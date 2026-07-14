"""Runtime adapter from signed curriculum evidence to public recommendations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from engine import recommender


DEFAULT_ASSET = Path(__file__).with_name("assets") / "curriculum_runtime_v1.json"
DEFAULT_VIDEO_CHUNKS = Path(__file__).resolve().parents[2] / "data/video_chunks/chemistry_chunks_v1.jsonl"
_ENTITY_PREFIXES = ("bv:", "season:", "series:")
_ORGANIC_MARKERS = (
    "有机",
    "烷烃",
    "烯烃",
    "炔烃",
    "芳香烃",
    "卤代烃",
    "醇",
    "酚",
    "醛",
    "酮",
    "羧酸",
    "酯",
)


class CurriculumRuntime:
    """Fail-closed signed candidate pool around ``engine.recommender``."""

    def __init__(self, payload: dict[str, Any]):
        if not isinstance(payload, dict):
            raise ValueError("curriculum runtime payload must be an object")
        self.version = str(payload.get("version") or "")
        self.provenance = copy.deepcopy(payload.get("provenance") or {})
        raw_track_map = copy.deepcopy(payload.get("track_map") or {})
        self.track_map = recommender.load_track_map(raw_track_map)
        self._active_entities = set(self.track_map["entities"])
        self._segments_by_node = {
            str(node): [copy.deepcopy(row) for row in rows if isinstance(row, dict)]
            for node, rows in (payload.get("segments_by_node") or {}).items()
            if isinstance(rows, list)
        }
        self._validate_segments()

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        video_chunks_path: Path | None = None,
        video_chunks: Iterable[dict[str, Any]] | None = None,
    ) -> "CurriculumRuntime":
        if video_chunks_path is not None and video_chunks is not None:
            raise ValueError("video_chunks_path and video_chunks are mutually exclusive")
        if video_chunks_path is not None:
            rows = _read_video_chunks(Path(video_chunks_path))
            return cls(_overlay_video_chunks(payload, rows, Path(video_chunks_path)))
        if video_chunks is not None:
            return cls(_overlay_video_chunks(payload, list(video_chunks), None))
        return cls(payload)

    @classmethod
    def from_default_asset(
        cls,
        path: Path = DEFAULT_ASSET,
        *,
        video_chunks_path: Path | None = DEFAULT_VIDEO_CHUNKS,
    ) -> "CurriculumRuntime":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if video_chunks_path is not None and Path(video_chunks_path).is_file():
            chunks_path = Path(video_chunks_path)
            value = _overlay_video_chunks(
                value, _read_video_chunks(chunks_path), chunks_path
            )
        return cls(value)

    def eligible_segments(self, node: str) -> list[dict[str, Any]]:
        """Return only exact-BV or canonical season/series signed candidates."""
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in self._segments_by_node.get(str(node), []):
            entity = canonical_entity(source.get("signed_entity"))
            if entity not in self._active_entities:
                continue
            bv = str(source.get("bv") or "").strip()
            try:
                p = int(source.get("p") or 1)
            except (TypeError, ValueError):
                continue
            segment_id = str(source.get("segment_id") or "")
            if not bv or p < 1 or not segment_id or segment_id in seen:
                continue
            row = copy.deepcopy(source)
            row["bv"], row["p"], row["signed_entity"] = bv, p, entity
            output.append(row)
            seen.add(segment_id)
        return output

    def recommend(
        self,
        *,
        node: str,
        belief: np.ndarray,
        grade: str,
        learning_purpose: str,
        budget: dict[str, Any],
        seen_segments: Iterable[Any],
        session_id: str,
        action_id: str,
    ) -> dict[str, Any]:
        return self.recommend_multi_node(
            target_nodes=[str(node)],
            beliefs={str(node): np.asarray(belief, dtype=float)},
            grade=grade,
            learning_purpose=learning_purpose,
            budget=budget,
            seen_segments=seen_segments,
            session_id=session_id,
            action_id=action_id,
        )

    def recommend_multi_node(
        self,
        *,
        target_nodes: Iterable[str],
        beliefs: dict[str, np.ndarray],
        grade: str,
        learning_purpose: str,
        budget: dict[str, Any],
        seen_segments: Iterable[Any],
        session_id: str,
        action_id: str,
    ) -> dict[str, Any]:
        ordered_nodes: list[str] = []
        for value in target_nodes:
            node = str(value).strip()
            if node and node not in ordered_nodes:
                ordered_nodes.append(node)

        budget_seconds = max(0, int(float(budget.get("rx_minutes") or 0) * 60))
        candidates_by_node: dict[str, list[dict[str, Any]]] = {}
        source_by_node_segment: dict[tuple[str, str], dict[str, Any]] = {}
        runtime_map = copy.deepcopy(self.track_map)
        for node in ordered_nodes:
            engine_candidates = copy.deepcopy(self.eligible_segments(node))
            for segment in engine_candidates:
                full_duration = max(
                    1,
                    int(
                        segment.get("full_video_duration_sec")
                        or segment.get("duration_sec")
                        or 1
                    ),
                )
                segment["full_video_duration_sec"] = full_duration
                anchor = _usable_anchor(node, segment)
                if anchor is not None:
                    start, end = float(anchor["start_sec"]), float(anchor["end_sec"])
                    segment["start_sec"], segment["end_sec"] = start, end
                    segment["duration_sec"] = end - start
                else:
                    segment.pop("start_sec", None)
                    segment.pop("end_sec", None)
                if (
                    budget_seconds > 0
                    and full_duration > budget_seconds
                    and anchor is None
                ):
                    # Legacy whole-video rows stay range-free and disclose that
                    # only the session-budget prefix was allocated.
                    segment["duration_sec"] = budget_seconds
                    segment["watch_scope"] = "from_start_within_session_budget"
                source_by_node_segment[(node, str(segment["segment_id"]))] = segment
                entity = segment["signed_entity"]
                # The pure recommender resolves exact BV then season. This
                # runtime-only alias proves the BV belongs to an active signed row.
                alias = f"bv:{segment['bv']}"
                if alias not in runtime_map["entities"]:
                    runtime_map["entities"][alias] = copy.deepcopy(
                        runtime_map["entities"][entity]
                    )
            candidates_by_node[node] = engine_candidates

        normalized_seen = _normalize_seen(seen_segments)
        sequence = 0

        def rec_id_factory() -> str:
            nonlocal sequence
            sequence += 1
            material = f"recommendation:v1:{session_id}:{action_id}:{sequence}".encode()
            return hashlib.sha256(material).hexdigest()[:32]

        result = recommender.recommend(
            grade,
            learning_purpose,
            ordered_nodes,
            {node: np.asarray(beliefs[node], dtype=float) for node in ordered_nodes},
            candidates_by_node,
            runtime_map,
            dict(budget),
            seen_segments=normalized_seen,
            rec_id_factory=rec_id_factory,
            session_id=str(session_id),
            action_id=str(action_id),
        )
        public: list[dict[str, Any]] = []
        bindings: dict[str, dict[str, Any]] = {}
        for served in result["recommendations"]:
            served_node = str(served["node"])
            source = source_by_node_segment[(served_node, str(served["segment_id"]))]
            anchor = _usable_anchor(served_node, source)
            start = int(float(anchor["start_sec"])) if anchor else None
            url = _bilibili_url(source["bv"], source["p"], start)
            rec_id = str(served["rec_id"])
            duration = (
                max(1, int(float(anchor["end_sec"]) - float(anchor["start_sec"])))
                if anchor
                else max(1, int(source.get("duration_sec") or 1))
            )
            full_duration = max(
                duration, int(source.get("full_video_duration_sec") or duration)
            )
            completion = str(
                source.get("completion_criterion")
                or "能独立复述步骤并完成同考点变式"
            )
            if source.get("watch_scope") == "from_start_within_session_budget":
                completion = (
                    f"先从开头观看本次分配的 {duration // 60} 分钟，写下前段的关键变量与因果关系；"
                    f"完整视频标准：{completion}"
                )
            public.append(
                {
                    "rec_id": rec_id,
                    "segment_id": str(source["segment_id"]),
                    "title": str(served.get("part_title") or source.get("video_title") or "观看讲解"),
                    "duration_seconds": duration,
                    "full_video_duration_seconds": full_duration,
                    "value": str(source.get("value") or "针对本次诊断证据补齐这一考点"),
                    "completion_criterion": completion,
                    "url": url,
                    "reason": str(served.get("reason") or ""),
                    "has_time_anchor": anchor is not None,
                    "watch_scope": source.get("watch_scope") or (
                        "exact_segment" if anchor else "full_video"
                    ),
                }
            )
            bindings[rec_id] = {
                "rec_id": rec_id,
                "session_id": str(session_id),
                "action_id": str(action_id),
                "node": served_node,
                "bv": source["bv"],
                "p": source["p"],
                "segment_id": str(source.get("segment_id") or f"{source['bv']}#P{source['p']:03d}"),
                "signed_entity": source["signed_entity"],
                "url": url,
            }
        return {
            "recommendations": public,
            "bindings": bindings,
            "rec_served": result["rec_served"],
            "warnings": list(result.get("warnings") or []),
            "status": result.get("status"),
        }

    def _validate_segments(self) -> None:
        identities: dict[str, tuple[Any, ...]] = {}
        for node, rows in self._segments_by_node.items():
            for row in rows:
                segment_id = str(row.get("segment_id") or "")
                if not segment_id:
                    raise ValueError(f"curriculum segment in {node!r} is missing segment_id")
                try:
                    physical_identity = (
                        str(row.get("bv") or ""),
                        int(row.get("p") or 1),
                        float(row["start_sec"]) if row.get("start_sec") is not None else None,
                        float(row["end_sec"]) if row.get("end_sec") is not None else None,
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"curriculum segment {segment_id!r} has invalid physical identity"
                    ) from exc
                previous = identities.get(segment_id)
                if previous is not None and previous != physical_identity:
                    raise ValueError(
                        f"curriculum segment physical identity conflict: {segment_id}"
                    )
                identities[segment_id] = physical_identity
                canonical_entity(row.get("signed_entity"))


def _read_video_chunks(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            row = copy.deepcopy(row)
            row["_video_chunks_line"] = line_number
            output.append(row)
    return output


def _overlay_video_chunks(
    payload: dict[str, Any],
    chunk_rows: list[dict[str, Any]],
    source_path: Path | None,
) -> dict[str, Any]:
    if not chunk_rows:
        return copy.deepcopy(payload)
    by_part: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for source in chunk_rows:
        if source.get("needs_human") is not False:
            continue
        try:
            bv = str(source.get("bv") or "").strip()
            p = int(source.get("p_number") or 1)
            start, end = float(source["start_sec"]), float(source["end_sec"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not bv or p < 1 or not 0 <= start < end:
            continue
        by_part.setdefault((bv, p), []).append(source)

    output = copy.deepcopy(payload)
    overlaid: dict[str, list[dict[str, Any]]] = {}
    for node, legacy_rows in (payload.get("segments_by_node") or {}).items():
        node_name = str(node)
        node_output: list[dict[str, Any]] = []
        seen_segment_ids: set[str] = set()
        for legacy in legacy_rows if isinstance(legacy_rows, list) else []:
            try:
                part_key = (str(legacy.get("bv") or ""), int(legacy.get("p") or 1))
            except (TypeError, ValueError):
                continue
            for chunk in by_part.get(part_key, []):
                topics = [str(value) for value in chunk.get("knowledge_topic") or []]
                if node_name not in topics:
                    continue
                segment_id = str(chunk.get("chunk_id") or "").strip()
                if not segment_id or segment_id in seen_segment_ids:
                    continue
                start, end = float(chunk["start_sec"]), float(chunk["end_sec"])
                row = copy.deepcopy(legacy)
                provenance = copy.deepcopy(row.get("provenance") or {})
                provenance.update({
                    "chunk_source": "video_chunks",
                    "video_chunks_line": chunk.get("_video_chunks_line"),
                })
                row.update({
                    "segment_id": segment_id,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": end - start,
                    "full_video_duration_sec": max(
                        1, int(float(legacy.get("duration_sec") or 1))
                    ),
                    "knowledge_topic": topics,
                    "chunk_source": "video_chunks",
                    "time_anchor": {
                        "chunk_id": segment_id,
                        "start_sec": start,
                        "end_sec": end,
                        "needs_human": False,
                        "source_line": chunk.get("_video_chunks_line"),
                        "chunk_source": "video_chunks",
                    },
                    "provenance": provenance,
                })
                node_output.append(row)
                seen_segment_ids.add(segment_id)
        if node_output:
            overlaid[node_name] = node_output
    output["segments_by_node"] = overlaid
    provenance = copy.deepcopy(output.get("provenance") or {})
    if source_path is not None:
        raw = Path(source_path).read_bytes()
        provenance["video_chunks"] = {
            "path": str(source_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "lines": len(chunk_rows),
        }
    else:
        provenance["video_chunks"] = {
            "path": None,
            "sha256": hashlib.sha256(
                json.dumps(chunk_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "lines": len(chunk_rows),
        }
    provenance["build_rule"] = (
        "trusted video chunks with exact runtime-node topic match; legacy rows are clone-only"
    )
    output["provenance"] = provenance
    return output


def canonical_entity(value: Any) -> str:
    normalized = str(value or "").strip()
    while normalized.startswith("season:") and normalized[len("season:") :].startswith(
        _ENTITY_PREFIXES
    ):
        normalized = normalized[len("season:") :]
    if not normalized.startswith(_ENTITY_PREFIXES):
        raise ValueError(f"non-canonical curriculum entity: {value!r}")
    prefix, identifier = normalized.split(":", 1)
    if not identifier or ":" in identifier:
        raise ValueError(f"non-canonical curriculum entity: {value!r}")
    return f"{prefix}:{identifier}"


def _normalize_seen(values: Iterable[Any]) -> set[tuple[str, int]]:
    output: set[tuple[str, int]] = set()
    for value in values or ():
        if isinstance(value, dict):
            bv, p = value.get("bv"), value.get("p")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            bv, p = value
        else:
            continue
        try:
            output.add((str(bv), int(p)))
        except (TypeError, ValueError):
            continue
    return output


def _usable_anchor(node: str, segment: Any) -> dict[str, Any] | None:
    if not isinstance(segment, dict):
        return None
    provenance = segment.get("provenance") or {}
    if (
        segment.get("chunk_source") == "video_chunks"
        or isinstance(provenance, dict)
        and provenance.get("chunk_source") == "video_chunks"
    ):
        value = segment
    else:
        if not _is_organic_node(node):
            return None
        value = segment.get("time_anchor")
        if not isinstance(value, dict) or value.get("needs_human") is not False:
            return None
    try:
        start, end = float(value["start_sec"]), float(value["end_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= start < end:
        return None
    return value


def _is_organic_node(node: str) -> bool:
    value = str(node)
    return any(marker in value for marker in _ORGANIC_MARKERS)


def _bilibili_url(bv: str, p: int, start: int | None) -> str:
    url = f"https://www.bilibili.com/video/{bv}?p={int(p)}"
    return f"{url}&t={start}" if start is not None else url
