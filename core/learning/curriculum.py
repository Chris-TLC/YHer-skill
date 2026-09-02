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
    def from_payload(cls, payload: dict[str, Any]) -> "CurriculumRuntime":
        return cls(payload)

    @classmethod
    def from_default_asset(cls, path: Path = DEFAULT_ASSET) -> "CurriculumRuntime":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(value)

    def eligible_segments(self, node: str) -> list[dict[str, Any]]:
        """Return only exact-BV or canonical season/series signed candidates."""
        output: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for source in self._segments_by_node.get(str(node), []):
            entity = canonical_entity(source.get("signed_entity"))
            if entity not in self._active_entities:
                continue
            bv = str(source.get("bv") or "").strip()
            try:
                p = int(source.get("p") or 1)
            except (TypeError, ValueError):
                continue
            if not bv or p < 1 or (bv, p) in seen:
                continue
            row = copy.deepcopy(source)
            row["bv"], row["p"], row["signed_entity"] = bv, p, entity
            output.append(row)
            seen.add((bv, p))
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
        candidates = self.eligible_segments(node)
        budget_seconds = max(0, int(float(budget.get("rx_minutes") or 0) * 60))
        engine_candidates = copy.deepcopy(candidates)
        for segment in engine_candidates:
            full_duration = max(1, int(segment.get("duration_sec") or 1))
            segment["full_video_duration_sec"] = full_duration
            if (
                budget_seconds > 0
                and full_duration > budget_seconds
                and _usable_anchor(str(node), segment.get("time_anchor")) is None
            ):
                # Video-level recommendation: allocate the first N minutes and
                # disclose the complete duration; no synthetic timestamp/deep link.
                segment["duration_sec"] = budget_seconds
                segment["watch_scope"] = "from_start_within_session_budget"
        runtime_map = copy.deepcopy(self.track_map)
        for segment in engine_candidates:
            entity = segment["signed_entity"]
            # The pure recommender currently resolves exact BV then season. A
            # runtime-only alias preserves its API while the adapter proves the
            # originating BV belongs to an active signed series/season.
            alias = f"bv:{segment['bv']}"
            if alias not in runtime_map["entities"]:
                runtime_map["entities"][alias] = copy.deepcopy(
                    runtime_map["entities"][entity]
                )

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
            [str(node)],
            {str(node): np.asarray(belief, dtype=float)},
            {str(node): engine_candidates},
            runtime_map,
            dict(budget),
            seen_segments=normalized_seen,
            rec_id_factory=rec_id_factory,
            session_id=str(session_id),
            action_id=str(action_id),
        )
        by_key = {(row["bv"], row["p"]): row for row in engine_candidates}
        public: list[dict[str, Any]] = []
        bindings: dict[str, dict[str, Any]] = {}
        for served in result["recommendations"]:
            source = by_key[(str(served["bv"]), int(served["p"]))]
            anchor = _usable_anchor(str(node), source.get("time_anchor"))
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
                    "title": str(served.get("part_title") or source.get("video_title") or "观看讲解"),
                    "duration_seconds": duration,
                    "full_video_duration_seconds": full_duration,
                    "value": str(source.get("value") or "针对本次诊断证据补齐这一考点"),
                    "completion_criterion": completion,
                    "url": url,
                    "reason": str(served.get("reason") or ""),
                    "has_time_anchor": anchor is not None,
                    "watch_scope": source.get("watch_scope") or "full_video",
                }
            )
            bindings[rec_id] = {
                "rec_id": rec_id,
                "session_id": str(session_id),
                "action_id": str(action_id),
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
        ids: set[str] = set()
        for node, rows in self._segments_by_node.items():
            for row in rows:
                segment_id = str(row.get("segment_id") or "")
                if not segment_id:
                    raise ValueError(f"curriculum segment in {node!r} is missing segment_id")
                scoped_id = f"{node}\0{segment_id}"
                if scoped_id in ids:
                    raise ValueError(f"duplicate curriculum segment: {node}/{segment_id}")
                ids.add(scoped_id)
                canonical_entity(row.get("signed_entity"))


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


def _usable_anchor(node: str, value: Any) -> dict[str, Any] | None:
    if not _is_organic_node(node) or not isinstance(value, dict):
        return None
    if value.get("needs_human") is not False:
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
