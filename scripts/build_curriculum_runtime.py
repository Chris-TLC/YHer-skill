#!/usr/bin/env python3
"""Build the deterministic signed curriculum asset consumed by the Demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT.parent
DEFAULT_TRACK_MAP = REPO_ROOT / "config/curriculum/track_map_v1.yaml"
DEFAULT_CATALOG = Path("/tmp/yher_channel_catalog/channel_catalog.jsonl")
DEFAULT_KG = REPO_ROOT / "data/knowledge_graph_150_enriched.jsonl"
DEFAULT_ORGANIC = TOOLS_ROOT / "scratchpad/organic_chunks_timestamped.jsonl"
DEFAULT_VIDEO_CHUNKS = REPO_ROOT / "data/video_chunks/chemistry_chunks_v1.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "core/learning/assets/curriculum_runtime_v1.json"
ENTITY_PREFIXES = ("bv:", "season:", "series:")
ORGANIC_MARKERS = (
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


def build_runtime(
    *,
    track_map_path: Path,
    catalog_path: Path,
    kg_path: Path,
    organic_path: Path | None = None,
    video_chunks_path: Path | None = None,
) -> dict[str, Any]:
    if organic_path is not None and video_chunks_path is not None:
        raise ValueError("organic_path and video_chunks_path are compatibility aliases; provide one")
    candidate_path = Path(video_chunks_path or organic_path) if (video_chunks_path or organic_path) else None
    raw_track_map = yaml.safe_load(Path(track_map_path).read_text(encoding="utf-8"))
    if not isinstance(raw_track_map, dict):
        raise ValueError("track map must be an object")
    active = {
        canonical_entity(row.get("entity"))
        for row in raw_track_map.get("entities") or []
        if row.get("needs_human") is False
        and str(row.get("reviewer") or "").strip()
        and row.get("evidence")
    }
    catalog = _catalog_index(catalog_path)
    anchors = _anchor_index(candidate_path) if candidate_path is not None else {}
    candidate_file_nonempty = False
    if candidate_path is not None:
        with candidate_path.open(encoding="utf-8") as handle:
            candidate_file_nonempty = any(raw.strip() for raw in handle)
    segments: dict[str, dict[Any, dict[str, Any]]] = {}
    for kg_line, kg_row in _read_jsonl_with_lines(kg_path):
        node = str(kg_row.get("node_id") or "").strip()
        parent = str(kg_row.get("parent_node") or "").strip()
        if not node:
            continue
        targets = [node, *([parent] if parent else [])]
        for video in kg_row.get("recommended_videos") or []:
            bv = str(video.get("bv") or "").strip()
            try:
                p = int(video.get("p_number") or 1)
            except (TypeError, ValueError):
                continue
            catalog_hit = catalog.get((bv, p))
            if not bv or catalog_hit is None:
                continue
            catalog_line, catalog_row = catalog_hit
            try:
                authoritative_duration = int(catalog_row.get("part_duration_sec"))
            except (TypeError, ValueError):
                continue
            if authoritative_duration < 1:
                continue
            signed_entity = _signed_entity(catalog_row, active)
            if signed_entity is None:
                continue
            for target in targets:
                selected = _select_anchors(target, anchors.get((bv, p), []))
                if candidate_file_nonempty and not selected:
                    continue
                for anchor in selected if candidate_file_nonempty else [None]:
                    full_duration = authoritative_duration
                    duration = (
                        float(anchor["end_sec"]) - float(anchor["start_sec"])
                        if anchor
                        else full_duration
                    )
                    provenance = {
                        "kg_line": kg_line,
                        "catalog_line": catalog_line,
                    }
                    if anchor:
                        provenance.update({
                            "chunk_source": "video_chunks",
                            "video_chunks_line": anchor["source_line"],
                        })
                    segment = {
                        "segment_id": (
                            str(anchor["chunk_id"]) if anchor else f"{bv}#P{p:03d}"
                        ),
                        "bv": bv,
                        "p": p,
                        "signed_entity": signed_entity,
                        "seg_type": _segment_type(video.get("type")),
                        "difficulty": str(video.get("difficulty") or "T2"),
                        "topic_match_ratio": 1.0,
                        "duration_sec": duration,
                        "full_video_duration_sec": full_duration,
                        "view": int(catalog_row.get("view") or 0),
                        "pubdate": str(catalog_row.get("pubdate") or ""),
                        "season_id": canonical_optional(catalog_row.get("season_id")),
                        "season_name": catalog_row.get("season_name"),
                        "season_order": catalog_row.get("season_order"),
                        "part_title": str(
                            catalog_row.get("part_title")
                            or catalog_row.get("video_title")
                            or ""
                        ),
                        "video_title": str(catalog_row.get("video_title") or ""),
                        "part_degrade_state": catalog_row.get("part_degrade_state"),
                        "value": str(video.get("what_you_learn") or ""),
                        "completion_criterion": str(video.get("completion_criterion") or ""),
                        "time_anchor": anchor,
                        "provenance": provenance,
                    }
                    if anchor:
                        segment.update({
                            "start_sec": float(anchor["start_sec"]),
                            "end_sec": float(anchor["end_sec"]),
                            "knowledge_topic": list(anchor["knowledge_topic"]),
                            "chunk_source": "video_chunks",
                        })
                    key = segment["segment_id"] if anchor else (bv, p)
                    segments.setdefault(target, {}).setdefault(key, segment)
    candidate_provenance = (
        _file_provenance(candidate_path)
        if candidate_path is not None
        else {"path": None, "sha256": None, "lines": 0}
    )
    return {
        "version": "curriculum_runtime_v1_20260713",
        "provenance": {
            "track_map": _file_provenance(track_map_path),
            "catalog": _file_provenance(catalog_path),
            "knowledge_graph": _file_provenance(kg_path),
            "organic_timestamps": candidate_provenance,
            "video_chunks": candidate_provenance,
            "build_rule": (
                "KG binding plus exact catalog row plus active exact-BV/season/series signature; "
                "non-empty candidate files are chunk-only and require exact topic plus needs_human=false"
            ),
        },
        "track_map": raw_track_map,
        "segments_by_node": {
            node: [rows[key] for key in sorted(rows)]
            for node, rows in sorted(segments.items())
        },
    }


def canonical_entity(value: Any) -> str:
    normalized = str(value or "").strip()
    while normalized.startswith("season:") and normalized[len("season:") :].startswith(
        ENTITY_PREFIXES
    ):
        normalized = normalized[len("season:") :]
    if not normalized.startswith(ENTITY_PREFIXES):
        raise ValueError(f"non-canonical entity: {value!r}")
    prefix, identifier = normalized.split(":", 1)
    if not identifier or ":" in identifier:
        raise ValueError(f"non-canonical entity: {value!r}")
    return f"{prefix}:{identifier}"


def canonical_optional(value: Any) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized.startswith(ENTITY_PREFIXES):
        return canonical_entity(normalized)
    return canonical_entity(f"season:{normalized}")


def _signed_entity(row: dict[str, Any], active: set[str]) -> str | None:
    exact = canonical_entity(f"bv:{row.get('bv')}")
    if exact in active:
        return exact
    season = canonical_optional(row.get("season_id"))
    return season if season in active else None


def _catalog_index(path: Path) -> dict[tuple[str, int], tuple[int, dict[str, Any]]]:
    output: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    for line_number, row in _read_jsonl_with_lines(path):
        key = (str(row.get("bv") or ""), int(row.get("p") or 1))
        if key in output:
            raise ValueError(f"duplicate catalog row: {key}")
        output[key] = (line_number, row)
    return output


def _anchor_index(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    output: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for line_number, row in _read_jsonl_with_lines(path):
        if row.get("needs_human") is not False:
            continue
        row = dict(row)
        row["source_line"] = line_number
        output.setdefault((str(row.get("bv") or ""), int(row.get("p_number") or 1)), []).append(row)
    return output


def _select_anchor(node: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected = _select_anchors(node, rows)
    return selected[0] if selected else None


def _select_anchors(node: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        topics = {str(value) for value in row.get("knowledge_topic") or []}
        if str(node) not in topics:
            continue
        try:
            start, end = float(row["start_sec"]), float(row["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end:
            output.append({
                "chunk_id": str(row.get("chunk_id") or ""),
                "start_sec": start,
                "end_sec": end,
                "knowledge_topic": list(row.get("knowledge_topic") or []),
                "needs_human": False,
                "source_line": int(row["source_line"]),
                "chunk_source": "video_chunks",
            })
    return output


def _segment_type(value: Any) -> str:
    return {
        "concept_intro": "concept_intro",
        "review_with_problems": "review",
        "exam_problem_drill": "drill",
    }.get(str(value or ""), str(value or "concept_intro"))


def _read_jsonl_with_lines(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            yield line_number, value


def _file_provenance(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    lines = 0
    with Path(path).open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            lines += 1
    return {"path": str(path), "sha256": digest.hexdigest(), "lines": lines}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-map", type=Path, default=DEFAULT_TRACK_MAP)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--knowledge-graph", type=Path, default=DEFAULT_KG)
    candidates = parser.add_mutually_exclusive_group()
    candidates.add_argument("--video-chunks", type=Path)
    candidates.add_argument("--organic-timestamps", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_runtime(
        track_map_path=args.track_map,
        catalog_path=args.catalog,
        kg_path=args.knowledge_graph,
        video_chunks_path=(
            args.video_chunks
            if args.video_chunks is not None
            else None if args.organic_timestamps is not None else DEFAULT_VIDEO_CHUNKS
        ),
        organic_path=args.organic_timestamps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "nodes": len(payload["segments_by_node"]),
                "segments": sum(len(rows) for rows in payload["segments_by_node"].values()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
