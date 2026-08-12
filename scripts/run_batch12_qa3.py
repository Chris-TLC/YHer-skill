#!/usr/bin/env python3
"""Batch 12 QA-3 rolling L0 candidate package.

This runner reads official v4 data and writes all deliverables under
`/tmp/yher_batch12_qa3` by default. It never applies changes to official data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TOOLS_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data.item_bank_v4 import iter_service_items  # noqa: E402
from scripts.run_batch8_ws2 import (  # noqa: E402
    SCHEMA_VERSION as WS2_SCHEMA_VERSION,
    build_vision_client,
    default_node_modules,
    find_leak_hits,
    is_blank_image,
    load_service_items,
    repair_bad_assets_batch,
    run_illustration_track,
    validate_latex_katex,
)
from scripts.run_batch10_qa1 import (  # noqa: E402
    FAILURE_PROMPT_RE,
    OMML_CACHE,
    SERVICE_MAP_PATH,
    V4_ITEMS,
    WS2_MEDIA_REF_MAP,
    WS2_REPAIRED_ASSETS,
    WS2_TRANSCRIPTS,
    build_group_dirs,
    candidate_row as qa1_candidate_row,
    collect_omml_by_sha,
    extract_mathml_via_libreoffice,
    file_md5,
    gate_text_values,
    image_extrema_min,
    iter_text_occurrences,
    load_jsonl,
    mark_bad_prompt_rows,
    mark_leak_rows,
    mathml_to_latex,
    repair_cached_latex,
    run_formula_rows,
    write_json,
    write_jsonl,
)


OUT_ROOT = Path("/tmp/yher_batch12_qa3")
MACHINE_AUDIT = Path("/tmp/yher_batch11_qa2/full_pool/machine_audit.jsonl")
BATCH10_UNREPAIRABLE = Path("/tmp/yher_batch10_qa1/answer_zone_assets/asset_repair/unrepairable.jsonl")
BATCH10_LITERAL_AFTER = Path("/tmp/yher_batch10_qa1/literal_scan/literal_after_comparison.jsonl")
WS2_ASSET_MANIFEST = REPO_ROOT / "data" / "ws2_assets_v1_candidate_20260703" / "asset_manifest.jsonl"
QA3_SCHEMA_VERSION = "qa3_batch12_candidate_v1"

LABEL_RE = re.compile(r"([A-D])([.．、])")
ANS_PREFIX_RE = re.compile(r"^ans_[0-9a-f]{8}_")
LITERAL_RE = re.compile(r"\[(?:formula|figure):[^\]]+\]|\[OMML\]")
OFFICIAL_READONLY_PATHS = [V4_ITEMS, WS2_TRANSCRIPTS, WS2_MEDIA_REF_MAP, OMML_CACHE]


@dataclass
class SplitResult:
    ok: bool
    segments: list[str]
    labels: list[str]
    reason: str = ""
    invalid_markers: list[dict[str, Any]] | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_join(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def stamp_pending(row: dict[str, Any], candidate_kind: str | None = None) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("schema_version", QA3_SCHEMA_VERSION)
    if candidate_kind:
        out.setdefault("candidate_kind", candidate_kind)
    out["review_status"] = "pending_user_or_claude"
    out["reviewer"] = ""
    return out


def qa3_candidate_row(kind: str, **fields: Any) -> dict[str, Any]:
    return stamp_pending(
        {
            "schema_version": QA3_SCHEMA_VERSION,
            "candidate_kind": kind,
            **fields,
        }
    )


def previous_char_blocks_split(ch: str) -> bool:
    return bool(re.fullmatch(r"[0-9A-Za-z.]", ch or "")) or ch == "．"


def strictly_increasing(labels: list[str]) -> bool:
    return all(labels[idx] < labels[idx + 1] for idx in range(len(labels) - 1))


def split_option_segments(text: str) -> SplitResult:
    """Split a text node at safe option labels.

    A split marker must be `[A-D][.．、]`. Markers in the middle of text are
    accepted only when the previous character is not a digit, ASCII letter, or
    decimal point. This keeps strings such as `0.5B.` out of main candidates.
    """

    raw = list(LABEL_RE.finditer(text or ""))
    if not raw:
        return SplitResult(False, [], [], "no_option_markers", [])

    valid: list[re.Match[str]] = []
    invalid: list[dict[str, Any]] = []
    for match in raw:
        if match.start() == 0:
            valid.append(match)
            continue
        prev = text[match.start() - 1]
        if previous_char_blocks_split(prev):
            invalid.append({"label": match.group(1), "index": match.start(), "previous_char": prev})
        else:
            valid.append(match)

    if invalid and (not valid or min(m.start() for m in valid) > min(m["index"] for m in invalid)):
        return SplitResult(False, [], [m.group(1) for m in raw], "no_valid_split_markers_after_invalid_boundary", invalid)
    if not valid:
        return SplitResult(False, [], [m.group(1) for m in raw], "no_valid_split_markers", invalid)

    labels = [m.group(1) for m in valid]
    if not strictly_increasing(labels):
        return SplitResult(False, [], labels, "option_labels_not_strictly_increasing", invalid)

    first = valid[0]
    prefix = text[: first.start()].strip()
    pieces: list[str] = []
    if prefix and first.group(1) != "A":
        pieces.append(prefix)
        start_points = [m.start() for m in valid]
    else:
        start_points = [0 if prefix else first.start(), *[m.start() for m in valid[1:]]]

    for idx, start in enumerate(start_points):
        end = start_points[idx + 1] if idx + 1 < len(start_points) else len(text)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

    if len(pieces) < 2:
        return SplitResult(False, pieces, labels, "split_would_not_create_multiple_segments", invalid)
    if norm_join("".join(pieces)) != norm_join(text):
        return SplitResult(False, pieces, labels, "join_mismatch", invalid)
    return SplitResult(True, pieces, labels, "", invalid)


def direct_text_occurrences(item: dict[str, Any], field: str, zone: str) -> Iterable[dict[str, Any]]:
    for block_idx, block in enumerate(item.get(field) or []):
        para = block.get("para") if isinstance(block, dict) else (block if isinstance(block, list) else [])
        if not isinstance(para, list):
            continue
        for seg_idx, seg in enumerate(para):
            if isinstance(seg, dict) and seg.get("type") == "text":
                yield {
                    "zone": zone,
                    "field": field,
                    "block_path": f"{field}[{block_idx}].para[{seg_idx}]",
                    "text": str(seg.get("text") or ""),
                    "parent_para_len": len(para),
                    "node_index": seg_idx,
                }


def build_option_split_rows(
    items: list[dict[str, Any]],
    stem_target_item_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    low_confidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    stats: Counter[str] = Counter()

    for item in items:
        item_id = str(item.get("item_id") or "")
        scan_plan: list[tuple[str, str, str]] = []
        if item_id in stem_target_item_ids:
            scan_plan.append(("stem_blocks", "stem", "machine_audit_option_no_sticky"))
        scan_plan.append(("answer_blocks_effective", "answer", "answer_zone_scan"))

        for field, zone, source in scan_plan:
            for occ in direct_text_occurrences(item, field, zone):
                text = occ["text"]
                if not LABEL_RE.search(text):
                    continue
                split = split_option_segments(text)
                key = (item_id, occ["block_path"], text)
                if key in seen:
                    continue
                seen.add(key)
                base = {
                    "item_id": item_id,
                    "group_key": item.get("group_key"),
                    "section_num": item.get("section_num"),
                    "q_num": item.get("q_num"),
                    "zone": zone,
                    "block_path": occ["block_path"],
                    "original_text": text,
                    "labels": split.labels,
                    "parent_para_len": occ["parent_para_len"],
                    "node_index": occ["node_index"],
                    "source": source,
                }
                if split.ok:
                    split_kind = "in_block" if occ["parent_para_len"] > 1 else "tight_text"
                    row = qa3_candidate_row(
                        "option_split_v2",
                        **base,
                        suggested_segments=split.segments,
                        split_kind=split_kind,
                        split_count=len(split.segments),
                    )
                    if norm_join("".join(row["suggested_segments"])) != norm_join(row["original_text"]):
                        stats["join_mismatch"] += 1
                        low_confidence.append(qa3_candidate_row("option_split_v2_low_confidence", **base, reason="join_mismatch"))
                    else:
                        candidates.append(row)
                        stats[f"candidate_{split_kind}"] += 1
                        stats[f"zone_{zone}"] += 1
                else:
                    low_confidence.append(
                        qa3_candidate_row(
                            "option_split_v2_low_confidence",
                            **base,
                            reason=split.reason,
                            invalid_markers=split.invalid_markers or [],
                        )
                    )
                    stats[f"low_{split.reason}"] += 1

    summary = {
        "candidate_rows": len(candidates),
        "candidate_items": len({r["item_id"] for r in candidates}),
        "low_confidence_rows": len(low_confidence),
        "low_confidence_items": len({r["item_id"] for r in low_confidence}),
        "by_split_kind": dict(Counter(r["split_kind"] for r in candidates)),
        "by_zone": dict(Counter(r["zone"] for r in candidates)),
        **dict(stats),
    }
    summary.setdefault("join_mismatch", 0)
    return sorted(candidates, key=lambda r: (r["item_id"], r["zone"], r["block_path"])), sorted(
        low_confidence, key=lambda r: (r["item_id"], r["zone"], r["block_path"])
    ), summary


def collect_option_target_ids(machine_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("item_id") or "")
        for row in machine_rows
        if row.get("option_no_sticky") is False
    }


def collect_media_unmapped_refs(machine_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in machine_rows:
        item_id = str(row.get("item_id") or "")
        group_key = str(row.get("group_key") or "")
        for ev in (row.get("evidence") or {}).get("no_degrade_placeholder") or []:
            reason = str(ev.get("reason") or "")
            if not reason.startswith("media_unmapped:"):
                continue
            media = reason.split(":", 1)[1]
            key = (group_key, media)
            agg = refs.setdefault(
                key,
                {"group_key": group_key, "media": media, "zones": set(), "item_ids": set(), "occurrence_count": 0},
            )
            agg["zones"].add(str(ev.get("zone") or row.get("zone") or ""))
            agg["item_ids"].add(item_id)
            agg["occurrence_count"] += 1
    out: list[dict[str, Any]] = []
    for agg in refs.values():
        out.append(
            {
                "group_key": agg["group_key"],
                "media": agg["media"],
                "zones": sorted(z for z in agg["zones"] if z),
                "item_ids": sorted(agg["item_ids"]),
                "occurrence_count": agg["occurrence_count"],
            }
        )
    return sorted(out, key=lambda r: (r["group_key"], r["media"]))


def existing_ref_map_dict(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    return {
        (str(row.get("group_key") or ""), str(row.get("media") or "")): str(row.get("asset_hash") or "")
        for row in rows
        if row.get("group_key") and row.get("media") and row.get("asset_hash")
    }


def resolve_repo_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def ref_manifest_index(manifest_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        for ref in row.get("sample_refs") or []:
            group_key = str(ref.get("group_key") or "")
            media = str(ref.get("media") or "")
            if not group_key or not media:
                continue
            entry = {"asset_hash": row.get("asset_hash"), "manifest_row": row, "sample_ref": ref}
            index[(group_key, media)].append(entry)
    return index


def build_refmap_fix_rows(
    media_refs: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    existing_ref_map: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = ref_manifest_index(manifest_rows)
    group_dirs = build_group_dirs()
    candidates: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for ref in media_refs:
        group_key = str(ref.get("group_key") or "")
        media = str(ref.get("media") or "")
        if existing_ref_map.get((group_key, media)):
            continue
        media_variants = [media]
        stripped = ANS_PREFIX_RE.sub("", media)
        if stripped != media:
            media_variants.append(stripped)
        matches: list[dict[str, Any]] = []
        for variant in media_variants:
            matches.extend(index.get((group_key, variant), []))
        by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for match in matches:
            if match.get("asset_hash"):
                by_hash[str(match["asset_hash"])].append(match)
        if len(by_hash) != 1:
            source_paths: list[Path] = []
            gdir = group_dirs.get(group_key)
            if gdir is not None:
                for variant in media_variants:
                    p = gdir / "assets" / variant
                    if p.exists() and p not in source_paths:
                        source_paths.append(p)
            if len(source_paths) == 1:
                chosen_source = source_paths[0]
                candidates.append(
                    qa3_candidate_row(
                        "refmap_fix",
                        **ref,
                        suggested_asset_hash=sha256_file(chosen_source),
                        matched_media=chosen_source.name,
                        source_path=str(chosen_source),
                        source_sha256=sha256_file(chosen_source),
                        source_size_bytes=chosen_source.stat().st_size,
                        match_count=1,
                        match_method="ws1_source_sha256",
                        in_ws2_manifest=False,
                        media_variants=media_variants,
                    )
                )
                continue
            unresolved.append(
                qa3_candidate_row(
                    "refmap_unresolvable",
                    **ref,
                    reason="no_unique_manifest_or_ws1_source_match" if by_hash else "no_manifest_or_ws1_source_match",
                    matched_asset_hashes=sorted(by_hash),
                    source_path_candidates=[str(p) for p in source_paths],
                    media_variants=media_variants,
                )
            )
            continue
        asset_hash, hash_matches = next(iter(by_hash.items()))
        source_paths = []
        for match in hash_matches:
            raw_path = str((match.get("sample_ref") or {}).get("asset_path") or "")
            if raw_path:
                p = resolve_repo_path(raw_path)
                if p.exists() and p not in source_paths:
                    source_paths.append(p)
        if not source_paths:
            unresolved.append(
                qa3_candidate_row(
                    "refmap_unresolvable",
                    **ref,
                    reason="source_asset_missing",
                    suggested_asset_hash=asset_hash,
                    media_variants=media_variants,
                )
            )
            continue
        chosen_source = source_paths[0]
        candidates.append(
            qa3_candidate_row(
                "refmap_fix",
                **ref,
                suggested_asset_hash=asset_hash,
                matched_media=str((hash_matches[0].get("sample_ref") or {}).get("media") or ""),
                source_path=str(chosen_source),
                source_sha256=sha256_file(chosen_source),
                source_size_bytes=chosen_source.stat().st_size,
                match_count=len(hash_matches),
                media_variants=media_variants,
            )
        )
    return sorted(candidates, key=lambda r: (r["group_key"], r["media"])), sorted(
        unresolved, key=lambda r: (r["group_key"], r["media"])
    )


def load_manifest_by_hash(path: Path = WS2_ASSET_MANIFEST) -> dict[str, dict[str, Any]]:
    return {str(row.get("asset_hash")): row for row in read_jsonl(path) if row.get("asset_hash")}


def load_batch10_unrepairable_hashes(path: Path = BATCH10_UNREPAIRABLE) -> set[str]:
    if not path.exists():
        repaired = {p.stem for p in WS2_REPAIRED_ASSETS.glob("*.png")}
        return repaired
    return {str(row.get("asset_hash") or "") for row in read_jsonl(path) if row.get("asset_hash")}


def select_manual_queue_rerender_targets(
    transcript_rows: list[dict[str, Any]],
    manifest_by_hash: dict[str, dict[str, Any]],
    excluded_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    targets: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    seen: set[str] = set()
    for row in sorted(transcript_rows, key=lambda r: str(r.get("asset_hash") or "")):
        asset_hash = str(row.get("asset_hash") or "")
        if not asset_hash or asset_hash in seen or row.get("pool") != "manual_queue":
            continue
        seen.add(asset_hash)
        if asset_hash in excluded_hashes:
            excluded["batch10_unrepairable"] += 1
            continue
        base = dict(manifest_by_hash.get(asset_hash) or {"asset_hash": asset_hash, "sample_refs": []})
        base["asset_hash"] = asset_hash
        base["asset_class"] = row.get("asset_class") or base.get("asset_class") or "illustration"
        base["source_pool"] = row.get("pool")
        base["source_fine_type"] = row.get("fine_type")
        base["source_latex_status"] = row.get("latex_status")
        base["source_apply_id"] = row.get("apply_id")
        base["source_scope"] = "batch12_manual_queue_rerender"
        targets.append(base)
    return targets, dict(excluded)


def sum_cost(rows: Iterable[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        meta = row.get("metadata") or {}
        try:
            total += float(meta.get("cost_yuan") or 0.0)
        except Exception:
            pass
    return round(total, 4)


def gate_transcription_candidates(
    rows: list[dict[str, Any]],
    task: str,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stamped = [stamp_pending(row) for row in rows]
    kept, prompt_rejected = mark_bad_prompt_rows(stamped, task)
    leak_rejected: list[dict[str, Any]] = []
    if mode == "transcript":
        kept, leak_rejected = mark_leak_rows(kept, task)
    kept = [stamp_pending(row) for row in kept]
    prompt_rejected = [stamp_pending(row) for row in prompt_rejected]
    leak_rejected = [stamp_pending(row) for row in leak_rejected]
    return kept, prompt_rejected, leak_rejected


def run_asset_rerender(
    out_dir: Path,
    client: Any | None,
    workers: int,
    skip_vision: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_rows = read_jsonl(WS2_TRANSCRIPTS)
    manifest_by_hash = load_manifest_by_hash()
    excluded_hashes = load_batch10_unrepairable_hashes()
    targets, excluded = select_manual_queue_rerender_targets(transcript_rows, manifest_by_hash, excluded_hashes)
    write_jsonl(out_dir / "manual_queue_rerender_targets.jsonl", [stamp_pending(row, "asset_rerender_target") for row in targets])
    write_json(out_dir / "manual_queue_exclusions.json", {"excluded_hashes": sorted(excluded_hashes), "counts": excluded})

    items_by_id = load_service_items()
    service_map = json.loads(SERVICE_MAP_PATH.read_text(encoding="utf-8")) if SERVICE_MAP_PATH.exists() else {}
    repaired, unrepairable, attempts = repair_bad_assets_batch(targets, out_dir, items_by_id, service_map)
    write_jsonl(out_dir / "asset_repair" / "repair_attempts.jsonl", attempts)
    write_jsonl(out_dir / "asset_repair" / "repaired_assets.jsonl", repaired)
    write_jsonl(out_dir / "asset_repair" / "unrepairable.jsonl", unrepairable)

    target_by_hash = {row["asset_hash"]: row for row in targets}
    repaired_hashes = {str(row.get("asset_hash") or "") for row in repaired}
    rescued_rows = [target_by_hash[h] for h in sorted(repaired_hashes) if h in target_by_hash]
    nonblank_failures = []
    for row in repaired:
        png = Path(str(row.get("repaired_png") or ""))
        if not png.exists() or is_blank_image(png):
            nonblank_failures.append(row.get("asset_hash"))

    formula_rows = [row for row in rescued_rows if row.get("asset_class") == "formula_image"]
    illustration_rows = [row for row in rescued_rows if row.get("asset_class") != "formula_image"]
    cache_dir = out_dir / "api_cache"
    raw_formula = run_formula_rows(
        formula_rows,
        out_dir,
        cache_dir,
        client,
        workers,
        skip_vision,
        "batch12_asset_rerender_formula",
    )
    formula_kept, formula_prompt_rejected, _formula_leak = gate_transcription_candidates(
        raw_formula,
        "12b_asset_rerender_formula",
        "formula",
    )
    formula_candidates = [row for row in formula_kept if row.get("latex_status") == "passed"]
    formula_failures = [row for row in formula_kept if row.get("latex_status") != "passed"]
    formula_dir = out_dir / "formula_latex"
    write_jsonl(formula_dir / "formula_latex_candidates.jsonl", formula_candidates)
    write_jsonl(formula_dir / "formula_latex_failure_prompt_rejected.jsonl", formula_prompt_rejected)
    write_jsonl(formula_dir / "formula_latex_failures.jsonl", formula_failures)

    raw_transcripts = run_illustration_track(
        illustration_rows,
        out_dir,
        client,
        cache_dir,
        workers,
        None,
        out_path=out_dir / "transcripts" / "transcript_candidates_raw.jsonl",
        skip_vision=skip_vision,
    )
    transcript_kept, transcript_prompt_rejected, transcript_leak_rejected = gate_transcription_candidates(
        raw_transcripts,
        "12b_asset_rerender_transcript",
        "transcript",
    )
    transcript_candidates = [row for row in transcript_kept if row.get("pool") in {"ai_seed", "display_only"}]
    transcript_manual = [row for row in transcript_kept if row.get("pool") not in {"ai_seed", "display_only"}]
    transcript_dir = out_dir / "transcripts"
    write_jsonl(transcript_dir / "transcript_candidates.jsonl", transcript_candidates)
    write_jsonl(transcript_dir / "transcript_manual_queue.jsonl", transcript_manual)
    write_jsonl(transcript_dir / "transcript_failure_prompt_rejected.jsonl", transcript_prompt_rejected)
    write_jsonl(transcript_dir / "transcript_leak_rejected.jsonl", transcript_leak_rejected)

    summary = {
        "manual_queue_assets": len({row.get("asset_hash") for row in transcript_rows if row.get("pool") == "manual_queue"}),
        "excluded_assets": excluded,
        "target_assets": len(targets),
        "repaired_nonblank_assets": len(repaired),
        "still_dead_assets": len(unrepairable),
        "nonblank_scan_failures": len(nonblank_failures),
        "formula_targets": len(formula_rows),
        "illustration_targets": len(illustration_rows),
        "formula_candidate_rows": len(formula_candidates),
        "formula_failures": len(formula_failures),
        "formula_failure_prompt_rejected": len(formula_prompt_rejected),
        "transcript_candidate_rows": len(transcript_candidates),
        "transcript_manual_queue_rows": len(transcript_manual),
        "transcript_failure_prompt_rejected": len(transcript_prompt_rejected),
        "transcript_leak_rejected": len(transcript_leak_rejected),
        "rescued_transcription_assets": len(formula_candidates) + len(transcript_candidates),
        "cost_yuan": round(sum_cost(raw_formula) + sum_cost(raw_transcripts), 4),
    }
    write_json(out_dir / "asset_rerender_summary.json", summary)
    return summary


def missing_omml_sources(
    omml_sources: dict[str, dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cached = {str(row.get("omml_sha1") or "") for row in cache_rows if row.get("omml_sha1")}
    return [omml_sources[sha] for sha in sorted(omml_sources) if sha not in cached]


def run_omml_backfill(items: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_rows = read_jsonl(OMML_CACHE)
    omml_sources = collect_omml_by_sha(items)
    targets = missing_omml_sources(omml_sources, cache_rows)
    write_jsonl(out_dir / "omml_backfill_targets.jsonl", [stamp_pending(row, "omml_backfill_target") for row in targets])

    node_modules = default_node_modules(out_dir)
    candidates: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    mathml_rows: list[dict[str, Any]] = []
    for source in targets:
        sha = str(source.get("omml_sha1") or "")
        attempts: list[dict[str, Any]] = []
        chosen_latex = ""
        chosen_compile: dict[str, Any] = {"ok": False, "engine": "none", "error": "not_attempted"}
        mathml = ""
        mathml_status = "not_attempted"

        if source.get("latex"):
            repaired = repair_cached_latex(str(source.get("latex") or ""))
            chosen_compile = validate_latex_katex(repaired, node_modules=node_modules) if repaired else {"ok": False, "error": "empty_source_latex"}
            attempts.append({"method": "source_latex_sanitized", "latex": repaired, "compile_result": chosen_compile})
            if chosen_compile.get("ok"):
                chosen_latex = repaired

        if not chosen_latex and source.get("omml"):
            mathml, mathml_status = extract_mathml_via_libreoffice(str(source["omml"]), out_dir / "work", sha)
            if mathml:
                mathml_latex = mathml_to_latex(mathml)
                mathml_compile = validate_latex_katex(mathml_latex, node_modules=node_modules) if mathml_latex else {"ok": False, "error": "empty_mathml_latex"}
                attempts.append({"method": "libreoffice_flat_xml_mathml_to_latex", "latex": mathml_latex, "compile_result": mathml_compile})
                if mathml_compile.get("ok"):
                    chosen_latex = mathml_latex
                    chosen_compile = mathml_compile
            mathml_rows.append({"omml_sha1": sha, "status": mathml_status, "mathml": mathml[:4000]})

        base = {
            "omml_sha1": sha,
            "occurrences": source.get("occurrences") or [],
            "attempts": attempts,
            "compile_result": chosen_compile,
            "mathml_status": mathml_status,
        }
        if chosen_latex and chosen_compile.get("ok"):
            candidates.append(
                qa3_candidate_row(
                    "omml_backfill",
                    **base,
                    latex=chosen_latex,
                    ok=True,
                    katex_ok=True,
                )
            )
        else:
            manual.append(
                qa3_candidate_row(
                    "omml_backfill_manual",
                    **base,
                    latex=chosen_latex,
                    ok=False,
                    katex_ok=False,
                    manual_reason=mathml_status if mathml_status != "ok" else (chosen_compile.get("error") or "compile_failed"),
                )
            )

    write_jsonl(out_dir / "omml_backfill_candidates.jsonl", candidates)
    write_jsonl(out_dir / "omml_backfill_manual_queue.jsonl", manual)
    write_jsonl(out_dir / "omml_backfill_mathml_samples.jsonl", mathml_rows)
    summary = {
        "missing_cache_rows": len(targets),
        "missing_occurrences": sum(len(row.get("occurrences") or []) for row in targets),
        "candidate_rows": len(candidates),
        "manual_queue_rows": len(manual),
        "candidate_compile_failures": sum(1 for row in candidates if not (row.get("compile_result") or {}).get("ok")),
        "compile_engines": dict(Counter(str((row.get("compile_result") or {}).get("engine") or "") for row in candidates)),
    }
    write_json(out_dir / "omml_backfill_summary.json", summary)
    return summary


def item_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("group_key") or ""), str(row.get("section_num") or ""), str(row.get("q_num") or ""))


def has_ws1_source(item: dict[str, Any], group_dirs: dict[str, Path]) -> bool:
    group_key = str(item.get("group_key") or "")
    if group_key in group_dirs:
        return True
    source_path = str(item.get("source_path") or "")
    if not source_path:
        return False
    p = Path(source_path)
    if not p.is_absolute():
        p = (REPO_ROOT / source_path).resolve()
    return p.exists()


def scan_literal_residual_accounting(
    items: list[dict[str, Any]],
    residual_item_ids: set[str],
    batch10_after_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    b10_by_key = {item_key(row): row for row in batch10_after_rows}
    dirs = build_group_dirs()
    rows: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item.get("item_id") or "")
        if item_id not in residual_item_ids:
            continue
        key = item_key(item)
        batch10 = b10_by_key.get(key)
        for field in ("stem_blocks", "answer_blocks_effective", "analysis_blocks"):
            zone = {"stem_blocks": "stem", "answer_blocks_effective": "answer", "analysis_blocks": "analysis"}[field]
            for occ in iter_text_occurrences(item.get(field) or [], field):
                text = str(occ.get("text") or "")
                for match in LITERAL_RE.finditer(text):
                    if not batch10:
                        origin = "not_in_batch10_10c_targets"
                    elif zone != "stem":
                        origin = "answer_or_analysis_zone_residual"
                    elif int(batch10.get("after_literal_rows") or 0) > 0:
                        origin = "batch10_unmatched_or_still_residual"
                    else:
                        origin = "post_apply_residual_requires_review"
                    rows.append(
                        qa3_candidate_row(
                            "literal_residual_accounting",
                            item_id=item_id,
                            group_key=item.get("group_key"),
                            section_num=item.get("section_num"),
                            q_num=item.get("q_num"),
                            literal_text=match.group(0),
                            literal_type="OMML" if match.group(0) == "[OMML]" else match.group(0).split(":", 1)[0].strip("["),
                            zone=zone,
                            block_path=occ.get("path"),
                            context=text[max(0, match.start() - 80) : match.end() + 80],
                            has_ws1_source=has_ws1_source(item, dirs),
                            batch10_10c_key_seen=bool(batch10),
                            batch10_10c_old_item_id=batch10.get("old_item_id") if batch10 else "",
                            batch10_10c_new_item_id=batch10.get("new_item_id") if batch10 else "",
                            batch10_10c_rerun_status=batch10.get("rerun_status") if batch10 else "",
                            batch10_10c_after_literal_rows=batch10.get("after_literal_rows") if batch10 else None,
                            residual_origin=origin,
                        )
                    )
    summary = {
        "residual_items_from_machine_audit": len(residual_item_ids),
        "accounting_rows": len(rows),
        "accounting_items": len({row["item_id"] for row in rows}),
        "by_zone": dict(Counter(row["zone"] for row in rows)),
        "by_origin": dict(Counter(row["residual_origin"] for row in rows)),
        "with_ws1_source": sum(1 for row in rows if row.get("has_ws1_source")),
    }
    return sorted(rows, key=lambda r: (r["item_id"], r["zone"], r["block_path"], r["literal_text"])), summary


def failure_prompt_hits(row: dict[str, Any]) -> list[str]:
    hits: list[str] = []
    for text in gate_text_values(row):
        hits.extend(match.group(0) for match in FAILURE_PROMPT_RE.finditer(text))
    return sorted(set(hits))


def validate_deliverables(out_root: Path, official_before: dict[str, str], official_after: dict[str, str]) -> dict[str, Any]:
    candidate_file_names = {
        "option_split_candidates.jsonl",
        "low_confidence.jsonl",
        "refmap_fix_candidates.jsonl",
        "refmap_unresolvable.jsonl",
        "media_unmapped_refs.jsonl",
        "manual_queue_rerender_targets.jsonl",
        "formula_latex_candidates.jsonl",
        "transcript_candidates.jsonl",
        "transcript_manual_queue.jsonl",
        "omml_backfill_targets.jsonl",
        "omml_backfill_candidates.jsonl",
        "omml_backfill_manual_queue.jsonl",
        "literal_residual_accounting.jsonl",
    }
    reviewer_nonempty = 0
    codex_reviewer = 0
    bad_status = 0
    candidate_rows = 0
    failure_prompt_kept_hits = 0
    leak_kept_hits = 0
    option_join_mismatch = 0
    for path in out_root.rglob("*.jsonl"):
        is_rejected = "rejected" in path.name or "manual_queue" in path.name
        for row in read_jsonl(path):
            if row.get("candidate_kind") or path.name in candidate_file_names:
                candidate_rows += 1
                if row.get("reviewer"):
                    reviewer_nonempty += 1
                if str(row.get("reviewer") or "").startswith("codex_"):
                    codex_reviewer += 1
                if row.get("review_status") != "pending_user_or_claude":
                    bad_status += 1
            if path.name == "option_split_candidates.jsonl":
                if norm_join("".join(row.get("suggested_segments") or [])) != norm_join(row.get("original_text") or ""):
                    option_join_mismatch += 1
            if not is_rejected and path.name in {"formula_latex_candidates.jsonl", "transcript_candidates.jsonl"}:
                if failure_prompt_hits(row):
                    failure_prompt_kept_hits += 1
                if path.name == "transcript_candidates.jsonl" and find_leak_hits({k: row.get(k) for k in ["summary", "elements", "text_in_image", "data_points", "uncertain", "transcript"]}):
                    leak_kept_hits += 1
    return {
        "official_md5_unchanged": official_before == official_after,
        "candidate_rows_checked": candidate_rows,
        "reviewer_nonempty": reviewer_nonempty,
        "codex_reviewer": codex_reviewer,
        "bad_review_status": bad_status,
        "failure_prompt_kept_hits": failure_prompt_kept_hits,
        "leak_kept_hits": leak_kept_hits,
        "option_join_mismatch": option_join_mismatch,
    }


def write_batch_report(out_root: Path, summaries: dict[str, Any], validation: dict[str, Any], started: float) -> None:
    lines = [
        "# Batch 12 QA-3 Rolling First Package Report",
        "",
        f"- elapsed_sec: {round(time.time() - started, 1)}",
        f"- output_root: `{out_root}`",
        f"- measured_vision_cost_yuan: {round(float((summaries.get('asset_rerender') or {}).get('cost_yuan') or 0.0), 4)}",
        f"- official_md5_unchanged: {validation['official_md5_unchanged']}",
        f"- reviewer_nonempty: {validation['reviewer_nonempty']}",
        f"- codex_reviewer: {validation['codex_reviewer']}",
        f"- bad_review_status: {validation['bad_review_status']}",
        f"- failure_prompt_kept_hits: {validation['failure_prompt_kept_hits']}",
        f"- leak_kept_hits: {validation['leak_kept_hits']}",
        f"- option_join_mismatch: {validation['option_join_mismatch']}",
        "",
    ]
    for key in ["option_split_v2", "refmap_fix", "asset_rerender", "omml_backfill", "literal_accounting"]:
        lines.extend([f"## {key}", "", "```json", json.dumps(summaries.get(key, {}), ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    lines.extend(
        [
            "## Discipline",
            "",
            "- L0 only: no official apply was performed.",
            "- Official item-bank/transcript/ref_map/OMML cache files were read-only inputs.",
            "- Candidate rows keep blank reviewer and `review_status=pending_user_or_claude`.",
            "- Failure-prompt and leak hits are excluded from kept transcription candidates.",
        ]
    )
    (out_root / "BATCH12_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-vision", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    out_root = args.out_root
    for name in ["option_split_v2", "refmap_fix", "asset_rerender", "omml_backfill", "literal_accounting"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)

    official_before = {str(path): file_md5(path) for path in OFFICIAL_READONLY_PATHS if path.exists()}
    items = list(iter_service_items())
    machine_rows = read_jsonl(MACHINE_AUDIT)
    summaries: dict[str, Any] = {}

    option_dir = out_root / "option_split_v2"
    stem_target_ids = collect_option_target_ids(machine_rows)
    option_candidates, low_confidence, option_summary = build_option_split_rows(items, stem_target_ids)
    write_jsonl(option_dir / "option_split_candidates.jsonl", option_candidates)
    write_jsonl(option_dir / "low_confidence.jsonl", low_confidence)
    write_json(option_dir / "option_split_summary.json", option_summary)
    summaries["option_split_v2"] = option_summary

    ref_dir = out_root / "refmap_fix"
    media_refs = collect_media_unmapped_refs(machine_rows)
    manifest_rows = read_jsonl(WS2_ASSET_MANIFEST)
    ref_candidates, ref_unresolved = build_refmap_fix_rows(media_refs, manifest_rows, existing_ref_map_dict(read_jsonl(WS2_MEDIA_REF_MAP)))
    write_jsonl(ref_dir / "media_unmapped_refs.jsonl", [stamp_pending(row, "media_unmapped_ref") for row in media_refs])
    write_jsonl(ref_dir / "refmap_fix_candidates.jsonl", ref_candidates)
    write_jsonl(ref_dir / "refmap_unresolvable.jsonl", ref_unresolved)
    ref_summary = {
        "media_unmapped_refs": len(media_refs),
        "refmap_fix_candidates": len(ref_candidates),
        "refmap_unresolvable": len(ref_unresolved),
        "mapped_occurrences": sum(int(row.get("occurrence_count") or 0) for row in ref_candidates),
        "unresolved_occurrences": sum(int(row.get("occurrence_count") or 0) for row in ref_unresolved),
    }
    write_json(ref_dir / "refmap_fix_summary.json", ref_summary)
    summaries["refmap_fix"] = ref_summary

    client = None if args.skip_vision else build_vision_client()
    summaries["asset_rerender"] = run_asset_rerender(out_root / "asset_rerender", client, args.workers, args.skip_vision)

    summaries["omml_backfill"] = run_omml_backfill(items, out_root / "omml_backfill")

    residual_ids = {
        str(row.get("item_id") or "")
        for row in machine_rows
        if row.get("no_asset_literal_residue") is False
    }
    literal_rows, literal_summary = scan_literal_residual_accounting(items, residual_ids, read_jsonl(BATCH10_LITERAL_AFTER))
    literal_dir = out_root / "literal_accounting"
    write_jsonl(literal_dir / "literal_residual_accounting.jsonl", literal_rows)
    write_json(literal_dir / "literal_residual_summary.json", literal_summary)
    summaries["literal_accounting"] = literal_summary

    official_after = {str(path): file_md5(path) for path in OFFICIAL_READONLY_PATHS if path.exists()}
    validation = validate_deliverables(out_root, official_before, official_after)
    summaries["validation"] = validation
    write_json(out_root / "batch12_summary.json", summaries)
    write_batch_report(out_root, summaries, validation, started)
    return 0 if all(
        [
            validation["official_md5_unchanged"],
            validation["reviewer_nonempty"] == 0,
            validation["codex_reviewer"] == 0,
            validation["bad_review_status"] == 0,
            validation["failure_prompt_kept_hits"] == 0,
            validation["leak_kept_hits"] == 0,
            validation["option_join_mismatch"] == 0,
        ]
    ) else 2


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
