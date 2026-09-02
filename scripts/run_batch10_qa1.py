#!/usr/bin/env python3
"""Batch 10 QA-1 data repair candidate package.

This is an L0 candidate runner. It reads official v4 data and writes all
deliverables under /tmp/yher_batch10_qa1 by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from lxml import etree

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TOOLS_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_batch8_ws2 import (  # noqa: E402
    BATCH8_CANDIDATE_ROOT,
    GOLD_LIST_PATH,
    SCHEMA_VERSION as WS2_SCHEMA_VERSION,
    SERVICE_MAP_PATH,
    SOFFICE,
    build_vision_client,
    crop_png,
    default_node_modules,
    filter_transcript_leaks,
    image_extrema_min,
    is_blank_image,
    load_jsonl,
    load_service_items,
    repair_bad_assets_batch,
    run_illustration_track,
    run_parallel,
    transcribe_formula_asset,
    validate_latex_katex,
    write_json,
    write_jsonl,
)


OUT_ROOT = Path("/tmp/yher_batch10_qa1")
QA_SCHEMA_VERSION = "qa1_candidate_v1"

V4_DIR = REPO_ROOT / "data" / "item_bank" / "v4"
V4_ITEMS = V4_DIR / "chemistry_v4_1_3329.jsonl"
WS2_TRANSCRIPTS = V4_DIR / "ws2_asset_transcripts_v1.jsonl"
WS2_MEDIA_REF_MAP = V4_DIR / "ws2_media_ref_map_v1.jsonl"
WS2_REPAIRED_ASSETS = V4_DIR / "ws2_repaired_assets"
WS2_ASSET_MANIFEST = REPO_ROOT / "data" / "ws2_assets_v1_candidate_20260703" / "asset_manifest.jsonl"
WS1_ROOT = REPO_ROOT / "data" / "ws1_batch_v4_20260703"
SOURCE_ROOT = TOOLS_ROOT / "上海化学卷合集"
OMML_CACHE = V4_DIR / "ws2_omml_latex_cache_v1.jsonl"

FAILURE_PROMPT_RE = re.compile(r"(无法识别|无有效|图片|抱歉|unable|cannot|sorry)", re.IGNORECASE)
FRAC_SUSPECT_RE = re.compile(
    r"\\frac\{(?:[0-9]{1,2}|[+\-−]|[0-9]{1,2}[+\-−]?|[+\-−]?[0-9]{1,2})\}"
    r"\{(?:[0-9]{1,2}|[+\-−]|[0-9]{1,2}[+\-−]?|[+\-−]?[0-9]{1,2})\}"
)
LITERAL_RE = re.compile(r"\[(?:formula|figure):[^\]]+\]|\[OMML\]")
OPTION_MARK_RE = re.compile(r"(?<![0-9A-Za-z.])([A-D][.．、])")
ASCII_MINUS = str.maketrans({"−": "-", "－": "-", "—": "-", "–": "-", "＋": "+"})


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_row(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": QA_SCHEMA_VERSION,
        "candidate_kind": kind,
        "review_status": "pending_user_or_claude",
        "reviewer": "",
        **fields,
    }


def sum_cost(rows: Iterable[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        meta = row.get("metadata") or {}
        try:
            total += float(meta.get("cost_yuan") or 0.0)
        except Exception:
            pass
    return round(total, 4)


def gate_text_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if row.get("latex"):
        values.append(str(row.get("latex") or ""))
    transcript = row.get("transcript")
    if isinstance(transcript, dict):
        for key in ("summary", "elements", "text_in_image", "data_points", "uncertain"):
            value = transcript.get(key)
            if isinstance(value, list):
                values.extend(str(v) for v in value)
            elif value:
                values.append(str(value))
    for key in ("summary", "elements", "text_in_image", "data_points", "uncertain"):
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(v) for v in value)
        elif value:
            values.append(str(value))
    runs = row.get("runs")
    if runs:
        values.append(json.dumps(runs, ensure_ascii=False))
    return values


def failure_prompt_hits(row: dict[str, Any]) -> list[str]:
    hits: list[str] = []
    for text in gate_text_values(row):
        hits.extend(match.group(0) for match in FAILURE_PROMPT_RE.finditer(text))
    return sorted(set(hits))


def has_failure_prompt_text(row: dict[str, Any]) -> bool:
    return bool(failure_prompt_hits(row))


def mark_bad_prompt_rows(rows: list[dict[str, Any]], task: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        hits = failure_prompt_hits(row)
        if not hits:
            kept.append(row)
            continue
        bad = json.loads(json.dumps(row, ensure_ascii=False))
        bad["schema_version"] = bad.get("schema_version") or QA_SCHEMA_VERSION
        bad["review_status"] = "pending_user_or_claude"
        bad["reviewer"] = ""
        bad["pool"] = "manual_queue"
        bad["pool_reason"] = "failure_prompt_text"
        bad["latex_status"] = "failed" if "latex" in bad else bad.get("latex_status")
        bad["qa1_rejection_task"] = task
        bad["qa1_rejection_reason"] = "failure_prompt_text"
        bad["qa1_failure_prompt_hits"] = hits
        rejected.append(bad)
    return kept, rejected


def mark_leak_rows(rows: list[dict[str, Any]], task: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, rejected = filter_transcript_leaks(rows)
    for row in rejected:
        row["schema_version"] = row.get("schema_version") or WS2_SCHEMA_VERSION
        row["review_status"] = "pending_user_or_claude"
        row["reviewer"] = ""
        row["pool"] = "leak_rejected"
        row["pool_reason"] = "answer_or_analysis_leak_pattern"
        row["qa1_rejection_task"] = task
    return kept, rejected


def sorted_unique(values: Iterable[Any]) -> list[str]:
    return sorted({str(v) for v in values if str(v)})


def zone_sort_key(zone: str) -> int:
    order = {"answer": 0, "analysis": 1, "stem": 2}
    return order.get(zone, 9)


def build_group_dirs() -> dict[str, Path]:
    groups: dict[str, Path] = {}
    if not WS1_ROOT.exists():
        return groups
    for gdir in WS1_ROOT.iterdir():
        if not gdir.is_dir() or not (gdir / "assets").is_dir():
            continue
        group_key = None
        summary = gdir / "summary.json"
        if summary.exists():
            try:
                group_key = json.loads(summary.read_text(encoding="utf-8")).get("group_key")
            except Exception:
                group_key = None
        group_key = group_key or re.sub(r"_[0-9a-f]{10}$", "", gdir.name)
        groups[str(group_key)] = gdir
    return groups


def iter_media_occurrences(node: Any, item: dict[str, Any], zone: str, path: str = "") -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        block_type = node.get("type")
        if block_type in {"formula", "figure"} and node.get("media"):
            yield {
                "group_key": item.get("group_key"),
                "media": node.get("media"),
                "block_type": block_type,
                "zone": zone,
                "item_id": item.get("item_id"),
                "q_num": item.get("q_num"),
                "section_num": item.get("section_num"),
                "block_path": path,
            }
        for key, value in node.items():
            yield from iter_media_occurrences(value, item, zone, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from iter_media_occurrences(value, item, zone, f"{path}[{idx}]")


def build_media_occurrence_map(items: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    refs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        for zone, field in (
            ("stem", "stem_blocks"),
            ("answer", "answer_blocks_effective"),
            ("analysis", "analysis_blocks"),
        ):
            for occ in iter_media_occurrences(item.get(field) or [], item, zone, field):
                refs[(str(occ.get("group_key") or ""), str(occ.get("media") or ""))].append(occ)
    return refs


def resolve_asset_path(group_dirs: dict[str, Path], group_key: str, media: str) -> Path | None:
    gdir = group_dirs.get(group_key)
    if not gdir:
        return None
    p = gdir / "assets" / media
    return p


def repo_relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def build_answer_zone_asset_rows(
    media_rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    group_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    occurrences = build_media_occurrence_map(items)
    by_hash: dict[str, dict[str, Any]] = {}
    for media_row in media_rows:
        if media_row.get("in_ws2_manifest") is not False:
            continue
        asset_hash = str(media_row.get("asset_hash") or "")
        if not asset_hash:
            continue
        group_key = str(media_row.get("group_key") or "")
        media = str(media_row.get("media") or "")
        path = resolve_asset_path(group_dirs, group_key, media)
        row = by_hash.setdefault(
            asset_hash,
            {
                "schema_version": WS2_SCHEMA_VERSION,
                "asset_hash": asset_hash,
                "asset_class": "",
                "source_scope": "answer_or_analysis_zone_unmanifested",
                "in_ws2_manifest": False,
                "zones": set(),
                "block_types": set(),
                "media_refs": [],
                "sample_refs": [],
                "ref_count": 0,
                "question_ids": set(),
                "original_ext": Path(media).suffix.lower(),
            },
        )
        row["zones"].update(str(z) for z in (media_row.get("zones") or []))
        row["media_refs"].append(media_row)
        row["ref_count"] += 1
        occs = occurrences.get((group_key, media), [])
        if not occs:
            occs = [
                {
                    "block_type": "unknown",
                    "zone": ",".join(media_row.get("zones") or []),
                    "item_id": "",
                    "q_num": "",
                    "section_num": "",
                    "block_path": "",
                    "group_key": group_key,
                    "media": media,
                }
            ]
        for occ in occs:
            block_type = str(occ.get("block_type") or "unknown")
            if block_type != "unknown":
                row["block_types"].add(block_type)
            if occ.get("item_id"):
                row["question_ids"].add(str(occ.get("item_id")))
            row["sample_refs"].append(
                {
                    "asset_path": repo_relative(path),
                    "block_type": block_type,
                    "group_key": group_key,
                    "media": media,
                    "question_id": occ.get("item_id") or "",
                    "q_num": occ.get("q_num"),
                    "section_num": occ.get("section_num"),
                    "zone": occ.get("zone"),
                    "block_path": occ.get("block_path"),
                }
            )
    rows: list[dict[str, Any]] = []
    for row in by_hash.values():
        block_types = set(row.pop("block_types"))
        row["asset_class"] = "formula_image" if block_types == {"formula"} else "illustration"
        row["block_types"] = sorted(block_types)
        row["zones"] = sorted(set(row["zones"]), key=zone_sort_key)
        row["question_count"] = len(row["question_ids"])
        row["question_ids"] = sorted(row["question_ids"])
        rows.append(row)
    return sorted(rows, key=lambda r: r["asset_hash"])


def write_transcription_outputs(
    rows: list[dict[str, Any]],
    out_path: Path,
    task: str,
    mode: str,
) -> dict[str, Any]:
    kept, prompt_rejected = mark_bad_prompt_rows(rows, task)
    leak_rejected: list[dict[str, Any]] = []
    if mode == "transcript":
        kept, leak_rejected = mark_leak_rows(kept, task)
    write_jsonl(out_path, kept)
    if prompt_rejected:
        write_jsonl(out_path.with_name(out_path.stem + "_failure_prompt_rejected.jsonl"), prompt_rejected)
    else:
        write_jsonl(out_path.with_name(out_path.stem + "_failure_prompt_rejected.jsonl"), [])
    if leak_rejected:
        write_jsonl(out_path.with_name(out_path.stem + "_leak_rejected.jsonl"), leak_rejected)
    elif mode == "transcript":
        write_jsonl(out_path.with_name(out_path.stem + "_leak_rejected.jsonl"), [])
    if mode == "formula":
        write_jsonl(out_path.with_name(out_path.stem + "_failures.jsonl"), [r for r in kept if r.get("latex_status") != "passed"])
    return {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "failure_prompt_rejected": len(prompt_rejected),
        "leak_rejected": len(leak_rejected),
        "cost_yuan": sum_cost(rows),
    }


def copy_official_repaired_assets(rows: list[dict[str, Any]], out_root: Path) -> int:
    target = out_root / "asset_repair" / "repaired"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        src = WS2_REPAIRED_ASSETS / f"{row['asset_hash']}.png"
        dst = target / f"{row['asset_hash']}.png"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def load_cached_answer_repair(
    answer_dir: Path,
    asset_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    repair_dir = answer_dir / "asset_repair"
    attempts_path = repair_dir / "repair_attempts.jsonl"
    repaired_path = repair_dir / "repaired_assets.jsonl"
    unrepairable_path = repair_dir / "unrepairable.jsonl"
    if not (attempts_path.exists() and repaired_path.exists() and unrepairable_path.exists()):
        return None
    expected = {str(row.get("asset_hash") or "") for row in asset_rows if row.get("asset_hash")}
    if not expected:
        return None
    attempts = load_jsonl(attempts_path)
    repaired = load_jsonl(repaired_path)
    unrepairable = load_jsonl(unrepairable_path)
    attempt_hashes = {str(row.get("asset_hash") or "") for row in attempts if row.get("asset_hash")}
    repaired_hashes = {str(row.get("asset_hash") or "") for row in repaired if row.get("asset_hash")}
    unrepairable_hashes = {str(row.get("asset_hash") or "") for row in unrepairable if row.get("asset_hash")}
    if attempt_hashes != expected:
        return None
    if repaired_hashes | unrepairable_hashes != expected:
        return None
    if repaired_hashes & unrepairable_hashes:
        return None
    repaired_dir = repair_dir / "repaired"
    for asset_hash in repaired_hashes:
        if not (repaired_dir / f"{asset_hash}.png").exists():
            return None
    return repaired, unrepairable, attempts


def run_formula_rows(
    rows: list[dict[str, Any]],
    out_root: Path,
    cache_dir: Path,
    client: Any | None,
    workers: int,
    skip_vision: bool,
    cache_track: str,
) -> list[dict[str, Any]]:
    node_modules = default_node_modules(out_root)
    row_by_hash = {str(row.get("asset_hash") or ""): row for row in rows}

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        return transcribe_formula_asset(
            row,
            out_root,
            client,
            cache_dir,
            node_modules,
            skip_vision=skip_vision,
            cache_track=cache_track,
        )

    results = run_parallel(rows, run_one, workers)
    for result in results:
        source = row_by_hash.get(str(result.get("asset_hash") or ""), {})
        meta = result.setdefault("metadata", {})
        for key in ("source_scope", "zones", "block_types", "question_ids", "fine_type", "source_pool", "original_latex"):
            if key in source:
                meta[key] = source.get(key)
    return results


def run_10a_answer_zone(
    items: list[dict[str, Any]],
    out_root: Path,
    client: Any | None,
    workers: int,
    skip_vision: bool,
) -> dict[str, Any]:
    answer_dir = out_root / "answer_zone_assets"
    answer_dir.mkdir(parents=True, exist_ok=True)
    media_rows = load_jsonl(WS2_MEDIA_REF_MAP)
    target_media_rows = [r for r in media_rows if r.get("in_ws2_manifest") is False]
    asset_rows = build_answer_zone_asset_rows(target_media_rows, items, build_group_dirs())
    write_jsonl(answer_dir / "answer_zone_asset_manifest.jsonl", asset_rows)
    write_jsonl(answer_dir / "answer_zone_media_refs.jsonl", target_media_rows)

    cached_repair = load_cached_answer_repair(answer_dir, asset_rows)
    if cached_repair:
        repaired, unrepairable, repair_results = cached_repair
    else:
        items_by_id = load_service_items()
        service_map = json.loads(SERVICE_MAP_PATH.read_text(encoding="utf-8")) if SERVICE_MAP_PATH.exists() else {}
        repaired, unrepairable, repair_results = repair_bad_assets_batch(asset_rows, answer_dir, items_by_id, service_map)
        write_jsonl(answer_dir / "asset_repair" / "repair_attempts.jsonl", repair_results)
        write_jsonl(answer_dir / "asset_repair" / "repaired_assets.jsonl", repaired)
        write_jsonl(answer_dir / "asset_repair" / "unrepairable.jsonl", unrepairable)

    formula_rows = [row for row in asset_rows if row.get("asset_class") == "formula_image"]
    illustration_rows = [row for row in asset_rows if row.get("asset_class") == "illustration"]
    cache_dir = answer_dir / "api_cache"
    raw_formula = run_formula_rows(formula_rows, answer_dir, cache_dir, client, workers, skip_vision, "answer_zone_formula")
    formula_out = answer_dir / "formula_latex" / "formula_latex_candidates.jsonl"
    formula_summary = write_transcription_outputs(raw_formula, formula_out, "10a_answer_zone_formula", "formula")

    raw_transcripts = run_illustration_track(
        illustration_rows,
        answer_dir,
        client,
        cache_dir,
        workers,
        None,
        out_path=answer_dir / "transcripts" / "transcript_candidates_raw.jsonl",
        skip_vision=skip_vision,
    )
    transcript_out = answer_dir / "transcripts" / "transcript_candidates.jsonl"
    transcript_summary = write_transcription_outputs(raw_transcripts, transcript_out, "10a_answer_zone_transcript", "transcript")

    summary = {
        "media_ref_rows": len(target_media_rows),
        "unique_assets": len(asset_rows),
        "formula_assets": len(formula_rows),
        "illustration_assets": len(illustration_rows),
        "repaired_or_collected_assets": len(repaired),
        "unrepairable_assets": len(unrepairable),
        "formula": formula_summary,
        "transcript": transcript_summary,
        "cost_yuan": round(formula_summary["cost_yuan"] + transcript_summary["cost_yuan"], 4),
    }
    write_json(answer_dir / "answer_zone_summary.json", summary)
    return summary


def load_manifest_by_hash() -> dict[str, dict[str, Any]]:
    return {row["asset_hash"]: row for row in load_jsonl(WS2_ASSET_MANIFEST) if row.get("asset_hash")}


def formula_asset_row(asset_hash: str, manifest_by_hash: dict[str, dict[str, Any]], **metadata: Any) -> dict[str, Any]:
    base = dict(manifest_by_hash.get(asset_hash) or {"asset_hash": asset_hash, "sample_refs": []})
    base["asset_class"] = "formula_image"
    base.update(metadata)
    return base


def run_10b_formula_backfill(
    out_root: Path,
    client: Any | None,
    workers: int,
    skip_vision: bool,
) -> dict[str, Any]:
    out_dir = out_root / "formula_backfill"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts = load_jsonl(WS2_TRANSCRIPTS)
    manifest_by_hash = load_manifest_by_hash()
    targets = [
        row
        for row in transcripts
        if row.get("pool") in {"ai_seed", "display_only"}
        and row.get("fine_type") in {"formula_fragment", "chemical_equation_image"}
        and not row.get("latex")
    ]
    rows = [
        formula_asset_row(
            row["asset_hash"],
            manifest_by_hash,
            source_scope="batch10_formula_backfill",
            fine_type=row.get("fine_type"),
            source_pool=row.get("pool"),
        )
        for row in targets
    ]
    write_jsonl(out_dir / "formula_backfill_targets.jsonl", rows)
    copied = copy_official_repaired_assets(rows, out_dir)
    raw = run_formula_rows(rows, out_dir, out_dir / "api_cache", client, workers, skip_vision, "formula_backfill")
    summary = write_transcription_outputs(raw, out_dir / "formula_backfill_candidates.jsonl", "10b_formula_backfill", "formula")
    summary.update({"target_rows": len(targets), "official_repaired_assets_copied": copied})
    write_json(out_dir / "formula_backfill_summary.json", summary)
    return summary


def is_isolated_prescript_fragment(latex: str) -> bool:
    text = str(latex or "").strip()
    if re.fullmatch(r"\d+\^-_\d+", text):
        return True
    inner = re.fullmatch(r"\\ce\{\s*(.*?)\s*\}", text)
    if not inner:
        return False
    value = inner.group(1)
    return bool(re.fullmatch(r"(?:\d+)?\^\{?[+\-−]\}?_\{?\d+\}?", value))


def normalize_isolated_prescript_fragment(latex: str) -> str | None:
    text = str(latex or "").strip()
    m = re.fullmatch(r"(?P<charge_num>\d+)\^-_(?P<sub>\d+)", text)
    if m:
        return f"{{}}^{{{m.group('charge_num')}-}}_{{{m.group('sub')}}}"
    m = re.fullmatch(r"\\ce\{\s*(?:(?P<charge_num>\d+))?\^\{?(?P<charge>[+\-−])\}?_\{?(?P<sub>\d+)\}?\s*\}", text)
    if m:
        charge_num = m.group("charge_num") or ""
        charge = m.group("charge").translate(ASCII_MINUS)
        return f"{{}}^{{{charge_num}{charge}}}_{{{m.group('sub')}}}"
    return None


def select_latex_form_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for row in rows:
        latex = str(row.get("latex") or "")
        reason = ""
        if FRAC_SUSPECT_RE.search(latex):
            reason = "simple_frac_suspect"
        elif is_isolated_prescript_fragment(latex):
            reason = "isolated_prescript_fragment"
        if reason and row.get("asset_hash") not in seen:
            copied = dict(row)
            copied["qa1_latex_form_reason"] = reason
            selected.append(copied)
            seen.add(row.get("asset_hash"))
    return selected


def run_10d_latex_form_fix(
    out_root: Path,
    client: Any | None,
    workers: int,
    skip_vision: bool,
) -> dict[str, Any]:
    out_dir = out_root / "latex_form_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts = load_jsonl(WS2_TRANSCRIPTS)
    targets = select_latex_form_targets([row for row in transcripts if row.get("latex")])
    manifest_by_hash = load_manifest_by_hash()
    deterministic_rows: list[dict[str, Any]] = []
    rerun_targets: list[dict[str, Any]] = []
    for row in targets:
        normalized = normalize_isolated_prescript_fragment(str(row.get("latex") or ""))
        if normalized:
            compile_result = validate_latex_katex(normalized, node_modules=default_node_modules(out_dir))
            deterministic_rows.append(
                candidate_row(
                    "latex_form_fix",
                    asset_hash=row.get("asset_hash"),
                    fix_method="deterministic_prescript_normalization",
                    original_latex=row.get("latex"),
                    suggested_latex=normalized,
                    compile_result=compile_result,
                    latex_status="passed" if compile_result.get("ok") else "failed",
                    review_basis="isolated fragment canonicalization",
                )
            )
        else:
            rerun_targets.append(
                formula_asset_row(
                    row["asset_hash"],
                    manifest_by_hash,
                    source_scope="batch10_latex_form_fix",
                    original_latex=row.get("latex"),
                    qa1_latex_form_reason=row.get("qa1_latex_form_reason"),
                )
            )
    write_jsonl(out_dir / "latex_form_fix_targets.jsonl", targets)
    copied = copy_official_repaired_assets(rerun_targets, out_dir)
    raw = run_formula_rows(rerun_targets, out_dir, out_dir / "api_cache", client, workers, skip_vision, "latex_form_fix")
    raw_by_hash = {row.get("asset_hash"): row for row in raw}
    before_by_hash = {row.get("asset_hash"): row for row in targets}
    before_after = []
    for asset_hash, new_row in sorted(raw_by_hash.items()):
        before = before_by_hash.get(asset_hash, {})
        merged = {
            **new_row,
            "original_latex": before.get("latex"),
            "suggested_latex": new_row.get("latex"),
            "qa1_latex_form_reason": before.get("qa1_latex_form_reason"),
        }
        before_after.append(merged)
    kept_rerun, rejected = mark_bad_prompt_rows(before_after, "10d_latex_form_fix")
    all_candidates = kept_rerun + deterministic_rows
    write_jsonl(out_dir / "latex_form_fix_candidates.jsonl", all_candidates)
    write_jsonl(out_dir / "latex_form_fix_failure_prompt_rejected.jsonl", rejected)
    write_jsonl(out_dir / "latex_form_fix_failures.jsonl", [r for r in all_candidates if r.get("latex_status") != "passed"])
    summary = {
        "targets": len(targets),
        "rerun_targets": len(rerun_targets),
        "deterministic_prescript_targets": len(deterministic_rows),
        "candidate_rows": len(all_candidates),
        "failure_prompt_rejected": len(rejected),
        "official_repaired_assets_copied": copied,
        "cost_yuan": sum_cost(raw),
    }
    write_json(out_dir / "latex_form_fix_summary.json", summary)
    return summary


def iter_text_occurrences(node: Any, path: str = "") -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        if node.get("type") == "text":
            yield {"path": path, "text": str(node.get("text") or ""), "block_type": "text", "cell": ""}
        elif node.get("type") == "table":
            for r_idx, row in enumerate(node.get("rows") or []):
                for c_idx, cell in enumerate(row):
                    yield from iter_text_occurrences(cell, f"{path}.rows[{r_idx}][{c_idx}]")
        else:
            for key, value in node.items():
                yield from iter_text_occurrences(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from iter_text_occurrences(value, f"{path}[{idx}]")
    elif isinstance(node, str):
        yield {"path": path, "text": node, "block_type": "string", "cell": ""}


def scope_for_item(item: dict[str, Any]) -> str:
    from core.data.item_bank_v4 import load_service_exclusions, service_blockers

    blockers = service_blockers(item, load_service_exclusions())
    if not blockers:
        return "service"
    if item.get("pool") == "main":
        return "main_blocked"
    return "other_pool"


def scan_literal_hits(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in items:
        scope = scope_for_item(item)
        for zone in ("stem_blocks", "answer_blocks_effective", "analysis_blocks"):
            for occurrence in iter_text_occurrences(item.get(zone) or [], zone):
                text = occurrence["text"]
                for match in LITERAL_RE.finditer(text):
                    hits.append(
                        candidate_row(
                            "literal_scan",
                            item_id=item.get("item_id"),
                            group_key=item.get("group_key"),
                            section_num=item.get("section_num"),
                            q_num=item.get("q_num"),
                            source_path=item.get("source_path"),
                            zone=zone,
                            block_path=occurrence.get("path"),
                            scope=scope,
                            literal_text=match.group(0),
                            literal_type="OMML" if match.group(0) == "[OMML]" else match.group(0).split(":", 1)[0].strip("["),
                            context=text[max(0, match.start() - 60) : match.end() + 60],
                        )
                    )
    return hits


def build_literal_targets(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_item: dict[str, dict[str, Any]] = {}
    for hit in hits:
        item_id = str(hit.get("item_id") or "")
        row = by_item.setdefault(
            item_id,
            candidate_row(
                "literal_scan_target",
                item_id=item_id,
                group_key=hit.get("group_key"),
                section_num=hit.get("section_num"),
                q_num=hit.get("q_num"),
                source_path=hit.get("source_path"),
                scope=hit.get("scope"),
                literal_count=0,
                literal_types=[],
            ),
        )
        row["literal_count"] += 1
        row["literal_types"] = sorted(set(row.get("literal_types") or []) | {hit.get("literal_type")})
    return sorted(by_item.values(), key=lambda r: (str(r.get("group_key")), str(r.get("section_num")), str(r.get("q_num")), str(r.get("item_id"))))


def source_paths_for_targets(targets: list[dict[str, Any]]) -> list[Path]:
    from scripts.build_batch6_artifacts import resolve_source_path, source_groups_by_key

    groups = source_groups_by_key()
    paths: set[str] = set()
    for target in targets:
        group = groups.get(str(target.get("group_key") or ""))
        if group:
            for source in group.get("unique_sources", []):
                raw = source.get("original_path") or source.get("path")
                if raw:
                    paths.add(resolve_source_path(str(raw)))
        elif target.get("source_path"):
            paths.add(resolve_source_path(str(target.get("source_path"))))
    return [Path(p) for p in sorted(paths) if Path(p).exists()]


def item_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("group_key") or ""), str(row.get("section_num") or ""), str(row.get("q_num") or ""))


def count_literals_in_item(row: dict[str, Any]) -> int:
    total = 0
    for zone in ("stem_blocks", "answer_blocks_effective", "analysis_blocks"):
        for occurrence in iter_text_occurrences(row.get(zone) or [], zone):
            total += len(LITERAL_RE.findall(occurrence["text"]))
    return total


def run_10c_literal_scan(items: list[dict[str, Any]], out_root: Path, skip_rerun: bool) -> dict[str, Any]:
    out_dir = out_root / "literal_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    hits = scan_literal_hits(items)
    targets = build_literal_targets(hits)
    write_jsonl(out_dir / "literal_hits_before.jsonl", hits)
    write_jsonl(out_dir / "literal_targets.jsonl", targets)
    source_paths = source_paths_for_targets(targets)
    source_list = out_dir / "target_source_list.txt"
    source_list.write_text("\n".join(str(p) for p in source_paths) + "\n", encoding="utf-8")
    rerun_root = out_dir / "ws1_segmentation" / "fixed_groups"
    if targets and not skip_rerun:
        from scripts.ws1_docx_extract_prototype import run_batch_extract_v4

        run_batch_extract_v4(SOURCE_ROOT, rerun_root, preview_limit=5, source_paths=source_paths)
        from scripts.build_batch6_artifacts import build_rerun_diff_outputs

        targets_by_id = {
            str(t["item_id"]): {
                "target_kind": "literal_scan",
                "problem_type": "literal_residual",
                "item_id": t["item_id"],
                "group_key": t.get("group_key"),
                "section_num": t.get("section_num"),
                "q_num": t.get("q_num"),
                "source_path": t.get("source_path"),
            }
            for t in targets
        }
        rerun_diff_summary = build_rerun_diff_outputs(out_dir, rerun_root, items, targets_by_id)
        fixed_path = out_dir / "ws1_segmentation" / "fixed_candidate_items.jsonl"
        fixed = load_jsonl(fixed_path)
        for row in fixed:
            row["schema_version"] = QA_SCHEMA_VERSION
            row["review_status"] = "pending_user_or_claude"
            row["reviewer"] = ""
        write_jsonl(fixed_path, fixed)
        candidate_by_key = {item_key(row): row for row in fixed}
    else:
        fixed = []
        candidate_by_key = {}
        rerun_diff_summary = {}

    after_rows = []
    target_items = {str(t["item_id"]): t for t in targets}
    for item in items:
        item_id = str(item.get("item_id") or "")
        if item_id not in target_items:
            continue
        candidate = candidate_by_key.get(item_key(item))
        after_count = count_literals_in_item(candidate) if candidate else count_literals_in_item(item)
        before_count = sum(1 for hit in hits if hit.get("item_id") == item_id)
        after_rows.append(
            candidate_row(
                "literal_after_comparison",
                old_item_id=item_id,
                new_item_id=candidate.get("item_id") if candidate else "",
                group_key=item.get("group_key"),
                section_num=item.get("section_num"),
                q_num=item.get("q_num"),
                before_literal_rows=before_count,
                after_literal_rows=after_count,
                rerun_status="skipped" if skip_rerun else ("rerun_matched" if candidate else "rerun_unmatched"),
                service_candidate_pass=after_count == 0,
            )
        )
    write_jsonl(out_dir / "literal_after_comparison.jsonl", after_rows)
    summary = {
        "literal_occurrences": len(hits),
        "target_items": len(targets),
        "target_groups": len({t.get("group_key") for t in targets}),
        "source_paths": len(source_paths),
        "rerun_skipped": skip_rerun,
        "candidate_items": len(fixed),
        "after_literal_rows": sum(row.get("after_literal_rows") or 0 for row in after_rows),
        "after_literal_rows_in_fixed_candidates": sum(count_literals_in_item(row) for row in fixed),
        "unmatched_target_items": sum(1 for row in after_rows if row.get("rerun_status") == "rerun_unmatched"),
        "unmatched_target_literal_rows": sum(
            row.get("after_literal_rows") or 0 for row in after_rows if row.get("rerun_status") == "rerun_unmatched"
        ),
    }
    if rerun_diff_summary:
        summary["raw_non_target_content_changes"] = rerun_diff_summary.get("raw_non_target_content_changes")
        summary["final_non_target_content_changes"] = rerun_diff_summary.get("final_non_target_content_changes")
        summary["final_collateral_rows"] = rerun_diff_summary.get("final_collateral_rows")
    write_json(out_dir / "literal_scan_summary.json", summary)
    return summary


ION_RULES = [
    {"rule_id": "ION_NH_PLUS_4", "pattern": r"(?<![A-Za-z0-9])NH[+＋]4(?![A-Za-z0-9])", "replacement": r"\\ce{NH4+}", "description": "NH+4 -> ammonium mhchem"},
    {"rule_id": "ION_SO4_2_MINUS", "pattern": r"(?<![A-Za-z0-9])SO(?:2[−\-－]4|42[−\-－])(?![A-Za-z0-9])", "replacement": r"\\ce{SO4^{2-}}", "description": "SO2-4/SO42- -> sulfate mhchem"},
    {"rule_id": "ION_CO3_2_MINUS", "pattern": r"(?<![A-Za-z0-9])(?:CO2/3|CO32[−\-－])(?![A-Za-z0-9])", "replacement": r"\\ce{CO3^{2-}}", "description": "CO2/3/CO32- -> carbonate mhchem"},
    {"rule_id": "ION_ALO2_MINUS", "pattern": r"(?<![A-Za-z0-9])AlO[−\-－]2(?![A-Za-z0-9])", "replacement": r"\\ce{AlO2-}", "description": "AlO-2 -> aluminate mhchem"},
    {"rule_id": "ION_BA2_PLUS", "pattern": r"(?<![A-Za-z0-9])Ba2[+＋](?![A-Za-z0-9])", "replacement": r"\\ce{Ba^{2+}}", "description": "Ba2+ -> barium mhchem"},
    {"rule_id": "ION_CA2_PLUS", "pattern": r"(?<![A-Za-z0-9])Ca2[+＋](?![A-Za-z0-9])", "replacement": r"\\ce{Ca^{2+}}", "description": "Ca2+ -> calcium mhchem"},
    {"rule_id": "ION_FE3_PLUS", "pattern": r"(?<![A-Za-z0-9])Fe3[+＋](?![A-Za-z0-9])", "replacement": r"\\ce{Fe^{3+}}", "description": "Fe3+ -> ferric mhchem"},
    {"rule_id": "ION_FE2_PLUS", "pattern": r"(?<![A-Za-z0-9])Fe2[+＋](?![A-Za-z0-9])", "replacement": r"\\ce{Fe^{2+}}", "description": "Fe2+ -> ferrous mhchem"},
    {"rule_id": "ELECTRON_CONFIG_EXPONENT", "pattern": r"\b([1-7][spdf])([1-9]|1[0-4])\b", "replacement": r"\1^{\2}", "description": "3d10 -> 3d^{10} in electron-configuration context"},
]


def apply_rule(text: str, rule: dict[str, str]) -> Iterable[tuple[re.Match[str], str]]:
    rx = re.compile(rule["pattern"])
    for match in rx.finditer(text):
        if rule["rule_id"] == "ELECTRON_CONFIG_EXPONENT":
            window = text[max(0, match.start() - 24) : match.end() + 24]
            if not re.search(r"电子|排布|轨道|基态|价电子", window):
                continue
        yield match, rx.sub(rule["replacement"], text, count=1)


def scan_text_ion_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, str]] = set()
    for item in items:
        for zone in ("stem_blocks", "answer_blocks_effective", "analysis_blocks"):
            for occurrence in iter_text_occurrences(item.get(zone) or [], zone):
                text = occurrence["text"]
                for rule in ION_RULES:
                    for match, suggested in apply_rule(text, rule):
                        key = (str(item.get("item_id")), zone, occurrence["path"], match.start(), rule["rule_id"])
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            candidate_row(
                                "text_ion_fix",
                                item_id=item.get("item_id"),
                                group_key=item.get("group_key"),
                                section_num=item.get("section_num"),
                                q_num=item.get("q_num"),
                                zone=zone,
                                block_path=occurrence.get("path"),
                                rule_id=rule["rule_id"],
                                matched_text=match.group(0),
                                suggested_replacement=re.sub(rule["pattern"], rule["replacement"], match.group(0), count=1),
                                original_text=text[:500],
                                suggested_rewrite=suggested[:500],
                                context=text[max(0, match.start() - 30) : match.end() + 30],
                            )
                        )
    return candidates


def run_10e_text_ion_fix(items: list[dict[str, Any]], out_root: Path) -> dict[str, Any]:
    out_dir = out_root / "text_ion_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = scan_text_ion_candidates(items)
    write_json(out_dir / "text_ion_fix_rules.json", ION_RULES)
    write_jsonl(out_dir / "text_ion_fix_candidates.jsonl", candidates)
    summary = {
        "candidate_rows": len(candidates),
        "items": len({row.get("item_id") for row in candidates}),
        "by_rule": dict(Counter(str(row.get("rule_id")) for row in candidates)),
    }
    write_json(out_dir / "text_ion_fix_summary.json", summary)
    return summary


def create_minimal_docx_with_omml(omml: str, path: Path) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"""
    doc = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><w:body><w:p><m:oMathPara>{omml}</m:oMathPara></w:p><w:sectPr/></w:body></w:document>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", doc)


def extract_mathml_via_libreoffice(omml: str, work_dir: Path, sha: str) -> tuple[str, str]:
    if not SOFFICE.exists():
        return "", f"soffice_missing:{SOFFICE}"
    docx = work_dir / f"{sha}.docx"
    out_dir = work_dir / "flat_xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    create_minimal_docx_with_omml(omml, docx)
    try:
        proc = subprocess.run(
            [str(SOFFICE), "--headless", "--convert-to", "xml", "--outdir", str(out_dir), str(docx)],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", "soffice_timeout_xml"
    xml_path = out_dir / f"{sha}.xml"
    if not xml_path.exists():
        fallback = out_dir / f"{docx.stem}.xml"
        xml_path = fallback if fallback.exists() else xml_path
    if not xml_path.exists():
        return "", (" ".join((proc.stdout or "", proc.stderr or "")).strip() or "soffice_no_xml")[:240]
    root = etree.fromstring(xml_path.read_bytes())
    math_nodes = root.xpath('//*[local-name()="math" and namespace-uri()="http://www.w3.org/1998/Math/MathML"]')
    if not math_nodes:
        return "", "mathml_not_found_in_flat_xml"
    return etree.tostring(math_nodes[0], encoding="unicode"), "ok"


def mathml_node_to_latex(node: etree._Element) -> str:
    local = etree.QName(node).localname
    text = "".join(node.itertext()).strip() if local in {"mi", "mn", "mo", "mtext"} else ""
    if local in {"mi", "mn", "mtext"}:
        return text
    if local == "mo":
        return {"⇌": r"\rightleftharpoons", "→": r"\to", "−": "-", "×": r"\times", "⋅": r"\cdot", "❑": ""}.get(text, text)
    children = [child for child in node if isinstance(child.tag, str)]
    if local in {"math", "semantics", "mrow"}:
        return "".join(mathml_node_to_latex(child) for child in children if etree.QName(child).localname != "annotation")
    if local == "msub" and len(children) >= 2:
        return f"{mathml_node_to_latex(children[0])}_{{{mathml_node_to_latex(children[1])}}}"
    if local == "msup" and len(children) >= 2:
        return f"{mathml_node_to_latex(children[0])}^{{{mathml_node_to_latex(children[1])}}}"
    if local == "msubsup" and len(children) >= 3:
        return f"{mathml_node_to_latex(children[0])}_{{{mathml_node_to_latex(children[1])}}}^{{{mathml_node_to_latex(children[2])}}}"
    if local == "mfrac" and len(children) >= 2:
        return f"\\frac{{{mathml_node_to_latex(children[0])}}}{{{mathml_node_to_latex(children[1])}}}"
    if local == "msqrt" and children:
        return f"\\sqrt{{{''.join(mathml_node_to_latex(child) for child in children)}}}"
    if local == "munderover" and children:
        base = mathml_node_to_latex(children[0])
        under = mathml_node_to_latex(children[1]) if len(children) > 1 else ""
        over = mathml_node_to_latex(children[2]) if len(children) > 2 else ""
        if over or under:
            return f"\\overset{{{over}}}{{\\underset{{{under}}}{{{base}}}}}"
        return base
    if local == "mover" and len(children) >= 2:
        return f"\\overset{{{mathml_node_to_latex(children[1])}}}{{{mathml_node_to_latex(children[0])}}}"
    if local == "munder" and len(children) >= 2:
        return f"\\underset{{{mathml_node_to_latex(children[1])}}}{{{mathml_node_to_latex(children[0])}}}"
    return "".join(mathml_node_to_latex(child) for child in children)


def mathml_to_latex(mathml: str) -> str:
    if not mathml:
        return ""
    root = etree.fromstring(mathml.encode("utf-8"))
    return re.sub(r"\s+", " ", mathml_node_to_latex(root)).strip()


def repair_cached_latex(latex: str | None) -> str:
    text = str(latex or "")
    text = text.translate(ASCII_MINUS)
    text = text.replace(r"\downright", r"\downarrow")
    text = text.replace(r"\upright", r"\uparrow")
    return text.strip()


def collect_omml_by_sha(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def walk(node: Any, item: dict[str, Any], zone: str, path: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "math_omml" and node.get("omml"):
                omml = str(node.get("omml") or "")
                sha = hashlib.sha1(omml.encode()).hexdigest()
                out.setdefault(
                    sha,
                    {
                        "omml_sha1": sha,
                        "omml": omml,
                        "occurrences": [],
                    },
                )["occurrences"].append(
                    {
                        "item_id": item.get("item_id"),
                        "group_key": item.get("group_key"),
                        "section_num": item.get("section_num"),
                        "q_num": item.get("q_num"),
                        "zone": zone,
                        "path": path,
                    }
                )
            for key, value in node.items():
                walk(value, item, zone, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                walk(value, item, zone, f"{path}[{idx}]")

    for item in items:
        for zone, field in (("stem", "stem_blocks"), ("answer", "answer_blocks_effective"), ("analysis", "analysis_blocks")):
            walk(item.get(field) or [], item, zone, field)
    return out


def run_10f_omml_retry(items: list[dict[str, Any]], out_root: Path) -> dict[str, Any]:
    out_dir = out_root / "omml_retry"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = load_jsonl(OMML_CACHE)
    failed = [row for row in cache if not row.get("katex_ok")]
    omml_by_sha = collect_omml_by_sha(items)
    candidates: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    mathml_rows: list[dict[str, Any]] = []
    node_modules = default_node_modules(out_dir)
    for row in failed:
        sha = str(row.get("omml_sha1") or "")
        source = omml_by_sha.get(sha, {})
        attempts: list[dict[str, Any]] = []
        repaired = repair_cached_latex(row.get("latex"))
        compile_result = validate_latex_katex(repaired, node_modules=node_modules) if repaired else {"ok": False, "error": "empty_cached_latex"}
        attempts.append({"method": "cached_latex_sanitized", "latex": repaired, "compile_result": compile_result})
        chosen_latex = repaired if compile_result.get("ok") else ""
        chosen_compile = compile_result
        mathml = ""
        mathml_status = "not_attempted"
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
        base = candidate_row(
            "omml_retry",
            omml_sha1=sha,
            old_latex=row.get("latex"),
            old_ok=row.get("ok"),
            old_katex_ok=row.get("katex_ok"),
            occurrences=source.get("occurrences") or [],
            attempts=attempts,
            suggested_latex=chosen_latex,
            compile_result=chosen_compile,
        )
        if chosen_latex and chosen_compile.get("ok"):
            candidates.append(base)
        else:
            base["manual_reason"] = mathml_status if mathml_status != "ok" else (chosen_compile.get("error") or "compile_failed")
            manual.append(base)
    write_jsonl(out_dir / "omml_retry_candidates.jsonl", candidates)
    write_jsonl(out_dir / "omml_retry_manual_queue.jsonl", manual)
    write_jsonl(out_dir / "omml_retry_mathml_samples.jsonl", mathml_rows)
    summary = {"failed_cache_rows": len(failed), "candidate_rows": len(candidates), "manual_queue_rows": len(manual)}
    write_json(out_dir / "omml_retry_summary.json", summary)
    return summary


def split_option_text(text: str) -> list[str]:
    matches = list(OPTION_MARK_RE.finditer(text))
    if len(matches) < 2:
        return [text]
    pieces: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
    prefix = text[: matches[0].start()].strip()
    if prefix and pieces:
        pieces[0] = prefix + pieces[0]
    return pieces or [text]


def scan_option_split_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in items:
        for zone in ("stem_blocks", "answer_blocks_effective"):
            for occurrence in iter_text_occurrences(item.get(zone) or [], zone):
                text = occurrence["text"]
                pieces = split_option_text(text)
                if len(pieces) < 2:
                    continue
                candidates.append(
                    candidate_row(
                        "option_split",
                        item_id=item.get("item_id"),
                        group_key=item.get("group_key"),
                        section_num=item.get("section_num"),
                        q_num=item.get("q_num"),
                        zone=zone,
                        block_path=occurrence.get("path"),
                        original_text=text[:800],
                        suggested_segments=pieces,
                        split_count=len(pieces),
                    )
                )
    return candidates


def run_10g_option_split(items: list[dict[str, Any]], out_root: Path) -> dict[str, Any]:
    out_dir = out_root / "option_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = scan_option_split_candidates(items)
    write_jsonl(out_dir / "option_split_candidates.jsonl", candidates)
    summary = {
        "candidate_rows": len(candidates),
        "items": len({row.get("item_id") for row in candidates}),
        "by_zone": dict(Counter(str(row.get("zone")) for row in candidates)),
    }
    write_json(out_dir / "option_split_summary.json", summary)
    return summary


def validate_outputs(out_root: Path) -> dict[str, Any]:
    reviewer_codex_hits = 0
    candidate_schema_bad = 0
    transcript_schema_bad = 0
    transcript_delivery_names = {
        "formula_latex_candidates.jsonl",
        "transcript_candidates.jsonl",
        "formula_backfill_candidates.jsonl",
        "latex_form_fix_candidates.jsonl",
    }
    for path in out_root.rglob("*.jsonl"):
        for row in load_jsonl(path):
            if str(row.get("reviewer") or "").startswith("codex_"):
                reviewer_codex_hits += 1
            if row.get("candidate_kind") and row.get("schema_version") != QA_SCHEMA_VERSION:
                candidate_schema_bad += 1
            if (
                path.name in transcript_delivery_names
                and row.get("asset_hash")
                and not row.get("candidate_kind")
                and row.get("schema_version") != WS2_SCHEMA_VERSION
            ):
                transcript_schema_bad += 1
    return {
        "reviewer_codex_hits": reviewer_codex_hits,
        "candidate_schema_bad": candidate_schema_bad,
        "transcript_schema_bad": transcript_schema_bad,
    }


def write_batch_report(out_root: Path, summaries: dict[str, Any], started: float, official_before: dict[str, str], official_after: dict[str, str]) -> None:
    total_cost = 0.0
    for summary in summaries.values():
        if isinstance(summary, dict):
            total_cost += float(summary.get("cost_yuan") or 0.0)
    validation = validate_outputs(out_root)
    lines = [
        "# Batch 10 QA-1 Report",
        "",
        f"- elapsed_sec: {round(time.time() - started, 1)}",
        f"- output_root: `{out_root}`",
        f"- measured_vision_cost_yuan: {round(total_cost, 4)}",
        f"- official_item_bank_md5_unchanged: {official_before == official_after}",
        f"- reviewer_codex_hits: {validation['reviewer_codex_hits']}",
        f"- candidate_schema_bad: {validation['candidate_schema_bad']}",
        f"- transcript_schema_bad: {validation['transcript_schema_bad']}",
        "",
    ]
    for key in ["10a", "10b", "10c", "10d", "10e", "10f", "10g"]:
        summary = summaries.get(key, {})
        lines.extend([f"## {key}", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    lines.extend(
        [
            "## Hard Gates",
            "",
            "- official data: read-only; writes are limited to `/tmp/yher_batch10_qa1` and runner/test files.",
            "- reviewer discipline: candidate/manual rows use `review_status=pending_user_or_claude` and blank `reviewer`.",
            "- failure prompt text: latex/transcript rows matching the required prompt-failure regex are written to rejected files.",
            "- leak gate: transcript leak hits are written to `*_leak_rejected.jsonl`.",
        ]
    )
    (out_root / "BATCH10_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_delivery_dirs(out_root: Path) -> None:
    for name in [
        "answer_zone_assets",
        "formula_backfill",
        "literal_scan",
        "latex_form_fix",
        "text_ion_fix",
        "omml_retry",
        "option_split",
    ]:
        (out_root / name).mkdir(parents=True, exist_ok=True)
    legacy_cache = out_root / "api_cache"
    if legacy_cache.exists():
        try:
            legacy_cache.rmdir()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-literal-rerun", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    out_root = args.out_root
    ensure_delivery_dirs(out_root)
    official_before = {str(path): file_md5(path) for path in [V4_ITEMS, WS2_TRANSCRIPTS, WS2_MEDIA_REF_MAP, OMML_CACHE] if path.exists()}
    items = load_jsonl(V4_ITEMS)
    client = None if args.skip_vision else build_vision_client()
    summaries: dict[str, Any] = {}
    summaries["10a"] = run_10a_answer_zone(items, out_root, client, args.workers, args.skip_vision)
    summaries["10b"] = run_10b_formula_backfill(out_root, client, args.workers, args.skip_vision)
    summaries["10c"] = run_10c_literal_scan(items, out_root, args.skip_literal_rerun)
    summaries["10d"] = run_10d_latex_form_fix(out_root, client, args.workers, args.skip_vision)
    summaries["10e"] = run_10e_text_ion_fix(items, out_root)
    summaries["10f"] = run_10f_omml_retry(items, out_root)
    summaries["10g"] = run_10g_option_split(items, out_root)
    official_after = {str(path): file_md5(path) for path in [V4_ITEMS, WS2_TRANSCRIPTS, WS2_MEDIA_REF_MAP, OMML_CACHE] if path.exists()}
    write_json(out_root / "batch10_summary.json", summaries)
    write_batch_report(out_root, summaries, started, official_before, official_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
