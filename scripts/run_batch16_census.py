#!/usr/bin/env python3
"""Batch16 QA-5 L0 package runner.

Reads official v4 data and writes all deliverables under /tmp/yher_batch16_qa5.
It never applies changes to official item bank, ref-map, transcripts, assets, or
R5 data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TOOLS_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data.item_bank_v4 import iter_service_items, load_service_pool, load_usability_r5  # noqa: E402
from scripts.qa_item_auditor import machine_audit_item  # noqa: E402
from scripts.ws1_docx_extract_prototype import (  # noqa: E402
    blocks_text,
    convert_doc_to_docx,
    merge_answers_into_questions,
    parse_docx_model,
)

OUT_ROOT = Path("/tmp/yher_batch16_qa5")
WS1_ROOT = REPO_ROOT / "data" / "ws1_batch_v4_20260703"
V4_DIR = REPO_ROOT / "data" / "item_bank" / "v4"
REF_MAP = V4_DIR / "ws2_media_ref_map_v1.jsonl"
TRANSCRIPTS = V4_DIR / "ws2_asset_transcripts_v1.jsonl"
OMML_CACHE = V4_DIR / "ws2_omml_latex_cache_v1.jsonl"
R5_PATH = V4_DIR / "usability_r5_v1.jsonl"
B14_USABILITY = Path("/tmp/yher_batch14_qa2r/usability_audit.jsonl")
B15_DIR = Path("/tmp/yher_batch15_qa4")
B15_SIGNED_VARIANTS = B15_DIR / "15c_variant_dispositions_signed.jsonl"
B15_15D = B15_DIR / "15d_split_candidates.jsonl"
B15_15E = B15_DIR / "15e_asset_dimensions.jsonl"
PRECISION_GOLD = Path("/tmp/yher_b16_precision_gold.jsonl")
RECALL_GOLD = Path("/tmp/yher_b16_recall_gold.jsonl")

B16_DIMS = [
    "subanswer_not_hollow",
    "no_hollow_mention",
    "no_fragment_literal",
    "stem_answer_sub_coverage",
    "answer_stem_match_flag",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stem_md5(item: dict[str, Any]) -> str:
    raw = json.dumps(item.get("stem_blocks") or [], ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def compact_len(text: str) -> int:
    return len(re.sub(r"[\s　,，.。;；:：、（）()\[\]【】]+", "", text or ""))


def short(text: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def pending_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["review_status"] = "pending_user_or_claude"
    out["reviewer"] = ""
    return out


def resolve_source_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    raw = Path(str(raw_path))
    if raw.is_absolute():
        return raw if raw.exists() else None
    candidates = [
        (REPO_ROOT / raw).resolve(),
        (TOOLS_ROOT / raw).resolve(),
    ]
    raw_text = str(raw_path)
    if raw_text.startswith("../"):
        candidates.append((TOOLS_ROOT / raw_text[3:]).resolve())
    candidates.append((Path("/tmp/yher_ws1_batch_v4/_converted_docx") / raw.name).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_sources_by_group(root: Path = WS1_ROOT) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return groups
    for path in root.glob("*/sources.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        group_key = row.get("group_key")
        if group_key:
            groups[str(group_key)] = row
    return groups


def checked_group_sources(group_sources: dict[str, Any] | None) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for source in (group_sources or {}).get("unique_sources") or []:
        path = resolve_source_path(source.get("path") or source.get("original_path"))
        checked.append(
            {
                "role": source.get("role"),
                "status": source.get("status"),
                "path": source.get("path") or source.get("original_path"),
                "resolved_path": str(path) if path else "",
                "exists": bool(path),
                "answer_marker_count": int(source.get("answer_marker_count") or 0),
            }
        )
    return checked


def answer_source_candidates(group_sources: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in (group_sources or {}).get("unique_sources") or []:
        role = str(source.get("role") or "")
        markers = int(source.get("answer_marker_count") or 0)
        if source.get("status") != "ok":
            continue
        if role not in {"analysis", "answer_key", "answer", "answer_only"} and markers <= 0:
            continue
        path = resolve_source_path(source.get("path") or source.get("original_path"))
        rows.append(
            {
                "role": role,
                "path": source.get("path") or source.get("original_path"),
                "resolved_path": str(path) if path else "",
                "path_exists": bool(path),
                "answer_marker_count": markers,
                "file_name": source.get("file_name") or source.get("name"),
            }
        )
    rows.sort(key=lambda row: (not row["path_exists"], -row["answer_marker_count"], row["role"]))
    return rows


def question_source_candidates(group_sources: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in (group_sources or {}).get("unique_sources") or []:
        if source.get("status") != "ok":
            continue
        role = str(source.get("role") or "")
        if role not in {"question_source", "analysis", "unknown"}:
            continue
        path = resolve_source_path(source.get("path") or source.get("original_path"))
        rows.append(
            {
                "role": role,
                "path": source.get("path") or source.get("original_path"),
                "resolved_path": str(path) if path else "",
                "path_exists": bool(path),
                "answer_marker_count": int(source.get("answer_marker_count") or 0),
            }
        )
    return rows


def iter_segments(blocks: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for block in blocks or []:
        para = block.get("para") if isinstance(block, dict) else block if isinstance(block, list) else []
        for seg in para or []:
            if isinstance(seg, dict):
                yield seg


def item_media_refs(item: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for field in ("stem_blocks", "answer_blocks_effective", "analysis_blocks"):
        for seg in iter_segments(item.get(field) or []):
            media = seg.get("media")
            if media:
                refs.append({"field": field, "media": str(media), "type": seg.get("type")})
    return refs


def load_ref_map_index(path: Path = REF_MAP) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line_no, row in enumerate(read_jsonl(path), start=1):
        group_key = str(row.get("group_key") or "")
        media = str(row.get("media") or "")
        if not group_key or not media:
            continue
        hit = {k: row.get(k) for k in ("asset_hash", "zones", "in_ws2_manifest")}
        hit["line_no"] = line_no
        index[(group_key, media)].append(hit)
    return index


def ref_map_hits_for_item(item: dict[str, Any], ref_map_index: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    group_key = str(item.get("group_key") or "")
    for ref in item_media_refs(item):
        for hit in ref_map_index.get((group_key, ref["media"]), []):
            hits.append({**ref, **hit})
    return hits


def load_signed_source_dead_ids(path: Path = B15_SIGNED_VARIANTS) -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        if row.get("disposition") == "exclude":
            ids.add(str(row.get("item_id") or ""))
    return ids


def classify_census_item(
    item: dict[str, Any],
    *,
    r5_row: dict[str, Any] | None,
    group_sources: dict[str, Any] | None,
    ref_map_index: dict[tuple[str, str], list[dict[str, Any]]],
    signed_source_dead_ids: set[str],
    usability_row: dict[str, Any] | None,
) -> dict[str, Any]:
    item_id = str(item.get("item_id") or "")
    answer_sources = answer_source_candidates(group_sources)
    question_sources = question_source_candidates(group_sources)
    ref_hits = ref_map_hits_for_item(item, ref_map_index)
    group_checked = checked_group_sources(group_sources)
    r5_reason = (r5_row or {}).get("r5_block_reason") or (r5_row or {}).get("usability_pool") or ""
    failed_dims = list((usability_row or {}).get("machine_failed_dimensions") or [])
    evidence = {
        "r5_block_reason": r5_reason,
        "usability_pool": (r5_row or {}).get("usability_pool"),
        "batch14_machine_failed_dimensions": failed_dims,
        "group_sources_checked": group_checked,
        "answer_source_candidates": answer_sources,
        "question_source_candidates": question_sources,
        "ref_map_hits": ref_hits[:20],
    }

    if item_id in signed_source_dead_ids or r5_reason == "exclusion:variant_bank_answer_misattributed":
        return pending_row(
            {
                "schema_version": "batch16_16a_census_v1",
                "item_id": item_id,
                "group_key": item.get("group_key", ""),
                "section_num": item.get("section_num"),
                "q_num": item.get("q_num"),
                "census_class": "source_dead",
                "subclass": "variant_bank_answer_misattributed",
                "evidence": {**evidence, "source_dead_basis": "batch15_15c_claude_signed_exclude"},
            }
        )

    subclasses: list[str] = []
    if any(row["path_exists"] and row["answer_marker_count"] > 0 for row in answer_sources):
        subclasses.append("hollow_group_source")
    if ref_hits:
        subclasses.append("media_linkable")
    if any(dim in failed_dims for dim in ("option_complete", "option_no_sticky", "stem_not_truncated")):
        subclasses.append("sticky_split_residue")
    if any(dim in failed_dims for dim in ("image_dimensions_normal", "table_complete")):
        subclasses.append("dimension_metadata_only")
    if r5_reason.startswith("exclusion:content_leak") or r5_reason.startswith("exclusion:partial_answer"):
        subclasses.append("hollow_group_source" if answer_sources else "dimension_metadata_only")
    if r5_reason.startswith("pool:fixable") and not subclasses:
        subclasses.append("dimension_metadata_only")

    if subclasses:
        return pending_row(
            {
                "schema_version": "batch16_16a_census_v1",
                "item_id": item_id,
                "group_key": item.get("group_key", ""),
                "section_num": item.get("section_num"),
                "q_num": item.get("q_num"),
                "census_class": "repairable_known_route",
                "subclass": sorted(set(subclasses)),
                "evidence": evidence,
            }
        )

    unresolved_media = [ref for ref in item_media_refs(item) if not ref_map_index.get((str(item.get("group_key") or ""), ref["media"]))]
    if unresolved_media and any(row["path_exists"] for row in question_sources):
        return pending_row(
            {
                "schema_version": "batch16_16a_census_v1",
                "item_id": item_id,
                "group_key": item.get("group_key", ""),
                "section_num": item.get("section_num"),
                "q_num": item.get("q_num"),
                "census_class": "repairable_visual_crop",
                "subclass": ["b13_manual_reanchor"],
                "estimated_pages": "requires_group_docx_render",
                "evidence": {**evidence, "unresolved_media_refs": unresolved_media[:20]},
            }
        )

    return pending_row(
        {
            "schema_version": "batch16_16a_census_v1",
            "item_id": item_id,
            "group_key": item.get("group_key", ""),
            "section_num": item.get("section_num"),
            "q_num": item.get("q_num"),
            "census_class": "unknown",
            "subclass": ["needs_claude_sampling"],
            "evidence": evidence,
        }
    )


def build_census(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    service_items = list(iter_service_items(apply_r5=False))
    items_by_id = {row["item_id"]: row for row in service_items}
    r5 = load_usability_r5()
    b14 = {row["item_id"]: row for row in read_jsonl(B14_USABILITY) if row.get("item_id")}
    sources = load_sources_by_group()
    ref_index = load_ref_map_index()
    source_dead_ids = load_signed_source_dead_ids()

    target_ids = [item_id for item_id, row in r5.items() if row.get("r5_serve") is not True and item_id in items_by_id]
    rows = [
        classify_census_item(
            items_by_id[item_id],
            r5_row=r5.get(item_id),
            group_sources=sources.get(str(items_by_id[item_id].get("group_key") or "")),
            ref_map_index=ref_index,
            signed_source_dead_ids=source_dead_ids,
            usability_row=b14.get(item_id),
        )
        for item_id in sorted(target_ids)
    ]
    class_counts = Counter(row["census_class"] for row in rows)
    subclass_counts: Counter[str] = Counter()
    for row in rows:
        subclasses = row.get("subclass")
        if isinstance(subclasses, list):
            subclass_counts.update(str(x) for x in subclasses)
        else:
            subclass_counts[str(subclasses)] += 1
    source_dead = [row for row in rows if row["census_class"] == "source_dead"]
    summary = {
        "schema_version": "batch16_16a_summary_v1",
        "audit_pool_2526": len(service_items),
        "r5_serve": sum(1 for row in r5.values() if row.get("r5_serve") is True and row.get("item_id") in items_by_id),
        "census_rows": len(rows),
        "expected_census_rows": len(service_items) - sum(1 for row in r5.values() if row.get("r5_serve") is True and row.get("item_id") in items_by_id),
        "class_counts": dict(class_counts),
        "subclass_counts": dict(subclass_counts),
        "source_dead_count": len(source_dead),
        "true_denominator_2526_minus_source_dead": len(service_items) - len(source_dead),
    }
    write_jsonl(out_dir / "16a_census.jsonl", rows)
    write_jsonl(out_dir / "16a_source_dead.jsonl", source_dead)
    write_json(out_dir / "16a_summary.json", summary)
    return rows, summary


def run_gold_regression(out_dir: Path) -> dict[str, Any]:
    service = load_service_pool(apply_r5=False)
    r5 = load_usability_r5()
    precision_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []

    for gold in read_jsonl(PRECISION_GOLD):
        item = service.get(str(gold.get("item_id") or ""))
        failed: list[str] = []
        evidence: dict[str, Any] = {}
        if item:
            audit = machine_audit_item(item, latex_compile_map={})
            failed = [dim for dim in B16_DIMS if not audit["dimensions"].get(dim)]
            evidence = {dim: audit["evidence"].get(dim, []) for dim in failed}
        precision_rows.append({**gold, "missing_item": item is None, "failed_b16_dims": failed, "evidence": evidence})

    for gold in read_jsonl(RECALL_GOLD):
        item_id = str(gold.get("item_id") or "")
        item = service.get(item_id)
        machine_failed: list[str] = []
        if item:
            audit = machine_audit_item(item, latex_compile_map={})
            machine_failed = list(audit.get("machine_failed_dimensions") or [])
        r5_row = r5.get(item_id) or {}
        r5_existing_hit = r5_row.get("r5_serve") is False and bool(r5_row.get("r5_block_reason"))
        recall_rows.append(
            {
                **gold,
                "missing_item": item is None,
                "machine_failed_dimensions": machine_failed,
                "auditor_hit": bool(machine_failed),
                "r5_existing_hit": r5_existing_hit,
                "r5_block_reason": r5_row.get("r5_block_reason"),
                "hit": bool(machine_failed) or r5_existing_hit,
            }
        )

    precision_hits = [row for row in precision_rows if row["failed_b16_dims"] or row["missing_item"]]
    recall_misses = [row for row in recall_rows if not row["hit"] or row["missing_item"]]
    result = {
        "schema_version": "batch16_16b_gold_regression_v1",
        "precision_total": len(precision_rows),
        "precision_hits": len(precision_hits),
        "precision_hit_by_dim": dict(Counter(dim for row in precision_rows for dim in row["failed_b16_dims"])),
        "recall_total": len(recall_rows),
        "recall_hits": len(recall_rows) - len(recall_misses),
        "recall_auditor_hits": sum(1 for row in recall_rows if row["auditor_hit"]),
        "recall_r5_existing_hits": sum(1 for row in recall_rows if (not row["auditor_hit"]) and row["r5_existing_hit"]),
        "recall_misses": len(recall_misses),
        "passed": len(precision_hits) == 0 and len(recall_misses) == 0,
        "precision_details": precision_rows,
        "recall_details": recall_rows,
    }
    write_json(out_dir / "16b_gold_regression.json", result)
    write_16b_dims_diff(out_dir / "16b_dims_diff.md", result)
    return result


def write_16b_dims_diff(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Batch16 16b Node-Aware Auditor Regression",
        "",
        "## Gold Gates",
        "",
        f"- Precision gold rows: {result['precision_total']}",
        f"- Precision B16-dim hits: {result['precision_hits']}",
        f"- Recall gold rows: {result['recall_total']}",
        f"- Recall total hits: {result['recall_hits']}",
        f"- Recall auditor hits: {result['recall_auditor_hits']}",
        f"- Recall existing R5 exclusion hits: {result['recall_r5_existing_hits']}",
        f"- Recall misses: {result['recall_misses']}",
        f"- Passed: {result['passed']}",
        "",
        "## Precision Hit By Dimension",
        "",
        json.dumps(result["precision_hit_by_dim"], ensure_ascii=False, indent=2, sort_keys=True),
        "",
    ]
    if result["precision_hits"]:
        lines.extend(["## Precision Hits", ""])
        for row in result["precision_details"]:
            if row["failed_b16_dims"] or row["missing_item"]:
                lines.append(f"- `{row.get('item_id')}`: {row.get('failed_b16_dims')} missing={row.get('missing_item')}")
    if result["recall_misses"]:
        lines.extend(["", "## Recall Misses", ""])
        for row in result["recall_details"]:
            if not row["hit"] or row["missing_item"]:
                lines.append(f"- `{row.get('item_id')}`: must_flag={row.get('must_flag')} missing={row.get('missing_item')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_rows_for_merge(answer_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in answer_sources:
        if not source.get("path_exists"):
            continue
        rows.append(
            {
                "role": source.get("role"),
                "path": source.get("resolved_path"),
                "original_path": source.get("path"),
            }
        )
    return rows


def answer_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    flat: list[dict[str, Any]] = []
    for para in blocks or []:
        flat.extend(para.get("para") or [])
    return blocks_text(flat)


def media_refs_from_blocks(blocks: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for para in blocks or []:
        for seg in para.get("para") or []:
            if isinstance(seg, dict) and seg.get("media"):
                refs.append(str(seg["media"]))
    return refs


def has_placeholder_text(blocks: list[dict[str, Any]]) -> bool:
    for para in blocks or []:
        for seg in para.get("para") or []:
            if not isinstance(seg, dict) or seg.get("type") != "text":
                continue
            text = re.sub(r"\s+", "", str(seg.get("text") or ""))
            if text in {"图", "略", "[image]", "image"}:
                return True
            if "[image]" in text.lower():
                return True
    return False


def build_refill_for_item(
    item: dict[str, Any],
    *,
    group_sources: dict[str, Any] | None,
    ref_map_index: dict[tuple[str, str], list[dict[str, Any]]],
    evidence_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    item_id = str(item.get("item_id") or "")
    answer_sources = answer_source_candidates(group_sources)
    usable_sources = source_rows_for_merge(answer_sources)
    base = {
        "schema_version": "batch16_16c_refill_candidate_v1",
        "item_id": item_id,
        "group_key": item.get("group_key", ""),
        "section_num": item.get("section_num"),
        "q_num": item.get("q_num"),
        "stem_md5_before": stem_md5(item),
        "stem_md5_after": stem_md5(item),
        "answer_source_candidates": answer_sources,
    }
    if not usable_sources:
        return None, pending_row({**base, "reason": "no_group_answer_source_on_disk"})

    target = {
        "q_num": item.get("q_num"),
        "section_num": item.get("section_num"),
        "question_id": f"{item.get('section_num') or 0}-{item.get('q_num')}",
        "answer_blocks": [],
        "analysis_blocks": [],
    }
    assets_dir = evidence_dir / item_id / "assets"
    merge_result = merge_answers_into_questions([target], usable_sources, assets_dir)
    if not target.get("answer_blocks"):
        # Fallback for documents whose q ids do not match the generated compound id.
        for source in usable_sources:
            path = Path(str(source["path"]))
            try:
                docx = path
                if path.suffix.lower() == ".doc":
                    converted, _error = convert_doc_to_docx(path, evidence_dir / "_converted_docx")
                    if converted:
                        docx = converted
                model = parse_docx_model(docx)
            except Exception:
                continue
            matches = [
                q
                for q in model.get("questions", [])
                if q.get("q_num") == item.get("q_num")
                and (item.get("section_num") is None or q.get("section_num") == item.get("section_num"))
                and q.get("answer_blocks")
            ]
            if len(matches) == 1:
                target["answer_blocks"] = matches[0].get("answer_blocks") or []
                target["analysis_blocks"] = matches[0].get("analysis_blocks") or []
                target["answer_source_path"] = source.get("original_path") or source.get("path")
                target["answer_source_role"] = source.get("role")
                break

    if not target.get("answer_blocks"):
        return None, pending_row({**base, "reason": "answer_not_located_in_group_source", "merge_result": merge_result})

    old_text = answer_text_from_blocks(item.get("answer_blocks_effective") or [])
    new_text = answer_text_from_blocks(target.get("answer_blocks") or [])
    old_len = compact_len(old_text)
    new_len = compact_len(new_text)
    media_refs = media_refs_from_blocks(target.get("answer_blocks") or []) + media_refs_from_blocks(target.get("analysis_blocks") or [])
    media_hits: dict[str, list[dict[str, Any]]] = {
        media: ref_map_index.get((str(item.get("group_key") or ""), media), []) for media in media_refs
    }
    missing_media = [media for media, hits in media_hits.items() if not hits]

    evidence = {
        "merge_result": merge_result,
        "answer_source_path": target.get("answer_source_path"),
        "answer_source_role": target.get("answer_source_role"),
        "old_answer_preview": short(old_text),
        "new_answer_preview": short(new_text),
        "old_answer_compact_len": old_len,
        "new_answer_compact_len": new_len,
        "media_refs": media_refs,
        "media_ref_hits": media_hits,
    }
    write_json(evidence_dir / item_id / "refill_evidence.json", evidence)

    if new_len == 0:
        return None, pending_row({**base, "reason": "new_answer_empty", "evidence": evidence})
    if has_placeholder_text(target.get("answer_blocks") or []):
        return None, pending_row({**base, "reason": "placeholder_text_in_new_answer", "evidence": evidence})
    if old_len and new_len < old_len * 0.6:
        return None, pending_row({**base, "reason": "new_answer_shrink_below_60_percent_guard", "evidence": evidence})
    if missing_media:
        return None, pending_row({**base, "reason": "media_ref_not_linked", "missing_media_refs": missing_media, "evidence": evidence})

    candidate = pending_row(
        {
            **base,
            "candidate_kind": "answer_refill_from_group_source",
            "new_answer_blocks": target.get("answer_blocks") or [],
            "new_analysis_blocks": target.get("analysis_blocks") or [],
            "evidence": evidence,
        }
    )
    return candidate, None


def build_16c_outputs(census_rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    service = load_service_pool(apply_r5=False)
    sources = load_sources_by_group()
    ref_index = load_ref_map_index()
    source_dead_ids = {row["item_id"] for row in census_rows if row["census_class"] == "source_dead"}
    recall_targets = [row for row in read_jsonl(RECALL_GOLD) if row.get("item_id") not in source_dead_ids]
    candidates: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    evidence_dir = out_dir / "16c_evidence"

    for target in recall_targets:
        item_id = str(target.get("item_id") or "")
        item = service.get(item_id)
        if item is None:
            manual.append(pending_row({"schema_version": "batch16_16c_refill_candidate_v1", "item_id": item_id, "reason": "target_not_in_audit_pool"}))
            continue
        candidate, manual_row = build_refill_for_item(
            item,
            group_sources=sources.get(str(item.get("group_key") or "")),
            ref_map_index=ref_index,
            evidence_dir=evidence_dir,
        )
        if candidate:
            candidates.append(candidate)
        if manual_row:
            manual.append(manual_row)

    write_jsonl(out_dir / "16c_refill_candidates.jsonl", candidates)
    write_jsonl(out_dir / "16c_manual.jsonl", manual)
    summary = {
        "targets": len(recall_targets),
        "candidates": len(candidates),
        "manual": len(manual),
        "manual_reason_counts": dict(Counter(row.get("reason", "") for row in manual)),
    }
    return candidates, manual, summary


def copy_16d_passthrough(out_dir: Path) -> dict[str, Any]:
    dst = out_dir / "16d_passthrough"
    dst.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Any] = {}
    for src in (B15_15D, B15_15E):
        if src.exists():
            target = dst / src.name
            shutil.copy2(src, target)
            copied[src.name] = {"rows": len(read_jsonl(target)), "path": str(target)}
        else:
            copied[src.name] = {"rows": 0, "missing": True}
    return copied


def governance_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = list(rows)
    return {
        "rows": len(rows),
        "reviewer_nonempty": sum(1 for row in rows if row.get("reviewer") not in ("", None)),
        "bad_review_status": sum(1 for row in rows if row.get("review_status") != "pending_user_or_claude"),
        "codex_reviewer_hits": sum(1 for row in rows if "codex" in str(row.get("reviewer") or "").lower()),
    }


def write_report(
    out_dir: Path,
    census_summary: dict[str, Any],
    gold: dict[str, Any],
    refill_summary: dict[str, Any] | None,
    passthrough: dict[str, Any] | None,
    validation: dict[str, Any],
) -> None:
    class_counts = census_summary.get("class_counts", {})
    lines = [
        "# Batch16 QA-5 Report",
        "",
        "## 16a 不可修宇宙普查",
        "",
        "| 项 | 数量 |",
        "|---|---:|",
        f"| 审计口径 | {census_summary.get('audit_pool_2526')} |",
        f"| R5 serve | {census_summary.get('r5_serve')} |",
        f"| 普查行数 | {census_summary.get('census_rows')} |",
        f"| source_dead | {census_summary.get('source_dead_count')} |",
        f"| 真实分母 2526-source_dead | {census_summary.get('true_denominator_2526_minus_source_dead')} |",
        "",
        "| 类别 | 数量 |",
        "|---|---:|",
    ]
    for key, value in sorted(class_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## 16b 双金标硬门",
            "",
            f"- precision: {gold.get('precision_hits')}/{gold.get('precision_total')} 命中",
            f"- recall: {gold.get('recall_hits')}/{gold.get('recall_total')} 命中",
            f"- recall auditor hits: {gold.get('recall_auditor_hits')}",
            f"- recall existing R5 exclusion hits: {gold.get('recall_r5_existing_hits')}",
            f"- passed: {gold.get('passed')}",
            "",
        ]
    )
    if refill_summary is not None:
        lines.extend(
            [
                "## 16c 空心回链候选",
                "",
                f"- targets: {refill_summary.get('targets')}",
                f"- candidates: {refill_summary.get('candidates')}",
                f"- manual: {refill_summary.get('manual')}",
                f"- manual reasons: `{refill_summary.get('manual_reason_counts')}`",
                "",
            ]
        )
    if passthrough is not None:
        lines.extend(["## 16d Passthrough", ""])
        for name, info in sorted(passthrough.items()):
            lines.append(f"- `{name}`: {info}")
        lines.append("")
    lines.extend(
        [
            "## Validation",
            "",
            "```json",
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    (out_dir / "BATCH16_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(out_dir: Path = OUT_ROOT) -> dict[str, Any]:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    official_paths = [REF_MAP, TRANSCRIPTS, OMML_CACHE, R5_PATH, V4_DIR / "chemistry_v4_1_3329.jsonl"]
    md5_before = {str(path): file_md5(path) for path in official_paths if path.exists()}

    census_rows, census_summary = build_census(out_dir)
    gold = run_gold_regression(out_dir)
    refill_summary: dict[str, Any] | None = None
    passthrough: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []

    if gold["passed"]:
        candidates, manual, refill_summary = build_16c_outputs(census_rows, out_dir)
        passthrough = copy_16d_passthrough(out_dir)

    md5_after = {str(path): file_md5(path) for path in official_paths if path.exists()}
    all_candidate_rows = [*census_rows, *candidates, *manual]
    validation = {
        "schema_version": "batch16_validation_v1",
        "elapsed_sec": round(time.time() - started, 2),
        "official_md5_unchanged": md5_before == md5_after,
        "official_md5_before": md5_before,
        "official_md5_after": md5_after,
        "census_rows": len(census_rows),
        "census_expected_rows": census_summary.get("expected_census_rows"),
        "census_full_coverage": len(census_rows) == census_summary.get("expected_census_rows") == 1319,
        "census_class_counts": census_summary.get("class_counts"),
        "true_denominator": census_summary.get("true_denominator_2526_minus_source_dead"),
        "gold_passed": gold.get("passed"),
        "precision_hits": gold.get("precision_hits"),
        "recall_misses": gold.get("recall_misses"),
        "ran_16c_16d": bool(gold.get("passed")),
        "governance": governance_counts(all_candidate_rows),
    }
    if refill_summary is not None:
        validation["16c"] = refill_summary
        validation["16c_placeholder_text_violations"] = sum(
            1 for row in candidates if has_placeholder_text(row.get("new_answer_blocks") or [])
        )
        validation["16c_stem_md5_changed"] = sum(1 for row in candidates if row.get("stem_md5_before") != row.get("stem_md5_after"))
    if passthrough is not None:
        validation["16d_passthrough"] = passthrough
    write_json(out_dir / "BATCH16_VALIDATION.json", validation)
    write_report(out_dir, census_summary, gold, refill_summary, passthrough, validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Batch16 QA-5 L0 census/regression/package.")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    validation = run(args.out_dir)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if validation.get("gold_passed") and validation.get("official_md5_unchanged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
