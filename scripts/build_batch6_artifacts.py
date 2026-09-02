#!/usr/bin/env python3
"""Build Codex Batch 6 audit artifacts under /tmp.

This script is intentionally read-only against official manifests. It gathers
the Batch 6 target rows, runs a bounded WS1 rerun for impacted source groups,
and writes reviewable queues/reports for Claude or the user to sign later.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT.parent
DEFAULT_OUT = Path("/tmp/yher_batch6_20260703")

V4_MANIFEST = REPO_ROOT / "data/item_bank/v4/chemistry_v4_3329.jsonl"
SERVICE_EXCLUSIONS = REPO_ROOT / "data/item_bank/v4/service_exclusions.jsonl"
V3_BANK = REPO_ROOT / "data/item_bank/chemistry_v3_6695.jsonl"
WS1_ROOT = REPO_ROOT / "data/ws1_batch_v4_20260703"
SOURCE_GROUPS = WS1_ROOT / "source_groups.json"
ALIGNMENT_QUEUE = REPO_ROOT / "data/ws3_apply_20260703/alignment_review_queue.jsonl"
SOURCE_ROOT = TOOLS_ROOT / "上海化学卷合集"

SCHEMA_VERSION = "ws3_schema_v4_candidate_2"

ROUND2_TARGETS = {"round2_042_equation", "round2_045_process"}
ANALYSIS_LEAK_RE = re.compile(r"【\s*(?:试题\s*解析|题目\s*解析)\s*】")
OLD_LEAK_RE = re.compile(r"【\s*解析\s*】")
OMML_LITERAL = "[OMML]"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_source_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def blocks_text(value: object, include_media_placeholder: bool = True) -> str:
    parts: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            block_type = node.get("type")
            if block_type == "text":
                parts.append(str(node.get("text", "")))
            elif block_type == "table":
                for row in node.get("rows") or []:
                    for cell in row:
                        walk(cell)
            elif block_type in {"formula", "figure", "math_omml"}:
                latex = str(node.get("latex") or "")
                if latex:
                    parts.append(latex)
                elif include_media_placeholder:
                    parts.append("图")
            else:
                for child in node.values():
                    walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, str):
            parts.append(node)

    walk(value)
    return "".join(parts)


def block_signature(item_or_question: dict) -> dict:
    if "answer_blocks_effective" in item_or_question:
        answer_key = "answer_blocks_effective"
    else:
        answer_key = "answer_blocks"
    return {
        "stem_blocks": item_or_question.get("stem_blocks") or [],
        "answer_blocks": item_or_question.get(answer_key) or [],
        "analysis_blocks": item_or_question.get("analysis_blocks") or [],
    }


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def item_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("group_key") or ""),
        str(row.get("section_num") or ""),
        str(row.get("q_num") or ""),
    )


def find_batch6_targets(items: list[dict], exclusions: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    by_id = {row["item_id"]: row for row in items}
    targets: list[dict] = []
    exclusion_reason_by_id = {row.get("item_id"): row.get("reason") for row in exclusions}
    for exclusion in exclusions:
        item = by_id.get(exclusion.get("item_id"))
        if not item:
            continue
        targets.append(
            {
                "target_kind": "service_exclusion",
                "problem_type": exclusion.get("reason"),
                "item_id": item["item_id"],
                "source_path": item.get("source_path"),
                "resolved_source_path": resolve_source_path(str(item.get("source_path") or "")),
                "group_key": item.get("group_key"),
                "section_num": item.get("section_num"),
                "q_num": item.get("q_num"),
                "stem_before": item.get("stem_text"),
                "pool": item.get("pool"),
            }
        )
    for item in items:
        gold = item.get("golden_candidate") or {}
        candidate_id = str(gold.get("candidate_id") or gold.get("round2_id") or "")
        if candidate_id in ROUND2_TARGETS:
            targets.append(
                {
                    "target_kind": "formal_gold",
                    "problem_type": "manual_split_required"
                    if candidate_id == "round2_042_equation"
                    else "stem_tail_cross_question_contamination",
                    "item_id": item["item_id"],
                    "source_path": item.get("source_path"),
                    "resolved_source_path": resolve_source_path(str(item.get("source_path") or "")),
                    "group_key": item.get("group_key"),
                    "section_num": item.get("section_num"),
                    "q_num": item.get("q_num"),
                    "stem_before": item.get("stem_text"),
                    "pool": item.get("pool"),
                    "gold_candidate_id": candidate_id,
                    "alignment_status": (item.get("alignment") or {}).get("status"),
                }
            )
    return targets, {row["item_id"]: row for row in targets}


def source_groups_by_key() -> dict[str, dict]:
    if not SOURCE_GROUPS.exists():
        return {}
    groups = json.loads(SOURCE_GROUPS.read_text(encoding="utf-8"))
    return {str(group.get("group_key") or ""): group for group in groups}


def target_source_paths(targets: list[dict], groups_by_key: dict[str, dict]) -> list[Path]:
    group_keys = {str(target.get("group_key") or "") for target in targets if target.get("group_key")}
    return source_paths_for_group_keys(group_keys, groups_by_key, targets)


def source_paths_for_group_keys(
    group_keys: set[str],
    groups_by_key: dict[str, dict],
    fallback_targets: list[dict] | None = None,
) -> list[Path]:
    paths: set[str] = set()
    for group_key in group_keys:
        group = groups_by_key.get(group_key)
        if group:
            for source in group.get("unique_sources", []):
                original = source.get("original_path") or source.get("path")
                if original:
                    paths.add(resolve_source_path(str(original)))
    for target in fallback_targets or []:
        if str(target.get("group_key") or "") not in group_keys and target.get("source_path"):
            paths.add(resolve_source_path(str(target["source_path"])))
    return [Path(path) for path in sorted(paths) if Path(path).exists()]


def build_segmentation_targets(out_dir: Path, targets: list[dict], groups_by_key: dict[str, dict]) -> None:
    rows = []
    for target in targets:
        group = groups_by_key.get(str(target.get("group_key") or "")) or {}
        rows.append(
            {
                **target,
                "raw_ws1_group": group,
            }
        )
    write_jsonl(out_dir / "ws1_segmentation/segmentation_targets.jsonl", rows)


def run_bounded_ws1_rerun(
    out_dir: Path,
    targets: list[dict],
    groups_by_key: dict[str, dict],
    skip_rerun: bool,
    extra_group_keys: set[str] | None = None,
) -> Path:
    rerun_root = out_dir / "ws1_segmentation/fixed_groups"
    group_keys = {str(target.get("group_key") or "") for target in targets if target.get("group_key")}
    group_keys.update(extra_group_keys or set())
    source_paths = source_paths_for_group_keys(group_keys, groups_by_key, targets)
    source_list_path = out_dir / "ws1_segmentation/target_source_list.txt"
    source_list_path.parent.mkdir(parents=True, exist_ok=True)
    source_list_path.write_text("\n".join(str(path) for path in source_paths) + "\n", encoding="utf-8")
    if skip_rerun:
        return rerun_root
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.ws1_docx_extract_prototype import run_batch_extract_v4

    run_batch_extract_v4(SOURCE_ROOT, rerun_root, preview_limit=5, source_paths=source_paths)
    return rerun_root


def build_candidate_item_from_question(
    question: dict,
    old_item: dict | None = None,
    candidate_origin: str = "batch6_ws1_bounded_rerun",
) -> dict:
    stem_text = blocks_text(question.get("stem_blocks") or [])
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": str(question.get("question_id") or ""),
        "source_question_id": str(question.get("question_id") or ""),
        "candidate_origin": candidate_origin,
        "group_key": question.get("group_key"),
        "source_path": question.get("source_path"),
        "answer_source_path": question.get("answer_source_path"),
        "q_num": question.get("q_num"),
        "section_num": question.get("section_num"),
        "stem_blocks": question.get("stem_blocks") or [],
        "answer_blocks_effective": question.get("answer_blocks") or [],
        "analysis_blocks": question.get("analysis_blocks") or [],
        "quality_flags": question.get("quality_flags") or [],
        "stem_text": stem_text,
        "old_item_id": old_item.get("item_id") if old_item else "",
        "old_source_question_id": old_item.get("source_question_id") if old_item else "",
    }


def build_candidate_item_from_official_item(item: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": str(item.get("item_id") or ""),
        "source_question_id": str(item.get("source_question_id") or item.get("item_id") or ""),
        "candidate_origin": "batch6_official_preserved_for_collateral_safety",
        "group_key": item.get("group_key"),
        "source_path": item.get("source_path"),
        "answer_source_path": item.get("answer_source_path"),
        "q_num": item.get("q_num"),
        "section_num": item.get("section_num"),
        "stem_blocks": item.get("stem_blocks") or [],
        "answer_blocks_effective": item.get("answer_blocks_effective") or [],
        "analysis_blocks": item.get("analysis_blocks") or [],
        "quality_flags": item.get("quality_flags") or [],
        "stem_text": item.get("stem_text") or blocks_text(item.get("stem_blocks") or []),
        "old_item_id": item.get("item_id"),
        "old_source_question_id": item.get("source_question_id"),
    }


def build_rerun_diff_outputs(out_dir: Path, rerun_root: Path, items: list[dict], targets_by_id: dict[str, dict]) -> dict:
    old_by_key = defaultdict(list)
    for item in items:
        old_by_key[item_key(item)].append(item)
    rerun_questions = load_jsonl(rerun_root / "questions_deduped.jsonl")
    new_by_key = defaultdict(list)
    for row in rerun_questions:
        new_by_key[item_key(row)].append(row)
    rerun_groups = {str(row.get("group_key") or "") for row in rerun_questions}
    rerun_groups.update(str(target.get("group_key") or "") for target in targets_by_id.values())

    mapping_rows = []
    raw_collateral_rows = []
    final_collateral_rows = []
    candidate_rows = []
    matched_new_ids: set[str] = set()
    manual_split_group_keys = {
        str(target.get("group_key") or "")
        for target in targets_by_id.values()
        if target.get("problem_type") == "manual_split_required"
        or target.get("gold_candidate_id") == "round2_042_equation"
    }
    old_rows_in_scope = [
        old
        for item_list in old_by_key.values()
        for old in item_list
        if item_key(old)[0] in rerun_groups
    ]
    old_rows_in_scope.sort(key=lambda row: (item_key(row), str(row.get("item_id") or "")))

    def best_new_for_old(old: dict, candidates: list[dict]) -> tuple[dict | None, int | None, float]:
        if not candidates:
            return None, None, 0.0
        old_text = str(old.get("stem_text") or "")
        scored = []
        for idx, candidate in enumerate(candidates):
            new_text = blocks_text(candidate.get("stem_blocks") or [])
            score = difflib.SequenceMatcher(None, old_text, new_text).ratio()
            scored.append((score, idx, candidate))
        scored.sort(key=lambda row: (-row[0], row[1]))
        score, idx, candidate = scored[0]
        return candidate, idx + 1, score

    def diff_kind(old: dict, new: dict) -> tuple[bool, bool, bool, str]:
        old_stem_text = str(old.get("stem_text") or "")
        new_stem_text = blocks_text(new.get("stem_blocks") or [])
        stem_text_changed = old_stem_text != new_stem_text
        old_answer_text = blocks_text(old.get("answer_blocks_effective") or [])
        new_answer_text = blocks_text(new.get("answer_blocks") or [])
        old_analysis_text = blocks_text(old.get("analysis_blocks") or [])
        new_analysis_text = blocks_text(new.get("analysis_blocks") or [])
        answer_or_analysis_text_changed = (old_answer_text != new_answer_text) or (old_analysis_text != new_analysis_text)
        block_json_changed = stable_json(block_signature(old)) != stable_json(block_signature(new))
        if stem_text_changed:
            reason = "stem_text_changed"
        elif answer_or_analysis_text_changed:
            reason = "answer_or_analysis_text_changed"
        elif block_json_changed:
            reason = "block_json_schema_or_asset_changed"
        else:
            reason = "no_content_change"
        return stem_text_changed, answer_or_analysis_text_changed, block_json_changed, reason

    for old in old_rows_in_scope:
        key = item_key(old)
        new, duplicate_rank, match_score = best_new_for_old(old, new_by_key.get(key, []))
        duplicate_count = len(old_by_key.get(key, []))
        is_target = old["item_id"] in targets_by_id
        if new:
            stem_changed, answer_analysis_changed, block_changed, detailed_reason = diff_kind(old, new)
            changed = block_changed
            reason = "target_fix" if is_target else "untouched"
            mapping_rows.append(
                {
                    "old_item_id": old.get("item_id"),
                    "new_item_id": new.get("question_id"),
                    "group_key": key[0],
                    "section_num": key[1],
                    "q_num": key[2],
                    "change_reason": reason,
                    "mapping_status": "matched",
                    "content_changed": changed,
                    "stem_text_changed": stem_changed,
                    "answer_or_analysis_text_changed": answer_analysis_changed,
                    "detailed_change_reason": detailed_reason,
                    "duplicate_key_count": duplicate_count,
                    "duplicate_key_match_rank": duplicate_rank,
                    "duplicate_key_stem_similarity": round(match_score, 6),
                }
            )
            matched_new_ids.add(str(new.get("question_id") or ""))
            if is_target:
                candidate_rows.append(build_candidate_item_from_question(new, old))
                final_collateral_rows.append(
                    {
                        "old_item_id": old.get("item_id"),
                        "new_item_id": new.get("question_id"),
                        "group_key": key[0],
                        "section_num": key[1],
                        "q_num": key[2],
                        "is_target": True,
                        "content_changed": changed,
                        "change_reason": detailed_reason if changed else "target_observed_no_content_change",
                        "stem_text_changed": stem_changed,
                        "answer_or_analysis_text_changed": answer_analysis_changed,
                        "candidate_origin": "batch6_ws1_bounded_rerun",
                        "duplicate_key_count": duplicate_count,
                        "duplicate_key_match_rank": duplicate_rank,
                        "duplicate_key_stem_similarity": round(match_score, 6),
                        "old_stem": old.get("stem_text", "")[:500],
                        "new_stem": blocks_text(new.get("stem_blocks") or [])[:500],
                    }
                )
            else:
                candidate_rows.append(build_candidate_item_from_official_item(old))
                if changed:
                    final_collateral_rows.append(
                        {
                            "old_item_id": old.get("item_id"),
                            "new_item_id": old.get("item_id"),
                            "raw_rerun_new_item_id": new.get("question_id"),
                            "group_key": key[0],
                            "section_num": key[1],
                            "q_num": key[2],
                            "is_target": False,
                            "content_changed": False,
                            "raw_rerun_content_changed": True,
                            "change_reason": "official_preserved_for_collateral_safety",
                            "raw_rerun_change_reason": detailed_reason,
                            "stem_text_changed": False,
                            "answer_or_analysis_text_changed": False,
                            "candidate_origin": "batch6_official_preserved_for_collateral_safety",
                            "duplicate_key_count": duplicate_count,
                            "duplicate_key_match_rank": duplicate_rank,
                            "duplicate_key_stem_similarity": round(match_score, 6),
                            "old_stem": old.get("stem_text", "")[:500],
                            "new_stem": old.get("stem_text", "")[:500],
                        }
                    )
            if changed or is_target:
                raw_collateral_rows.append(
                    {
                        "old_item_id": old.get("item_id"),
                        "new_item_id": new.get("question_id"),
                        "group_key": key[0],
                        "section_num": key[1],
                        "q_num": key[2],
                        "is_target": is_target,
                        "content_changed": changed,
                        "change_reason": detailed_reason if changed else "target_observed_no_content_change",
                        "stem_text_changed": stem_changed,
                        "answer_or_analysis_text_changed": answer_analysis_changed,
                        "duplicate_key_count": duplicate_count,
                        "duplicate_key_match_rank": duplicate_rank,
                        "duplicate_key_stem_similarity": round(match_score, 6),
                        "old_stem": old.get("stem_text", "")[:500],
                        "new_stem": blocks_text(new.get("stem_blocks") or [])[:500],
                    }
                )
        else:
            mapping_rows.append(
                {
                    "old_item_id": old.get("item_id"),
                    "new_item_id": "",
                    "group_key": key[0],
                    "section_num": key[1],
                    "q_num": key[2],
                    "change_reason": "target_fix" if is_target else "untouched",
                    "mapping_status": "removed_or_not_in_rerun",
                    "content_changed": is_target,
                    "stem_text_changed": is_target,
                    "answer_or_analysis_text_changed": False,
                    "detailed_change_reason": "target_removed" if is_target else "missing_from_bounded_rerun",
                    "duplicate_key_count": duplicate_count,
                    "duplicate_key_match_rank": None,
                    "duplicate_key_stem_similarity": 0.0,
                }
            )
            raw_collateral_rows.append(
                {
                    "old_item_id": old.get("item_id"),
                    "new_item_id": "",
                    "group_key": key[0],
                    "section_num": key[1],
                    "q_num": key[2],
                    "is_target": is_target,
                    "content_changed": is_target,
                    "change_reason": "target_removed" if is_target else "missing_from_bounded_rerun",
                    "old_stem": old.get("stem_text", "")[:500],
                    "new_stem": "",
                }
            )
            if is_target:
                target = targets_by_id.get(str(old.get("item_id") or ""), {})
                final_collateral_rows.append(
                    {
                        "old_item_id": old.get("item_id"),
                        "new_item_id": "",
                        "group_key": key[0],
                        "section_num": key[1],
                        "q_num": key[2],
                        "is_target": True,
                        "content_changed": True,
                        "change_reason": "target_removed"
                        if target.get("problem_type") != "omml_literal"
                        else "target_unmatched_in_rerun",
                        "candidate_origin": "",
                        "old_stem": old.get("stem_text", "")[:500],
                        "new_stem": "",
                    }
                )
            else:
                candidate_rows.append(build_candidate_item_from_official_item(old))
                final_collateral_rows.append(
                    {
                        "old_item_id": old.get("item_id"),
                        "new_item_id": old.get("item_id"),
                        "group_key": key[0],
                        "section_num": key[1],
                        "q_num": key[2],
                        "is_target": False,
                        "content_changed": False,
                        "raw_rerun_content_changed": False,
                        "change_reason": "official_preserved_for_collateral_safety",
                        "raw_rerun_change_reason": "missing_from_bounded_rerun",
                        "candidate_origin": "batch6_official_preserved_for_collateral_safety",
                        "old_stem": old.get("stem_text", "")[:500],
                        "new_stem": old.get("stem_text", "")[:500],
                    }
                )

    for key, new_rows in sorted(new_by_key.items()):
        unmatched_new_rows = [row for row in new_rows if str(row.get("question_id") or "") not in matched_new_ids]
        if key not in old_by_key:
            for new in unmatched_new_rows:
                include_as_split_candidate = key[0] in manual_split_group_keys
                mapping_rows.append(
                    {
                        "old_item_id": "",
                        "new_item_id": new.get("question_id"),
                        "group_key": key[0],
                        "section_num": key[1],
                        "q_num": key[2],
                        "change_reason": "split_new",
                        "mapping_status": "new_in_rerun",
                        "content_changed": True,
                        "candidate_included": include_as_split_candidate,
                        "candidate_exclusion_reason": "" if include_as_split_candidate else "new_key_not_in_manual_split_scope",
                    }
                )
                if include_as_split_candidate:
                    candidate_rows.append(
                        build_candidate_item_from_question(
                            new,
                            candidate_origin="batch6_ws1_split_candidate_pending_review",
                        )
                    )
                    final_collateral_rows.append(
                        {
                            "old_item_id": "",
                            "new_item_id": new.get("question_id"),
                            "group_key": key[0],
                            "section_num": key[1],
                            "q_num": key[2],
                            "is_target": True,
                            "content_changed": True,
                            "change_reason": "split_new_pending_user_or_claude_review",
                            "candidate_origin": "batch6_ws1_split_candidate_pending_review",
                            "old_stem": "",
                            "new_stem": blocks_text(new.get("stem_blocks") or [])[:500],
                        }
                    )
                raw_collateral_rows.append(
                    {
                        "old_item_id": "",
                        "new_item_id": new.get("question_id"),
                        "group_key": key[0],
                        "section_num": key[1],
                        "q_num": key[2],
                        "is_target": include_as_split_candidate,
                        "content_changed": True,
                        "candidate_included": include_as_split_candidate,
                        "change_reason": "split_new_or_new_question",
                        "old_stem": "",
                        "new_stem": blocks_text(new.get("stem_blocks") or [])[:500],
                    }
                )
        elif unmatched_new_rows:
            for new in unmatched_new_rows:
                include_as_split_candidate = key[0] in manual_split_group_keys
                mapping_rows.append(
                    {
                        "old_item_id": "",
                        "new_item_id": new.get("question_id"),
                        "group_key": key[0],
                        "section_num": key[1],
                        "q_num": key[2],
                        "change_reason": "split_new",
                        "mapping_status": "new_duplicate_key_in_rerun",
                        "content_changed": True,
                        "candidate_included": include_as_split_candidate,
                        "candidate_exclusion_reason": "" if include_as_split_candidate else "duplicate_key_extra_not_in_manual_split_scope",
                    }
                )
                if include_as_split_candidate:
                    candidate_rows.append(
                        build_candidate_item_from_question(
                            new,
                            candidate_origin="batch6_ws1_split_candidate_pending_review",
                        )
                    )
                    final_collateral_rows.append(
                        {
                            "old_item_id": "",
                            "new_item_id": new.get("question_id"),
                            "group_key": key[0],
                            "section_num": key[1],
                            "q_num": key[2],
                            "is_target": True,
                            "content_changed": True,
                            "change_reason": "split_new_duplicate_pending_user_or_claude_review",
                            "candidate_origin": "batch6_ws1_split_candidate_pending_review",
                            "old_stem": "",
                            "new_stem": blocks_text(new.get("stem_blocks") or [])[:500],
                        }
                    )
                raw_collateral_rows.append(
                    {
                        "old_item_id": "",
                        "new_item_id": new.get("question_id"),
                        "group_key": key[0],
                        "section_num": key[1],
                        "q_num": key[2],
                        "is_target": include_as_split_candidate,
                        "content_changed": True,
                        "candidate_included": include_as_split_candidate,
                        "change_reason": "split_new_or_new_duplicate_key",
                        "old_stem": "",
                        "new_stem": blocks_text(new.get("stem_blocks") or [])[:500],
                    }
                )

    unique_candidate_rows: dict[str, dict] = {}
    candidate_priority = {
        "batch6_ws1_bounded_rerun": 3,
        "batch6_ws1_split_candidate_pending_review": 2,
        "batch6_official_preserved_for_collateral_safety": 1,
    }
    for row in candidate_rows:
        item_id = str(row.get("item_id") or "")
        if not item_id:
            continue
        existing = unique_candidate_rows.get(item_id)
        existing_priority = candidate_priority.get(str((existing or {}).get("candidate_origin") or ""), 0)
        row_priority = candidate_priority.get(str(row.get("candidate_origin") or ""), 0)
        if existing is None or row_priority > existing_priority:
            unique_candidate_rows[item_id] = row

    write_jsonl(out_dir / "ws1_segmentation/rerun_id_mapping.jsonl", mapping_rows)
    write_jsonl(out_dir / "ws1_segmentation/raw_collateral_diff.jsonl", raw_collateral_rows)
    write_jsonl(out_dir / "ws1_segmentation/collateral_diff.jsonl", final_collateral_rows)
    write_jsonl(out_dir / "ws1_segmentation/fixed_candidate_items.jsonl", unique_candidate_rows.values())

    manual_split_rows = []
    for target in targets_by_id.values():
        if target.get("gold_candidate_id") == "round2_042_equation":
            manual_split_rows.append(
                {
                    "item_id": target["item_id"],
                    "group_key": target.get("group_key"),
                    "q_num": target.get("q_num"),
                    "review_status": "pending_user_or_claude",
                    "recommended_action": "inspect WS1 rerun and decide whether to split composite equation subparts",
                    "reviewer": "",
                    "stem_before": target.get("stem_before"),
                }
            )
    write_jsonl(out_dir / "ws1_segmentation/manual_split_review_queue.jsonl", manual_split_rows)
    write_jsonl(out_dir / "ws1_segmentation/split_plan.jsonl", [])
    return {
        "rerun_questions": len(rerun_questions),
        "candidate_items": len(unique_candidate_rows),
        "mapping_rows": len(mapping_rows),
        "raw_collateral_rows": len(raw_collateral_rows),
        "raw_non_target_content_changes": sum(
            1 for row in raw_collateral_rows if row.get("content_changed") and not row.get("is_target")
        ),
        "raw_collateral_reason_counts": dict(Counter(str(row.get("change_reason")) for row in raw_collateral_rows)),
        "raw_non_target_reason_counts": dict(
            Counter(str(row.get("change_reason")) for row in raw_collateral_rows if not row.get("is_target"))
        ),
        "final_collateral_rows": len(final_collateral_rows),
        "final_non_target_content_changes": sum(
            1 for row in final_collateral_rows if row.get("content_changed") and not row.get("is_target")
        ),
        "final_collateral_reason_counts": dict(Counter(str(row.get("change_reason")) for row in final_collateral_rows)),
        "final_non_target_reason_counts": dict(
            Counter(str(row.get("change_reason")) for row in final_collateral_rows if not row.get("is_target"))
        ),
        "target_rows_with_mapping": sum(1 for row in mapping_rows if row.get("old_item_id") in targets_by_id),
    }


def build_source_classification_outputs(out_dir: Path, groups_by_key: dict[str, dict], rerun_root: Path) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.ws1_docx_extract_prototype import classify_source_name, refine_source_role_with_probe

    rerun_records = {}
    rerun_records_path = rerun_root / "source_records.jsonl"
    if rerun_records_path.exists():
        for row in load_jsonl(rerun_records_path):
            original = str(row.get("original_path") or row.get("path") or "")
            if original:
                rerun_records[resolve_source_path(original)] = row

    audit_rows = []
    answer_only_rows = []
    for group in groups_by_key.values():
        for source in group.get("unique_sources", []):
            name = str(source.get("name") or source.get("file_name") or Path(str(source.get("original_path") or "")).name)
            new_role = classify_source_name(name)
            original_path = str(source.get("original_path") or source.get("path") or "")
            probe_record = rerun_records.get(resolve_source_path(original_path), {})
            role_record = refine_source_role_with_probe({**source, **probe_record, **new_role})
            old_role = str(source.get("role") or "")
            row = {
                "group_key": group.get("group_key"),
                "file_name": name,
                "original_path": original_path,
                "old_role": old_role,
                "new_role": role_record.get("role"),
                "role_changed": old_role != role_record.get("role"),
                "role_inferred_reason": role_record.get("role_inferred_reason", ""),
                "answer_only_evidence": role_record.get("answer_only_evidence", []),
                "answer_marker_count": role_record.get("answer_marker_count"),
                "answer_position": role_record.get("answer_position"),
                "text_char_count": role_record.get("text_char_count"),
                "q_start_count": role_record.get("q_start_count"),
                "question_prompt_count": role_record.get("question_prompt_count"),
                "answer_fragment_line_count": role_record.get("answer_fragment_line_count"),
                "option_marker_count": role_record.get("option_marker_count"),
                "route": role_record.get("route"),
                "paired_question_source": any(
                    s.get("role") == "question_source" for s in group.get("unique_sources", []) if s is not source
                ),
            }
            audit_rows.append(row)
            if role_record.get("role") == "answer_only":
                answer_only_rows.append(
                    {
                        **row,
                        "usable_for_answer_matching": True,
                        "question_output_allowed": False,
                    }
                )
    write_jsonl(out_dir / "source_classification/source_role_audit.jsonl", audit_rows)
    write_jsonl(out_dir / "source_classification/answer_only_docs.jsonl", answer_only_rows)
    return {"source_rows": len(audit_rows), "answer_only_docs": len(answer_only_rows)}


def load_v4_loader_functions():
    sys.path.insert(0, str(REPO_ROOT))
    from core.data.item_bank_v4 import has_effective_answer, load_service_exclusions, service_blockers

    return has_effective_answer, load_service_exclusions, service_blockers


def scope_for_item(item: dict, exclusions: dict, service_blockers) -> str:
    blockers = service_blockers(item, exclusions)
    if not blockers:
        return "service"
    if item.get("pool") == "main":
        return "main_blocked"
    return "other_pool"


def iter_block_occurrences(blocks: object, path: str = ""):
    if isinstance(blocks, dict):
        block_type = blocks.get("type")
        if block_type == "text" and OMML_LITERAL in str(blocks.get("text") or ""):
            yield {"block_type": "text", "block_path": path, "cell": "", "text": blocks.get("text")}
        elif block_type == "table":
            for r_idx, row in enumerate(blocks.get("rows") or []):
                for c_idx, cell in enumerate(row):
                    cell_path = f"{path}.rows[{r_idx}][{c_idx}]"
                    if isinstance(cell, str) and OMML_LITERAL in cell:
                        yield {"block_type": "table", "block_path": path, "cell": f"{r_idx},{c_idx}", "text": cell}
                    else:
                        yield from iter_block_occurrences(cell, cell_path)
        else:
            for key, value in blocks.items():
                yield from iter_block_occurrences(value, f"{path}.{key}" if path else key)
    elif isinstance(blocks, list):
        for idx, value in enumerate(blocks):
            yield from iter_block_occurrences(value, f"{path}[{idx}]")
    elif isinstance(blocks, str) and OMML_LITERAL in blocks:
        yield {"block_type": "text", "block_path": path, "cell": "", "text": blocks}


def omml_literal_rows_for_items(items: list[dict]) -> list[dict]:
    _, load_exclusions, service_blockers = load_v4_loader_functions()
    exclusions = load_exclusions(SERVICE_EXCLUSIONS)
    rows = []
    for item in items:
        scope = scope_for_item(item, exclusions, service_blockers)
        for zone in ("stem_blocks", "answer_blocks_effective", "analysis_blocks"):
            for occurrence in iter_block_occurrences(item.get(zone) or [], zone):
                rows.append(
                    {
                        "item_id": item.get("item_id"),
                        "group_key": item.get("group_key"),
                        "q_num": item.get("q_num"),
                        "section_num": item.get("section_num"),
                        "source_path": item.get("source_path"),
                        "zone": zone,
                        "block_type": occurrence["block_type"],
                        "block_path": occurrence["block_path"],
                        "cell": occurrence["cell"],
                        "scope": scope,
                        "stem_summary": str(item.get("stem_text") or "")[:240],
                        "literal_text": str(occurrence.get("text") or "")[:240],
                    }
                )
    return rows


def service_omml_group_keys(before_rows: list[dict]) -> set[str]:
    return {str(row.get("group_key") or "") for row in before_rows if row.get("scope") == "service" and row.get("group_key")}


def omml_targets_by_id(before_rows: list[dict]) -> dict[str, dict]:
    targets: dict[str, dict] = {}
    for row in before_rows:
        item_id = str(row.get("item_id") or "")
        if not item_id:
            continue
        existing = targets.setdefault(
            item_id,
            {
                "target_kind": "table_formula",
                "problem_type": "omml_literal",
                "item_id": item_id,
                "group_key": row.get("group_key"),
                "section_num": row.get("section_num"),
                "q_num": row.get("q_num"),
                "source_path": row.get("source_path"),
                "scope": row.get("scope"),
                "omml_literal_rows": 0,
                "block_types": set(),
            },
        )
        existing["omml_literal_rows"] += 1
        existing["block_types"].add(row.get("block_type"))
    for target in targets.values():
        target["block_types"] = sorted(str(value) for value in target["block_types"])
    return targets


def candidate_literal_count(candidate: dict) -> tuple[int, Counter[str]]:
    rows = []
    for zone in ("stem_blocks", "answer_blocks_effective", "analysis_blocks"):
        rows.extend(iter_block_occurrences(candidate.get(zone) or [], zone))
    return len(rows), Counter(str(row["block_type"]) for row in rows)


def build_omml_after_comparison_rows(
    before_rows: list[dict],
    candidate_rows: list[dict],
    rerun_group_keys: set[str],
) -> list[dict]:
    before_by_item: dict[str, list[dict]] = defaultdict(list)
    for row in before_rows:
        before_by_item[str(row.get("item_id") or "")].append(row)
    candidates_by_key: dict[tuple[str, str, str], dict] = {}
    for row in candidate_rows:
        if row.get("candidate_origin") in {
            "batch6_ws1_bounded_rerun",
            "batch6_ws1_split_candidate_pending_review",
        }:
            candidates_by_key[item_key(row)] = row

    after_rows = []
    for old_item_id, rows in sorted(before_by_item.items()):
        first = rows[0]
        key = (
            str(first.get("group_key") or ""),
            str(first.get("section_num") or ""),
            str(first.get("q_num") or ""),
        )
        candidate = candidates_by_key.get(key)
        scope = str(first.get("scope") or "")
        if candidate:
            literal_count, block_counts = candidate_literal_count(candidate)
            status = "rerun_matched"
            new_item_id = candidate.get("item_id")
        elif key[0] in rerun_group_keys:
            literal_count, block_counts = 0, Counter()
            status = "rerun_no_matching_candidate"
            new_item_id = ""
        elif scope != "service":
            literal_count, block_counts = len(rows), Counter(str(row.get("block_type")) for row in rows)
            status = "not_rerun_known_residual_debt"
            new_item_id = ""
        else:
            literal_count, block_counts = len(rows), Counter(str(row.get("block_type")) for row in rows)
            status = "service_not_rerun_gap"
            new_item_id = ""
        after_rows.append(
            {
                "old_item_id": old_item_id,
                "new_item_id": new_item_id,
                "group_key": key[0],
                "section_num": key[1],
                "q_num": key[2],
                "scope": scope,
                "before_literal_rows": len(rows),
                "before_block_types": dict(Counter(str(row.get("block_type")) for row in rows)),
                "after_literal_rows": literal_count,
                "after_block_types": dict(block_counts),
                "rerun_status": status,
                "service_candidate_pass": scope != "service" or (status == "rerun_matched" and literal_count == 0),
            }
        )
    return after_rows


def build_table_preview_samples(out_dir: Path, rerun_root: Path, after_rows: list[dict]) -> list[dict]:
    service_groups = [row["group_key"] for row in after_rows if row.get("scope") == "service" and row.get("rerun_status") == "rerun_matched"]
    wanted = set(service_groups)
    samples = []
    for preview in sorted(rerun_root.glob("*/preview.html")):
        summary_path = preview.parent / "summary.json"
        group_key = ""
        if summary_path.exists():
            try:
                group_key = str(json.loads(summary_path.read_text(encoding="utf-8")).get("group_key") or "")
            except Exception:
                group_key = ""
        if group_key in wanted or len(samples) < 5:
            samples.append(
                {
                    "group_key": group_key,
                    "preview_html": str(preview),
                    "status": "generated_preview_for_rerun_group",
                }
            )
        if len(samples) >= 5:
            break
    write_jsonl(out_dir / "table_formula/preview_samples.jsonl", samples)
    return samples


def build_table_formula_outputs(out_dir: Path, items: list[dict], rerun_root: Path | None = None) -> dict:
    rows = omml_literal_rows_for_items(items)
    write_jsonl(out_dir / "table_formula/omml_literal_before.jsonl", rows)
    candidate_rows = load_jsonl(out_dir / "ws1_segmentation/fixed_candidate_items.jsonl")
    rerun_group_keys = {
        str(row.get("group_key") or "")
        for row in candidate_rows
        if row.get("group_key")
        and row.get("candidate_origin")
        in {"batch6_ws1_bounded_rerun", "batch6_ws1_split_candidate_pending_review"}
    }
    after_rows = build_omml_after_comparison_rows(rows, candidate_rows, rerun_group_keys)
    write_jsonl(out_dir / "table_formula/omml_literal_after.jsonl", after_rows)
    preview_samples = build_table_preview_samples(out_dir, rerun_root or (out_dir / "ws1_segmentation/fixed_groups"), after_rows)
    schema_delta = """# Batch 6 Table Formula Schema Delta

- Existing official v4 may contain table cells as plain strings.
- Candidate v2 table cells may be either a string or a list of existing inline
  blocks: `text`, `formula`, `figure`, `table`, `math_omml`.
- No new inline block type is introduced.
- If LaTeX is available for OMML, it is attached as `latex` on the existing
  `math_omml` block: `{ "type": "math_omml", "omml": "...", "latex": "..." }`.
- `build_ws3_items_v4.blocks_text` and WS1 `blocks_text` recursively walk
  structured table cells so `stem_text` does not silently drop cell content.
"""
    (out_dir / "table_formula/schema_delta.md").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "table_formula/schema_delta.md").write_text(schema_delta, encoding="utf-8")
    return {
        "omml_literal_rows": len(rows),
        "unique_table_items": len({r["item_id"] for r in rows if r["block_type"] == "table"}),
        "unique_text_items": len({r["item_id"] for r in rows if r["block_type"] == "text"}),
        "scope_counts": dict(Counter(r["scope"] for r in rows)),
        "after_rows": len(after_rows),
        "after_status_counts": dict(Counter(row["rerun_status"] for row in after_rows)),
        "service_after_literal_rows": sum(int(row["after_literal_rows"]) for row in after_rows if row["scope"] == "service"),
        "service_after_failures": sum(1 for row in after_rows if row["scope"] == "service" and not row["service_candidate_pass"]),
        "preview_samples": len(preview_samples),
    }


def standard_solution_answers(item: dict) -> list[str]:
    sol = item.get("standard_solution") or {}
    answers = [str(a).strip() for a in sol.get("final_answers") or [] if str(a).strip()]
    standard = str(sol.get("standard_answer") or "").strip()
    if standard and standard not in answers:
        answers.append(standard)
    return answers


def build_no_answer_outputs(out_dir: Path, items: list[dict]) -> dict:
    has_effective_answer, load_exclusions, service_blockers = load_v4_loader_functions()
    exclusions = load_exclusions(SERVICE_EXCLUSIONS)
    v3_by_id = {row["item_id"]: row for row in load_jsonl(V3_BANK)}
    no_answer = [
        item
        for item in items
        if item.get("pool") == "main" and not has_effective_answer(item)
    ]
    no_answer_rows = []
    recoverable_rows = []
    review_rows = []
    excluded_rows = []
    for item in no_answer:
        alignment = item.get("alignment") or {}
        status = alignment.get("status")
        blockers = service_blockers(item, exclusions)
        base = {
            "item_id": item.get("item_id"),
            "group_key": item.get("group_key"),
            "q_num": item.get("q_num"),
            "section_num": item.get("section_num"),
            "source_path": item.get("source_path"),
            "alignment_status": status,
            "aligned_item_id": alignment.get("aligned_item_id"),
            "best_candidate_item_id": alignment.get("best_candidate_item_id"),
            "stem_summary": str(item.get("stem_text") or "")[:300],
            "service_blockers": blockers,
        }
        no_answer_rows.append(base)
        if status == "auto_inherited":
            old = v3_by_id.get(str(alignment.get("aligned_item_id") or ""))
            answers = standard_solution_answers(old or {}) if old else []
            if answers:
                recoverable_rows.append(
                    {
                        **base,
                        "schema_version": SCHEMA_VERSION,
                        "provenance": "v3_via_official_alignment",
                        "answer_text": "\n".join(answers),
                        "source_evidence": {
                            "v3_item_id": old.get("item_id") if old else "",
                            "v3_source": old.get("source") if old else "",
                            "v3_standard_solution": old.get("standard_solution") if old else {},
                        },
                        "confidence_reason": "alignment.status is auto_inherited and official v3 aligned item has standard_solution answers",
                    }
                )
            else:
                excluded_rows.append(
                    {
                        **base,
                        "review_status": "pending_user_or_claude",
                        "reviewer": "",
                        "recommended_action": "candidate_exclude_answerless_after_claude_or_user_review",
                        "reason": "auto_inherited aligned v3 item has no usable answer",
                    }
                )
        elif status in {"needs_review", "new"}:
            review_rows.append(
                {
                    **base,
                    "review_status": "pending_user_or_claude",
                    "reviewer": "",
                    "recommended_action": "manual_alignment_or_exclusion_review_required_before_any_answer_inheritance",
                    "reason": "alignment is not auto_inherited; v3 answer inheritance would be a manual alignment decision",
                }
            )
        else:
            excluded_rows.append(
                {
                    **base,
                    "review_status": "pending_user_or_claude",
                    "reviewer": "",
                    "recommended_action": "candidate_exclude_answerless_after_claude_or_user_review",
                    "reason": "no allowed recovery provenance found",
                }
            )
    write_jsonl(out_dir / "no_answer/no_effective_answer_138.jsonl", no_answer_rows)
    write_jsonl(out_dir / "no_answer/recoverable_answer_patch_candidates.jsonl", recoverable_rows)
    write_jsonl(out_dir / "no_answer/answer_inherit_review_queue.jsonl", review_rows)
    write_jsonl(out_dir / "no_answer/excluded_answerless_candidate_queue.jsonl", excluded_rows)
    return {
        "no_effective_answer": len(no_answer_rows),
        "recoverable": len(recoverable_rows),
        "inherit_review": len(review_rows),
        "excluded_candidates": len(excluded_rows),
        "alignment_counts": dict(Counter(row["alignment_status"] for row in no_answer_rows)),
    }


def build_leak_pattern_outputs(out_dir: Path, items: list[dict], exclusions: list[dict]) -> dict:
    exclusion_ids = {row.get("item_id") for row in exclusions}
    new_hits = []
    before_hits = []
    for item in items:
        stem = str(item.get("stem_text") or "")
        if OLD_LEAK_RE.search(stem):
            before_hits.append(item.get("item_id"))
        if ANALYSIS_LEAK_RE.search(stem):
            if item.get("item_id") in exclusion_ids:
                category = "known_service_exclusion"
            elif item.get("pool") == "main" and item.get("service_eligible") is True:
                category = "candidate_new_leak"
            else:
                category = "false_positive_review"
            new_hits.append(
                {
                    "item_id": item.get("item_id"),
                    "group_key": item.get("group_key"),
                    "q_num": item.get("q_num"),
                    "source_path": item.get("source_path"),
                    "category": category,
                    "stem_summary": stem[:500],
                }
            )
    write_jsonl(out_dir / "leak_pattern/new_hits.jsonl", new_hits)
    lines = [
        "# Batch 6 Leak Pattern Before/After",
        "",
        f"- Old `【解析】` stem hits: {len(before_hits)}",
        f"- New `【试题解析】`/`【题目解析】` stem hits: {len(new_hits)}",
        f"- New hit categories: {dict(Counter(row['category'] for row in new_hits))}",
        "- Scan is applied only to student-facing stem text, not to analysis_blocks.",
    ]
    path = out_dir / "leak_pattern/leak_scan_before_after.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"old_hits": len(before_hits), "new_hits": len(new_hits), "new_hit_categories": dict(Counter(row["category"] for row in new_hits))}


def build_alignment_queue_outputs(out_dir: Path) -> dict:
    queue = load_jsonl(ALIGNMENT_QUEUE)
    v3_by_id = {row["item_id"]: row for row in load_jsonl(V3_BANK)}
    kg_counts: Counter[str] = Counter()
    enriched = []
    for row in queue:
        old = v3_by_id.get(str(row.get("best_candidate_item_id") or ""))
        kg_nodes = list((old or {}).get("kg_nodes") or [])
        if not kg_nodes:
            kg_nodes = ["__missing_kg__"]
        kg_counts.update(kg_nodes)
        enriched.append((row, old, kg_nodes))

    ranked = []
    for row, old, kg_nodes in enriched:
        top_freq = max(kg_counts[node] for node in kg_nodes)
        reason_tags = []
        text = f"{row.get('new_stem','')} {row.get('candidate_stem','')}"
        if "图" in text:
            reason_tags.append("media_or_formula_text_loss_possible")
        ranked.append(
            {
                **row,
                "kg_nodes_from_best_candidate": kg_nodes,
                "kg_frequency_max": top_freq,
                "kg_frequency_sum": sum(kg_counts[node] for node in kg_nodes),
                "v3_source": (old or {}).get("source", ""),
                "review_status": "pending_user_or_claude",
                "reviewer": "",
                "recommended_action": "review_alignment_candidate",
                "reason_tags": reason_tags,
            }
        )
    ranked.sort(key=lambda r: (-int(r["kg_frequency_max"]), -float(r.get("similarity") or 0), str(r.get("group_key") or ""), int(r.get("q_num") or 0)))
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
        row["review_batch_id"] = f"batch_{(idx - 1) // 50 + 1:02d}"
    write_jsonl(out_dir / "alignment_queue/alignment_review_ranked_by_kg.jsonl", ranked)

    batches = defaultdict(list)
    for row in ranked:
        batches[row["review_batch_id"]].append(row)
    lines = [
        "# Batch 6 Alignment Review Batches",
        "",
        "KG join path: `alignment_review_queue.best_candidate_item_id` -> `data/item_bank/chemistry_v3_6695.jsonl.item_id` -> `kg_nodes`.",
        "",
        "Manual discipline: every row remains `review_status=pending_user_or_claude`; Codex does not produce final manual alignment or exclusion decisions.",
        "",
        "Review field template: `decision`, `reviewer`, `evidence`, `notes`. Reviewer must be user or Claude, never `codex_*`.",
        "",
    ]
    for batch_id in sorted(batches):
        rows = batches[batch_id]
        node_counter = Counter(node for row in rows for node in row["kg_nodes_from_best_candidate"])
        sim_counter = Counter(f"{int(float(row.get('similarity') or 0) * 10) / 10:.1f}x" for row in rows)
        lines.extend(
            [
                f"## {batch_id}",
                "",
                f"- Rows: {len(rows)}",
                f"- Top KG nodes: {', '.join(f'{node}({count})' for node, count in node_counter.most_common(8))}",
                f"- Similarity buckets: {dict(sim_counter)}",
                "- Expected decisions: manual_inherit / manual_do_not_inherit / manual_split_required / keep_pending.",
                "",
            ]
        )
    path = out_dir / "alignment_queue/alignment_review_batches.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "ranked_rows": len(ranked),
        "batches": len(batches),
        "kg_join_missing": sum(1 for row in ranked if row["kg_nodes_from_best_candidate"] == ["__missing_kg__"]),
        "top_kg_nodes": dict(kg_counts.most_common(20)),
    }


def write_report(out_dir: Path, summary: dict, official_md5_before: dict[str, str], official_md5_after: dict[str, str]) -> None:
    segmentation_targets = load_jsonl(out_dir / "ws1_segmentation/segmentation_targets.jsonl")
    mapping_rows = load_jsonl(out_dir / "ws1_segmentation/rerun_id_mapping.jsonl")
    candidate_rows = load_jsonl(out_dir / "ws1_segmentation/fixed_candidate_items.jsonl")
    answer_only_docs = load_jsonl(out_dir / "source_classification/answer_only_docs.jsonl")
    omml_before = load_jsonl(out_dir / "table_formula/omml_literal_before.jsonl")
    omml_after = load_jsonl(out_dir / "table_formula/omml_literal_after.jsonl")
    no_answer_rows = load_jsonl(out_dir / "no_answer/no_effective_answer_138.jsonl")
    recoverable_rows = load_jsonl(out_dir / "no_answer/recoverable_answer_patch_candidates.jsonl")
    inherit_review_rows = load_jsonl(out_dir / "no_answer/answer_inherit_review_queue.jsonl")
    excluded_answerless_rows = load_jsonl(out_dir / "no_answer/excluded_answerless_candidate_queue.jsonl")
    leak_hits = load_jsonl(out_dir / "leak_pattern/new_hits.jsonl")
    alignment_rows = load_jsonl(out_dir / "alignment_queue/alignment_review_ranked_by_kg.jsonl")

    target_by_id = {str(row.get("item_id") or ""): row for row in segmentation_targets}
    mapping_by_old = {str(row.get("old_item_id") or ""): row for row in mapping_rows if row.get("old_item_id")}
    candidate_by_old = {str(row.get("old_item_id") or ""): row for row in candidate_rows if row.get("old_item_id")}
    target_after_rows = []
    for old_id, target in sorted(target_by_id.items(), key=lambda row: (str(row[1].get("problem_type")), str(row[1].get("group_key")), str(row[1].get("q_num")))):
        mapping = mapping_by_old.get(old_id, {})
        candidate = candidate_by_old.get(old_id, {})
        stem_after = str(candidate.get("stem_text") or "")
        target_after_rows.append(
            {
                "old_item_id": old_id,
                "problem_type": target.get("problem_type"),
                "mapping_status": mapping.get("mapping_status", "missing"),
                "new_item_id": mapping.get("new_item_id", ""),
                "change_reason": mapping.get("detailed_change_reason", ""),
                "after_has_analysis_marker": bool(ANALYSIS_LEAK_RE.search(stem_after)),
            }
        )
    analysis_target_ids = {
        row["old_item_id"]
        for row in target_after_rows
        if row.get("problem_type") == "stem_is_analysis_text"
    }
    answer_fragment_target_ids = {
        row["old_item_id"]
        for row in target_after_rows
        if row.get("problem_type") in {"stem_is_answer_fragment", "stem_from_answers_only_doc"}
    }
    analysis_after_bad = sum(
        1
        for row in target_after_rows
        if row["old_item_id"] in analysis_target_ids and row.get("after_has_analysis_marker")
    )
    answer_fragment_removed_or_remapped = sum(
        1
        for row in target_after_rows
        if row["old_item_id"] in answer_fragment_target_ids
        and (row.get("mapping_status") == "removed_or_not_in_rerun" or row.get("change_reason") == "stem_text_changed")
    )
    omml_scope_item_counts = {
        scope: {
            "rows": sum(1 for row in omml_before if row.get("scope") == scope),
            "table_items": len({row.get("item_id") for row in omml_before if row.get("scope") == scope and row.get("block_type") == "table"}),
            "text_items": len({row.get("item_id") for row in omml_before if row.get("scope") == scope and row.get("block_type") == "text"}),
        }
        for scope in ("service", "main_blocked", "other_pool")
    }
    leak_false_positive_count = sum(1 for row in leak_hits if row.get("category") == "false_positive_review")
    alignment_similarity_buckets = Counter(f"{int(float(row.get('similarity') or 0) * 10) / 10:.1f}x" for row in alignment_rows)

    lines = [
        "# Codex Batch 6 Audit Report",
        "",
        "Discipline:",
        "",
        "> \"manual\" 对齐/排除决定只能由用户或 Claude 签字。Codex 遇到该人工判定的项,必须放进队列等审, 不得自批(reviewer 字段不得出现 codex_*)。产出先进 /tmp 审计,不碰 official,不清工作树,不泄 key。",
        "",
        "## Summary",
        "",
        f"- WS1 targets: {summary['targets']['total']} rows ({summary['targets']['by_problem_type']})",
        f"- Bounded rerun candidate items: {summary['rerun_diff']['candidate_items']}",
        f"- Raw non-target content changes observed in full bounded rerun: {summary['rerun_diff']['raw_non_target_content_changes']}",
        f"- Raw non-target change reasons: {summary['rerun_diff'].get('raw_non_target_reason_counts', {})}",
        f"- Final candidate non-target content changes after preserving official rows: {summary['rerun_diff']['final_non_target_content_changes']}",
        f"- Final candidate non-target reasons: {summary['rerun_diff'].get('final_non_target_reason_counts', {})}",
        f"- Answer-only docs detected: {summary['source_classification']['answer_only_docs']}",
        f"- `[OMML]` before rows: {summary['table_formula']['omml_literal_rows']}; table items={summary['table_formula']['unique_table_items']}; text items={summary['table_formula']['unique_text_items']}; scopes={summary['table_formula']['scope_counts']}",
        f"- `[OMML]` after rows: compared items={summary['table_formula'].get('after_rows')}; statuses={summary['table_formula'].get('after_status_counts')}; service_after_literal_rows={summary['table_formula'].get('service_after_literal_rows')}; service_after_failures={summary['table_formula'].get('service_after_failures')}; preview_samples={summary['table_formula'].get('preview_samples')}",
        f"- No-effective-answer classification: {summary['no_answer']}",
        f"- Leak scan: {summary['leak_pattern']}",
        f"- Alignment queue ranked: {summary['alignment_queue']['ranked_rows']} rows in {summary['alignment_queue']['batches']} batches; KG join missing={summary['alignment_queue']['kg_join_missing']}",
        "",
        "## WS1 Segmentation And Source Classification",
        "",
        f"- Before targets: 7 `stem_is_analysis_text`, 2 `stem_is_answer_fragment`, 2 `stem_from_answers_only_doc`, 1 `round2_045_process`, 1 `round2_042_equation`.",
        f"- After analysis-marker target stems still containing `【试题解析】`/`【题目解析】`: {analysis_after_bad}/7.",
        f"- After answer-fragment / answer-only targets removed or remapped to real stems: {answer_fragment_removed_or_remapped}/4.",
        f"- Target mapping outcomes: {dict(Counter(str(row.get('mapping_status')) for row in target_after_rows))}.",
        f"- Target detailed reasons: {dict(Counter(str(row.get('change_reason')) for row in target_after_rows))}.",
        f"- `round2_042_equation`: `manual_split_review_queue.jsonl` rows={len(load_jsonl(out_dir / 'ws1_segmentation/manual_split_review_queue.jsonl'))}; `split_plan.jsonl` rows={len(load_jsonl(out_dir / 'ws1_segmentation/split_plan.jsonl'))}; no Codex manual decision.",
        f"- `answer_only_docs.jsonl`: {len(answer_only_docs)} docs; names={[row.get('file_name') for row in answer_only_docs]}.",
        f"- Answer-only question output allowed rows: {sum(1 for row in answer_only_docs if row.get('question_output_allowed'))}.",
        "",
        "## Table Formula OMML",
        "",
        f"- Before all-library literal hits: rows={len(omml_before)}, table_items={summary['table_formula']['unique_table_items']}, text_items={summary['table_formula']['unique_text_items']}.",
        f"- Before by scope: {omml_scope_item_counts}.",
        f"- After comparison rows={len(omml_after)}, status_counts={dict(Counter(str(row.get('rerun_status')) for row in omml_after))}.",
        f"- Service after literal rows={summary['table_formula']['service_after_literal_rows']}; service failures={summary['table_formula']['service_after_failures']}.",
        f"- Known non-service residual debt rows={sum(int(row.get('after_literal_rows') or 0) for row in omml_after if row.get('rerun_status') == 'not_rerun_known_residual_debt')}.",
        f"- Preview sample rows={summary['table_formula']['preview_samples']}; schema delta=`table_formula/schema_delta.md`.",
        "",
        "## No-Answer Triage",
        "",
        f"- No-effective-answer rows from v4 loader R3={len(no_answer_rows)}; alignment counts={dict(Counter(str(row.get('alignment_status')) for row in no_answer_rows))}.",
        f"- Recoverable patch candidates={len(recoverable_rows)}; provenance counts={dict(Counter(str(row.get('provenance')) for row in recoverable_rows))}.",
        f"- Pending answer inheritance review queue={len(inherit_review_rows)}; excluded answerless candidate queue={len(excluded_answerless_rows)}.",
        f"- Queue total check: {len(recoverable_rows) + len(inherit_review_rows) + len(excluded_answerless_rows)}/138.",
        "",
        "## Leak Pattern Scan",
        "",
        f"- New pattern hits for `【试题解析】`/`【题目解析】`: {len(leak_hits)}; categories={dict(Counter(str(row.get('category')) for row in leak_hits))}.",
        f"- Known service exclusions caught={sum(1 for row in leak_hits if row.get('category') == 'known_service_exclusion')}/7.",
        f"- False-positive review rows={leak_false_positive_count}; scan applies to student-facing stem text only, not analysis_blocks.",
        "",
        "## Alignment Queue",
        "",
        "- KG join path: `alignment_review_queue.best_candidate_item_id` -> `data/item_bank/chemistry_v3_6695.jsonl.item_id` -> `kg_nodes`.",
        f"- Ranked queue rows={len(alignment_rows)}; batches={summary['alignment_queue']['batches']}; KG join missing={summary['alignment_queue']['kg_join_missing']}.",
        f"- Similarity buckets={dict(alignment_similarity_buckets)}.",
        f"- Review status counts={dict(Counter(str(row.get('review_status')) for row in alignment_rows))}; nonempty reviewers={sum(1 for row in alignment_rows if row.get('reviewer'))}.",
        "",
        "## Official Untouched Proof",
        "",
    ]
    for path, before in official_md5_before.items():
        lines.append(f"- `{path}` md5 before={before} after={official_md5_after.get(path)}")
    lines.extend(
        [
            "",
            "## Required Artifacts",
            "",
            "- `ws1_segmentation/segmentation_targets.jsonl`",
            "- `ws1_segmentation/rerun_id_mapping.jsonl`",
            "- `ws1_segmentation/raw_collateral_diff.jsonl`",
            "- `ws1_segmentation/collateral_diff.jsonl`",
            "- `source_classification/source_role_audit.jsonl`",
            "- `source_classification/answer_only_docs.jsonl`",
            "- `table_formula/omml_literal_before.jsonl`",
            "- `table_formula/omml_literal_after.jsonl`",
            "- `table_formula/schema_delta.md`",
            "- `table_formula/preview_samples.jsonl`",
            "- `no_answer/no_effective_answer_138.jsonl`",
            "- `no_answer/recoverable_answer_patch_candidates.jsonl`",
            "- `no_answer/answer_inherit_review_queue.jsonl`",
            "- `no_answer/excluded_answerless_candidate_queue.jsonl`",
            "- `leak_pattern/leak_scan_before_after.md`",
            "- `leak_pattern/new_hits.jsonl`",
            "- `alignment_queue/alignment_review_ranked_by_kg.jsonl`",
            "- `alignment_queue/alignment_review_batches.md`",
            "",
            "## Known Limits",
            "",
        "- This package does not write official v4 or service exclusion files.",
        "- Manual split/alignment/exclusion decisions remain pending for user or Claude.",
        "- Same-source answer extraction recovery is not auto-signed in this package.",
        "- The 25 auto_inherited no-answer rows currently have no usable v3 `standard_solution` answer in this local v3 bank, so they are emitted as pending exclusion candidates rather than recoverable patches.",
        "- The full bounded WS1 rerun still has non-target collateral differences; these are isolated in `raw_collateral_diff.jsonl`.",
        "- `fixed_candidate_items.jsonl` preserves official blocks for non-target rows, so `collateral_diff.jsonl` should have zero non-target content changes.",
        ]
    )
    (out_dir / "BATCH6_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-rerun", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    official_paths = {
        str(V4_MANIFEST): file_md5(V4_MANIFEST),
        str(SERVICE_EXCLUSIONS): file_md5(SERVICE_EXCLUSIONS),
    }

    items = load_jsonl(V4_MANIFEST)
    exclusions = load_jsonl(SERVICE_EXCLUSIONS)
    groups_by_key = source_groups_by_key()

    targets, targets_by_id = find_batch6_targets(items, exclusions)
    omml_before_rows = omml_literal_rows_for_items(items)
    omml_service_groups = service_omml_group_keys(omml_before_rows)
    diff_targets_by_id = {**targets_by_id, **omml_targets_by_id(omml_before_rows)}
    build_segmentation_targets(out_dir, targets, groups_by_key)
    rerun_root = run_bounded_ws1_rerun(out_dir, targets, groups_by_key, args.skip_rerun, omml_service_groups)
    rerun_diff = build_rerun_diff_outputs(out_dir, rerun_root, items, diff_targets_by_id)
    source_classification = build_source_classification_outputs(out_dir, groups_by_key, rerun_root)
    table_formula = build_table_formula_outputs(out_dir, items, rerun_root)
    no_answer = build_no_answer_outputs(out_dir, items)
    leak_pattern = build_leak_pattern_outputs(out_dir, items, exclusions)
    alignment_queue = build_alignment_queue_outputs(out_dir)

    summary = {
        "targets": {
            "total": len(targets),
            "by_problem_type": dict(Counter(str(row.get("problem_type")) for row in targets)),
        },
        "rerun_diff": rerun_diff,
        "source_classification": source_classification,
        "table_formula": table_formula,
        "no_answer": no_answer,
        "leak_pattern": leak_pattern,
        "alignment_queue": alignment_queue,
    }
    write_json(out_dir / "batch6_summary.json", summary)
    official_after = {
        str(V4_MANIFEST): file_md5(V4_MANIFEST),
        str(SERVICE_EXCLUSIONS): file_md5(SERVICE_EXCLUSIONS),
    }
    write_report(out_dir, summary, official_paths, official_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
