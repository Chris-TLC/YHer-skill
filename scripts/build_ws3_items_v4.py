#!/usr/bin/env python3
"""Build WS3 Schema v4 candidate items and alignment artifacts.

Writes only reviewable artifacts under /tmp by default. This does not mutate
official item manifests.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


AUTO_THRESHOLD = 0.92
REVIEW_THRESHOLD = 0.75
RESOLVED_ALIGNMENT_STATUSES = {
    "auto_inherited",
    "manual_inherited",
    "manual_do_not_inherit",
    "manual_split_required",
}

SUB_SUP_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₊": "+",
        "₋": "-",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def load_manual_overrides(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    overrides: dict[str, dict] = {}
    for row in load_jsonl(path):
        question_id = str(row.get("question_id") or "")
        aligned_item_id = str(row.get("aligned_item_id") or "")
        decision = str(row.get("decision") or "")
        if not question_id or not aligned_item_id:
            raise ValueError(f"manual override missing question_id/aligned_item_id: {row}")
        if decision != "manual_inherit":
            raise ValueError(f"unsupported manual override decision for {question_id}: {decision}")
        overrides[question_id] = {
            "aligned_item_id": aligned_item_id,
            "decision": decision,
            "reviewer": row.get("reviewer"),
            "evidence": row.get("evidence"),
        }
    return overrides


def load_manual_resolutions(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    resolutions: dict[str, dict] = {}
    decision_aliases = {
        "do_not_inherit": "manual_do_not_inherit",
        "manual_do_not_inherit": "manual_do_not_inherit",
        "manual_split_required": "manual_split_required",
    }
    for row in load_jsonl(path):
        question_id = str(row.get("question_id") or "")
        decision = decision_aliases.get(str(row.get("decision") or ""))
        if not question_id or not decision:
            raise ValueError(f"manual resolution missing question_id/decision: {row}")
        resolutions[question_id] = {
            "decision": decision,
            "reason": row.get("reason"),
            "reviewer": row.get("reviewer"),
            "evidence": row.get("evidence"),
        }
    return resolutions


def blocks_text(value: object, include_media_placeholder: bool = True) -> str:
    parts: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            block_type = node.get("type")
            if block_type == "text":
                parts.append(str(node.get("text", "")))
                return
            if block_type == "table":
                rows = node.get("rows") or []
                for row in rows:
                    for cell in row:
                        walk(cell)
                return
            if block_type in {"formula", "figure", "math_omml"}:
                latex = str(node.get("latex") or "")
                if latex:
                    parts.append(latex)
                elif include_media_placeholder:
                    parts.append("图")
                return
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, str):
            parts.append(node)

    walk(value)
    return "".join(parts)


def normalize_stem(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", (text or "").translate(SUB_SUP_TRANSLATION))
    normalized = re.sub(r"^\s*\d{1,3}[.、．]\s*", "", normalized)
    normalized = re.sub(r"^\s*[（(]?\d{2,4}[^）)]*[）)]", "", normalized)
    normalized = normalized.replace("图示", "")
    normalized = normalized.replace("____", "")
    normalized = re.sub(r"[\s_＿]+", "", normalized)
    normalized = re.sub(r"[（）()【】\[\]{}.,，。:：;；!！?？“”\"'、·\\/\\-—=＝]+", "", normalized)
    return normalized.lower()


def normalize_source_key(text: str) -> str:
    stem = Path(text or "").stem.strip()
    stem = re.sub(r"^\s*精品解析[:：]\s*", "", stem)
    stem = re.sub(r"\(\d+\)\s*$", "", stem).strip()
    stem = re.sub(r"（\d+）\s*$", "", stem).strip()
    for marker in (
        "原卷版",
        "空白卷",
        "考试版",
        "原卷",
        "解析版",
        "解析卷",
        "全解全析",
        "含解析",
        "参考答案",
        "答题卡",
        "（含解析）",
        "(含解析)",
        "精品解析：",
        "精品解析",
    ):
        stem = stem.replace(marker, "")
    stem = re.sub(r"[()（）]", "", stem)
    stem = re.sub(r"\s+", "", stem)
    stem = stem.replace("．", ".").replace("：", "").replace(":", "")
    return stem.strip("._- ")


def stem_grams(stem: str, size: int = 2) -> set[str]:
    if not stem:
        return set()
    if len(stem) < size:
        return {stem}
    return {stem[i : i + size] for i in range(len(stem) - size + 1)}


@dataclass
class OldRecord:
    item_id: str
    normalized_stem: str
    gram_set: set[str]
    item: dict


@dataclass
class AlignmentIndex:
    records: list[OldRecord]
    inverted: dict[str, list[int]]


def build_alignment_index(old_items: list[dict]) -> AlignmentIndex:
    records: list[OldRecord] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    for item in old_items:
        stem = normalize_stem(str(item.get("stem") or ""))
        if not stem:
            continue
        grams = stem_grams(stem)
        record = OldRecord(str(item.get("item_id") or ""), stem, grams, item)
        records.append(record)
        record_idx = len(records) - 1
        for gram in sorted(grams):
            inverted[gram].append(record_idx)
    return AlignmentIndex(records=records, inverted=dict(inverted))


def question_normalized_stem(question: dict) -> str:
    return normalize_stem(blocks_text(question.get("stem_blocks", []), include_media_placeholder=False))


def similarity_score(left: str, right: str, overlap: int, left_grams: set[str], right_grams: set[str]) -> tuple[float, float, float]:
    if not left or not right:
        return 0.0, 0.0, 0.0
    sequence_ratio = difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()
    containment = overlap / max(1, min(len(left_grams), len(right_grams)))
    dice = 2 * overlap / max(1, len(left_grams) + len(right_grams))
    return max(sequence_ratio, containment), sequence_ratio, dice


def align_question(question: dict, index: AlignmentIndex) -> dict:
    normalized = question_normalized_stem(question)
    grams = stem_grams(normalized)
    counts: Counter[int] = Counter()
    for gram in sorted(grams):
        for record_idx in index.inverted.get(gram, []):
            counts[record_idx] += 1

    best: tuple[int | None, float, float, float] = (None, 0.0, 0.0, 0.0)
    ordered_candidates = sorted(
        counts.items(),
        key=lambda item: (-item[1], index.records[item[0]].item_id, index.records[item[0]].normalized_stem),
    )[:120]
    for record_idx, overlap in ordered_candidates:
        record = index.records[record_idx]
        dice_hint = 2 * overlap / max(1, len(grams) + len(record.gram_set))
        if dice_hint < 0.18:
            continue
        score, sequence_ratio, dice = similarity_score(normalized, record.normalized_stem, overlap, grams, record.gram_set)
        if score > best[1]:
            best = (record_idx, score, sequence_ratio, dice)

    record_idx, score, sequence_ratio, dice = best
    old_item = index.records[record_idx].item if record_idx is not None else None
    status = "new"
    if score >= AUTO_THRESHOLD and old_item is not None:
        status = "auto_inherited"
    elif score >= REVIEW_THRESHOLD and old_item is not None:
        status = "needs_review"

    return {
        "status": status,
        "similarity": round(score, 6),
        "sequence_ratio": round(sequence_ratio, 6),
        "dice_ratio": round(dice, 6),
        "aligned_item_id": str(old_item.get("item_id")) if status == "auto_inherited" and old_item else None,
        "best_candidate_item_id": str(old_item.get("item_id")) if old_item else None,
        "best_candidate_stem": str(old_item.get("stem") or "") if old_item else "",
        "old_item": old_item,
    }


def apply_manual_override(question: dict, override: dict, old_by_item_id: dict[str, dict]) -> dict:
    old_item = old_by_item_id.get(str(override.get("aligned_item_id") or ""))
    if not old_item:
        raise ValueError(
            f"manual override for question {question.get('question_id')} references missing old item "
            f"{override.get('aligned_item_id')}"
        )

    normalized = question_normalized_stem(question)
    old_normalized = normalize_stem(str(old_item.get("stem") or ""))
    grams = stem_grams(normalized)
    old_grams = stem_grams(old_normalized)
    overlap = len(grams & old_grams)
    score, sequence_ratio, dice = similarity_score(normalized, old_normalized, overlap, grams, old_grams)
    manual_override = {
        "decision": override.get("decision"),
        "reviewer": override.get("reviewer"),
        "evidence": override.get("evidence"),
    }
    return {
        "status": "manual_inherited",
        "similarity": round(score, 6),
        "sequence_ratio": round(sequence_ratio, 6),
        "dice_ratio": round(dice, 6),
        "aligned_item_id": str(old_item.get("item_id") or ""),
        "best_candidate_item_id": str(old_item.get("item_id") or ""),
        "best_candidate_stem": str(old_item.get("stem") or ""),
        "old_item": old_item,
        "manual_override": manual_override,
    }


def apply_manual_resolution(alignment: dict, resolution: dict) -> dict:
    resolved = dict(alignment)
    resolved["status"] = resolution["decision"]
    resolved["aligned_item_id"] = None
    resolved["old_item"] = None
    resolved["manual_resolution"] = {
        "decision": resolution.get("decision"),
        "reason": resolution.get("reason"),
        "reviewer": resolution.get("reviewer"),
        "evidence": resolution.get("evidence"),
    }
    return resolved


def is_legacy_gaokao(question: dict) -> bool:
    group_key = str(question.get("group_key") or "")
    return bool(re.search(r"^(?:200[8-9]|201[0-6])年高考.*上海", group_key))


def has_answer_type_mismatch(question: dict) -> bool:
    return "answer_type_mismatch" in set(question.get("quality_flags") or [])


def effective_answer_blocks(question: dict) -> list[dict]:
    if has_answer_type_mismatch(question):
        return []
    return list(question.get("answer_blocks") or [])


def classify_pool(question: dict) -> dict:
    tags: list[str] = []
    if is_legacy_gaokao(question):
        tags.append("legacy")
    if has_answer_type_mismatch(question):
        tags.append("excluded_answerless")
    if not tags:
        tags.append("main")

    if "excluded_answerless" in tags:
        pool = "excluded_answerless"
    elif "legacy" in tags:
        pool = "legacy"
    else:
        pool = "main"
    return {
        "pool": pool,
        "pool_tags": tags,
        "service_eligible": pool == "main",
    }


def inherited_fields(old_item: dict) -> dict:
    return {
        "kg_nodes": old_item.get("kg_nodes") or [],
        "knowledge_points": old_item.get("knowledge_points") or [],
        "rubric": old_item.get("rubric") or [],
        "standard_solution": old_item.get("standard_solution") or {},
        "answer_verification": {
            "verification_status": old_item.get("verification_status"),
            "confidence": old_item.get("confidence"),
            "source_pipeline": old_item.get("_pipeline"),
        },
    }


def build_item_v4(question: dict, alignment: dict, golden_meta: dict | None = None) -> dict:
    pool = classify_pool(question)
    stem_text = blocks_text(question.get("stem_blocks", []), include_media_placeholder=True)
    item = {
        "schema_version": "ws3_schema_v4_candidate_1",
        "item_id": str(question.get("question_id") or question.get("question_id_base") or ""),
        "source_question_id": str(question.get("question_id") or ""),
        "pool": pool["pool"],
        "pool_tags": pool["pool_tags"],
        "service_eligible": pool["service_eligible"],
        "quality_flags": list(question.get("quality_flags") or []),
        "group_key": question.get("group_key"),
        "source_path": question.get("source_path"),
        "answer_source_path": question.get("answer_source_path"),
        "q_num": question.get("q_num"),
        "section_num": question.get("section_num"),
        "local_question_id": question.get("local_question_id"),
        "stem_blocks": question.get("stem_blocks") or [],
        "answer_blocks_effective": effective_answer_blocks(question),
        "answer_available": bool(effective_answer_blocks(question)),
        "analysis_blocks": question.get("analysis_blocks") or [],
        "stem_text": stem_text,
        "stem_normalized": question_normalized_stem(question),
        "stem_hash": question.get("stem_hash")
        or hashlib.sha1(question_normalized_stem(question).encode("utf-8")).hexdigest(),
        "alignment": {
            "status": alignment["status"],
            "similarity": alignment["similarity"],
            "sequence_ratio": alignment["sequence_ratio"],
            "dice_ratio": alignment["dice_ratio"],
            "aligned_item_id": alignment["aligned_item_id"],
            "best_candidate_item_id": alignment["best_candidate_item_id"],
        },
    }
    if alignment.get("manual_override"):
        item["alignment"]["manual_override"] = alignment["manual_override"]
    if alignment.get("manual_resolution"):
        item["alignment"]["manual_resolution"] = alignment["manual_resolution"]
    if golden_meta:
        item["golden_candidate"] = golden_meta
    if alignment["status"] in {"auto_inherited", "manual_inherited"} and alignment.get("old_item"):
        item.update(inherited_fields(alignment["old_item"]))
    return item


def load_formal_golden_metadata(golden_dir: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    if not golden_dir.exists():
        return metadata
    for path in sorted(golden_dir.glob("round2_*/question.json")):
        question = json.loads(path.read_text(encoding="utf-8"))
        candidate = dict(question.get("golden_candidate") or {})
        if candidate.get("set_role") != "formal":
            continue
        qid = str(question.get("question_id") or "")
        if qid:
            metadata[qid] = {
                "candidate_id": candidate.get("candidate_id") or path.parent.name,
                "category": candidate.get("category"),
                "set_role": candidate.get("set_role"),
            }
    return metadata


def review_queue_row(question: dict, alignment: dict) -> dict:
    return {
        "question_id": question.get("question_id"),
        "group_key": question.get("group_key"),
        "q_num": question.get("q_num"),
        "similarity": alignment["similarity"],
        "sequence_ratio": alignment["sequence_ratio"],
        "dice_ratio": alignment["dice_ratio"],
        "best_candidate_item_id": alignment["best_candidate_item_id"],
        "new_stem": blocks_text(question.get("stem_blocks", []), include_media_placeholder=True)[:700],
        "candidate_stem": alignment.get("best_candidate_stem", "")[:700],
    }


def write_alignment_report(
    path: Path,
    items: list[dict],
    queue: list[dict],
    gold_audit_rows: list[dict],
    golden_meta: dict[str, dict],
    summary: dict,
) -> None:
    pool_counts = Counter(item["pool"] for item in items)
    tag_counts = Counter(tag for item in items for tag in item.get("pool_tags", []))
    alignment_counts = Counter(item["alignment"]["status"] for item in items)
    formal_ids = set(golden_meta)
    hit_ids = {item["source_question_id"] for item in items if item["source_question_id"] in formal_ids}
    auto_gold_ids = {
        item["source_question_id"]
        for item in items
        if item["source_question_id"] in formal_ids and item["alignment"]["status"] == "auto_inherited"
    }
    inherited_gold_ids = {
        item["source_question_id"]
        for item in items
        if item["source_question_id"] in formal_ids
        and item["alignment"]["status"] in {"auto_inherited", "manual_inherited"}
    }
    resolved_gold_ids = {
        item["source_question_id"]
        for item in items
        if item["source_question_id"] in formal_ids and item["alignment"]["status"] in RESOLVED_ALIGNMENT_STATUSES
    }
    source_rank_hits = sum(1 for row in gold_audit_rows if row.get("source_rank_match") is True)
    override_candidates = sum(1 for row in gold_audit_rows if row.get("override_candidate") is True)
    manual_overrides = sum(1 for row in gold_audit_rows if row.get("gold_alignment_bucket") == "manual_override_inherited")
    manual_no_inherit = sum(1 for row in gold_audit_rows if row.get("gold_alignment_bucket") == "manual_do_not_inherit")
    manual_split_required = sum(1 for row in gold_audit_rows if row.get("gold_alignment_bucket") == "manual_split_required")
    unresolved_gold = sum(1 for row in gold_audit_rows if row.get("gold_alignment_bucket") == "manual_review_required")
    gold_gaps = sorted(formal_ids - resolved_gold_ids)
    sample_pool = [
        item
        for item in items
        if item["pool"] == "main" and item["alignment"]["status"] == "auto_inherited" and item.get("kg_nodes")
    ]
    sample = sample_pool[:20]
    if len(sample) < 20:
        sample = [item for item in items if item["alignment"]["status"] == "auto_inherited" and item.get("kg_nodes")][:20]

    lines = [
        "# WS3 Schema v4 Alignment Report",
        "",
        "## Summary",
        "",
        f"- Total items: {len(items)}",
        f"- Alignment counts: {dict(alignment_counts)}",
        f"- Review queue rows: {len(queue)}",
        f"- Review queue ratio: {summary['review_queue_ratio']:.2%}",
        f"- Main-pool review queue ratio: {summary['main_pool_review_queue_ratio']:.2%}",
        "",
        "## Pool Counts",
        "",
    ]
    for name, count in sorted(pool_counts.items()):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Pool Tag Counts", ""])
    for name, count in sorted(tag_counts.items()):
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## Golden Formal Set",
            "",
            f"- Present in items_v4: {len(hit_ids)} / {len(formal_ids)}",
            f"- Auto-inherited at >=0.92: {len(auto_gold_ids)} / {len(formal_ids)}",
            f"- Inherited with strict auto or manual override: {len(inherited_gold_ids)} / {len(formal_ids)}",
            f"- Gold alignment decisions resolved: {len(resolved_gold_ids)} / {len(formal_ids)}",
            f"- Manual overrides inherited: {manual_overrides} / {len(formal_ids)}",
            f"- Manual do-not-inherit decisions: {manual_no_inherit} / {len(formal_ids)}",
            f"- Manual split-required decisions: {manual_split_required} / {len(formal_ids)}",
            f"- Same-source rank matches: {source_rank_hits} / {len(formal_ids)}",
            f"- Source-rank override candidates: {override_candidates} / {len(formal_ids)}",
            f"- Unresolved manual-review gold rows: {unresolved_gold} / {len(formal_ids)}",
            "- Detailed audit: `gold_alignment_audit.jsonl`",
            "- Review candidates: `gold_alignment_review_candidates.jsonl`",
        ]
    )
    if gold_gaps:
        lines.append("- Golden rows still unresolved after strict auto/manual decisions:")
        for qid in gold_gaps[:50]:
            meta = golden_meta.get(qid, {})
            lines.append(f"  - `{meta.get('candidate_id', qid)}` question_id=`{qid}`")
    lines.extend(["", "## Manual Trust Sample For Auto-Inherited KG", ""])
    lines.append("| # | item_id | aligned_item_id | similarity | inherited_kg_nodes | stem_excerpt |")
    lines.append("|---|---|---|---:|---|---|")
    for idx, item in enumerate(sample, start=1):
        kg = "、".join(str(node) for node in item.get("kg_nodes", [])[:4])
        stem = str(item.get("stem_text", "")).replace("|", " ")[:80]
        lines.append(
            f"| {idx} | `{item['item_id']}` | `{item['alignment']['aligned_item_id']}` | "
            f"{item['alignment']['similarity']:.3f} | {kg} | {stem} |"
        )
    lines.extend(
        [
            "",
            "## Review Queue Policy",
            "",
            "- `similarity >= 0.92`: automatic inheritance.",
            "- `0.75 <= similarity < 0.92`: written to `alignment_review_queue.jsonl`.",
            "- `similarity < 0.75`: treated as a new question.",
            "- `answer_type_mismatch` rows keep raw blocks but have empty `answer_blocks_effective` and are not service eligible.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_rank_lookup(old_items: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    counters: Counter[str] = Counter()
    for item in old_items:
        source_key = normalize_source_key(str(item.get("source") or ""))
        if not source_key:
            continue
        counters[source_key] += 1
        item_id = str(item.get("item_id") or "")
        if item_id and item_id not in grouped[source_key]:
            grouped[source_key][item_id] = counters[source_key]
    return dict(grouped)


def build_gold_alignment_audit_rows(items: list[dict], old_items: list[dict]) -> list[dict]:
    rank_lookup = source_rank_lookup(old_items)
    rows: list[dict] = []
    for item in items:
        golden = item.get("golden_candidate")
        if not golden or golden.get("set_role") != "formal":
            continue
        source_key = normalize_source_key(str(item.get("group_key") or ""))
        best_candidate_item_id = item["alignment"].get("best_candidate_item_id")
        source_rank = rank_lookup.get(source_key, {}).get(str(best_candidate_item_id))
        q_num = item.get("q_num")
        source_rank_match = None
        if isinstance(source_rank, int) and isinstance(q_num, int):
            source_rank_match = source_rank == q_num
        alignment_status = item["alignment"].get("status")
        override_candidate = alignment_status != "auto_inherited" and source_rank_match is True
        if alignment_status == "auto_inherited":
            gold_alignment_bucket = "strict_text_auto"
        elif alignment_status == "manual_inherited":
            gold_alignment_bucket = "manual_override_inherited"
        elif alignment_status in {"manual_do_not_inherit", "manual_split_required"}:
            gold_alignment_bucket = alignment_status
        elif override_candidate:
            gold_alignment_bucket = "source_rank_override_candidate"
        else:
            gold_alignment_bucket = "manual_review_required"
        row = {
            "candidate_id": golden.get("candidate_id"),
            "question_id": item.get("source_question_id"),
            "set_role": golden.get("set_role"),
            "pool": item.get("pool"),
            "alignment_status": alignment_status,
            "gold_alignment_bucket": gold_alignment_bucket,
            "aligned_item_id": item["alignment"].get("aligned_item_id"),
            "best_candidate_item_id": best_candidate_item_id,
            "similarity": item["alignment"].get("similarity"),
            "q_num": q_num,
            "source_rank": source_rank,
            "source_rank_match": source_rank_match,
            "override_candidate": override_candidate,
            "needs_manual_review": alignment_status not in RESOLVED_ALIGNMENT_STATUSES,
        }
        if alignment_status == "manual_inherited":
            row["manual_override"] = True
        if alignment_status in {"manual_do_not_inherit", "manual_split_required"}:
            manual_resolution = item["alignment"].get("manual_resolution") or {}
            row["manual_resolution"] = True
            row["manual_resolution_reason"] = manual_resolution.get("reason")
        rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("candidate_id") or ""))


def build_gold_alignment_review_candidates(items: list[dict], gold_audit_rows: list[dict], old_items: list[dict]) -> list[dict]:
    item_by_question_id = {item.get("source_question_id"): item for item in items}
    old_by_item_id = {str(item.get("item_id") or ""): item for item in old_items}
    rows: list[dict] = []
    for audit in gold_audit_rows:
        if not audit.get("needs_manual_review"):
            continue
        question_id = audit.get("question_id")
        item = item_by_question_id.get(question_id, {})
        old_item = old_by_item_id.get(str(audit.get("best_candidate_item_id") or ""), {})
        rows.append(
            {
                "candidate_id": audit.get("candidate_id"),
                "question_id": question_id,
                "gold_alignment_bucket": audit.get("gold_alignment_bucket"),
                "best_candidate_item_id": audit.get("best_candidate_item_id"),
                "similarity": audit.get("similarity"),
                "q_num": audit.get("q_num"),
                "source_rank": audit.get("source_rank"),
                "source_rank_match": audit.get("source_rank_match"),
                "new_stem": str(item.get("stem_text") or "")[:700],
                "candidate_stem": str(old_item.get("stem") or "")[:700],
                "recommended_action": "manual_override_if_reviewer_confirms"
                if audit.get("override_candidate")
                else "manual_review_required",
            }
        )
    return rows


def build_ws3_outputs(
    questions_path: Path,
    old_items_path: Path,
    golden_dir: Path,
    out_dir: Path,
    manual_overrides_path: Path | None = None,
    manual_resolutions_path: Path | None = None,
) -> dict:
    questions = load_jsonl(questions_path)
    old_items = load_jsonl(old_items_path)
    golden_meta = load_formal_golden_metadata(golden_dir)
    manual_overrides = load_manual_overrides(manual_overrides_path)
    manual_resolutions = load_manual_resolutions(manual_resolutions_path)
    index = build_alignment_index(old_items)
    old_by_item_id = {str(item.get("item_id") or ""): item for item in old_items}

    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    queue: list[dict] = []
    for question in questions:
        alignment = align_question(question, index)
        qid = str(question.get("question_id") or "")
        if alignment["status"] != "auto_inherited" and qid in manual_overrides:
            alignment = apply_manual_override(question, manual_overrides[qid], old_by_item_id)
        elif alignment["status"] != "auto_inherited" and qid in manual_resolutions:
            alignment = apply_manual_resolution(alignment, manual_resolutions[qid])
        item = build_item_v4(question, alignment, golden_meta.get(qid))
        items.append(item)
        if alignment["status"] == "needs_review":
            queue.append(review_queue_row(question, alignment))

    total = len(items)
    main_items = [item for item in items if item["pool"] == "main"]
    main_queue_count = sum(1 for item in main_items if item["alignment"]["status"] == "needs_review")
    gold_audit_rows = build_gold_alignment_audit_rows(items, old_items)
    gold_review_rows = build_gold_alignment_review_candidates(items, gold_audit_rows, old_items)
    summary = {
        "total_items": total,
        "pool_counts": dict(Counter(item["pool"] for item in items)),
        "pool_tag_counts": dict(Counter(tag for item in items for tag in item.get("pool_tags", []))),
        "alignment_counts": dict(Counter(item["alignment"]["status"] for item in items)),
        "review_queue_rows": len(queue),
        "review_queue_ratio": len(queue) / total if total else 0.0,
        "main_pool_review_queue_ratio": main_queue_count / len(main_items) if main_items else 0.0,
    }

    write_jsonl(out_dir / "items_v4.jsonl", items)
    write_jsonl(out_dir / "alignment_review_queue.jsonl", queue)
    write_jsonl(out_dir / "gold_alignment_audit.jsonl", gold_audit_rows)
    write_jsonl(out_dir / "gold_alignment_review_candidates.jsonl", gold_review_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_alignment_report(out_dir / "alignment_report.md", items, queue, gold_audit_rows, golden_meta, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/ws1_batch_v4_20260703/questions_deduped.jsonl"))
    parser.add_argument("--old-items", type=Path, default=Path("data/item_bank/chemistry_v3_6695.jsonl"))
    parser.add_argument("--golden-dir", type=Path, default=Path("data/ws1_batch_v4_20260703/golden_round2"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/yher_ws3_v1"))
    parser.add_argument("--manual-overrides", type=Path, default=None)
    parser.add_argument("--manual-resolutions", type=Path, default=None)
    args = parser.parse_args()
    summary = build_ws3_outputs(
        args.questions,
        args.old_items,
        args.golden_dir,
        args.out_dir,
        args.manual_overrides,
        args.manual_resolutions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
