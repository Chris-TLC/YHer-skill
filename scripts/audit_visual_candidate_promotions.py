#!/usr/bin/env python3
"""Audit visual strong candidates before any official manifest promotion.

The script is intentionally conservative. It does not mutate official manifests.
It produces a promotion plan split into approved/manual_review/reject so a later
human or explicit promotion pass can decide what to copy into data/quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

SKILL_DIR = Path(__file__).parent.parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def by_item_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("item_id")): row for row in rows if row.get("item_id")}


def has_candidate_repair(understanding: dict[str, Any]) -> bool:
    repair = understanding.get("candidate_repair") or {}
    if not repair:
        return False
    if repair.get("official_source_has_answer") is False:
        return True
    if repair.get("do_not_copy_to_official_without_promotion_audit"):
        return True
    return bool(repair)


def has_cross_page_evidence(visual: dict[str, Any], understanding: dict[str, Any]) -> bool:
    return bool(
        visual.get("additional_image_paths")
        or visual.get("cross_page_evidence")
        or understanding.get("additional_image_paths")
        or understanding.get("cross_page_evidence")
    )


def has_structured_transcript_evidence(quality: dict[str, Any], understanding: dict[str, Any] | None) -> bool:
    if quality.get("visual_evidence_mode") == "structured_transcript":
        return True
    if not understanding:
        return False
    return bool(
        understanding.get("transcript_supported_strong")
        or understanding.get("visual_evidence_mode") == "structured_transcript"
    )


def audit_row(
    quality: dict[str, Any],
    visual: dict[str, Any] | None,
    understanding: dict[str, Any] | None,
    official_visual: dict[str, Any] | None,
) -> dict[str, Any]:
    item_id = str(quality.get("item_id") or "")
    review_reasons: list[str] = []
    reject_reasons: list[str] = []
    structured_transcript_evidence = has_structured_transcript_evidence(quality, understanding)

    if not quality.get("needs_image"):
        reject_reasons.append("not_visual_item")
    if not quality.get("strong"):
        reject_reasons.append("candidate_quality_not_strong")
    if quality.get("blocker_reasons"):
        reject_reasons.append("candidate_quality_has_blockers")
    if quality.get("strong_blocker_reasons"):
        reject_reasons.append("candidate_quality_has_strong_blockers")

    if not visual:
        reject_reasons.append("missing_candidate_visual_row")
    else:
        if visual.get("match_tier") != "strong":
            reject_reasons.append("visual_match_not_strong")
        if not visual.get("source_file"):
            reject_reasons.append("missing_source_file")
        if visual.get("declared_page") is None:
            reject_reasons.append("missing_declared_page")
        if not (visual.get("page_image_path") and visual.get("page_image_hash")):
            reject_reasons.append("missing_page_image_evidence")
        if not (visual.get("crop_path") and visual.get("crop_hash")) and not structured_transcript_evidence:
            reject_reasons.append("missing_crop_evidence")
        if (
            visual.get("crop_tier") not in {"item_crop", "item_crop_candidate"}
            and not structured_transcript_evidence
        ):
            reject_reasons.append("crop_tier_not_item_level")

    if not understanding:
        reject_reasons.append("missing_candidate_understanding_row")
    else:
        if not understanding.get("visible_pass"):
            reject_reasons.append("vl_visible_not_pass")
        if not understanding.get("answer_match"):
            reject_reasons.append("vl_answer_not_match")
        if not understanding.get("understanding_pass"):
            reject_reasons.append("vl_understanding_not_pass")
        if understanding.get("error_types"):
            reject_reasons.append("vl_has_error_types")
        if not understanding.get("model"):
            reject_reasons.append("missing_vl_model")
        if has_candidate_repair(understanding):
            review_reasons.append("candidate_answer_repair_requires_human_policy")

    if visual and understanding and has_cross_page_evidence(visual, understanding):
        review_reasons.append("cross_page_evidence_requires_promotion_policy")

    official_crop_hash = (official_visual or {}).get("crop_hash")
    official_crop_path = (official_visual or {}).get("crop_path")
    candidate_crop_hash = (visual or {}).get("crop_hash")
    if (
        official_visual
        and candidate_crop_hash
        and (official_crop_hash or official_crop_path)
        and official_crop_hash != candidate_crop_hash
    ):
        review_reasons.append("official_existing_crop_would_be_replaced")
    if official_visual and visual and visual.get("additional_image_paths") and not official_visual.get("additional_image_paths"):
        review_reasons.append("official_schema_needs_additional_image_paths")

    status = "approved"
    if reject_reasons:
        status = "reject"
    elif review_reasons:
        status = "manual_review"

    return {
        "item_id": item_id,
        "status": status,
        "promotion_scope": "understanding_structured_transcript"
        if structured_transcript_evidence
        else "visual_asset_and_understanding",
        "category": quality.get("category") or (visual or {}).get("category") or (understanding or {}).get("category") or "",
        "review_reasons": sorted(set(review_reasons)),
        "reject_reasons": sorted(set(reject_reasons)),
        "candidate_quality_evidence_id": quality.get("quality_evidence_id", ""),
        "candidate_crop_path": (visual or {}).get("crop_path", ""),
        "candidate_crop_hash": (visual or {}).get("crop_hash", ""),
        "candidate_page_image_path": (visual or {}).get("page_image_path", ""),
        "candidate_page_image_hash": (visual or {}).get("page_image_hash", ""),
        "additional_image_paths": (visual or {}).get("additional_image_paths") or (understanding or {}).get("additional_image_paths") or [],
        "vl_model": (understanding or {}).get("model", ""),
        "vl_raw_source": (understanding or {}).get("raw_source", ""),
        "candidate_repair": (understanding or {}).get("candidate_repair") or {},
    }


def audit_promotions(
    candidate_quality_path: Path,
    candidate_visual_path: Path,
    candidate_understanding_path: Path,
    official_visual_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    quality_rows = load_jsonl(Path(candidate_quality_path))
    visual_by_id = by_item_id(load_jsonl(Path(candidate_visual_path)))
    understanding_by_id = by_item_id(load_jsonl(Path(candidate_understanding_path)))
    official_visual_by_id = by_item_id(load_jsonl(Path(official_visual_path)))

    routed: dict[str, list[dict[str, Any]]] = {"approved": [], "manual_review": [], "reject": []}
    for quality in quality_rows:
        if not quality.get("needs_image") or not quality.get("strong"):
            continue
        row = audit_row(
            quality=quality,
            visual=visual_by_id.get(str(quality.get("item_id"))),
            understanding=understanding_by_id.get(str(quality.get("item_id"))),
            official_visual=official_visual_by_id.get(str(quality.get("item_id"))),
        )
        routed[row["status"]].append(row)

    reason_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    for status_rows in routed.values():
        for row in status_rows:
            reason_counter.update(row.get("review_reasons") or [])
            reason_counter.update(row.get("reject_reasons") or [])
            category_counter.update([str(row.get("category") or "other")])

    summary = {
        "candidate_visual_strong_rows": sum(len(rows) for rows in routed.values()),
        "approved": len(routed["approved"]),
        "manual_review": len(routed["manual_review"]),
        "reject": len(routed["reject"]),
        "reason_counts": dict(reason_counter.most_common()),
        "category_counts": dict(category_counter.most_common()),
        "official_manifest_changed": False,
    }
    return routed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit candidate visual strong rows before official promotion.")
    parser.add_argument("--candidate-quality", type=Path, required=True)
    parser.add_argument("--candidate-visual", type=Path, required=True)
    parser.add_argument("--candidate-understanding", type=Path, required=True)
    parser.add_argument("--official-visual", type=Path, default=SKILL_DIR / "data" / "quality" / "visual_asset_manifest.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    routed, summary = audit_promotions(
        candidate_quality_path=args.candidate_quality,
        candidate_visual_path=args.candidate_visual,
        candidate_understanding_path=args.candidate_understanding,
        official_visual_path=args.official_visual,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for status, rows in routed.items():
            write_jsonl(args.out_dir / f"{status}.jsonl", rows)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out_dir}")
    else:
        print("DRY RUN: pass --write to write promotion audit outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
