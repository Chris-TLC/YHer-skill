#!/usr/bin/env python3
"""Build a balanced visual item eval set from the visual asset manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_ITEM_BANK = SKILL_DIR / "data" / "item_bank" / "chemistry_v3_6695.jsonl"
DEFAULT_PDF_ITEMS = SKILL_DIR / "data" / "from_pdf" / "all_from_pdf_v3.jsonl"
DEFAULT_VISUAL_MANIFEST = SKILL_DIR / "data" / "quality" / "visual_asset_manifest.jsonl"
DEFAULT_EVAL_DIR = SKILL_DIR / "data" / "evals"
DEFAULT_PILOT = Path("/tmp/yher_multimodal_pilot.json")

REQUIRED_CATEGORIES = [
    "crystal_cell",
    "experiment_device",
    "process_flow",
    "chart_curve",
    "organic_structure",
    "electrochem_device",
    "other",
]

CROP_FIRST_TIERS = {"item_crop_candidate", "page_region_candidate"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def load_excluded_item_ids(paths: Iterable[Path] | None) -> set[str]:
    excluded: set[str] = set()
    for path in paths or []:
        for row in load_jsonl(Path(path)):
            item_id = row.get("item_id")
            if item_id:
                excluded.add(str(item_id))
    return excluded


def crop_first_rank(row: dict[str, Any]) -> int:
    if row.get("crop_tier") in CROP_FIRST_TIERS or row.get("crop_path"):
        return 0
    return 1


def load_crop_evidence(path: Path | None) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    if not path:
        return evidence
    for row in load_jsonl(Path(path)):
        item_id = row.get("item_id")
        if not item_id:
            continue
        evidence[str(item_id)] = {
            key: row.get(key)
            for key in ("crop_tier", "crop_path", "crop_hash", "bbox_pdf_points", "anchor_used")
            if row.get(key)
        }
    return evidence


def qid_from_pdf_item(item: dict[str, Any]) -> str:
    raw = f"{item.get('_source_file','')}|{item.get('q_num','')}|{item.get('stem','')[:40]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def pilot_seed_ids(pilot_path: Path, pdf_items_path: Path) -> list[str]:
    if not pilot_path.exists():
        return []
    pdf_by_key = {}
    for item in load_jsonl(pdf_items_path):
        pdf_by_key[(item.get("_source_file"), str(item.get("q_num")), item.get("_page"))] = item
    try:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    ids: list[str] = []
    for row in pilot.get("results", []):
        item = pdf_by_key.get((row.get("source_file"), str(row.get("q_num")), row.get("page")))
        if item:
            ids.append(qid_from_pdf_item(item))
    return ids


def build_eval_set(
    item_bank_path: Path = DEFAULT_ITEM_BANK,
    visual_manifest_path: Path = DEFAULT_VISUAL_MANIFEST,
    pdf_items_path: Path = DEFAULT_PDF_ITEMS,
    pilot_path: Path = DEFAULT_PILOT,
    exclude_items_paths: list[Path] | None = None,
    crop_evidence_path: Path | None = None,
    per_category: int = 3,
    max_items: int = 35,
    max_per_category: int | None = None,
    category_targets: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_item_ids = load_excluded_item_ids(exclude_items_paths)
    crop_evidence = load_crop_evidence(crop_evidence_path)
    items_by_id = {
        row.get("item_id"): row
        for row in load_jsonl(Path(item_bank_path))
        if row.get("item_id")
    }
    visual_rows = [
        row
        for row in load_jsonl(Path(visual_manifest_path))
        if row.get("item_id") in items_by_id
        and str(row.get("item_id")) not in excluded_item_ids
        and row.get("match_tier") == "strong"
        and row.get("page_image_path")
    ]
    for row in visual_rows:
        evidence = crop_evidence.get(str(row.get("item_id")))
        if evidence:
            row.update(evidence)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in visual_rows:
        buckets[row.get("category") or "other"].append(row)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    visual_by_id = {row["item_id"]: row for row in visual_rows}
    for item_id in pilot_seed_ids(Path(pilot_path), Path(pdf_items_path)):
        if item_id in excluded_item_ids:
            continue
        row = visual_by_id.get(item_id)
        if row and item_id not in seen:
            selected.append(row)
            seen.add(item_id)

    targets = category_targets or {}

    for category in REQUIRED_CATEGORIES:
        target_for_category = targets.get(category, per_category)
        if target_for_category <= 0:
            continue
        rows = sorted(
            buckets.get(category, []),
            key=lambda r: (
                crop_first_rank(r),
                r.get("difficulty") or "",
                r.get("question_type") or "",
                r.get("source_file") or "",
                r.get("item_id") or "",
            ),
        )
        category_existing = sum(1 for row in selected if row.get("category") == category)
        for row in rows:
            if category_existing >= target_for_category:
                break
            if row["item_id"] in seen:
                continue
            selected.append(row)
            seen.add(row["item_id"])
            category_existing += 1

    if len(selected) < max_items:
        category_counts = Counter(row.get("category") or "other" for row in selected)
        remaining = sorted(
            [row for row in visual_rows if row["item_id"] not in seen],
            key=lambda r: (
                crop_first_rank(r),
                r.get("category") or "",
                r.get("difficulty") or "",
                r.get("source_file") or "",
                r.get("item_id") or "",
            ),
        )
        category_cap = max_per_category if max_per_category and max_per_category > 0 else None
        for enforce_cap in (True, False):
            if len(selected) >= max_items:
                break
            for row in remaining:
                if row["item_id"] in seen:
                    continue
                category = row.get("category") or "other"
                if category in targets and category_counts[category] >= targets[category]:
                    continue
                if enforce_cap and category_cap is not None and category_counts[category] >= category_cap:
                    continue
                selected.append(row)
                seen.add(row["item_id"])
                category_counts[category] += 1
                if len(selected) >= max_items:
                    break

    eval_rows: list[dict[str, Any]] = []
    for row in selected[:max_items]:
        item = items_by_id[row["item_id"]]
        solution = item.get("standard_solution") or {}
        eval_rows.append(
            {
                "item_id": row["item_id"],
                "category": row.get("category", "other"),
                "source_file": row.get("source_file", ""),
                "source_path": row.get("source_path", ""),
                "page": row.get("declared_page"),
                "best_text_page": row.get("best_text_page"),
                "page_image_path": row.get("page_image_path", ""),
                "page_image_hash": row.get("page_image_hash", ""),
                "crop_path": row.get("crop_path"),
                "crop_hash": row.get("crop_hash"),
                "crop_tier": row.get("crop_tier"),
                "match_tier": row.get("match_tier"),
                "visible_anchors": row.get("visible_anchors", []),
                "question_type": item.get("question_type", row.get("question_type", "")),
                "difficulty": item.get("difficulty", row.get("difficulty", "")),
                "kg_nodes": item.get("kg_nodes", []),
                "stem": item.get("stem", ""),
                "options": item.get("options", {}),
                "standard_answer": solution.get("standard_answer", row.get("answer", "")),
                "final_answers": solution.get("final_answers", []),
                "rubric": item.get("rubric", []),
                "manual_review_status": "pending",
            }
        )

    summary = {
        "total_eval_items": len(eval_rows),
        "per_category_target": per_category,
        "max_items": max_items,
        "category_counts": dict(Counter(row["category"] for row in eval_rows)),
        "source_count": len(set(row["source_file"] for row in eval_rows)),
        "pilot_seeded": len([row for row in eval_rows if row["item_id"] in set(pilot_seed_ids(Path(pilot_path), Path(pdf_items_path)))]),
        "excluded_items": len(excluded_item_ids),
        "max_per_category": max_per_category,
        "category_targets": targets,
        "crop_first_candidates": sum(1 for row in eval_rows if row.get("crop_tier") in CROP_FIRST_TIERS or row.get("crop_path")),
        "crop_evidence_path": str(crop_evidence_path) if crop_evidence_path else "",
    }
    return eval_rows, summary


def parse_category_targets(raw: str | None) -> dict[str, int] | None:
    if not raw:
        return None
    targets: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid category target {part!r}; expected category=count")
        category, value = part.split("=", 1)
        category = category.strip()
        try:
            count = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"invalid count for category target {part!r}") from exc
        if count < 0:
            raise ValueError(f"invalid negative count for category target {part!r}")
        targets[category] = count
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Build balanced visual eval set.")
    parser.add_argument("--item-bank", type=Path, default=DEFAULT_ITEM_BANK)
    parser.add_argument("--visual-manifest", type=Path, default=DEFAULT_VISUAL_MANIFEST)
    parser.add_argument("--pdf-items", type=Path, default=DEFAULT_PDF_ITEMS)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--exclude-items", type=Path, action="append", default=[])
    parser.add_argument("--crop-evidence", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_EVAL_DIR / "visual_item_eval_set.jsonl")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_EVAL_DIR / "visual_item_eval_set_summary.json")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=35)
    parser.add_argument("--max-per-category", type=int, default=None)
    parser.add_argument(
        "--category-targets",
        default=None,
        help="Optional comma-separated category=count targets, e.g. crystal_cell=5,chart_curve=9.",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    try:
        category_targets = parse_category_targets(args.category_targets)
    except ValueError as exc:
        parser.error(str(exc))

    rows, summary = build_eval_set(
        item_bank_path=args.item_bank,
        visual_manifest_path=args.visual_manifest,
        pdf_items_path=args.pdf_items,
        pilot_path=args.pilot,
        exclude_items_paths=args.exclude_items,
        crop_evidence_path=args.crop_evidence,
        per_category=args.per_category,
        max_items=args.max_items,
        max_per_category=args.max_per_category,
        category_targets=category_targets,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        write_jsonl(args.out, rows)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out}")
        print(f"WROTE {args.summary_out}")
    else:
        print("DRY RUN: pass --write to write data/evals outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
