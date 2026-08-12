#!/usr/bin/env python3
"""Report full visual-item readability and strong-rate baseline.

This is intentionally read-only unless --write is passed. It does not call any
vision model; it summarizes existing manifests/results and reports whether API
keys are configured as booleans only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_ITEM_QUALITY = SKILL_DIR / "data" / "quality" / "item_quality_manifest.jsonl"
DEFAULT_VISUAL_MANIFEST = SKILL_DIR / "data" / "quality" / "visual_asset_manifest.jsonl"
DEFAULT_UNDERSTANDING_RESULTS = SKILL_DIR / "data" / "evals" / "visual_understanding_results.jsonl"
DEFAULT_OUT = SKILL_DIR / "data" / "quality" / "visual_quality_baseline_report.json"
DEFAULT_ENV = SKILL_DIR / ".env"


def load_jsonl_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, [{"path": str(path), "line": 0, "error": "missing_file"}]
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"path": str(path), "line": line_number, "error": str(exc)})
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                errors.append({"path": str(path), "line": line_number, "error": "row_not_object"})
    return rows, errors


def load_env_presence(env_path: Path = DEFAULT_ENV) -> dict[str, bool]:
    values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in {"OPENAI_API_KEY", "DASHSCOPE_API_KEY", "CODEX_API_KEY"}:
                values[key] = value.strip().strip('"').strip("'")
    for key in {"OPENAI_API_KEY", "DASHSCOPE_API_KEY", "CODEX_API_KEY"}:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return {
        "openai": bool(values.get("OPENAI_API_KEY")),
        "dashscope": bool(values.get("DASHSCOPE_API_KEY")),
        "codex": bool(values.get("CODEX_API_KEY")),
    }


def row_student_readable(row: dict[str, Any]) -> bool:
    if "student_readable" in row:
        return bool(row.get("student_readable"))
    return bool(row.get("usable_for_practice") and row.get("readability_status") == "pass")


def row_strong(row: dict[str, Any]) -> bool:
    if "strong" in row:
        return bool(row.get("strong"))
    return bool(row.get("usable_for_profile_evidence"))


def row_category(row: dict[str, Any], visual_by_id: dict[str, dict[str, Any]]) -> str:
    item_id = row.get("item_id")
    return str(row.get("category") or visual_by_id.get(item_id, {}).get("category") or "other")


def top_counter(counter: Counter[str], limit: int = 20) -> dict[str, int]:
    return dict(counter.most_common(limit))


def build_next_batch_candidates(
    visual_quality_rows: list[dict[str, Any]],
    understanding_by_id: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in visual_quality_rows:
        if row_strong(row):
            continue
        reasons = list(row.get("blocker_reasons") or [])
        reasons.extend(row.get("strong_blocker_reasons") or [])
        understanding = understanding_by_id.get(row.get("item_id"))
        if understanding:
            reasons.extend(understanding.get("error_types") or [])
        if row_student_readable(row) and not row_strong(row):
            priority = 0
            queue = "strong_review"
        elif row.get("visual_pipeline_stage") in {"asset_linked", "crop_readable"}:
            priority = 1
            queue = "crop_or_readability_review"
        else:
            priority = 2
            queue = row.get("review_queue") or "manual_review"
        candidates.append(
            {
                "item_id": row.get("item_id", ""),
                "category": row.get("category", "other"),
                "visual_pipeline_stage": row.get("visual_pipeline_stage", ""),
                "review_queue": queue,
                "reasons": sorted(set(str(reason) for reason in reasons if reason)),
                "_priority": priority,
            }
        )
    candidates.sort(key=lambda row: (row["_priority"], row["category"], row["item_id"]))
    for row in candidates:
        row.pop("_priority", None)
    return candidates[:limit]


def build_baseline_report(
    item_quality_path: Path = DEFAULT_ITEM_QUALITY,
    visual_manifest_path: Path = DEFAULT_VISUAL_MANIFEST,
    understanding_results_path: Path = DEFAULT_UNDERSTANDING_RESULTS,
    strong_target_rate: float = 0.8,
    env_path: Path = DEFAULT_ENV,
    next_batch_size: int = 200,
) -> dict[str, Any]:
    quality_rows, quality_errors = load_jsonl_with_errors(Path(item_quality_path))
    visual_rows, visual_errors = load_jsonl_with_errors(Path(visual_manifest_path))
    understanding_rows, understanding_errors = load_jsonl_with_errors(Path(understanding_results_path))

    visual_by_id = {row.get("item_id"): row for row in visual_rows if row.get("item_id")}
    understanding_by_id = {row.get("item_id"): row for row in understanding_rows if row.get("item_id")}
    visual_quality_rows = [row for row in quality_rows if row.get("needs_image")]
    if not visual_quality_rows:
        visual_quality_rows = [
            {
                "item_id": row.get("item_id"),
                "needs_image": True,
                "student_readable": bool(row.get("match_tier") == "strong" and row.get("page_image_path")),
                "strong": False,
                "visual_pipeline_stage": "asset_linked" if row.get("page_image_path") else "raw_visual_item",
                "category": row.get("category", "other"),
                "blocker_reasons": row.get("blocker_reasons") or [],
            }
            for row in visual_rows
        ]

    visual_denominator = len(visual_quality_rows)
    strong_target_count = math.ceil(max(0.0, min(1.0, strong_target_rate)) * visual_denominator)
    current_strong_count = sum(1 for row in visual_quality_rows if row_strong(row))
    student_readable_count = sum(1 for row in visual_quality_rows if row_student_readable(row))

    blocker_counter: Counter[str] = Counter()
    for row in visual_quality_rows:
        blocker_counter.update(str(reason) for reason in row.get("blocker_reasons") or [] if reason)
        blocker_counter.update(str(reason) for reason in row.get("strong_blocker_reasons") or [] if reason)
    for row in visual_rows:
        blocker_counter.update(str(reason) for reason in row.get("blocker_reasons") or [] if reason)
    for row in understanding_rows:
        blocker_counter.update(str(reason) for reason in row.get("error_types") or [] if reason)

    by_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "student_readable": 0,
            "strong": 0,
            "high_confidence_error": 0,
            "stages": Counter(),
        }
    )
    for row in visual_quality_rows:
        category = row_category(row, visual_by_id)
        stats = by_category[category]
        stats["total"] += 1
        stats["student_readable"] += int(row_student_readable(row))
        stats["strong"] += int(row_strong(row))
        stats["stages"].update([str(row.get("visual_pipeline_stage") or "")])
        understanding = understanding_by_id.get(row.get("item_id"), {})
        stats["high_confidence_error"] += int("high_confidence_error" in (understanding.get("error_types") or []))

    by_category_out: dict[str, dict[str, Any]] = {}
    for category, stats in sorted(by_category.items()):
        total = stats["total"] or 1
        by_category_out[category] = {
            "total": stats["total"],
            "student_readable": stats["student_readable"],
            "strong": stats["strong"],
            "strong_rate": round(stats["strong"] / total, 4),
            "student_readable_rate": round(stats["student_readable"] / total, 4),
            "high_confidence_error": stats["high_confidence_error"],
            "stages": dict(stats["stages"]),
        }

    recommendations: list[str] = []
    remaining_to_target = max(0, strong_target_count - current_strong_count)
    if remaining_to_target:
        recommendations.append(f"Need {remaining_to_target} additional visual items promoted to strong to reach {int(strong_target_rate * 100)}%.")
    if blocker_counter.get("missing_page_image"):
        recommendations.append("Prioritize missing page/crop image evidence before any model review.")
    if blocker_counter.get("high_confidence_error"):
        recommendations.append("Route high-confidence model errors to strong-model or human answer audit.")
    if student_readable_count < visual_denominator:
        recommendations.append("Keep non-student-readable items out of all student-facing flows until image/text evidence is fixed.")

    return {
        "report_type": "visual_quality_baseline",
        "strong_target_rate": strong_target_rate,
        "visual_denominator": visual_denominator,
        "strong_target_count": strong_target_count,
        "current_strong_count": current_strong_count,
        "current_strong_rate": round(current_strong_count / visual_denominator, 4) if visual_denominator else 0.0,
        "student_readable_count": student_readable_count,
        "student_readable_rate": round(student_readable_count / visual_denominator, 4) if visual_denominator else 0.0,
        "remaining_to_target": max(0, strong_target_count - current_strong_count),
        "visual_pipeline_stage": dict(Counter(str(row.get("visual_pipeline_stage") or "") for row in visual_quality_rows)),
        "review_queue": dict(Counter(str(row.get("review_queue") or "") for row in visual_quality_rows)),
        "visual_asset_tier": dict(Counter(str(row.get("match_tier") or "") for row in visual_rows)),
        "vl_summary": {
            "evaluated": len(understanding_rows),
            "visible_pass": sum(1 for row in understanding_rows if row.get("visible_pass")),
            "answer_match": sum(1 for row in understanding_rows if row.get("answer_match")),
            "understanding_pass": sum(1 for row in understanding_rows if row.get("understanding_pass")),
            "high_confidence_error": blocker_counter.get("high_confidence_error", 0),
            "models": dict(Counter(str(row.get("model") or "") for row in understanding_rows)),
        },
        "blocker_distribution": top_counter(blocker_counter),
        "by_category": by_category_out,
        "jsonl_errors": quality_errors + visual_errors + understanding_errors,
        "api_key_status": load_env_presence(env_path),
        "next_batch_size": next_batch_size,
        "next_batch_candidates": build_next_batch_candidates(visual_quality_rows, understanding_by_id, next_batch_size),
        "recommendations": recommendations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build read-only visual quality baseline report.")
    parser.add_argument("--item-quality", type=Path, default=DEFAULT_ITEM_QUALITY)
    parser.add_argument("--visual-manifest", type=Path, default=DEFAULT_VISUAL_MANIFEST)
    parser.add_argument("--understanding-results", type=Path, default=DEFAULT_UNDERSTANDING_RESULTS)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--target-rate", type=float, default=0.8)
    parser.add_argument("--next-batch-size", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    report = build_baseline_report(
        item_quality_path=args.item_quality,
        visual_manifest_path=args.visual_manifest,
        understanding_results_path=args.understanding_results,
        strong_target_rate=args.target_rate,
        env_path=args.env,
        next_batch_size=args.next_batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out}")
    else:
        print("DRY RUN: pass --write to write data/quality/visual_quality_baseline_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
