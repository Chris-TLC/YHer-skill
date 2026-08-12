#!/usr/bin/env python3
"""Build visual failure exclusions and manual audit queues from batch reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


PROVIDER_FAILURES = {"provider_error", "timeout", "transport_error"}


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


def batch_label(path: Path) -> str:
    name = path.name
    if name.endswith("_failure_report.jsonl"):
        return name[: -len("_failure_report.jsonl")]
    return path.stem


def has_high_confidence_error(rows: list[dict[str, Any]]) -> bool:
    return any("high_confidence_error" in (row.get("error_types") or []) for row in rows)


def is_provider_only(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        primary = str(row.get("primary_failure_category") or "")
        errors = {str(err) for err in (row.get("error_types") or [])}
        if primary not in PROVIDER_FAILURES and not errors.issubset(PROVIDER_FAILURES):
            return False
    return True


def compact_failure(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "batch": source,
        "primary_failure_category": row.get("primary_failure_category"),
        "error_types": row.get("error_types") or [],
        "visible_pass": row.get("visible_pass"),
        "answer_match": row.get("answer_match"),
        "understanding_pass": row.get("understanding_pass"),
        "confidence": row.get("confidence"),
        "crop_ready_for_promotion": row.get("crop_ready_for_promotion"),
    }


def recommended_action(exclude_reason: str) -> str:
    if exclude_reason == "high_confidence_error":
        return "answer_or_standard_key_audit"
    return "manual_visual_evidence_audit"


def build_failure_exclusion(
    failure_report_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources_by_id: dict[str, list[str]] = defaultdict(list)

    for path in failure_report_paths:
        source = batch_label(Path(path))
        for row in load_jsonl(Path(path)):
            item_id = row.get("item_id")
            if not item_id:
                continue
            copied = dict(row)
            copied["_source_batch"] = source
            grouped[str(item_id)].append(copied)
            sources_by_id[str(item_id)].append(source)

    exclusion_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    provider_only_items = 0

    for item_id in sorted(grouped):
        rows = grouped[item_id]
        category = str(rows[0].get("category") or "other")
        non_provider_rows = [row for row in rows if not is_provider_only([row])]
        if is_provider_only(rows):
            provider_only_items += 1
            continue
        if has_high_confidence_error(rows):
            exclude_reason = "high_confidence_error"
        elif len(non_provider_rows) >= 2:
            exclude_reason = "repeated_non_provider_failure"
        else:
            continue

        failures = [
            compact_failure(row, str(row.get("_source_batch") or "unknown"))
            for row in rows
        ]
        exclusion_row = {
            "item_id": item_id,
            "category": category,
            "exclude_reason": exclude_reason,
            "failure_attempts": len(rows),
            "source_batches": sorted(set(sources_by_id[item_id])),
            "failures": failures,
        }
        exclusion_rows.append(exclusion_row)
        audit_rows.append(
            {
                **exclusion_row,
                "recommended_action": recommended_action(exclude_reason),
                "audit_status": "pending",
            }
        )

    category_counter = Counter(row["category"] for row in exclusion_rows)
    reason_counter = Counter(row["exclude_reason"] for row in exclusion_rows)
    summary = {
        "source_files": [str(path) for path in failure_report_paths],
        "failure_records": sum(len(rows) for rows in grouped.values()),
        "unique_failure_items": len(grouped),
        "exclude_items": len(exclusion_rows),
        "provider_only_items": provider_only_items,
        "category_counts": dict(sorted(category_counter.items())),
        "exclude_reason_counts": dict(sorted(reason_counter.items())),
        "ordinary_batch_exclusion_policy": "high_confidence_or_repeated_non_provider_failure",
        "reason": "Exclude these from ordinary paid visual batches; review via answer/manual audit instead.",
    }
    return exclusion_rows, audit_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build visual failure exclusion and audit queue artifacts.")
    parser.add_argument("--failure-report", type=Path, action="append", required=True)
    parser.add_argument("--out-exclude", type=Path, required=True)
    parser.add_argument("--out-audit", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    exclusion_rows, audit_rows, summary = build_failure_exclusion(args.failure_report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        write_jsonl(args.out_exclude, exclusion_rows)
        write_jsonl(args.out_audit, audit_rows)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out_exclude}")
        print(f"WROTE {args.out_audit}")
        print(f"WROTE {args.summary_out}")
    else:
        print("DRY RUN: pass --write to write artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
