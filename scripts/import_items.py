#!/usr/bin/env python3
"""
Real-exam-item import pipeline (master blueprint B-1; runs offline, the engine layer does not depend on it).

Purpose: once Chris obtains real exam items, use this script to convert the raw items into item_bank/{topic}.jsonl.

Minimum format Chris must provide (5 fields per item, none optional):
  1. Stem (textualize data/figures; chemical formulas as plain text, e.g. N2+3H2⇌2NH3)
  2. Source (which year/paper/question number)
  3. Final answers (per sub-question)
  4. Solution steps
  5. Score per sub-question
Nice to have: official grading rubrics/scoring points (= ready-made rubric; prioritize real items with official rubrics).

Pipeline steps:
  parse -> auto-tag item type (vector) -> auto-tag KG nodes (vector) -> LLM semi-auto rubric split (MUST be human-reviewed)
       -> attach yihua-style explanation chunks -> attach videos -> write jsonl

Current implementation: a minimal usable version that writes directly from structured input (rubric provided by a human, or LLM-generated then human-reviewed).
Vector auto-attachment and LLM rubric splitting will hook in the retriever/llm_client when there is a real batch import need.

Usage example:
    python3 scripts/import_items.py --validate   # validate the existing item bank format
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

ITEM_BANK_DIR = SKILL_DIR / "data" / "item_bank"

REQUIRED_FIELDS = ["item_id", "stem", "kg_nodes", "standard_solution", "rubric"]
REQUIRED_RUBRIC_FIELDS = ["point_id", "desc", "keywords", "score"]


def validate_item(item: dict) -> list:
    """Validate whether an item conforms to the schema; return the list of problems (empty = valid)."""
    problems = []
    for f in REQUIRED_FIELDS:
        if f not in item or item[f] in (None, "", [], {}):
            problems.append(f"missing field: {f}")
    sol = item.get("standard_solution", {})
    if not sol.get("final_answers"):
        problems.append("standard_solution missing final_answers")
    if not sol.get("solution_steps"):
        problems.append("standard_solution missing solution_steps")
    for i, rp in enumerate(item.get("rubric", [])):
        for rf in REQUIRED_RUBRIC_FIELDS:
            if rf not in rp:
                problems.append(f"rubric[{i}] missing {rf}")
    if not any(rp.get("must_have") for rp in item.get("rubric", [])):
        problems.append("rubric has no must_have scoring points (at least 1 recommended)")
    return problems


def validate_bank() -> int:
    """Validate the whole item bank. Returns the number of invalid items."""
    if not ITEM_BANK_DIR.exists():
        print("Item bank directory does not exist:", ITEM_BANK_DIR)
        return 0
    total, bad = 0, 0
    for f in sorted(ITEM_BANK_DIR.glob("*.jsonl")):
        print(f"\n=== {f.name} ===")
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except Exception as e:
                print(f"  line {ln} JSON parse failed: {e}")
                bad += 1
                continue
            problems = validate_item(item)
            if problems:
                bad += 1
                print(f"  X {item.get('item_id','?')}: {'; '.join(problems)}")
            else:
                print(f"  OK {item.get('item_id')} ({len(item.get('rubric',[]))} scoring points)")
    print(f"\nTotal {total} items, {bad} with problems, {total-bad} valid.")
    return bad


def main():
    parser = argparse.ArgumentParser(description="Real-exam-item import pipeline")
    parser.add_argument("--validate", action="store_true", help="Validate the existing item bank format")
    args = parser.parse_args()

    if args.validate:
        sys.exit(1 if validate_bank() else 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
