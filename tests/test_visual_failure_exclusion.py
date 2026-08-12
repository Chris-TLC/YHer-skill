#!/usr/bin/env python3
"""Tests for visual failure exclusion and audit queue generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def failure(
    item_id: str,
    category: str,
    primary: str,
    error_types: list[str],
    confidence: float = 0.8,
) -> dict:
    return {
        "item_id": item_id,
        "category": category,
        "primary_failure_category": primary,
        "error_types": error_types,
        "visible_pass": "question_not_visible" not in error_types,
        "answer_match": "answer_mismatch" not in error_types,
        "understanding_pass": False,
        "confidence": confidence,
        "crop_ready_for_promotion": True,
    }


def test_build_failure_exclusion_excludes_high_confidence_and_repeated_failures(tmp_path: Path):
    from scripts.build_visual_failure_exclusion import build_failure_exclusion

    batch_a = tmp_path / "batch_008_failure_report.jsonl"
    batch_b = tmp_path / "batch_009_failure_report.jsonl"
    write_jsonl(
        batch_a,
        [
            failure("high_conf", "chart_curve", "high_confidence_error", ["answer_mismatch", "high_confidence_error"], 1.0),
            failure("repeat", "crystal_cell", "has_missing_or_uncertain", ["has_missing_or_uncertain"], 0.9),
            failure("transient_provider", "other", "provider_error", ["provider_error"], 0.0),
        ],
    )
    write_jsonl(
        batch_b,
        [
            failure("repeat", "crystal_cell", "question_not_visible_or_weak_evidence", ["question_not_visible_or_weak_evidence"], 0.85),
            failure("single_low_conf", "organic_structure", "answer_mismatch", ["answer_mismatch"], 0.3),
        ],
    )

    exclusion_rows, audit_rows, summary = build_failure_exclusion([batch_a, batch_b])

    assert [row["item_id"] for row in exclusion_rows] == ["high_conf", "repeat"]
    assert exclusion_rows[0]["exclude_reason"] == "high_confidence_error"
    assert exclusion_rows[1]["exclude_reason"] == "repeated_non_provider_failure"
    assert [row["item_id"] for row in audit_rows] == ["high_conf", "repeat"]
    assert audit_rows[0]["recommended_action"] == "answer_or_standard_key_audit"
    assert audit_rows[1]["recommended_action"] == "manual_visual_evidence_audit"
    assert summary["failure_records"] == 5
    assert summary["unique_failure_items"] == 4
    assert summary["exclude_items"] == 2
    assert summary["ordinary_batch_exclusion_policy"] == "high_confidence_or_repeated_non_provider_failure"


def test_build_failure_exclusion_does_not_exclude_transient_provider_only_failures(tmp_path: Path):
    from scripts.build_visual_failure_exclusion import build_failure_exclusion

    batch = tmp_path / "batch_010_failure_report.jsonl"
    write_jsonl(
        batch,
        [
            failure("provider_only", "other", "provider_error", ["provider_error"], 0.0),
            failure("low_conf_once", "chart_curve", "answer_mismatch", ["answer_mismatch"], 0.2),
        ],
    )

    exclusion_rows, audit_rows, summary = build_failure_exclusion([batch])

    assert exclusion_rows == []
    assert audit_rows == []
    assert summary["exclude_items"] == 0
    assert summary["provider_only_items"] == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    import tempfile

    for test in tests:
        try:
            with tempfile.TemporaryDirectory() as d:
                test(Path(d))
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 测试通过")
    sys.exit(0 if passed == len(tests) else 1)
