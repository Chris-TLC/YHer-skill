#!/usr/bin/env python3
"""Tests for semantic answer equivalence review helpers."""

from __future__ import annotations

from pathlib import Path
import sys

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))


def test_review_rows_uses_local_equivalence_without_model_call(tmp_path: Path):
    from scripts.review_answer_equivalence_openai import review_rows

    rows = [
        {
            "item_id": "i_local",
            "standard_answer": "冷却(降温)结晶",
            "model_answer": "冷却结晶（或降温结晶）",
        }
    ]

    results, summary = review_rows(rows, provider="gemini", model="gemini-3.1-pro-preview", env_path=tmp_path / ".env")

    assert summary["total"] == 1
    assert results[0]["decision"] == "equivalent"
    assert results[0]["reason"] == "local_exact_or_rule_match"


def test_review_rows_reports_missing_key_when_model_needed(tmp_path: Path):
    from scripts.review_answer_equivalence_openai import review_rows

    rows = [
        {
            "item_id": "i_needs_model",
            "standard_answer": "B",
            "model_answer": "C",
        }
    ]

    results, summary = review_rows(rows, provider="gemini", model="gemini-3.1-pro-preview", env_path=tmp_path / ".env")

    assert summary["error"] == "missing_api_key"
    assert results[0]["decision"] == "error"
    assert results[0]["error"] == "missing_api_key"


def test_review_rows_can_force_model_review_even_when_local_equivalent(tmp_path: Path):
    from scripts.review_answer_equivalence_openai import review_rows

    rows = [
        {
            "item_id": "i_force_model",
            "standard_answer": "硫酸镁熔点更高",
            "model_answer": "熔点更高",
        }
    ]

    results, summary = review_rows(
        rows,
        provider="gemini",
        model="gemini-3.1-pro-preview",
        env_path=tmp_path / ".env",
        force_model_review=True,
    )

    assert summary["error"] == "missing_api_key"
    assert results[0]["decision"] == "error"
    assert results[0]["error"] == "missing_api_key"


def test_answer_review_payload_can_upgrade_semantic_equivalence():
    from scripts.evaluate_visual_understanding import result_from_answer_review_payload

    eval_item = {
        "item_id": "i_semantic",
        "category": "chart_curve",
        "input_image_path": "/tmp/page.jpg",
        "stem": "解释题",
        "standard_answer": "H₂O₂ 的氧化性强于 O₂，亚硫酸氧化为硫酸 pH 显著下降",
        "model_answer": "H₂O₂ 和 O₂ 均能将弱酸 H₂SO₃ 氧化为强酸 H₂SO₄，使溶液 pH 下降；但在该条件下，H₂O₂ 氧化 H₂SO₃ 的反应速率远大于 O₂ 氧化 H₂SO₃ 的反应速率，因此通入 O₂ 后 pH 变化不明显，而加入 H₂O₂ 后 pH 迅速下降。",
        "visible_pass": True,
    }
    payload = {"decision": "equivalent", "confidence": 0.94, "reason": "same chemistry"}

    row = result_from_answer_review_payload(eval_item, payload, model="unit-model", raw_source="unit")

    assert row["answer_match"]
    assert row["understanding_pass"]
    assert row["profile_evidence_allowed"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    import tempfile

    for test in tests:
        try:
            if "tmp_path" in test.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    test(Path(d))
            else:
                test()
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 测试通过")
    sys.exit(0 if passed == len(tests) else 1)
