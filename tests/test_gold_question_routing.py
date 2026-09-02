#!/usr/bin/env python3
"""Tests for model-generated gold diagnostic question candidate routing."""

from __future__ import annotations

from pathlib import Path
import sys

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))


def valid_candidate() -> dict:
    return {
        "gold_id": "solution_entry_001",
        "hard_hole": "solution_three_balances",
        "kg_node": "溶液三大守恒",
        "diagnostic_axis": "entry",
        "difficulty": "T1",
        "answer_type": "single_choice",
        "prompt": "某盐溶液中同时存在 Na+、HA-、A2-、H+、OH-。判断下列哪一种列式优先体现电荷守恒。",
        "options": {"A": "c(Na+)+c(H+)=c(HA-)+2c(A2-)+c(OH-)", "B": "c(Na+)=c(HA-)+c(A2-)", "C": "c(H+)=c(OH-)", "D": "c(HA-)=c(A2-)"},
        "standard_answer": "A",
        "rubric": [
            {
                "point_id": "charge",
                "desc": "能把所有带电粒子按电荷数列入等式",
                "must_have": True,
                "score": 1,
                "accept_patterns": ["阳离子电荷总量等于阴离子电荷总量"],
                "reject_patterns": ["漏掉H+或OH-"],
            }
        ],
        "misconceptions": [
            {
                "wrong_pattern": "只写物料守恒",
                "reveals": "分不清电荷守恒和物料守恒",
                "profile_update": {"axis": "审题入口", "direction": "weaken", "tag": "守恒类型识别弱"},
                "recommended_remediation": "先补守恒类型判断，再练列式。",
            }
        ],
        "profile_evidence_rule": {
            "can_update_profile": True,
            "max_weight": "medium",
            "mastery_signal": "选 A 且解释电荷总量相等。",
            "weakness_signal": "选择物料守恒或漏写 H+/OH-。",
        },
        "verification_use": ["diagnosis", "post_video_verification"],
        "source_type": "model_candidate",
        "review_status": "silver_candidate",
        "risk_notes": [],
    }


def test_valid_candidate_routes_to_approved():
    from scripts.route_gold_question_candidates import route_candidates

    routed, summary = route_candidates([valid_candidate()])

    assert summary["approved"] == 1
    assert routed["approved"][0]["approved_label"] == "gold_v1_model_reviewed"
    assert routed["approved"][0]["production_profile_evidence_allowed"] is False


def test_visual_dependency_routes_to_reject():
    from scripts.route_gold_question_candidates import route_candidates

    row = valid_candidate()
    row["prompt"] = "如图所示，判断该装置中应使用哪种守恒。"

    routed, summary = route_candidates([row])

    assert summary["reject"] == 1
    assert "prompt_has_external_visual_dependency" in routed["reject"][0]["route_blockers"]


def test_schema_gap_routes_to_revise_when_recoverable():
    from scripts.route_gold_question_candidates import route_candidates

    row = valid_candidate()
    row["standard_answer"] = "AB"

    routed, summary = route_candidates([row])

    assert summary["revise"] == 1
    assert "single_choice_standard_answer_must_be_one_option" in routed["revise"][0]["route_blockers"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 测试通过")
    sys.exit(0 if passed == len(tests) else 1)
