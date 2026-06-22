#!/usr/bin/env python3
"""阶段一私教编排器的轻量测试。"""

import os
import sys
import unittest

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SKILL_DIR)

from core.private_tutor import ChemistryPrivateTutor, StudentProfile
from core.tutor_prompts import (
    STAGE1_TUTOR_SYSTEM_PROMPT,
    build_diagnostic_review_prompt,
    build_progressive_diagnosis_prompt,
    build_session_report_prompt,
    build_task_execution_prompt,
    extract_tagged_json,
    strip_tagged_json,
)


class PrivateTutorTest(unittest.TestCase):
    def test_prepare_equilibrium_session_without_retriever(self):
        tutor = ChemistryPrivateTutor()
        profile = StudentProfile(
            grade="高二",
            region="全国卷",
            goal="我今天下午想补化学平衡，尤其是平衡移动和转化率",
            time_budget_min=180,
            current_score=55,
            target_score=80,
        )

        session = tutor.prepare_session(profile)

        self.assertEqual(session.subject, "chemistry")
        self.assertEqual(session.topic_id, "chemical_equilibrium")
        self.assertEqual(session.topic_name, "化学平衡")
        self.assertGreaterEqual(len(session.diagnostic_questions), 5)
        self.assertEqual(len(session.tasks), 6)
        self.assertEqual(sum(t.duration_min for t in session.tasks), 180)
        self.assertEqual(session.current_task_id, "T1")
        self.assertIn("large_score_gap", session.risk_flags)

    def test_short_session_still_has_valid_task_durations(self):
        tutor = ChemistryPrivateTutor()
        profile = StudentProfile(goal="氧化还原配平老错", time_budget_min=45)

        session = tutor.prepare_session(profile)

        self.assertEqual(session.topic_id, "redox")
        self.assertEqual(sum(t.duration_min for t in session.tasks), 45)
        self.assertTrue(all(t.duration_min >= 5 for t in session.tasks))

    def test_no_llm_answer_snapshot_finds_empty_answers(self):
        tutor = ChemistryPrivateTutor()
        session = tutor.prepare_session(StudentProfile(goal="电化学正负极搞不清"))
        answers = {q["id"]: "" for q in session.diagnostic_questions}

        snapshot = tutor.estimate_answer_snapshot(session, answers)

        self.assertEqual(snapshot["confidence"], "low_without_llm")
        self.assertTrue(snapshot["weakest_axes"])
        self.assertTrue(snapshot["evidence"])

    def test_prompt_builders_include_core_constraints(self):
        tutor = ChemistryPrivateTutor()
        session = tutor.prepare_session(StudentProfile(goal="工艺流程产率怎么算"))
        answers = {q["id"]: "不会，看到流程题就不知道先看哪里" for q in session.diagnostic_questions}

        review_prompt = build_diagnostic_review_prompt(session, answers)
        progressive_prompt = build_progressive_diagnosis_prompt(
            session,
            session.diagnostic_questions[0],
            "我不知道先看哪里",
        )
        task_prompt = build_task_execution_prompt(session, "T1", "我先找原料可以吗")
        report_prompt = build_session_report_prompt(session, [{"role": "student", "content": "我会了"}])

        self.assertIn("真实能力状态", review_prompt)
        self.assertIn("DIAGNOSIS_JSON", progressive_prompt)
        self.assertIn("CURRENT_TASK", task_prompt)
        self.assertIn("CONTROL_JSON", task_prompt)
        self.assertIn("学生画像 JSON", report_prompt)
        self.assertIn("不要机械套固定格式", STAGE1_TUTOR_SYSTEM_PROMPT)

    def test_tagged_json_helpers(self):
        text = """可见内容

[CONTROL_JSON]
{"current_task":"T1","mastery":0.8,"decision":"advance","next_task":"T2"}
[/CONTROL_JSON]
"""
        parsed = extract_tagged_json(text, "CONTROL_JSON")
        visible = strip_tagged_json(text, "CONTROL_JSON")

        self.assertEqual(parsed["decision"], "advance")
        self.assertNotIn("CONTROL_JSON", visible)


if __name__ == "__main__":
    unittest.main()
