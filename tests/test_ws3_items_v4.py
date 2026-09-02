#!/usr/bin/env python3
"""Tests for WS3 Schema v4 item landing and alignment helpers."""

from __future__ import annotations

import json
from pathlib import Path


def text_para(text: str) -> dict:
    return {"para": [{"type": "text", "text": text}]}


def test_pool_tags_exclude_answer_type_mismatch_and_preserve_legacy_tag():
    from scripts.build_ws3_items_v4 import classify_pool, effective_answer_blocks

    question = {
        "group_key": "2012年高考化学试卷上海",
        "quality_flags": ["answer_type_mismatch"],
        "answer_blocks": [text_para("【答案】不是ABCD")],
    }

    pool = classify_pool(question)

    assert pool["pool"] == "excluded_answerless"
    assert pool["service_eligible"] is False
    assert pool["pool_tags"] == ["legacy", "excluded_answerless"]
    assert effective_answer_blocks(question) == []


def test_alignment_thresholds_do_not_silently_merge_low_confidence_matches():
    from scripts.build_ws3_items_v4 import align_question, build_alignment_index

    old_items = [
        {
            "item_id": "old-auto",
            "stem": "下列关于氯化钠性质的说法正确的是A溶于水B能燃烧C有毒D不导电",
            "kg_nodes": ["离子化合物"],
        },
        {
            "item_id": "old-review",
            "stem": "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未",
            "kg_nodes": ["部分重叠"],
        },
    ]
    index = build_alignment_index(old_items)

    auto = align_question(
        {"stem_blocks": [text_para("1. 下列关于氯化钠性质的说法正确的是 A.溶于水 B.能燃烧 C.有毒 D.不导电")]},
        index,
    )
    review = align_question(
        {"stem_blocks": [text_para("甲乙丙丁戊己庚辛壬癸甲甲甲甲")]},
        index,
    )
    new = align_question({"stem_blocks": [text_para("完全不同的新题干")]}, index)

    assert auto["status"] == "auto_inherited"
    assert auto["aligned_item_id"] == "old-auto"
    assert review["status"] == "needs_review"
    assert review["aligned_item_id"] is None
    assert new["status"] == "new"
    assert new["aligned_item_id"] is None


def test_blocks_text_extracts_text_from_structured_table_cells():
    from scripts.build_ws3_items_v4 import blocks_text

    value = [
        {
            "type": "table",
            "rows": [[
                [
                    {"type": "text", "text": "Ka="},
                    {"type": "math_omml", "omml": "<m:oMath/>", "latex": "10^{-7}"},
                ],
                "HClO",
            ]],
        }
    ]

    text = blocks_text(value, include_media_placeholder=False)

    assert "Ka=" in text
    assert "10^{-7}" in text
    assert "HClO" in text
    assert "'type'" not in text
    assert "omml" not in text


def test_build_ws3_outputs_items_report_and_review_queue(tmp_path: Path):
    from scripts.build_ws3_items_v4 import build_ws3_outputs

    ws1 = tmp_path / "questions_deduped.jsonl"
    old = tmp_path / "chemistry_v3.jsonl"
    gold = tmp_path / "golden_round2"
    out = tmp_path / "out"
    gold_case = gold / "round2_001_equation"
    gold_case.mkdir(parents=True)
    gold_review_case = gold / "round2_002_device"
    gold_review_case.mkdir(parents=True)

    questions = [
        {
            "question_id": "q-auto",
            "group_key": "2024届上海市测试卷",
            "stem_blocks": [text_para("2. 碳酸钙和盐酸反应生成二氧化碳，下列说法正确的是A.属于复分解反应")],
            "answer_blocks": [text_para("【答案】A")],
            "quality_flags": [],
        },
        {
            "question_id": "q-review",
            "group_key": "2024届上海市测试卷",
            "q_num": 2,
            "stem_blocks": [text_para("实验室制备氯气需要加热并吸收尾气")],
            "answer_blocks": [text_para("【答案】加热")],
            "quality_flags": [],
        },
    ]
    old_items = [
        {
            "item_id": "old-auto",
            "source": "2024届上海市测试卷（解析版）.docx",
            "stem": "碳酸钙和盐酸反应生成二氧化碳，下列说法正确的是A属于复分解反应",
            "kg_nodes": ["离子反应"],
            "rubric": [{"point_id": "ans"}],
            "standard_solution": {"standard_answer": "A"},
            "verification_status": "passed",
        },
        {
            "item_id": "old-review",
            "source": "2024届上海市测试卷（解析版）.docx",
            "stem": "实验室用二氧化锰和浓盐酸制备氯气时需要加热并使用氢氧化钠溶液吸收尾气",
            "kg_nodes": ["氯气"],
        },
    ]
    ws1.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n")
    old.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in old_items) + "\n")
    (gold_case / "question.json").write_text(
        json.dumps(
            {
                "question_id": "q-auto",
                "golden_candidate": {"candidate_id": "round2_001_equation", "set_role": "formal"},
            },
            ensure_ascii=False,
        )
    )
    (gold_review_case / "question.json").write_text(
        json.dumps(
            {
                "question_id": "q-review",
                "golden_candidate": {"candidate_id": "round2_002_device", "set_role": "formal"},
            },
            ensure_ascii=False,
        )
    )

    summary = build_ws3_outputs(ws1, old, gold, out)

    items = [json.loads(line) for line in (out / "items_v4.jsonl").read_text().splitlines()]
    queue = [json.loads(line) for line in (out / "alignment_review_queue.jsonl").read_text().splitlines()]
    gold_audit = [json.loads(line) for line in (out / "gold_alignment_audit.jsonl").read_text().splitlines()]
    gold_review = [json.loads(line) for line in (out / "gold_alignment_review_candidates.jsonl").read_text().splitlines()]
    report = (out / "alignment_report.md").read_text()

    assert summary["total_items"] == 2
    assert items[0]["alignment"]["aligned_item_id"] == "old-auto"
    assert items[0]["kg_nodes"] == ["离子反应"]
    assert len(queue) == 1
    assert queue[0]["question_id"] == "q-review"
    assert gold_audit[0] == {
        "candidate_id": "round2_001_equation",
        "question_id": "q-auto",
        "alignment_status": "auto_inherited",
        "gold_alignment_bucket": "strict_text_auto",
        "aligned_item_id": "old-auto",
        "best_candidate_item_id": "old-auto",
        "pool": "main",
        "set_role": "formal",
        "similarity": items[0]["alignment"]["similarity"],
        "source_rank_match": None,
        "source_rank": 1,
        "q_num": None,
        "override_candidate": False,
        "needs_manual_review": False,
    }
    assert gold_audit[1] == {
        "candidate_id": "round2_002_device",
        "question_id": "q-review",
        "alignment_status": "needs_review",
        "gold_alignment_bucket": "source_rank_override_candidate",
        "aligned_item_id": None,
        "best_candidate_item_id": "old-review",
        "pool": "main",
        "set_role": "formal",
        "similarity": items[1]["alignment"]["similarity"],
        "source_rank_match": True,
        "source_rank": 2,
        "q_num": 2,
        "override_candidate": True,
        "needs_manual_review": True,
    }
    assert gold_review == [
        {
            "candidate_id": "round2_002_device",
            "question_id": "q-review",
            "gold_alignment_bucket": "source_rank_override_candidate",
            "best_candidate_item_id": "old-review",
            "similarity": items[1]["alignment"]["similarity"],
            "q_num": 2,
            "source_rank": 2,
            "source_rank_match": True,
            "new_stem": "实验室制备氯气需要加热并吸收尾气",
            "candidate_stem": "实验室用二氧化锰和浓盐酸制备氯气时需要加热并使用氢氧化钠溶液吸收尾气",
            "recommended_action": "manual_override_if_reviewer_confirms",
        }
    ]
    assert "Golden Formal Set" in report
    assert "2 / 2" in report
    assert "Source-rank override candidates: 1 / 2" in report


def test_manual_alignment_override_is_explicit_and_audited(tmp_path: Path):
    from scripts.build_ws3_items_v4 import build_ws3_outputs

    ws1 = tmp_path / "questions_deduped.jsonl"
    old = tmp_path / "chemistry_v3.jsonl"
    gold = tmp_path / "golden_round2"
    out = tmp_path / "out"
    overrides = tmp_path / "manual_overrides.jsonl"
    gold_case = gold / "round2_001_device"
    gold_case.mkdir(parents=True)

    ws1.write_text(
        json.dumps(
            {
                "question_id": "q-manual",
                "group_key": "2024届上海市测试卷",
                "q_num": 7,
                "stem_blocks": [text_para("实验室制备氯气需要加热并吸收尾气")],
                "answer_blocks": [text_para("【答案】加热")],
                "quality_flags": [],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    old.write_text(
        json.dumps(
            {
                "item_id": "old-manual",
                "source": "2024届上海市测试卷（解析版）.docx",
                "stem": "实验室用二氧化锰和浓盐酸制备氯气时需要加热并使用氢氧化钠溶液吸收尾气",
                "kg_nodes": ["氯气"],
                "rubric": [{"point_id": "ans"}],
                "standard_solution": {"standard_answer": "加热"},
                "verification_status": "passed",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    (gold_case / "question.json").write_text(
        json.dumps(
            {
                "question_id": "q-manual",
                "golden_candidate": {"candidate_id": "round2_001_device", "set_role": "formal"},
            },
            ensure_ascii=False,
        )
    )
    overrides.write_text(
        json.dumps(
            {
                "question_id": "q-manual",
                "aligned_item_id": "old-manual",
                "decision": "manual_inherit",
                "reviewer": "codex_batch5_manual_audit",
                "evidence": "same source question confirmed; new stem lost formulas/figure text only",
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    summary = build_ws3_outputs(ws1, old, gold, out, overrides)

    items = [json.loads(line) for line in (out / "items_v4.jsonl").read_text().splitlines()]
    queue = [json.loads(line) for line in (out / "alignment_review_queue.jsonl").read_text().splitlines()]
    gold_audit = [json.loads(line) for line in (out / "gold_alignment_audit.jsonl").read_text().splitlines()]
    gold_review = [json.loads(line) for line in (out / "gold_alignment_review_candidates.jsonl").read_text().splitlines()]
    report = (out / "alignment_report.md").read_text()

    assert summary["alignment_counts"] == {"manual_inherited": 1}
    assert queue == []
    assert gold_review == []
    assert items[0]["alignment"]["status"] == "manual_inherited"
    assert items[0]["alignment"]["aligned_item_id"] == "old-manual"
    assert items[0]["alignment"]["manual_override"] == {
        "decision": "manual_inherit",
        "reviewer": "codex_batch5_manual_audit",
        "evidence": "same source question confirmed; new stem lost formulas/figure text only",
    }
    assert items[0]["kg_nodes"] == ["氯气"]
    assert gold_audit == [
        {
            "candidate_id": "round2_001_device",
            "question_id": "q-manual",
            "alignment_status": "manual_inherited",
            "gold_alignment_bucket": "manual_override_inherited",
            "aligned_item_id": "old-manual",
            "best_candidate_item_id": "old-manual",
            "pool": "main",
            "set_role": "formal",
            "similarity": items[0]["alignment"]["similarity"],
            "source_rank_match": False,
            "source_rank": 1,
            "q_num": 7,
            "override_candidate": False,
            "manual_override": True,
            "needs_manual_review": False,
        }
    ]
    assert "Manual overrides inherited: 1 / 1" in report


def test_manual_gold_resolution_records_do_not_inherit_without_kg_inheritance(tmp_path: Path):
    from scripts.build_ws3_items_v4 import build_ws3_outputs

    ws1 = tmp_path / "questions_deduped.jsonl"
    old = tmp_path / "chemistry_v3.jsonl"
    gold = tmp_path / "golden_round2"
    out = tmp_path / "out"
    resolutions = tmp_path / "manual_resolutions.jsonl"
    gold_case = gold / "round2_001_crystal"
    gold_case.mkdir(parents=True)

    ws1.write_text(
        json.dumps(
            {
                "question_id": "q-new-gold",
                "group_key": "2024届上海市测试卷",
                "q_num": 4,
                "stem_blocks": [text_para("全库没有精确旧题的金标题")],
                "answer_blocks": [text_para("【答案】B")],
                "quality_flags": [],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    old.write_text(
        json.dumps(
            {
                "item_id": "old-different",
                "source": "2024届上海市测试卷（解析版）.docx",
                "stem": "同源但是完全不同的旧题",
                "kg_nodes": ["不应继承"],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    (gold_case / "question.json").write_text(
        json.dumps(
            {
                "question_id": "q-new-gold",
                "golden_candidate": {"candidate_id": "round2_001_crystal", "set_role": "formal"},
            },
            ensure_ascii=False,
        )
    )
    resolutions.write_text(
        json.dumps(
            {
                "question_id": "q-new-gold",
                "decision": "manual_do_not_inherit",
                "reason": "no_safe_v3_match",
                "reviewer": "codex_batch5_manual_audit",
                "evidence": "same-source and global candidates are different questions",
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    summary = build_ws3_outputs(ws1, old, gold, out, None, resolutions)

    items = [json.loads(line) for line in (out / "items_v4.jsonl").read_text().splitlines()]
    queue = [json.loads(line) for line in (out / "alignment_review_queue.jsonl").read_text().splitlines()]
    gold_audit = [json.loads(line) for line in (out / "gold_alignment_audit.jsonl").read_text().splitlines()]
    gold_review = [json.loads(line) for line in (out / "gold_alignment_review_candidates.jsonl").read_text().splitlines()]
    report = (out / "alignment_report.md").read_text()

    assert summary["alignment_counts"] == {"manual_do_not_inherit": 1}
    assert queue == []
    assert gold_review == []
    assert "kg_nodes" not in items[0]
    assert items[0]["alignment"]["status"] == "manual_do_not_inherit"
    assert items[0]["alignment"]["aligned_item_id"] is None
    assert items[0]["alignment"]["manual_resolution"] == {
        "decision": "manual_do_not_inherit",
        "reason": "no_safe_v3_match",
        "reviewer": "codex_batch5_manual_audit",
        "evidence": "same-source and global candidates are different questions",
    }
    assert gold_audit == [
        {
            "candidate_id": "round2_001_crystal",
            "question_id": "q-new-gold",
            "alignment_status": "manual_do_not_inherit",
            "gold_alignment_bucket": "manual_do_not_inherit",
            "aligned_item_id": None,
            "best_candidate_item_id": items[0]["alignment"]["best_candidate_item_id"],
            "pool": "main",
            "set_role": "formal",
            "similarity": items[0]["alignment"]["similarity"],
            "source_rank_match": None,
            "source_rank": None,
            "q_num": 4,
            "override_candidate": False,
            "manual_resolution": True,
            "manual_resolution_reason": "no_safe_v3_match",
            "needs_manual_review": False,
        }
    ]
    assert "Gold alignment decisions resolved: 1 / 1" in report
    assert "Manual do-not-inherit decisions: 1 / 1" in report
    assert "Unresolved manual-review gold rows: 0 / 1" in report
