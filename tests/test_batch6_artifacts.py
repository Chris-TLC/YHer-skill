#!/usr/bin/env python3
"""Tests for Batch 6 artifact builder helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_service_omml_groups_are_selected_for_rerun():
    from scripts.build_batch6_artifacts import service_omml_group_keys

    rows = [
        {"group_key": "service-a", "scope": "service"},
        {"group_key": "blocked-b", "scope": "main_blocked"},
        {"group_key": "legacy-c", "scope": "other_pool"},
    ]

    assert service_omml_group_keys(rows) == {"service-a"}


def test_omml_after_comparison_marks_unrerun_other_pool_as_known_debt():
    from scripts.build_batch6_artifacts import build_omml_after_comparison_rows

    before_rows = [
        {
            "item_id": "old-service",
            "group_key": "g1",
            "section_num": 1,
            "q_num": 2,
            "scope": "service",
            "block_type": "table",
        },
        {
            "item_id": "old-legacy",
            "group_key": "g2",
            "section_num": 1,
            "q_num": 3,
            "scope": "other_pool",
            "block_type": "table",
        },
    ]
    candidate_rows = [
        {
            "item_id": "new-service",
            "candidate_origin": "batch6_ws1_bounded_rerun",
            "group_key": "g1",
            "section_num": 1,
            "q_num": 2,
            "stem_blocks": [{"para": [{"type": "text", "text": "fixed"}]}],
        }
    ]

    rows = build_omml_after_comparison_rows(before_rows, candidate_rows, rerun_group_keys={"g1"})

    by_old = {row["old_item_id"]: row for row in rows}
    assert by_old["old-service"]["after_literal_rows"] == 0
    assert by_old["old-service"]["rerun_status"] == "rerun_matched"
    assert by_old["old-legacy"]["rerun_status"] == "not_rerun_known_residual_debt"


def test_rerun_diff_preserves_non_targets_in_final_candidate_package():
    from scripts.build_batch6_artifacts import build_rerun_diff_outputs

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rerun_root = root / "fixed_groups"
        rerun_root.mkdir()
        questions = [
            {
                "question_id": "new-target",
                "group_key": "g",
                "section_num": 1,
                "q_num": 1,
                "stem_blocks": [{"para": [{"type": "text", "text": "fixed target"}]}],
                "answer_blocks": [],
                "analysis_blocks": [],
            },
            {
                "question_id": "new-non-target",
                "group_key": "g",
                "section_num": 1,
                "q_num": 2,
                "stem_blocks": [{"para": [{"type": "text", "text": "changed by rerun"}]}],
                "answer_blocks": [],
                "analysis_blocks": [],
            },
        ]
        (rerun_root / "questions_deduped.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions),
            encoding="utf-8",
        )
        items = [
            {
                "item_id": "old-target",
                "source_question_id": "old-target",
                "group_key": "g",
                "section_num": 1,
                "q_num": 1,
                "stem_text": "old target",
                "stem_blocks": [{"para": [{"type": "text", "text": "old target"}]}],
                "answer_blocks_effective": [],
                "analysis_blocks": [],
            },
            {
                "item_id": "old-non-target",
                "source_question_id": "old-non-target",
                "group_key": "g",
                "section_num": 1,
                "q_num": 2,
                "stem_text": "official non-target",
                "stem_blocks": [{"para": [{"type": "text", "text": "official non-target"}]}],
                "answer_blocks_effective": [],
                "analysis_blocks": [],
            },
        ]
        summary = build_rerun_diff_outputs(
            root,
            rerun_root,
            items,
            {"old-target": {"item_id": "old-target", "problem_type": "stem_is_analysis_text", "group_key": "g"}},
        )

        candidates = {row["item_id"]: row for row in read_jsonl(root / "ws1_segmentation/fixed_candidate_items.jsonl")}
        raw_diff = read_jsonl(root / "ws1_segmentation/raw_collateral_diff.jsonl")
        final_diff = read_jsonl(root / "ws1_segmentation/collateral_diff.jsonl")

        assert candidates["new-target"]["candidate_origin"] == "batch6_ws1_bounded_rerun"
        assert candidates["old-non-target"]["candidate_origin"] == "batch6_official_preserved_for_collateral_safety"
        assert candidates["old-non-target"]["stem_text"] == "official non-target"
        assert any(row["old_item_id"] == "old-non-target" and row["content_changed"] for row in raw_diff)
        assert any(row["old_item_id"] == "old-non-target" and not row["content_changed"] for row in final_diff)
        assert summary["raw_non_target_content_changes"] == 1
        assert summary["final_non_target_content_changes"] == 0
