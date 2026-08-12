#!/usr/bin/env python3
"""Focused contract tests for Batch 13 QA-3 crop rescue."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_select_source_respects_ans_prefix_and_exam_fallback():
    from scripts.run_batch13_qa3 import select_source_for_media

    sources = {
        "unique_sources": [
            {
                "role": "analysis",
                "status": "ok",
                "name": "某卷（考试版）.docx",
                "path": __file__,
                "answer_marker_count": 0,
            },
            {
                "role": "analysis",
                "status": "ok",
                "name": "某卷（解析版）.docx",
                "path": __file__,
                "answer_marker_count": 12,
            },
        ]
    }

    question = select_source_for_media(sources, "g", "image1.wmf")
    answer = select_source_for_media(sources, "g", "ans_abcd1234_image1.wmf")

    assert question.path == Path(__file__).resolve()
    assert "考试版" in question.source["name"]
    assert answer.path == Path(__file__).resolve()
    assert "解析版" in answer.source["name"]


def test_context_extraction_uses_left_and_right_text_anchors():
    from scripts.run_batch13_qa3 import iter_media_contexts_for_item

    item = {
        "item_id": "i1",
        "group_key": "g",
        "q_num": 7,
        "stem_blocks": [
            {
                "para": [
                    {"type": "text", "text": "如图所示装置中，"},
                    {"type": "figure", "media": "image1.wmf"},
                    {"type": "text", "text": "下列说法正确的是"},
                ]
            }
        ],
    }

    ctx = list(iter_media_contexts_for_item(item, "stem", "stem_blocks"))[0]

    assert ctx["media"] == "image1.wmf"
    assert ctx["block_type"] == "figure"
    assert ctx["anchor_before"] == "如图所示装置中，"
    assert ctx["anchor_after"] == "下列说法正确的是"
    assert "7" in ctx["anchor_text"]


def test_match_pages_is_conservative_for_ambiguous_hits():
    from scripts.run_batch13_qa3 import match_pages

    contexts = [{"anchor_before": "如图所示装置中", "anchor_after": "下列说法正确的是", "stem_head": ""}]
    pages = {
        1: "如图所示装置中 下列说法正确的是",
        2: "如图所示装置中 下列说法正确的是",
    }

    result = match_pages(pages, contexts)

    assert result["status"] == "page_ambiguous"


def test_match_pages_accepts_unique_combined_anchor():
    from scripts.run_batch13_qa3 import match_pages

    contexts = [{"anchor_before": "如图所示装置中", "anchor_after": "下列说法正确的是", "stem_head": ""}]
    pages = {1: "无关文字", 2: "如图所示装置中 下列说法正确的是"}

    result = match_pages(pages, contexts)

    assert result["status"] == "ok"
    assert result["page"] == 2


def test_bbox_normalization_and_plausibility():
    from scripts.run_batch13_qa3 import bbox_plausible, normalize_bbox

    bbox = normalize_bbox([100, 100, 300, 250], (2000, 3000), "normalized_1000")

    assert bbox == (200, 300, 600, 750)
    ok, reason, frac = bbox_plausible(bbox, (2000, 3000))
    assert ok is True
    assert reason == ""
    assert 0 < frac < 1


def test_pending_rows_keep_reviewer_empty_and_no_codex():
    from scripts.run_batch13_qa3 import pending_row

    row = pending_row({"group_key": "g", "media": "image1.wmf"}, status="kept")

    assert row["review_status"] == "pending_user_or_claude"
    assert row["reviewer"] == ""
    assert "codex_" not in json.dumps(row, ensure_ascii=False).lower()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            with tempfile.TemporaryDirectory() as _:
                test()
            passed += 1
            print(f"PASS {test.__name__}")
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
