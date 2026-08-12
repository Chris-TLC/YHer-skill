#!/usr/bin/env python3
"""Focused tests for QA-3 cross-node broken ion merge candidates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _item(para):
    return {
        "item_id": "item1",
        "group_key": "group1",
        "q_num": 1,
        "section_num": 1,
        "stem_blocks": [{"para": para}],
        "answer_blocks_effective": [],
        "analysis_blocks": [],
    }


def test_positive_so4_candidate_merges_left_text_and_keeps_right_text():
    from scripts.apply_qa3_crossnode_ions import build_crossnode_rows

    items = [
        _item(
            [
                {"type": "text", "text": "Ba²⁺+SO"},
                {"type": "formula", "media": "so4.wmf"},
                {"type": "text", "text": "=BaSO4↓"},
            ]
        )
    ]
    media_lookup = {
        ("group1", "so4.wmf"): "965f5aab47569cef512cd4ed333dbcb9287298318407ff55545f3b9397b48622"
    }

    candidates, manual, stats = build_crossnode_rows(items, media_lookup)

    assert len(candidates) == 1
    assert manual == []
    assert stats["rev_fail"] == 0
    row = candidates[0]
    assert row["head"] == "SO"
    assert row["ion"] == "SO42-"
    assert row["ion_group"] == "whitelist"
    assert row["cluster"] == "₄²⁻"
    assert row["original_left"] == "Ba²⁺+SO"
    assert row["replaced_left"] == "Ba²⁺+SO₄²⁻"
    assert row["original_right"] == "=BaSO4↓"
    assert row["review_status"] == "pending_user_or_claude"
    assert row["reviewer"] == ""
    assert row["candidate_kind"] == "crossnode_ion_merge"


def test_idempotent_left_text_with_existing_unicode_charge_is_skipped():
    from scripts.apply_qa3_crossnode_ions import build_crossnode_rows

    items = [
        _item(
            [
                {"type": "text", "text": "Ba²⁺+SO₄²⁻"},
                {"type": "formula", "media": "so4.wmf"},
                {"type": "text", "text": "=BaSO4↓"},
            ]
        )
    ]
    media_lookup = {
        ("group1", "so4.wmf"): "965f5aab47569cef512cd4ed333dbcb9287298318407ff55545f3b9397b48622"
    }

    candidates, manual, stats = build_crossnode_rows(items, media_lookup)

    assert candidates == []
    assert manual == []
    assert stats["head_sites"] == 0


def test_arrow_cluster_is_manual_tail_not_candidate():
    from scripts.apply_qa3_crossnode_ions import build_crossnode_rows

    items = [
        _item(
            [
                {"type": "text", "text": "NO"},
                {"type": "formula", "media": "arrow.wmf"},
                {"type": "text", "text": "HNO3"},
            ]
        )
    ]
    media_lookup = {("group1", "arrow.wmf"): "d627e6db2ce61f0b3"}

    candidates, manual, stats = build_crossnode_rows(items, media_lookup)

    assert candidates == []
    assert len(manual) == 1
    assert manual[0]["reason"] == "arrow_excluded"
    assert stats["manual_arrow_excluded"] == 1


def test_non_whitelist_cluster_is_manual_tail():
    from scripts.apply_qa3_crossnode_ions import build_crossnode_rows

    items = [
        _item(
            [
                {"type": "text", "text": "(S"},
                {"type": "formula", "media": "s22.wmf"},
                {"type": "text", "text": "中的S—S键"},
            ]
        )
    ]
    media_lookup = {
        ("group1", "s22.wmf"): "fa32cc3f5934e9304e33273921c9c038a2ab8935034c1488b736a575dc26406f"
    }

    candidates, manual, stats = build_crossnode_rows(items, media_lookup)

    assert candidates == []
    assert len(manual) == 1
    assert manual[0]["reason"] == "not_whitelist"
    assert manual[0]["ion"] == "S22-"
    assert stats["manual_not_whitelist"] == 1


def test_extension_oxygen_anion_is_marked_separately():
    from scripts.apply_qa3_crossnode_ions import build_crossnode_rows

    items = [
        _item(
            [
                {"type": "text", "text": "3ClO⁻+I-=3Cl⁻+IO"},
                {"type": "formula", "media": "io3.wmf"},
                {"type": "text", "text": "；IO"},
            ]
        )
    ]
    media_lookup = {
        ("group1", "io3.wmf"): "e2c78343001a25fb71fc7f80ddc03a05dc789d0a9435fe24d78db9639d480e10"
    }

    candidates, manual, stats = build_crossnode_rows(items, media_lookup)

    assert len(candidates) == 1
    assert manual == []
    assert candidates[0]["ion"] == "IO3-"
    assert candidates[0]["ion_group"] == "extension"
    assert candidates[0]["replaced_left"].endswith("IO₃⁻")
    assert stats["candidate_extension"] == 1


def test_unwind_round_trips_target_and_candidate_left_text():
    from scripts.apply_qa3_crossnode_ions import make_target, unwind

    target = make_target("HPO", "4", "2-")
    new_left = ")+2c(" + target

    assert target == "HPO₄²⁻"
    assert unwind(target) == "HPO42-"
    assert unwind(new_left) == ")+2c(HPO42-"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"PASS {test.__name__}")
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
