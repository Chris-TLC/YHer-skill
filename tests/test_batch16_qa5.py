#!/usr/bin/env python3
"""Focused tests for Batch16 QA-5 census and packaging helpers."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_answer_source_candidates_resolve_group_sources(tmp_path: Path):
    from scripts.run_batch16_census import answer_source_candidates

    source = tmp_path / "analysis.docx"
    source.write_bytes(b"docx")
    group = {
        "unique_sources": [
            {
                "role": "question_source",
                "status": "ok",
                "path": str(tmp_path / "paper.docx"),
                "answer_marker_count": 0,
            },
            {
                "role": "analysis",
                "status": "ok",
                "path": str(source),
                "answer_marker_count": 12,
            },
        ]
    }

    rows = answer_source_candidates(group)

    assert len(rows) == 1
    assert rows[0]["role"] == "analysis"
    assert rows[0]["path_exists"] is True
    assert rows[0]["answer_marker_count"] == 12


def test_census_prefers_signed_variant_source_dead(tmp_path: Path):
    from scripts.run_batch16_census import classify_census_item

    item = {"item_id": "bad", "group_key": "g", "q_num": 3, "stem_blocks": [], "answer_blocks_effective": []}

    row = classify_census_item(
        item,
        r5_row={"r5_serve": False, "r5_block_reason": "exclusion:variant_bank_answer_misattributed"},
        group_sources={"unique_sources": []},
        ref_map_index={},
        signed_source_dead_ids={"bad"},
        usability_row={},
    )

    assert row["census_class"] == "source_dead"
    assert row["subclass"] == "variant_bank_answer_misattributed"
    assert row["reviewer"] == ""
    assert row["review_status"] == "pending_user_or_claude"
    assert "codex_" not in json.dumps(row, ensure_ascii=False).lower()


def test_census_known_route_uses_group_answer_source(tmp_path: Path):
    from scripts.run_batch16_census import classify_census_item

    source = tmp_path / "answer.docx"
    source.write_bytes(b"docx")
    item = {"item_id": "i", "group_key": "g", "q_num": 7, "stem_blocks": [], "answer_blocks_effective": []}
    group_sources = {
        "unique_sources": [
            {"role": "answer_key", "status": "ok", "path": str(source), "answer_marker_count": 20}
        ]
    }

    row = classify_census_item(
        item,
        r5_row={"r5_serve": False, "r5_block_reason": "exclusion:hollow_content"},
        group_sources=group_sources,
        ref_map_index={},
        signed_source_dead_ids=set(),
        usability_row={},
    )

    assert row["census_class"] == "repairable_known_route"
    assert "hollow_group_source" in row["subclass"]
    assert row["evidence"]["answer_source_candidates"][0]["path_exists"] is True


def test_census_known_route_records_media_ref_hits():
    from scripts.run_batch16_census import classify_census_item

    item = {
        "item_id": "i",
        "group_key": "g",
        "q_num": 1,
        "stem_blocks": [],
        "answer_blocks_effective": [{"para": [{"type": "figure", "media": "ans_abcd_image1.png"}]}],
    }

    row = classify_census_item(
        item,
        r5_row={"r5_serve": False, "r5_block_reason": "exclusion:hollow_content"},
        group_sources={"unique_sources": []},
        ref_map_index={("g", "ans_abcd_image1.png"): [{"line_no": 12, "asset_hash": "a" * 64}]},
        signed_source_dead_ids=set(),
        usability_row={},
    )

    assert row["census_class"] == "repairable_known_route"
    assert "media_linkable" in row["subclass"]
    assert row["evidence"]["ref_map_hits"][0]["line_no"] == 12


def _run_focused() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp)) if "tmp_path" in fn.__code__.co_varnames else fn()
            print(f"PASS {name}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{failures} FAILURES" if failures else "\nALL PASS")
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_focused())
