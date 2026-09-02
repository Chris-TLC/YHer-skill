#!/usr/bin/env python3
"""Focused contract tests for Batch 10 QA-1 candidate generation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_failure_prompt_gate_rejects_latex_and_transcript_text():
    from scripts.run_batch10_qa1 import has_failure_prompt_text, mark_bad_prompt_rows

    rows = [
        {"asset_hash": "ok", "latex": r"\ce{SO4^{2-}}", "latex_status": "passed"},
        {"asset_hash": "bad_latex", "latex": "无法识别图片中的公式", "latex_status": "passed"},
        {"asset_hash": "bad_transcript", "summary": "sorry, cannot read this image", "pool": "ai_seed"},
        {
            "asset_hash": "bad_nested_run",
            "latex": r"\ce{NaCl}",
            "runs": {
                "temperature_0_1": {
                    "raw": {
                        "uncertain": ["图片内容模糊"],
                    }
                }
            },
        },
    ]

    assert has_failure_prompt_text(rows[1]) is True
    marked, rejected = mark_bad_prompt_rows(rows, "unit")

    assert [row["asset_hash"] for row in marked] == ["ok"]
    assert {row["asset_hash"] for row in rejected} == {"bad_latex", "bad_transcript", "bad_nested_run"}
    assert all(row.get("pool") == "manual_queue" for row in rejected)
    assert all(row.get("reviewer") == "" for row in rejected)
    assert all(row.get("review_status") == "pending_user_or_claude" for row in rejected)


def test_answer_zone_asset_rows_are_classified_from_blocks():
    from scripts.run_batch10_qa1 import build_answer_zone_asset_rows

    media_rows = [
        {"group_key": "g", "media": "a.wmf", "asset_hash": "h1", "in_ws2_manifest": False, "zones": ["answer"]},
        {"group_key": "g", "media": "b.png", "asset_hash": "h2", "in_ws2_manifest": False, "zones": ["analysis"]},
    ]
    items = [
        {
            "item_id": "item1",
            "group_key": "g",
            "q_num": 1,
            "answer_blocks_effective": [{"para": [{"type": "formula", "media": "a.wmf"}]}],
            "analysis_blocks": [{"para": [{"type": "figure", "media": "b.png"}]}],
        }
    ]
    group_dirs = {"g": Path("/tmp/gdir")}

    rows = build_answer_zone_asset_rows(media_rows, items, group_dirs)

    by_hash = {row["asset_hash"]: row for row in rows}
    assert by_hash["h1"]["asset_class"] == "formula_image"
    assert by_hash["h2"]["asset_class"] == "illustration"
    assert by_hash["h1"]["sample_refs"][0]["block_type"] == "formula"
    assert by_hash["h2"]["zones"] == ["analysis"]


def test_text_ion_candidates_are_pending_and_rule_based():
    from scripts.run_batch10_qa1 import scan_text_ion_candidates

    items = [
        {
            "item_id": "i1",
            "group_key": "g",
            "q_num": 1,
            "stem_blocks": [{"para": [{"type": "text", "text": "溶液中含NH+4和SO42-，不是0.5B。"}]}],
            "answer_blocks_effective": [],
            "analysis_blocks": [],
        }
    ]

    candidates = scan_text_ion_candidates(items)

    assert {row["rule_id"] for row in candidates} >= {"ION_NH_PLUS_4", "ION_SO4_2_MINUS"}
    assert all(row["schema_version"] == "qa1_candidate_v1" for row in candidates)
    assert all(row["review_status"] == "pending_user_or_claude" for row in candidates)
    assert all(row["reviewer"] == "" for row in candidates)
    assert all(row["matched_text"] != "0.5B" for row in candidates)


def test_option_split_rejects_decimal_and_letter_prefix_boundaries():
    from scripts.run_batch10_qa1 import split_option_text

    text = "A．酸性B．铁粉C．0.5B不是拆点D．结束"
    pieces = split_option_text(text)

    assert pieces == ["A．酸性", "B．铁粉", "C．0.5B不是拆点", "D．结束"]


def test_latex_form_target_selector_catches_frac_and_isolated_fragments():
    from scripts.run_batch10_qa1 import select_latex_form_targets

    rows = [
        {"asset_hash": "frac", "latex": r"\frac{2}{3}"},
        {"asset_hash": "frag", "latex": r"\ce{^{-}_{3}}"},
        {"asset_hash": "ok", "latex": r"\frac{a}{b}"},
    ]

    selected = select_latex_form_targets(rows)

    assert [row["asset_hash"] for row in selected] == ["frac", "frag"]


def test_candidate_rows_never_use_codex_reviewer():
    from scripts.run_batch10_qa1 import candidate_row

    row = candidate_row("x", item_id="i1")

    assert row["review_status"] == "pending_user_or_claude"
    assert row["reviewer"] == ""
    assert "codex" not in json.dumps(row, ensure_ascii=False).lower()


def test_validate_outputs_only_checks_delivery_transcript_files():
    from scripts.run_batch10_qa1 import validate_outputs

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        task_dir = root / "formula_backfill"
        task_dir.mkdir()
        (task_dir / "formula_backfill_targets.jsonl").write_text(
            json.dumps({"asset_hash": "target_without_schema"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (task_dir / "formula_backfill_candidates.jsonl").write_text(
            json.dumps({"asset_hash": "candidate_without_schema"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        validation = validate_outputs(root)

    assert validation["transcript_schema_bad"] == 1


def test_answer_repair_cache_requires_complete_hash_set():
    from scripts.run_batch10_qa1 import load_cached_answer_repair

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        repair_dir = root / "asset_repair"
        repaired_dir = repair_dir / "repaired"
        repaired_dir.mkdir(parents=True)
        (repaired_dir / "h1.png").write_bytes(b"not-a-real-png")
        rows = [{"asset_hash": "h1"}, {"asset_hash": "h2"}]
        (repair_dir / "repair_attempts.jsonl").write_text(
            "\n".join(json.dumps({"asset_hash": h}, ensure_ascii=False) for h in ["h1", "h2"]) + "\n",
            encoding="utf-8",
        )
        (repair_dir / "repaired_assets.jsonl").write_text(
            json.dumps({"asset_hash": "h1", "status": "repaired"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (repair_dir / "unrepairable.jsonl").write_text(
            json.dumps({"asset_hash": "h2", "status": "unrepairable"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        cached = load_cached_answer_repair(root, rows)

    assert cached is not None
    repaired, unrepairable, attempts = cached
    assert [row["asset_hash"] for row in repaired] == ["h1"]
    assert [row["asset_hash"] for row in unrepairable] == ["h2"]
    assert len(attempts) == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            with tempfile.TemporaryDirectory() as d:
                try:
                    test(Path(d))
                except TypeError:
                    test()
            passed += 1
            print(f"PASS {test.__name__}")
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
