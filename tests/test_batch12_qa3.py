#!/usr/bin/env python3
"""Focused contract tests for Batch 12 QA-3 L0 candidate generation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_split_segments_accepts_tight_join_and_rejects_decimal_false_split():
    from scripts.run_batch12_qa3 import split_option_segments

    ok = split_option_segments("酸性B. 熔点C. 沸点")
    bad = split_option_segments("0.5B. 不是选项C. 仍不应吞字")

    assert ok.ok is True
    assert ok.segments == ["酸性", "B. 熔点", "C. 沸点"]
    assert bad.ok is False
    assert "no_valid_split_markers" in bad.reason


def test_split_segments_keeps_strictly_increasing_tail_options():
    from scripts.run_batch12_qa3 import split_option_segments

    result = split_option_segments("C．稳定性 SiH4>PH3 D．原子半径N>C")

    assert result.ok is True
    assert result.labels == ["C", "D"]
    assert result.segments == ["C．稳定性 SiH4>PH3", "D．原子半径N>C"]
    assert "".join(result.segments).replace(" ", "") == "C．稳定性SiH4>PH3D．原子半径N>C"


def test_split_segments_low_confidence_for_non_increasing_sequence():
    from scripts.run_batch12_qa3 import split_option_segments

    result = split_option_segments("B. 前项A. 后项")

    assert result.ok is False
    assert result.reason == "option_labels_not_strictly_increasing"


def test_build_option_split_rows_stamps_candidates_and_low_confidence():
    from scripts.run_batch12_qa3 import build_option_split_rows

    items = [
        {
            "item_id": "i1",
            "group_key": "g",
            "section_num": 1,
            "q_num": 2,
            "stem_blocks": [{"para": [{"type": "text", "text": "C. 一项 D. 二项"}]}],
            "answer_blocks_effective": [{"para": [{"type": "text", "text": "0.5B. 不拆 C. 不拆"}]}],
        }
    ]
    target_ids = {"i1"}

    candidates, low, summary = build_option_split_rows(items, target_ids)

    assert len(candidates) == 1
    assert len(low) == 1
    row = candidates[0]
    assert row["split_kind"] == "tight_text"
    assert row["zone"] == "stem"
    assert row["block_path"] == "stem_blocks[0].para[0]"
    assert row["review_status"] == "pending_user_or_claude"
    assert row["reviewer"] == ""
    assert summary["join_mismatch"] == 0


def test_refmap_candidate_resolves_unique_manifest_ref_with_source_hash(tmp_path: Path):
    from scripts.run_batch12_qa3 import build_refmap_fix_rows

    source = tmp_path / "asset.wmf"
    source.write_bytes(b"wmf")
    manifest_rows = [
        {
            "asset_hash": "a" * 64,
            "sample_refs": [
                {
                    "group_key": "g",
                    "media": "image1.wmf",
                    "asset_path": str(source),
                }
            ],
        }
    ]
    refs = [{"group_key": "g", "media": "ans_abcd1234_image1.wmf", "zone": "answer", "item_ids": ["i1"]}]

    candidates, unresolved = build_refmap_fix_rows(refs, manifest_rows, existing_ref_map={})

    assert unresolved == []
    assert candidates[0]["suggested_asset_hash"] == "a" * 64
    assert candidates[0]["matched_media"] == "image1.wmf"
    assert candidates[0]["source_sha256"]
    assert candidates[0]["review_status"] == "pending_user_or_claude"
    assert candidates[0]["reviewer"] == ""


def test_manual_queue_targets_exclude_batch10_unrepairable_hashes():
    from scripts.run_batch12_qa3 import select_manual_queue_rerender_targets

    transcripts = [
        {"asset_hash": "a", "pool": "manual_queue", "asset_class": "illustration"},
        {"asset_hash": "b", "pool": "manual_queue", "asset_class": "formula_image"},
        {"asset_hash": "c", "pool": "ai_seed", "asset_class": "illustration"},
    ]
    manifest_by_hash = {
        "a": {"asset_hash": "a", "sample_refs": [{"asset_path": "/tmp/a.png"}]},
        "b": {"asset_hash": "b", "sample_refs": [{"asset_path": "/tmp/b.png"}]},
    }

    targets, excluded = select_manual_queue_rerender_targets(transcripts, manifest_by_hash, {"b"})

    assert [row["asset_hash"] for row in targets] == ["a"]
    assert excluded == {"batch10_unrepairable": 1}
    assert targets[0]["source_pool"] == "manual_queue"


def test_omml_backfill_keeps_only_missing_cache_keys():
    from scripts.run_batch12_qa3 import missing_omml_sources

    sources = {
        "a": {"omml_sha1": "a", "omml": "<m/>", "occurrences": []},
        "b": {"omml_sha1": "b", "omml": "<m/>", "occurrences": []},
    }

    missing = missing_omml_sources(sources, [{"omml_sha1": "a", "latex": "x"}])

    assert [row["omml_sha1"] for row in missing] == ["b"]


def test_gate_transcription_candidates_rejects_failure_prompt_and_leak():
    from scripts.run_batch12_qa3 import gate_transcription_candidates

    rows = [
        {"asset_hash": "ok", "summary": "烧杯和导管", "pool": "display_only"},
        {"asset_hash": "prompt", "summary": "sorry cannot read image", "pool": "display_only"},
        {"asset_hash": "leak", "summary": "【试题解析】泄漏", "pool": "display_only"},
    ]

    kept, prompt_rejected, leak_rejected = gate_transcription_candidates(rows, "unit", mode="transcript")

    assert [row["asset_hash"] for row in kept] == ["ok"]
    assert [row["asset_hash"] for row in prompt_rejected] == ["prompt"]
    assert [row["asset_hash"] for row in leak_rejected] == ["leak"]
    assert kept[0]["review_status"] == "pending_user_or_claude"
    assert kept[0]["reviewer"] == ""


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
