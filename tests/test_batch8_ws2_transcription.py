#!/usr/bin/env python3
"""Focused contract tests for Batch 8 WS2 asset repair and transcription."""

from __future__ import annotations

import inspect
import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def make_rgb_png(width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> bytes:
    rows = [b"\x00" + b"".join(bytes(rgb) for rgb in row) for row in pixels]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(b"".join(rows))) + png_chunk(b"IEND", b"")


def test_blank_detector_uses_extrema_min_threshold(tmp_path: Path):
    from scripts.run_batch8_ws2 import is_blank_image

    blank = tmp_path / "blank.png"
    nonblank = tmp_path / "nonblank.png"
    blank.write_bytes(make_rgb_png(2, 2, [[(250, 250, 250), (255, 255, 255)], [(251, 252, 253), (255, 255, 255)]]))
    nonblank.write_bytes(make_rgb_png(2, 2, [[(255, 255, 255), (249, 255, 255)], [(255, 255, 255), (255, 255, 255)]]))

    assert is_blank_image(blank) is True
    assert is_blank_image(nonblank) is False


def test_normalized_path_is_remapped_to_candidate_root(tmp_path: Path):
    from scripts.run_batch8_ws2 import normalized_png_for

    root = tmp_path / "ws2"
    expected = root / "normalized" / "abc123.png"

    assert normalized_png_for({"asset_hash": "abc123", "normalized_png": "/tmp/old/abc123.png"}, root) == expected


def test_illustration_prompt_only_uses_stem_text_prefix():
    from scripts.run_batch8_ws2 import build_illustration_prompts

    item = {
        "stem_text": "甲" * 350,
        "answer" + "_blocks": [{"para": [{"type": "text", "text": "正确" + "答案 A"}]}],
        "analysis" + "_blocks": [{"para": [{"type": "text", "text": "【解" + "析】秘密"}]}],
    }

    _system_prompt, user_prompt = build_illustration_prompts("plot_curve", item)

    assert "甲" * 300 in user_prompt
    assert "甲" * 301 not in user_prompt
    assert "正确" + "答案" not in user_prompt
    assert "【解" + "析】" not in user_prompt
    assert "秘密" not in user_prompt


def test_formula_prompt_contains_batch8p1_relation_rule_verbatim():
    from scripts.run_batch8_ws2 import RELATION_SYMBOL_TRANSCRIPTION_RULE, build_formula_prompts

    system_prompt, user_prompt = build_formula_prompts()
    joined = system_prompt + user_prompt

    assert RELATION_SYMBOL_TRANSCRIPTION_RULE in joined
    assert (
        "化学式之间的关系符号(=、→、⇌、↑、↓)必须严格按图面逐字转写。禁止依据化学习惯改写:图面是等号就写 =,即使你认为该反应可逆或应写箭头。"
        in joined
    )


def test_batch8p1_relation_target_selector_is_literal():
    from scripts.run_batch8_ws2 import select_batch8p1_relation_targets

    rows = [
        {"asset_hash": "a", "latex": r"\ce{A <=> B}"},
        {"asset_hash": "b", "latex": r"\ce{A -> B}"},
        {"asset_hash": "c", "latex": r"\ce{A = B}"},
        {"asset_hash": "d", "latex": r"\rightleftharpoons"},
    ]

    assert [row["asset_hash"] for row in select_batch8p1_relation_targets(rows)] == ["a", "b"]


def test_batch8p1_track_i_fallback_targets_keep_gold_source_rows():
    from scripts.run_batch8_ws2 import select_batch8p1_track_i_fallback_targets

    formula_rows = [
        {"asset_hash": "a", "latex_status": "failed", "latex": ""},
        {"asset_hash": "b", "latex_status": "passed", "latex": "x"},
        {"asset_hash": "c", "pool_reason": "pipeline_exception"},
    ]
    gold_rows = [{"asset_hash": "a", "latex_status": "failed", "latex": ""}, {"asset_hash": "g", "latex_status": "failed", "latex": ""}]

    targets = select_batch8p1_track_i_fallback_targets(formula_rows, gold_rows)

    assert [target["asset_hash"] for target in targets] == ["a", "a", "g"]
    assert [target["source_scope"] for target in targets] == [
        "formula_latex_candidates_latex_status_failed",
        "gold_blind_formula_latex_failures_named",
        "gold_blind_formula_latex_failures_named",
    ]


def test_leak_gate_catches_required_patterns():
    from scripts.run_batch8_ws2 import find_leak_hits

    assert find_leak_hits({"summary": "【试题解析】这里泄漏了"})
    assert find_leak_hits({"elements": ["故选B"]})
    assert find_leak_hits({"text_in_image": ["正确答案：C"]})
    assert not find_leak_hits({"summary": "图中显示烧杯和导管"})


def test_pool_assignment_matches_spec_thresholds():
    from scripts.run_batch8_ws2 import assign_pool

    assert assign_pool(True, 0.82, "plot_curve", "")["pool"] == "ai_seed"
    assert assign_pool(False, 0.70, "plot_curve", "文字少量不同")["pool"] == "display_only"
    assert assign_pool(True, 0.90, "broken_image", "")["pool"] == "manual_queue"
    assert assign_pool(True, 0.90, "icon_or_noise", "")["pool"] == "manual_queue"


def test_katex_validation_accepts_leading_minus_latex():
    from scripts.run_batch8_ws2 import validate_latex_katex

    result = validate_latex_katex("-2")

    assert result["ok"] is True


def test_transcript_row_has_required_schema_version():
    from scripts.run_batch8_ws2 import make_transcript_row

    row = make_transcript_row(
        asset_hash="h1",
        fine_type="plot_curve",
        merged={"summary": "图", "elements": [], "text_in_image": [], "data_points": [], "uncertain": ["不确定"], "confidence": 0.8},
        run_low={"fine_type": "plot_curve"},
        run_high={"fine_type": "plot_curve"},
        consistency={"consistent": True},
        metadata={},
    )

    assert row["schema_version"] == "ws2_transcript_v1"
    assert row["asset_hash"] == "h1"


def test_phash_groups_near_duplicate_images(tmp_path: Path):
    from scripts.run_batch8_ws2 import build_near_duplicate_groups

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    pixels_a = [
        [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
        [(255, 255, 255), (10, 10, 10), (20, 20, 20), (255, 255, 255)],
        [(255, 255, 255), (30, 30, 30), (40, 40, 40), (255, 255, 255)],
        [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)],
    ]
    pixels_b = [
        [(250, 250, 250), (250, 250, 250), (250, 250, 250), (250, 250, 250)],
        [(250, 250, 250), (12, 12, 12), (22, 22, 22), (250, 250, 250)],
        [(250, 250, 250), (32, 32, 32), (42, 42, 42), (250, 250, 250)],
        [(250, 250, 250), (250, 250, 250), (250, 250, 250), (250, 250, 250)],
    ]
    a.write_bytes(make_rgb_png(4, 4, pixels_a))
    b.write_bytes(make_rgb_png(4, 4, pixels_b))
    out_root = tmp_path / "out"
    norm = out_root / "asset_repair" / "repaired"
    norm.mkdir(parents=True)
    (norm / "hash_a.png").write_bytes(a.read_bytes())
    (norm / "hash_b.png").write_bytes(b.read_bytes())

    mapping = build_near_duplicate_groups([{"asset_hash": "hash_a"}, {"asset_hash": "hash_b"}], out_root)

    assert mapping["hash_a"] == "hash_a"
    assert mapping["hash_b"] == "hash_a"


def test_vision_client_accepts_temperature_parameter():
    from adapters.vision_client import VisionClient

    assert "temperature" in inspect.signature(VisionClient.read_page).parameters


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
