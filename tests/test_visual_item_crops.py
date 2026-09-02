#!/usr/bin/env python3
"""Unit-level checks for visual item crop helpers."""

from __future__ import annotations

from pathlib import Path
import sys

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))


def test_stem_anchors_prefers_normalized_stem_prefixes():
    from scripts.build_visual_item_crops import stem_anchors

    row = {
        "stem": "下列化学用语中正确的是（  ）\nA. 甲基的电子式",
        "visible_anchors": ["3", "下列化学用语中正确的是A甲基的电子式"],
    }

    anchors = stem_anchors(row)

    assert anchors[0].startswith("下列化学用语中正确的是")
    assert len(anchors) >= 2
    assert len(anchors) == len(set(anchors))


def test_source_pdf_path_uses_pdf_source_directly(tmp_path: Path):
    from scripts.build_visual_item_crops import source_pdf_path

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    assert source_pdf_path({"source_path": str(pdf)}) == pdf


def test_fallback_crop_fraction_is_larger_for_visual_heavy_categories():
    from scripts.build_visual_item_crops import fallback_crop_fraction

    assert fallback_crop_fraction({"category": "chart_curve"}) > fallback_crop_fraction({"category": "other"})
    assert fallback_crop_fraction({"category": "process_flow"}) > fallback_crop_fraction({"category": "other"})


def test_chart_curve_with_referenced_figure_expands_above_anchor():
    from scripts.build_visual_item_crops import context_padding_for

    row = {
        "category": "chart_curve",
        "stem": "上图反应历程中，表示使用了铁催化剂是______线，基元反应有______个。",
    }

    padding = context_padding_for(row)

    assert padding["top"] >= 180
    assert padding["bottom"] >= 260
    assert "referenced_figure_context" in padding["blockers"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            import tempfile

            if "tmp_path" in test.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    test(Path(d))
            else:
                test()
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 测试通过")
    sys.exit(0 if passed == len(tests) else 1)
