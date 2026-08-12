#!/usr/bin/env python3
"""Tests for static visual quality preview gating."""

from __future__ import annotations

import json
from pathlib import Path
import sys

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_preview_uses_student_readable_not_diagnosis_flag(tmp_path: Path):
    from scripts.build_visual_quality_preview import build_html

    eval_set = tmp_path / "visual_item_eval_set.jsonl"
    quality = tmp_path / "item_quality_manifest.jsonl"
    out = tmp_path / "preview.html"
    write_jsonl(
        eval_set,
        [
            {
                "item_id": "practice_only",
                "category": "experiment_device",
                "difficulty": "T2",
                "question_type": "选择题",
                "stem": "如图所示实验装置题",
                "options": {"A": "甲", "B": "乙"},
                "source_file": "paper.pdf",
                "page": 2,
                "page_image_path": "/tmp/p002.jpg",
            },
            {
                "item_id": "diagnosis_legacy_but_unreadable",
                "category": "process_flow",
                "difficulty": "T3",
                "question_type": "填空题",
                "stem": "如图所示流程题",
                "options": {},
                "source_file": "paper.pdf",
                "page": 3,
                "page_image_path": "",
            },
        ],
    )
    write_jsonl(
        quality,
        [
            {
                "item_id": "practice_only",
                "student_readable": True,
                "strong": False,
                "usable_for_diagnosis": False,
                "visual_pipeline_stage": "student_readable",
                "visual_asset_status": "strong",
                "readability_status": "pass",
                "llm_understanding_status": "weak",
                "blocker_reasons": [],
            },
            {
                "item_id": "diagnosis_legacy_but_unreadable",
                "student_readable": False,
                "strong": False,
                "usable_for_diagnosis": True,
                "visual_pipeline_stage": "raw_visual_item",
                "visual_asset_status": "reject",
                "readability_status": "reject",
                "llm_understanding_status": "reject",
                "blocker_reasons": ["missing_page_image"],
            },
        ],
    )

    html = build_html(eval_set_path=eval_set, quality_path=quality, out_path=out, limit=5)

    assert "practice_only" in html
    assert "student-readable" in html
    assert "not-student-readable" in html
    assert "diagnosis-strong" not in html
    assert "missing_page_image" in html


def test_preview_packages_existing_images_next_to_html(tmp_path: Path):
    from scripts.build_visual_quality_preview import build_html

    source_dir = tmp_path / "source_images"
    source_dir.mkdir()
    image = source_dir / "page one.png"
    image.write_bytes(b"fake-image")
    eval_set = tmp_path / "visual_item_eval_set.jsonl"
    quality = tmp_path / "item_quality_manifest.jsonl"
    out = tmp_path / "preview" / "visual_quality_preview.html"
    write_jsonl(
        eval_set,
        [
            {
                "item_id": "packaged_image",
                "category": "chart_curve",
                "difficulty": "T2",
                "question_type": "选择题",
                "stem": "如图所示曲线题",
                "options": {"A": "甲"},
                "source_file": "paper.pdf",
                "page": 1,
                "page_image_path": str(image),
            }
        ],
    )
    write_jsonl(
        quality,
        [
            {
                "item_id": "packaged_image",
                "student_readable": True,
                "strong": False,
                "visual_pipeline_stage": "student_readable",
                "visual_asset_status": "strong",
                "readability_status": "pass",
                "llm_understanding_status": "weak",
                "blocker_reasons": [],
            }
        ],
    )

    html = build_html(eval_set_path=eval_set, quality_path=quality, out_path=out, limit=1)

    packaged = out.parent / "visual_quality_preview_assets" / "packaged_image_page-one.png"
    assert packaged.read_bytes() == b"fake-image"
    assert 'src="visual_quality_preview_assets/packaged_image_page-one.png"' in html


def test_preview_declares_empty_favicon_to_keep_browser_console_clean(tmp_path: Path):
    from scripts.build_visual_quality_preview import build_html

    eval_set = tmp_path / "visual_item_eval_set.jsonl"
    quality = tmp_path / "item_quality_manifest.jsonl"
    out = tmp_path / "preview.html"
    write_jsonl(eval_set, [])
    write_jsonl(quality, [])

    html = build_html(eval_set_path=eval_set, quality_path=quality, out_path=out, limit=1)

    assert '<link rel="icon" href="data:,">' in html


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    import tempfile

    for test in tests:
        try:
            with tempfile.TemporaryDirectory() as d:
                test(Path(d))
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 测试通过")
    sys.exit(0 if passed == len(tests) else 1)
