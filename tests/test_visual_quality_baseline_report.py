#!/usr/bin/env python3
"""Tests for full visual quality baseline reporting."""

from __future__ import annotations

import json
import os
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


def test_baseline_report_counts_full_visual_denominator_and_blockers(tmp_path: Path):
    from scripts.report_visual_quality_baseline import build_baseline_report

    item_quality = tmp_path / "item_quality_manifest.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    understanding = tmp_path / "visual_understanding_results.jsonl"
    write_jsonl(
        item_quality,
        [
            {
                "item_id": "strong_chart",
                "needs_image": True,
                "student_readable": True,
                "strong": True,
                "visual_pipeline_stage": "strong",
                "category": "chart_curve",
                "blocker_reasons": [],
            },
            {
                "item_id": "student_only_flow",
                "needs_image": True,
                "student_readable": True,
                "strong": False,
                "visual_pipeline_stage": "student_readable",
                "category": "process_flow",
                "blocker_reasons": ["llm_understanding_not_strong"],
            },
            {
                "item_id": "missing_asset",
                "needs_image": True,
                "student_readable": False,
                "strong": False,
                "visual_pipeline_stage": "raw_visual_item",
                "category": "experiment_device",
                "blocker_reasons": ["missing_page_image"],
            },
            {
                "item_id": "text_ok",
                "needs_image": False,
                "student_readable": True,
                "strong": True,
                "visual_pipeline_stage": "text_ready",
                "category": "",
                "blocker_reasons": [],
            },
        ],
    )
    write_jsonl(
        visual_manifest,
        [
            {"item_id": "strong_chart", "category": "chart_curve", "match_tier": "strong", "blocker_reasons": []},
            {"item_id": "student_only_flow", "category": "process_flow", "match_tier": "strong", "blocker_reasons": []},
            {"item_id": "missing_asset", "category": "experiment_device", "match_tier": "reject", "blocker_reasons": ["missing_page_image"]},
        ],
    )
    write_jsonl(
        understanding,
        [
            {"item_id": "strong_chart", "category": "chart_curve", "understanding_pass": True, "error_types": []},
            {
                "item_id": "student_only_flow",
                "category": "process_flow",
                "understanding_pass": False,
                "error_types": ["answer_mismatch", "high_confidence_error"],
            },
        ],
    )
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        report = build_baseline_report(
            item_quality_path=item_quality,
            visual_manifest_path=visual_manifest,
            understanding_results_path=understanding,
            env_path=tmp_path / ".env",
            strong_target_rate=0.8,
        )
    finally:
        if old_key is not None:
            os.environ["OPENAI_API_KEY"] = old_key

    assert report["visual_denominator"] == 3
    assert report["strong_target_count"] == 3
    assert report["current_strong_count"] == 1
    assert report["student_readable_count"] == 2
    assert report["blocker_distribution"]["missing_page_image"] == 2
    assert report["blocker_distribution"]["high_confidence_error"] == 1
    assert report["by_category"]["chart_curve"]["strong"] == 1
    assert report["by_category"]["process_flow"]["student_readable"] == 1
    assert report["api_key_status"]["openai"] is False
    assert "recommendations" in report


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
