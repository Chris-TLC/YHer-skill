#!/usr/bin/env python3
"""Tests for balanced visual eval set selection."""

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


def item(item_id: str, category: str = "chart_curve") -> dict:
    return {
        "item_id": item_id,
        "stem": f"{item_id} stem",
        "options": {"A": "甲", "B": "乙"},
        "standard_solution": {"standard_answer": "A", "final_answers": ["A"]},
        "rubric": [{"point_id": "ans", "must_have": True}],
        "question_type": "选择题",
        "difficulty": "T2",
        "kg_nodes": [category],
    }


def visual(item_id: str, category: str, crop_tier: str = "page_only") -> dict:
    return {
        "item_id": item_id,
        "category": category,
        "match_tier": "strong",
        "page_image_path": f"/tmp/{item_id}.png",
        "page_image_hash": f"sha256:{item_id}",
        "source_file": "sample.pdf",
        "source_path": "/tmp/sample.pdf",
        "declared_page": 1,
        "best_text_page": 1,
        "crop_tier": crop_tier,
        "crop_path": f"/tmp/{item_id}_crop.png" if "crop" in crop_tier else "",
    }


def test_eval_set_prefers_crop_candidates_within_category(tmp_path: Path):
    from scripts.build_visual_eval_set import build_eval_set

    item_bank = tmp_path / "items.jsonl"
    manifest = tmp_path / "visual.jsonl"
    write_jsonl(item_bank, [item("a_page_first"), item("z_crop_first")])
    write_jsonl(
        manifest,
        [
            visual("a_page_first", "chart_curve", "page_only"),
            visual("z_crop_first", "chart_curve", "item_crop_candidate"),
        ],
    )

    rows, _ = build_eval_set(
        item_bank_path=item_bank,
        visual_manifest_path=manifest,
        pdf_items_path=tmp_path / "missing_pdf.jsonl",
        pilot_path=tmp_path / "missing_pilot.json",
        per_category=1,
        max_items=1,
    )

    assert [row["item_id"] for row in rows] == ["z_crop_first"]


def test_eval_set_excludes_items_from_previous_batch(tmp_path: Path):
    from scripts.build_visual_eval_set import build_eval_set

    item_bank = tmp_path / "items.jsonl"
    manifest = tmp_path / "visual.jsonl"
    previous = tmp_path / "previous.jsonl"
    write_jsonl(item_bank, [item("already_seen"), item("new_item")])
    write_jsonl(
        manifest,
        [
            visual("already_seen", "chart_curve", "item_crop_candidate"),
            visual("new_item", "chart_curve", "item_crop_candidate"),
        ],
    )
    write_jsonl(previous, [{"item_id": "already_seen"}])

    rows, summary = build_eval_set(
        item_bank_path=item_bank,
        visual_manifest_path=manifest,
        pdf_items_path=tmp_path / "missing_pdf.jsonl",
        pilot_path=tmp_path / "missing_pilot.json",
        exclude_items_paths=[previous],
        per_category=1,
        max_items=2,
    )

    assert [row["item_id"] for row in rows] == ["new_item"]
    assert summary["excluded_items"] == 1


def test_eval_set_uses_external_crop_evidence_for_crop_first(tmp_path: Path):
    from scripts.build_visual_eval_set import build_eval_set

    item_bank = tmp_path / "items.jsonl"
    manifest = tmp_path / "visual.jsonl"
    crop_evidence = tmp_path / "crops.jsonl"
    write_jsonl(item_bank, [item("a_page_first"), item("z_crop_first")])
    write_jsonl(
        manifest,
        [
            visual("a_page_first", "chart_curve", "page_only"),
            visual("z_crop_first", "chart_curve", "page_only"),
        ],
    )
    write_jsonl(
        crop_evidence,
        [
            {
                "item_id": "z_crop_first",
                "crop_tier": "item_crop_candidate",
                "crop_path": "/tmp/z_crop_first.png",
                "crop_hash": "sha256:z_crop_first",
            }
        ],
    )

    rows, summary = build_eval_set(
        item_bank_path=item_bank,
        visual_manifest_path=manifest,
        pdf_items_path=tmp_path / "missing_pdf.jsonl",
        pilot_path=tmp_path / "missing_pilot.json",
        crop_evidence_path=crop_evidence,
        per_category=1,
        max_items=1,
    )

    assert [row["item_id"] for row in rows] == ["z_crop_first"]
    assert rows[0]["crop_path"] == "/tmp/z_crop_first.png"
    assert rows[0]["crop_hash"] == "sha256:z_crop_first"
    assert summary["crop_first_candidates"] == 1


def test_eval_set_limits_category_overfill_when_alternatives_exist(tmp_path: Path):
    from scripts.build_visual_eval_set import build_eval_set

    item_bank = tmp_path / "items.jsonl"
    manifest = tmp_path / "visual.jsonl"
    rows = [item(f"chart_{i}", "chart_curve") for i in range(8)]
    rows += [item(f"flow_{i}", "process_flow") for i in range(8)]
    visuals = [visual(f"chart_{i}", "chart_curve", "item_crop_candidate") for i in range(8)]
    visuals += [visual(f"flow_{i}", "process_flow", "item_crop_candidate") for i in range(8)]
    write_jsonl(item_bank, rows)
    write_jsonl(manifest, visuals)

    selected, summary = build_eval_set(
        item_bank_path=item_bank,
        visual_manifest_path=manifest,
        pdf_items_path=tmp_path / "missing_pdf.jsonl",
        pilot_path=tmp_path / "missing_pilot.json",
        per_category=1,
        max_items=10,
        max_per_category=5,
    )

    assert summary["category_counts"]["chart_curve"] <= 5
    assert summary["category_counts"]["process_flow"] <= 5
    assert len(selected) == 10


def test_eval_set_respects_category_targets_for_full_batch_shape(tmp_path: Path):
    from scripts.build_visual_eval_set import build_eval_set

    item_bank = tmp_path / "items.jsonl"
    manifest = tmp_path / "visual.jsonl"
    item_rows = []
    visual_rows = []
    for category in [
        "crystal_cell",
        "experiment_device",
        "process_flow",
        "chart_curve",
        "organic_structure",
        "electrochem_device",
        "other",
    ]:
        for i in range(20):
            item_id = f"{category}_{i}"
            item_rows.append(item(item_id, category))
            visual_rows.append(visual(item_id, category, "item_crop_candidate"))
    write_jsonl(item_bank, item_rows)
    write_jsonl(manifest, visual_rows)

    selected, summary = build_eval_set(
        item_bank_path=item_bank,
        visual_manifest_path=manifest,
        pdf_items_path=tmp_path / "missing_pdf.jsonl",
        pilot_path=tmp_path / "missing_pilot.json",
        per_category=3,
        max_items=50,
        max_per_category=12,
        category_targets={
            "crystal_cell": 5,
            "experiment_device": 9,
            "process_flow": 4,
            "chart_curve": 9,
            "organic_structure": 9,
            "electrochem_device": 5,
            "other": 9,
        },
    )

    assert len(selected) == 50
    assert summary["category_counts"] == {
        "crystal_cell": 5,
        "experiment_device": 9,
        "process_flow": 4,
        "chart_curve": 9,
        "organic_structure": 9,
        "electrochem_device": 5,
        "other": 9,
    }
    assert summary["category_targets"]["crystal_cell"] == 5


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
