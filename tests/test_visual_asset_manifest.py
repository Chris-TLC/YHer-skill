#!/usr/bin/env python3
"""Tests for the visual asset manifest builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_visual_manifest_strong_requires_exact_source_image_and_matching_page(tmp_path: Path):
    from scripts.build_visual_asset_manifest import build_manifest
    from scripts.wire_data_to_engine import qid

    source_root = tmp_path / "papers"
    paper = source_root / "sample.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.4 fake")

    source_hash = hashlib.sha256(str(paper.resolve()).encode("utf-8")).hexdigest()[:12]
    page_image = tmp_path / "page_images_v3" / f"{source_hash}_p002.jpg"
    page_image.parent.mkdir(parents=True)
    page_image.write_bytes(b"x" * 2048)

    transcript = tmp_path / "full_markdown_v3" / "sample_transcript.md"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "--- 第1页 ---\n无关内容\n\n"
        "--- 第2页 ---\n2. 如图所示装置用于制取气体并检验生成气体的性质。\nA. 选项甲\nB. 选项乙\n",
        encoding="utf-8",
    )

    pdf_item = {
        "q_num": "2",
        "stem": "如图所示装置用于制取气体并检验生成气体的性质。\nA. 选项甲\nB. 选项乙",
        "options": {"A": "选项甲", "B": "选项乙"},
        "answer": "A",
        "question_type": "选择题",
        "difficulty": "T2",
        "diagram_description": "实验装置图",
        "_source_file": "sample.pdf",
        "_page": 2,
    }
    item_bank_row = {
        "item_id": qid(pdf_item),
        "source": "sample.pdf",
        "stem": pdf_item["stem"],
        "options": pdf_item["options"],
        "standard_solution": {"standard_answer": "A", "final_answers": ["A"]},
        "rubric": [{"point_id": "ans", "must_have": True}],
    }
    pdf_items = tmp_path / "all_from_pdf_v3.jsonl"
    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    write_jsonl(pdf_items, [pdf_item])
    write_jsonl(item_bank, [item_bank_row])

    manifest, summary = build_manifest(
        pdf_items_path=pdf_items,
        item_bank_path=item_bank,
        page_images_dir=tmp_path / "page_images_v3",
        transcript_dir=tmp_path / "full_markdown_v3",
        source_roots=[source_root],
    )

    assert summary["total_items"] == 1
    assert summary["image_like_items"] == 1
    assert summary["match_tiers"]["strong"] == 1
    assert summary["missing_page_image"] == 0
    row = manifest[0]
    assert row["item_id"] == item_bank_row["item_id"]
    assert row["source_path"] == str(paper)
    assert row["declared_page"] == 2
    assert row["best_text_page"] == 2
    assert row["page_image_path"] == str(page_image)
    assert row["page_image_hash"].startswith("sha256:")
    assert row["crop_tier"] == "page_only"
    assert row["match_tier"] == "strong"
    assert row["blocker_reasons"] == []


def test_visual_manifest_rejects_missing_image_and_marks_page_mismatch(tmp_path: Path):
    from scripts.build_visual_asset_manifest import build_manifest
    from scripts.wire_data_to_engine import qid

    source_root = tmp_path / "papers"
    paper = source_root / "sample.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.4 fake")

    transcript = tmp_path / "full_markdown_v3" / "sample_transcript.md"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        "--- 第1页 ---\n1. 如图所示流程用于制备物质。\nA. 选项甲\n"
        "--- 第2页 ---\n无关内容\n",
        encoding="utf-8",
    )

    pdf_item = {
        "q_num": "1",
        "stem": "如图所示流程用于制备物质。\nA. 选项甲",
        "options": {"A": "选项甲"},
        "answer": "A",
        "question_type": "选择题",
        "difficulty": "T2",
        "diagram_description": "流程图",
        "_source_file": "sample.pdf",
        "_page": 2,
    }
    item_bank_row = {
        "item_id": qid(pdf_item),
        "source": "sample.pdf",
        "stem": pdf_item["stem"],
        "options": pdf_item["options"],
        "standard_solution": {"standard_answer": "A", "final_answers": ["A"]},
        "rubric": [{"point_id": "ans", "must_have": True}],
    }
    pdf_items = tmp_path / "all_from_pdf_v3.jsonl"
    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    write_jsonl(pdf_items, [pdf_item])
    write_jsonl(item_bank, [item_bank_row])

    manifest, summary = build_manifest(
        pdf_items_path=pdf_items,
        item_bank_path=item_bank,
        page_images_dir=tmp_path / "page_images_v3",
        transcript_dir=tmp_path / "full_markdown_v3",
        source_roots=[source_root],
    )

    row = manifest[0]
    assert summary["match_tiers"]["reject"] == 1
    assert summary["page_mismatch"] == 1
    assert summary["missing_page_image"] == 1
    assert row["best_text_page"] == 1
    assert row["match_tier"] == "reject"
    assert "page_mismatch" in row["blocker_reasons"]
    assert "missing_page_image" in row["blocker_reasons"]


def test_lcs_scoring_is_bounded_for_long_pages(tmp_path: Path):
    from scripts.build_visual_asset_manifest import longest_common_substring_len

    started = time.perf_counter()
    score = longest_common_substring_len("甲" * 5000, "乙" * 5000)
    elapsed = time.perf_counter() - started

    assert score == 0
    assert elapsed < 0.2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    import tempfile

    for t in tests:
        try:
            with tempfile.TemporaryDirectory() as d:
                t(Path(d))
            passed += 1
            print(f"✅ {t.__name__}")
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 测试通过")
    sys.exit(0 if passed == len(tests) else 1)
