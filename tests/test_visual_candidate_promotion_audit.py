#!/usr/bin/env python3
"""Tests for candidate visual strong promotion audit planning."""

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


def quality_row(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "needs_image": True,
        "student_readable": True,
        "strong": True,
        "visual_pipeline_stage": "strong",
        "visual_asset_status": "strong",
        "readability_status": "pass",
        "llm_understanding_status": "strong",
        "answer_status": "verified",
        "rubric_status": "complete",
        "blocker_reasons": [],
        "strong_blocker_reasons": [],
        "usable_for_profile_evidence": True,
    }


def visual_row(item_id: str, crop_path: str, extra: dict | None = None) -> dict:
    row = {
        "item_id": item_id,
        "match_tier": "strong",
        "source_file": "paper.pdf",
        "declared_page": 3,
        "page_image_path": f"/tmp/{item_id}_page.jpg",
        "page_image_hash": f"sha256:{item_id}page",
        "crop_path": crop_path,
        "crop_hash": f"sha256:{item_id}crop",
        "crop_tier": "item_crop_candidate",
        "category": "chart_curve",
        "blocker_reasons": [],
    }
    if extra:
        row.update(extra)
    return row


def understanding_row(item_id: str, extra: dict | None = None) -> dict:
    row = {
        "item_id": item_id,
        "model": "gemini-3.1-pro-preview",
        "visible_pass": True,
        "answer_match": True,
        "understanding_pass": True,
        "profile_evidence_allowed": True,
        "confidence": 1.0,
        "error_types": [],
        "raw_source": "model_call",
    }
    if extra:
        row.update(extra)
    return row


def test_audit_approves_safe_candidate_with_complete_evidence(tmp_path: Path):
    from scripts.audit_visual_candidate_promotions import audit_promotions

    quality = tmp_path / "candidate_quality.jsonl"
    visual = tmp_path / "candidate_visual.jsonl"
    understanding = tmp_path / "candidate_understanding.jsonl"
    official = tmp_path / "official_visual.jsonl"

    write_jsonl(quality, [quality_row("safe")])
    write_jsonl(visual, [visual_row("safe", "/tmp/safe_crop.png")])
    write_jsonl(understanding, [understanding_row("safe")])
    write_jsonl(official, [visual_row("safe", "", {"crop_hash": ""})])

    routed, summary = audit_promotions(
        candidate_quality_path=quality,
        candidate_visual_path=visual,
        candidate_understanding_path=understanding,
        official_visual_path=official,
    )

    assert summary["approved"] == 1
    assert routed["approved"][0]["item_id"] == "safe"
    assert routed["approved"][0]["promotion_scope"] == "visual_asset_and_understanding"
    assert routed["approved"][0]["review_reasons"] == []


def test_audit_manual_reviews_replacing_existing_official_crop(tmp_path: Path):
    from scripts.audit_visual_candidate_promotions import audit_promotions

    quality = tmp_path / "candidate_quality.jsonl"
    visual = tmp_path / "candidate_visual.jsonl"
    understanding = tmp_path / "candidate_understanding.jsonl"
    official = tmp_path / "official_visual.jsonl"

    write_jsonl(quality, [quality_row("changed_crop")])
    write_jsonl(visual, [visual_row("changed_crop", "/tmp/changed_crop_new.png")])
    write_jsonl(understanding, [understanding_row("changed_crop")])
    write_jsonl(
        official,
        [
            visual_row(
                "changed_crop",
                "/tmp/changed_crop_old.png",
                {"crop_hash": "sha256:oldcrop"},
            )
        ],
    )

    routed, summary = audit_promotions(
        candidate_quality_path=quality,
        candidate_visual_path=visual,
        candidate_understanding_path=understanding,
        official_visual_path=official,
    )

    assert summary["manual_review"] == 1
    assert "official_existing_crop_would_be_replaced" in routed["manual_review"][0]["review_reasons"]


def test_audit_routes_cross_page_candidate_to_manual_review(tmp_path: Path):
    from scripts.audit_visual_candidate_promotions import audit_promotions

    quality = tmp_path / "candidate_quality.jsonl"
    visual = tmp_path / "candidate_visual.jsonl"
    understanding = tmp_path / "candidate_understanding.jsonl"
    official = tmp_path / "official_visual.jsonl"

    write_jsonl(quality, [quality_row("cross")])
    write_jsonl(
        visual,
        [
            visual_row(
                "cross",
                "/tmp/cross_crop.png",
                {"additional_image_paths": ["/tmp/cross_next_page.jpg"]},
            )
        ],
    )
    write_jsonl(understanding, [understanding_row("cross", {"additional_image_paths": ["/tmp/cross_next_page.jpg"]})])
    write_jsonl(official, [visual_row("cross", "")])

    routed, summary = audit_promotions(
        candidate_quality_path=quality,
        candidate_visual_path=visual,
        candidate_understanding_path=understanding,
        official_visual_path=official,
    )

    assert summary["manual_review"] == 1
    assert routed["manual_review"][0]["item_id"] == "cross"
    assert "cross_page_evidence_requires_promotion_policy" in routed["manual_review"][0]["review_reasons"]


def test_audit_routes_candidate_answer_repair_to_manual_review(tmp_path: Path):
    from scripts.audit_visual_candidate_promotions import audit_promotions

    quality = tmp_path / "candidate_quality.jsonl"
    visual = tmp_path / "candidate_visual.jsonl"
    understanding = tmp_path / "candidate_understanding.jsonl"
    official = tmp_path / "official_visual.jsonl"

    write_jsonl(quality, [quality_row("repaired")])
    write_jsonl(visual, [visual_row("repaired", "/tmp/repaired_crop.png")])
    write_jsonl(
        understanding,
        [
            understanding_row(
                "repaired",
                {
                    "candidate_repair": {
                        "official_source_has_answer": False,
                        "do_not_copy_to_official_without_promotion_audit": True,
                    }
                },
            )
        ],
    )
    write_jsonl(official, [visual_row("repaired", "")])

    routed, summary = audit_promotions(
        candidate_quality_path=quality,
        candidate_visual_path=visual,
        candidate_understanding_path=understanding,
        official_visual_path=official,
    )

    assert summary["manual_review"] == 1
    assert "candidate_answer_repair_requires_human_policy" in routed["manual_review"][0]["review_reasons"]


def test_audit_approves_structured_transcript_candidate_without_crop(tmp_path: Path):
    from scripts.audit_visual_candidate_promotions import audit_promotions

    quality = tmp_path / "candidate_quality.jsonl"
    visual = tmp_path / "candidate_visual.jsonl"
    understanding = tmp_path / "candidate_understanding.jsonl"
    official = tmp_path / "official_visual.jsonl"

    write_jsonl(
        quality,
        [
            {
                **quality_row("transcript"),
                "visual_evidence_mode": "structured_transcript",
                "crop_path": "",
                "crop_hash": "",
                "strong_blocker_reasons": [],
            }
        ],
    )
    write_jsonl(
        visual,
        [
            visual_row(
                "transcript",
                "",
                {
                    "crop_hash": "",
                    "crop_tier": "page_only",
                },
            )
        ],
    )
    write_jsonl(
        understanding,
        [
            understanding_row(
                "transcript",
                {
                    "visual_evidence_mode": "structured_transcript",
                    "transcript_supported_strong": True,
                },
            )
        ],
    )
    write_jsonl(official, [visual_row("transcript", "", {"crop_hash": "", "crop_tier": "page_only"})])

    routed, summary = audit_promotions(
        candidate_quality_path=quality,
        candidate_visual_path=visual,
        candidate_understanding_path=understanding,
        official_visual_path=official,
    )

    assert summary["approved"] == 1
    assert routed["approved"][0]["item_id"] == "transcript"
    assert routed["approved"][0]["promotion_scope"] == "understanding_structured_transcript"
    assert routed["approved"][0]["reject_reasons"] == []


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
