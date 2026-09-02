#!/usr/bin/env python3
"""Tests for applying approved visual candidate promotions."""

from __future__ import annotations

import json
import hashlib
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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def visual_row(item_id: str, crop_path: str | None = None, crop_hash: str | None = None) -> dict:
    return {
        "item_id": item_id,
        "source_file": "paper.pdf",
        "source_path": "/source/paper.pdf",
        "source_candidates": 1,
        "source_ambiguous": False,
        "declared_page": 1,
        "best_text_page": 1,
        "page_image_path": "/tmp/page.jpg",
        "page_image_hash": "sha256:page",
        "crop_path": crop_path,
        "crop_hash": crop_hash,
        "visible_anchors": ["anchor"],
        "declared_match_score": 0.99,
        "best_match_score": 0.99,
        "declared_text_tier": "strong",
        "best_text_tier": "strong",
        "match_tier": "strong",
        "crop_tier": "page_only" if not crop_path else "item_crop_candidate",
        "needs_image": True,
        "category": "chart_curve",
        "question_type": "选择题",
        "difficulty": "T2",
        "answer": "A",
        "blocker_reasons": [],
    }


def understanding_row(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "category": "chart_curve",
        "model": "gemini-3.1-pro-preview",
        "input_image_path": "/tmp/candidate_crop.png",
        "standard_answer": "A",
        "model_answer": "A",
        "confidence": 1.0,
        "visible_pass": True,
        "answer_match": True,
        "understanding_pass": True,
        "profile_evidence_allowed": True,
        "error_types": [],
        "raw_source": "model_call",
    }


def approval_row(item_id: str, crop_path: Path, crop_hash: str = "sha256:candidatecrop") -> dict:
    return {
        "item_id": item_id,
        "status": "approved",
        "promotion_scope": "visual_asset_and_understanding",
        "category": "chart_curve",
        "review_reasons": [],
        "reject_reasons": [],
        "candidate_crop_path": str(crop_path),
        "candidate_crop_hash": crop_hash,
        "candidate_page_image_path": "/tmp/page.jpg",
        "candidate_page_image_hash": "sha256:page",
        "additional_image_paths": [],
        "candidate_repair": {},
    }


def test_apply_approved_promotions_copies_crop_and_updates_manifests(tmp_path: Path):
    from scripts.promote_visual_candidate_approvals import apply_approved_promotions

    crop_src = tmp_path / "tmp_run" / "candidate_crop.png"
    crop_src.parent.mkdir(parents=True)
    crop_src.write_bytes(b"fake image bytes")
    crop_hash = sha256_digest(crop_src)

    official_visual = tmp_path / "official_visual.jsonl"
    official_understanding = tmp_path / "official_understanding.jsonl"
    candidate_visual = tmp_path / "candidate_visual.jsonl"
    candidate_understanding = tmp_path / "candidate_understanding.jsonl"
    approved = tmp_path / "approved.jsonl"
    out_visual = tmp_path / "out_visual.jsonl"
    out_understanding = tmp_path / "out_understanding.jsonl"
    stable_crop_dir = tmp_path / "stable_crops"

    write_jsonl(official_visual, [visual_row("safe")])
    write_jsonl(official_understanding, [])
    write_jsonl(candidate_visual, [visual_row("safe", str(crop_src), crop_hash)])
    write_jsonl(candidate_understanding, [understanding_row("safe")])
    write_jsonl(approved, [approval_row("safe", crop_src, crop_hash)])

    summary = apply_approved_promotions(
        approved_plan_path=approved,
        candidate_visual_path=candidate_visual,
        candidate_understanding_path=candidate_understanding,
        official_visual_path=official_visual,
        official_understanding_path=official_understanding,
        out_visual_path=out_visual,
        out_understanding_path=out_understanding,
        stable_crop_dir=stable_crop_dir,
    )

    assert summary["approved_count"] == 1
    assert summary["promoted_count"] == 1
    promoted_visual = read_jsonl(out_visual)[0]
    promoted_understanding = read_jsonl(out_understanding)[0]
    assert promoted_visual["item_id"] == "safe"
    assert promoted_visual["crop_tier"] == "item_crop_candidate"
    assert promoted_visual["crop_path"].startswith(str(stable_crop_dir))
    assert "/tmp/" not in promoted_visual["crop_path"]
    assert Path(promoted_visual["crop_path"]).read_bytes() == b"fake image bytes"
    assert promoted_understanding["item_id"] == "safe"
    assert promoted_understanding["input_image_path"] == promoted_visual["crop_path"]


def test_apply_approved_promotions_excludes_manual_review_rows(tmp_path: Path):
    from scripts.promote_visual_candidate_approvals import apply_approved_promotions

    crop_src = tmp_path / "candidate_crop.png"
    crop_src.write_bytes(b"fake image bytes")
    crop_hash = sha256_digest(crop_src)

    official_visual = tmp_path / "official_visual.jsonl"
    official_understanding = tmp_path / "official_understanding.jsonl"
    candidate_visual = tmp_path / "candidate_visual.jsonl"
    candidate_understanding = tmp_path / "candidate_understanding.jsonl"
    approved = tmp_path / "approved.jsonl"

    write_jsonl(official_visual, [visual_row("safe"), visual_row("manual")])
    write_jsonl(official_understanding, [])
    write_jsonl(
        candidate_visual,
        [
            visual_row("safe", str(crop_src), crop_hash),
            visual_row("manual", str(crop_src), crop_hash),
        ],
    )
    write_jsonl(candidate_understanding, [understanding_row("safe"), understanding_row("manual")])
    write_jsonl(
        approved,
        [
            approval_row("safe", crop_src, crop_hash),
            {**approval_row("manual", crop_src, crop_hash), "status": "manual_review"},
        ],
    )

    summary = apply_approved_promotions(
        approved_plan_path=approved,
        candidate_visual_path=candidate_visual,
        candidate_understanding_path=candidate_understanding,
        official_visual_path=official_visual,
        official_understanding_path=official_understanding,
        out_visual_path=tmp_path / "out_visual.jsonl",
        out_understanding_path=tmp_path / "out_understanding.jsonl",
        stable_crop_dir=tmp_path / "stable_crops",
    )

    assert summary["approved_count"] == 1
    assert summary["skipped_non_approved_count"] == 1
    rows = read_jsonl(tmp_path / "out_visual.jsonl")
    by_id = {row["item_id"]: row for row in rows}
    assert by_id["safe"]["crop_path"]
    assert by_id["manual"]["crop_path"] is None


def test_apply_approved_promotions_rejects_missing_crop_source(tmp_path: Path):
    from scripts.promote_visual_candidate_approvals import apply_approved_promotions

    missing_crop = tmp_path / "missing.png"
    official_visual = tmp_path / "official_visual.jsonl"
    official_understanding = tmp_path / "official_understanding.jsonl"
    candidate_visual = tmp_path / "candidate_visual.jsonl"
    candidate_understanding = tmp_path / "candidate_understanding.jsonl"
    approved = tmp_path / "approved.jsonl"

    write_jsonl(official_visual, [visual_row("broken")])
    write_jsonl(official_understanding, [])
    write_jsonl(candidate_visual, [visual_row("broken", str(missing_crop), "sha256:missing")])
    write_jsonl(candidate_understanding, [understanding_row("broken")])
    write_jsonl(approved, [approval_row("broken", missing_crop, "sha256:missing")])

    try:
        apply_approved_promotions(
            approved_plan_path=approved,
            candidate_visual_path=candidate_visual,
            candidate_understanding_path=candidate_understanding,
            official_visual_path=official_visual,
            official_understanding_path=official_understanding,
            out_visual_path=tmp_path / "out_visual.jsonl",
            out_understanding_path=tmp_path / "out_understanding.jsonl",
            stable_crop_dir=tmp_path / "stable_crops",
        )
    except ValueError as exc:
        assert "missing_candidate_crop_file" in str(exc)
    else:
        raise AssertionError("Expected missing crop source to fail promotion")


def test_apply_approved_promotions_can_apply_structured_transcript_without_crop(tmp_path: Path):
    from scripts.promote_visual_candidate_approvals import apply_approved_promotions

    official_visual = tmp_path / "official_visual.jsonl"
    official_understanding = tmp_path / "official_understanding.jsonl"
    candidate_visual = tmp_path / "candidate_visual.jsonl"
    candidate_understanding = tmp_path / "candidate_understanding.jsonl"
    approved = tmp_path / "approved.jsonl"
    out_visual = tmp_path / "out_visual.jsonl"
    out_understanding = tmp_path / "out_understanding.jsonl"

    write_jsonl(official_visual, [visual_row("transcript")])
    write_jsonl(official_understanding, [])
    write_jsonl(candidate_visual, [visual_row("transcript")])
    write_jsonl(
        candidate_understanding,
        [
            {
                **understanding_row("transcript"),
                "input_image_path": "/tmp/page.jpg",
                "visual_evidence_mode": "structured_transcript",
                "transcript_supported_strong": True,
            }
        ],
    )
    write_jsonl(
        approved,
        [
            {
                "item_id": "transcript",
                "status": "approved",
                "promotion_scope": "understanding_structured_transcript",
                "category": "chart_curve",
                "review_reasons": [],
                "reject_reasons": [],
                "candidate_crop_path": "",
                "candidate_crop_hash": "",
                "additional_image_paths": [],
                "candidate_repair": {},
            }
        ],
    )

    summary = apply_approved_promotions(
        approved_plan_path=approved,
        candidate_visual_path=candidate_visual,
        candidate_understanding_path=candidate_understanding,
        official_visual_path=official_visual,
        official_understanding_path=official_understanding,
        out_visual_path=out_visual,
        out_understanding_path=out_understanding,
        stable_crop_dir=tmp_path / "stable_crops",
    )

    assert summary["approved_count"] == 1
    assert summary["transcript_promoted_count"] == 1
    promoted_visual = read_jsonl(out_visual)[0]
    promoted_understanding = read_jsonl(out_understanding)[0]
    assert promoted_visual["crop_path"] is None
    assert promoted_visual["crop_hash"] is None
    assert promoted_understanding["item_id"] == "transcript"
    assert promoted_understanding["visual_evidence_mode"] == "structured_transcript"
    assert promoted_understanding["transcript_supported_strong"] is True
    assert promoted_understanding["promotion_status"] == "official_promoted_structured_transcript"


if __name__ == "__main__":
    import tempfile

    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            with tempfile.TemporaryDirectory() as directory:
                test(Path(directory))
            passed += 1
            print(f"PASS {test.__name__}")
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
    print(f"{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
