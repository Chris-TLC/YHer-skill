#!/usr/bin/env python3
"""Tests for item quality gating and repository purpose filtering."""

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


def test_quality_gate_blocks_missing_manifest_for_diagnosis(tmp_path: Path):
    from core.data.item_quality import ItemQualityGate

    gate = ItemQualityGate(tmp_path / "missing.jsonl")
    decision = gate.evaluate({"item_id": "i1"}, purpose="diagnosis")

    assert not decision.allowed
    assert "missing_quality_manifest" in decision.blockers


def test_quality_gate_allows_text_item_and_blocks_weak_visual_profile(tmp_path: Path):
    from core.data.item_quality import ItemQualityGate

    manifest = tmp_path / "item_quality_manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "item_id": "text_ok",
                "needs_image": False,
                "visual_asset_status": "not_required",
                "readability_status": "pass",
                "llm_understanding_status": "not_required",
                "answer_status": "verified",
                "rubric_status": "complete",
                "usable_for_diagnosis": True,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": True,
                "blocker_reasons": [],
            },
            {
                "item_id": "visual_weak",
                "needs_image": True,
                "visual_asset_status": "weak",
                "readability_status": "manual_review",
                "llm_understanding_status": "weak",
                "answer_status": "verified",
                "rubric_status": "complete",
                "page_image_path": "/tmp/weak_page.jpg",
                "page_image_hash": "sha256:weak",
                "usable_for_diagnosis": False,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": False,
                "blocker_reasons": [],
            },
            {
                "item_id": "visual_blocked",
                "needs_image": True,
                "visual_asset_status": "weak",
                "readability_status": "manual_review",
                "llm_understanding_status": "weak",
                "answer_status": "verified",
                "rubric_status": "complete",
                "page_image_path": "/tmp/blocked_page.jpg",
                "page_image_hash": "sha256:blocked",
                "usable_for_diagnosis": False,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": False,
                "blocker_reasons": ["page_mismatch"],
            },
        ],
    )

    gate = ItemQualityGate(manifest)
    assert gate.evaluate({"item_id": "text_ok"}, purpose="diagnosis").allowed

    weak_profile = gate.evaluate({"item_id": "visual_weak"}, purpose="profile_evidence")
    assert not weak_profile.allowed
    assert "visual_asset_not_strong" in weak_profile.blockers

    weak_practice = gate.evaluate({"item_id": "visual_weak"}, purpose="practice")
    assert weak_practice.allowed

    blocked_practice = gate.evaluate({"item_id": "visual_blocked"}, purpose="practice")
    assert not blocked_practice.allowed
    assert "page_mismatch" in blocked_practice.blockers


def test_quality_gate_requires_strong_for_diagnosis_profile_and_student_readable_for_practice(tmp_path: Path):
    from core.data.item_quality import ItemQualityGate

    manifest = tmp_path / "item_quality_manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "item_id": "student_only",
                "needs_image": True,
                "student_readable": True,
                "strong": False,
                "visual_pipeline_stage": "student_readable",
                "visual_asset_status": "strong",
                "readability_status": "pass",
                "llm_understanding_status": "weak",
                "answer_status": "verified",
                "rubric_status": "complete",
                "page_image_path": "/tmp/student_only_page.jpg",
                "page_image_hash": "sha256:studentonly",
                "usable_for_diagnosis": True,
                "usable_for_practice": False,
                "usable_for_teaching": False,
                "usable_for_profile_evidence": True,
                "blocker_reasons": [],
            },
            {
                "item_id": "not_readable",
                "needs_image": True,
                "student_readable": False,
                "strong": False,
                "visual_pipeline_stage": "asset_linked",
                "visual_asset_status": "weak",
                "readability_status": "manual_review",
                "llm_understanding_status": "weak",
                "answer_status": "verified",
                "rubric_status": "complete",
                "page_image_path": "/tmp/not_readable_page.jpg",
                "page_image_hash": "sha256:notreadable",
                "usable_for_diagnosis": False,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": False,
                "blocker_reasons": [],
            },
            {
                "item_id": "visual_strong",
                "needs_image": True,
                "student_readable": True,
                "strong": True,
                "visual_pipeline_stage": "strong",
                "visual_asset_status": "strong",
                "readability_status": "pass",
                "llm_understanding_status": "strong",
                "answer_status": "verified",
                "rubric_status": "complete",
                "source_file": "paper.pdf",
                "page": 2,
                "page_image_path": "/tmp/strong_page.jpg",
                "page_image_hash": "sha256:strongpage",
                "crop_path": "/tmp/strong_crop.png",
                "crop_hash": "sha256:strongcrop",
                "vl_model": "qwen3-vl-plus",
                "usable_for_diagnosis": True,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": True,
                "blocker_reasons": [],
            },
        ],
    )

    gate = ItemQualityGate(manifest)

    student_only_diag = gate.evaluate({"item_id": "student_only"}, purpose="diagnosis")
    assert not student_only_diag.allowed
    assert "not_strong" in student_only_diag.blockers

    student_only_profile = gate.evaluate({"item_id": "student_only"}, purpose="profile_evidence")
    assert not student_only_profile.allowed
    assert "not_strong" in student_only_profile.blockers

    assert gate.evaluate({"item_id": "student_only"}, purpose="practice").allowed
    assert gate.evaluate({"item_id": "student_only"}, purpose="teaching").allowed

    not_readable_practice = gate.evaluate({"item_id": "not_readable"}, purpose="practice")
    assert not not_readable_practice.allowed
    assert "not_student_readable" in not_readable_practice.blockers

    assert gate.evaluate({"item_id": "visual_strong"}, purpose="diagnosis").allowed
    assert gate.evaluate({"item_id": "visual_strong"}, purpose="profile_evidence").allowed


def test_quality_gate_defends_against_inconsistent_student_readable_and_strong_rows(tmp_path: Path):
    from core.data.item_quality import ItemQualityGate

    manifest = tmp_path / "item_quality_manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "item_id": "student_readable_without_image",
                "needs_image": True,
                "student_readable": True,
                "strong": False,
                "visual_asset_status": "strong",
                "readability_status": "pass",
                "llm_understanding_status": "weak",
                "answer_status": "verified",
                "rubric_status": "complete",
                "blocker_reasons": [],
            },
            {
                "item_id": "strong_without_evidence_chain",
                "needs_image": True,
                "student_readable": True,
                "strong": True,
                "visual_asset_status": "strong",
                "readability_status": "pass",
                "llm_understanding_status": "strong",
                "answer_status": "verified",
                "rubric_status": "complete",
                "page_image_path": "/tmp/page.jpg",
                "page_image_hash": "sha256:page",
                "blocker_reasons": [],
            },
        ],
    )
    gate = ItemQualityGate(manifest)

    practice = gate.evaluate({"item_id": "student_readable_without_image"}, purpose="practice")
    assert not practice.allowed
    assert "missing_student_image_evidence" in practice.blockers

    profile = gate.evaluate({"item_id": "strong_without_evidence_chain"}, purpose="profile_evidence")
    assert not profile.allowed
    assert "missing_crop_evidence_for_strong" in profile.blockers
    assert "missing_vl_result" in profile.blockers


def test_item_repository_filters_by_purpose_and_debug_all(tmp_path: Path):
    from core.data.item_repository import ItemRepository

    bank = tmp_path / "item_bank"
    write_jsonl(
        bank / "sample.jsonl",
        [
            {"item_id": "text_ok", "kg_nodes": ["化学平衡"], "question_type": "选择题"},
            {"item_id": "visual_weak", "kg_nodes": ["化学平衡"], "question_type": "选择题"},
        ],
    )
    quality = tmp_path / "quality" / "item_quality_manifest.jsonl"
    write_jsonl(
        quality,
        [
            {
                "item_id": "text_ok",
                "needs_image": False,
                "visual_asset_status": "not_required",
                "readability_status": "pass",
                "llm_understanding_status": "not_required",
                "answer_status": "verified",
                "rubric_status": "complete",
                "usable_for_diagnosis": True,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": True,
                "blocker_reasons": [],
            },
            {
                "item_id": "visual_weak",
                "needs_image": True,
                "visual_asset_status": "weak",
                "readability_status": "manual_review",
                "llm_understanding_status": "weak",
                "answer_status": "verified",
                "rubric_status": "complete",
                "page_image_path": "/tmp/visual_weak.jpg",
                "page_image_hash": "sha256:visualweak",
                "usable_for_diagnosis": False,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": False,
                "blocker_reasons": [],
            },
        ],
    )

    repo = ItemRepository(bank_dir=bank, quality_manifest_path=quality)
    diagnosis_ids = [item["item_id"] for item in repo.find_items(kg_node="化学平衡", purpose="diagnosis", limit=10)]
    profile_ids = [item["item_id"] for item in repo.find_items(kg_node="化学平衡", purpose="profile_evidence", limit=10)]
    practice_ids = [item["item_id"] for item in repo.find_items(kg_node="化学平衡", purpose="practice", limit=10)]
    debug_ids = [item["item_id"] for item in repo.find_items(kg_node="化学平衡", purpose="debug_all", limit=10)]

    assert diagnosis_ids == ["text_ok"]
    assert profile_ids == ["text_ok"]
    assert practice_ids == ["text_ok", "visual_weak"]
    assert debug_ids == ["text_ok", "visual_weak"]


def test_item_repository_default_purpose_is_safe_for_student_flows(tmp_path: Path):
    from core.data.item_repository import ItemRepository

    bank = tmp_path / "item_bank"
    write_jsonl(
        bank / "sample.jsonl",
        [
            {"item_id": "text_ok", "kg_nodes": ["化学平衡"], "question_type": "选择题"},
            {"item_id": "visual_weak", "kg_nodes": ["化学平衡"], "question_type": "选择题"},
        ],
    )
    quality = tmp_path / "quality" / "item_quality_manifest.jsonl"
    write_jsonl(
        quality,
        [
            {
                "item_id": "text_ok",
                "needs_image": False,
                "visual_asset_status": "not_required",
                "readability_status": "pass",
                "llm_understanding_status": "not_required",
                "answer_status": "verified",
                "rubric_status": "complete",
                "usable_for_diagnosis": True,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": True,
                "blocker_reasons": [],
            },
            {
                "item_id": "visual_weak",
                "needs_image": True,
                "visual_asset_status": "weak",
                "readability_status": "manual_review",
                "llm_understanding_status": "weak",
                "answer_status": "verified",
                "rubric_status": "complete",
                "page_image_path": "/tmp/visual_weak.jpg",
                "page_image_hash": "sha256:visualweak",
                "usable_for_diagnosis": False,
                "usable_for_practice": True,
                "usable_for_teaching": True,
                "usable_for_profile_evidence": False,
                "blocker_reasons": [],
            },
        ],
    )

    repo = ItemRepository(bank_dir=bank, quality_manifest_path=quality)
    default_ids = [item["item_id"] for item in repo.find_items(kg_node="化学平衡", limit=10)]
    debug_ids = [item["item_id"] for item in repo.find_items(kg_node="化学平衡", purpose="debug_all", limit=10)]

    assert default_ids == ["text_ok"]
    assert debug_ids == ["text_ok", "visual_weak"]


def test_build_quality_manifest_derives_text_and_visual_purpose_flags(tmp_path: Path):
    from scripts.build_item_quality_manifest import build_quality_manifest

    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    write_jsonl(
        item_bank,
        [
            {
                "item_id": "text_ok",
                "stem": "纯文本题干",
                "options": {"A": "甲", "B": "乙"},
                "confidence": 0.9,
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "A", "final_answers": ["A"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            },
            {
                "item_id": "visual_strong",
                "stem": "如图所示实验装置题",
                "confidence": 0.9,
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "B", "final_answers": ["B"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            },
            {
                "item_id": "visual_weak",
                "stem": "如图所示流程题",
                "confidence": 0.9,
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "C", "final_answers": ["C"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            },
        ],
    )
    write_jsonl(
        visual_manifest,
        [
            {
                "item_id": "visual_strong",
                "match_tier": "strong",
                "crop_tier": "page_only",
                "page_image_path": "/tmp/p001.jpg",
                "page_image_hash": "sha256:abc",
                "blocker_reasons": [],
            },
            {
                "item_id": "visual_weak",
                "match_tier": "weak",
                "crop_tier": "page_only",
                "page_image_path": "/tmp/p002.jpg",
                "page_image_hash": "sha256:def",
                "blocker_reasons": ["page_mismatch"],
            },
        ],
    )

    rows, summary = build_quality_manifest(item_bank_path=item_bank, visual_manifest_path=visual_manifest)
    by_id = {row["item_id"]: row for row in rows}

    assert summary["total_items"] == 3
    assert by_id["text_ok"]["usable_for_diagnosis"]
    assert by_id["text_ok"]["usable_for_profile_evidence"]
    assert by_id["visual_strong"]["needs_image"]
    assert by_id["visual_strong"]["visual_asset_status"] == "strong"
    assert by_id["visual_strong"]["readability_status"] == "pass"
    assert by_id["visual_strong"]["llm_understanding_status"] == "weak"
    assert not by_id["visual_strong"]["usable_for_diagnosis"]
    assert by_id["visual_strong"]["usable_for_practice"]
    assert not by_id["visual_strong"]["usable_for_profile_evidence"]
    assert "page_mismatch" in by_id["visual_weak"]["blocker_reasons"]
    assert not by_id["visual_weak"]["usable_for_practice"]


def test_build_quality_manifest_uses_strong_visual_understanding_results(tmp_path: Path):
    from scripts.build_item_quality_manifest import build_quality_manifest

    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    understanding_results = tmp_path / "visual_understanding_results.jsonl"
    write_jsonl(
        item_bank,
        [
            {
                "item_id": "visual_strong",
                "stem": "如图所示实验装置题",
                "confidence": 0.9,
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "B", "final_answers": ["B"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            }
        ],
    )
    write_jsonl(
        visual_manifest,
        [
            {
                "item_id": "visual_strong",
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 3,
                "match_tier": "strong",
                "crop_tier": "item_crop_candidate",
                "page_image_path": "/tmp/p001.jpg",
                "page_image_hash": "sha256:abc",
                "crop_path": "/tmp/c001.png",
                "crop_hash": "sha256:cropabc",
                "blocker_reasons": [],
            }
        ],
    )
    write_jsonl(
        understanding_results,
        [
            {
                "item_id": "visual_strong",
                "model": "qwen3-vl-plus",
                "understanding_pass": True,
                "visible_pass": True,
                "answer_match": True,
                "confidence": 0.91,
                "error_types": [],
            }
        ],
    )

    rows, _ = build_quality_manifest(
        item_bank_path=item_bank,
        visual_manifest_path=visual_manifest,
        understanding_results_path=understanding_results,
    )

    row = rows[0]
    assert row["llm_understanding_status"] == "strong"
    assert row["usable_for_diagnosis"]
    assert row["usable_for_profile_evidence"]


def test_build_quality_manifest_blocks_page_display_answer_leak(tmp_path: Path):
    from scripts.build_item_quality_manifest import build_quality_manifest

    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    understanding_results = tmp_path / "visual_understanding_results.jsonl"
    quarantine_list = tmp_path / "quarantine_list.txt"
    leaky_page = tmp_path / "leaky_page.jpg"
    write_jsonl(
        item_bank,
        [
            {
                "item_id": "leaky_page_item",
                "stem": "如图所示实验装置题",
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "B", "final_answers": ["B"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            }
        ],
    )
    write_jsonl(
        visual_manifest,
        [
            {
                "item_id": "leaky_page_item",
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 3,
                "match_tier": "strong",
                "crop_tier": "page_only",
                "page_image_path": str(leaky_page),
                "page_image_hash": "sha256:leakypage",
                "blocker_reasons": [],
            }
        ],
    )
    write_jsonl(
        understanding_results,
        [
            {
                "item_id": "leaky_page_item",
                "model": "qwen3-vl-plus",
                "understanding_pass": True,
                "visible_pass": True,
                "answer_match": True,
                "confidence": 0.91,
                "error_types": [],
            }
        ],
    )
    quarantine_list.write_text(str(leaky_page) + "\n", encoding="utf-8")

    rows, summary = build_quality_manifest(
        item_bank_path=item_bank,
        visual_manifest_path=visual_manifest,
        understanding_results_path=understanding_results,
        display_answer_leak_quarantine_path=quarantine_list,
    )

    row = rows[0]
    assert "display_answer_leak" in row["blocker_reasons"]
    assert not row["student_readable"]
    assert not row["usable_for_diagnosis"]
    assert not row["usable_for_practice"]
    assert not row["usable_for_teaching"]
    assert summary["blocker_reasons"]["display_answer_leak"] == 1


def test_build_quality_manifest_blocks_crop_display_answer_leak(tmp_path: Path):
    from scripts.build_item_quality_manifest import build_quality_manifest

    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    understanding_results = tmp_path / "visual_understanding_results.jsonl"
    quarantine_list = tmp_path / "quarantine_list.txt"
    clean_page = tmp_path / "clean_page.jpg"
    leaky_crop = tmp_path / "leaky_crop.png"
    write_jsonl(
        item_bank,
        [
            {
                "item_id": "leaky_crop_item",
                "stem": "如图所示曲线题",
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "A", "final_answers": ["A"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            }
        ],
    )
    write_jsonl(
        visual_manifest,
        [
            {
                "item_id": "leaky_crop_item",
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 2,
                "match_tier": "strong",
                "crop_tier": "item_crop_candidate",
                "page_image_path": str(clean_page),
                "page_image_hash": "sha256:cleanpage",
                "crop_path": str(leaky_crop),
                "crop_hash": "sha256:leakycrop",
                "blocker_reasons": [],
            }
        ],
    )
    write_jsonl(
        understanding_results,
        [
            {
                "item_id": "leaky_crop_item",
                "model": "qwen3-vl-plus",
                "understanding_pass": True,
                "visible_pass": True,
                "answer_match": True,
                "confidence": 0.95,
                "error_types": [],
            }
        ],
    )
    quarantine_list.write_text(str(leaky_crop) + "\n", encoding="utf-8")

    rows, summary = build_quality_manifest(
        item_bank_path=item_bank,
        visual_manifest_path=visual_manifest,
        understanding_results_path=understanding_results,
        display_answer_leak_quarantine_path=quarantine_list,
    )

    row = rows[0]
    assert row["display_image_path"] == str(leaky_crop)
    assert "display_answer_leak" in row["blocker_reasons"]
    assert "blocker_reasons_present" in row["strong_blocker_reasons"]
    assert not row["strong"]
    assert not row["usable_for_practice"]
    assert summary["display_answer_leak"] == 1


def test_build_quality_manifest_uses_explicit_answer_verification_overrides_only(tmp_path: Path):
    from scripts.build_item_quality_manifest import build_quality_manifest

    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    understanding_results = tmp_path / "visual_understanding_results.jsonl"
    answer_overrides = tmp_path / "answer_verification_overrides.jsonl"
    items = []
    visuals = []
    understandings = []
    for item_id in ["reviewed_ok", "still_suspect", "invalid_override"]:
        items.append(
            {
                "item_id": item_id,
                "stem": "如图所示实验装置题",
                "verification_status": "needs_review",
                "standard_solution": {"standard_answer": "B", "final_answers": ["B"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            }
        )
        visuals.append(
            {
                "item_id": item_id,
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 3,
                "match_tier": "strong",
                "crop_tier": "item_crop_candidate",
                "page_image_path": f"/tmp/{item_id}_page.jpg",
                "page_image_hash": f"sha256:{item_id}page",
                "crop_path": f"/tmp/{item_id}_crop.png",
                "crop_hash": f"sha256:{item_id}crop",
                "blocker_reasons": [],
            }
        )
        understandings.append(
            {
                "item_id": item_id,
                "model": "gemini-3.1-pro-preview",
                "understanding_pass": True,
                "visible_pass": True,
                "answer_match": True,
                "confidence": 0.94,
                "error_types": [],
            }
        )
    write_jsonl(item_bank, items)
    write_jsonl(visual_manifest, visuals)
    write_jsonl(understanding_results, understandings)
    write_jsonl(
        answer_overrides,
        [
            {
                "item_id": "reviewed_ok",
                "answer_status": "verified",
                "review_decision": "standard_answer_verified",
                "review_source": "answer_equivalence_review",
            },
            {
                "item_id": "invalid_override",
                "answer_status": "verified",
                "review_decision": "uncertain",
                "review_source": "answer_equivalence_review",
            },
        ],
    )

    rows, _ = build_quality_manifest(
        item_bank_path=item_bank,
        visual_manifest_path=visual_manifest,
        understanding_results_path=understanding_results,
        answer_verification_overrides_path=answer_overrides,
    )
    by_id = {row["item_id"]: row for row in rows}

    assert by_id["reviewed_ok"]["answer_status"] == "verified"
    assert by_id["reviewed_ok"]["strong"]
    assert by_id["reviewed_ok"]["answer_review_source"] == "answer_equivalence_review"

    assert by_id["still_suspect"]["answer_status"] == "suspect"
    assert not by_id["still_suspect"]["strong"]
    assert "answer_not_verified" in by_id["still_suspect"]["strong_blocker_reasons"]

    assert by_id["invalid_override"]["answer_status"] == "suspect"
    assert not by_id["invalid_override"]["strong"]


def test_build_quality_manifest_keeps_vl_answered_page_only_item_out_of_strong(tmp_path: Path):
    from scripts.build_item_quality_manifest import build_quality_manifest

    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    understanding_results = tmp_path / "visual_understanding_results.jsonl"
    write_jsonl(
        item_bank,
        [
            {
                "item_id": "vl_answered_page_only",
                "stem": "如图所示晶胞题",
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "4", "final_answers": ["4"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            }
        ],
    )
    write_jsonl(
        visual_manifest,
        [
            {
                "item_id": "vl_answered_page_only",
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 1,
                "match_tier": "strong",
                "crop_tier": "page_only",
                "page_image_path": "/tmp/p001.jpg",
                "page_image_hash": "sha256:abc",
                "crop_path": "",
                "crop_hash": "",
                "blocker_reasons": [],
            }
        ],
    )
    write_jsonl(
        understanding_results,
        [
            {
                "item_id": "vl_answered_page_only",
                "model": "qwen3-vl-plus",
                "understanding_pass": True,
                "visible_pass": True,
                "answer_match": True,
                "confidence": 0.91,
                "error_types": [],
            }
        ],
    )

    rows, _ = build_quality_manifest(
        item_bank_path=item_bank,
        visual_manifest_path=visual_manifest,
        understanding_results_path=understanding_results,
    )

    row = rows[0]
    assert row["student_readable"]
    assert not row["strong"]
    assert row["usable_for_practice"]
    assert not row["usable_for_profile_evidence"]
    assert row["visual_pipeline_stage"] == "strong_candidate"
    assert row["review_queue"] == "strong_review"
    assert "missing_crop_evidence_for_strong" in row["strong_blocker_reasons"]


def test_build_quality_manifest_allows_transcript_supported_visual_strong_without_crop(tmp_path: Path):
    from core.data.item_quality import ItemQualityGate
    from scripts.build_item_quality_manifest import build_quality_manifest

    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    understanding_results = tmp_path / "visual_understanding_results.jsonl"
    write_jsonl(
        item_bank,
        [
            {
                "item_id": "transcript_supported",
                "stem": "如图所示曲线题。（图：坐标系横轴为温度，纵轴为转化率；上方曲线标注p1，下方曲线标注p2。）",
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "p1<p2", "final_answers": ["p1<p2"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            }
        ],
    )
    write_jsonl(
        visual_manifest,
        [
            {
                "item_id": "transcript_supported",
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 7,
                "match_tier": "strong",
                "crop_tier": "page_only",
                "page_image_path": "/tmp/p007.jpg",
                "page_image_hash": "sha256:page007",
                "crop_path": "",
                "crop_hash": "",
                "blocker_reasons": [],
            }
        ],
    )
    write_jsonl(
        understanding_results,
        [
            {
                "item_id": "transcript_supported",
                "model": "gemini-3.1-pro-preview",
                "understanding_pass": True,
                "visible_pass": True,
                "answer_match": True,
                "confidence": 1.0,
                "error_types": [],
                "transcript_supported_strong": True,
                "visual_evidence_mode": "structured_transcript",
            }
        ],
    )

    rows, _ = build_quality_manifest(
        item_bank_path=item_bank,
        visual_manifest_path=visual_manifest,
        understanding_results_path=understanding_results,
    )
    manifest = tmp_path / "item_quality_manifest.jsonl"
    write_jsonl(manifest, rows)
    row = rows[0]

    assert row["strong"]
    assert row["visual_evidence_mode"] == "structured_transcript"
    assert "missing_crop_evidence_for_strong" not in row["strong_blocker_reasons"]
    assert ItemQualityGate(manifest).evaluate({"item_id": "transcript_supported"}, purpose="diagnosis").allowed
    assert ItemQualityGate(manifest).evaluate({"item_id": "transcript_supported"}, purpose="profile_evidence").allowed


def test_build_quality_manifest_emits_visual_state_machine_and_review_queue(tmp_path: Path):
    from scripts.build_item_quality_manifest import build_quality_manifest

    item_bank = tmp_path / "chemistry_v3_6695.jsonl"
    visual_manifest = tmp_path / "visual_asset_manifest.jsonl"
    understanding_results = tmp_path / "visual_understanding_results.jsonl"
    write_jsonl(
        item_bank,
        [
            {
                "item_id": "visual_student_only",
                "stem": "如图所示实验装置题",
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "B", "final_answers": ["B"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            },
            {
                "item_id": "visual_strong",
                "stem": "如图所示曲线题",
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "A", "final_answers": ["A"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            },
            {
                "item_id": "visual_missing_image",
                "stem": "如图所示流程题",
                "verification_status": "passed",
                "standard_solution": {"standard_answer": "C", "final_answers": ["C"]},
                "rubric": [{"point_id": "ans", "must_have": True}],
            },
        ],
    )
    write_jsonl(
        visual_manifest,
        [
            {
                "item_id": "visual_student_only",
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 3,
                "page_image_path": "/tmp/p003.jpg",
                "page_image_hash": "sha256:003",
                "crop_path": "",
                "crop_hash": "",
                "crop_tier": "page_only",
                "match_tier": "strong",
                "category": "experiment_device",
                "blocker_reasons": [],
            },
            {
                "item_id": "visual_strong",
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 4,
                "page_image_path": "/tmp/p004.jpg",
                "page_image_hash": "sha256:004",
                "crop_path": "/tmp/c004.png",
                "crop_hash": "sha256:c004",
                "crop_tier": "item_crop_candidate",
                "match_tier": "strong",
                "category": "chart_curve",
                "blocker_reasons": [],
            },
            {
                "item_id": "visual_missing_image",
                "source_file": "paper.pdf",
                "source_path": "/tmp/paper.pdf",
                "declared_page": 5,
                "page_image_path": "",
                "page_image_hash": "",
                "crop_tier": "missing",
                "match_tier": "reject",
                "category": "process_flow",
                "blocker_reasons": ["missing_page_image"],
            },
        ],
    )
    write_jsonl(
        understanding_results,
        [
            {
                "item_id": "visual_strong",
                "model": "qwen3-vl-plus",
                "visible_pass": True,
                "answer_match": True,
                "understanding_pass": True,
                "profile_evidence_allowed": True,
                "error_types": [],
            }
        ],
    )

    rows, summary = build_quality_manifest(
        item_bank_path=item_bank,
        visual_manifest_path=visual_manifest,
        understanding_results_path=understanding_results,
    )
    by_id = {row["item_id"]: row for row in rows}

    assert by_id["visual_student_only"]["student_readable"]
    assert not by_id["visual_student_only"]["strong"]
    assert by_id["visual_student_only"]["visual_pipeline_stage"] == "student_readable"
    assert by_id["visual_student_only"]["review_queue"] == "strong_review"
    assert by_id["visual_student_only"]["usable_for_practice"]
    assert not by_id["visual_student_only"]["usable_for_profile_evidence"]

    assert by_id["visual_strong"]["student_readable"]
    assert by_id["visual_strong"]["strong"]
    assert by_id["visual_strong"]["visual_pipeline_stage"] == "strong"
    assert by_id["visual_strong"]["review_queue"] == "none"
    assert by_id["visual_strong"]["source_file"] == "paper.pdf"
    assert by_id["visual_strong"]["page"] == 4
    assert by_id["visual_strong"]["page_image_hash"] == "sha256:004"
    assert by_id["visual_strong"]["crop_path"] == "/tmp/c004.png"
    assert by_id["visual_strong"]["vl_model"] == "qwen3-vl-plus"
    assert by_id["visual_strong"]["quality_evidence_id"].startswith("iq:")

    assert not by_id["visual_missing_image"]["student_readable"]
    assert by_id["visual_missing_image"]["visual_pipeline_stage"] == "raw_visual_item"
    assert by_id["visual_missing_image"]["review_queue"] == "quarantine"
    assert "missing_page_image" in by_id["visual_missing_image"]["blocker_reasons"]

    assert summary["student_readable"] == 2
    assert summary["strong"] == 1
    assert summary["visual_pipeline_stage"]["strong"] == 1


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
