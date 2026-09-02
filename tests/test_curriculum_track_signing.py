"""Contracts for evidence-backed curriculum track signing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.sign_curriculum_tracks import SigningError, build_signed_map, write_artifacts


def _draft() -> dict:
    return {
        "version": "curriculum_v1_draft",
        "subject": "chemistry",
        "tracks": ["foundation", "round1", "sprint", "topical", "scene"],
        "entities": [
            {
                "entity": "bv:exact",
                "name_hint": "基础课程",
                "track": "foundation",
                "confidence": "high",
                "needs_human": True,
                "reviewer": "",
                "note": "high fixture",
            },
            {
                "entity": "bv:mismatch",
                "name_hint": "草稿标题",
                "track": "sprint",
                "confidence": "high",
                "needs_human": True,
                "reviewer": "",
                "note": "mismatch fixture",
            },
            {
                "entity": "season:neutral",
                "name_hint": "中性课程",
                "track": "topical",
                "confidence": "low",
                "needs_human": True,
                "reviewer": "",
                "note": "low fixture",
            },
        ],
    }


def _catalog() -> list[dict]:
    return [
        {"entity_id": "bv:exact", "name": "基础课程", "subject": "chemistry"},
        {"entity_id": "bv:mismatch", "name": "真实标题", "subject": "chemistry"},
        {"entity_id": "season:neutral", "name": "中性课程", "subject": "chemistry"},
    ]


def test_only_exact_high_confidence_evidence_is_signed() -> None:
    signed, manifest = build_signed_map(
        _draft(),
        _catalog(),
        reviewer="codex_sol_20260713",
        reviewed_at="2026-07-13",
        catalog_source="seasons_catalog.jsonl",
        catalog_sha256="a" * 64,
        draft_sha256="b" * 64,
    )

    rows = {row["entity"]: row for row in signed["entities"]}
    assert rows["bv:exact"]["needs_human"] is False
    assert rows["bv:exact"]["reviewer"] == "codex_sol_20260713"
    assert rows["bv:exact"]["evidence"] == {
        "source_file": "seasons_catalog.jsonl",
        "source_line": 1,
        "catalog_entity_id": "bv:exact",
        "catalog_name": "基础课程",
        "match": "exact_entity_id_and_title",
        "reviewed_at": "2026-07-13",
    }
    assert rows["bv:mismatch"]["needs_human"] is True
    assert rows["bv:mismatch"]["reviewer"] == ""
    assert rows["bv:mismatch"]["neutral_reason"] == "catalog_title_mismatch"
    assert rows["season:neutral"]["needs_human"] is True
    assert rows["season:neutral"]["neutral_reason"] == "confidence_not_high"
    assert signed["version"] == "curriculum_v1_signed_20260713"
    assert signed["provenance"]["catalog_sha256"] == "a" * 64
    assert len(manifest) == 3
    assert {row["decision"] for row in manifest} == {"signed", "neutral"}


def test_duplicate_catalog_entity_is_rejected() -> None:
    catalog = [*_catalog(), _catalog()[0]]

    with pytest.raises(SigningError, match="duplicate catalog entity"):
        build_signed_map(
            _draft(),
            catalog,
            reviewer="codex_sol_20260713",
            reviewed_at="2026-07-13",
            catalog_source="catalog.jsonl",
            catalog_sha256="a" * 64,
            draft_sha256="b" * 64,
        )


def test_write_artifacts_preserves_one_manifest_row_per_entity(tmp_path: Path) -> None:
    signed, manifest = build_signed_map(
        _draft(),
        _catalog(),
        reviewer="codex_sol_20260713",
        reviewed_at="2026-07-13",
        catalog_source="catalog.jsonl",
        catalog_sha256="a" * 64,
        draft_sha256="b" * 64,
    )
    config = tmp_path / "config" / "track_map_v1.yaml"
    evidence = tmp_path / "delivery" / "track_signing_manifest.jsonl"
    rollback = tmp_path / "delivery" / "ROLLBACK.md"

    write_artifacts(
        signed_map=signed,
        evidence_manifest=manifest,
        config_path=config,
        manifest_path=evidence,
        rollback_path=rollback,
    )

    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    manifest_rows = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
    assert len(loaded["entities"]) == 3
    assert len(manifest_rows) == 3
    assert [row["entity"] for row in manifest_rows] == [
        "bv:exact",
        "bv:mismatch",
        "season:neutral",
    ]
    assert "rm config/curriculum/track_map_v1.yaml" in rollback.read_text(encoding="utf-8")


def test_versioned_product_config_has_43_evidence_backed_decisions() -> None:
    from engine.recommender import load_track_map

    repo_root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (repo_root / "config/curriculum/track_map_v1.yaml").read_text(encoding="utf-8")
    )
    loaded = load_track_map(raw)

    assert len(raw["entities"]) == 43
    assert len(loaded["entities"]) == 30
    assert len(loaded["neutral_entities"]) == 13
    assert all(
        row["reviewer"] == "codex_sol_20260713"
        and row["needs_human"] is False
        and row["evidence"]["match"] == "exact_entity_id_and_title"
        for row in raw["entities"]
        if row["entity"] in loaded["entities"]
    )
    assert all(
        row["reviewer"] == "" and row["needs_human"] is True
        for row in raw["entities"]
        if row["entity"] in loaded["neutral_entities"]
    )
