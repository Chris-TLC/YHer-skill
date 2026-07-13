from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

import analysis.prepare as prepare_module
from analysis.dataset import DatasetContractError
from analysis.prepare import prepare_journey, validate_frozen_manifest
from analysis.views import rebuild_views_from_events


def _raw_journey() -> dict[str, object]:
    persona = "confirmatory:T:P:matched:0:A"
    journey: dict[str, object] = {
        "record_type": "confirmatory_journey",
        "target_node": "T",
        "truth": "P",
        "condition": "matched",
        "replicate": 0,
        "arm": "A",
        "persona_id": persona,
        "terminal_reason": "confidence",
        "h1_h2_eligible": True,
        "held_out_outcomes": {"h1": True, "h2": False},
        "events": [
            {
                "target_node": "T",
                "truth": "P",
                "condition": "matched",
                "replicate": 0,
                "arm": "A",
                "persona_id": persona,
                "position": position,
                "response_noise": 0.1 * position,
                "posterior_belief": [0.1, 0.7, 0.1, 0.1],
                "production_should_stop": position == 4,
                "production_confidence_should_stop": position == 4,
                "item_id": f"item-{position}",
                "family_id": f"family-{position}",
                "item_type": "mcq" if position % 2 else "numeric",
                "generator_probability": 0.4,
                "production_correct_probabilities": [0.9, 0.3, 0.6, 0.2],
                "role": "prereq" if position == 1 else "local",
            }
            for position in range(1, 5)
        ],
    }
    canonical = rebuild_views_from_events(journey)
    journey["views"] = [
        {
            **view,
            "common_support_no_repeat": budget != 25,
            "common_support_set_sha256": f"hash-{budget}",
            "held_out_outcomes": {"h1": True, "h2": False},
            "held_out_family_scores": [
                {
                    "family_id": "held-1",
                    "item_id": "h1",
                    "outcome": True,
                    "p_hat": 0.8,
                    "squared_error": 0.04,
                },
                {
                    "family_id": "held-2",
                    "item_id": "h2",
                    "outcome": False,
                    "p_hat": 0.3,
                    "squared_error": 0.09,
                },
            ],
            "held_out_brier": 0.065,
        }
        for view, budget in zip(canonical, (9, 15, 25), strict=True)
    ]
    return journey


def test_prepare_journey_validates_raw_views_then_keeps_only_compact_audit_data() -> None:
    prepared = prepare_journey(_raw_journey())

    assert len(prepared["analysis_rows"]) == 3
    assert prepared["analysis_rows"][0].budget == 9
    assert prepared["analysis_rows"][0].held_out_brier == pytest.approx(0.065)
    assert prepared["analysis_rows"][0].common_support_no_repeat is True
    assert prepared["analysis_rows"][0].prerequisite_count == 1
    assert prepared["analysis_rows"][0].prerequisite_share == pytest.approx(0.25)
    assert prepared["analysis_rows"][0].direct_count == 3
    assert prepared["analysis_rows"][0].unique_item_count == 4
    assert prepared["analysis_rows"][0].unique_family_count == 4
    assert prepared["events"][0] == {
        "target_node": "T",
        "truth": "P",
        "condition": "matched",
        "replicate": 0,
        "arm": "A",
        "persona_id": "confirmatory:T:P:matched:0:A",
        "position": 1,
        "response_noise": 0.1,
        "item_id": "item-1",
        "family_id": "family-1",
    }
    assert prepared["views"][2] == {
        "nominal_budget": 25,
        "common_support_no_repeat": False,
        "common_support_set_sha256": "hash-25",
    }
    assert prepared["analysis_events"][0].item_type == "mcq"
    assert prepared["analysis_events"][0].generator_probability == 0.4
    assert prepared["analysis_events"][0].production_probability == 0.3
    assert prepared["analysis_events"][0].valid is True
    assert prepared["held_out_pairs"] == (
        ("held-1", "h1"),
        ("held-2", "h2"),
    )
    assert "provenance" not in prepared


def test_frozen_manifest_requires_complete_32400_journey_contract() -> None:
    valid = {
        "status": "complete",
        "full_grid_complete": True,
        "expected_journey_count": 32400,
        "selected_shard_count": 216,
        "full_shard_count": 216,
        "bootstrap_seed": 2026071301,
        "validation": {
            "expected_journeys": 32400,
            "replicates": 50,
            "arms": 3,
            "conditions": 2,
            "truth_states": 4,
            "open_nodes": 27,
        },
    }
    validate_frozen_manifest(valid)

    invalid = {**valid, "expected_journey_count": 32399}
    with pytest.raises(DatasetContractError, match="32,400"):
        validate_frozen_manifest(invalid)


def test_formal_manifest_requires_the_exact_frozen_identity_and_shard_topology() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "data/sim_store/confirmatory/confirmatory-v1/manifest.json"
    payload = path.read_bytes()
    manifest = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()

    validate_frozen_manifest(
        manifest,
        manifest_sha256=digest,
        repo_root=repo_root,
    )

    with pytest.raises(DatasetContractError, match="canonical manifest SHA-256"):
        validate_frozen_manifest(
            manifest,
            manifest_sha256="0" * 64,
            repo_root=repo_root,
        )

    changed = {**manifest, "run_id": "other-run"}
    with pytest.raises(DatasetContractError, match="run_id"):
        validate_frozen_manifest(
            changed,
            manifest_sha256=digest,
            repo_root=repo_root,
        )

    changed = {**manifest, "validation": dict(manifest["validation"])}
    changed["validation"]["config_sha256"] = "0" * 64
    with pytest.raises(DatasetContractError, match="config"):
        validate_frozen_manifest(
            changed,
            manifest_sha256=digest,
            repo_root=repo_root,
        )


def test_frozen_config_uses_canonical_json_hash_and_actual_tag_binding(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "experiments/config"
    config_dir.mkdir(parents=True)
    config = config_dir / "confirmatory_v1.json"
    shutil.copyfile(repo_root / "experiments/config/confirmatory_v1.json", config)
    manifest = {
        "config_sha256": prepare_module.FROZEN_CONFIG_SHA256,
        "validation": {"config_sha256": prepare_module.FROZEN_CONFIG_SHA256},
    }

    prepare_module._validate_frozen_config(tmp_path, manifest)
    prepare_module._validate_annotated_tag(repo_root)

    changed = json.loads(config.read_text(encoding="utf-8"))
    changed["max_items"] = 24
    config.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(DatasetContractError, match="config"):
        prepare_module._validate_frozen_config(tmp_path, manifest)
