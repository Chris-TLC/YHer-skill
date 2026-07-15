"""Offline contract tests for the Persona v2 foundation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


def _anchors(count: int = 25) -> list[dict[str, object]]:
    return [
        {
            "anchor_id": f"anchor-{index:02d}",
            "target_node": f"Node-{index:02d}",
            "failure_id": f"failure-{index:02d}",
            "failure_cause": f"cause text {index}",
            "failure_symptom": f"symptom text {index}",
            "curriculum_exposure": [f"Node-{index:02d}"],
        }
        for index in range(count)
    ]


def _items_for(anchor: dict[str, object], count: int = 5) -> list[dict[str, object]]:
    node = str(anchor["target_node"])
    return [
        {
            "item_id": f"item-{node}-{index}",
            "family_id": f"family-{node}-{index}",
            "node_ids": [node],
            "public_question": f"Public question {node} {index}",
            "options": {"A": "correct", "B": "wrong b", "C": "wrong c", "D": "wrong d"},
            "answer_values": ["A"],
            "scoring_mode": "mcq",
        }
        for index in range(count)
    ]


class FakeCatalog:
    def __init__(self, anchors: list[dict[str, object]]):
        self.items = {
            item["item_id"]: item
            for anchor in anchors
            for item in _items_for(anchor)
        }

    def for_node(self, node: str, *, deterministic_only: bool = True):
        del deterministic_only
        return [item for item in self.items.values() if node in item["node_ids"]]


def _import(name: str):
    """Turn a missing implementation into a normal RED assertion."""
    try:
        module = __import__(name, fromlist=["*"])
    except ImportError as exc:  # pragma: no cover - expected during RED
        pytest.fail(f"v2 implementation is missing: {name}: {exc}")
    return module


def test_persona_grid_has_fifty_clusters_and_one_hundred_paired_rows():
    grid = _import("experiments.llm_sim_v2.grid")

    rows = grid.build_persona_grid(_anchors(), seed=20260715)

    assert len(rows) == 100
    assert len({row.persona_id for row in rows}) == 50
    assert len({row.pair_id for row in rows}) == 50
    assert len({row.row_id for row in rows}) == 100
    assert {row.deficit_condition for row in rows} == {"deficit", "control"}
    assert {row.modality_condition for row in rows} == {"text_only"}
    for persona_id in {row.persona_id for row in rows}:
        pair = [row for row in rows if row.persona_id == persona_id]
        assert {row.deficit_condition for row in pair} == {"deficit", "control"}
        assert len({row.row_id for row in pair}) == 2
    assert {row.noise_parameters["level"] for row in rows} == {"low", "high"}
    assert sum(row.noise_parameters["level"] == "low" for row in rows) == 50
    assert sum(row.noise_parameters["level"] == "high" for row in rows) == 50


def test_v2_compatibility_modules_expose_the_standalone_builder_and_qa_contracts():
    personas = _import("experiments.llm_sim_v2.personas")
    qa = _import("experiments.llm_sim_v2.qa")
    assert personas.build_personas is not None
    assert qa.scan_blacklist is not None


def test_persona_grid_serialization_is_byte_deterministic_and_outcome_free():
    grid = _import("experiments.llm_sim_v2.grid")

    first = grid.build_persona_grid(_anchors(), seed=9)
    second = grid.build_persona_grid(list(reversed(_anchors())), seed=9)

    assert grid.serialize_grid(first) == grid.serialize_grid(second)
    payload = grid.serialize_grid(first)
    assert b"provider" not in payload
    assert b"outcome" not in payload
    assert hashlib.sha256(payload).hexdigest() == hashlib.sha256(grid.serialize_grid(second)).hexdigest()


def test_ability_band_is_anchor_deterministic_and_not_confounded_with_noise():
    grid = _import("experiments.llm_sim_v2.grid")
    rows = grid.build_persona_grid(_anchors(), seed=3)

    for anchor_id in {row.anchor_id for row in rows}:
        anchor_rows = [row for row in rows if row.anchor_id == anchor_id]
        assert len({row.ability_band for row in anchor_rows}) == 1
        assert {row.noise_parameters["level"] for row in anchor_rows} == {"low", "high"}
    for ability_band in {row.ability_band for row in rows}:
        assert {row.noise_parameters["level"] for row in rows if row.ability_band == ability_band} == {"low", "high"}


def test_persona_grid_rejects_duplicate_anchor_cluster_ids():
    grid = _import("experiments.llm_sim_v2.grid")
    anchors = _anchors()
    anchors[-1] = dict(anchors[0])
    with pytest.raises(ValueError, match="unique|duplicate"):
        grid.build_persona_grid(anchors)


def test_grid_serializer_rejects_rows_with_observation_or_provider_data():
    grid = _import("experiments.llm_sim_v2.grid")
    row = {
        "persona_id": "p",
        "pair_id": "pair",
        "row_id": "p:control",
        "target_node": "Node",
        "curriculum_exposure": ["Node"],
        "deficit_condition": "control",
        "local_skill_vector": {"ability_band": "lower"},
        "observable_error_policy": {},
        "noise_parameters": {"level": "low"},
        "modality_condition": "text_only",
        "seed": 1,
        "provider": "provider-a",
    }
    with pytest.raises(ValueError, match="provider|observed"):
        grid.serialize_grid([row])


def test_persona_schema_rejects_provider_or_observed_outcome_fields():
    models = _import("experiments.llm_sim_v2.models")

    with pytest.raises(ValueError, match="provider|outcome"):
        models.PersonaV2.from_mapping(
            {
                "persona_id": "p",
                "pair_id": "p",
                "row_id": "p:control",
                "target_node": "Node",
                "curriculum_exposure": ["Node"],
                "deficit_condition": "control",
                "local_skill_vector": {"ability_band": "medium"},
                "observable_error_policy": {"guessing": False},
                "noise_parameters": {"level": "low"},
                "modality_condition": "text_only",
                "seed": 1,
                "provider": "provider-a",
            }
        )


def test_panel_selects_exactly_four_valid_family_distinct_items_and_review_payload():
    panel = _import("experiments.llm_sim_v2.panel")
    anchor = _anchors(1)[0]
    catalog = FakeCatalog([anchor])

    selected = panel.select_calibration_items(anchor, catalog)
    review = panel.build_review_payload(anchor, catalog)

    assert len(selected) == 4
    assert len({item["family_id"] for item in selected}) == 4
    assert len({item["item_id"] for item in selected}) == 4
    assert [item["item_id"] for item in selected] == [item["item_id"] for item in panel.select_calibration_items(anchor, catalog)]
    assert review["failure_id"] == anchor["failure_id"]
    assert len(review["items"]) == 4
    assert {"item_id", "family_id", "public_question", "options", "correct_option", "failure_id", "failure_cause", "failure_symptom"} <= set(review["items"][0])


def test_panel_and_mapping_accept_bare_in_memory_item_iterables():
    panel = _import("experiments.llm_sim_v2.panel")
    mapping = _import("experiments.llm_sim_v2.mapping")
    anchor = _anchors(1)[0]
    items = _items_for(anchor)
    selected = panel.select_calibration_items(anchor, items)
    rows = [
        {
            "item_id": item["item_id"],
            "failure_id": anchor["failure_id"],
            "target_option": "B",
            "status": "mapped",
            "reviewer": "claude",
        }
        for item in selected
    ]
    assert len(mapping.normalize_target_option_map(rows, items=items)["rows"]) == 4


def test_panel_rejects_when_four_distinct_families_are_not_available():
    panel = _import("experiments.llm_sim_v2.panel")
    anchor = _anchors(1)[0]
    catalog = FakeCatalog([anchor])
    for item in catalog.items.values():
        item["family_id"] = "same-family"

    with pytest.raises(ValueError, match="four|family"):
        panel.select_calibration_items(anchor, catalog)


def test_panel_uses_the_v1_mechanical_answer_values_predicate():
    panel = _import("experiments.llm_sim_v2.panel")
    anchor = _anchors(1)[0]
    catalog = FakeCatalog([anchor])
    for item in catalog.items.values():
        item.pop("answer_values")
        item["correct_option"] = "A"

    with pytest.raises(ValueError, match="four|family"):
        panel.select_calibration_items(anchor, catalog)

    catalog = FakeCatalog([anchor])
    for item in catalog.items.values():
        item["scoring_mode"] = "MCQ"
    with pytest.raises(ValueError, match="four|family"):
        panel.select_calibration_items(anchor, catalog)


def _valid_mapping(panel_module):
    anchor = _anchors(1)[0]
    catalog = FakeCatalog([anchor])
    selected = panel_module.select_calibration_items(anchor, catalog)
    rows = [
        {
            "item_id": item["item_id"],
            "failure_id": anchor["failure_id"],
            "target_option": "B",
            "status": "mapped",
            "reviewer": "claude",
        }
        for item in selected
    ]
    return anchor, catalog, selected, rows


def test_target_option_map_normalizes_hashes_and_rejects_invalid_targets():
    panel = _import("experiments.llm_sim_v2.panel")
    mapping = _import("experiments.llm_sim_v2.mapping")
    anchor, catalog, selected, rows = _valid_mapping(panel)

    normalized = mapping.normalize_target_option_map(
        rows,
        catalog=catalog,
        expected_rows=[(item["item_id"], anchor["failure_id"]) for item in selected],
    )
    assert normalized["mapping_sha256"]
    assert normalized["target_set_hash"]
    assert normalized["rows"][0]["status"] == "mapped"
    assert normalized == mapping.normalize_target_option_map(
        list(reversed(rows)),
        catalog=catalog,
        expected_rows=[(item["item_id"], anchor["failure_id"]) for item in selected],
    )

    bad_correct = [dict(row, target_option="A") for row in rows]
    with pytest.raises(ValueError, match="correct"):
        mapping.normalize_target_option_map(bad_correct, catalog=catalog)

    bad_absent = [dict(row, target_option="Z") for row in rows]
    with pytest.raises(ValueError, match="option"):
        mapping.normalize_target_option_map(bad_absent, catalog=catalog)


def test_mapping_correct_answer_uses_the_frozen_answer_values_contract():
    panel = _import("experiments.llm_sim_v2.panel")
    mapping = _import("experiments.llm_sim_v2.mapping")
    anchor, catalog, selected, rows = _valid_mapping(panel)
    catalog.items[selected[0]["item_id"]]["correct_option"] = "B"

    normalized = mapping.normalize_target_option_map(rows, catalog=catalog)
    assert normalized["rows"][0]["target_option"] == "B"


def test_target_option_map_rejects_conflicts_missing_rows_and_post_observation_replacement():
    panel = _import("experiments.llm_sim_v2.panel")
    mapping = _import("experiments.llm_sim_v2.mapping")
    anchor, catalog, selected, rows = _valid_mapping(panel)
    expected = [(item["item_id"], anchor["failure_id"]) for item in selected]

    conflict = rows + [dict(rows[0], target_option="C")]
    with pytest.raises(ValueError, match="conflict|duplicate"):
        mapping.normalize_target_option_map(conflict, catalog=catalog)
    with pytest.raises(ValueError, match="missing"):
        mapping.normalize_target_option_map(rows[:-1], catalog=catalog, expected_rows=expected)
    with pytest.raises(ValueError, match="unexpected|extra"):
        mapping.normalize_target_option_map(
            rows + [{**rows[0], "item_id": "not-in-panel"}],
            catalog=catalog,
            expected_rows=expected,
        )

    frozen = mapping.normalize_target_option_map(rows, catalog=catalog, expected_rows=expected)
    with pytest.raises(ValueError, match="observ|replace|frozen"):
        mapping.normalize_target_option_map(
            [dict(rows[0], target_option="C")] + rows[1:],
            catalog=catalog,
            existing=frozen,
            observation_started=True,
        )

    frozen_started = dict(frozen, observation_started=True)
    with pytest.raises(ValueError, match="observ|replace|frozen"):
        mapping.normalize_target_option_map(
            [dict(rows[0], target_option="C")] + rows[1:],
            catalog=catalog,
            existing=frozen_started,
        )


def test_target_option_map_rejects_codex_manual_reviewer_and_accepts_explicit_ambiguity():
    panel = _import("experiments.llm_sim_v2.panel")
    mapping = _import("experiments.llm_sim_v2.mapping")
    anchor, catalog, selected, rows = _valid_mapping(panel)
    with pytest.raises(ValueError, match="reviewer|codex"):
        mapping.normalize_target_option_map(
            [dict(rows[0], reviewer="codex_manual")] + rows[1:],
            catalog=catalog,
        )

    ambiguous = [
        {
            "item_id": selected[0]["item_id"],
            "failure_id": anchor["failure_id"],
            "target_option": None,
            "status": "excluded_ambiguous",
            "reviewer_provenance": {"reviewer": "claude", "method": "blind_local_review"},
            "ambiguity_reason": "two distractors encode the same error",
        }
    ]
    normalized = mapping.normalize_target_option_map(ambiguous, catalog=catalog)
    assert normalized["rows"][0]["status"] == "excluded_ambiguous"


def test_v2_store_enforces_exact_run_id_phase_isolation_and_path_guards(tmp_path: Path):
    store_module = _import("experiments.llm_sim_v2.store")

    pilot = store_module.V2Store(tmp_path, phase="pilot")
    main = store_module.V2Store(tmp_path, phase="main")
    assert pilot.run_id == "llm-personas-v2-dual"
    assert pilot.root != main.root
    assert pilot.root.name == "pilot"
    assert main.root.name == "main"

    for bad in (
        tmp_path / "llm-personas-v1" / "pilot",
        tmp_path / "archive-v1-old" / "pilot",
        tmp_path / "official" / "llm-personas-v2-dual" / "pilot",
        tmp_path / "official-data" / "llm-personas-v2-dual" / "pilot",
        tmp_path / "local_store" / "llm-personas-v2-dual" / "pilot",
        tmp_path / "local-store" / "llm-personas-v2-dual" / "pilot",
        tmp_path / "study-log" / "llm-personas-v2-dual" / "pilot",
        tmp_path / "llm-personas-v2-other" / "pilot",
    ):
        with pytest.raises(ValueError):
            store_module.V2Store(bad)
    with pytest.raises(ValueError, match="run"):
        store_module.V2Store(tmp_path, run_id="llm-personas-v1", phase="pilot")


def test_provenance_hashes_declared_files_without_serializing_content_or_environment(tmp_path: Path):
    provenance = _import("experiments.llm_sim_v2.provenance")
    code = tmp_path / "code.py"
    config = tmp_path / "config.json"
    code.write_text("print('ok')\n", encoding="utf-8")
    config.write_text('{"seed": 1}\n', encoding="utf-8")
    os.environ["YHER_TEST_SECRET"] = "must-not-appear"

    manifest = provenance.hash_declared_files(tmp_path, ["code.py", "config.json"])

    assert manifest["files"][0]["path"] == "code.py"
    encoded = json.dumps(manifest, sort_keys=True)
    assert "must-not-appear" not in encoded
    assert "print('ok')" not in encoded


def test_frozen_git_proof_requires_byte_identity_and_pre_observation_timestamp(tmp_path: Path):
    provenance = _import("experiments.llm_sim_v2.provenance")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    plan = tmp_path / "plan.md"
    plan.write_text("frozen plan\n", encoding="utf-8")
    subprocess.run(["git", "add", "plan.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "freeze"], cwd=tmp_path, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()

    proof = provenance.verify_frozen_git_commit(
        tmp_path,
        commit=commit,
        declared_files={"plan.md": digest},
        observation_timestamp="2999-01-01T00:00:00Z",
    )
    assert proof["ok"] is True
    assert proof["byte_identical"] is True
    assert proof["precedes_observation"] is True

    plan.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte|identical"):
        provenance.verify_frozen_git_commit(
            tmp_path,
            commit=commit,
            declared_files={"plan.md": hashlib.sha256(plan.read_bytes()).hexdigest()},
            observation_timestamp="2999-01-01T00:00:00Z",
        )


def test_provenance_accepts_categorized_code_config_prompt_mapping_plan_sets(tmp_path: Path):
    provenance = _import("experiments.llm_sim_v2.provenance")
    for name in ("code.py", "config.json", "prompt.txt", "mapping.json", "plan.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    manifest = provenance.hash_declared_files(
        tmp_path,
        {
            "code": ["code.py"],
            "config": ["config.json"],
            "prompt": ["prompt.txt"],
            "mapping": ["mapping.json"],
            "plan": ["plan.md"],
        },
    )
    assert {row["path"] for row in manifest["files"]} == {
        "code.py",
        "config.json",
        "prompt.txt",
        "mapping.json",
        "plan.md",
    }
