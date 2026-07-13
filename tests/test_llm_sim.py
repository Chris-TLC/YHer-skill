"""Contract tests for the S2 simulated-persona facade.

These tests deliberately use an in-memory catalog and a fake transport.  A live
provider is never contacted from the offline suite.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest


def _item(
    item_id: str,
    *,
    node: str = "Target",
    difficulty: float = 0.5,
    answer: str = "A",
    options: dict[str, str] | None = None,
    distractor_map: dict[str, str] | None = None,
):
    class Item:
        pass

    item = Item()
    item.item_id = item_id
    item.family_id = f"family-{item_id}"
    item.node_ids = (node,)
    item.stem_text = f"Question {item_id}"
    item.stem_blocks = ()
    item.options = options or {"A": "correct", "B": "wrong-b", "C": "wrong-c", "D": "wrong-d"}
    item.answer_values = (answer,)
    item.answer_verification_status = "passed"
    item.scoring_mode = "mcq"
    item.item_type = "mcq"
    item.difficulty = difficulty
    item.has_media = False
    item.source_label = "synthetic-test"
    if distractor_map is not None:
        item.distractor_map = distractor_map
    return item


class FakeCatalog:
    def __init__(self):
        self.items = {
            item.item_id: item
            for item in (
                _item("i1", distractor_map={"failure-1": "B"}),
                _item("i2", difficulty=0.25, distractor_map={"failure-1": "C"}),
                _item("i3", difficulty=0.75, distractor_map={"failure-1": "B"}),
                _item("i4", difficulty=1.0, distractor_map={"failure-1": "C"}),
            )
        }
        self._prerequisites = {"Target": ()}

    def open_nodes(self, minimum_families: int = 5):
        return {"Target": 4}

    def for_node(self, node: str, *, deterministic_only: bool = False):
        return tuple(item for item in self.items.values() if node in item.node_ids)

    def prerequisites_for(self, node: str):
        return ()


class FakeKG:
    def all_nodes(self):
        class Failure:
            cause = "failure cause"
            symptom = "failure symptom"
            diagnostic_question = "Which option reflects the failure?"

        class Node:
            common_failures = [Failure()]

        nodes = []
        for index in range(25):
            node = Node()
            node.node_id = "Target" if index == 0 else f"Target-{index:02d}"
            nodes.append(node)
        return nodes


class FakeTransport:
    def __init__(self, responses=None):
        self.responses = None if responses is None else list(responses)
        self.calls = []

    def complete(self, *, provider, model, messages, timeout_seconds):
        self.calls.append({"provider": provider, "model": model, "messages": messages})
        if self.responses is None:
            weak = '"strength": "weak"' in messages[-1]["content"]
            content = (
                '{"answer":"B","rationale":"simulated misconception"}'
                if weak
                else '{"answer":"A","rationale":"ok"}'
            )
        else:
            content = self.responses.pop(0)
        return {
            "content": content,
            "model_returned": model,
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "cost_yuan": 0.01,
        }


def _paired_personas(pair_count: int = 25):
    rows = []
    for index in range(pair_count):
        pair_id = f"pair-{index:02d}"
        for strength in ("weak", "strong"):
            rows.append(
                {
                    "persona_id": f"{pair_id}:{strength}",
                    "pair_id": pair_id,
                    "strength": strength,
                    "target_node": "Target",
                    "failure_id": "failure-1",
                    "failure_cause": f"cause-{index}",
                    "failure_symptom": f"symptom-{index}",
                    "diagnostic_question": f"question-{index}",
                }
            )
    return rows


def test_persona_factory_builds_25_weak_strong_pairs_without_response_dependency():
    from experiments.llm_sim.personas import build_personas

    personas = build_personas(FakeKG(), pair_count=25, seed=7)
    assert len(personas) == 50
    assert {p.strength for p in personas} == {"weak", "strong"}
    assert len({p.pair_id for p in personas}) == 25
    assert len({p.target_node for p in personas}) == 25
    for pair_id in {p.pair_id for p in personas}:
        pair = [p for p in personas if p.pair_id == pair_id]
        assert {p.strength for p in pair} == {"weak", "strong"}
        assert all(p.annotation_source == "kg.common_failures" for p in pair)


def test_runner_consumes_frozen_persona_count_seed_version_and_prompt_version(
    monkeypatch, tmp_path: Path
):
    from experiments.llm_sim import runner as runner_module
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import LLMSimulationRunner, _messages
    from experiments.llm_sim.store import SimulationStore

    captured = {}

    def fake_build(_kg, **kwargs):
        captured.update(kwargs)
        return [
            Persona.from_mapping(
                {
                    "persona_id": "p:weak",
                    "pair_id": "p",
                    "strength": "weak",
                    "target_node": "Target",
                    "failure_id": "failure-1",
                }
            )
        ]

    monkeypatch.setattr(runner_module, "build_personas", fake_build)
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=FakeTransport(),
        repo_root=tmp_path,
    )
    assert captured["pair_count"] == runner.config.pair_count
    assert captured["seed"] == runner.config.study_seed
    assert (
        captured["seed_derivation_version"]
        == runner.config.persona_seed_derivation_version
    )
    prompt = _messages(
        runner.personas[0],
        FakeCatalog().items["i1"],
        prompt_version=runner.config.prompt_version,
    )
    assert runner.config.prompt_version in prompt[0]["content"]


def test_persona_factory_can_filter_to_catalog_eligible_nodes():
    from experiments.llm_sim.personas import build_personas

    personas = build_personas(
        FakeKG(),
        pair_count=1,
        seed=7,
        eligible_nodes={"Target"},
    )
    assert len(personas) == 2
    assert {persona.target_node for persona in personas} == {"Target"}


def test_persona_pairs_cover_distinct_nodes_before_reusing_second_failures():
    from experiments.llm_sim.personas import build_personas

    class MultiFailureKG:
        def all_nodes(self):
            nodes = []
            for index in range(25):
                failure_a = type(
                    "Failure",
                    (),
                    {"cause": f"cause-{index}-a", "symptom": "s", "diagnostic_question": "q"},
                )()
                failure_b = type(
                    "Failure",
                    (),
                    {"cause": f"cause-{index}-b", "symptom": "s", "diagnostic_question": "q"},
                )()
                node = type(
                    "Node",
                    (),
                    {"node_id": f"Node-{index:02d}", "common_failures": [failure_a, failure_b]},
                )()
                nodes.append(node)
            return nodes

    personas = build_personas(MultiFailureKG(), pair_count=25)
    assert len({persona.target_node for persona in personas}) == 25


def test_child_common_failure_can_anchor_an_open_catalog_parent():
    from experiments.llm_sim.personas import build_personas

    failure = {
        "cause": "child cause",
        "symptom": "child symptom",
        "diagnostic_question": "child question",
    }
    kg_rows = [
        {
            "node_id": "Target-child",
            "parent_node": "Target",
            "common_failures": [failure],
        }
    ]
    personas = build_personas(
        kg_rows,
        pair_count=1,
        eligible_nodes={"Target"},
    )
    assert {persona.target_node for persona in personas} == {"Target"}
    assert all(persona.failure_id.startswith("Target-child#") for persona in personas)


def test_parent_targets_are_interleaved_before_second_child_failure():
    from experiments.llm_sim.personas import build_personas

    rows = []
    eligible = set()
    for index in range(25):
        parent = f"Parent-{index:02d}"
        eligible.add(parent)
        for child in ("a", "b"):
            rows.append(
                {
                    "node_id": f"{parent}-{child}",
                    "parent_node": parent,
                    "common_failures": [
                        {
                            "cause": f"cause-{index}-{child}",
                            "symptom": "s",
                            "diagnostic_question": "q",
                        }
                    ],
                }
            )
    personas = build_personas(rows, pair_count=25, eligible_nodes=eligible)
    assert len({persona.target_node for persona in personas}) == 25


def test_catalog_target_aliases_use_only_the_frozen_mechanical_normalization():
    from experiments.llm_sim.personas import build_personas

    rows = [
        {
            "node_id": "Target_alias-child",
            "parent_node": "Target_alias",
            "common_failures": [
                {"cause": "cause", "symptom": "symptom", "diagnostic_question": "question"}
            ],
        }
    ]
    personas = build_personas(
        rows,
        pair_count=1,
        eligible_nodes={"Target/alias"},
    )
    assert {persona.target_node for persona in personas} == {"Target/alias"}


def test_calibration_ready_targets_use_only_the_panel_candidate_contract():
    from experiments.llm_sim.panel import (
        calibration_ready_target_nodes,
        is_calibration_candidate,
    )
    from experiments.llm_sim.personas import build_personas

    valid = [_item(f"good-{index}", node="Good") for index in range(4)]
    duplicate_family = _item("good-duplicate", node="Good")
    duplicate_family.family_id = valid[0].family_id
    invalid_item_id = _item("placeholder", node="Bad")
    invalid_item_id.item_id = ""
    invalid_family = _item("bad-family", node="Bad")
    invalid_family.family_id = ""
    invalid_scoring = _item("bad-numeric", node="Bad")
    invalid_scoring.scoring_mode = "numeric"
    invalid_options = _item("bad-options", node="Bad")
    invalid_options.options = {}
    invalid_answer = _item("bad-answer", node="Bad")
    invalid_answer.answer_values = ("Z",)
    bad_valid = [_item(f"bad-{index}", node="Bad") for index in range(3)]

    class Catalog:
        items = {
            item.item_id or "missing-id": item
            for item in (
                *valid,
                duplicate_family,
                invalid_item_id,
                invalid_family,
                invalid_scoring,
                invalid_options,
                invalid_answer,
                *bad_valid,
            )
        }

        def open_nodes(self):
            return {"Good": 5, "Bad": 9}

        def for_node(self, node, *, deterministic_only=False):
            return tuple(
                item for item in self.items.values() if node in item.node_ids
            )

    catalog = Catalog()
    assert all(is_calibration_candidate(item) for item in valid)
    assert not any(
        is_calibration_candidate(item)
        for item in (
            invalid_item_id,
            invalid_family,
            invalid_scoring,
            invalid_options,
            invalid_answer,
        )
    )
    assert calibration_ready_target_nodes(
        catalog, catalog.open_nodes()
    ) == frozenset({"Good"})
    with pytest.raises(ValueError, match="25 persona pairs are required"):
        build_personas(
            [
                {
                    "node_id": "Good",
                    "common_failures": [
                        {
                            "cause": f"cause-{index}",
                            "symptom": "symptom",
                            "diagnostic_question": "question",
                        }
                        for index in range(24)
                    ],
                }
            ],
            pair_count=25,
            eligible_nodes={"Good"},
            seed_derivation_version="yher-llm-persona-v2",
        )


def test_panel_freezes_before_first_provider_response_and_has_stable_hash(tmp_path: Path):
    from experiments.llm_sim.panel import freeze_manipulation_panel, load_frozen_panel

    panel = freeze_manipulation_panel(
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
            }
        ],
        catalog=FakeCatalog(),
        failure_id="failure-1",
        output_path=tmp_path / "manipulation_panel.json",
        study_seed=123,
    )
    assert panel["frozen"] is True
    assert panel["observation_started"] is False
    assert panel["simulated"] is True
    assert panel["persona_id"] == "llm-sim-study:manipulation-panel"
    assert panel["provider"] == "study_design"
    assert panel["model_id"] == "no-provider-observation"
    assert panel["panel_sha256"] == load_frozen_panel(tmp_path / "manipulation_panel.json")["panel_sha256"]
    assert panel["annotations"][0]["target_option"] == "B"
    assert panel["annotations"][0]["mapping_status"] == "mapped"
    calibration = panel["annotations"][0]["calibration_items"]
    assert len(calibration) == 4
    assert len({row["family_id"] for row in calibration}) == 4


def test_panel_accepts_catalog_level_machine_mapping_but_never_text_inference(tmp_path: Path):
    from experiments.llm_sim.panel import freeze_manipulation_panel

    catalog = FakeCatalog()
    for item in catalog.items.values():
        if hasattr(item, "distractor_map"):
            delattr(item, "distractor_map")
    catalog.distractor_map = {
        "i1": {"failure-1": "B"},
        "i2": {"failure-1": "C"},
        "i3": {"failure-1": "B"},
        "i4": {"failure-1": "C"},
    }
    panel = freeze_manipulation_panel(
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
            }
        ],
        catalog=catalog,
        failure_id="failure-1",
        output_path=tmp_path / "panel.json",
        study_seed=123,
    )
    assert panel["annotations"][0]["mapping_status"] == "mapped"


def test_independent_annotation_map_is_hashed_copied_and_used(tmp_path: Path):
    from experiments.llm_sim.panel import annotation_map_hash
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    catalog = FakeCatalog()
    for item in catalog.items.values():
        delattr(item, "distractor_map")
    annotation_path = tmp_path / "explicit_annotations.json"
    annotation_map = {
        "items": {
            "i1": {"failure-1": "B"},
            "i2": {"failure-1": "C"},
            "i3": {"failure-1": "B"},
            "i4": {"failure-1": "C"},
        }
    }
    annotation_path.write_text(json.dumps(annotation_map), encoding="utf-8")
    store = SimulationStore(tmp_path / "sim_store")
    runner = LLMSimulationRunner(
        catalog=catalog,
        kg=FakeKG(),
        store=store,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        annotation_map=annotation_map,
        annotation_map_source=annotation_path,
        repo_root=tmp_path,
    )
    preparation = runner.prepare()
    snapshot = store.read_json("annotation_map_snapshot.json")
    panel = store.read_json("manipulation_panel.json")
    assert preparation["mapped_count"] == 1
    assert preparation["annotation_map_sha256"] == annotation_map_hash(annotation_map)
    assert preparation["annotation_map_source"] == str(annotation_path.resolve())
    assert snapshot["annotation_map_sha256"] == preparation["annotation_map_sha256"]
    assert snapshot["source_path"] == str(annotation_path.resolve())
    assert snapshot["annotation_map"]["items"]["i1"]["failure-1"] == "B"
    assert panel["annotations"][0]["mapping_status"] == "mapped"


def test_frozen_panel_rejects_changed_independent_annotation_map(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    catalog = FakeCatalog()
    for item in catalog.items.values():
        delattr(item, "distractor_map")
    base = {
        "i1": {"failure-1": "B"},
        "i2": {"failure-1": "C"},
        "i3": {"failure-1": "B"},
        "i4": {"failure-1": "C"},
    }
    kwargs = {
        "catalog": catalog,
        "kg": FakeKG(),
        "store": SimulationStore(tmp_path / "sim_store"),
        "personas": [
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        "repo_root": tmp_path,
    }
    LLMSimulationRunner(**kwargs, annotation_map={"items": base}).prepare()
    changed = {**base, "i1": {"failure-1": "C"}}
    with pytest.raises(ValueError, match="does not match"):
        LLMSimulationRunner(
            **kwargs,
            annotation_map={"items": changed},
        ).prepare()


def test_calibration_is_separate_and_gates_journeys_by_strength(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=FakeTransport(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    result = runner.run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    calibration = result["calibration_attempts"]
    assert len(calibration) == 4
    assert all(row["record_type"] == "llm_sim_calibration_attempt" for row in calibration)
    eligibility = result["provider_eligibility"]["weak"]
    assert eligibility["n"] == 4
    assert eligibility["accuracy_denominator"] == 4
    assert eligibility["accuracy"] == 0.0
    assert eligibility["accuracy_band_pass"] is True
    assert result["journeys"][0]["events"][0]["record_type"] == "llm_sim_event"


def test_weak_manipulation_requires_positive_cluster_bootstrap_contrast():
    from experiments.llm_sim.config import load_frozen_config
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import _calibration_eligibility

    personas = [
        Persona.from_mapping(
            {
                "persona_id": f"p-{index}:weak",
                "pair_id": f"p-{index}",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        )
        for index in range(4)
    ]
    panel_rows = [{"mapping_status": "mapped"} for _ in personas]

    def attempts(hit: bool):
        return [
            {
                "persona_id": persona.persona_id,
                "correct": False,
                "target_misconception_hit": hit,
                "random_wrong_option_baseline": 1 / 3,
            }
            for persona in personas
            for _ in range(4)
        ]

    passed = _calibration_eligibility(
        strength="weak",
        personas=personas,
        attempts=attempts(True),
        panel_rows=panel_rows,
        calibration_ready=True,
        config=load_frozen_config(),
        prompt_revision=0,
    )
    assert passed["accuracy_gate"]["pass"] is True
    assert passed["target_gate"]["applicable"] is True
    assert passed["target_gate"]["bootstrap_resamples"] == 10_000
    assert passed["target_gate"]["contrast_ci95_lower"] > 0
    assert passed["target_gate"]["pass"] is True
    assert passed["eligible"] is True

    failed_v0 = _calibration_eligibility(
        strength="weak",
        personas=personas,
        attempts=attempts(False),
        panel_rows=panel_rows,
        calibration_ready=True,
        config=load_frozen_config(),
        prompt_revision=0,
    )
    assert failed_v0["accuracy_gate"]["pass"] is True
    assert failed_v0["target_gate"]["contrast_ci95_lower"] < 0
    assert failed_v0["target_gate"]["pass"] is False
    assert failed_v0["status"] == "prompt_rewrite_available"
    assert failed_v0["excluded_reason"] == "target_contrast_failed"

    failed_v1 = _calibration_eligibility(
        strength="weak",
        personas=personas,
        attempts=attempts(False),
        panel_rows=panel_rows,
        calibration_ready=True,
        config=load_frozen_config(),
        prompt_revision=1,
    )
    assert failed_v1["status"] == "excluded_post_calibration"


def test_strong_target_gate_is_not_applicable_and_mapping_absence_only_excludes_weak():
    from experiments.llm_sim.config import load_frozen_config
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import _calibration_eligibility

    config = load_frozen_config()
    strong = Persona.from_mapping(
        {
            "persona_id": "p:strong",
            "pair_id": "p",
            "strength": "strong",
            "target_node": "Target",
            "failure_id": "failure-1",
        }
    )
    strong_result = _calibration_eligibility(
        strength="strong",
        personas=[strong],
        attempts=[{"persona_id": strong.persona_id, "correct": True}] * 4,
        panel_rows=[{"mapping_status": "excluded_pre_outcome"}],
        calibration_ready=True,
        config=config,
        prompt_revision=0,
    )
    assert strong_result["accuracy_gate"]["pass"] is True
    assert strong_result["target_gate"]["applicable"] is False
    assert strong_result["target_gate"]["status"] == "not_applicable"
    assert strong_result["eligible"] is True

    weak = Persona.from_mapping(
        {
            "persona_id": "p:weak",
            "pair_id": "p",
            "strength": "weak",
            "target_node": "Target",
            "failure_id": "failure-1",
        }
    )
    weak_result = _calibration_eligibility(
        strength="weak",
        personas=[weak],
        attempts=[{"persona_id": weak.persona_id, "correct": False}] * 4,
        panel_rows=[{"mapping_status": "excluded_pre_outcome"}],
        calibration_ready=True,
        config=config,
        prompt_revision=0,
    )
    assert weak_result["accuracy_gate"]["pass"] is True
    assert weak_result["target_gate"]["status"] == "excluded_pre_outcome"
    assert weak_result["status"] == "excluded_pre_outcome"


def test_insufficient_calibration_mapping_excludes_strength_before_transport(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    catalog = FakeCatalog()
    delattr(catalog.items["i4"], "distractor_map")
    transport = FakeTransport()
    runner = LLMSimulationRunner(
        catalog=catalog,
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    result = runner.run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    assert result["provider_eligibility"]["weak"]["status"] == "excluded_pre_outcome"
    assert len(result["journeys"]) == 1
    assert len(transport.calls) == 5  # four calibration responses, then one journey


def test_unmapped_target_is_pre_outcome_excluded_not_semantic_fallback(tmp_path: Path):
    from experiments.llm_sim.panel import freeze_manipulation_panel

    panel = freeze_manipulation_panel(
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
            }
        ],
        catalog=FakeCatalog(),
        failure_id="missing-failure",
        output_path=tmp_path / "panel.json",
        study_seed=123,
    )
    annotation = panel["annotations"][0]
    assert annotation["mapping_status"] == "excluded_pre_outcome"
    assert annotation["exclusion_reason"] == "no_mechanical_target_option_mapping"
    assert annotation["target_option"] is None


def test_frozen_panel_refuses_a_different_pre_outcome_definition(tmp_path: Path):
    from experiments.llm_sim.panel import freeze_manipulation_panel

    path = tmp_path / "panel.json"
    common = {
        "personas": [
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
            }
        ],
        "catalog": FakeCatalog(),
        "failure_id": "failure-1",
        "output_path": path,
        "study_seed": 123,
    }
    first = freeze_manipulation_panel(**common)
    assert freeze_manipulation_panel(**common)["panel_sha256"] == first["panel_sha256"]
    with pytest.raises(FileExistsError, match="frozen"):
        freeze_manipulation_panel(**{**common, "study_seed": 124})


def test_loaded_panel_cannot_be_marked_as_post_observation(tmp_path: Path):
    from experiments.llm_sim.panel import freeze_manipulation_panel, load_frozen_panel

    path = tmp_path / "panel.json"
    freeze_manipulation_panel(
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
            }
        ],
        catalog=FakeCatalog(),
        failure_id="failure-1",
        output_path=path,
        study_seed=123,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["observation_started"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="observation"):
        load_frozen_panel(path)


def test_store_rejects_local_store_and_requires_simulated_envelope(tmp_path: Path):
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    record = {
        "simulated": True,
        "persona_id": "persona:0:weak",
        "provider": "deepseek",
        "model_id": "model-a",
        "record_type": "journey",
    }
    path = store.write_json("journeys/p0.json", record)
    assert json.loads(path.read_text(encoding="utf-8"))["simulated"] is True
    with pytest.raises(ValueError, match="local_store"):
        SimulationStore(tmp_path / "local_store")
    with pytest.raises(ValueError, match="simulated:true"):
        store.write_json("journeys/bad.json", {"record_type": "journey"})


def test_store_path_encoding_is_deterministic_collision_safe_for_slash_and_unicode(
    tmp_path: Path,
):
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    persona_ids = (
        "llm-pair:00:化学计量（摩尔/阿伏伽德罗）:weak",
        "llm-pair:00:化学计量_摩尔_阿伏伽德罗:weak",
        "literal/u-5YyW5a2m:weak",
        "literal%2Fu-5YyW5a2m:weak",
    )
    paths = [store.journey_relative_path("deepseek", value, "A") for value in persona_ids]
    assert len(set(paths)) == len(persona_ids)
    assert paths == [store.journey_relative_path("deepseek", value, "A") for value in persona_ids]
    assert all(len(path.parts) == 3 for path in paths)
    assert all("/" not in path.name and "\\" not in path.name for path in paths)


def test_slash_persona_id_is_preserved_in_records_while_path_is_encoded(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    persona_id = "pair/含斜杠:weak"
    store = SimulationStore(tmp_path / "sim_store")
    result = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=store,
        transport=FakeTransport(),
        personas=[
            {
                "persona_id": persona_id,
                "pair_id": "pair/含斜杠",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert result["journeys"][0]["persona_id"] == persona_id
    relative = store.journey_relative_path("deepseek", persona_id, "A")
    assert store.read_json(relative)["persona_id"] == persona_id


def test_production_50_persona_snapshot_and_paths_are_frozen_pre_observation(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    runner = LLMSimulationRunner(store=store, repo_root=tmp_path)
    preparation = runner.prepare()
    snapshot = store.read_json("persona_panel.json")
    assert len(runner.personas) == 50
    assert len(snapshot["personas"]) == 50
    assert snapshot["simulated"] is True
    assert snapshot["persona_id"] == "llm-sim-study:persona-panel"
    assert snapshot["provider"] == "study_design"
    assert snapshot["model_id"] == "no-provider-observation"
    assert snapshot["frozen"] is True
    assert snapshot["observation_started"] is False
    assert snapshot["frozen_pre_observation_utc"] == runner.config.frozen_pre_observation_utc
    assert preparation["persona_panel_path"] == "persona_panel.json"
    assert preparation["persona_panel_sha256"] == snapshot["persona_panel_sha256"]
    expected = [persona.to_dict() for persona in sorted(runner.personas, key=lambda row: row.persona_id)]
    assert snapshot["personas"] == expected
    assert snapshot["canonical_match"] is True
    assert snapshot["canonical_personas_sha256"] == snapshot["personas_sha256"]
    assert all(
        set(("failure_cause", "failure_symptom", "diagnostic_question")) <= set(row)
        for row in snapshot["personas"]
    )
    journey_paths = {
        store.journey_relative_path("deepseek", persona.persona_id, arm)
        for persona in runner.personas
        for arm in ("A", "B")
    }
    assert len(journey_paths) == 100


def test_production_persona_calibration_preflight_is_ready_and_byte_stable(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    stores = [
        SimulationStore(tmp_path / name / "sim_store")
        for name in ("first", "second")
    ]
    runners = [
        LLMSimulationRunner(
            store=store,
            repo_root=tmp_path / "protected_repo",
        )
        for store in stores
    ]
    preparations = [runner.prepare() for runner in runners]
    panels = [store.read_json("manipulation_panel.json") for store in stores]

    assert all(len(runner.personas) == 50 for runner in runners)
    assert all(
        len({persona.pair_id for persona in runner.personas}) == 25
        for runner in runners
    )
    assert all(
        persona.target_node != "同分异构体"
        for runner in runners
        for persona in runner.personas
    )
    for preparation, panel in zip(preparations, panels):
        assert preparation["provider_observations"] == 0
        assert preparation["mapped_count"] == 0
        assert preparation["excluded_pre_outcome_count"] == 50
        assert len(panel["annotations"]) == 50
        assert all(
            row["calibration_status"] == "ready"
            and len(row["calibration_items"]) == 4
            and len(
                {item["family_id"] for item in row["calibration_items"]}
            )
            == 4
            for row in panel["annotations"]
        )
        assert all(
            row["mapping_status"] == "excluded_pre_outcome"
            for row in panel["annotations"]
        )
    for relative in (
        "manipulation_panel.json",
        "persona_panel.json",
        "preparation_manifest.json",
    ):
        assert (stores[0].root / relative).read_bytes() == (
            stores[1].root / relative
        ).read_bytes()


def test_tampered_persona_snapshot_fails_before_transport(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    transport = FakeTransport()
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=store,
        transport=transport,
        personas=_paired_personas(1),
        repo_root=tmp_path,
    )
    runner.prepare()
    path = store.root / "persona_panel.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["personas"][0]["failure_cause"] = "tampered"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="persona panel"):
        runner.run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert transport.calls == []


def test_provider_runner_freezes_panel_before_transport_and_records_envelope(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    transport = FakeTransport()
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "manipulation_panel.json",
        repo_root=tmp_path,
    )
    result = runner.run_provider("deepseek", model="model-a", max_items=1)
    assert len(transport.calls) == 6  # 4 calibration + A and B
    assert result["panel_sha256"]
    assert result["config_sha256"]
    assert result["observation_started"] is True
    assert all("answer_values" not in json.dumps(call["messages"]) for call in transport.calls)
    for journey in result["journeys"]:
        assert journey["simulated"] is True
        assert journey["provider"] == "deepseek"
        assert journey["model_id"] == "model-a"
        assert journey["arm"] in {"A", "B"}
        assert journey["events"][0]["simulated"] is True
    assert result["manifest"]["reportability"]["reportable"] is False
    assert result["manifest"]["persona_panel_path"] == "persona_panel.json"
    assert result["manifest"]["persona_panel_sha256"] == result["manifest"][
        "preparation_persona_panel_sha256"
    ]
    assert result["manifest"]["reportability"]["formal_design_match"] is False
    assert "persona_composition" in result["manifest"]["reportability"]["formal_design_failures"]
    assert "max_items" in result["manifest"]["reportability"]["formal_design_failures"]


def test_runner_resume_skips_completed_persona_arm_and_accounts_tokens(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    transport = FakeTransport()
    kwargs = dict(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=store,
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    first = LLMSimulationRunner(**kwargs).run_provider("deepseek", model="model-a", max_items=1)
    assert first["accounting"]["requests"] == 6
    artifacts = first["manifest"]["artifacts"]
    assert {row["record_type"] for row in artifacts} == {
        "llm_sim_calibration",
        "llm_sim_journey",
        "llm_sim_provider_attempt",
    }
    assert len(artifacts) == 9  # six attempts, one calibration, and two arm finals
    for row in artifacts:
        artifact_path = store.root / row["path"]
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == row["sha256"]
    expected_aggregate = hashlib.sha256(
        json.dumps(
            artifacts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert first["manifest"]["artifact_aggregate_sha256"] == expected_aggregate
    created_at = first["manifest"]["created_at_utc"]
    transport.calls.clear()
    second = LLMSimulationRunner(**kwargs).run_provider("deepseek", model="model-a", max_items=1, resume=True)
    assert second["accounting"]["requests"] == 6
    assert second["accounting"]["retries"] == 0
    assert second["accounting"]["skipped"] == 2
    assert second["accounting"]["input_tokens"] == 60
    assert second["accounting"]["output_tokens"] == 24
    assert second["accounting"]["cost_yuan"] == 0.06
    assert transport.calls == []
    assert second["manifest"]["created_at_utc"] == created_at
    assert second["manifest"]["artifact_aggregate_sha256"] == expected_aggregate


def test_no_resume_refuses_existing_completed_raw_before_transport(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    base = dict(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=store,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    LLMSimulationRunner(**base, transport=FakeTransport()).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    journey_path = store.root / store.journey_relative_path("deepseek", "p:weak", "A")
    calibration_path = store.root / store.calibration_relative_path("deepseek", "p:weak")
    before = (journey_path.read_bytes(), calibration_path.read_bytes())
    transport = FakeTransport()
    with pytest.raises(FileExistsError, match="--no-resume"):
        LLMSimulationRunner(**base, transport=transport).run_provider(
            "deepseek",
            model="model-a",
            max_items=1,
            arms=("A",),
            resume=False,
        )
    assert transport.calls == []
    assert before == (journey_path.read_bytes(), calibration_path.read_bytes())


@pytest.mark.parametrize("checkpoint_kind", ("calibration", "journey"))
def test_no_resume_refuses_existing_partial_checkpoint_before_transport(
    tmp_path: Path, checkpoint_kind: str
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    transport = FakeTransport()
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=store,
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    runner.prepare()
    if checkpoint_kind == "calibration":
        relative = store.calibration_checkpoint_relative_path("deepseek", "p:weak")
        record_type = "llm_sim_calibration"
    else:
        relative = store.checkpoint_relative_path("deepseek", "p:weak", "A")
        record_type = "llm_sim_journey"
    partial = {
        "simulated": True,
        "run_id": runner.config.run_id,
        "persona_id": "p:weak",
        "provider": "deepseek",
        "model_id": "model-a",
        "record_type": record_type,
        "status": "in_progress",
        "events": [],
    }
    path = store.write_json(relative, partial)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="--no-resume"):
        runner.run_provider(
            "deepseek",
            model="model-a",
            max_items=1,
            arms=("A",),
            resume=False,
        )
    assert transport.calls == []
    assert path.read_bytes() == before


def test_store_immutable_write_never_replaces_completed_raw(tmp_path: Path):
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    original = {
        "simulated": True,
        "persona_id": "p",
        "provider": "deepseek",
        "model_id": "model-a",
        "record_type": "llm_sim_journey",
        "status": "complete",
    }
    path = store.write_json("journey.json", original, immutable=True)
    assert store.write_json("journey.json", original, immutable=True) == path
    with pytest.raises(FileExistsError, match="immutable"):
        store.write_json(
            "journey.json",
            {**original, "status": "incomplete"},
            immutable=True,
        )
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "complete"


def test_backoff_and_circuit_breaker_are_provider_local(tmp_path: Path):
    from experiments.llm_sim.runner import CircuitOpenError, ProviderCallPolicy

    policy = ProviderCallPolicy(max_attempts=2, failure_threshold=2, base_backoff_seconds=0)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise RuntimeError("429 rate limited")

    with pytest.raises(RuntimeError):
        policy.call(flaky)
    assert calls["n"] == 2
    with pytest.raises(CircuitOpenError):
        policy.call(lambda: "must-not-call")


def test_circuit_opens_at_failure_threshold_without_extra_attempt():
    from experiments.llm_sim.runner import ProviderCallPolicy

    policy = ProviderCallPolicy(
        max_attempts=3,
        failure_threshold=2,
        base_backoff_seconds=0,
    )
    calls = {"n": 0}

    def fail():
        calls["n"] += 1
        raise RuntimeError("503 unavailable")

    with pytest.raises(RuntimeError):
        policy.call(fail)
    assert calls["n"] == 2


def test_runner_policy_and_accuracy_thresholds_come_from_frozen_config(tmp_path: Path):
    from experiments.llm_sim.config import LLMSimConfig, load_frozen_config
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import LLMSimulationRunner, _calibration_eligibility
    from experiments.llm_sim.store import SimulationStore

    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=FakeTransport(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    runner.run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    policy = runner._policies["deepseek"]
    frozen_policy = runner.config.provider_policy
    assert policy.max_attempts == frozen_policy["max_attempts"]
    assert policy.failure_threshold == frozen_policy["failure_threshold"]
    assert policy.base_backoff_seconds == frozen_policy["base_backoff_seconds"]
    assert policy.max_backoff_seconds == frozen_policy["max_backoff_seconds"]
    assert policy.cooldown_seconds == frozen_policy["cooldown_seconds"]

    raw = json.loads(json.dumps(load_frozen_config().raw))
    raw["accuracy_bands"]["weak_upper_exclusive"] = 0.0
    custom = LLMSimConfig.from_mapping(raw)
    persona = Persona.from_mapping(
        {
            "persona_id": "p:weak",
            "pair_id": "p",
            "strength": "weak",
            "target_node": "Target",
            "failure_id": "failure-1",
        }
    )
    result = _calibration_eligibility(
        strength="weak",
        personas=[persona],
        attempts=[
            {
                "persona_id": persona.persona_id,
                "correct": False,
                "target_misconception_hit": True,
                "random_wrong_option_baseline": 1 / 3,
            }
        ]
        * 4,
        panel_rows=[{"mapping_status": "mapped"}],
        calibration_ready=True,
        config=custom,
        prompt_revision=0,
    )
    assert result["accuracy"] == 0.0
    assert result["accuracy_gate"]["pass"] is False
    assert result["status"] == "prompt_rewrite_available"


def test_request_accounting_includes_retry_attempts(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner, ProviderCallPolicy
    from experiments.llm_sim.store import SimulationStore

    class OnceFlaky(FakeTransport):
        def complete(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                raise RuntimeError("429 rate limited")
            return super().complete(**kwargs)

    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=OnceFlaky(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
        policy_factory=lambda: ProviderCallPolicy(
            max_attempts=2,
            failure_threshold=3,
            base_backoff_seconds=0,
        ),
    )
    result = runner.run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert result["accounting"]["requests"] == 6
    assert result["accounting"]["retries"] == 1


def test_partial_checkpoint_resumes_without_repeating_completed_item(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner, ProviderCallPolicy
    from experiments.llm_sim.store import SimulationStore

    class FailAfterOne(FakeTransport):
        def complete(self, **kwargs):
            # Four calibration calls must complete; then one journey item is
            # checkpointed before the second journey call fails.
            if len(self.calls) >= 5:
                self.calls.append(kwargs)
                raise RuntimeError("503 provider unavailable")
            return super().complete(**kwargs)

    store = SimulationStore(tmp_path / "sim_store")
    base = dict(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=store,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
        policy_factory=lambda: ProviderCallPolicy(
            max_attempts=1,
            failure_threshold=2,
            base_backoff_seconds=0,
        ),
    )
    with pytest.raises(RuntimeError, match="503"):
        LLMSimulationRunner(**base, transport=FailAfterOne()).run_provider(
            "deepseek", model="model-a", max_items=2, arms=("A",)
        )
    checkpoint = store.read_json(store.checkpoint_relative_path("deepseek", "p:weak", "A"))
    assert len(checkpoint["events"]) == 1

    resumed_transport = FakeTransport()
    result = LLMSimulationRunner(**base, transport=resumed_transport).run_provider(
        "deepseek", model="model-a", max_items=2, arms=("A",), resume=True
    )
    assert len(resumed_transport.calls) == 1
    assert result["journeys"][0]["actual_administered_count"] == 2


def test_crash_resume_with_invalid_event_is_equivalent_to_uninterrupted_run(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner, ProviderCallPolicy
    from experiments.llm_sim.store import SimulationStore
    from experiments.llm_sim.transport import ProviderNetworkError

    class InvalidThenCrash(FakeTransport):
        def complete(self, **kwargs):
            index = len(self.calls)
            if index < 4:
                return super().complete(**kwargs)
            self.calls.append(kwargs)
            if index == 4:
                return {
                    "content": '{"answer":"not-an-option"}',
                    "model_returned": kwargs["model"],
                    "usage": {"input_tokens": 10, "output_tokens": 4},
                    "cost_yuan": 0.01,
                }
            raise ProviderNetworkError()

    common = dict(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
        policy_factory=lambda: ProviderCallPolicy(
            max_attempts=1,
            failure_threshold=3,
            base_backoff_seconds=0,
        ),
    )
    resumed_store = SimulationStore(tmp_path / "resumed" / "sim_store")
    with pytest.raises(ProviderNetworkError):
        LLMSimulationRunner(
            **common,
            store=resumed_store,
            transport=InvalidThenCrash(),
        ).run_provider("deepseek", model="model-a", max_items=2, arms=("B",))
    checkpoint = resumed_store.read_json(
        resumed_store.checkpoint_relative_path("deepseek", "p:weak", "B")
    )
    assert checkpoint["events"][0]["update_applied"] is False
    resumed = LLMSimulationRunner(
        **common,
        store=resumed_store,
        transport=FakeTransport(responses=['{"answer":"A"}']),
    ).run_provider("deepseek", model="model-a", max_items=2, arms=("B",))[
        "journeys"
    ][0]

    uninterrupted_store = SimulationStore(tmp_path / "uninterrupted" / "sim_store")
    uninterrupted = LLMSimulationRunner(
        **common,
        store=uninterrupted_store,
        transport=FakeTransport(
            responses=[
                '{"answer":"B"}',
                '{"answer":"B"}',
                '{"answer":"B"}',
                '{"answer":"B"}',
                '{"answer":"not-an-option"}',
                '{"answer":"A"}',
            ]
        ),
    ).run_provider("deepseek", model="model-a", max_items=2, arms=("B",))[
        "journeys"
    ][0]
    comparable_keys = (
        "item_id",
        "score_status",
        "update_applied",
        "direct_answers_before",
        "direct_answers_after",
        "prior_belief",
        "posterior_belief",
    )
    assert [
        {key: event[key] for key in comparable_keys} for event in resumed["events"]
    ] == [
        {key: event[key] for key in comparable_keys}
        for event in uninterrupted["events"]
    ]
    assert resumed["final_belief"] == uninterrupted["final_belief"]
    assert resumed["terminal_reason"] == uninterrupted["terminal_reason"]


def test_sparse_item_pool_stops_incomplete_without_exact_item_recycling(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    class SparseCatalog(FakeCatalog):
        def __init__(self):
            only = _item("only-item", distractor_map={"failure-1": "B"})
            self.items = {only.item_id: only}
            self._prerequisites = {"Target": ()}

    transport = FakeTransport()
    journey = LLMSimulationRunner(
        catalog=SparseCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    ).run_provider("deepseek", model="model-a", max_items=3, arms=("B",))[
        "journeys"
    ][0]
    assert journey["status"] == "incomplete"
    assert journey["terminal_reason"] == "structural_failure_item_pool"
    assert journey["actual_administered_count"] == 1
    assert [event["item_id"] for event in journey["events"]] == ["only-item"]
    assert len(transport.calls) == 1


def test_structural_incomplete_final_is_idempotently_resumed_but_no_resume_refuses(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    class SparseCatalog(FakeCatalog):
        def __init__(self):
            only = _item("only-item", distractor_map={"failure-1": "B"})
            self.items = {only.item_id: only}
            self._prerequisites = {"Target": ()}

    store = SimulationStore(tmp_path / "sim_store")
    base = dict(
        catalog=SparseCatalog(),
        kg=FakeKG(),
        store=store,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    first_transport = FakeTransport()
    first = LLMSimulationRunner(**base, transport=first_transport).run_provider(
        "deepseek", model="model-a", max_items=3, arms=("B",)
    )
    relative = store.journey_relative_path("deepseek", "p:weak", "B")
    path = store.root / relative
    raw_before = path.read_bytes()
    artifact_sha = next(
        row["sha256"]
        for row in first["manifest"]["artifacts"]
        if row["path"] == relative.as_posix()
    )
    created_at = first["manifest"]["created_at_utc"]
    assert first["journeys"][0]["status"] == "incomplete"
    assert first["accounting"]["failed"] == 1
    assert first["accounting"]["completed"] == 0

    resume_transport = FakeTransport()
    resumed = LLMSimulationRunner(**base, transport=resume_transport).run_provider(
        "deepseek", model="model-a", max_items=3, arms=("B",), resume=True
    )
    assert resume_transport.calls == []
    assert resumed["accounting"]["skipped"] == 1
    assert resumed["accounting"]["failed"] == 1
    assert resumed["accounting"]["completed"] == 0
    assert resumed["accounting"]["completed_by_arm"] == {"B": 0}
    assert resumed["journeys"][0]["status"] == "incomplete"
    assert resumed["journeys"][0]["terminal_reason"] == "structural_failure_item_pool"
    assert path.read_bytes() == raw_before
    assert resumed["manifest"]["created_at_utc"] == created_at
    assert next(
        row["sha256"]
        for row in resumed["manifest"]["artifacts"]
        if row["path"] == relative.as_posix()
    ) == artifact_sha

    no_resume_transport = FakeTransport()
    with pytest.raises(FileExistsError, match="--no-resume"):
        LLMSimulationRunner(**base, transport=no_resume_transport).run_provider(
            "deepseek", model="model-a", max_items=3, arms=("B",), resume=False
        )
    assert no_resume_transport.calls == []
    assert path.read_bytes() == raw_before


def test_six_official_provider_specs_never_point_at_application_api():
    from experiments.llm_sim.transport import PROVIDER_SPECS

    assert set(PROVIDER_SPECS) == {"deepseek", "glm", "kimi", "minimax", "doubao", "tongyi"}
    assert all(spec.base_url.startswith("https://") for spec in PROVIDER_SPECS.values())
    assert all("8700" not in spec.base_url for spec in PROVIDER_SPECS.values())


def test_live_environment_reads_repo_dotenv_in_memory_with_process_precedence(
    monkeypatch, tmp_path: Path
):
    from experiments.llm_sim.transport import (
        HTTPProviderTransport,
        load_live_environment,
        model_from_environment,
    )

    monkeypatch.delenv("DOUBAO_API_KEY", raising=False)
    monkeypatch.delenv("DOUBAO_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "DOUBAO_API_KEY=file-test-key\n"
        "DOUBAO_MODEL=doubao-seed-2-1-pro-260628\n",
        encoding="utf-8",
    )
    file_environment = load_live_environment(repo_root=tmp_path, environ={})
    transport = HTTPProviderTransport.from_environment(
        "doubao",
        environment=file_environment,
    )
    assert transport.spec.name == "doubao"
    assert model_from_environment(
        "doubao",
        "old-default",
        environment=file_environment,
    ) == "doubao-seed-2-1-pro-260628"
    assert "DOUBAO_API_KEY" not in __import__("os").environ

    process_environment = load_live_environment(
        repo_root=tmp_path,
        environ={
            "DOUBAO_API_KEY": "process-test-key",
            "DOUBAO_MODEL": "process-model",
        },
    )
    assert model_from_environment(
        "doubao",
        "old-default",
        environment=process_environment,
    ) == "process-model"


def test_live_cli_requires_bytecode_disabled_from_process_start(monkeypatch, tmp_path: Path):
    import sys

    from experiments.llm_sim.cli import main

    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    output = tmp_path / "sim_store"
    with pytest.raises(SystemExit):
        main(
            [
                "--live",
                "--provider",
                "deepseek",
                "--output-root",
                str(output),
            ]
        )
    assert not output.exists()


def test_prepare_never_reads_live_dotenv(monkeypatch, tmp_path: Path):
    from experiments.llm_sim import runner as runner_module
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    monkeypatch.setattr(
        runner_module,
        "load_live_environment",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepare must not read live dotenv")
        ),
    )
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=None,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    assert runner.prepare()["provider_observations"] == 0


def test_preparation_and_provider_manifest_bind_git_code_config_and_seed(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=FakeTransport(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    preparation = runner.prepare()
    result = runner.run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    for manifest in (preparation, result["manifest"]):
        assert manifest["run_id"] == "llm-personas-v1"
        assert len(manifest["git_head"]) == 40
        assert len(manifest["code_sha256"]) == 64
        assert manifest["working_code_sha256"] == manifest["code_sha256"]
        assert len(manifest["head_code_sha256"]) == 64
        assert isinstance(manifest["code_matches_head"], bool)
        assert manifest["analysis_plan_commit"] == runner.config.analysis_plan_commit
        assert manifest["analysis_plan_is_ancestor"] is True
        assert (
            manifest["h5_analysis_plan_commit"]
            == runner.config.h5_analysis_plan_commit
        )
        assert (
            manifest["h5_analysis_plan_sha256"]
            == runner.config.h5_analysis_plan_sha256
        )
        assert (
            manifest["h5_analysis_plan_committed_at_utc"]
            == runner.config.h5_analysis_plan_committed_at_utc
        )
        assert manifest["h5_analysis_plan_verified"] is True
        assert len(manifest["official_input_sha256"]) == 64
        assert manifest["config_sha256"] == runner.config.sha256
        assert manifest["study_seed"] == runner.study_seed
    assert preparation["code_files"]
    assert "experiments/config/llm_sim_v1.json" in {
        row["path"] for row in preparation["code_files"]
    }
    assert all(
        {"path", "sha256", "head_sha256", "matches_head"} <= set(row)
        for row in preparation["code_files"]
    )
    assert preparation["official_inputs"]["catalog_item_count"] == 4
    assert "answer_values" not in json.dumps(preparation["official_inputs"])
    assert result["manifest"]["git_head"] == preparation["git_head"]
    assert result["manifest"]["code_sha256"] == preparation["code_sha256"]
    assert result["manifest"]["official_input_sha256"] == preparation["official_input_sha256"]


def test_live_fails_closed_when_code_or_head_changed_after_prepare(monkeypatch, tmp_path: Path):
    from experiments.llm_sim import runner as runner_module
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    current = {
        "git_head": "1" * 40,
        "code_sha256": "2" * 64,
        "working_code_sha256": "2" * 64,
        "head_code_sha256": "2" * 64,
        "code_matches_head": True,
        "code_files": [
            {
                "path": "runner.py",
                "sha256": "3" * 64,
                "head_sha256": "3" * 64,
                "matches_head": True,
            }
        ],
    }
    monkeypatch.setattr(
        runner_module,
        "collect_code_provenance",
        lambda _root: json.loads(json.dumps(current)),
    )
    monkeypatch.setattr(runner_module, "analysis_plan_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(
        runner_module,
        "verify_frozen_document_commit",
        lambda _root, **kwargs: {
            "commit": kwargs["commit"],
            "path": kwargs["relative_path"],
            "sha256": kwargs["sha256"],
            "committed_at_utc": kwargs["committed_at_utc"],
            "is_ancestor": True,
            "verified": True,
        },
    )
    transport = FakeTransport()
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    runner.prepare()
    current["git_head"] = "4" * 40
    current["code_sha256"] = "5" * 64
    with pytest.raises(RuntimeError, match="changed after S2 preparation"):
        runner.run_provider(
            "deepseek", model="model-a", max_items=1, arms=("A",)
        )
    assert transport.calls == []


def test_official_live_transport_requires_scoped_code_byte_equal_to_head_before_dotenv(
    monkeypatch, tmp_path: Path
):
    from experiments.llm_sim import runner as runner_module
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    current = {
        "git_head": "1" * 40,
        "code_sha256": "2" * 64,
        "working_code_sha256": "2" * 64,
        "head_code_sha256": "3" * 64,
        "code_matches_head": False,
        "code_files": [
            {
                "path": "experiments/llm_sim/runner.py",
                "sha256": "4" * 64,
                "head_sha256": "5" * 64,
                "matches_head": False,
            }
        ],
    }
    monkeypatch.setattr(
        runner_module,
        "collect_code_provenance",
        lambda _root: json.loads(json.dumps(current)),
    )
    monkeypatch.setattr(runner_module, "analysis_plan_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(
        runner_module,
        "verify_frozen_document_commit",
        lambda _root, **kwargs: {
            "commit": kwargs["commit"],
            "path": kwargs["relative_path"],
            "sha256": kwargs["sha256"],
            "committed_at_utc": kwargs["committed_at_utc"],
            "is_ancestor": True,
            "verified": True,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "load_live_environment",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dotenv must not be read before scoped HEAD binding")
        ),
    )
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=None,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="byte-equal to recorded HEAD"):
        runner.run_provider("deepseek", model="model-a", max_items=1, arms=("A",))


def test_live_fails_closed_when_official_question_input_changed_after_prepare(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    catalog = FakeCatalog()
    transport = FakeTransport()
    runner = LLMSimulationRunner(
        catalog=catalog,
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    runner.prepare()
    catalog.items["i1"].stem_text = "Question changed after preparation"
    with pytest.raises(RuntimeError, match="official input changed after S2 preparation"):
        runner.run_provider(
            "deepseek", model="model-a", max_items=1, arms=("A",)
        )
    assert transport.calls == []


def test_failed_prompt_v0_calibration_requires_rewrite_and_never_qualifies_cell(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    # A weak persona answering every calibration item correctly violates the
    # frozen <0.40 accuracy band.  Its journey is retained as an observation,
    # but cannot count toward a reportable provider-arm cell.
    transport = FakeTransport(responses=['{"answer":"A"}'] * 5)
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    result = runner.run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    assert result["provider_eligibility"]["weak"]["status"] == "prompt_rewrite_available"
    assert result["status"] == "calibration_rewrite_required"
    assert result["manifest"]["reportability"]["eligible_cell_completed"] == {"A": 0}
    assert result["manifest"]["reportability"]["completion_reportable"] is False
    assert result["journeys"][0]["status"] == "complete"


def test_failed_prompt_v1_calibration_is_excluded_post_calibration(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    transport = FakeTransport(responses=['{"answer":"A"}'] * 5)
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    v0 = runner.run_provider(
        "deepseek",
        model="model-a",
        max_items=1,
        arms=("A",),
        prompt_revision=0,
    )
    assert v0["status"] == "calibration_rewrite_required"
    runner.transport = FakeTransport(responses=['{"answer":"A"}'] * 5)
    result = runner.run_provider(
        "deepseek",
        model="model-a",
        max_items=1,
        arms=("A",),
        prompt_revision=1,
    )
    assert result["provider_eligibility"]["weak"]["status"] == "excluded_post_calibration"
    assert result["status"] == "excluded_post_calibration"
    assert result["manifest"]["reportability"]["eligible_cell_completed"] == {"A": 0}
    assert result["manifest"]["reportability"]["reportable"] is False


def test_prompt_v1_requires_matching_immutable_rewrite_required_v0_before_transport(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    transport = FakeTransport()
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "missing-v0" / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    with pytest.raises(ValueError, match="prompt revision 1 requires"):
        runner.run_provider(
            "deepseek",
            model="model-a",
            max_items=1,
            arms=("A",),
            prompt_revision=1,
        )
    assert transport.calls == []

    passing_transport = FakeTransport(
        responses=[
            '{"answer":"B"}',
            '{"answer":"C"}',
            '{"answer":"B"}',
            '{"answer":"C"}',
            '{"answer":"B"}',
        ]
    )
    passing = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "passing-v0" / "sim_store"),
        transport=passing_transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
    )
    assert passing.run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )["status"] != "calibration_rewrite_required"
    decision = passing.store.read_json("calibration_decisions/deepseek.json")
    assert decision["status"] != "calibration_rewrite_required"
    assert decision["decision_sha256"]
    provider_manifest_path = passing.store.root / "providers" / "deepseek.json"
    mutable_manifest = json.loads(provider_manifest_path.read_text(encoding="utf-8"))
    mutable_manifest["status"] = "calibration_rewrite_required"
    provider_manifest_path.write_text(json.dumps(mutable_manifest), encoding="utf-8")
    passing.transport = FakeTransport()
    with pytest.raises(ValueError, match="prompt revision 1 requires"):
        passing.run_provider(
            "deepseek",
            model="model-a",
            max_items=1,
            arms=("A",),
            prompt_revision=1,
        )
    assert passing.transport.calls == []


def test_frozen_llm_config_exposes_exact_s2_grid():
    from experiments.llm_sim.config import (
        FROZEN_H5_ANALYSIS_PLAN_COMMIT,
        FROZEN_H5_ANALYSIS_PLAN_COMMITTED_AT_UTC,
        FROZEN_H5_ANALYSIS_PLAN_SHA256,
        load_frozen_config,
    )

    config = load_frozen_config()
    assert config.run_id == "llm-personas-v1"
    assert config.pair_count == 25
    assert config.persona_count == 50
    assert config.arms == ("A", "B")
    assert config.providers == ("deepseek", "glm", "kimi", "minimax", "doubao", "tongyi")
    assert config.minimum_complete_per_cell == 45
    assert config.maximum_prompt_rewrites == 1
    assert config.study_seed == 2026071302
    assert config.max_items == 15
    assert config.weak_accuracy_upper == 0.4
    assert config.strong_accuracy_lower == 0.75
    assert config.manipulation_bootstrap_resamples == 10_000
    assert isinstance(config.manipulation_bootstrap_seed, int)
    assert config.provider_policy["max_attempts"] == 3
    assert config.frozen_pre_observation_utc.endswith("Z")
    assert config.persona_seed_derivation_version == "yher-llm-persona-v2"
    assert config.h5_analysis_plan_commit == FROZEN_H5_ANALYSIS_PLAN_COMMIT
    assert config.h5_analysis_plan_sha256 == FROZEN_H5_ANALYSIS_PLAN_SHA256
    assert (
        config.h5_analysis_plan_committed_at_utc
        == FROZEN_H5_ANALYSIS_PLAN_COMMITTED_AT_UTC
    )
    assert config.frozen_pre_observation_utc == config.h5_analysis_plan_committed_at_utc
    assert config.prompt_version == "yher-llm-persona-prompt-v1"
    assert config.manipulation_mapping_policy == "explicit_machine_annotation_only"


def test_h5_amendment_provenance_is_recomputed_from_git_commit():
    from experiments.llm_sim.config import load_frozen_config
    from experiments.llm_sim.provenance import (
        collect_code_provenance,
        verify_frozen_document_commit,
    )

    config = load_frozen_config()
    repo_root = Path(__file__).resolve().parents[1]
    head = collect_code_provenance(repo_root)["git_head"]
    proof = verify_frozen_document_commit(
        repo_root,
        commit=config.h5_analysis_plan_commit,
        relative_path="experiments/h5_analysis_plan.md",
        sha256=config.h5_analysis_plan_sha256,
        committed_at_utc=config.h5_analysis_plan_committed_at_utc,
        head=head,
    )
    assert proof == {
        "commit": config.h5_analysis_plan_commit,
        "path": "experiments/h5_analysis_plan.md",
        "sha256": config.h5_analysis_plan_sha256,
        "committed_at_utc": config.h5_analysis_plan_committed_at_utc,
        "is_ancestor": True,
        "verified": True,
    }
    with pytest.raises(RuntimeError, match="blob hash"):
        verify_frozen_document_commit(
            repo_root,
            commit=config.h5_analysis_plan_commit,
            relative_path="experiments/h5_analysis_plan.md",
            sha256="0" * 64,
            committed_at_utc=config.h5_analysis_plan_committed_at_utc,
            head=head,
        )
    with pytest.raises(RuntimeError, match="commit time"):
        verify_frozen_document_commit(
            repo_root,
            commit=config.h5_analysis_plan_commit,
            relative_path="experiments/h5_analysis_plan.md",
            sha256=config.h5_analysis_plan_sha256,
            committed_at_utc="2099-01-01T00:00:00Z",
            head=head,
        )


def test_formal_design_check_requires_exact_frozen_grid():
    from experiments.llm_sim.config import load_frozen_config
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import _formal_design_check

    config = load_frozen_config()
    personas = [Persona.from_mapping(row) for row in _paired_personas(25)]
    matched = _formal_design_check(
        personas=personas,
        study_seed=config.study_seed,
        max_items=config.max_items,
        arms=config.arms,
        config=config,
    )
    assert matched["match"] is True
    assert matched["composition"] == {"weak": 25, "strong": 25}
    assert matched["failures"] == []

    noncanonical = _formal_design_check(
        personas=personas,
        study_seed=config.study_seed,
        max_items=config.max_items,
        arms=config.arms,
        config=config,
        canonical_personas_sha256="a" * 64,
        persona_panel_personas_sha256="b" * 64,
    )
    assert noncanonical["match"] is False
    assert "canonical_persona_panel" in noncanonical["failures"]

    smoke = _formal_design_check(
        personas=personas[:2],
        study_seed=config.study_seed + 1,
        max_items=1,
        arms=("A",),
        config=config,
    )
    assert smoke["match"] is False
    assert set(smoke["failures"]) == {
        "persona_composition",
        "pair_composition",
        "study_seed",
        "max_items",
        "arms",
    }


@pytest.mark.parametrize(
    ("mutation", "failure_reason"),
    (
        (lambda rows: rows[1].__setitem__("target_node", "Other"), "pair_target_node"),
        (lambda rows: rows[1].__setitem__("failure_id", "failure-2"), "pair_failure_id"),
        (lambda rows: rows[1].__setitem__("seed", 99), "pair_seed"),
        (
            lambda rows: rows[1].__setitem__("failure_cause", "different cause"),
            "pair_failure_definition",
        ),
        (
            lambda rows: rows[1].__setitem__("persona_id", rows[0]["persona_id"]),
            "persona_id_uniqueness",
        ),
    ),
)
def test_formal_design_rejects_adversarial_pair_invariant_mismatches(
    mutation, failure_reason
):
    from experiments.llm_sim.config import load_frozen_config
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import _formal_design_check

    config = load_frozen_config()
    rows = _paired_personas(25)
    mutation(rows)
    result = _formal_design_check(
        personas=[Persona.from_mapping(row) for row in rows],
        study_seed=config.study_seed,
        max_items=config.max_items,
        arms=config.arms,
        config=config,
    )
    assert result["match"] is False
    assert failure_reason in result["failures"]


def test_cli_default_output_is_scoped_by_frozen_run_id():
    from experiments.llm_sim.cli import build_parser

    args = build_parser().parse_args([])
    assert args.output_root == "data/sim_store/llm_personas/llm-personas-v1"


def test_prepare_freezes_panel_without_resolving_transport(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    def forbidden_transport(_provider):
        raise AssertionError("prepare must not resolve or call a provider transport")

    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=forbidden_transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    summary = runner.prepare()
    assert summary["persona_count"] == 1
    assert summary["mapped_count"] == 1
    assert summary["provider_observations"] == 0


def test_cli_prepare_only_is_non_network_and_writes_only_sim_store(tmp_path: Path):
    from experiments.llm_sim.cli import main

    output = tmp_path / "sim_store"
    assert main(["--prepare-only", "--output-root", str(output)]) == 0
    assert (output / "manipulation_panel.json").is_file()
    manifest = json.loads((output / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider_observations"] == 0
    assert manifest["simulated"] is True


def test_returned_model_drift_is_not_silently_accepted(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner, ModelDriftError
    from experiments.llm_sim.store import SimulationStore

    class DriftTransport(FakeTransport):
        def complete(self, **kwargs):
            result = dict(super().complete(**kwargs))
            result["model_returned"] = "unexpected-model"
            return result

    store = SimulationStore(tmp_path / "sim_store")
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=store,
        transport=DriftTransport(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    with pytest.raises(ModelDriftError):
        runner.run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    manifest = store.read_json("providers/deepseek.json")
    assert manifest["failure_category"] == "model_id_drift"
    assert manifest["accounting"]["responses"] == 1
    assert manifest["accounting"]["input_tokens"] == 10
    assert manifest["accounting"]["output_tokens"] == 4
    assert manifest["accounting"]["cost_yuan"] == 0.01
    excluded = [
        row
        for row in manifest["artifacts"]
        if row["record_type"] == "llm_sim_excluded_response_accounting"
    ]
    assert len(excluded) == 1
    excluded_record = store.read_json(excluded[0]["path"])
    assert excluded_record["failure_category"] == "model_id_drift"
    assert excluded_record["usage"] == {"input_tokens": 10, "output_tokens": 4}
    assert "content" not in excluded_record


def test_missing_provider_configuration_is_pre_outcome_excluded_and_key_safe(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore
    from experiments.llm_sim.transport import ProviderConfigurationError

    def unavailable(_provider):
        raise ProviderConfigurationError("missing API key for provider deepseek")

    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=unavailable,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    result = runner.run_provider("deepseek", model="model-a", max_items=1)
    assert result["status"] == "excluded_pre_outcome"
    assert result["exclusion_reason"] == "provider_configuration_unavailable"
    serialized = json.dumps(result)
    assert "API key" not in serialized


def test_failure_categories_are_sanitized_and_exhaustive():
    from experiments.llm_sim.runner import (
        CircuitOpenError,
        ModelDriftError,
        _failure_category,
    )
    from experiments.llm_sim.transport import (
        ProviderHTTPError,
        ProviderNetworkError,
        ProviderProtocolError,
    )

    assert _failure_category(ModelDriftError("drift")) == "model_id_drift"
    assert _failure_category(ProviderHTTPError(429)) == "http_status"
    assert _failure_category(ProviderNetworkError()) == "network"
    assert _failure_category(ProviderProtocolError()) == "protocol"
    assert _failure_category(CircuitOpenError("open")) == "circuit"
    assert _failure_category(RuntimeError("secret response body")) == "unexpected"


def test_interruption_manifest_accounts_persisted_events_without_secret_error_text(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner, ProviderCallPolicy
    from experiments.llm_sim.store import SimulationStore
    from experiments.llm_sim.transport import ProviderNetworkError

    class OneThenNetwork(FakeTransport):
        def complete(self, **kwargs):
            if not self.calls:
                return super().complete(**kwargs)
            self.calls.append(kwargs)
            raise ProviderNetworkError()

    store = SimulationStore(tmp_path / "sim_store")
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=store,
        transport=OneThenNetwork(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        repo_root=tmp_path,
        policy_factory=lambda: ProviderCallPolicy(
            max_attempts=1,
            failure_threshold=3,
            base_backoff_seconds=0,
        ),
    )
    with pytest.raises(ProviderNetworkError):
        runner.run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    manifest = store.read_json("providers/deepseek.json")
    assert manifest["status"] == "interrupted_calibration"
    assert manifest["failure_category"] == "network"
    assert manifest["accounting"]["responses"] == 1
    assert manifest["accounting"]["input_tokens"] == 10
    assert manifest["accounting"]["output_tokens"] == 4
    assert manifest["accounting"]["cost_yuan"] == 0.01
    assert any(row["path"].endswith(".partial.json") for row in manifest["artifacts"])
    serialized = json.dumps(manifest)
    assert "secret response body" not in serialized
    assert "API key" not in serialized


def test_real_production_scoring_mastery_selector_are_used(monkeypatch, tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore
    from engine import mastery, selector

    observed = {"score": 0, "observe": 0, "select": 0}
    from core.learning import scoring

    original_score = scoring.score_item
    original_observe = mastery.observe
    original_select = selector.select_next

    def score(*args, **kwargs):
        observed["score"] += 1
        return original_score(*args, **kwargs)

    def observe(*args, **kwargs):
        observed["observe"] += 1
        return original_observe(*args, **kwargs)

    def select(*args, **kwargs):
        observed["select"] += 1
        return original_select(*args, **kwargs)

    monkeypatch.setattr(scoring, "score_item", score)
    monkeypatch.setattr(mastery, "observe", observe)
    monkeypatch.setattr(selector, "select_next", select)
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=FakeTransport(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    runner.run_provider("deepseek", model="model-a", max_items=1)
    assert observed["score"] == 6
    assert observed["observe"] == 2
    assert observed["select"] >= 1


def test_numeric_prompt_requests_a_numeric_string_without_mcq_example_or_answer_key():
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import _messages, _parse_answer

    numeric_item = _item("numeric-1")
    numeric_item.scoring_mode = "numeric"
    numeric_item.item_type = "numeric"
    numeric_item.options = {}
    numeric_item.answer_values = ("42",)
    persona = Persona(
        persona_id="p:strong",
        pair_id="p",
        strength="strong",
        target_node="Target",
        failure_id="failure-1",
        failure_cause="",
        failure_symptom="",
        diagnostic_question="",
    )
    messages = _messages(persona, numeric_item)
    serialized = json.dumps(messages, ensure_ascii=False)
    assert '"answer": "<numeric string>"' in messages[0]["content"]
    assert '"answer": "A"' not in messages[0]["content"]
    assert "42" not in serialized
    assert _parse_answer('{"answer":"3.50"}', response_kind="numeric") == "3.50"


def test_prompt_revision_one_is_a_substantive_frozen_rewrite_without_answer_leak():
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import _messages

    item = _item("i-rewrite", answer="D")
    weak = Persona(
        persona_id="p:weak",
        pair_id="p",
        strength="weak",
        target_node="Target",
        failure_id="failure-1",
        failure_cause="confuses oxidation and reduction",
        failure_symptom="selects the reversed direction",
        diagnostic_question="",
    )
    strong = Persona(
        persona_id="p:strong",
        pair_id="p",
        strength="strong",
        target_node="Target",
        failure_id="failure-1",
        failure_cause="",
        failure_symptom="",
        diagnostic_question="",
    )
    weak_v0 = _messages(weak, item, prompt_revision=0)
    weak_v1 = _messages(weak, item, prompt_revision=1)
    strong_v0 = _messages(strong, item, prompt_revision=0)
    strong_v1 = _messages(strong, item, prompt_revision=1)
    assert weak_v0 != weak_v1
    assert strong_v0 != strong_v1
    assert "do not answer as an expert" in weak_v1[0]["content"]
    assert "Recheck the chemistry and calculation" in strong_v1[0]["content"]
    for messages in (weak_v0, weak_v1, strong_v0, strong_v1):
        serialized = json.dumps(messages, ensure_ascii=False)
        assert '"answer": "D"' not in serialized
        assert "answer_values" not in serialized


def test_target_misconception_denominator_uses_only_the_frozen_target_item(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    transport = FakeTransport(
        responses=[
            '{"answer":"B","rationale":"calibration"}',
            '{"answer":"B","rationale":"calibration"}',
            '{"answer":"B","rationale":"calibration"}',
            '{"answer":"B","rationale":"calibration"}',
            '{"answer":"B","rationale":"wrong non-target"}',
            '{"answer":"B","rationale":"frozen target"}',
        ]
    )
    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=transport,
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    journey = runner.run_provider(
        "deepseek", model="model-a", max_items=2, arms=("B",)
    )["journeys"][0]
    assert journey["target_misconception_wrong_denominator"] == 1
    assert journey["target_misconception_hit_count"] == 1


def test_arm_a_can_select_real_prerequisite_items_without_direct_count(monkeypatch, tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore
    from engine import selector

    class PrereqCatalog(FakeCatalog):
        def __init__(self):
            super().__init__()
            prereq = _item("prereq-1", node="Prerequisite")
            self.items[prereq.item_id] = prereq
            self._prerequisites = {"Target": ("Prerequisite",)}

        def prerequisites_for(self, node: str):
            return self._prerequisites.get(node, ())

    original = selector.select_next

    def choose_prerequisite(candidates, *args, **kwargs):
        return next(row for row in candidates if row["role"] == "prereq")

    monkeypatch.setattr(selector, "select_next", choose_prerequisite)
    runner = LLMSimulationRunner(
        catalog=PrereqCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
        transport=FakeTransport(),
        personas=[
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        panel_path=tmp_path / "sim_store" / "panel.json",
        repo_root=tmp_path,
    )
    journey = runner.run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )["journeys"][0]
    assert journey["events"][0]["role"] == "prereq"
    assert journey["events"][0]["direct_answers_after"] == 0
    monkeypatch.setattr(selector, "select_next", original)


def _canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _one_persona_runner_kwargs(tmp_path: Path, *, store, transport):
    return {
        "catalog": FakeCatalog(),
        "kg": FakeKG(),
        "store": store,
        "transport": transport,
        "personas": [
            {
                "persona_id": "p:weak",
                "pair_id": "p",
                "strength": "weak",
                "target_node": "Target",
                "failure_id": "failure-1",
            }
        ],
        "repo_root": tmp_path,
    }


def test_attempt_ledger_survives_drift_then_resume_and_keeps_provider_excluded(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner, ModelDriftError
    from experiments.llm_sim.store import SimulationStore

    class DriftOnce(FakeTransport):
        def complete(self, **kwargs):
            result = super().complete(**kwargs)
            result["model_returned"] = "unexpected-model"
            return result

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(tmp_path, store=store, transport=DriftOnce())
    with pytest.raises(ModelDriftError):
        LLMSimulationRunner(**base).run_provider(
            "deepseek", model="model-a", max_items=1, arms=("A",)
        )

    resumed = LLMSimulationRunner(
        **{**base, "transport": FakeTransport()}
    ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))

    attempt_rows = [
        row
        for row in resumed["manifest"]["artifacts"]
        if row["record_type"] == "llm_sim_provider_attempt"
    ]
    assert len(attempt_rows) == 6
    assert resumed["accounting"]["requests"] == 6
    assert resumed["accounting"]["responses"] == 6
    assert resumed["accounting"]["input_tokens"] == 60
    assert resumed["accounting"]["output_tokens"] == 24
    assert resumed["accounting"]["cost_yuan"] == 0.06
    assert resumed["status"] == "excluded_model_drift"
    assert resumed["manifest"]["reportability"]["model_drift_detected"] is True
    assert resumed["manifest"]["reportability"]["reportable"] is False


def test_retry_attempt_ledger_and_totals_survive_a_no_call_resume(tmp_path: Path):
    from experiments.llm_sim.runner import LLMSimulationRunner, ProviderCallPolicy
    from experiments.llm_sim.store import SimulationStore

    class OnceFlaky(FakeTransport):
        def complete(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                raise RuntimeError("429 rate limited")
            return super().complete(**kwargs)

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=OnceFlaky()
    )
    base["policy_factory"] = lambda: ProviderCallPolicy(
        max_attempts=2,
        failure_threshold=3,
        base_backoff_seconds=0,
    )
    first = LLMSimulationRunner(**base).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    assert first["accounting"]["requests"] == 6
    assert first["accounting"]["retries"] == 1

    transport = FakeTransport()
    resumed = LLMSimulationRunner(
        **{**base, "transport": transport}
    ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))

    assert transport.calls == []
    assert resumed["accounting"]["requests"] == 6
    assert resumed["accounting"]["retries"] == 1
    attempts = [
        store.read_json(row["path"])
        for row in resumed["manifest"]["artifacts"]
        if row["record_type"] == "llm_sim_provider_attempt"
    ]
    assert sum(row["status"] == "failed" for row in attempts) == 1
    assert sum(row["retry_number"] == 1 for row in attempts) == 1


def test_complete_journey_validation_rejects_duplicate_over_budget_events():
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import _complete_existing

    persona = Persona.from_mapping(
        {
            "persona_id": "p:weak",
            "pair_id": "p",
            "strength": "weak",
            "target_node": "Target",
            "failure_id": "failure-1",
        }
    )
    event = {
        "simulated": True,
        "run_id": "llm-personas-v1",
        "persona_id": persona.persona_id,
        "provider": "deepseek",
        "model_id": "model-a",
        "record_type": "llm_sim_event",
        "pair_id": persona.pair_id,
        "strength": persona.strength,
        "arm": "A",
        "position": 1,
        "item_id": "i1",
    }
    record = {
        "simulated": True,
        "run_id": "llm-personas-v1",
        "persona_id": persona.persona_id,
        "provider": "deepseek",
        "model_id": "model-a",
        "record_type": "llm_sim_journey",
        "status": "complete",
        "pair_id": persona.pair_id,
        "strength": persona.strength,
        "arm": "A",
        "max_items": 1,
        "actual_administered_count": 2,
        "terminal_reason": "budget_exhausted",
        "events": [event, {**event, "position": 2}],
        "panel_sha256": "panel",
        "config_sha256": "config",
        "persona_panel_sha256": "personas",
        "study_seed": 7,
        "analysis_plan_commit": "plan",
        "prompt_version": "prompt",
        "prompt_revision": 0,
    }
    assert not _complete_existing(
        record,
        provider="deepseek",
        requested_model="model-a",
        persona=persona,
        arm="A",
        max_items=1,
        panel_sha="panel",
        prompt_revision=0,
        config_sha256="config",
        persona_panel_sha256="personas",
        study_seed=7,
        analysis_plan_commit="plan",
        prompt_version="prompt",
    )


def test_malformed_checkpoints_raise_instead_of_being_silently_restarted():
    from experiments.llm_sim.models import Persona
    from experiments.llm_sim.runner import (
        _resume_calibration_events,
        _resume_events,
    )

    persona = Persona.from_mapping(
        {
            "persona_id": "p:weak",
            "pair_id": "p",
            "strength": "weak",
            "target_node": "Target",
            "failure_id": "failure-1",
        }
    )
    envelope = {
        "simulated": True,
        "run_id": "llm-personas-v1",
        "persona_id": persona.persona_id,
        "provider": "deepseek",
        "model_id": "model-a",
    }
    bad_journey = {
        **envelope,
        "record_type": "llm_sim_journey",
        "status": "in_progress",
        "arm": "A",
        "max_items": 1,
        "panel_sha256": "panel",
        "prompt_version": "prompt",
        "prompt_revision": 0,
        "events": [
            {**envelope, "record_type": "llm_sim_event", "position": 1, "item_id": "i1"},
            {**envelope, "record_type": "llm_sim_event", "position": 2, "item_id": "i1"},
        ],
    }
    with pytest.raises(ValueError, match="checkpoint"):
        _resume_events(
            bad_journey,
            provider="deepseek",
            model="model-a",
            persona=persona,
            arm="A",
            max_items=1,
            panel_sha="panel",
            prompt_revision=0,
            prompt_version="prompt",
        )

    calibration_items = [{"item_id": f"i{index}"} for index in range(1, 5)]
    bad_calibration = {
        **envelope,
        "record_type": "llm_sim_calibration",
        "status": "in_progress",
        "panel_sha256": "panel",
        "prompt_version": "prompt",
        "prompt_revision": 0,
        "events": [
            {
                **envelope,
                "record_type": "llm_sim_calibration_attempt",
                "position": 1,
                "item_id": "i4",
            }
        ],
    }
    with pytest.raises(ValueError, match="calibration checkpoint"):
        _resume_calibration_events(
            bad_calibration,
            provider="deepseek",
            model="model-a",
            persona=persona,
            panel_sha="panel",
            prompt_revision=0,
            prompt_version="prompt",
            calibration_items=calibration_items,
        )


def test_self_rehashed_panel_is_rederived_and_rejected_before_transport(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=FakeTransport()
    )
    LLMSimulationRunner(**base).prepare()

    panel_path = store.root / "manipulation_panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel["annotations"][0]["target_option"] = "C"
    panel_core = {key: value for key, value in panel.items() if key != "panel_sha256"}
    panel["panel_sha256"] = _canonical_sha(panel_core)
    panel_path.write_text(json.dumps(panel, sort_keys=True), encoding="utf-8")

    persona_path = store.root / "persona_panel.json"
    persona_panel = json.loads(persona_path.read_text(encoding="utf-8"))
    persona_panel["manipulation_panel_sha256"] = panel["panel_sha256"]
    persona_core = {
        key: value
        for key, value in persona_panel.items()
        if key != "persona_panel_sha256"
    }
    persona_panel["persona_panel_sha256"] = _canonical_sha(persona_core)
    persona_path.write_text(json.dumps(persona_panel, sort_keys=True), encoding="utf-8")

    preparation_path = store.root / "preparation_manifest.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["panel_sha256"] = panel["panel_sha256"]
    preparation["persona_panel_sha256"] = persona_panel["persona_panel_sha256"]
    preparation_path.write_text(json.dumps(preparation, sort_keys=True), encoding="utf-8")

    transport = FakeTransport()
    with pytest.raises(ValueError, match="rederived manipulation panel"):
        LLMSimulationRunner(
            **{**base, "transport": transport}
        ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert transport.calls == []


def test_missing_response_model_id_is_protocol_failure_not_requested_model_fallback(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore
    from experiments.llm_sim.transport import ProviderProtocolError

    class MissingModel(FakeTransport):
        def complete(self, **kwargs):
            result = super().complete(**kwargs)
            del result["model_returned"]
            return result

    store = SimulationStore(tmp_path / "sim_store")
    with pytest.raises(ProviderProtocolError):
        LLMSimulationRunner(
            **_one_persona_runner_kwargs(
                tmp_path, store=store, transport=MissingModel()
            )
        ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    attempts = list((store.root / "attempts").rglob("*.json"))
    assert len(attempts) == 1
    record = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert record["returned_model_id"] is None
    assert record["model_id"] != "model-a"
    assert record["failure_category"] == "protocol"


@pytest.mark.parametrize(
    "prior_kind",
    ("provider_manifest", "calibration_decision", "excluded_response", "attempt"),
)
def test_no_resume_rejects_every_prior_observation_kind_before_transport(
    tmp_path: Path, prior_kind: str
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    calls = []

    def transport_factory(provider):
        calls.append(provider)
        return FakeTransport()

    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=transport_factory
    )
    runner = LLMSimulationRunner(**base)
    runner.prepare()
    common = {
        "simulated": True,
        "run_id": runner.config.run_id,
        "persona_id": "p:weak",
        "provider": "deepseek",
        "model_id": "model-a",
        "status": "recorded",
    }
    if prior_kind == "provider_manifest":
        relative = "providers/deepseek.json"
        record = {**common, "record_type": "llm_sim_provider_manifest", "artifacts": []}
    elif prior_kind == "calibration_decision":
        relative = "calibration_decisions/deepseek.json"
        record = {**common, "record_type": "llm_sim_calibration_decision"}
    elif prior_kind == "excluded_response":
        relative = store.excluded_response_relative_path(
            "deepseek", "p:weak", phase="calibration", position=1
        )
        record = {**common, "record_type": "llm_sim_excluded_response_accounting"}
    else:
        relative = "attempts/deepseek/prior.json"
        record = {**common, "record_type": "llm_sim_provider_attempt"}
    store.write_json(relative, record, immutable=True)

    with pytest.raises(FileExistsError, match="--no-resume"):
        runner.run_provider(
            "deepseek",
            model="model-a",
            max_items=1,
            arms=("A",),
            resume=False,
        )
    assert calls == []


def test_direct_http_transport_requires_python_b_but_fake_transport_remains_offline_seam(
    monkeypatch, tmp_path: Path
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore
    from experiments.llm_sim.transport import HTTPProviderTransport, provider_spec

    class SafeHTTP(HTTPProviderTransport):
        def complete(self, **kwargs):
            return {
                "content": '{"answer":"B"}',
                "model_returned": kwargs["model"],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "cost_yuan": 0.0,
            }

    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    http = SafeHTTP(provider_spec("deepseek"), "not-a-real-key")
    with pytest.raises(RuntimeError, match="Python -B"):
        LLMSimulationRunner(
            **_one_persona_runner_kwargs(
                tmp_path,
                store=SimulationStore(tmp_path / "http" / "sim_store"),
                transport=http,
            )
        ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))

    fake = FakeTransport()
    LLMSimulationRunner(
        **_one_persona_runner_kwargs(
            tmp_path,
            store=SimulationStore(tmp_path / "fake" / "sim_store"),
            transport=fake,
        )
    ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert fake.calls


@pytest.mark.parametrize(
    "missing",
    ("preparation_manifest.json", "manipulation_panel.json", "persona_panel.json"),
)
def test_missing_frozen_singleton_fails_instead_of_being_regenerated(
    tmp_path: Path, missing: str
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=FakeTransport()
    )
    LLMSimulationRunner(**base).prepare()
    (store.root / missing).unlink()
    transport = FakeTransport()

    with pytest.raises((FileNotFoundError, RuntimeError), match="frozen.*missing"):
        LLMSimulationRunner(
            **{**base, "transport": transport}
        ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert transport.calls == []


@pytest.mark.parametrize("artifact_kind", ("calibration", "journey"))
def test_rehashed_malformed_manifest_artifact_fails_before_transport_resolution(
    tmp_path: Path, artifact_kind: str
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=FakeTransport()
    )
    LLMSimulationRunner(**base).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    if artifact_kind == "calibration":
        relative = store.calibration_relative_path("deepseek", "p:weak")
        record = store.read_json(relative)
        record["events"][0]["item_id"] = record["events"][-1]["item_id"]
    else:
        relative = store.journey_relative_path("deepseek", "p:weak", "A")
        record = store.read_json(relative)
        duplicate = {**record["events"][0], "position": 2}
        record["events"].append(duplicate)
        record["actual_administered_count"] = 2
    artifact_path = store.root / relative
    artifact_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = store.root / "providers" / "deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["artifacts"]:
        if row["path"] == relative.as_posix():
            row["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    resolved = []

    def transport_factory(provider):
        resolved.append(provider)
        return FakeTransport()

    with pytest.raises(ValueError, match=f"provider manifest artifact.*{artifact_kind}"):
        LLMSimulationRunner(
            **{**base, "transport": transport_factory}
        ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert resolved == []


def test_rehashed_empty_confidence_final_fails_before_transport_resolution(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=FakeTransport()
    )
    LLMSimulationRunner(**base).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )

    relative = store.journey_relative_path("deepseek", "p:weak", "A")
    artifact_path = store.root / relative
    journey = store.read_json(relative)
    journey.update(
        {
            "status": "complete",
            "terminal_reason": "confidence",
            "events": [],
            "actual_administered_count": 0,
            "final_belief": [0.25, 0.25, 0.25, 0.25],
            "accuracy": None,
            "target_misconception_hit_count": 0,
            "target_misconception_wrong_denominator": 0,
        }
    )
    artifact_path.write_text(
        json.dumps(journey, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = store.root / "providers" / "deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["artifacts"]:
        if row["path"] == relative.as_posix():
            row["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    resolved = []
    with pytest.raises(ValueError, match="provider manifest artifact.*journey"):
        LLMSimulationRunner(
            **{
                **base,
                "transport": lambda provider: resolved.append(provider)
                or FakeTransport(),
            }
        ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert resolved == []


@pytest.mark.parametrize("source_kind", ("direct", "mapping", "callable"))
def test_every_resolved_http_transport_runs_the_official_ready_gate(
    monkeypatch, tmp_path: Path, source_kind: str
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore
    from experiments.llm_sim.transport import HTTPProviderTransport, provider_spec

    class SafeHTTP(HTTPProviderTransport):
        def __init__(self):
            super().__init__(provider_spec("deepseek"), "not-a-real-key")
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "content": '{"answer":"B"}',
                "model_returned": kwargs["model"],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "cost_yuan": 0.0,
            }

    store = SimulationStore(tmp_path / "sim_store")
    runner = LLMSimulationRunner(
        **_one_persona_runner_kwargs(
            tmp_path, store=store, transport=FakeTransport()
        )
    )
    runner.prepare()
    http = SafeHTTP()
    runner.transport = (
        http
        if source_kind == "direct"
        else {"deepseek": http}
        if source_kind == "mapping"
        else lambda _provider: http
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    gates = []

    def reject_not_ready():
        gates.append("official-ready")
        raise RuntimeError("official-ready")

    monkeypatch.setattr(runner, "_assert_official_live_ready", reject_not_ready)
    with pytest.raises(RuntimeError, match="official-ready"):
        runner.run_provider(
            "deepseek", model="model-a", max_items=1, arms=("A",)
        )
    assert gates == ["official-ready"]
    assert http.calls == []


def test_missing_frozen_calibration_item_writes_resumable_structural_terminal(
    monkeypatch, tmp_path: Path
):
    from experiments.llm_sim import runner as runner_module
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    original_lookup = runner_module._catalog_item_by_id

    def missing_second(catalog, item_id):
        return None if item_id == "i2" else original_lookup(catalog, item_id)

    monkeypatch.setattr(runner_module, "_catalog_item_by_id", missing_second)
    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=FakeTransport()
    )
    first = LLMSimulationRunner(**base).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    relative = store.calibration_relative_path("deepseek", "p:weak")
    path = store.root / relative
    before = path.read_bytes()
    calibration = store.read_json(relative)
    assert calibration["status"] == "structural_failure"
    assert calibration["failure_category"] == "catalog_item_missing"
    assert calibration["terminal_reason"] == "frozen_calibration_item_unavailable"
    assert calibration["expected_item_count"] == 4
    assert calibration["actual_administered_count"] == 1
    assert calibration["missing_item_id"] == "i2"
    assert first["manifest"]["reportability"]["reportable"] is False

    transport = FakeTransport()
    resumed = LLMSimulationRunner(
        **{**base, "transport": transport}
    ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert transport.calls == []
    assert path.read_bytes() == before
    for field in (
        "requests",
        "responses",
        "retries",
        "input_tokens",
        "output_tokens",
        "cost_yuan",
        "completed_by_arm",
    ):
        assert resumed["accounting"][field] == first["accounting"][field]


def test_repeated_model_drift_exclusions_are_append_only_and_fully_accounted(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner, ModelDriftError
    from experiments.llm_sim.store import SimulationStore

    class Drift(FakeTransport):
        def complete(self, **kwargs):
            result = super().complete(**kwargs)
            result["model_returned"] = "unexpected-model"
            return result

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(tmp_path, store=store, transport=Drift())
    for _ in range(2):
        with pytest.raises(ModelDriftError):
            LLMSimulationRunner(**base).run_provider(
                "deepseek", model="model-a", max_items=1, arms=("A",)
            )
    manifest = store.read_json("providers/deepseek.json")
    attempts = [
        row
        for row in manifest["artifacts"]
        if row["record_type"] == "llm_sim_provider_attempt"
    ]
    exclusions = [
        row
        for row in manifest["artifacts"]
        if row["record_type"] == "llm_sim_excluded_response_accounting"
    ]
    assert len(attempts) == 2
    assert len(exclusions) == 2
    assert len({row["path"] for row in exclusions}) == 2
    assert manifest["accounting"]["requests"] == 2
    assert manifest["accounting"]["responses"] == 2
    assert manifest["accounting"]["cost_yuan"] == 0.02
    records = [store.read_json(row["path"]) for row in exclusions]
    assert {row["schema_version"] for row in records} == {
        "yher.llm_sim.model_drift_exclusion.v1"
    }
    assert {row["exclusion_type"] for row in records} == {"model_id_drift"}
    assert {row["source_attempt_number"] for row in records} == {1, 2}


def test_belief_vector_requires_probability_mass_one():
    from experiments.llm_sim.runner import _belief_vector

    assert _belief_vector([0.25, 0.25, 0.25, 0.25])
    assert not _belief_vector([2.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("tamper", ("mass", "transition"))
def test_rehashed_journey_belief_tamper_fails_before_transport(
    tmp_path: Path, tamper: str
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=FakeTransport()
    )
    LLMSimulationRunner(**base).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    relative = store.journey_relative_path("deepseek", "p:weak", "A")
    path = store.root / relative
    journey = store.read_json(relative)
    if tamper == "mass":
        journey["events"][0]["posterior_belief"] = [2.0, 0.0, 0.0, 0.0]
        journey["final_belief"] = [2.0, 0.0, 0.0, 0.0]
    else:
        journey["events"][0]["prior_belief"] = [0.4, 0.2, 0.2, 0.2]
    path.write_text(
        json.dumps(journey, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = store.root / "providers" / "deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["artifacts"]:
        if row["path"] == relative.as_posix():
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    resolved = []
    with pytest.raises(ValueError, match="provider manifest artifact.*journey"):
        LLMSimulationRunner(
            **{
                **base,
                "transport": lambda provider: resolved.append(provider)
                or FakeTransport(),
            }
        ).run_provider("deepseek", model="model-a", max_items=1, arms=("A",))
    assert resolved == []


def test_frozen_annotation_snapshot_rehydrates_new_runner_without_source_map(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    annotation_map = {
        "items": {
            "i1": {"failure-1": "B"},
            "i2": {"failure-1": "C"},
            "i3": {"failure-1": "B"},
            "i4": {"failure-1": "C"},
        }
    }
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps(annotation_map), encoding="utf-8")
    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=FakeTransport()
    )
    LLMSimulationRunner(
        **base,
        annotation_map=annotation_map,
        annotation_map_source=source,
    ).prepare()
    snapshot_path = store.root / "annotation_map_snapshot.json"
    before = snapshot_path.read_bytes()

    result = LLMSimulationRunner(**base).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    assert result["calibration_attempts"]
    assert snapshot_path.read_bytes() == before


@pytest.mark.parametrize(
    ("max_items", "arms", "message"),
    ((1, ("A", "B"), "max_items"), (15, ("A",), "arms")),
)
def test_http_transport_requires_canonical_live_budget_and_both_arms(
    monkeypatch, tmp_path: Path, max_items: int, arms: tuple[str, ...], message: str
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore
    from experiments.llm_sim.transport import HTTPProviderTransport, provider_spec

    class SafeHTTP(HTTPProviderTransport):
        def __init__(self):
            super().__init__(provider_spec("deepseek"), "not-a-real-key")
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("canonical design must fail before HTTP")

    store = SimulationStore(tmp_path / "sim_store")
    runner = LLMSimulationRunner(
        **_one_persona_runner_kwargs(
            tmp_path, store=store, transport=FakeTransport()
        )
    )
    runner.prepare()
    http = SafeHTTP()
    runner.transport = http
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(runner, "_assert_official_live_ready", lambda: None)
    with pytest.raises(ValueError, match=message):
        runner.run_provider(
            "deepseek", model="model-a", max_items=max_items, arms=arms
        )
    assert http.calls == []


def test_live_cli_rejects_max_items_override_before_creating_store(
    monkeypatch, tmp_path: Path
):
    from experiments.llm_sim.cli import main

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    output = tmp_path / "sim_store"
    with pytest.raises(SystemExit):
        main(
            [
                "--live",
                "--provider",
                "deepseek",
                "--max-items",
                "1",
                "--output-root",
                str(output),
            ]
        )
    assert not output.exists()


def test_run_started_timestamp_is_bound_to_manifest_and_all_resume_attempts(
    tmp_path: Path,
):
    from experiments.llm_sim.runner import LLMSimulationRunner
    from experiments.llm_sim.store import SimulationStore

    store = SimulationStore(tmp_path / "sim_store")
    base = _one_persona_runner_kwargs(
        tmp_path, store=store, transport=FakeTransport()
    )
    first = LLMSimulationRunner(**base).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    started = first["manifest"]["run_started_at_utc"]
    resumed = LLMSimulationRunner(**base).run_provider(
        "deepseek", model="model-a", max_items=1, arms=("A",)
    )
    assert resumed["manifest"]["run_started_at_utc"] == started
    attempts = [
        store.read_json(row["path"])
        for row in resumed["manifest"]["artifacts"]
        if row["record_type"] == "llm_sim_provider_attempt"
    ]
    assert attempts
    assert {row["run_started_at_utc"] for row in attempts} == {started}


def test_http_transport_factory_has_no_unused_model_parameter():
    import inspect

    from experiments.llm_sim.transport import HTTPProviderTransport

    assert "model" not in inspect.signature(
        HTTPProviderTransport.from_environment
    ).parameters
