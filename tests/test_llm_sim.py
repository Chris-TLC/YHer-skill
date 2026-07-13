"""Contract tests for the S2 simulated-persona facade.

These tests deliberately use an in-memory catalog and a fake transport.  A live
provider is never contacted from the offline suite.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    transport.calls.clear()
    second = LLMSimulationRunner(**kwargs).run_provider("deepseek", model="model-a", max_items=1, resume=True)
    assert second["accounting"]["requests"] == 0
    assert second["accounting"]["skipped"] == 2
    assert second["accounting"]["input_tokens"] == 60
    assert second["accounting"]["output_tokens"] == 24
    assert second["accounting"]["cost_yuan"] == 0.06
    assert transport.calls == []


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
        assert len(manifest["official_input_sha256"]) == 64
        assert manifest["config_sha256"] == runner.config.sha256
        assert manifest["study_seed"] == runner.study_seed
    assert preparation["code_files"]
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
        "code_files": [{"path": "runner.py", "sha256": "3" * 64}],
    }
    monkeypatch.setattr(
        runner_module,
        "collect_code_provenance",
        lambda _root: json.loads(json.dumps(current)),
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


def test_frozen_llm_config_exposes_exact_s2_grid():
    from experiments.llm_sim.config import load_frozen_config

    config = load_frozen_config()
    assert config.run_id == "llm-personas-v1"
    assert config.pair_count == 25
    assert config.persona_count == 50
    assert config.arms == ("A", "B")
    assert config.providers == ("deepseek", "glm", "kimi", "minimax", "doubao", "tongyi")
    assert config.minimum_complete_per_cell == 45
    assert config.maximum_prompt_rewrites == 1
    assert config.manipulation_mapping_policy == "explicit_machine_annotation_only"


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

    runner = LLMSimulationRunner(
        catalog=FakeCatalog(),
        kg=FakeKG(),
        store=SimulationStore(tmp_path / "sim_store"),
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
