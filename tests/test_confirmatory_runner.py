"""Focused contracts for the frozen S1A confirmatory runner."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest


def test_frozen_config_exposes_exact_confirmatory_grid() -> None:
    try:
        from experiments.confirmatory.config import load_frozen_config
    except ModuleNotFoundError as exc:
        pytest.fail(f"confirmatory package is not implemented: {exc}")

    config = load_frozen_config()

    assert config.truth_states == ("M", "P", "C", "U")
    assert config.arms == ("A", "B", "C")
    assert config.conditions == ("matched", "misspecified")
    assert config.replicates == 50
    assert config.budgets == (9, 15, 25)
    assert config.max_items == 25
    assert config.stop_budget_items == 26
    assert config.expected_journeys(open_node_count=27) == 32_400


def _test_config():
    from experiments.confirmatory.config import ConfirmatoryConfig, load_frozen_config

    raw = copy.deepcopy(dict(load_frozen_config().raw))
    raw["replicates"] = 1
    return ConfirmatoryConfig.from_mapping(raw)


def _fake_pools():
    from experiments.confirmatory.models import EmpiricalItem, TargetPools

    local = tuple(
        EmpiricalItem(
            item_id=f"local-{index:02d}",
            family_id=f"local-family-{index:02d}",
            node_id="Target",
            difficulty=(0.25, 0.5, 0.75, 1.0)[index % 4],
            item_type="mcq" if index % 3 else "numeric",
            role="local",
        )
        for index in range(30)
    )
    prerequisite = tuple(
        EmpiricalItem(
            item_id=f"prereq-{index:02d}",
            family_id=f"prereq-family-{index:02d}",
            node_id="Prerequisite",
            difficulty=(0.25, 0.5, 0.75, 1.0)[index % 4],
            item_type="mcq",
            role="prereq",
        )
        for index in range(12)
    )
    held_out = tuple(
        EmpiricalItem(
            item_id=f"held-{index}",
            family_id=f"held-family-{index}",
            node_id="Target",
            difficulty=0.5,
            item_type="mcq",
            role="held_out",
        )
        for index in range(2)
    )
    return TargetPools(
        target_node="Target",
        local_items=local,
        prerequisite_items=prerequisite,
        held_out_items=held_out,
        held_out_family_ids=frozenset(item.family_id for item in held_out),
        h1_h2_eligible=True,
        common_support_no_repeat={9: True, 15: True, 25: True},
        common_support_set_sha256={
            9: "9" * 64,
            15: "a" * 64,
            25: "b" * 64,
        },
    )


def _fake_context(*, input_digest: str = "f"):
    from experiments.confirmatory.models import CatalogContext

    return CatalogContext(
        targets={"Target": _fake_pools()},
        h1_h2_eligible_targets=("Target",),
        h1_h2_excluded_targets=(),
        input_sha256={
            "fake_catalog": {"path": "fake", "sha256": input_digest * 64}
        },
    )


def _spec(*, truth: str = "P", condition: str = "matched", replicate: int = 0):
    from experiments.confirmatory.models import UnitSpec

    return UnitSpec(
        target_node="Target",
        truth=truth,
        condition=condition,
        replicate=replicate,
    )


def test_sha256_seed_derivation_uses_exact_first_128_bits() -> None:
    from experiments.confirmatory.randomness import replicate_seed_material, seed128

    material = replicate_seed_material(
        master_seed=20260713,
        target="Target",
        truth="P",
        condition="matched",
        replicate=0,
    )
    assert material == "yher-confirmatory-v1|20260713|Target|P|matched|0"
    expected = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:16], "big")
    assert seed128(material) == expected


def test_shared_response_noise_consumes_exact_frozen_replicate_seed() -> None:
    from experiments.confirmatory.randomness import SharedResponseStreams, seed128

    material = "yher-confirmatory-v1|20260713|Target|P|matched|0"
    streams = SharedResponseStreams.build(material, 3, _test_config())

    expected_rng = random.Random(seed128(material))
    assert streams.response_noise == tuple(expected_rng.random() for _ in range(3))


@pytest.mark.parametrize(
    "field_path",
    (
        ("analysis_plan_commit",),
        ("arms",),
        ("bootstrap_seed",),
        ("budgets",),
        ("c_probe_interval",),
        ("census_seed",),
        ("census_analysis_plan_commit",),
        ("conditions",),
        ("config_frozen_at_utc",),
        ("fixed_difficulty_ladder",),
        ("gap_threshold",),
        ("held_out_families",),
        ("master_seed",),
        ("max_items",),
        ("minimum_direct_answers",),
        ("minimum_prerequisite_families",),
        ("misspecified", "ability_offset_clip"),
        ("misspecified", "ability_offset_sd"),
        ("misspecified", "guess_range"),
        ("misspecified", "probability_clip"),
        ("misspecified", "slip_range"),
        ("provider",),
        ("replicates",),
        ("run_id",),
        ("schema_version",),
        ("seed_derivation_version",),
        ("stop_budget_items",),
        ("truth_states",),
    ),
)
def test_every_scientific_config_field_is_frozen(field_path: tuple[str, ...]) -> None:
    from experiments.confirmatory.config import ConfirmatoryConfig, load_frozen_config
    from experiments.confirmatory.runner import validate_definition

    raw = copy.deepcopy(dict(load_frozen_config().raw))
    parent = raw
    for field in field_path[:-1]:
        parent = parent[field]
    field = field_path[-1]
    current = parent[field]
    if isinstance(current, list):
        parent[field] = list(reversed(current))
    elif isinstance(current, int):
        parent[field] = current + 1
    elif isinstance(current, float):
        parent[field] = current + 0.01
    else:
        parent[field] = f"{current}-mutated"

    try:
        changed = ConfirmatoryConfig.from_mapping(raw)
    except ValueError:
        return
    with pytest.raises(ValueError, match="frozen confirmatory config"):
        validate_definition(changed, _fake_context())


def test_family_epoch_is_seeded_stable_and_exhausts_families_before_repeat() -> None:
    from experiments.confirmatory.allocation import FamilyEpoch

    pools = _fake_pools()
    first = FamilyEpoch(pools.local_items[:6], seed_material="stable")
    second = FamilyEpoch(pools.local_items[:6], seed_material="stable")

    first_epoch = [first.take_first().family_id for _ in range(6)]
    second_epoch = [second.take_first().family_id for _ in range(6)]
    repeated = first.take_first().family_id

    assert first_epoch == second_epoch
    assert len(set(first_epoch)) == 6
    assert repeated in set(first_epoch)


def test_fixed_allocator_never_uses_seed_to_override_exact_tie_break() -> None:
    from experiments.confirmatory.allocation import FixedLadderAllocator
    from experiments.confirmatory.models import EmpiricalItem

    items = (
        EmpiricalItem("z-item", "a-family", "Target", 0.5, "mcq", "local"),
        EmpiricalItem("a-item", "z-family", "Target", 0.5, "mcq", "local"),
        EmpiricalItem("a-item", "a-family", "Target", 0.75, "mcq", "local"),
    )
    allocator = FixedLadderAllocator(items)

    assert allocator.take(0.5).item_id == "z-item"
    assert allocator.take(0.5).item_id == "a-item"


def test_real_production_calls_and_arm_role_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.confirmatory import simulation
    from experiments.s0_census import require_simulated_event_envelope

    originals = {
        "observe": simulation.mastery.observe,
        "select_next": simulation.selector.select_next,
        "should_stop": simulation.selector.should_stop,
    }
    dynamic_calls = {name: 0 for name in originals}

    def wrap(name):
        def wrapped(*args, **kwargs):
            dynamic_calls[name] += 1
            return originals[name](*args, **kwargs)

        return wrapped

    for name in originals:
        monkeypatch.setattr(getattr(simulation, "mastery" if name == "observe" else "selector"), name, wrap(name))

    paired = simulation.run_paired_unit(_fake_pools(), _spec(), _test_config())
    journeys = {journey["arm"]: journey for journey in paired}
    total_administered = sum(row["actual_administered_count"] for row in paired)

    assert dynamic_calls["observe"] == total_administered
    expected_stop_calls = total_administered + sum(
        row["actual_administered_count"] == 25 for row in paired
    )
    assert dynamic_calls["should_stop"] == expected_stop_calls
    assert dynamic_calls["select_next"] == journeys["A"]["actual_administered_count"]
    assert journeys["A"]["call_counters"]["selector_select_next"] == journeys["A"]["actual_administered_count"]
    assert journeys["B"]["call_counters"]["selector_select_next"] == 0
    assert journeys["C"]["call_counters"]["selector_select_next"] == 0
    assert all(event["role"] == "local" for event in journeys["B"]["events"])
    assert all(
        event["role"] == ("prereq" if event["position"] % 3 == 0 else "local")
        for event in journeys["C"]["events"]
    )
    common_positions = min(row["actual_administered_count"] for row in paired)
    for position in range(common_positions):
        assert len({row["events"][position]["response_noise"] for row in paired}) == 1
    for journey in paired:
        require_simulated_event_envelope(journey)
        for event in journey["events"]:
            require_simulated_event_envelope(event)

    source = inspect.getsource(simulation)
    assert "mastery.observe(" in source
    assert "selector.select_next(" in source
    assert "selector.should_stop(" in source
    assert "def bayes_update" not in source
    assert "def select_next" not in source


def test_posteriors_are_production_bayes_and_misspecification_never_leaks() -> None:
    from engine import mastery
    from experiments.confirmatory.simulation import run_paired_unit

    journeys = run_paired_unit(
        _fake_pools(),
        _spec(truth="U", condition="misspecified"),
        _test_config(),
    )
    saw_generator_difference = False
    parameters_by_position: dict[int, set[tuple[float, float, float]]] = {}
    for journey in journeys:
        for event in journey["events"]:
            prior = np.asarray(event["prior_belief"], dtype=float)
            likelihood = np.asarray(event["production_inference_likelihood"], dtype=float)
            posterior = np.asarray(event["posterior_belief"], dtype=float)
            assert np.allclose(posterior, mastery.bayes_update(prior, likelihood))
            assert np.all(np.isfinite(posterior))
            assert np.all(posterior >= 0)
            assert np.isclose(posterior.sum(), 1.0)

            if event["role"] == "prereq":
                correct_probs = mastery.prereq_correct_probs(item_type=event["item_type"])
            else:
                correct_probs = mastery.local_correct_probs(
                    event["difficulty"], event["item_type"]
                )
            expected = (
                mastery.likelihood_correct(correct_probs)
                if event["correct"]
                else mastery.likelihood_wrong_binary(correct_probs)
            )
            assert np.allclose(likelihood, expected)
            saw_generator_difference |= not np.isclose(
                event["generator_probability"], correct_probs[mastery.U]
            )
            parameters = event["generator_parameters"]
            parameters_by_position.setdefault(event["position"], set()).add(
                (
                    parameters["slip"],
                    parameters["guess"],
                    parameters["ability_offset"],
                )
            )
    assert saw_generator_difference
    assert all(len(values) == 1 for values in parameters_by_position.values())


def test_prerequisite_observation_does_not_increment_direct_count() -> None:
    from experiments.confirmatory.simulation import run_paired_unit

    journey = next(
        row
        for row in run_paired_unit(_fake_pools(), _spec(), _test_config())
        if row["arm"] == "C"
    )
    for event in journey["events"]:
        expected_delta = 0 if event["role"] == "prereq" else 1
        assert event["direct_answers_after"] - event["direct_answers_before"] == expected_delta


def test_production_confidence_gate_and_item_25_reason_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine import mastery
    from experiments.confirmatory import simulation

    exact_gap = mastery.NodeBelief(np.array([0.70, 0.25, 0.03, 0.02]), direct_answers=3)
    insufficient = mastery.NodeBelief(np.array([0.80, 0.10, 0.05, 0.05]), direct_answers=2)
    assert not simulation.production_confidence_stop(exact_gap, "Target", asked=3)
    assert not simulation.production_confidence_stop(insufficient, "Target", asked=3)

    budgets_seen: list[int] = []

    def never_confident(*args, **kwargs):
        budgets_seen.append(kwargs["budget_items"])
        return False

    monkeypatch.setattr(simulation.selector, "should_stop", never_confident)
    journey = simulation.run_journey(_fake_pools(), _spec(truth="C"), _test_config(), "B", {})
    assert journey["actual_administered_count"] == 25
    assert journey["terminal_reason"] == "budget_exhausted"
    assert journey["converged"] is False
    assert budgets_seen == [25] * 25 + [26]

    monkeypatch.setattr(
        simulation.selector,
        "should_stop",
        lambda *args, **kwargs: kwargs["asked"] == 25,
    )
    journey = simulation.run_journey(_fake_pools(), _spec(truth="C"), _test_config(), "B", {})
    assert journey["terminal_reason"] == "confidence"
    assert journey["convergence_time"] == 25


def test_early_stop_carries_forward_without_inflating_actual_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.confirmatory import simulation

    monkeypatch.setattr(
        simulation.selector,
        "should_stop",
        lambda *args, **kwargs: kwargs["asked"] == 4,
    )
    journey = simulation.run_journey(_fake_pools(), _spec(), _test_config(), "B", {})

    assert journey["actual_administered_count"] == 4
    assert journey["convergence_time"] == 4
    assert [view["actual_administered_count"] for view in journey["views"]] == [4, 4, 4]
    assert all(view["carried_forward"] for view in journey["views"])
    assert all(view["belief"] == journey["final_belief"] for view in journey["views"])


def test_held_out_outcomes_are_paired_never_administered_and_do_not_update() -> None:
    from experiments.confirmatory.simulation import run_paired_unit, score_held_out

    pools = _fake_pools()
    journeys = run_paired_unit(pools, _spec(truth="M"), _test_config())
    held_ids = {item.item_id for item in pools.held_out_items}
    outcomes = []
    for journey in journeys:
        assert held_ids.isdisjoint(event["item_id"] for event in journey["events"])
        outcomes.append(journey["views"][0]["held_out_outcomes"])
        assert len(journey["views"][0]["held_out_family_scores"]) == 2
    assert outcomes[0] == outcomes[1] == outcomes[2]

    posterior = np.array([0.4, 0.3, 0.2, 0.1])
    before = posterior.copy()
    score_held_out(posterior, pools.held_out_items, {"held-0": True, "held-1": False})
    assert np.array_equal(posterior, before)


def test_matched_brier_is_explicitly_internal_calibration_only() -> None:
    from experiments.confirmatory.simulation import run_journey

    journey = run_journey(
        _fake_pools(),
        _spec(condition="matched"),
        _test_config(),
        "B",
        {},
    )

    assert {
        view["held_out_brier_interpretation"] for view in journey["views"]
    } == {"internal_calibration_only"}


def test_repeat_counters_match_recorded_items_and_families() -> None:
    from experiments.confirmatory.simulation import repetition_metrics

    metrics = repetition_metrics(
        item_ids=("a", "b", "a", "c"),
        family_ids=("f1", "f1", "f1", "f2"),
    )
    assert metrics == {
        "actual_administered_count": 4,
        "unique_item_count": 3,
        "unique_family_count": 2,
        "exact_item_repeat_count": 1,
        "family_repeat_count": 2,
        "exact_item_repeat_fraction": 0.25,
        "family_repeat_fraction": 0.5,
    }


def test_storage_is_envelope_validated_atomic_resume_safe_and_byte_stable(
    tmp_path: Path,
) -> None:
    from experiments.confirmatory.storage import write_shards_atomic

    envelope = {
        "simulated": True,
        "persona_id": "confirmatory:Target:P:matched:0:A",
        "provider": "programmatic",
        "model_id": "mastery:abc;selector:def;runner:123",
    }
    records = {
        "Target__P__matched": [
            {**envelope, "record_type": "confirmatory_journey", "replicate": value}
            for value in (1, 0)
        ],
        "Target__U__matched": [
            {
                **envelope,
                "persona_id": "confirmatory:Target:U:matched:0:A",
                "record_type": "confirmatory_journey",
                "replicate": 0,
            }
        ],
    }
    temp_root = tmp_path / "yher_sprint2"
    first = temp_root / "first"
    second = temp_root / "second"
    first_result = write_shards_atomic(
        records,
        output_dir=first,
        config_sha256="c" * 64,
        workers=1,
        resume=False,
        repo_root=tmp_path / "repo",
        temp_root=temp_root,
    )
    second_result = write_shards_atomic(
        dict(reversed(tuple(records.items()))),
        output_dir=second,
        config_sha256="c" * 64,
        workers=3,
        resume=False,
        repo_root=tmp_path / "repo",
        temp_root=temp_root,
    )
    assert first_result["written"] == second_result["written"] == 2
    assert {
        path.name: path.read_bytes() for path in first.glob("*.jsonl")
    } == {
        path.name: path.read_bytes() for path in second.glob("*.jsonl")
    }
    before = {path.name: path.read_bytes() for path in first.glob("*.jsonl")}
    resumed = write_shards_atomic(
        records,
        output_dir=first,
        config_sha256="c" * 64,
        workers=2,
        resume=True,
        repo_root=tmp_path / "repo",
        temp_root=temp_root,
    )
    assert resumed == {"written": 0, "skipped": 2}
    assert before == {path.name: path.read_bytes() for path in first.glob("*.jsonl")}

    invalid = copy.deepcopy(records)
    invalid["Target__P__matched"][0]["simulated"] = False
    with pytest.raises(ValueError, match="simulated event envelope"):
        write_shards_atomic(
            invalid,
            output_dir=temp_root / "invalid",
            config_sha256="c" * 64,
            workers=1,
            resume=False,
            repo_root=tmp_path / "repo",
            temp_root=temp_root,
        )

    nested_invalid = copy.deepcopy(records)
    nested_invalid["Target__P__matched"][0]["events"] = [
        {
            **envelope,
            "simulated": False,
            "record_type": "confirmatory_event",
        }
    ]
    with pytest.raises(ValueError, match="simulated event envelope"):
        write_shards_atomic(
            nested_invalid,
            output_dir=temp_root / "nested-invalid",
            config_sha256="c" * 64,
            workers=1,
            resume=False,
            repo_root=tmp_path / "repo",
            temp_root=temp_root,
        )


def test_real_catalog_validation_is_exact_and_validate_only_writes_no_outcomes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.confirmatory.catalog import load_catalog_context
    from experiments.confirmatory.cli import main

    config = _test_config()
    context = load_catalog_context(config)
    assert len(context.targets) == 27
    assert len(context.h1_h2_eligible_targets) == 23
    assert len(context.h1_h2_excluded_targets) == 4
    for pools in context.targets.values():
        administered_families = {
            item.family_id for item in (*pools.local_items, *pools.prerequisite_items)
        }
        assert pools.held_out_family_ids.isdisjoint(administered_families)
        assert set(pools.common_support_no_repeat) == {9, 15, 25}

    output_root = tmp_path / "yher_sprint2" / "validate-only"
    assert main(["--validate-only", "--output-root", str(output_root)]) == 0
    assert not output_root.exists()
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["open_nodes"] == 27
    assert rendered["h1_h2_eligible"] == 23
    assert rendered["h1_h2_excluded"] == 4
    assert rendered["truth_states"] == 4
    assert rendered["arms"] == 3
    assert rendered["conditions"] == 2
    assert rendered["replicates"] == 50
    assert rendered["expected_journeys"] == 32_400


def test_bounded_fake_execution_is_worker_and_resume_byte_identical(
    tmp_path: Path,
) -> None:
    from experiments.confirmatory.config import load_frozen_config
    from experiments.confirmatory.models import CatalogContext
    from experiments.confirmatory.runner import execute
    from experiments.confirmatory.storage import read_shard_records
    from experiments.s0_census import require_simulated_event_envelope

    config = load_frozen_config()
    pools = _fake_pools()
    context = CatalogContext(
        targets={"Target": pools},
        h1_h2_eligible_targets=("Target",),
        h1_h2_excluded_targets=(),
        input_sha256={"fake_catalog": {"path": "fake", "sha256": "f" * 64}},
    )
    repo = tmp_path / "repo"
    temp_root = tmp_path / "yher_sprint2"
    common = {
        "config": config,
        "run_id": "fake-smoke",
        "resume": False,
        "limit_shards": 2,
        "runner_commit": "a" * 40,
        "experiment_tag": "test-only-no-tag-created",
        "run_started_at_utc": "2026-07-13T13:20:00Z",
        "context": context,
        "repo_root": repo,
        "temp_root": temp_root,
        "verify_repository_binding": False,
    }
    first = execute(output_root=temp_root / "first", workers=1, **common)
    second = execute(output_root=temp_root / "second", workers=3, **common)
    first_dir = Path(first["manifest"]).parent
    second_dir = Path(second["manifest"]).parent

    def artifacts(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.iterdir())
            if path.is_file()
        }

    first_bytes = artifacts(first_dir)
    assert first_bytes == artifacts(second_dir)
    resumed = execute(
        output_root=temp_root / "first",
        workers=2,
        **{**common, "resume": True},
    )
    assert resumed["written"] == 0
    assert resumed["skipped"] == 2
    assert artifacts(first_dir) == first_bytes
    assert resumed["protected_filesystem_assertion"]["unchanged"] is True

    shard = next(first_dir.glob("shard-*.jsonl"))
    journey = read_shard_records(shard)[0]
    require_simulated_event_envelope(journey)
    assert journey["provenance"]["runner_commit"] == "a" * 40
    assert journey["provenance"]["experiment_tag"] == "test-only-no-tag-created"
    assert journey["provenance"]["input_sha256"] == context.input_sha256
    assert all(event["provenance"] == journey["provenance"] for event in journey["events"])
    manifest = json.loads(Path(resumed["manifest"]).read_text(encoding="utf-8"))
    assert manifest["run_started_at_utc"] == "2026-07-13T13:20:00Z"
    assert manifest["config_frozen_at_utc"] == config.raw["config_frozen_at_utc"]
    assert "created_at_utc" not in manifest


@pytest.mark.parametrize("changed_field", ("runner_commit", "experiment_tag", "input"))
def test_resume_rewrites_shards_when_frozen_provenance_changes(
    tmp_path: Path,
    changed_field: str,
) -> None:
    from experiments.confirmatory.config import load_frozen_config
    from experiments.confirmatory.models import CatalogContext
    from experiments.confirmatory.runner import execute
    from experiments.confirmatory.storage import read_shard_records

    config = load_frozen_config()
    context = CatalogContext(
        targets={"Target": _fake_pools()},
        h1_h2_eligible_targets=("Target",),
        h1_h2_excluded_targets=(),
        input_sha256={"fake_catalog": {"path": "fake", "sha256": "f" * 64}},
    )
    repo = tmp_path / "repo"
    temp_root = tmp_path / "yher_sprint2"
    output_root = temp_root / "resume-provenance"
    common = {
        "config": config,
        "output_root": output_root,
        "run_id": "fake-smoke",
        "workers": 1,
        "limit_shards": 1,
        "experiment_tag": "test-only-no-tag-created",
        "run_started_at_utc": "2026-07-13T13:20:00Z",
        "context": context,
        "repo_root": repo,
        "temp_root": temp_root,
        "verify_repository_binding": False,
    }

    execute(runner_commit="a" * 40, resume=False, **common)
    changed = dict(common)
    changed["runner_commit"] = "a" * 40
    if changed_field == "runner_commit":
        changed["runner_commit"] = "b" * 40
    elif changed_field == "experiment_tag":
        changed["experiment_tag"] = "test-only-second-tag"
    else:
        changed["context"] = _fake_context(input_digest="e")
    resumed = execute(resume=True, **changed)

    assert resumed["written"] == 1
    assert resumed["skipped"] == 0
    shard = next(Path(resumed["manifest"]).parent.glob("shard-*.jsonl"))
    records = read_shard_records(shard)
    assert {row["provenance"]["runner_commit"] for row in records} == {
        changed["runner_commit"]
    }
    assert {row["provenance"]["experiment_tag"] for row in records} == {
        changed["experiment_tag"]
    }
    assert {json.dumps(row["provenance"]["input_sha256"], sort_keys=True) for row in records} == {
        json.dumps(changed["context"].input_sha256, sort_keys=True)
    }


def _fake_execute_args(tmp_path: Path) -> dict[str, object]:
    temp_root = tmp_path / "yher_sprint2"
    return {
        "config": __import__(
            "experiments.confirmatory.config", fromlist=["load_frozen_config"]
        ).load_frozen_config(),
        "output_root": temp_root / "confirmatory",
        "run_id": "fake-smoke",
        "workers": 1,
        "resume": False,
        "limit_shards": 1,
        "runner_commit": "a" * 40,
        "experiment_tag": "test-only-no-tag-created",
        "run_started_at_utc": "2026-07-13T13:20:00Z",
        "context": _fake_context(),
        "repo_root": tmp_path / "repo",
        "temp_root": temp_root,
        "verify_repository_binding": False,
    }


def test_failed_isolation_guard_leaves_no_resumable_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.confirmatory import runner
    from experiments.s0_census import ProtectedWriteError

    @contextmanager
    def failing_guard(*args, **kwargs):
        yield {
            "before": {"digest": "a" * 64, "coverage": ["fake"]},
            "after": {"digest": "b" * 64, "coverage": ["fake"]},
            "unchanged": False,
        }
        raise ProtectedWriteError("injected protected write")

    monkeypatch.setattr(runner, "guarded_simulation_run", failing_guard)
    args = _fake_execute_args(tmp_path)
    output = Path(args["output_root"]) / str(args["run_id"])

    with pytest.raises(ProtectedWriteError, match="injected protected write"):
        runner.execute(**args)

    assert not list(output.glob("shard-*.jsonl"))
    assert not (output / "manifest.json").exists()


def test_resume_rebuilds_shard_without_successful_isolation_attestation(
    tmp_path: Path,
) -> None:
    from experiments.confirmatory.runner import execute

    args = _fake_execute_args(tmp_path)
    first = execute(**args)
    shard = next(Path(first["manifest"]).parent.glob("shard-*.jsonl"))
    lines = shard.read_bytes().splitlines(keepends=True)
    shard_manifest = json.loads(lines[0])
    assert shard_manifest["protected_filesystem_assertion"]["unchanged"] is True
    shard_manifest.pop("protected_filesystem_assertion")
    shard.write_bytes(
        json.dumps(
            shard_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        + b"".join(lines[1:])
    )

    resumed = execute(**{**args, "resume": True})

    assert resumed["written"] == 1
    assert resumed["skipped"] == 0


@pytest.mark.parametrize("run_id", (".", ".."))
def test_run_id_rejects_path_segments(tmp_path: Path, run_id: str) -> None:
    from experiments.confirmatory.runner import execute

    with pytest.raises(ValueError, match="run_id"):
        execute(**{**_fake_execute_args(tmp_path), "run_id": run_id})


def test_repository_outputs_are_confined_to_confirmatory_root(tmp_path: Path) -> None:
    from experiments.confirmatory.runner import execute

    args = _fake_execute_args(tmp_path)
    bad_root = Path(args["repo_root"]) / "data" / "sim_store" / "census"
    with pytest.raises(ValueError, match="confirmatory"):
        execute(**{**args, "output_root": bad_root})


def test_data_execution_requires_bytecode_disabled_from_process_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.confirmatory.runner import execute

    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    with pytest.raises(RuntimeError, match="python -B"):
        execute(**_fake_execute_args(tmp_path))


def test_run_start_cannot_predate_config_freeze(tmp_path: Path) -> None:
    from experiments.confirmatory.runner import execute

    with pytest.raises(ValueError, match="predate config freeze"):
        execute(
            **{
                **_fake_execute_args(tmp_path),
                "run_started_at_utc": "2026-07-13T13:00:00Z",
            }
        )


def test_repository_binding_requires_annotated_tag_head_and_clean_frozen_paths(
    tmp_path: Path,
) -> None:
    from experiments.confirmatory.runner import verify_repository_binding
    from experiments.confirmatory.provenance import FROZEN_REPOSITORY_PATHS

    assert "core/data/item_bank_v4.py" in FROZEN_REPOSITORY_PATHS
    assert "core/learning/scoring.py" in FROZEN_REPOSITORY_PATHS

    repo = tmp_path / "binding-repo"
    repo.mkdir()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Confirmatory Test")
    frozen = repo / "frozen.txt"
    frozen.write_text("frozen\n", encoding="utf-8")
    dependency = repo / "core" / "data" / "item_bank_v4.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("frozen dependency\n", encoding="utf-8")
    git("add", "frozen.txt", "core/data/item_bank_v4.py")
    git("commit", "-qm", "freeze")
    commit = git("rev-parse", "HEAD")
    git("tag", "-a", "experiment-freeze-test", "-m", "freeze test")

    evidence = verify_repository_binding(
        repo,
        runner_commit=commit,
        experiment_tag="experiment-freeze-test",
        analysis_plan_commit=commit,
        frozen_paths=("frozen.txt", "core/data/item_bank_v4.py"),
    )
    assert evidence["head"] == evidence["tag_commit"] == commit
    assert evidence["tag_type"] == "tag"

    dependency.write_text("dirty dependency\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen worktree"):
        verify_repository_binding(
            repo,
            runner_commit=commit,
            experiment_tag="experiment-freeze-test",
            analysis_plan_commit=commit,
            frozen_paths=("frozen.txt", "core/data/item_bank_v4.py"),
        )
