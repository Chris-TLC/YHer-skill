"""Contracts for the frozen Persona-v2 main-analysis engine."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


RUN_ID = "llm-personas-v2-dual"
PROVIDERS = ("deepseek", "glm", "kimi", "minimax", "doubao", "tongyi")
PERSONAS = tuple(f"persona-{index:02d}" for index in range(50))


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _carried_cost_fixture() -> dict[str, object]:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json"
        ).read_text(encoding="utf-8")
    )


def _expected_tasks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for persona_index, persona_id in enumerate(PERSONAS):
        anchor_id = f"anchor-{persona_index // 2:02d}"
        for response_arm in ("deficit", "control"):
            for ordinal in range(4):
                rows.append(
                    {
                        "task_id": f"c:{persona_id}:{response_arm}:{ordinal}",
                        "persona_id": persona_id,
                        "pair_id": f"pair-{persona_index:02d}",
                        "row_id": f"{persona_id}:{response_arm}",
                        "anchor_id": anchor_id,
                        "response_arm": response_arm,
                        "condition": "controlled",
                        "item_id": f"controlled-{anchor_id}-{ordinal}",
                        "correct_option": "A",
                        "is_stability_repeat": False,
                        "is_terminal": False,
                        "target_option": (
                            "B"
                            if ordinal == 0 and int(anchor_id.rsplit("-", 1)[1]) < 6
                            else None
                        ),
                        "random_wrong_option_baseline": (
                            1 / 3
                            if ordinal == 0 and int(anchor_id.rsplit("-", 1)[1]) < 6
                            else None
                        ),
                    }
                )
            for ordinal in range(2):
                rows.append(
                    {
                        "task_id": f"b:{persona_id}:{response_arm}:{ordinal}",
                        "persona_id": persona_id,
                        "pair_id": f"pair-{persona_index:02d}",
                        "row_id": f"{persona_id}:{response_arm}",
                        "anchor_id": anchor_id,
                        "response_arm": response_arm,
                        "condition": "blind",
                        "item_id": f"blind-{anchor_id}-{ordinal}",
                        "correct_option": "A",
                        "is_stability_repeat": False,
                        "is_terminal": ordinal == 1,
                        "target_option": None,
                        "random_wrong_option_baseline": None,
                    }
                )
            if persona_index < 10:
                rows.append(
                    {
                        "task_id": f"r:{persona_id}:{response_arm}",
                        "persona_id": persona_id,
                        "pair_id": f"pair-{persona_index:02d}",
                        "row_id": f"{persona_id}:{response_arm}",
                        "anchor_id": anchor_id,
                        "response_arm": response_arm,
                        "condition": "blind",
                        "item_id": f"blind-{anchor_id}-1",
                        "correct_option": "A",
                        "is_stability_repeat": True,
                        "is_terminal": True,
                        "target_option": None,
                        "random_wrong_option_baseline": None,
                    }
                )
    for row_index, row in enumerate(rows):
        task_id = str(row["task_id"])
        public_question = {
            "kind": "mcq",
            "stem_blocks": [],
            "stem_text": f"Public question {row_index}",
            "options": {
                "A": "public option A",
                "B": "public option B",
                "C": "public option C",
                "D": "public option D",
            },
            "difficulty": 0.5,
            "nodes": [],
            "source_label": "synthetic-test",
        }
        row.update(
            {
                "target_node": str(row["anchor_id"]),
                "family_id": str(row["item_id"]),
                "attempt_id": (
                    "stability" if row["is_stability_repeat"] else "primary"
                ),
                "logical_key": f"logical:{task_id}",
                "message_sha256": _sha({"messages": task_id}),
                "wire_message_sha256": _sha({"wire_messages": task_id}),
                "public_question": public_question,
                "item_contract": {
                    "item_id": row["item_id"],
                    "family_id": row["item_id"],
                    "public_question": public_question,
                    "options": public_question["options"],
                    "private_correct_option": "A",
                },
                "persona_contract": {
                    "persona_id": row["persona_id"],
                    "pair_id": row["pair_id"],
                    "row_id": row["row_id"],
                    "target_node": row["anchor_id"],
                    "deficit_condition": row["response_arm"],
                    "failure_id": "private-failure",
                    "failure_cause": "private-cause",
                    "failure_symptom": "private-symptom",
                    "observable_error_policy": {"private_policy": "hidden"},
                },
            }
        )
    return rows


def _provenance_fixture(
    tasks: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    task_ids = [str(row["task_id"]) for row in tasks]
    runtime_phase = {
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_set_sha256": "1" * 64,
        "providers": list(PROVIDERS),
    }
    runtime: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.runtime_task_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "freeze_commit": "a" * 40,
        "freeze_manifest_sha256": "b" * 64,
        "runtime_commit": "c" * 40,
        "frozen_at_utc": "2026-07-15T08:00:00Z",
        "prompt_revision": 0,
        "prompt_contract_sha256": "d" * 64,
        "prompt_ledger_sha256": "e" * 64,
        "runtime_files": [],
        "runtime_file_set_sha256": "f" * 64,
        "phases": {"main": runtime_phase},
    }
    runtime["runtime_task_manifest_sha256"] = _sha(runtime)
    prior = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/llm_sim_v2/prior_cost_ledger.json"
        ).read_text(encoding="utf-8")
    )
    phase: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.phase_provenance.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "main",
        "analysis_population": "main",
        "collection_mode": "formal",
        "development_only": False,
        "partial": False,
        "formal_analysis_eligible": True,
        "modality_condition": "text_only",
        "selected_providers": list(PROVIDERS),
        "frozen_providers": list(PROVIDERS),
        "task_limit": None,
        "freeze": {"freeze_manifest_sha256": "b" * 64},
        "source": {"source_set_sha256": "2" * 64},
        "target": {
            "target_set_hash": "3" * 64,
            "mapping_sha256": "4" * 64,
        },
        "grid_sha256": "5" * 64,
        "population_manifest_sha256": "6" * 64,
        "official_inputs_sha256": "7" * 64,
        "prompt": {
            "revision": 0,
            "prompt_contract_sha256": "d" * 64,
            "prompt_ledger_sha256": "e" * 64,
        },
        "runtime": {
            "runtime_task_manifest_sha256": runtime[
                "runtime_task_manifest_sha256"
            ],
            "execution_commit": "c" * 40,
            "runtime_file_set_sha256": "f" * 64,
        },
        "budget": {
            "prior_cost_ledger_sha256": prior["prior_cost_ledger_sha256"],
            "prior_known_cost_yuan": prior["known_cost_yuan"],
            "prior_ambiguity_reserve_yuan": prior[
                "pre_run_ambiguity_reserve_yuan"
            ],
            "prior_documented_cost_yuan": prior["pre_run_total_bound_yuan"],
            "carried_forward_cost_ledger_sha256": _carried_cost_fixture()[
                "carried_forward_cost_ledger_sha256"
            ],
            "source_phase_receipt_sha256": _carried_cost_fixture()[
                "source_phase_receipt_sha256"
            ],
            "source_record_set_sha256": _carried_cost_fixture()[
                "source_record_set_sha256"
            ],
            "carried_forward_known_cost_yuan": _carried_cost_fixture()[
                "known_cost_yuan"
            ],
            "carried_forward_unknown_reserve_yuan": _carried_cost_fixture()[
                "unknown_cost_reserve_yuan"
            ],
            "carried_forward_total_accounted_cost_yuan": _carried_cost_fixture()[
                "total_accounted_cost_yuan"
            ],
            "unknown_attempt_reserve_yuan": prior[
                "unknown_attempt_reserve_yuan"
            ],
            "soft_warning_yuan": 300.0,
            "hard_fuse_yuan": 450.0,
        },
        "task_roster": {
            "expected_task_count": len(task_ids),
            "expected_task_ids": task_ids,
            "task_set_sha256": "1" * 64,
            "frozen_task_count": len(task_ids),
            "frozen_task_set_sha256": "1" * 64,
        },
    }
    phase["phase_provenance_sha256"] = _sha(phase)
    return runtime, phase


def _rehash(value: dict[str, object], field: str) -> None:
    value.pop(field, None)
    value[field] = _sha(value)


def _active_contract_proof(
    runtime: dict[str, object], phase: dict[str, object]
) -> dict[str, object]:
    proof: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.active_analysis_contract_proof.v1",
        "ok": True,
        "runtime_git_verified": True,
        "contract_revalidated": True,
        "request_temperature": 0.0,
        "runtime_task_manifest_sha256": runtime[
            "runtime_task_manifest_sha256"
        ],
        "phase_provenance_sha256": phase["phase_provenance_sha256"],
        "source_set_sha256": phase["source"]["source_set_sha256"],  # type: ignore[index]
        "target_set_hash": phase["target"]["target_set_hash"],  # type: ignore[index]
        "carried_forward_cost_ledger_sha256": phase["budget"][  # type: ignore[index]
            "carried_forward_cost_ledger_sha256"
        ],
        "source_record_set_sha256": phase["budget"][  # type: ignore[index]
            "source_record_set_sha256"
        ],
        "provider_models": {
            provider: f"{provider}-model" for provider in PROVIDERS
        },
        "provider_attempt_policies": {
            provider: {
                "max_attempts": 4,
                "allowed_request_max_tokens": [512, 1024],
                "max_tokens": 512,
                "retry_max_tokens": 1024,
                "timeout_seconds": 60.0,
                "concurrency": 4,
                "failure_threshold": 3,
                "base_backoff_seconds": 1.0,
                "max_backoff_seconds": 30.0,
                "cooldown_seconds": 120.0,
                "jitter_fraction": 0.25,
            }
            for provider in PROVIDERS
        },
        "frozen_leakage_lexicon": [],
    }
    proof["active_analysis_contract_proof_sha256"] = _sha(proof)
    return proof


def test_formal_main_provenance_and_runtime_roster_are_fail_closed() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, validate_inputs

    tasks = _expected_tasks()
    runtime, phase = _provenance_fixture(tasks)
    proof = validate_inputs(
        phase_provenance=phase,
        runtime_manifest=runtime,
        expected_tasks=tasks,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )
    assert proof["expected_task_count"] == len(tasks)
    assert proof["persona_cluster_count"] == 50
    assert proof["providers"] == list(PROVIDERS)

    pilot = copy.deepcopy(phase)
    pilot["phase"] = "pilot"
    pilot["analysis_population"] = "pilot"
    _rehash(pilot, "phase_provenance_sha256")
    with pytest.raises(AnalysisContractError, match="formal main"):
        validate_inputs(
            phase_provenance=pilot,
            runtime_manifest=runtime,
            expected_tasks=tasks,
            active_contract_proof=_active_contract_proof(runtime, pilot),
        )

    partial = copy.deepcopy(phase)
    partial["partial"] = True
    partial["formal_analysis_eligible"] = False
    _rehash(partial, "phase_provenance_sha256")
    with pytest.raises(AnalysisContractError, match="partial|formal"):
        validate_inputs(
            phase_provenance=partial,
            runtime_manifest=runtime,
            expected_tasks=tasks,
            active_contract_proof=_active_contract_proof(runtime, partial),
        )

    drifted = copy.deepcopy(runtime)
    drifted["phases"]["main"]["task_ids"] = list(  # type: ignore[index]
        drifted["phases"]["main"]["task_ids"]  # type: ignore[index]
    )[:-1]
    drifted["phases"]["main"]["task_count"] -= 1  # type: ignore[index,operator]
    _rehash(drifted, "runtime_task_manifest_sha256")
    with pytest.raises(AnalysisContractError, match="roster|task"):
        validate_inputs(
            phase_provenance=phase,
            runtime_manifest=drifted,
            expected_tasks=tasks,
            active_contract_proof=_active_contract_proof(drifted, phase),
        )


@pytest.mark.parametrize(
    ("status", "parsed", "outcomes", "expected"),
    [
        (
            "complete",
            {"simulated": True, "answer": "A", "rationale": "ok"},
            {"is_correct": True},
            "correct_answer",
        ),
        (
            "complete",
            {"simulated": True, "answer": "B", "rationale": "ok"},
            {"is_correct": False},
            "incorrect_answer",
        ),
        (
            "complete",
            {"simulated": True, "answer": None, "rationale": "unsure"},
            {"is_correct": False},
            "abstention",
        ),
        ("excluded_schema", None, {"is_correct": None}, "technical_or_schema_failure"),
        (None, None, None, "technical_or_schema_failure"),
    ],
)
def test_controlled_response_state_is_mutually_exclusive(
    status: str | None,
    parsed: dict[str, object] | None,
    outcomes: dict[str, object] | None,
    expected: str,
) -> None:
    from experiments.llm_sim_v2.analyze import controlled_response_state

    record = (
        {"status": status, "parsed_output": parsed, "outcomes": outcomes}
        if status is not None
        else None
    )
    assert controlled_response_state(record) == expected


def test_cluster_bootstrap_is_seeded_provider_equal_and_counts_undefined() -> None:
    from experiments.llm_sim_v2.analyze import cluster_bootstrap_mean

    values = {
        "provider-a": {persona: 0.0 for persona in PERSONAS},
        "provider-b": {
            persona: (1.0 if index == 0 else None)
            for index, persona in enumerate(PERSONAS)
        },
    }
    first = cluster_bootstrap_mean(values, persona_ids=PERSONAS)
    second = cluster_bootstrap_mean(values, persona_ids=PERSONAS)

    assert first == second
    assert first["seed"] == 2026071503
    assert first["resamples"] == 10_000
    assert first["point_estimate"] == pytest.approx(0.5)
    assert 0 < first["undefined_resamples"] < 10_000
    assert first["defined_resamples"] + first["undefined_resamples"] == 10_000


def test_pairwise_agreement_keeps_nc_and_matches_known_cohen_kappa() -> None:
    from experiments.llm_sim_v2.analyze import pairwise_terminal_agreement

    result = pairwise_terminal_agreement(
        {
            "left": {"s1": "A", "s2": "A", "s3": "B", "s4": "NC"},
            "right": {"s1": "A", "s2": "B", "s3": "B", "s4": "NC"},
        },
        subjects=("s1", "s2", "s3", "s4"),
    )
    cell = result["pairs"][0]
    assert cell["denominator"] == 4
    assert cell["exact_agreement_numerator"] == 3
    assert cell["exact_agreement"] == pytest.approx(0.75)
    assert cell["cohen_kappa"] == pytest.approx(7 / 11)
    assert "NC" in result["categories"]


def _binding(phase: dict[str, object]) -> dict[str, object]:
    return {
        "collection_mode": "formal",
        "development_only": False,
        "partial": False,
        "formal_analysis_eligible": True,
        "phase_provenance_sha256": phase["phase_provenance_sha256"],
        "freeze_manifest_sha256": phase["freeze"]["freeze_manifest_sha256"],  # type: ignore[index]
        "source_set_sha256": phase["source"]["source_set_sha256"],  # type: ignore[index]
        "target_set_hash": phase["target"]["target_set_hash"],  # type: ignore[index]
        "grid_sha256": phase["grid_sha256"],
        "prompt_ledger_sha256": phase["prompt"]["prompt_ledger_sha256"],  # type: ignore[index]
        "prompt_revision": phase["prompt"]["revision"],  # type: ignore[index]
        "prompt_contract_sha256": phase["prompt"]["prompt_contract_sha256"],  # type: ignore[index]
        "runtime_task_manifest_sha256": phase["runtime"][  # type: ignore[index]
            "runtime_task_manifest_sha256"
        ],
        "execution_commit": phase["runtime"]["execution_commit"],  # type: ignore[index]
        "runtime_file_set_sha256": phase["runtime"][  # type: ignore[index]
            "runtime_file_set_sha256"
        ],
        "carried_forward_cost_ledger_sha256": phase["budget"][  # type: ignore[index]
            "carried_forward_cost_ledger_sha256"
        ],
        "source_phase_receipt_sha256": phase["budget"][  # type: ignore[index]
            "source_phase_receipt_sha256"
        ],
        "source_record_set_sha256": phase["budget"][  # type: ignore[index]
            "source_record_set_sha256"
        ],
        "carried_forward_total_accounted_cost_yuan": phase["budget"][  # type: ignore[index]
            "carried_forward_total_accounted_cost_yuan"
        ],
    }


def test_analyzer_uses_exact_runner_phase_provenance_binding() -> None:
    import experiments.llm_sim_v2.analyze as analyze
    from experiments.llm_sim_v2.runner import phase_provenance_binding

    tasks = _expected_tasks()
    _, phase = _provenance_fixture(tasks)
    assert analyze._provenance_binding(phase) == phase_provenance_binding(phase)


def test_active_contract_revalidation_passes_carried_forward_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import experiments.llm_sim_v2.analyze as analyze
    import experiments.llm_sim_v2.collect as collect
    import experiments.llm_sim_v2.runner as runner

    tasks = _expected_tasks()
    runtime, phase = _provenance_fixture(tasks)
    carried = _carried_cost_fixture()
    carried_path = (
        tmp_path
        / "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json"
    )
    _write_json_file(carried_path, carried)
    policy = SimpleNamespace(
        max_attempts=4,
        max_tokens=512,
        retry_max_tokens=1024,
        timeout_seconds=60.0,
        concurrency=4,
        failure_threshold=3,
        base_backoff_seconds=1.0,
        max_backoff_seconds=30.0,
        cooldown_seconds=120.0,
        jitter_fraction=0.25,
    )
    contract = SimpleNamespace(
        runtime_manifest=runtime,
        lexicon=(),
        provider_model=lambda provider: f"{provider}-model",
        provider_policy=lambda _provider: policy,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "load_runtime_contract", lambda _repo: contract)
    monkeypatch.setattr(
        runner,
        "verify_runtime_task_manifest",
        lambda *_args, **_kwargs: {
            "ok": True,
            "git_proof": {"byte_identical": True},
        },
    )
    monkeypatch.setattr(runner, "enumerate_tasks", lambda *_args, **_kwargs: [])

    def capture_phase(*_args, **kwargs):
        captured["carried_forward_cost"] = kwargs.get("carried_forward_cost")
        return {"contract_revalidated": True}

    monkeypatch.setattr(runner, "verify_phase_provenance_against_contract", capture_phase)
    monkeypatch.setattr(
        runner,
        "validate_formal_phase_provenance",
        lambda _phase: {"formal_analysis_eligible": True},
    )
    monkeypatch.setattr(
        collect,
        "verify_formal_carried_forward_cost_ledger",
        lambda value: dict(value),
    )

    analyze._validate_active_contract_inputs(
        tmp_path,
        runtime_manifest=runtime,
        phase_provenance=phase,
    )
    assert captured["carried_forward_cost"] == carried


def _response_record(
    task: dict[str, object],
    provider: str,
    phase: dict[str, object],
    *,
    answer: str | None = "A",
    status: str = "complete",
    rationale: str | None = None,
) -> dict[str, object]:
    parsed = (
        {
            "simulated": True,
            "answer": answer,
            "rationale": rationale or f"reason:{answer}",
            **(
                {"abstain": answer is None}
                if task["condition"] == "blind"
                else {}
            ),
        }
        if status == "complete"
        else None
    )
    is_correct = answer == "A" if status == "complete" else None
    target = task.get("target_option")
    target_hit = answer == target if target and status == "complete" else None
    compliance = (
        target_hit
        if target and task["response_arm"] == "deficit"
        else is_correct if target else None
    )
    return {
        "schema_version": "yher.llm_sim_v2.response_record.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "main",
        "analysis_population": "main",
        "collection_mode": "formal",
        "development_only": False,
        "partial": False,
        "formal_analysis_eligible": True,
        "provider": provider,
        "model_id": f"{provider}-model",
        "requested_model": f"{provider}-model",
        "prompt_revision": 0,
        "prompt_contract_sha256": "d" * 64,
        **{
            key: task[key]
            for key in (
                "task_id",
                "persona_id",
                "pair_id",
                "row_id",
                "anchor_id",
                "response_arm",
                "condition",
                "item_id",
                "target_node",
                "family_id",
                "logical_key",
                "message_sha256",
                "wire_message_sha256",
                "is_stability_repeat",
            )
        },
        "attempt_id": task["attempt_id"],
        "status": status,
        "error": "invalid_schema" if status == "excluded_schema" else None,
        "parsed_output": parsed,
        "outcomes": {
            "is_correct": is_correct,
            "target_option_hit": target_hit,
            "manipulation_compliance": compliance,
        },
        "attempts": [
            {
                "attempt": 1,
                "status": "response" if status == "complete" else "failed",
                "request_max_tokens": 512,
                "latency_ms": 1.0,
                "model_returned": f"{provider}-model",
                "finish_reason": "stop",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "cost_yuan": 0.0,
                "cost_known": True,
                "billing_ambiguity": False,
                "cost_reserve_yuan": 0.0,
                **(
                    {"error_category": "invalid_schema"}
                    if status == "excluded_schema"
                    else {}
                ),
            }
        ],
        "retry_count": 0,
        "known_cost_yuan": 0.0,
        "unknown_cost_reserve_yuan": 0.0,
        "cost_yuan": 0.0,
        "has_unknown_cost_attempts": False,
        "needs_user": False,
        "needs_user_reasons": [],
        "provenance": _binding(phase),
    }


def _condition_lifecycle_fixture(
    tasks: list[dict[str, object]],
    records: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for condition in ("controlled", "blind"):
        condition_tasks = [task for task in tasks if task["condition"] == condition]
        primary = [task for task in condition_tasks if not task["is_stability_repeat"]]
        repeats = [task for task in condition_tasks if task["is_stability_repeat"]]
        present = [task for task in condition_tasks if str(task["task_id"]) in records]
        invalid = sum(
            records[str(task["task_id"])]["status"] == "excluded_schema"
            for task in primary
            if str(task["task_id"]) in records
        )
        complete_clusters = 0
        for persona_id in PERSONAS:
            persona_tasks = [
                task for task in primary if task["persona_id"] == persona_id
            ]
            complete_clusters += bool(persona_tasks) and all(
                str(task["task_id"]) in records
                and records[str(task["task_id"])]["status"] == "complete"
                for task in persona_tasks
            )
        invalid_fraction = invalid / len(primary)
        output[condition] = {
            "expected_count": len(condition_tasks),
            "primary_expected_count": len(primary),
            "stability_repeat_expected_count": len(repeats),
            "present_count": len(present),
            "missing_count": len(condition_tasks) - len(present),
            "invalid_schema_count": invalid,
            "invalid_schema_fraction": invalid_fraction,
            "excluded_invalid_schema": condition == "blind"
            and invalid_fraction > 0.5,
            "complete_cluster_count": int(complete_clusters),
            "minimum_complete_clusters": 45,
            "minimum_complete_clusters_met": complete_clusters >= 45,
        }
    return output


def _refresh_provider_manifest(
    provider: str,
    tasks: list[dict[str, object]],
    records: dict[str, dict[str, dict[str, object]]],
    manifests: dict[str, dict[str, object]],
    *,
    provider_lifecycle: str | None = None,
) -> None:
    expected_ids = [str(task["task_id"]) for task in tasks]
    provider_records = records[provider]
    present_ids = [task_id for task_id in expected_ids if task_id in provider_records]
    missing_ids = [task_id for task_id in expected_ids if task_id not in provider_records]
    counts: dict[str, int] = {}
    for record in provider_records.values():
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    lifecycle = provider_lifecycle
    if lifecycle is None:
        if missing_ids:
            lifecycle = "partial_missing"
        elif any(status != "complete" for status in counts):
            lifecycle = "complete_with_exclusions"
        else:
            lifecycle = "complete"
    interrupted_ids = missing_ids if lifecycle == "interrupted" else []
    breaker_ids = missing_ids if lifecycle == "excluded_repeated_failure" else []
    classified = set(interrupted_ids) | set(breaker_ids)
    manifest = manifests[provider]
    needs_user_ids = [
        task_id
        for task_id in expected_ids
        if task_id in provider_records
        and provider_records[task_id].get("needs_user") is True
    ]
    unknown_attempt_count = sum(
        sum(
            attempt.get("cost_known") is False
            for attempt in record.get("attempts", [])
        )
        for record in provider_records.values()
    )
    manifest.update(
        {
            "provider_lifecycle": lifecycle,
            "record_count": len(present_ids),
            "complete_records": counts.get("complete", 0),
            "status_counts": counts,
            "returned_models": sorted(
                {
                    str(record["model_id"])
                    for record in provider_records.values()
                    if record.get("model_id")
                }
            ),
            "condition_lifecycle": _condition_lifecycle_fixture(
                tasks, provider_records
            ),
            "interruption": {
                "interrupted": lifecycle == "interrupted",
                "type": "test_interrupt" if lifecycle == "interrupted" else None,
            },
            "unavailable": {
                "unavailable": lifecycle == "unavailable",
                "error_category": "missing_credentials"
                if lifecycle == "unavailable"
                else None,
            },
            "needs_user": {
                "required": bool(needs_user_ids),
                "reason": "unknown_provider_billing_reserved"
                if needs_user_ids
                else None,
                "record_count": len(needs_user_ids),
                "record_task_ids": needs_user_ids,
                "unknown_cost_attempt_count": unknown_attempt_count,
            },
            "breaker": {
                "status": "open" if breaker_ids else "closed",
                "failure_threshold": 3,
                "consecutive_failures": 3 if breaker_ids else 0,
                "cooldown_seconds": 120.0,
                "opened_at_epoch": 1000.0 if breaker_ids else None,
                "opened_at_utc": (
                    "1970-01-01T00:16:40Z" if breaker_ids else None
                ),
                "resume_not_before_epoch": 1120.0 if breaker_ids else None,
                "resume_not_before_utc": (
                    "1970-01-01T00:18:40Z" if breaker_ids else None
                ),
            },
        }
    )
    manifest["lifecycle"] = {
        "expected_count": len(expected_ids),
        "present_count": len(present_ids),
        "missing_count": len(missing_ids),
        "interrupted_count": len(interrupted_ids),
        "fuse_skipped_count": 0,
        "breaker_skipped_count": len(breaker_ids),
        "expected_task_ids": expected_ids,
        "present_task_ids": present_ids,
        "missing_task_ids": missing_ids,
        "interrupted_task_ids": interrupted_ids,
        "fuse_skipped_task_ids": [],
        "breaker_skipped_task_ids": breaker_ids,
        "unclassified_missing_task_ids": [
            task_id for task_id in missing_ids if task_id not in classified
        ],
    }


def _synthetic_main_fixture() -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, dict[str, object]]],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    tasks = _expected_tasks()
    runtime, phase = _provenance_fixture(tasks)
    records: dict[str, dict[str, dict[str, object]]] = {
        provider: {} for provider in PROVIDERS
    }
    for provider in PROVIDERS:
        for task in tasks:
            task_id = str(task["task_id"])
            answer: str | None
            status = "complete"
            rationale = None
            if task["condition"] == "controlled":
                ordinal = int(task_id.rsplit(":", 1)[1])
                if task["response_arm"] == "deficit":
                    answer = "A" if ordinal == 3 else "B"
                else:
                    answer = "A" if ordinal < 3 else "B"
                if provider == "deepseek" and task_id == "c:persona-00:deficit:0":
                    continue
                if provider == "deepseek" and task_id == "c:persona-00:deficit:1":
                    answer = None
            else:
                persona_index = int(str(task["persona_id"]).rsplit("-", 1)[1])
                if provider == "glm" and persona_index % 10 == 0:
                    answer = "B"
                elif provider == "kimi":
                    answer = "B"
                else:
                    answer = "A"
                if provider == "tongyi" and task["is_stability_repeat"] is False:
                    status = "excluded_schema"
                if (
                    provider == "deepseek"
                    and task_id == "b:persona-00:deficit:1"
                ):
                    continue
                if task["is_stability_repeat"]:
                    rationale = (
                        f"repeat-different:{answer}"
                        if provider == "glm"
                        else f"reason:{answer}"
                    )
            records[provider][task_id] = _response_record(
                task,
                provider,
                phase,
                answer=answer,
                status=status,
                rationale=rationale,
            )

    manifests: dict[str, dict[str, object]] = {}
    expected_ids = [str(task["task_id"]) for task in tasks]
    for provider in PROVIDERS:
        present_ids = [task_id for task_id in expected_ids if task_id in records[provider]]
        missing_ids = [task_id for task_id in expected_ids if task_id not in records[provider]]
        status_counts: dict[str, int] = {}
        for record in records[provider].values():
            status = str(record["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        manifests[provider] = {
            "schema_version": "yher.llm_sim_v2.provider_manifest.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "phase": "main",
            "analysis_population": "main",
            "provider": provider,
            "provider_lifecycle": "partial_missing" if missing_ids else "complete",
            "collection_mode": "formal",
            "development_only": False,
            "partial": False,
            "formal_analysis_eligible": True,
            "prompt_revision": 0,
            "requested_model": f"{provider}-model",
            "returned_models": [f"{provider}-model"],
            "record_count": len(present_ids),
            "complete_records": status_counts.get("complete", 0),
            "status_counts": status_counts,
            "budget": {
                "total_cost_yuan": 0.0,
                "provider_record_known_cost_yuan": 0.0,
                "provider_record_unknown_reserve_yuan": 0.0,
                "provider_record_accounted_cost_yuan": 0.0,
                "soft_warning_triggered": False,
                "hard_fuse_triggered": False,
            },
            "lifecycle": {
                "expected_count": len(expected_ids),
                "present_count": len(present_ids),
                "missing_count": len(missing_ids),
                "interrupted_count": 0,
                "fuse_skipped_count": 0,
                "breaker_skipped_count": 0,
                "expected_task_ids": expected_ids,
                "present_task_ids": present_ids,
                "missing_task_ids": missing_ids,
                "interrupted_task_ids": [],
                "fuse_skipped_task_ids": [],
                "breaker_skipped_task_ids": [],
                "unclassified_missing_task_ids": missing_ids,
            },
            "provenance": _binding(phase),
        }
    for provider in PROVIDERS:
        _refresh_provider_manifest(provider, tasks, records, manifests)
    mapping = {
        "schema_version": "yher.llm_sim_v2.target_option_mapping.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "mapping_sha256": "4" * 64,
        "target_set_hash": "3" * 64,
        "confirmatory_target_misconception_hit_rate": False,
        "mapped_fraction": 0.06,
        "consensus": {
            "mapped_rows": 6,
            "excluded_ambiguous_rows": 94,
        },
    }
    return tasks, runtime, phase, records, manifests, mapping


def test_synthetic_main_computes_denominators_effects_blind_and_stability() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )

    assert result["analysis_population"] == "synthetic_nonformal"
    assert result["independent_cluster_unit"] == "persona_id"
    assert result["independent_cluster_count"] == 50
    assert result["expected_denominator"]["tasks_per_provider"] == len(tasks)
    assert result["expected_denominator"]["provider_task_cells"] == len(tasks) * 6
    provenance = result["collection_provenance"]
    assert provenance["schema_version"] == (
        "yher.llm_sim_v2.collection_provenance.v1"
    )
    assert provenance["temperature"] == 0.0
    assert provenance["top_p"] is None
    assert provenance["seed"] is None
    assert provenance["first_observation_at_utc"] is None
    assert provenance["provider_evidence_event_window_utc"] is None
    protocol = {row["provider"]: row for row in provenance["providers"]}
    assert set(protocol) == set(PROVIDERS)
    assert protocol["deepseek"] == {
        "provider": "deepseek",
        "requested_model": "deepseek-model",
        "returned_models": ["deepseek-model"],
        "observed_model_ids_match_request": True,
        "evidence_event_window_utc": None,
        "observed_request_max_tokens": [512],
        "max_tokens": 512,
        "retry_max_tokens": 1024,
        "timeout_seconds": 60.0,
        "concurrency": 4,
        "max_attempts": 4,
        "failure_threshold": 3,
        "base_backoff_seconds": 1.0,
        "max_backoff_seconds": 30.0,
        "cooldown_seconds": 120.0,
        "jitter_fraction": 0.25,
    }
    deepseek = next(
        row for row in result["provider_lifecycle"] if row["provider"] == "deepseek"
    )
    assert deepseek["missing_count"] == 2
    assert deepseek["requested_model"] == "deepseek-model"
    assert deepseek["returned_models"] == ["deepseek-model"]
    assert deepseek["observed_model_ids_match_request"] is True

    composition = result["controlled"]["composition"]
    for provider_row in composition["by_provider"]:
        for arm_row in provider_row["arms"]:
            assert sum(arm_row["counts"].values()) == arm_row["expected_denominator"]
    aggregate_counts = composition["aggregate_counts"]
    assert aggregate_counts["correct_answer"] > 0
    assert aggregate_counts["incorrect_answer"] > 0
    assert aggregate_counts["abstention"] == 1
    assert aggregate_counts["technical_or_schema_failure"] == 1

    effects = {
        row["metric_id"]: row
        for row in result["controlled"]["paired_effects"]
    }
    assert effects["conditional_answer_accuracy"]["orientation"] == "control_minus_deficit"
    assert effects["conditional_answer_accuracy"]["estimate"] > 0
    assert effects["correct_response_yield"]["estimate"] > 0
    assert effects["incorrect_response_yield"]["orientation"] == "deficit_minus_control"
    assert effects["incorrect_response_yield"]["estimate"] > 0
    assert effects["abstention_yield"]["estimate"] > 0
    assert effects["technical_or_schema_failure_yield"]["estimate"] > 0
    assert all(row["bootstrap"]["resamples"] == 10_000 for row in effects.values())
    assert all(row["bootstrap"]["seed"] == 2026071503 for row in effects.values())

    blind = result["blind"]
    assert blind["excluded_providers"] == ["tongyi"]
    assert blind["provider_schema"]["tongyi"]["invalid_schema_fraction"] == 1.0
    assert blind["agreement"]["nc_retained"] is True
    assert len(blind["agreement"]["pairs"]) == 10
    assert all(row["denominator"] == 100 for row in blind["agreement"]["pairs"])
    assert all(row["exact_agreement_ci95"] for row in blind["agreement"]["pairs"])
    assert all(
        row["exact_agreement_bootstrap"]["resamples"] == 10_000
        for row in blind["agreement"]["pairs"]
    )
    assert "NC" in blind["agreement"]["categories"]

    stability = {row["provider"]: row for row in blind["stability"]}
    assert stability["deepseek"]["expected_pairs"] == 20
    assert stability["deepseek"]["answer_agreement_numerator"] == 19
    assert stability["deepseek"]["canonical_complete_pair_numerator"] == 19
    assert stability["deepseek"]["canonical_complete_pair_denominator"] == 19
    assert stability["deepseek"]["answer_bootstrap"]["resamples"] == 10_000
    assert stability["deepseek"]["canonical_complete_pair_bootstrap"]["ci95"]
    assert stability["glm"]["answer_agreement_numerator"] == 20
    assert stability["glm"]["canonical_complete_pair_numerator"] == 0
    aggregate_stability = blind["stability_provider_equal_aggregate"]
    assert aggregate_stability["eligible_providers"] == [
        "deepseek",
        "glm",
        "kimi",
        "minimax",
        "doubao",
    ]
    assert aggregate_stability["answer"]["resamples"] == 10_000
    assert aggregate_stability["canonical_complete_pair"]["resamples"] == 10_000

    sparse = result["sparse_mapping_descriptive"]
    assert sparse["confirmatory"] is False
    assert sparse["mapped_mapping_rows"] == 6
    assert sparse["total_mapping_rows"] == 100
    assert sparse["target_set_hash"] == "3" * 64
    deficit_sparse = next(
        row
        for row in sparse["by_provider_and_arm"]
        if row["provider"] == "deepseek" and row["response_arm"] == "deficit"
    )
    assert deficit_sparse["random_wrong_option_baseline"] == pytest.approx(1 / 3)
    assert deficit_sparse["incorrect_answer_denominator"] > 0
    assert deficit_sparse["target_hit_minus_random_wrong_baseline"] is not None

    judge = result["judge_adjudication"]
    assert judge["case_manifest"]["selected_count"] == 120
    assert judge["case_manifest"]["selected_stratum_counts"] == {
        "disagreement": 120,
        "agreement": 0,
    }
    assert judge["analysis"]["status"] == "missing_all_judges"
    assert judge["analysis"]["missing_judges"] == ["claude", "gpt"]


def test_record_and_manifest_reconciliation_rejects_cross_phase_or_extra_cells() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    bad = copy.deepcopy(records)
    record = copy.deepcopy(next(iter(bad["glm"].values())))
    record["phase"] = "pilot"
    bad["glm"][str(record["task_id"])] = record
    with pytest.raises(AnalysisContractError, match="record.*main|phase"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=bad,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )

    extra = copy.deepcopy(records)
    record = copy.deepcopy(next(iter(extra["glm"].values())))
    record["task_id"] = "not-in-runtime-roster"
    extra["glm"]["not-in-runtime-roster"] = record
    with pytest.raises(AnalysisContractError, match="extra|roster"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=extra,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


@pytest.mark.parametrize(
    ("answer", "extra"),
    [
        pytest.param("Z", {}, id="unknown-option"),
        pytest.param("a", {}, id="lowercase-not-normalized"),
        pytest.param("A", {"unexpected": "field"}, id="extra-schema-key"),
    ],
)
def test_complete_record_requires_exact_runner_normalized_output(
    answer: str,
    extra: dict[str, object],
) -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset
    from experiments.llm_sim_v2.runner import compute_outcomes

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    task = next(task for task in tasks if task["condition"] == "controlled")
    record = records["glm"][str(task["task_id"])]
    record["parsed_output"] = {
        "simulated": True,
        "answer": answer,
        "rationale": "synthetic rationale",
        **extra,
    }
    record["outcomes"] = compute_outcomes(
        condition=str(task["condition"]),
        response_arm=str(task["response_arm"]),
        answer=answer,
        abstain=False,
        correct_option=str(task["correct_option"]),
        target_option=(
            str(task["target_option"])
            if task.get("target_option") is not None
            else None
        ),
    )

    with pytest.raises(AnalysisContractError, match="parsed|schema|option|normalized"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_complete_record_requires_successful_terminal_response_attempt() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    task = next(task for task in tasks if task["condition"] == "controlled")
    record = records["glm"][str(task["task_id"])]
    record["attempts"][-1].update(  # type: ignore[index]
        {"status": "failed", "error_category": "invalid_schema"}
    )

    with pytest.raises(AnalysisContractError, match="complete.*attempt|terminal.*response"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


@pytest.mark.parametrize(
    ("status", "error", "model_id", "attempt_model"),
    [
        pytest.param(
            "excluded_schema",
            "provider response schema keys drifted",
            "glm-model",
            "glm-model",
            id="schema-declared-with-response-attempt",
        ),
        pytest.param(
            "excluded_model_drift",
            "timeout",
            "drifted-model",
            "drifted-model",
            id="model-drift-with-wrong-error",
        ),
        pytest.param(
            "technical_failure",
            "timeout",
            "glm-model",
            "glm-model",
            id="technical-failure-with-response-attempt",
        ),
    ],
)
def test_noncomplete_record_status_must_match_terminal_attempt_semantics(
    status: str,
    error: str,
    model_id: str,
    attempt_model: str,
) -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    task = next(task for task in tasks if task["condition"] == "controlled")
    record = records["glm"][str(task["task_id"])]
    record.update(
        {
            "status": status,
            "error": error,
            "model_id": model_id,
            "parsed_output": None,
            "outcomes": {
                "is_correct": None,
                "target_option_hit": None,
                "manipulation_compliance": None,
            },
        }
    )
    record["attempts"][-1]["model_returned"] = attempt_model  # type: ignore[index]
    _refresh_provider_manifest("glm", tasks, records, manifests)

    with pytest.raises(
        AnalysisContractError,
        match="status|terminal.*attempt|schema|model.*drift|technical",
    ):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


@pytest.mark.parametrize("mutation", ["missing-finish-reason", "token-policy-drift"])
def test_response_attempt_requires_runner_fields_and_frozen_token_policy(
    mutation: str,
) -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    task = next(task for task in tasks if task["condition"] == "controlled")
    attempt = records["glm"][str(task["task_id"])]["attempts"][-1]
    if mutation == "missing-finish-reason":
        attempt.pop("finish_reason")  # type: ignore[union-attr]
    else:
        attempt["request_max_tokens"] = 999  # type: ignore[index]

    with pytest.raises(AnalysisContractError, match="attempt|token|response.*field"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_cli_accepts_main_output_base_and_result_dir_without_analysis_knobs() -> None:
    from experiments.llm_sim_v2.analyze import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-base",
            "/tmp/persona-v2-main-input",
            "--result-dir",
            "/tmp/persona-v2-main-results",
        ]
    )
    assert args.output_base == Path("/tmp/persona-v2-main-input")
    assert args.result_dir == Path("/tmp/persona-v2-main-results")
    with_judges = parser.parse_args(
        [
            "--output-base",
            "/tmp/persona-v2-main-input",
            "--result-dir",
            "/tmp/persona-v2-main-results-judged",
            "--judge-results-dir",
            "/tmp/persona-v2-judge-results",
        ]
    )
    assert with_judges.judge_results_dir == Path("/tmp/persona-v2-judge-results")
    prepared = parser.parse_args(
        [
            "--output-base",
            "/tmp/persona-v2-main-input",
            "--judge-results-dir",
            "/tmp/persona-v2-prepared-judge",
            "--prepare-judge-cases",
        ]
    )
    assert prepared.prepare_judge_cases is True
    assert prepared.result_dir is None
    help_text = parser.format_help()
    assert "--prepare-judge-cases" in help_text
    assert "--bootstrap-seed" not in help_text
    assert "--bootstrap-resamples" not in help_text


def test_missing_judge_result_directory_is_a_contract_error(tmp_path: Path) -> None:
    from experiments.llm_sim_v2.analyze import (
        AnalysisContractError,
        _load_judge_result_manifests,
    )

    with pytest.raises(AnalysisContractError, match="judge result.*missing|cannot resolve"):
        _load_judge_result_manifests(tmp_path / "missing-judge-results")


def test_final_run_analysis_rejects_missing_judge_root_before_output(
    tmp_path: Path,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    destination = tmp_path / "must-not-exist"
    with pytest.raises(
        analyze.AnalysisContractError, match="finalized judge run root"
    ):
        analyze.run_analysis(
            output_base=tmp_path / "collection",
            result_dir=destination,
            repo_root=tmp_path,
        )
    assert not destination.exists()


def test_machine_outputs_include_csv_figure_data_png_svg_and_hash_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    output = tmp_path / "analysis-results"
    judge_root, prepared_manifest = _prepare_and_finalize_zero_case_judge_run(
        analyze=analyze,
        output_base=output_base,
        repo=repo,
        tmp_path=tmp_path,
    )
    result = analyze.run_analysis(
        output_base=output_base,
        result_dir=output,
        repo_root=repo,
        judge_results_dir=judge_root,
    )
    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )

    assert result["analysis_mode"] == "formal_main"
    assert result["formal_analysis_eligible"] is True
    assert (
        result["judge_adjudication"]["case_manifest"]["case_manifest_sha256"]
        == prepared_manifest["case_manifest_sha256"]
    )
    with pytest.raises(
        analyze.AnalysisContractError, match="input artifact manifest|formal.*loader"
    ):
        analyze.write_analysis_outputs(result, tmp_path / "missing-input-manifest")
    input_manifest = json.loads(
        (output / "input_artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert not hasattr(analyze, "_FORMAL_PUBLICATION_SENTINEL")
    forged_proof = analyze._FormalPublicationProof(
        loader_bundle_sha256=result["formal_loader_bundle_sha256"],
        result_sha256=analyze._canonical_sha(result),
        input_artifact_manifest_sha256=analyze._canonical_sha(input_manifest),
        authorization_mac="0" * 64,
    )
    forged_destination = tmp_path / "forged-formal-proof"
    with pytest.raises(analyze.AnalysisContractError, match="publication.*proof"):
        analyze.write_analysis_outputs(
            result,
            forged_destination,
            input_artifact_manifest=input_manifest,
            _formal_publication_proof=forged_proof,
        )
    assert not forged_destination.exists()
    tampered_manifest = copy.deepcopy(input_manifest)
    tampered_manifest["input_file_count"] += 1
    with pytest.raises(
        analyze.AnalysisContractError,
        match="publication.*proof|input artifact manifest|binding",
    ):
        analyze.write_analysis_outputs(
            result,
            tmp_path / "tampered-input-manifest",
            input_artifact_manifest=tampered_manifest,
        )

    tampered_cost = copy.deepcopy(result)
    tampered_cost["cost_accounting"]["provider_phase"][0][
        "accounted_cost_yuan"
    ] = 1.0
    with pytest.raises(analyze.AnalysisContractError, match="publication.*cost"):
        analyze._validate_publication_cost_accounting(tampered_cost)
    assert not (tmp_path / "tampered-cost").exists()

    required = {
        "analysis_results.json",
        "input_artifact_manifest.json",
        "provider_lifecycle.csv",
        "controlled_composition.csv",
        "controlled_paired_effects.csv",
        "blind_agreement.csv",
        "blind_stability.csv",
        "sparse_mapping_descriptive.csv",
        "cost_by_provider_phase.csv",
        "cost_reconciliation_artifact_manifest.json",
        "judge/case_manifest.json",
        "judge/claude_input.jsonl",
        "judge/gpt_input.jsonl",
        "judge/judge_analysis.json",
        "judge/judge_category_counts.csv",
        "judge/judge_label_disagreements.json",
        "judge/judge_error_category_disagreements.json",
        "figure_data/controlled_composition.csv",
        "figure_data/blind_agreement.csv",
        "figure_data/blind_stability.csv",
        "figures/controlled_composition.png",
        "figures/controlled_composition.svg",
        "figures/blind_terminal_agreement.png",
        "figures/blind_terminal_agreement.svg",
        "figures/blind_output_stability.png",
        "figures/blind_output_stability.svg",
        "artifact_manifest.json",
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert required <= actual

    for name in (
        "controlled_composition",
        "blind_terminal_agreement",
        "blind_output_stability",
    ):
        png = output / "figures" / f"{name}.png"
        svg = output / "figures" / f"{name}.svg"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert png.stat().st_size > 10_000
        assert b"<svg" in svg.read_bytes()[:500]
        assert svg.stat().st_size > 5_000
    stability_svg = (
        output / "figures/blind_output_stability.svg"
    ).read_text(encoding="utf-8")
    assert "(excluded)" in stability_svg
    assert "NA" in stability_svg
    assert (output / "judge/claude_input.jsonl").read_bytes() == (
        output / "judge/gpt_input.jsonl"
    ).read_bytes()
    judge_analysis = json.loads(
        (output / "judge/judge_analysis.json").read_text(encoding="utf-8")
    )
    assert judge_analysis["status"] == "not_applicable_zero_cases"
    assert judge_analysis["missing_judges"] == []

    disk_manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert disk_manifest == manifest
    assert disk_manifest["target_set_hash"] == "3" * 64
    paths = {row["path"] for row in disk_manifest["artifacts"]}
    assert "artifact_manifest.json" not in paths
    assert paths == actual - {"artifact_manifest.json"}
    for row in disk_manifest["artifacts"]:
        path = output / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert path.stat().st_size == row["size"]

    payload = json.loads((output / "analysis_results.json").read_text(encoding="utf-8"))
    assert payload["bootstrap_contract"]["resamples"] == 10_000
    assert payload["outputs"]["figure_data_machine_readable"] is True
    assert payload["outputs"]["publication_figures"] == 3


def test_public_in_memory_analysis_is_nonformal_and_writer_refuses(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.analyze import (
        AnalysisContractError,
        analyze_dataset,
        write_analysis_outputs,
    )

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )

    assert result["schema_version"] == (
        "yher.llm_sim_v2.analysis_results.synthetic_nonformal.v1"
    )
    assert result["analysis_mode"] == "synthetic_nonformal"
    assert result["formal_analysis_eligible"] is False
    with pytest.raises(AnalysisContractError, match="nonformal|formal.*loader|publication"):
        write_analysis_outputs(result, tmp_path / "must-not-write")


def test_relabelled_nonformal_result_cannot_reach_publication_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze
    from experiments.llm_sim_v2.analyze import (
        AnalysisContractError,
        analyze_dataset,
        write_analysis_outputs,
    )

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )
    forged = copy.deepcopy(result)
    forged.update(
        {
            "schema_version": "yher.llm_sim_v2.analysis_results.v1",
            "analysis_population": "main",
            "analysis_mode": "formal_main",
            "formal_analysis_eligible": True,
            "publication_output_eligible": True,
        }
    )
    empty_files: list[dict[str, object]] = []
    input_manifest = {
        "schema_version": "yher.llm_sim_v2.analysis_input_artifact_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "analysis_population": "main",
        "files": empty_files,
        "input_file_count": 0,
        "record_file_count": 0,
        "input_file_set_sha256": _sha(empty_files),
    }
    forged["input_artifact_binding"] = {
        "input_file_count": 0,
        "record_file_count": 0,
        "input_file_set_sha256": input_manifest["input_file_set_sha256"],
        "input_artifact_manifest_sha256": _sha(input_manifest),
    }

    destination = tmp_path / "forged-publication"
    def forbidden_side_effect(*_args: object, **_kwargs: object) -> None:
        pytest.fail("publication side effect ran before proof validation")

    monkeypatch.setattr(analyze, "_load_pyplot", forbidden_side_effect)
    monkeypatch.setattr(analyze.tempfile, "mkdtemp", forbidden_side_effect)
    with pytest.raises(AnalysisContractError, match="publication.*proof|loader.*proof"):
        write_analysis_outputs(
            forged,
            destination,
            input_artifact_manifest=input_manifest,
        )
    assert not destination.exists()


def test_self_hashed_cost_cannot_bypass_formal_loader_proof() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    empty_files: list[dict[str, object]] = []
    fake_cost = {
        "schema_version": "yher.llm_sim_v2.cost_accounting.v1",
        "currency": "CNY",
        "provider_phase": [],
        "source": "fabricated",
        "cost_reconciliation_artifact_manifest": {
            "schema_version": (
                "yher.llm_sim_v2.cost_reconciliation_artifact_manifest.v1"
            ),
            "simulated": True,
            "run_id": RUN_ID,
            "metric_input": False,
            "file_count": 0,
            "files": empty_files,
            "file_set_sha256": _sha(empty_files),
        },
    }

    with pytest.raises(AnalysisContractError, match="loader.*proof|formal.*proof"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
            cost_accounting=fake_cost,
        )


def _write_json_file(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _filesystem_main_fixture(
    tmp_path: Path,
    *,
    anchor_state: str = "committed",
) -> tuple[Path, Path, list[dict[str, object]]]:
    if anchor_state not in {"committed", "untracked"}:
        raise ValueError("anchor_state must be committed or untracked")
    tasks = _expected_tasks()
    runtime, phase = _provenance_fixture(tasks)
    output_base = tmp_path / "collection"
    main_root = output_base / RUN_ID / "main"
    repo = tmp_path / "repo"
    prior = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/llm_sim_v2/prior_cost_ledger.json"
        ).read_text(encoding="utf-8")
    )
    carried = _carried_cost_fixture()
    _write_json_file(repo / "experiments/llm_sim_v2/prior_cost_ledger.json", prior)
    _write_json_file(
        repo
        / "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json",
        carried,
    )
    _write_json_file(main_root / "phase_provenance.json", phase)
    _write_json_file(
        main_root.parent / "run_budget_ledger.json",
        {
            "schema_version": "yher.llm_sim_v2.run_budget_ledger.v2",
            "simulated": True,
            "run_id": RUN_ID,
            "prior_cost_ledger_sha256": prior["prior_cost_ledger_sha256"],
            "prior_known_cost_yuan": prior["known_cost_yuan"],
            "prior_ambiguity_reserve_yuan": prior[
                "pre_run_ambiguity_reserve_yuan"
            ],
            "prior_documented_cost_yuan": prior["pre_run_total_bound_yuan"],
            "carried_forward_cost_ledger_sha256": carried[
                "carried_forward_cost_ledger_sha256"
            ],
            "carried_forward_source_phase": "pilot",
            "carried_forward_source_record_set_sha256": carried[
                "source_record_set_sha256"
            ],
            "carried_forward_source_phase_receipt_sha256": carried[
                "source_phase_receipt_sha256"
            ],
            "carried_forward_known_cost_yuan": carried["known_cost_yuan"],
            "carried_forward_unknown_reserve_yuan": carried[
                "unknown_cost_reserve_yuan"
            ],
            "carried_forward_total_accounted_cost_yuan": carried[
                "total_accounted_cost_yuan"
            ],
            "unknown_attempt_reserve_yuan": prior[
                "unknown_attempt_reserve_yuan"
            ],
            "immutable_record_known_cost_yuan": 0.0,
            "immutable_record_unknown_reserve_yuan": 0.0,
            "immutable_record_cost_yuan": 0.0,
            "total_known_cost_yuan": round(
                prior["known_cost_yuan"] + carried["known_cost_yuan"], 8
            ),
            "total_unknown_reserve_yuan": prior[
                "pre_run_ambiguity_reserve_yuan"
            ],
            "total_accounted_cost_yuan": round(
                prior["pre_run_total_bound_yuan"]
                + carried["total_accounted_cost_yuan"],
                8,
            ),
            "needs_user": False,
            "needs_user_reasons": [],
            "soft_warning_yuan": 300.0,
            "hard_fuse_yuan": 450.0,
            "updated_at_utc": "2026-07-15T12:00:00Z",
        },
    )
    _write_json_file(
        repo / "experiments/llm_sim_v2/runtime_task_manifest.json", runtime
    )
    _write_json_file(
        repo / "experiments/llm_sim_v2/frozen_v0/target_option_mapping.json",
        {
            "schema_version": "yher.llm_sim_v2.target_option_mapping.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "mapping_sha256": "4" * 64,
            "target_set_hash": "3" * 64,
            "confirmatory_target_misconception_hit_rate": False,
            "mapped_fraction": 0.06,
            "consensus": {
                "mapped_rows": 6,
                "excluded_ambiguous_rows": 94,
            },
        },
    )
    expected_ids = [str(task["task_id"]) for task in tasks]
    for provider in PROVIDERS:
        from experiments.llm_sim_v2.evidence import ProviderEvidenceLedger

        (main_root / "records" / provider).mkdir(parents=True, exist_ok=True)
        evidence = ProviderEvidenceLedger(
            main_root,
            run_id=RUN_ID,
            phase="main",
            provider=provider,
        )
        invocation = evidence.begin_invocation(
            expected_task_ids=expected_ids,
            resumed_task_ids=(),
        )
        finished_at = "2026-07-15T12:00:00Z"
        needs_user = {
            "required": False,
            "reason": None,
            "record_count": 0,
            "record_task_ids": [],
            "unknown_cost_attempt_count": 0,
        }
        manifest: dict[str, object] = {
                "schema_version": "yher.llm_sim_v2.provider_manifest.v1",
                "simulated": True,
                "run_id": RUN_ID,
                "phase": "main",
                "analysis_population": "main",
                "provider": provider,
                "provider_lifecycle": "partial_missing",
                "collection_mode": "formal",
                "development_only": False,
                "partial": False,
                "formal_analysis_eligible": True,
                "prompt_revision": 0,
                "requested_model": f"{provider}-model",
                "returned_models": [],
                "record_count": 0,
                "complete_records": 0,
                "status_counts": {},
                "condition_lifecycle": _condition_lifecycle_fixture(tasks, {}),
                "interruption": {"interrupted": False, "type": None},
                "unavailable": {"unavailable": False, "error_category": None},
                "needs_user": needs_user,
                "breaker": {
                    "status": "closed",
                    "failure_threshold": 3,
                    "consecutive_failures": 0,
                    "cooldown_seconds": 120.0,
                    "opened_at_epoch": None,
                    "opened_at_utc": None,
                    "resume_not_before_epoch": None,
                    "resume_not_before_utc": None,
                },
                "budget": {
                    "total_cost_yuan": 0.0,
                    "provider_record_known_cost_yuan": 0.0,
                    "provider_record_unknown_reserve_yuan": 0.0,
                    "provider_record_accounted_cost_yuan": 0.0,
                    "soft_warning_triggered": False,
                    "hard_fuse_triggered": False,
                },
                "lifecycle": {
                    "expected_count": len(expected_ids),
                    "present_count": 0,
                    "missing_count": len(expected_ids),
                    "interrupted_count": 0,
                    "fuse_skipped_count": 0,
                    "breaker_skipped_count": 0,
                    "expected_task_ids": expected_ids,
                    "present_task_ids": [],
                    "missing_task_ids": expected_ids,
                    "interrupted_task_ids": [],
                    "fuse_skipped_task_ids": [],
                    "breaker_skipped_task_ids": [],
                    "unclassified_missing_task_ids": expected_ids,
                },
                "provenance": _binding(phase),
                "finished_at_utc": finished_at,
        }
        from experiments.llm_sim_v2.evidence import build_provider_record_set

        manifest["record_set"] = build_provider_record_set(
            main_root,
            provider=provider,
            expected_task_ids=expected_ids,
        )
        event: dict[str, object] = {
            "schema_version": "yher.llm_sim_v2.provider_lifecycle_event.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "phase": "main",
            "analysis_population": "main",
            "provider": provider,
            "event_index": 0,
            "provider_lifecycle": manifest["provider_lifecycle"],
            "lifecycle": manifest["lifecycle"],
            "interruption": manifest["interruption"],
            "unavailable": manifest["unavailable"],
            "needs_user": needs_user,
            "provenance": manifest["provenance"],
            "finished_at_utc": finished_at,
        }
        event["lifecycle_event_sha256"] = _sha(event)
        event_relative = (
            f"provider_lifecycle/{provider}/"
            f"0000-{event['lifecycle_event_sha256']}.json"
        )
        manifest["lifecycle_history"] = [
            {
                "event_index": 0,
                "provider_lifecycle": manifest["provider_lifecycle"],
                "finished_at_utc": finished_at,
                "lifecycle_event_sha256": event["lifecycle_event_sha256"],
                "path": event_relative,
            }
        ]
        _write_json_file(main_root / event_relative, event)
        _write_json_file(
            main_root / "provider_manifests" / f"{provider}.json", manifest
        )
        evidence.finish_invocation(invocation, status="complete")
    from experiments.llm_sim_v2.evidence import write_phase_evidence_receipt

    receipt = write_phase_evidence_receipt(
        main_root,
        phase_provenance=phase,
        tasks=tasks,
    )
    anchor_path = (
        repo
        / "experiments/llm_sim_v2/evidence_anchors/main_phase_evidence_receipt.json"
    )
    anchored_receipt = write_phase_evidence_receipt(
        main_root,
        output=anchor_path,
        phase_provenance=phase,
        tasks=tasks,
    )
    assert anchored_receipt == receipt
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    if anchor_state == "committed":
        subprocess.run(
            ["git", "add", anchor_path.relative_to(repo).as_posix()],
            cwd=repo,
            check=True,
        )
        commit_args = ["git", "commit", "-q", "-m", "anchor main phase evidence"]
    else:
        commit_args = ["git", "commit", "-q", "--allow-empty", "-m", "baseline"]
    subprocess.run(
        [
            *commit_args,
            "--author",
            "Test Runner <tests@example.invalid>",
        ],
        cwd=repo,
        check=True,
        env={
            **os.environ,
            "GIT_COMMITTER_NAME": "Test Runner",
            "GIT_COMMITTER_EMAIL": "tests@example.invalid",
        },
    )
    if anchor_state == "committed":
        subprocess.run(
            [
                "git",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "descendant head",
                "--author",
                "Test Runner <tests@example.invalid>",
            ],
            cwd=repo,
            check=True,
            env={
                **os.environ,
                "GIT_COMMITTER_NAME": "Test Runner",
                "GIT_COMMITTER_EMAIL": "tests@example.invalid",
            },
        )
    return output_base, repo, tasks


def _prepare_and_finalize_zero_case_judge_run(
    *,
    analyze: object,
    output_base: Path,
    repo: Path,
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    from experiments.llm_sim_v2 import judge_execution

    root = tmp_path / "judge-results"
    manifest = analyze.prepare_judge_cases(  # type: ignore[attr-defined]
        output_base=output_base,
        judge_results_dir=root,
        repo_root=repo,
    )
    assert manifest["selected_count"] == 0
    judge_execution.mint_judge_budget_authority(
        case_manifest=manifest,
        output_root=root,
        repo_root=repo,
        run_budget_ledger=output_base / RUN_ID / "run_budget_ledger.json",
    )
    for family in ("gpt", "claude"):
        judge_execution.record_judge_family_disposition(
            case_manifest=manifest,
            output_root=root,
            judge_family=family,
            status="not_applicable_zero_cases",
            reason_code="selected_case_count_zero",
        )
    internal = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=manifest,
        output_root=root,
    )
    anchor = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=manifest,
        output_root=root,
        output=(
            repo
            / "experiments/llm_sim_v2/evidence_anchors/"
            "judge_run_evidence_receipt.json"
        ),
    )
    assert anchor.read_bytes() == internal.read_bytes()
    subprocess.run(
        ["git", "add", anchor.relative_to(repo).as_posix()], cwd=repo, check=True
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            "anchor finalized judge run",
            "--author",
            "Test Runner <tests@example.invalid>",
        ],
        cwd=repo,
        check=True,
        env={
            **os.environ,
            "GIT_COMMITTER_NAME": "Test Runner",
            "GIT_COMMITTER_EMAIL": "tests@example.invalid",
        },
    )
    return root, manifest


def test_prepare_judge_cases_writes_only_a_nonpublication_case_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    root = tmp_path / "prepared-judge"
    manifest = analyze.prepare_judge_cases(
        output_base=output_base,
        judge_results_dir=root,
        repo_root=repo,
    )

    assert manifest["selected_count"] == 0
    assert {path.relative_to(root).as_posix() for path in root.rglob("*")} == {
        "case_manifest.json"
    }
    loaded = analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)
    pre_adjudication = analyze.analyze_dataset(**loaded)
    assert pre_adjudication["analysis_mode"] == "formal_main_pre_adjudication"
    assert pre_adjudication["formal_analysis_eligible"] is False
    assert pre_adjudication["publication_output_eligible"] is False
    assert (
        pre_adjudication["judge_adjudication"]["case_manifest"]
        ["case_manifest_sha256"]
        == manifest["case_manifest_sha256"]
    )


def test_run_analysis_reconstructs_expected_denominator_when_record_store_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(
        analyze,
        "_load_expected_tasks",
        lambda _repo, _runtime: tasks,
    )
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    result_dir = tmp_path / "result"
    judge_root, _prepared_manifest = _prepare_and_finalize_zero_case_judge_run(
        analyze=analyze,
        output_base=output_base,
        repo=repo,
        tmp_path=tmp_path,
    )
    result = analyze.run_analysis(
        output_base=output_base,
        result_dir=result_dir,
        repo_root=repo,
        judge_results_dir=judge_root,
    )

    assert result["expected_denominator"]["tasks_per_provider"] == len(tasks)
    assert result["expected_denominator"]["provider_task_cells"] == len(tasks) * 6
    assert all(
        row["missing_count"] == len(tasks) for row in result["provider_lifecycle"]
    )
    assert result["blind"]["terminal_subject_count"] == 100
    assert result["blind"]["agreement"]["categories"] == ["NC"]
    conditional = next(
        row
        for row in result["controlled"]["paired_effects"]
        if row["metric_id"] == "conditional_answer_accuracy"
    )
    assert conditional["estimate"] is None
    assert conditional["bootstrap"]["undefined_resamples"] == 10_000
    assert (result_dir / "artifact_manifest.json").is_file()


@pytest.mark.parametrize(
    "missing_kind",
    ["internal_receipt", "committed_anchor", "provider_event"],
)
def test_formal_loader_requires_authoritative_phase_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_kind: str,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    main_root = output_base / RUN_ID / "main"
    if missing_kind == "internal_receipt":
        next((main_root / "evidence/phase_receipts").glob("*.json")).unlink()
    elif missing_kind == "committed_anchor":
        (
            repo
            / "experiments/llm_sim_v2/evidence_anchors/"
            "main_phase_evidence_receipt.json"
        ).unlink()
    else:
        next((main_root / "evidence/provider_events/deepseek").glob("*.json")).unlink()

    with pytest.raises(
        analyze.AnalysisContractError,
        match="phase evidence|phase receipt|provider evidence|event",
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_formal_loader_rejects_untracked_phase_evidence_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(
        tmp_path, anchor_state="untracked"
    )
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    anchor_relative = (
        "experiments/llm_sim_v2/evidence_anchors/"
        "main_phase_evidence_receipt.json"
    )
    assert subprocess.run(
        ["git", "ls-files", "--error-unmatch", anchor_relative],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).returncode != 0

    with pytest.raises(
        analyze.AnalysisContractError,
        match="phase evidence receipt.*not committed|repository HEAD",
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_formal_loader_rejects_committed_but_stale_phase_evidence_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze
    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        write_phase_evidence_receipt,
    )

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    main_root = output_base / RUN_ID / "main"
    phase = json.loads((main_root / "phase_provenance.json").read_text(encoding="utf-8"))
    expected_ids = [str(task["task_id"]) for task in tasks]
    ledger = ProviderEvidenceLedger(
        main_root,
        run_id=RUN_ID,
        phase="main",
        provider="deepseek",
    )
    invocation = ledger.begin_invocation(
        expected_task_ids=expected_ids,
        resumed_task_ids=(),
    )
    ledger.finish_invocation(invocation, status="complete")
    write_phase_evidence_receipt(
        main_root,
        phase_provenance=phase,
        tasks=tasks,
    )

    with pytest.raises(
        analyze.AnalysisContractError,
        match="committed main phase evidence receipt differs|stale",
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_formal_loader_binds_committed_receipt_and_every_provider_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    loaded = analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)
    phase_evidence = loaded["phase_evidence"]
    proof = phase_evidence["committed_anchor"]
    assert proof["anchor_is_ancestor_of_head"] is True
    assert proof["anchor_commit"] != proof["head_commit"]

    rows = {
        str(row["path"]): row
        for row in loaded["input_artifact_manifest"]["files"]
    }
    anchor_relative = (
        "repo/experiments/llm_sim_v2/evidence_anchors/"
        "main_phase_evidence_receipt.json"
    )
    assert rows[anchor_relative]["sha256"] == proof["sha256"]
    main_root = output_base / RUN_ID / "main"
    receipt_sha = phase_evidence["receipt"]["phase_evidence_receipt_sha256"]
    assert f"main/evidence/phase_receipts/{receipt_sha}.json" in rows
    expected_event_paths = {
        f"main/{path.relative_to(main_root).as_posix()}"
        for provider in PROVIDERS
        for path in sorted((main_root / "evidence/provider_events" / provider).glob("*.json"))
    }
    assert expected_event_paths
    assert expected_event_paths.issubset(rows)
    for relative in expected_event_paths:
        path = main_root / relative.removeprefix("main/")
        assert rows[relative]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert rows[relative]["size"] == path.stat().st_size


def test_formal_loader_revalidates_phase_evidence_after_hashing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    original = analyze._input_file_row
    mutated = False

    def mutate_after_hash(path: Path, *, relative: str) -> dict[str, object]:
        nonlocal mutated
        row = original(path, relative=relative)
        if not mutated and relative.startswith("main/evidence/provider_events/"):
            path.write_bytes(path.read_bytes() + b"\n")
            mutated = True
        return row

    monkeypatch.setattr(analyze, "_input_file_row", mutate_after_hash)
    with pytest.raises(
        analyze.AnalysisContractError,
        match="phase evidence|receipt|changed while analysis inputs",
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)
    assert mutated is True


def test_formal_loader_rejects_nonfinite_json_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    budget_path = output_base / RUN_ID / "run_budget_ledger.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["ignored_nonfinite_probe"] = "NONFINITE_PROBE"
    encoded = json.dumps(budget, ensure_ascii=False, sort_keys=True, indent=2)
    budget_path.write_text(
        encoded.replace('"NONFINITE_PROBE"', "NaN") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        analyze.AnalysisContractError,
        match="non-finite|JSON constant|cannot read run budget ledger",
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_formal_loader_rejects_non_phase_input_mutation_after_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    original = analyze._input_file_row
    mutated = False

    def mutate_runtime_after_hash(path: Path, *, relative: str) -> dict[str, object]:
        nonlocal mutated
        row = original(path, relative=relative)
        if (
            not mutated
            and relative
            == "repo/experiments/llm_sim_v2/runtime_task_manifest.json"
        ):
            path.write_bytes(path.read_bytes() + b"\n")
            mutated = True
        return row

    monkeypatch.setattr(analyze, "_input_file_row", mutate_runtime_after_hash)
    with pytest.raises(
        analyze.AnalysisContractError,
        match=(
            "changed while (analysis inputs|it was parsed)|input snapshot|"
            "runtime manifest"
        ),
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)
    assert mutated is True


def test_formal_loader_rejects_symlinked_phase_anchor_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    anchor_root = repo / "experiments/llm_sim_v2/evidence_anchors"
    target_root = repo / "phase-anchor-target"
    target_root.mkdir()
    for path in sorted(anchor_root.iterdir()):
        (target_root / path.name).write_bytes(path.read_bytes())
        path.unlink()
    anchor_root.rmdir()
    anchor_root.symlink_to(target_root, target_is_directory=True)

    with pytest.raises(
        analyze.AnalysisContractError,
        match="symlink|committed phase evidence",
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_loader_rejects_flat_or_unknown_record_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(
        analyze,
        "_load_expected_tasks",
        lambda _repo, _runtime: tasks,
    )
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    flat = output_base / RUN_ID / "main/records/legacy-flat.json"
    _write_json_file(flat, {"task_id": "not-allowed"})
    with pytest.raises(analyze.AnalysisContractError, match="provider namespace|record"):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_formal_record_v2_requires_raw_response_binding_and_strict_replay() -> None:
    import experiments.llm_sim_v2.analyze as analyze
    from experiments.llm_sim_v2.evidence import bind_response_content

    tasks, runtime, phase, records, _manifests, _mapping = _synthetic_main_fixture()
    task = tasks[0]
    v1_record = records["glm"][str(task["task_id"])]
    policy = _active_contract_proof(runtime, phase)["provider_attempt_policies"][
        "glm"
    ]
    analyze._validate_record(
        v1_record,
        provider="glm",
        task=task,
        phase=phase,
        expected_model="glm-model",
        attempt_policy=policy,
        formal_mode=False,
    )
    with pytest.raises(analyze.AnalysisContractError, match="record.*v2|schema"):
        analyze._validate_record(
            v1_record,
            provider="glm",
            task=task,
            phase=phase,
            expected_model="glm-model",
            attempt_policy=policy,
            formal_mode=True,
        )

    v2_record = copy.deepcopy(v1_record)
    v2_record["schema_version"] = "yher.llm_sim_v2.response_record.v2"
    raw = json.dumps(
        v2_record["parsed_output"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    v2_record["attempts"][-1].update(bind_response_content(raw))
    analyze._validate_record(
        v2_record,
        provider="glm",
        task=task,
        phase=phase,
        expected_model="glm-model",
        attempt_policy=policy,
        formal_mode=True,
    )

    retried_after_success = copy.deepcopy(v2_record)
    second_attempt = copy.deepcopy(retried_after_success["attempts"][0])
    second_attempt["attempt"] = 2
    retried_after_success["attempts"].append(second_attempt)
    retried_after_success["retry_count"] = 1
    with pytest.raises(
        analyze.AnalysisContractError,
        match="successful response must be final|shared strict replay",
    ):
        analyze._validate_record(
            retried_after_success,
            provider="glm",
            task=task,
            phase=phase,
            expected_model="glm-model",
            attempt_policy=policy,
            formal_mode=True,
        )

    tampered_binding = copy.deepcopy(v2_record)
    tampered_binding["attempts"][-1]["response_content_sha256"] = "0" * 64
    with pytest.raises(analyze.AnalysisContractError, match="response.*binding|digest"):
        analyze._validate_record(
            tampered_binding,
            provider="glm",
            task=task,
            phase=phase,
            expected_model="glm-model",
            attempt_policy=policy,
            formal_mode=True,
        )
    tampered_parse = copy.deepcopy(v2_record)
    tampered_parse["parsed_output"]["rationale"] = "changed after observation"
    with pytest.raises(analyze.AnalysisContractError, match="raw|replay|parsed"):
        analyze._validate_record(
            tampered_parse,
            provider="glm",
            task=task,
            phase=phase,
            expected_model="glm-model",
            attempt_policy=policy,
            formal_mode=True,
        )

    falsely_excluded = _response_record(
        task,
        "glm",
        phase,
        answer="A",
        status="excluded_schema",
    )
    falsely_excluded["schema_version"] = "yher.llm_sim_v2.response_record.v2"
    falsely_excluded["attempts"][-1].update(bind_response_content(raw))
    with pytest.raises(
        analyze.AnalysisContractError,
        match="invalid.schema.*(replay|strict)|raw.*valid|exclusion",
    ):
        analyze._validate_record(
            falsely_excluded,
            provider="glm",
            task=task,
            phase=phase,
            expected_model="glm-model",
            attempt_policy=policy,
            formal_mode=True,
        )


def test_formal_loader_recomputes_provider_record_set_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    manifest_path = (
        output_base / RUN_ID / "main/provider_manifests/deepseek.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_set"]["record_count"] = 1
    _write_json_file(manifest_path, manifest)

    with pytest.raises(analyze.AnalysisContractError, match="record.set|record.*bytes"):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_formal_loader_reconciles_carried_cost_ledger_to_phase_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    carried_path = (
        repo
        / "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json"
    )
    carried = json.loads(carried_path.read_text(encoding="utf-8"))
    carried["known_cost_yuan"] = 1.0
    carried["total_accounted_cost_yuan"] = 1.0
    _rehash(carried, "carried_forward_cost_ledger_sha256")
    _write_json_file(carried_path, carried)

    with pytest.raises(
        analyze.AnalysisContractError, match="carried.*phase|carried.*ledger|run budget"
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def _refresh_manifest_counts(
    provider: str,
    tasks: list[dict[str, object]],
    records: dict[str, dict[str, dict[str, object]]],
    manifests: dict[str, dict[str, object]],
) -> None:
    _refresh_provider_manifest(provider, tasks, records, manifests)


def test_blind_schema_exclusion_is_strictly_above_half_not_at_half() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    primary = [
        task
        for task in tasks
        if task["condition"] == "blind" and task["is_stability_repeat"] is False
    ]
    assert len(primary) == 200
    for task in primary[:100]:
        records["tongyi"][str(task["task_id"])] = _response_record(
            task, "tongyi", phase, answer="A"
        )
    _refresh_manifest_counts("tongyi", tasks, records, manifests)
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )

    schema = result["blind"]["provider_schema"]["tongyi"]
    assert schema["invalid_schema_fraction"] == 0.5
    assert schema["strictly_above_half"] is False
    assert result["blind"]["excluded_providers"] == ["tongyi"]
    assert schema["exclusion_reasons"] == ["minimum_complete_clusters"]
    assert len(result["blind"]["agreement"]["pairs"]) == 10


def test_fewer_than_two_blind_providers_is_disclosed_as_not_estimable() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    primary = [
        task
        for task in tasks
        if task["condition"] == "blind" and task["is_stability_repeat"] is False
    ]
    for provider in PROVIDERS:
        for task in primary:
            task_id = str(task["task_id"])
            if task_id not in records[provider]:
                continue
            records[provider][task_id] = _response_record(
                task,
                provider,
                phase,
                status="excluded_schema",
            )
        _refresh_manifest_counts(provider, tasks, records, manifests)
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )

    blind = result["blind"]
    assert blind["eligible_providers"] == []
    assert blind["excluded_providers"] == list(PROVIDERS)
    assert blind["agreement"]["status"] == "not_estimable"
    assert blind["agreement"]["pairs"] == []
    assert blind["technical_or_schema_failure_rate"]["estimate"] is None
    assert blind["technical_or_schema_failure_rate"]["bootstrap"][
        "undefined_resamples"
    ] == 10_000
    assert blind["multi_provider_descriptive"]["unanimous_fraction"] is None


def test_provider_lifecycle_missing_classification_must_reconcile() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    manifests["deepseek"]["lifecycle"]["interrupted_count"] = 1  # type: ignore[index]
    with pytest.raises(AnalysisContractError, match="lifecycle.*class|interrupted"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_partial_provider_cannot_fabricate_breaker_exclusion() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    _refresh_provider_manifest(
        "deepseek",
        tasks,
        records,
        manifests,
        provider_lifecycle="excluded_repeated_failure",
    )
    with pytest.raises(AnalysisContractError, match="breaker|failure.*threshold"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_real_runner_final_batch_threshold_is_a_valid_closed_breaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from experiments.llm_sim.transport import ProviderNetworkError
    import experiments.llm_sim_v2.runner as runner_module

    class SequenceTransport:
        def __init__(self, responses: list[BaseException]) -> None:
            self.responses = list(responses)
            self.calls = 0

        def complete(self, **_kwargs: object) -> dict[str, object]:
            self.calls += 1
            if not self.responses:
                raise AssertionError("unexpected provider call")
            raise self.responses.pop(0)

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(repo_root)
    tasks = [
        task
        for task in runner_module.enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    ][:3]
    assert len(tasks) == 3
    transport = SequenceTransport([ProviderNetworkError() for _ in range(3)])
    runner = runner_module.V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=transport,
        budget=runner_module.BudgetLedger(
            soft_warning_yuan=300.0,
            hard_fuse_yuan=450.0,
        ),
        sleep=lambda _seconds: None,
        random_value=lambda: 0.5,
        clock=lambda: 1000.0,
    )
    runner.policy = replace(
        runner.policy,
        concurrency=1,
        max_attempts=1,
        failure_threshold=3,
    )

    summary = runner.run_tasks(tasks)

    assert transport.calls == 3
    assert summary["lifecycle"]["present_count"] == 3
    assert summary["lifecycle"]["missing_count"] == 0
    assert summary["lifecycle"]["breaker_skipped_count"] == 0
    assert summary["breaker"]["status"] == "closed"
    assert summary["breaker"]["consecutive_failures"] == 3

    manifest = json.loads(
        (runner.store.root / "provider_manifests/deepseek.json").read_bytes()
    )
    records: dict[str, dict[str, object]] = {}
    for path in sorted((runner.store.root / "records/deepseek").glob("*.json")):
        record = json.loads(path.read_bytes())
        records[str(record["task_id"])] = record
    lifecycle = manifest["lifecycle"]

    from experiments.llm_sim_v2.analyze import _validate_breaker_state

    _validate_breaker_state(
        provider="deepseek",
        breaker=manifest["breaker"],
        breaker_ids=set(lifecycle["breaker_skipped_task_ids"]),
        expected_ids=lifecycle["expected_task_ids"],
        present_ids=lifecycle["present_task_ids"],
        missing_ids=lifecycle["missing_task_ids"],
        records=records,
        attempt_policy={
            "failure_threshold": runner.policy.failure_threshold,
            "concurrency": runner.policy.concurrency,
            "cooldown_seconds": runner.policy.cooldown_seconds,
        },
        formal_mode=True,
    )


def test_publication_figure_guard_rejects_overlapping_footer_and_legend() -> None:
    from matplotlib import pyplot as plt

    from experiments.llm_sim_v2.analyze import (
        AnalysisContractError,
        _assert_artists_do_not_overlap,
    )

    figure, axis = plt.subplots(figsize=(4, 3))
    axis.plot([0, 1], [0, 1], label="series")
    legend = figure.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02))
    footer = figure.text(0.5, 0.02, "footer", ha="center")
    try:
        with pytest.raises(AnalysisContractError, match="overlap"):
            _assert_artists_do_not_overlap(figure, legend, footer)
        footer.set_position((0.5, 0.18))
        _assert_artists_do_not_overlap(figure, legend, footer)
    finally:
        plt.close(figure)


def test_coordinated_source_rehash_is_rejected_against_active_contract() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    active_proof = _active_contract_proof(runtime, phase)
    phase["source"] = {"source_set_sha256": "9" * 64}
    _rehash(phase, "phase_provenance_sha256")
    for provider in PROVIDERS:
        binding = _binding(phase)
        manifests[provider]["provenance"] = binding
        for record in records[provider].values():
            record["provenance"] = binding

    with pytest.raises(AnalysisContractError, match="active|contract|source|provenance"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=active_proof,
        )


def test_record_attempt_costs_must_reconcile_before_metrics() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    record = next(iter(records["glm"].values()))
    record["attempts"] = [
        {
            "attempt": 1,
            "status": "response",
            "request_max_tokens": 512,
            "latency_ms": 1.0,
            "model_returned": "glm-model",
            "finish_reason": "stop",
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "cost_yuan": 123.45,
            "cost_known": True,
            "billing_ambiguity": False,
            "cost_reserve_yuan": 0.0,
        }
    ]
    record["known_cost_yuan"] = 0.0
    record["unknown_cost_reserve_yuan"] = 0.0
    record["cost_yuan"] = 0.0

    with pytest.raises(AnalysisContractError, match="cost|attempt|reconcile"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_model_and_outcome_fields_are_recomputed_from_runtime_task() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    record = next(
        row
        for row in records["kimi"].values()
        if row["status"] == "complete" and row["parsed_output"]["answer"] is not None
    )
    record["requested_model"] = "wrong-requested-model"
    record["model_id"] = "wrong-returned-model"
    record["outcomes"]["is_correct"] = not record["outcomes"]["is_correct"]

    with pytest.raises(AnalysisContractError, match="model|outcome|correct"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_attempt_returned_model_must_reconcile_to_record_model() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    record = next(
        row for row in records["glm"].values() if row["status"] == "complete"
    )
    record["attempts"][0]["model_returned"] = "attempt-level-drift"

    with pytest.raises(AnalysisContractError, match="attempt.*model|model.*attempt|drift"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_loader_requires_cumulative_run_budget_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    (output_base / RUN_ID / "run_budget_ledger.json").unlink()

    with pytest.raises(analyze.AnalysisContractError, match="run budget ledger|cost"):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


@pytest.mark.parametrize(
    ("field", "expected", "tampered"),
    [
        ("target_node", "node-a", "node-b"),
        ("family_id", "family-a", "family-b"),
        ("attempt_id", "primary", "repeat"),
        ("logical_key", "logical-a", "logical-b"),
        ("message_sha256", "1" * 64, "2" * 64),
        ("wire_message_sha256", "3" * 64, "4" * 64),
    ],
)
def test_full_runtime_task_identity_is_reconciled(
    field: str, expected: str, tampered: str
) -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    task = tasks[0]
    task[field] = expected
    records["glm"][str(task["task_id"])][field] = tampered

    with pytest.raises(AnalysisContractError, match="task identity|logical|message|family|target|attempt"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_provider_manifest_budget_must_reconcile_to_attempt_ledgers() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    manifests["glm"]["budget"] = {
        "total_cost_yuan": 999.0,
        "provider_record_known_cost_yuan": 999.0,
        "provider_record_unknown_reserve_yuan": 0.0,
        "provider_record_accounted_cost_yuan": 999.0,
        "soft_warning_triggered": True,
        "hard_fuse_triggered": True,
    }

    with pytest.raises(AnalysisContractError, match="provider.*budget|cost"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_loader_rejects_run_ledger_record_total_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    ledger_path = output_base / RUN_ID / "run_budget_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["immutable_record_cost_yuan"] = 999.0
    _write_json_file(ledger_path, ledger)

    with pytest.raises(analyze.AnalysisContractError, match="run budget ledger|record cost|reconcile"):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_loader_rejects_tampered_prior_cost_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    prior_path = repo / "experiments/llm_sim_v2/prior_cost_ledger.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior["prior_cost_ledger_sha256"] = "0" * 64
    _write_json_file(prior_path, prior)

    with pytest.raises(analyze.AnalysisContractError, match="prior cost ledger|digest|cost"):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_analysis_emits_provider_phase_cost_table() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )

    assert result["cost_accounting"]["currency"] == "CNY"
    rows = result["cost_accounting"]["provider_phase"]
    assert {(row["phase"], row["provider"]) for row in rows} == {
        ("main", provider) for provider in PROVIDERS
    }
    assert all(
        {
            "requests",
            "responses",
            "retries",
            "input_tokens",
            "output_tokens",
            "known_cost_yuan",
            "unknown_cost_reserve_yuan",
            "accounted_cost_yuan",
        }
        <= set(row)
        for row in rows
    )


def test_unavailable_provider_is_disclosed_but_ineligible_and_not_nc_stable() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    records["glm"].clear()
    _refresh_provider_manifest(
        "glm", tasks, records, manifests, provider_lifecycle="unavailable"
    )
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )

    glm_lifecycle = next(
        row for row in result["provider_lifecycle"] if row["provider"] == "glm"
    )
    assert glm_lifecycle["provider_lifecycle"] == "unavailable"
    assert glm_lifecycle["controlled_eligible"] is False
    assert glm_lifecycle["blind_eligible"] is False
    assert "glm" not in result["controlled"]["eligible_providers"]
    assert "glm" not in result["blind"]["eligible_providers"]
    glm_stability = next(
        row for row in result["blind"]["stability"] if row["provider"] == "glm"
    )
    assert glm_stability["status"] == "not_estimable_ineligible_lane"
    assert glm_stability["answer_agreement"] is None
    assert glm_stability["answer_agreement_denominator"] == 0


@pytest.mark.parametrize(
    "declared_lifecycle",
    ["unavailable", "interrupted", "excluded_repeated_failure", "fuse_open"],
)
def test_provider_lifecycle_is_recomputed_not_trusted_from_manifest_or_history(
    declared_lifecycle: str,
) -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    _refresh_provider_manifest(
        "glm",
        tasks,
        records,
        manifests,
        provider_lifecycle=declared_lifecycle,
    )
    manifests["glm"]["lifecycle_history"] = [
        {
            "event_index": 0,
            "provider_lifecycle": declared_lifecycle,
            "path": "provider_lifecycle/glm/0000-synthetic.json",
        }
    ]

    with pytest.raises(
        AnalysisContractError,
        match="lifecycle.*derived|lifecycle.*record|unavailable|interrupted|fuse|breaker",
    ):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_condition_eligibility_boundary_is_45_complete_clusters() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    def result_with_complete_clusters(complete_clusters: int) -> dict[str, object]:
        tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
        removed = set(PERSONAS[complete_clusters:])
        for task in tasks:
            if task["condition"] == "controlled" and task["persona_id"] in removed:
                records["minimax"].pop(str(task["task_id"]), None)
        _refresh_provider_manifest("minimax", tasks, records, manifests)
        return analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )

    at_boundary = result_with_complete_clusters(45)
    below_boundary = result_with_complete_clusters(44)
    assert "minimax" in at_boundary["controlled"]["eligible_providers"]
    assert "minimax" not in below_boundary["controlled"]["eligible_providers"]
    row = next(
        item
        for item in below_boundary["provider_lifecycle"]
        if item["provider"] == "minimax"
    )
    assert row["controlled_complete_cluster_count"] == 44
    assert "minimum_complete_clusters" in row["controlled_exclusion_reasons"]


def test_model_drift_excludes_lane_even_when_cluster_minimum_is_met() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    task = next(task for task in tasks if task["condition"] == "controlled")
    record = records["kimi"][str(task["task_id"])]
    record.update(
        {
            "model_id": "drifted-model",
            "status": "excluded_model_drift",
            "error": "returned_model_drift",
            "parsed_output": None,
            "outcomes": {
                "is_correct": None,
                "target_option_hit": None,
                "manipulation_compliance": None,
            },
        }
    )
    record["attempts"][0]["model_returned"] = "drifted-model"
    _refresh_provider_manifest("kimi", tasks, records, manifests)
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )

    assert "kimi" not in result["controlled"]["eligible_providers"]
    row = next(
        item for item in result["provider_lifecycle"] if item["provider"] == "kimi"
    )
    assert "model_drift" in row["controlled_exclusion_reasons"]


def test_sparse_mapping_incorrect_denominator_excludes_abstention() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    task = next(
        task
        for task in tasks
        if task["condition"] == "controlled"
        and task["response_arm"] == "deficit"
        and task["target_option"] is not None
    )
    records["glm"][str(task["task_id"])] = _response_record(
        task, "glm", phase, answer=None
    )
    _refresh_provider_manifest("glm", tasks, records, manifests)
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )
    row = next(
        item
        for item in result["sparse_mapping_descriptive"]["by_provider_and_arm"]
        if item["provider"] == "glm" and item["response_arm"] == "deficit"
    )
    expected = sum(
        record["status"] == "complete"
        and record["parsed_output"]["answer"] is not None
        and record["outcomes"]["is_correct"] is False
        for candidate in tasks
        if candidate["condition"] == "controlled"
        and candidate["response_arm"] == "deficit"
        and candidate["target_option"] is not None
        and (record := records["glm"].get(str(candidate["task_id"]))) is not None
    )
    assert row["incorrect_answer_denominator"] == expected


def test_provider_effects_have_cluster_cis_and_honest_denominator_vector() -> None:
    from experiments.llm_sim_v2.analyze import analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    for task in tasks:
        if (
            task["condition"] == "controlled"
            and task["persona_id"] == "persona-01"
            and task["response_arm"] == "deficit"
        ):
            records["glm"][str(task["task_id"])] = _response_record(
                task, "glm", phase, answer=None
            )
    _refresh_provider_manifest("glm", tasks, records, manifests)
    result = analyze_dataset(
        expected_tasks=tasks,
        runtime_manifest=runtime,
        phase_provenance=phase,
        records_by_provider=records,
        provider_manifests=manifests,
        mapping_manifest=mapping,
        active_contract_proof=_active_contract_proof(runtime, phase),
    )
    effect = next(
        row
        for row in result["controlled"]["paired_effects"]
        if row["metric_id"] == "conditional_answer_accuracy"
    )
    glm = next(row for row in effect["by_provider"] if row["provider"] == "glm")
    assert glm["paired_persona_denominator"] == 49
    assert glm["ci95"] is not None
    assert glm["bootstrap"]["resamples"] == 10_000
    assert effect["paired_persona_denominators"]["glm"] == 49
    assert effect["paired_persona_denominator_range"] == [49, 50]


def test_unknown_attempt_needs_user_reason_cannot_be_erased() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    record = next(iter(records["doubao"].values()))
    record["attempts"] = [
        {
            "attempt": 1,
            "status": "failed",
            "error_category": "timeout",
            "latency_ms": 120000.0,
            "cost_yuan": None,
            "cost_known": False,
            "billing_ambiguity": True,
            "cost_reserve_yuan": 10.0,
        }
    ]
    record["status"] = "technical_failure"
    record["error"] = "timeout"
    record["parsed_output"] = None
    record["outcomes"] = {
        "is_correct": None,
        "target_option_hit": None,
        "manipulation_compliance": None,
    }
    record["model_id"] = None
    record["known_cost_yuan"] = 0.0
    record["unknown_cost_reserve_yuan"] = 10.0
    record["cost_yuan"] = 10.0
    record["has_unknown_cost_attempts"] = True
    record["needs_user"] = True
    record["needs_user_reasons"] = []
    _refresh_provider_manifest("doubao", tasks, records, manifests)
    manifests["doubao"]["budget"].update(  # type: ignore[union-attr]
        {
            "provider_record_known_cost_yuan": 0.0,
            "provider_record_unknown_reserve_yuan": 10.0,
            "provider_record_accounted_cost_yuan": 10.0,
        }
    )

    with pytest.raises(AnalysisContractError, match="needs_user|billing|reserve"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_provider_manifest_needs_user_summary_must_reconcile() -> None:
    from experiments.llm_sim_v2.analyze import AnalysisContractError, analyze_dataset

    tasks, runtime, phase, records, manifests, mapping = _synthetic_main_fixture()
    record = next(iter(records["doubao"].values()))
    record.update(
        {
            "attempts": [
                {
                    "attempt": 1,
                    "status": "failed",
                    "error_category": "timeout",
                    "latency_ms": 120000.0,
                    "cost_yuan": None,
                    "cost_known": False,
                    "billing_ambiguity": True,
                    "cost_reserve_yuan": 10.0,
                }
            ],
            "status": "technical_failure",
            "error": "timeout",
            "parsed_output": None,
            "outcomes": {
                "is_correct": None,
                "target_option_hit": None,
                "manipulation_compliance": None,
            },
            "model_id": None,
            "known_cost_yuan": 0.0,
            "unknown_cost_reserve_yuan": 10.0,
            "cost_yuan": 10.0,
            "has_unknown_cost_attempts": True,
            "needs_user": True,
            "needs_user_reasons": ["unknown_provider_billing_reserved"],
        }
    )
    _refresh_provider_manifest("doubao", tasks, records, manifests)
    manifests["doubao"]["budget"].update(  # type: ignore[union-attr]
        {
            "provider_record_known_cost_yuan": 0.0,
            "provider_record_unknown_reserve_yuan": 10.0,
            "provider_record_accounted_cost_yuan": 10.0,
        }
    )
    manifests["doubao"]["needs_user"] = {
        "required": False,
        "reason": None,
        "record_count": 0,
        "record_task_ids": [],
        "unknown_cost_attempt_count": 0,
    }

    with pytest.raises(AnalysisContractError, match="provider.*needs_user|billing|unknown"):
        analyze_dataset(
            expected_tasks=tasks,
            runtime_manifest=runtime,
            phase_provenance=phase,
            records_by_provider=records,
            provider_manifests=manifests,
            mapping_manifest=mapping,
            active_contract_proof=_active_contract_proof(runtime, phase),
        )


def test_loader_rejects_tampered_lifecycle_history_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    manifest_path = output_base / RUN_ID / "main/provider_manifests/glm.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_path = output_base / RUN_ID / "main" / manifest["lifecycle_history"][0]["path"]
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["provider_lifecycle"] = "complete"
    _write_json_file(event_path, event)

    with pytest.raises(
        analyze.AnalysisContractError,
        match="lifecycle.*history|event|digest|phase evidence receipt.*stale",
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def test_pilot_cost_bytes_are_bound_outside_metric_input_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.llm_sim_v2.analyze as analyze

    output_base, repo, tasks = _filesystem_main_fixture(tmp_path)
    monkeypatch.setattr(analyze, "_load_expected_tasks", lambda _repo, _runtime: tasks)
    monkeypatch.setattr(
        analyze,
        "_validate_active_contract_inputs",
        lambda _repo, *, runtime_manifest, phase_provenance: _active_contract_proof(
            dict(runtime_manifest), dict(phase_provenance)
        ),
    )
    from experiments.llm_sim_v2.evidence import (
        bind_response_content,
        build_provider_record_set,
    )

    pilot_record = {
        "schema_version": "yher.llm_sim_v2.response_record.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "analysis_population": "pilot",
        "provider": "deepseek",
        "task_id": "pilot-cost-task",
        "attempts": [
            {
                "attempt": 1,
                "status": "response",
                "model_returned": "deepseek-model",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "cost_yuan": 0.0,
                "cost_known": True,
                "billing_ambiguity": False,
                "cost_reserve_yuan": 0.0,
                **bind_response_content("{}"),
            }
        ],
        "retry_count": 0,
        "known_cost_yuan": 0.0,
        "unknown_cost_reserve_yuan": 0.0,
        "cost_yuan": 0.0,
        "has_unknown_cost_attempts": False,
        "needs_user": False,
        "needs_user_reasons": [],
    }
    _write_json_file(
        output_base
        / RUN_ID
        / "pilot/records/deepseek/pilot-cost-task.json",
        pilot_record,
    )
    pilot_provider_manifest = {
        "schema_version": "yher.llm_sim_v2.provider_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "analysis_population": "pilot",
        "provider": "deepseek",
        "record_count": 1,
        "lifecycle": {
            "expected_task_ids": ["pilot-cost-task"],
        },
        "budget": {
            "provider_record_known_cost_yuan": 0.0,
            "provider_record_unknown_reserve_yuan": 0.0,
            "provider_record_accounted_cost_yuan": 0.0,
        },
        "needs_user": {
            "required": False,
            "reason": None,
            "record_count": 0,
            "record_task_ids": [],
            "unknown_cost_attempt_count": 0,
        },
    }
    pilot_provider_manifest["record_set"] = build_provider_record_set(
        output_base / RUN_ID / "pilot",
        provider="deepseek",
        expected_task_ids=["pilot-cost-task"],
    )
    _write_json_file(
        output_base
        / RUN_ID
        / "pilot/provider_manifests/deepseek.json",
        pilot_provider_manifest,
    )
    loaded = analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)

    metric_paths = {
        row["path"] for row in loaded["input_artifact_manifest"]["files"]
    }
    assert not any(
        path.startswith(("pilot/records/", "pilot/provider_manifests/"))
        for path in metric_paths
    )
    assert any("carried_forward_cost" in path for path in metric_paths)
    cost_manifest = loaded["cost_accounting"][
        "cost_reconciliation_artifact_manifest"
    ]
    assert cost_manifest["file_count"] == 2
    by_type = {row["artifact_type"]: row for row in cost_manifest["files"]}
    assert set(by_type) == {"provider_manifest", "response_record"}
    assert by_type["response_record"]["phase"] == "pilot"
    assert by_type["response_record"]["sha256"] == hashlib.sha256(
        (
            output_base
            / RUN_ID
            / "pilot/records/deepseek/pilot-cost-task.json"
        ).read_bytes()
    ).hexdigest()
    assert by_type["provider_manifest"]["sha256"] == hashlib.sha256(
        (
            output_base
            / RUN_ID
            / "pilot/provider_manifests/deepseek.json"
        ).read_bytes()
    ).hexdigest()

    pilot_record["schema_version"] = "yher.llm_sim_v2.response_record.v1"
    _write_json_file(
        output_base
        / RUN_ID
        / "pilot/records/deepseek/pilot-cost-task.json",
        pilot_record,
    )
    with pytest.raises(analyze.AnalysisContractError, match="pilot.*identity|v2"):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)
    pilot_record["schema_version"] = "yher.llm_sim_v2.response_record.v2"
    _write_json_file(
        output_base
        / RUN_ID
        / "pilot/records/deepseek/pilot-cost-task.json",
        pilot_record,
    )

    pilot_provider_manifest["budget"]["provider_record_known_cost_yuan"] = 1.0  # type: ignore[index]
    _write_json_file(
        output_base
        / RUN_ID
        / "pilot/provider_manifests/deepseek.json",
        pilot_provider_manifest,
    )
    with pytest.raises(
        analyze.AnalysisContractError,
        match="pilot.*provider.*budget|provider.*cost.*reconcile",
    ):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)

    pilot_provider_manifest["budget"]["provider_record_known_cost_yuan"] = "invalid"  # type: ignore[index]
    _write_json_file(
        output_base
        / RUN_ID
        / "pilot/provider_manifests/deepseek.json",
        pilot_provider_manifest,
    )
    with pytest.raises(analyze.AnalysisContractError, match="pilot.*provider.*budget"):
        analyze.load_analysis_inputs(output_base=output_base, repo_root=repo)


def _judge_candidate(index: int, *, stratum: str) -> dict[str, object]:
    public_question = {
        "kind": "mcq",
        "stem_blocks": [],
        "stem_text": f"Question {index}",
        "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "difficulty": 0.5,
        "nodes": [],
        "source_label": "synthetic-test",
    }
    return {
        "candidate_identity": f"provider-{index % 6}|task-{index}",
        "subject_id": f"subject-{index // 6}",
        "stratum": stratum,
        "public_question": public_question,
        "model_output": {
            "simulated": True,
            "answer": "A" if index % 2 == 0 else "B",
            "rationale": f"candidate rationale {index}",
            "abstain": False,
        },
        "persona": {
            "persona_id": f"private-persona-{index}",
            "pair_id": "private-pair",
            "row_id": "private-row",
            "target_node": "private-node",
            "deficit_condition": "deficit",
            "failure_id": "private-failure",
            "failure_cause": "private-cause",
            "failure_symptom": "private-symptom",
            "observable_error_policy": {"private": "policy"},
        },
        "item": {
            "item_id": f"item-{index}",
            "public_question": public_question,
            "options": public_question["options"],
            "private_correct_option": "A",
        },
    }


def test_judge_selection_uses_80_40_quota_fill_and_identical_case_bytes() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    assert hasattr(analyze, "build_judge_case_manifest"), "judge selector is missing"
    assert hasattr(analyze, "judge_input_bytes"), "judge byte renderer is missing"
    candidates = [
        *(_judge_candidate(index, stratum="disagreement") for index in range(100)),
        *(_judge_candidate(1000 + index, stratum="agreement") for index in range(100)),
    ]
    manifest = analyze.build_judge_case_manifest(
        candidates, frozen_leakage_lexicon=()
    )
    reversed_manifest = analyze.build_judge_case_manifest(
        list(reversed(candidates)), frozen_leakage_lexicon=()
    )
    assert reversed_manifest == manifest
    assert manifest["selected_count"] == 120
    assert manifest["selected_stratum_counts"] == {
        "disagreement": 80,
        "agreement": 40,
    }
    claude_bytes = analyze.judge_input_bytes(manifest)
    gpt_bytes = analyze.judge_input_bytes(manifest)
    assert claude_bytes == gpt_bytes
    assert hashlib.sha256(claude_bytes).hexdigest() == manifest["shared_input_sha256"]
    text = claude_bytes.decode("utf-8")
    for forbidden in (
        "provider-",
        "private-persona",
        "private_correct_option",
        "target_node",
        "failure_id",
    ):
        assert forbidden not in text

    filled = analyze.build_judge_case_manifest(
        [
            *(_judge_candidate(index, stratum="disagreement") for index in range(10)),
            *(
                _judge_candidate(2000 + index, stratum="agreement")
                for index in range(200)
            ),
        ],
        frozen_leakage_lexicon=(),
    )
    assert filled["selected_stratum_counts"] == {
        "disagreement": 10,
        "agreement": 110,
    }
    assert filled["cross_stratum_fill"] == 70


def test_judge_selection_exports_only_the_strict_question_whitelist() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    candidate = _judge_candidate(7, stratum="disagreement")
    candidate["public_question"]["nodes"] = ["private-node"]  # type: ignore[index]
    candidate["public_question"]["difficulty"] = 0.91  # type: ignore[index]
    candidate["public_question"]["source_label"] = "private-source"  # type: ignore[index]
    manifest = analyze.build_judge_case_manifest(
        [candidate], frozen_leakage_lexicon=()
    )

    assert manifest["schema_version"] == "yher.llm_sim_v2.judge_case_manifest.v2"
    assert manifest["question_field_whitelist"] == [
        "kind",
        "options",
        "stem_blocks",
        "stem_text",
    ]
    assert manifest["target_metadata_exported"] is False
    amendment = manifest["judge_amendment"]
    assert amendment["path"] == (
        "experiments/llm_sim_v2/judge_amendment_20260716.md"
    )
    assert len(amendment["sha256"]) == 64
    assert amendment["size"] > 0
    exported = analyze.judge_input_bytes(manifest).decode("utf-8")
    for forbidden in (
        "private-node",
        "private-source",
        '"nodes"',
        '"difficulty"',
        '"source_label"',
    ):
        assert forbidden not in exported
    for visible in ("Question 7", "one"):
        assert visible in exported


def test_judge_selection_excludes_question_text_containing_the_target_label() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    candidate = _judge_candidate(8, stratum="agreement")
    candidate["public_question"]["stem_text"] = "Question about private-node"  # type: ignore[index]
    candidate["item"]["public_question"] = candidate["public_question"]  # type: ignore[index]
    manifest = analyze.build_judge_case_manifest(
        [candidate], frozen_leakage_lexicon=()
    )

    assert manifest["selected_count"] == 0
    assert manifest["target_label_scan"]["excluded_candidate_count"] == 1
    assert manifest["target_label_scan"]["final_serialized_hit_count"] == 0
    assert "private-node" not in analyze.judge_input_bytes(manifest).decode("utf-8")


def test_judge_selection_excludes_model_output_containing_the_target_label() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    candidate = _judge_candidate(9, stratum="agreement")
    candidate["model_output"]["rationale"] = "This suggests private-node."  # type: ignore[index]
    manifest = analyze.build_judge_case_manifest(
        [candidate], frozen_leakage_lexicon=()
    )

    assert manifest["selected_count"] == 0
    assert manifest["target_label_scan"]["excluded_candidate_count"] == 1
    assert manifest["target_label_scan"]["final_serialized_hit_count"] == 0
    assert "private-node" not in analyze.judge_input_bytes(manifest).decode("utf-8")


def test_judge_selection_rejects_nested_or_encoded_private_fields() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    assert hasattr(analyze, "build_judge_case_manifest"), "judge selector is missing"
    nested = _judge_candidate(1, stratum="disagreement")
    nested["model_output"]["rationale"] = json.dumps(  # type: ignore[index]
        {"target_option": "B"}
    )
    with pytest.raises(analyze.AnalysisContractError, match="judge|leak|target"):
        analyze.build_judge_case_manifest([nested], frozen_leakage_lexicon=())


def test_judge_selection_excludes_and_discloses_provider_identity_without_redaction() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    identifying = _judge_candidate(1, stratum="disagreement")
    identifying["model_output"]["rationale"] = (  # type: ignore[index]
        "I am DeepSeek-V4-Pro and I choose B."
    )
    clean = _judge_candidate(2, stratum="agreement")
    clean_rationale = str(clean["model_output"]["rationale"])  # type: ignore[index]
    manifest = analyze.build_judge_case_manifest(
        [identifying, clean],
        frozen_leakage_lexicon=(),
        provider_identity_terms=("DeepSeek", "deepseek-v4-pro", "deepseek-chat"),
    )

    assert manifest["provider_identity_exported"] is False
    assert manifest["provider_identity_scan"]["excluded_candidate_count"] == 1
    assert manifest["provider_identity_scan"]["final_serialized_hit_count"] == 0
    assert len(
        manifest["provider_identity_scan"]["excluded_candidate_identity_sha256"]
    ) == 1
    exported = analyze.judge_input_bytes(manifest).decode("utf-8")
    assert "deepseek" not in exported.casefold()
    assert clean_rationale in exported
    assert "I am" not in exported


def _judge_result_manifest(
    case_manifest: dict[str, object],
    judge: str,
    *,
    disagree_case: str | None = None,
    error_category: str = "chemistry_reasoning",
    omit_error_category_case: str | None = None,
) -> dict[str, object]:
    results = []
    for case in case_manifest["cases"]:  # type: ignore[union-attr]
        case_id = str(case["case_id"])
        output = {
            "label": (
                "inconsistent" if case_id == disagree_case else "consistent"
            ),
            "error_category": error_category,
            "rationale": f"{judge} rationale",
            "simulated": True,
        }
        if case_id == omit_error_category_case:
            output.pop("error_category")
        results.append({"case_id": case_id, "output": output})
    manifest: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.judge_result_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "judge": judge,
        "case_manifest_sha256": case_manifest["case_manifest_sha256"],
        "results": results,
    }
    manifest["judge_result_manifest_sha256"] = _sha(manifest)
    return manifest


def test_judge_ingestion_rejects_legacy_unreceipted_result_manifest() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    cases = analyze.build_judge_case_manifest(
        [_judge_candidate(index, stratum="disagreement") for index in range(2)],
        frozen_leakage_lexicon=(),
    )
    claude = _judge_result_manifest(cases, "claude")
    with pytest.raises(
        analyze.AnalysisContractError,
        match="judge result|execution|receipt|hash",
    ):
        analyze.ingest_judge_results(
            cases, {"claude": claude, "gpt": None}
        )


def test_judge_ingestion_rejects_rehashed_protocol_or_amendment_drift() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    cases = analyze.build_judge_case_manifest(
        [_judge_candidate(1, stratum="agreement")],
        frozen_leakage_lexicon=(),
    )
    for field in ("judge_protocol", "judge_amendment"):
        drifted = copy.deepcopy(cases)
        drifted[field]["simulated" if field == "judge_protocol" else "size"] = (  # type: ignore[index]
            False if field == "judge_protocol" else 1
        )
        _rehash(drifted, "case_manifest_sha256")
        with pytest.raises(
            analyze.AnalysisContractError,
            match="case manifest|input bytes|drift",
        ):
            analyze.ingest_judge_results(
                drifted, {"claude": None, "gpt": None}
            )


def test_judge_ingestion_rejects_case_or_result_manifest_hash_drift() -> None:
    import experiments.llm_sim_v2.analyze as analyze

    assert hasattr(analyze, "build_judge_case_manifest"), "judge selector is missing"
    assert hasattr(analyze, "ingest_judge_results"), "judge ingestion is missing"
    cases = analyze.build_judge_case_manifest(
        [_judge_candidate(1, stratum="agreement")],
        frozen_leakage_lexicon=(),
    )
    result = _judge_result_manifest(cases, "claude")
    result["case_manifest_sha256"] = "0" * 64
    _rehash(result, "judge_result_manifest_sha256")
    with pytest.raises(analyze.AnalysisContractError, match="case manifest|judge result|hash"):
        analyze.ingest_judge_results(cases, {"claude": result, "gpt": None})
