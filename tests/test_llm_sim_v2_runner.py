"""Runtime contracts for the frozen Persona v2 provider collection."""

from __future__ import annotations

import copy
from email.utils import formatdate
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = REPO_ROOT / "experiments/llm_sim_v2/frozen_v0"


def _response(
    answer: str | None = "A",
    *,
    blind: bool = False,
    returned_model: str = "deepseek-v4-pro",
    cost: float = 0.01,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "simulated": True,
        "answer": answer,
        "rationale": "short synthetic response",
    }
    if blind:
        payload["abstain"] = answer is None
    return {
        "content": json.dumps(payload),
        "model_returned": returned_model,
        "finish_reason": "stop",
        "reasoning_tokens": 0,
        "request_max_tokens": 512,
        "latency_ms": 1.0,
        "http_status": 200,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "cost_yuan": cost,
    }


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract_with_revision_one():
    from dataclasses import replace

    from experiments.llm_sim_v2.runner import load_runtime_contract

    contract = load_runtime_contract(REPO_ROOT)
    ledger = copy.deepcopy(dict(contract.prompt_ledger))
    revision = copy.deepcopy(ledger["revisions"][0])
    revision.update(
        {
            "revision": 1,
            "parent_revision": 0,
            "reason": "pilot_engineering_failure_single_rewrite",
            "calibration_rewrite_required": True,
            "committed_at_utc": "2026-07-15T07:00:00Z",
            "observed_row_count": 0,
        }
    )
    revision.pop("prompt_contract_sha256", None)
    revision["prompt_contract_sha256"] = _canonical_sha(revision)
    ledger["current_revision"] = 1
    ledger["revisions"].append(revision)
    ledger.pop("prompt_ledger_sha256", None)
    ledger["prompt_ledger_sha256"] = _canonical_sha(ledger)
    return replace(contract, prompt_ledger=ledger), revision


class SequenceTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected provider call")
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return dict(result)


def test_frozen_task_enumerator_uses_paired_rows_and_exact_panel_support():
    from experiments.llm_sim_v2.runner import load_runtime_contract, enumerate_tasks

    contract = load_runtime_contract(REPO_ROOT)
    pilot = enumerate_tasks(contract, phase="pilot")
    main = enumerate_tasks(contract, phase="main")

    assert len({task.persona_id for task in pilot}) == 5
    assert len({task.persona_id for task in main}) == 50
    assert {task.response_arm for task in main} == {"deficit", "control"}
    assert sum(task.condition == "controlled" for task in main) == 400
    assert sum(task.condition == "blind" and not task.is_stability_repeat for task in main) == 1108
    assert sum(task.is_stability_repeat for task in main) == 20
    assert sum(task.condition == "controlled" for task in pilot) == 40
    assert sum(task.is_stability_repeat for task in pilot) == 10
    assert all(task.prompt_revision == 0 for task in main + pilot)
    assert len({task.logical_key for task in main}) == len(main)


@pytest.mark.parametrize(
    ("condition", "content", "expected"),
    [
        (
            "controlled",
            '{"simulated":true,"answer":"b","rationale":"ok"}',
            {"simulated": True, "answer": "B", "rationale": "ok"},
        ),
        (
            "blind",
            '{"simulated":true,"answer":null,"rationale":"unsure","abstain":true}',
            {
                "simulated": True,
                "answer": None,
                "rationale": "unsure",
                "abstain": True,
            },
        ),
    ],
)
def test_strict_response_parser_accepts_only_the_frozen_schema(
    condition: str, content: str, expected: dict[str, Any]
):
    from experiments.llm_sim_v2.runner import parse_provider_output

    assert parse_provider_output(
        content, condition=condition, option_keys={"A", "B", "C", "D"}
    ) == expected


@pytest.mark.parametrize(
    ("condition", "content"),
    [
        ("controlled", "```json\n{}\n```"),
        ("controlled", '{"simulated":false,"answer":"A","rationale":"x"}'),
        ("controlled", '{"simulated":true,"answer":"E","rationale":"x"}'),
        ("controlled", '{"simulated":true,"answer":"A","rationale":"x","extra":1}'),
        ("blind", '{"simulated":true,"answer":null,"rationale":"x","abstain":false}'),
        ("blind", '{"simulated":true,"answer":"A","rationale":"x"}'),
    ],
)
def test_strict_response_parser_rejects_schema_drift(condition: str, content: str):
    from experiments.llm_sim_v2.runner import InvalidProviderOutput, parse_provider_output

    with pytest.raises(InvalidProviderOutput):
        parse_provider_output(
            content, condition=condition, option_keys={"A", "B", "C", "D"}
        )


def test_runner_computes_sparse_compliance_without_provider_self_rating():
    from experiments.llm_sim_v2.runner import compute_outcomes

    deficit = compute_outcomes(
        condition="controlled",
        response_arm="deficit",
        answer="B",
        abstain=False,
        correct_option="A",
        target_option="B",
    )
    control = compute_outcomes(
        condition="controlled",
        response_arm="control",
        answer="A",
        abstain=False,
        correct_option="A",
        target_option="B",
    )
    unmapped = compute_outcomes(
        condition="controlled",
        response_arm="deficit",
        answer="C",
        abstain=False,
        correct_option="A",
        target_option=None,
    )

    assert deficit == {
        "is_correct": False,
        "target_option_hit": True,
        "manipulation_compliance": True,
    }
    assert control == {
        "is_correct": True,
        "target_option_hit": False,
        "manipulation_compliance": True,
    }
    assert unmapped["target_option_hit"] is None
    assert unmapped["manipulation_compliance"] is None


def test_execute_task_retries_same_messages_and_records_every_attempt(tmp_path: Path):
    from experiments.llm_sim.transport import ProviderNetworkError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        execute_task,
        load_runtime_contract,
        enumerate_tasks,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(task for task in enumerate_tasks(contract, phase="pilot") if task.condition == "controlled")
    transport = SequenceTransport([ProviderNetworkError(), _response(answer=task.correct_option)])
    budget = BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0)

    record = execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=transport,
        policy=contract.provider_policy("deepseek"),
        budget=budget,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert record["status"] == "complete"
    assert record["error"] is None
    assert len(record["attempts"]) == 2
    assert record["retry_count"] == 1
    assert transport.calls[0]["messages"] == transport.calls[1]["messages"]
    assert transport.calls[0]["model"] == transport.calls[1]["model"]
    assert record["message_sha256"] == task.message_sha256
    assert record["prompt_contract_sha256"] == task.prompt_contract_sha256
    assert record["prompt_contract_sha256"] != task.wire_message_sha256
    assert budget.total_cost_yuan == pytest.approx(10.01)
    assert record["known_cost_yuan"] == pytest.approx(0.01)
    assert record["unknown_cost_reserve_yuan"] == pytest.approx(10.0)
    assert record["needs_user"] is True


def test_truncated_attempt_usage_and_cost_are_counted_before_retry():
    from experiments.llm_sim.transport import ProviderTruncatedResponseError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        execute_task,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    truncated = ProviderTruncatedResponseError(
        finish_reason="length",
        request_max_tokens=1024,
        reasoning_tokens=900,
        returned_model="deepseek-v4-pro",
        usage={"input_tokens": 100, "output_tokens": 1024},
        cost_yuan=0.25,
        latency_ms=5.0,
    )
    transport = SequenceTransport(
        [truncated, _response(answer=task.correct_option, cost=0.10)]
    )
    budget = BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0)

    record = execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=transport,
        policy=contract.provider_policy("deepseek"),
        budget=budget,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert record["status"] == "complete"
    assert record["error"] is None
    assert record["cost_yuan"] == pytest.approx(0.35)
    assert budget.total_cost_yuan == pytest.approx(0.35)
    assert record["attempts"][0] == {
        "attempt": 1,
        "status": "failed",
        "error_category": "truncated_length",
        "request_max_tokens": 1024,
        "latency_ms": 5.0,
        "model_returned": "deepseek-v4-pro",
        "finish_reason": "length",
        "reasoning_tokens": 900,
        "usage": {"input_tokens": 100, "output_tokens": 1024},
        "cost_yuan": 0.25,
        "cost_known": True,
        "billing_ambiguity": False,
        "cost_reserve_yuan": 0.0,
    }
    assert record["attempts"][1]["request_max_tokens"] == 2048
    assert [call["max_tokens"] for call in transport.calls] == [1024, 2048]


def test_truncated_attempt_model_drift_is_terminal_before_valid_retry():
    from experiments.llm_sim.transport import ProviderTruncatedResponseError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        enumerate_tasks,
        execute_task,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    truncated = ProviderTruncatedResponseError(
        finish_reason="length",
        request_max_tokens=1024,
        reasoning_tokens=900,
        returned_model="wrong-model",
        usage={"input_tokens": 100, "output_tokens": 1024},
        cost_yuan=0.25,
        latency_ms=5.0,
    )
    transport = SequenceTransport(
        [truncated, _response(answer=task.correct_option, cost=0.10)]
    )
    budget = BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0)

    record = execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=transport,
        policy=contract.provider_policy("deepseek"),
        budget=budget,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert record["status"] == "excluded_model_drift"
    assert record["error"] == "returned_model_drift"
    assert record["model_id"] == "wrong-model"
    assert len(record["attempts"]) == 1
    assert record["attempts"][0]["request_max_tokens"] == 1024
    assert len(transport.calls) == 1
    assert budget.total_cost_yuan == pytest.approx(0.25)


def test_execute_task_excludes_model_drift_and_never_parses_it_as_valid():
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        execute_task,
        load_runtime_contract,
        enumerate_tasks,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(task for task in enumerate_tasks(contract, phase="pilot") if task.condition == "controlled")
    transport = SequenceTransport([_response(returned_model="unexpected-model")])
    record = execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=transport,
        policy=contract.provider_policy("deepseek"),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert record["status"] == "excluded_model_drift"
    assert record["parsed_output"] is None
    assert len(transport.calls) == 1


def test_provider_runner_is_resumable_and_phase_isolated(tmp_path: Path):
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        load_runtime_contract,
        enumerate_tasks,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(task for task in enumerate_tasks(contract, phase="pilot") if task.condition == "controlled")
    first_transport = SequenceTransport([_response(answer=task.correct_option)])
    first = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=first_transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    summary = first.run_tasks([task])
    assert summary["complete_records"] == 1
    assert len(first_transport.calls) == 1

    second_transport = SequenceTransport([])
    second = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=second_transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    resumed = second.run_tasks([task])
    assert resumed["resumed_records"] == 1
    assert len(second_transport.calls) == 0
    assert (tmp_path / "llm-personas-v2-dual/pilot").is_dir()
    assert not (tmp_path / "llm-personas-v2-dual/main").exists()


def test_provider_runner_scopes_identical_tasks_by_provider(tmp_path: Path):
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    for provider in ("deepseek", "doubao"):
        model = contract.provider_model(provider)
        runner = V2ProviderRunner(
            contract=contract,
            output_base=tmp_path,
            phase="pilot",
            provider=provider,
            transport=SequenceTransport(
                [_response(answer=task.correct_option, returned_model=model)]
            ),
            budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
            sleep=lambda _: None,
            random_value=lambda: 0.5,
        )
        assert runner.run_tasks([task])["complete_records"] == 1

    root = tmp_path / "llm-personas-v2-dual/pilot/records"
    assert (root / "deepseek" / f"{task.task_id}.json").is_file()
    assert (root / "doubao" / f"{task.task_id}.json").is_file()
    manifests = tmp_path / "llm-personas-v2-dual/pilot/provider_manifests"
    assert (manifests / "deepseek.json").is_file()
    assert (manifests / "doubao.json").is_file()


def test_budget_ledger_hard_fuse_stops_new_calls_after_recorded_cost():
    from experiments.llm_sim_v2.runner import BudgetFuseOpen, BudgetLedger

    budget = BudgetLedger(
        soft_warning_yuan=1.0,
        hard_fuse_yuan=2.0,
        initial_cost_yuan=0.5,
    )
    budget.add_cost(1.25)
    assert budget.soft_warning_triggered is True
    assert budget.hard_fuse_triggered is False
    budget.add_cost(0.75)
    assert budget.hard_fuse_triggered is True
    with pytest.raises(BudgetFuseOpen):
        budget.assert_new_call_allowed()


def test_existing_phase_cost_rebuilds_one_budget_across_all_provider_records(tmp_path: Path):
    from experiments.llm_sim_v2.collect import existing_phase_cost

    root = tmp_path / "llm-personas-v2-dual/main/records"
    for provider, costs in {"deepseek": [0.1, 0.2], "doubao": [0.3]}.items():
        provider_root = root / provider
        provider_root.mkdir(parents=True)
        for index, cost in enumerate(costs):
            (provider_root / f"{index}.json").write_text(
                json.dumps(
                        {
                            "run_id": "llm-personas-v2-dual",
                            "phase": "main",
                            "analysis_population": "main",
                            "attempts": [
                                {
                                    "attempt": 1,
                                    "status": "response",
                                    "cost_yuan": cost,
                                    "cost_known": True,
                                    "billing_ambiguity": False,
                                    "cost_reserve_yuan": 0.0,
                                }
                            ],
                            "known_cost_yuan": cost,
                            "unknown_cost_reserve_yuan": 0.0,
                            "cost_yuan": cost,
                            "has_unknown_cost_attempts": False,
                            "needs_user": False,
                        }
                ),
                encoding="utf-8",
            )
    assert existing_phase_cost(tmp_path, phase="main") == pytest.approx(0.6)

    (root / "doubao/0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="record|cost|phase|run"):
        existing_phase_cost(tmp_path, phase="main")


def test_collection_entrypoint_requires_explicit_live_and_partial_acknowledgement(tmp_path: Path):
    from experiments.llm_sim_v2.collect import build_parser, run_collection

    parser = build_parser()
    offline = parser.parse_args(
        ["--repo-root", str(REPO_ROOT), "--phase", "pilot", "--output-base", str(tmp_path)]
    )
    with pytest.raises(SystemExit, match="--live"):
        run_collection(offline)

    partial = parser.parse_args(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--phase",
            "pilot",
            "--output-base",
            str(tmp_path),
            "--live",
            "--limit",
            "1",
        ]
    )
    with pytest.raises(SystemExit, match="allow-partial"):
        run_collection(partial)


def test_runtime_task_manifest_binds_both_phase_task_sets_and_runtime_commit():
    from experiments.llm_sim_v2.runner import (
        build_runtime_task_manifest,
        load_runtime_contract,
        verify_runtime_task_manifest,
    )

    contract = load_runtime_contract(REPO_ROOT)
    manifest = build_runtime_task_manifest(
        contract,
        runtime_commit="a" * 40,
        frozen_at_utc="2026-07-15T06:30:00Z",
    )

    assert manifest["freeze_commit"] == "e3c0d4dbe6f37303d9eac86ecd9c1af823f152b9"
    assert manifest["runtime_commit"] == "a" * 40
    assert manifest["phases"]["pilot"]["task_count"] == 128
    assert manifest["phases"]["main"]["task_count"] == 1528
    assert len(manifest["phases"]["pilot"]["task_ids"]) == 128
    assert len(manifest["phases"]["main"]["task_ids"]) == 1528
    assert len(manifest["runtime_task_manifest_sha256"]) == 64
    assert verify_runtime_task_manifest(contract, manifest)["ok"] is True

    manifest["phases"]["main"]["task_ids"].pop()
    with pytest.raises(ValueError, match="task|manifest|digest"):
        verify_runtime_task_manifest(contract, manifest)


def test_collection_scope_requires_partial_ack_for_provider_override_and_marks_output():
    from experiments.llm_sim_v2.collect import resolve_collection_scope

    frozen = ("deepseek", "doubao")
    with pytest.raises(SystemExit, match="allow-partial"):
        resolve_collection_scope(
            frozen_providers=frozen,
            requested_providers=("doubao",),
            limit=None,
            allow_partial=False,
        )

    scope = resolve_collection_scope(
        frozen_providers=frozen,
        requested_providers=("doubao",),
        limit=None,
        allow_partial=True,
    )
    assert scope == {
        "collection_mode": "development_partial",
        "development_only": True,
        "partial": True,
        "formal_analysis_eligible": False,
        "frozen_providers": ["deepseek", "doubao"],
        "selected_providers": ["doubao"],
        "task_limit": None,
    }

    with pytest.raises(SystemExit, match="duplicate"):
        resolve_collection_scope(
            frozen_providers=frozen,
            requested_providers=("deepseek", "deepseek"),
            limit=None,
            allow_partial=True,
        )


def test_formal_collection_scope_requires_exact_frozen_population():
    from experiments.llm_sim_v2.collect import resolve_collection_scope

    scope = resolve_collection_scope(
        frozen_providers=("deepseek", "doubao"),
        requested_providers=None,
        limit=None,
        allow_partial=False,
    )
    assert scope["collection_mode"] == "formal"
    assert scope["development_only"] is False
    assert scope["partial"] is False
    assert scope["formal_analysis_eligible"] is True


def test_existing_run_cost_rebuilds_budget_across_pilot_and_main(tmp_path: Path):
    from experiments.llm_sim_v2.collect import existing_run_cost

    for phase, provider, cost in (
        ("pilot", "deepseek", 0.1),
        ("pilot", "doubao", 0.2),
        ("main", "deepseek", 0.3),
    ):
        path = (
            tmp_path
            / "llm-personas-v2-dual"
            / phase
            / "records"
            / provider
            / f"{provider}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "run_id": "llm-personas-v2-dual",
                    "phase": phase,
                    "analysis_population": phase,
                    "provider": provider,
                    "task_id": f"{phase}-{provider}",
                    "attempts": [
                        {
                            "attempt": 1,
                            "status": "response",
                            "cost_yuan": cost,
                            "cost_known": True,
                            "billing_ambiguity": False,
                            "cost_reserve_yuan": 0.0,
                        }
                    ],
                    "known_cost_yuan": cost,
                    "unknown_cost_reserve_yuan": 0.0,
                    "cost_yuan": cost,
                    "has_unknown_cost_attempts": False,
                    "needs_user": False,
                }
            ),
            encoding="utf-8",
        )

    assert existing_run_cost(tmp_path) == pytest.approx(0.6)


def test_prior_cost_ledger_is_machine_bound_and_cli_override_is_removed():
    from experiments.llm_sim_v2.collect import build_parser
    from experiments.llm_sim_v2.runner import (
        RUNTIME_PATHS,
        load_runtime_contract,
        verify_prior_cost_ledger,
    )

    contract = load_runtime_contract(REPO_ROOT)
    ledger = contract.prior_cost_ledger
    assert ledger["known_cost_yuan"] == pytest.approx(0.54466321)
    assert ledger["pre_run_ambiguity_reserve_yuan"] == pytest.approx(0.113)
    assert ledger["pre_run_total_bound_yuan"] == pytest.approx(0.65766321)
    assert ledger["unknown_attempt_reserve_yuan"] == pytest.approx(10.0)
    assert verify_prior_cost_ledger(ledger)["ok"] is True
    assert "experiments/llm_sim_v2/prior_cost_ledger.json" in RUNTIME_PATHS

    drifted = copy.deepcopy(dict(ledger))
    drifted["known_cost_entries"][0]["cost_yuan"] = 0.0
    drifted.pop("prior_cost_ledger_sha256", None)
    drifted["prior_cost_ledger_sha256"] = _canonical_sha(drifted)
    with pytest.raises(ValueError, match="prior cost ledger"):
        verify_prior_cost_ledger(drifted)

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--phase",
                "pilot",
                "--prior-documented-cost",
                "0.0",
            ]
        )


def test_run_budget_ledger_uses_machine_prior_and_separates_known_from_reserve(
    tmp_path: Path,
):
    from experiments.llm_sim_v2.collect import reconcile_run_budget_ledger
    from experiments.llm_sim_v2.runner import load_runtime_contract

    contract = load_runtime_contract(REPO_ROOT)
    ledger = reconcile_run_budget_ledger(
        tmp_path,
        prior_cost_ledger=contract.prior_cost_ledger,
        soft_warning_yuan=300.0,
        hard_fuse_yuan=450.0,
    )
    assert ledger["prior_known_cost_yuan"] == pytest.approx(0.54466321)
    assert ledger["prior_ambiguity_reserve_yuan"] == pytest.approx(0.113)
    assert ledger["prior_documented_cost_yuan"] == pytest.approx(0.65766321)
    assert ledger["prior_cost_ledger_sha256"] == contract.prior_cost_ledger[
        "prior_cost_ledger_sha256"
    ]
    assert ledger["immutable_record_cost_yuan"] == 0.0
    assert ledger["immutable_record_known_cost_yuan"] == 0.0
    assert ledger["immutable_record_unknown_reserve_yuan"] == 0.0
    assert ledger["total_known_cost_yuan"] == pytest.approx(0.54466321)
    assert ledger["total_unknown_reserve_yuan"] == pytest.approx(0.113)
    assert ledger["total_accounted_cost_yuan"] == pytest.approx(0.65766321)
    assert ledger["needs_user"] is False
    assert ledger["hard_fuse_yuan"] == 450.0
    ledger_path = tmp_path / "llm-personas-v2-dual/run_budget_ledger.json"
    assert json.loads(ledger_path.read_text(encoding="utf-8")) == ledger


def test_existing_run_cost_rejects_attempt_total_mismatch_and_counts_reserve(
    tmp_path: Path,
):
    from experiments.llm_sim_v2.collect import existing_run_cost

    path = (
        tmp_path
        / "llm-personas-v2-dual/pilot/records/deepseek/fake-task.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": "llm-personas-v2-dual",
        "phase": "pilot",
        "analysis_population": "pilot",
        "provider": "deepseek",
        "task_id": "fake-task",
        "attempts": [
            {
                "attempt": 1,
                "status": "response",
                "cost_yuan": 5.0,
                "cost_known": True,
                "billing_ambiguity": False,
                "cost_reserve_yuan": 0.0,
            }
        ],
        "known_cost_yuan": 0.0,
        "unknown_cost_reserve_yuan": 0.0,
        "cost_yuan": 0.0,
        "has_unknown_cost_attempts": False,
        "needs_user": False,
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="attempt|reconcil"):
        existing_run_cost(tmp_path)

    record.update(
        {
            "attempts": [
                {
                    "attempt": 1,
                    "status": "failed",
                    "cost_yuan": None,
                    "cost_known": False,
                    "billing_ambiguity": True,
                    "cost_reserve_yuan": 10.0,
                }
            ],
            "known_cost_yuan": 0.0,
            "unknown_cost_reserve_yuan": 10.0,
            "cost_yuan": 10.0,
            "has_unknown_cost_attempts": True,
            "needs_user": True,
        }
    )
    path.write_text(json.dumps(record), encoding="utf-8")
    assert existing_run_cost(tmp_path) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("soft_warning", "hard_fuse"),
    [(301.0, 450.0), (300.0, 451.0), (299.0, 450.0)],
)
def test_formal_budget_thresholds_cannot_drift_or_raise_hard_fuse(
    soft_warning: float,
    hard_fuse: float,
):
    from experiments.llm_sim_v2.collect import validate_budget_thresholds

    with pytest.raises(SystemExit, match="frozen|300|450"):
        validate_budget_thresholds(
            soft_warning_yuan=soft_warning,
            hard_fuse_yuan=hard_fuse,
            frozen_soft_warning_yuan=300.0,
            frozen_hard_fuse_yuan=450.0,
        )


def test_prompt_revision_one_is_blocked_until_pilot_failure_evidence_is_bound():
    from experiments.llm_sim_v2.runner import enumerate_tasks

    contract, _ = _contract_with_revision_one()
    with pytest.raises(
        ValueError,
        match=(
            "prompt revision 1 blocked until committed pilot-failure evidence is bound"
        ),
    ):
        enumerate_tasks(contract, phase="pilot")


def test_stale_active_prompt_revision_pointer_is_rejected():
    from dataclasses import replace

    from experiments.llm_sim_v2.runner import enumerate_tasks, load_runtime_contract

    contract = load_runtime_contract(REPO_ROOT)
    ledger = copy.deepcopy(dict(contract.prompt_ledger))
    ledger["current_revision"] = 1
    ledger.pop("prompt_ledger_sha256", None)
    ledger["prompt_ledger_sha256"] = _canonical_sha(ledger)
    with pytest.raises(ValueError, match="revision|ledger"):
        enumerate_tasks(replace(contract, prompt_ledger=ledger), phase="pilot")


def test_active_prompt_revision_rejects_prompt_file_byte_drift():
    from dataclasses import replace

    from experiments.llm_sim_v2.runner import enumerate_tasks, load_runtime_contract

    contract = load_runtime_contract(REPO_ROOT)
    ledger = copy.deepcopy(dict(contract.prompt_ledger))
    revision = ledger["revisions"][0]
    revision["prompt_files"][0]["sha256"] = "0" * 64
    revision.pop("prompt_contract_sha256", None)
    revision["prompt_contract_sha256"] = _canonical_sha(revision)
    ledger.pop("prompt_ledger_sha256", None)
    ledger["prompt_ledger_sha256"] = _canonical_sha(ledger)
    with pytest.raises(ValueError, match="prompt|byte|sha"):
        enumerate_tasks(replace(contract, prompt_ledger=ledger), phase="pilot")


def test_provider_policy_binds_frozen_backoff_breaker_and_doubao_limits():
    from experiments.llm_sim_v2.runner import load_runtime_contract

    contract = load_runtime_contract(REPO_ROOT)
    deepseek = contract.provider_policy("deepseek")
    doubao = contract.provider_policy("doubao")
    assert (
        deepseek.failure_threshold,
        deepseek.base_backoff_seconds,
        deepseek.max_backoff_seconds,
        deepseek.cooldown_seconds,
        deepseek.jitter_fraction,
    ) == (3, 1.0, 30.0, 120.0, 0.25)
    assert doubao.concurrency == 2
    assert doubao.base_backoff_seconds == 2.0
    assert doubao.max_backoff_seconds == 60.0
    assert doubao.cooldown_seconds == 180.0


def test_retry_honors_retry_after_and_frozen_jitter():
    from experiments.llm_sim.transport import ProviderHTTPError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        enumerate_tasks,
        execute_task,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    transport = SequenceTransport(
        [
            ProviderHTTPError(429, retry_after_seconds=7.0),
            _response(answer=task.correct_option),
        ]
    )
    delays: list[float] = []
    execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=transport,
        policy=contract.provider_policy("deepseek"),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=delays.append,
        random_value=lambda: 1.0,
    )
    assert delays == [7.0]


def test_timeout_attempt_reserves_cost_and_marks_record_and_lifecycle_needs_user(
    tmp_path: Path,
):
    from experiments.llm_sim.transport import ProviderNetworkError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        execute_task,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    record = execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=SequenceTransport([ProviderNetworkError()] * 4),
        policy=contract.provider_policy("deepseek"),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        unknown_attempt_reserve_yuan=contract.prior_cost_ledger[
            "unknown_attempt_reserve_yuan"
        ],
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    assert record["status"] == "technical_failure"
    assert record["model_id"] is None
    assert record["has_unknown_cost_attempts"] is True
    assert record["known_cost_yuan"] == 0.0
    assert record["unknown_cost_reserve_yuan"] == pytest.approx(40.0)
    assert record["cost_yuan"] == pytest.approx(40.0)
    assert record["needs_user"] is True
    assert all(attempt["cost_known"] is False for attempt in record["attempts"])
    assert all(attempt["billing_ambiguity"] is True for attempt in record["attempts"])
    assert all(
        attempt["cost_reserve_yuan"] == pytest.approx(10.0)
        for attempt in record["attempts"]
    )

    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([ProviderNetworkError()] * 4),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    summary = runner.run_tasks([task])
    assert summary["needs_user"]["required"] is True
    assert summary["needs_user"]["unknown_cost_attempt_count"] == 4
    lifecycle_path = next(
        (
            tmp_path
            / "llm-personas-v2-dual/pilot/provider_lifecycle/deepseek"
        ).glob("*.json")
    )
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["needs_user"]["required"] is True


def test_paid_attempt_that_opens_fuse_is_still_returned_for_immutable_storage():
    from experiments.llm_sim.transport import ProviderTruncatedResponseError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        enumerate_tasks,
        execute_task,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    truncated = ProviderTruncatedResponseError(
        finish_reason="length",
        request_max_tokens=1024,
        reasoning_tokens=900,
        returned_model="deepseek-v4-pro",
        usage={"input_tokens": 100, "output_tokens": 1024},
        cost_yuan=0.25,
        latency_ms=5.0,
    )
    record = execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=SequenceTransport([truncated]),
        policy=contract.provider_policy("deepseek"),
        budget=BudgetLedger(soft_warning_yuan=0.1, hard_fuse_yuan=0.2),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    assert record["status"] == "technical_failure"
    assert record["error"] == "budget_fuse_open_after_attempt"
    assert record["cost_yuan"] == pytest.approx(0.25)
    assert len(record["attempts"]) == 1


def _phase_provenance_fixture(*, partial: bool = False):
    from experiments.llm_sim_v2.collect import resolve_collection_scope
    from experiments.llm_sim_v2.runner import (
        build_phase_provenance,
        build_runtime_task_manifest,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    runtime = build_runtime_task_manifest(
        contract,
        runtime_commit="a" * 40,
        frozen_at_utc="2026-07-15T07:05:00Z",
    )
    proof = {
        "schema_version": "yher.llm_sim_v2.runtime_task_manifest_proof.v1",
        "ok": True,
        "run_id": "llm-personas-v2-dual",
        "runtime_task_manifest_sha256": runtime[
            "runtime_task_manifest_sha256"
        ],
        "runtime_commit": runtime["runtime_commit"],
        "git_proof": {
            "schema_version": "yher.llm_sim_v2.git_proof.v1",
            "ok": True,
            "commit": runtime["runtime_commit"],
            "current_head": runtime["runtime_commit"],
            "ancestor_of_head": True,
            "observation_timestamp": contract.freeze_proof[
                "observation_timestamp"
            ],
            "precedes_observation": True,
            "byte_identical": True,
            "files": runtime["runtime_files"],
        },
    }
    frozen_providers = tuple(contract.config["pilot"]["providers"])
    selected_providers = ("deepseek",) if partial else None
    scope = resolve_collection_scope(
        frozen_providers=frozen_providers,
        requested_providers=selected_providers,
        limit=1 if partial else None,
        allow_partial=partial,
    )
    tasks = enumerate_tasks(contract, phase="pilot")
    if partial:
        tasks = [next(task for task in tasks if task.condition == "controlled")]
    phase = build_phase_provenance(
        contract,
        runtime_manifest=runtime,
        runtime_proof=proof,
        phase="pilot",
        tasks=tasks,
        collection_scope=scope,
        prior_cost_ledger=contract.prior_cost_ledger,
        first_observation_at_utc=contract.freeze_proof[
            "observation_timestamp"
        ],
    )
    return contract, runtime, tasks, phase


def test_phase_provenance_serializes_all_frozen_and_execution_bindings(tmp_path: Path):
    from experiments.llm_sim_v2.runner import (
        validate_formal_phase_provenance,
        write_phase_provenance,
    )

    contract, runtime, tasks, phase = _phase_provenance_fixture()
    assert phase["freeze"]["freeze_manifest_sha256"] == contract.freeze_manifest[
        "freeze_manifest_sha256"
    ]
    assert phase["source"]["source_set_sha256"] == contract.freeze_manifest[
        "source_set_sha256"
    ]
    assert phase["target"]["target_set_hash"] == contract.config["target_set_hash"]
    assert phase["grid_sha256"] == contract.freeze_manifest["grid_sha256"]
    assert phase["prompt"]["revision"] == 0
    assert phase["runtime"]["execution_commit"] == "a" * 40
    assert phase["runtime"]["execution_files"] == runtime["runtime_files"]
    assert phase["task_roster"]["expected_task_count"] == len(tasks) == 128
    assert validate_formal_phase_provenance(phase)["ok"] is True

    path = write_phase_provenance(tmp_path, phase=phase)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == phase


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("freeze", "freeze_manifest_sha256"),
        ("source", "source_set_sha256"),
        ("target", "target_set_hash"),
        ("prompt", "prompt_ledger_sha256"),
        ("runtime", "runtime_file_set_sha256"),
        ("budget", "prior_cost_ledger_sha256"),
    ],
)
def test_rehashed_stored_phase_provenance_is_revalidated_against_active_contract(
    section: str,
    field: str,
):
    from experiments.llm_sim_v2.runner import (
        verify_phase_provenance_against_contract,
    )

    contract, runtime, tasks, phase = _phase_provenance_fixture()
    runtime_proof = {
        "schema_version": "yher.llm_sim_v2.runtime_task_manifest_proof.v1",
        "ok": True,
        "run_id": "llm-personas-v2-dual",
        "runtime_task_manifest_sha256": runtime[
            "runtime_task_manifest_sha256"
        ],
        "runtime_commit": runtime["runtime_commit"],
        "git_proof": phase["runtime"]["git_proof"],
    }
    scope = {
        key: phase[key]
        for key in (
            "collection_mode",
            "development_only",
            "partial",
            "formal_analysis_eligible",
            "frozen_providers",
            "selected_providers",
            "task_limit",
        )
    }
    assert verify_phase_provenance_against_contract(
        phase,
        contract=contract,
        runtime_manifest=runtime,
        runtime_proof=runtime_proof,
        tasks=tasks,
        collection_scope=scope,
    )["ok"] is True

    tampered = copy.deepcopy(phase)
    tampered[section][field] = "0" * 64
    tampered.pop("phase_provenance_sha256", None)
    tampered["phase_provenance_sha256"] = _canonical_sha(tampered)
    with pytest.raises(ValueError, match="phase provenance.*contract"):
        verify_phase_provenance_against_contract(
            tampered,
            contract=contract,
            runtime_manifest=runtime,
            runtime_proof=runtime_proof,
            tasks=tasks,
            collection_scope=scope,
        )


def test_formal_analyzer_rejects_partial_phase_provenance():
    from experiments.llm_sim_v2.runner import validate_formal_phase_provenance

    _, _, _, phase = _phase_provenance_fixture(partial=True)
    assert phase["development_only"] is True
    assert phase["partial"] is True
    assert phase["formal_analysis_eligible"] is False
    with pytest.raises(ValueError, match="partial|development|formal"):
        validate_formal_phase_provenance(phase)


def test_response_and_provider_manifest_bind_phase_provenance(tmp_path: Path):
    from experiments.llm_sim_v2.runner import BudgetLedger, V2ProviderRunner

    contract, _, tasks, phase = _phase_provenance_fixture(partial=True)
    task = next(task for task in tasks if task.condition == "controlled")
    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([_response(answer=task.correct_option)]),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        phase_provenance=phase,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    summary = runner.run_tasks([task])
    record_path = (
        tmp_path
        / "llm-personas-v2-dual/pilot/records/deepseek"
        / f"{task.task_id}.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["collection_mode"] == "development_partial"
    assert record["development_only"] is True
    assert record["partial"] is True
    assert record["formal_analysis_eligible"] is False
    binding = record["provenance"]
    assert binding["phase_provenance_sha256"] == phase[
        "phase_provenance_sha256"
    ]
    assert binding["freeze_manifest_sha256"] == phase["freeze"][
        "freeze_manifest_sha256"
    ]
    assert binding["runtime_task_manifest_sha256"] == phase["runtime"][
        "runtime_task_manifest_sha256"
    ]
    assert binding["execution_commit"] == "a" * 40
    assert summary["provenance"] == binding


def test_fuse_lifecycle_reports_every_expected_missing_and_skipped_task(tmp_path: Path):
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    tasks = enumerate_tasks(contract, phase="pilot")[:5]
    transport = SequenceTransport([])
    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=transport,
        budget=BudgetLedger(
            soft_warning_yuan=0.5,
            hard_fuse_yuan=1.0,
            initial_cost_yuan=1.0,
        ),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    summary = runner.run_tasks(tasks)
    lifecycle = summary["lifecycle"]
    assert lifecycle["expected_count"] == 5
    assert lifecycle["present_count"] == 0
    assert lifecycle["missing_count"] == 5
    assert lifecycle["interrupted_count"] == 0
    assert lifecycle["fuse_skipped_count"] == 5
    assert lifecycle["fuse_skipped_task_ids"] == [task.task_id for task in tasks]
    assert summary["provider_lifecycle"] == "fuse_open"
    assert len(transport.calls) == 0


def test_keyboard_interrupt_always_writes_interrupted_provider_manifest(tmp_path: Path):
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = enumerate_tasks(contract, phase="pilot")[0]
    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([KeyboardInterrupt()]),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run_tasks([task])

    path = tmp_path / "llm-personas-v2-dual/pilot/provider_manifests/deepseek.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["provider_lifecycle"] == "interrupted"
    assert manifest["lifecycle"]["expected_count"] == 1
    assert manifest["lifecycle"]["present_count"] == 0
    assert manifest["lifecycle"]["missing_count"] == 1
    assert manifest["lifecycle"]["interrupted_count"] == 1
    assert manifest["lifecycle"]["interrupted_task_ids"] == [task.task_id]


def test_provider_breaker_uses_bounded_batches_and_discloses_skipped_tasks(tmp_path: Path):
    from dataclasses import replace

    from experiments.llm_sim.transport import ProviderNetworkError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    tasks = [
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    ][:10]
    transport = SequenceTransport([ProviderNetworkError() for _ in range(3)])
    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
        clock=lambda: 1000.0,
    )
    runner.policy = replace(runner.policy, concurrency=1, max_attempts=1)
    summary = runner.run_tasks(tasks)
    assert len(transport.calls) == 3
    assert summary["provider_lifecycle"] == "excluded_repeated_failure"
    assert summary["lifecycle"]["present_count"] == 3
    assert summary["lifecycle"]["missing_count"] == 7
    assert summary["lifecycle"]["breaker_skipped_count"] == 7
    assert summary["breaker"]["status"] == "open"
    assert summary["breaker"]["failure_threshold"] == 3
    assert summary["breaker"]["resume_not_before_epoch"] == 1120.0


def test_blind_schema_fraction_and_complete_cluster_gate_are_manifested(tmp_path: Path):
    from dataclasses import replace

    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    tasks = [
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "blind"
    ][:2]
    transport = SequenceTransport(
        [_response(answer=task.correct_option, blind=False) for task in tasks]
    )
    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    runner.policy = replace(
        runner.policy,
        concurrency=1,
        max_attempts=1,
        failure_threshold=3,
    )
    summary = runner.run_tasks(tasks)
    blind = summary["condition_lifecycle"]["blind"]
    assert blind["expected_count"] == 2
    assert blind["invalid_schema_count"] == 2
    assert blind["invalid_schema_fraction"] == 1.0
    assert blind["excluded_invalid_schema"] is True
    assert blind["complete_cluster_count"] == 0
    assert blind["minimum_complete_clusters"] == 45
    assert blind["minimum_complete_clusters_met"] is False


def test_resume_rebuilds_breaker_from_immutable_failure_prefix_without_manifest(
    tmp_path: Path,
):
    from dataclasses import replace

    from experiments.llm_sim.transport import ProviderNetworkError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    tasks = [
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    ][:5]
    first = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([ProviderNetworkError(), ProviderNetworkError()]),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    first.policy = replace(first.policy, concurrency=1, max_attempts=1)
    first.run_tasks(tasks[:2])
    manifest_path = (
        tmp_path
        / "llm-personas-v2-dual/pilot/provider_manifests/deepseek.json"
    )
    manifest_path.unlink()

    transport = SequenceTransport([ProviderNetworkError()])
    resumed = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
        clock=lambda: 2000.0,
    )
    resumed.policy = replace(resumed.policy, concurrency=1, max_attempts=1)
    summary = resumed.run_tasks(tasks)
    assert len(transport.calls) == 1
    assert summary["resumed_records"] == 2
    assert summary["lifecycle"]["present_count"] == 3
    assert summary["lifecycle"]["breaker_skipped_count"] == 2
    assert summary["breaker"]["consecutive_failures"] == 3


def test_production_retry_default_uses_system_random_jitter(monkeypatch):
    import experiments.llm_sim_v2.runner as runner_module
    from experiments.llm_sim.transport import ProviderHTTPError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        enumerate_tasks,
        execute_task,
        load_runtime_contract,
    )

    class FixedSystemRandom:
        def random(self) -> float:
            return 1.0

    monkeypatch.setattr(runner_module, "_SYSTEM_RANDOM", FixedSystemRandom())
    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    delays: list[float] = []
    execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=SequenceTransport(
            [ProviderHTTPError(429), _response(answer=task.correct_option)]
        ),
        policy=contract.provider_policy("deepseek"),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=delays.append,
        random_value=None,
    )
    assert delays == [1.25]


def test_retry_after_seconds_is_capped_by_provider_policy():
    from experiments.llm_sim.transport import ProviderHTTPError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        enumerate_tasks,
        execute_task,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    delays: list[float] = []
    execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=SequenceTransport(
            [
                ProviderHTTPError(429, retry_after_seconds=86_400),
                _response(answer=task.correct_option),
            ]
        ),
        policy=contract.provider_policy("deepseek"),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=delays.append,
        random_value=lambda: 0.5,
    )
    assert delays == [30.0]


def test_retry_after_http_date_is_parsed_then_capped():
    from experiments.llm_sim.transport import ProviderHTTPError
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        enumerate_tasks,
        execute_task,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    now = 2_000_000_000.0
    delays: list[float] = []
    error = ProviderHTTPError(429)
    error.retry_after_seconds = formatdate(now + 60.0, usegmt=True)
    execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=SequenceTransport([error, _response(answer=task.correct_option)]),
        policy=contract.provider_policy("deepseek"),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=delays.append,
        random_value=lambda: 0.5,
        wall_time=lambda: now,
    )
    assert delays == [30.0]


def test_interrupt_then_resume_retains_immutable_lifecycle_history(tmp_path: Path):
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    task = enumerate_tasks(contract, phase="pilot")[0]
    interrupted = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([KeyboardInterrupt()]),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    with pytest.raises(KeyboardInterrupt):
        interrupted.run_tasks([task])

    resumed = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport(
            [
                _response(
                    answer=task.correct_option,
                    blind=task.condition == "blind",
                )
            ]
        ),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    manifest = resumed.run_tasks([task])
    assert manifest["provider_lifecycle"] == "complete"
    assert [row["provider_lifecycle"] for row in manifest["lifecycle_history"]] == [
        "interrupted",
        "complete",
    ]
    event_paths = sorted(
        (
            tmp_path
            / "llm-personas-v2-dual/pilot/provider_lifecycle/deepseek"
        ).glob("*.json")
    )
    assert len(event_paths) == 2
    assert all(json.loads(path.read_text())["event_index"] == index for index, path in enumerate(event_paths))


def test_collection_records_unavailable_provider_and_continues_frozen_roster(
    tmp_path: Path,
    monkeypatch,
):
    from dataclasses import replace

    import experiments.llm_sim_v2.collect as collect_module
    from experiments.llm_sim_v2.runner import (
        build_runtime_task_manifest,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    runtime = build_runtime_task_manifest(
        contract,
        runtime_commit="a" * 40,
        frozen_at_utc="2026-07-15T07:05:00Z",
    )
    contract = replace(contract, runtime_manifest=runtime)
    task = enumerate_tasks(contract, phase="pilot")[0]
    proof = {
        "schema_version": "yher.llm_sim_v2.runtime_task_manifest_proof.v1",
        "ok": True,
        "run_id": "llm-personas-v2-dual",
        "runtime_task_manifest_sha256": runtime[
            "runtime_task_manifest_sha256"
        ],
        "runtime_commit": runtime["runtime_commit"],
        "git_proof": {
            "schema_version": "yher.llm_sim_v2.git_proof.v1",
            "ok": True,
            "commit": runtime["runtime_commit"],
            "current_head": runtime["runtime_commit"],
            "ancestor_of_head": True,
            "observation_timestamp": "2026-07-15T07:10:00Z",
            "precedes_observation": True,
            "byte_identical": True,
            "files": runtime["runtime_files"],
        },
    }
    monkeypatch.setattr(collect_module, "load_runtime_contract", lambda _: contract)
    monkeypatch.setattr(
        collect_module,
        "verify_runtime_task_manifest",
        lambda *args, **kwargs: proof,
    )
    calls: list[str] = []

    def transport_for(provider: str, **kwargs: Any):
        calls.append(provider)
        if provider == "deepseek":
            raise RuntimeError("synthetic unavailable")
        return SequenceTransport(
            [
                _response(
                    answer=task.correct_option,
                    blind=task.condition == "blind",
                    returned_model=contract.provider_model("doubao"),
                )
            ]
        )

    monkeypatch.setattr(
        collect_module.HTTPProviderTransport,
        "from_environment",
        staticmethod(transport_for),
    )
    args = collect_module.build_parser().parse_args(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--phase",
            "pilot",
            "--output-base",
            str(tmp_path),
            "--provider",
            "deepseek",
            "--provider",
            "doubao",
            "--limit",
            "1",
            "--allow-partial",
            "--live",
        ]
    )
    results = collect_module.run_collection(args)
    assert calls == ["deepseek", "doubao"]
    assert [row["provider_lifecycle"] for row in results] == [
        "unavailable",
        "complete",
    ]
    unavailable = json.loads(
        (
            tmp_path
            / "llm-personas-v2-dual/pilot/provider_manifests/deepseek.json"
        ).read_text()
    )
    assert unavailable["unavailable"]["unavailable"] is True
    assert unavailable["unavailable"]["error_category"] == (
        "transport_configuration_or_credential_unavailable"
    )
    assert "synthetic unavailable" not in json.dumps(unavailable)


def test_blind_schema_exclusion_uses_78_primary_tasks_not_10_repeats(tmp_path: Path):
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    contract = load_runtime_contract(REPO_ROOT)
    blind = [
        task
        for task in enumerate_tasks(contract, phase="pilot")
        if task.condition == "blind"
    ]
    primary = [task for task in blind if not task.is_stability_repeat]
    repeats = [task for task in blind if task.is_stability_repeat]
    assert (len(primary), len(repeats)) == (78, 10)
    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([]),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
    )
    records = {
        task.task_id: {
            "status": "excluded_schema" if index < 40 else "complete"
        }
        for index, task in enumerate(primary)
    }
    records.update({task.task_id: {"status": "complete"} for task in repeats})
    lifecycle = runner._condition_lifecycle(blind, records)["blind"]
    assert lifecycle["expected_count"] == 88
    assert lifecycle["primary_expected_count"] == 78
    assert lifecycle["stability_repeat_expected_count"] == 10
    assert lifecycle["invalid_schema_count"] == 40
    assert lifecycle["invalid_schema_fraction"] == pytest.approx(40 / 78)
    assert lifecycle["excluded_invalid_schema"] is True

    records[primary[39].task_id] = {"status": "complete"}
    boundary = runner._condition_lifecycle(blind, records)["blind"]
    assert boundary["invalid_schema_count"] == 39
    assert boundary["invalid_schema_fraction"] == 0.5
    assert boundary["excluded_invalid_schema"] is False
