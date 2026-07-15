"""Runtime contracts for the frozen Persona v2 provider collection."""

from __future__ import annotations

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


class SequenceTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected provider call")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
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
    assert len(record["attempts"]) == 2
    assert record["retry_count"] == 1
    assert transport.calls[0]["messages"] == transport.calls[1]["messages"]
    assert transport.calls[0]["model"] == transport.calls[1]["model"]
    assert record["message_sha256"] == task.message_sha256
    assert record["prompt_contract_sha256"] == task.prompt_contract_sha256
    assert record["prompt_contract_sha256"] != task.wire_message_sha256
    assert budget.total_cost_yuan == pytest.approx(0.01)


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


def test_budget_ledger_hard_fuse_stops_new_calls_after_recorded_cost():
    from experiments.llm_sim_v2.runner import BudgetFuseOpen, BudgetLedger

    budget = BudgetLedger(soft_warning_yuan=1.0, hard_fuse_yuan=2.0)
    budget.add_cost(1.25)
    assert budget.soft_warning_triggered is True
    assert budget.hard_fuse_triggered is False
    budget.add_cost(0.75)
    assert budget.hard_fuse_triggered is True
    with pytest.raises(BudgetFuseOpen):
        budget.assert_new_call_allowed()


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
