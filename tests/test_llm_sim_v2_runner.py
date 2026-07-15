"""Runtime contracts for the frozen Persona v2 provider collection."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from email.utils import formatdate
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
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


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _git_commit_timestamp(commit: str) -> str:
    epoch = int(
        subprocess.check_output(
            ["git", "show", "-s", "--format=%ct", commit],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    )
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _git_parent(commit: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}^"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


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


def _write_runner_phase_stub(
    phase_root: Path,
    *,
    phase: str,
    providers: tuple[str, ...],
    task_ids: tuple[str, ...],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.phase_provenance.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "phase": phase,
        "analysis_population": phase,
        "collection_mode": "development_partial",
        "development_only": True,
        "partial": True,
        "formal_analysis_eligible": False,
        "selected_providers": list(providers),
        "frozen_providers": list(providers),
        "task_roster": {
            "expected_task_count": len(task_ids),
            "expected_task_ids": list(task_ids),
        },
        "budget": {
            "carried_forward_cost_ledger_sha256": None,
            "source_phase_receipt_sha256": None,
            "carried_forward_known_cost_yuan": 0.0,
            "carried_forward_unknown_reserve_yuan": 0.0,
            "carried_forward_total_accounted_cost_yuan": 0.0,
        },
    }
    artifact["phase_provenance_sha256"] = _canonical_sha(artifact)
    phase_root.mkdir(parents=True, exist_ok=True)
    (phase_root / "phase_provenance.json").write_text(
        json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


class _BlockingProcessTransport:
    def __init__(
        self,
        *,
        task: Any,
        call_log: str,
        started: object,
        release: object,
        block: bool,
    ) -> None:
        self.task = task
        self.call_log = call_log
        self.started = started
        self.release = release
        self.block = block
        self.calls = 0

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        descriptor = os.open(
            self.call_log,
            os.O_CREAT | os.O_WRONLY | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        finally:
            os.close(descriptor)
        if self.block:
            self.started.set()
            self.release.wait(5)
        return _response(
            answer=self.task.correct_option,
            blind=self.task.condition == "blind",
            returned_model=kwargs["model"],
        )


def _runner_process_worker(
    contract: Any,
    task: Any,
    output_base: str,
    call_log: str,
    started: object,
    release: object,
    results: object,
    block: bool,
) -> None:
    from experiments.llm_sim_v2.runner import BudgetLedger, V2ProviderRunner

    transport = _BlockingProcessTransport(
        task=task,
        call_log=call_log,
        started=started,
        release=release,
        block=block,
    )
    summary = V2ProviderRunner(
        contract=contract,
        output_base=output_base,
        phase="pilot",
        provider="deepseek",
        transport=transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    ).run_tasks([task])
    results.put(
        {
            "transport_calls": transport.calls,
            "receipt_calls": summary["evidence_receipt"]["provider_call_count"],
        }
    )


def _crash_open_invocation_worker(phase_root: str, task_id: str) -> None:
    from experiments.llm_sim_v2.evidence import ProviderEvidenceLedger

    ledger = ProviderEvidenceLedger(
        Path(phase_root),
        run_id="llm-personas-v2-dual",
        phase="pilot",
        provider="deepseek",
    )
    ledger.begin_invocation(expected_task_ids=(task_id,), resumed_task_ids=())
    os._exit(0)


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


def test_response_attempt_retains_exact_content_for_strict_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.runner as runner_module
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        execute_task,
        parse_provider_output,
    )

    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in runner_module.enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    raw_content = (
        ' { "simulated" : true, "answer" : "a", '
        '"rationale" : "exact UTF-8: 化学" }\n'
    )
    response = _response(answer=task.correct_option)
    response["content"] = raw_content

    record = execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=SequenceTransport([response]),
        policy=contract.provider_policy("deepseek"),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    attempt = record["attempts"][0]
    assert attempt["response_content"] == raw_content
    assert attempt["response_content_utf8_bytes"] == len(
        raw_content.encode("utf-8")
    )
    assert attempt["response_content_sha256"] == hashlib.sha256(
        raw_content.encode("utf-8")
    ).hexdigest()
    assert parse_provider_output(
        attempt["response_content"],
        condition=task.condition,
        option_keys={str(key).upper() for key in task.item["options"]},
    ) == record["parsed_output"]


def test_invalid_schema_attempt_retains_exact_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import experiments.llm_sim_v2.runner as runner_module
    from experiments.llm_sim_v2.runner import BudgetLedger, execute_task

    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in runner_module.enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    raw_content = (
        '{"simulated":true,"answer":"A","answer":"B",'
        '"rationale":"duplicate key"}'
    )
    response = _response(answer=task.correct_option)
    response["content"] = raw_content

    record = execute_task(
        task,
        provider="deepseek",
        model="deepseek-v4-pro",
        transport=SequenceTransport([response]),
        policy=replace(contract.provider_policy("deepseek"), max_attempts=1),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert record["status"] == "excluded_schema"
    assert record["attempts"][0]["response_content"] == raw_content
    assert record["attempts"][0]["response_content_sha256"] == hashlib.sha256(
        raw_content.encode("utf-8")
    ).hexdigest()


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
        "provider_response_received": False,
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


def test_runner_manifest_binds_record_bytes_and_resume_receipt_counts_zero_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.runner as runner_module
    from experiments.llm_sim_v2.runner import BudgetLedger, V2ProviderRunner

    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in runner_module.enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
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
    collected = first.run_tasks([task])
    record_path = (
        tmp_path
        / "llm-personas-v2-dual/pilot/records/deepseek"
        / f"{task.task_id}.json"
    )

    assert collected["record_set"]["records"] == [
        {
            "attempt_count": 1,
            "bytes": record_path.stat().st_size,
            "path": f"records/deepseek/{task.task_id}.json",
            "response_attempt_count": 1,
            "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
            "task_id": task.task_id,
        }
    ]
    assert collected["evidence_receipt"]["provider_call_count"] == 1

    resumed_transport = SequenceTransport([])
    resumed = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=resumed_transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    ).run_tasks([task])

    assert resumed_transport.calls == []
    assert resumed["evidence_receipt"]["invocation_kind"] == "resume"
    assert resumed["evidence_receipt"]["provider_call_count"] == 0
    assert resumed["evidence_receipt"]["before_store"]["file_set_sha256"]
    assert resumed["evidence_receipt"]["after_store"]["file_set_sha256"]

    from experiments.llm_sim_v2.evidence import build_phase_evidence_receipt

    phase_root = tmp_path / "llm-personas-v2-dual/pilot"
    phase = _write_runner_phase_stub(
        phase_root,
        phase="pilot",
        providers=("deepseek",),
        task_ids=(task.task_id,),
    )
    phase_receipt = build_phase_evidence_receipt(
        phase_root,
        phase_provenance=phase,
        tasks=(task,),
    )
    provider_anchor = phase_receipt["providers"]["deepseek"]
    assert (
        provider_anchor["evidence_chain_head_sha256"]
        == resumed["evidence_receipt"]["event_sha256"]
    )
    assert (
        provider_anchor["record_set_sha256"]
        == resumed["record_set"]["record_set_sha256"]
    )


def test_new_runner_rejects_legacy_response_record_without_raw_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.runner as runner_module
    from experiments.llm_sim_v2.runner import BudgetLedger, V2ProviderRunner

    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in runner_module.enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    first = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([_response(answer=task.correct_option)]),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    first.run_tasks([task])
    record_path = (
        tmp_path
        / "llm-personas-v2-dual/pilot/records/deepseek"
        / f"{task.task_id}.json"
    )
    legacy = json.loads(record_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = "yher.llm_sim_v2.response_record.v1"
    for attempt in legacy["attempts"]:
        attempt.pop("provider_response_received", None)
        attempt.pop("response_content", None)
        attempt.pop("response_content_utf8_bytes", None)
        attempt.pop("response_content_sha256", None)
    record_path.write_text(
        json.dumps(legacy, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="existing v2 record"):
        V2ProviderRunner(
            contract=contract,
            output_base=tmp_path,
            phase="pilot",
            provider="deepseek",
            transport=SequenceTransport([]),
            budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
            sleep=lambda _: None,
            random_value=lambda: 0.5,
        ).run_tasks([task])


def test_provider_runner_process_lock_prevents_duplicate_transport_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in runner_module.enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    context = multiprocessing.get_context("fork")
    started = context.Event()
    release = context.Event()
    results = context.Queue()
    call_log = tmp_path / "transport-calls.txt"
    first = context.Process(
        target=_runner_process_worker,
        args=(
            contract,
            task,
            str(tmp_path),
            str(call_log),
            started,
            release,
            results,
            True,
        ),
    )
    second = context.Process(
        target=_runner_process_worker,
        args=(
            contract,
            task,
            str(tmp_path),
            str(call_log),
            started,
            release,
            results,
            False,
        ),
    )
    first.start()
    assert started.wait(3)
    second.start()
    release.set()
    first.join(10)
    second.join(10)
    assert first.exitcode == second.exitcode == 0
    rows = [results.get(timeout=1), results.get(timeout=1)]
    assert sorted(row["transport_calls"] for row in rows) == [0, 1]
    assert sorted(row["receipt_calls"] for row in rows) == [0, 1]
    assert call_log.read_text(encoding="ascii").splitlines() == [
        str(first.pid)
    ]


def test_hard_crash_open_invocation_blocks_next_runner_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.runner as runner_module
    from experiments.llm_sim_v2.runner import BudgetLedger, V2ProviderRunner

    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(REPO_ROOT)
    task = next(
        task
        for task in runner_module.enumerate_tasks(contract, phase="pilot")
        if task.condition == "controlled"
    )
    phase_root = tmp_path / "llm-personas-v2-dual/pilot"
    context = multiprocessing.get_context("fork")
    crashed = context.Process(
        target=_crash_open_invocation_worker,
        args=(str(phase_root), task.task_id),
    )
    crashed.start()
    crashed.join(5)
    assert crashed.exitcode == 0

    transport = SequenceTransport([_response(answer=task.correct_option)])
    with pytest.raises(ValueError, match="unmatched invocation"):
        V2ProviderRunner(
            contract=contract,
            output_base=tmp_path,
            phase="pilot",
            provider="deepseek",
            transport=transport,
            budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
            sleep=lambda _: None,
            random_value=lambda: 0.5,
        ).run_tasks([task])
    assert transport.calls == []


def test_carried_forward_cost_ledger_is_separate_and_strictly_added(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.collect import reconcile_run_budget_ledger
    from experiments.llm_sim_v2.evidence import canonical_sha256

    prior_path = REPO_ROOT / "experiments/llm_sim_v2/prior_cost_ledger.json"
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)
    carried = {
        "schema_version": "yher.llm_sim_v2.carried_forward_cost_ledger.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "source_phase": "pilot",
        "source_record_set_sha256": "a" * 64,
        "source_phase_receipt_sha256": "b" * 64,
        "known_cost_yuan": 1.91386592,
        "unknown_cost_reserve_yuan": 0.0,
        "total_accounted_cost_yuan": 1.91386592,
    }
    carried["carried_forward_cost_ledger_sha256"] = canonical_sha256(carried)

    ledger = reconcile_run_budget_ledger(
        tmp_path,
        prior_cost_ledger=prior,
        carried_forward_cost_ledger=carried,
        soft_warning_yuan=300.0,
        hard_fuse_yuan=450.0,
    )

    assert prior_path.read_bytes() == prior_bytes
    assert ledger["prior_documented_cost_yuan"] == pytest.approx(0.65766321)
    assert ledger["carried_forward_known_cost_yuan"] == pytest.approx(1.91386592)
    assert ledger["carried_forward_unknown_reserve_yuan"] == 0.0
    assert ledger["carried_forward_total_accounted_cost_yuan"] == pytest.approx(
        1.91386592
    )
    assert ledger["total_accounted_cost_yuan"] == pytest.approx(2.57152913)
    assert (
        ledger["carried_forward_cost_ledger_sha256"]
        == carried["carried_forward_cost_ledger_sha256"]
    )


def test_carried_forward_cost_ledger_rejects_coordinated_total_drift(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.collect import reconcile_run_budget_ledger
    from experiments.llm_sim_v2.evidence import canonical_sha256

    prior = json.loads(
        (REPO_ROOT / "experiments/llm_sim_v2/prior_cost_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    carried = {
        "schema_version": "yher.llm_sim_v2.carried_forward_cost_ledger.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "source_phase": "pilot",
        "source_record_set_sha256": "a" * 64,
        "source_phase_receipt_sha256": "b" * 64,
        "known_cost_yuan": 1.0,
        "unknown_cost_reserve_yuan": 0.0,
        "total_accounted_cost_yuan": 2.0,
    }
    carried["carried_forward_cost_ledger_sha256"] = canonical_sha256(carried)

    with pytest.raises(ValueError, match="carried-forward cost"):
        reconcile_run_budget_ledger(
            tmp_path,
            prior_cost_ledger=prior,
            carried_forward_cost_ledger=carried,
            soft_warning_yuan=300.0,
            hard_fuse_yuan=450.0,
        )


def test_formal_carried_forward_cost_requires_exact_reviewed_anchor() -> None:
    from experiments.llm_sim_v2.collect import (
        verify_formal_carried_forward_cost_ledger,
    )
    from experiments.llm_sim_v2.evidence import canonical_sha256

    reviewed = json.loads(
        (
            REPO_ROOT
            / "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json"
        ).read_text(encoding="utf-8")
    )
    assert verify_formal_carried_forward_cost_ledger(reviewed)[
        "total_accounted_cost_yuan"
    ] == pytest.approx(1.91386592)

    with pytest.raises(ValueError, match="reviewed carried-forward"):
        verify_formal_carried_forward_cost_ledger(None)
    counterfeit = {
        "schema_version": "yher.llm_sim_v2.carried_forward_cost_ledger.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "source_phase": "pilot",
        "source_record_set_sha256": "a" * 64,
        "source_phase_receipt_sha256": "b" * 64,
        "known_cost_yuan": 0.0,
        "unknown_cost_reserve_yuan": 0.0,
        "total_accounted_cost_yuan": 0.0,
    }
    counterfeit["carried_forward_cost_ledger_sha256"] = canonical_sha256(
        counterfeit
    )
    with pytest.raises(ValueError, match="reviewed carried-forward"):
        verify_formal_carried_forward_cost_ledger(counterfeit)


def test_collection_parser_exposes_evidence_and_carried_cost_inputs() -> None:
    from experiments.llm_sim_v2.collect import build_parser

    parsed = build_parser().parse_args(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--phase",
            "pilot",
            "--carried-forward-cost-ledger",
            "legacy-cost.json",
            "--phase-receipt-output",
            "tracked-phase-anchor.json",
        ]
    )

    assert parsed.carried_forward_cost_ledger == "legacy-cost.json"
    assert parsed.phase_receipt_output == "tracked-phase-anchor.json"


def test_phase_provenance_binds_carried_forward_digest_and_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.runner as runner_module
    from experiments.llm_sim_v2.collect import resolve_collection_scope

    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(REPO_ROOT)
    head = _git_head()
    runtime = runner_module.build_runtime_task_manifest(
        contract,
        runtime_commit=head,
        frozen_at_utc="2026-07-15T12:00:00Z",
    )
    runtime_proof = {
        "ok": True,
        "runtime_task_manifest_sha256": runtime["runtime_task_manifest_sha256"],
        "runtime_commit": head,
        "git_proof": {
            "ok": True,
            "byte_identical": True,
            "commit": head,
        },
    }
    tasks = runner_module.enumerate_tasks(contract, phase="pilot")
    scope = resolve_collection_scope(
        frozen_providers=contract.config["pilot"]["providers"],
        requested_providers=None,
        limit=None,
        allow_partial=False,
    )
    carried = json.loads(
        (
            REPO_ROOT
            / "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json"
        ).read_text(encoding="utf-8")
    )

    phase = runner_module.build_phase_provenance(
        contract,
        runtime_manifest=runtime,
        runtime_proof=runtime_proof,
        phase="pilot",
        tasks=tasks,
        collection_scope=scope,
        prior_cost_ledger=contract.prior_cost_ledger,
        carried_forward_cost=carried,
        first_observation_at_utc="2026-07-15T12:00:01Z",
    )

    assert (
        phase["budget"]["carried_forward_cost_ledger_sha256"]
        == "87ff7a3d08df64d67df2372188d6f707ddb4deb295a85300aae0b6190d48be35"
    )
    assert (
        phase["budget"]["source_phase_receipt_sha256"]
        == "2ea161a0f29e3a8bac1eeee38cb238f7cb722a38b809997172b933e063e82999"
    )
    assert (
        phase["budget"]["source_record_set_sha256"]
        == "1490e7d35410717614ba9fd2d37e1eda12d824fa247bff29ae29ff9a0b6c58c0"
    )
    assert phase["budget"]["carried_forward_known_cost_yuan"] == pytest.approx(
        1.91386592
    )
    assert phase["budget"]["carried_forward_total_accounted_cost_yuan"] == (
        pytest.approx(1.91386592)
    )

    with pytest.raises(ValueError, match="reviewed carried-forward"):
        runner_module.build_phase_provenance(
            contract,
            runtime_manifest=runtime,
            runtime_proof=runtime_proof,
            phase="pilot",
            tasks=tasks,
            collection_scope=scope,
            prior_cost_ledger=contract.prior_cost_ledger,
            carried_forward_cost=None,
            first_observation_at_utc="2026-07-15T12:00:01Z",
        )


def test_unavailable_provider_emits_zero_call_evidence_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.runner as runner_module
    from experiments.llm_sim_v2.evidence import build_phase_evidence_receipt
    from experiments.llm_sim_v2.runner import BudgetLedger, V2ProviderRunner

    monkeypatch.setattr(
        runner_module,
        "RUNTIME_MANIFEST_REL",
        Path("experiments/llm_sim_v2/runtime_task_manifest.not-present.json"),
    )
    contract = runner_module.load_runtime_contract(REPO_ROOT)
    task = runner_module.enumerate_tasks(contract, phase="pilot")[0]
    summary = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=None,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
    ).write_unavailable_manifest(
        [task],
        error_category="transport_configuration_or_credential_unavailable",
    )

    assert summary["evidence_receipt"]["provider_call_count"] == 0
    assert summary["evidence_receipt"]["status"] == "unavailable"
    phase_root = tmp_path / "llm-personas-v2-dual/pilot"
    phase = _write_runner_phase_stub(
        phase_root,
        phase="pilot",
        providers=("deepseek",),
        task_ids=(task.task_id,),
    )
    phase_receipt = build_phase_evidence_receipt(
        phase_root,
        phase_provenance=phase,
        tasks=(task,),
    )
    assert (
        phase_receipt["providers"]["deepseek"]["evidence_chain_head_sha256"]
        == summary["evidence_receipt"]["event_sha256"]
    )


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
    import experiments.llm_sim_v2.runner as runner_module

    original_manifest = runner_module.RUNTIME_MANIFEST_REL
    runner_module.RUNTIME_MANIFEST_REL = Path(
        "experiments/llm_sim_v2/runtime_task_manifest.not-present.json"
    )
    try:
        contract = runner_module.load_runtime_contract(REPO_ROOT)
    finally:
        runner_module.RUNTIME_MANIFEST_REL = original_manifest
    runtime_commit = str(contract.freeze_proof["commit"])
    runtime = runner_module.build_runtime_task_manifest(
        contract,
        runtime_commit=runtime_commit,
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
            "commit_timestamp_utc": _git_commit_timestamp(runtime_commit),
            "observation_timestamp": contract.freeze_proof[
                "observation_timestamp"
            ],
            "precedes_observation": True,
            "byte_identical": True,
            "files": [
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "byte_identical": True,
                }
                for row in runtime["runtime_files"]
            ],
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
    tasks = runner_module.enumerate_tasks(contract, phase="pilot")
    if partial:
        tasks = [next(task for task in tasks if task.condition == "controlled")]
    carried = None
    if not partial:
        carried = json.loads(
            (
                REPO_ROOT
                / "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json"
            ).read_text(encoding="utf-8")
        )
    phase = runner_module.build_phase_provenance(
        contract,
        runtime_manifest=runtime,
        runtime_proof=proof,
        phase="pilot",
        tasks=tasks,
        collection_scope=scope,
        prior_cost_ledger=contract.prior_cost_ledger,
        carried_forward_cost=carried,
        first_observation_at_utc=contract.freeze_proof[
            "observation_timestamp"
        ],
    )
    return contract, runtime, tasks, phase


def _reviewed_carried() -> dict[str, Any]:
    return json.loads(
        (
            REPO_ROOT
            / "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json"
        ).read_text(encoding="utf-8")
    )


def _active_runtime_proof(
    runtime: dict[str, Any],
    phase: dict[str, Any],
) -> dict[str, Any]:
    git_proof = copy.deepcopy(phase["runtime"]["git_proof"])
    git_proof["current_head"] = _git_head()
    git_proof["commit_timestamp_utc"] = _git_commit_timestamp(
        runtime["runtime_commit"]
    )
    return {
        "schema_version": "yher.llm_sim_v2.runtime_task_manifest_proof.v1",
        "ok": True,
        "run_id": "llm-personas-v2-dual",
        "runtime_task_manifest_sha256": runtime[
            "runtime_task_manifest_sha256"
        ],
        "runtime_commit": runtime["runtime_commit"],
        "git_proof": git_proof,
    }


def _phase_scope(phase: dict[str, Any]) -> dict[str, Any]:
    return {
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


def _rehash_phase(phase: dict[str, Any]) -> None:
    phase.pop("phase_provenance_sha256", None)
    phase["phase_provenance_sha256"] = _canonical_sha(phase)


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
    assert phase["runtime"]["execution_commit"] == runtime["runtime_commit"]
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
    runtime_proof = _active_runtime_proof(runtime, phase)
    scope = _phase_scope(phase)
    assert verify_phase_provenance_against_contract(
        phase,
        contract=contract,
        runtime_manifest=runtime,
        runtime_proof=runtime_proof,
        tasks=tasks,
        collection_scope=scope,
        carried_forward_cost=_reviewed_carried(),
    )["ok"] is True

    tampered = copy.deepcopy(phase)
    tampered[section][field] = "0" * 64
    _rehash_phase(tampered)
    with pytest.raises(ValueError, match="phase provenance.*contract"):
        verify_phase_provenance_against_contract(
            tampered,
            contract=contract,
            runtime_manifest=runtime,
            runtime_proof=runtime_proof,
            tasks=tasks,
            collection_scope=scope,
            carried_forward_cost=_reviewed_carried(),
        )


def test_rehashed_phase_rejects_coordinated_precommit_observation_timestamps():
    from experiments.llm_sim_v2.runner import (
        verify_phase_provenance_against_contract,
    )

    contract, runtime, tasks, phase = _phase_provenance_fixture()
    runtime_proof = _active_runtime_proof(runtime, phase)
    tampered = copy.deepcopy(phase)
    ancient = "2000-01-01T00:00:00Z"
    tampered["first_observation_at_utc"] = ancient
    tampered["freeze"]["git_proof"]["observation_timestamp"] = ancient
    tampered["runtime"]["git_proof"]["observation_timestamp"] = ancient
    tampered["freeze"]["git_proof"]["precedes_observation"] = True
    tampered["runtime"]["git_proof"]["precedes_observation"] = True
    _rehash_phase(tampered)

    with pytest.raises(ValueError, match="commit|observation|temporal"):
        verify_phase_provenance_against_contract(
            tampered,
            contract=contract,
            runtime_manifest=runtime,
            runtime_proof=runtime_proof,
            tasks=tasks,
            collection_scope=_phase_scope(phase),
            carried_forward_cost=_reviewed_carried(),
        )


@pytest.mark.parametrize("binding", ["freeze", "runtime"])
def test_rehashed_phase_rejects_stored_commit_timestamp_drift(binding: str):
    from experiments.llm_sim_v2.runner import (
        verify_phase_provenance_against_contract,
    )

    contract, runtime, tasks, phase = _phase_provenance_fixture()
    runtime_proof = _active_runtime_proof(runtime, phase)
    tampered = copy.deepcopy(phase)
    tampered[binding]["git_proof"]["commit_timestamp_utc"] = (
        "1999-01-01T00:00:00+00:00"
    )
    _rehash_phase(tampered)

    with pytest.raises(ValueError, match="commit timestamp|git proof|temporal"):
        verify_phase_provenance_against_contract(
            tampered,
            contract=contract,
            runtime_manifest=runtime,
            runtime_proof=runtime_proof,
            tasks=tasks,
            collection_scope=_phase_scope(phase),
            carried_forward_cost=_reviewed_carried(),
        )


def test_rehashed_phase_rejects_impossible_stored_runtime_head_ancestry():
    from experiments.llm_sim_v2.runner import (
        FROZEN_COMMIT,
        verify_phase_provenance_against_contract,
    )

    contract, runtime, tasks, phase = _phase_provenance_fixture()
    runtime_proof = _active_runtime_proof(runtime, phase)
    tampered = copy.deepcopy(phase)
    tampered["runtime"]["git_proof"]["current_head"] = _git_parent(
        FROZEN_COMMIT
    )
    _rehash_phase(tampered)

    with pytest.raises(ValueError, match="current_head|ancestor|ancestry"):
        verify_phase_provenance_against_contract(
            tampered,
            contract=contract,
            runtime_manifest=runtime,
            runtime_proof=runtime_proof,
            tasks=tasks,
            collection_scope=_phase_scope(phase),
            carried_forward_cost=_reviewed_carried(),
        )


def test_stored_runtime_head_may_precede_active_head_when_ancestry_is_valid():
    from experiments.llm_sim_v2.runner import (
        verify_phase_provenance_against_contract,
    )

    contract, runtime, tasks, phase = _phase_provenance_fixture()
    assert phase["runtime"]["git_proof"]["current_head"] == runtime[
        "runtime_commit"
    ]
    assert runtime["runtime_commit"] != _git_head()
    assert verify_phase_provenance_against_contract(
        phase,
        contract=contract,
        runtime_manifest=runtime,
        runtime_proof=_active_runtime_proof(runtime, phase),
        tasks=tasks,
        collection_scope=_phase_scope(phase),
        carried_forward_cost=_reviewed_carried(),
    )["contract_revalidated"] is True


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
    assert binding["execution_commit"] == phase["runtime"]["execution_commit"]
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
    )

    contract, _, tasks, phase = _phase_provenance_fixture(partial=True)
    task = tasks[0]
    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([KeyboardInterrupt()]),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        phase_provenance=phase,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run_tasks([task])

    path = tmp_path / "llm-personas-v2-dual/pilot/provider_manifests/deepseek.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["provider_lifecycle"] == "interrupted"
    assert manifest["lifecycle"]["expected_count"] == 1
    assert manifest["lifecycle"]["present_count"] == 1
    assert manifest["lifecycle"]["missing_count"] == 0
    assert manifest["lifecycle"]["interrupted_count"] == 0
    assert manifest["needs_user"] == {
        "required": True,
        "reason": "unknown_provider_billing_reserved",
        "record_count": 1,
        "record_task_ids": [task.task_id],
        "unknown_cost_attempt_count": 1,
    }
    assert manifest["budget"]["provider_record_unknown_reserve_yuan"] == 10.0


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


def test_interrupt_after_transport_entry_persists_reserve_and_zero_call_resume(
    tmp_path: Path,
):
    from experiments.llm_sim_v2.evidence import build_phase_evidence_receipt
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        write_phase_provenance,
    )

    contract, _, tasks, phase = _phase_provenance_fixture(partial=True)
    task = tasks[0]
    write_phase_provenance(tmp_path, phase=phase)
    interrupted = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=SequenceTransport([KeyboardInterrupt()]),
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        phase_provenance=phase,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    with pytest.raises(KeyboardInterrupt):
        interrupted.run_tasks([task])

    record_path = (
        tmp_path
        / "llm-personas-v2-dual/pilot/records/deepseek"
        / f"{task.task_id}.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "technical_failure"
    assert record["attempts"] == [
        {
            "attempt": 1,
            "status": "failed",
            "error_category": "interrupted_provider_call",
            "request_max_tokens": 1024,
            "cost_yuan": None,
            "cost_known": False,
            "billing_ambiguity": True,
            "cost_reserve_yuan": 10.0,
            "provider_response_received": False,
        }
    ]
    assert record["unknown_cost_reserve_yuan"] == 10.0
    assert record["needs_user"] is True

    resume_transport = SequenceTransport(
        [
            _response(
                answer=task.correct_option,
                blind=task.condition == "blind",
            )
        ]
    )
    resumed = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=resume_transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        phase_provenance=phase,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    resumed.run_tasks([task])
    assert resume_transport.calls == []
    receipt = build_phase_evidence_receipt(
        tmp_path / "llm-personas-v2-dual/pilot",
        phase_provenance=phase,
        tasks=tasks,
    )
    assert receipt["providers"]["deepseek"]["provider_call_count"] == 1


def test_resume_rejects_valid_response_relabelled_technical_failure(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        execute_task,
    )

    contract, _, tasks, phase = _phase_provenance_fixture(partial=True)
    task = tasks[0]
    resume_transport = SequenceTransport([])
    runner = V2ProviderRunner(
        contract=contract,
        output_base=tmp_path,
        phase="pilot",
        provider="deepseek",
        transport=resume_transport,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        phase_provenance=phase,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    record = execute_task(
        task,
        provider="deepseek",
        model=runner.model,
        transport=SequenceTransport(
            [
                _response(
                    answer=task.correct_option,
                    blind=task.condition == "blind",
                )
            ]
        ),
        policy=runner.policy,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        provenance=runner.provenance,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    record.update(
        {
            "status": "technical_failure",
            "error": "network_timeout",
            "parsed_output": None,
            "outcomes": {
                "is_correct": None,
                "target_option_hit": None,
                "manipulation_compliance": None,
            },
        }
    )
    path = runner._record_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="status semantics"):
        runner.run_tasks(tasks)
    assert resume_transport.calls == []


def test_invalid_schema_then_network_failure_uses_final_attempt_status() -> None:
    from dataclasses import replace

    from experiments.llm_sim.transport import ProviderNetworkError
    from experiments.llm_sim_v2.evidence import validate_v2_response_record
    from experiments.llm_sim_v2.runner import BudgetLedger, execute_task

    contract, _, tasks, _ = _phase_provenance_fixture(partial=True)
    task = tasks[0]
    invalid_response = _response(answer=task.correct_option)
    invalid_response["content"] = "not-json"
    policy = replace(contract.provider_policy("deepseek"), max_attempts=2)
    record = execute_task(
        task,
        provider="deepseek",
        model=contract.provider_model("deepseek"),
        transport=SequenceTransport(
            [invalid_response, ProviderNetworkError()]
        ),
        policy=policy,
        budget=BudgetLedger(soft_warning_yuan=300.0, hard_fuse_yuan=450.0),
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert record["status"] == "technical_failure"
    assert record["attempts"][-1]["provider_response_received"] is False
    validate_v2_response_record(
        record,
        provider="deepseek",
        requested_model=contract.provider_model("deepseek"),
        phase="pilot",
        task=task,
        expected_provenance={},
    )


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
