"""Append-only evidence receipts for Persona-v2 collection."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path


RUN_ID = "llm-personas-v2-dual"


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_phase_stub(
    phase_root: Path,
    *,
    providers: tuple[str, ...],
    task_ids: tuple[str, ...],
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.phase_provenance.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": phase_root.name,
        "analysis_population": phase_root.name,
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
    artifact["phase_provenance_sha256"] = _sha(artifact)
    phase_root.mkdir(parents=True, exist_ok=True)
    (phase_root / "phase_provenance.json").write_text(
        json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def _task_contract(task_id: str = "a") -> dict[str, object]:
    return {
        "task_id": task_id,
        "logical_key": f"logical-{task_id}",
        "phase": "pilot",
        "analysis_population": "pilot",
        "condition": "controlled",
        "response_arm": "control",
        "option_keys": ["A", "B"],
        "correct_option": "A",
        "target_option": "B",
        "requested_model": "frozen-model",
        "wire_message_sha256": "a" * 64,
    }


def _v2_record(task: dict[str, object]) -> dict[str, object]:
    content = json.dumps(
        {"simulated": True, "answer": "A", "rationale": "synthetic"},
        sort_keys=True,
    )
    attempt = {
        "attempt": 1,
        "status": "response",
        "request_max_tokens": 1024,
        "model_returned": task["requested_model"],
        "finish_reason": "stop",
        "usage": {"input_tokens": 2, "output_tokens": 3},
        "cost_yuan": 0.01,
        "cost_known": True,
        "billing_ambiguity": False,
        "cost_reserve_yuan": 0.0,
        "provider_response_received": True,
        "response_content": content,
        "response_content_utf8_bytes": len(content.encode("utf-8")),
        "response_content_sha256": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
    }
    return {
        "schema_version": "yher.llm_sim_v2.response_record.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": task["phase"],
        "analysis_population": task["analysis_population"],
        "provider": "deepseek",
        "requested_model": task["requested_model"],
        "model_id": task["requested_model"],
        "task_id": task["task_id"],
        "logical_key": task["logical_key"],
        "condition": task["condition"],
        "response_arm": task["response_arm"],
        "wire_message_sha256": task["wire_message_sha256"],
        "status": "complete",
        "error": None,
        "parsed_output": {
            "simulated": True,
            "answer": "A",
            "rationale": "synthetic",
        },
        "outcomes": {
            "is_correct": True,
            "target_option_hit": False,
            "manipulation_compliance": True,
        },
        "attempts": [attempt],
        "retry_count": 0,
        "known_cost_yuan": 0.01,
        "unknown_cost_reserve_yuan": 0.0,
        "cost_yuan": 0.01,
        "has_unknown_cost_attempts": False,
        "needs_user": False,
        "needs_user_reasons": [],
        "provenance": {},
    }


def _write_manifest(
    phase_root: Path,
    *,
    provider: str,
    expected_task_ids: tuple[str, ...],
) -> None:
    from experiments.llm_sim_v2.evidence import build_provider_record_set

    (phase_root / "records" / provider).mkdir(parents=True, exist_ok=True)
    record_set = build_provider_record_set(
        phase_root,
        provider=provider,
        expected_task_ids=expected_task_ids,
    )
    path = phase_root / "provider_manifests" / f"{provider}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "provider": provider,
                "phase": phase_root.name,
                "requested_model": "frozen-model",
                "lifecycle": {"expected_task_ids": list(expected_task_ids)},
                "record_set": record_set,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _multiprocess_lock_worker(
    phase_root_text: str,
    started: object,
    release: object,
    results: object,
    first: bool,
) -> None:
    from experiments.llm_sim_v2.evidence import ProviderEvidenceLedger

    phase_root = Path(phase_root_text)
    record_path = phase_root / "records/deepseek/a.json"
    ledger = ProviderEvidenceLedger(
        phase_root,
        run_id=RUN_ID,
        phase="pilot",
        provider="deepseek",
    )
    with ledger.provider_lock():
        resumed = record_path.is_file()
        invocation = ledger.begin_invocation(
            expected_task_ids=("a",),
            resumed_task_ids=("a",) if resumed else (),
        )
        calls = 0
        if not resumed:
            calls = 1
            ledger.record_provider_call_started(
                task_id="a",
                attempt=1,
                model="frozen-model",
                request_max_tokens=1024,
                wire_message_sha256="a" * 64,
            )
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(
                json.dumps(
                    {
                        "task_id": "a",
                        "requested_model": "frozen-model",
                        "wire_message_sha256": "a" * 64,
                        "attempts": [
                            {"attempt": 1, "request_max_tokens": 1024}
                        ],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if first:
                started.set()
                release.wait(5)
        ledger.finish_invocation(invocation, status="complete")
    results.put(calls)


def _write_record(path: Path, task_id: str, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"task_id": task_id, "answer": answer}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_provider_record_set_binds_exact_record_bytes(tmp_path: Path) -> None:
    from experiments.llm_sim_v2.evidence import build_provider_record_set

    phase_root = tmp_path / RUN_ID / "pilot"
    first = phase_root / "records/deepseek/a.json"
    second = phase_root / "records/deepseek/b.json"
    _write_record(first, "a", "A")
    _write_record(second, "b", "B")

    original = build_provider_record_set(
        phase_root,
        provider="deepseek",
        expected_task_ids=("a", "b", "c"),
    )

    assert original["record_count"] == 2
    assert original["missing_task_ids"] == ["c"]
    assert original["records"][0] == {
        "attempt_count": 0,
        "bytes": first.stat().st_size,
        "path": "records/deepseek/a.json",
        "response_attempt_count": 0,
        "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "task_id": "a",
    }
    assert len(original["record_set_sha256"]) == 64

    _write_record(first, "a", "C")
    changed = build_provider_record_set(
        phase_root,
        provider="deepseek",
        expected_task_ids=("a", "b", "c"),
    )
    assert changed["record_set_sha256"] != original["record_set_sha256"]


def test_resume_receipt_persists_zero_actual_calls_and_hash_chain(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.evidence import ProviderEvidenceLedger

    phase_root = tmp_path / RUN_ID / "pilot"
    _write_record(phase_root / "records/deepseek/a.json", "a", "A")
    ledger = ProviderEvidenceLedger(
        phase_root,
        run_id=RUN_ID,
        phase="pilot",
        provider="deepseek",
    )

    invocation = ledger.begin_invocation(
        expected_task_ids=("a",),
        resumed_task_ids=("a",),
    )
    (phase_root / "provider_manifests").mkdir(parents=True)
    (phase_root / "provider_manifests/deepseek.json").write_text(
        "{}\n", encoding="utf-8"
    )
    receipt = ledger.finish_invocation(invocation, status="complete")
    events = ledger.read_events()

    assert receipt["invocation_kind"] == "resume"
    assert receipt["provider_call_count"] == 0
    assert receipt["before_store"]["file_set_sha256"]
    assert receipt["after_store"]["file_set_sha256"]
    assert (
        receipt["before_store"]["file_set_sha256"]
        != receipt["after_store"]["file_set_sha256"]
    )
    assert [event["event_type"] for event in events] == [
        "invocation_started",
        "invocation_finished",
    ]
    assert events[0]["previous_event_sha256"] is None
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]
    assert events[1]["provider_call_count"] == 0


def test_public_event_history_rejects_unmatched_invocation(tmp_path: Path) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import ProviderEvidenceLedger

    phase_root = tmp_path / RUN_ID / "pilot"
    ledger = ProviderEvidenceLedger(
        phase_root,
        run_id=RUN_ID,
        phase="pilot",
        provider="deepseek",
    )
    ledger.begin_invocation(expected_task_ids=("a",), resumed_task_ids=())

    with pytest.raises(ValueError, match="unmatched invocation"):
        ledger.read_events()


def test_call_started_event_is_counted_in_invocation_receipt(tmp_path: Path) -> None:
    from experiments.llm_sim_v2.evidence import ProviderEvidenceLedger

    phase_root = tmp_path / RUN_ID / "main"
    ledger = ProviderEvidenceLedger(
        phase_root,
        run_id=RUN_ID,
        phase="main",
        provider="doubao",
    )
    invocation = ledger.begin_invocation(
        expected_task_ids=("task-1",),
        resumed_task_ids=(),
    )
    ledger.record_provider_call_started(
        task_id="task-1",
        attempt=1,
        model="frozen-model",
        request_max_tokens=1024,
        wire_message_sha256="a" * 64,
    )
    receipt = ledger.finish_invocation(invocation, status="complete")

    assert receipt["provider_call_count"] == 1
    assert receipt["provider_call_event_sha256s"] == [
        ledger.read_events()[1]["event_sha256"]
    ]


def test_phase_receipt_is_deterministic_and_git_anchor_ready(tmp_path: Path) -> None:
    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
        build_provider_record_set,
        write_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    task = _task_contract()
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    record = phase_root / "records/deepseek/a.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(_v2_record(task), sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    record_set = build_provider_record_set(
        phase_root, provider="deepseek", expected_task_ids=("a",)
    )
    ledger = ProviderEvidenceLedger(
        phase_root,
        run_id=RUN_ID,
        phase="pilot",
        provider="deepseek",
    )
    invocation = ledger.begin_invocation(
        expected_task_ids=("a",), resumed_task_ids=()
    )
    ledger.record_provider_call_started(
        task_id="a",
        attempt=1,
        model="frozen-model",
        request_max_tokens=1024,
        wire_message_sha256="a" * 64,
    )
    ledger.finish_invocation(invocation, status="complete")

    first = build_phase_evidence_receipt(
        phase_root, phase_provenance=phase, tasks=(task,)
    )
    second = build_phase_evidence_receipt(
        phase_root, phase_provenance=phase, tasks=(task,)
    )
    output = tmp_path / "tracked-anchor.json"
    written = write_phase_evidence_receipt(
        phase_root,
        output=output,
        phase_provenance=phase,
        tasks=(task,),
    )

    assert first == second == written
    assert first["run_id"] == RUN_ID
    assert first["phase"] == "pilot"
    assert first["authority"] == "post_invocation_phase_receipt"
    assert (
        first["providers"]["deepseek"]["record_set_sha256"]
        == record_set["record_set_sha256"]
    )
    assert first["providers"]["deepseek"]["evidence_chain_head_sha256"]
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert len(first["phase_evidence_receipt_sha256"]) == 64


def test_phase_receipt_recomputes_provider_record_set_before_anchoring(
    tmp_path: Path,
) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
        build_provider_record_set,
    )

    phase_root = tmp_path / RUN_ID / "main"
    task = _task_contract()
    task.update({"phase": "main", "analysis_population": "main"})
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    record = phase_root / "records/deepseek/a.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(_v2_record(task), sort_keys=True) + "\n", encoding="utf-8"
    )
    record_set = build_provider_record_set(
        phase_root,
        provider="deepseek",
        expected_task_ids=("a",),
    )
    manifest = phase_root / "provider_manifests/deepseek.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "provider": "deepseek",
                "phase": "main",
                "requested_model": "frozen-model",
                "lifecycle": {"expected_task_ids": ["a"]},
                "record_set": record_set,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = ProviderEvidenceLedger(
        phase_root,
        run_id=RUN_ID,
        phase="main",
        provider="deepseek",
    )
    invocation = ledger.begin_invocation(
        expected_task_ids=("a",), resumed_task_ids=()
    )
    ledger.record_provider_call_started(
        task_id="a",
        attempt=1,
        model="frozen-model",
        request_max_tokens=1024,
        wire_message_sha256="a" * 64,
    )
    ledger.finish_invocation(invocation, status="complete")
    changed = _v2_record(task)
    changed["parsed_output"] = {
        "simulated": True,
        "answer": "B",
        "rationale": "changed",
    }
    record.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="record-set binding"):
        build_phase_evidence_receipt(
            phase_root, phase_provenance=phase, tasks=(task,)
        )


def test_phase_receipt_rejects_provider_root_set_drift(tmp_path: Path) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=()
    )
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=())
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    invocation = ledger.begin_invocation(expected_task_ids=(), resumed_task_ids=())
    ledger.finish_invocation(invocation, status="unavailable")
    (phase_root / "records/doubao").mkdir(parents=True)

    with pytest.raises(ValueError, match="provider set"):
        build_phase_evidence_receipt(
            phase_root, phase_provenance=phase, tasks=()
        )


def test_phase_receipt_validates_unmanifested_event_provider(tmp_path: Path) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    (phase_root / "records/deepseek").mkdir(parents=True)
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    ledger.begin_invocation(expected_task_ids=("a",), resumed_task_ids=())

    with pytest.raises(ValueError, match="unmatched invocation"):
        build_phase_evidence_receipt(
            phase_root,
            phase_provenance=phase,
            tasks=(_task_contract(),),
        )


def test_phase_receipt_rejects_v1_even_when_manifest_binds_it(
    tmp_path: Path,
) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    record = phase_root / "records/deepseek/a.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps({"schema_version": "yher.llm_sim_v2.response_record.v1", "task_id": "a", "attempts": []}) + "\n",
        encoding="utf-8",
    )
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    invocation = ledger.begin_invocation(
        expected_task_ids=("a",), resumed_task_ids=("a",)
    )
    ledger.finish_invocation(invocation, status="complete")

    with pytest.raises(ValueError, match="response_record.v2"):
        build_phase_evidence_receipt(
            phase_root,
            phase_provenance=phase,
            tasks=(_task_contract(),),
        )


def test_phase_receipt_replays_raw_content_and_reconciles_call_events(
    tmp_path: Path,
) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    task = _task_contract()
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    record = _v2_record(task)
    record["parsed_output"] = {
        "simulated": True,
        "answer": "B",
        "rationale": "tampered",
    }
    path = phase_root / "records/deepseek/a.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    invocation = ledger.begin_invocation(expected_task_ids=("a",), resumed_task_ids=())
    ledger.record_provider_call_started(
        task_id="a",
        attempt=1,
        model="frozen-model",
        request_max_tokens=1024,
        wire_message_sha256="a" * 64,
    )
    ledger.finish_invocation(invocation, status="complete")

    with pytest.raises(ValueError, match="strict replay"):
        build_phase_evidence_receipt(
            phase_root, phase_provenance=phase, tasks=(task,)
        )


def test_phase_receipt_rejects_call_attempt_identity_mismatch(
    tmp_path: Path,
) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    task = _task_contract()
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    path = phase_root / "records/deepseek/a.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(_v2_record(task), sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    invocation = ledger.begin_invocation(expected_task_ids=("a",), resumed_task_ids=())
    ledger.record_provider_call_started(
        task_id="a",
        attempt=1,
        model="wrong-model",
        request_max_tokens=1024,
        wire_message_sha256="a" * 64,
    )
    ledger.finish_invocation(invocation, status="complete")

    with pytest.raises(ValueError, match="call.*attempt"):
        build_phase_evidence_receipt(
            phase_root, phase_provenance=phase, tasks=(task,)
        )


def test_phase_receipt_rejects_noncomplete_record_semantic_drift(
    tmp_path: Path,
) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    task = _task_contract()
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    record = _v2_record(task)
    record["status"] = "technical_failure"
    record["error"] = "network_timeout"
    path = phase_root / "records/deepseek/a.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    invocation = ledger.begin_invocation(expected_task_ids=("a",), resumed_task_ids=())
    ledger.record_provider_call_started(
        task_id="a",
        attempt=1,
        model="frozen-model",
        request_max_tokens=1024,
        wire_message_sha256="a" * 64,
    )
    ledger.finish_invocation(invocation, status="complete")

    with pytest.raises(ValueError, match="record status semantics"):
        build_phase_evidence_receipt(
            phase_root, phase_provenance=phase, tasks=(task,)
        )


def test_phase_receipt_rejects_valid_raw_json_relabelled_excluded_schema(
    tmp_path: Path,
) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    task = _task_contract()
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    record = _v2_record(task)
    record.update(
        {
            "status": "excluded_schema",
            "error": "invalid_schema",
            "parsed_output": None,
            "outcomes": {
                "is_correct": None,
                "target_option_hit": None,
                "manipulation_compliance": None,
            },
        }
    )
    record["attempts"][-1].update(
        {"status": "failed", "error_category": "invalid_schema"}
    )
    path = phase_root / "records/deepseek/a.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    invocation = ledger.begin_invocation(expected_task_ids=("a",), resumed_task_ids=())
    ledger.record_provider_call_started(
        task_id="a",
        attempt=1,
        model="frozen-model",
        request_max_tokens=1024,
        wire_message_sha256="a" * 64,
    )
    ledger.finish_invocation(invocation, status="complete")

    with pytest.raises(ValueError, match="invalid-schema.*strict parser"):
        build_phase_evidence_receipt(
            phase_root, phase_provenance=phase, tasks=(task,)
        )


def test_phase_receipt_rejects_successful_response_followed_by_retry(
    tmp_path: Path,
) -> None:
    import copy
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    task = _task_contract()
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    record = _v2_record(task)
    first_attempt = copy.deepcopy(record["attempts"][0])
    second_attempt = copy.deepcopy(first_attempt)
    second_attempt["attempt"] = 2
    record["attempts"] = [first_attempt, second_attempt]
    record["retry_count"] = 1
    record["known_cost_yuan"] = 0.02
    record["cost_yuan"] = 0.02
    path = phase_root / "records/deepseek/a.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    invocation = ledger.begin_invocation(expected_task_ids=("a",), resumed_task_ids=())
    for attempt in (1, 2):
        ledger.record_provider_call_started(
            task_id="a",
            attempt=attempt,
            model="frozen-model",
            request_max_tokens=1024,
            wire_message_sha256="a" * 64,
        )
    ledger.finish_invocation(invocation, status="complete")

    with pytest.raises(ValueError, match="successful response must be final"):
        build_phase_evidence_receipt(
            phase_root, phase_provenance=phase, tasks=(task,)
        )


def test_phase_receipt_rejects_nonresponse_attempt_retaining_raw_content(
    tmp_path: Path,
) -> None:
    import pytest

    from experiments.llm_sim_v2.evidence import (
        ProviderEvidenceLedger,
        build_phase_evidence_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    task = _task_contract()
    phase = _write_phase_stub(
        phase_root, providers=("deepseek",), task_ids=("a",)
    )
    record = _v2_record(task)
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
    record["attempts"][-1].update(
        {
            "status": "failed",
            "error_category": "network_timeout",
            "provider_response_received": False,
        }
    )
    path = phase_root / "records/deepseek/a.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    _write_manifest(phase_root, provider="deepseek", expected_task_ids=("a",))
    ledger = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    )
    invocation = ledger.begin_invocation(expected_task_ids=("a",), resumed_task_ids=())
    ledger.record_provider_call_started(
        task_id="a",
        attempt=1,
        model="frozen-model",
        request_max_tokens=1024,
        wire_message_sha256="a" * 64,
    )
    ledger.finish_invocation(invocation, status="complete")

    with pytest.raises(ValueError, match="non-response attempt retains raw content"):
        build_phase_evidence_receipt(
            phase_root, phase_provenance=phase, tasks=(task,)
        )


def test_provider_lock_serializes_processes_and_loser_makes_zero_calls(
    tmp_path: Path,
) -> None:
    phase_root = tmp_path / RUN_ID / "pilot"
    context = multiprocessing.get_context("fork")
    started = context.Event()
    release = context.Event()
    results = context.Queue()
    first = context.Process(
        target=_multiprocess_lock_worker,
        args=(str(phase_root), started, release, results, True),
    )
    second = context.Process(
        target=_multiprocess_lock_worker,
        args=(str(phase_root), started, release, results, False),
    )
    first.start()
    assert started.wait(3)
    second.start()
    release.set()
    first.join(5)
    second.join(5)
    assert first.exitcode == second.exitcode == 0
    assert sorted([results.get(timeout=1), results.get(timeout=1)]) == [0, 1]

    from experiments.llm_sim_v2.evidence import ProviderEvidenceLedger

    events = ProviderEvidenceLedger(
        phase_root, run_id=RUN_ID, phase="pilot", provider="deepseek"
    ).read_events()
    assert [event["event_type"] for event in events] == [
        "invocation_started",
        "provider_call_started",
        "invocation_finished",
        "invocation_started",
        "invocation_finished",
    ]
    assert sum(event["event_type"] == "provider_call_started" for event in events) == 1


def test_legacy_receipt_and_carried_ledger_are_generated_from_bound_sources(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.collect import verify_carried_forward_cost_ledger
    from experiments.llm_sim_v2.evidence import (
        build_carried_forward_cost_ledger,
        build_phase_source_file_set,
        build_retrospective_legacy_receipt,
        canonical_sha256,
        validate_retrospective_legacy_receipt,
        write_carried_forward_cost_ledger,
        write_retrospective_legacy_receipt,
    )

    phase_root = tmp_path / RUN_ID / "pilot"
    per_provider_cost = 0.95693296
    for provider, task_id in (("deepseek", "a"), ("doubao", "b")):
        record_path = phase_root / "records" / provider / f"{task_id}.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "schema_version": "yher.llm_sim_v2.response_record.v1",
                    "task_id": task_id,
                    "attempts": [
                        {
                            "attempt": 1,
                            "status": "response",
                            "cost_known": True,
                            "cost_yuan": per_provider_cost,
                            "cost_reserve_yuan": 0.0,
                            "billing_ambiguity": False,
                        }
                    ],
                    "known_cost_yuan": per_provider_cost,
                    "unknown_cost_reserve_yuan": 0.0,
                    "cost_yuan": per_provider_cost,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path = phase_root / "provider_manifests" / f"{provider}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "provider": provider,
                    "lifecycle": {"expected_task_ids": [task_id]},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    source_set = build_phase_source_file_set(phase_root)
    audit = {
        "schema_version": "yher.llm_sim_v2.formal_pilot_audit.v1",
        "decision": "GO",
        "source_binding": {
            "pilot_source_files": source_set["files"],
            "pilot_source_file_count": source_set["file_count"],
            "pilot_source_set_sha256": source_set["file_set_sha256"],
        },
        "accounting": {
            "known_cost_yuan": 1.91386592,
            "unknown_cost_reserve_yuan": 0.0,
            "accounted_cost_yuan": 1.91386592,
        },
    }
    audit["pilot_audit_sha256"] = canonical_sha256(audit)
    before = build_phase_source_file_set(phase_root)

    receipt = build_retrospective_legacy_receipt(
        phase_root,
        audit_report=audit,
        expected_audit_sha256=audit["pilot_audit_sha256"],
        expected_known_cost_yuan=1.91386592,
        expected_unknown_reserve_yuan=0.0,
    )
    assert validate_retrospective_legacy_receipt(
        receipt,
        phase_root=phase_root,
        audit_report=audit,
        expected_audit_sha256=audit["pilot_audit_sha256"],
        expected_known_cost_yuan=1.91386592,
        expected_unknown_reserve_yuan=0.0,
    )["ok"] is True
    assert build_phase_source_file_set(phase_root) == before
    assert receipt["evidence_quality"] == "legacy_parsed_only_no_raw_content"
    assert receipt["formal_release_gate_eligible"] is False
    assert receipt["raw_content_bound_response_attempt_count"] == 0
    assert receipt["provider_record_sets"]["deepseek"]["record_count"] == 1
    assert receipt["known_cost_yuan"] == 1.91386592

    carried = build_carried_forward_cost_ledger(receipt)
    assert verify_carried_forward_cost_ledger(carried)[
        "total_accounted_cost_yuan"
    ] == 1.91386592
    assert (
        carried["source_phase_receipt_sha256"]
        == receipt["retrospective_legacy_receipt_sha256"]
    )
    receipt_path = tmp_path / "anchors/legacy-receipt.json"
    carried_path = tmp_path / "anchors/carried-cost.json"
    write_retrospective_legacy_receipt(receipt, output=receipt_path)
    write_carried_forward_cost_ledger(carried, output=carried_path)
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert json.loads(carried_path.read_text(encoding="utf-8")) == carried
