"""Formal, read-only pilot approval gate for Persona-v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).parents[1]
RUN_ID = "llm-personas-v2-dual"


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class _LocalValidTransport:
    """Pure in-process fixture transport; it performs no network I/O."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, object]],
        timeout_seconds: float,
        max_tokens: int,
    ) -> dict[str, object]:
        self.call_count += 1
        del provider, timeout_seconds, max_tokens
        blind = '"condition":"blind"' in str(messages[-1]["content"])
        payload: dict[str, object] = {
            "simulated": True,
            "answer": "A",
            "rationale": "offline audit fixture",
        }
        if blind:
            payload["abstain"] = False
        return {
            "content": json.dumps(payload, ensure_ascii=False),
            "model_returned": model,
            "finish_reason": "stop",
            "usage": {"input_tokens": 20, "output_tokens": 8},
            "cost_yuan": 0.001,
            "latency_ms": 1.0,
        }


@pytest.fixture()
def formal_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, object, list[object]]:
    import experiments.llm_sim_v2.audit_pilot as audit_module
    from experiments.llm_sim_v2.collect import (
        resolve_collection_scope,
        verify_formal_carried_forward_cost_ledger,
    )
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        build_phase_provenance,
        enumerate_tasks,
        load_runtime_contract,
        verify_runtime_task_manifest,
        write_phase_provenance,
    )

    auditor_sha = hashlib.sha256(Path(audit_module.__file__).read_bytes()).hexdigest()
    monkeypatch.setattr(
        audit_module,
        "_audit_implementation_proof",
        lambda repo: {
            "passed": True,
            "path": "experiments/llm_sim_v2/audit_pilot.py",
            "auditor_implementation_sha256": auditor_sha,
            "committed_blob_sha256": auditor_sha,
            "audit_git_head": "f" * 40,
            "auditor_commit": "e" * 40,
        },
        raising=False,
    )
    monkeypatch.setattr(
        audit_module,
        "_git_head",
        lambda repo: "b" * 40,
        raising=False,
    )

    def committed_file_proof(
        repo: Path, path: Path, *, head: str | None = None
    ) -> dict[str, object]:
        del repo
        resolved = Path(path)
        if "anchor-a" in resolved.name:
            commit = "a" * 40
            proof_path = (
                "experiments/llm_sim_v2/evidence_anchors/"
                "replacement_pilot_phase_anchor_a.json"
            )
        elif "anchor-b" in resolved.name or "transition" in resolved.name:
            commit = "c" * 40
            proof_path = (
                "experiments/llm_sim_v2/evidence_anchors/"
                + (
                    "replacement_pilot_phase_anchor_b.json"
                    if "anchor-b" in resolved.name
                    else "replacement_pilot_resume_transition.json"
                )
            )
        else:
            commit = "d" * 40
            proof_path = f"experiments/llm_sim_v2/evidence_anchors/{resolved.name}"
        return {
            "passed": True,
            "path": proof_path,
            "file_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "committed_blob_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "git_head": head or "f" * 40,
            "anchor_commit": commit,
        }

    monkeypatch.setattr(
        audit_module,
        "_git_committed_file_proof",
        committed_file_proof,
        raising=False,
    )
    monkeypatch.setattr(
        audit_module,
        "_git_is_ancestor",
        lambda repo, ancestor, descendant, strict=False: (
            ancestor != descendant if strict else True
        ),
        raising=False,
    )
    contract = load_runtime_contract(REPO_ROOT)
    runtime = contract.runtime_manifest
    assert isinstance(runtime, dict)
    proof = verify_runtime_task_manifest(contract, runtime, verify_git=True)
    tasks = enumerate_tasks(contract, phase="pilot")
    providers = tuple(contract.config["pilot"]["providers"])
    scope = resolve_collection_scope(
        frozen_providers=providers,
        requested_providers=None,
        limit=None,
        allow_partial=False,
    )
    carried_forward_cost = verify_formal_carried_forward_cost_ledger(
        json.loads(
            (
                REPO_ROOT
                / "experiments/llm_sim_v2/evidence_anchors/"
                "legacy_pilot_carried_forward_cost.json"
            ).read_text(encoding="utf-8")
        )
    )
    phase = build_phase_provenance(
        contract,
        runtime_manifest=runtime,
        runtime_proof=proof,
        phase="pilot",
        tasks=tasks,
        collection_scope=scope,
        prior_cost_ledger=contract.prior_cost_ledger,
        carried_forward_cost=carried_forward_cost,
        first_observation_at_utc=proof["git_proof"]["observation_timestamp"],
    )
    write_phase_provenance(tmp_path, phase=phase)
    budget = BudgetLedger(
        soft_warning_yuan=300.0,
        hard_fuse_yuan=450.0,
        initial_cost_yuan=round(
            float(contract.prior_cost_ledger["pre_run_total_bound_yuan"])
            + float(carried_forward_cost["total_accounted_cost_yuan"]),
            8,
        ),
    )
    for provider in providers:
        runner = V2ProviderRunner(
            contract=contract,
            output_base=tmp_path,
            phase="pilot",
            provider=provider,
            transport=_LocalValidTransport(),
            budget=budget,
            phase_provenance=phase,
            sleep=lambda _: None,
            random_value=lambda: 0.5,
        )
        runner.run_tasks(tasks)
    return tmp_path / RUN_ID / "pilot", contract, tasks


def _blocking_codes(report: dict[str, object]) -> set[str]:
    reasons = report["blocking_reasons"]
    assert isinstance(reasons, list)
    return {str(row["code"]) for row in reasons}


def _first_record(pilot_root: Path, provider: str = "deepseek") -> Path:
    return sorted((pilot_root / "records" / provider).glob("*.json"))[0]


def _anchor_a_path(pilot_root: Path) -> Path:
    return pilot_root.parents[1] / "pilot-phase-anchor-a.json"


def _anchor_b_path(pilot_root: Path) -> Path:
    return pilot_root.parents[1] / "pilot-phase-anchor-b.json"


def _resume_receipt_path(pilot_root: Path) -> Path:
    return pilot_root.parents[1] / "pilot-resume-transition.json"


def _resume_receipt(pilot_root: Path) -> dict[str, object]:
    return json.loads(_resume_receipt_path(pilot_root).read_text(encoding="utf-8"))


def _audit_with_zero_call_resume(pilot_root: Path) -> dict[str, object]:
    import experiments.llm_sim_v2.audit_pilot as audit_module
    from experiments.llm_sim_v2.audit_pilot import (
        audit_formal_pilot,
        snapshot_resume_state,
    )
    from experiments.llm_sim_v2.evidence import write_phase_evidence_receipt
    from experiments.llm_sim_v2.runner import enumerate_tasks, load_runtime_contract

    before = snapshot_resume_state(pilot_root)
    after = before
    receipt = None
    receipt_path = None
    if pilot_root.name == "pilot":
        try:
            anchor_a_path = _anchor_a_path(pilot_root)
            anchor_b_path = _anchor_b_path(pilot_root)
            receipt_path = _resume_receipt_path(pilot_root)
            if not anchor_a_path.exists():
                contract = load_runtime_contract(REPO_ROOT)
                write_phase_evidence_receipt(
                    pilot_root,
                    output=anchor_a_path,
                    phase_provenance=json.loads(
                        (pilot_root / "phase_provenance.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                    tasks=enumerate_tasks(contract, phase="pilot"),
                )
            probe = audit_module.run_zero_call_resume_probe(
                repo_root=REPO_ROOT,
                pilot_root=pilot_root,
                anchor_a_path=anchor_a_path,
                anchor_b_path=anchor_b_path,
                transition_receipt_path=receipt_path,
            )
            before = probe["resume_before"]
            after = probe["resume_after"]
            receipt = probe["receipt"]
        except (audit_module.PilotAuditError, ValueError):
            # Invalid stores fail preflight before the bomb transport can be reached.
            pass
    return audit_formal_pilot(
        repo_root=REPO_ROOT,
        pilot_root=pilot_root,
        resume_before=before,
        resume_after=after,
        resume_receipt=receipt,
        resume_receipt_path=receipt_path if receipt is not None else None,
        anchor_a_path=_anchor_a_path(pilot_root),
        anchor_b_path=_anchor_b_path(pilot_root),
    )


def _audit_existing_protocol(pilot_root: Path) -> dict[str, object]:
    from experiments.llm_sim_v2.audit_pilot import (
        audit_formal_pilot,
        snapshot_resume_state,
    )

    snapshot = snapshot_resume_state(pilot_root)
    return audit_formal_pilot(
        repo_root=REPO_ROOT,
        pilot_root=pilot_root,
        resume_before=snapshot,
        resume_after=snapshot,
        resume_receipt=_resume_receipt(pilot_root),
        resume_receipt_path=_resume_receipt_path(pilot_root),
        anchor_a_path=_anchor_a_path(pilot_root),
        anchor_b_path=_anchor_b_path(pilot_root),
    )


def test_complete_formal_pilot_is_hash_bound_go(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        audit_formal_pilot,
        snapshot_resume_state,
        verify_pilot_audit_output,
        write_pilot_audit,
    )

    pilot_root, _, _ = formal_pilot
    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "GO", report["blocking_reasons"]
    assert report["formal_pilot_approved"] is True
    assert report["expected_providers"] == ["deepseek", "doubao"]
    assert report["expected_task_count_per_provider"] == 128
    assert report["gates"]["formal_scope"]["passed"] is True
    assert report["gates"]["text_only_and_leakage"]["passed"] is True
    assert report["gates"]["pilot_main_isolation"]["passed"] is True
    assert report["gates"]["phase_evidence_receipts"]["passed"] is True
    assert report["gates"]["billing_authorization_resolution"] == {
        "passed": True,
        "evidence": {
            "status": "not_required",
            "unknown_attempt_count": 0,
        },
    }
    assert report["gates"]["zero_call_resume"]["passed"] is True
    assert report["accounting"]["pre_collection_total_yuan"] == pytest.approx(
        2.57152913
    )
    assert report["blocking_reasons"] == []
    audit_module = REPO_ROOT / "experiments/llm_sim_v2/audit_pilot.py"
    assert report["source_binding"]["auditor_implementation_sha256"] == (
        hashlib.sha256(audit_module.read_bytes()).hexdigest()
    )
    assert len(report["source_binding"]["audit_git_head"]) == 40
    assert (
        report["source_binding"][
            "billing_authorization_resolution_file_sha256"
        ]
        is None
    )
    assert (
        report["source_binding"]["billing_authorization_resolution_sha256"]
        is None
    )
    for provider in ("deepseek", "doubao"):
        provider_report = report["providers"][provider]
        assert provider_report["expected_task_state_count"] == 128
        assert provider_report["unexplained_task_ids"] == []
        assert provider_report["state_counts"] == {"record_complete": 128}
        assert provider_report["resume_evidence"] == {
            "resumed_record_count": 128,
            "lifecycle_event_count": 2,
        }
        for condition, expected in (("controlled", 40), ("blind", 88)):
            cell = provider_report["condition_cells"][condition]
            assert cell["expected_count"] == expected
            assert cell["complete_fraction"] == 1.0
            assert cell["invalid_schema_fraction"] == 0.0
            assert cell["passes"] is True

    payload = dict(report)
    advertised = payload.pop("pilot_audit_sha256")
    assert advertised == _canonical_sha(payload)

    output = tmp_path / "audit-output"
    snapshot = snapshot_resume_state(pilot_root)
    receipt_path = _resume_receipt_path(pilot_root)
    manifest = write_pilot_audit(
        report,
        output,
        repo_root=REPO_ROOT,
        pilot_root=pilot_root,
        resume_before=snapshot,
        resume_after=snapshot,
        resume_receipt=_resume_receipt(pilot_root),
        resume_receipt_path=receipt_path,
        anchor_a_path=_anchor_a_path(pilot_root),
        anchor_b_path=_anchor_b_path(pilot_root),
    )
    assert (output / "pilot_gate.json").is_file()
    assert (output / "PILOT_GATE_REPORT.md").is_file()
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(
        (output / "pilot_gate.json").read_bytes()
    ).hexdigest()
    assert "GO" in (output / "PILOT_GATE_REPORT.md").read_text(encoding="utf-8")
    assert manifest["artifact_manifest_sha256"]
    assert verify_pilot_audit_output(output)["ok"] is True


def test_zero_call_resume_probe_emits_git_verifiable_receipt(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
) -> None:
    import experiments.llm_sim_v2.audit_pilot as audit_module

    probe = getattr(audit_module, "run_zero_call_resume_probe", None)
    assert probe is not None
    pilot_root, _, _ = formal_pilot
    from experiments.llm_sim_v2.evidence import write_phase_evidence_receipt
    from experiments.llm_sim_v2.runner import enumerate_tasks, load_runtime_contract

    anchor_a_path = tmp_path / "pilot-phase-anchor-a.json"
    anchor_b_path = tmp_path / "pilot-phase-anchor-b.json"
    receipt_path = tmp_path / "pilot-resume-transition.json"
    contract = load_runtime_contract(REPO_ROOT)
    tasks = enumerate_tasks(contract, phase="pilot")
    write_phase_evidence_receipt(
        pilot_root,
        output=anchor_a_path,
        phase_provenance=json.loads(
            (pilot_root / "phase_provenance.json").read_text(encoding="utf-8")
        ),
        tasks=tasks,
    )

    result = probe(
        repo_root=REPO_ROOT,
        pilot_root=pilot_root,
        anchor_a_path=anchor_a_path,
        anchor_b_path=anchor_b_path,
        transition_receipt_path=receipt_path,
    )

    receipt = result["receipt"]
    assert receipt["schema_version"] == (
        "yher.llm_sim_v2.pilot_resume_transition_receipt.v2"
    )
    assert receipt["resume_execution_head"] == "b" * 40
    assert receipt["anchor_a"]["phase_evidence_receipt_sha256"]
    assert receipt["anchor_b"]["phase_evidence_receipt_sha256"]
    assert receipt["anchor_a"] != receipt["anchor_b"]
    assert receipt["provider_call_count_delta"] == 0
    assert receipt["records_unchanged"] is True
    for provider in ("deepseek", "doubao"):
        delta = receipt["provider_deltas"][provider]
        assert delta["evidence_event_count_delta"] == 2
        assert delta["provider_call_count_delta"] == 0
        assert delta["record_count_delta"] == 0
        assert delta["record_set_unchanged"] is True
        assert [row["event_type"] for row in delta["added_events"]] == [
            "invocation_started",
            "invocation_finished",
        ]
        assert delta["added_events"][1]["invocation_kind"] == "resume"
        assert delta["added_events"][1]["resumed_record_count"] == 128
    assert receipt_path.is_file()
    missing_receipt_report = audit_module.audit_formal_pilot(
        repo_root=REPO_ROOT,
        pilot_root=pilot_root,
        resume_before=result["resume_before"],
        resume_after=result["resume_after"],
        anchor_a_path=anchor_a_path,
        anchor_b_path=anchor_b_path,
    )
    assert missing_receipt_report["decision"] == "BLOCK"
    assert "zero_call_resume_receipt" in _blocking_codes(missing_receipt_report)
    report = audit_module.audit_formal_pilot(
        repo_root=REPO_ROOT,
        pilot_root=pilot_root,
        resume_before=result["resume_before"],
        resume_after=result["resume_after"],
        resume_receipt=receipt,
        resume_receipt_path=receipt_path,
        anchor_a_path=anchor_a_path,
        anchor_b_path=anchor_b_path,
    )
    assert report["decision"] == "GO", report["blocking_reasons"]


def test_probe_rejects_receipt_inserted_after_anchor_a_before_resume(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        PilotAuditError,
        run_zero_call_resume_probe,
    )
    from experiments.llm_sim_v2.evidence import write_phase_evidence_receipt

    pilot_root, _, tasks = formal_pilot
    phase = json.loads(
        (pilot_root / "phase_provenance.json").read_text(encoding="utf-8")
    )
    anchor_a_path = tmp_path / "pilot-phase-anchor-a.json"
    anchor_b_path = tmp_path / "pilot-phase-anchor-b.json"
    transition_path = tmp_path / "pilot-resume-transition.json"
    internal_a = write_phase_evidence_receipt(
        pilot_root,
        phase_provenance=phase,
        tasks=tasks,
    )
    anchor_a = write_phase_evidence_receipt(
        pilot_root,
        output=anchor_a_path,
        phase_provenance=phase,
        tasks=tasks,
    )
    assert anchor_a == internal_a

    rogue = json.loads(json.dumps(anchor_a))
    rogue["store_snapshot"]["total_bytes"] += 1
    rogue.pop("phase_evidence_receipt_sha256")
    rogue["phase_evidence_receipt_sha256"] = _canonical_sha(rogue)
    rogue_path = (
        pilot_root
        / "evidence/phase_receipts"
        / f"{rogue['phase_evidence_receipt_sha256']}.json"
    )
    rogue_path.write_text(
        json.dumps(rogue, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PilotAuditError, match="phase-receipt inventory"):
        run_zero_call_resume_probe(
            repo_root=REPO_ROOT,
            pilot_root=pilot_root,
            anchor_a_path=anchor_a_path,
            anchor_b_path=anchor_b_path,
            transition_receipt_path=transition_path,
        )

    assert not anchor_b_path.exists()
    assert not transition_path.exists()


def test_resume_receipt_binds_running_auditor_bytes(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
) -> None:
    import experiments.llm_sim_v2.audit_pilot as audit_module

    pilot_root, _, _ = formal_pilot
    del tmp_path
    assert _audit_with_zero_call_resume(pilot_root)["decision"] == "GO"
    receipt_path = _resume_receipt_path(pilot_root)
    receipt = _resume_receipt(pilot_root)
    receipt["auditor_implementation_sha256"] = "0" * 64
    receipt.pop("transition_receipt_sha256")
    receipt["transition_receipt_sha256"] = _canonical_sha(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    snapshot = audit_module.snapshot_resume_state(pilot_root)

    report = audit_module.audit_formal_pilot(
        repo_root=REPO_ROOT,
        pilot_root=pilot_root,
        resume_before=snapshot,
        resume_after=snapshot,
        resume_receipt=receipt,
        resume_receipt_path=receipt_path,
        anchor_a_path=_anchor_a_path(pilot_root),
        anchor_b_path=_anchor_b_path(pilot_root),
    )

    assert report["decision"] == "BLOCK"
    assert "zero_call_resume_receipt" in _blocking_codes(report)


def test_protocol_rejects_missing_tampered_and_stale_phase_anchors(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    from experiments.llm_sim_v2.runner import (
        BudgetLedger,
        V2ProviderRunner,
        enumerate_tasks,
        load_runtime_contract,
    )

    pilot_root, _, _ = formal_pilot
    assert _audit_with_zero_call_resume(pilot_root)["decision"] == "GO"
    a_path = _anchor_a_path(pilot_root)
    b_path = _anchor_b_path(pilot_root)

    for missing_path in (a_path, b_path):
        backup = missing_path.with_suffix(".missing")
        missing_path.rename(backup)
        try:
            report = _audit_existing_protocol(pilot_root)
            assert report["decision"] == "BLOCK"
            assert "zero_call_resume_receipt" in _blocking_codes(report)
        finally:
            backup.rename(missing_path)

    for tampered_path in (a_path, b_path):
        original = tampered_path.read_bytes()
        row = json.loads(original)
        row["store_snapshot"]["total_bytes"] += 1
        row.pop("phase_evidence_receipt_sha256")
        row["phase_evidence_receipt_sha256"] = _canonical_sha(row)
        tampered_path.write_text(json.dumps(row), encoding="utf-8")
        try:
            report = _audit_existing_protocol(pilot_root)
            assert report["decision"] == "BLOCK"
            assert "zero_call_resume_receipt" in _blocking_codes(report)
        finally:
            tampered_path.write_bytes(original)

    contract = load_runtime_contract(REPO_ROOT)
    tasks = enumerate_tasks(contract, phase="pilot")
    phase = json.loads(
        (pilot_root / "phase_provenance.json").read_text(encoding="utf-8")
    )

    class BombTransport:
        def complete(self, **_: object) -> dict[str, object]:
            raise AssertionError("stale-B test reached the provider transport")

    baseline = _audit_existing_protocol(pilot_root)
    runner = V2ProviderRunner(
        contract=contract,
        output_base=pilot_root.parents[1],
        phase="pilot",
        provider="deepseek",
        transport=BombTransport(),
        budget=BudgetLedger(
            soft_warning_yuan=300.0,
            hard_fuse_yuan=450.0,
            initial_cost_yuan=float(
                baseline["accounting"]["total_accounted_cost_yuan"]
            ),
        ),
        phase_provenance=phase,
    )
    runner.run_tasks(tasks)
    stale = _audit_existing_protocol(pilot_root)
    assert stale["decision"] == "BLOCK"
    assert "zero_call_resume_receipt" in _blocking_codes(stale)


def test_protocol_rejects_uncommitted_and_nonancestor_git_anchors(
    formal_pilot: tuple[Path, object, list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.audit_pilot as audit_module

    pilot_root, _, _ = formal_pilot
    assert _audit_with_zero_call_resume(pilot_root)["decision"] == "GO"
    committed_proof = audit_module._git_committed_file_proof
    ancestry = audit_module._git_is_ancestor

    for marker in ("anchor-a", "anchor-b"):
        def uncommitted(
            repo: Path,
            path: Path,
            *,
            head: str | None = None,
            marker: str = marker,
        ) -> dict[str, object]:
            proof = dict(committed_proof(repo, path, head=head))
            if marker in Path(path).name:
                proof["passed"] = False
                proof["committed_blob_sha256"] = None
            return proof

        with monkeypatch.context() as scoped:
            scoped.setattr(audit_module, "_git_committed_file_proof", uncommitted)
            report = _audit_existing_protocol(pilot_root)
        assert report["decision"] == "BLOCK"
        assert "zero_call_resume_receipt" in _blocking_codes(report)

    for blocked_edge in (("a" * 40, "b" * 40), ("b" * 40, "c" * 40)):
        def nonancestor(
            repo: Path,
            ancestor: str,
            descendant: str,
            *,
            strict: bool = False,
            blocked_edge: tuple[str, str] = blocked_edge,
        ) -> bool:
            if (ancestor, descendant) == blocked_edge:
                return False
            return ancestry(repo, ancestor, descendant, strict=strict)

        with monkeypatch.context() as scoped:
            scoped.setattr(audit_module, "_git_is_ancestor", nonancestor)
            report = _audit_existing_protocol(pilot_root)
        assert report["decision"] == "BLOCK"
        assert "zero_call_resume_receipt" in _blocking_codes(report)


def test_protocol_rejects_call_delta_and_wrong_resume_kind(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    assert _audit_with_zero_call_resume(pilot_root)["decision"] == "GO"
    transition_path = _resume_receipt_path(pilot_root)
    original = transition_path.read_bytes()

    mutations = (
        lambda row: row.__setitem__("provider_call_count_delta", 1),
        lambda row: row["provider_deltas"]["deepseek"]["added_events"][1].__setitem__(
            "invocation_kind", "mixed_resume"
        ),
        lambda row: row.update(
            {
                "before_resume_snapshot_sha256": "0" * 64,
                "after_resume_snapshot_sha256": "1" * 64,
                "before_record_count": 0,
                "after_record_count": 1,
                "before_attempt_count": 2,
                "after_attempt_count": 3,
            }
        ),
    )
    for mutate in mutations:
        row = json.loads(original)
        mutate(row)
        row.pop("transition_receipt_sha256")
        row["transition_receipt_sha256"] = _canonical_sha(row)
        transition_path.write_text(json.dumps(row), encoding="utf-8")
        try:
            report = _audit_existing_protocol(pilot_root)
            assert report["decision"] == "BLOCK"
            assert "zero_call_resume_receipt" in _blocking_codes(report)
        finally:
            transition_path.write_bytes(original)


def test_protocol_rejects_unbound_internal_phase_receipt(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    assert _audit_with_zero_call_resume(pilot_root)["decision"] == "GO"

    fake = json.loads(_anchor_a_path(pilot_root).read_text(encoding="utf-8"))
    fake["store_snapshot"]["total_bytes"] += 1
    fake.pop("phase_evidence_receipt_sha256")
    fake["phase_evidence_receipt_sha256"] = _canonical_sha(fake)
    rogue = (
        pilot_root
        / "evidence/phase_receipts"
        / f"{fake['phase_evidence_receipt_sha256']}.json"
    )
    rogue.write_text(
        json.dumps(fake, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    report = _audit_existing_protocol(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "zero_call_resume_receipt" in _blocking_codes(report)


def test_prior_only_budget_reset_is_rejected(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    baseline = _audit_with_zero_call_resume(pilot_root)
    assert baseline["decision"] == "GO"
    assert baseline["accounting"]["pre_collection_total_yuan"] == pytest.approx(
        2.57152913
    )
    for provider in ("deepseek", "doubao"):
        path = pilot_root / "provider_manifests" / f"{provider}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["budget"]["total_cost_yuan"] = round(
            float(row["budget"]["total_cost_yuan"]) - 1.91386592,
            8,
        )
        path.write_text(json.dumps(row), encoding="utf-8")

    report = _audit_existing_protocol(pilot_root)
    assert report["decision"] == "BLOCK"
    assert "budget_reconciliation" in _blocking_codes(report)


def test_git_file_proof_uses_real_commits_and_strict_ancestry(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        _git_committed_file_proof,
        _git_is_ancestor,
    )

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.email", "audit@example.invalid")
    git("config", "user.name", "Pilot Audit Test")
    anchor_a = repo / "anchor-a.json"
    anchor_a.write_text("{\"anchor\":\"A\"}\n", encoding="utf-8")
    git("add", "anchor-a.json")
    git("commit", "-m", "anchor A")
    a_commit = git("rev-parse", "HEAD")
    (repo / "execution.txt").write_text("resume head\n", encoding="utf-8")
    git("add", "execution.txt")
    git("commit", "-m", "resume execution head")
    execution_head = git("rev-parse", "HEAD")
    anchor_b = repo / "anchor-b.json"
    transition = repo / "transition.json"
    anchor_b.write_text("{\"anchor\":\"B\"}\n", encoding="utf-8")
    transition.write_text("{\"transition\":2}\n", encoding="utf-8")
    git("add", "anchor-b.json", "transition.json")
    git("commit", "-m", "anchor B and transition")
    b_commit = git("rev-parse", "HEAD")
    (repo / "audit.txt").write_text("audit head\n", encoding="utf-8")
    git("add", "audit.txt")
    git("commit", "-m", "audit head")
    audit_head = git("rev-parse", "HEAD")

    a_proof = _git_committed_file_proof(repo, anchor_a, head=audit_head)
    b_proof = _git_committed_file_proof(repo, anchor_b, head=audit_head)
    assert a_proof["passed"] is True
    assert a_proof["anchor_commit"] == a_commit
    assert b_proof["passed"] is True
    assert b_proof["anchor_commit"] == b_commit
    assert _git_is_ancestor(repo, a_commit, execution_head, strict=True)
    assert _git_is_ancestor(repo, execution_head, b_commit, strict=True)
    assert _git_is_ancestor(repo, b_commit, audit_head)
    assert not _git_is_ancestor(repo, b_commit, b_commit, strict=True)

    anchor_b.write_text("uncommitted drift\n", encoding="utf-8")
    assert _git_committed_file_proof(repo, anchor_b, head=audit_head)[
        "passed"
    ] is False


def test_uncommitted_auditor_bytes_block_formal_go(
    formal_pilot: tuple[Path, object, list[object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.audit_pilot as audit_module

    pilot_root, _, _ = formal_pilot
    monkeypatch.setattr(
        audit_module,
        "_audit_implementation_proof",
        lambda repo: {
            "passed": False,
            "path": "experiments/llm_sim_v2/audit_pilot.py",
            "auditor_implementation_sha256": "a" * 64,
            "committed_blob_sha256": None,
            "audit_git_head": "f" * 40,
            "auditor_commit": None,
        },
    )

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "auditor_git_provenance" in _blocking_codes(report)
    assert report["gates"]["auditor_git_provenance"]["passed"] is False


def test_writer_rejects_stale_output_entries(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        PilotAuditError,
        snapshot_resume_state,
        write_pilot_audit,
    )

    pilot_root, _, _ = formal_pilot
    report = _audit_with_zero_call_resume(pilot_root)
    snapshot = snapshot_resume_state(pilot_root)
    output = tmp_path / "stale-output"
    output.mkdir()
    (output / "note.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(PilotAuditError, match="empty"):
        write_pilot_audit(
            report,
            output,
            repo_root=REPO_ROOT,
            pilot_root=pilot_root,
            resume_before=snapshot,
            resume_after=snapshot,
            resume_receipt=_resume_receipt(pilot_root),
            resume_receipt_path=_resume_receipt_path(pilot_root),
        )


def test_unexplained_or_extra_task_id_blocks_formal_pilot(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    from experiments.llm_sim_v2.audit_pilot import audit_formal_pilot

    pilot_root, _, _ = formal_pilot
    source = _first_record(pilot_root)
    extra = source.with_name("f" * 64 + ".json")
    shutil.copyfile(source, extra)

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert report["formal_pilot_approved"] is False
    assert "unexplained_record_ids" in _blocking_codes(report)
    assert "f" * 64 in report["providers"]["deepseek"]["unexplained_task_ids"]


def test_rogue_record_root_entry_blocks_formal_pilot(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    (pilot_root / "records" / "rogue.json").write_text("{}", encoding="utf-8")

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "provider_roster_drift" in _blocking_codes(report)


def test_rogue_manifest_note_blocks_formal_pilot(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    (pilot_root / "provider_manifests" / "note.txt").write_text(
        "unbound", encoding="utf-8"
    )

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "provider_roster_drift" in _blocking_codes(report)


def test_model_and_record_provenance_drift_block(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    from experiments.llm_sim_v2.audit_pilot import audit_formal_pilot

    pilot_root, _, _ = formal_pilot
    path = _first_record(pilot_root, "doubao")
    row = json.loads(path.read_text(encoding="utf-8"))
    row["model_id"] = "drifted-model"
    row["provenance"]["execution_commit"] = "0" * 40
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert {"model_drift", "record_provenance_drift"} <= _blocking_codes(report)


def test_condition_threshold_and_cost_reconciliation_block(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    from experiments.llm_sim_v2.audit_pilot import audit_formal_pilot

    pilot_root, _, tasks = formal_pilot
    controlled_ids = [
        task.task_id for task in tasks if task.condition == "controlled"
    ][:9]
    for task_id in controlled_ids:
        path = pilot_root / "records" / "deepseek" / f"{task_id}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["status"] = "excluded_schema"
        row["parsed_output"] = None
        path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    cost_path = _first_record(pilot_root, "doubao")
    cost_row = json.loads(cost_path.read_text(encoding="utf-8"))
    cost_row["known_cost_yuan"] = 99.0
    cost_path.write_text(json.dumps(cost_row, ensure_ascii=False), encoding="utf-8")

    report = _audit_with_zero_call_resume(pilot_root)

    controlled = report["providers"]["deepseek"]["condition_cells"]["controlled"]
    assert controlled["complete_fraction"] == pytest.approx(31 / 40)
    assert controlled["invalid_schema_fraction"] == pytest.approx(9 / 40)
    assert controlled["passes"] is False
    assert {"condition_cell_threshold", "budget_reconciliation"} <= _blocking_codes(
        report
    )


def test_complete_record_payload_is_revalidated(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    path = _first_record(pilot_root, "deepseek")
    row = json.loads(path.read_text(encoding="utf-8"))
    row["parsed_output"] = {
        "simulated": True,
        "answer": "A",
        "rationale": "",
    }
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "response_semantics" in _blocking_codes(report)


def test_complete_record_requires_positive_metered_usage(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    path = _first_record(pilot_root, "deepseek")
    row = json.loads(path.read_text(encoding="utf-8"))
    row["attempts"][-1]["usage"] = None
    row["attempts"][-1]["cost_yuan"] = 0.0
    row["known_cost_yuan"] = 0.0
    row["cost_yuan"] = 0.0
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "response_semantics" in _blocking_codes(report)


def test_successful_attempt_cannot_be_followed_by_another_call(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    path = _first_record(pilot_root, "deepseek")
    row = json.loads(path.read_text(encoding="utf-8"))
    duplicate = dict(row["attempts"][-1])
    duplicate["attempt"] = 2
    row["attempts"].append(duplicate)
    row["retry_count"] = 1
    row["known_cost_yuan"] = 0.002
    row["cost_yuan"] = 0.002
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "attempt_reconciliation" in _blocking_codes(report)


def test_unknown_billing_reserve_requires_user_resolution_before_go(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    path = _first_record(pilot_root, "deepseek")
    row = json.loads(path.read_text(encoding="utf-8"))
    response = dict(row["attempts"][-1])
    response["attempt"] = 2
    row["attempts"] = [
        {
            "attempt": 1,
            "status": "failed",
            "error_category": "network",
            "latency_ms": 1.0,
            "cost_yuan": None,
            "cost_known": False,
            "billing_ambiguity": True,
            "cost_reserve_yuan": 10.0,
        },
        response,
    ]
    row["retry_count"] = 1
    row["unknown_cost_reserve_yuan"] = 10.0
    row["cost_yuan"] = 10.001
    row["has_unknown_cost_attempts"] = True
    row["needs_user"] = True
    row["needs_user_reasons"] = ["unknown_provider_billing_reserved"]
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "needs_user_unresolved" in _blocking_codes(report)


def test_partial_phase_and_non_pilot_root_never_approve(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    from experiments.llm_sim_v2.audit_pilot import audit_formal_pilot

    pilot_root, _, _ = formal_pilot
    phase_path = pilot_root / "phase_provenance.json"
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    phase["collection_mode"] = "development_partial"
    phase["development_only"] = True
    phase["partial"] = True
    phase["formal_analysis_eligible"] = False
    phase.pop("phase_provenance_sha256")
    phase["phase_provenance_sha256"] = _canonical_sha(phase)
    phase_path.write_text(json.dumps(phase, ensure_ascii=False), encoding="utf-8")

    report = _audit_with_zero_call_resume(pilot_root)
    assert report["decision"] == "BLOCK"
    assert "formal_scope" in _blocking_codes(report)

    renamed = pilot_root.with_name("development-pilot")
    pilot_root.rename(renamed)
    report = _audit_with_zero_call_resume(renamed)
    assert report["decision"] == "BLOCK"
    assert "pilot_main_isolation" in _blocking_codes(report)


def test_sibling_main_store_blocks_pilot_approval(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    (pilot_root.parent / "main").mkdir()

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "pilot_main_isolation" in _blocking_codes(report)


def test_child_main_store_blocks_pilot_approval(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    pilot_root, _, _ = formal_pilot
    (pilot_root / "main").mkdir()

    report = _audit_with_zero_call_resume(pilot_root)

    assert report["decision"] == "BLOCK"
    assert "pilot_main_isolation" in _blocking_codes(report)


def test_verifier_rejects_rehashed_go_with_failed_gate(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        PilotAuditError,
        verify_pilot_audit,
    )

    pilot_root, _, _ = formal_pilot
    (pilot_root.parent / "main").mkdir()
    forged = _audit_with_zero_call_resume(pilot_root)
    assert forged["decision"] == "BLOCK"
    forged["decision"] = "GO"
    forged["formal_pilot_approved"] = True
    forged["blocking_reasons"] = []
    forged.pop("pilot_audit_sha256")
    forged["pilot_audit_sha256"] = _canonical_sha(forged)

    with pytest.raises(PilotAuditError, match="failed gate"):
        verify_pilot_audit(forged)


def test_verifier_rejects_structurally_gutted_rehashed_go(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        PilotAuditError,
        verify_pilot_audit,
    )

    pilot_root, _, _ = formal_pilot
    valid = _audit_with_zero_call_resume(pilot_root)
    assert valid["decision"] == "GO"
    forged = json.loads(json.dumps(valid))
    forged["gates"] = {"fake": {"passed": True}}
    forged["providers"] = {}
    forged.pop("pilot_audit_sha256")
    forged["pilot_audit_sha256"] = _canonical_sha(forged)

    with pytest.raises(PilotAuditError, match="gate set"):
        verify_pilot_audit(forged)

    for gate_name in ("phase_evidence_receipts", "zero_call_resume"):
        forged = json.loads(json.dumps(valid))
        forged["gates"][gate_name]["evidence"] = {}
        forged.pop("pilot_audit_sha256")
        forged["pilot_audit_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
            verify_pilot_audit(forged)

    for field in (
        "carried_forward_cost_ledger_sha256",
        "carried_forward_source_phase_receipt_sha256",
        "carried_forward_source_record_set_sha256",
    ):
        forged = json.loads(json.dumps(valid))
        forged["accounting"][field] = "0" * 64
        forged.pop("pilot_audit_sha256")
        forged["pilot_audit_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
            verify_pilot_audit(forged)

    for field in (
        "anchor_a_file_sha256",
        "anchor_b_file_sha256",
        "transition_receipt_file_sha256",
    ):
        forged = json.loads(json.dumps(valid))
        forged["source_binding"][field] = "0" * 64
        forged.pop("pilot_audit_sha256")
        forged["pilot_audit_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
            verify_pilot_audit(forged)

    transition_proof = valid["gates"]["zero_call_resume"]["evidence"][
        "git_anchored_receipt"
    ]
    for source_field, proof_name in (
        ("anchor_a_file_sha256", "anchor_a"),
        ("anchor_b_file_sha256", "anchor_b"),
        ("transition_receipt_file_sha256", "transition"),
    ):
        forged = json.loads(json.dumps(valid))
        forged["source_binding"][source_field] = "0" * 64
        forged["gates"]["zero_call_resume"]["evidence"][
            "git_anchored_receipt"
        ][proof_name]["file_sha256"] = "0" * 64
        forged.pop("pilot_audit_sha256")
        forged["pilot_audit_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
            verify_pilot_audit(forged)

    forged = json.loads(json.dumps(valid))
    gate_tree = forged["gates"]["phase_evidence_receipts"]["evidence"][
        "evidence_tree"
    ]
    rogue_digest = "0" * 64
    exemplar = next(iter(gate_tree["phase_receipts"].values()))
    gate_tree["phase_receipts"][rogue_digest] = {
        **exemplar,
        "path": f"evidence/phase_receipts/{rogue_digest}.json",
    }
    forged.pop("pilot_audit_sha256")
    forged["pilot_audit_sha256"] = _canonical_sha(forged)
    with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
        verify_pilot_audit(forged)

    billing_mutations = (
        lambda row: row["gates"]["billing_authorization_resolution"][
            "evidence"
        ].update({"status": "applied"}),
        lambda row: row["gates"]["billing_authorization_resolution"][
            "evidence"
        ].update({"unknown_attempt_count": 1}),
        lambda row: row["source_binding"].update(
            {
                "billing_authorization_resolution_file_sha256": "1" * 64,
                "billing_authorization_resolution_sha256": "2" * 64,
            }
        ),
    )
    for mutate in billing_mutations:
        forged = json.loads(json.dumps(valid))
        mutate(forged)
        forged.pop("pilot_audit_sha256")
        forged["pilot_audit_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError, match="billing authorization"):
            verify_pilot_audit(forged)

    applied = json.loads(json.dumps(valid))
    applied["accounting"]["unknown_attempt_count"] = 1
    applied["accounting"]["unknown_cost_reserve_yuan"] = 10.0
    applied["accounting"]["accounted_cost_yuan"] = round(
        float(applied["accounting"]["accounted_cost_yuan"]) + 10.0, 8
    )
    applied["accounting"]["total_accounted_cost_yuan"] = round(
        float(applied["accounting"]["total_accounted_cost_yuan"]) + 10.0, 8
    )
    deepseek = applied["providers"]["deepseek"]
    deepseek["accounting"]["unknown_attempt_count"] = 1
    deepseek["accounting"]["unknown_cost_reserve_yuan"] = 10.0
    deepseek["accounting"]["accounted_cost_yuan"] = round(
        float(deepseek["accounting"]["accounted_cost_yuan"]) + 10.0, 8
    )
    deepseek["disclosures"]["needs_user"] = {
        "required": True,
        "reason": "unknown_provider_billing_reserved",
        "record_count": 1,
        "record_task_ids": ["reserved-task"],
        "unknown_cost_attempt_count": 1,
    }
    transition = applied["gates"]["zero_call_resume"]["evidence"][
        "git_anchored_receipt"
    ]
    applied_proof = {
        "status": "applied",
        "billing_fact_status": "unresolved_reserved",
        "action_disposition": "continue_under_preexisting_budget_authorization",
        "billing_authorization_resolution_sha256": "2" * 64,
        "file_sha256": "1" * 64,
        "committed_blob_sha256": "1" * 64,
        "receipt_commit": transition["resume_execution_head"],
        "audit_head": applied["source_binding"]["audit_git_head"],
        "anchor_a_commit": transition["anchor_a"]["anchor_commit"],
        "anchor_a_phase_evidence_receipt_sha256": transition[
            "anchor_a_phase_evidence_receipt_sha256"
        ],
        "unknown_attempt_count": 1,
        "unknown_attempt_set_sha256": "3" * 64,
        "unknown_reserve_yuan": 10.0,
        "total_accounted_cost_yuan": applied["accounting"][
            "total_accounted_cost_yuan"
        ],
        "hard_fuse_yuan": applied["accounting"]["hard_fuse_yuan"],
    }
    applied["gates"]["billing_authorization_resolution"] = {
        "passed": True,
        "evidence": applied_proof,
    }
    applied["source_binding"].update(
        {
            "billing_authorization_resolution_file_sha256": "1" * 64,
            "billing_authorization_resolution_sha256": "2" * 64,
        }
    )
    applied.pop("pilot_audit_sha256")
    applied["pilot_audit_sha256"] = _canonical_sha(applied)
    assert verify_pilot_audit(applied)["ok"] is True

    applied_mutations = (
        lambda row: row["gates"]["billing_authorization_resolution"][
            "evidence"
        ].update({"receipt_commit": "9" * 40}),
        lambda row: row["gates"]["billing_authorization_resolution"][
            "evidence"
        ].update({"anchor_a_commit": "8" * 40}),
        lambda row: row["gates"]["billing_authorization_resolution"][
            "evidence"
        ].update({"unknown_reserve_yuan": 0.0}),
        lambda row: row["gates"]["billing_authorization_resolution"][
            "evidence"
        ].update({"unknown_attempt_count": True}),
        lambda row: row["providers"]["deepseek"]["disclosures"][
            "needs_user"
        ].update({"required": False}),
    )
    for mutate in applied_mutations:
        forged = json.loads(json.dumps(applied))
        mutate(forged)
        forged.pop("pilot_audit_sha256")
        forged["pilot_audit_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError, match="billing authorization"):
            verify_pilot_audit(forged)

    for field, value in (
        ("resume_execution_head", "z" * 40),
        ("transition_receipt_sha256", "z" * 64),
    ):
        forged = json.loads(json.dumps(valid))
        forged["gates"]["zero_call_resume"]["evidence"][
            "git_anchored_receipt"
        ][field] = value
        forged.pop("pilot_audit_sha256")
        forged["pilot_audit_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
            verify_pilot_audit(forged)

    forged = json.loads(json.dumps(valid))
    git_proof = forged["gates"]["zero_call_resume"]["evidence"][
        "git_anchored_receipt"
    ]
    del git_proof["anchor_a"]["path"]
    forged.pop("pilot_audit_sha256")
    forged["pilot_audit_sha256"] = _canonical_sha(forged)
    with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
        verify_pilot_audit(forged)

    forged = json.loads(json.dumps(valid))
    git_proof = forged["gates"]["zero_call_resume"]["evidence"][
        "git_anchored_receipt"
    ]
    git_proof["anchor_b"]["path"] = git_proof["anchor_a"]["path"]
    forged.pop("pilot_audit_sha256")
    forged["pilot_audit_sha256"] = _canonical_sha(forged)
    with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
        verify_pilot_audit(forged)

    for field in (
        "top_level_entries",
        "provider_event_counts",
        "provider_lock_files",
    ):
        forged = json.loads(json.dumps(valid))
        gate_tree = forged["gates"]["phase_evidence_receipts"]["evidence"][
            "evidence_tree"
        ]
        transition_tree = forged["gates"]["zero_call_resume"]["evidence"][
            "git_anchored_receipt"
        ]["evidence_tree"]
        del gate_tree[field]
        del transition_tree[field]
        forged.pop("pilot_audit_sha256")
        forged["pilot_audit_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
            verify_pilot_audit(forged)

    forged = json.loads(json.dumps(valid))
    for tree in (
        forged["gates"]["phase_evidence_receipts"]["evidence"]["evidence_tree"],
        forged["gates"]["zero_call_resume"]["evidence"][
            "git_anchored_receipt"
        ]["evidence_tree"],
    ):
        tree["provider_event_counts"] = {"deepseek": 0, "doubao": 0}
    forged.pop("pilot_audit_sha256")
    forged["pilot_audit_sha256"] = _canonical_sha(forged)
    with pytest.raises(PilotAuditError, match="Evidence-v2 transition proof"):
        verify_pilot_audit(forged)


def test_writer_reaudits_bound_sources_before_emitting_go(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        PilotAuditError,
        snapshot_resume_state,
        write_pilot_audit,
    )

    pilot_root, _, _ = formal_pilot
    forged = _audit_with_zero_call_resume(pilot_root)
    snapshot = snapshot_resume_state(pilot_root)
    forged["accounting"]["known_cost_yuan"] = 0.0
    forged.pop("pilot_audit_sha256")
    forged["pilot_audit_sha256"] = _canonical_sha(forged)

    with pytest.raises(PilotAuditError, match="source re-audit"):
        write_pilot_audit(
            forged,
            tmp_path / "forged-output",
            repo_root=REPO_ROOT,
            pilot_root=pilot_root,
            resume_before=snapshot,
            resume_after=snapshot,
            resume_receipt=_resume_receipt(pilot_root),
            resume_receipt_path=_resume_receipt_path(pilot_root),
        )


def test_authoritative_verifier_reaudits_bound_sources(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    import experiments.llm_sim_v2.audit_pilot as audit_module

    authoritative = getattr(
        audit_module, "verify_pilot_audit_against_sources", None
    )
    assert authoritative is not None
    pilot_root, _, _ = formal_pilot
    report = _audit_with_zero_call_resume(pilot_root)
    snapshot = audit_module.snapshot_resume_state(pilot_root)

    assert authoritative(
        report,
        repo_root=REPO_ROOT,
        pilot_root=pilot_root,
        resume_before=snapshot,
        resume_after=snapshot,
        resume_receipt=_resume_receipt(pilot_root),
        resume_receipt_path=_resume_receipt_path(pilot_root),
    )["verification_scope"] == "authoritative_source_reaudit"

    forged = json.loads(json.dumps(report))
    forged["accounting"]["known_cost_yuan"] = 0.0
    forged.pop("pilot_audit_sha256")
    forged["pilot_audit_sha256"] = _canonical_sha(forged)
    with pytest.raises(audit_module.PilotAuditError, match="source re-audit"):
        authoritative(
            forged,
            repo_root=REPO_ROOT,
            pilot_root=pilot_root,
            resume_before=snapshot,
            resume_after=snapshot,
            resume_receipt=_resume_receipt(pilot_root),
            resume_receipt_path=_resume_receipt_path(pilot_root),
        )


def test_writer_rejects_output_inside_pilot_run_root(
    formal_pilot: tuple[Path, object, list[object]],
) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        PilotAuditError,
        snapshot_resume_state,
        write_pilot_audit,
    )

    pilot_root, _, _ = formal_pilot
    report = _audit_with_zero_call_resume(pilot_root)
    snapshot = snapshot_resume_state(pilot_root)
    forbidden_output = pilot_root.parent / "main"

    with pytest.raises(PilotAuditError, match="outside the pilot run root"):
        write_pilot_audit(
            report,
            forbidden_output,
            repo_root=REPO_ROOT,
            pilot_root=pilot_root,
            resume_before=snapshot,
            resume_after=snapshot,
            resume_receipt=_resume_receipt(pilot_root),
            resume_receipt_path=_resume_receipt_path(pilot_root),
        )
    assert not forbidden_output.exists()


def test_writer_does_not_publish_go_when_post_write_reaudit_changes(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.audit_pilot as audit_module

    pilot_root, _, _ = formal_pilot
    report = _audit_with_zero_call_resume(pilot_root)
    snapshot = audit_module.snapshot_resume_state(pilot_root)
    output = tmp_path / "atomic-output"
    real_audit = audit_module.audit_formal_pilot
    calls = 0

    def changing_audit(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        regenerated = real_audit(**kwargs)
        if calls == 2:
            regenerated["pilot_audit_sha256"] = "0" * 64
        return regenerated

    monkeypatch.setattr(audit_module, "audit_formal_pilot", changing_audit)

    with pytest.raises(audit_module.PilotAuditError, match="source changed"):
        audit_module.write_pilot_audit(
            report,
            output,
            repo_root=REPO_ROOT,
            pilot_root=pilot_root,
            resume_before=snapshot,
            resume_after=snapshot,
            resume_receipt=_resume_receipt(pilot_root),
            resume_receipt_path=_resume_receipt_path(pilot_root),
        )
    assert not output.exists()


def test_cli_accepts_anchored_transition_without_legacy_snapshots(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2.audit_pilot import main

    pilot_root, _, _ = formal_pilot
    assert _audit_with_zero_call_resume(pilot_root)["decision"] == "GO"

    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--pilot-root",
            str(pilot_root),
            "--resume-receipt",
            str(_resume_receipt_path(pilot_root)),
            "--anchor-a",
            str(_anchor_a_path(pilot_root)),
            "--anchor-b",
            str(_anchor_b_path(pilot_root)),
            "--output",
            str(tmp_path / "cli-audit"),
        ]
    )

    assert exit_code == 0


def test_cli_prepares_billing_resolution_as_a_separate_commit_stage(
    formal_pilot: tuple[Path, object, list[object]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import experiments.llm_sim_v2.audit_pilot as audit_module

    pilot_root, _, _ = formal_pilot
    anchor_a = _anchor_a_path(pilot_root)
    anchor_a.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def prepare(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"billing_fact_status": "unresolved_reserved"}

    monkeypatch.setattr(
        audit_module, "prepare_billing_authorization_resolution", prepare
    )
    common = [
        "--repo-root",
        str(REPO_ROOT),
        "--pilot-root",
        str(pilot_root),
        "--anchor-a",
        str(anchor_a),
    ]

    assert audit_module.main([*common, "--prepare-billing-resolution"]) == 0
    assert captured["anchor_a_path"] == anchor_a
    assert json.loads(capsys.readouterr().out)["billing_fact_status"] == (
        "unresolved_reserved"
    )

    for conflicting in ("--prepare-anchor-a", "--run-resume"):
        with pytest.raises(
            audit_module.PilotAuditError, match="separate commit stages"
        ):
            audit_module.main(
                [
                    *common,
                    "--prepare-billing-resolution",
                    conflicting,
                ]
            )


def test_resume_requires_billing_resolution_commit_at_execution_head(
    formal_pilot: tuple[Path, object, list[object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.llm_sim_v2.audit_pilot as audit_module
    from experiments.llm_sim_v2.evidence import write_phase_evidence_receipt

    pilot_root, _, tasks = formal_pilot
    phase = json.loads(
        (pilot_root / "phase_provenance.json").read_text(encoding="utf-8")
    )
    anchor_a = tmp_path / "pilot-phase-anchor-a.json"
    anchor_b = tmp_path / "pilot-phase-anchor-b.json"
    transition = tmp_path / "pilot-resume-transition.json"
    write_phase_evidence_receipt(
        pilot_root,
        output=anchor_a,
        phase_provenance=phase,
        tasks=tasks,
    )
    monkeypatch.setattr(
        audit_module,
        "audit_formal_pilot",
        lambda **_: {
            "blocking_reasons": [],
            "accounting": {"total_accounted_cost_yuan": 24.60518803},
            "gates": {
                "billing_authorization_resolution": {
                    "passed": True,
                    "evidence": {
                        "status": "applied",
                        "receipt_commit": "d" * 40,
                    },
                }
            },
        },
    )

    with pytest.raises(
        audit_module.PilotAuditError, match="billing resolution commit"
    ):
        audit_module.run_zero_call_resume_probe(
            repo_root=REPO_ROOT,
            pilot_root=pilot_root,
            anchor_a_path=anchor_a,
            anchor_b_path=anchor_b,
            transition_receipt_path=transition,
        )

    assert not anchor_b.exists()
    assert not transition.exists()


def test_resume_audit_allows_only_new_immutable_records(tmp_path: Path) -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        compare_resume_audits,
        snapshot_resume_state,
    )

    pilot = tmp_path / "pilot"
    records = pilot / "records" / "deepseek"
    records.mkdir(parents=True)
    first = records / "a.json"
    first.write_text(json.dumps({"task_id": "a", "attempts": [{"attempt": 1}]}))
    before = snapshot_resume_state(pilot)
    second = records / "b.json"
    second.write_text(json.dumps({"task_id": "b", "attempts": [{"attempt": 1}]}))
    after_add = snapshot_resume_state(pilot)

    comparison = compare_resume_audits(before, after_add)
    assert comparison["ok"] is True
    assert comparison["added_record_keys"] == ["deepseek/b"]
    assert comparison["mutated_record_keys"] == []

    formal_comparison = compare_resume_audits(
        before, after_add, require_zero_calls=True
    )
    assert formal_comparison["ok"] is False
    assert formal_comparison["require_zero_calls"] is True
    assert formal_comparison["added_record_keys"] == ["deepseek/b"]
    assert formal_comparison["added_attempt_count"] == 1
    assert formal_comparison["formal_blocking_reasons"] == [
        "added_records_after_completed_pilot",
        "added_attempts_after_completed_pilot",
    ]

    first.write_text(
        json.dumps(
            {"task_id": "a", "attempts": [{"attempt": 1}, {"attempt": 2}]}
        )
    )
    after_mutation = snapshot_resume_state(pilot)
    comparison = compare_resume_audits(after_add, after_mutation)
    assert comparison["ok"] is False
    assert comparison["mutated_record_keys"] == ["deepseek/a"]
    assert comparison["attempt_count_changes"] == [
        {"record_key": "deepseek/a", "before": 1, "after": 2}
    ]


def test_billing_authorization_resolution_requires_exact_reserved_attempts() -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        PilotAuditError,
        build_billing_authorization_resolution_payload,
        validate_billing_authorization_resolution_payload,
    )

    attempts = [
        {
            "provider": provider,
            "task_id": character * 64,
            "attempt": attempt,
            "record": {
                "path": f"records/{provider}/{character * 64}.json",
                "bytes": 1234 + attempt,
                "sha256": character * 64,
                "attempt_sha256": ("a" if provider == "deepseek" else "b") * 64,
            },
            "request": {
                "model": "model-frozen",
                "max_tokens": 1024,
                "wire_message_sha256": ("c" if provider == "deepseek" else "d") * 64,
            },
            "failure": {
                "error_category": "network_timeout",
                "provider_response_received": False,
                "cost_known": False,
                "billing_ambiguity": True,
                "cost_yuan": None,
                "cost_reserve_yuan": 10.0,
            },
            "provider_call_event": {
                "path": f"evidence/provider_events/{provider}/event-{attempt}.json",
                "bytes": 456 + attempt,
                "file_sha256": ("e" if provider == "deepseek" else "f") * 64,
                "event_sha256": ("1" if provider == "deepseek" else "2") * 64,
                "event_index": attempt,
                "invocation_id": f"invocation-{provider}",
            },
        }
        for provider, character, attempt in (
            ("deepseek", "3", 4),
            ("doubao", "4", 1),
        )
    ]
    anchor = {
        "phase_evidence_receipt_sha256": "5" * 64,
        "file_sha256": "6" * 64,
        "file_bytes": 9000,
        "phase_provenance_sha256": "7" * 64,
        "phase_provenance_file_sha256": "8" * 64,
        "store_snapshot": {"file_set_sha256": "9" * 64},
        "providers": {
            provider: {
                "provider_manifest_sha256": character * 64,
                "record_set_sha256": digit * 64,
                "evidence_chain_head_sha256": event * 64,
            }
            for provider, character, digit, event in (
                ("deepseek", "a", "b", "c"),
                ("doubao", "d", "e", "f"),
            )
        },
    }
    accounting = {
        "pre_collection_total_yuan": 2.57152913,
        "phase_known_cost_yuan": 2.0336589,
        "phase_unknown_reserve_yuan": 20.0,
        "phase_accounted_cost_yuan": 22.0336589,
        "total_known_cost_yuan": 4.49218803,
        "total_unknown_reserve_yuan": 20.113,
        "total_accounted_cost_yuan": 24.60518803,
        "hard_fuse_yuan": 450.0,
        "remaining_headroom_yuan": 425.39481197,
    }
    authorization = {
        "analysis_plan": {
            "path": "experiments/h5v2_analysis_plan.md",
            "sha256": "a" * 64,
            "committed_blob_sha256": "a" * 64,
            "anchor_commit": "b" * 40,
        },
        "dispatch_brief": {
            "path": "PROJECT_HANDOFF/codex_briefs/brief.md",
            "sha256": "c" * 64,
        },
        "hard_fuse_policy": "CNY 450 is the only additional-confirmation fuse.",
        "self_review_policy": "Codex may self-review and self-sign with dated evidence.",
    }
    receipt = build_billing_authorization_resolution_payload(
        unknown_attempts=attempts,
        anchor_a=anchor,
        accounting=accounting,
        authorization=authorization,
        runtime_task_manifest_sha256="d" * 64,
        freeze_manifest_sha256="e" * 64,
        reviewer="codex_budget_resolution",
        review_date="2026-07-16",
    )

    assert validate_billing_authorization_resolution_payload(
        receipt,
        expected_unknown_attempts=attempts,
        expected_anchor_a=anchor,
        expected_accounting=accounting,
        expected_authorization=authorization,
        expected_runtime_task_manifest_sha256="d" * 64,
        expected_freeze_manifest_sha256="e" * 64,
    ) == receipt["billing_authorization_resolution_sha256"]
    assert receipt["billing_fact_status"] == "unresolved_reserved"
    assert receipt["action_disposition"] == (
        "continue_under_preexisting_budget_authorization"
    )

    def use_bool_for_doubao_attempt(row: dict[str, object]) -> None:
        row["unknown_attempts"][1]["attempt"] = True
        row["unknown_attempt_set_sha256"] = _canonical_sha(
            row["unknown_attempts"]
        )

    def use_bool_for_doubao_event_index(row: dict[str, object]) -> None:
        row["unknown_attempts"][1]["provider_call_event"][
            "event_index"
        ] = True
        row["unknown_attempt_set_sha256"] = _canonical_sha(
            row["unknown_attempts"]
        )

    def use_int_for_failure_boolean(row: dict[str, object]) -> None:
        row["unknown_attempts"][1]["failure"][
            "provider_response_received"
        ] = 0
        row["unknown_attempt_set_sha256"] = _canonical_sha(
            row["unknown_attempts"]
        )

    mutations = (
        lambda row: row["unknown_attempts"].pop(),
        lambda row: row["unknown_attempts"].append(row["unknown_attempts"][0]),
        use_bool_for_doubao_attempt,
        use_bool_for_doubao_event_index,
        use_int_for_failure_boolean,
        lambda row: row["unknown_attempts"][0]["record"].__setitem__(
            "attempt_sha256", "0" * 64
        ),
        lambda row: row["unknown_attempts"][0][
            "provider_call_event"
        ].__setitem__("event_sha256", "0" * 64),
        lambda row: row["accounting"].__setitem__("phase_unknown_reserve_yuan", 0),
        lambda row: row["accounting"].__setitem__(
            "total_accounted_cost_yuan", 1.0
        ),
        lambda row: row["anchor_a"].__setitem__("file_sha256", "0" * 64),
        lambda row: row["authorization"].__setitem__(
            "hard_fuse_policy", "confirmation waived"
        ),
        lambda row: row.__setitem__("runtime_task_manifest_sha256", "0" * 64),
        lambda row: row.__setitem__("freeze_manifest_sha256", "0" * 64),
        lambda row: row.__setitem__("reviewer", "codex_"),
        lambda row: row.__setitem__("review_date", "not-a-date"),
        lambda row: row.__setitem__("billing_fact_status", "resolved"),
        lambda row: row["safeguards"].__setitem__(
            "other_audit_blockers_waived", 0
        ),
        lambda row: row["safeguards"].__setitem__(
            "scientific_outcomes_unmodified", False
        ),
    )
    for mutate in mutations:
        forged = json.loads(json.dumps(receipt))
        mutate(forged)
        forged.pop("billing_authorization_resolution_sha256")
        forged["billing_authorization_resolution_sha256"] = _canonical_sha(forged)
        with pytest.raises(PilotAuditError):
            validate_billing_authorization_resolution_payload(
                forged,
                expected_unknown_attempts=attempts,
                expected_anchor_a=anchor,
                expected_accounting=accounting,
                expected_authorization=authorization,
                expected_runtime_task_manifest_sha256="d" * 64,
                expected_freeze_manifest_sha256="e" * 64,
            )


def test_billing_authorization_resolution_cannot_override_hard_fuse() -> None:
    from experiments.llm_sim_v2.audit_pilot import (
        PilotAuditError,
        build_billing_authorization_resolution_payload,
    )

    with pytest.raises(PilotAuditError, match="hard fuse"):
        build_billing_authorization_resolution_payload(
            unknown_attempts=[{"provider": "deepseek"}],
            anchor_a={"phase_evidence_receipt_sha256": "a" * 64},
            accounting={
                "phase_unknown_reserve_yuan": 10.0,
                "total_accounted_cost_yuan": 450.0,
                "hard_fuse_yuan": 450.0,
                "remaining_headroom_yuan": 0.0,
            },
            authorization={},
            runtime_task_manifest_sha256="b" * 64,
            freeze_manifest_sha256="c" * 64,
            reviewer="codex_budget_resolution",
            review_date="2026-07-16",
        )
