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
