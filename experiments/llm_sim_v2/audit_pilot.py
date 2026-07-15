"""Read-only, fail-closed approval gate for the formal Persona-v2 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from .prompts import assert_blind_no_leakage
from .collect import verify_formal_carried_forward_cost_ledger
from .evidence import (
    REVIEWED_CARRIED_LEDGER_SHA256,
    REVIEWED_LEGACY_KNOWN_COST_YUAN,
    REVIEWED_LEGACY_RECEIPT_SHA256,
    REVIEWED_LEGACY_RECORD_SET_SHA256,
    build_phase_evidence_receipt,
    validate_v2_response_record,
    write_phase_evidence_receipt,
)
from .runner import (
    BudgetLedger,
    FROZEN_COMMIT,
    Task,
    V2ProviderRunner,
    compute_outcomes,
    enumerate_tasks,
    load_runtime_contract,
    parse_provider_output,
    phase_provenance_binding,
    validate_formal_phase_provenance,
    verify_phase_provenance_against_contract,
    verify_runtime_task_manifest,
)
from .store import RUN_ID


EXPECTED_PILOT_PROVIDERS = ("deepseek", "doubao")
EXPECTED_TASKS_PER_PROVIDER = 128
MIN_COMPLETE_FRACTION = 0.80
MAX_INVALID_SCHEMA_FRACTION = 0.20
DEFAULT_OUTPUT_DIR = Path("/tmp/yher_h5v2/formal_pilot_audit")
DEFAULT_ANCHOR_A = Path(
    "experiments/llm_sim_v2/evidence_anchors/replacement_pilot_phase_anchor_a.json"
)
DEFAULT_ANCHOR_B = Path(
    "experiments/llm_sim_v2/evidence_anchors/replacement_pilot_phase_anchor_b.json"
)
DEFAULT_TRANSITION_RECEIPT = Path(
    "experiments/llm_sim_v2/evidence_anchors/replacement_pilot_resume_transition.json"
)
DEFAULT_BILLING_AUTHORIZATION_RESOLUTION = Path(
    "experiments/llm_sim_v2/evidence_anchors/"
    "replacement_pilot_unknown_billing_budget_disposition.json"
)
ANALYSIS_PLAN_RELATIVE = Path("experiments/h5v2_analysis_plan.md")
DISPATCH_BRIEF_RELATIVE = Path(
    "PROJECT_HANDOFF/codex_briefs/"
    "2026-07-15_persona双条件v2与期刊论文总攻.md"
)
REVIEWED_DISPATCH_BRIEF_SHA256 = (
    "9245652e033b20b9f1094ff4c4cfd6b5cb0fbbad26e3ff3d216e2a6c9261f75a"
)
HARD_FUSE_POLICY = "CNY 450 is the only additional-confirmation fuse."
SELF_REVIEW_POLICY = "Codex may self-review and self-sign with dated evidence."
_BILLING_REVIEWER_PATTERN = re.compile(r"codex_[a-z0-9][a-z0-9_]*")
EXPECTED_GATE_NAMES = (
    "auditor_git_provenance",
    "runtime_and_phase_provenance",
    "formal_scope",
    "exact_task_states",
    "condition_cells",
    "model_identity",
    "text_only_and_leakage",
    "pilot_main_isolation",
    "phase_evidence_receipts",
    "billing_authorization_resolution",
    "accounting_reconciliation",
    "lifecycle_disclosures",
    "zero_call_resume",
)


class PilotAuditError(ValueError):
    """Raised when an audit input cannot be parsed without guessing."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_implementation_proof(repo_root: Path) -> dict[str, Any]:
    relative = Path("experiments/llm_sim_v2/audit_pilot.py")
    current_path = Path(__file__).resolve()
    expected_path = (repo_root / relative).resolve(strict=False)
    current_sha = sha256_file(current_path)

    def git(*args: str, binary: bool = False) -> bytes | str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout if binary else result.stdout.decode("utf-8").strip()

    head: str | None = None
    committed_sha: str | None = None
    auditor_commit: str | None = None
    try:
        head = str(git("rev-parse", "HEAD"))
        git("ls-files", "--error-unmatch", relative.as_posix())
        committed_bytes = git("show", f"HEAD:{relative.as_posix()}", binary=True)
        assert isinstance(committed_bytes, bytes)
        committed_sha = hashlib.sha256(committed_bytes).hexdigest()
        auditor_commit = str(
            git("log", "-1", "--format=%H", "--", relative.as_posix())
        ) or None
    except (AssertionError, OSError, subprocess.CalledProcessError, UnicodeError):
        pass
    passed = (
        current_path == expected_path
        and isinstance(head, str)
        and len(head) == 40
        and committed_sha == current_sha
        and isinstance(auditor_commit, str)
        and len(auditor_commit) == 40
    )
    return {
        "passed": passed,
        "path": relative.as_posix(),
        "auditor_implementation_sha256": current_sha,
        "committed_blob_sha256": committed_sha,
        "audit_git_head": head,
        "auditor_commit": auditor_commit,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotAuditError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise PilotAuditError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _safe_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError as exc:
        raise PilotAuditError("pilot artifact escapes the formal pilot root") from exc


def _source_file_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PilotAuditError(f"pilot source contains a symlink: {path}")
        if not path.is_file():
            continue
        rows.append(
            {
                "path": _safe_relative(root, path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _phase_store_file_rows(root: Path) -> list[dict[str, Any]]:
    """Mirror the evidence writer's store snapshot with path-level rows."""

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PilotAuditError(f"pilot source contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = _safe_relative(root, path)
        if relative.startswith("evidence/phase_receipts/"):
            continue
        if path.name.startswith(".") and ".tmp-" in path.name:
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _verify_phase_receipt_envelope(receipt: Mapping[str, Any]) -> str:
    payload = dict(receipt)
    advertised = payload.pop("phase_evidence_receipt_sha256", None)
    if (
        receipt.get("schema_version")
        != "yher.llm_sim_v2.phase_evidence_receipt.v1"
        or receipt.get("simulated") is not True
        or receipt.get("run_id") != RUN_ID
        or receipt.get("phase") != "pilot"
        or receipt.get("authority") != "post_invocation_phase_receipt"
        or not isinstance(advertised, str)
        or advertised != canonical_sha256(payload)
    ):
        raise PilotAuditError("phase evidence receipt envelope is invalid")
    return advertised


def _phase_anchor_summary(
    receipt: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    return {
        "phase_evidence_receipt_sha256": _verify_phase_receipt_envelope(receipt),
        "file_sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
    }


def _validate_evidence_tree(
    pilot_root: Path,
    *,
    providers: Sequence[str],
    required_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    evidence_root = pilot_root / "evidence"
    expected_top = {"phase_receipts", "provider_events", "provider_locks"}
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise PilotAuditError("pilot evidence root is missing or invalid")
    actual_top = {path.name for path in evidence_root.iterdir()}
    if actual_top != expected_top or any(path.is_symlink() for path in evidence_root.iterdir()):
        raise PilotAuditError("pilot evidence root has an unexplained entry")

    provider_set = set(providers)
    event_root = evidence_root / "provider_events"
    if (
        not event_root.is_dir()
        or event_root.is_symlink()
        or {path.name for path in event_root.iterdir()} != provider_set
        or any(path.is_symlink() or not path.is_dir() for path in event_root.iterdir())
    ):
        raise PilotAuditError("pilot provider evidence roster is invalid")
    event_counts: dict[str, int] = {}
    for provider in providers:
        entries = list((event_root / provider).iterdir())
        if any(
            path.is_symlink() or not path.is_file() or path.suffix != ".json"
            for path in entries
        ):
            raise PilotAuditError("pilot provider evidence stream has an unbound entry")
        event_counts[provider] = len(entries)

    lock_root = evidence_root / "provider_locks"
    expected_locks = {f"{provider}.lock" for provider in providers}
    if (
        not lock_root.is_dir()
        or lock_root.is_symlink()
        or {path.name for path in lock_root.iterdir()} != expected_locks
        or any(path.is_symlink() or not path.is_file() for path in lock_root.iterdir())
    ):
        raise PilotAuditError("pilot provider evidence lock set is invalid")

    receipt_root = evidence_root / "phase_receipts"
    if receipt_root.is_symlink() or not receipt_root.is_dir():
        raise PilotAuditError("pilot internal phase receipt root is missing")
    receipts: dict[str, dict[str, Any]] = {}
    for path in receipt_root.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise PilotAuditError("pilot internal phase receipt root has an unbound entry")
        receipt = _read_json(path)
        digest = _verify_phase_receipt_envelope(receipt)
        if path.name != f"{digest}.json" or digest in receipts:
            raise PilotAuditError("pilot internal phase receipt identity is invalid")
        receipts[digest] = {
            "path": _safe_relative(pilot_root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    required_digests = {
        _verify_phase_receipt_envelope(receipt) for receipt in required_receipts
    }
    if not required_digests <= set(receipts):
        raise PilotAuditError("pilot internal A/B phase receipt copy is missing")
    receipt_files = sorted(
        (dict(row) for row in receipts.values()),
        key=lambda row: str(row["path"]),
    )
    return {
        "top_level_entries": sorted(actual_top),
        "provider_event_counts": event_counts,
        "provider_lock_files": sorted(expected_locks),
        "phase_receipts": receipts,
        "phase_receipt_files": receipt_files,
        "phase_receipt_file_set_sha256": canonical_sha256(receipt_files),
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    head = result.stdout.strip()
    if len(head) != 40:
        raise PilotAuditError("active Git HEAD is invalid")
    return head


def _git_committed_file_proof(
    repo_root: Path,
    path: Path,
    *,
    head: str | None = None,
) -> dict[str, Any]:
    target = str(head or _git_head(repo_root))
    resolved = path.expanduser().resolve(strict=True)
    try:
        relative = resolved.relative_to(repo_root).as_posix()
    except ValueError:
        relative = ""
    current_sha = sha256_file(resolved)
    committed_sha: str | None = None
    anchor_commit: str | None = None
    resolved_head: str | None = None
    try:
        if not relative:
            raise ValueError
        resolved_head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", target],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        tracked = subprocess.run(
            ["git", "-C", str(repo_root), "ls-tree", "--name-only", resolved_head, "--", relative],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.splitlines()
        if tracked != [relative]:
            raise ValueError
        committed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{resolved_head}:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
        committed_sha = hashlib.sha256(committed).hexdigest()
        anchor_commit = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%H", resolved_head, "--", relative],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        pass
    passed = (
        bool(relative)
        and len(resolved_head or "") == 40
        and committed_sha == current_sha
        and len(anchor_commit or "") == 40
    )
    return {
        "passed": passed,
        "path": relative or str(resolved),
        "file_sha256": current_sha,
        "committed_blob_sha256": committed_sha,
        "git_head": resolved_head,
        "anchor_commit": anchor_commit,
    }


def _git_is_ancestor(
    repo_root: Path,
    ancestor: str,
    descendant: str,
    *,
    strict: bool = False,
) -> bool:
    if strict and ancestor == descendant:
        return False
    result = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


class _Blockers:
    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._seen: set[str] = set()

    def add(
        self,
        code: str,
        message: str,
        *,
        provider: str | None = None,
        task_id: str | None = None,
    ) -> None:
        row: dict[str, Any] = {"code": code, "message": message}
        if provider is not None:
            row["provider"] = provider
        if task_id is not None:
            row["task_id"] = task_id
        key = canonical_sha256(row)
        if key not in self._seen:
            self._seen.add(key)
            self._rows.append(row)

    def has(self, *codes: str) -> bool:
        wanted = set(codes)
        return any(str(row["code"]) in wanted for row in self._rows)

    def rows(self) -> list[dict[str, Any]]:
        return sorted(
            self._rows,
            key=lambda row: (
                str(row["code"]),
                str(row.get("provider") or ""),
                str(row.get("task_id") or ""),
                str(row["message"]),
            ),
        )


def _phase_binding(phase: Mapping[str, Any]) -> dict[str, Any]:
    return phase_provenance_binding(phase)


def _reviewed_carried_forward_cost(repo_root: Path) -> dict[str, Any]:
    path = (
        repo_root
        / "experiments/llm_sim_v2/evidence_anchors/"
        "legacy_pilot_carried_forward_cost.json"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotAuditError(
            "reviewed carried-forward cost ledger is missing or invalid"
        ) from exc
    return verify_formal_carried_forward_cost_ledger(value)


def _scope_from_phase(phase: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: phase.get(key)
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


def _text_only_messages(value: Any) -> bool:
    if isinstance(value, Mapping):
        forbidden = {"image", "image_url", "input_image", "media_url", "audio_url"}
        if any(str(key).strip().lower() in forbidden for key in value):
            return False
        return all(_text_only_messages(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_text_only_messages(child) for child in value)
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        try:
            return _text_only_messages(json.loads(value))
        except json.JSONDecodeError:
            return True
    return not isinstance(value, (bytes, bytearray, memoryview))


def _audit_text_and_leakage(contract: Any, tasks: Sequence[Task]) -> dict[str, Any]:
    persona_by_row = {
        str(row["row_id"]): row for row in contract.personas if isinstance(row, Mapping)
    }
    blind_checked = 0
    violations: list[str] = []
    if contract.config.get("modality_condition") != "text_only":
        violations.append("frozen modality is not text_only")
    for task in tasks:
        if not _text_only_messages(task.messages) or not _text_only_messages(
            task.wire_messages
        ):
            violations.append(f"non-text message payload: {task.task_id}")
        if task.condition != "blind":
            continue
        blind_checked += 1
        persona = persona_by_row.get(task.row_id)
        if persona is None:
            violations.append(f"blind task lacks frozen persona row: {task.task_id}")
            continue
        try:
            assert_blind_no_leakage(
                task.messages,
                persona=persona,
                item=task.item,
                frozen_leakage_lexicon=contract.lexicon,
            )
        except AssertionError as exc:
            violations.append(f"{task.task_id}: {exc}")
    return {
        "passed": not violations,
        "modality_condition": contract.config.get("modality_condition"),
        "blind_task_count_checked": blind_checked,
        "violations": sorted(set(violations)),
        "proof_basis": (
            "re-rendered committed tasks plus frozen recursive blind-leakage assertion"
        ),
    }


def _record_identity(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "logical_key": task.logical_key,
        "persona_id": task.persona_id,
        "pair_id": task.pair_id,
        "row_id": task.row_id,
        "anchor_id": task.anchor_id,
        "target_node": task.target_node,
        "response_arm": task.response_arm,
        "condition": task.condition,
        "item_id": task.item_id,
        "family_id": task.family_id,
        "is_stability_repeat": task.is_stability_repeat,
        "attempt_id": task.attempt_id,
        "message_sha256": task.message_sha256,
        "wire_message_sha256": task.wire_message_sha256,
        "prompt_revision": task.prompt_revision,
        "prompt_contract_sha256": task.prompt_contract_sha256,
    }


def _finite_nonnegative(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("value must be finite and non-negative")
    return parsed


def _audit_record(
    record: Mapping[str, Any],
    *,
    task: Task,
    provider: str,
    model: str,
    max_attempts: int,
    max_tokens: int,
    retry_max_tokens: int,
    unknown_attempt_reserve_yuan: float,
    phase_binding: Mapping[str, Any],
    blockers: _Blockers,
) -> dict[str, Any]:
    identity_errors = [
        field
        for field, expected in _record_identity(task).items()
        if record.get(field) != expected
    ]
    envelope = {
        "schema_version": "yher.llm_sim_v2.response_record.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "analysis_population": "pilot",
        "collection_mode": "formal",
        "development_only": False,
        "partial": False,
        "formal_analysis_eligible": True,
        "provider": provider,
        "requested_model": model,
    }
    identity_errors.extend(
        field for field, expected in envelope.items() if record.get(field) != expected
    )
    if identity_errors:
        blockers.add(
            "record_identity_drift",
            "record differs from frozen task/envelope: "
            + ", ".join(sorted(set(identity_errors))),
            provider=provider,
            task_id=task.task_id,
        )
    if record.get("provenance") != phase_binding:
        blockers.add(
            "record_provenance_drift",
            "record provenance differs from verified formal phase binding",
            provider=provider,
            task_id=task.task_id,
        )
    try:
        validate_v2_response_record(
            record,
            provider=provider,
            requested_model=model,
            phase="pilot",
            task=task,
            expected_provenance=phase_binding,
        )
    except (TypeError, ValueError) as exc:
        blockers.add(
            "record_identity_drift",
            f"response_record.v2 strict replay failed: {exc}",
            provider=provider,
            task_id=task.task_id,
        )

    status = str(record.get("status") or "")
    allowed_statuses = {
        "complete",
        "excluded_schema",
        "excluded_model_drift",
        "technical_failure",
    }
    if status not in allowed_statuses:
        blockers.add(
            "record_identity_drift",
            f"unknown record status: {status!r}",
            provider=provider,
            task_id=task.task_id,
        )
    returned_model = record.get("model_id")
    if status == "excluded_model_drift" or (
        returned_model not in (None, "") and returned_model != model
    ):
        blockers.add(
            "model_drift",
            f"returned model differs from frozen model {model}",
            provider=provider,
            task_id=task.task_id,
        )
    if status in {"complete", "excluded_schema"} and returned_model != model:
        blockers.add(
            "model_drift",
            "response-bearing record lacks the exact frozen model identity",
            provider=provider,
            task_id=task.task_id,
        )

    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        blockers.add(
            "attempt_reconciliation",
            "record has no attempt ledger",
            provider=provider,
            task_id=task.task_id,
        )
        attempts = []
    if len(attempts) > max_attempts or record.get("retry_count") != max(
        0, len(attempts) - 1
    ):
        blockers.add(
            "attempt_reconciliation",
            "attempt count exceeds policy or retry_count does not reconcile",
            provider=provider,
            task_id=task.task_id,
        )

    known_cost = 0.0
    reserve = 0.0
    unknown_attempts = 0
    input_tokens = 0
    output_tokens = 0
    token_error = False
    attempt_error = False
    model_attempt_drift = False
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping) or attempt.get("attempt") != index:
            attempt_error = True
            continue
        attempt_status = attempt.get("status")
        if attempt_status not in {"response", "failed"}:
            attempt_error = True
        if index < len(attempts) and (
            attempt_status != "failed"
            or not isinstance(attempt.get("error_category"), str)
            or not str(attempt.get("error_category")).strip()
        ):
            attempt_error = True
        requested = attempt.get("request_max_tokens")
        if requested is not None and int(requested) not in {
            int(max_tokens),
            int(retry_max_tokens),
        }:
            attempt_error = True
        attempt_model = str(attempt.get("model_returned") or "")
        if attempt_model and attempt_model != model:
            model_attempt_drift = True
        usage = attempt.get("usage")
        if usage is not None:
            if not isinstance(usage, Mapping):
                token_error = True
            else:
                try:
                    input_value = int(usage.get("input_tokens") or 0)
                    output_value = int(usage.get("output_tokens") or 0)
                    if input_value < 0 or output_value < 0:
                        raise ValueError
                    input_tokens += input_value
                    output_tokens += output_value
                except (TypeError, ValueError):
                    token_error = True
        try:
            if attempt.get("cost_known") is True:
                cost = _finite_nonnegative(attempt.get("cost_yuan"))
                if (
                    float(attempt.get("cost_reserve_yuan", 0.0)) != 0.0
                    or attempt.get("billing_ambiguity") is not False
                ):
                    raise ValueError
                known_cost += cost
            elif attempt.get("cost_known") is False:
                attempt_reserve = _finite_nonnegative(
                    attempt.get("cost_reserve_yuan")
                )
                if (
                    attempt.get("cost_yuan") is not None
                    or attempt.get("billing_ambiguity") is not True
                    or not math.isclose(
                        attempt_reserve,
                        unknown_attempt_reserve_yuan,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    )
                ):
                    raise ValueError
                reserve += attempt_reserve
                unknown_attempts += 1
            else:
                raise ValueError
        except (TypeError, ValueError):
            attempt_error = True
    if attempt_error:
        blockers.add(
            "attempt_reconciliation",
            "attempt sequence, token policy, or per-attempt billing identity is invalid",
            provider=provider,
            task_id=task.task_id,
        )
    if token_error:
        blockers.add(
            "token_reconciliation",
            "attempt token usage is invalid",
            provider=provider,
            task_id=task.task_id,
        )
    if model_attempt_drift:
        blockers.add(
            "model_drift",
            "an attempt returned a model outside the frozen provider binding",
            provider=provider,
            task_id=task.task_id,
        )

    null_outcomes = {
        "is_correct": None,
        "target_option_hit": None,
        "manipulation_compliance": None,
    }
    response_semantics_valid = True
    parsed = record.get("parsed_output")
    outcomes = record.get("outcomes")
    final_attempt = attempts[-1] if attempts else {}
    if status == "complete":
        try:
            if not isinstance(parsed, Mapping):
                raise ValueError
            option_keys = {
                str(key).strip().upper()
                for key in (task.item.get("options") or {})
            }
            reparsed = parse_provider_output(
                json.dumps(parsed, ensure_ascii=False),
                condition=task.condition,
                option_keys=option_keys,
            )
            expected_outcomes = compute_outcomes(
                condition=task.condition,
                response_arm=task.response_arm,
                answer=reparsed.get("answer"),
                abstain=bool(reparsed.get("abstain")),
                correct_option=task.correct_option,
                target_option=task.target_option,
            )
            final_usage = final_attempt.get("usage")
            metering_valid = (
                isinstance(final_usage, Mapping)
                and int(final_usage.get("input_tokens") or 0)
                + int(final_usage.get("output_tokens") or 0)
                > 0
                and final_attempt.get("cost_known") is True
                and float(final_attempt.get("cost_yuan")) > 0.0
            )
            response_semantics_valid = (
                dict(parsed) == reparsed
                and outcomes == expected_outcomes
                and record.get("error") is None
                and isinstance(final_attempt, Mapping)
                and final_attempt.get("status") == "response"
                and "error_category" not in final_attempt
                and metering_valid
            )
        except (TypeError, ValueError):
            response_semantics_valid = False
    elif status == "excluded_schema":
        response_semantics_valid = (
            parsed is None
            and outcomes == null_outcomes
            and isinstance(record.get("error"), str)
            and bool(str(record.get("error")).strip())
            and isinstance(final_attempt, Mapping)
            and final_attempt.get("status") == "failed"
            and final_attempt.get("error_category") == "invalid_schema"
        )
    elif status == "technical_failure":
        response_semantics_valid = (
            parsed is None
            and outcomes == null_outcomes
            and isinstance(record.get("error"), str)
            and bool(str(record.get("error")).strip())
            and isinstance(final_attempt, Mapping)
            and final_attempt.get("status") == "failed"
        )
    elif status == "excluded_model_drift":
        response_semantics_valid = parsed is None and outcomes == null_outcomes
    if not response_semantics_valid:
        blockers.add(
            "response_semantics",
            "record status, parsed output, outcomes, and final attempt do not reconcile",
            provider=provider,
            task_id=task.task_id,
        )

    known_cost = round(known_cost, 8)
    reserve = round(reserve, 8)
    accounted = round(known_cost + reserve, 8)
    try:
        stored_known = float(record.get("known_cost_yuan"))
        stored_reserve = float(record.get("unknown_cost_reserve_yuan"))
        stored_accounted = float(record.get("cost_yuan"))
        cost_matches = (
            math.isclose(stored_known, known_cost, rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(stored_reserve, reserve, rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(stored_accounted, accounted, rel_tol=0.0, abs_tol=1e-8)
            and record.get("has_unknown_cost_attempts") is bool(unknown_attempts)
            and record.get("needs_user") is bool(unknown_attempts)
            and record.get("needs_user_reasons")
            == (
                ["unknown_provider_billing_reserved"]
                if unknown_attempts
                else []
            )
        )
    except (TypeError, ValueError):
        cost_matches = False
    if not cost_matches:
        blockers.add(
            "budget_reconciliation",
            "record totals do not reconcile with immutable attempts/reserves",
            provider=provider,
            task_id=task.task_id,
        )
    return {
        "status": status,
        "attempt_count": len(attempts),
        "unknown_attempt_count": unknown_attempts,
        "known_cost_yuan": known_cost,
        "unknown_cost_reserve_yuan": reserve,
        "accounted_cost_yuan": accounted,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _validate_lifecycle_history(
    pilot_root: Path,
    *,
    provider: str,
    manifest: Mapping[str, Any],
    phase_binding: Mapping[str, Any],
    blockers: _Blockers,
) -> list[dict[str, Any]]:
    root = pilot_root / "provider_lifecycle" / provider
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        blockers.add(
            "lifecycle_integrity",
            "provider lifecycle history directory is missing",
            provider=provider,
        )
        return rows
    unexpected_entries = sorted(
        path.name
        for path in root.iterdir()
        if not path.is_file() or path.suffix != ".json"
    )
    if unexpected_entries:
        blockers.add(
            "lifecycle_integrity",
            "provider lifecycle directory contains unexplained entries: "
            + ", ".join(unexpected_entries),
            provider=provider,
        )
    for index, path in enumerate(sorted(root.glob("*.json"))):
        try:
            row = _read_json(path)
            payload = dict(row)
            advertised = payload.pop("lifecycle_event_sha256", None)
            valid = (
                row.get("schema_version")
                == "yher.llm_sim_v2.provider_lifecycle_event.v1"
                and row.get("simulated") is True
                and row.get("run_id") == RUN_ID
                and row.get("phase") == "pilot"
                and row.get("analysis_population") == "pilot"
                and row.get("provider") == provider
                and row.get("event_index") == index
                and row.get("provenance") == phase_binding
                and advertised == canonical_sha256(payload)
                and path.name == f"{index:04d}-{advertised}.json"
            )
        except (PilotAuditError, TypeError, ValueError):
            valid = False
            row = {}
        if not valid:
            blockers.add(
                "lifecycle_integrity",
                f"provider lifecycle event is invalid: {path.name}",
                provider=provider,
            )
        rows.append(row)
    history = manifest.get("lifecycle_history")
    expected_history = [
        {
            "event_index": row.get("event_index"),
            "provider_lifecycle": row.get("provider_lifecycle"),
            "finished_at_utc": row.get("finished_at_utc"),
            "lifecycle_event_sha256": row.get("lifecycle_event_sha256"),
            "path": (
                f"provider_lifecycle/{provider}/"
                f"{int(row.get('event_index') or 0):04d}-"
                f"{row.get('lifecycle_event_sha256')}.json"
            ),
        }
        for row in rows
    ]
    if not rows or history != expected_history:
        blockers.add(
            "lifecycle_integrity",
            "provider lifecycle_history does not bind the immutable event chain",
            provider=provider,
        )
    if rows:
        last = rows[-1]
        for field in (
            "provider_lifecycle",
            "lifecycle",
            "interruption",
            "unavailable",
            "needs_user",
            "provenance",
            "finished_at_utc",
        ):
            if last.get(field) != manifest.get(field):
                blockers.add(
                    "lifecycle_integrity",
                    f"latest lifecycle event differs from provider manifest: {field}",
                    provider=provider,
                )
    return rows


def _audit_provider(
    pilot_root: Path,
    *,
    provider: str,
    tasks: Sequence[Task],
    contract: Any,
    phase_binding: Mapping[str, Any],
    blockers: _Blockers,
) -> dict[str, Any]:
    expected_ids = [task.task_id for task in tasks]
    expected_set = set(expected_ids)
    task_by_id = {task.task_id: task for task in tasks}
    model = contract.provider_model(provider)
    policy = contract.provider_policy(provider)
    records_root = pilot_root / "records" / provider
    record_paths = sorted(records_root.glob("*.json")) if records_root.is_dir() else []
    unexpected_entries = (
        sorted(
            path.name
            for path in records_root.iterdir()
            if not path.is_file() or path.suffix != ".json"
        )
        if records_root.is_dir()
        else []
    )
    if not records_root.is_dir():
        blockers.add(
            "provider_records_missing",
            "provider records directory is missing",
            provider=provider,
        )
    if unexpected_entries:
        blockers.add(
            "unexplained_record_ids",
            "provider records directory contains non-record entries: "
            + ", ".join(unexpected_entries),
            provider=provider,
        )

    file_ids = [path.stem for path in record_paths]
    unexplained = sorted(set(file_ids) - expected_set)
    if unexplained:
        blockers.add(
            "unexplained_record_ids",
            "record filenames contain task IDs outside the frozen pilot roster",
            provider=provider,
        )
    records: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for path in record_paths:
        file_task_id = path.stem
        try:
            record = _read_json(path)
        except PilotAuditError:
            blockers.add(
                "record_identity_drift",
                "record JSON cannot be parsed",
                provider=provider,
                task_id=file_task_id,
            )
            continue
        if record.get("task_id") != file_task_id:
            blockers.add(
                "unexplained_record_ids",
                "record filename and internal task_id differ",
                provider=provider,
                task_id=file_task_id,
            )
        if file_task_id not in expected_set:
            continue
        records[file_task_id] = record
        metrics[file_task_id] = _audit_record(
            record,
            task=task_by_id[file_task_id],
            provider=provider,
            model=model,
            max_attempts=policy.max_attempts,
            max_tokens=policy.max_tokens,
            retry_max_tokens=policy.retry_max_tokens,
            unknown_attempt_reserve_yuan=float(
                contract.prior_cost_ledger["unknown_attempt_reserve_yuan"]
            ),
            phase_binding=phase_binding,
            blockers=blockers,
        )

    manifest_path = pilot_root / "provider_manifests" / f"{provider}.json"
    manifest: dict[str, Any] = {}
    try:
        manifest = _read_json(manifest_path)
    except PilotAuditError:
        blockers.add(
            "provider_manifest_missing",
            "provider manifest is missing or invalid",
            provider=provider,
        )
    formal_fields = {
        "schema_version": "yher.llm_sim_v2.provider_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "analysis_population": "pilot",
        "provider": provider,
        "collection_mode": "formal",
        "development_only": False,
        "partial": False,
        "formal_analysis_eligible": True,
        "requested_model": model,
        "freeze_commit": FROZEN_COMMIT,
        "prompt_revision": tasks[0].prompt_revision,
        "provenance": phase_binding,
    }
    drifted_manifest_fields = [
        field for field, expected in formal_fields.items() if manifest.get(field) != expected
    ]
    if drifted_manifest_fields:
        blockers.add(
            "formal_scope",
            "provider manifest is partial, development-only, or provenance-drifted: "
            + ", ".join(sorted(drifted_manifest_fields)),
            provider=provider,
        )

    present_ids = [task_id for task_id in expected_ids if task_id in records]
    missing_ids = [task_id for task_id in expected_ids if task_id not in records]
    lifecycle = manifest.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        lifecycle = {}
        blockers.add(
            "lifecycle_integrity",
            "provider lifecycle summary is missing",
            provider=provider,
        )
    lifecycle_lists: dict[str, list[str]] = {}
    for field in (
        "expected_task_ids",
        "present_task_ids",
        "missing_task_ids",
        "interrupted_task_ids",
        "fuse_skipped_task_ids",
        "breaker_skipped_task_ids",
        "unclassified_missing_task_ids",
    ):
        raw = lifecycle.get(field)
        lifecycle_lists[field] = [str(value) for value in raw] if isinstance(raw, list) else []
    lifecycle_valid = (
        lifecycle.get("expected_count") == EXPECTED_TASKS_PER_PROVIDER
        and lifecycle_lists["expected_task_ids"] == expected_ids
        and lifecycle.get("present_count") == len(present_ids)
        and lifecycle_lists["present_task_ids"] == present_ids
        and lifecycle.get("missing_count") == len(missing_ids)
        and lifecycle_lists["missing_task_ids"] == missing_ids
        and lifecycle.get("interrupted_count")
        == len(lifecycle_lists["interrupted_task_ids"])
        and lifecycle.get("fuse_skipped_count")
        == len(lifecycle_lists["fuse_skipped_task_ids"])
        and lifecycle.get("breaker_skipped_count")
        == len(lifecycle_lists["breaker_skipped_task_ids"])
    )
    classified_lists = {
        "missing_interrupted": lifecycle_lists["interrupted_task_ids"],
        "missing_fuse_skipped": lifecycle_lists["fuse_skipped_task_ids"],
        "missing_breaker_skipped": lifecycle_lists["breaker_skipped_task_ids"],
        "missing_unclassified": lifecycle_lists["unclassified_missing_task_ids"],
    }
    seen_classified: set[str] = set()
    for values in classified_lists.values():
        if (
            len(values) != len(set(values))
            or not set(values) <= set(missing_ids)
            or seen_classified & set(values)
        ):
            lifecycle_valid = False
        seen_classified.update(values)
    unavailable = manifest.get("unavailable")
    unavailable_flag = isinstance(unavailable, Mapping) and unavailable.get(
        "unavailable"
    ) is True
    if not lifecycle_valid:
        blockers.add(
            "lifecycle_integrity",
            "provider lifecycle counts or task-ID partitions do not reconcile",
            provider=provider,
        )

    states: dict[str, str] = {}
    for task_id in expected_ids:
        if task_id in metrics:
            states[task_id] = f"record_{metrics[task_id]['status']}"
            continue
        if unavailable_flag and task_id in missing_ids:
            states[task_id] = "missing_unavailable"
            continue
        matches = [
            state for state, values in classified_lists.items() if task_id in values
        ]
        states[task_id] = (
            matches[0]
            if len(matches) == 1 and matches[0] != "missing_unclassified"
            else "missing_unexplained"
        )
    unexplained_states = sorted(
        task_id for task_id, state in states.items() if state == "missing_unexplained"
    )
    if unexplained_states:
        blockers.add(
            "unexplained_task_states",
            "expected task IDs lack exactly one disclosed lifecycle state",
            provider=provider,
        )
    if missing_ids:
        blockers.add(
            "incomplete_pilot_state",
            "formal pilot still has expected task IDs without immutable records",
            provider=provider,
        )

    provider_lifecycle = str(manifest.get("provider_lifecycle") or "missing")
    allowed_complete_lifecycles = {"complete", "complete_with_exclusions"}
    if provider_lifecycle not in allowed_complete_lifecycles:
        blockers.add(
            "provider_lifecycle_not_complete",
            f"provider lifecycle is {provider_lifecycle}",
            provider=provider,
        )
    lifecycle_history = _validate_lifecycle_history(
        pilot_root,
        provider=provider,
        manifest=manifest,
        phase_binding=phase_binding,
        blockers=blockers,
    )

    status_counts = Counter(value["status"] for value in metrics.values())
    if (
        manifest.get("record_count") != len(records)
        or manifest.get("complete_records") != status_counts.get("complete", 0)
        or manifest.get("status_counts") != dict(sorted(status_counts.items()))
        or manifest.get("returned_models")
        != sorted(
            {
                str(record["model_id"])
                for record in records.values()
                if record.get("model_id")
            }
        )
    ):
        blockers.add(
            "provider_manifest_reconciliation",
            "provider manifest record/status/model totals are stale",
            provider=provider,
        )

    condition_cells: dict[str, Any] = {}
    for condition in ("controlled", "blind"):
        condition_ids = [
            task.task_id for task in tasks if task.condition == condition
        ]
        complete = sum(
            metrics.get(task_id, {}).get("status") == "complete"
            for task_id in condition_ids
        )
        invalid = sum(
            metrics.get(task_id, {}).get("status") == "excluded_schema"
            for task_id in condition_ids
        )
        expected = len(condition_ids)
        complete_fraction = complete / expected
        invalid_fraction = invalid / expected
        passes = (
            complete_fraction >= MIN_COMPLETE_FRACTION
            and invalid_fraction <= MAX_INVALID_SCHEMA_FRACTION
        )
        condition_cells[condition] = {
            "expected_count": expected,
            "complete_count": complete,
            "invalid_schema_count": invalid,
            "complete_fraction": complete_fraction,
            "invalid_schema_fraction": invalid_fraction,
            "minimum_complete_fraction": MIN_COMPLETE_FRACTION,
            "maximum_invalid_schema_fraction": MAX_INVALID_SCHEMA_FRACTION,
            "passes": passes,
        }
        if not passes:
            blockers.add(
                "condition_cell_threshold",
                f"{condition} complete/invalid fractions are outside 80%/20%",
                provider=provider,
            )

    provider_totals = {
        "attempt_count": sum(value["attempt_count"] for value in metrics.values()),
        "unknown_attempt_count": sum(
            value["unknown_attempt_count"] for value in metrics.values()
        ),
        "input_tokens": sum(value["input_tokens"] for value in metrics.values()),
        "output_tokens": sum(value["output_tokens"] for value in metrics.values()),
        "known_cost_yuan": round(
            sum(value["known_cost_yuan"] for value in metrics.values()), 8
        ),
        "unknown_cost_reserve_yuan": round(
            sum(value["unknown_cost_reserve_yuan"] for value in metrics.values()),
            8,
        ),
        "accounted_cost_yuan": round(
            sum(value["accounted_cost_yuan"] for value in metrics.values()), 8
        ),
    }
    budget = manifest.get("budget")
    needs_user = manifest.get("needs_user")
    needs_user_ids = [
        task_id
        for task_id in expected_ids
        if metrics.get(task_id, {}).get("unknown_attempt_count", 0) > 0
    ]
    expected_needs_user = {
        "required": bool(needs_user_ids),
        "reason": (
            "unknown_provider_billing_reserved" if needs_user_ids else None
        ),
        "record_count": len(needs_user_ids),
        "record_task_ids": needs_user_ids,
        "unknown_cost_attempt_count": provider_totals["unknown_attempt_count"],
    }
    try:
        budget_matches = isinstance(budget, Mapping) and (
            math.isclose(
                float(budget.get("provider_record_known_cost_yuan", -1)),
                provider_totals["known_cost_yuan"],
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(budget.get("provider_record_unknown_reserve_yuan", -1)),
                provider_totals["unknown_cost_reserve_yuan"],
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(budget.get("provider_record_accounted_cost_yuan", -1)),
                provider_totals["accounted_cost_yuan"],
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and isinstance(needs_user, Mapping)
            and needs_user.get("unknown_cost_attempt_count")
            == provider_totals["unknown_attempt_count"]
        )
    except (TypeError, ValueError):
        budget_matches = False
    if not budget_matches:
        blockers.add(
            "budget_reconciliation",
            "provider manifest budget/reserve totals differ from attempts",
            provider=provider,
        )
    if needs_user != expected_needs_user:
        blockers.add(
            "needs_user_manifest_reconciliation",
            "provider needs_user disclosure differs from immutable attempt reserves",
            provider=provider,
        )
    state_counts = dict(sorted(Counter(states.values()).items()))
    provider_totals["reported_cumulative_total_cost_yuan"] = (
        float(budget.get("total_cost_yuan"))
        if isinstance(budget, Mapping)
        and isinstance(budget.get("total_cost_yuan"), (int, float))
        else None
    )
    provider_totals["reported_soft_warning_triggered"] = (
        budget.get("soft_warning_triggered") if isinstance(budget, Mapping) else None
    )
    provider_totals["reported_hard_fuse_triggered"] = (
        budget.get("hard_fuse_triggered") if isinstance(budget, Mapping) else None
    )
    return {
        "expected_task_state_count": len(states),
        "expected_task_ids_sha256": canonical_sha256(expected_ids),
        "observed_record_count": len(records),
        "state_counts": state_counts,
        "unexplained_task_ids": unexplained,
        "unexplained_task_state_ids": unexplained_states,
        "condition_cells": condition_cells,
        "model": model,
        "provider_lifecycle": provider_lifecycle,
        "resume_evidence": {
            "resumed_record_count": manifest.get("resumed_records"),
            "lifecycle_event_count": len(lifecycle_history),
        },
        "disclosures": {
            "interruption": manifest.get("interruption"),
            "unavailable": manifest.get("unavailable"),
            "needs_user": manifest.get("needs_user"),
            "fuse_skipped_task_ids": lifecycle_lists["fuse_skipped_task_ids"],
            "breaker_skipped_task_ids": lifecycle_lists[
                "breaker_skipped_task_ids"
            ],
            "interrupted_task_ids": lifecycle_lists["interrupted_task_ids"],
        },
        "accounting": provider_totals,
        "record_set_sha256": canonical_sha256(
            [
                {
                    "task_id": path.stem,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "attempt_count": len(
                        records.get(path.stem, {}).get("attempts", ())
                    ),
                }
                for path in record_paths
            ]
        ),
    }


def _gate(
    passed: bool,
    *,
    evidence: Mapping[str, Any] | Sequence[Any] | str,
) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence": evidence}


def audit_formal_pilot(
    *,
    repo_root: Path | str,
    pilot_root: Path | str,
    resume_before: Mapping[str, Any] | None = None,
    resume_after: Mapping[str, Any] | None = None,
    resume_receipt: Mapping[str, Any] | None = None,
    resume_receipt_path: Path | str | None = None,
    anchor_a_path: Path | str | None = None,
    anchor_b_path: Path | str | None = None,
) -> dict[str, Any]:
    """Audit one formal pilot root without reading a main or provider endpoint."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    pilot = Path(pilot_root).expanduser().resolve(strict=True)
    transition_hint = (
        Path(resume_receipt_path).expanduser().resolve(strict=False)
        if resume_receipt_path is not None
        else None
    )
    sibling_a = (
        transition_hint.with_name("pilot-phase-anchor-a.json")
        if transition_hint is not None
        else None
    )
    sibling_b = (
        transition_hint.with_name("pilot-phase-anchor-b.json")
        if transition_hint is not None
        else None
    )
    a_path = Path(
        anchor_a_path
        or (sibling_a if sibling_a is not None and sibling_a.exists() else repo / DEFAULT_ANCHOR_A)
    ).expanduser().resolve(strict=False)
    b_path = Path(
        anchor_b_path
        or (sibling_b if sibling_b is not None and sibling_b.exists() else repo / DEFAULT_ANCHOR_B)
    ).expanduser().resolve(strict=False)
    blockers = _Blockers()
    contract = load_runtime_contract(repo)
    runtime_manifest = contract.runtime_manifest
    if not isinstance(runtime_manifest, Mapping):
        raise PilotAuditError("committed runtime task manifest is missing")
    runtime_proof = verify_runtime_task_manifest(
        contract, runtime_manifest, verify_git=True
    )
    carried_forward_cost = _reviewed_carried_forward_cost(repo)
    auditor_proof = _audit_implementation_proof(repo)
    if not auditor_proof["passed"]:
        blockers.add(
            "auditor_git_provenance",
            "running auditor bytes are not committed unchanged at the active Git HEAD",
        )
    tasks = enumerate_tasks(contract, phase="pilot")
    expected_ids = [task.task_id for task in tasks]
    runtime_pilot = runtime_manifest.get("phases", {}).get("pilot")
    runtime_roster_pass = (
        isinstance(runtime_pilot, Mapping)
        and runtime_pilot.get("task_count") == EXPECTED_TASKS_PER_PROVIDER
        and list(runtime_pilot.get("task_ids") or ()) == expected_ids
        and len(tasks) == EXPECTED_TASKS_PER_PROVIDER
        and len(set(expected_ids)) == EXPECTED_TASKS_PER_PROVIDER
    )
    if not runtime_roster_pass:
        blockers.add(
            "runtime_roster_drift",
            "committed pilot runtime roster is not exactly 128 unique task IDs",
        )

    providers = tuple(str(value) for value in contract.config["pilot"]["providers"])
    provider_contract_pass = providers == EXPECTED_PILOT_PROVIDERS and (
        isinstance(runtime_pilot, Mapping)
        and tuple(runtime_pilot.get("providers") or ()) == EXPECTED_PILOT_PROVIDERS
    )
    if not provider_contract_pass:
        blockers.add(
            "provider_roster_drift",
            "formal pilot providers differ from frozen deepseek/doubao roster",
        )

    expected_pilot_entries = {
        "evidence",
        "phase_provenance.json",
        "provider_lifecycle",
        "provider_manifests",
        "records",
    }
    pilot_entries = {path.name for path in pilot.iterdir()}
    isolation_pass = (
        pilot.name == "pilot"
        and pilot.parent.name == RUN_ID
        and contract.config["pilot"].get("physical_phase") == "pilot"
        and contract.config["pilot"].get("excluded_from_main_analysis") is True
        and pilot_entries == expected_pilot_entries
        and not (pilot / "main").exists()
        and not (pilot.parent / "main").exists()
    )
    if not isolation_pass:
        blockers.add(
            "pilot_main_isolation",
            "pilot root or frozen exclusion contract does not prove physical isolation",
        )

    phase_path = pilot / "phase_provenance.json"
    try:
        phase = _read_json(phase_path)
    except PilotAuditError as exc:
        phase = {}
        blockers.add("phase_provenance", str(exc))
    formal_scope_pass = (
        phase.get("collection_mode") == "formal"
        and phase.get("development_only") is False
        and phase.get("partial") is False
        and phase.get("formal_analysis_eligible") is True
        and phase.get("phase") == "pilot"
        and phase.get("analysis_population") == "pilot"
        and phase.get("modality_condition") == "text_only"
        and list(phase.get("frozen_providers") or ()) == list(providers)
        and list(phase.get("selected_providers") or ()) == list(providers)
        and phase.get("task_limit") is None
    )
    if not formal_scope_pass:
        blockers.add(
            "formal_scope",
            "phase provenance is partial, development-only, or not pilot formal",
        )
    provenance_pass = False
    phase_binding: dict[str, Any] = {}
    if phase:
        try:
            validate_formal_phase_provenance(phase)
            verify_phase_provenance_against_contract(
                phase,
                contract=contract,
                runtime_manifest=runtime_manifest,
                runtime_proof=runtime_proof,
                tasks=tasks,
                collection_scope=_scope_from_phase(phase),
                carried_forward_cost=carried_forward_cost,
            )
            phase_binding = _phase_binding(phase)
            provenance_pass = True
        except (KeyError, TypeError, ValueError) as exc:
            blockers.add(
                "phase_provenance",
                f"stored phase provenance failed active-contract verification: {exc}",
            )
    if not phase_binding:
        phase_binding = {"invalid_phase_provenance": True}

    phase_receipt_pass = False
    current_phase_receipt: dict[str, Any] | None = None
    evidence_tree: dict[str, Any] | None = None
    if phase and provenance_pass:
        try:
            current_phase_receipt = build_phase_evidence_receipt(
                pilot,
                phase_provenance=phase,
                tasks=tasks,
            )
            evidence_tree = _validate_evidence_tree(
                pilot,
                providers=providers,
            )
            phase_receipt_pass = True
        except (KeyError, OSError, TypeError, ValueError) as exc:
            blockers.add(
                "phase_evidence_receipt",
                f"Evidence-v2 phase receipt reconstruction failed: {exc}",
            )

    text_proof = _audit_text_and_leakage(contract, tasks)
    if phase.get("modality_condition") != "text_only":
        text_proof["passed"] = False
        text_proof["violations"] = sorted(
            set(text_proof["violations"] + ["phase provenance is not text_only"])
        )
    if not text_proof["passed"]:
        blockers.add(
            "text_only_and_leakage",
            "frozen task rendering failed text-only or blind-leakage proof",
        )

    records_root = pilot / "records"
    manifest_root = pilot / "provider_manifests"
    lifecycle_root = pilot / "provider_lifecycle"
    observed_record_providers = (
        sorted(path.name for path in records_root.iterdir() if path.is_dir())
        if records_root.is_dir()
        else []
    )
    manifest_files = (
        sorted(path.name for path in manifest_root.glob("*.json"))
        if manifest_root.is_dir()
        else []
    )
    observed_lifecycle_providers = (
        sorted(path.name for path in lifecycle_root.iterdir() if path.is_dir())
        if lifecycle_root.is_dir()
        else []
    )
    expected_manifest_files = sorted(f"{provider}.json" for provider in providers)
    disk_provider_pass = (
        records_root.is_dir()
        and all(path.is_dir() for path in records_root.iterdir())
        and observed_record_providers == sorted(providers)
        and manifest_root.is_dir()
        and all(path.is_file() for path in manifest_root.iterdir())
        and sorted(path.name for path in manifest_root.iterdir())
        == expected_manifest_files
        and manifest_files == expected_manifest_files
        and lifecycle_root.is_dir()
        and all(path.is_dir() for path in lifecycle_root.iterdir())
        and observed_lifecycle_providers == sorted(providers)
    )
    if not disk_provider_pass:
        blockers.add(
            "provider_roster_drift",
            "pilot store provider directories/manifests differ from frozen roster",
        )

    provider_reports = {
        provider: _audit_provider(
            pilot,
            provider=provider,
            tasks=tasks,
            contract=contract,
            phase_binding=phase_binding,
            blockers=blockers,
        )
        for provider in providers
    }
    accounting = {
        field: round(
            sum(float(report["accounting"][field]) for report in provider_reports.values()),
            8,
        )
        for field in (
            "attempt_count",
            "unknown_attempt_count",
            "input_tokens",
            "output_tokens",
            "known_cost_yuan",
            "unknown_cost_reserve_yuan",
            "accounted_cost_yuan",
        )
    }
    for field in ("attempt_count", "unknown_attempt_count", "input_tokens", "output_tokens"):
        accounting[field] = int(accounting[field])
    prior = contract.prior_cost_ledger
    carried_total = float(carried_forward_cost["total_accounted_cost_yuan"])
    pre_collection_total = round(
        float(prior["pre_run_total_bound_yuan"]) + carried_total,
        8,
    )
    running_total = pre_collection_total
    final_total = round(
        running_total + float(accounting["accounted_cost_yuan"]),
        8,
    )
    cumulative_provider_totals: dict[str, float] = {}
    for provider in providers:
        provider_accounting = provider_reports[provider]["accounting"]
        running_total = round(
            running_total + float(provider_accounting["accounted_cost_yuan"]),
            8,
        )
        cumulative_provider_totals[provider] = running_total
        reported_total = provider_accounting[
            "reported_cumulative_total_cost_yuan"
        ]
        resume_evidence = provider_reports[provider]["resume_evidence"]
        fully_resumed = (
            resume_evidence["resumed_record_count"] == EXPECTED_TASKS_PER_PROVIDER
            and resume_evidence["lifecycle_event_count"] >= 2
        )
        expected_reported_total = final_total if fully_resumed else running_total
        expected_soft = expected_reported_total >= float(
            contract.config["budget_yuan"]["soft_warning"]
        )
        expected_hard = expected_reported_total >= float(
            contract.config["budget_yuan"]["hard_fuse"]
        )
        if (
            reported_total is None
            or not math.isclose(
                float(reported_total),
                expected_reported_total,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or provider_accounting["reported_soft_warning_triggered"]
            is not expected_soft
            or provider_accounting["reported_hard_fuse_triggered"]
            is not expected_hard
        ):
            blockers.add(
                "budget_reconciliation",
                "provider cumulative ledger/fuse flags differ from immutable attempts",
                provider=provider,
            )
    accounting.update(
        {
            "prior_cost_ledger_sha256": prior["prior_cost_ledger_sha256"],
            "prior_known_cost_yuan": prior["known_cost_yuan"],
            "prior_ambiguity_reserve_yuan": prior[
                "pre_run_ambiguity_reserve_yuan"
            ],
            "prior_total_bound_yuan": prior["pre_run_total_bound_yuan"],
            "carried_forward_cost_ledger_sha256": carried_forward_cost[
                "carried_forward_cost_ledger_sha256"
            ],
            "carried_forward_source_phase_receipt_sha256": carried_forward_cost[
                "source_phase_receipt_sha256"
            ],
            "carried_forward_source_record_set_sha256": carried_forward_cost[
                "source_record_set_sha256"
            ],
            "carried_forward_known_cost_yuan": carried_forward_cost[
                "known_cost_yuan"
            ],
            "carried_forward_unknown_reserve_yuan": carried_forward_cost[
                "unknown_cost_reserve_yuan"
            ],
            "carried_forward_total_accounted_cost_yuan": carried_total,
            "pre_collection_total_yuan": pre_collection_total,
            "total_accounted_cost_yuan": final_total,
            "soft_warning_yuan": contract.config["budget_yuan"]["soft_warning"],
            "hard_fuse_yuan": contract.config["budget_yuan"]["hard_fuse"],
            "cumulative_provider_totals_yuan": cumulative_provider_totals,
        }
    )
    if accounting["total_accounted_cost_yuan"] >= float(
        accounting["hard_fuse_yuan"]
    ):
        blockers.add(
            "budget_fuse",
            "reconciled total accounted cost reaches the frozen CNY 450 fuse",
        )

    billing_resolution_required = int(accounting["unknown_attempt_count"]) > 0
    billing_resolution_pass = not billing_resolution_required
    billing_resolution_proof: dict[str, Any] = {
        "status": "not_required",
        "unknown_attempt_count": int(accounting["unknown_attempt_count"]),
    }
    if billing_resolution_required:
        try:
            billing_resolution_proof = verify_billing_authorization_resolution(
                repo_root=repo,
                pilot_root=pilot,
                anchor_a_path=a_path,
                resolution_path=(
                    repo / DEFAULT_BILLING_AUTHORIZATION_RESOLUTION
                ),
                contract=contract,
                audit_head=str(auditor_proof.get("audit_git_head") or ""),
            )
            billing_resolution_pass = (
                billing_resolution_proof["unknown_attempt_count"]
                == accounting["unknown_attempt_count"]
                and math.isclose(
                    float(billing_resolution_proof["unknown_reserve_yuan"]),
                    float(accounting["unknown_cost_reserve_yuan"]),
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                and math.isclose(
                    float(billing_resolution_proof["total_accounted_cost_yuan"]),
                    float(accounting["total_accounted_cost_yuan"]),
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                and float(billing_resolution_proof["total_accounted_cost_yuan"])
                < float(billing_resolution_proof["hard_fuse_yuan"])
            )
            if not billing_resolution_pass:
                raise PilotAuditError(
                    "billing authorization accounting differs from pilot audit"
                )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            billing_resolution_proof = {
                "status": "missing_or_invalid",
                "unknown_attempt_count": int(accounting["unknown_attempt_count"]),
                "reason": str(exc),
            }
            billing_resolution_pass = False
        if not billing_resolution_pass:
            for provider in EXPECTED_PILOT_PROVIDERS:
                if int(
                    provider_reports[provider]["accounting"][
                        "unknown_attempt_count"
                    ]
                ) > 0:
                    blockers.add(
                        "needs_user_unresolved",
                        "unknown provider billing lacks a valid preauthorized disposition",
                        provider=provider,
                    )

    source_files = _source_file_rows(pilot)
    source_set_sha = canonical_sha256(source_files)
    condition_pass = all(
        cell["passes"]
        for provider_report in provider_reports.values()
        for cell in provider_report["condition_cells"].values()
    )
    exact_states_pass = all(
        report["expected_task_state_count"] == EXPECTED_TASKS_PER_PROVIDER
        and not report["unexplained_task_ids"]
        and not report["unexplained_task_state_ids"]
        for report in provider_reports.values()
    ) and not blockers.has("unexplained_record_ids", "unexplained_task_states")
    lifecycle_pass = not blockers.has(
        "lifecycle_integrity",
        "provider_lifecycle_not_complete",
        "incomplete_pilot_state",
        "provider_manifest_missing",
        "provider_manifest_reconciliation",
    )
    accounting_pass = not blockers.has(
        "budget_reconciliation",
        "attempt_reconciliation",
        "token_reconciliation",
        "budget_fuse",
        "needs_user_manifest_reconciliation",
        "needs_user_unresolved",
    )
    model_pass = not blockers.has("model_drift")
    record_provenance_pass = not blockers.has(
        "record_identity_drift", "record_provenance_drift"
    )
    resume_comparison: dict[str, Any]
    current_snapshot = snapshot_resume_state(pilot)
    expected_complete_records = len(providers) * EXPECTED_TASKS_PER_PROVIDER
    if resume_before is None and resume_after is None:
        resume_comparison = {
            "ok": True,
            "require_zero_calls": True,
            "supplemental_legacy_snapshots": "not_supplied",
            "formal_blocking_reasons": [],
            "after_matches_audited_root": True,
            "expected_complete_record_count": expected_complete_records,
            "full_complete_record_count": current_snapshot.get("record_count")
            == expected_complete_records,
        }
    elif resume_before is None or resume_after is None:
        resume_comparison = {
            "ok": False,
            "require_zero_calls": True,
            "supplemental_legacy_snapshots": "incomplete_pair",
            "formal_blocking_reasons": ["resume_snapshot_pair_incomplete"],
            "after_matches_audited_root": False,
            "expected_complete_record_count": expected_complete_records,
            "full_complete_record_count": False,
        }
        blockers.add(
            "zero_call_resume",
            "legacy resume snapshots must be supplied as a complete before/after pair",
        )
    else:
        resume_comparison = compare_resume_audits(
            resume_before,
            resume_after,
            require_zero_calls=True,
        )
        after_to_current = compare_resume_audits(
            resume_after,
            current_snapshot,
            require_zero_calls=True,
        )
        resume_comparison["after_matches_audited_root"] = after_to_current["ok"]
        resume_comparison["expected_complete_record_count"] = (
            expected_complete_records
        )
        resume_comparison["full_complete_record_count"] = (
            resume_before.get("record_count") == expected_complete_records
            and resume_after.get("record_count") == expected_complete_records
        )
    resume_comparison["provider_resume_evidence"] = {
        provider: report["resume_evidence"]
        for provider, report in provider_reports.items()
    }
    resume_comparison["runner_resume_observed"] = all(
        evidence["resumed_record_count"] == EXPECTED_TASKS_PER_PROVIDER
        and evidence["lifecycle_event_count"] >= 2
        for evidence in resume_comparison["provider_resume_evidence"].values()
    )
    resume_pass = bool(
        resume_comparison["ok"]
        and resume_comparison["after_matches_audited_root"]
        and resume_comparison["full_complete_record_count"]
        and resume_comparison["runner_resume_observed"]
    )
    if not resume_pass and not blockers.has("zero_call_resume"):
        blockers.add(
            "zero_call_resume",
            "post-completion resume added calls/records, changed bytes, or was incomplete",
        )
    resume_receipt_proof: dict[str, Any] | None = None
    if resume_receipt is None and resume_receipt_path is None:
        blockers.add(
            "zero_call_resume_receipt",
            "formal approval requires a Git-anchored zero-call resume receipt",
        )
        resume_pass = False
    else:
        try:
            if resume_receipt is None or resume_receipt_path is None:
                raise PilotAuditError("resume receipt and path must be supplied together")
            resume_receipt_proof = verify_zero_call_resume_receipt(
                resume_receipt,
                repo_root=repo,
                pilot_root=pilot,
                receipt_path=resume_receipt_path,
                anchor_a_path=a_path,
                anchor_b_path=b_path,
                phase_provenance=phase,
                tasks=tasks,
                audit_head=str(auditor_proof.get("audit_git_head") or ""),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            blockers.add(
                "zero_call_resume_receipt",
                f"zero-call resume receipt failed verification: {exc}",
            )
            resume_pass = False
    if resume_receipt_proof is not None:
        resume_comparison["git_anchored_receipt"] = resume_receipt_proof
    gates = {
        "auditor_git_provenance": _gate(
            bool(auditor_proof["passed"]), evidence=auditor_proof
        ),
        "runtime_and_phase_provenance": _gate(
            runtime_roster_pass and provenance_pass and record_provenance_pass,
            evidence={
                "runtime_task_manifest_sha256": runtime_manifest[
                    "runtime_task_manifest_sha256"
                ],
                "runtime_commit": runtime_manifest["runtime_commit"],
                "phase_provenance_sha256": phase.get("phase_provenance_sha256"),
            },
        ),
        "formal_scope": _gate(
            formal_scope_pass and disk_provider_pass and provider_contract_pass,
            evidence={
                "frozen_providers": list(providers),
                "selected_providers": phase.get("selected_providers"),
                "partial": phase.get("partial"),
                "development_only": phase.get("development_only"),
            },
        ),
        "exact_task_states": _gate(
            exact_states_pass,
            evidence={
                provider: report["state_counts"]
                for provider, report in provider_reports.items()
            },
        ),
        "condition_cells": _gate(
            condition_pass,
            evidence={
                provider: report["condition_cells"]
                for provider, report in provider_reports.items()
            },
        ),
        "model_identity": _gate(
            model_pass,
            evidence={
                provider: report["model"]
                for provider, report in provider_reports.items()
            },
        ),
        "text_only_and_leakage": _gate(
            bool(text_proof["passed"]), evidence=text_proof
        ),
        "pilot_main_isolation": _gate(
            isolation_pass,
            evidence={
                "phase_root_name": pilot.name,
                "run_root_name": pilot.parent.name,
                "pilot_top_level_entries": sorted(pilot_entries),
                "excluded_from_main_analysis": contract.config["pilot"].get(
                    "excluded_from_main_analysis"
                ),
            },
        ),
        "phase_evidence_receipts": _gate(
            phase_receipt_pass,
            evidence={
                "current_phase_evidence_receipt_sha256": (
                    current_phase_receipt.get("phase_evidence_receipt_sha256")
                    if current_phase_receipt
                    else None
                ),
                "evidence_tree": evidence_tree,
            },
        ),
        "billing_authorization_resolution": _gate(
            billing_resolution_pass,
            evidence=billing_resolution_proof,
        ),
        "accounting_reconciliation": _gate(
            accounting_pass, evidence=accounting
        ),
        "lifecycle_disclosures": _gate(
            lifecycle_pass,
            evidence={
                provider: report["disclosures"]
                for provider, report in provider_reports.items()
            },
        ),
        "zero_call_resume": _gate(resume_pass, evidence=resume_comparison),
    }
    blocking_rows = blockers.rows()
    report: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.formal_pilot_audit.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "decision": "BLOCK" if blocking_rows else "GO",
        "formal_pilot_approved": not blocking_rows,
        "audit_scope": "formal_pilot_root_plus_committed_runtime_freeze_contract",
        "provider_calls_performed": 0,
        "expected_providers": list(providers),
        "expected_task_count_per_provider": EXPECTED_TASKS_PER_PROVIDER,
        "thresholds": {
            "minimum_condition_complete_fraction": MIN_COMPLETE_FRACTION,
            "maximum_condition_invalid_schema_fraction": (
                MAX_INVALID_SCHEMA_FRACTION
            ),
        },
        "gates": gates,
        "providers": provider_reports,
        "accounting": accounting,
        "blocking_reasons": blocking_rows,
        "source_binding": {
            "auditor_implementation_sha256": auditor_proof[
                "auditor_implementation_sha256"
            ],
            "auditor_committed_blob_sha256": auditor_proof[
                "committed_blob_sha256"
            ],
            "audit_git_head": auditor_proof["audit_git_head"],
            "auditor_commit": auditor_proof["auditor_commit"],
            "pilot_source_file_count": len(source_files),
            "pilot_source_set_sha256": source_set_sha,
            "pilot_source_files": source_files,
            "runtime_task_manifest_sha256": runtime_manifest[
                "runtime_task_manifest_sha256"
            ],
            "freeze_manifest_sha256": contract.freeze_manifest[
                "freeze_manifest_sha256"
            ],
            "phase_provenance_file_sha256": (
                sha256_file(phase_path) if phase_path.is_file() else None
            ),
            "current_phase_evidence_receipt_sha256": (
                current_phase_receipt.get("phase_evidence_receipt_sha256")
                if current_phase_receipt
                else None
            ),
            "anchor_a_file_sha256": (
                sha256_file(a_path) if a_path.is_file() else None
            ),
            "anchor_b_file_sha256": (
                sha256_file(b_path) if b_path.is_file() else None
            ),
            "transition_receipt_file_sha256": (
                sha256_file(Path(resume_receipt_path).expanduser().resolve(strict=False))
                if resume_receipt_path is not None
                and Path(resume_receipt_path).expanduser().resolve(strict=False).is_file()
                else None
            ),
            "billing_authorization_resolution_file_sha256": (
                billing_resolution_proof.get("file_sha256")
                if billing_resolution_proof.get("status") == "applied"
                else None
            ),
            "billing_authorization_resolution_sha256": (
                billing_resolution_proof.get(
                    "billing_authorization_resolution_sha256"
                )
                if billing_resolution_proof.get("status") == "applied"
                else None
            ),
        },
    }
    report["pilot_audit_sha256"] = canonical_sha256(report)
    return report


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_canonical_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == value
        and value not in {".", ".."}
    )


def _unknown_billing_failure_is_exact(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "error_category",
            "provider_response_received",
            "cost_known",
            "billing_ambiguity",
            "cost_yuan",
            "cost_reserve_yuan",
        }
        and value.get("error_category") == "network_timeout"
        and value.get("provider_response_received") is False
        and value.get("cost_known") is False
        and value.get("billing_ambiguity") is True
        and value.get("cost_yuan") is None
        and type(value.get("cost_reserve_yuan")) in {int, float}
        and float(value.get("cost_reserve_yuan")) == 10.0
    )


def _validate_unknown_billing_attempt_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise PilotAuditError("billing authorization lacks reserved attempts")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int]] = set()
    for value in rows:
        if not isinstance(value, Mapping) or set(value) != {
            "provider",
            "task_id",
            "attempt",
            "record",
            "request",
            "failure",
            "provider_call_event",
        }:
            raise PilotAuditError("billing authorization attempt row is invalid")
        provider = value.get("provider")
        task_id = value.get("task_id")
        attempt = value.get("attempt")
        record = value.get("record")
        request = value.get("request")
        failure = value.get("failure")
        event = value.get("provider_call_event")
        if (
            provider not in EXPECTED_PILOT_PROVIDERS
            or not _is_lower_hex(task_id, 64)
            or type(attempt) is not int
            or attempt < 1
            or not isinstance(record, Mapping)
            or set(record) != {"path", "bytes", "sha256", "attempt_sha256"}
            or record.get("path") != f"records/{provider}/{task_id}.json"
            or type(record.get("bytes")) is not int
            or int(record.get("bytes", 0)) <= 0
            or not _is_lower_hex(record.get("sha256"), 64)
            or not _is_lower_hex(record.get("attempt_sha256"), 64)
            or not isinstance(request, Mapping)
            or set(request) != {"model", "max_tokens", "wire_message_sha256"}
            or not isinstance(request.get("model"), str)
            or not request.get("model")
            or type(request.get("max_tokens")) is not int
            or int(request.get("max_tokens", 0)) <= 0
            or not _is_lower_hex(request.get("wire_message_sha256"), 64)
            or not _unknown_billing_failure_is_exact(failure)
            or not isinstance(event, Mapping)
            or set(event)
            != {
                "path",
                "bytes",
                "file_sha256",
                "event_sha256",
                "event_index",
                "invocation_id",
            }
            or not _is_canonical_relative_path(event.get("path"))
            or not str(event.get("path")).startswith(
                f"evidence/provider_events/{provider}/"
            )
            or type(event.get("bytes")) is not int
            or int(event.get("bytes", 0)) <= 0
            or not _is_lower_hex(event.get("file_sha256"), 64)
            or not _is_lower_hex(event.get("event_sha256"), 64)
            or type(event.get("event_index")) is not int
            or int(event.get("event_index", -1)) < 0
            or not isinstance(event.get("invocation_id"), str)
            or not event.get("invocation_id")
        ):
            raise PilotAuditError("billing authorization attempt evidence is invalid")
        identity = (str(provider), str(task_id), int(attempt))
        if identity in identities:
            raise PilotAuditError("billing authorization repeats an attempt")
        identities.add(identity)
        normalized.append(json.loads(json.dumps(value, ensure_ascii=False)))
    ordered = sorted(
        normalized,
        key=lambda row: (str(row["provider"]), str(row["task_id"]), int(row["attempt"])),
    )
    if normalized != ordered:
        raise PilotAuditError("billing authorization attempt rows are not sorted")
    return normalized


def _billing_reviewer_and_date_are_valid(reviewer: Any, review_date: Any) -> bool:
    if (
        not isinstance(reviewer, str)
        or _BILLING_REVIEWER_PATTERN.fullmatch(reviewer) is None
        or not isinstance(review_date, str)
    ):
        return False
    try:
        parsed = date.fromisoformat(review_date)
    except ValueError:
        return False
    return parsed.isoformat() == review_date


def _billing_safeguards_are_exact(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "original_needs_user_disclosure_retained",
            "billing_ambiguity_retained",
            "reserve_remains_accounted",
            "scientific_outcomes_unmodified",
            "other_audit_blockers_waived",
        }
        and value.get("original_needs_user_disclosure_retained") is True
        and value.get("billing_ambiguity_retained") is True
        and value.get("reserve_remains_accounted") is True
        and value.get("scientific_outcomes_unmodified") is True
        and value.get("other_audit_blockers_waived") is False
    )


def build_billing_authorization_resolution_payload(
    *,
    unknown_attempts: Sequence[Mapping[str, Any]],
    anchor_a: Mapping[str, Any],
    accounting: Mapping[str, Any],
    authorization: Mapping[str, Any],
    runtime_task_manifest_sha256: str,
    freeze_manifest_sha256: str,
    reviewer: str,
    review_date: str,
) -> dict[str, Any]:
    if not _billing_reviewer_and_date_are_valid(reviewer, review_date):
        raise PilotAuditError("billing authorization reviewer is invalid")
    try:
        total = float(accounting["total_accounted_cost_yuan"])
        hard_fuse = float(accounting["hard_fuse_yuan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotAuditError("billing authorization accounting is invalid") from exc
    if not math.isfinite(total) or not math.isfinite(hard_fuse) or total >= hard_fuse:
        raise PilotAuditError("billing authorization cannot override the hard fuse")
    attempts = sorted(
        (json.loads(json.dumps(row, ensure_ascii=False)) for row in unknown_attempts),
        key=lambda row: (str(row["provider"]), str(row["task_id"]), int(row["attempt"])),
    )
    if not attempts:
        raise PilotAuditError("billing authorization lacks reserved attempts")
    receipt: dict[str, Any] = {
        "schema_version": (
            "yher.llm_sim_v2.billing_authorization_resolution.v1"
        ),
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "authority": "post_observation_budget_governance",
        "billing_fact_status": "unresolved_reserved",
        "action_disposition": (
            "continue_under_preexisting_budget_authorization"
        ),
        "unknown_attempts": attempts,
        "unknown_attempt_set_sha256": canonical_sha256(attempts),
        "anchor_a": json.loads(json.dumps(anchor_a, ensure_ascii=False)),
        "accounting": json.loads(json.dumps(accounting, ensure_ascii=False)),
        "authorization": json.loads(json.dumps(authorization, ensure_ascii=False)),
        "runtime_task_manifest_sha256": runtime_task_manifest_sha256,
        "freeze_manifest_sha256": freeze_manifest_sha256,
        "safeguards": {
            "original_needs_user_disclosure_retained": True,
            "billing_ambiguity_retained": True,
            "reserve_remains_accounted": True,
            "scientific_outcomes_unmodified": True,
            "other_audit_blockers_waived": False,
        },
        "reviewer": reviewer,
        "review_date": review_date,
    }
    receipt["billing_authorization_resolution_sha256"] = canonical_sha256(receipt)
    validate_billing_authorization_resolution_payload(
        receipt,
        expected_unknown_attempts=attempts,
        expected_anchor_a=anchor_a,
        expected_accounting=accounting,
        expected_authorization=authorization,
        expected_runtime_task_manifest_sha256=runtime_task_manifest_sha256,
        expected_freeze_manifest_sha256=freeze_manifest_sha256,
    )
    return receipt


def validate_billing_authorization_resolution_payload(
    receipt: Mapping[str, Any],
    *,
    expected_unknown_attempts: Sequence[Mapping[str, Any]],
    expected_anchor_a: Mapping[str, Any],
    expected_accounting: Mapping[str, Any],
    expected_authorization: Mapping[str, Any],
    expected_runtime_task_manifest_sha256: str,
    expected_freeze_manifest_sha256: str,
) -> str:
    payload = dict(receipt)
    advertised = payload.pop("billing_authorization_resolution_sha256", None)
    if (
        set(payload)
        != {
            "schema_version",
            "simulated",
            "run_id",
            "phase",
            "authority",
            "billing_fact_status",
            "action_disposition",
            "unknown_attempts",
            "unknown_attempt_set_sha256",
            "anchor_a",
            "accounting",
            "authorization",
            "runtime_task_manifest_sha256",
            "freeze_manifest_sha256",
            "safeguards",
            "reviewer",
            "review_date",
        }
        or receipt.get("schema_version")
        != "yher.llm_sim_v2.billing_authorization_resolution.v1"
        or receipt.get("simulated") is not True
        or receipt.get("run_id") != RUN_ID
        or receipt.get("phase") != "pilot"
        or receipt.get("authority") != "post_observation_budget_governance"
        or receipt.get("billing_fact_status") != "unresolved_reserved"
        or receipt.get("action_disposition")
        != "continue_under_preexisting_budget_authorization"
        or advertised != canonical_sha256(payload)
    ):
        raise PilotAuditError("billing authorization envelope is invalid")
    attempts = _validate_unknown_billing_attempt_rows(receipt.get("unknown_attempts"))
    expected_attempts = _validate_unknown_billing_attempt_rows(
        list(expected_unknown_attempts)
    )
    if (
        attempts != expected_attempts
        or receipt.get("unknown_attempt_set_sha256") != canonical_sha256(attempts)
        or receipt.get("anchor_a") != dict(expected_anchor_a)
        or receipt.get("accounting") != dict(expected_accounting)
        or receipt.get("authorization") != dict(expected_authorization)
        or receipt.get("runtime_task_manifest_sha256")
        != expected_runtime_task_manifest_sha256
        or receipt.get("freeze_manifest_sha256") != expected_freeze_manifest_sha256
        or not _billing_safeguards_are_exact(receipt.get("safeguards"))
        or not _billing_reviewer_and_date_are_valid(
            receipt.get("reviewer"), receipt.get("review_date")
        )
    ):
        raise PilotAuditError("billing authorization differs from source evidence")
    accounting = receipt["accounting"]
    try:
        count = len(attempts)
        phase_known = float(accounting["phase_known_cost_yuan"])
        phase_reserve = float(accounting["phase_unknown_reserve_yuan"])
        phase_total = float(accounting["phase_accounted_cost_yuan"])
        pre_total = float(accounting["pre_collection_total_yuan"])
        total_known = float(accounting["total_known_cost_yuan"])
        total_reserve = float(accounting["total_unknown_reserve_yuan"])
        total = float(accounting["total_accounted_cost_yuan"])
        hard_fuse = float(accounting["hard_fuse_yuan"])
        headroom = float(accounting["remaining_headroom_yuan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotAuditError("billing authorization accounting is invalid") from exc
    if not (
        set(accounting)
        == {
            "pre_collection_total_yuan",
            "phase_known_cost_yuan",
            "phase_unknown_reserve_yuan",
            "phase_accounted_cost_yuan",
            "total_known_cost_yuan",
            "total_unknown_reserve_yuan",
            "total_accounted_cost_yuan",
            "hard_fuse_yuan",
            "remaining_headroom_yuan",
        }
        and math.isclose(phase_reserve, count * 10.0, rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(phase_total, phase_known + phase_reserve, rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(total, pre_total + phase_total, rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(total, total_known + total_reserve, rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(headroom, hard_fuse - total, rel_tol=0.0, abs_tol=1e-8)
        and total < hard_fuse
    ):
        raise PilotAuditError("billing authorization reaches or misstates the hard fuse")
    return str(advertised)


def _workspace_dispatch_brief_path(repo_root: Path) -> Path:
    for ancestor in (repo_root, *repo_root.parents):
        candidate = ancestor / DISPATCH_BRIEF_RELATIVE
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise PilotAuditError("billing authorization dispatch brief is missing")


def _billing_authorization_sources(
    repo_root: Path, *, audit_head: str
) -> dict[str, Any]:
    plan_path = repo_root / ANALYSIS_PLAN_RELATIVE
    plan_proof = _git_committed_file_proof(repo_root, plan_path, head=audit_head)
    brief_path = _workspace_dispatch_brief_path(repo_root)
    plan_text = plan_path.read_text(encoding="utf-8")
    brief_text = brief_path.read_text(encoding="utf-8")
    if not (
        plan_proof.get("passed") is True
        and plan_proof.get("path") == ANALYSIS_PLAN_RELATIVE.as_posix()
        and plan_proof.get("file_sha256")
        == plan_proof.get("committed_blob_sha256")
        and "CNY 450 is a hard fuse" in plan_text
        and "enters `needs_user`" in plan_text
        and sha256_file(brief_path) == REVIEWED_DISPATCH_BRIEF_SHA256
        and "累计超 **¥450** 时熔断进 needs_user 等用户确认" in brief_text
        and "Codex 可自审自签" in brief_text
    ):
        raise PilotAuditError("billing authorization policy sources are invalid")
    return {
        "analysis_plan": {
            "path": ANALYSIS_PLAN_RELATIVE.as_posix(),
            "sha256": plan_proof["file_sha256"],
            "committed_blob_sha256": plan_proof["committed_blob_sha256"],
            "anchor_commit": plan_proof["anchor_commit"],
        },
        "dispatch_brief": {
            "path": DISPATCH_BRIEF_RELATIVE.as_posix(),
            "sha256": REVIEWED_DISPATCH_BRIEF_SHA256,
        },
        "hard_fuse_policy": HARD_FUSE_POLICY,
        "self_review_policy": SELF_REVIEW_POLICY,
    }


def _billing_anchor_a_binding(
    anchor_a: Mapping[str, Any], anchor_a_path: Path
) -> dict[str, Any]:
    digest = _verify_phase_receipt_envelope(anchor_a)
    providers = anchor_a.get("providers")
    if not isinstance(providers, Mapping) or set(providers) != set(
        EXPECTED_PILOT_PROVIDERS
    ):
        raise PilotAuditError("billing authorization anchor A provider set is invalid")
    provider_bindings: dict[str, Any] = {}
    for provider in EXPECTED_PILOT_PROVIDERS:
        row = providers[provider]
        if not isinstance(row, Mapping):
            raise PilotAuditError("billing authorization anchor A provider is invalid")
        provider_bindings[provider] = {
            field: row.get(field)
            for field in (
                "provider_manifest_sha256",
                "record_set_sha256",
                "evidence_chain_head_sha256",
            )
        }
        if any(
            not _is_lower_hex(value, 64)
            for value in provider_bindings[provider].values()
        ):
            raise PilotAuditError("billing authorization anchor A hashes are invalid")
    store_snapshot = anchor_a.get("store_snapshot")
    if not isinstance(store_snapshot, Mapping):
        raise PilotAuditError("billing authorization anchor A store is invalid")
    return {
        "phase_evidence_receipt_sha256": digest,
        "file_sha256": sha256_file(anchor_a_path),
        "file_bytes": anchor_a_path.stat().st_size,
        "phase_provenance_sha256": anchor_a.get("phase_provenance_sha256"),
        "phase_provenance_file_sha256": anchor_a.get(
            "phase_provenance_file_sha256"
        ),
        "store_snapshot": dict(store_snapshot),
        "providers": provider_bindings,
    }


def _billing_unknown_attempt_rows(pilot_root: Path) -> list[dict[str, Any]]:
    event_by_identity: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for provider in EXPECTED_PILOT_PROVIDERS:
        event_root = pilot_root / "evidence/provider_events" / provider
        for path in sorted(event_root.glob("*.json")):
            event = _read_json(path)
            if event.get("event_type") != "provider_call_started":
                continue
            payload = dict(event)
            advertised = payload.pop("event_sha256", None)
            try:
                identity = (
                    provider,
                    str(event["task_id"]),
                    int(event["attempt"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PilotAuditError("billing provider-call identity is invalid") from exc
            if (
                event.get("schema_version")
                != "yher.llm_sim_v2.provider_evidence_event.v1"
                or event.get("simulated") is not True
                or event.get("run_id") != RUN_ID
                or event.get("phase") != "pilot"
                or event.get("provider") != provider
                or not _is_lower_hex(advertised, 64)
                or advertised != canonical_sha256(payload)
                or identity in event_by_identity
            ):
                raise PilotAuditError("billing provider-call event is invalid")
            event_by_identity[identity] = (path, event)

    rows: list[dict[str, Any]] = []
    for provider in EXPECTED_PILOT_PROVIDERS:
        records_root = pilot_root / "records" / provider
        for record_path in sorted(records_root.glob("*.json")):
            record = _read_json(record_path)
            attempts = record.get("attempts")
            if not isinstance(attempts, list):
                raise PilotAuditError("billing source record lacks attempts")
            unknown = [
                attempt
                for attempt in attempts
                if isinstance(attempt, Mapping)
                and (
                    attempt.get("billing_ambiguity") is True
                    or attempt.get("cost_known") is False
                    or float(attempt.get("cost_reserve_yuan") or 0.0) > 0.0
                )
            ]
            if not unknown:
                continue
            if (
                record.get("provider") != provider
                or record.get("task_id") != record_path.stem
                or record.get("needs_user") is not True
                or record.get("needs_user_reasons")
                != ["unknown_provider_billing_reserved"]
                or not math.isclose(
                    float(record.get("unknown_cost_reserve_yuan") or 0.0),
                    sum(float(attempt.get("cost_reserve_yuan") or 0.0) for attempt in unknown),
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ):
                raise PilotAuditError("billing source record disclosure is invalid")
            for attempt_value in unknown:
                attempt = dict(attempt_value)
                try:
                    attempt_number = int(attempt["attempt"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise PilotAuditError("billing source attempt is invalid") from exc
                identity = (provider, record_path.stem, attempt_number)
                event_pair = event_by_identity.get(identity)
                if event_pair is None:
                    raise PilotAuditError("billing source attempt lacks provider-call event")
                event_path, event = event_pair
                failure = {
                    "error_category": attempt.get("error_category"),
                    "provider_response_received": attempt.get(
                        "provider_response_received"
                    ),
                    "cost_known": attempt.get("cost_known"),
                    "billing_ambiguity": attempt.get("billing_ambiguity"),
                    "cost_yuan": attempt.get("cost_yuan"),
                    "cost_reserve_yuan": attempt.get("cost_reserve_yuan"),
                }
                request = {
                    "model": record.get("requested_model"),
                    "max_tokens": attempt.get("request_max_tokens"),
                    "wire_message_sha256": record.get("wire_message_sha256"),
                }
                if (
                    not _unknown_billing_failure_is_exact(failure)
                    or event.get("model") != request["model"]
                    or event.get("request_max_tokens") != request["max_tokens"]
                    or event.get("wire_message_sha256")
                    != request["wire_message_sha256"]
                ):
                    raise PilotAuditError("billing attempt and event differ")
                rows.append(
                    {
                        "provider": provider,
                        "task_id": record_path.stem,
                        "attempt": attempt_number,
                        "record": {
                            "path": _safe_relative(pilot_root, record_path),
                            "bytes": record_path.stat().st_size,
                            "sha256": sha256_file(record_path),
                            "attempt_sha256": canonical_sha256(attempt),
                        },
                        "request": request,
                        "failure": failure,
                        "provider_call_event": {
                            "path": _safe_relative(pilot_root, event_path),
                            "bytes": event_path.stat().st_size,
                            "file_sha256": sha256_file(event_path),
                            "event_sha256": event["event_sha256"],
                            "event_index": event["event_index"],
                            "invocation_id": event["invocation_id"],
                        },
                    }
                )
    rows.sort(
        key=lambda row: (str(row["provider"]), str(row["task_id"]), int(row["attempt"]))
    )
    return _validate_unknown_billing_attempt_rows(rows)


def _billing_accounting(
    pilot_root: Path,
    *,
    prior_cost_ledger: Mapping[str, Any],
    carried_forward_cost: Mapping[str, Any],
) -> dict[str, float]:
    phase_known = 0.0
    phase_reserve = 0.0
    for provider in EXPECTED_PILOT_PROVIDERS:
        for path in sorted((pilot_root / "records" / provider).glob("*.json")):
            record = _read_json(path)
            phase_known += float(record.get("known_cost_yuan") or 0.0)
            phase_reserve += float(record.get("unknown_cost_reserve_yuan") or 0.0)
    phase_known = round(phase_known, 8)
    phase_reserve = round(phase_reserve, 8)
    phase_total = round(phase_known + phase_reserve, 8)
    pre_total = round(
        float(prior_cost_ledger["pre_run_total_bound_yuan"])
        + float(carried_forward_cost["total_accounted_cost_yuan"]),
        8,
    )
    total_known = round(
        float(prior_cost_ledger["known_cost_yuan"])
        + float(carried_forward_cost["known_cost_yuan"])
        + phase_known,
        8,
    )
    total_reserve = round(
        float(prior_cost_ledger["pre_run_ambiguity_reserve_yuan"])
        + float(carried_forward_cost["unknown_cost_reserve_yuan"])
        + phase_reserve,
        8,
    )
    total = round(pre_total + phase_total, 8)
    hard_fuse = 450.0
    return {
        "pre_collection_total_yuan": pre_total,
        "phase_known_cost_yuan": phase_known,
        "phase_unknown_reserve_yuan": phase_reserve,
        "phase_accounted_cost_yuan": phase_total,
        "total_known_cost_yuan": total_known,
        "total_unknown_reserve_yuan": total_reserve,
        "total_accounted_cost_yuan": total,
        "hard_fuse_yuan": hard_fuse,
        "remaining_headroom_yuan": round(hard_fuse - total, 8),
    }


def prepare_billing_authorization_resolution(
    *,
    repo_root: Path | str,
    pilot_root: Path | str,
    anchor_a_path: Path | str,
    output_path: Path | str = DEFAULT_BILLING_AUTHORIZATION_RESOLUTION,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve(strict=True)
    pilot = Path(pilot_root).expanduser().resolve(strict=True)
    a_path = Path(anchor_a_path).expanduser().resolve(strict=True)
    output = Path(output_path)
    if not output.is_absolute():
        output = repo / output
    output = output.expanduser().resolve(strict=False)
    if output != (repo / DEFAULT_BILLING_AUTHORIZATION_RESOLUTION).resolve(
        strict=False
    ):
        raise PilotAuditError("billing authorization path is not the fixed path")
    contract = load_runtime_contract(repo)
    phase = _read_json(pilot / "phase_provenance.json")
    tasks = enumerate_tasks(contract, phase="pilot")
    anchor_a = _read_json(a_path)
    rebuilt_a = build_phase_evidence_receipt(
        pilot, phase_provenance=phase, tasks=tasks
    )
    if anchor_a != rebuilt_a:
        raise PilotAuditError("billing authorization anchor A is stale")
    evidence = _validate_evidence_tree(
        pilot,
        providers=EXPECTED_PILOT_PROVIDERS,
        required_receipts=(anchor_a,),
    )
    if set(evidence["phase_receipts"]) != {
        str(anchor_a["phase_evidence_receipt_sha256"])
    }:
        raise PilotAuditError("billing authorization requires the sole internal A")
    head = _git_head(repo)
    a_git = _git_committed_file_proof(repo, a_path, head=head)
    if a_git.get("passed") is not True:
        raise PilotAuditError("billing authorization requires committed anchor A")
    carried = _reviewed_carried_forward_cost(repo)
    receipt = build_billing_authorization_resolution_payload(
        unknown_attempts=_billing_unknown_attempt_rows(pilot),
        anchor_a=_billing_anchor_a_binding(anchor_a, a_path),
        accounting=_billing_accounting(
            pilot,
            prior_cost_ledger=contract.prior_cost_ledger,
            carried_forward_cost=carried,
        ),
        authorization=_billing_authorization_sources(repo, audit_head=head),
        runtime_task_manifest_sha256=str(
            contract.runtime_manifest["runtime_task_manifest_sha256"]
        ),
        freeze_manifest_sha256=str(
            contract.freeze_manifest["freeze_manifest_sha256"]
        ),
        reviewer="codex_budget_resolution_2026_07_16",
        review_date="2026-07-16",
    )
    _write_immutable_json(output, receipt)
    return receipt


def verify_billing_authorization_resolution(
    *,
    repo_root: Path,
    pilot_root: Path,
    anchor_a_path: Path,
    resolution_path: Path,
    contract: Any,
    audit_head: str,
) -> dict[str, Any]:
    expected_path = (repo_root / DEFAULT_BILLING_AUTHORIZATION_RESOLUTION).resolve(
        strict=False
    )
    path = resolution_path.expanduser().resolve(strict=True)
    if path != expected_path:
        raise PilotAuditError("billing authorization path differs from fixed path")
    receipt = _read_json(path)
    anchor_a = _read_json(anchor_a_path)
    carried = _reviewed_carried_forward_cost(repo_root)
    digest = validate_billing_authorization_resolution_payload(
        receipt,
        expected_unknown_attempts=_billing_unknown_attempt_rows(pilot_root),
        expected_anchor_a=_billing_anchor_a_binding(anchor_a, anchor_a_path),
        expected_accounting=_billing_accounting(
            pilot_root,
            prior_cost_ledger=contract.prior_cost_ledger,
            carried_forward_cost=carried,
        ),
        expected_authorization=_billing_authorization_sources(
            repo_root, audit_head=audit_head
        ),
        expected_runtime_task_manifest_sha256=str(
            contract.runtime_manifest["runtime_task_manifest_sha256"]
        ),
        expected_freeze_manifest_sha256=str(
            contract.freeze_manifest["freeze_manifest_sha256"]
        ),
    )
    resolution_git = _git_committed_file_proof(repo_root, path, head=audit_head)
    anchor_a_git = _git_committed_file_proof(
        repo_root, anchor_a_path, head=audit_head
    )
    if not (
        resolution_git.get("passed") is True
        and resolution_git.get("path")
        == DEFAULT_BILLING_AUTHORIZATION_RESOLUTION.as_posix()
        and anchor_a_git.get("passed") is True
        and _git_is_ancestor(
            repo_root,
            str(anchor_a_git.get("anchor_commit") or ""),
            str(resolution_git.get("anchor_commit") or ""),
            strict=True,
        )
    ):
        raise PilotAuditError(
            "billing authorization must be committed after anchor A"
        )
    accounting = receipt["accounting"]
    return {
        "status": "applied",
        "billing_fact_status": "unresolved_reserved",
        "action_disposition": (
            "continue_under_preexisting_budget_authorization"
        ),
        "billing_authorization_resolution_sha256": digest,
        "file_sha256": resolution_git["file_sha256"],
        "committed_blob_sha256": resolution_git["committed_blob_sha256"],
        "receipt_commit": resolution_git["anchor_commit"],
        "audit_head": resolution_git["git_head"],
        "anchor_a_commit": anchor_a_git["anchor_commit"],
        "anchor_a_phase_evidence_receipt_sha256": anchor_a[
            "phase_evidence_receipt_sha256"
        ],
        "unknown_attempt_count": len(receipt["unknown_attempts"]),
        "unknown_attempt_set_sha256": receipt["unknown_attempt_set_sha256"],
        "unknown_reserve_yuan": accounting["phase_unknown_reserve_yuan"],
        "total_accounted_cost_yuan": accounting["total_accounted_cost_yuan"],
        "hard_fuse_yuan": accounting["hard_fuse_yuan"],
    }


def _evidence_v2_structural_links_are_exact(
    *,
    phase_evidence: Mapping[str, Any],
    transition_proof: Mapping[str, Any],
    source: Mapping[str, Any],
    source_files: Sequence[Mapping[str, Any]],
    auditor_proof: Mapping[str, Any],
) -> bool:
    """Cross-bind every duplicated Evidence-v2 identity in a stored GO report."""

    try:
        if set(phase_evidence) != {
            "current_phase_evidence_receipt_sha256",
            "evidence_tree",
        } or set(transition_proof) != {
            "ok",
            "verification_scope",
            "transition_receipt_sha256",
            "anchor_a_phase_evidence_receipt_sha256",
            "anchor_b_phase_evidence_receipt_sha256",
            "anchor_a",
            "anchor_b",
            "transition",
            "resume_execution_head",
            "audit_head",
            "evidence_tree",
            "phase_receipt_inventory",
        }:
            return False
        if (
            transition_proof.get("ok") is not True
            or transition_proof.get("verification_scope")
            != "git_anchored_phase_a_zero_call_resume_phase_b"
            or not _is_lower_hex(
                transition_proof.get("transition_receipt_sha256"), 64
            )
            or not _is_lower_hex(
                transition_proof.get("resume_execution_head"), 40
            )
        ):
            return False

        anchor_a_proof = transition_proof["anchor_a"]
        anchor_b_proof = transition_proof["anchor_b"]
        transition_git_proof = transition_proof["transition"]
        proofs = (anchor_a_proof, anchor_b_proof, transition_git_proof)
        if not all(isinstance(proof, Mapping) for proof in proofs):
            return False

        anchor_a_digest = transition_proof[
            "anchor_a_phase_evidence_receipt_sha256"
        ]
        anchor_b_digest = transition_proof[
            "anchor_b_phase_evidence_receipt_sha256"
        ]
        if (
            not _is_lower_hex(anchor_a_digest, 64)
            or not _is_lower_hex(anchor_b_digest, 64)
            or anchor_a_digest == anchor_b_digest
        ):
            return False

        gate_tree = phase_evidence["evidence_tree"]
        transition_tree = transition_proof["evidence_tree"]
        if (
            not isinstance(gate_tree, Mapping)
            or not isinstance(transition_tree, Mapping)
            or dict(gate_tree) != dict(transition_tree)
            or set(gate_tree) != {
                "top_level_entries",
                "provider_event_counts",
                "provider_lock_files",
                "phase_receipts",
                "phase_receipt_files",
                "phase_receipt_file_set_sha256",
            }
            or gate_tree.get("top_level_entries")
            != ["phase_receipts", "provider_events", "provider_locks"]
        ):
            return False

        normalized_source_files: list[dict[str, Any]] = []
        for row in source_files:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"path", "bytes", "sha256"}
                or not _is_canonical_relative_path(row.get("path"))
                or not isinstance(row.get("bytes"), int)
                or row.get("bytes", -1) < 0
                or not _is_lower_hex(row.get("sha256"), 64)
            ):
                return False
            normalized_source_files.append(dict(row))
        if (
            normalized_source_files
            != sorted(normalized_source_files, key=lambda row: str(row["path"]))
            or len({str(row["path"]) for row in normalized_source_files})
            != len(normalized_source_files)
        ):
            return False

        expected_provider_set = set(EXPECTED_PILOT_PROVIDERS)
        event_counts = Counter({provider: 0 for provider in EXPECTED_PILOT_PROVIDERS})
        source_lock_paths: set[str] = set()
        for row in normalized_source_files:
            path = str(row["path"])
            if path.startswith("evidence/provider_events/"):
                parts = Path(path).parts
                if (
                    len(parts) != 4
                    or parts[:2] != ("evidence", "provider_events")
                    or parts[2] not in expected_provider_set
                    or Path(parts[3]).suffix != ".json"
                ):
                    return False
                event_counts[parts[2]] += 1
            elif path.startswith("evidence/provider_locks/"):
                source_lock_paths.add(path)
        expected_lock_files = sorted(
            f"{provider}.lock" for provider in EXPECTED_PILOT_PROVIDERS
        )
        if (
            gate_tree.get("provider_event_counts")
            != {
                provider: event_counts[provider]
                for provider in EXPECTED_PILOT_PROVIDERS
            }
            or any(event_counts[provider] <= 0 for provider in EXPECTED_PILOT_PROVIDERS)
            or gate_tree.get("provider_lock_files") != expected_lock_files
            or source_lock_paths
            != {
                f"evidence/provider_locks/{filename}"
                for filename in expected_lock_files
            }
        ):
            return False

        receipts = gate_tree["phase_receipts"]
        receipt_files = gate_tree["phase_receipt_files"]
        if (
            not isinstance(receipts, Mapping)
            or set(receipts) != {anchor_a_digest, anchor_b_digest}
            or not isinstance(receipt_files, list)
        ):
            return False

        expected_rows: list[dict[str, Any]] = []
        for digest, proof in (
            (anchor_a_digest, anchor_a_proof),
            (anchor_b_digest, anchor_b_proof),
        ):
            row = receipts[digest]
            if (
                not isinstance(row, Mapping)
                or set(row) != {"path", "bytes", "sha256"}
                or row.get("path")
                != f"evidence/phase_receipts/{digest}.json"
                or not isinstance(row.get("bytes"), int)
                or row.get("bytes", 0) <= 0
                or row.get("sha256") != proof.get("file_sha256")
            ):
                return False
            expected_rows.append(dict(row))
        expected_rows.sort(key=lambda row: str(row["path"]))
        if (
            receipt_files != expected_rows
            or gate_tree.get("phase_receipt_file_set_sha256")
            != canonical_sha256(expected_rows)
        ):
            return False

        inventory = transition_proof["phase_receipt_inventory"]
        if not isinstance(inventory, Mapping) or set(inventory) != {
            "before_file_count",
            "after_file_count",
            "before_file_set_sha256",
            "after_file_set_sha256",
            "added_phase_evidence_receipt_sha256",
        }:
            return False
        anchor_a_row = next(
            row
            for row in expected_rows
            if row["path"]
            == f"evidence/phase_receipts/{anchor_a_digest}.json"
        )
        if (
            inventory.get("before_file_count") != 1
            or inventory.get("after_file_count") != 2
            or inventory.get("before_file_set_sha256")
            != canonical_sha256([anchor_a_row])
            or inventory.get("after_file_set_sha256")
            != canonical_sha256(expected_rows)
            or inventory.get("added_phase_evidence_receipt_sha256")
            != anchor_b_digest
        ):
            return False

        source_receipt_rows = sorted(
            (
                dict(row)
                for row in normalized_source_files
                if str(row.get("path") or "").startswith(
                    "evidence/phase_receipts/"
                )
            ),
            key=lambda row: str(row["path"]),
        )
        if source_receipt_rows != expected_rows:
            return False

        proof_hashes = (
            (
                source.get("anchor_a_file_sha256"),
                anchor_a_proof,
                DEFAULT_ANCHOR_A.as_posix(),
            ),
            (
                source.get("anchor_b_file_sha256"),
                anchor_b_proof,
                DEFAULT_ANCHOR_B.as_posix(),
            ),
            (
                source.get("transition_receipt_file_sha256"),
                transition_git_proof,
                DEFAULT_TRANSITION_RECEIPT.as_posix(),
            ),
        )
        if any(
            set(proof)
            != {
                "passed",
                "path",
                "file_sha256",
                "committed_blob_sha256",
                "git_head",
                "anchor_commit",
            }
            or proof.get("passed") is not True
            or proof.get("path") != expected_path
            or not _is_lower_hex(proof.get("file_sha256"), 64)
            or proof.get("file_sha256") != proof.get("committed_blob_sha256")
            or source_hash != proof.get("file_sha256")
            or not _is_lower_hex(proof.get("git_head"), 40)
            or not _is_lower_hex(proof.get("anchor_commit"), 40)
            for source_hash, proof, expected_path in proof_hashes
        ):
            return False

        audit_head = transition_proof.get("audit_head")
        execution_head = transition_proof.get("resume_execution_head")
        anchor_a_commit = anchor_a_proof.get("anchor_commit")
        anchor_b_commit = anchor_b_proof.get("anchor_commit")
        if (
            not _is_lower_hex(audit_head, 40)
            or source.get("audit_git_head") != audit_head
            or any(
                proof.get("git_head") != audit_head
                for _, proof, _ in proof_hashes
            )
            or anchor_b_commit
            != transition_git_proof.get("anchor_commit")
            or len({anchor_a_commit, execution_head, anchor_b_commit}) != 3
            or execution_head == audit_head
        ):
            return False

        current_digest = phase_evidence.get(
            "current_phase_evidence_receipt_sha256"
        )
        if not (
            current_digest == anchor_b_digest
            and source.get("current_phase_evidence_receipt_sha256")
            == anchor_b_digest
            and inventory.get("added_phase_evidence_receipt_sha256")
            == anchor_b_digest
        ):
            return False

        if (
            set(auditor_proof)
            != {
                "passed",
                "path",
                "auditor_implementation_sha256",
                "committed_blob_sha256",
                "audit_git_head",
                "auditor_commit",
            }
            or auditor_proof.get("passed") is not True
            or auditor_proof.get("path")
            != "experiments/llm_sim_v2/audit_pilot.py"
            or not _is_lower_hex(
                auditor_proof.get("auditor_implementation_sha256"), 64
            )
            or auditor_proof.get("auditor_implementation_sha256")
            != auditor_proof.get("committed_blob_sha256")
            or not _is_lower_hex(auditor_proof.get("audit_git_head"), 40)
            or not _is_lower_hex(auditor_proof.get("auditor_commit"), 40)
            or source.get("auditor_implementation_sha256")
            != auditor_proof.get("auditor_implementation_sha256")
            or source.get("auditor_committed_blob_sha256")
            != auditor_proof.get("committed_blob_sha256")
            or source.get("auditor_implementation_sha256")
            != source.get("auditor_committed_blob_sha256")
            or source.get("audit_git_head") != auditor_proof.get("audit_git_head")
            or source.get("auditor_commit") != auditor_proof.get("auditor_commit")
        ):
            return False
    except (KeyError, StopIteration, TypeError, ValueError):
        return False
    return True


def _billing_resolution_structure_is_exact(
    *,
    gate: Any,
    source: Any,
    accounting: Any,
    providers: Any,
) -> bool:
    if not all(
        isinstance(value, Mapping)
        for value in (gate, source, accounting, providers)
    ):
        return False
    evidence = gate.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    if type(accounting.get("unknown_attempt_count")) is not int:
        return False
    try:
        unknown_count = int(accounting["unknown_attempt_count"])
        unknown_reserve = float(accounting["unknown_cost_reserve_yuan"])
        total_accounted = float(accounting["total_accounted_cost_yuan"])
        hard_fuse = float(accounting["hard_fuse_yuan"])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        unknown_count < 0
        or unknown_reserve < 0
        or not all(
            math.isfinite(value)
            for value in (unknown_reserve, total_accounted, hard_fuse)
        )
    ):
        return False

    provider_unknown_count = 0
    provider_unknown_reserve = 0.0
    for provider in EXPECTED_PILOT_PROVIDERS:
        row = providers.get(provider)
        if not isinstance(row, Mapping):
            return False
        provider_accounting = row.get("accounting")
        disclosures = row.get("disclosures")
        needs_user = (
            disclosures.get("needs_user")
            if isinstance(disclosures, Mapping)
            else None
        )
        if not isinstance(provider_accounting, Mapping) or not isinstance(
            needs_user, Mapping
        ):
            return False
        if type(provider_accounting.get("unknown_attempt_count")) is not int:
            return False
        try:
            provider_count = int(provider_accounting["unknown_attempt_count"])
            provider_reserve = float(
                provider_accounting["unknown_cost_reserve_yuan"]
            )
        except (KeyError, TypeError, ValueError):
            return False
        if provider_count < 0 or provider_reserve < 0:
            return False
        provider_unknown_count += provider_count
        provider_unknown_reserve += provider_reserve
        if (
            needs_user.get("required") is not bool(provider_count)
            or needs_user.get("reason")
            != ("unknown_provider_billing_reserved" if provider_count else None)
            or needs_user.get("unknown_cost_attempt_count") != provider_count
            or type(needs_user.get("unknown_cost_attempt_count")) is not int
            or type(needs_user.get("record_count")) is not int
            or int(needs_user.get("record_count", -1)) < (1 if provider_count else 0)
            or not isinstance(needs_user.get("record_task_ids"), list)
            or len(needs_user.get("record_task_ids", []))
            != int(needs_user.get("record_count", -1))
        ):
            return False
    if (
        provider_unknown_count != unknown_count
        or not math.isclose(
            provider_unknown_reserve,
            unknown_reserve,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    ):
        return False

    status = evidence.get("status")
    source_file_sha = source.get(
        "billing_authorization_resolution_file_sha256"
    )
    source_receipt_sha = source.get(
        "billing_authorization_resolution_sha256"
    )
    if status == "not_required":
        return (
            set(evidence) == {"status", "unknown_attempt_count"}
            and gate.get("passed") is True
            and type(evidence.get("unknown_attempt_count")) is int
            and unknown_count == 0
            and unknown_reserve == 0.0
            and evidence.get("unknown_attempt_count") == 0
            and source_file_sha is None
            and source_receipt_sha is None
        )
    if status == "missing_or_invalid":
        return (
            set(evidence) == {"status", "unknown_attempt_count", "reason"}
            and gate.get("passed") is False
            and type(evidence.get("unknown_attempt_count")) is int
            and unknown_count > 0
            and evidence.get("unknown_attempt_count") == unknown_count
            and isinstance(evidence.get("reason"), str)
            and bool(evidence.get("reason"))
            and source_file_sha is None
            and source_receipt_sha is None
        )
    if status != "applied":
        return False
    expected_fields = {
        "status",
        "billing_fact_status",
        "action_disposition",
        "billing_authorization_resolution_sha256",
        "file_sha256",
        "committed_blob_sha256",
        "receipt_commit",
        "audit_head",
        "anchor_a_commit",
        "anchor_a_phase_evidence_receipt_sha256",
        "unknown_attempt_count",
        "unknown_attempt_set_sha256",
        "unknown_reserve_yuan",
        "total_accounted_cost_yuan",
        "hard_fuse_yuan",
    }
    try:
        proof_reserve = float(evidence["unknown_reserve_yuan"])
        proof_total = float(evidence["total_accounted_cost_yuan"])
        proof_fuse = float(evidence["hard_fuse_yuan"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        set(evidence) == expected_fields
        and gate.get("passed") is True
        and type(evidence.get("unknown_attempt_count")) is int
        and unknown_count > 0
        and unknown_reserve > 0.0
        and total_accounted < hard_fuse
        and evidence.get("billing_fact_status") == "unresolved_reserved"
        and evidence.get("action_disposition")
        == "continue_under_preexisting_budget_authorization"
        and evidence.get("unknown_attempt_count") == unknown_count
        and math.isclose(
            proof_reserve, unknown_reserve, rel_tol=0.0, abs_tol=1e-8
        )
        and math.isclose(
            proof_total, total_accounted, rel_tol=0.0, abs_tol=1e-8
        )
        and math.isclose(proof_fuse, hard_fuse, rel_tol=0.0, abs_tol=1e-8)
        and proof_total < proof_fuse
        and all(
            _is_lower_hex(evidence.get(field), length)
            for field, length in (
                ("billing_authorization_resolution_sha256", 64),
                ("file_sha256", 64),
                ("committed_blob_sha256", 64),
                ("receipt_commit", 40),
                ("audit_head", 40),
                ("anchor_a_commit", 40),
                ("anchor_a_phase_evidence_receipt_sha256", 64),
                ("unknown_attempt_set_sha256", 64),
            )
        )
        and evidence.get("file_sha256")
        == evidence.get("committed_blob_sha256")
        == source_file_sha
        and evidence.get("billing_authorization_resolution_sha256")
        == source_receipt_sha
        and evidence.get("audit_head") == source.get("audit_git_head")
    )


def verify_pilot_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    """Verify report structure and internal hashes without authenticating sources."""
    if (
        report.get("schema_version")
        != "yher.llm_sim_v2.formal_pilot_audit.v1"
        or report.get("simulated") is not True
        or report.get("run_id") != RUN_ID
        or report.get("phase") != "pilot"
        or report.get("decision") not in {"GO", "BLOCK"}
        or report.get("formal_pilot_approved")
        is not (report.get("decision") == "GO")
    ):
        raise PilotAuditError("pilot audit envelope is invalid")
    payload = dict(report)
    advertised = payload.pop("pilot_audit_sha256", None)
    if advertised != canonical_sha256(payload):
        raise PilotAuditError("pilot audit digest mismatch")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(EXPECTED_GATE_NAMES):
        raise PilotAuditError("pilot audit gate set is invalid")
    if any(
        not isinstance(gate, Mapping) or gate.get("passed") is not True
        for gate in gates.values()
    ):
        all_gates_pass = False
    else:
        all_gates_pass = True
    providers = report.get("providers")
    if (
        report.get("provider_calls_performed") != 0
        or report.get("expected_providers") != list(EXPECTED_PILOT_PROVIDERS)
        or report.get("expected_task_count_per_provider")
        != EXPECTED_TASKS_PER_PROVIDER
        or not isinstance(providers, Mapping)
        or set(providers) != set(EXPECTED_PILOT_PROVIDERS)
    ):
        raise PilotAuditError("pilot audit provider set is invalid")
    source = report.get("source_binding")
    source_files = source.get("pilot_source_files") if isinstance(source, Mapping) else None
    if (
        not isinstance(source_files, list)
        or source.get("pilot_source_file_count") != len(source_files)
        or source.get("pilot_source_set_sha256") != canonical_sha256(source_files)
        or not isinstance(source.get("auditor_implementation_sha256"), str)
        or len(source.get("auditor_implementation_sha256", "")) != 64
        or not isinstance(source.get("audit_git_head"), str)
        or len(source.get("audit_git_head", "")) != 40
        or not isinstance(source.get("runtime_task_manifest_sha256"), str)
        or not isinstance(source.get("freeze_manifest_sha256"), str)
    ):
        raise PilotAuditError("pilot audit source binding is invalid")
    accounting = report.get("accounting")
    try:
        accounting_valid = isinstance(accounting, Mapping) and all(
            math.isfinite(float(accounting[field])) and float(accounting[field]) >= 0
            for field in (
                "attempt_count",
                "unknown_attempt_count",
                "input_tokens",
                "output_tokens",
                "known_cost_yuan",
                "unknown_cost_reserve_yuan",
                "accounted_cost_yuan",
                "total_accounted_cost_yuan",
                "soft_warning_yuan",
                "hard_fuse_yuan",
            )
        )
    except (KeyError, TypeError, ValueError):
        accounting_valid = False
    if not accounting_valid:
        raise PilotAuditError("pilot audit accounting is invalid")
    if not _billing_resolution_structure_is_exact(
        gate=gates["billing_authorization_resolution"],
        source=source,
        accounting=accounting,
        providers=providers,
    ):
        raise PilotAuditError("pilot audit billing authorization proof is invalid")
    try:
        reviewed_cost_shape = (
            math.isclose(
                float(accounting["pre_collection_total_yuan"]),
                2.57152913,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(accounting["carried_forward_total_accounted_cost_yuan"]),
                REVIEWED_LEGACY_KNOWN_COST_YUAN,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(accounting["carried_forward_known_cost_yuan"]),
                REVIEWED_LEGACY_KNOWN_COST_YUAN,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(accounting["carried_forward_unknown_reserve_yuan"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(accounting["total_accounted_cost_yuan"]),
                float(accounting["pre_collection_total_yuan"])
                + float(accounting["accounted_cost_yuan"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and accounting["carried_forward_cost_ledger_sha256"]
            == REVIEWED_CARRIED_LEDGER_SHA256
            and accounting["carried_forward_source_phase_receipt_sha256"]
            == REVIEWED_LEGACY_RECEIPT_SHA256
            and accounting["carried_forward_source_record_set_sha256"]
            == REVIEWED_LEGACY_RECORD_SET_SHA256
        )
    except (KeyError, TypeError, ValueError):
        reviewed_cost_shape = False
    blocking_reasons = report.get("blocking_reasons")
    if report.get("decision") == "GO" and blocking_reasons != []:
        raise PilotAuditError("GO audit retains blocking reasons")
    if report.get("decision") == "GO" and not all_gates_pass:
        raise PilotAuditError("GO audit retains a failed gate")
    if report.get("decision") == "GO":
        phase_gate = gates["phase_evidence_receipts"]
        phase_evidence = phase_gate.get("evidence")
        zero_gate = gates["zero_call_resume"]
        resume_evidence = zero_gate.get("evidence")
        transition_proof = (
            resume_evidence.get("git_anchored_receipt")
            if isinstance(resume_evidence, Mapping)
            else None
        )
        billing_evidence = gates["billing_authorization_resolution"].get(
            "evidence"
        )
        anchor_proofs = (
            (
                transition_proof.get("anchor_a"),
                transition_proof.get("anchor_b"),
                transition_proof.get("transition"),
            )
            if isinstance(transition_proof, Mapping)
            else ()
        )
        inventory_proof = (
            transition_proof.get("phase_receipt_inventory")
            if isinstance(transition_proof, Mapping)
            else None
        )
        auditor_gate_proof = gates["auditor_git_provenance"].get("evidence")
        exact_structural_links = (
            _evidence_v2_structural_links_are_exact(
                phase_evidence=phase_evidence,
                transition_proof=transition_proof,
                source=source,
                source_files=source_files,
                auditor_proof=auditor_gate_proof,
            )
            if all(
                isinstance(value, Mapping)
                for value in (
                    phase_evidence,
                    transition_proof,
                    source,
                    auditor_gate_proof,
                )
            )
            else False
        )
        evidence_v2_shape = (
            isinstance(phase_evidence, Mapping)
            and len(
                str(
                    phase_evidence.get("current_phase_evidence_receipt_sha256")
                    or ""
                )
            )
            == 64
            and isinstance(phase_evidence.get("evidence_tree"), Mapping)
            and len(phase_evidence["evidence_tree"].get("phase_receipts", {})) >= 2
            and isinstance(resume_evidence, Mapping)
            and resume_evidence.get("runner_resume_observed") is True
            and resume_evidence.get("full_complete_record_count") is True
            and isinstance(transition_proof, Mapping)
            and transition_proof.get("ok") is True
            and transition_proof.get("verification_scope")
            == "git_anchored_phase_a_zero_call_resume_phase_b"
            and len(str(transition_proof.get("transition_receipt_sha256") or ""))
            == 64
            and len(str(transition_proof.get("resume_execution_head") or "")) == 40
            and len(str(transition_proof.get("audit_head") or "")) == 40
            and isinstance(inventory_proof, Mapping)
            and len(str(inventory_proof.get("after_file_set_sha256") or "")) == 64
            and phase_evidence["evidence_tree"].get(
                "phase_receipt_file_set_sha256"
            )
            == inventory_proof.get("after_file_set_sha256")
            and phase_evidence.get("current_phase_evidence_receipt_sha256")
            == transition_proof.get("anchor_b_phase_evidence_receipt_sha256")
            and len(anchor_proofs) == 3
            and all(
                isinstance(proof, Mapping)
                and proof.get("passed") is True
                and len(str(proof.get("file_sha256") or "")) == 64
                and len(str(proof.get("committed_blob_sha256") or "")) == 64
                and len(str(proof.get("anchor_commit") or "")) == 40
                for proof in anchor_proofs
            )
            and isinstance(source, Mapping)
            and source.get("anchor_a_file_sha256")
            == anchor_proofs[0].get("file_sha256")
            and source.get("anchor_b_file_sha256")
            == anchor_proofs[1].get("file_sha256")
            and source.get("transition_receipt_file_sha256")
            == anchor_proofs[2].get("file_sha256")
            and exact_structural_links
            and reviewed_cost_shape
        )
        if not evidence_v2_shape:
            raise PilotAuditError("GO audit lacks its Evidence-v2 transition proof")
        if (
            isinstance(billing_evidence, Mapping)
            and billing_evidence.get("status") == "applied"
            and (
                not isinstance(transition_proof, Mapping)
                or billing_evidence.get("receipt_commit")
                != transition_proof.get("resume_execution_head")
                or billing_evidence.get("anchor_a_commit")
                != transition_proof.get("anchor_a", {}).get("anchor_commit")
                or billing_evidence.get(
                    "anchor_a_phase_evidence_receipt_sha256"
                )
                != transition_proof.get(
                    "anchor_a_phase_evidence_receipt_sha256"
                )
            )
        ):
            raise PilotAuditError(
                "GO audit billing authorization transition binding is invalid"
            )
        go_provider_shape = all(
            isinstance(row, Mapping)
            and row.get("expected_task_state_count") == EXPECTED_TASKS_PER_PROVIDER
            and row.get("observed_record_count") == EXPECTED_TASKS_PER_PROVIDER
            and row.get("unexplained_task_ids") == []
            and row.get("unexplained_task_state_ids") == []
            and row.get("resume_evidence")
            == {
                "resumed_record_count": EXPECTED_TASKS_PER_PROVIDER,
                "lifecycle_event_count": row.get("resume_evidence", {}).get(
                    "lifecycle_event_count"
                ),
            }
            and row.get("resume_evidence", {}).get("lifecycle_event_count", 0) >= 2
            and isinstance(row.get("condition_cells"), Mapping)
            and set(row["condition_cells"]) == {"controlled", "blind"}
            and all(
                isinstance(cell, Mapping) and cell.get("passes") is True
                for cell in row["condition_cells"].values()
            )
            for row in providers.values()
        )
        if not go_provider_shape:
            raise PilotAuditError("GO audit provider evidence is incomplete")
        if float(accounting["total_accounted_cost_yuan"]) >= float(
            accounting["hard_fuse_yuan"]
        ):
            raise PilotAuditError("GO audit reaches the hard budget fuse")
    if report.get("decision") == "BLOCK" and (
        not isinstance(blocking_reasons, list) or not blocking_reasons
    ):
        raise PilotAuditError("BLOCK audit lacks a blocking reason")
    return {
        "ok": True,
        "verification_scope": "structural_only",
        "pilot_audit_sha256": advertised,
    }


def verify_pilot_audit_against_sources(
    report: Mapping[str, Any],
    *,
    repo_root: Path | str,
    pilot_root: Path | str,
    resume_before: Mapping[str, Any] | None,
    resume_after: Mapping[str, Any] | None,
    resume_receipt: Mapping[str, Any] | None,
    resume_receipt_path: Path | str | None,
    anchor_a_path: Path | str | None = None,
    anchor_b_path: Path | str | None = None,
) -> dict[str, Any]:
    """Authoritatively regenerate an audit from its Git-bound source store."""

    structural = verify_pilot_audit(report)
    regenerated = audit_formal_pilot(
        repo_root=repo_root,
        pilot_root=pilot_root,
        resume_before=resume_before,
        resume_after=resume_after,
        resume_receipt=resume_receipt,
        resume_receipt_path=resume_receipt_path,
        anchor_a_path=anchor_a_path,
        anchor_b_path=anchor_b_path,
    )
    if regenerated["pilot_audit_sha256"] != report.get("pilot_audit_sha256"):
        raise PilotAuditError(
            "source re-audit differs from the supplied pilot audit report"
        )
    return {
        "ok": True,
        "verification_scope": "authoritative_source_reaudit",
        "pilot_audit_sha256": structural["pilot_audit_sha256"],
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Persona v2 Formal Pilot Gate",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"Audit SHA-256: `{report['pilot_audit_sha256']}`",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate_name, gate in report["gates"].items():
        lines.append(f"| `{gate_name}` | {'PASS' if gate['passed'] else 'BLOCK'} |")
    lines.extend(
        [
            "",
            "## Providers",
            "",
            "| Provider | Records | Controlled complete | Blind complete | Lifecycle |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for provider, row in report["providers"].items():
        controlled = row["condition_cells"]["controlled"]["complete_fraction"]
        blind = row["condition_cells"]["blind"]["complete_fraction"]
        lines.append(
            f"| {provider} | {row['observed_record_count']}/128 | "
            f"{controlled:.1%} | {blind:.1%} | {row['provider_lifecycle']} |"
        )
    lines.extend(["", "## Blocking Reasons", ""])
    reasons = report["blocking_reasons"]
    if reasons:
        for reason in reasons:
            suffix = ""
            if reason.get("provider"):
                suffix += f" provider={reason['provider']}"
            if reason.get("task_id"):
                suffix += f" task={reason['task_id']}"
            lines.append(f"- `{reason['code']}`:{suffix} {reason['message']}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "Pilot observations remain simulated, text-only, physically isolated, "
            "and excluded from main analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def write_pilot_audit(
    report: Mapping[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    repo_root: Path | str,
    pilot_root: Path | str,
    resume_before: Mapping[str, Any] | None,
    resume_after: Mapping[str, Any] | None,
    resume_receipt: Mapping[str, Any] | None,
    resume_receipt_path: Path | str | None,
    anchor_a_path: Path | str | None = None,
    anchor_b_path: Path | str | None = None,
) -> dict[str, Any]:
    pilot_path = Path(pilot_root).expanduser().resolve(strict=True)
    run_root = pilot_path.parent
    root = Path(output_dir).expanduser().resolve(strict=False)
    if root == run_root or run_root in root.parents:
        raise PilotAuditError("audit output must remain outside the pilot run root")
    verify_pilot_audit_against_sources(
        report,
        repo_root=repo_root,
        pilot_root=pilot_root,
        resume_before=resume_before,
        resume_after=resume_after,
        resume_receipt=resume_receipt,
        resume_receipt_path=resume_receipt_path,
        anchor_a_path=anchor_a_path,
        anchor_b_path=anchor_b_path,
    )
    if root.exists():
        raise PilotAuditError(
            "audit output path must be absent, not a stale or empty directory"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    json_bytes = json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    markdown_bytes = _markdown_report(report).encode("utf-8")
    with tempfile.TemporaryDirectory(
        dir=root.parent, prefix=f".{root.name}.staging-"
    ) as temporary:
        staging = Path(temporary)
        json_path = staging / "pilot_gate.json"
        markdown_path = staging / "PILOT_GATE_REPORT.md"
        json_path.write_bytes(json_bytes)
        markdown_path.write_bytes(markdown_bytes)
        manifest = {
            "schema_version": "yher.llm_sim_v2.formal_pilot_audit_output.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "decision": report["decision"],
            "pilot_audit_sha256": report["pilot_audit_sha256"],
            "artifacts": [
                {
                    "filename": json_path.name,
                    "bytes": len(json_bytes),
                    "sha256": hashlib.sha256(json_bytes).hexdigest(),
                },
                {
                    "filename": markdown_path.name,
                    "bytes": len(markdown_bytes),
                    "sha256": hashlib.sha256(markdown_bytes).hexdigest(),
                },
            ],
        }
        manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
        (staging / "artifact_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        verify_pilot_audit_output(staging)
        post_write = audit_formal_pilot(
            repo_root=repo_root,
            pilot_root=pilot_path,
            resume_before=resume_before,
            resume_after=resume_after,
            resume_receipt=resume_receipt,
            resume_receipt_path=resume_receipt_path,
            anchor_a_path=anchor_a_path,
            anchor_b_path=anchor_b_path,
        )
        if post_write["pilot_audit_sha256"] != report.get("pilot_audit_sha256"):
            raise PilotAuditError(
                "pilot source changed while audit artifacts were emitted"
            )
        staging.rename(root)
    verify_pilot_audit_output(root)
    return manifest


def verify_pilot_audit_output(output_dir: Path | str) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve(strict=True)
    expected_files = {
        "PILOT_GATE_REPORT.md",
        "artifact_manifest.json",
        "pilot_gate.json",
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_files:
        raise PilotAuditError("pilot audit output file set is invalid")
    manifest = _read_json(root / "artifact_manifest.json")
    payload = dict(manifest)
    advertised = payload.pop("artifact_manifest_sha256", None)
    if (
        manifest.get("schema_version")
        != "yher.llm_sim_v2.formal_pilot_audit_output.v1"
        or manifest.get("simulated") is not True
        or manifest.get("run_id") != RUN_ID
        or advertised != canonical_sha256(payload)
    ):
        raise PilotAuditError("pilot audit artifact manifest is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {
        row.get("filename") for row in artifacts if isinstance(row, Mapping)
    } != {"pilot_gate.json", "PILOT_GATE_REPORT.md"} or len(artifacts) != 2:
        raise PilotAuditError("pilot audit artifact inventory is invalid")
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise PilotAuditError("pilot audit artifact row is invalid")
        path = root / str(row["filename"])
        if (
            not path.is_file()
            or row.get("bytes") != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise PilotAuditError("pilot audit artifact bytes differ from manifest")
    report = _read_json(root / "pilot_gate.json")
    verify_pilot_audit(report)
    if (
        manifest.get("decision") != report.get("decision")
        or manifest.get("pilot_audit_sha256") != report.get("pilot_audit_sha256")
    ):
        raise PilotAuditError("pilot audit report and artifact manifest differ")
    return {
        "ok": True,
        "verification_scope": "structural_artifact_only",
        "artifact_manifest_sha256": advertised,
        "pilot_audit_sha256": report["pilot_audit_sha256"],
    }


def snapshot_resume_state(pilot_root: Path | str) -> dict[str, Any]:
    root = Path(pilot_root).expanduser().resolve(strict=True)
    records_root = root / "records"
    records: dict[str, Any] = {}
    if records_root.is_dir():
        for provider_root in sorted(path for path in records_root.iterdir() if path.is_dir()):
            provider = provider_root.name
            for path in sorted(provider_root.glob("*.json")):
                row = _read_json(path)
                attempts = row.get("attempts")
                if not isinstance(attempts, list):
                    raise PilotAuditError(f"resume record lacks attempt list: {path}")
                record_key = f"{provider}/{path.stem}"
                if record_key in records:
                    raise PilotAuditError(f"duplicate resume record key: {record_key}")
                records[record_key] = {
                    "task_id": row.get("task_id"),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "attempt_count": len(attempts),
                }
    snapshot: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.resume_snapshot.v1",
        "run_id": RUN_ID,
        "phase": "pilot",
        "records": records,
        "record_count": len(records),
        "attempt_count": sum(int(row["attempt_count"]) for row in records.values()),
    }
    snapshot["resume_snapshot_sha256"] = canonical_sha256(snapshot)
    return snapshot


def _verify_resume_snapshot(snapshot: Mapping[str, Any]) -> None:
    payload = dict(snapshot)
    advertised = payload.pop("resume_snapshot_sha256", None)
    if (
        snapshot.get("schema_version") != "yher.llm_sim_v2.resume_snapshot.v1"
        or snapshot.get("run_id") != RUN_ID
        or snapshot.get("phase") != "pilot"
        or advertised != canonical_sha256(payload)
    ):
        raise PilotAuditError("resume snapshot is invalid")


def compare_resume_audits(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    require_zero_calls: bool = False,
) -> dict[str, Any]:
    _verify_resume_snapshot(before)
    _verify_resume_snapshot(after)
    before_records = before["records"]
    after_records = after["records"]
    if not isinstance(before_records, Mapping) or not isinstance(after_records, Mapping):
        raise PilotAuditError("resume snapshot records are invalid")
    before_keys = set(before_records)
    after_keys = set(after_records)
    removed = sorted(before_keys - after_keys)
    added = sorted(after_keys - before_keys)
    mutated = sorted(
        key
        for key in before_keys & after_keys
        if before_records[key].get("sha256") != after_records[key].get("sha256")
        or before_records[key].get("task_id") != after_records[key].get("task_id")
    )
    attempt_changes = [
        {
            "record_key": key,
            "before": before_records[key].get("attempt_count"),
            "after": after_records[key].get("attempt_count"),
        }
        for key in sorted(before_keys & after_keys)
        if before_records[key].get("attempt_count")
        != after_records[key].get("attempt_count")
    ]
    added_attempt_count = sum(
        int(after_records[key].get("attempt_count") or 0) for key in added
    ) + sum(
        max(
            0,
            int(change["after"] or 0) - int(change["before"] or 0),
        )
        for change in attempt_changes
    )
    formal_blocking_reasons: list[str] = []
    if require_zero_calls and added:
        formal_blocking_reasons.append("added_records_after_completed_pilot")
    if require_zero_calls and added_attempt_count:
        formal_blocking_reasons.append("added_attempts_after_completed_pilot")
    comparison: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.resume_audit.v1",
        "ok": (
            not removed
            and not mutated
            and not attempt_changes
            and not formal_blocking_reasons
        ),
        "require_zero_calls": bool(require_zero_calls),
        "before_snapshot_sha256": before["resume_snapshot_sha256"],
        "after_snapshot_sha256": after["resume_snapshot_sha256"],
        "before_record_count": before["record_count"],
        "after_record_count": after["record_count"],
        "before_attempt_count": before["attempt_count"],
        "after_attempt_count": after["attempt_count"],
        "added_record_keys": added,
        "removed_record_keys": removed,
        "mutated_record_keys": mutated,
        "attempt_count_changes": attempt_changes,
        "added_attempt_count": added_attempt_count,
        "formal_blocking_reasons": formal_blocking_reasons,
    }
    comparison["resume_audit_sha256"] = canonical_sha256(comparison)
    return comparison


def _store_delta(
    before_rows: Sequence[Mapping[str, Any]],
    after_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before = {str(row["path"]): dict(row) for row in before_rows}
    after = {str(row["path"]): dict(row) for row in after_rows}
    if len(before) != len(before_rows) or len(after) != len(after_rows):
        raise PilotAuditError("phase store delta contains duplicate paths")
    added = [after[path] for path in sorted(set(after) - set(before))]
    removed = [before[path] for path in sorted(set(before) - set(after))]
    mutated = [
        {
            "path": path,
            "before_bytes": before[path]["bytes"],
            "before_sha256": before[path]["sha256"],
            "after_bytes": after[path]["bytes"],
            "after_sha256": after[path]["sha256"],
        }
        for path in sorted(set(before) & set(after))
        if before[path] != after[path]
    ]
    return {
        "before_files": [before[path] for path in sorted(before)],
        "after_files": [after[path] for path in sorted(after)],
        "added_files": added,
        "removed_files": removed,
        "mutated_files": mutated,
    }


def _verify_phase_receipt_inventory_delta(
    pilot_root: Path,
    inventory_delta: Mapping[str, Any],
    *,
    anchor_a: Mapping[str, Any],
    anchor_b: Mapping[str, Any],
    evidence_tree: Mapping[str, Any],
) -> dict[str, Any]:
    before_rows = inventory_delta.get("before_files")
    after_rows = inventory_delta.get("after_files")
    current_rows = evidence_tree.get("phase_receipt_files")
    if not all(isinstance(value, list) for value in (before_rows, after_rows, current_rows)):
        raise PilotAuditError("transition lacks its internal phase-receipt inventory")
    recomputed = _store_delta(before_rows, after_rows)
    if dict(inventory_delta) != recomputed or after_rows != current_rows:
        raise PilotAuditError("internal phase-receipt inventory delta is stale")
    if recomputed["removed_files"] or recomputed["mutated_files"]:
        raise PilotAuditError("zero-call resume removed or mutated a phase receipt")

    a_digest = _verify_phase_receipt_envelope(anchor_a)
    b_digest = _verify_phase_receipt_envelope(anchor_b)
    if a_digest == b_digest:
        raise PilotAuditError("phase anchors A and B must be distinct receipts")
    a_path = f"evidence/phase_receipts/{a_digest}.json"
    b_path = f"evidence/phase_receipts/{b_digest}.json"
    before_by_path = {str(row.get("path")): dict(row) for row in before_rows}
    after_by_path = {str(row.get("path")): dict(row) for row in after_rows}
    if (
        len(before_by_path) != len(before_rows)
        or len(after_by_path) != len(after_rows)
        or set(before_by_path) != {a_path}
        or set(after_by_path) != {a_path, b_path}
        or a_path not in before_by_path
        or b_path in before_by_path
        or a_path not in after_by_path
        or b_path not in after_by_path
        or [str(row.get("path")) for row in recomputed["added_files"]] != [b_path]
    ):
        raise PilotAuditError(
            "zero-call resume must append only the internal phase anchor B receipt"
        )
    for digest, row in ((a_digest, after_by_path[a_path]), (b_digest, after_by_path[b_path])):
        path = pilot_root / "evidence/phase_receipts" / f"{digest}.json"
        if (
            row.get("sha256") != sha256_file(path)
            or row.get("bytes") != path.stat().st_size
        ):
            raise PilotAuditError("internal phase anchor inventory differs from disk")
    return {
        "before_file_count": len(before_rows),
        "after_file_count": len(after_rows),
        "before_file_set_sha256": canonical_sha256(before_rows),
        "after_file_set_sha256": canonical_sha256(after_rows),
        "added_phase_evidence_receipt_sha256": b_digest,
    }


def _transition_event_rows(
    pilot_root: Path,
    *,
    provider: str,
    start_index: int,
    stop_index: int,
) -> list[dict[str, Any]]:
    root = pilot_root / "evidence" / "provider_events" / provider
    paths = sorted(root.glob("*.json")) if root.is_dir() else []
    if len(paths) != stop_index or not 0 <= start_index <= stop_index:
        raise PilotAuditError("provider transition event range is invalid")
    rows: list[dict[str, Any]] = []
    for path in paths[start_index:stop_index]:
        event = _read_json(path)
        rows.append(
            {
                "path": _safe_relative(pilot_root, path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "event_index": event.get("event_index"),
                "event_type": event.get("event_type"),
                "event_sha256": event.get("event_sha256"),
                "previous_event_sha256": event.get("previous_event_sha256"),
                "invocation_id": event.get("invocation_id"),
                "start_event_sha256": event.get("start_event_sha256"),
                "status": event.get("status"),
                "invocation_kind": event.get("invocation_kind"),
                "expected_task_count": event.get("expected_task_count"),
                "resumed_record_count": event.get("resumed_record_count"),
                "provider_call_count": event.get("provider_call_count"),
                "provider_call_count_before": event.get(
                    "provider_call_count_before"
                ),
                "provider_call_count_after": event.get(
                    "provider_call_count_after"
                ),
                "before_store": event.get("before_store"),
                "after_store": event.get("after_store"),
            }
        )
    return rows


def _provider_transition_delta(
    pilot_root: Path,
    *,
    provider: str,
    anchor_a: Mapping[str, Any],
    anchor_b: Mapping[str, Any],
) -> dict[str, Any]:
    before = anchor_a["providers"][provider]
    after = anchor_b["providers"][provider]
    start_index = int(before["evidence_event_count"])
    stop_index = int(after["evidence_event_count"])
    return {
        "evidence_event_count_before": start_index,
        "evidence_event_count_after": stop_index,
        "evidence_event_count_delta": stop_index - start_index,
        "evidence_chain_head_before": before["evidence_chain_head_sha256"],
        "evidence_chain_head_after": after["evidence_chain_head_sha256"],
        "provider_call_count_before": before["provider_call_count"],
        "provider_call_count_after": after["provider_call_count"],
        "provider_call_count_delta": int(after["provider_call_count"])
        - int(before["provider_call_count"]),
        "record_count_before": before["record_count"],
        "record_count_after": after["record_count"],
        "record_count_delta": int(after["record_count"])
        - int(before["record_count"]),
        "record_set_sha256_before": before["record_set_sha256"],
        "record_set_sha256_after": after["record_set_sha256"],
        "record_set_unchanged": before["record_set_sha256"]
        == after["record_set_sha256"],
        "validated_attempt_count_before": before["validated_attempt_count"],
        "validated_attempt_count_after": after["validated_attempt_count"],
        "added_events": _transition_event_rows(
            pilot_root,
            provider=provider,
            start_index=start_index,
            stop_index=stop_index,
        ),
    }


def _assert_provider_transition(
    provider: str,
    delta: Mapping[str, Any],
) -> None:
    events = delta.get("added_events")
    if not isinstance(events, list) or len(events) != 2:
        raise PilotAuditError(f"{provider} resume must add exactly two evidence events")
    start, finish = events
    valid = (
        delta.get("evidence_event_count_delta") == 2
        and delta.get("provider_call_count_delta") == 0
        and delta.get("record_count_delta") == 0
        and delta.get("record_set_unchanged") is True
        and delta.get("validated_attempt_count_before")
        == delta.get("validated_attempt_count_after")
        and start.get("event_type") == "invocation_started"
        and finish.get("event_type") == "invocation_finished"
        and start.get("event_index") == delta.get("evidence_event_count_before")
        and finish.get("event_index")
        == int(delta.get("evidence_event_count_before") or 0) + 1
        and start.get("previous_event_sha256")
        == delta.get("evidence_chain_head_before")
        and finish.get("previous_event_sha256") == start.get("event_sha256")
        and finish.get("event_sha256") == delta.get("evidence_chain_head_after")
        and start.get("invocation_id") == finish.get("invocation_id")
        and finish.get("start_event_sha256") == start.get("event_sha256")
        and start.get("expected_task_count") == EXPECTED_TASKS_PER_PROVIDER
        and finish.get("expected_task_count") == EXPECTED_TASKS_PER_PROVIDER
        and start.get("resumed_record_count") == EXPECTED_TASKS_PER_PROVIDER
        and finish.get("resumed_record_count") == EXPECTED_TASKS_PER_PROVIDER
        and start.get("provider_call_count_before")
        == delta.get("provider_call_count_before")
        and finish.get("provider_call_count_before")
        == delta.get("provider_call_count_before")
        and finish.get("provider_call_count_after")
        == delta.get("provider_call_count_before")
        and finish.get("provider_call_count") == 0
        and finish.get("invocation_kind") == "resume"
        and finish.get("status") == "complete"
    )
    if not valid:
        raise PilotAuditError(
            f"{provider} resume evidence is not one closed zero-call resume invocation"
        )


def _snapshot_from_file_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row["path"]))
    return {
        "schema_version": "yher.llm_sim_v2.phase_store_snapshot.v1",
        "run_id": RUN_ID,
        "phase": "pilot",
        "file_count": len(ordered),
        "total_bytes": sum(int(row["bytes"]) for row in ordered),
        "file_set_sha256": canonical_sha256(ordered),
    }


def _assert_ordered_provider_transition_chain(
    provider_deltas: Mapping[str, Mapping[str, Any]],
    store_delta: Mapping[str, Any],
    *,
    anchor_a_store: Mapping[str, Any],
    anchor_b_store: Mapping[str, Any],
) -> None:
    before_rows = store_delta["before_files"]
    after_rows = store_delta["after_files"]
    state = {str(row["path"]): dict(row) for row in before_rows}
    if _snapshot_from_file_rows(list(state.values())) != dict(anchor_a_store):
        raise PilotAuditError("ordered transition does not begin at phase anchor A")
    added = {str(row["path"]): dict(row) for row in store_delta["added_files"]}
    mutated = {
        str(row["path"]): dict(row) for row in store_delta["mutated_files"]
    }
    for provider in EXPECTED_PILOT_PROVIDERS:
        start, finish = provider_deltas[provider]["added_events"]
        before_store = _snapshot_from_file_rows(list(state.values()))
        if (
            start.get("before_store") != before_store
            or finish.get("before_store") != before_store
        ):
            raise PilotAuditError(
                f"{provider} resume does not start at the preceding phase state"
            )
        start_path = str(start["path"])
        state[start_path] = {
            key: start[key] for key in ("path", "bytes", "sha256")
        }
        lifecycle_paths = [
            path
            for path in added
            if path.startswith(f"provider_lifecycle/{provider}/")
        ]
        if len(lifecycle_paths) != 1:
            raise PilotAuditError(
                f"{provider} resume must add one lifecycle disclosure"
            )
        lifecycle_path = lifecycle_paths[0]
        state[lifecycle_path] = added[lifecycle_path]
        manifest_path = f"provider_manifests/{provider}.json"
        manifest_change = mutated.get(manifest_path)
        if manifest_change is None:
            raise PilotAuditError(f"{provider} resume manifest delta is missing")
        state[manifest_path] = {
            "path": manifest_path,
            "bytes": manifest_change["after_bytes"],
            "sha256": manifest_change["after_sha256"],
        }
        if finish.get("after_store") != _snapshot_from_file_rows(
            list(state.values())
        ):
            raise PilotAuditError(
                f"{provider} finish receipt does not bind its pre-finish store"
            )
        finish_path = str(finish["path"])
        state[finish_path] = {
            key: finish[key] for key in ("path", "bytes", "sha256")
        }
    if (
        _snapshot_from_file_rows(list(state.values())) != dict(anchor_b_store)
        or sorted(state.values(), key=lambda row: str(row["path"])) != after_rows
    ):
        raise PilotAuditError("ordered transition does not terminate at phase anchor B")


def _verify_store_delta(
    pilot_root: Path,
    store_delta: Mapping[str, Any],
    *,
    anchor_a_store: Mapping[str, Any],
    anchor_b_store: Mapping[str, Any],
) -> None:
    before_rows = store_delta.get("before_files")
    after_rows = store_delta.get("after_files")
    if not isinstance(before_rows, list) or not isinstance(after_rows, list):
        raise PilotAuditError("transition receipt lacks its phase store rows")
    if (
        len(before_rows) != anchor_a_store.get("file_count")
        or sum(int(row.get("bytes") or 0) for row in before_rows)
        != anchor_a_store.get("total_bytes")
        or canonical_sha256(before_rows) != anchor_a_store.get("file_set_sha256")
        or len(after_rows) != anchor_b_store.get("file_count")
        or sum(int(row.get("bytes") or 0) for row in after_rows)
        != anchor_b_store.get("total_bytes")
        or canonical_sha256(after_rows) != anchor_b_store.get("file_set_sha256")
        or after_rows != _phase_store_file_rows(pilot_root)
    ):
        raise PilotAuditError("transition phase store rows differ from A/B or disk")
    recomputed = _store_delta(before_rows, after_rows)
    if dict(store_delta) != recomputed:
        raise PilotAuditError("transition phase store delta is stale or invalid")
    if recomputed["removed_files"]:
        raise PilotAuditError("zero-call resume removed phase-store evidence")
    expected_added_prefixes = {
        *(f"evidence/provider_events/{provider}/" for provider in EXPECTED_PILOT_PROVIDERS),
        *(f"provider_lifecycle/{provider}/" for provider in EXPECTED_PILOT_PROVIDERS),
    }
    added_paths = [str(row["path"]) for row in recomputed["added_files"]]
    if len(added_paths) != 6 or any(
        sum(path.startswith(prefix) for prefix in expected_added_prefixes) != 1
        for path in added_paths
    ):
        raise PilotAuditError("zero-call resume added an unexplained phase-store file")
    mutated_paths = {
        str(row["path"]) for row in recomputed["mutated_files"]
    }
    if mutated_paths != {
        f"provider_manifests/{provider}.json"
        for provider in EXPECTED_PILOT_PROVIDERS
    }:
        raise PilotAuditError("zero-call resume mutated an unexplained phase-store file")


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise PilotAuditError(f"immutable protocol artifact differs: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise PilotAuditError("protocol artifact temporary path already exists")
    temporary.write_bytes(payload)
    temporary.rename(path)


def verify_zero_call_resume_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path | str,
    pilot_root: Path | str,
    receipt_path: Path | str,
    anchor_a_path: Path | str,
    anchor_b_path: Path | str,
    phase_provenance: Mapping[str, Any],
    tasks: Sequence[Task],
    audit_head: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve(strict=True)
    pilot = Path(pilot_root).expanduser().resolve(strict=True)
    path = Path(receipt_path).expanduser().resolve(strict=True)
    a_path = Path(anchor_a_path).expanduser().resolve(strict=True)
    b_path = Path(anchor_b_path).expanduser().resolve(strict=True)
    stored = _read_json(path)
    if stored != dict(receipt):
        raise PilotAuditError("resume transition file differs from supplied receipt")
    payload = dict(receipt)
    advertised = payload.pop("transition_receipt_sha256", None)
    if (
        receipt.get("schema_version")
        != "yher.llm_sim_v2.pilot_resume_transition_receipt.v2"
        or receipt.get("simulated") is not True
        or receipt.get("run_id") != RUN_ID
        or receipt.get("phase") != "pilot"
        or receipt.get("providers") != list(EXPECTED_PILOT_PROVIDERS)
        or receipt.get("provider_call_count_delta") != 0
        or receipt.get("records_unchanged") is not True
        or len(str(receipt.get("resume_execution_head") or "")) != 40
        or advertised != canonical_sha256(payload)
    ):
        raise PilotAuditError("zero-call resume transition envelope is invalid")
    current_snapshot = snapshot_resume_state(pilot)
    if (
        receipt.get("before_resume_snapshot_sha256")
        != current_snapshot["resume_snapshot_sha256"]
        or receipt.get("after_resume_snapshot_sha256")
        != current_snapshot["resume_snapshot_sha256"]
        or receipt.get("before_record_count") != current_snapshot["record_count"]
        or receipt.get("after_record_count") != current_snapshot["record_count"]
        or receipt.get("before_attempt_count") != current_snapshot["attempt_count"]
        or receipt.get("after_attempt_count") != current_snapshot["attempt_count"]
    ):
        raise PilotAuditError(
            "zero-call resume snapshot declarations differ from the current record set"
        )
    contract = load_runtime_contract(repo)
    if (
        receipt.get("auditor_implementation_sha256")
        != sha256_file(Path(__file__).resolve())
        or receipt.get("runtime_task_manifest_sha256")
        != contract.runtime_manifest.get("runtime_task_manifest_sha256")
    ):
        raise PilotAuditError("resume transition code/runtime binding is invalid")

    anchor_a = _read_json(a_path)
    anchor_b = _read_json(b_path)
    a_summary = _phase_anchor_summary(anchor_a, a_path)
    b_summary = _phase_anchor_summary(anchor_b, b_path)
    if receipt.get("anchor_a") != a_summary or receipt.get("anchor_b") != b_summary:
        raise PilotAuditError("resume transition does not bind exact A/B receipt bytes")
    rebuilt_b = build_phase_evidence_receipt(
        pilot,
        phase_provenance=phase_provenance,
        tasks=tasks,
    )
    if anchor_b != rebuilt_b:
        raise PilotAuditError("phase anchor B is stale relative to the current pilot store")
    evidence_tree = _validate_evidence_tree(
        pilot,
        providers=EXPECTED_PILOT_PROVIDERS,
        required_receipts=(anchor_a, anchor_b),
    )
    for anchor, summary in ((anchor_a, a_summary), (anchor_b, b_summary)):
        digest = str(anchor["phase_evidence_receipt_sha256"])
        if evidence_tree["phase_receipts"][digest]["sha256"] != summary["file_sha256"]:
            raise PilotAuditError("internal and external phase anchor bytes differ")
    inventory_delta = receipt.get("phase_receipt_inventory_delta")
    if not isinstance(inventory_delta, Mapping):
        raise PilotAuditError("resume transition lacks phase-receipt inventory evidence")
    inventory_proof = _verify_phase_receipt_inventory_delta(
        pilot,
        inventory_delta,
        anchor_a=anchor_a,
        anchor_b=anchor_b,
        evidence_tree=evidence_tree,
    )

    provider_deltas = receipt.get("provider_deltas")
    if not isinstance(provider_deltas, Mapping) or set(provider_deltas) != set(
        EXPECTED_PILOT_PROVIDERS
    ):
        raise PilotAuditError("resume transition provider delta set is invalid")
    expected_deltas: dict[str, Any] = {}
    for provider in EXPECTED_PILOT_PROVIDERS:
        expected = _provider_transition_delta(
            pilot,
            provider=provider,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
        )
        _assert_provider_transition(
            provider,
            expected,
        )
        expected_deltas[provider] = expected
    if dict(provider_deltas) != expected_deltas:
        raise PilotAuditError("resume transition provider event delta is stale or invalid")
    if sum(
        int(row["provider_call_count_delta"])
        for row in expected_deltas.values()
    ) != 0:
        raise PilotAuditError("resume transition entered a provider transport")
    _verify_store_delta(
        pilot,
        receipt.get("store_delta") if isinstance(receipt.get("store_delta"), Mapping) else {},
        anchor_a_store=anchor_a["store_snapshot"],
        anchor_b_store=anchor_b["store_snapshot"],
    )
    _assert_ordered_provider_transition_chain(
        expected_deltas,
        receipt["store_delta"],
        anchor_a_store=anchor_a["store_snapshot"],
        anchor_b_store=anchor_b["store_snapshot"],
    )

    resolved_audit_head = str(audit_head or _git_head(repo))
    a_git = _git_committed_file_proof(repo, a_path, head=resolved_audit_head)
    b_git = _git_committed_file_proof(repo, b_path, head=resolved_audit_head)
    transition_git = _git_committed_file_proof(repo, path, head=resolved_audit_head)
    execution_head = str(receipt["resume_execution_head"])
    a_commit = str(a_git.get("anchor_commit") or "")
    b_commit = str(b_git.get("anchor_commit") or "")
    transition_commit = str(transition_git.get("anchor_commit") or "")
    git_order_pass = (
        a_git.get("passed") is True
        and b_git.get("passed") is True
        and transition_git.get("passed") is True
        and a_git.get("git_head") == resolved_audit_head
        and b_git.get("git_head") == resolved_audit_head
        and transition_git.get("git_head") == resolved_audit_head
        and b_commit == transition_commit
        and _git_is_ancestor(repo, a_commit, execution_head, strict=True)
        and _git_is_ancestor(repo, execution_head, b_commit, strict=True)
        and _git_is_ancestor(repo, b_commit, resolved_audit_head)
    )
    if not git_order_pass:
        raise PilotAuditError(
            "Git proof must satisfy A commit < resume execution HEAD < "
            "B/transition commit <= audit HEAD"
        )
    return {
        "ok": True,
        "verification_scope": "git_anchored_phase_a_zero_call_resume_phase_b",
        "transition_receipt_sha256": advertised,
        "anchor_a_phase_evidence_receipt_sha256": a_summary[
            "phase_evidence_receipt_sha256"
        ],
        "anchor_b_phase_evidence_receipt_sha256": b_summary[
            "phase_evidence_receipt_sha256"
        ],
        "anchor_a": a_git,
        "anchor_b": b_git,
        "transition": transition_git,
        "resume_execution_head": execution_head,
        "audit_head": resolved_audit_head,
        "evidence_tree": evidence_tree,
        "phase_receipt_inventory": inventory_proof,
    }


def run_zero_call_resume_probe(
    *,
    repo_root: Path | str,
    pilot_root: Path | str,
    anchor_a_path: Path | str,
    anchor_b_path: Path | str,
    transition_receipt_path: Path | str,
) -> dict[str, Any]:
    """Exercise the real resume path with a transport that cannot call a provider."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    pilot = Path(pilot_root).expanduser().resolve(strict=True)
    a_path = Path(anchor_a_path).expanduser().resolve(strict=True)
    b_path = Path(anchor_b_path).expanduser().resolve(strict=False)
    transition_path = Path(transition_receipt_path).expanduser().resolve(strict=False)
    for protocol_path in (a_path, b_path, transition_path):
        if protocol_path == pilot.parent or pilot.parent in protocol_path.parents:
            raise PilotAuditError("A/B/transition artifacts must remain outside the run root")
    if b_path.exists() or transition_path.exists():
        raise PilotAuditError("anchor B and transition paths must not already exist")

    contract = load_runtime_contract(repo)
    tasks = enumerate_tasks(contract, phase="pilot")
    phase = _read_json(pilot / "phase_provenance.json")
    anchor_a = _read_json(a_path)
    rebuilt_a = build_phase_evidence_receipt(
        pilot,
        phase_provenance=phase,
        tasks=tasks,
    )
    if anchor_a != rebuilt_a:
        raise PilotAuditError("phase anchor A is stale before the resume execution")
    internal_a = write_phase_evidence_receipt(
        pilot,
        phase_provenance=phase,
        tasks=tasks,
    )
    if internal_a != anchor_a:
        raise PilotAuditError("internal and external phase anchor A differ")
    evidence_a = _validate_evidence_tree(
        pilot,
        providers=EXPECTED_PILOT_PROVIDERS,
        required_receipts=(anchor_a,),
    )
    anchor_a_digest = str(anchor_a["phase_evidence_receipt_sha256"])
    if set(evidence_a["phase_receipts"]) != {anchor_a_digest}:
        raise PilotAuditError(
            "phase-receipt inventory before resume must contain only anchor A"
        )
    execution_head = _git_head(repo)
    a_git = _git_committed_file_proof(repo, a_path, head=execution_head)
    if not (
        a_git.get("passed") is True
        and a_git.get("git_head") == execution_head
        and _git_is_ancestor(
            repo,
            str(a_git.get("anchor_commit") or ""),
            execution_head,
            strict=True,
        )
    ):
        raise PilotAuditError(
            "phase anchor A must be committed unchanged before a descendant resume HEAD"
        )

    before = snapshot_resume_state(pilot)
    preflight = audit_formal_pilot(
        repo_root=repo,
        pilot_root=pilot,
        resume_before=before,
        resume_after=before,
        anchor_a_path=a_path,
        anchor_b_path=b_path,
    )
    allowed_preflight_blockers = {
        "zero_call_resume",
        "zero_call_resume_receipt",
        "phase_evidence_receipt",
    }
    unexpected = [
        row
        for row in preflight["blocking_reasons"]
        if row.get("code") not in allowed_preflight_blockers
    ]
    if unexpected:
        raise PilotAuditError(
            "zero-call resume preflight has non-resume blockers: "
            + ", ".join(sorted({str(row["code"]) for row in unexpected}))
        )
    billing_preflight = preflight.get("gates", {}).get(
        "billing_authorization_resolution", {}
    )
    billing_evidence = (
        billing_preflight.get("evidence")
        if isinstance(billing_preflight, Mapping)
        else None
    )
    if (
        isinstance(billing_evidence, Mapping)
        and billing_evidence.get("status") == "applied"
        and billing_evidence.get("receipt_commit") != execution_head
    ):
        raise PilotAuditError(
            "billing resolution commit must equal the zero-call resume execution HEAD"
        )
    before_files = _phase_store_file_rows(pilot)
    budget = BudgetLedger(
        soft_warning_yuan=float(contract.config["budget_yuan"]["soft_warning"]),
        hard_fuse_yuan=float(contract.config["budget_yuan"]["hard_fuse"]),
        initial_cost_yuan=float(preflight["accounting"]["total_accounted_cost_yuan"]),
    )

    class ProviderCallAttempted(BaseException):
        pass

    class BombTransport:
        def __init__(self) -> None:
            self.call_count = 0

        def complete(self, **_: Any) -> dict[str, Any]:
            self.call_count += 1
            raise ProviderCallAttempted

    transports: list[BombTransport] = []
    try:
        for provider in EXPECTED_PILOT_PROVIDERS:
            transport = BombTransport()
            transports.append(transport)
            runner = V2ProviderRunner(
                contract=contract,
                output_base=pilot.parents[1],
                phase="pilot",
                provider=provider,
                transport=transport,
                budget=budget,
                phase_provenance=phase,
            )
            runner.run_tasks(tasks)
    except ProviderCallAttempted as exc:
        raise PilotAuditError(
            "zero-call resume probe reached the provider transport"
        ) from exc
    provider_calls = sum(transport.call_count for transport in transports)
    after = snapshot_resume_state(pilot)
    comparison = compare_resume_audits(before, after, require_zero_calls=True)
    after_files = _phase_store_file_rows(pilot)
    anchor_b = write_phase_evidence_receipt(
        pilot,
        phase_provenance=phase,
        tasks=tasks,
    )
    external_b = write_phase_evidence_receipt(
        pilot,
        output=b_path,
        phase_provenance=phase,
        tasks=tasks,
    )
    if external_b != anchor_b:
        raise PilotAuditError("internal and external phase anchor B differ")
    evidence_b = _validate_evidence_tree(
        pilot,
        providers=EXPECTED_PILOT_PROVIDERS,
        required_receipts=(anchor_a, anchor_b),
    )
    phase_receipt_inventory_delta = _store_delta(
        evidence_a["phase_receipt_files"],
        evidence_b["phase_receipt_files"],
    )
    _verify_phase_receipt_inventory_delta(
        pilot,
        phase_receipt_inventory_delta,
        anchor_a=anchor_a,
        anchor_b=anchor_b,
        evidence_tree=evidence_b,
    )
    provider_deltas = {
        provider: _provider_transition_delta(
            pilot,
            provider=provider,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
        )
        for provider in EXPECTED_PILOT_PROVIDERS
    }
    for provider, delta in provider_deltas.items():
        _assert_provider_transition(
            provider,
            delta,
        )
    store_delta = _store_delta(before_files, after_files)
    _verify_store_delta(
        pilot,
        store_delta,
        anchor_a_store=anchor_a["store_snapshot"],
        anchor_b_store=anchor_b["store_snapshot"],
    )
    _assert_ordered_provider_transition_chain(
        provider_deltas,
        store_delta,
        anchor_a_store=anchor_a["store_snapshot"],
        anchor_b_store=anchor_b["store_snapshot"],
    )
    receipt: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.pilot_resume_transition_receipt.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "providers": list(EXPECTED_PILOT_PROVIDERS),
        "anchor_a": _phase_anchor_summary(anchor_a, a_path),
        "anchor_b": _phase_anchor_summary(anchor_b, b_path),
        "resume_execution_head": execution_head,
        "provider_call_count_delta": provider_calls,
        "records_unchanged": bool(comparison["ok"] and provider_calls == 0),
        "before_resume_snapshot_sha256": before["resume_snapshot_sha256"],
        "after_resume_snapshot_sha256": after["resume_snapshot_sha256"],
        "before_record_count": before["record_count"],
        "after_record_count": after["record_count"],
        "before_attempt_count": before["attempt_count"],
        "after_attempt_count": after["attempt_count"],
        "provider_deltas": provider_deltas,
        "store_delta": store_delta,
        "phase_receipt_inventory_delta": phase_receipt_inventory_delta,
        "runtime_task_manifest_sha256": contract.runtime_manifest[
            "runtime_task_manifest_sha256"
        ],
        "auditor_implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt["transition_receipt_sha256"] = canonical_sha256(receipt)
    _write_immutable_json(transition_path, receipt)
    return {
        "receipt": receipt,
        "receipt_path": str(transition_path),
        "anchor_a": anchor_a,
        "anchor_a_path": str(a_path),
        "anchor_b": anchor_b,
        "anchor_b_path": str(b_path),
        "resume_before": before,
        "resume_after": after,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--resume-before", type=Path)
    parser.add_argument("--resume-after", type=Path)
    parser.add_argument("--resume-receipt", "--resume-transition", dest="resume_receipt", type=Path)
    parser.add_argument("--anchor-a", type=Path)
    parser.add_argument("--anchor-b", type=Path)
    parser.add_argument("--prepare-anchor-a", action="store_true")
    parser.add_argument("--prepare-billing-resolution", action="store_true")
    parser.add_argument("--run-resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = args.repo_root.expanduser().resolve(strict=True)
    pilot = args.pilot_root.expanduser().resolve(strict=True)
    anchor_a_path = args.anchor_a or repo / DEFAULT_ANCHOR_A
    anchor_b_path = args.anchor_b or repo / DEFAULT_ANCHOR_B
    transition_path = args.resume_receipt or repo / DEFAULT_TRANSITION_RECEIPT
    if sum(
        bool(value)
        for value in (
            args.prepare_anchor_a,
            args.prepare_billing_resolution,
            args.run_resume,
        )
    ) > 1:
        raise PilotAuditError("protocol actions are separate commit stages")
    if args.prepare_anchor_a:
        contract = load_runtime_contract(repo)
        tasks = enumerate_tasks(contract, phase="pilot")
        phase = _read_json(pilot / "phase_provenance.json")
        internal = write_phase_evidence_receipt(
            pilot,
            phase_provenance=phase,
            tasks=tasks,
        )
        external = write_phase_evidence_receipt(
            pilot,
            output=anchor_a_path,
            phase_provenance=phase,
            tasks=tasks,
        )
        if internal != external:
            raise PilotAuditError("internal and external phase anchor A differ")
        print(json.dumps(external, ensure_ascii=False, sort_keys=True))
        return 0
    if args.prepare_billing_resolution:
        receipt = prepare_billing_authorization_resolution(
            repo_root=repo,
            pilot_root=pilot,
            anchor_a_path=anchor_a_path,
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if args.run_resume:
        result = run_zero_call_resume_probe(
            repo_root=repo,
            pilot_root=pilot,
            anchor_a_path=anchor_a_path,
            anchor_b_path=anchor_b_path,
            transition_receipt_path=transition_path,
        )
        print(json.dumps(result["receipt"], ensure_ascii=False, sort_keys=True))
        return 0
    resume_before = _read_json(args.resume_before) if args.resume_before else None
    resume_after = _read_json(args.resume_after) if args.resume_after else None
    resume_receipt = _read_json(transition_path) if transition_path.is_file() else None
    report = audit_formal_pilot(
        repo_root=repo,
        pilot_root=pilot,
        resume_before=resume_before,
        resume_after=resume_after,
        resume_receipt=resume_receipt,
        resume_receipt_path=transition_path if resume_receipt is not None else None,
        anchor_a_path=anchor_a_path,
        anchor_b_path=anchor_b_path,
    )
    manifest = write_pilot_audit(
        report,
        args.output,
        repo_root=repo,
        pilot_root=pilot,
        resume_before=resume_before,
        resume_after=resume_after,
        resume_receipt=resume_receipt,
        resume_receipt_path=transition_path if resume_receipt is not None else None,
        anchor_a_path=anchor_a_path,
        anchor_b_path=anchor_b_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if report["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
