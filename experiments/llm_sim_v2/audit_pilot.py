"""Read-only, fail-closed approval gate for the formal Persona-v2 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .prompts import assert_blind_no_leakage
from .runner import (
    BudgetLedger,
    FROZEN_COMMIT,
    Task,
    V2ProviderRunner,
    compute_outcomes,
    enumerate_tasks,
    load_runtime_contract,
    parse_provider_output,
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
EXPECTED_GATE_NAMES = (
    "auditor_git_provenance",
    "runtime_and_phase_provenance",
    "formal_scope",
    "exact_task_states",
    "condition_cells",
    "model_identity",
    "text_only_and_leakage",
    "pilot_main_isolation",
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
    return {
        "collection_mode": phase["collection_mode"],
        "development_only": phase["development_only"],
        "partial": phase["partial"],
        "formal_analysis_eligible": phase["formal_analysis_eligible"],
        "phase_provenance_sha256": phase["phase_provenance_sha256"],
        "freeze_manifest_sha256": phase["freeze"]["freeze_manifest_sha256"],
        "source_set_sha256": phase["source"]["source_set_sha256"],
        "target_set_hash": phase["target"]["target_set_hash"],
        "grid_sha256": phase["grid_sha256"],
        "prompt_ledger_sha256": phase["prompt"]["prompt_ledger_sha256"],
        "prompt_revision": phase["prompt"]["revision"],
        "prompt_contract_sha256": phase["prompt"]["prompt_contract_sha256"],
        "runtime_task_manifest_sha256": phase["runtime"][
            "runtime_task_manifest_sha256"
        ],
        "execution_commit": phase["runtime"]["execution_commit"],
        "runtime_file_set_sha256": phase["runtime"]["runtime_file_set_sha256"],
    }


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
        "schema_version": "yher.llm_sim_v2.response_record.v1",
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
    if needs_user_ids:
        blockers.add(
            "needs_user_unresolved",
            "unknown provider billing remains unresolved for completed pilot records",
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
) -> dict[str, Any]:
    """Audit one formal pilot root without reading a main or provider endpoint."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    pilot = Path(pilot_root).expanduser().resolve(strict=True)
    blockers = _Blockers()
    contract = load_runtime_contract(repo)
    runtime_manifest = contract.runtime_manifest
    if not isinstance(runtime_manifest, Mapping):
        raise PilotAuditError("committed runtime task manifest is missing")
    runtime_proof = verify_runtime_task_manifest(
        contract, runtime_manifest, verify_git=True
    )
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
    running_total = float(prior["pre_run_total_bound_yuan"])
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
    if resume_before is None or resume_after is None:
        resume_comparison = {
            "ok": False,
            "require_zero_calls": True,
            "formal_blocking_reasons": ["resume_snapshots_missing"],
        }
        resume_pass = False
        blockers.add(
            "zero_call_resume",
            "formal approval requires before/after snapshots from an immediate resume",
        )
    else:
        resume_comparison = compare_resume_audits(
            resume_before,
            resume_after,
            require_zero_calls=True,
        )
        current_snapshot = snapshot_resume_state(pilot)
        after_to_current = compare_resume_audits(
            resume_after,
            current_snapshot,
            require_zero_calls=True,
        )
        expected_complete_records = len(providers) * EXPECTED_TASKS_PER_PROVIDER
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
        if not resume_pass:
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
        },
    }
    report["pilot_audit_sha256"] = canonical_sha256(report)
    return report


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
    blocking_reasons = report.get("blocking_reasons")
    if report.get("decision") == "GO" and blocking_reasons != []:
        raise PilotAuditError("GO audit retains blocking reasons")
    if report.get("decision") == "GO" and not all_gates_pass:
        raise PilotAuditError("GO audit retains a failed gate")
    if report.get("decision") == "GO":
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


def _resume_receipt_git_proof(
    repo_root: Path, receipt_path: Path
) -> dict[str, Any]:
    path = receipt_path.expanduser().resolve(strict=True)
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError:
        relative = ""
    current_sha = sha256_file(path)

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
    anchor_commit: str | None = None
    try:
        if not relative:
            raise ValueError
        head = str(git("rev-parse", "HEAD"))
        git("ls-files", "--error-unmatch", relative)
        committed = git("show", f"HEAD:{relative}", binary=True)
        assert isinstance(committed, bytes)
        committed_sha = hashlib.sha256(committed).hexdigest()
        anchor_commit = str(git("log", "-1", "--format=%H", "--", relative)) or None
    except (
        AssertionError,
        OSError,
        subprocess.CalledProcessError,
        UnicodeError,
        ValueError,
    ):
        pass
    return {
        "passed": (
            bool(relative)
            and committed_sha == current_sha
            and isinstance(head, str)
            and len(head) == 40
            and isinstance(anchor_commit, str)
            and len(anchor_commit) == 40
        ),
        "path": relative or str(path),
        "receipt_file_sha256": current_sha,
        "committed_blob_sha256": committed_sha,
        "anchor_git_head": head,
        "anchor_commit": anchor_commit,
    }


def verify_zero_call_resume_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path | str,
    pilot_root: Path | str,
    receipt_path: Path | str,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve(strict=True)
    pilot = Path(pilot_root).expanduser().resolve(strict=True)
    path = Path(receipt_path).expanduser().resolve(strict=True)
    stored = _read_json(path)
    if stored != dict(receipt):
        raise PilotAuditError("resume receipt file differs from supplied receipt")
    payload = dict(receipt)
    advertised = payload.pop("resume_receipt_sha256", None)
    if (
        receipt.get("schema_version")
        != "yher.llm_sim_v2.zero_call_resume_receipt.v1"
        or receipt.get("simulated") is not True
        or receipt.get("run_id") != RUN_ID
        or receipt.get("phase") != "pilot"
        or receipt.get("providers") != list(EXPECTED_PILOT_PROVIDERS)
        or receipt.get("provider_call_count") != 0
        or receipt.get("records_unchanged") is not True
        or receipt.get("before_record_count")
        != len(EXPECTED_PILOT_PROVIDERS) * EXPECTED_TASKS_PER_PROVIDER
        or receipt.get("after_record_count")
        != len(EXPECTED_PILOT_PROVIDERS) * EXPECTED_TASKS_PER_PROVIDER
        or receipt.get("before_resume_snapshot_sha256")
        != receipt.get("after_resume_snapshot_sha256")
        or advertised != canonical_sha256(payload)
    ):
        raise PilotAuditError("zero-call resume receipt envelope is invalid")
    added_lifecycle = receipt.get("added_lifecycle_files")
    if (
        not isinstance(added_lifecycle, list)
        or len(added_lifecycle) != len(EXPECTED_PILOT_PROVIDERS)
        or {
            str(path_value).split("/")[1]
            for path_value in added_lifecycle
            if str(path_value).startswith("provider_lifecycle/")
            and len(str(path_value).split("/")) == 3
        }
        != set(EXPECTED_PILOT_PROVIDERS)
    ):
        raise PilotAuditError("resume receipt lifecycle increment is invalid")
    contract = load_runtime_contract(repo)
    if (
        receipt.get("auditor_implementation_sha256")
        != sha256_file(Path(__file__).resolve())
        or receipt.get("runtime_task_manifest_sha256")
        != contract.runtime_manifest.get("runtime_task_manifest_sha256")
    ):
        raise PilotAuditError("resume receipt code/runtime binding is invalid")
    current_snapshot = snapshot_resume_state(pilot)
    current_files = _source_file_rows(pilot)
    if (
        current_snapshot["resume_snapshot_sha256"]
        != receipt.get("after_resume_snapshot_sha256")
        or canonical_sha256(current_files)
        != receipt.get("after_pilot_source_set_sha256")
        or len(current_files) != receipt.get("after_pilot_source_file_count")
    ):
        raise PilotAuditError("pilot store differs from anchored resume receipt")
    anchor = _resume_receipt_git_proof(repo, path)
    if not anchor["passed"]:
        raise PilotAuditError("resume receipt is not committed unchanged in Git")
    return {
        "ok": True,
        "verification_scope": "git_anchored_zero_call_resume_receipt",
        "resume_receipt_sha256": advertised,
        "anchor": anchor,
    }


def run_zero_call_resume_probe(
    *,
    repo_root: Path | str,
    pilot_root: Path | str,
    receipt_path: Path | str,
) -> dict[str, Any]:
    """Exercise the real resume path with a transport that cannot call a provider."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    pilot = Path(pilot_root).expanduser().resolve(strict=True)
    receipt_file = Path(receipt_path).expanduser().resolve(strict=False)
    if pilot.parent == receipt_file or pilot.parent in receipt_file.parents:
        raise PilotAuditError("resume receipt must remain outside the pilot run root")
    if receipt_file.exists():
        raise PilotAuditError("resume receipt path must not already exist")
    before = snapshot_resume_state(pilot)
    preflight = audit_formal_pilot(
        repo_root=repo,
        pilot_root=pilot,
        resume_before=before,
        resume_after=before,
    )
    allowed_preflight_blockers = {
        "zero_call_resume",
        "zero_call_resume_receipt",
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
    contract = load_runtime_contract(repo)
    tasks = enumerate_tasks(contract, phase="pilot")
    phase = _read_json(pilot / "phase_provenance.json")
    before_files = _source_file_rows(pilot)
    before_lifecycle = {
        row["path"]: row
        for row in before_files
        if str(row["path"]).startswith("provider_lifecycle/")
    }
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
    after_files = _source_file_rows(pilot)
    after_lifecycle = {
        row["path"]: row
        for row in after_files
        if str(row["path"]).startswith("provider_lifecycle/")
    }
    added_lifecycle = sorted(set(after_lifecycle) - set(before_lifecycle))
    receipt: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.zero_call_resume_receipt.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "providers": list(EXPECTED_PILOT_PROVIDERS),
        "provider_call_count": provider_calls,
        "records_unchanged": bool(comparison["ok"] and provider_calls == 0),
        "before_record_count": before["record_count"],
        "after_record_count": after["record_count"],
        "before_attempt_count": before["attempt_count"],
        "after_attempt_count": after["attempt_count"],
        "before_resume_snapshot_sha256": before["resume_snapshot_sha256"],
        "after_resume_snapshot_sha256": after["resume_snapshot_sha256"],
        "before_pilot_source_file_count": len(before_files),
        "after_pilot_source_file_count": len(after_files),
        "before_pilot_source_set_sha256": canonical_sha256(before_files),
        "after_pilot_source_set_sha256": canonical_sha256(after_files),
        "before_lifecycle_set_sha256": canonical_sha256(before_lifecycle),
        "after_lifecycle_set_sha256": canonical_sha256(after_lifecycle),
        "added_lifecycle_files": added_lifecycle,
        "runtime_task_manifest_sha256": contract.runtime_manifest[
            "runtime_task_manifest_sha256"
        ],
        "auditor_implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt["resume_receipt_sha256"] = canonical_sha256(receipt)
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_file.with_name(f".{receipt_file.name}.tmp")
    if temporary.exists():
        raise PilotAuditError("resume receipt temporary path already exists")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.rename(receipt_file)
    return {
        "receipt": receipt,
        "receipt_path": str(receipt_file),
        "resume_before": before,
        "resume_after": after,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--resume-before", type=Path)
    parser.add_argument("--resume-after", type=Path)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resume_before = _read_json(args.resume_before) if args.resume_before else None
    resume_after = _read_json(args.resume_after) if args.resume_after else None
    resume_receipt = _read_json(args.resume_receipt) if args.resume_receipt else None
    report = audit_formal_pilot(
        repo_root=args.repo_root,
        pilot_root=args.pilot_root,
        resume_before=resume_before,
        resume_after=resume_after,
        resume_receipt=resume_receipt,
        resume_receipt_path=args.resume_receipt,
    )
    manifest = write_pilot_audit(
        report,
        args.output,
        repo_root=args.repo_root,
        pilot_root=args.pilot_root,
        resume_before=resume_before,
        resume_after=resume_after,
        resume_receipt=resume_receipt,
        resume_receipt_path=args.resume_receipt,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if report["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
