"""Fail-closed execution evidence for one isolated Persona-v2 judge pass."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any
import uuid

from .keys import canonical_key


RUN_ID = "llm-personas-v2-dual"
CASE_SCHEMA = "yher.llm_sim_v2.judge_case_manifest.v2"
RECEIPT_SCHEMA = "yher.llm_sim_v2.judge_execution_receipt.v2"
FAILED_RECEIPT_SCHEMA = "yher.llm_sim_v2.judge_failed_execution_receipt.v2"
FAMILY_DISPOSITION_SCHEMA = "yher.llm_sim_v2.judge_family_disposition_receipt.v1"
RUN_EVIDENCE_RECEIPT_SCHEMA = "yher.llm_sim_v2.judge_run_evidence_receipt.v1"
BUDGET_AUTHORITY_SCHEMA = "yher.llm_sim_v2.judge_budget_authority.v1"
TRANSPORT_RESPONSE_SCHEMA = "yher.llm_sim_v2.judge_transport_response.v2"
BATCH_SIZE = 10
MAX_ATTEMPTS_PER_BATCH = 2
JUDGE_FAMILIES = frozenset({"claude", "gpt"})
QUESTION_FIELD_WHITELIST = ["kind", "options", "stem_blocks", "stem_text"]
JUDGE_AMENDMENT_PATH = "experiments/llm_sim_v2/judge_amendment_20260716.md"
JUDGE_RUN_ANCHOR_PATH = (
    "experiments/llm_sim_v2/evidence_anchors/judge_run_evidence_receipt.json"
)
MAIN_PHASE_ANCHOR_PATH = (
    "experiments/llm_sim_v2/evidence_anchors/main_phase_evidence_receipt.json"
)
PRODUCTION_UNKNOWN_RESERVE_YUAN = Decimal("10")
HARD_FUSE_YUAN = Decimal("450")
JUDGE_LABELS = frozenset(
    {"consistent", "inconsistent", "unknown", "insufficient_evidence"}
)
REQUIRED_OUTPUT_FIELDS = ["error_category", "label", "rationale", "simulated"]
_FORBIDDEN_JUDGE_KEYS = frozenset(
    {
        "correct_option",
        "deficit_condition",
        "difficulty",
        "answer_values",
        "anchor_id",
        "authenticity",
        "authenticity_score",
        "failure_cause",
        "failure_id",
        "failure_symptom",
        "mapping_status",
        "misconception_id",
        "nodes",
        "observable_error_policy",
        "pair_id",
        "persona_id",
        "private_correct_option",
        "provider",
        "realism",
        "realism_score",
        "row_id",
        "source_label",
        "target_node",
        "target_label",
        "target_option",
        "truthfulness",
        "truthfulness_score",
    }
)
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}\Z")
_RUN_THREAD_LOCK = threading.RLock()


class JudgeExecutionError(ValueError):
    """Raised when an isolated judge execution cannot be proven valid."""


class JudgeSchemaError(JudgeExecutionError):
    """A retryable outer-response or normalized-output schema failure."""


class JudgeIsolationError(JudgeExecutionError):
    """A non-retryable judge identity or tool-isolation failure."""


class JudgeTransportError(RuntimeError):
    """A retryable failure before a parseable transport response is returned."""

    def __init__(
        self,
        message: str,
        *,
        raw_outer_response: bytes | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        known_cost_yuan: int | float | None = None,
        unknown_cost_reserve_yuan: int | float = 0,
    ) -> None:
        super().__init__(message)
        self.raw_outer_response = raw_outer_response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.known_cost_yuan = known_cost_yuan
        self.unknown_cost_reserve_yuan = unknown_cost_reserve_yuan


class JudgePassFailed(JudgeExecutionError):
    """A failed isolated pass whose immutable evidence receipt was finalized."""

    def __init__(self, message: str, *, receipt_path: Path, receipt: Mapping[str, Any]):
        super().__init__(message)
        self.receipt_path = receipt_path
        self.receipt_sha256 = str(receipt["failed_execution_receipt_sha256"])
        self.receipt = dict(receipt)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _strict_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise JudgeExecutionError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise JudgeExecutionError(f"{label} contains non-finite JSON: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JudgeExecutionError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise JudgeExecutionError(f"{label} must be a JSON object")
    return value


def _load_object(value: Mapping[str, Any] | str | Path, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise JudgeExecutionError(f"{label} path must be a regular file")
    return _strict_json_bytes(path.read_bytes(), label=label)


def _frozen_judge_protocol() -> dict[str, Any]:
    path = Path(__file__).with_name("judge_protocol_v1.json")
    if path.is_symlink() or not path.is_file():
        raise JudgeExecutionError("frozen judge protocol file is unavailable")
    return _strict_json_bytes(path.read_bytes(), label="frozen judge protocol")


def _frozen_judge_amendment_binding() -> dict[str, Any]:
    path = Path(__file__).with_name("judge_amendment_20260716.md")
    if path.is_symlink() or not path.is_file():
        raise JudgeExecutionError("outcome-blind judge amendment file is unavailable")
    data = path.read_bytes()
    return {
        "path": JUDGE_AMENDMENT_PATH,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _self_hash(value: Mapping[str, Any], field: str, *, label: str) -> str:
    payload = dict(value)
    advertised = payload.pop(field, None)
    calculated = canonical_sha256(payload)
    if not isinstance(advertised, str) or advertised != calculated:
        raise JudgeExecutionError(f"{label} self-hash mismatch")
    return advertised


def _judge_input_bytes(cases: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        canonical_json_bytes(
            {"case_id": case["case_id"], "messages": case["judge_messages"]}
        )
        + b"\n"
        for case in cases
    )


def _assert_no_target_metadata(value: Any, *, _depth: int = 0) -> None:
    if _depth > 30:
        raise JudgeExecutionError("judge input nesting exceeds the leakage scanner limit")
    if isinstance(value, Mapping):
        normalized_keys = {canonical_key(key) for key in value}
        forbidden = normalized_keys & _FORBIDDEN_JUDGE_KEYS
        if forbidden:
            raise JudgeExecutionError(
                "judge input contains target metadata leakage: " + ", ".join(sorted(forbidden))
            )
        for key, child in value.items():
            if canonical_key(key) == "public_question":
                if not isinstance(child, Mapping) or sorted(child) != QUESTION_FIELD_WHITELIST:
                    raise JudgeExecutionError(
                        "judge public question exceeds the frozen metadata whitelist"
                    )
            _assert_no_target_metadata(child, _depth=_depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_target_metadata(child, _depth=_depth + 1)
        return
    if not isinstance(value, str):
        return
    stripped = value.strip()
    if not stripped.startswith(("{", "[", '"')):
        return

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in values:
            if key in result:
                raise JudgeExecutionError("judge input contains duplicate nested JSON keys")
            result[key] = child
        return result

    try:
        decoded = json.loads(
            stripped,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                JudgeExecutionError(f"judge input contains non-finite JSON: {token}")
            ),
        )
    except json.JSONDecodeError:
        return
    if decoded != value:
        _assert_no_target_metadata(decoded, _depth=_depth + 1)


def _validate_case_manifest(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    manifest = dict(value)
    _self_hash(manifest, "case_manifest_sha256", label="judge case manifest")
    protocol = manifest.get("judge_protocol")
    cases = manifest.get("cases")
    whitelist = manifest.get("question_field_whitelist")
    if (
        manifest.get("schema_version") != CASE_SCHEMA
        or manifest.get("simulated") is not True
        or manifest.get("run_id") != RUN_ID
        or manifest.get("analysis_population") != "main"
        or manifest.get("target_metadata_exported") is not False
        or manifest.get("provider_identity_exported") is not False
        or whitelist != QUESTION_FIELD_WHITELIST
        or not isinstance(protocol, Mapping)
        or not isinstance(cases, list)
        or manifest.get("selected_count") != len(cases)
    ):
        raise JudgeExecutionError("judge case manifest violates the v2 blind contract")
    if manifest.get("judge_protocol_sha256") != canonical_sha256(protocol):
        raise JudgeExecutionError("judge protocol hash mismatch")
    if dict(protocol) != _frozen_judge_protocol():
        raise JudgeExecutionError("judge protocol differs from the frozen protocol file")
    if manifest.get("judge_amendment") != _frozen_judge_amendment_binding():
        raise JudgeExecutionError("judge amendment binding differs from committed bytes")
    labels = protocol.get("label_definitions")
    categories = protocol.get("error_categories")
    category_policy = protocol.get("label_category_policy")
    if (
        protocol.get("schema_version") != "yher.llm_sim_v2.judge_protocol.v1"
        or protocol.get("simulated") is not True
        or not isinstance(protocol.get("scope"), str)
        or not str(protocol.get("scope")).strip()
        or protocol.get("question_field_whitelist") != whitelist
        or not isinstance(labels, Mapping)
        or set(labels) != JUDGE_LABELS
        or any(not isinstance(text, str) or not text.strip() for text in labels.values())
        or not isinstance(categories, Mapping)
        or len(categories) != 9
        or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(text, str)
            or not text.strip()
            for name, text in categories.items()
        )
        or protocol.get("required_output_fields") != REQUIRED_OUTPUT_FIELDS
        or not isinstance(category_policy, Mapping)
        or set(category_policy) != JUDGE_LABELS
        or any(
            not isinstance(allowed, list)
            or not allowed
            or len(allowed) != len(set(allowed))
            or any(category not in categories for category in allowed)
            for allowed in category_policy.values()
        )
        or set().union(*(set(allowed) for allowed in category_policy.values()))
        != set(categories)
    ):
        raise JudgeExecutionError("judge protocol is incomplete or inconsistent")
    normalized_cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise JudgeExecutionError("judge case must be an object")
        case = dict(raw)
        case_id = case.get("case_id")
        messages = case.get("judge_messages")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
            or not isinstance(messages, list)
            or not messages
            or case.get("judge_input_sha256") != canonical_sha256(messages)
        ):
            raise JudgeExecutionError("judge case identity or input hash is invalid")
        seen.add(case_id)
        _assert_no_target_metadata(messages)
        normalized_cases.append(case)
    input_bytes = _judge_input_bytes(normalized_cases)
    if manifest.get("shared_input_sha256") != hashlib.sha256(input_bytes).hexdigest():
        raise JudgeExecutionError("shared judge input hash mismatch")
    manifest["cases"] = normalized_cases
    return manifest, input_bytes


def _decimal(value: Any, *, label: str, allow_null: bool = False) -> Decimal | None:
    if value is None and allow_null:
        return None
    if isinstance(value, bool):
        raise JudgeExecutionError(f"{label} must be a non-negative number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise JudgeExecutionError(f"{label} must be a non-negative number") from exc
    if not result.is_finite() or result < 0:
        raise JudgeExecutionError(f"{label} must be a non-negative number")
    return result


def _number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _validate_output(
    value: Any, *, category_policy: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or list(sorted(value)) != sorted(REQUIRED_OUTPUT_FIELDS):
        raise JudgeExecutionError("judge output fields differ from the frozen protocol")
    output = dict(value)
    label = output.get("label")
    category = output.get("error_category")
    rationale = output.get("rationale")
    if (
        label not in JUDGE_LABELS
        or category not in category_policy[label]
        or not isinstance(rationale, str)
        or not rationale.strip()
        or output.get("simulated") is not True
    ):
        raise JudgeExecutionError("judge output violates the frozen protocol")
    return output


def _validate_raw_transport(value: Any) -> bytes:
    expected = {
        "stdout_base64",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_bytes",
        "stderr_sha256",
        "returncode",
        "environment_exported",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise JudgeSchemaError("raw CLI transport binding is invalid")
    try:
        stdout = base64.b64decode(str(value["stdout_base64"]), validate=True)
    except ValueError as exc:
        raise JudgeSchemaError("raw CLI stdout base64 is invalid") from exc
    stderr_bytes = value.get("stderr_bytes")
    returncode = value.get("returncode")
    if (
        value.get("stdout_bytes") != len(stdout)
        or value.get("stdout_sha256") != hashlib.sha256(stdout).hexdigest()
        or not isinstance(stderr_bytes, int)
        or isinstance(stderr_bytes, bool)
        or stderr_bytes < 0
        or not isinstance(value.get("stderr_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("stderr_sha256")))
        or not isinstance(returncode, int)
        or isinstance(returncode, bool)
        or value.get("environment_exported") is not False
    ):
        raise JudgeSchemaError("raw CLI stdout hash or metadata drifted")
    return stdout


def _validate_transport_response(
    data: bytes,
    *,
    case_ids: list[str],
    exact_model: str,
    category_policy: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = _strict_json_bytes(data, label="judge transport response")
    expected_fields = {
        "schema_version",
        "simulated",
        "transport_reported_models",
        "transport_reported_model_source",
        "transport_request_id",
        "results",
        "usage",
        "billing",
        "tool_calls",
    }
    if (
        frozenset(value)
        not in {
            frozenset(expected_fields),
            frozenset({*expected_fields, "raw_transport"}),
        }
        or value.get("schema_version") != TRANSPORT_RESPONSE_SCHEMA
        or value.get("simulated") is not True
        or not isinstance(value.get("transport_request_id"), str)
        or not str(value.get("transport_request_id")).strip()
        or not isinstance(value.get("tool_calls"), list)
    ):
        raise JudgeSchemaError("judge transport response envelope is invalid")
    if "raw_transport" in value:
        _validate_raw_transport(value["raw_transport"])
    reported_models = value.get("transport_reported_models")
    reported_source = value.get("transport_reported_model_source")
    if (
        not isinstance(reported_models, list)
        or any(
            not isinstance(model, str) or not _MODEL_NAME.fullmatch(model)
            for model in reported_models
        )
        or len(reported_models) != len(set(reported_models))
        or (bool(reported_models) != isinstance(reported_source, str))
        or isinstance(reported_source, str) and not reported_source.strip()
    ):
        raise JudgeSchemaError("judge transport-reported model evidence is invalid")
    if reported_models and reported_models != [exact_model]:
        raise JudgeIsolationError("judge transport-reported model identity drifted")
    if value.get("tool_calls"):
        raise JudgeIsolationError("judge tool isolation failed")
    results = value.get("results")
    usage = value.get("usage")
    billing = value.get("billing")
    if (
        not isinstance(results, list)
        or not isinstance(usage, Mapping)
        or set(usage) != {"input_tokens", "output_tokens"}
        or not isinstance(billing, Mapping)
        or set(billing) != {"known_cost_yuan", "unknown_cost_reserve_yuan"}
    ):
        raise JudgeExecutionError("judge transport response schema is invalid")
    normalized: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    for row in results:
        if not isinstance(row, Mapping) or set(row) != {"case_id", "output"}:
            raise JudgeExecutionError("judge result row schema is invalid")
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            raise JudgeExecutionError("judge result case ID is invalid")
        observed_ids.append(case_id)
        normalized.append(
            {
                "case_id": case_id,
                "output": _validate_output(
                    row.get("output"), category_policy=category_policy
                ),
            }
        )
    if observed_ids != case_ids:
        raise JudgeExecutionError("judge result coverage or order differs from its batch")
    for field in ("input_tokens", "output_tokens"):
        token_value = usage.get(field)
        if not isinstance(token_value, int) or isinstance(token_value, bool) or token_value < 0:
            raise JudgeExecutionError("judge token accounting is invalid")
    known = _decimal(
        billing.get("known_cost_yuan"), label="known judge cost", allow_null=True
    )
    reserve = _decimal(
        billing.get("unknown_cost_reserve_yuan"), label="unknown judge reserve"
    )
    assert reserve is not None
    if known is None and reserve == 0:
        raise JudgeExecutionError("unknown judge billing requires a positive reserve")
    value["_known_cost_decimal"] = known
    value["_reserve_decimal"] = reserve
    return value, normalized


def _response_accounting(
    data: bytes,
) -> tuple[int, int, Decimal | None, Decimal, dict[str, Any] | None]:
    """Recover billable metadata even when result coverage/schema later fails."""

    try:
        value = _strict_json_bytes(data, label="judge transport response")
    except JudgeExecutionError:
        return 0, 0, None, Decimal("0"), None
    usage = value.get("usage")
    billing = value.get("billing")
    if not isinstance(usage, Mapping) or not isinstance(billing, Mapping):
        return 0, 0, None, Decimal("0"), value
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or input_tokens < 0
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens < 0
    ):
        input_tokens = 0
        output_tokens = 0
    try:
        known = _decimal(
            billing.get("known_cost_yuan"),
            label="known judge cost",
            allow_null=True,
        )
        reserve = _decimal(
            billing.get("unknown_cost_reserve_yuan"),
            label="unknown judge reserve",
        )
    except JudgeExecutionError:
        return input_tokens, output_tokens, None, Decimal("0"), value
    assert reserve is not None
    return input_tokens, output_tokens, known, reserve, value


class FixtureJudgeTransport:
    """Deterministic no-network transport used only for contract tests."""

    name = "fixture"
    tools_disabled = True
    fresh_execution = True

    def __init__(self, responses: Sequence[Mapping[str, Any] | bytes | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[bytes] = []

    def command_argv(self, exact_model: str) -> list[str]:
        return ["fixture-judge", "--model", exact_model, "--tools", "disabled"]

    def executable_evidence(self) -> dict[str, Any]:
        return {
            "transport_class": (
                "experiments.llm_sim_v2.judge_execution.FixtureJudgeTransport"
            ),
            "configured_binary": "fixture-judge",
            "resolved_binary_realpath": None,
            "binary_bytes": 0,
            "binary_sha256": hashlib.sha256(b"").hexdigest(),
            "version_argv": [],
            "version_returncode": 0,
            "version_stdout": "fixture-only",
            "version_stdout_bytes": len(b"fixture-only"),
            "version_stdout_sha256": hashlib.sha256(b"fixture-only").hexdigest(),
            "version_stderr_bytes": 0,
            "version_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        }

    def invoke(self, request: bytes, *, exact_model: str, attempt_id: str) -> bytes:
        del exact_model, attempt_id
        self.requests.append(request)
        if not self._responses:
            raise JudgeTransportError("fixture response queue exhausted")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            if isinstance(response, JudgeTransportError):
                raise response
            raise JudgeTransportError(type(response).__name__) from response
        if isinstance(response, bytes):
            return response
        return canonical_json_bytes(response)


def _validate_exact_model(value: str) -> str:
    if not isinstance(value, str) or not _MODEL_NAME.fullmatch(value):
        raise JudgeExecutionError("exact judge model name is invalid")
    return value


def _cli_prompt(request: bytes) -> bytes:
    try:
        request_value = _strict_json_bytes(request, label="judge batch request")
    except JudgeExecutionError as exc:
        raise JudgeExecutionError("judge CLI received an invalid batch request") from exc
    return (
        "You are a fresh, isolated blind adjudicator. Do not use tools, files, web "
        "search, prior conversations, or external context. Judge only the cases in "
        "the request. Return one strict JSON object with exactly one `results` array; "
        "each row must contain `case_id` and `output`, in input order. Do not wrap "
        "JSON in markdown.\n\n"
        + canonical_json_bytes(request_value).decode("utf-8")
    ).encode("utf-8")


def _raw_cli_binding(
    *, stdout: bytes, stderr: bytes, returncode: int
) -> dict[str, Any]:
    return {
        "stdout_base64": base64.b64encode(stdout).decode("ascii"),
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "returncode": returncode,
        "environment_exported": False,
    }


def _codex_response_from_raw_cli(
    *,
    raw_transport: Mapping[str, Any],
    exact_model: str,
    attempt_id: str,
    unknown_cost_reserve_yuan: int | float,
) -> dict[str, Any]:
    stdout = _validate_raw_transport(raw_transport)
    if raw_transport.get("returncode") != 0:
        raise JudgeSchemaError("successful Codex judge transport has nonzero exit status")
    events: list[dict[str, Any]] = []
    try:
        events = [
            _strict_json_bytes(line, label="Codex JSONL event")
            for line in stdout.splitlines()
            if line.strip()
        ]
    except JudgeExecutionError:
        events = []
    final_text: str | None = None
    thread_id: str | None = None
    tool_calls: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    for event in events:
        if isinstance(event.get("thread_id"), str):
            thread_id = str(event["thread_id"])
        if event.get("type") == "thread.started" and isinstance(
            event.get("thread_id"), str
        ):
            thread_id = str(event["thread_id"])
        item = event.get("item")
        if isinstance(item, Mapping):
            item_type = str(item.get("type") or "")
            if item_type == "agent_message" and isinstance(item.get("text"), str):
                final_text = str(item["text"])
            elif item_type and item_type not in {
                "reasoning",
                "plan",
                "error",
            }:
                tool_calls.append(
                    {
                        "type_sha256": hashlib.sha256(
                            item_type.encode()
                        ).hexdigest()
                    }
                )
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            if isinstance(usage.get("input_tokens"), int):
                input_tokens = max(input_tokens, int(usage["input_tokens"]))
            if isinstance(usage.get("output_tokens"), int):
                output_tokens = max(output_tokens, int(usage["output_tokens"]))
    parsed: dict[str, Any] | None = None
    if final_text is not None:
        try:
            parsed = _strict_json_bytes(
                final_text.encode("utf-8"), label="Codex final judge response"
            )
        except JudgeExecutionError:
            parsed = None
    return {
        "schema_version": TRANSPORT_RESPONSE_SCHEMA,
        "simulated": True,
        "transport_reported_models": [],
        "transport_reported_model_source": None,
        "transport_request_id": thread_id or attempt_id,
        "results": parsed.get("results") if isinstance(parsed, Mapping) else None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "billing": {
            "known_cost_yuan": None,
            "unknown_cost_reserve_yuan": unknown_cost_reserve_yuan,
        },
        "tool_calls": tool_calls,
        "raw_transport": dict(raw_transport),
    }


def _claude_response_from_raw_cli(
    *,
    raw_transport: Mapping[str, Any],
    exact_model: str,
    attempt_id: str,
    unknown_cost_reserve_yuan: int | float,
) -> dict[str, Any]:
    stdout = _validate_raw_transport(raw_transport)
    if raw_transport.get("returncode") != 0:
        raise JudgeSchemaError("successful Claude judge transport has nonzero exit status")
    try:
        outer = _strict_json_bytes(stdout, label="Claude CLI response")
    except JudgeExecutionError:
        outer = {}
    result_text = outer.get("result")
    parsed: dict[str, Any] | None = None
    if isinstance(result_text, str):
        try:
            parsed = _strict_json_bytes(
                result_text.encode("utf-8"), label="Claude final judge response"
            )
        except JudgeExecutionError:
            parsed = None
    usage = outer.get("usage")
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, Mapping):
        if isinstance(usage.get("input_tokens"), int):
            input_tokens = int(usage["input_tokens"])
        if isinstance(usage.get("output_tokens"), int):
            output_tokens = int(usage["output_tokens"])
    model_usage = outer.get("modelUsage")
    reported_models = sorted(str(model) for model in model_usage) if isinstance(
        model_usage, Mapping
    ) else []
    return {
        "schema_version": TRANSPORT_RESPONSE_SCHEMA,
        "simulated": True,
        "transport_reported_models": reported_models,
        "transport_reported_model_source": (
            "raw_cli.modelUsage_keys" if reported_models else None
        ),
        "transport_request_id": str(outer.get("session_id") or attempt_id),
        "results": parsed.get("results") if isinstance(parsed, Mapping) else None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "billing": {
            "known_cost_yuan": None,
            "unknown_cost_reserve_yuan": unknown_cost_reserve_yuan,
        },
        "tool_calls": [],
        "raw_transport": dict(raw_transport),
    }


class _BaseCLIJudgeTransport:
    tools_disabled = True
    fresh_execution = True
    trusted_binary_name = ""

    def __init__(
        self,
        *,
        binary: str,
        timeout_seconds: float = 900.0,
    ) -> None:
        if not isinstance(binary, str) or not binary.strip():
            raise JudgeExecutionError("judge CLI binary is required")
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise JudgeExecutionError("judge CLI timeout must be positive")
        self.binary = binary
        self.unknown_cost_reserve_yuan = PRODUCTION_UNKNOWN_RESERVE_YUAN
        self.timeout_seconds = float(timeout_seconds)
        self._bound_binary: str | None = None

    def _resolved_binary(self) -> str:
        if os.sep in self.binary:
            path = Path(self.binary)
            if not path.is_file() or not os.access(path, os.X_OK):
                raise JudgeExecutionError(f"judge CLI unavailable: {self.binary}")
            return str(path)
        resolved = shutil.which(self.binary)
        if resolved is None:
            raise JudgeExecutionError(f"judge CLI unavailable: {self.binary}")
        return resolved

    def preflight(self) -> None:
        """Prove the configured executable exists before evidence paths are created."""

        self._resolved_binary()

    def executable_evidence(self) -> dict[str, Any]:
        if not self.trusted_binary_name:
            raise JudgeExecutionError("judge transport lacks a trusted executable name")
        resolved = Path(self._resolved_binary()).resolve(strict=True)
        trusted_value = shutil.which(self.trusted_binary_name)
        if trusted_value is None:
            raise JudgeExecutionError(
                f"trusted judge CLI unavailable: {self.trusted_binary_name}"
            )
        trusted = Path(trusted_value).resolve(strict=True)
        if resolved != trusted or not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise JudgeExecutionError("judge executable substitution is forbidden")
        data = resolved.read_bytes()
        version_argv = [str(resolved), "--version"]
        try:
            completed = subprocess.run(
                version_argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=min(self.timeout_seconds, 30.0),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise JudgeExecutionError("judge executable version probe failed") from exc
        try:
            version_stdout = completed.stdout.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise JudgeExecutionError("judge executable version output is not UTF-8") from exc
        if completed.returncode != 0 or not version_stdout:
            raise JudgeExecutionError("judge executable version output is invalid")
        normalized_version = version_stdout.lower()
        version_identity_valid = (
            normalized_version.startswith("codex-cli ")
            if self.trusted_binary_name == "codex"
            else "claude" in normalized_version
        )
        if not version_identity_valid:
            raise JudgeExecutionError("judge executable version identity is invalid")
        self._bound_binary = str(resolved)
        class_name = f"{type(self).__module__}.{type(self).__qualname__}"
        return {
            "transport_class": class_name,
            "configured_binary": self.binary,
            "resolved_binary_realpath": str(resolved),
            "binary_bytes": len(data),
            "binary_sha256": hashlib.sha256(data).hexdigest(),
            "version_argv": version_argv,
            "version_returncode": completed.returncode,
            "version_stdout": version_stdout,
            "version_stdout_bytes": len(version_stdout.encode("utf-8")),
            "version_stdout_sha256": hashlib.sha256(
                version_stdout.encode("utf-8")
            ).hexdigest(),
            "version_stderr_bytes": len(completed.stderr),
            "version_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        }

    def _run(self, argv: list[str], prompt: bytes) -> subprocess.CompletedProcess[bytes]:
        self.preflight()
        try:
            with tempfile.TemporaryDirectory(prefix="yher-judge-isolated-") as directory:
                return subprocess.run(
                    argv,
                    input=prompt,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=directory,
                    timeout=self.timeout_seconds,
                    check=False,
                )
        except FileNotFoundError as exc:
            raise JudgeExecutionError(f"judge CLI unavailable: {self.binary}") from exc
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, bytes) else b""
            stderr = exc.stderr if isinstance(exc.stderr, bytes) else b""
            raise JudgeTransportError(
                "judge CLI timeout",
                raw_outer_response=canonical_json_bytes(
                    {
                        "schema_version": "yher.llm_sim_v2.judge_cli_failure.v1",
                        "raw_transport": _raw_cli_binding(
                            stdout=stdout, stderr=stderr, returncode=124
                        ),
                    }
                ),
                unknown_cost_reserve_yuan=_number(
                    self.unknown_cost_reserve_yuan
                ),
            ) from exc

    def _failed_process(
        self, completed: subprocess.CompletedProcess[bytes]
    ) -> JudgeTransportError:
        return JudgeTransportError(
            "judge CLI exited unsuccessfully",
            raw_outer_response=canonical_json_bytes(
                {
                    "schema_version": "yher.llm_sim_v2.judge_cli_failure.v1",
                    "raw_transport": _raw_cli_binding(
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        returncode=completed.returncode,
                    ),
                }
            ),
            unknown_cost_reserve_yuan=_number(self.unknown_cost_reserve_yuan),
        )


class CodexCLIJudgeTransport(_BaseCLIJudgeTransport):
    """Fresh noninteractive Codex process with every exposed tool family disabled."""

    name = "codex_cli"
    judge_family = "gpt"
    trusted_binary_name = "codex"

    def command_argv(self, exact_model: str) -> list[str]:
        model = _validate_exact_model(exact_model)
        return [
            self._bound_binary or self.binary,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--config",
            "mcp_servers={}",
            "--disable",
            "shell_tool",
            "--disable",
            "unified_exec",
            "--disable",
            "unified_exec_zsh_fork",
            "--disable",
            "shell_zsh_fork",
            "--disable",
            "apply_patch_freeform",
            "--disable",
            "browser_use",
            "--disable",
            "in_app_browser",
            "--disable",
            "computer_use",
            "--disable",
            "multi_agent",
            "--disable",
            "remote_plugin",
            "--disable",
            "workspace_dependencies",
            "--color",
            "never",
            "--json",
            "-",
        ]

    def invoke(self, request: bytes, *, exact_model: str, attempt_id: str) -> bytes:
        argv = self.command_argv(exact_model)
        completed = self._run(argv, _cli_prompt(request))
        if completed.returncode != 0:
            raise self._failed_process(completed)
        response = _codex_response_from_raw_cli(
            raw_transport=_raw_cli_binding(
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            ),
            exact_model=exact_model,
            attempt_id=attempt_id,
            unknown_cost_reserve_yuan=_number(self.unknown_cost_reserve_yuan),
        )
        return canonical_json_bytes(response)


class ClaudeCLIJudgeTransport(_BaseCLIJudgeTransport):
    """Fresh one-shot Claude process with an explicitly empty tool allowlist."""

    name = "claude_cli"
    judge_family = "claude"
    trusted_binary_name = "claude"

    def command_argv(self, exact_model: str) -> list[str]:
        model = _validate_exact_model(exact_model)
        return [
            self._bound_binary or self.binary,
            "-p",
            "--model",
            model,
            "--tools",
            "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--output-format",
            "json",
        ]

    def invoke(self, request: bytes, *, exact_model: str, attempt_id: str) -> bytes:
        argv = self.command_argv(exact_model)
        completed = self._run(argv, _cli_prompt(request))
        if completed.returncode != 0:
            raise self._failed_process(completed)
        response = _claude_response_from_raw_cli(
            raw_transport=_raw_cli_binding(
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            ),
            exact_model=exact_model,
            attempt_id=attempt_id,
            unknown_cost_reserve_yuan=_number(self.unknown_cost_reserve_yuan),
        )
        return canonical_json_bytes(response)


def _validate_executable_evidence(
    value: Any,
    *,
    transport_name: str,
    allow_fixture: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JudgeExecutionError("judge executable evidence is missing")
    evidence = dict(value)
    expected_fields = {
        "transport_class",
        "configured_binary",
        "resolved_binary_realpath",
        "binary_bytes",
        "binary_sha256",
        "version_argv",
        "version_returncode",
        "version_stdout",
        "version_stdout_bytes",
        "version_stdout_sha256",
        "version_stderr_bytes",
        "version_stderr_sha256",
    }
    if set(evidence) != expected_fields:
        raise JudgeExecutionError("judge executable evidence schema is invalid")
    expected_class = {
        "fixture": "experiments.llm_sim_v2.judge_execution.FixtureJudgeTransport",
        "codex_cli": "experiments.llm_sim_v2.judge_execution.CodexCLIJudgeTransport",
        "claude_cli": "experiments.llm_sim_v2.judge_execution.ClaudeCLIJudgeTransport",
    }.get(transport_name)
    allowed_classes = {expected_class}
    if transport_name != "fixture" and isinstance(expected_class, str):
        allowed_classes.add(f"__main__.{expected_class.rsplit('.', 1)[-1]}")
    if evidence.get("transport_class") not in allowed_classes:
        raise JudgeExecutionError("judge transport class evidence drifted")
    if transport_name == "fixture":
        expected = FixtureJudgeTransport([]).executable_evidence()
        if not allow_fixture or evidence != expected:
            raise JudgeExecutionError("fixture executable evidence is not admissible")
        return evidence
    trusted_name = {"codex_cli": "codex", "claude_cli": "claude"}.get(
        transport_name
    )
    trusted_value = shutil.which(str(trusted_name or ""))
    resolved_value = evidence.get("resolved_binary_realpath")
    if (
        trusted_value is None
        or not isinstance(resolved_value, str)
        or Path(resolved_value).resolve(strict=True)
        != Path(trusted_value).resolve(strict=True)
    ):
        raise JudgeExecutionError("judge executable is not the trusted installed CLI")
    resolved = Path(resolved_value)
    data = resolved.read_bytes()
    version_argv = [str(resolved), "--version"]
    try:
        completed = subprocess.run(
            version_argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30.0,
            check=False,
        )
        version_stdout = completed.stdout.decode("utf-8").strip()
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
        raise JudgeExecutionError("judge executable evidence cannot be replayed") from exc
    if (
        evidence.get("binary_bytes") != len(data)
        or evidence.get("binary_sha256") != hashlib.sha256(data).hexdigest()
        or evidence.get("version_argv") != version_argv
        or evidence.get("version_returncode") != completed.returncode
        or evidence.get("version_stdout") != version_stdout
        or evidence.get("version_stdout_bytes") != len(version_stdout.encode("utf-8"))
        or evidence.get("version_stdout_sha256")
        != hashlib.sha256(version_stdout.encode("utf-8")).hexdigest()
        or evidence.get("version_stderr_bytes") != len(completed.stderr)
        or evidence.get("version_stderr_sha256")
        != hashlib.sha256(completed.stderr).hexdigest()
    ):
        raise JudgeExecutionError("judge executable bytes or version evidence drifted")
    return evidence


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise JudgeExecutionError(f"immutable artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _judge_root(value: str | Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_symlink():
        raise JudgeExecutionError("judge output root cannot be a symlink")
    return supplied.resolve(strict=False)


@contextmanager
def _judge_run_lock(root: Path):
    """Serialize every mutation of one judge run across threads and processes."""

    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.judge-run.lock"
    if lock_path.is_symlink():
        raise JudgeExecutionError("judge run lock cannot be a symlink")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    with _RUN_THREAD_LOCK:
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _serialized_judge_run_mutation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        root = _judge_root(kwargs["output_root"])
        with _judge_run_lock(root):
            return function(*args, **kwargs)

    return wrapped


def _case_manifest_path(root: Path) -> Path:
    return root / "case_manifest.json"


def _bind_case_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
    path = _case_manifest_path(root)
    payload = canonical_json_bytes(manifest) + b"\n"
    root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise JudgeExecutionError(
                "judge root case manifest differs from the frozen run binding"
            )
        return path
    _write_new(path, payload)
    return path


def _committed_main_phase_anchor(repo_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = Path(repo_root).expanduser()
    if supplied.is_symlink():
        raise JudgeExecutionError("formal collection repository root cannot be a symlink")
    try:
        repo = supplied.resolve(strict=True)
        top = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=repo,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        ).resolve(strict=True)
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        committed = subprocess.check_output(
            ["git", "show", f"{head}:{MAIN_PHASE_ANCHOR_PATH}"],
            cwd=repo,
            stderr=subprocess.STDOUT,
        )
        anchor_commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", MAIN_PHASE_ANCHOR_PATH],
            cwd=repo,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", anchor_commit, head],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise JudgeExecutionError(
            "formal main phase evidence anchor is not committed"
        ) from exc
    anchor = repo / MAIN_PHASE_ANCHOR_PATH
    if (
        top != repo
        or not anchor_commit
        or ancestry.returncode != 0
        or anchor.is_symlink()
        or not anchor.is_file()
        or anchor.read_bytes() != committed
    ):
        raise JudgeExecutionError(
            "formal main phase evidence anchor differs from committed Git bytes"
        )
    value = _strict_json_bytes(committed, label="committed main phase evidence receipt")
    payload = dict(value)
    advertised = payload.pop("phase_evidence_receipt_sha256", None)
    if (
        value.get("schema_version")
        != "yher.llm_sim_v2.phase_evidence_receipt.v1"
        or value.get("simulated") is not True
        or value.get("run_id") != RUN_ID
        or value.get("phase") != "main"
        or value.get("authority") != "post_invocation_phase_receipt"
        or not isinstance(value.get("store_snapshot"), Mapping)
        or not isinstance(value.get("providers"), Mapping)
        or advertised != canonical_sha256(payload)
    ):
        raise JudgeExecutionError("committed main phase evidence receipt is invalid")
    proof = {
        "repository_root_realpath": str(repo),
        "anchor_relative_path": MAIN_PHASE_ANCHOR_PATH,
        "anchor_sha256": hashlib.sha256(committed).hexdigest(),
        "anchor_bytes": len(committed),
        "phase_evidence_receipt_sha256": advertised,
        "head_commit_at_mint": head,
        "anchor_commit": anchor_commit,
        "anchor_is_ancestor_of_head": True,
        "working_tree_matches_head_blob": True,
    }
    return value, proof


def _run_budget_source(path_value: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = Path(path_value).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise JudgeExecutionError("formal run budget ledger must be a regular file")
    path = supplied.resolve(strict=True)
    data = path.read_bytes()
    value = _strict_json_bytes(data, label="formal run budget ledger")
    known = _decimal(value.get("total_known_cost_yuan"), label="budget total known cost")
    reserve = _decimal(
        value.get("total_unknown_reserve_yuan"),
        label="budget total unknown reserve",
    )
    total = _decimal(
        value.get("total_accounted_cost_yuan"), label="budget total accounted cost"
    )
    hard = _decimal(value.get("hard_fuse_yuan"), label="budget hard fuse")
    assert known is not None and reserve is not None and total is not None and hard is not None
    if (
        value.get("schema_version") != "yher.llm_sim_v2.run_budget_ledger.v2"
        or value.get("simulated") is not True
        or value.get("run_id") != RUN_ID
        or known + reserve != total
        or hard != HARD_FUSE_YUAN
        or total >= hard
    ):
        raise JudgeExecutionError("formal run budget ledger identity or totals are invalid")
    return value, {
        "source_path_realpath": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "total_known_cost_yuan": _number(known),
        "total_unknown_reserve_yuan": _number(reserve),
        "total_accounted_cost_yuan": _number(total),
    }


@_serialized_judge_run_mutation
def bind_prepared_judge_case_manifest(
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
) -> Path:
    """Install the deterministic pre-adjudication case manifest into a new run."""

    manifest, _ = _validate_case_manifest(
        _load_object(case_manifest, label="prepared judge case manifest")
    )
    root = _judge_root(output_root)
    if root.exists():
        raise JudgeExecutionError("prepared judge run root already exists")
    return _bind_case_manifest(root, manifest)


@_serialized_judge_run_mutation
def mint_judge_budget_authority(
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    run_budget_ledger: str | Path,
) -> Path:
    """Mint the one immutable judge budget authority from formal collection bytes."""

    manifest, _ = _validate_case_manifest(
        _load_object(case_manifest, label="judge case manifest")
    )
    _phase_receipt, formal_collection = _committed_main_phase_anchor(repo_root)
    _ledger, ledger_binding = _run_budget_source(run_budget_ledger)
    root = _judge_root(output_root)
    bound_case_path = _case_manifest_path(root)
    if (
        bound_case_path.is_symlink()
        or not bound_case_path.is_file()
        or _load_object(bound_case_path, label="prepared judge case manifest")
        != manifest
        or {path.name for path in root.iterdir()} != {"case_manifest.json"}
    ):
        raise JudgeExecutionError(
            "judge budget authority requires the sole prepared case manifest"
        )
    authority: dict[str, Any] = {
        "schema_version": BUDGET_AUTHORITY_SCHEMA,
        "simulated": True,
        "run_id": RUN_ID,
        "authority_kind": "formal_collection_committed_anchor_and_live_cost_ledger",
        "case_manifest_sha256": manifest["case_manifest_sha256"],
        "formal_collection": formal_collection,
        "run_budget_ledger": ledger_binding,
        "baseline_accounted_cost_yuan": ledger_binding[
            "total_accounted_cost_yuan"
        ],
        "unknown_reserve_per_attempt_yuan": _number(
            PRODUCTION_UNKNOWN_RESERVE_YUAN
        ),
        "hard_fuse_yuan": _number(HARD_FUSE_YUAN),
        "minted_at_utc": _utc_now(),
    }
    authority["judge_budget_authority_sha256"] = canonical_sha256(authority)
    path = root / "budget_authority.json"
    _write_new(path, canonical_json_bytes(authority) + b"\n")
    return path


def _fixture_budget_authority(
    root: Path,
    *,
    manifest: Mapping[str, Any],
) -> Path:
    path = root / "budget_authority.json"
    if path.exists():
        return path
    authority: dict[str, Any] = {
        "schema_version": BUDGET_AUTHORITY_SCHEMA,
        "simulated": True,
        "run_id": RUN_ID,
        "authority_kind": "test_fixture",
        "case_manifest_sha256": manifest["case_manifest_sha256"],
        "formal_collection": None,
        "run_budget_ledger": None,
        "baseline_accounted_cost_yuan": 0,
        "unknown_reserve_per_attempt_yuan": _number(
            PRODUCTION_UNKNOWN_RESERVE_YUAN
        ),
        "hard_fuse_yuan": _number(HARD_FUSE_YUAN),
        "minted_at_utc": _utc_now(),
    }
    authority["judge_budget_authority_sha256"] = canonical_sha256(authority)
    _write_new(path, canonical_json_bytes(authority) + b"\n")
    return path


def load_judge_budget_authority(
    output_root: str | Path,
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    manifest, _ = _validate_case_manifest(
        _load_object(case_manifest, label="judge case manifest")
    )
    root = _judge_root(output_root)
    path = root / "budget_authority.json"
    value = _load_object(path, label="judge budget authority")
    expected_fields = {
        "schema_version",
        "simulated",
        "run_id",
        "authority_kind",
        "case_manifest_sha256",
        "formal_collection",
        "run_budget_ledger",
        "baseline_accounted_cost_yuan",
        "unknown_reserve_per_attempt_yuan",
        "hard_fuse_yuan",
        "minted_at_utc",
        "judge_budget_authority_sha256",
    }
    _self_hash(value, "judge_budget_authority_sha256", label="judge budget authority")
    baseline = _decimal(
        value.get("baseline_accounted_cost_yuan"), label="judge budget baseline"
    )
    reserve = _decimal(
        value.get("unknown_reserve_per_attempt_yuan"),
        label="judge unknown reserve per attempt",
    )
    hard = _decimal(value.get("hard_fuse_yuan"), label="judge hard fuse")
    assert baseline is not None and reserve is not None and hard is not None
    if (
        set(value) != expected_fields
        or value.get("schema_version") != BUDGET_AUTHORITY_SCHEMA
        or value.get("simulated") is not True
        or value.get("run_id") != RUN_ID
        or value.get("case_manifest_sha256") != manifest["case_manifest_sha256"]
        or reserve != PRODUCTION_UNKNOWN_RESERVE_YUAN
        or hard != HARD_FUSE_YUAN
        or baseline >= hard
    ):
        raise JudgeExecutionError("judge budget authority identity is invalid")
    _utc_timestamp(value.get("minted_at_utc"), label="judge budget mint timestamp")
    kind = value.get("authority_kind")
    if kind == "test_fixture":
        if not allow_fixture or value.get("formal_collection") is not None or value.get(
            "run_budget_ledger"
        ) is not None or baseline != 0:
            raise JudgeExecutionError("fixture judge budget authority is not production eligible")
        return value
    if kind != "formal_collection_committed_anchor_and_live_cost_ledger":
        raise JudgeExecutionError("judge budget authority source is invalid")
    formal = value.get("formal_collection")
    ledger = value.get("run_budget_ledger")
    if not isinstance(formal, Mapping) or not isinstance(ledger, Mapping):
        raise JudgeExecutionError("judge budget authority source binding is incomplete")
    _receipt, current_formal = _committed_main_phase_anchor(
        str(formal.get("repository_root_realpath") or "")
    )
    current_head = current_formal.pop("head_commit_at_mint")
    stored_formal = dict(formal)
    stored_head = stored_formal.pop("head_commit_at_mint", None)
    if (
        current_formal != stored_formal
        or not isinstance(stored_head, str)
        or not stored_head
        or not isinstance(current_head, str)
    ):
        raise JudgeExecutionError("committed formal collection authority changed")
    _ledger_value, current_ledger = _run_budget_source(
        str(ledger.get("source_path_realpath") or "")
    )
    if current_ledger != dict(ledger) or current_ledger[
        "total_accounted_cost_yuan"
    ] != value["baseline_accounted_cost_yuan"]:
        raise JudgeExecutionError("formal run budget ledger source changed after mint")
    return value


def _budget_receipt_binding(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "judge_budget_authority_sha256": authority[
            "judge_budget_authority_sha256"
        ],
        "authority_kind": authority["authority_kind"],
        "baseline_accounted_cost_yuan": authority[
            "baseline_accounted_cost_yuan"
        ],
        "unknown_reserve_per_attempt_yuan": authority[
            "unknown_reserve_per_attempt_yuan"
        ],
        "hard_fuse_yuan": authority["hard_fuse_yuan"],
    }


def _terminal_execution_receipts(root: Path) -> list[Path]:
    executions = root / "executions"
    if not executions.exists():
        return []
    if executions.is_symlink() or not executions.is_dir():
        raise JudgeExecutionError("judge executions root is unsafe")
    receipts: list[Path] = []
    for family_root in sorted(executions.iterdir()):
        if (
            family_root.name not in JUDGE_FAMILIES
            or family_root.is_symlink()
            or not family_root.is_dir()
        ):
            raise JudgeExecutionError("judge executions contain an unexpected family")
        children = sorted(family_root.iterdir())
        if len(children) != 1 or children[0].is_symlink() or not children[0].is_dir():
            raise JudgeExecutionError("judge family contains an unsealed execution root")
        execution_root = children[0]
        candidates = [
            execution_root / "execution_receipt.json",
            execution_root / "failed_execution_receipt.json",
        ]
        terminal = [path for path in candidates if path.is_file() and not path.is_symlink()]
        if len(terminal) != 1:
            raise JudgeExecutionError("judge family contains an unsealed execution root")
        receipts.append(terminal[0])
    return receipts


def _accrued_judge_accounting(root: Path) -> Decimal:
    total = Decimal("0")
    for receipt_path in _terminal_execution_receipts(root):
        value = _load_object(receipt_path, label="prior judge execution receipt")
        accounting = value.get("accounting")
        if not isinstance(accounting, Mapping):
            raise JudgeExecutionError("prior judge accounting is invalid")
        amount = _decimal(
            accounting.get("accounted_cost_yuan"), label="prior judge accounted cost"
        )
        assert amount is not None
        total += amount
    return total


def _family_disposition_path(root: Path, judge_family: str) -> Path:
    return root / "family_dispositions" / f"{judge_family}.json"


def _assert_family_slot_available(root: Path, judge_family: str) -> None:
    if (root / "judge_run_evidence_receipt.json").exists():
        raise JudgeExecutionError("judge run is finalized and every family slot is sealed")
    if (
        (root / "executions" / judge_family).exists()
        or _family_disposition_path(root, judge_family).exists()
    ):
        raise JudgeExecutionError(
            f"judge family slot already has its single allowed disposition: {judge_family}"
        )


@_serialized_judge_run_mutation
def record_judge_family_disposition(
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    judge_family: str,
    status: str,
    reason_code: str,
) -> Path:
    """Seal one family without an invocation as unavailable or not applicable."""

    manifest, _ = _validate_case_manifest(
        _load_object(case_manifest, label="judge case manifest")
    )
    if judge_family not in JUDGE_FAMILIES:
        raise JudgeExecutionError("judge family must be exactly claude or gpt")
    if status not in {"unavailable", "not_applicable_zero_cases"}:
        raise JudgeExecutionError("judge family disposition status is invalid")
    if not isinstance(reason_code, str) or not re.fullmatch(
        r"[a-z][a-z0-9_]{2,79}", reason_code
    ):
        raise JudgeExecutionError("judge family disposition reason code is invalid")
    case_count = len(manifest["cases"])
    if (status == "not_applicable_zero_cases") != (case_count == 0):
        raise JudgeExecutionError(
            "zero-case judge disposition must match the frozen selected case count"
    )
    root = _judge_root(output_root)
    _accrued_judge_accounting(root)
    _assert_next_family_order(root, manifest=manifest, judge_family=judge_family)
    _assert_family_slot_available(root, judge_family)
    _bind_case_manifest(root, manifest)
    receipt: dict[str, Any] = {
        "schema_version": FAMILY_DISPOSITION_SCHEMA,
        "simulated": True,
        "run_id": RUN_ID,
        "judge_family": judge_family,
        "status": status,
        "reason_code": reason_code,
        "case_manifest_sha256": manifest["case_manifest_sha256"],
        "case_count": case_count,
        "created_at_utc": _utc_now(),
        "accounting": {
            "request_count": 0,
            "known_cost_yuan": 0,
            "unknown_cost_reserve_yuan": 0,
            "accounted_cost_yuan": 0,
        },
    }
    receipt["family_disposition_receipt_sha256"] = canonical_sha256(receipt)
    path = _family_disposition_path(root, judge_family)
    _write_new(path, canonical_json_bytes(receipt) + b"\n")
    return path


def _load_family_disposition(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    judge_family: str,
) -> dict[str, Any]:
    value = _load_object(path, label=f"{judge_family} judge family disposition")
    expected_fields = {
        "schema_version",
        "simulated",
        "run_id",
        "judge_family",
        "status",
        "reason_code",
        "case_manifest_sha256",
        "case_count",
        "created_at_utc",
        "accounting",
        "family_disposition_receipt_sha256",
    }
    _self_hash(
        value,
        "family_disposition_receipt_sha256",
        label="judge family disposition",
    )
    case_count = len(manifest["cases"])
    status = value.get("status")
    expected_accounting = {
        "request_count": 0,
        "known_cost_yuan": 0,
        "unknown_cost_reserve_yuan": 0,
        "accounted_cost_yuan": 0,
    }
    if (
        set(value) != expected_fields
        or value.get("schema_version") != FAMILY_DISPOSITION_SCHEMA
        or value.get("simulated") is not True
        or value.get("run_id") != RUN_ID
        or value.get("judge_family") != judge_family
        or status not in {"unavailable", "not_applicable_zero_cases"}
        or (status == "not_applicable_zero_cases") != (case_count == 0)
        or value.get("case_manifest_sha256") != manifest["case_manifest_sha256"]
        or value.get("case_count") != case_count
        or value.get("accounting") != expected_accounting
        or not isinstance(value.get("reason_code"), str)
        or not re.fullmatch(r"[a-z][a-z0-9_]{2,79}", value["reason_code"])
    ):
        raise JudgeExecutionError("judge family disposition receipt is invalid")
    _utc_timestamp(value.get("created_at_utc"), label="judge disposition timestamp")
    return value


def _family_terminal_times(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    judge_family: str,
) -> tuple[datetime, datetime] | None:
    disposition_path = _family_disposition_path(root, judge_family)
    family_root = root / "executions" / judge_family
    if disposition_path.exists() and family_root.exists():
        raise JudgeExecutionError("judge family has both execution and disposition evidence")
    if disposition_path.exists():
        disposition = _load_family_disposition(
            disposition_path,
            manifest=manifest,
            judge_family=judge_family,
        )
        created = _utc_timestamp(
            disposition["created_at_utc"], label="judge disposition timestamp"
        )
        return created, created
    if not family_root.exists():
        return None
    receipts = [
        path
        for path in _terminal_execution_receipts(root)
        if path.parents[1].name == judge_family
    ]
    if len(receipts) != 1:
        raise JudgeExecutionError("judge family does not have one terminal execution")
    receipt = _load_object(receipts[0], label=f"{judge_family} terminal receipt")
    identity = receipt.get("identity")
    expected_schema = (
        RECEIPT_SCHEMA
        if receipts[0].name == "execution_receipt.json"
        else FAILED_RECEIPT_SCHEMA
    )
    expected_status = (
        "complete" if receipts[0].name == "execution_receipt.json" else "failed"
    )
    if (
        receipt.get("schema_version") != expected_schema
        or receipt.get("status") != expected_status
        or not isinstance(identity, Mapping)
        or identity.get("judge_family") != judge_family
    ):
        raise JudgeExecutionError("judge family terminal receipt identity is invalid")
    started = _utc_timestamp(
        receipt.get("started_at_utc"), label="judge execution start timestamp"
    )
    completed = _utc_timestamp(
        receipt.get("completed_at_utc"), label="judge execution completion timestamp"
    )
    if completed < started:
        raise JudgeExecutionError("judge execution timestamp order is invalid")
    return started, completed


def _assert_next_family_order(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    judge_family: str,
) -> None:
    if not manifest["cases"]:
        return
    gpt_times = _family_terminal_times(root, manifest=manifest, judge_family="gpt")
    claude_times = _family_terminal_times(
        root, manifest=manifest, judge_family="claude"
    )
    if judge_family == "gpt" and claude_times is not None:
        raise JudgeExecutionError("GPT must be sealed before any Claude family action")
    if judge_family == "claude" and gpt_times is None:
        raise JudgeExecutionError("GPT must be terminal before Claude starts")


def _assert_final_family_order(
    root: Path,
    *,
    manifest: Mapping[str, Any],
) -> None:
    if not manifest["cases"]:
        return
    gpt_times = _family_terminal_times(root, manifest=manifest, judge_family="gpt")
    claude_times = _family_terminal_times(
        root, manifest=manifest, judge_family="claude"
    )
    if gpt_times is None or claude_times is None or gpt_times[1] > claude_times[0]:
        raise JudgeExecutionError("final judge evidence does not prove GPT-before-Claude")


def _tree_binding(root: Path) -> dict[str, Any]:
    excluded = {root / "judge_run_evidence_receipt.json"}
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    if root.is_dir():
        for current_value, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(current_value)
            if current != root:
                directories.append(current.relative_to(root).as_posix())
            for name in directory_names:
                if (current / name).is_symlink():
                    raise JudgeExecutionError("judge run tree contains a symlink")
            for name in file_names:
                path = current / name
                if path in excluded:
                    continue
                if path.is_symlink() or not path.is_file():
                    raise JudgeExecutionError("judge run tree contains an unsafe entry")
                data = path.read_bytes()
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
    files.sort(key=lambda row: row["path"])
    directories.sort()
    return {
        "files": files,
        "directories": directories,
        "file_count": len(files),
        "file_set_sha256": canonical_sha256(files),
        "directory_set_sha256": canonical_sha256(directories),
    }


def _family_slot_binding(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    judge_family: str,
    budget_binding: Mapping[str, Any],
    allow_fixture: bool,
) -> dict[str, Any]:
    disposition_path = _family_disposition_path(root, judge_family)
    family_root = root / "executions" / judge_family
    if disposition_path.exists() and family_root.exists():
        raise JudgeExecutionError("judge family has both execution and disposition evidence")
    if disposition_path.exists():
        disposition = _load_family_disposition(
            disposition_path, manifest=manifest, judge_family=judge_family
        )
        return {
            "status": disposition["status"],
            "execution_id": None,
            "requested_model": None,
            "transport": None,
            "transport_reported_models": [],
            "receipt_path": disposition_path.relative_to(root).as_posix(),
            "receipt_sha256": disposition[
                "family_disposition_receipt_sha256"
            ],
            "accounting": disposition["accounting"],
        }
    if not family_root.is_dir() or family_root.is_symlink():
        raise JudgeExecutionError(f"judge family slot is unresolved: {judge_family}")
    children = sorted(family_root.iterdir())
    if len(children) != 1 or children[0].is_symlink() or not children[0].is_dir():
        raise JudgeExecutionError("judge family must contain exactly one execution")
    execution_root = children[0]
    complete_path = execution_root / "execution_receipt.json"
    failed_path = execution_root / "failed_execution_receipt.json"
    if complete_path.is_file() == failed_path.is_file():
        raise JudgeExecutionError("judge execution receipt status is ambiguous")
    if complete_path.is_file():
        receipt_path = complete_path
        receipt = validate_execution_receipt(
            receipt_path,
            manifest,
            judge_family,
            allow_fixture=allow_fixture,
        )
        status = "complete"
        digest = receipt["execution_receipt_sha256"]
    else:
        receipt_path = failed_path
        receipt = validate_failed_execution_receipt(
            receipt_path,
            manifest,
            judge_family,
            allow_fixture=allow_fixture,
        )
        status = "failed"
        digest = receipt["failed_execution_receipt_sha256"]
    if receipt.get("budget_authority") != dict(budget_binding):
        raise JudgeExecutionError(
            "judge execution budget authority differs from finalized run authority"
        )
    identity = receipt["identity"]
    return {
        "status": status,
        "execution_id": identity["execution_id"],
        "requested_model": identity["requested_model"],
        "transport": identity["transport"],
        "transport_reported_models": list(identity["transport_reported_models"]),
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": digest,
        "accounting": receipt["accounting"],
    }


def build_judge_run_evidence_receipt(
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    manifest, input_bytes = _validate_case_manifest(
        _load_object(case_manifest, label="judge case manifest")
    )
    root = _judge_root(output_root)
    bound_case_path = _case_manifest_path(root)
    if (
        not bound_case_path.is_file()
        or bound_case_path.is_symlink()
        or _load_object(bound_case_path, label="bound judge case manifest") != manifest
    ):
        raise JudgeExecutionError("judge run lacks its exact case manifest bytes")
    if allow_fixture and not (root / "budget_authority.json").exists():
        _fixture_budget_authority(root, manifest=manifest)
    authority = load_judge_budget_authority(
        root,
        case_manifest=manifest,
        allow_fixture=allow_fixture,
    )
    budget_binding = _budget_receipt_binding(authority)
    slots = {
        family: _family_slot_binding(
            root,
            manifest=manifest,
            judge_family=family,
            budget_binding=budget_binding,
            allow_fixture=allow_fixture,
        )
        for family in ("claude", "gpt")
    }
    _assert_final_family_order(root, manifest=manifest)
    baseline = _decimal(
        authority["baseline_accounted_cost_yuan"], label="judge budget baseline"
    )
    hard_fuse = _decimal(authority["hard_fuse_yuan"], label="judge hard fuse")
    assert baseline is not None and hard_fuse is not None
    finalized_judge_cost = Decimal("0")
    for family, slot in slots.items():
        accounting = slot.get("accounting")
        if not isinstance(accounting, Mapping):
            raise JudgeExecutionError(f"{family} judge accounting is invalid")
        amount = _decimal(
            accounting.get("accounted_cost_yuan"),
            label=f"{family} finalized judge cost",
        )
        assert amount is not None
        finalized_judge_cost += amount
    if baseline + finalized_judge_cost >= hard_fuse:
        raise JudgeExecutionError("finalized judge run reaches the CNY 450 hard fuse")
    receipt: dict[str, Any] = {
        "schema_version": RUN_EVIDENCE_RECEIPT_SCHEMA,
        "simulated": True,
        "run_id": RUN_ID,
        "status": "finalized",
        "case_binding": {
            "case_manifest_sha256": manifest["case_manifest_sha256"],
            "shared_input_sha256": manifest["shared_input_sha256"],
            "input_bytes": len(input_bytes),
            "case_count": len(manifest["cases"]),
        },
        "budget_authority": budget_binding,
        "family_slots": slots,
        "artifact_tree": _tree_binding(root),
    }
    receipt["judge_run_evidence_receipt_sha256"] = canonical_sha256(receipt)
    return receipt


@_serialized_judge_run_mutation
def write_judge_run_evidence_receipt(
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    output: str | Path | None = None,
    allow_fixture: bool = False,
) -> Path:
    root = _judge_root(output_root)
    receipt = build_judge_run_evidence_receipt(
        case_manifest=case_manifest,
        output_root=root,
        allow_fixture=allow_fixture,
    )
    destination = (
        root / "judge_run_evidence_receipt.json"
        if output is None
        else Path(output).expanduser().resolve(strict=False)
    )
    if output is not None and not destination.as_posix().endswith(
        "/" + JUDGE_RUN_ANCHOR_PATH
    ):
        raise JudgeExecutionError("external judge run receipt must use the fixed anchor path")
    payload = canonical_json_bytes(receipt) + b"\n"
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
            raise JudgeExecutionError("immutable judge run evidence receipt already differs")
        return destination
    _write_new(destination, payload)
    return destination


def validate_judge_run_evidence_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    value = _load_object(receipt, label="judge run evidence receipt")
    _self_hash(
        value,
        "judge_run_evidence_receipt_sha256",
        label="judge run evidence receipt",
    )
    rebuilt = build_judge_run_evidence_receipt(
        case_manifest=case_manifest,
        output_root=output_root,
        allow_fixture=allow_fixture,
    )
    if value != rebuilt:
        raise JudgeExecutionError("judge run evidence receipt differs from exact tree")
    return value


def _attempt_artifact(
    *,
    attempt_id: str,
    batch_id: str,
    request: bytes,
    response: bytes,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "yher.llm_sim_v2.judge_raw_attempt.v1",
            "attempt_id": attempt_id,
            "batch_id": batch_id,
            "request_base64": base64.b64encode(request).decode("ascii"),
            "request_sha256": hashlib.sha256(request).hexdigest(),
            "raw_outer_response_base64": base64.b64encode(response).decode("ascii"),
            "raw_outer_response_sha256": hashlib.sha256(response).hexdigest(),
        }
    ) + b"\n"


def _transport_error_response(error: JudgeTransportError) -> bytes:
    known = _decimal(
        error.known_cost_yuan,
        label="transport-error known judge cost",
        allow_null=True,
    )
    reserve = _decimal(
        error.unknown_cost_reserve_yuan,
        label="transport-error unknown judge reserve",
    )
    assert reserve is not None
    source = error.raw_outer_response or b""
    return canonical_json_bytes(
        {
            "schema_version": "yher.llm_sim_v2.judge_transport_error.v1",
            "error_type": type(error).__name__,
            "message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            "raw_failure_base64": base64.b64encode(source).decode("ascii"),
            "raw_failure_bytes": len(source),
            "raw_failure_sha256": hashlib.sha256(source).hexdigest(),
            "usage": {
                "input_tokens": error.input_tokens,
                "output_tokens": error.output_tokens,
            },
            "billing": {
                "known_cost_yuan": None if known is None else _number(known),
                "unknown_cost_reserve_yuan": _number(reserve),
            },
        }
    )


def _execution_error_response(error: BaseException) -> bytes:
    error_type = f"{type(error).__module__}.{type(error).__qualname__}"
    message_sha256 = hashlib.sha256(str(error).encode("utf-8")).hexdigest()
    return canonical_json_bytes(
        {
            "schema_version": "yher.llm_sim_v2.judge_execution_error.v1",
            "error_type": error_type,
            "message_sha256": message_sha256,
            "call_may_have_begun": True,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "billing": {
                "known_cost_yuan": None,
                "unknown_cost_reserve_yuan": _number(
                    PRODUCTION_UNKNOWN_RESERVE_YUAN
                ),
            },
        }
    )


def _batch_request_bytes(
    *,
    manifest: Mapping[str, Any],
    batch: Sequence[Mapping[str, Any]],
    batch_index: int,
    judge_family: str,
    exact_model: str,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "yher.llm_sim_v2.judge_batch_request.v1",
            "simulated": True,
            "judge_family": judge_family,
            "requested_model": exact_model,
            "case_manifest_sha256": manifest["case_manifest_sha256"],
            "shared_input_sha256": manifest["shared_input_sha256"],
            "judge_protocol_sha256": manifest["judge_protocol_sha256"],
            "judge_amendment_sha256": manifest["judge_amendment"]["sha256"],
            "batch_id": f"batch-{batch_index:04d}",
            "cases": [
                {"case_id": case["case_id"], "messages": case["judge_messages"]}
                for case in batch
            ],
            "required_output_fields": REQUIRED_OUTPUT_FIELDS,
        }
    )


@_serialized_judge_run_mutation
def execute_judge_pass(
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    judge_family: str,
    exact_model: str,
    transport: Any,
) -> Path:
    """Execute one family only after run-wide ordering and residue checks."""

    manifest, _ = _validate_case_manifest(
        _load_object(case_manifest, label="judge case manifest")
    )
    if not manifest["cases"]:
        raise JudgeExecutionError(
            "zero-case judge runs require not_applicable_zero_cases dispositions"
        )
    root = _judge_root(output_root)
    _accrued_judge_accounting(root)
    _assert_next_family_order(root, manifest=manifest, judge_family=judge_family)
    return _execute_judge_pass_locked(
        case_manifest=manifest,
        output_root=root,
        judge_family=judge_family,
        exact_model=exact_model,
        transport=transport,
    )


def _execute_judge_pass_locked(
    *,
    case_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    judge_family: str,
    exact_model: str,
    transport: Any,
) -> Path:
    """Execute one fresh isolated pass and return its immutable receipt path."""

    manifest, input_bytes = _validate_case_manifest(
        _load_object(case_manifest, label="judge case manifest")
    )
    if judge_family not in JUDGE_FAMILIES:
        raise JudgeExecutionError("judge family must be exactly claude or gpt")
    _validate_exact_model(exact_model)
    root = _judge_root(output_root)
    _assert_family_slot_available(root, judge_family)
    if (
        getattr(transport, "tools_disabled", None) is not True
        or getattr(transport, "fresh_execution", None) is not True
        or not isinstance(getattr(transport, "name", None), str)
    ):
        raise JudgeExecutionError("judge transport cannot prove fresh tool-free isolation")
    expected_transport_type = {
        "fixture": FixtureJudgeTransport,
        "codex_cli": CodexCLIJudgeTransport,
        "claude_cli": ClaudeCLIJudgeTransport,
    }.get(getattr(transport, "name", None))
    if expected_transport_type is None or type(transport) is not expected_transport_type:
        raise JudgeExecutionError("judge transport class substitution is forbidden")
    transport_family = getattr(transport, "judge_family", None)
    if transport_family is not None and transport_family != judge_family:
        raise JudgeExecutionError("judge transport family does not match its frozen slot")
    _bind_case_manifest(root, manifest)
    if transport.name == "fixture" and not (root / "budget_authority.json").exists():
        _fixture_budget_authority(root, manifest=manifest)
    authority = load_judge_budget_authority(
        root,
        case_manifest=manifest,
        allow_fixture=transport.name == "fixture",
    )
    budget_binding = _budget_receipt_binding(authority)
    baseline_cost = _decimal(
        authority["baseline_accounted_cost_yuan"], label="judge budget baseline"
    )
    next_unknown_reserve = _decimal(
        authority["unknown_reserve_per_attempt_yuan"],
        label="judge next-attempt reserve",
    )
    hard_fuse = _decimal(authority["hard_fuse_yuan"], label="judge hard fuse")
    assert baseline_cost is not None and next_unknown_reserve is not None and hard_fuse is not None
    preexisting_judge_cost = _accrued_judge_accounting(root)
    executable_evidence = transport.executable_evidence()
    command = transport.command_argv(exact_model)
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) for part in command)
    ):
        raise JudgeExecutionError("judge transport command is not explicit")
    empty_indexes = [index for index, part in enumerate(command) if part == ""]
    allowed_empty_indexes: list[int] = []
    if transport.name == "claude_cli" and command.count("--tools") == 1:
        allowed_empty_indexes = [command.index("--tools") + 1]
    if empty_indexes != allowed_empty_indexes:
        raise JudgeExecutionError("judge transport command contains an empty argument")
    strict_configuration_verified = (
        transport.name == "codex_cli"
        and all(
            flag in command
            for flag in (
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--strict-config",
            )
        )
        or transport.name == "claude_cli"
        and all(
            flag in command
            for flag in (
                "--no-session-persistence",
                "--strict-mcp-config",
                "--disable-slash-commands",
            )
        )
        and command[command.index("--tools") + 1] == ""
    )
    execution_id = str(uuid.uuid4())
    executions_root = root / "executions"
    executions_root.mkdir(parents=True, exist_ok=True)
    family_root = executions_root / judge_family
    try:
        family_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise JudgeExecutionError("judge family already has its single execution") from exc
    execution_root = family_root / execution_id
    execution_root.mkdir(mode=0o700)
    started_at = _utc_now()
    cases = manifest["cases"]
    protocol = manifest["judge_protocol"]
    category_policy = protocol["label_category_policy"]
    attempts: list[dict[str, Any]] = []
    raw_artifacts: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    input_tokens = 0
    output_tokens = 0
    known_cost = Decimal("0")
    reserve_cost = Decimal("0")
    request_count = 0
    retry_count = 0
    schema_error_count = 0
    transport_error_count = 0
    terminal_failure: dict[str, Any] | None = None
    for batch_index, offset in enumerate(range(0, len(cases), BATCH_SIZE)):
        batch = cases[offset : offset + BATCH_SIZE]
        batch_id = f"batch-{batch_index:04d}"
        case_ids = [str(case["case_id"]) for case in batch]
        request = _batch_request_bytes(
            manifest=manifest,
            batch=batch,
            batch_index=batch_index,
            judge_family=judge_family,
            exact_model=exact_model,
        )
        for attempt_number in range(1, MAX_ATTEMPTS_PER_BATCH + 1):
            projected_cost = (
                baseline_cost
                + preexisting_judge_cost
                + known_cost
                + reserve_cost
                + next_unknown_reserve
            )
            if projected_cost >= hard_fuse:
                terminal_failure = {
                    "reason": "budget_fuse_blocked",
                    "terminal_status": "fuse_blocked",
                    "error_type": "JudgeBudgetFuseError",
                    "error_message_sha256": hashlib.sha256(
                        b"judge budget would reach the frozen hard fuse"
                    ).hexdigest(),
                }
                break
            attempt_id = str(uuid.uuid4())
            attempt_started = _utc_now()
            request_count += 1
            if attempt_number > 1:
                retry_count += 1
            status = "success"
            batch_rows: list[dict[str, Any]] = []
            response: dict[str, Any] | None = None
            transport_error: JudgeTransportError | None = None
            execution_exception: BaseException | None = None
            isolation_error: JudgeIsolationError | None = None
            schema_exception: JudgeExecutionError | None = None
            try:
                raw_response = transport.invoke(
                    request, exact_model=exact_model, attempt_id=attempt_id
                )
                if not isinstance(raw_response, bytes):
                    raise JudgeExecutionError("transport must return raw bytes")
            except JudgeTransportError as exc:
                status = "transport_error"
                transport_error = exc
                transport_error_count += 1
                raw_response = _transport_error_response(exc)
            except BaseException as exc:
                status = "execution_error"
                execution_exception = exc
                raw_response = _execution_error_response(exc)
            if execution_exception is not None:
                attempt_input_tokens = 0
                attempt_output_tokens = 0
                attempt_known = None
                attempt_reserve = PRODUCTION_UNKNOWN_RESERVE_YUAN
                outer = None
            elif transport_error is None:
                (
                    attempt_input_tokens,
                    attempt_output_tokens,
                    attempt_known,
                    attempt_reserve,
                    outer,
                ) = _response_accounting(raw_response)
                try:
                    response, batch_rows = _validate_transport_response(
                        raw_response,
                        case_ids=case_ids,
                        exact_model=exact_model,
                        category_policy=category_policy,
                    )
                except JudgeIsolationError as exc:
                    status = "isolation_error"
                    isolation_error = exc
                except JudgeExecutionError as exc:
                    status = "schema_error"
                    schema_exception = exc
                    schema_error_count += 1
            else:
                attempt_input_tokens = transport_error.input_tokens
                attempt_output_tokens = transport_error.output_tokens
                attempt_known = _decimal(
                    transport_error.known_cost_yuan,
                    label="transport-error known judge cost",
                    allow_null=True,
                )
                attempt_reserve = _decimal(
                    transport_error.unknown_cost_reserve_yuan,
                    label="transport-error unknown judge reserve",
                )
                assert attempt_reserve is not None
                outer = None
            if transport.name in {"codex_cli", "claude_cli"} and attempt_known is None:
                attempt_reserve = PRODUCTION_UNKNOWN_RESERVE_YUAN
                if transport_error is not None:
                    transport_error.unknown_cost_reserve_yuan = _number(
                        PRODUCTION_UNKNOWN_RESERVE_YUAN
                    )
                    raw_response = _transport_error_response(transport_error)
            raw_relative = Path("raw_attempts") / f"{len(attempts):04d}-{attempt_id}.json"
            artifact_bytes = _attempt_artifact(
                attempt_id=attempt_id,
                batch_id=batch_id,
                request=request,
                response=raw_response,
            )
            raw_path = execution_root / raw_relative
            _write_new(raw_path, artifact_bytes)
            raw_binding = {
                "path": raw_relative.as_posix(),
                "bytes": len(artifact_bytes),
                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            }
            raw_artifacts.append(raw_binding)
            input_tokens += attempt_input_tokens
            output_tokens += attempt_output_tokens
            if isinstance(attempt_known, Decimal):
                known_cost += attempt_known
            reserve_cost += attempt_reserve
            attempt: dict[str, Any] = {
                "attempt_id": attempt_id,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "attempt_number": attempt_number,
                "case_ids": case_ids,
                "request_sha256": hashlib.sha256(request).hexdigest(),
                "started_at_utc": attempt_started,
                "completed_at_utc": _utc_now(),
                "status": status,
                "retry_reason": (
                    status
                    if status in {"schema_error", "transport_error"}
                    else None
                ),
                "transport_reported_models": (
                    list(response["transport_reported_models"])
                    if response is not None
                    else list(outer.get("transport_reported_models") or [])
                    if isinstance(outer, Mapping)
                    else []
                ),
                "transport_reported_model_source": (
                    response["transport_reported_model_source"]
                    if response is not None
                    else outer.get("transport_reported_model_source")
                    if isinstance(outer, Mapping)
                    else None
                ),
                "transport_request_id": (
                    response["transport_request_id"]
                    if response is not None
                    else outer.get("transport_request_id")
                    if isinstance(outer, Mapping)
                    else None
                ),
                "raw_outer_response_sha256": hashlib.sha256(raw_response).hexdigest(),
                "output_sha256": (
                    canonical_sha256(batch_rows) if status == "success" else None
                ),
                "input_tokens": attempt_input_tokens,
                "output_tokens": attempt_output_tokens,
                "known_cost_yuan": (
                    None if attempt_known is None else _number(attempt_known)
                ),
                "unknown_cost_reserve_yuan": _number(attempt_reserve),
                "tools_used": bool(
                    isinstance(outer, Mapping) and outer.get("tool_calls")
                ),
                "raw_artifact": raw_binding,
            }
            attempt["attempt_sha256"] = canonical_sha256(attempt)
            attempts.append(attempt)
            if execution_exception is not None:
                terminal_failure = {
                    "reason": "execution_exception",
                    "terminal_status": "execution_error",
                    "error_type": (
                        f"{type(execution_exception).__module__}."
                        f"{type(execution_exception).__qualname__}"
                    ),
                    "error_message_sha256": hashlib.sha256(
                        str(execution_exception).encode("utf-8")
                    ).hexdigest(),
                }
                break
            if isolation_error is not None:
                terminal_failure = {
                    "reason": "isolation_failure",
                    "terminal_status": "isolation_error",
                    "error_type": type(isolation_error).__name__,
                    "error_message_sha256": hashlib.sha256(
                        str(isolation_error).encode("utf-8")
                    ).hexdigest(),
                }
                break
            if status == "success":
                normalized_rows.extend(batch_rows)
                break
            if attempt_number == MAX_ATTEMPTS_PER_BATCH:
                terminal_failure = {
                    "reason": f"{status.removesuffix('_error')}_retries_exhausted",
                    "terminal_status": status,
                    "error_type": (
                        type(transport_error).__name__
                        if transport_error is not None
                        else type(schema_exception).__name__
                        if schema_exception is not None
                        else "JudgeSchemaError"
                    ),
                    "error_message_sha256": hashlib.sha256(
                        (
                            str(transport_error)
                            if transport_error is not None
                            else str(schema_exception)
                            if schema_exception is not None
                            else "judge output schema retries exhausted"
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                break
        if terminal_failure is not None:
            break
    normalized_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in normalized_rows
    )
    normalized_relative = Path("normalized_results.jsonl")
    _write_new(execution_root / normalized_relative, normalized_bytes)
    attempt_set = [
        {"attempt_id": row["attempt_id"], "attempt_sha256": row["attempt_sha256"]}
        for row in attempts
    ]
    completed_at = _utc_now()
    common_receipt: dict[str, Any] = {
        "simulated": True,
        "run_id": RUN_ID,
        "case_binding": {
            "case_manifest_schema_version": manifest["schema_version"],
            "case_manifest_sha256": manifest["case_manifest_sha256"],
            "shared_input_sha256": manifest["shared_input_sha256"],
            "input_bytes": len(input_bytes),
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "case_count": len(cases),
            "ordered_case_ids_sha256": canonical_sha256(
                [case["case_id"] for case in cases]
            ),
            "judge_protocol_sha256": manifest["judge_protocol_sha256"],
            "judge_amendment_sha256": manifest["judge_amendment"]["sha256"],
        },
        "budget_authority": budget_binding,
        "identity": {
            "judge_family": judge_family,
            "requested_model": exact_model,
            "requested_model_verification_source": (
                "fixture_command_argv"
                if transport.name == "fixture"
                else "strict_cli_command_argv"
            ),
            "transport_reported_models": sorted(
                {
                    model
                    for attempt in attempts
                    for model in attempt["transport_reported_models"]
                }
            ),
            "transport_reported_model_verification_source": next(
                (
                    attempt["transport_reported_model_source"]
                    for attempt in attempts
                    if attempt["transport_reported_model_source"] is not None
                ),
                None,
            ),
            "transport": transport.name,
            "execution_id": execution_id,
        },
        "executable_evidence": executable_evidence,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "policy": {
            "batch_size": BATCH_SIZE,
            "max_attempts_per_batch": MAX_ATTEMPTS_PER_BATCH,
            "retryable_errors": ["schema_error", "transport_error"],
            "content_retry_allowed": False,
        },
        "isolation": {
            "fresh_execution": True,
            "resumed_session": False,
            "prior_conversation_context": False,
            "tools_disabled": True,
            "tools_used": any(bool(row["tools_used"]) for row in attempts),
            "no_external_case_data": True,
            "raw_environment_exported": False,
            "secrets_exported": False,
        },
        "command": {
            "argv": command,
            "argv_sha256": canonical_sha256(command),
            "environment_exported": False,
            "strict_configuration_verified_from_argv": strict_configuration_verified,
        },
        "attempts": attempts,
        "ordered_attempt_ids": [row["attempt_id"] for row in attempts],
        "attempt_set_sha256": canonical_sha256(attempt_set),
        "raw_artifacts": raw_artifacts,
        "raw_artifact_set_sha256": canonical_sha256(raw_artifacts),
        "normalized_results": {
            "path": normalized_relative.as_posix(),
            "bytes": len(normalized_bytes),
            "sha256": hashlib.sha256(normalized_bytes).hexdigest(),
            "row_count": len(normalized_rows),
            "ordered_output_sha256": canonical_sha256(
                [canonical_sha256(row) for row in normalized_rows]
            ),
        },
        "accounting": {
            "request_count": request_count,
            "retry_count": retry_count,
            "transport_error_count": transport_error_count,
            "schema_error_count": schema_error_count,
            "content_retry_count": 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "known_cost_yuan": _number(known_cost),
            "unknown_cost_reserve_yuan": _number(reserve_cost),
            "accounted_cost_yuan": _number(known_cost + reserve_cost),
        },
    }
    if terminal_failure is not None:
        terminal_attempt = (
            None
            if terminal_failure.get("reason") == "budget_fuse_blocked"
            else attempts[-1]
            if attempts
            else None
        )
        completed_case_ids = [str(row["case_id"]) for row in normalized_rows]
        reason = str(terminal_failure["reason"])
        reason_code = f"judge_{reason}"
        needs_user_reasons = [reason_code]
        if reserve_cost > 0:
            needs_user_reasons.append("unknown_judge_billing_reserved")
        failure = {
            **terminal_failure,
            "terminal_attempt_id": (
                terminal_attempt["attempt_id"] if terminal_attempt is not None else None
            ),
            "terminal_batch_id": (
                terminal_attempt["batch_id"] if terminal_attempt is not None else None
            ),
            "completed_case_count": len(completed_case_ids),
            "missing_case_count": len(cases) - len(completed_case_ids),
            "ordered_completed_case_ids_sha256": canonical_sha256(
                completed_case_ids
            ),
        }
        receipt = {
            "schema_version": FAILED_RECEIPT_SCHEMA,
            "status": "failed",
            **common_receipt,
            "failure": failure,
            "needs_user": True,
            "needs_user_reasons": needs_user_reasons,
        }
        receipt["failed_execution_receipt_sha256"] = canonical_sha256(receipt)
        receipt_path = execution_root / "failed_execution_receipt.json"
        _write_new(receipt_path, canonical_json_bytes(receipt) + b"\n")
        validate_failed_execution_receipt(
            receipt_path,
            manifest,
            judge_family,
            allow_fixture=transport.name == "fixture",
        )
        raise JudgePassFailed(
            f"judge pass failed: {reason}",
            receipt_path=receipt_path,
            receipt=receipt,
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        **common_receipt,
    }
    receipt["execution_receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = execution_root / "execution_receipt.json"
    _write_new(receipt_path, canonical_json_bytes(receipt) + b"\n")
    validate_execution_receipt(
        receipt_path,
        manifest,
        judge_family,
        allow_fixture=transport.name == "fixture",
    )
    return receipt_path


def _utc_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise JudgeExecutionError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JudgeExecutionError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise JudgeExecutionError(f"{label} must be in UTC")
    return parsed


def _validate_receipt(
    receipt: Mapping[str, Any] | str | Path,
    case_manifest: Mapping[str, Any] | str | Path,
    expected_judge: str,
    *,
    artifact_root: str | Path | None = None,
    allow_fixture: bool = False,
    expect_failed: bool,
) -> dict[str, Any]:
    """Validate a receipt and, when locatable, every bound artifact byte."""

    receipt_path = Path(receipt) if isinstance(receipt, (str, Path)) else None
    receipt_label = "failed judge execution receipt" if expect_failed else "judge execution receipt"
    value = _load_object(receipt, label=receipt_label)
    manifest, input_bytes = _validate_case_manifest(
        _load_object(case_manifest, label="judge case manifest")
    )
    if not expect_failed and (
        value.get("schema_version") == FAILED_RECEIPT_SCHEMA
        or value.get("status") == "failed"
    ):
        raise JudgeExecutionError(
            "failed judge receipt is not a completed publication result"
        )
    if expect_failed and (
        value.get("schema_version") == RECEIPT_SCHEMA
        or value.get("status") == "complete"
    ):
        raise JudgeExecutionError("completed judge receipt is not failed evidence")
    receipt_hash_field = (
        "failed_execution_receipt_sha256"
        if expect_failed
        else "execution_receipt_sha256"
    )
    _self_hash(value, receipt_hash_field, label=receipt_label)
    identity = value.get("identity")
    binding = value.get("case_binding")
    policy = value.get("policy")
    isolation = value.get("isolation")
    command = value.get("command")
    budget_authority = value.get("budget_authority")
    executable_evidence = value.get("executable_evidence")
    expected_policy = {
        "batch_size": BATCH_SIZE,
        "max_attempts_per_batch": MAX_ATTEMPTS_PER_BATCH,
        "retryable_errors": ["schema_error", "transport_error"],
        "content_retry_allowed": False,
    }
    expected_isolation = {
        "fresh_execution": True,
        "resumed_session": False,
        "prior_conversation_context": False,
        "tools_disabled": True,
        "tools_used": False,
        "no_external_case_data": True,
        "raw_environment_exported": False,
        "secrets_exported": False,
    }
    expected_receipt_fields = {
        "schema_version",
        "simulated",
        "run_id",
        "status",
        "case_binding",
        "budget_authority",
        "executable_evidence",
        "identity",
        "started_at_utc",
        "completed_at_utc",
        "policy",
        "isolation",
        "command",
        "attempts",
        "ordered_attempt_ids",
        "attempt_set_sha256",
        "raw_artifacts",
        "raw_artifact_set_sha256",
        "normalized_results",
        "accounting",
        receipt_hash_field,
    }
    if expect_failed:
        expected_receipt_fields.update(
            {"failure", "needs_user", "needs_user_reasons"}
        )
    expected_case_ids = [str(case["case_id"]) for case in manifest["cases"]]
    expected_binding = {
        "case_manifest_schema_version": manifest["schema_version"],
        "case_manifest_sha256": manifest["case_manifest_sha256"],
        "shared_input_sha256": manifest["shared_input_sha256"],
        "input_bytes": len(input_bytes),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "case_count": len(expected_case_ids),
        "ordered_case_ids_sha256": canonical_sha256(expected_case_ids),
        "judge_protocol_sha256": manifest["judge_protocol_sha256"],
        "judge_amendment_sha256": manifest["judge_amendment"]["sha256"],
    }
    if (
        set(value) != expected_receipt_fields
        or value.get("schema_version")
        != (FAILED_RECEIPT_SCHEMA if expect_failed else RECEIPT_SCHEMA)
        or value.get("simulated") is not True
        or value.get("run_id") != RUN_ID
        or value.get("status") != ("failed" if expect_failed else "complete")
        or not isinstance(identity, Mapping)
        or set(identity)
        != {
            "judge_family",
            "requested_model",
            "requested_model_verification_source",
            "transport_reported_models",
            "transport_reported_model_verification_source",
            "transport",
            "execution_id",
        }
        or identity.get("judge_family") != expected_judge
        or not isinstance(identity.get("requested_model"), str)
        or not identity.get("requested_model")
        or identity.get("requested_model_verification_source")
        not in {"strict_cli_command_argv", "fixture_command_argv"}
        or not isinstance(identity.get("transport_reported_models"), list)
        or any(
            not isinstance(model, str) or not _MODEL_NAME.fullmatch(model)
            for model in identity.get("transport_reported_models", [])
        )
        or (
            bool(identity.get("transport_reported_models"))
            != isinstance(
                identity.get("transport_reported_model_verification_source"), str
            )
        )
        or not isinstance(identity.get("transport"), str)
        or not isinstance(identity.get("execution_id"), str)
        or binding != expected_binding
        or not isinstance(budget_authority, Mapping)
        or set(budget_authority)
        != {
            "judge_budget_authority_sha256",
            "authority_kind",
            "baseline_accounted_cost_yuan",
            "unknown_reserve_per_attempt_yuan",
            "hard_fuse_yuan",
        }
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(budget_authority.get("judge_budget_authority_sha256")),
        )
        or budget_authority.get("unknown_reserve_per_attempt_yuan")
        != _number(PRODUCTION_UNKNOWN_RESERVE_YUAN)
        or budget_authority.get("hard_fuse_yuan") != _number(HARD_FUSE_YUAN)
        or policy != expected_policy
        or (
            isolation != expected_isolation
            if not expect_failed
            else not isinstance(isolation, Mapping)
            or set(isolation) != set(expected_isolation)
            or any(
                isolation.get(key) != expected
                for key, expected in expected_isolation.items()
                if key != "tools_used"
            )
            or not isinstance(isolation.get("tools_used"), bool)
        )
        or not isinstance(command, Mapping)
        or set(command)
        != {
            "argv",
            "argv_sha256",
            "environment_exported",
            "strict_configuration_verified_from_argv",
        }
        or command.get("environment_exported") is not False
        or not isinstance(command.get("strict_configuration_verified_from_argv"), bool)
        or not isinstance(command.get("argv"), list)
        or command.get("argv_sha256") != canonical_sha256(command.get("argv"))
    ):
        raise JudgeExecutionError("judge execution receipt identity or input binding failed")
    try:
        parsed_execution_id = uuid.UUID(str(identity["execution_id"]))
    except (ValueError, TypeError, AttributeError) as exc:
        raise JudgeExecutionError("judge execution ID is not a UUID") from exc
    if str(parsed_execution_id) != identity["execution_id"]:
        raise JudgeExecutionError("judge execution ID is not canonical")
    argv = command["argv"]
    model = str(identity["requested_model"])
    if model not in argv:
        raise JudgeExecutionError("exact judge model is absent from the bound command")
    transport_name = str(identity["transport"])
    _validate_executable_evidence(
        executable_evidence,
        transport_name=transport_name,
        allow_fixture=allow_fixture,
    )
    if transport_name == "fixture":
        if not allow_fixture:
            raise JudgeExecutionError(
                "fixture judge receipts are not eligible for production validation"
            )
        if budget_authority.get("authority_kind") not in {
            "test_fixture",
            "formal_collection_committed_anchor_and_live_cost_ledger",
        }:
            raise JudgeExecutionError("fixture judge budget authority is invalid")
        if (
            identity.get("requested_model_verification_source")
            != "fixture_command_argv"
            or command.get("strict_configuration_verified_from_argv") is not False
        ):
            raise JudgeExecutionError("fixture judge command provenance is invalid")
        expected_argv = [
            "fixture-judge",
            "--model",
            model,
            "--tools",
            "disabled",
        ]
    elif transport_name == "codex_cli" and expected_judge == "gpt":
        if (
            budget_authority.get("authority_kind")
            != "formal_collection_committed_anchor_and_live_cost_ledger"
        ):
            raise JudgeExecutionError("production judge lacks formal budget authority")
        if (
            identity.get("requested_model_verification_source")
            != "strict_cli_command_argv"
            or command.get("strict_configuration_verified_from_argv") is not True
        ):
            raise JudgeExecutionError("Codex strict configuration evidence is invalid")
        if not argv or not isinstance(argv[0], str):
            raise JudgeExecutionError("Codex judge command is invalid")
        expected_argv = CodexCLIJudgeTransport(binary=argv[0]).command_argv(model)
    elif transport_name == "claude_cli" and expected_judge == "claude":
        if (
            budget_authority.get("authority_kind")
            != "formal_collection_committed_anchor_and_live_cost_ledger"
        ):
            raise JudgeExecutionError("production judge lacks formal budget authority")
        if (
            identity.get("requested_model_verification_source")
            != "strict_cli_command_argv"
            or command.get("strict_configuration_verified_from_argv") is not True
        ):
            raise JudgeExecutionError("Claude strict configuration evidence is invalid")
        if not argv or not isinstance(argv[0], str):
            raise JudgeExecutionError("Claude judge command is invalid")
        expected_argv = ClaudeCLIJudgeTransport(binary=argv[0]).command_argv(model)
    else:
        raise JudgeExecutionError(
            "judge receipt transport is not an allowed production family"
        )
    if argv != expected_argv:
        raise JudgeExecutionError("bound judge command differs from the tool-free command")
    if transport_name != "fixture" and argv[0] != executable_evidence.get(
        "resolved_binary_realpath"
    ):
        raise JudgeExecutionError("judge command executable differs from bound realpath")
    started = _utc_timestamp(value.get("started_at_utc"), label="judge start timestamp")
    completed = _utc_timestamp(
        value.get("completed_at_utc"), label="judge completion timestamp"
    )
    if completed < started:
        raise JudgeExecutionError("judge execution timestamp order is invalid")
    attempts = value.get("attempts")
    raw_artifacts = value.get("raw_artifacts")
    normalized = value.get("normalized_results")
    if (
        not isinstance(attempts, list)
        or (
            not attempts
            and (
                not expect_failed
                or not isinstance(value.get("failure"), Mapping)
                or value["failure"].get("reason") != "budget_fuse_blocked"
            )
        )
        or not isinstance(raw_artifacts, list)
        or not isinstance(normalized, Mapping)
    ):
        raise JudgeExecutionError("judge execution receipt artifact roster is invalid")
    for attempt in attempts:
        expected_attempt_fields = {
            "attempt_id",
            "batch_id",
            "batch_index",
            "attempt_number",
            "case_ids",
            "request_sha256",
            "started_at_utc",
            "completed_at_utc",
            "status",
            "retry_reason",
            "transport_reported_models",
            "transport_reported_model_source",
            "transport_request_id",
            "raw_outer_response_sha256",
            "output_sha256",
            "input_tokens",
            "output_tokens",
            "known_cost_yuan",
            "unknown_cost_reserve_yuan",
            "tools_used",
            "raw_artifact",
            "attempt_sha256",
        }
        if not isinstance(attempt, Mapping) or set(attempt) != expected_attempt_fields:
            raise JudgeExecutionError("judge attempt receipt schema is invalid")
        _self_hash(attempt, "attempt_sha256", label="judge attempt")
    attempt_set = [
        {"attempt_id": row["attempt_id"], "attempt_sha256": row["attempt_sha256"]}
        for row in attempts
    ]
    if (
        value.get("ordered_attempt_ids") != [row["attempt_id"] for row in attempts]
        or value.get("attempt_set_sha256") != canonical_sha256(attempt_set)
        or value.get("raw_artifact_set_sha256") != canonical_sha256(raw_artifacts)
    ):
        raise JudgeExecutionError("judge attempt or raw-artifact set hash mismatch")
    if raw_artifacts != [row.get("raw_artifact") for row in attempts]:
        raise JudgeExecutionError("raw artifact order differs from the attempt order")
    expected_batches = [
        expected_case_ids[offset : offset + BATCH_SIZE]
        for offset in range(0, len(expected_case_ids), BATCH_SIZE)
    ]
    successful_case_ids: list[str] = []
    request_count = len(attempts)
    retry_count = 0
    schema_error_count = 0
    transport_error_count = 0
    isolation_error_count = 0
    input_tokens = 0
    output_tokens = 0
    known_cost = Decimal("0")
    reserve_cost = Decimal("0")
    previous_batch_index = -1
    previous_attempt_number = 0
    previous_status: str | None = None
    previous_attempt_completed = started
    seen_attempt_ids: set[str] = set()
    allowed_statuses = {"success", "schema_error", "transport_error"}
    if expect_failed:
        allowed_statuses.update({"isolation_error", "execution_error"})
    for attempt in attempts:
        status = attempt.get("status")
        batch_index = attempt.get("batch_index")
        attempt_number = attempt.get("attempt_number")
        case_ids = attempt.get("case_ids")
        attempt_id = attempt.get("attempt_id")
        attempt_reported_models = attempt.get("transport_reported_models")
        attempt_reported_source = attempt.get("transport_reported_model_source")
        if (
            status not in allowed_statuses
            or not isinstance(batch_index, int)
            or isinstance(batch_index, bool)
            or batch_index < 0
            or not isinstance(attempt_number, int)
            or isinstance(attempt_number, bool)
            or attempt_number not in {1, 2}
            or not isinstance(case_ids, list)
            or not case_ids
            or not isinstance(attempt_id, str)
            or attempt_id in seen_attempt_ids
            or not isinstance(attempt.get("tools_used"), bool)
            or not isinstance(attempt_reported_models, list)
            or any(
                not isinstance(reported_model, str)
                or not _MODEL_NAME.fullmatch(reported_model)
                for reported_model in attempt_reported_models
            )
            or len(attempt_reported_models) != len(set(attempt_reported_models))
            or (
                bool(attempt_reported_models)
                != isinstance(attempt_reported_source, str)
            )
            or (
                status != "isolation_error"
                and attempt.get("tools_used") is not False
            )
        ):
            raise JudgeExecutionError("judge attempt lifecycle is invalid")
        try:
            parsed_attempt_id = uuid.UUID(attempt_id)
        except (ValueError, AttributeError) as exc:
            raise JudgeExecutionError("judge attempt ID is not a UUID") from exc
        if str(parsed_attempt_id) != attempt_id:
            raise JudgeExecutionError("judge attempt ID is not canonical")
        attempt_started = _utc_timestamp(
            attempt.get("started_at_utc"), label="judge attempt start timestamp"
        )
        attempt_completed = _utc_timestamp(
            attempt.get("completed_at_utc"), label="judge attempt completion timestamp"
        )
        if (
            attempt_started < started
            or attempt_completed < attempt_started
            or attempt_completed > completed
            or attempt_started < previous_attempt_completed
        ):
            raise JudgeExecutionError("judge attempt timestamp order is invalid")
        previous_attempt_completed = attempt_completed
        seen_attempt_ids.add(attempt_id)
        if (
            batch_index >= len(expected_batches)
            or case_ids != expected_batches[batch_index]
            or attempt.get("batch_id") != f"batch-{batch_index:04d}"
        ):
            raise JudgeExecutionError("judge attempt cases differ from the frozen batch")
        if batch_index == previous_batch_index:
            if attempt_number != previous_attempt_number + 1:
                raise JudgeExecutionError("judge retry numbering is not consecutive")
            if previous_status not in {"schema_error", "transport_error"}:
                raise JudgeExecutionError("judge retry did not follow a retryable failure")
        else:
            if batch_index != previous_batch_index + 1 or attempt_number != 1:
                raise JudgeExecutionError("judge batch ordering is invalid")
        previous_batch_index = batch_index
        previous_attempt_number = attempt_number
        previous_status = str(status)
        if (
            status == "success"
            and (attempt.get("retry_reason") is not None or not attempt.get("output_sha256"))
        ):
            raise JudgeExecutionError("successful judge attempt status fields are invalid")
        if status in {"schema_error", "transport_error"} and (
            attempt.get("retry_reason") != status
            or attempt.get("output_sha256") is not None
        ):
            raise JudgeExecutionError("failed judge attempt status fields are invalid")
        if status == "isolation_error" and (
            attempt.get("retry_reason") is not None
            or attempt.get("output_sha256") is not None
        ):
            raise JudgeExecutionError("isolation attempt status fields are invalid")
        if status == "execution_error" and (
            attempt.get("retry_reason") is not None
            or attempt.get("output_sha256") is not None
            or attempt.get("transport_request_id") is not None
            or attempt.get("transport_reported_models") != []
            or attempt.get("transport_reported_model_source") is not None
        ):
            raise JudgeExecutionError("execution-error attempt status fields are invalid")
        if attempt_number > 1:
            retry_count += 1
        if status == "schema_error":
            schema_error_count += 1
        elif status == "transport_error":
            transport_error_count += 1
        elif status == "isolation_error":
            isolation_error_count += 1
        elif status == "execution_error":
            pass
        else:
            reported_models = attempt.get("transport_reported_models")
            reported_source = attempt.get("transport_reported_model_source")
            if (
                not isinstance(reported_models, list)
                or reported_models not in ([], [model])
                or (bool(reported_models) != isinstance(reported_source, str))
            ):
                raise JudgeExecutionError(
                    "successful judge attempt transport-reported model drifted"
                )
            successful_case_ids.extend(str(case_id) for case_id in case_ids)
        attempt_input = attempt.get("input_tokens")
        attempt_output = attempt.get("output_tokens")
        if (
            not isinstance(attempt_input, int)
            or isinstance(attempt_input, bool)
            or attempt_input < 0
            or not isinstance(attempt_output, int)
            or isinstance(attempt_output, bool)
            or attempt_output < 0
        ):
            raise JudgeExecutionError("judge attempt token accounting is invalid")
        input_tokens += attempt_input
        output_tokens += attempt_output
        known = _decimal(
            attempt.get("known_cost_yuan"),
            label="attempt known judge cost",
            allow_null=True,
        )
        reserve = _decimal(
            attempt.get("unknown_cost_reserve_yuan"),
            label="attempt unknown judge reserve",
        )
        assert reserve is not None
        if known is not None:
            known_cost += known
        reserve_cost += reserve
    aggregate_reported_models = sorted(
        {
            reported_model
            for attempt in attempts
            for reported_model in attempt["transport_reported_models"]
        }
    )
    aggregate_sources = sorted(
        {
            str(attempt["transport_reported_model_source"])
            for attempt in attempts
            if attempt["transport_reported_model_source"] is not None
        }
    )
    if (
        identity.get("transport_reported_models") != aggregate_reported_models
        or identity.get("transport_reported_model_verification_source")
        != (aggregate_sources[0] if len(aggregate_sources) == 1 else None)
        or len(aggregate_sources) > 1
    ):
        raise JudgeExecutionError("judge aggregate model evidence differs from attempts")
    if (
        successful_case_ids != expected_case_ids
        if not expect_failed
        else expected_case_ids[: len(successful_case_ids)] != successful_case_ids
        or len(successful_case_ids) >= len(expected_case_ids)
    ):
        raise JudgeExecutionError("successful judge attempts do not match receipt status")
    accounting = value.get("accounting")
    expected_accounting = {
        "request_count": request_count,
        "retry_count": retry_count,
        "transport_error_count": transport_error_count,
        "schema_error_count": schema_error_count,
        "content_retry_count": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "known_cost_yuan": _number(known_cost),
        "unknown_cost_reserve_yuan": _number(reserve_cost),
        "accounted_cost_yuan": _number(known_cost + reserve_cost),
    }
    if accounting != expected_accounting:
        raise JudgeExecutionError("judge aggregate accounting differs from attempts")
    if isolation.get("tools_used") != any(
        bool(row.get("tools_used")) for row in attempts
    ):
        raise JudgeExecutionError("judge isolation summary differs from attempts")
    if expect_failed:
        failure = value.get("failure")
        terminal_attempt = (
            None
            if isinstance(failure, Mapping)
            and failure.get("reason") == "budget_fuse_blocked"
            else attempts[-1]
            if attempts
            else None
        )
        terminal_status = (
            terminal_attempt.get("status")
            if terminal_attempt is not None
            else "fuse_blocked"
        )
        if terminal_status == "fuse_blocked":
            expected_reason = "budget_fuse_blocked"
            terminal_shape_valid = isolation_error_count == 0
        elif terminal_status == "execution_error":
            expected_reason = "execution_exception"
            terminal_shape_valid = isolation_error_count == 0
        elif terminal_status == "isolation_error":
            expected_reason = "isolation_failure"
            terminal_shape_valid = isolation_error_count == 1
        elif terminal_status in {"schema_error", "transport_error"}:
            expected_reason = (
                f"{str(terminal_status).removesuffix('_error')}_retries_exhausted"
            )
            terminal_shape_valid = (
                terminal_attempt.get("attempt_number")
                == MAX_ATTEMPTS_PER_BATCH
                and isolation_error_count == 0
            )
        else:
            expected_reason = ""
            terminal_shape_valid = False
        expected_failure_fields = {
            "reason",
            "terminal_status",
            "error_type",
            "error_message_sha256",
            "terminal_attempt_id",
            "terminal_batch_id",
            "completed_case_count",
            "missing_case_count",
            "ordered_completed_case_ids_sha256",
        }
        if (
            not terminal_shape_valid
            or not isinstance(failure, Mapping)
            or set(failure) != expected_failure_fields
            or failure.get("reason") != expected_reason
            or failure.get("terminal_status") != terminal_status
            or not isinstance(failure.get("error_type"), str)
            or not str(failure.get("error_type")).strip()
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(failure.get("error_message_sha256"))
            )
            or failure.get("terminal_attempt_id")
            != (
                terminal_attempt.get("attempt_id")
                if terminal_attempt is not None
                else None
            )
            or failure.get("terminal_batch_id")
            != (
                terminal_attempt.get("batch_id")
                if terminal_attempt is not None
                else None
            )
            or failure.get("completed_case_count") != len(successful_case_ids)
            or failure.get("missing_case_count")
            != len(expected_case_ids) - len(successful_case_ids)
            or failure.get("ordered_completed_case_ids_sha256")
            != canonical_sha256(successful_case_ids)
        ):
            raise JudgeExecutionError("failed judge receipt terminal proof is invalid")
        expected_needs_user_reasons = [f"judge_{expected_reason}"]
        if reserve_cost > 0:
            expected_needs_user_reasons.append("unknown_judge_billing_reserved")
        if (
            value.get("needs_user") is not True
            or value.get("needs_user_reasons") != expected_needs_user_reasons
        ):
            raise JudgeExecutionError("failed judge needs_user accounting is invalid")
    root: Path | None
    if artifact_root is not None:
        root_value = Path(artifact_root).expanduser()
        if root_value.is_symlink():
            raise JudgeExecutionError("judge artifact root cannot be a symlink")
        try:
            root = root_value.resolve(strict=True)
        except OSError as exc:
            raise JudgeExecutionError("judge artifact root is missing") from exc
    elif receipt_path is not None:
        receipt_value = receipt_path.expanduser()
        if receipt_value.parent.is_symlink():
            raise JudgeExecutionError("judge artifact root cannot be a symlink")
        root = receipt_value.resolve().parent
    else:
        root = None
    if root is not None:
        if not root.is_dir() or root.name != identity["execution_id"]:
            raise JudgeExecutionError("artifact root does not match the execution ID")
        expected_files = {
            (
                "failed_execution_receipt.json"
                if expect_failed
                else "execution_receipt.json"
            ),
            str(normalized.get("path") or ""),
            *(str(row.get("path") or "") for row in raw_artifacts),
        }
        observed_files: set[str] = set()
        for current_value, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(current_value)
            for name in directory_names:
                if (current / name).is_symlink():
                    raise JudgeExecutionError(
                        "judge execution tree contains an unbound symlink"
                    )
            for name in file_names:
                path = current / name
                if path.is_symlink() or not path.is_file():
                    raise JudgeExecutionError(
                        "judge execution tree contains an unsafe entry"
                    )
                observed_files.add(path.relative_to(root).as_posix())
        if observed_files != expected_files:
            raise JudgeExecutionError(
                "judge execution tree contains an unbound or missing file set"
            )
        for artifact in raw_artifacts:
            if not isinstance(artifact, Mapping):
                raise JudgeExecutionError("raw judge artifact binding is invalid")
            relative = Path(str(artifact.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise JudgeExecutionError("raw judge artifact path is unsafe")
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise JudgeExecutionError("raw judge artifact is missing or unsafe")
            data = path.read_bytes()
            if (
                artifact.get("bytes") != len(data)
                or artifact.get("sha256") != hashlib.sha256(data).hexdigest()
            ):
                raise JudgeExecutionError("raw judge artifact bytes drifted")
        category_policy = manifest["judge_protocol"]["label_category_policy"]
        replayed_success_rows: list[dict[str, Any]] = []
        replayed_terminal_error: tuple[str, str] | None = None
        for artifact_index, (attempt, artifact) in enumerate(
            zip(attempts, raw_artifacts, strict=True)
        ):
            expected_raw_relative = (
                f"raw_attempts/{artifact_index:04d}-{attempt['attempt_id']}.json"
            )
            if set(artifact) != {"path", "bytes", "sha256"} or artifact.get(
                "path"
            ) != expected_raw_relative:
                raise JudgeExecutionError("raw judge artifact roster drifted")
            raw_path = root / Path(str(artifact["path"]))
            raw_value = _strict_json_bytes(
                raw_path.read_bytes(), label="raw judge attempt artifact"
            )
            if (
                set(raw_value)
                != {
                    "schema_version",
                    "attempt_id",
                    "batch_id",
                    "request_base64",
                    "request_sha256",
                    "raw_outer_response_base64",
                    "raw_outer_response_sha256",
                }
                or
                raw_value.get("schema_version")
                != "yher.llm_sim_v2.judge_raw_attempt.v1"
                or raw_value.get("attempt_id") != attempt.get("attempt_id")
                or raw_value.get("batch_id") != attempt.get("batch_id")
                or raw_value.get("request_sha256") != attempt.get("request_sha256")
                or raw_value.get("raw_outer_response_sha256")
                != attempt.get("raw_outer_response_sha256")
            ):
                raise JudgeExecutionError("raw judge attempt binding drifted")
            try:
                request_bytes = base64.b64decode(
                    str(raw_value["request_base64"]), validate=True
                )
                response_bytes = base64.b64decode(
                    str(raw_value["raw_outer_response_base64"]), validate=True
                )
            except (KeyError, ValueError) as exc:
                raise JudgeExecutionError("raw judge attempt base64 is invalid") from exc
            if (
                hashlib.sha256(request_bytes).hexdigest()
                != attempt.get("request_sha256")
                or hashlib.sha256(response_bytes).hexdigest()
                != attempt.get("raw_outer_response_sha256")
            ):
                raise JudgeExecutionError("raw judge request or response hash drifted")
            batch_index = int(attempt["batch_index"])
            batch_start = batch_index * BATCH_SIZE
            batch = manifest["cases"][batch_start : batch_start + BATCH_SIZE]
            expected_request = _batch_request_bytes(
                manifest=manifest,
                batch=batch,
                batch_index=batch_index,
                judge_family=expected_judge,
                exact_model=model,
            )
            if request_bytes != expected_request:
                raise JudgeExecutionError(
                    "raw judge request differs from the exact frozen batch request"
                )
            status = str(attempt["status"])
            if status == "execution_error":
                execution_failure = _strict_json_bytes(
                    response_bytes, label="raw judge execution failure"
                )
                if (
                    set(execution_failure)
                    != {
                        "schema_version",
                        "error_type",
                        "message_sha256",
                        "call_may_have_begun",
                        "usage",
                        "billing",
                    }
                    or execution_failure.get("schema_version")
                    != "yher.llm_sim_v2.judge_execution_error.v1"
                    or execution_failure.get("call_may_have_begun") is not True
                    or not isinstance(execution_failure.get("error_type"), str)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(execution_failure.get("message_sha256")),
                    )
                    or execution_failure.get("usage")
                    != {"input_tokens": 0, "output_tokens": 0}
                    or execution_failure.get("billing")
                    != {
                        "known_cost_yuan": None,
                        "unknown_cost_reserve_yuan": _number(
                            PRODUCTION_UNKNOWN_RESERVE_YUAN
                        ),
                    }
                    or attempt.get("input_tokens") != 0
                    or attempt.get("output_tokens") != 0
                    or attempt.get("known_cost_yuan") is not None
                    or attempt.get("unknown_cost_reserve_yuan")
                    != _number(PRODUCTION_UNKNOWN_RESERVE_YUAN)
                ):
                    raise JudgeExecutionError(
                        "raw judge execution failure evidence drifted"
                    )
                if expect_failed and artifact_index == len(attempts) - 1:
                    replayed_terminal_error = (
                        str(execution_failure["error_type"]),
                        str(execution_failure["message_sha256"]),
                    )
                continue
            if status == "transport_error":
                transport_failure = _strict_json_bytes(
                    response_bytes, label="raw judge transport failure"
                )
                expected_failure_fields = {
                    "schema_version",
                    "error_type",
                    "message_sha256",
                    "raw_failure_base64",
                    "raw_failure_bytes",
                    "raw_failure_sha256",
                    "usage",
                    "billing",
                }
                if (
                    set(transport_failure) != expected_failure_fields
                    or transport_failure.get("schema_version")
                    != "yher.llm_sim_v2.judge_transport_error.v1"
                    or not isinstance(transport_failure.get("error_type"), str)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(transport_failure.get("message_sha256")),
                    )
                ):
                    raise JudgeExecutionError(
                        "raw judge transport failure envelope drifted"
                    )
                try:
                    raw_failure = base64.b64decode(
                        str(transport_failure["raw_failure_base64"]), validate=True
                    )
                except ValueError as exc:
                    raise JudgeExecutionError(
                        "raw judge transport failure base64 drifted"
                    ) from exc
                if (
                    transport_failure.get("raw_failure_bytes") != len(raw_failure)
                    or transport_failure.get("raw_failure_sha256")
                    != hashlib.sha256(raw_failure).hexdigest()
                ):
                    raise JudgeExecutionError(
                        "raw judge transport failure bytes drifted"
                    )
                usage = transport_failure.get("usage")
                billing = transport_failure.get("billing")
                if (
                    not isinstance(usage, Mapping)
                    or set(usage) != {"input_tokens", "output_tokens"}
                    or not isinstance(billing, Mapping)
                    or set(billing)
                    != {"known_cost_yuan", "unknown_cost_reserve_yuan"}
                ):
                    raise JudgeExecutionError(
                        "raw judge transport failure accounting drifted"
                    )
                failure_known = _decimal(
                    billing.get("known_cost_yuan"),
                    label="raw transport-error known judge cost",
                    allow_null=True,
                )
                failure_reserve = _decimal(
                    billing.get("unknown_cost_reserve_yuan"),
                    label="raw transport-error unknown judge reserve",
                )
                assert failure_reserve is not None
                if (
                    attempt.get("input_tokens") != usage.get("input_tokens")
                    or attempt.get("output_tokens") != usage.get("output_tokens")
                    or attempt.get("known_cost_yuan")
                    != (
                        None
                        if failure_known is None
                        else _number(failure_known)
                    )
                    or attempt.get("unknown_cost_reserve_yuan")
                    != _number(failure_reserve)
                ):
                    raise JudgeExecutionError(
                        "raw judge transport failure accounting differs from the attempt"
                    )
                if expect_failed and artifact_index == len(attempts) - 1:
                    replayed_terminal_error = (
                        str(transport_failure["error_type"]),
                        str(transport_failure["message_sha256"]),
                    )
                continue
            if transport_name in {"codex_cli", "claude_cli"}:
                raw_outer = _strict_json_bytes(
                    response_bytes, label="raw judge CLI envelope"
                )
                raw_transport = raw_outer.get("raw_transport")
                billing = raw_outer.get("billing")
                if not isinstance(raw_transport, Mapping) or not isinstance(
                    billing, Mapping
                ):
                    raise JudgeExecutionError(
                        "production judge envelope lacks raw CLI evidence"
                    )
                reserve = _decimal(
                    billing.get("unknown_cost_reserve_yuan"),
                    label="raw CLI unknown judge reserve",
                )
                assert reserve is not None
                parser = (
                    _codex_response_from_raw_cli
                    if transport_name == "codex_cli"
                    else _claude_response_from_raw_cli
                )
                try:
                    derived_outer = parser(
                        raw_transport=raw_transport,
                        exact_model=model,
                        attempt_id=str(attempt["attempt_id"]),
                        unknown_cost_reserve_yuan=_number(reserve),
                    )
                except JudgeExecutionError as exc:
                    raise JudgeExecutionError(
                        "raw judge CLI stdout cannot be replayed"
                    ) from exc
                if raw_outer != derived_outer:
                    raise JudgeExecutionError(
                        "judge envelope differs from raw CLI stdout replay"
                    )
            replay_input, replay_output, replay_known, replay_reserve, replay_outer = (
                _response_accounting(response_bytes)
            )
            try:
                replay_response, replay_rows = _validate_transport_response(
                    response_bytes,
                    case_ids=[str(case_id) for case_id in attempt["case_ids"]],
                    exact_model=model,
                    category_policy=category_policy,
                )
            except JudgeIsolationError as exc:
                if not expect_failed or status != "isolation_error":
                    raise JudgeExecutionError(
                        "raw judge replay proves an isolation failure"
                    ) from exc
                replay_response = None
                replay_rows = []
                if artifact_index == len(attempts) - 1:
                    replayed_terminal_error = (
                        type(exc).__name__,
                        hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                    )
            except JudgeExecutionError as exc:
                if status != "schema_error":
                    raise JudgeExecutionError(
                        "raw judge replay does not support the recorded success status"
                    ) from exc
                replay_response = None
                replay_rows = []
                if expect_failed and artifact_index == len(attempts) - 1:
                    replayed_terminal_error = (
                        type(exc).__name__,
                        hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                    )
            else:
                if status != "success":
                    raise JudgeExecutionError(
                        "raw judge replay disproves the recorded retryable status"
                    )
            replay_known_value = None if replay_known is None else _number(replay_known)
            if (
                attempt.get("input_tokens") != replay_input
                or attempt.get("output_tokens") != replay_output
                or attempt.get("known_cost_yuan") != replay_known_value
                or attempt.get("unknown_cost_reserve_yuan")
                != _number(replay_reserve)
            ):
                raise JudgeExecutionError(
                    "raw judge replay accounting differs from the attempt"
                )
            replay_identity = (
                replay_response if replay_response is not None else replay_outer
            )
            if (
                attempt.get("transport_reported_models")
                != (
                    replay_identity.get("transport_reported_models")
                    if isinstance(replay_identity, Mapping)
                    else []
                )
                or attempt.get("transport_reported_model_source")
                != (
                    replay_identity.get("transport_reported_model_source")
                    if isinstance(replay_identity, Mapping)
                    else None
                )
                or attempt.get("transport_request_id")
                != (
                    replay_identity.get("transport_request_id")
                    if isinstance(replay_identity, Mapping)
                    else None
                )
            ):
                raise JudgeExecutionError(
                    "raw judge replay identity differs from the attempt"
                )
            if attempt.get("tools_used") != bool(
                isinstance(replay_outer, Mapping)
                and replay_outer.get("tool_calls")
            ):
                raise JudgeExecutionError(
                    "raw judge replay tool-use status differs from the attempt"
                )
            if status == "success":
                if attempt.get("output_sha256") != canonical_sha256(replay_rows):
                    raise JudgeExecutionError(
                        "raw judge replay output differs from the attempt"
                    )
                replayed_success_rows.extend(replay_rows)
        normalized_relative = Path(str(normalized.get("path") or ""))
        if (
            normalized_relative.as_posix() != "normalized_results.jsonl"
            or normalized_relative.is_absolute()
            or ".." in normalized_relative.parts
        ):
            raise JudgeExecutionError("normalized judge artifact path is unsafe")
        normalized_path = root / normalized_relative
        if normalized_path.is_symlink() or not normalized_path.is_file():
            raise JudgeExecutionError("normalized judge artifact is missing or unsafe")
        data = normalized_path.read_bytes()
        if (
            normalized.get("bytes") != len(data)
            or normalized.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise JudgeExecutionError("normalized judge artifact bytes drifted")
        if data and any(not line.strip() for line in data.splitlines()):
            raise JudgeExecutionError("normalized judge JSONL is empty or has blank lines")
        rows = [
            _strict_json_bytes(line, label="normalized judge result row")
            for line in data.splitlines()
        ]
        if (
            normalized.get("row_count") != len(rows)
            or [row.get("case_id") for row in rows] != successful_case_ids
            or normalized.get("ordered_output_sha256")
            != canonical_sha256([canonical_sha256(row) for row in rows])
        ):
            raise JudgeExecutionError("normalized judge result order or hash drifted")
        for row in rows:
            if set(row) != {"case_id", "output"}:
                raise JudgeExecutionError("normalized judge result row schema drifted")
            _validate_output(row.get("output"), category_policy=category_policy)
        if rows != replayed_success_rows:
            raise JudgeExecutionError(
                "normalized judge results differ from raw response replay"
            )
        if (
            expect_failed
            and value["failure"].get("reason") != "budget_fuse_blocked"
            and replayed_terminal_error
            != (
                str(value["failure"]["error_type"]),
                str(value["failure"]["error_message_sha256"]),
            )
        ):
            raise JudgeExecutionError(
                "failed judge terminal error differs from raw response replay"
            )
        row_offset = 0
        for attempt in attempts:
            if attempt["status"] != "success":
                continue
            count = len(attempt["case_ids"])
            batch_rows = rows[row_offset : row_offset + count]
            if attempt.get("output_sha256") != canonical_sha256(batch_rows):
                raise JudgeExecutionError("judge attempt output hash drifted")
            row_offset += count
    return value


def validate_execution_receipt(
    receipt: Mapping[str, Any] | str | Path,
    case_manifest: Mapping[str, Any] | str | Path,
    expected_judge: str,
    *,
    artifact_root: str | Path | None = None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Validate one completed judge pass and all locatable artifact bytes."""

    return _validate_receipt(
        receipt,
        case_manifest,
        expected_judge,
        artifact_root=artifact_root,
        allow_fixture=allow_fixture,
        expect_failed=False,
    )


def validate_failed_execution_receipt(
    receipt: Mapping[str, Any] | str | Path,
    case_manifest: Mapping[str, Any] | str | Path,
    expected_judge: str,
    *,
    artifact_root: str | Path | None = None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Validate one failed judge pass without making it publication-eligible."""

    return _validate_receipt(
        receipt,
        case_manifest,
        expected_judge,
        artifact_root=artifact_root,
        allow_fixture=allow_fixture,
        expect_failed=True,
    )


@_serialized_judge_run_mutation
def import_judge_pass(
    *,
    source_receipt: str | Path,
    case_manifest: Mapping[str, Any] | str | Path,
    expected_judge: str,
    output_root: str | Path,
    allow_fixture: bool = False,
) -> Path:
    """Import a byte-identical execution tree after full receipt replay."""

    manifest, _input_bytes = _validate_case_manifest(
        _load_object(case_manifest, label="judge case manifest")
    )
    if not allow_fixture:
        raise JudgeExecutionError(
            "production judge imports are disabled; execute in the canonical run root"
        )
    if expected_judge not in JUDGE_FAMILIES:
        raise JudgeExecutionError("judge family must be exactly claude or gpt")
    source_path = Path(source_receipt).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise JudgeExecutionError("source judge receipt must be a regular file")
    source_path = source_path.resolve()
    receipt = validate_execution_receipt(
        source_path,
        manifest,
        expected_judge,
        allow_fixture=allow_fixture,
    )
    source_root = source_path.parent
    execution_id = str(receipt["identity"]["execution_id"])
    try:
        source_run_root = source_path.parents[3]
    except IndexError as exc:
        raise JudgeExecutionError("source judge receipt is outside a fixed run root") from exc
    expected_source_relative = (
        Path("executions")
        / expected_judge
        / execution_id
        / "execution_receipt.json"
    )
    if source_path.relative_to(source_run_root) != expected_source_relative:
        raise JudgeExecutionError(
            "source judge receipt must use executions/<family>/<execution_id>"
        )
    source_case_path = source_run_root / "case_manifest.json"
    source_authority_path = source_run_root / "budget_authority.json"
    if (
        not source_case_path.is_file()
        or source_case_path.is_symlink()
        or _load_object(source_case_path, label="source run case manifest") != manifest
        or not source_authority_path.is_file()
        or source_authority_path.is_symlink()
    ):
        raise JudgeExecutionError("source judge run authority is missing or drifted")
    source_authority = load_judge_budget_authority(
        source_run_root,
        case_manifest=manifest,
        allow_fixture=allow_fixture,
    )
    if receipt.get("budget_authority") != _budget_receipt_binding(source_authority):
        raise JudgeExecutionError(
            "source judge receipt differs from its run budget authority"
        )
    expected_files = {
        "execution_receipt.json",
        str(receipt["normalized_results"]["path"]),
        *(str(row["path"]) for row in receipt["raw_artifacts"]),
    }
    observed_files: set[str] = set()
    for current_value, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        current = Path(current_value)
        for name in directory_names:
            if (current / name).is_symlink():
                raise JudgeExecutionError("source judge tree contains an unbound symlink")
        for name in file_names:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise JudgeExecutionError("source judge tree contains an unsafe entry")
            observed_files.add(path.relative_to(source_root).as_posix())
    if observed_files != expected_files:
        raise JudgeExecutionError("source judge tree contains unbound or missing files")
    root = _judge_root(output_root)
    _accrued_judge_accounting(root)
    _assert_next_family_order(
        root, manifest=manifest, judge_family=expected_judge
    )
    _assert_family_slot_available(root, expected_judge)
    _bind_case_manifest(root, manifest)
    target_authority_path = root / "budget_authority.json"
    if not target_authority_path.exists():
        _write_new(target_authority_path, source_authority_path.read_bytes())
    elif (
        target_authority_path.is_symlink()
        or not target_authority_path.is_file()
        or target_authority_path.read_bytes() != source_authority_path.read_bytes()
    ):
        raise JudgeExecutionError(
            "judge import target uses a different run budget authority"
        )
    authority = load_judge_budget_authority(
        root,
        case_manifest=manifest,
        allow_fixture=allow_fixture,
    )
    if receipt.get("budget_authority") != _budget_receipt_binding(authority):
        raise JudgeExecutionError(
            "imported judge execution uses a different run budget authority"
        )
    destination_root = (
        root
        / "executions"
        / expected_judge
        / execution_id
    )
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise JudgeExecutionError("imported judge execution already exists") from exc
    for relative_value in sorted(expected_files - {"execution_receipt.json"}):
        relative = Path(relative_value)
        _write_new(destination_root / relative, (source_root / relative).read_bytes())
    destination_receipt = destination_root / "execution_receipt.json"
    _write_new(destination_receipt, source_path.read_bytes())
    validate_execution_receipt(
        destination_receipt,
        manifest,
        expected_judge,
        allow_fixture=allow_fixture,
    )
    return destination_receipt


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute, verify, or import one isolated Persona-v2 judge pass."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--case-manifest", required=True)
    execute_parser.add_argument("--output-root", required=True)
    execute_parser.add_argument(
        "--judge-family", required=True, choices=sorted(JUDGE_FAMILIES)
    )
    execute_parser.add_argument("--model", required=True)
    execute_parser.add_argument(
        "--transport", required=True, choices=["claude-cli", "codex-cli"]
    )
    execute_parser.add_argument("--binary", required=True)
    execute_parser.add_argument("--timeout-seconds", type=float, default=900.0)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--case-manifest", required=True)
    verify_parser.add_argument("--execution-receipt", required=True)
    verify_parser.add_argument(
        "--judge-family", required=True, choices=sorted(JUDGE_FAMILIES)
    )

    import_parser = commands.add_parser("import")
    import_parser.add_argument("--case-manifest", required=True)
    import_parser.add_argument("--source-receipt", required=True)
    import_parser.add_argument("--output-root", required=True)
    import_parser.add_argument(
        "--judge-family", required=True, choices=sorted(JUDGE_FAMILIES)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the production-only judge evidence CLI without exposing environment data."""

    arguments = _argument_parser().parse_args(argv)
    try:
        if arguments.command == "execute":
            transport_class = (
                ClaudeCLIJudgeTransport
                if arguments.transport == "claude-cli"
                else CodexCLIJudgeTransport
            )
            transport = transport_class(
                binary=arguments.binary,
                timeout_seconds=arguments.timeout_seconds,
            )
            receipt_path = execute_judge_pass(
                case_manifest=arguments.case_manifest,
                output_root=arguments.output_root,
                judge_family=arguments.judge_family,
                exact_model=arguments.model,
                transport=transport,
            )
            receipt = validate_execution_receipt(
                receipt_path,
                arguments.case_manifest,
                arguments.judge_family,
            )
        elif arguments.command == "verify":
            receipt_path = Path(arguments.execution_receipt)
            preview = _load_object(receipt_path, label="judge execution receipt")
            validator = (
                validate_failed_execution_receipt
                if preview.get("schema_version") == FAILED_RECEIPT_SCHEMA
                else validate_execution_receipt
            )
            receipt = validator(
                receipt_path, arguments.case_manifest, arguments.judge_family
            )
        else:
            receipt_path = import_judge_pass(
                source_receipt=arguments.source_receipt,
                case_manifest=arguments.case_manifest,
                expected_judge=arguments.judge_family,
                output_root=arguments.output_root,
            )
            receipt = validate_execution_receipt(
                receipt_path,
                arguments.case_manifest,
                arguments.judge_family,
            )
    except JudgePassFailed as exc:
        receipt_path = exc.receipt_path
        receipt = exc.receipt
        print(
            canonical_json_bytes(
                {
                    "execution_receipt_path": str(receipt_path.resolve()),
                    "execution_receipt_sha256": exc.receipt_sha256,
                    "status": "failed",
                    "judge_family": receipt["identity"]["judge_family"],
                    "requested_model": receipt["identity"]["requested_model"],
                    "transport": receipt["identity"]["transport"],
                    "needs_user": receipt["needs_user"],
                    "needs_user_reasons": receipt["needs_user_reasons"],
                }
            ).decode("utf-8")
        )
        print(f"judge execution failed: {exc}", file=sys.stderr)
        return 2
    except JudgeExecutionError as exc:
        print(f"judge execution error: {exc}", file=sys.stderr)
        return 2
    receipt_hash_field = (
        "failed_execution_receipt_sha256"
        if receipt.get("status") == "failed"
        else "execution_receipt_sha256"
    )
    print(
        canonical_json_bytes(
            {
                "execution_receipt_path": str(receipt_path.resolve()),
                "execution_receipt_sha256": receipt[receipt_hash_field],
                "status": receipt["status"],
                "judge_family": receipt["identity"]["judge_family"],
                "requested_model": receipt["identity"]["requested_model"],
                "transport": receipt["identity"]["transport"],
                "needs_user": bool(receipt.get("needs_user", False)),
                "needs_user_reasons": list(
                    receipt.get("needs_user_reasons") or []
                ),
            }
        ).decode("utf-8")
    )
    return 0


__all__ = [
    "BATCH_SIZE",
    "FAILED_RECEIPT_SCHEMA",
    "MAX_ATTEMPTS_PER_BATCH",
    "ClaudeCLIJudgeTransport",
    "CodexCLIJudgeTransport",
    "FixtureJudgeTransport",
    "JudgeExecutionError",
    "JudgeIsolationError",
    "JudgePassFailed",
    "JudgeSchemaError",
    "JudgeTransportError",
    "bind_prepared_judge_case_manifest",
    "build_judge_run_evidence_receipt",
    "canonical_json_bytes",
    "canonical_sha256",
    "execute_judge_pass",
    "import_judge_pass",
    "load_judge_budget_authority",
    "main",
    "mint_judge_budget_authority",
    "record_judge_family_disposition",
    "validate_execution_receipt",
    "validate_failed_execution_receipt",
    "validate_judge_run_evidence_receipt",
    "write_judge_run_evidence_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
