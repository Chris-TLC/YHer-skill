"""Execution-evidence contracts for isolated Persona-v2 judge passes."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading

import pytest


RUN_ID = "llm-personas-v2-dual"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _case_manifest(count: int = 2) -> dict[str, object]:
    cases = []
    for index in range(count):
        messages = [
            {
                "role": "user",
                "simulated": True,
                "content": f"Public chemistry case {index}",
            }
        ]
        cases.append(
            {
                "case_id": f"case-{index:03d}",
                "judge_messages": messages,
                "judge_input_sha256": _sha(messages),
            }
        )
    shared_input = b"".join(
        _canonical_bytes(
            {"case_id": case["case_id"], "messages": case["judge_messages"]}
        )
        + b"\n"
        for case in cases
    )
    question_field_whitelist = ["kind", "options", "stem_blocks", "stem_text"]
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/llm_sim_v2/judge_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    amendment_bytes = (
        Path(__file__).resolve().parents[1]
        / "experiments/llm_sim_v2/judge_amendment_20260716.md"
    ).read_bytes()
    manifest: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.judge_case_manifest.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "analysis_population": "main",
        "target_labels_exported": False,
        "target_metadata_exported": False,
        "provider_identity_exported": False,
        "question_field_whitelist": question_field_whitelist,
        "judge_protocol": protocol,
        "judge_protocol_sha256": _sha(protocol),
        "judge_amendment": {
            "path": "experiments/llm_sim_v2/judge_amendment_20260716.md",
            "sha256": hashlib.sha256(amendment_bytes).hexdigest(),
            "size": len(amendment_bytes),
        },
        "selected_count": count,
        "cases": cases,
        "shared_input_sha256": hashlib.sha256(shared_input).hexdigest(),
    }
    manifest["case_manifest_sha256"] = _sha(manifest)
    return manifest


def _output(case_id: str, label: str = "consistent") -> dict[str, object]:
    return {
        "case_id": case_id,
        "output": {
            "label": label,
            "error_category": "none",
            "rationale": "The selected answer follows the visible reasoning.",
            "simulated": True,
        },
    }


def _transport_response(
    case_ids: list[str],
    *,
    model: str = "fixture-gpt-exact",
) -> dict[str, object]:
    return {
        "schema_version": "yher.llm_sim_v2.judge_transport_response.v2",
        "simulated": True,
        "transport_reported_models": [model],
        "transport_reported_model_source": "fixture_response",
        "transport_request_id": "fixture-request-001",
        "results": [_output(case_id) for case_id in case_ids],
        "usage": {"input_tokens": 101, "output_tokens": 43},
        "billing": {
            "known_cost_yuan": 0.125,
            "unknown_cost_reserve_yuan": 0.0,
        },
        "tool_calls": [],
    }


def _self_hash(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field)
    return _sha(payload)


def _rehash_case_manifest(manifest: dict[str, object]) -> None:
    cases = manifest["cases"]  # type: ignore[assignment]
    for case in cases:  # type: ignore[union-attr]
        case["judge_input_sha256"] = _sha(case["judge_messages"])
    shared_input = b"".join(
        _canonical_bytes(
            {"case_id": case["case_id"], "messages": case["judge_messages"]}
        )
        + b"\n"
        for case in cases  # type: ignore[union-attr]
    )
    manifest["shared_input_sha256"] = hashlib.sha256(shared_input).hexdigest()
    manifest.pop("case_manifest_sha256", None)
    manifest["case_manifest_sha256"] = _sha(manifest)


def _rehash_receipt(receipt: dict[str, object]) -> None:
    receipt.pop("execution_receipt_sha256", None)
    receipt["execution_receipt_sha256"] = _sha(receipt)


def _rehash_failed_receipt(receipt: dict[str, object]) -> None:
    receipt.pop("failed_execution_receipt_sha256", None)
    receipt["failed_execution_receipt_sha256"] = _sha(receipt)


def _rewrite_raw_attempt(
    receipt_path: Path,
    receipt: dict[str, object],
    attempt_index: int,
    *,
    request: dict[str, object] | None = None,
    response: dict[str, object] | None = None,
) -> None:
    attempt = receipt["attempts"][attempt_index]  # type: ignore[index]
    binding = attempt["raw_artifact"]
    raw_path = receipt_path.parent / binding["path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if request is not None:
        request_bytes = _canonical_bytes(request)
        raw["request_base64"] = base64.b64encode(request_bytes).decode("ascii")
        raw["request_sha256"] = hashlib.sha256(request_bytes).hexdigest()
        attempt["request_sha256"] = raw["request_sha256"]
    if response is not None:
        response_bytes = _canonical_bytes(response)
        raw["raw_outer_response_base64"] = base64.b64encode(response_bytes).decode(
            "ascii"
        )
        raw["raw_outer_response_sha256"] = hashlib.sha256(response_bytes).hexdigest()
        attempt["raw_outer_response_sha256"] = raw["raw_outer_response_sha256"]
    raw_bytes = _canonical_bytes(raw) + b"\n"
    raw_path.write_bytes(raw_bytes)
    rewritten_binding = {
        "path": binding["path"],
        "bytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    attempt["raw_artifact"] = rewritten_binding
    receipt["raw_artifacts"][attempt_index] = rewritten_binding  # type: ignore[index]
    attempt.pop("attempt_sha256")
    attempt["attempt_sha256"] = _sha(attempt)
    receipt["attempt_set_sha256"] = _sha(
        [
            {
                "attempt_id": row["attempt_id"],
                "attempt_sha256": row["attempt_sha256"],
            }
            for row in receipt["attempts"]  # type: ignore[union-attr]
        ]
    )
    receipt["raw_artifact_set_sha256"] = _sha(receipt["raw_artifacts"])
    _rehash_receipt(receipt)


def test_fixture_execution_writes_bound_immutable_evidence(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest()
    case_ids = [str(row["case_id"]) for row in manifest["cases"]]  # type: ignore[index]
    transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(case_ids)]
    )

    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=transport,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    execution_root = receipt_path.parent
    assert receipt_path.name == "execution_receipt.json"
    assert receipt["schema_version"] == "yher.llm_sim_v2.judge_execution_receipt.v2"
    assert receipt["status"] == "complete"
    assert receipt["case_binding"]["case_manifest_sha256"] == manifest[
        "case_manifest_sha256"
    ]
    assert receipt["case_binding"]["shared_input_sha256"] == manifest[
        "shared_input_sha256"
    ]
    assert receipt["identity"]["judge_family"] == "gpt"
    assert receipt["identity"]["requested_model"] == "fixture-gpt-exact"
    assert receipt["identity"]["transport"] == "fixture"
    assert receipt["isolation"] == {
        "fresh_execution": True,
        "resumed_session": False,
        "prior_conversation_context": False,
        "tools_disabled": True,
        "tools_used": False,
        "no_external_case_data": True,
        "raw_environment_exported": False,
        "secrets_exported": False,
    }
    assert receipt["policy"] == {
        "batch_size": 10,
        "max_attempts_per_batch": 2,
        "retryable_errors": ["schema_error", "transport_error"],
        "content_retry_allowed": False,
    }
    assert len(receipt["attempts"]) == 1
    attempt = receipt["attempts"][0]
    assert attempt["status"] == "success"
    assert attempt["attempt_sha256"] == _self_hash(attempt, "attempt_sha256")
    raw_path = execution_root / attempt["raw_artifact"]["path"]
    assert raw_path.is_file()
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == attempt["raw_artifact"][
        "sha256"
    ]
    normalized = execution_root / receipt["normalized_results"]["path"]
    assert normalized.is_file()
    assert [json.loads(line) for line in normalized.read_text().splitlines()] == [
        _output(case_id) for case_id in case_ids
    ]
    assert receipt["accounting"] == {
        "request_count": 1,
        "retry_count": 0,
        "transport_error_count": 0,
        "schema_error_count": 0,
        "content_retry_count": 0,
        "input_tokens": 101,
        "output_tokens": 43,
        "known_cost_yuan": 0.125,
        "unknown_cost_reserve_yuan": 0.0,
        "accounted_cost_yuan": 0.125,
    }
    assert receipt["execution_receipt_sha256"] == _self_hash(
        receipt, "execution_receipt_sha256"
    )
    assert judge_execution.validate_execution_receipt(
        receipt_path, manifest, "gpt", allow_fixture=True
    )["execution_receipt_sha256"] == receipt["execution_receipt_sha256"]


def test_retries_only_schema_and_transport_errors_and_keeps_every_attempt(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=11)
    case_ids = [str(row["case_id"]) for row in manifest["cases"]]  # type: ignore[index]
    bad_schema = _transport_response(case_ids[:10])
    bad_schema["results"] = bad_schema["results"][:-1]  # type: ignore[index]
    transport = judge_execution.FixtureJudgeTransport(
        [
            bad_schema,
            _transport_response(case_ids[:10]),
            judge_execution.JudgeTransportError(
                "fixture timeout",
                unknown_cost_reserve_yuan=0.25,
            ),
            _transport_response(case_ids[10:]),
        ]
    )

    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=transport,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    attempts = receipt["attempts"]
    assert [row["status"] for row in attempts] == [
        "schema_error",
        "success",
        "transport_error",
        "success",
    ]
    assert [row["attempt_number"] for row in attempts] == [1, 2, 1, 2]
    assert receipt["ordered_attempt_ids"] == [row["attempt_id"] for row in attempts]
    assert len(receipt["raw_artifacts"]) == 4
    for attempt in attempts:
        raw_path = receipt_path.parent / attempt["raw_artifact"]["path"]
        assert raw_path.is_file()
        assert attempt["attempt_sha256"] == _self_hash(attempt, "attempt_sha256")
    assert receipt["accounting"] == {
        "request_count": 4,
        "retry_count": 2,
        "transport_error_count": 1,
        "schema_error_count": 1,
        "content_retry_count": 0,
        "input_tokens": 303,
        "output_tokens": 129,
        "known_cost_yuan": 0.375,
        "unknown_cost_reserve_yuan": 0.25,
        "accounted_cost_yuan": 0.625,
    }
    rows = [
        json.loads(line)
        for line in (receipt_path.parent / "normalized_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["case_id"] for row in rows] == case_ids


def test_execution_rejects_rehashed_nested_target_metadata_before_transport(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    case = manifest["cases"][0]  # type: ignore[index]
    case["judge_messages"][0]["content"] = json.dumps(  # type: ignore[index]
        {
            "public_question": {
                "kind": "mcq",
                "stem_blocks": [],
                "stem_text": "Allowed public text",
                "options": {"A": "one", "B": "two"},
                "nodes": ["private-target-node"],
            }
        },
        ensure_ascii=False,
    )
    _rehash_case_manifest(manifest)
    transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )

    with pytest.raises(judge_execution.JudgeExecutionError, match="target|metadata|leak"):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    assert transport.requests == []
    assert not (tmp_path / "judge").exists()


def test_execution_rejects_self_consistent_alternative_judge_protocol(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    protocol = manifest["judge_protocol"]  # type: ignore[assignment]
    protocol["label_definitions"]["consistent"] += " Altered after freeze."  # type: ignore[index]
    manifest["judge_protocol_sha256"] = _sha(protocol)
    manifest.pop("case_manifest_sha256")
    manifest["case_manifest_sha256"] = _sha(manifest)
    transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )

    with pytest.raises(judge_execution.JudgeExecutionError, match="protocol|frozen"):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    assert transport.requests == []


@pytest.mark.parametrize("mutation", ["missing", "hash", "size", "path"])
def test_execution_requires_exact_outcome_blind_amendment_binding(
    tmp_path: Path, mutation: str
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    if mutation == "missing":
        manifest.pop("judge_amendment")
    elif mutation == "hash":
        manifest["judge_amendment"]["sha256"] = "0" * 64  # type: ignore[index]
    elif mutation == "size":
        manifest["judge_amendment"]["size"] = 1  # type: ignore[index]
    else:
        manifest["judge_amendment"]["path"] = "elsewhere.md"  # type: ignore[index]
    manifest.pop("case_manifest_sha256")
    manifest["case_manifest_sha256"] = _sha(manifest)
    transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )

    with pytest.raises(judge_execution.JudgeExecutionError, match="amendment"):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    assert transport.requests == []


@pytest.mark.parametrize("mutation", ["isolation", "accounting", "model"])
def test_receipt_validation_rejects_cross_field_forgery(
    tmp_path: Path, mutation: str
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=transport,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "isolation":
        receipt["isolation"]["tools_used"] = True
    elif mutation == "accounting":
        receipt["accounting"]["request_count"] = 999
    else:
        receipt["identity"]["requested_model"] = "different-model"
    _rehash_receipt(receipt)

    with pytest.raises(judge_execution.JudgeExecutionError):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


def test_model_drift_is_an_isolation_failure_and_is_never_retried(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    transport = judge_execution.FixtureJudgeTransport(
        [
            _transport_response(["case-000"], model="wrong-model"),
            _transport_response(["case-000"]),
        ]
    )

    with pytest.raises(judge_execution.JudgeExecutionError, match="model|isolation"):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    assert len(transport.requests) == 1


def test_valid_unknown_content_is_accepted_without_content_retry(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    response = _transport_response(["case-000"])
    response["results"][0]["output"] = {  # type: ignore[index]
        "label": "unknown",
        "error_category": "not_applicable",
        "rationale": "The visible response cannot be classified reliably.",
        "simulated": True,
    }
    transport = judge_execution.FixtureJudgeTransport([response])

    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=transport,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["accounting"]["request_count"] == 1
    assert receipt["accounting"]["retry_count"] == 0
    assert receipt["accounting"]["content_retry_count"] == 0


def test_cli_commands_pin_model_fresh_session_and_tools_disabled() -> None:
    from experiments.llm_sim_v2 import judge_execution

    codex = judge_execution.CodexCLIJudgeTransport(
        binary="/definitely/missing/codex",
    )
    codex_argv = codex.command_argv("gpt-exact-model")
    assert codex_argv[:2] == ["/definitely/missing/codex", "exec"]
    for flag in (
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--json",
    ):
        assert flag in codex_argv
    assert codex_argv[codex_argv.index("--model") + 1] == "gpt-exact-model"
    assert codex_argv[codex_argv.index("--sandbox") + 1] == "read-only"
    assert 'tools.state="limited"' not in codex_argv
    assert "tools.enabled_tools=[]" not in codex_argv
    assert "mcp_servers={}" in codex_argv
    for feature in (
        "shell_tool",
        "unified_exec",
        "unified_exec_zsh_fork",
        "shell_zsh_fork",
        "apply_patch_freeform",
        "browser_use",
        "in_app_browser",
        "computer_use",
        "multi_agent",
        "remote_plugin",
        "workspace_dependencies",
    ):
        assert ["--disable", feature] == codex_argv[
            codex_argv.index(feature) - 1 : codex_argv.index(feature) + 1
        ]

    claude = judge_execution.ClaudeCLIJudgeTransport(
        binary="/definitely/missing/claude",
    )
    claude_argv = claude.command_argv("claude-exact-model")
    assert claude_argv[0] == "/definitely/missing/claude"
    assert "-p" in claude_argv
    assert claude_argv[claude_argv.index("--model") + 1] == "claude-exact-model"
    assert claude_argv[claude_argv.index("--tools") + 1] == ""
    assert "--no-session-persistence" in claude_argv
    assert claude_argv[claude_argv.index("--output-format") + 1] == "json"
    assert "--strict-mcp-config" in claude_argv
    assert not any("API_KEY" in part or "=" in part and "KEY" in part for part in [
        *codex_argv,
        *claude_argv,
    ])


@pytest.mark.parametrize(
    ("transport_name", "factory"),
    [
        (
            "codex",
            lambda module: module.CodexCLIJudgeTransport(
                binary="/definitely/missing/codex",
            ),
        ),
        (
            "claude",
            lambda module: module.ClaudeCLIJudgeTransport(
                binary="/definitely/missing/claude",
            ),
        ),
    ],
)
def test_cli_transport_fails_closed_when_binary_is_unavailable(
    transport_name: str, factory: object
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    transport = factory(judge_execution)  # type: ignore[operator]
    with pytest.raises(judge_execution.JudgeExecutionError, match="unavailable"):
        transport.invoke(
            b"{}",
            exact_model=f"{transport_name}-exact-model",
            attempt_id="00000000-0000-4000-8000-000000000000",
        )


def test_unavailable_production_binary_fails_before_execution_directory(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    _authority, _ledger, output = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=0.0
    )
    with pytest.raises(judge_execution.JudgeExecutionError, match="unavailable"):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=output,
            judge_family="gpt",
            exact_model="gpt-exact-model",
            transport=judge_execution.CodexCLIJudgeTransport(
                binary="/definitely/missing/codex",
            ),
        )

    assert not (output / "executions").exists()


def test_import_copies_only_a_fully_validated_isolated_execution(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=2)
    source_receipt = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "source",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000", "case-001"])]
        ),
    )

    imported_receipt = judge_execution.import_judge_pass(
        source_receipt=source_receipt,
        case_manifest=manifest,
        expected_judge="gpt",
        output_root=tmp_path / "imported",
        allow_fixture=True,
    )

    assert imported_receipt.read_bytes() == source_receipt.read_bytes()
    assert imported_receipt.parent.name == source_receipt.parent.name
    source_files = {
        path.relative_to(source_receipt.parent): path.read_bytes()
        for path in source_receipt.parent.rglob("*")
        if path.is_file()
    }
    imported_files = {
        path.relative_to(imported_receipt.parent): path.read_bytes()
        for path in imported_receipt.parent.rglob("*")
        if path.is_file()
    }
    assert imported_files == source_files
    assert judge_execution.validate_execution_receipt(
        imported_receipt, manifest, "gpt", allow_fixture=True
    )["execution_receipt_sha256"]

    (source_receipt.parent / "unbound.txt").write_text("not evidence", encoding="utf-8")
    with pytest.raises(judge_execution.JudgeExecutionError, match="unbound|tree"):
        judge_execution.import_judge_pass(
            source_receipt=source_receipt,
            case_manifest=manifest,
            expected_judge="gpt",
            output_root=tmp_path / "second-import",
            allow_fixture=True,
        )


def test_production_import_is_disabled_before_target_mutation(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    source_receipt = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "source",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    target = tmp_path / "production-import"

    with pytest.raises(judge_execution.JudgeExecutionError, match="production.*disabled"):
        judge_execution.import_judge_pass(
            source_receipt=source_receipt,
            case_manifest=manifest,
            expected_judge="gpt",
            output_root=target,
        )

    assert not target.exists()


def test_execution_rejects_symlink_output_root_before_any_transport(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "judge-link"
    output.symlink_to(target, target_is_directory=True)
    transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )

    with pytest.raises(judge_execution.JudgeExecutionError, match="symlink"):
        judge_execution.execute_judge_pass(
            case_manifest=_case_manifest(count=1),
            output_root=output,
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    assert transport.requests == []
    assert list(target.iterdir()) == []


def test_raw_cli_stdout_binding_is_validated_before_accepting_results(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    response = _transport_response(["case-000"])
    stdout = b'{"type":"item.completed"}\n'
    response["raw_transport"] = {
        "stdout_base64": base64.b64encode(stdout).decode("ascii"),
        "stdout_bytes": len(stdout),
        "stdout_sha256": "0" * 64,
        "stderr_bytes": 0,
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "returncode": 0,
        "environment_exported": False,
    }
    transport = judge_execution.FixtureJudgeTransport([response, response])

    with pytest.raises(judge_execution.JudgeExecutionError, match="raw|schema"):
        judge_execution.execute_judge_pass(
            case_manifest=_case_manifest(count=1),
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    assert len(transport.requests) == 2


def test_fixture_receipt_is_rejected_by_default_formal_validation(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )

    with pytest.raises(judge_execution.JudgeExecutionError, match="fixture|production"):
        judge_execution.validate_execution_receipt(receipt_path, manifest, "gpt")


def test_codex_cli_parser_builds_a_production_eligible_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    model = "gpt-exact-model"
    manifest = _case_manifest(count=1)
    _authority, _ledger, output_root = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=0.0
    )
    events = [
        {"type": "thread.started", "thread_id": "codex-thread-001"},
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": json.dumps(
                    {"results": [_output("case-000")]},
                    ensure_ascii=False,
                ),
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 77, "output_tokens": 31}},
    ]
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=b"".join(_canonical_bytes(event) + b"\n" for event in events),
        stderr=b"",
    )
    binary = shutil.which("codex")
    assert binary is not None
    transport = judge_execution.CodexCLIJudgeTransport(binary=binary)
    monkeypatch.setattr(transport, "preflight", lambda: None)
    monkeypatch.setattr(transport, "_run", lambda argv, prompt: completed)

    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=output_root,
        judge_family="gpt",
        exact_model=model,
        transport=transport,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["identity"]["transport"] == "codex_cli"
    assert receipt["accounting"]["input_tokens"] == 77
    assert receipt["accounting"]["output_tokens"] == 31
    assert receipt["accounting"]["known_cost_yuan"] == 0
    assert receipt["accounting"]["unknown_cost_reserve_yuan"] == 10
    assert judge_execution.validate_execution_receipt(
        receipt_path, manifest, "gpt"
    )["identity"]["requested_model"] == model


def test_claude_cli_parser_preserves_raw_reported_model_evidence() -> None:
    from experiments.llm_sim_v2 import judge_execution

    model = "claude-exact-model"
    outer = {
        "result": json.dumps(
            {"results": [_output("case-000")]},
            ensure_ascii=False,
        ),
        "session_id": "claude-session-001",
        "usage": {"input_tokens": 81, "output_tokens": 29},
        "modelUsage": {model: {"inputTokens": 81, "outputTokens": 29}},
    }
    response = judge_execution._claude_response_from_raw_cli(
        raw_transport=judge_execution._raw_cli_binding(
            stdout=_canonical_bytes(outer), stderr=b"", returncode=0
        ),
        exact_model=model,
        attempt_id="00000000-0000-4000-8000-000000000000",
        unknown_cost_reserve_yuan=10,
    )
    assert response["transport_reported_models"] == [model]
    assert response["transport_reported_model_source"] == "raw_cli.modelUsage_keys"
    assert response["usage"] == {"input_tokens": 81, "output_tokens": 29}


def test_claude_cli_rejects_multiple_transport_reported_model_identities() -> None:
    from experiments.llm_sim_v2 import judge_execution

    model = "claude-exact-model"
    outer = {
        "result": json.dumps(
            {"results": [_output("case-000")]},
            ensure_ascii=False,
        ),
        "session_id": "claude-session-001",
        "usage": {"input_tokens": 81, "output_tokens": 29},
        "modelUsage": {
            model: {"inputTokens": 81, "outputTokens": 29},
            "claude-unapproved-model": {"inputTokens": 1, "outputTokens": 1},
        },
    }
    response = judge_execution._claude_response_from_raw_cli(
        raw_transport=judge_execution._raw_cli_binding(
            stdout=_canonical_bytes(outer), stderr=b"", returncode=0
        ),
        exact_model=model,
        attempt_id="00000000-0000-4000-8000-000000000000",
        unknown_cost_reserve_yuan=10,
    )
    protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/llm_sim_v2/judge_protocol_v1.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="model|isolation|identity",
    ):
        judge_execution._validate_transport_response(
            _canonical_bytes(response),
            case_ids=["case-000"],
            exact_model=model,
            category_policy=protocol["label_category_policy"],
        )


def test_module_cli_verify_is_production_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    _authority, _ledger, output_root = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=0.0
    )
    case_path = tmp_path / "cases.json"
    case_path.write_bytes(_canonical_bytes(manifest) + b"\n")
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=output_root,
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )

    exit_code = judge_execution.main(
        [
            "verify",
            "--case-manifest",
            str(case_path),
            "--execution-receipt",
            str(receipt_path),
            "--judge-family",
            "gpt",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "fixture" in captured.err


def test_receipt_replay_binds_failed_retry_to_the_same_frozen_batch(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=2)
    bad = _transport_response(["case-000", "case-001"])
    bad["results"] = bad["results"][:-1]  # type: ignore[index]
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [bad, _transport_response(["case-000", "case-001"])]
        ),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["attempts"][0]["case_ids"] = ["unrelated-case"]
    first = receipt["attempts"][0]
    first.pop("attempt_sha256")
    first["attempt_sha256"] = _sha(first)
    receipt["attempt_set_sha256"] = _sha(
        [
            {
                "attempt_id": row["attempt_id"],
                "attempt_sha256": row["attempt_sha256"],
            }
            for row in receipt["attempts"]
        ]
    )
    _rehash_receipt(receipt)

    with pytest.raises(judge_execution.JudgeExecutionError, match="batch|case"):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


def test_receipt_replays_raw_success_into_the_normalized_rows(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    changed = _transport_response(["case-000"])
    changed["results"][0]["output"]["rationale"] = "A different valid rationale."  # type: ignore[index]
    _rewrite_raw_attempt(receipt_path, receipt, 0, response=changed)

    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="raw|normalized|output|replay",
    ):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


def test_receipt_cannot_relabel_a_valid_first_response_as_retryable(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=2)
    invalid = _transport_response(["case-000", "case-001"])
    invalid["results"] = invalid["results"][:-1]  # type: ignore[index]
    valid = _transport_response(["case-000", "case-001"])
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport([invalid, valid]),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["attempts"][0]["status"] == "schema_error"
    _rewrite_raw_attempt(receipt_path, receipt, 0, response=valid)

    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="retry|status|raw|replay",
    ):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


def test_receipt_reconstructs_the_exact_frozen_batch_request(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw_path = receipt_path.parent / receipt["raw_artifacts"][0]["path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    request = json.loads(base64.b64decode(raw["request_base64"]))
    request["requested_model"] = "substituted-model"
    _rewrite_raw_attempt(receipt_path, receipt, 0, request=request)

    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="request|batch|frozen",
    ):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("case_manifest_schema_version", "forged.schema"),
        ("case_count", 999),
        ("ordered_case_ids_sha256", "0" * 64),
    ],
)
def test_receipt_rejects_rehashed_case_binding_fields(
    tmp_path: Path, field: str, replacement: object
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["case_binding"][field] = replacement
    _rehash_receipt(receipt)

    with pytest.raises(judge_execution.JudgeExecutionError, match="binding|identity"):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


def test_receipt_rejects_rehashed_attempt_timestamps_outside_the_pass(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    attempt = receipt["attempts"][0]
    attempt["started_at_utc"] = "1999-01-01T00:00:00Z"
    attempt.pop("attempt_sha256")
    attempt["attempt_sha256"] = _sha(attempt)
    receipt["attempt_set_sha256"] = _sha(
        [{"attempt_id": attempt["attempt_id"], "attempt_sha256": attempt["attempt_sha256"]}]
    )
    _rehash_receipt(receipt)

    with pytest.raises(judge_execution.JudgeExecutionError, match="timestamp|time"):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


def test_receipt_validation_rejects_symlink_or_unbound_execution_tree(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    alias = tmp_path / "alias"
    alias.symlink_to(receipt_path.parent, target_is_directory=True)
    with pytest.raises(judge_execution.JudgeExecutionError, match="symlink|artifact root"):
        judge_execution.validate_execution_receipt(
            receipt_path,
            manifest,
            "gpt",
            artifact_root=alias,
            allow_fixture=True,
        )

    (receipt_path.parent / "unbound.txt").write_text("unbound", encoding="utf-8")
    with pytest.raises(judge_execution.JudgeExecutionError, match="unbound|file set|tree"):
        judge_execution.validate_execution_receipt(
            receipt_path,
            manifest,
            "gpt",
            allow_fixture=True,
        )


def test_transport_error_reserve_is_replayed_from_the_raw_failure(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=11)
    case_ids = [str(row["case_id"]) for row in manifest["cases"]]  # type: ignore[index]
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [
                _transport_response(case_ids[:10]),
                judge_execution.JudgeTransportError(
                    "fixture timeout",
                    unknown_cost_reserve_yuan=0.25,
                ),
                _transport_response(case_ids[10:]),
            ]
        ),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    failed = receipt["attempts"][1]
    assert failed["status"] == "transport_error"
    failed["unknown_cost_reserve_yuan"] = 0
    failed.pop("attempt_sha256")
    failed["attempt_sha256"] = _sha(failed)
    receipt["attempt_set_sha256"] = _sha(
        [
            {
                "attempt_id": row["attempt_id"],
                "attempt_sha256": row["attempt_sha256"],
            }
            for row in receipt["attempts"]
        ]
    )
    receipt["accounting"]["unknown_cost_reserve_yuan"] = 0
    receipt["accounting"]["accounted_cost_yuan"] = receipt["accounting"][
        "known_cost_yuan"
    ]
    _rehash_receipt(receipt)

    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="transport|accounting|reserve|raw",
    ):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


@pytest.mark.parametrize("target", ["receipt", "identity", "attempt"])
def test_receipt_v1_rejects_rehashed_undeclared_fields(
    tmp_path: Path, target: str
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path / "judge",
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if target == "receipt":
        receipt["undeclared"] = True
    elif target == "identity":
        receipt["identity"]["undeclared"] = True
    else:
        attempt = receipt["attempts"][0]
        attempt["undeclared"] = True
        attempt.pop("attempt_sha256")
        attempt["attempt_sha256"] = _sha(attempt)
        receipt["attempt_set_sha256"] = _sha(
            [
                {
                    "attempt_id": attempt["attempt_id"],
                    "attempt_sha256": attempt["attempt_sha256"],
                }
            ]
        )
    _rehash_receipt(receipt)

    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="schema|field|identity|lifecycle",
    ):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


@pytest.mark.parametrize("judge", ["gpt"])
def test_production_envelope_is_replayed_from_raw_cli_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    judge: str,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    _authority, _ledger, output_root = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=0.0
    )
    model = f"{judge}-exact-model"
    if judge == "gpt":
        stdout = b"".join(
            _canonical_bytes(event) + b"\n"
            for event in [
                {"type": "thread.started", "thread_id": "codex-thread-001"},
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": json.dumps(
                            {"results": [_output("case-000")]},
                            ensure_ascii=False,
                        ),
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 77, "output_tokens": 31},
                },
            ]
        )
        binary = shutil.which("codex")
        assert binary is not None
        transport = judge_execution.CodexCLIJudgeTransport(binary=binary)
    else:
        stdout = _canonical_bytes(
            {
                "result": json.dumps(
                    {"results": [_output("case-000")]}, ensure_ascii=False
                ),
                "session_id": "claude-session-001",
                "usage": {"input_tokens": 81, "output_tokens": 29},
                "modelUsage": {model: {"inputTokens": 81, "outputTokens": 29}},
            }
        )
        transport = judge_execution.ClaudeCLIJudgeTransport(
            binary="/not-invoked/claude"
        )
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=b""
    )
    monkeypatch.setattr(transport, "preflight", lambda: None)
    monkeypatch.setattr(transport, "_run", lambda argv, prompt: completed)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=output_root,
        judge_family=judge,
        exact_model=model,
        transport=transport,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw_path = receipt_path.parent / receipt["raw_artifacts"][0]["path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    response = json.loads(base64.b64decode(raw["raw_outer_response_base64"]))
    response["results"][0]["output"]["rationale"] = "Envelope-only substitution."
    _rewrite_raw_attempt(receipt_path, receipt, 0, response=response)

    rows = response["results"]
    normalized_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    normalized_path = receipt_path.parent / receipt["normalized_results"]["path"]
    normalized_path.write_bytes(normalized_bytes)
    receipt["normalized_results"] = {
        "path": "normalized_results.jsonl",
        "bytes": len(normalized_bytes),
        "sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "row_count": 1,
        "ordered_output_sha256": _sha([_sha(row) for row in rows]),
    }
    attempt = receipt["attempts"][0]
    attempt["output_sha256"] = _sha(rows)
    attempt.pop("attempt_sha256")
    attempt["attempt_sha256"] = _sha(attempt)
    receipt["attempt_set_sha256"] = _sha(
        [{"attempt_id": attempt["attempt_id"], "attempt_sha256": attempt["attempt_sha256"]}]
    )
    _rehash_receipt(receipt)

    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="CLI|stdout|transport|raw|envelope",
    ):
        judge_execution.validate_execution_receipt(
            receipt,
            manifest,
            judge,
            artifact_root=receipt_path.parent,
        )


def test_schema_exhaustion_writes_a_verifiable_failed_execution_receipt(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    invalid = _transport_response(["case-000"])
    invalid["results"] = []
    with pytest.raises(judge_execution.JudgeExecutionError) as captured:
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=judge_execution.FixtureJudgeTransport([invalid, invalid]),
        )

    assert type(captured.value).__name__ == "JudgePassFailed"
    receipt_path = captured.value.receipt_path  # type: ignore[attr-defined]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_path.name == "failed_execution_receipt.json"
    assert receipt["status"] == "failed"
    assert receipt["failure"]["reason"] == "schema_retries_exhausted"
    assert receipt["failure"]["terminal_status"] == "schema_error"
    assert receipt["failure"]["completed_case_count"] == 0
    assert receipt["accounting"]["request_count"] == 2
    assert receipt["accounting"]["retry_count"] == 1
    assert receipt["accounting"]["schema_error_count"] == 2
    assert len(receipt["raw_artifacts"]) == 2
    assert (receipt_path.parent / "normalized_results.jsonl").read_bytes() == b""
    assert judge_execution.validate_failed_execution_receipt(
        receipt_path, manifest, "gpt", allow_fixture=True
    )["failed_execution_receipt_sha256"] == receipt[
        "failed_execution_receipt_sha256"
    ]
    with pytest.raises(judge_execution.JudgeExecutionError, match="failed|status|schema"):
        judge_execution.validate_execution_receipt(
            receipt_path, manifest, "gpt", allow_fixture=True
        )


def test_transport_exhaustion_binds_reserve_and_needs_user(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    errors = [
        judge_execution.JudgeTransportError(
            "timeout one", unknown_cost_reserve_yuan=0.25
        ),
        judge_execution.JudgeTransportError(
            "timeout two", unknown_cost_reserve_yuan=0.5
        ),
    ]
    with pytest.raises(judge_execution.JudgeExecutionError) as captured:
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=judge_execution.FixtureJudgeTransport(errors),
        )

    receipt = json.loads(  # type: ignore[attr-defined]
        captured.value.receipt_path.read_text(encoding="utf-8")
    )
    assert receipt["failure"]["reason"] == "transport_retries_exhausted"
    assert receipt["accounting"]["unknown_cost_reserve_yuan"] == 0.75
    assert receipt["accounting"]["accounted_cost_yuan"] == 0.75
    assert receipt["needs_user"] is True
    assert "unknown_judge_billing_reserved" in receipt["needs_user_reasons"]
    assert "judge_transport_retries_exhausted" in receipt["needs_user_reasons"]


def test_isolation_failure_writes_one_nonretryable_failed_receipt(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    drifted = _transport_response(["case-000"], model="wrong-model")
    with pytest.raises(judge_execution.JudgeExecutionError) as captured:
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=judge_execution.FixtureJudgeTransport([drifted]),
        )

    receipt = json.loads(  # type: ignore[attr-defined]
        captured.value.receipt_path.read_text(encoding="utf-8")
    )
    assert receipt["failure"]["reason"] == "isolation_failure"
    assert receipt["failure"]["terminal_status"] == "isolation_error"
    assert receipt["accounting"]["request_count"] == 1
    assert receipt["accounting"]["retry_count"] == 0
    assert len(receipt["attempts"]) == 1
    assert receipt["needs_user"] is True
    assert "judge_isolation_failure" in receipt["needs_user_reasons"]


def test_cli_failed_execute_prints_receipt_and_verify_accepts_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    case_path = tmp_path / "cases.json"
    case_path.write_bytes(_canonical_bytes(manifest) + b"\n")
    _authority, _ledger, output_root = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=0.0
    )
    events = [
        {"type": "thread.started", "thread_id": "codex-failed-thread"},
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": json.dumps({"results": []}),
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}},
    ]
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=b"".join(_canonical_bytes(event) + b"\n" for event in events),
        stderr=b"",
    )
    monkeypatch.setattr(
        judge_execution.CodexCLIJudgeTransport,
        "preflight",
        lambda self: None,
    )
    monkeypatch.setattr(
        judge_execution.CodexCLIJudgeTransport,
        "_run",
        lambda self, argv, prompt: completed,
    )
    binary = shutil.which("codex")
    assert binary is not None

    exit_code = judge_execution.main(
        [
            "execute",
            "--case-manifest",
            str(case_path),
            "--output-root",
            str(output_root),
            "--judge-family",
            "gpt",
            "--model",
            "gpt-exact-model",
            "--transport",
            "codex-cli",
            "--binary",
            binary,
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "failed"
    assert payload["needs_user"] is True
    receipt_path = Path(payload["execution_receipt_path"])
    assert receipt_path.is_file()
    assert payload["execution_receipt_sha256"] == json.loads(
        receipt_path.read_text(encoding="utf-8")
    )["failed_execution_receipt_sha256"]
    assert "failed" in captured.err

    verify_code = judge_execution.main(
        [
            "verify",
            "--case-manifest",
            str(case_path),
            "--execution-receipt",
            str(receipt_path),
            "--judge-family",
            "gpt",
        ]
    )
    verified = capsys.readouterr()
    assert verify_code == 0
    assert json.loads(verified.out)["status"] == "failed"


def test_failed_receipt_replays_terminal_error_identity_from_raw_bytes(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    drifted = _transport_response(["case-000"], model="wrong-model")
    with pytest.raises(judge_execution.JudgeExecutionError) as captured:
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=tmp_path / "judge",
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=judge_execution.FixtureJudgeTransport([drifted]),
        )
    receipt_path = captured.value.receipt_path  # type: ignore[attr-defined]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["failure"]["error_type"] = "SubstitutedIsolationError"
    receipt["failure"]["error_message_sha256"] = "0" * 64
    _rehash_failed_receipt(receipt)

    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="terminal|raw|error|failure",
    ):
        judge_execution.validate_failed_execution_receipt(
            receipt,
            manifest,
            "gpt",
            artifact_root=receipt_path.parent,
            allow_fixture=True,
        )


def test_fixed_judge_root_allows_only_one_execution_per_family(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    root = tmp_path / "judge-results"
    first = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=root,
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    execution_id = json.loads(first.read_text(encoding="utf-8"))["identity"][
        "execution_id"
    ]
    assert first.relative_to(root).as_posix() == (
        f"executions/gpt/{execution_id}/execution_receipt.json"
    )

    second_transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )
    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="family|slot|already|single",
    ):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=root,
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=second_transport,
        )
    assert second_transport.requests == []


def test_zero_case_execution_fails_without_poisoning_a_family_slot(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    root = tmp_path / "judge-results"
    transport = judge_execution.FixtureJudgeTransport([])
    with pytest.raises(judge_execution.JudgeExecutionError, match="zero-case"):
        judge_execution.execute_judge_pass(
            case_manifest=_case_manifest(count=0),
            output_root=root,
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    assert transport.requests == []
    assert not root.exists()


def test_claude_cannot_start_or_be_disposed_before_gpt_is_terminal(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    root = tmp_path / "judge-results"
    transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )
    with pytest.raises(judge_execution.JudgeExecutionError, match="GPT.*before Claude"):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=root,
            judge_family="claude",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )
    with pytest.raises(judge_execution.JudgeExecutionError, match="GPT.*before Claude"):
        judge_execution.record_judge_family_disposition(
            case_manifest=manifest,
            output_root=root,
            judge_family="claude",
            status="unavailable",
            reason_code="production_cli_unavailable",
        )

    assert transport.requests == []
    assert not root.exists()


def test_unsealed_execution_root_blocks_every_later_family_call(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    root = tmp_path / "judge-results"
    (root / "executions/gpt/crashed-execution").mkdir(parents=True)
    transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )

    with pytest.raises(judge_execution.JudgeExecutionError, match="unsealed"):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=root,
            judge_family="claude",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    assert transport.requests == []


def test_run_lock_serializes_gpt_then_claude_family_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    root = tmp_path / "judge-results"
    first = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )
    second = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_done = threading.Event()
    errors: list[BaseException] = []
    original_invoke = judge_execution.FixtureJudgeTransport.invoke

    def blocking_invoke(self, *args, **kwargs):
        if self is first:
            first_entered.set()
            if not release_first.wait(2):
                raise RuntimeError("test did not release the first judge pass")
        return original_invoke(self, *args, **kwargs)

    monkeypatch.setattr(
        judge_execution.FixtureJudgeTransport, "invoke", blocking_invoke
    )

    def run_gpt() -> None:
        try:
            judge_execution.execute_judge_pass(
                case_manifest=manifest,
                output_root=root,
                judge_family="gpt",
                exact_model="fixture-gpt-exact",
                transport=first,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def run_claude() -> None:
        second_started.set()
        try:
            judge_execution.execute_judge_pass(
                case_manifest=manifest,
                output_root=root,
                judge_family="claude",
                exact_model="fixture-gpt-exact",
                transport=second,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            second_done.set()

    gpt_thread = threading.Thread(target=run_gpt)
    claude_thread = threading.Thread(target=run_claude)
    gpt_thread.start()
    assert first_entered.wait(1)
    claude_thread.start()
    assert second_started.wait(1)
    assert not second_done.wait(0.1)
    assert second.requests == []
    release_first.set()
    gpt_thread.join(2)
    claude_thread.join(2)

    assert not gpt_thread.is_alive()
    assert not claude_thread.is_alive()
    assert errors == []
    assert len(first.requests) == 1
    assert len(second.requests) == 1


def test_run_evidence_receipt_binds_complete_and_unavailable_slots_and_exact_tree(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    root = tmp_path / "judge-results"
    completed = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=root,
        judge_family="gpt",
        exact_model="fixture-gpt-exact",
        transport=judge_execution.FixtureJudgeTransport(
            [_transport_response(["case-000"])]
        ),
    )
    unavailable = judge_execution.record_judge_family_disposition(
        case_manifest=manifest,
        output_root=root,
        judge_family="claude",
        status="unavailable",
        reason_code="production_cli_unavailable",
    )
    internal = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=manifest,
        output_root=root,
        allow_fixture=True,
    )
    receipt = json.loads(internal.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "yher.llm_sim_v2.judge_run_evidence_receipt.v1"
    assert receipt["family_slots"]["gpt"]["status"] == "complete"
    assert receipt["family_slots"]["claude"]["status"] == "unavailable"
    assert receipt["family_slots"]["gpt"]["receipt_sha256"] == json.loads(
        completed.read_text(encoding="utf-8")
    )["execution_receipt_sha256"]
    assert receipt["family_slots"]["claude"]["receipt_sha256"] == json.loads(
        unavailable.read_text(encoding="utf-8")
    )["family_disposition_receipt_sha256"]
    expected_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != internal
    )
    assert [row["path"] for row in receipt["artifact_tree"]["files"]] == expected_files
    assert receipt["judge_run_evidence_receipt_sha256"] == _self_hash(
        receipt, "judge_run_evidence_receipt_sha256"
    )
    assert judge_execution.validate_judge_run_evidence_receipt(
        internal,
        case_manifest=manifest,
        output_root=root,
        allow_fixture=True,
    )["judge_run_evidence_receipt_sha256"] == receipt[
        "judge_run_evidence_receipt_sha256"
    ]

    fixed_anchor = (
        tmp_path
        / "repo/experiments/llm_sim_v2/evidence_anchors/"
        "judge_run_evidence_receipt.json"
    )
    external = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=manifest,
        output_root=root,
        output=fixed_anchor,
        allow_fixture=True,
    )
    assert external == fixed_anchor
    assert external.read_bytes() == internal.read_bytes()


def test_run_evidence_receipt_keeps_failed_and_zero_case_dispositions(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    failed_manifest = _case_manifest(count=1)
    failed_root = tmp_path / "failed"
    invalid = _transport_response(["case-000"])
    invalid["results"] = []
    with pytest.raises(judge_execution.JudgePassFailed) as captured:
        judge_execution.execute_judge_pass(
            case_manifest=failed_manifest,
            output_root=failed_root,
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=judge_execution.FixtureJudgeTransport([invalid, invalid]),
        )
    judge_execution.record_judge_family_disposition(
        case_manifest=failed_manifest,
        output_root=failed_root,
        judge_family="claude",
        status="unavailable",
        reason_code="production_cli_unavailable",
    )
    failed_run_path = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=failed_manifest,
        output_root=failed_root,
        allow_fixture=True,
    )
    failed_run = json.loads(failed_run_path.read_text(encoding="utf-8"))
    assert failed_run["family_slots"]["gpt"]["status"] == "failed"
    assert failed_run["family_slots"]["gpt"]["receipt_sha256"] == (
        captured.value.receipt_sha256
    )
    assert failed_run["family_slots"]["gpt"]["accounting"] == (
        captured.value.receipt["accounting"]
    )

    zero_manifest = _case_manifest(count=0)
    zero_root = tmp_path / "zero"
    for family in ("claude", "gpt"):
        judge_execution.record_judge_family_disposition(
            case_manifest=zero_manifest,
            output_root=zero_root,
            judge_family=family,
            status="not_applicable_zero_cases",
            reason_code="selected_case_count_zero",
        )
    zero_run_path = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=zero_manifest,
        output_root=zero_root,
        allow_fixture=True,
    )
    zero_run = json.loads(zero_run_path.read_text(encoding="utf-8"))
    assert {
        family: slot["status"]
        for family, slot in zero_run["family_slots"].items()
    } == {
        "claude": "not_applicable_zero_cases",
        "gpt": "not_applicable_zero_cases",
    }
    assert not (zero_root / "executions").exists()


def _mint_test_budget_authority(
    tmp_path: Path,
    *,
    manifest: dict[str, object],
    baseline_yuan: float,
) -> tuple[Path, Path, Path]:
    from experiments.llm_sim_v2 import judge_execution

    repo = tmp_path / "repo"
    anchor = (
        repo
        / "experiments/llm_sim_v2/evidence_anchors/"
        "main_phase_evidence_receipt.json"
    )
    anchor.parent.mkdir(parents=True)
    phase_receipt: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.phase_evidence_receipt.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "main",
        "authority": "post_invocation_phase_receipt",
        "phase_provenance_sha256": "a" * 64,
        "phase_provenance_file_sha256": "b" * 64,
        "store_snapshot": {"snapshot_sha256": "c" * 64},
        "providers": {},
    }
    phase_receipt["phase_evidence_receipt_sha256"] = _sha(phase_receipt)
    anchor.write_bytes(_canonical_bytes(phase_receipt) + b"\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "judge-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Judge Test"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "bind formal collection receipt"],
        cwd=repo,
        check=True,
    )
    ledger_path = tmp_path / "run_budget_ledger.json"
    known = round(max(0.0, baseline_yuan - 1.0), 8)
    reserve = round(baseline_yuan - known, 8)
    ledger = {
        "schema_version": "yher.llm_sim_v2.run_budget_ledger.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "total_known_cost_yuan": known,
        "total_unknown_reserve_yuan": reserve,
        "total_accounted_cost_yuan": baseline_yuan,
        "hard_fuse_yuan": 450.0,
        "updated_at_utc": "2026-07-16T00:00:00Z",
    }
    ledger_path.write_bytes(_canonical_bytes(ledger) + b"\n")
    root = tmp_path / "judge-results"
    judge_execution.bind_prepared_judge_case_manifest(
        case_manifest=manifest,
        output_root=root,
    )
    authority_path = judge_execution.mint_judge_budget_authority(
        case_manifest=manifest,
        output_root=root,
        repo_root=repo,
        run_budget_ledger=ledger_path,
    )
    return authority_path, ledger_path, root


def test_production_reserve_is_fixed_and_not_caller_selected() -> None:
    from experiments.llm_sim_v2 import judge_execution

    codex = judge_execution.CodexCLIJudgeTransport(
        binary="/definitely/missing/codex"
    )
    claude = judge_execution.ClaudeCLIJudgeTransport(
        binary="/definitely/missing/claude"
    )
    assert codex.unknown_cost_reserve_yuan == 10
    assert claude.unknown_cost_reserve_yuan == 10
    with pytest.raises(TypeError):
        judge_execution.CodexCLIJudgeTransport(
            binary="/definitely/missing/codex",
            unknown_cost_reserve_yuan=1,
        )


def test_budget_authority_binds_committed_collection_receipt_and_live_ledger(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    authority_path, ledger_path, root = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=123.5
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    assert authority["schema_version"] == "yher.llm_sim_v2.judge_budget_authority.v1"
    assert authority["baseline_accounted_cost_yuan"] == 123.5
    assert authority["unknown_reserve_per_attempt_yuan"] == 10
    assert authority["hard_fuse_yuan"] == 450
    assert authority["formal_collection"]["anchor_relative_path"] == (
        "experiments/llm_sim_v2/evidence_anchors/"
        "main_phase_evidence_receipt.json"
    )
    assert authority["formal_collection"]["working_tree_matches_head_blob"] is True
    assert authority["judge_budget_authority_sha256"] == _self_hash(
        authority, "judge_budget_authority_sha256"
    )
    assert judge_execution.load_judge_budget_authority(
        root,
        case_manifest=manifest,
    )["judge_budget_authority_sha256"] == authority[
        "judge_budget_authority_sha256"
    ]

    ledger_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="budget|ledger|source|hash|changed",
    ):
        judge_execution.load_judge_budget_authority(
            root,
            case_manifest=manifest,
        )


def test_budget_fuse_blocks_before_every_call_and_preserves_prior_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    _authority, _ledger, root = _mint_test_budget_authority(
        tmp_path / "at-boundary",
        manifest=manifest,
        baseline_yuan=440.0,
    )
    blocked_transport = judge_execution.FixtureJudgeTransport(
        [_transport_response(["case-000"])]
    )
    with pytest.raises(judge_execution.JudgePassFailed) as blocked:
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=root,
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=blocked_transport,
        )
    assert blocked_transport.requests == []
    assert blocked.value.receipt["failure"]["reason"] == "budget_fuse_blocked"
    assert blocked.value.receipt["attempts"] == []
    assert blocked.value.receipt["accounting"]["request_count"] == 0
    assert blocked.value.receipt["accounting"]["unknown_cost_reserve_yuan"] == 0

    _authority, _ledger, retry_root = _mint_test_budget_authority(
        tmp_path / "before-retry",
        manifest=manifest,
        baseline_yuan=430.0,
    )
    binary = shutil.which("codex")
    assert binary is not None
    transport = judge_execution.CodexCLIJudgeTransport(binary=binary)
    monkeypatch.setattr(transport, "preflight", lambda: None)
    calls = 0

    def fail_once(request: bytes, *, exact_model: str, attempt_id: str) -> bytes:
        nonlocal calls
        del request, exact_model, attempt_id
        calls += 1
        raise judge_execution.JudgeTransportError("timeout after dispatch")

    monkeypatch.setattr(transport, "invoke", fail_once)
    with pytest.raises(judge_execution.JudgePassFailed) as retry_blocked:
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=retry_root,
            judge_family="gpt",
            exact_model="gpt-exact-model",
            transport=transport,
        )
    assert calls == 1
    receipt = retry_blocked.value.receipt
    assert receipt["failure"]["reason"] == "budget_fuse_blocked"
    assert receipt["accounting"]["request_count"] == 1
    assert receipt["accounting"]["unknown_cost_reserve_yuan"] == 10
    assert receipt["accounting"]["accounted_cost_yuan"] == 10


def test_codex_receipt_does_not_synthesize_a_transport_reported_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    binary = shutil.which("codex")
    assert binary is not None
    manifest = _case_manifest(count=1)
    _authority, _ledger, root = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=0.0
    )
    events = [
        {"type": "thread.started", "thread_id": "codex-thread-honest-model"},
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": json.dumps({"results": [_output("case-000")]}),
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}},
    ]
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=b"".join(_canonical_bytes(event) + b"\n" for event in events),
        stderr=b"",
    )
    transport = judge_execution.CodexCLIJudgeTransport(binary=binary)
    monkeypatch.setattr(transport, "_run", lambda argv, prompt: completed)

    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=root,
        judge_family="gpt",
        exact_model="gpt-5.6-sol",
        transport=transport,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "yher.llm_sim_v2.judge_execution_receipt.v2"
    assert receipt["identity"]["requested_model"] == "gpt-5.6-sol"
    assert receipt["identity"]["requested_model_verification_source"] == (
        "strict_cli_command_argv"
    )
    assert receipt["identity"]["transport_reported_models"] == []
    assert receipt["identity"]["transport_reported_model_verification_source"] is None
    assert receipt["attempts"][0]["transport_reported_models"] == []
    assert "returned_model" not in receipt["attempts"][0]
    executable = receipt["executable_evidence"]
    resolved = str(Path(binary).resolve(strict=True))
    assert executable["resolved_binary_realpath"] == resolved
    assert executable["binary_sha256"] == hashlib.sha256(
        Path(resolved).read_bytes()
    ).hexdigest()
    assert executable["transport_class"] == (
        "experiments.llm_sim_v2.judge_execution.CodexCLIJudgeTransport"
    )
    assert executable["version_returncode"] == 0
    assert executable["version_stdout"].startswith("codex-cli ")
    assert receipt["command"]["argv"][0] == resolved
    assert receipt["command"]["strict_configuration_verified_from_argv"] is True


def test_python_module_entrypoint_codex_class_identity_replays_canonically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    binary = shutil.which("codex")
    assert binary is not None
    manifest = _case_manifest(count=1)
    _authority, _ledger, root = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=0.0
    )
    events = [
        {"type": "thread.started", "thread_id": "codex-module-entrypoint"},
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": json.dumps({"results": [_output("case-000")]}),
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}},
    ]
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=b"".join(_canonical_bytes(event) + b"\n" for event in events),
        stderr=b"",
    )
    transport = judge_execution.CodexCLIJudgeTransport(binary=binary)
    monkeypatch.setattr(transport, "_run", lambda argv, prompt: completed)
    receipt_path = judge_execution.execute_judge_pass(
        case_manifest=manifest,
        output_root=root,
        judge_family="gpt",
        exact_model="gpt-5.6-sol",
        transport=transport,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["executable_evidence"]["transport_class"] = (
        "__main__.CodexCLIJudgeTransport"
    )
    _rehash_receipt(receipt)
    receipt_path.write_bytes(_canonical_bytes(receipt) + b"\n")

    assert judge_execution.validate_execution_receipt(
        receipt_path,
        manifest,
        "gpt",
    )["execution_receipt_sha256"] == receipt["execution_receipt_sha256"]


def test_claude_parser_claims_only_raw_model_usage_identity() -> None:
    from experiments.llm_sim_v2 import judge_execution

    model = "claude-exact-model"
    stdout = _canonical_bytes(
        {
            "result": json.dumps({"results": [_output("case-000")]}),
            "session_id": "claude-session-model-proof",
            "usage": {"input_tokens": 8, "output_tokens": 4},
            "modelUsage": {model: {"inputTokens": 8, "outputTokens": 4}},
        }
    )
    parsed = judge_execution._claude_response_from_raw_cli(
        raw_transport=judge_execution._raw_cli_binding(
            stdout=stdout, stderr=b"", returncode=0
        ),
        exact_model=model,
        attempt_id="00000000-0000-4000-8000-000000000000",
        unknown_cost_reserve_yuan=10,
    )
    assert parsed["transport_reported_models"] == [model]
    assert parsed["transport_reported_model_source"] == "raw_cli.modelUsage_keys"
    assert "returned_model" not in parsed


def test_production_transport_rejects_fake_executable_and_subclass_before_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    manifest = _case_manifest(count=1)
    fake_binary = tmp_path / "fake/codex"
    fake_binary.parent.mkdir()
    fake_binary.write_text("#!/bin/sh\necho 'codex-cli 0.144.2'\n", encoding="utf-8")
    fake_binary.chmod(0o700)
    _authority, _ledger, fake_root = _mint_test_budget_authority(
        tmp_path / "fake-run", manifest=manifest, baseline_yuan=0.0
    )
    fake = judge_execution.CodexCLIJudgeTransport(binary=str(fake_binary))
    fake_calls = 0

    def forbidden_run(argv: list[str], prompt: bytes) -> subprocess.CompletedProcess[bytes]:
        nonlocal fake_calls
        del argv, prompt
        fake_calls += 1
        raise AssertionError("fake production executable reached transport")

    monkeypatch.setattr(fake, "_run", forbidden_run)
    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="trusted|executable|binary|substitution",
    ):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=fake_root,
            judge_family="gpt",
            exact_model="gpt-5.6-sol",
            transport=fake,
        )
    assert fake_calls == 0

    real_binary = shutil.which("codex")
    assert real_binary is not None

    class SubstitutedCodex(judge_execution.CodexCLIJudgeTransport):
        pass

    _authority, _ledger, subclass_root = _mint_test_budget_authority(
        tmp_path / "subclass-run", manifest=manifest, baseline_yuan=0.0
    )
    substituted = SubstitutedCodex(binary=real_binary)
    subclass_calls = 0

    def forbidden_subclass_run(
        argv: list[str], prompt: bytes
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal subclass_calls
        del argv, prompt
        subclass_calls += 1
        raise AssertionError("substituted transport reached invocation")

    monkeypatch.setattr(substituted, "_run", forbidden_subclass_run)
    with pytest.raises(
        judge_execution.JudgeExecutionError,
        match="transport|class|substitution",
    ):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=subclass_root,
            judge_family="gpt",
            exact_model="gpt-5.6-sol",
            transport=substituted,
        )
    assert subclass_calls == 0


@pytest.mark.parametrize(
    "failure_kind",
    ["non_bytes", "oserror", "runtime_error", "base_exception"],
)
def test_every_post_root_execution_exception_finalizes_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    class SyntheticBaseFailure(BaseException):
        pass

    binary = shutil.which("codex")
    assert binary is not None
    manifest = _case_manifest(count=1)
    _authority, _ledger, root = _mint_test_budget_authority(
        tmp_path, manifest=manifest, baseline_yuan=0.0
    )
    transport = judge_execution.CodexCLIJudgeTransport(binary=binary)
    calls = 0

    def injected_failure(
        request: bytes, *, exact_model: str, attempt_id: str
    ) -> bytes:
        nonlocal calls
        del request, exact_model, attempt_id
        calls += 1
        if failure_kind == "non_bytes":
            return object()  # type: ignore[return-value]
        if failure_kind == "oserror":
            raise FileNotFoundError("binary disappeared after provenance")
        if failure_kind == "runtime_error":
            raise RuntimeError("unexpected transport implementation failure")
        raise SyntheticBaseFailure("base failure after invocation began")

    monkeypatch.setattr(transport, "invoke", injected_failure)
    with pytest.raises(judge_execution.JudgePassFailed) as captured:
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=root,
            judge_family="gpt",
            exact_model="gpt-5.6-sol",
            transport=transport,
        )
    assert calls == 1
    receipt_path = captured.value.receipt_path
    receipt = captured.value.receipt
    assert receipt["status"] == "failed"
    assert receipt["failure"]["reason"] == "execution_exception"
    assert receipt["failure"]["terminal_status"] == "execution_error"
    assert receipt["accounting"]["request_count"] == 1
    assert receipt["accounting"]["unknown_cost_reserve_yuan"] == 10
    assert receipt["accounting"]["accounted_cost_yuan"] == 10
    assert len(receipt["attempts"]) == 1
    attempt = receipt["attempts"][0]
    assert attempt["status"] == "execution_error"
    assert attempt["unknown_cost_reserve_yuan"] == 10
    raw = json.loads(
        (receipt_path.parent / attempt["raw_artifact"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    error = json.loads(base64.b64decode(raw["raw_outer_response_base64"]))
    assert error["schema_version"] == "yher.llm_sim_v2.judge_execution_error.v1"
    assert error["call_may_have_begun"] is True
    assert error["billing"]["unknown_cost_reserve_yuan"] == 10
    assert error["error_type"] == receipt["failure"]["error_type"]
    assert error["message_sha256"] == receipt["failure"]["error_message_sha256"]
    assert judge_execution.validate_failed_execution_receipt(
        receipt_path,
        manifest,
        "gpt",
    )["failed_execution_receipt_sha256"] == receipt[
        "failed_execution_receipt_sha256"
    ]

    second = judge_execution.CodexCLIJudgeTransport(binary=binary)
    second_calls = 0

    def should_not_run(
        request: bytes, *, exact_model: str, attempt_id: str
    ) -> bytes:
        nonlocal second_calls
        del request, exact_model, attempt_id
        second_calls += 1
        raise AssertionError("sealed failed family was rerun")

    monkeypatch.setattr(second, "invoke", should_not_run)
    with pytest.raises(judge_execution.JudgeExecutionError, match="slot|single|already"):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=root,
            judge_family="gpt",
            exact_model="gpt-5.6-sol",
            transport=second,
        )
    assert second_calls == 0
