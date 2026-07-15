"""Provider runner for the frozen Persona v2 dual-condition study.

This module owns only the v2 task contract and v2 store.  The HTTP transport is
the stateless, already-stressed OpenAI-compatible boundary from ``llm_sim``;
no v1 panel, persona, run, or result artifact is read.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments.llm_sim.transport import (
    ProviderHTTPError,
    ProviderNetworkError,
    ProviderTruncatedResponseError,
    ProviderTransport,
    provider_spec,
    transport_policy,
)

from .freeze import validate_analysis_population, verify_freeze_manifest
from .grid import grid_sha256
from .prompts import render_blind_prompt, render_controlled_prompt
from .provenance import verify_frozen_git_commit
from .public import public_question_payload
from .store import RUN_ID, V2Store


FROZEN_COMMIT = "e3c0d4dbe6f37303d9eac86ecd9c1af823f152b9"
FROZEN_DIR_REL = Path("experiments/llm_sim_v2/frozen_v0")
FROZEN_PLAN_REL = Path("experiments/h5v2_analysis_plan.md")
RUNTIME_MANIFEST_REL = Path("experiments/llm_sim_v2/runtime_task_manifest.json")
RUNTIME_PATHS = (
    "experiments/llm_sim_v2/collect.py",
    "experiments/llm_sim_v2/runner.py",
)
_JSON_KEYS = {"simulated", "answer", "rationale"}
_BLIND_JSON_KEYS = _JSON_KEYS | {"abstain"}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


@dataclass(frozen=True)
class ProviderPolicy:
    max_tokens: int
    retry_max_tokens: int
    timeout_seconds: float
    concurrency: int
    max_attempts: int


@dataclass(frozen=True)
class Task:
    phase: str
    analysis_population: str
    persona_id: str
    pair_id: str
    row_id: str
    anchor_id: str
    target_node: str
    response_arm: str
    condition: str
    item_id: str
    family_id: str
    item: Mapping[str, Any]
    correct_option: str
    target_option: str | None
    messages: tuple[Mapping[str, Any], ...]
    wire_messages: tuple[Mapping[str, str], ...]
    prompt_revision: int
    prompt_contract_sha256: str
    is_stability_repeat: bool
    attempt_id: str
    task_id: str
    logical_key: str
    message_sha256: str
    wire_message_sha256: str


@dataclass(frozen=True)
class RuntimeContract:
    repo_root: Path
    config: Mapping[str, Any]
    mapping: Mapping[str, Any]
    personas: tuple[Mapping[str, Any], ...]
    panel: Mapping[str, Any]
    lexicon: tuple[str, ...]
    population: Mapping[str, Any]
    prompt_ledger: Mapping[str, Any]
    freeze_manifest: Mapping[str, Any]
    freeze_proof: Mapping[str, Any]
    runtime_manifest: Mapping[str, Any] | None = None

    def provider_policy(self, provider: str) -> ProviderPolicy:
        name = str(provider).strip().lower()
        raw = self.config["providers"].get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"provider is not in the frozen v2 registry: {provider}")
        return ProviderPolicy(
            max_tokens=int(raw["max_tokens"]),
            retry_max_tokens=int(raw["retry_max_tokens"]),
            timeout_seconds=float(raw["timeout_seconds"]),
            concurrency=int(raw["concurrency"]),
            max_attempts=int(raw["max_attempts"]),
        )

    def provider_model(self, provider: str) -> str:
        raw = self.config["providers"].get(str(provider).strip().lower())
        if not isinstance(raw, Mapping) or not str(raw.get("model") or "").strip():
            raise ValueError(f"provider model is not frozen: {provider}")
        return str(raw["model"])


class InvalidProviderOutput(ValueError):
    """Provider returned content outside the frozen response schema."""


class BudgetFuseOpen(RuntimeError):
    pass


class BudgetLedger:
    def __init__(
        self,
        *,
        soft_warning_yuan: float,
        hard_fuse_yuan: float,
        initial_cost_yuan: float = 0.0,
    ) -> None:
        if not 0 <= soft_warning_yuan < hard_fuse_yuan:
            raise ValueError("budget thresholds must satisfy 0 <= soft < hard")
        if float(initial_cost_yuan) < 0:
            raise ValueError("initial budget cost must be non-negative")
        self.soft_warning_yuan = float(soft_warning_yuan)
        self.hard_fuse_yuan = float(hard_fuse_yuan)
        self.total_cost_yuan = float(initial_cost_yuan)
        self._lock = threading.Lock()

    @property
    def soft_warning_triggered(self) -> bool:
        with self._lock:
            return self.total_cost_yuan >= self.soft_warning_yuan

    @property
    def hard_fuse_triggered(self) -> bool:
        with self._lock:
            return self.total_cost_yuan >= self.hard_fuse_yuan

    def add_cost(self, value: float) -> None:
        cost = max(0.0, float(value))
        with self._lock:
            self.total_cost_yuan += cost

    def assert_new_call_allowed(self) -> None:
        with self._lock:
            if self.total_cost_yuan >= self.hard_fuse_yuan:
                raise BudgetFuseOpen("v2 hard budget fuse is open")


def _strict_json_object(content: str) -> dict[str, Any]:
    def pairs(pairs_value: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs_value:
            if key in output:
                raise InvalidProviderOutput("duplicate JSON response key")
            output[key] = value
        return output

    def reject_constant(value: str) -> Any:
        raise InvalidProviderOutput(f"invalid JSON constant: {value}")

    try:
        value = json.loads(
            content,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except InvalidProviderOutput:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise InvalidProviderOutput("provider content is not strict JSON") from exc
    if not isinstance(value, dict):
        raise InvalidProviderOutput("provider response must be a JSON object")
    return value


def parse_provider_output(
    content: str,
    *,
    condition: str,
    option_keys: set[str],
) -> dict[str, Any]:
    condition_name = str(condition).strip().lower()
    if condition_name not in {"controlled", "blind"}:
        raise ValueError("condition must be controlled or blind")
    value = _strict_json_object(content)
    expected_keys = _BLIND_JSON_KEYS if condition_name == "blind" else _JSON_KEYS
    if set(value) != expected_keys:
        raise InvalidProviderOutput("provider response schema keys drifted")
    if value.get("simulated") is not True:
        raise InvalidProviderOutput("provider response requires simulated:true")
    answer = value.get("answer")
    if answer is not None:
        answer = str(answer).strip().upper()
        if answer not in option_keys:
            raise InvalidProviderOutput("provider answer is not a known option")
    rationale = value.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise InvalidProviderOutput("provider rationale must be a non-empty string")
    output: dict[str, Any] = {
        "simulated": True,
        "answer": answer,
        "rationale": rationale.strip(),
    }
    if condition_name == "blind":
        abstain = value.get("abstain")
        if not isinstance(abstain, bool) or abstain != (answer is None):
            raise InvalidProviderOutput("blind abstain flag is inconsistent with answer")
        output["abstain"] = abstain
    return output


def compute_outcomes(
    *,
    condition: str,
    response_arm: str,
    answer: str | None,
    abstain: bool,
    correct_option: str,
    target_option: str | None,
) -> dict[str, Any]:
    if condition != "controlled":
        return {
            "is_correct": answer == correct_option,
            "target_option_hit": None,
            "manipulation_compliance": None,
        }
    is_correct = answer == correct_option
    target_hit = answer == target_option if target_option is not None else None
    if target_option is None:
        compliance = None
    elif response_arm == "deficit":
        compliance = target_hit
    elif response_arm == "control":
        compliance = is_correct
    else:
        raise ValueError("response_arm must be deficit or control")
    return {
        "is_correct": is_correct,
        "target_option_hit": target_hit,
        "manipulation_compliance": compliance,
    }


def _wire_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, str], ...]:
    output: list[Mapping[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("rendered message must be an object")
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("rendered message is not wire-compatible")
        output.append({"role": role, "content": content})
    return tuple(output)


def _task_hash(payload: Mapping[str, Any]) -> str:
    return _sha(payload)


def _persona_rows(contract: RuntimeContract, phase: str) -> list[Mapping[str, Any]]:
    block = contract.population.get(phase)
    if not isinstance(block, Mapping):
        raise ValueError(f"population manifest has no phase: {phase}")
    allowed_ids = set(str(value) for value in block.get("persona_ids", ()))
    rows = [row for row in contract.personas if str(row.get("persona_id")) in allowed_ids]
    if len({str(row.get("persona_id")) for row in rows}) != len(allowed_ids):
        raise ValueError("population manifest persona membership does not match frozen grid")
    return sorted(rows, key=lambda row: str(row.get("row_id")))


def enumerate_tasks(contract: RuntimeContract, *, phase: str) -> list[Task]:
    phase_name = str(phase).strip().lower()
    if phase_name not in {"pilot", "main"}:
        raise ValueError("phase must be pilot or main")
    rows = _persona_rows(contract, phase_name)
    panels = {
        str(anchor["anchor_id"]): anchor for anchor in contract.panel["anchors"]
    }
    mapping_index = {
        (str(row["item_id"]), str(row["failure_id"])): row
        for row in contract.mapping["rows"]
    }
    repeat_ids = set(contract.config["pilot" if phase_name == "pilot" else "blind"].get("persona_ids", ()))
    if phase_name == "pilot":
        repeat_ids = set(contract.config["pilot"].get("persona_ids", ()))
    else:
        repeat_ids = set(contract.config["blind"].get("terminal_repeat_persona_ids", ()))
        if not repeat_ids:
            # The frozen config stores the count; derive the same deterministic
            # subset from the population block when the explicit list is absent.
            repeat_count = int(contract.config["blind"].get("terminal_repeat_subset_persona_count", 0))
            ranked_ids = [
                str(row["persona_id"])
                for row in sorted(
                    (row for row in rows if row.get("deficit_condition") == "deficit"),
                    key=lambda row: (
                        hashlib.sha256(f"repeat|{row['persona_id']}".encode()).hexdigest(),
                        str(row["persona_id"]),
                    ),
                )
            ]
            repeat_ids = set(ranked_ids[:repeat_count])
    tasks: list[Task] = []
    for persona in rows:
        anchor = panels.get(str(persona.get("anchor_id")))
        if anchor is None:
            raise ValueError("persona row is not bound to a frozen panel anchor")
        items = anchor.get("items")
        if not isinstance(items, list) or len(items) < 4:
            raise ValueError("panel anchor has fewer than four items")
        for condition in ("controlled", "blind"):
            selected = items[:4] if condition == "controlled" else items
            for item_index, item in enumerate(selected):
                tasks.append(
                    _make_task(
                        contract,
                        persona,
                        phase_name,
                        condition,
                        item,
                        mapping_index,
                        is_stability_repeat=False,
                        attempt_id="primary",
                    )
                )
            if condition == "blind" and str(persona.get("persona_id")) in repeat_ids:
                tasks.append(
                    _make_task(
                        contract,
                        persona,
                        phase_name,
                        condition,
                        items[-1],
                        mapping_index,
                        is_stability_repeat=True,
                        attempt_id="stability",
                    )
                )
    tasks.sort(key=lambda task: task.logical_key)
    return tasks


def _make_task(
    contract: RuntimeContract,
    persona: Mapping[str, Any],
    phase: str,
    condition: str,
    item: Mapping[str, Any],
    mapping_index: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    is_stability_repeat: bool,
    attempt_id: str,
) -> Task:
    response_arm = str(persona.get("deficit_condition") or "")
    item_id = str(item.get("item_id") or "")
    failure_id = str(persona.get("failure_id") or "")
    mapping = mapping_index.get((item_id, failure_id))
    target_option = (
        str(mapping.get("target_option") or "").strip().upper()
        if mapping and mapping.get("status") == "mapped"
        else None
    )
    public = public_question_payload(item)
    item_for_prompt = dict(item)
    item_for_prompt["public_question"] = public
    if condition == "controlled":
        messages = render_controlled_prompt(persona, item_for_prompt)
    else:
        messages = render_blind_prompt(
            persona,
            item_for_prompt,
            frozen_leakage_lexicon=contract.lexicon,
        )
    wire = _wire_messages(messages)
    logical_payload = {
        "phase": phase,
        "analysis_population": phase,
        "persona_id": str(persona["persona_id"]),
        "response_arm": response_arm,
        "condition": condition,
        "item_id": item_id,
        "attempt_id": attempt_id,
        "prompt_revision": 0,
    }
    logical_key = json.dumps(logical_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    task_id = _task_hash(logical_payload)
    return Task(
        phase=phase,
        analysis_population=phase,
        persona_id=str(persona["persona_id"]),
        pair_id=str(persona["pair_id"]),
        row_id=str(persona["row_id"]),
        anchor_id=str(persona["anchor_id"]),
        target_node=str(persona["target_node"]),
        response_arm=response_arm,
        condition=condition,
        item_id=item_id,
        family_id=str(item.get("family_id") or ""),
        item=item_for_prompt,
        correct_option=str(item.get("private_correct_option") or item.get("correct_option") or "").upper(),
        target_option=target_option,
        messages=tuple(messages),
        wire_messages=wire,
        prompt_revision=0,
        prompt_contract_sha256=str(
            contract.prompt_ledger["revisions"][0]["prompt_contract_sha256"]
        ),
        is_stability_repeat=is_stability_repeat,
        attempt_id=attempt_id,
        task_id=task_id,
        logical_key=logical_key,
        message_sha256=_sha(messages),
        wire_message_sha256=_sha(wire),
    )


def load_runtime_contract(repo_root: str | Path) -> RuntimeContract:
    root = Path(repo_root).expanduser().resolve(strict=True)
    frozen = root / FROZEN_DIR_REL
    manifest = _read_json(frozen / "freeze_manifest.json")
    verify_freeze_manifest(root, manifest)
    proof = verify_frozen_git_commit(
        root,
        commit=FROZEN_COMMIT,
        declared_files=manifest["frozen_files"],
        observation_timestamp=_utc_now(),
    )
    config = _read_json(frozen / "study_config.json")
    mapping = _read_json(frozen / "target_option_mapping.json")
    personas_artifact = _read_json(frozen / "persona_grid.json")
    panel = _read_json(frozen / "blind_panel.json")
    lexicon_artifact = _read_json(frozen / "leakage_lexicon.json")
    population = _read_json(frozen / "population_manifest.json")
    prompt_ledger = _read_json(frozen / "prompt_revision_ledger.json")
    source_manifest = _read_json(frozen / "source_manifest.json")
    from .official import verify_source_manifest

    verify_source_manifest(root, source_manifest)
    if config.get("run_id") != RUN_ID or config.get("modality_condition") != "text_only":
        raise ValueError("runtime config is not the frozen v2 contract")
    if config.get("mapping_sha256") != mapping.get("mapping_sha256"):
        raise ValueError("runtime mapping hash does not match config")
    if config.get("target_set_hash") != mapping.get("target_set_hash"):
        raise ValueError("runtime target-set hash does not match config")
    if config.get("leakage_lexicon_sha256") != lexicon_artifact.get("sha256"):
        raise ValueError("runtime leakage lexicon hash does not match config")
    if population.get("population_manifest_sha256") != manifest.get("population_manifest_sha256"):
        raise ValueError("runtime population manifest hash does not match freeze")
    if prompt_ledger.get("prompt_ledger_sha256") != manifest.get("prompt_ledger_sha256"):
        raise ValueError("runtime prompt ledger hash does not match freeze")
    if personas_artifact.get("grid_sha256") != manifest.get("grid_sha256"):
        raise ValueError("runtime persona grid hash does not match freeze")
    if panel.get("panel_sha256") is None:
        raise ValueError("runtime blind panel is not hashed")
    personas = tuple(personas_artifact.get("rows") or ())
    if not personas or not all(isinstance(row, Mapping) for row in personas):
        raise ValueError("runtime persona grid is empty")
    for phase in ("pilot", "main"):
        validate_analysis_population(
            [
                {
                    "simulated": True,
                    "run_id": RUN_ID,
                    "phase": phase,
                    "analysis_population": phase,
                }
            ],
            phase=phase,
        )
    runtime_manifest_path = root / RUNTIME_MANIFEST_REL
    runtime_manifest = (
        _read_json(runtime_manifest_path) if runtime_manifest_path.is_file() else None
    )
    contract = RuntimeContract(
        repo_root=root,
        config=config,
        mapping=mapping,
        personas=personas,
        panel=panel,
        lexicon=tuple(str(term) for term in lexicon_artifact.get("terms") or ()),
        population=population,
        prompt_ledger=prompt_ledger,
        freeze_manifest=manifest,
        freeze_proof=proof,
        runtime_manifest=runtime_manifest,
    )
    if runtime_manifest is not None:
        verify_runtime_task_manifest(contract, runtime_manifest, verify_git=False)
    return contract


def _validate_runtime_timestamp(value: str) -> str:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("runtime frozen_at_utc must be ISO-8601") from exc
    if not text.endswith("Z") or parsed.tzinfo is None:
        raise ValueError("runtime frozen_at_utc must be UTC")
    return text


def build_runtime_task_manifest(
    contract: RuntimeContract,
    *,
    runtime_commit: str,
    frozen_at_utc: str,
) -> dict[str, Any]:
    commit = str(runtime_commit).strip().lower()
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise ValueError("runtime commit must be a full lowercase git SHA")
    phases: dict[str, Any] = {}
    for phase in ("pilot", "main"):
        tasks = enumerate_tasks(contract, phase=phase)
        task_ids = [task.task_id for task in tasks]
        rows = [
            {
                "task_id": task.task_id,
                "logical_key": task.logical_key,
                "message_sha256": task.message_sha256,
                "wire_message_sha256": task.wire_message_sha256,
                "prompt_contract_sha256": task.prompt_contract_sha256,
            }
            for task in tasks
        ]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"runtime {phase} tasks contain duplicate task IDs")
        phases[phase] = {
            "task_count": len(tasks),
            "task_ids": task_ids,
            "task_set_sha256": _sha(rows),
            "providers": list(contract.config[phase]["providers"]),
        }
    runtime_files = []
    for relative in RUNTIME_PATHS:
        data = (contract.repo_root / relative).read_bytes()
        runtime_files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
        )
    manifest = {
        "schema_version": "yher.llm_sim_v2.runtime_task_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "freeze_commit": FROZEN_COMMIT,
        "freeze_manifest_sha256": contract.freeze_manifest["freeze_manifest_sha256"],
        "runtime_commit": commit,
        "frozen_at_utc": _validate_runtime_timestamp(frozen_at_utc),
        "prompt_revision": 0,
        "runtime_files": runtime_files,
        "runtime_file_set_sha256": _sha(runtime_files),
        "phases": phases,
    }
    manifest["runtime_task_manifest_sha256"] = _sha(manifest)
    return manifest


def verify_runtime_task_manifest(
    contract: RuntimeContract,
    manifest: Mapping[str, Any],
    *,
    verify_git: bool = False,
) -> dict[str, Any]:
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != "yher.llm_sim_v2.runtime_task_manifest.v1"
        or manifest.get("run_id") != RUN_ID
        or manifest.get("freeze_commit") != FROZEN_COMMIT
        or manifest.get("freeze_manifest_sha256")
        != contract.freeze_manifest.get("freeze_manifest_sha256")
    ):
        raise ValueError("runtime task manifest envelope does not match the frozen study")
    advertised = manifest.get("runtime_task_manifest_sha256")
    payload = dict(manifest)
    payload.pop("runtime_task_manifest_sha256", None)
    if advertised != _sha(payload):
        raise ValueError("runtime task manifest digest mismatch")
    expected = build_runtime_task_manifest(
        contract,
        runtime_commit=str(manifest.get("runtime_commit") or ""),
        frozen_at_utc=str(manifest.get("frozen_at_utc") or ""),
    )
    if expected != manifest:
        raise ValueError("runtime task manifest task set drifted")
    git_proof = None
    if verify_git:
        git_proof = verify_frozen_git_commit(
            contract.repo_root,
            commit=str(manifest["runtime_commit"]),
            declared_files=list(manifest["runtime_files"]),
            observation_timestamp=_utc_now(),
        )
    return {
        "schema_version": "yher.llm_sim_v2.runtime_task_manifest_proof.v1",
        "ok": True,
        "run_id": RUN_ID,
        "runtime_task_manifest_sha256": advertised,
        "runtime_commit": manifest["runtime_commit"],
        "git_proof": git_proof,
    }


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (ProviderNetworkError, ProviderTruncatedResponseError)):
        return True
    return isinstance(exc, ProviderHTTPError) and (exc.status == 429 or exc.status >= 500)


def _error_category(exc: Exception) -> str:
    if isinstance(exc, ProviderTruncatedResponseError):
        return "truncated_length"
    if isinstance(exc, ProviderNetworkError):
        return "network_timeout"
    if isinstance(exc, ProviderHTTPError):
        return f"http_{exc.status}"
    if isinstance(exc, InvalidProviderOutput):
        return "invalid_schema"
    return "protocol_or_unexpected"


def _backoff(policy: ProviderPolicy, retry_index: int) -> float:
    return min(30.0, max(0.0, policy.timeout_seconds / 60.0) * (2**retry_index) / 10.0)


def execute_task(
    task: Task,
    *,
    provider: str,
    model: str,
    transport: ProviderTransport,
    policy: ProviderPolicy,
    budget: BudgetLedger,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = lambda: 0.5,
) -> dict[str, Any]:
    del random_value  # deterministic retries are part of the frozen contract
    attempts: list[dict[str, Any]] = []
    request_max_tokens = policy.max_tokens
    parsed: dict[str, Any] | None = None
    returned_model: str | None = None
    final_status = "technical_failure"
    final_error: str | None = None
    for attempt_index in range(policy.max_attempts):
        budget.assert_new_call_allowed()
        started = time.monotonic()
        try:
            response = transport.complete(
                provider=provider,
                model=model,
                messages=list(task.wire_messages),
                timeout_seconds=policy.timeout_seconds,
                max_tokens=request_max_tokens,
            )
            cost = max(0.0, float(response.get("cost_yuan") or 0.0))
            budget.add_cost(cost)
            returned_model = str(response.get("model_returned") or "").strip()
            usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "response",
                    "latency_ms": round(float(response.get("latency_ms") or (time.monotonic() - started) * 1000), 3),
                    "model_returned": returned_model,
                    "finish_reason": str(response.get("finish_reason") or ""),
                    "usage": {
                        "input_tokens": max(0, int(usage.get("input_tokens") or 0)),
                        "output_tokens": max(0, int(usage.get("output_tokens") or 0)),
                    },
                    "cost_yuan": cost,
                }
            )
            if returned_model != model:
                final_status = "excluded_model_drift"
                final_error = "returned_model_drift"
                break
            try:
                option_keys = {
                    str(key).strip().upper()
                    for key in (task.item.get("options") or {})
                }
                parsed = parse_provider_output(
                    str(response.get("content") or ""),
                    condition=task.condition,
                    option_keys=option_keys,
                )
                final_status = "complete"
                final_error = None
                break
            except InvalidProviderOutput as exc:
                final_status = "excluded_schema"
                final_error = str(exc)
                if attempt_index + 1 >= policy.max_attempts:
                    break
                sleep(_backoff(policy, attempt_index))
                continue
        except BudgetFuseOpen:
            raise
        except Exception as exc:
            final_error = _error_category(exc)
            if isinstance(exc, ProviderTruncatedResponseError):
                truncated_usage = exc.usage
                truncated_cost = max(0.0, float(exc.cost_yuan))
                budget.add_cost(truncated_cost)
                attempt = {
                    "attempt": attempt_index + 1,
                    "status": "failed",
                    "error_category": final_error,
                    "latency_ms": round(float(exc.latency_ms), 3),
                    "model_returned": str(exc.returned_model or ""),
                    "finish_reason": str(exc.finish_reason or ""),
                    "reasoning_tokens": max(0, int(exc.reasoning_tokens)),
                    "usage": {
                        "input_tokens": max(0, int(truncated_usage.get("input_tokens") or 0)),
                        "output_tokens": max(0, int(truncated_usage.get("output_tokens") or 0)),
                    },
                    "cost_yuan": truncated_cost,
                }
            else:
                attempt = {
                    "attempt": attempt_index + 1,
                    "status": "failed",
                    "error_category": final_error,
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                }
            attempts.append(
                attempt
            )
            if attempt_index + 1 >= policy.max_attempts or not _retryable(exc):
                break
            if isinstance(exc, ProviderTruncatedResponseError):
                request_max_tokens = policy.retry_max_tokens
            sleep(_backoff(policy, attempt_index))

    outcomes = compute_outcomes(
        condition=task.condition,
        response_arm=task.response_arm,
        answer=parsed.get("answer") if parsed else None,
        abstain=bool(parsed.get("abstain")) if parsed else False,
        correct_option=task.correct_option,
        target_option=task.target_option,
    ) if parsed else {
        "is_correct": None,
        "target_option_hit": None,
        "manipulation_compliance": None,
    }
    return {
        "schema_version": "yher.llm_sim_v2.response_record.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": task.phase,
        "analysis_population": task.analysis_population,
        "provider": provider,
        "model_id": returned_model or model,
        "requested_model": model,
        "prompt_revision": task.prompt_revision,
        "prompt_contract_sha256": task.prompt_contract_sha256,
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
        "status": final_status,
        "error": final_error,
        "parsed_output": parsed,
        "outcomes": outcomes,
        "attempts": attempts,
        "retry_count": max(0, len(attempts) - 1),
        "cost_yuan": round(sum(float(row.get("cost_yuan") or 0.0) for row in attempts), 8),
    }


class V2ProviderRunner:
    def __init__(
        self,
        *,
        contract: RuntimeContract,
        output_base: str | Path,
        phase: str,
        provider: str,
        transport: ProviderTransport,
        budget: BudgetLedger,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = lambda: 0.5,
    ) -> None:
        phase_name = str(phase).strip().lower()
        if phase_name not in {"pilot", "main"}:
            raise ValueError("phase must be pilot or main")
        self.contract = contract
        self.phase = phase_name
        self.provider = str(provider).strip().lower()
        self.model = contract.provider_model(self.provider)
        self.policy = contract.provider_policy(self.provider)
        self.transport = transport
        self.budget = budget
        self.sleep = sleep
        self.random_value = random_value
        self.store = V2Store(output_base, phase=phase_name)
        self._write_lock = threading.Lock()

    def _record_path(self, task: Task) -> Path:
        return self.store.path(
            Path("records") / self.provider / f"{task.task_id}.json"
        )

    def _read_existing(self, task: Task) -> dict[str, Any] | None:
        path = self._record_path(task)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise ValueError("existing v2 record is not an object")
        if (
            record.get("task_id") != task.task_id
            or record.get("logical_key") != task.logical_key
            or record.get("phase") != self.phase
            or record.get("analysis_population") != self.phase
            or record.get("provider") != self.provider
            or record.get("requested_model") != self.model
        ):
            raise ValueError("existing v2 record does not match the frozen task")
        return dict(record)

    def _write_record(self, task: Task, record: Mapping[str, Any]) -> None:
        with self._write_lock:
            self.store.write_json(
                Path("records") / self.provider / f"{task.task_id}.json",
                record,
                immutable=True,
            )

    def run_tasks(self, tasks: Sequence[Task]) -> dict[str, Any]:
        selected = [task for task in tasks if task.phase == self.phase]
        existing: dict[str, dict[str, Any]] = {}
        pending: list[Task] = []
        for task in selected:
            record = self._read_existing(task)
            if record is None:
                pending.append(task)
            else:
                existing[task.task_id] = record
        records: dict[str, dict[str, Any]] = dict(existing)
        fuse_open = False
        with ThreadPoolExecutor(max_workers=self.policy.concurrency) as pool:
            futures = {
                pool.submit(
                    execute_task,
                    task,
                    provider=self.provider,
                    model=self.model,
                    transport=self.transport,
                    policy=self.policy,
                    budget=self.budget,
                    sleep=self.sleep,
                    random_value=self.random_value,
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record = future.result()
                except BudgetFuseOpen:
                    fuse_open = True
                    continue
                self._write_record(task, record)
                records[task.task_id] = record
        counts = Counter(str(record.get("status")) for record in records.values())
        summary = {
            "schema_version": "yher.llm_sim_v2.provider_manifest.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "phase": self.phase,
            "analysis_population": self.phase,
            "provider": self.provider,
            "requested_model": self.model,
            "returned_models": sorted({str(record.get("model_id")) for record in records.values()}),
            "freeze_commit": FROZEN_COMMIT,
            "prompt_revision": 0,
            "record_count": len(records),
            "complete_records": counts.get("complete", 0),
            "resumed_records": len(existing),
            "status_counts": dict(sorted(counts.items())),
            "budget": {
                "total_cost_yuan": round(self.budget.total_cost_yuan, 8),
                "soft_warning_triggered": self.budget.soft_warning_triggered,
                "hard_fuse_triggered": self.budget.hard_fuse_triggered or fuse_open,
            },
            "finished_at_utc": _utc_now(),
        }
        self.store.write_json(
            Path("provider_manifests") / f"{self.provider}.json",
            summary,
            immutable=False,
        )
        return summary


__all__ = [
    "BudgetFuseOpen",
    "BudgetLedger",
    "InvalidProviderOutput",
    "ProviderPolicy",
    "RuntimeContract",
    "Task",
    "V2ProviderRunner",
    "compute_outcomes",
    "build_runtime_task_manifest",
    "enumerate_tasks",
    "execute_task",
    "load_runtime_contract",
    "parse_provider_output",
    "verify_runtime_task_manifest",
]
