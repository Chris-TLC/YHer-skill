"""Provider runner for the frozen Persona v2 dual-condition study.

This module owns only the v2 task contract and v2 store.  The HTTP transport is
the stateless, already-stressed OpenAI-compatible boundary from ``llm_sim``;
no v1 panel, persona, run, or result artifact is read.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

from experiments.llm_sim.transport import (
    ProviderHTTPError,
    ProviderNetworkError,
    ProviderTruncatedResponseError,
    ProviderTransport,
    transport_policy,
)

from .freeze import (
    build_rendered_prompt_contract_hashes,
    validate_analysis_population,
    verify_freeze_manifest,
)
from .prompts import render_blind_prompt, render_controlled_prompt
from .provenance import verify_frozen_git_commit
from .public import public_question_payload
from .store import RUN_ID, V2Store
from .evidence import (
    ProviderEvidenceLedger,
    REVIEWED_CARRIED_LEDGER_SHA256,
    REVIEWED_LEGACY_KNOWN_COST_YUAN,
    REVIEWED_LEGACY_RECEIPT_SHA256,
    REVIEWED_LEGACY_RECORD_SET_SHA256,
    bind_response_content,
    build_provider_record_set,
    validate_v2_response_record,
)


FROZEN_COMMIT = "e3c0d4dbe6f37303d9eac86ecd9c1af823f152b9"
FROZEN_DIR_REL = Path("experiments/llm_sim_v2/frozen_v0")
FROZEN_PLAN_REL = Path("experiments/h5v2_analysis_plan.md")
PRIOR_COST_LEDGER_REL = Path("experiments/llm_sim_v2/prior_cost_ledger.json")
PRIOR_COST_LEDGER_SHA256 = (
    "962c8eeb45702a3a38740db296d574773dac204ac8162b557388e295f011d58e"
)
RUNTIME_MANIFEST_REL = Path("experiments/llm_sim_v2/runtime_task_manifest.json")
RUNTIME_PATHS = (
    "experiments/config/llm_transport_v2.json",
    "experiments/llm_sim/transport.py",
    "experiments/llm_sim_v2/collect.py",
    "experiments/llm_sim_v2/evidence.py",
    "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json",
    "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_retrospective_receipt.json",
    "experiments/llm_sim_v2/prior_cost_ledger.json",
    "experiments/llm_sim_v2/runner.py",
    "experiments/llm_sim_v2/store.py",
)
_JSON_KEYS = {"simulated", "answer", "rationale"}
_BLIND_JSON_KEYS = _JSON_KEYS | {"abstain"}
_SYSTEM_RANDOM = random.SystemRandom()
UNKNOWN_ATTEMPT_RESERVE_YUAN = 10.0


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


def _utc_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def verify_prior_cost_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    expected_entries = [
        {
            "evidence_id": "mapping_crosscheck",
            "evidence_sha256": (
                "e2f1e5533f0b962c60f140d0d1a07f36d03ab37d5531741c1b8ad554009a7a69"
            ),
            "cost_yuan": 0.28948744,
        },
        {
            "evidence_id": "provider_probe_summary",
            "evidence_sha256": (
                "7482afead0198adf0d12b444059a52efbe550d373e0f017346736d402d5350a2"
            ),
            "cost_yuan": 0.11171617,
        },
        {
            "evidence_id": "quarantine_56_record_canonical_row_manifest",
            "evidence_sha256": (
                "b3e7b4d7a15b51e00cae849ce738772e2f1ce8e52006afdb33dc44080711a7a0"
            ),
            "cost_yuan": 0.1434596,
        },
    ]
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("schema_version")
        != "yher.llm_sim_v2.prior_cost_ledger.v1"
        or ledger.get("simulated") is not True
        or ledger.get("run_id") != RUN_ID
        or ledger.get("currency") != "CNY"
        or ledger.get("known_cost_entries") != expected_entries
        or float(ledger.get("known_cost_yuan", -1)) != 0.54466321
        or float(ledger.get("pre_run_ambiguity_reserve_yuan", -1)) != 0.113
        or float(ledger.get("pre_run_total_bound_yuan", -1)) != 0.65766321
        or float(ledger.get("unknown_attempt_reserve_yuan", -1)) != 10.0
    ):
        raise ValueError("prior cost ledger content is not the reviewed bound")
    payload = dict(ledger)
    advertised = payload.pop("prior_cost_ledger_sha256", None)
    if advertised != _sha(payload) or advertised != PRIOR_COST_LEDGER_SHA256:
        raise ValueError("prior cost ledger digest is not the reviewed bound")
    if round(sum(float(row["cost_yuan"]) for row in expected_entries), 8) != float(
        ledger["known_cost_yuan"]
    ) or round(
        float(ledger["known_cost_yuan"])
        + float(ledger["pre_run_ambiguity_reserve_yuan"]),
        8,
    ) != float(ledger["pre_run_total_bound_yuan"]):
        raise ValueError("prior cost ledger totals do not reconcile")
    return {
        "schema_version": "yher.llm_sim_v2.prior_cost_ledger_proof.v1",
        "ok": True,
        "run_id": RUN_ID,
        "prior_cost_ledger_sha256": advertised,
    }


@dataclass(frozen=True)
class ProviderPolicy:
    max_tokens: int
    retry_max_tokens: int
    timeout_seconds: float
    concurrency: int
    max_attempts: int
    failure_threshold: int
    base_backoff_seconds: float
    max_backoff_seconds: float
    cooldown_seconds: float
    jitter_fraction: float


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
    source_manifest: Mapping[str, Any]
    prior_cost_ledger: Mapping[str, Any]
    freeze_manifest: Mapping[str, Any]
    freeze_proof: Mapping[str, Any]
    runtime_manifest: Mapping[str, Any] | None = None
    validated_prompt_revision: Mapping[str, Any] | None = None
    validated_prompt_ledger_sha256: str | None = None

    def provider_policy(self, provider: str) -> ProviderPolicy:
        name = str(provider).strip().lower()
        raw = self.config["providers"].get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"provider is not in the frozen v2 registry: {provider}")
        runtime = transport_policy(name, version="v2")
        frozen_values = {
            "max_tokens": int(raw["max_tokens"]),
            "retry_max_tokens": int(raw["retry_max_tokens"]),
            "timeout_seconds": float(raw["timeout_seconds"]),
            "concurrency": int(raw["concurrency"]),
            "max_attempts": int(raw["max_attempts"]),
        }
        if any(getattr(runtime, key) != value for key, value in frozen_values.items()):
            raise ValueError(f"transport policy drifted from frozen provider config: {name}")
        return ProviderPolicy(
            max_tokens=runtime.max_tokens,
            retry_max_tokens=runtime.retry_max_tokens,
            timeout_seconds=runtime.timeout_seconds,
            concurrency=runtime.concurrency,
            max_attempts=runtime.max_attempts,
            failure_threshold=runtime.failure_threshold,
            base_backoff_seconds=runtime.base_backoff_seconds,
            max_backoff_seconds=runtime.max_backoff_seconds,
            cooldown_seconds=runtime.cooldown_seconds,
            jitter_fraction=runtime.jitter_fraction,
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


def active_prompt_revision(contract: RuntimeContract) -> dict[str, Any]:
    ledger = contract.prompt_ledger
    if (
        ledger.get("schema_version") != "yher.llm_sim_v2.prompt_revision_ledger.v1"
        or ledger.get("simulated") is not True
        or ledger.get("run_id") != RUN_ID
        or ledger.get("maximum_prompt_rewrites") != 1
    ):
        raise ValueError("prompt revision ledger envelope is invalid")
    advertised = ledger.get("prompt_ledger_sha256")
    ledger_payload = dict(ledger)
    ledger_payload.pop("prompt_ledger_sha256", None)
    if advertised != _sha(ledger_payload):
        raise ValueError("prompt revision ledger digest mismatch")
    current = ledger.get("current_revision")
    if not isinstance(current, int) or isinstance(current, bool) or current not in {0, 1}:
        raise ValueError("prompt revision ledger current revision is invalid")
    cached = contract.validated_prompt_revision
    if (
        isinstance(cached, Mapping)
        and contract.validated_prompt_ledger_sha256 == advertised
        and cached.get("revision") == current
    ):
        files_match = True
        for row in cached.get("prompt_files", ()):
            relative = Path(str(row.get("path") or ""))
            data = (contract.repo_root / relative).read_bytes()
            files_match = files_match and (
                row.get("sha256") == hashlib.sha256(data).hexdigest()
                and row.get("size") == len(data)
            )
        if files_match:
            return dict(cached)
    revisions = ledger.get("revisions")
    if not isinstance(revisions, list) or len(revisions) != current + 1:
        raise ValueError("prompt revision ledger is stale or incomplete")
    revision_ids = [row.get("revision") if isinstance(row, Mapping) else None for row in revisions]
    if revision_ids != list(range(current + 1)):
        raise ValueError("prompt revision ledger transition sequence is invalid")
    for index, raw_revision in enumerate(revisions):
        if not isinstance(raw_revision, Mapping):
            raise ValueError("prompt revision ledger row is invalid")
        revision = dict(raw_revision)
        contract_sha = revision.pop("prompt_contract_sha256", None)
        if contract_sha != _sha(revision):
            raise ValueError("prompt revision contract digest mismatch")
        expected_parent = None if index == 0 else index - 1
        if raw_revision.get("parent_revision") != expected_parent:
            raise ValueError("prompt revision parent transition is invalid")
    active = dict(revisions[current])
    if active.get("status") != "pre_observation_frozen":
        raise ValueError("active prompt revision is not frozen")
    if current == 1:
        raise ValueError(
            "prompt revision 1 blocked until committed pilot-failure evidence is bound"
        )
    _validate_runtime_timestamp(str(active.get("committed_at_utc") or ""))
    prompt_files = active.get("prompt_files")
    if not isinstance(prompt_files, list) or not prompt_files:
        raise ValueError("active prompt revision has no prompt files")
    seen_paths: set[str] = set()
    for row in prompt_files:
        if not isinstance(row, Mapping):
            raise ValueError("active prompt revision file row is invalid")
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("active prompt revision path is invalid")
        normalized = relative.as_posix()
        if normalized in seen_paths:
            raise ValueError("active prompt revision repeats a prompt file")
        seen_paths.add(normalized)
        data = (contract.repo_root / relative).read_bytes()
        if (
            row.get("sha256") != hashlib.sha256(data).hexdigest()
            or row.get("size") != len(data)
        ):
            raise ValueError("active prompt file is not byte-identical")
    rendered = build_rendered_prompt_contract_hashes(
        contract.personas,
        contract.panel,
        {"terms": list(contract.lexicon)},
    )
    if active.get("rendered_contract_sha256") != rendered:
        raise ValueError("active rendered prompt contract hash drifted")
    expected_bindings = {
        "mapping_sha256": contract.mapping.get("mapping_sha256"),
        "grid_sha256": contract.freeze_manifest.get("grid_sha256"),
        "panel_sha256": contract.panel.get("panel_sha256"),
        "blind_lexicon_sha256": contract.config.get("leakage_lexicon_sha256"),
    }
    for field, expected in expected_bindings.items():
        if active.get(field) != expected:
            raise ValueError(f"active prompt revision {field} binding drifted")
    return active


def enumerate_tasks(contract: RuntimeContract, *, phase: str) -> list[Task]:
    phase_name = str(phase).strip().lower()
    if phase_name not in {"pilot", "main"}:
        raise ValueError("phase must be pilot or main")
    prompt_revision = active_prompt_revision(contract)
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
                        prompt_revision,
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
                        prompt_revision,
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
    prompt_revision: Mapping[str, Any],
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
        "prompt_revision": int(prompt_revision["revision"]),
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
        prompt_revision=int(prompt_revision["revision"]),
        prompt_contract_sha256=str(prompt_revision["prompt_contract_sha256"]),
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
    prior_cost_ledger = _read_json(root / PRIOR_COST_LEDGER_REL)
    from .official import verify_source_manifest

    verify_source_manifest(root, source_manifest)
    verify_prior_cost_ledger(prior_cost_ledger)
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
        source_manifest=source_manifest,
        prior_cost_ledger=prior_cost_ledger,
        freeze_manifest=manifest,
        freeze_proof=proof,
        runtime_manifest=runtime_manifest,
    )
    prompt_revision = active_prompt_revision(contract)
    contract = replace(
        contract,
        validated_prompt_revision=prompt_revision,
        validated_prompt_ledger_sha256=str(prompt_ledger["prompt_ledger_sha256"]),
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
        "prompt_revision": int(active_prompt_revision(contract)["revision"]),
        "prompt_contract_sha256": str(
            active_prompt_revision(contract)["prompt_contract_sha256"]
        ),
        "prompt_ledger_sha256": contract.prompt_ledger["prompt_ledger_sha256"],
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


def _frozen_file_sha256(contract: RuntimeContract, relative: str) -> str:
    matches = [
        row
        for row in contract.freeze_manifest.get("frozen_files", ())
        if isinstance(row, Mapping) and row.get("path") == relative
    ]
    if len(matches) != 1 or not str(matches[0].get("sha256") or ""):
        raise ValueError(f"freeze manifest does not bind {relative}")
    return str(matches[0]["sha256"])


def build_phase_provenance(
    contract: RuntimeContract,
    *,
    runtime_manifest: Mapping[str, Any],
    runtime_proof: Mapping[str, Any],
    phase: str,
    tasks: Sequence[Task],
    collection_scope: Mapping[str, Any],
    prior_cost_ledger: Mapping[str, Any],
    carried_forward_cost: Mapping[str, Any] | None = None,
    first_observation_at_utc: str,
) -> dict[str, Any]:
    phase_name = str(phase).strip().lower()
    if phase_name not in {"pilot", "main"}:
        raise ValueError("phase must be pilot or main")
    verify_runtime_task_manifest(contract, runtime_manifest, verify_git=False)
    git_proof = runtime_proof.get("git_proof")
    if (
        runtime_proof.get("ok") is not True
        or runtime_proof.get("runtime_task_manifest_sha256")
        != runtime_manifest.get("runtime_task_manifest_sha256")
        or runtime_proof.get("runtime_commit") != runtime_manifest.get("runtime_commit")
        or not isinstance(git_proof, Mapping)
        or git_proof.get("ok") is not True
        or git_proof.get("byte_identical") is not True
        or git_proof.get("commit") != runtime_manifest.get("runtime_commit")
    ):
        raise ValueError("phase provenance requires a byte-identical runtime git proof")
    timestamp = _validate_runtime_timestamp(first_observation_at_utc)
    selected_tasks = [task for task in tasks if task.phase == phase_name]
    if len(selected_tasks) != len(tasks):
        raise ValueError("phase provenance task roster crosses phase boundaries")
    task_ids = [task.task_id for task in selected_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("phase provenance task roster contains duplicates")
    runtime_phase = runtime_manifest.get("phases", {}).get(phase_name)
    if not isinstance(runtime_phase, Mapping):
        raise ValueError("runtime manifest lacks the requested phase")
    required_scope = {
        "collection_mode",
        "development_only",
        "partial",
        "formal_analysis_eligible",
        "frozen_providers",
        "selected_providers",
        "task_limit",
    }
    if not isinstance(collection_scope, Mapping) or set(collection_scope) != required_scope:
        raise ValueError("phase collection scope is incomplete")
    frozen_providers = list(contract.config[phase_name]["providers"])
    if list(collection_scope["frozen_providers"]) != frozen_providers:
        raise ValueError("phase collection scope frozen providers drifted")
    formal = collection_scope.get("collection_mode") == "formal"
    if formal and (
        collection_scope.get("development_only") is not False
        or collection_scope.get("partial") is not False
        or collection_scope.get("formal_analysis_eligible") is not True
        or list(collection_scope["selected_providers"]) != frozen_providers
        or task_ids != list(runtime_phase.get("task_ids") or ())
    ):
        raise ValueError("formal phase scope does not match the frozen population")
    if not formal and collection_scope.get("formal_analysis_eligible") is not False:
        raise ValueError("partial phase scope cannot be formal-analysis eligible")
    verify_prior_cost_ledger(prior_cost_ledger)
    if dict(prior_cost_ledger) != dict(contract.prior_cost_ledger):
        raise ValueError("phase provenance prior cost ledger differs from runtime contract")
    carried = dict(carried_forward_cost or {})
    carried_digest = carried.get("carried_forward_cost_ledger_sha256")
    carried_receipt = carried.get("source_phase_receipt_sha256")
    carried_record_set = carried.get("source_record_set_sha256")
    try:
        carried_known = float(carried.get("known_cost_yuan", 0.0))
        carried_reserve = float(carried.get("unknown_cost_reserve_yuan", 0.0))
        carried_total = float(carried.get("total_accounted_cost_yuan", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("phase provenance carried-forward cost is invalid") from exc
    if carried_digest is not None:
        for value in (carried_digest, carried_receipt, carried_record_set):
            text = str(value or "")
            if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
                raise ValueError("phase provenance carried-forward binding is invalid")
    elif carried_receipt is not None or carried_record_set is not None or any(
        value != 0.0 for value in (carried_known, carried_reserve, carried_total)
    ):
        raise ValueError("phase provenance carried-forward binding is invalid")
    if (
        any(not math.isfinite(value) or value < 0 for value in (
            carried_known,
            carried_reserve,
            carried_total,
        ))
        or not math.isclose(
            carried_total,
            carried_known + carried_reserve,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    ):
        raise ValueError("phase provenance carried-forward totals are invalid")
    if formal and not (
        carried_digest == REVIEWED_CARRIED_LEDGER_SHA256
        and carried_record_set == REVIEWED_LEGACY_RECORD_SET_SHA256
        and carried_receipt == REVIEWED_LEGACY_RECEIPT_SHA256
        and carried_known == REVIEWED_LEGACY_KNOWN_COST_YUAN
        and carried_reserve == 0.0
        and carried_total == REVIEWED_LEGACY_KNOWN_COST_YUAN
    ):
        raise ValueError("formal phase requires the reviewed carried-forward ledger")
    prompt = active_prompt_revision(contract)
    roster_rows = [
        {
            "task_id": task.task_id,
            "logical_key": task.logical_key,
            "message_sha256": task.message_sha256,
            "wire_message_sha256": task.wire_message_sha256,
            "prompt_contract_sha256": task.prompt_contract_sha256,
        }
        for task in selected_tasks
    ]
    artifact: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.phase_provenance.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": phase_name,
        "analysis_population": phase_name,
        "collection_mode": collection_scope["collection_mode"],
        "development_only": bool(collection_scope["development_only"]),
        "partial": bool(collection_scope["partial"]),
        "formal_analysis_eligible": bool(
            collection_scope["formal_analysis_eligible"]
        ),
        "modality_condition": contract.config["modality_condition"],
        "selected_providers": list(collection_scope["selected_providers"]),
        "frozen_providers": frozen_providers,
        "task_limit": collection_scope["task_limit"],
        "first_observation_at_utc": timestamp,
        "freeze": {
            "freeze_commit": FROZEN_COMMIT,
            "freeze_manifest_sha256": contract.freeze_manifest[
                "freeze_manifest_sha256"
            ],
            "artifact_set_sha256": contract.freeze_manifest["artifact_set_sha256"],
            "frozen_at_utc": contract.freeze_manifest["frozen_at_utc"],
            "git_proof": dict(contract.freeze_proof),
        },
        "source": {
            "source_set_sha256": contract.freeze_manifest["source_set_sha256"],
            "source_manifest_file_sha256": _frozen_file_sha256(
                contract,
                "experiments/llm_sim_v2/frozen_v0/source_manifest.json",
            ),
            "files": list(contract.source_manifest["files"]),
        },
        "target": {
            "target_set_hash": contract.config["target_set_hash"],
            "mapping_sha256": contract.mapping["mapping_sha256"],
        },
        "grid_sha256": contract.freeze_manifest["grid_sha256"],
        "population_manifest_sha256": contract.freeze_manifest[
            "population_manifest_sha256"
        ],
        "official_inputs_sha256": contract.freeze_manifest[
            "official_inputs_sha256"
        ],
        "prompt": {
            "revision": prompt["revision"],
            "prompt_contract_sha256": prompt["prompt_contract_sha256"],
            "prompt_ledger_sha256": contract.prompt_ledger[
                "prompt_ledger_sha256"
            ],
            "rendered_contract_sha256": dict(
                prompt["rendered_contract_sha256"]
            ),
            "prompt_files": list(prompt["prompt_files"]),
        },
        "runtime": {
            "runtime_task_manifest_sha256": runtime_manifest[
                "runtime_task_manifest_sha256"
            ],
            "execution_commit": runtime_manifest["runtime_commit"],
            "runtime_file_set_sha256": runtime_manifest[
                "runtime_file_set_sha256"
            ],
            "execution_files": list(runtime_manifest["runtime_files"]),
            "git_proof": dict(git_proof),
        },
        "task_roster": {
            "expected_task_count": len(selected_tasks),
            "expected_task_ids": task_ids,
            "task_set_sha256": _sha(roster_rows),
            "frozen_task_count": int(runtime_phase["task_count"]),
            "frozen_task_set_sha256": runtime_phase["task_set_sha256"],
        },
        "budget": {
            "prior_cost_ledger_sha256": prior_cost_ledger[
                "prior_cost_ledger_sha256"
            ],
            "prior_known_cost_yuan": prior_cost_ledger["known_cost_yuan"],
            "prior_ambiguity_reserve_yuan": prior_cost_ledger[
                "pre_run_ambiguity_reserve_yuan"
            ],
            "prior_documented_cost_yuan": prior_cost_ledger[
                "pre_run_total_bound_yuan"
            ],
            "carried_forward_cost_ledger_sha256": carried_digest,
            "source_phase_receipt_sha256": carried_receipt,
            "source_record_set_sha256": carried_record_set,
            "carried_forward_known_cost_yuan": round(carried_known, 8),
            "carried_forward_unknown_reserve_yuan": round(carried_reserve, 8),
            "carried_forward_total_accounted_cost_yuan": round(carried_total, 8),
            "unknown_attempt_reserve_yuan": prior_cost_ledger[
                "unknown_attempt_reserve_yuan"
            ],
            "soft_warning_yuan": float(contract.config["budget_yuan"]["soft_warning"]),
            "hard_fuse_yuan": float(contract.config["budget_yuan"]["hard_fuse"]),
        },
    }
    artifact["phase_provenance_sha256"] = _sha(artifact)
    return artifact


def verify_phase_provenance(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema_version") != "yher.llm_sim_v2.phase_provenance.v1"
        or artifact.get("simulated") is not True
        or artifact.get("run_id") != RUN_ID
        or artifact.get("phase") not in {"pilot", "main"}
        or artifact.get("analysis_population") != artifact.get("phase")
    ):
        raise ValueError("phase provenance envelope is invalid")
    payload = dict(artifact)
    advertised = payload.pop("phase_provenance_sha256", None)
    if advertised != _sha(payload):
        raise ValueError("phase provenance digest mismatch")
    roster = artifact.get("task_roster")
    if (
        not isinstance(roster, Mapping)
        or roster.get("expected_task_count")
        != len(roster.get("expected_task_ids") or ())
        or len(set(roster.get("expected_task_ids") or ()))
        != roster.get("expected_task_count")
    ):
        raise ValueError("phase provenance task roster is invalid")
    return {
        "schema_version": "yher.llm_sim_v2.phase_provenance_proof.v1",
        "ok": True,
        "phase_provenance_sha256": advertised,
        "formal_analysis_eligible": artifact.get("formal_analysis_eligible") is True,
    }


def _validated_stored_git_proof(
    stored: Any,
    active: Any,
    *,
    repo_root: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(stored, Mapping) or not isinstance(active, Mapping):
        raise ValueError(f"phase provenance {label} git proof is missing")
    static_fields = (
        "schema_version",
        "ok",
        "commit",
        "ancestor_of_head",
        "byte_identical",
        "files",
    )
    if (
        any(stored.get(field) != active.get(field) for field in static_fields)
        or stored.get("ok") is not True
        or stored.get("ancestor_of_head") is not True
        or stored.get("byte_identical") is not True
        or stored.get("precedes_observation") is not True
    ):
        raise ValueError(f"phase provenance {label} git proof drifted")

    def parse_utc(value: Any, *, field: str) -> datetime:
        text = str(value or "").strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"phase provenance {label} {field} is invalid"
            ) from exc
        offset = parsed.utcoffset()
        if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError(f"phase provenance {label} {field} is not UTC")
        return parsed.astimezone(timezone.utc)

    def resolve_commit(value: Any, *, field: str) -> str:
        commit = str(value or "").strip().lower()
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise ValueError(f"phase provenance {label} {field} is invalid")
        try:
            resolved = subprocess.check_output(
                ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"phase provenance {label} {field} is not a git commit"
            ) from exc
        if resolved != commit:
            raise ValueError(f"phase provenance {label} {field} identity drifted")
        return commit

    def require_ancestor(ancestor: str, descendant: str, *, relation: str) -> None:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            capture_output=True,
        )
        if result.returncode == 1:
            raise ValueError(f"phase provenance {label} {relation} ancestry is invalid")
        if result.returncode != 0:
            raise RuntimeError(
                f"cannot verify phase provenance {label} {relation} ancestry"
            )

    bound_commit = resolve_commit(stored.get("commit"), field="commit")
    stored_head = resolve_commit(stored.get("current_head"), field="current_head")
    active_head = resolve_commit(
        active.get("current_head"), field="active current_head"
    )
    repository_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    if active_head != repository_head:
        raise ValueError(f"phase provenance {label} active current_head is stale")
    require_ancestor(bound_commit, stored_head, relation="commit-to-stored-head")
    require_ancestor(stored_head, active_head, relation="stored-to-active-head")

    stored_commit_text = str(stored.get("commit_timestamp_utc") or "").strip()
    active_commit_text = str(active.get("commit_timestamp_utc") or "").strip()
    actual_commit_epoch = int(
        subprocess.check_output(
            ["git", "show", "-s", "--format=%ct", bound_commit],
            cwd=repo_root,
            text=True,
        ).strip()
    )
    actual_commit_text = datetime.fromtimestamp(
        actual_commit_epoch,
        tz=timezone.utc,
    ).isoformat()
    if (
        stored_commit_text != active_commit_text
        or active_commit_text != actual_commit_text
    ):
        raise ValueError(f"phase provenance {label} commit timestamp drifted")
    commit_timestamp = parse_utc(stored_commit_text, field="commit timestamp")
    observation_text = _validate_runtime_timestamp(
        str(stored.get("observation_timestamp") or "")
    )
    observation_timestamp = parse_utc(
        observation_text,
        field="observation timestamp",
    )
    if not commit_timestamp < observation_timestamp:
        raise ValueError(
            f"phase provenance {label} commit does not precede observation"
        )
    return dict(stored)


def verify_phase_provenance_against_contract(
    artifact: Mapping[str, Any],
    *,
    contract: RuntimeContract,
    runtime_manifest: Mapping[str, Any],
    runtime_proof: Mapping[str, Any],
    tasks: Sequence[Task],
    collection_scope: Mapping[str, Any],
    carried_forward_cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proof = verify_phase_provenance(artifact)
    freeze_binding = artifact.get("freeze")
    runtime_binding = artifact.get("runtime")
    if not isinstance(freeze_binding, Mapping) or not isinstance(
        runtime_binding, Mapping
    ):
        raise ValueError("phase provenance contract bindings are missing")
    stored_freeze_proof = _validated_stored_git_proof(
        freeze_binding.get("git_proof"),
        contract.freeze_proof,
        repo_root=contract.repo_root,
        label="freeze",
    )
    stored_runtime_proof = _validated_stored_git_proof(
        runtime_binding.get("git_proof"),
        runtime_proof.get("git_proof"),
        repo_root=contract.repo_root,
        label="runtime",
    )
    first_observation = _validate_runtime_timestamp(
        str(artifact.get("first_observation_at_utc") or "")
    )
    if stored_runtime_proof["observation_timestamp"] != first_observation:
        raise ValueError("phase provenance runtime proof is not the first observation")
    if datetime.fromisoformat(
        stored_freeze_proof["observation_timestamp"].replace("Z", "+00:00")
    ) > datetime.fromisoformat(first_observation.replace("Z", "+00:00")):
        raise ValueError("phase provenance freeze proof postdates observation")
    expected_contract = replace(contract, freeze_proof=stored_freeze_proof)
    bound_runtime_proof = dict(runtime_proof)
    bound_runtime_proof["git_proof"] = stored_runtime_proof
    expected = build_phase_provenance(
        expected_contract,
        runtime_manifest=runtime_manifest,
        runtime_proof=bound_runtime_proof,
        phase=str(artifact["phase"]),
        tasks=tasks,
        collection_scope=collection_scope,
        prior_cost_ledger=contract.prior_cost_ledger,
        carried_forward_cost=carried_forward_cost,
        first_observation_at_utc=first_observation,
    )
    if dict(artifact) != expected:
        raise ValueError("stored phase provenance differs from the active contract")
    return {**proof, "contract_revalidated": True}


def validate_formal_phase_provenance(artifact: Mapping[str, Any]) -> dict[str, Any]:
    proof = verify_phase_provenance(artifact)
    if (
        artifact.get("collection_mode") != "formal"
        or artifact.get("development_only") is not False
        or artifact.get("partial") is not False
        or artifact.get("formal_analysis_eligible") is not True
    ):
        raise ValueError("partial or development-only phase is ineligible for formal analysis")
    budget = artifact.get("budget")
    if not isinstance(budget, Mapping) or not (
        budget.get("carried_forward_cost_ledger_sha256")
        == REVIEWED_CARRIED_LEDGER_SHA256
        and budget.get("source_phase_receipt_sha256")
        == REVIEWED_LEGACY_RECEIPT_SHA256
        and budget.get("source_record_set_sha256")
        == REVIEWED_LEGACY_RECORD_SET_SHA256
        and float(budget.get("carried_forward_known_cost_yuan", -1.0))
        == REVIEWED_LEGACY_KNOWN_COST_YUAN
        and float(budget.get("carried_forward_unknown_reserve_yuan", -1.0)) == 0.0
        and float(budget.get("carried_forward_total_accounted_cost_yuan", -1.0))
        == REVIEWED_LEGACY_KNOWN_COST_YUAN
    ):
        raise ValueError("formal phase lacks the reviewed carried-forward ledger")
    return proof


def write_phase_provenance(
    output_base: str | Path,
    *,
    phase: Mapping[str, Any],
) -> Path:
    verify_phase_provenance(phase)
    store = V2Store(output_base, phase=str(phase["phase"]))
    return store.write_json("phase_provenance.json", phase, immutable=True)


def phase_provenance_binding(artifact: Mapping[str, Any]) -> dict[str, Any]:
    verify_phase_provenance(artifact)
    return {
        "collection_mode": artifact["collection_mode"],
        "development_only": artifact["development_only"],
        "partial": artifact["partial"],
        "formal_analysis_eligible": artifact["formal_analysis_eligible"],
        "phase_provenance_sha256": artifact["phase_provenance_sha256"],
        "freeze_manifest_sha256": artifact["freeze"]["freeze_manifest_sha256"],
        "source_set_sha256": artifact["source"]["source_set_sha256"],
        "target_set_hash": artifact["target"]["target_set_hash"],
        "grid_sha256": artifact["grid_sha256"],
        "prompt_ledger_sha256": artifact["prompt"]["prompt_ledger_sha256"],
        "prompt_revision": artifact["prompt"]["revision"],
        "prompt_contract_sha256": artifact["prompt"]["prompt_contract_sha256"],
        "runtime_task_manifest_sha256": artifact["runtime"][
            "runtime_task_manifest_sha256"
        ],
        "execution_commit": artifact["runtime"]["execution_commit"],
        "runtime_file_set_sha256": artifact["runtime"][
            "runtime_file_set_sha256"
        ],
        "carried_forward_cost_ledger_sha256": artifact["budget"].get(
            "carried_forward_cost_ledger_sha256"
        ),
        "source_phase_receipt_sha256": artifact["budget"].get(
            "source_phase_receipt_sha256"
        ),
        "source_record_set_sha256": artifact["budget"].get(
            "source_record_set_sha256"
        ),
        "carried_forward_total_accounted_cost_yuan": artifact["budget"].get(
            "carried_forward_total_accounted_cost_yuan", 0.0
        ),
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


def _backoff(
    policy: ProviderPolicy,
    retry_index: int,
    *,
    random_value: Callable[[], float],
    retry_after_seconds: Any = None,
    wall_time: Callable[[], float] = time.time,
) -> float:
    base = min(
        policy.max_backoff_seconds,
        policy.base_backoff_seconds * (2**retry_index),
    )
    sample = min(1.0, max(0.0, float(random_value())))
    jitter = 1.0 + policy.jitter_fraction * (2.0 * sample - 1.0)
    retry_delay = 0.0
    if retry_after_seconds not in (None, ""):
        try:
            retry_delay = float(retry_after_seconds)
        except (TypeError, ValueError):
            try:
                parsed = parsedate_to_datetime(str(retry_after_seconds))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                retry_delay = parsed.timestamp() - float(wall_time())
            except (TypeError, ValueError, OverflowError):
                retry_delay = 0.0
    if not math.isfinite(retry_delay):
        retry_delay = 0.0
    return min(
        policy.max_backoff_seconds,
        max(max(0.0, retry_delay), max(0.0, base * jitter)),
    )


def _build_response_record(
    task: Task,
    *,
    provider: str,
    model: str,
    returned_model: str | None,
    final_status: str,
    final_error: str | None,
    parsed: Mapping[str, Any] | None,
    attempts: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    parsed_output = dict(parsed) if parsed is not None else None
    outcomes = (
        compute_outcomes(
            condition=task.condition,
            response_arm=task.response_arm,
            answer=parsed_output.get("answer") if parsed_output else None,
            abstain=bool(parsed_output.get("abstain")) if parsed_output else False,
            correct_option=task.correct_option,
            target_option=task.target_option,
        )
        if parsed_output
        else {
            "is_correct": None,
            "target_option_hit": None,
            "manipulation_compliance": None,
        }
    )
    attempt_rows = [dict(row) for row in attempts]
    known_cost_yuan = round(
        sum(
            float(row.get("cost_yuan") or 0.0)
            for row in attempt_rows
            if row.get("cost_known") is True
        ),
        8,
    )
    unknown_cost_reserve_yuan = round(
        sum(
            float(row.get("cost_reserve_yuan") or 0.0)
            for row in attempt_rows
            if row.get("cost_known") is False
        ),
        8,
    )
    has_unknown_cost_attempts = unknown_cost_reserve_yuan > 0.0
    return {
        "schema_version": "yher.llm_sim_v2.response_record.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": task.phase,
        "analysis_population": task.analysis_population,
        "collection_mode": provenance.get("collection_mode", "development_partial"),
        "development_only": provenance.get("development_only", True),
        "partial": provenance.get("partial", True),
        "formal_analysis_eligible": provenance.get("formal_analysis_eligible", False),
        "provider": provider,
        "model_id": returned_model,
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
        "parsed_output": parsed_output,
        "outcomes": outcomes,
        "attempts": attempt_rows,
        "retry_count": max(0, len(attempt_rows) - 1),
        "known_cost_yuan": known_cost_yuan,
        "unknown_cost_reserve_yuan": unknown_cost_reserve_yuan,
        "cost_yuan": round(known_cost_yuan + unknown_cost_reserve_yuan, 8),
        "has_unknown_cost_attempts": has_unknown_cost_attempts,
        "needs_user": has_unknown_cost_attempts,
        "needs_user_reasons": (
            ["unknown_provider_billing_reserved"]
            if has_unknown_cost_attempts
            else []
        ),
        "provenance": dict(provenance),
    }


def execute_task(
    task: Task,
    *,
    provider: str,
    model: str,
    transport: ProviderTransport,
    policy: ProviderPolicy,
    budget: BudgetLedger,
    provenance: Mapping[str, Any] | None = None,
    unknown_attempt_reserve_yuan: float = UNKNOWN_ATTEMPT_RESERVE_YUAN,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] | None = None,
    wall_time: Callable[[], float] = time.time,
    on_provider_call_started: Callable[[Mapping[str, Any]], None] | None = None,
    on_provider_call_interrupted: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    unknown_reserve = float(unknown_attempt_reserve_yuan)
    if (
        not math.isfinite(unknown_reserve)
        or unknown_reserve != UNKNOWN_ATTEMPT_RESERVE_YUAN
    ):
        raise ValueError("unknown attempt reserve must equal the frozen CNY 10 policy")
    retry_random = random_value or _SYSTEM_RANDOM.random
    if (
        on_provider_call_started is not None
        and on_provider_call_interrupted is None
    ):
        raise ValueError(
            "provider call evidence requires terminal interruption persistence"
        )
    provenance_binding = dict(provenance or {})
    attempts: list[dict[str, Any]] = []
    request_max_tokens = policy.max_tokens
    parsed: dict[str, Any] | None = None
    returned_model: str | None = None
    final_status = "technical_failure"
    final_error: str | None = None
    for attempt_index in range(policy.max_attempts):
        try:
            budget.assert_new_call_allowed()
        except BudgetFuseOpen:
            if not attempts:
                raise
            final_status = "technical_failure"
            final_error = "budget_fuse_open_after_attempt"
            break
        started = time.monotonic()
        try:
            if on_provider_call_started is not None:
                on_provider_call_started(
                    {
                        "task_id": task.task_id,
                        "attempt": attempt_index + 1,
                        "model": model,
                        "request_max_tokens": request_max_tokens,
                        "wire_message_sha256": task.wire_message_sha256,
                    }
                )
            try:
                response = transport.complete(
                    provider=provider,
                    model=model,
                    messages=list(task.wire_messages),
                    timeout_seconds=policy.timeout_seconds,
                    max_tokens=request_max_tokens,
                )
            except Exception:
                raise
            except BaseException:
                if on_provider_call_interrupted is None:
                    raise
                budget.add_cost(unknown_reserve)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "failed",
                        "error_category": "interrupted_provider_call",
                        "request_max_tokens": request_max_tokens,
                        "cost_yuan": None,
                        "cost_known": False,
                        "billing_ambiguity": True,
                        "cost_reserve_yuan": unknown_reserve,
                        "provider_response_received": False,
                    }
                )
                interrupted_record = _build_response_record(
                    task,
                    provider=provider,
                    model=model,
                    returned_model=returned_model,
                    final_status="technical_failure",
                    final_error="interrupted_provider_call",
                    parsed=None,
                    attempts=attempts,
                    provenance=provenance_binding,
                )
                on_provider_call_interrupted(interrupted_record)
                raise
            cost = max(0.0, float(response.get("cost_yuan") or 0.0))
            budget.add_cost(cost)
            returned_model = str(response.get("model_returned") or "").strip()
            usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
            response_content = str(response.get("content") or "")
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "response",
                    "request_max_tokens": request_max_tokens,
                    "latency_ms": round(float(response.get("latency_ms") or (time.monotonic() - started) * 1000), 3),
                    "model_returned": returned_model,
                    "finish_reason": str(response.get("finish_reason") or ""),
                    "usage": {
                        "input_tokens": max(0, int(usage.get("input_tokens") or 0)),
                        "output_tokens": max(0, int(usage.get("output_tokens") or 0)),
                    },
                    "cost_yuan": cost,
                    "cost_known": True,
                    "billing_ambiguity": False,
                    "cost_reserve_yuan": 0.0,
                    **bind_response_content(response_content),
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
                    response_content,
                    condition=task.condition,
                    option_keys=option_keys,
                )
                final_status = "complete"
                final_error = None
                break
            except InvalidProviderOutput as exc:
                final_status = "excluded_schema"
                final_error = str(exc)
                attempts[-1].update(
                    {
                        "status": "failed",
                        "error_category": "invalid_schema",
                    }
                )
                if attempt_index + 1 >= policy.max_attempts:
                    break
                sleep(
                    _backoff(
                        policy,
                        attempt_index,
                        random_value=retry_random,
                        wall_time=wall_time,
                    )
                )
                continue
        except BudgetFuseOpen:
            raise
        except Exception as exc:
            final_status = "technical_failure"
            final_error = _error_category(exc)
            if isinstance(exc, ProviderTruncatedResponseError):
                truncated_usage = exc.usage
                truncated_cost = max(0.0, float(exc.cost_yuan))
                budget.add_cost(truncated_cost)
                returned_model = str(exc.returned_model or "").strip() or returned_model
                attempt = {
                    "attempt": attempt_index + 1,
                    "status": "failed",
                    "error_category": final_error,
                    "request_max_tokens": int(exc.request_max_tokens),
                    "latency_ms": round(float(exc.latency_ms), 3),
                    "model_returned": str(exc.returned_model or ""),
                    "finish_reason": str(exc.finish_reason or ""),
                    "reasoning_tokens": max(0, int(exc.reasoning_tokens)),
                    "usage": {
                        "input_tokens": max(0, int(truncated_usage.get("input_tokens") or 0)),
                        "output_tokens": max(0, int(truncated_usage.get("output_tokens") or 0)),
                    },
                    "cost_yuan": truncated_cost,
                    "cost_known": True,
                    "billing_ambiguity": False,
                    "cost_reserve_yuan": 0.0,
                    "provider_response_received": False,
                }
            else:
                budget.add_cost(unknown_reserve)
                attempt = {
                    "attempt": attempt_index + 1,
                    "status": "failed",
                    "error_category": final_error,
                    "request_max_tokens": request_max_tokens,
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "cost_yuan": None,
                    "cost_known": False,
                    "billing_ambiguity": True,
                    "cost_reserve_yuan": unknown_reserve,
                    "provider_response_received": False,
                }
            attempts.append(
                attempt
            )
            if (
                isinstance(exc, ProviderTruncatedResponseError)
                and returned_model != model
            ):
                final_status = "excluded_model_drift"
                final_error = "returned_model_drift"
                break
            if attempt_index + 1 >= policy.max_attempts or not _retryable(exc):
                break
            if isinstance(exc, ProviderTruncatedResponseError):
                request_max_tokens = policy.retry_max_tokens
            sleep(
                _backoff(
                    policy,
                    attempt_index,
                    random_value=retry_random,
                    retry_after_seconds=getattr(exc, "retry_after_seconds", None),
                    wall_time=wall_time,
                )
            )

    return _build_response_record(
        task,
        provider=provider,
        model=model,
        returned_model=returned_model,
        final_status=final_status,
        final_error=final_error,
        parsed=parsed,
        attempts=attempts,
        provenance=provenance_binding,
    )


class V2ProviderRunner:
    def __init__(
        self,
        *,
        contract: RuntimeContract,
        output_base: str | Path,
        phase: str,
        provider: str,
        transport: ProviderTransport | None,
        budget: BudgetLedger,
        phase_provenance: Mapping[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.time,
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
        self.random_value = random_value or _SYSTEM_RANDOM.random
        self.clock = clock
        self.store = V2Store(output_base, phase=phase_name)
        if phase_provenance is not None:
            if phase_provenance.get("phase") != phase_name:
                raise ValueError("phase provenance does not match provider runner phase")
            self.provenance = phase_provenance_binding(phase_provenance)
            self.phase_provenance = dict(phase_provenance)
        else:
            self.provenance = {}
            self.phase_provenance = None
        self._write_lock = threading.Lock()
        self.evidence = ProviderEvidenceLedger(
            self.store.root,
            run_id=RUN_ID,
            phase=self.phase,
            provider=self.provider,
        )

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
        try:
            validate_v2_response_record(
                record,
                provider=self.provider,
                requested_model=self.model,
                phase=self.phase,
                task=task,
                expected_provenance=self.provenance,
            )
        except ValueError as exc:
            raise ValueError(
                f"existing v2 record failed evidence validation: {exc}"
            ) from exc
        return dict(record)

    def _write_record(self, task: Task, record: Mapping[str, Any]) -> None:
        with self._write_lock:
            self.store.write_json(
                Path("records") / self.provider / f"{task.task_id}.json",
                record,
                immutable=True,
            )

    def _execute_and_write(self, task: Task) -> dict[str, Any]:
        if self.transport is None:
            raise RuntimeError("provider transport is unavailable")
        record = execute_task(
            task,
            provider=self.provider,
            model=self.model,
            transport=self.transport,
            policy=self.policy,
            budget=self.budget,
            provenance=self.provenance,
            unknown_attempt_reserve_yuan=self.contract.prior_cost_ledger[
                "unknown_attempt_reserve_yuan"
            ],
            sleep=self.sleep,
            random_value=self.random_value,
            on_provider_call_started=lambda event: self.evidence.record_provider_call_started(
                task_id=str(event["task_id"]),
                attempt=int(event["attempt"]),
                model=str(event["model"]),
                request_max_tokens=int(event["request_max_tokens"]),
                wire_message_sha256=str(event["wire_message_sha256"]),
            ),
            on_provider_call_interrupted=lambda record: self._write_record(task, record),
        )
        self._write_record(task, record)
        return record

    def _condition_lifecycle(
        self,
        selected: Sequence[Task],
        records: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        minimum = int(
            self.contract.config["provider_exclusion"][
                "minimum_complete_clusters_per_condition"
            ]
        )
        threshold = float(
            self.contract.config["provider_exclusion"][
                "blind_invalid_schema_fraction_strictly_above"
            ]
        )
        output: dict[str, Any] = {}
        for condition in ("controlled", "blind"):
            condition_tasks = [task for task in selected if task.condition == condition]
            primary_tasks = [
                task for task in condition_tasks if not task.is_stability_repeat
            ]
            stability_repeat_tasks = [
                task for task in condition_tasks if task.is_stability_repeat
            ]
            condition_records = {
                task.task_id: records[task.task_id]
                for task in condition_tasks
                if task.task_id in records
            }
            primary_records = {
                task.task_id: records[task.task_id]
                for task in primary_tasks
                if task.task_id in records
            }
            invalid = sum(
                row.get("status") == "excluded_schema"
                for row in primary_records.values()
            )
            expected_count = len(condition_tasks)
            primary_expected_count = len(primary_tasks)
            invalid_fraction = (
                invalid / primary_expected_count if primary_expected_count else None
            )
            grouped: dict[str, list[Task]] = {}
            for task in primary_tasks:
                grouped.setdefault(task.persona_id, []).append(task)
            complete_clusters = sum(
                bool(tasks)
                and all(
                    task.task_id in records
                    and records[task.task_id].get("status") == "complete"
                    for task in tasks
                )
                for tasks in grouped.values()
            )
            output[condition] = {
                "expected_count": expected_count,
                "primary_expected_count": primary_expected_count,
                "stability_repeat_expected_count": len(stability_repeat_tasks),
                "present_count": len(condition_records),
                "missing_count": expected_count - len(condition_records),
                "invalid_schema_count": invalid,
                "invalid_schema_fraction": invalid_fraction,
                "excluded_invalid_schema": bool(
                    condition == "blind"
                    and invalid_fraction is not None
                    and invalid_fraction > threshold
                ),
                "complete_cluster_count": complete_clusters,
                "minimum_complete_clusters": minimum,
                "minimum_complete_clusters_met": complete_clusters >= minimum,
            }
        return output

    def _lifecycle_events(self) -> list[dict[str, Any]]:
        root = self.store.path(Path("provider_lifecycle") / self.provider)
        if not root.exists():
            return []
        events: list[dict[str, Any]] = []
        for path in sorted(root.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError("provider lifecycle event is not an object")
            payload = dict(value)
            advertised = payload.pop("lifecycle_event_sha256", None)
            if (
                value.get("schema_version")
                != "yher.llm_sim_v2.provider_lifecycle_event.v1"
                or value.get("run_id") != RUN_ID
                or value.get("phase") != self.phase
                or value.get("analysis_population") != self.phase
                or value.get("provider") != self.provider
                or value.get("event_index") != len(events)
                or advertised != _sha(payload)
                or not path.name.endswith(f"-{advertised}.json")
            ):
                raise ValueError("provider lifecycle event history is invalid")
            events.append(dict(value))
        return events

    def _append_lifecycle_event(self, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
        events = self._lifecycle_events()
        event: dict[str, Any] = {
            "schema_version": "yher.llm_sim_v2.provider_lifecycle_event.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "phase": self.phase,
            "analysis_population": self.phase,
            "provider": self.provider,
            "event_index": len(events),
            "provider_lifecycle": summary["provider_lifecycle"],
            "lifecycle": dict(summary["lifecycle"]),
            "interruption": dict(summary["interruption"]),
            "unavailable": dict(summary["unavailable"]),
            "needs_user": dict(summary["needs_user"]),
            "provenance": dict(summary["provenance"]),
            "finished_at_utc": summary["finished_at_utc"],
        }
        event["lifecycle_event_sha256"] = _sha(event)
        relative = (
            Path("provider_lifecycle")
            / self.provider
            / f"{len(events):04d}-{event['lifecycle_event_sha256']}.json"
        )
        self.store.write_json(relative, event, immutable=True)
        events.append(event)
        return [
            {
                "event_index": row["event_index"],
                "provider_lifecycle": row["provider_lifecycle"],
                "finished_at_utc": row["finished_at_utc"],
                "lifecycle_event_sha256": row["lifecycle_event_sha256"],
                "path": (
                    f"provider_lifecycle/{self.provider}/"
                    f"{int(row['event_index']):04d}-{row['lifecycle_event_sha256']}.json"
                ),
            }
            for row in events
        ]

    def _write_manifest(
        self,
        *,
        selected: Sequence[Task],
        records: Mapping[str, Mapping[str, Any]],
        resumed_count: int,
        fuse_skipped: set[str],
        breaker_skipped: set[str],
        interrupted: bool,
        interruption_type: str | None,
        breaker_opened_at_epoch: float | None,
        consecutive_failures: int,
        unavailable_error_category: str | None = None,
    ) -> dict[str, Any]:
        expected_ids = [task.task_id for task in selected]
        present_ids = [task_id for task_id in expected_ids if task_id in records]
        missing_ids = [task_id for task_id in expected_ids if task_id not in records]
        missing_set = set(missing_ids)
        fuse_ids = [task_id for task_id in expected_ids if task_id in fuse_skipped]
        breaker_ids = [task_id for task_id in expected_ids if task_id in breaker_skipped]
        classified = set(fuse_ids) | set(breaker_ids)
        interrupted_ids = (
            [task_id for task_id in missing_ids if task_id not in classified]
            if interrupted
            else []
        )
        counts = Counter(str(record.get("status")) for record in records.values())
        needs_user_ids = [
            task_id
            for task_id in expected_ids
            if task_id in records and records[task_id].get("needs_user") is True
        ]
        unknown_cost_attempt_count = sum(
            sum(
                attempt.get("cost_known") is False
                for attempt in record.get("attempts", ())
                if isinstance(attempt, Mapping)
            )
            for record in records.values()
        )
        record_known_cost_yuan = round(
            sum(float(record.get("known_cost_yuan") or 0.0) for record in records.values()),
            8,
        )
        record_unknown_reserve_yuan = round(
            sum(
                float(record.get("unknown_cost_reserve_yuan") or 0.0)
                for record in records.values()
            ),
            8,
        )
        if unavailable_error_category is not None:
            provider_lifecycle = "unavailable"
        elif interrupted:
            provider_lifecycle = "interrupted"
        elif fuse_ids:
            provider_lifecycle = "fuse_open"
        elif breaker_ids:
            provider_lifecycle = "excluded_repeated_failure"
        elif missing_ids:
            provider_lifecycle = "partial_missing"
        elif any(status != "complete" for status in counts):
            provider_lifecycle = "complete_with_exclusions"
        else:
            provider_lifecycle = "complete"
        resume_epoch = (
            breaker_opened_at_epoch + self.policy.cooldown_seconds
            if breaker_opened_at_epoch is not None
            else None
        )
        scope = self.phase_provenance or {}
        (self.store.root / "records" / self.provider).mkdir(
            parents=True,
            exist_ok=True,
        )
        record_set = build_provider_record_set(
            self.store.root,
            provider=self.provider,
            expected_task_ids=expected_ids,
        )
        summary = {
            "schema_version": "yher.llm_sim_v2.provider_manifest.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "phase": self.phase,
            "analysis_population": self.phase,
            "provider": self.provider,
            "provider_lifecycle": provider_lifecycle,
            "collection_mode": scope.get("collection_mode", "development_partial"),
            "development_only": scope.get("development_only", True),
            "partial": scope.get("partial", True),
            "formal_analysis_eligible": scope.get("formal_analysis_eligible", False),
            "requested_model": self.model,
            "returned_models": sorted(
                {
                    str(record["model_id"])
                    for record in records.values()
                    if record.get("model_id")
                }
            ),
            "freeze_commit": FROZEN_COMMIT,
            "prompt_revision": active_prompt_revision(self.contract)["revision"],
            "record_count": len(records),
            "record_set": record_set,
            "complete_records": counts.get("complete", 0),
            "resumed_records": resumed_count,
            "status_counts": dict(sorted(counts.items())),
            "lifecycle": {
                "expected_count": len(expected_ids),
                "present_count": len(present_ids),
                "missing_count": len(missing_ids),
                "interrupted_count": len(interrupted_ids),
                "fuse_skipped_count": len(fuse_ids),
                "breaker_skipped_count": len(breaker_ids),
                "expected_task_ids": expected_ids,
                "present_task_ids": present_ids,
                "missing_task_ids": missing_ids,
                "interrupted_task_ids": interrupted_ids,
                "fuse_skipped_task_ids": fuse_ids,
                "breaker_skipped_task_ids": breaker_ids,
                "unclassified_missing_task_ids": [
                    task_id
                    for task_id in missing_ids
                    if task_id not in classified and task_id not in interrupted_ids
                ],
            },
            "condition_lifecycle": self._condition_lifecycle(selected, records),
            "interruption": {
                "interrupted": interrupted,
                "type": interruption_type,
            },
            "unavailable": {
                "unavailable": unavailable_error_category is not None,
                "error_category": unavailable_error_category,
            },
            "needs_user": {
                "required": bool(needs_user_ids),
                "reason": (
                    "unknown_provider_billing_reserved"
                    if needs_user_ids
                    else None
                ),
                "record_count": len(needs_user_ids),
                "record_task_ids": needs_user_ids,
                "unknown_cost_attempt_count": unknown_cost_attempt_count,
            },
            "breaker": {
                "status": "open" if breaker_ids else "closed",
                "failure_threshold": self.policy.failure_threshold,
                "consecutive_failures": consecutive_failures,
                "cooldown_seconds": self.policy.cooldown_seconds,
                "opened_at_epoch": breaker_opened_at_epoch,
                "opened_at_utc": (
                    _utc_from_epoch(breaker_opened_at_epoch)
                    if breaker_opened_at_epoch is not None
                    else None
                ),
                "resume_not_before_epoch": resume_epoch,
                "resume_not_before_utc": (
                    _utc_from_epoch(resume_epoch) if resume_epoch is not None else None
                ),
            },
            "provenance": dict(self.provenance),
            "budget": {
                "total_cost_yuan": round(self.budget.total_cost_yuan, 8),
                "provider_record_known_cost_yuan": record_known_cost_yuan,
                "provider_record_unknown_reserve_yuan": (
                    record_unknown_reserve_yuan
                ),
                "provider_record_accounted_cost_yuan": round(
                    record_known_cost_yuan + record_unknown_reserve_yuan,
                    8,
                ),
                "soft_warning_triggered": self.budget.soft_warning_triggered,
                "hard_fuse_triggered": self.budget.hard_fuse_triggered
                or bool(fuse_ids),
            },
            "finished_at_utc": _utc_now(),
        }
        if set(present_ids) & missing_set:
            raise AssertionError("provider lifecycle present/missing sets overlap")
        summary["lifecycle_history"] = self._append_lifecycle_event(summary)
        self.store.write_json(
            Path("provider_manifests") / f"{self.provider}.json",
            summary,
            immutable=False,
        )
        return summary

    def write_unavailable_manifest(
        self,
        tasks: Sequence[Task],
        *,
        error_category: str,
    ) -> dict[str, Any]:
        with self.evidence.provider_lock():
            return self._write_unavailable_manifest_locked(
                tasks,
                error_category=error_category,
            )

    def _write_unavailable_manifest_locked(
        self,
        tasks: Sequence[Task],
        *,
        error_category: str,
    ) -> dict[str, Any]:
        selected = [task for task in tasks if task.phase == self.phase]
        if self.phase_provenance is not None and [task.task_id for task in selected] != list(
            self.phase_provenance["task_roster"]["expected_task_ids"]
        ):
            raise ValueError("provider task roster differs from phase provenance")
        records = {
            task.task_id: record
            for task in selected
            if (record := self._read_existing(task)) is not None
        }
        invocation = self.evidence.begin_invocation(
            expected_task_ids=[task.task_id for task in selected],
            resumed_task_ids=list(records),
        )
        summary = self._write_manifest(
            selected=selected,
            records=records,
            resumed_count=len(records),
            fuse_skipped=set(),
            breaker_skipped=set(),
            interrupted=False,
            interruption_type=None,
            breaker_opened_at_epoch=None,
            consecutive_failures=0,
            unavailable_error_category=str(error_category),
        )
        receipt = self.evidence.finish_invocation(
            invocation,
            status="unavailable",
        )
        summary["evidence_receipt"] = receipt
        return summary

    def run_tasks(self, tasks: Sequence[Task]) -> dict[str, Any]:
        with self.evidence.provider_lock():
            return self._run_tasks_locked(tasks)

    def _run_tasks_locked(self, tasks: Sequence[Task]) -> dict[str, Any]:
        selected = [task for task in tasks if task.phase == self.phase]
        if len({task.task_id for task in selected}) != len(selected):
            raise ValueError("provider task roster contains duplicate task IDs")
        if self.phase_provenance is not None and [task.task_id for task in selected] != list(
            self.phase_provenance["task_roster"]["expected_task_ids"]
        ):
            raise ValueError("provider task roster differs from phase provenance")
        existing: dict[str, dict[str, Any]] = {}
        pending: list[Task] = []
        for task in selected:
            record = self._read_existing(task)
            if record is None:
                pending.append(task)
            else:
                existing[task.task_id] = record
        records: dict[str, dict[str, Any]] = dict(existing)
        fuse_skipped: set[str] = set()
        breaker_skipped: set[str] = set()
        interrupted = False
        interruption_type: str | None = None
        breaker_opened_at_epoch: float | None = None
        consecutive_failures = 0
        for task in selected:
            record = existing.get(task.task_id)
            if record is None:
                break
            if record.get("status") == "complete":
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        previous = self.store.read_json(
            Path("provider_manifests") / f"{self.provider}.json"
        )
        previous_breaker = previous.get("breaker") if isinstance(previous, Mapping) else None
        if isinstance(previous_breaker, Mapping):
            resume_epoch = previous_breaker.get("resume_not_before_epoch")
            if (
                previous_breaker.get("status") == "open"
                and isinstance(resume_epoch, (int, float))
                and self.clock() < float(resume_epoch)
            ):
                breaker_opened_at_epoch = float(previous_breaker["opened_at_epoch"])
                consecutive_failures = int(
                    previous_breaker.get("consecutive_failures") or 0
                )
                breaker_skipped.update(task.task_id for task in pending)
                pending.clear()
            elif previous_breaker.get("status") == "open":
                consecutive_failures = 0
        if consecutive_failures >= self.policy.failure_threshold and pending:
            breaker_opened_at_epoch = self.clock()
            breaker_skipped.update(task.task_id for task in pending)
            pending.clear()
        invocation = self.evidence.begin_invocation(
            expected_task_ids=[task.task_id for task in selected],
            resumed_task_ids=list(existing),
        )
        summary: dict[str, Any] | None = None
        invocation_status = "complete"
        try:
            while pending:
                if self.budget.hard_fuse_triggered:
                    fuse_skipped.update(task.task_id for task in pending)
                    pending.clear()
                    break
                batch = pending[: self.policy.concurrency]
                del pending[: len(batch)]
                batch_records: dict[str, dict[str, Any]] = {}
                pool = ThreadPoolExecutor(max_workers=len(batch))
                futures = {
                    pool.submit(self._execute_and_write, task): task for task in batch
                }
                try:
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            record = future.result()
                        except BudgetFuseOpen:
                            fuse_skipped.add(task.task_id)
                            continue
                        records[task.task_id] = record
                        batch_records[task.task_id] = record
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
                for task in batch:
                    record = batch_records.get(task.task_id)
                    if record is None:
                        continue
                    if record.get("status") == "complete":
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= self.policy.failure_threshold:
                            breaker_opened_at_epoch = self.clock()
                            breaker_skipped.update(task.task_id for task in pending)
                            pending.clear()
                            break
                if self.budget.hard_fuse_triggered and pending:
                    fuse_skipped.update(task.task_id for task in pending)
                    pending.clear()
        except BaseException as exc:
            interrupted = True
            interruption_type = type(exc).__name__
            invocation_status = "interrupted"
            raise
        finally:
            # Workers persist before returning, so a completed in-flight response
            # remains resumable even when the coordinator is interrupted.
            for task in selected:
                if task.task_id not in records:
                    record = self._read_existing(task)
                    if record is not None:
                        records[task.task_id] = record
            summary = self._write_manifest(
                selected=selected,
                records=records,
                resumed_count=len(existing),
                fuse_skipped=fuse_skipped,
                breaker_skipped=breaker_skipped,
                interrupted=interrupted,
                interruption_type=interruption_type,
                breaker_opened_at_epoch=breaker_opened_at_epoch,
                consecutive_failures=consecutive_failures,
            )
            receipt = self.evidence.finish_invocation(
                invocation,
                status=invocation_status,
            )
            summary["evidence_receipt"] = receipt
        assert summary is not None
        return summary


__all__ = [
    "BudgetFuseOpen",
    "BudgetLedger",
    "InvalidProviderOutput",
    "ProviderPolicy",
    "RUNTIME_PATHS",
    "RuntimeContract",
    "Task",
    "UNKNOWN_ATTEMPT_RESERVE_YUAN",
    "V2ProviderRunner",
    "active_prompt_revision",
    "build_phase_provenance",
    "compute_outcomes",
    "build_runtime_task_manifest",
    "enumerate_tasks",
    "execute_task",
    "load_runtime_contract",
    "parse_provider_output",
    "phase_provenance_binding",
    "validate_formal_phase_provenance",
    "verify_phase_provenance",
    "verify_phase_provenance_against_contract",
    "verify_prior_cost_ledger",
    "verify_runtime_task_manifest",
    "write_phase_provenance",
]
