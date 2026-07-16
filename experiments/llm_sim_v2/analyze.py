"""Fail-closed analysis for the Persona-v2 dual-condition main study."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ID = "llm-personas-v2-dual"
FROZEN_PROVIDERS = (
    "deepseek",
    "glm",
    "kimi",
    "minimax",
    "doubao",
    "tongyi",
)
BOOTSTRAP_SEED = 2026071503
BOOTSTRAP_RESAMPLES = 10_000
CONTROLLED_STATES = (
    "correct_answer",
    "incorrect_answer",
    "abstention",
    "technical_or_schema_failure",
)
_PROVIDER_IDENTITY_ALIASES = {
    "deepseek": ("deepseek",),
    "glm": ("glm", "zhipu"),
    "kimi": ("kimi", "moonshot"),
    "minimax": ("minimax",),
    "doubao": ("doubao", "bytedance"),
    "tongyi": ("tongyi", "qwen", "dashscope", "alibaba"),
}
_UNICODE_ESCAPE_PATTERN = re.compile(r"\\u([0-9a-fA-F]{4})")
_PROVIDER_POLICY_FIELDS = {
    "max_attempts",
    "allowed_request_max_tokens",
    "max_tokens",
    "retry_max_tokens",
    "timeout_seconds",
    "concurrency",
    "failure_threshold",
    "base_backoff_seconds",
    "max_backoff_seconds",
    "cooldown_seconds",
    "jitter_fraction",
}
_CARRIED_COST_LEDGER_REL = Path(
    "experiments/llm_sim_v2/evidence_anchors/legacy_pilot_carried_forward_cost.json"
)
_MAIN_PHASE_RECEIPT_REL = Path(
    "experiments/llm_sim_v2/evidence_anchors/main_phase_evidence_receipt.json"
)
_JUDGE_RUN_RECEIPT_REL = Path(
    "experiments/llm_sim_v2/evidence_anchors/judge_run_evidence_receipt.json"
)


class AnalysisContractError(ValueError):
    """Raised before analysis when frozen input identity cannot be proven."""


_FORMAL_LOADER_SENTINEL = object()


class _FormalLoaderProof:
    __slots__ = ("bundle_sha256",)

    def __init__(self, sentinel: object, *, bundle_sha256: str) -> None:
        if sentinel is not _FORMAL_LOADER_SENTINEL:
            raise AnalysisContractError("formal loader proof cannot be constructed externally")
        self.bundle_sha256 = bundle_sha256


class _FormalPublicationProof:
    __slots__ = (
        "loader_bundle_sha256",
        "result_sha256",
        "input_artifact_manifest_sha256",
        "authorization_mac",
    )

    def __init__(
        self,
        *,
        loader_bundle_sha256: str,
        result_sha256: str,
        input_artifact_manifest_sha256: str,
        authorization_mac: str,
    ) -> None:
        self.loader_bundle_sha256 = loader_bundle_sha256
        self.result_sha256 = result_sha256
        self.input_artifact_manifest_sha256 = input_artifact_manifest_sha256
        self.authorization_mac = authorization_mac


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _verify_internal_digest(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    advertised = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    if not isinstance(advertised, str) or advertised != _canonical_sha(payload):
        raise AnalysisContractError(f"{label} digest mismatch")
    return advertised


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisContractError(f"{label} must be an object")
    return value


def _utc_timestamp(value: Any, label: str) -> str:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnalysisContractError(f"{label} is not an ISO-8601 timestamp") from exc
    if not text.endswith("Z") or parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise AnalysisContractError(f"{label} must be an explicit UTC timestamp")
    return text


_FORMAL_LOADER_BUNDLE_FIELDS = (
    "expected_tasks",
    "runtime_manifest",
    "phase_provenance",
    "records_by_provider",
    "provider_manifests",
    "mapping_manifest",
    "active_contract_proof",
    "cost_accounting",
    "judge_result_manifests",
    "judge_run_evidence",
    "judge_artifact_roots",
    "judge_artifact_sources",
    "input_artifact_manifest",
    "phase_evidence",
)


def _formal_loader_bundle_sha256(values: Mapping[str, Any]) -> str:
    if any(field not in values for field in _FORMAL_LOADER_BUNDLE_FIELDS):
        raise AnalysisContractError("formal loader bundle is incomplete")
    return _canonical_sha(
        {field: values[field] for field in _FORMAL_LOADER_BUNDLE_FIELDS}
    )


def validate_inputs(
    *,
    phase_provenance: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    expected_tasks: Sequence[Mapping[str, Any]],
    active_contract_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate formal-main provenance and the expected task denominator."""

    phase = _mapping(phase_provenance, "phase provenance")
    runtime = _mapping(runtime_manifest, "runtime manifest")
    _verify_internal_digest(
        phase, field="phase_provenance_sha256", label="phase provenance"
    )
    _verify_internal_digest(
        runtime,
        field="runtime_task_manifest_sha256",
        label="runtime task manifest",
    )
    active = _mapping(active_contract_proof, "active analysis contract proof")
    active_digest = _verify_internal_digest(
        active,
        field="active_analysis_contract_proof_sha256",
        label="active analysis contract proof",
    )
    phase_source = _mapping(phase.get("source"), "phase source binding")
    phase_target = _mapping(phase.get("target"), "phase target binding")
    phase_budget = _mapping(phase.get("budget"), "phase budget binding")
    provider_models = active.get("provider_models")
    provider_attempt_policies = active.get("provider_attempt_policies")
    if (
        active.get("schema_version")
        != "yher.llm_sim_v2.active_analysis_contract_proof.v1"
        or active.get("ok") is not True
        or active.get("runtime_git_verified") is not True
        or active.get("contract_revalidated") is not True
        or active.get("request_temperature") != 0.0
        or active.get("runtime_task_manifest_sha256")
        != runtime.get("runtime_task_manifest_sha256")
        or active.get("phase_provenance_sha256")
        != phase.get("phase_provenance_sha256")
        or active.get("source_set_sha256") != phase_source.get("source_set_sha256")
        or active.get("target_set_hash") != phase_target.get("target_set_hash")
        or active.get("carried_forward_cost_ledger_sha256")
        != phase_budget.get("carried_forward_cost_ledger_sha256")
        or active.get("source_record_set_sha256")
        != phase_budget.get("source_record_set_sha256")
        or not isinstance(provider_models, Mapping)
        or set(provider_models) != set(FROZEN_PROVIDERS)
        or any(not str(provider_models[provider]).strip() for provider in FROZEN_PROVIDERS)
        or not isinstance(provider_attempt_policies, Mapping)
        or set(provider_attempt_policies) != set(FROZEN_PROVIDERS)
        or not isinstance(active.get("frozen_leakage_lexicon"), list)
    ):
        raise AnalysisContractError(
            "phase/runtime identity is not bound to the active runner contract"
        )
    for provider in FROZEN_PROVIDERS:
        policy = _mapping(
            provider_attempt_policies[provider],
            f"{provider} active attempt policy",
        )
        allowed_tokens = policy.get("allowed_request_max_tokens")
        try:
            max_tokens = int(policy.get("max_tokens"))
            retry_max_tokens = int(policy.get("retry_max_tokens"))
            timeout_seconds = float(policy.get("timeout_seconds"))
            base_backoff_seconds = float(policy.get("base_backoff_seconds"))
            max_backoff_seconds = float(policy.get("max_backoff_seconds"))
            cooldown_seconds = float(policy.get("cooldown_seconds"))
            jitter_fraction = float(policy.get("jitter_fraction"))
        except (TypeError, ValueError) as exc:
            raise AnalysisContractError(
                f"{provider} active attempt policy is invalid"
            ) from exc
        if (
            set(policy) != _PROVIDER_POLICY_FIELDS
            or
            not isinstance(policy.get("max_attempts"), int)
            or isinstance(policy.get("max_attempts"), bool)
            or int(policy["max_attempts"]) < 1
            or not isinstance(allowed_tokens, list)
            or not allowed_tokens
            or len(allowed_tokens) != len(set(allowed_tokens))
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                for value in allowed_tokens
            )
            or not isinstance(policy.get("concurrency"), int)
            or isinstance(policy.get("concurrency"), bool)
            or int(policy["concurrency"]) < 1
            or not isinstance(policy.get("failure_threshold"), int)
            or isinstance(policy.get("failure_threshold"), bool)
            or int(policy["failure_threshold"]) < 1
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
            or max_tokens < 1
            or retry_max_tokens < max_tokens
            or allowed_tokens != sorted({max_tokens, retry_max_tokens})
            or not math.isfinite(base_backoff_seconds)
            or base_backoff_seconds < 0.0
            or not math.isfinite(max_backoff_seconds)
            or max_backoff_seconds < base_backoff_seconds
            or not math.isfinite(cooldown_seconds)
            or cooldown_seconds <= 0.0
            or not math.isfinite(jitter_fraction)
            or not 0.0 <= jitter_fraction <= 1.0
        ):
            raise AnalysisContractError(
                f"{provider} active attempt policy is invalid"
            )
    if (
        phase.get("schema_version") != "yher.llm_sim_v2.phase_provenance.v1"
        or phase.get("simulated") is not True
        or phase.get("run_id") != RUN_ID
        or phase.get("phase") != "main"
        or phase.get("analysis_population") != "main"
    ):
        raise AnalysisContractError("analysis accepts only simulated formal main provenance")
    if (
        phase.get("collection_mode") != "formal"
        or phase.get("development_only") is not False
        or phase.get("partial") is not False
        or phase.get("formal_analysis_eligible") is not True
        or phase.get("task_limit") is not None
    ):
        raise AnalysisContractError("partial or development collection is not formal-analysis eligible")
    if phase.get("modality_condition") != "text_only":
        raise AnalysisContractError("formal main provenance must bind text_only modality")
    if (
        runtime.get("schema_version")
        != "yher.llm_sim_v2.runtime_task_manifest.v1"
        or runtime.get("simulated") is not True
        or runtime.get("run_id") != RUN_ID
    ):
        raise AnalysisContractError("runtime task manifest envelope is invalid")

    selected = phase.get("selected_providers")
    frozen = phase.get("frozen_providers")
    if selected != list(FROZEN_PROVIDERS) or frozen != list(FROZEN_PROVIDERS):
        raise AnalysisContractError("formal main must retain the frozen six-provider roster")
    phases = _mapping(runtime.get("phases"), "runtime phases")
    runtime_main = _mapping(phases.get("main"), "runtime main phase")
    if runtime_main.get("providers") != list(FROZEN_PROVIDERS):
        raise AnalysisContractError("runtime main provider roster drifted")

    roster = _mapping(phase.get("task_roster"), "phase task roster")
    phase_ids = roster.get("expected_task_ids")
    runtime_ids = runtime_main.get("task_ids")
    if not isinstance(phase_ids, list) or not all(
        isinstance(value, str) and value for value in phase_ids
    ):
        raise AnalysisContractError("phase task roster IDs are invalid")
    if len(phase_ids) != len(set(phase_ids)):
        raise AnalysisContractError("phase task roster contains duplicate IDs")
    task_rows = tuple(expected_tasks)
    task_ids = [str(row.get("task_id") or "") for row in task_rows]
    if (
        runtime_ids != phase_ids
        or task_ids != phase_ids
        or roster.get("expected_task_count") != len(phase_ids)
        or roster.get("frozen_task_count") != len(phase_ids)
        or runtime_main.get("task_count") != len(phase_ids)
        or roster.get("task_set_sha256") != runtime_main.get("task_set_sha256")
        or roster.get("frozen_task_set_sha256")
        != runtime_main.get("task_set_sha256")
    ):
        raise AnalysisContractError("runtime, phase, and reconstructed task rosters differ")
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise AnalysisContractError("reconstructed task roster is invalid")

    phase_runtime = _mapping(phase.get("runtime"), "phase runtime binding")
    if (
        phase_runtime.get("runtime_task_manifest_sha256")
        != runtime.get("runtime_task_manifest_sha256")
        or phase_runtime.get("execution_commit") != runtime.get("runtime_commit")
        or phase_runtime.get("runtime_file_set_sha256")
        != runtime.get("runtime_file_set_sha256")
    ):
        raise AnalysisContractError("phase provenance runtime binding drifted")
    phase_prompt = _mapping(phase.get("prompt"), "phase prompt binding")
    if (
        phase_prompt.get("revision") != runtime.get("prompt_revision")
        or phase_prompt.get("prompt_contract_sha256")
        != runtime.get("prompt_contract_sha256")
        or phase_prompt.get("prompt_ledger_sha256")
        != runtime.get("prompt_ledger_sha256")
    ):
        raise AnalysisContractError("phase provenance prompt binding drifted")
    phase_freeze = _mapping(phase.get("freeze"), "phase freeze binding")
    if phase_freeze.get("freeze_manifest_sha256") != runtime.get(
        "freeze_manifest_sha256"
    ):
        raise AnalysisContractError("phase provenance freeze binding drifted")

    required_task_fields = {
        "task_id",
        "persona_id",
        "pair_id",
        "row_id",
        "anchor_id",
        "response_arm",
        "condition",
        "item_id",
        "correct_option",
        "target_node",
        "family_id",
        "attempt_id",
        "logical_key",
        "message_sha256",
        "wire_message_sha256",
        "public_question",
        "item_contract",
        "persona_contract",
        "is_stability_repeat",
        "is_terminal",
        "target_option",
        "random_wrong_option_baseline",
    }
    for row in task_rows:
        if not isinstance(row, Mapping) or not required_task_fields.issubset(row):
            raise AnalysisContractError("reconstructed task row is incomplete")
        if row.get("response_arm") not in {"deficit", "control"}:
            raise AnalysisContractError("task response arm drifted")
        if row.get("condition") not in {"controlled", "blind"}:
            raise AnalysisContractError("task condition drifted")
        if row.get("is_stability_repeat") is True and row.get("condition") != "blind":
            raise AnalysisContractError("stability repeat must be a blind task")
        target_option = row.get("target_option")
        random_baseline = row.get("random_wrong_option_baseline")
        if target_option is None and random_baseline is not None:
            raise AnalysisContractError("unmapped task cannot define a random-wrong baseline")
        if target_option is not None and (
            not isinstance(random_baseline, (int, float))
            or isinstance(random_baseline, bool)
            or not 0.0 < float(random_baseline) < 1.0
        ):
            raise AnalysisContractError("mapped task random-wrong baseline is invalid")
    personas = sorted({str(row["persona_id"]) for row in task_rows})
    if len(personas) != 50:
        raise AnalysisContractError("formal main requires exactly 50 persona_id clusters")
    for persona_id in personas:
        arms = {
            str(row["response_arm"])
            for row in task_rows
            if row["persona_id"] == persona_id
        }
        if arms != {"deficit", "control"}:
            raise AnalysisContractError("every persona cluster requires paired response arms")
    return {
        "schema_version": "yher.llm_sim_v2.analysis_input_proof.v1",
        "ok": True,
        "analysis_population": "main",
        "expected_task_count": len(task_rows),
        "persona_cluster_count": len(personas),
        "providers": list(FROZEN_PROVIDERS),
        "runtime_task_manifest_sha256": runtime["runtime_task_manifest_sha256"],
        "phase_provenance_sha256": phase["phase_provenance_sha256"],
        "active_analysis_contract_proof_sha256": active_digest,
    }


def controlled_response_state(record: Mapping[str, Any] | None) -> str:
    """Classify one expected controlled cell into the frozen four states."""

    if not isinstance(record, Mapping) or record.get("status") != "complete":
        return "technical_or_schema_failure"
    parsed = record.get("parsed_output")
    if not isinstance(parsed, Mapping):
        raise AnalysisContractError("complete record lacks a parsed output")
    if parsed.get("answer") is None:
        return "abstention"
    outcomes = record.get("outcomes")
    if not isinstance(outcomes, Mapping) or not isinstance(
        outcomes.get("is_correct"), bool
    ):
        raise AnalysisContractError("complete answered record lacks correctness")
    return "correct_answer" if outcomes["is_correct"] else "incorrect_answer"


def cluster_bootstrap_mean(
    values_by_provider: Mapping[str, Mapping[str, float | None]],
    *,
    persona_ids: Sequence[str],
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap whole persona clusters and average fixed providers equally."""

    personas = tuple(str(value) for value in persona_ids)
    if not personas or len(personas) != len(set(personas)):
        raise AnalysisContractError("bootstrap persona IDs must be unique and non-empty")
    providers = tuple(sorted(str(value) for value in values_by_provider))
    if not providers or int(resamples) < 1:
        raise AnalysisContractError("bootstrap requires providers and positive resamples")
    arrays: list[np.ndarray] = []
    provider_points: dict[str, float | None] = {}
    for provider in providers:
        unknown = set(values_by_provider[provider]) - set(personas)
        if unknown:
            raise AnalysisContractError("bootstrap values contain an unknown persona")
        array = np.asarray(
            [
                np.nan
                if values_by_provider[provider].get(persona) is None
                else float(values_by_provider[provider][persona])
                for persona in personas
            ],
            dtype=float,
        )
        if np.any(np.isinf(array)):
            raise AnalysisContractError("bootstrap values must be finite or null")
        finite = array[np.isfinite(array)]
        provider_points[provider] = float(finite.mean()) if finite.size else None
        arrays.append(array)
    point = (
        float(np.mean([value for value in provider_points.values() if value is not None]))
        if all(value is not None for value in provider_points.values())
        else None
    )
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(personas), size=(int(resamples), len(personas)))
    estimates: list[np.ndarray] = []
    for array in arrays:
        sampled = array[draws]
        denominator = np.isfinite(sampled).sum(axis=1)
        numerator = np.nansum(sampled, axis=1)
        estimates.append(
            np.divide(
                numerator,
                denominator,
                out=np.full(int(resamples), np.nan, dtype=float),
                where=denominator > 0,
            )
        )
    stacked = np.vstack(estimates)
    aggregate = np.mean(stacked, axis=0)
    aggregate[~np.all(np.isfinite(stacked), axis=0)] = np.nan
    defined = aggregate[np.isfinite(aggregate)]
    interval = (
        [
            float(np.quantile(defined, 0.025)),
            float(np.quantile(defined, 0.975)),
        ]
        if defined.size
        else None
    )
    return {
        "point_estimate": point,
        "ci95": interval,
        "seed": int(seed),
        "resamples": int(resamples),
        "defined_resamples": int(defined.size),
        "undefined_resamples": int(resamples) - int(defined.size),
        "provider_equal_weighting": True,
        "provider_point_estimates": provider_points,
    }


def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        raise AnalysisContractError("Cohen kappa requires paired non-empty ratings")
    categories = sorted(set(left) | set(right))
    count = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / count
    expected = sum(
        (left.count(category) / count) * (right.count(category) / count)
        for category in categories
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    value = (observed - expected) / (1.0 - expected)
    return float(value) if math.isfinite(value) else None


def pairwise_terminal_agreement(
    ratings_by_provider: Mapping[str, Mapping[str, str]],
    *,
    subjects: Sequence[str],
) -> dict[str, Any]:
    """Report pairwise exact agreement and Cohen kappa, retaining NC."""

    subject_ids = tuple(str(value) for value in subjects)
    if not subject_ids or len(subject_ids) != len(set(subject_ids)):
        raise AnalysisContractError("agreement subjects must be unique and non-empty")
    providers = tuple(sorted(str(value) for value in ratings_by_provider))
    if len(providers) < 2:
        raise AnalysisContractError("pairwise agreement requires at least two providers")
    normalized: dict[str, list[str]] = {}
    for provider in providers:
        unknown = set(ratings_by_provider[provider]) - set(subject_ids)
        if unknown:
            raise AnalysisContractError("agreement ratings contain an unknown subject")
        normalized[provider] = [
            str(ratings_by_provider[provider].get(subject, "NC"))
            for subject in subject_ids
        ]
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(providers):
        for right in providers[left_index + 1 :]:
            left_values = normalized[left]
            right_values = normalized[right]
            numerator = sum(
                a == b for a, b in zip(left_values, right_values, strict=True)
            )
            pairs.append(
                {
                    "provider_left": left,
                    "provider_right": right,
                    "exact_agreement_numerator": numerator,
                    "denominator": len(subject_ids),
                    "exact_agreement": numerator / len(subject_ids),
                    "cohen_kappa": _cohen_kappa(left_values, right_values),
                }
            )
    categories = sorted(
        {value for values in normalized.values() for value in values} | {"NC"}
    )
    return {
        "subjects": list(subject_ids),
        "providers": list(providers),
        "categories": categories,
        "nc_retained": True,
        "pairs": pairs,
    }


def judge_input_bytes(case_manifest: Mapping[str, Any]) -> bytes:
    """Render canonical JSONL bytes shared identically by both frozen judges."""

    cases = case_manifest.get("cases")
    if not isinstance(cases, list):
        raise AnalysisContractError("judge case manifest has no case list")
    rows: list[bytes] = []
    seen: set[str] = set()
    for case_value in cases:
        case = _mapping(case_value, "judge case")
        case_id = str(case.get("case_id") or "")
        messages = case.get("judge_messages")
        if (
            not case_id
            or case_id in seen
            or not isinstance(messages, list)
            or not messages
        ):
            raise AnalysisContractError("judge case ID or messages are invalid")
        seen.add(case_id)
        rows.append(_canonical_bytes({"case_id": case_id, "messages": messages}) + b"\n")
    return b"".join(rows)


def _normalize_identity_term(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def _identity_string_surfaces(value: Any, *, _depth: int = 0) -> list[str]:
    if _depth > 20:
        return []
    if isinstance(value, Mapping):
        return [
            surface
            for key, child in value.items()
            for surface in (
                *_identity_string_surfaces(key, _depth=_depth + 1),
                *_identity_string_surfaces(child, _depth=_depth + 1),
            )
        ]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            surface
            for child in value
            for surface in _identity_string_surfaces(child, _depth=_depth + 1)
        ]
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return []
    if not isinstance(value, str):
        return []
    normalized = _normalize_identity_term(value)
    surfaces = [normalized] if normalized else []
    decoded_unicode = _UNICODE_ESCAPE_PATTERN.sub(
        lambda match: chr(int(match.group(1), 16)), value
    )
    decoded_normalized = _normalize_identity_term(decoded_unicode)
    if decoded_normalized and decoded_normalized not in surfaces:
        surfaces.append(decoded_normalized)
    stripped = value.strip()
    if stripped.startswith(("{", "[", '"')):
        try:
            nested = json.loads(stripped)
        except json.JSONDecodeError:
            nested = value
        if nested != value:
            surfaces.extend(
                _identity_string_surfaces(nested, _depth=_depth + 1)
            )
    return surfaces


def _provider_identity_hits(value: Any, normalized_terms: Sequence[str]) -> list[str]:
    surfaces = _identity_string_surfaces(value)
    return sorted(
        {
            term
            for term in normalized_terms
            if term and any(term in surface for surface in surfaces)
        }
    )


def build_judge_case_manifest(
    candidates: Sequence[Mapping[str, Any]],
    *,
    frozen_leakage_lexicon: Sequence[str],
    provider_identity_terms: Sequence[str] = (),
) -> dict[str, Any]:
    """Select the frozen 80/40 exploratory adjudication sample and export it."""

    from .judge_protocol import (
        JUDGE_PUBLIC_SCHEMA_KEYS,
        judge_protocol,
        judge_public_question_payload,
        render_judge_export,
    )

    protocol = judge_protocol()
    amendment_path = Path(__file__).with_name("judge_amendment_20260716.md")
    amendment_bytes = amendment_path.read_bytes()
    normalized: dict[str, list[dict[str, Any]]] = {
        "disagreement": [],
        "agreement": [],
    }
    identity_terms = sorted(
        {
            normalized_term
            for value in provider_identity_terms
            if (normalized_term := _normalize_identity_term(value))
        }
    )
    pre_exclusion_counts = {"disagreement": 0, "agreement": 0}
    excluded_identity_rows: list[dict[str, Any]] = []
    excluded_target_label_rows: list[dict[str, Any]] = []
    target_terms_by_case: dict[str, tuple[str, ...]] = {}
    identities: set[str] = set()
    ordered_candidates = sorted(
        candidates,
        key=lambda value: (
            str(value.get("candidate_identity") or "")
            if isinstance(value, Mapping)
            else ""
        ),
    )
    for candidate_value in ordered_candidates:
        candidate = _mapping(candidate_value, "judge candidate")
        identity = str(candidate.get("candidate_identity") or "")
        stratum = str(candidate.get("stratum") or "")
        public_question = candidate.get("public_question")
        model_output = candidate.get("model_output")
        persona = candidate.get("persona")
        item = candidate.get("item")
        if (
            not identity
            or identity in identities
            or stratum not in normalized
            or not isinstance(public_question, Mapping)
            or not isinstance(model_output, Mapping)
            or not isinstance(persona, Mapping)
            or not isinstance(item, Mapping)
        ):
            raise AnalysisContractError("judge candidate identity or payload is invalid")
        identities.add(identity)
        pre_exclusion_counts[stratum] += 1
        identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        allowed_output_fields = {"simulated", "answer", "rationale", "abstain"}
        if (
            set(model_output) - allowed_output_fields
            or model_output.get("simulated") is not True
            or not isinstance(model_output.get("rationale"), str)
            or not str(model_output.get("rationale")).strip()
            or model_output.get("answer") is not None
            and not isinstance(model_output.get("answer"), str)
            or model_output.get("abstain")
            is not (model_output.get("answer") is None)
        ):
            raise AnalysisContractError("judge candidate output surface is invalid")
        judge_question = judge_public_question_payload(
            {"public_question": dict(public_question)}
        )
        target_label_terms = tuple(
            sorted(
                {
                    normalized_term
                    for value in (
                        persona.get("target_node"),
                        persona.get("target_label"),
                        item.get("target_node"),
                        item.get("target_label"),
                    )
                    if value is not None
                    and (normalized_term := _normalize_identity_term(value))
                }
            )
        )
        target_label_hits = _provider_identity_hits(
            judge_question, target_label_terms
        )
        if target_label_hits:
            excluded_target_label_rows.append(
                {
                    "candidate_identity_sha256": identity_digest,
                    "stratum": stratum,
                    "matched_target_label_sha256": [
                        hashlib.sha256(term.encode("utf-8")).hexdigest()
                        for term in target_label_hits
                    ],
                }
            )
            continue
        try:
            judge_messages = render_judge_export(
                public_question=judge_question,
                model_output=dict(model_output),
            )
        except (AssertionError, TypeError, ValueError) as exc:
            raise AnalysisContractError(
                f"judge candidate contains target-label leakage: {exc}"
            ) from exc
        payload_target_label_hits = _provider_identity_hits(
            judge_messages, target_label_terms
        )
        if payload_target_label_hits:
            excluded_target_label_rows.append(
                {
                    "candidate_identity_sha256": identity_digest,
                    "stratum": stratum,
                    "matched_target_label_sha256": [
                        hashlib.sha256(term.encode("utf-8")).hexdigest()
                        for term in payload_target_label_hits
                    ],
                }
            )
            continue
        identity_hits = _provider_identity_hits(judge_messages, identity_terms)
        if identity_hits:
            excluded_identity_rows.append(
                {
                    "candidate_identity_sha256": identity_digest,
                    "stratum": stratum,
                    "matched_identity_term_sha256": [
                        hashlib.sha256(term.encode("utf-8")).hexdigest()
                        for term in identity_hits
                    ],
                }
            )
            continue
        selection_digest = hashlib.sha256(
            f"{BOOTSTRAP_SEED}|{identity}".encode("utf-8")
        ).hexdigest()
        case_row = {
            "candidate_identity_sha256": identity_digest,
            "selection_sha256": selection_digest,
            "stratum": stratum,
            "case_id": "case-"
            + hashlib.sha256(
                f"judge-case|{BOOTSTRAP_SEED}|{identity}".encode("utf-8")
            ).hexdigest()[:24],
            "judge_messages": judge_messages,
            "judge_input_sha256": _canonical_sha(judge_messages),
        }
        normalized[stratum].append(case_row)
        target_terms_by_case[case_row["case_id"]] = target_label_terms
    for values in normalized.values():
        values.sort(key=lambda row: (row["selection_sha256"], row["case_id"]))
    excluded_identity_rows.sort(
        key=lambda row: (row["candidate_identity_sha256"], row["stratum"])
    )
    excluded_target_label_rows.sort(
        key=lambda row: (row["candidate_identity_sha256"], row["stratum"])
    )
    disagreement = normalized["disagreement"]
    agreement = normalized["agreement"]
    selected_disagreement = disagreement[:80]
    selected_agreement = agreement[:40]
    cross_fill = 0
    if len(selected_disagreement) < 80:
        capacity = min(
            80 - len(selected_disagreement),
            len(agreement) - len(selected_agreement),
        )
        selected_agreement.extend(agreement[40 : 40 + capacity])
        cross_fill += capacity
    if len(selected_agreement) < 40:
        capacity = min(
            40 - len(selected_agreement),
            len(disagreement) - len(selected_disagreement),
        )
        selected_disagreement.extend(disagreement[80 : 80 + capacity])
        cross_fill += capacity
    selected = sorted(
        [*selected_disagreement, *selected_agreement],
        key=lambda row: (row["selection_sha256"], row["case_id"]),
    )
    if len(selected) > 120 or len({row["case_id"] for row in selected}) != len(selected):
        raise AnalysisContractError("judge selection is not a unique at-most-120 set")
    manifest: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.judge_case_manifest.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "analysis_population": "main",
        "exploratory": True,
        "seed_label": str(BOOTSTRAP_SEED),
        "quota": {"disagreement": 80, "agreement": 40, "maximum": 120},
        "pre_exclusion_candidate_stratum_counts": pre_exclusion_counts,
        "candidate_stratum_counts": {
            "disagreement": len(disagreement),
            "agreement": len(agreement),
        },
        "selected_stratum_counts": {
            "disagreement": len(selected_disagreement),
            "agreement": len(selected_agreement),
        },
        "cross_stratum_fill": cross_fill,
        "selected_count": len(selected),
        "opaque_case_ids": True,
        "judge_protocol": protocol,
        "judge_protocol_sha256": _canonical_sha(protocol),
        "judge_amendment": {
            "path": "experiments/llm_sim_v2/judge_amendment_20260716.md",
            "sha256": hashlib.sha256(amendment_bytes).hexdigest(),
            "size": len(amendment_bytes),
        },
        "question_field_whitelist": sorted(JUDGE_PUBLIC_SCHEMA_KEYS),
        "target_metadata_exported": False,
        "target_labels_exported": False,
        "target_label_scan": {
            "policy": "exclude_candidate_when_its_exact_target_label_occurs_in_sanitized_judge_payload",
            "excluded_candidate_count": len(excluded_target_label_rows),
            "excluded_candidates": excluded_target_label_rows,
            "final_serialized_hit_count": None,
        },
        "provider_identity_scan": {
            "policy": "exclude_candidate_without_redaction_before_selection",
            "identity_term_count": len(identity_terms),
            "identity_term_set_sha256": _canonical_sha(identity_terms),
            "excluded_candidate_count": len(excluded_identity_rows),
            "excluded_candidate_identity_sha256": [
                row["candidate_identity_sha256"] for row in excluded_identity_rows
            ],
            "excluded_candidates": excluded_identity_rows,
            "final_serialized_bytes_scanned": True,
            "final_serialized_hit_count": None,
        },
        "cases": selected,
    }
    shared_bytes = judge_input_bytes(manifest)
    final_identity_hits = _provider_identity_hits(shared_bytes, identity_terms)
    if final_identity_hits:
        raise AnalysisContractError(
            "provider identity remained in final serialized judge bytes"
        )
    manifest["provider_identity_scan"]["final_serialized_hit_count"] = 0
    final_target_label_hits = sum(
        bool(_provider_identity_hits(row["judge_messages"], target_terms_by_case[row["case_id"]]))
        for row in selected
    )
    if final_target_label_hits:
        raise AnalysisContractError("target label remained in final judge case bytes")
    manifest["target_label_scan"]["final_serialized_hit_count"] = 0
    manifest["provider_identity_exported"] = False
    manifest["shared_input_sha256"] = hashlib.sha256(shared_bytes).hexdigest()
    manifest["judge_inputs"] = {
        "claude": manifest["shared_input_sha256"],
        "gpt": manifest["shared_input_sha256"],
    }
    manifest["case_manifest_sha256"] = _canonical_sha(manifest)
    return manifest


def ingest_judge_results(
    case_manifest: Mapping[str, Any],
    judge_result_manifests: Mapping[str, Mapping[str, Any] | None],
    *,
    judge_artifact_roots: Mapping[str, str] | None = None,
    judge_run_evidence: Mapping[str, Any] | None = None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Validate two frozen judge passes and report descriptive agreement only."""

    from .judge_execution import JudgeExecutionError, validate_execution_receipt
    from .judge_protocol import (
        JUDGE_PUBLIC_SCHEMA_KEYS,
        judge_protocol,
        validate_judge_output,
    )

    cases = _mapping(case_manifest, "judge case manifest")
    payload = dict(cases)
    advertised_case_sha = payload.pop("case_manifest_sha256", None)
    protocol = judge_protocol()
    amendment_bytes = Path(__file__).with_name(
        "judge_amendment_20260716.md"
    ).read_bytes()
    amendment_binding = {
        "path": "experiments/llm_sim_v2/judge_amendment_20260716.md",
        "sha256": hashlib.sha256(amendment_bytes).hexdigest(),
        "size": len(amendment_bytes),
    }
    if (
        cases.get("schema_version") != "yher.llm_sim_v2.judge_case_manifest.v2"
        or advertised_case_sha != _canonical_sha(payload)
        or cases.get("shared_input_sha256")
        != hashlib.sha256(judge_input_bytes(cases)).hexdigest()
        or cases.get("judge_protocol") != protocol
        or cases.get("judge_protocol_sha256") != _canonical_sha(protocol)
        or cases.get("judge_amendment") != amendment_binding
        or cases.get("question_field_whitelist")
        != sorted(JUDGE_PUBLIC_SCHEMA_KEYS)
        or cases.get("target_metadata_exported") is not False
        or cases.get("target_labels_exported") is not False
    ):
        raise AnalysisContractError("judge case manifest hash or input bytes drifted")
    if set(judge_result_manifests) != {"claude", "gpt"}:
        raise AnalysisContractError("judge results must disclose both frozen judge slots")
    case_ids = [str(row["case_id"]) for row in cases.get("cases", ())]
    run_receipt: Mapping[str, Any] | None = None
    if judge_run_evidence is not None:
        run_receipt = _mapping(
            judge_run_evidence.get("receipt"), "judge run evidence receipt"
        )
        receipt_payload = dict(run_receipt)
        advertised_run_sha = receipt_payload.pop(
            "judge_run_evidence_receipt_sha256", None
        )
        family_slots = run_receipt.get("family_slots")
        case_binding = run_receipt.get("case_binding")
        if (
            run_receipt.get("schema_version")
            != "yher.llm_sim_v2.judge_run_evidence_receipt.v1"
            or run_receipt.get("simulated") is not True
            or run_receipt.get("run_id") != RUN_ID
            or run_receipt.get("status") != "finalized"
            or advertised_run_sha != _canonical_sha(receipt_payload)
            or not isinstance(family_slots, Mapping)
            or set(family_slots) != {"claude", "gpt"}
            or not isinstance(case_binding, Mapping)
            or case_binding.get("case_manifest_sha256") != advertised_case_sha
            or case_binding.get("shared_input_sha256")
            != cases.get("shared_input_sha256")
            or case_binding.get("case_count") != len(case_ids)
        ):
            raise AnalysisContractError(
                "judge run evidence receipt does not bind the analyzed case manifest"
            )
        for judge in ("claude", "gpt"):
            slot = _mapping(family_slots[judge], f"{judge} judge family slot")
            result = judge_result_manifests[judge]
            if (slot.get("status") == "complete") != (result is not None):
                raise AnalysisContractError(
                    f"{judge} judge result availability differs from finalized run evidence"
                )
            if result is not None:
                embedded = _mapping(
                    result.get("execution_receipt"),
                    f"{judge} embedded judge execution receipt",
                )
                if (
                    slot.get("receipt_sha256")
                    != embedded.get("execution_receipt_sha256")
                    or slot.get("accounting") != embedded.get("accounting")
                ):
                    raise AnalysisContractError(
                        f"{judge} judge result differs from finalized family slot"
                    )
    if not case_ids:
        if any(judge_result_manifests.values()):
            raise AnalysisContractError("zero-case judge run cannot contain judge results")
        if run_receipt is not None and any(
            _mapping(
                _mapping(run_receipt["family_slots"], "judge family slots")[judge],
                f"{judge} judge family slot",
            ).get("status")
            != "not_applicable_zero_cases"
            for judge in ("claude", "gpt")
        ):
            raise AnalysisContractError(
                "zero-case judge run must record both not-applicable dispositions"
            )
        return {
            "schema_version": "yher.llm_sim_v2.judge_analysis.v2",
            "simulated": True,
            "run_id": RUN_ID,
            "exploratory": True,
            "case_manifest_sha256": advertised_case_sha,
            "selected_count": 0,
            "cases": [],
            "result_manifest_sha256": {},
            "execution_receipt_sha256": {},
            "execution_ids": {},
            "judge_families": {},
            "judge_models": {},
            "judge_transports": {},
            "judge_accounting": {},
            "expected_judges": ["claude", "gpt"],
            "available_judges": [],
            "missing_judges": [],
            "status": "not_applicable_zero_cases",
            "category_counts": {},
            "pairwise_label_agreement": None,
            "pairwise_error_category_agreement": None,
            "label_disagreement_examples": [],
            "error_category_disagreement_examples": [],
        }
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    manifest_hashes: dict[str, str] = {}
    missing: list[str] = []
    category_counts: dict[str, dict[str, Any]] = {}
    execution_receipt_hashes: dict[str, str] = {}
    execution_ids: dict[str, str] = {}
    judge_families: dict[str, str] = {}
    judge_models: dict[str, str] = {}
    judge_transports: dict[str, str] = {}
    attempt_ids_by_judge: dict[str, set[str]] = {}
    raw_hashes_by_judge: dict[str, set[str]] = {}
    judge_accounting: dict[str, dict[str, Any]] = {}
    for judge in ("claude", "gpt"):
        raw = judge_result_manifests[judge]
        if raw is None:
            missing.append(judge)
            continue
        result_manifest = _mapping(raw, f"{judge} judge result manifest")
        result_payload = dict(result_manifest)
        advertised = result_payload.pop("judge_result_manifest_sha256", None)
        results = result_manifest.get("results")
        receipt_value = result_manifest.get("execution_receipt")
        if (
            result_manifest.get("schema_version")
            != "yher.llm_sim_v2.judge_result_manifest.v2"
            or result_manifest.get("simulated") is not True
            or result_manifest.get("run_id") != RUN_ID
            or result_manifest.get("judge") != judge
            or result_manifest.get("case_manifest_sha256") != advertised_case_sha
            or advertised != _canonical_sha(result_payload)
            or not isinstance(receipt_value, Mapping)
            or not isinstance(results, list)
            or [str(row.get("case_id") or "") for row in results if isinstance(row, Mapping)]
            != case_ids
        ):
            raise AnalysisContractError(
                f"{judge} judge result or case manifest hash drifted"
            )
        try:
            receipt = validate_execution_receipt(
                receipt_value,
                cases,
                judge,
                artifact_root=(
                    judge_artifact_roots.get(judge)
                    if judge_artifact_roots is not None
                    else None
                ),
                allow_fixture=allow_fixture,
            )
        except JudgeExecutionError as exc:
            raise AnalysisContractError(
                f"{judge} judge execution receipt is invalid: {exc}"
            ) from exc
        identity = _mapping(receipt.get("identity"), "judge execution identity")
        expected_transport = (
            "fixture"
            if allow_fixture
            else {"claude": "claude_cli", "gpt": "codex_cli"}[judge]
        )
        if identity.get("transport") != expected_transport:
            raise AnalysisContractError(
                f"{judge} judge execution transport is not independently admissible"
            )
        normalized_receipt = _mapping(
            receipt.get("normalized_results"), "judge normalized result binding"
        )
        if (
            normalized_receipt.get("row_count") != len(results)
            or normalized_receipt.get("ordered_output_sha256")
            != _canonical_sha([_canonical_sha(row) for row in results])
        ):
            raise AnalysisContractError(
                f"{judge} judge normalized result receipt differs from results"
            )
        outputs: dict[str, dict[str, Any]] = {}
        try:
            for row in results:
                row_mapping = _mapping(row, f"{judge} judge result row")
                case_id = str(row_mapping.get("case_id") or "")
                output = validate_judge_output(
                    _mapping(row_mapping.get("output"), "judge output")
                )
                outputs[case_id] = output
        except (TypeError, ValueError) as exc:
            raise AnalysisContractError(f"{judge} judge output is invalid: {exc}") from exc
        if len(outputs) != len(case_ids):
            raise AnalysisContractError(f"{judge} judge results repeat a case ID")
        normalized[judge] = outputs
        manifest_hashes[judge] = str(advertised)
        execution_receipt_hashes[judge] = str(
            receipt["execution_receipt_sha256"]
        )
        execution_ids[judge] = str(identity["execution_id"])
        judge_families[judge] = str(identity["judge_family"])
        judge_models[judge] = str(identity["requested_model"])
        judge_transports[judge] = str(identity["transport"])
        attempt_ids_by_judge[judge] = {
            str(value) for value in receipt["ordered_attempt_ids"]
        }
        raw_hashes_by_judge[judge] = {
            str(_mapping(row, "judge raw artifact")["sha256"])
            for row in receipt["raw_artifacts"]
        }
        judge_accounting[judge] = dict(
            _mapping(receipt.get("accounting"), "judge execution accounting")
        )
        label_counts = Counter(output["label"] for output in outputs.values())
        error_counts = Counter(
            str(output["error_category"])
            for output in outputs.values()
            if output.get("error_category")
        )
        category_counts[judge] = {
            "labels": dict(sorted(label_counts.items())),
            "error_categories": dict(sorted(error_counts.items())),
        }
    pairwise_label: dict[str, Any] | None = None
    pairwise_error_category: dict[str, Any] | None = None
    label_disagreements: list[dict[str, Any]] = []
    error_category_disagreements: list[dict[str, Any]] = []
    if not missing:
        if (
            judge_families != {"claude": "claude", "gpt": "gpt"}
            or len(set(execution_ids.values())) != 2
            or len(set(judge_models.values())) != 2
            or attempt_ids_by_judge["claude"] & attempt_ids_by_judge["gpt"]
            or raw_hashes_by_judge["claude"] & raw_hashes_by_judge["gpt"]
        ):
            raise AnalysisContractError(
                "judge executions are not independent across families, models, attempts, and raw artifacts"
            )
        left = [normalized["claude"][case_id]["label"] for case_id in case_ids]
        right = [normalized["gpt"][case_id]["label"] for case_id in case_ids]
        if case_ids:
            numerator = sum(
                left_value == right_value
                for left_value, right_value in zip(left, right, strict=True)
            )
            pairwise_label = {
                "metric": "judge_label_agreement",
                "judges": ["claude", "gpt"],
                "exact_agreement_numerator": numerator,
                "denominator": len(case_ids),
                "exact_agreement": numerator / len(case_ids),
                "cohen_kappa": _cohen_kappa(left, right),
            }
            paired_error_cases = [
                case_id
                for case_id in case_ids
                if normalized["claude"][case_id].get("error_category")
                and normalized["gpt"][case_id].get("error_category")
            ]
            left_error = [
                str(normalized["claude"][case_id]["error_category"])
                for case_id in paired_error_cases
            ]
            right_error = [
                str(normalized["gpt"][case_id]["error_category"])
                for case_id in paired_error_cases
            ]
            error_numerator = sum(
                left_value == right_value
                for left_value, right_value in zip(
                    left_error, right_error, strict=True
                )
            )
            error_denominator = len(paired_error_cases)
            pairwise_error_category = {
                "metric": "judge_error_category_agreement",
                "judges": ["claude", "gpt"],
                "denominator_policy": "paired_nonmissing_error_categories",
                "total_case_count": len(case_ids),
                "denominator": error_denominator,
                "missing_any_count": len(case_ids) - error_denominator,
                "exact_agreement_numerator": error_numerator,
                "exact_agreement": (
                    error_numerator / error_denominator
                    if error_denominator
                    else None
                ),
                "cohen_kappa": (
                    _cohen_kappa(left_error, right_error)
                    if error_denominator
                    else None
                ),
            }
        for case_id in case_ids:
            claude = normalized["claude"][case_id]
            gpt = normalized["gpt"][case_id]
            if claude["label"] != gpt["label"]:
                label_disagreements.append(
                    {
                        "case_id": case_id,
                        "claude": {
                            field: claude.get(field)
                            for field in ("label", "error_category", "rationale")
                        },
                        "gpt": {
                            field: gpt.get(field)
                            for field in ("label", "error_category", "rationale")
                        },
                    }
                )
            if (
                claude.get("error_category")
                and gpt.get("error_category")
                and claude["error_category"] != gpt["error_category"]
            ):
                error_category_disagreements.append(
                    {
                        "case_id": case_id,
                        "claude": {
                            field: claude.get(field)
                            for field in ("label", "error_category", "rationale")
                        },
                        "gpt": {
                            field: gpt.get(field)
                            for field in ("label", "error_category", "rationale")
                        },
                    }
                )
    return {
        "schema_version": "yher.llm_sim_v2.judge_analysis.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "exploratory": True,
        "case_manifest_sha256": advertised_case_sha,
        "selected_count": len(case_ids),
        "cases": case_ids,
        "result_manifest_sha256": manifest_hashes,
        "execution_receipt_sha256": execution_receipt_hashes,
        "execution_ids": execution_ids,
        "judge_families": judge_families,
        "judge_models": judge_models,
        "judge_transports": judge_transports,
        "judge_accounting": judge_accounting,
        "expected_judges": ["claude", "gpt"],
        "available_judges": [
            judge for judge in ("claude", "gpt") if judge in normalized
        ],
        "missing_judges": missing,
        "status": (
            "complete"
            if not missing
            else "missing_all_judges"
            if len(missing) == 2
            else "partial_missing_judge"
        ),
        "category_counts": category_counts,
        "pairwise_label_agreement": pairwise_label,
        "pairwise_error_category_agreement": pairwise_error_category,
        "label_disagreement_examples": label_disagreements[:10],
        "error_category_disagreement_examples": error_category_disagreements[:10],
    }


_RECORD_STATUSES = {
    "complete",
    "excluded_schema",
    "excluded_model_drift",
    "technical_failure",
}


def _provenance_binding(phase: Mapping[str, Any]) -> dict[str, Any]:
    from .runner import phase_provenance_binding

    try:
        return phase_provenance_binding(phase)
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisContractError(
            f"phase provenance cannot produce the runner binding: {exc}"
        ) from exc


def _validate_mapping_manifest(
    mapping_manifest: Mapping[str, Any],
    *,
    phase: Mapping[str, Any],
    expected_tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mapping_value = _mapping(mapping_manifest, "target-option mapping")
    target_binding = _mapping(phase.get("target"), "phase target binding")
    consensus = _mapping(mapping_value.get("consensus"), "mapping consensus")
    if (
        mapping_value.get("mapping_sha256") != target_binding.get("mapping_sha256")
        or mapping_value.get("target_set_hash") != target_binding.get("target_set_hash")
        or mapping_value.get("confirmatory_target_misconception_hit_rate") is not False
        or consensus.get("mapped_rows") != 6
        or consensus.get("excluded_ambiguous_rows") != 94
        or not math.isclose(float(mapping_value.get("mapped_fraction", -1.0)), 0.06)
    ):
        raise AnalysisContractError("sparse mapping identity or degradation status drifted")
    mapped_task_rows = {
        (str(row["anchor_id"]), str(row["item_id"]))
        for row in expected_tasks
        if row.get("condition") == "controlled" and row.get("target_option") is not None
    }
    if len(mapped_task_rows) != 6:
        raise AnalysisContractError("reconstructed roster does not contain six mapped rows")
    return {
        "confirmatory": False,
        "status": "sparse_descriptive_only",
        "mapped_mapping_rows": 6,
        "excluded_ambiguous_mapping_rows": 94,
        "total_mapping_rows": 100,
        "mapped_fraction": 0.06,
        "mapping_sha256": mapping_value["mapping_sha256"],
        "target_set_hash": mapping_value["target_set_hash"],
    }


def _validate_record(
    record: Mapping[str, Any],
    *,
    provider: str,
    task: Mapping[str, Any],
    phase: Mapping[str, Any],
    expected_model: str,
    attempt_policy: Mapping[str, Any],
    formal_mode: bool,
) -> None:
    schema_version = record.get("schema_version")
    if formal_mode and schema_version != "yher.llm_sim_v2.response_record.v2":
        raise AnalysisContractError(
            "formal record schema must be yher.llm_sim_v2.response_record.v2"
        )
    if (
        schema_version
        not in (
            {"yher.llm_sim_v2.response_record.v2"}
            if formal_mode
            else {
                "yher.llm_sim_v2.response_record.v1",
                "yher.llm_sim_v2.response_record.v2",
            }
        )
        or record.get("simulated") is not True
        or record.get("run_id") != RUN_ID
        or record.get("phase") != "main"
        or record.get("analysis_population") != "main"
    ):
        raise AnalysisContractError("record must be a simulated formal main record")
    if (
        record.get("collection_mode") != "formal"
        or record.get("development_only") is not False
        or record.get("partial") is not False
        or record.get("formal_analysis_eligible") is not True
    ):
        raise AnalysisContractError("record is partial or ineligible for formal main analysis")
    if record.get("provider") != provider:
        raise AnalysisContractError("record provider differs from its provider namespace")
    if record.get("requested_model") != expected_model:
        raise AnalysisContractError("record requested model differs from active contract")
    for field in (
        "task_id",
        "persona_id",
        "pair_id",
        "row_id",
        "anchor_id",
        "response_arm",
        "condition",
        "item_id",
        "target_node",
        "family_id",
        "attempt_id",
        "logical_key",
        "message_sha256",
        "wire_message_sha256",
        "is_stability_repeat",
    ):
        if record.get(field) != task.get(field):
            raise AnalysisContractError(f"record task identity drifted: {field}")
    if record.get("prompt_revision") != _mapping(
        phase.get("prompt"), "phase prompt binding"
    ).get("revision") or record.get("prompt_contract_sha256") != _mapping(
        phase.get("prompt"), "phase prompt binding"
    ).get("prompt_contract_sha256"):
        raise AnalysisContractError("record prompt binding drifted")
    if record.get("provenance") != _provenance_binding(phase):
        raise AnalysisContractError("record provenance binding drifted")
    if formal_mode:
        from .evidence import validate_v2_response_record

        item_contract = _mapping(task.get("item_contract"), "task item contract")
        evidence_task = dict(task)
        evidence_task["option_keys"] = tuple(
            str(key).strip().upper()
            for key in _mapping(
                item_contract.get("options"), "task option mapping"
            )
        )
        try:
            validate_v2_response_record(
                record,
                provider=provider,
                requested_model=expected_model,
                phase="main",
                task=evidence_task,
                expected_provenance=_provenance_binding(phase),
            )
        except (TypeError, ValueError) as exc:
            raise AnalysisContractError(
                f"formal response record fails shared strict replay: {exc}"
            ) from exc
    status = record.get("status")
    if status not in _RECORD_STATUSES:
        raise AnalysisContractError("record status is outside the runner lifecycle schema")
    parsed = record.get("parsed_output")
    if status == "complete":
        from .runner import InvalidProviderOutput, parse_provider_output

        if (
            not isinstance(parsed, Mapping)
            or parsed.get("simulated") is not True
            or parsed.get("answer") is not None
            and not isinstance(parsed.get("answer"), str)
            or not isinstance(parsed.get("rationale"), str)
            or not str(parsed.get("rationale")).strip()
        ):
            raise AnalysisContractError("complete record parsed output is invalid")
        if task.get("condition") == "blind" and parsed.get("abstain") is not (
            parsed.get("answer") is None
        ):
            raise AnalysisContractError("blind abstention flag is inconsistent")
        item_contract = _mapping(task.get("item_contract"), "task item contract")
        options = _mapping(item_contract.get("options"), "task option mapping")
        option_keys = {str(key).strip().upper() for key in options}
        if not formal_mode:
            try:
                normalized = parse_provider_output(
                    json.dumps(
                        dict(parsed),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    condition=str(task["condition"]),
                    option_keys=option_keys,
                )
            except (InvalidProviderOutput, TypeError, ValueError) as exc:
                raise AnalysisContractError(
                    f"complete record parsed output violates runner schema: {exc}"
                ) from exc
            if dict(parsed) != normalized:
                raise AnalysisContractError(
                    "complete record parsed output is not runner-normalized"
                )
    elif parsed is not None:
        raise AnalysisContractError("non-complete record cannot have parsed output")
    if status == "complete" and record.get("model_id") != expected_model:
        raise AnalysisContractError("complete record returned model differs from active contract")
    if status == "excluded_model_drift" and record.get("model_id") == expected_model:
        raise AnalysisContractError("model-drift exclusion lacks returned-model drift")

    from .collect import _reconciled_record_cost
    from .runner import compute_outcomes

    try:
        reconciled_cost = _reconciled_record_cost(
            record,
            path=Path(f"<memory>/{provider}/{record.get('task_id')}.json"),
            unknown_attempt_reserve_yuan=10.0,
        )
    except ValueError as exc:
        raise AnalysisContractError(f"record attempt cost does not reconcile: {exc}") from exc
    expected_needs_user_reasons = (
        ["unknown_provider_billing_reserved"]
        if reconciled_cost["unknown_attempt_count"] > 0
        else []
    )
    if record.get("needs_user_reasons") != expected_needs_user_reasons:
        raise AnalysisContractError(
            "record needs_user billing-reserve reasons do not reconcile"
        )
    attempts = record.get("attempts")
    max_attempts = int(attempt_policy["max_attempts"])
    allowed_request_max_tokens = set(
        int(value) for value in attempt_policy["allowed_request_max_tokens"]
    )
    if (
        not isinstance(attempts, list)
        or len(attempts) > max_attempts
        or [row.get("attempt") for row in attempts if isinstance(row, Mapping)]
        != list(range(1, len(attempts) + 1))
        or record.get("retry_count") != len(attempts) - 1
    ):
        raise AnalysisContractError("record attempt/retry ledger does not reconcile")
    attempt_models: list[str] = []
    from .evidence import validate_response_content_binding

    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise AnalysisContractError("record attempt row is invalid")
        attempt_status = attempt.get("status")
        if attempt_status not in {"response", "failed"}:
            raise AnalysisContractError("record attempt status is invalid")
        request_max_tokens = attempt.get("request_max_tokens")
        if request_max_tokens is not None and (
            not isinstance(request_max_tokens, int)
            or isinstance(request_max_tokens, bool)
            or request_max_tokens not in allowed_request_max_tokens
        ):
            raise AnalysisContractError("record attempt token policy drifted")
        response_received = attempt.get("provider_response_received")
        if formal_mode:
            if not isinstance(response_received, bool):
                raise AnalysisContractError(
                    "formal v2 attempt lacks provider-response receipt state"
                )
            if response_received:
                try:
                    validate_response_content_binding(attempt)
                except (TypeError, ValueError) as exc:
                    raise AnalysisContractError(
                        f"provider response content binding is invalid: {exc}"
                    ) from exc
            elif any(
                field in attempt
                for field in (
                    "response_content",
                    "response_content_utf8_bytes",
                    "response_content_sha256",
                )
            ):
                raise AnalysisContractError(
                    "non-response attempt carries provider response content"
                )
            if response_received is True and attempt_status == "failed":
                if attempt.get("error_category") != "invalid_schema":
                    raise AnalysisContractError(
                        "failed provider-response attempt is not an invalid-schema retry"
                    )
                from .runner import InvalidProviderOutput, parse_provider_output

                item_contract = _mapping(
                    task.get("item_contract"), "task item contract"
                )
                option_keys = {
                    str(key).strip().upper()
                    for key in _mapping(
                        item_contract.get("options"), "task option mapping"
                    )
                }
                try:
                    parse_provider_output(
                        str(attempt["response_content"]),
                        condition=str(task["condition"]),
                        option_keys=option_keys,
                    )
                except InvalidProviderOutput:
                    pass
                else:
                    raise AnalysisContractError(
                        "invalid-schema exclusion differs from raw strict replay"
                    )
        if attempt_status == "response" and formal_mode and response_received is not True:
            raise AnalysisContractError(
                "successful response attempt lacks bound provider content"
            )
        if attempt_status == "response" or (formal_mode and response_received is True):
            usage = attempt.get("usage")
            if (
                request_max_tokens is None
                or not isinstance(attempt.get("finish_reason"), str)
                or not isinstance(attempt.get("model_returned"), str)
                or not isinstance(attempt.get("latency_ms"), (int, float))
                or isinstance(attempt.get("latency_ms"), bool)
                or not math.isfinite(float(attempt["latency_ms"]))
                or float(attempt["latency_ms"]) < 0.0
                or not isinstance(usage, Mapping)
                or any(
                    not isinstance(usage.get(field), int)
                    or isinstance(usage.get(field), bool)
                    or int(usage[field]) < 0
                    for field in ("input_tokens", "output_tokens")
                )
            ):
                raise AnalysisContractError(
                    "response attempt lacks required runner fields"
                )
        if "model_returned" in attempt:
            model_value = attempt.get("model_returned")
            if not isinstance(model_value, str):
                raise AnalysisContractError("record attempt returned model is invalid")
            attempt_models.append(model_value)
    if attempt_models and record.get("model_id") != attempt_models[-1]:
        raise AnalysisContractError(
            "record model does not reconcile to the final attempt returned model"
        )
    if any(model != expected_model for model in attempt_models) and status != (
        "excluded_model_drift"
    ):
        raise AnalysisContractError(
            "attempt-level model drift was not excluded from analysis"
        )
    terminal_attempt = _mapping(attempts[-1], "terminal record attempt")
    terminal_status = terminal_attempt.get("status")
    record_error = record.get("error")
    if terminal_status not in {"response", "failed"}:
        raise AnalysisContractError("terminal attempt status is outside runner schema")
    if status == "complete":
        if (
            terminal_status != "response"
            or terminal_attempt.get("error_category") is not None
            or record_error is not None
        ):
            raise AnalysisContractError(
                "complete record requires a successful terminal response attempt"
            )
        if formal_mode:
            from .runner import InvalidProviderOutput, parse_provider_output

            item_contract = _mapping(task.get("item_contract"), "task item contract")
            option_keys = {
                str(key).strip().upper()
                for key in _mapping(
                    item_contract.get("options"), "task option mapping"
                )
            }
            try:
                replayed = parse_provider_output(
                    str(terminal_attempt["response_content"]),
                    condition=str(task["condition"]),
                    option_keys=option_keys,
                )
            except (InvalidProviderOutput, KeyError, TypeError, ValueError) as exc:
                raise AnalysisContractError(
                    f"formal raw provider response cannot be strictly replayed: {exc}"
                ) from exc
            if replayed != dict(_mapping(parsed, "complete parsed output")):
                raise AnalysisContractError(
                    "stored parsed output differs from raw strict replay"
                )
    elif status == "excluded_schema":
        if (
            terminal_status != "failed"
            or terminal_attempt.get("error_category") != "invalid_schema"
            or not isinstance(record_error, str)
            or not record_error.strip()
            or record.get("model_id") != expected_model
        ):
            raise AnalysisContractError(
                "excluded-schema record does not match terminal attempt semantics"
            )
    elif status == "excluded_model_drift":
        if (
            record_error != "returned_model_drift"
            or not isinstance(record.get("model_id"), str)
            or not str(record.get("model_id")).strip()
            or record.get("model_id") == expected_model
            or terminal_attempt.get("model_returned") != record.get("model_id")
        ):
            raise AnalysisContractError(
                "excluded model-drift record does not match terminal attempt semantics"
            )
    elif status == "technical_failure":
        if (
            terminal_status != "failed"
            or not isinstance(terminal_attempt.get("error_category"), str)
            or not str(terminal_attempt.get("error_category")).strip()
            or not isinstance(record_error, str)
            or not record_error.strip()
            or record_error == "returned_model_drift"
            or record.get("model_id") not in {None, expected_model}
        ):
            raise AnalysisContractError(
                "technical-failure record does not match terminal attempt semantics"
            )
    expected_outcomes = (
        compute_outcomes(
            condition=str(task["condition"]),
            response_arm=str(task["response_arm"]),
            answer=parsed.get("answer") if isinstance(parsed, Mapping) else None,
            abstain=bool(parsed.get("abstain")) if isinstance(parsed, Mapping) else False,
            correct_option=str(task["correct_option"]),
            target_option=(
                str(task["target_option"])
                if task.get("target_option") is not None
                else None
            ),
        )
        if status == "complete"
        else {
            "is_correct": None,
            "target_option_hit": None,
            "manipulation_compliance": None,
        }
    )
    if record.get("outcomes") != expected_outcomes:
        raise AnalysisContractError("record outcomes differ from runtime-task recomputation")


def _recomputed_condition_lifecycle(
    expected_tasks: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for condition in ("controlled", "blind"):
        condition_tasks = [
            task for task in expected_tasks if task["condition"] == condition
        ]
        primary_tasks = [
            task for task in condition_tasks if task["is_stability_repeat"] is False
        ]
        repeat_tasks = [
            task for task in condition_tasks if task["is_stability_repeat"] is True
        ]
        present = {
            str(task["task_id"]): records[str(task["task_id"])]
            for task in condition_tasks
            if str(task["task_id"]) in records
        }
        primary_present = {
            str(task["task_id"]): records[str(task["task_id"])]
            for task in primary_tasks
            if str(task["task_id"]) in records
        }
        invalid = sum(
            record.get("status") == "excluded_schema"
            for record in primary_present.values()
        )
        invalid_fraction = invalid / len(primary_tasks) if primary_tasks else None
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for task in primary_tasks:
            grouped.setdefault(str(task["persona_id"]), []).append(task)
        complete_clusters = sum(
            bool(tasks)
            and all(
                str(task["task_id"]) in records
                and records[str(task["task_id"])].get("status") == "complete"
                for task in tasks
            )
            for tasks in grouped.values()
        )
        output[condition] = {
            "expected_count": len(condition_tasks),
            "primary_expected_count": len(primary_tasks),
            "stability_repeat_expected_count": len(repeat_tasks),
            "present_count": len(present),
            "missing_count": len(condition_tasks) - len(present),
            "invalid_schema_count": invalid,
            "invalid_schema_fraction": invalid_fraction,
            "excluded_invalid_schema": bool(
                condition == "blind"
                and invalid_fraction is not None
                and invalid_fraction > 0.5
            ),
            "complete_cluster_count": complete_clusters,
            "minimum_complete_clusters": 45,
            "minimum_complete_clusters_met": complete_clusters >= 45,
        }
    return output


def _validate_breaker_state(
    *,
    provider: str,
    breaker: Mapping[str, Any],
    breaker_ids: set[str],
    expected_ids: Sequence[str],
    present_ids: Sequence[str],
    missing_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    attempt_policy: Mapping[str, Any],
    formal_mode: bool,
) -> None:
    threshold = int(attempt_policy["failure_threshold"])
    concurrency = int(attempt_policy["concurrency"])
    cooldown = float(attempt_policy["cooldown_seconds"])
    consecutive_failures = breaker.get("consecutive_failures")
    if (
        breaker.get("failure_threshold") != threshold
        or not math.isclose(
            float(breaker.get("cooldown_seconds", -1.0)),
            cooldown,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not isinstance(consecutive_failures, int)
        or isinstance(consecutive_failures, bool)
        or consecutive_failures < 0
    ):
        raise AnalysisContractError(
            f"{provider} breaker policy differs from the active runner contract"
        )
    failure_run = 0
    threshold_positions: list[int] = []
    for index, task_id in enumerate(present_ids):
        if records[task_id].get("status") == "complete":
            failure_run = 0
        else:
            failure_run += 1
            if failure_run >= threshold:
                threshold_positions.append(index)
    if breaker_ids:
        try:
            opened_epoch = float(breaker.get("opened_at_epoch"))
            resume_epoch = float(breaker.get("resume_not_before_epoch"))
        except (TypeError, ValueError) as exc:
            raise AnalysisContractError(
                f"{provider} breaker timing evidence is invalid"
            ) from exc
        prefix_ids = list(expected_ids[: len(present_ids)])
        missing_suffix = list(expected_ids[len(present_ids) :])
        if (
            breaker.get("status") != "open"
            or consecutive_failures < threshold
            or list(present_ids) != prefix_ids
            or sorted(breaker_ids, key=expected_ids.index) != missing_suffix
            or set(breaker_ids) != set(missing_ids)
            or not threshold_positions
            or not any(
                position >= max(0, len(present_ids) - concurrency)
                for position in threshold_positions
            )
            or not math.isfinite(opened_epoch)
            or not math.isfinite(resume_epoch)
            or not math.isclose(
                resume_epoch,
                opened_epoch + cooldown,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not isinstance(breaker.get("opened_at_utc"), str)
            or not isinstance(breaker.get("resume_not_before_utc"), str)
        ):
            raise AnalysisContractError(
                f"{provider} breaker exclusion lacks a threshold-backed transition"
            )
        return

    timing_fields = (
        "opened_at_epoch",
        "opened_at_utc",
        "resume_not_before_epoch",
        "resume_not_before_utc",
    )
    has_timing = any(breaker.get(field) is not None for field in timing_fields)
    final_batch_start = max(0, len(present_ids) - concurrency)
    threshold_reached_in_final_batch = bool(threshold_positions) and min(
        threshold_positions
    ) >= final_batch_start
    valid_closed_threshold_terminal = False
    if consecutive_failures >= threshold:
        if has_timing:
            try:
                opened_epoch = float(breaker.get("opened_at_epoch"))
                resume_epoch = float(breaker.get("resume_not_before_epoch"))
            except (TypeError, ValueError):
                opened_epoch = math.nan
                resume_epoch = math.nan
            timing_valid = (
                math.isfinite(opened_epoch)
                and math.isfinite(resume_epoch)
                and math.isclose(
                    resume_epoch,
                    opened_epoch + cooldown,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and isinstance(breaker.get("opened_at_utc"), str)
                and isinstance(breaker.get("resume_not_before_utc"), str)
            )
            # The runner records the opening instant even when the last bounded
            # batch leaves no pending task to classify as skipped.
            valid_closed_threshold_terminal = (
                consecutive_failures == threshold and timing_valid
            )
        else:
            # A zero-call resume reconstructs the terminal failure run from
            # immutable records and intentionally clears old timing.
            valid_closed_threshold_terminal = consecutive_failures == failure_run
        valid_closed_threshold_terminal = (
            valid_closed_threshold_terminal
            and not missing_ids
            and threshold_reached_in_final_batch
        )
    if (
        breaker.get("status") != "closed"
        or (
            consecutive_failures < threshold
            and (
                has_timing
                or (formal_mode and consecutive_failures != failure_run)
            )
        )
        or (
            consecutive_failures >= threshold
            and not valid_closed_threshold_terminal
        )
    ):
        raise AnalysisContractError(
            f"{provider} closed breaker state does not reconcile"
        )


def _validate_provider_inputs(
    *,
    expected_tasks: Sequence[Mapping[str, Any]],
    phase: Mapping[str, Any],
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
    provider_manifests: Mapping[str, Mapping[str, Any]],
    active_contract_proof: Mapping[str, Any],
    formal_mode: bool,
) -> list[dict[str, Any]]:
    if set(records_by_provider) != set(FROZEN_PROVIDERS) or set(
        provider_manifests
    ) != set(FROZEN_PROVIDERS):
        raise AnalysisContractError("all frozen providers require records and lifecycle manifests")
    task_index = {str(row["task_id"]): row for row in expected_tasks}
    expected_ids = list(task_index)
    expected_set = set(expected_ids)
    binding = _provenance_binding(phase)
    provider_models = _mapping(
        active_contract_proof.get("provider_models"), "active provider models"
    )
    provider_attempt_policies = _mapping(
        active_contract_proof.get("provider_attempt_policies"),
        "active provider attempt policies",
    )
    lifecycle_rows: list[dict[str, Any]] = []
    for provider in FROZEN_PROVIDERS:
        records = records_by_provider[provider]
        manifest = _mapping(provider_manifests[provider], f"{provider} provider manifest")
        attempt_policy = _mapping(
            provider_attempt_policies[provider],
            f"{provider} active attempt policy",
        )
        extra = set(records) - expected_set
        if extra:
            raise AnalysisContractError(f"{provider} records contain extra tasks outside roster")
        for task_id, record in records.items():
            if not isinstance(record, Mapping) or record.get("task_id") != task_id:
                raise AnalysisContractError(f"{provider} record path/task identity mismatch")
            _validate_record(
                record,
                provider=provider,
                task=task_index[task_id],
                phase=phase,
                expected_model=str(provider_models[provider]),
                attempt_policy=attempt_policy,
                formal_mode=formal_mode,
            )
        present_ids = [task_id for task_id in expected_ids if task_id in records]
        missing_ids = [task_id for task_id in expected_ids if task_id not in records]
        lifecycle = _mapping(manifest.get("lifecycle"), f"{provider} lifecycle")
        if (
            manifest.get("schema_version")
            != "yher.llm_sim_v2.provider_manifest.v1"
            or manifest.get("simulated") is not True
            or manifest.get("run_id") != RUN_ID
            or manifest.get("phase") != "main"
            or manifest.get("analysis_population") != "main"
            or manifest.get("provider") != provider
            or manifest.get("collection_mode") != "formal"
            or manifest.get("development_only") is not False
            or manifest.get("partial") is not False
            or manifest.get("formal_analysis_eligible") is not True
            or manifest.get("prompt_revision")
            != _mapping(phase.get("prompt"), "phase prompt binding")["revision"]
            or manifest.get("provenance") != binding
        ):
            raise AnalysisContractError(f"{provider} provider manifest is not formal main")
        if (
            lifecycle.get("expected_task_ids") != expected_ids
            or lifecycle.get("present_task_ids") != present_ids
            or lifecycle.get("missing_task_ids") != missing_ids
            or lifecycle.get("expected_count") != len(expected_ids)
            or lifecycle.get("present_count") != len(present_ids)
            or lifecycle.get("missing_count") != len(missing_ids)
            or manifest.get("record_count") != len(present_ids)
        ):
            raise AnalysisContractError(f"{provider} lifecycle does not reconcile to records")
        if formal_mode:
            record_set = _mapping(
                manifest.get("record_set"), f"{provider} provider record set"
            )
            record_rows = record_set.get("records")
            if (
                record_set.get("schema_version")
                != "yher.llm_sim_v2.provider_record_set.v1"
                or record_set.get("run_id") != RUN_ID
                or record_set.get("phase") != "main"
                or record_set.get("provider") != provider
                or record_set.get("expected_task_count") != len(expected_ids)
                or record_set.get("expected_task_ids_sha256")
                != _canonical_sha(expected_ids)
                or record_set.get("record_count") != len(present_ids)
                or record_set.get("missing_task_ids") != missing_ids
                or record_set.get("unexpected_task_ids") != []
                or not isinstance(record_rows, list)
                or [str(row.get("task_id")) for row in record_rows if isinstance(row, Mapping)]
                != sorted(present_ids)
            ):
                raise AnalysisContractError(
                    f"{provider} provider record set does not reconcile to formal records"
                )
        allowed_provider_lifecycles = {
            "complete",
            "complete_with_exclusions",
            "partial_missing",
            "fuse_open",
            "excluded_repeated_failure",
            "interrupted",
            "unavailable",
        }
        if manifest.get("provider_lifecycle") not in allowed_provider_lifecycles:
            raise AnalysisContractError(f"{provider} provider lifecycle value is invalid")
        classified_sets: list[set[str]] = []
        classified_by_name: dict[str, set[str]] = {}
        for prefix in ("interrupted", "fuse_skipped", "breaker_skipped"):
            ids = lifecycle.get(f"{prefix}_task_ids")
            count = lifecycle.get(f"{prefix}_count")
            if (
                not isinstance(ids, list)
                or len(ids) != len(set(ids))
                or count != len(ids)
                or not set(ids).issubset(missing_ids)
            ):
                raise AnalysisContractError(
                    f"{provider} lifecycle {prefix} classification does not reconcile"
                )
            classified_by_name[prefix] = set(ids)
            classified_sets.append(classified_by_name[prefix])
        unclassified = lifecycle.get("unclassified_missing_task_ids")
        if (
            not isinstance(unclassified, list)
            or len(unclassified) != len(set(unclassified))
            or not set(unclassified).issubset(missing_ids)
        ):
            raise AnalysisContractError(
                f"{provider} lifecycle unclassified missing set does not reconcile"
            )
        classified_sets.append(set(unclassified))
        if any(
            left & right
            for index, left in enumerate(classified_sets)
            for right in classified_sets[index + 1 :]
        ) or set().union(*classified_sets) != set(missing_ids):
            raise AnalysisContractError(
                f"{provider} lifecycle missing classifications are not an exact partition"
            )
        status_counts = Counter(str(record.get("status")) for record in records.values())
        if manifest.get("status_counts") != dict(status_counts) or manifest.get(
            "complete_records"
        ) != status_counts.get("complete", 0):
            raise AnalysisContractError(f"{provider} manifest status counts do not reconcile")
        condition_lifecycle = _mapping(
            manifest.get("condition_lifecycle"),
            f"{provider} condition lifecycle",
        )
        recomputed_conditions = _recomputed_condition_lifecycle(
            expected_tasks, records
        )
        if condition_lifecycle != recomputed_conditions:
            raise AnalysisContractError(
                f"{provider} condition lifecycle does not reconcile to records"
            )
        interruption = _mapping(
            manifest.get("interruption"), f"{provider} interruption"
        )
        unavailable = _mapping(
            manifest.get("unavailable"), f"{provider} unavailable state"
        )
        interrupted_flag = interruption.get("interrupted")
        unavailable_flag = unavailable.get("unavailable")
        if (
            not isinstance(interrupted_flag, bool)
            or not isinstance(unavailable_flag, bool)
            or (interrupted_flag is False and interruption.get("type") is not None)
            or (
                interrupted_flag is True
                and not isinstance(interruption.get("type"), str)
            )
            or (unavailable_flag is False and unavailable.get("error_category") is not None)
            or (
                unavailable_flag is True
                and (
                    not isinstance(unavailable.get("error_category"), str)
                    or not str(unavailable.get("error_category")).strip()
                )
            )
        ):
            raise AnalysisContractError(
                f"{provider} lifecycle state flags do not reconcile"
            )
        interrupted_ids = classified_by_name["interrupted"]
        fuse_ids = classified_by_name["fuse_skipped"]
        breaker_ids = classified_by_name["breaker_skipped"]
        breaker = _mapping(manifest.get("breaker"), f"{provider} breaker state")
        _validate_breaker_state(
            provider=provider,
            breaker=breaker,
            breaker_ids=breaker_ids,
            expected_ids=expected_ids,
            present_ids=present_ids,
            missing_ids=missing_ids,
            records=records,
            attempt_policy=attempt_policy,
            formal_mode=formal_mode,
        )
        budget = _mapping(manifest.get("budget"), f"{provider} provider budget")
        if fuse_ids and budget.get("hard_fuse_triggered") is not True:
            raise AnalysisContractError(
                f"{provider} fuse-skipped tasks lack hard-fuse evidence"
            )
        if unavailable_flag:
            if records or not missing_ids or interrupted_ids or fuse_ids or breaker_ids:
                raise AnalysisContractError(
                    f"{provider} unavailable lifecycle contradicts attempted records"
                )
            recomputed_provider_lifecycle = "unavailable"
        elif interrupted_flag:
            if not interrupted_ids:
                raise AnalysisContractError(
                    f"{provider} interrupted lifecycle lacks interrupted missing tasks"
                )
            recomputed_provider_lifecycle = "interrupted"
        elif interrupted_ids:
            raise AnalysisContractError(
                f"{provider} interrupted task partition lacks interruption evidence"
            )
        elif fuse_ids:
            recomputed_provider_lifecycle = "fuse_open"
        elif breaker_ids:
            recomputed_provider_lifecycle = "excluded_repeated_failure"
        elif missing_ids:
            recomputed_provider_lifecycle = "partial_missing"
        elif any(status != "complete" for status in status_counts):
            recomputed_provider_lifecycle = "complete_with_exclusions"
        else:
            recomputed_provider_lifecycle = "complete"
        declared_provider_lifecycle = str(manifest.get("provider_lifecycle"))
        if declared_provider_lifecycle != recomputed_provider_lifecycle:
            raise AnalysisContractError(
                f"{provider} provider lifecycle differs from record-derived lifecycle"
            )
        history = manifest.get("lifecycle_history")
        if history is not None:
            if (
                not isinstance(history, list)
                or not history
                or not isinstance(history[-1], Mapping)
                or history[-1].get("provider_lifecycle")
                != recomputed_provider_lifecycle
            ):
                raise AnalysisContractError(
                    f"{provider} latest lifecycle history differs from record-derived lifecycle"
                )
        provider_lifecycle = recomputed_provider_lifecycle
        returned_models = sorted(
            {
                str(record["model_id"])
                for record in records.values()
                if record.get("model_id")
            }
        )
        if (
            manifest.get("requested_model") != provider_models[provider]
            or manifest.get("returned_models") != returned_models
        ):
            raise AnalysisContractError(f"{provider} provider model set does not reconcile")
        known_cost = round(
            sum(float(record["known_cost_yuan"]) for record in records.values()),
            8,
        )
        reserve_cost = round(
            sum(
                float(record["unknown_cost_reserve_yuan"])
                for record in records.values()
            ),
            8,
        )
        if (
            not math.isclose(
                float(budget.get("provider_record_known_cost_yuan", -1.0)),
                known_cost,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or not math.isclose(
                float(budget.get("provider_record_unknown_reserve_yuan", -1.0)),
                reserve_cost,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or not math.isclose(
                float(budget.get("provider_record_accounted_cost_yuan", -1.0)),
                round(known_cost + reserve_cost, 8),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            raise AnalysisContractError(
                f"{provider} provider budget does not reconcile to record attempts"
            )
        needs_user_ids = [
            task_id
            for task_id in expected_ids
            if task_id in records and records[task_id].get("needs_user") is True
        ]
        unknown_attempt_count = sum(
            sum(
                attempt.get("cost_known") is False
                for attempt in record.get("attempts", ())
                if isinstance(attempt, Mapping)
            )
            for record in records.values()
        )
        expected_needs_user = {
            "required": bool(needs_user_ids),
            "reason": (
                "unknown_provider_billing_reserved" if needs_user_ids else None
            ),
            "record_count": len(needs_user_ids),
            "record_task_ids": needs_user_ids,
            "unknown_cost_attempt_count": unknown_attempt_count,
        }
        if manifest.get("needs_user") != expected_needs_user:
            raise AnalysisContractError(
                f"{provider} provider needs_user billing summary does not reconcile"
            )
        condition_eligibility: dict[str, tuple[bool, list[str]]] = {}
        provider_hard_reason = {
            "unavailable": "unavailable",
            "interrupted": "interrupted",
            "excluded_repeated_failure": "repeated_failure",
        }.get(provider_lifecycle)
        for condition in ("controlled", "blind"):
            reasons: list[str] = []
            if provider_hard_reason is not None:
                reasons.append(provider_hard_reason)
            if any(
                task["condition"] == condition
                and str(task["task_id"]) in records
                and records[str(task["task_id"])].get("status")
                == "excluded_model_drift"
                for task in expected_tasks
            ):
                reasons.append("model_drift")
            lane = recomputed_conditions[condition]
            if lane["complete_cluster_count"] < lane["minimum_complete_clusters"]:
                reasons.append("minimum_complete_clusters")
            if condition == "blind" and lane["excluded_invalid_schema"]:
                reasons.append("invalid_schema_strictly_above_half")
            condition_eligibility[condition] = (not reasons, reasons)
        lifecycle_rows.append(
            {
                "provider": provider,
                "provider_lifecycle": provider_lifecycle,
                "recomputed_provider_lifecycle": provider_lifecycle,
                "requested_model": str(manifest["requested_model"]),
                "returned_models": returned_models,
                "observed_model_ids_match_request": (
                    all(
                        model == str(manifest["requested_model"])
                        for model in returned_models
                    )
                    if returned_models
                    else None
                ),
                "expected_count": len(expected_ids),
                "present_count": len(present_ids),
                "missing_count": len(missing_ids),
                "status_counts": dict(sorted(status_counts.items())),
                "missing_task_ids": missing_ids,
                "known_cost_yuan": known_cost,
                "unknown_cost_reserve_yuan": reserve_cost,
                "accounted_cost_yuan": round(known_cost + reserve_cost, 8),
                "controlled_complete_cluster_count": recomputed_conditions[
                    "controlled"
                ]["complete_cluster_count"],
                "controlled_eligible": condition_eligibility["controlled"][0],
                "controlled_exclusion_reasons": condition_eligibility[
                    "controlled"
                ][1],
                "blind_complete_cluster_count": recomputed_conditions["blind"][
                    "complete_cluster_count"
                ],
                "blind_eligible": condition_eligibility["blind"][0],
                "blind_exclusion_reasons": condition_eligibility["blind"][1],
            }
        )
    return lifecycle_rows


def _provider_phase_cost_rows(
    records_by_phase_provider: Mapping[
        str, Mapping[str, Mapping[str, Mapping[str, Any]]]
    ],
) -> list[dict[str, Any]]:
    """Summarize requests, tokens, retries, and reconciled CNY by lane."""

    from .collect import _reconciled_record_cost

    rows: list[dict[str, Any]] = []
    for phase in sorted(records_by_phase_provider):
        for provider in sorted(records_by_phase_provider[phase]):
            records = records_by_phase_provider[phase][provider]
            requests = responses = retries = input_tokens = output_tokens = 0
            known = reserve = 0.0
            unknown_attempts = 0
            for task_id, record in records.items():
                try:
                    cost = _reconciled_record_cost(
                        record,
                        path=Path(f"<memory>/{phase}/{provider}/{task_id}.json"),
                        unknown_attempt_reserve_yuan=10.0,
                    )
                except ValueError as exc:
                    raise AnalysisContractError(
                        f"{phase}/{provider} record cost does not reconcile: {exc}"
                    ) from exc
                attempts = record.get("attempts")
                if not isinstance(attempts, list):
                    raise AnalysisContractError("record attempt ledger is invalid")
                requests += len(attempts)
                retries += int(record.get("retry_count") or 0)
                known += cost["known_cost_yuan"]
                reserve += cost["unknown_cost_reserve_yuan"]
                unknown_attempts += int(cost["unknown_attempt_count"])
                for attempt in attempts:
                    if not isinstance(attempt, Mapping):
                        raise AnalysisContractError("record attempt row is invalid")
                    usage = attempt.get("usage")
                    if attempt.get("cost_known") is True:
                        responses += 1
                        if not isinstance(usage, Mapping):
                            raise AnalysisContractError(
                                "known-cost provider response lacks token usage"
                            )
                        for field in ("input_tokens", "output_tokens"):
                            value = usage.get(field)
                            if (
                                not isinstance(value, int)
                                or isinstance(value, bool)
                                or value < 0
                            ):
                                raise AnalysisContractError(
                                    "provider response token usage is invalid"
                                )
                        input_tokens += int(usage["input_tokens"])
                        output_tokens += int(usage["output_tokens"])
            known = round(known, 8)
            reserve = round(reserve, 8)
            rows.append(
                {
                    "phase": phase,
                    "provider": provider,
                    "record_count": len(records),
                    "requests": requests,
                    "responses": responses,
                    "retries": retries,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "known_cost_yuan": known,
                    "unknown_cost_reserve_yuan": reserve,
                    "accounted_cost_yuan": round(known + reserve, 8),
                    "unknown_attempt_count": unknown_attempts,
                    "needs_user": unknown_attempts > 0,
                }
            )
    return rows


def _controlled_analysis(
    *,
    expected_tasks: Sequence[Mapping[str, Any]],
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
    persona_ids: Sequence[str],
    eligible_providers: Sequence[str],
) -> dict[str, Any]:
    controlled = [row for row in expected_tasks if row["condition"] == "controlled"]
    by_provider: list[dict[str, Any]] = []
    eligible = tuple(sorted(str(provider) for provider in eligible_providers))
    aggregate_counts = Counter({state: 0 for state in CONTROLLED_STATES})
    all_provider_counts = Counter({state: 0 for state in CONTROLLED_STATES})
    persona_metrics: dict[str, dict[str, dict[str, float | None]]] = {
        metric: {provider: {} for provider in FROZEN_PROVIDERS}
        for metric in (
            "conditional_answer_accuracy",
            "correct_response_yield",
            "incorrect_response_yield",
            "abstention_yield",
            "technical_or_schema_failure_yield",
        )
    }
    for provider in FROZEN_PROVIDERS:
        provider_arms: list[dict[str, Any]] = []
        for arm in ("deficit", "control"):
            arm_tasks = [row for row in controlled if row["response_arm"] == arm]
            states = [
                controlled_response_state(
                    records_by_provider[provider].get(str(task["task_id"]))
                )
                for task in arm_tasks
            ]
            counts = Counter(states)
            normalized_counts = {state: counts.get(state, 0) for state in CONTROLLED_STATES}
            all_provider_counts.update(normalized_counts)
            if provider in eligible:
                aggregate_counts.update(normalized_counts)
            answered = counts["correct_answer"] + counts["incorrect_answer"]
            provider_arms.append(
                {
                    "response_arm": arm,
                    "expected_denominator": len(arm_tasks),
                    "counts": normalized_counts,
                    "rates": {
                        state: normalized_counts[state] / len(arm_tasks)
                        for state in CONTROLLED_STATES
                    },
                    "conditional_answer_accuracy": (
                        counts["correct_answer"] / answered if answered else None
                    ),
                    "conditional_answer_denominator": answered,
                }
            )
        by_provider.append({"provider": provider, "arms": provider_arms})

        for persona_id in persona_ids:
            arm_values: dict[str, dict[str, float | None]] = {}
            for arm in ("deficit", "control"):
                tasks = [
                    row
                    for row in controlled
                    if row["persona_id"] == persona_id and row["response_arm"] == arm
                ]
                states = [
                    controlled_response_state(
                        records_by_provider[provider].get(str(task["task_id"]))
                    )
                    for task in tasks
                ]
                counts = Counter(states)
                answered = counts["correct_answer"] + counts["incorrect_answer"]
                arm_values[arm] = {
                    "conditional_answer_accuracy": (
                        counts["correct_answer"] / answered if answered else None
                    ),
                    "correct_response_yield": counts["correct_answer"] / len(tasks),
                    "incorrect_response_yield": counts["incorrect_answer"] / len(tasks),
                    "abstention_yield": counts["abstention"] / len(tasks),
                    "technical_or_schema_failure_yield": counts[
                        "technical_or_schema_failure"
                    ]
                    / len(tasks),
                }
            for metric in persona_metrics:
                deficit = arm_values["deficit"][metric]
                control = arm_values["control"][metric]
                if deficit is None or control is None:
                    value = None
                elif metric in {
                    "conditional_answer_accuracy",
                    "correct_response_yield",
                }:
                    value = control - deficit
                else:
                    value = deficit - control
                persona_metrics[metric][provider][persona_id] = value

    effects: list[dict[str, Any]] = []
    for metric, provider_values in persona_metrics.items():
        eligible_values = {
            provider: provider_values[provider] for provider in eligible
        }
        if eligible_values:
            bootstrap = cluster_bootstrap_mean(
                eligible_values, persona_ids=persona_ids
            )
        else:
            bootstrap = {
                "point_estimate": None,
                "ci95": None,
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
                "defined_resamples": 0,
                "undefined_resamples": BOOTSTRAP_RESAMPLES,
                "provider_equal_weighting": True,
                "provider_point_estimates": {},
            }
        orientation = (
            "control_minus_deficit"
            if metric in {"conditional_answer_accuracy", "correct_response_yield"}
            else "deficit_minus_control"
        )
        provider_rows: list[dict[str, Any]] = []
        for provider in FROZEN_PROVIDERS:
            provider_bootstrap = cluster_bootstrap_mean(
                {provider: provider_values[provider]},
                persona_ids=persona_ids,
            )
            provider_rows.append(
                {
                    "provider": provider,
                    "included_in_aggregate": provider in eligible,
                    "estimate": provider_bootstrap["point_estimate"],
                    "ci95": provider_bootstrap["ci95"],
                    "paired_persona_denominator": sum(
                        value is not None
                        for value in provider_values[provider].values()
                    ),
                    "bootstrap": provider_bootstrap,
                }
            )
        denominators = {
            row["provider"]: row["paired_persona_denominator"]
            for row in provider_rows
            if row["included_in_aggregate"]
        }
        effects.append(
            {
                "metric_id": metric,
                "orientation": orientation,
                "estimate": bootstrap["point_estimate"],
                "ci95": bootstrap["ci95"],
                "eligible_providers": list(eligible),
                "paired_persona_denominators": denominators,
                "paired_persona_denominator_range": (
                    [min(denominators.values()), max(denominators.values())]
                    if denominators
                    else None
                ),
                "by_provider": provider_rows,
                "bootstrap": bootstrap,
            }
        )
    return {
        "eligible_providers": list(eligible),
        "excluded_providers": [
            provider for provider in FROZEN_PROVIDERS if provider not in eligible
        ],
        "composition": {
            "states": list(CONTROLLED_STATES),
            "expected_tasks_per_provider": len(controlled),
            "by_provider": by_provider,
            "aggregate_counts": {
                state: aggregate_counts[state] for state in CONTROLLED_STATES
            },
            "all_provider_counts": {
                state: all_provider_counts[state] for state in CONTROLLED_STATES
            },
        },
        "paired_effects": effects,
    }


def _terminal_category(record: Mapping[str, Any] | None) -> str:
    if not isinstance(record, Mapping) or record.get("status") != "complete":
        return "NC"
    parsed = _mapping(record.get("parsed_output"), "complete blind parsed output")
    answer = parsed.get("answer")
    return "ABSTAIN" if answer is None else str(answer).strip().upper()


def _canonical_output(record: Mapping[str, Any] | None) -> bytes | None:
    if not isinstance(record, Mapping) or record.get("status") != "complete":
        return None
    parsed = record.get("parsed_output")
    return _canonical_bytes(parsed) if isinstance(parsed, Mapping) else None


def _blind_analysis(
    *,
    expected_tasks: Sequence[Mapping[str, Any]],
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
    persona_ids: Sequence[str],
    lifecycle_by_provider: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    primary = [
        row
        for row in expected_tasks
        if row["condition"] == "blind" and row["is_stability_repeat"] is False
    ]
    terminal = [row for row in primary if row["is_terminal"] is True]
    repeats = [
        row
        for row in expected_tasks
        if row["condition"] == "blind" and row["is_stability_repeat"] is True
    ]
    if len(terminal) != 100:
        raise AnalysisContractError("frozen main requires one terminal blind task per paired row")
    if len(repeats) != 20:
        raise AnalysisContractError("frozen main requires twenty terminal stability repeats")
    terminal_keys = {
        (str(row["persona_id"]), str(row["response_arm"]), str(row["item_id"])): row
        for row in terminal
    }
    if len(terminal_keys) != len(terminal):
        raise AnalysisContractError("terminal blind roster is not unique")
    subjects = [
        f"{persona_id}|{arm}"
        for persona_id in persona_ids
        for arm in ("deficit", "control")
    ]
    provider_schema: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    ratings: dict[str, dict[str, str]] = {}
    failure_values: dict[str, dict[str, float | None]] = {}
    for provider in FROZEN_PROVIDERS:
        invalid = sum(
            records_by_provider[provider].get(str(task["task_id"]), {}).get("status")
            == "excluded_schema"
            for task in primary
        )
        fraction = invalid / len(primary)
        lifecycle = _mapping(
            lifecycle_by_provider.get(provider), f"{provider} lifecycle eligibility"
        )
        is_excluded = lifecycle.get("blind_eligible") is not True
        if is_excluded:
            excluded.append(provider)
        provider_schema[provider] = {
            "expected_primary_blind_tasks": len(primary),
            "invalid_schema_count": invalid,
            "invalid_schema_fraction": fraction,
            "strictly_above_half": fraction > 0.5,
            "invalid_schema_strictly_above_half": fraction > 0.5,
            "excluded_from_blind_aggregate": is_excluded,
            "complete_cluster_count": lifecycle.get(
                "blind_complete_cluster_count"
            ),
            "exclusion_reasons": list(
                lifecycle.get("blind_exclusion_reasons") or ()
            ),
        }
        ratings[provider] = {}
        for task in terminal:
            subject = f"{task['persona_id']}|{task['response_arm']}"
            ratings[provider][subject] = _terminal_category(
                records_by_provider[provider].get(str(task["task_id"]))
            )
        failure_values[provider] = {}
        for persona_id in persona_ids:
            persona_tasks = [row for row in primary if row["persona_id"] == persona_id]
            failure_values[provider][persona_id] = sum(
                not isinstance(records_by_provider[provider].get(str(task["task_id"])), Mapping)
                or records_by_provider[provider][str(task["task_id"])].get("status")
                != "complete"
                for task in persona_tasks
            ) / len(persona_tasks)
    eligible = [provider for provider in FROZEN_PROVIDERS if provider not in excluded]
    if len(eligible) >= 2:
        agreement = pairwise_terminal_agreement(
            {provider: ratings[provider] for provider in eligible},
            subjects=subjects,
        )
        agreement["status"] = "estimated"
    else:
        visible_categories = sorted(
            {
                value
                for provider in eligible
                for value in ratings[provider].values()
            }
            | {"NC"}
        )
        agreement = {
            "status": "not_estimable",
            "reason": "fewer_than_two_blind_eligible_providers",
            "subjects": subjects,
            "providers": eligible,
            "categories": visible_categories,
            "nc_retained": True,
            "pairs": [],
        }
    for pair in agreement["pairs"]:
        left = str(pair["provider_left"])
        right = str(pair["provider_right"])
        persona_values = {
            persona_id: sum(
                ratings[left][f"{persona_id}|{arm}"]
                == ratings[right][f"{persona_id}|{arm}"]
                for arm in ("deficit", "control")
            )
            / 2.0
            for persona_id in persona_ids
        }
        bootstrap = cluster_bootstrap_mean(
            {f"{left}__{right}": persona_values},
            persona_ids=persona_ids,
        )
        pair["exact_agreement_ci95"] = bootstrap["ci95"]
        pair["exact_agreement_bootstrap"] = bootstrap
    if eligible:
        failure_bootstrap = cluster_bootstrap_mean(
            {provider: failure_values[provider] for provider in eligible},
            persona_ids=persona_ids,
        )
    else:
        failure_bootstrap = {
            "point_estimate": None,
            "ci95": None,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "defined_resamples": 0,
            "undefined_resamples": BOOTSTRAP_RESAMPLES,
            "provider_equal_weighting": True,
            "provider_point_estimates": {},
        }

    stability: list[dict[str, Any]] = []
    answer_values_by_provider: dict[str, dict[str, float | None]] = {}
    canonical_values_by_provider: dict[str, dict[str, float | None]] = {}
    for provider in FROZEN_PROVIDERS:
        answer_matches = 0
        complete_pairs = 0
        canonical_matches = 0
        nc_nc_matches = 0
        answer_flags: dict[str, list[float]] = {persona_id: [] for persona_id in persona_ids}
        canonical_flags: dict[str, list[float]] = {
            persona_id: [] for persona_id in persona_ids
        }
        for repeat in repeats:
            key = (
                str(repeat["persona_id"]),
                str(repeat["response_arm"]),
                str(repeat["item_id"]),
            )
            primary_task = terminal_keys.get(key)
            if primary_task is None:
                raise AnalysisContractError("stability repeat lacks its frozen terminal primary")
            primary_record = records_by_provider[provider].get(str(primary_task["task_id"]))
            repeat_record = records_by_provider[provider].get(str(repeat["task_id"]))
            primary_category = _terminal_category(primary_record)
            repeat_category = _terminal_category(repeat_record)
            if primary_category == repeat_category:
                answer_matches += 1
                if primary_category == "NC":
                    nc_nc_matches += 1
            answer_flags[str(repeat["persona_id"])].append(
                float(primary_category == repeat_category)
            )
            primary_output = _canonical_output(primary_record)
            repeat_output = _canonical_output(repeat_record)
            if primary_output is not None and repeat_output is not None:
                complete_pairs += 1
                if primary_output == repeat_output:
                    canonical_matches += 1
                canonical_flags[str(repeat["persona_id"])].append(
                    float(primary_output == repeat_output)
                )
        answer_persona = {
            persona_id: (
                sum(answer_flags[persona_id]) / len(answer_flags[persona_id])
                if answer_flags[persona_id]
                else None
            )
            for persona_id in persona_ids
        }
        canonical_persona = {
            persona_id: (
                sum(canonical_flags[persona_id]) / len(canonical_flags[persona_id])
                if canonical_flags[persona_id]
                else None
            )
            for persona_id in persona_ids
        }
        answer_values_by_provider[provider] = answer_persona
        canonical_values_by_provider[provider] = canonical_persona
        if provider in eligible:
            answer_bootstrap = cluster_bootstrap_mean(
                {provider: answer_persona}, persona_ids=persona_ids
            )
            canonical_bootstrap = cluster_bootstrap_mean(
                {provider: canonical_persona}, persona_ids=persona_ids
            )
            answer_denominator = len(repeats)
            answer_numerator: int | None = answer_matches
            answer_agreement: float | None = answer_matches / len(repeats)
            row_status = "estimated"
        else:
            answer_bootstrap = {
                "point_estimate": None,
                "ci95": None,
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
                "defined_resamples": 0,
                "undefined_resamples": BOOTSTRAP_RESAMPLES,
                "provider_equal_weighting": True,
                "provider_point_estimates": {},
            }
            canonical_bootstrap = dict(answer_bootstrap)
            answer_denominator = 0
            answer_numerator = None
            answer_agreement = None
            row_status = "not_estimable_ineligible_lane"
        stability.append(
            {
                "provider": provider,
                "excluded_from_blind_aggregate": provider in excluded,
                "status": row_status,
                "expected_pairs": len(repeats),
                "answer_agreement_numerator": answer_numerator,
                "answer_agreement_denominator": answer_denominator,
                "answer_agreement": answer_agreement,
                "answer_bootstrap": answer_bootstrap,
                "nc_nc_agreement_count": nc_nc_matches,
                "canonical_complete_pair_numerator": canonical_matches,
                "canonical_complete_pair_denominator": complete_pairs,
                "canonical_complete_pair_stability": (
                    canonical_matches / complete_pairs if complete_pairs else None
                ),
                "canonical_complete_pair_bootstrap": canonical_bootstrap,
                "canonical_itt_yield": canonical_matches / len(repeats),
            }
        )
    if eligible:
        aggregate_answer_stability = cluster_bootstrap_mean(
            {provider: answer_values_by_provider[provider] for provider in eligible},
            persona_ids=persona_ids,
        )
        aggregate_canonical_stability = cluster_bootstrap_mean(
            {
                provider: canonical_values_by_provider[provider]
                for provider in eligible
            },
            persona_ids=persona_ids,
        )
    else:
        aggregate_answer_stability = {
            "point_estimate": None,
            "ci95": None,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "defined_resamples": 0,
            "undefined_resamples": BOOTSTRAP_RESAMPLES,
            "provider_equal_weighting": True,
            "provider_point_estimates": {},
        }
        aggregate_canonical_stability = dict(aggregate_answer_stability)
    unanimous = (
        sum(
            len({ratings[provider][subject] for provider in eligible}) == 1
            for subject in subjects
        )
        if len(eligible) >= 2
        else None
    )
    return {
        "primary_terminal_definition": "frozen_final_blind_item",
        "terminal_subject_count": len(subjects),
        "terminal_categories": agreement["categories"],
        "provider_schema": provider_schema,
        "eligible_providers": eligible,
        "excluded_providers": excluded,
        "agreement": agreement,
        "multi_provider_descriptive": {
            "rectangular_providers": eligible,
            "subjects": len(subjects),
            "unanimous_numerator": unanimous,
            "unanimous_fraction": (
                unanimous / len(subjects) if unanimous is not None else None
            ),
            "status": "estimated" if unanimous is not None else "not_estimable",
        },
        "technical_or_schema_failure_rate": {
            "estimate": failure_bootstrap["point_estimate"],
            "ci95": failure_bootstrap["ci95"],
            "bootstrap": failure_bootstrap,
        },
        "stability": stability,
        "stability_provider_equal_aggregate": {
            "eligible_providers": eligible,
            "answer": aggregate_answer_stability,
            "canonical_complete_pair": aggregate_canonical_stability,
        },
    }


def _sparse_mapping_analysis(
    *,
    support: Mapping[str, Any],
    expected_tasks: Sequence[Mapping[str, Any]],
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    mapped_tasks = [
        row
        for row in expected_tasks
        if row["condition"] == "controlled" and row.get("target_option") is not None
    ]
    by_provider_arm: list[dict[str, Any]] = []
    for provider in FROZEN_PROVIDERS:
        for arm in ("deficit", "control"):
            tasks = [row for row in mapped_tasks if row["response_arm"] == arm]
            task_records = [
                (
                    task,
                    records_by_provider[provider].get(str(task["task_id"])),
                )
                for task in tasks
            ]
            complete = [
                (task, record)
                for task, record in task_records
                if isinstance(record, Mapping) and record.get("status") == "complete"
            ]
            target_hits = sum(
                _mapping(record.get("outcomes"), "mapped task outcomes").get(
                    "target_option_hit"
                )
                is True
                for _, record in complete
            )
            compliance = sum(
                _mapping(record.get("outcomes"), "mapped task outcomes").get(
                    "manipulation_compliance"
                )
                is True
                for _, record in complete
            )
            incorrect = [
                (task, record)
                for task, record in complete
                if _mapping(record.get("parsed_output"), "mapped parsed output").get(
                    "answer"
                )
                is not None
                if _mapping(record.get("outcomes"), "mapped task outcomes").get(
                    "is_correct"
                )
                is False
            ]
            incorrect_hits = sum(
                _mapping(record.get("outcomes"), "mapped task outcomes").get(
                    "target_option_hit"
                )
                is True
                for _, record in incorrect
            )
            baseline = (
                sum(float(task["random_wrong_option_baseline"]) for task, _ in incorrect)
                / len(incorrect)
                if incorrect
                else None
            )
            conditional_hit_rate = (
                incorrect_hits / len(incorrect) if incorrect else None
            )
            by_provider_arm.append(
                {
                    "provider": provider,
                    "response_arm": arm,
                    "expected_task_cells": len(tasks),
                    "complete_task_cells": len(complete),
                    "target_option_hit_numerator": target_hits,
                    "target_option_hit_denominator": len(complete),
                    "target_option_hit_rate": (
                        target_hits / len(complete) if complete else None
                    ),
                    "manipulation_compliance_numerator": compliance,
                    "manipulation_compliance_denominator": len(complete),
                    "manipulation_compliance_rate": (
                        compliance / len(complete) if complete else None
                    ),
                    "incorrect_answer_denominator": len(incorrect),
                    "target_option_hit_among_incorrect_numerator": incorrect_hits,
                    "target_option_hit_among_incorrect_rate": conditional_hit_rate,
                    "random_wrong_option_baseline": baseline,
                    "target_hit_minus_random_wrong_baseline": (
                        conditional_hit_rate - baseline
                        if conditional_hit_rate is not None and baseline is not None
                        else None
                    ),
                }
            )
    return {**support, "by_provider_and_arm": by_provider_arm}


def _judge_candidates_from_terminal_records(
    *,
    expected_tasks: Sequence[Mapping[str, Any]],
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
    eligible_providers: Sequence[str],
) -> list[dict[str, Any]]:
    eligible = tuple(str(provider) for provider in eligible_providers)
    if len(eligible) < 2:
        return []
    terminal = sorted(
        (
            task
            for task in expected_tasks
            if task["condition"] == "blind"
            and task["is_stability_repeat"] is False
            and task["is_terminal"] is True
        ),
        key=lambda task: str(task["task_id"]),
    )
    candidates: list[dict[str, Any]] = []
    for task in terminal:
        task_id = str(task["task_id"])
        subject = f"{task['persona_id']}|{task['response_arm']}"
        categories = {
            _terminal_category(records_by_provider[provider].get(task_id))
            for provider in eligible
        }
        stratum = "disagreement" if len(categories) > 1 else "agreement"
        for provider in eligible:
            record = records_by_provider[provider].get(task_id)
            if not isinstance(record, Mapping) or record.get("status") != "complete":
                continue
            parsed = _mapping(record.get("parsed_output"), "terminal parsed output")
            candidates.append(
                {
                    "candidate_identity": _canonical_sha(
                        {
                            "provider": provider,
                            "task_id": task_id,
                            "model_id": record.get("model_id"),
                            "phase_provenance_sha256": _mapping(
                                record.get("provenance"), "record provenance"
                            ).get("phase_provenance_sha256"),
                        }
                    ),
                    "subject_id": subject,
                    "stratum": stratum,
                    "public_question": _mapping(
                        task.get("public_question"), "terminal public question"
                    ),
                    "model_output": dict(parsed),
                    "persona": _mapping(
                        task.get("persona_contract"), "terminal persona contract"
                    ),
                    "item": _mapping(
                        task.get("item_contract"), "terminal item contract"
                    ),
                }
            )
    return candidates


def _collect_provider_identity_terms(
    *,
    active_contract_proof: Mapping[str, Any],
    provider_manifests: Mapping[str, Mapping[str, Any]],
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[str]:
    terms = {
        alias
        for provider in FROZEN_PROVIDERS
        for alias in (provider, *_PROVIDER_IDENTITY_ALIASES[provider])
    }
    provider_models = _mapping(
        active_contract_proof.get("provider_models"), "active provider models"
    )
    for provider in FROZEN_PROVIDERS:
        terms.add(str(provider_models[provider]))
        manifest = provider_manifests[provider]
        terms.add(str(manifest.get("requested_model") or ""))
        returned_models = manifest.get("returned_models")
        if isinstance(returned_models, list):
            terms.update(str(value) for value in returned_models if value)
        for record in records_by_provider[provider].values():
            terms.update(
                str(value)
                for value in (
                    record.get("requested_model"),
                    record.get("model_id"),
                )
                if value
            )
            attempts = record.get("attempts")
            if isinstance(attempts, list):
                terms.update(
                    str(attempt.get("model_returned"))
                    for attempt in attempts
                    if isinstance(attempt, Mapping) and attempt.get("model_returned")
                )
    return sorted(value for value in terms if value.strip())


def _prepare_cost_accounting(
    *,
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
    cost_accounting: Mapping[str, Any] | None,
    formal_mode: bool,
) -> dict[str, Any]:
    recomputed_main_rows = _provider_phase_cost_rows(
        {"main": records_by_provider}
    )
    if not formal_mode:
        if cost_accounting is not None:
            raise AnalysisContractError(
                "formal cost accounting requires a loader-generated proof"
            )
        cost_result: dict[str, Any] = {
            "schema_version": "yher.llm_sim_v2.cost_accounting.nonformal.v1",
            "currency": "CNY",
            "provider_phase": recomputed_main_rows,
            "source": "synthetic_nonformal_recomputed_records",
            "cost_reconciliation_artifact_manifest": {
                "schema_version": (
                    "yher.llm_sim_v2.cost_reconciliation_artifact_manifest.v1"
                ),
                "simulated": True,
                "run_id": RUN_ID,
                "metric_input": False,
                "file_count": 0,
                "files": [],
                "file_set_sha256": _canonical_sha([]),
            },
        }
    else:
        cost_result = dict(_mapping(cost_accounting, "formal cost accounting"))
        if (
            cost_result.get("schema_version")
            != "yher.llm_sim_v2.cost_accounting.v1"
            or cost_result.get("currency") != "CNY"
            or not isinstance(cost_result.get("provider_phase"), list)
            or not isinstance(cost_result.get("judge"), list)
            or not isinstance(
                cost_result.get("cost_reconciliation_artifact_manifest"), Mapping
            )
        ):
            raise AnalysisContractError("formal cost accounting envelope is invalid")
        actual_main_rows = sorted(
            (
                dict(_mapping(row, "formal provider-phase cost row"))
                for row in cost_result["provider_phase"]
                if isinstance(row, Mapping) and row.get("phase") == "main"
            ),
            key=lambda row: str(row.get("provider")),
        )
        if actual_main_rows != sorted(
            recomputed_main_rows, key=lambda row: str(row.get("provider"))
        ):
            raise AnalysisContractError(
                "formal main provider-phase costs differ from validated records"
            )
    cost_manifest = _mapping(
        cost_result.get("cost_reconciliation_artifact_manifest"),
        "cost reconciliation artifact manifest",
    )
    cost_files = cost_manifest.get("files")
    if (
        cost_manifest.get("schema_version")
        != "yher.llm_sim_v2.cost_reconciliation_artifact_manifest.v1"
        or cost_manifest.get("metric_input") is not False
        or not isinstance(cost_files, list)
        or cost_manifest.get("file_count") != len(cost_files)
        or cost_manifest.get("file_set_sha256") != _canonical_sha(cost_files)
        or any(
            not isinstance(row, Mapping)
            or row.get("included_in_metrics") is not False
            for row in cost_files
        )
    ):
        raise AnalysisContractError(
            "cost reconciliation artifact manifest is invalid"
        )
    return cost_result


def _collection_provenance(
    *,
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
    provider_manifests: Mapping[str, Mapping[str, Any]],
    active_contract_proof: Mapping[str, Any],
    phase_provenance: Mapping[str, Any],
    phase_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    policies = _mapping(
        active_contract_proof.get("provider_attempt_policies"),
        "active provider attempt policies",
    )
    formal_window: dict[str, Any] | None = None
    provider_windows: Mapping[str, Any] = {}
    time_semantics = "immutable_provider_evidence_recorded_at_utc"
    if phase_evidence is not None:
        formal_window = dict(
            _mapping(
                phase_evidence.get("provider_evidence_event_window_utc"),
                "formal provider evidence window",
            )
        )
        provider_windows = _mapping(
            phase_evidence.get("provider_event_windows"),
            "formal provider event windows",
        )
        if phase_evidence.get("time_semantics") != time_semantics:
            raise AnalysisContractError("formal provider evidence time semantics drifted")
    rows: list[dict[str, Any]] = []
    for provider in FROZEN_PROVIDERS:
        manifest = _mapping(
            provider_manifests[provider], f"{provider} provider manifest"
        )
        policy = _mapping(policies[provider], f"{provider} active attempt policy")
        requested_model = str(manifest["requested_model"])
        returned_models = [str(value) for value in manifest["returned_models"]]
        observed_tokens = sorted(
            {
                int(attempt["request_max_tokens"])
                for record in records_by_provider[provider].values()
                for attempt in record.get("attempts", ())
                if isinstance(attempt, Mapping)
                and isinstance(attempt.get("request_max_tokens"), int)
                and not isinstance(attempt.get("request_max_tokens"), bool)
            }
        )
        rows.append(
            {
                "provider": provider,
                "requested_model": requested_model,
                "returned_models": returned_models,
                "observed_model_ids_match_request": (
                    all(model == requested_model for model in returned_models)
                    if returned_models
                    else None
                ),
                "evidence_event_window_utc": (
                    dict(
                        _mapping(
                            provider_windows[provider],
                            f"{provider} evidence window",
                        )
                    )
                    if phase_evidence is not None
                    else None
                ),
                "observed_request_max_tokens": observed_tokens,
                "max_tokens": int(policy["max_tokens"]),
                "retry_max_tokens": int(policy["retry_max_tokens"]),
                "timeout_seconds": float(policy["timeout_seconds"]),
                "concurrency": int(policy["concurrency"]),
                "max_attempts": int(policy["max_attempts"]),
                "failure_threshold": int(policy["failure_threshold"]),
                "base_backoff_seconds": float(policy["base_backoff_seconds"]),
                "max_backoff_seconds": float(policy["max_backoff_seconds"]),
                "cooldown_seconds": float(policy["cooldown_seconds"]),
                "jitter_fraction": float(policy["jitter_fraction"]),
            }
        )
    return {
        "schema_version": "yher.llm_sim_v2.collection_provenance.v1",
        "temperature": float(active_contract_proof["request_temperature"]),
        "top_p": None,
        "seed": None,
        "time_semantics": time_semantics,
        "first_observation_at_utc": (
            _utc_timestamp(
                phase_provenance.get("first_observation_at_utc"),
                "phase first_observation_at_utc",
            )
            if phase_provenance.get("first_observation_at_utc") is not None
            else None
        ),
        "provider_evidence_event_window_utc": formal_window,
        "active_analysis_contract_proof_sha256": active_contract_proof[
            "active_analysis_contract_proof_sha256"
        ],
        "providers": rows,
    }


def analyze_dataset(
    *,
    expected_tasks: Sequence[Mapping[str, Any]],
    runtime_manifest: Mapping[str, Any],
    phase_provenance: Mapping[str, Any],
    records_by_provider: Mapping[str, Mapping[str, Mapping[str, Any]]],
    provider_manifests: Mapping[str, Mapping[str, Any]],
    mapping_manifest: Mapping[str, Any],
    active_contract_proof: Mapping[str, Any],
    cost_accounting: Mapping[str, Any] | None = None,
    judge_result_manifests: Mapping[str, Mapping[str, Any] | None] | None = None,
    judge_run_evidence: Mapping[str, Any] | None = None,
    judge_artifact_roots: Mapping[str, str] | None = None,
    judge_artifact_sources: Mapping[str, str] | None = None,
    input_artifact_manifest: Mapping[str, Any] | None = None,
    phase_evidence: Mapping[str, Any] | None = None,
    _formal_loader_proof: _FormalLoaderProof | None = None,
) -> dict[str, Any]:
    """Analyze records; only a filesystem loader capability can produce formal output."""

    loader_bundle = {
        "expected_tasks": expected_tasks,
        "runtime_manifest": runtime_manifest,
        "phase_provenance": phase_provenance,
        "records_by_provider": records_by_provider,
        "provider_manifests": provider_manifests,
        "mapping_manifest": mapping_manifest,
        "active_contract_proof": active_contract_proof,
        "cost_accounting": cost_accounting,
        "judge_result_manifests": judge_result_manifests,
        "judge_run_evidence": judge_run_evidence,
        "judge_artifact_roots": judge_artifact_roots,
        "judge_artifact_sources": judge_artifact_sources,
        "input_artifact_manifest": input_artifact_manifest,
        "phase_evidence": phase_evidence,
    }
    formal_mode = _formal_loader_proof is not None
    if formal_mode:
        if (
            not isinstance(_formal_loader_proof, _FormalLoaderProof)
            or _formal_loader_proof.bundle_sha256
            != _formal_loader_bundle_sha256(loader_bundle)
            or cost_accounting is None
            or input_artifact_manifest is None
            or judge_result_manifests is None
            or judge_artifact_roots is None
            or judge_artifact_sources is None
            or phase_evidence is None
            or bool(judge_artifact_sources) != (judge_run_evidence is not None)
        ):
            raise AnalysisContractError("formal loader proof does not bind analysis inputs")
    elif (
        cost_accounting is not None
        or input_artifact_manifest is not None
        or phase_evidence is not None
        or judge_run_evidence is not None
        or judge_artifact_roots is not None
        or judge_artifact_sources is not None
    ):
        raise AnalysisContractError(
            "formal cost/input artifacts require a loader-generated proof"
        )
    finalized_formal_mode = formal_mode and judge_run_evidence is not None

    input_proof = validate_inputs(
        phase_provenance=phase_provenance,
        runtime_manifest=runtime_manifest,
        expected_tasks=expected_tasks,
        active_contract_proof=active_contract_proof,
    )
    mapping_support = _validate_mapping_manifest(
        mapping_manifest,
        phase=phase_provenance,
        expected_tasks=expected_tasks,
    )
    lifecycle = _validate_provider_inputs(
        expected_tasks=expected_tasks,
        phase=phase_provenance,
        records_by_provider=records_by_provider,
        provider_manifests=provider_manifests,
        active_contract_proof=active_contract_proof,
        formal_mode=formal_mode,
    )
    collection_provenance = _collection_provenance(
        records_by_provider=records_by_provider,
        provider_manifests=provider_manifests,
        active_contract_proof=active_contract_proof,
        phase_provenance=phase_provenance,
        phase_evidence=phase_evidence,
    )
    cost_result = _prepare_cost_accounting(
        records_by_provider=records_by_provider,
        cost_accounting=cost_accounting,
        formal_mode=formal_mode,
    )
    persona_ids = sorted({str(row["persona_id"]) for row in expected_tasks})
    lifecycle_index = {str(row["provider"]): row for row in lifecycle}
    controlled_eligible = [
        provider
        for provider in FROZEN_PROVIDERS
        if lifecycle_index[provider]["controlled_eligible"] is True
    ]
    controlled = _controlled_analysis(
        expected_tasks=expected_tasks,
        records_by_provider=records_by_provider,
        persona_ids=persona_ids,
        eligible_providers=controlled_eligible,
    )
    blind = _blind_analysis(
        expected_tasks=expected_tasks,
        records_by_provider=records_by_provider,
        persona_ids=persona_ids,
        lifecycle_by_provider=lifecycle_index,
    )
    sparse = _sparse_mapping_analysis(
        support=mapping_support,
        expected_tasks=expected_tasks,
        records_by_provider=records_by_provider,
    )
    judge_candidates = _judge_candidates_from_terminal_records(
        expected_tasks=expected_tasks,
        records_by_provider=records_by_provider,
        eligible_providers=blind["eligible_providers"],
    )
    case_manifest = build_judge_case_manifest(
        judge_candidates,
        frozen_leakage_lexicon=tuple(
            str(value)
            for value in active_contract_proof.get("frozen_leakage_lexicon", ())
        ),
        provider_identity_terms=_collect_provider_identity_terms(
            active_contract_proof=active_contract_proof,
            provider_manifests=provider_manifests,
            records_by_provider=records_by_provider,
        ),
    )
    judge_analysis = ingest_judge_results(
        case_manifest,
        judge_result_manifests
        if judge_result_manifests is not None
        else {"claude": None, "gpt": None},
        judge_artifact_roots=judge_artifact_roots,
        judge_run_evidence=judge_run_evidence,
    )
    if formal_mode:
        cost_judges = {
            str(row.get("judge")): _mapping(row, "formal judge cost row")
            for row in cost_result.get("judge", ())
            if isinstance(row, Mapping)
        }
        analysis_accounting = _mapping(
            judge_analysis.get("judge_accounting"),
            "formal judge analysis accounting",
        )
        complete_cost_judges = {
            judge: row
            for judge, row in cost_judges.items()
            if row.get("status") == "complete"
        }
        if set(complete_cost_judges) != set(analysis_accounting):
            raise AnalysisContractError(
                "formal judge costs differ from available adjudications"
            )
        for judge, accounting_value in analysis_accounting.items():
            accounting = _mapping(
                accounting_value, f"{judge} judge analysis accounting"
            )
            row = complete_cost_judges[judge]
            if any(
                row.get(field) != accounting.get(field)
                for field in (
                    "request_count",
                    "retry_count",
                    "transport_error_count",
                    "schema_error_count",
                    "content_retry_count",
                    "input_tokens",
                    "output_tokens",
                    "known_cost_yuan",
                    "unknown_cost_reserve_yuan",
                    "accounted_cost_yuan",
                )
            ):
                raise AnalysisContractError(
                    f"{judge} judge cost differs from the validated execution receipt"
                )
    return {
        "schema_version": (
            "yher.llm_sim_v2.analysis_results.v1"
            if finalized_formal_mode
            else "yher.llm_sim_v2.analysis_results.pre_adjudication.v1"
            if formal_mode
            else "yher.llm_sim_v2.analysis_results.synthetic_nonformal.v1"
        ),
        "simulated": True,
        "run_id": RUN_ID,
        "analysis_population": "main" if formal_mode else "synthetic_nonformal",
        "analysis_mode": (
            "formal_main"
            if finalized_formal_mode
            else "formal_main_pre_adjudication"
            if formal_mode
            else "synthetic_nonformal"
        ),
        "formal_analysis_eligible": finalized_formal_mode,
        "publication_output_eligible": finalized_formal_mode,
        "formal_loader_bundle_sha256": (
            _formal_loader_proof.bundle_sha256
            if formal_mode and _formal_loader_proof is not None
            else None
        ),
        "modality_condition": "text_only",
        "independent_cluster_unit": "persona_id",
        "independent_cluster_count": len(persona_ids),
        "repeated_measure_factors": ["provider", "response_arm", "condition", "item"],
        "bootstrap_contract": {
            "cluster_unit": "persona_id",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_interval": "two-sided percentile 95%",
            "provider_equal_weighting": True,
            "undefined_resamples_retained": True,
        },
        "expected_denominator": {
            "tasks_per_provider": len(expected_tasks),
            "provider_count": len(FROZEN_PROVIDERS),
            "provider_task_cells": len(expected_tasks) * len(FROZEN_PROVIDERS),
            "source": "committed_runtime_task_manifest",
            "filesystem_glob_defines_denominator": False,
        },
        "input_proof": input_proof,
        "input_artifact_binding": (
            {
                "input_file_count": input_artifact_manifest["input_file_count"],
                "record_file_count": input_artifact_manifest["record_file_count"],
                "input_file_set_sha256": input_artifact_manifest[
                    "input_file_set_sha256"
                ],
                "input_artifact_manifest_sha256": _canonical_sha(
                    input_artifact_manifest
                ),
            }
            if formal_mode and input_artifact_manifest is not None
            else None
        ),
        "phase_evidence_binding": (
            {
                "phase_evidence_receipt_sha256": _mapping(
                    phase_evidence.get("receipt"), "formal phase evidence receipt"
                )["phase_evidence_receipt_sha256"],
                "committed_anchor_sha256": _mapping(
                    phase_evidence.get("committed_anchor"),
                    "formal committed phase evidence anchor",
                )["sha256"],
            }
            if formal_mode and phase_evidence is not None
            else None
        ),
        "provider_lifecycle": lifecycle,
        "collection_provenance": collection_provenance,
        "controlled": controlled,
        "blind": blind,
        "sparse_mapping_descriptive": sparse,
        "cost_accounting": cost_result,
        "judge_adjudication": {
            "case_manifest": case_manifest,
            "analysis": judge_analysis,
            "run_evidence_binding": (
                {
                    "schema_version": (
                        "yher.llm_sim_v2.formal_judge_run_evidence_binding.v1"
                    ),
                    "judge_run_evidence_receipt_sha256": _mapping(
                        judge_run_evidence.get("receipt"),
                        "formal judge run evidence receipt",
                    )["judge_run_evidence_receipt_sha256"],
                    "committed_anchor_sha256": _mapping(
                        judge_run_evidence.get("committed_anchor"),
                        "formal committed judge run anchor",
                    )["sha256"],
                    "family_slots": _mapping(
                        judge_run_evidence.get("receipt"),
                        "formal judge run evidence receipt",
                    )["family_slots"],
                }
                if judge_run_evidence is not None
                else None
            ),
            "result_manifests": (
                dict(judge_result_manifests)
                if judge_result_manifests is not None
                else {"claude": None, "gpt": None}
            ),
        },
        "claim_boundary": (
            "independent simulated text-only response-channel stress test; "
            "not human participants, learner trajectories, or educational efficacy"
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for field in fieldnames
                    for value in (row.get(field),)
                }
            )


def _composition_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    composition = _mapping(
        _mapping(result.get("controlled"), "controlled results").get("composition"),
        "controlled composition",
    )
    for provider_row in composition.get("by_provider", ()):
        for arm_row in provider_row["arms"]:
            counts = arm_row["counts"]
            rates = arm_row["rates"]
            for state in CONTROLLED_STATES:
                output.append(
                    {
                        "provider": provider_row["provider"],
                        "response_arm": arm_row["response_arm"],
                        "state": state,
                        "count": counts[state],
                        "expected_denominator": arm_row["expected_denominator"],
                        "rate": rates[state],
                        "conditional_answer_accuracy": arm_row[
                            "conditional_answer_accuracy"
                        ],
                        "conditional_answer_denominator": arm_row[
                            "conditional_answer_denominator"
                        ],
                    }
                )
    return output


def _effect_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    effects = _mapping(result.get("controlled"), "controlled results").get(
        "paired_effects", ()
    )
    for effect in effects:
        ci = effect.get("ci95") or [None, None]
        output.append(
            {
                "metric_id": effect["metric_id"],
                "orientation": effect["orientation"],
                "scope": "provider_equal_aggregate",
                "provider": "aggregate",
                "estimate": effect["estimate"],
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "paired_persona_denominator": None,
                "paired_persona_denominators": effect[
                    "paired_persona_denominators"
                ],
                "paired_persona_denominator_range": effect[
                    "paired_persona_denominator_range"
                ],
                "bootstrap_seed": effect["bootstrap"]["seed"],
                "bootstrap_resamples": effect["bootstrap"]["resamples"],
                "undefined_resamples": effect["bootstrap"]["undefined_resamples"],
            }
        )
        for provider_row in effect["by_provider"]:
            provider_ci = provider_row.get("ci95") or [None, None]
            output.append(
                {
                    "metric_id": effect["metric_id"],
                    "orientation": effect["orientation"],
                    "scope": "provider",
                    "provider": provider_row["provider"],
                    "estimate": provider_row["estimate"],
                    "ci95_low": provider_ci[0],
                    "ci95_high": provider_ci[1],
                    "paired_persona_denominator": provider_row[
                        "paired_persona_denominator"
                    ],
                    "paired_persona_denominators": None,
                    "paired_persona_denominator_range": None,
                    "bootstrap_seed": provider_row["bootstrap"]["seed"],
                    "bootstrap_resamples": provider_row["bootstrap"]["resamples"],
                    "undefined_resamples": provider_row["bootstrap"][
                        "undefined_resamples"
                    ],
                }
            )
    return output


def _agreement_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    blind = _mapping(result.get("blind"), "blind results")
    agreement = _mapping(blind.get("agreement"), "blind agreement")
    return [dict(row) for row in agreement.get("pairs", ())]


def _stability_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    blind = _mapping(result.get("blind"), "blind results")
    return [dict(row) for row in blind.get("stability", ())]


def _mapping_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    sparse = _mapping(result.get("sparse_mapping_descriptive"), "sparse mapping")
    shared = {
        "confirmatory": sparse["confirmatory"],
        "status": sparse["status"],
        "mapped_mapping_rows": sparse["mapped_mapping_rows"],
        "total_mapping_rows": sparse["total_mapping_rows"],
        "mapped_fraction": sparse["mapped_fraction"],
        "mapping_sha256": sparse["mapping_sha256"],
        "target_set_hash": sparse["target_set_hash"],
    }
    return [{**shared, **dict(row)} for row in sparse["by_provider_and_arm"]]


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        matplotlib.rcParams["svg.hashsalt"] = "yher-persona-v2-analysis"
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise AnalysisContractError(
            "publication figures require the repository's matplotlib dependency"
        ) from exc
    return plt


def _save_figure(figure: Any, base: Path, plt: Any) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "YHer Persona-v2 analyzer"},
    )
    figure.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None, "Creator": "YHer Persona-v2 analyzer"},
    )
    plt.close(figure)
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    if not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") or b"<svg" not in svg.read_bytes()[:500]:
        raise AnalysisContractError("publication figure rendering failed")


def _assert_artists_do_not_overlap(figure: Any, *artists: Any) -> None:
    """Fail before export when named figure artists occupy the same pixels."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer=renderer) for artist in artists]
    for left_index, left in enumerate(boxes):
        for right in boxes[left_index + 1 :]:
            if left.overlaps(right):
                raise AnalysisContractError("publication figure artists overlap")


def _plot_controlled_composition(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    plt = _load_pyplot()
    providers = list(FROZEN_PROVIDERS)
    arms = ("deficit", "control")
    lookup = {
        (str(row["provider"]), str(row["response_arm"]), str(row["state"])): float(
            row["rate"]
        )
        for row in rows
    }
    figure, axis = plt.subplots(figsize=(10.8, 5.9))
    figure.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.27)
    colors = {
        "correct_answer": "#237a57",
        "incorrect_answer": "#c44e52",
        "abstention": "#e3a018",
        "technical_or_schema_failure": "#777777",
    }
    x_values = np.arange(len(providers) * len(arms), dtype=float)
    bottom = np.zeros(len(x_values), dtype=float)
    for state in CONTROLLED_STATES:
        values = np.asarray(
            [
                lookup[(provider, arm, state)]
                for provider in providers
                for arm in arms
            ],
            dtype=float,
        )
        axis.bar(
            x_values,
            values,
            bottom=bottom,
            color=colors[state],
            width=0.78,
            label=state.replace("_", " "),
        )
        bottom += values
    axis.set_xticks(
        x_values,
        [
            f"{provider}\n{arm}"
            for provider in providers
            for arm in ("D", "C")
        ],
        fontsize=8.5,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Share of expected controlled tasks")
    axis.set_title("Controlled response composition on the frozen expected denominator")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
    )
    footer = figure.text(
        0.5,
        0.018,
        "D = deficit; C = control. Technical/schema failures remain in the denominator.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    _assert_artists_do_not_overlap(figure, legend, footer)
    _save_figure(figure, output, plt)


def _plot_blind_agreement(
    rows: Sequence[Mapping[str, Any]],
    providers: Sequence[str],
    output: Path,
) -> None:
    plt = _load_pyplot()
    names = list(providers)
    if len(names) < 2:
        figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.5,
            0.56,
            "Pairwise agreement not estimable",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
        )
        axis.text(
            0.5,
            0.43,
            "Fewer than two providers passed the frozen blind schema gate.",
            ha="center",
            va="center",
            fontsize=10,
            color="#444444",
        )
        _save_figure(figure, output, plt)
        return
    index = {provider: offset for offset, provider in enumerate(names)}
    matrix = np.eye(len(names), dtype=float)
    denominators = np.zeros_like(matrix, dtype=int)
    for row in rows:
        left = index[str(row["provider_left"])]
        right = index[str(row["provider_right"])]
        matrix[left, right] = matrix[right, left] = float(row["exact_agreement"])
        denominators[left, right] = denominators[right, left] = int(row["denominator"])
    figure, axis = plt.subplots(figsize=(7.4, 6.3), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_title("Blind terminal exact agreement (NC retained)")
    for left in range(len(names)):
        for right in range(len(names)):
            label = "1.00" if left == right else f"{matrix[left, right]:.2f}\nn={denominators[left, right]}"
            axis.text(
                right,
                left,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[left, right] < 0.65 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Exact agreement")
    figure.text(
        0.01,
        0.01,
        "Providers above the frozen >50% invalid-schema threshold are excluded from this matrix.",
        fontsize=8,
        color="#444444",
    )
    _save_figure(figure, output, plt)


def _plot_stability(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    plt = _load_pyplot()
    providers = [str(row["provider"]) for row in rows]
    display_names = [
        f"{row['provider']}\n(excluded)"
        if row["excluded_from_blind_aggregate"]
        else str(row["provider"])
        for row in rows
    ]
    answer = [
        float(row["answer_agreement"])
        if row["answer_agreement"] is not None
        else np.nan
        for row in rows
    ]
    canonical = [
        float(row["canonical_complete_pair_stability"])
        if row["canonical_complete_pair_stability"] is not None
        else np.nan
        for row in rows
    ]
    x_values = np.arange(len(providers), dtype=float)
    figure, axis = plt.subplots(figsize=(9.2, 5.5))
    figure.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.25)
    for index, row in enumerate(rows):
        if row["excluded_from_blind_aggregate"]:
            axis.axvspan(index - 0.46, index + 0.46, color="#eeeeee", zorder=0)
    axis.bar(x_values - 0.18, answer, width=0.36, color="#2878b5", label="Answer category")
    axis.bar(
        x_values + 0.18,
        canonical,
        width=0.36,
        color="#e07a32",
        label="Canonical output | complete pair",
    )
    for index, value in enumerate(canonical):
        if not math.isfinite(value):
            axis.text(
                x_values[index] + 0.18,
                0.025,
                "NA",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                color="#9a4d18",
            )
    for index, value in enumerate(answer):
        if not math.isfinite(value):
            axis.text(
                x_values[index] - 0.18,
                0.025,
                "NA",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                color="#1b527c",
            )
    axis.set_xticks(x_values, display_names)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Stability fraction")
    axis.set_title("Frozen terminal-repeat output stability")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.055),
    )
    footer = figure.text(
        0.5,
        0.015,
        "Answer denominator is the frozen repeat roster; canonical equality is conditional on two complete parses.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    _assert_artists_do_not_overlap(figure, legend, footer)
    _save_figure(figure, output, plt)


def validate_judge_run_execution_snapshot(
    snapshot_manifest: Mapping[str, Any] | str | Path,
    *,
    snapshot_root: str | Path,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Replay a byte-complete W3 snapshot of the single finalized judge run."""

    supplied_root = Path(snapshot_root).expanduser()
    if supplied_root.is_symlink():
        raise AnalysisContractError("judge snapshot root cannot be a symlink")
    try:
        root = supplied_root.resolve(strict=True)
    except OSError as exc:
        raise AnalysisContractError("judge snapshot root cannot resolve") from exc
    if not root.is_dir():
        raise AnalysisContractError("judge snapshot root is not a directory")
    manifest_path = root / "snapshot_manifest.json"
    if isinstance(snapshot_manifest, Mapping):
        value = dict(snapshot_manifest)
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or _strict_json(manifest_path, "judge snapshot manifest") != value
        ):
            raise AnalysisContractError(
                "judge snapshot manifest differs from its fixed on-disk path"
            )
    else:
        supplied_manifest = Path(snapshot_manifest).expanduser()
        if supplied_manifest.is_symlink():
            raise AnalysisContractError("judge snapshot manifest cannot be a symlink")
        try:
            resolved_manifest = supplied_manifest.resolve(strict=True)
        except OSError as exc:
            raise AnalysisContractError("judge snapshot manifest cannot resolve") from exc
        if resolved_manifest != manifest_path.resolve(strict=False):
            raise AnalysisContractError(
                "judge snapshot manifest must use snapshot_manifest.json"
            )
        value = _strict_json(resolved_manifest, "judge snapshot manifest")
    advertised = value.get("snapshot_manifest_sha256")
    payload = dict(value)
    payload.pop("snapshot_manifest_sha256", None)
    expected_fields = {
        "schema_version",
        "simulated",
        "run_id",
        "case_manifest_sha256",
        "source_judge_run_evidence_receipt_sha256",
        "family_slots",
        "files",
        "file_count",
        "file_set_sha256",
        "directories",
        "directory_count",
        "directory_set_sha256",
        "snapshot_manifest_sha256",
    }
    files = value.get("files")
    directories = value.get("directories")
    if (
        set(value) != expected_fields
        or value.get("schema_version")
        != "yher.llm_sim_v2.judge_run_execution_snapshot_manifest.v1"
        or value.get("simulated") is not True
        or value.get("run_id") != RUN_ID
        or advertised != _canonical_sha(payload)
        or not isinstance(files, list)
        or not isinstance(directories, list)
        or value.get("file_count") != len(files)
        or value.get("file_set_sha256") != _canonical_sha(files)
        or value.get("directory_count") != len(directories)
        or value.get("directory_set_sha256") != _canonical_sha(directories)
        or directories != sorted(set(directories))
        or not directories
        or directories[0] != "run"
    ):
        raise AnalysisContractError("judge run snapshot manifest is invalid")
    file_index: dict[str, Mapping[str, Any]] = {}
    for row_value in files:
        row = _mapping(row_value, "judge snapshot file row")
        relative_value = row.get("path")
        if (
            set(row) != {"path", "sha256", "size"}
            or not isinstance(relative_value, str)
            or relative_value in file_index
            or not relative_value.startswith("run/")
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256")))
            or not isinstance(row.get("size"), int)
            or isinstance(row.get("size"), bool)
            or int(row["size"]) < 0
        ):
            raise AnalysisContractError("judge snapshot file binding is invalid")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise AnalysisContractError("judge snapshot file path is unsafe")
        file_index[relative_value] = row
    if list(file_index) != sorted(file_index):
        raise AnalysisContractError("judge snapshot files must be canonically ordered")
    for directory_value in directories:
        if not isinstance(directory_value, str):
            raise AnalysisContractError("judge snapshot directory binding is invalid")
        relative = Path(directory_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or (directory_value != "run" and not directory_value.startswith("run/"))
        ):
            raise AnalysisContractError("judge snapshot directory path is unsafe")

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for current_value, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_value)
        if current != root:
            actual_directories.add(current.relative_to(root).as_posix())
        for name in directory_names:
            directory = current / name
            if directory.is_symlink() or not directory.is_dir():
                raise AnalysisContractError(
                    "judge snapshot contains an unsafe directory"
                )
        for name in file_names:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise AnalysisContractError("judge snapshot contains an unsafe entry")
            actual_files.add(path.relative_to(root).as_posix())
    if (
        actual_files != set(file_index) | {"snapshot_manifest.json"}
        or actual_directories != set(directories)
    ):
        raise AnalysisContractError(
            "judge snapshot contains an unbound or missing file or directory set"
        )
    for relative, row in file_index.items():
        data = (root / relative).read_bytes()
        if (
            row.get("sha256") != hashlib.sha256(data).hexdigest()
            or row.get("size") != len(data)
        ):
            raise AnalysisContractError(
                f"judge snapshot file bytes drifted: {relative}"
            )

    run_root = root / "run"
    case_path = run_root / "case_manifest.json"
    receipt_path = run_root / "judge_run_evidence_receipt.json"
    case_manifest = _strict_json(case_path, "snapshotted judge case manifest")
    receipt = _strict_json(receipt_path, "snapshotted judge run receipt")
    try:
        from .judge_execution import (
            JudgeExecutionError,
            validate_judge_run_evidence_receipt,
        )

        validated = validate_judge_run_evidence_receipt(
            receipt,
            case_manifest=case_manifest,
            output_root=run_root,
            allow_fixture=allow_fixture,
        )
    except JudgeExecutionError as exc:
        raise AnalysisContractError(
            f"snapshotted judge run cannot replay: {exc}"
        ) from exc
    if (
        validated != receipt
        or value.get("case_manifest_sha256")
        != case_manifest.get("case_manifest_sha256")
        or value.get("source_judge_run_evidence_receipt_sha256")
        != receipt.get("judge_run_evidence_receipt_sha256")
        or value.get("family_slots") != receipt.get("family_slots")
    ):
        raise AnalysisContractError(
            "judge snapshot manifest differs from the replayed run evidence"
        )
    return value


def _stage_judge_execution_snapshots(
    *,
    staging: Path,
    judge_artifact_sources: Mapping[str, str],
    input_artifact_manifest: Mapping[str, Any],
    judge_result_manifests: Mapping[str, Mapping[str, Any] | None] | None = None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Copy the entire finalized judge run under one replayable snapshot."""

    del judge_result_manifests
    input_files = input_artifact_manifest.get("files")
    if not isinstance(input_files, list):
        raise AnalysisContractError("judge snapshot input manifest is invalid")
    input_index = {
        str(row.get("path")): row
        for row in input_files
        if isinstance(row, Mapping)
        and str(row.get("path") or "").startswith("judge-results/")
    }
    if not judge_artifact_sources:
        if input_index:
            raise AnalysisContractError(
                "judge snapshot inputs exist without artifact sources"
            )
        return {}
    if set(judge_artifact_sources) != set(input_index):
        raise AnalysisContractError(
            "judge snapshot source roster differs from loader-bound inputs"
        )
    case_key = "judge-results/case_manifest.json"
    if case_key not in judge_artifact_sources:
        raise AnalysisContractError("judge snapshot lacks the run case manifest")
    case_source = Path(judge_artifact_sources[case_key])
    if not case_source.is_absolute() or case_source.is_symlink():
        raise AnalysisContractError("judge snapshot case source is unsafe")
    try:
        source_root = case_source.resolve(strict=True).parent
    except OSError as exc:
        raise AnalysisContractError("judge snapshot source root cannot resolve") from exc

    source_bytes: dict[str, bytes] = {}
    for source_key, source_value in sorted(judge_artifact_sources.items()):
        relative_value = source_key.removeprefix("judge-results/")
        relative = Path(relative_value)
        source = Path(source_value)
        expected_source = source_root / relative
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or not source.is_absolute()
            or source.is_symlink()
        ):
            raise AnalysisContractError("judge snapshot source path is unsafe")
        try:
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise AnalysisContractError("judge snapshot source is missing") from exc
        if resolved_source != expected_source.resolve(strict=False) or not resolved_source.is_file():
            raise AnalysisContractError(
                "judge snapshot source is outside the single finalized run root"
            )
        data = resolved_source.read_bytes()
        binding = _mapping(input_index[source_key], "judge snapshot input binding")
        if (
            binding.get("sha256") != hashlib.sha256(data).hexdigest()
            or binding.get("size") != len(data)
        ):
            raise AnalysisContractError(
                f"judge snapshot source changed after input binding: {source_key}"
            )
        source_bytes[relative_value] = data

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for current_value, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        current = Path(current_value)
        if current != source_root:
            observed_directories.add(current.relative_to(source_root).as_posix())
        for name in directory_names:
            directory = current / name
            if directory.is_symlink() or not directory.is_dir():
                raise AnalysisContractError("judge source run contains a symlink")
        for name in file_names:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise AnalysisContractError("judge source run contains an unsafe entry")
            observed_files.add(path.relative_to(source_root).as_posix())
    if observed_files != set(source_bytes):
        raise AnalysisContractError(
            "judge snapshot source roster is not the complete finalized run root"
        )
    case_manifest = json.loads(source_bytes["case_manifest.json"])
    receipt = json.loads(source_bytes["judge_run_evidence_receipt.json"])
    try:
        from .judge_execution import (
            JudgeExecutionError,
            validate_judge_run_evidence_receipt,
        )

        validate_judge_run_evidence_receipt(
            receipt,
            case_manifest=case_manifest,
            output_root=source_root,
            allow_fixture=allow_fixture,
        )
    except (JudgeExecutionError, TypeError, ValueError) as exc:
        raise AnalysisContractError(
            f"judge snapshot source run cannot replay: {exc}"
        ) from exc

    files = [
        {
            "path": f"run/{relative}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        for relative, data in sorted(source_bytes.items())
    ]
    directories = ["run", *(f"run/{value}" for value in sorted(observed_directories))]
    snapshot: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.judge_run_execution_snapshot_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "case_manifest_sha256": case_manifest["case_manifest_sha256"],
        "source_judge_run_evidence_receipt_sha256": receipt[
            "judge_run_evidence_receipt_sha256"
        ],
        "family_slots": receipt["family_slots"],
        "files": files,
        "file_count": len(files),
        "file_set_sha256": _canonical_sha(files),
        "directories": directories,
        "directory_count": len(directories),
        "directory_set_sha256": _canonical_sha(directories),
    }
    snapshot["snapshot_manifest_sha256"] = _canonical_sha(snapshot)
    destination = staging / "judge-snapshots"
    if destination.exists():
        raise AnalysisContractError("judge snapshot destination already exists")
    try:
        for relative, data in sorted(source_bytes.items()):
            target = destination / "run" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        _write_json(destination / "snapshot_manifest.json", snapshot)
        validate_judge_run_execution_snapshot(
            destination / "snapshot_manifest.json",
            snapshot_root=destination,
            allow_fixture=allow_fixture,
        )
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    return snapshot


def _validate_publication_cost_accounting(result: Mapping[str, Any]) -> None:
    cost = _mapping(result.get("cost_accounting"), "formal publication cost accounting")
    rows = cost.get("provider_phase")
    if (
        cost.get("schema_version") != "yher.llm_sim_v2.cost_accounting.v1"
        or cost.get("currency") != "CNY"
        or cost.get("source")
        != "reconciled_prior_carried_forward_and_immutable_attempt_ledgers"
        or not isinstance(rows, list)
    ):
        raise AnalysisContractError("formal publication cost accounting is invalid")

    integer_fields = (
        "record_count",
        "requests",
        "responses",
        "retries",
        "input_tokens",
        "output_tokens",
        "unknown_attempt_count",
    )
    money_fields = (
        "known_cost_yuan",
        "unknown_cost_reserve_yuan",
        "accounted_cost_yuan",
    )
    normalized_rows: list[dict[str, Any]] = []
    for value in rows:
        row = dict(_mapping(value, "formal publication provider-phase cost row"))
        if row.get("phase") not in {"pilot", "main"} or row.get(
            "provider"
        ) not in FROZEN_PROVIDERS:
            raise AnalysisContractError(
                "formal publication provider-phase cost identity is invalid"
            )
        if any(
            not isinstance(row.get(field), int)
            or isinstance(row.get(field), bool)
            or int(row[field]) < 0
            for field in integer_fields
        ):
            raise AnalysisContractError(
                "formal publication provider-phase counters are invalid"
            )
        amounts: dict[str, float] = {}
        for field in money_fields:
            raw = row.get(field)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise AnalysisContractError(
                    "formal publication provider-phase cost is invalid"
                )
            amount = float(raw)
            if not math.isfinite(amount) or amount < 0.0:
                raise AnalysisContractError(
                    "formal publication provider-phase cost is invalid"
                )
            amounts[field] = amount
        if (
            not math.isclose(
                amounts["accounted_cost_yuan"],
                amounts["known_cost_yuan"]
                + amounts["unknown_cost_reserve_yuan"],
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            or row.get("needs_user") is not (
                int(row["unknown_attempt_count"]) > 0
            )
        ):
            raise AnalysisContractError(
                "formal publication provider-phase cost does not reconcile"
            )
        normalized_rows.append(row)

    identities = [(str(row["phase"]), str(row["provider"])) for row in normalized_rows]
    if (
        len(identities) != len(set(identities))
        or identities != sorted(identities)
        or {
            provider
            for phase, provider in identities
            if phase == "main"
        }
        != set(FROZEN_PROVIDERS)
    ):
        raise AnalysisContractError(
            "formal publication provider-phase rows are incomplete or unordered"
        )

    judge_rows_value = cost.get("judge")
    if not isinstance(judge_rows_value, list):
        raise AnalysisContractError("formal publication judge costs are invalid")
    judge_integer_fields = (
        "request_count",
        "retry_count",
        "transport_error_count",
        "schema_error_count",
        "content_retry_count",
        "input_tokens",
        "output_tokens",
    )
    normalized_judge_rows: list[dict[str, Any]] = []
    for value in judge_rows_value:
        row = dict(_mapping(value, "formal publication judge cost row"))
        if (
            row.get("judge") not in {"claude", "gpt"}
            or row.get("status") not in {"complete", "failed"}
            or not isinstance(row.get("execution_id"), str)
            or not isinstance(row.get("requested_model"), str)
            or not isinstance(row.get("transport_reported_models"), list)
            or any(
                not isinstance(model, str) or not model
                for model in row.get("transport_reported_models", ())
            )
            or row.get("transport")
            != {"claude": "claude_cli", "gpt": "codex_cli"}[row["judge"]]
            or not isinstance(row.get("execution_receipt_sha256"), str)
            or len(row["execution_receipt_sha256"]) != 64
            or any(
                not isinstance(row.get(field), int)
                or isinstance(row.get(field), bool)
                or int(row[field]) < 0
                for field in judge_integer_fields
            )
        ):
            raise AnalysisContractError("formal publication judge cost row is invalid")
        judge_amounts: dict[str, float] = {}
        for field in money_fields:
            raw = row.get(field)
            if (
                not isinstance(raw, (int, float))
                or isinstance(raw, bool)
                or not math.isfinite(float(raw))
                or float(raw) < 0.0
            ):
                raise AnalysisContractError(
                    "formal publication judge cost row is invalid"
                )
            judge_amounts[field] = float(raw)
        if not math.isclose(
            judge_amounts["accounted_cost_yuan"],
            judge_amounts["known_cost_yuan"]
            + judge_amounts["unknown_cost_reserve_yuan"],
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise AnalysisContractError(
                "formal publication judge cost does not reconcile"
            )
        normalized_judge_rows.append(row)
    judge_identities = [str(row["judge"]) for row in normalized_judge_rows]
    if (
        len(judge_identities) != len(set(judge_identities))
        or judge_identities != [
            judge for judge in ("claude", "gpt") if judge in judge_identities
        ]
    ):
        raise AnalysisContractError(
            "formal publication judge cost identities are duplicated or unordered"
        )
    judge_known = round(
        sum(float(row["known_cost_yuan"]) for row in normalized_judge_rows), 8
    )
    judge_reserve = round(
        sum(
            float(row["unknown_cost_reserve_yuan"])
            for row in normalized_judge_rows
        ),
        8,
    )
    judge_accounted = round(
        sum(float(row["accounted_cost_yuan"]) for row in normalized_judge_rows),
        8,
    )

    immutable = _mapping(
        cost.get("immutable_record_totals"), "formal immutable record totals"
    )
    row_known = round(sum(float(row["known_cost_yuan"]) for row in normalized_rows), 8)
    row_reserve = round(
        sum(float(row["unknown_cost_reserve_yuan"]) for row in normalized_rows), 8
    )
    row_accounted = round(
        sum(float(row["accounted_cost_yuan"]) for row in normalized_rows), 8
    )
    row_unknown_attempts = sum(
        int(row["unknown_attempt_count"]) for row in normalized_rows
    )
    try:
        immutable_matches = (
            math.isclose(
                float(immutable["known_cost_yuan"]),
                row_known,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(immutable["unknown_cost_reserve_yuan"]),
                row_reserve,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(immutable["accounted_cost_yuan"]),
                row_accounted,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and immutable["unknown_attempt_count"] == row_unknown_attempts
        )
        collection_known = round(
            float(cost["prior_known_cost_yuan"])
            + float(cost["carried_forward_known_cost_yuan"])
            + row_known,
            8,
        )
        collection_reserve = round(
            float(cost["prior_ambiguity_reserve_yuan"])
            + float(cost["carried_forward_unknown_reserve_yuan"])
            + row_reserve,
            8,
        )
        collection_accounted = round(
            float(cost["prior_accounted_cost_yuan"])
            + float(cost["carried_forward_accounted_cost_yuan"])
            + row_accounted,
            8,
        )
        expected_known = round(collection_known + judge_known, 8)
        expected_reserve = round(collection_reserve + judge_reserve, 8)
        expected_accounted = round(collection_accounted + judge_accounted, 8)
        cumulative_matches = (
            math.isclose(
                float(cost["collection_total_known_cost_yuan"]),
                collection_known,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(cost["collection_total_unknown_reserve_yuan"]),
                collection_reserve,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(cost["collection_total_accounted_cost_yuan"]),
                collection_accounted,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(cost["judge_total_known_cost_yuan"]),
                judge_known,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(cost["judge_total_unknown_reserve_yuan"]),
                judge_reserve,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(cost["judge_total_accounted_cost_yuan"]),
                judge_accounted,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and
            math.isclose(
                float(cost["total_known_cost_yuan"]),
                expected_known,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(cost["total_unknown_reserve_yuan"]),
                expected_reserve,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(cost["total_accounted_cost_yuan"]),
                expected_accounted,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                expected_accounted,
                expected_known + expected_reserve,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and float(cost["soft_warning_yuan"]) == 300.0
            and float(cost["hard_fuse_yuan"]) == 450.0
            and expected_accounted < 450.0
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisContractError(
            "formal publication cumulative cost inputs are invalid"
        ) from exc
    if not immutable_matches or not cumulative_matches:
        raise AnalysisContractError(
            "formal publication cumulative cost totals do not reconcile"
        )

    manifest = _mapping(
        cost.get("cost_reconciliation_artifact_manifest"),
        "formal publication cost reconciliation artifact manifest",
    )
    files = manifest.get("files")
    if (
        manifest.get("schema_version")
        != "yher.llm_sim_v2.cost_reconciliation_artifact_manifest.v1"
        or manifest.get("simulated") is not True
        or manifest.get("run_id") != RUN_ID
        or manifest.get("metric_input") is not False
        or not isinstance(files, list)
        or manifest.get("file_count") != len(files)
        or manifest.get("file_set_sha256") != _canonical_sha(files)
        or any(
            not isinstance(row, Mapping)
            or row.get("included_in_metrics") is not False
            for row in files
        )
    ):
        raise AnalysisContractError(
            "formal publication cost reconciliation manifest is invalid"
        )


def write_analysis_outputs(
    result: Mapping[str, Any],
    result_dir: str | Path,
    *,
    input_artifact_manifest: Mapping[str, Any] | None = None,
    judge_artifact_sources: Mapping[str, str] | None = None,
    _formal_publication_proof: _FormalPublicationProof | None = None,
) -> dict[str, Any]:
    """Write one complete machine-readable and publication-figure package."""

    if not _verify_formal_publication_proof(
        _formal_publication_proof,
        result=result,
        input_artifact_manifest=input_artifact_manifest,
    ):
        raise AnalysisContractError("publication requires an exact formal loader proof")

    if result.get("analysis_mode") == "synthetic_nonformal":
        raise AnalysisContractError(
            "synthetic nonformal analysis cannot produce publication outputs"
        )
    if (
        result.get("schema_version") != "yher.llm_sim_v2.analysis_results.v1"
        or result.get("run_id") != RUN_ID
        or result.get("analysis_population") != "main"
        or result.get("analysis_mode") != "formal_main"
        or result.get("formal_analysis_eligible") is not True
        or result.get("publication_output_eligible") is not True
        or result.get("independent_cluster_count") != 50
    ):
        raise AnalysisContractError("analysis result envelope is not formal main")
    if input_artifact_manifest is None:
        raise AnalysisContractError(
            "formal publication output requires the loader-bound input artifact manifest"
        )
    input_manifest = _mapping(
        input_artifact_manifest, "input artifact manifest"
    )
    if not isinstance(judge_artifact_sources, Mapping):
        raise AnalysisContractError(
            "formal publication requires loader-bound judge artifact sources"
        )
    input_files = input_manifest.get("files")
    if (
        input_manifest.get("schema_version")
        != "yher.llm_sim_v2.analysis_input_artifact_manifest.v1"
        or input_manifest.get("run_id") != RUN_ID
        or input_manifest.get("analysis_population") != "main"
        or not isinstance(input_files, list)
        or input_manifest.get("input_file_count") != len(input_files)
        or input_manifest.get("record_file_count")
        != sum(
            1
            for row in input_files
            if isinstance(row, Mapping)
            and str(row.get("path") or "").startswith("main/records/")
        )
        or input_manifest.get("input_file_set_sha256") != _canonical_sha(input_files)
    ):
        raise AnalysisContractError("input artifact manifest envelope is invalid")
    expected_input_binding = {
        "input_file_count": input_manifest["input_file_count"],
        "record_file_count": input_manifest["record_file_count"],
        "input_file_set_sha256": input_manifest["input_file_set_sha256"],
        "input_artifact_manifest_sha256": _canonical_sha(input_manifest),
    }
    if result.get("input_artifact_binding") != expected_input_binding:
        raise AnalysisContractError(
            "formal result input artifact binding does not match the supplied manifest"
        )
    _validate_publication_cost_accounting(result)
    destination = Path(result_dir).expanduser().resolve(strict=False)
    if destination.exists():
        raise AnalysisContractError("result directory already exists; refusing to overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        composition = _composition_rows(result)
        effects = _effect_rows(result)
        agreement = _agreement_rows(result)
        stability = _stability_rows(result)
        mapping_rows = _mapping_rows(result)
        cost_rows = [
            dict(row)
            for row in _mapping(
                result.get("cost_accounting"), "cost accounting"
            ).get("provider_phase", ())
        ]
        cost_reconciliation_manifest = _mapping(
            _mapping(result.get("cost_accounting"), "cost accounting").get(
                "cost_reconciliation_artifact_manifest"
            ),
            "cost reconciliation artifact manifest",
        )
        judge = _mapping(result.get("judge_adjudication"), "judge adjudication")
        case_manifest = _mapping(judge.get("case_manifest"), "judge case manifest")
        judge_analysis = _mapping(judge.get("analysis"), "judge analysis")
        judge_category_rows: list[dict[str, Any]] = []
        for judge_name, counts_value in sorted(
            _mapping(
                judge_analysis.get("category_counts"), "judge category counts"
            ).items()
        ):
            counts = _mapping(counts_value, "judge category count row")
            for category_type, values in (
                ("label", counts.get("labels")),
                ("error_category", counts.get("error_categories")),
            ):
                for category, count in sorted(
                    _mapping(values, "judge category values").items()
                ):
                    judge_category_rows.append(
                        {
                            "judge": judge_name,
                            "category_type": category_type,
                            "category": category,
                            "count": count,
                        }
                    )
        lifecycle_rows = [dict(row) for row in result["provider_lifecycle"]]
        for row in lifecycle_rows:
            row["status_counts"] = dict(row["status_counts"])

        _write_csv(
            staging / "provider_lifecycle.csv",
            (
                "provider",
                "provider_lifecycle",
                "requested_model",
                "returned_models",
                "observed_model_ids_match_request",
                "expected_count",
                "present_count",
                "missing_count",
                "status_counts",
                "missing_task_ids",
            ),
            lifecycle_rows,
        )
        composition_fields = (
            "provider",
            "response_arm",
            "state",
            "count",
            "expected_denominator",
            "rate",
            "conditional_answer_accuracy",
            "conditional_answer_denominator",
        )
        _write_csv(staging / "controlled_composition.csv", composition_fields, composition)
        _write_csv(
            staging / "controlled_paired_effects.csv",
            (
                "metric_id",
                "orientation",
                "scope",
                "provider",
                "estimate",
                "ci95_low",
                "ci95_high",
                "paired_persona_denominator",
                "paired_persona_denominators",
                "paired_persona_denominator_range",
                "bootstrap_seed",
                "bootstrap_resamples",
                "undefined_resamples",
            ),
            effects,
        )
        agreement_fields = (
            "provider_left",
            "provider_right",
            "exact_agreement_numerator",
            "denominator",
            "exact_agreement",
            "cohen_kappa",
            "exact_agreement_ci95",
            "exact_agreement_bootstrap",
        )
        _write_csv(staging / "blind_agreement.csv", agreement_fields, agreement)
        stability_fields = tuple(stability[0])
        _write_csv(staging / "blind_stability.csv", stability_fields, stability)
        mapping_fields = tuple(mapping_rows[0])
        _write_csv(staging / "sparse_mapping_descriptive.csv", mapping_fields, mapping_rows)
        _write_csv(
            staging / "cost_by_provider_phase.csv",
            (
                "phase",
                "provider",
                "record_count",
                "requests",
                "responses",
                "retries",
                "input_tokens",
                "output_tokens",
                "known_cost_yuan",
                "unknown_cost_reserve_yuan",
                "accounted_cost_yuan",
                "unknown_attempt_count",
                "needs_user",
            ),
            cost_rows,
        )
        _write_json(
            staging / "cost_reconciliation_artifact_manifest.json",
            cost_reconciliation_manifest,
        )
        _write_json(staging / "judge/case_manifest.json", case_manifest)
        shared_judge_bytes = judge_input_bytes(case_manifest)
        if hashlib.sha256(shared_judge_bytes).hexdigest() != case_manifest.get(
            "shared_input_sha256"
        ):
            raise AnalysisContractError("judge shared input hash drifted before write")
        for judge_name in ("claude", "gpt"):
            path = staging / f"judge/{judge_name}_input.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(shared_judge_bytes)
        _write_json(staging / "judge/judge_analysis.json", judge_analysis)
        _write_csv(
            staging / "judge/judge_category_counts.csv",
            ("judge", "category_type", "category", "count"),
            judge_category_rows,
        )
        _write_json(
            staging / "judge/judge_label_disagreements.json",
            list(judge_analysis.get("label_disagreement_examples") or ()),
        )
        _write_json(
            staging / "judge/judge_error_category_disagreements.json",
            list(
                judge_analysis.get("error_category_disagreement_examples") or ()
            ),
        )
        result_manifests = _mapping(
            judge.get("result_manifests"), "judge result manifests"
        )
        _stage_judge_execution_snapshots(
            staging=staging,
            judge_result_manifests={
                judge_name: (
                    _mapping(value, f"{judge_name} judge result manifest")
                    if value is not None
                    else None
                )
                for judge_name, value in result_manifests.items()
            },
            judge_artifact_sources=judge_artifact_sources,
            input_artifact_manifest=input_manifest,
        )
        _write_csv(
            staging / "figure_data/controlled_composition.csv",
            composition_fields,
            composition,
        )
        _write_csv(
            staging / "figure_data/blind_agreement.csv", agreement_fields, agreement
        )
        _write_csv(
            staging / "figure_data/blind_stability.csv", stability_fields, stability
        )

        _plot_controlled_composition(
            composition, staging / "figures/controlled_composition"
        )
        _plot_blind_agreement(
            agreement,
            result["blind"]["eligible_providers"],
            staging / "figures/blind_terminal_agreement",
        )
        _plot_stability(stability, staging / "figures/blind_output_stability")

        payload = copy.deepcopy(dict(result))
        payload["outputs"] = {
            "machine_json": True,
            "machine_csv_tables": 8,
            "figure_data_machine_readable": True,
            "publication_figures": 3,
            "publication_formats": ["png_300_dpi", "svg"],
            "judge_case_export": True,
            "judge_shared_input_sha256": case_manifest["shared_input_sha256"],
        }
        _write_json(staging / "input_artifact_manifest.json", input_manifest)
        _write_json(staging / "analysis_results.json", payload)
        artifacts = []
        for path in sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda value: value.relative_to(staging).as_posix(),
        ):
            relative = path.relative_to(staging).as_posix()
            artifacts.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": path.stat().st_size,
                }
            )
        sparse = _mapping(result.get("sparse_mapping_descriptive"), "sparse mapping")
        manifest = {
            "schema_version": "yher.llm_sim_v2.analysis_artifact_manifest.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "analysis_population": "main",
            "target_set_hash": sparse["target_set_hash"],
            "runtime_task_manifest_sha256": result["input_proof"][
                "runtime_task_manifest_sha256"
            ],
            "phase_provenance_sha256": result["input_proof"][
                "phase_provenance_sha256"
            ],
            "judge_case_manifest_sha256": case_manifest[
                "case_manifest_sha256"
            ],
            "judge_result_manifest_sha256": judge_analysis[
                "result_manifest_sha256"
            ],
            "artifacts": artifacts,
            "artifact_set_sha256": _canonical_sha(artifacts),
        }
        _write_json(staging / "artifact_manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            if key in output:
                raise AnalysisContractError(f"{label} contains duplicate JSON keys")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise AnalysisContractError(
            f"{label} contains non-finite JSON constant: {value}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AnalysisContractError(f"{label} must be a JSON object")
    return value


def _load_judge_result_manifests(
    judge_results_dir: str | Path | None,
    *,
    repo_root: str | Path | None = None,
    allow_fixture: bool = False,
) -> tuple[
    dict[str, Mapping[str, Any] | None],
    list[tuple[str, Path]],
    dict[str, str],
    dict[str, Any] | None,
]:
    """Load one finalized judge run and its exact committed evidence anchor."""

    output: dict[str, Mapping[str, Any] | None] = {
        "claude": None,
        "gpt": None,
    }
    if judge_results_dir is None:
        return output, [], {}, None
    supplied_root = Path(judge_results_dir).expanduser()
    if supplied_root.is_symlink():
        raise AnalysisContractError("judge result root cannot be a symlink")
    try:
        root = supplied_root.resolve(strict=True)
    except OSError as exc:
        raise AnalysisContractError(
            f"judge result directory is missing or cannot resolve: {judge_results_dir}"
        ) from exc
    if not root.is_dir():
        raise AnalysisContractError("judge result path is not a directory")
    if repo_root is None:
        raise AnalysisContractError(
            "judge run requires the repository containing its committed fixed anchor"
        )
    supplied_repo = Path(repo_root).expanduser()
    if supplied_repo.is_symlink():
        raise AnalysisContractError("judge anchor repository cannot be a symlink")
    try:
        repo = supplied_repo.resolve(strict=True)
    except OSError as exc:
        raise AnalysisContractError("judge anchor repository cannot resolve") from exc
    if not repo.is_dir():
        raise AnalysisContractError("judge anchor repository is not a directory")

    case_path = root / "case_manifest.json"
    internal_receipt_path = root / "judge_run_evidence_receipt.json"
    budget_path = root / "budget_authority.json"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (case_path, internal_receipt_path, budget_path)
    ):
        raise AnalysisContractError(
            "judge run lacks its case, budget, or internal run evidence receipt"
        )
    case_manifest = _strict_json(case_path, "judge run case manifest")
    internal_receipt = _strict_json(
        internal_receipt_path, "internal judge run evidence receipt"
    )
    external_anchor_path, committed_anchor = _verify_committed_repo_file(
        repo, _JUDGE_RUN_RECEIPT_REL
    )
    if external_anchor_path.read_bytes() != internal_receipt_path.read_bytes():
        raise AnalysisContractError(
            "internal judge run receipt differs from the exact committed fixed anchor"
        )
    try:
        from .judge_execution import (
            JudgeExecutionError,
            validate_judge_run_evidence_receipt,
        )

        validated_run = validate_judge_run_evidence_receipt(
            internal_receipt,
            case_manifest=case_manifest,
            output_root=root,
            allow_fixture=allow_fixture,
        )
    except JudgeExecutionError as exc:
        raise AnalysisContractError(
            f"judge run evidence cannot replay the exact root: {exc}"
        ) from exc
    if validated_run != internal_receipt:
        raise AnalysisContractError("judge run evidence changed during validation")

    observed_files: dict[str, Path] = {}
    observed_directories: set[str] = set()
    for current_value, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_value)
        if current != root:
            observed_directories.add(current.relative_to(root).as_posix())
        for name in directory_names:
            directory = current / name
            if directory.is_symlink() or not directory.is_dir():
                raise AnalysisContractError("judge run tree contains an unsafe directory")
        for name in file_names:
            artifact = current / name
            if artifact.is_symlink() or not artifact.is_file():
                raise AnalysisContractError("judge run tree contains an unsafe entry")
            observed_files[artifact.relative_to(root).as_posix()] = artifact

    expected_files = {
        "case_manifest.json",
        "budget_authority.json",
        "judge_run_evidence_receipt.json",
    }
    expected_directories: set[str] = set()
    artifact_roots: dict[str, str] = {}
    family_receipts: dict[str, Mapping[str, Any]] = {}
    family_slots = _mapping(
        internal_receipt.get("family_slots"), "judge run family slots"
    )
    for judge in ("claude", "gpt"):
        slot = _mapping(family_slots.get(judge), f"{judge} judge run family slot")
        status = slot.get("status")
        receipt_relative = Path(str(slot.get("receipt_path") or ""))
        if (
            receipt_relative.is_absolute()
            or not receipt_relative.parts
            or ".." in receipt_relative.parts
        ):
            raise AnalysisContractError(f"{judge} judge slot receipt path is unsafe")
        receipt_path = root / receipt_relative
        expected_files.add(receipt_relative.as_posix())
        parent = receipt_relative.parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        family_receipt = _strict_json(
            receipt_path, f"{judge} judge family receipt"
        )
        family_receipts[judge] = family_receipt
        result_path = root / f"{judge}.json"
        if status in {"unavailable", "not_applicable_zero_cases"}:
            if result_path.exists() or slot.get("execution_id") is not None:
                raise AnalysisContractError(
                    f"{judge} disposition slot cannot contain an execution result"
                )
            continue
        if status not in {"complete", "failed"}:
            raise AnalysisContractError(f"{judge} judge run slot status is invalid")
        execution_id = slot.get("execution_id")
        expected_receipt_relative = (
            Path("executions")
            / judge
            / str(execution_id or "")
            / (
                "execution_receipt.json"
                if status == "complete"
                else "failed_execution_receipt.json"
            )
        )
        if receipt_relative != expected_receipt_relative:
            raise AnalysisContractError(
                f"{judge} judge execution path differs from its finalized family slot"
            )
        execution_root = receipt_path.parent
        normalized = _mapping(
            family_receipt.get("normalized_results"),
            f"{judge} normalized judge result binding",
        )
        artifact_bindings = [normalized]
        raw_artifacts = family_receipt.get("raw_artifacts")
        if not isinstance(raw_artifacts, list):
            raise AnalysisContractError(f"{judge} raw judge artifact list is invalid")
        artifact_bindings.extend(
            _mapping(row, f"{judge} raw judge artifact") for row in raw_artifacts
        )
        for binding in artifact_bindings:
            relative = Path(str(binding.get("path") or ""))
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
            ):
                raise AnalysisContractError(f"{judge} judge artifact path is unsafe")
            artifact_relative = receipt_relative.parent / relative
            expected_files.add(artifact_relative.as_posix())
            parent = artifact_relative.parent
            while parent.as_posix() != ".":
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        if status == "failed":
            if result_path.exists():
                raise AnalysisContractError(
                    f"{judge} failed execution cannot have a result manifest"
                )
            continue
        if result_path.is_symlink() or not result_path.is_file():
            raise AnalysisContractError(
                f"{judge} complete execution lacks its bound result manifest"
            )
        expected_files.add(f"{judge}.json")
        result_manifest = _strict_json(result_path, f"{judge} judge result manifest")
        if (
            result_manifest.get("schema_version")
            != "yher.llm_sim_v2.judge_result_manifest.v2"
            or result_manifest.get("simulated") is not True
            or result_manifest.get("run_id") != RUN_ID
            or result_manifest.get("judge") != judge
            or result_manifest.get("execution_receipt_path")
            != receipt_relative.as_posix()
            or result_manifest.get("execution_receipt") != family_receipt
        ):
            raise AnalysisContractError(
                f"{judge} judge result manifest envelope is invalid"
            )
        _verify_internal_digest(
            result_manifest,
            field="judge_result_manifest_sha256",
            label=f"{judge} judge result manifest",
        )
        artifact_roots[judge] = str(execution_root.resolve())
        output[judge] = result_manifest

    if set(observed_files) != expected_files or observed_directories != expected_directories:
        raise AnalysisContractError(
            "judge run contains an unbound or missing file or directory set"
        )
    paths = sorted(observed_files.items())
    run_evidence: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.formal_judge_run_evidence_binding.v1",
        "receipt": internal_receipt,
        "committed_anchor": committed_anchor,
        "family_receipts": family_receipts,
    }
    ingest_judge_results(
        case_manifest,
        output,
        judge_artifact_roots=artifact_roots,
        judge_run_evidence=run_evidence,
        allow_fixture=allow_fixture,
    )
    return output, paths, artifact_roots, run_evidence


def _judge_cost_accounting(
    judge_result_manifests: Mapping[str, Mapping[str, Any] | None],
    *,
    judge_run_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    integer_fields = (
        "request_count",
        "retry_count",
        "transport_error_count",
        "schema_error_count",
        "content_retry_count",
        "input_tokens",
        "output_tokens",
    )
    money_fields = (
        "known_cost_yuan",
        "unknown_cost_reserve_yuan",
        "accounted_cost_yuan",
    )
    run_slots: Mapping[str, Any] | None = None
    family_receipts: Mapping[str, Any] | None = None
    if judge_run_evidence is not None:
        run_receipt = _mapping(
            judge_run_evidence.get("receipt"), "judge cost run evidence receipt"
        )
        run_slots = _mapping(
            run_receipt.get("family_slots"), "judge cost family slots"
        )
        family_receipts = _mapping(
            judge_run_evidence.get("family_receipts"),
            "judge cost family receipts",
        )
    for judge in ("claude", "gpt"):
        result_manifest = judge_result_manifests.get(judge)
        status = "complete" if result_manifest is not None else None
        if run_slots is not None:
            slot = _mapping(run_slots.get(judge), f"{judge} judge cost family slot")
            status = str(slot.get("status") or "")
            if status not in {"complete", "failed"}:
                continue
            assert family_receipts is not None
            receipt = _mapping(
                family_receipts.get(judge), f"{judge} judge family receipt"
            )
            if status == "complete" and (
                result_manifest is None
                or result_manifest.get("execution_receipt") != receipt
            ):
                raise AnalysisContractError(
                    f"{judge} completed judge cost lacks its result receipt"
                )
            if status == "failed" and result_manifest is not None:
                raise AnalysisContractError(
                    f"{judge} failed judge cost cannot have a result manifest"
                )
        else:
            if result_manifest is None:
                continue
            receipt = _mapping(
                result_manifest.get("execution_receipt"),
                f"{judge} judge execution receipt",
            )
        identity = _mapping(
            receipt.get("identity"), f"{judge} judge execution identity"
        )
        accounting = _mapping(
            receipt.get("accounting"), f"{judge} judge execution accounting"
        )
        if any(
            not isinstance(accounting.get(field), int)
            or isinstance(accounting.get(field), bool)
            or int(accounting[field]) < 0
            for field in integer_fields
        ):
            raise AnalysisContractError(f"{judge} judge counters are invalid")
        amounts: dict[str, float] = {}
        for field in money_fields:
            value = accounting.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise AnalysisContractError(f"{judge} judge cost is invalid")
            amounts[field] = round(float(value), 8)
        if not math.isclose(
            amounts["accounted_cost_yuan"],
            amounts["known_cost_yuan"] + amounts["unknown_cost_reserve_yuan"],
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise AnalysisContractError(f"{judge} judge cost does not reconcile")
        rows.append(
            {
                "judge": judge,
                "status": status,
                "execution_id": identity.get("execution_id"),
                "requested_model": identity.get("requested_model"),
                "transport_reported_models": list(
                    identity.get("transport_reported_models") or ()
                ),
                "transport": identity.get("transport"),
                "execution_receipt_sha256": receipt.get(
                    "execution_receipt_sha256"
                    if status == "complete"
                    else "failed_execution_receipt_sha256"
                ),
                **{field: int(accounting[field]) for field in integer_fields},
                **amounts,
            }
        )
    return {
        "schema_version": "yher.llm_sim_v2.judge_cost_accounting.v1",
        "currency": "CNY",
        "rows": rows,
        "total_known_cost_yuan": round(
            sum(float(row["known_cost_yuan"]) for row in rows), 8
        ),
        "total_unknown_reserve_yuan": round(
            sum(float(row["unknown_cost_reserve_yuan"]) for row in rows), 8
        ),
        "total_accounted_cost_yuan": round(
            sum(float(row["accounted_cost_yuan"]) for row in rows), 8
        ),
    }


def _merge_judge_cost_accounting(
    collection_cost: Mapping[str, Any],
    judge_cost: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(collection_cost)
    try:
        collection_known = float(collection_cost["total_known_cost_yuan"])
        collection_reserve = float(collection_cost["total_unknown_reserve_yuan"])
        collection_accounted = float(collection_cost["total_accounted_cost_yuan"])
        judge_known = float(judge_cost["total_known_cost_yuan"])
        judge_reserve = float(judge_cost["total_unknown_reserve_yuan"])
        judge_accounted = float(judge_cost["total_accounted_cost_yuan"])
        hard_fuse = float(collection_cost["hard_fuse_yuan"])
        soft_warning = float(collection_cost["soft_warning_yuan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisContractError("judge cumulative cost inputs are invalid") from exc
    values = (
        collection_known,
        collection_reserve,
        collection_accounted,
        judge_known,
        judge_reserve,
        judge_accounted,
        hard_fuse,
        soft_warning,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise AnalysisContractError("judge cumulative cost inputs are invalid")
    if (
        not math.isclose(
            collection_accounted,
            collection_known + collection_reserve,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        or not math.isclose(
            judge_accounted,
            judge_known + judge_reserve,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        or hard_fuse != 450.0
        or soft_warning != 300.0
    ):
        raise AnalysisContractError("judge cumulative cost totals do not reconcile")
    total_known = round(collection_known + judge_known, 8)
    total_reserve = round(collection_reserve + judge_reserve, 8)
    total_accounted = round(collection_accounted + judge_accounted, 8)
    if total_accounted >= hard_fuse:
        raise AnalysisContractError(
            "judge cost reaches the CNY 450 hard fuse; user authorization is required"
        )
    reasons = list(collection_cost.get("needs_user_reasons") or ())
    if judge_reserve > 0.0 and "unknown_judge_billing_reserved" not in reasons:
        reasons.append("unknown_judge_billing_reserved")
    rows = judge_cost.get("rows")
    if not isinstance(rows, list):
        raise AnalysisContractError("judge cost rows are invalid")
    merged.update(
        {
            "collection_total_known_cost_yuan": round(collection_known, 8),
            "collection_total_unknown_reserve_yuan": round(
                collection_reserve, 8
            ),
            "collection_total_accounted_cost_yuan": round(
                collection_accounted, 8
            ),
            "judge": [dict(_mapping(row, "judge cost row")) for row in rows],
            "judge_total_known_cost_yuan": round(judge_known, 8),
            "judge_total_unknown_reserve_yuan": round(judge_reserve, 8),
            "judge_total_accounted_cost_yuan": round(judge_accounted, 8),
            "total_known_cost_yuan": total_known,
            "total_unknown_reserve_yuan": total_reserve,
            "total_accounted_cost_yuan": total_accounted,
            "needs_user": bool(collection_cost.get("needs_user"))
            or judge_reserve > 0.0,
            "needs_user_reasons": reasons,
        }
    )
    return merged


def _main_store_root(output_base: str | Path) -> Path:
    from .store import V2Store

    return V2Store(output_base, phase="main").root


def _load_expected_tasks(
    repo_root: Path, runtime_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Reconstruct task metadata from the frozen contract, never the record glob."""

    from .runner import enumerate_tasks, load_runtime_contract

    contract = load_runtime_contract(repo_root)
    if contract.runtime_manifest != runtime_manifest:
        raise AnalysisContractError("loaded runtime contract differs from runtime manifest bytes")
    anchors = contract.panel.get("anchors")
    if not isinstance(anchors, list):
        raise AnalysisContractError("frozen blind panel lacks anchors")
    terminal_by_anchor: dict[str, str] = {}
    persona_index = {
        str(row.get("persona_id")): dict(row) for row in contract.personas
    }
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise AnalysisContractError("frozen blind panel anchor is invalid")
        items = anchor.get("items")
        if not isinstance(items, list) or not items:
            raise AnalysisContractError("frozen blind panel anchor has no items")
        terminal_by_anchor[str(anchor.get("anchor_id") or "")] = str(
            _mapping(items[-1], "terminal blind item").get("item_id") or ""
        )
    output: list[dict[str, Any]] = []
    for task in enumerate_tasks(contract, phase="main"):
        terminal_item = terminal_by_anchor.get(task.anchor_id)
        if not terminal_item:
            raise AnalysisContractError("runtime task anchor lacks a frozen terminal item")
        persona_contract = persona_index.get(task.persona_id)
        if persona_contract is None:
            raise AnalysisContractError("runtime task lacks its frozen persona contract")
        public_question = _mapping(
            task.item.get("public_question"), "runtime task public question"
        )
        output.append(
            {
                "task_id": task.task_id,
                "phase": task.phase,
                "analysis_population": task.analysis_population,
                "persona_id": task.persona_id,
                "pair_id": task.pair_id,
                "row_id": task.row_id,
                "anchor_id": task.anchor_id,
                "response_arm": task.response_arm,
                "condition": task.condition,
                "item_id": task.item_id,
                "correct_option": task.correct_option,
                "target_node": task.target_node,
                "family_id": task.family_id,
                "attempt_id": task.attempt_id,
                "logical_key": task.logical_key,
                "message_sha256": task.message_sha256,
                "wire_message_sha256": task.wire_message_sha256,
                "prompt_revision": task.prompt_revision,
                "prompt_contract_sha256": task.prompt_contract_sha256,
                "option_keys": tuple(
                    str(key).strip().upper()
                    for key in _mapping(
                        task.item.get("options"), "runtime task option mapping"
                    )
                ),
                "public_question": dict(public_question),
                "item_contract": dict(task.item),
                "persona_contract": dict(persona_contract),
                "is_stability_repeat": task.is_stability_repeat,
                "is_terminal": bool(
                    task.condition == "blind" and task.item_id == terminal_item
                ),
                "target_option": task.target_option,
                "random_wrong_option_baseline": (
                    1.0 / (len(task.item.get("options") or {}) - 1)
                    if task.target_option is not None
                    and len(task.item.get("options") or {}) > 1
                    else None
                ),
            }
        )
    return output


def _validate_active_contract_inputs(
    repo_root: Path,
    *,
    runtime_manifest: Mapping[str, Any],
    phase_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate stored main provenance against the active runner contract."""

    from .runner import (
        enumerate_tasks,
        load_runtime_contract,
        validate_formal_phase_provenance,
        verify_phase_provenance_against_contract,
        verify_runtime_task_manifest,
    )
    from .collect import verify_formal_carried_forward_cost_ledger

    try:
        contract = load_runtime_contract(repo_root)
        if contract.runtime_manifest != runtime_manifest:
            raise ValueError("active runtime manifest differs from stored bytes")
        runtime_proof = verify_runtime_task_manifest(
            contract,
            runtime_manifest,
            verify_git=True,
        )
        tasks = enumerate_tasks(contract, phase="main")
        carried_path = repo_root / _CARRIED_COST_LEDGER_REL
        carried_forward_cost = verify_formal_carried_forward_cost_ledger(
            _strict_json(carried_path, "reviewed carried-forward cost ledger")
        )
        scope = {
            field: phase_provenance[field]
            for field in (
                "collection_mode",
                "development_only",
                "partial",
                "formal_analysis_eligible",
                "frozen_providers",
                "selected_providers",
                "task_limit",
            )
        }
        phase_proof = verify_phase_provenance_against_contract(
            phase_provenance,
            contract=contract,
            runtime_manifest=runtime_manifest,
            runtime_proof=runtime_proof,
            tasks=tasks,
            collection_scope=scope,
            carried_forward_cost=carried_forward_cost,
        )
        formal_proof = validate_formal_phase_provenance(phase_provenance)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise AnalysisContractError(
            f"active runner contract revalidation failed: {exc}"
        ) from exc
    if (
        runtime_proof.get("ok") is not True
        or not isinstance(runtime_proof.get("git_proof"), Mapping)
        or runtime_proof["git_proof"].get("byte_identical") is not True
        or phase_proof.get("contract_revalidated") is not True
        or formal_proof.get("formal_analysis_eligible") is not True
    ):
        raise AnalysisContractError("active runner contract proof is incomplete")
    proof: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.active_analysis_contract_proof.v1",
        "ok": True,
        "runtime_git_verified": True,
        "contract_revalidated": True,
        "request_temperature": 0.0,
        "runtime_task_manifest_sha256": runtime_manifest[
            "runtime_task_manifest_sha256"
        ],
        "phase_provenance_sha256": phase_provenance[
            "phase_provenance_sha256"
        ],
        "source_set_sha256": _mapping(
            phase_provenance.get("source"), "phase source binding"
        )["source_set_sha256"],
        "target_set_hash": _mapping(
            phase_provenance.get("target"), "phase target binding"
        )["target_set_hash"],
        "carried_forward_cost_ledger_sha256": carried_forward_cost[
            "carried_forward_cost_ledger_sha256"
        ],
        "source_record_set_sha256": carried_forward_cost[
            "source_record_set_sha256"
        ],
        "provider_models": {
            provider: contract.provider_model(provider)
            for provider in FROZEN_PROVIDERS
        },
        "provider_attempt_policies": {
            provider: {
                "max_attempts": contract.provider_policy(provider).max_attempts,
                "allowed_request_max_tokens": sorted(
                    {
                        contract.provider_policy(provider).max_tokens,
                        contract.provider_policy(provider).retry_max_tokens,
                    }
                ),
                "max_tokens": contract.provider_policy(provider).max_tokens,
                "retry_max_tokens": contract.provider_policy(provider).retry_max_tokens,
                "timeout_seconds": contract.provider_policy(provider).timeout_seconds,
                "concurrency": contract.provider_policy(provider).concurrency,
                "failure_threshold": contract.provider_policy(provider).failure_threshold,
                "base_backoff_seconds": contract.provider_policy(
                    provider
                ).base_backoff_seconds,
                "max_backoff_seconds": contract.provider_policy(
                    provider
                ).max_backoff_seconds,
                "cooldown_seconds": contract.provider_policy(provider).cooldown_seconds,
                "jitter_fraction": contract.provider_policy(provider).jitter_fraction,
            }
            for provider in FROZEN_PROVIDERS
        },
        "frozen_leakage_lexicon": list(contract.lexicon),
    }
    proof["active_analysis_contract_proof_sha256"] = _canonical_sha(proof)
    return proof


def _input_file_row(path: Path, *, relative: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _verify_committed_repo_file(
    repo_root: Path,
    relative: Path,
) -> tuple[Path, dict[str, Any]]:
    if relative.is_absolute() or ".." in relative.parts:
        raise AnalysisContractError("committed evidence path must be repository-relative")
    logical_path = repo_root / relative
    candidate = repo_root
    for component in relative.parts:
        candidate /= component
        if candidate.is_symlink():
            raise AnalysisContractError(
                "committed phase evidence cannot use a symlinked path"
            )
    path = logical_path.resolve(strict=False)
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise AnalysisContractError("committed evidence path escapes the repository") from exc
    if not path.is_file() or path.is_symlink():
        raise AnalysisContractError(f"committed phase evidence is missing: {path}")
    try:
        top = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        ).resolve(strict=True)
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        committed = subprocess.check_output(
            ["git", "show", f"{head}:{relative.as_posix()}"],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
        )
        anchor_commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", relative.as_posix()],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", anchor_commit, head],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisContractError(
            "phase evidence receipt is not committed at repository HEAD"
        ) from exc
    working = path.read_bytes()
    if (
        top != repo_root
        or not anchor_commit
        or ancestry.returncode != 0
        or committed != working
    ):
        raise AnalysisContractError(
            "phase evidence receipt bytes are not anchored in the current Git history"
        )
    return path, {
        "schema_version": "yher.llm_sim_v2.committed_phase_evidence_proof.v1",
        "relative_path": relative.as_posix(),
        "head_commit": head,
        "anchor_commit": anchor_commit,
        "anchor_is_ancestor_of_head": True,
        "sha256": hashlib.sha256(working).hexdigest(),
        "size": len(working),
        "working_tree_matches_head_blob": True,
    }


def _validate_phase_evidence(
    *,
    main_root: Path,
    repo_root: Path,
    phase_provenance: Mapping[str, Any],
    tasks: Sequence[Any],
) -> tuple[dict[str, Any], Path, Path, list[tuple[str, Path]]]:
    from .evidence import build_phase_evidence_receipt

    evidence_tasks: list[Any] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            evidence_tasks.append(task)
            continue
        evidence_task = dict(task)
        evidence_task.setdefault("phase", "main")
        evidence_task.setdefault("analysis_population", "main")
        prompt_binding = _mapping(
            phase_provenance.get("prompt"), "phase evidence prompt binding"
        )
        evidence_task.setdefault("prompt_revision", prompt_binding.get("revision"))
        evidence_task.setdefault(
            "prompt_contract_sha256",
            prompt_binding.get("prompt_contract_sha256"),
        )
        if "option_keys" not in evidence_task:
            item_contract = _mapping(
                evidence_task.get("item_contract"), "phase evidence item contract"
            )
            evidence_task["option_keys"] = tuple(
                str(key).strip().upper()
                for key in _mapping(
                    item_contract.get("options"), "phase evidence option mapping"
                )
            )
        evidence_tasks.append(evidence_task)
    try:
        rebuilt = build_phase_evidence_receipt(
            main_root,
            phase_provenance=phase_provenance,
            tasks=evidence_tasks,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise AnalysisContractError(
            f"formal main phase evidence cannot be rebuilt: {exc}"
        ) from exc
    receipt_sha = str(rebuilt.get("phase_evidence_receipt_sha256") or "")
    internal_path = main_root / "evidence" / "phase_receipts" / f"{receipt_sha}.json"
    if not internal_path.is_file() or _strict_json(
        internal_path, "current internal phase evidence receipt"
    ) != rebuilt:
        raise AnalysisContractError(
            "current internal phase evidence receipt is missing or stale"
        )
    anchor_path, git_proof = _verify_committed_repo_file(
        repo_root,
        _MAIN_PHASE_RECEIPT_REL,
    )
    if _strict_json(anchor_path, "committed main phase evidence receipt") != rebuilt:
        raise AnalysisContractError(
            "committed main phase evidence receipt differs from rebuilt store evidence"
        )
    event_root = main_root / "evidence" / "provider_events"
    event_paths: list[tuple[str, Path]] = []
    event_times: dict[str, list[tuple[datetime, str]]] = {
        provider: [] for provider in FROZEN_PROVIDERS
    }
    for provider in FROZEN_PROVIDERS:
        provider_root = event_root / provider
        if not provider_root.is_dir():
            raise AnalysisContractError(
                f"{provider} provider evidence event directory is missing"
            )
        paths = sorted(provider_root.iterdir())
        if not paths or any(
            path.is_symlink() or not path.is_file() or path.suffix != ".json"
            for path in paths
        ):
            raise AnalysisContractError(
                f"{provider} provider evidence event bytes are incomplete"
            )
        for path in paths:
            event = _strict_json(path, f"{provider} provider evidence event")
            if event.get("provider") != provider:
                raise AnalysisContractError(
                    f"{provider} provider evidence event identity drifted"
                )
            recorded_at = _utc_timestamp(
                event.get("recorded_at_utc"),
                f"{provider} provider evidence recorded_at_utc",
            )
            parsed = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
            event_times[provider].append((parsed, recorded_at))
            event_paths.append((provider, path))
    provider_event_windows = {
        provider: {
            "started_at_utc": min(event_times[provider])[1],
            "finished_at_utc": max(event_times[provider])[1],
        }
        for provider in FROZEN_PROVIDERS
    }
    all_event_times = [
        value for provider_times in event_times.values() for value in provider_times
    ]
    phase_evidence = {
        "schema_version": "yher.llm_sim_v2.formal_phase_evidence_binding.v1",
        "receipt": rebuilt,
        "committed_anchor": git_proof,
        "time_semantics": "immutable_provider_evidence_recorded_at_utc",
        "provider_evidence_event_window_utc": {
            "started_at_utc": min(all_event_times)[1],
            "finished_at_utc": max(all_event_times)[1],
        },
        "provider_event_windows": provider_event_windows,
    }
    return phase_evidence, anchor_path, internal_path, event_paths


def _reconcile_cost_ledgers(
    *,
    output_base: str | Path,
    repo_root: Path,
    phase_provenance: Mapping[str, Any],
    run_budget_ledger: Mapping[str, Any],
    main_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    judge_result_manifests: Mapping[str, Mapping[str, Any] | None],
    judge_run_evidence: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Path, Path, list[tuple[str, str, Path]]]:
    """Rebuild immutable pilot/main costs and reconcile the cumulative ledger."""

    from .collect import _stored_record_costs, verify_carried_forward_cost_ledger
    from .runner import verify_prior_cost_ledger

    prior_path = repo_root / "experiments/llm_sim_v2/prior_cost_ledger.json"
    if not prior_path.is_file():
        raise AnalysisContractError(f"prior cost ledger is missing: {prior_path}")
    prior = _strict_json(prior_path, "prior cost ledger")
    try:
        verify_prior_cost_ledger(prior)
    except (TypeError, ValueError) as exc:
        raise AnalysisContractError(f"prior cost ledger verification failed: {exc}") from exc
    carried_path = repo_root / _CARRIED_COST_LEDGER_REL
    if not carried_path.is_file():
        raise AnalysisContractError(
            f"carried-forward cost ledger is missing: {carried_path}"
        )
    carried_raw = _strict_json(carried_path, "carried-forward cost ledger")
    try:
        carried = verify_carried_forward_cost_ledger(carried_raw)
    except (TypeError, ValueError) as exc:
        raise AnalysisContractError(
            f"carried-forward cost ledger verification failed: {exc}"
        ) from exc
    phase_budget = _mapping(phase_provenance.get("budget"), "phase budget binding")
    if (
        phase_budget.get("prior_cost_ledger_sha256")
        != prior.get("prior_cost_ledger_sha256")
        or float(phase_budget.get("prior_known_cost_yuan", -1.0))
        != float(prior["known_cost_yuan"])
        or float(phase_budget.get("prior_ambiguity_reserve_yuan", -1.0))
        != float(prior["pre_run_ambiguity_reserve_yuan"])
        or float(phase_budget.get("prior_documented_cost_yuan", -1.0))
        != float(prior["pre_run_total_bound_yuan"])
        or float(phase_budget.get("unknown_attempt_reserve_yuan", -1.0))
        != float(prior["unknown_attempt_reserve_yuan"])
        or float(phase_budget.get("soft_warning_yuan", -1.0)) != 300.0
        or float(phase_budget.get("hard_fuse_yuan", -1.0)) != 450.0
        or phase_budget.get("carried_forward_cost_ledger_sha256")
        != carried.get("carried_forward_cost_ledger_sha256")
        or phase_budget.get("source_phase_receipt_sha256")
        != carried.get("source_phase_receipt_sha256")
        or not math.isclose(
            float(phase_budget.get("carried_forward_known_cost_yuan", -1.0)),
            float(carried["known_cost_yuan"]),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        or not math.isclose(
            float(
                phase_budget.get("carried_forward_unknown_reserve_yuan", -1.0)
            ),
            float(carried["unknown_cost_reserve_yuan"]),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        or not math.isclose(
            float(
                phase_budget.get("carried_forward_total_accounted_cost_yuan", -1.0)
            ),
            float(carried["total_accounted_cost_yuan"]),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    ):
        raise AnalysisContractError(
            "phase budget binding differs from prior or carried-forward cost ledger"
        )
    try:
        stored = _stored_record_costs(
            output_base,
            phases=("pilot", "main"),
            unknown_attempt_reserve_yuan=float(prior["unknown_attempt_reserve_yuan"]),
        )
    except ValueError as exc:
        raise AnalysisContractError(f"stored run record cost reconciliation failed: {exc}") from exc
    expected = {
        "schema_version": "yher.llm_sim_v2.run_budget_ledger.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "prior_cost_ledger_sha256": prior["prior_cost_ledger_sha256"],
        "prior_known_cost_yuan": round(float(prior["known_cost_yuan"]), 8),
        "prior_ambiguity_reserve_yuan": round(
            float(prior["pre_run_ambiguity_reserve_yuan"]), 8
        ),
        "prior_documented_cost_yuan": round(
            float(prior["pre_run_total_bound_yuan"]), 8
        ),
        "carried_forward_cost_ledger_sha256": carried[
            "carried_forward_cost_ledger_sha256"
        ],
        "carried_forward_source_phase": carried["source_phase"],
        "carried_forward_source_record_set_sha256": carried[
            "source_record_set_sha256"
        ],
        "carried_forward_source_phase_receipt_sha256": carried[
            "source_phase_receipt_sha256"
        ],
        "carried_forward_known_cost_yuan": carried["known_cost_yuan"],
        "carried_forward_unknown_reserve_yuan": carried[
            "unknown_cost_reserve_yuan"
        ],
        "carried_forward_total_accounted_cost_yuan": carried[
            "total_accounted_cost_yuan"
        ],
        "unknown_attempt_reserve_yuan": float(
            prior["unknown_attempt_reserve_yuan"]
        ),
        "immutable_record_known_cost_yuan": stored["known_cost_yuan"],
        "immutable_record_unknown_reserve_yuan": stored[
            "unknown_cost_reserve_yuan"
        ],
        "immutable_record_cost_yuan": stored["accounted_cost_yuan"],
        "total_known_cost_yuan": round(
            float(prior["known_cost_yuan"])
            + float(carried["known_cost_yuan"])
            + stored["known_cost_yuan"],
            8,
        ),
        "total_unknown_reserve_yuan": round(
            float(prior["pre_run_ambiguity_reserve_yuan"])
            + float(carried["unknown_cost_reserve_yuan"])
            + stored["unknown_cost_reserve_yuan"],
            8,
        ),
        "total_accounted_cost_yuan": round(
            float(prior["pre_run_total_bound_yuan"])
            + float(carried["total_accounted_cost_yuan"])
            + stored["accounted_cost_yuan"],
            8,
        ),
        "needs_user": (
            stored["unknown_attempt_count"] > 0
            or float(carried["unknown_cost_reserve_yuan"]) > 0.0
        ),
        "needs_user_reasons": (
            ["unknown_provider_billing_reserved"]
            if stored["unknown_attempt_count"] > 0
            or float(carried["unknown_cost_reserve_yuan"]) > 0.0
            else []
        ),
        "soft_warning_yuan": 300.0,
        "hard_fuse_yuan": 450.0,
    }
    for field, value in expected.items():
        actual = run_budget_ledger.get(field)
        if isinstance(value, float):
            try:
                matches = math.isclose(
                    float(actual), value, rel_tol=0.0, abs_tol=1e-8
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == value
        if not matches:
            raise AnalysisContractError(
                f"run budget ledger does not reconcile: {field}"
            )

    run_root = _main_store_root(output_base).parent
    phase_records: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {
        "main": {
            provider: dict(records) for provider, records in main_records.items()
        }
    }
    cost_only_paths: list[tuple[str, str, Path]] = []
    pilot_manifest_paths: list[tuple[str, Path]] = []
    pilot_root = run_root / "pilot/records"
    if pilot_root.exists():
        if not pilot_root.is_dir():
            raise AnalysisContractError("pilot cost record path is not a directory")
        phase_records["pilot"] = {}
        for provider_dir in sorted(pilot_root.iterdir()):
            if not provider_dir.is_dir() or provider_dir.name not in {"deepseek", "doubao"}:
                raise AnalysisContractError("pilot cost records use an unknown provider namespace")
            provider_records: dict[str, Mapping[str, Any]] = {}
            for path in sorted(provider_dir.iterdir()):
                if not path.is_file() or path.suffix != ".json":
                    raise AnalysisContractError("pilot cost record namespace is invalid")
                record = _strict_json(path, "pilot cost response record")
                task_id = path.stem
                if (
                    record.get("schema_version")
                    != "yher.llm_sim_v2.response_record.v2"
                    or record.get("run_id") != RUN_ID
                    or record.get("phase") != "pilot"
                    or record.get("analysis_population") != "pilot"
                    or record.get("provider") != provider_dir.name
                    or record.get("task_id") != task_id
                ):
                    raise AnalysisContractError("pilot cost response identity is invalid")
                attempts = record.get("attempts")
                if not isinstance(attempts, list) or not attempts:
                    raise AnalysisContractError(
                        "pilot v2 cost response lacks attempt evidence"
                    )
                from .evidence import validate_response_content_binding

                for attempt in attempts:
                    if (
                        not isinstance(attempt, Mapping)
                        or not isinstance(
                            attempt.get("provider_response_received"), bool
                        )
                    ):
                        raise AnalysisContractError(
                            "pilot v2 attempt lacks provider-response receipt state"
                        )
                    if attempt["provider_response_received"] is True:
                        try:
                            validate_response_content_binding(attempt)
                        except (TypeError, ValueError) as exc:
                            raise AnalysisContractError(
                                f"pilot provider response binding is invalid: {exc}"
                            ) from exc
                provider_records[task_id] = record
                cost_only_paths.append((provider_dir.name, task_id, path))
            phase_records["pilot"][provider_dir.name] = provider_records
        pilot_manifest_root = run_root / "pilot/provider_manifests"
        if not pilot_manifest_root.is_dir():
            raise AnalysisContractError("pilot provider manifest directory is missing")
        pilot_manifest_entries = {
            path.stem: path for path in pilot_manifest_root.iterdir()
        }
        if set(pilot_manifest_entries) != set(phase_records["pilot"]) or any(
            not path.is_file() or path.suffix != ".json"
            for path in pilot_manifest_entries.values()
        ):
            raise AnalysisContractError(
                "pilot provider manifests do not match pilot record namespaces"
            )
        pilot_manifest_paths.extend(sorted(pilot_manifest_entries.items()))
    rows = _provider_phase_cost_rows(phase_records)
    pilot_cost_rows = {
        str(row["provider"]): row for row in rows if row["phase"] == "pilot"
    }
    for provider, path in pilot_manifest_paths:
        manifest = _strict_json(path, f"{provider} pilot provider manifest")
        if (
            manifest.get("schema_version")
            != "yher.llm_sim_v2.provider_manifest.v1"
            or manifest.get("simulated") is not True
            or manifest.get("run_id") != RUN_ID
            or manifest.get("phase") != "pilot"
            or manifest.get("analysis_population") != "pilot"
            or manifest.get("provider") != provider
        ):
            raise AnalysisContractError(
                f"{provider} pilot provider manifest identity is invalid"
            )
        lifecycle = _mapping(
            manifest.get("lifecycle"), f"{provider} pilot lifecycle"
        )
        expected_task_ids = lifecycle.get("expected_task_ids")
        if not isinstance(expected_task_ids, list):
            raise AnalysisContractError(
                f"{provider} pilot provider manifest lacks expected record roster"
            )
        from .evidence import build_provider_record_set

        try:
            recomputed_record_set = build_provider_record_set(
                run_root / "pilot",
                provider=provider,
                expected_task_ids=[str(value) for value in expected_task_ids],
            )
        except (OSError, TypeError, ValueError) as exc:
            raise AnalysisContractError(
                f"{provider} pilot provider record-set bytes cannot be verified: {exc}"
            ) from exc
        if manifest.get("record_set") != recomputed_record_set:
            raise AnalysisContractError(
                f"{provider} pilot provider record-set binding differs from disk bytes"
            )
        row = pilot_cost_rows[provider]
        budget = _mapping(
            manifest.get("budget"), f"{provider} pilot provider budget"
        )
        expected_budget = {
            "provider_record_known_cost_yuan": row["known_cost_yuan"],
            "provider_record_unknown_reserve_yuan": row[
                "unknown_cost_reserve_yuan"
            ],
            "provider_record_accounted_cost_yuan": row["accounted_cost_yuan"],
        }
        try:
            budget_matches = all(
                math.isclose(
                    float(budget.get(field, -1.0)),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
                for field, expected in expected_budget.items()
            )
        except (TypeError, ValueError):
            budget_matches = False
        needs_user = _mapping(
            manifest.get("needs_user"), f"{provider} pilot provider needs_user"
        )
        if (
            manifest.get("record_count") != row["record_count"]
            or not budget_matches
            or needs_user.get("required") is not row["needs_user"]
            or needs_user.get("unknown_cost_attempt_count")
            != row["unknown_attempt_count"]
        ):
            raise AnalysisContractError(
                f"{provider} pilot provider budget does not reconcile to record attempts"
            )
    cost_input_files = [
        {
            "artifact_type": "response_record",
            "phase": "pilot",
            "provider": provider,
            "task_id": task_id,
            "path": f"pilot/records/{provider}/{task_id}.json",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
            "purpose": "cost_reconciliation_only",
            "included_in_metrics": False,
        }
        for provider, task_id, path in sorted(cost_only_paths)
    ]
    cost_input_files.extend(
        {
            "artifact_type": "provider_manifest",
            "phase": "pilot",
            "provider": provider,
            "path": f"pilot/provider_manifests/{provider}.json",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
            "purpose": "cost_reconciliation_only",
            "included_in_metrics": False,
        }
        for provider, path in pilot_manifest_paths
    )
    cost_input_files.sort(
        key=lambda row: (
            str(row["phase"]),
            str(row["provider"]),
            str(row["artifact_type"]),
            str(row["path"]),
        )
    )
    cost_input_manifest = {
        "schema_version": "yher.llm_sim_v2.cost_reconciliation_artifact_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "metric_input": False,
        "file_count": len(cost_input_files),
        "files": cost_input_files,
        "file_set_sha256": _canonical_sha(cost_input_files),
    }
    collection_cost = {
            "schema_version": "yher.llm_sim_v2.cost_accounting.v1",
            "currency": "CNY",
            "prior_cost_ledger_sha256": prior["prior_cost_ledger_sha256"],
            "prior_known_cost_yuan": prior["known_cost_yuan"],
            "prior_ambiguity_reserve_yuan": prior[
                "pre_run_ambiguity_reserve_yuan"
            ],
            "prior_accounted_cost_yuan": prior["pre_run_total_bound_yuan"],
            "carried_forward_cost_ledger_sha256": carried[
                "carried_forward_cost_ledger_sha256"
            ],
            "carried_forward_known_cost_yuan": carried["known_cost_yuan"],
            "carried_forward_unknown_reserve_yuan": carried[
                "unknown_cost_reserve_yuan"
            ],
            "carried_forward_accounted_cost_yuan": carried[
                "total_accounted_cost_yuan"
            ],
            "provider_phase": rows,
            "immutable_record_totals": stored,
            "total_known_cost_yuan": expected["total_known_cost_yuan"],
            "total_unknown_reserve_yuan": expected["total_unknown_reserve_yuan"],
            "total_accounted_cost_yuan": expected["total_accounted_cost_yuan"],
            "unknown_attempt_reserve_yuan": prior[
                "unknown_attempt_reserve_yuan"
            ],
            "needs_user": expected["needs_user"],
            "needs_user_reasons": expected["needs_user_reasons"],
            "soft_warning_yuan": 300.0,
            "hard_fuse_yuan": 450.0,
            "source": (
                "reconciled_prior_carried_forward_and_immutable_attempt_ledgers"
            ),
            "cost_reconciliation_artifact_manifest": cost_input_manifest,
        }
    cost_accounting = _merge_judge_cost_accounting(
        collection_cost,
        _judge_cost_accounting(
            judge_result_manifests,
            judge_run_evidence=judge_run_evidence,
        ),
    )
    return (
        cost_accounting,
        prior_path,
        carried_path,
        cost_only_paths,
    )


def _validate_lifecycle_histories(
    main_root: Path,
    provider_manifests: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, int, Path]]:
    """Validate immutable lifecycle event bytes and current-manifest binding."""

    lifecycle_root = main_root / "provider_lifecycle"
    if not lifecycle_root.is_dir():
        raise AnalysisContractError("provider lifecycle history directory is missing")
    root_entries = {entry.name: entry for entry in lifecycle_root.iterdir()}
    if set(root_entries) != set(FROZEN_PROVIDERS) or any(
        not entry.is_dir() for entry in root_entries.values()
    ):
        raise AnalysisContractError(
            "provider lifecycle history requires exactly six provider namespaces"
        )
    loaded_paths: list[tuple[str, int, Path]] = []
    for provider in FROZEN_PROVIDERS:
        manifest = _mapping(
            provider_manifests[provider], f"{provider} provider manifest"
        )
        history = manifest.get("lifecycle_history")
        if not isinstance(history, list) or not history:
            raise AnalysisContractError(
                f"{provider} lifecycle history is missing or empty"
            )
        referenced: set[str] = set()
        events: list[Mapping[str, Any]] = []
        for index, row_value in enumerate(history):
            row = _mapping(row_value, f"{provider} lifecycle history row")
            relative = Path(str(row.get("path") or ""))
            expected_parent = Path("provider_lifecycle") / provider
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parent != expected_parent
                or row.get("event_index") != index
            ):
                raise AnalysisContractError(
                    f"{provider} lifecycle history path/index is invalid"
                )
            normalized = relative.as_posix()
            if normalized in referenced:
                raise AnalysisContractError(
                    f"{provider} lifecycle history repeats an event path"
                )
            referenced.add(normalized)
            path = main_root / relative
            if not path.is_file():
                raise AnalysisContractError(
                    f"{provider} lifecycle history event is missing: {path}"
                )
            event = _strict_json(path, f"{provider} lifecycle history event")
            payload = dict(event)
            advertised = payload.pop("lifecycle_event_sha256", None)
            if (
                event.get("schema_version")
                != "yher.llm_sim_v2.provider_lifecycle_event.v1"
                or event.get("simulated") is not True
                or event.get("run_id") != RUN_ID
                or event.get("phase") != "main"
                or event.get("analysis_population") != "main"
                or event.get("provider") != provider
                or event.get("event_index") != index
                or advertised != _canonical_sha(payload)
                or row.get("lifecycle_event_sha256") != advertised
                or row.get("provider_lifecycle")
                != event.get("provider_lifecycle")
                or row.get("finished_at_utc") != event.get("finished_at_utc")
                or path.name != f"{index:04d}-{advertised}.json"
            ):
                raise AnalysisContractError(
                    f"{provider} lifecycle history event digest or binding drifted"
                )
            events.append(event)
            loaded_paths.append((provider, index, path))
        actual = {
            path.name for path in root_entries[provider].iterdir() if path.is_file()
        }
        expected = {Path(value).name for value in referenced}
        if actual != expected or any(
            not path.is_file() for path in root_entries[provider].iterdir()
        ):
            raise AnalysisContractError(
                f"{provider} lifecycle history directory differs from manifest"
            )
        latest = events[-1]
        for field in (
            "provider_lifecycle",
            "lifecycle",
            "interruption",
            "unavailable",
            "needs_user",
            "provenance",
            "finished_at_utc",
        ):
            if latest.get(field) != manifest.get(field):
                raise AnalysisContractError(
                    f"{provider} current manifest differs from lifecycle history: {field}"
                )
    return loaded_paths


def load_analysis_inputs(
    *,
    output_base: str | Path,
    repo_root: str | Path,
    judge_results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load only provider-scoped formal-main artifacts and bind every input byte."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    main_root = _main_store_root(output_base)
    if not main_root.is_dir():
        raise AnalysisContractError(f"formal main store is missing: {main_root}")
    phase_path = main_root / "phase_provenance.json"
    runtime_path = repo / "experiments/llm_sim_v2/runtime_task_manifest.json"
    mapping_path = repo / "experiments/llm_sim_v2/frozen_v0/target_option_mapping.json"
    for path, label in (
        (phase_path, "phase provenance"),
        (runtime_path, "runtime task manifest"),
        (mapping_path, "target-option mapping"),
    ):
        if not path.is_file():
            raise AnalysisContractError(f"{label} is missing: {path}")
    bound_input_rows: dict[str, dict[str, Any]] = {}

    def load_bound_json(path: Path, label: str, *, relative: str) -> dict[str, Any]:
        before = _input_file_row(path, relative=relative)
        value = _strict_json(path, label)
        after = _input_file_row(path, relative=relative)
        if before != after:
            raise AnalysisContractError(
                f"analysis input changed while it was parsed: {relative}"
            )
        bound_input_rows[relative] = before
        return value

    phase = load_bound_json(
        phase_path,
        "phase provenance",
        relative="main/phase_provenance.json",
    )
    runtime = load_bound_json(
        runtime_path,
        "runtime task manifest",
        relative="repo/experiments/llm_sim_v2/runtime_task_manifest.json",
    )
    mapping_manifest = load_bound_json(
        mapping_path,
        "target-option mapping",
        relative=(
            "repo/experiments/llm_sim_v2/frozen_v0/target_option_mapping.json"
        ),
    )
    active_contract_proof = _validate_active_contract_inputs(
        repo,
        runtime_manifest=runtime,
        phase_provenance=phase,
    )
    expected_tasks = _load_expected_tasks(repo, runtime)
    validate_inputs(
        phase_provenance=phase,
        runtime_manifest=runtime,
        expected_tasks=expected_tasks,
        active_contract_proof=active_contract_proof,
    )
    (
        phase_evidence,
        phase_evidence_anchor_path,
        internal_phase_receipt_path,
        provider_evidence_event_paths,
    ) = _validate_phase_evidence(
        main_root=main_root,
        repo_root=repo,
        phase_provenance=phase,
        tasks=expected_tasks,
    )

    run_budget_path = main_root.parent / "run_budget_ledger.json"
    if not run_budget_path.is_file():
        raise AnalysisContractError(
            f"cumulative run budget ledger is missing: {run_budget_path}"
        )
    run_budget_ledger = load_bound_json(
        run_budget_path,
        "run budget ledger",
        relative="run/run_budget_ledger.json",
    )

    manifest_root = main_root / "provider_manifests"
    if not manifest_root.is_dir():
        raise AnalysisContractError("formal main provider manifest directory is missing")
    manifest_entries = {entry.name: entry for entry in manifest_root.iterdir()}
    expected_manifest_names = {f"{provider}.json" for provider in FROZEN_PROVIDERS}
    if set(manifest_entries) != expected_manifest_names or any(
        not entry.is_file() for entry in manifest_entries.values()
    ):
        raise AnalysisContractError("formal main requires exactly six provider manifests")
    provider_manifests = {
        provider: load_bound_json(
            manifest_entries[f"{provider}.json"],
            f"{provider} provider manifest",
            relative=f"main/provider_manifests/{provider}.json",
        )
        for provider in FROZEN_PROVIDERS
    }
    lifecycle_event_paths = _validate_lifecycle_histories(
        main_root, provider_manifests
    )

    records_by_provider: dict[str, dict[str, dict[str, Any]]] = {
        provider: {} for provider in FROZEN_PROVIDERS
    }
    record_paths: list[tuple[str, str, Path]] = []
    records_root = main_root / "records"
    if records_root.exists():
        if not records_root.is_dir():
            raise AnalysisContractError("formal main records path is not a directory")
        entries = list(records_root.iterdir())
        unexpected = [
            entry
            for entry in entries
            if not entry.is_dir() or entry.name not in FROZEN_PROVIDERS
        ]
        if unexpected:
            raise AnalysisContractError(
                "records must use a frozen provider namespace; flat records are forbidden"
            )
        for provider_dir in entries:
            provider = provider_dir.name
            for record_path in provider_dir.iterdir():
                if not record_path.is_file() or record_path.suffix != ".json":
                    raise AnalysisContractError(
                        f"{provider} record namespace contains a non-JSON artifact"
                    )
                task_id = record_path.stem
                if task_id in records_by_provider[provider]:
                    raise AnalysisContractError(f"duplicate record task ID for {provider}")
                record = load_bound_json(
                    record_path,
                    f"{provider} response record",
                    relative=f"main/records/{provider}/{task_id}.json",
                )
                if record.get("task_id") != task_id:
                    raise AnalysisContractError(
                        f"{provider} response filename does not match task_id"
                    )
                records_by_provider[provider][task_id] = record
                record_paths.append((provider, task_id, record_path))

    from .evidence import build_provider_record_set

    expected_task_ids = [str(task["task_id"]) for task in expected_tasks]
    for provider in FROZEN_PROVIDERS:
        try:
            recomputed_record_set = build_provider_record_set(
                main_root,
                provider=provider,
                expected_task_ids=expected_task_ids,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise AnalysisContractError(
                f"{provider} provider record-set bytes cannot be verified: {exc}"
            ) from exc
        if provider_manifests[provider].get("record_set") != recomputed_record_set:
            raise AnalysisContractError(
                f"{provider} provider record-set binding differs from disk bytes"
            )

    prior_cost_relative = "repo/experiments/llm_sim_v2/prior_cost_ledger.json"
    carried_cost_relative = (
        "repo/experiments/llm_sim_v2/evidence_anchors/"
        "legacy_pilot_carried_forward_cost.json"
    )
    expected_prior_cost_path = repo / prior_cost_relative.removeprefix("repo/")
    expected_carried_cost_path = repo / carried_cost_relative.removeprefix("repo/")
    bound_input_rows[prior_cost_relative] = _input_file_row(
        expected_prior_cost_path,
        relative=prior_cost_relative,
    )
    bound_input_rows[carried_cost_relative] = _input_file_row(
        expected_carried_cost_path,
        relative=carried_cost_relative,
    )
    (
        judge_result_manifests,
        judge_result_paths,
        judge_artifact_roots,
        judge_run_evidence,
    ) = _load_judge_result_manifests(
        judge_results_dir,
        repo_root=repo,
    )
    (
        cost_accounting,
        prior_cost_path,
        carried_cost_path,
        cost_only_paths,
    ) = _reconcile_cost_ledgers(
        output_base=output_base,
        repo_root=repo,
        phase_provenance=phase,
        run_budget_ledger=run_budget_ledger,
        main_records=records_by_provider,
        judge_result_manifests=judge_result_manifests,
        judge_run_evidence=judge_run_evidence,
    )
    if (
        prior_cost_path != expected_prior_cost_path
        or carried_cost_path != expected_carried_cost_path
        or _input_file_row(prior_cost_path, relative=prior_cost_relative)
        != bound_input_rows[prior_cost_relative]
        or _input_file_row(carried_cost_path, relative=carried_cost_relative)
        != bound_input_rows[carried_cost_relative]
    ):
        raise AnalysisContractError(
            "cost ledger input changed while cost accounting was reconciled"
        )
    input_files = [
        bound_input_rows["main/phase_provenance.json"],
        _input_file_row(
            phase_evidence_anchor_path,
            relative=f"repo/{_MAIN_PHASE_RECEIPT_REL.as_posix()}",
        ),
        _input_file_row(
            internal_phase_receipt_path,
            relative=(
                "main/"
                f"{internal_phase_receipt_path.relative_to(main_root).as_posix()}"
            ),
        ),
        bound_input_rows[
            "repo/experiments/llm_sim_v2/runtime_task_manifest.json"
        ],
        bound_input_rows[
            "repo/experiments/llm_sim_v2/frozen_v0/target_option_mapping.json"
        ],
        bound_input_rows[prior_cost_relative],
        bound_input_rows[carried_cost_relative],
        bound_input_rows["run/run_budget_ledger.json"],
    ]
    input_files.extend(
        _input_file_row(
            path,
            relative=f"main/evidence/provider_events/{provider}/{path.name}",
        )
        for provider, path in provider_evidence_event_paths
    )
    input_files.extend(
        bound_input_rows[f"main/provider_manifests/{provider}.json"]
        for provider in FROZEN_PROVIDERS
    )
    input_files.extend(
        _input_file_row(
            path,
            relative=f"main/provider_lifecycle/{provider}/{path.name}",
        )
        for provider, _index, path in lifecycle_event_paths
    )
    input_files.extend(
        bound_input_rows[f"main/records/{provider}/{task_id}.json"]
        for provider, task_id, path in sorted(record_paths)
    )
    input_files.extend(
        _input_file_row(
            path,
            relative=f"judge-results/{relative}",
        )
        for relative, path in judge_result_paths
    )
    if judge_run_evidence is not None:
        input_files.append(
            _input_file_row(
                repo / _JUDGE_RUN_RECEIPT_REL,
                relative=f"repo/{_JUDGE_RUN_RECEIPT_REL.as_posix()}",
            )
        )
    (
        final_phase_evidence,
        final_anchor_path,
        final_internal_path,
        final_event_paths,
    ) = _validate_phase_evidence(
        main_root=main_root,
        repo_root=repo,
        phase_provenance=phase,
        tasks=expected_tasks,
    )
    if (
        final_phase_evidence != phase_evidence
        or final_anchor_path != phase_evidence_anchor_path
        or final_internal_path != internal_phase_receipt_path
        or final_event_paths != provider_evidence_event_paths
    ):
        raise AnalysisContractError(
            "formal phase evidence changed while analysis inputs were loaded"
        )
    anchor_rows = [
        row
        for row in input_files
        if row.get("path") == f"repo/{_MAIN_PHASE_RECEIPT_REL.as_posix()}"
    ]
    committed_anchor = _mapping(
        phase_evidence.get("committed_anchor"), "committed phase evidence proof"
    )
    if (
        len(anchor_rows) != 1
        or anchor_rows[0].get("sha256") != committed_anchor.get("sha256")
        or anchor_rows[0].get("size") != committed_anchor.get("size")
    ):
        raise AnalysisContractError(
            "analysis input manifest does not bind the committed phase evidence bytes"
        )
    judge_sources = {
        f"judge-results/{relative}": path
        for relative, path in judge_result_paths
    }
    for row in input_files:
        relative = str(row.get("path") or "")
        if relative.startswith("repo/"):
            source_path = repo / relative.removeprefix("repo/")
        elif relative.startswith("main/"):
            source_path = main_root / relative.removeprefix("main/")
        elif relative.startswith("run/"):
            source_path = main_root.parent / relative.removeprefix("run/")
        elif relative in judge_sources:
            source_path = judge_sources[relative]
        else:
            raise AnalysisContractError(
                f"analysis input manifest path is outside known roots: {relative}"
            )
        if _input_file_row(source_path, relative=relative) != row:
            raise AnalysisContractError(
                f"analysis input changed while analysis inputs were loaded: {relative}"
            )
    input_artifact_manifest = {
        "schema_version": "yher.llm_sim_v2.analysis_input_artifact_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "analysis_population": "main",
        "files": input_files,
        "input_file_count": len(input_files),
        "record_file_count": len(record_paths),
        "input_file_set_sha256": _canonical_sha(input_files),
    }
    loaded = {
        "expected_tasks": expected_tasks,
        "runtime_manifest": runtime,
        "phase_provenance": phase,
        "records_by_provider": records_by_provider,
        "provider_manifests": provider_manifests,
        "mapping_manifest": mapping_manifest,
        "active_contract_proof": active_contract_proof,
        "cost_accounting": cost_accounting,
        "judge_result_manifests": judge_result_manifests,
        "judge_run_evidence": judge_run_evidence,
        "judge_artifact_roots": judge_artifact_roots,
        "judge_artifact_sources": {
            f"judge-results/{relative}": str(path.resolve())
            for relative, path in judge_result_paths
        },
        "input_artifact_manifest": input_artifact_manifest,
        "phase_evidence": phase_evidence,
    }
    loaded["_formal_loader_proof"] = _FormalLoaderProof(
        _FORMAL_LOADER_SENTINEL,
        bundle_sha256=_formal_loader_bundle_sha256(loaded),
    )
    return loaded


def prepare_judge_cases(
    *,
    output_base: str | Path,
    judge_results_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze only the deterministic blind case manifest; never publish W3 output."""

    repo = (
        Path(repo_root).expanduser().resolve(strict=True)
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    loaded = load_analysis_inputs(
        output_base=output_base,
        repo_root=repo,
        judge_results_dir=None,
    )
    result = analyze_dataset(**loaded)
    if (
        result.get("schema_version")
        != "yher.llm_sim_v2.analysis_results.pre_adjudication.v1"
        or result.get("analysis_mode") != "formal_main_pre_adjudication"
        or result.get("formal_analysis_eligible") is not False
        or result.get("publication_output_eligible") is not False
    ):
        raise AnalysisContractError("judge preparation unexpectedly became publishable")
    judge = _mapping(result.get("judge_adjudication"), "judge preparation")
    case_manifest = dict(
        _mapping(judge.get("case_manifest"), "prepared judge case manifest")
    )
    analysis = _mapping(judge.get("analysis"), "prepared judge analysis")
    if analysis.get("status") not in {
        "missing_all_judges",
        "not_applicable_zero_cases",
    }:
        raise AnalysisContractError("judge preparation contains adjudication results")
    from .judge_execution import bind_prepared_judge_case_manifest

    path = bind_prepared_judge_case_manifest(
        case_manifest=case_manifest,
        output_root=judge_results_dir,
    )
    if _strict_json(path, "installed prepared judge case manifest") != case_manifest:
        raise AnalysisContractError("prepared judge case bytes changed during installation")
    return case_manifest


def _build_formal_run_analysis_authority():
    authority_key = os.urandom(32)

    def proof_payload(
        *,
        loader_bundle_sha256: str,
        result_sha256: str,
        input_artifact_manifest_sha256: str,
    ) -> dict[str, str]:
        return {
            "domain": "yher.llm_sim_v2.formal_publication.v1",
            "loader_bundle_sha256": loader_bundle_sha256,
            "result_sha256": result_sha256,
            "input_artifact_manifest_sha256": input_artifact_manifest_sha256,
        }

    def authorization_mac(payload: Mapping[str, Any]) -> str:
        return hmac.new(
            authority_key,
            _canonical_bytes(payload),
            hashlib.sha256,
        ).hexdigest()

    def verify_formal_publication_proof(
        proof: _FormalPublicationProof | None,
        *,
        result: Mapping[str, Any],
        input_artifact_manifest: Mapping[str, Any] | None,
    ) -> bool:
        if not isinstance(proof, _FormalPublicationProof) or not isinstance(
            input_artifact_manifest, Mapping
        ):
            return False
        try:
            result_sha = _canonical_sha(result)
            input_sha = _canonical_sha(input_artifact_manifest)
            payload = proof_payload(
                loader_bundle_sha256=proof.loader_bundle_sha256,
                result_sha256=result_sha,
                input_artifact_manifest_sha256=input_sha,
            )
            return (
                proof.result_sha256 == result_sha
                and proof.input_artifact_manifest_sha256 == input_sha
                and result.get("formal_loader_bundle_sha256")
                == proof.loader_bundle_sha256
                and isinstance(proof.authorization_mac, str)
                and hmac.compare_digest(
                    proof.authorization_mac,
                    authorization_mac(payload),
                )
            )
        except (TypeError, ValueError):
            return False

    def run_analysis(
        *,
        output_base: str | Path,
        result_dir: str | Path,
        repo_root: str | Path | None = None,
        judge_results_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load, analyze, and atomically publish one formal-main package."""

        if judge_results_dir is None:
            raise AnalysisContractError(
                "final formal analysis requires a finalized judge run root"
            )
        repo = (
            Path(repo_root).expanduser().resolve(strict=True)
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        loaded = load_analysis_inputs(
            output_base=output_base,
            repo_root=repo,
            judge_results_dir=judge_results_dir,
        )
        input_manifest = loaded["input_artifact_manifest"]
        loader_proof = loaded["_formal_loader_proof"]
        result = analyze_dataset(**loaded)
        result_sha = _canonical_sha(result)
        input_sha = _canonical_sha(input_manifest)
        payload = proof_payload(
            loader_bundle_sha256=loader_proof.bundle_sha256,
            result_sha256=result_sha,
            input_artifact_manifest_sha256=input_sha,
        )
        publication_proof = _FormalPublicationProof(
            loader_bundle_sha256=loader_proof.bundle_sha256,
            result_sha256=result_sha,
            input_artifact_manifest_sha256=input_sha,
            authorization_mac=authorization_mac(payload),
        )
        write_analysis_outputs(
            result,
            result_dir,
            input_artifact_manifest=input_manifest,
            judge_artifact_sources=loaded["judge_artifact_sources"],
            _formal_publication_proof=publication_proof,
        )
        return result

    return run_analysis, verify_formal_publication_proof


run_analysis, _verify_formal_publication_proof = _build_formal_run_analysis_authority()
del _build_formal_run_analysis_authority


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen Persona-v2 formal main collection."
    )
    parser.add_argument(
        "--output-base",
        required=True,
        type=Path,
        help="Collection base containing llm-personas-v2-dual/main.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="New final directory for machine results and publication figures.",
    )
    parser.add_argument(
        "--judge-results-dir",
        type=Path,
        help=(
            "Judge run root: a new case-only root in preparation mode, or the "
            "finalized anchored run root in final mode."
        ),
    )
    parser.add_argument(
        "--prepare-judge-cases",
        action="store_true",
        help="Freeze only the nonpublication judge case manifest.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.prepare_judge_cases:
            if args.judge_results_dir is None or args.result_dir is not None:
                raise AnalysisContractError(
                    "judge preparation requires --judge-results-dir and forbids --result-dir"
                )
            case_manifest = prepare_judge_cases(
                output_base=args.output_base,
                judge_results_dir=args.judge_results_dir,
                repo_root=args.repo_root,
            )
            summary = {
                "run_id": RUN_ID,
                "analysis_population": "main",
                "mode": "formal_main_pre_adjudication",
                "publication_output_eligible": False,
                "selected_count": case_manifest["selected_count"],
                "case_manifest_sha256": case_manifest["case_manifest_sha256"],
                "judge_results_dir": str(args.judge_results_dir),
            }
        else:
            if args.result_dir is None or args.judge_results_dir is None:
                raise AnalysisContractError(
                    "final analysis requires --result-dir and --judge-results-dir"
                )
            result = run_analysis(
                output_base=args.output_base,
                result_dir=args.result_dir,
                repo_root=args.repo_root,
                judge_results_dir=args.judge_results_dir,
            )
            summary = {
                "run_id": result["run_id"],
                "analysis_population": result["analysis_population"],
                "persona_clusters": result["independent_cluster_count"],
                "result_dir": str(args.result_dir),
            }
    except AnalysisContractError as exc:
        parser.exit(2, f"analysis contract error: {exc}\n")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "AnalysisContractError",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONTROLLED_STATES",
    "cluster_bootstrap_mean",
    "controlled_response_state",
    "analyze_dataset",
    "build_parser",
    "pairwise_terminal_agreement",
    "load_analysis_inputs",
    "main",
    "prepare_judge_cases",
    "run_analysis",
    "validate_inputs",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI contract
    raise SystemExit(main())
