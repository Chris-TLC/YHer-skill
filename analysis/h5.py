"""Fail-closed collection and analysis for the frozen H5 LLM-persona study."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score

from experiments.llm_sim.config import load_frozen_config
from experiments.llm_sim.provenance import CODE_PATTERNS, verify_frozen_document_commit

from .dataset import DatasetContractError
from .provenance import verify_analysis_provenance


PRODUCTION_STATES = ("M", "P", "C", "U")
TERMINAL_CATEGORIES = PRODUCTION_STATES + ("NC",)
FROZEN_PROVIDERS = (
    "deepseek",
    "glm",
    "kimi",
    "minimax",
    "doubao",
    "tongyi",
)
BOOTSTRAP_SEED = 2026071303
BOOTSTRAP_ITERATIONS = 10_000
H5_RESULT_IDS = (
    "H5_QUALIFYING_PROVIDER_COUNT",
    "H5_MINIMUM_COMPLETED_PERSONAS_PER_QUALIFYING_CELL",
    "H5_WEAK_ACCURACY_GATE",
    "H5_STRONG_ACCURACY_GATE",
    "H5_MISCONCEPTION_HIT_RATE_CONTRAST",
    "H5_MISCONCEPTION_HIT_RATE_CONTRAST_CI95",
)
H5_LIFECYCLE_DENOMINATOR_FIELDS = (
    "frozen_provider_count",
    "collected_provider_count",
    "qualifying_provider_count",
    "invalid_calibration_schema_provider_count",
    "invalid_provider_artifact_count",
    "missing_provider_count",
    "missing_required_revision_provider_count",
    "network_interruption_provider_count",
    "model_drift_exclusion_provider_count",
    "provider_configuration_exclusion_provider_count",
    "pre_outcome_design_exclusion_provider_count",
    "technical_interruption_provider_count",
    "post_calibration_exclusion_provider_count",
    "provider_lifecycle_counts",
)
PROGRAMMATIC_H5_PENDING_STATUS = "PROGRAMMATIC_COMPLETE_H5_PENDING"
H5_EVALUATED_STATUS = "COMPLETE_H5_EVALUATED"
H5_EXCLUDED_STATUS = "COMPLETE_H5_EXCLUDED_PRE_OUTCOME"
COLLECTION_LOCK_RELATIVE = Path("experiments/config/h5_collection_lock.json")
H5_OUTPUT_FILES = frozenset(
    {
        "agreement_matrix.json",
        "artifact_manifest.json",
        "h5_metric_registry.json",
        "h5_results.json",
        "manipulation_matrix.json",
        "provider_ledger.json",
        "figures/provider_agreement.png",
        "figures/provider_agreement.svg",
        "figures/manipulation_checks.png",
        "figures/manipulation_checks.svg",
    }
)


class H5ContractError(ValueError):
    """Raised when frozen H5 inputs or outputs violate their contract."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_GRAFT_FILE": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=True,
            capture_output=True,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise H5ContractError(
            f"git provenance command failed: git {' '.join(args)}"
        ) from exc
    return completed.stdout


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise H5ContractError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise H5ContractError(f"{label} must be an RFC3339 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise H5ContractError(f"{label} must be an RFC3339 UTC timestamp")
    return parsed


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise H5ContractError(f"cannot read {label}: {path}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise H5ContractError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise H5ContractError(f"{label} must be a JSON object")
    return value, payload, hashlib.sha256(payload).hexdigest()


def _relative_path(root: Path, raw: Any, label: str) -> tuple[Path, str]:
    relative = Path(str(raw or ""))
    if not str(raw or "").strip() or relative.is_absolute() or ".." in relative.parts:
        raise H5ContractError(f"{label} path must be a confined relative path")
    root_resolved = root.resolve(strict=True)
    resolved = (root_resolved / relative).resolve(strict=False)
    if not resolved.is_relative_to(root_resolved):
        raise H5ContractError(f"{label} path escapes the frozen root")
    return resolved, relative.as_posix()


def _require_envelope(
    record: Mapping[str, Any],
    label: str,
    *,
    provider: str | None = None,
) -> None:
    if record.get("simulated") is not True:
        raise H5ContractError(f"{label} simulated-data envelope is missing")
    for field in ("persona_id", "provider", "model_id"):
        if not isinstance(record.get(field), str) or not str(record[field]).strip():
            raise H5ContractError(f"{label} simulated-data envelope is missing {field}")
    if provider is not None and record.get("provider") != provider:
        raise H5ContractError(f"{label} provider envelope does not match {provider}")


def _internal_hash(record: Mapping[str, Any], field: str, label: str) -> None:
    supplied = record.get(field)
    core = {key: value for key, value in record.items() if key != field}
    if supplied != _canonical_sha(core):
        raise H5ContractError(f"{label} internal hash mismatch")


def _validate_persona_panel(
    raw_root: Path,
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    path, _ = _relative_path(
        raw_root, preparation.get("persona_panel_path"), "persona panel"
    )
    panel, _, _ = _read_json(path, "persona panel")
    _require_envelope(panel, "persona panel", provider="study_design")
    _internal_hash(panel, "persona_panel_sha256", "persona panel")
    if panel.get("persona_panel_sha256") != preparation.get("persona_panel_sha256"):
        raise H5ContractError("persona panel hash differs from preparation")
    if any(
        (
            panel.get("persona_seed_derivation_version")
            != preparation.get("persona_seed_derivation_version"),
            panel.get("frozen_pre_observation_utc")
            != preparation.get("frozen_pre_observation_utc"),
        )
    ):
        raise H5ContractError("persona panel differs from the frozen amendment")
    personas = panel.get("personas")
    if not isinstance(personas, list) or len(personas) != 50:
        raise H5ContractError("persona panel must contain exactly 50 personas")
    if panel.get("personas_sha256") != _canonical_sha(personas):
        raise H5ContractError("persona panel personas hash mismatch")
    if (
        panel.get("canonical_match") is not True
        or panel.get("canonical_personas_sha256") != panel.get("personas_sha256")
    ):
        raise H5ContractError("persona panel is not the canonical frozen grid")
    ids = [str(row.get("persona_id") or "") for row in personas if isinstance(row, Mapping)]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise H5ContractError("persona panel IDs must be 50 unique values")
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for row in personas:
        if not isinstance(row, Mapping):
            raise H5ContractError("persona panel row must be an object")
        pairs.setdefault(str(row.get("pair_id") or ""), []).append(row)
    if len(pairs) != 25 or any(
        len(rows) != 2
        or sorted(str(row.get("strength")) for row in rows) != ["strong", "weak"]
        or len({row.get("target_node") for row in rows}) != 1
        or len({row.get("failure_id") for row in rows}) != 1
        or len({row.get("seed") for row in rows}) != 1
        for rows in pairs.values()
    ):
        raise H5ContractError("persona panel does not contain 25 valid pairs")
    return panel


def _validate_manipulation_panel(
    raw_root: Path,
    preparation: Mapping[str, Any],
    persona_panel: Mapping[str, Any],
) -> dict[str, Any]:
    path = raw_root / "manipulation_panel.json"
    panel, _, _ = _read_json(path, "manipulation panel")
    _require_envelope(panel, "manipulation panel", provider="study_design")
    _internal_hash(panel, "panel_sha256", "manipulation panel")
    if any(
        (
            panel.get("panel_sha256") != preparation.get("panel_sha256"),
            panel.get("panel_sha256")
            != persona_panel.get("manipulation_panel_sha256"),
            panel.get("personas_sha256") != persona_panel.get("personas_sha256"),
            panel.get("frozen") is not True,
            panel.get("observation_started") is not False,
        )
    ):
        raise H5ContractError("manipulation panel differs from frozen preparation")
    annotations = panel.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 50:
        raise H5ContractError("manipulation panel must contain 50 annotations")
    annotation_hash = panel.get("annotation_map_sha256")
    snapshot_name = preparation.get("annotation_map_snapshot")
    normalized_map: dict[str, Any] = {
        "schema_version": "yher.llm_sim.annotation_map.v1",
        "items": {},
    }
    if annotation_hash is None:
        if snapshot_name is not None:
            raise H5ContractError("annotation snapshot exists without a frozen map")
    else:
        snapshot_path, _ = _relative_path(
            raw_root, snapshot_name, "annotation map snapshot"
        )
        snapshot, _, _ = _read_json(snapshot_path, "annotation map snapshot")
        _require_envelope(snapshot, "annotation map snapshot", provider="study_design")
        try:
            from experiments.llm_sim.panel import normalize_annotation_map

            normalized_map = normalize_annotation_map(
                snapshot.get("annotation_map") or {}
            )
        except (TypeError, ValueError) as exc:
            raise H5ContractError("annotation map snapshot is invalid") from exc
        if any(
            (
                snapshot.get("annotation_map_sha256") != annotation_hash,
                _canonical_sha(normalized_map) != annotation_hash,
                snapshot.get("panel_sha256") != panel.get("panel_sha256"),
            )
        ):
            raise H5ContractError("annotation map snapshot hash mismatch")
    coverage = _mapping_coverage(annotations, normalized_map)
    return {**panel, "_h5_mapping_coverage": coverage}


def _mapping_coverage(
    annotations: Sequence[Mapping[str, Any]],
    annotation_map: Mapping[str, Any],
) -> dict[str, Any]:
    map_items = annotation_map.get("items")
    if not isinstance(map_items, Mapping):
        map_items = {}
    declared = [row for row in annotations if row.get("mapping_status") == "mapped"]
    required_entries = 0
    covered_entries = 0
    covered_personas = 0
    for annotation in declared:
        failure_id = str(annotation.get("failure_id") or "")
        items = annotation.get("calibration_items")
        if not isinstance(items, list):
            items = []
        persona_complete = len(items) == 4
        required_entries += len(items)
        for item in items:
            if not isinstance(item, Mapping):
                persona_complete = False
                continue
            item_id = str(item.get("item_id") or "")
            target = str(item.get("target_option") or "").upper()
            failures = map_items.get(item_id)
            covered = bool(
                item.get("mapping_status") == "mapped"
                and isinstance(failures, Mapping)
                and str(failures.get(failure_id) or "").upper() == target
                and target
            )
            covered_entries += covered
            persona_complete = persona_complete and covered
        covered_personas += persona_complete
    complete = bool(
        len(declared) == 50
        and covered_personas == 50
        and required_entries == 200
        and covered_entries == 200
    )
    return {
        "declared_mapped_personas": len(declared),
        "covered_mapped_personas": covered_personas,
        "required_entries": required_entries,
        "covered_entries": covered_entries,
        "coverage_rate": (
            covered_entries / required_entries if required_entries else 0.0
        ),
        "complete": complete,
    }


def _validate_preparation(
    raw_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    preparation_path = raw_root / "preparation_manifest.json"
    preparation, _, preparation_file_sha = _read_json(
        preparation_path, "preparation manifest"
    )
    _require_envelope(preparation, "preparation manifest", provider="study_design")
    if any(
        (
            preparation.get("record_type") != "llm_sim_preparation_manifest",
            preparation.get("status") != "panel_frozen",
            preparation.get("provider_observations") != 0,
            preparation.get("persona_count") != 50,
            preparation.get("code_matches_head") is not True,
            preparation.get("analysis_plan_is_ancestor") is not True,
        )
    ):
        raise H5ContractError("preparation manifest is not a frozen pre-observation study")
    config_path = repo_root / "experiments/config/llm_sim_v1.json"
    try:
        frozen_config = load_frozen_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise H5ContractError("frozen S2 config violates the frozen amendment") from exc
    config = dict(frozen_config.raw)
    if frozen_config.sha256 != preparation.get("config_sha256"):
        raise H5ContractError("frozen config hash differs from preparation")
    if any(
        (
            tuple(config.get("providers") or ()) != FROZEN_PROVIDERS,
            config.get("persona_count") != 50,
            config.get("pair_count") != 25,
            tuple(config.get("arms") or ()) != ("A", "B"),
            config.get("max_items") != 15,
            config.get("study_seed") != preparation.get("study_seed"),
            config.get("run_id") != preparation.get("run_id"),
            config.get("analysis_plan_commit")
            != preparation.get("analysis_plan_commit"),
            config.get("h5_analysis_plan_commit")
            != preparation.get("h5_analysis_plan_commit"),
            config.get("h5_analysis_plan_sha256")
            != preparation.get("h5_analysis_plan_sha256"),
            config.get("h5_analysis_plan_committed_at_utc")
            != preparation.get("h5_analysis_plan_committed_at_utc"),
            config.get("persona_seed_derivation_version")
            != preparation.get("persona_seed_derivation_version"),
            config.get("prompt_version") != preparation.get("prompt_version"),
            config.get("frozen_pre_observation_utc")
            != preparation.get("frozen_pre_observation_utc"),
            preparation.get("h5_analysis_plan_verified") is not True,
        )
    ):
        raise H5ContractError("frozen config grid differs from preparation")
    try:
        verify_frozen_document_commit(
            repo_root,
            commit=str(preparation.get("h5_analysis_plan_commit") or ""),
            relative_path="experiments/h5_analysis_plan.md",
            sha256=str(preparation.get("h5_analysis_plan_sha256") or ""),
            committed_at_utc=str(
                preparation.get("h5_analysis_plan_committed_at_utc") or ""
            ),
            head=str(preparation.get("git_head") or ""),
        )
    except RuntimeError as exc:
        raise H5ContractError("frozen H5 amendment provenance is invalid") from exc
    code_rows = preparation.get("code_files")
    if not isinstance(code_rows, list) or not code_rows:
        raise H5ContractError("preparation code provenance is missing")
    expected_paths: set[str] = set()
    for pattern in CODE_PATTERNS:
        matches = tuple(
            path for path in repo_root.glob(pattern) if path.is_file()
        )
        if not matches:
            raise H5ContractError(f"frozen S2 code pattern is missing: {pattern}")
        expected_paths.update(path.relative_to(repo_root).as_posix() for path in matches)
    ordered_paths = tuple(sorted(expected_paths))
    claimed_paths = tuple(
        str(row.get("path") or "") if isinstance(row, Mapping) else ""
        for row in code_rows
    )
    if claimed_paths != ordered_paths:
        raise H5ContractError("preparation code provenance lacks the exact frozen scope")
    git_head = str(preparation.get("git_head") or "")
    if re.fullmatch(r"[0-9a-f]{40}", git_head) is None:
        raise H5ContractError("preparation git HEAD is invalid")
    _git(repo_root, "cat-file", "-e", f"{git_head}^{{commit}}")
    _git(repo_root, "merge-base", "--is-ancestor", git_head, "HEAD")
    digest_rows = []
    for row, relative in zip(code_rows, ordered_paths, strict=True):
        if not isinstance(row, Mapping):
            raise H5ContractError("preparation code provenance row is invalid")
        path, normalized = _relative_path(repo_root, relative, "code provenance")
        committed = _git(repo_root, "show", f"{git_head}:{normalized}")
        committed_sha = hashlib.sha256(committed).hexdigest()
        actual = _file_sha(path)
        if actual != committed_sha:
            raise H5ContractError(
                f"code hash differs from committed S2 git blob: {normalized}"
            )
        if (
            row.get("sha256") != actual
            or row.get("head_sha256") != committed_sha
            or row.get("matches_head") is not True
        ):
            raise H5ContractError(f"code git provenance mismatch: {normalized}")
        digest_rows.append({"path": normalized, "sha256": actual})
    if _canonical_sha(digest_rows) != preparation.get("code_sha256"):
        raise H5ContractError("aggregate code hash mismatch")
    official_inputs = preparation.get("official_inputs")
    if not isinstance(official_inputs, Mapping):
        raise H5ContractError("official input provenance is missing")
    if _canonical_sha(official_inputs) != preparation.get("official_input_sha256"):
        raise H5ContractError("official input aggregate hash mismatch")
    source_files = official_inputs.get("source_files")
    if not isinstance(source_files, list):
        raise H5ContractError("official source file provenance is missing")
    for row in source_files:
        if not isinstance(row, Mapping):
            raise H5ContractError("official source file row is invalid")
        path = Path(str(row.get("path") or "")).expanduser().resolve(strict=False)
        if not path.is_file():
            raise H5ContractError("official source file is missing")
        payload = path.read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != row.get("sha256")
            or len(payload) != row.get("bytes")
        ):
            raise H5ContractError("official source file hash mismatch")
    persona_panel = _validate_persona_panel(raw_root, preparation)
    manipulation_panel = _validate_manipulation_panel(
        raw_root, preparation, persona_panel
    )
    return (
        preparation,
        preparation_file_sha,
        persona_panel,
        manipulation_panel,
    )


def _validate_artifacts(
    raw_root: Path,
    manifest: Mapping[str, Any],
    provider: str,
) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise H5ContractError(f"{provider} provider artifact list is missing")
    if _canonical_sha(artifacts) != manifest.get("artifact_aggregate_sha256"):
        raise H5ContractError(f"{provider} provider artifact aggregate hash mismatch")
    seen: set[str] = set()
    validated = []
    records: list[Mapping[str, Any]] = []
    requested_model = str(manifest.get("model_id") or "")
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise H5ContractError(f"{provider} artifact row is invalid")
        path, relative = _relative_path(raw_root, row.get("path"), "provider artifact")
        if relative in seen:
            raise H5ContractError(f"{provider} artifact path is duplicated")
        seen.add(relative)
        record, _, sha = _read_json(path, f"{provider} artifact")
        if sha != row.get("sha256"):
            raise H5ContractError(f"{provider} artifact hash mismatch: {relative}")
        _require_envelope(record, f"{provider} artifact", provider=provider)
        if record.get("run_id") != manifest.get("run_id"):
            raise H5ContractError(f"{provider} artifact run envelope mismatch")
        record_type = record.get("record_type")
        if record_type == "llm_sim_provider_attempt":
            _validate_provider_attempt(record, manifest, provider)
        elif record_type == "llm_sim_excluded_response_accounting":
            _validate_drift_exclusion(record, manifest, provider)
        elif record_type == "llm_sim_calibration":
            events = record.get("events")
            if not isinstance(events, list) or any(
                not isinstance(event, Mapping)
                or not isinstance(event.get("correct"), bool)
                for event in events
            ):
                raise H5ContractError(
                    f"{provider} calibration artifact has invalid scored events"
                )
        elif record_type == "llm_sim_journey":
            if record.get("model_id") != requested_model:
                raise H5ContractError(f"{provider} artifact has unapproved model drift")
            _validate_journey_semantics(record, provider)
        elif record.get("model_id") != requested_model:
            raise H5ContractError(f"{provider} artifact has unapproved model drift")
        for field in ("record_type", "persona_id", "status", "arm"):
            if row.get(field) != record.get(field):
                raise H5ContractError(
                    f"{provider} artifact index differs on {field}: {relative}"
                )
        validated.append(dict(row))
        records.append(record)
    drift_attempts = {
        (
            record.get("persona_id"),
            record.get("phase"),
            record.get("arm"),
            record.get("position"),
            record.get("attempt_number"),
        ): record
        for record in records
        if record.get("record_type") == "llm_sim_provider_attempt"
        and record.get("failure_category") == "model_id_drift"
    }
    exclusion_keys = set()
    for record in records:
        if record.get("record_type") != "llm_sim_excluded_response_accounting":
            continue
        key = (
            record.get("persona_id"),
            record.get("phase"),
            record.get("arm"),
            record.get("position"),
            record.get("source_attempt_number"),
        )
        attempt = drift_attempts.get(key)
        if attempt is None or any(
            (
                record.get("requested_model_id") != attempt.get("requested_model_id"),
                record.get("returned_model_id") != attempt.get("returned_model_id"),
                record.get("item_id") != attempt.get("item_id"),
            )
        ):
            raise H5ContractError(f"{provider} drift exclusion lacks its typed attempt")
        exclusion_keys.add(key)
    if set(drift_attempts) != exclusion_keys:
        raise H5ContractError(f"{provider} typed drift attempt lacks exclusion accounting")
    return [dict(record) for record in records]


def _validate_journey_semantics(
    record: Mapping[str, Any], provider: str
) -> None:
    try:
        from engine import mastery
        from experiments.llm_sim.runner import (
            _belief_vector,
            _journey_transitions_are_valid,
        )

        events = record.get("events")
        final_belief = record.get("final_belief")
        max_items = record.get("max_items")
        actual_count = record.get("actual_administered_count")
        valid = bool(
            isinstance(events, list)
            and _belief_vector(final_belief)
            and isinstance(max_items, int)
            and not isinstance(max_items, bool)
            and 1 <= max_items <= 15
            and isinstance(actual_count, int)
            and not isinstance(actual_count, bool)
            and actual_count == len(events)
            and len(events) <= max_items
            and len(events) <= 15
        )
        status = record.get("status")
        terminal_reason = record.get("terminal_reason")
        if valid and status == "complete":
            valid = bool(
                1 <= len(events) <= 15
                and terminal_reason in {"confidence", "budget_exhausted"}
                and (
                    terminal_reason != "budget_exhausted"
                    or len(events) == max_items
                )
            )
        elif valid and status == "incomplete":
            valid = bool(
                (
                    terminal_reason == "structural_failure_no_items"
                    and len(events) == 0
                )
                or (
                    terminal_reason == "structural_failure_item_pool"
                    and 0 < len(events) < max_items
                )
            )
        else:
            valid = False
        if valid and events:
            valid = bool(
                all(isinstance(event, Mapping) for event in events)
                and _journey_transitions_are_valid(events)
                and list(final_belief) == list(events[-1].get("posterior_belief") or ())
            )
        elif valid:
            valid = list(final_belief) == mastery.UNIFORM.tolist()
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise H5ContractError(f"{provider} journey belief transition semantics are invalid")


def _usage_is_valid(record: Mapping[str, Any]) -> bool:
    usage = record.get("usage")
    try:
        return bool(
            isinstance(usage, Mapping)
            and int(usage.get("input_tokens", -1)) >= 0
            and int(usage.get("output_tokens", -1)) >= 0
            and math.isfinite(float(record.get("cost_yuan", -1.0)))
            and float(record.get("cost_yuan", -1.0)) >= 0.0
        )
    except (TypeError, ValueError):
        return False


def _validate_provider_attempt(
    record: Mapping[str, Any], manifest: Mapping[str, Any], provider: str
) -> None:
    requested = str(manifest.get("model_id") or "")
    returned = record.get("returned_model_id")
    status = record.get("status")
    try:
        common = all(
            (
                record.get("record_type") == "llm_sim_provider_attempt",
                record.get("schema_version") == "yher.llm_sim.provider_attempt.v1",
                record.get("run_id") == manifest.get("run_id"),
                record.get("requested_model_id") == requested,
                record.get("panel_sha256") == manifest.get("panel_sha256"),
                record.get("prompt_version") == manifest.get("prompt_version"),
                record.get("prompt_revision") == manifest.get("prompt_revision"),
                record.get("run_started_at_utc")
                == manifest.get("run_started_at_utc"),
                record.get("phase") in {"calibration", "journey"},
                record.get("arm") in {None, "A", "B"},
                int(record.get("position", -1)) >= 1,
                int(record.get("attempt_number", -1)) >= 1,
                int(record.get("retry_number", -1)) >= 0,
                bool(str(record.get("item_id") or "")),
                _usage_is_valid(record),
            )
        )
    except (TypeError, ValueError, OverflowError):
        common = False
    if status == "response":
        outcome = all(
            (
                returned == requested,
                record.get("model_id") == requested,
                record.get("response_received") is True,
                record.get("failure_category") is None,
                record.get("exclusion_type") is None,
            )
        )
    elif status == "model_drift":
        outcome = all(
            (
                isinstance(returned, str),
                bool(str(returned or "")),
                returned != requested,
                record.get("model_id") == returned,
                record.get("response_received") is True,
                record.get("failure_category") == "model_id_drift",
                record.get("exclusion_type") == "model_id_drift",
                manifest.get("failure_category") == "model_id_drift",
            )
        )
    elif status == "protocol_failure":
        outcome = all(
            (
                returned is None,
                record.get("model_id") == "missing-provider-model-id",
                isinstance(record.get("response_received"), bool),
                record.get("failure_category") == "protocol",
                record.get("exclusion_type") is None,
            )
        )
    elif status == "failed":
        outcome = all(
            (
                returned is None,
                record.get("model_id") == requested,
                record.get("response_received") is False,
                isinstance(record.get("failure_category"), str),
                record.get("exclusion_type") is None,
            )
        )
    else:
        outcome = False
    if not common or not outcome:
        raise H5ContractError(f"{provider} typed provider attempt is invalid")


def _validate_drift_exclusion(
    record: Mapping[str, Any], manifest: Mapping[str, Any], provider: str
) -> None:
    requested = str(manifest.get("model_id") or "")
    returned = record.get("returned_model_id")
    if not all(
        (
            record.get("schema_version")
            == "yher.llm_sim.model_drift_exclusion.v1",
            record.get("status") == "excluded_response",
            record.get("failure_category") == "model_id_drift",
            record.get("exclusion_type") == "model_id_drift",
            record.get("requested_model_id") == requested,
            record.get("requested_model") == requested,
            isinstance(returned, str),
            bool(str(returned or "")),
            returned != requested,
            record.get("returned_model") == returned,
            record.get("model_id") == returned,
            int(record.get("source_attempt_number", -1)) >= 1,
            record.get("panel_sha256") == manifest.get("panel_sha256"),
            record.get("prompt_revision") == manifest.get("prompt_revision"),
            _usage_is_valid(record),
        )
    ):
        raise H5ContractError(f"{provider} model-drift exclusion is invalid")


def _validate_provider_manifest(
    raw_root: Path,
    preparation: Mapping[str, Any],
    provider: str,
    prompt_revision: int,
) -> tuple[dict[str, Any], str, str]:
    suffix = f"__prompt-v{prompt_revision}" if prompt_revision else ""
    relative = f"providers/{provider}{suffix}.json"
    path, relative = _relative_path(raw_root, relative, "provider manifest")
    manifest, _, sha = _read_json(path, f"{provider} provider manifest")
    _require_envelope(manifest, f"{provider} provider manifest", provider=provider)
    expected = {
        "record_type": "llm_sim_provider_manifest",
        "run_id": preparation.get("run_id"),
        "prompt_revision": prompt_revision,
        "prompt_version": preparation.get("prompt_version"),
        "persona_seed_derivation_version": preparation.get(
            "persona_seed_derivation_version"
        ),
        "panel_sha256": preparation.get("panel_sha256"),
        "persona_panel_sha256": preparation.get("persona_panel_sha256"),
        "config_sha256": preparation.get("config_sha256"),
        "study_seed": preparation.get("study_seed"),
        "git_head": preparation.get("git_head"),
        "code_sha256": preparation.get("code_sha256"),
        "analysis_plan_commit": preparation.get("analysis_plan_commit"),
        "h5_analysis_plan_commit": preparation.get("h5_analysis_plan_commit"),
        "h5_analysis_plan_sha256": preparation.get("h5_analysis_plan_sha256"),
        "h5_analysis_plan_committed_at_utc": preparation.get(
            "h5_analysis_plan_committed_at_utc"
        ),
        "official_input_sha256": preparation.get("official_input_sha256"),
        "persona_count": 50,
        "arms": ["A", "B"],
        "max_items": 15,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise H5ContractError(f"{provider} provider manifest provenance mismatch")
    started = _utc_timestamp(
        manifest.get("run_started_at_utc"),
        f"{provider} run_started_at_utc",
    )
    frozen = _utc_timestamp(
        preparation.get("frozen_pre_observation_utc"),
        "frozen_pre_observation_utc",
    )
    if started <= frozen:
        raise H5ContractError(
            f"{provider} run_started_at_utc must be strictly after the frozen amendment"
        )
    records = _validate_artifacts(raw_root, manifest, provider)
    _validated_provider_accounting(records, manifest, provider)
    return manifest, sha, relative


def _validate_rewrite_decision(
    raw_root: Path,
    preparation: Mapping[str, Any],
    provider: str,
) -> tuple[dict[str, Any], str, str] | None:
    relative = f"calibration_decisions/{provider}.json"
    candidate = raw_root / relative
    if not candidate.is_file() and not candidate.is_symlink():
        return None
    path, relative = _relative_path(raw_root, relative, "calibration decision")
    decision, _, sha = _read_json(path, f"{provider} calibration decision")
    _require_envelope(decision, f"{provider} calibration decision", provider=provider)
    _internal_hash(decision, "decision_sha256", f"{provider} calibration decision")
    expected = {
        "record_type": "llm_sim_calibration_decision",
        "run_id": preparation.get("run_id"),
        "prompt_revision": 0,
        "prompt_version": preparation.get("prompt_version"),
        "persona_seed_derivation_version": preparation.get(
            "persona_seed_derivation_version"
        ),
        "panel_sha256": preparation.get("panel_sha256"),
        "persona_panel_sha256": preparation.get("persona_panel_sha256"),
        "config_sha256": preparation.get("config_sha256"),
        "study_seed": preparation.get("study_seed"),
        "h5_analysis_plan_commit": preparation.get("h5_analysis_plan_commit"),
        "h5_analysis_plan_sha256": preparation.get("h5_analysis_plan_sha256"),
        "h5_analysis_plan_committed_at_utc": preparation.get(
            "h5_analysis_plan_committed_at_utc"
        ),
        "arms": ["A", "B"],
        "max_items": 15,
    }
    if any(decision.get(key) != value for key, value in expected.items()):
        raise H5ContractError(f"{provider} calibration decision provenance mismatch")
    artifacts = decision.get("calibration_artifacts")
    if not isinstance(artifacts, list) or _canonical_sha(artifacts) != decision.get(
        "calibration_artifact_aggregate_sha256"
    ):
        raise H5ContractError(f"{provider} calibration decision artifact hash mismatch")
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise H5ContractError(f"{provider} calibration decision artifact is invalid")
        artifact_path, relative_artifact = _relative_path(
            raw_root, row.get("path"), "calibration decision artifact"
        )
        if _file_sha(artifact_path) != row.get("sha256"):
            raise H5ContractError(
                f"{provider} calibration decision artifact hash mismatch: {relative_artifact}"
            )
    return decision, sha, relative


def _require_v0_rewrite_pair(
    provider: str,
    v0: Mapping[str, Any],
    decision_result: tuple[dict[str, Any], str, str] | None,
) -> tuple[dict[str, Any], str, str]:
    if decision_result is None:
        raise H5ContractError(f"{provider} v0 rewrite lacks an immutable decision")
    decision, file_sha, relative = decision_result
    if any(
        (
            v0.get("status") != "calibration_rewrite_required",
            decision.get("status") != "calibration_rewrite_required",
            decision.get("model_id") != v0.get("model_id"),
            v0.get("calibration_decision_path") != relative,
            v0.get("calibration_decision_sha256") != decision.get("decision_sha256"),
        )
    ):
        raise H5ContractError(f"{provider} v0 and rewrite decision are inconsistent")
    return decision, file_sha, relative


def _write_immutable_json(path: Path, record: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        + b"\n"
    )
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise H5ContractError(f"immutable H5 artifact already differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _provider_invalid_evidence(
    raw_root: Path,
    provider: str,
) -> list[dict[str, Any]]:
    """Hash only the fixed provider-local lifecycle files that are safe to read."""

    lifecycle_relatives = (
        f"providers/{provider}.json",
        f"calibration_decisions/{provider}.json",
        f"providers/{provider}__prompt-v1.json",
    )
    root = raw_root.resolve(strict=True)
    evidence: dict[str, tuple[str | None, str]] = {}
    lifecycle_payloads: list[bytes] = []

    def bind(relative: str, *, include_missing: bool) -> bytes | None:
        rel = Path(relative)
        if not relative.strip() or rel.is_absolute() or ".." in rel.parts:
            return None
        path = raw_root / rel
        try:
            if path.is_symlink():
                target = os.readlink(path)
                evidence[rel.as_posix()] = (
                    hashlib.sha256(os.fsencode(target)).hexdigest(),
                    "symlink",
                )
                return None
            if not path.resolve(strict=False).is_relative_to(root):
                evidence[rel.as_posix()] = (
                    hashlib.sha256(os.fsencode(relative)).hexdigest(),
                    "escaped",
                )
                return None
            if not path.is_file():
                if include_missing:
                    evidence[rel.as_posix()] = (None, "missing")
                return None
            payload = path.read_bytes()
        except OSError:
            evidence[rel.as_posix()] = (None, "unreadable")
            return None
        evidence[rel.as_posix()] = (hashlib.sha256(payload).hexdigest(), "file")
        return payload

    for relative in lifecycle_relatives:
        payload = bind(relative, include_missing=False)
        if payload is not None:
            lifecycle_payloads.append(payload)
    for payload in lifecycle_payloads:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        for key in ("artifacts", "calibration_artifacts"):
            rows = value.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, Mapping) and isinstance(row.get("path"), str):
                    bind(str(row["path"]), include_missing=True)
    return [
        {
            "path": relative,
            "sha256": evidence[relative][0],
            "kind": evidence[relative][1],
        }
        for relative in sorted(evidence)
    ]


def _invalid_provider_row(
    raw_root: Path,
    provider: str,
    *,
    reason_code: str = "strict_provider_validation_failed",
) -> dict[str, Any]:
    evidence = _provider_invalid_evidence(raw_root, provider)
    by_path = {str(row["path"]): row.get("sha256") for row in evidence}
    v0_relative = f"providers/{provider}.json"
    decision_relative = f"calibration_decisions/{provider}.json"
    return {
        "provider": provider,
        "collection_status": "invalid_excluded",
        "prompt_revision": None,
        "manifest_path": v0_relative if v0_relative in by_path else None,
        "manifest_sha256": by_path.get(v0_relative),
        "model_id": None,
        "v0_manifest_path": v0_relative if v0_relative in by_path else None,
        "v0_manifest_sha256": by_path.get(v0_relative),
        "calibration_decision_path": (
            decision_relative if decision_relative in by_path else None
        ),
        "calibration_decision_sha256": by_path.get(decision_relative),
        "invalid_reason_code": reason_code,
        "invalid_evidence": evidence,
    }


def _invalid_provider_reason_code(error: Exception) -> str:
    if (
        isinstance(error, H5ContractError)
        and "calibration artifact has invalid scored events" in str(error)
    ):
        return "invalid_calibration_schema"
    return "strict_provider_validation_failed"


def _build_provider_collection_row(
    raw_root: Path,
    preparation: Mapping[str, Any],
    provider: str,
) -> dict[str, Any]:
    """Validate one provider, downgrading only provider-local failures."""

    v0_path = raw_root / "providers" / f"{provider}.json"
    try:
        decision_result = _validate_rewrite_decision(
            raw_root, preparation, provider
        )
        if not v0_path.is_file() and not v0_path.is_symlink():
            if decision_result is not None:
                raise H5ContractError(
                    f"{provider} calibration decision exists without a v0 manifest"
                )
            return {
                "provider": provider,
                "collection_status": "missing",
                "prompt_revision": None,
                "manifest_path": None,
                "manifest_sha256": None,
                "model_id": None,
                "v0_manifest_path": None,
                "v0_manifest_sha256": None,
                "calibration_decision_path": None,
                "calibration_decision_sha256": None,
            }
        v0, v0_sha, v0_relative = _validate_provider_manifest(
            raw_root, preparation, provider, 0
        )
        rewrite_marked = any(
            (
                v0.get("status") == "calibration_rewrite_required",
                decision_result is not None
                and decision_result[0].get("status")
                == "calibration_rewrite_required",
            )
        )
        if rewrite_marked:
            _require_v0_rewrite_pair(provider, v0, decision_result)
        selected_revision = 1 if rewrite_marked else 0
        if selected_revision == 1:
            selected_path = raw_root / "providers" / f"{provider}__prompt-v1.json"
            if not selected_path.is_file() and not selected_path.is_symlink():
                assert decision_result is not None
                return {
                    "provider": provider,
                    "collection_status": "missing_required_revision",
                    "prompt_revision": 1,
                    "manifest_path": None,
                    "manifest_sha256": None,
                    "model_id": v0.get("model_id"),
                    "v0_manifest_path": v0_relative,
                    "v0_manifest_sha256": v0_sha,
                    "calibration_decision_path": decision_result[2],
                    "calibration_decision_sha256": decision_result[1],
                }
            selected, selected_sha, selected_relative = _validate_provider_manifest(
                raw_root, preparation, provider, 1
            )
            if selected.get("model_id") != v0.get("model_id"):
                raise H5ContractError(f"{provider} prompt revision has model drift")
        else:
            selected, selected_sha, selected_relative = v0, v0_sha, v0_relative
        return {
            "provider": provider,
            "collection_status": "collected",
            "prompt_revision": selected_revision,
            "manifest_path": selected_relative,
            "manifest_sha256": selected_sha,
            "model_id": selected.get("model_id"),
            "v0_manifest_path": v0_relative,
            "v0_manifest_sha256": v0_sha,
            "calibration_decision_path": (
                decision_result[2] if decision_result is not None else None
            ),
            "calibration_decision_sha256": (
                decision_result[1] if decision_result is not None else None
            ),
        }
    except (H5ContractError, OSError, TypeError, ValueError, OverflowError) as error:
        return _invalid_provider_row(
            raw_root,
            provider,
            reason_code=_invalid_provider_reason_code(error),
        )


def _build_collection(
    raw_root: Path | str,
    repo_root: Path | str,
) -> dict[str, Any]:
    """Build the exact six-provider collection value without writing or locking it."""

    raw = Path(raw_root).expanduser().resolve(strict=True)
    repo = Path(repo_root).expanduser().resolve(strict=True)
    preparation, preparation_sha, persona_panel, panel = _validate_preparation(
        raw, repo
    )
    qwen_alias_paths = (
        raw / "providers/qwen.json",
        raw / "providers/qwen__prompt-v1.json",
        raw / "calibration_decisions/qwen.json",
    )
    if any(path.is_file() or path.is_symlink() for path in qwen_alias_paths):
        raise H5ContractError(
            "qwen is not a provider alias; use provider=tongyi and retain qwen only in model_id"
        )
    provider_rows = []
    for provider in FROZEN_PROVIDERS:
        provider_rows.append(
            _build_provider_collection_row(raw, preparation, provider)
        )
    core = {
        "simulated": True,
        "run_id": preparation.get("run_id"),
        "persona_id": "llm-sim-study:h5-collection",
        "provider": "study_design",
        "model_id": "no-provider-observation",
        "record_type": "llm_sim_h5_collection_manifest",
        "schema_version": "yher.h5.collection.v1",
        "preparation_path": "preparation_manifest.json",
        "preparation_sha256": preparation_sha,
        "panel_sha256": panel.get("panel_sha256"),
        "persona_panel_sha256": persona_panel.get("persona_panel_sha256"),
        "personas_sha256": persona_panel.get("personas_sha256"),
        "annotation_map_sha256": panel.get("annotation_map_sha256"),
        "config_sha256": preparation.get("config_sha256"),
        "study_seed": preparation.get("study_seed"),
        "s2_git_head": preparation.get("git_head"),
        "s2_code_sha256": preparation.get("code_sha256"),
        "official_input_sha256": preparation.get("official_input_sha256"),
        "analysis_plan_commit": preparation.get("analysis_plan_commit"),
        "h5_analysis_plan_commit": preparation.get("h5_analysis_plan_commit"),
        "h5_analysis_plan_sha256": preparation.get("h5_analysis_plan_sha256"),
        "h5_analysis_plan_committed_at_utc": preparation.get(
            "h5_analysis_plan_committed_at_utc"
        ),
        "frozen_providers": list(FROZEN_PROVIDERS),
        "providers": provider_rows,
    }
    collection = {**core, "collection_sha256": _canonical_sha(core)}
    return collection


def _collection_lock_value(collection: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "record_type": "llm_sim_h5_collection_lock",
        "schema_version": "yher.h5.collection_lock.v1",
        "run_id": collection.get("run_id"),
        "collection_sha256": collection.get("collection_sha256"),
        "preparation_sha256": collection.get("preparation_sha256"),
        "config_sha256": collection.get("config_sha256"),
        "s2_git_head": collection.get("s2_git_head"),
        "h5_analysis_plan_commit": collection.get("h5_analysis_plan_commit"),
    }
    return {**core, "collection_lock_sha256": _canonical_sha(core)}


def write_collection_lock(
    raw_root: Path | str,
    output_path: Path | str,
    *,
    repo_root: Path | str = Path("."),
) -> dict[str, Any]:
    """Write the one canonical lock that must be committed before finalization."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    destination = Path(output_path).expanduser().resolve(strict=False)
    canonical = (repo / COLLECTION_LOCK_RELATIVE).resolve(strict=False)
    if destination != canonical:
        raise H5ContractError(
            f"collection lock must use canonical path: {COLLECTION_LOCK_RELATIVE}"
        )
    lock = _collection_lock_value(_build_collection(raw_root, repo))
    _write_immutable_json(destination, lock)
    return lock


def _verify_committed_collection_lock(
    repo_root: Path,
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    path = repo_root / COLLECTION_LOCK_RELATIVE
    lock, payload, _ = _read_json(path, "committed H5 collection lock")
    required = {
        "record_type",
        "schema_version",
        "run_id",
        "collection_sha256",
        "preparation_sha256",
        "config_sha256",
        "s2_git_head",
        "h5_analysis_plan_commit",
        "collection_lock_sha256",
    }
    if set(lock) != required:
        raise H5ContractError("committed collection lock fields are not canonical")
    _internal_hash(lock, "collection_lock_sha256", "committed collection lock")
    expected = _collection_lock_value(collection)
    if lock != expected:
        raise H5ContractError("committed collection lock differs from H5 collection")
    relative = COLLECTION_LOCK_RELATIVE.as_posix()
    additions = tuple(
        line
        for line in _git(
            repo_root,
            "log",
            "--reverse",
            "--diff-filter=A",
            "--format=%H",
            "--",
            relative,
        )
        .decode("ascii")
        .splitlines()
        if line
    )
    if not additions or re.fullmatch(r"[0-9a-f]{40}", additions[0]) is None:
        raise H5ContractError("collection lock has no canonical add-commit anchor")
    anchor_commit = additions[0]
    _git(repo_root, "merge-base", "--is-ancestor", anchor_commit, "HEAD")
    anchored = _git(repo_root, "show", f"{anchor_commit}:{relative}")
    committed = _git(repo_root, "show", f"HEAD:{relative}")
    if anchored != payload or committed != payload:
        raise H5ContractError("collection lock is not committed unchanged at HEAD")
    return lock


def finalize_collection(
    raw_root: Path | str,
    output_path: Path | str,
    *,
    repo_root: Path | str = Path("."),
) -> dict[str, Any]:
    """Freeze the externally locked six-provider S2 collection."""

    repo = Path(repo_root).expanduser().resolve(strict=True)
    collection = _build_collection(raw_root, repo)
    _verify_committed_collection_lock(repo, collection)
    _write_immutable_json(Path(output_path), collection)
    return collection


def terminal_category(journey: Mapping[str, Any] | None) -> str:
    """Return M/P/C/U for a valid terminal journey and NC otherwise."""

    if not journey or journey.get("status") != "complete":
        return "NC"
    if journey.get("terminal_reason") not in {"confidence", "budget_exhausted"}:
        return "NC"
    belief = journey.get("final_belief")
    if not isinstance(belief, list) or len(belief) != len(PRODUCTION_STATES):
        raise H5ContractError("complete journey final_belief must have four states")
    try:
        values = np.asarray([float(value) for value in belief], dtype=float)
    except (TypeError, ValueError) as exc:
        raise H5ContractError("complete journey final_belief is not numeric") from exc
    if not np.all(np.isfinite(values)):
        raise H5ContractError("complete journey final_belief is not finite")
    return PRODUCTION_STATES[int(np.argmax(values))]


def fleiss_kappa(
    ratings: Sequence[Sequence[str]],
    *,
    categories: Sequence[str] = TERMINAL_CATEGORIES,
) -> float:
    """Compute Fleiss' kappa from a rectangular subject-by-rater matrix."""

    if not ratings:
        raise H5ContractError("Fleiss kappa requires at least one subject")
    category_values = tuple(str(value) for value in categories)
    if not category_values or len(set(category_values)) != len(category_values):
        raise H5ContractError("Fleiss categories must be unique and non-empty")
    rater_count = len(ratings[0])
    if rater_count < 2 or any(len(row) != rater_count for row in ratings):
        raise H5ContractError("Fleiss ratings must be rectangular with two raters")
    index = {value: offset for offset, value in enumerate(category_values)}
    counts = np.zeros((len(ratings), len(category_values)), dtype=float)
    for subject_index, row in enumerate(ratings):
        for rating in row:
            try:
                counts[subject_index, index[str(rating)]] += 1.0
            except KeyError as exc:
                raise H5ContractError(f"unknown Fleiss category: {rating!r}") from exc
    agreement = (
        np.square(counts).sum(axis=1) - rater_count
    ) / (rater_count * (rater_count - 1))
    observed = float(agreement.mean())
    proportions = counts.sum(axis=0) / (len(ratings) * rater_count)
    expected = float(np.square(proportions).sum())
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return float((observed - expected) / (1.0 - expected))


def pairwise_cohen_kappa(
    ratings_by_provider: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Compute sklearn Cohen kappa on each provider pair's common subjects."""

    providers = sorted(str(provider) for provider in ratings_by_provider)
    output: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for left in providers:
        output[left] = {}
        left_rows = ratings_by_provider[left]
        for right in providers:
            right_rows = ratings_by_provider[right]
            subjects = sorted(set(left_rows) & set(right_rows))
            if not subjects:
                value: float | None = None
            elif left == right:
                value = 1.0
            else:
                raw = float(
                    cohen_kappa_score(
                        [left_rows[subject] for subject in subjects],
                        [right_rows[subject] for subject in subjects],
                        labels=list(TERMINAL_CATEGORIES),
                    )
                )
                value = raw if math.isfinite(raw) else None
            output[left][right] = {
                "kappa": value,
                "n_subject": len(subjects),
            }
    return output


def persona_cluster_contrast_bootstrap(
    observations: Mapping[
        str, Mapping[str, Mapping[str, float | int]]
    ],
    *,
    persona_ids: Sequence[str],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap persona clusters while weighting included providers equally."""

    personas = tuple(str(value) for value in persona_ids)
    if not personas or len(set(personas)) != len(personas):
        raise H5ContractError("bootstrap persona_ids must be unique and non-empty")
    providers = tuple(sorted(str(value) for value in observations))
    if not providers:
        raise H5ContractError("bootstrap requires at least one provider")
    if int(iterations) < 1:
        raise H5ContractError("bootstrap iterations must be positive")
    index = {persona_id: offset for offset, persona_id in enumerate(personas)}
    numerators = np.zeros((len(providers), len(personas)), dtype=float)
    denominators = np.zeros_like(numerators)
    for provider_index, provider in enumerate(providers):
        for persona_id, row in observations[provider].items():
            if persona_id not in index:
                raise H5ContractError("bootstrap observation has an unknown persona")
            offset = index[persona_id]
            numerator = float(row.get("numerator", 0.0))
            denominator = float(row.get("denominator", 0.0))
            if not math.isfinite(numerator) or not math.isfinite(denominator):
                raise H5ContractError("bootstrap observation must be finite")
            if denominator < 0:
                raise H5ContractError("bootstrap denominator cannot be negative")
            numerators[provider_index, offset] += numerator
            denominators[provider_index, offset] += denominator
    totals = denominators.sum(axis=1)
    if np.any(totals <= 0):
        raise H5ContractError("every bootstrap provider needs observations")
    point = float(np.mean(numerators.sum(axis=1) / totals))
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(
        0,
        len(personas),
        size=(int(iterations), len(personas)),
    )
    provider_estimates = []
    for provider_index in range(len(providers)):
        sampled_numerator = numerators[provider_index][samples].sum(axis=1)
        sampled_denominator = denominators[provider_index][samples].sum(axis=1)
        estimate = np.divide(
            sampled_numerator,
            sampled_denominator,
            out=np.full(int(iterations), np.nan, dtype=float),
            where=sampled_denominator > 0,
        )
        provider_estimates.append(estimate)
    distribution = np.nanmean(np.vstack(provider_estimates), axis=0)
    if not np.all(np.isfinite(distribution)):
        raise H5ContractError("bootstrap produced an empty persona resample")
    low, high = np.quantile(distribution, (0.025, 0.975), method="linear")
    return {
        "point": point,
        "ci95": [float(low), float(high)],
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_unit": "persona_id",
        "provider_weighting": "equal",
        "provider_count": len(providers),
    }


def _validate_collection_manifest(
    collection_path: Path,
    raw_root: Path,
    repo_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    collection, _, _ = _read_json(collection_path, "H5 collection manifest")
    _require_envelope(collection, "H5 collection manifest", provider="study_design")
    _internal_hash(collection, "collection_sha256", "H5 collection manifest")
    _verify_committed_collection_lock(repo_root, collection)
    if tuple(collection.get("frozen_providers") or ()) != FROZEN_PROVIDERS:
        raise H5ContractError("H5 collection provider grid differs from the frozen six")
    preparation, preparation_sha, persona_panel, panel = _validate_preparation(
        raw_root, repo_root
    )
    expected_collection = {
        "preparation_path": "preparation_manifest.json",
        "preparation_sha256": preparation_sha,
        "panel_sha256": panel.get("panel_sha256"),
        "persona_panel_sha256": persona_panel.get("persona_panel_sha256"),
        "personas_sha256": persona_panel.get("personas_sha256"),
        "annotation_map_sha256": panel.get("annotation_map_sha256"),
        "config_sha256": preparation.get("config_sha256"),
        "study_seed": preparation.get("study_seed"),
        "s2_git_head": preparation.get("git_head"),
        "s2_code_sha256": preparation.get("code_sha256"),
        "official_input_sha256": preparation.get("official_input_sha256"),
        "analysis_plan_commit": preparation.get("analysis_plan_commit"),
        "h5_analysis_plan_commit": preparation.get("h5_analysis_plan_commit"),
        "h5_analysis_plan_sha256": preparation.get("h5_analysis_plan_sha256"),
        "h5_analysis_plan_committed_at_utc": preparation.get(
            "h5_analysis_plan_committed_at_utc"
        ),
    }
    if any(collection.get(key) != value for key, value in expected_collection.items()):
        raise H5ContractError("H5 collection differs from current frozen preparation")
    rows = collection.get("providers")
    if not isinstance(rows, list) or [row.get("provider") for row in rows] != list(
        FROZEN_PROVIDERS
    ):
        raise H5ContractError("H5 collection must list the exact six providers in order")
    validated_rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise H5ContractError("H5 collection provider row is invalid")
        provider = str(row["provider"])
        status = row.get("collection_status")
        if status == "collected":
            revision = int(row.get("prompt_revision", -1))
            if revision not in (0, 1):
                raise H5ContractError(f"{provider} collection revision is invalid")
            v0, v0_sha, v0_relative = _validate_provider_manifest(
                raw_root, preparation, provider, 0
            )
            if (
                row.get("v0_manifest_path") != v0_relative
                or row.get("v0_manifest_sha256") != v0_sha
            ):
                raise H5ContractError(f"{provider} collection v0 hash mismatch")
            manifest, manifest_sha, relative = _validate_provider_manifest(
                raw_root, preparation, provider, revision
            )
            if (
                row.get("manifest_path") != relative
                or row.get("manifest_sha256") != manifest_sha
                or row.get("model_id") != manifest.get("model_id")
            ):
                raise H5ContractError(f"{provider} collection manifest hash mismatch")
            decision_result = _validate_rewrite_decision(
                raw_root, preparation, provider
            )
            if revision == 1:
                _require_v0_rewrite_pair(provider, v0, decision_result)
                if (
                    decision_result is None
                    or row.get("calibration_decision_path") != decision_result[2]
                    or row.get("calibration_decision_sha256") != decision_result[1]
                    or manifest.get("model_id") != v0.get("model_id")
                ):
                    raise H5ContractError(
                        f"{provider} collection lacks immutable v0 rewrite decision"
                    )
            elif any(
                (
                    v0.get("status") == "calibration_rewrite_required",
                    decision_result is not None
                    and decision_result[0].get("status")
                    == "calibration_rewrite_required",
                )
            ):
                raise H5ContractError(f"{provider} revision zero bypasses v0 rewrite")
            validated_rows.append(
                {
                    **dict(row),
                    "manifest": manifest,
                    "v0_manifest": v0,
                }
            )
        elif status in {"missing", "missing_required_revision"}:
            if status == "missing" and any(
                row.get(key) is not None
                for key in ("manifest_path", "manifest_sha256", "prompt_revision")
            ):
                raise H5ContractError(f"{provider} missing row names an observation")
            if status == "missing":
                if (
                    (raw_root / "providers" / f"{provider}.json").is_file()
                    or (raw_root / "providers" / f"{provider}.json").is_symlink()
                    or (
                        raw_root / "calibration_decisions" / f"{provider}.json"
                    ).is_file()
                    or (
                        raw_root / "calibration_decisions" / f"{provider}.json"
                    ).is_symlink()
                ):
                    raise H5ContractError(f"{provider} missing row now has an observation")
                validated_rows.append(dict(row))
            else:
                v0, v0_sha, v0_relative = _validate_provider_manifest(
                    raw_root, preparation, provider, 0
                )
                decision_result = _validate_rewrite_decision(
                    raw_root, preparation, provider
                )
                _require_v0_rewrite_pair(provider, v0, decision_result)
                if (
                    row.get("v0_manifest_path") != v0_relative
                    or row.get("v0_manifest_sha256") != v0_sha
                    or decision_result is None
                    or row.get("calibration_decision_path") != decision_result[2]
                    or row.get("calibration_decision_sha256") != decision_result[1]
                    or (
                        raw_root / "providers" / f"{provider}__prompt-v1.json"
                    ).is_file()
                    or (
                        raw_root / "providers" / f"{provider}__prompt-v1.json"
                    ).is_symlink()
                ):
                    raise H5ContractError(
                        f"{provider} missing revision bindings changed"
                )
                validated_rows.append({**dict(row), "v0_manifest": v0})
        elif status == "invalid_excluded":
            expected_row = _build_provider_collection_row(
                raw_root, preparation, provider
            )
            if dict(row) != expected_row:
                raise H5ContractError(
                    f"{provider} invalid provider evidence binding changed"
                )
            validated_rows.append(dict(row))
        else:
            raise H5ContractError(f"{provider} collection status is invalid")
    return collection, preparation, persona_panel, panel, validated_rows


def _artifact_records(
    raw_root: Path,
    provider: str,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for row in manifest.get("artifacts") or ():
        path, _ = _relative_path(raw_root, row.get("path"), "provider artifact")
        record, _, sha = _read_json(path, f"{provider} provider artifact")
        if sha != row.get("sha256"):
            raise H5ContractError(f"{provider} provider artifact hash changed")
        output.append(record)
    return output


def _event_accounting(
    records: Sequence[Mapping[str, Any]],
    provider: str,
    model_id: str,
) -> dict[str, float | int]:
    attempts = [
        record
        for record in records
        if record.get("record_type") == "llm_sim_provider_attempt"
    ]
    exclusions = [
        record
        for record in records
        if record.get("record_type") == "llm_sim_excluded_response_accounting"
    ]
    if attempts:
        failure_reasons: dict[str, int] = {}
        for attempt in attempts:
            category = attempt.get("failure_category")
            if category is not None:
                key = str(category)
                failure_reasons[key] = failure_reasons.get(key, 0) + 1
        return {
            "requests": len(attempts),
            "responses": sum(
                attempt.get("response_received") is True for attempt in attempts
            ),
            "retries": sum(
                int(attempt.get("retry_number", -1)) > 0 for attempt in attempts
            ),
            "failed_requests": sum(
                attempt.get("status") != "response" for attempt in attempts
            ),
            "input_tokens": sum(
                int((attempt.get("usage") or {}).get("input_tokens") or 0)
                for attempt in attempts
            ),
            "output_tokens": sum(
                int((attempt.get("usage") or {}).get("output_tokens") or 0)
                for attempt in attempts
            ),
            "cost_yuan": round(
                sum(float(attempt.get("cost_yuan") or 0.0) for attempt in attempts),
                12,
            ),
            "drift_count": sum(
                attempt.get("failure_category") == "model_id_drift"
                for attempt in attempts
            ),
            "technical_failure_count": sum(
                attempt.get("status") in {"protocol_failure", "failed"}
                for attempt in attempts
            ),
            "failure_reasons": failure_reasons,
            "returned_model_ids": sorted(
                {
                    str(attempt["returned_model_id"])
                    for attempt in attempts
                    if attempt.get("returned_model_id")
                }
            ),
            "excluded_response_count": len(exclusions),
        }
    events: list[Mapping[str, Any]] = []
    for record in records:
        if record.get("record_type") == "llm_sim_excluded_response_accounting":
            events.append(record)
            continue
        raw_events = record.get("events")
        if raw_events is None:
            raw_events = []
        if not isinstance(raw_events, list):
            raise H5ContractError(f"{provider} artifact events must be a list")
        for event in raw_events:
            if not isinstance(event, Mapping):
                raise H5ContractError(f"{provider} artifact event is invalid")
            _require_envelope(event, f"{provider} event", provider=provider)
            if event.get("model_id") != model_id:
                raise H5ContractError(f"{provider} event has unapproved model drift")
            events.append(event)
    return {
        "requests": len(events),
        "responses": len(events),
        "retries": 0,
        "failed_requests": 0,
        "input_tokens": sum(
            int((event.get("usage") or {}).get("input_tokens") or 0)
            for event in events
        ),
        "output_tokens": sum(
            int((event.get("usage") or {}).get("output_tokens") or 0)
            for event in events
        ),
        "cost_yuan": round(
            sum(float(event.get("cost_yuan") or 0.0) for event in events), 12
        ),
        "drift_count": 0,
        "technical_failure_count": 0,
        "failure_reasons": {},
        "returned_model_ids": [model_id] if events else [],
        "excluded_response_count": len(exclusions),
    }


def _empty_provider_accounting(status: str) -> dict[str, Any]:
    return {
        "requests": 0,
        "responses": 0,
        "retries": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_yuan": 0.0,
        "failed_requests": 0,
        "drift_count": 0,
        "failure_reasons": {},
        "excluded_response_count": 0,
        "accounting_status": status,
    }


def _invalid_provider_accounting(
    raw_root: Path,
    collection_row: Mapping[str, Any],
    provider: str,
) -> dict[str, Any]:
    """Recover incurred cost only from attempt bytes bound into the collection."""

    bound = {
        str(row.get("path") or ""): row
        for row in collection_row.get("invalid_evidence") or ()
        if isinstance(row, Mapping)
        and row.get("kind") == "file"
        and isinstance(row.get("sha256"), str)
    }
    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()
    invalid_attempt_count = 0
    for relative in (
        f"providers/{provider}.json",
        f"providers/{provider}__prompt-v1.json",
    ):
        manifest_evidence = bound.get(relative)
        if manifest_evidence is None:
            continue
        try:
            manifest_path, _ = _relative_path(raw_root, relative, "invalid manifest")
            if manifest_path.is_symlink():
                continue
            manifest, _, manifest_sha = _read_json(
                manifest_path, f"{provider} invalid provider manifest"
            )
            if manifest_sha != manifest_evidence.get("sha256"):
                continue
            _require_envelope(
                manifest, f"{provider} invalid provider manifest", provider=provider
            )
        except (H5ContractError, OSError, TypeError, ValueError, OverflowError):
            continue
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if (
                not isinstance(artifact, Mapping)
                or artifact.get("record_type") != "llm_sim_provider_attempt"
            ):
                continue
            artifact_relative = str(artifact.get("path") or "")
            if artifact_relative in seen:
                continue
            artifact_evidence = bound.get(artifact_relative)
            if artifact_evidence is None:
                invalid_attempt_count += 1
                continue
            try:
                path, normalized = _relative_path(
                    raw_root, artifact_relative, "invalid provider attempt"
                )
                if path.is_symlink():
                    raise H5ContractError("invalid provider attempt cannot be a symlink")
                record, _, sha = _read_json(path, f"{provider} invalid provider attempt")
                if any(
                    (
                        normalized != artifact_relative,
                        sha != artifact.get("sha256"),
                        sha != artifact_evidence.get("sha256"),
                    )
                ):
                    raise H5ContractError("invalid provider attempt hash differs")
                _require_envelope(
                    record, f"{provider} invalid provider attempt", provider=provider
                )
                _validate_provider_attempt(record, manifest, provider)
            except (H5ContractError, OSError, TypeError, ValueError, OverflowError):
                invalid_attempt_count += 1
                continue
            seen.add(artifact_relative)
            attempts.append(record)
    if not attempts:
        return _empty_provider_accounting("unavailable")
    accounting = _event_accounting(attempts, provider, "")
    status = (
        "validated_manifest_bound_attempts"
        if invalid_attempt_count == 0
        else "partial_validated_manifest_bound_attempts"
    )
    return {
        "requests": int(accounting["requests"]),
        "responses": int(accounting["responses"]),
        "retries": int(accounting["retries"]),
        "input_tokens": int(accounting["input_tokens"]),
        "output_tokens": int(accounting["output_tokens"]),
        "cost_yuan": float(accounting["cost_yuan"]),
        "failed_requests": int(accounting["failed_requests"]),
        "drift_count": int(accounting["drift_count"]),
        "failure_reasons": dict(accounting["failure_reasons"]),
        "excluded_response_count": int(accounting["excluded_response_count"]),
        "accounting_status": status,
    }


def _validated_provider_accounting(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    provider: str,
) -> dict[str, Any]:
    """Reconstruct billing from typed attempts and verify the manifest claim."""

    model_id = str(manifest.get("model_id") or "")
    accounting = _event_accounting(records, provider, model_id)
    try:
        response_attempts = {
            (
                str(record.get("persona_id") or ""),
                str(record.get("phase") or ""),
                record.get("arm"),
                int(record.get("position", -1)),
                str(record.get("item_id") or ""),
            )
            for record in records
            if record.get("record_type") == "llm_sim_provider_attempt"
            and record.get("status") == "response"
        }
        for record in records:
            record_type = record.get("record_type")
            if record_type not in {"llm_sim_calibration", "llm_sim_journey"}:
                continue
            phase = "calibration" if record_type == "llm_sim_calibration" else "journey"
            arm = record.get("arm") if phase == "journey" else None
            for event in record.get("events") or ():
                key = (
                    str(record.get("persona_id") or ""),
                    phase,
                    arm,
                    int(event.get("position", -1)),
                    str(event.get("item_id") or ""),
                )
                if key not in response_attempts:
                    raise H5ContractError(
                        f"{provider} outcome event lacks a manifest-bound response attempt"
                    )
        manifest_accounting = manifest.get("accounting")
        if not isinstance(manifest_accounting, Mapping):
            raise H5ContractError(f"{provider} manifest accounting is missing")
        for key in (
            "requests",
            "responses",
            "retries",
            "failed_requests",
            "input_tokens",
            "output_tokens",
        ):
            if int(manifest_accounting.get(key, -1)) != int(accounting[key]):
                raise H5ContractError(f"{provider} manifest {key} accounting mismatch")
        claimed_cost = float(manifest_accounting.get("cost_yuan", -1.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise H5ContractError(f"{provider} manifest accounting is invalid") from error
    if not math.isfinite(claimed_cost) or not math.isclose(
        claimed_cost,
        float(accounting["cost_yuan"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise H5ContractError(f"{provider} manifest cost accounting mismatch")
    if bool(manifest_accounting.get("model_drift_detected")) != bool(
        accounting["drift_count"]
    ):
        raise H5ContractError(f"{provider} manifest drift accounting mismatch")
    if sorted(manifest_accounting.get("returned_model_ids") or []) != accounting[
        "returned_model_ids"
    ]:
        raise H5ContractError(f"{provider} manifest returned-model accounting mismatch")
    ledger = {
        "requests": int(accounting["requests"]),
        "responses": int(accounting["responses"]),
        "retries": int(accounting["retries"]),
        "failed_requests": int(accounting["failed_requests"]),
        "input_tokens": int(accounting["input_tokens"]),
        "output_tokens": int(accounting["output_tokens"]),
        "cost_yuan": float(accounting["cost_yuan"]),
        "drift_count": int(accounting["drift_count"]),
        "failure_reasons": dict(accounting["failure_reasons"]),
        "excluded_response_count": int(accounting["excluded_response_count"]),
        "accounting_status": "validated_manifest_bound_attempts",
    }
    if any(int(ledger[key]) < 0 for key in ("requests", "responses", "retries")):
        raise H5ContractError(f"{provider} manifest accounting cannot be negative")
    return ledger


def _merge_validated_provider_accounting(
    ledgers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not ledgers:
        return _empty_provider_accounting("unavailable")
    failure_reasons: dict[str, int] = {}
    for ledger in ledgers:
        for reason, count in (ledger.get("failure_reasons") or {}).items():
            key = str(reason)
            failure_reasons[key] = failure_reasons.get(key, 0) + int(count)
    merged = {
        key: sum(int(ledger.get(key) or 0) for ledger in ledgers)
        for key in (
            "requests",
            "responses",
            "retries",
            "input_tokens",
            "output_tokens",
            "failed_requests",
            "drift_count",
            "excluded_response_count",
        )
    }
    return {
        **merged,
        "cost_yuan": round(
            sum(float(ledger.get("cost_yuan") or 0.0) for ledger in ledgers),
            12,
        ),
        "failure_reasons": failure_reasons,
        "accounting_status": "validated_manifest_bound_attempts",
    }


def _validated_collection_row_accounting(
    raw_root: Path,
    collection_row: Mapping[str, Any],
    provider: str,
) -> dict[str, Any]:
    status = collection_row.get("collection_status")
    manifests: list[Mapping[str, Any]] = []
    if status == "missing_required_revision":
        v0_manifest = collection_row.get("v0_manifest")
        if not isinstance(v0_manifest, Mapping):
            raise H5ContractError(f"{provider} missing revision lacks its bound v0 manifest")
        manifests.append(v0_manifest)
    elif status == "collected":
        if collection_row.get("prompt_revision") == 1:
            v0_manifest = collection_row.get("v0_manifest")
            if not isinstance(v0_manifest, Mapping):
                raise H5ContractError(f"{provider} prompt revision lacks its bound v0 manifest")
            manifests.append(v0_manifest)
        manifest = collection_row.get("manifest")
        if not isinstance(manifest, Mapping):
            raise H5ContractError(f"{provider} collected row lacks its selected manifest")
        manifests.append(manifest)
    else:
        raise H5ContractError(f"{provider} collection row has no validated accounting")
    ledgers = []
    for manifest in manifests:
        records = _artifact_records(raw_root, provider, manifest)
        ledgers.append(_validated_provider_accounting(records, manifest, provider))
    return _merge_validated_provider_accounting(ledgers)


def _excluded_provider_analysis_result(
    collection_row: Mapping[str, Any],
    personas: Sequence[Mapping[str, Any]],
    *,
    mapping_available: bool,
    ledger: Mapping[str, Any],
    reason_code: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, dict[str, float | int]],
    dict[str, Any],
]:
    provider = str(collection_row["provider"])
    status = "invalid_excluded" if reason_code else str(
        collection_row.get("collection_status")
    )
    subjects = tuple(
        f"{row['persona_id']}|{arm}" for row in personas for arm in ("A", "B")
    )
    manifest = collection_row.get("manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    return (
        {
            "provider": provider,
            "collection_status": status,
            "provider_status": manifest.get("status"),
            "failure_category": manifest.get("failure_category"),
            "exclusion_reason": manifest.get("exclusion_reason"),
            "prompt_revision": collection_row.get("prompt_revision"),
            "model_id": collection_row.get("model_id"),
            "manifest_path": collection_row.get("manifest_path"),
            "manifest_sha256": collection_row.get("manifest_sha256"),
            "invalid_reason_code": reason_code
            or collection_row.get("invalid_reason_code"),
            "completed_by_arm": {"A": 0, "B": 0},
            "observed_journeys": 0,
            "missing_journeys": 100,
            "structural_incomplete_journeys": 0,
            "weak_accuracy": None,
            "strong_accuracy": None,
            "weak_accuracy_gate": False,
            "strong_accuracy_gate": False,
            "target_gate_measurable": mapping_available,
            "target_gate_pass": False,
            "target_contrast": None,
            "formal_design_match": False,
            "calibration_grid_valid": False,
            "calibration_structural_personas": 50,
            "drift_count": int(ledger.get("drift_count") or 0),
            "technical_valid": False,
            "raw_complete": False,
            "qualifies": False,
            "exclusion_reasons": [status],
        },
        {subject: "NC" for subject in subjects},
        {},
        dict(ledger),
    )


def _provider_analysis(
    *,
    raw_root: Path,
    collection_row: Mapping[str, Any],
    preparation: Mapping[str, Any],
    personas: Sequence[Mapping[str, Any]],
    panel: Mapping[str, Any],
    mapping_available: bool,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, dict[str, float | int]],
    dict[str, float | int],
]:
    provider = str(collection_row["provider"])
    persona_ids = tuple(str(row["persona_id"]) for row in personas)
    persona_set = set(persona_ids)
    if collection_row.get("collection_status") != "collected":
        if collection_row.get("collection_status") == "invalid_excluded":
            ledger = _invalid_provider_accounting(raw_root, collection_row, provider)
        elif collection_row.get("collection_status") == "missing_required_revision":
            ledger = _validated_collection_row_accounting(
                raw_root, collection_row, provider
            )
        else:
            ledger = _empty_provider_accounting("not_collected")
        return _excluded_provider_analysis_result(
            collection_row,
            personas,
            mapping_available=mapping_available,
            ledger=ledger,
        )
    manifest = collection_row["manifest"]
    model_id = str(manifest.get("model_id") or "")
    records = _artifact_records(raw_root, provider, manifest)
    ledger = _validated_collection_row_accounting(raw_root, collection_row, provider)
    accounting = _event_accounting(records, provider, model_id)
    response_attempts = {
        (
            str(record.get("persona_id") or ""),
            str(record.get("phase") or ""),
            record.get("arm"),
            int(record.get("position", -1)),
            str(record.get("item_id") or ""),
        )
        for record in records
        if record.get("record_type") == "llm_sim_provider_attempt"
        and record.get("status") == "response"
    }

    journeys: dict[tuple[str, str], Mapping[str, Any]] = {}
    calibration_records: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        record_type = record.get("record_type")
        if record_type == "llm_sim_calibration":
            calibration_records.setdefault(
                str(record.get("persona_id") or ""), []
            ).append(record)
            continue
        if record_type != "llm_sim_journey":
            continue
        persona_id = str(record.get("persona_id") or "")
        arm = str(record.get("arm") or "")
        if persona_id not in persona_set or arm not in {"A", "B"}:
            raise H5ContractError(f"{provider} journey is outside the frozen persona/arm grid")
        key = (persona_id, arm)
        if key in journeys:
            raise H5ContractError(f"{provider} journey grid contains a duplicate")
        expected = {
            "run_id": preparation.get("run_id"),
            "provider": provider,
            "model_id": model_id,
            "max_items": 15,
            "panel_sha256": preparation.get("panel_sha256"),
            "config_sha256": preparation.get("config_sha256"),
            "persona_panel_sha256": preparation.get("persona_panel_sha256"),
            "study_seed": preparation.get("study_seed"),
            "analysis_plan_commit": preparation.get("analysis_plan_commit"),
            "prompt_revision": collection_row.get("prompt_revision"),
        }
        if any(record.get(field) != value for field, value in expected.items()):
            raise H5ContractError(f"{provider} journey provenance mismatch")
        journeys[key] = record
    annotations = {
        str(row.get("persona_id") or ""): row
        for row in panel.get("annotations") or ()
        if isinstance(row, Mapping)
    }
    calibration_events: list[Mapping[str, Any]] = []
    structural_calibration = sum(
        len(values)
        for persona_id, values in calibration_records.items()
        if persona_id not in persona_set
    )
    for persona in personas:
        persona_id = str(persona["persona_id"])
        records_for_persona = calibration_records.get(persona_id, [])
        annotation = annotations.get(persona_id) or {}
        expected_items = list(annotation.get("calibration_items") or ())
        valid = len(records_for_persona) == 1 and len(expected_items) == 4
        if valid:
            record = records_for_persona[0]
            events = record.get("events")
            valid = bool(
                record.get("status") == "complete"
                and record.get("run_id") == preparation.get("run_id")
                and record.get("provider") == provider
                and record.get("model_id") == model_id
                and record.get("strength") == persona.get("strength")
                and record.get("panel_sha256") == preparation.get("panel_sha256")
                and record.get("prompt_revision")
                == collection_row.get("prompt_revision")
                and isinstance(events, list)
                and len(events) == 4
            )
        else:
            events = []
        if valid:
            expected_ids = [str(item.get("item_id") or "") for item in expected_items]
            observed_ids = [str(event.get("item_id") or "") for event in events]
            valid = bool(
                len(set(expected_ids)) == 4
                and observed_ids == expected_ids
                and [event.get("position") for event in events] == [1, 2, 3, 4]
                and all(
                    isinstance(event, Mapping)
                    and event.get("record_type") == "llm_sim_calibration_attempt"
                    and event.get("run_id") == preparation.get("run_id")
                    and event.get("provider") == provider
                    and event.get("model_id") == model_id
                    and event.get("persona_id") == persona_id
                    and event.get("strength") == persona.get("strength")
                    for event in events
                )
            )
        if valid:
            valid_attempts = all(
                (
                    persona_id,
                    "calibration",
                    None,
                    int(event.get("position", -1)),
                    str(event.get("item_id") or ""),
                )
                in response_attempts
                for event in events
            )
            if not valid_attempts:
                raise H5ContractError(
                    f"{provider} calibration outcome lacks a response attempt"
                )
        if valid:
            calibration_events.extend(events)
        else:
            structural_calibration += 1
    completed_by_arm = {
        arm: sum(
            record.get("status") == "complete"
            for (persona_id, record_arm), record in journeys.items()
            if record_arm == arm
        )
        for arm in ("A", "B")
    }
    structural = sum(
        record.get("status") != "complete" for record in journeys.values()
    )
    ratings = {
        f"{persona_id}|{arm}": terminal_category(journeys.get((persona_id, arm)))
        for persona_id in persona_ids
        for arm in ("A", "B")
    }
    weak_events = [
        event for event in calibration_events if event.get("strength") == "weak"
    ]
    strong_events = [
        event for event in calibration_events if event.get("strength") == "strong"
    ]
    weak_accuracy = (
        sum(event.get("correct") is True for event in weak_events) / len(weak_events)
        if weak_events
        else None
    )
    strong_accuracy = (
        sum(event.get("correct") is True for event in strong_events)
        / len(strong_events)
        if strong_events
        else None
    )
    weak_pass = weak_accuracy is not None and weak_accuracy < 0.4
    strong_pass = strong_accuracy is not None and strong_accuracy > 0.75
    contrast_by_persona: dict[str, dict[str, float | int]] = {}
    for event in weak_events:
        hit = event.get("target_misconception_hit")
        baseline = event.get("random_wrong_option_baseline")
        if event.get("correct") is not False or not isinstance(hit, bool) or baseline is None:
            continue
        persona_id = str(event.get("persona_id") or "")
        row = contrast_by_persona.setdefault(
            persona_id, {"numerator": 0.0, "denominator": 0}
        )
        row["numerator"] = float(row["numerator"]) + float(hit) - float(baseline)
        row["denominator"] = int(row["denominator"]) + 1
    if contrast_by_persona:
        provider_contrast = sum(
            float(row["numerator"]) for row in contrast_by_persona.values()
        ) / sum(int(row["denominator"]) for row in contrast_by_persona.values())
        provider_bootstrap = persona_cluster_contrast_bootstrap(
            {provider: contrast_by_persona},
            persona_ids=persona_ids,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED,
        )
        target_pass = provider_bootstrap["ci95"][0] > 0.0
    else:
        provider_contrast = None
        target_pass = False
    formal = True
    calibration_grid_valid = structural_calibration == 0
    drift_count = int(accounting["drift_count"])
    technical_valid = bool(
        calibration_grid_valid
        and drift_count == 0
        and int(accounting["technical_failure_count"]) == 0
    )
    completion_pass = all(completed_by_arm[arm] >= 45 for arm in ("A", "B"))
    reasons = []
    if not completion_pass:
        reasons.append("completion")
    if not formal:
        reasons.append("formal_design")
    if not calibration_grid_valid:
        reasons.append("calibration_structure")
    if not weak_pass:
        reasons.append("weak_accuracy")
    if not strong_pass:
        reasons.append("strong_accuracy")
    if mapping_available and not target_pass:
        reasons.append("target_misconception")
    if drift_count:
        reasons.append("model_drift")
    elif not technical_valid:
        reasons.append("technical_failure")
    qualifies = not reasons and mapping_available
    matrix = {
        "provider": provider,
        "collection_status": "collected",
        "provider_status": manifest.get("status"),
        "failure_category": manifest.get("failure_category"),
        "exclusion_reason": manifest.get("exclusion_reason"),
        "prompt_revision": collection_row.get("prompt_revision"),
        "model_id": model_id,
        "completed_by_arm": completed_by_arm,
        "observed_journeys": len(journeys),
        "missing_journeys": 100 - len(journeys),
        "structural_incomplete_journeys": structural,
        "weak_accuracy": weak_accuracy,
        "strong_accuracy": strong_accuracy,
        "weak_accuracy_gate": weak_pass,
        "strong_accuracy_gate": strong_pass,
        "target_gate_measurable": mapping_available and bool(contrast_by_persona),
        "target_gate_pass": target_pass,
        "target_contrast": provider_contrast,
        "formal_design_match": formal,
        "calibration_grid_valid": calibration_grid_valid,
        "calibration_structural_personas": structural_calibration,
        "drift_count": drift_count,
        "technical_valid": technical_valid,
        "raw_complete": (
            completion_pass and len(journeys) == 100 and technical_valid
        ),
        "qualifies": qualifies,
        "exclusion_reasons": reasons,
    }
    return matrix, ratings, contrast_by_persona, ledger


def _analysis_provenance() -> dict[str, Any]:
    path = Path(__file__).resolve()
    root = path.parents[1]
    relative = path.relative_to(root).as_posix()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    head_sha = None
    if commit:
        try:
            payload = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
            head_sha = hashlib.sha256(payload).hexdigest()
        except (OSError, subprocess.CalledProcessError):
            head_sha = None
    working_sha = _file_sha(path)
    return {
        "analysis_commit": commit,
        "analysis_code_sha256": working_sha,
        "analysis_code_files": [
            {
                "path": relative,
                "sha256": working_sha,
                "head_sha256": head_sha,
                "matches_head": head_sha == working_sha,
            }
        ],
    }


def _h5_analysis_provenance(
    repo_root: Path,
    verified: Mapping[str, object] | None,
) -> dict[str, Any]:
    if verified is None:
        return _analysis_provenance()
    claimed = dict(verified)
    try:
        verify_analysis_provenance(repo_root, claimed)
        files = claimed["analysis_code_files"]
        commit = claimed["analysis_commit"]
        if not isinstance(files, Mapping):
            raise DatasetContractError("analysis code files are missing")
        h5_sha = files.get("analysis/h5.py")
        if (
            not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or not isinstance(h5_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", h5_sha) is None
        ):
            raise DatasetContractError("analysis/h5.py provenance is invalid")
    except (DatasetContractError, KeyError) as exc:
        raise H5ContractError("verified H5 analysis provenance is invalid") from exc
    return {
        "analysis_commit": commit,
        "analysis_code_sha256": h5_sha,
        "analysis_code_files": [
            {
                "path": "analysis/h5.py",
                "sha256": h5_sha,
                "head_sha256": h5_sha,
                "matches_head": True,
            }
        ],
    }


def _write_immutable_value(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise H5ContractError(f"immutable H5 artifact already differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    suffix = path.suffix.lower()
    metadata = (
        {"Date": None, "Creator": "YHer H5 deterministic analysis"}
        if suffix == ".svg"
        else {"Software": "YHer H5 deterministic analysis"}
    )
    try:
        figure.savefig(
            temporary,
            format=suffix.lstrip("."),
            dpi=160,
            metadata=metadata,
        )
        payload = temporary.read_bytes()
        if path.exists():
            if path.read_bytes() != payload:
                raise H5ContractError(f"deterministic figure drift: {path.name}")
        else:
            os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _render_figures(
    output_root: Path,
    provider_matrix: Sequence[Mapping[str, Any]],
    agreement: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, str]]]:
    matplotlib.rcParams["svg.hashsalt"] = "yher-h5-v1"
    figure_dir = output_root / "figures"
    providers = list(agreement.get("providers") or ())
    pairwise = agreement.get("pairwise") or {}
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    if providers:
        values = np.full((len(providers), len(providers)), np.nan)
        for i, left in enumerate(providers):
            for j, right in enumerate(providers):
                value = (pairwise.get(left) or {}).get(right, {}).get("kappa")
                if value is not None:
                    values[i, j] = float(value)
        image = axis.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        figure.colorbar(image, ax=axis, label="Cohen kappa")
        axis.set_xticks(range(len(providers)), providers, rotation=35, ha="right")
        axis.set_yticks(range(len(providers)), providers)
        for i, left in enumerate(providers):
            for j, right in enumerate(providers):
                cell = (pairwise.get(left) or {}).get(right, {})
                value = cell.get("kappa")
                text = "NA" if value is None else f"{float(value):.2f}"
                axis.text(j, i, f"{text}\nn={cell.get('n_subject', 0)}", ha="center", va="center", fontsize=7)
    else:
        axis.text(0.5, 0.5, "No two-provider agreement set", ha="center", va="center")
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title(f"Provider agreement ({agreement.get('scope')})")
    agreement_paths = []
    for suffix in ("png", "svg"):
        path = figure_dir / f"provider_agreement.{suffix}"
        _save_figure(figure, path)
        agreement_paths.append(path)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), constrained_layout=True)
    collected = [row for row in provider_matrix if row.get("collection_status") == "collected"]
    if collected:
        labels = [str(row["provider"]) for row in collected]
        x = np.arange(len(labels))
        axes[0].bar(x - 0.18, [float(row.get("weak_accuracy") or 0.0) for row in collected], width=0.36, color="#D55E00", label="Weak")
        axes[0].bar(x + 0.18, [float(row.get("strong_accuracy") or 0.0) for row in collected], width=0.36, color="#0072B2", label="Strong")
        axes[0].axhline(0.4, color="#555555", linestyle="--", linewidth=0.8)
        axes[0].axhline(0.75, color="#555555", linestyle=":", linewidth=0.8)
        axes[0].set_xticks(x, labels, rotation=35, ha="right")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_ylabel("Calibration accuracy")
        axes[0].legend(frameon=False)
        contrasts = [row.get("target_contrast") for row in collected]
        axes[1].bar(x, [float(value) if value is not None else 0.0 for value in contrasts], color="#009E73")
        axes[1].axhline(0.0, color="#333333", linewidth=0.8)
        axes[1].set_xticks(x, labels, rotation=35, ha="right")
        axes[1].set_ylabel("Target-option contrast")
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No collected provider observations", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        f"Manipulation checks: {hypothesis.get('analysis_status')}"
    )
    manipulation_paths = []
    for suffix in ("png", "svg"):
        path = figure_dir / f"manipulation_checks.{suffix}"
        _save_figure(figure, path)
        manipulation_paths.append(path)
    plt.close(figure)

    def reference(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": _file_sha(path),
        }

    return {
        "FIG_PROVIDER_AGREEMENT": {
            path.suffix.lstrip("."): reference(path) for path in agreement_paths
        },
        "FIG_MANIPULATION_CHECKS": {
            path.suffix.lstrip("."): reference(path) for path in manipulation_paths
        },
    }


def _display_record(
    *,
    metric_id: str,
    value: Any,
    ci95: list[float] | None,
    numerator: float,
    denominator: int,
    weighting: str,
    n_pair: int,
    registry_sha: str,
) -> dict[str, Any]:
    return {
        "registry_metric_id": metric_id,
        "value": value,
        "ci95": ci95,
        "numerator": float(numerator),
        "denominator": int(denominator),
        "weighting": weighting,
        "n_target": 50,
        "n_pair": int(n_pair),
        "artifact": "h5_metric_registry.json",
        "artifact_sha256": registry_sha,
    }


def _reject_stale_output(output: Path, collection_file: Path) -> None:
    if not output.exists():
        return
    allowed = set(H5_OUTPUT_FILES)
    if collection_file.is_relative_to(output):
        allowed.add(collection_file.relative_to(output).as_posix())
    unexpected = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).as_posix() not in allowed
    )
    if unexpected:
        raise H5ContractError(
            "unexpected stale H5 analysis output: " + ", ".join(unexpected)
        )


def analyze_collection(
    collection_path: Path | str,
    output_dir: Path | str,
    *,
    raw_root: Path | str | None = None,
    repo_root: Path | str = Path("."),
    verified_analysis_provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Analyze only manifest-listed H5 artifacts and write deterministic outputs."""

    collection_file = Path(collection_path).expanduser().resolve(strict=True)
    raw = (
        Path(raw_root).expanduser().resolve(strict=True)
        if raw_root is not None
        else collection_file.parent
    )
    repo = Path(repo_root).expanduser().resolve(strict=True)
    analysis_provenance = _h5_analysis_provenance(
        repo,
        verified_analysis_provenance,
    )
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output == raw or output.is_relative_to(raw):
        raise H5ContractError("H5 analysis output cannot be inside the raw sim_store")
    _reject_stale_output(output, collection_file)
    collection, preparation, persona_panel, panel, provider_rows = (
        _validate_collection_manifest(collection_file, raw, repo)
    )
    personas = tuple(persona_panel["personas"])
    persona_ids = tuple(str(row["persona_id"]) for row in personas)
    mapping_coverage = dict(panel.get("_h5_mapping_coverage") or {})
    mapping_available = mapping_coverage.get("complete") is True
    provider_matrix = []
    ratings_by_provider: dict[str, dict[str, str]] = {}
    contrast_observations: dict[
        str, dict[str, dict[str, float | int]]
    ] = {}
    ledger_rows = []
    for collection_row in provider_rows:
        try:
            matrix, ratings, contrast, ledger = _provider_analysis(
                raw_root=raw,
                collection_row=collection_row,
                preparation=preparation,
                personas=personas,
                panel=panel,
                mapping_available=mapping_available,
            )
        except (H5ContractError, OSError, TypeError, ValueError, OverflowError):
            provider = str(collection_row["provider"])
            try:
                ledger = _validated_collection_row_accounting(
                    raw, collection_row, provider
                )
            except (
                H5ContractError,
                OSError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                ledger = _empty_provider_accounting("unavailable")
            matrix, ratings, contrast, ledger = _excluded_provider_analysis_result(
                collection_row,
                personas,
                mapping_available=mapping_available,
                ledger=ledger,
                reason_code="analysis_provider_validation_failed",
            )
        provider_matrix.append(matrix)
        provider = str(matrix["provider"])
        if matrix["collection_status"] == "collected":
            ratings_by_provider[provider] = ratings
        if contrast:
            contrast_observations[provider] = contrast
        ledger_rows.append(
            {
                "provider": provider,
                "model_id": matrix.get("model_id"),
                "prompt_revision": matrix.get("prompt_revision"),
                **ledger,
            }
        )
    qualifying = [str(row["provider"]) for row in provider_matrix if row["qualifies"]]
    raw_complete = [str(row["provider"]) for row in provider_matrix if row["raw_complete"]]
    if len(qualifying) >= 2:
        agreement_providers = qualifying
        agreement_scope = "qualifying_providers"
    elif len(raw_complete) >= 2:
        agreement_providers = raw_complete
        agreement_scope = "exploratory_unqualified"
    else:
        agreement_providers = []
        agreement_scope = "unavailable"
    agreement_ratings = {
        provider: ratings_by_provider[provider] for provider in agreement_providers
    }
    pairwise = pairwise_cohen_kappa(agreement_ratings) if agreement_ratings else {}
    subjects = tuple(f"{persona_id}|{arm}" for persona_id in persona_ids for arm in ("A", "B"))
    fleiss = (
        fleiss_kappa(
            [
                [ratings_by_provider[provider][subject] for provider in agreement_providers]
                for subject in subjects
            ]
        )
        if len(agreement_providers) >= 2
        else None
    )
    agreement = {
        "scope": agreement_scope,
        "providers": agreement_providers,
        "subject_count": len(subjects),
        "categories": list(TERMINAL_CATEGORIES),
        "fleiss_kappa": fleiss,
        "pairwise": pairwise,
    }
    qualifying_contrasts = {
        provider: contrast_observations[provider]
        for provider in qualifying
        if provider in contrast_observations
    }
    contrast = (
        persona_cluster_contrast_bootstrap(
            qualifying_contrasts,
            persona_ids=persona_ids,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED,
        )
        if qualifying_contrasts
        else None
    )
    response_available = any(int(row["responses"]) > 0 for row in ledger_rows)
    minimum_complete = min(
        (
            min(int(row["completed_by_arm"][arm]) for arm in ("A", "B"))
            for row in provider_matrix
            if row["qualifies"]
        ),
        default=0,
    )
    weak_gate = bool(qualifying) and all(
        bool(row["weak_accuracy_gate"])
        for row in provider_matrix
        if row["qualifies"]
    )
    strong_gate = bool(qualifying) and all(
        bool(row["strong_accuracy_gate"])
        for row in provider_matrix
        if row["qualifies"]
    )
    contrast_point = float(contrast["point"]) if contrast else 0.0
    contrast_ci = list(contrast["ci95"]) if contrast else [0.0, 0.0]
    if not mapping_available:
        analysis_status = "excluded_pre_outcome"
        decision = None
        branch_reason = "no_explicit_machine_annotation_map"
    elif not response_available:
        analysis_status = "pending_input"
        decision = None
        branch_reason = "validated_S2_provider_panel_not_supplied"
    else:
        analysis_status = "complete"
        hit_gate = contrast is not None and contrast_ci[0] > 0.0
        accuracy_gate = weak_gate and strong_gate
        if len(qualifying) >= 5 and minimum_complete >= 45 and accuracy_gate and hit_gate:
            decision = "supported"
            branch_reason = "supported"
        elif len(qualifying) < 4:
            decision = "not_supported"
            branch_reason = "too_few_providers"
        elif not accuracy_gate and not hit_gate:
            decision = "not_supported"
            branch_reason = "neither_manipulation_gate"
        elif len(qualifying) < 5:
            decision = "partially_supported"
            branch_reason = "provider_coverage_partial"
        else:
            decision = "partially_supported"
            branch_reason = "mixed"
    hypothesis = {
        "analysis_status": analysis_status,
        "decision": decision,
        "branch_reason": branch_reason,
        "predicate_inputs": {
            "qualifying_provider_count": len(qualifying),
            "minimum_completed_personas_per_qualifying_cell": minimum_complete,
            "weak_accuracy_gate": weak_gate,
            "strong_accuracy_gate": strong_gate,
            "misconception_hit_point": contrast_point if mapping_available else None,
            "misconception_hit_ci_low": contrast_ci[0] if mapping_available else None,
            "misconception_hit_ci_high": contrast_ci[1] if mapping_available else None,
            "annotation_map_sha256": panel.get("annotation_map_sha256"),
            "collection_sha256": collection.get("collection_sha256"),
            "preparation_sha256": collection.get("preparation_sha256"),
            "mapping_covered_entries": mapping_coverage.get("covered_entries"),
            "mapping_required_entries": mapping_coverage.get("required_entries"),
        },
    }
    totals = {
        key: sum(int(row[key]) for row in ledger_rows)
        for key in ("requests", "responses", "retries", "input_tokens", "output_tokens")
    }
    totals["cost_yuan"] = round(
        sum(float(row["cost_yuan"]) for row in ledger_rows), 12
    )
    ledger = {"providers": ledger_rows, "totals": totals}
    lifecycle_providers: dict[str, list[str]] = {}
    for row in provider_matrix:
        collection_status = row.get("collection_status")
        provider_status = row.get("provider_status")
        failure_category = row.get("failure_category")
        exclusion_reason = row.get("exclusion_reason")
        if collection_status == "invalid_excluded":
            category = (
                "invalid_calibration_schema"
                if row.get("invalid_reason_code") == "invalid_calibration_schema"
                else "invalid_provider_artifact"
            )
        elif collection_status in {"missing", "missing_required_revision"}:
            category = str(collection_status)
        elif provider_status == "excluded_post_calibration":
            category = "post_calibration_exclusion"
        elif provider_status == "excluded_model_drift" or failure_category == "model_id_drift":
            category = "model_drift_exclusion"
        elif provider_status == "excluded_pre_outcome":
            category = (
                "provider_configuration_exclusion"
                if exclusion_reason == "provider_configuration_unavailable"
                else "pre_outcome_design_exclusion"
            )
        elif provider_status in {"interrupted", "interrupted_calibration"}:
            category = (
                "network_interruption"
                if failure_category in {"network", "circuit"}
                else "technical_interruption"
            )
        elif provider_status in {"complete", "partial"}:
            category = "collected"
        else:
            category = "technical_interruption"
        lifecycle_providers.setdefault(category, []).append(str(row["provider"]))
    lifecycle_counts = {
        category: len(providers)
        for category, providers in sorted(lifecycle_providers.items())
    }
    provider_exclusion_disclosure = {
        category: providers
        for category, providers in sorted(lifecycle_providers.items())
        if category != "collected"
    }
    denominators = {
        "frozen_provider_count": 6,
        "collected_provider_count": sum(
            row["collection_status"] == "collected" for row in provider_matrix
        ),
        "qualifying_provider_count": len(qualifying),
        "frozen_persona_count": 50,
        "frozen_subject_count": 100,
        "observed_journeys": sum(int(row["observed_journeys"]) for row in provider_matrix),
        "missing_journeys": sum(int(row["missing_journeys"]) for row in provider_matrix),
        "structural_incomplete_journeys": sum(
            int(row["structural_incomplete_journeys"]) for row in provider_matrix
        ),
        "structural_calibration_personas": sum(
            int(row.get("calibration_structural_personas") or 0)
            for row in provider_matrix
        ),
        "model_drift_attempts": sum(
            int(row.get("drift_count") or 0) for row in provider_matrix
        ),
        "mapping_required_entries": mapping_coverage.get("required_entries"),
        "mapping_covered_entries": mapping_coverage.get("covered_entries"),
        "invalid_calibration_schema_provider_count": lifecycle_counts.get(
            "invalid_calibration_schema", 0
        ),
        "invalid_provider_artifact_count": lifecycle_counts.get(
            "invalid_provider_artifact", 0
        ),
        "missing_provider_count": lifecycle_counts.get("missing", 0),
        "missing_required_revision_provider_count": lifecycle_counts.get(
            "missing_required_revision", 0
        ),
        "network_interruption_provider_count": lifecycle_counts.get(
            "network_interruption", 0
        ),
        "model_drift_exclusion_provider_count": lifecycle_counts.get(
            "model_drift_exclusion", 0
        ),
        "provider_configuration_exclusion_provider_count": lifecycle_counts.get(
            "provider_configuration_exclusion", 0
        ),
        "pre_outcome_design_exclusion_provider_count": lifecycle_counts.get(
            "pre_outcome_design_exclusion", 0
        ),
        "technical_interruption_provider_count": lifecycle_counts.get(
            "technical_interruption", 0
        ),
        "post_calibration_exclusion_provider_count": lifecycle_counts.get(
            "post_calibration_exclusion", 0
        ),
        "provider_lifecycle_counts": lifecycle_counts,
        "excluded_provider_cells": sum(not row["qualifies"] for row in provider_matrix) * 2,
        "excluded_persona_cells": sum(
            100
            if row["collection_status"] != "collected"
            else int(row["missing_journeys"])
            + int(row["structural_incomplete_journeys"])
            for row in provider_matrix
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable_value(output / "provider_ledger.json", ledger)
    _write_immutable_value(output / "agreement_matrix.json", agreement)
    _write_immutable_value(output / "manipulation_matrix.json", provider_matrix)
    registry_rows: list[dict[str, Any]] = []
    if analysis_status == "complete":
        registry_rows = [
            {
                "metric_id": "h5.qualifying_provider_count",
                "value": float(len(qualifying)),
                "numerator": float(len(qualifying)),
                "denominator": 6,
                "weighting": "provider_equal",
                "n_target": 50,
                "n_pair": 100,
                "raw_hash": collection["collection_sha256"],
                "ci_low": None,
                "ci_high": None,
            },
            {
                "metric_id": "h5.minimum_completed_personas_per_qualifying_cell",
                "value": float(minimum_complete),
                "numerator": float(minimum_complete),
                "denominator": 1,
                "weighting": "minimum_provider_arm_cell",
                "n_target": 50,
                "n_pair": 100,
                "raw_hash": collection["collection_sha256"],
                "ci_low": None,
                "ci_high": None,
            },
            {
                "metric_id": "h5.weak_accuracy_gate",
                "value": weak_gate,
                "numerator": float(weak_gate),
                "denominator": 1,
                "weighting": "frozen_boolean_gate",
                "n_target": 50,
                "n_pair": 100,
                "raw_hash": collection["collection_sha256"],
                "ci_low": None,
                "ci_high": None,
            },
            {
                "metric_id": "h5.strong_accuracy_gate",
                "value": strong_gate,
                "numerator": float(strong_gate),
                "denominator": 1,
                "weighting": "frozen_boolean_gate",
                "n_target": 50,
                "n_pair": 100,
                "raw_hash": collection["collection_sha256"],
                "ci_low": None,
                "ci_high": None,
            },
            {
                "metric_id": "h5.misconception_hit_rate_contrast",
                "value": contrast_point,
                "numerator": contrast_point,
                "denominator": max(1, len(qualifying)),
                "weighting": "provider_equal_persona_cluster",
                "n_target": 50,
                "n_pair": 100,
                "raw_hash": collection["collection_sha256"],
                "ci_low": contrast_ci[0],
                "ci_high": contrast_ci[1],
            },
        ]
    _write_immutable_value(output / "h5_metric_registry.json", registry_rows)
    registry_sha = _file_sha(output / "h5_metric_registry.json")
    metrics: dict[str, Any] = {result_id: None for result_id in H5_RESULT_IDS}
    if analysis_status == "complete":
        metrics = {
            "H5_QUALIFYING_PROVIDER_COUNT": _display_record(
                metric_id="h5.qualifying_provider_count",
                value=len(qualifying),
                ci95=None,
                numerator=len(qualifying),
                denominator=6,
                weighting="provider_equal",
                n_pair=100,
                registry_sha=registry_sha,
            ),
            "H5_MINIMUM_COMPLETED_PERSONAS_PER_QUALIFYING_CELL": _display_record(
                metric_id="h5.minimum_completed_personas_per_qualifying_cell",
                value=minimum_complete,
                ci95=None,
                numerator=minimum_complete,
                denominator=1,
                weighting="minimum_provider_arm_cell",
                n_pair=100,
                registry_sha=registry_sha,
            ),
            "H5_WEAK_ACCURACY_GATE": _display_record(
                metric_id="h5.weak_accuracy_gate",
                value=weak_gate,
                ci95=None,
                numerator=float(weak_gate),
                denominator=1,
                weighting="frozen_boolean_gate",
                n_pair=100,
                registry_sha=registry_sha,
            ),
            "H5_STRONG_ACCURACY_GATE": _display_record(
                metric_id="h5.strong_accuracy_gate",
                value=strong_gate,
                ci95=None,
                numerator=float(strong_gate),
                denominator=1,
                weighting="frozen_boolean_gate",
                n_pair=100,
                registry_sha=registry_sha,
            ),
            "H5_MISCONCEPTION_HIT_RATE_CONTRAST": _display_record(
                metric_id="h5.misconception_hit_rate_contrast",
                value=contrast_point,
                ci95=contrast_ci,
                numerator=contrast_point,
                denominator=max(1, len(qualifying)),
                weighting="provider_equal_persona_cluster",
                n_pair=100,
                registry_sha=registry_sha,
            ),
            "H5_MISCONCEPTION_HIT_RATE_CONTRAST_CI95": _display_record(
                metric_id="h5.misconception_hit_rate_contrast",
                value=contrast_ci,
                ci95=contrast_ci,
                numerator=contrast_point,
                denominator=max(1, len(qualifying)),
                weighting="provider_equal_persona_cluster",
                n_pair=100,
                registry_sha=registry_sha,
            ),
        }
    figures = _render_figures(output, provider_matrix, agreement, hypothesis)
    provenance = {
        "s2": {
            "preparation_sha256": collection.get("preparation_sha256"),
            "collection_sha256": collection.get("collection_sha256"),
            "panel_sha256": collection.get("panel_sha256"),
            "persona_panel_sha256": collection.get("persona_panel_sha256"),
            "config_sha256": collection.get("config_sha256"),
            "code_commit": collection.get("s2_git_head"),
            "code_sha256": collection.get("s2_code_sha256"),
            "official_input_sha256": collection.get("official_input_sha256"),
        },
        "analysis": analysis_provenance,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }
    artifact_paths = [
        output / "agreement_matrix.json",
        output / "h5_metric_registry.json",
        output / "manipulation_matrix.json",
        output / "provider_ledger.json",
        output / "figures/provider_agreement.png",
        output / "figures/provider_agreement.svg",
        output / "figures/manipulation_checks.png",
        output / "figures/manipulation_checks.svg",
    ]
    artifact_rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": _file_sha(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(artifact_paths)
    ]
    artifact_core = {
        "simulated": True,
        "run_id": collection.get("run_id"),
        "persona_id": "llm-sim-study:h5-analysis-artifacts",
        "provider": "analysis",
        "model_id": "no-provider-observation",
        "record_type": "llm_sim_h5_artifact_manifest",
        "schema_version": "yher.h5.artifacts.v1",
        "collection_sha256": collection.get("collection_sha256"),
        "provenance": provenance,
        "artifacts": artifact_rows,
    }
    artifact_manifest = {
        **artifact_core,
        "artifact_manifest_sha256": _canonical_sha(artifact_core),
    }
    _write_immutable_json(output / "artifact_manifest.json", artifact_manifest)
    result_core = {
        "simulated": True,
        "run_id": collection.get("run_id"),
        "persona_id": "llm-sim-study:h5-results",
        "provider": "analysis",
        "model_id": "no-provider-observation",
        "record_type": "llm_sim_h5_results",
        "schema_version": "yher.h5.results.v1",
        "status": analysis_status,
        "hypothesis": hypothesis,
        "metrics": metrics,
        "provider_matrix": provider_matrix,
        "agreement": agreement,
        "ledger": ledger,
        "denominators": denominators,
        "provider_exclusion_disclosure": provider_exclusion_disclosure,
        "mapping_coverage": mapping_coverage,
        "figures": figures,
        "artifact_manifest": "artifact_manifest.json",
        "artifact_manifest_sha256": _file_sha(output / "artifact_manifest.json"),
        "artifact_manifest_internal_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "provenance": provenance,
    }
    result = {**result_core, "h5_results_sha256": _canonical_sha(result_core)}
    _write_immutable_json(output / "h5_results.json", result)
    # A concurrent or post-finalization raw mutation must never yield a package.
    _validate_collection_manifest(collection_file, raw, repo)
    return result


def _json_value(path: Path, label: str) -> tuple[Any, str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise H5ContractError(f"cannot read {label}") from exc
    return value, hashlib.sha256(payload).hexdigest()


def _verify_reference(root: Path, reference: Mapping[str, Any], label: str) -> Path:
    path, _ = _relative_path(root, reference.get("path"), label)
    if _file_sha(path) != reference.get("sha256"):
        raise H5ContractError(f"{label} hash mismatch")
    return path


def _verify_reference_tree(root: Path, value: Any, label: str) -> list[Path]:
    if isinstance(value, Mapping) and set(value) >= {"path", "sha256"}:
        return [_verify_reference(root, value, label)]
    if isinstance(value, Mapping):
        paths = []
        for key, child in value.items():
            paths.extend(_verify_reference_tree(root, child, f"{label}.{key}"))
        return paths
    raise H5ContractError(f"{label} must contain hashed artifact references")


def _validate_s3_for_merge(
    payload: Mapping[str, Any], artifact_root: Path
) -> None:
    from .paper import PROGRAMMATIC_IDS, REQUIRED_FIGURE_IDS

    if payload.get("schema_version") != "yher.paper-results.v1":
        raise H5ContractError("S3 contract schema is invalid")
    if payload.get("status") != PROGRAMMATIC_H5_PENDING_STATUS:
        raise H5ContractError(
            "S3 contract lifecycle must be PROGRAMMATIC_COMPLETE_H5_PENDING"
        )
    metrics = payload.get("metrics")
    decisions = payload.get("decisions")
    hypotheses = payload.get("hypotheses")
    if not isinstance(metrics, Mapping) or not isinstance(decisions, Mapping):
        raise H5ContractError("S3 contract metrics or decisions are missing")
    if not isinstance(hypotheses, Mapping):
        raise H5ContractError("S3 contract hypotheses are missing")
    if payload.get("decision_details") not in (None, hypotheses):
        raise H5ContractError("S3 decision_details differs from hypotheses")
    for hypothesis in ("H1", "H2", "H3", "H4"):
        detail = hypotheses.get(hypothesis)
        if (
            not isinstance(detail, Mapping)
            or detail.get("analysis_status") != "complete"
            or detail.get("decision") != decisions.get(hypothesis)
        ):
            raise H5ContractError(f"S3 {hypothesis} is not complete and consistent")
    registry_ref = {
        "path": payload.get("analysis_artifact"),
        "sha256": payload.get("analysis_artifact_sha256"),
    }
    registry_path = _verify_reference(artifact_root, registry_ref, "S3 registry")
    registry_value, registry_sha = _json_value(registry_path, "S3 registry")
    if not isinstance(registry_value, list):
        raise H5ContractError("S3 registry must be a JSON list")
    registry: dict[str, Mapping[str, Any]] = {}
    for row in registry_value:
        if not isinstance(row, Mapping) or not isinstance(row.get("metric_id"), str):
            raise H5ContractError("S3 registry row is invalid")
        if row["metric_id"] in registry:
            raise H5ContractError("S3 registry has duplicate metric IDs")
        registry[str(row["metric_id"])] = row
    for result_id in PROGRAMMATIC_IDS:
        display = metrics.get(result_id)
        if not isinstance(display, Mapping):
            raise H5ContractError(f"S3 metric is missing: {result_id}")
        display_registry = _verify_reference(
            artifact_root,
            {
                "path": display.get("artifact"),
                "sha256": display.get("artifact_sha256"),
            },
            f"S3 display registry {result_id}",
        )
        if not display_registry.is_file():
            raise H5ContractError(f"S3 display registry is missing for {result_id}")
        row = registry.get(str(display.get("registry_metric_id") or ""))
        if row is None:
            raise H5ContractError(f"S3 registry metric is missing for {result_id}")
        expected_value = (
            [row.get("ci_low"), row.get("ci_high")]
            if isinstance(display.get("value"), list)
            else row.get("value")
        )
        if display.get("value") != expected_value:
            raise H5ContractError(f"S3 registry value differs for {result_id}")
        for field in ("numerator", "denominator", "weighting", "n_target", "n_pair"):
            if display.get(field) != row.get(field):
                raise H5ContractError(
                    f"S3 registry field {field} differs for {result_id}"
                )
    figures = payload.get("figures")
    if not isinstance(figures, Mapping):
        raise H5ContractError("S3 contract figures are missing")
    for figure_id in sorted(
        REQUIRED_FIGURE_IDS
        - {"FIG_PROVIDER_AGREEMENT", "FIG_MANIPULATION_CHECKS"}
    ):
        if figures.get(figure_id) is None:
            raise H5ContractError(f"S3 figure is missing: {figure_id}")
        _verify_reference_tree(artifact_root, figures[figure_id], figure_id)


def _reject_sensitive_keys(value: Any, path: str = "H5 results") -> None:
    forbidden = {
        "api_key",
        "authorization",
        "content",
        "raw_error",
        "response_body",
        "environment",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise H5ContractError(f"{path} contains forbidden sensitive field {key}")
            _reject_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, f"{path}[{index}]")


def _validate_h5_results(result_path: Path) -> tuple[dict[str, Any], Path]:
    root = result_path.parent.resolve(strict=True)
    result, _, _ = _read_json(result_path, "H5 results")
    _require_envelope(result, "H5 results", provider="analysis")
    _internal_hash(result, "h5_results_sha256", "H5 results")
    _reject_sensitive_keys(result)
    manifest_path, _ = _relative_path(
        root, result.get("artifact_manifest"), "H5 artifact manifest"
    )
    manifest, _, manifest_file_sha = _read_json(
        manifest_path, "H5 artifact manifest"
    )
    _require_envelope(manifest, "H5 artifact manifest", provider="analysis")
    _internal_hash(manifest, "artifact_manifest_sha256", "H5 artifact manifest")
    if (
        manifest_file_sha != result.get("artifact_manifest_sha256")
        or manifest.get("artifact_manifest_sha256")
        != result.get("artifact_manifest_internal_sha256")
        or manifest.get("collection_sha256")
        != (result.get("provenance") or {}).get("s2", {}).get("collection_sha256")
    ):
        raise H5ContractError("H5 artifact manifest hash or provenance mismatch")
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise H5ContractError("H5 artifact manifest rows are missing")
    listed = set()
    for row in artifact_rows:
        if not isinstance(row, Mapping):
            raise H5ContractError("H5 artifact manifest row is invalid")
        path, relative = _relative_path(root, row.get("path"), "H5 artifact")
        if relative in listed:
            raise H5ContractError("H5 artifact manifest path is duplicated")
        listed.add(relative)
        if (
            _file_sha(path) != row.get("sha256")
            or path.stat().st_size != row.get("bytes")
        ):
            raise H5ContractError(f"H5 artifact hash mismatch: {relative}")
    expected_manifest_artifacts = H5_OUTPUT_FILES - {
        "artifact_manifest.json",
        "h5_results.json",
    }
    if listed != expected_manifest_artifacts:
        raise H5ContractError(
            "H5 artifact manifest paths differ from the frozen output set"
        )
    figures = result.get("figures")
    if not isinstance(figures, Mapping):
        raise H5ContractError("H5 result figures are missing")
    for figure_id in ("FIG_PROVIDER_AGREEMENT", "FIG_MANIPULATION_CHECKS"):
        references = _verify_reference_tree(root, figures.get(figure_id), figure_id)
        if not any(path.suffix.lower() == ".png" for path in references):
            raise H5ContractError(f"{figure_id} lacks a PNG")
    status = result.get("status")
    hypothesis = result.get("hypothesis")
    metrics = result.get("metrics")
    if status not in {"complete", "excluded_pre_outcome", "pending_input"}:
        raise H5ContractError("H5 lifecycle status is invalid")
    if not isinstance(hypothesis, Mapping) or hypothesis.get("analysis_status") != status:
        raise H5ContractError("H5 hypothesis lifecycle differs from result status")
    if not isinstance(metrics, Mapping) or set(metrics) != set(H5_RESULT_IDS):
        raise H5ContractError("H5 result metric IDs differ from the frozen contract")
    if status == "complete":
        registry_path = root / "h5_metric_registry.json"
        _, registry_sha = _json_value(registry_path, "H5 metric registry")
        for result_id in H5_RESULT_IDS:
            display = metrics.get(result_id)
            if (
                not isinstance(display, Mapping)
                or display.get("artifact") != "h5_metric_registry.json"
                or display.get("artifact_sha256") != registry_sha
            ):
                raise H5ContractError(f"H5 metric registry hash differs for {result_id}")
        from .paper import PREDICATE_RESULT_BINDINGS, derive_hypothesis_branch

        predicate_inputs = hypothesis.get("predicate_inputs")
        if not isinstance(predicate_inputs, Mapping):
            raise H5ContractError("H5 predicate inputs are missing")
        branch = derive_hypothesis_branch("H5", predicate_inputs)
        if (
            hypothesis.get("decision") != branch.decision
            or hypothesis.get("branch_reason") != branch.reason_key
        ):
            raise H5ContractError("H5 decision branch differs from frozen predicates")
        for predicate, (result_id, selector) in PREDICATE_RESULT_BINDINGS["H5"].items():
            display = metrics[result_id]
            raw_value = display.get("value")
            expected = (
                raw_value
                if selector == "value"
                else raw_value[0 if selector == "ci_low" else 1]
            )
            actual = predicate_inputs.get(predicate)
            if isinstance(expected, bool):
                matches = isinstance(actual, bool) and actual is expected
            else:
                try:
                    matches = math.isclose(
                        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    matches = False
            if not matches:
                raise H5ContractError(
                    f"H5 predicate {predicate} differs from display metric"
                )
    elif status == "excluded_pre_outcome":
        if hypothesis.get("decision") is not None:
            raise H5ContractError("excluded H5 decision must be null")
        if not str(hypothesis.get("branch_reason") or "").strip():
            raise H5ContractError("excluded H5 branch reason is missing")
    elif any(metrics.get(result_id) is not None for result_id in H5_RESULT_IDS):
        raise H5ContractError("non-evaluated H5 lifecycle must keep metrics null")
    return result, root


def _copy_immutable(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() == payload:
            return
        raise H5ContractError(f"immutable merged H5 artifact differs: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _prefix_reference_paths(value: Any, prefix: str) -> Any:
    if isinstance(value, Mapping):
        output = {key: _prefix_reference_paths(child, prefix) for key, child in value.items()}
        if set(value) >= {"path", "sha256"}:
            output["path"] = f"{prefix}/{value['path']}"
        if set(value) >= {"artifact", "artifact_sha256"}:
            output["artifact"] = f"{prefix}/{value['artifact']}"
        return output
    if isinstance(value, list):
        return [_prefix_reference_paths(child, prefix) for child in value]
    return copy.deepcopy(value)


def _atomic_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _contract_bytes(original: str, updated: Mapping[str, Any]) -> bytes:
    from .paper import CONTRACT_BEGIN, CONTRACT_END

    if original.count(CONTRACT_BEGIN) != 1 or original.count(CONTRACT_END) != 1:
        raise H5ContractError("results contract marker pair is invalid")
    start = original.index(CONTRACT_BEGIN)
    end = original.index(CONTRACT_END, start) + len(CONTRACT_END)
    block = (
        f"{CONTRACT_BEGIN}\n\n```json\n"
        + json.dumps(updated, ensure_ascii=False, sort_keys=True, indent=2)
        + f"\n```\n\n{CONTRACT_END}"
    )
    return (original[:start] + block + original[end:]).encode("utf-8")


def _idempotent_final_merge(
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
    result_path: Path,
    artifacts: Path,
) -> dict[str, Any]:
    _validate_final_h5_surface(payload, result, result_path, artifacts)
    try:
        from .paper import _validate_contract

        _validate_contract(payload, artifacts)
    except Exception as exc:
        raise H5ContractError("final H5 contract is not binder-valid") from exc
    return dict(payload)


def _validate_final_h5_surface(
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
    result_path: Path,
    artifacts: Path,
) -> None:
    from .paper import H5_IDS

    status = str(result["status"])
    expected_status = {
        "complete": H5_EVALUATED_STATUS,
        "excluded_pre_outcome": H5_EXCLUDED_STATUS,
    }.get(status)
    if payload.get("status") != expected_status:
        raise H5ContractError("final H5 contract conflicts with supplied results")
    expected_hypothesis = copy.deepcopy(result["hypothesis"])
    if status == "excluded_pre_outcome":
        expected_hypothesis["predicate_inputs"]["evidence_path"] = "h5/h5_results.json"
        expected_hypothesis["predicate_inputs"]["evidence_sha256"] = _file_sha(
            result_path
        )
    expected_metrics = _prefix_reference_paths(result["metrics"], "h5")
    if status == "complete":
        for result_id in H5_IDS:
            display = dict(expected_metrics[result_id])
            display["artifact"] = payload.get("analysis_artifact")
            display["artifact_sha256"] = payload.get("analysis_artifact_sha256")
            display["raw_hash"] = payload.get("h5_collection_manifest_sha256")
            expected_metrics[result_id] = display
    expected_figures = _prefix_reference_paths(result["figures"], "h5")
    result_denominators = result.get("denominators") or {}
    if any(
        (
            payload.get("h5_results_sha256") != result.get("h5_results_sha256"),
            payload.get("h5_results_file_sha256") != _file_sha(result_path),
            payload.get("hypotheses", {}).get("H5") != expected_hypothesis,
            payload.get("decisions", {}).get("H5")
            != expected_hypothesis.get("decision"),
            any(payload.get("metrics", {}).get(key) != expected_metrics[key] for key in H5_IDS),
            any(
                payload.get("figures", {}).get(key) != expected_figures[key]
                for key in ("FIG_PROVIDER_AGREEMENT", "FIG_MANIPULATION_CHECKS")
            ),
            any(
                payload.get("denominators", {}).get(field)
                != result_denominators.get(field)
                for field in H5_LIFECYCLE_DENOMINATOR_FIELDS
            ),
            payload.get("h5_provider_exclusion_disclosure")
            != result.get("provider_exclusion_disclosure"),
        )
    ):
        raise H5ContractError("final H5 contract conflicts with supplied results")
    if payload.get("decision_details") is not None and payload.get(
        "decision_details", {}
    ).get("H5") != expected_hypothesis:
        raise H5ContractError("final H5 decision_details conflicts with results")
    for figure_id in ("FIG_PROVIDER_AGREEMENT", "FIG_MANIPULATION_CHECKS"):
        _verify_reference_tree(
            artifacts, payload["figures"][figure_id], f"final {figure_id}"
        )
    if status == "complete":
        registry_path = _verify_reference(
            artifacts,
            {
                "path": payload.get("analysis_artifact"),
                "sha256": payload.get("analysis_artifact_sha256"),
            },
            "final merged registry",
        )
        for result_id in H5_IDS:
            display = payload["metrics"][result_id]
            _verify_reference(
                artifacts,
                {
                    "path": display.get("artifact"),
                    "sha256": display.get("artifact_sha256"),
                },
                f"final {result_id}",
            )
            if display.get("raw_hash") != payload.get(
                "h5_collection_manifest_sha256"
            ):
                raise H5ContractError(f"final {result_id} raw hash is invalid")
        results_path = _verify_reference(
            artifacts,
            {
                "path": payload.get("results_artifact"),
                "sha256": payload.get("results_artifact_sha256"),
            },
            "final merged results provenance",
        )
        manifest_path = _verify_reference(
            artifacts,
            {
                "path": payload.get("artifact_manifest"),
                "sha256": payload.get("artifact_manifest_sha256"),
            },
            "final merged artifact manifest",
        )
        results_document, _ = _json_value(results_path, "final merged results")
        manifest_document, _ = _json_value(
            manifest_path, "final merged artifact manifest"
        )
        files = manifest_document.get("files") if isinstance(manifest_document, Mapping) else None
        if (
            not isinstance(results_document, Mapping)
            or results_document.get("numeric_source") != payload.get("analysis_artifact")
            or not isinstance(files, Mapping)
            or files.get(payload.get("analysis_artifact"))
            != payload.get("analysis_artifact_sha256")
            or files.get(payload.get("results_artifact"))
            != payload.get("results_artifact_sha256")
            or registry_path != (artifacts / str(payload.get("analysis_artifact"))).resolve()
        ):
            raise H5ContractError("final merged provenance chain is invalid")
    else:
        _verify_reference(
            artifacts,
            {
                "path": expected_hypothesis["predicate_inputs"].get("evidence_path"),
                "sha256": expected_hypothesis["predicate_inputs"].get(
                    "evidence_sha256"
                ),
            },
            "final H5 exclusion evidence",
        )


def merge_h5_results(
    contract_path: Path | str,
    h5_results_path: Path | str,
    *,
    artifact_root: Path | str,
) -> dict[str, Any]:
    """Validate S3 and atomically merge only H5-owned contract surfaces."""

    from .paper import H5_IDS, load_results_contract

    contract = Path(contract_path)
    artifacts = Path(artifact_root).expanduser().resolve(strict=True)
    payload = load_results_contract(contract)
    result_path = Path(h5_results_path).expanduser().resolve(strict=True)
    result, h5_root = _validate_h5_results(result_path)
    status = str(result["status"])
    if status == "pending_input":
        raise H5ContractError("pending_input H5 results cannot be merged into paper-final")
    if payload.get("status") in {H5_EVALUATED_STATUS, H5_EXCLUDED_STATUS}:
        return _idempotent_final_merge(payload, result, result_path, artifacts)
    _validate_s3_for_merge(payload, artifacts)
    # Validate every source before copying anything or replacing the marker.
    source_paths = [h5_root / relative for relative in sorted(H5_OUTPUT_FILES)]
    if result_path != (h5_root / "h5_results.json").resolve(strict=True):
        raise H5ContractError("H5 results path differs from the frozen output set")
    unique_sources = sorted(set(source_paths))
    planned: dict[Path, bytes] = {
        Path("h5") / source.relative_to(h5_root): source.read_bytes()
        for source in unique_sources
    }

    updated = copy.deepcopy(payload)
    updated["status"] = {
        "complete": H5_EVALUATED_STATUS,
        "excluded_pre_outcome": H5_EXCLUDED_STATUS,
    }[status]
    h5_metrics = _prefix_reference_paths(result["metrics"], "h5")
    for result_id in H5_IDS:
        updated["metrics"][result_id] = h5_metrics[result_id]
    h5_hypothesis = copy.deepcopy(result["hypothesis"])
    if status == "excluded_pre_outcome":
        h5_hypothesis["predicate_inputs"]["evidence_path"] = "h5/h5_results.json"
        h5_hypothesis["predicate_inputs"]["evidence_sha256"] = _file_sha(
            result_path
        )
    updated["decisions"]["H5"] = h5_hypothesis["decision"]
    updated["hypotheses"]["H5"] = h5_hypothesis
    if updated.get("decision_details") is not None:
        updated["decision_details"]["H5"] = copy.deepcopy(h5_hypothesis)
    prefixed_figures = _prefix_reference_paths(result["figures"], "h5")
    for figure_id in ("FIG_PROVIDER_AGREEMENT", "FIG_MANIPULATION_CHECKS"):
        updated["figures"][figure_id] = prefixed_figures[figure_id]
    denominators = result.get("denominators") or {}
    updated["denominators"]["excluded_provider_cells"] = denominators.get(
        "excluded_provider_cells"
    )
    updated["denominators"]["excluded_persona_cells"] = denominators.get(
        "excluded_persona_cells"
    )
    for field in H5_LIFECYCLE_DENOMINATOR_FIELDS:
        updated["denominators"][field] = copy.deepcopy(denominators.get(field))
    updated["h5_provider_exclusion_disclosure"] = copy.deepcopy(
        result.get("provider_exclusion_disclosure")
    )
    provenance = result.get("provenance") or {}
    updated["h5_collection_manifest_sha256"] = (
        provenance.get("s2") or {}
    ).get("collection_sha256")
    updated["h5_results_sha256"] = result.get("h5_results_sha256")
    updated["h5_results_file_sha256"] = _file_sha(result_path)
    updated["h5_artifact_manifest_sha256"] = result.get(
        "artifact_manifest_sha256"
    )
    updated["h5_artifact_manifest_internal_sha256"] = result.get(
        "artifact_manifest_internal_sha256"
    )
    updated["h5_analysis_code_sha256"] = (
        provenance.get("analysis") or {}
    ).get("analysis_code_sha256")
    if status == "complete":
        source_registry_path, _ = _relative_path(
            artifacts,
            payload.get("analysis_artifact"),
            "S3 analysis registry",
        )
        source_registry, _ = _json_value(source_registry_path, "S3 analysis registry")
        h5_registry, _ = _json_value(
            h5_root / "h5_metric_registry.json", "H5 metric registry"
        )
        if not isinstance(source_registry, list) or not isinstance(h5_registry, list):
            raise H5ContractError("merged metric registries must be JSON arrays")
        merged_by_id: dict[str, Any] = {}
        for row in [*source_registry, *h5_registry]:
            if not isinstance(row, Mapping) or not isinstance(row.get("metric_id"), str):
                raise H5ContractError("merged metric registry row is invalid")
            metric_id = str(row["metric_id"])
            if metric_id in merged_by_id and merged_by_id[metric_id] != row:
                raise H5ContractError(f"merged metric registry conflicts on {metric_id}")
            merged_by_id[metric_id] = dict(row)
        merged_registry = [merged_by_id[key] for key in sorted(merged_by_id)]
        merged_registry_payload = (
            json.dumps(merged_registry, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
        planned[Path("h5/merged_metric_registry.json")] = merged_registry_payload
        updated["analysis_artifact"] = "h5/merged_metric_registry.json"
        updated["analysis_artifact_sha256"] = hashlib.sha256(
            merged_registry_payload
        ).hexdigest()
        h5_raw_hash = updated.get("h5_collection_manifest_sha256")
        for result_id, display in updated["metrics"].items():
            if not isinstance(display, Mapping):
                continue
            normalized_display = dict(display)
            normalized_display["artifact"] = updated["analysis_artifact"]
            normalized_display["artifact_sha256"] = updated[
                "analysis_artifact_sha256"
            ]
            normalized_display["raw_hash"] = (
                h5_raw_hash if result_id in H5_IDS else updated.get("raw_hash")
            )
            updated["metrics"][result_id] = normalized_display
        results_path, _ = _relative_path(
            artifacts, payload.get("results_artifact"), "S3 results provenance"
        )
        results_document, _ = _json_value(results_path, "S3 results provenance")
        if not isinstance(results_document, Mapping):
            raise H5ContractError("S3 results provenance must be an object")
        merged_results = {
            **dict(results_document),
            "numeric_source": updated["analysis_artifact"],
        }
        merged_results_payload = (
            json.dumps(merged_results, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
        merged_results_relative = Path("h5/merged_results.json")
        planned[merged_results_relative] = merged_results_payload
        updated["results_artifact"] = merged_results_relative.as_posix()
        updated["results_artifact_sha256"] = hashlib.sha256(
            merged_results_payload
        ).hexdigest()

        manifest_path, _ = _relative_path(
            artifacts, payload.get("artifact_manifest"), "S3 artifact manifest"
        )
        manifest_document, _ = _json_value(manifest_path, "S3 artifact manifest")
        if not isinstance(manifest_document, Mapping) or not isinstance(
            manifest_document.get("files"), Mapping
        ):
            raise H5ContractError("S3 artifact manifest file map is missing")
        merged_files = dict(manifest_document["files"])
        merged_files[updated["analysis_artifact"]] = updated[
            "analysis_artifact_sha256"
        ]
        merged_files[updated["results_artifact"]] = updated[
            "results_artifact_sha256"
        ]
        merged_manifest = {**dict(manifest_document), "files": merged_files}
        merged_manifest_payload = (
            json.dumps(merged_manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
        merged_manifest_relative = Path("h5/merged_artifact_manifest.json")
        planned[merged_manifest_relative] = merged_manifest_payload
        updated["artifact_manifest"] = merged_manifest_relative.as_posix()
        updated["artifact_manifest_sha256"] = hashlib.sha256(
            merged_manifest_payload
        ).hexdigest()
    original = contract.read_text(encoding="utf-8")
    merged_payload = _contract_bytes(original, updated)
    for relative, value in planned.items():
        destination = artifacts / relative
        if destination.exists() and destination.read_bytes() != value:
            raise H5ContractError(f"immutable merged H5 artifact differs: {destination}")

    with tempfile.TemporaryDirectory(
        prefix=".h5-merge-stage-", dir=artifacts.parent
    ) as temporary_root:
        staged = Path(temporary_root) / "artifacts"
        shutil.copytree(artifacts, staged)
        for relative, value in planned.items():
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
        _validate_final_h5_surface(updated, result, result_path, staged)
        try:
            from .paper import _validate_contract

            _validate_contract(updated, staged)
        except Exception as exc:
            raise H5ContractError(
                "staged merged H5 contract is not binder-valid"
            ) from exc

    original_paths = {path.relative_to(artifacts) for path in artifacts.rglob("*")}
    original_contract = contract.read_bytes()
    try:
        for relative, value in sorted(planned.items(), key=lambda item: item[0].as_posix()):
            destination = artifacts / relative
            if destination.exists():
                continue
            _atomic_payload(destination, value)
        if merged_payload != original_contract:
            _atomic_payload(contract, merged_payload)
    except Exception:
        if contract.read_bytes() != original_contract:
            _atomic_payload(contract, original_contract)
        for path in sorted(
            artifacts.rglob("*"), key=lambda value: len(value.parts), reverse=True
        ):
            relative = path.relative_to(artifacts)
            if relative in original_paths:
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        raise
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YHer frozen H5 collection and analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser(
        "lock",
        help="generate the canonical post-collection lock for review and commit",
    )
    lock.add_argument("--raw-root", type=Path, required=True)
    lock.add_argument("--repo-root", type=Path, default=Path("."))
    finalize = subparsers.add_parser("finalize", help="freeze exact S2 provider inputs")
    finalize.add_argument("--raw-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--repo-root", type=Path, default=Path("."))
    analyze = subparsers.add_parser("analyze", help="analyze one frozen collection")
    analyze.add_argument("--collection", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--raw-root", type=Path)
    analyze.add_argument("--repo-root", type=Path, default=Path("."))
    merge = subparsers.add_parser("merge", help="merge H5 into a validated S3 contract")
    merge.add_argument("--contract", type=Path, required=True)
    merge.add_argument("--h5-results", type=Path, required=True)
    merge.add_argument("--artifact-root", type=Path, required=True)
    run = subparsers.add_parser("run", help="finalize and analyze, optionally merge")
    run.add_argument("--raw-root", type=Path, required=True)
    run.add_argument("--collection", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--contract", type=Path)
    run.add_argument("--artifact-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lock":
        repo = args.repo_root.expanduser().resolve(strict=True)
        result = write_collection_lock(
            args.raw_root,
            repo / COLLECTION_LOCK_RELATIVE,
            repo_root=repo,
        )
        summary = {
            "collection_sha256": result["collection_sha256"],
            "lock": COLLECTION_LOCK_RELATIVE.as_posix(),
            "next": "review and commit the lock before finalize/analyze",
        }
    elif args.command == "finalize":
        result = finalize_collection(args.raw_root, args.output, repo_root=args.repo_root)
        summary = {"collection_sha256": result["collection_sha256"]}
    elif args.command == "analyze":
        result = analyze_collection(
            args.collection,
            args.output_dir,
            raw_root=args.raw_root,
            repo_root=args.repo_root,
        )
        summary = {"status": result["status"], "decision": result["hypothesis"]["decision"]}
    elif args.command == "merge":
        result = merge_h5_results(
            args.contract,
            args.h5_results,
            artifact_root=args.artifact_root,
        )
        summary = {"status": result["status"]}
    else:
        finalize_collection(args.raw_root, args.collection, repo_root=args.repo_root)
        result = analyze_collection(
            args.collection,
            args.output_dir,
            raw_root=args.raw_root,
            repo_root=args.repo_root,
        )
        if (args.contract is None) != (args.artifact_root is None):
            raise H5ContractError("run requires both --contract and --artifact-root")
        if args.contract is not None:
            merged = merge_h5_results(
                args.contract,
                args.output_dir / "h5_results.json",
                artifact_root=args.artifact_root,
            )
            summary = {"status": result["status"], "contract_status": merged["status"]}
        else:
            summary = {"status": result["status"]}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
