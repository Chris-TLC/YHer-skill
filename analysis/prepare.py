"""Compact validated projection of the frozen raw simulation records."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .bootstrap import BOOTSTRAP_SEED
from .dataset import DatasetContractError
from .metrics import AnalysisEvent, AnalysisRow
from .views import rebuild_views_from_events, validate_raw_views


EXPECTED_JOURNEYS = 32_400
EXPECTED_SHARDS = 216
EXPECTED_REPLICATES = 50
FROZEN_MANIFEST_SHA256 = "2c68cada6c2229e6860d46fca4e4f65b3df674bfc4652b4a947934ba05e76dd3"
FROZEN_RUN_ID = "confirmatory-v1"
FROZEN_RUNNER_COMMIT = "33536b4a810d297166e4f1c0f036bb9c70f1a979"
FROZEN_EXPERIMENT_TAG = "experiment-freeze-20260713"
FROZEN_CONFIG_SHA256 = "9020726dcc118ae5f7d4c4879421a2b54f13f7a3ffc61119d76bc0577dfb2501"
FROZEN_ANALYSIS_PLAN_COMMIT = "6c559b6f2f8cbe9ab61808c351df3743dca1a0be"
FROZEN_ANALYSIS_PLAN_SHA256 = "662e6844dfbabf8942d787d6b7c2ef37d92c995f22e5968563fa0d9861317fd5"
FROZEN_CENSUS_PLAN_COMMIT = "ed42a4fa96c3e357b701e04da74a9f6a54c36b92"
FROZEN_CENSUS_SEED = 2026071300
FROZEN_STATES = ("M", "P", "C", "U")
FROZEN_CONDITIONS = ("matched", "misspecified")
FROZEN_ARMS = ("A", "B", "C")
FROZEN_TARGETS = frozenset(
    {
        "元素周期律/周期表",
        "分子间作用力/氢键",
        "分离提纯",
        "化学平衡",
        "化学计量（摩尔/阿伏伽德罗）",
        "化学键",
        "原子结构",
        "反应热/焓变",
        "同分异构体",
        "基本操作",
        "弱电解质电离平衡",
        "晶体结构",
        "有机推断",
        "氧化还原反应",
        "氮及其化合物",
        "氯及其化合物",
        "烷烃",
        "物质分类",
        "物质制备",
        "物质检验",
        "电解质与非电解质",
        "盐类水解",
        "硫及其化合物",
        "芳香烃",
        "误差分析",
        "铁及其化合物",
        "镁铝及其化合物",
    }
)


def validate_frozen_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str | None = None,
    repo_root: Path | str | None = None,
) -> None:
    validation = manifest.get("validation")
    required = {
        "expected_journeys": EXPECTED_JOURNEYS,
        "replicates": EXPECTED_REPLICATES,
        "arms": 3,
        "conditions": 2,
        "truth_states": 4,
        "open_nodes": 27,
    }
    if (
        manifest.get("status") != "complete"
        or manifest.get("full_grid_complete") is not True
        or manifest.get("expected_journey_count") != EXPECTED_JOURNEYS
        or manifest.get("selected_shard_count") != EXPECTED_SHARDS
        or manifest.get("full_shard_count") != EXPECTED_SHARDS
        or manifest.get("bootstrap_seed") != BOOTSTRAP_SEED
        or not isinstance(validation, Mapping)
        or any(validation.get(key) != value for key, value in required.items())
    ):
        raise DatasetContractError(
            "formal analysis requires the complete 32,400-journey, 216-shard "
            "50-replicate frozen manifest"
        )
    if manifest_sha256 is not None:
        _validate_canonical_identity(manifest, manifest_sha256)
        if repo_root is None:
            raise DatasetContractError(
                "canonical manifest validation requires the repository root"
            )
        root = Path(repo_root)
        _validate_frozen_config(root, manifest)
        _validate_annotated_tag(root)
        _validate_frozen_analysis_plan(root)


def _validate_canonical_identity(
    manifest: Mapping[str, Any], manifest_sha256: str
) -> None:
    if manifest_sha256 != FROZEN_MANIFEST_SHA256:
        raise DatasetContractError(
            "formal analysis requires canonical manifest SHA-256 "
            f"{FROZEN_MANIFEST_SHA256}"
        )
    required = {
        "record_type": "confirmatory_run_manifest",
        "simulated": True,
        "persona_id": "confirmatory-run:confirmatory-v1",
        "provider": "programmatic",
        "run_id": FROZEN_RUN_ID,
        "status": "complete",
        "runner_commit": FROZEN_RUNNER_COMMIT,
        "repository_head": FROZEN_RUNNER_COMMIT,
        "experiment_tag": FROZEN_EXPERIMENT_TAG,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "analysis_plan_commit": FROZEN_ANALYSIS_PLAN_COMMIT,
        "census_analysis_plan_commit": FROZEN_CENSUS_PLAN_COMMIT,
        "census_seed": FROZEN_CENSUS_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "selected_shard_count": EXPECTED_SHARDS,
        "full_shard_count": EXPECTED_SHARDS,
        "expected_journey_count": EXPECTED_JOURNEYS,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise DatasetContractError(
                f"canonical manifest {field} differs from the frozen identity"
            )
    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or FROZEN_RUNNER_COMMIT not in model_id:
        raise DatasetContractError("canonical manifest model_id is not runner-bound")
    validation = manifest.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("config_sha256") != FROZEN_CONFIG_SHA256
    ):
        raise DatasetContractError(
            "canonical manifest validation config SHA-256 is invalid"
        )

    repository = manifest.get("repository_binding")
    if not isinstance(repository, Mapping) or any(
        repository.get(key) != value
        for key, value in {
            "analysis_plan_commit": FROZEN_ANALYSIS_PLAN_COMMIT,
            "head": FROZEN_RUNNER_COMMIT,
            "tag": FROZEN_EXPERIMENT_TAG,
            "tag_commit": FROZEN_RUNNER_COMMIT,
            "tag_type": "tag",
            "verified": True,
        }.items()
    ):
        raise DatasetContractError("canonical manifest repository binding is invalid")
    isolation = manifest.get("protected_filesystem_assertion")
    if (
        not isinstance(isolation, Mapping)
        or isolation.get("unchanged") is not True
        or int(isolation.get("attested_shard_count", -1)) != EXPECTED_SHARDS
        or not isinstance(isolation.get("attestations_sha256"), str)
    ):
        raise DatasetContractError("canonical manifest isolation assertion is invalid")

    inputs = manifest.get("input_sha256")
    if not isinstance(inputs, Mapping) or any(
        (inputs.get(name) or {}).get("sha256") != expected
        for name, expected in {
            "confirmatory_analysis_plan": FROZEN_ANALYSIS_PLAN_SHA256,
            "census_summary": "b4ca29eaec60b413de36d94ad9fffcaa9a747adfcb6f38a917d4c2a72d76cd18",
            "census_records": "28a3f8d38f25003834a7a3fe996c3851ecd8d4ba48923099cbc39ab827ac6afe",
        }.items()
    ):
        raise DatasetContractError("canonical manifest census/input identity is invalid")
    _validate_shard_topology(manifest.get("shards"))


def _validate_shard_topology(value: object) -> None:
    if not isinstance(value, list) or len(value) != EXPECTED_SHARDS:
        raise DatasetContractError("canonical manifest shard topology is not 216 cells")
    observed: set[tuple[str, str, str]] = set()
    filenames: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise DatasetContractError("canonical manifest shard row is invalid")
        filename = row.get("filename")
        shard_id = row.get("shard_id")
        digest = row.get("sha256")
        if (
            not isinstance(filename, str)
            or filename in filenames
            or not re.fullmatch(r"shard-[0-9a-f]{20}\.jsonl", filename)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(shard_id, str)
        ):
            raise DatasetContractError("canonical manifest shard identity is invalid")
        filenames.add(filename)
        try:
            fields = dict(part.split("=", 1) for part in shard_id.split("|"))
            cell = (fields["target"], fields["truth"], fields["condition"])
        except (KeyError, ValueError) as exc:
            raise DatasetContractError("canonical manifest shard_id is invalid") from exc
        observed.add(cell)
    expected = {
        (target, truth, condition)
        for target in FROZEN_TARGETS
        for truth in FROZEN_STATES
        for condition in FROZEN_CONDITIONS
    }
    if observed != expected:
        raise DatasetContractError("canonical manifest target/state/condition topology differs")


def _validate_frozen_analysis_plan(repo_root: Path) -> None:
    plan = repo_root / "experiments/analysis_plan.md"
    try:
        working = plan.read_bytes()
        committed = subprocess.run(
            (
                "git",
                "show",
                f"{FROZEN_ANALYSIS_PLAN_COMMIT}:experiments/analysis_plan.md",
            ),
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DatasetContractError("cannot verify the frozen analysis-plan bytes") from exc
    if working != committed or hashlib.sha256(working).hexdigest() != (
        FROZEN_ANALYSIS_PLAN_SHA256
    ):
        raise DatasetContractError(
            "current analysis-plan bytes differ from the frozen git blob/hash"
        )


def _validate_frozen_config(
    repo_root: Path, manifest: Mapping[str, Any]
) -> None:
    config = repo_root / "experiments/config/confirmatory_v1.json"
    try:
        value = json.loads(config.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetContractError("cannot read the frozen confirmatory config") from exc
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    validation = manifest.get("validation")
    if (
        digest != FROZEN_CONFIG_SHA256
        or manifest.get("config_sha256") != FROZEN_CONFIG_SHA256
        or not isinstance(validation, Mapping)
        or validation.get("config_sha256") != FROZEN_CONFIG_SHA256
    ):
        raise DatasetContractError(
            "confirmatory config bytes or manifest config SHA-256 differ from frozen identity"
        )


def _validate_annotated_tag(repo_root: Path) -> None:
    try:
        tag_type = subprocess.run(
            ("git", "cat-file", "-t", FROZEN_EXPERIMENT_TAG),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tag_commit = subprocess.run(
            ("git", "rev-parse", f"{FROZEN_EXPERIMENT_TAG}^{{}}"),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DatasetContractError("cannot verify the frozen annotated tag") from exc
    if tag_type != "tag" or tag_commit != FROZEN_RUNNER_COMMIT:
        raise DatasetContractError(
            "frozen experiment tag is not an annotated tag bound to the runner commit"
        )


def prepare_journey(raw: dict[str, Any]) -> dict[str, Any]:
    canonical = rebuild_views_from_events(raw)
    held_out_pairs = validate_raw_views(raw, canonical)
    raw_views = raw["views"]
    compact_views: list[dict[str, Any]] = []
    analysis_rows: list[AnalysisRow] = []
    for rebuilt, stored in zip(canonical, raw_views, strict=True):
        budget = int(rebuilt["nominal_budget"])
        compact_views.append(
            {
                "nominal_budget": budget,
                "common_support_no_repeat": bool(
                    stored["common_support_no_repeat"]
                ),
                "common_support_set_sha256": str(
                    stored["common_support_set_sha256"]
                ),
            }
        )
        analysis_rows.append(
            AnalysisRow(
                target=str(raw["target_node"]),
                truth=str(raw["truth"]),
                condition=str(raw["condition"]),
                replicate=int(raw["replicate"]),
                arm=str(raw["arm"]),
                budget=budget,
                argmax=(
                    str(rebuilt["argmax"])
                    if rebuilt["argmax"] is not None
                    else None
                ),
                converged=bool(rebuilt["converged"]),
                convergence_time=(
                    int(rebuilt["convergence_time"])
                    if rebuilt["convergence_time"] is not None
                    else None
                ),
                actual_administered_count=int(
                    rebuilt["actual_administered_count"]
                ),
                held_out_brier=(
                    float(stored["held_out_brier"])
                    if stored.get("held_out_brier") is not None
                    else None
                ),
                exact_item_repeat_fraction=float(
                    rebuilt["exact_item_repeat_fraction"]
                ),
                family_repeat_fraction=float(rebuilt["family_repeat_fraction"]),
                h1_h2_eligible=bool(raw["h1_h2_eligible"]),
                common_support_no_repeat=bool(
                    stored["common_support_no_repeat"]
                ),
                valid=bool(rebuilt["valid"]),
                prerequisite_count=int(rebuilt["prerequisite_count"]),
                prerequisite_share=float(rebuilt["prerequisite_share"]),
                direct_count=int(rebuilt["direct_count"]),
                unique_item_count=int(rebuilt["unique_item_count"]),
                unique_family_count=int(rebuilt["unique_family_count"]),
                exclusion_reason=(
                    None if rebuilt["valid"] else str(rebuilt["terminal_reason"])
                ),
            )
        )

    event_keys = (
        "target_node",
        "truth",
        "condition",
        "replicate",
        "arm",
        "persona_id",
        "position",
        "response_noise",
        "item_id",
        "family_id",
    )
    truth = str(raw["truth"])
    try:
        truth_index = FROZEN_STATES.index(truth)
    except ValueError as exc:
        raise DatasetContractError(f"unknown truth state in journey: {truth}") from exc
    journey_valid = all(row.valid for row in analysis_rows)
    exclusion_reason = None if journey_valid else str(raw.get("terminal_reason", ""))
    analysis_events: list[AnalysisEvent] = []
    for event in raw["events"]:
        item_type = event.get("item_type")
        generator_probability = event.get("generator_probability")
        production_probabilities = event.get("production_correct_probabilities")
        if item_type not in {"mcq", "numeric"}:
            raise DatasetContractError("event item_type must be mcq or numeric")
        if not isinstance(production_probabilities, list) or len(
            production_probabilities
        ) != len(FROZEN_STATES):
            raise DatasetContractError(
                "event production_correct_probabilities must have four states"
            )
        try:
            generator_value = float(generator_probability)
            production_value = float(production_probabilities[truth_index])
        except (TypeError, ValueError) as exc:
            raise DatasetContractError("event probabilities are not numeric") from exc
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (generator_value, production_value)
        ):
            raise DatasetContractError("event probabilities are outside [0, 1]")
        analysis_events.append(
            AnalysisEvent(
                target=str(raw["target_node"]),
                truth=truth,
                condition=str(raw["condition"]),
                replicate=int(raw["replicate"]),
                arm=str(raw["arm"]),
                position=int(event["position"]),
                item_type=str(item_type),
                generator_probability=generator_value,
                production_probability=production_value,
                valid=journey_valid,
                exclusion_reason=exclusion_reason,
            )
        )
    return {
        "target_node": str(raw["target_node"]),
        "truth": str(raw["truth"]),
        "condition": str(raw["condition"]),
        "replicate": int(raw["replicate"]),
        "arm": str(raw["arm"]),
        "persona_id": str(raw["persona_id"]),
        "held_out_outcomes": dict(raw["held_out_outcomes"]),
        "held_out_pairs": held_out_pairs,
        "events": [
            {key: event[key] for key in event_keys}
            for event in raw["events"]
        ],
        "analysis_events": tuple(analysis_events),
        "views": compact_views,
        "analysis_rows": tuple(analysis_rows),
    }
