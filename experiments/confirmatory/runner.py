"""Shard planning, execution, resume validation, and run provenance."""

from __future__ import annotations

import hashlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.s0_census import guarded_simulation_run

from .catalog import load_catalog_context
from .config import REPO_ROOT
from .models import CatalogContext, UnitSpec
from . import provenance
from .simulation import production_model_id, run_paired_unit
from .storage import (
    read_shard_records,
    shard_file_path,
    validate_shard,
    write_run_manifest_atomic,
    write_shards_atomic,
)

verify_repository_binding = provenance.verify_repository_binding


@dataclass(frozen=True)
class ShardSpec:
    target_node: str
    truth: str
    condition: str

    @property
    def shard_id(self) -> str:
        return (
            f"target={self.target_node}|truth={self.truth}|"
            f"condition={self.condition}"
        )


def validate_definition(config, context: CatalogContext | None = None) -> dict[str, Any]:
    provenance.assert_frozen_config(config)
    catalog_context = context or load_catalog_context(config)
    return {
        "open_nodes": len(catalog_context.targets),
        "h1_h2_eligible": len(catalog_context.h1_h2_eligible_targets),
        "h1_h2_excluded": len(catalog_context.h1_h2_excluded_targets),
        "truth_states": len(config.truth_states),
        "arms": len(config.arms),
        "conditions": len(config.conditions),
        "replicates": config.replicates,
        "expected_journeys": config.expected_journeys(
            open_node_count=len(catalog_context.targets)
        ),
        "config_sha256": config.sha256,
        "common_support_targets": {
            str(budget): sum(
                pools.common_support_no_repeat[budget]
                for pools in catalog_context.targets.values()
            )
            for budget in config.budgets
        },
        "common_support_set_sha256": {
            str(budget): next(iter(catalog_context.targets.values()))
            .common_support_set_sha256[budget]
            for budget in config.budgets
        },
    }


def plan_shards(config, context: CatalogContext) -> tuple[ShardSpec, ...]:
    return tuple(
        ShardSpec(target, truth, condition)
        for target in sorted(context.targets)
        for truth in config.truth_states
        for condition in config.conditions
    )


def execute(
    config,
    *,
    output_root: str | Path,
    run_id: str,
    workers: int,
    resume: bool,
    limit_shards: int | None,
    runner_commit: str,
    experiment_tag: str,
    run_started_at_utc: str,
    context: CatalogContext | None = None,
    repo_root: str | Path = REPO_ROOT,
    temp_root: str | Path = "/tmp/yher_sprint2",
    verify_repository_binding: bool = True,
) -> dict[str, Any]:
    if not sys.dont_write_bytecode:
        raise RuntimeError(
            "confirmatory data execution requires python -B or "
            "PYTHONDONTWRITEBYTECODE=1 from process start"
        )
    return _execute_guarded(
        config,
        output_root=output_root,
        run_id=run_id,
        workers=workers,
        resume=resume,
        limit_shards=limit_shards,
        runner_commit=runner_commit,
        experiment_tag=experiment_tag,
        run_started_at_utc=run_started_at_utc,
        context=context,
        repo_root=repo_root,
        temp_root=temp_root,
        verify_repository_binding=verify_repository_binding,
    )


def _execute_guarded(
    config,
    *,
    output_root: str | Path,
    run_id: str,
    workers: int,
    resume: bool,
    limit_shards: int | None,
    runner_commit: str,
    experiment_tag: str,
    run_started_at_utc: str,
    context: CatalogContext | None = None,
    repo_root: str | Path = REPO_ROOT,
    temp_root: str | Path = "/tmp/yher_sprint2",
    verify_repository_binding: bool = True,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", run_id):
        raise ValueError("run_id must be one safe non-dot path segment")
    if not re.fullmatch(r"[0-9a-f]{40}", runner_commit):
        raise ValueError("runner_commit must be a full lowercase git SHA")
    if not experiment_tag.strip():
        raise ValueError("experiment_tag must be supplied at execution")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", run_started_at_utc):
        raise ValueError("run_started_at_utc must be an explicit UTC ISO-8601 timestamp")
    provenance.validate_run_timestamp(config, run_started_at_utc)
    if workers < 1:
        raise ValueError("workers must be positive")
    repository_binding = (
        provenance.verify_repository_binding(
            repo_root,
            runner_commit=runner_commit,
            experiment_tag=experiment_tag,
            analysis_plan_commit=str(config.raw["analysis_plan_commit"]),
        )
        if verify_repository_binding
        else {"verified": False, "reason": "test_only_injected_context"}
    )
    catalog_context = context or load_catalog_context(config)
    validation = validate_definition(config, catalog_context)
    output = provenance.confirmatory_output_path(
        output_root,
        run_id=run_id,
        repo_root=repo_root,
        temp_root=temp_root,
    )
    specs = list(plan_shards(config, catalog_context))
    if limit_shards is not None:
        if limit_shards < 1:
            raise ValueError("limit_shards must be positive")
        specs = specs[:limit_shards]
    model_id = production_model_id(runner_commit)
    manifest_metadata = {
        "analysis_plan_commit": str(config.raw["analysis_plan_commit"]),
        "census_analysis_plan_commit": str(
            config.raw["census_analysis_plan_commit"]
        ),
        "runner_commit": runner_commit,
        "experiment_tag": experiment_tag,
        "config_sha256": config.sha256,
        "seed_derivation_version": str(config.raw["seed_derivation_version"]),
        "master_seed": config.master_seed,
        "bootstrap_seed": int(config.raw["bootstrap_seed"]),
        "census_seed": int(config.raw["census_seed"]),
        "input_sha256": catalog_context.input_sha256,
        "config_frozen_at_utc": str(config.raw["config_frozen_at_utc"]),
        "run_started_at_utc": run_started_at_utc,
        "repository_head": runner_commit,
        "repository_binding": repository_binding,
    }
    totals = {"written": 0, "skipped": 0}
    for start in range(0, len(specs), workers):
        batch = specs[start : start + workers]
        pending = [
            spec
            for spec in batch
            if not (
                resume
                and _complete_shard(
                    shard_file_path(output, spec.shard_id),
                    config=config,
                    shard_spec=spec,
                    model_id=model_id,
                    provenance=manifest_metadata,
                )
            )
        ]
        totals["skipped"] += len(batch) - len(pending)
        built: list[tuple[str, list[dict[str, Any]]]] = []
        assertion: dict[str, Any] | None = None
        if pending:
            with guarded_simulation_run(repo_root) as isolation:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    built = list(
                        executor.map(
                            lambda spec: (
                                spec.shard_id,
                                build_shard_records(
                                    spec,
                                    config=config,
                                    context=catalog_context,
                                    model_id=model_id,
                                    provenance=manifest_metadata,
                                ),
                            ),
                            pending,
                        )
                    )
            assertion = provenance.isolation_assertion(isolation)
        if built:
            result = write_shards_atomic(
                dict(built),
                output_dir=output,
                config_sha256=config.sha256,
                workers=workers,
                resume=False,
                repo_root=repo_root,
                temp_root=temp_root,
                manifest_metadata={
                    **manifest_metadata,
                    "protected_filesystem_assertion": assertion,
                },
            )
            totals["written"] += result["written"]
        for spec in pending:
            if not _complete_shard(
                shard_file_path(output, spec.shard_id),
                config=config,
                shard_spec=spec,
                model_id=model_id,
                provenance=manifest_metadata,
            ):
                raise ValueError(f"written shard is incomplete: {spec.shard_id}")

    shard_entries = []
    for spec in specs:
        path = shard_file_path(output, spec.shard_id)
        shard_entries.append(
            {
                "shard_id": spec.shard_id,
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    full_shard_count = len(catalog_context.targets) * len(config.truth_states) * len(
        config.conditions
    )
    run_isolation = provenance.aggregate_isolation_assertions(
        [shard_file_path(output, spec.shard_id) for spec in specs]
    )
    manifest = {
        "simulated": True,
        "persona_id": f"confirmatory-run:{run_id}",
        "provider": str(config.raw["provider"]),
        "model_id": model_id,
        "record_type": "confirmatory_run_manifest",
        "run_id": run_id,
        "status": "complete" if len(specs) == full_shard_count else "smoke_partial",
        "full_grid_complete": len(specs) == full_shard_count,
        "selected_shard_count": len(specs),
        "full_shard_count": full_shard_count,
        "expected_journey_count": len(specs) * config.replicates * len(config.arms),
        "shards": shard_entries,
        "validation": validation,
        "protected_filesystem_assertion": run_isolation,
        **manifest_metadata,
    }
    manifest_path = write_run_manifest_atomic(
        manifest,
        output_dir=output,
        repo_root=repo_root,
        temp_root=temp_root,
    )
    return {
        **totals,
        "selected_shards": len(specs),
        "manifest": str(manifest_path),
        "status": manifest["status"],
        "protected_filesystem_assertion": run_isolation,
    }


def build_shard_records(
    shard_spec: ShardSpec,
    *,
    config,
    context: CatalogContext,
    model_id: str,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pools = context.targets[shard_spec.target_node]
    records: list[dict[str, Any]] = []
    for replicate in range(config.replicates):
        unit = UnitSpec(
            target_node=shard_spec.target_node,
            truth=shard_spec.truth,
            condition=shard_spec.condition,
            replicate=replicate,
        )
        records.extend(
            run_paired_unit(
                pools,
                unit,
                config,
                model_id=model_id,
                provenance=provenance,
            )
        )
    validate_shard_records(records, config=config, shard_spec=shard_spec)
    return records


def validate_shard_records(
    records: Iterable[Mapping[str, Any]],
    *,
    config,
    shard_spec: ShardSpec,
    model_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    rows = list(records)
    expected_count = config.replicates * len(config.arms)
    if len(rows) != expected_count:
        raise ValueError(f"shard must contain {expected_count} journeys")
    expected_keys = {
        (replicate, arm)
        for replicate in range(config.replicates)
        for arm in config.arms
    }
    actual_keys: set[tuple[int, str]] = set()
    for row in rows:
        if row.get("record_type") != "confirmatory_journey":
            raise ValueError("shard contains a non-journey record")
        if model_id is not None and row.get("model_id") != model_id:
            raise ValueError("journey model_id differs from the frozen runner")
        if provenance is not None and row.get("provenance") != dict(provenance):
            raise ValueError("journey provenance differs from the frozen run")
        if (
            row.get("target_node"),
            row.get("truth"),
            row.get("condition"),
        ) != (
            shard_spec.target_node,
            shard_spec.truth,
            shard_spec.condition,
        ):
            raise ValueError("journey is in the wrong target/truth/condition shard")
        key = (int(row["replicate"]), str(row["arm"]))
        actual_keys.add(key)
        actual = int(row["actual_administered_count"])
        events = list(row["events"])
        if provenance is not None and any(
            event.get("provenance") != dict(provenance) for event in events
        ):
            raise ValueError("event provenance differs from the frozen run")
        if actual != len(events):
            raise ValueError("actual administered count differs from event count")
        counters = row["call_counters"]
        if int(counters["mastery_observe"]) != actual:
            raise ValueError("observe calls must equal administrations")
        expected_stop_calls = actual + int(actual == config.max_items)
        if int(counters["selector_should_stop"]) != expected_stop_calls:
            raise ValueError("should_stop calls must equal administrations")
        expected_select = actual if row["arm"] == "A" else 0
        if int(counters["selector_select_next"]) != expected_select:
            raise ValueError("select_next call count violates arm contract")
        if row["terminal_reason"] == "budget_exhausted" and row["converged"]:
            raise ValueError("budget exhaustion cannot be labeled converged")
        if tuple(view["nominal_budget"] for view in row["views"]) != config.budgets:
            raise ValueError("journey nominal views differ from frozen budgets")
    if actual_keys != expected_keys:
        raise ValueError("shard is missing a replicate/arm pairing")


def _complete_shard(
    path: Path,
    *,
    config,
    shard_spec: ShardSpec,
    model_id: str,
    provenance: Mapping[str, Any],
) -> bool:
    expected_manifest = {
        **dict(provenance),
        "shard_id": shard_spec.shard_id,
        "model_id": model_id,
    }
    if not validate_shard(
        path,
        config_sha256=config.sha256,
        expected_manifest=expected_manifest,
        require_isolation_attestation=True,
    ):
        return False
    try:
        validate_shard_records(
            read_shard_records(path),
            config=config,
            shard_spec=shard_spec,
            model_id=model_id,
            provenance=provenance,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True
