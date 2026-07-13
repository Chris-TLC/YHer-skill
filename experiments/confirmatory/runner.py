"""Shard planning, execution, resume validation, and run provenance."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.s0_census import guarded_simulation_run, require_simulation_output_path

from .catalog import load_catalog_context
from .config import REPO_ROOT
from .models import CatalogContext, UnitSpec
from .simulation import production_model_id, run_paired_unit
from .storage import (
    read_shard_records,
    shard_file_path,
    validate_shard,
    write_run_manifest_atomic,
    write_shards_atomic,
)


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
    _assert_frozen_config(config)
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
    context: CatalogContext | None = None,
    repo_root: str | Path = REPO_ROOT,
    temp_root: str | Path = "/tmp/yher_sprint2",
) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    with guarded_simulation_run(repo_root) as isolation:
        result = _execute_guarded(
            config,
            output_root=output_root,
            run_id=run_id,
            workers=workers,
            resume=resume,
            limit_shards=limit_shards,
            runner_commit=runner_commit,
            experiment_tag=experiment_tag,
            context=context,
            repo_root=repo_root,
            temp_root=temp_root,
        )
    assertion = {
        "before_sha256": isolation["before"]["digest"],
        "after_sha256": isolation["after"]["digest"],
        "unchanged": isolation["unchanged"],
        "coverage": isolation["before"]["coverage"],
    }
    manifest_path = Path(result["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protected_filesystem_assertion"] = assertion
    write_run_manifest_atomic(
        manifest,
        output_dir=manifest_path.parent,
        repo_root=repo_root,
        temp_root=temp_root,
    )
    return {**result, "protected_filesystem_assertion": assertion}


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
    context: CatalogContext | None = None,
    repo_root: str | Path = REPO_ROOT,
    temp_root: str | Path = "/tmp/yher_sprint2",
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore, and dash")
    if not re.fullmatch(r"[0-9a-f]{40}", runner_commit):
        raise ValueError("runner_commit must be a full lowercase git SHA")
    if not experiment_tag.strip():
        raise ValueError("experiment_tag must be supplied at execution")
    if workers < 1:
        raise ValueError("workers must be positive")
    catalog_context = context or load_catalog_context(config)
    validation = validate_definition(config, catalog_context)
    output = require_simulation_output_path(
        Path(output_root) / run_id,
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
        "runner_commit": runner_commit,
        "experiment_tag": experiment_tag,
        "config_sha256": config.sha256,
        "seed_derivation_version": str(config.raw["seed_derivation_version"]),
        "master_seed": config.master_seed,
        "bootstrap_seed": int(config.raw["bootstrap_seed"]),
        "census_seed": int(config.raw["census_seed"]),
        "input_sha256": catalog_context.input_sha256,
        "created_at_utc": str(config.raw["artifact_created_at_utc"]),
        "repository_head": runner_commit,
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
                )
            )
        ]
        totals["skipped"] += len(batch) - len(pending)
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
        if built:
            result = write_shards_atomic(
                dict(built),
                output_dir=output,
                config_sha256=config.sha256,
                workers=workers,
                resume=False,
                repo_root=repo_root,
                temp_root=temp_root,
                manifest_metadata=manifest_metadata,
            )
            totals["written"] += result["written"]
        for spec in pending:
            if not _complete_shard(
                shard_file_path(output, spec.shard_id),
                config=config,
                shard_spec=spec,
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
        if actual != len(events):
            raise ValueError("actual administered count differs from event count")
        counters = row["call_counters"]
        if int(counters["mastery_observe"]) != actual:
            raise ValueError("observe calls must equal administrations")
        if int(counters["selector_should_stop"]) != actual:
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


def _complete_shard(path: Path, *, config, shard_spec: ShardSpec) -> bool:
    if not validate_shard(path, config_sha256=config.sha256):
        return False
    try:
        validate_shard_records(
            read_shard_records(path),
            config=config,
            shard_spec=shard_spec,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _assert_frozen_config(config) -> None:
    exact = {
        "truth_states": ("M", "P", "C", "U"),
        "arms": ("A", "B", "C"),
        "conditions": ("matched", "misspecified"),
        "budgets": (9, 15, 25),
    }
    for field, expected in exact.items():
        if tuple(getattr(config, field)) != expected:
            raise ValueError(f"frozen confirmatory {field} changed")
    scalar = {
        "replicates": 50,
        "max_items": 25,
        "stop_budget_items": 26,
        "master_seed": 20260713,
    }
    for field, expected in scalar.items():
        if int(getattr(config, field)) != expected:
            raise ValueError(f"frozen confirmatory {field} changed")
    if tuple(float(value) for value in config.raw["fixed_difficulty_ladder"]) != (
        0.25,
        0.5,
        0.75,
        1.0,
    ):
        raise ValueError("frozen fixed difficulty ladder changed")
