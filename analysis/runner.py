"""End-to-end entry point for the frozen programmatic analysis."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .contracts import validate_programmatic_grid
from .dataset import DatasetContractError, load_manifest_dataset
from .metrics import AnalysisRow
from .no_repeat import validate_no_repeat_sets
from .prepare import (
    EXPECTED_JOURNEYS,
    EXPECTED_REPLICATES,
    prepare_journey,
    validate_frozen_manifest,
)
from .provenance import collect_analysis_provenance, verify_analysis_provenance
from .results import (
    ResultBundle,
    build_results,
    replace_results_contract,
    write_results,
)


def run_formal_analysis(
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    results_contract_path: Path | str | None = None,
    verified_analysis_provenance: Mapping[str, object] | None = None,
) -> ResultBundle:
    manifest_file = Path(manifest_path).resolve()
    _assert_output_paths_do_not_overlap_raw(
        manifest_file,
        output_dir,
        results_contract_path,
    )
    try:
        manifest_bytes = manifest_file.read_bytes()
        manifest_preview = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"cannot read manifest: {manifest_file}") from exc
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    repo_root = Path(__file__).resolve().parents[1]
    validate_frozen_manifest(
        manifest_preview,
        manifest_sha256=manifest_sha256,
        repo_root=repo_root,
    )
    if verified_analysis_provenance is None:
        analysis_provenance = collect_analysis_provenance(repo_root)
    else:
        analysis_provenance = dict(verified_analysis_provenance)
        verify_analysis_provenance(repo_root, analysis_provenance)

    dataset = load_manifest_dataset(
        manifest_file,
        projector=prepare_journey,
        manifest_bytes=manifest_bytes,
    )
    grid = validate_programmatic_grid(
        dataset.journeys,
        expected_journeys=EXPECTED_JOURNEYS,
        expected_replicates=EXPECTED_REPLICATES,
        invalid_primary_keys=dataset.invalid_primary_keys,
    )
    no_repeat = validate_no_repeat_sets(dataset.journeys, dataset.manifest)
    rows = tuple(
        row
        for journey in dataset.journeys
        for row in journey["analysis_rows"]
    )
    events = tuple(
        event
        for journey in dataset.journeys
        for event in journey["analysis_events"]
    )
    rows, events = _complete_schema_invalid_pairs(
        rows,
        events,
        invalid_primary_keys=dataset.invalid_primary_keys,
    )
    _validate_canonical_rows(rows)
    validation = dataset.manifest["validation"]
    for budget, support in no_repeat.items():
        if validation["common_support_targets"][str(budget)] != support.n_target:
            raise DatasetContractError(
                f"manifest no-repeat count mismatch at budget {budget}"
            )
        if validation["common_support_set_sha256"][str(budget)] != support.sha256:
            raise DatasetContractError(
                f"manifest no-repeat hash mismatch at budget {budget}"
            )

    bundle = build_results(
        rows,
        events=events,
        raw_hash=dataset.raw_hash,
        no_repeat_sets=no_repeat,
        bootstrap_iterations=10_000,
        source_provenance={
            "run_id": str(dataset.manifest.get("run_id", "")),
            "run_started_at_utc": str(
                dataset.manifest.get("run_started_at_utc", "")
            ),
            "runner_commit": str(dataset.manifest.get("runner_commit", "")),
            "experiment_tag": str(dataset.manifest.get("experiment_tag", "")),
            "config_sha256": str(dataset.manifest.get("config_sha256", "")),
            "source_manifest_sha256": manifest_sha256,
            "analysis_plan_commit": str(
                dataset.manifest.get("analysis_plan_commit", "")
            ),
            "analysis_plan_sha256": str(
                (dataset.manifest.get("input_sha256") or {})
                .get("confirmatory_analysis_plan", {})
                .get("sha256", "")
            ),
        },
        analysis_provenance=analysis_provenance,
        schema_invalid_count=dataset.invalid_record_count,
        schema_invalid_reasons=dataset.invalid_reasons,
        intended_journey_count=dataset.intended_journey_count,
    )
    bundle = replace(
        bundle,
        validation={
            **bundle.validation,
            "journey_count": grid.journey_count,
            "programmatic_primary_key_count": grid.primary_key_count,
            "paired_replicate_count": grid.pair_count,
            "target_count": grid.target_count,
            "manifest_shard_count": len(dataset.shard_paths),
            "raw_hash": dataset.raw_hash,
            "schema_invalid_count": dataset.invalid_record_count,
            "schema_invalid_reasons": dict(dataset.invalid_reasons),
            "intended_journey_count": dataset.intended_journey_count,
        },
    )
    contract_block = write_results(bundle, output_dir)
    if results_contract_path is not None:
        replace_results_contract(
            results_contract_path,
            contract_block.read_text(encoding="utf-8"),
        )
    return bundle


def _complete_schema_invalid_pairs(
    rows: tuple[AnalysisRow, ...],
    events: tuple[object, ...],
    *,
    invalid_primary_keys: tuple[tuple[object, ...], ...],
):
    from .metrics import AnalysisEvent

    if not invalid_primary_keys:
        return rows, events
    invalid_pairs = {tuple(key[:4]) for key in invalid_primary_keys}
    completed_rows = [
        replace(row, valid=False, exclusion_reason="paired_schema_invalid")
        if (row.target, row.truth, row.condition, row.replicate) in invalid_pairs
        else row
        for row in rows
    ]
    completed_events = [
        replace(event, valid=False, exclusion_reason="paired_schema_invalid")
        if isinstance(event, AnalysisEvent)
        and (event.target, event.truth, event.condition, event.replicate)
        in invalid_pairs
        else event
        for event in events
    ]
    existing = {
        (row.target, row.truth, row.condition, row.replicate, row.arm, row.budget)
        for row in completed_rows
    }
    for raw_key in invalid_primary_keys:
        target, truth, condition, replicate, arm = raw_key
        target_rows = [row for row in rows if row.target == str(target)]
        if not target_rows:
            raise DatasetContractError(
                f"cannot recover quarantine metadata for target {target!r}"
            )
        for budget in (9, 15, 25):
            key = (str(target), str(truth), str(condition), int(replicate), str(arm), budget)
            if key in existing:
                continue
            exemplar = next(
                (row for row in target_rows if row.budget == budget),
                target_rows[0],
            )
            completed_rows.append(
                AnalysisRow(
                    target=str(target),
                    truth=str(truth),
                    condition=str(condition),
                    replicate=int(replicate),
                    arm=str(arm),
                    budget=budget,
                    argmax=None,
                    converged=False,
                    convergence_time=None,
                    actual_administered_count=0,
                    held_out_brier=None,
                    exact_item_repeat_fraction=0.0,
                    family_repeat_fraction=0.0,
                    h1_h2_eligible=exemplar.h1_h2_eligible,
                    common_support_no_repeat=exemplar.common_support_no_repeat,
                    valid=False,
                    exclusion_reason="paired_schema_invalid",
                )
            )
            existing.add(key)
    completed_rows.sort(
        key=lambda row: (
            row.target,
            row.truth,
            row.condition,
            row.replicate,
            row.arm,
            row.budget,
        )
    )
    completed_events.sort(
        key=lambda event: (
            getattr(event, "target", ""),
            getattr(event, "truth", ""),
            getattr(event, "condition", ""),
            getattr(event, "replicate", -1),
            getattr(event, "arm", ""),
            getattr(event, "position", -1),
        )
    )
    return tuple(completed_rows), tuple(completed_events)


def _assert_output_paths_do_not_overlap_raw(
    manifest_path: Path | str,
    output_dir: Path | str,
    results_contract_path: Path | str | None,
) -> None:
    raw_root = Path(manifest_path).resolve().parent
    destinations = [Path(output_dir).resolve()]
    if results_contract_path is not None:
        destinations.append(Path(results_contract_path).resolve())
    if any(
        destination == raw_root
        or destination in raw_root.parents
        or raw_root in destination.parents
        for destination in destinations
    ):
        raise DatasetContractError(
            "formal analysis output paths cannot overlap raw collection"
        )


def _validate_canonical_rows(rows: tuple[AnalysisRow, ...]) -> None:
    expected = EXPECTED_JOURNEYS * 3
    if len(rows) != expected:
        raise DatasetContractError(
            f"canonical view count mismatch: {len(rows)} != {expected}"
        )
    keys = {
        (
            row.target,
            row.truth,
            row.condition,
            row.replicate,
            row.arm,
            row.budget,
        )
        for row in rows
    }
    if len(keys) != expected:
        raise DatasetContractError("duplicate canonical analysis key")
    budgets_by_journey: dict[tuple[object, ...], set[int]] = {}
    for row in rows:
        key = (row.target, row.truth, row.condition, row.replicate, row.arm)
        budgets_by_journey.setdefault(key, set()).add(row.budget)
    if any(budgets != {9, 15, 25} for budgets in budgets_by_journey.values()):
        raise DatasetContractError("canonical journey lacks a 9/15/25 budget view")
