from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shutil

import pytest


ROW_WEIGHT = "equal_target_then_replicate"
PAIR_WEIGHT = "equal_target_then_paired_replicate"
TIME_WEIGHT = "equal_target_then_weighted_median; NC=budget+1"
DEGRADATION_WEIGHT = (
    "equal_target_then_paired_replicate; "
    "matched_minus_misspecified_condition_contrast"
)
ITEM_WEIGHT = (
    "equal_target_then_event; misspecified_event_level; "
    "target_stratified_paired_replicate_resample; journey_cluster_preserved; "
    "bootstrap_iterations=10000; diagnostic_only_not_item_type_H1_H2_estimand"
)
POSTHOC_ROW_WEIGHT = "exploratory_posthoc; equal_target_then_replicate"
POSTHOC_PAIR_WEIGHT = (
    "exploratory_posthoc; equal_target_then_replicate paired indicator difference"
)


def _expected_estimand_rows() -> tuple[tuple[object, ...], ...]:
    all_truth = ("M", "P", "C", "U")
    return (
        ("p_rescue.full.matched.b15.arm_A", "eligible", ("matched",), None, 15, ("P",), ("A",), ROW_WEIGHT, "target_replicate_rows"),
        ("p_rescue.full.matched.b15.arm_B", "eligible", ("matched",), None, 15, ("P",), ("B",), ROW_WEIGHT, "target_replicate_rows"),
        ("h1.primary.matched.b15.rescue_A_minus_B", "eligible", ("matched",), None, 15, ("P",), ("A", "B"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h1.no_repeat.matched.b15.arm_A", "common_b15", ("matched",), None, 15, ("P",), ("A",), ROW_WEIGHT, "target_replicate_rows"),
        ("h1.no_repeat.matched.b15.arm_B", "common_b15", ("matched",), None, 15, ("P",), ("B",), ROW_WEIGHT, "target_replicate_rows"),
        ("h1.no_repeat.matched.b15.rescue_A_minus_B", "common_b15", ("matched",), None, 15, ("P",), ("A", "B"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("c_misdiagnosis.full.matched.b9.arm_A", "eligible", ("matched",), None, 9, ("C",), ("A",), ROW_WEIGHT, "target_replicate_rows"),
        ("c_misdiagnosis.full.matched.b9.arm_B", "eligible", ("matched",), None, 9, ("C",), ("B",), ROW_WEIGHT, "target_replicate_rows"),
        ("c_misdiagnosis.full.matched.b9.arm_C", "eligible", ("matched",), None, 9, ("C",), ("C",), ROW_WEIGHT, "target_replicate_rows"),
        ("h2.primary.matched.b9.harm_C_minus_A", "eligible", ("matched",), None, 9, ("C",), ("C", "A"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h2.primary.matched.b9.no_harm_A_minus_B", "eligible", ("matched",), None, 9, ("C",), ("A", "B"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h2.no_repeat.matched.b9.arm_A", "common_b9", ("matched",), None, 9, ("C",), ("A",), ROW_WEIGHT, "target_replicate_rows"),
        ("h2.no_repeat.matched.b9.arm_B", "common_b9", ("matched",), None, 9, ("C",), ("B",), ROW_WEIGHT, "target_replicate_rows"),
        ("h2.no_repeat.matched.b9.arm_C", "common_b9", ("matched",), None, 9, ("C",), ("C",), ROW_WEIGHT, "target_replicate_rows"),
        ("h2.no_repeat.matched.b9.harm_C_minus_A", "common_b9", ("matched",), None, 9, ("C",), ("C", "A"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h2.no_repeat.matched.b9.no_harm_A_minus_B", "common_b9", ("matched",), None, 9, ("C",), ("A", "B"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h3.matched.b15.terminal_accuracy_A_minus_B", "full", ("matched",), None, 15, all_truth, ("A", "B"), PAIR_WEIGHT, "paired_target_state_replicate_contrasts"),
        ("h3.matched.b15.time_to_confidence.arm_A", "full", ("matched",), None, 15, all_truth, ("A",), TIME_WEIGHT, "target_state_replicate_rows_time_coded"),
        ("h3.matched.b15.time_to_confidence.arm_B", "full", ("matched",), None, 15, all_truth, ("B",), TIME_WEIGHT, "target_state_replicate_rows_time_coded"),
        ("h4.misspecified.b15.rescue_A_minus_B", "eligible", ("misspecified",), None, 15, ("P",), ("A", "B"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h4.misspecified.b9.harm_C_minus_A", "eligible", ("misspecified",), None, 9, ("C",), ("C", "A"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h4.misspecified.b9.no_harm_A_minus_B", "eligible", ("misspecified",), None, 9, ("C",), ("A", "B"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h4.degradation.h1_rescue.matched_minus_misspecified", "eligible", ("matched", "misspecified"), "matched_minus_misspecified", 15, ("P",), ("A", "B"), DEGRADATION_WEIGHT, "paired_condition_target_replicate_contrasts"),
        ("h4.degradation.h2_harm.matched_minus_misspecified", "eligible", ("matched", "misspecified"), "matched_minus_misspecified", 9, ("C",), ("C", "A"), DEGRADATION_WEIGHT, "paired_condition_target_replicate_contrasts"),
        ("h4.degradation.h2_no_harm.matched_minus_misspecified", "eligible", ("matched", "misspecified"), "matched_minus_misspecified", 9, ("C",), ("A", "B"), DEGRADATION_WEIGHT, "paired_condition_target_replicate_contrasts"),
        ("h1.no_repeat.misspecified.b15.rescue_A_minus_B", "common_b15", ("misspecified",), None, 15, ("P",), ("A", "B"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h2.no_repeat.misspecified.b9.harm_C_minus_A", "common_b9", ("misspecified",), None, 9, ("C",), ("C", "A"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("h2.no_repeat.misspecified.b9.no_harm_A_minus_B", "common_b9", ("misspecified",), None, 9, ("C",), ("A", "B"), PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("misspecification.item_type.mcq.generator_minus_production", "item_type_mcq", ("misspecified",), None, None, all_truth, ("generator_probability", "production_probability"), ITEM_WEIGHT, "misspecified_event_pairs"),
        ("misspecification.item_type.numeric.generator_minus_production", "item_type_numeric", ("misspecified",), None, None, all_truth, ("generator_probability", "production_probability"), ITEM_WEIGHT, "misspecified_event_pairs"),
        ("outcome_by_view.matched.b15.arm_A.truth_P.terminal_accuracy", "full", ("matched",), None, 15, ("P",), ("A",), ROW_WEIGHT, "target_replicate_rows"),
        ("outcome_by_view.matched.b15.arm_A.truth_P.correct_convergence", "full", ("matched",), None, 15, ("P",), ("A",), ROW_WEIGHT, "target_replicate_rows"),
        ("exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.terminal_accuracy", "eligible", ("matched",), None, 15, ("P",), ("A",), POSTHOC_ROW_WEIGHT, "target_replicate_rows"),
        ("exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.correct_convergence", "eligible", ("matched",), None, 15, ("P",), ("A",), POSTHOC_ROW_WEIGHT, "target_replicate_rows"),
        ("exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.terminal_minus_correct_convergence", "eligible", ("matched",), None, 15, ("P",), ("A",), POSTHOC_PAIR_WEIGHT, "paired_target_replicate_contrasts"),
        ("exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.direct_count", "eligible", ("matched",), None, 15, ("P",), ("A",), POSTHOC_ROW_WEIGHT, "target_replicate_rows"),
        ("exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.prerequisite_count", "eligible", ("matched",), None, 15, ("P",), ("A",), POSTHOC_ROW_WEIGHT, "target_replicate_rows"),
        ("exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.prerequisite_share", "eligible", ("matched",), None, 15, ("P",), ("A",), POSTHOC_ROW_WEIGHT, "target_replicate_rows"),
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_raw_manifest(
    root: Path,
    *,
    targets: tuple[str, ...] = ("alpha", "beta", "gamma"),
    eligible: tuple[str, ...] = ("alpha", "beta"),
    common_by_budget: dict[int, tuple[str, ...]] | None = None,
) -> Path:
    common_by_budget = common_by_budget or {
        9: ("alpha",),
        15: ("alpha",),
        25: (),
    }
    rows: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    for index, target in enumerate(targets):
        filename = f"shard-{index}.jsonl"
        row = {
            "target_node": target,
            "condition": "misspecified",
            "truth": "M",
            "replicate": 0,
            "arm": "A",
            "h1_h2_eligible": target in eligible,
            "events": [
                {"item_type": "mcq"},
                *([{"item_type": "numeric"}] if target in eligible else []),
            ],
            "views": [
                {
                    "nominal_budget": budget,
                    "common_support_no_repeat": target in common_by_budget[budget],
                }
                for budget in (9, 15, 25)
            ],
        }
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        (root / filename).write_text(payload, encoding="utf-8")
        shards.append(
            {
                "filename": filename,
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "shard_id": f"target={target}|truth=M|condition=misspecified",
            }
        )
    manifest = {
        "record_type": "confirmatory_run_manifest",
        "run_id": "confirmatory-v1",
        "status": "complete",
        "simulated": True,
        "experiment_tag": "test-freeze",
        "run_started_at_utc": "2026-07-13T00:00:00Z",
        "runner_commit": "1" * 40,
        "analysis_plan_commit": "2" * 40,
        "config_sha256": "config-hash",
        "expected_journey_count": len(targets),
        "full_shard_count": len(targets),
        "selected_shard_count": len(targets),
        "full_grid_complete": True,
        "shards": shards,
        "input_sha256": {
            "confirmatory_analysis_plan": {"sha256": "plan-hash"}
        },
        "validation": {
            "open_nodes": len(targets),
            "h1_h2_eligible": len(eligible),
            "h1_h2_excluded": len(targets) - len(eligible),
            "common_support_targets": {
                str(budget): len(common_by_budget[budget])
                for budget in (9, 15, 25)
            },
            "expected_journeys": len(targets),
        },
    }
    path = root / "manifest.json"
    path.write_bytes(_canonical(manifest) + b"\n")
    return path


def _write_registry(
    path: Path,
    *,
    raw_hash: str = "raw-aggregate",
    rows: list[dict[str, object]] | None = None,
) -> Path:
    rows = rows or [
        {
            "metric_id": "h1.primary.matched.b15.rescue_A_minus_B",
            "value": 0.12,
            "numerator": 12,
            "denominator": 100,
            "weighting": PAIR_WEIGHT,
            "n_target": 2,
            "n_pair": 100,
            "raw_hash": raw_hash,
            "ci_low": 0.1,
            "ci_high": 0.14,
        },
        {
            "metric_id": "outcome_by_view.matched.b15.arm_A.truth_P.terminal_accuracy",
            "value": 0.8,
            "numerator": 80,
            "denominator": 100,
            "weighting": "equal_target_then_replicate",
            "n_target": 3,
            "n_pair": 100,
            "raw_hash": raw_hash,
            "ci_low": 0.7,
            "ci_high": 0.9,
        },
        {
            "metric_id": "outcome_by_view.matched.b15.arm_A.truth_P.correct_convergence",
            "value": 0.2,
            "numerator": 20,
            "denominator": 100,
            "weighting": "equal_target_then_replicate",
            "n_target": 3,
            "n_pair": 100,
            "raw_hash": raw_hash,
            "ci_low": 0.1,
            "ci_high": 0.3,
        },
        {
            "metric_id": "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.terminal_accuracy",
            "value": 0.9,
            "numerator": 90,
            "denominator": 100,
            "weighting": "exploratory_posthoc; equal_target_then_replicate",
            "n_target": 2,
            "n_pair": 100,
            "raw_hash": raw_hash,
            "ci_low": 0.8,
            "ci_high": 0.95,
        },
        {
            "metric_id": "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.correct_convergence",
            "value": 0.3,
            "numerator": 30,
            "denominator": 100,
            "weighting": "exploratory_posthoc; equal_target_then_replicate",
            "n_target": 2,
            "n_pair": 100,
            "raw_hash": raw_hash,
            "ci_low": 0.2,
            "ci_high": 0.4,
        },
        {
            "metric_id": "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.terminal_minus_correct_convergence",
            "value": 0.6,
            "numerator": 60,
            "denominator": 100,
            "weighting": "exploratory_posthoc; equal_target_then_replicate paired indicator difference",
            "n_target": 2,
            "n_pair": 100,
            "raw_hash": raw_hash,
            "ci_low": 0.5,
            "ci_high": 0.7,
        },
    ]
    path.write_text(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _raw_aggregate_hash(raw_manifest: Path) -> str:
    manifest = json.loads(raw_manifest.read_text(encoding="utf-8"))
    shards = [
        {
            "filename": row["filename"],
            "sha256": hashlib.sha256(
                (raw_manifest.parent / row["filename"]).read_bytes()
            ).hexdigest(),
        }
        for row in manifest["shards"]
    ]
    return hashlib.sha256(
        _canonical(
            {
                "manifest_sha256": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
                "shards": shards,
            }
        )
    ).hexdigest()


def _complete_registry_rows(raw_hash: str) -> list[dict[str, object]]:
    from experiments import journal_binder

    rows: list[dict[str, object]] = []
    for metric_id in journal_binder.REPORTABLE_METRIC_IDS:
        if metric_id.startswith("misspecification.item_type.mcq."):
            n_target = 3
            weighting = ITEM_WEIGHT
        elif metric_id.startswith("misspecification.item_type.numeric."):
            n_target = 2
            weighting = ITEM_WEIGHT
        elif metric_id.startswith("outcome_by_view.") or metric_id.startswith("h3."):
            n_target = 3
            weighting = "equal_target_then_replicate"
        elif ".no_repeat." in metric_id:
            n_target = 1
            weighting = "equal_target_then_replicate"
        else:
            n_target = 2
            weighting = "equal_target_then_replicate"
        if "time_to_confidence" in metric_id:
            weighting = "equal_target_then_weighted_median; NC=budget+1"
        elif metric_id.endswith("terminal_minus_correct_convergence"):
            weighting = (
                "exploratory_posthoc; equal_target_then_replicate "
                "paired indicator difference"
            )
        elif metric_id.startswith("exploratory_posthoc."):
            weighting = "exploratory_posthoc; equal_target_then_replicate"
        elif not metric_id.startswith("misspecification.item_type.") and (
            "_minus_" in metric_id
            or ".degradation." in metric_id
            or metric_id.endswith("rescue_A_minus_B")
            or metric_id.endswith("harm_C_minus_A")
            or metric_id.endswith("no_harm_A_minus_B")
        ):
            weighting = PAIR_WEIGHT
            if ".degradation." in metric_id:
                weighting = DEGRADATION_WEIGHT
        rows.append(
            {
                "metric_id": metric_id,
                "value": 0.5,
                "numerator": 50,
                "denominator": 100,
                "weighting": weighting,
                "n_target": n_target,
                "n_pair": 100,
                "raw_hash": raw_hash,
                "ci_low": 0.4,
                "ci_high": 0.6,
            }
        )
    return rows


def _write_complete_h1_h4_bundle(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    raw_hash = _raw_aggregate_hash(raw)
    registry = _write_registry(
        root / "metric_registry.json",
        rows=_complete_registry_rows(raw_hash),
    )
    registry_ids = [
        row["metric_id"] for row in json.loads(registry.read_text(encoding="utf-8"))
    ]
    raw_manifest = json.loads(raw.read_text(encoding="utf-8"))
    source_provenance = {
        "analysis_plan_commit": raw_manifest["analysis_plan_commit"],
        "analysis_plan_sha256": raw_manifest["input_sha256"][
            "confirmatory_analysis_plan"
        ]["sha256"],
        "config_sha256": raw_manifest["config_sha256"],
        "experiment_tag": raw_manifest["experiment_tag"],
        "run_id": raw_manifest["run_id"],
        "run_started_at_utc": raw_manifest["run_started_at_utc"],
        "runner_commit": raw_manifest["runner_commit"],
        "source_manifest_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    }
    decisions = {
        "H1": "partially_supported",
        "H2": "not_supported",
        "H3": "supported",
        "H4": "supported",
        "H5": None,
    }
    hypothesis_evidence = {
        "H1": [
            "p_rescue.full.matched.b15.arm_A",
            "h1.primary.matched.b15.rescue_A_minus_B",
        ],
        "H2": [
            "h2.primary.matched.b9.harm_C_minus_A",
            "h2.primary.matched.b9.no_harm_A_minus_B",
        ],
        "H3": [
            "h3.matched.b15.terminal_accuracy_A_minus_B",
            "h3.matched.b15.time_to_confidence.arm_A",
            "h3.matched.b15.time_to_confidence.arm_B",
        ],
        "H4": list(
            __import__(
                "experiments.journal_binder",
                fromlist=["EXPECTED_HYPOTHESIS_EVIDENCE"],
            ).EXPECTED_HYPOTHESIS_EVIDENCE["H4"]
        ),
        "H5": [],
    }
    validation = {
        "manifest_shard_count": 3,
        "intended_journey_count": 3,
        "journey_count": 3,
        "programmatic_journey_count": 3,
        "programmatic_primary_key_count": 3,
        "valid_journey_count": 3,
        "structural_failure_count": 0,
        "schema_invalid_count": 0,
        "target_count": 3,
        "full_target_count": 3,
        "h1_h2_eligible_target_count": 2,
        "common_support_target_count_b9": 1,
        "common_support_target_count_b15": 1,
        "common_support_target_count_b25": 0,
        "raw_hash": raw_hash,
    }
    results_payload = {
        "source_provenance": source_provenance,
        "raw_hash": raw_hash,
        "registry_metric_ids": registry_ids,
        "decisions": decisions,
        "hypothesis_evidence": hypothesis_evidence,
        "validation": validation,
        "analysis_code_sha256": "3" * 64,
        "analysis_commit": "4" * 40,
    }
    results = root / "results.json"
    results.write_bytes(_canonical(results_payload) + b"\n")
    artifact_payload = {
        "files": {
            "metric_registry.json": hashlib.sha256(registry.read_bytes()).hexdigest(),
            "results.json": hashlib.sha256(results.read_bytes()).hexdigest(),
        },
        "source_provenance": source_provenance,
        "raw_hash": raw_hash,
        "registry_metric_ids": registry_ids,
        "analysis_code_sha256": results_payload["analysis_code_sha256"],
        "analysis_commit": results_payload["analysis_commit"],
    }
    artifact_manifest = root / "artifact_manifest.json"
    artifact_manifest.write_bytes(_canonical(artifact_payload) + b"\n")
    return {
        "raw": raw,
        "registry": registry,
        "results": results,
        "artifact_manifest": artifact_manifest,
    }


def _write_p2(root: Path, *, raw_manifest: Path) -> Path:
    root.mkdir()
    sources = root / "sources"
    sources.mkdir()
    nodes = ["alpha", "beta", *[f"library-node-{index:02d}" for index in range(2, 13)]]
    node_counts = {node: (4 if node in {"alpha", "beta"} else 5) for node in nodes}
    for node in nodes[2:7]:
        node_counts[node] += 1
    assert sum(node_counts.values()) == 68

    candidate_rows: list[dict[str, object]] = []
    runtime_rows: dict[str, list[dict[str, object]]] = {}
    for node in nodes:
        seen_parts: set[tuple[str, int]] = set()
        for index in range(node_counts[node]):
            if node == "alpha":
                bv = "BV-P2-0" if index < 2 else "BV-P2-1"
            elif node == "beta":
                bv = "BV-P2-1" if index < 2 else "BV-P2-2"
            else:
                bv = f"BV-{node}-{index:02d}"
            row = {
                "chunk_id": f"{bv}#P001#{node}-{index:02d}",
                "bv": bv,
                "p_number": 1,
                "start_sec": float(index * 10),
                "end_sec": float(index * 10 + 8),
                "knowledge_topic": [node],
                "align_ratio": 1.0,
                "text_repaired_v2": f"trusted {node} {index}",
                "needs_human": False,
            }
            candidate_rows.append(row)
            part = (bv, 1)
            if part not in seen_parts:
                runtime_rows.setdefault(node, []).append({"bv": bv, "p": 1})
                seen_parts.add(part)
    candidate_path = sources / "trusted_candidates.jsonl"
    candidate_path.write_bytes(
        b"".join(_canonical(row) + b"\n" for row in candidate_rows)
    )
    runtime_path = sources / "runtime.json"
    runtime_path.write_bytes(
        _canonical({"segments_by_node": runtime_rows, "version": "test"}) + b"\n"
    )

    exact_rows = [
        row
        for row in candidate_rows
        if row["knowledge_topic"] in (["alpha"], ["beta"])
    ]
    exact_ids = [str(row["chunk_id"]) for row in exact_rows]
    exact_digest = hashlib.sha256(
        _canonical(sorted(exact_rows, key=lambda row: str(row["chunk_id"])))
    ).hexdigest()
    spec_path = Path(__file__).parents[1] / "experiments/p2_illustrative_analysis_plan.md"
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    claim_boundary = (
        "supply_bound_algorithmic_illustration_not_learning_benefit_or_external_validation"
    )
    input_manifest = {
        "schema_version": "yher.p2.input_manifest.v1",
        "claim_boundary": claim_boundary,
        "simulated": True,
        "illustrative": True,
        "external_validity": False,
        "hash_gate_status": "pass",
        "selector": {
            "budget_seconds": 600,
            "binary_role_slot_saturation": True,
            "physical_source_no_repeat": True,
            "decimal_precision_digits": 60,
        },
        "bootstrap": {
            "resamples": 10000,
            "seed": 2026071505,
            "rng": "numpy.PCG64",
            "cluster_count_per_stratum": 50,
            "fixed_target_strata": ["alpha", "beta"],
            "arms_and_truths_paired_within_target_replicate": True,
        },
        "candidate_subset": {
            "exact_chunk_ids": exact_ids,
            "row_count": 8,
            "physical_source_count": 3,
            "canonical_sha256": exact_digest,
            "canonical_serialization": "canonical sorted JSON array",
            "audit_declared_digest_is_gate": False,
            "audit_declared_unreproduced": "f" * 64,
        },
        "h1_h4_product_margins": {
            "condition": "matched",
            "checkpoint_nominal_budget": 15,
            "posterior_order": ["M", "P", "C", "U"],
            "invalid_belief_read_policy": "never_read_fail_closed",
            "matched_shard_count": 0,
            "matched_shard_sha256": {},
            "journey_count": 0,
        },
        "source_files": {
            "trusted_candidate_jsonl": {
                "path": str(candidate_path),
                "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            },
            "signed_runtime_metadata": {
                "path": str(runtime_path),
                "sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            },
            "h1_h4_raw_manifest": {
                "path": str(raw_manifest),
                "sha256": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
            },
        },
        "spec": {
            "sha256": spec_hash,
            "committed_bytes_sha256": spec_hash,
            "commit": "5" * 40,
            "precedes_outcome_computation": True,
        },
    }
    input_bytes = _canonical(input_manifest) + b"\n"
    (root / "input_manifest.json").write_bytes(input_bytes)

    metric_names = (
        "selected_seconds",
        "selected_segment_count",
        "mismatched_selected_seconds",
        "missed_available_seconds",
        "unused_budget_seconds",
        "unobtainable_truth_slots",
        "unsupported_posterior_mass",
        "structural_failure_node_fraction",
        "missed_diagnostic_structural_failure_seconds",
        "missed_posterior_selection_seconds",
        "missed_budget_constraint_seconds",
    )
    overall: list[dict[str, object]] = []
    for arm_index, arm in enumerate(("oracle", "A", "B", "C")):
        selected = float(120 + arm_index * 10)
        structural = 0.5 if arm == "C" else 0.0
        overall.append(
            {
                "arm": arm,
                "selected_seconds": selected,
                "selected_minutes": selected / 60,
                "selected_segment_count": 2.0,
                "mismatched_selected_seconds": float(arm_index * 5),
                "mismatched_selected_minutes": float(arm_index * 5) / 60,
                "missed_available_seconds": float(arm_index * 3),
                "missed_available_supply_minutes": float(arm_index * 3) / 60,
                "unused_budget_seconds": 600.0 - selected,
                "unused_budget_minutes": (600.0 - selected) / 60,
                "unobtainable_truth_slots": 1.0,
                "unsupported_posterior_mass": 1.0,
                "structural_failure_node_fraction": structural,
                "missed_diagnostic_structural_failure_seconds": (
                    9.0 if arm == "C" else 0.0
                ),
                "missed_posterior_selection_seconds": 0.0,
                "missed_budget_constraint_seconds": float(arm_index * 3),
                "unobtainable_supply_minutes": None,
                "unobtainable_reason": "no_frozen_role_compatible_dose",
                "analytic_integration_terms": 40000,
                "analytic_terms_are_not_sample_size": True,
            }
        )
    overall_by_arm = {str(row["arm"]): row for row in overall}
    bootstrap_overall = [
        {
            "arm": arm,
            "metric": metric,
            "point": overall_by_arm[arm][metric],
            "ci95_low": overall_by_arm[arm][metric],
            "ci95_high": overall_by_arm[arm][metric],
            "defined_resamples": 10000,
        }
        for arm in ("oracle", "A", "B", "C")
        for metric in metric_names
    ]
    contrast_names = (
        "A_minus_oracle",
        "B_minus_oracle",
        "A_minus_B",
        "C_minus_oracle_ITT",
    )
    bootstrap_contrasts = [
        {
            "contrast": contrast,
            "metric": metric,
            "point": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "defined_resamples": 10000,
            "arm_c_structural_failure_annotation": (
                "failed_node_fraction=0.5"
                if contrast == "C_minus_oracle_ITT"
                else None
            ),
        }
        for contrast in contrast_names
        for metric in metric_names
    ]
    truth_cells = [
        {
            "arm": arm,
            "truth_basic": truth_basic,
            "truth_alkane": truth_alkane,
            "selected_seconds": 120.0,
            "selected_segment_count": 2.0,
            "mismatched_selected_seconds": 0.0,
            "missed_available_seconds": 0.0,
            "unused_budget_seconds": 480.0,
            "unobtainable_truth_slots": 1.0,
            "unsupported_posterior_mass": 1.0,
            "structural_failure_node_fraction": 0.5 if arm == "C" else 0.0,
            "missed_diagnostic_structural_failure_seconds": (
                9.0 if arm == "C" else 0.0
            ),
            "missed_posterior_selection_seconds": 0.0,
            "missed_budget_constraint_seconds": 0.0,
        }
        for arm in ("oracle", "A", "B", "C")
        for truth_basic in ("M", "P", "C", "U")
        for truth_alkane in ("M", "P", "C", "U")
    ]
    summary = {
        "schema_version": "yher.p2.summary.v1",
        "claim_boundary": claim_boundary,
        "simulated": True,
        "illustrative": True,
        "external_validity": False,
        "spec_hash": spec_hash,
        "budget_seconds": 600,
        "exact_overlap_targets": ["alpha", "beta"],
        "candidate_row_count": 8,
        "physical_source_count": 3,
        "reporting_unit": (
            "two_fixed_target_strata_each_with_50_programmatic_replicate_clusters"
        ),
        "profile_row_count": 160000,
        "unique_selector_trace_count": 1,
        "selector_trace_deduplication": (
            "profile rows reference exact trace hashes; identical selector inputs share one trace"
        ),
        "scalar_composite_computed": False,
        "overall": overall,
        "bootstrap_overall": bootstrap_overall,
        "bootstrap_contrasts": bootstrap_contrasts,
        "truth_cells": truth_cells,
        "unavailable_minute_field_policy": {
            "value": None,
            "reason": "no_frozen_role_compatible_dose",
        },
    }
    summary_bytes = _canonical(summary) + b"\n"
    (root / "summary.json").write_bytes(summary_bytes)
    p2_publication_files = {
        "figure_data.json": _canonical(
            {
                "schema_version": "yher.p2.figure_data.v1",
                "simulated": True,
                "illustrative": True,
                "external_validity": False,
            }
        )
        + b"\n",
        "p2_supply_bound_illustration.png": b"\x89PNG\r\n\x1a\np2",
        "p2_supply_bound_illustration.svg": b"<svg>p2</svg>\n",
    }
    for filename, payload in p2_publication_files.items():
        (root / filename).write_bytes(payload)
    output_manifest = {
        "schema_version": "yher.p2.output_manifest.v1",
        "claim_boundary": claim_boundary,
        "simulated": True,
        "illustrative": True,
        "external_validity": False,
        "manifest_self_hash_policy": "output_manifest_excluded_from_its_own_artifact_list",
        "profile_row_count": 160000,
        "unique_selector_trace_count": 1,
        "artifacts": [
            {
                "filename": "summary.json",
                "bytes": len(summary_bytes),
                "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            },
            {
                "filename": "input_manifest.json",
                "bytes": len(input_bytes),
                "sha256": hashlib.sha256(input_bytes).hexdigest(),
            },
            *[
                {
                    "filename": filename,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for filename, payload in p2_publication_files.items()
            ],
        ],
    }
    (root / "output_manifest.json").write_bytes(_canonical(output_manifest) + b"\n")
    return root


def _self_hash(payload: dict[str, object], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(_canonical(value)).hexdigest()


PERSONA_PROVIDERS = ("deepseek", "glm", "kimi", "minimax", "doubao", "tongyi")
PERSONA_STATES = (
    "correct_answer",
    "incorrect_answer",
    "abstention",
    "technical_or_schema_failure",
)


def _persona_bootstrap(
    point: float | None,
    *,
    provider_points: dict[str, float | None],
) -> dict[str, object]:
    return {
        "point_estimate": point,
        "ci95": None if point is None else [max(0.0, point - 0.05), min(1.0, point + 0.05)],
        "seed": 2026071503,
        "resamples": 10000,
        "defined_resamples": 0 if point is None else 10000,
        "undefined_resamples": 10000 if point is None else 0,
        "provider_equal_weighting": True,
        "provider_point_estimates": provider_points,
    }


def _persona_controlled_surface() -> dict[str, object]:
    deficit_counts = {
        "correct_answer": 5,
        "incorrect_answer": 40,
        "abstention": 3,
        "technical_or_schema_failure": 2,
    }
    control_counts = {
        "correct_answer": 45,
        "incorrect_answer": 3,
        "abstention": 1,
        "technical_or_schema_failure": 1,
    }
    by_provider: list[dict[str, object]] = []
    for provider in PERSONA_PROVIDERS:
        arms: list[dict[str, object]] = []
        for arm, counts in (("deficit", deficit_counts), ("control", control_counts)):
            answered = counts["correct_answer"] + counts["incorrect_answer"]
            arms.append(
                {
                    "response_arm": arm,
                    "expected_denominator": 50,
                    "counts": dict(counts),
                    "rates": {state: counts[state] / 50 for state in PERSONA_STATES},
                    "conditional_answer_accuracy": counts["correct_answer"] / answered,
                    "conditional_answer_denominator": answered,
                }
            )
        by_provider.append({"provider": provider, "arms": arms})

    all_counts = {
        state: len(PERSONA_PROVIDERS) * (deficit_counts[state] + control_counts[state])
        for state in PERSONA_STATES
    }
    metric_specs = (
        ("conditional_answer_accuracy", "control_minus_deficit", 0.80),
        ("correct_response_yield", "control_minus_deficit", 0.80),
        ("incorrect_response_yield", "deficit_minus_control", 0.74),
        ("abstention_yield", "deficit_minus_control", 0.04),
        ("technical_or_schema_failure_yield", "deficit_minus_control", 0.02),
    )
    effects: list[dict[str, object]] = []
    for metric, orientation, estimate in metric_specs:
        provider_points = {provider: estimate for provider in PERSONA_PROVIDERS}
        effects.append(
            {
                "metric_id": metric,
                "orientation": orientation,
                "estimate": estimate,
                "ci95": [max(0.0, estimate - 0.05), min(1.0, estimate + 0.05)],
                "eligible_providers": list(PERSONA_PROVIDERS),
                "paired_persona_denominators": {
                    provider: 50 for provider in PERSONA_PROVIDERS
                },
                "paired_persona_denominator_range": [50, 50],
                "by_provider": [
                    {
                        "provider": provider,
                        "included_in_aggregate": True,
                        "estimate": estimate,
                        "ci95": [
                            max(0.0, estimate - 0.05),
                            min(1.0, estimate + 0.05),
                        ],
                        "paired_persona_denominator": 50,
                        "bootstrap": _persona_bootstrap(
                            estimate, provider_points={provider: estimate}
                        ),
                    }
                    for provider in PERSONA_PROVIDERS
                ],
                "bootstrap": _persona_bootstrap(
                    estimate, provider_points=provider_points
                ),
            }
        )
    return {
        "eligible_providers": list(PERSONA_PROVIDERS),
        "excluded_providers": [],
        "composition": {
            "states": list(PERSONA_STATES),
            "expected_tasks_per_provider": 100,
            "by_provider": by_provider,
            "aggregate_counts": all_counts,
            "all_provider_counts": all_counts,
        },
        "paired_effects": effects,
    }


def _persona_blind_surface() -> dict[str, object]:
    subjects = [
        f"persona-{index:02d}|{arm}"
        for index in range(50)
        for arm in ("deficit", "control")
    ]
    pairs: list[dict[str, object]] = []
    sorted_providers = sorted(PERSONA_PROVIDERS)
    for left_index, left in enumerate(sorted_providers):
        for right in sorted_providers[left_index + 1 :]:
            pairs.append(
                {
                    "provider_left": left,
                    "provider_right": right,
                    "exact_agreement_numerator": 80,
                    "denominator": 100,
                    "exact_agreement": 0.8,
                    "cohen_kappa": 0.6,
                    "exact_agreement_ci95": [0.75, 0.85],
                    "exact_agreement_bootstrap": _persona_bootstrap(
                        0.8, provider_points={f"{left}__{right}": 0.8}
                    ),
                }
            )
    provider_schema = {
        provider: {
            "expected_primary_blind_tasks": 100,
            "invalid_schema_count": 0,
            "invalid_schema_fraction": 0.0,
            "strictly_above_half": False,
            "invalid_schema_strictly_above_half": False,
            "excluded_from_blind_aggregate": False,
            "complete_cluster_count": 50,
            "exclusion_reasons": [],
        }
        for provider in PERSONA_PROVIDERS
    }
    stability = [
        {
            "provider": provider,
            "excluded_from_blind_aggregate": False,
            "status": "estimated",
            "expected_pairs": 20,
            "answer_agreement_numerator": 18,
            "answer_agreement_denominator": 20,
            "answer_agreement": 0.9,
            "answer_bootstrap": _persona_bootstrap(
                0.9, provider_points={provider: 0.9}
            ),
            "nc_nc_agreement_count": 0,
            "canonical_complete_pair_numerator": 16,
            "canonical_complete_pair_denominator": 20,
            "canonical_complete_pair_stability": 0.8,
            "canonical_complete_pair_bootstrap": _persona_bootstrap(
                0.8, provider_points={provider: 0.8}
            ),
            "canonical_itt_yield": 0.8,
        }
        for provider in PERSONA_PROVIDERS
    ]
    provider_points = {provider: 0.9 for provider in PERSONA_PROVIDERS}
    canonical_points = {provider: 0.8 for provider in PERSONA_PROVIDERS}
    return {
        "primary_terminal_definition": "frozen_final_blind_item",
        "terminal_subject_count": 100,
        "terminal_categories": ["A", "B", "NC"],
        "provider_schema": provider_schema,
        "eligible_providers": list(PERSONA_PROVIDERS),
        "excluded_providers": [],
        "agreement": {
            "subjects": subjects,
            "providers": sorted_providers,
            "categories": ["A", "B", "NC"],
            "nc_retained": True,
            "pairs": pairs,
            "status": "estimated",
        },
        "multi_provider_descriptive": {
            "rectangular_providers": list(PERSONA_PROVIDERS),
            "subjects": 100,
            "unanimous_numerator": 70,
            "unanimous_fraction": 0.7,
            "status": "estimated",
        },
        "technical_or_schema_failure_rate": {
            "estimate": 0.02,
            "ci95": [0.01, 0.03],
            "bootstrap": _persona_bootstrap(
                0.02,
                provider_points={provider: 0.02 for provider in PERSONA_PROVIDERS},
            ),
        },
        "stability": stability,
        "stability_provider_equal_aggregate": {
            "eligible_providers": list(PERSONA_PROVIDERS),
            "answer": _persona_bootstrap(0.9, provider_points=provider_points),
            "canonical_complete_pair": _persona_bootstrap(
                0.8, provider_points=canonical_points
            ),
        },
    }


def _write_persona_judge_fixture(
    root: Path, *, judge_profile: str
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    from experiments.llm_sim_v2 import analyze, judge_execution

    selected_count = 0 if judge_profile == "zero_cases" else 2
    candidates: list[dict[str, object]] = []
    for index in range(selected_count):
        question = {
            "kind": "mcq",
            "stem_blocks": [],
            "stem_text": f"Public judge question {index}",
            "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
            "difficulty": 0.5,
            "nodes": ["private-node"],
            "source_label": "private-source",
        }
        candidates.append(
            {
                "candidate_identity": f"provider|task-{index:03d}",
                "stratum": "agreement",
                "public_question": question,
                "model_output": {
                    "simulated": True,
                    "answer": "A",
                    "rationale": f"Candidate rationale {index}",
                    "abstain": False,
                },
                "persona": {
                    "persona_id": f"private-persona-{index:03d}",
                    "target_node": "private-node",
                },
                "item": {
                    "item_id": f"private-item-{index:03d}",
                    "public_question": question,
                    "options": question["options"],
                },
            }
        )
    case_manifest = analyze.build_judge_case_manifest(
        candidates, frozen_leakage_lexicon=()
    )
    case_ids = [str(row["case_id"]) for row in case_manifest["cases"]]
    run_root = root.parent / f".{root.name}-judge-run-source"
    judges = {
        "both_complete": ("gpt", "claude"),
        "gpt_only": ("gpt",),
        "zero_cases": (),
    }[judge_profile]
    result_manifests: dict[str, dict[str, object]] = {}
    artifact_roots: dict[str, str] = {}
    for judge in judges:
        model = f"fixture-{judge}-exact"
        rows = [
            {
                "case_id": case_id,
                "output": {
                    "label": "consistent",
                    "error_category": "none",
                    "rationale": f"{judge} rationale for {case_id}",
                    "simulated": True,
                },
            }
            for case_id in case_ids
        ]
        responses = [
            {
                "schema_version": "yher.llm_sim_v2.judge_transport_response.v2",
                "simulated": True,
                "transport_reported_models": [model],
                "transport_reported_model_source": "fixture_response",
                "transport_request_id": f"fixture-{judge}-request-{offset // 10:03d}",
                "results": rows[offset : offset + 10],
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "billing": {
                    "known_cost_yuan": 0.125,
                    "unknown_cost_reserve_yuan": 0,
                },
                "tool_calls": [],
            }
            for offset in range(0, len(rows), 10)
        ]
        receipt_path = judge_execution.execute_judge_pass(
            case_manifest=case_manifest,
            output_root=run_root,
            judge_family=judge,
            exact_model=model,
            transport=judge_execution.FixtureJudgeTransport(responses),
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest: dict[str, object] = {
            "schema_version": "yher.llm_sim_v2.judge_result_manifest.v2",
            "simulated": True,
            "run_id": "llm-personas-v2-dual",
            "judge": judge,
            "case_manifest_sha256": case_manifest["case_manifest_sha256"],
            "execution_receipt_path": receipt_path.relative_to(run_root).as_posix(),
            "execution_receipt": receipt,
            "results": rows,
        }
        manifest["judge_result_manifest_sha256"] = (
            judge_execution.canonical_sha256(manifest)
        )
        result_path = run_root / f"{judge}.json"
        result_path.write_bytes(judge_execution.canonical_json_bytes(manifest) + b"\n")
        result_manifests[judge] = manifest
        artifact_roots[judge] = str(receipt_path.parent.resolve())

    if judge_profile == "gpt_only":
        judge_execution.record_judge_family_disposition(
            case_manifest=case_manifest,
            output_root=run_root,
            judge_family="claude",
            status="unavailable",
            reason_code="production_cli_unavailable",
        )
    elif judge_profile == "zero_cases":
        for judge in ("claude", "gpt"):
            judge_execution.record_judge_family_disposition(
                case_manifest=case_manifest,
                output_root=run_root,
                judge_family=judge,
                status="not_applicable_zero_cases",
                reason_code="selected_case_count_zero",
            )

    run_receipt_path = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=case_manifest,
        output_root=run_root,
        allow_fixture=True,
    )
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    result_slots = {
        judge: result_manifests.get(judge) for judge in ("claude", "gpt")
    }
    judge_analysis = analyze.ingest_judge_results(
        case_manifest,
        result_slots,
        judge_artifact_roots=artifact_roots,
        judge_run_evidence={"receipt": run_receipt},
        allow_fixture=True,
    )

    source_files = sorted(
        (path for path in run_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(run_root).as_posix(),
    )
    input_files = [
        {
            "path": f"judge-results/{path.relative_to(run_root).as_posix()}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in source_files
    ]
    input_manifest: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.analysis_input_artifact_manifest.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "analysis_population": "main",
        "files": input_files,
        "input_file_count": len(input_files),
        "record_file_count": 0,
        "input_file_set_sha256": hashlib.sha256(_canonical(input_files)).hexdigest(),
    }
    sources = {
        f"judge-results/{path.relative_to(run_root).as_posix()}": str(path.resolve())
        for path in source_files
    }
    snapshot = analyze._stage_judge_execution_snapshots(
        staging=root,
        judge_artifact_sources=sources,
        input_artifact_manifest=input_manifest,
        allow_fixture=True,
    )
    return (
        case_manifest,
        result_manifests,
        judge_analysis,
        input_manifest,
        snapshot,
    )


def _write_persona_v2_bundle(
    root: Path,
    *,
    pilot_overlap: bool = False,
    judge_profile: str = "both_complete",
) -> Path:
    if judge_profile not in {"both_complete", "gpt_only", "zero_cases"}:
        raise ValueError("unsupported judge fixture profile")
    root.mkdir()
    analysis_root = root / "analysis"
    evidence_root = root / "evidence"
    analysis_root.mkdir()
    evidence_root.mkdir()
    providers = list(PERSONA_PROVIDERS)
    judges = {
        "both_complete": ("claude", "gpt"),
        "gpt_only": ("gpt",),
        "zero_cases": (),
    }[judge_profile]
    main_ids = [f"main-task-{index:03d}" for index in range(220)]
    pilot_ids = [main_ids[0]] if pilot_overlap else ["pilot-task-0"]
    runtime: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.runtime_task_manifest.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "freeze_commit": "6" * 40,
        "freeze_manifest_sha256": "7" * 64,
        "runtime_commit": "8" * 40,
        "frozen_at_utc": "2026-07-15T00:00:00Z",
        "prompt_revision": 0,
        "prompt_contract_sha256": "9" * 64,
        "prompt_ledger_sha256": "a" * 64,
        "runtime_files": [],
        "runtime_file_set_sha256": hashlib.sha256(_canonical([])).hexdigest(),
        "phases": {
            "pilot": {
                "task_count": len(pilot_ids),
                "task_ids": pilot_ids,
                "task_set_sha256": hashlib.sha256(_canonical(pilot_ids)).hexdigest(),
                "providers": providers[:2],
            },
            "main": {
                "task_count": len(main_ids),
                "task_ids": main_ids,
                "task_set_sha256": hashlib.sha256(_canonical(main_ids)).hexdigest(),
                "providers": providers,
            },
        },
    }
    runtime["runtime_task_manifest_sha256"] = _self_hash(
        runtime, "runtime_task_manifest_sha256"
    )
    runtime_path = evidence_root / "runtime_task_manifest.json"
    runtime_path.write_bytes(_canonical(runtime) + b"\n")

    mapping_rows = [
        {
            "item_id": f"item-{index:03d}",
            "failure_id": f"failure-{index:03d}",
            "target_option": "B" if index < 6 else None,
            "status": "mapped" if index < 6 else "excluded_ambiguous",
            "reviewer_provenance": {"kind": "cross_model_consensus"},
            **(
                {}
                if index < 6
                else {"ambiguity_reason": "no_unique_target_option"}
            ),
        }
        for index in range(100)
    ]
    mapping_sha = hashlib.sha256(_canonical(mapping_rows)).hexdigest()
    mapped_targets = [
        {
            "item_id": row["item_id"],
            "failure_id": row["failure_id"],
            "target_option": row["target_option"],
        }
        for row in mapping_rows
        if row["status"] == "mapped"
    ]
    target_hash = hashlib.sha256(_canonical(mapped_targets)).hexdigest()
    mapping = {
        "schema_version": "yher.llm_sim_v2.target_option_map.v1",
        "frozen": True,
        "observation_started": False,
        "candidate_frame_sha256": "b" * 64,
        "crosscheck_provenance": {"kind": "cross_model_consensus"},
        "rows": mapping_rows,
        "mapping_sha256": mapping_sha,
        "target_set_hash": target_hash,
        "mapped_fraction": 0.06,
        "confirmatory_target_misconception_hit_rate": False,
        "consensus": {
            "draft_sha256": "c" * 64,
            "crosscheck_sha256": "d" * 64,
            "mapped_rows": 6,
            "excluded_ambiguous_rows": 94,
        },
    }
    mapping_path = evidence_root / "target_option_mapping.json"
    mapping_path.write_bytes(_canonical(mapping) + b"\n")

    phase: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.phase_provenance.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "phase": "main",
        "analysis_population": "main",
        "collection_mode": "formal",
        "development_only": False,
        "partial": False,
        "formal_analysis_eligible": True,
        "modality_condition": "text_only",
        "selected_providers": providers,
        "frozen_providers": providers,
        "task_limit": None,
        "target": {
            "target_set_hash": target_hash,
            "mapping_sha256": mapping_sha,
        },
        "runtime": {
            "runtime_task_manifest_sha256": runtime[
                "runtime_task_manifest_sha256"
            ],
            "execution_commit": runtime["runtime_commit"],
            "runtime_file_set_sha256": runtime["runtime_file_set_sha256"],
        },
        "task_roster": {
            "expected_task_count": len(main_ids),
            "expected_task_ids": main_ids,
            "task_set_sha256": hashlib.sha256(_canonical(main_ids)).hexdigest(),
            "frozen_task_count": len(main_ids),
            "frozen_task_set_sha256": hashlib.sha256(_canonical(main_ids)).hexdigest(),
        },
    }
    phase["phase_provenance_sha256"] = _self_hash(
        phase, "phase_provenance_sha256"
    )
    phase_path = evidence_root / "phase_provenance.json"
    phase_path.write_bytes(_canonical(phase) + b"\n")

    lifecycle = [
        {
            "provider": provider,
            "provider_lifecycle": "complete",
            "recomputed_provider_lifecycle": "complete",
            "expected_count": len(main_ids),
            "present_count": len(main_ids),
            "missing_count": 0,
            "missing_task_ids": [],
            "status_counts": {"complete": len(main_ids)},
            "known_cost_yuan": 1.0,
            "unknown_cost_reserve_yuan": 0.0,
            "accounted_cost_yuan": 1.0,
            "controlled_complete_cluster_count": 50,
            "controlled_eligible": True,
            "controlled_exclusion_reasons": [],
            "blind_complete_cluster_count": 50,
            "blind_eligible": True,
            "blind_exclusion_reasons": [],
        }
        for provider in providers
    ]
    (
        judge_case_manifest,
        judge_result_manifests,
        judge_analysis,
        analysis_input_manifest,
        judge_run_snapshot,
    ) = _write_persona_judge_fixture(root, judge_profile=judge_profile)
    judge_result_bytes = {
        judge: _canonical(manifest) + b"\n"
        for judge, manifest in judge_result_manifests.items()
    }
    result = {
        "schema_version": "yher.llm_sim_v2.analysis_results.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "analysis_population": "main",
        "modality_condition": "text_only",
        "independent_cluster_count": 50,
        "independent_cluster_unit": "persona_id",
        "repeated_measure_factors": ["provider", "response_arm", "condition", "item"],
        "claim_boundary": (
            "independent simulated text-only response-channel stress test; "
            "not human participants, learner trajectories, or educational efficacy"
        ),
        "input_proof": {
            "schema_version": "yher.llm_sim_v2.analysis_input_proof.v1",
            "ok": True,
            "analysis_population": "main",
            "persona_cluster_count": 50,
            "providers": providers,
            "expected_task_count": len(main_ids),
            "runtime_task_manifest_sha256": runtime[
                "runtime_task_manifest_sha256"
            ],
            "phase_provenance_sha256": phase["phase_provenance_sha256"],
        },
        "expected_denominator": {
            "source": "committed_runtime_task_manifest",
            "filesystem_glob_defines_denominator": False,
            "provider_count": len(providers),
            "tasks_per_provider": len(main_ids),
            "provider_task_cells": len(providers) * len(main_ids),
        },
        "input_artifact_binding": {
            "input_file_count": analysis_input_manifest["input_file_count"],
            "record_file_count": analysis_input_manifest["record_file_count"],
            "input_file_set_sha256": analysis_input_manifest[
                "input_file_set_sha256"
            ],
            "input_artifact_manifest_sha256": hashlib.sha256(
                _canonical(analysis_input_manifest)
            ).hexdigest(),
        },
        "bootstrap_contract": {
            "cluster_unit": "persona_id",
            "provider_equal_weighting": True,
            "resamples": 10000,
            "seed": 2026071503,
            "confidence_interval": "two-sided percentile 95%",
            "undefined_resamples_retained": True,
        },
        "provider_lifecycle": lifecycle,
        "sparse_mapping_descriptive": {
            "status": "sparse_descriptive_only",
            "confirmatory": False,
            "mapped_mapping_rows": 6,
            "excluded_ambiguous_mapping_rows": 94,
            "total_mapping_rows": 100,
            "mapped_fraction": 0.06,
            "mapping_sha256": mapping_sha,
            "target_set_hash": target_hash,
            "by_provider_and_arm": [],
        },
        "controlled": _persona_controlled_surface(),
        "blind": _persona_blind_surface(),
        "judge_adjudication": {
            "case_manifest": judge_case_manifest,
            "analysis": judge_analysis,
            "run_evidence_binding": {
                "schema_version": (
                    "yher.llm_sim_v2.formal_judge_run_evidence_binding.v1"
                ),
                "judge_run_evidence_receipt_sha256": judge_run_snapshot[
                    "source_judge_run_evidence_receipt_sha256"
                ],
                "committed_anchor_sha256": "e" * 64,
                "family_slots": judge_run_snapshot["family_slots"],
            },
            "result_manifests": {
                judge: judge_result_manifests.get(judge)
                for judge in ("claude", "gpt")
            },
        },
        "outputs": {
            "machine_json": True,
            "machine_csv_tables": 8,
            "figure_data_machine_readable": True,
            "publication_figures": 3,
            "publication_formats": ["png_300_dpi", "svg"],
            "judge_case_export": True,
            "judge_shared_input_sha256": "1" * 64,
        },
    }
    result_path = analysis_root / "analysis_results.json"
    result_path.write_bytes(_canonical(result) + b"\n")
    input_artifact_path = analysis_root / "input_artifact_manifest.json"
    input_artifact_path.write_bytes(_canonical(analysis_input_manifest) + b"\n")
    judge_result_paths: dict[str, Path] = {}
    for judge, payload in judge_result_bytes.items():
        path = root / f"judge-snapshots/run/{judge}.json"
        assert path.read_bytes() == payload
        judge_result_paths[judge] = path
    snapshot_manifest_path = root / "judge-snapshots/snapshot_manifest.json"
    assert json.loads(snapshot_manifest_path.read_text(encoding="utf-8")) == (
        judge_run_snapshot
    )
    snapshot_publication_files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(
            (path for path in (root / "judge-snapshots").rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    }
    publication_files = {
        "input_artifact_manifest.json": input_artifact_path.read_bytes(),
        "provider_lifecycle.csv": b"provider,provider_lifecycle\ndeepseek,complete\n",
        "controlled_composition.csv": b"provider,response_arm,state,count\ndeepseek,deficit,incorrect_answer,40\n",
        "controlled_paired_effects.csv": b"metric_id,estimate\nconditional_answer_accuracy,0.8\n",
        "blind_agreement.csv": b"provider_left,provider_right,exact_agreement\ndeepseek,glm,0.8\n",
        "blind_stability.csv": b"provider,answer_agreement\ndeepseek,0.9\n",
        "figure_data/controlled_composition.csv": b"provider,response_arm,state,count\ndeepseek,deficit,incorrect_answer,40\n",
        "figure_data/blind_agreement.csv": b"provider_left,provider_right,exact_agreement\ndeepseek,glm,0.8\n",
        "figure_data/blind_stability.csv": b"provider,answer_agreement\ndeepseek,0.9\n",
        "figures/controlled_composition.png": b"\x89PNG\r\n\x1a\ncontrolled",
        "figures/controlled_composition.svg": b"<svg>controlled</svg>\n",
        "figures/blind_terminal_agreement.png": b"\x89PNG\r\n\x1a\nagreement",
        "figures/blind_terminal_agreement.svg": b"<svg>agreement</svg>\n",
        "figures/blind_output_stability.png": b"\x89PNG\r\n\x1a\nstability",
        "figures/blind_output_stability.svg": b"<svg>stability</svg>\n",
        "judge/case_manifest.json": _canonical(judge_case_manifest) + b"\n",
        "judge/judge_analysis.json": _canonical(judge_analysis) + b"\n",
        **snapshot_publication_files,
    }
    for relative, payload in publication_files.items():
        path = analysis_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    artifact_paths = [
        result_path,
        *(analysis_root / name for name in publication_files),
    ]
    artifacts = [
        {
            "path": path.relative_to(analysis_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": len(path.read_bytes()),
        }
        for path in artifact_paths
    ]
    analysis_manifest = {
        "schema_version": "yher.llm_sim_v2.analysis_artifact_manifest.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "analysis_population": "main",
        "target_set_hash": target_hash,
        "runtime_task_manifest_sha256": runtime[
            "runtime_task_manifest_sha256"
        ],
        "phase_provenance_sha256": phase["phase_provenance_sha256"],
        "artifacts": artifacts,
        "artifact_set_sha256": hashlib.sha256(_canonical(artifacts)).hexdigest(),
    }
    analysis_manifest_path = analysis_root / "artifact_manifest.json"
    analysis_manifest_path.write_bytes(_canonical(analysis_manifest) + b"\n")

    bundle_files = {
        "analysis_results": result_path,
        "analysis_artifact_manifest": analysis_manifest_path,
        "analysis_input_artifact_manifest": input_artifact_path,
        "phase_provenance": phase_path,
        "runtime_task_manifest": runtime_path,
        "mapping_manifest": mapping_path,
        **{
            f"{judge}_judge_result_manifest": judge_result_paths[judge]
            for judge in judges
        },
        "judge_run_execution_snapshot_manifest": snapshot_manifest_path,
    }
    binding_manifest = {
        "schema_version": "yher.journal_binder.persona_v2_bundle.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "analysis_population": "main",
        "files": {
            role: {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for role, path in bundle_files.items()
        },
    }
    (root / "binding_manifest.json").write_bytes(_canonical(binding_manifest) + b"\n")
    return root


def _rewrite_persona_result_bundle(
    bundle: Path,
    mutate: object,
) -> None:
    analysis_root = bundle / "analysis"
    result_path = analysis_root / "analysis_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(result)
    result_path.write_bytes(_canonical(result) + b"\n")

    judge_files = {
        "judge/case_manifest.json": result["judge_adjudication"]["case_manifest"],
        "judge/judge_analysis.json": result["judge_adjudication"]["analysis"],
    }
    for relative, payload in judge_files.items():
        (analysis_root / relative).write_bytes(_canonical(payload) + b"\n")

    artifact_path = analysis_root / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result_entry = next(
        row for row in artifact["artifacts"] if row["path"] == "analysis_results.json"
    )
    result_entry["sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    result_entry["size"] = len(result_path.read_bytes())
    for relative in judge_files:
        path = analysis_root / relative
        row = next(item for item in artifact["artifacts"] if item["path"] == relative)
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row["size"] = len(path.read_bytes())
    artifact["artifact_set_sha256"] = hashlib.sha256(
        _canonical(artifact["artifacts"])
    ).hexdigest()
    artifact_path.write_bytes(_canonical(artifact) + b"\n")

    binding_path = bundle / "binding_manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["files"]["analysis_results"]["sha256"] = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    binding["files"]["analysis_artifact_manifest"]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    binding_path.write_bytes(_canonical(binding) + b"\n")


def _rewrite_persona_input_manifest_bundle(
    bundle: Path,
    mutate: object,
) -> None:
    analysis_root = bundle / "analysis"
    input_path = analysis_root / "input_artifact_manifest.json"
    input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(input_manifest)
    input_manifest["input_file_count"] = len(input_manifest["files"])
    input_manifest["input_file_set_sha256"] = hashlib.sha256(
        _canonical(input_manifest["files"])
    ).hexdigest()
    input_path.write_bytes(_canonical(input_manifest) + b"\n")

    def bind_input(result: dict[str, object]) -> None:
        input_binding = result["input_artifact_binding"]
        assert isinstance(input_binding, dict)
        input_binding.update(
            {
                "input_file_count": input_manifest["input_file_count"],
                "record_file_count": input_manifest["record_file_count"],
                "input_file_set_sha256": input_manifest["input_file_set_sha256"],
                "input_artifact_manifest_sha256": hashlib.sha256(
                    _canonical(input_manifest)
                ).hexdigest(),
            }
        )

    _rewrite_persona_result_bundle(bundle, bind_input)
    artifact_path = analysis_root / "artifact_manifest.json"
    artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
    input_row = next(
        row
        for row in artifact_manifest["artifacts"]
        if row["path"] == "input_artifact_manifest.json"
    )
    input_row["sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    input_row["size"] = input_path.stat().st_size
    artifact_manifest["artifact_set_sha256"] = hashlib.sha256(
        _canonical(artifact_manifest["artifacts"])
    ).hexdigest()
    artifact_path.write_bytes(_canonical(artifact_manifest) + b"\n")
    binding_path = bundle / "binding_manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["files"]["analysis_input_artifact_manifest"]["sha256"] = (
        hashlib.sha256(input_path.read_bytes()).hexdigest()
    )
    binding["files"]["analysis_artifact_manifest"]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    binding_path.write_bytes(_canonical(binding) + b"\n")


def test_raw_manifest_must_be_an_explicit_override(tmp_path: Path) -> None:
    from experiments import journal_binder

    with pytest.raises(journal_binder.BinderError, match="raw_manifest_path"):
        journal_binder.build_binder(
            raw_manifest_path=None,
            registry_path=tmp_path / "registry.json",
        )


def test_complete_mode_requires_results_and_paper_artifact_manifest(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    registry = _write_registry(tmp_path / "registry.json")

    with pytest.raises(journal_binder.BinderError, match="results artifact is required"):
        journal_binder.build_binder(
            raw_manifest_path=raw,
            registry_path=registry,
            results_path=None,
            p2_dir=None,
            require_complete=True,
        )


def test_complete_mode_rejects_nonfrozen_raw_manifest(tmp_path: Path) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    payload = json.loads(raw.read_text(encoding="utf-8"))
    payload["status"] = "partial"
    raw.write_bytes(_canonical(payload) + b"\n")
    registry = _write_registry(tmp_path / "registry.json")

    with pytest.raises(journal_binder.BinderError, match="confirmatory-v1.*complete"):
        journal_binder.build_binder(
            raw_manifest_path=raw,
            registry_path=registry,
            results_path=None,
            p2_dir=None,
            require_complete=True,
        )


def test_raw_manifest_requires_declared_sha256_for_every_shard(tmp_path: Path) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    payload = json.loads(raw.read_text(encoding="utf-8"))
    del payload["shards"][0]["sha256"]
    raw.write_bytes(_canonical(payload) + b"\n")
    registry = _write_registry(tmp_path / "registry.json")

    with pytest.raises(journal_binder.BinderError, match="declared SHA-256"):
        journal_binder.build_binder(
            raw_manifest_path=raw,
            registry_path=registry,
            p2_dir=None,
            require_complete=False,
        )


def test_complete_binding_requires_paper_artifact_manifest(tmp_path: Path) -> None:
    from experiments import journal_binder

    bundle = _write_complete_h1_h4_bundle(tmp_path)
    bundle["artifact_manifest"].unlink()

    with pytest.raises(journal_binder.BinderError, match="paper artifact manifest"):
        journal_binder.build_binder(
            raw_manifest_path=bundle["raw"],
            registry_path=bundle["registry"],
            results_path=bundle["results"],
            p2_dir=None,
            require_complete=True,
        )


def test_complete_binding_rejects_unrelated_results_source_manifest(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_complete_h1_h4_bundle(tmp_path)
    results = json.loads(bundle["results"].read_text(encoding="utf-8"))
    results["source_provenance"]["source_manifest_sha256"] = "f" * 64
    bundle["results"].write_bytes(_canonical(results) + b"\n")
    artifact = json.loads(
        bundle["artifact_manifest"].read_text(encoding="utf-8")
    )
    artifact["files"]["results.json"] = hashlib.sha256(
        bundle["results"].read_bytes()
    ).hexdigest()
    artifact["source_provenance"]["source_manifest_sha256"] = "f" * 64
    bundle["artifact_manifest"].write_bytes(_canonical(artifact) + b"\n")

    with pytest.raises(journal_binder.BinderError, match="source manifest"):
        journal_binder.build_binder(
            raw_manifest_path=bundle["raw"],
            registry_path=bundle["registry"],
            results_path=bundle["results"],
            p2_dir=None,
            require_complete=True,
        )


def test_complete_binding_rejects_inverted_frozen_decisions(tmp_path: Path) -> None:
    from experiments import journal_binder

    bundle = _write_complete_h1_h4_bundle(tmp_path)
    results = json.loads(bundle["results"].read_text(encoding="utf-8"))
    results["decisions"].update(
        {
            "H1": "supported",
            "H2": "supported",
            "H3": "not_supported",
            "H4": "not_supported",
        }
    )
    bundle["results"].write_bytes(_canonical(results) + b"\n")
    artifact = json.loads(
        bundle["artifact_manifest"].read_text(encoding="utf-8")
    )
    artifact["files"]["results.json"] = hashlib.sha256(
        bundle["results"].read_bytes()
    ).hexdigest()
    bundle["artifact_manifest"].write_bytes(_canonical(artifact) + b"\n")

    with pytest.raises(journal_binder.BinderError, match="frozen H1-H4 decisions"):
        journal_binder.build_binder(
            raw_manifest_path=bundle["raw"],
            registry_path=bundle["registry"],
            results_path=bundle["results"],
            p2_dir=None,
            require_complete=True,
        )


def test_frozen_h4_hypothesis_evidence_preserves_analysis_artifact_order() -> None:
    from experiments import journal_binder

    assert journal_binder.EXPECTED_HYPOTHESIS_EVIDENCE["H4"] == [
        "h4.misspecified.b15.rescue_A_minus_B",
        "h4.misspecified.b9.harm_C_minus_A",
        "h4.degradation.h1_rescue.matched_minus_misspecified",
        "h4.degradation.h2_harm.matched_minus_misspecified",
        "h4.misspecified.b9.no_harm_A_minus_B",
        "h4.degradation.h2_no_harm.matched_minus_misspecified",
        "h1.no_repeat.misspecified.b15.rescue_A_minus_B",
        "h2.no_repeat.misspecified.b9.harm_C_minus_A",
        "h2.no_repeat.misspecified.b9.no_harm_A_minus_B",
        "misspecification.item_type.mcq.generator_minus_production",
        "misspecification.item_type.numeric.generator_minus_production",
    ]


def test_complete_binding_rejects_registry_not_bound_by_paper_manifest(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_complete_h1_h4_bundle(tmp_path)
    artifact = json.loads(
        bundle["artifact_manifest"].read_text(encoding="utf-8")
    )
    artifact["files"]["metric_registry.json"] = "0" * 64
    bundle["artifact_manifest"].write_bytes(_canonical(artifact) + b"\n")

    with pytest.raises(journal_binder.BinderError, match="metric registry.*paper artifact"):
        journal_binder.build_binder(
            raw_manifest_path=bundle["raw"],
            registry_path=bundle["registry"],
            results_path=bundle["results"],
            p2_dir=None,
            require_complete=True,
        )


def test_production_mode_rejects_an_incomplete_reportable_registry(tmp_path: Path) -> None:
    from experiments import journal_binder

    bundle = _write_complete_h1_h4_bundle(tmp_path)
    rows = json.loads(bundle["registry"].read_text(encoding="utf-8"))[:-1]
    _write_registry(bundle["registry"], rows=rows)
    registry_ids = [row["metric_id"] for row in rows]
    results = json.loads(bundle["results"].read_text(encoding="utf-8"))
    results["registry_metric_ids"] = registry_ids
    bundle["results"].write_bytes(_canonical(results) + b"\n")
    artifact = json.loads(
        bundle["artifact_manifest"].read_text(encoding="utf-8")
    )
    artifact["registry_metric_ids"] = registry_ids
    artifact["files"]["metric_registry.json"] = hashlib.sha256(
        bundle["registry"].read_bytes()
    ).hexdigest()
    artifact["files"]["results.json"] = hashlib.sha256(
        bundle["results"].read_bytes()
    ).hexdigest()
    bundle["artifact_manifest"].write_bytes(_canonical(artifact) + b"\n")

    with pytest.raises(journal_binder.BinderError, match="registry is incomplete"):
        journal_binder.build_binder(
            raw_manifest_path=bundle["raw"],
            registry_path=bundle["registry"],
            results_path=bundle["results"],
            p2_dir=None,
        )


def test_binder_binds_support_identity_and_rejects_cross_support_pairs(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    registry = _write_registry(tmp_path / "registry.json")

    bound = journal_binder.build_binder(
        raw_manifest_path=raw,
        registry_path=registry,
        p2_dir=None,
        require_complete=False,
    )

    metric = bound["metrics"]["h1.primary.matched.b15.rescue_A_minus_B"]
    assert metric["target_set_hash"] == bound["supports"]["eligible"]["target_set_hash"]
    assert metric["target_roster"] == ["alpha", "beta"]
    assert metric["filter_predicate"]
    assert metric["weighting"] == PAIR_WEIGHT
    assert metric["numerator"] == 12
    assert metric["denominator"] == 100
    assert metric["source_hashes"]["raw_manifest_sha256"] == hashlib.sha256(
        raw.read_bytes()
    ).hexdigest()
    assert metric["source_hashes"]["config_sha256"] == "config-hash"
    assert metric["source_hashes"]["analysis_plan_sha256"] == "plan-hash"

    pair = bound["same_support_pairs"]["full_27_terminal_vs_correct_convergence"]
    assert pair["target_set_hash"] == bound["supports"]["full"]["target_set_hash"]
    assert pair["comparison_status"] == "bound"
    assert pair["post_hoc"] is True
    assert bound["metrics"][pair["left_metric_id"]]["post_hoc"] is True
    with pytest.raises(journal_binder.BinderError, match="support"):
        journal_binder.assert_comparable(
            bound["metrics"]["outcome_by_view.matched.b15.arm_A.truth_P.terminal_accuracy"],
            metric,
        )


def test_persona_v2_slot_is_pending_and_rejects_invented_values() -> None:
    from experiments import journal_binder

    slot = journal_binder.persona_v2_slot()
    assert slot["status"] == "pending_formal_w3_artifacts"
    assert slot["value"] is None
    with pytest.raises(journal_binder.BinderError, match="Persona-v2"):
        journal_binder.bind_persona_v2_value(slot, 0.5)


def test_persona_v2_formal_w3_bundle_has_a_verified_success_path(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    assert hasattr(journal_binder, "bind_persona_v2_artifacts"), (
        "formal W3 evidence needs an artifact-level binder"
    )
    bundle = _write_persona_v2_bundle(tmp_path / "persona")

    bound = journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)

    assert bound["status"] == "bound_formal_w3"
    assert bound["run_id"] == "llm-personas-v2-dual"
    assert bound["analysis_population"] == "main"
    assert bound["persona_cluster_count"] == 50
    assert bound["independent_unit"] == "persona_id"
    assert bound["provider_role"] == "repeated_measurement"
    assert bound["modality_condition"] == "text_only"
    assert bound["pilot_exclusion"]["task_rosters_disjoint"] is True
    assert bound["pilot_exclusion"]["pilot_task_count"] == 1
    assert bound["pilot_exclusion"]["main_task_count"] == 220
    assert len(bound["provider_lifecycle"]) == 6
    assert bound["mapping"]["mapped_mapping_rows"] == 6
    assert bound["mapping"]["total_mapping_rows"] == 100
    assert len(bound["source_hashes"]["analysis_results_sha256"]) == 64
    assert len(bound["source_hashes"]["analysis_artifact_set_sha256"]) == 64
    assert bound["controlled"]["paired_effects"][0][
        "paired_persona_denominator_range"
    ] == [50, 50]
    assert bound["blind"]["agreement"]["pairs"][0]["denominator"] == 100
    assert bound["blind"]["stability"][0]["answer_agreement_denominator"] == 20
    assert bound["judge_adjudication"]["analysis"]["status"] == "complete"
    assert bound["judge_adjudication"]["analysis"]["schema_version"] == (
        "yher.llm_sim_v2.judge_analysis.v2"
    )
    assert set(
        bound["judge_adjudication"]["analysis"]["execution_receipt_sha256"]
    ) == {"claude", "gpt"}
    assert set(bound["publication_assets"]["main_persona_composite_sources"]) == {
        "controlled_composition",
        "blind_terminal_agreement",
        "blind_output_stability",
    }


def test_persona_v2_accepts_exact_gpt_only_claude_missing_profile(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(
        tmp_path / "persona",
        judge_profile="gpt_only",
    )

    bound = journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)

    analysis = bound["judge_adjudication"]["analysis"]
    assert analysis["status"] == "partial_missing_judge"
    assert analysis["expected_judges"] == ["claude", "gpt"]
    assert analysis["available_judges"] == ["gpt"]
    assert analysis["missing_judges"] == ["claude"]
    assert analysis["pairwise_label_agreement"] is None
    assert analysis["pairwise_error_category_agreement"] is None
    assert set(analysis["result_manifest_sha256"]) == {"gpt"}
    snapshot_paths = set(
        bound["publication_assets"]["judge_execution_snapshots"]
    )
    assert "judge-snapshots/run/family_dispositions/claude.json" in snapshot_paths
    assert "judge-snapshots/run/gpt.json" in snapshot_paths
    assert any(
        path.startswith("judge-snapshots/run/executions/gpt/")
        and path.endswith("/execution_receipt.json")
        for path in snapshot_paths
    )
    assert not any(path == "judge-snapshots/run/claude.json" for path in snapshot_paths)
    assert not any(
        role.startswith("claude_") for role in bound["source_artifacts"]
    )


def test_persona_v2_rejects_renamed_identical_judge_result_role(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(
        tmp_path / "persona", judge_profile="gpt_only"
    )
    canonical = bundle / "judge-snapshots/run/gpt.json"
    renamed = bundle / "renamed-identical-gpt-result.json"
    shutil.copyfile(canonical, renamed)
    binding_path = bundle / "binding_manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["files"]["gpt_judge_result_manifest"]["path"] = renamed.name
    binding_path.write_bytes(_canonical(binding) + b"\n")

    with pytest.raises(journal_binder.BinderError, match="fixed.*path|canonical.*path"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_symlinked_canonical_bundle_role(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    canonical = bundle / "analysis/analysis_results.json"
    outside = tmp_path / "outside-analysis-results.json"
    shutil.move(canonical, outside)
    canonical.symlink_to(outside)

    with pytest.raises(journal_binder.BinderError, match="symlink|unsafe|regular"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_symlinked_analysis_artifact(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    artifact = bundle / "analysis/figures/blind_output_stability.png"
    outside = tmp_path / "outside-stability.png"
    shutil.move(artifact, outside)
    artifact.symlink_to(outside)

    with pytest.raises(journal_binder.BinderError, match="symlink|unsafe|regular"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_duplicate_binding_role_json_key(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(
        tmp_path / "persona", judge_profile="gpt_only"
    )
    binding_path = bundle / "binding_manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    role_parts: list[bytes] = []
    for role, descriptor in binding["files"].items():
        encoded = _canonical(role) + b":" + _canonical(descriptor)
        role_parts.append(encoded)
        if role == "gpt_judge_result_manifest":
            role_parts.append(encoded)
    raw_files = b"{" + b",".join(role_parts) + b"}"
    binding_path.write_bytes(
        b'{"analysis_population":"main","files":'
        + raw_files
        + b',"run_id":"llm-personas-v2-dual"'
        + b',"schema_version":"yher.journal_binder.persona_v2_bundle.v1"'
        + b',"simulated":true}\n'
    )

    with pytest.raises(journal_binder.BinderError, match="duplicate JSON key"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_stray_embedded_result_for_unavailable_judge(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(
        tmp_path / "persona", judge_profile="gpt_only"
    )
    _rewrite_persona_result_bundle(
        bundle,
        lambda result: result["judge_adjudication"]["result_manifests"].__setitem__(
            "claude", {"stray": True}
        ),
    )

    with pytest.raises(journal_binder.BinderError, match="result manifest.*profile"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


@pytest.mark.parametrize("mutation", ["missing", "mismatched"])
def test_persona_v2_rejects_incomplete_or_mismatched_embedded_result_manifest(
    tmp_path: Path,
    mutation: str,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona")

    def mutate(result: dict[str, object]) -> None:
        adjudication = result["judge_adjudication"]
        assert isinstance(adjudication, dict)
        manifests = adjudication["result_manifests"]
        assert isinstance(manifests, dict)
        if mutation == "missing":
            manifests.pop("gpt")
        else:
            gpt = manifests["gpt"]
            assert isinstance(gpt, dict)
            gpt["run_id"] = "tampered-run"

    _rewrite_persona_result_bundle(bundle, mutate)

    with pytest.raises(journal_binder.BinderError, match="result manifest.*profile"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_judge_run_evidence_binding_drift(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    _rewrite_persona_result_bundle(
        bundle,
        lambda result: result["judge_adjudication"]["run_evidence_binding"].__setitem__(
            "judge_run_evidence_receipt_sha256", "0" * 64
        ),
    )

    with pytest.raises(journal_binder.BinderError, match="run evidence binding.*drift"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_zero_case_rejects_bound_judge_result_input_row(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(
        tmp_path / "persona", judge_profile="zero_cases"
    )
    _rewrite_persona_input_manifest_bundle(
        bundle,
        lambda manifest: manifest["files"].append(
            {
                "path": "judge-results/gpt.json",
                "sha256": "0" * 64,
                "size": 0,
            }
        ),
    )

    with pytest.raises(journal_binder.BinderError, match="judge.*input.*snapshot"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_snapshot_raw_attempt_missing_from_analysis_inputs(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(
        tmp_path / "persona", judge_profile="gpt_only"
    )

    def remove_raw_attempt(manifest: dict[str, object]) -> None:
        files = manifest["files"]
        assert isinstance(files, list)
        raw_rows = [row for row in files if "/raw_attempts/" in row["path"]]
        assert raw_rows
        files.remove(raw_rows[0])

    _rewrite_persona_input_manifest_bundle(bundle, remove_raw_attempt)

    with pytest.raises(journal_binder.BinderError, match="judge.*input.*snapshot"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def _persona_snapshot_file(bundle: Path, *, judge: str, kind: str) -> Path:
    snapshot_path = bundle / "judge-snapshots/snapshot_manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    matches = []
    for row in snapshot["files"]:
        relative = str(row["path"])
        if not relative.startswith(f"run/executions/{judge}/"):
            continue
        if kind == "normalized_results" and relative.endswith(
            "/normalized_results.jsonl"
        ):
            matches.append(relative)
        elif kind == "raw_attempt" and "/raw_attempts/" in relative:
            matches.append(relative)
    assert len(matches) == 1
    return snapshot_path.parent / matches[0]


def _rehash_persona_snapshot_tree(bundle: Path) -> None:
    snapshot_path = bundle / "judge-snapshots/snapshot_manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for row in snapshot["files"]:
        path = snapshot_path.parent / row["path"]
        if path.is_file():
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["size"] = len(path.read_bytes())
    snapshot["file_count"] = len(snapshot["files"])
    snapshot["file_set_sha256"] = hashlib.sha256(
        _canonical(snapshot["files"])
    ).hexdigest()
    snapshot["snapshot_manifest_sha256"] = _self_hash(
        snapshot, "snapshot_manifest_sha256"
    )
    snapshot_path.write_bytes(_canonical(snapshot) + b"\n")

    analysis_root = bundle / "analysis"
    artifact_path = analysis_root / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for row in artifact["artifacts"]:
        path = analysis_root / row["path"]
        if path.is_file():
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["size"] = len(path.read_bytes())
    artifact["artifact_set_sha256"] = hashlib.sha256(
        _canonical(artifact["artifacts"])
    ).hexdigest()
    artifact_path.write_bytes(_canonical(artifact) + b"\n")

    binding_path = bundle / "binding_manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["files"]["judge_run_execution_snapshot_manifest"]["sha256"] = (
        hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    )
    binding["files"]["analysis_artifact_manifest"]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    binding_path.write_bytes(_canonical(binding) + b"\n")


def test_persona_v2_rejects_rehashed_raw_snapshot_that_differs_from_receipt(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    raw_path = _persona_snapshot_file(bundle, judge="gpt", kind="raw_attempt")
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    _rehash_persona_snapshot_tree(bundle)

    with pytest.raises(
        journal_binder.BinderError, match="snapshot cannot replay|exact tree|raw"
    ):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


@pytest.mark.parametrize(
    "kind",
    (
        "normalized_results",
        "raw_attempt",
    ),
)
def test_persona_v2_rejects_missing_required_snapshot_file(
    tmp_path: Path,
    kind: str,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona", judge_profile="gpt_only")
    _persona_snapshot_file(bundle, judge="gpt", kind=kind).unlink()

    with pytest.raises(
        journal_binder.BinderError, match="snapshot cannot replay|missing file|file set"
    ):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_missing_judge_profile_cannot_report_pairwise_agreement(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona", judge_profile="gpt_only")
    _rewrite_persona_result_bundle(
        bundle,
        lambda result: result["judge_adjudication"]["analysis"].__setitem__(
            "pairwise_label_agreement",
            {
                "metric": "judge_label_agreement",
                "judges": ["claude", "gpt"],
                "exact_agreement_numerator": 1,
                "denominator": 1,
                "exact_agreement": 1.0,
                "cohen_kappa": 1.0,
            },
        ),
    )

    with pytest.raises(journal_binder.BinderError, match="missing judge.*pairwise"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: result["controlled"].__setitem__("paired_effects", []),
            "paired effect",
        ),
        (
            lambda result: result["controlled"]["paired_effects"][0].__setitem__(
                "paired_persona_denominator_range", [49, 50]
            ),
            "denominator",
        ),
        (
            lambda result: result["blind"]["agreement"]["pairs"][0].__setitem__(
                "denominator", 99
            ),
            "agreement",
        ),
        (
            lambda result: result["blind"]["stability"][0].__setitem__(
                "answer_agreement_numerator", 21
            ),
            "stability",
        ),
        (
            lambda result: result["provider_lifecycle"][0].__setitem__(
                "controlled_eligible", False
            ),
            "lifecycle|eligible",
        ),
        (
            lambda result: result["judge_adjudication"]["analysis"].__setitem__(
                "schema_version", "yher.llm_sim_v2.judge_analysis.v1"
            ),
            "judge.*v2|schema",
        ),
        (
            lambda result: result["judge_adjudication"]["analysis"][
                "judge_models"
            ].__setitem__("gpt", "claude-opus-4"),
            "judge.*independent|model|execution receipt",
        ),
        (
            lambda result: result["judge_adjudication"]["analysis"][
                "judge_transports"
            ].__setitem__("gpt", "claude_cli"),
            "transport",
        ),
        (
            lambda result: result["judge_adjudication"]["analysis"][
                "judge_accounting"
            ]["gpt"].__setitem__("accounted_cost_yuan", 9.0),
            "accounting",
        ),
        (
            lambda result: result["judge_adjudication"]["analysis"][
                "execution_receipt_sha256"
            ].__setitem__("gpt", "0" * 64),
            "receipt.*(?:hash|drift)|judge result",
        ),
    ),
)
def test_persona_v2_rejects_incomplete_or_unreconciled_quantitative_surfaces(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    _rewrite_persona_result_bundle(bundle, mutation)

    with pytest.raises(journal_binder.BinderError, match=message):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_requires_hash_bound_publication_tables_and_figures(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    artifact_path = bundle / "analysis/artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["artifacts"] = [
        row
        for row in artifact["artifacts"]
        if row["path"] != "figures/blind_output_stability.png"
    ]
    artifact["artifact_set_sha256"] = hashlib.sha256(
        _canonical(artifact["artifacts"])
    ).hexdigest()
    artifact_path.write_bytes(_canonical(artifact) + b"\n")
    binding_path = bundle / "binding_manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["files"]["analysis_artifact_manifest"]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    binding_path.write_bytes(_canonical(binding) + b"\n")

    with pytest.raises(journal_binder.BinderError, match="publication artifact"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_noncanonical_analysis_artifact_path_alias(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    artifact_path = bundle / "analysis/artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    original = next(
        row
        for row in artifact["artifacts"]
        if row["path"] == "controlled_composition.csv"
    )
    artifact["artifacts"].append(
        {
            **original,
            "path": "./controlled_composition.csv",
        }
    )
    artifact["artifact_set_sha256"] = hashlib.sha256(
        _canonical(artifact["artifacts"])
    ).hexdigest()
    artifact_path.write_bytes(_canonical(artifact) + b"\n")
    binding_path = bundle / "binding_manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["files"]["analysis_artifact_manifest"]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    binding_path.write_bytes(_canonical(binding) + b"\n")

    with pytest.raises(journal_binder.BinderError, match="non-canonical|path.*unsafe"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_pilot_tasks_in_the_main_roster(tmp_path: Path) -> None:
    from experiments import journal_binder

    assert hasattr(journal_binder, "bind_persona_v2_artifacts")
    bundle = _write_persona_v2_bundle(tmp_path / "persona", pilot_overlap=True)

    with pytest.raises(journal_binder.BinderError, match="pilot.*main.*disjoint"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_persona_v2_rejects_unbound_analysis_output_bytes(tmp_path: Path) -> None:
    from experiments import journal_binder

    assert hasattr(journal_binder, "bind_persona_v2_artifacts")
    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    result_path = bundle / "analysis/analysis_results.json"
    result_path.write_bytes(result_path.read_bytes() + b" ")

    with pytest.raises(journal_binder.BinderError, match="bundle file SHA-256"):
        journal_binder.bind_persona_v2_artifacts(bundle, allow_fixture=True)


def test_build_binder_uses_the_formal_persona_artifact_binder(tmp_path: Path) -> None:
    from experiments import journal_binder

    assert "persona_v2_dir" in inspect.signature(
        journal_binder.build_binder
    ).parameters
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    registry = _write_registry(tmp_path / "registry.json")
    persona = _write_persona_v2_bundle(tmp_path / "persona")

    bound = journal_binder.build_binder(
        raw_manifest_path=raw,
        registry_path=registry,
        persona_v2_dir=persona,
        p2_dir=None,
        require_complete=False,
        allow_fixture=True,
    )

    assert bound["persona_v2"]["status"] == "bound_formal_w3"


def test_cli_exposes_paper_manifest_and_persona_bundle_inputs() -> None:
    from experiments import journal_binder

    destinations = {action.dest for action in journal_binder._parser()._actions}
    assert "paper_artifact_manifest" in destinations
    assert "persona_v2_dir" in destinations


def test_item_type_diagnostics_bind_exact_rosters_from_raw_events(tmp_path: Path) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    registry = _write_registry(tmp_path / "registry.json")
    rows = json.loads(registry.read_text(encoding="utf-8"))
    for item_type, n_target in (("mcq", 3), ("numeric", 2)):
        rows.append(
            {
                "metric_id": f"misspecification.item_type.{item_type}.generator_minus_production",
                "value": 0.01,
                "numerator": 1,
                "denominator": 100,
                "weighting": ITEM_WEIGHT,
                "n_target": n_target,
                "n_pair": 50,
                "raw_hash": "raw-aggregate",
                "ci_low": 0.0,
                "ci_high": 0.02,
            }
        )
    _write_registry(registry, rows=rows)

    bound = journal_binder.build_binder(
        raw_manifest_path=raw,
        registry_path=registry,
        p2_dir=None,
        require_complete=False,
    )

    mcq = bound["metrics"]["misspecification.item_type.mcq.generator_minus_production"]
    numeric = bound["metrics"][
        "misspecification.item_type.numeric.generator_minus_production"
    ]
    assert mcq["target_roster"] == ["alpha", "beta", "gamma"]
    assert numeric["target_roster"] == ["alpha", "beta"]
    assert mcq["target_set_hash"] != numeric["target_set_hash"]
    assert mcq["claim_class"] == "exploratory_generator_diagnostic"
    assert "condition == misspecified; condition == misspecified" not in mcq[
        "filter_predicate"
    ]
    assert bound["unbound_metrics"] == {}
    assert not any("item_type" in gap for gap in bound["evidence_gaps"])


def test_h4_predicates_bind_the_state_specific_estimands(tmp_path: Path) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    registry = _write_registry(tmp_path / "registry.json")
    rows = json.loads(registry.read_text(encoding="utf-8"))
    for metric_id in (
        "h4.misspecified.b15.rescue_A_minus_B",
        "h4.misspecified.b9.harm_C_minus_A",
    ):
        rows.append(
            {
                "metric_id": metric_id,
                "value": 0.01,
                "numerator": 1,
                "denominator": 100,
                "weighting": "equal_target_then_paired_replicate",
                "n_target": 2,
                "n_pair": 100,
                "raw_hash": "raw-aggregate",
                "ci_low": 0.0,
                "ci_high": 0.02,
            }
        )
    _write_registry(registry, rows=rows)

    bound = journal_binder.build_binder(
        raw_manifest_path=raw,
        registry_path=registry,
        p2_dir=None,
        require_complete=False,
    )

    assert "truth == P" in bound["metrics"][
        "h4.misspecified.b15.rescue_A_minus_B"
    ]["filter_predicate"]
    assert "truth == C" in bound["metrics"][
        "h4.misspecified.b9.harm_C_minus_A"
    ]["filter_predicate"]


def test_all_38_reportable_metrics_have_an_exact_immutable_estimand_spec() -> None:
    from experiments import journal_binder

    assert hasattr(journal_binder, "ESTIMAND_SPECS"), (
        "journal binder must expose the frozen 38-metric estimand table"
    )
    expected = {
        row[0]: {
            "support_id": row[1],
            "conditions": row[2],
            "condition_contrast": row[3],
            "budget": row[4],
            "truth_states": row[5],
            "arms": row[6],
            "weighting": row[7],
            "denominator_kind": row[8],
        }
        for row in _expected_estimand_rows()
    }
    actual = {
        metric_id: dict(spec)
        for metric_id, spec in journal_binder.ESTIMAND_SPECS.items()
    }
    assert len(expected) == 38
    assert tuple(expected) == journal_binder.REPORTABLE_METRIC_IDS
    assert actual == expected


def test_h4_degradation_filters_are_exact_condition_contrasts_at_frozen_budgets(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    registry = _write_registry(
        tmp_path / "registry.json",
        rows=_complete_registry_rows(_raw_aggregate_hash(raw)),
    )

    bound = journal_binder.build_binder(
        raw_manifest_path=raw,
        registry_path=registry,
        p2_dir=None,
        require_complete=False,
    )

    expected = {
        "h4.degradation.h1_rescue.matched_minus_misspecified": (15, ("P",)),
        "h4.degradation.h2_harm.matched_minus_misspecified": (9, ("C",)),
        "h4.degradation.h2_no_harm.matched_minus_misspecified": (9, ("C",)),
    }
    for metric_id, (budget, truth_states) in expected.items():
        metric = bound["metrics"][metric_id]
        assert "filter_dimensions" in metric
        assert metric["filter_dimensions"]["conditions"] == [
            "matched",
            "misspecified",
        ]
        assert (
            metric["filter_dimensions"]["condition_contrast"]
            == "matched_minus_misspecified"
        )
        assert metric["filter_dimensions"]["budget"] == budget
        assert metric["filter_dimensions"]["truth_states"] == list(truth_states)
        assert "condition == misspecified" not in metric["filter_predicate"]


def test_metric_weighting_must_match_the_frozen_estimand_table(tmp_path: Path) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    rows = _complete_registry_rows(_raw_aggregate_hash(raw))
    metric_id = "h4.degradation.h2_harm.matched_minus_misspecified"
    next(row for row in rows if row["metric_id"] == metric_id)["weighting"] = (
        "invented_weighting"
    )
    registry = _write_registry(tmp_path / "registry.json", rows=rows)

    with pytest.raises(journal_binder.BinderError, match="weighting.*frozen estimand"):
        journal_binder.build_binder(
            raw_manifest_path=raw,
            registry_path=registry,
            p2_dir=None,
            require_complete=False,
        )


def test_paired_difference_requires_equal_denominator_pair_count_and_filters(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    rows = _complete_registry_rows(_raw_aggregate_hash(raw))
    difference_id = (
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A."
        "terminal_minus_correct_convergence"
    )
    difference = next(row for row in rows if row["metric_id"] == difference_id)
    difference["denominator"] = 999
    difference["n_pair"] = 999
    registry = _write_registry(tmp_path / "registry.json", rows=rows)

    with pytest.raises(journal_binder.BinderError, match="paired difference denominator"):
        journal_binder.build_binder(
            raw_manifest_path=raw,
            registry_path=registry,
            p2_dir=None,
            require_complete=False,
        )


def test_p2_binding_is_hash_bound_and_preserves_claim_boundary(tmp_path: Path) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    registry = _write_registry(tmp_path / "registry.json")
    p2 = _write_p2(tmp_path / "p2", raw_manifest=raw)

    bound = journal_binder.build_binder(
        raw_manifest_path=raw,
        registry_path=registry,
        p2_dir=p2,
        require_complete=False,
    )
    p2_bound = bound["p2"]
    assert p2_bound["illustrative"] is True
    assert p2_bound["simulated"] is True
    assert p2_bound["external_validity"] is False
    assert p2_bound["claim_boundary"].startswith("supply_bound_algorithmic")
    assert p2_bound["unobtainable_supply_minutes"] is None
    assert p2_bound["exact_overlap_targets"] == ["alpha", "beta"]
    assert p2_bound["budget_seconds"] == 600
    assert [row["arm"] for row in p2_bound["overall"]] == [
        "oracle",
        "A",
        "B",
        "C",
    ]
    assert len(p2_bound["bootstrap_overall"]) == 44
    assert len(p2_bound["bootstrap_contrasts"]) == 44
    assert p2_bound["library_boundary"]["node_count"] == 13
    assert p2_bound["library_boundary"]["trusted_exact_segment_assignments"] == 68
    assert p2_bound["exact_overlap_boundary"] == {
        "target_count": 2,
        "candidate_row_count": 8,
        "physical_source_count": 3,
    }
    assert p2_bound["source_hashes"]["summary_sha256"] == hashlib.sha256(
        (p2 / "summary.json").read_bytes()
    ).hexdigest()
    input_manifest = json.loads(
        (p2 / "input_manifest.json").read_text(encoding="utf-8")
    )
    assert p2_bound["source_hashes"]["trusted_candidate_jsonl_sha256"] == (
        input_manifest["source_files"]["trusted_candidate_jsonl"]["sha256"]
    )
    assert p2_bound["source_hashes"]["p2_spec_sha256"] == input_manifest["spec"][
        "sha256"
    ]
    assert p2_bound["publication_assets"]["supplement_figure_png"][
        "relative_path"
    ] == "p2_supply_bound_illustration.png"
    assert p2_bound["publication_assets"]["supplement_figure_svg"][
        "relative_path"
    ] == "p2_supply_bound_illustration.svg"


def test_p2_requires_an_input_manifest(tmp_path: Path) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    p2 = _write_p2(tmp_path / "p2", raw_manifest=raw)
    (p2 / "input_manifest.json").unlink()

    with pytest.raises(journal_binder.BinderError, match="input manifest"):
        journal_binder._bind_p2(p2)


def test_p2_recursively_rejects_nested_claim_fields(tmp_path: Path) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    p2 = _write_p2(tmp_path / "p2", raw_manifest=raw)
    summary = json.loads((p2 / "summary.json").read_text(encoding="utf-8"))
    summary["overall"][0]["nested"] = {"learning_benefit": 999}
    summary_bytes = _canonical(summary) + b"\n"
    (p2 / "summary.json").write_bytes(summary_bytes)
    output = json.loads((p2 / "output_manifest.json").read_text(encoding="utf-8"))
    summary_entry = next(
        row for row in output["artifacts"] if row["filename"] == "summary.json"
    )
    summary_entry["bytes"] = len(summary_bytes)
    summary_entry["sha256"] = hashlib.sha256(summary_bytes).hexdigest()
    (p2 / "output_manifest.json").write_bytes(_canonical(output) + b"\n")

    with pytest.raises(journal_binder.BinderError, match="prohibited claim field"):
        journal_binder._bind_p2(p2)


def test_p2_verifies_source_file_bytes_instead_of_trusting_hash_gate(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = _write_raw_manifest(raw_root)
    p2 = _write_p2(tmp_path / "p2", raw_manifest=raw)
    input_manifest = json.loads(
        (p2 / "input_manifest.json").read_text(encoding="utf-8")
    )
    candidate_path = Path(
        input_manifest["source_files"]["trusted_candidate_jsonl"]["path"]
    )
    candidate_path.write_bytes(candidate_path.read_bytes() + b"{}\n")

    with pytest.raises(journal_binder.BinderError, match="source file SHA-256"):
        journal_binder._bind_p2(p2)


def _complete_binder(tmp_path: Path) -> dict[str, object]:
    from experiments import journal_binder

    bundle = _write_complete_h1_h4_bundle(tmp_path)
    return journal_binder.build_binder(
        raw_manifest_path=bundle["raw"],
        registry_path=bundle["registry"],
        results_path=bundle["results"],
        p2_dir=None,
        require_complete=True,
    )


def _complete_journal_binder(
    tmp_path: Path,
    *,
    judge_profile: str = "both_complete",
) -> dict[str, object]:
    from experiments import journal_binder

    bundle = _write_complete_h1_h4_bundle(tmp_path / "h1-h4")
    persona = _write_persona_v2_bundle(
        tmp_path / "persona",
        judge_profile=judge_profile,
    )
    p2 = _write_p2(tmp_path / "p2", raw_manifest=bundle["raw"])
    return journal_binder.build_binder(
        raw_manifest_path=bundle["raw"],
        registry_path=bundle["registry"],
        results_path=bundle["results"],
        persona_v2_dir=persona,
        p2_dir=p2,
        require_complete=True,
        allow_fixture=True,
    )


def test_manuscript_slots_are_deterministically_rendered_from_bound_values(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    assert hasattr(journal_binder, "render_manuscript_slots")
    binder = _complete_binder(tmp_path)

    first = journal_binder.render_manuscript_slots(binder)
    second = journal_binder.render_manuscript_slots(binder)

    assert first == second
    assert first["schema_version"] == "yher.journal_binder.manuscript_slots.v1"
    assert "H1 | partially_supported" in first["hypothesis_decisions_markdown"]
    assert "3 intended" in first["execution_integrity_markdown"]
    assert "pending" in first["persona_v2_markdown"].lower()
    assert "full_27_terminal_vs_correct_convergence" in first[
        "same_support_convergence_markdown"
    ]
    assert len(first["content_sha256"]) == 64


def test_bound_persona_and_p2_slots_are_compact_complete_and_claim_bounded(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    slots = journal_binder.render_manuscript_slots(
        _complete_journal_binder(tmp_path)
    )
    persona = slots["persona_v2_markdown"]
    p2 = slots["p2_markdown"]

    for text in (
        "50 persona_id clusters",
        "provider as a repeated measurement",
        "Conditional answer accuracy",
        "80.0%",
        "paired persona denominator 50-50",
        "Blind technical/schema failure rate",
        "2.0%",
        "Blind terminal exact agreement",
        "80.0%",
        "Repeat answer stability",
        "90.0%",
        "Cross-model judge label agreement",
        "2/2",
        "exploratory",
        "not human behavioral validity",
    ):
        assert text in persona
    assert persona.count("<figure") == 1
    assert persona.count("<img ") == 3
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" in persona
    assert "assets/persona_v2/controlled_composition.png" in persona
    assert "assets/persona_v2/blind_terminal_agreement.png" in persona
    assert "assets/persona_v2/blind_output_stability.png" in persona

    for text in (
        "2 targets",
        "8 candidate rows",
        "3 physical sources",
        "600-second analytic budget",
        "P/U role-compatible dose minutes are unavailable and remain null",
        "Arm C",
        "0.500",
        "diagnostic structural failure",
        "supply scarcity",
        "illustrative",
        "does not estimate learning benefit",
    ):
        assert text in p2
    assert p2.count("| oracle |") == 1
    assert p2.count("| A |") == 1
    assert p2.count("| B |") == 1
    assert p2.count("| C |") == 1


def test_gpt_only_slots_disclose_missing_claude_without_pairwise_claim(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    slots = journal_binder.render_manuscript_slots(
        _complete_journal_binder(tmp_path, judge_profile="gpt_only")
    )

    persona = slots["persona_v2_markdown"]
    abstract = slots["bound_abstract_results_markdown"]
    assert "GPT-only exploratory coding" in persona
    assert "Claude judge was unavailable" in persona
    assert "pairwise judge agreement was not estimable" in persona
    assert "GPT-only coding" in abstract
    assert "Claude unavailable" in abstract
    assert "pairwise agreement not estimable" in abstract
    for text in (persona, abstract):
        assert "Cross-model judge label agreement" not in text
        assert "110/120" not in text
    assert "P2 (illustrative;" in abstract


def test_bound_abstract_result_slot_is_compact_but_keeps_core_limits(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    slots = journal_binder.render_manuscript_slots(
        _complete_journal_binder(tmp_path, judge_profile="gpt_only")
    )
    abstract = slots["bound_abstract_results_markdown"]

    assert len(journal_manuscript.ABSTRACT_WORD_PATTERN.findall(abstract)) <= 45
    for text in (
        "95% CI",
        "blind agreement",
        "stability",
        "failure",
        "GPT-only",
        "Claude unavailable",
        "pairwise agreement not estimable",
        "P2",
        "illustrative",
        "Arm-C structural-failure",
    ):
        assert text in abstract


def test_write_binder_publishes_an_atomic_generation_with_slot_fragments(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder

    binder = _complete_binder(tmp_path / "inputs")
    output = tmp_path / "output"

    manifest = journal_binder.write_binder(binder, output)

    current = output / "current"
    assert current.is_symlink()
    generation = current.resolve()
    assert generation.parent == (output / "generations").resolve()
    assert (generation / "journal_binder.json").is_file()
    assert (generation / "manuscript_slots.json").is_file()
    assert (generation / "artifact_manifest.json").is_file()
    disk_manifest = json.loads(
        (generation / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == disk_manifest
    assert {row["filename"] for row in manifest["artifacts"]} == {
        "journal_binder.json",
        "manuscript_slots.json",
    }


def test_interrupted_publish_keeps_the_previous_generation_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import journal_binder

    assert hasattr(journal_binder, "os"), "atomic writer must use replace semantics"
    output = tmp_path / "output"
    old_binder = _complete_binder(tmp_path / "old-inputs")
    journal_binder.write_binder(old_binder, output)
    previous_target = (output / "current").resolve()
    previous_bytes = (previous_target / "journal_binder.json").read_bytes()

    new_binder = json.loads(json.dumps(old_binder))
    new_binder["evidence_gaps"] = ["synthetic-new-generation"]
    real_replace = journal_binder.os.replace

    def interrupted_replace(source: object, destination: object) -> None:
        if Path(destination) == output / "current":
            raise OSError("synthetic publish interruption")
        real_replace(source, destination)

    monkeypatch.setattr(journal_binder.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="synthetic publish interruption"):
        journal_binder.write_binder(new_binder, output)

    assert (output / "current").resolve() == previous_target
    assert (previous_target / "journal_binder.json").read_bytes() == previous_bytes


def _write_finalizer_template(path: Path, *, persona_body: str = "TEMPLATE ONLY") -> Path:
    path.write_text(
        """# Bound Journal Manuscript

## Structured Abstract

**Results:** Frozen H1-H4 results are reported above.

<!-- BEGIN RESULT SLOT: BOUND_ABSTRACT_RESULTS -->
TEMPLATE ONLY
<!-- END RESULT SLOT: BOUND_ABSTRACT_RESULTS -->

## 4. Results

### 4.5 Persona-v2

<!-- BEGIN RESULT SLOT: PERSONA_V2_DUAL -->
"""
        + persona_body
        + """
<!-- END RESULT SLOT: PERSONA_V2_DUAL -->

### 4.6 P2 illustration

<!-- BEGIN RESULT SLOT: P2_ILLUSTRATIVE -->
TEMPLATE ONLY
<!-- END RESULT SLOT: P2_ILLUSTRATIVE -->

## References
""",
        encoding="utf-8",
    )
    return path


def _write_reference_fixture(path: Path) -> Path:
    path.write_bytes(
        _canonical({"schema_version": "yher.verified-references.v1", "references": []})
        + b"\n"
    )
    return path


def test_journal_finalizer_binds_template_binder_slots_assets_and_final_bytes(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    binder = _complete_journal_binder(tmp_path / "inputs")
    binder_root = tmp_path / "binder"
    journal_binder.write_binder(binder, binder_root)
    binder_generation = (binder_root / "current").resolve()
    template = _write_finalizer_template(tmp_path / "template.md")
    references = _write_reference_fixture(tmp_path / "references.json")

    manifest = journal_manuscript.finalize_manuscript(
        template_path=template,
        binder_generation=binder_root / "current",
        references_path=references,
        output_dir=tmp_path / "final",
        expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        expected_binder_generation_id=binder_generation.name,
    )

    final_generation = (tmp_path / "final/current").resolve()
    final_text = (final_generation / "journal_main.md").read_text(encoding="utf-8")
    assert manifest["template"]["sha256"] == hashlib.sha256(
        template.read_bytes()
    ).hexdigest()
    assert manifest["binder"]["generation_id"] == binder_generation.name
    assert len(manifest["binder"]["journal_binder_sha256"]) == 64
    assert len(manifest["slots"]["manuscript_slots_sha256"]) == 64
    assert len(manifest["final_manuscript"]["sha256"]) == 64
    assert manifest["structured_abstract"]["maximum_words"] == 300
    assert 0 < manifest["structured_abstract"]["word_count"] <= 300
    assert "BEGIN RESULT SLOT" not in final_text
    assert "Persona v2: conditional-accuracy shift=" in final_text
    assert "P2 (illustrative;" in final_text
    assert "Formal Persona-v2 main results" in final_text
    assert "Illustrative P2 is supply-bound" in final_text
    for relative in (
        "assets/persona_v2/controlled_composition.png",
        "assets/persona_v2/blind_terminal_agreement.png",
        "assets/persona_v2/blind_output_stability.png",
        "assets/supplement/p2/p2_supply_bound_illustration.png",
        "assets/supplement/p2/p2_supply_bound_illustration.svg",
    ):
        assert (final_generation / relative).is_file()
    verified = journal_manuscript.verify_finalized_generation(
        final_generation, references_path=references
    )
    assert verified == manifest


def test_journal_finalizer_preserves_honest_gpt_only_judge_disclosure(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    binder = _complete_journal_binder(
        tmp_path / "inputs",
        judge_profile="gpt_only",
    )
    binder_root = tmp_path / "binder"
    journal_binder.write_binder(binder, binder_root)
    binder_generation = (binder_root / "current").resolve()
    template = _write_finalizer_template(tmp_path / "template.md")
    references = _write_reference_fixture(tmp_path / "references.json")

    journal_manuscript.finalize_manuscript(
        template_path=template,
        binder_generation=binder_root / "current",
        references_path=references,
        output_dir=tmp_path / "final",
        expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        expected_binder_generation_id=binder_generation.name,
    )

    final_text = (tmp_path / "final/current/journal_main.md").read_text(
        encoding="utf-8"
    )
    assert final_text.count("GPT-only exploratory coding") == 1
    assert final_text.count("Claude judge was unavailable") == 1
    assert final_text.count("pairwise judge agreement was not estimable") == 1
    assert final_text.count("GPT-only coding") == 1
    assert final_text.count("Claude unavailable") == 1
    assert final_text.count("pairwise agreement not estimable") == 1
    assert "Cross-model judge label agreement" not in final_text
    assert "110/120" not in final_text


def test_repository_journal_template_finalizes_without_stale_result_prose(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    binder = _complete_journal_binder(
        tmp_path / "inputs",
        judge_profile="gpt_only",
    )
    binder_root = tmp_path / "binder"
    journal_binder.write_binder(binder, binder_root)
    binder_generation = (binder_root / "current").resolve()
    template = Path(__file__).parents[1] / "docs/paper/journal_main.md"
    references = Path(__file__).parents[1] / "docs/paper/references.json"

    journal_manuscript.finalize_manuscript(
        template_path=template,
        binder_generation=binder_root / "current",
        references_path=references,
        output_dir=tmp_path / "final",
        expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        expected_binder_generation_id=binder_generation.name,
    )

    final_text = (tmp_path / "final/current/journal_main.md").read_text(
        encoding="utf-8"
    )
    journal_manuscript.audit_manuscript(final_text, references_path=references)
    assert journal_manuscript.structured_abstract_word_count(final_text) <= 300
    for stale in (
        "BEGIN RESULT SLOT",
        "no Persona-v2 outcome enters this version",
        "slot remains empty",
        "absent by design",
        "P2 component will",
        "P2 will compare",
        "no Persona-v2 figure",
    ):
        assert stale.lower() not in final_text.lower()
    assert "Persona v2: conditional-accuracy shift=" in final_text
    assert "P2 (illustrative;" in final_text


def test_journal_finalizer_rejects_overlong_fully_bound_structured_abstract(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    binder_root = tmp_path / "binder"
    journal_binder.write_binder(
        _complete_journal_binder(tmp_path / "inputs"), binder_root
    )
    binder_generation = (binder_root / "current").resolve()
    template = _write_finalizer_template(tmp_path / "template.md")
    template.write_text(
        template.read_text(encoding="utf-8").replace(
            "**Results:** Frozen H1-H4 results are reported above.",
            "**Results:** " + " ".join(f"word{index}" for index in range(301)),
        ),
        encoding="utf-8",
    )
    references = _write_reference_fixture(tmp_path / "references.json")

    with pytest.raises(
        journal_manuscript.FinalizationError,
        match="Structured Abstract.*300",
    ):
        journal_manuscript.finalize_manuscript(
            template_path=template,
            binder_generation=binder_root / "current",
            references_path=references,
            output_dir=tmp_path / "final",
            expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
            expected_binder_generation_id=binder_generation.name,
        )
    assert not (tmp_path / "final/current").exists()


def test_journal_finalizer_rejects_pending_binder_and_manual_slot_numbers(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    references = _write_reference_fixture(tmp_path / "references.json")
    pending_root = tmp_path / "pending-binder"
    journal_binder.write_binder(_complete_binder(tmp_path / "pending-inputs"), pending_root)
    pending_generation = (pending_root / "current").resolve()
    template = _write_finalizer_template(tmp_path / "template.md")
    with pytest.raises(journal_manuscript.FinalizationError, match="fully bound"):
        journal_manuscript.finalize_manuscript(
            template_path=template,
            binder_generation=pending_root / "current",
            references_path=references,
            output_dir=tmp_path / "pending-final",
            expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
            expected_binder_generation_id=pending_generation.name,
        )

    complete_root = tmp_path / "complete-binder"
    journal_binder.write_binder(
        _complete_journal_binder(tmp_path / "complete-inputs"), complete_root
    )
    complete_generation = (complete_root / "current").resolve()
    manual = _write_finalizer_template(
        tmp_path / "manual.md", persona_body="TEMPLATE ONLY: 99.9%"
    )
    with pytest.raises(journal_manuscript.FinalizationError, match="manual result number"):
        journal_manuscript.finalize_manuscript(
            template_path=manual,
            binder_generation=complete_root / "current",
            references_path=references,
            output_dir=tmp_path / "manual-final",
            expected_template_sha256=hashlib.sha256(manual.read_bytes()).hexdigest(),
            expected_binder_generation_id=complete_generation.name,
        )

    stale_prose = _write_finalizer_template(tmp_path / "stale-prose.md")
    stale_prose.write_text(
        stale_prose.read_text(encoding="utf-8")
        + "\nPersona-v2 results are absent by design until a later insertion.\n",
        encoding="utf-8",
    )
    with pytest.raises(journal_manuscript.FinalizationError, match="pending manuscript"):
        journal_manuscript.finalize_manuscript(
            template_path=stale_prose,
            binder_generation=complete_root / "current",
            references_path=references,
            output_dir=tmp_path / "stale-prose-final",
            expected_template_sha256=hashlib.sha256(
                stale_prose.read_bytes()
            ).hexdigest(),
            expected_binder_generation_id=complete_generation.name,
        )


def test_journal_finalizer_rejects_stale_binder_source_and_post_finalize_drift(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    references = _write_reference_fixture(tmp_path / "references.json")
    binder = _complete_journal_binder(tmp_path / "inputs")
    binder_root = tmp_path / "binder"
    journal_binder.write_binder(binder, binder_root)
    binder_generation = (binder_root / "current").resolve()
    template = _write_finalizer_template(tmp_path / "template.md")
    persona_source = Path(
        binder["persona_v2"]["publication_assets"][
            "main_persona_composite_sources"
        ]["controlled_composition"]["source_path"]
    )
    original = persona_source.read_bytes()
    persona_source.write_bytes(original + b"drift")
    with pytest.raises(journal_manuscript.FinalizationError, match="stale binder source"):
        journal_manuscript.finalize_manuscript(
            template_path=template,
            binder_generation=binder_root / "current",
            references_path=references,
            output_dir=tmp_path / "stale-final",
            expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
            expected_binder_generation_id=binder_generation.name,
        )
    persona_source.write_bytes(original)

    journal_manuscript.finalize_manuscript(
        template_path=template,
        binder_generation=binder_root / "current",
        references_path=references,
        output_dir=tmp_path / "final",
        expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        expected_binder_generation_id=binder_generation.name,
    )
    final_generation = (tmp_path / "final/current").resolve()
    manuscript = final_generation / "journal_main.md"
    manuscript.write_text(
        manuscript.read_text(encoding="utf-8").replace("80.0%", "81.0%", 1),
        encoding="utf-8",
    )
    with pytest.raises(journal_manuscript.FinalizationError, match="final manuscript.*drift"):
        journal_manuscript.verify_finalized_generation(
            final_generation, references_path=references
        )


def test_journal_finalizer_rechecks_primary_binder_sources(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    references = _write_reference_fixture(tmp_path / "references.json")
    binder = _complete_journal_binder(tmp_path / "inputs")
    assert set(binder["source_artifacts"]) == {
        "raw_manifest",
        "metric_registry",
        "results",
        "paper_artifact_manifest",
    }
    assert "analysis_results" in binder["persona_v2"]["source_artifacts"]
    binder_root = tmp_path / "binder"
    journal_binder.write_binder(binder, binder_root)
    generation = (binder_root / "current").resolve()
    template = _write_finalizer_template(tmp_path / "template.md")
    registry = Path(binder["source_artifacts"]["metric_registry"]["source_path"])
    original = registry.read_bytes()
    registry.write_bytes(original + b" ")

    with pytest.raises(journal_manuscript.FinalizationError, match="stale binder source"):
        journal_manuscript.finalize_manuscript(
            template_path=template,
            binder_generation=binder_root / "current",
            references_path=references,
            output_dir=tmp_path / "stale-final",
            expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
            expected_binder_generation_id=generation.name,
        )
    registry.write_bytes(original)


def test_journal_pdf_metadata_binds_pdf_to_finalized_manuscript(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript

    references = _write_reference_fixture(tmp_path / "references.json")
    binder_root = tmp_path / "binder"
    journal_binder.write_binder(
        _complete_journal_binder(tmp_path / "inputs"), binder_root
    )
    binder_generation = (binder_root / "current").resolve()
    template = _write_finalizer_template(tmp_path / "template.md")
    journal_manuscript.finalize_manuscript(
        template_path=template,
        binder_generation=binder_root / "current",
        references_path=references,
        output_dir=tmp_path / "final",
        expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        expected_binder_generation_id=binder_generation.name,
    )
    final_generation = (tmp_path / "final/current").resolve()
    pdf = tmp_path / "journal_main.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsynthetic-test-pdf\n%%EOF\n")
    metadata_path = tmp_path / "journal_main.pdf.metadata.json"

    metadata = journal_manuscript.write_pdf_metadata(
        pdf_path=pdf,
        finalized_generation=final_generation,
        references_path=references,
        output_path=metadata_path,
    )

    assert metadata["schema_version"] == "yher.journal_pdf.metadata.v1"
    assert metadata["pdf"]["sha256"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert metadata["finalized_manuscript"]["sha256"] == hashlib.sha256(
        (final_generation / "journal_main.md").read_bytes()
    ).hexdigest()
    assert metadata["binder_generation_id"] == binder_generation.name
    assert journal_manuscript.verify_pdf_metadata(
        metadata_path, references_path=references
    ) == metadata

    pdf.write_bytes(pdf.read_bytes() + b"drift")
    with pytest.raises(journal_manuscript.FinalizationError, match="PDF bytes drifted"):
        journal_manuscript.verify_pdf_metadata(
            metadata_path, references_path=references
        )
