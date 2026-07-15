from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

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
        ],
    }
    (root / "output_manifest.json").write_bytes(_canonical(output_manifest) + b"\n")
    return root


def _self_hash(payload: dict[str, object], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_persona_v2_bundle(root: Path, *, pilot_overlap: bool = False) -> Path:
    root.mkdir()
    providers = ["deepseek", "glm", "kimi", "minimax", "doubao", "tongyi"]
    main_ids = ["main-task-0", "main-task-1"]
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
    runtime_path = root / "runtime_task_manifest.json"
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
    mapping_path = root / "mapping_manifest.json"
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
    phase_path = root / "phase_provenance.json"
    phase_path.write_bytes(_canonical(phase) + b"\n")

    lifecycle = [
        {
            "provider": provider,
            "provider_lifecycle": "complete",
            "expected_count": len(main_ids),
            "present_count": len(main_ids),
            "missing_count": 0,
            "missing_task_ids": [],
            "status_counts": {"complete": len(main_ids)},
        }
        for provider in providers
    ]
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
        "controlled": {"composition": [], "paired_effects": []},
        "blind": {
            "eligible_providers": providers,
            "excluded_providers": [],
            "agreement": {},
            "stability": [],
        },
        "outputs": {
            "machine_json": True,
            "machine_csv_tables": 0,
            "figure_data_machine_readable": True,
            "publication_figures": 0,
            "publication_formats": [],
        },
    }
    result_path = root / "analysis_results.json"
    result_path.write_bytes(_canonical(result) + b"\n")
    artifacts = [
        {
            "path": "analysis_results.json",
            "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "size": len(result_path.read_bytes()),
        }
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
    analysis_manifest_path = root / "artifact_manifest.json"
    analysis_manifest_path.write_bytes(_canonical(analysis_manifest) + b"\n")

    bundle_files = {
        "analysis_results": result_path,
        "analysis_artifact_manifest": analysis_manifest_path,
        "phase_provenance": phase_path,
        "runtime_task_manifest": runtime_path,
        "mapping_manifest": mapping_path,
    }
    binding_manifest = {
        "schema_version": "yher.journal_binder.persona_v2_bundle.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "analysis_population": "main",
        "files": {
            role: {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for role, path in bundle_files.items()
        },
    }
    (root / "binding_manifest.json").write_bytes(_canonical(binding_manifest) + b"\n")
    return root


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

    bound = journal_binder.bind_persona_v2_artifacts(bundle)

    assert bound["status"] == "bound_formal_w3"
    assert bound["run_id"] == "llm-personas-v2-dual"
    assert bound["analysis_population"] == "main"
    assert bound["persona_cluster_count"] == 50
    assert bound["independent_unit"] == "persona_id"
    assert bound["provider_role"] == "repeated_measurement"
    assert bound["modality_condition"] == "text_only"
    assert bound["pilot_exclusion"]["task_rosters_disjoint"] is True
    assert bound["pilot_exclusion"]["pilot_task_count"] == 1
    assert bound["pilot_exclusion"]["main_task_count"] == 2
    assert len(bound["provider_lifecycle"]) == 6
    assert bound["mapping"]["mapped_mapping_rows"] == 6
    assert bound["mapping"]["total_mapping_rows"] == 100
    assert len(bound["source_hashes"]["analysis_results_sha256"]) == 64
    assert len(bound["source_hashes"]["analysis_artifact_set_sha256"]) == 64


def test_persona_v2_rejects_pilot_tasks_in_the_main_roster(tmp_path: Path) -> None:
    from experiments import journal_binder

    assert hasattr(journal_binder, "bind_persona_v2_artifacts")
    bundle = _write_persona_v2_bundle(tmp_path / "persona", pilot_overlap=True)

    with pytest.raises(journal_binder.BinderError, match="pilot.*main.*disjoint"):
        journal_binder.bind_persona_v2_artifacts(bundle)


def test_persona_v2_rejects_unbound_analysis_output_bytes(tmp_path: Path) -> None:
    from experiments import journal_binder

    assert hasattr(journal_binder, "bind_persona_v2_artifacts")
    bundle = _write_persona_v2_bundle(tmp_path / "persona")
    result_path = bundle / "analysis_results.json"
    result_path.write_bytes(result_path.read_bytes() + b" ")

    with pytest.raises(journal_binder.BinderError, match="bundle file SHA-256"):
        journal_binder.bind_persona_v2_artifacts(bundle)


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
