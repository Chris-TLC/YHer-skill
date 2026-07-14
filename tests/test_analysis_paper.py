from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from analysis.paper import (
    PaperContractError,
    PaperDriftError,
    bind_papers,
    build_parser,
    derive_hypothesis_branch,
    load_results_contract,
)


BEGIN_RESULTS = "<!-- BEGIN S3 GENERATED RESULTS -->"
END_RESULTS = "<!-- END S3 GENERATED RESULTS -->"
BEGIN_STATUS = "<!-- BEGIN PAPER GENERATED STATUS -->"
END_STATUS = "<!-- END PAPER GENERATED STATUS -->"
BEGIN_PAPER_RESULTS = "<!-- BEGIN PAPER GENERATED RESULTS -->"
END_PAPER_RESULTS = "<!-- END PAPER GENERATED RESULTS -->"
BEGIN_DISCUSSION = "<!-- BEGIN PAPER GENERATED DISCUSSION -->"
END_DISCUSSION = "<!-- END PAPER GENERATED DISCUSSION -->"
BEGIN_ABSTRACT_EN = "<!-- BEGIN PAPER GENERATED ABSTRACT FINDINGS EN -->"
END_ABSTRACT_EN = "<!-- END PAPER GENERATED ABSTRACT FINDINGS EN -->"
BEGIN_ABSTRACT_ZH = "<!-- BEGIN PAPER GENERATED ABSTRACT FINDINGS ZH -->"
END_ABSTRACT_ZH = "<!-- END PAPER GENERATED ABSTRACT FINDINGS ZH -->"
BEGIN_DEFENSE_DIGEST = "<!-- BEGIN PAPER GENERATED DEFENSE DIGEST -->"
END_DEFENSE_DIGEST = "<!-- END PAPER GENERATED DEFENSE DIGEST -->"


H1_IDS = {
    "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS_CI95",
    "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT",
    "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT_CI95",
}
H2_IDS = {
    *{
        f"H2_C_{arm}_MISDIAGNOSIS_MATCHED_B9_{population}"
        for arm in ("A", "B", "C")
        for population in ("ELIGIBLE_STRESS", "COMMON_SUPPORT")
    },
    *{
        f"H2_C_{contrast}_MISDIAGNOSIS_MATCHED_B9_{population}{suffix}"
        for contrast in ("C_MINUS_A", "A_MINUS_B")
        for population in ("ELIGIBLE_STRESS", "COMMON_SUPPORT")
        for suffix in ("", "_CI95")
    },
}
H3_IDS = {
    "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET",
    "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET_CI95",
    "H3_A_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
    "H3_B_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
}
H4_IDS = {
    "H4_H1_RESCUE_MISSPECIFIED_B15_ELIGIBLE_STRESS",
    "H4_H2_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS",
    "H4_H1_MATCHED_TO_MISSPECIFIED_DEGRADATION",
    "H4_H2_MATCHED_TO_MISSPECIFIED_DEGRADATION",
    "H4_H2_NO_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS",
    "H4_H2_NO_HARM_MATCHED_TO_MISSPECIFIED_DEGRADATION",
    "H4_H1_RESCUE_MISSPECIFIED_B15_COMMON_SUPPORT",
    "H4_H1_RESCUE_MISSPECIFIED_B15_COMMON_SUPPORT_CI95",
    "H4_H2_HARM_MISSPECIFIED_B9_COMMON_SUPPORT",
    "H4_H2_HARM_MISSPECIFIED_B9_COMMON_SUPPORT_CI95",
    "H4_H2_A_MINUS_B_MISDIAGNOSIS_MISSPECIFIED_B9_COMMON_SUPPORT",
    "H4_H2_A_MINUS_B_MISDIAGNOSIS_MISSPECIFIED_B9_COMMON_SUPPORT_CI95",
    "H4_MCQ_GENERATOR_MINUS_PRODUCTION_MISSPECIFIED_DIAGNOSTIC",
    "H4_NUMERIC_GENERATOR_MINUS_PRODUCTION_MISSPECIFIED_DIAGNOSTIC",
}
ITEM_TYPE_DIAGNOSTIC_IDS = {
    "H4_MCQ_GENERATOR_MINUS_PRODUCTION_MISSPECIFIED_DIAGNOSTIC",
    "H4_NUMERIC_GENERATOR_MINUS_PRODUCTION_MISSPECIFIED_DIAGNOSTIC",
}
H5_IDS = {
    "H5_QUALIFYING_PROVIDER_COUNT",
    "H5_MINIMUM_COMPLETED_PERSONAS_PER_QUALIFYING_CELL",
    "H5_WEAK_ACCURACY_GATE",
    "H5_STRONG_ACCURACY_GATE",
    "H5_MISCONCEPTION_HIT_RATE_CONTRAST",
    "H5_MISCONCEPTION_HIT_RATE_CONTRAST_CI95",
}
REQUIRED_PROGRAMMATIC_IDS = H1_IDS | H2_IDS | H3_IDS | H4_IDS
REQUIRED_FIGURES = {
    "FIG_P_RESCUE",
    "FIG_C_PROBE_HARM",
    "FIG_MATCHED_VS_MISSPECIFIED",
    "FIG_CONFUSION_MATRICES",
    "FIG_HELDOUT_BRIER",
    "FIG_CONVERGENCE_DISTRIBUTION",
    "FIG_MISSPECIFICATION_BY_ITEM_TYPE",
    "FIG_PROVIDER_AGREEMENT",
    "FIG_MANIPULATION_CHECKS",
}
YAU_PROGRAMMATIC_IDS = {
    "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS_CI95",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT_CI95",
    "H2_C_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
    "H2_C_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
    "H2_C_C_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
    "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
    "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
    "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
    "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
    "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT",
    "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95",
    "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT",
    "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95",
    "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET",
    "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET_CI95",
    "H3_A_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
    "H3_B_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
    "H4_H1_RESCUE_MISSPECIFIED_B15_ELIGIBLE_STRESS",
    "H4_H2_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS",
    "H4_H2_NO_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS",
    "H4_H1_RESCUE_MISSPECIFIED_B15_COMMON_SUPPORT_CI95",
    "H4_H2_HARM_MISSPECIFIED_B9_COMMON_SUPPORT_CI95",
    "H4_H2_A_MINUS_B_MISDIAGNOSIS_MISSPECIFIED_B9_COMMON_SUPPORT_CI95",
}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FROZEN_RUNNER_COMMIT = "33536b4a810d297166e4f1c0f036bb9c70f1a979"
FROZEN_EXPERIMENT_TAG = "experiment-freeze-20260713"
FROZEN_CONFIG_SHA256 = "9020726dcc118ae5f7d4c4879421a2b54f13f7a3ffc61119d76bc0577dfb2501"
FROZEN_MANIFEST_SHA256 = "2c68cada6c2229e6860d46fca4e4f65b3df674bfc4652b4a947934ba05e76dd3"
FROZEN_ANALYSIS_PLAN_COMMIT = "6c559b6f2f8cbe9ab61808c351df3743dca1a0be"
FROZEN_ANALYSIS_PLAN_SHA256 = "662e6844dfbabf8942d787d6b7c2ef37d92c995f22e5968563fa0d9861317fd5"
FIXTURE_RAW_HASH = "3" * 64


@pytest.fixture(autouse=True)
def _isolate_repository_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis.paper import _manuscript_skeleton_sha256

    monkeypatch.setattr(
        "analysis.paper.FROZEN_MANUSCRIPT_SKELETON_SHA256",
        (
            _manuscript_skeleton_sha256(
                _paper_template("Main"), include_zh=False
            ),
            _manuscript_skeleton_sha256(
                _paper_template("Yau"), include_zh=True
            ),
        ),
    )
    monkeypatch.setattr(
        "analysis.paper._validate_repository_provenance",
        lambda _source, _analysis: None,
    )
    monkeypatch.setattr(
        "analysis.paper._validate_programmatic_replay",
        lambda _payload, _artifact_root: None,
    )
    monkeypatch.setattr(
        "analysis.paper._validate_h5_replay",
        lambda _payload, _artifact_root: None,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paper_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_paper_value(item) for item in value) + "]"
    number = float(value)  # fixture display values are already contract-validated
    return str(int(number)) if number.is_integer() else f"{number:.6f}"


def _display_record(
    result_id: str,
    *,
    artifact_sha: str,
    value: float | bool | list[float] = 0.2,
    ci95: list[float] | None = None,
) -> dict[str, object]:
    from analysis.results import _contract_metric_specs

    interval = ci95 if ci95 is not None else [0.1, 0.3]
    if result_id.endswith("_CI95"):
        value = interval
    h5_metric_ids = {
        "H5_QUALIFYING_PROVIDER_COUNT": "h5.qualifying_provider_count",
        "H5_MINIMUM_COMPLETED_PERSONAS_PER_QUALIFYING_CELL": (
            "h5.minimum_completed_personas_per_qualifying_cell"
        ),
        "H5_WEAK_ACCURACY_GATE": "h5.weak_accuracy_gate",
        "H5_STRONG_ACCURACY_GATE": "h5.strong_accuracy_gate",
        "H5_MISCONCEPTION_HIT_RATE_CONTRAST": (
            "h5.misconception_hit_rate_contrast"
        ),
        "H5_MISCONCEPTION_HIT_RATE_CONTRAST_CI95": (
            "h5.misconception_hit_rate_contrast"
        ),
    }
    return {
        "registry_metric_id": (
            h5_metric_ids[result_id]
            if result_id in h5_metric_ids
            else _contract_metric_specs()[result_id][0]
        ),
        "value": value,
        "ci95": interval,
        "numerator": 20.0,
        "denominator": 100,
        "weighting": "equal_target_then_replicate",
        "n_target": 23,
        "n_pair": 1150,
        "artifact": "metric_registry.json",
        "artifact_sha256": artifact_sha,
    }


def _set_metric(
    payload: dict[str, object],
    result_id: str,
    value: float | bool | list[float],
    *,
    ci95: list[float] | None = None,
) -> None:
    record = payload["metrics"][result_id]  # type: ignore[index]
    record["value"] = value  # type: ignore[index]
    if ci95 is not None:
        record["ci95"] = ci95  # type: ignore[index]


def _sync_fixture_registry(
    artifact_root: Path, payload: dict[str, object]
) -> str:
    from analysis.results import expected_programmatic_registry_ids

    registry_by_id: dict[str, dict[str, object]] = {}
    for result_id, record in payload["metrics"].items():  # type: ignore[union-attr]
        if record is None:
            continue
        value = record["value"]
        ci95 = record.get("ci95")
        row = {
                "metric_id": record["registry_metric_id"],
                "value": (
                    (float(value[0]) + float(value[1])) / 2
                    if isinstance(value, list)
                    else value
                ),
                "numerator": record["numerator"],
                "denominator": record["denominator"],
                "weighting": record["weighting"],
                "n_target": record["n_target"],
                "n_pair": record["n_pair"],
                "raw_hash": (
                    payload["h5_collection_manifest_sha256"]
                    if result_id in H5_IDS
                    else payload["raw_hash"]
                ),
                "ci_low": ci95[0] if isinstance(ci95, list) else None,
                "ci_high": ci95[1] if isinstance(ci95, list) else None,
            }
        existing = registry_by_id.get(str(row["metric_id"]))
        if existing is None or not result_id.endswith("_CI95"):
            registry_by_id[str(row["metric_id"])] = row
    counts = {
        budget: int(payload["denominators"][f"common_support_target_count_b{budget}"])
        for budget in (9, 15, 25)
    }
    for metric_id in expected_programmatic_registry_ids(counts):
        registry_by_id.setdefault(
            metric_id,
            {
                "metric_id": metric_id,
                "value": 0.2,
                "numerator": 20.0,
                "denominator": 100,
                "weighting": "fixture_generated_metric",
                "n_target": 23,
                "n_pair": 100,
                "raw_hash": payload["raw_hash"],
                "ci_low": 0.1,
                "ci_high": 0.3,
            },
        )
    registry_rows = [registry_by_id[key] for key in sorted(registry_by_id)]
    registry = artifact_root / str(payload["analysis_artifact"])
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(registry_rows, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    registry_sha = _sha(registry)
    payload["analysis_artifact_sha256"] = registry_sha
    for result_id, record in payload["metrics"].items():  # type: ignore[union-attr]
        if record is None:
            continue
        record["artifact"] = payload["analysis_artifact"]
        record["artifact_sha256"] = registry_sha
        record["raw_hash"] = (
            payload["h5_collection_manifest_sha256"]
            if result_id in H5_IDS
            else payload["raw_hash"]
        )
    _sync_fixture_provenance_artifacts(artifact_root, payload)
    return registry_sha


def _sync_fixture_provenance_artifacts(
    artifact_root: Path, payload: dict[str, object]
) -> None:
    source = {
        "run_id": payload["run_id"],
        "run_started_at_utc": payload["source_run_started_at_utc"],
        "runner_commit": payload["runner_commit"],
        "experiment_tag": payload["experiment_tag"],
        "config_sha256": payload["config_sha256"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "analysis_plan_commit": payload["analysis_plan_commit"],
        "analysis_plan_sha256": payload["analysis_plan_sha256"],
    }
    analysis = {
        key: payload[key]
        for key in (
            "analysis_commit",
            "analysis_code_committed_at_utc",
            "analysis_code_sha256",
            "analysis_code_files",
        )
    }
    policy_source = Path(__file__).parents[1] / "analysis/static_audit_policy.json"
    policy_relative = "static_audit_policy.json"
    policy_path = artifact_root / policy_relative
    policy_path.write_bytes(policy_source.read_bytes())
    payload["static_audit_policy"] = {
        "path": policy_relative,
        "sha256": _sha(policy_path),
    }
    h5_results = artifact_root / "h5" / "h5_results.json"
    h5_results.parent.mkdir(parents=True, exist_ok=True)
    h5_results.write_text(
        json.dumps(
            {
                "denominators": dict(payload["denominators"]),
                "provider_exclusion_disclosure": payload.get(
                    "h5_provider_exclusion_disclosure"
                ),
                "ledger": {
                    "totals": {
                        "requests": 10_849,
                        "responses": 10_842,
                        "retries": 5,
                        "input_tokens": 5_571_972,
                        "output_tokens": 1_201_999,
                        "cost_yuan": 103.51977121,
                    }
                },
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    payload["h5_results_file_sha256"] = _sha(h5_results)
    h5_hypothesis = (payload.get("hypotheses") or {}).get("H5")
    if (
        isinstance(h5_hypothesis, dict)
        and h5_hypothesis.get("analysis_status") == "excluded_pre_outcome"
    ):
        h5_hypothesis["predicate_inputs"]["evidence_path"] = "h5/h5_results.json"
        h5_hypothesis["predicate_inputs"]["evidence_sha256"] = payload[
            "h5_results_file_sha256"
        ]
        payload["decision_details"] = copy.deepcopy(payload["hypotheses"])
    results_relative = str(payload.get("results_artifact", "results.json"))
    results = artifact_root / results_relative
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        json.dumps(
            {
                "raw_hash": payload["raw_hash"],
                "source_run_started_at_utc": payload["source_run_started_at_utc"],
                "analysis_code_committed_at_utc": payload[
                    "analysis_code_committed_at_utc"
                ],
                "analysis_timestamp_policy": payload["analysis_timestamp_policy"],
                "source_provenance": source,
                "analysis_provenance": analysis,
                "numeric_source": payload["analysis_artifact"],
                "registry_metric_ids": sorted(
                    record["registry_metric_id"]
                    for record in payload["metrics"].values()
                    if isinstance(record, dict)
                ),
                "conditional_metric_audit": payload["conditional_metric_audit"],
                "validation": payload["denominators"],
                "static_audit_policy": payload["static_audit_policy"],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_relative = str(
        payload.get("artifact_manifest", "artifact_manifest.json")
    )
    artifact_manifest = artifact_root / manifest_relative
    artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
    artifact_manifest.write_text(
        json.dumps(
            {
                "raw_hash": payload["raw_hash"],
                "source_run_started_at_utc": payload["source_run_started_at_utc"],
                "analysis_code_committed_at_utc": payload[
                    "analysis_code_committed_at_utc"
                ],
                "analysis_timestamp_policy": payload["analysis_timestamp_policy"],
                "source_provenance": source,
                "analysis_provenance": analysis,
                "conditional_metric_audit": payload["conditional_metric_audit"],
                "static_audit_policy": payload["static_audit_policy"],
                "registry_metric_ids": sorted(
                    record["registry_metric_id"]
                    for record in payload["metrics"].values()
                    if isinstance(record, dict)
                ),
                "files": {
                    str(payload["analysis_artifact"]): _sha(
                        artifact_root / str(payload["analysis_artifact"])
                    ),
                    results_relative: _sha(results),
                    policy_relative: _sha(policy_path),
                    "h5/h5_results.json": _sha(h5_results),
                },
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    payload["results_artifact"] = results_relative
    payload["results_artifact_sha256"] = _sha(results)
    payload["artifact_manifest"] = manifest_relative
    payload["artifact_manifest_sha256"] = _sha(artifact_manifest)


def _valid_payload(artifact_root: Path, *, h5: str = "excluded") -> dict[str, object]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    registry = artifact_root / "metric_registry.json"
    registry_sha = "0" * 64
    figure_dir = artifact_root / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure_names = (
        "p_rescue.png",
        "c_harm.png",
        "misspecified.png",
        "confusion_terminal.png",
        "confusion_decision.png",
        "heldout_brier.png",
        "convergence.png",
        "misspecification_by_item_type.png",
        "provider_agreement.png",
        "manipulation_checks.png",
    )
    for name in figure_names:
        (figure_dir / name).write_bytes(PNG_MAGIC + name.encode("ascii"))

    metrics = {
        result_id: _display_record(result_id, artifact_sha=registry_sha)
        for result_id in REQUIRED_PROGRAMMATIC_IDS
    }
    metrics.update({result_id: None for result_id in H5_IDS})

    # Predicate inputs and their display records are deliberately duplicated so the
    # binder can detect a stale or manually altered decision surface.
    _set_metric(
        {"metrics": metrics},
        "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
        0.60,
    )
    _set_metric(
        {"metrics": metrics},
        "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
        0.20,
        ci95=[0.05, 0.35],
    )
    _set_metric(
        {"metrics": metrics},
        "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS_CI95",
        [0.05, 0.35],
        ci95=[0.05, 0.35],
    )
    _set_metric(
        {"metrics": metrics},
        "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
        0.20,
        ci95=[0.05, 0.35],
    )
    _set_metric(
        {"metrics": metrics},
        "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
        [0.05, 0.35],
        ci95=[0.05, 0.35],
    )
    _set_metric(
        {"metrics": metrics},
        "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
        0.01,
        ci95=[-0.02, 0.04],
    )
    _set_metric(
        {"metrics": metrics},
        "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
        [-0.02, 0.04],
        ci95=[-0.02, 0.04],
    )
    _set_metric(
        {"metrics": metrics},
        "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET",
        0.02,
        ci95=[0.00, 0.04],
    )
    _set_metric(
        {"metrics": metrics},
        "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET_CI95",
        [0.00, 0.04],
        ci95=[0.00, 0.04],
    )
    _set_metric(
        {"metrics": metrics},
        "H3_A_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
        10.0,
    )
    _set_metric(
        {"metrics": metrics},
        "H3_B_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
        11.0,
    )
    _set_metric(
        {"metrics": metrics},
        "H4_H1_RESCUE_MISSPECIFIED_B15_ELIGIBLE_STRESS",
        0.10,
    )
    _set_metric(
        {"metrics": metrics},
        "H4_H2_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS",
        0.08,
    )
    for result_id, value, interval, event_count, journey_count in (
        (
            "H4_MCQ_GENERATOR_MINUS_PRODUCTION_MISSPECIFIED_DIAGNOSTIC",
            0.04,
            [0.02, 0.06],
            12_000,
            1_100,
        ),
        (
            "H4_NUMERIC_GENERATOR_MINUS_PRODUCTION_MISSPECIFIED_DIAGNOSTIC",
            0.12,
            [0.08, 0.16],
            2_400,
            800,
        ),
    ):
        _set_metric({"metrics": metrics}, result_id, value, ci95=interval)
        metrics[result_id]["numerator"] = value * event_count
        metrics[result_id]["denominator"] = event_count
        metrics[result_id]["n_target"] = 23
        metrics[result_id]["n_pair"] = journey_count
        metrics[result_id]["weighting"] = (
            "equal_target_then_event; misspecified_event_level; "
            "target_stratified_paired_replicate_resample; "
            "journey_cluster_preserved; bootstrap_iterations=10000; "
            "diagnostic_only_not_item_type_H1_H2_estimand"
        )

    hypotheses: dict[str, dict[str, object]] = {
        "H1": {
            "analysis_status": "complete",
            "decision": "supported",
            "branch_reason": "a_rate_at_least_0_50_and_rescue_ci_strictly_positive",
            "predicate_inputs": {
                "a_rate": 0.60,
                "a_rate_threshold": 0.50,
                "rescue_point": 0.20,
                "rescue_ci_low": 0.05,
                "rescue_ci_strict_threshold": 0.0,
            },
        },
        "H2": {
            "analysis_status": "complete",
            "decision": "supported",
            "branch_reason": "harm_ci_strictly_positive_and_no_harm_ci_below_0_05",
            "predicate_inputs": {
                "harm_point": 0.20,
                "harm_ci_low": 0.05,
                "no_harm_point": 0.01,
                "no_harm_ci_high": 0.04,
                "noninferiority_margin": 0.05,
            },
        },
        "H3": {
            "analysis_status": "complete",
            "decision": "supported",
            "branch_reason": "accuracy_ci_nonnegative_and_median_A_no_longer_than_B",
            "predicate_inputs": {
                "accuracy_point": 0.02,
                "accuracy_ci_low": 0.0,
                "accuracy_ci_threshold": 0.0,
                "median_a": 10.0,
                "median_b": 11.0,
                "nonconvergence_encoding": 16.0,
            },
        },
        "H4": {
            "analysis_status": "complete",
            "decision": "supported",
            "branch_reason": "h1_rescue_and_h2_harm_directions_persist",
            "predicate_inputs": {
                "rescue_point": 0.10,
                "harm_point": 0.08,
                "strict_direction_threshold": 0.0,
            },
        },
    }

    if h5 == "excluded":
        hypotheses["H5"] = {
            "analysis_status": "excluded_pre_outcome",
            "decision": None,
            "branch_reason": "no_explicit_machine_annotation_map",
            "predicate_inputs": {
                "source_manifest_sha256": "a" * 64,
                "panel_sha256": "b" * 64,
                "pre_outcome_mapping_exclusions": 50,
                "evidence_path": "h5/h5_results.json",
                "evidence_sha256": "0" * 64,
            },
        }
    else:
        metrics.update(
            {
                result_id: _display_record(result_id, artifact_sha=registry_sha)
                for result_id in H5_IDS
            }
        )
        h5_values: dict[
            str, tuple[float | bool | list[float], list[float] | None]
        ] = {
            "H5_QUALIFYING_PROVIDER_COUNT": (5.0, None),
            "H5_MINIMUM_COMPLETED_PERSONAS_PER_QUALIFYING_CELL": (45.0, None),
            "H5_WEAK_ACCURACY_GATE": (True, None),
            "H5_STRONG_ACCURACY_GATE": (True, None),
            "H5_MISCONCEPTION_HIT_RATE_CONTRAST": (0.15, [0.02, 0.28]),
            "H5_MISCONCEPTION_HIT_RATE_CONTRAST_CI95": (
                [0.02, 0.28],
                [0.02, 0.28],
            ),
        }
        for result_id, (value, interval) in h5_values.items():
            _set_metric(
                {"metrics": metrics},
                result_id,
                value,
                ci95=interval,
            )
        hypotheses["H5"] = {
            "analysis_status": "complete",
            "decision": "supported",
            "branch_reason": "provider_coverage_and_both_manipulation_gates_pass",
            "predicate_inputs": {
                "qualifying_provider_count": 5,
                "minimum_completed_personas_per_qualifying_cell": 45,
                "weak_accuracy_gate": True,
                "strong_accuracy_gate": True,
                "misconception_hit_ci_low": 0.02,
                "maximum_prompt_rewrites": 1,
            },
        }

    decisions = {key: value["decision"] for key, value in hypotheses.items()}
    audited_registry_id = metrics[
        "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"
    ]["registry_metric_id"]
    metrics[
        "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"
    ]["weighting"] = (
        "equal_target_then_replicate; zero_denominator_policy=record_NA_no_redraw; "
        "bootstrap_attempted_iterations=100; bootstrap_defined_iterations=99"
    )
    def png(name: str) -> dict[str, str]:
        return {"path": f"figures/{name}", "sha256": _sha(figure_dir / name)}

    payload = {
        "schema_version": "yher.paper-results.v1",
        "status": (
            "COMPLETE_H5_EXCLUDED_PRE_OUTCOME"
            if h5 == "excluded"
            else "COMPLETE_H5_EVALUATED"
        ),
        "run_id": "confirmatory-v1",
        "raw_hash": FIXTURE_RAW_HASH,
        "source_run_started_at_utc": "2026-07-13T13:23:07Z",
        "analysis_code_committed_at_utc": "2026-07-14T00:00:00Z",
        "analysis_timestamp_policy": "analysis_code_commit_time_for_byte_determinism",
        "runner_commit": FROZEN_RUNNER_COMMIT,
        "experiment_tag": FROZEN_EXPERIMENT_TAG,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "analysis_plan_commit": FROZEN_ANALYSIS_PLAN_COMMIT,
        "analysis_plan_sha256": FROZEN_ANALYSIS_PLAN_SHA256,
        "analysis_commit": "4" * 40,
        "analysis_code_sha256": "5" * 64,
        "analysis_code_files": {
            "analysis/results.py": "6" * 64,
            "experiments/analysis_plan.md": FROZEN_ANALYSIS_PLAN_SHA256,
        },
        "conditional_metric_audit": {
            audited_registry_id: {
                "attempted_iterations": 100,
                "defined_iterations": 99,
                "all_targets_undefined_iterations": 1,
                "undefined_target_iterations": {"Target Alpha": 2},
                "observed_undefined_targets": [],
                "point_declared_target_count": 4,
                "point_defined_target_count": 4,
                "point_undefined_target_count": 0,
                "point_undefined_target_ids": [],
                "redraw_count": 0,
            }
        },
        "denominators": {
            "full_target_count": 27,
            "h1_h2_eligible_target_count": 23,
            "common_support_target_count_b9": 9,
            "common_support_target_count_b15": 4,
            "common_support_target_count_b25": 1,
            "valid_journey_count": 30800,
            "structural_failure_count": 1600,
            "schema_invalid_count": 0,
            "schema_invalid_reasons": {},
            "intended_journey_count": 32400,
            "estimand_excluded_journey_count": 1600,
            "estimand_exclusion_reasons": {"structural_failure_item_pool": 1600},
            "estimand_exclusion_arms": {"C": 1600},
            "estimand_exclusion_targets": [
                "Target Alpha",
                "Target Beta",
                "Target Delta",
                "Target Gamma"
            ],
            "excluded_provider_cells": 12,
            "excluded_persona_cells": 464,
            "frozen_provider_count": 6,
            "collected_provider_count": 3 if h5 == "excluded" else 5,
            "qualifying_provider_count": 0 if h5 == "excluded" else 5,
            "invalid_calibration_schema_provider_count": (
                3 if h5 == "excluded" else 0
            ),
            "invalid_provider_artifact_count": 0,
            "missing_provider_count": 0,
            "missing_required_revision_provider_count": 0,
            "network_interruption_provider_count": 1 if h5 == "excluded" else 0,
            "model_drift_exclusion_provider_count": 0,
            "provider_configuration_exclusion_provider_count": 0,
            "pre_outcome_design_exclusion_provider_count": 0,
            "technical_interruption_provider_count": 0,
            "post_calibration_exclusion_provider_count": 2 if h5 == "excluded" else 0,
            "provider_lifecycle_counts": (
                {
                    "invalid_calibration_schema": 3,
                    "network_interruption": 1,
                    "post_calibration_exclusion": 2,
                }
                if h5 == "excluded"
                else {"collected": 5, "missing": 1}
            ),
        },
        "analysis_artifact": (
            "metric_registry.json"
            if h5 == "excluded"
            else "h5/merged_metric_registry.json"
        ),
        "results_artifact": (
            "results.json" if h5 == "excluded" else "h5/merged_results.json"
        ),
        "artifact_manifest": (
            "artifact_manifest.json"
            if h5 == "excluded"
            else "h5/merged_artifact_manifest.json"
        ),
        "h5_collection_manifest_sha256": "8" * 64,
        "h5_provider_exclusion_disclosure": {
            "invalid_calibration_schema": ["deepseek", "glm", "kimi"],
            "network_interruption": ["doubao"],
            "post_calibration_exclusion": ["minimax", "tongyi"],
        },
        "analysis_artifact_sha256": registry_sha,
        "metrics": metrics,
        "hypotheses": hypotheses,
        "decision_details": copy.deepcopy(hypotheses),
        "decisions": decisions,
        "figures": {
            "FIG_P_RESCUE": {"png": png("p_rescue.png")},
            "FIG_C_PROBE_HARM": {"png": png("c_harm.png")},
            "FIG_MATCHED_VS_MISSPECIFIED": {"png": png("misspecified.png")},
            "FIG_CONFUSION_MATRICES": {
                "terminal_png": png("confusion_terminal.png"),
                "decision_png": png("confusion_decision.png"),
            },
            "FIG_HELDOUT_BRIER": {"png": png("heldout_brier.png")},
            "FIG_CONVERGENCE_DISTRIBUTION": {"png": png("convergence.png")},
            "FIG_MISSPECIFICATION_BY_ITEM_TYPE": {
                "png": png("misspecification_by_item_type.png")
            },
            "FIG_PROVIDER_AGREEMENT": {"png": png("provider_agreement.png")},
            "FIG_MANIPULATION_CHECKS": {"png": png("manipulation_checks.png")},
        },
    }
    _sync_fixture_registry(artifact_root, payload)
    return payload


def _write_contract(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        "# Human text containing {\"ignored\": true}\n\n"
        f"{BEGIN_RESULTS}\n\n```json\n"
        f"{json.dumps(payload, sort_keys=True, indent=2)}\n"
        f"```\n\n{END_RESULTS}\n",
        encoding="utf-8",
    )


def _paper_template(name: str) -> str:
    abstract = (
        f"{BEGIN_ABSTRACT_EN}\nPENDING abstract findings\n{END_ABSTRACT_EN}\n\n"
    )
    if name == "Yau":
        abstract += (
            f"{BEGIN_ABSTRACT_ZH}\nPENDING Chinese findings\n{END_ABSTRACT_ZH}\n\n"
        )
    return (
        f"# {name}\n\n"
        + abstract
        + f"{BEGIN_STATUS}\nPENDING status\n{END_STATUS}\n\n"
        "## Results\n\n"
        f"{BEGIN_PAPER_RESULTS}\nPENDING results\n{END_PAPER_RESULTS}\n\n"
        "## Discussion\n\n"
        f"{BEGIN_DISCUSSION}\nIf H1 is supported, placeholder.\n{END_DISCUSSION}\n\n"
        "## References\n\nReference text must survive.\n"
    )


def _defense_template() -> str:
    return (
        "# Defense\n\n"
        f"{BEGIN_DEFENSE_DIGEST}\nPENDING machine digest\n{END_DEFENSE_DIGEST}\n\n"
        "Human defense text must survive.\n"
    )


def _write_papers(root: Path) -> tuple[Path, Path]:
    main = root / "main.md"
    yau = root / "yau.md"
    main.write_text(_paper_template("Main"), encoding="utf-8")
    yau.write_text(_paper_template("Yau"), encoding="utf-8")
    return main, yau


def _fixture(tmp_path: Path, *, h5: str = "excluded") -> tuple[Path, Path, Path, Path]:
    artifacts = tmp_path / "artifacts"
    payload = _valid_payload(artifacts, h5=h5)
    contract = tmp_path / "results_contract.md"
    _write_contract(contract, payload)
    main, yau = _write_papers(tmp_path)
    return contract, artifacts, main, yau


@pytest.mark.parametrize(
    ("hypothesis", "inputs", "decision", "reason"),
    (
        (
            "H1",
            {"a_rate": 0.50, "rescue_point": 0.10, "rescue_ci_low": 0.001},
            "supported",
            "supported",
        ),
        (
            "H1",
            {"a_rate": 0.50, "rescue_point": 0.10, "rescue_ci_low": 0.0},
            "partially_supported",
            "mixed",
        ),
        (
            "H1",
            {"a_rate": 0.49, "rescue_point": 0.0, "rescue_ci_low": -0.01},
            "not_supported",
            "low_rate_and_no_rescue",
        ),
        (
            "H2",
            {
                "harm_point": 0.10,
                "harm_ci_low": 0.01,
                "no_harm_point": 0.01,
                "no_harm_ci_high": 0.049,
            },
            "supported",
            "supported",
        ),
        (
            "H2",
            {
                "harm_point": 0.0,
                "harm_ci_low": -0.01,
                "no_harm_point": 0.05,
                "no_harm_ci_high": 0.06,
            },
            "not_supported",
            "harm_nonpositive_and_a_inferior",
        ),
        (
            "H2",
            {
                "harm_point": 0.0,
                "harm_ci_low": -0.01,
                "no_harm_point": 0.01,
                "no_harm_ci_high": 0.04,
            },
            "not_supported",
            "harm_nonpositive",
        ),
        (
            "H2",
            {
                "harm_point": 0.10,
                "harm_ci_low": 0.01,
                "no_harm_point": 0.05,
                "no_harm_ci_high": 0.06,
            },
            "not_supported",
            "a_inferior",
        ),
        (
            "H2",
            {
                "harm_point": 0.10,
                "harm_ci_low": 0.0,
                "no_harm_point": 0.01,
                "no_harm_ci_high": 0.05,
            },
            "partially_supported",
            "imprecise",
        ),
        (
            "H3",
            {"accuracy_point": 0.0, "accuracy_ci_low": 0.0, "median_a": 10, "median_b": 10},
            "supported",
            "supported",
        ),
        (
            "H3",
            {"accuracy_point": -0.01, "accuracy_ci_low": -0.02, "median_a": 11, "median_b": 10},
            "not_supported",
            "both_favor_b",
        ),
        (
            "H4",
            {"rescue_point": 0.01, "harm_point": -0.01},
            "partially_supported",
            "one_direction",
        ),
        (
            "H5",
            {
                "qualifying_provider_count": 5,
                "minimum_completed_personas_per_qualifying_cell": 45,
                "weak_accuracy_gate": True,
                "strong_accuracy_gate": True,
                "misconception_hit_ci_low": 0.001,
            },
            "supported",
            "supported",
        ),
        (
            "H5",
            {
                "qualifying_provider_count": 4,
                "minimum_completed_personas_per_qualifying_cell": 45,
                "weak_accuracy_gate": True,
                "strong_accuracy_gate": True,
                "misconception_hit_ci_low": 0.001,
            },
            "partially_supported",
            "provider_coverage_partial",
        ),
        (
            "H5",
            {
                "qualifying_provider_count": 3,
                "minimum_completed_personas_per_qualifying_cell": 45,
                "weak_accuracy_gate": True,
                "strong_accuracy_gate": True,
                "misconception_hit_ci_low": 0.001,
            },
            "not_supported",
            "too_few_providers",
        ),
    ),
)
def test_frozen_branch_truth_table_and_boundaries(
    hypothesis: str,
    inputs: dict[str, object],
    decision: str,
    reason: str,
) -> None:
    branch = derive_hypothesis_branch(hypothesis, inputs)

    assert branch.decision == decision
    assert branch.reason_key == reason


def test_contract_parser_reads_only_the_marker_json(tmp_path: Path) -> None:
    contract, _, _, _ = _fixture(tmp_path)

    payload = load_results_contract(contract)

    assert payload["schema_version"] == "yher.paper-results.v1"
    assert "ignored" not in payload


@pytest.mark.parametrize(
    "status",
    (
        "COMPLETE",
        "BOGUS",
        "PROGRAMMATIC_COMPLETE_H5_PENDING",
        "COMPLETE_H5_EVALUATED_EXTRA",
    ),
)
def test_binder_accepts_only_the_two_final_lifecycle_statuses(
    tmp_path: Path, status: str
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["status"] = status
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="lifecycle status"):
        bind_papers(contract, artifacts, main, yau)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_run_started_at_utc", "2026-07-13T13:23:07+00:00"),
        ("analysis_code_committed_at_utc", "2026-07-14T00:00:00.000Z"),
        ("analysis_timestamp_policy", "BOGUS"),
    ),
)
def test_binder_requires_exact_utc_timestamps_and_timestamp_policy(
    tmp_path: Path, field: str, value: str
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload[field] = value
    _sync_fixture_provenance_artifacts(artifacts, payload)
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="timestamp|policy"):
        bind_papers(contract, artifacts, main, yau)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runner_commit", "0" * 40),
        ("experiment_tag", "BOGUS"),
        ("config_sha256", "0" * 64),
        ("source_manifest_sha256", "0" * 64),
        ("analysis_plan_commit", "0" * 40),
        ("analysis_plan_sha256", "0" * 64),
        ("analysis_commit", "0" * 40),
        ("analysis_code_sha256", "0" * 64),
    ),
)
def test_binder_rejects_placeholder_or_noncanonical_provenance(
    tmp_path: Path, field: str, value: str
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload[field] = value
    _sync_fixture_provenance_artifacts(artifacts, payload)
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="provenance|canonical"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_cross_binds_results_and_artifact_manifest_provenance(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    results_path = artifacts / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["source_provenance"]["runner_commit"] = "7" * 40
    results_path.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
    payload["results_artifact_sha256"] = _sha(results_path)
    artifact_manifest_path = artifacts / "artifact_manifest.json"
    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    artifact_manifest["files"]["results.json"] = _sha(results_path)
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, sort_keys=True, indent=2) + "\n"
    )
    payload["artifact_manifest_sha256"] = _sha(artifact_manifest_path)
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="results.*provenance"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_machine_renders_exclusion_and_static_audit_disclosures(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)

    bind_papers(contract, artifacts, main, yau)

    main_text = main.read_text(encoding="utf-8")
    assert "1,600" in main_text
    assert "structural_failure_item_pool" in main_text
    assert "Arm C" in main_text
    for target in ("Target Alpha", "Target Beta", "Target Delta", "Target Gamma"):
        assert target in main_text
    assert "post-collection static audit" in main_text
    assert "no denominator redraw" in main_text
    assert "result direction" in main_text
    assert "invalid calibration schema: `deepseek`, `glm`, `kimi`" in main_text
    assert "network interruption: `doubao`" in main_text
    assert "post calibration exclusion: `minimax`, `tongyi`" in main_text

    yau_text = yau.read_text(encoding="utf-8")
    assert "Machine integrity summary" in yau_text
    assert "1,600 predeclared estimand exclusions" in yau_text
    assert "invalid calibration schema=deepseek/glm/kimi" in yau_text
    assert "network interruption=doubao" in yau_text
    assert "post calibration exclusion=minimax/tongyi" in yau_text


def test_binder_labels_interval_bound_item_type_outputs_as_diagnostics_only(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)

    bind_papers(contract, artifacts, main, yau)

    text = main.read_text(encoding="utf-8")
    assert "item-type generator diagnostic" in text
    assert "mixed trajectories" in text
    assert "not item-type H1/H2 outcome estimands" in text
    assert "MCQ gap=0.040000" in text
    assert "95% CI [0.020000, 0.060000]" in text
    assert "12,000 events / 1,100 journeys / 23 targets" in text
    assert "Numeric gap=0.120000" in text
    assert "2,400 events / 800 journeys / 23 targets" in text
    assert "item-type generator diagnostic" not in yau.read_text(encoding="utf-8")

    payload = load_results_contract(contract)
    diagnostic = payload["metrics"][
        "H4_MCQ_GENERATOR_MINUS_PRODUCTION_MISSPECIFIED_DIAGNOSTIC"
    ]
    diagnostic["weighting"] = "equal_target_then_event"
    _sync_fixture_registry(artifacts, payload)
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="item-type diagnostic weighting"):
        bind_papers(contract, artifacts, main, yau)


def test_registry_rows_bind_raw_hash_and_all_displays_use_one_registry(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    registry_path = artifacts / "metric_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry[0]["raw_hash"] = "7" * 64
    registry_path.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n")
    registry_sha = _sha(registry_path)
    payload["analysis_artifact_sha256"] = registry_sha
    for record in payload["metrics"].values():
        if record is None:
            continue
        record["artifact_sha256"] = registry_sha
    _sync_fixture_provenance_artifacts(artifacts, payload)
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="raw_hash"):
        bind_papers(contract, artifacts, main, yau)

    payload = _valid_payload(artifacts)
    result_id = "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"
    alternate = artifacts / "alternate_registry.json"
    alternate.write_bytes((artifacts / "metric_registry.json").read_bytes())
    payload["metrics"][result_id]["artifact"] = alternate.name
    payload["metrics"][result_id]["artifact_sha256"] = _sha(alternate)
    _write_contract(contract, payload)
    with pytest.raises(PaperContractError, match="single analysis registry"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_rejects_contract_only_display_registry_remap(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    arm_a = "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"
    arm_b = "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"
    payload["metrics"][arm_b] = copy.deepcopy(payload["metrics"][arm_a])
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="canonical registry mapping"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_rejects_rehashed_non_display_registry_row(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    registry_path = artifacts / str(payload["analysis_artifact"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    forged = dict(registry[0])
    forged["metric_id"] = "forged.non_display.extra"
    registry.append(forged)
    registry_path.write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    registry_sha = _sha(registry_path)
    payload["analysis_artifact_sha256"] = registry_sha
    for record in payload["metrics"].values():
        if isinstance(record, dict):
            record["artifact_sha256"] = registry_sha
    _sync_fixture_provenance_artifacts(artifacts, payload)
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="missing or extra IDs"):
        bind_papers(contract, artifacts, main, yau)


def test_replay_rejects_fully_rehashed_programmatic_numeric_forgery(
    tmp_path: Path,
) -> None:
    from analysis.paper import _validate_replayed_programmatic_surface

    contract, artifacts, _, _ = _fixture(tmp_path)
    trusted = tmp_path / "trusted-replay"
    shutil.copytree(artifacts, trusted)
    shutil.copyfile(contract, trusted / "results_contract_block.md")
    payload = load_results_contract(contract)
    registry_path = artifacts / "metric_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry[0]["value"] = float(registry[0]["value"]) + 0.123
    registry_path.write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    forged_sha = _sha(registry_path)
    payload["analysis_artifact_sha256"] = forged_sha
    for record in payload["metrics"].values():
        if isinstance(record, dict):
            record["artifact_sha256"] = forged_sha
    results_path = artifacts / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results_path.write_text(
        json.dumps(results, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    payload["results_artifact_sha256"] = _sha(results_path)
    manifest_path = artifacts / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["metric_registry.json"] = forged_sha
    manifest["files"]["results.json"] = payload["results_artifact_sha256"]
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    payload["artifact_manifest_sha256"] = _sha(manifest_path)

    with pytest.raises(PaperContractError, match="replayed programmatic artifact"):
        _validate_replayed_programmatic_surface(payload, artifacts, trusted)


def test_binder_rejects_self_consistent_contract_exclusion_forgery(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["denominators"].update(
        {
            "estimand_excluded_journey_count": 1,
            "estimand_exclusion_reasons": {"forged_reason": 1},
            "estimand_exclusion_arms": {"A": 1},
            "estimand_exclusion_targets": ["Forged Target"],
        }
    )
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="results validation"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_rejects_self_consistent_source_denominator_forgery(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["denominators"]["full_target_count"] = 26
    payload["denominators"]["h1_h2_eligible_target_count"] = 22
    payload["denominators"]["common_support_target_count_b9"] = 8
    _sync_fixture_provenance_artifacts(artifacts, payload)
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="source manifest denominator"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_rejects_contract_h5_denominator_forgery(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["denominators"]["excluded_provider_cells"] = ["forged"]
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="H5 denominator"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_requires_repository_verified_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)

    def reject(_source: object, _analysis: object) -> None:
        raise PaperContractError("analysis git provenance is invalid")

    monkeypatch.setattr("analysis.paper._validate_repository_provenance", reject)
    with pytest.raises(PaperContractError, match="git provenance"):
        bind_papers(contract, artifacts, main, yau)


def test_repository_provenance_wraps_missing_commit_as_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    source = {
        "run_id": "confirmatory-v1",
        "run_started_at_utc": "2026-07-13T13:23:07Z",
        "runner_commit": FROZEN_RUNNER_COMMIT,
        "experiment_tag": FROZEN_EXPERIMENT_TAG,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "analysis_plan_commit": FROZEN_ANALYSIS_PLAN_COMMIT,
        "analysis_plan_sha256": FROZEN_ANALYSIS_PLAN_SHA256,
    }
    analysis = {
        "analysis_commit": "7" * 40,
        "analysis_code_committed_at_utc": "2025-01-01T00:00:00Z",
        "analysis_code_sha256": "8" * 64,
        "analysis_code_files": {"analysis/results.py": "8" * 64},
    }
    from analysis.paper import _validate_repository_provenance

    with pytest.raises(PaperContractError, match="git provenance"):
        _validate_repository_provenance(source, analysis)


def test_final_binding_requires_every_frozen_stress_and_common_support_id(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    missing = "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95"
    del payload["metrics"][missing]
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match=missing):
        bind_papers(contract, artifacts, main, yau)


@pytest.mark.parametrize("figure_id", sorted(REQUIRED_FIGURES))
def test_final_binding_requires_every_brief_minimum_figure_even_when_h5_excluded(
    tmp_path: Path,
    figure_id: str,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    del payload["figures"][figure_id]
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match=figure_id):
        bind_papers(contract, artifacts, main, yau)


def test_every_required_figure_must_supply_a_verified_png(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    svg = artifacts / "figures" / "p_rescue.svg"
    svg.write_text("<svg/>\n", encoding="utf-8")
    payload = load_results_contract(contract)
    payload["figures"]["FIG_P_RESCUE"] = {
        "svg": {"path": "figures/p_rescue.svg", "sha256": _sha(svg)}
    }
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="FIG_P_RESCUE.*PNG"):
        bind_papers(contract, artifacts, main, yau)


def test_final_binding_rejects_pending_h5_but_accepts_hashed_pre_outcome_exclusion(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["hypotheses"]["H5"] = {
        "analysis_status": "pending_input",
        "decision": None,
        "branch_reason": "waiting",
        "predicate_inputs": {},
    }
    payload["decision_details"] = copy.deepcopy(payload["hypotheses"])
    payload["decisions"]["H5"] = None
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="H5.*pending"):
        bind_papers(contract, artifacts, main, yau)

    excluded = _valid_payload(artifacts, h5="excluded")
    _write_contract(contract, excluded)
    bind_papers(contract, artifacts, main, yau)

    text = main.read_text(encoding="utf-8")
    assert "excluded pre-outcome" in text
    assert "H5" in text
    assert "PENDING" not in text
    assert "If H" not in text


def test_h5_exclusion_evidence_path_and_hash_must_resolve(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["hypotheses"]["H5"]["predicate_inputs"]["evidence_sha256"] = "0" * 64
    payload["decision_details"] = copy.deepcopy(payload["hypotheses"])
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="H5 exclusion evidence.*hash"):
        bind_papers(contract, artifacts, main, yau)


def test_display_records_must_equal_the_registry_rows_semantically(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    result_id = "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"
    payload["metrics"][result_id]["numerator"] = 21.0
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="registry semantic mismatch.*numerator"):
        bind_papers(contract, artifacts, main, yau)


def test_free_form_h5_exclusion_reason_is_rejected(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["hypotheses"]["H5"]["branch_reason"] = "50 mappings unavailable"
    payload["decision_details"] = copy.deepcopy(payload["hypotheses"])
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="H5 exclusion reason"):
        bind_papers(contract, artifacts, main, yau)


def test_known_s3_branch_reason_must_match_recomputed_predicates(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["hypotheses"]["H1"]["branch_reason"] = (
        "a_rate_below_0_50_and_rescue_point_nonpositive"
    )
    payload["decision_details"] = copy.deepcopy(payload["hypotheses"])
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="H1.*branch_reason"):
        bind_papers(contract, artifacts, main, yau)


def test_decisions_are_recomputed_from_display_records_and_h2_reasons_stay_distinct(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    h2 = payload["hypotheses"]["H2"]
    h2["decision"] = "not_supported"
    h2["predicate_inputs"].update(
        {
            "harm_point": 0.20,
            "harm_ci_low": 0.05,
            "no_harm_point": 0.05,
            "no_harm_ci_high": 0.07,
        }
    )
    h2["branch_reason"] = "A_inferior_to_B_margin"
    payload["decision_details"] = copy.deepcopy(payload["hypotheses"])
    payload["decisions"]["H2"] = "not_supported"
    _set_metric(
        payload,
        "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
        0.05,
        ci95=[0.03, 0.07],
    )
    _set_metric(
        payload,
        "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
        [0.03, 0.07],
        ci95=[0.03, 0.07],
    )
    _sync_fixture_registry(artifacts, payload)
    _write_contract(contract, payload)

    bind_papers(contract, artifacts, main, yau)

    discussion = main.read_text(encoding="utf-8")
    assert "fixed-quota harm contrast remained positive" in discussion
    assert "belief-triggered arm failed the no-harm requirement" in discussion
    assert "harm pattern did not survive" not in discussion

    payload["hypotheses"]["H2"]["predicate_inputs"]["harm_point"] = -0.01
    payload["hypotheses"]["H2"]["branch_reason"] = "harm_nonpositive"
    payload["decision_details"] = copy.deepcopy(payload["hypotheses"])
    _write_contract(contract, payload)
    with pytest.raises(PaperContractError, match="predicate.*display"):
        bind_papers(contract, artifacts, main, yau)


def test_binding_is_idempotent_and_yau_is_a_compact_consistent_publication_surface(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path, h5="evaluated")
    payload = load_results_contract(contract)

    bind_papers(contract, artifacts, main, yau)
    first_main = main.read_bytes()
    first_yau = yau.read_bytes()
    bind_papers(contract, artifacts, main, yau)

    assert main.read_bytes() == first_main
    assert yau.read_bytes() == first_yau
    main_text = first_main.decode("utf-8")
    yau_text = first_yau.decode("utf-8")
    main_results = main_text.split(BEGIN_PAPER_RESULTS, 1)[1].split(END_PAPER_RESULTS, 1)[0]
    yau_results = yau_text.split(BEGIN_PAPER_RESULTS, 1)[1].split(END_PAPER_RESULTS, 1)[0]
    assert main_results != yau_results
    assert "PENDING" not in main_text + yau_text
    assert "If H" not in main_text + yau_text
    main_ids = set(re.findall(r"result `([^`]+)`", main_results))
    yau_ids = set(re.findall(r"result `([^`]+)`", yau_results))
    assert main_ids == REQUIRED_PROGRAMMATIC_IDS | H5_IDS
    assert yau_ids == YAU_PROGRAMMATIC_IDS | H5_IDS
    assert yau_ids < main_ids
    assert "<!-- BEGIN YAU MACHINE AUDIT -->" in yau_results
    assert "<!-- END YAU MACHINE AUDIT -->" in yau_results
    visible_yau_results = yau_results.split("<!-- BEGIN YAU MACHINE AUDIT -->", 1)[0]
    assert "H1: A P convergence" in visible_yau_results
    assert "H2: C-state misdiagnosis" in visible_yau_results
    assert "H3: A-B terminal accuracy" in visible_yau_results
    assert "H4: misspecified rescue" in visible_yau_results
    assert "no-repeat C-A=" in visible_yau_results
    assert "fixed-probe harm=" in visible_yau_results
    assert re.search(
        r"fixed-probe harm=\+[^\n]+\(95% CI [^)]+\)", visible_yau_results
    )
    assert "H1_P_A_CORRECT_CONVERGENCE" not in visible_yau_results
    assert "H2_C_C_MINUS_A_MISDIAGNOSIS" not in visible_yau_results

    for result_id in yau_ids:
        record = payload["metrics"][result_id]
        registry_id = record["registry_metric_id"]
        assert f"registry `{registry_id}`" in main_results
        assert f"registry `{registry_id}`" in yau_results
        pattern = rf"result `{re.escape(result_id)}` / registry `[^`]+`: value=([^;]+)"
        main_value = re.search(pattern, main_results)
        yau_value = re.search(pattern, yau_results)
        assert main_value is not None and yau_value is not None
        assert main_value.group(1) == yau_value.group(1)
        assert main_value.group(1) == _paper_value(record["value"])

    # The source hash appears in the audit lifecycle record and source sentence; the
    # H5-results hash appears once in the audit lifecycle record.
    assert yau_results.count(f"sha256:{payload['analysis_artifact_sha256']}") == 2
    assert yau_results.count(f"sha256:{payload['h5_results_file_sha256']}") == 1
    main_abstract = main_text.split(BEGIN_ABSTRACT_EN, 1)[1].split(END_ABSTRACT_EN, 1)[0]
    yau_abstract_en = yau_text.split(BEGIN_ABSTRACT_EN, 1)[1].split(END_ABSTRACT_EN, 1)[0]
    yau_abstract_zh = yau_text.split(BEGIN_ABSTRACT_ZH, 1)[1].split(END_ABSTRACT_ZH, 1)[0]
    for abstract in (main_abstract, yau_abstract_en, yau_abstract_zh):
        assert all(hypothesis in abstract for hypothesis in ("H1", "H2", "H3", "H4", "H5"))
        assert "PENDING" not in abstract
    assert "A rate" in main_abstract
    assert "rescue 95% CI" in main_abstract
    assert "harm 95% CI" in main_abstract
    assert "no-harm 95% CI" in main_abstract


def test_both_render_paths_disclose_machine_bound_h5_exclusion_cells(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)

    bind_papers(contract, artifacts, main, yau)

    for manuscript in (main, yau):
        text = manuscript.read_text(encoding="utf-8")
        assert "12 excluded provider cells" in text
        assert "464 excluded persona cells" in text


def test_yau_machine_audit_binds_every_visible_non_metric_number(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)

    bind_papers(contract, artifacts, main, yau)

    audit = yau.read_text(encoding="utf-8").split(
        "<!-- BEGIN YAU MACHINE AUDIT -->", 1
    )[1].split("<!-- END YAU MACHINE AUDIT -->", 1)[0]
    expected_denominators = {
        "intended_journeys": "intended_journey_count",
        "valid_journeys": "valid_journey_count",
        "estimand_excluded_journeys": "estimand_excluded_journey_count",
        "frozen_providers": "frozen_provider_count",
        "collected_providers": "collected_provider_count",
        "qualifying_providers": "qualifying_provider_count",
        "excluded_provider_cells": "excluded_provider_cells",
        "excluded_persona_cells": "excluded_persona_cells",
    }
    for label, field in expected_denominators.items():
        assert f"{label}={payload['denominators'][field]:,}" in audit
    assert "lifecycle `YAU_VISIBLE_LIFECYCLE_DENOMINATORS`" in audit
    assert (
        f"h5_results_sha256=sha256:{payload['h5_results_file_sha256']}" in audit
    )
    assert (
        f"source_artifact_sha256=sha256:{payload['analysis_artifact_sha256']}" in audit
    )


def test_yau_labels_stress_and_common_support_denominators_from_records(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    for result_id in H1_IDS:
        payload["metrics"][result_id]["denominator"] = (  # type: ignore[index]
            200 if "COMMON_SUPPORT" in result_id else 1_150
        )
    for result_id in H2_IDS:
        payload["metrics"][result_id]["denominator"] = (  # type: ignore[index]
            450 if "COMMON_SUPPORT" in result_id else 1_150
        )
    _sync_fixture_registry(artifacts, payload)
    _write_contract(contract, payload)

    bind_papers(contract, artifacts, main, yau)

    visible = yau.read_text(encoding="utf-8").split(
        "<!-- BEGIN YAU MACHINE AUDIT -->", 1
    )[0]
    assert "stress n=1,150/arm; common-support n=200/arm" in visible
    assert "stress n=1,150/arm; common-support n=450/arm" in visible


def test_defense_digest_is_contract_generated_and_check_detects_drift(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    defense = tmp_path / "defense_pack.md"
    defense.write_text(_defense_template(), encoding="utf-8")

    bind_papers(contract, artifacts, main, yau, defense_path=defense)

    text = defense.read_text(encoding="utf-8")
    digest = text.split(BEGIN_DEFENSE_DIGEST, 1)[1].split(END_DEFENSE_DIGEST, 1)[0]
    assert "PENDING" not in digest
    assert "H1=supported" in digest
    assert "H2=supported" in digest
    assert "H3=supported" in digest
    assert "H4=supported" in digest
    assert "H5=not evaluated (excluded pre-outcome)" in digest
    assert "12 excluded provider cells" in digest
    assert "464 excluded persona cells" in digest
    assert "requests=10,849" in digest
    assert "input tokens=5,571,972" in digest
    assert "output tokens=1,201,999" in digest
    assert "CNY 103.51977121" in digest
    assert "result `H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS`" in digest
    assert "Human defense text must survive." in text

    defense.write_text(text.replace("requests=10,849", "requests=10,850"), encoding="utf-8")
    drifted = defense.read_bytes()
    with pytest.raises(PaperDriftError, match="defense_pack.md"):
        bind_papers(
            contract,
            artifacts,
            main,
            yau,
            defense_path=defense,
            check=True,
        )
    assert defense.read_bytes() == drifted


def test_contract_pngs_are_copied_with_deterministic_names_and_relative_links(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    output = tmp_path / "publication-assets"

    bind_papers(
        contract,
        artifacts,
        main,
        yau,
        figure_output_dir=output,
    )

    copied = sorted(output.glob("*.png"))
    assert len(copied) == 10
    assert not list(output.glob("*.svg"))
    source_hashes = {
        _sha(path): path.read_bytes() for path in (artifacts / "figures").glob("*.png")
    }
    for path in copied:
        digest = _sha(path)
        assert path.read_bytes() == source_hashes[digest]
        assert digest[:12] in path.name

    main_text = main.read_text(encoding="utf-8")
    yau_text = yau.read_text(encoding="utf-8")
    main_links = re.findall(r"!\[[^]]+\]\(([^)]+\.png)\)", main_text)
    yau_links = re.findall(r"!\[[^]]+\]\(([^)]+\.png)\)", yau_text)
    assert len(main_links) == 10
    assert 2 <= len(yau_links) < len(main_links)
    assert set(yau_links) <= set(main_links)
    yau_selected = yau_text.split("### Selected verified figures", 1)[1]
    assert yau_selected.count("::: {.figure-grid}") == 1
    assert yau_selected.count("\n:::\n") == 1
    assert yau_selected.index("::: {.figure-grid}") < yau_selected.index("![")
    assert yau_selected.rindex("\n:::") > yau_selected.rindex("![")
    assert ")\n\n![" in yau_selected
    assert "![C-state misdiagnosis across item budgets]" in yau_selected
    assert "![P-state rescue across item budgets]" in yau_selected
    assert "FIG_" not in yau_selected
    assert " png]" not in yau_selected
    assert all(not Path(link).is_absolute() for link in main_links + yau_links)
    assert all(link.startswith("publication-assets/") for link in main_links + yau_links)

    payload = load_results_contract(contract)
    source_sentence = (
        f"sha256:{payload['analysis_artifact_sha256']}"
    )
    assert source_sentence in main_text
    assert source_sentence in yau_text


def test_figure_paths_are_confined_and_every_supplied_hash_is_verified(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    payload = load_results_contract(contract)
    payload["figures"]["FIG_P_RESCUE"]["png"]["sha256"] = "0" * 64
    _write_contract(contract, payload)

    with pytest.raises(PaperContractError, match="FIG_P_RESCUE.*hash"):
        bind_papers(contract, artifacts, main, yau)

    payload = _valid_payload(artifacts)
    payload["figures"]["FIG_P_RESCUE"]["png"]["path"] = "../outside.png"
    _write_contract(contract, payload)
    with pytest.raises(PaperContractError, match="outside artifact root"):
        bind_papers(contract, artifacts, main, yau)


def test_check_mode_detects_manual_drift_without_writing(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    bind_papers(contract, artifacts, main, yau)
    clean_main = main.read_bytes()

    bind_papers(contract, artifacts, main, yau, check=True)
    main.write_text(
        main.read_text(encoding="utf-8").replace("supported", "manually changed", 1),
        encoding="utf-8",
    )
    drifted = main.read_bytes()

    with pytest.raises(PaperDriftError, match="main.md"):
        bind_papers(contract, artifacts, main, yau, check=True)

    assert main.read_bytes() == drifted
    assert main.read_bytes() != clean_main


def test_binder_rejects_hand_filled_claim_outside_generated_regions(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    bind_papers(contract, artifacts, main, yau)
    main.write_text(
        main.read_text(encoding="utf-8")
        + "\nH1 was supported with a confirmatory rescue of 99%.\n",
        encoding="utf-8",
    )
    forged = main.read_bytes()

    with pytest.raises(PaperContractError, match="skeleton"):
        bind_papers(contract, artifacts, main, yau)
    with pytest.raises(PaperContractError, match="skeleton"):
        bind_papers(contract, artifacts, main, yau, check=True)
    assert main.read_bytes() == forged


def _authorize_skeleton_amendment(
    monkeypatch: pytest.MonkeyPatch,
    manifest: Path,
    *,
    manuscript: str,
    frozen_text: str,
    amended_text: str,
    include_zh: bool = False,
    classification: str = "non_outcome_editorial_audit_correction",
    reviewer: object = None,
    review_status: str = "ai_draft_pending_user_or_claude_review",
    evidence_anchors: object | None = None,
) -> dict[str, object]:
    def normalized(text: str) -> str:
        marker_pairs = [
            (BEGIN_STATUS, END_STATUS),
            (BEGIN_ABSTRACT_EN, END_ABSTRACT_EN),
            (BEGIN_PAPER_RESULTS, END_PAPER_RESULTS),
            (BEGIN_DISCUSSION, END_DISCUSSION),
        ]
        if include_zh:
            marker_pairs.insert(2, (BEGIN_ABSTRACT_ZH, END_ABSTRACT_ZH))
        for begin, end in marker_pairs:
            start = text.index(begin)
            finish = text.index(end, start) + len(end)
            text = (
                text[:start]
                + f"{begin}\n<machine-generated-content>\n{end}"
                + text[finish:]
            )
        return text

    frozen_normalized = normalized(frozen_text)
    amended_normalized = normalized(amended_text)
    frozen_sha256 = hashlib.sha256(frozen_normalized.encode("utf-8")).hexdigest()
    amended_sha256 = hashlib.sha256(amended_normalized.encode("utf-8")).hexdigest()
    normalized_diff = manifest.with_name(f"{manifest.stem}.{manuscript}.patch")
    normalized_diff.write_text(
        "".join(
            difflib.unified_diff(
                frozen_normalized.splitlines(keepends=True),
                amended_normalized.splitlines(keepends=True),
                fromfile=f"frozen-original/{manuscript}",
                tofile=f"current/{manuscript}",
                n=0,
            )
        ),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "schema_version": "yher.paper-skeleton-amendments.v1",
        "policy": "post_collection_non_outcome_editorial_only",
        "amendments": [
            {
                "amendment_id": "2026-07-14-test-honesty-correction",
                "manuscript": manuscript,
                "from_skeleton_sha256": frozen_sha256,
                "to_skeleton_sha256": amended_sha256,
                "classification": classification,
                "generated_outcome_regions_unchanged": True,
                "outcome_knowledge_status": "post_collection_non_outcome_only",
                "recorded_at_utc": "2026-07-14T00:00:00Z",
                "reviewer": reviewer,
                "review_status": review_status,
                "affected_sections": ["Audit disclosure"],
                "rationale": "Correct a non-outcome audit disclosure after collection.",
                "evidence_anchors": (
                    evidence_anchors
                    if evidence_anchors is not None
                    else [
                        {
                            "path": "tests/test_analysis_paper.py",
                            "sha256": _sha(Path(__file__)),
                            "locator": "skeleton amendment contract tests",
                        }
                    ]
                ),
                "normalized_diff_path": normalized_diff.name,
                "normalized_diff_sha256": _sha(normalized_diff),
            }
        ],
    }
    manifest.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "analysis.paper.MANUSCRIPT_SKELETON_AMENDMENT_PATH",
        manifest,
        raising=False,
    )
    monkeypatch.setattr(
        "analysis.paper.MANUSCRIPT_SKELETON_AMENDMENT_SHA256",
        _sha(manifest),
        raising=False,
    )
    return payload


def test_binder_accepts_only_a_hash_pinned_non_outcome_skeleton_amendment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    frozen_text = main.read_text(encoding="utf-8")
    main.write_text(
        frozen_text.replace(
            "Reference text must survive.",
            "Corrected non-outcome isolation disclosure.",
        ),
        encoding="utf-8",
    )
    amended_text = main.read_text(encoding="utf-8")
    _authorize_skeleton_amendment(
        monkeypatch,
        tmp_path / "paper_skeleton_amendments.json",
        manuscript="main.md",
        frozen_text=frozen_text,
        amended_text=amended_text,
    )

    bind_papers(contract, artifacts, main, yau)

    assert "Corrected non-outcome isolation disclosure." in main.read_text(
        encoding="utf-8"
    )


def test_binder_rejects_codex_reviewer_as_a_false_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    frozen_text = main.read_text(encoding="utf-8")
    main.write_text(frozen_text + "\nAudit correction.\n", encoding="utf-8")
    _authorize_skeleton_amendment(
        monkeypatch,
        tmp_path / "paper_skeleton_amendments.json",
        manuscript="main.md",
        frozen_text=frozen_text,
        amended_text=main.read_text(encoding="utf-8"),
        reviewer="codex_post_collection_audit",
        evidence_anchors=["tests/test_analysis_paper.py"],
    )

    with pytest.raises(PaperContractError, match="pending review"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_rejects_unrelated_but_hash_pinned_normalized_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    frozen_text = main.read_text(encoding="utf-8")
    main.write_text(frozen_text + "\nAudit correction.\n", encoding="utf-8")
    manifest = tmp_path / "paper_skeleton_amendments.json"
    payload = _authorize_skeleton_amendment(
        monkeypatch,
        manifest,
        manuscript="main.md",
        frozen_text=frozen_text,
        amended_text=main.read_text(encoding="utf-8"),
    )
    amendment = payload["amendments"][0]  # type: ignore[index]
    diff_path = manifest.parent / str(amendment["normalized_diff_path"])
    diff_path.write_text(
        "--- frozen-original/main.md\n+++ current/main.md\n"
        "@@ -1 +1 @@\n-Unrelated frozen text\n+Unrelated amended text\n",
        encoding="utf-8",
    )
    amendment["normalized_diff_sha256"] = _sha(diff_path)
    manifest.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "analysis.paper.MANUSCRIPT_SKELETON_AMENDMENT_SHA256", _sha(manifest)
    )

    with pytest.raises(PaperContractError, match="diff does not reconstruct"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_rejects_machine_local_session_evidence_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    frozen_text = main.read_text(encoding="utf-8")
    main.write_text(frozen_text + "\nAudit correction.\n", encoding="utf-8")
    _authorize_skeleton_amendment(
        monkeypatch,
        tmp_path / "paper_skeleton_amendments.json",
        manuscript="main.md",
        frozen_text=frozen_text,
        amended_text=main.read_text(encoding="utf-8"),
        evidence_anchors=[
            {
                "path": (
                    "/Users/mac/.codex/sessions/2026/07/14/"
                    "machine-local-session.jsonl"
                ),
                "sha256": "1" * 64,
                "locator": "machine-local evidence",
            }
        ],
    )

    with pytest.raises(PaperContractError, match="evidence anchor"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_rejects_tampered_skeleton_amendment_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    frozen_text = main.read_text(encoding="utf-8")
    main.write_text(frozen_text + "\nAudit correction.\n")
    amended_text = main.read_text(encoding="utf-8")
    manifest = tmp_path / "paper_skeleton_amendments.json"
    _authorize_skeleton_amendment(
        monkeypatch,
        manifest,
        manuscript="main.md",
        frozen_text=frozen_text,
        amended_text=amended_text,
    )
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(PaperContractError, match="amendment manifest hash"):
        bind_papers(contract, artifacts, main, yau)


def test_binder_rejects_tampered_skeleton_amendment_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    frozen_text = main.read_text(encoding="utf-8")
    main.write_text(frozen_text + "\nAudit correction.\n")
    amended_text = main.read_text(encoding="utf-8")
    manifest = tmp_path / "paper_skeleton_amendments.json"
    payload = _authorize_skeleton_amendment(
        monkeypatch,
        manifest,
        manuscript="main.md",
        frozen_text=frozen_text,
        amended_text=amended_text,
    )
    diff_path = manifest.parent / str(
        payload["amendments"][0]["normalized_diff_path"]  # type: ignore[index]
    )
    diff_path.write_text(
        diff_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8"
    )

    with pytest.raises(PaperContractError, match="amendment diff hash"):
        bind_papers(contract, artifacts, main, yau)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("classification", "outcome_reinterpretation", "classification"),
        ("from_skeleton_sha256", "f" * 64, "frozen predecessor"),
        ("generated_outcome_regions_unchanged", False, "generated outcome regions"),
        ("outcome_knowledge_status", "outcome_rewrite", "outcome knowledge"),
        ("reviewer", "", "reviewer"),
        ("review_status", "approved_by_user", "pending review"),
        ("affected_sections", [], "affected sections"),
    ),
)
def test_binder_rejects_non_editorial_or_unanchored_skeleton_amendments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    frozen_text = main.read_text(encoding="utf-8")
    main.write_text(frozen_text + "\nAudit correction.\n")
    amended_text = main.read_text(encoding="utf-8")
    manifest = tmp_path / "paper_skeleton_amendments.json"
    payload = _authorize_skeleton_amendment(
        monkeypatch,
        manifest,
        manuscript="main.md",
        frozen_text=frozen_text,
        amended_text=amended_text,
    )
    payload["amendments"][0][field] = value  # type: ignore[index]
    manifest.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "analysis.paper.MANUSCRIPT_SKELETON_AMENDMENT_SHA256", _sha(manifest)
    )

    with pytest.raises(PaperContractError, match=message):
        bind_papers(contract, artifacts, main, yau)


def test_repository_manuscript_skeletons_are_frozen_or_explicitly_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    from analysis.paper import (
        FROZEN_MANUSCRIPT_SKELETON_SHA256,
        _normalize_manuscript_skeleton,
        _validate_manuscript_skeleton,
    )

    repo_root = Path(__file__).parents[1]
    for index, (relative, include_zh) in enumerate(
        (
            ("docs/paper/main.md", False),
            ("docs/paper/yau_award_4page.md", True),
        )
    ):
        manuscript = repo_root / relative
        observed = _normalize_manuscript_skeleton(
            manuscript.read_text(encoding="utf-8"), include_zh=include_zh
        )
        _validate_manuscript_skeleton(
            manuscript.name,
            observed,
            FROZEN_MANUSCRIPT_SKELETON_SHA256[index],
        )


def test_h5_replay_injects_contract_analysis_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    import analysis.h5 as h5
    import analysis.paper as paper

    repo = tmp_path / "repo"
    fake_module = repo / "analysis/paper.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("# synthetic module location\n", encoding="utf-8")
    (repo / "data/sim_store/llm_personas/llm-personas-v1").mkdir(parents=True)
    monkeypatch.setattr(paper, "__file__", str(fake_module))
    monkeypatch.setattr(h5, "finalize_collection", lambda *args, **kwargs: {})
    captured: dict[str, object] = {}

    def analyze(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(h5, "analyze_collection", analyze)
    monkeypatch.setattr(
        paper,
        "_validate_replayed_h5_surface",
        lambda *args, **kwargs: None,
    )
    provenance = {
        "analysis_commit": "4" * 40,
        "analysis_code_committed_at_utc": "2026-07-14T00:00:00Z",
        "analysis_code_sha256": "5" * 64,
        "analysis_code_files": {"analysis/h5.py": "6" * 64},
    }
    payload = {**provenance, "status": "COMPLETE_H5_EXCLUDED_PRE_OUTCOME"}

    paper._validate_h5_replay(payload, tmp_path)

    assert captured["verified_analysis_provenance"] == provenance


def test_check_mode_detects_missing_or_modified_copied_pngs(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    output = tmp_path / "generated"
    bind_papers(contract, artifacts, main, yau, figure_output_dir=output)
    image = next(output.glob("*.png"))
    expected = image.read_bytes()

    image.write_bytes(PNG_MAGIC + b"manual drift")
    with pytest.raises(PaperDriftError, match="figure.*drift"):
        bind_papers(
            contract,
            artifacts,
            main,
            yau,
            figure_output_dir=output,
            check=True,
        )

    bind_papers(contract, artifacts, main, yau, figure_output_dir=output)
    assert image.read_bytes() == expected
    image.unlink()
    with pytest.raises(PaperDriftError, match="figure.*missing"):
        bind_papers(
            contract,
            artifacts,
            main,
            yau,
            figure_output_dir=output,
            check=True,
        )


def test_generated_png_set_is_exact_and_cleanup_preserves_non_png_files(
    tmp_path: Path,
) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    output = tmp_path / "generated"
    bind_papers(contract, artifacts, main, yau, figure_output_dir=output)
    stale = output / "stale.png"
    keep = output / "notes.txt"
    stale.write_bytes(PNG_MAGIC + b"stale")
    keep.write_text("human-owned\n", encoding="utf-8")

    with pytest.raises(PaperDriftError, match="figure stale"):
        bind_papers(
            contract,
            artifacts,
            main,
            yau,
            figure_output_dir=output,
            check=True,
        )

    bind_papers(contract, artifacts, main, yau, figure_output_dir=output)
    assert not stale.exists()
    assert keep.read_text(encoding="utf-8") == "human-owned\n"


def test_cli_exposes_figure_output_directory_and_defaults_to_defense_pack() -> None:
    args = build_parser().parse_args(["--figure-output-dir", "paper-assets"])

    assert args.figure_output_dir == Path("paper-assets")
    assert args.defense == Path("docs/paper/defense_pack.md")


def test_marker_failure_is_detected_before_either_manuscript_changes(tmp_path: Path) -> None:
    contract, artifacts, main, yau = _fixture(tmp_path)
    yau.write_text(yau.read_text(encoding="utf-8").replace(END_DISCUSSION, ""), encoding="utf-8")
    before_main = main.read_bytes()
    before_yau = yau.read_bytes()

    with pytest.raises(PaperContractError, match="marker"):
        bind_papers(contract, artifacts, main, yau)

    assert main.read_bytes() == before_main
    assert yau.read_bytes() == before_yau
    assert not (tmp_path / "generated").exists()
