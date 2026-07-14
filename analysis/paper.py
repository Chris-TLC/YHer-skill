"""Fail-closed binding of machine results into the two paper manuscripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .dataset import DatasetContractError

from .hypotheses import (
    h1_branch_reason,
    h2_branch_reason,
    h3_branch_reason,
    h4_branch_reason,
)
from .prepare import (
    FROZEN_ANALYSIS_PLAN_COMMIT,
    FROZEN_ANALYSIS_PLAN_SHA256,
    FROZEN_CONFIG_SHA256,
    FROZEN_EXPERIMENT_TAG,
    FROZEN_MANIFEST_SHA256,
    FROZEN_RUN_ID,
    FROZEN_RUNNER_COMMIT,
)
from .provenance import verify_analysis_provenance
from .results import _contract_metric_specs, expected_programmatic_registry_ids


CONTRACT_BEGIN = "<!-- BEGIN S3 GENERATED RESULTS -->"
CONTRACT_END = "<!-- END S3 GENERATED RESULTS -->"
STATUS_BEGIN = "<!-- BEGIN PAPER GENERATED STATUS -->"
STATUS_END = "<!-- END PAPER GENERATED STATUS -->"
RESULTS_BEGIN = "<!-- BEGIN PAPER GENERATED RESULTS -->"
RESULTS_END = "<!-- END PAPER GENERATED RESULTS -->"
DISCUSSION_BEGIN = "<!-- BEGIN PAPER GENERATED DISCUSSION -->"
DISCUSSION_END = "<!-- END PAPER GENERATED DISCUSSION -->"
ABSTRACT_EN_BEGIN = "<!-- BEGIN PAPER GENERATED ABSTRACT FINDINGS EN -->"
ABSTRACT_EN_END = "<!-- END PAPER GENERATED ABSTRACT FINDINGS EN -->"
ABSTRACT_ZH_BEGIN = "<!-- BEGIN PAPER GENERATED ABSTRACT FINDINGS ZH -->"
ABSTRACT_ZH_END = "<!-- END PAPER GENERATED ABSTRACT FINDINGS ZH -->"

DEFAULT_CONTRACT = Path("docs/paper/results_contract.md")
DEFAULT_ARTIFACT_ROOT = Path("/tmp/yher_sprint2/paper_results")
DEFAULT_MAIN = Path("docs/paper/main.md")
DEFAULT_YAU = Path("docs/paper/yau_award_4page.md")
DEFAULT_FIGURE_OUTPUT = Path("docs/paper/generated")


H1_IDS = (
    "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS_CI95",
    "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT",
    "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT_CI95",
)
H2_IDS = tuple(
    [
        f"H2_C_{arm}_MISDIAGNOSIS_MATCHED_B9_{population}"
        for population in ("ELIGIBLE_STRESS", "COMMON_SUPPORT")
        for arm in ("A", "B", "C")
    ]
    + [
        f"H2_C_{contrast}_MISDIAGNOSIS_MATCHED_B9_{population}{suffix}"
        for population in ("ELIGIBLE_STRESS", "COMMON_SUPPORT")
        for contrast in ("C_MINUS_A", "A_MINUS_B")
        for suffix in ("", "_CI95")
    ]
)
H3_IDS = (
    "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET",
    "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET_CI95",
    "H3_A_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
    "H3_B_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
)
H4_IDS = (
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
)
ITEM_TYPE_DIAGNOSTIC_IDS = H4_IDS[-2:]
H5_IDS = (
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
PROGRAMMATIC_IDS = H1_IDS + H2_IDS + H3_IDS + H4_IDS
REQUIRED_FIGURE_IDS = frozenset(
    {
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
)
YAU_PROGRAMMATIC_IDS = (
    "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS_CI95",
    "H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT_CI95",
    "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
    "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
    "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95",
    "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95",
    "H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET_CI95",
    "H3_A_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
    "H3_B_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET",
    "H4_H1_RESCUE_MISSPECIFIED_B15_ELIGIBLE_STRESS",
    "H4_H2_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS",
    "H4_H2_NO_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS",
    "H4_H1_RESCUE_MISSPECIFIED_B15_COMMON_SUPPORT_CI95",
    "H4_H2_HARM_MISSPECIFIED_B9_COMMON_SUPPORT_CI95",
    "H4_H2_A_MINUS_B_MISDIAGNOSIS_MISSPECIFIED_B9_COMMON_SUPPORT_CI95",
)
YAU_FIGURE_IDS = frozenset(
    {
        "FIG_P_RESCUE",
        "FIG_C_PROBE_HARM",
        "FIG_MATCHED_VS_MISSPECIFIED",
        "FIG_MANIPULATION_CHECKS",
    }
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FINAL_STATUSES = frozenset(
    {"COMPLETE_H5_EVALUATED", "COMPLETE_H5_EXCLUDED_PRE_OUTCOME"}
)
ANALYSIS_TIMESTAMP_POLICY = "analysis_code_commit_time_for_byte_determinism"
FROZEN_MANUSCRIPT_SKELETON_SHA256 = (
    "751a719870a18b4fb2699815240401db88c74f3580d3a3e60154d5a520e7e5ff",
    "1afff22d5f8ca706bafe61ba73588a44175cf0b10e7f992b20e74f2fe2eb6045",
)
MANUSCRIPT_SKELETON_AMENDMENT_PATH = (
    Path(__file__).parents[1]
    / "experiments/config/paper_skeleton_amendments_v1.json"
)
MANUSCRIPT_SKELETON_AMENDMENT_SHA256 = (
    "302010d76d52655f06a16d5889f9a4f53e3d28af41279b870b39ab7eca027c09"
)

PREDICATE_RESULT_BINDINGS: Mapping[str, Mapping[str, tuple[str, str]]] = {
    "H1": {
        "a_rate": (H1_IDS[0], "value"),
        "rescue_point": (H1_IDS[2], "value"),
        "rescue_ci_low": (H1_IDS[3], "ci_low"),
    },
    "H2": {
        "harm_point": (
            "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
            "value",
        ),
        "harm_ci_low": (
            "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
            "ci_low",
        ),
        "no_harm_point": (
            "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
            "value",
        ),
        "no_harm_ci_high": (
            "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95",
            "ci_high",
        ),
    },
    "H3": {
        "accuracy_point": (H3_IDS[0], "value"),
        "accuracy_ci_low": (H3_IDS[1], "ci_low"),
        "median_a": (H3_IDS[2], "value"),
        "median_b": (H3_IDS[3], "value"),
    },
    "H4": {
        "rescue_point": (H4_IDS[0], "value"),
        "harm_point": (H4_IDS[1], "value"),
    },
    "H5": {
        "qualifying_provider_count": (H5_IDS[0], "value"),
        "minimum_completed_personas_per_qualifying_cell": (H5_IDS[1], "value"),
        "weak_accuracy_gate": (H5_IDS[2], "value"),
        "strong_accuracy_gate": (H5_IDS[3], "value"),
        "misconception_hit_ci_low": (H5_IDS[5], "ci_low"),
    },
}

FROZEN_CONSTANTS: Mapping[str, Mapping[str, float]] = {
    "H1": {
        "a_rate_threshold": 0.50,
        "rescue_ci_strict_threshold": 0.0,
    },
    "H2": {"noninferiority_margin": 0.05},
    "H3": {
        "accuracy_ci_threshold": 0.0,
        "nonconvergence_encoding": 16.0,
    },
    "H4": {"strict_direction_threshold": 0.0},
}


class PaperContractError(ValueError):
    """Raised when a result contract or paper template is not safe to bind."""


class PaperDriftError(PaperContractError):
    """Raised by check mode when generated manuscript blocks are stale or edited."""


@dataclass(frozen=True)
class HypothesisBranch:
    decision: str
    reason_key: str


@dataclass(frozen=True)
class FigureReference:
    figure_id: str
    key_path: tuple[str, ...]
    source_relative_path: str
    source_path: Path
    sha256: str

    @property
    def is_png(self) -> bool:
        return self.source_path.suffix.lower() == ".png"

    @property
    def output_name(self) -> str:
        figure = _slug(self.figure_id)
        key = _slug("-".join(self.key_path) or "png")
        return f"{figure}-{key}-{self.sha256[:12]}.png"


@dataclass(frozen=True)
class ValidatedContract:
    payload: Mapping[str, Any]
    hypotheses: Mapping[str, Mapping[str, Any]]
    branches: Mapping[str, HypothesisBranch | None]
    required_ids: tuple[str, ...]
    figure_references: Mapping[str, tuple[FigureReference, ...]]


def derive_hypothesis_branch(
    hypothesis: str, predicate_inputs: Mapping[str, object]
) -> HypothesisBranch:
    """Apply the frozen H1-H5 ordered branches and retain the reason sub-branch."""

    if hypothesis == "H1":
        a_rate = _number(predicate_inputs, "a_rate")
        rescue_point = _number(predicate_inputs, "rescue_point")
        rescue_ci_low = _number(predicate_inputs, "rescue_ci_low")
        if a_rate >= 0.50 and rescue_ci_low > 0.0:
            return HypothesisBranch("supported", "supported")
        if a_rate < 0.50 and rescue_point <= 0.0:
            return HypothesisBranch("not_supported", "low_rate_and_no_rescue")
        return HypothesisBranch("partially_supported", "mixed")

    if hypothesis == "H2":
        harm_point = _number(predicate_inputs, "harm_point")
        harm_ci_low = _number(predicate_inputs, "harm_ci_low")
        no_harm_point = _number(predicate_inputs, "no_harm_point")
        no_harm_ci_high = _number(predicate_inputs, "no_harm_ci_high")
        if harm_ci_low > 0.0 and no_harm_ci_high < 0.05:
            return HypothesisBranch("supported", "supported")
        harm_nonpositive = harm_point <= 0.0
        a_inferior = no_harm_point >= 0.05
        if harm_nonpositive and a_inferior:
            return HypothesisBranch(
                "not_supported", "harm_nonpositive_and_a_inferior"
            )
        if harm_nonpositive:
            return HypothesisBranch("not_supported", "harm_nonpositive")
        if a_inferior:
            return HypothesisBranch("not_supported", "a_inferior")
        return HypothesisBranch("partially_supported", "imprecise")

    if hypothesis == "H3":
        accuracy_point = _number(predicate_inputs, "accuracy_point")
        accuracy_ci_low = _number(predicate_inputs, "accuracy_ci_low")
        median_a = _number(predicate_inputs, "median_a")
        median_b = _number(predicate_inputs, "median_b")
        if accuracy_ci_low >= 0.0 and median_a <= median_b:
            return HypothesisBranch("supported", "supported")
        if accuracy_point < 0.0 and median_a > median_b:
            return HypothesisBranch("not_supported", "both_favor_b")
        return HypothesisBranch("partially_supported", "mixed")

    if hypothesis == "H4":
        rescue_point = _number(predicate_inputs, "rescue_point")
        harm_point = _number(predicate_inputs, "harm_point")
        if rescue_point > 0.0 and harm_point > 0.0:
            return HypothesisBranch("supported", "supported")
        if rescue_point <= 0.0 and harm_point <= 0.0:
            return HypothesisBranch("not_supported", "neither_direction")
        return HypothesisBranch("partially_supported", "one_direction")

    if hypothesis == "H5":
        providers = _number(predicate_inputs, "qualifying_provider_count")
        minimum_complete = _number(
            predicate_inputs, "minimum_completed_personas_per_qualifying_cell"
        )
        weak_gate = _boolean(predicate_inputs, "weak_accuracy_gate")
        strong_gate = _boolean(predicate_inputs, "strong_accuracy_gate")
        hit_ci_low = _number(predicate_inputs, "misconception_hit_ci_low")
        accuracy_gate = weak_gate and strong_gate
        hit_gate = hit_ci_low > 0.0
        if (
            providers >= 5
            and minimum_complete >= 45
            and accuracy_gate
            and hit_gate
        ):
            return HypothesisBranch("supported", "supported")
        if providers < 4:
            return HypothesisBranch("not_supported", "too_few_providers")
        if not accuracy_gate and not hit_gate:
            return HypothesisBranch("not_supported", "neither_manipulation_gate")
        if providers < 5:
            return HypothesisBranch(
                "partially_supported", "provider_coverage_partial"
            )
        return HypothesisBranch("partially_supported", "mixed")

    raise PaperContractError(f"unknown hypothesis: {hypothesis!r}")


def load_results_contract(path: Path | str) -> dict[str, Any]:
    """Parse only the single JSON fence inside the machine-owned contract markers."""

    contract_path = Path(path)
    try:
        text = contract_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PaperContractError(f"cannot read results contract: {contract_path}") from exc
    if text.count(CONTRACT_BEGIN) != 1 or text.count(CONTRACT_END) != 1:
        raise PaperContractError("results contract must contain one marker pair")
    start = text.index(CONTRACT_BEGIN) + len(CONTRACT_BEGIN)
    end = text.index(CONTRACT_END, start)
    match = re.fullmatch(
        r"\s*```json\s*\n(?P<payload>.*)\n```\s*",
        text[start:end],
        flags=re.DOTALL,
    )
    if match is None:
        raise PaperContractError("generated contract block must contain one JSON fence")
    try:
        payload = json.loads(match.group("payload"))
    except json.JSONDecodeError as exc:
        raise PaperContractError("generated contract JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise PaperContractError("generated contract JSON must be an object")
    return payload


def bind_papers(
    contract_path: Path | str,
    artifact_root: Path | str,
    main_path: Path | str,
    yau_path: Path | str,
    *,
    figure_output_dir: Path | str | None = None,
    check: bool = False,
) -> Mapping[str, str]:
    """Validate the contract and bind generated blocks into both manuscripts."""

    payload = load_results_contract(contract_path)
    validated = _validate_contract(payload, Path(artifact_root))
    status = _render_status(validated)
    abstract_en = _render_abstract_findings(validated, language="en")
    abstract_zh = _render_abstract_findings(validated, language="zh")
    discussion = _render_discussion(validated)
    paths = (Path(main_path), Path(yau_path))
    output_dir = (
        Path(figure_output_dir)
        if figure_output_dir is not None
        else paths[0].parent / "generated"
    )
    originals: dict[Path, str] = {}
    expected: dict[Path, str] = {}
    for index, path in enumerate(paths):
        try:
            original = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PaperContractError(f"cannot read manuscript: {path}") from exc
        observed_skeleton_sha = _manuscript_skeleton_sha256(
            original, include_zh=index == 1
        )
        _validate_manuscript_skeleton(
            path.name,
            observed_skeleton_sha,
            FROZEN_MANUSCRIPT_SKELETON_SHA256[index],
        )
        originals[path] = original
        results = (
            _render_main_results(validated, path, output_dir)
            if index == 0
            else _render_yau_results(validated, path, output_dir)
        )
        updated = _replace_generated_block(original, STATUS_BEGIN, STATUS_END, status)
        updated = _replace_generated_block(
            updated,
            ABSTRACT_EN_BEGIN,
            ABSTRACT_EN_END,
            abstract_en,
        )
        if index == 1:
            updated = _replace_generated_block(
                updated,
                ABSTRACT_ZH_BEGIN,
                ABSTRACT_ZH_END,
                abstract_zh,
            )
        updated = _replace_generated_block(updated, RESULTS_BEGIN, RESULTS_END, results)
        updated = _replace_generated_block(
            updated, DISCUSSION_BEGIN, DISCUSSION_END, discussion
        )
        expected[path] = updated

    manuscript_drift = [path for path in paths if originals[path] != expected[path]]
    figure_outputs = _expected_figure_outputs(validated, output_dir)
    stale_pngs = _stale_figure_outputs(figure_outputs, output_dir)
    figure_drift = _figure_output_drift(figure_outputs, stale_pngs)
    if check:
        if manuscript_drift:
            raise PaperDriftError(
                "generated paper drift: "
                + ", ".join(path.name for path in manuscript_drift)
            )
        if figure_drift:
            path, reason = figure_drift[0]
            raise PaperDriftError(f"generated figure {reason}: {path.name}")
        return {str(path): expected[path] for path in paths}

    if manuscript_drift or figure_drift:
        _replace_outputs_atomically(
            paths,
            originals,
            expected,
            figure_outputs,
            stale_pngs,
        )
    return {str(path): expected[path] for path in paths}


def _validate_contract(
    payload: Mapping[str, Any], artifact_root: Path
) -> ValidatedContract:
    status = payload.get("status")
    if status not in FINAL_STATUSES:
        raise PaperContractError(
            "final paper lifecycle status must be COMPLETE_H5_EVALUATED or "
            "COMPLETE_H5_EXCLUDED_PRE_OUTCOME"
        )
    metrics = payload.get("metrics")
    hypotheses = payload.get("hypotheses")
    decisions = payload.get("decisions")
    if not isinstance(metrics, Mapping):
        raise PaperContractError("contract metrics must be an object")
    if not isinstance(hypotheses, Mapping):
        raise PaperContractError("contract hypotheses must be an object")
    if not isinstance(decisions, Mapping):
        raise PaperContractError("contract decisions must be an object")
    registry_path = _verify_artifact_reference(
        "analysis artifact",
        payload.get("analysis_artifact"),
        payload.get("analysis_artifact_sha256"),
        artifact_root,
    )
    _validate_provenance_chain(payload, artifact_root, registry_path)
    decision_details = payload.get("decision_details")
    if decision_details is not None and decision_details != hypotheses:
        raise PaperContractError("decision_details alias differs from hypotheses")

    normalized_hypotheses: dict[str, Mapping[str, Any]] = {}
    branches: dict[str, HypothesisBranch | None] = {}
    for hypothesis in ("H1", "H2", "H3", "H4"):
        detail = _hypothesis_detail(hypotheses, hypothesis)
        if detail.get("analysis_status") != "complete":
            raise PaperContractError(
                f"{hypothesis} analysis_status must be complete in final mode"
            )
        branch = _validate_evaluated_hypothesis(hypothesis, detail)
        _require_decision_consistency(decisions, hypothesis, branch.decision)
        normalized_hypotheses[hypothesis] = detail
        branches[hypothesis] = branch

    h5 = _hypothesis_detail(hypotheses, "H5")
    h5_status = h5.get("analysis_status")
    if h5_status == "complete":
        branch = _validate_evaluated_hypothesis("H5", h5)
        _require_decision_consistency(decisions, "H5", branch.decision)
        h5_required = H5_IDS
        branches["H5"] = branch
    elif h5_status == "excluded_pre_outcome":
        if h5.get("decision") is not None or decisions.get("H5") is not None:
            raise PaperContractError(
                "H5 excluded_pre_outcome must have a null decision"
            )
        if h5.get("branch_reason") != "no_explicit_machine_annotation_map":
            raise PaperContractError("H5 exclusion reason is not canonical")
        predicate_inputs = h5.get("predicate_inputs")
        if not isinstance(predicate_inputs, Mapping):
            raise PaperContractError("H5 exclusion predicate_inputs must be an object")
        if any(metrics.get(result_id) is not None for result_id in H5_IDS):
            raise PaperContractError("H5 excluded_pre_outcome metrics must be null")
        if (
            predicate_inputs.get("evidence_path") != "h5/h5_results.json"
            or predicate_inputs.get("evidence_sha256")
            != payload.get("h5_results_file_sha256")
        ):
            raise PaperContractError(
                "H5 exclusion evidence path or hash is not canonical"
            )
        _verify_artifact_reference(
            "H5 exclusion evidence",
            predicate_inputs.get("evidence_path"),
            predicate_inputs.get("evidence_sha256"),
            artifact_root,
        )
        h5_required = ()
        branches["H5"] = None
    else:
        raise PaperContractError(
            f"H5 analysis_status {h5_status!r} is pending or invalid in final mode"
        )
    normalized_hypotheses["H5"] = h5
    expected_status = (
        "COMPLETE_H5_EVALUATED"
        if h5_status == "complete"
        else "COMPLETE_H5_EXCLUDED_PRE_OUTCOME"
    )
    if status != expected_status:
        raise PaperContractError(
            "final paper lifecycle status differs from the H5 analysis state"
        )

    required_ids = PROGRAMMATIC_IDS + h5_required
    canonical_metric_ids = set(PROGRAMMATIC_IDS + H5_IDS)
    missing_metric_ids = canonical_metric_ids - set(metrics)
    extra_metric_ids = set(metrics) - canonical_metric_ids
    if missing_metric_ids or extra_metric_ids:
        raise PaperContractError(
            "contract metric ID set is not canonical; missing="
            + ",".join(sorted(missing_metric_ids))
            + "; extra="
            + ",".join(sorted(extra_metric_ids))
        )
    for result_id in required_ids:
        if result_id not in metrics or metrics[result_id] is None:
            raise PaperContractError(f"required result ID is missing: {result_id}")
        expected_registry_id = _expected_registry_metric_id(result_id)
        if metrics[result_id].get("registry_metric_id") != expected_registry_id:
            raise PaperContractError(
                f"display record {result_id} differs from canonical registry mapping"
            )
        _validate_display_record(
            result_id,
            metrics[result_id],
            artifact_root,
            registry_artifact=str(payload["analysis_artifact"]),
            registry_sha256=str(payload["analysis_artifact_sha256"]),
        )
    results_document = _load_json_mapping(
        artifact_root / str(payload["results_artifact"]),
        "results provenance artifact",
    )
    _validate_registry_semantics(
        metrics,
        required_ids,
        registry_path,
        programmatic_raw_hash=str(payload["raw_hash"]),
        h5_raw_hash=(
            str(payload["h5_collection_manifest_sha256"])
            if h5_status == "complete"
            and payload.get("h5_collection_manifest_sha256") is not None
            else None
        ),
        programmatic_registry_ids=expected_programmatic_registry_ids(
            {
                9: int(payload["denominators"]["common_support_target_count_b9"]),
                15: int(payload["denominators"]["common_support_target_count_b15"]),
                25: int(payload["denominators"]["common_support_target_count_b25"]),
            }
        ),
        conditional_metric_audit=payload.get("conditional_metric_audit"),
    )
    _validate_item_type_diagnostics(metrics)
    _validate_reporting_disclosures(
        payload,
        results_document,
        artifact_root,
        source_manifest=_canonical_source_manifest(payload),
    )
    _validate_programmatic_replay(payload, artifact_root)
    _validate_h5_replay(payload, artifact_root)

    for hypothesis in ("H1", "H2", "H3", "H4"):
        _validate_predicates_against_display(
            hypothesis,
            normalized_hypotheses[hypothesis]["predicate_inputs"],
            metrics,
        )
    if h5_status == "complete":
        _validate_predicates_against_display(
            "H5", normalized_hypotheses["H5"]["predicate_inputs"], metrics
        )

    figures = payload.get("figures")
    if not isinstance(figures, Mapping):
        raise PaperContractError("contract figures must be an object")
    required_figures = REQUIRED_FIGURE_IDS
    references: dict[str, tuple[FigureReference, ...]] = {}
    for figure_id, value in figures.items():
        if value is None:
            if figure_id in required_figures:
                raise PaperContractError(f"required figure is missing: {figure_id}")
            continue
        figure_references = tuple(
            _validate_figure_tree(str(figure_id), value, artifact_root)
        )
        if figure_id in required_figures and not any(
            reference.is_png for reference in figure_references
        ):
            raise PaperContractError(f"{figure_id} must supply a verified PNG")
        references[str(figure_id)] = figure_references
    missing_figures = required_figures - references.keys()
    if missing_figures:
        raise PaperContractError(
            "required figures are missing: " + ", ".join(sorted(missing_figures))
        )

    return ValidatedContract(
        payload=payload,
        hypotheses=normalized_hypotheses,
        branches=branches,
        required_ids=required_ids,
        figure_references=references,
    )


def _hypothesis_detail(
    hypotheses: Mapping[str, Any], hypothesis: str
) -> Mapping[str, Any]:
    detail = hypotheses.get(hypothesis)
    if not isinstance(detail, Mapping):
        raise PaperContractError(f"{hypothesis} hypothesis detail is missing")
    return detail


def _validate_evaluated_hypothesis(
    hypothesis: str, detail: Mapping[str, Any]
) -> HypothesisBranch:
    _require_branch_reason(hypothesis, detail)
    predicate_inputs = detail.get("predicate_inputs")
    if not isinstance(predicate_inputs, Mapping):
        raise PaperContractError(f"{hypothesis} predicate_inputs must be an object")
    for key, expected in FROZEN_CONSTANTS.get(hypothesis, {}).items():
        actual = _number(predicate_inputs, key)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise PaperContractError(
                f"{hypothesis} frozen predicate constant {key} changed"
            )
    if hypothesis == "H5":
        rewrites = predicate_inputs.get("maximum_prompt_rewrites", 1)
        if _number({"value": rewrites}, "value") > 1:
            raise PaperContractError("H5 maximum_prompt_rewrites exceeds one")
    branch = derive_hypothesis_branch(hypothesis, predicate_inputs)
    if detail.get("decision") != branch.decision:
        raise PaperContractError(
            f"{hypothesis} decision differs from frozen predicate branch"
        )
    expected_reason = _expected_s3_branch_reason(hypothesis, predicate_inputs)
    if expected_reason is not None and detail.get("branch_reason") != expected_reason:
        raise PaperContractError(
            f"{hypothesis} branch_reason differs from recomputed S3 reason"
        )
    return branch


def _expected_s3_branch_reason(
    hypothesis: str, predicate_inputs: Mapping[str, object]
) -> str | None:
    if hypothesis == "H1":
        return h1_branch_reason(
            a_rate=_number(predicate_inputs, "a_rate"),
            rescue_point=_number(predicate_inputs, "rescue_point"),
            rescue_ci_low=_number(predicate_inputs, "rescue_ci_low"),
        )
    if hypothesis == "H2":
        return h2_branch_reason(
            _number(predicate_inputs, "harm_point"),
            _number(predicate_inputs, "harm_ci_low"),
            _number(predicate_inputs, "no_harm_point"),
            _number(predicate_inputs, "no_harm_ci_high"),
        )
    if hypothesis == "H3":
        return h3_branch_reason(
            _number(predicate_inputs, "accuracy_point"),
            _number(predicate_inputs, "accuracy_ci_low"),
            _number(predicate_inputs, "median_a"),
            _number(predicate_inputs, "median_b"),
        )
    if hypothesis == "H4":
        return h4_branch_reason(
            _number(predicate_inputs, "rescue_point"),
            _number(predicate_inputs, "harm_point"),
        )
    # S2 owns H5 reason vocabulary. The binder uses its recomputed branch for prose
    # and treats H5's free-form reason as audit metadata only.
    return None


def _require_branch_reason(hypothesis: str, detail: Mapping[str, Any]) -> None:
    reason = detail.get("branch_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise PaperContractError(f"{hypothesis} branch_reason is missing")


def _require_decision_consistency(
    decisions: Mapping[str, Any], hypothesis: str, expected: str
) -> None:
    if decisions.get(hypothesis) != expected:
        raise PaperContractError(
            f"top-level {hypothesis} decision differs from hypothesis detail"
        )


def _validate_provenance_chain(
    payload: Mapping[str, Any], artifact_root: Path, registry_path: Path
) -> None:
    if payload.get("analysis_timestamp_policy") != ANALYSIS_TIMESTAMP_POLICY:
        raise PaperContractError("analysis timestamp policy is not the frozen policy")
    for field in ("source_run_started_at_utc", "analysis_code_committed_at_utc"):
        _require_rfc3339_utc(payload.get(field), field)

    canonical = {
        "run_id": FROZEN_RUN_ID,
        "runner_commit": FROZEN_RUNNER_COMMIT,
        "experiment_tag": FROZEN_EXPERIMENT_TAG,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "analysis_plan_commit": FROZEN_ANALYSIS_PLAN_COMMIT,
        "analysis_plan_sha256": FROZEN_ANALYSIS_PLAN_SHA256,
    }
    for field, expected in canonical.items():
        if payload.get(field) != expected:
            raise PaperContractError(
                f"canonical provenance mismatch for {field}"
            )

    raw_hash = _require_nonzero_hex(payload.get("raw_hash"), 64, "raw_hash")
    analysis_commit = _require_nonzero_hex(
        payload.get("analysis_commit"), 40, "analysis_commit"
    )
    analysis_code_sha = _require_nonzero_hex(
        payload.get("analysis_code_sha256"), 64, "analysis_code_sha256"
    )
    files = payload.get("analysis_code_files")
    if not isinstance(files, Mapping) or not files:
        raise PaperContractError("analysis code provenance files are missing")
    normalized_files: dict[str, str] = {}
    for path, digest in files.items():
        if not isinstance(path, str) or not path:
            raise PaperContractError("analysis code provenance path is invalid")
        normalized_files[path] = _require_nonzero_hex(
            digest, 64, f"analysis_code_files.{path}"
        )
    if normalized_files.get("experiments/analysis_plan.md") != (
        FROZEN_ANALYSIS_PLAN_SHA256
    ):
        raise PaperContractError(
            "canonical provenance does not bind the frozen analysis-plan bytes"
        )

    source_provenance = {
        "run_id": FROZEN_RUN_ID,
        "run_started_at_utc": payload["source_run_started_at_utc"],
        "runner_commit": FROZEN_RUNNER_COMMIT,
        "experiment_tag": FROZEN_EXPERIMENT_TAG,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "analysis_plan_commit": FROZEN_ANALYSIS_PLAN_COMMIT,
        "analysis_plan_sha256": FROZEN_ANALYSIS_PLAN_SHA256,
    }
    analysis_provenance = {
        "analysis_commit": analysis_commit,
        "analysis_code_committed_at_utc": payload[
            "analysis_code_committed_at_utc"
        ],
        "analysis_code_sha256": analysis_code_sha,
        "analysis_code_files": normalized_files,
    }
    _validate_repository_provenance(source_provenance, analysis_provenance)
    results_path = _verify_artifact_reference(
        "results provenance artifact",
        payload.get("results_artifact"),
        payload.get("results_artifact_sha256"),
        artifact_root,
    )
    manifest_path = _verify_artifact_reference(
        "artifact manifest",
        payload.get("artifact_manifest"),
        payload.get("artifact_manifest_sha256"),
        artifact_root,
    )
    results = _load_json_mapping(results_path, "results provenance artifact")
    artifact_manifest = _load_json_mapping(manifest_path, "artifact manifest")
    core = {
        "raw_hash": raw_hash,
        "source_run_started_at_utc": payload["source_run_started_at_utc"],
        "analysis_code_committed_at_utc": payload[
            "analysis_code_committed_at_utc"
        ],
        "analysis_timestamp_policy": ANALYSIS_TIMESTAMP_POLICY,
    }
    for label, document in (
        ("results", results),
        ("artifact manifest", artifact_manifest),
    ):
        if any(document.get(key) != value for key, value in core.items()):
            raise PaperContractError(f"{label} provenance core differs from contract")
        if document.get("source_provenance") != source_provenance:
            raise PaperContractError(
                f"{label} source provenance differs from contract"
            )
        if document.get("analysis_provenance") != analysis_provenance:
            raise PaperContractError(
                f"{label} analysis provenance differs from contract"
            )

    registry_relative = str(payload["analysis_artifact"])
    registry_sha = str(payload["analysis_artifact_sha256"])
    if results.get("numeric_source") != registry_relative:
        raise PaperContractError("results numeric source differs from analysis registry")
    manifest_files = artifact_manifest.get("files")
    if not isinstance(manifest_files, Mapping):
        raise PaperContractError("artifact manifest file map is missing")
    if manifest_files.get(registry_relative) != registry_sha:
        raise PaperContractError("artifact manifest does not bind the analysis registry")
    if manifest_files.get(str(payload["results_artifact"])) != payload.get(
        "results_artifact_sha256"
    ):
        raise PaperContractError("artifact manifest does not bind results provenance")
    policy = payload.get("static_audit_policy")
    if not isinstance(policy, Mapping) or set(policy) != {"path", "sha256"}:
        raise PaperContractError("static audit policy reference is invalid")
    policy_path = _verify_artifact_reference(
        "static audit policy",
        policy.get("path"),
        policy.get("sha256"),
        artifact_root,
    )
    policy_value = _load_json_mapping(policy_path, "static audit policy")
    if (
        policy_value.get("basis")
        != "independent_static_review_before_result_interpretation"
        or policy_value.get("frozen_analysis_plan_modified") is not False
        or policy_value.get("result_direction_used") is not False
        or not isinstance(
            policy_value.get("conditional_metric_zero_denominator"), Mapping
        )
        or policy_value["conditional_metric_zero_denominator"].get(
            "redraw_until_defined"
        )
        is not False
    ):
        raise PaperContractError("static audit policy content is invalid")
    if manifest_files.get(str(policy["path"])) != policy["sha256"]:
        raise PaperContractError("artifact manifest does not bind static audit policy")
    audit = payload.get("conditional_metric_audit")
    if results.get("conditional_metric_audit") != audit:
        raise PaperContractError("results conditional metric audit differs from contract")
    if artifact_manifest.get("conditional_metric_audit") != audit:
        raise PaperContractError(
            "artifact manifest conditional metric audit differs from contract"
        )
    if results.get("static_audit_policy") != policy:
        raise PaperContractError("results static audit policy differs from contract")
    if artifact_manifest.get("static_audit_policy") != policy:
        raise PaperContractError(
            "artifact manifest static audit policy differs from contract"
        )
    if artifact_manifest.get("registry_metric_ids") != results.get(
        "registry_metric_ids"
    ):
        raise PaperContractError("registry ID inventory differs across artifacts")
    if registry_path != (artifact_root.resolve() / registry_relative).resolve():
        raise PaperContractError("analysis registry path differs across provenance")


def _validate_repository_provenance(
    source_provenance: Mapping[str, object],
    analysis_provenance: Mapping[str, object],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        verify_analysis_provenance(repo_root, dict(analysis_provenance))
    except DatasetContractError as exc:
        raise PaperContractError("analysis git provenance is invalid") from exc
    manifest = (
        repo_root
        / "data"
        / "sim_store"
        / "confirmatory"
        / str(source_provenance["run_id"])
        / "manifest.json"
    )
    try:
        payload = manifest.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperContractError("canonical source manifest is invalid") from exc
    if hashlib.sha256(payload).hexdigest() != source_provenance[
        "source_manifest_sha256"
    ]:
        raise PaperContractError("canonical source manifest hash differs from contract")
    if not isinstance(value, Mapping) or value.get("run_started_at_utc") != (
        source_provenance["run_started_at_utc"]
    ):
        raise PaperContractError("source run timestamp differs from canonical manifest")


def _canonical_source_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    path = (
        repo_root
        / "data"
        / "sim_store"
        / "confirmatory"
        / str(payload.get("run_id") or "")
        / "manifest.json"
    )
    try:
        source = path.read_bytes()
        value = json.loads(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperContractError("canonical source manifest is invalid") from exc
    if (
        hashlib.sha256(source).hexdigest() != payload.get("source_manifest_sha256")
        or not isinstance(value, Mapping)
    ):
        raise PaperContractError("canonical source manifest differs from contract")
    return value


def _validate_programmatic_replay(
    payload: Mapping[str, Any], artifact_root: Path
) -> None:
    from .runner import run_formal_analysis

    repo_root = Path(__file__).resolve().parents[1]
    manifest = (
        repo_root
        / "data"
        / "sim_store"
        / "confirmatory"
        / str(payload.get("run_id") or "")
        / "manifest.json"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="yher-paper-replay-") as temporary:
            replay_root = Path(temporary) / "programmatic"
            run_formal_analysis(
                manifest,
                replay_root,
                verified_analysis_provenance={
                    "analysis_commit": payload["analysis_commit"],
                    "analysis_code_committed_at_utc": payload[
                        "analysis_code_committed_at_utc"
                    ],
                    "analysis_code_sha256": payload["analysis_code_sha256"],
                    "analysis_code_files": payload["analysis_code_files"],
                },
            )
            _validate_replayed_programmatic_surface(
                payload,
                artifact_root,
                replay_root,
            )
    except PaperContractError:
        raise
    except Exception as exc:
        raise PaperContractError(
            "canonical programmatic replay failed closed"
        ) from exc


def _validate_replayed_programmatic_surface(
    payload: Mapping[str, Any],
    artifact_root: Path,
    replay_root: Path,
) -> None:
    replay_contract_path = replay_root / "results_contract_block.md"
    replay = load_results_contract(replay_contract_path)
    replay_files = tuple(
        sorted(
            path
            for path in replay_root.rglob("*")
            if path.is_file() and path != replay_contract_path
        )
    )
    for replay_path in replay_files:
        relative = replay_path.relative_to(replay_root)
        claimed_path = artifact_root / relative
        if not claimed_path.is_file() or claimed_path.read_bytes() != replay_path.read_bytes():
            raise PaperContractError(
                f"replayed programmatic artifact differs: {relative.as_posix()}"
            )

    scalar_fields = (
        "run_id",
        "runner_commit",
        "experiment_tag",
        "config_sha256",
        "source_manifest_sha256",
        "analysis_plan_commit",
        "analysis_plan_sha256",
        "raw_hash",
        "source_run_started_at_utc",
        "analysis_code_committed_at_utc",
        "analysis_timestamp_policy",
        "analysis_commit",
        "analysis_code_sha256",
        "analysis_code_files",
        "conditional_metric_audit",
        "static_audit_policy",
        "tables",
    )
    for field in scalar_fields:
        if payload.get(field) != replay.get(field):
            raise PaperContractError(
                f"replayed programmatic contract differs: {field}"
            )

    claimed_metrics = payload.get("metrics")
    replay_metrics = replay.get("metrics")
    if not isinstance(claimed_metrics, Mapping) or not isinstance(
        replay_metrics, Mapping
    ):
        raise PaperContractError("replayed programmatic metrics are missing")
    metric_fields = (
        "registry_metric_id",
        "value",
        "ci95",
        "numerator",
        "denominator",
        "weighting",
        "n_target",
        "n_pair",
        "raw_hash",
    )
    for result_id in PROGRAMMATIC_IDS:
        claimed = claimed_metrics.get(result_id)
        expected = replay_metrics.get(result_id)
        if not isinstance(claimed, Mapping) or not isinstance(expected, Mapping):
            raise PaperContractError(
                f"replayed programmatic display is missing: {result_id}"
            )
        if any(claimed.get(field) != expected.get(field) for field in metric_fields):
            raise PaperContractError(
                f"replayed programmatic display differs: {result_id}"
            )

    for field in ("decisions", "hypotheses", "decision_details"):
        claimed = payload.get(field)
        expected = replay.get(field)
        if not isinstance(claimed, Mapping) or not isinstance(expected, Mapping):
            raise PaperContractError(f"replayed programmatic {field} is missing")
        for hypothesis in ("H1", "H2", "H3", "H4"):
            if claimed.get(hypothesis) != expected.get(hypothesis):
                raise PaperContractError(
                    f"replayed programmatic {field} differs: {hypothesis}"
                )

    claimed_denominators = payload.get("denominators")
    replay_denominators = replay.get("denominators")
    if not isinstance(claimed_denominators, Mapping) or not isinstance(
        replay_denominators, Mapping
    ):
        raise PaperContractError("replayed programmatic denominators are missing")
    for field, value in replay_denominators.items():
        if field in {"excluded_provider_cells", "excluded_persona_cells"}:
            continue
        if claimed_denominators.get(field) != value:
            raise PaperContractError(
                f"replayed programmatic denominator differs: {field}"
            )

    claimed_figures = payload.get("figures")
    replay_figures = replay.get("figures")
    if not isinstance(claimed_figures, Mapping) or not isinstance(
        replay_figures, Mapping
    ):
        raise PaperContractError("replayed programmatic figures are missing")
    for figure_id in REQUIRED_FIGURE_IDS - {
        "FIG_PROVIDER_AGREEMENT",
        "FIG_MANIPULATION_CHECKS",
    }:
        if claimed_figures.get(figure_id) != replay_figures.get(figure_id):
            raise PaperContractError(
                f"replayed programmatic figure differs: {figure_id}"
            )

    replay_registry = _load_registry_rows(
        replay_root / str(replay["analysis_artifact"]),
        "replayed programmatic registry",
    )
    claimed_registry = _load_registry_rows(
        artifact_root / str(payload["analysis_artifact"]),
        "claimed analysis registry",
    )
    programmatic_ids = set(replay_registry)
    if {
        metric_id: claimed_registry.get(metric_id) for metric_id in programmatic_ids
    } != replay_registry:
        raise PaperContractError("replayed programmatic registry differs")


def _load_registry_rows(path: Path, label: str) -> dict[str, Mapping[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperContractError(f"{label} is invalid") from exc
    if not isinstance(rows, list):
        raise PaperContractError(f"{label} must be an array")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("metric_id"), str):
            raise PaperContractError(f"{label} row is invalid")
        metric_id = str(row["metric_id"])
        if metric_id in output:
            raise PaperContractError(f"{label} contains duplicate IDs")
        output[metric_id] = row
    return output


def _validate_h5_replay(payload: Mapping[str, Any], artifact_root: Path) -> None:
    from .h5 import analyze_collection, finalize_collection

    repo_root = Path(__file__).resolve().parents[1]
    raw_root = repo_root / "data/sim_store/llm_personas/llm-personas-v1"
    try:
        raw_root = raw_root.resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix="yher-h5-paper-replay-") as temporary:
            replay_root = Path(temporary) / "h5"
            collection = Path(temporary) / "h5_collection_manifest.json"
            finalize_collection(raw_root, collection, repo_root=repo_root)
            analyze_collection(
                collection,
                replay_root,
                raw_root=raw_root,
                repo_root=repo_root,
                verified_analysis_provenance={
                    "analysis_commit": payload["analysis_commit"],
                    "analysis_code_committed_at_utc": payload[
                        "analysis_code_committed_at_utc"
                    ],
                    "analysis_code_sha256": payload["analysis_code_sha256"],
                    "analysis_code_files": payload["analysis_code_files"],
                },
            )
            _validate_replayed_h5_surface(payload, artifact_root, replay_root)
    except PaperContractError:
        raise
    except Exception as exc:
        raise PaperContractError("canonical H5 replay failed closed") from exc


def _validate_replayed_h5_surface(
    payload: Mapping[str, Any],
    artifact_root: Path,
    replay_root: Path,
) -> None:
    replay_results = _load_json_mapping(
        replay_root / "h5_results.json", "replayed H5 results"
    )
    replay_manifest = _load_json_mapping(
        replay_root / "artifact_manifest.json", "replayed H5 artifact manifest"
    )
    h5_root = artifact_root / "h5"
    artifact_rows = replay_manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise PaperContractError("replayed H5 artifact inventory is missing")
    listed_paths = []
    for row in artifact_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise PaperContractError("replayed H5 artifact inventory row is invalid")
        listed_paths.append(str(row["path"]))
    copied_paths = tuple(
        sorted({"artifact_manifest.json", "h5_results.json", *listed_paths})
    )
    for relative in copied_paths:
        claimed = h5_root / relative
        replayed = replay_root / relative
        if not claimed.is_file() or claimed.read_bytes() != replayed.read_bytes():
            raise PaperContractError(f"replayed H5 artifact differs: h5/{relative}")

    expected_status = str(replay_results.get("status"))
    lifecycle = {
        "complete": "COMPLETE_H5_EVALUATED",
        "excluded_pre_outcome": "COMPLETE_H5_EXCLUDED_PRE_OUTCOME",
    }
    if expected_status not in lifecycle or payload.get("status") != lifecycle[expected_status]:
        raise PaperContractError("replayed H5 lifecycle differs")
    if payload.get("h5_results_file_sha256") != hashlib.sha256(
        (replay_root / "h5_results.json").read_bytes()
    ).hexdigest():
        raise PaperContractError("replayed H5 results file hash differs")
    for field, expected in (
        ("h5_results_sha256", replay_results.get("h5_results_sha256")),
        (
            "h5_artifact_manifest_sha256",
            replay_results.get("artifact_manifest_sha256"),
        ),
        (
            "h5_artifact_manifest_internal_sha256",
            replay_results.get("artifact_manifest_internal_sha256"),
        ),
        (
            "h5_analysis_code_sha256",
            (replay_results.get("provenance") or {})
            .get("analysis", {})
            .get("analysis_code_sha256"),
        ),
        (
            "h5_collection_manifest_sha256",
            (replay_results.get("provenance") or {})
            .get("s2", {})
            .get("collection_sha256"),
        ),
    ):
        if payload.get(field) != expected:
            raise PaperContractError(f"replayed H5 contract differs: {field}")
    h5_analysis = (replay_results.get("provenance") or {}).get("analysis")
    claimed_analysis_files = payload.get("analysis_code_files")
    if not isinstance(h5_analysis, Mapping) or not isinstance(
        claimed_analysis_files, Mapping
    ):
        raise PaperContractError("replayed H5 analysis provenance is missing")
    h5_files = h5_analysis.get("analysis_code_files")
    if (
        h5_analysis.get("analysis_commit") != payload.get("analysis_commit")
        or h5_analysis.get("analysis_code_sha256")
        != claimed_analysis_files.get("analysis/h5.py")
        or not isinstance(h5_files, list)
        or len(h5_files) != 1
        or h5_files[0].get("path") != "analysis/h5.py"
        or h5_files[0].get("sha256")
        != claimed_analysis_files.get("analysis/h5.py")
        or h5_files[0].get("matches_head") is not True
        or h5_files[0].get("head_sha256")
        != claimed_analysis_files.get("analysis/h5.py")
    ):
        raise PaperContractError("replayed H5 analysis provenance differs")
    if replay_manifest.get("collection_sha256") != payload.get(
        "h5_collection_manifest_sha256"
    ):
        raise PaperContractError("replayed H5 artifact collection hash differs")

    hypotheses = payload.get("hypotheses")
    decisions = payload.get("decisions")
    metrics = payload.get("metrics")
    figures = payload.get("figures")
    denominators = payload.get("denominators")
    if not all(
        isinstance(value, Mapping)
        for value in (hypotheses, decisions, metrics, figures, denominators)
    ):
        raise PaperContractError("replayed H5 contract surface is missing")
    expected_hypothesis = dict(replay_results["hypothesis"])
    if expected_status == "excluded_pre_outcome":
        expected_hypothesis["predicate_inputs"] = dict(
            expected_hypothesis["predicate_inputs"]
        )
        expected_hypothesis["predicate_inputs"]["evidence_path"] = (
            "h5/h5_results.json"
        )
        expected_hypothesis["predicate_inputs"]["evidence_sha256"] = hashlib.sha256(
            (replay_root / "h5_results.json").read_bytes()
        ).hexdigest()
    if hypotheses.get("H5") != expected_hypothesis:
        raise PaperContractError("replayed H5 hypothesis differs")
    if decisions.get("H5") != expected_hypothesis.get("decision"):
        raise PaperContractError("replayed H5 decision differs")

    expected_metrics = replay_results.get("metrics")
    if not isinstance(expected_metrics, Mapping):
        raise PaperContractError("replayed H5 metrics are missing")
    if expected_status == "complete":
        registry = _load_registry_rows(
            replay_root / "h5_metric_registry.json", "replayed H5 registry"
        )
        expected_ids = {
            "h5.qualifying_provider_count",
            "h5.minimum_completed_personas_per_qualifying_cell",
            "h5.weak_accuracy_gate",
            "h5.strong_accuracy_gate",
            "h5.misconception_hit_rate_contrast",
        }
        if set(registry) != expected_ids:
            raise PaperContractError("replayed H5 registry ID set differs")
        claimed_registry_path = _verify_artifact_reference(
            "claimed merged registry",
            payload.get("analysis_artifact"),
            payload.get("analysis_artifact_sha256"),
            artifact_root,
        )
        claimed_registry = _load_registry_rows(
            claimed_registry_path, "claimed merged registry"
        )
        claimed_h5_ids = {
            metric_id for metric_id in claimed_registry if metric_id.startswith("h5.")
        }
        if claimed_h5_ids != expected_ids or {
            metric_id: claimed_registry[metric_id] for metric_id in expected_ids
        } != registry:
            raise PaperContractError("replayed H5 registry differs")
        metric_fields = (
            "registry_metric_id",
            "value",
            "ci95",
            "numerator",
            "denominator",
            "weighting",
            "n_target",
            "n_pair",
        )
        for result_id in H5_IDS:
            claimed = metrics.get(result_id)
            expected = expected_metrics.get(result_id)
            if not isinstance(claimed, Mapping) or not isinstance(expected, Mapping):
                raise PaperContractError(f"replayed H5 display is missing: {result_id}")
            if any(claimed.get(field) != expected.get(field) for field in metric_fields):
                raise PaperContractError(f"replayed H5 display differs: {result_id}")
            if claimed.get("raw_hash") != payload.get(
                "h5_collection_manifest_sha256"
            ):
                raise PaperContractError(f"replayed H5 raw hash differs: {result_id}")
    elif any(metrics.get(result_id) is not None for result_id in H5_IDS):
        raise PaperContractError("replayed excluded H5 metrics must be null")

    replay_figures = replay_results.get("figures")
    if not isinstance(replay_figures, Mapping):
        raise PaperContractError("replayed H5 figures are missing")
    for figure_id in ("FIG_PROVIDER_AGREEMENT", "FIG_MANIPULATION_CHECKS"):
        expected = _prefix_h5_paths(replay_figures.get(figure_id))
        if figures.get(figure_id) != expected:
            raise PaperContractError(f"replayed H5 figure differs: {figure_id}")
    replay_denominators = replay_results.get("denominators")
    if not isinstance(replay_denominators, Mapping):
        raise PaperContractError("replayed H5 denominators are missing")
    for field in (
        "excluded_provider_cells",
        "excluded_persona_cells",
        *H5_LIFECYCLE_DENOMINATOR_FIELDS,
    ):
        if denominators.get(field) != replay_denominators.get(field):
            raise PaperContractError(f"replayed H5 denominator differs: {field}")
    if payload.get("h5_provider_exclusion_disclosure") != replay_results.get(
        "provider_exclusion_disclosure"
    ):
        raise PaperContractError("replayed H5 provider disclosure differs")


def _prefix_h5_paths(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): (
                f"h5/{child}"
                if key == "path" and isinstance(child, str)
                else _prefix_h5_paths(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_prefix_h5_paths(child) for child in value]
    return value


def _expected_registry_metric_id(result_id: str) -> str:
    if result_id in PROGRAMMATIC_IDS:
        return _contract_metric_specs()[result_id][0]
    h5_mapping = {
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
    try:
        return h5_mapping[result_id]
    except KeyError as exc:
        raise PaperContractError(f"unknown display result ID: {result_id}") from exc


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperContractError(f"{label} JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise PaperContractError(f"{label} JSON must be an object")
    return value


def _require_rfc3339_utc(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ) is None:
        raise PaperContractError(
            f"{field} timestamp must be exact RFC3339 UTC seconds"
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PaperContractError(f"{field} timestamp is invalid") from exc
    return value


def _require_nonzero_hex(value: object, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
        or value == "0" * length
    ):
        raise PaperContractError(f"provenance {field} is invalid or placeholder")
    return value


def _validate_display_record(
    result_id: str,
    value: object,
    artifact_root: Path,
    *,
    registry_artifact: str,
    registry_sha256: str,
) -> None:
    if not isinstance(value, Mapping):
        raise PaperContractError(f"display record is invalid: {result_id}")
    required = {
        "registry_metric_id",
        "value",
        "numerator",
        "denominator",
        "weighting",
        "n_target",
        "n_pair",
        "raw_hash",
        "artifact",
        "artifact_sha256",
    }
    missing = required - value.keys()
    if missing:
        raise PaperContractError(
            f"display record {result_id} is missing: {', '.join(sorted(missing))}"
        )
    if not isinstance(value["registry_metric_id"], str) or not value[
        "registry_metric_id"
    ]:
        raise PaperContractError(f"display record {result_id} lacks registry ID")
    _validate_display_value(result_id, value["value"])
    ci95 = value.get("ci95")
    if ci95 is not None:
        _interval(ci95, result_id)
    for field in ("numerator", "denominator", "n_target", "n_pair"):
        _finite_number(value[field], f"{result_id}.{field}")
    if not _is_sha256(value["raw_hash"]):
        raise PaperContractError(f"display record {result_id} raw_hash is invalid")
    if (
        value["artifact"] != registry_artifact
        or value["artifact_sha256"] != registry_sha256
    ):
        raise PaperContractError(
            f"display record {result_id} does not use the single analysis registry"
        )


def _validate_registry_semantics(
    metrics: Mapping[str, Any],
    required_ids: Sequence[str],
    registry_path: Path,
    *,
    programmatic_raw_hash: str,
    h5_raw_hash: str | None,
    programmatic_registry_ids: object,
    conditional_metric_audit: object,
) -> None:
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperContractError("metric registry JSON is invalid") from exc
    if not isinstance(payload, list):
        raise PaperContractError("metric registry JSON must be an array")
    registry: dict[str, Mapping[str, Any]] = {}
    for value in payload:
        if not isinstance(value, Mapping):
            raise PaperContractError("metric registry row is invalid")
        metric_id = value.get("metric_id")
        if not isinstance(metric_id, str) or not metric_id:
            raise PaperContractError("metric registry row lacks metric_id")
        if metric_id in registry:
            raise PaperContractError(f"duplicate metric registry ID: {metric_id}")
        registry[metric_id] = value

    if (
        not isinstance(programmatic_registry_ids, (list, tuple, set, frozenset))
        or len(programmatic_registry_ids) != len(set(programmatic_registry_ids))
        or any(not isinstance(metric_id, str) for metric_id in programmatic_registry_ids)
    ):
        raise PaperContractError("programmatic registry ID inventory is invalid")
    h5_registry_ids = {
        str(metrics[result_id]["registry_metric_id"])
        for result_id in H5_IDS
        if isinstance(metrics.get(result_id), Mapping)
    }
    expected_registry_ids = set(programmatic_registry_ids) | h5_registry_ids
    if set(registry) != expected_registry_ids:
        raise PaperContractError("metric registry contains missing or extra IDs")
    for metric_id, row in registry.items():
        expected_hash = (
            h5_raw_hash if metric_id in h5_registry_ids else programmatic_raw_hash
        )
        if not expected_hash or row.get("raw_hash") != expected_hash:
            raise PaperContractError(
                f"registry raw_hash classification mismatch: {metric_id}"
            )

    for result_id in required_ids:
        display = metrics[result_id]
        registry_id = str(display["registry_metric_id"])
        try:
            row = registry[registry_id]
        except KeyError as exc:
            raise PaperContractError(
                f"registry metric is missing for {result_id}: {registry_id}"
            ) from exc
        if isinstance(display["value"], list):
            _require_registry_interval_equal(
                result_id,
                display["value"],
                row,
                field="value",
            )
        else:
            _require_registry_value_equal(
                result_id,
                "value",
                display["value"],
                row.get("value"),
            )
        if display.get("ci95") is not None:
            _require_registry_interval_equal(
                result_id,
                display["ci95"],
                row,
                field="ci95",
            )
        for field in ("numerator", "denominator", "n_target", "n_pair"):
            _require_registry_value_equal(
                result_id,
                field,
                display[field],
                row.get(field),
            )
        if display["weighting"] != row.get("weighting"):
            raise PaperContractError(
                f"registry semantic mismatch for {result_id}: weighting"
            )
        expected_raw_hash = (
            h5_raw_hash
            if result_id in H5_IDS and h5_raw_hash is not None
            else programmatic_raw_hash
        )
        if (
            display["raw_hash"] != row.get("raw_hash")
            or display["raw_hash"] != expected_raw_hash
        ):
            raise PaperContractError(
                f"registry semantic mismatch for {result_id}: raw_hash"
            )

    if not isinstance(conditional_metric_audit, Mapping):
        raise PaperContractError("conditional metric audit must be an object")
    for metric_id, audit in conditional_metric_audit.items():
        if not isinstance(metric_id, str) or metric_id not in registry:
            raise PaperContractError(
                f"conditional metric audit registry row is missing: {metric_id}"
            )
        if not isinstance(audit, Mapping):
            raise PaperContractError(
                f"conditional metric audit is invalid: {metric_id}"
            )
        attempted = audit.get("attempted_iterations")
        defined = audit.get("defined_iterations")
        all_undefined = audit.get("all_targets_undefined_iterations")
        if (
            not isinstance(attempted, int)
            or attempted <= 0
            or not isinstance(defined, int)
            or not 0 < defined <= attempted
            or not isinstance(all_undefined, int)
            or all_undefined != attempted - defined
            or audit.get("redraw_count") != 0
        ):
            raise PaperContractError(
                f"conditional metric audit iteration accounting is invalid: {metric_id}"
            )
        weighting = str(registry[metric_id].get("weighting", ""))
        if (
            "zero_denominator_policy=record_NA_no_redraw" not in weighting
            or f"bootstrap_attempted_iterations={attempted}" not in weighting
            or f"bootstrap_defined_iterations={defined}" not in weighting
        ):
            raise PaperContractError(
                f"conditional metric audit differs from registry weighting: {metric_id}"
            )


def _validate_reporting_disclosures(
    payload: Mapping[str, Any],
    results: Mapping[str, Any],
    artifact_root: Path,
    *,
    source_manifest: Mapping[str, Any],
) -> None:
    denominators = payload.get("denominators")
    if not isinstance(denominators, Mapping):
        raise PaperContractError("contract denominators must be an object")
    count = denominators.get("estimand_excluded_journey_count")
    reasons = denominators.get("estimand_exclusion_reasons")
    arms = denominators.get("estimand_exclusion_arms")
    targets = denominators.get("estimand_exclusion_targets")
    if not isinstance(count, int) or count < 0:
        raise PaperContractError("estimand exclusion count is invalid")
    for label, values in (("reason", reasons), ("arm", arms)):
        if (
            not isinstance(values, Mapping)
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, int)
                or value <= 0
                for key, value in values.items()
            )
            or sum(values.values()) != count
        ):
            raise PaperContractError(
                f"estimand exclusion {label} accounting is invalid"
            )
    if (
        not isinstance(targets, list)
        or len(targets) != len(set(targets))
        or any(not isinstance(target, str) or not target for target in targets)
        or (count > 0 and not targets)
    ):
        raise PaperContractError("estimand exclusion target disclosure is invalid")
    validation = results.get("validation")
    if not isinstance(validation, Mapping):
        raise PaperContractError("results validation denominators are missing")
    fields = (
        "full_target_count",
        "h1_h2_eligible_target_count",
        "common_support_target_count_b9",
        "common_support_target_count_b15",
        "common_support_target_count_b25",
        "valid_journey_count",
        "structural_failure_count",
        "schema_invalid_count",
        "schema_invalid_reasons",
        "intended_journey_count",
        "estimand_excluded_journey_count",
        "estimand_exclusion_reasons",
        "estimand_exclusion_arms",
        "estimand_exclusion_targets",
    )
    for field in fields:
        if denominators.get(field) != validation.get(field):
            raise PaperContractError(
                f"contract denominator differs from results validation: {field}"
            )
    source_validation = source_manifest.get("validation")
    if not isinstance(source_validation, Mapping):
        raise PaperContractError("canonical source manifest validation is missing")
    source_denominators = {
        "full_target_count": source_validation.get("open_nodes"),
        "h1_h2_eligible_target_count": source_validation.get("h1_h2_eligible"),
        "common_support_target_count_b9": (
            source_validation.get("common_support_targets") or {}
        ).get("9"),
        "common_support_target_count_b15": (
            source_validation.get("common_support_targets") or {}
        ).get("15"),
        "common_support_target_count_b25": (
            source_validation.get("common_support_targets") or {}
        ).get("25"),
        "intended_journey_count": source_manifest.get("expected_journey_count"),
    }
    for field, expected in source_denominators.items():
        if denominators.get(field) != expected:
            raise PaperContractError(
                f"source manifest denominator differs from contract: {field}"
            )
    h5_results_path = _verify_artifact_reference(
        "H5 results",
        "h5/h5_results.json",
        payload.get("h5_results_file_sha256"),
        artifact_root,
    )
    h5_results = _load_json_mapping(h5_results_path, "H5 results")
    h5_denominators = h5_results.get("denominators")
    if not isinstance(h5_denominators, Mapping):
        raise PaperContractError("H5 results denominators are missing")
    for field in (
        "excluded_provider_cells",
        "excluded_persona_cells",
        *H5_LIFECYCLE_DENOMINATOR_FIELDS,
    ):
        if denominators.get(field) != h5_denominators.get(field):
            raise PaperContractError(
                f"contract H5 denominator differs from H5 results: {field}"
            )
    if payload.get("h5_provider_exclusion_disclosure") != h5_results.get(
        "provider_exclusion_disclosure"
    ):
        raise PaperContractError(
            "contract H5 provider disclosure differs from H5 results"
        )


def _validate_item_type_diagnostics(metrics: Mapping[str, Any]) -> None:
    required_weighting = (
        "target_stratified_paired_replicate_resample",
        "journey_cluster_preserved",
        "bootstrap_iterations=10000",
        "diagnostic_only_not_item_type_H1_H2_estimand",
    )
    for result_id in ITEM_TYPE_DIAGNOSTIC_IDS:
        record = metrics.get(result_id)
        if not isinstance(record, Mapping):
            raise PaperContractError(f"item-type diagnostic is missing: {result_id}")
        weighting = str(record.get("weighting", ""))
        if any(token not in weighting for token in required_weighting):
            raise PaperContractError(
                f"item-type diagnostic weighting is invalid: {result_id}"
            )
        _interval(record.get("ci95"), result_id)
        event_count = _finite_number(record.get("denominator"), result_id)
        journey_count = _finite_number(record.get("n_pair"), result_id)
        target_count = _finite_number(record.get("n_target"), result_id)
        if not event_count >= journey_count >= target_count > 0:
            raise PaperContractError(
                f"item-type diagnostic counts are invalid: {result_id}"
            )


def _require_registry_interval_equal(
    result_id: str,
    expected: object,
    row: Mapping[str, Any],
    *,
    field: str,
) -> None:
    low, high = _interval(expected, f"{result_id}.{field}")
    for label, expected_value, key in (
        ("ci_low", low, "ci_low"),
        ("ci_high", high, "ci_high"),
    ):
        _require_registry_value_equal(
            result_id,
            label,
            expected_value,
            row.get(key),
        )


def _require_registry_value_equal(
    result_id: str,
    field: str,
    expected: object,
    actual: object,
) -> None:
    if isinstance(expected, bool):
        matches = isinstance(actual, bool) and actual is expected
    else:
        try:
            matches = math.isclose(
                _finite_number(expected, f"{result_id}.{field}"),
                _finite_number(actual, f"registry.{field}"),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except PaperContractError:
            matches = False
    if not matches:
        raise PaperContractError(
            f"registry semantic mismatch for {result_id}: {field}"
        )


def _validate_display_value(result_id: str, value: object) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, list):
        _interval(value, result_id)
        return
    _finite_number(value, f"{result_id}.value")


def _validate_predicates_against_display(
    hypothesis: str,
    predicate_inputs: object,
    metrics: Mapping[str, Any],
) -> None:
    if not isinstance(predicate_inputs, Mapping):
        raise PaperContractError(f"{hypothesis} predicate_inputs must be an object")
    for predicate, (result_id, selector) in PREDICATE_RESULT_BINDINGS[
        hypothesis
    ].items():
        expected = _display_selected_value(metrics[result_id], selector, result_id)
        actual = predicate_inputs.get(predicate)
        if isinstance(expected, bool):
            matches = isinstance(actual, bool) and actual is expected
        else:
            try:
                matches = math.isclose(
                    _finite_number(actual, f"{hypothesis}.{predicate}"),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            except PaperContractError:
                matches = False
        if not matches:
            raise PaperContractError(
                f"{hypothesis} predicate {predicate} differs from display record {result_id}"
            )


def _display_selected_value(
    record: object, selector: str, result_id: str
) -> float | bool:
    if not isinstance(record, Mapping):
        raise PaperContractError(f"display record is invalid: {result_id}")
    if selector == "value":
        value = record.get("value")
        if isinstance(value, bool):
            return value
        return _finite_number(value, f"{result_id}.value")
    low, high = _interval(record.get("value"), result_id)
    return low if selector == "ci_low" else high


def _validate_figure_tree(
    figure_id: str,
    value: object,
    artifact_root: Path,
    key_path: tuple[str, ...] = (),
) -> list[FigureReference]:
    if not isinstance(value, Mapping):
        raise PaperContractError(f"{figure_id} figure record is invalid")
    if "path" in value or "sha256" in value:
        if set(value) != {"path", "sha256"}:
            raise PaperContractError(f"{figure_id} figure reference is incomplete")
        path = value["path"]
        sha256 = value["sha256"]
        resolved = _verify_artifact_reference(figure_id, path, sha256, artifact_root)
        if resolved.suffix.lower() == ".png" and not resolved.read_bytes().startswith(
            PNG_MAGIC
        ):
            raise PaperContractError(f"{figure_id} PNG signature is invalid: {path}")
        return [
            FigureReference(
                figure_id=figure_id,
                key_path=key_path,
                source_relative_path=str(path),
                source_path=resolved,
                sha256=str(sha256),
            )
        ]
    references: list[FigureReference] = []
    for key, child in value.items():
        references.extend(
            _validate_figure_tree(
                figure_id,
                child,
                artifact_root,
                key_path + (str(key),),
            )
        )
    if not references:
        raise PaperContractError(f"{figure_id} has no artifact references")
    return references


def _verify_artifact_reference(
    label: str,
    path_value: object,
    sha_value: object,
    artifact_root: Path,
) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise PaperContractError(f"{label} artifact path is invalid")
    if not _is_sha256(sha_value):
        raise PaperContractError(f"{label} artifact hash is invalid")
    relative = Path(path_value)
    if relative.is_absolute():
        raise PaperContractError(f"{label} artifact path is outside artifact root")
    root = artifact_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PaperContractError(
            f"{label} artifact path is outside artifact root"
        ) from exc
    if not resolved.is_file():
        raise PaperContractError(f"{label} artifact does not exist: {path_value}")
    observed = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if observed != sha_value:
        raise PaperContractError(f"{label} artifact hash mismatch: {path_value}")
    return resolved


def _render_status(validated: ValidatedContract) -> str:
    h5_status = validated.hypotheses["H5"]["analysis_status"]
    if h5_status == "excluded_pre_outcome":
        h5_clause = "H5 was excluded pre-outcome with machine-bound evidence"
    else:
        h5_clause = "H5 was evaluated under its frozen gates"
    return (
        "**Manuscript status:** confirmatory result binding complete; "
        f"H1-H4 were evaluated and {h5_clause}. All result values and decisions in "
        "the generated sections come from the validated machine contract."
    )


def _render_abstract_findings(
    validated: ValidatedContract,
    *,
    language: str,
) -> str:
    metrics = validated.payload["metrics"]
    decisions = []
    for hypothesis in ("H1", "H2", "H3", "H4", "H5"):
        analysis_status, decision, _ = _decision_cells(validated, hypothesis)
        decisions.append(f"{hypothesis}={decision} ({analysis_status})")
    a_rate = _format_value(metrics[H1_IDS[0]]["value"])
    rescue_ci = _format_value(metrics[H1_IDS[3]]["value"])
    harm_id = "H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95"
    no_harm_id = "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95"
    harm_ci = _format_value(metrics[harm_id]["value"])
    no_harm_ci = _format_value(metrics[no_harm_id]["value"])
    decision_text = "; ".join(decisions)
    if language == "en":
        return (
            f"**Confirmatory findings.** {decision_text}. H1 A rate={a_rate}; "
            f"rescue 95% CI={rescue_ci}. H2 harm 95% CI={harm_ci}; "
            f"no-harm 95% CI={no_harm_ci}. These are simulated finite-budget "
            "results, not evidence of human learning or educational efficacy."
        )
    if language == "zh":
        return (
            f"**确证结果。** {decision_text}。H1 A 臂收敛率={a_rate}；"
            f"救活效应 95% CI={rescue_ci}。H2 伤害效应 95% CI={harm_ci}；"
            f"无伤害对照 95% CI={no_harm_ci}。上述结果仅限于有限题量"
            "仿真，不证明真人学习效果或教育效能。"
        )
    raise PaperContractError(f"unsupported abstract findings language: {language}")


def _render_main_results(
    validated: ValidatedContract,
    manuscript_path: Path,
    figure_output_dir: Path,
) -> str:
    metrics = validated.payload["metrics"]
    lines = [
        "The table and audit records below are generated directly from the validated "
        "results contract. No confirmatory value in this section is hand-entered.",
        "",
        *_render_integrity_disclosure(validated),
        "",
        "| Hypothesis | Analysis status | Decision | Machine branch |",
        "|---|---|---|---|",
    ]
    for hypothesis in ("H1", "H2", "H3", "H4", "H5"):
        analysis_status, decision, machine_branch = _decision_cells(
            validated, hypothesis
        )
        lines.append(
            f"| {hypothesis} | {analysis_status} | {decision} | `{machine_branch}` |"
        )

    grouped = {
        "H1": H1_IDS,
        "H2": H2_IDS,
        "H3": H3_IDS,
        "H4": H4_IDS,
        "H5": H5_IDS if validated.branches["H5"] is not None else (),
    }
    lines.extend(["", "### Machine display records", ""])
    for hypothesis, result_ids in grouped.items():
        lines.append(f"**{hypothesis}.**")
        if not result_ids:
            lines.append(
                "No outcome estimate was generated because the analysis was excluded "
                "pre-outcome; the decision table records this as not evaluated."
            )
        for result_id in result_ids:
            lines.append(_format_display_record(result_id, metrics[result_id]))
        lines.append("")

    lines.extend([_source_artifact_sentence(validated), ""])
    lines.extend(["### Verified publication figures", ""])
    lines.extend(
        _render_figure_links(validated, manuscript_path, figure_output_dir)
    )
    return "\n".join(lines).rstrip()


def _render_yau_results(
    validated: ValidatedContract,
    manuscript_path: Path,
    figure_output_dir: Path,
) -> str:
    metrics = validated.payload["metrics"]
    compact_ids = YAU_PROGRAMMATIC_IDS + (
        H5_IDS if validated.branches["H5"] is not None else ()
    )
    h1_n = int(metrics[H1_IDS[0]]["denominator"])
    h2_n = int(
        metrics["H2_C_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS"][
            "denominator"
        ]
    )
    h5_evidence = (
        "excluded pre-outcome; 0 qualifying providers; no outcome decision"
        if validated.branches["H5"] is None
        else (
            f"qualifying providers={_format_value(metrics[H5_IDS[0]]['value'])}; "
            f"minimum completed per cell="
            f"{_format_value(metrics[H5_IDS[1]]['value'])}"
        )
    )
    visible_rows = (
        (
            "H1: A P convergence",
            f"A={_format_percent(metrics[H1_IDS[0]]['value'])}; "
            f"B={_format_percent(metrics[H1_IDS[1]]['value'])}; "
            f"A-B={_format_pp(metrics[H1_IDS[2]]['value'])} "
            f"(95% CI {_format_pp_interval(metrics[H1_IDS[3]]['value'])}); "
            f"no-repeat A-B={_format_pp(metrics[H1_IDS[6]]['value'])} "
            f"({_format_pp_interval(metrics[H1_IDS[7]]['value'])}); n={h1_n:,}/arm",
        ),
        (
            "H2: C-state misdiagnosis",
            "A="
            f"{_format_percent(metrics['H2_C_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS']['value'])}; "
            "B="
            f"{_format_percent(metrics['H2_C_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS']['value'])}; "
            "C="
            f"{_format_percent(metrics['H2_C_C_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS']['value'])}; "
            "C-A="
            f"{_format_pp(metrics['H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS']['value'])} "
            "(95% CI "
            f"{_format_pp_interval(metrics['H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95']['value'])}); "
            "A-B="
            f"{_format_pp(metrics['H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS']['value'])} "
            "(95% CI "
            f"{_format_pp_interval(metrics['H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95']['value'])}); "
            "no-repeat C-A="
            f"{_format_pp(metrics['H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT']['value'])} "
            "(95% CI "
            f"{_format_pp_interval(metrics['H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95']['value'])}); "
            "A-B="
            f"{_format_pp(metrics['H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT']['value'])} "
            "(95% CI "
            f"{_format_pp_interval(metrics['H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95']['value'])}); "
            f"n={h2_n:,}/arm",
        ),
        (
            "H3: A-B terminal accuracy",
            f"{_format_pp(metrics[H3_IDS[0]]['value'])} "
            f"(95% CI {_format_pp_interval(metrics[H3_IDS[1]]['value'])}); "
            f"median convergence A={_format_value(metrics[H3_IDS[2]]['value'])}, "
            f"B={_format_value(metrics[H3_IDS[3]]['value'])} items",
        ),
        (
            "H4: misspecified rescue and harm",
            f"rescue={_format_pp(metrics[H4_IDS[0]]['value'])} "
            f"(95% CI {_format_pp_interval(metrics[H4_IDS[0]]['ci95'])}); "
            f"fixed-probe harm={_format_pp(metrics[H4_IDS[1]]['value'])} "
            f"(95% CI {_format_pp_interval(metrics[H4_IDS[1]]['ci95'])})",
        ),
        ("H5: LLM personas", h5_evidence),
    )
    lines = [
        "This compact table is generated from the same validated contract as the main "
        "paper. It reports only decision-driving primary evidence and the mandatory "
        "stress/common-support sensitivity evidence.",
        "",
        _render_yau_integrity_disclosure(validated),
        "",
        "| Claim | Decision | Compact machine evidence |",
        "|---|---|---|",
    ]
    for hypothesis, (claim, evidence) in zip(
        ("H1", "H2", "H3", "H4", "H5"), visible_rows, strict=True
    ):
        analysis_status, decision, _ = _decision_cells(validated, hypothesis)
        lines.append(f"| {claim} ({analysis_status}) | {decision} | {evidence} |")
    lines.extend(["", "<!-- BEGIN YAU MACHINE AUDIT -->"])
    for result_id in compact_ids:
        lines.append(_format_compact_display_record(result_id, metrics[result_id]))
    lines.append("<!-- END YAU MACHINE AUDIT -->")
    lines.extend(["", _source_artifact_sentence(validated), ""])
    lines.extend(["### Selected verified figures", ""])
    lines.extend(
        _render_figure_links(
            validated,
            manuscript_path,
            figure_output_dir,
            figure_ids=YAU_FIGURE_IDS,
        )
    )
    return "\n".join(lines).rstrip()


def _render_yau_integrity_disclosure(validated: ValidatedContract) -> str:
    denominators = validated.payload["denominators"]
    provider_exclusions = validated.payload.get("h5_provider_exclusion_disclosure")
    provider_parts = []
    if isinstance(provider_exclusions, Mapping):
        for category, providers in sorted(provider_exclusions.items()):
            if not isinstance(providers, list) or any(
                not isinstance(provider, str) or not provider for provider in providers
            ):
                raise PaperContractError("H5 provider exclusion disclosure is invalid")
            provider_parts.append(
                f"{str(category).replace('_', ' ')}="
                + "/".join(str(provider) for provider in providers)
            )
    h5_clause = (
        "H5 excluded pre-outcome with 0 qualifying providers"
        if validated.branches["H5"] is None
        else "H5 evaluated under its frozen gates"
    )
    provider_clause = "; ".join(provider_parts) or "no provider exclusions"
    return (
        f"**Machine integrity summary.** {int(denominators['valid_journey_count']):,} "
        f"valid of {int(denominators['intended_journey_count']):,} intended "
        f"programmatic journeys; {int(denominators['estimand_excluded_journey_count']):,} "
        f"predeclared estimand exclusions; {h5_clause}. Provider lifecycle: "
        f"{provider_clause}."
    )


def _render_integrity_disclosure(validated: ValidatedContract) -> list[str]:
    denominators = validated.payload["denominators"]
    count = int(denominators["estimand_excluded_journey_count"])
    reasons = ", ".join(
        f"`{reason}` ({int(reason_count):,})"
        for reason, reason_count in sorted(
            denominators["estimand_exclusion_reasons"].items()
        )
    )
    arms = ", ".join(
        f"Arm {arm} ({int(arm_count):,})"
        for arm, arm_count in sorted(denominators["estimand_exclusion_arms"].items())
    )
    targets = ", ".join(
        f"`{target}`" for target in denominators["estimand_exclusion_targets"]
    )
    exclusion = (
        f"**Machine exclusion disclosure.** {count:,} intended journeys were excluded "
        f"from every estimand. Reasons: {reasons or 'none'}. Arms: {arms or 'none'}. "
        f"Affected targets: {targets or 'none'}."
    )
    policy = validated.payload["static_audit_policy"]
    audit = (
        "**Machine post-collection static audit policy.** The independently reviewed "
        "conditional-metric rule records undefined target/draw denominators as NA, "
        "uses no denominator redraw, and discloses attempted and defined bootstrap "
        "iterations. This clarification was adopted from static review, not from any "
        f"result direction; the binder verified policy artifact `{policy['path']}`."
    )
    provider_identity_disclosure = None
    provider_exclusions = validated.payload.get("h5_provider_exclusion_disclosure")
    if isinstance(provider_exclusions, Mapping):
        identity_parts = []
        for category, providers in sorted(provider_exclusions.items()):
            if not isinstance(providers, list) or any(
                not isinstance(provider, str) or not provider for provider in providers
            ):
                raise PaperContractError("H5 provider exclusion disclosure is invalid")
            names = ", ".join(f"`{provider}`" for provider in providers)
            identity_parts.append(
                f"{str(category).replace('_', ' ')}: {names or 'none'}"
            )
        provider_identity_disclosure = (
            "**H5 provider identities by lifecycle.** "
            + "; ".join(identity_parts)
            + ". Immutable evidence is bound in `h5/h5_results.json`."
        )

    lifecycle = denominators.get("provider_lifecycle_counts")
    lifecycle_disclosure = None
    if isinstance(lifecycle, Mapping):
        lifecycle_disclosure = (
            "**H5 provider lifecycle disclosure.** Of the six frozen providers: "
            "invalid calibration schema="
            f"{int(lifecycle.get('invalid_calibration_schema', 0))}; invalid provider "
            f"artifact={int(lifecycle.get('invalid_provider_artifact', 0))}; "
            f"missing={int(lifecycle.get('missing', 0))}; missing required revision="
            f"{int(lifecycle.get('missing_required_revision', 0))}; network interruption="
            f"{int(lifecycle.get('network_interruption', 0))}; model-drift exclusion="
            f"{int(lifecycle.get('model_drift_exclusion', 0))}; provider-configuration "
            "exclusion="
            f"{int(lifecycle.get('provider_configuration_exclusion', 0))}; pre-outcome "
            "design exclusion="
            f"{int(lifecycle.get('pre_outcome_design_exclusion', 0))}; technical "
            f"interruption={int(lifecycle.get('technical_interruption', 0))}; "
            "post-calibration exclusion="
            f"{int(lifecycle.get('post_calibration_exclusion', 0))}; other collected="
            f"{int(lifecycle.get('collected', 0))}. Invalid, missing, interrupted, and "
            "post-calibration-excluded providers do not enter qualifying-provider metrics. "
            "Immutable evidence is bound in `h5/h5_results.json`."
        )
    metrics = validated.payload["metrics"]
    diagnostic_parts = []
    for label, result_id in zip(
        ("MCQ", "Numeric"), ITEM_TYPE_DIAGNOSTIC_IDS, strict=True
    ):
        record = metrics[result_id]
        interval = _interval(record["ci95"], result_id)
        diagnostic_parts.append(
            f"{label} gap={_format_value(record['value'])}; 95% CI "
            f"{_format_value(list(interval))}; {int(record['denominator']):,} events / "
            f"{int(record['n_pair']):,} journeys / {int(record['n_target']):,} targets"
        )
    diagnostic = (
        "**Machine item-type generator diagnostic.** Because administered journeys "
        "are mixed trajectories, these probability-gap summaries are generator "
        "diagnostics, not item-type H1/H2 outcome estimands. "
        + ". ".join(diagnostic_parts)
        + "."
    )
    disclosures = [exclusion, "", audit]
    if lifecycle_disclosure is not None:
        disclosures.extend(("", lifecycle_disclosure))
    if provider_identity_disclosure is not None:
        disclosures.extend(("", provider_identity_disclosure))
    disclosures.extend(("", diagnostic))
    return disclosures


def _decision_cells(
    validated: ValidatedContract, hypothesis: str
) -> tuple[str, str, str]:
    branch = validated.branches[hypothesis]
    if branch is None:
        return "excluded pre-outcome", "not evaluated", "excluded_pre_outcome"
    return "complete", branch.decision, branch.reason_key


def _format_display_record(result_id: str, record: Mapping[str, Any]) -> str:
    registry_id = str(record["registry_metric_id"])
    value = _format_value(record["value"])
    ci95 = record.get("ci95")
    interval = "" if ci95 is None or isinstance(record["value"], list) else (
        f"; ci95={_format_value(ci95)}"
    )
    return (
        f"- result `{result_id}` / registry `{registry_id}`: value={value}{interval}; "
        f"numerator={_format_value(record['numerator'])}; "
        f"denominator={_format_value(record['denominator'])}; "
        f"n_target={_format_value(record['n_target'])}; "
        f"n_pair={_format_value(record['n_pair'])}."
    )


def _format_compact_display_record(
    result_id: str, record: Mapping[str, Any]
) -> str:
    value = _format_value(record["value"])
    ci95 = record.get("ci95")
    interval = "" if ci95 is None or isinstance(record["value"], list) else (
        f"; ci95={_format_value(ci95)}"
    )
    return (
        f"result `{result_id}` / registry `{record['registry_metric_id']}`: "
        f"value={value}{interval}; denominator={_format_value(record['denominator'])}"
    )


def _source_artifact_sentence(validated: ValidatedContract) -> str:
    return (
        "Source artifact: "
        f"`{validated.payload['analysis_artifact']}` "
        f"(`sha256:{validated.payload['analysis_artifact_sha256']}`). "
        "Publication PNG bytes were copied only after contract hash verification."
    )


def _render_figure_links(
    validated: ValidatedContract,
    manuscript_path: Path,
    figure_output_dir: Path,
    *,
    figure_ids: frozenset[str] | None = None,
) -> list[str]:
    lines: list[str] = []
    manuscript_dir = manuscript_path.parent.resolve()
    output_dir = figure_output_dir.resolve()
    for figure_id, references in sorted(validated.figure_references.items()):
        if figure_ids is not None and figure_id not in figure_ids:
            continue
        for reference in references:
            if not reference.is_png:
                continue
            destination = output_dir / reference.output_name
            relative = Path(os.path.relpath(destination, manuscript_dir)).as_posix()
            detail = "/".join(reference.key_path) if reference.key_path else "png"
            label = f"{figure_id} {detail}"
            lines.append(f"![{label}]({relative})")
    return lines


def _render_discussion(validated: ValidatedContract) -> str:
    lines = [
        _discussion_h1(validated.branches["H1"]),
        _discussion_h2(validated.branches["H2"]),
        _discussion_h3(validated.branches["H3"]),
        _discussion_h4(validated.branches["H4"]),
        _discussion_h5(validated),
        (
            "These findings remain limited to simulated, finite-budget diagnosis in the "
            "tested catalog. They do not establish structural non-identifiability, human "
            "validity, learning gains, or educational efficacy."
        ),
    ]
    return "\n\n".join(lines)


def _discussion_h1(branch: HypothesisBranch | None) -> str:
    assert branch is not None
    if branch.decision == "supported":
        return (
            "**H1.** The frozen P-state rescue criterion was supported. The result "
            "supports a finite-budget advantage for belief-triggered prerequisite "
            "evidence under the tested model and catalog, not an asymptotic or human claim."
        )
    if branch.decision == "not_supported":
        return (
            "**H1.** The frozen P-state rescue criterion was not supported: the "
            "belief-triggered arm missed the required level and the rescue contrast was "
            "not positive. The production-bound result therefore overrides the pilot "
            "expectation."
        )
    return (
        "**H1.** The frozen P-state rescue criterion was partially supported. Some "
        "required evidence favored rescue, but the full conjunction did not pass; the "
        "paper therefore makes no full-support claim."
    )


def _discussion_h2(branch: HypothesisBranch | None) -> str:
    assert branch is not None
    if branch.reason_key == "supported":
        return (
            "**H2.** The probe-policy harm criterion was supported. Under the tested "
            "conditions, fixed-quota probing harmed C-state classification while the "
            "belief-triggered policy met the frozen no-harm comparison."
        )
    if branch.reason_key == "harm_nonpositive_and_a_inferior":
        return (
            "**H2.** The criterion was not supported for both frozen reasons: the "
            "fixed-quota harm contrast was non-positive and the belief-triggered arm "
            "failed the no-harm requirement relative to the local-only arm."
        )
    if branch.reason_key == "harm_nonpositive":
        return (
            "**H2.** The criterion was not supported because the fixed-quota harm "
            "contrast was non-positive. The pilot harm pattern did not survive the "
            "production-bound comparison."
        )
    if branch.reason_key == "a_inferior":
        return (
            "**H2.** The criterion was not supported. The fixed-quota harm contrast "
            "remained positive, but the belief-triggered arm failed the no-harm "
            "requirement relative to the local-only arm; this is not a reversal of the "
            "fixed-quota pattern."
        )
    return (
        "**H2.** The criterion was partially supported. The point directions did not "
        "trigger a frozen failure branch, but interval evidence did not establish the "
        "full harm-and-no-harm conjunction."
    )


def _discussion_h3(branch: HypothesisBranch | None) -> str:
    assert branch is not None
    if branch.decision == "supported":
        outcome = "supported"
    elif branch.decision == "not_supported":
        outcome = "not supported because both accuracy and convergence time favored B"
    else:
        outcome = "partially supported because the two sanity criteria were mixed"
    return (
        f"**H3.** The subordinate adaptive sanity check was {outcome}. It remains "
        "secondary because adaptive-testing efficiency is established prior art."
    )


def _discussion_h4(branch: HypothesisBranch | None) -> str:
    assert branch is not None
    if branch.decision == "supported":
        outcome = "both predicted directions persisted"
    elif branch.decision == "not_supported":
        outcome = "neither predicted direction persisted"
    else:
        outcome = "only one predicted direction persisted"
    return (
        f"**H4.** Under the frozen misspecification stress, {outcome}. This result "
        "describes sensitivity to one declared perturbation family and cannot establish "
        "behavioral realism."
    )


def _discussion_h5(validated: ValidatedContract) -> str:
    branch = validated.branches["H5"]
    if branch is None:
        return (
            "**H5.** The LLM-persona hypothesis was excluded pre-outcome because its "
            "machine mapping gate was unavailable. It was not evaluated and is not "
            "reported as a negative finding."
        )
    if branch.decision == "supported":
        outcome = "met all frozen completion and manipulation gates"
    elif branch.decision == "not_supported":
        outcome = "met a frozen not-supported branch"
    else:
        outcome = "met only part of the frozen completion and manipulation criteria"
    return (
        f"**H5.** The secondary LLM-persona analysis {outcome}. Any statement is "
        "limited to qualifying simulated provider cells and is not evidence about humans."
    )


def _replace_generated_block(
    text: str, begin: str, end: str, content: str
) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise PaperContractError(f"manuscript marker pair is invalid: {begin}")
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    replacement = f"{begin}\n{content.rstrip()}\n{end}"
    return text[:start] + replacement + text[finish:]


def _manuscript_skeleton_sha256(text: str, *, include_zh: bool) -> str:
    marker_pairs = [
        (STATUS_BEGIN, STATUS_END),
        (ABSTRACT_EN_BEGIN, ABSTRACT_EN_END),
        (RESULTS_BEGIN, RESULTS_END),
        (DISCUSSION_BEGIN, DISCUSSION_END),
    ]
    if include_zh:
        marker_pairs.insert(2, (ABSTRACT_ZH_BEGIN, ABSTRACT_ZH_END))
    normalized = text
    for begin, end in marker_pairs:
        normalized = _replace_generated_block(
            normalized,
            begin,
            end,
            "<machine-generated-content>",
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_manuscript_skeleton(
    manuscript: str,
    observed_sha256: str,
    frozen_sha256: str,
) -> None:
    if observed_sha256 == frozen_sha256:
        return

    try:
        manifest_bytes = MANUSCRIPT_SKELETON_AMENDMENT_PATH.read_bytes()
    except OSError as exc:
        raise PaperContractError(
            f"manuscript skeleton differs from frozen template: {manuscript}; "
            "audited amendment manifest is unavailable"
        ) from exc
    if not _is_sha256(MANUSCRIPT_SKELETON_AMENDMENT_SHA256) or (
        hashlib.sha256(manifest_bytes).hexdigest()
        != MANUSCRIPT_SKELETON_AMENDMENT_SHA256
    ):
        raise PaperContractError("manuscript skeleton amendment manifest hash mismatch")
    try:
        payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaperContractError("manuscript skeleton amendment manifest is invalid") from exc
    if not isinstance(payload, Mapping):
        raise PaperContractError("manuscript skeleton amendment manifest is invalid")
    if (
        payload.get("schema_version") != "yher.paper-skeleton-amendments.v1"
        or payload.get("policy") != "post_collection_non_outcome_editorial_only"
    ):
        raise PaperContractError("manuscript skeleton amendment policy is invalid")
    amendments = payload.get("amendments")
    if not isinstance(amendments, list):
        raise PaperContractError("manuscript skeleton amendments must be a list")
    matching = [
        row
        for row in amendments
        if isinstance(row, Mapping) and row.get("manuscript") == manuscript
    ]
    if len(matching) != 1:
        raise PaperContractError(
            f"manuscript skeleton amendment is not unique: {manuscript}"
        )
    amendment = matching[0]
    if amendment.get("classification") != "non_outcome_editorial_audit_correction":
        raise PaperContractError("manuscript skeleton amendment classification is invalid")
    if amendment.get("generated_outcome_regions_unchanged") is not True:
        raise PaperContractError(
            "manuscript skeleton amendment must leave generated outcome regions unchanged"
        )
    if amendment.get("outcome_knowledge_status") != (
        "post_collection_non_outcome_only"
    ):
        raise PaperContractError(
            "manuscript skeleton amendment outcome knowledge status is invalid"
        )
    if amendment.get("from_skeleton_sha256") != frozen_sha256:
        raise PaperContractError(
            "manuscript skeleton amendment has the wrong frozen predecessor"
        )
    if amendment.get("to_skeleton_sha256") != observed_sha256:
        raise PaperContractError(
            f"manuscript skeleton amendment does not authorize: {manuscript}"
        )
    if not isinstance(amendment.get("amendment_id"), str) or not str(
        amendment["amendment_id"]
    ).strip():
        raise PaperContractError("manuscript skeleton amendment ID is invalid")
    if not isinstance(amendment.get("rationale"), str) or not str(
        amendment["rationale"]
    ).strip():
        raise PaperContractError("manuscript skeleton amendment rationale is invalid")
    evidence = amendment.get("evidence_anchors")
    if not isinstance(evidence, list) or not evidence or any(
        not isinstance(anchor, str) or not anchor.strip() for anchor in evidence
    ):
        raise PaperContractError(
            "manuscript skeleton amendment evidence anchors are invalid"
        )
    affected_sections = amendment.get("affected_sections")
    if not isinstance(affected_sections, list) or not affected_sections or any(
        not isinstance(section, str) or not section.strip()
        for section in affected_sections
    ):
        raise PaperContractError(
            "manuscript skeleton amendment affected sections are invalid"
        )
    reviewer = amendment.get("reviewer")
    if not isinstance(reviewer, str) or re.fullmatch(
        r"codex_[a-z0-9_]+", reviewer
    ) is None:
        raise PaperContractError("manuscript skeleton amendment reviewer is invalid")
    _require_rfc3339_utc(
        amendment.get("recorded_at_utc"),
        "manuscript skeleton amendment recorded_at_utc",
    )
    diff_relative = amendment.get("normalized_diff_path")
    diff_sha256 = amendment.get("normalized_diff_sha256")
    if not isinstance(diff_relative, str) or not diff_relative:
        raise PaperContractError("manuscript skeleton amendment diff path is invalid")
    diff_root = MANUSCRIPT_SKELETON_AMENDMENT_PATH.parent.resolve()
    diff_path = (diff_root / diff_relative).resolve()
    try:
        diff_path.relative_to(diff_root)
    except ValueError as exc:
        raise PaperContractError(
            "manuscript skeleton amendment diff path is outside manifest directory"
        ) from exc
    try:
        diff_bytes = diff_path.read_bytes()
    except OSError as exc:
        raise PaperContractError(
            "manuscript skeleton amendment diff is unavailable"
        ) from exc
    if not _is_sha256(diff_sha256) or hashlib.sha256(diff_bytes).hexdigest() != diff_sha256:
        raise PaperContractError("manuscript skeleton amendment diff hash mismatch")


def _expected_figure_outputs(
    validated: ValidatedContract, output_dir: Path
) -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    for references in validated.figure_references.values():
        for reference in references:
            if not reference.is_png:
                continue
            content = reference.source_path.read_bytes()
            if hashlib.sha256(content).hexdigest() != reference.sha256:
                raise PaperContractError(
                    f"{reference.figure_id} source PNG changed after validation"
                )
            if not content.startswith(PNG_MAGIC):
                raise PaperContractError(
                    f"{reference.figure_id} source PNG signature changed"
                )
            destination = output_dir / reference.output_name
            previous = outputs.get(destination)
            if previous is not None and previous != content:
                raise PaperContractError(
                    f"deterministic figure filename collision: {destination.name}"
                )
            outputs[destination] = content
    return outputs


def _figure_output_drift(
    expected: Mapping[Path, bytes],
    stale: Sequence[Path],
) -> list[tuple[Path, str]]:
    drift: list[tuple[Path, str]] = []
    for path, content in expected.items():
        if not path.is_file():
            drift.append((path, "missing"))
        elif path.read_bytes() != content:
            drift.append((path, "drift"))
    drift.extend((path, "stale") for path in stale)
    return drift


def _stale_figure_outputs(
    expected: Mapping[Path, bytes], output_dir: Path
) -> tuple[Path, ...]:
    if not output_dir.is_dir():
        return ()
    expected_paths = frozenset(path.resolve() for path in expected)
    return tuple(
        path
        for path in sorted(output_dir.glob("*.png"))
        if path.resolve() not in expected_paths
    )


def _replace_outputs_atomically(
    paths: Sequence[Path],
    originals: Mapping[Path, str],
    expected: Mapping[Path, str],
    figure_outputs: Mapping[Path, bytes],
    stale_pngs: Sequence[Path],
) -> None:
    manuscript_changes = [path for path in paths if originals[path] != expected[path]]
    figure_changes = [
        path
        for path, content in figure_outputs.items()
        if not path.is_file() or path.read_bytes() != content
    ]
    manuscript_temps: dict[Path, Path] = {}
    figure_temps: dict[Path, Path] = {}
    replaced_manuscripts: list[Path] = []
    replaced_figures: list[Path] = []
    removed_stale: list[Path] = []
    original_figures: dict[Path, bytes | None] = {
        path: path.read_bytes() if path.is_file() else None for path in figure_changes
    }
    original_stale = {path: path.read_bytes() for path in stale_pngs}
    try:
        for path in manuscript_changes:
            temp = path.with_name(f".{path.name}.paper-tmp")
            temp.write_text(expected[path], encoding="utf-8", newline="\n")
            manuscript_temps[path] = temp
        for path in figure_changes:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.name}.paper-tmp")
            temp.write_bytes(figure_outputs[path])
            figure_temps[path] = temp
        # Images land first so a manuscript never points at a missing new artifact.
        for path in figure_changes:
            figure_temps[path].replace(path)
            replaced_figures.append(path)
        for path in stale_pngs:
            path.unlink()
            removed_stale.append(path)
        for path in manuscript_changes:
            manuscript_temps[path].replace(path)
            replaced_manuscripts.append(path)
    except OSError as exc:
        for path in replaced_manuscripts:
            rollback = path.with_name(f".{path.name}.paper-rollback")
            rollback.write_text(originals[path], encoding="utf-8", newline="\n")
            rollback.replace(path)
        for path in replaced_figures:
            original = original_figures[path]
            if original is None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            else:
                rollback = path.with_name(f".{path.name}.paper-rollback")
                rollback.write_bytes(original)
                rollback.replace(path)
        for path in removed_stale:
            rollback = path.with_name(f".{path.name}.paper-rollback")
            rollback.write_bytes(original_stale[path])
            rollback.replace(path)
        raise PaperContractError(
            "could not atomically replace manuscripts and figures"
        ) from exc
    finally:
        for temp in (*manuscript_temps.values(), *figure_temps.values()):
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "figure"


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    number = _finite_number(value, "display value")
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}"


def _format_percent(value: object) -> str:
    return f"{_finite_number(value, 'percentage') * 100.0:.2f}%"


def _format_pp(value: object) -> str:
    points = _finite_number(value, "percentage-point effect") * 100.0
    return f"{points:+.2f} pp"


def _format_pp_interval(value: object) -> str:
    low, high = _interval(value, "percentage-point interval")
    return f"{low * 100.0:.2f} to {high * 100.0:.2f} pp"


def _number(values: Mapping[str, object], key: str) -> float:
    if key not in values:
        raise PaperContractError(f"predicate input is missing: {key}")
    return _finite_number(values[key], key)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperContractError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PaperContractError(f"{label} must be a finite number")
    return number


def _boolean(values: Mapping[str, object], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise PaperContractError(f"predicate input {key} must be boolean")
    return value


def _interval(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise PaperContractError(f"{label} must contain a two-sided interval")
    low = _finite_number(value[0], f"{label}.low")
    high = _finite_number(value[1], f"{label}.high")
    if low > high:
        raise PaperContractError(f"{label} interval is reversed")
    return low, high


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value != "0" * 64
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind validated confirmatory results into both paper manuscripts."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--yau", type=Path, default=DEFAULT_YAU)
    parser.add_argument(
        "--figure-output-dir",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT,
        help="Directory for verified contract-listed publication PNG copies.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail without writing when generated manuscript blocks are stale.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bind_papers(
            args.contract,
            args.artifact_root,
            args.main,
            args.yau,
            figure_output_dir=args.figure_output_dir,
            check=args.check,
        )
    except PaperContractError as exc:
        print(f"paper binder error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
