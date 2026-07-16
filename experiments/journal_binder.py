"""Support-aware evidence binder for the journal manuscript.

The binder reads the frozen programmatic run only through an explicit manifest
path. It never writes source data and rejects comparisons across target support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


DEFAULT_REGISTRY = Path("/tmp/yher_sprint2/paper_results_768785c/metric_registry.json")
DEFAULT_RESULTS = Path("/tmp/yher_sprint2/paper_results_768785c/results.json")
DEFAULT_P2_DIR = Path("/tmp/yher_h5v2/p2")
DEFAULT_OUTPUT_DIR = Path("/tmp/yher_h5v2/journal_binder")

P2_CLAIM_BOUNDARY = (
    "supply_bound_algorithmic_illustration_not_learning_benefit_or_external_validation"
)
P2_SPEC_PATH = Path(__file__).with_name("p2_illustrative_analysis_plan.md")
P2_ARMS = ("oracle", "A", "B", "C")
P2_BOOTSTRAP_METRICS = (
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
P2_CONTRASTS = (
    "A_minus_oracle",
    "B_minus_oracle",
    "A_minus_B",
    "C_minus_oracle_ITT",
)
P2_PROHIBITED_CLAIM_FIELDS = frozenset(
    {
        "learning_benefit",
        "remediation_dose",
        "prescription_efficacy",
        "human_population_effect",
        "actual_harm_or_benefit_minutes",
    }
)

PERSONA_PROVIDERS = (
    "deepseek",
    "glm",
    "kimi",
    "minimax",
    "doubao",
    "tongyi",
)
PERSONA_CONTROLLED_STATES = (
    "correct_answer",
    "incorrect_answer",
    "abstention",
    "technical_or_schema_failure",
)
PERSONA_EFFECT_ORIENTATIONS = {
    "conditional_answer_accuracy": "control_minus_deficit",
    "correct_response_yield": "control_minus_deficit",
    "incorrect_response_yield": "deficit_minus_control",
    "abstention_yield": "deficit_minus_control",
    "technical_or_schema_failure_yield": "deficit_minus_control",
}
PERSONA_PUBLICATION_ARTIFACTS = {
    "tables": (
        "provider_lifecycle.csv",
        "controlled_composition.csv",
        "controlled_paired_effects.csv",
        "blind_agreement.csv",
        "blind_stability.csv",
    ),
    "figure_data": (
        "figure_data/controlled_composition.csv",
        "figure_data/blind_agreement.csv",
        "figure_data/blind_stability.csv",
    ),
    "figures": (
        "figures/controlled_composition.png",
        "figures/controlled_composition.svg",
        "figures/blind_terminal_agreement.png",
        "figures/blind_terminal_agreement.svg",
        "figures/blind_output_stability.png",
        "figures/blind_output_stability.svg",
    ),
}
PROGRAMMATIC_PUBLICATION_ASSETS = {
    "generated/fig-p-rescue-png-c36a76849139.png": "figures/p_rescue.png",
    "generated/fig-c-probe-harm-png-e5e22d30fb2c.png": (
        "figures/c_misdiagnosis.png"
    ),
    "generated/fig-matched-vs-misspecified-png-39c4270e4169.png": (
        "figures/matched_vs_misspecified.png"
    ),
}
P2_PUBLICATION_ARTIFACTS = (
    "figure_data.json",
    "p2_supply_bound_illustration.png",
    "p2_supply_bound_illustration.svg",
)

PAIR_SPECS = {
    "full_27_terminal_vs_correct_convergence": {
        "support": "full",
        "left": "outcome_by_view.matched.b15.arm_A.truth_P.terminal_accuracy",
        "right": "outcome_by_view.matched.b15.arm_A.truth_P.correct_convergence",
        "difference": None,
        "claim_class": "exploratory_posthoc",
    },
    "eligible_23_terminal_vs_correct_convergence": {
        "support": "eligible",
        "left": (
            "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A."
            "terminal_accuracy"
        ),
        "right": (
            "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A."
            "correct_convergence"
        ),
        "difference": (
            "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A."
            "terminal_minus_correct_convergence"
        ),
        "claim_class": "exploratory_posthoc",
    },
}

H1_REPORTABLE = (
    "p_rescue.full.matched.b15.arm_A",
    "p_rescue.full.matched.b15.arm_B",
    "h1.primary.matched.b15.rescue_A_minus_B",
    "h1.no_repeat.matched.b15.arm_A",
    "h1.no_repeat.matched.b15.arm_B",
    "h1.no_repeat.matched.b15.rescue_A_minus_B",
)
H2_REPORTABLE = (
    "c_misdiagnosis.full.matched.b9.arm_A",
    "c_misdiagnosis.full.matched.b9.arm_B",
    "c_misdiagnosis.full.matched.b9.arm_C",
    "h2.primary.matched.b9.harm_C_minus_A",
    "h2.primary.matched.b9.no_harm_A_minus_B",
    "h2.no_repeat.matched.b9.arm_A",
    "h2.no_repeat.matched.b9.arm_B",
    "h2.no_repeat.matched.b9.arm_C",
    "h2.no_repeat.matched.b9.harm_C_minus_A",
    "h2.no_repeat.matched.b9.no_harm_A_minus_B",
)
H3_REPORTABLE = (
    "h3.matched.b15.terminal_accuracy_A_minus_B",
    "h3.matched.b15.time_to_confidence.arm_A",
    "h3.matched.b15.time_to_confidence.arm_B",
)
H4_REPORTABLE = (
    "h4.misspecified.b15.rescue_A_minus_B",
    "h4.misspecified.b9.harm_C_minus_A",
    "h4.misspecified.b9.no_harm_A_minus_B",
    "h4.degradation.h1_rescue.matched_minus_misspecified",
    "h4.degradation.h2_harm.matched_minus_misspecified",
    "h4.degradation.h2_no_harm.matched_minus_misspecified",
    "h1.no_repeat.misspecified.b15.rescue_A_minus_B",
    "h2.no_repeat.misspecified.b9.harm_C_minus_A",
    "h2.no_repeat.misspecified.b9.no_harm_A_minus_B",
    "misspecification.item_type.mcq.generator_minus_production",
    "misspecification.item_type.numeric.generator_minus_production",
)
EXPECTED_HYPOTHESIS_DECISIONS = {
    "H1": "partially_supported",
    "H2": "not_supported",
    "H3": "supported",
    "H4": "supported",
    "H5": None,
}
EXPECTED_HYPOTHESIS_EVIDENCE = {
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
    "H4": [
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
    ],
    "H5": [],
}
UNBOUND_DIAGNOSTIC_IDS: frozenset[str] = frozenset()
MECHANISM_REPORTABLE = tuple(
    spec[key]
    for spec in PAIR_SPECS.values()
    for key in ("left", "right", "difference")
    if spec[key] is not None
) + (
    "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.direct_count",
    "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.prerequisite_count",
    "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.prerequisite_share",
)
POSTHOC_METRIC_IDS = frozenset(MECHANISM_REPORTABLE)
REPORTABLE_METRIC_IDS = tuple(
    dict.fromkeys(
        H1_REPORTABLE
        + H2_REPORTABLE
        + H3_REPORTABLE
        + H4_REPORTABLE
        + MECHANISM_REPORTABLE
    )
)

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


def _estimand(
    support_id: str,
    conditions: tuple[str, ...],
    condition_contrast: str | None,
    budget: int | None,
    truth_states: tuple[str, ...],
    arms: tuple[str, ...],
    weighting: str,
    denominator_kind: str,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "support_id": support_id,
            "conditions": conditions,
            "condition_contrast": condition_contrast,
            "budget": budget,
            "truth_states": truth_states,
            "arms": arms,
            "weighting": weighting,
            "denominator_kind": denominator_kind,
        }
    )


_ALL_TRUTH = ("M", "P", "C", "U")
ESTIMAND_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "p_rescue.full.matched.b15.arm_A": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("A",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "p_rescue.full.matched.b15.arm_B": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("B",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "h1.primary.matched.b15.rescue_A_minus_B": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("A", "B"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h1.no_repeat.matched.b15.arm_A": _estimand(
            "common_b15", ("matched",), None, 15, ("P",), ("A",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "h1.no_repeat.matched.b15.arm_B": _estimand(
            "common_b15", ("matched",), None, 15, ("P",), ("B",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "h1.no_repeat.matched.b15.rescue_A_minus_B": _estimand(
            "common_b15", ("matched",), None, 15, ("P",), ("A", "B"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "c_misdiagnosis.full.matched.b9.arm_A": _estimand(
            "eligible", ("matched",), None, 9, ("C",), ("A",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "c_misdiagnosis.full.matched.b9.arm_B": _estimand(
            "eligible", ("matched",), None, 9, ("C",), ("B",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "c_misdiagnosis.full.matched.b9.arm_C": _estimand(
            "eligible", ("matched",), None, 9, ("C",), ("C",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "h2.primary.matched.b9.harm_C_minus_A": _estimand(
            "eligible", ("matched",), None, 9, ("C",), ("C", "A"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h2.primary.matched.b9.no_harm_A_minus_B": _estimand(
            "eligible", ("matched",), None, 9, ("C",), ("A", "B"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h2.no_repeat.matched.b9.arm_A": _estimand(
            "common_b9", ("matched",), None, 9, ("C",), ("A",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "h2.no_repeat.matched.b9.arm_B": _estimand(
            "common_b9", ("matched",), None, 9, ("C",), ("B",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "h2.no_repeat.matched.b9.arm_C": _estimand(
            "common_b9", ("matched",), None, 9, ("C",), ("C",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "h2.no_repeat.matched.b9.harm_C_minus_A": _estimand(
            "common_b9", ("matched",), None, 9, ("C",), ("C", "A"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h2.no_repeat.matched.b9.no_harm_A_minus_B": _estimand(
            "common_b9", ("matched",), None, 9, ("C",), ("A", "B"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h3.matched.b15.terminal_accuracy_A_minus_B": _estimand(
            "full", ("matched",), None, 15, _ALL_TRUTH, ("A", "B"),
            PAIR_WEIGHT, "paired_target_state_replicate_contrasts",
        ),
        "h3.matched.b15.time_to_confidence.arm_A": _estimand(
            "full", ("matched",), None, 15, _ALL_TRUTH, ("A",), TIME_WEIGHT,
            "target_state_replicate_rows_time_coded",
        ),
        "h3.matched.b15.time_to_confidence.arm_B": _estimand(
            "full", ("matched",), None, 15, _ALL_TRUTH, ("B",), TIME_WEIGHT,
            "target_state_replicate_rows_time_coded",
        ),
        "h4.misspecified.b15.rescue_A_minus_B": _estimand(
            "eligible", ("misspecified",), None, 15, ("P",), ("A", "B"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h4.misspecified.b9.harm_C_minus_A": _estimand(
            "eligible", ("misspecified",), None, 9, ("C",), ("C", "A"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h4.misspecified.b9.no_harm_A_minus_B": _estimand(
            "eligible", ("misspecified",), None, 9, ("C",), ("A", "B"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h4.degradation.h1_rescue.matched_minus_misspecified": _estimand(
            "eligible", ("matched", "misspecified"), "matched_minus_misspecified",
            15, ("P",), ("A", "B"), DEGRADATION_WEIGHT,
            "paired_condition_target_replicate_contrasts",
        ),
        "h4.degradation.h2_harm.matched_minus_misspecified": _estimand(
            "eligible", ("matched", "misspecified"), "matched_minus_misspecified",
            9, ("C",), ("C", "A"), DEGRADATION_WEIGHT,
            "paired_condition_target_replicate_contrasts",
        ),
        "h4.degradation.h2_no_harm.matched_minus_misspecified": _estimand(
            "eligible", ("matched", "misspecified"), "matched_minus_misspecified",
            9, ("C",), ("A", "B"), DEGRADATION_WEIGHT,
            "paired_condition_target_replicate_contrasts",
        ),
        "h1.no_repeat.misspecified.b15.rescue_A_minus_B": _estimand(
            "common_b15", ("misspecified",), None, 15, ("P",), ("A", "B"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h2.no_repeat.misspecified.b9.harm_C_minus_A": _estimand(
            "common_b9", ("misspecified",), None, 9, ("C",), ("C", "A"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "h2.no_repeat.misspecified.b9.no_harm_A_minus_B": _estimand(
            "common_b9", ("misspecified",), None, 9, ("C",), ("A", "B"),
            PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "misspecification.item_type.mcq.generator_minus_production": _estimand(
            "item_type_mcq", ("misspecified",), None, None, _ALL_TRUTH,
            ("generator_probability", "production_probability"), ITEM_WEIGHT,
            "misspecified_event_pairs",
        ),
        "misspecification.item_type.numeric.generator_minus_production": _estimand(
            "item_type_numeric", ("misspecified",), None, None, _ALL_TRUTH,
            ("generator_probability", "production_probability"), ITEM_WEIGHT,
            "misspecified_event_pairs",
        ),
        "outcome_by_view.matched.b15.arm_A.truth_P.terminal_accuracy": _estimand(
            "full", ("matched",), None, 15, ("P",), ("A",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "outcome_by_view.matched.b15.arm_A.truth_P.correct_convergence": _estimand(
            "full", ("matched",), None, 15, ("P",), ("A",), ROW_WEIGHT,
            "target_replicate_rows",
        ),
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.terminal_accuracy": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("A",),
            POSTHOC_ROW_WEIGHT, "target_replicate_rows",
        ),
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.correct_convergence": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("A",),
            POSTHOC_ROW_WEIGHT, "target_replicate_rows",
        ),
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.terminal_minus_correct_convergence": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("A",),
            POSTHOC_PAIR_WEIGHT, "paired_target_replicate_contrasts",
        ),
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.direct_count": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("A",),
            POSTHOC_ROW_WEIGHT, "target_replicate_rows",
        ),
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.prerequisite_count": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("A",),
            POSTHOC_ROW_WEIGHT, "target_replicate_rows",
        ),
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.prerequisite_share": _estimand(
            "eligible", ("matched",), None, 15, ("P",), ("A",),
            POSTHOC_ROW_WEIGHT, "target_replicate_rows",
        ),
    }
)
if tuple(ESTIMAND_SPECS) != REPORTABLE_METRIC_IDS:
    raise RuntimeError("estimand table must exactly cover the reportable metric order")

REQUIRED_METRIC_FIELDS = frozenset(
    {
        "metric_id",
        "value",
        "numerator",
        "denominator",
        "weighting",
        "n_target",
        "n_pair",
        "raw_hash",
        "ci_low",
        "ci_high",
    }
)


class BinderError(ValueError):
    """Raised when evidence cannot be bound without changing its claim surface."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_set_hash(targets: Sequence[str]) -> str:
    roster = sorted(str(target) for target in targets)
    if len(roster) != len(set(roster)):
        raise BinderError("target roster contains duplicates")
    return hashlib.sha256(canonical_json_bytes(roster)).hexdigest()


def _safe_shard_path(root: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename:
        raise BinderError("raw manifest shard filename is invalid")
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise BinderError(f"raw manifest shard path escapes its root: {filename}")
    return root / relative


def _extract_supports(
    raw_manifest_path: Path, *, require_frozen: bool = False
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_bytes = raw_manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise BinderError("raw manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise BinderError("raw manifest root must be an object")
    if require_frozen and (
        manifest.get("record_type") != "confirmatory_run_manifest"
        or manifest.get("run_id") != "confirmatory-v1"
        or manifest.get("status") != "complete"
    ):
        raise BinderError(
            "complete binding requires the confirmatory-v1 raw manifest in complete status"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise BinderError("raw manifest has no shards")

    target_flags: dict[str, set[bool]] = {}
    common_flags: dict[str, dict[int, set[bool]]] = {}
    item_type_targets: dict[str, set[str]] = {"mcq": set(), "numeric": set()}
    shard_hashes: dict[str, str] = {}
    ordered_shards: list[dict[str, str]] = []
    shard_root = raw_manifest_path.parent
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise BinderError("raw manifest shard entry is not an object")
        path = _safe_shard_path(shard_root, shard.get("filename"))
        if not path.is_file():
            raise BinderError(f"raw manifest shard is missing: {path.name}")
        actual_hash = sha256_file(path)
        expected_hash = shard.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise BinderError(
                f"raw manifest shard lacks a declared SHA-256: {path.name}"
            )
        if expected_hash != actual_hash:
            raise BinderError(f"raw manifest shard SHA-256 drift: {path.name}")
        shard_hashes[path.name] = actual_hash
        ordered_shards.append({"filename": path.name, "sha256": actual_hash})
        shard_id = shard.get("shard_id")
        if not isinstance(shard_id, str):
            raise BinderError(f"raw manifest shard_id is invalid: {path.name}")
        match = re.fullmatch(r"target=(.+?)\|truth=[^|]+\|condition=[^|]+", shard_id)
        if match is None:
            raise BinderError(f"raw manifest shard_id cannot be parsed: {shard_id}")
        shard_target = match.group(1)

        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BinderError(
                        f"raw shard has invalid JSON at {path.name}:{line_number}"
                    ) from exc
                if not isinstance(row, Mapping) or "target_node" not in row:
                    continue
                target = row.get("target_node")
                if target != shard_target:
                    raise BinderError(f"raw shard target mismatch: {path.name}")
                eligible = row.get("h1_h2_eligible")
                if not isinstance(eligible, bool):
                    raise BinderError(f"raw row lacks eligibility flag: {path.name}")
                target_flags.setdefault(shard_target, set()).add(eligible)
                per_budget = common_flags.setdefault(shard_target, {})
                views = row.get("views")
                if not isinstance(views, list):
                    raise BinderError(f"raw row lacks views: {path.name}")
                for view in views:
                    if not isinstance(view, Mapping):
                        raise BinderError(f"raw view is not an object: {path.name}")
                    budget = view.get("nominal_budget")
                    flag = view.get("common_support_no_repeat")
                    if budget not in (9, 15, 25) or not isinstance(flag, bool):
                        continue
                    per_budget.setdefault(int(budget), set()).add(flag)
                if row.get("condition") == "misspecified":
                    events = row.get("events")
                    if not isinstance(events, list):
                        raise BinderError(f"raw row lacks event records: {path.name}")
                    for event in events:
                        if not isinstance(event, Mapping):
                            raise BinderError(f"raw event is not an object: {path.name}")
                        item_type = event.get("item_type")
                        if item_type in item_type_targets:
                            item_type_targets[str(item_type)].add(shard_target)

    if set(target_flags) != set(common_flags):
        raise BinderError("raw support extraction is incomplete")
    for target, values in target_flags.items():
        if len(values) != 1:
            raise BinderError(f"eligibility flag is inconsistent for target: {target}")
        for budget in (9, 15, 25):
            flags = common_flags[target].get(budget)
            if not flags or len(flags) != 1:
                raise BinderError(
                    f"common-support flag is inconsistent for {target} at b{budget}"
                )

    full = sorted(target_flags)
    eligible = sorted(target for target, values in target_flags.items() if True in values)
    rosters = {
        "full": full,
        "eligible": eligible,
        **{
            f"common_b{budget}": sorted(
                target
                for target, by_budget in common_flags.items()
                if True in by_budget[budget]
            )
            for budget in (9, 15, 25)
        },
        "item_type_mcq": sorted(item_type_targets["mcq"]),
        "item_type_numeric": sorted(item_type_targets["numeric"]),
    }
    predicates = {
        "full": "target_node in frozen manifest target roster",
        "eligible": "h1_h2_eligible == true",
        "common_b9": "views[nominal_budget == 9].common_support_no_repeat == true",
        "common_b15": "views[nominal_budget == 15].common_support_no_repeat == true",
        "common_b25": "views[nominal_budget == 25].common_support_no_repeat == true",
        "item_type_mcq": (
            "target has at least one misspecified event.item_type == mcq"
        ),
        "item_type_numeric": (
            "target has at least one misspecified event.item_type == numeric"
        ),
    }
    supports = {
        support_id: {
            "support_id": support_id,
            "target_set_hash": target_set_hash(roster),
            "target_set_hash_algorithm": (
                "sha256(canonical_json(exact_sorted_target_roster))"
            ),
            "target_roster": roster,
            "n_target": len(roster),
            "support_predicate": predicates[support_id],
        }
        for support_id, roster in rosters.items()
    }
    if len(full) == 27:
        supports["full_27"] = supports["full"]
    if len(eligible) == 23:
        supports["eligible_23"] = supports["eligible"]
    common_aliases = {9: 9, 15: 4, 25: 1}
    for budget, expected_count in common_aliases.items():
        support = supports[f"common_b{budget}"]
        if support["n_target"] == expected_count:
            supports[f"common_{expected_count}_b{budget}"] = support
    for item_type, expected_count in (("mcq", 27), ("numeric", 13)):
        support = supports[f"item_type_{item_type}"]
        if support["n_target"] == expected_count:
            supports[f"item_type_{item_type}_{expected_count}"] = support

    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    source_hashes = {
        "raw_manifest_sha256": manifest_sha256,
        "raw_shard_set_sha256": hashlib.sha256(
            canonical_json_bytes(shard_hashes)
        ).hexdigest(),
        "raw_aggregate_hash": hashlib.sha256(
            canonical_json_bytes(
                {"manifest_sha256": manifest_sha256, "shards": ordered_shards}
            )
        ).hexdigest(),
    }
    config_sha256 = manifest.get("config_sha256")
    if isinstance(config_sha256, str) and config_sha256:
        source_hashes["config_sha256"] = config_sha256
    input_sha256 = manifest.get("input_sha256")
    if isinstance(input_sha256, Mapping):
        plan = input_sha256.get("confirmatory_analysis_plan")
        if isinstance(plan, Mapping) and isinstance(plan.get("sha256"), str):
            source_hashes["analysis_plan_sha256"] = str(plan["sha256"])
    return supports, source_hashes


def _metric_support_id(metric_id: str) -> str:
    spec = ESTIMAND_SPECS.get(metric_id)
    if spec is None:
        raise BinderError(f"metric estimand is undefined: {metric_id}")
    return str(spec["support_id"])


def _unbound_metric(
    row: Mapping[str, Any], *, source_hashes: Mapping[str, str]
) -> dict[str, Any]:
    """Retain a diagnostic row while refusing to invent its target roster."""
    return {
        **dict(row),
        "status": "unbound_missing_exact_target_roster",
        "support_id": None,
        "target_set_hash": None,
        "target_roster": None,
        "filter_predicate": None,
        "denominator_definition": (
            "registry denominator retained; exact target roster was not retained "
            "in the supplied evidence bundle"
        ),
        "claim_class": "exploratory_diagnostic_unbound",
        "post_hoc": True,
        "source_hashes": dict(source_hashes),
    }


def _filter_dimensions(
    metric_id: str, support: Mapping[str, Any]
) -> dict[str, Any]:
    spec = ESTIMAND_SPECS[metric_id]
    return {
        "support_id": support["support_id"],
        "target_set_hash": support["target_set_hash"],
        "conditions": list(spec["conditions"]),
        "condition_contrast": spec["condition_contrast"],
        "budget": spec["budget"],
        "truth_states": list(spec["truth_states"]),
        "arms": list(spec["arms"]),
    }


def _filter_predicate(metric_id: str, support: Mapping[str, Any]) -> str:
    spec = ESTIMAND_SPECS[metric_id]
    conditions = tuple(str(value) for value in spec["conditions"])
    if spec["condition_contrast"] is None:
        condition_clause = f"condition == {conditions[0]}"
    else:
        condition_clause = (
            "conditions in {"
            + ",".join(conditions)
            + "}; paired condition contrast == "
            + str(spec["condition_contrast"])
        )
    clauses = [condition_clause, str(support["support_predicate"])]
    if spec["budget"] is not None:
        clauses.extend(
            [
                f"views.nominal_budget == {spec['budget']}",
                "views.valid == true",
            ]
        )
    truth_states = tuple(str(value) for value in spec["truth_states"])
    if len(truth_states) == 1:
        clauses.append(f"truth == {truth_states[0]}")
    else:
        clauses.append("truth in {" + ",".join(truth_states) + "}")
    arms = tuple(str(value) for value in spec["arms"])
    if len(arms) == 1:
        clauses.append(f"arm == {arms[0]}")
    elif arms == ("generator_probability", "production_probability"):
        clauses.append("paired generator_probability minus production_probability")
    else:
        clauses.append("paired arms in contrast order == (" + ",".join(arms) + ")")
    if str(spec["support_id"]).startswith("item_type_"):
        clauses.extend(
            [
                f"event.item_type == {str(spec['support_id']).removeprefix('item_type_')}",
                "journey cluster preserved",
            ]
        )
    return "; ".join(clauses)


def _denominator_definition(metric_id: str, row: Mapping[str, Any]) -> str:
    kind = str(ESTIMAND_SPECS[metric_id]["denominator_kind"])
    if kind == "misspecified_event_pairs":
        return (
            f"{row['denominator']} misspecified journey events nested in "
            f"{row['n_pair']} journeys across {row['n_target']} exact targets"
        )
    if kind == "target_state_replicate_rows_time_coded":
        return (
            f"{row['denominator']} target-state-replicate rows; numerator is the "
            "sum of time-coded audit rows (NC=budget+1), not a success count"
        )
    if kind == "paired_condition_target_replicate_contrasts":
        return (
            f"{row['denominator']} paired matched-minus-misspecified "
            f"target-replicate contrasts across {row['n_target']} exact targets"
        )
    if kind in {
        "paired_target_replicate_contrasts",
        "paired_target_state_replicate_contrasts",
    }:
        return (
            f"{row['denominator']} paired target-replicate contrasts across "
            f"{row['n_target']} exact targets"
        )
    return (
        f"{row['denominator']} target-replicate rows across "
        f"{row['n_target']} exact targets"
    )


def _strict_json_bytes(data: bytes, *, label: str) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise BinderError(f"{label} contains a duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise BinderError(f"{label} contains a non-finite JSON constant: {value}")

    try:
        return json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BinderError(f"{label} is not valid JSON") from exc


def _load_registry(path: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    raw_bytes = path.read_bytes()
    payload = _strict_json_bytes(raw_bytes, label="metric registry")
    if not isinstance(payload, list):
        raise BinderError("metric registry root must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            raise BinderError("metric registry row is not an object")
        metric_id = row.get("metric_id")
        if not isinstance(metric_id, str) or not metric_id:
            raise BinderError("metric registry row has no metric_id")
        if metric_id in by_id:
            raise BinderError(f"duplicate metric registry row: {metric_id}")
        by_id[metric_id] = row
    return by_id, {"metric_registry_sha256": hashlib.sha256(raw_bytes).hexdigest()}


def _load_json_object(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    raw_bytes = path.read_bytes()
    payload = _strict_json_bytes(raw_bytes, label=label)
    if not isinstance(payload, Mapping):
        raise BinderError(f"{label} root must be an object")
    return payload, raw_bytes


def _validate_complete_h1_h4_evidence(
    *,
    raw_manifest_path: Path,
    registry_path: Path,
    results_path: Path,
    paper_artifact_manifest_path: Path,
    supports: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    registry_rows: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, dict[str, Any]]]:
    if not paper_artifact_manifest_path.is_file():
        raise BinderError(
            "paper artifact manifest is required for a complete binding"
        )
    raw_manifest, _ = _load_json_object(raw_manifest_path, label="raw manifest")
    paper_manifest, paper_manifest_bytes = _load_json_object(
        paper_artifact_manifest_path, label="paper artifact manifest"
    )

    files = paper_manifest.get("files")
    if not isinstance(files, Mapping):
        raise BinderError("paper artifact manifest has no file hash map")
    if files.get("metric_registry.json") != sha256_file(registry_path):
        raise BinderError("metric registry is not hash-bound by paper artifact manifest")
    if files.get("results.json") != sha256_file(results_path):
        raise BinderError("results artifact is not hash-bound by paper artifact manifest")

    results_source = results.get("source_provenance")
    paper_source = paper_manifest.get("source_provenance")
    if not isinstance(results_source, Mapping) or not isinstance(paper_source, Mapping):
        raise BinderError("results and paper artifact source provenance are required")
    if dict(results_source) != dict(paper_source):
        raise BinderError("results and paper artifact source provenance differ")

    input_sha256 = raw_manifest.get("input_sha256")
    plan = (
        input_sha256.get("confirmatory_analysis_plan")
        if isinstance(input_sha256, Mapping)
        else None
    )
    expected_source = {
        "analysis_plan_commit": raw_manifest.get("analysis_plan_commit"),
        "analysis_plan_sha256": (
            plan.get("sha256") if isinstance(plan, Mapping) else None
        ),
        "config_sha256": raw_manifest.get("config_sha256"),
        "experiment_tag": raw_manifest.get("experiment_tag"),
        "run_id": raw_manifest.get("run_id"),
        "run_started_at_utc": raw_manifest.get("run_started_at_utc"),
        "runner_commit": raw_manifest.get("runner_commit"),
        "source_manifest_sha256": source_hashes["raw_manifest_sha256"],
    }
    for field, expected in expected_source.items():
        if expected is None or results_source.get(field) != expected:
            label = "source manifest" if field == "source_manifest_sha256" else field
            raise BinderError(f"results {label} differs from frozen raw manifest")

    raw_hash = source_hashes["raw_aggregate_hash"]
    if results.get("raw_hash") != raw_hash or paper_manifest.get("raw_hash") != raw_hash:
        raise BinderError("raw aggregate hash is not cross-bound across all artifacts")

    registry_ids = list(registry_rows)
    if (
        results.get("registry_metric_ids") != registry_ids
        or paper_manifest.get("registry_metric_ids") != registry_ids
    ):
        raise BinderError("registry metric IDs are not cross-bound across all artifacts")
    if results.get("decisions") != EXPECTED_HYPOTHESIS_DECISIONS:
        raise BinderError("results differ from the frozen H1-H4 decisions")
    if results.get("hypothesis_evidence") != EXPECTED_HYPOTHESIS_EVIDENCE:
        raise BinderError("results hypothesis evidence differs from the frozen registry")
    if any(
        metric_id not in registry_rows
        for metric_ids in EXPECTED_HYPOTHESIS_EVIDENCE.values()
        for metric_id in metric_ids
    ):
        raise BinderError("frozen hypothesis evidence is missing from metric registry")

    for field in ("analysis_code_sha256", "analysis_commit"):
        value = results.get(field)
        if not isinstance(value, str) or not value or paper_manifest.get(field) != value:
            raise BinderError(f"analysis identity is not cross-bound: {field}")

    validation = results.get("validation")
    raw_validation = raw_manifest.get("validation")
    if not isinstance(validation, Mapping) or not isinstance(raw_validation, Mapping):
        raise BinderError("raw and results validation counts are required")
    common = raw_validation.get("common_support_targets")
    if not isinstance(common, Mapping):
        raise BinderError("raw common-support validation counts are required")
    shard_count = len(raw_manifest.get("shards") or ())
    intended = raw_manifest.get("expected_journey_count")
    expected_counts = {
        "manifest_shard_count": shard_count,
        "intended_journey_count": intended,
        "journey_count": intended,
        "programmatic_journey_count": intended,
        "programmatic_primary_key_count": intended,
        "target_count": supports["full"]["n_target"],
        "full_target_count": supports["full"]["n_target"],
        "h1_h2_eligible_target_count": supports["eligible"]["n_target"],
        "common_support_target_count_b9": supports["common_b9"]["n_target"],
        "common_support_target_count_b15": supports["common_b15"]["n_target"],
        "common_support_target_count_b25": supports["common_b25"]["n_target"],
        "raw_hash": raw_hash,
    }
    for field, expected in expected_counts.items():
        if validation.get(field) != expected:
            raise BinderError(f"results validation count differs from raw evidence: {field}")
    if raw_manifest.get("full_shard_count") != shard_count or raw_manifest.get(
        "selected_shard_count"
    ) != shard_count:
        raise BinderError("raw manifest shard counts are not complete")
    if raw_manifest.get("full_grid_complete") is not True:
        raise BinderError("raw manifest full grid is not complete")
    raw_count_expectations = {
        "open_nodes": supports["full"]["n_target"],
        "h1_h2_eligible": supports["eligible"]["n_target"],
        "expected_journeys": intended,
    }
    for field, expected in raw_count_expectations.items():
        if raw_validation.get(field) != expected:
            raise BinderError(f"raw validation count differs from extracted support: {field}")
    for budget in (9, 15, 25):
        if common.get(str(budget)) != supports[f"common_b{budget}"]["n_target"]:
            raise BinderError(
                f"raw common-support count differs from extracted support: b{budget}"
            )
    try:
        valid_count = int(validation["valid_journey_count"])
        structural_count = int(validation["structural_failure_count"])
        schema_invalid_count = int(validation["schema_invalid_count"])
        intended_count = int(intended)
    except (KeyError, TypeError, ValueError) as exc:
        raise BinderError("results lifecycle validation counts are invalid") from exc
    if valid_count + structural_count + schema_invalid_count != intended_count:
        raise BinderError("results lifecycle validation counts do not reconcile")

    source_updates = {
        "paper_artifact_manifest_sha256": hashlib.sha256(
            paper_manifest_bytes
        ).hexdigest(),
    }
    execution_integrity = {
        "intended_journey_count": intended_count,
        "valid_journey_count": valid_count,
        "structural_failure_count": structural_count,
        "schema_invalid_count": schema_invalid_count,
        "manifest_shard_count": shard_count,
        "raw_hash": raw_hash,
    }
    publication_assets: dict[str, dict[str, Any]] = {}
    for destination, relative_value in PROGRAMMATIC_PUBLICATION_ASSETS.items():
        relative = Path(relative_value)
        expected_hash = files.get(relative.as_posix())
        if not isinstance(expected_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ) is None:
            raise BinderError(
                f"programmatic publication figure is not manifest-bound: {relative}"
            )
        source, data = _read_relative_regular_file(
            paper_artifact_manifest_path.parent,
            relative,
            label=f"programmatic publication figure {relative}",
        )
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise BinderError(f"programmatic publication figure drifted: {relative}")
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise BinderError(f"programmatic publication figure is not PNG: {relative}")
        publication_assets[destination] = {
            "relative_path": relative.as_posix(),
            "source_path": str(source.resolve()),
            "sha256": expected_hash,
            "bytes": len(data),
        }
    return source_updates, execution_integrity, publication_assets


def _validate_metric(
    row: Mapping[str, Any],
    *,
    support: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    missing = REQUIRED_METRIC_FIELDS - set(row)
    if missing:
        raise BinderError(f"metric is missing required fields: {sorted(missing)}")
    metric_id = str(row["metric_id"])
    if int(row["n_target"]) != int(support["n_target"]):
        raise BinderError(
            f"metric target count differs from exact support: {metric_id}"
        )
    if float(row["denominator"]) <= 0:
        raise BinderError(f"metric denominator is not positive: {metric_id}")
    try:
        n_pair = int(row["n_pair"])
    except (TypeError, ValueError) as exc:
        raise BinderError(f"metric pair count is invalid: {metric_id}") from exc
    if n_pair <= 0 or float(row["n_pair"]) != float(n_pair):
        raise BinderError(f"metric pair count is invalid: {metric_id}")
    for field in ("value", "numerator", "denominator", "ci_low", "ci_high"):
        value = row[field]
        if value is not None and not math.isfinite(float(value)):
            raise BinderError(f"metric field is not finite: {metric_id}.{field}")
    if not isinstance(row["weighting"], str) or not row["weighting"]:
        raise BinderError(f"metric has no weighting: {metric_id}")
    expected_weighting = ESTIMAND_SPECS[metric_id]["weighting"]
    if row["weighting"] != expected_weighting:
        raise BinderError(
            f"metric weighting differs from frozen estimand: {metric_id}"
        )
    if metric_id.startswith("misspecification.item_type."):
        claim_class = "exploratory_generator_diagnostic"
    elif metric_id in POSTHOC_METRIC_IDS:
        claim_class = "exploratory_posthoc"
    else:
        claim_class = "confirmatory"
    return {
        **dict(row),
        "support_id": support["support_id"],
        "target_set_hash": support["target_set_hash"],
        "target_roster": list(support["target_roster"]),
        "filter_dimensions": _filter_dimensions(metric_id, support),
        "filter_predicate": _filter_predicate(metric_id, support),
        "weighting": row["weighting"],
        "denominator_kind": ESTIMAND_SPECS[metric_id]["denominator_kind"],
        "denominator_definition": _denominator_definition(metric_id, row),
        "claim_class": claim_class,
        "post_hoc": claim_class == "exploratory_posthoc",
        "source_hashes": dict(source_hashes),
    }


def assert_comparable(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    fields = (
        "target_set_hash",
        "target_roster",
        "filter_dimensions",
        "weighting",
        "denominator_definition",
    )
    mismatched = [field for field in fields if left.get(field) != right.get(field)]
    if mismatched:
        raise BinderError(
            "cross-support or non-identical estimand comparison rejected: "
            + ", ".join(mismatched)
        )


def _assert_same_support(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    fields = ("support_id", "target_set_hash", "target_roster")
    mismatched = [field for field in fields if left.get(field) != right.get(field)]
    if mismatched:
        raise BinderError(
            "cross-support comparison rejected: " + ", ".join(mismatched)
        )


def _same_support_pairs(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pair_id, spec in PAIR_SPECS.items():
        left = metrics[str(spec["left"])]
        right = metrics[str(spec["right"])]
        assert_comparable(left, right)
        if float(left["denominator"]) != float(right["denominator"]):
            raise BinderError(f"same-support pair denominator mismatch: {pair_id}")
        if int(left["n_pair"]) != int(right["n_pair"]):
            raise BinderError(f"same-support pair count mismatch: {pair_id}")
        difference_id = spec["difference"]
        if difference_id is None:
            difference = {
                "metric_id": None,
                "value": float(left["value"]) - float(right["value"]),
                "numerator": float(left["numerator"]) - float(right["numerator"]),
                "denominator": left["denominator"],
                "n_pair": left["n_pair"],
                "filter_dimensions": left["filter_dimensions"],
                "weighting": "audit_arithmetic_only",
                "denominator_definition": left["denominator_definition"],
                "ci_low": None,
                "ci_high": None,
                "status": "audit_arithmetic_only_no_bound_bootstrap_ci",
            }
        else:
            bound_difference = metrics[str(difference_id)]
            _assert_same_support(left, bound_difference)
            if float(bound_difference["denominator"]) != float(left["denominator"]):
                raise BinderError(
                    f"paired difference denominator mismatch: {pair_id}"
                )
            if int(bound_difference["n_pair"]) != int(left["n_pair"]):
                raise BinderError(f"paired difference pair count mismatch: {pair_id}")
            if bound_difference["filter_dimensions"] != left["filter_dimensions"]:
                raise BinderError(
                    f"paired difference filter dimensions mismatch: {pair_id}"
                )
            difference = {
                key: bound_difference[key]
                for key in (
                    "metric_id",
                    "value",
                    "numerator",
                    "denominator",
                    "n_pair",
                    "filter_dimensions",
                    "weighting",
                    "denominator_definition",
                    "ci_low",
                    "ci_high",
                )
            }
            difference["status"] = "bound_paired_metric"
        output[pair_id] = {
            "pair_id": pair_id,
            "comparison_status": "bound",
            "support_id": left["support_id"],
            "target_set_hash": left["target_set_hash"],
            "target_roster": left["target_roster"],
            "filter_predicate": left["filter_predicate"],
            "weighting": left["weighting"],
            "denominator_definition": left["denominator_definition"],
            "left_metric_id": left["metric_id"],
            "right_metric_id": right["metric_id"],
            "left_value": left["value"],
            "right_value": right["value"],
            "difference": difference,
            "claim_class": spec["claim_class"],
            "post_hoc": spec["claim_class"] == "exploratory_posthoc",
            "source_hashes": left["source_hashes"],
        }
    return output


def _persona_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BinderError(f"Persona-v2 {label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise BinderError(f"Persona-v2 {label} is not finite")
    if minimum is not None and number < minimum:
        raise BinderError(f"Persona-v2 {label} is below its valid range")
    if maximum is not None and number > maximum:
        raise BinderError(f"Persona-v2 {label} is above its valid range")
    return number


def _persona_integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BinderError(f"Persona-v2 {label} is not an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise BinderError(f"Persona-v2 {label} is outside its valid range")
    return value


def _persona_same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _validate_persona_ci(
    value: Any,
    *,
    point: Any,
    label: str,
) -> list[float] | None:
    if point is None:
        if value is not None:
            raise BinderError(f"Persona-v2 {label} CI exists without an estimate")
        return None
    estimate = _persona_number(point, label=f"{label} estimate")
    if not isinstance(value, list) or len(value) != 2:
        raise BinderError(f"Persona-v2 {label} CI is invalid")
    low = _persona_number(value[0], label=f"{label} CI low")
    high = _persona_number(value[1], label=f"{label} CI high")
    if low > estimate or estimate > high:
        raise BinderError(f"Persona-v2 {label} CI does not contain its estimate")
    return [low, high]


def _validate_persona_bootstrap(
    value: Any,
    *,
    label: str,
    point: Any,
    provider_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BinderError(f"Persona-v2 {label} bootstrap is missing")
    _assert_exact_keys(
        value,
        {
            "point_estimate",
            "ci95",
            "seed",
            "resamples",
            "defined_resamples",
            "undefined_resamples",
            "provider_equal_weighting",
            "provider_point_estimates",
        },
        f"Persona-v2 {label} bootstrap",
    )
    if (
        value.get("seed") != 2026071503
        or value.get("resamples") != 10000
        or value.get("provider_equal_weighting") is not True
    ):
        raise BinderError(f"Persona-v2 {label} bootstrap contract drifted")
    defined = _persona_integer(
        value.get("defined_resamples"),
        label=f"{label} defined resamples",
        maximum=10000,
    )
    undefined = _persona_integer(
        value.get("undefined_resamples"),
        label=f"{label} undefined resamples",
        maximum=10000,
    )
    if defined + undefined != 10000:
        raise BinderError(f"Persona-v2 {label} bootstrap resamples do not reconcile")
    if not _persona_same_number(value.get("point_estimate"), point):
        raise BinderError(f"Persona-v2 {label} bootstrap estimate drifted")
    _validate_persona_ci(value.get("ci95"), point=point, label=f"{label} bootstrap")
    points = value.get("provider_point_estimates")
    if not isinstance(points, Mapping):
        raise BinderError(f"Persona-v2 {label} provider points are missing")
    if provider_keys is not None and set(points) != provider_keys:
        raise BinderError(f"Persona-v2 {label} provider-point roster drifted")
    for provider, provider_point in points.items():
        if not isinstance(provider, str) or not provider:
            raise BinderError(f"Persona-v2 {label} provider-point key is invalid")
        if provider_point is not None:
            _persona_number(provider_point, label=f"{label} provider point")
    return dict(value)


def _persona_asset(
    *,
    path: Path,
    root: Path,
    digest: str,
    size: int,
) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "source_path": str(path.resolve()),
        "sha256": digest,
        "bytes": size,
    }


def _source_artifact(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "relative_path": relative_path or path.name,
        "source_path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _validate_persona_controlled(
    value: Any,
    *,
    providers: Sequence[str],
    lifecycle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BinderError("Persona-v2 controlled result surface is missing")
    _assert_exact_keys(
        value,
        {"eligible_providers", "excluded_providers", "composition", "paired_effects"},
        "Persona-v2 controlled results",
    )
    provider_list = list(providers)
    eligible = [
        provider
        for provider in provider_list
        if lifecycle[provider].get("controlled_eligible") is True
    ]
    excluded = [provider for provider in provider_list if provider not in eligible]
    if value.get("eligible_providers") != eligible or value.get("excluded_providers") != excluded:
        raise BinderError("Persona-v2 controlled eligibility differs from lifecycle")

    composition = value.get("composition")
    if not isinstance(composition, Mapping):
        raise BinderError("Persona-v2 controlled composition is missing")
    _assert_exact_keys(
        composition,
        {
            "states",
            "expected_tasks_per_provider",
            "by_provider",
            "aggregate_counts",
            "all_provider_counts",
        },
        "Persona-v2 controlled composition",
    )
    if composition.get("states") != list(PERSONA_CONTROLLED_STATES):
        raise BinderError("Persona-v2 controlled state roster drifted")
    by_provider = composition.get("by_provider")
    if not isinstance(by_provider, list) or [row.get("provider") for row in by_provider if isinstance(row, Mapping)] != provider_list:
        raise BinderError("Persona-v2 controlled provider roster drifted")
    recomputed_all = {state: 0 for state in PERSONA_CONTROLLED_STATES}
    recomputed_eligible = {state: 0 for state in PERSONA_CONTROLLED_STATES}
    expected_per_provider: int | None = None
    for provider_row in by_provider:
        if not isinstance(provider_row, Mapping):
            raise BinderError("Persona-v2 controlled provider row is invalid")
        _assert_exact_keys(provider_row, {"provider", "arms"}, "Persona-v2 controlled provider row")
        provider = str(provider_row["provider"])
        arms = provider_row.get("arms")
        if not isinstance(arms, list) or [row.get("response_arm") for row in arms if isinstance(row, Mapping)] != ["deficit", "control"]:
            raise BinderError("Persona-v2 controlled arm roster drifted")
        provider_total = 0
        for arm_row in arms:
            if not isinstance(arm_row, Mapping):
                raise BinderError("Persona-v2 controlled arm row is invalid")
            _assert_exact_keys(
                arm_row,
                {
                    "response_arm",
                    "expected_denominator",
                    "counts",
                    "rates",
                    "conditional_answer_accuracy",
                    "conditional_answer_denominator",
                },
                "Persona-v2 controlled arm row",
            )
            denominator = _persona_integer(
                arm_row.get("expected_denominator"), label="controlled arm denominator"
            )
            counts = arm_row.get("counts")
            rates = arm_row.get("rates")
            if not isinstance(counts, Mapping) or set(counts) != set(PERSONA_CONTROLLED_STATES):
                raise BinderError("Persona-v2 controlled count state set drifted")
            if not isinstance(rates, Mapping) or set(rates) != set(PERSONA_CONTROLLED_STATES):
                raise BinderError("Persona-v2 controlled rate state set drifted")
            normalized_counts = {
                state: _persona_integer(counts[state], label=f"controlled {state} count")
                for state in PERSONA_CONTROLLED_STATES
            }
            if sum(normalized_counts.values()) != denominator:
                raise BinderError("Persona-v2 controlled count denominator does not reconcile")
            for state, count in normalized_counts.items():
                rate = _persona_number(
                    rates[state], label=f"controlled {state} rate", minimum=0.0, maximum=1.0
                )
                if not math.isclose(rate, count / denominator, abs_tol=1e-12):
                    raise BinderError("Persona-v2 controlled rate denominator does not reconcile")
                recomputed_all[state] += count
                if provider in eligible:
                    recomputed_eligible[state] += count
            answered = normalized_counts["correct_answer"] + normalized_counts["incorrect_answer"]
            if arm_row.get("conditional_answer_denominator") != answered:
                raise BinderError("Persona-v2 controlled conditional denominator drifted")
            expected_accuracy = normalized_counts["correct_answer"] / answered if answered else None
            if not _persona_same_number(arm_row.get("conditional_answer_accuracy"), expected_accuracy):
                raise BinderError("Persona-v2 controlled conditional accuracy drifted")
            provider_total += denominator
        if expected_per_provider is None:
            expected_per_provider = provider_total
        elif provider_total != expected_per_provider:
            raise BinderError("Persona-v2 controlled provider denominators differ")
    if (
        composition.get("expected_tasks_per_provider") != expected_per_provider
        or composition.get("all_provider_counts") != recomputed_all
        or composition.get("aggregate_counts") != recomputed_eligible
    ):
        raise BinderError("Persona-v2 controlled composition totals do not reconcile")

    effects = value.get("paired_effects")
    if not isinstance(effects, list) or len(effects) != len(PERSONA_EFFECT_ORIENTATIONS):
        raise BinderError("Persona-v2 controlled paired effect table is incomplete")
    effect_index: dict[str, dict[str, Any]] = {}
    for effect in effects:
        if not isinstance(effect, Mapping):
            raise BinderError("Persona-v2 controlled paired effect row is invalid")
        _assert_exact_keys(
            effect,
            {
                "metric_id",
                "orientation",
                "estimate",
                "ci95",
                "eligible_providers",
                "paired_persona_denominators",
                "paired_persona_denominator_range",
                "by_provider",
                "bootstrap",
            },
            "Persona-v2 controlled paired effect row",
        )
        metric = str(effect.get("metric_id") or "")
        if metric in effect_index or PERSONA_EFFECT_ORIENTATIONS.get(metric) != effect.get("orientation"):
            raise BinderError("Persona-v2 controlled paired effect metric or orientation drifted")
        if effect.get("eligible_providers") != eligible:
            raise BinderError("Persona-v2 controlled paired effect eligibility drifted")
        estimate = effect.get("estimate")
        if estimate is not None:
            _persona_number(estimate, label=f"controlled {metric} estimate", minimum=-1.0, maximum=1.0)
        _validate_persona_ci(effect.get("ci95"), point=estimate, label=f"controlled {metric}")
        denominators = effect.get("paired_persona_denominators")
        if not isinstance(denominators, Mapping) or set(denominators) != set(eligible):
            raise BinderError("Persona-v2 controlled paired denominator roster drifted")
        normalized_denominators = {
            provider: _persona_integer(
                denominators[provider],
                label=f"controlled {metric} paired persona denominator",
                maximum=50,
            )
            for provider in eligible
        }
        expected_range = (
            [min(normalized_denominators.values()), max(normalized_denominators.values())]
            if normalized_denominators
            else None
        )
        if effect.get("paired_persona_denominator_range") != expected_range:
            raise BinderError("Persona-v2 controlled paired denominator range drifted")
        provider_rows = effect.get("by_provider")
        if not isinstance(provider_rows, list) or [row.get("provider") for row in provider_rows if isinstance(row, Mapping)] != provider_list:
            raise BinderError("Persona-v2 controlled provider effect roster drifted")
        for provider_row in provider_rows:
            if not isinstance(provider_row, Mapping):
                raise BinderError("Persona-v2 controlled provider effect row is invalid")
            _assert_exact_keys(
                provider_row,
                {
                    "provider",
                    "included_in_aggregate",
                    "estimate",
                    "ci95",
                    "paired_persona_denominator",
                    "bootstrap",
                },
                "Persona-v2 controlled provider effect row",
            )
            provider = str(provider_row["provider"])
            if provider_row.get("included_in_aggregate") is not (provider in eligible):
                raise BinderError("Persona-v2 controlled provider effect inclusion drifted")
            denominator = _persona_integer(
                provider_row.get("paired_persona_denominator"),
                label=f"controlled {metric} provider denominator",
                maximum=50,
            )
            if provider in eligible and denominator != normalized_denominators[provider]:
                raise BinderError("Persona-v2 controlled provider denominator drifted")
            provider_estimate = provider_row.get("estimate")
            if provider_estimate is not None:
                _persona_number(provider_estimate, label=f"controlled {metric} provider estimate", minimum=-1.0, maximum=1.0)
            _validate_persona_ci(provider_row.get("ci95"), point=provider_estimate, label=f"controlled {metric} provider")
            _validate_persona_bootstrap(
                provider_row.get("bootstrap"),
                label=f"controlled {metric} {provider}",
                point=provider_estimate,
                provider_keys={provider},
            )
        _validate_persona_bootstrap(
            effect.get("bootstrap"),
            label=f"controlled {metric} aggregate",
            point=estimate,
            provider_keys=set(eligible),
        )
        effect_index[metric] = dict(effect)
    if set(effect_index) != set(PERSONA_EFFECT_ORIENTATIONS):
        raise BinderError("Persona-v2 controlled paired effect metric set drifted")
    return dict(value)


def _validate_persona_blind(
    value: Any,
    *,
    providers: Sequence[str],
    lifecycle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BinderError("Persona-v2 blind result surface is missing")
    _assert_exact_keys(
        value,
        {
            "primary_terminal_definition",
            "terminal_subject_count",
            "terminal_categories",
            "provider_schema",
            "eligible_providers",
            "excluded_providers",
            "agreement",
            "multi_provider_descriptive",
            "technical_or_schema_failure_rate",
            "stability",
            "stability_provider_equal_aggregate",
        },
        "Persona-v2 blind results",
    )
    provider_list = list(providers)
    eligible = [
        provider
        for provider in provider_list
        if lifecycle[provider].get("blind_eligible") is True
    ]
    excluded = [provider for provider in provider_list if provider not in eligible]
    if (
        value.get("primary_terminal_definition") != "frozen_final_blind_item"
        or value.get("terminal_subject_count") != 100
        or value.get("eligible_providers") != eligible
        or value.get("excluded_providers") != excluded
    ):
        raise BinderError("Persona-v2 blind denominator or lifecycle eligibility drifted")
    categories = value.get("terminal_categories")
    if not isinstance(categories, list) or "NC" not in categories or len(categories) != len(set(categories)):
        raise BinderError("Persona-v2 blind terminal category roster is invalid")

    provider_schema = value.get("provider_schema")
    if not isinstance(provider_schema, Mapping) or set(provider_schema) != set(provider_list):
        raise BinderError("Persona-v2 blind provider schema roster drifted")
    primary_denominator: int | None = None
    for provider in provider_list:
        row = provider_schema[provider]
        if not isinstance(row, Mapping):
            raise BinderError("Persona-v2 blind provider schema row is invalid")
        _assert_exact_keys(
            row,
            {
                "expected_primary_blind_tasks",
                "invalid_schema_count",
                "invalid_schema_fraction",
                "strictly_above_half",
                "invalid_schema_strictly_above_half",
                "excluded_from_blind_aggregate",
                "complete_cluster_count",
                "exclusion_reasons",
            },
            "Persona-v2 blind provider schema row",
        )
        expected = _persona_integer(row.get("expected_primary_blind_tasks"), label="blind primary denominator", minimum=1)
        invalid = _persona_integer(row.get("invalid_schema_count"), label="blind invalid-schema count", maximum=expected)
        fraction = _persona_number(row.get("invalid_schema_fraction"), label="blind invalid-schema fraction", minimum=0.0, maximum=1.0)
        above = fraction > 0.5
        if (
            not math.isclose(fraction, invalid / expected, abs_tol=1e-12)
            or row.get("strictly_above_half") is not above
            or row.get("invalid_schema_strictly_above_half") is not above
            or row.get("excluded_from_blind_aggregate") is not (provider in excluded)
            or row.get("complete_cluster_count") != lifecycle[provider].get("blind_complete_cluster_count")
            or row.get("exclusion_reasons") != lifecycle[provider].get("blind_exclusion_reasons")
        ):
            raise BinderError("Persona-v2 blind provider schema does not reconcile")
        if primary_denominator is None:
            primary_denominator = expected
        elif expected != primary_denominator:
            raise BinderError("Persona-v2 blind primary denominators differ")

    agreement = value.get("agreement")
    if not isinstance(agreement, Mapping):
        raise BinderError("Persona-v2 blind agreement is missing")
    if agreement.get("status") != ("estimated" if len(eligible) >= 2 else "not_estimable"):
        raise BinderError("Persona-v2 blind agreement status drifted")
    if agreement.get("nc_retained") is not True or agreement.get("categories") != categories:
        raise BinderError("Persona-v2 blind agreement NC policy drifted")
    if set(agreement.get("providers") or ()) != set(eligible):
        raise BinderError("Persona-v2 blind agreement provider roster drifted")
    subjects = agreement.get("subjects")
    if not isinstance(subjects, list) or len(subjects) != 100 or len(subjects) != len(set(subjects)):
        raise BinderError("Persona-v2 blind agreement subject denominator drifted")
    pairs = agreement.get("pairs")
    expected_pairs = len(eligible) * (len(eligible) - 1) // 2
    if not isinstance(pairs, list) or len(pairs) != expected_pairs:
        raise BinderError("Persona-v2 blind agreement pair table is incomplete")
    seen_pairs: set[frozenset[str]] = set()
    for row in pairs:
        if not isinstance(row, Mapping):
            raise BinderError("Persona-v2 blind agreement row is invalid")
        left = str(row.get("provider_left") or "")
        right = str(row.get("provider_right") or "")
        key = frozenset((left, right))
        denominator = _persona_integer(row.get("denominator"), label="blind agreement denominator")
        numerator = _persona_integer(row.get("exact_agreement_numerator"), label="blind agreement numerator", maximum=denominator)
        estimate = _persona_number(row.get("exact_agreement"), label="blind agreement estimate", minimum=0.0, maximum=1.0)
        if (
            left == right
            or left not in eligible
            or right not in eligible
            or key in seen_pairs
            or denominator != 100
            or not math.isclose(estimate, numerator / denominator, abs_tol=1e-12)
        ):
            raise BinderError("Persona-v2 blind agreement denominator or pair drifted")
        kappa = row.get("cohen_kappa")
        if kappa is not None:
            _persona_number(kappa, label="blind Cohen kappa", minimum=-1.0, maximum=1.0)
        _validate_persona_ci(row.get("exact_agreement_ci95"), point=estimate, label="blind agreement")
        _validate_persona_bootstrap(
            row.get("exact_agreement_bootstrap"),
            label=f"blind agreement {left}/{right}",
            point=estimate,
        )
        seen_pairs.add(key)

    descriptive = value.get("multi_provider_descriptive")
    if not isinstance(descriptive, Mapping) or set(descriptive.get("rectangular_providers") or ()) != set(eligible) or descriptive.get("subjects") != 100:
        raise BinderError("Persona-v2 blind descriptive denominator drifted")
    if len(eligible) >= 2:
        numerator = _persona_integer(descriptive.get("unanimous_numerator"), label="blind unanimous numerator", maximum=100)
        fraction = _persona_number(descriptive.get("unanimous_fraction"), label="blind unanimous fraction", minimum=0.0, maximum=1.0)
        if descriptive.get("status") != "estimated" or not math.isclose(fraction, numerator / 100, abs_tol=1e-12):
            raise BinderError("Persona-v2 blind unanimous fraction drifted")

    failure = value.get("technical_or_schema_failure_rate")
    if not isinstance(failure, Mapping):
        raise BinderError("Persona-v2 blind failure-rate surface is missing")
    estimate = failure.get("estimate")
    if estimate is not None:
        _persona_number(estimate, label="blind failure rate", minimum=0.0, maximum=1.0)
    _validate_persona_ci(failure.get("ci95"), point=estimate, label="blind failure rate")
    _validate_persona_bootstrap(
        failure.get("bootstrap"),
        label="blind failure rate",
        point=estimate,
        provider_keys=set(eligible),
    )

    stability = value.get("stability")
    if not isinstance(stability, list) or [row.get("provider") for row in stability if isinstance(row, Mapping)] != provider_list:
        raise BinderError("Persona-v2 blind stability provider roster drifted")
    for row in stability:
        if not isinstance(row, Mapping):
            raise BinderError("Persona-v2 blind stability row is invalid")
        provider = str(row.get("provider") or "")
        expected = _persona_integer(row.get("expected_pairs"), label="blind stability expected pairs")
        answer_denominator = _persona_integer(row.get("answer_agreement_denominator"), label="blind stability answer denominator", maximum=expected)
        answer_numerator = row.get("answer_agreement_numerator")
        answer = row.get("answer_agreement")
        if provider in eligible:
            numerator = _persona_integer(answer_numerator, label="blind stability answer numerator", maximum=answer_denominator)
            answer_value = _persona_number(answer, label="blind stability answer", minimum=0.0, maximum=1.0)
            if row.get("status") != "estimated" or answer_denominator != expected or not math.isclose(answer_value, numerator / answer_denominator, abs_tol=1e-12):
                raise BinderError("Persona-v2 blind stability answer denominator drifted")
        elif answer_numerator is not None or answer is not None or answer_denominator != 0:
            raise BinderError("Persona-v2 blind ineligible stability row is estimable")
        if row.get("excluded_from_blind_aggregate") is not (provider in excluded):
            raise BinderError("Persona-v2 blind stability lifecycle inclusion drifted")
        complete_denominator = _persona_integer(row.get("canonical_complete_pair_denominator"), label="blind stability canonical denominator", maximum=expected)
        complete_numerator = _persona_integer(row.get("canonical_complete_pair_numerator"), label="blind stability canonical numerator", maximum=complete_denominator)
        canonical = row.get("canonical_complete_pair_stability")
        expected_canonical = complete_numerator / complete_denominator if complete_denominator else None
        if not _persona_same_number(canonical, expected_canonical):
            raise BinderError("Persona-v2 blind stability canonical denominator drifted")
        _persona_number(row.get("canonical_itt_yield"), label="blind stability ITT yield", minimum=0.0, maximum=1.0)
        _persona_integer(row.get("nc_nc_agreement_count"), label="blind stability NC count", maximum=expected)
        _validate_persona_bootstrap(row.get("answer_bootstrap"), label=f"blind stability answer {provider}", point=answer, provider_keys={provider} if provider in eligible else set())
        _validate_persona_bootstrap(row.get("canonical_complete_pair_bootstrap"), label=f"blind stability canonical {provider}", point=canonical if provider in eligible else None, provider_keys={provider} if provider in eligible else set())

    aggregate = value.get("stability_provider_equal_aggregate")
    if not isinstance(aggregate, Mapping) or aggregate.get("eligible_providers") != eligible:
        raise BinderError("Persona-v2 blind aggregate stability roster drifted")
    for key in ("answer", "canonical_complete_pair"):
        bootstrap = aggregate.get(key)
        point = bootstrap.get("point_estimate") if isinstance(bootstrap, Mapping) else None
        _validate_persona_bootstrap(
            bootstrap,
            label=f"blind aggregate stability {key}",
            point=point,
            provider_keys=set(eligible),
        )
    return dict(value)


def _validate_persona_judge(
    value: Any, *, allow_fixture: bool = False
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BinderError("Persona-v2 judge adjudication is missing")
    _assert_exact_keys(
        value,
        {
            "case_manifest",
            "analysis",
            "run_evidence_binding",
            "result_manifests",
        },
        "Persona-v2 judge adjudication",
    )
    case_manifest = value.get("case_manifest")
    analysis = value.get("analysis")
    run_evidence_binding = value.get("run_evidence_binding")
    result_manifests = value.get("result_manifests")
    if (
        not isinstance(case_manifest, Mapping)
        or not isinstance(analysis, Mapping)
        or not isinstance(run_evidence_binding, Mapping)
        or not isinstance(result_manifests, Mapping)
    ):
        raise BinderError("Persona-v2 judge adjudication surfaces are incomplete")
    _assert_exact_keys(
        run_evidence_binding,
        {
            "schema_version",
            "judge_run_evidence_receipt_sha256",
            "committed_anchor_sha256",
            "family_slots",
        },
        "Persona-v2 judge run evidence binding",
    )
    family_slots = run_evidence_binding.get("family_slots")
    if (
        run_evidence_binding.get("schema_version")
        != "yher.llm_sim_v2.formal_judge_run_evidence_binding.v1"
        or not isinstance(
            run_evidence_binding.get("judge_run_evidence_receipt_sha256"), str
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(run_evidence_binding.get("judge_run_evidence_receipt_sha256")),
        )
        is None
        or not isinstance(run_evidence_binding.get("committed_anchor_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(run_evidence_binding.get("committed_anchor_sha256")),
        )
        is None
        or not isinstance(family_slots, Mapping)
        or set(family_slots) != {"claude", "gpt"}
        or not all(isinstance(slot, Mapping) for slot in family_slots.values())
    ):
        raise BinderError("Persona-v2 judge run evidence binding is invalid")
    case_payload = dict(case_manifest)
    case_sha = case_payload.pop("case_manifest_sha256", None)
    if (
        case_manifest.get("schema_version")
        != "yher.llm_sim_v2.judge_case_manifest.v2"
        or case_manifest.get("simulated") is not True
        or case_manifest.get("run_id") != "llm-personas-v2-dual"
        or case_manifest.get("analysis_population") != "main"
        or case_manifest.get("exploratory") is not True
        or case_sha != hashlib.sha256(canonical_json_bytes(case_payload)).hexdigest()
    ):
        raise BinderError("Persona-v2 judge case schema v2 or self-hash drifted")
    if (
        case_manifest.get("target_labels_exported") is not False
        or case_manifest.get("target_metadata_exported") is not False
        or case_manifest.get("provider_identity_exported") is not False
        or case_manifest.get("question_field_whitelist")
        != ["kind", "options", "stem_blocks", "stem_text"]
    ):
        raise BinderError("Persona-v2 judge input exported a prohibited identity")
    if not isinstance(case_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", case_sha):
        raise BinderError("Persona-v2 judge case manifest hash is invalid")
    amendment = case_manifest.get("judge_amendment")
    if (
        not isinstance(case_manifest.get("judge_protocol_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(case_manifest.get("judge_protocol_sha256"))
        )
        is None
        or not isinstance(amendment, Mapping)
        or amendment.get("path")
        != "experiments/llm_sim_v2/judge_amendment_20260716.md"
        or not isinstance(amendment.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(amendment.get("sha256"))) is None
        or not isinstance(amendment.get("size"), int)
        or amendment.get("size", 0) <= 0
    ):
        raise BinderError("Persona-v2 judge protocol or amendment binding drifted")
    selected_count = _persona_integer(
        case_manifest.get("selected_count"),
        label="judge selected case count",
        minimum=0,
        maximum=120,
    )
    cases = case_manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != selected_count:
        raise BinderError("Persona-v2 judge selected case roster is incomplete")
    case_ids = [
        str(row.get("case_id") or "")
        for row in cases
        if isinstance(row, Mapping)
    ]
    if (
        len(case_ids) != selected_count
        or any(not value for value in case_ids)
        or len(case_ids) != len(set(case_ids))
    ):
        raise BinderError("Persona-v2 judge selected case IDs are invalid")
    _assert_exact_keys(
        analysis,
        {
            "schema_version",
            "simulated",
            "run_id",
            "exploratory",
            "case_manifest_sha256",
            "selected_count",
            "cases",
            "result_manifest_sha256",
            "execution_receipt_sha256",
            "execution_ids",
            "judge_families",
            "judge_models",
            "judge_transports",
            "judge_accounting",
            "expected_judges",
            "available_judges",
            "missing_judges",
            "status",
            "category_counts",
            "pairwise_label_agreement",
            "pairwise_error_category_agreement",
            "label_disagreement_examples",
            "error_category_disagreement_examples",
        },
        "Persona-v2 judge analysis v2",
    )
    if (
        analysis.get("case_manifest_sha256") != case_sha
        or analysis.get("selected_count") != selected_count
        or analysis.get("cases") != case_ids
    ):
        raise BinderError("Persona-v2 judge analysis/case hash drifted")
    if (
        analysis.get("schema_version") != "yher.llm_sim_v2.judge_analysis.v2"
        or analysis.get("simulated") is not True
        or analysis.get("run_id") != "llm-personas-v2-dual"
        or analysis.get("exploratory") is not True
        or analysis.get("expected_judges") != ["claude", "gpt"]
    ):
        raise BinderError("Persona-v2 judge analysis v2 envelope drifted")
    status = analysis.get("status")
    available = analysis.get("available_judges")
    missing = analysis.get("missing_judges")
    if status == "complete":
        expected_available = ["claude", "gpt"]
        expected_missing: list[str] = []
    elif status == "partial_missing_judge":
        expected_available = ["gpt"]
        expected_missing = ["claude"]
    elif status == "missing_all_judges":
        expected_available = []
        expected_missing = ["claude", "gpt"]
    elif status == "not_applicable_zero_cases":
        expected_available = []
        expected_missing = []
    else:
        raise BinderError("Persona-v2 judge analysis status is invalid")
    if available != expected_available or missing != expected_missing:
        raise BinderError("Persona-v2 judge availability profile drifted")
    if set(result_manifests) != {"claude", "gpt"}:
        raise BinderError("Persona-v2 judge result manifest profile is incomplete")
    available_set = set(expected_available)
    for judge in ("claude", "gpt"):
        embedded = result_manifests.get(judge)
        if judge in available_set:
            if not isinstance(embedded, Mapping):
                raise BinderError("Persona-v2 judge result manifest profile is incomplete")
        elif embedded is not None:
            raise BinderError("Persona-v2 judge result manifest profile has a stray result")
    if status in {
        "complete",
        "partial_missing_judge",
        "missing_all_judges",
    } and selected_count == 0:
        raise BinderError("Persona-v2 applicable judge profile has no selected cases")
    if status == "not_applicable_zero_cases":
        empty_map_fields = (
            "result_manifest_sha256",
            "execution_receipt_sha256",
            "execution_ids",
            "judge_families",
            "judge_models",
            "judge_transports",
            "judge_accounting",
            "category_counts",
        )
        if (
            selected_count != 0
            or cases != []
            or case_manifest.get("selected_stratum_counts")
            != {"disagreement": 0, "agreement": 0}
            or any(analysis.get(field) != {} for field in empty_map_fields)
            or analysis.get("pairwise_label_agreement") is not None
            or analysis.get("pairwise_error_category_agreement") is not None
            or analysis.get("label_disagreement_examples") != []
            or analysis.get("error_category_disagreement_examples") != []
            or result_manifests != {"claude": None, "gpt": None}
        ):
            raise BinderError("Persona-v2 zero-case judge profile drifted")
        return {
            "case_manifest": dict(case_manifest),
            "analysis": dict(analysis),
            "run_evidence_binding": dict(run_evidence_binding),
            "result_manifests": dict(result_manifests),
        }
    if status == "missing_all_judges":
        empty_map_fields = (
            "result_manifest_sha256",
            "execution_receipt_sha256",
            "execution_ids",
            "judge_families",
            "judge_models",
            "judge_transports",
            "judge_accounting",
            "category_counts",
        )
        if (
            any(analysis.get(field) != {} for field in empty_map_fields)
            or analysis.get("pairwise_label_agreement") is not None
            or analysis.get("pairwise_error_category_agreement") is not None
            or analysis.get("label_disagreement_examples") != []
            or analysis.get("error_category_disagreement_examples") != []
            or result_manifests != {"claude": None, "gpt": None}
        ):
            raise BinderError("Persona-v2 missing-all-judges profile drifted")
        return {
            "case_manifest": dict(case_manifest),
            "analysis": dict(analysis),
            "run_evidence_binding": dict(run_evidence_binding),
            "result_manifests": dict(result_manifests),
        }
    for field in ("result_manifest_sha256", "execution_receipt_sha256"):
        hashes = analysis.get(field)
        if not isinstance(hashes, Mapping) or set(hashes) != available_set:
            raise BinderError(f"Persona-v2 judge {field} values are incomplete")
        for digest in hashes.values():
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise BinderError(f"Persona-v2 judge {field} value is invalid")
    execution_ids = analysis.get("execution_ids")
    models = analysis.get("judge_models")
    families = analysis.get("judge_families")
    transports = analysis.get("judge_transports")
    if (
        not isinstance(execution_ids, Mapping)
        or set(execution_ids) != available_set
        or not all(isinstance(value, str) and value for value in execution_ids.values())
        or len(set(execution_ids.values())) != len(available_set)
        or not isinstance(models, Mapping)
        or set(models) != available_set
        or not all(isinstance(value, str) and value for value in models.values())
        or len(set(models.values())) != len(available_set)
        or families != {judge: judge for judge in expected_available}
    ):
        raise BinderError(
            "Persona-v2 judge executions do not prove independent families, models, and IDs"
        )
    expected_transports = {
        judge: (
            "fixture"
            if allow_fixture
            else "claude_cli"
            if judge == "claude"
            else "codex_cli"
        )
        for judge in expected_available
    }
    if transports != expected_transports:
        raise BinderError("Persona-v2 judge transport routes are not independent")
    accounting = analysis.get("judge_accounting")
    if not isinstance(accounting, Mapping) or set(accounting) != available_set:
        raise BinderError("Persona-v2 judge accounting is incomplete")
    accounting_keys = {
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
    }
    for judge, row in accounting.items():
        if not isinstance(row, Mapping) or set(row) != accounting_keys:
            raise BinderError("Persona-v2 judge accounting schema drifted")
        for field in (
            "request_count",
            "retry_count",
            "transport_error_count",
            "schema_error_count",
            "content_retry_count",
            "input_tokens",
            "output_tokens",
        ):
            _persona_integer(
                row.get(field), label=f"judge {judge} accounting {field}"
            )
        if row.get("request_count", 0) <= 0:
            raise BinderError("Persona-v2 judge accounting has no request")
        known = _persona_number(
            row.get("known_cost_yuan"),
            label=f"judge {judge} known cost",
            minimum=0.0,
        )
        reserve = _persona_number(
            row.get("unknown_cost_reserve_yuan"),
            label=f"judge {judge} reserve cost",
            minimum=0.0,
        )
        accounted = _persona_number(
            row.get("accounted_cost_yuan"),
            label=f"judge {judge} accounted cost",
            minimum=0.0,
        )
        if not math.isclose(accounted, known + reserve, abs_tol=1e-8):
            raise BinderError("Persona-v2 judge accounting does not reconcile")
    category_counts = analysis.get("category_counts")
    if not isinstance(category_counts, Mapping) or set(category_counts) != available_set:
        raise BinderError("Persona-v2 judge category counts are incomplete")
    if not isinstance(analysis.get("label_disagreement_examples"), list) or not isinstance(
        analysis.get("error_category_disagreement_examples"), list
    ):
        raise BinderError("Persona-v2 judge disagreement examples are invalid")
    pairwise_names = (
        "pairwise_label_agreement",
        "pairwise_error_category_agreement",
    )
    if status == "partial_missing_judge":
        if any(analysis.get(name) is not None for name in pairwise_names):
            raise BinderError("Persona-v2 missing judge profile reports pairwise agreement")
        if analysis.get("label_disagreement_examples") != [] or analysis.get(
            "error_category_disagreement_examples"
        ) != []:
            raise BinderError("Persona-v2 missing judge profile reports disagreements")
        return {
            "case_manifest": dict(case_manifest),
            "analysis": dict(analysis),
            "run_evidence_binding": dict(run_evidence_binding),
            "result_manifests": dict(result_manifests),
        }
    for name in pairwise_names:
        row = analysis.get(name)
        if not isinstance(row, Mapping):
            raise BinderError(f"Persona-v2 judge {name} is missing")
        denominator = _persona_integer(row.get("denominator"), label=f"judge {name} denominator")
        numerator = _persona_integer(row.get("exact_agreement_numerator"), label=f"judge {name} numerator", maximum=denominator)
        estimate = row.get("exact_agreement")
        if denominator:
            estimate_value = _persona_number(estimate, label=f"judge {name} estimate", minimum=0.0, maximum=1.0)
            if not math.isclose(estimate_value, numerator / denominator, abs_tol=1e-12):
                raise BinderError(f"Persona-v2 judge {name} denominator drifted")
        if name == "pairwise_label_agreement" and denominator != selected_count:
            raise BinderError("Persona-v2 judge label denominator differs from selected cases")
        if name == "pairwise_error_category_agreement":
            total = _persona_integer(
                row.get("total_case_count"), label="judge error-category total"
            )
            missing = _persona_integer(
                row.get("missing_any_count"), label="judge error-category missing"
            )
            if total != selected_count or denominator + missing != total:
                raise BinderError("Persona-v2 judge error-category denominator drifted")
    return {
        "case_manifest": dict(case_manifest),
        "analysis": dict(analysis),
        "run_evidence_binding": dict(run_evidence_binding),
        "result_manifests": dict(result_manifests),
    }


def _validate_persona_judge_sources(
    *,
    input_manifest: Mapping[str, Any],
    result_input_binding: Any,
    judge_results: Mapping[str, Mapping[str, Any]],
    judge_result_paths: Mapping[str, Path],
    judge_adjudication: Mapping[str, Any],
) -> None:
    files = input_manifest.get("files")
    if (
        input_manifest.get("schema_version")
        != "yher.llm_sim_v2.analysis_input_artifact_manifest.v1"
        or input_manifest.get("simulated") is not True
        or input_manifest.get("run_id") != "llm-personas-v2-dual"
        or input_manifest.get("analysis_population") != "main"
        or not isinstance(files, list)
        or input_manifest.get("input_file_count") != len(files)
        or input_manifest.get("input_file_set_sha256")
        != hashlib.sha256(canonical_json_bytes(files)).hexdigest()
    ):
        raise BinderError("Persona-v2 analysis input artifact manifest drifted")
    expected_input_binding = {
        "input_file_count": input_manifest["input_file_count"],
        "record_file_count": input_manifest.get("record_file_count"),
        "input_file_set_sha256": input_manifest["input_file_set_sha256"],
        "input_artifact_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(input_manifest)
        ).hexdigest(),
    }
    if result_input_binding != expected_input_binding:
        raise BinderError("Persona-v2 result/input artifact binding drifted")
    input_index: dict[str, Mapping[str, Any]] = {}
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size"}:
            raise BinderError("Persona-v2 analysis input artifact row is invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or relative in input_index:
            raise BinderError("Persona-v2 analysis input artifact path is invalid")
        input_index[relative] = row

    case_manifest = judge_adjudication["case_manifest"]
    analysis = judge_adjudication["analysis"]
    case_ids = [str(row["case_id"]) for row in case_manifest["cases"]]
    available_judges = tuple(analysis["available_judges"])
    embedded_results = judge_adjudication.get("result_manifests")
    if (
        set(judge_results) != set(available_judges)
        or set(judge_result_paths) != set(available_judges)
        or not isinstance(embedded_results, Mapping)
        or any(
            embedded_results.get(judge) != judge_results[judge]
            for judge in available_judges
        )
    ):
        raise BinderError("Persona-v2 judge result manifest profile differs from analysis")
    for missing_judge in analysis["missing_judges"]:
        if f"judge-results/{missing_judge}.json" in input_index:
            raise BinderError("Persona-v2 missing judge has a bound result artifact")
    attempt_sets: dict[str, set[str]] = {}
    raw_sets: dict[str, set[str]] = {}
    for judge in available_judges:
        result_manifest = judge_results[judge]
        payload = dict(result_manifest)
        advertised_result_sha = payload.pop("judge_result_manifest_sha256", None)
        receipt = result_manifest.get("execution_receipt")
        results = result_manifest.get("results")
        if (
            result_manifest.get("schema_version")
            != "yher.llm_sim_v2.judge_result_manifest.v2"
            or result_manifest.get("simulated") is not True
            or result_manifest.get("run_id") != "llm-personas-v2-dual"
            or result_manifest.get("judge") != judge
            or result_manifest.get("case_manifest_sha256")
            != case_manifest["case_manifest_sha256"]
            or advertised_result_sha
            != hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            or advertised_result_sha
            != analysis["result_manifest_sha256"][judge]
            or not isinstance(receipt, Mapping)
            or not isinstance(results, list)
            or [
                str(row.get("case_id") or "")
                for row in results
                if isinstance(row, Mapping)
            ]
            != case_ids
        ):
            raise BinderError(f"Persona-v2 {judge} judge result manifest drifted")
        receipt_payload = dict(receipt)
        advertised_receipt_sha = receipt_payload.pop(
            "execution_receipt_sha256", None
        )
        identity = receipt.get("identity")
        attempts = receipt.get("ordered_attempt_ids")
        raw_artifacts = receipt.get("raw_artifacts")
        if (
            receipt.get("schema_version")
            != "yher.llm_sim_v2.judge_execution_receipt.v2"
            or advertised_receipt_sha
            != hashlib.sha256(canonical_json_bytes(receipt_payload)).hexdigest()
            or advertised_receipt_sha
            != analysis["execution_receipt_sha256"][judge]
            or not isinstance(identity, Mapping)
            or identity.get("judge_family") != analysis["judge_families"][judge]
            or identity.get("requested_model") != analysis["judge_models"][judge]
            or identity.get("transport") != analysis["judge_transports"][judge]
            or identity.get("execution_id") != analysis["execution_ids"][judge]
            or receipt.get("accounting") != analysis["judge_accounting"][judge]
            or not isinstance(attempts, list)
            or not attempts
            or len(attempts) != len(set(attempts))
            or not all(isinstance(value, str) and value for value in attempts)
            or not isinstance(raw_artifacts, list)
            or not raw_artifacts
        ):
            raise BinderError(f"Persona-v2 {judge} judge execution receipt drifted")
        raw_hashes: set[str] = set()
        for artifact in raw_artifacts:
            digest = artifact.get("sha256") if isinstance(artifact, Mapping) else None
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise BinderError(
                    f"Persona-v2 {judge} judge raw artifact binding drifted"
                )
            raw_hashes.add(digest)
        if len(raw_hashes) != len(raw_artifacts):
            raise BinderError(f"Persona-v2 {judge} judge raw artifact set repeats")
        attempt_sets[judge] = set(attempts)
        raw_sets[judge] = raw_hashes
        relative = f"judge-results/{judge}.json"
        input_row = input_index.get(relative)
        result_path = judge_result_paths[judge]
        if (
            input_row is None
            or input_row.get("sha256") != sha256_file(result_path)
            or input_row.get("size") != result_path.stat().st_size
        ):
            raise BinderError(
                f"Persona-v2 {judge} judge result is not hash-bound by analysis inputs"
            )
    if len(available_judges) == 2:
        left, right = available_judges
        if attempt_sets[left] & attempt_sets[right]:
            raise BinderError("Persona-v2 judge attempt artifact sets are not independent")
        if raw_sets[left] & raw_sets[right]:
            raise BinderError("Persona-v2 judge raw artifact sets are not independent")


def _validate_persona_judge_run_snapshot(
    *,
    root: Path,
    snapshot_manifest: Mapping[str, Any],
    snapshot_manifest_path: Path,
    judge_results: Mapping[str, Mapping[str, Any]],
    judge_analysis: Mapping[str, Any],
    allow_fixture: bool = False,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    expected_manifest_path = root / "judge-snapshots/snapshot_manifest.json"
    try:
        actual_manifest_path = snapshot_manifest_path.resolve(strict=True)
    except OSError as exc:
        raise BinderError("Persona-v2 judge run snapshot manifest is missing") from exc
    if actual_manifest_path != expected_manifest_path.resolve(strict=False):
        raise BinderError("Persona-v2 judge run snapshot must use its fixed bundle path")

    from experiments.llm_sim_v2.analyze import (
        AnalysisContractError,
        validate_judge_run_execution_snapshot,
    )

    try:
        validated = validate_judge_run_execution_snapshot(
            snapshot_manifest,
            snapshot_root=snapshot_manifest_path.parent,
            allow_fixture=allow_fixture,
        )
    except AnalysisContractError as exc:
        raise BinderError(f"Persona-v2 judge run snapshot cannot replay: {exc}") from exc
    if validated != dict(snapshot_manifest):
        raise BinderError("Persona-v2 judge run snapshot changed during replay")

    available_judges = tuple(judge_analysis["available_judges"])
    if set(judge_results) != set(available_judges):
        raise BinderError("Persona-v2 judge run snapshot result profile drifted")
    expected_statuses = {
        "complete": {"claude": "complete", "gpt": "complete"},
        "partial_missing_judge": {"claude": "unavailable", "gpt": "complete"},
        "missing_all_judges": {"claude": "unavailable", "gpt": "failed"},
        "not_applicable_zero_cases": {
            "claude": "not_applicable_zero_cases",
            "gpt": "not_applicable_zero_cases",
        },
    }
    status = str(judge_analysis.get("status") or "")
    family_slots = snapshot_manifest.get("family_slots")
    if status not in expected_statuses or not isinstance(family_slots, Mapping):
        raise BinderError("Persona-v2 judge run snapshot profile is invalid")
    actual_statuses = {
        family: slot.get("status") if isinstance(slot, Mapping) else None
        for family, slot in family_slots.items()
    }
    if actual_statuses != expected_statuses[status]:
        raise BinderError("Persona-v2 judge run snapshot family slots drifted")

    for judge in available_judges:
        slot = family_slots[judge]
        result_manifest = judge_results[judge]
        receipt = result_manifest.get("execution_receipt")
        if not isinstance(slot, Mapping) or not isinstance(receipt, Mapping):
            raise BinderError(f"Persona-v2 {judge} judge run slot is invalid")
        identity = receipt.get("identity")
        if not isinstance(identity, Mapping):
            raise BinderError(f"Persona-v2 {judge} judge receipt identity is invalid")
        if (
            result_manifest.get("execution_receipt_path") != slot.get("receipt_path")
            or slot.get("receipt_sha256")
            != judge_analysis["execution_receipt_sha256"][judge]
            or slot.get("execution_id") != judge_analysis["execution_ids"][judge]
            or slot.get("requested_model") != judge_analysis["judge_models"][judge]
            or slot.get("transport") != judge_analysis["judge_transports"][judge]
            or slot.get("accounting") != judge_analysis["judge_accounting"][judge]
            or identity.get("execution_id") != slot.get("execution_id")
            or identity.get("requested_model") != slot.get("requested_model")
            or identity.get("transport") != slot.get("transport")
        ):
            raise BinderError(
                f"Persona-v2 {judge} judge result differs from the finalized run slot"
            )

    expected_artifact_paths = {"judge-snapshots/snapshot_manifest.json"}
    bound_files: dict[str, dict[str, Any]] = {}
    files = snapshot_manifest.get("files")
    if not isinstance(files, list):
        raise BinderError("Persona-v2 judge run snapshot file roster is invalid")
    for row in files:
        if not isinstance(row, Mapping):
            raise BinderError("Persona-v2 judge run snapshot row is invalid")
        relative = str(row.get("path") or "")
        artifact_relative = f"judge-snapshots/{relative}"
        path = snapshot_manifest_path.parent / relative
        expected_artifact_paths.add(artifact_relative)
        bound_files[artifact_relative] = {
            "relative_path": artifact_relative,
            "source_path": str(path.resolve()),
            "sha256": str(row["sha256"]),
            "bytes": int(row["size"]),
            "role": "judge_run_evidence",
            "judge": None,
        }
    return expected_artifact_paths, bound_files


def _validate_persona_judge_snapshot_input_binding(
    *,
    input_manifest: Mapping[str, Any],
    snapshot_manifest: Mapping[str, Any],
) -> None:
    input_files = input_manifest.get("files")
    snapshot_files = snapshot_manifest.get("files")
    if not isinstance(input_files, list) or not isinstance(snapshot_files, list):
        raise BinderError("Persona-v2 judge input/snapshot rosters are invalid")
    actual = {
        str(row["path"]): row
        for row in input_files
        if isinstance(row, Mapping)
        and str(row.get("path") or "").startswith("judge-results/")
    }
    expected: dict[str, Mapping[str, Any]] = {}
    for row in snapshot_files:
        if not isinstance(row, Mapping):
            raise BinderError("Persona-v2 judge input/snapshot rosters are invalid")
        snapshot_path = str(row.get("path") or "")
        if not snapshot_path.startswith("run/"):
            raise BinderError("Persona-v2 judge input/snapshot path is invalid")
        expected[f"judge-results/{snapshot_path.removeprefix('run/')}"] = row
    if set(actual) != set(expected):
        raise BinderError("Persona-v2 judge input roster differs from canonical snapshot")
    for relative, snapshot_row in expected.items():
        input_row = actual[relative]
        if (
            input_row.get("sha256") != snapshot_row.get("sha256")
            or input_row.get("size") != snapshot_row.get("size")
        ):
            raise BinderError(
                "Persona-v2 judge input bytes differ from canonical snapshot"
            )


def persona_v2_slot() -> dict[str, Any]:
    return {
        "schema_version": "yher.journal_binder.persona_v2_pending.v1",
        "status": "pending_formal_w3_artifacts",
        "value": None,
        "metric_registry_sha256": None,
        "target_set_hash": None,
        "persona_cluster_count": 50,
        "independent_unit": "persona_id",
        "provider_role": "repeated_measurement",
        "required_run_id": "llm-personas-v2-dual",
        "accepts_unbound_values": False,
    }


def bind_persona_v2_value(slot: Mapping[str, Any], value: Any) -> Mapping[str, Any]:
    if slot.get("status") == "pending_formal_w3_artifacts" or value is not None:
        raise BinderError(
            "Persona-v2 values cannot be bound before formal W3 artifacts exist"
        )
    return slot


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, label: str
) -> str:
    advertised = payload.get(field)
    copy = dict(payload)
    copy.pop(field, None)
    computed = hashlib.sha256(canonical_json_bytes(copy)).hexdigest()
    if advertised != computed:
        raise BinderError(f"{label} self-hash drift")
    return computed


def _read_relative_regular_file(
    root: Path, relative: Path, *, label: str
) -> tuple[Path, bytes]:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise BinderError(f"{label} path escapes its root")
    if root.is_symlink() or not root.is_dir():
        raise BinderError(f"{label} root is missing, unsafe, or a symlink")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        descriptor = os.open(root, directory_flags)
        descriptors.append(descriptor)
        for part in relative.parts[:-1]:
            descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        file_descriptor = os.open(
            relative.parts[-1], file_flags, dir_fd=descriptor
        )
        descriptors.append(file_descriptor)
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise BinderError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise BinderError(f"{label} is missing") from exc
    except OSError as exc:
        raise BinderError(f"{label} is unsafe or contains a symlink") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return root / relative, b"".join(chunks)


def _load_bound_bundle_file(
    root: Path, entry: Mapping[str, Any], *, role: str
) -> tuple[Path, Mapping[str, Any], str]:
    _assert_exact_keys(entry, {"path", "sha256"}, f"Persona-v2 bundle {role}")
    path_value = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise BinderError(f"Persona-v2 bundle file path is invalid: {role}")
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise BinderError(f"Persona-v2 bundle file path escapes root: {role}")
    path, data = _read_relative_regular_file(
        root, relative, label=f"Persona-v2 bundle file {role}"
    )
    actual_hash = hashlib.sha256(data).hexdigest()
    if expected_hash != actual_hash:
        raise BinderError(f"Persona-v2 bundle file SHA-256 drift: {role}")
    payload = _strict_json_bytes(data, label=f"Persona-v2 {role}")
    if not isinstance(payload, Mapping):
        raise BinderError(f"Persona-v2 {role} root must be an object")
    return path, payload, actual_hash


def bind_persona_v2_artifacts(
    bundle_dir: Path | str, *, allow_fixture: bool = False
) -> dict[str, Any]:
    root = Path(bundle_dir)
    binding_path, binding_bytes = _read_relative_regular_file(
        root,
        Path("binding_manifest.json"),
        label="Persona-v2 formal W3 binding manifest",
    )
    binding = _strict_json_bytes(
        binding_bytes, label="Persona-v2 binding manifest"
    )
    if not isinstance(binding, Mapping):
        raise BinderError("Persona-v2 binding manifest root must be an object")
    _assert_exact_keys(
        binding,
        {"schema_version", "simulated", "run_id", "analysis_population", "files"},
        "Persona-v2 binding manifest",
    )
    if (
        binding.get("schema_version")
        != "yher.journal_binder.persona_v2_bundle.v1"
        or binding.get("simulated") is not True
        or binding.get("run_id") != "llm-personas-v2-dual"
        or binding.get("analysis_population") != "main"
    ):
        raise BinderError("Persona-v2 binding manifest envelope is invalid")
    files = binding.get("files")
    base_roles = {
        "analysis_results",
        "analysis_artifact_manifest",
        "analysis_input_artifact_manifest",
        "phase_provenance",
        "runtime_task_manifest",
        "mapping_manifest",
    }
    snapshot_roles = {"judge_run_execution_snapshot_manifest"}
    gpt_roles = {"gpt_judge_result_manifest"}
    claude_roles = {"claude_judge_result_manifest"}
    allowed_role_sets = (
        base_roles | snapshot_roles,
        base_roles | snapshot_roles | gpt_roles,
        base_roles | snapshot_roles | gpt_roles | claude_roles,
    )
    if not isinstance(files, Mapping) or set(files) not in allowed_role_sets:
        raise BinderError("Persona-v2 binding manifest file set is incomplete")
    expected_roles = set(files)
    canonical_role_paths = {
        "analysis_results": "analysis/analysis_results.json",
        "analysis_artifact_manifest": "analysis/artifact_manifest.json",
        "analysis_input_artifact_manifest": (
            "analysis/input_artifact_manifest.json"
        ),
        "phase_provenance": "evidence/phase_provenance.json",
        "runtime_task_manifest": "evidence/runtime_task_manifest.json",
        "mapping_manifest": "evidence/target_option_mapping.json",
        "judge_run_execution_snapshot_manifest": (
            "judge-snapshots/snapshot_manifest.json"
        ),
        "gpt_judge_result_manifest": "judge-snapshots/run/gpt.json",
        "claude_judge_result_manifest": "judge-snapshots/run/claude.json",
    }
    for role in expected_roles:
        descriptor = files.get(role)
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("path") != canonical_role_paths[role]
        ):
            raise BinderError(f"Persona-v2 {role} must use its canonical fixed path")
    role_judges = {
        judge
        for judge, roles in (("gpt", gpt_roles), ("claude", claude_roles))
        if roles <= expected_roles
    }
    loaded: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    file_hashes: dict[str, str] = {}
    for role in sorted(expected_roles):
        entry = files.get(role)
        if not isinstance(entry, Mapping):
            raise BinderError(f"Persona-v2 bundle file entry is invalid: {role}")
        path, payload, digest = _load_bound_bundle_file(root, entry, role=role)
        paths[role] = path
        loaded[role] = payload
        file_hashes[f"{role}_sha256"] = digest

    runtime = loaded["runtime_task_manifest"]
    if (
        runtime.get("schema_version")
        != "yher.llm_sim_v2.runtime_task_manifest.v1"
        or runtime.get("simulated") is not True
        or runtime.get("run_id") != "llm-personas-v2-dual"
    ):
        raise BinderError("Persona-v2 runtime task manifest envelope is invalid")
    runtime_hash = _verify_self_hash(
        runtime,
        field="runtime_task_manifest_sha256",
        label="Persona-v2 runtime task manifest",
    )
    phases = runtime.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != {"pilot", "main"}:
        raise BinderError("Persona-v2 runtime manifest must bind pilot and main phases")
    phase_tasks: dict[str, list[str]] = {}
    phase_providers: dict[str, list[str]] = {}
    for phase_name in ("pilot", "main"):
        phase_row = phases.get(phase_name)
        if not isinstance(phase_row, Mapping):
            raise BinderError(f"Persona-v2 runtime {phase_name} phase is invalid")
        task_ids = phase_row.get("task_ids")
        providers = phase_row.get("providers")
        if (
            not isinstance(task_ids, list)
            or not all(isinstance(value, str) and value for value in task_ids)
            or len(task_ids) != len(set(task_ids))
            or phase_row.get("task_count") != len(task_ids)
            or not isinstance(providers, list)
            or not all(isinstance(value, str) and value for value in providers)
            or len(providers) != len(set(providers))
        ):
            raise BinderError(f"Persona-v2 runtime {phase_name} roster is invalid")
        phase_tasks[phase_name] = list(task_ids)
        phase_providers[phase_name] = list(providers)
    if set(phase_tasks["pilot"]) & set(phase_tasks["main"]):
        raise BinderError("Persona-v2 pilot and main task rosters must be disjoint")

    phase = loaded["phase_provenance"]
    if (
        phase.get("schema_version") != "yher.llm_sim_v2.phase_provenance.v1"
        or phase.get("simulated") is not True
        or phase.get("run_id") != "llm-personas-v2-dual"
        or phase.get("phase") != "main"
        or phase.get("analysis_population") != "main"
        or phase.get("collection_mode") != "formal"
        or phase.get("development_only") is not False
        or phase.get("partial") is not False
        or phase.get("formal_analysis_eligible") is not True
        or phase.get("modality_condition") != "text_only"
        or phase.get("task_limit") is not None
    ):
        raise BinderError("Persona-v2 phase provenance is not formal main evidence")
    phase_hash = _verify_self_hash(
        phase,
        field="phase_provenance_sha256",
        label="Persona-v2 phase provenance",
    )
    selected_providers = phase.get("selected_providers")
    frozen_providers = phase.get("frozen_providers")
    if (
        selected_providers != frozen_providers
        or selected_providers != phase_providers["main"]
    ):
        raise BinderError("Persona-v2 main provider roster differs from runtime freeze")
    task_roster = phase.get("task_roster")
    phase_runtime = phase.get("runtime")
    phase_target = phase.get("target")
    if not all(
        isinstance(value, Mapping)
        for value in (task_roster, phase_runtime, phase_target)
    ):
        raise BinderError("Persona-v2 phase provenance bindings are incomplete")
    if (
        task_roster.get("expected_task_ids") != phase_tasks["main"]
        or task_roster.get("expected_task_count") != len(phase_tasks["main"])
        or task_roster.get("frozen_task_count") != len(phase_tasks["main"])
        or task_roster.get("task_set_sha256")
        != phases["main"].get("task_set_sha256")
        or task_roster.get("frozen_task_set_sha256")
        != phases["main"].get("task_set_sha256")
    ):
        raise BinderError("Persona-v2 main task roster differs from runtime freeze")
    if (
        phase_runtime.get("runtime_task_manifest_sha256") != runtime_hash
        or phase_runtime.get("execution_commit") != runtime.get("runtime_commit")
        or phase_runtime.get("runtime_file_set_sha256")
        != runtime.get("runtime_file_set_sha256")
    ):
        raise BinderError("Persona-v2 phase/runtime identity drifted")

    mapping = loaded["mapping_manifest"]
    rows = mapping.get("rows")
    consensus = mapping.get("consensus")
    if (
        mapping.get("schema_version") != "yher.llm_sim_v2.target_option_map.v1"
        or mapping.get("frozen") is not True
        or not isinstance(rows, list)
        or not all(isinstance(row, Mapping) for row in rows)
        or not isinstance(consensus, Mapping)
    ):
        raise BinderError("Persona-v2 mapping manifest envelope is invalid")
    mapping_hash = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    mapped_targets = [
        {
            "item_id": row.get("item_id"),
            "failure_id": row.get("failure_id"),
            "target_option": row.get("target_option"),
        }
        for row in rows
        if row.get("status") == "mapped"
    ]
    target_hash = hashlib.sha256(canonical_json_bytes(mapped_targets)).hexdigest()
    if (
        mapping.get("mapping_sha256") != mapping_hash
        or mapping.get("target_set_hash") != target_hash
        or mapping.get("confirmatory_target_misconception_hit_rate") is not False
        or mapping.get("mapped_fraction") != 0.06
        or consensus.get("mapped_rows") != 6
        or consensus.get("excluded_ambiguous_rows") != 94
        or len(rows) != 100
        or phase_target.get("mapping_sha256") != mapping_hash
        or phase_target.get("target_set_hash") != target_hash
    ):
        raise BinderError("Persona-v2 mapping identity or sparse boundary drifted")

    result = loaded["analysis_results"]
    if (
        result.get("schema_version") != "yher.llm_sim_v2.analysis_results.v1"
        or result.get("simulated") is not True
        or result.get("run_id") != "llm-personas-v2-dual"
        or result.get("analysis_population") != "main"
        or result.get("modality_condition") != "text_only"
        or result.get("independent_cluster_count") != 50
        or result.get("independent_cluster_unit") != "persona_id"
        or result.get("repeated_measure_factors")
        != ["provider", "response_arm", "condition", "item"]
    ):
        raise BinderError("Persona-v2 formal analysis result envelope is invalid")
    claim_boundary = result.get("claim_boundary")
    if not isinstance(claim_boundary, str) or "not human participants" not in claim_boundary:
        raise BinderError("Persona-v2 claim boundary is missing")
    input_proof = result.get("input_proof")
    expected_denominator = result.get("expected_denominator")
    bootstrap = result.get("bootstrap_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (input_proof, expected_denominator, bootstrap)
    ):
        raise BinderError("Persona-v2 formal analysis contracts are incomplete")
    providers = list(selected_providers)
    main_task_count = len(phase_tasks["main"])
    if (
        input_proof.get("schema_version")
        != "yher.llm_sim_v2.analysis_input_proof.v1"
        or input_proof.get("ok") is not True
        or input_proof.get("analysis_population") != "main"
        or input_proof.get("persona_cluster_count") != 50
        or input_proof.get("providers") != providers
        or input_proof.get("expected_task_count") != main_task_count
        or input_proof.get("runtime_task_manifest_sha256") != runtime_hash
        or input_proof.get("phase_provenance_sha256") != phase_hash
    ):
        raise BinderError("Persona-v2 input proof is not bound to runtime and phase")
    if (
        expected_denominator.get("source") != "committed_runtime_task_manifest"
        or expected_denominator.get("filesystem_glob_defines_denominator") is not False
        or expected_denominator.get("provider_count") != len(providers)
        or expected_denominator.get("tasks_per_provider") != main_task_count
        or expected_denominator.get("provider_task_cells")
        != len(providers) * main_task_count
        or bootstrap.get("cluster_unit") != "persona_id"
        or bootstrap.get("provider_equal_weighting") is not True
        or bootstrap.get("resamples") != 10000
        or bootstrap.get("seed") != 2026071503
    ):
        raise BinderError("Persona-v2 denominator or bootstrap contract drifted")

    lifecycle = result.get("provider_lifecycle")
    if not isinstance(lifecycle, list) or len(lifecycle) != len(providers):
        raise BinderError("Persona-v2 provider lifecycle table is incomplete")
    allowed_lifecycles = {
        "complete",
        "complete_with_exclusions",
        "partial_missing",
        "interrupted",
        "unavailable",
        "fuse_open",
        "excluded_repeated_failure",
    }
    lifecycle_index: dict[str, dict[str, Any]] = {}
    for row in lifecycle:
        if not isinstance(row, Mapping):
            raise BinderError("Persona-v2 provider lifecycle row is not an object")
        _assert_exact_keys(
            row,
            {
                "provider",
                "provider_lifecycle",
                "recomputed_provider_lifecycle",
                "expected_count",
                "present_count",
                "missing_count",
                "status_counts",
                "missing_task_ids",
                "known_cost_yuan",
                "unknown_cost_reserve_yuan",
                "accounted_cost_yuan",
                "controlled_complete_cluster_count",
                "controlled_eligible",
                "controlled_exclusion_reasons",
                "blind_complete_cluster_count",
                "blind_eligible",
                "blind_exclusion_reasons",
            },
            "Persona-v2 provider lifecycle row",
        )
        provider = row.get("provider")
        status_counts = row.get("status_counts")
        missing_ids = row.get("missing_task_ids")
        if (
            provider not in providers
            or provider in lifecycle_index
            or row.get("provider_lifecycle") not in allowed_lifecycles
            or row.get("recomputed_provider_lifecycle")
            != row.get("provider_lifecycle")
            or row.get("expected_count") != main_task_count
            or not isinstance(missing_ids, list)
            or not set(missing_ids).issubset(set(phase_tasks["main"]))
            or row.get("missing_count") != len(missing_ids)
            or row.get("present_count") + row.get("missing_count") != main_task_count
            or not isinstance(status_counts, Mapping)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in status_counts.values()
            )
            or sum(status_counts.values()) != row.get("present_count")
            or not isinstance(row.get("controlled_eligible"), bool)
            or not isinstance(row.get("blind_eligible"), bool)
            or not isinstance(row.get("controlled_exclusion_reasons"), list)
            or not isinstance(row.get("blind_exclusion_reasons"), list)
        ):
            raise BinderError("Persona-v2 provider lifecycle does not reconcile")
        known = _persona_number(
            row.get("known_cost_yuan"),
            label=f"{provider} lifecycle known cost",
            minimum=0.0,
        )
        reserve = _persona_number(
            row.get("unknown_cost_reserve_yuan"),
            label=f"{provider} lifecycle reserve cost",
            minimum=0.0,
        )
        accounted = _persona_number(
            row.get("accounted_cost_yuan"),
            label=f"{provider} lifecycle accounted cost",
            minimum=0.0,
        )
        if not math.isclose(accounted, known + reserve, abs_tol=1e-8):
            raise BinderError("Persona-v2 provider lifecycle cost does not reconcile")
        for condition in ("controlled", "blind"):
            _persona_integer(
                row.get(f"{condition}_complete_cluster_count"),
                label=f"{provider} {condition} complete clusters",
                maximum=50,
            )
        lifecycle_index[str(provider)] = dict(row)
    if set(lifecycle_index) != set(providers):
        raise BinderError("Persona-v2 provider lifecycle roster drifted")

    sparse = result.get("sparse_mapping_descriptive")
    if not isinstance(sparse, Mapping) or (
        sparse.get("status") != "sparse_descriptive_only"
        or sparse.get("confirmatory") is not False
        or sparse.get("mapped_mapping_rows") != 6
        or sparse.get("excluded_ambiguous_mapping_rows") != 94
        or sparse.get("total_mapping_rows") != 100
        or sparse.get("mapped_fraction") != 0.06
        or sparse.get("mapping_sha256") != mapping_hash
        or sparse.get("target_set_hash") != target_hash
    ):
        raise BinderError("Persona-v2 analysis mapping surface drifted")
    controlled = _validate_persona_controlled(
        result.get("controlled"), providers=providers, lifecycle=lifecycle_index
    )
    blind = _validate_persona_blind(
        result.get("blind"), providers=providers, lifecycle=lifecycle_index
    )
    judge_adjudication = _validate_persona_judge(
        result.get("judge_adjudication"), allow_fixture=allow_fixture
    )
    available_judges = tuple(judge_adjudication["analysis"]["available_judges"])
    if role_judges != set(available_judges):
        raise BinderError("Persona-v2 judge binding role profile differs from analysis")
    _validate_persona_judge_sources(
        input_manifest=loaded["analysis_input_artifact_manifest"],
        result_input_binding=result.get("input_artifact_binding"),
        judge_results={
            judge: loaded[f"{judge}_judge_result_manifest"]
            for judge in available_judges
        },
        judge_result_paths={
            judge: paths[f"{judge}_judge_result_manifest"]
            for judge in available_judges
        },
        judge_adjudication=judge_adjudication,
    )
    snapshot_artifact_paths, snapshot_bound_files = (
        _validate_persona_judge_run_snapshot(
            root=root,
            snapshot_manifest=loaded["judge_run_execution_snapshot_manifest"],
            snapshot_manifest_path=paths[
                "judge_run_execution_snapshot_manifest"
            ],
            judge_results={
                judge: loaded[f"{judge}_judge_result_manifest"]
                for judge in available_judges
            },
            judge_analysis=judge_adjudication["analysis"],
            allow_fixture=allow_fixture,
        )
    )
    run_evidence_binding = judge_adjudication["run_evidence_binding"]
    snapshot_manifest = loaded["judge_run_execution_snapshot_manifest"]
    _validate_persona_judge_snapshot_input_binding(
        input_manifest=loaded["analysis_input_artifact_manifest"],
        snapshot_manifest=snapshot_manifest,
    )
    if (
        run_evidence_binding["judge_run_evidence_receipt_sha256"]
        != snapshot_manifest.get("source_judge_run_evidence_receipt_sha256")
        or run_evidence_binding["family_slots"]
        != snapshot_manifest.get("family_slots")
    ):
        raise BinderError("Persona-v2 judge run evidence binding drifted from snapshot")

    outputs = result.get("outputs")
    if not isinstance(outputs, Mapping) or (
        outputs.get("machine_json") is not True
        or outputs.get("machine_csv_tables") != 8
        or outputs.get("figure_data_machine_readable") is not True
        or outputs.get("publication_figures") != 3
        or outputs.get("publication_formats") != ["png_300_dpi", "svg"]
        or outputs.get("judge_case_export") is not True
        or not isinstance(outputs.get("judge_shared_input_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(outputs.get("judge_shared_input_sha256")))
        is None
    ):
        raise BinderError("Persona-v2 publication output contract drifted")

    analysis_manifest = loaded["analysis_artifact_manifest"]
    if (
        analysis_manifest.get("schema_version")
        != "yher.llm_sim_v2.analysis_artifact_manifest.v1"
        or analysis_manifest.get("simulated") is not True
        or analysis_manifest.get("run_id") != "llm-personas-v2-dual"
        or analysis_manifest.get("analysis_population") != "main"
        or analysis_manifest.get("target_set_hash") != target_hash
        or analysis_manifest.get("runtime_task_manifest_sha256") != runtime_hash
        or analysis_manifest.get("phase_provenance_sha256") != phase_hash
    ):
        raise BinderError("Persona-v2 analysis artifact manifest envelope drifted")
    artifacts = analysis_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BinderError("Persona-v2 analysis artifact manifest is empty")
    if analysis_manifest.get("artifact_set_sha256") != hashlib.sha256(
        canonical_json_bytes(artifacts)
    ).hexdigest():
        raise BinderError("Persona-v2 analysis artifact set hash drifted")
    analysis_root = paths["analysis_artifact_manifest"].parent
    bound_result = False
    seen_artifacts: set[str] = set()
    artifact_index: dict[str, dict[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise BinderError("Persona-v2 analysis artifact row is not an object")
        _assert_exact_keys(
            row, {"path", "sha256", "size"}, "Persona-v2 analysis artifact"
        )
        relative_value = row.get("path")
        if not isinstance(relative_value, str) or not relative_value:
            raise BinderError("Persona-v2 analysis artifact path is invalid")
        relative = Path(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_value
            or relative_value in seen_artifacts
        ):
            raise BinderError("Persona-v2 analysis artifact path is unsafe or duplicated")
        path, artifact_bytes = _read_relative_regular_file(
            analysis_root,
            relative,
            label=f"Persona-v2 analysis artifact {relative_value}",
        )
        if (
            row.get("sha256") != hashlib.sha256(artifact_bytes).hexdigest()
            or row.get("size") != len(artifact_bytes)
        ):
            raise BinderError(f"Persona-v2 analysis artifact drift: {relative_value}")
        if path.resolve() == paths["analysis_results"].resolve():
            bound_result = True
        seen_artifacts.add(relative_value)
        artifact_index[relative_value] = _persona_asset(
            path=path,
            root=analysis_root,
            digest=str(row["sha256"]),
            size=int(row["size"]),
        )
    if not bound_result:
        raise BinderError("Persona-v2 analysis results are not bound by output manifest")
    required_publication = {
        relative
        for group in PERSONA_PUBLICATION_ARTIFACTS.values()
        for relative in group
    }
    missing_publication = required_publication - set(artifact_index)
    if missing_publication:
        raise BinderError(
            "Persona-v2 publication artifact set is incomplete: "
            + ", ".join(sorted(missing_publication))
        )
    input_artifact_entry = artifact_index.get("input_artifact_manifest.json")
    if (
        input_artifact_entry is None
        or Path(input_artifact_entry["source_path"]).resolve()
        != paths["analysis_input_artifact_manifest"].resolve()
    ):
        raise BinderError(
            "Persona-v2 analysis input artifact manifest is not publication-bound"
        )
    missing_snapshot_artifacts = snapshot_artifact_paths - set(artifact_index)
    if missing_snapshot_artifacts:
        raise BinderError(
            "Persona-v2 judge snapshot tree is not publication-bound: "
            + ", ".join(sorted(missing_snapshot_artifacts))
        )
    for relative, descriptor in snapshot_bound_files.items():
        artifact_descriptor = artifact_index[relative]
        if (
            artifact_descriptor["sha256"] != descriptor["sha256"]
            or artifact_descriptor["bytes"] != descriptor["bytes"]
        ):
            raise BinderError(
                f"Persona-v2 judge snapshot/output artifact drifted: {relative}"
            )
    for relative in PERSONA_PUBLICATION_ARTIFACTS["figures"]:
        payload = Path(artifact_index[relative]["source_path"]).read_bytes()
        if relative.endswith(".png") and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise BinderError(f"Persona-v2 publication artifact is not PNG: {relative}")
        if relative.endswith(".svg") and b"<svg" not in payload[:500]:
            raise BinderError(f"Persona-v2 publication artifact is not SVG: {relative}")
    for relative, expected in (
        ("judge/case_manifest.json", judge_adjudication["case_manifest"]),
        ("judge/judge_analysis.json", judge_adjudication["analysis"]),
    ):
        if relative not in artifact_index:
            raise BinderError(f"Persona-v2 judge publication artifact is missing: {relative}")
        artifact_payload, _ = _load_json_object(
            Path(artifact_index[relative]["source_path"]),
            label=f"Persona-v2 {relative}",
        )
        if artifact_payload != expected:
            raise BinderError(f"Persona-v2 judge publication artifact drifted: {relative}")

    source_hashes = {
        **file_hashes,
        "binding_manifest_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        "runtime_task_manifest_sha256": runtime_hash,
        "phase_provenance_sha256": phase_hash,
        "mapping_sha256": mapping_hash,
        "target_set_hash": target_hash,
        "analysis_artifact_set_sha256": str(
            analysis_manifest["artifact_set_sha256"]
        ),
    }
    publication_assets = {
        "main_persona_composite_sources": {
            "controlled_composition": artifact_index[
                "figures/controlled_composition.png"
            ],
            "blind_terminal_agreement": artifact_index[
                "figures/blind_terminal_agreement.png"
            ],
            "blind_output_stability": artifact_index[
                "figures/blind_output_stability.png"
            ],
        },
        "tables": {
            relative: artifact_index[relative]
            for relative in PERSONA_PUBLICATION_ARTIFACTS["tables"]
        },
        "figure_data": {
            relative: artifact_index[relative]
            for relative in PERSONA_PUBLICATION_ARTIFACTS["figure_data"]
        },
        "supplement_figures": {
            relative: artifact_index[relative]
            for relative in PERSONA_PUBLICATION_ARTIFACTS["figures"]
        },
        "judge_execution_snapshots": snapshot_bound_files,
    }
    source_artifacts = {
        role: _source_artifact(path, relative_path=path.relative_to(root).as_posix())
        for role, path in paths.items()
    }
    source_artifacts["binding_manifest"] = _source_artifact(
        binding_path, relative_path="binding_manifest.json"
    )
    return {
        "schema_version": "yher.journal_binder.persona_v2_bound.v1",
        "status": "bound_formal_w3",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "analysis_population": "main",
        "modality_condition": "text_only",
        "persona_cluster_count": 50,
        "independent_unit": "persona_id",
        "provider_role": "repeated_measurement",
        "provider_count": len(providers),
        "pilot_exclusion": {
            "task_rosters_disjoint": True,
            "pilot_task_count": len(phase_tasks["pilot"]),
            "main_task_count": main_task_count,
            "main_phase_provenance_bound": True,
        },
        "bootstrap_contract": dict(bootstrap),
        "expected_denominator": dict(expected_denominator),
        "provider_lifecycle": [lifecycle_index[value] for value in providers],
        "mapping": {
            "status": sparse["status"],
            "confirmatory": False,
            "mapped_mapping_rows": 6,
            "excluded_ambiguous_mapping_rows": 94,
            "total_mapping_rows": 100,
            "mapped_fraction": 0.06,
            "mapping_sha256": mapping_hash,
            "target_set_hash": target_hash,
        },
        "controlled": controlled,
        "blind": blind,
        "judge_adjudication": judge_adjudication,
        "publication_assets": publication_assets,
        "source_artifacts": source_artifacts,
        "claim_boundary": claim_boundary,
        "source_hashes": source_hashes,
    }


def _assert_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise BinderError(
            f"{label} fields differ from frozen schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _reject_prohibited_claim_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in P2_PROHIBITED_CLAIM_FIELDS:
                raise BinderError(f"P2 prohibited claim field at {path}.{key_text}")
            _reject_prohibited_claim_fields(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_prohibited_claim_fields(child, path=f"{path}[{index}]")


def _p2_finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BinderError(f"P2 numeric field is invalid: {label}") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise BinderError(f"P2 numeric field is invalid: {label}")
    return number


def _p2_source_path(source: Mapping[str, Any], *, role: str) -> Path:
    _assert_exact_keys(source, {"path", "sha256"}, f"P2 source {role}")
    path_value = source.get("path")
    expected_hash = source.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise BinderError(f"P2 source path is invalid: {role}")
    if not isinstance(expected_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ) is None:
        raise BinderError(f"P2 source SHA-256 is invalid: {role}")
    path = Path(path_value)
    resolved, data = _read_absolute_regular_file(path, label=f"P2 source {role}")
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise BinderError(f"P2 source file SHA-256 drift: {role}")
    return resolved


def _read_absolute_regular_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    if path.is_symlink():
        raise BinderError(f"{label} is unsafe or contains a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BinderError(f"{label} is missing, unsafe, or contains a symlink") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise BinderError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return path.resolve(strict=True), b"".join(chunks)


def _p2_read_candidates(path: Path) -> list[Mapping[str, Any]]:
    _, data = _read_absolute_regular_file(path, label="P2 trusted candidate source")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _strict_json_bytes(
                line,
                label=f"P2 trusted candidate source line {line_number}",
            )
        except BinderError as exc:
            raise BinderError(
                f"P2 trusted candidate source has invalid JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, Mapping):
            raise BinderError("P2 trusted candidate row is not an object")
        rows.append(row)
    if not rows:
        raise BinderError("P2 trusted candidate source is empty")
    return rows


def _p2_library_boundary(
    *, candidate_rows: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, set[str]]]:
    segments_by_node = runtime.get("segments_by_node")
    if not isinstance(segments_by_node, Mapping):
        raise BinderError("P2 signed runtime metadata has no segments_by_node map")
    by_part: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    by_chunk_id: dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in by_chunk_id:
            raise BinderError("P2 trusted candidate chunk IDs are invalid or duplicated")
        by_chunk_id[chunk_id] = row
        if row.get("needs_human") is not False:
            continue
        try:
            bv = str(row.get("bv") or "")
            part = int(row.get("p_number") or 1)
            start = float(row["start_sec"])
            end = float(row["end_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BinderError("P2 trusted candidate timing identity is invalid") from exc
        topics = row.get("knowledge_topic")
        if (
            not bv
            or part < 1
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end
            or not isinstance(topics, list)
            or not all(isinstance(topic, str) and topic for topic in topics)
        ):
            raise BinderError("P2 trusted candidate identity is invalid")
        by_part.setdefault((bv, part), []).append(row)

    assignments: dict[str, set[str]] = {}
    for node, legacy_rows in segments_by_node.items():
        node_name = str(node)
        if not isinstance(legacy_rows, list):
            raise BinderError("P2 runtime segment rows are invalid")
        matched: set[str] = set()
        for legacy in legacy_rows:
            if not isinstance(legacy, Mapping):
                raise BinderError("P2 runtime segment row is not an object")
            try:
                part_key = (str(legacy.get("bv") or ""), int(legacy.get("p") or 1))
            except (TypeError, ValueError) as exc:
                raise BinderError("P2 runtime physical source identity is invalid") from exc
            for candidate in by_part.get(part_key, ()):
                if node_name in candidate.get("knowledge_topic", ()):
                    matched.add(str(candidate["chunk_id"]))
        if matched:
            assignments[node_name] = matched
    node_roster = sorted(assignments)
    assignment_count = sum(len(values) for values in assignments.values())
    if len(node_roster) != 13 or assignment_count != 68:
        raise BinderError("P2 wider library boundary must remain 13 nodes and 68 segments")
    unique_chunks = sorted({value for values in assignments.values() for value in values})
    boundary = {
        "node_count": len(node_roster),
        "trusted_exact_segment_assignments": assignment_count,
        "unique_chunk_count": len(unique_chunks),
        "node_roster": node_roster,
        "node_set_hash": target_set_hash(node_roster),
    }
    return boundary, by_chunk_id, assignments


def _bind_p2(p2_dir: Path | None) -> dict[str, Any]:
    if p2_dir is None:
        return {
            "status": "not_requested",
            "illustrative": True,
            "value": None,
        }
    summary_path = p2_dir / "summary.json"
    input_manifest_path = p2_dir / "input_manifest.json"
    output_manifest_path = p2_dir / "output_manifest.json"
    _, summary_bytes = _read_relative_regular_file(
        p2_dir, Path("summary.json"), label="P2 summary"
    )
    _, input_bytes = _read_relative_regular_file(
        p2_dir, Path("input_manifest.json"), label="P2 input manifest"
    )
    _, output_manifest_bytes = _read_relative_regular_file(
        p2_dir, Path("output_manifest.json"), label="P2 output manifest"
    )
    summary = _strict_json_bytes(summary_bytes, label="P2 summary")
    input_manifest = _strict_json_bytes(input_bytes, label="P2 input manifest")
    output_manifest = _strict_json_bytes(
        output_manifest_bytes, label="P2 output manifest"
    )
    if not all(
        isinstance(value, Mapping)
        for value in (summary, input_manifest, output_manifest)
    ):
        raise BinderError("P2 summary or manifest root is not an object")
    _reject_prohibited_claim_fields(summary)

    _assert_exact_keys(
        summary,
        {
            "schema_version",
            "claim_boundary",
            "simulated",
            "illustrative",
            "external_validity",
            "spec_hash",
            "budget_seconds",
            "exact_overlap_targets",
            "candidate_row_count",
            "physical_source_count",
            "reporting_unit",
            "profile_row_count",
            "unique_selector_trace_count",
            "selector_trace_deduplication",
            "scalar_composite_computed",
            "overall",
            "bootstrap_overall",
            "bootstrap_contrasts",
            "truth_cells",
            "unavailable_minute_field_policy",
        },
        "P2 summary",
    )
    _assert_exact_keys(
        input_manifest,
        {
            "schema_version",
            "claim_boundary",
            "simulated",
            "illustrative",
            "external_validity",
            "hash_gate_status",
            "selector",
            "bootstrap",
            "candidate_subset",
            "h1_h4_product_margins",
            "source_files",
            "spec",
        },
        "P2 input manifest",
    )
    _assert_exact_keys(
        output_manifest,
        {
            "schema_version",
            "claim_boundary",
            "simulated",
            "illustrative",
            "external_validity",
            "manifest_self_hash_policy",
            "profile_row_count",
            "unique_selector_trace_count",
            "artifacts",
        },
        "P2 output manifest",
    )
    if summary.get("schema_version") != "yher.p2.summary.v1":
        raise BinderError("P2 summary schema drift")
    if input_manifest.get("schema_version") != "yher.p2.input_manifest.v1":
        raise BinderError("P2 input manifest schema drift")
    if output_manifest.get("schema_version") != "yher.p2.output_manifest.v1":
        raise BinderError("P2 output manifest schema drift")
    for source in (summary, input_manifest, output_manifest):
        if source.get("claim_boundary") != P2_CLAIM_BOUNDARY:
            raise BinderError("P2 claim boundary drift")
        if source.get("simulated") is not True or source.get("illustrative") is not True:
            raise BinderError("P2 must remain simulated and illustrative")
        if source.get("external_validity") is not False:
            raise BinderError("P2 cannot claim external validity")

    artifacts = output_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BinderError("P2 output manifest has no artifact list")
    seen_artifacts: set[str] = set()
    artifact_index: dict[str, dict[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise BinderError("P2 output artifact row is not an object")
        _assert_exact_keys(row, {"filename", "bytes", "sha256"}, "P2 output artifact")
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename or filename in seen_artifacts:
            raise BinderError("P2 output artifact filename is invalid or duplicated")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise BinderError("P2 output artifact path escapes its root")
        artifact_path, artifact_bytes = _read_relative_regular_file(
            p2_dir, relative, label=f"P2 output artifact {filename}"
        )
        if (
            row.get("sha256") != hashlib.sha256(artifact_bytes).hexdigest()
            or row.get("bytes") != len(artifact_bytes)
        ):
            raise BinderError(f"P2 output artifact hash or size drift: {filename}")
        seen_artifacts.add(filename)
        artifact_index[filename] = {
            "relative_path": filename,
            "source_path": str(artifact_path.resolve()),
            "sha256": str(row["sha256"]),
            "bytes": int(row["bytes"]),
        }
    if not {"summary.json", "input_manifest.json"}.issubset(seen_artifacts):
        raise BinderError("P2 output manifest does not bind summary and input manifest")
    missing_publication = set(P2_PUBLICATION_ARTIFACTS) - seen_artifacts
    if missing_publication:
        raise BinderError(
            "P2 publication artifact set is incomplete: "
            + ", ".join(sorted(missing_publication))
        )
    for relative in (
        "p2_supply_bound_illustration.png",
        "p2_supply_bound_illustration.svg",
    ):
        payload = Path(artifact_index[relative]["source_path"]).read_bytes()
        if relative.endswith(".png") and not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise BinderError("P2 publication PNG artifact is invalid")
        if relative.endswith(".svg") and b"<svg" not in payload[:500]:
            raise BinderError("P2 publication SVG artifact is invalid")

    selector = input_manifest.get("selector")
    bootstrap_contract = input_manifest.get("bootstrap")
    subset = input_manifest.get("candidate_subset")
    margins = input_manifest.get("h1_h4_product_margins")
    source_files = input_manifest.get("source_files")
    spec = input_manifest.get("spec")
    if not all(
        isinstance(value, Mapping)
        for value in (selector, bootstrap_contract, subset, margins, source_files, spec)
    ):
        raise BinderError("P2 input manifest contracts are incomplete")
    _assert_exact_keys(
        selector,
        {
            "budget_seconds",
            "binary_role_slot_saturation",
            "physical_source_no_repeat",
            "decimal_precision_digits",
        },
        "P2 selector",
    )
    if (
        selector.get("budget_seconds") != 600
        or selector.get("binary_role_slot_saturation") is not True
        or selector.get("physical_source_no_repeat") is not True
        or int(selector.get("decimal_precision_digits", 0)) < 50
        or summary.get("budget_seconds") != 600
    ):
        raise BinderError("P2 selector budget or frozen constraints drifted")
    _assert_exact_keys(
        bootstrap_contract,
        {
            "resamples",
            "seed",
            "rng",
            "cluster_count_per_stratum",
            "fixed_target_strata",
            "arms_and_truths_paired_within_target_replicate",
        },
        "P2 bootstrap contract",
    )
    if (
        bootstrap_contract.get("resamples") != 10000
        or bootstrap_contract.get("seed") != 2026071505
        or bootstrap_contract.get("cluster_count_per_stratum") != 50
        or bootstrap_contract.get("arms_and_truths_paired_within_target_replicate")
        is not True
    ):
        raise BinderError("P2 bootstrap contract drifted")

    _assert_exact_keys(
        source_files,
        {
            "trusted_candidate_jsonl",
            "signed_runtime_metadata",
            "h1_h4_raw_manifest",
        },
        "P2 source file map",
    )
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "input_manifest_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "output_manifest_sha256": hashlib.sha256(output_manifest_bytes).hexdigest(),
    }
    for role in (
        "trusted_candidate_jsonl",
        "signed_runtime_metadata",
        "h1_h4_raw_manifest",
    ):
        source = source_files.get(role)
        if not isinstance(source, Mapping):
            raise BinderError(f"P2 source entry is invalid: {role}")
        source_paths[role] = _p2_source_path(source, role=role)
        source_hashes[f"{role}_sha256"] = str(source["sha256"])
    if input_manifest.get("hash_gate_status") != "pass":
        raise BinderError("P2 input hash gate did not pass")

    _assert_exact_keys(
        spec,
        {"sha256", "committed_bytes_sha256", "commit", "precedes_outcome_computation"},
        "P2 spec binding",
    )
    spec_sha = sha256_file(P2_SPEC_PATH)
    if (
        spec.get("sha256") != spec_sha
        or spec.get("committed_bytes_sha256") != spec_sha
        or spec.get("precedes_outcome_computation") is not True
        or summary.get("spec_hash") != spec_sha
    ):
        raise BinderError("P2 frozen spec bytes are not cross-bound")
    spec_commit = spec.get("commit")
    if not isinstance(spec_commit, str) or re.fullmatch(r"[0-9a-f]{40}", spec_commit) is None:
        raise BinderError("P2 spec commit is invalid")
    repository = Path(__file__).resolve().parents[1]
    try:
        committed_spec = subprocess.check_output(
            [
                "git",
                "show",
                f"{spec_commit}:experiments/p2_illustrative_analysis_plan.md",
            ],
            cwd=repository,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", spec_commit, "HEAD"],
            cwd=repository,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        commit_epoch = int(
            subprocess.check_output(
                ["git", "show", "-s", "--format=%ct", spec_commit],
                cwd=repository,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise BinderError("P2 spec commit cannot be verified") from exc
    if hashlib.sha256(committed_spec).hexdigest() != spec_sha:
        raise BinderError("P2 committed spec bytes differ from the bound spec")
    if commit_epoch >= min(
        summary_path.stat().st_mtime,
        input_manifest_path.stat().st_mtime,
        output_manifest_path.stat().st_mtime,
    ):
        raise BinderError("P2 spec commit does not precede outcome artifacts")
    source_hashes["p2_spec_sha256"] = spec_sha

    candidate_rows = _p2_read_candidates(source_paths["trusted_candidate_jsonl"])
    runtime, _ = _load_json_object(
        source_paths["signed_runtime_metadata"], label="P2 signed runtime metadata"
    )
    library_boundary, by_chunk_id, assignments = _p2_library_boundary(
        candidate_rows=candidate_rows, runtime=runtime
    )
    raw_manifest, _ = _load_json_object(
        source_paths["h1_h4_raw_manifest"], label="P2 H1-H4 raw manifest"
    )
    raw_shards = raw_manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise BinderError("P2 H1-H4 raw manifest has no shards")
    h1_h4_targets: set[str] = set()
    for row in raw_shards:
        if not isinstance(row, Mapping) or not isinstance(row.get("shard_id"), str):
            raise BinderError("P2 H1-H4 raw shard identity is invalid")
        match = re.fullmatch(
            r"target=(.+?)\|truth=[^|]+\|condition=[^|]+", str(row["shard_id"])
        )
        if match is None:
            raise BinderError("P2 H1-H4 raw shard identity is invalid")
        h1_h4_targets.add(match.group(1))
    overlap = sorted(set(library_boundary["node_roster"]) & h1_h4_targets)
    exact_overlap_targets = summary.get("exact_overlap_targets")
    if (
        not isinstance(exact_overlap_targets, list)
        or exact_overlap_targets != overlap
        or len(overlap) != 2
        or bootstrap_contract.get("fixed_target_strata") != overlap
    ):
        raise BinderError("P2 exact overlap must remain the frozen two targets")

    _assert_exact_keys(
        subset,
        {
            "exact_chunk_ids",
            "row_count",
            "physical_source_count",
            "canonical_sha256",
            "canonical_serialization",
            "audit_declared_digest_is_gate",
            "audit_declared_unreproduced",
        },
        "P2 candidate subset",
    )
    exact_ids = subset.get("exact_chunk_ids")
    if (
        not isinstance(exact_ids, list)
        or len(exact_ids) != 8
        or len(set(exact_ids)) != 8
        or not all(isinstance(value, str) and value in by_chunk_id for value in exact_ids)
    ):
        raise BinderError("P2 candidate subset must contain eight exact source rows")
    exact_rows = [by_chunk_id[str(value)] for value in exact_ids]
    canonical_subset_sha = hashlib.sha256(
        canonical_json_bytes(sorted(exact_rows, key=lambda row: str(row["chunk_id"])))
    ).hexdigest()
    exact_physical_sources: set[tuple[str, int]] = set()
    for row in exact_rows:
        topics = [str(value) for value in row.get("knowledge_topic", ())]
        if len(set(topics) & set(overlap)) != 1:
            raise BinderError("P2 candidate row is not bound to exactly one overlap target")
        exact_physical_sources.add((str(row.get("bv") or ""), int(row.get("p_number") or 1)))
    overlap_assignment_ids = {
        chunk_id for target in overlap for chunk_id in assignments[target]
    }
    if (
        subset.get("row_count") != 8
        or subset.get("physical_source_count") != 3
        or subset.get("canonical_sha256") != canonical_subset_sha
        or subset.get("audit_declared_digest_is_gate") is not False
        or set(exact_ids) != overlap_assignment_ids
        or len(exact_physical_sources) != 3
        or summary.get("candidate_row_count") != 8
        or summary.get("physical_source_count") != 3
    ):
        raise BinderError("P2 exact-overlap 2/8/3 boundary drifted")

    overall = summary.get("overall")
    if not isinstance(overall, list) or [row.get("arm") for row in overall if isinstance(row, Mapping)] != list(P2_ARMS):
        raise BinderError("P2 summary must contain exactly the four frozen arms")
    overall_keys = {
        "arm",
        "selected_seconds",
        "selected_minutes",
        "selected_segment_count",
        "mismatched_selected_seconds",
        "mismatched_selected_minutes",
        "missed_available_seconds",
        "missed_available_supply_minutes",
        "unused_budget_seconds",
        "unused_budget_minutes",
        "unobtainable_truth_slots",
        "unsupported_posterior_mass",
        "structural_failure_node_fraction",
        "missed_diagnostic_structural_failure_seconds",
        "missed_posterior_selection_seconds",
        "missed_budget_constraint_seconds",
        "unobtainable_supply_minutes",
        "unobtainable_reason",
        "analytic_integration_terms",
        "analytic_terms_are_not_sample_size",
    }
    sanitized_overall: list[dict[str, Any]] = []
    overall_index: dict[str, Mapping[str, Any]] = {}
    for row in overall:
        if not isinstance(row, Mapping):
            raise BinderError("P2 overall row is not an object")
        _assert_exact_keys(row, overall_keys, "P2 overall row")
        arm = str(row["arm"])
        if row.get("unobtainable_supply_minutes") is not None or row.get(
            "unobtainable_reason"
        ) != "no_frozen_role_compatible_dose":
            raise BinderError("P2 unobtainable supply field must remain null")
        if row.get("analytic_terms_are_not_sample_size") is not True:
            raise BinderError("P2 analytic terms cannot be presented as sample size")
        for field in overall_keys - {
            "arm",
            "unobtainable_supply_minutes",
            "unobtainable_reason",
            "analytic_terms_are_not_sample_size",
        }:
            _p2_finite(row[field], label=f"overall.{arm}.{field}", nonnegative=True)
        selected = _p2_finite(row["selected_seconds"], label=f"overall.{arm}.selected")
        unused = _p2_finite(row["unused_budget_seconds"], label=f"overall.{arm}.unused")
        if (
            not math.isclose(selected + unused, 600.0, abs_tol=1e-8)
            or not math.isclose(float(row["selected_minutes"]), selected / 60, abs_tol=1e-8)
            or not math.isclose(
                float(row["mismatched_selected_minutes"]),
                float(row["mismatched_selected_seconds"]) / 60,
                abs_tol=1e-8,
            )
            or not math.isclose(
                float(row["missed_available_supply_minutes"]),
                float(row["missed_available_seconds"]) / 60,
                abs_tol=1e-8,
            )
            or not math.isclose(float(row["unused_budget_minutes"]), unused / 60, abs_tol=1e-8)
        ):
            raise BinderError("P2 overall seconds/minutes or budget identity drifted")
        structural = float(row["structural_failure_node_fraction"])
        if (arm == "C" and structural != 0.5) or (arm != "C" and structural != 0.0):
            raise BinderError("P2 Arm C structural-failure annotation drifted")
        overall_index[arm] = row
        sanitized_overall.append({key: row[key] for key in sorted(overall_keys)})

    bootstrap_overall = summary.get("bootstrap_overall")
    bootstrap_contrasts = summary.get("bootstrap_contrasts")
    if not isinstance(bootstrap_overall, list) or not isinstance(
        bootstrap_contrasts, list
    ):
        raise BinderError("P2 bootstrap outputs are missing")
    expected_overall_keys = {
        (arm, metric) for arm in P2_ARMS for metric in P2_BOOTSTRAP_METRICS
    }
    seen_overall_keys: set[tuple[str, str]] = set()
    sanitized_bootstrap_overall: list[dict[str, Any]] = []
    for row in bootstrap_overall:
        if not isinstance(row, Mapping):
            raise BinderError("P2 bootstrap overall row is not an object")
        _assert_exact_keys(
            row,
            {"arm", "metric", "point", "ci95_low", "ci95_high", "defined_resamples"},
            "P2 bootstrap overall row",
        )
        key = (str(row["arm"]), str(row["metric"]))
        if key in seen_overall_keys or key not in expected_overall_keys:
            raise BinderError("P2 bootstrap overall key set drifted")
        point = _p2_finite(row["point"], label=f"bootstrap_overall.{key}.point")
        low = _p2_finite(row["ci95_low"], label=f"bootstrap_overall.{key}.low")
        high = _p2_finite(row["ci95_high"], label=f"bootstrap_overall.{key}.high")
        if (
            row.get("defined_resamples") != 10000
            or low > high
            or not math.isclose(point, float(overall_index[key[0]][key[1]]), abs_tol=1e-8)
        ):
            raise BinderError("P2 bootstrap overall estimate drifted")
        seen_overall_keys.add(key)
        sanitized_bootstrap_overall.append(dict(row))
    if seen_overall_keys != expected_overall_keys:
        raise BinderError("P2 bootstrap overall table is incomplete")

    expected_contrast_keys = {
        (contrast, metric)
        for contrast in P2_CONTRASTS
        for metric in P2_BOOTSTRAP_METRICS
    }
    seen_contrast_keys: set[tuple[str, str]] = set()
    sanitized_bootstrap_contrasts: list[dict[str, Any]] = []
    for row in bootstrap_contrasts:
        if not isinstance(row, Mapping):
            raise BinderError("P2 bootstrap contrast row is not an object")
        _assert_exact_keys(
            row,
            {
                "contrast",
                "metric",
                "point",
                "ci95_low",
                "ci95_high",
                "defined_resamples",
                "arm_c_structural_failure_annotation",
            },
            "P2 bootstrap contrast row",
        )
        key = (str(row["contrast"]), str(row["metric"]))
        if key in seen_contrast_keys or key not in expected_contrast_keys:
            raise BinderError("P2 bootstrap contrast key set drifted")
        point = _p2_finite(row["point"], label=f"bootstrap_contrast.{key}.point")
        low = _p2_finite(row["ci95_low"], label=f"bootstrap_contrast.{key}.low")
        high = _p2_finite(row["ci95_high"], label=f"bootstrap_contrast.{key}.high")
        expected_annotation = (
            "failed_node_fraction=0.5" if key[0] == "C_minus_oracle_ITT" else None
        )
        if (
            row.get("defined_resamples") != 10000
            or low > high
            or row.get("arm_c_structural_failure_annotation") != expected_annotation
            or not low <= point <= high
        ):
            raise BinderError("P2 bootstrap contrast estimate or annotation drifted")
        seen_contrast_keys.add(key)
        sanitized_bootstrap_contrasts.append(dict(row))
    if seen_contrast_keys != expected_contrast_keys:
        raise BinderError("P2 bootstrap contrast table is incomplete")

    truth_cells = summary.get("truth_cells")
    truth_cell_keys = {
        "arm",
        "truth_basic",
        "truth_alkane",
        *P2_BOOTSTRAP_METRICS,
    }
    expected_truth_keys = {
        (arm, left, right)
        for arm in P2_ARMS
        for left in _ALL_TRUTH
        for right in _ALL_TRUTH
    }
    seen_truth_keys: set[tuple[str, str, str]] = set()
    if not isinstance(truth_cells, list):
        raise BinderError("P2 truth-cell table is missing")
    for row in truth_cells:
        if not isinstance(row, Mapping):
            raise BinderError("P2 truth-cell row is not an object")
        _assert_exact_keys(row, truth_cell_keys, "P2 truth-cell row")
        key = (str(row["arm"]), str(row["truth_basic"]), str(row["truth_alkane"]))
        if key in seen_truth_keys or key not in expected_truth_keys:
            raise BinderError("P2 truth-cell key set drifted")
        for metric in P2_BOOTSTRAP_METRICS:
            _p2_finite(row[metric], label=f"truth_cell.{key}.{metric}", nonnegative=True)
        seen_truth_keys.add(key)
    if seen_truth_keys != expected_truth_keys:
        raise BinderError("P2 truth-cell table is incomplete")

    unavailable_policy = summary.get("unavailable_minute_field_policy")
    if not isinstance(unavailable_policy, Mapping):
        raise BinderError("P2 unavailable-minute policy is missing")
    _assert_exact_keys(unavailable_policy, {"value", "reason"}, "P2 unavailable policy")
    if unavailable_policy != {
        "value": None,
        "reason": "no_frozen_role_compatible_dose",
    }:
        raise BinderError("P2 unavailable-minute policy drifted")
    if (
        summary.get("scalar_composite_computed") is not False
        or summary.get("reporting_unit")
        != "two_fixed_target_strata_each_with_50_programmatic_replicate_clusters"
        or output_manifest.get("profile_row_count") != summary.get("profile_row_count")
        or output_manifest.get("unique_selector_trace_count")
        != summary.get("unique_selector_trace_count")
    ):
        raise BinderError("P2 reporting-unit or output count contract drifted")

    return {
        "status": "bound",
        "schema_version": "yher.p2.summary.v1",
        "claim_boundary": P2_CLAIM_BOUNDARY,
        "simulated": True,
        "illustrative": True,
        "external_validity": False,
        "budget_seconds": 600,
        "exact_overlap_targets": list(exact_overlap_targets),
        "candidate_row_count": 8,
        "physical_source_count": 3,
        "library_boundary": library_boundary,
        "exact_overlap_boundary": {
            "target_count": 2,
            "candidate_row_count": 8,
            "physical_source_count": 3,
        },
        "overall": sanitized_overall,
        "bootstrap_overall": sanitized_bootstrap_overall,
        "bootstrap_contrasts": sanitized_bootstrap_contrasts,
        "bootstrap_contract": {
            "resamples": 10000,
            "seed": 2026071505,
            "cluster_count_per_stratum": 50,
            "arms_and_truths_paired_within_target_replicate": True,
        },
        "unobtainable_supply_minutes": None,
        "unobtainable_reason": "no_frozen_role_compatible_dose",
        "claim_exclusions": sorted(P2_PROHIBITED_CLAIM_FIELDS),
        "publication_assets": {
            "main_table_source": artifact_index["summary.json"],
            "figure_data": artifact_index["figure_data.json"],
            "supplement_figure_png": artifact_index[
                "p2_supply_bound_illustration.png"
            ],
            "supplement_figure_svg": artifact_index[
                "p2_supply_bound_illustration.svg"
            ],
        },
        "source_artifacts": {
            "summary": _source_artifact(summary_path),
            "input_manifest": _source_artifact(input_manifest_path),
            "output_manifest": _source_artifact(output_manifest_path),
            **{
                role: _source_artifact(path)
                for role, path in source_paths.items()
            },
        },
        "source_hashes": source_hashes,
    }


def build_binder(
    *,
    raw_manifest_path: Path | str | None,
    registry_path: Path | str = DEFAULT_REGISTRY,
    results_path: Path | str | None = None,
    paper_artifact_manifest_path: Path | str | None = None,
    persona_v2_dir: Path | str | None = None,
    p2_dir: Path | str | None = DEFAULT_P2_DIR,
    require_complete: bool = True,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    if raw_manifest_path is None:
        raise BinderError(
            "raw_manifest_path is required; no primary-manifest default is permitted"
        )
    raw_path = Path(raw_manifest_path)
    if not raw_path.is_file():
        raise BinderError(f"raw_manifest_path does not exist: {raw_path}")
    registry = Path(registry_path)
    if not registry.is_file():
        raise BinderError(f"metric registry does not exist: {registry}")

    supports, raw_source_hashes = _extract_supports(
        raw_path, require_frozen=require_complete
    )
    registry_rows, registry_source_hashes = _load_registry(registry)
    source_hashes = {**raw_source_hashes, **registry_source_hashes}
    results: Mapping[str, Any] | None = None
    results_file: Path | None = None
    if require_complete and results_path is None:
        raise BinderError("results artifact is required for a complete binding")
    if results_path is not None:
        results_file = Path(results_path)
        if not results_file.is_file():
            raise BinderError(f"results artifact does not exist: {results_file}")
        try:
            loaded_results = json.loads(results_file.read_bytes())
        except json.JSONDecodeError as exc:
            raise BinderError("results artifact is not valid JSON") from exc
        if not isinstance(loaded_results, Mapping):
            raise BinderError("results artifact root must be an object")
        results = loaded_results
        source_hashes["results_sha256"] = sha256_file(results_file)
        for key in ("analysis_code_sha256", "analysis_commit"):
            value = loaded_results.get(key)
            if isinstance(value, str) and value:
                source_hashes[key] = value

    available_ids = set(registry_rows)
    missing_reportable = set(REPORTABLE_METRIC_IDS) - available_ids
    if require_complete and missing_reportable:
        raise BinderError(
            "reportable metric registry is incomplete: "
            + ", ".join(sorted(missing_reportable))
        )
    selected_ids = [
        metric_id for metric_id in REPORTABLE_METRIC_IDS if metric_id in available_ids
    ]
    required_pair_ids = {
        str(spec[key])
        for spec in PAIR_SPECS.values()
        for key in ("left", "right")
    }
    missing_pair_ids = required_pair_ids - available_ids
    if missing_pair_ids:
        raise BinderError(f"same-support pair metrics are missing: {sorted(missing_pair_ids)}")
    raw_hashes = {str(registry_rows[metric_id]["raw_hash"]) for metric_id in selected_ids}
    if len(raw_hashes) != 1:
        raise BinderError("reportable metrics do not share one raw aggregate hash")
    raw_aggregate_hash = next(iter(raw_hashes))
    if results is not None and results.get("raw_hash") != raw_aggregate_hash:
        raise BinderError("results raw hash differs from reportable metric registry")
    if require_complete and raw_source_hashes["raw_aggregate_hash"] != raw_aggregate_hash:
        raise BinderError("registry raw hash differs from frozen raw manifest and shards")
    source_hashes["raw_aggregate_hash"] = raw_aggregate_hash

    execution_integrity: dict[str, Any] | None = None
    programmatic_publication_assets: dict[str, dict[str, Any]] = {}
    paper_artifact_file: Path | None = None
    if require_complete:
        if results is None or results_file is None:
            raise BinderError("results artifact is required for a complete binding")
        artifact_path = (
            Path(paper_artifact_manifest_path)
            if paper_artifact_manifest_path is not None
            else results_file.parent / "artifact_manifest.json"
        )
        paper_artifact_file = artifact_path
        (
            source_updates,
            execution_integrity,
            programmatic_publication_assets,
        ) = _validate_complete_h1_h4_evidence(
            raw_manifest_path=raw_path,
            registry_path=registry,
            results_path=results_file,
            paper_artifact_manifest_path=artifact_path,
            supports=supports,
            source_hashes=source_hashes,
            registry_rows=registry_rows,
            results=results,
        )
        source_hashes.update(source_updates)

    metrics: dict[str, dict[str, Any]] = {}
    unbound_metrics: dict[str, dict[str, Any]] = {}
    for metric_id in selected_ids:
        if metric_id in UNBOUND_DIAGNOSTIC_IDS:
            unbound_metrics[metric_id] = _unbound_metric(
                registry_rows[metric_id], source_hashes=source_hashes
            )
            continue
        support_id = _metric_support_id(metric_id)
        metrics[metric_id] = _validate_metric(
            registry_rows[metric_id],
            support=supports[support_id],
            source_hashes=source_hashes,
        )
    pairs = _same_support_pairs(metrics)
    persona_v2 = (
        persona_v2_slot()
        if persona_v2_dir is None
        else bind_persona_v2_artifacts(
            Path(persona_v2_dir), allow_fixture=allow_fixture
        )
    )
    p2_bound = _bind_p2(None if p2_dir is None else Path(p2_dir))
    if (
        p2_bound.get("status") == "bound"
        and p2_bound.get("source_hashes", {}).get("h1_h4_raw_manifest_sha256")
        != raw_source_hashes["raw_manifest_sha256"]
    ):
        raise BinderError("P2 and H1-H4 binder use different raw manifests")
    source_artifacts = {
        "raw_manifest": _source_artifact(raw_path),
        "metric_registry": _source_artifact(registry),
    }
    if results_file is not None:
        source_artifacts["results"] = _source_artifact(results_file)
    if paper_artifact_file is not None:
        source_artifacts["paper_artifact_manifest"] = _source_artifact(
            paper_artifact_file
        )
    return {
        "schema_version": "yher.journal_support_binder.v1",
        "status": (
            "bound_with_persona_v2_pending"
            if persona_v2["status"] == "pending_formal_w3_artifacts"
            else "bound"
        ),
        "simulated": True,
        "support_comparison_policy": "reject_unless_support_filter_weighting_denominator_match",
        "raw_manifest_input_policy": "explicit_path_override_required",
        "supports": supports,
        "metrics": metrics,
        "unbound_metrics": unbound_metrics,
        "hypotheses": {
            "H1": {
                "decision": (
                    results["decisions"]["H1"]
                    if results is not None
                    else EXPECTED_HYPOTHESIS_DECISIONS["H1"]
                ),
                "metric_ids": list(H1_REPORTABLE),
            },
            "H2": {
                "decision": (
                    results["decisions"]["H2"]
                    if results is not None
                    else EXPECTED_HYPOTHESIS_DECISIONS["H2"]
                ),
                "metric_ids": list(H2_REPORTABLE),
            },
            "H3": {
                "decision": (
                    results["decisions"]["H3"]
                    if results is not None
                    else EXPECTED_HYPOTHESIS_DECISIONS["H3"]
                ),
                "metric_ids": list(H3_REPORTABLE),
            },
            "H4": {
                "decision": (
                    results["decisions"]["H4"]
                    if results is not None
                    else EXPECTED_HYPOTHESIS_DECISIONS["H4"]
                ),
                "metric_ids": list(H4_REPORTABLE),
            },
        },
        "execution_integrity": execution_integrity,
        "programmatic_publication_assets": programmatic_publication_assets,
        "same_support_pairs": pairs,
        "persona_v2": persona_v2,
        "p2": p2_bound,
        "source_artifacts": source_artifacts,
        "source_hashes": source_hashes,
        "evidence_gaps": [
            "full_27_gap_has_no_machine_bound_bootstrap_interval",
            *(
                ["persona_v2_pending_formal_w3_artifacts"]
                if persona_v2["status"] == "pending_formal_w3_artifacts"
                else []
            ),
        ],
    }


def _display_number(value: Any) -> str:
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:.6g}"


def _metric_display(metric: Mapping[str, Any]) -> str:
    return (
        f"{_display_number(metric['numerator'])}/"
        f"{_display_number(metric['denominator'])} "
        f"({100.0 * float(metric['value']):.1f}%)"
    )


def _percent_display(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.1f}%"


def _ci_display(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "NA"
    return f"{_percent_display(value[0])} to {_percent_display(value[1])}"


def _bound_metric(binder: Mapping[str, Any], metric_id: str) -> Mapping[str, Any]:
    metrics = binder.get("metrics")
    metric = metrics.get(metric_id) if isinstance(metrics, Mapping) else None
    if not isinstance(metric, Mapping):
        raise BinderError(f"bound manuscript metric is missing: {metric_id}")
    return metric


def _metric_ci_text(metric: Mapping[str, Any], *, percent: bool = True) -> str:
    low = metric.get("ci_low")
    high = metric.get("ci_high")
    if low is None or high is None:
        return "95% CI unavailable"
    if percent:
        return f"95% CI {_percent_display(low)} to {_percent_display(high)}"
    return f"95% CI {_display_number(low)} to {_display_number(high)}"


def _contrast_text(metric: Mapping[str, Any]) -> str:
    return (
        f"{_display_number(metric['numerator'])}/"
        f"{_display_number(metric['denominator'])} paired contrast; "
        f"{100.0 * float(metric['value']):.1f} percentage points; "
        f"{_metric_ci_text(metric)}"
    )


def _decision_text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise BinderError("bound hypothesis decision is invalid")
    return value.replace("_", " ")


def render_manuscript_slots(binder: Mapping[str, Any]) -> dict[str, Any]:
    if binder.get("schema_version") != "yher.journal_support_binder.v1":
        raise BinderError("manuscript slots require a journal support binder")
    hypotheses = binder.get("hypotheses")
    pairs = binder.get("same_support_pairs")
    persona = binder.get("persona_v2")
    p2 = binder.get("p2")
    if not all(isinstance(value, Mapping) for value in (hypotheses, pairs, persona, p2)):
        raise BinderError("binder manuscript surfaces are incomplete")

    decision_lines = ["| Hypothesis | Frozen decision |", "|---|---|"]
    for hypothesis in ("H1", "H2", "H3", "H4"):
        row = hypotheses.get(hypothesis)
        if not isinstance(row, Mapping) or not isinstance(row.get("decision"), str):
            raise BinderError("binder hypothesis decision surface is incomplete")
        decision_lines.append(
            f"| {hypothesis} | {_decision_text(row['decision'])} |"
        )

    convergence_lines = [
        "| Pair ID | Support | Terminal accuracy | Correct convergence | Difference |",
        "|---|---|---:|---:|---:|",
    ]
    for pair_id in PAIR_SPECS:
        pair = pairs.get(pair_id)
        if not isinstance(pair, Mapping):
            raise BinderError(f"binder same-support pair is missing: {pair_id}")
        left = binder["metrics"].get(pair["left_metric_id"])
        right = binder["metrics"].get(pair["right_metric_id"])
        difference = pair.get("difference")
        if not all(isinstance(value, Mapping) for value in (left, right, difference)):
            raise BinderError(f"binder same-support pair is incomplete: {pair_id}")
        convergence_lines.append(
            "| "
            + pair_id
            + " | "
            + str(pair["support_id"])
            + " | "
            + _metric_display(left)
            + " | "
            + _metric_display(right)
            + " | "
            + f"{100.0 * float(difference['value']):.1f} pp |"
        )

    execution = binder.get("execution_integrity")
    if isinstance(execution, Mapping):
        full_support = binder["supports"]["full"]
        eligible_support = binder["supports"]["eligible"]
        execution_markdown = (
            f"{execution['intended_journey_count']} intended journeys; "
            f"{execution['valid_journey_count']} valid; "
            f"{execution['structural_failure_count']} structural failures; "
            f"{execution['schema_invalid_count']} schema-invalid. The full analysis "
            f"support contains {full_support['n_target']} targets; the mechanically "
            f"eligible H1/H2 support contains {eligible_support['n_target']} targets."
        )
    else:
        execution_markdown = "Execution-integrity counts are not bound in this draft."

    if persona.get("status") == "bound_formal_w3":
        pilot = persona.get("pilot_exclusion")
        if not isinstance(pilot, Mapping) or pilot.get("task_rosters_disjoint") is not True:
            raise BinderError("bound Persona-v2 slot lacks pilot exclusion proof")
        controlled = persona.get("controlled")
        blind = persona.get("blind")
        judge = persona.get("judge_adjudication")
        if not all(isinstance(value, Mapping) for value in (controlled, blind, judge)):
            raise BinderError("bound Persona-v2 quantitative surfaces are incomplete")
        effects = controlled.get("paired_effects")
        if not isinstance(effects, list):
            raise BinderError("bound Persona-v2 paired effects are missing")
        effect_index = {
            str(row.get("metric_id")): row
            for row in effects
            if isinstance(row, Mapping)
        }
        effect_labels = {
            "conditional_answer_accuracy": "Conditional answer accuracy",
            "correct_response_yield": "Correct-response yield",
            "incorrect_response_yield": "Incorrect-response yield",
            "abstention_yield": "Abstention yield",
            "technical_or_schema_failure_yield": "Technical/schema-failure yield",
        }
        conditional_effect = effect_index.get("conditional_answer_accuracy")
        if not isinstance(conditional_effect, Mapping):
            raise BinderError("bound Persona-v2 conditional-accuracy effect is missing")
        lifecycle = persona.get("provider_lifecycle")
        if not isinstance(lifecycle, list) or len(lifecycle) != len(PERSONA_PROVIDERS):
            raise BinderError("bound Persona-v2 provider lifecycle is incomplete")
        lifecycle_lines = [
            "| Provider lifecycle | Records | Controlled status | Blind status | Exclusion reasons |",
            "|---|---:|---|---|---|",
        ]
        for lifecycle_row in lifecycle:
            if not isinstance(lifecycle_row, Mapping):
                raise BinderError("bound Persona-v2 provider lifecycle row is invalid")
            provider_name = str(lifecycle_row.get("provider") or "")
            expected = _persona_integer(
                lifecycle_row.get("expected_count"),
                label=f"{provider_name} expected lifecycle records",
            )
            missing_count = _persona_integer(
                lifecycle_row.get("missing_count"),
                label=f"{provider_name} missing lifecycle records",
                maximum=expected,
            )
            controlled_eligible = lifecycle_row.get("controlled_eligible") is True
            blind_eligible = lifecycle_row.get("blind_eligible") is True
            reasons = sorted(
                {
                    str(reason)
                    for field in (
                        "controlled_exclusion_reasons",
                        "blind_exclusion_reasons",
                    )
                    for reason in (lifecycle_row.get(field) or [])
                }
            )
            lifecycle_lines.append(
                f"| {provider_name} | {expected - missing_count:,}/{expected:,} | "
                f"{'controlled eligible' if controlled_eligible else 'controlled excluded'} | "
                f"{'blind eligible' if blind_eligible else 'blind excluded'} | "
                f"{', '.join(reasons) if reasons else 'none'} |"
            )
        persona_lines = [
            (
                "Formal Persona-v2 main results use 50 persona_id clusters, provider "
                "as a repeated measurement, text-only modality, and disjoint "
                "pilot/main task rosters. These simulated response-channel estimates "
                "measure instruction following and robustness, not human behavioral "
                "validity."
            ),
            "",
            *lifecycle_lines,
            "",
            "| Controlled paired outcome | Oriented effect | 95% cluster CI | Denominator |",
            "|---|---:|---:|---|",
        ]
        for metric in PERSONA_EFFECT_ORIENTATIONS:
            row = effect_index.get(metric)
            if not isinstance(row, Mapping):
                raise BinderError(f"bound Persona-v2 effect is missing: {metric}")
            denominator_range = row.get("paired_persona_denominator_range")
            denominator_text = (
                f"paired persona denominator {denominator_range[0]}-{denominator_range[1]}"
                if isinstance(denominator_range, list) and len(denominator_range) == 2
                else "paired persona denominator NA"
            )
            persona_lines.append(
                f"| {effect_labels[metric]} | {_percent_display(row.get('estimate'))} "
                f"({row['orientation']}) | {_ci_display(row.get('ci95'))} | "
                f"{denominator_text} |"
            )

        agreement = blind.get("agreement")
        stability = blind.get("stability_provider_equal_aggregate")
        failure = blind.get("technical_or_schema_failure_rate")
        judge_analysis = judge.get("analysis")
        if not all(
            isinstance(value, Mapping)
            for value in (agreement, stability, failure, judge_analysis)
        ):
            raise BinderError("bound Persona-v2 blind or judge surface is incomplete")
        agreement_rows = agreement.get("pairs")
        if not isinstance(agreement_rows, list) or not agreement_rows:
            raise BinderError("bound Persona-v2 blind agreement is not estimable")
        agreement_values = [float(row["exact_agreement"]) for row in agreement_rows]
        answer_stability = stability.get("answer")
        judge_label = judge_analysis.get("pairwise_label_agreement")
        if not isinstance(answer_stability, Mapping):
            raise BinderError("bound Persona-v2 stability is missing")
        judge_status = judge_analysis.get("status")
        if judge_status == "complete":
            if not isinstance(judge_label, Mapping):
                raise BinderError("bound Persona-v2 judge agreement is missing")
            judge_row = (
                "| Cross-model judge label agreement (exploratory) | "
                f"{judge_label['exact_agreement_numerator']}/"
                f"{judge_label['denominator']} "
                f"({_percent_display(judge_label.get('exact_agreement'))}) | "
                "descriptive agreement only |"
            )
            judge_summary = (
                f"judge agreement={judge_label['exact_agreement_numerator']}/"
                f"{judge_label['denominator']} "
                f"({_percent_display(judge_label.get('exact_agreement'))}; exploratory)"
            )
        elif judge_status == "partial_missing_judge":
            if judge_label is not None:
                raise BinderError("bound missing-judge profile reports pairwise agreement")
            judge_row = (
                "| Automated error coding (exploratory) | GPT-only exploratory coding | "
                "Claude judge was unavailable; pairwise judge agreement was not estimable |"
            )
            judge_summary = (
                "GPT-only coding; Claude unavailable; pairwise agreement not estimable"
            )
        elif judge_status == "missing_all_judges":
            if judge_label is not None:
                raise BinderError("bound missing-all-judges profile reports agreement")
            unavailable = (
                "Automated error coding was unavailable: GPT judge failed and Claude "
                "judge was unavailable; pairwise judge agreement was not estimable"
            )
            judge_row = (
                "| Automated error coding (exploratory) | Unavailable | "
                f"{unavailable} |"
            )
            judge_summary = unavailable
        elif judge_status == "not_applicable_zero_cases":
            if judge_label is not None:
                raise BinderError("bound zero-case profile reports pairwise agreement")
            not_applicable = (
                "exploratory adjudication was not applicable because no outcome-blind "
                "cases survived structural exclusion"
            )
            judge_row = (
                "| Automated error coding (exploratory) | Not applicable | "
                f"{not_applicable} |"
            )
            judge_summary = not_applicable
        else:
            raise BinderError("bound Persona-v2 judge status is invalid")
        persona_lines.extend(
            [
                "",
                "| Blind/exploratory outcome | Estimate | Denominator/interval |",
                "|---|---:|---|",
                (
                    "| Blind technical/schema failure rate | "
                    f"{_percent_display(failure.get('estimate'))} | "
                    f"{_ci_display(failure.get('ci95'))} |"
                ),
                (
                    "| Blind terminal exact agreement | "
                    f"{_percent_display(min(agreement_values))} to "
                    f"{_percent_display(max(agreement_values))} | "
                    f"{len(agreement_values)} provider pairs; 100 paired terminal subjects each |"
                ),
                (
                    "| Repeat answer stability | "
                    f"{_percent_display(answer_stability.get('point_estimate'))} | "
                    f"{_ci_display(answer_stability.get('ci95'))} |"
                ),
                judge_row,
                "",
                '<figure class="persona-v2-composite">',
                (
                    '<div class="persona-v2-panels" style="display:grid;'
                    'grid-template-columns:repeat(3,minmax(0,1fr));gap:3mm;'
                    'break-inside:avoid-page">'
                ),
                '<img src="assets/persona_v2/controlled_composition.png" alt="Controlled response composition by provider and response arm">',
                '<img src="assets/persona_v2/blind_terminal_agreement.png" alt="Blind terminal agreement across eligible providers">',
                '<img src="assets/persona_v2/blind_output_stability.png" alt="Frozen repeat-output stability by provider">',
                "</div>",
                (
                    "<figcaption>Figure. Compact Persona-v2 composite: controlled "
                    "composition, blind terminal agreement, and repeat stability. "
                    "Detailed provider panels remain in the supplement.</figcaption>"
                ),
                "</figure>",
            ]
        )
        persona_markdown = "\n".join(persona_lines)
        persona_abstract = (
            "Persona v2: conditional-accuracy shift="
            f"{_percent_display(conditional_effect.get('estimate'))} "
            f"(95% CI {_ci_display(conditional_effect.get('ci95'))}); "
            "blind agreement="
            f"{_percent_display(min(agreement_values))}-"
            f"{_percent_display(max(agreement_values))}, stability="
            f"{_percent_display(answer_stability.get('point_estimate'))}, failure="
            f"{_percent_display(failure.get('estimate'))}; {judge_summary}."
        )
    elif persona.get("status") == "pending_formal_w3_artifacts":
        persona_markdown = (
            "Formal Persona-v2 W3 artifacts are pending; no Persona-v2 result value "
            "is bound."
        )
        persona_abstract = "Persona-v2 results are not bound in this draft."
    else:
        raise BinderError("Persona-v2 slot status is invalid")

    if p2.get("status") == "bound":
        boundary = p2.get("library_boundary")
        overlap = p2.get("exact_overlap_boundary")
        overall = p2.get("overall")
        if (
            not isinstance(boundary, Mapping)
            or not isinstance(overlap, Mapping)
            or not isinstance(overall, list)
        ):
            raise BinderError("bound P2 slot is incomplete")
        p2_lines = [
            (
                f"Illustrative P2 is supply-bound to {boundary['node_count']} nodes "
                f"and {boundary['trusted_exact_segment_assignments']} trusted exact "
                "node-chunk assignments spanning "
                f"{boundary['unique_chunk_count']} unique chunks. The fixed analytic "
                "overlap contains "
                f"{overlap['target_count']} targets, {overlap['candidate_row_count']} "
                f"candidate rows, and {overlap['physical_source_count']} physical "
                "sources under a 600-second analytic budget. This illustrative "
                "analysis does not estimate "
                "learning benefit or external remediation quality."
            ),
            "",
            "P/U role-compatible dose minutes are unavailable and remain null because "
            "the frozen library has no role-compatible dose definition for those states.",
            "",
            "| Arm | Mechanically mismatched selected min | Missed available-supply min | Diagnostic structural-failure node fraction |",
            "|---|---:|---:|---:|",
        ]
        for row in overall:
            if not isinstance(row, Mapping):
                raise BinderError("bound P2 overall row is invalid")
            p2_lines.append(
                f"| {row['arm']} | {float(row['mismatched_selected_minutes']):.3f} | "
                f"{float(row['missed_available_supply_minutes']):.3f} | "
                f"{float(row['structural_failure_node_fraction']):.3f} |"
            )
        p2_lines.extend(
            [
                "",
                (
                    "Arm C's failed-node fraction of 0.500 is a diagnostic structural "
                    "failure and is retained intention-to-treat; it is not a zero-cost "
                    "prescription. Missed available-supply minutes instead quantify "
                    "supply scarcity within the fixed trusted library."
                ),
            ]
        )
        p2_markdown = "\n".join(p2_lines)
        overall_index = {
            str(row["arm"]): row for row in overall if isinstance(row, Mapping)
        }
        if set(overall_index) != set(P2_ARMS):
            raise BinderError("bound P2 abstract arm roster drifted")
        p2_abstract = (
            f"P2 (illustrative; {overlap['target_count']} targets/"
            f"{overlap['candidate_row_count']} rows/"
            f"{overlap['physical_source_count']} sources): mismatched minutes "
            "A/B/C="
            f"{float(overall_index['A']['mismatched_selected_minutes']):.3f}/"
            f"{float(overall_index['B']['mismatched_selected_minutes']):.3f}/"
            f"{float(overall_index['C']['mismatched_selected_minutes']):.3f}; "
            "Arm-C structural-failure fraction="
            f"{float(overall_index['C']['structural_failure_node_fraction']):.3f}."
        )
    elif p2.get("status") == "not_requested":
        p2_markdown = "P2 is not bound in this binder generation."
        p2_abstract = "P2 is not bound in this draft."
    else:
        raise BinderError("P2 slot status is invalid")

    h1_a = _bound_metric(binder, "p_rescue.full.matched.b15.arm_A")
    h1_b = _bound_metric(binder, "p_rescue.full.matched.b15.arm_B")
    h1_difference = _bound_metric(
        binder, "h1.primary.matched.b15.rescue_A_minus_B"
    )
    h1_nr_a = _bound_metric(binder, "h1.no_repeat.matched.b15.arm_A")
    h1_nr_b = _bound_metric(binder, "h1.no_repeat.matched.b15.arm_B")
    h1_nr_difference = _bound_metric(
        binder, "h1.no_repeat.matched.b15.rescue_A_minus_B"
    )
    h2_a = _bound_metric(binder, "c_misdiagnosis.full.matched.b9.arm_A")
    h2_b = _bound_metric(binder, "c_misdiagnosis.full.matched.b9.arm_B")
    h2_c = _bound_metric(binder, "c_misdiagnosis.full.matched.b9.arm_C")
    h2_harm = _bound_metric(binder, "h2.primary.matched.b9.harm_C_minus_A")
    h2_no_harm = _bound_metric(
        binder, "h2.primary.matched.b9.no_harm_A_minus_B"
    )
    h2_nr_a = _bound_metric(binder, "h2.no_repeat.matched.b9.arm_A")
    h2_nr_b = _bound_metric(binder, "h2.no_repeat.matched.b9.arm_B")
    h2_nr_c = _bound_metric(binder, "h2.no_repeat.matched.b9.arm_C")
    h2_nr_harm = _bound_metric(binder, "h2.no_repeat.matched.b9.harm_C_minus_A")
    h2_nr_no_harm = _bound_metric(
        binder, "h2.no_repeat.matched.b9.no_harm_A_minus_B"
    )
    h3_accuracy = _bound_metric(
        binder, "h3.matched.b15.terminal_accuracy_A_minus_B"
    )
    h3_time_a = _bound_metric(binder, "h3.matched.b15.time_to_confidence.arm_A")
    h3_time_b = _bound_metric(binder, "h3.matched.b15.time_to_confidence.arm_B")
    h4_rescue = _bound_metric(binder, "h4.misspecified.b15.rescue_A_minus_B")
    h4_harm = _bound_metric(binder, "h4.misspecified.b9.harm_C_minus_A")
    h4_no_harm = _bound_metric(binder, "h4.misspecified.b9.no_harm_A_minus_B")
    h4_rescue_degradation = _bound_metric(
        binder, "h4.degradation.h1_rescue.matched_minus_misspecified"
    )
    h4_harm_degradation = _bound_metric(
        binder, "h4.degradation.h2_harm.matched_minus_misspecified"
    )
    h4_no_harm_degradation = _bound_metric(
        binder, "h4.degradation.h2_no_harm.matched_minus_misspecified"
    )
    direct_count = _bound_metric(
        binder,
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.direct_count",
    )
    prerequisite_count = _bound_metric(
        binder,
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.prerequisite_count",
    )
    prerequisite_share = _bound_metric(
        binder,
        "exploratory_posthoc.prerequisite.truth_P.matched.b15.arm_A.prerequisite_share",
    )

    programmatic_abstract = (
        f"At 15 items, adaptive P-state correct convergence was "
        f"{_metric_display(h1_a)} versus {_metric_display(h1_b)} for the local ladder; "
        f"the paired difference was {100.0 * float(h1_difference['value']):.1f} points "
        f"({_metric_ci_text(h1_difference)}). H1 was partially supported because the "
        "adaptive rate missed the frozen 50% criterion. At nine items, fixed insertion "
        f"minus adaptive C-state misdiagnosis was {100.0 * float(h2_harm['value']):.1f} "
        f"points, while adaptive minus local was {100.0 * float(h2_no_harm['value']):.1f} "
        "points; H2 was not supported. H3 and the direction-only H4 check were supported."
    )
    primary_results = "\n\n".join(
        [
            "### 4.2 Prerequisite-gap rescue and reasoning-chain harm",
            (
                f"At budget 15 on the {h1_a['n_target']}-target eligible stress support, "
                f"Arm A P-state correct confident convergence was {_metric_display(h1_a)} "
                f"({_metric_ci_text(h1_a)}), versus {_metric_display(h1_b)} "
                f"({_metric_ci_text(h1_b)}) for Arm B. The paired A-minus-B contrast was "
                f"{_contrast_text(h1_difference)}. The direction favored adaptive probing, "
                "but the Arm-A rate did not reach the frozen 50% threshold. H1 was "
                "partially supported."
            ),
            (
                f"On the no-repeat common support ({h1_nr_a['n_target']} targets; "
                f"{_display_number(h1_nr_a['denominator'])} paired cases), Arm A was "
                f"{_metric_display(h1_nr_a)} and Arm B was {_metric_display(h1_nr_b)}; "
                f"the contrast was {_contrast_text(h1_nr_difference)}. This jointly "
                "changed support-and-repeat estimand was much weaker, so its difference "
                "from the broad estimate cannot be attributed to repetition alone."
            ),
            (
                f"At budget 9 on the {h2_a['n_target']}-target support, C-state "
                f"misdiagnosis was {_metric_display(h2_a)} for Arm A, "
                f"{_metric_display(h2_b)} for Arm B, and {_metric_display(h2_c)} for Arm C. "
                f"C minus A was {_contrast_text(h2_harm)}. A minus B was "
                f"{_contrast_text(h2_no_harm)}, exceeding the frozen +5-point no-harm "
                "margin. H2 was not supported."
            ),
            (
                f"On the {h2_nr_a['n_target']}-target no-repeat common support, the "
                f"corresponding rates were {_metric_display(h2_nr_a)} for A, "
                f"{_metric_display(h2_nr_b)} for B, and {_metric_display(h2_nr_c)} for C. "
                f"C minus A was {_contrast_text(h2_nr_harm)}, whereas A minus B was "
                f"{_contrast_text(h2_nr_no_harm)}."
            ),
            (
                "![Figure 1. P-state correct convergence across item budgets. The figure "
                "is reused from the verified programmatic artifact and does not include "
                "Persona-v2 evidence.](generated/fig-p-rescue-png-c36a76849139.png)"
            ),
            (
                "![Figure 2. C-state misdiagnosis across item budgets. Arm labels refer "
                "only to the programmatic study.]"
                "(generated/fig-c-probe-harm-png-e5e22d30fb2c.png)"
            ),
            "### 4.3 Adaptive sanity check and misspecification",
            (
                f"On the full {h3_accuracy['n_target']}-target matched set at budget 15, "
                f"Arm A exceeded Arm B in terminal accuracy by "
                f"{_contrast_text(h3_accuracy)}. Median time to confidence was "
                f"{_display_number(h3_time_a['value'])} items for A "
                f"({_metric_ci_text(h3_time_a, percent=False)}) and "
                f"{_display_number(h3_time_b['value'])} items for B "
                f"({_metric_ci_text(h3_time_b, percent=False)}). H3 was supported; this "
                "is an implementation sanity check, not an algorithmic novelty claim."
            ),
            (
                f"Under the misspecified generator, P-state A minus B was "
                f"{_contrast_text(h4_rescue)}; C-state C minus A was "
                f"{_contrast_text(h4_harm)}; and C-state A minus B was "
                f"{_contrast_text(h4_no_harm)}. Matched minus misspecified degradation "
                f"was {100.0 * float(h4_rescue_degradation['value']):.1f} points for H1 "
                f"({_metric_ci_text(h4_rescue_degradation)}), "
                f"{100.0 * float(h4_harm_degradation['value']):.1f} points for fixed-quota "
                f"harm ({_metric_ci_text(h4_harm_degradation)}), and "
                f"{100.0 * float(h4_no_harm_degradation['value']):.1f} points for adaptive "
                f"minus local ({_metric_ci_text(h4_no_harm_degradation)}). H4 was "
                "supported under its direction-only rule; this does not establish "
                "robustness to untested generator families."
            ),
            (
                "![Figure 3. Matched and misspecified contrast estimates. The perturbation "
                "is one declared synthetic sensitivity condition.]"
                "(generated/fig-matched-vs-misspecified-png-39c4270e4169.png)"
            ),
        ]
    )

    convergence_lines.extend(
        [
            "",
            (
                "Rows are not cross-compared: target-set hashes are "
                + "; ".join(
                    f"{pair_id}={pairs[pair_id]['target_set_hash']}"
                    for pair_id in PAIR_SPECS
                )
                + "."
            ),
            (
                "The full-support difference is arithmetic only because no dedicated "
                "bootstrap interval is bound. The eligible-support difference retains "
                "its paired interval."
            ),
            (
                f"Post-hoc selector composition on the eligible support averaged "
                f"{float(direct_count['value']):.2f} direct and "
                f"{float(prerequisite_count['value']):.2f} prerequisite items by budget "
                f"15; prerequisite share was {_percent_display(prerequisite_share['value'])} "
                f"({_metric_ci_text(prerequisite_share)}). This selector-stopping mismatch "
                "is exploratory and does not revise H1."
            ),
        ]
    )

    slots: dict[str, Any] = {
        "schema_version": "yher.journal_binder.manuscript_slots.v1",
        "programmatic_abstract_results_markdown": programmatic_abstract,
        "hypothesis_decisions_markdown": "\n".join(decision_lines),
        "same_support_convergence_markdown": "\n".join(convergence_lines),
        "execution_integrity_markdown": execution_markdown,
        "primary_h1_h4_results_markdown": primary_results,
        "bound_abstract_results_markdown": f"{persona_abstract} {p2_abstract}",
        "persona_v2_markdown": persona_markdown,
        "p2_markdown": p2_markdown,
    }
    slots["content_sha256"] = hashlib.sha256(canonical_json_bytes(slots)).hexdigest()
    return slots


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )


def _write_synced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_binder(binder: Mapping[str, Any], output_dir: Path | str) -> dict[str, Any]:
    root = Path(output_dir)
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    slots = render_manuscript_slots(binder)
    binder_bytes = _pretty_json_bytes(binder)
    slots_bytes = _pretty_json_bytes(slots)
    binder_sha = hashlib.sha256(binder_bytes).hexdigest()
    slots_sha = hashlib.sha256(slots_bytes).hexdigest()
    generation_id = hashlib.sha256(
        canonical_json_bytes(
            {"journal_binder.json": binder_sha, "manuscript_slots.json": slots_sha}
        )
    ).hexdigest()[:24]
    artifacts = [
        {
            "filename": "journal_binder.json",
            "bytes": len(binder_bytes),
            "sha256": binder_sha,
        },
        {
            "filename": "manuscript_slots.json",
            "bytes": len(slots_bytes),
            "sha256": slots_sha,
        },
    ]
    artifact_manifest = {
        "schema_version": "yher.journal_support_binder.output.v2",
        "generation_id": generation_id,
        "generation_set_sha256": hashlib.sha256(
            canonical_json_bytes(artifacts)
        ).hexdigest(),
        "artifacts": artifacts,
    }
    manifest_bytes = _pretty_json_bytes(artifact_manifest)
    final_dir = generations / generation_id
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=generations))
    link_temp = root / f".current.{generation_id}.{os.getpid()}.tmp"
    try:
        _write_synced(staging / "journal_binder.json", binder_bytes)
        _write_synced(staging / "manuscript_slots.json", slots_bytes)
        _write_synced(staging / "artifact_manifest.json", manifest_bytes)
        _fsync_directory(staging)
        if final_dir.exists():
            existing = final_dir / "artifact_manifest.json"
            if not existing.is_file() or existing.read_bytes() != manifest_bytes:
                raise BinderError("existing binder generation content drifted")
            shutil.rmtree(staging)
        else:
            os.replace(staging, final_dir)
            _fsync_directory(generations)
        if link_temp.exists() or link_temp.is_symlink():
            link_temp.unlink()
        os.symlink(Path("generations") / generation_id, link_temp)
        os.replace(link_temp, root / "current")
        _fsync_directory(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if link_temp.exists() or link_temp.is_symlink():
            link_temp.unlink()
        raise
    return artifact_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--paper-artifact-manifest", type=Path)
    parser.add_argument("--persona-v2-dir", type=Path)
    parser.add_argument("--p2-dir", type=Path, default=DEFAULT_P2_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binder = build_binder(
        raw_manifest_path=args.raw_manifest,
        registry_path=args.registry,
        results_path=args.results,
        paper_artifact_manifest_path=args.paper_artifact_manifest,
        persona_v2_dir=args.persona_v2_dir,
        p2_dir=args.p2_dir,
    )
    manifest = write_binder(binder, args.output)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
