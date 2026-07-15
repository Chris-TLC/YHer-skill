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


def _load_registry(path: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise BinderError("metric registry is not valid JSON") from exc
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
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise BinderError(f"{label} is not valid JSON") from exc
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
) -> tuple[dict[str, str], dict[str, Any]]:
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
    return source_updates, execution_integrity


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
    path = root / relative
    if not path.is_file():
        raise BinderError(f"Persona-v2 bundle file is missing: {role}")
    actual_hash = sha256_file(path)
    if expected_hash != actual_hash:
        raise BinderError(f"Persona-v2 bundle file SHA-256 drift: {role}")
    payload, _ = _load_json_object(path, label=f"Persona-v2 {role}")
    return path, payload, actual_hash


def bind_persona_v2_artifacts(bundle_dir: Path | str) -> dict[str, Any]:
    root = Path(bundle_dir)
    binding_path = root / "binding_manifest.json"
    if not binding_path.is_file():
        raise BinderError("Persona-v2 formal W3 binding manifest is missing")
    binding, binding_bytes = _load_json_object(
        binding_path, label="Persona-v2 binding manifest"
    )
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
    expected_roles = {
        "analysis_results",
        "analysis_artifact_manifest",
        "phase_provenance",
        "runtime_task_manifest",
        "mapping_manifest",
    }
    if not isinstance(files, Mapping) or set(files) != expected_roles:
        raise BinderError("Persona-v2 binding manifest file set is incomplete")
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
        provider = row.get("provider")
        status_counts = row.get("status_counts")
        missing_ids = row.get("missing_task_ids")
        if (
            provider not in providers
            or provider in lifecycle_index
            or row.get("provider_lifecycle") not in allowed_lifecycles
            or row.get("expected_count") != main_task_count
            or not isinstance(missing_ids, list)
            or not set(missing_ids).issubset(set(phase_tasks["main"]))
            or row.get("missing_count") != len(missing_ids)
            or row.get("present_count") + row.get("missing_count") != main_task_count
            or not isinstance(status_counts, Mapping)
            or sum(int(value) for value in status_counts.values())
            != row.get("present_count")
        ):
            raise BinderError("Persona-v2 provider lifecycle does not reconcile")
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
    if not isinstance(result.get("controlled"), Mapping) or not isinstance(
        result.get("blind"), Mapping
    ):
        raise BinderError("Persona-v2 controlled or blind result surface is missing")

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
        if relative.is_absolute() or ".." in relative.parts or relative_value in seen_artifacts:
            raise BinderError("Persona-v2 analysis artifact path is unsafe or duplicated")
        path = analysis_root / relative
        if not path.is_file():
            raise BinderError(f"Persona-v2 analysis artifact is missing: {relative_value}")
        artifact_bytes = path.read_bytes()
        if (
            row.get("sha256") != hashlib.sha256(artifact_bytes).hexdigest()
            or row.get("size") != len(artifact_bytes)
        ):
            raise BinderError(f"Persona-v2 analysis artifact drift: {relative_value}")
        if path.resolve() == paths["analysis_results"].resolve():
            bound_result = True
        seen_artifacts.add(relative_value)
    if not bound_result:
        raise BinderError("Persona-v2 analysis results are not bound by output manifest")

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
        "controlled": dict(result["controlled"]),
        "blind": dict(result["blind"]),
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
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise BinderError(f"P2 source file SHA-256 drift: {role}")
    return path


def _p2_read_candidates(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BinderError(
                    f"P2 trusted candidate source has invalid JSON at line {line_number}"
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
    if not summary_path.is_file() or not output_manifest_path.is_file():
        raise BinderError("P2 summary or output manifest is missing")
    if not input_manifest_path.is_file():
        raise BinderError("P2 input manifest is required")
    summary, summary_bytes = _load_json_object(summary_path, label="P2 summary")
    input_manifest, input_bytes = _load_json_object(
        input_manifest_path, label="P2 input manifest"
    )
    output_manifest, output_manifest_bytes = _load_json_object(
        output_manifest_path, label="P2 output manifest"
    )
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
        artifact_path = p2_dir / relative
        if not artifact_path.is_file():
            raise BinderError(f"P2 output artifact is missing: {filename}")
        artifact_bytes = artifact_path.read_bytes()
        if (
            row.get("sha256") != hashlib.sha256(artifact_bytes).hexdigest()
            or row.get("bytes") != len(artifact_bytes)
        ):
            raise BinderError(f"P2 output artifact hash or size drift: {filename}")
        seen_artifacts.add(filename)
    if not {"summary.json", "input_manifest.json"}.issubset(seen_artifacts):
        raise BinderError("P2 output manifest does not bind summary and input manifest")

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
    if require_complete:
        if results is None or results_file is None:
            raise BinderError("results artifact is required for a complete binding")
        artifact_path = (
            Path(paper_artifact_manifest_path)
            if paper_artifact_manifest_path is not None
            else results_file.parent / "artifact_manifest.json"
        )
        source_updates, execution_integrity = _validate_complete_h1_h4_evidence(
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
        else bind_persona_v2_artifacts(Path(persona_v2_dir))
    )
    p2_bound = _bind_p2(None if p2_dir is None else Path(p2_dir))
    if (
        p2_bound.get("status") == "bound"
        and p2_bound.get("source_hashes", {}).get("h1_h4_raw_manifest_sha256")
        != raw_source_hashes["raw_manifest_sha256"]
    ):
        raise BinderError("P2 and H1-H4 binder use different raw manifests")
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
        "same_support_pairs": pairs,
        "persona_v2": persona_v2,
        "p2": p2_bound,
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
    return str(int(number)) if number.is_integer() else f"{number:.6g}"


def _metric_display(metric: Mapping[str, Any]) -> str:
    return (
        f"{_display_number(metric['numerator'])}/"
        f"{_display_number(metric['denominator'])} "
        f"({100.0 * float(metric['value']):.1f}%)"
    )


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
        decision_lines.append(f"| {hypothesis} | {row['decision']} |")

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
        execution_markdown = (
            f"{execution['intended_journey_count']} intended journeys; "
            f"{execution['valid_journey_count']} valid; "
            f"{execution['structural_failure_count']} structural failures; "
            f"{execution['schema_invalid_count']} schema-invalid."
        )
    else:
        execution_markdown = "Execution-integrity counts are not bound in this draft."

    if persona.get("status") == "bound_formal_w3":
        pilot = persona.get("pilot_exclusion")
        if not isinstance(pilot, Mapping) or pilot.get("task_rosters_disjoint") is not True:
            raise BinderError("bound Persona-v2 slot lacks pilot exclusion proof")
        persona_markdown = (
            "Formal Persona-v2 main results are machine-bound for 50 persona_id "
            "clusters, with provider as a repeated measurement, text-only modality, "
            "and disjoint pilot/main task rosters."
        )
    elif persona.get("status") == "pending_formal_w3_artifacts":
        persona_markdown = (
            "Formal Persona-v2 W3 artifacts are pending; no Persona-v2 result value "
            "is bound."
        )
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
                f"Illustrative P2 is supply-bound to {boundary['node_count']} nodes/"
                f"{boundary['trusted_exact_segment_assignments']} trusted exact segment "
                f"assignments; the analysis overlap is {overlap['target_count']} targets/"
                f"{overlap['candidate_row_count']} rows/{overlap['physical_source_count']} "
                "physical sources under a 600-second budget."
            ),
            "",
            "| Arm | Mechanically mismatched selected min | Missed available-supply min |",
            "|---|---:|---:|",
        ]
        for row in overall:
            if not isinstance(row, Mapping):
                raise BinderError("bound P2 overall row is invalid")
            p2_lines.append(
                f"| {row['arm']} | {float(row['mismatched_selected_minutes']):.3f} | "
                f"{float(row['missed_available_supply_minutes']):.3f} |"
            )
        p2_markdown = "\n".join(p2_lines)
    elif p2.get("status") == "not_requested":
        p2_markdown = "P2 is not bound in this binder generation."
    else:
        raise BinderError("P2 slot status is invalid")

    slots: dict[str, Any] = {
        "schema_version": "yher.journal_binder.manuscript_slots.v1",
        "hypothesis_decisions_markdown": "\n".join(decision_lines),
        "same_support_convergence_markdown": "\n".join(convergence_lines),
        "execution_integrity_markdown": execution_markdown,
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
