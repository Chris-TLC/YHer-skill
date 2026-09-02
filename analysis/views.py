"""Independent checkpoint reconstruction from immutable event streams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from engine import selector

from .dataset import DatasetContractError


STATES = ("M", "P", "C", "U")
DEFAULT_BUDGETS = (9, 15, 25)


def rebuild_views_from_events(
    journey: Mapping[str, Any],
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> tuple[dict[str, Any], ...]:
    """Rebuild checkpoint fields without trusting the stored ``views`` array."""

    events = journey.get("events")
    if not isinstance(events, list):
        raise DatasetContractError("canonical view reconstruction requires an events list")
    if not events:
        if not str(journey.get("terminal_reason", "")).startswith(
            "structural_failure"
        ):
            raise DatasetContractError("an empty event stream must be a structural failure")
        reason = str(journey["terminal_reason"])
        return tuple(
            {
                "nominal_budget": int(budget),
                "actual_administered_count": 0,
                "belief": None,
                "argmax": None,
                "converged": False,
                "convergence_time": None,
                "terminal_reason": reason,
                "carried_forward": False,
                "valid": False,
                "incomplete": True,
                "severe_misdiagnosis_all_terminal": False,
                "severe_misdiagnosis_converged_only": None,
                "unique_item_count": 0,
                "unique_family_count": 0,
                "exact_item_repeat_count": 0,
                "family_repeat_count": 0,
                "exact_item_repeat_fraction": 0.0,
                "family_repeat_fraction": 0.0,
                "prerequisite_count": 0,
                "prerequisite_share": 0.0,
                "direct_count": 0,
            }
            for budget in budgets
        )
    for expected_position, event in enumerate(events, start=1):
        if not isinstance(event, Mapping) or event.get("position") != expected_position:
            raise DatasetContractError(
                f"event positions are not contiguous at {expected_position}"
            )
    confidence_positions = _validate_and_recompute_stop_flags(journey, events)
    convergence_time = min(confidence_positions) if confidence_positions else None
    actual = len(events)
    truth = str(journey.get("truth"))
    top_terminal_reason = str(journey.get("terminal_reason"))
    if convergence_time is not None:
        if convergence_time != actual or top_terminal_reason != "confidence":
            raise DatasetContractError(
                "event stream terminal reason differs from recomputed confidence stop"
            )
        journey_valid = True
        exclusion_reason = None
    elif top_terminal_reason == "budget_exhausted":
        journey_valid = actual == 25
        exclusion_reason = None if journey_valid else "incomplete_budget_exhausted"
    elif top_terminal_reason.startswith("structural_failure"):
        journey_valid = False
        exclusion_reason = top_terminal_reason
    else:
        raise DatasetContractError(
            "journey terminal reason differs from recomputed stop state"
        )
    max_budget = max(int(value) for value in budgets)
    output: list[dict[str, Any]] = []

    for raw_budget in budgets:
        budget = int(raw_budget)
        view_actual = min(actual, budget)
        event = events[view_actual - 1]
        belief = _belief(event.get("posterior_belief"))
        argmax = STATES[max(range(len(STATES)), key=belief.__getitem__)]
        converged = convergence_time is not None and convergence_time <= budget
        prefix = events[:view_actual]
        repeats = _repeat_metrics(prefix)
        prerequisite_count = sum(event.get("role") == "prereq" for event in prefix)
        direct_count = sum(event.get("role") == "local" for event in prefix)
        if prerequisite_count + direct_count != view_actual:
            raise DatasetContractError("event role must be local or prereq")
        terminal_reason = (
            str(exclusion_reason)
            if exclusion_reason is not None
            else "confidence"
            if converged
            else top_terminal_reason
            if actual < budget or budget == max_budget
            else "checkpoint_nonterminal"
        )
        severe = (truth, argmax) in {("M", "U"), ("U", "M")}
        output.append(
            {
                "nominal_budget": budget,
                "actual_administered_count": view_actual,
                "belief": list(belief),
                "argmax": argmax,
                "converged": converged,
                "convergence_time": convergence_time if converged else None,
                "terminal_reason": terminal_reason,
                "carried_forward": (
                    top_terminal_reason == "confidence" and actual < budget
                ),
                "valid": journey_valid,
                "incomplete": not journey_valid,
                "severe_misdiagnosis_all_terminal": severe,
                "severe_misdiagnosis_converged_only": severe if converged else None,
                "prerequisite_count": prerequisite_count,
                "prerequisite_share": prerequisite_count / view_actual,
                "direct_count": direct_count,
                **repeats,
            }
        )
    return tuple(output)


def validate_raw_views(
    journey: Mapping[str, Any],
    canonical: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Require stored views to match reconstruction and recompute Brier from atoms."""

    raw_views = journey.get("views")
    if not isinstance(raw_views, list) or len(raw_views) != len(canonical):
        raise DatasetContractError("raw view count does not match canonical views")
    core_keys = (
        "nominal_budget",
        "actual_administered_count",
        "belief",
        "argmax",
        "converged",
        "convergence_time",
        "terminal_reason",
        "carried_forward",
        "valid",
        "incomplete",
        "severe_misdiagnosis_all_terminal",
        "severe_misdiagnosis_converged_only",
        "unique_item_count",
        "unique_family_count",
        "exact_item_repeat_count",
        "family_repeat_count",
        "exact_item_repeat_fraction",
        "family_repeat_fraction",
    )
    held_out_pairs: tuple[tuple[str, str], ...] | None = None
    for raw, rebuilt in zip(raw_views, canonical, strict=True):
        if not isinstance(raw, Mapping):
            raise DatasetContractError("raw view is not an object")
        for key in core_keys:
            if not _equal(raw.get(key), rebuilt.get(key)):
                raise DatasetContractError(
                    f"raw view mismatch for budget {rebuilt['nominal_budget']}: {key}"
                )
        pairs = _validate_brier_atoms(raw, journey.get("held_out_outcomes"))
        if pairs is None:
            continue
        if held_out_pairs is None:
            held_out_pairs = pairs
        elif pairs != held_out_pairs:
            raise DatasetContractError(
                "held-out family/item pairs are not fixed across budget views"
            )

    if held_out_pairs is None:
        return ()
    held_items = {item_id for _, item_id in held_out_pairs}
    held_families = {family_id for family_id, _ in held_out_pairs}
    administered_items = {str(event.get("item_id")) for event in journey["events"]}
    administered_families = {
        str(event.get("family_id")) for event in journey["events"]
    }
    if held_items & administered_items:
        raise DatasetContractError("held-out item was administered in the journey")
    if held_families & administered_families:
        raise DatasetContractError("held-out family was administered in the journey")
    return held_out_pairs


def _validate_and_recompute_stop_flags(
    journey: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> list[int]:
    target = str(journey.get("target_node") or "__single_target__")
    direct_count = 0
    confidence_positions: list[int] = []
    for event in events:
        position = int(event["position"])
        role = event.get("role")
        if role == "local":
            direct_count += 1
        elif role != "prereq":
            raise DatasetContractError("event role must be local or prereq")
        belief = _belief(event.get("posterior_belief"))
        beliefs = {target: np.asarray(belief, dtype=float)}
        direct_answers = {target: direct_count}
        expected_policy = selector.should_stop(
            beliefs,
            [target],
            direct_answers=direct_answers,
            budget_items=25,
            asked=position,
        )
        expected_confidence = selector.should_stop(
            beliefs,
            [target],
            direct_answers=direct_answers,
            budget_items=26,
            asked=position,
        )
        stored_policy = event.get("production_should_stop")
        stored_confidence = event.get("production_confidence_should_stop")
        if (
            not isinstance(stored_policy, bool)
            or not isinstance(stored_confidence, bool)
            or stored_policy is not expected_policy
            or stored_confidence is not expected_confidence
        ):
            raise DatasetContractError(
                f"stored stop flag differs from event-prefix recomputation at {position}"
            )
        if expected_confidence:
            confidence_positions.append(position)
    return confidence_positions


def _belief(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != len(STATES):
        raise DatasetContractError("posterior belief must have four states")
    belief = tuple(float(item) for item in value)
    if not all(math.isfinite(item) and item >= 0.0 for item in belief):
        raise DatasetContractError("posterior belief contains invalid values")
    if not math.isclose(sum(belief), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise DatasetContractError("posterior belief does not sum to one")
    return belief  # type: ignore[return-value]


def _repeat_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = [str(event.get("item_id")) for event in events]
    families = [str(event.get("family_id")) for event in events]
    actual = len(events)
    unique_items = len(set(items))
    unique_families = len(set(families))
    item_repeats = actual - unique_items
    family_repeats = actual - unique_families
    return {
        "unique_item_count": unique_items,
        "unique_family_count": unique_families,
        "exact_item_repeat_count": item_repeats,
        "family_repeat_count": family_repeats,
        "exact_item_repeat_fraction": item_repeats / actual if actual else 0.0,
        "family_repeat_fraction": family_repeats / actual if actual else 0.0,
    }


def _validate_brier_atoms(
    view: Mapping[str, Any], journey_outcomes: object
) -> tuple[tuple[str, str], ...] | None:
    atoms = view.get("held_out_family_scores")
    if view.get("valid") is False and not atoms and view.get("held_out_brier") is None:
        return None
    if not isinstance(journey_outcomes, Mapping) or len(journey_outcomes) != 2:
        raise DatasetContractError("journey must bind exactly two held-out outcomes")
    if not all(
        isinstance(item_id, str) and item_id and isinstance(outcome, bool)
        for item_id, outcome in journey_outcomes.items()
    ):
        raise DatasetContractError("journey held-out outcomes are malformed")
    if view.get("held_out_outcomes") != journey_outcomes:
        raise DatasetContractError("view held-out outcomes differ from the journey")
    if not isinstance(atoms, list) or len(atoms) != 2:
        raise DatasetContractError("held-out Brier requires exactly two atomic scores")
    family_ids = [atom.get("family_id") for atom in atoms if isinstance(atom, Mapping)]
    item_ids = [atom.get("item_id") for atom in atoms if isinstance(atom, Mapping)]
    if len(family_ids) != 2 or any(
        not isinstance(value, str) or not value for value in family_ids
    ) or len(set(family_ids)) != 2:
        raise DatasetContractError("Brier atoms require two distinct held-out families")
    if len(item_ids) != 2 or any(
        not isinstance(value, str) or not value for value in item_ids
    ) or len(set(item_ids)) != 2 or set(item_ids) != set(journey_outcomes):
        raise DatasetContractError("Brier atom held-out item IDs differ from outcomes")
    squared_errors: list[float] = []
    for atom in atoms:
        if not isinstance(atom, Mapping) or not isinstance(atom.get("outcome"), bool):
            raise DatasetContractError("Brier atom is malformed")
        if atom["outcome"] is not journey_outcomes[atom["item_id"]]:
            raise DatasetContractError("Brier atom held-out outcome mismatch")
        p_hat = float(atom.get("p_hat"))
        if not math.isfinite(p_hat) or not 0.0 <= p_hat <= 1.0:
            raise DatasetContractError("Brier atom p_hat is outside [0, 1]")
        expected = (p_hat - float(atom["outcome"])) ** 2
        if not _equal(atom.get("squared_error"), expected):
            raise DatasetContractError("Brier atom squared_error mismatch")
        squared_errors.append(expected)
    expected_brier = sum(squared_errors) / len(squared_errors)
    if not _equal(view.get("held_out_brier"), expected_brier):
        raise DatasetContractError("held_out_brier does not equal atomic mean")
    return tuple(sorted(zip(family_ids, item_ids, strict=True)))


def _equal(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(
        right, (int, float)
    ) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-12)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right
