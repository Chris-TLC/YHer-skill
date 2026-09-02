from __future__ import annotations

from copy import deepcopy

import pytest

from analysis.dataset import DatasetContractError
from analysis.views import rebuild_views_from_events, validate_raw_views


def _event(position: int, *, confidence: bool = False) -> dict[str, object]:
    belief = [0.25, 0.25, 0.25, 0.25]
    belief[min(position // 7, 3)] += 0.2
    belief[0] -= 0.2
    if confidence:
        belief = [0.8, 0.05, 0.1, 0.05]
    family = f"family-{position if position <= 19 else position - 19}"
    item = f"item-{position if position <= 19 else position - 19}"
    return {
        "position": position,
        "posterior_belief": belief,
        "production_should_stop": confidence or position == 25,
        "production_confidence_should_stop": confidence,
        "family_id": family,
        "item_id": item,
        "role": "prereq" if position % 3 == 0 else "local",
    }


def _brier_fields() -> dict[str, object]:
    return {
        "held_out_outcomes": {
            "held-item-1": True,
            "held-item-2": False,
        },
        "held_out_family_scores": [
            {
                "family_id": "held-1",
                "item_id": "held-item-1",
                "outcome": True,
                "p_hat": 0.8,
                "squared_error": 0.04,
            },
            {
                "family_id": "held-2",
                "item_id": "held-item-2",
                "outcome": False,
                "p_hat": 0.3,
                "squared_error": 0.09,
            },
        ],
        "held_out_brier": 0.065,
    }


def _raw_view(canonical: dict[str, object]) -> dict[str, object]:
    return {**canonical, **_brier_fields()}


def test_rebuilds_early_stop_views_from_event_prefix_and_carries_forward() -> None:
    journey = {
        "truth": "M",
        "terminal_reason": "confidence",
        "held_out_outcomes": {
            "held-item-1": True,
            "held-item-2": False,
        },
        "events": [_event(i, confidence=i == 4) for i in range(1, 5)],
    }

    views = rebuild_views_from_events(journey)

    assert [view["nominal_budget"] for view in views] == [9, 15, 25]
    assert all(view["actual_administered_count"] == 4 for view in views)
    assert all(view["converged"] is True for view in views)
    assert all(view["convergence_time"] == 4 for view in views)
    assert all(view["terminal_reason"] == "confidence" for view in views)
    assert all(view["carried_forward"] is True for view in views)
    assert all(view["exact_item_repeat_count"] == 0 for view in views)


def test_rebuilds_budget_views_and_repeat_metrics_from_exact_event_prefixes() -> None:
    journey = {
        "truth": "U",
        "terminal_reason": "budget_exhausted",
        "events": [_event(i) for i in range(1, 26)],
    }

    views = rebuild_views_from_events(journey)

    assert [view["actual_administered_count"] for view in views] == [9, 15, 25]
    assert [view["terminal_reason"] for view in views] == [
        "checkpoint_nonterminal",
        "checkpoint_nonterminal",
        "budget_exhausted",
    ]
    assert [view["argmax"] for view in views] == ["P", "C", "U"]
    assert views[-1]["unique_item_count"] == 19
    assert views[-1]["exact_item_repeat_count"] == 6
    assert views[-1]["family_repeat_fraction"] == pytest.approx(6 / 25)
    assert [view["prerequisite_count"] for view in views] == [3, 5, 8]
    assert views[0]["prerequisite_share"] == pytest.approx(1 / 3)
    assert [view["direct_count"] for view in views] == [6, 10, 17]


def test_raw_views_must_equal_event_reconstruction_and_atomic_brier() -> None:
    journey = {
        "truth": "M",
        "terminal_reason": "confidence",
        "held_out_outcomes": {
            "held-item-1": True,
            "held-item-2": False,
        },
        "events": [_event(i, confidence=i == 4) for i in range(1, 5)],
    }
    canonical = rebuild_views_from_events(journey)
    journey["views"] = [_raw_view(view) for view in canonical]

    validate_raw_views(journey, canonical)

    drifted = deepcopy(journey)
    drifted["views"][0]["argmax"] = "U"
    with pytest.raises(DatasetContractError, match="raw view mismatch"):
        validate_raw_views(drifted, canonical)

    drifted = deepcopy(journey)
    drifted["views"][0]["held_out_family_scores"][0]["squared_error"] = 0.5
    with pytest.raises(DatasetContractError, match="Brier atom"):
        validate_raw_views(drifted, canonical)

    drifted = deepcopy(journey)
    drifted["views"][0]["held_out_brier"] = 0.5
    with pytest.raises(DatasetContractError, match="held_out_brier"):
        validate_raw_views(drifted, canonical)

    drifted = deepcopy(journey)
    drifted["views"][0]["held_out_family_scores"][1]["family_id"] = "held-1"
    with pytest.raises(DatasetContractError, match="two distinct held-out families"):
        validate_raw_views(drifted, canonical)

    drifted = deepcopy(journey)
    drifted["views"][0]["held_out_family_scores"][1]["item_id"] = "other-item"
    with pytest.raises(DatasetContractError, match="held-out item IDs"):
        validate_raw_views(drifted, canonical)

    drifted = deepcopy(journey)
    drifted["views"][0]["held_out_family_scores"][0]["outcome"] = False
    with pytest.raises(DatasetContractError, match="held-out outcome"):
        validate_raw_views(drifted, canonical)

    drifted = deepcopy(journey)
    drifted["views"][1]["held_out_family_scores"][0]["family_id"] = "held-other"
    with pytest.raises(DatasetContractError, match="fixed across budget views"):
        validate_raw_views(drifted, canonical)

    drifted = deepcopy(journey)
    drifted["events"][0]["item_id"] = "held-item-1"
    with pytest.raises(DatasetContractError, match="held-out item.*administered"):
        validate_raw_views(drifted, canonical)

    drifted = deepcopy(journey)
    drifted["events"][0]["family_id"] = "held-1"
    with pytest.raises(DatasetContractError, match="held-out family.*administered"):
        validate_raw_views(drifted, canonical)


def test_stop_flags_are_recomputed_from_posterior_and_local_event_count() -> None:
    journey = {
        "target_node": "T",
        "truth": "M",
        "terminal_reason": "confidence",
        "events": [_event(i, confidence=i == 4) for i in range(1, 5)],
    }
    rebuild_views_from_events(journey)

    drifted = deepcopy(journey)
    drifted["events"][0]["production_should_stop"] = True
    drifted["events"][0]["production_confidence_should_stop"] = True
    with pytest.raises(DatasetContractError, match="stored stop flag"):
        rebuild_views_from_events(drifted)


def test_item_25_uses_budget_26_to_distinguish_exhaustion_from_confidence() -> None:
    journey = {
        "target_node": "T",
        "truth": "U",
        "terminal_reason": "budget_exhausted",
        "events": [_event(i) for i in range(1, 26)],
    }

    views = rebuild_views_from_events(journey)

    assert views[-1]["converged"] is False
    assert views[-1]["terminal_reason"] == "budget_exhausted"

    drifted = deepcopy(journey)
    drifted["events"][-1]["production_confidence_should_stop"] = True
    with pytest.raises(DatasetContractError, match="stored stop flag"):
        rebuild_views_from_events(drifted)


def test_short_budget_exhaustion_is_an_invalid_incomplete_journey() -> None:
    journey = {
        "target_node": "T",
        "truth": "U",
        "terminal_reason": "budget_exhausted",
        "events": [_event(1), _event(2)],
    }

    views = rebuild_views_from_events(journey)

    assert all(view["valid"] is False for view in views)
    assert all(view["incomplete"] is True for view in views)
    assert all(
        view["terminal_reason"] == "incomplete_budget_exhausted" for view in views
    )


def test_empty_structural_failure_rebuilds_three_invalid_nc_views() -> None:
    journey = {
        "truth": "P",
        "terminal_reason": "structural_failure",
        "events": [],
    }

    canonical = rebuild_views_from_events(journey)

    assert [view["nominal_budget"] for view in canonical] == [9, 15, 25]
    assert all(view["actual_administered_count"] == 0 for view in canonical)
    assert all(view["argmax"] is None for view in canonical)
    assert all(view["valid"] is False for view in canonical)
    assert all(view["incomplete"] is True for view in canonical)
    assert all(view["terminal_reason"] == "structural_failure" for view in canonical)
    assert all(view["prerequisite_count"] == 0 for view in canonical)
    assert all(view["prerequisite_share"] == 0.0 for view in canonical)
    assert all(view["direct_count"] == 0 for view in canonical)
    journey["views"] = [dict(view) for view in canonical]
    validate_raw_views(journey, canonical)
