from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from analysis.bootstrap import (
    BOOTSTRAP_SEED,
    bootstrap_category_rate_distributions,
    bootstrap_contrast_distribution,
    bootstrap_event_mean_distribution,
    bootstrap_median_time_distribution,
    bootstrap_mean_distribution,
    bootstrap_rate_distribution,
    bootstrap_ratio_distribution,
    bootstrap_value_distribution,
    percentile_interval,
    replicate_weights,
)
from analysis.metrics import AnalysisEvent, AnalysisRow


def _row(
    target: str,
    replicate: int,
    arm: str,
    *,
    correct: bool,
    condition: str = "matched",
) -> AnalysisRow:
    return AnalysisRow(
        target=target,
        truth="P",
        condition=condition,
        replicate=replicate,
        arm=arm,
        budget=15,
        argmax="P" if correct else "U",
        converged=correct,
        convergence_time=5 if correct else None,
        actual_administered_count=15,
        held_out_brier=0.1 + replicate / 100,
        exact_item_repeat_fraction=0.0,
        family_repeat_fraction=0.0,
        h1_h2_eligible=True,
        common_support_no_repeat=True,
    )


def test_replicate_weights_are_seeded_by_target_truth_condition_only() -> None:
    assert BOOTSTRAP_SEED == 2026071301
    first = replicate_weights(
        seed=BOOTSTRAP_SEED,
        iteration=7,
        target="T",
        truth="P",
        condition="matched",
        replicate_ids=(0, 1, 2, 3),
    )
    repeated = replicate_weights(
        seed=BOOTSTRAP_SEED,
        iteration=7,
        target="T",
        truth="P",
        condition="matched",
        replicate_ids=(0, 1, 2, 3),
    )
    misspecified = replicate_weights(
        seed=BOOTSTRAP_SEED,
        iteration=7,
        target="T",
        truth="P",
        condition="misspecified",
        replicate_ids=(0, 1, 2, 3),
    )

    assert first == repeated
    assert sum(first.values()) == 4
    assert first != misspecified


def test_bootstrap_rate_keeps_target_set_fixed_and_equally_weighted() -> None:
    rows = tuple(
        _row(target, replicate, "A", correct=target == "T1")
        for target in ("T1", "T2")
        for replicate in range(4)
    )

    values = bootstrap_rate_distribution(
        rows,
        outcome="correct_convergence",
        iterations=20,
        seed=BOOTSTRAP_SEED,
    )

    assert values == (0.5,) * 20


def test_bootstrap_contrast_preserves_paired_replicates_across_arms() -> None:
    rows = (
        _row("T", 0, "A", correct=True),
        _row("T", 0, "B", correct=False),
        _row("T", 1, "A", correct=False),
        _row("T", 1, "B", correct=True),
    )

    first = bootstrap_contrast_distribution(
        rows,
        left_arm="A",
        right_arm="B",
        outcome="correct_convergence",
        iterations=50,
        seed=BOOTSTRAP_SEED,
    )
    second = bootstrap_contrast_distribution(
        rows,
        left_arm="A",
        right_arm="B",
        outcome="correct_convergence",
        iterations=50,
        seed=BOOTSTRAP_SEED,
    )

    assert first == second
    assert set(first) <= {-1.0, 0.0, 1.0}
    assert {-1.0, 0.0, 1.0} <= set(first)


def test_bootstrap_estimands_exclude_invalid_rows_and_invalid_pairs() -> None:
    rate_rows = (
        _row("T", 0, "A", correct=True),
        replace(_row("T", 1, "A", correct=False), valid=False),
    )
    contrast_rows = (
        _row("T", 0, "A", correct=True),
        _row("T", 0, "B", correct=False),
        _row("T", 1, "A", correct=False),
        replace(_row("T", 1, "B", correct=True), valid=False),
    )

    assert bootstrap_rate_distribution(
        rate_rows,
        outcome="correct_convergence",
        iterations=10,
    ) == (1.0,) * 10
    assert bootstrap_contrast_distribution(
        contrast_rows,
        left_arm="A",
        right_arm="B",
        outcome="correct_convergence",
        iterations=10,
    ) == (1.0,) * 10


def test_percentile_interval_uses_deterministic_linear_interpolation() -> None:
    assert percentile_interval((0.0, 1.0, 2.0, 3.0, 4.0), alpha=0.20) == (0.4, 3.6)


def test_numeric_bootstrap_reuses_the_same_cell_weights_for_heldout_values() -> None:
    rows = tuple(_row("T", replicate, "A", correct=True) for replicate in range(4))
    weights = replicate_weights(
        seed=BOOTSTRAP_SEED,
        iteration=0,
        target="T",
        truth="P",
        condition="matched",
        replicate_ids=(0, 1, 2, 3),
    )
    expected = sum(
        weights[row.replicate] * row.held_out_brier
        for row in rows
        if row.replicate in weights
    ) / 4

    values = bootstrap_mean_distribution(
        rows,
        field="held_out_brier",
        iterations=1,
        seed=BOOTSTRAP_SEED,
    )

    assert values == pytest.approx((expected,))


def test_category_bootstrap_generates_a_complete_distribution_in_one_pass() -> None:
    rows = (
        _row("T", 0, "A", correct=True),
        _row("T", 1, "A", correct=False),
    )

    distributions = bootstrap_category_rate_distributions(
        rows,
        categories=(5, 16),
        category=lambda row: 5 if row.converged else 16,
        iterations=20,
        seed=BOOTSTRAP_SEED,
    )

    assert set(distributions) == {5, 16}
    assert all(len(values) == 20 for values in distributions.values())
    assert all(
        converged + nonconverged == 1.0
        for converged, nonconverged in zip(
            distributions[5], distributions[16], strict=True
        )
    )


def test_median_time_bootstrap_encodes_nonconvergence_as_budget_plus_one() -> None:
    rows = (
        _row("T", 0, "A", correct=True),
        _row("T", 1, "A", correct=False),
    )

    values = bootstrap_median_time_distribution(
        rows,
        iterations=20,
        seed=BOOTSTRAP_SEED,
    )

    assert len(values) == 20
    assert set(values) <= {5.0, 10.5, 16.0}


def test_median_time_bootstrap_keeps_targets_equal_after_invalid_exclusions() -> None:
    rows = (
        _row("T1", 0, "A", correct=True),
        _row("T1", 1, "A", correct=True),
        _row("T2", 0, "A", correct=False),
        replace(_row("T2", 1, "A", correct=True), valid=False),
    )

    values = bootstrap_median_time_distribution(
        rows,
        iterations=20,
        seed=BOOTSTRAP_SEED,
    )

    assert values == (10.5,) * 20


def test_ratio_bootstrap_resamples_all_journeys_before_applying_denominator() -> None:
    rows = (
        _row("T", 0, "A", correct=True),
        _row("T", 1, "A", correct=False),
        _row("T", 2, "A", correct=True),
        _row("T", 3, "A", correct=False),
    )
    weights = replicate_weights(
        seed=BOOTSTRAP_SEED,
        iteration=0,
        target="T",
        truth="P",
        condition="matched",
        replicate_ids=(0, 1, 2, 3),
    )
    expected_numerator = weights.get(0, 0)
    expected_denominator = weights.get(0, 0) + weights.get(2, 0)

    values = bootstrap_ratio_distribution(
        rows,
        numerator=lambda row: row.replicate == 0,
        denominator=lambda row: row.converged,
        iterations=1,
        seed=BOOTSTRAP_SEED,
    )

    assert values == (expected_numerator / expected_denominator,)


def test_sparse_ratio_bootstrap_records_undefined_draws_without_redrawing() -> None:
    rows = tuple(
        _row(target, replicate, "A", correct=replicate == 0)
        for target in ("T1", "T2")
        for replicate in range(8)
    )

    first_audit: dict[str, object] = {}
    first = bootstrap_ratio_distribution(
        rows,
        numerator=lambda row: row.replicate == 0,
        denominator=lambda row: row.replicate == 0,
        iterations=200,
        seed=BOOTSTRAP_SEED,
        audit=first_audit,
    )
    second_audit: dict[str, object] = {}
    second = bootstrap_ratio_distribution(
        rows,
        numerator=lambda row: row.replicate == 0,
        denominator=lambda row: row.replicate == 0,
        iterations=200,
        seed=BOOTSTRAP_SEED,
        audit=second_audit,
    )

    assert first == second
    assert first_audit == second_audit
    assert set(first) == {1.0}
    assert first_audit["attempted_iterations"] == 200
    assert first_audit["defined_iterations"] == len(first)
    assert first_audit["all_targets_undefined_iterations"] == 200 - len(first)
    assert first_audit["all_targets_undefined_iterations"] > 0
    assert first_audit["undefined_target_iterations"]["T1"] > 0
    assert first_audit["undefined_target_iterations"]["T2"] > 0
    assert first_audit["redraw_count"] == 0


def test_post_collection_static_audit_policy_is_explicit_and_not_outcome_tuned() -> None:
    policy_path = Path(__file__).parents[1] / "analysis/static_audit_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert policy["basis"] == "independent_static_review_before_result_interpretation"
    assert policy["frozen_analysis_plan_modified"] is False
    denominator = policy["conditional_metric_zero_denominator"]
    assert denominator["redraw_until_defined"] is False
    assert denominator["point_target"] == "record_NA_and_exclude_from_conditional_mean"
    assert denominator["all_targets_undefined_bootstrap_iteration"] == (
        "record_NA_and_exclude_from_percentile_sample"
    )
    lifecycle = policy["h5_provider_schema_lifecycle_clarification"]
    assert lifecycle["adopted_date"] == "2026-07-14"
    assert lifecycle["scope"] == "post_collection_schema_lifecycle_clarification"
    assert lifecycle["result_direction_used"] is False
    assert lifecycle["h1_h4_decision_rules_modified"] is False


def test_value_bootstrap_supports_paired_within_row_derived_effects() -> None:
    rows = tuple(_row("T", replicate, "A", correct=True) for replicate in range(4))
    weights = replicate_weights(
        seed=BOOTSTRAP_SEED,
        iteration=0,
        target="T",
        truth="P",
        condition="matched",
        replicate_ids=(0, 1, 2, 3),
    )
    expected = sum(
        weights.get(row.replicate, 0) * (1.0 if row.replicate == 0 else -1.0)
        for row in rows
    ) / 4

    values = bootstrap_value_distribution(
        rows,
        value=lambda row: 1.0 if row.replicate == 0 else -1.0,
        iterations=1,
        seed=BOOTSTRAP_SEED,
    )

    assert values == pytest.approx((expected,))


def _event(
    target: str,
    replicate: int,
    position: int,
    value: float,
) -> AnalysisEvent:
    return AnalysisEvent(
        target=target,
        truth="P",
        condition="misspecified",
        replicate=replicate,
        arm="A",
        position=position,
        item_type="mcq",
        generator_probability=value,
        production_probability=0.0,
    )


def test_event_bootstrap_resamples_whole_journey_clusters() -> None:
    events = (
        _event("T", 0, 1, 0.0),
        _event("T", 0, 2, 0.0),
        _event("T", 1, 1, 1.0),
        _event("T", 1, 2, 1.0),
    )

    values = bootstrap_event_mean_distribution(
        events,
        value=lambda event: event.generator_probability,
        iterations=200,
        seed=BOOTSTRAP_SEED,
    )

    assert set(values) == {0.0, 0.5, 1.0}
    assert 0.25 not in values and 0.75 not in values


def test_event_bootstrap_keeps_targets_equal_despite_unequal_event_counts() -> None:
    events = tuple(
        [
            _event("T1", 0, 1, 0.0),
            _event("T1", 0, 2, 0.0),
            _event("T1", 0, 3, 0.0),
            _event("T1", 0, 4, 0.0),
        ]
        + [_event("T2", 0, 1, 1.0)]
    )

    values = bootstrap_event_mean_distribution(
        events,
        value=lambda event: event.generator_probability,
        iterations=20,
        seed=BOOTSTRAP_SEED,
    )

    assert values == (0.5,) * 20
