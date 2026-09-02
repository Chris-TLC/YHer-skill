from __future__ import annotations

from dataclasses import replace

import pytest

from analysis.metrics import (
    AnalysisRow,
    contrast_metric,
    confusion_matrix,
    rate_metric,
    severe_metrics,
    time_or_budget_plus_one,
)


def _row(
    target: str,
    replicate: int,
    *,
    truth: str,
    argmax: str,
    converged: bool,
) -> AnalysisRow:
    return AnalysisRow(
        target=target,
        truth=truth,
        condition="matched",
        replicate=replicate,
        arm="A",
        budget=9,
        argmax=argmax,
        converged=converged,
        convergence_time=4 if converged else None,
        actual_administered_count=9,
        held_out_brier=0.1 * (replicate + 1),
        exact_item_repeat_fraction=0.0,
        family_repeat_fraction=0.0,
        h1_h2_eligible=True,
        common_support_no_repeat=True,
    )


def test_misdiagnosis_uses_converged_wrong_numerator_and_all_journeys_denominator() -> None:
    rows = (
        _row("T1", 0, truth="C", argmax="P", converged=True),
        _row("T1", 1, truth="C", argmax="P", converged=False),
        _row("T2", 0, truth="C", argmax="C", converged=True),
        _row("T2", 1, truth="C", argmax="C", converged=False),
    )

    metric = rate_metric(
        "h2.arm_a.misdiagnosis",
        rows,
        outcome="misdiagnosis",
        raw_hash="raw",
    )

    assert metric.numerator == 1
    assert metric.denominator == 4
    assert metric.value == pytest.approx(0.25)
    assert metric.weighting == "equal_target_then_replicate"
    assert metric.n_target == 2
    assert metric.n_pair == 4
    assert metric.raw_hash == "raw"


def test_confusions_include_all_terminal_argmax_and_nonconvergence_column() -> None:
    rows = (
        _row("T", 0, truth="M", argmax="U", converged=True),
        _row("T", 1, truth="M", argmax="P", converged=False),
        _row("T", 2, truth="U", argmax="U", converged=True),
    )

    terminal = confusion_matrix(rows, decision=False)
    decision = confusion_matrix(rows, decision=True)

    assert terminal.shape == (4, 4)
    assert terminal.count("M", "U") == 1
    assert terminal.count("M", "P") == 1
    assert decision.shape == (4, 5)
    assert decision.count("M", "U") == 1
    assert decision.count("M", "NC") == 1
    assert decision.row_denominator("M") == 2


def test_structural_failure_is_excluded_from_both_confusion_estimands() -> None:
    valid = _row("T", 0, truth="M", argmax="M", converged=True)
    structural = replace(
        _row("T", 1, truth="M", argmax="U", converged=False),
        valid=False,
    )

    terminal = confusion_matrix((valid, structural), decision=False)
    decision = confusion_matrix((valid, structural), decision=True)

    assert terminal.row_denominator("M") == 1
    assert terminal.count("M", "U") == 0
    assert decision.row_denominator("M") == 1
    assert decision.count("M", "NC") == 0


def test_invalid_rows_are_excluded_from_rate_denominators() -> None:
    valid = _row("T", 0, truth="C", argmax="P", converged=True)
    invalid = replace(
        _row("T", 1, truth="C", argmax="P", converged=True),
        valid=False,
    )

    metric = rate_metric(
        "valid-only",
        (valid, invalid),
        outcome="misdiagnosis",
        raw_hash="raw",
    )

    assert metric.numerator == 1
    assert metric.denominator == 1


def test_severe_swap_reports_the_three_frozen_denominators() -> None:
    rows = (
        _row("T", 0, truth="M", argmax="U", converged=True),
        _row("T", 1, truth="M", argmax="U", converged=False),
        _row("T", 2, truth="U", argmax="M", converged=True),
        _row("T", 3, truth="U", argmax="P", converged=True),
        _row("T", 4, truth="P", argmax="M", converged=True),
        _row("T", 5, truth="C", argmax="U", converged=False),
    )

    metrics = severe_metrics(rows, raw_hash="raw")

    assert metrics["all_journeys"].numerator == 3
    assert metrics["all_journeys"].denominator == 6
    assert metrics["all_converged_journeys"].numerator == 2
    assert metrics["all_converged_journeys"].denominator == 4
    assert metrics["truth_M_or_U_all_journeys"].numerator == 3
    assert metrics["truth_M_or_U_all_journeys"].denominator == 4

    with_structural = rows + (
        replace(_row("T", 6, truth="M", argmax="U", converged=False), valid=False),
    )
    metrics = severe_metrics(with_structural, raw_hash="raw")
    assert metrics["all_journeys"].denominator == 6
    assert metrics["all_journeys"].numerator == 3
    assert metrics["all_converged_journeys"].denominator == 4
    assert metrics["all_converged_journeys"].numerator == 2
    assert metrics["truth_M_or_U_all_journeys"].denominator == 4
    assert metrics["truth_M_or_U_all_journeys"].numerator == 3


def test_h3_time_scalar_encodes_nonconvergence_as_budget_plus_one() -> None:
    converged = _row("T", 0, truth="M", argmax="M", converged=True)
    nonconverged = _row("T", 1, truth="M", argmax="P", converged=False)

    assert time_or_budget_plus_one(converged) == 4
    assert time_or_budget_plus_one(nonconverged) == 10


def test_point_contrast_is_replicate_paired_and_target_weighted() -> None:
    rows = (
        _row("T1", 0, truth="P", argmax="P", converged=True),
        _row("T1", 0, truth="P", argmax="U", converged=False),
        _row("T2", 0, truth="P", argmax="U", converged=False),
        _row("T2", 0, truth="P", argmax="P", converged=True),
    )
    rows = tuple(
        row.__class__(**{**row.__dict__, "arm": arm})
        for row, arm in zip(rows, ("A", "B", "A", "B"), strict=True)
    )

    metric = contrast_metric(
        "paired",
        rows,
        left_arm="A",
        right_arm="B",
        outcome="correct_convergence",
        raw_hash="raw",
    )

    assert metric.value == 0.0
    assert metric.numerator == 0.0
    assert metric.denominator == 2
    assert metric.n_pair == 2


def test_paired_contrast_drops_the_whole_pair_when_either_arm_is_invalid() -> None:
    rows = (
        replace(_row("T1", 0, truth="P", argmax="P", converged=True), arm="A"),
        replace(
            _row("T1", 0, truth="P", argmax="U", converged=False),
            arm="B",
            valid=False,
        ),
        replace(_row("T2", 0, truth="P", argmax="P", converged=True), arm="A"),
        replace(_row("T2", 0, truth="P", argmax="U", converged=False), arm="B"),
    )

    metric = contrast_metric(
        "paired-valid-only",
        rows,
        left_arm="A",
        right_arm="B",
        outcome="correct_convergence",
        raw_hash="raw",
    )

    assert metric.value == 1.0
    assert metric.denominator == 1
    assert metric.n_target == 1
