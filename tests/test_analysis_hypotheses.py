from __future__ import annotations

from analysis.hypotheses import (
    decide_h1,
    decide_h2,
    decide_h3,
    decide_h4,
    h1_branch_reason,
    h2_branch_reason,
    h3_branch_reason,
    h4_branch_reason,
)


def test_h1_follows_the_frozen_ordered_branches() -> None:
    assert decide_h1(a_rate=0.50, rescue_point=0.20, rescue_ci_low=0.01) == "supported"
    assert decide_h1(a_rate=0.49, rescue_point=0.00, rescue_ci_low=-0.1) == "not_supported"
    assert decide_h1(a_rate=0.60, rescue_point=0.10, rescue_ci_low=-0.1) == "partially_supported"


def test_h2_follows_harm_and_no_harm_branches() -> None:
    assert (
        decide_h2(
            harm_point=0.20,
            harm_ci_low=0.01,
            no_harm_point=0.01,
            no_harm_ci_high=0.04,
        )
        == "supported"
    )
    assert (
        decide_h2(
            harm_point=0.00,
            harm_ci_low=-0.1,
            no_harm_point=0.01,
            no_harm_ci_high=0.04,
        )
        == "not_supported"
    )
    assert (
        decide_h2(
            harm_point=0.20,
            harm_ci_low=-0.01,
            no_harm_point=0.03,
            no_harm_ci_high=0.06,
        )
        == "partially_supported"
    )


def test_h3_is_explicitly_subordinate_accuracy_and_time_sanity() -> None:
    assert (
        decide_h3(
            accuracy_point=0.01,
            accuracy_ci_low=0.00,
            median_a=10,
            median_b=10,
        )
        == "supported"
    )
    assert (
        decide_h3(
            accuracy_point=-0.01,
            accuracy_ci_low=-0.02,
            median_a=11,
            median_b=10,
        )
        == "not_supported"
    )
    assert (
        decide_h3(
            accuracy_point=0.01,
            accuracy_ci_low=-0.02,
            median_a=11,
            median_b=10,
        )
        == "partially_supported"
    )


def test_h4_uses_direction_only_under_misspecification() -> None:
    assert decide_h4(rescue_point=0.01, harm_point=0.01) == "supported"
    assert decide_h4(rescue_point=0.00, harm_point=-0.01) == "not_supported"
    assert decide_h4(rescue_point=0.01, harm_point=-0.01) == "partially_supported"


def test_branch_reasons_are_stable_and_distinguish_h2_failure_modes() -> None:
    assert h1_branch_reason(a_rate=0.5, rescue_point=0.1, rescue_ci_low=0.01) == (
        "a_rate_at_least_0_50_and_rescue_ci_strictly_positive"
    )
    assert h2_branch_reason(0.0, -0.1, 0.1, 0.2) == "harm_nonpositive"
    assert h2_branch_reason(0.1, -0.1, 0.05, 0.2) == (
        "A_inferior_to_B_margin"
    )
    assert h2_branch_reason(0.1, -0.1, 0.01, 0.04) == (
        "harm_interval_inconclusive"
    )
    assert h2_branch_reason(0.1, 0.01, 0.01, 0.05) == (
        "no_harm_interval_inconclusive"
    )
    assert h3_branch_reason(0.1, 0.0, 9.0, 10.0) == (
        "accuracy_ci_nonnegative_and_median_A_no_longer_than_B"
    )
    assert h4_branch_reason(0.1, -0.1) == "only_h1_rescue_direction_persists"
