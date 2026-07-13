"""Pre-specified H1-H4 decision branches from the frozen analysis plan."""

from __future__ import annotations

from typing import Literal


Decision = Literal["supported", "partially_supported", "not_supported"]


def decide_h1(*, a_rate: float, rescue_point: float, rescue_ci_low: float) -> Decision:
    if a_rate >= 0.50 and rescue_ci_low > 0.0:
        return "supported"
    if a_rate < 0.50 and rescue_point <= 0.0:
        return "not_supported"
    return "partially_supported"


def decide_h2(
    *,
    harm_point: float,
    harm_ci_low: float,
    no_harm_point: float,
    no_harm_ci_high: float,
) -> Decision:
    if harm_ci_low > 0.0 and no_harm_ci_high < 0.05:
        return "supported"
    if harm_point <= 0.0 or no_harm_point >= 0.05:
        return "not_supported"
    return "partially_supported"


def decide_h3(
    *,
    accuracy_point: float,
    accuracy_ci_low: float,
    median_a: float,
    median_b: float,
) -> Decision:
    if accuracy_ci_low >= 0.0 and median_a <= median_b:
        return "supported"
    if accuracy_point < 0.0 and median_a > median_b:
        return "not_supported"
    return "partially_supported"


def decide_h4(*, rescue_point: float, harm_point: float) -> Decision:
    if rescue_point > 0.0 and harm_point > 0.0:
        return "supported"
    if rescue_point <= 0.0 and harm_point <= 0.0:
        return "not_supported"
    return "partially_supported"


def h1_branch_reason(
    *, a_rate: float, rescue_point: float, rescue_ci_low: float
) -> str:
    if a_rate >= 0.50 and rescue_ci_low > 0.0:
        return "a_rate_at_least_0_50_and_rescue_ci_strictly_positive"
    if a_rate < 0.50 and rescue_point <= 0.0:
        return "a_rate_below_0_50_and_rescue_point_nonpositive"
    if a_rate < 0.50:
        return "rescue_direction_positive_but_rate_threshold_not_met"
    return "rate_threshold_met_but_rescue_interval_inconclusive"


def h2_branch_reason(
    harm_point: float,
    harm_ci_low: float,
    no_harm_point: float,
    no_harm_ci_high: float,
) -> str:
    if harm_ci_low > 0.0 and no_harm_ci_high < 0.05:
        return "harm_ci_strictly_positive_and_no_harm_ci_below_0_05"
    if harm_point <= 0.0:
        return "harm_nonpositive"
    if no_harm_point >= 0.05:
        return "A_inferior_to_B_margin"
    if harm_ci_low <= 0.0 and no_harm_ci_high >= 0.05:
        return "harm_and_no_harm_intervals_inconclusive"
    if harm_ci_low <= 0.0:
        return "harm_interval_inconclusive"
    return "no_harm_interval_inconclusive"


def h3_branch_reason(
    accuracy_point: float,
    accuracy_ci_low: float,
    median_a: float,
    median_b: float,
) -> str:
    if accuracy_ci_low >= 0.0 and median_a <= median_b:
        return "accuracy_ci_nonnegative_and_median_A_no_longer_than_B"
    if accuracy_point < 0.0 and median_a > median_b:
        return "accuracy_direction_and_median_time_favor_B"
    if accuracy_ci_low < 0.0 and median_a > median_b:
        return "accuracy_interval_and_median_time_inconclusive"
    if accuracy_ci_low < 0.0:
        return "accuracy_interval_inconclusive"
    return "median_time_direction_favors_B"


def h4_branch_reason(rescue_point: float, harm_point: float) -> str:
    if rescue_point > 0.0 and harm_point > 0.0:
        return "h1_rescue_and_h2_harm_directions_persist"
    if rescue_point <= 0.0 and harm_point <= 0.0:
        return "neither_h1_rescue_nor_h2_harm_direction_persists"
    if rescue_point > 0.0:
        return "only_h1_rescue_direction_persists"
    return "only_h2_harm_direction_persists"
