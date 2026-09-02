"""Frozen point estimands and auditable metric records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Literal

from .dataset import DatasetContractError


STATES = ("M", "P", "C", "U")


@dataclass(frozen=True)
class AnalysisRow:
    target: str
    truth: str
    condition: str
    replicate: int
    arm: str
    budget: int
    argmax: str | None
    converged: bool
    convergence_time: int | None
    actual_administered_count: int
    held_out_brier: float | None
    exact_item_repeat_fraction: float
    family_repeat_fraction: float
    h1_h2_eligible: bool
    common_support_no_repeat: bool
    valid: bool = True
    prerequisite_count: int = 0
    prerequisite_share: float = 0.0
    direct_count: int = 0
    unique_item_count: int = 0
    unique_family_count: int = 0
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class AnalysisEvent:
    target: str
    truth: str
    condition: str
    replicate: int
    arm: str
    position: int
    item_type: str
    generator_probability: float
    production_probability: float
    valid: bool = True
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class MetricValue:
    metric_id: str
    value: float
    numerator: float
    denominator: int
    weighting: str
    n_target: int
    n_pair: int
    raw_hash: str
    ci_low: float | None = None
    ci_high: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConfusionMatrix:
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    counts: dict[tuple[str, str], int]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.columns))

    def count(self, truth: str, decision: str) -> int:
        return self.counts[(truth, decision)]

    def row_denominator(self, truth: str) -> int:
        return sum(self.count(truth, column) for column in self.columns)


Outcome = Literal[
    "correct_convergence",
    "misdiagnosis",
    "terminal_accuracy",
    "convergence",
    "severe_swap",
]


def rate_metric(
    metric_id: str,
    rows: Iterable[AnalysisRow],
    *,
    outcome: Outcome,
    raw_hash: str,
) -> MetricValue:
    indicator = outcome_indicator(outcome)
    return _rate(
        metric_id,
        tuple(row for row in rows if row.valid),
        indicator=indicator,
        raw_hash=raw_hash,
    )


def contrast_metric(
    metric_id: str,
    rows: Iterable[AnalysisRow],
    *,
    left_arm: str,
    right_arm: str,
    outcome: Outcome,
    raw_hash: str,
) -> MetricValue:
    indicator = outcome_indicator(outcome)
    pairs: dict[tuple[str, str, str, int, int], dict[str, AnalysisRow]] = defaultdict(dict)
    for row in rows:
        if row.arm not in {left_arm, right_arm}:
            continue
        key = (row.target, row.truth, row.condition, row.replicate, row.budget)
        if row.arm in pairs[key]:
            raise DatasetContractError(f"duplicate contrast row for {key!r}, {row.arm}")
        pairs[key][row.arm] = row
    if not pairs:
        raise DatasetContractError(f"metric {metric_id} has no paired rows")
    by_target: dict[str, list[float]] = defaultdict(list)
    numerator = 0.0
    valid_pair_count = 0
    for key, arms in pairs.items():
        if set(arms) != {left_arm, right_arm}:
            raise DatasetContractError(f"unpaired contrast row for {key!r}")
        if not all(row.valid for row in arms.values()):
            continue
        difference = float(indicator(arms[left_arm])) - float(
            indicator(arms[right_arm])
        )
        numerator += difference
        by_target[key[0]].append(difference)
        valid_pair_count += 1
    if not by_target:
        raise DatasetContractError(f"metric {metric_id} has no valid paired rows")
    target_values = [sum(values) / len(values) for values in by_target.values()]
    return MetricValue(
        metric_id=metric_id,
        value=sum(target_values) / len(target_values),
        numerator=numerator,
        denominator=valid_pair_count,
        weighting="equal_target_then_paired_replicate",
        n_target=len(by_target),
        n_pair=valid_pair_count,
        raw_hash=raw_hash,
    )


def severe_metrics(
    rows: Iterable[AnalysisRow], *, raw_hash: str
) -> dict[str, MetricValue]:
    values = tuple(row for row in rows if row.valid)
    return {
        "all_journeys": _rate(
            "severe.all_journeys",
            values,
            indicator=_severe,
            raw_hash=raw_hash,
        ),
        "all_converged_journeys": _rate(
            "severe.all_converged_journeys",
            tuple(row for row in values if row.converged),
            indicator=_severe,
            raw_hash=raw_hash,
        ),
        "truth_M_or_U_all_journeys": _rate(
            "severe.truth_M_or_U_all_journeys",
            tuple(row for row in values if row.truth in {"M", "U"}),
            indicator=_severe,
            raw_hash=raw_hash,
        ),
    }


def confusion_matrix(
    rows: Iterable[AnalysisRow], *, decision: bool
) -> ConfusionMatrix:
    columns = (*STATES, "NC") if decision else STATES
    counts = {(truth, column): 0 for truth in STATES for column in columns}
    for row in rows:
        if row.truth not in STATES:
            raise DatasetContractError("confusion row has an unknown state")
        if not row.valid:
            continue
        if not decision and row.argmax not in STATES:
            continue
        column = (
            row.argmax
            if row.valid and row.converged and row.argmax in STATES
            else "NC"
            if decision
            else row.argmax
        )
        counts[(row.truth, column)] += 1
    return ConfusionMatrix(rows=STATES, columns=columns, counts=counts)


def _rate(
    metric_id: str,
    rows: tuple[AnalysisRow, ...],
    *,
    indicator: Callable[[AnalysisRow], bool],
    raw_hash: str,
) -> MetricValue:
    if not rows:
        raise DatasetContractError(f"metric {metric_id} has an empty denominator")
    grouped: dict[str, list[AnalysisRow]] = defaultdict(list)
    for row in rows:
        grouped[row.target].append(row)
    target_rates: list[float] = []
    numerator = 0
    for target_rows in grouped.values():
        target_numerator = sum(bool(indicator(row)) for row in target_rows)
        numerator += target_numerator
        target_rates.append(target_numerator / len(target_rows))
    return MetricValue(
        metric_id=metric_id,
        value=sum(target_rates) / len(target_rates),
        numerator=float(numerator),
        denominator=len(rows),
        weighting="equal_target_then_replicate",
        n_target=len(grouped),
        n_pair=len(rows),
        raw_hash=raw_hash,
    )


def outcome_indicator(outcome: Outcome) -> Callable[[AnalysisRow], bool]:
    if outcome == "correct_convergence":
        return lambda row: row.valid and row.converged and row.argmax == row.truth
    if outcome == "misdiagnosis":
        return lambda row: row.valid and row.converged and row.argmax != row.truth
    if outcome == "terminal_accuracy":
        return lambda row: row.valid and row.argmax == row.truth
    if outcome == "convergence":
        return lambda row: row.valid and row.converged
    if outcome == "severe_swap":
        return _severe
    raise DatasetContractError(f"unknown outcome: {outcome}")


def _severe(row: AnalysisRow) -> bool:
    return row.valid and (row.truth, row.argmax) in {("M", "U"), ("U", "M")}


def time_or_budget_plus_one(row: AnalysisRow) -> int:
    """Encode non-convergence as right-censored at one item beyond the budget."""

    return int(row.convergence_time) if row.valid and row.converged else row.budget + 1
