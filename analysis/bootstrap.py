"""Deterministic target-stratified, replicate-paired bootstrap."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import math
import random
from typing import Callable, Hashable, Iterable, Literal, Mapping, MutableMapping, TypeVar

import numpy as np

from .dataset import DatasetContractError
from .metrics import (
    AnalysisEvent,
    AnalysisRow,
    Outcome,
    outcome_indicator,
    time_or_budget_plus_one,
)


BOOTSTRAP_SEED = 2026071301
BOOTSTRAP_ITERATIONS = 10_000
NumericField = Literal[
    "held_out_brier",
    "exact_item_repeat_fraction",
    "family_repeat_fraction",
    "actual_administered_count",
    "prerequisite_count",
    "prerequisite_share",
    "direct_count",
    "unique_item_count",
    "unique_family_count",
]
Category = TypeVar("Category", bound=Hashable)


def percentile_interval(
    values: Iterable[float], *, alpha: float = 0.05
) -> tuple[float, float]:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise DatasetContractError("cannot compute an interval from no values")
    if not 0.0 < alpha < 1.0:
        raise DatasetContractError("alpha must be between zero and one")

    def percentile(probability: float) -> float:
        rank = probability * (len(ordered) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[lower]
        fraction = rank - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    return percentile(alpha / 2), percentile(1.0 - alpha / 2)


def replicate_weights(
    *,
    seed: int,
    iteration: int,
    target: str,
    truth: str,
    condition: str,
    replicate_ids: tuple[int, ...],
) -> dict[int, int]:
    """Draw one cell's replicate multiplicities, independent of arm/budget/metric."""

    ids = tuple(sorted(replicate_ids))
    if not ids or len(ids) != len(set(ids)):
        raise DatasetContractError("bootstrap replicate IDs must be unique and non-empty")
    material = (
        f"yher-analysis-bootstrap-v1|{seed}|{iteration}|{target}|{truth}|"
        f"{condition}"
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    draws = random.Random(derived_seed).choices(ids, k=len(ids))
    return dict(Counter(draws))


def bootstrap_rate_distribution(
    rows: Iterable[AnalysisRow],
    *,
    outcome: Outcome,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    indicator = outcome_indicator(outcome)
    cells = _single_arm_cells(
        tuple(row for row in rows if row.valid),
        indicator=indicator,
    )
    return _bootstrap_cell_mean(cells, iterations=iterations, seed=seed)


def bootstrap_mean_distribution(
    rows: Iterable[AnalysisRow],
    *,
    field: NumericField,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    cells: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        if not row.valid:
            continue
        cell = (row.target, row.truth, row.condition)
        if row.replicate in cells[cell]:
            raise DatasetContractError(
                f"duplicate numeric bootstrap row for {cell!r}, "
                f"replicate {row.replicate}"
            )
        value = float(getattr(row, field))
        if not math.isfinite(value):
            raise DatasetContractError(f"numeric bootstrap field {field} is not finite")
        cells[cell][row.replicate] = value
    if not cells:
        raise DatasetContractError("numeric bootstrap has no rows")

    return _bootstrap_cell_mean(cells, iterations=iterations, seed=seed)


def bootstrap_value_distribution(
    rows: Iterable[AnalysisRow],
    *,
    value: Callable[[AnalysisRow], float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    """Bootstrap a derived per-row scalar while preserving the frozen cell weights."""

    cells = _single_arm_cells(
        tuple(row for row in rows if row.valid),
        indicator=value,
    )
    if any(
        not math.isfinite(item)
        for cell_values in cells.values()
        for item in cell_values.values()
    ):
        raise DatasetContractError("derived bootstrap value is not finite")
    return _bootstrap_cell_mean(cells, iterations=iterations, seed=seed)


def bootstrap_event_mean_distribution(
    events: Iterable[AnalysisEvent],
    *,
    value: Callable[[AnalysisEvent], float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    """Target-stratified event mean with whole-journey clusters preserved.

    The frozen replicate draw is shared across arms. All selected events from an
    arm journey therefore receive one multiplicity, while targets remain equally
    weighted regardless of their event counts.
    """

    cells: dict[
        tuple[str, str, str], dict[int, list[float]]
    ] = defaultdict(dict)
    for event in events:
        if not event.valid:
            continue
        event_value = float(value(event))
        if not math.isfinite(event_value):
            raise DatasetContractError("event bootstrap value is not finite")
        cell = (event.target, event.truth, event.condition)
        aggregate = cells[cell].setdefault(event.replicate, [0.0, 0.0])
        aggregate[0] += event_value
        aggregate[1] += 1.0
    if not cells:
        raise DatasetContractError("event bootstrap has no valid events")

    targets = tuple(sorted({cell[0] for cell in cells}))
    output = np.zeros(iterations, dtype=float)
    for target in targets:
        target_numerator = np.zeros(iterations, dtype=float)
        target_denominator = np.zeros(iterations, dtype=float)
        for (cell_target, truth, condition), replicates in cells.items():
            if cell_target != target:
                continue
            ids = tuple(sorted(replicates))
            weights = _replicate_weight_matrix(
                seed, iterations, target, truth, condition, ids
            )
            target_numerator += weights @ np.asarray(
                [replicates[replicate][0] for replicate in ids], dtype=float
            )
            target_denominator += weights @ np.asarray(
                [replicates[replicate][1] for replicate in ids], dtype=float
            )
        if np.any(target_denominator <= 0):
            raise DatasetContractError(
                f"event bootstrap target {target!r} has an empty draw denominator"
            )
        output += target_numerator / target_denominator
    output /= len(targets)
    return tuple(float(item) for item in output)


def bootstrap_ratio_distribution(
    rows: Iterable[AnalysisRow],
    *,
    numerator: Callable[[AnalysisRow], bool],
    denominator: Callable[[AnalysisRow], bool],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    audit: MutableMapping[str, object] | None = None,
) -> tuple[float, ...]:
    """Resample full cells, then compute a target-equal numerator/denominator ratio."""

    cells: dict[
        tuple[str, str, str], dict[int, tuple[float, float]]
    ] = defaultdict(dict)
    for row in rows:
        if not row.valid:
            continue
        cell = (row.target, row.truth, row.condition)
        if row.replicate in cells[cell]:
            raise DatasetContractError(
                f"duplicate ratio bootstrap row for {cell!r}, "
                f"replicate {row.replicate}"
            )
        cells[cell][row.replicate] = (
            float(bool(numerator(row))),
            float(bool(denominator(row))),
        )
    if not cells:
        raise DatasetContractError("ratio bootstrap has no rows")

    targets = tuple(sorted({key[0] for key in cells}))
    target_values = np.full((iterations, len(targets)), np.nan, dtype=float)
    undefined_target_iterations: dict[str, int] = {}
    observed_undefined_targets: list[str] = []
    for target_index, target in enumerate(targets):
        target_numerator = np.zeros(iterations, dtype=float)
        target_denominator = np.zeros(iterations, dtype=float)
        observed_denominator = 0.0
        for (cell_target, truth, condition), values in cells.items():
            if cell_target != target:
                continue
            ids = tuple(sorted(values))
            weights = _replicate_weight_matrix(
                seed, iterations, target, truth, condition, ids
            )
            target_numerator += weights @ np.asarray(
                [values[replicate][0] for replicate in ids], dtype=float
            )
            target_denominator += weights @ np.asarray(
                [values[replicate][1] for replicate in ids], dtype=float
            )
            observed_denominator += sum(value[1] for value in values.values())
        if observed_denominator <= 0:
            observed_undefined_targets.append(target)
            undefined_target_iterations[target] = iterations
            continue
        defined = target_denominator > 0
        undefined_target_iterations[target] = int((~defined).sum())
        target_values[defined, target_index] = (
            target_numerator[defined] / target_denominator[defined]
        )
    defined_target_counts = np.sum(np.isfinite(target_values), axis=1)
    defined_iterations = defined_target_counts > 0
    if not np.any(defined_iterations):
        raise DatasetContractError("ratio bootstrap has no active target denominator")
    output = np.nansum(target_values[defined_iterations], axis=1) / (
        defined_target_counts[defined_iterations]
    )
    if audit is not None:
        audit.clear()
        audit.update(
            {
                "attempted_iterations": iterations,
                "defined_iterations": int(defined_iterations.sum()),
                "all_targets_undefined_iterations": int(
                    (~defined_iterations).sum()
                ),
                "undefined_target_iterations": {
                    target: undefined_target_iterations[target]
                    for target in sorted(undefined_target_iterations)
                },
                "observed_undefined_targets": sorted(observed_undefined_targets),
                "redraw_count": 0,
            }
        )
    return tuple(float(value) for value in output)


def bootstrap_category_rate_distributions(
    rows: Iterable[AnalysisRow],
    *,
    categories: tuple[Category, ...],
    category: Callable[[AnalysisRow], Category],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[Category, tuple[float, ...]]:
    """Bootstrap every category share with one set of cell resamples."""

    if not categories or len(categories) != len(set(categories)):
        raise DatasetContractError("bootstrap categories must be unique and non-empty")
    allowed = frozenset(categories)
    cells: dict[tuple[str, str, str], dict[int, Category]] = defaultdict(dict)
    for row in rows:
        if not row.valid:
            continue
        cell = (row.target, row.truth, row.condition)
        if row.replicate in cells[cell]:
            raise DatasetContractError(
                f"duplicate categorical bootstrap row for {cell!r}, "
                f"replicate {row.replicate}"
            )
        value = category(row)
        if value not in allowed:
            raise DatasetContractError(
                f"categorical bootstrap value {value!r} is not declared"
            )
        cells[cell][row.replicate] = value
    if not cells:
        raise DatasetContractError("categorical bootstrap has no rows")

    targets = tuple(sorted({key[0] for key in cells}))
    category_index = {value: index for index, value in enumerate(categories)}
    output = np.zeros((iterations, len(categories)), dtype=float)
    for target in targets:
        target_counts = np.zeros((iterations, len(categories)), dtype=float)
        denominator = 0
        for (cell_target, truth, condition), values in cells.items():
            if cell_target != target:
                continue
            ids = tuple(sorted(values))
            one_hot = np.zeros((len(ids), len(categories)), dtype=float)
            for index, replicate in enumerate(ids):
                one_hot[index, category_index[values[replicate]]] = 1.0
            weights = _replicate_weight_matrix(
                seed, iterations, target, truth, condition, ids
            )
            target_counts += weights @ one_hot
            denominator += len(ids)
        output += target_counts / denominator
    output /= len(targets)
    return {
        value: tuple(float(item) for item in output[:, category_index[value]])
        for value in categories
    }


def bootstrap_median_time_distribution(
    rows: Iterable[AnalysisRow],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    """Bootstrap a target-equal median with NC encoded as budget plus one."""

    cells: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        if not row.valid:
            continue
        cell = (row.target, row.truth, row.condition)
        if row.replicate in cells[cell]:
            raise DatasetContractError(
                f"duplicate median bootstrap row for {cell!r}, "
                f"replicate {row.replicate}"
            )
        cells[cell][row.replicate] = float(time_or_budget_plus_one(row))
    if not cells:
        raise DatasetContractError("median bootstrap has no rows")

    ordered_values = tuple(
        sorted({value for cell_values in cells.values() for value in cell_values.values()})
    )
    value_index = {value: index for index, value in enumerate(ordered_values)}
    targets = tuple(sorted({cell[0] for cell in cells}))
    probabilities = np.zeros((iterations, len(ordered_values)), dtype=float)
    for target in targets:
        target_counts = np.zeros((iterations, len(ordered_values)), dtype=float)
        target_total = 0
        for (cell_target, truth, condition), values in cells.items():
            if cell_target != target:
                continue
            ids = tuple(sorted(values))
            one_hot = np.zeros((len(ids), len(ordered_values)), dtype=float)
            for index, replicate in enumerate(ids):
                one_hot[index, value_index[values[replicate]]] = 1.0
            weights = _replicate_weight_matrix(
                seed, iterations, target, truth, condition, ids
            )
            target_counts += weights @ one_hot
            target_total += len(ids)
        probabilities += target_counts / target_total
    probabilities /= len(targets)
    cumulative = np.cumsum(probabilities, axis=1)
    lower_indices = np.argmax(cumulative >= (0.5 - 1e-12), axis=1)
    upper_indices = np.argmax(cumulative > (0.5 + 1e-12), axis=1)
    values_array = np.asarray(ordered_values, dtype=float)
    medians = (values_array[lower_indices] + values_array[upper_indices]) / 2.0
    return tuple(float(value) for value in medians)


def bootstrap_contrast_distribution(
    rows: Iterable[AnalysisRow],
    *,
    left_arm: str,
    right_arm: str,
    outcome: Outcome,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    indicator = outcome_indicator(outcome)
    cells: dict[tuple[str, str, str], dict[int, dict[str, AnalysisRow]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row.arm not in {left_arm, right_arm}:
            continue
        cell = (row.target, row.truth, row.condition)
        arms = cells[cell][row.replicate]
        if row.arm in arms:
            raise DatasetContractError(
                f"duplicate bootstrap row for {cell!r}, replicate {row.replicate}, "
                f"arm {row.arm}"
            )
        arms[row.arm] = row
    if not cells:
        raise DatasetContractError("bootstrap contrast has no rows")
    for cell, replicates in cells.items():
        for replicate, arms in replicates.items():
            if set(arms) != {left_arm, right_arm}:
                raise DatasetContractError(
                    f"unpaired bootstrap arms for {cell!r}, replicate {replicate}"
                )

    difference_cells: dict[tuple[str, str, str], dict[int, float]] = {}
    for cell, replicates in cells.items():
        valid_differences = {
            replicate: float(indicator(arms[left_arm]))
            - float(indicator(arms[right_arm]))
            for replicate, arms in replicates.items()
            if arms[left_arm].valid and arms[right_arm].valid
        }
        if valid_differences:
            difference_cells[cell] = valid_differences
    if not difference_cells:
        raise DatasetContractError("bootstrap contrast has no valid paired rows")
    return _bootstrap_cell_mean(
        difference_cells,
        iterations=iterations,
        seed=seed,
    )


def clear_bootstrap_weight_cache() -> None:
    """Release cached matrices after a complete analysis bundle is built."""

    _replicate_weight_matrix.cache_clear()


@lru_cache(maxsize=512)
def _replicate_weight_matrix(
    seed: int,
    iterations: int,
    target: str,
    truth: str,
    condition: str,
    replicate_ids: tuple[int, ...],
):
    if iterations <= 0:
        raise DatasetContractError("bootstrap iterations must be positive")
    ids = tuple(sorted(replicate_ids))
    if not ids or len(ids) != len(set(ids)):
        raise DatasetContractError("bootstrap replicate IDs must be unique and non-empty")
    if len(ids) > 255:
        raise DatasetContractError("bootstrap weight cache supports at most 255 replicates")
    index = {replicate: position for position, replicate in enumerate(ids)}
    matrix = np.zeros((iterations, len(ids)), dtype=np.uint8)
    for iteration in range(iterations):
        weights = replicate_weights(
            seed=seed,
            iteration=iteration,
            target=target,
            truth=truth,
            condition=condition,
            replicate_ids=ids,
        )
        for replicate, count in weights.items():
            matrix[iteration, index[replicate]] = count
    matrix.flags.writeable = False
    return matrix


def _bootstrap_cell_mean(
    cells: Mapping[tuple[str, str, str], Mapping[int, float]],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, ...]:
    targets = tuple(sorted({key[0] for key in cells}))
    output = np.zeros(iterations, dtype=float)
    for target in targets:
        numerator = np.zeros(iterations, dtype=float)
        denominator = 0
        for (cell_target, truth, condition), values in cells.items():
            if cell_target != target:
                continue
            ids = tuple(sorted(values))
            weights = _replicate_weight_matrix(
                seed, iterations, target, truth, condition, ids
            )
            numerator += weights @ np.asarray(
                [values[replicate] for replicate in ids], dtype=float
            )
            denominator += len(ids)
        output += numerator / denominator
    output /= len(targets)
    return tuple(float(value) for value in output)


def _single_arm_cells(
    rows: tuple[AnalysisRow, ...],
    *,
    indicator,
) -> dict[tuple[str, str, str], dict[int, float]]:
    cells: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        cell = (row.target, row.truth, row.condition)
        if row.replicate in cells[cell]:
            raise DatasetContractError(
                f"duplicate bootstrap row for {cell!r}, replicate {row.replicate}"
            )
        cells[cell][row.replicate] = float(indicator(row))
    if not cells:
        raise DatasetContractError("bootstrap rate has no rows")
    return cells
