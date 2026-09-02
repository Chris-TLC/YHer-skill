"""Integrity contracts for programmatic confirmatory journeys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .dataset import DatasetContractError


PROGRAMMATIC_KEY = ("target_node", "truth", "condition", "replicate", "arm")
PAIR_KEY = ("target_node", "truth", "condition", "replicate")
ARMS = frozenset({"A", "B", "C"})


@dataclass(frozen=True)
class GridValidation:
    journey_count: int
    primary_key_count: int
    pair_count: int
    target_count: int


def _value(row: Mapping[str, Any], key: str) -> Any:
    try:
        return row[key]
    except KeyError as exc:
        raise DatasetContractError(f"journey missing {key}") from exc


def validate_programmatic_grid(
    journeys: Iterable[Mapping[str, Any]],
    *,
    expected_journeys: int,
    expected_replicates: int,
    invalid_primary_keys: Iterable[tuple[object, ...]] = (),
) -> GridValidation:
    """Validate the frozen primary key and all within-replicate pairing contracts."""

    rows = tuple(journeys)
    invalid_keys = tuple(invalid_primary_keys)
    if len(rows) + len(invalid_keys) != expected_journeys:
        raise DatasetContractError(
            "journey intention count mismatch: "
            f"{len(rows)} valid + {len(invalid_keys)} invalid != {expected_journeys}"
        )

    by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    pairs: dict[tuple[Any, ...], dict[str, Mapping[str, Any] | None]] = {}
    replicate_sets: dict[tuple[str, str, str, str], set[int]] = {}

    for primary_key in invalid_keys:
        if len(primary_key) != len(PROGRAMMATIC_KEY) or primary_key in by_key:
            raise DatasetContractError(
                f"invalid quarantined programmatic primary key: {primary_key!r}"
            )
        target, truth, condition, replicate, arm = primary_key
        if not isinstance(replicate, int) or arm not in ARMS:
            raise DatasetContractError(
                f"invalid quarantined programmatic primary key: {primary_key!r}"
            )
        by_key[primary_key] = {}
        pairs.setdefault((target, truth, condition, replicate), {})[str(arm)] = None
        replicate_sets.setdefault(
            (str(target), str(truth), str(condition), str(arm)), set()
        ).add(replicate)

    for row in rows:
        primary_key = tuple(_value(row, field) for field in PROGRAMMATIC_KEY)
        if primary_key in by_key:
            raise DatasetContractError(
                f"duplicate programmatic primary key: {primary_key!r}"
            )
        by_key[primary_key] = row
        target, truth, condition, replicate, arm = primary_key
        if not isinstance(replicate, int):
            raise DatasetContractError(f"replicate is not an integer: {primary_key!r}")
        if arm not in ARMS:
            raise DatasetContractError(f"unexpected arm: {arm!r}")

        pair_key = (target, truth, condition, replicate)
        pairs.setdefault(pair_key, {})[str(arm)] = row
        replicate_sets.setdefault(
            (str(target), str(truth), str(condition), str(arm)), set()
        ).add(replicate)
        _validate_event_prefix(row, primary_key)

    for pair_key, arms in pairs.items():
        if set(arms) != ARMS:
            raise DatasetContractError(
                f"paired arms for {pair_key!r} are {sorted(arms)!r}, "
                f"expected {sorted(ARMS)!r}"
            )
        if all(row is not None for row in arms.values()):
            _validate_cross_arm_pair(
                pair_key,
                {arm: row for arm, row in arms.items() if row is not None},
            )

    expected_ids = set(range(expected_replicates))
    for cell, observed in replicate_sets.items():
        if observed != expected_ids:
            raise DatasetContractError(
                f"replicate IDs for {cell!r} are {sorted(observed)!r}, "
                f"expected {sorted(expected_ids)!r}"
            )

    return GridValidation(
        journey_count=len(by_key),
        primary_key_count=len(by_key),
        pair_count=len(pairs),
        target_count=len({str(key[0]) for key in by_key}),
    )


def _validate_event_prefix(
    row: Mapping[str, Any], primary_key: tuple[Any, ...]
) -> None:
    events = row.get("events")
    if not isinstance(events, list):
        raise DatasetContractError(f"events is not a list for {primary_key!r}")
    persona_id = row.get("persona_id")
    for expected_position, event in enumerate(events, start=1):
        if not isinstance(event, Mapping):
            raise DatasetContractError(f"event prefix is not an object for {primary_key!r}")
        event_key = tuple(event.get(field) for field in PROGRAMMATIC_KEY)
        if (
            event_key != primary_key
            or event.get("position") != expected_position
            or event.get("persona_id") != persona_id
        ):
            raise DatasetContractError(
                f"event prefix mismatch for {primary_key!r} at position "
                f"{expected_position}"
            )


def _validate_cross_arm_pair(
    pair_key: tuple[Any, ...], arms: Mapping[str, Mapping[str, Any]]
) -> None:
    reference = arms["A"]
    reference_noise = tuple(event.get("response_noise") for event in reference["events"])
    reference_outcomes = reference.get("held_out_outcomes")
    reference_pairs = reference.get("held_out_pairs")
    for arm in ("B", "C"):
        row = arms[arm]
        noises = tuple(event.get("response_noise") for event in row["events"])
        prefix_length = min(len(reference_noise), len(noises))
        if noises[:prefix_length] != reference_noise[:prefix_length]:
            raise DatasetContractError(
                f"response-noise prefix mismatch for {pair_key!r}, arm {arm}"
            )
        if row.get("held_out_outcomes") != reference_outcomes:
            raise DatasetContractError(
                f"held-out outcomes mismatch for {pair_key!r}, arm {arm}"
            )
        if row.get("held_out_pairs") != reference_pairs:
            raise DatasetContractError(
                f"held-out family/item pairs mismatch for {pair_key!r}, arm {arm}"
            )
