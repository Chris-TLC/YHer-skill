"""Capacity-explicit common-support set validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from .dataset import DatasetContractError


@dataclass(frozen=True)
class NoRepeatSet:
    budget: int
    targets: tuple[str, ...]
    n_target: int
    sha256: str


def validate_no_repeat_sets(
    journeys: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[int, NoRepeatSet]:
    """Rebuild and validate each budget's three-arm common-support target set."""

    flags: dict[int, dict[str, set[bool]]] = {}
    hashes: dict[int, set[str]] = {}
    for journey in journeys:
        target = str(journey.get("target_node"))
        events = journey.get("events")
        if not isinstance(events, list):
            raise DatasetContractError(f"events is not a list for target {target}")
        views = journey.get("views")
        if not isinstance(views, list):
            raise DatasetContractError(f"views is not a list for target {target}")
        for view in views:
            if not isinstance(view, Mapping):
                raise DatasetContractError(f"view is not an object for target {target}")
            budget = int(view.get("nominal_budget"))
            flag = view.get("common_support_no_repeat")
            set_hash = view.get("common_support_set_sha256")
            if not isinstance(flag, bool):
                raise DatasetContractError(
                    f"no-repeat flag is not boolean for {target} at {budget}"
                )
            if not isinstance(set_hash, str) or not set_hash:
                raise DatasetContractError(
                    f"no-repeat set hash is missing for {target} at {budget}"
                )
            if flag and not _event_prefix_has_no_repeats(events, budget):
                raise DatasetContractError(
                    f"declared no-repeat member {target} repeats by budget {budget}"
                )
            flags.setdefault(budget, {}).setdefault(target, set()).add(flag)
            hashes.setdefault(budget, set()).add(set_hash)

    output: dict[int, NoRepeatSet] = {}
    for budget in sorted(flags):
        for target, observed in flags[budget].items():
            if len(observed) != 1:
                raise DatasetContractError(
                    f"no-repeat flag drift for {target} at budget {budget}"
                )
        targets = tuple(
            sorted(
                target
                for target, observed in flags[budget].items()
                if next(iter(observed))
            )
        )
        expected_hash = _set_hash(budget, targets, manifest)
        if hashes[budget] != {expected_hash}:
            raise DatasetContractError(
                f"no-repeat set hash mismatch at budget {budget}: "
                f"{sorted(hashes[budget])!r} != {expected_hash}"
            )
        output[budget] = NoRepeatSet(
            budget=budget,
            targets=targets,
            n_target=len(targets),
            sha256=expected_hash,
        )
    return output


def _event_prefix_has_no_repeats(
    events: list[object], budget: int
) -> bool:
    items: list[str] = []
    families: list[str] = []
    for event in events[:budget]:
        if not isinstance(event, Mapping):
            raise DatasetContractError("no-repeat event is not an object")
        item_id = event.get("item_id")
        family_id = event.get("family_id")
        if not isinstance(item_id, str) or not item_id:
            raise DatasetContractError("no-repeat event lacks item_id")
        if not isinstance(family_id, str) or not family_id:
            raise DatasetContractError("no-repeat event lacks family_id")
        items.append(item_id)
        families.append(family_id)
    return len(items) == len(set(items)) and len(families) == len(set(families))


def _set_hash(
    budget: int,
    targets: tuple[str, ...],
    manifest: Mapping[str, Any],
) -> str:
    try:
        input_sha = manifest["input_sha256"]
        material = {
            "budget": budget,
            "targets": list(targets),
            "config_sha256": manifest["config_sha256"],
            "census_summary_sha256": input_sha["census_summary"]["sha256"],
            "census_records_sha256": input_sha["census_records"]["sha256"],
        }
    except (KeyError, TypeError) as exc:
        raise DatasetContractError("manifest lacks no-repeat hash inputs") from exc
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
