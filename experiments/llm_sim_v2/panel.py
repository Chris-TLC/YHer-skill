"""Offline calibration panel selection and review-payload construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .public import public_question_payload


CALIBRATION_ITEMS_PER_ANCHOR = 4
BLIND_ITEMS_PER_ANCHOR_MAX = 25


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _items(catalog: Any, node: str) -> list[Any]:
    if hasattr(catalog, "for_node"):
        try:
            return list(catalog.for_node(node, deterministic_only=True))
        except TypeError:
            return list(catalog.for_node(node))
    raw = _value(catalog, "items", None)
    if raw is None and isinstance(catalog, Mapping):
        raw = catalog.values()
    if raw is None and isinstance(catalog, Iterable) and not isinstance(catalog, (str, bytes)):
        raw = catalog
    raw = raw or ()
    if isinstance(raw, Mapping):
        raw = raw.values()
    return [
        item
        for item in (raw or ())
        if node in tuple(_value(item, "node_ids", ()) or ())
    ]


def _candidate(item: Any) -> dict[str, Any] | None:
    item_id = str(_value(item, "item_id", "") or "").strip()
    family_id = str(_value(item, "family_id", "") or "").strip()
    scoring_mode = str(_value(item, "scoring_mode", "") or "").strip()
    public_question = public_question_payload(item)
    raw_options = public_question.get("options", {}) or {}
    if not isinstance(raw_options, Mapping):
        return None
    options = {str(key).strip().upper(): str(value) for key, value in raw_options.items()}
    answer_values = _value(item, "answer_values", ()) or ()
    correct = str(answer_values[0]).strip().upper() if answer_values else None
    if not item_id or not family_id or scoring_mode != "mcq" or not options or correct not in options:
        return None
    return {
        "item_id": item_id,
        "family_id": family_id,
        "public_question": public_question,
        "options": dict(sorted(options.items())),
        "correct_option": correct,
        "private_correct_option": correct,
    }


def is_calibration_candidate(item: Any) -> bool:
    return _candidate(item) is not None


def select_calibration_items(
    anchor: Mapping[str, Any] | Any,
    catalog: Any,
    *,
    count: int = CALIBRATION_ITEMS_PER_ANCHOR,
) -> list[dict[str, Any]]:
    """Select the first four valid items from distinct item families."""

    if count != CALIBRATION_ITEMS_PER_ANCHOR:
        raise ValueError("the v2 calibration panel requires exactly four items")
    node = str(_value(anchor, "target_node", "") or "").strip()
    if not node:
        raise ValueError("anchor target_node must be non-empty")
    candidates = [candidate for item in _items(catalog, node) if (candidate := _candidate(item))]
    candidates.sort(key=lambda value: (value["family_id"], value["item_id"]))
    selected: list[dict[str, Any]] = []
    families: set[str] = set()
    for candidate in candidates:
        if candidate["family_id"] in families:
            continue
        selected.append(candidate)
        families.add(candidate["family_id"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("anchor has fewer than four family-distinct valid MCQ calibration items")
    return selected


def select_blind_items(
    anchor: Mapping[str, Any] | Any,
    catalog: Any,
    *,
    maximum: int = BLIND_ITEMS_PER_ANCHOR_MAX,
) -> list[dict[str, Any]]:
    """Reuse the four calibration items and add up to 21 distinct families."""

    if maximum != BLIND_ITEMS_PER_ANCHOR_MAX:
        raise ValueError("the frozen v2 blind panel maximum is exactly 25 items")
    calibration = select_calibration_items(anchor, catalog)
    node = str(_value(anchor, "target_node", "") or "").strip()
    candidates = [candidate for item in _items(catalog, node) if (candidate := _candidate(item))]
    candidates.sort(key=lambda value: (value["family_id"], value["item_id"]))
    selected = list(calibration)
    families = {row["family_id"] for row in selected}
    item_ids = {row["item_id"] for row in selected}
    for candidate in candidates:
        if candidate["family_id"] in families or candidate["item_id"] in item_ids:
            continue
        selected.append(candidate)
        families.add(candidate["family_id"])
        item_ids.add(candidate["item_id"])
        if len(selected) == maximum:
            break
    return selected


def build_review_payload(anchor: Mapping[str, Any] | Any, catalog: Any) -> dict[str, Any]:
    """Return a local review payload; this function never writes it."""

    failure_id = str(_value(anchor, "failure_id", "") or "")
    failure_cause = str(_value(anchor, "failure_cause", "") or "")
    failure_symptom = str(_value(anchor, "failure_symptom", "") or "")
    items = []
    for candidate in select_calibration_items(anchor, catalog):
        items.append(
            {
                **candidate,
                "failure_id": failure_id,
                "failure_cause": failure_cause,
                "failure_symptom": failure_symptom,
            }
        )
    return {
        "anchor_id": str(_value(anchor, "anchor_id", "") or ""),
        "target_node": str(_value(anchor, "target_node", "") or ""),
        "failure_id": failure_id,
        "failure_cause": failure_cause,
        "failure_symptom": failure_symptom,
        "items": items,
    }


def export_review_payload(anchor: Mapping[str, Any] | Any, catalog: Any) -> dict[str, Any]:
    return build_review_payload(anchor, catalog)


build_calibration_panel = select_calibration_items


__all__ = [
    "BLIND_ITEMS_PER_ANCHOR_MAX",
    "CALIBRATION_ITEMS_PER_ANCHOR",
    "is_calibration_candidate",
    "select_blind_items",
    "select_calibration_items",
    "build_review_payload",
    "export_review_payload",
]
