"""Pure journey and checkpoint metrics."""

from __future__ import annotations

from typing import Any


def repetition_metrics(
    *, item_ids: tuple[str, ...], family_ids: tuple[str, ...]
) -> dict[str, Any]:
    actual = len(item_ids)
    unique_items = len(set(item_ids))
    unique_families = len(set(family_ids))
    item_repeats = actual - unique_items
    family_repeats = actual - unique_families
    return {
        "actual_administered_count": actual,
        "unique_item_count": unique_items,
        "unique_family_count": unique_families,
        "exact_item_repeat_count": item_repeats,
        "family_repeat_count": family_repeats,
        "exact_item_repeat_fraction": item_repeats / actual if actual else 0.0,
        "family_repeat_fraction": family_repeats / actual if actual else 0.0,
    }


def is_severe_misdiagnosis(truth: str, diagnosis: str) -> bool:
    return (truth, diagnosis) in {("M", "U"), ("U", "M")}
