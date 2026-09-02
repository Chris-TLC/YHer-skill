"""Adapters from item-bank fields to the validated selector contract."""
from __future__ import annotations

import math
from typing import Any, Mapping


DIFFICULTY_TIERS = {"T1": 0.25, "T2": 0.5, "T3": 0.75, "T4": 1.0}


def adapt_difficulty(value: Any) -> float:
    """Map T1-T4 or a normalized numeric value to difficulty in [0, 1]."""
    if isinstance(value, str):
        tier = DIFFICULTY_TIERS.get(value.strip().upper())
        if tier is not None:
            return tier
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"不支持的 difficulty: {value!r}") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"不支持的 difficulty: {value!r}") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"difficulty 必须在 [0, 1]: {value!r}")
    return numeric


def adapt_question_type(value: Any) -> str:
    """Map bank question_type values to mastery/selector's mcq|numeric enum."""
    normalized = str(value or "").strip().lower()
    if normalized in {"mcq", "numeric"}:
        return normalized
    if "选择" in normalized or "选题" in normalized:
        return "mcq"
    if any(token in normalized for token in ("填空", "计算", "简答", "综合")):
        return "numeric"
    raise ValueError(f"不支持的 question_type: {value!r}")


def adapt_selector_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return a selector-ready copy without mutating the item-bank record."""
    adapted = dict(item)
    adapted["difficulty"] = adapt_difficulty(item.get("difficulty"))
    adapted["item_type"] = adapt_question_type(
        item.get("item_type", item.get("question_type"))
    )
    return adapted
