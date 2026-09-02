"""Deterministic family traversal and fixed-arm allocation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from .models import EmpiricalItem
from .randomness import seed128


class FamilyEpoch:
    """Expose one stable representative per family before any family repeats."""

    def __init__(self, items: Iterable[EmpiricalItem], *, seed_material: str):
        grouped: dict[str, list[EmpiricalItem]] = defaultdict(list)
        for item in items:
            grouped[item.family_id].append(item)
        self._by_family = {
            family: tuple(sorted(rows, key=lambda row: row.item_id))
            for family, rows in grouped.items()
        }
        if not self._by_family:
            raise ValueError("family epoch requires at least one item")
        self._seed_material = seed_material
        self._epoch = -1
        self._remaining: list[str] = []
        self._start_epoch()

    def _start_epoch(self) -> None:
        self._epoch += 1
        self._remaining = sorted(self._by_family)
        random.Random(
            seed128(f"{self._seed_material}|epoch|{self._epoch}")
        ).shuffle(self._remaining)

    def candidates(self) -> tuple[EmpiricalItem, ...]:
        if not self._remaining:
            self._start_epoch()
        return tuple(self._by_family[family][0] for family in self._remaining)

    def consume(self, item: EmpiricalItem) -> None:
        try:
            self._remaining.remove(item.family_id)
        except ValueError as exc:
            raise ValueError("selected item is not in the active family epoch") from exc

    def take_first(self) -> EmpiricalItem:
        item = self.candidates()[0]
        self.consume(item)
        return item


class FixedLadderAllocator:
    """Choose by the frozen (distance, family_id, item_id) tuple."""

    def __init__(self, items: Iterable[EmpiricalItem]):
        self._items = tuple(items)
        if not self._items:
            raise ValueError("fixed allocator requires at least one item")
        self._families = frozenset(item.family_id for item in self._items)
        self._remaining = set(self._families)

    def take(
        self,
        requested_difficulty: float,
        *,
        excluded_families: frozenset[str] = frozenset(),
    ) -> EmpiricalItem:
        if not self._remaining:
            self._remaining = set(self._families)
        candidates = [
            item
            for item in self._items
            if item.family_id in self._remaining
            and item.family_id not in excluded_families
        ]
        if not candidates:
            candidates = [
                item for item in self._items if item.family_id in self._remaining
            ]
        selected = min(
            candidates,
            key=lambda item: (
                abs(item.difficulty - requested_difficulty),
                item.family_id,
                item.item_id,
            ),
        )
        self._remaining.remove(selected.family_id)
        return selected


def precompute_common_support(
    local_items: tuple[EmpiricalItem, ...],
    prerequisite_items: tuple[EmpiricalItem, ...],
    budgets: tuple[int, ...],
    *,
    probe_interval: int,
) -> dict[int, bool]:
    local_families = {item.family_id for item in local_items}
    output: dict[int, bool] = {}
    for budget in budgets:
        output[int(budget)] = (
            len(local_families) >= budget
            and _fixed_c_can_fill_without_repeat(
                local_items,
                prerequisite_items,
                budget,
                probe_interval=probe_interval,
            )
        )
    return output


def _fixed_c_can_fill_without_repeat(
    local_items: tuple[EmpiricalItem, ...],
    prerequisite_items: tuple[EmpiricalItem, ...],
    budget: int,
    *,
    probe_interval: int,
) -> bool:
    used: set[str] = set()
    ladder = (0.25, 0.5, 0.75, 1.0)
    for position in range(1, budget + 1):
        pool = prerequisite_items if position % probe_interval == 0 else local_items
        candidates = [item for item in pool if item.family_id not in used]
        if not candidates:
            return False
        requested = ladder[(position - 1) % len(ladder)]
        selected = min(
            candidates,
            key=lambda item: (
                abs(item.difficulty - requested),
                item.family_id,
                item.item_id,
            ),
        )
        used.add(selected.family_id)
    return True
