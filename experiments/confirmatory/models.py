"""Small immutable value objects shared by confirmatory runner modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class EmpiricalItem:
    item_id: str
    family_id: str
    node_id: str
    difficulty: float
    item_type: str
    role: str

    def selector_item(self, target_node: str) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "family_id": self.family_id,
            "node": self.node_id,
            "target_node": target_node,
            "difficulty": self.difficulty,
            "item_type": self.item_type,
            "role": self.role,
            "holdout": False,
        }


@dataclass(frozen=True)
class TargetPools:
    target_node: str
    local_items: tuple[EmpiricalItem, ...]
    prerequisite_items: tuple[EmpiricalItem, ...]
    held_out_items: tuple[EmpiricalItem, ...]
    held_out_family_ids: frozenset[str]
    h1_h2_eligible: bool
    common_support_no_repeat: Mapping[int, bool]
    common_support_set_sha256: Mapping[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogContext:
    targets: Mapping[str, TargetPools]
    h1_h2_eligible_targets: tuple[str, ...]
    h1_h2_excluded_targets: tuple[str, ...]
    input_sha256: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class UnitSpec:
    target_node: str
    truth: str
    condition: str
    replicate: int

    @property
    def persona_prefix(self) -> str:
        return (
            f"confirmatory:{self.target_node}:{self.truth}:"
            f"{self.condition}:{self.replicate}"
        )
