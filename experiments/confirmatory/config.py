"""Machine-readable configuration for the frozen confirmatory study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments" / "config" / "confirmatory_v1.json"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class ConfirmatoryConfig:
    raw: Mapping[str, Any]
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConfirmatoryConfig":
        raw = json.loads(json.dumps(value))
        _validate_config(raw)
        return cls(raw=raw, sha256=hashlib.sha256(canonical_json_bytes(raw)).hexdigest())

    @property
    def truth_states(self) -> tuple[str, ...]:
        return tuple(self.raw["truth_states"])

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(self.raw["arms"])

    @property
    def conditions(self) -> tuple[str, ...]:
        return tuple(self.raw["conditions"])

    @property
    def replicates(self) -> int:
        return int(self.raw["replicates"])

    @property
    def budgets(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["budgets"])

    @property
    def max_items(self) -> int:
        return int(self.raw["max_items"])

    @property
    def stop_budget_items(self) -> int:
        return int(self.raw["stop_budget_items"])

    @property
    def master_seed(self) -> int:
        return int(self.raw["master_seed"])

    def expected_journeys(self, *, open_node_count: int) -> int:
        return (
            int(open_node_count)
            * len(self.truth_states)
            * len(self.arms)
            * len(self.conditions)
            * self.replicates
        )


def load_frozen_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ConfirmatoryConfig:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("confirmatory config must be a JSON object")
    return ConfirmatoryConfig.from_mapping(value)


def _validate_config(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "seed_derivation_version",
        "master_seed",
        "truth_states",
        "arms",
        "conditions",
        "replicates",
        "budgets",
        "max_items",
        "stop_budget_items",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError("confirmatory config missing: " + ", ".join(missing))
    if int(value["max_items"]) != max(int(item) for item in value["budgets"]):
        raise ValueError("max_items must equal the largest nominal budget")
    if int(value["stop_budget_items"]) <= int(value["max_items"]):
        raise ValueError("stop_budget_items must discriminate confidence at max_items")
