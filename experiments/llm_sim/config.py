"""Frozen S2 definition and canonical configuration hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments" / "config" / "llm_sim_v1.json"
FROZEN_RUN_ID = "llm-personas-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class LLMSimConfig:
    raw: Mapping[str, Any]
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LLMSimConfig":
        raw = json.loads(json.dumps(value))
        _validate(raw)
        return cls(raw=raw, sha256=hashlib.sha256(canonical_json_bytes(raw)).hexdigest())

    @property
    def pair_count(self) -> int:
        return int(self.raw["pair_count"])

    @property
    def run_id(self) -> str:
        return str(self.raw["run_id"])

    @property
    def persona_count(self) -> int:
        return int(self.raw["persona_count"])

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["arms"])

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["providers"])

    @property
    def max_items(self) -> int:
        return int(self.raw["max_items"])

    @property
    def minimum_complete_per_cell(self) -> int:
        return int(self.raw["minimum_complete_per_cell"])

    @property
    def maximum_prompt_rewrites(self) -> int:
        return int(self.raw["maximum_prompt_rewrites"])

    @property
    def manipulation_mapping_policy(self) -> str:
        return str(self.raw["manipulation_mapping_policy"])


def load_frozen_config(path: str | Path = DEFAULT_CONFIG_PATH) -> LLMSimConfig:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("LLM simulation config must be an object")
    return LLMSimConfig.from_mapping(value)


def _validate(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "analysis_plan_commit",
        "persona_seed_derivation_version",
        "prompt_version",
        "study_seed",
        "pair_count",
        "persona_count",
        "arms",
        "providers",
        "max_items",
        "minimum_complete_per_cell",
        "maximum_prompt_rewrites",
        "accuracy_bands",
        "manipulation_mapping_policy",
        "provider_policy",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("LLM simulation config missing: " + ", ".join(missing))
    if int(value["pair_count"]) * 2 != int(value["persona_count"]):
        raise ValueError("persona_count must be exactly two per pair")
    if str(value["run_id"]) != FROZEN_RUN_ID:
        raise ValueError("S2 run_id differs from the frozen ingestion contract")
    if tuple(value["arms"]) != ("A", "B"):
        raise ValueError("S2 arms must be exactly A and B")
    expected_providers = ("deepseek", "glm", "kimi", "minimax", "doubao", "tongyi")
    if tuple(value["providers"]) != expected_providers:
        raise ValueError("S2 providers differ from the frozen six-provider grid")
    if not 0 < int(value["minimum_complete_per_cell"]) <= int(value["persona_count"]):
        raise ValueError("minimum_complete_per_cell is outside the persona count")
    if str(value["manipulation_mapping_policy"]) != "explicit_machine_annotation_only":
        raise ValueError("semantic target-option inference is forbidden")
    if not __import__("re").fullmatch(r"[0-9a-f]{40}", str(value["analysis_plan_commit"])):
        raise ValueError("analysis_plan_commit must be a full lowercase git SHA")
