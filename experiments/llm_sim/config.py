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
FROZEN_H5_ANALYSIS_PLAN_COMMIT = "289be3bc4634336a8598ad80c0de084afdeba51d"
FROZEN_H5_ANALYSIS_PLAN_SHA256 = (
    "3ac258fe1d819cc857162588dead3d03e0ba414771269bf04f8ce9ec0ad99260"
)
FROZEN_H5_ANALYSIS_PLAN_COMMITTED_AT_UTC = "2026-07-13T18:59:52Z"


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
    def analysis_plan_commit(self) -> str:
        return str(self.raw["analysis_plan_commit"])

    @property
    def h5_analysis_plan_commit(self) -> str:
        return str(self.raw["h5_analysis_plan_commit"])

    @property
    def h5_analysis_plan_sha256(self) -> str:
        return str(self.raw["h5_analysis_plan_sha256"])

    @property
    def h5_analysis_plan_committed_at_utc(self) -> str:
        return str(self.raw["h5_analysis_plan_committed_at_utc"])

    @property
    def persona_seed_derivation_version(self) -> str:
        return str(self.raw["persona_seed_derivation_version"])

    @property
    def prompt_version(self) -> str:
        return str(self.raw["prompt_version"])

    @property
    def frozen_pre_observation_utc(self) -> str:
        return str(self.raw["frozen_pre_observation_utc"])

    @property
    def study_seed(self) -> int:
        return int(self.raw["study_seed"])

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
    def weak_accuracy_upper(self) -> float:
        return float(self.raw["accuracy_bands"]["weak_upper_exclusive"])

    @property
    def strong_accuracy_lower(self) -> float:
        return float(self.raw["accuracy_bands"]["strong_lower_exclusive"])

    @property
    def manipulation_bootstrap_seed(self) -> int:
        return int(self.raw["manipulation_bootstrap"]["seed"])

    @property
    def manipulation_bootstrap_resamples(self) -> int:
        return int(self.raw["manipulation_bootstrap"]["resamples"])

    @property
    def provider_policy(self) -> dict[str, int | float]:
        return {
            "max_attempts": int(self.raw["provider_policy"]["max_attempts"]),
            "failure_threshold": int(self.raw["provider_policy"]["failure_threshold"]),
            "base_backoff_seconds": float(
                self.raw["provider_policy"]["base_backoff_seconds"]
            ),
            "max_backoff_seconds": float(
                self.raw["provider_policy"]["max_backoff_seconds"]
            ),
            "cooldown_seconds": float(
                self.raw["provider_policy"]["cooldown_seconds"]
            ),
        }

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
        "h5_analysis_plan_commit",
        "h5_analysis_plan_sha256",
        "h5_analysis_plan_committed_at_utc",
        "persona_seed_derivation_version",
        "prompt_version",
        "frozen_pre_observation_utc",
        "study_seed",
        "pair_count",
        "persona_count",
        "arms",
        "providers",
        "max_items",
        "minimum_complete_per_cell",
        "maximum_prompt_rewrites",
        "accuracy_bands",
        "manipulation_bootstrap",
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
    if str(value["persona_seed_derivation_version"]) != "yher-llm-persona-v2":
        raise ValueError("persona_seed_derivation_version differs from the frozen amendment")
    if not str(value["prompt_version"]).strip():
        raise ValueError("prompt_version must be non-empty")
    if not 0 < int(value["minimum_complete_per_cell"]) <= int(value["persona_count"]):
        raise ValueError("minimum_complete_per_cell is outside the persona count")
    weak_upper = float(value["accuracy_bands"]["weak_upper_exclusive"])
    strong_lower = float(value["accuracy_bands"]["strong_lower_exclusive"])
    if not 0.0 <= weak_upper < strong_lower <= 1.0:
        raise ValueError("accuracy bands must be ordered within [0, 1]")
    bootstrap = value["manipulation_bootstrap"]
    if (
        int(bootstrap["resamples"]) != 10_000
        or int(bootstrap["seed"]) < 0
        or float(bootstrap["confidence_level"]) != 0.95
        or str(bootstrap["cluster_unit"]) != "persona_id"
    ):
        raise ValueError("manipulation bootstrap differs from the frozen contract")
    timestamp = str(value["frozen_pre_observation_utc"])
    try:
        parsed_timestamp = __import__("datetime").datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("frozen_pre_observation_utc must be ISO-8601") from exc
    if not timestamp.endswith("Z") or parsed_timestamp.tzinfo is None:
        raise ValueError("frozen_pre_observation_utc must be UTC")
    if str(value["manipulation_mapping_policy"]) != "explicit_machine_annotation_only":
        raise ValueError("semantic target-option inference is forbidden")
    if not __import__("re").fullmatch(r"[0-9a-f]{40}", str(value["analysis_plan_commit"])):
        raise ValueError("analysis_plan_commit must be a full lowercase git SHA")
    if str(value["h5_analysis_plan_commit"]) != FROZEN_H5_ANALYSIS_PLAN_COMMIT:
        raise ValueError("h5_analysis_plan_commit differs from the frozen amendment")
    if str(value["h5_analysis_plan_sha256"]) != FROZEN_H5_ANALYSIS_PLAN_SHA256:
        raise ValueError("h5_analysis_plan_sha256 differs from the frozen amendment")
    if (
        str(value["h5_analysis_plan_committed_at_utc"])
        != FROZEN_H5_ANALYSIS_PLAN_COMMITTED_AT_UTC
    ):
        raise ValueError(
            "h5_analysis_plan_committed_at_utc differs from the frozen amendment"
        )
    if timestamp != FROZEN_H5_ANALYSIS_PLAN_COMMITTED_AT_UTC:
        raise ValueError("frozen_pre_observation_utc must equal the amendment commit time")
