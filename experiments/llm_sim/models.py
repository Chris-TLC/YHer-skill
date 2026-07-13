"""Small immutable contracts for the LLM-persona study."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Persona:
    """One simulated student; weak/strong rows share a pair annotation."""

    persona_id: str
    pair_id: str
    strength: str
    target_node: str
    failure_id: str
    failure_cause: str
    failure_symptom: str
    diagnostic_question: str
    annotation_source: str = "kg.common_failures"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.strength not in {"weak", "strong"}:
            raise ValueError("persona strength must be weak or strong")
        for field_name in (
            "persona_id",
            "pair_id",
            "target_node",
            "failure_id",
            "annotation_source",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"persona {field_name} must be non-empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Persona":
        return cls(
            persona_id=str(value["persona_id"]),
            pair_id=str(value["pair_id"]),
            strength=str(value["strength"]),
            target_node=str(value["target_node"]),
            failure_id=str(value.get("failure_id") or ""),
            failure_cause=str(value.get("failure_cause") or ""),
            failure_symptom=str(value.get("failure_symptom") or ""),
            diagnostic_question=str(value.get("diagnostic_question") or ""),
            annotation_source=str(value.get("annotation_source") or "kg.common_failures"),
            seed=int(value.get("seed") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "pair_id": self.pair_id,
            "strength": self.strength,
            "target_node": self.target_node,
            "failure_id": self.failure_id,
            "failure_cause": self.failure_cause,
            "failure_symptom": self.failure_symptom,
            "diagnostic_question": self.diagnostic_question,
            "annotation_source": self.annotation_source,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ProviderSpec:
    """Official endpoint and environment binding for one provider."""

    name: str
    base_url: str
    model_default: str
    key_env: str
    pricing: Mapping[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyPanel:
    """An immutable, pre-outcome manipulation annotation panel."""

    panel_sha256: str
    study_seed: int
    personas_sha256: str
    annotations: tuple[Mapping[str, Any], ...]
    frozen: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "yher.llm_sim.manipulation_panel.v1",
            "frozen": self.frozen,
            "observation_started": False,
            "study_seed": self.study_seed,
            "personas_sha256": self.personas_sha256,
            "annotations": [dict(row) for row in self.annotations],
            "panel_sha256": self.panel_sha256,
        }
