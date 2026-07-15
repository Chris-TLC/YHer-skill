"""Immutable, outcome-free Persona v2 records."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .keys import canonical_key


_FORBIDDEN_PERSONA_KEYS = {
    "provider",
    "provider_id",
    "model",
    "model_id",
    "response",
    "responses",
    "outcome",
    "outcomes",
    "result",
    "accuracy",
    "observed_at",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _scan_forbidden(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if canonical_key(key) in _FORBIDDEN_PERSONA_KEYS:
                return str(key)
            found = _scan_forbidden(item)
            if found:
                return found
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            found = _scan_forbidden(item)
            if found:
                return found
    return None


@dataclass(frozen=True)
class PersonaV2:
    """One paired response row in the pre-observation v2 grid.

    ``persona_id`` identifies the independent cluster and is shared by the
    deficit/control rows.  ``row_id`` identifies the concrete repeated variant.
    Latent failure annotations remain separate from the observable policy and
    are never needed by a blind renderer.
    """

    persona_id: str
    pair_id: str
    row_id: str
    target_node: str
    curriculum_exposure: Any
    deficit_condition: str
    local_skill_vector: Mapping[str, Any]
    observable_error_policy: Mapping[str, Any]
    noise_parameters: Mapping[str, Any]
    modality_condition: str = "text_only"
    seed: int = 0
    ability_band: str | None = None
    anchor_id: str | None = None
    failure_id: str | None = None
    failure_cause: str | None = None
    failure_symptom: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("persona_id", "pair_id", "row_id", "target_node"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"persona {field_name} must be non-empty")
            object.__setattr__(self, field_name, value)
        condition = str(self.deficit_condition).strip().lower()
        if condition not in {"deficit", "control"}:
            raise ValueError("deficit_condition must be deficit or control")
        object.__setattr__(self, "deficit_condition", condition)
        modality = str(self.modality_condition).strip()
        if modality != "text_only":
            raise ValueError("modality_condition must be exactly text_only")
        object.__setattr__(self, "modality_condition", modality)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("persona seed must be an integer")
        forbidden = _scan_forbidden(
            {
                "curriculum_exposure": self.curriculum_exposure,
                "local_skill_vector": self.local_skill_vector,
                "observable_error_policy": self.observable_error_policy,
                "noise_parameters": self.noise_parameters,
            }
        )
        if forbidden:
            raise ValueError(f"persona schema cannot contain observed field {forbidden}")
        object.__setattr__(self, "curriculum_exposure", _freeze(self.curriculum_exposure))
        object.__setattr__(self, "local_skill_vector", _freeze(self.local_skill_vector))
        object.__setattr__(self, "observable_error_policy", _freeze(self.observable_error_policy))
        object.__setattr__(self, "noise_parameters", _freeze(self.noise_parameters))
        for field_name in (
            "ability_band",
            "anchor_id",
            "failure_id",
            "failure_cause",
            "failure_symptom",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, str(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PersonaV2":
        if not isinstance(value, Mapping):
            raise ValueError("persona row must be a mapping")
        forbidden = _scan_forbidden(value)
        if forbidden:
            raise ValueError(f"persona schema cannot contain observed field {forbidden}")
        persona_id = str(value.get("persona_id") or "").strip()
        condition = str(value.get("deficit_condition") or "").strip().lower()
        row_id = str(value.get("row_id") or f"{persona_id}:{condition}")
        return cls(
            persona_id=persona_id,
            pair_id=str(value.get("pair_id") or ""),
            row_id=row_id,
            target_node=str(value.get("target_node") or ""),
            curriculum_exposure=value.get("curriculum_exposure", ()),
            deficit_condition=condition,
            local_skill_vector=value.get("local_skill_vector", {}),
            observable_error_policy=value.get("observable_error_policy", {}),
            noise_parameters=value.get("noise_parameters", {}),
            modality_condition=str(value.get("modality_condition") or "text_only"),
            seed=value.get("seed", 0),
            ability_band=value.get("ability_band"),
            anchor_id=value.get("anchor_id"),
            failure_id=value.get("failure_id"),
            failure_cause=value.get("failure_cause"),
            failure_symptom=value.get("failure_symptom"),
        )

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "persona_id": self.persona_id,
            "pair_id": self.pair_id,
            "row_id": self.row_id,
            "target_node": self.target_node,
            "curriculum_exposure": _thaw(self.curriculum_exposure),
            "deficit_condition": self.deficit_condition,
            "local_skill_vector": _thaw(self.local_skill_vector),
            "observable_error_policy": _thaw(self.observable_error_policy),
            "noise_parameters": _thaw(self.noise_parameters),
            "modality_condition": self.modality_condition,
            "seed": self.seed,
        }
        for field_name in (
            "ability_band",
            "anchor_id",
            "failure_id",
            "failure_cause",
            "failure_symptom",
        ):
            value = getattr(self, field_name)
            if value is not None:
                output[field_name] = value
        return output


PersonaV2Row = PersonaV2


__all__ = ["PersonaV2", "PersonaV2Row"]
