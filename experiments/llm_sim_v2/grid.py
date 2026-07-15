"""Deterministic 25-anchor x 2-noise Persona v2 grid construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .models import PersonaV2


ANCHOR_COUNT = 25
NOISE_LEVELS = ("low", "high")


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _seed_for(seed: int, index: int, anchor_id: str, noise: str) -> int:
    material = f"persona-v2|{seed}|{index}|{anchor_id}|{noise}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _anchor_rows(anchors: Sequence[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for anchor in anchors:
        if not isinstance(anchor, Mapping) and not hasattr(anchor, "__dict__"):
            raise ValueError("each anchor must be a mapping or record")
        forbidden = {"provider", "outcome", "response", "model_id"}
        if isinstance(anchor, Mapping) and forbidden.intersection(str(key).lower() for key in anchor):
            raise ValueError("persona anchors cannot contain provider or outcome data")
        target_node = str(_value(anchor, "target_node", "")).strip()
        failure_id = str(_value(anchor, "failure_id", "")).strip()
        anchor_id = str(_value(anchor, "anchor_id", "") or f"{target_node}:{failure_id}").strip()
        if not anchor_id or not target_node or not failure_id:
            raise ValueError("each anchor requires anchor_id, target_node, and failure_id")
        normalized.append(
            {
                "anchor_id": anchor_id,
                "target_node": target_node,
                "failure_id": failure_id,
                "failure_cause": str(_value(anchor, "failure_cause", "") or "").strip(),
                "failure_symptom": str(_value(anchor, "failure_symptom", "") or "").strip(),
                "curriculum_exposure": _value(anchor, "curriculum_exposure", (target_node,)),
            }
        )
        if not normalized[-1]["failure_cause"] or not normalized[-1]["failure_symptom"]:
            raise ValueError("each anchor requires non-empty failure cause and symptom")
    normalized.sort(key=lambda row: (row["anchor_id"], row["target_node"], row["failure_id"]))
    return normalized


def build_persona_grid(
    anchors: Sequence[Any],
    *,
    seed: int = 20260715,
    noise_levels: Sequence[str] = NOISE_LEVELS,
) -> list[PersonaV2]:
    """Build exactly 50 independent clusters and their 100 paired rows."""

    normalized = _anchor_rows(anchors)
    if len(normalized) != ANCHOR_COUNT:
        raise ValueError(f"exactly {ANCHOR_COUNT} deterministic anchors are required")
    anchor_ids = [row["anchor_id"] for row in normalized]
    if len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("persona anchors must have unique anchor_id values")
    levels = tuple(str(level).strip().lower() for level in noise_levels)
    if levels != NOISE_LEVELS:
        raise ValueError("noise_levels must be exactly ('low', 'high')")

    rows: list[PersonaV2] = []
    for anchor_index, anchor in enumerate(normalized):
        for noise_index, noise_level in enumerate(levels):
            cluster_index = anchor_index * len(levels) + noise_index
            persona_id = f"persona-v2:{anchor['anchor_id']}:{noise_level}"
            pair_id = f"pair-v2:{cluster_index:02d}"
            ability_band = "lower" if anchor_index % 2 == 0 else "higher"
            row_seed = _seed_for(seed, cluster_index, anchor["anchor_id"], noise_level)
            noise = {
                "level": noise_level,
                "hesitation_rate": 0.15 if noise_level == "low" else 0.45,
                "slip_rate": 0.05 if noise_level == "low" else 0.20,
            }
            base_skill = 0.35 if ability_band == "lower" else 0.75
            for condition in ("deficit", "control"):
                adjustment = -0.12 if condition == "deficit" else 0.0
                skill = {
                    "ability_band": ability_band,
                    "prerequisite_skill": round(max(0.0, base_skill + adjustment), 3),
                    "target_skill": round(max(0.0, base_skill + adjustment), 3),
                    "reasoning_skill": round(max(0.0, base_skill + adjustment), 3),
                }
                if condition == "deficit":
                    policy = {
                        "strategy": "apply_observed_failure_pattern",
                        "observable_behavior": "select the response pattern matching the predeclared cause and symptom",
                        "cause": anchor["failure_cause"],
                        "symptom": anchor["failure_symptom"],
                    }
                else:
                    policy = {"strategy": "solve_normally"}
                rows.append(
                    PersonaV2(
                        persona_id=persona_id,
                        pair_id=pair_id,
                        row_id=f"{persona_id}:{condition}",
                        target_node=anchor["target_node"],
                        curriculum_exposure=anchor["curriculum_exposure"],
                        deficit_condition=condition,
                        local_skill_vector=skill,
                        observable_error_policy=policy,
                        noise_parameters=noise,
                        modality_condition="text_only",
                        seed=row_seed,
                        ability_band=ability_band,
                        anchor_id=anchor["anchor_id"],
                        failure_id=anchor["failure_id"],
                        failure_cause=anchor["failure_cause"],
                        failure_symptom=anchor["failure_symptom"],
                    )
                )
    return rows


def build_persona_rows(*args: Any, **kwargs: Any) -> list[PersonaV2]:
    return build_persona_grid(*args, **kwargs)


def serialize_grid(rows: Sequence[PersonaV2 | Mapping[str, Any]]) -> bytes:
    payload = []
    for row in rows:
        normalized = row if isinstance(row, PersonaV2) else PersonaV2.from_mapping(row)
        payload.append(normalized.to_dict())
    payload.sort(key=lambda value: str(value.get("row_id", "")))
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def grid_sha256(rows: Sequence[PersonaV2 | Mapping[str, Any]]) -> str:
    return hashlib.sha256(serialize_grid(rows)).hexdigest()


__all__ = [
    "ANCHOR_COUNT",
    "NOISE_LEVELS",
    "build_persona_grid",
    "build_persona_rows",
    "serialize_grid",
    "grid_sha256",
]
