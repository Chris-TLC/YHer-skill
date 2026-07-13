"""SHA-256-derived deterministic random streams for paired journeys."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


def seed128(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:16], "big")


def replicate_seed_material(
    *,
    master_seed: int,
    target: str,
    truth: str,
    condition: str,
    replicate: int,
) -> str:
    return (
        f"yher-confirmatory-v1|{master_seed}|{target}|{truth}|"
        f"{condition}|{replicate}"
    )


def arm_seed_material(base: str, arm: str, purpose: str) -> str:
    return f"{base}|{arm}|{purpose}"


def held_out_seed_material(
    *,
    master_seed: int,
    target: str,
    truth: str,
    condition: str,
    replicate: int,
    family_id: str,
) -> str:
    return (
        f"yher-heldout-outcome-v1|{master_seed}|{target}|{truth}|"
        f"{condition}|{replicate}|{family_id}"
    )


@dataclass(frozen=True)
class SharedResponseStreams:
    response_noise: tuple[float, ...]
    slips: tuple[float, ...]
    guesses: tuple[float, ...]
    ability_offset: float

    @classmethod
    def build(cls, base_material: str, max_items: int, config) -> "SharedResponseStreams":
        response_rng = random.Random(seed128(f"{base_material}|response-noise"))
        slip_rng = random.Random(seed128(f"{base_material}|slip"))
        guess_rng = random.Random(seed128(f"{base_material}|guess"))
        ability_rng = random.Random(seed128(f"{base_material}|ability"))
        misspecified = config.raw["misspecified"]
        slip_low, slip_high = map(float, misspecified["slip_range"])
        guess_low, guess_high = map(float, misspecified["guess_range"])
        clip_low, clip_high = map(float, misspecified["ability_offset_clip"])
        ability = ability_rng.gauss(0.0, float(misspecified["ability_offset_sd"]))
        return cls(
            response_noise=tuple(response_rng.random() for _ in range(max_items)),
            slips=tuple(slip_rng.uniform(slip_low, slip_high) for _ in range(max_items)),
            guesses=tuple(guess_rng.uniform(guess_low, guess_high) for _ in range(max_items)),
            ability_offset=max(clip_low, min(clip_high, ability)),
        )
