"""Matched/misspecified response generation and held-out scoring."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np

from engine import mastery

from .models import EmpiricalItem, TargetPools, UnitSpec
from .randomness import SharedResponseStreams, held_out_seed_material, seed128


TRUTH_INDEX = {state: index for index, state in enumerate(mastery.STATES)}


def sample_held_out(
    pools: TargetPools,
    spec: UnitSpec,
    config,
    streams: SharedResponseStreams,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in pools.held_out_items:
        material = held_out_seed_material(
            master_seed=config.master_seed,
            target=spec.target_node,
            truth=spec.truth,
            condition=spec.condition,
            replicate=spec.replicate,
            family_id=item.family_id,
        )
        rng = random.Random(seed128(material))
        correct_probs = production_correct_probs(item)
        if spec.condition == "matched":
            probability = float(correct_probs[TRUTH_INDEX[spec.truth]])
            slip = guess = None
        else:
            misspecified = config.raw["misspecified"]
            slip = rng.uniform(*map(float, misspecified["slip_range"]))
            guess = rng.uniform(*map(float, misspecified["guess_range"]))
            probability = misspecified_probability(
                item,
                TRUTH_INDEX[spec.truth],
                slip,
                guess,
                streams.ability_offset,
                config,
            )
        outcome = rng.random() < probability
        output[item.item_id] = {
            "family_id": item.family_id,
            "outcome": bool(outcome),
            "generator_probability": probability,
            "slip": slip,
            "guess": guess,
            "ability_offset": streams.ability_offset,
            "seed_material": material,
        }
    return output


def generator_probability(
    item: EmpiricalItem,
    spec: UnitSpec,
    streams: SharedResponseStreams,
    position: int,
    production_correct_probabilities: np.ndarray,
    config,
) -> tuple[float, dict[str, float | None]]:
    if spec.condition == "matched":
        return float(production_correct_probabilities[TRUTH_INDEX[spec.truth]]), {
            "slip": None,
            "guess": None,
            "ability_offset": None,
        }
    slip = streams.slips[position - 1]
    guess = streams.guesses[position - 1]
    probability = misspecified_probability(
        item,
        TRUTH_INDEX[spec.truth],
        slip,
        guess,
        streams.ability_offset,
        config,
    )
    return probability, {
        "slip": slip,
        "guess": guess,
        "ability_offset": streams.ability_offset,
    }


def misspecified_probability(
    item: EmpiricalItem,
    truth_index: int,
    slip: float,
    guess: float,
    ability_offset: float,
    config,
) -> float:
    if item.role == "prereq":
        values = np.array(
            [1.0 - slip, guess, mastery.PREREQ_CORRECT_C, mastery.PREREQ_U_DEFAULT]
        )
    else:
        values = np.array(
            [1.0 - slip, guess + mastery.DELTA_P, 0.7 - 0.2 * item.difficulty, guess]
        )
    low, high = map(float, config.raw["misspecified"]["probability_clip"])
    return float(np.clip(values[truth_index] + ability_offset, low, high))


def production_correct_probs(item: EmpiricalItem) -> np.ndarray:
    if item.role == "prereq":
        return mastery.prereq_correct_probs(item_type=item.item_type)
    return mastery.local_correct_probs(item.difficulty, item.item_type)


def score_held_out(
    posterior: np.ndarray,
    held_out_items: tuple[EmpiricalItem, ...],
    outcomes: Mapping[str, bool],
) -> dict[str, Any]:
    belief = np.asarray(posterior, dtype=float)
    records = []
    for item in held_out_items:
        if item.item_id not in outcomes:
            continue
        correct_probs = mastery.local_correct_probs(item.difficulty, item.item_type)
        prediction = float(np.dot(belief, correct_probs))
        outcome = bool(outcomes[item.item_id])
        squared_error = (prediction - float(outcome)) ** 2
        records.append(
            {
                "family_id": item.family_id,
                "item_id": item.item_id,
                "p_hat": prediction,
                "outcome": outcome,
                "squared_error": squared_error,
            }
        )
    brier = float(np.mean([row["squared_error"] for row in records])) if records else None
    return {"held_out_family_scores": records, "held_out_brier": brier}
