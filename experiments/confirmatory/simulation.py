"""Production-engine observation loop for paired confirmatory journeys."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from engine import mastery, selector

from .allocation import FamilyEpoch, FixedLadderAllocator
from .metrics import is_severe_misdiagnosis as _is_severe
from .metrics import repetition_metrics
from .models import EmpiricalItem, TargetPools, UnitSpec
from .randomness import (
    SharedResponseStreams,
    arm_seed_material,
    replicate_seed_material,
)
from .response import generator_probability as _generator_probability
from .response import production_correct_probs as _production_correct_probs
from .response import sample_held_out as _sample_held_out
from .response import score_held_out


def production_model_id(runner_commit: str = "uncommitted") -> str:
    root = Path(__file__).resolve().parents[2]
    mastery_hash = hashlib.sha256((root / "engine" / "mastery.py").read_bytes()).hexdigest()
    selector_hash = hashlib.sha256((root / "engine" / "selector.py").read_bytes()).hexdigest()
    return (
        f"production-engine:mastery@{mastery_hash[:16]};"
        f"selector@{selector_hash[:16]};runner@{runner_commit}"
    )


def production_confidence_stop(
    node: mastery.NodeBelief,
    target_node: str,
    *,
    asked: int,
    stop_budget_items: int = 26,
) -> bool:
    belief = mastery.get_belief(node, float(asked))
    return selector.should_stop(
        {target_node: belief},
        [target_node],
        direct_answers={target_node: node.direct_answers},
        budget_items=stop_budget_items,
        asked=asked,
    )


def run_paired_unit(
    pools: TargetPools,
    spec: UnitSpec,
    config,
    *,
    model_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    _validate_unit(spec, pools, config)
    base_material = _base_material(spec, config)
    streams = SharedResponseStreams.build(base_material, config.max_items, config)
    held_out = _sample_held_out(pools, spec, config, streams)
    bound_model_id = model_id or production_model_id()
    return [
        run_journey(
            pools,
            spec,
            config,
            arm,
            held_out,
            streams=streams,
            model_id=bound_model_id,
            provenance=provenance,
        )
        for arm in config.arms
    ]


def run_journey(
    pools: TargetPools,
    spec: UnitSpec,
    config,
    arm: str,
    held_out_outcomes: Mapping[str, Any],
    *,
    streams: SharedResponseStreams | None = None,
    model_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_unit(spec, pools, config)
    if arm not in config.arms:
        raise ValueError(f"unknown arm: {arm}")
    base_material = _base_material(spec, config)
    shared = streams or SharedResponseStreams.build(base_material, config.max_items, config)
    held_out = dict(held_out_outcomes) or _sample_held_out(
        pools, spec, config, shared
    )
    bound_model_id = model_id or production_model_id()
    envelope = _envelope(spec, arm, config, bound_model_id, provenance)
    node = mastery.NodeBelief(mastery.UNIFORM.copy())
    snapshots: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    call_counters = {
        "mastery_observe": 0,
        "selector_select_next": 0,
        "selector_should_stop": 0,
    }
    terminal_reason: str | None = None
    convergence_time: int | None = None
    fixed_seen_families: set[str] = set()

    adaptive_items = pools.local_items + (
        pools.prerequisite_items if pools.h1_h2_eligible else ()
    )
    adaptive_epoch = (
        FamilyEpoch(
            adaptive_items,
            seed_material=arm_seed_material(base_material, "A", "family-epoch"),
        )
        if arm == "A" and adaptive_items
        else None
    )
    local_allocator = (
        FixedLadderAllocator(pools.local_items) if arm in {"B", "C"} else None
    )
    prereq_allocator = (
        FixedLadderAllocator(pools.prerequisite_items)
        if arm == "C" and pools.prerequisite_items
        else None
    )
    ladder = tuple(float(value) for value in config.raw["fixed_difficulty_ladder"])
    probe_interval = int(config.raw["c_probe_interval"])

    for position in range(1, config.max_items + 1):
        now = float(position)
        belief_before_selection = mastery.get_belief(node, now)
        eig: float | None = None
        selected: EmpiricalItem | None
        if arm == "A":
            assert adaptive_epoch is not None
            candidates = adaptive_epoch.candidates()
            candidate_rows = [item.selector_item(spec.target_node) for item in candidates]
            has_unseen = any(item.item_id not in seen_ids for item in candidates)
            selector_seen = seen_ids if has_unseen else set()
            call_counters["selector_select_next"] += 1
            selected_row = selector.select_next(
                candidate_rows,
                {spec.target_node: belief_before_selection},
                [spec.target_node],
                seen_ids=selector_seen,
                prereq_available=pools.h1_h2_eligible,
                asked_per_node={spec.target_node: node.direct_answers},
            )
            if selected_row is None:
                selected = None
            else:
                selected = next(
                    item for item in candidates if item.item_id == selected_row["item_id"]
                )
                eig = float(
                    selector.item_eig(
                        selected_row,
                        {spec.target_node: belief_before_selection},
                    )
                )
                adaptive_epoch.consume(selected)
        else:
            requested = ladder[(position - 1) % len(ladder)]
            if arm == "C" and position % probe_interval == 0:
                selected = (
                    prereq_allocator.take(
                        requested,
                        excluded_families=frozenset(fixed_seen_families),
                    )
                    if prereq_allocator
                    else None
                )
            else:
                assert local_allocator is not None
                selected = local_allocator.take(
                    requested,
                    excluded_families=frozenset(fixed_seen_families),
                )

        if selected is None:
            terminal_reason = "structural_failure"
            break

        correct_probs = _production_correct_probs(selected)
        generator_probability, generator_details = _generator_probability(
            selected,
            spec,
            shared,
            position,
            correct_probs,
            config,
        )
        correct = shared.response_noise[position - 1] < generator_probability
        likelihood = (
            mastery.likelihood_correct(correct_probs)
            if correct
            else mastery.likelihood_wrong_binary(correct_probs)
        )
        prior = mastery.get_belief(node, now)
        direct_before = node.direct_answers
        call_counters["mastery_observe"] += 1
        mastery.observe(
            node,
            likelihood,
            now,
            is_direct=selected.role == "local",
        )
        posterior = mastery.get_belief(node, now)
        direct_after = node.direct_answers
        seen_ids.add(selected.item_id)
        if arm in {"B", "C"}:
            fixed_seen_families.add(selected.family_id)
        call_counters["selector_should_stop"] += 1
        policy_stop = production_confidence_stop(
            node,
            spec.target_node,
            asked=position,
            stop_budget_items=config.max_items,
        )
        confidence_stop = policy_stop
        if position == config.max_items:
            call_counters["selector_should_stop"] += 1
            confidence_stop = production_confidence_stop(
                node,
                spec.target_node,
                asked=position,
                stop_budget_items=config.stop_budget_items,
            )
        event = {
            **envelope,
            "record_type": "confirmatory_event",
            "target_node": spec.target_node,
            "truth": spec.truth,
            "condition": spec.condition,
            "replicate": spec.replicate,
            "arm": arm,
            "position": position,
            "role": selected.role,
            "item_id": selected.item_id,
            "family_id": selected.family_id,
            "node_id": selected.node_id,
            "difficulty": selected.difficulty,
            "item_type": selected.item_type,
            "correct": bool(correct),
            "response_noise": shared.response_noise[position - 1],
            "generator_probability": generator_probability,
            "generator_parameters": generator_details,
            "production_correct_probabilities": correct_probs.tolist(),
            "production_inference_likelihood": likelihood.tolist(),
            "prior_belief": prior.tolist(),
            "posterior_belief": posterior.tolist(),
            "direct_answers_before": direct_before,
            "direct_answers_after": direct_after,
            "eig": eig,
            "production_should_stop": bool(policy_stop),
            "production_confidence_should_stop": bool(confidence_stop),
        }
        events.append(event)
        snapshots[position] = {
            "belief": posterior.tolist(),
            "direct_answers": direct_after,
        }
        if confidence_stop:
            terminal_reason = "confidence"
            convergence_time = position
            break
        if position == config.max_items:
            terminal_reason = "budget_exhausted"

    actual = len(events)
    if terminal_reason is None:
        terminal_reason = "structural_failure"
    final_belief = (
        np.asarray(snapshots[actual]["belief"], dtype=float)
        if actual
        else mastery.get_belief(node, 0.0)
    )
    converged = terminal_reason == "confidence"
    final_argmax = mastery.STATES[int(np.argmax(final_belief))]
    views = _build_views(
        config=config,
        pools=pools,
        spec=spec,
        snapshots=snapshots,
        actual=actual,
        terminal_reason=terminal_reason,
        convergence_time=convergence_time,
        final_belief=final_belief,
        held_out=held_out,
        events=events,
    )
    repeats = repetition_metrics(
        item_ids=tuple(event["item_id"] for event in events),
        family_ids=tuple(event["family_id"] for event in events),
    )
    severe = _is_severe(spec.truth, final_argmax)
    return {
        **envelope,
        "record_type": "confirmatory_journey",
        "target_node": spec.target_node,
        "truth": spec.truth,
        "condition": spec.condition,
        "replicate": spec.replicate,
        "arm": arm,
        "h1_h2_eligible": pools.h1_h2_eligible,
        "events": events,
        "views": views,
        "final_belief": final_belief.tolist(),
        "final_argmax": final_argmax,
        "converged": converged,
        "convergence_time": convergence_time,
        "terminal_reason": terminal_reason,
        "severe_misdiagnosis_all_terminal": severe,
        "severe_misdiagnosis_converged_only": severe if converged else None,
        "administered_roles": [event["role"] for event in events],
        "administered_item_ids": [event["item_id"] for event in events],
        "administered_family_ids": [event["family_id"] for event in events],
        "held_out_outcomes": {
            item_id: bool(details["outcome"]) for item_id, details in held_out.items()
        },
        "call_counters": call_counters,
        **repeats,
    }


def _build_views(
    *,
    config,
    pools: TargetPools,
    spec: UnitSpec,
    snapshots: Mapping[int, Mapping[str, Any]],
    actual: int,
    terminal_reason: str,
    convergence_time: int | None,
    final_belief: np.ndarray,
    held_out: Mapping[str, Mapping[str, Any]],
    events: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    views = []
    outcomes = {item_id: bool(row["outcome"]) for item_id, row in held_out.items()}
    for budget in config.budgets:
        if actual >= budget:
            view_actual = budget
            belief = np.asarray(snapshots[budget]["belief"], dtype=float)
        else:
            view_actual = actual
            belief = final_belief.copy()
        converged = convergence_time is not None and convergence_time <= budget
        argmax = mastery.STATES[int(np.argmax(belief))]
        held_out_score = score_held_out(belief, pools.held_out_items, outcomes)
        prefix_events = events[:view_actual]
        prefix_repeats = repetition_metrics(
            item_ids=tuple(str(event["item_id"]) for event in prefix_events),
            family_ids=tuple(str(event["family_id"]) for event in prefix_events),
        )
        views.append(
            {
                "nominal_budget": budget,
                "actual_administered_count": view_actual,
                "belief": belief.tolist(),
                "argmax": argmax,
                "converged": converged,
                "convergence_time": convergence_time if converged else None,
                "terminal_reason": (
                    "confidence"
                    if converged
                    else terminal_reason
                    if actual < budget or budget == config.max_items
                    else "checkpoint_nonterminal"
                ),
                "carried_forward": terminal_reason == "confidence" and actual < budget,
                "valid": terminal_reason != "structural_failure",
                "incomplete": terminal_reason == "structural_failure",
                "severe_misdiagnosis_all_terminal": _is_severe(spec.truth, argmax),
                "severe_misdiagnosis_converged_only": (
                    _is_severe(spec.truth, argmax) if converged else None
                ),
                "common_support_no_repeat": bool(
                    pools.common_support_no_repeat.get(budget, False)
                ),
                "common_support_set_sha256": pools.common_support_set_sha256.get(
                    budget, ""
                ),
                "held_out_outcomes": outcomes,
                "held_out_brier_interpretation": (
                    "internal_calibration_only"
                    if spec.condition == "matched"
                    else "misspecified_stress_condition"
                ),
                **held_out_score,
                **prefix_repeats,
            }
        )
    return views


def _base_material(spec: UnitSpec, config) -> str:
    return replicate_seed_material(
        master_seed=config.master_seed,
        target=spec.target_node,
        truth=spec.truth,
        condition=spec.condition,
        replicate=spec.replicate,
    )


def _envelope(
    spec: UnitSpec,
    arm: str,
    config,
    model_id: str,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    envelope = {
        "simulated": True,
        "persona_id": f"{spec.persona_prefix}:{arm}",
        "provider": str(config.raw["provider"]),
        "model_id": model_id,
    }
    if provenance is not None:
        envelope["provenance"] = dict(provenance)
    return envelope


def _validate_unit(spec: UnitSpec, pools: TargetPools, config) -> None:
    if spec.target_node != pools.target_node:
        raise ValueError("unit target does not match target pools")
    if spec.truth not in config.truth_states:
        raise ValueError(f"unknown truth state: {spec.truth}")
    if spec.condition not in config.conditions:
        raise ValueError(f"unknown generator condition: {spec.condition}")
    if spec.replicate < 0 or spec.replicate >= config.replicates:
        raise ValueError(f"replicate outside frozen range: {spec.replicate}")
