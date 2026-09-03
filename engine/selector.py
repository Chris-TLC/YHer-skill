#!/usr/bin/env python3
"""
engine/selector.py — exact expected-information-gain item selector (design doc
block 1 "item selector", lines 107-116)
================================================================================
EIG(candidate j) = H(b_n(j)) − Σ_o P(o|b_n(j),j)·H(b_n(j) updated by o)
  where n(j) = the node whose belief is directly updated by answering the item:
    local-node item → k; prerequisite item → k (via the cross-node likelihood
    table); lateral item → the neighboring node itself.
The session goal = minimize the joint entropy over all target nodes (external voice #2).

Four states × dozens of candidates per node × ≤5 observations = exhaustive exact
computation, microsecond scale, no approximation.

Red lines / constraints:
  · holdout items must not enter selection (inherited from the parent doc).
  · prerequisite-item candidates are gated on the alignment table delivery (#4 prereq_available).
  · T0 hard constraint: prerequisite items trigger belief updates — emitted
    naturally by the EIG formula (when P/U compete, prerequisite items have the
    highest EIG; when C dominates, local-node items score higher), with no fixed quota.
  · At least 2 items per target node per session.
  · Beliefs are always read via mastery.get_belief (this module never reads the stored state .b directly).
Convergence terminates in should_stop: gap>0.45 and ≥3 direct answers (#9), or budget exhaustion.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from engine import mastery as m

TOP1_STOP = 0.80               # stop: posterior argmax ≥ 0.80 (no gap precedent in the literature; switched to the threshold family)
MIN_DIRECT_ANSWERS = 4         # stop: minimum direct evidence (min_length 4–5, red-team ruling)
MIN_ITEMS_PER_TARGET = 2       # minimum items per target node per session


def entropy(b: np.ndarray) -> float:
    """Shannon entropy (bits). Uniform over four states = 2 bits."""
    b = np.asarray(b, dtype=float)
    nz = b[b > 1e-12]
    return float(-(nz * np.log2(nz)).sum())


def _observation_likelihoods(d: float, item_type: str, role: str,
                             prereq_u_predict: float):
    """Return the list of likelihood vectors for this item's possible observations [(likelihood vector, ), ...].
    Binary degenerate tier (distractor_map entirely empty, a real stored-data condition): observations are correct/wrong only."""
    if role == "prereq":
        cp = m.prereq_correct_probs(prereq_u_predict, item_type)
    else:
        cp = m.local_correct_probs(d, item_type)
    return [m.likelihood_correct(cp), m.likelihood_wrong_binary(cp)]


def _expected_posterior_entropy(b: np.ndarray, likelihoods) -> float:
    """Σ_o P(o|b)·H(b updated by o). P(o|b) = Σ_s L_o(s)·b(s)."""
    b = np.asarray(b, dtype=float)
    total = 0.0
    for L in likelihoods:
        po = float((L * b).sum())               # marginal probability of this observation
        if po <= 1e-12:
            continue
        total += po * entropy(m.bayes_update(b, L))
    return total


def eig_local(b: np.ndarray, d: float, item_type: str = "mcq") -> float:
    """EIG of a local-node item with respect to its node's belief."""
    L = _observation_likelihoods(d, item_type, "local", m.PREREQ_U_DEFAULT)
    return entropy(b) - _expected_posterior_entropy(b, L)


def item_eig(item: Dict, beliefs: Dict[str, np.ndarray],
             prereq_u_predict: float = m.PREREQ_U_DEFAULT) -> float:
    """EIG of a candidate item, computed with respect to the node it updates (#2).
    item: {role, target_node, difficulty, item_type}.
    beliefs: {node: projected belief vector} (the caller has already projected via get_belief)."""
    role = item.get("role", "local")
    d = item.get("difficulty", 0.5)
    itype = item.get("item_type", "mcq")
    if role == "lateral":
        # Lateral items update the neighboring node itself → EIG comes from the neighbor's entropy drop (zero information about b_k)
        nb = item.get("target_node")
        b_nb = beliefs.get(nb)
        if b_nb is None:
            return 0.0
        L = _observation_likelihoods(d, itype, "local", prereq_u_predict)
        return entropy(b_nb) - _expected_posterior_entropy(b_nb, L)
    # local / prereq both update the target node k's belief, but via their own likelihood tables
    k = item.get("target_node")
    b_k = beliefs.get(k)
    if b_k is None:
        return 0.0
    L = _observation_likelihoods(d, itype, role, prereq_u_predict)
    return entropy(b_k) - _expected_posterior_entropy(b_k, L)


def _candidate_pool(candidates, seen_ids, prereq_available):
    """Filter: holdout items barred, already-attempted excluded, prerequisite items gated when the alignment table is absent (#4)."""
    seen_ids = seen_ids or set()
    pool = []
    for it in candidates:
        if it.get("holdout"):
            continue                              # holdout red line
        if it.get("item_id") in seen_ids:
            continue
        if it.get("role") == "prereq" and not prereq_available:
            continue                              # #4 gating
        pool.append(it)
    return pool


def select_next(candidates: Sequence[Dict], beliefs: Dict[str, np.ndarray],
                target_nodes: Sequence[str], *,
                seen_ids: Optional[set] = None,
                prereq_available: bool = False,
                asked_per_node: Optional[Dict[str, int]] = None,
                prereq_u_predict: float = m.PREREQ_U_DEFAULT) -> Optional[Dict]:
    """Pick the next item: maximize EIG under the coverage constraint (≥2 items per target node).
    Coverage outranks pure EIG: when a target node is under-tested
    (<MIN_ITEMS_PER_TARGET), first pick the highest-EIG candidate from the
    under-tested nodes."""
    pool = _candidate_pool(candidates, seen_ids, prereq_available)
    if not pool:
        return None
    asked_per_node = asked_per_node or {}

    # Coverage constraint: under-tested target nodes
    under = [n for n in target_nodes if asked_per_node.get(n, 0) < MIN_ITEMS_PER_TARGET]
    if under:
        under_set = set(under)
        under_pool = [it for it in pool if it.get("target_node") in under_set]
        if under_pool:
            pool = under_pool

    best, best_eig = None, -1.0
    for it in pool:
        e = item_eig(it, beliefs, prereq_u_predict)
        if e > best_eig:
            best, best_eig = it, e
    return best


def should_stop(beliefs: Dict[str, np.ndarray], target_nodes: Sequence[str], *,
                direct_answers: Dict[str, int], budget_items: int,
                asked: int) -> bool:
    """Stopping condition (2026-08-13 audit replacement):
      [P(top1)≥0.80 and ≥4 direct answers on this node] (satisfied by all target
      nodes)  or  budget exhaustion.
      The old gap>0.45 rule had no precedent in the literature and misjudged early
      stopping 10.3% of the time (red team 1); it has been retired. The gap value
      is kept by callers as a UI display quantity, no longer a stopping criterion."""
    if asked >= budget_items:
        return True                               # budget exhausted
    for n in target_nodes:
        b = beliefs.get(n)
        if b is None:
            return False
        top1 = float(np.max(np.asarray(b, dtype=float)))
        if top1 < TOP1_STOP:
            return False
        if direct_answers.get(n, 0) < MIN_DIRECT_ANSWERS:
            return False                          # insufficient evidence, do not converge
    return True
