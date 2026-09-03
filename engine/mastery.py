#!/usr/bin/env python3
"""
engine/mastery.py — cause-of-error belief engine (diagnosis-engine design doc
block 1, 2026-07-08)
===================================================================
Bayesian updates over a four-state belief {M mastered / P prerequisite gap /
C unstable reasoning chain / U completely lost}, per-item-type slip/guess,
two-stage distractor decomposition, hierarchical priors, and FSRS one-way
decay projection.

Spec source: mac-unknown-design-20260707-230853.md (line numbers noted in comments).
Scope: this module is the belief kernel. The EIG selector lives in selector.py
(stage 1, pending a T0 decision on the gap threshold).

Red lines (design doc line 152):
  · The decay formula lives only in _project_decay; get_belief is the sole read
    entry — reading node.b directly, bypassing it, is forbidden.
  · The one-way projection only redistributes M→C and never touches P/U.
  · The stored state b never contains decay; the update prior = the projected b;
    forgetting for a given time span happens exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── State indices (fixed order for the whole module) ────────────────────
M, P, C, U = 0, 1, 2, 3
STATES = ("M", "P", "C", "U")
UNIFORM = np.full(4, 0.25)

# ── Observation parameters (design doc lines 86-87, module constants + comments; upgrade trigger at line 86) ──
EPS = 0.10                        # slip / carelessness, shared across item types
GAMMA = {"mcq": 0.25, "numeric": 0.03}   # guess: four-choice MCQ / plain numeric short-answer
DELTA_P = 0.10                    # δ_p, missing prerequisite but may half-know this step

# ── Cross-node likelihoods (line 88) ──
PREREQ_CORRECT_C = 0.80           # P(correct on prerequisite|C): those with unstable reasoning usually still have the prerequisite
PREREQ_U_DEFAULT = 0.75           # P(correct on prerequisite|U): default when no prerequisite belief data exists (frozen at session snapshot)

# ── Hierarchical priors (lines 119-122) ──
LAMBDA_PRIOR = 0.6                # b_k(0) = λ·block posterior + (1−λ)·global prior
ETA_PREREQ = 0.4                  # prerequisite propagation strength

# ── FSRS decay + stability S (lines 49/150-151; 2026-08-13 audit: old hand-set ×3/×2/×0.5 had no damping → S ballooned to 4608 days; replaced with the 4.5 formula) ──
S0 = 3.0                          # initial S (days)
S_FIRST_REVIEW_MULT = 3.0         # first review multiplies S by 3 (consolidated knowledge stabilizes quickly; same as 4.5's w0)
S_MAX = 500.0                     # damping cap (days), aligned with FSRS's official default max stability
# FSRS-4.5 stability growth weights (w7-w12, taken from the official public parameter table; the simplified version keeps only the 5 relevant to this project)
_W7 = 0.9                         # success-factor base
_W8 = 0.0                         # success-factor exponent
_W10 = 10.0                       # S-exponent denominator
_W11 = 0.72                       # failure multiplier
_W12 = 0.90                       # success verdict threshold
DAY_SECONDS = 86400

# ── LLM slow-path clamping (line 103, #8) ──
LIKELIHOOD_RATIO_CAP = 3.0        # hard likelihood-ratio cap: one misread Chinese sentence is at most worth one item's worth of evidence


# ══════════════════════════════════════════════════════════════════════
# Belief stored state
# ══════════════════════════════════════════════════════════════════════
@dataclass
class NodeBelief:
    """Stored belief state for a (student, node) pair. b never contains decay; decay is projected lazily in get_belief."""
    b: np.ndarray                              # [P(M),P(P),P(C),P(U)], stored state
    S: Optional[float] = None                  # stability (days), None = not initialized
    last_review_at: Optional[float] = None     # timestamp of the last event (seconds)
    direct_answers: int = 0                    # direct answer count on this node (#9 convergence gate)
    subject: str = "chemistry"                 # schema increment list, item 7

    def __post_init__(self) -> None:
        try:
            belief = np.asarray(self.b, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("belief must be a 4-dimensional numeric probability") from exc
        if belief.shape != (4,):
            raise ValueError(f"belief must be 4-dimensional, actual shape={belief.shape}")
        if not np.all(np.isfinite(belief)) or np.any(belief < 0):
            raise ValueError("belief must be finite and non-negative")
        if not np.isclose(belief.sum(), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("belief probabilities must sum to approximately 1")
        if self.S is not None and (not np.isfinite(self.S) or self.S <= 0):
            raise ValueError("S must be a finite positive number or None")
        if self.direct_answers < 0:
            raise ValueError("direct_answers must not be negative")
        self.b = belief


# ══════════════════════════════════════════════════════════════════════
# Likelihood tables (lines 87-91)
# ══════════════════════════════════════════════════════════════════════
def local_correct_probs(d: float, item_type: str = "mcq") -> np.ndarray:
    """Per-state probability of answering a local-node item correctly (line 87). d∈[0,1] is the normalized difficulty."""
    g = GAMMA[item_type]
    return np.array([1 - EPS, g + DELTA_P, 0.7 - 0.2 * d, g])


def prereq_correct_probs(prereq_u_predict: float = PREREQ_U_DEFAULT,
                         item_type: str = "mcq") -> np.ndarray:
    """Per-state probability of answering a prerequisite-node item correctly (line 88).
       prereq_u_predict = the prerequisite node's own predicted correct rate, frozen at the session-start snapshot (#12)."""
    g = GAMMA[item_type]
    return np.array([1 - EPS, g, PREREQ_CORRECT_C, prereq_u_predict])


def likelihood_correct(correct_probs: np.ndarray) -> np.ndarray:
    """Observation = correct; likelihood vector = P(correct|s)."""
    return np.asarray(correct_probs, dtype=float)


def likelihood_wrong_binary(correct_probs: np.ndarray) -> np.ndarray:
    """Observation = wrong, no distractor_map (binary degenerate case, line 91): likelihood = 1−P(correct|s)."""
    return 1.0 - np.asarray(correct_probs, dtype=float)


def default_distractor_error() -> np.ndarray:
    """P(d|wrong,s) template (lines 89-90): rows = states [M,P,C,U], columns = [typical misconception, other, other].
       Pinned: C choosing the typical misconception = 0.60, U/M uniform; the P row is a template value
       (overridden per item once the error-cause code tables ship at scale).
       Each row is normalized (Σ_d P(d|wrong,s)=1).
    """
    return np.array([
        [1 / 3, 1 / 3, 1 / 3],     # M: random slip
        [0.45, 0.275, 0.275],      # P: biased toward the typical misconception
        [0.60, 0.20, 0.20],        # C: typical misconception 0.60
        [1 / 3, 1 / 3, 1 / 3],     # U: uniform
    ])


def likelihood_wrong_distractor(correct_probs: np.ndarray, derr: np.ndarray,
                                chosen_idx: int) -> np.ndarray:
    """Observation = wrong with distractor chosen_idx picked; two-stage decomposition (line 89):
       P(pick d|s) = P(wrong|s)·P(d|wrong,s) = (1−P(correct|s))·derr[s, chosen_idx]."""
    return (1.0 - np.asarray(correct_probs, dtype=float)) * derr[:, chosen_idx]


# ══════════════════════════════════════════════════════════════════════
# Bayesian update (line 95)
# ══════════════════════════════════════════════════════════════════════
def bayes_update(b: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """One-line Bayes: b'(s) ∝ P(observation|s)·b(s), then normalized."""
    post = np.asarray(b, dtype=float) * np.asarray(likelihood, dtype=float)
    total = post.sum()
    if total <= 0:                    # degenerate protection: all-zero likelihood → belief unchanged
        return np.asarray(b, dtype=float).copy()
    return post / total


# ══════════════════════════════════════════════════════════════════════
# LLM slow-path likelihood clamping (line 103, #8; failure mode: hallucinated illegal values)
# ══════════════════════════════════════════════════════════════════════
def _cap_likelihood_ratio(L: np.ndarray, cap: float) -> np.ndarray:
    """Squeeze the likelihood ratio max/min down to ≤ cap (linear contraction in log space, order-preserving)."""
    L = L / L.sum()
    L = np.clip(L, 1e-12, None)
    if L.max() / L.min() <= cap:
        return L / L.sum()
    logL = np.log(L)
    logL -= logL.mean()
    span = logL.max() - logL.min()
    logL *= np.log(cap) / span        # compress the log span down to log(cap)
    Lc = np.exp(logL)
    return Lc / Lc.sum()


def sanitize_llm_likelihood(L_raw, confidence: float):
    """Sanitize the likelihood vector from the follow-up-question judge (#8, line 103).
    Returns (L_clean, was_illegal). The raw L_raw is logged verbatim by the caller (silent-failure path ①).
      · Illegal (shape/NaN/negative/non-positive sum) → uniform distribution = zero information.
      · Power flattening: L' = L^confidence (conf=0 → uniform; conf=1 → unchanged).
      · Hard likelihood-ratio cap at 3:1."""
    L = np.asarray(L_raw, dtype=float) if _is_numeric_seq(L_raw) else None
    if (L is None or L.shape != (4,) or not np.all(np.isfinite(L))
            or np.any(L < 0) or L.sum() <= 0):
        return UNIFORM.copy(), True
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return UNIFORM.copy(), True
    if not np.isfinite(conf):
        return UNIFORM.copy(), True
    L = L / L.sum()
    conf = float(np.clip(conf, 0.0, 1.0))
    L = L ** conf                     # power flattening
    L = L / L.sum()
    L = _cap_likelihood_ratio(L, LIKELIHOOD_RATIO_CAP)
    return L, False


def _is_numeric_seq(x) -> bool:
    try:
        arr = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        return False
    return arr.ndim == 1


# ══════════════════════════════════════════════════════════════════════
# Hierarchical priors (lines 119-122)
# ══════════════════════════════════════════════════════════════════════
def hierarchical_prior(block_beliefs, block_counts, global_prior: np.ndarray,
                       prereq_belief: Optional[np.ndarray] = None,
                       prereq_available: bool = False) -> np.ndarray:
    """b_k(0) = λ·block-level posterior + (1−λ)·global prior (line 119).
       The block-level posterior = the weighted average of the block's node beliefs
       b by cumulative answer counts (line 121);
       with no answers in the block → the λ term falls back to the global prior.
       Prerequisite propagation (line 122, #4 gated on prereq_available):
         the P component += η·max(0, P_prereq(U)+P_prereq(P)−0.5), then renormalized."""
    global_prior = np.asarray(global_prior, dtype=float)
    counts = np.asarray(block_counts, dtype=float) if len(block_counts) else np.array([])
    if counts.size and counts.sum() > 0:
        stacked = np.stack([np.asarray(x, dtype=float) for x in block_beliefs])
        block_post = (stacked * counts[:, None]).sum(0) / counts.sum()
    else:
        block_post = global_prior     # no answers in the block → fall back
    b0 = LAMBDA_PRIOR * block_post + (1 - LAMBDA_PRIOR) * global_prior
    if prereq_available and prereq_belief is not None:   # #4: enabled only after the alignment table is delivered
        pq = np.asarray(prereq_belief, dtype=float)
        b0 = b0.copy()
        b0[P] += ETA_PREREQ * max(0.0, pq[U] + pq[P] - 0.5)
    return b0 / b0.sum()              # renormalize after the addition


# ══════════════════════════════════════════════════════════════════════
# FSRS decay projection + get_belief as the sole read entry (lines 150-153, red line)
# ══════════════════════════════════════════════════════════════════════
def recall_probability(dt_days: float, S: float) -> float:
    """FSRS classic retrievability R(t) = (1 + t/(9S))^(−1) (lines 49/150)."""
    return (1.0 + dt_days / (9.0 * S)) ** (-1.0)


def _project_decay(b_stored: np.ndarray, S, last_review_at, now) -> np.ndarray:
    """★The only place the decay formula lives★ (line 152). One-way projection: the (1−R) share of M flows into C,
       P/U untouched. Idempotent, never written back, stored state never contains decay."""
    b = np.asarray(b_stored, dtype=float).copy()
    if S is None or last_review_at is None:
        return b                       # S not initialized → no decay
    dt_days = max(0.0, (now - last_review_at) / DAY_SECONDS)
    R = recall_probability(dt_days, S)
    lost = (1.0 - R) * b[M]            # mass drained from M
    b[M] = R * b[M]
    b[C] = b[C] + lost                # getting rusty ≈ the phenotype of an unstable reasoning chain (line 153)
    return b                           # P/U untouched; total mass conserved, still normalized


def get_belief(node: NodeBelief, now: float) -> np.ndarray:
    """★The sole read entry★ (line 152, red line). The selector / planner / report /
       memory layers must all read beliefs through this function; reading node.b
       directly is forbidden."""
    return _project_decay(node.b, node.S, node.last_review_at, now)


def stored_belief(node: NodeBelief) -> np.ndarray:
    """An **explicit** read of the stored (undecayed) belief. Exactly one legitimate use:
       the planner's review criterion needs to compare the drop from 'stored P(M)' to
       'projected P(M)' (line 154) — this is not a bypass; the unprojected value is
       wanted on purpose. Use get_belief for everything else. The red-line tests allow
       this accessor but still block bare .b."""
    return np.asarray(node.b, dtype=float).copy()


# ══════════════════════════════════════════════════════════════════════
# Observation update (real answer / retest events)
# ══════════════════════════════════════════════════════════════════════
def observe(node: NodeBelief, likelihood: np.ndarray, now: float,
            is_direct: bool = True, held_out_passed: bool = False) -> NodeBelief:
    """Real answer / retest event (line 152): the prior = the projected b (the student's true state right now);
       the update result is written back to the stored state and last_review_at is
       reset (forgetting for a given time span happens exactly once)."""
    b_prior = get_belief(node, now)               # project once (forgetting happens only here)
    node.b = bayes_update(b_prior, likelihood)    # write back: the state as of now, without future decay
    node.last_review_at = now                      # reset → next Δt is counted from now
    if is_direct:
        node.direct_answers += 1
    maybe_init_S(node, now, held_out_passed=held_out_passed)
    return node


# ══════════════════════════════════════════════════════════════════════
# Stability S initialization and review updates (lines 130/150-151; #9)
# ══════════════════════════════════════════════════════════════════════
def maybe_init_S(node: NodeBelief, now: float, held_out_passed: bool = False) -> None:
    """S₀ trigger (line 130, #9): P(M) crosses 0.7 for the first time with ≥2 direct
       answers on this node, or a held-out pass. A pure-prior push (too few direct
       answers) does not trigger — prevents prior echoes from igniting a fake review chain."""
    if node.S is not None:
        return
    if (node.b[M] > 0.7 and node.direct_answers >= 2) or held_out_passed:
        node.S = S0
        node.last_review_at = now


def review_update_S(node: NodeBelief, passed: bool, now: float,
                    first_review: bool = False) -> NodeBelief:
    """Review-based S update (lines 150-151; 2026-08-13 audit: switched to the FSRS-4.5 damped formula):
       pass → first time ×3 (w0), afterwards via the 4.5 damped growth term; fail → ×w11=0.72.
       S is clamped to [1, S_MAX] to prevent undamped inflation (the old formula reached
       S=4608 days after ×10 iterations; under 4.5 damping it is ≈73 days)."""
    if node.S is None:
        return node
    if passed and not first_review:
        f = _W12                                   # r=1.0 success
        growth = (1.0 + f * (node.S ** (-_W8) + _W7) * (11.0 / 9.0 - node.S / _W10))
        node.S = float(min(max(node.S * growth, 1.0), S_MAX))
    elif passed:
        node.S = float(min(node.S * S_FIRST_REVIEW_MULT, S_MAX))
    else:
        node.S = float(min(node.S * _W11, S_MAX))
    node.last_review_at = now
    return node
