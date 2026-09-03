#!/usr/bin/env python3
"""
engine/memory.py — three-layer long-term memory L2/L3 (diagnosis-engine design doc
block 3, 2026-07-07)
==========================================================================
Spec source (sole authority): mac-unknown-design-20260707-230853.md block 3
(lines 147-185). L1 (state layer = mastery + FSRS decay) already lives in
mastery.py; this module owns L2 episodic memory + L3 narrative distillation.

A port of Hermes' "periodically distill skills" pattern: after a year, acting
like a real person comes not from stuffing a year of context in, but from
replacing the context with a ≤500-character distillate (constant tokens).

Boundaries (pure functions, zero new infrastructure):
  · The five L2 admission rules are hardcoded; no LLM seat-of-the-pants judgment
    (lines 159-163).
  · The three retrieval scoring weights w_r/w_s/w_v are hardcoded (line 167); the
    BGE-M3 similarity is made an **injectable function** (tests pass fake vectors,
    production wires the real BGE-M3 + sqlite-vec) — this module never touches the
    vector store.
  · For L3 distillation this module only judges the **trigger** (lazy: >7 days +
    new L2); the LLM call belongs to the wiring layer.
  · The anti-robotic gate overrides everything (lines 183-185): at most 1 reference
    per session opening, and only when today's diagnosis touches a node with old events.
  · **#7 self-verification loop isolation (red line)**: the L3 profile feeds only
    expression touchpoints and is barred from evidence touchpoints (the follow-up
    judge). This module provides injectable_for(touchpoint) to distinguish them
    explicitly; judge touchpoints always return empty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from engine import mastery as m

# ── The five L2 admission types (same source as study_schema.L2_EVENT_TYPES, restated here for judgment) ──
L2_TYPES = ("belief_flip", "breakthrough", "stubborn_error",
            "high_info_followup", "high_efficacy_watch")

# ── Admission thresholds (lines 159-162, module constants) ──
BELIEF_FLIP_UP = 0.7              # P(M) crossing 0.7 for the first time
BELIEF_FLIP_DOWN = 0.5           # P(M) dropping below 0.5
STUBBORN_ERROR_COUNT = 3         # the ≥3rd occurrence of the same error code
HIGH_INFO_CONF = 0.7             # follow-up judge confidence ≥0.7 to enter L2 (low confidence only gets logged)
HIGH_EFFICACY_L1_DELTA = 0.3     # a single Bayesian update after a watched segment with ‖Δb‖₁ ≥ 0.3

# ── Retrieval scoring weights (lines 167-168, module constants) ──
W_RECENCY, W_SALIENCE, W_VECTOR = 0.4, 0.3, 0.3
RECENCY_TAU_DAYS = 30.0
SALIENCE_WEIGHT = {                # salience weight by type via table lookup (lines 168-169)
    "breakthrough": 1.0, "stubborn_error": 1.0,
    "belief_flip": 0.8, "high_info_followup": 0.9, "high_efficacy_watch": 0.6,
}

# ── L3 distillation trigger (line 175, hardcoded) ──
DISTILL_INTERVAL_DAYS = 7.0

# ── Anti-robotic gate (lines 183-184, hard rule) ──
MAX_REFERENCES_PER_SESSION = 1   # at most 1 reference per session opening

# ── Touchpoint classification (#7 self-verification loop isolation, line 104 red line) ──
EXPRESSION_TOUCHPOINTS = frozenset({"opening", "report"})   # expression class: L3 injectable
EVIDENCE_TOUCHPOINTS = frozenset({"followup_triage"})       # evidence class: L3 injection forbidden


# ══════════════════════════════════════════════════════════════════════
# L2 events (in-memory representation; persisted via study_schema.l2_event_row)
# ══════════════════════════════════════════════════════════════════════
@dataclass
class L2Event:
    """An L2 episodic-memory event. The vector is prefilled by the wiring layer using BGE-M3 (summary + node name); this module only consumes it."""
    ts: float
    node: str
    event_type: str
    summary: str
    related_event_id: Optional[str] = None
    vector: Optional[Sequence[float]] = None
    subject: str = "chemistry"


# ══════════════════════════════════════════════════════════════════════
# L2 admission judgment (five hardcoded rules; returns event_type or None)
# ══════════════════════════════════════════════════════════════════════
def admit_belief_flip(pm_before: float, pm_after: float) -> bool:
    """Belief flip: P(M) crosses 0.7 for the first time or drops below 0.5."""
    crossed_up = pm_before < BELIEF_FLIP_UP <= pm_after
    crossed_down = pm_before >= BELIEF_FLIP_DOWN > pm_after
    return crossed_up or crossed_down


def admit_stubborn_error(error_code_count: int) -> bool:
    """Stubborn error: the ≥3rd cumulative occurrence of the same error code."""
    return error_code_count >= STUBBORN_ERROR_COUNT


def admit_high_info_followup(confidence: float) -> bool:
    """High-information follow-up: judge confidence ≥0.7 to enter L2 (low confidence only leaves an event log, preventing low-signal flooding)."""
    return confidence >= HIGH_INFO_CONF


def admit_high_efficacy_watch(b_before: Sequence[float], b_after: Sequence[float]) -> bool:
    """High-efficacy watch: a single Bayesian update after a watched segment with ‖Δb‖₁ ≥ 0.3."""
    a = [float(x) for x in b_before]
    c = [float(x) for x in b_after]
    if len(a) != len(c):
        raise ValueError("belief vector dimensions must match")
    l1 = sum(abs(ai - ci) for ai, ci in zip(a, c))
    return l1 >= HIGH_EFFICACY_L1_DELTA


def classify_admission(*, pm_before: Optional[float] = None, pm_after: Optional[float] = None,
                       held_out_passed: bool = False, error_code_count: Optional[int] = None,
                       followup_confidence: Optional[float] = None,
                       b_before: Optional[Sequence[float]] = None,
                       b_after: Optional[Sequence[float]] = None) -> Optional[str]:
    """Judge one event against the five admission rules; returns the admitted event_type or None (not admitted).
    Priority: breakthrough > belief flip > stubborn error > high-efficacy watch >
    high-information follow-up (when one event hits multiple rules, keep the
    strongest salience)."""
    if held_out_passed:
        return "breakthrough"
    if pm_before is not None and pm_after is not None and admit_belief_flip(pm_before, pm_after):
        return "belief_flip"
    if error_code_count is not None and admit_stubborn_error(error_code_count):
        return "stubborn_error"
    if (b_before is not None and b_after is not None
            and admit_high_efficacy_watch(b_before, b_after)):
        return "high_efficacy_watch"
    if followup_confidence is not None and admit_high_info_followup(followup_confidence):
        return "high_info_followup"
    return None


# ══════════════════════════════════════════════════════════════════════
# L2 retrieval scoring (the "human-like attention" function form, all weights hardcoded; lines 165-170)
# ══════════════════════════════════════════════════════════════════════
def _cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """Default cosine similarity. Missing vectors → 0 (the vector term contributes nothing; degrades to recency+salience)."""
    if a is None or b is None:
        return 0.0
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def retrieval_score(event: L2Event, now: float, context_vector: Optional[Sequence[float]],
                    sim_fn: Callable[[Optional[Sequence[float]], Optional[Sequence[float]]], float] = _cosine
                    ) -> float:
    """score = w_r·exp(−Δt/30 days) + w_s·salience weight + w_v·cos(event vector, today's context vector).
    sim_fn is injectable (production wires the real BGE-M3; default cosine is for tests)."""
    dt_days = max(0.0, (now - event.ts) / m.DAY_SECONDS)
    recency = math.exp(-dt_days / RECENCY_TAU_DAYS)
    salience = SALIENCE_WEIGHT.get(event.event_type, 0.5)
    vec = sim_fn(event.vector, context_vector)
    return W_RECENCY * recency + W_SALIENCE * salience + W_VECTOR * vec


def rank_events(events: Sequence[L2Event], now: float,
                context_vector: Optional[Sequence[float]] = None,
                sim_fn: Callable = _cosine) -> List[L2Event]:
    """Descending retrieval score; ties broken by recency (deterministic)."""
    scored = [(retrieval_score(e, now, context_vector, sim_fn), -e.ts, i, e)
              for i, e in enumerate(events)]
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [e for *_, e in scored]


# ══════════════════════════════════════════════════════════════════════
# Anti-robotic gate (overrides everything; existence is separated from expression, lines 183-185)
# ══════════════════════════════════════════════════════════════════════
def select_references(events: Sequence[L2Event], today_nodes: Sequence[str], now: float,
                      context_vector: Optional[Sequence[float]] = None,
                      sim_fn: Callable = _cosine,
                      max_refs: int = MAX_REFERENCES_PER_SESSION) -> List[L2Event]:
    """L2 events citable in the opening: citation is allowed **only when today's
    diagnosis touches a node that has old events**, at most max_refs per session.
    Expression must serve the current action; aimless small talk is forbidden.
    L1/L2/L3 are all computed every session; what reaches the screen is constrained
    by this gate (existence ≠ expression)."""
    today = set(today_nodes)
    eligible = [e for e in events if e.node in today]     # only reference nodes touched today
    ranked = rank_events(eligible, now, context_vector, sim_fn)
    limit = min(MAX_REFERENCES_PER_SESSION, max(0, int(max_refs)))
    return ranked[:limit]


# ══════════════════════════════════════════════════════════════════════
# L3 distillation trigger (lazy; this module only judges the trigger, the LLM call belongs to the wiring layer; line 175)
# ══════════════════════════════════════════════════════════════════════
def should_distill(last_distill_at: Optional[float], now: float,
                   new_l2_since_last: int) -> bool:
    """Lazy trigger at session opening: >7 days since the last distillation AND new
    L2 events in between → trigger. Never distilled before (last_distill_at=None)
    with new L2 events → trigger."""
    if new_l2_since_last <= 0:
        return False
    if last_distill_at is None:
        return True
    return (now - last_distill_at) / m.DAY_SECONDS > DISTILL_INTERVAL_DAYS


def distill_inputs(l2_events: Sequence[L2Event], belief_deltas: Dict[str, float],
                   prev_profile: Optional[Dict]) -> Dict:
    """Assemble the distillation input (the LLM reads [last week's L2 + L1 belief deltas + the previous narrative] → a ≤500-character profile).
    This module only assembles the input and never calls the LLM; the profile
    structure the LLM outputs is in design doc line 178."""
    return {
        "l2_events": [{"ts": e.ts, "node": e.node, "type": e.event_type,
                       "summary": e.summary} for e in l2_events],
        "belief_deltas": dict(belief_deltas),
        "prev_profile": prev_profile or {},
        "expected_schema": ["学习节奏", "方法偏好", "顽固病灶", "近期突破",
                            "沟通特征", "语言画像"],
    }


# ══════════════════════════════════════════════════════════════════════
# #7 self-verification loop isolation (red line: the L3 profile feeds only expression touchpoints, barred from evidence touchpoints)
# ══════════════════════════════════════════════════════════════════════
def injectable_for(touchpoint: str, l3_profile: Optional[Dict]) -> Optional[Dict]:
    """Return the L3 profile injectable at this touchpoint. **Evidence touchpoints
    (the follow-up judge) always return None** — otherwise the profile writing
    'stubborn flaw = X' → the judge becomes biased toward seeing X → L2 records it
    again → distillation affirms X even more, and the inference layer self-reinforces.
    Expression touchpoints (opening/report) get the profile itself."""
    if touchpoint in EVIDENCE_TOUCHPOINTS:
        return None                            # red line: no injection into evidence touchpoints
    if touchpoint in EXPRESSION_TOUCHPOINTS:
        return l3_profile
    return None                                # unknown touchpoints denied by default (safe default)


# ══════════════════════════════════════════════════════════════════════
# Growth curve (L1 weekly snapshots; line 155, free output; gated on a minimum answer count)
# ══════════════════════════════════════════════════════════════════════
MIN_ANSWERS_FOR_CURVE = 10       # only shown after a cumulative ≥10 answers (avoid small-sample jitter misleading)


def curve_eligible(cumulative_answers: int) -> bool:
    """Growth-curve display criterion: this node has a cumulative ≥10 answers (same source as the L3 lighting criterion)."""
    return cumulative_answers >= MIN_ANSWERS_FOR_CURVE
