#!/usr/bin/env python3
"""
engine/planner.py — time-budget planner (design doc block 4, lines 187-205)
================================================================
Maximize expected mastery gain under the budget constraint (v1 greedy + table
lookup, no optimizer). Four budget tiers (30min/1h/2h/3h+); 30min = the shallow
diagnosis scope (user decision of 2026-07-08). Outputs session_budget to the
selector (EIG loop termination budget) and the recommender (constraints).

Components:
  · session_budget(tier): table lookup for the diagnostic item count / target node
    count / prescription segment count / durations / follow-up question caps.
  · plan_reviews: review insertion (stored P(M)>0.7 but projected below 0.6 →
    insert 2 items; #9 throttling ≥7 days, ≤2 nodes per session) — the consumer
    of the decay machine.
  · check_exhaustion: budget-exhaustion degradation (diagnostic overruns ×1.5 →
    mark unconverged + archive the prescription + no forced retest).
  · explain_plan: print the schedule and its rationale to the student (explicitly
    requested by the user).
"""
from __future__ import annotations

from typing import Dict, Optional

from engine import mastery as m

# Four-tier budget table (lines 192-198; module constants + comments; values tunable,
# function shape not to be changed).
# 30min: user decision of 2026-07-08 to switch to the shallow diagnosis scope (M/non-M split, no promise of cause convergence).
BUDGET_TABLE = {
    "30min": {"mode": "shallow", "diagnostic_items": 9, "diagnostic_minutes": 12,
              "target_nodes": 1, "rx_segments": 1, "rx_minutes": 8,
              "retest_items": 3, "followup_max": 2, "buffer_minutes": 1},
    "1h":    {"mode": "full", "diagnostic_items": 15, "diagnostic_minutes": 20,
              "target_nodes": 2, "rx_segments": 2, "rx_minutes": 15,
              "retest_items": 5, "followup_max": 3, "buffer_minutes": 10},
    "2h":    {"mode": "full", "diagnostic_items": 25, "diagnostic_minutes": 40,
              "target_nodes": 2, "rx_segments": 4, "rx_minutes": 30,
              "retest_items": 10, "followup_max": 5, "buffer_minutes": 20},
    "3h+":   {"mode": "full", "diagnostic_items": 25, "diagnostic_minutes": 40,
              "target_nodes": 2, "rx_segments": 4, "rx_minutes": 30,
              "retest_items": 10, "followup_max": 5, "buffer_minutes": 30,
              "two_rounds": True},   # two complete loops; don't fill the ending, wrap up honestly
}

REVIEW_STORE_THRESHOLD = 0.7        # only consider a review when stored P(M) is above this
REVIEW_DECAY_THRESHOLD = 0.6        # projection dropping below this triggers a review
REVIEW_ITEMS = 2                    # a review inserts 2 items
REVIEW_THROTTLE_DAYS = 7            # #9: ≥7 days between review prompts for the same node
REVIEW_MAX_NODES = 2               # ≤2 nodes reviewed per session
EXHAUSTION_MULT = 1.5               # diagnostic overrun ×1.5 triggers degradation


def session_budget(tier: str) -> Dict:
    """Table lookup for the tier's budget. Unknown tiers default to 1h."""
    return dict(BUDGET_TABLE.get(tier, BUDGET_TABLE["1h"]))


def estimate_minutes(budget: Dict) -> float:
    """Accounting estimate: diagnosis + prescription + retest + buffer, used to verify
    each tier stays ≤ budget×1.1. Retests are ~0.6 min/item (faster than diagnostic
    items: pure answering, no follow-ups)."""
    diag = budget["diagnostic_minutes"]
    rx = budget["rx_minutes"]
    retest = budget["retest_items"] * 0.6
    buf = budget["buffer_minutes"]
    total = diag + rx + retest + buf
    if budget.get("two_rounds"):
        total = (diag + rx + retest) * 2 + buf      # two complete loops
    return total


def plan_reviews(nodes: Dict[str, m.NodeBelief], now: float,
                 last_review_prompt: Optional[Dict[str, float]] = None) -> Dict:
    """Review insertion (consumer of the decay machine, line 154).
    For each node: stored P(M)>0.7 but the get_belief projection drops below 0.6
    → candidate for review. #9 throttling: don't re-prompt the same node if its last
    prompt was <7 days ago; ≤2 nodes per session."""
    last_review_prompt = last_review_prompt or {}
    candidates = []
    for name, node in nodes.items():
        stored = m.stored_belief(node)              # explicit stored-state read (the review criterion needs the unprojected value)
        if stored[m.M] <= REVIEW_STORE_THRESHOLD:
            continue                                # stored state isn't high to begin with; not a review target
        proj = m.get_belief(node, now)              # read the projection via the sole entry point
        if proj[m.M] >= REVIEW_DECAY_THRESHOLD:
            continue                                # hasn't dropped below; no review needed
        last = last_review_prompt.get(name)
        if last is not None and (now - last) < REVIEW_THROTTLE_DAYS * m.DAY_SECONDS:
            continue                                # throttle: don't re-prompt within 7 days
        drop = stored[m.M] - proj[m.M]              # the drop
        candidates.append((drop, name))
    candidates.sort(reverse=True)                   # larger drops first
    chosen = [name for _, name in candidates[:REVIEW_MAX_NODES]]
    return {"review_nodes": chosen,
            "review_items": REVIEW_ITEMS if chosen else 0}


def check_exhaustion(budget: Dict, elapsed_diagnostic_min: float) -> Dict:
    """Budget-exhaustion degradation (line 203, silent-failure path ②).
    Actual diagnostic time > the tier's diagnostic budget ×1.5 → end the diagnosis
    immediately (even unconverged), still issue the prescription but don't force a
    retest; the state goes to the server log and resumes next time."""
    threshold = budget["diagnostic_minutes"] * EXHAUSTION_MULT
    if elapsed_diagnostic_min > threshold:
        return {"force_stop": True, "force_retest": False,
                "message": "今天时间到了，处方已存好，下次打开从看段开始。"}
    return {"force_stop": False, "force_retest": True, "message": ""}


def explain_plan(budget: Dict) -> str:
    """Print the schedule and its rationale to the student (line 200, explicitly requested by the user). Numbers are generated live from the budget table."""
    if budget["mode"] == "shallow":
        return (f"你今天有约 30 分钟：先用 {budget['diagnostic_items']} 道题快速定位"
                f"「哪些点掌握了、哪些还没」，看 1 段视频补最弱的，再 "
                f"{budget['retest_items']} 题确认。时间有限，今天只做快速定位、不深挖病因。")
    return (f"你今天有这些时间：约 {budget['diagnostic_minutes']} 分钟定位问题（{budget['diagnostic_items']} 题），"
            f"{budget['rx_minutes']} 分钟看 {budget['rx_segments']} 段视频，"
            f"{int(budget['retest_items']*0.6)+1} 分钟证明补上了（{budget['retest_items']} 题），"
            f"剩下 {budget['buffer_minutes']} 分钟是缓冲。")
