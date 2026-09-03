#!/usr/bin/env python3
"""
Teaching engine (master blueprint B-3 + chapter 3, "alive teacher" feel).

Fixes "explanations too superficial" + injects "alive teacher" feel, via five mechanisms:
1. Standard-solution anchor: inject the real-item standard solution + rubric +
   exam-pattern skeleton; the LLM rephrases the standard solution in the
   yihuier voice instead of improvising.
2. Yihuier style layer: inject corpus + the eight thinking moves, owning the
   "how to explain" dimension.
3. Multi-angle explanation strategy library ("alive teacher" ②): switch tracks
   when stuck (logic → numeric → analogy → counter-example), not just slow down.
4. Vicious-cycle detection: consecutive wrong answers plus a new question thrown
   last round → force a return to the minimal concept, no new questions.
5. Tiered deep-explaining + anti-robotic rules: depth follows state; restrained,
   boundary-respecting, never performative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.data.knowledge_repository import ExamPattern, KGNode, get_knowledge_repository


# ── Multi-angle explanation strategies (soul of "alive teacher" ②) ─────
# N orthogonal ways to explain the same knowledge point. When stuck, switch
# tracks instead of slowing down the same explanation.
TEACHING_ANGLES = {
    "logic_derivation": "逻辑推导：从原理一步步推出结论",
    "numeric_example": "数值实例：代入具体数字算给学生看",
    "analogy_visual": "类比/画图：用形象比喻或图像让学生直观理解",
    "counter_example": "反例对比：故意给一个错的，让学生看清为什么错",
}
ANGLE_ORDER = ["logic_derivation", "numeric_example", "analogy_visual", "counter_example"]


@dataclass
class TeachingContext:
    """The "three pillars" injected into the teaching prompt + "alive teacher" materials."""

    node_id: str = ""
    # Correctness layer
    standard_solution: Optional[Dict[str, Any]] = None  # from the item bank (injected when present)
    rubric_desc: List[str] = field(default_factory=list)  # scoring-point / criterion descriptions
    solving_skeleton: List[Dict[str, str]] = field(default_factory=list)  # exam-pattern solving_steps
    setter_traps: List[Dict[str, str]] = field(default_factory=list)  # setter traps
    # Soul layer
    yihuier_chunks: List[str] = field(default_factory=list)  # yihuier explanation snippets (for tone reference)
    thinking_moves: List[str] = field(default_factory=list)  # the eight thinking moves
    # Funnel exit
    recommended_video: Optional[Dict[str, Any]] = None
    # Teaching control
    suggested_angle: str = "logic_derivation"  # suggested explanation angle for this round
    used_angles: List[str] = field(default_factory=list)


@dataclass
class ViciousCycleFlag:
    triggered: bool = False
    node_id: str = ""
    fail_streak: int = 0
    reason: str = ""


class TeachEngine:
    def __init__(self, repo=None, retriever=None):
        self.repo = repo or get_knowledge_repository()
        self.retriever = retriever

    # === Build the teaching context (plugs into the moat) ===

    def build_teaching_context(
        self,
        node_id: str,
        item: Optional[Dict[str, Any]] = None,
        pattern_id: str = "",
        used_angles: Optional[List[str]] = None,
    ) -> TeachingContext:
        node: Optional[KGNode] = self.repo.find_node(node_id)
        ctx = TeachingContext(node_id=node_id, used_angles=list(used_angles or []))

        # Correctness layer: the item bank's standard solution (strongest anchor)
        if item:
            ctx.standard_solution = item.get("standard_solution")
            ctx.rubric_desc = [p.get("desc", "") for p in item.get("rubric", [])]
        # Without an item-bank entry, fall back to KG criteria as the rubric
        if not ctx.rubric_desc and node:
            ctx.rubric_desc = node.mastery_rubric

        # Exam-pattern solving skeleton
        pattern: Optional[ExamPattern] = None
        if pattern_id:
            pattern = self.repo.get_pattern(pattern_id)
        elif node:
            pats = self.repo.find_patterns_for_node(node_id)
            pattern = pats[0] if pats else None
        if pattern:
            ctx.solving_skeleton = [
                {"step": str(s.step), "action": s.action, "why": s.why,
                 "common_mistake": s.common_mistake}
                for s in pattern.solving_steps
            ]
            ctx.setter_traps = [
                {"trap": t.trap_type, "wants": t.what_setter_wants, "avoid": t.how_to_avoid}
                for t in pattern.setter_traps[:5]
            ]

        # Soul layer: yihuier explanations + thinking moves
        if node:
            ctx.thinking_moves = node.thinking_patterns
            if node.videos:
                v = node.videos[0]
                ctx.recommended_video = {
                    "bv": v.bv, "p_number": v.p_number, "url": v.url,
                    "completion_criterion": v.completion_criterion,
                }
        if self.retriever is not None:
            try:
                chunks = self.retriever.retrieve(node_id, top_k=3)
                ctx.yihuier_chunks = [
                    c.get("text_preview", "")[:200] for c in chunks if c.get("text_preview")
                ]
            except Exception:
                pass

        # Suggested explanation angle
        ctx.suggested_angle = self.next_angle(ctx.used_angles)
        return ctx

    # === Multi-angle explanation strategy ("alive teacher" ②) ===

    def next_angle(self, used_angles: List[str]) -> str:
        """Pick an explanation angle that hasn't been used. If all are used, fall back to logic derivation."""
        for a in ANGLE_ORDER:
            if a not in used_angles:
                return a
        return ANGLE_ORDER[0]

    def select_depth(self, affect: str, fail_streak: int) -> str:
        """
        Tiered deep-explaining:
        - deep: consecutive wrong answers or frustration → substitute data line by line and explain the causal chain thoroughly.
        - confirm: in good shape, only needs confirmation → just probe, don't lecture.
        - normal: default.
        """
        if fail_streak >= 2 or affect in ("frustrated", "挫败"):
            return "deep"
        if affect in ("confident", "自信") and fail_streak == 0:
            return "confirm"
        return "normal"

    # === Vicious-cycle detection (core of the "alive" feel: interrupt "didn't get it → new question → wrong again") ===

    def detect_vicious_cycle(
        self, node_id: str, recent_masteries: List[float], last_tutor_action: str
    ) -> ViciousCycleFlag:
        """
        Same node with 2+ consecutive mastery<0.5 and a new question thrown
        last round → vicious cycle.
        """
        low_streak = 0
        for m in reversed(recent_masteries):
            if m < 0.5:
                low_streak += 1
            else:
                break
        threw_new = last_tutor_action in ("worked_example", "adaptive_practice")
        if low_streak >= 2 and threw_new:
            return ViciousCycleFlag(
                triggered=True, node_id=node_id, fail_streak=low_streak,
                reason=f"同一知识点连续{low_streak}轮未过且上轮抛了新题，强制回最小概念。",
            )
        return ViciousCycleFlag(fail_streak=low_streak)
