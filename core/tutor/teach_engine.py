#!/usr/bin/env python3
"""
教学引擎（总蓝图 B-3 + 第3章活人感）。

根治"讲解太表面"+ 注入"活人感"，五个机制：
1. 标准解锚点：注入真题标准解 + rubric + 题型骨架，LLM 把标准解翻成一化儿语气，不自由发挥。
2. 一化儿风格层：注入语料 + 八大招式，负责"怎么讲"。
3. 多讲法策略库（活人感②）：卡住时换轨（逻辑→数值→类比→反例），不是讲慢。
4. 恶性循环检测：连续答错且上轮抛了新题 → 强制回最小概念，禁止再抛新题。
5. 分层深讲 + 反人机准则：按状态控制深度；克制、有边界、不表演。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.data.knowledge_repository import ExamPattern, KGNode, get_knowledge_repository


# ── 多讲法策略（活人感②的灵魂）──────────────────────────────
# 同一知识点的 N 种正交讲法。卡住时换轨，而非把同一种讲法讲慢。
TEACHING_ANGLES = {
    "logic_derivation": "逻辑推导：从原理一步步推出结论",
    "numeric_example": "数值实例：代入具体数字算给学生看",
    "analogy_visual": "类比/画图：用形象比喻或图像让学生直观理解",
    "counter_example": "反例对比：故意给一个错的，让学生看清为什么错",
}
ANGLE_ORDER = ["logic_derivation", "numeric_example", "analogy_visual", "counter_example"]


@dataclass
class TeachingContext:
    """注入教学 prompt 的"三柱子"+ 活人感材料。"""

    node_id: str = ""
    # 正确性层
    standard_solution: Optional[Dict[str, Any]] = None  # 来自 item_bank（有则注入）
    rubric_desc: List[str] = field(default_factory=list)  # 得分点/判据描述
    solving_skeleton: List[Dict[str, str]] = field(default_factory=list)  # 题型 solving_steps
    setter_traps: List[Dict[str, str]] = field(default_factory=list)  # 出题人陷阱
    # 灵魂层
    yihuier_chunks: List[str] = field(default_factory=list)  # 一化儿讲法片段（参考语气）
    thinking_moves: List[str] = field(default_factory=list)  # 八大招式
    # 引流出口
    recommended_video: Optional[Dict[str, Any]] = None
    # 教学控制
    suggested_angle: str = "logic_derivation"  # 本轮建议讲法角度
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

    # === 构建教学上下文（接进护城河）===

    def build_teaching_context(
        self,
        node_id: str,
        item: Optional[Dict[str, Any]] = None,
        pattern_id: str = "",
        used_angles: Optional[List[str]] = None,
    ) -> TeachingContext:
        node: Optional[KGNode] = self.repo.find_node(node_id)
        ctx = TeachingContext(node_id=node_id, used_angles=list(used_angles or []))

        # 正确性层：题库标准解（最强锚点）
        if item:
            ctx.standard_solution = item.get("standard_solution")
            ctx.rubric_desc = [p.get("desc", "") for p in item.get("rubric", [])]
        # 无题库则退化到 KG 判据当 rubric
        if not ctx.rubric_desc and node:
            ctx.rubric_desc = node.mastery_rubric

        # 题型解题骨架
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

        # 灵魂层：一化儿讲法 + 招式
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

        # 建议讲法角度
        ctx.suggested_angle = self.next_angle(ctx.used_angles)
        return ctx

    # === 多讲法策略（活人感②）===

    def next_angle(self, used_angles: List[str]) -> str:
        """选一个还没用过的讲法角度。全用过则回到逻辑推导。"""
        for a in ANGLE_ORDER:
            if a not in used_angles:
                return a
        return ANGLE_ORDER[0]

    def select_depth(self, affect: str, fail_streak: int) -> str:
        """
        分层深讲：
        - deep：连续答错或挫败 → 逐行代入数据讲透因果链。
        - confirm：状态好、只需确认 → 只追问不灌输。
        - normal：默认。
        """
        if fail_streak >= 2 or affect in ("frustrated", "挫败"):
            return "deep"
        if affect in ("confident", "自信") and fail_streak == 0:
            return "confirm"
        return "normal"

    # === 恶性循环检测（活人感核心：打断"没懂→抛新题→又错"）===

    def detect_vicious_cycle(
        self, node_id: str, recent_masteries: List[float], last_tutor_action: str
    ) -> ViciousCycleFlag:
        """
        同一节点连续 2+ 次 mastery<0.5，且上轮抛了新题 → 恶性循环。
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
