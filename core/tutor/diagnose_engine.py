#!/usr/bin/env python3
"""
诊断引擎（总蓝图 B-2）。

根治"诊断不准"：取代"LLM 瞎猜 mastery + 关键词占卜"。
核心：用真题 rubric 得分点 + KG judgment_criteria 做客观校验，LLM 只是辅助。

三个客观机制：
1. check_against_rubric：学生答案与 rubric 得分点逐条比对，算客观命中率。
2. estimate_mastery：客观分为主、LLM 自评为辅融合；漏 must_have 得分点 → 封顶。
3. L1-L4 逐层诊断：保留高质量人工题作金标准，无精校题的节点用 KG common_failures 动态兜底。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.data.knowledge_repository import KGNode, get_knowledge_repository
from core.tutor.profile_model import MasteryRecord, now_iso


# ── 诊断数据结构 ──────────────────────────────────────────

@dataclass
class RubricPoint:
    """一个评分点（题库 item 的 rubric 元素，或从 KG 判据转化）。"""

    point_id: str
    desc: str
    keywords: List[str] = field(default_factory=list)
    score: float = 1.0
    must_have: bool = False
    kg_node: str = ""


@dataclass
class RubricCheckResult:
    item_id: str = ""
    hit_points: List[str] = field(default_factory=list)
    missed_points: List[str] = field(default_factory=list)
    objective_score: float = 0.0  # 命中分 / 总分
    missed_must_have: List[str] = field(default_factory=list)


@dataclass
class MasteryEstimate:
    value: float = 0.5
    objective_component: float = 0.0
    llm_component: float = 0.5
    weight_objective: float = 0.5
    confidence: str = "low"
    weak_axes: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def to_record(self) -> MasteryRecord:
        source = "rubric" if self.weight_objective >= 0.6 else (
            "mixed" if self.objective_component > 0 else "llm"
        )
        conf = {"high": 0.8, "medium": 0.5, "low": 0.3}.get(self.confidence, 0.3)
        return MasteryRecord(
            value=self.value, evidence=list(self.evidence),
            last_updated=now_iso(), source=source, confidence=conf,
        )


@dataclass
class DiagnosticQuestion:
    """一道诊断题（L1-L4）。"""

    id: str
    level: str  # "L1 基础概念" ...
    axis: str
    prompt: str
    look_for: str = ""
    source: str = "kg"  # gold(人工精校) | kg(动态) | self_report


# ── 诊断引擎 ──────────────────────────────────────────────

class DiagnoseEngine:
    def __init__(self, repo=None):
        self.repo = repo or get_knowledge_repository()

    # === 客观校验 ===

    def check_against_rubric(
        self, student_text: str, rubric: List[RubricPoint],
        point_verdicts: Optional[Dict[str, bool]] = None,
    ) -> RubricCheckResult:
        """学生答案与 rubric 逐条比对。

        命中判定优先级（解决"关键词在但答错"的误判，如"转化率升高"vs"转化率降低"）：
        1. 若提供 point_verdicts（LLM 对每个得分点的真伪判定），以它为准——这是最可靠的。
        2. 否则退回关键词命中（快、零成本，但对文字表述题易误判）。
        """
        text = (student_text or "").lower()
        verdicts = point_verdicts or {}
        hit, missed, missed_must = [], [], []
        total = sum(p.score for p in rubric) or 1.0
        gained = 0.0
        for p in rubric:
            if p.point_id in verdicts:
                is_hit = bool(verdicts[p.point_id])  # LLM 判定优先
            else:
                keys = [k.lower() for k in p.keywords if k]
                is_hit = any(k in text for k in keys) if keys else False
            if is_hit:
                hit.append(p.point_id)
                gained += p.score
            else:
                missed.append(p.point_id)
                if p.must_have:
                    missed_must.append(p.point_id)
        return RubricCheckResult(
            hit_points=hit, missed_points=missed,
            objective_score=round(gained / total, 3),
            missed_must_have=missed_must,
        )

    def estimate_mastery(
        self,
        rubric_result: Optional[RubricCheckResult],
        llm_self_score: float,
        has_numeric_answer: bool = False,
        weak_axes: Optional[List[str]] = None,
        evidence: Optional[List[str]] = None,
    ) -> MasteryEstimate:
        """
        融合客观 rubric 和 LLM 自评。
        - 计算/有标准答案题：客观为主(0.75)。
        - 文字表述题：LLM 话语权更大(0.45)。
        - 漏掉 must_have 得分点：mastery 封顶 0.6，LLM 不能抬分。
        """
        llm = max(0.0, min(1.0, float(llm_self_score)))
        evidence = list(evidence or [])
        if rubric_result is None:
            # 无客观锚点（如开放问答），只能用 LLM，标 low 置信
            return MasteryEstimate(
                value=llm, objective_component=0.0, llm_component=llm,
                weight_objective=0.0, confidence="low",
                weak_axes=list(weak_axes or []), evidence=evidence,
            )
        objective = rubric_result.objective_score
        w_obj = 0.75 if has_numeric_answer else 0.45
        value = w_obj * objective + (1 - w_obj) * llm
        # 硬封顶：漏掉必拿得分点
        if rubric_result.missed_must_have:
            value = min(value, 0.6)
            evidence.append(f"漏掉必拿得分点: {rubric_result.missed_must_have}")
        confidence = "high" if has_numeric_answer and not rubric_result.missed_must_have else "medium"
        return MasteryEstimate(
            value=round(value, 3),
            objective_component=objective,
            llm_component=llm,
            weight_objective=w_obj,
            confidence=confidence,
            weak_axes=list(weak_axes or []),
            evidence=evidence,
        )

    # === KG 判据 → rubric（让开放题也有客观锚点）===

    def kg_criteria_as_rubric(self, node_id: str) -> List[RubricPoint]:
        """把 KG 节点的掌握判据转成 rubric（无关键词，供 LLM 对照打分用）。"""
        node = self.repo.find_node(node_id)
        if not node:
            return []
        return [
            RubricPoint(
                point_id=f"{node.node_id}-c{i}",
                desc=crit, keywords=[], score=1.0,
                must_have=(i == 0), kg_node=node.node_id,
            )
            for i, crit in enumerate(node.mastery_rubric)
        ]

    # === L1-L4 逐层诊断 ===

    def build_progressive_questions(
        self, node_id: str, gold_bank: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> List[DiagnosticQuestion]:
        """
        生成逐层诊断题。
        优先用高质量人工精校题(gold_bank)；没有的节点用 KG common_failures 动态兜底。
        这是批判分析的关键修正：不一刀推倒高质量硬编码题。
        """
        node = self.repo.find_node(node_id)
        questions: List[DiagnosticQuestion] = []

        # L0 自评（永远第一问）
        questions.append(DiagnosticQuestion(
            id="self-report", level="L0 自我定位", axis="metacognition",
            prompt="你觉得自己现在最卡的是：概念听不懂、题型不会进、步骤不稳、计算容易错，"
                   "还是看答案懂自己做不出？举一个最近的例子。",
            look_for="能否把'不会'具体化。", source="self_report",
        ))

        # 金标准题（人工精校）优先
        gold = (gold_bank or {}).get(node_id) if gold_bank else None
        if gold:
            for q in gold:
                questions.append(DiagnosticQuestion(
                    id=q.get("id", ""), level=q.get("level", ""),
                    axis=q.get("axis", "concept"), prompt=q.get("prompt", ""),
                    look_for=q.get("look_for", ""), source="gold",
                ))
            return questions[:6]

        # 动态兜底：用 KG common_failures（每条自带高质量诊断问）
        if node:
            levels = ["L1 基础概念", "L2 入口判断", "L3 应试坑点", "L4 综合迁移"]
            for i, cf in enumerate(node.diagnostic_questions()[:4]):
                questions.append(DiagnosticQuestion(
                    id=f"{node.node_id}-l{i+1}",
                    level=levels[i] if i < len(levels) else f"L{i+1}",
                    axis=node.thinking_patterns[0] if node.thinking_patterns else "concept",
                    prompt=cf.diagnostic_question,
                    look_for=f"避开：{cf.symptom}", source="kg",
                ))
        return questions[:6]

    def decide_next_level(
        self, current_index: int, mastery: float, total: int
    ) -> str:
        """逐层升降级。≥0.7 升，<0.5 留同层，连续卡→进执行。"""
        if current_index + 1 >= total:
            return "finish_to_execution"
        if mastery >= 0.7:
            return "ask_next"
        if mastery < 0.5:
            return "ask_same"
        return "ask_next"
