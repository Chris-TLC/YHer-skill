#!/usr/bin/env python3
"""
知识资产仓库：把 65 节点知识图谱 + 13 题型库这些"金矿"读成引擎能直接用的结构。

设计原则（来自总蓝图第 2 章）：
- engine/data 层零界面依赖，纯数据进、dataclass 出。
- 这一层是"护城河接线"的物理入口：批判分析指出现有 KG 的
  judgment_criteria_for_mastery / common_failures / solving_steps
  从来没被编排层真正用上。这个仓库就是把它们暴露出来。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KG_FILE = DATA_DIR / "knowledge_graph_full.jsonl"
EXAM_PATTERN_FILE = DATA_DIR / "exam_patterns_curated.jsonl"


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class CommonFailure:
    """一条常见错误：它自带一个高质量诊断问题，是 L2 诊断的金标准来源。"""

    cause: str
    symptom: str
    diagnostic_question: str


@dataclass
class RecommendedVideo:
    """一化儿真实视频（引流出口）。绝不编造，只从 KG 读。"""

    bv: str
    p_number: int
    what_you_learn: str = ""
    completion_criterion: str = ""
    duration_min: float = 0.0
    difficulty: str = ""

    @property
    def url(self) -> str:
        return f"https://www.bilibili.com/video/{self.bv}?p={self.p_number}"


@dataclass
class KGNode:
    """一个知识图谱节点。它已经自带了我们需要的几乎一切。"""

    node_id: str
    category: str = ""
    prerequisites: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    difficulty: str = "T2"
    exam_weight: str = "中"
    notes: str = ""
    # 金矿字段：
    common_exam_patterns: List[str] = field(default_factory=list)
    common_failures: List[CommonFailure] = field(default_factory=list)
    judgment_criteria: List[str] = field(default_factory=list)  # = 现成的掌握 rubric
    thinking_patterns: List[str] = field(default_factory=list)
    videos: List[RecommendedVideo] = field(default_factory=list)

    @property
    def mastery_rubric(self) -> List[str]:
        """掌握判据就是这个节点的客观 rubric。"""
        return self.judgment_criteria

    def diagnostic_questions(self) -> List[CommonFailure]:
        """每条常见错误自带一个诊断问题。"""
        return self.common_failures


@dataclass
class SolvingStep:
    step: int
    action: str
    why: str = ""
    common_mistake: str = ""


@dataclass
class SetterTrap:
    """出题人视角：一个陷阱 + 怎么避开。"""

    trap_type: str
    what_setter_wants: str
    how_to_avoid: str


@dataclass
class ExamPattern:
    """一个题型：自带解题骨架和出题人视角，是教学引擎的解题锚点。"""

    pattern_id: str
    category: str = ""
    subcategory: str = ""
    score_value: str = ""
    linked_topics: List[str] = field(default_factory=list)
    solving_steps: List[SolvingStep] = field(default_factory=list)
    setter_traps: List[SetterTrap] = field(default_factory=list)
    common_traps: List[str] = field(default_factory=list)
    completion_criterion: str = ""


# ── 仓库 ──────────────────────────────────────────────────

class KnowledgeRepository:
    """统一读取 KG + 题型库。模块级缓存，首次加载后常驻。"""

    def __init__(self, kg_file: Optional[Path] = None, pattern_file: Optional[Path] = None):
        self._kg_file = Path(kg_file) if kg_file else KG_FILE
        self._pattern_file = Path(pattern_file) if pattern_file else EXAM_PATTERN_FILE
        self._nodes: Dict[str, KGNode] = {}
        self._patterns: Dict[str, ExamPattern] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load_kg()
        self._load_patterns()
        self._loaded = True

    def _load_kg(self):
        if not self._kg_file.exists():
            return
        with open(self._kg_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                node = KGNode(
                    node_id=raw["node_id"],
                    category=raw.get("category", ""),
                    prerequisites=raw.get("prerequisites", []),
                    successors=raw.get("successors", []),
                    difficulty=raw.get("difficulty", "T2"),
                    exam_weight=raw.get("exam_weight", "中"),
                    notes=raw.get("notes", ""),
                    common_exam_patterns=raw.get("common_exam_patterns", []),
                    common_failures=[
                        CommonFailure(
                            cause=cf.get("cause", ""),
                            symptom=cf.get("symptom", ""),
                            diagnostic_question=cf.get("diagnostic_question", ""),
                        )
                        for cf in raw.get("common_failures", [])
                    ],
                    judgment_criteria=raw.get("judgment_criteria_for_mastery", []),
                    thinking_patterns=raw.get("thinking_patterns_used", []),
                    videos=[
                        RecommendedVideo(
                            bv=v.get("bv", ""),
                            p_number=v.get("p_number", 1),
                            what_you_learn=v.get("what_you_learn", ""),
                            completion_criterion=v.get("completion_criterion", ""),
                            duration_min=v.get("duration_min", 0.0),
                            difficulty=v.get("difficulty", ""),
                        )
                        for v in raw.get("recommended_videos", [])
                        if v.get("bv")
                    ],
                )
                self._nodes[node.node_id] = node

    def _load_patterns(self):
        if not self._pattern_file.exists():
            return
        with open(self._pattern_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                pattern = ExamPattern(
                    pattern_id=raw["pattern_id"],
                    category=raw.get("category", ""),
                    subcategory=raw.get("subcategory", ""),
                    score_value=str(raw.get("score_value", "")),
                    linked_topics=raw.get("linked_knowledge_topics", []),
                    solving_steps=[
                        SolvingStep(
                            step=s.get("step", i + 1),
                            action=s.get("action", ""),
                            why=s.get("why", ""),
                            common_mistake=s.get("common_mistake", ""),
                        )
                        for i, s in enumerate(raw.get("solving_steps", []))
                    ],
                    setter_traps=[
                        SetterTrap(
                            trap_type=t.get("trap_type", ""),
                            what_setter_wants=t.get("what_setter_wants", ""),
                            how_to_avoid=t.get("how_to_avoid", ""),
                        )
                        for t in raw.get("setter_perspective", [])
                    ],
                    common_traps=raw.get("common_traps", []),
                    completion_criterion=raw.get("completion_criterion", ""),
                )
                self._patterns[pattern.pattern_id] = pattern

    # ── 对外接口 ──────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[KGNode]:
        self._ensure_loaded()
        return self._nodes.get(node_id)

    def all_nodes(self) -> List[KGNode]:
        self._ensure_loaded()
        return list(self._nodes.values())

    def find_node(self, text: str) -> Optional[KGNode]:
        """按文本模糊匹配一个最相关的节点（别名/包含）。"""
        self._ensure_loaded()
        if not text:
            return None
        # 先精确，再包含，再被包含
        if text in self._nodes:
            return self._nodes[text]
        for nid, node in self._nodes.items():
            if nid in text or text in nid:
                return node
        return None

    def get_prerequisites(self, node_id: str, depth: int = 2) -> List[str]:
        """反查前置链路（保序去重）。"""
        self._ensure_loaded()
        visited, result, queue = set(), [], [node_id]
        for _ in range(depth):
            nxt = []
            for nid in queue:
                if nid in visited:
                    continue
                visited.add(nid)
                node = self._nodes.get(nid)
                if not node:
                    continue
                for p in node.prerequisites:
                    if p not in visited:
                        result.append(p)
                        nxt.append(p)
            queue = nxt
        # 去重保序
        seen, out = set(), []
        for r in result:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def get_pattern(self, pattern_id: str) -> Optional[ExamPattern]:
        self._ensure_loaded()
        return self._patterns.get(pattern_id)

    def all_patterns(self) -> List[ExamPattern]:
        self._ensure_loaded()
        return list(self._patterns.values())

    def find_patterns_for_node(self, node_id: str) -> List[ExamPattern]:
        """找出与某知识点关联的题型。"""
        self._ensure_loaded()
        out = []
        for p in self._patterns.values():
            if any(node_id in t or t in node_id for t in p.linked_topics):
                out.append(p)
        return out


@lru_cache(maxsize=1)
def get_knowledge_repository() -> KnowledgeRepository:
    """全局单例。"""
    return KnowledgeRepository()
