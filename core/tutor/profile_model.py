#!/usr/bin/env python3
"""
两层学生模型（总蓝图 B-6）。

为阶段二多学科统一学生模型预留：
- UniversalAbility：通用能力层，跨学科共享（化学的"计算粗心"和物理的"计算粗心"汇到同一维）。
- SubjectAbility：学科能力层，每科一份，挂自己的 KG 节点掌握度。
- MasteryRecord：每个知识点掌握度带证据带来源（rubric 客观的可信度高于 LLM 自评）。

设计原则：纯 dataclass，可 to_dict/from_dict 与本地 JSON / Supabase 互转。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── 通用能力 6 维（跨学科共享）─────────────────────────────
UNIVERSAL_AXES = {
    "reading": "审题抓关键词",
    "procedure": "固定流程执行",
    "metacognition": "自我判断",
    "calculation": "计算与表达",
    "transfer": "变式迁移",
    "persistence": "学习耐性",
}


@dataclass
class UniversalAbility:
    """通用能力层。0-1 区间，0.5 为未知中性。"""

    reading: float = 0.5
    procedure: float = 0.5
    metacognition: float = 0.5
    calculation: float = 0.5
    transfer: float = 0.5
    persistence: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "UniversalAbility":
        d = d or {}
        return cls(**{k: float(d.get(k, 0.5)) for k in UNIVERSAL_AXES})

    def weakest(self, n: int = 3) -> List[str]:
        return [k for k, _ in sorted(self.to_dict().items(), key=lambda x: x[1])[:n]]


@dataclass
class MasteryRecord:
    """一个知识点的掌握度，带证据和来源。"""

    value: float = 0.5
    evidence: List[str] = field(default_factory=list)  # "在2023全国甲T28漏了三段式"
    last_updated: str = ""
    source: str = "llm"  # rubric | llm | mixed —— rubric 来源可信度最高
    confidence: float = 0.3
    belief: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    stability: Optional[float] = None
    last_review_at: Optional[float] = None
    direct_answers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MasteryRecord":
        return cls(
            value=float(d.get("value", 0.5)),
            evidence=list(d.get("evidence", [])),
            last_updated=d.get("last_updated", ""),
            source=d.get("source", "llm"),
            confidence=float(d.get("confidence", 0.3)),
            belief=[float(x) for x in d.get("belief", [0.25, 0.25, 0.25, 0.25])],
            stability=None if d.get("stability") is None else float(d["stability"]),
            last_review_at=(
                None if d.get("last_review_at") is None else float(d["last_review_at"])
            ),
            direct_answers=int(d.get("direct_answers", 0)),
        )


@dataclass
class SubjectAbility:
    """学科能力层，每科一份。"""

    subject: str = "chemistry"
    kg_mastery: Dict[str, MasteryRecord] = field(default_factory=dict)
    pattern_mastery: Dict[str, float] = field(default_factory=dict)

    @property
    def weak_nodes(self) -> List[str]:
        return [nid for nid, r in self.kg_mastery.items() if r.value < 0.6]

    @property
    def mastered_nodes(self) -> List[str]:
        return [nid for nid, r in self.kg_mastery.items() if r.value >= 0.8]

    def update_node(self, node_id: str, record: MasteryRecord):
        """更新某节点掌握度。rubric 来源优先覆盖 llm 来源。"""
        old = self.kg_mastery.get(node_id)
        if old and old.source == "rubric" and record.source == "llm":
            # 已有客观证据，不让 LLM 自评覆盖客观判分
            old.evidence.extend(record.evidence)
            old.evidence = old.evidence[-6:]
            return
        self.kg_mastery[node_id] = record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "kg_mastery": {k: v.to_dict() for k, v in self.kg_mastery.items()},
            "pattern_mastery": dict(self.pattern_mastery),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubjectAbility":
        return cls(
            subject=d.get("subject", "chemistry"),
            kg_mastery={
                k: MasteryRecord.from_dict(v) for k, v in (d.get("kg_mastery", {}) or {}).items()
            },
            pattern_mastery=dict(d.get("pattern_mastery", {}) or {}),
        )


@dataclass
class StudentModel:
    """完整学生模型。阶段一只有 chemistry，阶段二加 physics/math…"""

    user_id: str = "local_demo"
    grade: str = "高二"
    region: str = "全国卷"
    exam_system: str = "高考"
    goals: str = ""
    learning_purpose: str = "review"
    universal: UniversalAbility = field(default_factory=UniversalAbility)
    subjects: Dict[str, SubjectAbility] = field(default_factory=dict)

    def subject(self, name: str = "chemistry") -> SubjectAbility:
        if name not in self.subjects:
            self.subjects[name] = SubjectAbility(subject=name)
        return self.subjects[name]

    def get_weak_nodes(self, subject: str = "chemistry") -> List[str]:
        return self.subject(subject).weak_nodes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "grade": self.grade,
            "region": self.region,
            "exam_system": self.exam_system,
            "goals": self.goals,
            "learning_purpose": self.learning_purpose,
            "universal": self.universal.to_dict(),
            "subjects": {k: v.to_dict() for k, v in self.subjects.items()},
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "StudentModel":
        d = d or {}
        return cls(
            user_id=d.get("user_id", "local_demo"),
            grade=d.get("grade", "高二"),
            region=d.get("region", "全国卷"),
            exam_system=d.get("exam_system", "高考"),
            goals=d.get("goals", ""),
            learning_purpose=d.get("learning_purpose", "review"),
            universal=UniversalAbility.from_dict(d.get("universal")),
            subjects={
                k: SubjectAbility.from_dict(v) for k, v in (d.get("subjects", {}) or {}).items()
            },
        )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
