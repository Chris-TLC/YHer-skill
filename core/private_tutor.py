#!/usr/bin/env python3
"""
Stage 1: orchestration layer for the single-subject chemistry AI tutor.

This module does not replace the original RAG Q&A; instead it adds a
tutoring loop on top of it:
"diagnose -> schedule tasks -> execute -> recap -> update profile".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
from typing import Any, Dict, List, Optional

from core.diagnose import detect_complexity, detect_grade, diagnose_query


ABILITY_AXES = {
    "concept": "概念理解",
    "exam_entry": "题型入口识别",
    "procedure": "固定流程执行",
    "calculation": "计算与表达",
    "reading": "审题抓关键词",
    "transfer": "变式迁移",
    "metacognition": "自我判断",
}

TOPIC_LABELS = {
    "auto": "自动识别",
    "chemical_equilibrium": "化学平衡",
    "electrochemistry": "电化学",
    "redox": "氧化还原",
    "experiment": "实验综合",
    "process_flow": "工艺流程",
    "general": "高考化学综合",
}


TOPIC_LIBRARY = {
    "chemical_equilibrium": {
        "name": "化学平衡",
        "aliases": ["化学平衡", "勒夏特列", "平衡移动", "转化率", "K值", "平衡常数"],
        "prerequisites": ["物质的量", "反应速率", "可逆反应", "浓度变化"],
        "common_faults": [
            "把平衡正向移动直接等同于某反应物转化率升高",
            "分不清 K 只受温度影响，Q 会随浓度和压强变化",
            "不会用三段式把文字条件翻译成浓度变化",
        ],
        "diagnostic_questions": [
            {
                "id": "eq-q-vs-k",
                "axis": "concept",
                "prompt": "只改变某物质浓度时，平衡常数 K 会不会变？反应商 Q 会不会变？用一句话说明。",
                "look_for": "K 只受温度影响；Q 会立刻随浓度变。",
            },
            {
                "id": "eq-conversion",
                "axis": "exam_entry",
                "prompt": "向 N2 + 3H2 ⇌ 2NH3 体系中再充入 N2，H2 的转化率和 N2 的转化率分别可能怎样变化？",
                "look_for": "能区分被加入物和另一反应物的转化率变化。",
            },
            {
                "id": "eq-three-line",
                "axis": "procedure",
                "prompt": "如果题目给初始量、变化量、平衡量，你会先画什么表？表里三行分别是什么？",
                "look_for": "初始、变化、平衡三段式。",
            },
        ],
    },
    "electrochemistry": {
        "name": "电化学",
        "aliases": ["电化学", "原电池", "电解池", "电极", "阴极", "阳极", "正极", "负极"],
        "prerequisites": ["氧化还原", "离子方程式", "电子守恒", "金属活动性"],
        "common_faults": [
            "原电池正负极和电解池阴阳极混用",
            "不会先判断反应自发方向",
            "电极方程式不守恒或漏写介质",
        ],
        "diagnostic_questions": [
            {
                "id": "ec-cell-type",
                "axis": "exam_entry",
                "prompt": "看到装置图，你第一步怎么判断它是原电池还是电解池？",
                "look_for": "找外接电源；无电源看自发氧化还原。",
            },
            {
                "id": "ec-electrode",
                "axis": "concept",
                "prompt": "原电池的负极、电解池的阳极，分别发生氧化还是还原？",
                "look_for": "负极氧化，阳极氧化。",
            },
            {
                "id": "ec-equation",
                "axis": "procedure",
                "prompt": "写电极方程式时，你会按哪三步检查守恒？",
                "look_for": "元素、电荷、电子/介质守恒。",
            },
        ],
    },
    "redox": {
        "name": "氧化还原",
        "aliases": ["氧化还原", "配平", "化合价", "电子转移", "氧化剂", "还原剂"],
        "prerequisites": ["元素化合价", "离子方程式", "物质分类"],
        "common_faults": [
            "化合价找不准，导致电子转移数错",
            "先凑系数而不是先守恒电子",
            "酸碱介质下 H、O 配平顺序混乱",
        ],
        "diagnostic_questions": [
            {
                "id": "rx-valence",
                "axis": "concept",
                "prompt": "判断氧化剂、还原剂前，你第一步必须先标什么？",
                "look_for": "先标变价元素的化合价。",
            },
            {
                "id": "rx-electron",
                "axis": "procedure",
                "prompt": "氧化还原配平时，电子守恒那一步具体在守什么？",
                "look_for": "升高总数等于降低总数。",
            },
            {
                "id": "rx-medium",
                "axis": "transfer",
                "prompt": "酸性和碱性条件下，配 H、O 时常用的微粒有什么不同？",
                "look_for": "酸性常用 H+、H2O；碱性常用 OH-、H2O。",
            },
        ],
    },
    "experiment": {
        "name": "实验综合",
        "aliases": ["实验", "滴定", "定量", "定性", "装置", "误差分析", "实验综合"],
        "prerequisites": ["物质性质", "离子反应", "氧化还原", "化学计算"],
        "common_faults": [
            "看实验题只看装置，不先判断实验目的",
            "误差分析不会追踪到最终计算式",
            "现象、结论、操作之间缺因果链",
        ],
        "diagnostic_questions": [
            {
                "id": "ex-purpose",
                "axis": "reading",
                "prompt": "实验综合题开头给一堆装置，你第一眼应该先圈什么信息？",
                "look_for": "实验目的、待测量、关键试剂和控制变量。",
            },
            {
                "id": "ex-error",
                "axis": "transfer",
                "prompt": "误差分析题里，为什么不能只说'偏大偏小'，还要追到哪个表达式？",
                "look_for": "追到最终计算式或物质的量关系。",
            },
            {
                "id": "ex-causality",
                "axis": "procedure",
                "prompt": "写实验现象和结论时，怎样避免只背答案？",
                "look_for": "操作引起现象，现象支持结论。",
            },
        ],
    },
    "process_flow": {
        "name": "工艺流程",
        "aliases": ["工艺流程", "流程题", "产率", "浸出", "除杂", "沉淀", "滤液", "滤渣"],
        "prerequisites": ["离子反应", "沉淀溶解平衡", "氧化还原", "元素化合物"],
        "common_faults": [
            "不先读目标产物，导致流程方向迷路",
            "除杂和转化目的混淆",
            "产率计算漏掉纯度、转化率或守恒关系",
        ],
        "diagnostic_questions": [
            {
                "id": "pf-target",
                "axis": "reading",
                "prompt": "工艺流程题第一步应该先找原料，还是先找目标产物？为什么？",
                "look_for": "先找目标产物，再倒推每步目的。",
            },
            {
                "id": "pf-operation",
                "axis": "exam_entry",
                "prompt": "看到'调 pH 除杂'，你会立刻联想到哪类知识？",
                "look_for": "沉淀溶解平衡、离子沉淀顺序、Ksp。",
            },
            {
                "id": "pf-yield",
                "axis": "calculation",
                "prompt": "产率计算时，你会优先找哪条守恒线？",
                "look_for": "目标元素守恒或电子守恒。",
            },
        ],
    },
    "general": {
        "name": "高考化学综合",
        "aliases": [],
        "prerequisites": ["物质的量", "离子反应", "氧化还原", "元素化合物"],
        "common_faults": [
            "只问答案，不先定位知识点和题型入口",
            "不会把文字题翻译成守恒、方程或图像",
            "错因复盘停留在'粗心'，没有追到具体动作",
        ],
        "diagnostic_questions": [
            {
                "id": "gen-topic",
                "axis": "metacognition",
                "prompt": "你现在最想补的是概念听不懂、题型不会进、还是做题老算错？选一个并举例。",
                "look_for": "能说出具体症状，而不是只说不会。",
            },
            {
                "id": "gen-process",
                "axis": "procedure",
                "prompt": "遇到一道不会的化学题，你通常前三步会做什么？",
                "look_for": "圈关键词、判知识点/题型、列关系。",
            },
            {
                "id": "gen-review",
                "axis": "metacognition",
                "prompt": "最近一次化学错题，你觉得真正错因是什么？不要写'粗心'。",
                "look_for": "能定位到概念、审题、流程、计算或迁移。",
            },
        ],
    },
}


PROGRESSIVE_DIAGNOSTIC_BANK = {
    "chemical_equilibrium": [
        {
            "id": "eq-l1-k-q",
            "level": "L1 基础概念",
            "axis": "concept",
            "prompt": "只升高温度、只加 N2、只加催化剂，这三种情况里，平衡常数 K 分别会不会变？",
            "look_for": "K 只随温度变；浓度、压强、催化剂不改变 K。",
            "advance_if": "能明确区分 K 和 Q。",
        },
        {
            "id": "eq-l2-q-direction",
            "level": "L2 入口判断",
            "axis": "exam_entry",
            "prompt": "对于 N2(g)+3H2(g)⇌2NH3(g)，恒温恒容下加入 N2。请用 Q 和 K 判断平衡移动方向。",
            "look_for": "加入 N2 后 Q 变小，Q<K，平衡正向移动。",
            "advance_if": "能用 Q/K，而不是只背勒夏特列。",
        },
        {
            "id": "eq-l3-conversion-trap",
            "level": "L3 应试坑点",
            "axis": "transfer",
            "prompt": "同样加入 N2，H2 的转化率和 N2 的转化率一定都升高吗？分别说明原因。",
            "look_for": "H2 转化率升高；N2 因分母加入而不一定升高，常见结论是降低。",
            "advance_if": "能分清平衡移动、物质的量、转化率分母。",
        },
        {
            "id": "eq-l4-data",
            "level": "L4 全国卷风格原创诊断",
            "axis": "procedure",
            "prompt": "恒容 2 L 容器中充入 1 mol N2 和 3 mol H2，平衡时 NH3 为 1 mol。若再充入 1 mol N2，请写出第二次平衡前 Q 如何变化，并说出下一步三段式该列哪些量。",
            "look_for": "先算/比较 Q 变小，再列初始、变化、平衡；不要求完整求解但要有三段式意识。",
            "advance_if": "能把文字条件翻译成三段式。",
        },
    ],
    "electrochemistry": [
        {
            "id": "ec-l1-oxidation",
            "level": "L1 基础概念",
            "axis": "concept",
            "prompt": "原电池负极、电解池阳极分别发生氧化还是还原？用一句话说明记忆依据。",
            "look_for": "负极氧化，阳极氧化；失电子是氧化。",
            "advance_if": "正负极/阴阳极不混。",
        },
        {
            "id": "ec-l2-cell-type",
            "level": "L2 入口判断",
            "axis": "exam_entry",
            "prompt": "看到一个电化学装置图，你第一步看什么来判断原电池还是电解池？第二步看什么判断电极反应？",
            "look_for": "先看外接电源，再看自发氧化还原/离子放电顺序。",
            "advance_if": "能形成判题顺序。",
        },
        {
            "id": "ec-l3-electrode-equation",
            "level": "L3 应试坑点",
            "axis": "procedure",
            "prompt": "酸性介质中写 MnO4- → Mn2+ 的电极反应，你会按什么顺序补 H、O、电荷和电子？",
            "look_for": "补 H2O、H+，再用电子配电荷；最终 MnO4-+8H++5e-=Mn2++4H2O。",
            "advance_if": "能用守恒写半反应。",
        },
        {
            "id": "ec-l4-data",
            "level": "L4 全国卷风格原创诊断",
            "axis": "transfer",
            "prompt": "惰性电极电解 CuSO4 溶液一段时间后，两极产物如何判断？如果溶液 pH 变化，你会追哪条守恒线？",
            "look_for": "阴极 Cu2+ 得电子，阳极水/OH- 失电子放 O2；追电子守恒和溶液离子变化。",
            "advance_if": "能综合电极、产物、溶液变化。",
        },
    ],
    "redox": [
        {
            "id": "rx-l1-valence",
            "level": "L1 基础概念",
            "axis": "concept",
            "prompt": "判断氧化剂和还原剂之前，第一步必须标什么？升价和降价分别对应什么过程？",
            "look_for": "标化合价；升价失电子被氧化，降价得电子被还原。",
            "advance_if": "能稳定判断升降价。",
        },
        {
            "id": "rx-l2-balance",
            "level": "L2 入口判断",
            "axis": "procedure",
            "prompt": "配平氧化还原方程时，为什么不能先硬凑系数？电子守恒那一步到底守的是什么？",
            "look_for": "升高总价数=降低总价数/失电子数=得电子数。",
            "advance_if": "知道先守电子。",
        },
        {
            "id": "rx-l3-medium",
            "level": "L3 应试坑点",
            "axis": "transfer",
            "prompt": "酸性条件和碱性条件下补 H、O 时常用哪些粒子？如果题目给 OH-，你还会不会直接加 H+？",
            "look_for": "酸性用 H+/H2O，碱性用 OH-/H2O；介质不能乱用。",
            "advance_if": "能处理介质限制。",
        },
        {
            "id": "rx-l4-data",
            "level": "L4 全国卷风格原创诊断",
            "axis": "calculation",
            "prompt": "某反应中 Fe2+ 被氧化为 Fe3+，Cr2O7^2- 被还原为 Cr3+。请先不完整配平，只写电子转移倍数关系。",
            "look_for": "Fe2+ 每个失 1e-，Cr2O7^2- 中 2 个 Cr 从 +6 到 +3 共得 6e-，Fe2+ 系数配 6。",
            "advance_if": "能抓住电子倍数。",
        },
    ],
    "experiment": [
        {
            "id": "ex-l1-purpose",
            "level": "L1 基础入口",
            "axis": "reading",
            "prompt": "实验综合题给装置和一堆试剂时，第一眼先找实验目的、反应原理，还是先看每个仪器名称？为什么？",
            "look_for": "先找目的和原理，仪器服务于目的。",
            "advance_if": "不被装置细节带跑。",
        },
        {
            "id": "ex-l2-control",
            "level": "L2 操作逻辑",
            "axis": "procedure",
            "prompt": "如果题目问'为什么要先通 N2 一段时间'，你会从哪个角度答？",
            "look_for": "排空气/氧气/水蒸气，防止干扰或氧化。",
            "advance_if": "能把操作和干扰因素连接。",
        },
        {
            "id": "ex-l3-error",
            "level": "L3 应试坑点",
            "axis": "transfer",
            "prompt": "滴定误差分析里，锥形瓶洗后有少量蒸馏水，会不会影响待测浓度？为什么不能只凭直觉判断？",
            "look_for": "不影响待测物质的量；要追最终计算式。",
            "advance_if": "能追公式而非凭感觉。",
        },
        {
            "id": "ex-l4-data",
            "level": "L4 上海/全国卷风格原创诊断",
            "axis": "calculation",
            "prompt": "用标准酸滴定未知碱，若滴定管尖嘴开始有气泡、终点时气泡消失，测得碱浓度偏大还是偏小？请追到 V标 的变化。",
            "look_for": "读出的标准液体积偏大，计算出的待测浓度偏大。",
            "advance_if": "能把操作误差传到数据变量。",
        },
    ],
    "process_flow": [
        {
            "id": "pf-l1-target",
            "level": "L1 基础入口",
            "axis": "reading",
            "prompt": "工艺流程题第一眼先找原料还是目标产物？找到以后下一步做什么？",
            "look_for": "先找目标产物，再倒推每步目的和元素去向。",
            "advance_if": "能建立流程方向。",
        },
        {
            "id": "pf-l2-operation",
            "level": "L2 操作目的",
            "axis": "exam_entry",
            "prompt": "看到'调 pH 使 Fe3+ 沉淀而 Mg2+ 不沉淀'，你立刻联想到哪个知识点？怎么判断 pH 范围？",
            "look_for": "Ksp/沉淀溶解平衡；用离子浓度和 Ksp 判断。",
            "advance_if": "能把操作对应知识点。",
        },
        {
            "id": "pf-l3-yield",
            "level": "L3 应试坑点",
            "axis": "calculation",
            "prompt": "产率计算遇到原料纯度、转化率、产品质量时，你先抓哪条守恒线？",
            "look_for": "目标元素守恒；逐层扣纯度、转化率、摩尔质量。",
            "advance_if": "能建立计算路线。",
        },
        {
            "id": "pf-l4-data",
            "level": "L4 全国卷风格原创诊断",
            "axis": "transfer",
            "prompt": "某矿样含 NiO，流程经酸浸、除 Fe、沉 NiCO3、煅烧得 NiO。若问'除 Fe 前为什么先氧化 Fe2+'，你怎么答到得分点？",
            "look_for": "Fe2+ 氧化为 Fe3+，更易水解/沉淀除去，且控制 pH 不损失目标离子。",
            "advance_if": "能说出操作目的和选择性。",
        },
    ],
    "general": [
        {
            "id": "gen-l1-self",
            "level": "L1 自我定位",
            "axis": "metacognition",
            "prompt": "你最近化学错题最常见的是概念不会、题型入口不会、步骤乱、计算错，还是看答案懂自己做不出？举一个例子。",
            "look_for": "能把不会具体化。",
            "advance_if": "能定位症状。",
        },
        {
            "id": "gen-l2-entry",
            "level": "L2 入口判断",
            "axis": "exam_entry",
            "prompt": "遇到一道陌生化学题，你前三个动作是什么？请按顺序写。",
            "look_for": "圈关键词、判知识点/题型、列关系/守恒。",
            "advance_if": "有稳定开题流程。",
        },
        {
            "id": "gen-l3-exam",
            "level": "L3 应试策略",
            "axis": "reading",
            "prompt": "如果一道题文字很多，你如何判断哪些条件是核心条件，哪些只是背景？",
            "look_for": "围绕设问、目标产物、变量和数据关系筛选。",
            "advance_if": "能按设问筛条件。",
        },
        {
            "id": "gen-l4-transfer",
            "level": "L4 综合迁移",
            "axis": "transfer",
            "prompt": "请说一个你'听懂了但换题不会'的例子。你觉得换题后变的是外壳，还是核心关系也变了？",
            "look_for": "能区分题目情境和核心模型。",
            "advance_if": "具备迁移复盘意识。",
        },
    ],
}


@dataclass
class StudentProfile:
    """Minimal student profile for stage 1."""

    user_id: str = "local_demo"
    grade: str = "高二"
    region: str = "全国卷"
    exam_system: str = "高考"
    topic_id: str = "auto"
    exam_style: str = "全国卷风格"
    current_score: Optional[float] = None
    target_score: Optional[float] = None
    school_level: str = ""
    goal: str = "补化学薄弱点"
    time_budget_min: int = 180
    learning_preference: str = "先诊断再讲解"
    fatigue_level: str = "正常"
    known_weaknesses: List[str] = field(default_factory=list)

    @property
    def score_gap(self) -> Optional[float]:
        if self.current_score is None or self.target_score is None:
            return None
        return max(self.target_score - self.current_score, 0)


@dataclass
class TutorTask:
    """An executable task within a single managed learning session."""

    task_id: str
    title: str
    phase: str
    duration_min: int
    teaching_mode: str
    objective: str
    student_action: str
    success_criteria: List[str]
    tutor_instruction: str
    resource_query: str = ""
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"
    mastery_gate: float = 0.72
    max_extend_min: int = 15


@dataclass
class TutorSession:
    """State of the stage 1 private-tutor learning cabin."""

    session_id: str
    subject: str
    goal: str
    topic_id: str
    topic_name: str
    profile: StudentProfile
    diagnosis: Dict[str, Any]
    ability_hypothesis: Dict[str, Any]
    diagnostic_questions: List[Dict[str, Any]]
    tasks: List[TutorTask]
    recommended_videos: List[Dict[str, Any]]
    created_at: str
    current_task_id: str = "T1"
    evidence_log: List[Dict[str, Any]] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)


class ChemistryPrivateTutor:
    """Orchestrator for the single-subject chemistry private tutor."""

    def __init__(self, retriever=None):
        self.retriever = retriever

    def prepare_session(self, profile: StudentProfile) -> TutorSession:
        """Build a diagnostic package + managed plan from the student profile and goal."""
        goal = profile.goal.strip() or "补化学薄弱点"
        diagnosis = self._diagnose(goal, profile)
        topic_id = profile.topic_id if profile.topic_id in TOPIC_LIBRARY else "auto"
        if topic_id == "auto":
            topic_id = self._match_topic(goal, diagnosis)
        topic = TOPIC_LIBRARY[topic_id]

        ability_hypothesis = self._build_ability_hypothesis(profile, diagnosis, topic)
        diagnostic_questions = self._build_diagnostic_questions(profile, topic, diagnosis)
        recommended_videos = self._collect_videos(diagnosis)
        tasks = self._build_task_queue(profile, topic, diagnosis)

        session_id = self._make_session_id(profile.user_id, goal)
        return TutorSession(
            session_id=session_id,
            subject="chemistry",
            goal=goal,
            topic_id=topic_id,
            topic_name=topic["name"],
            profile=profile,
            diagnosis=diagnosis,
            ability_hypothesis=ability_hypothesis,
            diagnostic_questions=diagnostic_questions,
            tasks=tasks,
            recommended_videos=recommended_videos,
            created_at=datetime.now().isoformat(timespec="seconds"),
            current_task_id=tasks[0].task_id if tasks else "T1",
            risk_flags=self._infer_risk_flags(profile, diagnosis),
        )

    def estimate_answer_snapshot(
        self,
        session: TutorSession,
        answers: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Fallback heuristic scoring when no LLM is available.

        This is not the final grading; it only lets the demo produce a
        structured profile even without an API. In production, grading should
        be done jointly by the LLM and the item bank's standard answers.
        """
        axis_scores = {axis: 3 for axis in ABILITY_AXES}
        evidence = []
        for question in session.diagnostic_questions:
            qid = question["id"]
            axis = question.get("axis", "metacognition")
            answer = (answers.get(qid) or "").strip()
            if not answer:
                axis_scores[axis] = min(axis_scores[axis], 1)
                evidence.append(f"{qid}: 未作答")
                continue
            weak_terms = ["不会", "不懂", "蒙", "乱", "不知道", "不清楚", "忘了"]
            strong_terms = ["因为", "所以", "先", "再", "守恒", "判断", "本质"]
            weak_hit = any(t in answer for t in weak_terms)
            strong_hit = sum(1 for t in strong_terms if t in answer)
            if len(answer) < 12 or weak_hit:
                axis_scores[axis] = min(axis_scores[axis], 2)
            elif strong_hit >= 2 and len(answer) >= 28:
                axis_scores[axis] = max(axis_scores[axis], 4)
            evidence.append(f"{qid}: {answer[:40]}")

        weakest = sorted(axis_scores.items(), key=lambda x: x[1])[:3]
        return {
            "axis_scores": axis_scores,
            "weakest_axes": [
                {"axis": axis, "name": ABILITY_AXES[axis], "score": score}
                for axis, score in weakest
            ],
            "evidence": evidence,
            "confidence": "low_without_llm",
            "next_action": self._choose_next_action(weakest[0][0] if weakest else "concept"),
        }

    def first_question(self, session: TutorSession) -> Dict[str, Any]:
        """Return the first question of the progressive diagnosis."""
        if session.diagnostic_questions:
            return session.diagnostic_questions[0]
        return PROGRESSIVE_DIAGNOSTIC_BANK["general"][0]

    def builtin_next_question(
        self,
        session: TutorSession,
        answered_count: int,
        decision: str = "ask_next",
    ) -> Optional[Dict[str, Any]]:
        """Next-question strategy when there is no LLM or JSON parsing fails."""
        questions = session.diagnostic_questions
        if not questions:
            return None

        if decision in ("finish", "finish_to_execution"):
            return None
        if decision in ("ask_easier", "ask_same"):
            idx = min(max(answered_count - 1, 0), len(questions) - 1)
            return questions[idx]
        idx = min(answered_count, len(questions) - 1)
        if answered_count >= len(questions):
            return None
        return questions[idx]

    def task_index(self, session: TutorSession, task_id: str) -> int:
        for idx, task in enumerate(session.tasks):
            if task.task_id == task_id:
                return idx
        return 0

    def next_task_id(self, session: TutorSession, task_id: str) -> str:
        idx = self.task_index(session, task_id)
        if idx + 1 < len(session.tasks):
            return session.tasks[idx + 1].task_id
        return session.tasks[-1].task_id

    def previous_task_id(self, session: TutorSession, task_id: str) -> str:
        idx = self.task_index(session, task_id)
        if idx > 0:
            return session.tasks[idx - 1].task_id
        return session.tasks[0].task_id

    def session_to_dict(self, session: TutorSession) -> Dict[str, Any]:
        return asdict(session)

    def plan_markdown(self, session: TutorSession) -> str:
        """Render the learning-cabin plan as Markdown for CLI/Web display."""
        lines = [
            f"### {session.topic_name} Private Tutoring Session",
            f"- Goal: {session.goal}",
            f"- Duration: {sum(t.duration_min for t in session.tasks)} min",
            f"- Initial assessment: {session.ability_hypothesis.get('summary', '')}",
            "",
            "#### Diagnostic Questions",
        ]
        for i, q in enumerate(session.diagnostic_questions, 1):
            axis_name = ABILITY_AXES.get(q.get("axis", ""), q.get("axis", ""))
            lines.append(f"{i}. [{axis_name}] {q['prompt']}")

        lines.extend(["", "#### Managed Task Queue"])
        for task in session.tasks:
            criteria = "；".join(task.success_criteria[:2])
            lines.append(
                f"- {task.task_id} | {task.duration_min} min | {task.title}: "
                f"{task.objective} (pass: {criteria})"
            )

        if session.recommended_videos:
            lines.extend(["", "#### Recommended Videos"])
            for video in session.recommended_videos[:3]:
                bv = video.get("bv", "")
                pn = video.get("p_number", 1)
                title = video.get("short_title") or video.get("video_title", "")
                collection = video.get("collection", "Other videos")
                lines.append(
                    f"- 【{collection}】P{pn}：{title} "
                    f"https://www.bilibili.com/video/{bv}?p={pn}"
                )
        return "\n".join(lines)

    def _diagnose(self, goal: str, profile: StudentProfile) -> Dict[str, Any]:
        query = f"{profile.grade} {profile.region} {goal}"
        if self.retriever is None:
            return {
                "grade_signal": detect_grade(query),
                "complexity": detect_complexity(query),
                "related_nodes": [],
                "missing_prereqs": [],
                "exam_patterns": [],
                "thinking_names": [],
                "chunks": [],
                "recommended_videos": [],
                "elapsed_ms": 0,
            }
        return diagnose_query(query, self.retriever)

    def _match_topic(self, goal: str, diagnosis: Dict[str, Any]) -> str:
        text = " ".join([
            goal,
            " ".join(diagnosis.get("related_nodes", [])),
            " ".join(diagnosis.get("missing_prereqs", [])),
            " ".join(diagnosis.get("exam_patterns", [])),
        ])
        scores = {}
        for topic_id, topic in TOPIC_LIBRARY.items():
            if topic_id == "general":
                continue
            scores[topic_id] = sum(1 for alias in topic["aliases"] if alias and alias in text)
        best_topic, best_score = max(scores.items(), key=lambda x: x[1])
        return best_topic if best_score > 0 else "general"

    def _build_ability_hypothesis(
        self,
        profile: StudentProfile,
        diagnosis: Dict[str, Any],
        topic: Dict[str, Any],
    ) -> Dict[str, Any]:
        missing = diagnosis.get("missing_prereqs") or topic["prerequisites"][:2]
        thinking = diagnosis.get("thinking_names") or ["根因式诊断", "固定流程", "审题关键词"]

        axis_priority = ["concept", "exam_entry", "procedure"]
        if any(k in profile.goal for k in ["算", "计算", "产率"]):
            axis_priority.insert(0, "calculation")
        if any(k in profile.goal for k in ["看不懂", "审题", "流程", "实验"]):
            axis_priority.insert(0, "reading")
        if profile.score_gap and profile.score_gap >= 20:
            axis_priority.append("metacognition")

        summary = (
            f"先假设卡点在 {topic['name']} 的"
            f"{'、'.join(ABILITY_AXES[a] for a in axis_priority[:3])}，"
            f"优先检查前置：{'、'.join(missing[:3])}。"
        )
        return {
            "summary": summary,
            "suspected_missing_prereqs": missing[:5],
            "priority_axes": axis_priority[:4],
            "recommended_thinking": thinking[:4],
            "score_gap": profile.score_gap,
        }

    def _build_diagnostic_questions(
        self,
        profile: StudentProfile,
        topic: Dict[str, Any],
        diagnosis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        questions = []
        questions.append({
            "id": "self-report",
            "axis": "metacognition",
            "prompt": (
                "你觉得自己现在最卡的是：概念听不懂、题型不会进、"
                "步骤不稳定、计算容易错，还是看答案懂自己做不出？举一个最近的例子。"
            ),
            "look_for": "学生能否把'不会'具体化。",
        })
        topic_id = self._topic_id_by_name(topic.get("name", ""))
        questions.extend(PROGRESSIVE_DIAGNOSTIC_BANK.get(topic_id, topic["diagnostic_questions"]))

        prereqs = diagnosis.get("missing_prereqs") or topic["prerequisites"][:2]
        if prereqs:
            questions.append({
                "id": "prereq-check",
                "axis": "concept",
                "prompt": f"这节课要先排查 {'、'.join(prereqs[:3])}。你分别给自己打几分？0=完全不会，5=能讲给别人。",
                "look_for": "前置知识自评是否可信。",
            })

        questions.append({
            "id": "time-energy",
            "axis": "metacognition",
            "prompt": f"今天你有 {profile.time_budget_min} 分钟，当前精神状态如何？想先看讲解、先做题，还是先被追问？",
            "look_for": "确定节奏和教学入口。",
        })
        return questions[:8]

    def _build_task_queue(
        self,
        profile: StudentProfile,
        topic: Dict[str, Any],
        diagnosis: Dict[str, Any],
    ) -> List[TutorTask]:
        total = max(45, min(int(profile.time_budget_min or 180), 360))
        phases = [
            ("T1", "诊断校准", "diagnosis", 0.10),
            ("T2", "根因补洞", "root_patch", 0.20),
            ("T3", "题型入口训练", "guided_drill", 0.20),
            ("T4", "视频定点巩固", "video_checkpoint", 0.12),
            ("T5", "变式迁移训练", "adaptive_practice", 0.25),
            ("T6", "错因复盘与下次计划", "recap", 0.13),
        ]
        durations = self._allocate_minutes(total, [p[3] for p in phases])
        missing = diagnosis.get("missing_prereqs") or topic["prerequisites"][:3]
        common_faults = topic.get("common_faults", [])

        return [
            TutorTask(
                task_id="T1",
                title="诊断校准",
                phase="diagnosis",
                duration_min=durations[0],
                teaching_mode="追问 + 小测",
                objective="把'不会'压缩成可验证的 1-3 个真实卡点。",
                student_action="回答诊断题，不会也要写出自己卡住的那一步。",
                success_criteria=["能定位至少 1 个真实错因", "能确认今天优先补哪一层"],
                tutor_instruction="不要急着讲题，先用学生答案判断卡点属于概念、入口、流程、计算还是迁移。",
                mastery_gate=0.60,
            ),
            TutorTask(
                task_id="T2",
                title="根因补洞",
                phase="root_patch",
                duration_min=durations[1],
                teaching_mode="短讲解 + 反问确认",
                objective=f"补齐 {topic['name']} 的关键前置：{'、'.join(missing[:3])}。",
                student_action="听一个最短解释，然后用自己的话复述核心判断。",
                success_criteria=["能复述核心概念", "能回答 2 个反向追问"],
                tutor_instruction="每次只补一个根因；讲完必须追问，不允许连续大段灌输。",
                depends_on=["T1"],
                mastery_gate=0.68,
            ),
            TutorTask(
                task_id="T3",
                title="题型入口训练",
                phase="guided_drill",
                duration_min=durations[2],
                teaching_mode="例题拆解 + 固定流程",
                objective=f"把 {topic['name']} 的题型入口变成可重复动作。",
                student_action="按 AI 给的步骤拆一道典型题。",
                success_criteria=["能说出第一眼看什么", "能按固定流程走完一题"],
                tutor_instruction="先让学生圈关键词，再给流程；如果学生跳步，立刻拉回第一步。",
                resource_query=f"{topic['name']} 题型 固定流程",
                depends_on=["T2"],
                mastery_gate=0.72,
            ),
            TutorTask(
                task_id="T4",
                title="视频定点巩固",
                phase="video_checkpoint",
                duration_min=durations[3],
                teaching_mode="视频片段 + 观看任务",
                objective="用真实老师视频巩固刚补过的根因，而不是泛泛看网课。",
                student_action="看指定视频片段，并带着一个完成标准回来。",
                success_criteria=["能复述视频里的关键判断", "能说出它解决了哪个错因"],
                tutor_instruction="只能推荐检索得到的真实 BV+P；必须给观看目的和完成标准。",
                resource_query=f"{topic['name']} 一化儿 视频",
                depends_on=["T2"],
                mastery_gate=0.65,
            ),
            TutorTask(
                task_id="T5",
                title="变式迁移训练",
                phase="adaptive_practice",
                duration_min=durations[4],
                teaching_mode="变式题 + 即时批改",
                objective=f"针对高频坑训练迁移：{'；'.join(common_faults[:2])}。",
                student_action="连续做 2-4 道变式题，每题写出判断依据。",
                success_criteria=["同类题正确率达到 70%+", "能解释每个选项或每步计算"],
                tutor_instruction="根据学生上一题表现调难度；错了先判错因，再决定补概念还是换题。",
                resource_query=f"{topic['name']} 变式 易错",
                depends_on=["T3", "T4"],
                # Historical planner only; canonical student sessions use engine.planner.
                mastery_gate=0.75,
            ),
            TutorTask(
                task_id="T6",
                title="错因复盘与下次计划",
                phase="recap",
                duration_min=durations[5],
                teaching_mode="复盘 + 学生档案更新",
                objective="生成本次能力变化、残留弱点和下一次学习任务。",
                student_action="确认哪些地方真的会了，哪些还需要继续练。",
                success_criteria=["形成 3 条错因档案", "给出下一次可执行任务"],
                tutor_instruction="把复盘写成学生画像更新，不要只写鼓励语。",
                depends_on=["T5"],
                mastery_gate=0.70,
            ),
        ]

    def _topic_id_by_name(self, topic_name: str) -> str:
        for topic_id, topic in TOPIC_LIBRARY.items():
            if topic.get("name") == topic_name:
                return topic_id
        return "general"

    def _collect_videos(self, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        videos = []
        seen = set()
        for video in diagnosis.get("recommended_videos", []):
            bv = video.get("bv", "")
            pn = video.get("p_number", 1)
            key = (bv, pn)
            if not bv or key in seen:
                continue
            seen.add(key)
            videos.append(video)
            if len(videos) >= 5:
                break
        return videos

    def _infer_risk_flags(
        self,
        profile: StudentProfile,
        diagnosis: Dict[str, Any],
    ) -> List[str]:
        flags = []
        if profile.time_budget_min >= 240:
            flags.append("long_session_fatigue")
        if profile.score_gap and profile.score_gap >= 25:
            flags.append("large_score_gap")
        if diagnosis.get("complexity") in ("complex", "diagnostic"):
            flags.append("needs_root_cause_check")
        if not diagnosis.get("recommended_videos"):
            flags.append("video_evidence_limited")
        return flags

    def _choose_next_action(self, weakest_axis: str) -> str:
        return {
            "concept": "先补一个最小概念，再立刻反问确认。",
            "exam_entry": "先训练第一眼看题入口，不急着完整解题。",
            "procedure": "给固定流程，让学生按步骤复述并做一题。",
            "calculation": "把计算拆成守恒线和代入线，先查式子再查算术。",
            "reading": "先圈关键词和题目目的，再决定用哪个知识点。",
            "transfer": "用同核不同壳的变式题训练迁移。",
            "metacognition": "先让学生把'不会'具体化为一个可观察错误。",
        }.get(weakest_axis, "先追问确认真实卡点。")

    def _allocate_minutes(self, total: int, ratios: List[float]) -> List[int]:
        mins = [max(5, round(total * r / 5) * 5) for r in ratios]
        diff = total - sum(mins)
        mins[-1] += diff
        if mins[-1] < 5:
            deficit = 5 - mins[-1]
            mins[-1] = 5
            for i in range(len(mins) - 2, -1, -1):
                take = min(deficit, max(0, mins[i] - 5))
                mins[i] -= take
                deficit -= take
                if deficit == 0:
                    break
        return mins

    def _make_session_id(self, user_id: str, goal: str) -> str:
        raw = f"{user_id}|{goal}|{datetime.now().isoformat(timespec='seconds')}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"stage1_{digest}"
