#!/usr/bin/env python3
"""
engine/memory.py — 三层长期记忆 L2/L3（诊断引擎设计文档块 3，2026-07-07）
==========================================================================
规格源（唯一权威）：mac-unknown-design-20260707-230853.md 块 3（第 147-185 行）。
L1（状态层=掌握度+FSRS 衰减）已在 mastery.py；本模块管 L2 情节记忆 + L3 叙事蒸馏。

Hermes "定期提炼 skill" 模式的移植：一年后像真人不是靠塞一年上下文,
是靠 ≤500 字蒸馏物替代上下文（token 恒定）。

边界（纯函数,零新基础设施）：
  · L2 准入五类规则写死,不靠 LLM 拍脑袋（第 159-163 行）。
  · 检索打分三项写死 w_r/w_s/w_v（第 167 行）;BGE-M3 相似度做成**可注入函数**
    （测试传假向量,生产接真 BGE-M3+sqlite-vec）,本模块不碰向量库。
  · L3 蒸馏本模块只判**触发**（懒触发,>7 天+新 L2）,LLM 调用归接线层。
  · 反人机闸门凌驾一切（第 183-185 行）:每 session 开场至多引用 1 条,
    仅当今日诊断触到有旧事件的节点才允许引用。
  · **#7 自证回路隔离(红线)**:L3 档案只喂表达类触点,禁入证据类触点(追问判定器)。
    本模块提供 injectable_for(touchpoint) 显式区分,判定器触点永远返回空。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from engine import mastery as m

# ── L2 准入五类（与 study_schema.L2_EVENT_TYPES 同源,此处复述为判定用）──
L2_TYPES = ("belief_flip", "breakthrough", "stubborn_error",
            "high_info_followup", "high_efficacy_watch")

# ── 准入阈值（第 159-162 行,模块常量）──
BELIEF_FLIP_UP = 0.7              # P(M) 首过 0.7
BELIEF_FLIP_DOWN = 0.5           # P(M) 跌破 0.5
STUBBORN_ERROR_COUNT = 3         # 同错因码第 ≥3 次
HIGH_INFO_CONF = 0.7             # 追问判定器置信度 ≥0.7 才入 L2（低置信只留日志）
HIGH_EFFICACY_L1_DELTA = 0.3     # 看完段后单次贝叶斯更新 ‖Δb‖₁ ≥ 0.3

# ── 检索打分权重（第 167-168 行,模块常量）──
W_RECENCY, W_SALIENCE, W_VECTOR = 0.4, 0.3, 0.3
RECENCY_TAU_DAYS = 30.0
SALIENCE_WEIGHT = {                # 显著性权重按类型查表（第 168-169 行）
    "breakthrough": 1.0, "stubborn_error": 1.0,
    "belief_flip": 0.8, "high_info_followup": 0.9, "high_efficacy_watch": 0.6,
}

# ── L3 蒸馏触发（第 175 行,写死）──
DISTILL_INTERVAL_DAYS = 7.0

# ── 反人机闸门（第 183-184 行,硬规则）──
MAX_REFERENCES_PER_SESSION = 1   # 每 session 开场至多引用 1 条

# ── 触点分类（#7 自证回路隔离,第 104 行红线）──
EXPRESSION_TOUCHPOINTS = frozenset({"opening", "report"})   # 表达类:可注入 L3
EVIDENCE_TOUCHPOINTS = frozenset({"followup_triage"})       # 证据类:禁注入 L3


# ══════════════════════════════════════════════════════════════════════
# L2 事件（内存表示;落盘用 study_schema.l2_event_row）
# ══════════════════════════════════════════════════════════════════════
@dataclass
class L2Event:
    """L2 情节记忆事件。vector 由接线层用 BGE-M3(摘要+节点名) 预填,本模块只消费。"""
    ts: float
    node: str
    event_type: str
    summary: str
    related_event_id: Optional[str] = None
    vector: Optional[Sequence[float]] = None
    subject: str = "chemistry"


# ══════════════════════════════════════════════════════════════════════
# L2 准入判定（五类规则写死;返回 event_type 或 None）
# ══════════════════════════════════════════════════════════════════════
def admit_belief_flip(pm_before: float, pm_after: float) -> bool:
    """信念翻转:P(M) 首过 0.7 或跌破 0.5。"""
    crossed_up = pm_before < BELIEF_FLIP_UP <= pm_after
    crossed_down = pm_before >= BELIEF_FLIP_DOWN > pm_after
    return crossed_up or crossed_down


def admit_stubborn_error(error_code_count: int) -> bool:
    """顽固错误:同错因码累计第 ≥3 次。"""
    return error_code_count >= STUBBORN_ERROR_COUNT


def admit_high_info_followup(confidence: float) -> bool:
    """高信息追问:判定器置信度 ≥0.7 才入 L2（低置信只留事件日志,防低信号灌满）。"""
    return confidence >= HIGH_INFO_CONF


def admit_high_efficacy_watch(b_before: Sequence[float], b_after: Sequence[float]) -> bool:
    """高疗效观看:看完段后单次贝叶斯更新 ‖Δb‖₁ ≥ 0.3。"""
    a = [float(x) for x in b_before]
    c = [float(x) for x in b_after]
    if len(a) != len(c):
        raise ValueError("belief 向量维度必须一致")
    l1 = sum(abs(ai - ci) for ai, ci in zip(a, c))
    return l1 >= HIGH_EFFICACY_L1_DELTA


def classify_admission(*, pm_before: Optional[float] = None, pm_after: Optional[float] = None,
                       held_out_passed: bool = False, error_code_count: Optional[int] = None,
                       followup_confidence: Optional[float] = None,
                       b_before: Optional[Sequence[float]] = None,
                       b_after: Optional[Sequence[float]] = None) -> Optional[str]:
    """把一次事件按五类准入规则判定,返回准入的 event_type 或 None（不准入）。
    优先级:突破 > 信念翻转 > 顽固错误 > 高疗效观看 > 高信息追问（同事件多条命中取最强显著性）。"""
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
# L2 检索打分（"人性化注意力"函数形式,各项写死;第 165-170 行）
# ══════════════════════════════════════════════════════════════════════
def _cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """默认余弦相似度。向量缺失 → 0（向量项零贡献,退化为 recency+salience）。"""
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
    """score = w_r·exp(−Δt/30天) + w_s·显著性权重 + w_v·cos(事件向量, 今日上下文向量)。
    sim_fn 可注入（生产接真 BGE-M3;默认余弦供测试）。"""
    dt_days = max(0.0, (now - event.ts) / m.DAY_SECONDS)
    recency = math.exp(-dt_days / RECENCY_TAU_DAYS)
    salience = SALIENCE_WEIGHT.get(event.event_type, 0.5)
    vec = sim_fn(event.vector, context_vector)
    return W_RECENCY * recency + W_SALIENCE * salience + W_VECTOR * vec


def rank_events(events: Sequence[L2Event], now: float,
                context_vector: Optional[Sequence[float]] = None,
                sim_fn: Callable = _cosine) -> List[L2Event]:
    """按检索分降序;同分按时间新者优先（确定性）。"""
    scored = [(retrieval_score(e, now, context_vector, sim_fn), -e.ts, i, e)
              for i, e in enumerate(events)]
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [e for *_, e in scored]


# ══════════════════════════════════════════════════════════════════════
# 反人机闸门（凌驾一切;存在与表达分离,第 183-185 行）
# ══════════════════════════════════════════════════════════════════════
def select_references(events: Sequence[L2Event], today_nodes: Sequence[str], now: float,
                      context_vector: Optional[Sequence[float]] = None,
                      sim_fn: Callable = _cosine,
                      max_refs: int = MAX_REFERENCES_PER_SESSION) -> List[L2Event]:
    """开场可引用的 L2 事件:**仅当今日诊断触到有旧事件的节点**才允许引用,
    每 session 至多 max_refs 条。表达必须服务当下动作,禁止无目的寒暄。
    L1/L2/L3 每 session 都在算,上屏受此闸门约束（存在≠表达）。"""
    today = set(today_nodes)
    eligible = [e for e in events if e.node in today]     # 只引用今日触到的节点
    ranked = rank_events(eligible, now, context_vector, sim_fn)
    limit = min(MAX_REFERENCES_PER_SESSION, max(0, int(max_refs)))
    return ranked[:limit]


# ══════════════════════════════════════════════════════════════════════
# L3 蒸馏触发（懒触发;本模块只判触发,LLM 调用归接线层;第 175 行）
# ══════════════════════════════════════════════════════════════════════
def should_distill(last_distill_at: Optional[float], now: float,
                   new_l2_since_last: int) -> bool:
    """session 开场懒触发:距上次蒸馏 >7 天 且 期间有新 L2 事件 → 触发。
    从未蒸馏过（last_distill_at=None）且有新 L2 → 触发。"""
    if new_l2_since_last <= 0:
        return False
    if last_distill_at is None:
        return True
    return (now - last_distill_at) / m.DAY_SECONDS > DISTILL_INTERVAL_DAYS


def distill_inputs(l2_events: Sequence[L2Event], belief_deltas: Dict[str, float],
                   prev_profile: Optional[Dict]) -> Dict:
    """组装蒸馏输入（LLM 读【上周 L2 + L1 信念变化量 + 上一版叙事】→ ≤500 字档案）。
    本模块只组装输入,不调 LLM;LLM 输出的 profile 结构见设计文档第 178 行。"""
    return {
        "l2_events": [{"ts": e.ts, "node": e.node, "type": e.event_type,
                       "summary": e.summary} for e in l2_events],
        "belief_deltas": dict(belief_deltas),
        "prev_profile": prev_profile or {},
        "expected_schema": ["学习节奏", "方法偏好", "顽固病灶", "近期突破",
                            "沟通特征", "语言画像"],
    }


# ══════════════════════════════════════════════════════════════════════
# #7 自证回路隔离（红线:L3 档案只喂表达类,禁入证据类触点）
# ══════════════════════════════════════════════════════════════════════
def injectable_for(touchpoint: str, l3_profile: Optional[Dict]) -> Optional[Dict]:
    """返回该触点可注入的 L3 档案。**证据类触点（追问判定器）永远返回 None**——
    否则档案写'顽疾=X'→判定器倾向看见 X→L2 再记→蒸馏更肯定 X,推断层自我强化。
    表达类触点（开场/报告）返回档案本身。"""
    if touchpoint in EVIDENCE_TOUCHPOINTS:
        return None                            # 红线:证据类禁注入
    if touchpoint in EXPRESSION_TOUCHPOINTS:
        return l3_profile
    return None                                # 未知触点默认拒绝（安全默认）


# ══════════════════════════════════════════════════════════════════════
# 成长曲线（L1 周快照;第 155 行,免费产出;最低作答量门控）
# ══════════════════════════════════════════════════════════════════════
MIN_ANSWERS_FOR_CURVE = 10       # 累计 ≥10 次才上屏（避免小样本抖动误导）


def curve_eligible(cumulative_answers: int) -> bool:
    """成长曲线上屏判据:该节点累计作答 ≥10 次（与 L3 点亮判据同源）。"""
    return cumulative_answers >= MIN_ANSWERS_FOR_CURVE
