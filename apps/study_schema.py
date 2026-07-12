#!/usr/bin/env python3
"""
apps/study_schema.py — 诊断引擎数据地基增量 8 项 schema（2026-07-08）
====================================================================
规格源：mac-unknown-design-20260707-230853.md「数据地基增量清单」（第 207-216 行）。
原则：字段现在建全（事后补不回），v1 逻辑没用到的字段也要有位置；
     本模块只定义字段/构造/规范化（纯函数，零 fastapi 依赖），不含 serve 逻辑。
消费侧：engine/（mastery/recommender/memory/planner）从这些日志重算一切派生视图
     ——"一切派生视图可从事件日志全量重算"铁律（设计文档 Constraints）。

8 项对照：
  1 session.time_budget          → normalize_time_budget / TIME_BUDGET_MINUTES
  2 propensity 快照（最高优先）   → propensity_snapshot
  3 watch_proxy（服务端时间戳）   → watch_proxy_record（两级样本 + retest_delay 分层）
  4 掌握度 S + last_review_at    → mastery_state_row
  5 疗效表 {α,β,对照臂}          → efficacy_row
  6 L2 事件 + L3 档案(prompt版本) → l2_event_row / l3_profile_row
  7 subject 字段（现在锁死）      → stamp（所有落盘行统一注入）
  8 LLM 三元组 {错因码,置信度,L}  → followup_triage_row（存三元组而非结论）
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

SCHEMA_VERSION = "yher_diag_v1"      # 诊断引擎地基 v1；字段口径变更时 +1，新旧不混算
SUBJECT_DEFAULT = "chemistry"        # 第 7 项（补充点④，现在锁死）

# ── 第 1 项 · session.time_budget（分钟；设计文档第 189 行四档）─────────
TIME_BUDGET_MINUTES = {"30min": 30, "1h": 60, "2h": 120, "3h+": 180}


def normalize_time_budget(value: Any) -> Optional[int]:
    """档位字符串或分钟数 → 分钟 int；无法解析 → None（照录原值，不猜）。"""
    if isinstance(value, str) and value in TIME_BUDGET_MINUTES:
        return TIME_BUDGET_MINUTES[value]
    try:
        n = int(value)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


# ── 第 7 项 · subject + 版本戳（所有落盘行经此注入）────────────────────
def stamp(row: Dict[str, Any], subject: str = SUBJECT_DEFAULT) -> Dict[str, Any]:
    row.setdefault("subject", subject)
    row.setdefault("schema_version", SCHEMA_VERSION)
    return row


# ── 第 2 项 · propensity 快照（第 139 行；最高优先级，事后补不回）───────
def propensity_snapshot(candidate_ids: Sequence[str], scores: Any, chosen_id: str,
                        position: int = 1, randomized: bool = True, *,
                        node: str = "", session_id: str = "",
                        truncate_at: int = 500) -> Dict[str, Any]:
    """推荐/抽题事件的 propensity 快照 {候选集, 各分, 展示位次, randomized}。
    v1 随机抽题：scores 传 {"type":"uniform","value":1/N}；
    recommender 实装后：scores 传逐候选分数列表（与 candidates 对齐）。
    候选集超过 truncate_at 时截断并标记（candidates_total 保留全量数）。"""
    ids = list(candidate_ids)
    return {
        "kind": "propensity",
        "node": node,
        "session_id": session_id,
        "candidates": ids[:truncate_at],
        "candidates_total": len(ids),
        "candidates_truncated": len(ids) > truncate_at,
        "scores": scores,
        "chosen_id": chosen_id,
        "position": position,
        "randomized": bool(randomized),
    }


# ── 第 3 项 · watch_proxy（第 144 行 #6/#11；best-effort 标注）──────────
def watch_proxy_record(leave_ts: float, return_ts: float, seg_seconds: float,
                       self_report: Optional[str] = None,
                       retest_delay: Optional[str] = None, *,
                       session_id: str = "", segment_id: str = "",
                       bvid: str = "") -> Dict[str, Any]:
    """段观看代理事件。口径（#6 钉死）：
      离开时刻＝「跳转点击」API 事件的服务端时间戳；
      返回时刻＝该学生下一次任意 API 请求的服务端时间戳。
    两级样本（第 144 行）：
      时长达标(dwell ≥ 0.7×段长) 且自报看完 → A 级（进疗效表 α/β）；
      仅时长达标无自报            → B 级（存档不入 α/β）；
      时长不达标                  → none（闭环未走完，非样本）。
    #11 复测时延分层：retest_delay=="same_session" 时 A 级降 B（短时记忆测不出巩固）。
    字段标 best-effort：真实观看不可得（B 站站外无遥测），v2 自托管后口径版本 +1。"""
    dwell = max(0.0, float(return_ts) - float(leave_ts))
    dwell_ok = seg_seconds > 0 and dwell >= 0.7 * float(seg_seconds)
    if dwell_ok and self_report == "finished":
        level = "A"
    elif dwell_ok:
        level = "B"
    else:
        level = "none"
    if level == "A" and retest_delay == "same_session":
        level = "B"
    return {
        "kind": "watch_proxy",
        "session_id": session_id,
        "segment_id": segment_id,
        "bvid": bvid,
        "leave_ts": float(leave_ts),
        "return_ts": float(return_ts),
        "dwell_seconds": round(dwell, 3),
        "seg_seconds": float(seg_seconds),
        "dwell_ok": dwell_ok,
        "self_report": self_report,          # "finished" / "unfinished" / None
        "sample_level": level,               # "A" / "B" / "none"
        "retest_delay": retest_delay,        # "same_session"/"cross_session"/None(再测时回填)
        "best_effort": True,
    }


# ── 第 4 项 · 掌握度状态行（S + last_review_at；第 150 行）──────────────
def mastery_state_row(user_id: str, node: str, belief: Sequence[float],
                      S: Optional[float], last_review_at: Optional[float],
                      direct_answers: int = 0) -> Dict[str, Any]:
    """掌握度状态持久化行（append-only，最新行为准；可从事件日志全量重算）。
    belief=[P(M),P(P),P(C),P(U)] 存储态——永不含衰减（红线），
    衰减只在读取时经 engine.mastery.get_belief 懒投影。"""
    try:
        b = [float(x) for x in belief]
    except (TypeError, ValueError) as exc:
        raise ValueError("belief must be a numeric probability vector") from exc
    if len(b) != 4:
        raise ValueError(f"belief must be 4-dim, got {len(b)}")
    if any(not math.isfinite(value) or value < 0 for value in b):
        raise ValueError("belief values must be finite and non-negative")
    if not math.isclose(sum(b), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("belief probabilities must sum to 1")

    if S is not None:
        try:
            stability = float(S)
        except (TypeError, ValueError) as exc:
            raise ValueError("S must be a positive finite number or None") from exc
        if not math.isfinite(stability) or stability <= 0:
            raise ValueError("S must be a positive finite number or None")
    else:
        stability = None

    if last_review_at is not None:
        try:
            review_timestamp = float(last_review_at)
        except (TypeError, ValueError) as exc:
            raise ValueError("last_review_at must be finite or None") from exc
        if not math.isfinite(review_timestamp):
            raise ValueError("last_review_at must be finite or None")
    else:
        review_timestamp = None

    if isinstance(direct_answers, bool) or not isinstance(direct_answers, int) \
            or direct_answers < 0:
        raise ValueError("direct_answers must be a non-negative integer")
    return {
        "kind": "mastery_state",
        "user_id": user_id,
        "node": node,
        "belief": b,
        "S": stability,
        "last_review_at": review_timestamp,
        "direct_answers": direct_answers,
    }


# ── 第 5 项 · 疗效表行（{α,β,对照臂标记}；第 138/141 行）────────────────
def efficacy_row(segment_id: str, error_code: str, alpha: float, beta: float,
                 control_arm: bool = False) -> Dict[str, Any]:
    """疗效(段g,错因e) 存 {α,β} 而非比率（Beta-Binomial 收缩）；
    对照臂数据单独标记（control_arm=True 的行不与处方臂混算）。"""
    return {
        "kind": "efficacy",
        "segment_id": segment_id,
        "error_code": error_code,
        "alpha": float(alpha),
        "beta": float(beta),
        "control_arm": bool(control_arm),
    }


# ── 第 6 项 · L2 事件 + L3 叙事档案（第 157-181 行）─────────────────────
L2_EVENT_TYPES = ("belief_flip", "breakthrough", "stubborn_error",
                  "high_info_followup", "high_efficacy_watch")   # 准入五类（第 159-162 行）


def l2_event_row(user_id: str, node: str, event_type: str, summary: str,
                 related_event_id: Optional[str] = None) -> Dict[str, Any]:
    """L2 情节记忆事件：{时间, 节点, 类型, 一句话摘要, 关联事件id}。
    准入规则由 engine/memory.py 判定（本函数只钉字段），类型必须在五类内。"""
    if event_type not in L2_EVENT_TYPES:
        raise ValueError(f"event_type must be one of {L2_EVENT_TYPES}, got {event_type!r}")
    return {
        "kind": "l2_event",
        "user_id": user_id,
        "node": node,
        "event_type": event_type,
        "summary": str(summary)[:500],
        "related_event_id": related_event_id,
    }


def l3_profile_row(user_id: str, profile: Dict[str, Any],
                   distill_prompt_version: str) -> Dict[str, Any]:
    """L3 叙事档案（≤500 字结构化）；蒸馏 prompt 版本化入库（N3：换模型全量重蒸）。"""
    return {
        "kind": "l3_profile",
        "user_id": user_id,
        "profile": profile,
        "distill_prompt_version": str(distill_prompt_version),
    }


# ── 第 8 项 · LLM 慢路径三元组（第 100-104 行；存三元组而非结论）─────────
def followup_triage_row(item_id: str, utterance: str, error_code: str,
                        confidence: float, likelihood_vector: Any, *,
                        session_id: str = "", node: str = "") -> Dict[str, Any]:
    """追问判定器单次调用的三元组 {错因码, 置信度, 似然向量 L} 原样入日志——
    换 LLM 可重算历史（N3）。非法 L 的净化在引擎消费侧
    （engine.mastery.sanitize_llm_likelihood，静默失败路径①：钳位均匀＋原始照录），
    日志层只作 wellformed 轻量标注，不改写原始值。"""
    wf = _likelihood_wellformed(likelihood_vector)
    return {
        "kind": "followup_triage",
        "session_id": session_id,
        "item_id": item_id,
        "node": node,
        "utterance": str(utterance)[:1000],
        "error_code": str(error_code),
        "confidence": confidence,
        "likelihood_vector": likelihood_vector,   # 原样照录，不净化
        "wellformed": wf,
    }


def _likelihood_wellformed(L: Any) -> bool:
    if not isinstance(L, (list, tuple)) or len(L) != 4:
        return False
    try:
        vals = [float(x) for x in L]
    except (TypeError, ValueError):
        return False
    return all(v >= 0 and v == v and v != float("inf") for v in vals) and sum(vals) > 0
