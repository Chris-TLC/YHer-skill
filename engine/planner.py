#!/usr/bin/env python3
"""
engine/planner.py — 时间预算规划器（设计文档块 4，第 187-205 行）
================================================================
预算约束下最大化预期掌握增益（v1 贪心+查表，不上优化器）。
预算表（30min/1h-6h）；30min = 浅诊断口径（用户 2026-07-08 拍板）。
输出 session_budget 给 selector（EIG 循环终止预算）与 recommender（约束）。

组件：
  · session_budget(tier)：查表出诊断题数/目标节点数/处方段数/时长/追问次数上限。
  · plan_reviews：复核插入（存储 P(M)>0.7 但投影跌破 0.6 → 插 2 题；#9 节流 ≥7 天、
    每 session ≤2 节点）——衰减机器消费方。
  · check_exhaustion：预算耗尽降级（诊断超时 ×1.5 → 未收敛标记+处方存档+不强制再测）。
  · explain_plan：把安排与理由印给学生（用户明确要求）。
"""
from __future__ import annotations

from typing import Dict, Optional

from engine import mastery as m

# 产品预算表。3h+ 仅在 session_budget 输入边界解析为 3h。
# 30min: 用户 2026-07-08 拍板改浅诊断口径（M/非M 二分，不承诺病因收敛）。
BUDGET_TABLE = {
    "30min": {"total_minutes": 30, "mode": "shallow", "diagnostic_items": 9, "diagnostic_minutes": 12,
              "target_nodes": 1, "rx_segments": 1, "rx_minutes": 8,
              "retest_items": 3, "followup_max": 2, "buffer_minutes": 1},
    "1h":    {"total_minutes": 60, "mode": "full", "diagnostic_items": 15, "diagnostic_minutes": 20,
              "target_nodes": 2, "rx_segments": 2, "rx_minutes": 25,
              "retest_items": 5, "followup_max": 3, "buffer_minutes": 12},
    "2h":    {"total_minutes": 120, "mode": "full", "diagnostic_items": 25, "diagnostic_minutes": 40,
              "target_nodes": 2, "rx_segments": 4, "rx_minutes": 60,
              "retest_items": 10, "followup_max": 5, "buffer_minutes": 14},
    "3h":    {"total_minutes": 180, "mode": "full", "diagnostic_items": 25, "diagnostic_minutes": 40,
              "target_nodes": 2, "rx_segments": 8, "rx_minutes": 120,
              "retest_items": 10, "followup_max": 5, "buffer_minutes": 14},
    "4h":    {"total_minutes": 240, "mode": "full", "diagnostic_items": 25, "diagnostic_minutes": 40,
              "target_nodes": 2, "rx_segments": 12, "rx_minutes": 180,
              "retest_items": 10, "followup_max": 5, "buffer_minutes": 14},
    "5h":    {"total_minutes": 300, "mode": "full", "diagnostic_items": 25, "diagnostic_minutes": 40,
              "target_nodes": 2, "rx_segments": 16, "rx_minutes": 240,
              "retest_items": 10, "followup_max": 5, "buffer_minutes": 14},
    "6h":    {"total_minutes": 360, "mode": "full", "diagnostic_items": 25, "diagnostic_minutes": 40,
              "target_nodes": 2, "rx_segments": 20, "rx_minutes": 300,
              "retest_items": 10, "followup_max": 5, "buffer_minutes": 14},
}

_BUDGET_ALIASES = {"3h+": "3h"}

REVIEW_STORE_THRESHOLD = 0.7        # 存储 P(M) 高于此才考虑复核
REVIEW_DECAY_THRESHOLD = 0.6        # 投影跌破此触发复核
REVIEW_ITEMS = 2                    # 复核插 2 题
REVIEW_THROTTLE_DAYS = 7            # #9：同节点复核提示间隔 ≥7 天
REVIEW_MAX_NODES = 2               # 每 session 复核 ≤2 节点
EXHAUSTION_MULT = 1.5               # 诊断超时 ×1.5 触发降级


def session_budget(tier: str) -> Dict:
    """查表返回该档预算；未知档闭合失败。"""
    canonical = _BUDGET_ALIASES.get(tier, tier)
    if canonical not in BUDGET_TABLE:
        raise ValueError(f"unsupported session budget: {tier!r}")
    return dict(BUDGET_TABLE[canonical])


def estimate_minutes(budget: Dict) -> float:
    """账目估算：诊断+处方+再测+缓冲，用于验收各档 ≤ 预算×1.1。
    再测按 ~0.6min/题（比诊断题快，纯作答无追问）。"""
    diag = budget["diagnostic_minutes"]
    rx = budget["rx_minutes"]
    retest = budget["retest_items"] * 0.6
    buf = budget["buffer_minutes"]
    return diag + rx + retest + buf


def plan_reviews(nodes: Dict[str, m.NodeBelief], now: float,
                 last_review_prompt: Optional[Dict[str, float]] = None) -> Dict:
    """复核插入（衰减机器消费方，第 154 行）。
    对每个节点：存储 P(M)>0.7 但 get_belief 投影后跌破 0.6 → 候选复核。
    #9 节流：同节点上次提示 <7 天不重提；每 session ≤2 节点。"""
    last_review_prompt = last_review_prompt or {}
    candidates = []
    for name, node in nodes.items():
        stored = m.stored_belief(node)              # 显式存储态读（复核判据需未投影值）
        if stored[m.M] <= REVIEW_STORE_THRESHOLD:
            continue                                # 存储态本就不高，非复核对象
        proj = m.get_belief(node, now)              # 唯一入口读投影
        if proj[m.M] >= REVIEW_DECAY_THRESHOLD:
            continue                                # 没跌破，不用复核
        last = last_review_prompt.get(name)
        if last is not None and (now - last) < REVIEW_THROTTLE_DAYS * m.DAY_SECONDS:
            continue                                # 节流：<7 天不重提
        drop = stored[m.M] - proj[m.M]              # 跌幅
        candidates.append((drop, name))
    candidates.sort(reverse=True)                   # 跌幅大的优先
    chosen = [name for _, name in candidates[:REVIEW_MAX_NODES]]
    return {"review_nodes": chosen,
            "review_items": REVIEW_ITEMS if chosen else 0}


def check_exhaustion(budget: Dict, elapsed_diagnostic_min: float) -> Dict:
    """预算耗尽降级（第 203 行，静默失败路径②）。
    诊断实际用时 > 档诊断预算 ×1.5 → 立即结束诊断（哪怕未收敛），
    处方照开但不强制再测，状态落服务端日志下次续上。"""
    threshold = budget["diagnostic_minutes"] * EXHAUSTION_MULT
    if elapsed_diagnostic_min > threshold:
        return {"force_stop": True, "force_retest": False,
                "message": "今天时间到了，处方已存好，下次打开从看段开始。"}
    return {"force_stop": False, "force_retest": True, "message": ""}


def explain_plan(budget: Dict) -> str:
    """把安排与理由印给学生（第 200 行，用户明确要求）。数字实时从预算表生成。"""
    total = budget["total_minutes"]
    time_label = (
        f"{total // 60}小时（{total} 分钟）" if total >= 60 else f"{total} 分钟"
    )
    plan = (
        f"你今天有 {time_label}：诊断最多 {budget['diagnostic_items']} 道、最多 "
        f"{budget['diagnostic_minutes']} 分钟；视频学习最多 {budget['rx_segments']} 段、最多 "
        f"{budget['rx_minutes']} 分钟；再做 {budget['retest_items']} 道复测，预留 "
        f"{budget['buffer_minutes']} 分钟缓冲。"
    )
    if budget["mode"] == "shallow":
        return plan + "时间有限，今天只做快速定位、不深挖病因。"
    return plan + "更长的学习时段会把新增时间用于学习，而不会增加诊断题量。"
