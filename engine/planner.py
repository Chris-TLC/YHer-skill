#!/usr/bin/env python3
"""
engine/planner.py — 时间预算规划器（设计文档块 4，第 187-205 行）
================================================================
预算约束下最大化预期掌握增益（v1 贪心+查表，不上优化器）。
四档预算表（30min/1h/2h/3h+）；30min = 浅诊断口径（用户 2026-07-08 拍板）。
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

# 四档预算表（第 192-198 行；模块常量+注释，可调不改函数形式）。
# 30min: 用户 2026-07-08 拍板改浅诊断口径（M/非M 二分，不承诺病因收敛）。
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
              "two_rounds": True},   # 两轮完整闭环；结尾不填满、诚实收手
}

REVIEW_STORE_THRESHOLD = 0.7        # 存储 P(M) 高于此才考虑复核
REVIEW_DECAY_THRESHOLD = 0.6        # 投影跌破此触发复核
REVIEW_ITEMS = 2                    # 复核插 2 题
REVIEW_THROTTLE_DAYS = 7            # #9：同节点复核提示间隔 ≥7 天
REVIEW_MAX_NODES = 2               # 每 session 复核 ≤2 节点
EXHAUSTION_MULT = 1.5               # 诊断超时 ×1.5 触发降级


def session_budget(tier: str) -> Dict:
    """查表返回该档预算。未知档默认 1h。"""
    return dict(BUDGET_TABLE.get(tier, BUDGET_TABLE["1h"]))


def estimate_minutes(budget: Dict) -> float:
    """账目估算：诊断+处方+再测+缓冲，用于验收各档 ≤ 预算×1.1。
    再测按 ~0.6min/题（比诊断题快，纯作答无追问）。"""
    diag = budget["diagnostic_minutes"]
    rx = budget["rx_minutes"]
    retest = budget["retest_items"] * 0.6
    buf = budget["buffer_minutes"]
    total = diag + rx + retest + buf
    if budget.get("two_rounds"):
        total = (diag + rx + retest) * 2 + buf      # 两轮闭环
    return total


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
    if budget["mode"] == "shallow":
        return (f"你今天有约 30 分钟：先用 {budget['diagnostic_items']} 道题快速定位"
                f"「哪些点掌握了、哪些还没」，看 1 段视频补最弱的，再 "
                f"{budget['retest_items']} 题确认。时间有限，今天只做快速定位、不深挖病因。")
    return (f"你今天有这些时间：约 {budget['diagnostic_minutes']} 分钟定位问题（{budget['diagnostic_items']} 题），"
            f"{budget['rx_minutes']} 分钟看 {budget['rx_segments']} 段视频，"
            f"{int(budget['retest_items']*0.6)+1} 分钟证明补上了（{budget['retest_items']} 题），"
            f"剩下 {budget['buffer_minutes']} 分钟是缓冲。")
