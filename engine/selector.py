#!/usr/bin/env python3
"""
engine/selector.py — 精确期望信息增益选题器（设计文档块 1 选题器，第 107-116 行）
================================================================================
EIG(候选题 j) = H(b_n(j)) − Σ_o P(o|b_n(j),j)·H(b_n(j) 更新后(o))
  其中 n(j) = 该题作答后信念被直接更新的节点：
    本节点题→k；前置题→k（经跨节点似然表）；横向题→邻节点自身。
session 目标 = 降低全体目标节点的联合熵（外部声部 #2）。

四状态×每节点几十候选×≤5 观测结果 = 穷举精确计算，微秒级，不近似。

红线/约束：
  · holdout 题禁进选题（继承母文档）。
  · 前置题候选门控于对齐表交付（#4 prereq_available）。
  · T0 硬约束：前置题信念触发——由 EIG 公式自然输出（P/U 竞争时前置题 EIG 最高、
    C 主导时本节点题 EIG 更高），不设固定配额。
  · 每 session 目标节点 ≥2 题。
  · 信念一律经 mastery.get_belief 读（本模块不直读存储态 .b）。
收敛终止在 should_stop：gap>0.45 且直接作答≥3 题（#9），或预算耗尽。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from engine import mastery as m

GAP_THRESHOLD = 0.45           # 用户 2026-07-08 拍板维持
MIN_DIRECT_ANSWERS = 3         # #9：直接证据下限（防先验回声）
MIN_ITEMS_PER_TARGET = 2       # 每 session 目标节点最低题量


def entropy(b: np.ndarray) -> float:
    """香农熵（bit）。四状态均匀 = 2 bit。"""
    b = np.asarray(b, dtype=float)
    nz = b[b > 1e-12]
    return float(-(nz * np.log2(nz)).sum())


def _observation_likelihoods(d: float, item_type: str, role: str,
                             prereq_u_predict: float):
    """返回该题各观测结果的似然向量列表 [(似然向量, ), ...]。
    二元退化档（distractor_map 全空，存量真实条件）：观测=对/错两种。"""
    if role == "prereq":
        cp = m.prereq_correct_probs(prereq_u_predict, item_type)
    else:
        cp = m.local_correct_probs(d, item_type)
    return [m.likelihood_correct(cp), m.likelihood_wrong_binary(cp)]


def _expected_posterior_entropy(b: np.ndarray, likelihoods) -> float:
    """Σ_o P(o|b)·H(b 更新后(o))。P(o|b) = Σ_s L_o(s)·b(s)。"""
    b = np.asarray(b, dtype=float)
    total = 0.0
    for L in likelihoods:
        po = float((L * b).sum())               # 该观测的边际概率
        if po <= 1e-12:
            continue
        total += po * entropy(m.bayes_update(b, L))
    return total


def eig_local(b: np.ndarray, d: float, item_type: str = "mcq") -> float:
    """本节点题对其节点信念的 EIG。"""
    L = _observation_likelihoods(d, item_type, "local", m.PREREQ_U_DEFAULT)
    return entropy(b) - _expected_posterior_entropy(b, L)


def item_eig(item: Dict, beliefs: Dict[str, np.ndarray],
             prereq_u_predict: float = m.PREREQ_U_DEFAULT) -> float:
    """候选题 EIG，按其所更新节点计算（#2）。
    item: {role, target_node, difficulty, item_type}。
    beliefs: {node: 投影后信念向量}（调用方已用 get_belief 投影）。"""
    role = item.get("role", "local")
    d = item.get("difficulty", 0.5)
    itype = item.get("item_type", "mcq")
    if role == "lateral":
        # 横向题更新邻节点自身 → EIG 来自邻节点的熵下降（对 b_k 为零信息）
        nb = item.get("target_node")
        b_nb = beliefs.get(nb)
        if b_nb is None:
            return 0.0
        L = _observation_likelihoods(d, itype, "local", prereq_u_predict)
        return entropy(b_nb) - _expected_posterior_entropy(b_nb, L)
    # local / prereq 都更新目标节点 k 的信念，但用各自的似然表
    k = item.get("target_node")
    b_k = beliefs.get(k)
    if b_k is None:
        return 0.0
    L = _observation_likelihoods(d, itype, role, prereq_u_predict)
    return entropy(b_k) - _expected_posterior_entropy(b_k, L)


def _candidate_pool(candidates, seen_ids, prereq_available):
    """过滤：holdout 禁取、已做排除、对齐表缺席时前置题门控（#4）。"""
    seen_ids = seen_ids or set()
    pool = []
    for it in candidates:
        if it.get("holdout"):
            continue                              # holdout 红线
        if it.get("item_id") in seen_ids:
            continue
        if it.get("role") == "prereq" and not prereq_available:
            continue                              # #4 门控
        pool.append(it)
    return pool


def select_next(candidates: Sequence[Dict], beliefs: Dict[str, np.ndarray],
                target_nodes: Sequence[str], *,
                seen_ids: Optional[set] = None,
                prereq_available: bool = False,
                asked_per_node: Optional[Dict[str, int]] = None,
                prereq_u_predict: float = m.PREREQ_U_DEFAULT) -> Optional[Dict]:
    """选下一题：最大化 EIG，服从覆盖约束（每目标节点 ≥2 题）。
    覆盖约束优先于纯 EIG：有目标节点欠测（<MIN_ITEMS_PER_TARGET）时，
    先在欠测节点的候选里挑 EIG 最高者。"""
    pool = _candidate_pool(candidates, seen_ids, prereq_available)
    if not pool:
        return None
    asked_per_node = asked_per_node or {}

    # 覆盖约束：欠测的目标节点
    under = [n for n in target_nodes if asked_per_node.get(n, 0) < MIN_ITEMS_PER_TARGET]
    if under:
        under_set = set(under)
        under_pool = [it for it in pool if it.get("target_node") in under_set]
        if under_pool:
            pool = under_pool

    best, best_eig = None, -1.0
    for it in pool:
        e = item_eig(it, beliefs, prereq_u_predict)
        if e > best_eig:
            best, best_eig = it, e
    return best


def should_stop(beliefs: Dict[str, np.ndarray], target_nodes: Sequence[str], *,
                direct_answers: Dict[str, int], budget_items: int,
                asked: int) -> bool:
    """终止条件（第 128 行）：
      [gap>0.45 且本节点直接作答≥3 题]（全部目标节点满足）  或  预算耗尽。"""
    if asked >= budget_items:
        return True                               # 预算耗尽
    for n in target_nodes:
        b = beliefs.get(n)
        if b is None:
            return False
        srt = np.sort(np.asarray(b, dtype=float))
        gap = srt[-1] - srt[-2]
        if gap <= GAP_THRESHOLD:
            return False
        if direct_answers.get(n, 0) < MIN_DIRECT_ANSWERS:
            return False                          # #9：证据量不足不收敛
    return True
