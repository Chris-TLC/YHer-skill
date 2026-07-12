#!/usr/bin/env python3
"""
engine/mastery.py — 病因信念引擎（诊断引擎设计文档块 1，2026-07-08）
===================================================================
四状态信念 {M 已掌握 / P 前置缺失 / C 思维链不稳 / U 完全不会} 的贝叶斯更新、
按题型 slip/guess、干扰项两段分解、层级先验、FSRS 单向衰减投影。

规格源：mac-unknown-design-20260707-230853.md（行号在注释里标注）。
范围：本模块是信念内核。EIG 选题器在 selector.py（阶段 1，待 T0 拍板 gap 阈值）。

红线（设计文档第 152 行）：
  · 衰减公式只存在于 _project_decay，get_belief 是唯一读入口——禁止旁路直读 node.b。
  · 单向投影只重分配 M→C，不碰 P/U。
  · 存储态 b 永不含衰减；更新先验＝投影后的 b；同段时间遗忘只发生一次。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── 状态索引（全模块固定顺序）─────────────────────────────────────────
M, P, C, U = 0, 1, 2, 3
STATES = ("M", "P", "C", "U")
UNIFORM = np.full(4, 0.25)

# ── 观测参数（设计文档第 86-87 行，模块常量＋注释；升级触发器见第 86 行）──
EPS = 0.10                        # slip 粗心，各题型共用
GAMMA = {"mcq": 0.25, "numeric": 0.03}   # guess：四选一 / 纯数值短填
DELTA_P = 0.10                    # δ_p，前置缺但本步可能半会

# ── 跨节点似然（第 88 行）──
PREREQ_CORRECT_C = 0.80           # P(对前置|C)：思路不稳者前置通常在
PREREQ_U_DEFAULT = 0.75           # P(对前置|U)：无前置信念数据时缺省（session 快照冻结）

# ── 层级先验（第 119-122 行）──
LAMBDA_PRIOR = 0.6                # b_k(0)=λ·板块后验+(1−λ)·全局先验
ETA_PREREQ = 0.4                  # 前置传播幅度

# ── FSRS 衰减 + 稳定性 S（第 49/150-151 行）──
S0 = 3.0                          # S 初值（天）
S_GAIN = 1.0                      # 常规复核增益 S×(1+增益)
S_FIRST_REVIEW_MULT = 3.0         # 首次复核 S×3（已巩固知识快速稳定）
S_FAIL_MULT = 0.5                 # 复核失败 S×0.5
DAY_SECONDS = 86400

# ── LLM 慢路径钳位（第 103 行 #8）──
LIKELIHOOD_RATIO_CAP = 3.0        # 似然比硬封顶：一句被误读的中文最多顶一道题量级


# ══════════════════════════════════════════════════════════════════════
# 信念存储态
# ══════════════════════════════════════════════════════════════════════
@dataclass
class NodeBelief:
    """(学生, 节点) 的信念存储态。b 永不含衰减；衰减在 get_belief 里懒投影。"""
    b: np.ndarray                              # [P(M),P(P),P(C),P(U)]，存储态
    S: Optional[float] = None                  # 稳定性（天），None=未初始化
    last_review_at: Optional[float] = None     # 上次事件时间戳（秒）
    direct_answers: int = 0                    # 本节点直接作答计数（#9 收敛门）
    subject: str = "chemistry"                 # schema 增量清单第 7 项

    def __post_init__(self) -> None:
        try:
            belief = np.asarray(self.b, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("belief 必须是四维数值概率") from exc
        if belief.shape != (4,):
            raise ValueError(f"belief 必须是四维，实际 shape={belief.shape}")
        if not np.all(np.isfinite(belief)) or np.any(belief < 0):
            raise ValueError("belief 必须为有限非负数")
        if not np.isclose(belief.sum(), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("belief 概率和必须约等于 1")
        if self.S is not None and (not np.isfinite(self.S) or self.S <= 0):
            raise ValueError("S 必须为有限正数或 None")
        if self.direct_answers < 0:
            raise ValueError("direct_answers 不得为负")
        self.b = belief


# ══════════════════════════════════════════════════════════════════════
# 似然表（第 87-91 行）
# ══════════════════════════════════════════════════════════════════════
def local_correct_probs(d: float, item_type: str = "mcq") -> np.ndarray:
    """本节点题各状态答对概率（第 87 行）。d∈[0,1] 归一化难度。"""
    g = GAMMA[item_type]
    return np.array([1 - EPS, g + DELTA_P, 0.7 - 0.2 * d, g])


def prereq_correct_probs(prereq_u_predict: float = PREREQ_U_DEFAULT,
                         item_type: str = "mcq") -> np.ndarray:
    """前置节点题各状态答对概率（第 88 行）。
       prereq_u_predict＝前置节点自身预测答对率，session 开始快照冻结（#12）。"""
    g = GAMMA[item_type]
    return np.array([1 - EPS, g, PREREQ_CORRECT_C, prereq_u_predict])


def likelihood_correct(correct_probs: np.ndarray) -> np.ndarray:
    """观测＝答对，似然向量＝P(对|s)。"""
    return np.asarray(correct_probs, dtype=float)


def likelihood_wrong_binary(correct_probs: np.ndarray) -> np.ndarray:
    """观测＝答错、无 distractor_map（二元退化，第 91 行）：似然＝1−P(对|s)。"""
    return 1.0 - np.asarray(correct_probs, dtype=float)


def default_distractor_error() -> np.ndarray:
    """P(d|错,s) 模板（第 89-90 行）：行=状态[M,P,C,U]，列=[典型误解, 其他, 其他]。
       钉死：C 选典型误解=0.60，U/M 均匀；P 行为模板值（错因码表量产时逐题覆盖）。
       每行归一（Σ_d P(d|错,s)=1）。"""
    return np.array([
        [1 / 3, 1 / 3, 1 / 3],     # M：slip 随机
        [0.45, 0.275, 0.275],      # P：偏典型误解
        [0.60, 0.20, 0.20],        # C：典型误解 0.60
        [1 / 3, 1 / 3, 1 / 3],     # U：均匀
    ])


def likelihood_wrong_distractor(correct_probs: np.ndarray, derr: np.ndarray,
                                chosen_idx: int) -> np.ndarray:
    """观测＝答错且选了干扰项 chosen_idx，两段分解（第 89 行）：
       P(选d|s) = P(错|s)·P(d|错,s) = (1−P(对|s))·derr[s, chosen_idx]。"""
    return (1.0 - np.asarray(correct_probs, dtype=float)) * derr[:, chosen_idx]


# ══════════════════════════════════════════════════════════════════════
# 贝叶斯更新（第 95 行）
# ══════════════════════════════════════════════════════════════════════
def bayes_update(b: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """一行贝叶斯：b'(s) ∝ P(观测|s)·b(s)，归一化。"""
    post = np.asarray(b, dtype=float) * np.asarray(likelihood, dtype=float)
    total = post.sum()
    if total <= 0:                    # 退化保护：似然全零 → 信念不变
        return np.asarray(b, dtype=float).copy()
    return post / total


# ══════════════════════════════════════════════════════════════════════
# LLM 慢路径似然钳位（第 103 行 #8；失败模式：幻觉非法值）
# ══════════════════════════════════════════════════════════════════════
def _cap_likelihood_ratio(L: np.ndarray, cap: float) -> np.ndarray:
    """把似然比 max/min 压到 ≤ cap（对数空间线性收缩，保序）。"""
    L = L / L.sum()
    L = np.clip(L, 1e-12, None)
    if L.max() / L.min() <= cap:
        return L / L.sum()
    logL = np.log(L)
    logL -= logL.mean()
    span = logL.max() - logL.min()
    logL *= np.log(cap) / span        # 压缩对数跨度到 log(cap)
    Lc = np.exp(logL)
    return Lc / Lc.sum()


def sanitize_llm_likelihood(L_raw, confidence: float):
    """追问判定器的似然向量净化（#8，第 103 行）。
    返回 (L_clean, was_illegal)。原始 L_raw 由调用方照录日志（静默失败路径①）。
      · 非法（形状/NaN/负/非正和）→ 均匀分布＝零信息。
      · 幂次压平：L' = L^置信度（conf=0 → 均匀；conf=1 → 原样）。
      · 似然比硬封顶 3:1。"""
    L = np.asarray(L_raw, dtype=float) if _is_numeric_seq(L_raw) else None
    if (L is None or L.shape != (4,) or not np.all(np.isfinite(L))
            or np.any(L < 0) or L.sum() <= 0):
        return UNIFORM.copy(), True
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return UNIFORM.copy(), True
    if not np.isfinite(conf):
        return UNIFORM.copy(), True
    L = L / L.sum()
    conf = float(np.clip(conf, 0.0, 1.0))
    L = L ** conf                     # 幂次压平
    L = L / L.sum()
    L = _cap_likelihood_ratio(L, LIKELIHOOD_RATIO_CAP)
    return L, False


def _is_numeric_seq(x) -> bool:
    try:
        arr = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        return False
    return arr.ndim == 1


# ══════════════════════════════════════════════════════════════════════
# 层级先验（第 119-122 行）
# ══════════════════════════════════════════════════════════════════════
def hierarchical_prior(block_beliefs, block_counts, global_prior: np.ndarray,
                       prereq_belief: Optional[np.ndarray] = None,
                       prereq_available: bool = False) -> np.ndarray:
    """b_k(0) = λ·板块级后验 + (1−λ)·全局先验（第 119 行）。
       板块级后验＝板块内各节点 b 按累计作答数加权平均（第 121 行）；
       板块内无作答 → λ 项回退全局先验。
       前置传播（第 122 行，#4 门控 prereq_available）：
         P 分量 += η·max(0, P_前置(U)+P_前置(P)−0.5)，加完归一。"""
    global_prior = np.asarray(global_prior, dtype=float)
    counts = np.asarray(block_counts, dtype=float) if len(block_counts) else np.array([])
    if counts.size and counts.sum() > 0:
        stacked = np.stack([np.asarray(x, dtype=float) for x in block_beliefs])
        block_post = (stacked * counts[:, None]).sum(0) / counts.sum()
    else:
        block_post = global_prior     # 板块无作答 → 回退
    b0 = LAMBDA_PRIOR * block_post + (1 - LAMBDA_PRIOR) * global_prior
    if prereq_available and prereq_belief is not None:   # #4：对齐表交付后才启用
        pq = np.asarray(prereq_belief, dtype=float)
        b0 = b0.copy()
        b0[P] += ETA_PREREQ * max(0.0, pq[U] + pq[P] - 0.5)
    return b0 / b0.sum()              # 加完归一


# ══════════════════════════════════════════════════════════════════════
# FSRS 衰减投影 + get_belief 唯一读入口（第 150-153 行，红线）
# ══════════════════════════════════════════════════════════════════════
def recall_probability(dt_days: float, S: float) -> float:
    """FSRS 经典可提取性 R(t) = (1 + t/(9S))^(−1)（第 49/150 行）。"""
    return (1.0 + dt_days / (9.0 * S)) ** (-1.0)


def _project_decay(b_stored: np.ndarray, S, last_review_at, now) -> np.ndarray:
    """★衰减公式的唯一所在地★（第 152 行）。单向投影：M 中 (1−R) 比例流入 C，
       P/U 不碰。幂等、不回写、存储态永不含衰减。"""
    b = np.asarray(b_stored, dtype=float).copy()
    if S is None or last_review_at is None:
        return b                       # S 未初始化 → 无衰减
    dt_days = max(0.0, (now - last_review_at) / DAY_SECONDS)
    R = recall_probability(dt_days, S)
    lost = (1.0 - R) * b[M]            # M 流失的质量
    b[M] = R * b[M]
    b[C] = b[C] + lost                # 生疏 ≈ 思维链不稳的表现型（第 153 行）
    return b                           # P/U 未动，总质量守恒仍归一


def get_belief(node: NodeBelief, now: float) -> np.ndarray:
    """★唯一读入口★（第 152 行红线）。选题器/规划器/报告/记忆层一律经此读信念，
       禁止直读 node.b。"""
    return _project_decay(node.b, node.S, node.last_review_at, now)


def stored_belief(node: NodeBelief) -> np.ndarray:
    """存储态（未衰减）信念的**显式**读取。仅一个合法用途：规划器复核判据需要
       比较『存储态 P(M)』与『投影后 P(M)』的跌幅（第 154 行）——这不是旁路，
       是刻意要未投影值。除此之外一律用 get_belief。红线测试放行本访问器、仍拦裸 .b。"""
    return np.asarray(node.b, dtype=float).copy()


# ══════════════════════════════════════════════════════════════════════
# 观测更新（真实作答/再测事件）
# ══════════════════════════════════════════════════════════════════════
def observe(node: NodeBelief, likelihood: np.ndarray, now: float,
            is_direct: bool = True, held_out_passed: bool = False) -> NodeBelief:
    """真实作答/再测事件（第 152 行）：先验＝投影后的 b（学生此刻真实状态），
       更新结果写回存储态并重置 last_review_at（同段时间遗忘只发生一次）。"""
    b_prior = get_belief(node, now)               # 投影一次（遗忘只在此发生）
    node.b = bayes_update(b_prior, likelihood)    # 写回：此刻更新态，不含未来衰减
    node.last_review_at = now                      # 重置 → 下次从 now 起算 Δt
    if is_direct:
        node.direct_answers += 1
    maybe_init_S(node, now, held_out_passed=held_out_passed)
    return node


# ══════════════════════════════════════════════════════════════════════
# 稳定性 S 的初始化与复核更新（第 130/150-151 行；#9）
# ══════════════════════════════════════════════════════════════════════
def maybe_init_S(node: NodeBelief, now: float, held_out_passed: bool = False) -> None:
    """S₀ 触发（第 130 行 #9）：P(M) 首过 0.7 且本节点直接作答 ≥2 题，或 held-out 通过。
       纯先验推高（直接作答不足）不触发——防先验回声点燃虚假复核链。"""
    if node.S is not None:
        return
    if (node.b[M] > 0.7 and node.direct_answers >= 2) or held_out_passed:
        node.S = S0
        node.last_review_at = now


def review_update_S(node: NodeBelief, passed: bool, now: float,
                    first_review: bool = False) -> NodeBelief:
    """复核更新 S（第 150-151 行）：通过则增长（首次 ×3，此后 ×(1+增益)），失败 ×0.5。"""
    if node.S is None:
        return node
    if passed:
        node.S *= S_FIRST_REVIEW_MULT if first_review else (1.0 + S_GAIN)
    else:
        node.S *= S_FAIL_MULT
    node.last_review_at = now
    return node
