#!/usr/bin/env python3
"""
tests/test_mastery.py — engine/mastery.py 的 TDD 契约测试（先红后绿）
范围：设计文档 mastery 组断言 + 外部声部 #8/#9/#12 守护。
  不含 EIG selector（那是 selector.py / 阶段 1，等用户拍板 gap 阈值后建）。
运行：python3 tests/test_mastery.py   （pytest 也可发现 test_ 函数）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import mastery as m  # noqa: E402

M, P, C, U = m.M, m.P, m.C, m.U
DAY = m.DAY_SECONDS


# ── 似然：题型 ε/γ、二元退化、干扰项两段分解、守恒 ─────────────────────
def test_local_correct_probs_mcq():
    p = m.local_correct_probs(d=0.5, item_type="mcq")
    assert np.allclose(p, [0.90, 0.35, 0.60, 0.25]), p  # 设计文档第87行


def test_local_correct_probs_numeric_gamma():
    p = m.local_correct_probs(d=0.5, item_type="numeric")
    assert np.allclose(p, [0.90, 0.13, 0.60, 0.03]), p  # γ_数值=0.03, P(对|P)=γ+δ_p


def test_binary_degradation_when_distractor_missing():
    cp = m.local_correct_probs(0.5)
    assert np.allclose(m.likelihood_correct(cp), cp)
    assert np.allclose(m.likelihood_wrong_binary(cp), 1 - cp)  # 二元退化（第91行）


def test_distractor_error_rows_sum_to_one():
    derr = m.default_distractor_error()          # Σ_d P(d|错,s)=1（守恒断言）
    assert np.allclose(derr.sum(axis=1), 1.0), derr.sum(axis=1)


def test_distractor_likelihood_two_stage():
    cp = m.local_correct_probs(0.5)
    derr = m.default_distractor_error()
    L = m.likelihood_wrong_distractor(cp, derr, chosen_idx=0)  # P(选d|s)=P(错|s)·P(d|错,s)
    assert np.allclose(L, (1 - cp) * derr[:, 0]), L


# ── #12 跨节点 U 语义：前置题答对 → P(U):P(P) 比值上升 ────────────────
def test_prereq_correct_raises_U_over_P():
    b = np.array([0.1, 0.4, 0.1, 0.4])           # P/U 竞争
    ratio_before = b[U] / b[P]
    cp = m.prereq_correct_probs()                # P(对前置|U)=0.75 > P(对前置|P)=0.25
    b2 = m.bayes_update(b, m.likelihood_correct(cp))
    assert b2[U] / b2[P] > ratio_before          # 前置答对把质量推向 U


def test_prereq_u_default_frozen_value():
    cp = m.prereq_correct_probs()                # 无前置信念数据 → 缺省 0.75（第88行）
    assert np.isclose(cp[U], 0.75) and np.isclose(cp[P], 0.25) and np.isclose(cp[C], 0.80)


# ── 追问三元组：低置信零信息；#8 幂次压平 + 3:1 封顶；非法钳位 ─────────
def test_low_confidence_followup_zero_info():
    b = np.array([0.4, 0.2, 0.3, 0.1])
    L_raw = [0.05, 0.2, 0.7, 0.05]               # 强指向 C
    L, illegal = m.sanitize_llm_likelihood(L_raw, confidence=0.0)
    assert not illegal
    assert np.allclose(L, 0.25)                  # conf=0 → 均匀 → 零信息
    assert np.allclose(m.bayes_update(b, L), b)  # 信念零变化


def test_confidence_power_scales_evidence():
    # 用弱证据（比值 2:1 < 3:1 封顶阈值）隔离测幂次压平的单调性——
    # 强证据下 conf∈[0.5,1] 都会被 3:1 封顶饱和、失去区分（那是 #8 的正确行为，
    # 由 test_likelihood_ratio_capped_3to1 独立覆盖）。
    L_raw = [0.2, 0.2, 0.4, 0.2]
    L_full, _ = m.sanitize_llm_likelihood(L_raw, confidence=1.0)
    L_half, _ = m.sanitize_llm_likelihood(L_raw, confidence=0.5)
    L_zero, _ = m.sanitize_llm_likelihood(L_raw, confidence=0.0)
    d_full = np.abs(L_full - 0.25).sum()
    d_half = np.abs(L_half - 0.25).sum()
    d_zero = np.abs(L_zero - 0.25).sum()
    assert d_zero < d_half < d_full       # 置信越低越接近均匀
    assert np.isclose(d_zero, 0.0)        # conf=0 → 恰均匀 = 零信息


def test_likelihood_ratio_capped_3to1():
    L_raw = [0.85, 0.05, 0.05, 0.05]             # 原始 17:1
    L, illegal = m.sanitize_llm_likelihood(L_raw, confidence=1.0)
    assert not illegal
    assert L.max() / L.min() <= 3.0 + 1e-6       # #8 硬封顶 3:1（第103行）


def test_illegal_llm_likelihood_clamped():
    for bad in ([-0.1, 0.5, 0.3, 0.3],           # 负数
                [0.1, 0.2, 0.3],                  # 缺维
                [0.0, 0.0, 0.0, 0.0],             # 全零/不归一
                [float("nan"), 0.3, 0.3, 0.3]):   # NaN
        L, illegal = m.sanitize_llm_likelihood(bad, confidence=1.0)
        assert illegal, bad
        assert np.allclose(L, 0.25), (bad, L)     # 钳位为均匀 = 零信息（原始由调用方照录）


# ── FSRS 投影：R(t) 单调、幂等、单向 M→C 不碰 P/U、唯一入口 ───────────
def test_recall_monotonic_decreasing():
    S = 3.0
    rs = [m.recall_probability(t, S) for t in (0, 1, 5, 30, 100)]
    assert all(a > b for a, b in zip(rs, rs[1:]))
    assert np.isclose(rs[0], 1.0)                 # t=0 → R=1


def test_get_belief_idempotent_and_readonly():
    node = m.NodeBelief(b=np.array([0.6, 0.1, 0.1, 0.2]), S=3.0, last_review_at=0.0)
    stored_before = node.b.copy()
    b1 = m.get_belief(node, now=10 * DAY)
    b2 = m.get_belief(node, now=10 * DAY)
    assert np.allclose(b1, b2)                    # 幂等
    assert np.allclose(node.b, stored_before)     # 不回写存储态


def test_decay_only_M_to_C_not_PU():
    node = m.NodeBelief(b=np.array([0.6, 0.15, 0.05, 0.20]), S=2.0, last_review_at=0.0)
    proj = m.get_belief(node, now=60 * DAY)       # 大 Δt
    assert proj[M] < node.b[M]                    # M 流失
    assert proj[C] > node.b[C]                    # 流入 C
    assert np.isclose(proj[P], node.b[P])         # P 不碰（红线）
    assert np.isclose(proj[U], node.b[U])         # U 不碰（红线）
    assert np.isclose(proj.sum(), 1.0)


def test_no_decay_before_S_init():
    node = m.NodeBelief(b=np.array([0.7, 0.1, 0.1, 0.1]), S=None, last_review_at=0.0)
    assert np.allclose(m.get_belief(node, now=999 * DAY), node.b)  # S 未初始化 → 无衰减


# ── 单向投影铁律：更新先验=投影 b；存储态永不含衰减；同段遗忘只一次 ──
def test_update_prior_is_projected_belief():
    node = m.NodeBelief(b=np.array([0.6, 0.1, 0.1, 0.2]), S=3.0, last_review_at=0.0)
    now = 20 * DAY
    proj = m.get_belief(node, now)                # 学生此刻真实状态
    cp = m.local_correct_probs(0.5)
    expected = m.bayes_update(proj, m.likelihood_correct(cp))
    m.observe(node, m.likelihood_correct(cp), now, is_direct=True)
    assert np.allclose(node.b, expected)          # 更新先验=投影 b，不是存储原值


def test_stored_belief_resets_review_time_no_double_decay():
    node = m.NodeBelief(b=np.array([0.6, 0.1, 0.1, 0.2]), S=3.0, last_review_at=0.0)
    now = 20 * DAY
    m.observe(node, m.likelihood_correct(m.local_correct_probs(0.5)), now)
    assert node.last_review_at == now             # 重置 → 同段时间遗忘只发生一次
    assert np.allclose(m.get_belief(node, now), node.b)  # dt=0 → 不再二次扣减


# ── 层级先验：板块无作答回退全局；前置传播 η + #4 门控 + 归一 ──────────
def test_hierarchical_prior_fallback_global():
    g = np.array([0.4, 0.2, 0.2, 0.2])
    b0 = m.hierarchical_prior(block_beliefs=[], block_counts=[], global_prior=g)
    assert np.allclose(b0, g)                     # 板块内无作答 → 回退全局（第121行）


def test_hierarchical_prior_weighted_block():
    g = np.array([0.25, 0.25, 0.25, 0.25])
    bb = [np.array([0.8, 0.1, 0.05, 0.05]), np.array([0.2, 0.3, 0.3, 0.2])]
    b0 = m.hierarchical_prior(bb, [30, 10], g)    # 加权平均 → λ 混合
    assert np.isclose(b0.sum(), 1.0)
    assert b0[M] > g[M]                            # 板块偏 M → 先验偏 M


def test_prereq_propagation_eta_and_gate():
    g = np.array([0.25, 0.25, 0.25, 0.25])
    prereq = np.array([0.1, 0.3, 0.1, 0.5])       # P_前置(U)+P_前置(P)=0.8 >0.5
    # 门控关闭（对齐表未交付，#4）→ P 分量不加
    b_off = m.hierarchical_prior([], [], g, prereq_belief=prereq, prereq_available=False)
    assert np.allclose(b_off, g)
    # 门控开启 → P 分量 += η·max(0, 0.8−0.5)=0.4·0.3=0.12，加完归一
    b_on = m.hierarchical_prior([], [], g, prereq_belief=prereq, prereq_available=True)
    assert np.isclose(b_on.sum(), 1.0)
    assert b_on[P] > b_off[P]


# ── #9 S₀ 触发需直接证据；复核 S 增长/回落 ───────────────────────────
def test_S0_requires_direct_evidence_not_prior_echo():
    # P(M)>0.7 但直接作答<2 → 纯先验推高，不触发 S₀（防虚假复核链）
    node = m.NodeBelief(b=np.array([0.8, 0.1, 0.05, 0.05]), S=None,
                        last_review_at=None, direct_answers=1)
    m.maybe_init_S(node, now=DAY)
    assert node.S is None
    node.direct_answers = 2                        # 补足直接证据
    m.maybe_init_S(node, now=DAY)
    assert node.S == m.S0


def test_S0_triggers_on_heldout_pass():
    node = m.NodeBelief(b=np.array([0.5, 0.2, 0.2, 0.1]), S=None,
                        last_review_at=None, direct_answers=0)
    m.maybe_init_S(node, now=DAY, held_out_passed=True)
    assert node.S == m.S0


def test_S_growth_first_review_and_fail():
    node = m.NodeBelief(b=np.array([0.8, 0.1, 0.05, 0.05]), S=3.0, last_review_at=0.0)
    m.review_update_S(node, passed=True, now=DAY, first_review=True)
    assert np.isclose(node.S, 9.0)                 # 首次复核 ×3（第150行）
    m.review_update_S(node, passed=False, now=2 * DAY)
    assert np.isclose(node.S, 4.5)                 # 失败 ×0.5（第151行）


def test_node_belief_rejects_invalid_probability_vectors():
    bad_vectors = (
        [0.5, 0.5],
        [0.25, 0.25, 0.25, float("nan")],
        [0.5, 0.3, 0.3, -0.1],
        [0.4, 0.2, 0.2, 0.1],
    )
    for bad in bad_vectors:
        try:
            m.NodeBelief(b=bad)
            assert False, f"应拒绝非法 belief: {bad}"
        except ValueError:
            pass


def test_node_belief_rejects_invalid_stability_and_answer_count():
    for kwargs in ({"S": 0.0}, {"S": -1.0}, {"S": float("inf")},
                   {"direct_answers": -1}):
        try:
            m.NodeBelief(b=[0.25, 0.25, 0.25, 0.25], **kwargs)
            assert False, f"应拒绝非法状态: {kwargs}"
        except ValueError:
            pass


# ── 红线：get_belief 是唯一衰减入口（grep 无旁路直读存储态 .b）──────────
def test_get_belief_is_only_decay_entry_no_bypass():
    """扫描 engine/ 下除 mastery.py 外的模块，禁止裸直读 node.b（须经 get_belief）。
    放行两个显式访问器：get_belief（投影读）、stored_belief（复核判据的存储态读）。
    selector/planner/memory 加入后此红线自动生效。"""
    import glob
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for path in glob.glob(os.path.join(root, "engine", "*.py")):
        if os.path.basename(path) in ("mastery.py", "__init__.py"):
            continue
        with open(path) as f:
            src = f.read()
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue                       # 注释行不算
            if re.search(r"\.b\b", line) and "get_belief" not in line \
                    and "stored_belief" not in line:
                offenders.append(f"{os.path.basename(path)}:{i}: {line.strip()}")
    assert not offenders, "旁路裸直读存储态信念（须经 get_belief/stored_belief）:\n" + "\n".join(offenders)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"  PASS {n}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {n}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
