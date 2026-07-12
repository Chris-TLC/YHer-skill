#!/usr/bin/env python3
"""
tests/test_selector.py — engine/selector.py 的 TDD 契约测试（先红后绿）
范围：设计文档选题器规格（第 107-116 行）+ 外部声部 #2/#4 + T0 硬约束。
运行：python3 tests/test_selector.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import mastery as m       # noqa: E402
from engine import selector as sel    # noqa: E402

M, P, C, U = m.M, m.P, m.C, m.U


def _item(iid, node, itype="mcq", d=0.5, role="local", target_node=None):
    """构造候选题。role: local(本节点)/prereq(前置)/lateral(横向邻节点)。"""
    return {"item_id": iid, "node": node, "item_type": itype, "difficulty": d,
            "role": role, "target_node": target_node or node}


# ── 熵与 EIG 基础 ─────────────────────────────────────────────────────
def test_entropy_uniform_is_2bit():
    assert abs(sel.entropy(m.UNIFORM) - 2.0) < 1e-9      # 四状态均匀 = 2 bit


def test_entropy_certain_is_zero():
    assert sel.entropy(np.array([1.0, 0, 0, 0])) < 1e-9


def test_eig_nonnegative():
    b = np.array([0.4, 0.2, 0.3, 0.1])
    eig = sel.eig_local(b, d=0.5, item_type="mcq")
    assert eig >= -1e-9                                   # 信息增益非负


def test_eig_zero_when_belief_certain():
    b = np.array([0.98, 0.01, 0.005, 0.005])              # 已收敛
    assert sel.eig_local(b, d=0.5) < 0.05                 # 几乎无信息可得


# ── #2：EIG 按"该题所更新节点"算；横向题 EIG 来自邻节点自身熵 ──────────
def test_lateral_eig_from_neighbor_entropy_not_bk():
    b_k = np.array([0.9, 0.03, 0.04, 0.03])               # k 已收敛
    b_neighbor = m.UNIFORM.copy()                          # 邻节点全不确定
    beliefs = {"k": b_k, "nb": b_neighbor}
    lateral = _item("L1", "nb", role="lateral", target_node="nb")
    eig = sel.item_eig(lateral, beliefs)
    # 横向题 EIG 来自邻节点的熵下降，不是对 b_k（对 b_k 应为 0）
    assert eig > 0.1


def test_converged_node_flows_to_neighbor():
    """k 收敛且邻节点未测 → 必选邻节点题（联合熵目标的输出）。"""
    b_k = np.array([0.95, 0.02, 0.02, 0.01])
    beliefs = {"k": b_k, "nb": m.UNIFORM.copy()}
    cands = [_item("localk", "k", role="local", target_node="k"),
             _item("lateralnb", "nb", role="lateral", target_node="nb")]
    pick = sel.select_next(cands, beliefs, target_nodes=["k", "nb"])
    assert pick["item_id"] == "lateralnb"


# ── P/U 竞争 → 必选前置题（跨节点似然差 0.75 vs 0.25 远大于本节点）──────
def test_PU_competition_selects_prereq():
    b_k = np.array([0.05, 0.45, 0.05, 0.45])              # P/U 竞争
    beliefs = {"k": b_k, "pre": np.array([0.7, 0.1, 0.1, 0.1])}
    cands = [_item("localk", "k", role="local", target_node="k"),
             _item("prereqp", "pre", role="prereq", target_node="k")]
    pick = sel.select_next(cands, beliefs, target_nodes=["k"],
                           prereq_available=True)
    assert pick["item_id"] == "prereqp"                   # 前置题对 b_k 消歧力最大


# ── #4：对齐表缺席（prereq_available=False）→ 前置候选禁用 ──────────────
def test_prereq_gated_when_alignment_absent():
    b_k = np.array([0.05, 0.45, 0.05, 0.45])
    beliefs = {"k": b_k, "pre": np.array([0.7, 0.1, 0.1, 0.1])}
    cands = [_item("localk", "k", role="local", target_node="k"),
             _item("prereqp", "pre", role="prereq", target_node="k")]
    pick = sel.select_next(cands, beliefs, target_nodes=["k"],
                           prereq_available=False)
    assert pick["item_id"] == "localk"                    # 门控关：前置题不可选


# ── T0 硬约束：前置题信念触发，非固定配额 ──────────────────────────────
def test_prereq_not_selected_when_C_dominates():
    """C 主导（非 P/U 竞争）时不该出前置题——T0 证明固定插前置对 C 有害。"""
    b_k = np.array([0.1, 0.05, 0.8, 0.05])                # C 主导
    beliefs = {"k": b_k, "pre": np.array([0.7, 0.1, 0.1, 0.1])}
    cands = [_item("localk", "k", role="local", target_node="k"),
             _item("prereqp", "pre", role="prereq", target_node="k")]
    pick = sel.select_next(cands, beliefs, target_nodes=["k"],
                           prereq_available=True)
    assert pick["item_id"] == "localk"                    # C 主导 → 本节点题信息增益更高


# ── 掌握差信念 → 选横向不选更难（不推更难题）─────────────────────────
def test_weak_mastery_picks_lateral_not_harder():
    b_k = np.array([0.9, 0.03, 0.04, 0.03])               # k 掌握
    beliefs = {"k": b_k, "nb": m.UNIFORM.copy()}
    cands = [_item("hard_k", "k", role="local", d=0.9, target_node="k"),
             _item("lateral_nb", "nb", role="lateral", target_node="nb")]
    pick = sel.select_next(cands, beliefs, target_nodes=["k", "nb"])
    assert pick["item_id"] == "lateral_nb"                # 掌握后画边界，不推更难


# ── holdout 禁取红线 ──────────────────────────────────────────────────
def test_holdout_never_selected():
    b_k = m.UNIFORM.copy()
    beliefs = {"k": b_k}
    cands = [_item("normal", "k", role="local", target_node="k"),
             {**_item("hold", "k", role="local", target_node="k"), "holdout": True}]
    pick = sel.select_next(cands, beliefs, target_nodes=["k"])
    assert pick["item_id"] == "normal"                    # holdout 题禁进选题


def test_seen_items_excluded():
    beliefs = {"k": m.UNIFORM.copy()}
    cands = [_item("a", "k", target_node="k"), _item("b", "k", target_node="k")]
    pick = sel.select_next(cands, beliefs, target_nodes=["k"], seen_ids={"a"})
    assert pick["item_id"] == "b"


# ── 收敛终止（gap>0.45 且直接作答≥3 题）+ 预算耗尽 ─────────────────────
def test_should_stop_on_convergence():
    b = {"k": np.array([0.7, 0.1, 0.15, 0.05])}           # gap=0.55>0.45
    assert sel.should_stop(b, target_nodes=["k"], direct_answers={"k": 3},
                           budget_items=15, asked=5)


def test_should_not_stop_under_3_direct_answers():
    b = {"k": np.array([0.8, 0.1, 0.05, 0.05])}           # gap 高但证据不足
    assert not sel.should_stop(b, target_nodes=["k"], direct_answers={"k": 2},
                               budget_items=15, asked=5)   # #9：<3 题不收敛


def test_should_stop_on_budget_exhausted():
    b = {"k": m.UNIFORM.copy()}                            # 没收敛
    assert sel.should_stop(b, target_nodes=["k"], direct_answers={"k": 5},
                           budget_items=9, asked=9)        # 预算耗尽也停


# ── 每 session 目标节点 ≥2 题（覆盖约束）──────────────────────────────
def test_target_node_min_coverage():
    """两个目标节点，其一还没测够 2 题时，选题偏向欠测节点。"""
    beliefs = {"k1": np.array([0.7, 0.1, 0.15, 0.05]), "k2": m.UNIFORM.copy()}
    cands = [_item("k1c", "k1", target_node="k1"), _item("k2c", "k2", target_node="k2")]
    pick = sel.select_next(cands, beliefs, target_nodes=["k1", "k2"],
                           asked_per_node={"k1": 3, "k2": 0})
    assert pick["target_node"] == "k2"                    # k2 欠测 → 优先


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
