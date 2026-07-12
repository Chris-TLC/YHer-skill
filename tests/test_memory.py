#!/usr/bin/env python3
"""
tests/test_memory.py — engine/memory.py 的 TDD 契约测试(先红后绿)
范围:诊断引擎设计文档块 3(第 147-185 行)L2/L3 三层记忆规格。
运行:python3 tests/test_memory.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import memory as mem   # noqa: E402
from engine import mastery as m    # noqa: E402

DAY = m.DAY_SECONDS


def ev(ts, node, etype, summary="s", vector=None):
    return mem.L2Event(ts=ts, node=node, event_type=etype, summary=summary, vector=vector)


# ── L2 准入五类(第 159-162 行)────────────────────────────────────────
def test_admit_breakthrough_held_out():
    assert mem.classify_admission(held_out_passed=True) == "breakthrough"


def test_admit_belief_flip_up_and_down():
    assert mem.classify_admission(pm_before=0.5, pm_after=0.75) == "belief_flip"   # 首过0.7
    assert mem.classify_admission(pm_before=0.6, pm_after=0.4) == "belief_flip"    # 跌破0.5


def test_no_flip_when_not_crossing():
    assert mem.classify_admission(pm_before=0.72, pm_after=0.75) is None   # 已在0.7上,非首过
    assert mem.classify_admission(pm_before=0.55, pm_after=0.52) is None   # 未跌破0.5


def test_admit_stubborn_error_third_time():
    assert mem.classify_admission(error_code_count=3) == "stubborn_error"
    assert mem.classify_admission(error_code_count=2) is None


def test_admit_high_info_followup_gate():
    assert mem.classify_admission(followup_confidence=0.7) == "high_info_followup"
    assert mem.classify_admission(followup_confidence=0.69) is None       # 低置信不入L2


def test_admit_high_efficacy_watch_l1_delta():
    """看完段后 ‖Δb‖₁≥0.3 → 高疗效观看。"""
    before = [0.3, 0.2, 0.3, 0.2]
    after = [0.6, 0.1, 0.2, 0.1]                                          # ‖Δb‖₁=0.6
    assert mem.classify_admission(b_before=before, b_after=after) == "high_efficacy_watch"
    small = [0.32, 0.2, 0.28, 0.2]                                        # ‖Δb‖₁=0.04
    assert mem.classify_admission(b_before=before, b_after=small) is None


def test_breakthrough_priority_over_flip():
    """同事件多命中取最强:held_out(突破)优先于信念翻转。"""
    r = mem.classify_admission(held_out_passed=True, pm_before=0.5, pm_after=0.8)
    assert r == "breakthrough"


# ── L2 检索打分(第 167-170 行)────────────────────────────────────────
def test_retrieval_recency_decays():
    """近的事件 recency 项更高。"""
    now = 100 * DAY
    recent = mem.retrieval_score(ev(99 * DAY, "n", "belief_flip"), now, None)
    old = mem.retrieval_score(ev(10 * DAY, "n", "belief_flip"), now, None)
    assert recent > old


def test_salience_weight_table():
    """突破/顽固错误显著性=1.0 > 高疗效观看=0.6。"""
    now = 50 * DAY
    bt = mem.retrieval_score(ev(50 * DAY, "n", "breakthrough"), now, None)
    hw = mem.retrieval_score(ev(50 * DAY, "n", "high_efficacy_watch"), now, None)
    assert bt > hw


def test_vector_similarity_injectable():
    """sim_fn 可注入;向量项参与打分。"""
    now = 50 * DAY
    e = ev(50 * DAY, "n", "belief_flip", vector=[1.0, 0.0])
    hi = mem.retrieval_score(e, now, [1.0, 0.0])          # 完全对齐 cos=1
    lo = mem.retrieval_score(e, now, [0.0, 1.0])          # 正交 cos=0
    assert hi > lo


def test_vector_missing_degrades_gracefully():
    """向量缺失 → 向量项 0,退化为 recency+salience,不崩。"""
    now = 50 * DAY
    s = mem.retrieval_score(ev(50 * DAY, "n", "belief_flip", vector=None), now, None)
    assert s >= 0


def test_numpy_vectors_do_not_use_ambiguous_truthiness():
    assert np.isclose(mem._cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 1.0)


def test_mismatched_vector_dimensions_degrade_to_zero_similarity():
    assert mem._cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_mismatched_belief_dimensions_are_rejected():
    try:
        mem.admit_high_efficacy_watch([0.5, 0.5], [0.4, 0.3, 0.2, 0.1])
        assert False, "应拒绝不同维 belief"
    except ValueError:
        pass


def test_rank_events_deterministic():
    """同分按时间新者优先(确定性)。"""
    now = 100 * DAY
    events = [ev(10 * DAY, "n", "belief_flip"), ev(90 * DAY, "n", "belief_flip")]
    ranked = mem.rank_events(events, now)
    assert ranked[0].ts == 90 * DAY


# ── 反人机闸门(第 183-184 行)─────────────────────────────────────────
def test_only_reference_today_touched_nodes():
    """仅当今日诊断触到有旧事件的节点才允许引用。"""
    events = [ev(50 * DAY, "电化学", "breakthrough"), ev(50 * DAY, "水解", "belief_flip")]
    refs = mem.select_references(events, today_nodes=["电化学"], now=60 * DAY)
    assert len(refs) == 1
    assert refs[0].node == "电化学"                       # 水解今日没触到,不引用


def test_max_one_reference_per_session():
    """每 session 开场至多引用 1 条。"""
    events = [ev(50 * DAY, "n", "breakthrough"), ev(51 * DAY, "n", "stubborn_error")]
    refs = mem.select_references(events, today_nodes=["n"], now=60 * DAY)
    assert len(refs) <= 1


def test_max_refs_argument_cannot_bypass_session_cap():
    events = [ev(50 * DAY, "n", "breakthrough"), ev(51 * DAY, "n", "stubborn_error")]
    refs = mem.select_references(events, today_nodes=["n"], now=60 * DAY, max_refs=99)
    assert len(refs) == 1


def test_no_reference_when_no_today_overlap():
    """今日节点与旧事件零交集 → 不引用(禁无目的寒暄)。"""
    events = [ev(50 * DAY, "电化学", "breakthrough")]
    refs = mem.select_references(events, today_nodes=["有机推断"], now=60 * DAY)
    assert refs == []


# ── L3 蒸馏触发(第 175 行)────────────────────────────────────────────
def test_distill_first_time_with_new_l2():
    """从未蒸馏 + 有新 L2 → 触发。"""
    assert mem.should_distill(None, now=10 * DAY, new_l2_since_last=3) is True


def test_distill_not_without_new_l2():
    """无新 L2 事件 → 不触发(哪怕很久没蒸)。"""
    assert mem.should_distill(0, now=100 * DAY, new_l2_since_last=0) is False


def test_distill_7day_gate():
    """距上次 >7 天且有新 L2 → 触发;<7 天不触发。"""
    assert mem.should_distill(0, now=8 * DAY, new_l2_since_last=1) is True
    assert mem.should_distill(0, now=6 * DAY, new_l2_since_last=1) is False


def test_distill_inputs_assembled():
    """蒸馏输入组装:含 L2 事件 + 信念变化 + 上版档案 + 期望 schema。"""
    events = [ev(50 * DAY, "n", "breakthrough", summary="过了held-out")]
    inp = mem.distill_inputs(events, {"n": 0.3}, {"顽固病灶": ["旧"]})
    assert inp["l2_events"][0]["node"] == "n"
    assert inp["belief_deltas"]["n"] == 0.3
    assert inp["prev_profile"] == {"顽固病灶": ["旧"]}
    assert "语言画像" in inp["expected_schema"]


# ── #7 自证回路隔离(红线,第 104 行)──────────────────────────────────
def test_l3_not_injected_to_followup_triage():
    """L3 档案禁入证据类触点(追问判定器)——防推断层自我强化。"""
    profile = {"顽固病灶": ["盐类水解"]}
    assert mem.injectable_for("followup_triage", profile) is None


def test_l3_injected_to_expression_touchpoints():
    """L3 档案可入表达类触点(开场/报告)。"""
    profile = {"学习节奏": "慢热"}
    assert mem.injectable_for("opening", profile) == profile
    assert mem.injectable_for("report", profile) == profile


def test_unknown_touchpoint_default_deny():
    """未知触点默认拒绝(安全默认)。"""
    assert mem.injectable_for("mystery_touchpoint", {"x": 1}) is None


# ── 成长曲线门控(第 155 行)───────────────────────────────────────────
def test_curve_min_answers_gate():
    """累计 ≥10 次才上屏(避免小样本抖动)。"""
    assert mem.curve_eligible(10) is True
    assert mem.curve_eligible(9) is False


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
