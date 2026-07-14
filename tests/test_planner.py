#!/usr/bin/env python3
"""
tests/test_planner.py — engine/planner.py 的 TDD 契约测试（先红后绿）
范围：设计文档块 4（第 187-205 行）+ 用户 2026-07-08 拍板 30min 浅诊断 + #9 复核节流。
运行：python3 tests/test_planner.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import mastery as m       # noqa: E402
from engine import planner as pl      # noqa: E402

DAY = m.DAY_SECONDS

HOURLY_BUDGETS = {
    "1h": (60, 15, 20, 2, 2, 25, 5, 3, 12),
    "2h": (120, 25, 40, 2, 4, 60, 10, 5, 14),
    "3h": (180, 25, 40, 2, 8, 120, 10, 5, 14),
    "4h": (240, 25, 40, 2, 12, 180, 10, 5, 14),
    "5h": (300, 25, 40, 2, 16, 240, 10, 5, 14),
    "6h": (360, 25, 40, 2, 20, 300, 10, 5, 14),
}
HOURLY_FIELDS = (
    "total_minutes",
    "diagnostic_items",
    "diagnostic_minutes",
    "target_nodes",
    "rx_segments",
    "rx_minutes",
    "retest_items",
    "followup_max",
    "buffer_minutes",
)


# ── 预算表 ────────────────────────────────────────────────────────────
def test_budget_table_all_tiers():
    for tier in ("30min", "1h", "2h", "3h", "4h", "5h", "6h", "3h+"):
        b = pl.session_budget(tier)
        assert b["diagnostic_items"] > 0
        assert b["target_nodes"] >= 1
        assert b["followup_max"] >= 0


def test_budget_monotonic_with_time():
    d30 = pl.session_budget("30min")["diagnostic_items"]
    d1h = pl.session_budget("1h")["diagnostic_items"]
    d2h = pl.session_budget("2h")["diagnostic_items"]
    # Early budgets grow to T0; 2h+ stays capped at 25 without weakening safety gates.
    assert d30 < d1h < d2h


def test_budget_matches_design_values():
    assert pl.session_budget("1h")["diagnostic_items"] == 15   # 第 195 行
    assert pl.session_budget("2h")["diagnostic_items"] == 25   # 第 196 行


def test_canonical_hourly_tiers_have_exact_total_minutes_and_alias():
    assert pl.session_budget("30min").get("total_minutes") == 30
    for tier, expected in HOURLY_BUDGETS.items():
        budget = pl.session_budget(tier)
        assert tuple(budget.get(field) for field in HOURLY_FIELDS) == expected
    assert pl.session_budget("3h+") == pl.session_budget("3h")


def test_diagnostic_budget_caps_at_t0_after_two_hours():
    assert 9 < pl.session_budget("1h")["diagnostic_items"] < 25
    for tier in ("2h", "3h", "4h", "5h", "6h"):
        budget = pl.session_budget(tier)
        # T0 deliberately caps diagnosis at 25 items; longer sessions add learning time.
        assert budget["diagnostic_items"] == 25
        assert budget["diagnostic_minutes"] == 40
        assert budget["followup_max"] <= 5


def test_recommendation_time_grows_after_two_hours_without_more_diagnosis():
    tiers = ("2h", "3h", "4h", "5h", "6h")
    rx = [pl.session_budget(tier)["rx_minutes"] for tier in tiers]
    assert rx == sorted(rx)
    assert len(set(rx)) == len(rx)
    assert all(pl.session_budget(tier)["diagnostic_items"] == 25 for tier in tiers)


def test_canonical_budgets_have_no_two_rounds_and_exact_accounting():
    for tier in ("30min", "1h", "2h", "3h", "4h", "5h", "6h"):
        budget = pl.session_budget(tier)
        assert "two_rounds" not in budget
    for tier in ("1h", "2h", "3h", "4h", "5h", "6h"):
        budget = pl.session_budget(tier)
        assert pl.estimate_minutes(budget) == int(tier[:-1]) * 60
    assert pl.estimate_minutes(pl.session_budget("30min")) <= 30 * 1.1


def test_unknown_budget_values_fail_closed():
    for value in ("90min", "7h", ""):
        with pytest.raises(ValueError):
            pl.session_budget(value)


# ── 30min 浅诊断口径（用户 2026-07-08 拍板）────────────────────────────
def test_30min_is_shallow_diagnosis():
    b = pl.session_budget("30min")
    assert b["mode"] == "shallow"                   # 浅诊断：M/非M 二分，不承诺病因收敛
    assert b["target_nodes"] == 1                   # 单节点
    assert 8 <= b["diagnostic_items"] <= 10         # 8-10 题（第 194 行）


def test_higher_tiers_are_full_diagnosis():
    for tier in ("1h", "2h", "3h", "4h", "5h", "6h", "3h+"):
        assert pl.session_budget(tier)["mode"] == "full"


# ── 账目不超预算（各档最坏 ≤ 预算×1.1）─────────────────────────────────
def test_budget_accounting_within_110pct():
    totals = {
        "30min": 30,
        "1h": 60,
        "2h": 120,
        "3h": 180,
        "4h": 240,
        "5h": 300,
        "6h": 360,
    }
    for tier, minutes in totals.items():
        est = pl.estimate_minutes(pl.session_budget(tier))
        assert est <= minutes * 1.1, (tier, est)


# ── 复核插入（存储 P(M)>0.7 但投影跌破 0.6 → 插 2 题）──────────────────
def test_review_insertion_when_decayed():
    # 存储态 P(M)=0.8，S 小 + 久未复习 → 投影跌破 0.6
    node = m.NodeBelief(b=np.array([0.8, 0.05, 0.1, 0.05]), S=2.0,
                        last_review_at=0.0)
    plan = pl.plan_reviews({"电化学": node}, now=60 * DAY,
                           last_review_prompt={})
    assert "电化学" in plan["review_nodes"]
    assert plan["review_items"] == 2                # 插 2 题


def test_no_review_when_fresh():
    node = m.NodeBelief(b=np.array([0.8, 0.05, 0.1, 0.05]), S=100.0,
                        last_review_at=0.0)          # S 大 → 不衰减
    plan = pl.plan_reviews({"电化学": node}, now=1 * DAY, last_review_prompt={})
    assert plan["review_nodes"] == []


def test_review_throttle_7days(): # noqa
    """#9 节流：同节点复核提示间隔 ≥7 天。"""
    node = m.NodeBelief(b=np.array([0.8, 0.05, 0.1, 0.05]), S=2.0, last_review_at=0.0)
    # 3 天前刚提示过 → 不重提
    plan = pl.plan_reviews({"电化学": node}, now=60 * DAY,
                           last_review_prompt={"电化学": 57 * DAY})
    assert "电化学" not in plan["review_nodes"]


def test_review_max_2_nodes_per_session():
    nodes = {f"n{i}": m.NodeBelief(b=np.array([0.8, 0.05, 0.1, 0.05]), S=2.0,
                                   last_review_at=0.0) for i in range(5)}
    plan = pl.plan_reviews(nodes, now=60 * DAY, last_review_prompt={})
    assert len(plan["review_nodes"]) <= 2           # 每 session ≤2 节点


# ── 预算耗尽降级（诊断超时 ×1.5 → 未收敛标记 + 不强制再测）─────────────
def test_budget_exhaustion_degradation():
    b = pl.session_budget("1h")
    # 用时超诊断预算 ×1.5
    deg = pl.check_exhaustion(b, elapsed_diagnostic_min=b["diagnostic_minutes"] * 1.6)
    assert deg["force_stop"] is True
    assert deg["force_retest"] is False             # 处方存档但不强制再测
    assert "下次" in deg["message"]                  # 断点续上文案


def test_no_degradation_within_budget():
    b = pl.session_budget("1h")
    deg = pl.check_exhaustion(b, elapsed_diagnostic_min=b["diagnostic_minutes"] * 0.9)
    assert deg["force_stop"] is False


# ── 规划解释（理由印给学生）──────────────────────────────────────────
def test_plan_explanation_generated():
    text = pl.explain_plan(pl.session_budget("1h"))
    assert "分钟" in text or "题" in text            # 有具体安排数字
    assert len(text) > 10


def test_six_hour_plan_explains_caps_and_learning_allocation():
    text = pl.explain_plan(pl.session_budget("6h"))
    assert "6小时" in text or "360分钟" in text or "360 分钟" in text
    assert "25" in text and "诊断" in text
    assert "最多 20 段" in text
    assert "更长" in text and "学习" in text and "诊断" in text


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
