#!/usr/bin/env python3
"""
tests/test_study_schema.py — apps/study_schema.py 的字段契约测试（零 fastapi 依赖）
覆盖数据地基增量 8 项 schema 的构造/规范化/分级逻辑。
运行：python3 tests/test_study_schema.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from apps import study_schema as sch  # noqa: E402


# ── 第 1 项 time_budget ────────────────────────────────────────────────
def test_normalize_time_budget():
    assert sch.normalize_time_budget("1h") == 60
    assert sch.normalize_time_budget("30min") == 30
    assert sch.normalize_time_budget("4h") == 240
    assert sch.normalize_time_budget("6h") == 360
    assert sch.normalize_time_budget("3h+") == 180
    assert sch.normalize_time_budget(45) == 45
    assert sch.normalize_time_budget("garbage") is None    # 无法解析 → None，不猜
    assert sch.normalize_time_budget(0) is None


# ── 第 7 项 subject + 版本戳 ───────────────────────────────────────────
def test_stamp_injects_subject_and_version():
    r = sch.stamp({"event": "x"})
    assert r["subject"] == "chemistry"
    assert r["schema_version"] == sch.SCHEMA_VERSION
    r2 = sch.stamp({"subject": "physics"})               # 已有不覆盖
    assert r2["subject"] == "physics"


# ── 第 2 项 propensity 快照 ────────────────────────────────────────────
def test_propensity_snapshot_structure():
    snap = sch.propensity_snapshot(["a", "b", "c"], {"type": "uniform", "value": 1 / 3},
                                   chosen_id="b", position=1, randomized=True, node="电化学")
    assert snap["kind"] == "propensity"
    assert snap["candidates_total"] == 3 and not snap["candidates_truncated"]
    assert snap["chosen_id"] == "b" and snap["randomized"] is True
    assert snap["node"] == "电化学"


def test_propensity_snapshot_truncates_large_candidate_set():
    big = [f"i{n}" for n in range(1200)]
    snap = sch.propensity_snapshot(big, {"type": "uniform"}, "i5", truncate_at=500)
    assert snap["candidates_total"] == 1200
    assert snap["candidates_truncated"] is True
    assert len(snap["candidates"]) == 500                # 截断但保留全量计数


# ── 第 3 项 watch_proxy 两级样本 + #11 降级 ────────────────────────────
def test_watch_proxy_A_level():
    r = sch.watch_proxy_record(leave_ts=0, return_ts=100, seg_seconds=100,
                               self_report="finished")
    assert r["dwell_ok"] and r["sample_level"] == "A"     # 时长达标 + 自报看完


def test_watch_proxy_B_level_no_self_report():
    r = sch.watch_proxy_record(leave_ts=0, return_ts=100, seg_seconds=100,
                               self_report=None)
    assert r["dwell_ok"] and r["sample_level"] == "B"     # 时长达标无自报 → 不进 α/β


def test_watch_proxy_none_when_dwell_short():
    r = sch.watch_proxy_record(leave_ts=0, return_ts=50, seg_seconds=100,
                               self_report="finished")
    assert not r["dwell_ok"] and r["sample_level"] == "none"  # 停留<0.7×段长


def test_watch_proxy_same_session_downgrades_A_to_B():
    r = sch.watch_proxy_record(leave_ts=0, return_ts=100, seg_seconds=100,
                               self_report="finished", retest_delay="same_session")
    assert r["sample_level"] == "B"                       # #11 短时记忆降级


# ── 第 4 项 掌握度状态 ─────────────────────────────────────────────────
def test_mastery_state_row_validates_4dim():
    r = sch.mastery_state_row("u1", "电化学", [0.6, 0.1, 0.1, 0.2], S=3.0, last_review_at=0.0)
    assert r["belief"] == [0.6, 0.1, 0.1, 0.2] and r["S"] == 3.0
    r2 = sch.mastery_state_row("u1", "n", [0.5, 0.3, 0.2, 0.0], S=None, last_review_at=None)
    assert r2["S"] is None and r2["last_review_at"] is None
    try:
        sch.mastery_state_row("u1", "n", [0.5, 0.5], S=None, last_review_at=None)
        assert False, "应对非 4 维 belief 报错"
    except ValueError:
        pass


# ── 第 5 项 疗效表 ─────────────────────────────────────────────────────
def test_mastery_state_row_rejects_invalid_probability_distribution():
    invalid_beliefs = (
        [0.5, 0.5],
        [0.25, 0.25, 0.25, float("nan")],
        [0.25, 0.25, 0.25, float("inf")],
        [0.5, 0.3, 0.3, -0.1],
        [0.4, 0.2, 0.2, 0.1],
        ["not-a-number", 0.2, 0.3, 0.5],
    )
    for belief in invalid_beliefs:
        try:
            sch.mastery_state_row("u1", "n", belief, S=None, last_review_at=None)
            assert False, f"应拒绝非法 belief: {belief}"
        except ValueError:
            pass


def test_mastery_state_row_rejects_invalid_metadata_scalars():
    valid = [0.25, 0.25, 0.25, 0.25]
    invalid_kwargs = (
        {"S": 0.0, "last_review_at": None, "direct_answers": 0},
        {"S": -1.0, "last_review_at": None, "direct_answers": 0},
        {"S": float("nan"), "last_review_at": None, "direct_answers": 0},
        {"S": float("inf"), "last_review_at": None, "direct_answers": 0},
        {"S": None, "last_review_at": float("nan"), "direct_answers": 0},
        {"S": None, "last_review_at": float("inf"), "direct_answers": 0},
        {"S": None, "last_review_at": None, "direct_answers": -1},
        {"S": None, "last_review_at": None, "direct_answers": 1.5},
        {"S": None, "last_review_at": None, "direct_answers": "1"},
        {"S": None, "last_review_at": None, "direct_answers": True},
    )
    for kwargs in invalid_kwargs:
        try:
            sch.mastery_state_row("u1", "n", valid, **kwargs)
            assert False, f"应拒绝非法掌握度元数据: {kwargs}"
        except (TypeError, ValueError):
            pass


def test_efficacy_row_control_arm():
    r = sch.efficacy_row("seg1", "E-过量误判", alpha=3.0, beta=2.0, control_arm=True)
    assert r["alpha"] == 3.0 and r["beta"] == 2.0 and r["control_arm"] is True


# ── 第 6 项 L2/L3 ──────────────────────────────────────────────────────
def test_l2_event_type_validated():
    r = sch.l2_event_row("u1", "电化学", "breakthrough", "held-out 首次通过")
    assert r["event_type"] == "breakthrough"
    try:
        sch.l2_event_row("u1", "n", "random_type", "x")
        assert False, "应拒绝非五类准入类型"
    except ValueError:
        pass


def test_l3_profile_versioned():
    r = sch.l3_profile_row("u1", {"顽固病灶": ["过量误判"]}, distill_prompt_version="p1")
    assert r["distill_prompt_version"] == "p1"


# ── 第 8 项 LLM 三元组：原样照录 + wellformed 标注 ─────────────────────
def test_followup_triage_records_raw_likelihood():
    r = sch.followup_triage_row("item1", "我以为过量就全反应了", "E-过量误判",
                                confidence=0.85, likelihood_vector=[0.05, 0.2, 0.7, 0.05])
    assert r["likelihood_vector"] == [0.05, 0.2, 0.7, 0.05]   # 原样，不净化
    assert r["confidence"] == 0.85 and r["wellformed"] is True


def test_followup_triage_flags_illegal_but_keeps_raw():
    r = sch.followup_triage_row("item1", "…", "E-x", confidence=0.5,
                                likelihood_vector=[0.1, 0.2])   # 缺维
    assert r["wellformed"] is False
    assert r["likelihood_vector"] == [0.1, 0.2]                # 非法也照录（引擎侧净化）


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
