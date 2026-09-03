#!/usr/bin/env python3
"""
tests/test_recommender.py — engine/recommender.py 的 TDD 契约测试(先红后绿)
范围:设计文档 §4 的 28 条断言(eng-review E1-E8 + CEO 复审 F1-F5 定稿)。
运行:python3 tests/test_recommender.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import recommender as rec   # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# fixtures(参数传表,E6:测试造假表零 mock)
# ══════════════════════════════════════════════════════════════════════
def raw_track_map(entities_extra=None):
    """标准 5 轨 track_map(YAML 原始形)。entities 预注册每轨样例 bv。"""
    tm = {
        "version": "curriculum_v1",
        "tracks": [
            {"id": "foundation",
             "audience": {"高一": {"preview": 1.0, "review": 1.0, "exam_prep": 0.6},
                          "高二": {"preview": 1.0, "review": 1.0, "exam_prep": 0.6},
                          "高三": {"preview": 0.6, "review": 0.4, "exam_prep": 0.2}},
             "diagnostic_unlock": ["U", "P"]},
            {"id": "round1",
             "audience": {"高一": {"preview": 0.0, "review": 0.2, "exam_prep": 0.2},
                          "高二": {"preview": 0.2, "review": 0.6, "exam_prep": 0.6},
                          "高三": {"preview": 0.6, "review": 1.0, "exam_prep": 1.0}},
             "mastery_gate": 0.6},
            {"id": "sprint",
             "audience": {"高一": {"preview": 0.0, "review": 0.2, "exam_prep": 0.6},
                          "高二": {"preview": 0.0, "review": 0.2, "exam_prep": 0.8},
                          "高三": {"preview": 0.2, "review": 0.6, "exam_prep": 1.0}},
             "mastery_gate": 0.6},
            {"id": "topical",
             "audience": {"高一": {"preview": 0.6, "review": 0.8, "exam_prep": 0.6},
                          "高二": {"preview": 0.6, "review": 0.8, "exam_prep": 0.8},
                          "高三": {"preview": 0.6, "review": 0.8, "exam_prep": 0.8}}},
            {"id": "scene",
             "audience": {"高一": {"preview": 0.0, "review": 0.0, "exam_prep": 0.0},
                          "高二": {"preview": 0.0, "review": 0.0, "exam_prep": 0.0},
                          "高三": {"preview": 0.0, "review": 0.0, "exam_prep": 0.0}}},
        ],
        "entities": [
            {"entity": "bv:F1", "track": "foundation", "reviewer": "user_chris",
             "needs_human": False, "evidence": "fixture:foundation"},
            {"entity": "bv:R1", "track": "round1", "reviewer": "user_chris",
             "needs_human": False, "evidence": "fixture:round1"},
            {"entity": "bv:S1", "track": "sprint", "reviewer": "user_chris",
             "needs_human": False, "evidence": "fixture:sprint"},
            {"entity": "bv:T1", "track": "topical", "reviewer": "user_chris",
             "needs_human": False, "evidence": "fixture:topical"},
            {"entity": "bv:SC1", "track": "scene", "reviewer": "user_chris",
             "needs_human": False, "evidence": "fixture:scene"},
        ],
    }
    if entities_extra:
        tm["entities"].extend(entities_extra)
    return tm


def tm():
    return rec.load_track_map(raw_track_map())


def seg(bv, p=1, seg_type="concept_intro", diff="T2", tmr=1.0, dur=480,
        view=1000, pubdate="2023-01-01", season_id=None, season_order=None,
        season_name=None, part_title=None, video_title="视频总标题", degrade=None):
    return {"bv": bv, "p": p, "seg_type": seg_type, "difficulty": diff,
            "topic_match_ratio": tmr, "duration_sec": dur, "view": view,
            "pubdate": pubdate, "season_id": season_id, "season_order": season_order,
            "season_name": season_name,
            "part_title": part_title if part_title is not None else f"真标题P{p}",
            "video_title": video_title, "part_degrade_state": degrade,
            "start_sec": 0, "end_sec": dur}


def blf(mm, pp, cc, uu):
    return np.array([mm, pp, cc, uu], dtype=float)


NODE = "盐类水解"
B_U = blf(0.1, 0.1, 0.1, 0.7)       # argmax=U, P(M)=0.1
B_C = blf(0.35, 0.03, 0.60, 0.02)   # argmax=C, P(M)=0.35(<0.6,必被 gate 压)
B_M_HI = blf(0.75, 0.10, 0.10, 0.05)  # argmax=M, P(M)=0.75
B_UNIF = rec.np.full(4, 0.25)


def _run(grade, purpose, belief, segs, budget=None, **kw):
    budget = budget or {"mode": "full", "rx_minutes": 15, "rx_segments": 2}
    return rec.recommend(grade, purpose, [NODE], {NODE: belief},
                         {NODE: segs}, tm(), budget, **kw)


# ══════════════════════════════════════════════════════════════════════
# 核心路由断言
# ══════════════════════════════════════════════════════════════════════
def test_01_gaosan_sprint_U_crosses_to_foundation():
    """高三+考前冲刺+U → 必出 foundation 概念段,理由含跨轨解释。"""
    out = _run("高三", "exam_prep", B_U,
               [seg("F1", seg_type="concept_intro"), seg("S1", seg_type="drill")])
    assert out["recommendations"], "应有推荐"
    top = out["recommendations"][0]
    assert top["track_id"] == "foundation"
    assert "基础大合集" in top["reason"]        # 跨轨理由含轨道名


def test_02_gaosan_sprint_C_not_foundation_concept():
    """高三+考前冲刺+C → 出方法段(topical),不出 foundation 基础概念段。
       注:C 态 argmax=C ⟹ P(M)<0.5<0.6,必被 sprint/round1 gate 压;C 的正确落点是
       无门槛的 topical(专项=方法段),不是 sprint。断言字面'sprint/round1'实为'方法段'。"""
    out = _run("高三", "exam_prep", B_C,
               [seg("F1", seg_type="concept_intro"), seg("T1", seg_type="method")])
    assert out["recommendations"]
    assert out["recommendations"][0]["track_id"] != "foundation"
    assert out["recommendations"][0]["track_id"] == "topical"


def test_03a_gaoyi_review_no_round1():
    """高一+巩固复习 → foundation(1.0)压 round1(0.2),输出不含 round1。"""
    out = _run("高一", "review", B_U,
               [seg("F1", seg_type="concept_intro"), seg("R1", seg_type="review")])
    tids = {r["track_id"] for r in out["recommendations"]}
    assert "round1" not in tids
    assert "foundation" in tids


def test_03b_fallback_round1_gate_hard_excluded():
    """兜底时 round1 被 mastery_gate 硬排除:高一预习(round1 audience=0)+只有 round1 候选
       +P(M)<0.6 → 全零兜底但 gate 压制轨排除 → 暂无视频。"""
    out = _run("高一", "preview", B_U, [seg("R1", seg_type="review")])
    assert out["status"] == "no_segment"
    assert out["recommendations"] == []


def test_04_gate_suppress_round1_unlock_foundation():
    """P(M)=0.3+argmax=U+高三复习 → round1 gate 压 0.2,foundation unlock 升 1.0 → foundation。"""
    b = blf(0.3, 0.1, 0.1, 0.5)                 # P(M)=0.3, argmax=U
    out = _run("高三", "review", b,
               [seg("F1", seg_type="concept_intro"), seg("R1", seg_type="review")])
    assert out["recommendations"][0]["track_id"] == "foundation"


def test_04b_M_state_round1_passes_gate():
    """P(M)=0.75+argmax=M+高三复习 → round1 gate 放行(0.75≥0.6),推 round1 复习段。"""
    out = _run("高三", "review", B_M_HI,
               [seg("F1", seg_type="review"), seg("R1", seg_type="review")])
    assert out["recommendations"][0]["track_id"] == "round1"


def test_05_tiebreak_newer_version_wins():
    """同节点两段除 pubdate 外同分 → 新版胜出。"""
    old = seg("F1", p=1, pubdate="2020-01-01")
    new = seg("F1", p=2, pubdate="2024-06-01")
    out = _run("高二", "review", B_U, [old, new],
               budget={"mode": "full", "rx_minutes": 15, "rx_segments": 1})
    assert out["recommendations"][0]["p"] == 2   # 新版


def test_06_missing_entity_falls_back_foundation():
    """track_map 缺实体映射 → foundation 回退 + warning。"""
    warns = []
    t = rec.resolve_track(seg("BV_UNKNOWN"), tm(), warns)
    assert t == "foundation"
    assert any("foundation fallback" in w for w in warns)


def test_07_budget_hard_cap():
    """总时长超预算 → 硬淘汰,多段累计不超预算。"""
    out = _run("高二", "review", B_U,
               [seg("F1", p=1, dur=600), seg("F1", p=2, dur=600), seg("F1", p=3, dur=600)],
               budget={"mode": "full", "rx_minutes": 15, "rx_segments": 5})  # 900s 预算
    total = sum(s["end_sec"] - s["start_sec"] for s in out["recommendations"])
    assert total <= 900


def test_08_cross_track_reason_has_track_name():
    """跨轨发生时理由字符串必含轨道名。"""
    out = _run("高三", "exam_prep", B_U, [seg("F1", seg_type="concept_intro")])
    assert "基础大合集" in out["recommendations"][0]["reason"]


def test_reason_uses_full_session_time_remaining_after_allocated_video():
    out = _run(
        "高二",
        "review",
        B_U,
        [seg("F1", dur=480)],
        budget={
            "mode": "shallow",
            "rx_minutes": 8,
            "rx_segments": 1,
            "session_remaining_minutes": 30,
        },
    )

    assert "今天还剩约22分钟" in out["recommendations"][0]["reason"]


def test_09_efficacy_v1_is_one():
    """Efficacy v1 恒 1.0 → 快照 efficacy=1.0,排序等价 W_track×Match。"""
    out = _run("高三", "exam_prep", B_U, [seg("F1", seg_type="concept_intro")])
    assert out["rec_served"]["served"][0]["efficacy"] == 1.0


def test_10_incomplete_signatures_stay_neutral_and_fall_back():
    """未签实体保留供审计，但不得激活声明的轨道。"""
    incomplete = (
        {"reviewer": "", "needs_human": False, "evidence": "catalog:X"},
        {"reviewer": "user_chris", "needs_human": True, "evidence": "catalog:X"},
        {"reviewer": "claude", "evidence": "catalog:X"},
        {"reviewer": "user_chris", "needs_human": False, "evidence": "  "},
    )
    for index, signature in enumerate(incomplete):
        entity_id = f"bv:UNSIGNED{index}"
        entity = {"entity": entity_id, "track": "sprint", **signature}
        track_map = rec.load_track_map(raw_track_map(entities_extra=[entity]))

        assert entity_id not in track_map["entities"]
        assert entity_id in track_map["neutral_entities"]
        warnings = []
        assert rec.resolve_track(seg(f"UNSIGNED{index}"), track_map, warnings) == "foundation"
        assert warnings


def test_10b_complete_user_claude_and_authorized_codex_signatures_activate():
    """既有人工 signer 与本轮授权 Codex signer 均兼容完整证据。"""
    signed = [
        {"entity": "bv:USER", "track": "foundation", "reviewer": "user_chris",
         "needs_human": False, "evidence": "catalog:user"},
        {"entity": "bv:CLAUDE", "track": "round1", "reviewer": "claude",
         "needs_human": False, "evidence": "catalog:claude"},
        {"entity": "bv:CODEX", "track": "sprint", "reviewer": "codex_sol_20260713",
         "needs_human": False, "evidence": "catalog:codex"},
    ]
    track_map = rec.load_track_map(raw_track_map(entities_extra=signed))
    assert all(entity["entity"] in track_map["entities"] for entity in signed)


def test_10c_authorized_codex_requires_closed_review_and_evidence():
    signed = {"entity": "bv:X", "track": "foundation",
              "reviewer": "codex_sol_20260713", "needs_human": False,
              "evidence": "catalog:name=X; bv=BVX"}
    for incomplete in (
        {**signed, "needs_human": True},
        {**signed, "evidence": ""},
    ):
        track_map = rec.load_track_map(raw_track_map(entities_extra=[incomplete]))
        assert "bv:X" not in track_map["entities"]
        assert "bv:X" in track_map["neutral_entities"]

    try:
        rec.load_track_map(raw_track_map(entities_extra=[
            {**signed, "reviewer": "codex_auto"},
        ]))
        assert False, "应拒绝未授权 Codex signer"
    except ValueError:
        pass


def test_11_fallback_audience_zero_upgraded():
    """全零兜底:候选全落 scene 轨(audience=0 无 gate)→ 按 Match 重排 + 理由含升轨。"""
    out = _run("高三", "review", B_U, [seg("SC1", seg_type="concept_intro")])
    assert out["status"] == "ok"
    assert out["recommendations"][0]["track_id"] == "scene"
    assert any(w for w in out["warnings"] if "upgraded" in w or "fallback" in w)


def test_12_same_track_unlock_then_gate():
    """同轨双门:unlock+gate,state=U,投影 P(M)=0.3 → 先 unlock 抬 1.0、后 gate 压 0.2。"""
    raw = raw_track_map()
    # 给 foundation 加一个 mastery_gate 造双门
    for t in raw["tracks"]:
        if t["id"] == "foundation":
            t["mastery_gate"] = 0.6
    t2 = rec.load_track_map(raw)
    w, crossed = rec.w_track("foundation", "高三", "exam_prep", "U", 0.3, t2, "full")
    assert abs(w - 0.2) < 1e-9                   # unlock→1.0 后 gate→0.2
    assert crossed is True


def test_13_multinode_entropy_priority():
    """多节点汇合:熵高者优先拿名额,每节点至少 1 段。"""
    a, b = "N_high", "N_low"
    beliefs = {a: rec.np.full(4, 0.25), b: blf(0.9, 0.04, 0.03, 0.03)}
    segs = {a: [seg("F1", seg_type="concept_intro")],
            b: [seg("F1", p=2, seg_type="review")]}
    out = rec.recommend("高二", "review", [a, b], beliefs, segs, tm(),
                        {"mode": "full", "rx_minutes": 60, "rx_segments": 3})
    nodes = [r["node"] for r in out["recommendations"]]
    assert a in nodes and b in nodes            # 每节点≥1
    assert nodes[0] == a                        # 熵高的排前


def test_14_missing_audience_rejected():
    """track_map 任一轨缺 audience → 拒绝加载。"""
    raw = raw_track_map()
    raw["tracks"].append({"id": "broken", "diagnostic_unlock": []})   # 无 audience
    try:
        rec.load_track_map(raw)
        assert False, "应拒绝缺 audience 的轨道"
    except ValueError as e:
        assert "audience" in str(e)


def test_15_efficacy_table_shapes_ranking():
    """Efficacy 传表 → 排序按乘子生效(v2 换真值零改动,公式形状先行)。"""
    good = seg("F1", p=1, seg_type="concept_intro")
    bad = seg("F1", p=2, seg_type="concept_intro")
    table = {("F1", 1): 0.1, ("F1", 2): 1.0}    # p=2 疗效高
    out = rec.recommend("高三", "exam_prep", [NODE], {NODE: B_U}, {NODE: [good, bad]},
                        tm(), {"mode": "full", "rx_minutes": 15, "rx_segments": 1},
                        efficacy_table=table)
    assert out["recommendations"][0]["p"] == 2   # 疗效高者胜


def test_16_normalize_grade_fold():
    """年级折叠:高一上/高二下→高一/高二;未知→高二+warning。"""
    assert rec.normalize_profile("高二上", "review")[0] == "高二"
    assert rec.normalize_profile("高一下", "review")[0] == "高一"
    g, _, w = rec.normalize_profile("大一", "review")
    assert g == "高二" and w


def test_17_normalize_purpose_map():
    """目的映射:薄弱突破/空串/未知 → review + warning。"""
    assert rec.normalize_profile("高二", "薄弱突破")[1] == "review"
    _, q, w = rec.normalize_profile("高二", "")
    assert q == "review" and w
    _, q2, w2 = rec.normalize_profile("高二", "随便写的")
    assert q2 == "review" and w2


def test_18_normalize_passthrough():
    """标准三值 preview/review/exam_prep 直通不变形。"""
    for p in ("preview", "review", "exam_prep"):
        g, q, w = rec.normalize_profile("高二", p)
        assert q == p and not w


def test_19_resolve_bv_over_season():
    """解析优先级:片段同时命中 bv 实体与 season 实体 → bv 精确实体生效。"""
    raw = raw_track_map(entities_extra=[{"entity": "season:99", "track": "round1",
                                         "reviewer": "user_chris", "needs_human": False,
                                         "evidence": "fixture:season-99"}])
    t = rec.load_track_map(raw)
    s = seg("F1", season_id=99)                  # bv:F1→foundation, season:99→round1
    assert rec.resolve_track(s, t) == "foundation"


def test_20_proxy_match_directionality():
    """代理映射方向性:foundation+concept 对 U 的 Match > sprint+刷题。"""
    rx = rec.prescription("U", "full")
    m_concept = rec.match(seg("F1", seg_type="concept_intro", diff="T2"), rx)
    m_drill = rec.match(seg("S1", seg_type="drill", diff="T2"), rx)
    assert m_concept > m_drill


def test_21_tiebreak_view_and_null_safe():
    """tiebreak 3/4 级:同分同日期→view 高胜;含 null 字段进排序不抛异常。"""
    hi = seg("F1", p=1, view=9999, season_order=None)
    lo = seg("F1", p=2, view=10, season_order=None)
    out = _run("高二", "review", B_U, [hi, lo],
               budget={"mode": "full", "rx_minutes": 15, "rx_segments": 1})
    assert out["recommendations"][0]["p"] == 1   # view 高胜
    # null 字段(view/season_order/pubdate 全缺)不抛
    nul = seg("F1", p=3, view=None, pubdate=None, season_order=None)
    out2 = _run("高二", "review", B_U, [nul])
    assert out2["status"] == "ok"


def test_22_six_elements_and_part_title_fallback():
    """输出六要素+rec_id;序号退化段回退 video_title。"""
    out = _run("高三", "exam_prep", B_U,
               [seg("F1", part_title="讲盐类水解", degrade=None)])
    r = out["recommendations"][0]
    for k in ("bv", "p", "start_sec", "end_sec", "track_id", "reason", "part_title", "rec_id"):
        assert k in r, f"缺六要素/rec_id 字段 {k}"
    assert r["part_title"] == "讲盐类水解"        # 正常段
    # 退化段回退 video_title
    out2 = _run("高三", "exam_prep", B_U,
                [seg("F1", part_title="P76", video_title="真·视频名", degrade="ordinal_degraded")])
    assert out2["recommendations"][0]["part_title"] == "真·视频名"


def test_23_shallow_nonM_foundation_no_unlock():
    """shallow 档:非M → foundation 概念段,不触发跨轨解锁;M → 画像轨复习段。"""
    budget = {"mode": "shallow", "rx_minutes": 8, "rx_segments": 1}
    # 非M(argmax=U)→ foundation 概念,不 crossed
    out = _run("高三", "exam_prep", B_U,
               [seg("F1", seg_type="concept_intro"), seg("S1", seg_type="drill")],
               budget=budget)
    assert out["recommendations"][0]["track_id"] == "foundation"
    assert out["rec_served"]["served"][0]["crossed_track"] is False   # shallow 不 unlock
    # M → 画像轨(高三 review round1)复习段
    out2 = rec.recommend("高三", "review", ["N"], {"N": B_M_HI},
                        {"N": [seg("R1", seg_type="review")]}, tm(),
                        {"mode": "shallow", "rx_minutes": 8, "rx_segments": 1})
    assert out2["recommendations"][0]["track_id"] == "round1"


def test_24_wiring_contract_real_fields():
    """接线契约:真实字段(高二上/薄弱突破)全链路走通,无 KeyError,输出六要素。"""
    out = _run("高二上", "薄弱突破", B_U, [seg("F1", seg_type="concept_intro")])
    assert out["status"] == "ok"
    assert out["rec_served"]["grade_norm"] == "高二"
    assert out["rec_served"]["purpose_norm"] == "review"
    assert "rec_id" in out["recommendations"][0]


def test_25_rec_served_snapshot():
    """rec_served 快照含 rec_id/分量/mode/跨轨/未服务 top-k。"""
    out = _run("高三", "exam_prep", B_U,
               [seg("F1", seg_type="concept_intro"), seg("S1", seg_type="drill")])
    snap = out["rec_served"]
    assert snap["mode"] == "full"
    s0 = snap["served"][0]
    for k in ("rec_id", "w_track", "match", "efficacy", "score", "crossed_track"):
        assert k in s0, f"快照缺分量 {k}"
    assert "unserved_topk" in snap


def test_26_log_fail_open_retry_queue():
    """日志写失败:fail-open(不抛)+响亮报错+进内存重试队列;恢复后 flush。"""
    q = []

    def bad(_rec):
        raise IOError("disk full")

    r = rec.append_rec_served({"snap": 1}, bad, q)
    assert r["ok"] is False and r["error"] is not None   # 响亮,非静默
    assert len(q) == 1                                    # 进队列,不丢

    written = []
    r2 = rec.append_rec_served({"snap": 2}, written.append, q)
    assert r2["ok"] and r2["flushed"] == 1 and len(q) == 0   # 恢复后 flush 积压


def test_27_dedup_and_exhaustion():
    """防重复:seen 排除首轮输出;全部候选已看 → 暂无视频(不崩不重复)。"""
    segs = [seg("F1", p=1, seg_type="concept_intro")]
    first = _run("高三", "exam_prep", B_U, segs)
    served = {(r["bv"], r["p"]) for r in first["recommendations"]}
    assert served
    # 传入 seen → 第二轮不重复该段(候选耗尽 → 暂无视频)
    second = _run("高三", "exam_prep", B_U, segs, seen_segments=served)
    assert second["status"] == "no_segment"
    assert second["recommendations"] == []


def test_28_track_map_rejects_duplicate_or_dangling_ids():
    duplicate_track = raw_track_map()
    duplicate_track["tracks"].append(duplicate_track["tracks"][0].copy())
    duplicate_entity = raw_track_map(entities_extra=[
        {"entity": "bv:F1", "track": "foundation", "reviewer": "user_chris",
         "needs_human": False, "evidence": "fixture:duplicate"}
    ])
    dangling = raw_track_map(entities_extra=[
        {"entity": "bv:X", "track": "missing", "reviewer": "user_chris",
         "needs_human": False, "evidence": "fixture:dangling"}
    ])
    for bad in (duplicate_track, duplicate_entity, dangling):
        try:
            rec.load_track_map(bad)
            assert False, "应拒绝重复或悬空 ID"
        except ValueError:
            pass


def test_29_track_map_and_efficacy_values_stay_in_unit_interval():
    raw = raw_track_map()
    raw["tracks"][0]["audience"]["高一"]["preview"] = 1.1
    try:
        rec.load_track_map(raw)
        assert False, "应拒绝越界 audience"
    except ValueError:
        pass
    try:
        rec.efficacy(seg("F1"), rec.prescription("U"), {("F1", 1): -0.1})
        assert False, "应拒绝越界 efficacy"
    except ValueError:
        pass


def test_30_recommend_validates_belief_shape_and_explicit_segment_ids():
    try:
        rec.recommend("高二", "review", [NODE], {NODE: [0.5, 0.5]},
                      {NODE: [seg("F1")]}, tm(),
                      {"mode": "full", "rx_minutes": 15, "rx_segments": 1})
        assert False, "应拒绝非四维 belief"
    except ValueError:
        pass

    duplicate = [{**seg("F1", p=1), "segment_id": "s1"},
                 {**seg("F1", p=2), "segment_id": "s1"}]
    try:
        _run("高二", "review", B_U, duplicate)
        assert False, "应拒绝重复 segment_id"
    except ValueError:
        pass


def test_31_same_video_part_is_never_served_twice_across_nodes():
    a, b = "A", "B"
    shared_a = {**seg("F1", p=1), "segment_id": "a-segment"}
    shared_b = {**seg("F1", p=1), "segment_id": "b-segment"}
    out = rec.recommend("高二", "review", [a, b], {a: B_U, b: B_U},
                        {a: [shared_a], b: [shared_b]}, tm(),
                        {"mode": "full", "rx_minutes": 60, "rx_segments": 2})
    served = [(row["bv"], row["p"]) for row in out["recommendations"]]
    assert len(served) == len(set(served))


def test_32_first_segment_cannot_exceed_entire_budget():
    out = _run("高二", "review", B_U, [seg("F1", dur=601)],
               budget={"mode": "full", "rx_minutes": 10, "rx_segments": 1})
    assert out["recommendations"] == []
    assert out["status"] == "no_segment"


def test_33_draft_string_tracks_expand_to_approved_five_track_defaults():
    draft = raw_track_map()
    draft["tracks"] = ["foundation", "round1", "sprint", "topical", "scene"]
    expanded = rec.load_track_map(draft)
    approved = rec.load_track_map(raw_track_map())
    assert expanded["tracks"] == approved["tracks"]
    assert set(rec.DEFAULT_TRACK_CONFIG) == {
        "foundation", "round1", "sprint", "topical", "scene",
    }
    expanded["tracks"]["foundation"]["audience"]["高一"]["preview"] = 0.0
    reloaded = rec.load_track_map({**draft, "entities": []})
    assert reloaded["tracks"]["foundation"]["audience"]["高一"]["preview"] == 1.0


def test_34_draft_string_tracks_reject_unknown_track_id():
    draft = {"version": "draft", "tracks": ["foundation", "unknown"], "entities": []}
    try:
        rec.load_track_map(draft)
        assert False, "未知字符串轨道不得被默认生成"
    except ValueError as exc:
        assert "unknown" in str(exc)


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
