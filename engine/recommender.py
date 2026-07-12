#!/usr/bin/env python3
"""
engine/recommender.py — 视频格局层评分制路由器（curriculum layer，2026-07-08）
==============================================================================
规格源（唯一权威）：~/.gstack/projects/Tools/mac-unknown-design-20260708-212231.md
  §2 评分公式 + §4 测试断言（28 条，eng-review + CEO 复审定稿）。

五段串联：画像(年级+目的) → 默认轨道 → 诊断病因 → 轨道内匹配(可跨轨) → 段落精选。
评分：Score(s) = W_track(轨道×画像×诊断解锁×掌握门) × Match(内容匹配) × Efficacy(疗效)

红线（与 selector 同款）：
  · 本模块不直读存储态 node.b——只接收调用方经 get_belief 投影后的 np.array 信念。
  · 熵函数复用 selector.entropy（log2 版），不自建自然对数版（DRY，eng-review E3）。
  · 纯函数 + 参数传表：catalog/track_map 由调用方 load_curriculum 一次加载后传入，
    落盘归接线层（引擎零 IO，eng-review E6/F1）。

工程决议（eng-review E1-E8 + CEO 复审 F1-F5，见设计文档决策表）。
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine import mastery as m
from engine import selector as sel   # 复用 entropy（E3）

# ── 画像正规化表（E1；上游 grade_detail/learning_purpose 是无校验自由文本）──
GRADE_MAP = {
    "高一": "高一", "高一上": "高一", "高一下": "高一",
    "高二": "高二", "高二上": "高二", "高二下": "高二",
    "高三": "高三",
}
PURPOSE_MAP = {
    "preview": "preview", "review": "review", "exam_prep": "exam_prep",
    "预习新课": "preview", "巩固复习": "review", "考前冲刺": "exam_prep",
    "薄弱突破": "review",              # 已有测试锁定的自由值 → review
}
DEFAULT_GRADE = "高二"
DEFAULT_PURPOSE = "review"

# ── 处方段类型族（v1 代理字段，E2；seg_type 来自视频级 type）──
CONCEPT_TYPES = {"concept_intro", "concept", "知识点串讲"}
DRILL_TYPES = {"method", "exercise", "problem", "drill", "题刷刷", "刷题"}
REVIEW_TYPES = {"review", "advanced", "复习", "拔高"}

# ── 难度档位（chunk difficulty_tier 众数的代理）──
_TIER = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}

# ── 轨道展示名（输出理由用人话，CEO 复审 #4）──
TRACK_DISPLAY = {
    "foundation": "基础大合集", "round1": "一轮复习", "sprint": "刷题冲刺",
    "topical": "专项突破", "scene": "场景特供",
}

EPSILON = 0.05                        # 平局分桶粒度
TOPK = 3                              # 每节点候选 top-k
DEFAULT_MODE = "full"
AUTHORIZED_CODEX_REVIEWER = "codex_sol_20260713"


def _unit_interval(value, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须为 [0,1] 数值") from exc
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} 必须在 [0,1]")
    return numeric


# ══════════════════════════════════════════════════════════════════════
# 画像正规化（E1；recommender 入口必经，脏值挡在门口）
# ══════════════════════════════════════════════════════════════════════
def normalize_profile(grade, learning_purpose) -> Tuple[str, str, List[str]]:
    """(年级, 目的) → (g, q, warnings)。未知值安全默认 + warning，绝不 KeyError。"""
    warnings: List[str] = []
    g = GRADE_MAP.get((grade or "").strip())
    if g is None:
        warnings.append(f"unknown grade {grade!r} → {DEFAULT_GRADE}")
        g = DEFAULT_GRADE
    q = PURPOSE_MAP.get((learning_purpose or "").strip())
    if q is None:
        warnings.append(f"unknown purpose {learning_purpose!r} → {DEFAULT_PURPOSE}")
        q = DEFAULT_PURPOSE
    return g, q, warnings


# ══════════════════════════════════════════════════════════════════════
# 课程表加载 + 校验（E6 参数传表；断言 10/14）
# ══════════════════════════════════════════════════════════════════════
def load_track_map(raw: Dict) -> Dict:
    """把 YAML 原始结构（tracks/entities 为列表）正规化为 O(1) 查表 dict，并校验：
       · 任一轨道缺 audience 块 → ValueError（断言 14）。
       · 重复/悬空 ID、越界权重、未完成的 Codex 签字 → ValueError。"""
    tracks: Dict[str, Dict] = {}
    for t in raw.get("tracks", []):
        tid = t["id"]
        if tid in tracks:
            raise ValueError(f"重复 track id: {tid!r}")
        if "audience" not in t or not t["audience"]:
            raise ValueError(f"track {tid!r} 缺 audience 块，拒绝加载")
        for grade, purposes in t["audience"].items():
            if not isinstance(purposes, dict) or not purposes:
                raise ValueError(f"track {tid!r} audience[{grade!r}] 非法")
            for purpose, value in purposes.items():
                _unit_interval(value, f"track {tid!r} audience[{grade!r}][{purpose!r}]")
        if "efficacy" in t:
            _unit_interval(t["efficacy"], f"track {tid!r} efficacy")
        tracks[tid] = {
            "audience": t["audience"],
            "diagnostic_unlock": list(t.get("diagnostic_unlock", [])),
            "mastery_gate": t.get("mastery_gate"),
        }
    entities: Dict[str, Dict] = {}
    for e in raw.get("entities", []):
        entity_id = e["entity"]
        if entity_id in entities:
            raise ValueError(f"重复 entity id: {entity_id!r}")
        if e["track"] not in tracks:
            raise ValueError(f"entity {entity_id!r} 引用不存在的 track {e['track']!r}")
        reviewer = str(e.get("reviewer", ""))
        if reviewer.startswith("codex_") and reviewer != AUTHORIZED_CODEX_REVIEWER:
            raise ValueError(f"entity {entity_id!r} reviewer={reviewer} 未授权")
        if reviewer == AUTHORIZED_CODEX_REVIEWER:
            if e.get("needs_human") is not False:
                raise ValueError(f"entity {entity_id!r} 仍 needs_human，拒绝加载")
            if not str(e.get("evidence", "")).strip():
                raise ValueError(f"entity {entity_id!r} 缺签字证据")
        if "efficacy" in e:
            _unit_interval(e["efficacy"], f"entity {entity_id!r} efficacy")
        entities[entity_id] = {
            "track": e["track"], "reviewer": reviewer,
            "needs_human": e.get("needs_human"), "evidence": e.get("evidence"),
        }
    return {"tracks": tracks, "entities": entities,
            "version": raw.get("version", "curriculum_v1")}


# ══════════════════════════════════════════════════════════════════════
# 片段 → 轨道解析（E4；断言 19：bv 精确 > season > foundation 回退）
# ══════════════════════════════════════════════════════════════════════
def resolve_track(segment: Dict, track_map: Dict, warnings: Optional[List[str]] = None) -> str:
    """解析优先级：bv 精确实体 > season 实体 > foundation 回退（+warning，断言 6）。"""
    ents = track_map["entities"]
    bv_key = "bv:" + str(segment.get("bv"))
    if bv_key in ents:
        return ents[bv_key]["track"]
    sid = segment.get("season_id")
    if sid is not None:
        s_key = "season:" + str(sid)
        if s_key in ents:
            return ents[s_key]["track"]
    if warnings is not None:
        warnings.append(f"segment {segment.get('bv')}#{segment.get('p')} 无实体映射 → foundation 回退")
    return "foundation"


# ══════════════════════════════════════════════════════════════════════
# 状态判据 + 处方分叉（3A；shallow 二分，E7）
# ══════════════════════════════════════════════════════════════════════
def node_state(b_n: np.ndarray, mode: str = DEFAULT_MODE) -> str:
    """节点状态。full 档：argmax → M/P/C/U。shallow 档：M/非M 二分（E7）。"""
    idx = int(np.argmax(np.asarray(b_n, dtype=float)))
    if mode == "shallow":
        return "M" if idx == m.M else "nonM"
    return m.STATES[idx]


def prescription(state: str, mode: str = DEFAULT_MODE) -> Dict:
    """病因 → 需要的段类型/难度/来源节点（设计文档第一步）。
       shallow 非M 强制 foundation 概念段（保守安全默认，不触发 unlock，E7）。"""
    if mode == "shallow":
        if state == "M":
            return {"source": "self", "type_pref": REVIEW_TYPES, "diff": "any", "force_track": None}
        return {"source": "self", "type_pref": CONCEPT_TYPES, "diff": "low", "force_track": "foundation"}
    return {
        "P": {"source": "prereq", "type_pref": CONCEPT_TYPES, "diff": "low", "force_track": None},
        "U": {"source": "self", "type_pref": CONCEPT_TYPES, "diff": "low", "force_track": None},
        "C": {"source": "self", "type_pref": DRILL_TYPES, "diff": "mid", "force_track": None},
        "M": {"source": "self", "type_pref": REVIEW_TYPES, "diff": "any", "force_track": None},
    }[state]


# ══════════════════════════════════════════════════════════════════════
# W_track（三步定序：基础权重 → unlock 抬升 → gate 压制）
# ══════════════════════════════════════════════════════════════════════
def _base_weight(track_id: str, g: str, q: str, track_map: Dict) -> float:
    aud = track_map["tracks"][track_id]["audience"]
    return float(aud.get(g, {}).get(q, 0.0))


def gate_blocked(track_id: str, pm_proj: float, track_map: Dict) -> bool:
    """掌握门是否压制该轨道（投影态 P(M) < 门槛）。兜底硬排除据此判断（断言 11）。"""
    gate = track_map["tracks"][track_id].get("mastery_gate")
    return gate is not None and pm_proj < gate


def w_track(track_id: str, g: str, q: str, state: str, pm_proj: float,
            track_map: Dict, mode: str) -> Tuple[float, bool]:
    """返回 (最终权重, 是否跨轨)。三步：基础 → unlock（仅 full）→ gate。
       跨轨 = full 档 unlock 把基础权重<1.0 的轨道抬到 1.0（断言 1/4/8/12）。"""
    track = track_map["tracks"][track_id]
    base = _base_weight(track_id, g, q, track_map)
    w = base
    crossed = False
    if mode != "shallow" and state in track["diagnostic_unlock"]:
        if base < 1.0:
            crossed = True
        w = max(w, 1.0)                        # unlock 抬升
    gate = track.get("mastery_gate")
    if gate is not None and pm_proj < gate:
        w = min(w, 0.2)                        # gate 终裁（断言 12：先 unlock 后 gate）
    return w, crossed


# ══════════════════════════════════════════════════════════════════════
# Match（v1 代理字段）+ Efficacy（v1 占位）
# ══════════════════════════════════════════════════════════════════════
def _difficulty_tier(d) -> int:
    if isinstance(d, (int, float)):
        return int(d)
    s = str(d).upper()
    for k, v in _TIER.items():
        if k in s:
            return v
    return 2


def _type_fit(seg_type: str, type_pref) -> float:
    """精确匹配 1.0；同族 0.6；跨族(概念↔刷题) 0.3。"""
    if seg_type in type_pref:
        return 1.0
    seg_concept = seg_type in CONCEPT_TYPES
    want_concept = any(t in CONCEPT_TYPES for t in type_pref)
    seg_drill = seg_type in DRILL_TYPES
    want_drill = any(t in DRILL_TYPES for t in type_pref)
    if (seg_concept and want_drill) or (seg_drill and want_concept):
        return 0.3                            # 跨族错配
    return 0.6                                # 中性/同族


def _diff_fit(tier: int, pref: str) -> float:
    if pref == "any":
        return 1.0
    if pref == "low":
        return {1: 1.0, 2: 1.0, 3: 0.6, 4: 0.3}.get(tier, 0.6)
    if pref == "mid":
        return {1: 0.6, 2: 1.0, 3: 1.0, 4: 0.6}.get(tier, 0.6)
    return 0.6


def match(segment: Dict, rx: Dict) -> float:
    """topic_match_ratio × 类型匹配 × 难度匹配 ∈ [0,1]（断言 20 方向性）。"""
    tmr = float(segment.get("topic_match_ratio", segment.get("checks", {}).get("topic_match_ratio", 0.0)))
    tf = _type_fit(segment.get("seg_type", ""), rx["type_pref"])
    df = _diff_fit(_difficulty_tier(segment.get("difficulty", "T2")), rx["diff"])
    return tmr * tf * df


def efficacy(segment: Dict, rx: Dict, table: Optional[Dict] = None) -> float:
    """v1 恒 1.0 占位（E2/断言 9）。传入 table 时按 (bv,p) 取后验均值（断言 15，公式形状先行）。"""
    if table is None:
        return 1.0
    value = table.get((segment.get("bv"), segment.get("p")), 1.0)
    return _unit_interval(value, f"efficacy[{segment.get('bv')!r},{segment.get('p')!r}]")


# ══════════════════════════════════════════════════════════════════════
# 单段评分
# ══════════════════════════════════════════════════════════════════════
def _score_one(segment: Dict, g: str, q: str, b_n: np.ndarray, state: str,
               track_map: Dict, mode: str, rx: Dict,
               efficacy_table: Optional[Dict]) -> Dict:
    warns: List[str] = []
    tid = resolve_track(segment, track_map, warns)
    pm_proj = float(np.asarray(b_n, dtype=float)[m.M])
    w, crossed = w_track(tid, g, q, state, pm_proj, track_map, mode)
    # shallow force_track：非 foundation 段直接压零（E7，非 unlock 路径）
    if rx.get("force_track") and tid != rx["force_track"]:
        w = 0.0
        crossed = False
    mt = match(segment, rx)
    ef = efficacy(segment, rx, efficacy_table)
    return {"track_id": tid, "w_track": w, "match": mt, "efficacy": ef,
            "score": w * mt * ef, "crossed": crossed, "state": state,
            "gate_blocked": gate_blocked(tid, pm_proj, track_map), "warns": warns}


# ══════════════════════════════════════════════════════════════════════
# 平局排序键（哨兵值容错，F/断言 5/21）
# ══════════════════════════════════════════════════════════════════════
def _pubdate_ts(segment: Dict) -> int:
    """'2023-09-01' → 20230901；缺失 → 0（最旧，新版优先的反向安全面）。"""
    p = segment.get("pubdate")
    if not p:
        return 0
    try:
        return int(str(p).replace("-", "")[:8])
    except (ValueError, TypeError):
        return 0


def _sort_key(entry: Tuple[Dict, float, Dict]):
    """(−Score_bucket, −pubdate, −view, season_order)。null 字段用哨兵，排序永不抛异常。"""
    seg, score, _ = entry
    bucket = round(score / EPSILON)
    view = seg.get("view") or 0
    so = seg.get("season_order")
    so = float("inf") if so is None else so
    return (-bucket, -_pubdate_ts(seg), -view, so)


# ══════════════════════════════════════════════════════════════════════
# 全零兜底（降级阶梯，断言 11）
# ══════════════════════════════════════════════════════════════════════
def _fallback(scored: List[Tuple[Dict, float, Dict]], track_map: Dict,
              b_n: np.ndarray, warnings: List[str], node: str
              ) -> List[Tuple[Dict, float, Dict]]:
    """某节点全员 Score=0：只重纳「audience 归零、非 gate 压制」的轨道，按 Match 重排。
       gate 压制的轨道依然排除（掌握不足的后门不开）。"""
    pm_proj = float(np.asarray(b_n, dtype=float)[m.M])
    readmit = []
    for seg, sc, comp in scored:
        if comp["gate_blocked"]:
            continue                          # 硬排除，不豁免
        # 升轨：忽略 W_track，按 Match 重排；标记跨轨以便理由说明
        comp2 = dict(comp)
        comp2["crossed"] = True
        comp2["upgraded"] = True
        readmit.append((seg, comp["match"], comp2))
    if readmit:
        warnings.append(f"节点 {node} 全零兜底：{len(readmit)} 段升轨（缺基础轨讲解）")
    return readmit


# ══════════════════════════════════════════════════════════════════════
# 理由构建（表达层，模板拼装；跨轨强制含轨道名，断言 8/22）
# ══════════════════════════════════════════════════════════════════════
def _part_title(segment: Dict) -> str:
    """part_title 正常返回；序号/重复退化 → 回退 video_title（断言 22）。"""
    pt = segment.get("part_title")
    deg = segment.get("part_degrade_state")
    if not pt or deg in ("ordinal_degraded", "duplicate_degraded"):
        return segment.get("video_title") or pt or ""
    return pt


def _build_reason(segment: Dict, comp: Dict, budget_left_min: int) -> str:
    parts: List[str] = []
    tname = TRACK_DISPLAY.get(comp["track_id"], comp["track_id"])
    if comp.get("crossed"):
        parts.append(f"你这个点还没打牢，先回「{tname}」把这节看了再往下——冲刺题做了也白做。")
    st = comp.get("state")
    if st and st not in ("M", "nonM"):
        parts.append(f"诊断显示你在这个点卡在「{st}」。")
    sn, so = segment.get("season_name"), segment.get("season_order")
    if sn and so is not None:
        parts.append(f"去《{sn}》第{so}讲。")
    dur_min = round((segment.get("duration_sec") or 0) / 60)
    parts.append(f"这段约{dur_min}分钟，今天还剩约{budget_left_min}分钟。")
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════
# 主入口：recommend（五段串联 + 多节点汇合 + rec_served 快照）
# ══════════════════════════════════════════════════════════════════════
def recommend(grade, learning_purpose, target_nodes: Sequence[str],
              beliefs_proj: Dict[str, np.ndarray], segments_by_node: Dict[str, List[Dict]],
              track_map: Dict, budget: Dict, *,
              seen_segments: Optional[set] = None, prereq_map: Optional[Dict] = None,
              efficacy_table: Optional[Dict] = None,
              rec_id_factory=lambda: uuid.uuid4().hex) -> Dict:
    """返回 {recommendations, rec_served, warnings, status}。
       beliefs_proj：调用方已用 get_belief 投影的 {node: np.array}（红线：本模块不碰 .b）。
       budget：planner.session_budget(tier)。seen_segments：已推 (bv,p) 集合（防重复，断言 27）。"""
    g, q, warnings = normalize_profile(grade, learning_purpose)
    mode = budget.get("mode", DEFAULT_MODE)
    rx_minutes = int(budget.get("rx_minutes", 15))
    rx_segments = int(budget.get("rx_segments", 2))
    seen = set(seen_segments or ())
    prereq_map = prereq_map or {}

    segment_ids = set()
    for pool in segments_by_node.values():
        for segment in pool:
            segment_id = segment.get("segment_id")
            if segment_id is None:
                continue
            if segment_id in segment_ids:
                raise ValueError(f"重复 segment_id: {segment_id!r}")
            segment_ids.add(segment_id)

    # 1. 每节点评分 + 熵
    node_pools: Dict[str, List[Tuple[Dict, float, Dict]]] = {}
    node_entropy: Dict[str, float] = {}
    for node in target_nodes:
        try:
            b = np.asarray(beliefs_proj[node], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"节点 {node!r} belief 无效") from exc
        if (b.shape != (4,) or not np.all(np.isfinite(b)) or np.any(b < 0)
                or not np.isclose(b.sum(), 1.0, rtol=0.0, atol=1e-6)):
            raise ValueError(f"节点 {node!r} belief 必须是四维概率分布")
        node_entropy[node] = sel.entropy(b)
        state = node_state(b, mode)
        rx = prescription(state, mode)
        src = prereq_map.get(node) if (rx["source"] == "prereq" and node in prereq_map) else node
        pool = [s for s in segments_by_node.get(src, [])
                if (s.get("bv"), s.get("p")) not in seen]
        scored = []
        for s in pool:
            comp = _score_one(s, g, q, b, state, track_map, mode, rx, efficacy_table)
            warnings.extend(comp.pop("warns"))
            scored.append((s, comp["score"], comp))
        # 全零兜底
        if scored and all(sc <= 0 for _, sc, _ in scored):
            scored = _fallback(scored, track_map, b, warnings, node)
        scored.sort(key=_sort_key)
        node_pools[node] = scored[:TOPK]

    # 2. 多节点汇合：段名额按熵降序分配，每节点≥1（断言 13）
    order = sorted(target_nodes, key=lambda n: (-node_entropy[n], n))
    counts = {n: 0 for n in order}
    slots = rx_segments
    for n in order:                            # 第一轮：熵序每节点 1 段
        if slots <= 0:
            break
        if node_pools[n]:
            counts[n] = 1
            slots -= 1
    i = 0
    while slots > 0 and any(counts[n] < len(node_pools[n]) for n in order):
        n = order[i % len(order)]              # 富余给熵最高者（2h/3h+ 档）
        if counts[n] < len(node_pools[n]):
            counts[n] += 1
            slots -= 1
        i += 1
        if i > 4096:
            break

    # 3. 全局排序（熵降序 → 节点内 Score 降序）+ 预算硬约束
    picks: List[Tuple[str, Dict, float, Dict]] = []
    for n in order:
        for seg, sc, comp in node_pools[n][:counts[n]]:
            picks.append((n, seg, sc, comp))
    picks.sort(key=lambda x: (-node_entropy[x[0]], -x[2]))

    recommendations: List[Dict] = []
    served_snap: List[Dict] = []
    budget_sec = rx_minutes * 60
    used_sec = 0
    served_ids = set()
    for n, seg, sc, comp in picks:
        dur = int(seg.get("duration_sec") or 0)
        segment_key = (seg.get("bv"), seg.get("p"))
        if segment_key in served_ids:
            continue
        if used_sec + dur > budget_sec:
            continue                           # 超预算截断（宁缺勿滥，断言 7）
        used_sec += dur
        left_min = max(0, (budget_sec - used_sec) // 60)
        rid = rec_id_factory()
        rec = {
            "rec_id": rid, "node": n,
            "bv": seg.get("bv"), "p": seg.get("p"),
            "start_sec": seg.get("start_sec"), "end_sec": seg.get("end_sec"),
            "track_id": comp["track_id"],
            "part_title": _part_title(seg),
            "reason": _build_reason(seg, comp, left_min),
        }
        recommendations.append(rec)
        served_ids.add(segment_key)
        served_snap.append({
            "rec_id": rid, "bv": seg.get("bv"), "p": seg.get("p"), "node": n,
            "track_id": comp["track_id"], "w_track": comp["w_track"],
            "match": comp["match"], "efficacy": comp["efficacy"], "score": comp["score"],
            "crossed_track": comp["crossed"],
        })

    # 4. 未服务 top-k 快照（护城河原料，断言 25）
    unserved = []
    for n in order:
        for seg, sc, comp in node_pools[n]:
            if (seg.get("bv"), seg.get("p")) not in served_ids:
                unserved.append({"bv": seg.get("bv"), "p": seg.get("p"),
                                 "node": n, "score": sc})
    unserved.sort(key=lambda x: -x["score"])

    status = "ok" if recommendations else "no_segment"
    if status == "no_segment":
        warnings.append("本次无合适视频段（候选耗尽或全被 gate 排除）")

    rec_served = {"mode": mode, "grade_norm": g, "purpose_norm": q,
                  "served": served_snap, "unserved_topk": unserved[:TOPK]}
    return {"recommendations": recommendations, "rec_served": rec_served,
            "warnings": warnings, "status": status,
            "message": "本节点暂无合适视频" if status == "no_segment" else ""}


# ══════════════════════════════════════════════════════════════════════
# rec_served 落盘（接线层用；fail-open + 内存重试队列，F2/F5/断言 26）
# ══════════════════════════════════════════════════════════════════════
def append_rec_served(snapshot: Dict, writer, retry_queue: List, *,
                      queue_alert_threshold: int = 10,
                      max_queue_size: int = 100) -> Dict:
    """接线层调用。writer(record) 可能抛异常（磁盘满/权限）。
       fail-open：写失败不阻断推荐服务；瞬时故障进内存重试队列，先 flush 积压再写本条；
       持久故障丢数据但返回 error 可见（零静默失败）。返回 {ok, flushed, queued, error}。"""
    result = {"ok": False, "flushed": 0, "queued": len(retry_queue),
              "dropped": 0, "error": None}
    # 先尝试 flush 积压（瞬时故障恢复后不丢）
    still_queued = []
    for rec in retry_queue:
        try:
            writer(rec)
            result["flushed"] += 1
        except Exception:                      # noqa: BLE001 — fail-open 边界，下游可见
            still_queued.append(rec)
    retry_queue[:] = still_queued
    # 写本条
    try:
        writer(snapshot)
        result["ok"] = True
    except Exception as e:                      # noqa: BLE001
        if snapshot not in retry_queue:
            retry_queue.append(snapshot)
        limit = max(0, int(max_queue_size))
        overflow = max(0, len(retry_queue) - limit)
        if overflow:
            del retry_queue[:overflow]
            result["dropped"] = overflow
        result["error"] = f"{type(e).__name__}: {e}"   # 响亮报错，非静默吞
    result["queued"] = len(retry_queue)
    if result["queued"] > queue_alert_threshold:
        result["error"] = (result["error"] or "") + f" | 重试队列积压 {result['queued']} 条 [ERROR]"
    return result
