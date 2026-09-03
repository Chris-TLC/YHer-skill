#!/usr/bin/env python3
"""
engine/recommender.py — score-based router for the video landscape layer (curriculum
layer, 2026-07-08)
==============================================================================
Spec source (sole authority): ~/.gstack/projects/Tools/mac-unknown-design-20260708-212231.md
  §2 scoring formula + §4 test assertions (28 items, finalized by eng-review + CEO re-review).

Five-stage chain: profile (grade + purpose) → default track → diagnosed cause →
in-track matching (may cross tracks) → segment selection.
Scoring: Score(s) = W_track (track × profile × diagnostic unlock × mastery gate) × Match (content match) × Efficacy

Red lines (same flavor as selector):
  · This module never reads the stored belief state directly — it only receives
    the projected np.array beliefs from the caller via get_belief.
  · The entropy function is reused from selector.entropy (log2 version); no
    self-built natural-log version (DRY, eng-review E3).
  · Pure functions + tables passed as parameters: catalog/track_map are loaded
    once by the caller via load_curriculum and passed in; persistence belongs to
    the wiring layer (engine does zero IO, eng-review E6/F1).

Engineering resolutions (eng-review E1-E8 + CEO re-review F1-F5; see the design
doc's decision table).
"""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine import mastery as m
from engine import selector as sel   # reuse entropy (E3)

# ── Profile normalization tables (E1; upstream grade_detail/learning_purpose are free text without validation) ──
GRADE_MAP = {
    "高一": "高一", "高一上": "高一", "高一下": "高一",
    "高二": "高二", "高二上": "高二", "高二下": "高二",
    "高三": "高三",
}
PURPOSE_MAP = {
    "preview": "preview", "review": "review", "exam_prep": "exam_prep",
    "预习新课": "preview", "巩固复习": "review", "考前冲刺": "exam_prep",
    "薄弱突破": "review",              # free value already locked in by existing tests → review
}
DEFAULT_GRADE = "高二"
DEFAULT_PURPOSE = "review"

# ── Prescription segment-type families (v1 proxy fields, E2; seg_type comes from the video-level type) ──
CONCEPT_TYPES = {"concept_intro", "concept", "知识点串讲"}
DRILL_TYPES = {"method", "exercise", "problem", "drill", "题刷刷", "刷题"}
REVIEW_TYPES = {"review", "advanced", "复习", "拔高"}

# ── Difficulty tiers (proxy for the mode of chunk difficulty_tier) ──
_TIER = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}

# ── Track display names (plain-language reasons in output, CEO re-review #4) ──
TRACK_DISPLAY = {
    "foundation": "基础大合集", "round1": "一轮复习", "sprint": "刷题冲刺",
    "topical": "专项突破", "scene": "场景特供",
}

EPSILON = 0.05                        # tie-bucketing granularity
TOPK = 3                              # top-k candidates per node
DEFAULT_MODE = "full"
AUTHORIZED_CODEX_REVIEWER = "codex_sol_20260713"

# The real curriculum draft only lists track IDs; these are the approved v1 routing parameters.
DEFAULT_TRACK_CONFIG = {
    "foundation": {
        "audience": {
            "高一": {"preview": 1.0, "review": 1.0, "exam_prep": 0.6},
            "高二": {"preview": 1.0, "review": 1.0, "exam_prep": 0.6},
            "高三": {"preview": 0.6, "review": 0.4, "exam_prep": 0.2},
        },
        "diagnostic_unlock": ["U", "P"],
    },
    "round1": {
        "audience": {
            "高一": {"preview": 0.0, "review": 0.2, "exam_prep": 0.2},
            "高二": {"preview": 0.2, "review": 0.6, "exam_prep": 0.6},
            "高三": {"preview": 0.6, "review": 1.0, "exam_prep": 1.0},
        },
        "mastery_gate": 0.6,
    },
    "sprint": {
        "audience": {
            "高一": {"preview": 0.0, "review": 0.2, "exam_prep": 0.6},
            "高二": {"preview": 0.0, "review": 0.2, "exam_prep": 0.8},
            "高三": {"preview": 0.2, "review": 0.6, "exam_prep": 1.0},
        },
        "mastery_gate": 0.6,
    },
    "topical": {
        "audience": {
            "高一": {"preview": 0.6, "review": 0.8, "exam_prep": 0.6},
            "高二": {"preview": 0.6, "review": 0.8, "exam_prep": 0.8},
            "高三": {"preview": 0.6, "review": 0.8, "exam_prep": 0.8},
        },
    },
    "scene": {
        "audience": {
            "高一": {"preview": 0.0, "review": 0.0, "exam_prep": 0.0},
            "高二": {"preview": 0.0, "review": 0.0, "exam_prep": 0.0},
            "高三": {"preview": 0.0, "review": 0.0, "exam_prep": 0.0},
        },
    },
}


def _unit_interval(value, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number in [0,1]") from exc
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return numeric


# ══════════════════════════════════════════════════════════════════════
# Profile normalization (E1; every recommender entry passes through it, dirty values are stopped at the door)
# ══════════════════════════════════════════════════════════════════════
def normalize_profile(grade, learning_purpose) -> Tuple[str, str, List[str]]:
    """(grade, purpose) → (g, q, warnings). Unknown values get a safe default + warning; never raises KeyError."""
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
# Curriculum loading + validation (E6 tables passed as parameters; assertions 10/14)
# ══════════════════════════════════════════════════════════════════════
def load_track_map(raw: Dict) -> Dict:
    """Normalize the raw YAML structure (tracks/entities as lists) into an O(1) lookup dict, and validate:
       · Any track missing its audience block → ValueError (assertion 14).
       · Duplicate/dangling IDs, out-of-range weights, unauthorized Codex signers → ValueError.
       · Entities with incomplete signatures go into neutral_entities and stay out of track routing."""
    tracks: Dict[str, Dict] = {}
    for track_row in raw.get("tracks", []):
        if isinstance(track_row, str):
            if track_row not in DEFAULT_TRACK_CONFIG:
                raise ValueError(f"unknown track id: {track_row!r}")
            t = {"id": track_row, **copy.deepcopy(DEFAULT_TRACK_CONFIG[track_row])}
        elif isinstance(track_row, dict):
            t = track_row
        else:
            raise ValueError(f"track must be an ID string or a full object, got {type(track_row).__name__}")
        tid = t["id"]
        if tid in tracks:
            raise ValueError(f"duplicate track id: {tid!r}")
        if "audience" not in t or not t["audience"]:
            raise ValueError(f"track {tid!r} is missing its audience block; refusing to load")
        for grade, purposes in t["audience"].items():
            if not isinstance(purposes, dict) or not purposes:
                raise ValueError(f"track {tid!r} audience[{grade!r}] is invalid")
            for purpose, value in purposes.items():
                _unit_interval(value, f"track {tid!r} audience[{grade!r}][{purpose!r}]")
        if "efficacy" in t:
            _unit_interval(t["efficacy"], f"track {tid!r} efficacy")
        tracks[tid] = {
            "audience": copy.deepcopy(t["audience"]),
            "diagnostic_unlock": list(t.get("diagnostic_unlock", [])),
            "mastery_gate": t.get("mastery_gate"),
        }
    entities: Dict[str, Dict] = {}
    neutral_entities: Dict[str, Dict] = {}
    seen_entity_ids = set()
    for e in raw.get("entities", []):
        entity_id = e["entity"]
        if entity_id in seen_entity_ids:
            raise ValueError(f"duplicate entity id: {entity_id!r}")
        seen_entity_ids.add(entity_id)
        if e["track"] not in tracks:
            raise ValueError(f"entity {entity_id!r} references nonexistent track {e['track']!r}")
        reviewer = str(e.get("reviewer") or "").strip()
        if reviewer.startswith("codex_") and reviewer != AUTHORIZED_CODEX_REVIEWER:
            raise ValueError(f"entity {entity_id!r} reviewer={reviewer} is unauthorized")
        if "efficacy" in e:
            _unit_interval(e["efficacy"], f"entity {entity_id!r} efficacy")
        evidence = e.get("evidence")
        signature_complete = (
            bool(reviewer)
            and e.get("needs_human") is False
            and bool(evidence)
            and (not isinstance(evidence, str) or bool(evidence.strip()))
        )
        normalized = {
            "track": e["track"], "reviewer": reviewer,
            "needs_human": e.get("needs_human"), "evidence": evidence,
        }
        if signature_complete:
            entities[entity_id] = normalized
        else:
            neutral_entities[entity_id] = normalized
    return {"tracks": tracks, "entities": entities,
            "neutral_entities": neutral_entities,
            "version": raw.get("version", "curriculum_v1")}


# ══════════════════════════════════════════════════════════════════════
# Segment → track resolution (E4; assertion 19: exact bv > season > foundation fallback)
# ══════════════════════════════════════════════════════════════════════
def resolve_track(segment: Dict, track_map: Dict, warnings: Optional[List[str]] = None) -> str:
    """Resolution priority: exact bv entity > season entity > foundation fallback (+warning, assertion 6)."""
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
        warnings.append(f"segment {segment.get('bv')}#{segment.get('p')} has no entity mapping → foundation fallback")
    return "foundation"


# ══════════════════════════════════════════════════════════════════════
# State criterion + prescription fork (3A; shallow binary split, E7)
# ══════════════════════════════════════════════════════════════════════
def node_state(b_n: np.ndarray, mode: str = DEFAULT_MODE) -> str:
    """Node state. full tier: argmax → M/P/C/U. shallow tier: M/non-M binary split (E7)."""
    idx = int(np.argmax(np.asarray(b_n, dtype=float)))
    if mode == "shallow":
        return "M" if idx == m.M else "nonM"
    return m.STATES[idx]


def prescription(state: str, mode: str = DEFAULT_MODE) -> Dict:
    """Cause → needed segment type / difficulty / source node (design doc step 1).
       shallow non-M forces foundation concept segments (conservative safe default, no unlock triggered, E7)."""
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
# W_track (three ordered steps: base weight → unlock boost → gate suppression)
# ══════════════════════════════════════════════════════════════════════
def _base_weight(track_id: str, g: str, q: str, track_map: Dict) -> float:
    aud = track_map["tracks"][track_id]["audience"]
    return float(aud.get(g, {}).get(q, 0.0))


def gate_blocked(track_id: str, pm_proj: float, track_map: Dict) -> bool:
    """Whether the mastery gate suppresses this track (projected P(M) < threshold). The fallback hard exclusion relies on this (assertion 11)."""
    gate = track_map["tracks"][track_id].get("mastery_gate")
    return gate is not None and pm_proj < gate


def w_track(track_id: str, g: str, q: str, state: str, pm_proj: float,
            track_map: Dict, mode: str) -> Tuple[float, bool]:
    """Returns (final weight, whether the track was crossed). Three steps: base → unlock (full only) → gate.
       Cross-track = a full-tier unlock lifts a track whose base weight <1.0 up to 1.0 (assertions 1/4/8/12)."""
    track = track_map["tracks"][track_id]
    base = _base_weight(track_id, g, q, track_map)
    w = base
    crossed = False
    if mode != "shallow" and state in track["diagnostic_unlock"]:
        if base < 1.0:
            crossed = True
        w = max(w, 1.0)                        # unlock boost
    gate = track.get("mastery_gate")
    if gate is not None and pm_proj < gate:
        w = min(w, 0.2)                        # gate final verdict (assertion 12: unlock first, gate after)
    return w, crossed


# ══════════════════════════════════════════════════════════════════════
# Match (v1 proxy fields) + Efficacy (v1 placeholder)
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
    """Exact match 1.0; same family 0.6; cross-family (concept ↔ drill) 0.3."""
    if seg_type in type_pref:
        return 1.0
    seg_concept = seg_type in CONCEPT_TYPES
    want_concept = any(t in CONCEPT_TYPES for t in type_pref)
    seg_drill = seg_type in DRILL_TYPES
    want_drill = any(t in DRILL_TYPES for t in type_pref)
    if (seg_concept and want_drill) or (seg_drill and want_concept):
        return 0.3                            # cross-family mismatch
    return 0.6                                # neutral / same family


def _diff_fit(tier: int, pref: str) -> float:
    if pref == "any":
        return 1.0
    if pref == "low":
        return {1: 1.0, 2: 1.0, 3: 0.6, 4: 0.3}.get(tier, 0.6)
    if pref == "mid":
        return {1: 0.6, 2: 1.0, 3: 1.0, 4: 0.6}.get(tier, 0.6)
    return 0.6


def match(segment: Dict, rx: Dict) -> float:
    """topic_match_ratio × type match × difficulty match ∈ [0,1] (assertion 20 directionality)."""
    tmr = float(segment.get("topic_match_ratio", segment.get("checks", {}).get("topic_match_ratio", 0.0)))
    tf = _type_fit(segment.get("seg_type", ""), rx["type_pref"])
    df = _diff_fit(_difficulty_tier(segment.get("difficulty", "T2")), rx["diff"])
    return tmr * tf * df


def efficacy(segment: Dict, rx: Dict, table: Optional[Dict] = None) -> float:
    """v1 constant-1.0 placeholder (E2/assertion 9). With a table passed in, look up the posterior mean by (bv,p) (assertion 15, formula shape first)."""
    if table is None:
        return 1.0
    value = table.get((segment.get("bv"), segment.get("p")), 1.0)
    return _unit_interval(value, f"efficacy[{segment.get('bv')!r},{segment.get('p')!r}]")


# ══════════════════════════════════════════════════════════════════════
# Single-segment scoring
# ══════════════════════════════════════════════════════════════════════
def _score_one(segment: Dict, g: str, q: str, b_n: np.ndarray, state: str,
               track_map: Dict, mode: str, rx: Dict,
               efficacy_table: Optional[Dict]) -> Dict:
    warns: List[str] = []
    tid = resolve_track(segment, track_map, warns)
    pm_proj = float(np.asarray(b_n, dtype=float)[m.M])
    w, crossed = w_track(tid, g, q, state, pm_proj, track_map, mode)
    # shallow force_track: non-foundation segments are zeroed directly (E7, not an unlock path)
    if rx.get("force_track") and tid != rx["force_track"]:
        w = 0.0
        crossed = False
    mt = match(segment, rx)
    ef = efficacy(segment, rx, efficacy_table)
    return {"track_id": tid, "w_track": w, "match": mt, "efficacy": ef,
            "score": w * mt * ef, "crossed": crossed, "state": state,
            "gate_blocked": gate_blocked(tid, pm_proj, track_map), "warns": warns}


# ══════════════════════════════════════════════════════════════════════
# Tie-break sort key (sentinel values for fault tolerance, F/assertions 5/21)
# ══════════════════════════════════════════════════════════════════════
def _pubdate_ts(segment: Dict) -> int:
    """'2023-09-01' → 20230901; missing → 0 (oldest, the reverse-safe face of newer-first)."""
    p = segment.get("pubdate")
    if not p:
        return 0
    try:
        return int(str(p).replace("-", "")[:8])
    except (ValueError, TypeError):
        return 0


def _sort_key(entry: Tuple[Dict, float, Dict]):
    """(−Score_bucket, −pubdate, −view, season_order). Sentinel values for null fields, so sorting never raises."""
    seg, score, _ = entry
    bucket = round(score / EPSILON)
    view = seg.get("view") or 0
    so = seg.get("season_order")
    so = float("inf") if so is None else so
    return (-bucket, -_pubdate_ts(seg), -view, so)


# ══════════════════════════════════════════════════════════════════════
# All-zero fallback (degradation ladder, assertion 11)
# ══════════════════════════════════════════════════════════════════════
def _fallback(scored: List[Tuple[Dict, float, Dict]], track_map: Dict,
              b_n: np.ndarray, warnings: List[str], node: str
              ) -> List[Tuple[Dict, float, Dict]]:
    """A node where every segment scores 0: readmit only tracks whose audience
       score is zero without being gate-suppressed, re-ranked by Match.
       Gate-suppressed tracks stay excluded (no back door for insufficient mastery)."""
    pm_proj = float(np.asarray(b_n, dtype=float)[m.M])
    readmit = []
    for seg, sc, comp in scored:
        if comp["gate_blocked"]:
            continue                          # hard exclusion, no exemption
        # Upgrade: ignore W_track and re-rank by Match; mark crossed so the reason can explain it
        comp2 = dict(comp)
        comp2["crossed"] = True
        comp2["upgraded"] = True
        readmit.append((seg, comp["match"], comp2))
    if readmit:
        warnings.append(f"node {node} all-zero fallback: {len(readmit)} segments upgraded (missing foundational-track coverage)")
    return readmit


# ══════════════════════════════════════════════════════════════════════
# Reason construction (expression layer, template assembly; cross-track reasons must name the track, assertions 8/22)
# ══════════════════════════════════════════════════════════════════════
def _part_title(segment: Dict) -> str:
    """Return part_title normally; on ordinal/duplicate degradation → fall back to video_title (assertion 22)."""
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
# Main entry: recommend (five-stage chain + multi-node merge + rec_served snapshot)
# ══════════════════════════════════════════════════════════════════════
def _optional_business_id(value, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be a non-empty string")
    return normalized


def recommend(grade, learning_purpose, target_nodes: Sequence[str],
              beliefs_proj: Dict[str, np.ndarray], segments_by_node: Dict[str, List[Dict]],
              track_map: Dict, budget: Dict, *,
              seen_segments: Optional[set] = None, prereq_map: Optional[Dict] = None,
              efficacy_table: Optional[Dict] = None,
              rec_id_factory=lambda: uuid.uuid4().hex,
              session_id: Optional[str] = None,
              action_id: Optional[str] = None) -> Dict:
    """Returns {recommendations, rec_served, warnings, status}.
       beliefs_proj: {node: np.array} already projected by the caller via get_belief
       (red line: this module never touches the stored belief directly).
       budget: planner.session_budget(tier). seen_segments: the set of already-pushed
       (bv,p) pairs (dedup, assertion 27)."""
    normalized_session_id = _optional_business_id(session_id, "session_id")
    normalized_action_id = _optional_business_id(action_id, "action_id")
    if (normalized_session_id is None) != (normalized_action_id is None):
        raise ValueError("session_id and action_id must be provided as a pair")
    if normalized_session_id is not None:
        business_payload = json.dumps(
            [normalized_session_id, normalized_action_id],
            ensure_ascii=False, separators=(",", ":"),
        )
        business_event_id = "rec_served:" + hashlib.sha256(
            business_payload.encode("utf-8")
        ).hexdigest()
    else:
        business_event_id = None

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
                raise ValueError(f"duplicate segment_id: {segment_id!r}")
            segment_ids.add(segment_id)

    # 1. Per-node scoring + entropy
    node_pools: Dict[str, List[Tuple[Dict, float, Dict]]] = {}
    node_entropy: Dict[str, float] = {}
    for node in target_nodes:
        try:
            b = np.asarray(beliefs_proj[node], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"node {node!r} has an invalid belief") from exc
        if (b.shape != (4,) or not np.all(np.isfinite(b)) or np.any(b < 0)
                or not np.isclose(b.sum(), 1.0, rtol=0.0, atol=1e-6)):
            raise ValueError(f"node {node!r} belief must be a 4-dimensional probability distribution")
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
        # All-zero fallback
        if scored and all(sc <= 0 for _, sc, _ in scored):
            scored = _fallback(scored, track_map, b, warnings, node)
        scored.sort(key=_sort_key)
        node_pools[node] = scored[:TOPK]

    # 2. Multi-node merge: segment slots are allocated in descending entropy order, ≥1 per node (assertion 13)
    order = sorted(target_nodes, key=lambda n: (-node_entropy[n], n))
    counts = {n: 0 for n in order}
    slots = rx_segments
    for n in order:                            # round one: 1 segment per node in entropy order
        if slots <= 0:
            break
        if node_pools[n]:
            counts[n] = 1
            slots -= 1
    i = 0
    while slots > 0 and any(counts[n] < len(node_pools[n]) for n in order):
        n = order[i % len(order)]              # surplus goes to the highest-entropy nodes (2h/3h+ tiers)
        if counts[n] < len(node_pools[n]):
            counts[n] += 1
            slots -= 1
        i += 1
        if i > 4096:
            break

    # 3. Global sort (descending entropy → descending Score within node) + hard budget constraint
    picks: List[Tuple[str, Dict, float, Dict]] = []
    for n in order:
        for seg, sc, comp in node_pools[n][:counts[n]]:
            picks.append((n, seg, sc, comp))
    picks.sort(key=lambda x: (-node_entropy[x[0]], -x[2]))

    recommendations: List[Dict] = []
    served_snap: List[Dict] = []
    budget_sec = rx_minutes * 60
    try:
        reason_budget_sec = max(
            0, int(float(budget.get("session_remaining_minutes")) * 60)
        )
    except (TypeError, ValueError, OverflowError):
        reason_budget_sec = budget_sec
    used_sec = 0
    served_ids = set()
    for n, seg, sc, comp in picks:
        dur = int(seg.get("duration_sec") or 0)
        segment_key = (seg.get("bv"), seg.get("p"))
        if segment_key in served_ids:
            continue
        if used_sec + dur > budget_sec:
            continue                           # over-budget cutoff (better to omit than to serve junk, assertion 7)
        used_sec += dur
        left_min = max(0, (reason_budget_sec - used_sec) // 60)
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

    # 4. Unserved top-k snapshot (moat material, assertion 25)
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

    if business_event_id is None:
        legacy_basis = served_snap[0]["rec_id"] if served_snap else uuid.uuid4().hex
        event_id = f"rec_served:legacy:{legacy_basis}"
        idempotency_mode = "legacy_rec_id"
    else:
        event_id = business_event_id
        idempotency_mode = "business_key"
    rec_served = {"event_id": event_id,
                  "session_id": normalized_session_id,
                  "action_id": normalized_action_id,
                  "idempotency_mode": idempotency_mode,
                  "mode": mode, "grade_norm": g, "purpose_norm": q,
                  "served": served_snap, "unserved_topk": unserved[:TOPK]}
    return {"recommendations": recommendations, "rec_served": rec_served,
            "warnings": warnings, "status": status,
            "message": "本节点暂无合适视频" if status == "no_segment" else ""}


# ══════════════════════════════════════════════════════════════════════
# rec_served persistence (used by the wiring layer; fail-open + in-memory retry queue, F2/F5/assertion 26)
# ══════════════════════════════════════════════════════════════════════
def append_rec_served(snapshot: Dict, writer, retry_queue: List, *,
                      queue_alert_threshold: int = 10,
                      max_queue_size: int = 100) -> Dict:
    """Called by the wiring layer. writer(record) may raise (disk full / permissions).
       fail-open: a write failure must not block recommendation serving; transient
       failures go into the in-memory retry queue, flushing the backlog before
       writing this record; persistent failures drop data but return a visible
       error (zero silent failures). Returns {ok, flushed, queued, error}."""
    result = {"ok": False, "flushed": 0, "queued": len(retry_queue),
              "dropped": 0, "error": None}
    # Try flushing the backlog first (nothing lost after transient failures recover)
    still_queued = []
    for rec in retry_queue:
        try:
            writer(rec)
            result["flushed"] += 1
        except Exception:                      # noqa: BLE001 — fail-open boundary, visible downstream
            still_queued.append(rec)
    retry_queue[:] = still_queued
    # Write this record
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
        result["error"] = f"{type(e).__name__}: {e}"   # loud error, not silently swallowed
    result["queued"] = len(retry_queue)
    if result["queued"] > queue_alert_threshold:
        result["error"] = (result["error"] or "") + f" | 重试队列积压 {result['queued']} 条 [ERROR]"
    return result
