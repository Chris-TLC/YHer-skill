#!/usr/bin/env python3
"""WS4 v4 渲染 API(与现有 v3 路由并存,零侵入):
  GET /api/v4/items/{item_id}/rir      → 渲染中间表示(RIR)
  GET /api/v4/assets/{asset_hash}.png  → normalized/修复件图源
  GET /api/v4/raw_assets               → 原始 png/jpeg 直显(答案区兜底)
  POST /api/render_report              → 前端 KaTeX/渲染错误上报到 /tmp
只读服务池(item_bank_v4 R1-R4),不暴露 excluded/legacy 池。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from core.data.item_bank_v4 import load_service_pool
from core.render.block_renderer import item_to_rir, _Ctx

router = APIRouter(tags=["v4-render"])

REPORT_DIR = Path("/tmp/yher_batch9_ws4")
REPORT_PATH = REPORT_DIR / "render_reports.jsonl"

_pool_cache = None


def _pool():
    global _pool_cache
    if _pool_cache is None:
        # 预览/审计面(127.0.0.1):必须能渲染 fixable/blocked 池的题供 QA 逐题核查,
        # 故显式关 R5(2026-07-06 R5 apply)。学生端不走本模块,R5 门在 item_bank_v4 默认生效。
        _pool_cache = load_service_pool(apply_r5=False)
    return _pool_cache


def _scrub_report_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            lk = str(k).lower()
            if any(token in lk for token in ("key", "token", "secret", "password")):
                out[str(k)] = "[redacted]"
            else:
                out[str(k)] = _scrub_report_value(v)
        return out
    if isinstance(value, list):
        return [_scrub_report_value(v) for v in value[:50]]
    if isinstance(value, str):
        return value[:1000]
    return value


def append_render_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "reviewer": "",
        "payload": _scrub_report_value(payload),
    }
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"ok": True, "path": str(REPORT_PATH)}


@router.get("/api/v4/items/{item_id}/rir")
def get_item_rir(item_id: str, zones: str = Query("stem", description="逗号分隔: stem,answer,analysis")):
    item = _pool().get(item_id)
    if item is None:
        raise HTTPException(404, "item not in service pool")
    zs = tuple(z.strip() for z in zones.split(",") if z.strip() in ("stem", "answer", "analysis"))
    return item_to_rir(item, zones=zs or ("stem",))


@router.get("/api/v4/assets/{asset_hash}.png")
def get_asset(asset_hash: str):
    if not asset_hash.isalnum() or len(asset_hash) != 64:
        raise HTTPException(400, "bad asset hash")
    from core.data.ws2_transcripts import resolve_image_path
    p = resolve_image_path(asset_hash)
    if p is None:
        raise HTTPException(404, "asset not found")
    return FileResponse(p, media_type="image/png")


@router.get("/api/v4/raw_assets")
def get_raw_asset(group_key: str, media: str):
    p = _Ctx.get().raw_asset_path(group_key, media)
    if p is None or p.suffix.lower() not in (".png", ".jpeg", ".jpg"):
        raise HTTPException(404, "raw asset not found or not directly displayable")
    mt = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(p, media_type=mt)


@router.post("/api/render_report")
def render_report(payload: Dict[str, Any]):
    return append_render_report(payload)


# ============================================================================
# 练习舱 v1(2026-07-06,D2-A Track A③):学生口径 study API。
# 与上面的审计面(_pool,apply_r5=False)严格分离:study 池走 R5 默认口径,
# 学生只可能抽到白名单题。埋点/报坏题落 data/study_logs/(非 official item_bank)。
# ============================================================================

import random as _random

try:  # study_schema 与本模块同在 apps/；兼容包内 import 与直跑两种路径
    from apps import study_schema as sch
except ImportError:  # pragma: no cover
    import study_schema as sch

STUDY_LOG_DIR = Path(__file__).parent.parent / "data" / "study_logs"

_study_pool_cache = None


def _study_pool():
    global _study_pool_cache
    if _study_pool_cache is None:
        _study_pool_cache = load_service_pool()  # apply_r5=True 默认:学生端唯一合法口径
    return _study_pool_cache


def _study_append(fname: str, row: Dict[str, Any]) -> str:
    STUDY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = STUDY_LOG_DIR / fname
    row = dict(row)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    sch.stamp(row)  # 增量清单第 7 项：所有落盘行注入 subject + schema_version（现在锁死）
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_scrub_report_value(row), ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


@router.get("/api/v4/study/nodes")
def study_nodes():
    """白名单池的节点清单(名称+可用题数),供前端选节点。"""
    counts: Dict[str, int] = {}
    for it in _study_pool().values():
        for n in (it.get("kg_nodes") or ["其他"]):
            counts[n] = counts.get(n, 0) + 1
    return {"total_items": len(_study_pool()),
            "nodes": [{"node": k, "count": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]}


@router.post("/api/v4/study/next")
def study_next(payload: Dict[str, Any]):
    """抽下一题:R5 白名单 ∩ 节点过滤 − 本会话已做。payload={node?, seen_ids:[...]}"""
    node = payload.get("node") or ""
    seen = set(payload.get("seen_ids") or [])
    pool = _study_pool()
    cands = [iid for iid, it in pool.items()
             if iid not in seen and (not node or node in (it.get("kg_nodes") or ["其他"]))]
    if not cands:
        return {"done": True, "remaining": 0}
    iid = _random.choice(cands)
    it = pool[iid]
    # 增量清单第 2 项 propensity 快照（最高优先，事后补不回）：v1 随机抽题也照记
    # {候选集, 各分, 展示位次, randomized}，抽题逻辑本身不变；日后 recommender
    # 实装时 scores 换逐候选分数列表。失败不影响 serve（try/except 静默降级）。
    try:
        _study_append(
            f"propensity_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl",
            sch.propensity_snapshot(
                cands, {"type": "uniform", "value": round(1.0 / len(cands), 8)}, iid,
                position=1, randomized=True,
                node=node, session_id=str(payload.get("session_id") or "")))
    except Exception:
        pass
    return {"done": False, "item_id": iid, "remaining": len(cands) - 1,
            "kg_nodes": it.get("kg_nodes") or [], "group_key": it.get("group_key", "")}


@router.post("/api/v4/study/event")
def study_event(payload: Dict[str, Any]):
    """埋点:session_start/item_shown/answer_revealed/self_mark/next/session_end。"""
    if not payload.get("event"):
        raise HTTPException(400, "event required")
    # 增量清单第 1 项 session.time_budget：session_start 带档位/分钟数时规范化补
    # time_budget_minutes 字段（原值照录，不改事件语义）。
    if payload.get("event") == "session_start" and "time_budget" in payload:
        payload = dict(payload)
        payload["time_budget_minutes"] = sch.normalize_time_budget(payload.get("time_budget"))
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = _study_append(f"events_{day}.jsonl", payload)
    return {"ok": True, "path": path}


@router.post("/api/v4/study/report_bad")
def study_report_bad(payload: Dict[str, Any]):
    """报坏题:最高优先审计信号,进队列等 Claude/用户签(治理v2外部收敛器)。"""
    if not payload.get("item_id"):
        raise HTTPException(400, "item_id required")
    row = dict(payload)
    row["status"] = "pending"
    row["reviewer"] = ""
    path = _study_append("bad_reports.jsonl", row)
    return {"ok": True, "path": path}


# ── 诊断引擎数据地基增量埋点（2026-07-08，schema 见 apps/study_schema.py）──
# 第 3/8 项在此开前端端点；第 4(掌握度 S+last_review_at)/5(疗效 α,β,对照臂)/
# 6(L2 事件+L3 档案) 项为引擎侧产物：字段定义已在 study_schema.py 钉死，
# engine/（mastery/recommender/memory）实装时直接 import 构造行并 append
# 到 data/study_logs/，不走学生 HTTP 面。


@router.post("/api/v4/study/watch_proxy")
def study_watch_proxy(payload: Dict[str, Any]):
    """增量清单第 3 项 watch_proxy（#6 服务端时间戳口径 + #11 两级样本）。
    前端返回本站时 POST {leave_ts, return_ts, seg_seconds, self_report?,
    retest_delay?, segment_id?, bvid?, session_id?}；缺 return_ts 时以本次
    请求的服务端时间戳为返回时刻（"下一次任意 API 请求"口径的直接体现）。"""
    for k in ("leave_ts", "seg_seconds"):
        if k not in payload:
            raise HTTPException(400, f"{k} required")
    return_ts = payload.get("return_ts")
    if return_ts is None:
        return_ts = datetime.now(timezone.utc).timestamp()
    try:
        row = sch.watch_proxy_record(
            float(payload["leave_ts"]), float(return_ts), float(payload["seg_seconds"]),
            self_report=payload.get("self_report"),
            retest_delay=payload.get("retest_delay"),
            session_id=str(payload.get("session_id") or ""),
            segment_id=str(payload.get("segment_id") or ""),
            bvid=str(payload.get("bvid") or ""))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"bad watch_proxy payload: {e}")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = _study_append(f"watch_proxy_{day}.jsonl", row)
    return {"ok": True, "path": path, "sample_level": row["sample_level"]}


@router.post("/api/v4/study/followup_triage")
def study_followup_triage(payload: Dict[str, Any]):
    """增量清单第 8 项：追问判定器三元组 {错因码, 置信度, 似然向量} 原样入日志
    （存三元组而非结论，换 LLM 可重算历史）。似然向量的净化/钳位在引擎消费侧
    （engine.mastery.sanitize_llm_likelihood），日志层只标 wellformed 不改写。"""
    if not payload.get("item_id"):
        raise HTTPException(400, "item_id required")
    row = sch.followup_triage_row(
        str(payload["item_id"]), payload.get("utterance", ""),
        payload.get("error_code", ""), payload.get("confidence"),
        payload.get("likelihood_vector"),
        session_id=str(payload.get("session_id") or ""),
        node=str(payload.get("node") or ""))
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = _study_append(f"followup_triage_{day}.jsonl", row)
    return {"ok": True, "path": path, "wellformed": row["wellformed"]}
