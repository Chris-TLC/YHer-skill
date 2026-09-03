#!/usr/bin/env python3
"""
DEPRECATED: legacy v3/v4 application retained only for historical tools.

The authoritative student application is ``apps.demo_api`` on port 8700.
All legacy ``/session*`` student routes return HTTP 410.

FastAPI backend (historical master blueprint A: the HTTP↔engine translation layer).

Today Streamlit imports the engine directly; tomorrow the iOS/Android app will hit
these HTTP endpoints and call the same engine. Both paths behave identically →
zero rework at the app stage.

Endpoints:
  POST /session              create a learning cabin
  GET  /session/{sid}        fetch session state
  GET  /session/{sid}/first_question   first diagnostic question
  POST /session/{sid}/diagnose         run one diagnosis turn
  POST /session/{sid}/execute          run one teaching turn
  POST /session/{sid}/report           generate the recap
  POST /upload/homework      L1 image-upload pipeline (homework/papers)
  GET  /health

Run: uvicorn apps.api_server:app --reload --port 8600
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv(SKILL_DIR / ".env")

from adapters.store import LocalJsonStore
from apps.security import RequestGuardMiddleware, UploadSecurityError, save_image_upload
from core.tutor.llm_bridge import make_llm_caller
from core.tutor.session_orchestrator import SessionOrchestrator, TutorSession
from core.tutor.product_loop import build_nodes_contract, build_next_plan, build_video_recommendation
from core.data.knowledge_repository import get_knowledge_repository
from core.data.item_repository import get_item_repository

app = FastAPI(title="YHer 化学私教 API", version="1.0")
app.add_middleware(
    RequestGuardMiddleware,
    max_body_bytes=512 * 1024,
    body_limit_overrides={"/upload/homework": 10 * 1024 * 1024},
    max_requests=120,
    window_seconds=60,
    max_rate_keys=4096,
)

LEGACY_GONE_DETAIL = "legacy student flow retired; use /api/demo/* on port 8700"


@app.middleware("http")
async def reject_legacy_student_flow(request: Request, call_next):
    if request.url.path == "/session" or request.url.path.startswith("/session/"):
        return JSONResponse(status_code=410, content={"detail": LEGACY_GONE_DETAIL})
    return await call_next(request)

# WS4 (2026-07-04): v4-rendered RIR routes, coexisting with v3 with zero intrusion; switching the service source is an explicit WS6 action.
from apps.api_v4_render import router as _v4_render_router  # noqa: E402
app.include_router(_v4_render_router)

STORE = LocalJsonStore()
UPLOAD_DIR = SKILL_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_MAX_BYTES = 8 * 1024 * 1024


# ── Request models ────────────────────────────────────────
class CreateSessionReq(BaseModel):
    user_id: str = "demo"
    goal: str = "补化学薄弱点"
    node_id: str = ""
    grade: str = "高二"
    grade_detail: str = ""
    learning_purpose: str = ""
    region: str = "全国卷"
    time_budget_min: int = 180
    provider: str = "deepseek"
    api_key: Optional[str] = None


class DiagnoseReq(BaseModel):
    question: Dict[str, Any]
    answer: str
    provider: str = "deepseek"
    api_key: Optional[str] = None


class ExecuteReq(BaseModel):
    message: str
    time_context: Optional[Dict[str, Any]] = None
    provider: str = "deepseek"
    api_key: Optional[str] = None


class ReportReq(BaseModel):
    provider: str = "deepseek"
    api_key: Optional[str] = None


class VerificationReq(BaseModel):
    answers: Dict[str, str] = {}


# ── Utilities ─────────────────────────────────────────────
def _orchestrator(provider: str, api_key: Optional[str]) -> SessionOrchestrator:
    key = (api_key or "").strip() or os.getenv(f"{provider.upper()}_API_KEY", "")
    caller = None
    if key:
        try:
            caller = make_llm_caller(provider, api_key=key)
        except Exception:
            caller = None
    return SessionOrchestrator(llm_caller=caller)


def _load_session(sid: str) -> TutorSession:
    data = STORE.load_session(sid)
    if not data:
        raise HTTPException(404, f"session {sid} not found")
    return TutorSession(**data)


def _save_session(s: TutorSession):
    from dataclasses import asdict
    STORE.save_session(s.session_id, asdict(s))


# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health():
    from core.data.item_repository import get_item_repository
    from core.data.knowledge_repository import get_knowledge_repository
    return {
        "status": "ok",
        "kg_nodes": len(get_knowledge_repository().all_nodes()),
        "item_bank": get_item_repository().count(),
    }


# ── New-flow endpoints (for the native HTML frontend) ─────
@app.get("/nodes")
def list_nodes(category: Optional[str] = None, q: Optional[str] = None):
    """List the selectable knowledge points (for the home-page picker). Filter by category / search by text."""
    if not category and not q:
        return build_nodes_contract()
    kr = get_knowledge_repository()
    out = []
    for n in kr.all_nodes():
        if category and n.category != category:
            continue
        if q and q not in n.node_id:
            continue
        out.append({
            "node_id": n.node_id, "category": n.category,
            "difficulty": getattr(n, "difficulty", "T2"),
            "exam_weight": getattr(n, "exam_weight", ""),
            "has_video": bool(getattr(n, "videos", [])),
        })
    # Group by category for easy frontend display
    cats = {}
    for x in out:
        cats.setdefault(x["category"], []).append(x)
    return {"total": len(out), "by_category": cats, "nodes": out}


@app.get("/node/{node_id}/videos")
def node_videos(node_id: str):
    """Fetch the yihuier videos recommended for a node (core feature). Falls back to the parent node when a child node has no videos.

    Note: when a node name contains '/', putting it in the URL path makes the
    server treat it as a path separator → 404 (e.g. 醛/酮, 糖类/油脂, 8 nodes in
    total). The frontend should use the query-parameter variant
    /node_videos?node_id= for such nodes.
    """
    return _node_videos_impl(node_id)


@app.get("/node_videos")
def node_videos_query(node_id: str):
    """Query-parameter variant of the video endpoint. Safe for node names containing '/' (unaffected by URL path segmentation)."""
    return _node_videos_impl(node_id)


def _node_videos_impl(node_id: str):
    return build_video_recommendation(node_id)


@app.get("/session/{sid}/profile")
def session_profile(sid: str):
    """Fetch the current ability profile (radar chart / mastery data).

    Prefer the already-persisted subject_ability.kg_mastery; otherwise aggregate
    the real mastery live from this session's diag_history (split by diagnostic
    levels L1-L4), so students see their actual answering performance rather
    than hardcoded demo data.
    """
    s = _load_session(sid)
    kg_mastery = {}
    prof = getattr(s, "subject_ability", None) or {}
    if isinstance(prof, dict):
        kg_mastery = (prof.get("kg_mastery") or {})

    # Stored profile empty → aggregate live from the diagnosis history
    if not kg_mastery:
        kg_mastery = _aggregate_mastery_from_diag(s)

    return {
        "session_id": sid,
        "kg_mastery": kg_mastery,
        "node_id": s.node_id,
    }


# Diagnostic level → ability-axis dimension (L0 self-assessment doesn't count toward mastery)
_LEVEL_DIM = {
    "L1": "基础概念", "L2": "应用迁移", "L3": "综合推理", "L4": "拔高难题",
}
# Fixed ability axes for the radar chart (guarantees it renders; the order is the radar axis order)
_RADAR_DIMS = ["基础概念", "应用迁移", "综合推理", "审题入口", "整体掌握"]

def _aggregate_mastery_from_diag(s):
    """Aggregate the real mastery from diag_history into stable multi-dimensional radar data.

    Principles:
      - ability axes the student actually answered → use the real mastery mean of that axis (real data);
      - axes not covered by the diagnosis → use the overall mastery as a conservative
        baseline (slightly lowered, indicating insufficient checking), so the radar
        takes shape without fabricating high scores;
      - L0 self-assessment doesn't count toward mastery.
    Dimensions: 基础概念 / 应用迁移 / 综合推理 / 审题入口 / 整体掌握.
    """
    diag = getattr(s, "diag_history", None) or []
    by_dim = {}            # ability axis → [mastery, ...]
    all_scores = []        # mastery of all answers (excluding L0)
    for h in diag:
        q = h.get("question", {}) or {}
        level = (q.get("level") or "")
        if level.startswith("L0"):   # self-assessment doesn't count toward mastery
            continue
        m = h.get("mastery")
        if not isinstance(m, (int, float)):
            continue
        m = float(m)
        all_scores.append(m)
        # Map by level onto the corresponding ability axis
        code = level.split()[0] if level else ""
        dim = _LEVEL_DIM.get(code)
        if dim:
            by_dim.setdefault(dim, []).append(m)
        # Also feed the "审题入口" axis by axis tag (entry-judgment / reading / classification items)
        axis = (q.get("axis") or "").lower()
        if any(k in axis for k in ("入口", "判断", "审题", "classify", "entry")):
            by_dim.setdefault("审题入口", []).append(m)

    if not all_scores:
        return {}

    overall = round(sum(all_scores) / len(all_scores), 3)
    # Conservative baseline for uncovered dimensions: overall mastery lowered one notch (never below 0)
    baseline = round(max(0.0, overall - 0.08), 3)

    out = {}
    for dim in _RADAR_DIMS:
        if dim == "整体掌握":
            out[dim] = overall
        elif dim in by_dim:
            out[dim] = round(sum(by_dim[dim]) / len(by_dim[dim]), 3)
        else:
            out[dim] = baseline
    return out


@app.post("/session")
def create_session(req: CreateSessionReq):
    orch = _orchestrator(req.provider, req.api_key)
    s = orch.create_session(
        req.user_id, req.goal, node_id=req.node_id, grade=req.grade,
        region=req.region, time_budget_min=req.time_budget_min,
        grade_detail=req.grade_detail, learning_purpose=req.learning_purpose,
    )
    _save_session(s)
    return {"session_id": s.session_id, "node_id": s.node_id,
            "tasks": s.tasks, "goal": s.goal,
            "grade_detail": s.grade_detail,
            "learning_purpose": s.learning_purpose}


@app.get("/session/{sid}")
def get_session(sid: str):
    return _load_session(sid).__dict__


@app.get("/session/{sid}/first_question")
def first_question(sid: str):
    s = _load_session(sid)
    return SessionOrchestrator(llm_caller=None).first_question(s)


@app.get("/session/{sid}/diagnosis_prep")
def diagnosis_prep(sid: str):
    s = _load_session(sid)
    return SessionOrchestrator(llm_caller=None).prepare_diagnosis(s)


@app.post("/session/{sid}/diagnose")
def diagnose(sid: str, req: DiagnoseReq):
    s = _load_session(sid)
    orch = _orchestrator(req.provider, req.api_key)
    result = orch.run_diagnosis_turn(s, req.question, req.answer)
    _save_session(s)
    result["cost_yuan"] = s.cost_yuan
    return result


@app.post("/session/{sid}/execute")
def execute(sid: str, req: ExecuteReq):
    s = _load_session(sid)
    orch = _orchestrator(req.provider, req.api_key)
    result = orch.run_execution_turn(s, req.message, time_ctx=req.time_context)
    _save_session(s)
    result["cost_yuan"] = s.cost_yuan
    return result


@app.post("/session/{sid}/report")
def report(sid: str, req: ReportReq):
    s = _load_session(sid)
    orch = _orchestrator(req.provider, req.api_key)
    result = orch.run_report(s)
    _save_session(s)
    result["cost_yuan"] = s.cost_yuan
    return result


@app.get("/session/{sid}/verification")
def verification(sid: str):
    s = _load_session(sid)
    return SessionOrchestrator(llm_caller=None).build_post_video_verification(s)


@app.post("/session/{sid}/verification")
def submit_verification(sid: str, req: VerificationReq):
    s = _load_session(sid)
    result = SessionOrchestrator(llm_caller=None).submit_post_video_verification(s, req.answers)
    _save_session(s)
    return result


@app.get("/session/{sid}/next_plan")
def next_plan(sid: str, verification_passed: bool = False):
    s = _load_session(sid)
    return build_next_plan(s.node_id, verification_passed=verification_passed)


@app.post("/upload/homework")
async def upload_homework(user_id: str = Form("demo"), file: UploadFile = File(...)):
    """
    L1 image-upload pipeline (master blueprint chapter 4).

    Current L1: receive the image, store it, return a structured placeholder.
    L2 printed-text OCR / L3 answer recognition / L4 handwritten work are
    enhanced layer by layer once the vision model is wired in; the pipeline
    needs no rework.
    """
    try:
        saved = await save_image_upload(
            file,
            user_id=user_id,
            upload_dir=UPLOAD_DIR,
            max_bytes=UPLOAD_MAX_BYTES,
        )
    except UploadSecurityError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    return {
        **saved, "stored": True,
        "vision_status": "pending",
        "note": "L1 pipeline has received and stored the image. Printed-text OCR / answer recognition / handwritten-work recognition will be wired into the vision model layer by layer.",
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8600)


# WS4 (2026-07-04) static-page mount: must come after all route definitions
# (mount("/") is a catch-all; mounting it early would shadow the v3 API routes).
# Serves apps/web/ (v4_preview.html / rir_renderer.js).
from fastapi.staticfiles import StaticFiles  # noqa: E402
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "web"), html=True), name="web")
