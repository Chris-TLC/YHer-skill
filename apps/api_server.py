#!/usr/bin/env python3
"""
DEPRECATED: legacy v3/v4 application retained only for historical tools.

The authoritative student application is ``apps.demo_api`` on port 8700.
All legacy ``/session*`` student routes return HTTP 410.

FastAPI 后端（历史总蓝图 A：HTTP↔引擎翻译层）。

今天 Streamlit 直接 import 引擎；明天 iOS/安卓 App 走这些 HTTP 端点，调同一个引擎。
两条路行为一致 → App 阶段零返工。

端点：
  POST /session              创建学习舱
  GET  /session/{sid}        取会话状态
  GET  /session/{sid}/first_question   首个诊断题
  POST /session/{sid}/diagnose         诊断一轮
  POST /session/{sid}/execute          执行教学一轮
  POST /session/{sid}/report           生成复盘
  POST /upload/homework      L1 图片上传管线（作业/卷子）
  GET  /health

运行：uvicorn apps.api_server:app --reload --port 8600
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


STORE = LocalJsonStore()
UPLOAD_DIR = SKILL_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_MAX_BYTES = 8 * 1024 * 1024


# ── 请求模型 ──────────────────────────────────────────────
class CreateSessionReq(BaseModel):
    user_id: str = "demo"
    goal: str = "补化学薄弱点"
    node_id: str = ""
    grade: str = "高二"
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


# ── 工具 ──────────────────────────────────────────────────
def _orchestrator(provider: str, api_key: Optional[str]) -> SessionOrchestrator:
    key = (api_key or "").strip() or os.getenv(f"{provider.upper()}_API_KEY", "")
    caller = make_llm_caller(provider, api_key=key) if key else None
    return SessionOrchestrator(llm_caller=caller)


def _load_session(sid: str) -> TutorSession:
    data = STORE.load_session(sid)
    if not data:
        raise HTTPException(404, f"会话 {sid} 不存在")
    return TutorSession(**data)


def _save_session(s: TutorSession):
    from dataclasses import asdict
    STORE.save_session(s.session_id, asdict(s))


# ── 端点 ──────────────────────────────────────────────────
@app.get("/health")
def health():
    from core.data.item_repository import get_item_repository
    from core.data.knowledge_repository import get_knowledge_repository
    return {
        "status": "ok",
        "kg_nodes": len(get_knowledge_repository().all_nodes()),
        "item_bank": get_item_repository().count(),
    }


@app.post("/session")
def create_session(req: CreateSessionReq):
    orch = _orchestrator(req.provider, req.api_key)
    s = orch.create_session(
        req.user_id, req.goal, node_id=req.node_id, grade=req.grade,
        region=req.region, time_budget_min=req.time_budget_min,
    )
    _save_session(s)
    return {"session_id": s.session_id, "node_id": s.node_id,
            "tasks": s.tasks, "goal": s.goal}


@app.get("/session/{sid}")
def get_session(sid: str):
    return _load_session(sid).__dict__


@app.get("/session/{sid}/first_question")
def first_question(sid: str):
    s = _load_session(sid)
    return SessionOrchestrator(llm_caller=None).first_question(s)


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


@app.post("/upload/homework")
async def upload_homework(user_id: str = Form("demo"), file: UploadFile = File(...)):
    """
    L1 图片上传管线（总蓝图第4章）。

    当前 L1：收图、存储、返回结构化占位。
    L2 印刷体OCR / L3 答案识别 / L4 手写过程，接入视觉模型后逐层增强，管线不返工。
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
        "note": "L1 管线已收图存储。印刷体OCR/答案识别/手写过程识别将逐层接入视觉模型。",
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8600)
