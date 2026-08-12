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

# WS4(2026-07-04):v4 渲染 RIR 路由,与 v3 并存零侵入;切换服务源是 WS6 的显式动作。
from apps.api_v4_render import router as _v4_render_router  # noqa: E402
app.include_router(_v4_render_router)

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


# ── 工具 ──────────────────────────────────────────────────
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


# ── 新流程端点(原生HTML前端用)──────────────────────────────
@app.get("/nodes")
def list_nodes(category: Optional[str] = None, q: Optional[str] = None):
    """列出可选知识点(首页选择用)。按类别筛/按文字搜。"""
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
    # 按类别分组返回,方便前端展示
    cats = {}
    for x in out:
        cats.setdefault(x["category"], []).append(x)
    return {"total": len(out), "by_category": cats, "nodes": out}


@app.get("/node/{node_id}/videos")
def node_videos(node_id: str):
    """取某节点推荐的一化儿视频(核心功能)。子节点无视频则回退到父节点。

    注意:节点名含 '/'(如 醛/酮、糖类/油脂 共8个)时,放进 URL 路径会被
    服务器当作路径分隔符 → 404。这类节点前端应改用 /node_videos?node_id= 查询参数版。
    """
    return _node_videos_impl(node_id)


@app.get("/node_videos")
def node_videos_query(node_id: str):
    """查询参数版视频端点。对含 '/' 的节点名安全(不受 URL 路径分段影响)。"""
    return _node_videos_impl(node_id)


def _node_videos_impl(node_id: str):
    return build_video_recommendation(node_id)


@app.get("/session/{sid}/profile")
def session_profile(sid: str):
    """取当前能力画像(雷达图/掌握度数据)。

    优先用已落库的 subject_ability.kg_mastery;
    否则从本次诊断的 diag_history 实时聚合出真实掌握度(按诊断层级 L1-L4 分维),
    保证学生看到的是自己真实答题表现,而非写死的演示数据。
    """
    s = _load_session(sid)
    kg_mastery = {}
    prof = getattr(s, "subject_ability", None) or {}
    if isinstance(prof, dict):
        kg_mastery = (prof.get("kg_mastery") or {})

    # 落库画像为空 → 从诊断历史实时聚合
    if not kg_mastery:
        kg_mastery = _aggregate_mastery_from_diag(s)

    return {
        "session_id": sid,
        "kg_mastery": kg_mastery,
        "node_id": s.node_id,
    }


# 诊断层级 → 能力面维度(L0自评不计入掌握度)
_LEVEL_DIM = {
    "L1": "基础概念", "L2": "应用迁移", "L3": "综合推理", "L4": "拔高难题",
}
# 雷达图固定能力面(保证成形;顺序即雷达轴顺序)
_RADAR_DIMS = ["基础概念", "应用迁移", "综合推理", "审题入口", "整体掌握"]

def _aggregate_mastery_from_diag(s):
    """从 diag_history 聚合真实掌握度,产出稳定的多维雷达数据。

    原则:
      - 真正做过题的能力面 → 用该面真实 mastery 均值(真数据);
      - 没被诊断覆盖的面 → 用整体掌握度作保守基线(略下调,表示未充分检验),
        既让雷达成形,又不臆造高分;
      - L0 自评不计入掌握度。
    维度: 基础概念 / 应用迁移 / 综合推理 / 审题入口 / 整体掌握。
    """
    diag = getattr(s, "diag_history", None) or []
    by_dim = {}            # 能力面 → [mastery,...]
    all_scores = []        # 全部答题(L0除外)的 mastery
    for h in diag:
        q = h.get("question", {}) or {}
        level = (q.get("level") or "")
        if level.startswith("L0"):   # 自评不计入掌握度
            continue
        m = h.get("mastery")
        if not isinstance(m, (int, float)):
            continue
        m = float(m)
        all_scores.append(m)
        # 按 level 落到对应能力面
        code = level.split()[0] if level else ""
        dim = _LEVEL_DIM.get(code)
        if dim:
            by_dim.setdefault(dim, []).append(m)
        # 按 axis 补充"审题入口"面(入口判断/审题/分类类考点)
        axis = (q.get("axis") or "").lower()
        if any(k in axis for k in ("入口", "判断", "审题", "classify", "entry")):
            by_dim.setdefault("审题入口", []).append(m)

    if not all_scores:
        return {}

    overall = round(sum(all_scores) / len(all_scores), 3)
    # 未覆盖维度的保守基线:整体掌握度下调一档(不低于0)
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


# WS4(2026-07-04)静态页挂载:必须在全部路由定义之后(mount("/") 是 catch-all,
# 提前挂会遮蔽 v3 API 路由)。serve apps/web/(v4_preview.html / rir_renderer.js)。
from fastapi.staticfiles import StaticFiles  # noqa: E402
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "web"), html=True), name="web")
