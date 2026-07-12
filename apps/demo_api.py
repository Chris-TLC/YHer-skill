"""One same-origin FastAPI application for the authoritative learning loop."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from adapters.store import LocalJsonStore
from core.learning.item_catalog import ItemCatalog
from core.learning.session_service import SessionError, SessionService

from .demo_schemas import StartSessionRequest, SubmitAnswerRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "web"
ENGINE_VERSION = "five-engine-contract-20260713"


def create_app(
    *,
    catalog: ItemCatalog | None = None,
    store=None,
    clock: Callable[[], float] = time.time,
    id_factory: Callable[[], str] | None = None,
    llm_grader=None,
    static_dir: Path | None = DEFAULT_STATIC_DIR,
) -> FastAPI:
    catalog = catalog or ItemCatalog.from_default_data()
    store = store or LocalJsonStore()
    service = SessionService(
        catalog,
        store,
        clock=clock,
        id_factory=id_factory,
        llm_grader=llm_grader,
    )
    started_at = datetime.fromtimestamp(float(clock()), timezone.utc).isoformat()
    git_identity = _git_identity()
    app = FastAPI(title="YHer Chemistry Demo", version="2.0.0")
    app.state.catalog = catalog
    app.state.store = store
    app.state.session_service = service

    @app.get("/health")
    def health():
        stats = catalog.stats()
        return {
            "status": "ok",
            "git": dict(git_identity),
            "engine_version": ENGINE_VERSION,
            "started_at": started_at,
            "data": catalog.data_metadata(),
            "counts": {
                "r5": stats.r5_rows,
                "trusted": stats.trusted_items,
                "rejected": stats.rejected_items,
                "families": stats.families,
                "open_nodes": len(catalog.open_nodes()),
            },
        }

    @app.get("/api/demo/nodes")
    def nodes():
        return {
            "nodes": [
                {"node": node, "deterministic_families": count}
                for node, count in catalog.open_nodes().items()
            ],
            "default_node": "氧化还原反应",
            "synthetic": False,
        }

    @app.post("/api/demo/sessions", status_code=201)
    def start_session(request: StartSessionRequest):
        try:
            session = service.start_session(
                request.user_id,
                request.node,
                request.budget_tier,
                grade=request.grade,
                learning_purpose=request.learning_purpose,
            )
            assignment = service.next_assignment(session["session_id"])
        except SessionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return {**session, "assignment": assignment}

    @app.get("/api/demo/sessions/{session_id}/next")
    def next_assignment(session_id: str):
        try:
            return service.next_assignment(session_id)
        except SessionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.get("/api/demo/sessions/{session_id}")
    def session_view(session_id: str):
        try:
            return service.session_view(session_id)
        except SessionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.get("/api/demo/sessions/{session_id}/report")
    def report(session_id: str):
        try:
            return service.report(session_id)
        except SessionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/api/demo/sessions/{session_id}/submit")
    def submit(session_id: str, request: SubmitAnswerRequest):
        try:
            return service.submit(
                session_id,
                request.assignment_id,
                request.submission_id,
                request.answer,
            )
        except SessionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/api/demo/sessions/{session_id}/pause")
    def pause(session_id: str):
        try:
            return service.pause_session(session_id)
        except SessionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.post("/api/demo/sessions/{session_id}/resume")
    def resume(session_id: str):
        try:
            view = service.resume_session(session_id)
            return {**view, "assignment": service.next_assignment(session_id)}
        except SessionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # Register the catch-all mount last so API and health routes always win.
    if static_dir is not None:
        path = Path(static_dir)
        if path.is_dir():
            app.mount("/", StaticFiles(directory=str(path), html=True), name="student-web")
    return app


def _git_identity() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
    }


app = create_app()
