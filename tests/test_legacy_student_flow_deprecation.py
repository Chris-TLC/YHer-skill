"""Safety contracts for student flows superseded by ``apps.demo_api``."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_post_video_verification_is_hard_deprecated() -> None:
    from core.tutor.session_orchestrator import (
        LegacyStudentFlowDeprecated,
        SessionOrchestrator,
    )

    session = SessionOrchestrator(llm_caller=None).create_session(
        "legacy-test-user",
        "旧复测入口",
        node_id="氧化还原反应",
    )

    with pytest.raises(LegacyStudentFlowDeprecated):
        SessionOrchestrator(llm_caller=None).build_post_video_verification(session)
    with pytest.raises(LegacyStudentFlowDeprecated):
        SessionOrchestrator(llm_caller=None).submit_post_video_verification(session, {})


def test_legacy_runtime_sources_contain_no_demo_mastery_constants() -> None:
    prohibited = (
        "0." + "78",
        "0." + "42",
        "+" + "0.16",
    )
    for relative in (
        "core/tutor/session_orchestrator.py",
        "core/private_tutor.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        for marker in prohibited:
            assert marker not in source, f"{relative} still contains {marker}"


@pytest.mark.parametrize(
    ("method", "path", "body"),
    (
        ("post", "/session", {"user_id": "legacy", "node_id": "氧化还原反应"}),
        ("get", "/session/missing", None),
        ("get", "/session/missing/first_question", None),
        ("get", "/session/missing/diagnosis_prep", None),
        ("post", "/session/missing/diagnose", {"question": {}, "answer": ""}),
        ("post", "/session/missing/execute", {"message": ""}),
        ("post", "/session/missing/report", {}),
        ("get", "/session/missing/verification", None),
        ("post", "/session/missing/verification", {"answers": {}}),
        ("get", "/session/missing/next_plan", None),
    ),
)
def test_legacy_http_student_routes_are_gone(method: str, path: str, body: dict | None) -> None:
    from apps.api_server import app

    response = TestClient(app).request(
        method.upper(),
        path,
        **({"json": body} if body is not None else {}),
    )

    assert response.status_code == 410
    assert response.json()["detail"] == "legacy student flow retired; use /api/demo/* on port 8700"
