"""Static contracts for the canonical same-origin Demo student UI."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "apps" / "web" / "index.html"
CSS = ROOT / "apps" / "web" / "app.css"
JS = ROOT / "apps" / "web" / "app.js"
RIR = ROOT / "apps" / "web" / "rir_renderer.js"
KATEX = ROOT / "apps" / "web" / "vendor" / "katex"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_canonical_page_uses_external_assets_and_semantic_workflow_regions():
    html = _read(INDEX)

    assert '<link rel="stylesheet" href="app.css">' in html
    assert '<script defer src="app.js"></script>' in html
    assert 'href="vendor/katex/katex.min.css"' in html
    assert 'src="vendor/katex/katex.min.js"' in html
    assert 'src="vendor/katex/contrib/mhchem.min.js"' in html
    assert '<link rel="icon" href="data:,">' in html
    assert 'viewport-fit=cover' in html
    for marker in (
        'id="setup-view"',
        'id="session-view"',
        'id="report-view"',
        'id="status-region"',
        'aria-live="polite"',
        '<form id="setup-form"',
    ):
        assert marker in html
    assert "<style" not in html
    assert "<script>" not in html
    assert "https://" not in html
    assert "http://" not in html


def test_frontend_calls_only_the_canonical_same_origin_demo_namespace():
    js = _read(JS)
    paths = set(re.findall(r"[\"'`](/api/[A-Za-z0-9_{}$./-]+)", js))

    assert paths
    assert all(path.startswith("/api/demo/") for path in paths)
    assert 'request("/api/demo/nodes"' in js
    assert 'request("/api/demo/sessions"' in js
    assert "/api/demo/sessions/${state.sessionId}/submit" in js
    for stale in ("8600", "8504", "/api/v4", "node_videos", "diagnosis_prep"):
        assert stale not in js


def test_frontend_never_scores_or_persists_private_question_state():
    js = _read(JS)

    for forbidden in (
        "standard_answer",
        "final_answers",
        "0.78",
        "0.42",
        "+0.16",
        "item_id",
        "family_id",
        "normalizeAnswer",
        "isCorrect",
    ):
        assert forbidden not in js
    assert "submission_id: submissionId" in js
    assert "assignment_id: state.assignment.assignment_id" in js
    assert "answer: state.draftAnswer" in js
    assert "JSON.stringify(state" not in js
    assert "yher_demo_session_id" in js
    assert "yher_demo_user_id" in js
    assert "localStorage.setItem(STORAGE.session" in js
    assert "localStorage.setItem(STORAGE.user" in js


def test_phase_gates_delay_diagnostic_and_held_out_correctness():
    js = _read(JS)

    assert 'practice: "learning"' in js
    assert 'return phase === "learning";' in js
    assert 'phase === "diagnostic" || phase === "held_out"' in js
    assert "result.is_correct" in js
    assert "renderNeutralSubmission" in js
    assert "renderPracticeFeedback" in js
    assert "fetchReport" in js
    assert "phaseFromServer" in js


def test_frontend_reads_the_canonical_server_progress_and_timing_shape():
    js = _read(JS)

    assert "progress.answered" in js
    assert "timing.elapsed_minutes" in js
    assert "timing.remaining_minutes" in js
    assert "timing.budget_minutes" in js
    assert 'request(path, {method: "GET"})' in js
    assert "payload && payload.assignment_id" in js


def test_abandon_and_resume_are_server_owned_and_have_no_stale_host_fallback():
    html = _read(INDEX)
    js = _read(JS)
    rir = _read(RIR)

    assert 'id="pause-button"' in html
    assert "/api/demo/sessions/${state.sessionId}/pause" in js
    assert "pauseSession" in js
    assert "error && error.status === 400" in js
    assert "8600" not in rir


def test_rendering_is_dom_safe_and_public_rir_is_allowlisted():
    js = _read(JS)

    assert ".innerHTML" not in js
    assert "document.write" not in js
    assert "eval(" not in js
    assert "PUBLIC_RIR_ZONES" in js
    assert 'new Set(["stem", "options", "feedback"])' in js
    assert "assertPublicAssignment" in js


def test_required_responsive_accessible_and_degraded_states_exist():
    html = _read(INDEX)
    css = _read(CSS)
    js = _read(JS)

    for text in ("加载", "重试", "恢复进度", "SYNTHETIC_DEMO"):
        assert text in html + js
    for selector_or_rule in (
        "max-width: 600px",
        "env(safe-area-inset-bottom)",
        "overflow-x: hidden",
        "overflow-x: auto",
        "prefers-reduced-motion",
        ":focus-visible",
        "font-size: 16px",
        "--card-radius: 8px",
    ):
        assert selector_or_rule in css
    assert "linear-gradient" not in css
    assert "radial-gradient" not in css
    assert "letter-spacing: -" not in css
    assert 'role="radiogroup"' in html
    assert "focusCurrentView" in js


def test_synthetic_marker_depends_only_on_server_payload():
    js = _read(JS)

    assert "payload.synthetic === true" in js
    assert "setSyntheticMarker" in js
    assert "synthetic: true" not in js


def test_report_outcome_codes_are_rendered_as_student_facing_chinese():
    js = _read(JS)

    assert "REPORT_OUTCOME_COPY" in js
    assert 'verified: "验证通过"' in js
    assert 'needs_reinforcement: "继续补强"' in js
    assert 'partial: "本次学习已保存"' in js
    assert "REPORT_OUTCOME_COPY[report.outcome]" in js


def test_learning_checkpoint_uses_explicit_ack_instead_of_replaying_next():
    js = _read(JS)
    checkpoint_renderer = js.split("function renderLearningCheckpoint", 1)[1].split(
        "function acknowledgeLearning", 1
    )[0]

    assert "/api/demo/sessions/${state.sessionId}/learning/ack" in js
    assert "action_id: state.learningActionId" in js
    assert "acknowledgeLearning" in checkpoint_renderer
    assert "advanceSession" not in checkpoint_renderer


def test_recommendation_watch_events_are_bound_to_the_issued_rec_id():
    js = _read(JS)

    assert "/api/demo/sessions/${state.sessionId}/watch" in js
    assert "rec_id: recommendation.rec_id" in js
    assert "watched_seconds:" in js
    assert "completed:" in js
    assert "recordRecommendationWatch" in js
    assert "recordRecommendationWatch(recommendation, false).catch" in js


def test_learning_checkpoint_renders_deep_explanation_and_recommendation_evidence():
    js = _read(JS)

    for field in (
        "explanation.diagnosis",
        "explanation.worked_example",
        "explanation.causal_chain",
        "explanation.exam_strategy",
        "recommendation.title",
        "recommendation.value",
        "recommendation.completion_criterion",
    ):
        assert field in js


def test_report_renders_session_delta_seven_day_reminder_and_failure_plan():
    js = _read(JS)

    for field in (
        "report.delta",
        "report.seven_day_review_at",
        "report.error_summary",
        "report.reinforcement_plan",
    ):
        assert field in js


def test_report_maps_the_canonical_four_state_belief_shape():
    js = _read(JS)

    assert 'P: ["P", "p", "prerequisite_gap"]' in js
    assert 'C: ["C", "c", "concept_confusion"]' in js
    assert 'U: ["U", "u", "uncertain"]' in js
    assert 'P: "前置缺口"' in js


def test_client_timeout_covers_the_llm_full_response_gate_with_a_bounded_buffer():
    js = _read(JS)
    match = re.search(r"controller\.abort\(\); \}, (\d+)\);", js)

    assert match is not None
    timeout_ms = int(match.group(1))
    assert 25_000 <= timeout_ms <= 30_000
    assert 'if (error && error.name === "AbortError")' in js
    assert 'timeout.status = 408' in js
    assert "请求超时，答案和进度仍保留在当前页面。" in js


def test_question_media_is_restricted_to_same_origin():
    js = _read(JS)
    media_function = js.split("function safeMediaUrl", 1)[1].split("function renderRirZone", 1)[0]

    assert "url.origin === location.origin" in media_function
    assert 'url.protocol === "https:"' not in media_function


def test_local_katex_distribution_is_complete_for_offline_rendering():
    css = KATEX / "katex.min.css"
    js = KATEX / "katex.min.js"
    mhchem = KATEX / "contrib" / "mhchem.min.js"
    license_file = KATEX / "LICENSE"
    fonts = list((KATEX / "fonts").glob("*.woff2"))

    assert css.stat().st_size > 10_000
    assert js.stat().st_size > 100_000
    assert mhchem.stat().st_size > 20_000
    assert license_file.stat().st_size > 500
    assert len(fonts) >= 20
