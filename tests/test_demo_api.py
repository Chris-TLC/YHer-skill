"""HTTP boundary contracts for the one canonical Demo API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from adapters.store.memory import MemoryStore
from apps.demo_api import create_app
from core.learning.item_catalog import CatalogItem, ItemCatalog


NODE = "氧化还原反应"
FORBIDDEN = {"standard_answer", "final_answers", "answer", "rubric", "analysis"}


def _catalog() -> ItemCatalog:
    return ItemCatalog.from_items(
        [
            CatalogItem(
                item_id=f"private-{index}",
                family_id=f"family-{index}",
                aligned_item_id=f"v3-{index}",
                alignment_status="auto_inherited",
                node_ids=(NODE,),
                stem_blocks=({"para": [{"type": "text", "text": f"question {index}"}]},),
                stem_text=f"question {index}",
                stem_hash=f"hash-{index}",
                stem_normalized=f"question{index}",
                options={"A": "yes", "B": "no"},
                difficulty=(index + 1) / 10,
                item_type="mcq",
                scoring_mode="mcq",
                answer_values=("A",),
                answer_verification_status="passed",
                source_label="fixture",
            )
            for index in range(10)
        ]
    )


def _forbidden_paths(value, path="$"):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN:
                found.append(f"{path}.{key}")
            if key.lower() == "zones" and isinstance(child, list):
                if {str(x).lower() for x in child} & {"answer", "analysis"}:
                    found.append(f"{path}.{key}")
            found.extend(_forbidden_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{path}[{index}]"))
    return found


def _client(tmp_path) -> TestClient:
    app = create_app(catalog=_catalog(), store=MemoryStore(), static_dir=None)
    return TestClient(app)


def test_health_exposes_runtime_and_data_contract(tmp_path):
    response = _client(tmp_path).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["git"]["sha"]
    assert payload["git"]["branch"] == "demo-overnight-20260712"
    assert payload["engine_version"]
    assert "started_at" in payload
    assert payload["counts"]["trusted"] == 10
    assert payload["counts"]["open_nodes"] == 1


def test_pre_submit_response_recursively_contains_no_answer_or_analysis_keys(tmp_path):
    client = _client(tmp_path)
    started = client.post(
        "/api/demo/sessions",
        json={"user_id": "student", "node": NODE, "budget_tier": "30min"},
    )

    assert started.status_code == 201
    payload = started.json()
    assert payload["assignment"]["assignment_id"]
    assert not _forbidden_paths(payload)
    assert "private-" not in started.text


def test_start_persists_grade_and_learning_purpose_and_forbids_extra_fields(tmp_path):
    client = _client(tmp_path)
    accepted = client.post(
        "/api/demo/sessions",
        json={
            "user_id": "student",
            "node": NODE,
            "budget_tier": "1h",
            "grade": "高三",
            "learning_purpose": "exam_prep",
        },
    )
    rejected = client.post(
        "/api/demo/sessions",
        json={
            "user_id": "student",
            "node": NODE,
            "budget_tier": "1h",
            "grade": "高三",
            "learning_purpose": "exam_prep",
            "synthetic": True,
        },
    )

    assert accepted.status_code == 201
    assert accepted.json()["grade"] == "高三"
    assert accepted.json()["learning_purpose"] == "exam_prep"
    assert rejected.status_code == 422


def test_submit_schema_accepts_only_opaque_assignment_submission_and_answer(tmp_path):
    client = _client(tmp_path)
    started = client.post(
        "/api/demo/sessions",
        json={"user_id": "student", "node": NODE, "budget_tier": "30min"},
    ).json()
    sid = started["session_id"]
    aid = started["assignment"]["assignment_id"]

    rejected = client.post(
        f"/api/demo/sessions/{sid}/submit",
        json={
            "assignment_id": aid,
            "submission_id": "submission-1",
            "answer": "A",
            "item_id": "private-0",
        },
    )
    assert rejected.status_code == 422

    accepted = client.post(
        f"/api/demo/sessions/{sid}/submit",
        json={"assignment_id": aid, "submission_id": "submission-1", "answer": "A"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["accepted"] is True
    assert accepted.json()["next_action"] == "next"
    assert "correct" not in accepted.json()
    assert "is_correct" not in accepted.json()
    assert "belief" not in accepted.json()


def test_assignment_is_bound_to_its_server_session(tmp_path):
    client = _client(tmp_path)
    one = client.post(
        "/api/demo/sessions", json={"user_id": "one", "node": NODE, "budget_tier": "30min"}
    ).json()
    two = client.post(
        "/api/demo/sessions", json={"user_id": "two", "node": NODE, "budget_tier": "30min"}
    ).json()

    response = client.post(
        f"/api/demo/sessions/{two['session_id']}/submit",
        json={
            "assignment_id": one["assignment"]["assignment_id"],
            "submission_id": "cross-session",
            "answer": "A",
        },
    )
    assert response.status_code == 404


def test_session_view_and_report_are_projected_and_hide_raw_profile_details(tmp_path):
    client = _client(tmp_path)
    started = client.post(
        "/api/demo/sessions", json={"user_id": "reporter", "node": NODE, "budget_tier": "30min"}
    ).json()
    sid = started["session_id"]
    aid = started["assignment"]["assignment_id"]
    client.post(
        f"/api/demo/sessions/{sid}/submit",
        json={"assignment_id": aid, "submission_id": "report-1", "answer": "A"},
    )

    view = client.get(f"/api/demo/sessions/{sid}")
    early_report = client.get(f"/api/demo/sessions/{sid}/report")
    raw_profile = client.get("/api/demo/profiles/reporter")

    assert view.status_code == 200
    assert early_report.status_code == 409
    assert "mastery_probability" not in early_report.text
    assert "delta" not in early_report.text
    assert raw_profile.status_code == 404
    assert "assignments" not in view.json()
    assert "item_id" not in view.text
    next_step = client.get(f"/api/demo/sessions/{sid}/next").json()
    for index in range(20):
        if next_step.get("phase") == "learning":
            acknowledged = client.post(
                f"/api/demo/sessions/{sid}/learning/ack",
                json={"action_id": next_step["action_id"]},
            )
            assert acknowledged.status_code == 200
            next_step = client.get(f"/api/demo/sessions/{sid}/next").json()
        if next_step.get("done"):
            break
        client.post(
            f"/api/demo/sessions/{sid}/submit",
            json={
                "assignment_id": next_step["assignment_id"],
                "submission_id": f"report-drive-{index}",
                "answer": "A",
            },
        )
        next_step = client.get(f"/api/demo/sessions/{sid}/next").json()
    assert next_step.get("done") is True
    report = client.get(f"/api/demo/sessions/{sid}/report")
    assert report.status_code == 200
    summary = report.json()
    assert {
        "node",
        "outcome",
        "state_label",
        "mastery_probability",
        "delta",
        "evidence_count",
        "as_of",
        "review_due_at",
    }.issubset(summary)
    assert "node_summary" not in summary
    assert summary["outcome"] == "verified"
    assert summary["state_label"] not in {"M", "P", "C", "U"}
    assert summary["evidence_count"] >= 1
    assert "event_id" not in report.text
    assert "evidence" not in report.text.replace("evidence_count", "")


def test_active_report_hides_both_correct_and_wrong_trajectories(tmp_path):
    client = _client(tmp_path)
    for user_id, answer in (("active-correct", "A"), ("active-wrong", "B")):
        started = client.post(
            "/api/demo/sessions", json={"user_id": user_id, "node": NODE, "budget_tier": "30min"}
        ).json()
        client.post(
            f"/api/demo/sessions/{started['session_id']}/submit",
            json={
                "assignment_id": started["assignment"]["assignment_id"],
                "submission_id": f"{user_id}-one",
                "answer": answer,
            },
        )
        report = client.get(f"/api/demo/sessions/{started['session_id']}/report")
        assert report.status_code == 409
        assert "mastery_probability" not in report.text
        assert "state_label" not in report.text


def test_resume_is_idempotent_for_an_active_session_and_returns_same_assignment(tmp_path):
    client = _client(tmp_path)
    started = client.post(
        "/api/demo/sessions", json={"user_id": "refresh", "node": NODE, "budget_tier": "30min"}
    ).json()

    resumed = client.post(f"/api/demo/sessions/{started['session_id']}/resume")

    assert resumed.status_code == 200
    assert resumed.json()["assignment"]["assignment_id"] == started["assignment"]["assignment_id"]


def test_api_routes_are_registered_before_optional_same_origin_static_mount(tmp_path):
    static_dir = tmp_path / "web"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("demo", encoding="utf-8")
    app = create_app(catalog=_catalog(), store=MemoryStore(), static_dir=static_dir)
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/").text == "demo"
    assert app.routes[-1].name == "student-web"
