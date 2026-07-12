"""Product contracts for the canonical narrow chemistry loop."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from adapters.store.memory import MemoryStore
from apps.demo_api import create_app
from core.learning.item_catalog import CatalogItem, ItemCatalog
from core.learning.session_service import SessionService


ROOT = Path(__file__).resolve().parents[1]
NODE = "氧化还原反应"
RUN_DEMO = ROOT / "deploy" / "run_demo.sh"
DEMO_REQUIREMENTS = ROOT / "requirements-demo.txt"


def _fixture_catalog() -> ItemCatalog:
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
                options={"A": "正确", "B": "错误"},
                difficulty=(index + 1) / 10,
                item_type="mcq",
                scoring_mode="mcq",
                answer_values=("A",),
                source_label="fixture",
            )
            for index in range(10)
        ]
    )


def _complete(service: SessionService, store: MemoryStore, user_id: str, answer: str) -> dict:
    session_id = service.start_session(user_id, NODE, "30min")["session_id"]
    for index in range(30):
        step = service.next_assignment(session_id)
        if step.get("phase") == "learning":
            service.ack_learning(session_id, step["action_id"])
            continue
        if step.get("done"):
            return step["report"]
        assert "answer" not in step["question"]
        assert "standard_answer" not in step["question"]
        service.submit(
            session_id,
            step["assignment_id"],
            f"{user_id}-{index}",
            answer,
        )
    raise AssertionError("finite canonical session did not complete")


def test_actual_catalog_opens_only_nodes_with_five_deterministic_families() -> None:
    catalog = ItemCatalog.from_default_data()
    opened = catalog.open_nodes()

    assert opened[NODE] >= 5
    assert opened["化学平衡"] >= 5
    assert "实验综合大题" not in opened
    assert "溶液三大守恒" not in opened
    assert "电极反应方程式" not in opened
    assert all(count >= 5 for count in opened.values())


def test_canonical_nodes_endpoint_defaults_to_oxidation_and_hides_closed_nodes() -> None:
    catalog = ItemCatalog.from_default_data()
    client = TestClient(create_app(catalog=catalog, store=MemoryStore(), static_dir=None))

    response = client.get("/api/demo/nodes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_node"] == NODE
    visible = {row["node"] for row in payload["nodes"]}
    assert NODE in visible
    assert "实验综合大题" not in visible
    assert "溶液三大守恒" not in visible


def test_canonical_held_out_pass_and_failure_have_honest_outcomes() -> None:
    catalog = _fixture_catalog()
    passed_store, failed_store = MemoryStore(), MemoryStore()
    passed = _complete(SessionService(catalog, passed_store), passed_store, "passed", "A")
    failed = _complete(SessionService(catalog, failed_store), failed_store, "failed", "B")

    assert passed["outcome"] == "verified"
    assert failed["outcome"] == "needs_reinforcement"
    assert passed["evidence_count"] > 0
    assert failed["evidence_count"] > 0
    assert passed["mastery_probability"] != failed["mastery_probability"]


def test_unknown_or_closed_node_is_rejected_without_an_assignment() -> None:
    client = TestClient(
        create_app(catalog=_fixture_catalog(), store=MemoryStore(), static_dir=None)
    )

    response = client.post(
        "/api/demo/sessions",
        json={"user_id": "student", "node": "不存在的化学节点", "budget_tier": "30min"},
    )

    assert response.status_code == 400
    assert "assignment" not in response.text


def test_frontend_renders_canonical_degraded_retry_state() -> None:
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "apps" / "web" / "app.js").read_text(encoding="utf-8")

    assert 'id="degraded-panel"' in html
    assert 'id="degraded-message"' in html
    assert 'id="retry-button"' in html
    assert "showDegraded" in js


def test_frontend_report_uses_stable_numeric_alignment() -> None:
    css = (ROOT / "apps" / "web" / "app.css").read_text(encoding="utf-8")

    assert ".report-metrics" in css
    assert "font-variant-numeric: tabular-nums" in css
    assert ".report-metrics dd" in css


def test_frontend_declares_empty_favicon_for_clean_qa_logs() -> None:
    html = (ROOT / "apps" / "web" / "index.html").read_text(encoding="utf-8")

    assert '<link rel="icon" href="data:,">' in html


def test_run_demo_is_a_single_worker_local_canonical_entrypoint() -> None:
    script = RUN_DEMO.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in script
    assert "apps.demo_api:app" in script
    assert "--host 127.0.0.1" in script
    assert '"${YHER_DEMO_PORT:-8700}"' in script
    assert "--workers 1" in script
    assert "--env-file" in script
    assert "YHER_ENABLE_PAID_LLM" in script
    assert ".venv-demo" in script
    assert "-m venv" in script
    assert "requirements-demo.txt" in script
    for forbidden in ("kill", "pkill", "8504", "8600", "--reload", "0.0.0.0"):
        assert forbidden not in script


def test_demo_bootstrap_has_a_small_explicit_runtime_dependency_set() -> None:
    requirements = DEMO_REQUIREMENTS.read_text(encoding="utf-8")

    for package in ("fastapi", "uvicorn", "numpy", "pyyaml", "openai", "python-dotenv"):
        assert package in requirements.lower()
    for unrelated in ("streamlit", "sentence-transformers", "supabase"):
        assert unrelated not in requirements.lower()
