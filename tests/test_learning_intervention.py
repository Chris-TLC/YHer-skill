"""Contracts for the diagnosis-to-learning intervention checkpoint."""

from __future__ import annotations

import copy
import importlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adapters.store.memory import MemoryStore
from core.learning.item_catalog import CatalogItem, ItemCatalog
from core.learning.session_service import SessionService
from core.learning.events import project_student
from core.learning.session_service import AssignmentError
from apps.demo_api import create_app


NODE = "氧化还原反应"


def _catalog() -> ItemCatalog:
    return ItemCatalog.from_items(
        [
            CatalogItem(
                item_id=f"item-{index}",
                family_id=f"family-{index}",
                aligned_item_id=f"v3-{index}",
                alignment_status="auto_inherited",
                node_ids=(NODE,),
                stem_blocks=({"para": [{"type": "text", "text": f"question {index}"}]},),
                stem_text=f"question {index}",
                stem_hash=f"hash-{index}",
                stem_normalized=f"question{index}",
                options={"A": "correct", "B": "wrong"},
                difficulty=(index + 1) / 10,
                item_type="mcq",
                scoring_mode="mcq",
                answer_values=("A",),
                rubric=({"point_id": "p1", "desc": f"criterion {index}"},),
                answer_verification_status="passed",
                answer_verification_confidence=0.99,
                solution_steps=(f"standard solution {index}",),
                solution_key_insight=f"verified insight {index}",
                source_label=f"202{index % 5}上海卷",
            )
            for index in range(10)
        ]
    )


class FakeCurriculum:
    def recommend(self, **kwargs):
        session_id = kwargs["session_id"]
        action_id = kwargs["action_id"]
        return {
            "recommendations": [],
            "bindings": {},
            "rec_served": {
                "event_id": f"rec:{session_id}:{action_id}",
                "session_id": session_id,
                "action_id": action_id,
                "served": [],
                "unserved_topk": [],
            },
        }


class CapturingBudgetCurriculum(FakeCurriculum):
    def __init__(self):
        self.budgets: list[dict] = []

    def recommend(self, **kwargs):
        self.budgets.append(dict(kwargs["budget"]))
        return super().recommend(**kwargs)


class CapturingExplanationProvider:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []

    def generate(self, *, prompt: str, context: dict):
        self.calls.append((prompt, context))
        if self.fail:
            raise ConnectionError("provider unavailable")
        return {
            "content": {
                "title": "电子守恒复盘",
                "diagnosis": "先确定变价元素。",
                "worked_example": "把题干中的 2 mol 代入电子守恒关系。",
                "causal_chain": ["化合价改变", "电子转移", "系数配平"],
                "exam_strategy": ["先标价，再验电荷"],
                "analogy_used": False,
                "model": "must-not-leak",
            },
            "usage": {"input_tokens": 321, "output_tokens": 123},
            "cost_yuan": 0.031,
            "model_returned": "must-not-leak",
        }


class BlockingExplanationProvider(CapturingExplanationProvider):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, *, prompt: str, context: dict):
        self.entered.set()
        assert self.release.wait(timeout=5)
        return super().generate(prompt=prompt, context=context)


def test_production_env_uses_fast_explanation_model_without_changing_grader_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adapters import llm_client as llm_module

    class FakeLLMClient:
        instances: list["FakeLLMClient"] = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.calls: list[list[dict]] = []
            self.instances.append(self)

        def chat(self, messages, **_kwargs):
            self.calls.append(messages)
            return {
                "content": json.dumps(
                    {
                        "correct": False,
                        "error_code": "missing_conservation",
                        "confidence": 0.8,
                        "likelihood": [0.2, 0.2, 0.6, 0.2],
                    }
                ),
                "usage": {"input_tokens": 120, "output_tokens": 40},
                "cost_yuan": 0.004,
                "model_returned": "internal-only",
            }

    monkeypatch.setenv("YHER_ENABLE_PAID_LLM", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.setattr(llm_module, "LLMClient", FakeLLMClient)

    app = create_app(
        catalog=_catalog(), store=MemoryStore(), curriculum=FakeCurriculum(), static_dir=None
    )
    service = app.state.session_service

    assert len(FakeLLMClient.instances) == 2
    assert service.explanation_provider.client is FakeLLMClient.instances[0]
    assert service.llm_grader.client is FakeLLMClient.instances[1]
    assert FakeLLMClient.instances[0].init_kwargs["model"] == "deepseek-chat"
    assert "model" not in FakeLLMClient.instances[1].init_kwargs

    free_item = CatalogItem(
        item_id="free-item",
        family_id="free-family",
        aligned_item_id="free-aligned",
        alignment_status="auto_inherited",
        node_ids=(NODE,),
        stem_blocks=({"para": [{"type": "text", "text": "解释电子守恒"}]},),
        stem_text="解释电子守恒",
        stem_hash="free-hash",
        stem_normalized="解释电子守恒",
        options={},
        difficulty=0.8,
        item_type="free",
        scoring_mode="free_llm",
        answer_values=("氧化失电子数等于还原得电子数",),
        rubric=({"point_id": "conservation", "desc": "写出电子守恒"},),
        source_label="2025上海卷",
    )
    graded = service.llm_grader(free_item, "只写了化合价变化")

    assert graded == {
        "correct": False,
        "error_code": "missing_conservation",
        "confidence": 0.8,
        "likelihood": [0.2, 0.2, 0.6, 0.2],
        "usage": {"input_tokens": 120, "output_tokens": 40},
        "cost_yuan": 0.004,
    }
    assert "写出电子守恒" in FakeLLMClient.instances[1].calls[0][1]["content"]
    assert "只写了化合价变化" in FakeLLMClient.instances[1].calls[0][1]["content"]


def test_deepseek_downgrade_guard_only_applies_to_requested_pro_model() -> None:
    from adapters.llm_client import LLMClient

    client = object.__new__(LLMClient)
    client.provider = "deepseek"
    client.model = "deepseek-chat"
    client._validate_model("deepseek-chat")

    client.model = "deepseek-v4-pro"
    with pytest.raises(ValueError, match="模型降级"):
        client._validate_model("deepseek-chat")


def test_openai_compatible_transport_timeout_stays_below_product_deadline() -> None:
    from types import SimpleNamespace

    from adapters.llm_client import LLMClient

    class CapturingCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                model="deepseek-chat",
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    prompt_cache_hit_tokens=0,
                ),
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            )

    completions = CapturingCompletions()
    client = object.__new__(LLMClient)
    client.provider = "deepseek"
    client.model = "deepseek-chat"
    client.config = LLMClient.PROVIDER_CONFIGS["deepseek"]
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    client._chat_openai([{"role": "user", "content": "test"}], 16, 0.0)

    assert completions.kwargs["timeout"] == 18.0


def test_slow_explanation_does_not_block_health_or_another_sessions_next() -> None:
    provider = BlockingExplanationProvider()
    store = MemoryStore()
    app = create_app(
        catalog=_catalog(),
        store=store,
        curriculum=FakeCurriculum(),
        explanation_provider=provider,
        static_dir=None,
    )
    service = app.state.session_service
    slow_session = service.start_session("slow-provider", NODE, "30min")["session_id"]
    service.next_assignment(slow_session)
    state = store.load_session(slow_session)
    assignment = next(iter(state["assignments"].values()))
    assignment["submitted"] = True
    state["current_assignment_id"] = None
    state["phase"] = "learning"
    store.save_session(slow_session, state)
    store.append_event(
        "slow-provider",
        {
            "kind": "answer_scored",
            "session_id": slow_session,
            "assignment_id": assignment["assignment_id"],
            "phase": "diagnostic",
            "correct": True,
        },
    )
    other_session = service.start_session("fast-path", NODE, "30min")["session_id"]

    slow_client, fast_client = TestClient(app), TestClient(app)
    with ThreadPoolExecutor(max_workers=1) as executor:
        slow = executor.submit(
            slow_client.get, f"/api/demo/sessions/{slow_session}/next"
        )
        assert provider.entered.wait(timeout=2)
        started = time.perf_counter()
        health = fast_client.get("/health")
        health_ms = (time.perf_counter() - started) * 1_000
        started = time.perf_counter()
        deterministic = fast_client.get(f"/api/demo/sessions/{other_session}/next")
        deterministic_ms = (time.perf_counter() - started) * 1_000
        provider.release.set()
        slow_response = slow.result(timeout=5)

    assert health.status_code == 200
    assert health_ms < 500
    assert deterministic.status_code == 200
    assert deterministic.json()["phase"] == "diagnostic"
    assert deterministic_ms < 500
    assert slow_response.status_code == 200
    assert slow_response.json()["phase"] == "learning"


class FailLearningCheckpointSaveStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.fail_learning_save = False

    def save_session(self, session_id, session):
        if (
            self.fail_learning_save
            and session.get("phase") == "learning"
            and (session.get("learning") or {}).get("view") is not None
        ):
            self.fail_learning_save = False
            raise OSError("simulated checkpoint save crash")
        super().save_session(session_id, session)


class LosingWatchAppendStore(MemoryStore):
    lose_next_watch = False

    def append_event_once(self, user_id, event):
        if self.lose_next_watch and event.get("kind") == "watch_proxy":
            self.lose_next_watch = False
            competing = dict(event)
            competing["watched_seconds"] = float(event["watched_seconds"]) + 1
            competing["completed"] = not bool(event["completed"])
            assert super().append_event_once(user_id, competing) is True
            return False
        return super().append_event_once(user_id, event)


def _drive_to_learning(service: SessionService, session_id: str) -> dict:
    for index in range(20):
        step = service.next_assignment(session_id)
        if step.get("phase") == "learning":
            return step
        service.submit(session_id, step["assignment_id"], f"diag-{index}", "A")
    raise AssertionError("diagnosis never reached learning checkpoint")


def test_learning_checkpoint_blocks_practice_until_explicit_idempotent_ack() -> None:
    store = MemoryStore()
    service = SessionService(_catalog(), store, curriculum=FakeCurriculum())
    session_id = service.start_session("student", NODE, "30min")["session_id"]

    checkpoint = _drive_to_learning(service, session_id)
    replay = service.next_assignment(session_id)

    assert checkpoint["phase"] == "learning"
    assert checkpoint["next_action"] == "ack_learning"
    assert replay == checkpoint
    assert store.load_session(session_id)["asked"]["practice"] == 0

    acknowledged = service.ack_learning(session_id, checkpoint["action_id"])
    acknowledged_again = service.ack_learning(session_id, checkpoint["action_id"])
    next_assignment = service.next_assignment(session_id)

    assert acknowledged == acknowledged_again
    assert acknowledged["accepted"] is True
    assert next_assignment["phase"] in {"practice", "held_out"}
    session = store.load_session(session_id)
    for left, right in (
        ("diagnostic", "practice"),
        ("diagnostic", "held_out"),
        ("practice", "held_out"),
    ):
        assert set(session["partitions"][left]["ids"]).isdisjoint(
            session["partitions"][right]["ids"]
        )
        assert set(session["partitions"][left]["families"]).isdisjoint(
            session["partitions"][right]["families"]
        )


def test_pre_m3_session_without_learning_state_is_migrated_on_load() -> None:
    store = MemoryStore()
    service = SessionService(_catalog(), store, curriculum=FakeCurriculum())
    session_id = service.start_session("legacy-session", NODE, "30min")["session_id"]
    legacy = store.load_session(session_id)
    legacy.pop("learning")
    store.save_session(session_id, legacy)

    checkpoint = _drive_to_learning(service, session_id)

    assert checkpoint["phase"] == "learning"
    migrated = store.load_session(session_id)["learning"]
    assert migrated["action_id"] == checkpoint["action_id"]
    assert migrated["acked"] is False


def test_explanation_prompt_binds_private_evidence_but_public_view_hides_provider_metadata() -> None:
    provider = CapturingExplanationProvider()
    store = MemoryStore()
    service = SessionService(
        _catalog(),
        store,
        curriculum=FakeCurriculum(),
        explanation_provider=provider,
    )
    session_id = service.start_session("explained", NODE, "30min")["session_id"]

    checkpoint = _drive_to_learning(service, session_id)

    assert len(provider.calls) == 1
    prompt, context = provider.calls[0]
    assert context["node"] == NODE
    assert context["evidence"]
    evidence = context["evidence"][0]
    assert evidence["difficulty"] in {(index + 1) / 10 for index in range(10)}
    assert evidence["source"] in {f"202{index % 5}上海卷" for index in range(10)}
    assert evidence["solution_steps"]
    assert evidence["expected_response"]
    assert evidence["result"] in {"correct", "incorrect", "deferred"}
    for required in ("数据代入", "变量", "因果链", "难度", "比喻", "必要"):
        assert required in prompt
    assert checkpoint["explanation"]["status"] == "generated"
    assert checkpoint["explanation"]["title"] == f"{NODE}标准解复盘"
    assert "model" not in checkpoint["explanation"]
    assert "provider" not in str(checkpoint).lower()
    assert "must-not-leak" not in str(checkpoint)
    generation_events = [
        event for event in store.all_events("explained") if event["kind"] == "explanation_generated"
    ]
    assert len(generation_events) == 1
    assert generation_events[0]["usage"] == {"input_tokens": 321, "output_tokens": 123}
    assert generation_events[0]["cost_yuan"] == 0.031
    assert "model" not in generation_events[0]
    assert "provider" not in generation_events[0]


def test_unverified_incomplete_item_never_reaches_the_explanation_provider() -> None:
    catalog = _catalog()
    trusted = catalog.items["item-1"]
    incomplete = CatalogItem(
        item_id="incomplete",
        family_id="incomplete-family",
        aligned_item_id="incomplete-v3",
        alignment_status="auto_inherited",
        node_ids=(NODE,),
        stem_blocks=(
            {
                "para": [
                    {
                        "type": "text",
                        "text": "6FeCl2 + NaClO3 + 6HCl = ___ + 3H2O + ___",
                    }
                ]
            },
        ),
        stem_text="6FeCl2 + NaClO3 + 6HCl = ___ + 3H2O + ___",
        stem_hash="incomplete-hash",
        stem_normalized="incomplete",
        options={},
        difficulty=0.5,
        item_type="free",
        scoring_mode="free_llm",
        answer_values=("6FeCl3",),
        rubric=({"point_id": "ans", "desc": "6FeCl3"},),
        answer_verification_status="needs_review",
        answer_verification_confidence=0.623,
        solution_steps=(),
        source_label="incomplete source",
    )
    service = SessionService(ItemCatalog.from_items([incomplete, trusted]), MemoryStore())
    session = {
        "session_id": "source-contract",
        "user_id": "source-contract-user",
        "node": NODE,
        "grade": "高二",
        "learning_purpose": "review",
        "assignments": {
            "unverified-assignment": {
                "assignment_id": "unverified-assignment",
                "item_id": incomplete.item_id,
                "phase": "diagnostic",
                "submitted": True,
            },
            "verified-assignment": {
                "assignment_id": "verified-assignment",
                "item_id": trusted.item_id,
                "phase": "diagnostic",
                "submitted": True,
            },
        },
    }
    service.store.append_event(
        session["user_id"],
        {
            "kind": "answer_scored",
            "session_id": session["session_id"],
            "assignment_id": "unverified-assignment",
            "phase": "diagnostic",
            "correct": True,
        },
    )
    service.store.append_event(
        session["user_id"],
        {
            "kind": "answer_scored",
            "session_id": session["session_id"],
            "assignment_id": "verified-assignment",
            "phase": "diagnostic",
            "correct": True,
        },
    )

    context = service._explanation_context(session)
    serialized = json.dumps(context, ensure_ascii=False)

    assert context["result_summary"] == {
        "total": 2,
        "correct": 2,
        "incorrect": 0,
        "deferred": 0,
    }
    assert len(context["evidence"]) == 1
    assert context["evidence"][0]["node"] == NODE
    assert context["evidence"][0]["question"] == trusted.stem_text
    assert context["evidence"][0]["solution_steps"] == list(trusted.solution_steps)
    assert "analysis_blocks" not in context["evidence"][0]
    assert incomplete.stem_text not in serialized
    assert "6FeCl3" not in serialized


def test_authoritative_projection_titles_a_prerequisite_anchor_with_its_node() -> None:
    module = importlib.import_module("core.learning.explanations")
    context = {
        "node": NODE,
        "result_summary": {
            "total": 2,
            "correct": 0,
            "incorrect": 2,
            "deferred": 0,
        },
        "evidence": [
            {
                "node": "化学计量（摩尔/阿伏伽德罗）",
                "question": "等质量的乙烯与丙烯，碳氢质量比是否相等？",
                "difficulty": 0.25,
                "source": "2022上海卷",
                "result": "incorrect",
                "expected_response": ["相等"],
                "solution_steps": ["二者最简式均为CH2，因此碳氢质量比相等。"],
                "key_insight": "先比较最简式。",
            }
        ],
    }

    explanation, _audit = module.generate_explanation(None, context)

    assert explanation["title"] == "化学计量（摩尔/阿伏伽德罗）标准解复盘"


def test_authoritative_projection_builds_scaffold_only_from_verified_evidence() -> None:
    module = importlib.import_module("core.learning.explanations")

    class InventingProvider:
        def generate(self, **_kwargs):
            return {
                "content": {
                    "title": "模型标题",
                    "diagnosis": "模型诊断",
                    "worked_example": "模型新增产物X，价态从+7降到-3。",
                    "causal_chain": ["模型新增产物X"],
                    "exam_strategy": ["猜测反应产物"],
                    "analogy_used": True,
                }
            }

    question = "向150mL稀硝酸中加入6.4g铜，判断铜能否完全消耗。"
    key_insight = "先由质量和浓度求物质的量，再按方程式比较。"
    steps = [
        "6.4g铜为0.1mol。",
        "稀硝酸为0.45mol，按已核验方程式比较后铜完全消耗。",
    ]
    context = {
        "node": NODE,
        "result_summary": {
            "total": 6,
            "correct": 1,
            "incorrect": 5,
            "deferred": 0,
        },
        "evidence": [
            {
                "node": NODE,
                "question": question,
                "difficulty": 0.9,
                "source": "2022上海卷",
                "result": "incorrect",
                "expected_response": ["A"],
                "solution_steps": steps,
                "key_insight": key_insight,
            }
        ],
    }

    explanation, _audit = module.generate_explanation(InventingProvider(), context)
    rendered = json.dumps(explanation, ensure_ascii=False)

    assert "零基础起点" in explanation["worked_example"]
    assert "难度支架" in explanation["worked_example"]
    assert "错题支架" in explanation["worked_example"]
    assert "5 道未通过" in explanation["worked_example"]
    assert "已核验步骤" in explanation["worked_example"]
    assert question in explanation["worked_example"]
    assert key_insight in explanation["worked_example"]
    assert all(step in explanation["worked_example"] for step in steps)
    assert "标准答案：A" in explanation["worked_example"]
    assert explanation["causal_chain"] == steps
    assert "模型新增产物X" not in rendered
    assert "价态从+7降到-3" not in rendered


def test_authoritative_projection_replaces_an_incomplete_key_insight_fragment() -> None:
    module = importlib.import_module("core.learning.explanations")
    fragment = "先由质量求物质的量，再根据化学方程"
    context = {
        "node": NODE,
        "result_summary": {
            "total": 1,
            "correct": 1,
            "incorrect": 0,
            "deferred": 0,
        },
        "evidence": [
            {
                "node": NODE,
                "question": "比较反应物的物质的量。",
                "difficulty": 0.25,
                "source": "上海卷",
                "result": "correct",
                "expected_response": ["A"],
                "solution_steps": ["按已核验标准解完成比较。"],
                "key_insight": fragment,
            }
        ],
    }

    explanation, _audit = module.generate_explanation(None, context)
    zero_start = explanation["worked_example"].splitlines()[0]

    assert fragment not in zero_start
    assert zero_start == "零基础起点：先明确题目要求、已知条件和待求结论。"


def test_authoritative_projection_scales_depth_for_difficulty_and_errors() -> None:
    module = importlib.import_module("core.learning.explanations")
    question = "比较两种给定物质的量。"
    key_insight = "先统一单位，再比较题目给定量。"
    steps = ["读取题干数据。", "统一单位后比较。"]

    def projection(*, difficulty: float, correct: int, incorrect: int) -> dict:
        context = {
            "node": NODE,
            "result_summary": {
                "total": correct + incorrect,
                "correct": correct,
                "incorrect": incorrect,
                "deferred": 0,
            },
            "evidence": [
                {
                    "node": NODE,
                    "question": question,
                    "difficulty": difficulty,
                    "source": "上海卷",
                    "result": "correct" if not incorrect else "incorrect",
                    "expected_response": ["相等"],
                    "solution_steps": steps,
                    "key_insight": key_insight,
                }
            ],
        }
        explanation, _audit = module.generate_explanation(None, context)
        return explanation

    baseline = projection(difficulty=0.25, correct=3, incorrect=0)
    high_difficulty = projection(difficulty=0.9, correct=3, incorrect=0)
    many_errors = projection(difficulty=0.25, correct=2, incorrect=6)
    high_with_errors = projection(difficulty=0.9, correct=9, incorrect=8)

    for explanation in (baseline, high_difficulty, many_errors, high_with_errors):
        assert question in explanation["worked_example"]
        assert key_insight in explanation["worked_example"]
        assert all(step in explanation["worked_example"] for step in steps)
        assert "标准答案：相等" in explanation["worked_example"]
    assert "难度支架" not in baseline["worked_example"]
    assert "错题支架" not in baseline["worked_example"]
    assert "难度支架" in high_difficulty["worked_example"]
    assert "错题支架" not in high_difficulty["worked_example"]
    assert "难度支架" not in many_errors["worked_example"]
    assert "错题支架" in many_errors["worked_example"]
    assert "6 道未通过" in many_errors["worked_example"]
    assert "难度支架" in high_with_errors["worked_example"]
    assert "错题支架" in high_with_errors["worked_example"]
    assert "8 道未通过" in high_with_errors["worked_example"]
    assert "验算闭环" in high_difficulty["worked_example"]
    assert "验算闭环" in many_errors["worked_example"]
    assert "验算闭环" in high_with_errors["worked_example"]
    assert len(high_difficulty["worked_example"]) > len(baseline["worked_example"])
    assert len(many_errors["worked_example"]) > len(baseline["worked_example"])
    assert len(high_with_errors["worked_example"]) > len(high_difficulty["worked_example"])


def test_authoritative_projection_removes_wrong_chemistry_and_all_correct_blame() -> None:
    module = importlib.import_module("core.learning.explanations")

    class ContradictoryProvider:
        def generate(self, **_kwargs):
            return {
                "content": {
                    "title": "错误标题",
                    "diagnosis": "你作答错误，而且基础薄弱。",
                    "worked_example": (
                        "NaClO3 中没有 Cl；Cl 从 +5 变为 0 得 5e，"
                        "即使 6 与 5 不等也可写出 3Cl2。"
                    ),
                    "causal_chain": ["6FeCl2 只有 6 个 Cl"],
                    "exam_strategy": ["忽略电子守恒"],
                    "analogy_used": True,
                },
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "cost_yuan": 0.01,
            }

    context = {
        "node": NODE,
        "result_summary": {
            "total": 4,
            "correct": 4,
            "incorrect": 0,
            "deferred": 0,
            "unverified_for_teaching": 1,
        },
        "evidence": [
            {
                "question": "在给定条件下，哪种物质能完全消耗？",
                "difficulty": 0.75,
                "source": "2022上海卷",
                "result": "correct",
                "expected_response": ["A"],
                "solution_steps": [
                    "6.4g铜为0.1mol，稀硝酸为0.45mol；按方程式稀硝酸过量，铜完全消耗。"
                ],
                "key_insight": "先由质量和浓度求物质的量，再按方程式比较。",
            }
        ],
    }

    explanation, audit = module.generate_explanation(ContradictoryProvider(), context)
    rendered = json.dumps(explanation, ensure_ascii=False)

    assert explanation["status"] == "generated"
    assert "4/4" in explanation["diagnosis"]
    assert "作答错误" not in rendered
    assert "薄弱" not in rendered
    assert "NaClO3" not in rendered
    assert "3Cl2" not in rendered
    assert "6 与 5" not in rendered
    assert context["evidence"][0]["solution_steps"][0] in explanation["worked_example"]
    assert explanation["causal_chain"] == context["evidence"][0]["solution_steps"]
    assert explanation["analogy_used"] is False
    assert audit["generation_status"] == "generated"
    assert audit["grounding_status"] == "authoritative_projection"


def test_no_verified_solution_skips_provider_instead_of_guessing() -> None:
    module = importlib.import_module("core.learning.explanations")

    class MustNotRun:
        def generate(self, **_kwargs):
            raise AssertionError("unverified evidence must not be sent to the provider")

    explanation, audit = module.generate_explanation(
        MustNotRun(),
        {
            "node": NODE,
            "result_summary": {
                "total": 1,
                "correct": 1,
                "incorrect": 0,
                "deferred": 0,
                "unverified_for_teaching": 1,
            },
            "evidence": [],
        },
    )

    assert explanation["status"] == "evidence_fallback"
    assert "1/1" in explanation["diagnosis"]
    assert audit["generation_status"] == "evidence_fallback"
    assert audit["grounding_status"] == "no_verified_solution"


def test_explanation_provider_failure_is_honest_and_does_not_block_ack() -> None:
    provider = CapturingExplanationProvider(fail=True)
    store = MemoryStore()
    service = SessionService(
        _catalog(),
        store,
        curriculum=FakeCurriculum(),
        explanation_provider=provider,
    )
    session_id = service.start_session("offline", NODE, "30min")["session_id"]

    checkpoint = _drive_to_learning(service, session_id)

    assert checkpoint["explanation"]["status"] == "offline_fallback"
    assert checkpoint["explanation"]["causal_chain"]
    assert checkpoint["explanation"]["analogy_used"] is False
    assert "provider unavailable" not in str(checkpoint)
    acknowledged = service.ack_learning(session_id, checkpoint["action_id"])
    assert acknowledged["accepted"] is True
    generation_events = [
        event for event in store.all_events("offline") if event["kind"] == "explanation_generated"
    ]
    assert len(generation_events) == 1
    assert generation_events[0]["generation_status"] == "offline_fallback"


def test_recommendation_receives_same_full_session_time_as_learning_view() -> None:
    curriculum = CapturingBudgetCurriculum()
    service = SessionService(
        _catalog(), MemoryStore(), clock=lambda: 1_000.0, curriculum=curriculum
    )
    session_id = service.start_session(
        "timed-recommendation", NODE, "30min"
    )["session_id"]

    checkpoint = _drive_to_learning(service, session_id)

    assert checkpoint["timing"]["remaining_minutes"] == 30.0
    assert curriculum.budgets == [
        {
            **service.store.load_session(session_id)["budget"],
            "session_remaining_minutes": 30.0,
        }
    ]


def test_billed_invalid_explanation_retains_usage_and_cost_in_fallback_audit() -> None:
    module = importlib.import_module("core.learning.explanations")

    class BilledInvalidProvider:
        def generate(self, **_kwargs):
            return {
                "content": "",
                "usage": {"input_tokens": 3_242, "output_tokens": 1_400},
                "cost_yuan": 0.018911,
                "model_returned": "must-not-be-retained",
            }

    explanation, audit = module.generate_explanation(
        BilledInvalidProvider(),
        {
            "node": NODE,
            "evidence": [
                {
                    "question": "已核验例题",
                    "source": "测试卷",
                    "result": "incorrect",
                    "expected_response": ["A"],
                    "solution_steps": ["标准步骤"],
                }
            ],
        },
    )

    assert explanation["status"] == "offline_fallback"
    assert audit["generation_status"] == "offline_fallback"
    assert audit["usage"] == {"input_tokens": 3_242, "output_tokens": 1_400}
    assert audit["cost_yuan"] == 0.018911
    assert "model" not in audit
    assert "provider" not in audit


def test_explanation_prompt_makes_authoritative_answers_and_conservation_hard_constraints() -> None:
    module = importlib.import_module("core.learning.explanations")

    prompt = module.build_explanation_prompt(
        {
            "node": NODE,
            "evidence": [
                {
                    "question": "配平 6FeCl2 + NaClO3 + 6HCl",
                    "expected_response": ["6FeCl3", "NaCl"],
                    "criteria": [{"desc": "电子守恒与原子守恒"}],
                    "result": "incorrect",
                }
            ],
        }
    )

    for required in (
        "expected_response 是权威答案",
        "不得引入权威答案中没有的产物",
        "原子守恒",
        "电荷守恒",
        "电子得失守恒",
        "任何一步不自洽就不得输出",
    ):
        assert required in prompt


@pytest.mark.parametrize(
    "bad_text",
    (
        "NaClO3 中 Cl 从 +5 价降到 0 价，得到 5 个电子，通常生成 Cl2。",
        "NaClO3 中无 Cl，因此先写成 6FeCl3·NaCl，再改为 NaCl。",
        "6FeCl2 中有 6 个 Cl；检查时再按 6×2=12 个 Cl。",
        "O2F2 中 O 从 +1 到 0，四份 O2F2 共得到 1×4=4 个电子。",
        "失电子 6 个、得电子 5 个，电子得失不相等，但仍可继续配平。",
    ),
)
def test_observed_hard_chemistry_contradictions_fail_closed(bad_text: str) -> None:
    module = importlib.import_module("core.learning.explanations")

    class BadProvider:
        def generate(self, **_kwargs):
            return {
                "content": {
                    "title": "错误讲解",
                    "diagnosis": "需要复盘。",
                    "worked_example": bad_text,
                    "causal_chain": [bad_text],
                    "exam_strategy": ["最后验算"],
                    "analogy_used": False,
                },
                "usage": {"input_tokens": 2_066, "output_tokens": 822},
                "cost_yuan": 0.0116123,
            }

    context = {
        "node": NODE,
        "evidence": [
            {
                "question": (
                    "6FeCl2 + NaClO3 + 6HCl = 6FeCl3 + 3H2O + NaCl；"
                    "H2S + 4O2F2 = SF6 + 2HF + 4O2"
                ),
                "expected_response": ["6FeCl3", "NaCl", "1:4"],
                "criteria": [{"desc": "原子、电子和电荷守恒"}],
                "solution_steps": ["按权威标准解逐项核对原子、电子和电荷守恒。"],
                "result": "incorrect",
            }
        ],
    }

    explanation, audit = module.generate_explanation(BadProvider(), context)

    assert explanation["status"] == "generated"
    assert audit["generation_status"] == "generated"
    assert audit["quality_status"] == "projected"
    assert audit["quality_failures"]
    assert audit["usage"] == {"input_tokens": 2_066, "output_tokens": 822}
    assert audit["cost_yuan"] == pytest.approx(0.0116123)
    assert bad_text not in str(explanation)


def test_semantic_validator_rejection_aggregates_billed_generation_and_review() -> None:
    module = importlib.import_module("core.learning.explanations")

    class ReviewedProvider:
        def generate(self, **_kwargs):
            return {
                "content": {
                    "title": "表面完整",
                    "diagnosis": "需要复盘。",
                    "worked_example": "逐步配平并检查。",
                    "causal_chain": ["标价", "守恒", "复核"],
                    "exam_strategy": ["最后验算"],
                    "analogy_used": False,
                },
                "usage": {"input_tokens": 800, "output_tokens": 400},
                "cost_yuan": 0.01,
            }

        def validate(self, **_kwargs):
            return {
                "valid": False,
                "errors": ["产物与 expected_response 冲突"],
                "usage": {"input_tokens": 500, "output_tokens": 80},
                "cost_yuan": 0.003,
            }

    context = {
        "node": NODE,
        "evidence": [
            {
                "question": "配平氧化还原反应",
                "expected_response": ["6FeCl3", "NaCl"],
                "criteria": [{"desc": "电子守恒"}],
                "solution_steps": ["按权威标准解完成配平并检查守恒。"],
                "result": "incorrect",
            }
        ],
    }

    explanation, audit = module.generate_explanation(ReviewedProvider(), context)

    assert explanation["status"] == "generated"
    assert audit["generation_status"] == "generated"
    assert audit["grounding_status"] == "authoritative_projection"
    assert audit["usage"] == {"input_tokens": 800, "output_tokens": 400}
    assert audit["cost_yuan"] == pytest.approx(0.01)


def test_chat_explanation_semantic_review_is_evidence_bound_and_deterministic() -> None:
    module = importlib.import_module("core.learning.explanations")

    class FakeReviewClient:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return {
                "content": json.dumps(
                    {"valid": False, "errors": ["电子得失不守恒"]}, ensure_ascii=False
                ),
                "usage": {"input_tokens": 500, "output_tokens": 80},
                "cost_yuan": 0.003,
            }

    client = FakeReviewClient()
    provider = module.ChatExplanationProvider(client)
    context = {
        "node": NODE,
        "evidence": [
            {
                "question": "配平方程式",
                "expected_response": ["6FeCl3", "NaCl"],
                "criteria": [{"desc": "电子守恒"}],
            }
        ],
    }
    result = provider.validate(
        explanation={
            "title": "讲解",
            "diagnosis": "复盘",
            "worked_example": "逐步计算",
            "causal_chain": ["标价", "守恒"],
            "exam_strategy": ["验算"],
            "analogy_used": False,
        },
        context=context,
    )

    assert result["valid"] is False
    assert result["errors"] == ["电子得失不守恒"]
    messages, kwargs = client.calls[0]
    assert "expected_response" in messages[-1]["content"]
    assert "原子守恒" in messages[-1]["content"]
    assert "电荷守恒" in messages[-1]["content"]
    assert "电子得失守恒" in messages[-1]["content"]
    assert kwargs["temperature"] == 0.0


def test_offline_fallback_is_evidence_bound_and_safe_for_non_redox_nodes() -> None:
    module = importlib.import_module("core.learning.explanations")
    fallback = module.offline_fallback(
        {
            "node": "化学平衡",
            "evidence": [
                {
                    "question": "增大压强后，平衡如何移动？",
                    "source": "2024上海卷",
                    "result": "incorrect",
                    "criteria": [{"desc": "比较反应前后气体物质的量"}],
                }
            ],
        }
    )

    rendered = str(fallback)
    assert "增大压强" in rendered
    assert "2024上海卷" in rendered
    assert "化合价" not in rendered
    assert "电子转移" not in rendered
    assert "电子守恒" not in rendered


def test_chat_explanation_provider_scales_depth_with_actual_difficulty() -> None:
    module = importlib.import_module("core.learning.explanations")

    class FakeChatClient:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return {
                "content": (
                    '{"title":"t","diagnosis":"d","worked_example":"w",'
                    '"causal_chain":["c"],"exam_strategy":["s"],"analogy_used":false}'
                ),
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "cost_yuan": 0.01,
                "model_returned": "internal-only",
            }

    client = FakeChatClient()
    provider = module.ChatExplanationProvider(client)
    base = {"node": NODE, "grade": "高二", "learning_purpose": "review"}

    provider.generate(prompt="low", context={**base, "evidence": [{"difficulty": 0.2}]})
    provider.generate(prompt="high", context={**base, "evidence": [{"difficulty": 0.9}]})

    assert client.calls[0][1]["max_tokens"] < client.calls[1][1]["max_tokens"]
    assert client.calls[0][1]["temperature"] == 0.2
    assert client.calls[1][0][-1] == {"role": "user", "content": "high"}


@pytest.mark.parametrize(
    ("difficulty", "expected_tokens"),
    ((0.2, 3_200), (0.5, 4_200), (0.9, 5_200)),
)
def test_reasoning_model_budget_covers_reasoning_and_structured_explanation(
    difficulty: float, expected_tokens: int
) -> None:
    module = importlib.import_module("core.learning.explanations")

    assert module._dynamic_max_tokens(
        {"evidence": [{"difficulty": difficulty}]}
    ) == expected_tokens


def _runtime_payload() -> dict:
    track_map = {
        "version": "fixture-v1",
        "tracks": ["foundation", "sprint"],
        "entities": [
            {
                "entity": "series:1916889",
                "track": "foundation",
                "reviewer": "codex_sol_20260713",
                "needs_human": False,
                "evidence": {"source_line": 1},
            },
            {
                "entity": "bv:NEUTRAL",
                "track": "sprint",
                "reviewer": "",
                "needs_human": True,
                "evidence": {"source_line": 2},
            },
            {
                "entity": "bv:ORGANIC",
                "track": "foundation",
                "reviewer": "codex_sol_20260713",
                "needs_human": False,
                "evidence": {"source_line": 3},
            },
            {
                "entity": "bv:NONORGANIC",
                "track": "foundation",
                "reviewer": "codex_sol_20260713",
                "needs_human": False,
                "evidence": {"source_line": 4},
            },
        ],
    }

    def segment(bv: str, title: str, entity: str, *, anchor=None) -> dict:
        return {
            "segment_id": f"{bv}#P001",
            "bv": bv,
            "p": 1,
            "signed_entity": entity,
            "seg_type": "concept_intro",
            "difficulty": "T1",
            "topic_match_ratio": 1.0,
            "duration_sec": 300,
            "view": 100,
            "pubdate": "2026-01-01",
            "part_title": title,
            "video_title": title,
            "part_degrade_state": "real_title",
            "value": f"{title} value",
            "completion_criterion": f"{title} criterion",
            "time_anchor": anchor,
        }

    return {
        "version": "fixture-runtime-v1",
        "provenance": {"catalog_sha256": "a" * 64},
        "track_map": track_map,
        "segments_by_node": {
            NODE: [
                segment("SIGNED", "signed series", "series:1916889"),
                segment("NEUTRAL", "neutral candidate", "bv:NEUTRAL"),
            ],
            "有机推断": [
                segment(
                    "ORGANIC",
                    "organic segment",
                    "bv:ORGANIC",
                    anchor={"start_sec": 75.2, "end_sec": 155.0, "needs_human": False},
                )
            ],
            "氧化还原反应-概念": [
                segment(
                    "NONORGANIC",
                    "nonorganic video",
                    "bv:NONORGANIC",
                    anchor={"start_sec": 80.0, "end_sec": 160.0, "needs_human": False},
                )
            ],
        },
    }


def _curriculum_class():
    return importlib.import_module("core.learning.curriculum").CurriculumRuntime


def test_curriculum_excludes_neutral_candidates_instead_of_foundation_fallback() -> None:
    runtime = _curriculum_class().from_payload(_runtime_payload())

    result = runtime.recommend(
        node=NODE,
        belief=np.full(4, 0.25),
        grade="高二",
        learning_purpose="review",
        budget={"mode": "full", "rx_minutes": 20, "rx_segments": 3},
        seen_segments=set(),
        session_id="session-one",
        action_id="action-one",
    )

    assert [row["title"] for row in result["recommendations"]] == ["signed series"]
    assert all("bv" not in row and "p" not in row for row in result["recommendations"])
    assert {binding["bv"] for binding in result["bindings"].values()} == {"SIGNED"}
    assert result["rec_served"]["session_id"] == "session-one"
    assert result["rec_served"]["action_id"] == "action-one"


def test_default_runtime_omits_legacy_part_without_trusted_exact_topic_chunk() -> None:
    runtime = _curriculum_class().from_default_asset()
    candidates = runtime.eligible_segments(NODE)

    assert candidates == []


def test_default_runtime_returns_honest_no_segment_without_trusted_chunk() -> None:
    runtime = _curriculum_class().from_default_asset()

    result = runtime.recommend(
        node=NODE,
        belief=np.full(4, 0.25),
        grade="高二",
        learning_purpose="review",
        budget={"mode": "shallow", "rx_minutes": 8, "rx_segments": 1},
        seen_segments=set(),
        session_id="real-oxidation",
        action_id="learn",
    )

    assert result["recommendations"] == []
    assert result["status"] == "no_segment"


def test_default_runtime_keeps_all_exact_chunks_and_serves_by_segment_id() -> None:
    runtime = _curriculum_class().from_default_asset()
    node = "滴定-曲线与原理"
    candidates = runtime.eligible_segments(node)

    assert [row["segment_id"] for row in candidates] == [
        "BV1QG4y1X7e2#P111#c000",
        "BV1QG4y1X7e2#P111#c001",
        "BV1QG4y1X7e2#P111#c002",
    ]
    assert all(row["provenance"]["chunk_source"] == "video_chunks" for row in candidates)

    result = runtime.recommend(
        node=node,
        belief=np.full(4, 0.25),
        grade="高二",
        learning_purpose="review",
        budget={"mode": "full", "rx_minutes": 20, "rx_segments": 3},
        seen_segments=set(),
        session_id="chunk-lookup",
        action_id="learn",
    )

    recommendation = result["recommendations"][0]
    segment_id = "BV1QG4y1X7e2#P111#c000"
    assert recommendation["segment_id"] == segment_id
    assert recommendation["duration_seconds"] == 601
    assert result["bindings"][recommendation["rec_id"]]["segment_id"] == segment_id
    assert result["rec_served"]["served"][0]["segment_id"] == segment_id
    assert "t=3" in recommendation["url"]


def test_runtime_builder_drops_catalog_rows_without_authoritative_duration(tmp_path: Path) -> None:
    import yaml
    from scripts.build_curriculum_runtime import build_runtime

    track_map = tmp_path / "track.yaml"
    catalog = tmp_path / "catalog.jsonl"
    kg = tmp_path / "kg.jsonl"
    organic = tmp_path / "organic.jsonl"
    track_map.write_text(
        yaml.safe_dump(
            {
                "tracks": ["foundation"],
                "entities": [
                    {
                        "entity": "bv:MISSING",
                        "track": "foundation",
                        "reviewer": "codex_sol_20260713",
                        "needs_human": False,
                        "evidence": {"source_line": 1},
                    }
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    catalog.write_text(
        json.dumps(
            {
                "bv": "MISSING",
                "p": 1,
                "part_duration_sec": None,
                "part_title": "missing duration",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    kg.write_text(
        json.dumps(
            {
                "node_id": NODE,
                "recommended_videos": [{"bv": "MISSING", "p_number": 1}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    organic.write_text("", encoding="utf-8")

    payload = build_runtime(
        track_map_path=track_map,
        catalog_path=catalog,
        kg_path=kg,
        organic_path=organic,
    )

    assert payload["segments_by_node"] == {}


def test_runtime_builder_organic_alias_emits_all_trusted_nonorganic_chunks(
    tmp_path: Path,
) -> None:
    import inspect
    import yaml
    from scripts.build_curriculum_runtime import build_runtime

    track_map = tmp_path / "track.yaml"
    catalog = tmp_path / "catalog.jsonl"
    kg = tmp_path / "kg.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    track_map.write_text(
        yaml.safe_dump(
            {
                "tracks": ["foundation"],
                "entities": [{
                    "entity": "bv:SIGNED", "track": "foundation",
                    "reviewer": "codex_sol_20260713", "needs_human": False,
                    "evidence": {"source_line": 1},
                }],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    catalog.write_text(
        json.dumps({
            "bv": "SIGNED", "p": 1, "part_duration_sec": 900,
            "part_title": "非有机整讲", "video_title": "非有机合集",
            "season_name": "非有机合集", "season_order": 4,
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    kg.write_text(
        json.dumps({
            "node_id": "化学平衡-平衡移动",
            "recommended_videos": [{
                "bv": "SIGNED", "p_number": 1, "type": "concept_intro",
                "what_you_learn": "理解平衡移动",
                "completion_criterion": "能解释移动方向",
            }],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    chunk_rows = [
        {
            "chunk_id": "chunk-first", "bv": "SIGNED", "p_number": 1,
            "start_sec": 65, "end_sec": 155,
            "knowledge_topic": ["化学平衡-平衡移动"], "align_ratio": 0.95,
            "text_repaired_v2": "第一段", "needs_human": False,
        },
        {
            "chunk_id": "chunk-second", "bv": "SIGNED", "p_number": 1,
            "start_sec": 200, "end_sec": 320,
            "knowledge_topic": ["化学平衡-平衡移动"], "align_ratio": 0.90,
            "text_repaired_v2": "第二段", "needs_human": False,
        },
    ]
    chunks.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunk_rows),
        encoding="utf-8",
    )

    payload = build_runtime(
        track_map_path=track_map,
        catalog_path=catalog,
        kg_path=kg,
        organic_path=chunks,
    )
    rows = payload["segments_by_node"]["化学平衡-平衡移动"]

    assert [row["segment_id"] for row in rows] == ["chunk-first", "chunk-second"]
    assert [(row["start_sec"], row["end_sec"]) for row in rows] == [
        (65.0, 155.0), (200.0, 320.0),
    ]
    assert all(row["provenance"]["chunk_source"] == "video_chunks" for row in rows)
    assert payload["provenance"]["video_chunks"]["lines"] == 2

    assert "video_chunks_path" in inspect.signature(build_runtime).parameters
    alias_payload = build_runtime(
        track_map_path=track_map,
        catalog_path=catalog,
        kg_path=kg,
        video_chunks_path=chunks,
    )
    assert alias_payload["segments_by_node"] == payload["segments_by_node"]


def test_organic_anchor_requires_exact_topic_evidence() -> None:
    from scripts.build_curriculum_runtime import _select_anchor

    row = {
        "chunk_id": "chunk",
        "start_sec": 10,
        "end_sec": 20,
        "needs_human": False,
        "source_line": 1,
        "knowledge_topic": ["羧酸/酯"],
    }

    assert _select_anchor("有机推断", [row]) is None
    assert _select_anchor("羧酸/酯", [row])["chunk_id"] == "chunk"


def test_timestamp_links_require_organic_node_and_signed_nonhuman_anchor() -> None:
    runtime = _curriculum_class().from_payload(_runtime_payload())
    common = {
        "belief": np.full(4, 0.25),
        "grade": "高二",
        "learning_purpose": "review",
        "budget": {"mode": "full", "rx_minutes": 20, "rx_segments": 1},
        "seen_segments": set(),
    }

    organic = runtime.recommend(
        node="有机推断", session_id="organic", action_id="learn", **common
    )["recommendations"][0]
    nonorganic = runtime.recommend(
        node="氧化还原反应-概念",
        session_id="nonorganic",
        action_id="learn",
        **common,
    )["recommendations"][0]

    assert organic["url"].startswith("https://www.bilibili.com/video/")
    assert "t=75" in organic["url"]
    assert organic["has_time_anchor"] is True
    assert "t=" not in nonorganic["url"]
    assert nonorganic["has_time_anchor"] is False


def test_trusted_chunk_provenance_enables_nonorganic_timestamp_link() -> None:
    payload = _runtime_payload()
    segment = payload["segments_by_node"]["氧化还原反应-概念"][0]
    segment.update({
        "segment_id": "trusted-nonorganic-chunk",
        "start_sec": 80.0,
        "end_sec": 160.0,
        "duration_sec": 80,
        "provenance": {"chunk_source": "video_chunks"},
    })
    runtime = _curriculum_class().from_payload(payload)

    trusted = runtime.recommend(
        node="氧化还原反应-概念",
        belief=np.full(4, 0.25),
        grade="高二",
        learning_purpose="review",
        budget={"mode": "full", "rx_minutes": 20, "rx_segments": 1},
        seen_segments=set(),
        session_id="trusted-nonorganic",
        action_id="learn",
    )["recommendations"][0]

    assert "t=80" in trusted["url"]
    assert trusted["has_time_anchor"] is True
    assert trusted["segment_id"] == "trusted-nonorganic-chunk"
    assert "01:20-02:40" in trusted["reason"]


def test_unmarked_nonorganic_bounds_do_not_fabricate_range_in_reason() -> None:
    payload = _runtime_payload()
    segment = payload["segments_by_node"]["氧化还原反应-概念"][0]
    segment.update({"start_sec": 80.0, "end_sec": 160.0})
    runtime = _curriculum_class().from_payload(payload)

    legacy = runtime.recommend(
        node="氧化还原反应-概念",
        belief=np.full(4, 0.25),
        grade="高二",
        learning_purpose="review",
        budget={"mode": "full", "rx_minutes": 20, "rx_segments": 1},
        seen_segments=set(),
        session_id="unmarked-nonorganic",
        action_id="learn",
    )["recommendations"][0]

    assert "t=" not in legacy["url"]
    assert legacy["has_time_anchor"] is False
    assert "从开头观看本次分配时长" in legacy["reason"]
    assert "01:20-02:40" not in legacy["reason"]


def test_multilabel_chunk_reuse_requires_identical_physical_identity() -> None:
    payload = _runtime_payload()
    shared = payload["segments_by_node"]["有机推断"][0]
    shared.update({
        "segment_id": "shared-chunk",
        "start_sec": 75.2,
        "end_sec": 155.0,
        "duration_sec": 79,
        "provenance": {"chunk_source": "video_chunks"},
    })
    payload["segments_by_node"] = {
        "标签甲": [{**copy.deepcopy(shared), "value": "甲节点价值"}],
        "标签乙": [{
            **copy.deepcopy(shared),
            "value": "乙节点价值",
            "completion_criterion": "乙节点完成标准",
        }],
    }

    runtime = _curriculum_class().from_payload(payload)
    assert runtime.eligible_segments("标签甲")[0]["segment_id"] == "shared-chunk"
    assert runtime.eligible_segments("标签乙")[0]["value"] == "乙节点价值"

    conflict = copy.deepcopy(payload)
    conflict["segments_by_node"]["标签乙"][0]["end_sec"] = 156.0
    with pytest.raises(ValueError, match="physical identity conflict"):
        _curriculum_class().from_payload(conflict)


def test_rec_served_is_append_once_and_checkpoint_replay_is_stable() -> None:
    runtime = _curriculum_class().from_payload(_runtime_payload())
    store = MemoryStore()
    service = SessionService(_catalog(), store, curriculum=runtime)
    session_id = service.start_session("served-once", NODE, "30min")["session_id"]

    first = _drive_to_learning(service, session_id)
    second = service.next_assignment(session_id)

    assert second == first
    assert first["recommendations"]
    events = store.all_events("served-once")
    assert len([event for event in events if event["kind"] == "rec_served"]) == 1
    assert len([event for event in events if event["kind"] == "learning_checkpoint_ready"]) == 1
    rec_event = next(event for event in events if event["kind"] == "rec_served")
    assert rec_event["session_id"] == session_id
    assert rec_event["action_id"] == first["action_id"]
    assert rec_event["served"]


def test_checkpoint_crash_retry_reuses_explanation_and_rec_events() -> None:
    provider = CapturingExplanationProvider()
    runtime = _curriculum_class().from_payload(_runtime_payload())
    store = FailLearningCheckpointSaveStore()
    service = SessionService(
        _catalog(), store, curriculum=runtime, explanation_provider=provider
    )
    session_id = service.start_session("checkpoint-crash", NODE, "30min")["session_id"]
    store.fail_learning_save = True

    with pytest.raises(OSError, match="checkpoint save crash"):
        _drive_to_learning(service, session_id)
    recovered = service.next_assignment(session_id)

    assert recovered["phase"] == "learning"
    assert len(provider.calls) == 1
    events = store.all_events("checkpoint-crash")
    assert len([event for event in events if event["kind"] == "explanation_generated"]) == 1
    assert len([event for event in events if event["kind"] == "rec_served"]) == 1
    assert len([event for event in events if event["kind"] == "learning_checkpoint_ready"]) == 1


def test_watch_is_bound_to_opaque_rec_and_projects_seen_segments_across_sessions() -> None:
    runtime = _curriculum_class().from_payload(_runtime_payload())
    store = MemoryStore()
    service = SessionService(_catalog(), store, curriculum=runtime)
    first_session = service.start_session("watcher", NODE, "30min")["session_id"]
    checkpoint = _drive_to_learning(service, first_session)
    rec_id = checkpoint["recommendations"][0]["rec_id"]

    watched = service.record_watch(
        first_session,
        rec_id,
        "watch-action-1",
        watched_seconds=90,
        completed=False,
    )
    replay = service.record_watch(
        first_session,
        rec_id,
        "watch-action-1",
        watched_seconds=90,
        completed=False,
    )

    assert replay == watched
    assert watched == {
        "accepted": True,
        "rec_id": rec_id,
        "completed": False,
        "next_action": "continue_learning",
    }
    events = store.all_events("watcher")
    assert len([event for event in events if event["kind"] == "watch_proxy"]) == 1
    assert len([event for event in events if event["kind"] == "seen_segments"]) == 1
    projected = project_student("watcher", events).to_dict()
    assert projected["seen_segments"] == [{"bv": "SIGNED", "p": 1}]

    foreign_session = service.start_session("other", NODE, "30min")["session_id"]
    with pytest.raises(AssignmentError):
        service.record_watch(
            foreign_session,
            rec_id,
            "cross-session",
            watched_seconds=10,
            completed=False,
        )

    second_session = service.start_session("watcher", NODE, "30min")["session_id"]
    second_checkpoint = _drive_to_learning(service, second_session)
    assert second_checkpoint["recommendations"] == []


def test_watch_lost_append_race_reloads_and_rejects_changed_payload() -> None:
    runtime = _curriculum_class().from_payload(_runtime_payload())
    store = LosingWatchAppendStore()
    service = SessionService(_catalog(), store, curriculum=runtime)
    session_id = service.start_session("watch-race", NODE, "30min")["session_id"]
    checkpoint = _drive_to_learning(service, session_id)
    store.lose_next_watch = True

    with pytest.raises(AssignmentError, match="watch retry changed"):
        service.record_watch(
            session_id,
            checkpoint["recommendations"][0]["rec_id"],
            "racing-watch",
            watched_seconds=30,
            completed=False,
        )


def test_services_for_same_store_share_process_session_lock(tmp_path: Path) -> None:
    from adapters.store.local_json import LocalJsonStore

    memory = MemoryStore()
    first = SessionService(_catalog(), memory)
    second = SessionService(_catalog(), memory)
    assert first._session_lock("same") is second._session_lock("same")

    local_one = SessionService(_catalog(), LocalJsonStore(tmp_path))
    local_two = SessionService(_catalog(), LocalJsonStore(tmp_path))
    assert local_one._session_lock("same") is local_two._session_lock("same")


def _api_client() -> TestClient:
    app = create_app(
        catalog=_catalog(),
        store=MemoryStore(),
        curriculum=_curriculum_class().from_payload(_runtime_payload()),
        static_dir=None,
    )
    return TestClient(app)


def _api_drive_to_learning(client: TestClient, session_id: str, first: dict) -> dict:
    step = first
    for index in range(20):
        if step.get("phase") == "learning":
            return step
        submitted = client.post(
            f"/api/demo/sessions/{session_id}/submit",
            json={
                "assignment_id": step["assignment_id"],
                "submission_id": f"api-diag-{index}",
                "answer": "A",
            },
        )
        assert submitted.status_code == 200
        step = client.get(f"/api/demo/sessions/{session_id}/next").json()
    raise AssertionError("API never reached learning checkpoint")


def test_api_requires_learning_ack_and_watch_is_bound_to_issued_rec() -> None:
    client = _api_client()
    started = client.post(
        "/api/demo/sessions",
        json={"user_id": "api-student", "node": NODE, "budget_tier": "30min"},
    ).json()
    session_id = started["session_id"]
    checkpoint = _api_drive_to_learning(client, session_id, started["assignment"])
    rec_id = checkpoint["recommendations"][0]["rec_id"]

    replay = client.get(f"/api/demo/sessions/{session_id}/next")
    acknowledged = client.post(
        f"/api/demo/sessions/{session_id}/learning/ack",
        json={"action_id": checkpoint["action_id"]},
    )
    acknowledged_again = client.post(
        f"/api/demo/sessions/{session_id}/learning/ack",
        json={"action_id": checkpoint["action_id"]},
    )
    watched = client.post(
        f"/api/demo/sessions/{session_id}/watch",
        json={
            "rec_id": rec_id,
            "watch_id": "api-watch-1",
            "watched_seconds": 75,
            "completed": True,
        },
    )
    unknown = client.post(
        f"/api/demo/sessions/{session_id}/watch",
        json={
            "rec_id": "not-issued",
            "watch_id": "api-watch-2",
            "watched_seconds": 1,
            "completed": False,
        },
    )

    assert replay.json() == checkpoint
    assert acknowledged.status_code == 200
    assert acknowledged_again.json() == acknowledged.json()
    assert watched.status_code == 200
    assert watched.json()["next_action"] == "ack_learning"
    assert unknown.status_code == 404
    assert client.get(f"/api/demo/sessions/{session_id}/next").json()["phase"] in {
        "practice",
        "held_out",
    }


def test_canonical_api_rejects_hostile_origin_and_oversized_body() -> None:
    client = _api_client()

    hostile = client.get("/health", headers={"Origin": "https://attacker.example"})
    oversized = client.post(
        "/api/demo/sessions",
        content=b"x" * (512 * 1024 + 1),
        headers={"Content-Type": "application/json"},
    )

    assert hostile.status_code == 403
    assert hostile.json() == {"detail": "cross_origin_forbidden"}
    assert oversized.status_code == 413
    assert oversized.json() == {"detail": "request_too_large"}
    assert "access-control-allow-origin" not in hostile.headers


def _complete_session(service: SessionService, user_id: str, answer: str, prefix: str) -> str:
    session_id = service.start_session(user_id, NODE, "30min")["session_id"]
    for index in range(40):
        step = service.next_assignment(session_id)
        if step.get("phase") == "learning":
            service.ack_learning(session_id, step["action_id"])
            continue
        if step.get("done"):
            return session_id
        service.submit(
            session_id,
            step["assignment_id"],
            f"{prefix}-{index}",
            answer,
        )
    raise AssertionError("session did not complete")


def test_report_counts_only_session_evidence_and_has_distinct_seven_day_reminder() -> None:
    store = MemoryStore()
    service = SessionService(_catalog(), store, curriculum=FakeCurriculum())
    _complete_session(service, "returning-report", "A", "first")
    second_session = _complete_session(service, "returning-report", "A", "second")
    fresh_store = MemoryStore()
    fresh_service = SessionService(_catalog(), fresh_store, curriculum=FakeCurriculum())
    fresh_session = _complete_session(fresh_service, "fresh-report", "A", "fresh")

    report = service.report(second_session)
    fresh_report = fresh_service.report(fresh_session)
    second_events = [
        event
        for event in store.all_events("returning-report")
        if event.get("kind") == "answer_scored"
        and event.get("session_id") == second_session
        and event.get("node") == NODE
        and event.get("update_applied") is True
    ]
    all_answer_events = [
        event
        for event in store.all_events("returning-report")
        if event.get("kind") == "answer_scored" and event.get("node") == NODE
    ]

    assert report["evidence_count"] == len(second_events)
    assert report["evidence_count"] < len(all_answer_events)
    assert report["mastery_probability"] > fresh_report["mastery_probability"]
    assert isinstance(report["delta"], float)
    assert set(report["belief"]) == {
        "mastered",
        "prerequisite_gap",
        "concept_confusion",
        "uncertain",
    }
    assert sum(report["belief"].values()) == pytest.approx(1.0)
    assert report["belief"]["mastered"] == report["mastery_probability"]
    assert report["review_due_at"]
    assert report["seven_day_review_at"]
    assert report["seven_day_review_at"] != report["review_due_at"]
    as_of = datetime.fromisoformat(report["as_of"])
    reminder = datetime.fromisoformat(report["seven_day_review_at"])
    assert (reminder - as_of).total_seconds() == 7 * 24 * 60 * 60


def test_failed_held_out_report_has_rule_based_error_summary_and_reinforcement_plan() -> None:
    store = MemoryStore()
    service = SessionService(_catalog(), store, curriculum=FakeCurriculum())
    session_id = _complete_session(service, "failed-report", "B", "failed")

    report = service.report(session_id)

    assert report["outcome"] == "needs_reinforcement"
    assert report["error_summary"]["failed_held_out"] == 2
    assert report["error_summary"]["cause_codes"]
    assert report["reinforcement_plan"]["steps"]
    assert report["reinforcement_plan"]["next_check"] == "different_held_out_family"
    assert "疗效" not in str(report)
