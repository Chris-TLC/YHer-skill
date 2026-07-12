"""Contracts for the diagnosis-to-learning intervention checkpoint."""

from __future__ import annotations

import importlib
from datetime import datetime
import json
from pathlib import Path

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


def test_production_env_reuses_one_client_for_explanations_and_free_response_grading(
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

    assert len(FakeLLMClient.instances) == 1
    assert service.explanation_provider.client is FakeLLMClient.instances[0]
    assert service.llm_grader.client is FakeLLMClient.instances[0]

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
    assert "写出电子守恒" in FakeLLMClient.instances[0].calls[0][1]["content"]
    assert "只写了化合价变化" in FakeLLMClient.instances[0].calls[0][1]["content"]


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
    assert evidence["criteria"]
    assert evidence["result"] in {"correct", "incorrect", "deferred"}
    for required in ("数据代入", "变量", "因果链", "难度", "比喻", "必要"):
        assert required in prompt
    assert checkpoint["explanation"]["status"] == "generated"
    assert checkpoint["explanation"]["title"] == "电子守恒复盘"
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


def test_default_runtime_resolves_real_oxidation_candidate_through_signed_series() -> None:
    runtime = _curriculum_class().from_default_asset()
    candidates = runtime.eligible_segments(NODE)

    resolved = [
        row
        for row in candidates
        if row["bv"] == "BV1Qi4y1R7tW" and row["p"] == 19
    ]
    assert resolved
    assert resolved[0]["signed_entity"] == "series:1916889"
    assert "season:series:" not in str(resolved[0])


def test_default_runtime_routes_real_oxidation_video_within_eight_minute_rx_budget() -> None:
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

    assert result["recommendations"]
    recommendation = result["recommendations"][0]
    assert recommendation["url"].startswith("https://www.bilibili.com/video/")
    assert "t=" not in recommendation["url"]
    assert recommendation["has_time_anchor"] is False
    assert "bv" not in recommendation and "p" not in recommendation
    assert recommendation["duration_seconds"] == 8 * 60
    assert recommendation["full_video_duration_seconds"] > recommendation["duration_seconds"]
    assert recommendation["watch_scope"] == "from_start_within_session_budget"


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
