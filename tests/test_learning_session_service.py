"""End-to-end contracts for assignments, partitions and event projection."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from adapters.store.local_json import LocalJsonStore
from adapters.store.memory import MemoryStore
from core.learning.events import project_student
from core.learning.item_catalog import CatalogItem, ItemCatalog
from core.learning.session_service import AssignmentError, SessionError, SessionService
from core.learning import session_service as session_module


NODE = "氧化还原反应"


def _catalog() -> ItemCatalog:
    difficulties = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 0.2, 0.4, 0.6, 0.8, 0.3, 0.9)
    items = []
    for index, difficulty in enumerate(difficulties):
        items.append(
            CatalogItem(
                item_id=f"item-{index:02d}",
                family_id=f"family-{index:02d}",
                aligned_item_id=f"v3-{index:02d}",
                alignment_status="auto_inherited",
                node_ids=(NODE,),
                stem_blocks=({"para": [{"type": "text", "text": f"question {index}"}]},),
                stem_text=f"question {index}",
                stem_hash=f"hash-{index}",
                stem_normalized=f"question{index}",
                options={"A": "correct", "B": "wrong"},
                difficulty=difficulty,
                item_type="mcq",
                scoring_mode="mcq",
                answer_values=("A",),
                source_label="synthetic fixture",
            )
        )
    return ItemCatalog.from_items(items)


class Clock:
    def __init__(self, value: float = 1_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class FailingStudentSaveStore(MemoryStore):
    fail_next_student_save = False

    def save_student(self, user_id, model):
        if self.fail_next_student_save:
            self.fail_next_student_save = False
            raise OSError("simulated crash after event append")
        super().save_student(user_id, model)


class BlockingFirstProjectionStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.block_next_save = False
        self.first_save_entered = threading.Event()
        self.release_first_save = threading.Event()

    def save_student(self, user_id, model):
        if self.block_next_save:
            self.block_next_save = False
            self.first_save_entered.set()
            assert self.release_first_save.wait(timeout=5)
        super().save_student(user_id, model)


def _service(store=None, clock=None) -> SessionService:
    return SessionService(_catalog(), store or MemoryStore(), clock=clock or Clock())


def _private_item_id(store: MemoryStore, session_id: str, assignment_id: str) -> str:
    session = store.load_session(session_id)
    return session["assignments"][assignment_id]["item_id"]


def test_start_freezes_two_held_out_families_and_all_partitions_are_disjoint():
    store = MemoryStore()
    service = _service(store)
    started = service.start_session("student-1", NODE, "30min")
    session = store.load_session(started["session_id"])

    assert len(session["partitions"]["held_out"]["families"]) == 2
    for left, right in (("diagnostic", "practice"), ("diagnostic", "held_out"), ("practice", "held_out")):
        assert set(session["partitions"][left]["ids"]).isdisjoint(
            session["partitions"][right]["ids"]
        )
        assert set(session["partitions"][left]["families"]).isdisjoint(
            session["partitions"][right]["families"]
        )


def test_assignment_is_opaque_and_submission_cannot_name_an_item():
    store = MemoryStore()
    service = _service(store)
    session_id = service.start_session("student-1", NODE, "30min")["session_id"]
    assignment = service.next_assignment(session_id)

    assert len(assignment["assignment_id"]) == 32
    int(assignment["assignment_id"], 16)
    assert "item_id" not in assignment
    assert "answer" not in assignment["question"]
    with pytest.raises(AssignmentError):
        service.submit(session_id, "item-00", "submission-1", "A")


def test_correct_and_wrong_paths_change_belief_and_next_eig_choice():
    correct_store, wrong_store = MemoryStore(), MemoryStore()
    correct, wrong = _service(correct_store), _service(wrong_store)
    correct_sid = correct.start_session("correct", NODE, "30min")["session_id"]
    wrong_sid = wrong.start_session("wrong", NODE, "30min")["session_id"]
    first_correct = correct.next_assignment(correct_sid)
    first_wrong = wrong.next_assignment(wrong_sid)
    assert first_correct["question"]["difficulty"] == first_wrong["question"]["difficulty"]

    correct.submit(correct_sid, first_correct["assignment_id"], "c-1", "A")
    wrong.submit(wrong_sid, first_wrong["assignment_id"], "w-1", "B")
    next_correct = correct.next_assignment(correct_sid)
    next_wrong = wrong.next_assignment(wrong_sid)

    assert correct.get_profile("correct")["subjects"]["chemistry"]["kg_mastery"][NODE]["belief"] != (
        wrong.get_profile("wrong")["subjects"]["chemistry"]["kg_mastery"][NODE]["belief"]
    )
    assert next_correct["question"]["difficulty"] != next_wrong["question"]["difficulty"]


def test_append_once_projection_matches_persisted_student_and_continues_next_session():
    store = MemoryStore()
    service = _service(store)
    sid = service.start_session("returning", NODE, "30min")["session_id"]
    assignment = service.next_assignment(sid)
    service.submit(sid, assignment["assignment_id"], "one", "A")
    assignment_two = service.next_assignment(sid)
    service.submit(sid, assignment_two["assignment_id"], "two", "B")

    persisted = store.load_student("returning")
    replayed = project_student("returning", store.all_events("returning")).to_dict()
    assert persisted == replayed
    assert len(persisted["subjects"]["chemistry"]["kg_mastery"][NODE]["evidence"]) == 2

    before = deepcopy(persisted["subjects"]["chemistry"]["kg_mastery"][NODE]["belief"])
    service.start_session("returning", NODE, "1h")
    after = store.load_student("returning")["subjects"]["chemistry"]["kg_mastery"][NODE]["belief"]
    assert after == before


def test_pause_resume_and_budget_exhaustion_preserve_server_checkpoint():
    store, clock = MemoryStore(), Clock()
    service = _service(store, clock)
    sid = service.start_session("paused", NODE, "30min")["session_id"]
    assignment = service.next_assignment(sid)
    service.pause_session(sid)
    with pytest.raises(AssignmentError):
        service.next_assignment(sid)
    clock.value += 60 * 60
    resumed = service.resume_session(sid)
    assert resumed["phase"] == "diagnostic"
    assert service.next_assignment(sid)["assignment_id"] == assignment["assignment_id"]

    service.submit(sid, assignment["assignment_id"], "paused-1", "A")
    clock.value += 19 * 60
    exhausted = service.next_assignment(sid)
    assert exhausted["budget_exhausted"] is True
    assert store.load_session(sid)["checkpoint"]["resume_phase"] in {"practice", "held_out", "complete"}


def test_held_out_replay_is_idempotent():
    store = MemoryStore()
    service = _service(store)
    sid = service.start_session("held-out", NODE, "30min")["session_id"]

    # Drive the finite state machine; no test-only production hook is used.
    assignment = service.next_assignment(sid)
    for index in range(30):
        if assignment.get("phase") == "held_out":
            break
        service.submit(sid, assignment["assignment_id"], f"drive-{index}", "A")
        assignment = service.next_assignment(sid)
    assert assignment["phase"] == "held_out"

    first = service.submit(sid, assignment["assignment_id"], "held-1", "A")
    event_count = len(store.all_events("held-out"))
    replay = service.submit(sid, assignment["assignment_id"], "held-1", "A")
    assert replay == first
    assert len(store.all_events("held-out")) == event_count
    with pytest.raises(AssignmentError):
        service.submit(sid, assignment["assignment_id"], "held-2", "B")


def test_returning_strong_student_still_gets_two_direct_questions_this_session():
    store = MemoryStore()
    service = _service(store)
    first_sid = service.start_session("strong-returner", NODE, "30min")["session_id"]
    assignment = service.next_assignment(first_sid)
    for index in range(8):
        if assignment.get("phase") != "diagnostic":
            break
        service.submit(first_sid, assignment["assignment_id"], f"first-{index}", "A")
        assignment = service.next_assignment(first_sid)

    second_sid = service.start_session("strong-returner", NODE, "30min")["session_id"]
    one = service.next_assignment(second_sid)
    service.submit(second_sid, one["assignment_id"], "second-1", "A")
    two = service.next_assignment(second_sid)

    assert one["phase"] == "diagnostic"
    assert two["phase"] == "diagnostic"


def test_retry_recovers_when_event_append_won_but_projection_save_crashed():
    store = FailingStudentSaveStore()
    service = _service(store)
    sid = service.start_session("crash-retry", NODE, "30min")["session_id"]
    assignment = service.next_assignment(sid)
    store.fail_next_student_save = True

    with pytest.raises(OSError, match="simulated crash"):
        service.submit(sid, assignment["assignment_id"], "stable-submission", "A")
    answer_events = [row for row in store.all_events("crash-retry") if row["kind"] == "answer_scored"]
    assert len(answer_events) == 1

    recovered = service.submit(sid, assignment["assignment_id"], "stable-submission", "A")
    answer_events = [row for row in store.all_events("crash-retry") if row["kind"] == "answer_scored"]
    assert recovered["accepted"] is True
    assert len(answer_events) == 1
    assert store.load_session(sid)["assignments"][assignment["assignment_id"]]["submitted"] is True


def test_corrupt_partition_overlap_is_rejected_on_load():
    store = MemoryStore()
    service = _service(store)
    sid = service.start_session("corrupt", NODE, "30min")["session_id"]
    session = store.load_session(sid)
    session["partitions"]["held_out"]["ids"].append(
        session["partitions"]["diagnostic"]["ids"][0]
    )
    session["partitions"]["held_out"]["families"].append(
        session["partitions"]["diagnostic"]["families"][0]
    )
    session["partitions"]["held_out"]["roles"][
        session["partitions"]["diagnostic"]["ids"][0]
    ] = "local"
    store.save_session(sid, session)

    with pytest.raises(SessionError, match="partition overlap"):
        service.next_assignment(sid)


def test_actual_catalog_never_places_media_only_questions_in_session_partitions():
    catalog = ItemCatalog.from_default_data()
    store = MemoryStore()
    service = SessionService(catalog, store)
    sid = service.start_session("actual-media-gate", NODE, "30min")["session_id"]
    session = store.load_session(sid)
    partition_ids = {
        item_id
        for partition in session["partitions"].values()
        for item_id in partition["ids"]
    }

    assert partition_ids
    assert all(catalog.items[item_id].has_media is False for item_id in partition_ids)
    practice_ids = session["partitions"]["practice"]["ids"]
    assert practice_ids
    assert catalog.items[practice_ids[0]].scoring_mode == "free_llm"


def test_done_response_embeds_the_same_safe_report():
    store = MemoryStore()
    service = _service(store)
    sid = service.start_session("done-report", NODE, "30min")["session_id"]
    for index in range(30):
        next_step = service.next_assignment(sid)
        if next_step.get("done"):
            break
        service.submit(sid, next_step["assignment_id"], f"done-{index}", "A")
    else:
        raise AssertionError("finite session did not complete")

    assert next_step["report"] == service.report(sid)
    assert next_step["phase"] == "complete"
    assert next_step["report"]["outcome"] == "verified"
    assert next_step["report"]["state_label"] not in {"M", "P", "C", "U"}


def test_local_store_append_once_is_safe_across_preinitialized_writers(tmp_path):
    first = LocalJsonStore(tmp_path)
    second = LocalJsonStore(tmp_path)
    event = {"event_id": "same-event", "kind": "answer_scored"}

    assert first.append_event_once("student", event) is True
    assert second.append_event_once("student", event) is False
    assert first.all_events("student") == [event]


def test_two_services_cannot_accept_different_submissions_for_one_assignment(monkeypatch):
    store = MemoryStore()
    first = _service(store)
    second = _service(store)
    sid = first.start_session("concurrent", NODE, "30min")["session_id"]
    assignment = first.next_assignment(sid)
    barrier = threading.Barrier(2)
    original_score = session_module.score_item

    def synchronized_score(*args, **kwargs):
        barrier.wait(timeout=5)
        return original_score(*args, **kwargs)

    monkeypatch.setattr(session_module, "score_item", synchronized_score)

    def submit(service, submission_id, answer):
        try:
            service.submit(sid, assignment["assignment_id"], submission_id, answer)
            return "accepted"
        except AssignmentError as exc:
            return f"rejected:{exc.status_code}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                lambda args: submit(*args),
                ((first, "concurrent-a", "A"), (second, "concurrent-b", "B")),
            )
        )

    answer_events = [event for event in store.all_events("concurrent") if event["kind"] == "answer_scored"]
    profile = store.load_student("concurrent")
    session = store.load_session(sid)
    assert sorted(outcomes) == ["accepted", "rejected:409"]
    assert len(answer_events) == 1
    assert len(profile["subjects"]["chemistry"]["kg_mastery"][NODE]["evidence"]) == 1
    assert session["asked"]["diagnostic"] == 1


def test_pause_cannot_be_overwritten_by_an_inflight_submit(monkeypatch):
    store = MemoryStore()
    service = _service(store)
    sid = service.start_session("pause-race", NODE, "30min")["session_id"]
    assignment = service.next_assignment(sid)
    score_entered = threading.Event()
    release_score = threading.Event()
    original_score = session_module.score_item

    def blocked_score(*args, **kwargs):
        score_entered.set()
        assert release_score.wait(timeout=5)
        return original_score(*args, **kwargs)

    monkeypatch.setattr(session_module, "score_item", blocked_score)
    with ThreadPoolExecutor(max_workers=2) as executor:
        submitted = executor.submit(
            service.submit,
            sid,
            assignment["assignment_id"],
            "pause-race-submit",
            "A",
        )
        assert score_entered.wait(timeout=5)
        paused = executor.submit(service.pause_session, sid)
        assert paused.done() is False
        release_score.set()
        submitted.result(timeout=5)
        paused.result(timeout=5)

    session = store.load_session(sid)
    kinds = [event["kind"] for event in store.all_events("pause-race")]
    assert session["phase"] == "paused"
    assert session["status"] == "paused"
    assert kinds.count("answer_scored") == 1
    assert kinds.count("session_paused") == 1


def test_actual_no_key_free_practice_is_deferred_without_profile_update():
    catalog = ItemCatalog.from_default_data()
    store = MemoryStore()
    service = SessionService(catalog, store)
    sid = service.start_session("offline-free", NODE, "30min")["session_id"]
    assignment = service.next_assignment(sid)
    for index in range(20):
        if assignment.get("phase") == "practice":
            break
        service.submit(sid, assignment["assignment_id"], f"offline-diag-{index}", "A")
        assignment = service.next_assignment(sid)
    assert assignment["phase"] == "practice"
    before = len(
        store.load_student("offline-free")["subjects"]["chemistry"]["kg_mastery"][NODE][
            "evidence"
        ]
    )

    result = service.submit(sid, assignment["assignment_id"], "offline-practice", "work")
    after = len(
        store.load_student("offline-free")["subjects"]["chemistry"]["kg_mastery"][NODE][
            "evidence"
        ]
    )
    assert result["status"] == "deferred"
    assert result["degraded"] is True
    assert result["is_correct"] is None
    assert after == before


def test_completed_wrong_held_out_is_needs_reinforcement():
    store = MemoryStore()
    service = _service(store)
    sid = service.start_session("needs-more", NODE, "30min")["session_id"]
    for index in range(30):
        next_step = service.next_assignment(sid)
        if next_step.get("done"):
            break
        service.submit(sid, next_step["assignment_id"], f"wrong-{index}", "B")
    assert next_step["report"]["outcome"] == "needs_reinforcement"


def test_resumed_budget_exhaustion_remains_partial_after_completion():
    store, clock = MemoryStore(), Clock()
    service = _service(store, clock)
    sid = service.start_session("partial", NODE, "30min")["session_id"]
    assignment = service.next_assignment(sid)
    service.submit(sid, assignment["assignment_id"], "partial-first", "A")
    clock.value += 19 * 60
    assert service.next_assignment(sid)["budget_exhausted"] is True
    service.resume_session(sid)
    for index in range(10):
        next_step = service.next_assignment(sid)
        if next_step.get("done"):
            break
        service.submit(sid, next_step["assignment_id"], f"partial-{index}", "A")
    assert next_step["report"]["outcome"] == "partial"


def test_same_user_cross_session_projection_cannot_be_overwritten_by_stale_save():
    store = BlockingFirstProjectionStore()
    service = _service(store)
    first_sid = service.start_session("same-user", NODE, "30min")["session_id"]
    second_sid = service.start_session("same-user", NODE, "30min")["session_id"]
    first_assignment = service.next_assignment(first_sid)
    second_assignment = service.next_assignment(second_sid)
    store.block_next_save = True

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.submit,
            first_sid,
            first_assignment["assignment_id"],
            "cross-session-1",
            "A",
        )
        assert store.first_save_entered.wait(timeout=5)
        second = executor.submit(
            service.submit,
            second_sid,
            second_assignment["assignment_id"],
            "cross-session-2",
            "B",
        )
        threading.Event().wait(0.05)
        store.release_first_save.set()
        first.result(timeout=5)
        second.result(timeout=5)

    events = store.all_events("same-user")
    replayed = project_student("same-user", events).to_dict()
    persisted = store.load_student("same-user")
    assert len([event for event in events if event["kind"] == "answer_scored"]) == 2
    assert persisted == replayed
    assert len(persisted["subjects"]["chemistry"]["kg_mastery"][NODE]["evidence"]) == 2

    store.save_student("same-user", {"user_id": "same-user", "subjects": {}})
    assert service.get_profile("same-user") == replayed


def test_actual_oxidation_diagnosis_uses_aggregated_prerequisite_eig_candidates():
    catalog = ItemCatalog.from_default_data()
    store = MemoryStore()
    service = SessionService(catalog, store)
    sid = service.start_session("prereq-eig", NODE, "30min")["session_id"]
    session = store.load_session(sid)
    roles = session["partitions"]["diagnostic"]["roles"]
    assert "prereq" in roles.values()
    prereq_ids = {item_id for item_id, role in roles.items() if role == "prereq"}
    assert prereq_ids
    assert all(
        set(catalog.items[item_id].node_ids) & set(catalog.prerequisites_for(NODE))
        for item_id in prereq_ids
    )

    local_answers = 0
    for index in range(8):
        assignment = service.next_assignment(sid)
        if assignment["role"] == "prereq":
            break
        local_answers += 1
        service.submit(sid, assignment["assignment_id"], f"prereq-local-{index}", "Z")
    assert assignment["role"] == "prereq"
    service.submit(sid, assignment["assignment_id"], "prereq-answer", "Z")
    record = store.load_student("prereq-eig")["subjects"]["chemistry"]["kg_mastery"][NODE]
    assert record["direct_answers"] == local_answers


def test_total_budget_can_pause_after_practice_before_held_out():
    store, clock = MemoryStore(), Clock()
    service = _service(store, clock)
    sid = service.start_session("total-budget", NODE, "30min")["session_id"]
    for index in range(20):
        assignment = service.next_assignment(sid)
        if assignment.get("phase") == "practice":
            break
        service.submit(sid, assignment["assignment_id"], f"budget-diag-{index}", "A")
    assert assignment["phase"] == "practice"
    service.submit(sid, assignment["assignment_id"], "budget-practice", "A")
    clock.value += 31 * 60

    exhausted = service.next_assignment(sid)

    assert exhausted["budget_exhausted"] is True
    assert exhausted["phase"] == "paused"
    assert store.load_session(sid)["checkpoint"]["resume_phase"] == "held_out"


def test_held_out_events_apply_fsrs_stability_updates():
    correct = [0.9, 0.1, 0.1, 0.1]
    wrong = [0.1, 0.9, 0.9, 0.9]
    events = [
        {"event_id": "start", "kind": "session_started", "occurred_at": 1.0},
        *[
            {
                "event_id": f"diag-{index}",
                "kind": "answer_scored",
                "occurred_at": float(index + 2),
                "node": NODE,
                "phase": "diagnostic",
                "correct": True,
                "confidence": 1.0,
                "likelihood": correct,
                "update_applied": True,
                "is_direct": True,
            }
            for index in range(2)
        ],
        {
            "event_id": "held-pass",
            "kind": "answer_scored",
            "occurred_at": 4.0,
            "node": NODE,
            "phase": "held_out",
            "correct": True,
            "confidence": 1.0,
            "likelihood": correct,
            "update_applied": True,
            "is_direct": True,
        },
    ]
    passed = project_student("fsrs", events).to_dict()
    assert passed["subjects"]["chemistry"]["kg_mastery"][NODE]["stability"] == 9.0

    events.append(
        {
            "event_id": "held-fail",
            "kind": "answer_scored",
            "occurred_at": 5.0,
            "node": NODE,
            "phase": "held_out",
            "correct": False,
            "confidence": 1.0,
            "likelihood": wrong,
            "update_applied": True,
            "is_direct": True,
        }
    )
    failed = project_student("fsrs", events).to_dict()
    assert failed["subjects"]["chemistry"]["kg_mastery"][NODE]["stability"] == 4.5


def test_completed_report_is_frozen_at_that_sessions_last_event():
    store = MemoryStore()
    service = _service(store)

    def complete(user_id, answer, prefix):
        sid = service.start_session(user_id, NODE, "30min")["session_id"]
        for index in range(30):
            step = service.next_assignment(sid)
            if step.get("done"):
                return sid
            service.submit(sid, step["assignment_id"], f"{prefix}-{index}", answer)
        raise AssertionError("session did not complete")

    first_sid = complete("frozen-report", "A", "first")
    before = service.report(first_sid)
    complete("frozen-report", "B", "second")
    after = service.report(first_sid)

    assert after == before
