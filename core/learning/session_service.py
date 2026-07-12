"""Authoritative session state machine for diagnosis, practice and held-out work."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from core.tutor.profile_model import StudentModel
from engine import mastery, planner, selector

from .events import all_events, append_once, project_student
from .item_catalog import CatalogItem, ItemCatalog
from .scoring import LLMGrader, score_item


_TOTAL_MINUTES = {"30min": 30, "1h": 60, "2h": 120, "3h+": 180}
_PHASE_ORDER = ("diagnostic", "practice", "held_out")
_FORBIDDEN_PUBLIC_KEYS = {
    "item_id",
    "content_family_id",
    "family_id",
    "standard_answer",
    "final_answers",
    "solution_answers",
    "answer",
    "rubric",
    "analysis",
    "correct_option",
}


class SessionError(RuntimeError):
    status_code = 400


class SessionNotFound(SessionError):
    status_code = 404


class AssignmentError(SessionError):
    def __init__(self, message: str, *, status_code: int = 404):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class _Selection:
    item: CatalogItem
    eig: float | None


class SessionService:
    def __init__(
        self,
        catalog: ItemCatalog,
        store,
        *,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] | None = None,
        llm_grader: LLMGrader | None = None,
        synthetic: bool = False,
    ):
        self.catalog = catalog
        self.store = store
        self.clock = clock
        self.id_factory = id_factory or (lambda: secrets.token_hex(16))
        self.llm_grader = llm_grader
        self.synthetic = bool(synthetic)
        self._session_locks: dict[str, threading.RLock] = {}
        self._session_locks_guard = threading.Lock()
        self._user_locks: dict[str, threading.RLock] = {}
        self._user_locks_guard = threading.Lock()

    def start_session(
        self,
        user_id: str,
        node: str,
        budget_tier: str,
        *,
        grade: str = "高二",
        learning_purpose: str = "review",
    ) -> dict[str, Any]:
        user_id, node = str(user_id).strip(), str(node).strip()
        if not user_id or not node:
            raise SessionError("user_id and node are required")
        if budget_tier not in planner.BUDGET_TABLE:
            raise SessionError("unsupported budget_tier")
        if node not in self.catalog.open_nodes():
            raise SessionError("node is not open for deterministic diagnosis")
        candidates = _one_per_family(self.catalog.for_node(node, deterministic_only=True))
        if len(candidates) < 5:
            raise SessionError("node needs at least five independent deterministic families")

        # Freeze held-out before any adaptive selection. Stable ordering makes
        # replay/audit deterministic while item IDs remain server-private.
        ordered = sorted(candidates, key=lambda item: (item.family_id, item.item_id))
        held_out = ordered[-2:]
        remaining = ordered[:-2]
        held_out_families = {item.family_id for item in held_out}
        free_practice = sorted(
            _one_per_family(
                item
                for item in self.catalog.for_node(node)
                if item.scoring_mode == "free_llm"
                and not item.has_media
                and item.family_id not in held_out_families
            ),
            key=lambda item: (item.family_id, item.item_id),
        )
        practice: list[CatalogItem] = []
        diagnostic = remaining
        for candidate in free_practice:
            without_family = [item for item in remaining if item.family_id != candidate.family_id]
            if len(without_family) >= 3:
                practice, diagnostic = [candidate], without_family
                break
        if not practice and len(remaining) > 3:
            practice, diagnostic = remaining[-1:], remaining[:-1]
        diagnostic_roles = {item.item_id: "local" for item in diagnostic}
        used_families = {item.family_id for item in (*diagnostic, *practice, *held_out)}
        for prerequisite in self.catalog.prerequisites_for(node):
            added = 0
            for item in _one_per_family(
                self.catalog.for_node(prerequisite, deterministic_only=True)
            ):
                if item.family_id in used_families:
                    continue
                diagnostic.append(item)
                diagnostic_roles[item.item_id] = "prereq"
                used_families.add(item.family_id)
                added += 1
                if added >= 3:
                    break
        now = float(self.clock())
        session_id = self._new_id()
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "node": node,
            "budget_tier": budget_tier,
            "budget": planner.session_budget(budget_tier),
            "grade": str(grade or "高二"),
            "learning_purpose": str(learning_purpose or "review"),
            "synthetic": self.synthetic,
            "phase": "diagnostic",
            "status": "active",
            "created_at": now,
            "phase_started_at": now,
            "paused_at": None,
            "paused_seconds": 0.0,
            "budget_exhausted": False,
            "completed_at": None,
            "partitions": {
                "diagnostic": _partition(diagnostic, diagnostic_roles),
                "practice": _partition(practice),
                "held_out": _partition(held_out),
            },
            "assignments": {},
            "submission_ids": {},
            "current_assignment_id": None,
            "asked": {phase: 0 for phase in _PHASE_ORDER},
            "direct_diagnostic_answers": 0,
            "checkpoint": {"resume_phase": "diagnostic"},
        }
        self._persist_session(session)
        event = {
            "event_id": self._new_id(),
            "kind": "session_started",
            "occurred_at": now,
            "session_id": session_id,
            "user_id": user_id,
            "node": node,
            "budget_tier": budget_tier,
            "grade": session["grade"],
            "learning_purpose": session["learning_purpose"],
            "synthetic": self.synthetic,
        }
        append_once(self.store, user_id, event)
        self._save_projection(user_id)
        return self._session_view(session)

    def next_assignment(self, session_id: str) -> dict[str, Any]:
        with self._session_lock(session_id):
            return self._next_assignment_locked(session_id)

    def _next_assignment_locked(self, session_id: str) -> dict[str, Any]:
        session = self._load_session(session_id)
        if session["phase"] == "paused":
            raise AssignmentError("session is paused", status_code=409)
        if session["phase"] == "complete":
            self._ensure_completion_event(session)
            return self._done_view(session)
        current_id = session.get("current_assignment_id")
        if current_id:
            current = session["assignments"][current_id]
            if not current.get("submitted"):
                return current["view"]
        if session["phase"] == "held_out" and not self._remaining_items(session, "held_out"):
            self._complete_session(session)
            return self._done_view(session)

        if session["phase"] == "diagnostic" and self._diagnostic_budget_exhausted(session):
            resume_phase = self._phase_after_diagnostic(session)
            return self._pause_for_budget(session, resume_phase)
        if not session["budget_exhausted"] and self._total_budget_exhausted(session):
            resume_phase = (
                session["phase"]
                if self._remaining_items(session, session["phase"])
                else self._next_phase(session)
            )
            return self._pause_for_budget(session, resume_phase)

        selection = self._select_for_current_phase(session)
        while selection is None:
            session["phase"] = self._next_phase(session)
            if session["phase"] == "complete":
                self._complete_session(session)
                return self._done_view(session)
            selection = self._select_for_current_phase(session)

        assignment_id = self._new_id()
        role = session["partitions"][session["phase"]]["roles"][selection.item.item_id]
        view = {
            "assignment_id": assignment_id,
            "phase": session["phase"],
            "role": role,
            "question": selection.item.public_question(),
            "progress": self._progress(session),
            "timing": self._timing(session),
            "synthetic": session["synthetic"],
            "budget_exhausted": False,
            "next_action": "submit",
        }
        _assert_public_safe(view)
        session["assignments"][assignment_id] = {
            "assignment_id": assignment_id,
            "item_id": selection.item.item_id,
            "family_id": selection.item.family_id,
            "phase": session["phase"],
            "role": role,
            "selector_eig": selection.eig,
            "created_at": float(self.clock()),
            "submitted": False,
            "view": view,
        }
        session["current_assignment_id"] = assignment_id
        self._persist_session(session)
        return view

    def submit(
        self,
        session_id: str,
        assignment_id: str,
        submission_id: str,
        answer: str,
    ) -> dict[str, Any]:
        with self._session_lock(session_id):
            return self._submit_locked(session_id, assignment_id, submission_id, answer)

    def _submit_locked(
        self,
        session_id: str,
        assignment_id: str,
        submission_id: str,
        answer: str,
    ) -> dict[str, Any]:
        session = self._load_session(session_id)
        assignment = session["assignments"].get(str(assignment_id))
        if assignment is None:
            raise AssignmentError("assignment not found in this session")
        prior_assignment = session["submission_ids"].get(str(submission_id))
        if prior_assignment and prior_assignment != assignment_id:
            raise AssignmentError("submission_id already used", status_code=409)
        if assignment.get("submitted"):
            if (
                assignment.get("submission_id") == submission_id
                and assignment.get("submission_sha256") == _submission_hash(answer)
            ):
                return assignment["result"]
            raise AssignmentError("assignment already submitted", status_code=409)
        if session["phase"] == "paused":
            raise AssignmentError("session is paused", status_code=409)
        if assignment["phase"] != session["phase"]:
            raise AssignmentError("assignment phase does not match session phase", status_code=409)

        existing = self._answer_event(session["user_id"], assignment_id)
        if existing is not None:
            return self._recover_submission(session, assignment, submission_id, answer, existing)

        item = self.catalog.items[assignment["item_id"]]
        scored = score_item(item, str(answer), self.llm_grader)
        likelihood = _likelihood_for_assignment(scored, item, assignment["role"])
        now = float(self.clock())
        event = {
            "event_id": _answer_event_id(session_id, assignment_id),
            "kind": "answer_scored",
            "occurred_at": now,
            "session_id": session_id,
            "user_id": session["user_id"],
            "submission_id": str(submission_id),
            "submission_sha256": _submission_hash(answer),
            "assignment_id": str(assignment_id),
            "item_id": item.item_id,
            "family_id": item.family_id,
            "node": session["node"],
            "phase": assignment["phase"],
            "correct": scored.correct,
            "status": scored.status,
            "error_code": scored.error_code,
            "confidence": scored.confidence,
            "likelihood": list(likelihood),
            "update_applied": scored.update_allowed,
            "is_direct": assignment["role"] != "prereq",
            "synthetic": session["synthetic"],
        }
        appended = append_once(self.store, session["user_id"], event)
        if not appended:
            existing = self._answer_event(session["user_id"], assignment_id)
            if existing is None:
                raise AssignmentError("event id collision", status_code=409)
            event = existing
        return self._recover_submission(session, assignment, submission_id, answer, event)

    def pause_session(self, session_id: str) -> dict[str, Any]:
        with self._session_lock(session_id):
            return self._pause_session_locked(session_id)

    def _pause_session_locked(self, session_id: str) -> dict[str, Any]:
        session = self._load_session(session_id)
        if session["phase"] in {"paused", "complete"}:
            return self._session_view(session)
        now = float(self.clock())
        session["checkpoint"]["resume_phase"] = session["phase"]
        session["phase"] = "paused"
        session["status"] = "paused"
        session["paused_at"] = now
        self._persist_session(session)
        self._append_control_event(session, "session_paused", now)
        return self._session_view(session)

    def resume_session(self, session_id: str) -> dict[str, Any]:
        with self._session_lock(session_id):
            return self._resume_session_locked(session_id)

    def _resume_session_locked(self, session_id: str) -> dict[str, Any]:
        session = self._load_session(session_id)
        if session["phase"] != "paused":
            if session["phase"] == "complete":
                return self._done_view(session)
            return self._session_view(session)
        now = float(self.clock())
        paused_at = float(session.get("paused_at") or now)
        session["paused_seconds"] += max(0.0, now - paused_at)
        session["phase"] = session["checkpoint"].get("resume_phase") or "diagnostic"
        session["status"] = "active"
        session["paused_at"] = None
        self._persist_session(session)
        self._append_control_event(session, "session_resumed", now)
        return self._session_view(session)

    def get_profile(self, user_id: str) -> dict[str, Any]:
        return self._save_projection(user_id)

    def session_view(self, session_id: str) -> dict[str, Any]:
        return self._session_view(self._load_session(session_id))

    def report(self, session_id: str) -> dict[str, Any]:
        session = self._load_session(session_id)
        if session["phase"] != "complete":
            raise AssignmentError("report not available until complete", status_code=409)
        self._ensure_completion_event(session)
        user_id, node = session["user_id"], session["node"]
        all_user_events = all_events(self.store, user_id)
        completion_index = next(
            (
                index
                for index in range(len(all_user_events) - 1, -1, -1)
                if all_user_events[index].get("kind") == "session_completed"
                and all_user_events[index].get("session_id") == session_id
            ),
            None,
        )
        if completion_index is None:
            completion_index = max(
                (
                    index
                    for index, event in enumerate(all_user_events)
                    if event.get("session_id") == session_id
                ),
                default=-1,
            )
        events = all_user_events[: completion_index + 1]
        held_out_events = [
            event
            for event in events
            if event.get("kind") == "answer_scored"
            and event.get("session_id") == session_id
            and event.get("phase") == "held_out"
        ]
        if session.get("budget_exhausted"):
            outcome = "partial"
        elif (
            len(held_out_events) == len(session["partitions"]["held_out"]["ids"])
            and held_out_events
            and all(event.get("correct") is True for event in held_out_events)
        ):
            outcome = "verified"
        else:
            outcome = "needs_reinforcement"
        model = project_student(user_id, events)
        record = model.subject("chemistry").kg_mastery.get(node)
        now = (
            float(events[-1].get("occurred_at") or session.get("completed_at") or self.clock())
            if events
            else float(session.get("completed_at") or self.clock())
        )
        as_of = datetime.fromtimestamp(now, timezone.utc).isoformat()
        if record is None:
            summary = {
                "node": node,
                "outcome": outcome,
                "state_label": "当前证据不足",
                "mastery_probability": None,
                "delta": None,
                "evidence_count": 0,
                "as_of": as_of,
                "review_due_at": None,
            }
        else:
            state = mastery.NodeBelief(
                b=np.asarray(record.belief, dtype=float),
                S=record.stability,
                last_review_at=record.last_review_at,
                direct_answers=record.direct_answers,
            )
            projected = mastery.get_belief(state, now)
            session_answers = [
                event
                for event in events
                if event.get("kind") == "answer_scored"
                and event.get("session_id") == session_id
                and event.get("node") == node
                and event.get("update_applied") is True
            ]
            delta = None
            if session_answers:
                without_session = [event for event in events if event not in session_answers]
                baseline_model = project_student(user_id, without_session)
                baseline_record = baseline_model.subject("chemistry").kg_mastery.get(node)
                baseline = 0.25
                if baseline_record is not None:
                    baseline_state = mastery.NodeBelief(
                        b=np.asarray(baseline_record.belief, dtype=float),
                        S=baseline_record.stability,
                        last_review_at=baseline_record.last_review_at,
                        direct_answers=baseline_record.direct_answers,
                    )
                    baseline = float(mastery.get_belief(baseline_state, now)[mastery.M])
                delta = round(float(projected[mastery.M]) - baseline, 6)
            summary = {
                "node": node,
                "outcome": outcome,
                "state_label": _state_label(int(np.argmax(projected))),
                "mastery_probability": round(float(projected[mastery.M]), 6),
                "delta": delta,
                "evidence_count": len(record.evidence),
                "as_of": as_of,
                "review_due_at": _review_due_at(record.stability, record.last_review_at),
            }
        return {
            "session_id": session_id,
            "phase": session["phase"],
            "status": session["status"],
            **summary,
            "timing": self._timing(session),
            "synthetic": session["synthetic"],
        }

    def _recover_submission(
        self,
        session: dict[str, Any],
        assignment: dict[str, Any],
        submission_id: str,
        answer: str,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        if str(event.get("submission_id")) != str(submission_id):
            raise AssignmentError("assignment already has a different submission", status_code=409)
        if str(event.get("submission_sha256")) != _submission_hash(answer):
            raise AssignmentError("submission retry changed the answer", status_code=409)
        self._save_projection(session["user_id"])
        session["asked"][assignment["phase"]] += 1
        if assignment["phase"] == "diagnostic" and assignment["role"] != "prereq":
            session["direct_diagnostic_answers"] += 1
        result = self._public_submission_result(session, assignment, event)
        assignment.update(
            {
                "submitted": True,
                "submission_id": str(submission_id),
                "submission_sha256": _submission_hash(answer),
                "result": result,
            }
        )
        session["submission_ids"][str(submission_id)] = assignment["assignment_id"]
        session["current_assignment_id"] = None
        self._persist_session(session)
        return result

    def _public_submission_result(
        self,
        session: dict[str, Any],
        assignment: dict[str, Any],
        event: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "accepted": True,
            "status": "accepted" if event.get("status") == "scored" else "deferred",
            "phase": assignment["phase"],
            "timing": self._timing(session),
            "synthetic": session["synthetic"],
            "next_action": "next",
        }
        if assignment["phase"] == "practice":
            result["is_correct"] = event.get("correct")
            result["cause"] = event.get("error_code")
        if event.get("status") == "deferred":
            result["degraded"] = True
            result["capability_message"] = "自由作答稍后判定；确定性题和学习进度不受影响。"
        return result

    def _answer_event(self, user_id: str, assignment_id: str) -> dict[str, Any] | None:
        for event in all_events(self.store, user_id):
            if event.get("kind") == "answer_scored" and event.get("assignment_id") == assignment_id:
                return event
        return None

    def _select_for_current_phase(self, session: dict[str, Any]) -> _Selection | None:
        phase = session["phase"]
        remaining = self._remaining_items(session, phase)
        if not remaining:
            return None
        if phase != "diagnostic":
            return _Selection(remaining[0], None)

        belief = self._belief(session["user_id"], session["node"])
        current_direct = int(session["direct_diagnostic_answers"])
        budget_items = min(
            int(session["budget"]["diagnostic_items"]),
            len(session["partitions"]["diagnostic"]["ids"]),
        )
        if selector.should_stop(
            {session["node"]: belief},
            [session["node"]],
            direct_answers={session["node"]: current_direct},
            budget_items=budget_items,
            asked=session["asked"]["diagnostic"],
        ):
            return None
        role_map = session["partitions"]["diagnostic"]["roles"]
        candidates = [
            {
                "item_id": item.item_id,
                "target_node": session["node"],
                "role": role_map[item.item_id],
                "difficulty": item.difficulty,
                "item_type": "numeric" if item.scoring_mode == "numeric" else "mcq",
                "holdout": False,
            }
            for item in remaining
        ]
        chosen = selector.select_next(
            candidates,
            {session["node"]: belief},
            [session["node"]],
            seen_ids=set(),
            asked_per_node={session["node"]: session["asked"]["diagnostic"]},
            prereq_available=any(candidate["role"] == "prereq" for candidate in candidates),
        )
        if chosen is None:
            return None
        item = self.catalog.items[chosen["item_id"]]
        eig = selector.item_eig(chosen, {session["node"]: belief})
        return _Selection(item, float(eig))

    def _remaining_items(self, session: dict[str, Any], phase: str) -> list[CatalogItem]:
        assigned_ids = {
            row["item_id"] for row in session["assignments"].values() if row["phase"] == phase
        }
        return [
            self.catalog.items[item_id]
            for item_id in session["partitions"][phase]["ids"]
            if item_id not in assigned_ids
        ]

    def _diagnostic_budget_exhausted(self, session: dict[str, Any]) -> bool:
        elapsed = (float(self.clock()) - session["phase_started_at"] - session["paused_seconds"]) / 60
        return bool(planner.check_exhaustion(session["budget"], elapsed)["force_stop"])

    def _total_budget_exhausted(self, session: dict[str, Any]) -> bool:
        return self._timing(session)["remaining_minutes"] <= 0

    def _pause_for_budget(
        self, session: dict[str, Any], resume_phase: str
    ) -> dict[str, Any]:
        session["budget_exhausted"] = True
        session["status"] = "partial"
        session["phase"] = "paused"
        session["paused_at"] = float(self.clock())
        session["checkpoint"]["resume_phase"] = resume_phase
        self._persist_session(session)
        return {
            **self._session_view(session),
            "done": True,
            "budget_exhausted": True,
            "message": "今天时间到了，进度已保存，下次从断点继续。",
            "next_action": "resume",
        }

    def _phase_after_diagnostic(self, session: dict[str, Any]) -> str:
        if self._remaining_items(session, "practice"):
            return "practice"
        if self._remaining_items(session, "held_out"):
            return "held_out"
        return "complete"

    def _next_phase(self, session: dict[str, Any]) -> str:
        if session["phase"] == "diagnostic":
            return self._phase_after_diagnostic(session)
        if session["phase"] == "practice":
            return "held_out" if self._remaining_items(session, "held_out") else "complete"
        return "complete"

    def _belief(self, user_id: str, node: str) -> np.ndarray:
        model = StudentModel.from_dict(self.get_profile(user_id))
        record = model.subject("chemistry").kg_mastery.get(node)
        if record is None:
            return np.full(4, 0.25)
        state = mastery.NodeBelief(
            b=np.asarray(record.belief, dtype=float),
            S=record.stability,
            last_review_at=record.last_review_at,
            direct_answers=record.direct_answers,
        )
        return mastery.get_belief(state, float(self.clock()))

    def _save_projection(self, user_id: str) -> dict[str, Any]:
        with self._user_lock(user_id):
            model = project_student(user_id, all_events(self.store, user_id))
            payload = model.to_dict()
            self.store.save_student(user_id, payload)
            return payload

    def _append_control_event(self, session: dict[str, Any], kind: str, now: float) -> None:
        append_once(
            self.store,
            session["user_id"],
            {
                "event_id": self._new_id(),
                "kind": kind,
                "occurred_at": now,
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "node": session["node"],
                "synthetic": session["synthetic"],
            },
        )

    def _complete_session(self, session: dict[str, Any]) -> None:
        now = float(self.clock())
        session["phase"] = "complete"
        session["status"] = "complete"
        session["completed_at"] = now
        self._persist_session(session)
        self._ensure_completion_event(session)

    def _ensure_completion_event(self, session: dict[str, Any]) -> None:
        append_once(
            self.store,
            session["user_id"],
            {
                "event_id": _session_event_id("session_completed", session["session_id"]),
                "kind": "session_completed",
                "occurred_at": float(session.get("completed_at") or self.clock()),
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "node": session["node"],
                "synthetic": session["synthetic"],
            },
        )

    def _load_session(self, session_id: str) -> dict[str, Any]:
        session = self.store.load_session(str(session_id))
        if session is None:
            raise SessionNotFound("session not found")
        _assert_session_invariants(session)
        return session

    def _persist_session(self, session: dict[str, Any]) -> None:
        _assert_session_invariants(session)
        self.store.save_session(session["session_id"], session)

    def _new_id(self) -> str:
        value = str(self.id_factory())
        if not value:
            raise RuntimeError("id_factory returned an empty id")
        return value

    def _session_lock(self, session_id: str) -> threading.RLock:
        key = str(session_id)
        with self._session_locks_guard:
            return self._session_locks.setdefault(key, threading.RLock())

    def _user_lock(self, user_id: str) -> threading.RLock:
        key = str(user_id)
        with self._user_locks_guard:
            return self._user_locks.setdefault(key, threading.RLock())

    def _session_view(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            "session_id": session["session_id"],
            "user_id": session["user_id"],
            "node": session["node"],
            "budget_tier": session["budget_tier"],
            "grade": session["grade"],
            "learning_purpose": session["learning_purpose"],
            "phase": session["phase"],
            "status": session["status"],
            "timing": self._timing(session),
            "synthetic": session["synthetic"],
        }

    def _done_view(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._session_view(session),
            "done": True,
            "budget_exhausted": session["budget_exhausted"],
            "next_action": "complete",
            "report": self.report(session["session_id"]),
        }

    def _progress(self, session: dict[str, Any]) -> dict[str, int]:
        phase = session["phase"]
        total = len(session["partitions"][phase]["ids"])
        if phase == "diagnostic":
            total = min(total, int(session["budget"]["diagnostic_items"]))
        return {
            "answered": int(session["asked"][phase]),
            "total": total,
        }

    def _timing(self, session: dict[str, Any]) -> dict[str, float]:
        total = float(_TOTAL_MINUTES[session["budget_tier"]])
        effective_now = (
            float(session["completed_at"])
            if session.get("completed_at") is not None
            else float(self.clock())
        )
        elapsed = max(
            0.0,
            (effective_now - session["created_at"] - session["paused_seconds"]) / 60,
        )
        if session["phase"] == "paused" and session.get("paused_at") is not None:
            elapsed = max(
                0.0,
                (float(session["paused_at"]) - session["created_at"] - session["paused_seconds"]) / 60,
            )
        return {
            "budget_minutes": total,
            "elapsed_minutes": round(elapsed, 3),
            "remaining_minutes": round(max(0.0, total - elapsed), 3),
        }


def _partition(
    items: list[CatalogItem], roles: dict[str, str] | None = None
) -> dict[str, Any]:
    role_map = roles or {item.item_id: "local" for item in items}
    return {
        "ids": [item.item_id for item in items],
        "families": [item.family_id for item in items],
        "roles": {item.item_id: role_map[item.item_id] for item in items},
    }


def _answer_event_id(session_id: str, assignment_id: str) -> str:
    material = f"answer:v2:{session_id}:{assignment_id}".encode()
    return hashlib.sha256(material).hexdigest()


def _session_event_id(kind: str, session_id: str) -> str:
    material = f"session:v1:{kind}:{session_id}".encode()
    return hashlib.sha256(material).hexdigest()


def _submission_hash(answer: Any) -> str:
    return hashlib.sha256(str(answer).encode("utf-8")).hexdigest()


def _likelihood_for_assignment(scored, item: CatalogItem, role: str) -> tuple[float, ...]:
    if role != "prereq" or scored.correct is None or not scored.update_allowed:
        return tuple(scored.likelihood)
    engine_type = "numeric" if item.scoring_mode == "numeric" else "mcq"
    probabilities = mastery.prereq_correct_probs(item_type=engine_type)
    likelihood = (
        mastery.likelihood_correct(probabilities)
        if scored.correct
        else mastery.likelihood_wrong_binary(probabilities)
    )
    return tuple(float(value) for value in likelihood)


def _assert_session_invariants(session: dict[str, Any]) -> None:
    partitions = session.get("partitions") or {}
    for phase in _PHASE_ORDER:
        partition = partitions.get(phase) or {}
        ids = list(partition.get("ids") or [])
        families = list(partition.get("families") or [])
        if len(ids) != len(set(ids)) or len(families) != len(set(families)):
            raise SessionError(f"duplicate rows inside {phase} partition")
        roles = partition.get("roles") or {}
        if set(roles) != set(ids) or not set(roles.values()) <= {"local", "prereq"}:
            raise SessionError(f"invalid role map inside {phase} partition")
    for index, left in enumerate(_PHASE_ORDER):
        for right in _PHASE_ORDER[index + 1 :]:
            if set(partitions[left]["ids"]) & set(partitions[right]["ids"]):
                raise SessionError(f"partition overlap by item id: {left}/{right}")
            if set(partitions[left]["families"]) & set(partitions[right]["families"]):
                raise SessionError(f"partition overlap by family: {left}/{right}")
    for assignment in (session.get("assignments") or {}).values():
        phase = assignment.get("phase")
        if phase not in _PHASE_ORDER:
            raise SessionError("assignment has invalid phase")
        partition = partitions[phase]
        if assignment.get("item_id") not in partition["ids"]:
            raise SessionError("assignment item is outside its phase partition")
        if assignment.get("family_id") not in partition["families"]:
            raise SessionError("assignment family is outside its phase partition")
    current_id = session.get("current_assignment_id")
    if current_id:
        assignment = session["assignments"].get(current_id)
        if assignment is None:
            raise SessionError("current assignment is missing")
        expected_phase = (
            session.get("checkpoint", {}).get("resume_phase")
            if session.get("phase") == "paused"
            else session.get("phase")
        )
        if assignment.get("phase") != expected_phase:
            raise SessionError("current assignment phase mismatch")


def _review_due_at(stability: float | None, last_review_at: float | None) -> str | None:
    if stability is None or last_review_at is None:
        return None
    # R(t)=0.8 under the same FSRS projection used by mastery.get_belief.
    days = 9.0 * float(stability) * (1.0 / 0.8 - 1.0)
    due = float(last_review_at) + days * mastery.DAY_SECONDS
    return datetime.fromtimestamp(due, timezone.utc).isoformat()


def _state_label(index: int) -> str:
    return {
        mastery.M: "证据显示已掌握",
        mastery.P: "前置知识仍需补强",
        mastery.C: "思路链条仍需巩固",
        mastery.U: "当前证据不足",
    }[index]


def _one_per_family(items) -> list[CatalogItem]:
    chosen: dict[str, CatalogItem] = {}
    for item in sorted(items, key=lambda row: (row.family_id, row.item_id)):
        chosen.setdefault(item.family_id, item)
    return list(chosen.values())


def _assert_public_safe(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_PUBLIC_KEYS:
                raise RuntimeError(f"private key leaked at {path}.{key}")
            if lowered == "zones" and isinstance(child, (list, tuple, set)):
                if {str(zone).lower() for zone in child} & {"answer", "analysis"}:
                    raise RuntimeError(f"private RIR zone leaked at {path}.{key}")
            _assert_public_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_public_safe(child, f"{path}[{index}]")
