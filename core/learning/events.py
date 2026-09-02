"""Append-once learning events and deterministic StudentModel projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from core.tutor.profile_model import MasteryRecord, StudentModel
from engine import mastery


@dataclass
class ProjectedStudentModel(StudentModel):
    """StudentModel projection extended with replayable video exposure state."""

    seen_segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "seen_segments": [dict(row) for row in self.seen_segments]}


def append_once(store, user_id: str, event: dict[str, Any]) -> bool:
    """Persist one immutable event before any derived state is written."""
    if not event.get("event_id"):
        raise ValueError("event_id required")
    method = getattr(store, "append_event_once", None)
    if method is not None:
        return bool(method(user_id, event))
    existing = store.recent_events(user_id, limit=100_000)
    if any(row.get("event_id") == event["event_id"] for row in existing):
        return False
    store.append_event(user_id, event)
    return True


def all_events(store, user_id: str) -> list[dict[str, Any]]:
    method = getattr(store, "all_events", None)
    if method is not None:
        return list(method(user_id))
    return list(store.recent_events(user_id, limit=100_000))


def project_student(user_id: str, events: Iterable[dict[str, Any]]) -> StudentModel:
    """Rebuild the online profile only from the append-only event history."""
    model = ProjectedStudentModel(user_id=user_id)
    held_out_reviews: dict[str, int] = {}
    seen_segment_keys: set[tuple[str, int]] = set()
    for event in events:
        kind = event.get("kind")
        if kind == "session_started":
            model.grade = str(event.get("grade") or model.grade)
            model.learning_purpose = str(event.get("learning_purpose") or model.learning_purpose)
            continue
        if kind == "seen_segments":
            try:
                key = (str(event["bv"]), int(event["p"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key[0] and key[1] >= 1 and key not in seen_segment_keys:
                model.seen_segments.append({"bv": key[0], "p": key[1]})
                seen_segment_keys.add(key)
            continue
        if kind != "answer_scored" or event.get("update_applied") is not True:
            continue
        node = str(event.get("node") or "")
        likelihood = event.get("likelihood")
        if not node or not _valid_likelihood(likelihood):
            continue
        record = model.subject("chemistry").kg_mastery.get(node)
        belief = record.belief if record else [0.25, 0.25, 0.25, 0.25]
        state = mastery.NodeBelief(
            b=np.asarray(belief, dtype=float),
            S=record.stability if record else None,
            last_review_at=record.last_review_at if record else None,
            direct_answers=record.direct_answers if record else 0,
        )
        occurred_at = float(event.get("occurred_at") or 0.0)
        held_out_passed = event.get("phase") == "held_out" and event.get("correct") is True
        had_stability = state.S is not None
        mastery.observe(
            state,
            np.asarray(likelihood, dtype=float),
            occurred_at,
            is_direct=event.get("is_direct") is not False,
            held_out_passed=held_out_passed,
        )
        if event.get("phase") == "held_out":
            review_index = held_out_reviews.get(node, 0)
            if had_stability and isinstance(event.get("correct"), bool):
                mastery.review_update_S(
                    state,
                    passed=event["correct"],
                    now=occurred_at,
                    first_review=review_index == 0,
                )
            held_out_reviews[node] = review_index + 1
        model.subject("chemistry").kg_mastery[node] = MasteryRecord(
            value=float(state.b[mastery.M]),
            evidence=[*(record.evidence if record else []), str(event.get("event_id"))],
            last_updated=datetime.fromtimestamp(occurred_at, timezone.utc).isoformat(),
            source="rubric",
            confidence=float(event.get("confidence", 1.0)),
            belief=[float(value) for value in state.b],
            stability=state.S,
            last_review_at=state.last_review_at,
            direct_answers=state.direct_answers,
        )
    return model


def _valid_likelihood(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        array.shape == (4,)
        and bool(np.all(np.isfinite(array)))
        and bool(np.all(array >= 0))
        and float(array.sum()) > 0
    )
