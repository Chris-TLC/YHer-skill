"""Side-effect-free StorePort implementation for tests and injected Demo runs."""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any


class MemoryStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._students: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}

    def load_student(self, user_id: str):
        with self._lock:
            value = self._students.get(user_id)
            return deepcopy(value) if value is not None else None

    def save_student(self, user_id: str, model: dict[str, Any]) -> None:
        with self._lock:
            self._students[user_id] = deepcopy(model)

    def save_session(self, session_id: str, session: dict[str, Any]) -> None:
        with self._lock:
            self._sessions[session_id] = deepcopy(session)

    def load_session(self, session_id: str):
        with self._lock:
            value = self._sessions.get(session_id)
            return deepcopy(value) if value is not None else None

    def append_event(self, user_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.setdefault(user_id, []).append(deepcopy(event))

    def append_event_once(self, user_id: str, event: dict[str, Any]) -> bool:
        event_id = str(event.get("event_id") or "")
        if not event_id:
            raise ValueError("event_id required")
        with self._lock:
            rows = self._events.setdefault(user_id, [])
            if any(str(row.get("event_id") or "") == event_id for row in rows):
                return False
            rows.append(deepcopy(event))
            return True

    def recent_events(self, user_id: str, limit: int = 20):
        limit = max(0, int(limit))
        with self._lock:
            return [] if limit == 0 else deepcopy(self._events.get(user_id, [])[-limit:])

    def all_events(self, user_id: str):
        with self._lock:
            return deepcopy(self._events.get(user_id, []))
