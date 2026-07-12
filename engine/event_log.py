"""Small persistent JSONL event log with restart-safe idempotency."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Mapping


def event_key(record: Mapping[str, Any]) -> str:
    for field in ("event_id", "rec_id"):
        value = record.get(field)
        if value not in (None, ""):
            return f"{field}:{value}"
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonlEventLog:
    """Callable writer that appends each logical record at most once."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._known = self._load_known_keys()

    def _load_known_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        known: set[str] = set()
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    known.add(event_key(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"{self.path}:{line_number} 非法 JSONL") from exc
        return known

    def __call__(self, record: Mapping[str, Any]) -> bool:
        return self.append(record)

    def append(self, record: Mapping[str, Any]) -> bool:
        key = event_key(record)
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if key in self._known:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
            self._known.add(key)
        return True
