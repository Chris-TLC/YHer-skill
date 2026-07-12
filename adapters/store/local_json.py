#!/usr/bin/env python3
"""
本地 JSON 存储（零配置，Chris 本地拿给同学测时用）。

数据落在 data/local_store/ 下，按 user_id 分文件。
云端上线时换成 SupabaseStore，引擎层零改动。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.event_log import JsonlEventLog

SKILL_DIR = Path(__file__).parent.parent.parent
DEFAULT_STORE_DIR = SKILL_DIR / "data" / "local_store"


def _safe_key(key: str) -> str:
    """把 user_id/session_id 变成安全文件名。"""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", key)[:80] or "anon"


class LocalJsonStore:
    """文件级 JSON 存储。简单、可检查、零依赖。"""

    def __init__(self, store_dir: Optional[Path] = None):
        self.root = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.students = self.root / "students"
        self.sessions = self.root / "sessions"
        self.events = self.root / "events"
        self._write_lock = threading.RLock()
        for d in (self.students, self.sessions, self.events):
            d.mkdir(parents=True, exist_ok=True)

    # ── 学生模型 ──────────────────────────────────────────
    def load_student(self, user_id: str) -> Optional[Dict[str, Any]]:
        path = self.students / f"{_safe_key(user_id)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_student(self, user_id: str, model: Dict[str, Any]) -> None:
        path = self.students / f"{_safe_key(user_id)}.json"
        self._atomic_write_json(path, model)

    # ── 会话 ──────────────────────────────────────────────
    def save_session(self, session_id: str, session: Dict[str, Any]) -> None:
        path = self.sessions / f"{_safe_key(session_id)}.json"
        self._atomic_write_json(path, session)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self.sessions / f"{_safe_key(session_id)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ── 学习事件（活人感③）─────────────────────────────────
    def append_event(self, user_id: str, event: Dict[str, Any]) -> None:
        path = self.events / f"{_safe_key(user_id)}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def append_event_once(self, user_id: str, event: Dict[str, Any]) -> bool:
        """Append an event id once. Callers project state only after this succeeds."""
        event_id = str(event.get("event_id") or "")
        if not event_id:
            raise ValueError("event_id required")
        path = self.events / f"{_safe_key(user_id)}.jsonl"
        appended = JsonlEventLog(path).append(event)
        if appended:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        return appended

    def recent_events(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(0, int(limit))
        return [] if limit == 0 else self.all_events(user_id)[-limit:]

    def all_events(self, user_id: str) -> List[Dict[str, Any]]:
        path = self.events / f"{_safe_key(user_id)}.jsonl"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def _atomic_write_json(self, path: Path, value: Dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        with self._write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
