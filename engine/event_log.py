"""Small persistent JSONL event log with restart-safe idempotency."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

ANCHOR_BYTES = 4096

# 事件溯源字段（2026-08-13 审计 E8：无 source 卡住一切校准；对齐 Caliper 1.2 / xAPI）
EVENT_SOURCE = ("real", "qa", "synthetic")
EVENT_SCHEMA_VERSION = 2


def with_provenance(record: Mapping[str, Any], *,
                    source: str = "real",
                    schema_version: int = EVENT_SCHEMA_VERSION) -> dict[str, Any]:
    """给事件记录附加溯源字段（source∈{real,qa,synthetic} + schema_version）。
    兼容旧记录:缺省视为 real（历史事件均来自真实交互）。"""
    if source not in EVENT_SOURCE:
        raise ValueError(f"unknown event source: {source!r}")
    enriched = dict(record)
    enriched.setdefault("source", source)
    enriched.setdefault("schema_version", schema_version)
    return enriched


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
        self._offset = 0
        self._line_count = 0
        self._file_identity: tuple[int, int] | None = None
        empty_anchor = hashlib.sha256(b"").digest()
        self._prefix_anchor = empty_anchor
        self._tail_anchor = empty_anchor
        self._known = self._load_known_keys()

    def _load_known_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open("rb") as handle:
            # fcntl is intentionally POSIX-only: production targets macOS/Linux.
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                stat = os.fstat(handle.fileno())
                self._file_identity = (stat.st_dev, stat.st_ino)
                known, self._line_count = self._read_tail(handle, 0)
                self._offset = handle.tell()
                self._capture_anchors(handle)
                return known
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_tail(self, handle, start_line: int) -> tuple[set[str], int]:
        known: set[str] = set()
        line_number = start_line
        for raw_line in handle:
            line_number += 1
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{self.path}:{line_number} 非法 JSONL") from exc
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, Mapping):
                    raise TypeError("event row must be an object")
                known.add(event_key(record))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{self.path}:{line_number} 非法 JSONL") from exc
        return known, line_number

    def _anchors_at(self, handle, offset: int) -> tuple[bytes, bytes]:
        original_position = handle.tell()
        try:
            prefix_size = min(offset, ANCHOR_BYTES)
            handle.seek(0)
            prefix = handle.read(prefix_size)
            tail_start = max(0, offset - ANCHOR_BYTES)
            handle.seek(tail_start)
            tail = handle.read(offset - tail_start)
        finally:
            handle.seek(original_position)
        return hashlib.sha256(prefix).digest(), hashlib.sha256(tail).digest()

    def _capture_anchors(self, handle) -> None:
        self._prefix_anchor, self._tail_anchor = self._anchors_at(handle, self._offset)

    def _generation_matches(self, handle, identity: tuple[int, int], size: int) -> bool:
        if identity != self._file_identity or size < self._offset:
            return False
        current_prefix, current_tail = self._anchors_at(handle, self._offset)
        return (current_prefix == self._prefix_anchor
                and current_tail == self._tail_anchor)

    def _reset_index(self, identity: tuple[int, int]) -> None:
        self._known.clear()
        self._offset = 0
        self._line_count = 0
        self._file_identity = identity
        empty_anchor = hashlib.sha256(b"").digest()
        self._prefix_anchor = empty_anchor
        self._tail_anchor = empty_anchor

    def __call__(self, record: Mapping[str, Any]) -> bool:
        return self.append(record)

    def append(self, record: Mapping[str, Any]) -> bool:
        key = event_key(record)
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    stat = os.fstat(handle.fileno())
                    identity = (stat.st_dev, stat.st_ino)
                    if not self._generation_matches(handle, identity, stat.st_size):
                        self._reset_index(identity)

                    if stat.st_size > self._offset:
                        handle.seek(self._offset)
                        new_keys, self._line_count = self._read_tail(
                            handle, self._line_count
                        )
                        self._known.update(new_keys)
                        self._offset = handle.tell()
                        self._capture_anchors(handle)
                    if key in self._known:
                        return False
                    handle.seek(0, os.SEEK_END)
                    handle.write(encoded)
                    handle.flush()
                    self._known.add(key)
                    self._line_count += 1
                    self._offset = handle.tell()
                    self._file_identity = identity
                    self._capture_anchors(handle)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True
