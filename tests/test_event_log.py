import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from engine import recommender as rec
from engine.event_log import JsonlEventLog


def test_rec_served_jsonl_is_persistent_and_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rec_served.jsonl"
        writer = JsonlEventLog(path)
        queue = []
        snapshot = {"event_id": "evt-1", "served": [{"rec_id": "r1"}]}
        assert rec.append_rec_served(snapshot, writer, queue)["ok"]
        assert rec.append_rec_served(snapshot, writer, queue)["ok"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows == [snapshot]


def test_two_preinitialized_writers_append_same_event_only_once(tmp_path):
    path = tmp_path / "events.jsonl"
    first = JsonlEventLog(path)
    second = JsonlEventLog(path)
    event = {"event_id": "shared", "kind": "rec_served"}

    assert first.append(event) is True
    assert second.append(event) is False
    assert [json.loads(line) for line in path.read_text().splitlines()] == [event]


def test_existing_invalid_jsonl_is_rejected_explicitly(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event_id":"valid"}\nnot-json\n', encoding="utf-8")

    try:
        JsonlEventLog(path)
        assert False, "非法旧 JSONL 必须显式报错"
    except ValueError as exc:
        assert f"{path}:2" in str(exc)


def test_rec_served_retry_queue_is_bounded_and_reports_drops():
    def broken(_record):
        raise OSError("disk unavailable")

    queue = []
    for event_id in ("evt-1", "evt-2", "evt-3"):
        result = rec.append_rec_served({"event_id": event_id}, broken, queue,
                                       max_queue_size=2)
    assert [row["event_id"] for row in queue] == ["evt-2", "evt-3"]
    assert result["dropped"] == 1


def _real_recommend(rec_id, **business_key):
    track_map = rec.load_track_map({
        "tracks": [{
            "id": "foundation",
            "audience": {"高二": {"review": 1.0}},
            "diagnostic_unlock": ["U", "P"],
        }],
        "entities": [{
            "entity": "bv:F1", "track": "foundation", "reviewer": "user_chris",
            "needs_human": False, "evidence": "fixture:event-log",
        }],
    })
    segment = {
        "segment_id": "seg-1", "bv": "F1", "p": 1,
        "seg_type": "concept_intro", "difficulty": "T2",
        "topic_match_ratio": 1.0, "duration_sec": 120,
        "start_sec": 0, "end_sec": 120, "view": 1,
        "pubdate": "2026-07-13", "part_title": "氧化还原基础",
    }
    return rec.recommend(
        "高二", "review", ["氧化还原"],
        {"氧化还原": np.array([0.1, 0.1, 0.1, 0.7])},
        {"氧化还原": [segment]}, track_map,
        {"mode": "full", "rx_minutes": 30, "rx_segments": 1},
        rec_id_factory=lambda: rec_id,
        **business_key,
    )


def test_real_recommend_retry_has_stable_business_event_id(tmp_path):
    first = _real_recommend("rec-first", session_id="session-1", action_id="action-1")
    retry = _real_recommend("rec-retry", session_id="session-1", action_id="action-1")
    assert first["recommendations"][0]["rec_id"] != retry["recommendations"][0]["rec_id"]
    assert first["rec_served"]["event_id"] == retry["rec_served"]["event_id"]
    assert first["rec_served"]["event_id"].startswith("rec_served:")
    assert first["rec_served"]["session_id"] == "session-1"
    assert first["rec_served"]["action_id"] == "action-1"

    writer = JsonlEventLog(tmp_path / "rec_served.jsonl")
    assert writer.append(first["rec_served"]) is True
    assert writer.append(retry["rec_served"]) is False
    rows = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert rows == [first["rec_served"]]


def test_recommend_business_key_is_pairwise_and_legacy_ids_never_collapse():
    for partial in ({"session_id": "session-1"}, {"action_id": "action-1"}):
        with pytest.raises(ValueError):
            _real_recommend("rec-partial", **partial)

    first = _real_recommend("legacy-first")
    second = _real_recommend("legacy-second")
    assert first["rec_served"]["idempotency_mode"] == "legacy_rec_id"
    assert second["rec_served"]["idempotency_mode"] == "legacy_rec_id"
    assert first["rec_served"]["event_id"] != second["rec_served"]["event_id"]


def test_recommend_rejects_explicit_blank_business_ids():
    invalid_keys = (
        {"session_id": "", "action_id": ""},
        {"session_id": "   ", "action_id": "\t"},
        {"session_id": "", "action_id": "action-1"},
        {"session_id": "session-1", "action_id": " "},
    )
    for business_key in invalid_keys:
        with pytest.raises(ValueError):
            _real_recommend("rec-explicit-blank", **business_key)


def test_incremental_writer_reads_only_new_tail_after_2000_appends(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = JsonlEventLog(path)
    for index in range(2000):
        assert writer.append({"event_id": f"main-{index}"}) is True
    offset_before_external_append = writer._offset

    tail_scans = []
    original = writer._read_tail

    def spy(handle, start_line):
        tail_scans.append((handle.tell(), path.stat().st_size))
        return original(handle, start_line)

    writer._read_tail = spy
    other = JsonlEventLog(path)
    assert other.append({"event_id": "external"}) is True
    size_after_external_append = path.stat().st_size
    assert writer.append({"event_id": "main-final"}) is True

    assert tail_scans == [(offset_before_external_append, size_after_external_append)]
    start, end = tail_scans[0]
    assert start > 0
    assert end - start < 200
    assert writer._offset == path.stat().st_size


def test_incremental_writer_rebuilds_index_after_truncate(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = JsonlEventLog(path)
    event = {"event_id": "repeat-after-truncate"}
    assert writer.append(event) is True
    path.write_text("", encoding="utf-8")
    assert writer.append(event) is True
    assert [json.loads(line) for line in path.read_text().splitlines()] == [event]


def test_incremental_writer_detects_copytruncate_refill_past_old_offset(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = JsonlEventLog(path)
    stale_event = {"event_id": "stale-generation"}
    for index in range(50):
        assert writer.append({"event_id": f"old-{index}"}) is True
    assert writer.append(stale_event) is True
    old_offset = writer._offset
    old_inode = path.stat().st_ino

    replacement = {
        "event_id": "replacement-generation",
        "padding": "x" * (old_offset + 128),
    }
    path.write_text(json.dumps(replacement) + "\n", encoding="utf-8")
    assert path.stat().st_ino == old_inode
    assert path.stat().st_size >= old_offset

    assert writer.append(stale_event) is True
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event_id"] for row in rows] == [
        "replacement-generation", "stale-generation",
    ]
