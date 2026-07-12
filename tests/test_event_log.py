import json
import tempfile
from pathlib import Path

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


def test_rec_served_retry_queue_is_bounded_and_reports_drops():
    def broken(_record):
        raise OSError("disk unavailable")

    queue = []
    for event_id in ("evt-1", "evt-2", "evt-3"):
        result = rec.append_rec_served({"event_id": event_id}, broken, queue,
                                       max_queue_size=2)
    assert [row["event_id"] for row in queue] == ["evt-2", "evt-3"]
    assert result["dropped"] == 1
