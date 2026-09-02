"""Reversibility contracts for authorized official-data edits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_prepare_finalize_and_rollback_preserve_backup(tmp_path: Path) -> None:
    from scripts.official_change_guard import finalize, prepare, rollback

    workspace = tmp_path / "repo"
    target = workspace / "data" / "item_bank" / "v4" / "rows.jsonl"
    target.parent.mkdir(parents=True)
    original = '{"id":1,"status":"pending"}\n{"id":2,"status":"pending"}\n'
    target.write_text(original, encoding="utf-8")
    delivery = tmp_path / "delivery"

    snapshot = prepare(
        workspace=workspace,
        delivery_dir=delivery,
        step="bad_reports",
        date="20260713",
        paths=[target],
    )
    backup = workspace / "data" / "_backup_pre_bad_reports_20260713" / "item_bank" / "v4" / "rows.jsonl"
    assert backup.read_text(encoding="utf-8") == original

    target.write_text(
        '{"id":1,"status":"resolved"}\n{"id":2,"status":"pending"}\n',
        encoding="utf-8",
    )
    manifest_path = finalize(snapshot)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["step"] == "bad_reports"
    assert manifest["rollback_command"].endswith(f" rollback --manifest {manifest_path}")
    assert manifest["files"][0]["before_sha256"] != manifest["files"][0]["after_sha256"]
    assert manifest["files"][0]["changed_lines"] == [
        {
            "operation": "replace",
            "before_line": 1,
            "after_line": 1,
            "before": '{"id":1,"status":"pending"}',
            "after": '{"id":1,"status":"resolved"}',
        }
    ]

    rollback(manifest_path)
    assert target.read_text(encoding="utf-8") == original
    assert backup.read_text(encoding="utf-8") == original


def test_prepare_never_overwrites_an_existing_backup(tmp_path: Path) -> None:
    from scripts.official_change_guard import GuardError, prepare

    workspace = tmp_path / "repo"
    target = workspace / "data" / "rows.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    backup_root = workspace / "data" / "_backup_pre_once_20260713"
    backup_root.mkdir()

    with pytest.raises(GuardError, match="backup already exists"):
        prepare(
            workspace=workspace,
            delivery_dir=tmp_path / "delivery",
            step="once",
            date="20260713",
            paths=[target],
        )


def test_prepare_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    from scripts.official_change_guard import GuardError, prepare

    workspace = tmp_path / "repo"
    (workspace / "data").mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(GuardError, match="outside workspace"):
        prepare(
            workspace=workspace,
            delivery_dir=tmp_path / "delivery",
            step="escape",
            date="20260713",
            paths=[outside],
        )
