#!/usr/bin/env python3
"""Prepare, audit, and roll back authorized official-data changes."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


class GuardError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validated_token(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise GuardError(f"invalid {label}")
    return value


def _relative_to(path: Path, parent: Path, message: str) -> Path:
    try:
        return path.relative_to(parent)
    except ValueError as exc:
        raise GuardError(message) from exc


def prepare(
    *,
    workspace: Path,
    delivery_dir: Path,
    step: str,
    date: str,
    paths: Iterable[Path],
) -> Path:
    workspace = Path(workspace).resolve()
    delivery_dir = Path(delivery_dir).resolve()
    step = _validated_token(str(step), "step")
    date = _validated_token(str(date), "date")
    data_root = (workspace / "data").resolve()
    targets = [Path(path).resolve() for path in paths]
    if not targets:
        raise GuardError("at least one path is required")

    relative_targets: list[tuple[Path, Path]] = []
    for target in targets:
        _relative_to(target, workspace, "path outside workspace")
        relative = _relative_to(target, data_root, "path outside official data root")
        if not target.is_file():
            raise GuardError(f"official file missing: {target}")
        relative_targets.append((target, relative))

    backup_root = data_root / f"_backup_pre_{step}_{date}"
    if backup_root.exists():
        raise GuardError(f"backup already exists: {backup_root}")
    manifest_dir = delivery_dir / "manifest" / step
    snapshot_path = manifest_dir / "snapshot.json"
    if manifest_dir.exists():
        raise GuardError(f"manifest directory already exists: {manifest_dir}")

    backup_root.mkdir(parents=True)
    files: list[dict[str, str]] = []
    for target, relative in relative_targets:
        backup = backup_root / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        files.append(
            {
                "path": str(target.relative_to(workspace)),
                "backup": str(backup.relative_to(workspace)),
                "before_sha256": _sha256(backup),
            }
        )

    snapshot = {
        "schema_version": "yher_official_guard_v1",
        "workspace": str(workspace),
        "delivery_dir": str(delivery_dir),
        "step": step,
        "date": date,
        "backup_root": str(backup_root.relative_to(workspace)),
        "files": files,
    }
    _write_json(snapshot_path, snapshot)
    return snapshot_path


def _changed_lines(before: list[str], after: list[str]) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for operation, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        width = max(before_end - before_start, after_end - after_start)
        for offset in range(width):
            before_index = before_start + offset
            after_index = after_start + offset
            changed.append(
                {
                    "operation": operation,
                    "before_line": before_index + 1 if before_index < before_end else None,
                    "after_line": after_index + 1 if after_index < after_end else None,
                    "before": before[before_index] if before_index < before_end else None,
                    "after": after[after_index] if after_index < after_end else None,
                }
            )
    return changed


def finalize(snapshot_path: Path) -> Path:
    snapshot_path = Path(snapshot_path).resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    workspace = Path(snapshot["workspace"]).resolve()
    files: list[dict[str, Any]] = []

    for entry in snapshot["files"]:
        target = workspace / entry["path"]
        backup = workspace / entry["backup"]
        if not target.is_file() or not backup.is_file():
            raise GuardError(f"target or backup missing for {entry['path']}")
        before = backup.read_text(encoding="utf-8").splitlines()
        after = target.read_text(encoding="utf-8").splitlines()
        files.append(
            {
                **entry,
                "after_sha256": _sha256(target),
                "before_line_count": len(before),
                "after_line_count": len(after),
                "changed_lines": _changed_lines(before, after),
            }
        )

    manifest_path = snapshot_path.parent / "manifest.json"
    if manifest_path.exists():
        raise GuardError(f"manifest already exists: {manifest_path}")
    script_path = Path(__file__).resolve()
    manifest = {
        **{key: value for key, value in snapshot.items() if key != "files"},
        "files": files,
        "rollback_command": f"python3 {script_path} rollback --manifest {manifest_path}",
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def rollback(manifest_path: Path) -> None:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace = Path(manifest["workspace"]).resolve()
    for entry in manifest["files"]:
        target = workspace / entry["path"]
        backup = workspace / entry["backup"]
        if not backup.is_file():
            raise GuardError(f"backup missing: {backup}")
        if _sha256(backup) != entry["before_sha256"]:
            raise GuardError(f"backup hash mismatch: {backup}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--workspace", type=Path, required=True)
    prepare_parser.add_argument("--delivery-dir", type=Path, required=True)
    prepare_parser.add_argument("--step", required=True)
    prepare_parser.add_argument("--date", required=True)
    prepare_parser.add_argument("--path", action="append", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--snapshot", type=Path, required=True)
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(
            workspace=args.workspace,
            delivery_dir=args.delivery_dir,
            step=args.step,
            date=args.date,
            paths=args.path,
        )
        print(result)
    elif args.command == "finalize":
        print(finalize(args.snapshot))
    else:
        rollback(args.manifest)
        print("rollback complete; backups retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
