"""Byte-level provenance helpers with secret-safe manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _rooted_path(root: Path, relative: str | Path) -> tuple[str, Path]:
    path_value = Path(relative)
    if path_value.is_absolute() or ".." in path_value.parts:
        raise ValueError("provenance paths must be relative to the repository")
    normalized = path_value.as_posix()
    path = (root / path_value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("provenance path escapes repository")
    return normalized, path


def _reject_sensitive_path(relative: str) -> None:
    lowered = relative.lower()
    if lowered == ".env" or lowered.startswith(".env.") or any(token in lowered for token in ("secret", "credential", "api_key", "apikey")):
        raise ValueError("secret-bearing paths cannot enter provenance")


def hash_declared_files(
    repo_root: str | Path,
    declared_paths: Sequence[str | Path] | Mapping[str, Any],
) -> dict[str, Any]:
    """Hash only the declared bytes; no file content or environment is serialized."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    entries: list[tuple[str | None, str | Path]] = []
    categories = {"code", "config", "prompt", "mapping", "plan"}
    if isinstance(declared_paths, Mapping) and set(declared_paths).issubset(categories):
        for category, category_paths in declared_paths.items():
            values = [category_paths] if isinstance(category_paths, (str, Path)) else list(category_paths)
            entries.extend((str(category), value) for value in values)
    elif isinstance(declared_paths, Mapping):
        entries = [(None, value) for value in declared_paths.keys()]
    else:
        entries = [(None, value) for value in declared_paths]
    if not entries:
        raise ValueError("declared provenance file set cannot be empty")
    rows: list[dict[str, Any]] = []
    for category, relative in entries:
        normalized, path = _rooted_path(root, relative)
        _reject_sensitive_path(normalized)
        data = path.read_bytes()
        row = {
            "path": normalized,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        if category is not None:
            row["category"] = category
        rows.append(row)
    rows.sort(key=lambda row: row["path"])
    return {
        "schema_version": "yher.llm_sim_v2.provenance.v1",
        "files": rows,
        "file_set_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
    }


def _declared_rows(value: Mapping[str, Any] | Sequence[Any]) -> list[dict[str, str]]:
    if isinstance(value, Mapping):
        if "files" in value:
            value = value["files"]
        else:
            value = [{"path": path, "sha256": digest} for path, digest in value.items()]
    rows: list[dict[str, str]] = []
    for row in value:
        if isinstance(row, Mapping):
            path = str(row.get("path") or "")
            digest = str(row.get("sha256") or "")
        else:
            path, digest = row
            path, digest = str(path), str(digest)
        if not path or len(digest) != 64:
            raise ValueError("declared provenance rows require path and sha256")
        rows.append({"path": path, "sha256": digest.lower()})
    if not rows:
        raise ValueError("declared provenance file set cannot be empty")
    return sorted(rows, key=lambda row: row["path"])


def _observation_epoch(value: str | int | float | datetime) -> float:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.timestamp()


def verify_frozen_git_commit(
    repo_root: str | Path,
    *,
    commit: str,
    declared_files: Mapping[str, Any] | Sequence[Any],
    observation_timestamp: str | int | float | datetime,
) -> dict[str, Any]:
    """Verify commit blobs equal the declared and current bytes before observation."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    commit_id = subprocess.check_output(["git", "rev-parse", "--verify", commit], cwd=root, text=True).strip()
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit_id, current_head],
        cwd=root,
        capture_output=True,
    )
    if ancestry.returncode == 1:
        raise ValueError("frozen commit is not an ancestor of current HEAD")
    if ancestry.returncode != 0:
        raise RuntimeError("cannot verify frozen commit ancestry")
    rows = _declared_rows(declared_files)
    commit_epoch = float(subprocess.check_output(["git", "show", "-s", "--format=%ct", commit_id], cwd=root, text=True).strip())
    observation_epoch = _observation_epoch(observation_timestamp)
    file_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized, current_path = _rooted_path(root, row["path"])
        _reject_sensitive_path(normalized)
        try:
            committed = subprocess.check_output(["git", "show", f"{commit_id}:{normalized}"], cwd=root)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"frozen commit is missing declared file {normalized}") from exc
        committed_sha = hashlib.sha256(committed).hexdigest()
        current = current_path.read_bytes()
        current_sha = hashlib.sha256(current).hexdigest()
        if committed_sha != row["sha256"] or current_sha != committed_sha:
            raise ValueError(f"frozen file is not byte-identical: {normalized}")
        file_rows.append({"path": normalized, "sha256": committed_sha, "byte_identical": True})
    precedes = commit_epoch < observation_epoch
    if not precedes:
        raise ValueError("frozen commit does not precede observation timestamp")
    return {
        "schema_version": "yher.llm_sim_v2.git_proof.v1",
        "ok": True,
        "commit": commit_id,
        "current_head": current_head,
        "ancestor_of_head": True,
        "commit_timestamp_utc": datetime.fromtimestamp(commit_epoch, tz=timezone.utc).isoformat(),
        "observation_timestamp": str(observation_timestamp),
        "precedes_observation": True,
        "byte_identical": True,
        "files": file_rows,
    }


def verify_frozen_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return verify_frozen_git_commit(*args, **kwargs)


def hash_file_set(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return hash_declared_files(*args, **kwargs)


hash_provenance_files = hash_declared_files
verify_frozen_provenance = verify_frozen_git_commit


__all__ = [
    "hash_declared_files",
    "hash_file_set",
    "verify_frozen_git_commit",
    "verify_frozen_commit",
]
