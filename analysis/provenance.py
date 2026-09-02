"""Fail-closed provenance for the committed S3 analysis implementation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

from .dataset import DatasetContractError


_STATIC_ANALYSIS_PATHS = {
    "Makefile",
    "requirements.txt",
    "engine/mastery.py",
    "engine/selector.py",
    "core/data/item_bank_v4.py",
    "core/data/knowledge_repository.py",
    "core/learning/item_catalog.py",
    "core/learning/scoring.py",
    "experiments/analysis_plan.md",
    "experiments/config/confirmatory_v1.json",
    "experiments/config/llm_sim_v1.json",
    "experiments/h5_analysis_plan.md",
    "experiments/s0_census.py",
    "analysis/static_audit_policy.json",
}


def default_analysis_paths(repo_root: Path | str) -> tuple[str, ...]:
    root = Path(repo_root).resolve()
    paths = set(_STATIC_ANALYSIS_PATHS)
    paths.update(
        str(path.relative_to(root)) for path in (root / "analysis").rglob("*.py")
    )
    paths.update(
        str(path.relative_to(root))
        for path in (root / "tests").glob("test_analysis_*.py")
    )
    paths.update(
        str(path.relative_to(root))
        for path in (root / "experiments/llm_sim").glob("*.py")
    )
    collection_lock = root / "experiments/config/h5_collection_lock.json"
    if collection_lock.is_file():
        paths.add(str(collection_lock.relative_to(root)))
    return tuple(sorted(paths))


def collect_analysis_provenance(
    repo_root: Path | str,
    *,
    relative_paths: Iterable[str] | None = None,
) -> dict[str, object]:
    """Bind selected analysis bytes to HEAD without inspecting unrelated dirt."""

    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    try:
        commit_epoch = int(
            _git(root, "show", "-s", "--format=%ct", commit)
            .decode("ascii")
            .strip()
        )
    except ValueError as exc:
        raise DatasetContractError("analysis commit timestamp is invalid") from exc
    committed_at_utc = datetime.fromtimestamp(
        commit_epoch, tz=timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    requested = tuple(
        sorted(relative_paths if relative_paths is not None else default_analysis_paths(root))
    )
    if not requested or len(requested) != len(set(requested)):
        raise DatasetContractError("analysis provenance paths must be unique and non-empty")

    file_hashes: dict[str, str] = {}
    for relative in requested:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise DatasetContractError(f"unsafe analysis provenance path: {relative}")
        path = (root / relative_path).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise DatasetContractError(
                f"analysis provenance path is missing or outside repository: {relative}"
            )
        try:
            committed = _git(root, "show", f"HEAD:{relative}")
        except DatasetContractError as exc:
            raise DatasetContractError(
                f"analysis provenance path is not tracked at HEAD: {relative}"
            ) from exc
        actual = path.read_bytes()
        if actual != committed:
            raise DatasetContractError(
                f"analysis provenance path differs from HEAD: {relative}"
            )
        file_hashes[relative] = hashlib.sha256(actual).hexdigest()

    binding = json.dumps(
        file_hashes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "analysis_commit": commit,
        "analysis_code_committed_at_utc": committed_at_utc,
        "analysis_code_sha256": hashlib.sha256(binding).hexdigest(),
        "analysis_code_files": file_hashes,
    }


def verify_analysis_provenance(
    repo_root: Path | str,
    provenance: object,
) -> None:
    """Verify a claimed analysis binding directly against committed git blobs."""

    if not isinstance(provenance, dict):
        raise DatasetContractError("analysis provenance must be an object")
    root = Path(repo_root).resolve()
    commit = provenance.get("analysis_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise DatasetContractError("analysis commit is invalid")
    _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    try:
        _git(root, "merge-base", "--is-ancestor", commit, "HEAD")
    except DatasetContractError as exc:
        raise DatasetContractError(
            "analysis commit is not reachable from HEAD"
        ) from exc
    try:
        epoch = int(
            _git(root, "show", "-s", "--format=%ct", commit)
            .decode("ascii")
            .strip()
        )
    except ValueError as exc:
        raise DatasetContractError("analysis commit timestamp is invalid") from exc
    expected_timestamp = datetime.fromtimestamp(
        epoch, tz=timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    if provenance.get("analysis_code_committed_at_utc") != expected_timestamp:
        raise DatasetContractError("analysis provenance timestamp differs from commit")

    claimed = provenance.get("analysis_code_files")
    expected_paths = _analysis_paths_at_commit(root, commit)
    if not isinstance(claimed, dict) or set(claimed) != set(expected_paths):
        raise DatasetContractError("analysis provenance lacks the exact committed scope")
    observed: dict[str, str] = {}
    for relative in expected_paths:
        blob = _git(root, "show", f"{commit}:{relative}")
        observed[relative] = hashlib.sha256(blob).hexdigest()
    if claimed != observed:
        raise DatasetContractError("analysis provenance file hash differs from commit")
    binding = json.dumps(
        observed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if provenance.get("analysis_code_sha256") != hashlib.sha256(binding).hexdigest():
        raise DatasetContractError("analysis provenance aggregate hash differs from commit")
    working_paths = default_analysis_paths(root)
    if set(working_paths) != set(expected_paths):
        raise DatasetContractError("working analysis scope differs from analysis commit")
    for relative, expected_sha in observed.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise DatasetContractError(
                f"working analysis path differs from commit: {relative}"
            )


def _analysis_paths_at_commit(root: Path, commit: str) -> tuple[str, ...]:
    try:
        tree_paths = _git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            commit,
            "--",
            "analysis",
            "experiments/llm_sim",
            "experiments/config/h5_collection_lock.json",
            "tests",
        ).decode("utf-8").split("\0")
    except UnicodeDecodeError as exc:
        raise DatasetContractError("analysis commit contains a non-UTF-8 path") from exc

    paths = set(_STATIC_ANALYSIS_PATHS)
    for relative in tree_paths:
        directory, separator, name = relative.partition("/")
        if not separator:
            continue
        if directory == "analysis" and relative.endswith(".py"):
            paths.add(relative)
        if (
            directory == "tests"
            and name.startswith("test_analysis_")
            and name.endswith(".py")
        ):
            paths.add(relative)
        if directory == "experiments" and relative.startswith(
            "experiments/llm_sim/"
        ) and relative.endswith(".py"):
            paths.add(relative)
        if relative == "experiments/config/h5_collection_lock.json":
            paths.add(relative)
    return tuple(sorted(paths))


def _git(root: Path, *args: str) -> bytes:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_GRAFT_FILE": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=True,
            capture_output=True,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DatasetContractError(
            f"git provenance command failed: git {' '.join(args)}"
        ) from exc
    return completed.stdout
