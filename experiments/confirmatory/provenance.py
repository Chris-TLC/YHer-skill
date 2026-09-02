"""Frozen configuration, repository, output-path, and isolation provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.s0_census import require_simulation_output_path

from .config import load_frozen_config
from .storage import read_shard_manifest


FROZEN_REPOSITORY_PATHS = (
    "experiments/analysis_plan.md",
    "experiments/config/confirmatory_v1.json",
    "experiments/confirmatory",
    "experiments/s0_census.py",
    "engine/mastery.py",
    "engine/selector.py",
    "core/data/item_bank_v4.py",
    "core/learning/item_catalog.py",
    "core/learning/scoring.py",
)


def assert_frozen_config(config) -> None:
    expected = load_frozen_config()
    if config.sha256 != expected.sha256 or dict(config.raw) != dict(expected.raw):
        raise ValueError("frozen confirmatory config changed")


def validate_run_timestamp(config, run_started_at_utc: str) -> None:
    try:
        run_started = datetime.fromisoformat(run_started_at_utc.replace("Z", "+00:00"))
        config_frozen = datetime.fromisoformat(
            str(config.raw["config_frozen_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("run and config timestamps must be valid ISO-8601 UTC") from exc
    if run_started < config_frozen:
        raise ValueError("run_started_at_utc cannot predate config freeze")


def verify_repository_binding(
    repo_root: str | Path,
    *,
    runner_commit: str,
    experiment_tag: str,
    analysis_plan_commit: str,
    frozen_paths: tuple[str, ...] = FROZEN_REPOSITORY_PATHS,
) -> dict[str, Any]:
    repository = Path(repo_root).resolve(strict=True)

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ("git", "-C", str(repository), *args),
            check=False,
            capture_output=True,
            text=True,
        )
        if check and completed.returncode != 0:
            raise ValueError(
                "repository binding failed: "
                + (completed.stderr.strip() or "git command failed")
            )
        return completed

    head = git("rev-parse", "HEAD").stdout.strip()
    if head != runner_commit:
        raise ValueError("repository HEAD does not equal runner_commit")
    tag_type = git("cat-file", "-t", f"refs/tags/{experiment_tag}").stdout.strip()
    if tag_type != "tag":
        raise ValueError("experiment tag must be annotated")
    tag_commit = git("rev-list", "-n", "1", experiment_tag).stdout.strip()
    if tag_commit != runner_commit:
        raise ValueError("experiment tag does not resolve to runner_commit")
    ancestry = git(
        "merge-base",
        "--is-ancestor",
        analysis_plan_commit,
        runner_commit,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("analysis-plan commit is not an ancestor of runner_commit")
    status = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *frozen_paths,
    ).stdout.strip()
    if status:
        raise ValueError("frozen worktree paths differ from runner_commit")
    return {
        "verified": True,
        "head": head,
        "tag": experiment_tag,
        "tag_type": tag_type,
        "tag_commit": tag_commit,
        "analysis_plan_commit": analysis_plan_commit,
        "frozen_paths": list(frozen_paths),
    }


def confirmatory_output_path(
    output_root: str | Path,
    *,
    run_id: str,
    repo_root: str | Path,
    temp_root: str | Path,
) -> Path:
    candidate = require_simulation_output_path(
        Path(output_root) / run_id,
        repo_root=repo_root,
        temp_root=temp_root,
    )
    repository = Path(repo_root).resolve(strict=False)
    simulation_root = (repository / "data" / "sim_store").resolve(strict=False)
    confirmatory_root = (simulation_root / "confirmatory").resolve(strict=False)
    if candidate == simulation_root or candidate.is_relative_to(simulation_root):
        if candidate.parent != confirmatory_root:
            raise ValueError(
                "repository output must be data/sim_store/confirmatory/<run_id>"
            )
    return candidate


def isolation_assertion(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "before_sha256": evidence["before"]["digest"],
        "after_sha256": evidence["after"]["digest"],
        "unchanged": evidence["unchanged"],
        "coverage": evidence["before"]["coverage"],
    }


def aggregate_isolation_assertions(paths: Iterable[Path]) -> dict[str, Any]:
    assertions = [
        read_shard_manifest(path)["protected_filesystem_assertion"] for path in paths
    ]
    if not assertions or not all(row.get("unchanged") is True for row in assertions):
        raise ValueError("every selected shard requires a successful isolation attestation")
    material = json.dumps(
        assertions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "before_sha256": assertions[0]["before_sha256"],
        "after_sha256": assertions[-1]["after_sha256"],
        "unchanged": all(
            row["before_sha256"] == row["after_sha256"] for row in assertions
        ),
        "coverage": assertions[0]["coverage"],
        "attested_shard_count": len(assertions),
        "attestations_sha256": hashlib.sha256(material).hexdigest(),
    }
