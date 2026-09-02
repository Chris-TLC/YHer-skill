from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from analysis.dataset import DatasetContractError
from analysis.provenance import collect_analysis_provenance, verify_analysis_provenance


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_default_scope(root: Path) -> None:
    files = {
        "Makefile": "paper-results:\n\t@true\n",
        "requirements.txt": "numpy\n",
        "engine/mastery.py": "# frozen production mastery\n",
        "engine/selector.py": "# frozen production selector\n",
        "core/data/item_bank_v4.py": "# frozen item loader\n",
        "core/data/knowledge_repository.py": "# frozen knowledge loader\n",
        "core/learning/item_catalog.py": "# frozen item catalog\n",
        "core/learning/scoring.py": "# frozen scoring\n",
        "experiments/analysis_plan.md": "frozen\n",
        "experiments/config/confirmatory_v1.json": "{}\n",
        "experiments/config/llm_sim_v1.json": "{}\n",
        "experiments/h5_analysis_plan.md": "frozen h5\n",
        "experiments/llm_sim/runner.py": "# frozen H5 runner\n",
        "experiments/s0_census.py": "# frozen census\n",
        "analysis/static_audit_policy.json": "{}\n",
        "analysis/a.py": "VALUE = 1\n",
        "tests/test_analysis_a.py": "def test_a(): assert True\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_analysis_provenance_is_deterministic_scoped_and_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "analysis").mkdir(parents=True)
    (root / "analysis/a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "Makefile").write_text("paper-results:\n\t@true\n", encoding="utf-8")
    (root / "unrelated.txt").write_text("clean\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "analysis@example.invalid")
    _git(root, "config", "user.name", "Analysis Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    scope = ("Makefile", "analysis/a.py")

    first = collect_analysis_provenance(root, relative_paths=scope)
    second = collect_analysis_provenance(root, relative_paths=scope)

    assert first == second
    assert set(first["analysis_code_files"]) == set(scope)
    assert len(str(first["analysis_code_sha256"])) == 64
    assert str(first["analysis_code_committed_at_utc"]).endswith("Z")

    (root / "unrelated.txt").write_text("dirty but out of scope\n", encoding="utf-8")
    assert collect_analysis_provenance(root, relative_paths=scope) == first

    (root / "analysis/a.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(DatasetContractError, match="differs from HEAD"):
        collect_analysis_provenance(root, relative_paths=scope)


def test_verifier_recomputes_exact_committed_scope_hashes_and_timestamp(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write_default_scope(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "analysis@example.invalid")
    _git(root, "config", "user.name", "Analysis Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    provenance = collect_analysis_provenance(root)

    verify_analysis_provenance(root, provenance)

    forged = dict(provenance)
    forged["analysis_code_committed_at_utc"] = "2025-01-01T00:00:00Z"
    with pytest.raises(DatasetContractError, match="timestamp"):
        verify_analysis_provenance(root, forged)
    forged = dict(provenance)
    forged["analysis_code_files"] = {
        **provenance["analysis_code_files"],
        "analysis/extra.py": "7" * 64,
    }
    with pytest.raises(DatasetContractError, match="exact committed scope"):
        verify_analysis_provenance(root, forged)


def test_default_scope_binds_programmatic_replay_engine_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write_default_scope(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "analysis@example.invalid")
    _git(root, "config", "user.name", "Analysis Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")

    provenance = collect_analysis_provenance(root)
    assert {"engine/mastery.py", "engine/selector.py"} <= set(
        provenance["analysis_code_files"]
    )
    (root / "engine/selector.py").write_text("# forged selector\n", encoding="utf-8")
    with pytest.raises(DatasetContractError, match="differs from HEAD"):
        collect_analysis_provenance(root)


def test_default_scope_binds_h5_replay_dependencies(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_default_scope(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "analysis@example.invalid")
    _git(root, "config", "user.name", "Analysis Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")

    provenance = collect_analysis_provenance(root)
    assert "experiments/llm_sim/runner.py" in provenance["analysis_code_files"]
    (root / "experiments/llm_sim/runner.py").write_text(
        "# forged H5 runner\n", encoding="utf-8"
    )
    with pytest.raises(DatasetContractError, match="differs from HEAD"):
        collect_analysis_provenance(root)


@pytest.mark.parametrize(
    "hidden_relative",
    ("analysis/hidden.py", "tests/test_analysis_hidden.py"),
)
def test_verifier_derives_scope_from_claimed_commit_tree(
    tmp_path: Path,
    hidden_relative: str,
) -> None:
    root = tmp_path / "repo"
    _write_default_scope(root)
    hidden = root / hidden_relative
    hidden.write_text("HIDDEN = True\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "analysis@example.invalid")
    _git(root, "config", "user.name", "Analysis Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture with hidden scope file")

    hidden.unlink()
    provenance = collect_analysis_provenance(root)
    assert hidden_relative not in provenance["analysis_code_files"]

    with pytest.raises(DatasetContractError, match="exact committed scope"):
        verify_analysis_provenance(root, provenance)


def test_verifier_rejects_commit_not_reachable_from_head(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_default_scope(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "analysis@example.invalid")
    _git(root, "config", "user.name", "Analysis Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "trusted head")
    trusted_branch = _git(root, "branch", "--show-current")

    _git(root, "switch", "-qc", "attack")
    (root / "analysis/a.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(root, "add", "analysis/a.py")
    _git(root, "commit", "-qm", "untrusted provenance commit")
    provenance = collect_analysis_provenance(root)
    attack_commit = str(provenance["analysis_commit"])

    _git(root, "switch", "-q", trusted_branch)
    _git(root, "update-ref", "-d", "refs/heads/attack")
    assert _git(root, "cat-file", "-t", attack_commit) == "commit"

    with pytest.raises(DatasetContractError, match="reachable from HEAD"):
        verify_analysis_provenance(root, provenance)


def test_verifier_ignores_git_replacement_objects(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_default_scope(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "analysis@example.invalid")
    _git(root, "config", "user.name", "Analysis Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "common parent")
    common_parent = _git(root, "rev-parse", "HEAD")
    trusted_branch = _git(root, "branch", "--show-current")

    (root / "analysis/a.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(root, "add", "analysis/a.py")
    _git(root, "commit", "-qm", "trusted analysis")
    trusted_commit = _git(root, "rev-parse", "HEAD")

    _git(root, "switch", "-qc", "replacement", common_parent)
    (root / "analysis/a.py").write_text("VALUE = 999\n", encoding="utf-8")
    _git(root, "add", "analysis/a.py")
    _git(root, "commit", "-qm", "replacement analysis")
    replacement_provenance = collect_analysis_provenance(root)
    replacement_commit = str(replacement_provenance["analysis_commit"])

    _git(root, "switch", "-q", trusted_branch)
    _git(root, "replace", trusted_commit, replacement_commit)
    forged = {**replacement_provenance, "analysis_commit": trusted_commit}

    with pytest.raises(DatasetContractError, match="file hash differs from commit"):
        verify_analysis_provenance(root, forged)


def test_provenance_ignores_inherited_git_repository_redirection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "trusted"
    _write_default_scope(root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "analysis@example.invalid")
    _git(root, "config", "user.name", "Analysis Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "trusted analysis")
    trusted_commit = _git(root, "rev-parse", "HEAD")

    foreign = tmp_path / "foreign"
    _write_default_scope(foreign)
    _git(foreign, "init", "-q")
    _git(foreign, "config", "user.email", "attacker@example.invalid")
    _git(foreign, "config", "user.name", "Foreign Repository")
    _git(foreign, "add", ".")
    _git(foreign, "commit", "-qm", "foreign analysis")
    foreign_provenance = collect_analysis_provenance(foreign)
    assert foreign_provenance["analysis_commit"] != trusted_commit

    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(root))
    monkeypatch.setenv("GIT_COMMON_DIR", str(foreign / ".git"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(foreign / ".git/objects"))
    monkeypatch.setenv("GIT_GRAFT_FILE", str(foreign / ".git/info/grafts"))

    collected = collect_analysis_provenance(root)
    assert collected["analysis_commit"] == trusted_commit
    verify_analysis_provenance(root, collected)
    with pytest.raises(DatasetContractError, match="git provenance command failed"):
        verify_analysis_provenance(root, foreign_provenance)
