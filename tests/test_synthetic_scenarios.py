"""Contracts for physically isolated, explicitly synthetic Demo replays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = REPO_ROOT / "demo" / "synthetic_scenarios"
REAL_LOG_ROOTS = (
    REPO_ROOT / "data" / "local_store",
    REPO_ROOT / "data" / "study_logs",
)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_manifest_exists_before_loading_the_synthetic_suite() -> None:
    assert (SCENARIO_ROOT / "manifest.json").is_file()


def test_suite_has_exact_balanced_matrix_and_honest_node_availability() -> None:
    from demo.synthetic_scenarios.validate import load_suite, validate_suite

    manifest, scenarios = load_suite(SCENARIO_ROOT)
    summary = validate_suite(manifest, scenarios)

    assert summary["scenario_count"] == 24
    assert summary["episode_count"] == 32
    assert summary["single_session_scenarios"] == 16
    assert summary["double_session_scenarios"] == 8
    assert summary["planned_node_count"] == 28
    assert summary["current_open_node_count"] == 27
    assert summary["closed_planned_nodes"] == ["化学反应速率"]
    assert summary["budget_outcome_counts"] == {
        f"{budget}:{outcome}": 2
        for budget in ("30min", "1h", "2h")
        for outcome in ("verified", "needs_reinforcement", "partial", "paused")
    }
    assert all(scenario["synthetic"] is True for scenario in scenarios)


def test_replay_is_deterministic_offline_and_does_not_touch_real_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from demo.synthetic_scenarios.replay import replay_suite

    before = {str(path): _tree_digest(path) for path in REAL_LOG_ROOTS}

    def deny_network(*_args, **_kwargs):
        raise AssertionError("synthetic replay attempted network access")

    monkeypatch.setattr("socket.create_connection", deny_network)
    first = replay_suite(SCENARIO_ROOT, tmp_path / "run-a")
    second = replay_suite(SCENARIO_ROOT, tmp_path / "run-b")

    assert first["digest"] == second["digest"]
    assert first["scenario_count"] == 24
    assert first["episode_count"] == 32
    assert first["expected_closed_episodes"] == 1
    assert first["unexpected_failures"] == []
    assert first["all_persisted_rows_synthetic"] is True
    assert {str(path): _tree_digest(path) for path in REAL_LOG_ROOTS} == before


@pytest.mark.parametrize("real_root", REAL_LOG_ROOTS)
def test_replay_refuses_real_log_destinations(real_root: Path) -> None:
    from demo.synthetic_scenarios.replay import assert_isolated_output

    with pytest.raises(ValueError, match="synthetic replay output"):
        assert_isolated_output(real_root)


def test_replay_refuses_every_repository_destination() -> None:
    from demo.synthetic_scenarios.replay import assert_isolated_output

    with pytest.raises(ValueError, match="synthetic replay output"):
        assert_isolated_output(SCENARIO_ROOT / "_runs")
