"""S0 contracts for isolated, provenance-bound simulation census output."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_output_guard_allows_only_sim_store_and_sprint_tmp(tmp_path: Path) -> None:
    from experiments.s0_census import require_simulation_output_path

    repo = tmp_path / "repo"
    sim_store = repo / "data" / "sim_store"
    sprint_tmp = tmp_path / "yher_sprint2"
    outside = tmp_path / "outside"
    sim_store.mkdir(parents=True)
    sprint_tmp.mkdir()
    outside.mkdir()

    assert require_simulation_output_path(
        sim_store / "census" / "rows.jsonl", repo_root=repo, temp_root=sprint_tmp
    ) == (sim_store / "census" / "rows.jsonl").resolve()
    assert require_simulation_output_path(
        sprint_tmp / "WORKLOG.md", repo_root=repo, temp_root=sprint_tmp
    ) == (sprint_tmp / "WORKLOG.md").resolve()

    for forbidden in (
        repo / "data" / "local_store" / "events.jsonl",
        repo / "data" / "sim_store-not-allowed" / "rows.jsonl",
        outside / "rows.jsonl",
    ):
        with pytest.raises(ValueError, match="simulation output path"):
            require_simulation_output_path(
                forbidden, repo_root=repo, temp_root=sprint_tmp
            )

    (sim_store / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="simulation output path"):
        require_simulation_output_path(
            sim_store / "escape" / "rows.jsonl",
            repo_root=repo,
            temp_root=sprint_tmp,
        )

    redirected_repo = tmp_path / "redirected-repo"
    (redirected_repo / "data").mkdir(parents=True)
    (redirected_repo / "data" / "sim_store").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(ValueError, match="simulation output path"):
        require_simulation_output_path(
            redirected_repo / "data" / "sim_store" / "rows.jsonl",
            repo_root=redirected_repo,
            temp_root=sprint_tmp,
        )


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"simulated": False, "persona_id": "p", "provider": "local", "model_id": "m"},
        {"simulated": True, "persona_id": " ", "provider": "local", "model_id": "m"},
        {"simulated": True, "persona_id": "p", "provider": "", "model_id": "m"},
        {"simulated": True, "persona_id": "p", "provider": "local", "model_id": None},
    ],
)
def test_simulated_event_envelope_rejects_missing_or_false_fields(event: dict) -> None:
    from experiments.s0_census import require_simulated_event_envelope

    with pytest.raises(ValueError, match="simulated event envelope"):
        require_simulated_event_envelope(event)


def test_simulated_event_envelope_accepts_explicit_nonempty_identity() -> None:
    from experiments.s0_census import require_simulated_event_envelope

    event = {
        "simulated": True,
        "persona_id": "census:all-open-nodes",
        "provider": "local_programmatic",
        "model_id": "deterministic-census-v1",
    }
    assert require_simulated_event_envelope(event) is event


def test_guarded_run_detects_student_snapshot_log_and_cache_writes(tmp_path: Path) -> None:
    from experiments.s0_census import (
        ProtectedWriteError,
        guarded_simulation_run,
        protected_filesystem_fingerprint,
    )

    repo = tmp_path / "repo"
    student = repo / "data" / "local_store" / "students" / "profile.json"
    session = repo / "data" / "local_store" / "sessions" / "session.json"
    rec_served = repo / "data" / "local_store" / "events" / "rec_served.jsonl"
    study_log = repo / "data" / "study_logs" / "events.jsonl"
    cache = repo / "core" / "__pycache__" / "module.pyc"
    named_cache = repo / "data" / "item_bank" / "v4" / "ws2_omml_latex_cache_v1.jsonl"
    for path in (student, session, rec_served, study_log, cache, named_cache):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    before = protected_filesystem_fingerprint(repo)
    assert before["digest"]
    assert "data/local_store/students" in before["coverage"]
    assert "data/local_store/sessions" in before["coverage"]
    assert "data/local_store/events" in before["coverage"]
    assert "data/study_logs" in before["coverage"]
    assert "repository_cache_paths" in before["coverage"]
    assert "data/item_bank/v4/ws2_omml_latex_cache_v1.jsonl" in before["entries"]

    with guarded_simulation_run(repo):
        output = repo / "data" / "sim_store" / "census" / "ok.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
    assert protected_filesystem_fingerprint(repo) == before

    with pytest.raises(ProtectedWriteError, match="protected simulation state changed"):
        with guarded_simulation_run(repo):
            session.write_text("mutated", encoding="utf-8")


class _FakeCatalog:
    def __init__(self) -> None:
        self.items = {
            "p1": SimpleNamespace(
                item_id="p1",
                family_id="family-a",
                node_ids=("alpha(x)_y",),
                deterministic=True,
            ),
            "p2": SimpleNamespace(
                item_id="p2",
                family_id="family-a",
                node_ids=("alpha(x)_y",),
                deterministic=True,
            ),
            "p3": SimpleNamespace(
                item_id="p3",
                family_id="family-b",
                node_ids=("alpha(x)_y",),
                deterministic=True,
            ),
            "p4": SimpleNamespace(
                item_id="p4",
                family_id="family-c",
                node_ids=("alpha(x)_y",),
                deterministic=False,
            ),
            **{
                f"t{index}": SimpleNamespace(
                    item_id=f"t{index}",
                    family_id=f"target-family-{index}",
                    node_ids=("Target",),
                    deterministic=True,
                )
                for index in range(1, 7)
            },
            **{
                f"n{index}": SimpleNamespace(
                    item_id=f"n{index}",
                    family_id=f"none-family-{index}",
                    node_ids=("NoPrereq",),
                    deterministic=True,
                )
                for index in range(1, 6)
            },
        }

    def open_nodes(self) -> dict[str, int]:
        return {"NoPrereq": 5, "Target": 6}

    def prerequisites_for(self, node: str) -> tuple[str, ...]:
        return {
            "Target": ("alpha（x）/y", "unmatched"),
            "NoPrereq": (),
        }[node]

    def for_node(self, node: str, *, deterministic_only: bool = False) -> tuple:
        rows = tuple(item for item in self.items.values() if node in item.node_ids)
        if deterministic_only:
            rows = tuple(item for item in rows if item.deterministic)
        return rows


def test_census_uses_only_mechanical_label_normalization_and_family_counts() -> None:
    from experiments.s0_census import build_prerequisite_census

    first = build_prerequisite_census(_FakeCatalog())
    second = build_prerequisite_census(_FakeCatalog())
    assert first == second
    assert first["open_node_count"] == 2
    assert first["h1_h2_eligible_count"] == 1
    assert first["structurally_prerequisite_free_count"] == 1

    target = next(row for row in first["nodes"] if row["target_node"] == "Target")
    assert target["h1_h2_eligible"] is True
    assert target["local_deterministic_item_count"] == 6
    assert target["local_independent_family_count"] == 6
    assert target["no_repeat_local_capacity_after_holdout"] == 4
    assert target["no_repeat_local_budget_eligible"] == {"9": False, "15": False, "25": False}

    resolved, unmatched = target["prerequisites"]
    assert resolved == {
        "raw_label": "alpha（x）/y",
        "normalized_label": "alpha(x)_y",
        "resolved_label": "alpha(x)_y",
        "deterministic_r5_item_count": 3,
        "independent_deterministic_family_count": 2,
        "has_two_independent_deterministic_families": True,
    }
    assert unmatched["raw_label"] == "unmatched"
    assert unmatched["resolved_label"] is None
    assert unmatched["deterministic_r5_item_count"] == 0
    assert unmatched["independent_deterministic_family_count"] == 0
    assert unmatched["has_two_independent_deterministic_families"] is False


def test_real_prerequisite_census_matches_frozen_sanity() -> None:
    from core.learning.item_catalog import ItemCatalog
    from experiments.s0_census import build_prerequisite_census

    census = build_prerequisite_census(ItemCatalog.from_default_data())
    assert census["open_node_count"] == 27
    assert census["h1_h2_eligible_count"] == 23
    assert census["structurally_prerequisite_free_count"] == 4
    assert census["insufficient_prerequisite_coverage_count"] == 0
    assert len(census["nodes"]) == 27
    assert all(
        {
            "raw_label",
            "normalized_label",
            "resolved_label",
            "deterministic_r5_item_count",
            "independent_deterministic_family_count",
            "has_two_independent_deterministic_families",
        }
        <= prerequisite.keys()
        for node in census["nodes"]
        for prerequisite in node["prerequisites"]
    )


def test_census_artifacts_wrap_every_json_record_with_envelope_and_provenance(
    tmp_path: Path,
) -> None:
    from experiments.s0_census import write_census_artifacts

    repo = tmp_path / "repo"
    temp_root = tmp_path / "yher_sprint2"
    output_dir = repo / "data" / "sim_store" / "census"
    worklog = temp_root / "WORKLOG.md"
    census = {
        "open_node_count": 1,
        "h1_h2_eligible_count": 1,
        "structurally_prerequisite_free_count": 0,
        "insufficient_prerequisite_coverage_count": 0,
        "nodes": [{"target_node": "Target", "h1_h2_eligible": True, "prerequisites": []}],
    }
    provenance = {
        "repository_head": "a" * 40,
        "analysis_plan_commit": "b" * 40,
        "config_sha256": "c" * 64,
        "input_sha256": {"kg": "d" * 64, "r5": "e" * 64},
    }

    paths = write_census_artifacts(
        census,
        provenance,
        output_dir=output_dir,
        worklog_path=worklog,
        repo_root=repo,
        temp_root=temp_root,
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in paths["records"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    for record in [summary, *rows]:
        assert record["simulated"] is True
        assert record["persona_id"]
        assert record["provider"] == "local_programmatic"
        assert record["model_id"] == "deterministic-census-v1"
        assert record["provenance"] == provenance
    assert "eligible targets: 1/1" in worklog.read_text(encoding="utf-8")
