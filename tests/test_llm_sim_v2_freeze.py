"""Pre-observation W0.1/W1 contracts for official inputs and mapping freeze."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_item(index: int, *, node: str) -> dict[str, object]:
    return {
        "item_id": f"item-{node}-{index}",
        "family_id": f"family-{node}-{index}",
        "node_ids": [node],
        "public_question": {
            "kind": "mcq",
            "stem_blocks": [{"text": f"Question {index}"}],
            "stem_text": f"Question {index}",
            "options": {"A": "correct", "B": "wrong B", "C": "wrong C", "D": "wrong D"},
            "difficulty": 0.5,
            "nodes": [node],
            "source_label": "fixture",
        },
        "options": {"A": "correct", "B": "wrong B", "C": "wrong C", "D": "wrong D"},
        "answer_values": ["A"],
        "scoring_mode": "mcq",
    }


class FakeCatalog:
    def __init__(self, targets: int = 26) -> None:
        self.items = {
            item["item_id"]: item
            for target_index in range(targets)
            for item in (
                _fake_item(item_index, node=f"Target-{target_index:02d}")
                for item_index in range(5)
            )
        }

    def open_nodes(self) -> dict[str, int]:
        return {f"Target-{index:02d}": 5 for index in range(26)}

    def for_node(self, node: str, *, deterministic_only: bool = True):
        del deterministic_only
        return tuple(item for item in self.items.values() if node in item["node_ids"])

    def stats(self):
        return type(
            "Stats",
            (),
            {"r5_rows": 130, "trusted_items": 130, "rejected_items": 0, "families": 130},
        )()


def _kg_rows(targets: int = 26) -> list[dict[str, object]]:
    return [
        {
            "node_id": f"Target-{index:02d}",
            "common_failures": [
                {
                    "cause": f"cause-{index}-0",
                    "symptom": f"symptom-{index}-0",
                    "diagnostic_question": f"diagnostic-{index}-0",
                },
                {
                    "cause": f"cause-{index}-1",
                    "symptom": f"symptom-{index}-1",
                    "diagnostic_question": f"diagnostic-{index}-1",
                },
            ],
        }
        for index in range(targets)
    ]


def _candidate(anchor_id: str = "anchor-00") -> dict[str, object]:
    item = _fake_item(0, node="Target-00")
    return {
        "anchor_id": anchor_id,
        "target_node": "Target-00",
        "failure_id": "Target-00#failure-00",
        "failure_cause": "cause",
        "failure_symptom": "symptom",
        "item_id": item["item_id"],
        "family_id": item["family_id"],
        "public_question": item["public_question"],
        "options": item["options"],
        "correct_option": "A",
        "wrong_options": ["B", "C", "D"],
    }


def test_source_manifest_binds_missing_files_as_explicit_state(tmp_path: Path):
    from experiments.llm_sim_v2.official import build_source_manifest, verify_source_manifest

    present = tmp_path / "present.jsonl"
    missing = tmp_path / "missing.jsonl"
    present.write_text("{}\n", encoding="utf-8")
    sources = {"present": present, "service_exclusions": missing}

    manifest = build_source_manifest(tmp_path, sources=sources)
    rows = {row["role"]: row for row in manifest["files"]}
    assert rows["present"]["exists"] is True
    assert rows["present"]["sha256"]
    assert rows["service_exclusions"] == {
        "role": "service_exclusions",
        "path": "missing.jsonl",
        "exists": False,
        "sha256": None,
        "size": None,
    }
    assert verify_source_manifest(tmp_path, manifest)["ok"] is True

    missing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drift|state|exists|source"):
        verify_source_manifest(tmp_path, manifest)


def test_anchor_derivation_is_target_round_robin_and_does_not_import_v1():
    from experiments.llm_sim_v2 import official

    anchors = official.derive_anchor_roster(
        _kg_rows(),
        eligible_nodes=set(FakeCatalog().open_nodes()),
        pair_count=25,
    )

    assert len(anchors) == 25
    assert len({row["target_node"] for row in anchors}) == 25
    assert all(row["failure_id"].endswith("#failure-00") for row in anchors)
    assert "experiments.llm_sim" not in inspect.getsource(official)


def test_official_input_builder_is_deterministic_and_matches_read_only_production_roster():
    from experiments.llm_sim_v2.official import build_official_study_inputs

    first = build_official_study_inputs(REPO_ROOT)
    second = build_official_study_inputs(REPO_ROOT)

    assert first == second
    assert first["run_id"] == "llm-personas-v2-dual"
    assert first["modality_condition"] == "text_only"
    assert first["counts"] == {
        "open_targets": 27,
        "calibration_ready_targets": 26,
        "selected_anchors": 25,
        "calibration_candidates": 100,
    }
    assert first["excluded_open_targets"] == ["同分异构体"]
    assert first["unselected_ready_targets"] == ["镁铝及其化合物"]
    assert len({row["anchor_id"] for row in first["candidates"]}) == 25
    assert len({row["item_id"] for row in first["candidates"]}) == 100
    assert len({row["family_id"] for row in first["candidates"]}) == 100
    assert len(first["roster_sha256"]) == 64
    exclusions = next(
        row for row in first["source_manifest"]["files"] if row["role"] == "service_exclusions"
    )
    assert exclusions["exists"] is False


def test_official_anchors_expand_to_exactly_fifty_clusters_and_one_hundred_rows():
    from experiments.llm_sim_v2.grid import build_persona_grid
    from experiments.llm_sim_v2.official import build_official_study_inputs

    inputs = build_official_study_inputs(REPO_ROOT)
    rows = build_persona_grid(inputs["anchors"], seed=20260715)

    assert len(rows) == 100
    assert len({row.persona_id for row in rows}) == 50


def test_twenty_percent_audit_sample_covers_twenty_anchors_and_is_outcome_independent():
    from experiments.llm_sim_v2.official import select_mapping_audit_sample

    rows = []
    for anchor_index in range(25):
        for item_index in range(4):
            row = _candidate(f"anchor-{anchor_index:02d}")
            row["item_id"] = f"item-{anchor_index:02d}-{item_index}"
            row["family_id"] = f"family-{anchor_index:02d}-{item_index}"
            row["target_option"] = "B"
            rows.append(row)
    first = select_mapping_audit_sample(rows, seed=20260715)
    changed = [dict(row, target_option="C") for row in reversed(rows)]
    second = select_mapping_audit_sample(changed, seed=20260715)

    assert len(first) == 20
    assert len({row["anchor_id"] for row in first}) == 20
    assert {(row["anchor_id"], row["item_id"]) for row in first} == {
        (row["anchor_id"], row["item_id"]) for row in second
    }


def test_consensus_mapping_maps_only_exact_independent_agreement():
    from experiments.llm_sim_v2.official import build_consensus_mapping

    candidate = _candidate()
    item = _fake_item(0, node="Target-00")
    codex = [{"item_id": candidate["item_id"], "failure_id": candidate["failure_id"], "status": "mapped", "target_option": "B"}]
    crosscheck = copy.deepcopy(codex)

    mapped = build_consensus_mapping([candidate], codex, crosscheck, items={item["item_id"]: item})
    assert mapped["rows"][0]["status"] == "mapped"
    assert mapped["rows"][0]["target_option"] == "B"
    assert mapped["rows"][0]["reviewer_provenance"]["drafted_by"].startswith("codex_")
    assert mapped["rows"][0]["reviewer_provenance"]["crosschecked_by"] == "deepseek_chat"

    crosscheck[0]["target_option"] = "C"
    excluded = build_consensus_mapping([candidate], codex, crosscheck, items={item["item_id"]: item})
    assert excluded["rows"][0]["status"] == "excluded_ambiguous"
    assert excluded["rows"][0]["target_option"] is None
    assert "disagreement" in excluded["rows"][0]["ambiguity_reason"]


@pytest.mark.parametrize("bad_status", ["technical_error", "schema_error", "timeout"])
def test_consensus_mapping_never_converts_technical_failures_to_ambiguity(bad_status: str):
    from experiments.llm_sim_v2.official import build_consensus_mapping

    candidate = _candidate()
    item = _fake_item(0, node="Target-00")
    codex = [{"item_id": candidate["item_id"], "failure_id": candidate["failure_id"], "status": "mapped", "target_option": "B"}]
    crosscheck = [{**codex[0], "status": bad_status, "target_option": None}]

    with pytest.raises(ValueError, match="technical|schema|timeout|crosscheck|status"):
        build_consensus_mapping([candidate], codex, crosscheck, items={item["item_id"]: item})
