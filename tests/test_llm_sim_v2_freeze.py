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


def test_blind_panel_reuses_four_calibration_items_then_adds_at_most_twenty_one_families():
    from experiments.llm_sim_v2.panel import select_blind_items, select_calibration_items

    catalog = FakeCatalog(targets=1)
    for index in range(5, 32):
        item = _fake_item(index, node="Target-00")
        catalog.items[item["item_id"]] = item
    anchor = {"target_node": "Target-00"}

    calibration = select_calibration_items(anchor, catalog)
    blind = select_blind_items(anchor, catalog)

    assert blind[:4] == calibration
    assert len(blind) == 25
    assert len({row["family_id"] for row in blind}) == 25
    assert len({row["item_id"] for row in blind}) == 25


def test_study_config_prefreezes_low_mapping_coverage_degradation_and_provider_matrix():
    from experiments.llm_sim_v2.freeze import build_leakage_lexicon, build_study_config
    from experiments.llm_sim_v2.grid import build_persona_grid
    from experiments.llm_sim_v2.mapping import normalize_target_option_map

    anchors = [
        {
            "anchor_id": f"anchor-{index:02d}",
            "target_node": f"Target-{index:02d}",
            "failure_id": f"Target-{index:02d}#failure-00",
            "failure_cause": f"cause-{index}",
            "failure_symptom": f"symptom-{index}",
        }
        for index in range(25)
    ]
    personas = build_persona_grid(anchors, seed=20260715)
    items = {
        f"item-{index:03d}": {
            "item_id": f"item-{index:03d}",
            "options": {"A": "correct", "B": "wrong", "C": "other"},
            "answer_values": ["A"],
        }
        for index in range(100)
    }
    mapping_rows = []
    for index in range(100):
        mapped = index < 6
        row = {
            "item_id": f"item-{index:03d}",
            "failure_id": f"failure-{index:03d}",
            "status": "mapped" if mapped else "excluded_ambiguous",
            "target_option": "B" if mapped else None,
            "reviewer_provenance": {
                "method": "independent_dual_model_consensus",
                "drafted_by": "codex_gpt_5_6_sol_ultra",
                "crosschecked_by": "deepseek_chat",
            },
        }
        if not mapped:
            row["ambiguity_reason"] = "independent_mapping_disagreement"
        mapping_rows.append(row)
    expected = [
        (f"item-{index:03d}", f"failure-{index:03d}") for index in range(100)
    ]
    mapping = normalize_target_option_map(
        mapping_rows,
        items=items,
        expected_rows=expected,
    )
    mapping["consensus"] = {
        "mapped_rows": 6,
        "excluded_ambiguous_rows": 94,
        "draft_sha256": "c" * 64,
        "crosscheck_sha256": "d" * 64,
    }
    blind_panel = {
        "anchors": [
            {
                "anchor_id": anchor["anchor_id"],
                "items": [
                    {
                        "item_id": f"{anchor['anchor_id']}-item-{item_index:02d}",
                        "family_id": f"{anchor['anchor_id']}-family-{item_index:02d}",
                    }
                    for item_index in range(25)
                ],
                "calibration_item_ids": [
                    f"{anchor['anchor_id']}-item-{item_index:02d}"
                    for item_index in range(4)
                ],
            }
            for anchor in anchors
        ]
    }
    leakage_lexicon = build_leakage_lexicon(anchors)

    config = build_study_config(
        personas=personas,
        mapping=mapping,
        blind_panel=blind_panel,
        leakage_lexicon=leakage_lexicon,
        frozen_at_utc="2026-07-15T05:30:00Z",
    )

    assert config["run_id"] == "llm-personas-v2-dual"
    assert config["cluster_unit"] == "persona_id"
    assert config["cluster_count"] == 50
    assert config["paired_response_rows"] == 100
    assert config["repeated_measure_factors"] == ["provider", "response_arm"]
    assert config["mapping_gate"] == {
        "mapped_rows": 6,
        "total_rows": 100,
        "mapped_fraction": 0.06,
        "minimum_fraction": 0.6,
        "passed": False,
        "confirmatory_target_misconception_hit_rate": False,
        "sparse_descriptive_only": True,
    }
    assert config["controlled"]["items_per_row"] == 4
    assert config["blind"]["maximum_items_per_row"] == 25
    assert config["pilot"]["excluded_from_main_analysis"] is True
    assert len(config["pilot"]["persona_ids"]) == 5
    assert config["maximum_prompt_rewrites"] == 1
    assert config["leakage_lexicon_sha256"] == leakage_lexicon["sha256"]
    assert {
        provider: (row["model"], row["concurrency"])
        for provider, row in config["providers"].items()
    } == {
        "deepseek": ("deepseek-v4-pro", 4),
        "glm": ("glm-4-plus", 4),
        "kimi": ("moonshot-v1-128k", 4),
        "minimax": ("abab6.5s-chat", 4),
        "doubao": ("doubao-seed-2-0-mini-260428", 2),
        "tongyi": ("qwen-max", 4),
    }


def test_study_config_rejects_mapping_hash_or_consensus_tampering():
    from experiments.llm_sim_v2.freeze import build_study_config

    personas = []
    for index in range(50):
        for condition in ("deficit", "control"):
            personas.append(
                {
                    "persona_id": f"persona-{index:02d}",
                    "deficit_condition": condition,
                }
            )
    mapping = {
        "schema_version": "yher.llm_sim_v2.target_option_map.v1",
        "frozen": True,
        "observation_started": False,
        "rows": [],
        "mapping_sha256": "0" * 64,
        "target_set_hash": "0" * 64,
        "consensus": {
            "mapped_rows": 0,
            "excluded_ambiguous_rows": 100,
            "draft_sha256": "1" * 64,
            "crosscheck_sha256": "2" * 64,
        },
    }
    for index in range(100):
        mapping["rows"].append(
            {
                "item_id": f"item-{index:03d}",
                "failure_id": f"failure-{index:03d}",
                "target_option": None,
                "status": "excluded_ambiguous",
                "reviewer_provenance": {
                    "drafted_by": "codex_gpt_5_6_sol_ultra",
                    "crosschecked_by": "deepseek_chat",
                },
                "ambiguity_reason": "independent_mapping_disagreement",
            }
        )
    panel = {
        "anchors": [
            {
                "anchor_id": f"anchor-{anchor:02d}",
                "calibration_item_ids": [f"a{anchor}-i{item}" for item in range(4)],
                "items": [
                    {"item_id": f"a{anchor}-i{item}", "family_id": f"a{anchor}-f{item}"}
                    for item in range(4)
                ],
            }
            for anchor in range(25)
        ]
    }
    lexicon = {"terms": ["failure phrase"], "sha256": "3" * 64}

    with pytest.raises(ValueError, match="mapping.*sha|digest|consensus"):
        build_study_config(
            personas=personas,
            mapping=mapping,
            blind_panel=panel,
            leakage_lexicon=lexicon,
            frozen_at_utc="2026-07-15T05:30:00Z",
        )


def test_population_loader_rejects_pilot_records_from_main_analysis():
    from experiments.llm_sim_v2.freeze import validate_analysis_population

    main = {
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "phase": "main",
        "analysis_population": "main",
        "persona_id": "persona-00",
    }
    assert validate_analysis_population([main], phase="main") == [main]

    pilot = {**main, "phase": "pilot", "analysis_population": "pilot"}
    with pytest.raises(ValueError, match="pilot|population|phase"):
        validate_analysis_population([main, pilot], phase="main")


def test_leakage_lexicon_is_deterministic_and_excludes_allowed_curriculum_node_names():
    from experiments.llm_sim_v2.freeze import build_leakage_lexicon

    anchors = [
        {
            "target_node": "Acid-base equilibrium",
            "failure_id": "acid-base#failure-00",
            "failure_cause": "confuses buffer capacity with pH",
            "failure_symptom": "selects the dilution distractor",
            "diagnostic_question": "Which statement reveals that confusion?",
        }
    ]
    first = build_leakage_lexicon(anchors)
    second = build_leakage_lexicon(list(reversed(anchors)))

    assert first == second
    assert len(first["sha256"]) == 64
    assert "Acid-base equilibrium" not in first["terms"]
    assert "confuses buffer capacity with pH" in first["terms"]
    assert "selects the dilution distractor" in first["terms"]


def test_population_manifest_keeps_pilot_and_main_physically_separate_without_global_id_exclusion():
    from experiments.llm_sim_v2.freeze import build_population_manifest

    config = {
        "run_id": "llm-personas-v2-dual",
        "study_seed": 20260715,
        "response_arms": ["deficit", "control"],
        "pilot": {
            "providers": ["deepseek", "doubao"],
            "persona_ids": [f"persona-{index:02d}" for index in range(5)],
            "excluded_from_main_analysis": True,
            "physical_phase": "pilot",
        },
        "main": {
            "providers": ["deepseek", "glm", "kimi", "minimax", "doubao", "tongyi"],
            "persona_ids": [f"persona-{index:02d}" for index in range(50)],
            "physical_phase": "main",
        },
    }

    manifest = build_population_manifest(config)

    assert manifest["pilot"]["root_relative"].endswith("/pilot")
    assert manifest["main"]["root_relative"].endswith("/main")
    assert manifest["pilot"]["include_in_main"] is False
    assert manifest["main"]["include_in_main"] is True
    assert set(manifest["pilot"]["persona_ids"]) <= set(manifest["main"]["persona_ids"])
    assert manifest["ingestion_policy"]["pilot_records_never_join_main"] is True
    assert manifest["ingestion_policy"]["same_cluster_recollection_allowed"] is True
    assert manifest["population_manifest_sha256"]


def test_prompt_revision_zero_ledger_binds_prompt_files_and_rendered_contracts(tmp_path: Path):
    from experiments.llm_sim_v2.freeze import build_prompt_revision_ledger

    (tmp_path / "blind.py").write_text("BLIND = 1\n", encoding="utf-8")
    (tmp_path / "public.py").write_text("PUBLIC = 1\n", encoding="utf-8")
    ledger = build_prompt_revision_ledger(
        tmp_path,
        prompt_paths=["blind.py", "public.py"],
        rendered_contract_sha256={
            "controlled": "1" * 64,
            "blind": "2" * 64,
            "judge": "3" * 64,
        },
        leakage_lexicon_sha256="4" * 64,
        mapping_sha256="5" * 64,
        grid_sha256="6" * 64,
        panel_sha256="7" * 64,
        frozen_at_utc="2026-07-15T05:30:00Z",
    )

    assert ledger["current_revision"] == 0
    assert ledger["maximum_prompt_rewrites"] == 1
    assert len(ledger["revisions"]) == 1
    revision = ledger["revisions"][0]
    assert revision["revision"] == 0
    assert revision["parent_revision"] is None
    assert revision["calibration_rewrite_required"] is False
    assert revision["observed_row_count"] == 0
    assert len(revision["prompt_files"]) == 2
    assert revision["rendered_contract_sha256"]["blind"] == "2" * 64
    assert ledger["prompt_ledger_sha256"]


def test_freeze_manifest_recomputes_declared_bytes_and_rejects_drift(tmp_path: Path):
    from experiments.llm_sim_v2.freeze import build_freeze_manifest, verify_freeze_manifest

    (tmp_path / "plan.md").write_text("frozen plan\n", encoding="utf-8")
    (tmp_path / "mapping.json").write_text("{}\n", encoding="utf-8")
    manifest = build_freeze_manifest(
        tmp_path,
        declared_paths={"plan": ["plan.md"], "mapping": ["mapping.json"]},
        frozen_at_utc="2026-07-15T05:30:00Z",
        summary_hashes={
            "analysis_plan_sha256": "1" * 64,
            "source_set_sha256": "2" * 64,
            "official_inputs_sha256": "3" * 64,
            "grid_sha256": "4" * 64,
            "mapping_sha256": "5" * 64,
            "target_set_hash": "6" * 64,
            "prompt_ledger_sha256": "7" * 64,
            "population_manifest_sha256": "8" * 64,
        },
    )
    assert verify_freeze_manifest(tmp_path, manifest)["ok"] is True

    (tmp_path / "mapping.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="drift|hash|bytes|manifest"):
        verify_freeze_manifest(tmp_path, manifest)


def test_blind_panel_artifact_binds_calibration_prefix_and_global_unique_support():
    from experiments.llm_sim_v2.freeze import build_blind_panel

    catalog = FakeCatalog(targets=26)
    for target_index in range(25):
        node = f"Target-{target_index:02d}"
        for item_index in range(5, 30):
            item = _fake_item(item_index, node=node)
            catalog.items[item["item_id"]] = item
    anchors = [
        {"anchor_id": f"anchor-{index:02d}", "target_node": f"Target-{index:02d}"}
        for index in range(25)
    ]

    panel = build_blind_panel(anchors, catalog)

    assert len(panel["anchors"]) == 25
    assert panel["counts"]["minimum_items_per_anchor"] == 25
    assert panel["counts"]["maximum_items_per_anchor"] == 25
    all_items = []
    for anchor in panel["anchors"]:
        assert anchor["calibration_item_ids"] == [
            item["item_id"] for item in anchor["items"][:4]
        ]
        assert all(item["role"] == "calibration" for item in anchor["items"][:4])
        assert all(item["role"] == "diagnostic" for item in anchor["items"][4:])
        all_items.extend(anchor["items"])
    assert len({item["item_id"] for item in all_items}) == len(all_items)
    assert len({item["family_id"] for item in all_items}) == len(all_items)
    assert len(panel["panel_sha256"]) == 64


def test_rendered_prompt_contract_hashes_cover_controlled_blind_and_judge():
    from experiments.llm_sim_v2.freeze import build_rendered_prompt_contract_hashes

    persona = {
        "persona_id": "persona-00",
        "pair_id": "pair-00",
        "row_id": "persona-00:deficit",
        "anchor_id": "anchor-00",
        "target_node": "Target-00",
        "curriculum_exposure": ["Target-00"],
        "deficit_condition": "deficit",
        "local_skill_vector": {"ability_band": "lower", "target_skill": 0.2},
        "observable_error_policy": {"strategy": "apply_observed_failure_pattern"},
        "noise_parameters": {"level": "low", "hesitation_rate": 0.1},
        "modality_condition": "text_only",
        "seed": 1,
        "failure_id": "failure-00",
        "failure_cause": "confuses two rules",
        "failure_symptom": "selects a distractor",
    }
    items = []
    for index in range(4):
        item = _candidate("anchor-00")
        item["item_id"] = f"item-{index}"
        item["family_id"] = f"family-{index}"
        item["role"] = "calibration"
        items.append(item)
    panel = {
        "anchors": [
            {
                "anchor_id": "anchor-00",
                "calibration_item_ids": [item["item_id"] for item in items],
                "items": items,
            }
        ]
    }
    lexicon = {
        "schema_version": "yher.llm_sim_v2.leakage_lexicon.v1",
        "terms": ["confuses two rules", "selects a distractor"],
    }

    hashes = build_rendered_prompt_contract_hashes([persona], panel, lexicon)

    assert set(hashes) == {"controlled", "blind", "judge"}
    assert all(len(digest) == 64 for digest in hashes.values())
    assert len(set(hashes.values())) == 3


def test_h5v2_plan_contains_the_prefrozen_degradation_and_hard_analysis_gates():
    plan_path = REPO_ROOT / "experiments" / "h5v2_analysis_plan.md"
    assert plan_path.is_file()
    text = " ".join(plan_path.read_text(encoding="utf-8").lower().split())

    required_phrases = (
        "llm-personas-v2-dual",
        "6 of 100",
        "removed from confirmatory analysis",
        "pilot data are physically isolated and excluded",
        "exactly one prompt rewrite",
        "persona_id (n=50)",
        "provider and response arm are repeated measurements",
        "10,000",
        "unknown",
        "insufficient_evidence",
        "text_only",
    )
    for phrase in required_phrases:
        assert phrase in text

    assert "600 learners" not in text.lower()
