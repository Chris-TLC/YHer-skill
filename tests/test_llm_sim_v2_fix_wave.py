"""Regression contracts from the independent W0 review and controller audit."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


RUN_ID = "llm-personas-v2-dual"


class ActualCatalogItem:
    def __init__(self, index: int = 0, *, node: str = "Node-00") -> None:
        self.item_id = f"real-item-{index}"
        self.family_id = f"real-family-{index}"
        self.node_ids = (node,)
        self.stem_blocks = (
            {"para": [{"type": "text", "text": f"Structured stem {index}"}]},
        )
        self.stem_text = f"Structured stem {index}"
        self.options = {"A": "correct", "B": "public misconception phrase", "C": "other"}
        self.item_type = "mcq"
        self.scoring_mode = "mcq"
        self.answer_values = ("A",)
        self.difficulty = 0.5
        self.source_label = "actual-style-fixture"

    def public_question(self) -> dict[str, object]:
        return {
            "kind": self.item_type,
            "stem_blocks": list(self.stem_blocks),
            "stem_text": self.stem_text,
            "options": dict(self.options),
            "difficulty": self.difficulty,
            "nodes": list(self.node_ids),
            "source_label": self.source_label,
        }


class ActualCatalog:
    def __init__(self, count: int = 4, *, node: str = "Node-00") -> None:
        self.items = {
            item.item_id: item for item in (ActualCatalogItem(index, node=node) for index in range(count))
        }

    def for_node(self, node: str, *, deterministic_only: bool = True):
        del deterministic_only
        return tuple(item for item in self.items.values() if node in item.node_ids)


def _anchor(index: int = 0) -> dict[str, object]:
    return {
        "anchor_id": f"anchor-{index:02d}",
        "target_node": f"Node-{index:02d}",
        "failure_id": f"failure-{index:02d}",
        "failure_cause": f"cause prose {index}",
        "failure_symptom": f"symptom prose {index}",
        "curriculum_exposure": [f"Node-{index:02d}"],
    }


def _anchors() -> list[dict[str, object]]:
    return [_anchor(index) for index in range(25)]


def _row(condition: str = "deficit") -> dict[str, object]:
    return {
        "persona_id": "persona-00",
        "pair_id": "pair-00",
        "row_id": f"persona-00:{condition}",
        "target_node": "Node-00",
        "curriculum_exposure": ["Node-00"],
        "deficit_condition": condition,
        "local_skill_vector": {
            "ability_band": "lower",
            "prerequisite_skill": 0.2 if condition == "deficit" else 0.6,
            "target_skill": 0.2 if condition == "deficit" else 0.6,
            "reasoning_skill": 0.3 if condition == "deficit" else 0.7,
        },
        "observable_error_policy": (
            {
                "strategy": "apply_observed_failure_pattern",
                "cause": "cause prose 0",
                "symptom": "symptom prose 0",
            }
            if condition == "deficit"
            else {"strategy": "solve_normally"}
        ),
        "noise_parameters": {"level": "low", "hesitation_rate": 0.15},
        "modality_condition": "text_only",
        "seed": 7,
        "failure_id": "failure-00",
        "failure_cause": "cause prose 0",
        "failure_symptom": "symptom prose 0",
    }


def _content(messages: list[dict[str, object]], index: int = 1) -> dict[str, object]:
    return json.loads(str(messages[index]["content"]))


def _mapping_fixture():
    catalog = ActualCatalog()
    expected = [(item.item_id, "failure-00") for item in catalog.items.values()]
    rows = [
        {
            "item_id": item_id,
            "failure_id": failure_id,
            "target_option": "B",
            "status": "mapped",
            "reviewer": "claude",
        }
        for item_id, failure_id in expected
    ]
    return catalog, expected, rows


def test_callable_public_question_is_preserved_in_panel_and_both_prompts():
    from experiments.llm_sim_v2.panel import build_review_payload, select_calibration_items
    from experiments.llm_sim_v2.prompts import render_blind_prompt, render_controlled_prompt

    item = ActualCatalogItem()
    catalog = ActualCatalog()
    selected = select_calibration_items(_anchor(), catalog)
    review = build_review_payload(_anchor(), catalog)
    controlled = _content(render_controlled_prompt(_row(), item))
    blind = _content(render_blind_prompt(_row(), item))

    assert selected[0]["public_question"] == catalog.items[selected[0]["item_id"]].public_question()
    assert review["items"][0]["public_question"] == catalog.items[review["items"][0]["item_id"]].public_question()
    assert controlled["public_question"] == item.public_question()
    assert blind["public_question"] == item.public_question()
    assert "bound method" not in json.dumps(review, ensure_ascii=False)


def test_public_question_rejects_private_target_metadata_even_when_structured():
    from experiments.llm_sim_v2.prompts import render_blind_prompt

    item = ActualCatalogItem()
    base = vars(item) | {"public_question": item.public_question()}
    top_level = copy.deepcopy(base)
    top_level["public_question"]["target_option"] = "B"
    with pytest.raises(ValueError, match="public|private|target_option"):
        render_blind_prompt(_row(), top_level)

    nested = copy.deepcopy(base)
    nested["public_question"]["stem_blocks"].append(
        {"hidden": {"failure_cause": "private target prose"}}
    )
    with pytest.raises(ValueError, match="public|private|failure_cause"):
        render_blind_prompt(_row(), nested)

    for private_key in (
        "provider",
        "model",
        "response",
        "outcome",
        "run_id",
        "target_node",
        "candidate_output",
        "secret_token",
        "persona_id",
        "pair_id",
        "row_id",
        "anchor_id",
        "deficit_condition",
        "seed",
        "modality_condition",
        "local_skill_vector",
        "noise_parameters",
        "curriculum_exposure",
    ):
        poisoned = copy.deepcopy(base)
        poisoned["public_question"]["stem_blocks"][0][private_key] = "private metadata"
        with pytest.raises(ValueError, match="public|private|metadata"):
            render_blind_prompt(_row(), poisoned)

    unknown_top_level = copy.deepcopy(base)
    unknown_top_level["public_question"]["secret_token"] = "private metadata"
    with pytest.raises(ValueError, match="public|schema|secret"):
        render_blind_prompt(_row(), unknown_top_level)


def test_blind_leakage_validation_allows_only_the_exact_public_subtree():
    from experiments.llm_sim_v2.prompts import assert_blind_no_leakage, render_blind_prompt

    item = ActualCatalogItem()
    row = _row()
    messages = render_blind_prompt(
        row,
        item,
        frozen_leakage_lexicon=("public misconception phrase",),
    )
    assert_blind_no_leakage(
        messages,
        persona=row,
        item=item,
        frozen_leakage_lexicon=("public misconception phrase",),
    )

    lookalike = copy.deepcopy(messages)
    payload = _content(lookalike)
    payload["metadata"] = {"question": {"failure_cause": row["failure_cause"]}}
    lookalike[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="failure_cause|cause prose"):
        assert_blind_no_leakage(lookalike, persona=row, item=item)

    latent_metadata = copy.deepcopy(messages)
    payload = _content(latent_metadata)
    payload["metadata"] = {
        "deficit_condition": row["deficit_condition"],
        "persona_id": row["persona_id"],
    }
    latent_metadata[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="deficit_condition|persona_id"):
        assert_blind_no_leakage(latent_metadata, persona=row, item=item)

    for forbidden_key in ("target-option", "failure-cause", "failure‐cause"):
        lexical_variant = copy.deepcopy(messages)
        payload = _content(lexical_variant)
        payload["metadata"] = {forbidden_key: "private target metadata"}
        lexical_variant[1]["content"] = json.dumps(payload, ensure_ascii=False)
        with pytest.raises(AssertionError, match="forbidden|target|failure"):
            assert_blind_no_leakage(lexical_variant, persona=row, item=item)

    modified_public = copy.deepcopy(messages)
    payload = _content(modified_public)
    payload["public_question"]["hidden"] = {"failure_symptom": row["failure_symptom"]}
    modified_public[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="failure_symptom|symptom prose|public"):
        assert_blind_no_leakage(modified_public, persona=row, item=item)

    double_encoded = copy.deepcopy(messages)
    payload = _content(double_encoded)
    payload["wrapper"] = json.dumps(
        json.dumps({"public_text": {"observable_error_policy": row["observable_error_policy"]}})
    )
    double_encoded[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="observable_error_policy|cause prose"):
        assert_blind_no_leakage(double_encoded, persona=row, item=item)

    quoted_unicode_json = copy.deepcopy(messages)
    payload = _content(quoted_unicode_json)
    payload["wrapper"] = json.dumps(r'{"failure\u005fid":"redacted"}')
    quoted_unicode_json[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="failure_id"):
        assert_blind_no_leakage(quoted_unicode_json, persona=row, item=item)

    policy_value = copy.deepcopy(messages)
    payload = _content(policy_value)
    payload["metadata"] = {"note": row["observable_error_policy"]["strategy"]}
    policy_value[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="failure pattern|apply_observed"):
        assert_blind_no_leakage(policy_value, persona=row, item=item)


def test_judge_export_requires_context_and_rejects_injected_target_annotation():
    from experiments.llm_sim_v2.prompts import (
        assert_judge_no_target_labels,
        render_blind_prompt,
        render_judge_export,
    )

    item = ActualCatalogItem()
    row = _row()
    lexicon = ("private misconception token",)
    blind = render_blind_prompt(row, item, frozen_leakage_lexicon=lexicon)
    observed = {"simulated": True, "answer": "B", "rationale": "observed candidate output"}

    with pytest.raises(TypeError):
        render_judge_export(blind_messages=blind, model_output=observed)

    judge = render_judge_export(
        blind_messages=blind,
        model_output=observed,
        persona=row,
        item=item,
        frozen_leakage_lexicon=lexicon,
    )
    assert _content(judge)["candidate_output"] == observed

    with pytest.raises(AssertionError, match="target_option|target"):
        render_judge_export(
            blind_messages=blind,
            model_output={**observed, "target_option": "B"},
            persona=row,
            item=item,
            frozen_leakage_lexicon=lexicon,
        )

    tampered = copy.deepcopy(judge)
    payload = _content(tampered)
    payload["candidate_output"]["rationale"] = row["failure_cause"]
    tampered[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="candidate output|differs"):
        assert_judge_no_target_labels(
            tampered,
            persona=row,
            item=item,
            frozen_leakage_lexicon=lexicon,
            observed_output=observed,
        )


def test_same_cluster_blind_arms_differ_only_in_permitted_local_skill_vector():
    from experiments.llm_sim_v2.grid import build_persona_grid
    from experiments.llm_sim_v2.prompts import assert_blind_no_leakage, render_blind_prompt

    rows = build_persona_grid(_anchors(), seed=11)
    persona_id = rows[0].persona_id
    pair = [row for row in rows if row.persona_id == persona_id]
    deficit = next(row for row in pair if row.deficit_condition == "deficit")
    control = next(row for row in pair if row.deficit_condition == "control")
    item = ActualCatalogItem(node=deficit.target_node)
    deficit_messages = render_blind_prompt(deficit, item)
    control_messages = render_blind_prompt(control, item)
    deficit_payload = _content(deficit_messages)
    control_payload = _content(control_messages)

    assert deficit_payload["local_skill_vector"] != control_payload["local_skill_vector"]
    assert {key: value for key, value in deficit_payload.items() if key != "local_skill_vector"} == {
        key: value for key, value in control_payload.items() if key != "local_skill_vector"
    }
    assert_blind_no_leakage(deficit_messages, persona=deficit, item=item)
    assert_blind_no_leakage(control_messages, persona=control, item=item)


def test_controlled_policy_is_concrete_and_compliance_is_not_self_reported():
    from experiments.llm_sim_v2.grid import build_persona_grid
    from experiments.llm_sim_v2.prompts import render_blind_prompt, render_controlled_prompt

    rows = build_persona_grid(_anchors(), seed=12)
    persona_id = rows[0].persona_id
    pair = [row for row in rows if row.persona_id == persona_id]
    deficit = next(row for row in pair if row.deficit_condition == "deficit")
    control = next(row for row in pair if row.deficit_condition == "control")
    item = ActualCatalogItem(node=deficit.target_node)

    assert deficit.observable_error_policy["strategy"] == "apply_observed_failure_pattern"
    assert deficit.observable_error_policy["cause"] == deficit.failure_cause
    assert deficit.observable_error_policy["symptom"] == deficit.failure_symptom
    assert control.observable_error_policy == {"strategy": "solve_normally"}
    deficit_controlled = _content(render_controlled_prompt(deficit, item))
    control_controlled = _content(render_controlled_prompt(control, item))
    blind = _content(render_blind_prompt(deficit, item))
    assert deficit_controlled["observable_error_policy"]["cause"] == deficit.failure_cause
    assert control_controlled["observable_error_policy"] == {"strategy": "solve_normally"}
    assert "observable_error_policy" not in blind
    assert "manipulation_compliance" not in deficit_controlled["output_schema"]


def test_store_rejects_official_path_variants_and_enforces_v2_envelope(tmp_path: Path):
    from experiments.llm_sim_v2.store import V2Store

    for bad in (
        tmp_path / "item_bank" / "v4",
        tmp_path / "item-bank-v4",
        tmp_path / "knowledge_graph" / "official",
        tmp_path / "knowledge-graph-150",
        tmp_path / "official-catalog",
        tmp_path / "archive_item_bank_v4",
        tmp_path / "archive_knowledge_graph_150",
        tmp_path / "local.store",
        tmp_path / "archive-study-log",
    ):
        with pytest.raises(ValueError):
            V2Store(bad, phase="pilot")

    pilot = V2Store(tmp_path / "safe", phase="pilot")
    valid = {
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "analysis_population": "pilot",
    }
    assert pilot.write_json("record.json", valid, immutable=True).is_file()
    for index, schema_version in enumerate(
        ("yher.llm_sim_v2.target_option_map.v1", "yher.llm_sim_v2.git_proof.v1")
    ):
        assert pilot.write_json(
            f"schema-{index}.json",
            {**valid, "schema_version": schema_version},
            immutable=True,
        ).is_file()
    for bad_record in (
        {"simulated": True, "run_id": "llm-personas-v1", "phase": "pilot", "analysis_population": "pilot"},
        {"simulated": True, "run_id": RUN_ID, "phase": "main", "analysis_population": "pilot"},
        {"simulated": True, "run_id": RUN_ID, "phase": "pilot", "analysis_population": "main"},
        {"simulated": True, "run_id": RUN_ID, "phase": "pilot"},
        {"simulated": True, "run_id": RUN_ID, "phase": "pilot", "analysis_population": "pilot", "schema_version": "yher.llm_sim_v1.legacy"},
    ):
        with pytest.raises(ValueError):
            pilot.write_json("bad.json", bad_record)

    pilot.path("placed.json").write_text(
        json.dumps({"simulated": True, "run_id": "llm-personas-v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_id|phase|envelope|v1"):
        pilot.read_json("placed.json")


def test_mapping_requires_expected_catalog_complete_rows_and_known_answer():
    from experiments.llm_sim_v2.mapping import normalize_target_option_map

    catalog, expected, rows = _mapping_fixture()
    with pytest.raises(ValueError, match="expected"):
        normalize_target_option_map(rows, catalog=catalog)
    with pytest.raises(ValueError, match="expected"):
        normalize_target_option_map(rows, catalog=catalog, expected_rows=[])
    with pytest.raises(ValueError, match="catalog|items"):
        normalize_target_option_map(rows, items=[], expected_rows=expected)
    with pytest.raises(ValueError, match="missing"):
        normalize_target_option_map(rows[:-1], catalog=catalog, expected_rows=expected)
    with pytest.raises(ValueError, match="unexpected|extra"):
        normalize_target_option_map(rows + [{**rows[0], "item_id": "extra"}], catalog=catalog, expected_rows=expected)

    unknown = ActualCatalog()
    unknown.items[expected[0][0]].answer_values = ()
    with pytest.raises(ValueError, match="correct answer|answer"):
        normalize_target_option_map(rows, catalog=unknown, expected_rows=expected)


def test_mapping_hash_accessors_recompute_and_reject_advertised_tampering():
    from experiments.llm_sim_v2.mapping import mapping_sha256, normalize_target_option_map, target_set_hash

    catalog, expected, rows = _mapping_fixture()
    normalized = normalize_target_option_map(rows, catalog=catalog, expected_rows=expected)
    assert mapping_sha256(normalized) == normalized["mapping_sha256"]
    assert target_set_hash(normalized) == normalized["target_set_hash"]
    with pytest.raises(ValueError, match="hash|sha256|tamper"):
        mapping_sha256({**normalized, "mapping_sha256": "0" * 64})
    with pytest.raises(ValueError, match="hash|sha256|tamper"):
        target_set_hash({**normalized, "target_set_hash": "f" * 64})


def test_manual_mapping_reviewer_rule_is_separate_from_code_gate_self_signing():
    from experiments.llm_sim_v2 import mapping

    assert "manual" in mapping.REVIEWER_PROVENANCE_POLICY.lower()
    assert "code/test" in mapping.REVIEWER_PROVENANCE_POLICY.lower()
    assert "codex" in mapping.REVIEWER_PROVENANCE_POLICY.lower()
    catalog, expected, rows = _mapping_fixture()
    with pytest.raises(ValueError, match="codex|reviewer"):
        mapping.normalize_target_option_map(
            [{**rows[0], "reviewer": "codex_gate"}, *rows[1:]],
            catalog=catalog,
            expected_rows=expected,
        )
    accepted = [{**rows[0], "reviewer_provenance": {"reviewer": "claude", "drafted_by": "codex_agent", "crosschecked_by": "model"}}, *rows[1:]]
    assert mapping.normalize_target_option_map(accepted, catalog=catalog, expected_rows=expected)["rows"][0]["reviewer_provenance"]["reviewer"] == "claude"
    for provenance in (
        {"metadata": {"reviewer": "codex_gate"}},
        {"foo": [{"reviewed_by": "codex_gate"}]},
    ):
        with pytest.raises(ValueError, match="codex|reviewer"):
            mapping.normalize_target_option_map(
                [{**rows[0], "reviewer_provenance": provenance}, *rows[1:]],
                catalog=catalog,
                expected_rows=expected,
            )


def test_manuscript_qa_scans_wrapped_claims_from_the_correct_starting_line():
    from experiments.llm_sim_v2.manuscript_qa import scan_manuscript_text

    text = "Safe introduction.\nC&E:AI is described as\nan SCIE journal.\nThis is first-of-the-kind work.\n"
    findings = scan_manuscript_text(text)
    ceai = next(finding for finding in findings if finding.category == "ceai_index_claim")
    novelty = next(finding for finding in findings if finding.category == "novelty_claim")
    assert ceai.line == 2
    assert novelty.line == 4


def test_frozen_git_commit_must_be_ancestor_of_current_head(tmp_path: Path):
    from experiments.llm_sim_v2.provenance import verify_frozen_git_commit

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    plan = tmp_path / "plan.md"
    plan.write_text("frozen plan\n", encoding="utf-8")
    subprocess.run(["git", "add", "plan.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    subprocess.run(["git", "checkout", "-q", "-b", "side"], cwd=tmp_path, check=True)
    (tmp_path / "side.txt").write_text("side\n", encoding="utf-8")
    subprocess.run(["git", "add", "side.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "side"], cwd=tmp_path, check=True)
    side = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    subprocess.run(["git", "checkout", "-q", "--detach", base], cwd=tmp_path, check=True)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="ancestor"):
        verify_frozen_git_commit(
            tmp_path,
            commit=side,
            declared_files={"plan.md": digest},
            observation_timestamp="2999-01-01T00:00:00Z",
        )


def test_frozen_git_commit_rejects_an_empty_declared_file_set(tmp_path: Path):
    from experiments.llm_sim_v2.provenance import verify_frozen_git_commit

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    plan = tmp_path / "plan.md"
    plan.write_text("frozen plan\n", encoding="utf-8")
    subprocess.run(["git", "add", "plan.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()

    with pytest.raises(ValueError, match="empty|declared|file set"):
        verify_frozen_git_commit(
            tmp_path,
            commit=commit,
            declared_files=[],
            observation_timestamp="2999-01-01T00:00:00Z",
        )
