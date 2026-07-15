"""Adversarial regressions for the final Persona v2 W0 hard gates."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


RUN_ID = "llm-personas-v2-dual"


def _item() -> dict[str, object]:
    return {
        "item_id": "item-1",
        "family_id": "family-1",
        "node_ids": ["Node-1"],
        "public_question": {
            "kind": "mcq",
            "stem_blocks": [{"text": "Which option is correct?"}],
            "stem_text": "Which option is correct?",
            "options": {"A": "correct", "B": "wrong", "C": "other"},
            "difficulty": 0.5,
            "nodes": ["Node-1"],
            "source_label": "fixture",
        },
        "options": {"A": "correct", "B": "wrong", "C": "other"},
        "answer_values": ["A"],
        "scoring_mode": "mcq",
    }


def _persona() -> dict[str, object]:
    return {
        "persona_id": "persona-1",
        "pair_id": "pair-1",
        "row_id": "persona-1:deficit",
        "target_node": "Node-1",
        "curriculum_exposure": ["Node-1"],
        "deficit_condition": "deficit",
        "local_skill_vector": {"target_skill": 0.2},
        "observable_error_policy": {"strategy": "apply failure", "cause": "private cause"},
        "noise_parameters": {"level": "low"},
        "modality_condition": "text_only",
        "seed": 1,
        "failure_id": "failure-1",
        "failure_cause": "private cause",
        "failure_symptom": "private symptom",
    }


def _mapping_fixture() -> tuple[dict[str, object], list[tuple[str, str]], list[dict[str, object]]]:
    item = _item()
    catalog = {str(item["item_id"]): item}
    expected = [(str(item["item_id"]), "failure-1")]
    rows = [
        {
            "item_id": item["item_id"],
            "failure_id": "failure-1",
            "target_option": "B",
            "status": "mapped",
            "reviewer_provenance": {"reviewer": "claude", "method": "manual_review"},
        }
    ]
    return catalog, expected, rows


@pytest.mark.parametrize("private_key", ["targetOption", "failureCause", "personaId", "failure‐Symptom"])
def test_public_question_rejects_camel_case_and_unicode_private_keys(private_key: str):
    from experiments.llm_sim_v2.public import public_question_payload

    poisoned = _item()
    poisoned["public_question"] = copy.deepcopy(poisoned["public_question"])
    poisoned["public_question"]["stem_blocks"].append({"hidden": {private_key: "private"}})

    with pytest.raises(ValueError, match="private|target|failure|persona"):
        public_question_payload(poisoned)


def test_public_question_requires_canonical_top_level_schema_keys():
    from experiments.llm_sim_v2.public import public_question_payload

    item = _item()
    public = copy.deepcopy(item["public_question"])
    public["sourceLabel"] = public.pop("source_label")
    item["public_question"] = public

    with pytest.raises(ValueError, match="canonical|schema|source"):
        public_question_payload(item)


@pytest.mark.parametrize("private_key", ["targetOption", "failureCause", "personaId", "failure‐Symptom"])
def test_blind_scanner_rejects_camel_case_and_unicode_private_keys(private_key: str):
    from experiments.llm_sim_v2.prompts import assert_blind_no_leakage, render_blind_prompt

    messages = render_blind_prompt(_persona(), _item())
    payload = json.loads(messages[1]["content"])
    payload["metadata"] = {private_key: "private"}
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False)

    with pytest.raises(AssertionError, match="forbidden|target|failure|persona"):
        assert_blind_no_leakage(messages, persona=_persona(), item=_item())


@pytest.mark.parametrize("observed_key", ["providerId", "modelId", "observedAt", "observed‐at"])
def test_persona_schema_uses_canonical_sensitive_key_normalization(observed_key: str):
    from experiments.llm_sim_v2.models import PersonaV2

    row = _persona()
    row["noise_parameters"] = {observed_key: "observed"}

    with pytest.raises(ValueError, match="observed|provider|model"):
        PersonaV2.from_mapping(row)


def test_persona_mapping_rejects_boolean_seed_instead_of_coercing_it_to_one():
    from experiments.llm_sim_v2.models import PersonaV2

    row = _persona()
    row["seed"] = True
    with pytest.raises(ValueError, match="seed|integer"):
        PersonaV2.from_mapping(row)


def test_persona_grid_rejects_observed_camel_case_fields_in_anchor_records():
    from experiments.llm_sim_v2.grid import build_persona_grid

    anchors = []
    for index in range(25):
        anchors.append(
            {
                "anchor_id": f"anchor-{index}",
                "target_node": f"Node-{index}",
                "failure_id": f"failure-{index}",
                "failure_cause": f"cause-{index}",
                "failure_symptom": f"symptom-{index}",
                "curriculum_exposure": [f"Node-{index}"],
            }
        )
    anchors[0]["providerId"] = "observed-provider"

    with pytest.raises(ValueError, match="provider|observed"):
        build_persona_grid(anchors)


def test_judge_candidate_binding_distinguishes_omitted_from_explicit_null_and_rejects_aliases():
    from experiments.llm_sim_v2.prompts import (
        assert_judge_no_target_labels,
        render_blind_prompt,
        render_judge_export,
    )

    blind = render_blind_prompt(_persona(), _item())
    export = render_judge_export(
        blind_messages=blind,
        model_output=None,
        persona=_persona(),
        item=_item(),
        frozen_leakage_lexicon=(),
    )
    assert json.loads(export[1]["content"])["candidate_output"] is None
    assert_judge_no_target_labels(
        export,
        persona=_persona(),
        item=_item(),
        observed_output=None,
    )

    tampered = copy.deepcopy(export)
    payload = json.loads(tampered[1]["content"])
    payload.pop("candidate_output")
    payload["candidateOutput"] = {"answer": "B"}
    tampered[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="candidate|supplied|canonical"):
        assert_judge_no_target_labels(
            tampered,
            persona=_persona(),
            item=_item(),
            observed_output=None,
        )

    with pytest.raises(AssertionError, match="candidate|forbidden"):
        assert_judge_no_target_labels(export, persona=_persona(), item=_item())


def test_locked_mapping_verifies_existing_advertised_hashes_before_row_comparison():
    from experiments.llm_sim_v2.mapping import normalize_target_option_map

    catalog, expected, rows = _mapping_fixture()
    frozen = normalize_target_option_map(rows, items=catalog, expected_rows=expected)
    tampered_existing = copy.deepcopy(frozen)
    tampered_existing["rows"][0]["target_option"] = "C"
    replacement = copy.deepcopy(rows)
    replacement[0]["target_option"] = "C"

    with pytest.raises(ValueError, match="hash|sha256|target_set|advertised"):
        normalize_target_option_map(
            replacement,
            items=catalog,
            expected_rows=expected,
            existing=tampered_existing,
            observation_started=True,
        )


def test_locked_mapping_requires_both_frozen_hashes_and_validates_each_one():
    from experiments.llm_sim_v2.mapping import normalize_target_option_map

    catalog, expected, rows = _mapping_fixture()
    frozen = normalize_target_option_map(rows, items=catalog, expected_rows=expected)
    for missing in ("mapping_sha256", "target_set_hash"):
        incomplete = copy.deepcopy(frozen)
        incomplete.pop(missing)
        with pytest.raises(ValueError, match="hash|sha256|target_set|frozen"):
            normalize_target_option_map(
                rows,
                items=catalog,
                expected_rows=expected,
                existing=incomplete,
                observation_started=True,
            )

    stale_target = copy.deepcopy(frozen)
    stale_target["rows"][0]["target_option"] = "C"
    stale_target["mapping_sha256"] = hashlib.sha256(
        json.dumps(
            stale_target["rows"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    replacement = copy.deepcopy(rows)
    replacement[0]["target_option"] = "C"
    with pytest.raises(ValueError, match="target_set|hash|advertised"):
        normalize_target_option_map(
            replacement,
            items=catalog,
            expected_rows=expected,
            existing=stale_target,
            observation_started=True,
        )


def test_normalized_mapping_owns_a_deep_copy_of_reviewer_provenance():
    from experiments.llm_sim_v2.mapping import mapping_sha256, normalize_target_option_map

    catalog, expected, rows = _mapping_fixture()
    provenance = rows[0]["reviewer_provenance"]
    frozen = normalize_target_option_map(rows, items=catalog, expected_rows=expected)
    provenance["reviewer"] = "codex_after_freeze"

    assert frozen["rows"][0]["reviewer_provenance"]["reviewer"] == "claude"
    assert mapping_sha256(frozen) == frozen["mapping_sha256"]


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        "claude",
        [],
        {"manualReviewer": "codex_agent"},
        {"reviewerName": "codexAgent"},
        {"reviewer_name": "codex agent"},
        {"signer": "codex‐agent"},
        {"reviewer": {"name": "codex_agent"}},
        {"metadata": {"reviewed by": "codex_agent"}},
        {"metadata": {"reviewed‐by": "codex_agent"}},
    ],
)
def test_mapping_requires_structured_provenance_and_normalizes_signer_fields(provenance: object):
    from experiments.llm_sim_v2.mapping import normalize_target_option_map

    catalog, expected, rows = _mapping_fixture()
    rows[0]["reviewer_provenance"] = provenance

    with pytest.raises(ValueError, match="provenance|reviewer|codex|structured"):
        normalize_target_option_map(rows, items=catalog, expected_rows=expected)


def test_mapping_allows_codex_only_in_non_signer_provenance_fields():
    from experiments.llm_sim_v2.mapping import normalize_target_option_map

    catalog, expected, rows = _mapping_fixture()
    rows[0]["reviewer_provenance"] = {
        "drafted_by": "codex_agent",
        "crosschecked_by": "deepseek-chat",
        "method": "independent_model_crosscheck",
    }

    normalized = normalize_target_option_map(rows, items=catalog, expected_rows=expected)
    assert normalized["rows"][0]["reviewer_provenance"]["drafted_by"] == "codex_agent"


def test_mapping_rejects_a_codex_reviewer_hidden_beside_safe_provenance():
    from experiments.llm_sim_v2.mapping import normalize_target_option_map

    catalog, expected, rows = _mapping_fixture()
    rows[0]["reviewer"] = "codex_manual"

    with pytest.raises(ValueError, match="codex|reviewer"):
        normalize_target_option_map(rows, items=catalog, expected_rows=expected)


def test_judge_binding_requires_exactly_one_candidate_output():
    from experiments.llm_sim_v2.prompts import assert_judge_no_target_labels, render_blind_prompt, render_judge_export

    observed = {"answer": "B", "rationale": "observed"}
    export = render_judge_export(
        blind_messages=render_blind_prompt(_persona(), _item()),
        model_output=observed,
        persona=_persona(),
        item=_item(),
        frozen_leakage_lexicon=(),
    )

    missing = copy.deepcopy(export)
    payload = json.loads(missing[1]["content"])
    payload.pop("candidate_output")
    missing[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="exactly one|candidate"):
        assert_judge_no_target_labels(missing, persona=_persona(), item=_item(), observed_output=observed)

    duplicate = copy.deepcopy(export)
    payload = json.loads(duplicate[1]["content"])
    payload["wrapper"] = {"candidate_output": observed}
    duplicate[1]["content"] = json.dumps(payload)
    with pytest.raises(AssertionError, match="exactly one|candidate"):
        assert_judge_no_target_labels(duplicate, persona=_persona(), item=_item(), observed_output=observed)


@pytest.mark.parametrize(
    "nested",
    [
        {"metadata": {"run_id": "llm-personas-v1"}},
        {"events": [{"runId": "llm-personas-v1"}]},
        {"metadata": {"study‐run‐id": "archive/llm-personas-v1"}},
    ],
)
def test_store_recursively_rejects_nested_v1_run_envelopes(tmp_path: Path, nested: dict[str, object]):
    from experiments.llm_sim_v2.store import V2Store

    record = {
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "analysis_population": "pilot",
        "schema_version": "yher.llm_sim_v2.observation.v1",
        **nested,
    }

    with pytest.raises(ValueError, match="v1|envelope|run"):
        V2Store(tmp_path, phase="pilot").write_json("bad.json", record)


@pytest.mark.parametrize("key", ["sourceRunId", "parent_run_id"])
def test_store_rejects_v1_references_in_nested_run_id_fields(
    tmp_path_factory: pytest.TempPathFactory,
    key: str,
):
    from experiments.llm_sim_v2.store import V2Store

    record = {
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "main",
        "analysis_population": "main",
        "metadata": {key: "llm-personas-v1"},
    }
    with pytest.raises(ValueError, match="v1|run|envelope"):
        V2Store(tmp_path_factory.mktemp("safe_store"), phase="main").write_json("bad.json", record)


@pytest.mark.parametrize(
    "field",
    ["authenticity_score", "authenticityScore", "truthfulness", "truthfulness‐score", "realism_score"],
)
def test_judge_output_rejects_authenticity_truthfulness_and_realism_fields(field: str):
    from experiments.llm_sim_v2.prompts import validate_judge_output

    with pytest.raises(ValueError, match="agreement|error|field|authentic|truth|realism"):
        validate_judge_output({"label": "consistent", field: 0.99})


def test_judge_output_accepts_only_agreement_and_error_category_contract():
    from experiments.llm_sim_v2.prompts import validate_judge_output

    result = validate_judge_output(
        {
            "label": "inconsistent",
            "error_category": "reasoning_mismatch",
            "rationale": "The answer and rationale disagree.",
        }
    )
    assert result == {
        "label": "inconsistent",
        "error_category": "reasoning_mismatch",
        "rationale": "The answer and rationale disagree.",
        "simulated": True,
    }
    with pytest.raises(ValueError, match="field|agreement|error"):
        validate_judge_output({"label": "consistent", "confidence": 0.9})


@pytest.mark.parametrize(
    "claim,category",
    [
        ("We evaluated n=600 simulated learners.", "sample_size_claim"),
        ("The panel contains 600 synthetic learners.", "sample_size_claim"),
        ("This approximates the distribution of actual students.", "real_student_distribution"),
    ],
)
def test_manuscript_qa_catches_sample_and_distribution_variants(claim: str, category: str):
    from experiments.llm_sim_v2.manuscript_qa import scan_manuscript_text

    assert category in {finding.category for finding in scan_manuscript_text(claim)}


def test_frozen_provenance_rejects_stale_file_set_digest(tmp_path: Path):
    from experiments.llm_sim_v2.provenance import hash_declared_files, verify_frozen_git_commit

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "mapping.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "plan.md").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "mapping.json", "plan.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "freeze"], cwd=tmp_path, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    manifest = hash_declared_files(tmp_path, ["mapping.json", "plan.md"])
    stale = copy.deepcopy(manifest)
    stale["files"] = [row for row in stale["files"] if row["path"] != "mapping.json"]

    with pytest.raises(ValueError, match="file.set|digest|sha256|manifest"):
        verify_frozen_git_commit(
            tmp_path,
            commit=commit,
            declared_files=stale,
            observation_timestamp="2999-01-01T00:00:00Z",
        )


@pytest.mark.parametrize("tamper", ["size", "duplicate"])
def test_frozen_provenance_validates_structured_manifest_rows(tmp_path: Path, tamper: str):
    from experiments.llm_sim_v2.provenance import hash_declared_files, verify_frozen_git_commit

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "plan.md").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "plan.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "freeze"], cwd=tmp_path, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    manifest = hash_declared_files(tmp_path, ["plan.md"])
    if tamper == "size":
        manifest["files"][0]["size"] += 1
    else:
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    manifest["file_set_sha256"] = hashlib.sha256(
        json.dumps(
            manifest["files"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="size|duplicate|manifest|file"):
        verify_frozen_git_commit(
            tmp_path,
            commit=commit,
            declared_files=manifest,
            observation_timestamp="2999-01-01T00:00:00Z",
        )
