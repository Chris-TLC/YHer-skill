"""Prompt, leakage, and manuscript QA contracts for Persona v2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _anchor() -> dict[str, object]:
    return {
        "anchor_id": "anchor-00",
        "target_node": "Acid-base",
        "failure_id": "failure-00",
        "failure_cause": "confuses conjugate pairs",
        "failure_symptom": "selects the wrong proton donor",
        "curriculum_exposure": ["Acid-base"],
    }


def _row() -> dict[str, object]:
    return {
        "persona_id": "persona-00-low",
        "pair_id": "pair-00-low",
        "row_id": "persona-00-low:deficit",
        "target_node": "Acid-base",
        "curriculum_exposure": ["Acid-base"],
        "deficit_condition": "deficit",
        "local_skill_vector": {"ability_band": "low", "acid_base": 0.2},
        "observable_error_policy": {"guessing": "sometimes", "omit_work": True},
        "noise_parameters": {"level": "low", "hesitation_rate": 0.2},
        "modality_condition": "text_only",
        "seed": 7,
        "failure_id": "failure-00",
        "failure_cause": "confuses conjugate pairs",
        "failure_symptom": "selects the wrong proton donor",
    }


def _item() -> dict[str, object]:
    return {
        "item_id": "item-00",
        "family_id": "family-00",
        "public_question": "Which option identifies the proton donor?",
        "options": {"A": "correct donor", "B": "wrong donor", "C": "other", "D": "other"},
        "answer_values": ["A"],
    }


def _import(name: str):
    try:
        module = __import__(name, fromlist=["*"])
    except ImportError as exc:  # pragma: no cover - expected during RED
        pytest.fail(f"v2 implementation is missing: {name}: {exc}")
    return module


def test_controlled_prompt_requests_strict_json_and_exposes_compliance_policy():
    prompts = _import("experiments.llm_sim_v2.prompts")

    messages = prompts.render_controlled_prompt(_row(), _item())
    text = json.dumps(messages, ensure_ascii=False)

    assert '"simulated": true' in text
    assert "strict JSON object" in text
    assert "manipulation_compliance" in text
    assert "observable_error_policy" in text


def test_prompt_renderers_accept_catalog_record_objects():
    prompts = _import("experiments.llm_sim_v2.prompts")

    class Item:
        public_question = "Record-backed public question"
        options = {"A": "one", "B": "two"}

    controlled = prompts.render_controlled_prompt(_row(), Item())
    blind = prompts.render_blind_prompt(_row(), Item())
    assert "Record-backed public question" in json.dumps(controlled)
    assert "Record-backed public question" in json.dumps(blind)


def test_blind_prompt_contains_only_allowed_context_and_leakage_scan_is_clean():
    prompts = _import("experiments.llm_sim_v2.prompts")

    messages = prompts.render_blind_prompt(_row(), _item(), frozen_leakage_lexicon=("misconception-token",))
    serialized = json.dumps(messages, ensure_ascii=False).lower()

    assert '"simulated": true' in serialized
    for forbidden in (
        "failure_id",
        "failure_cause",
        "failure_symptom",
        "target_option",
        "observable_error_policy",
        "misconception-token",
    ):
        assert forbidden not in serialized
    prompts.assert_blind_no_leakage(messages, persona=_row(), item=_item(), frozen_leakage_lexicon=("misconception-token",))


def test_blind_and_judge_scanners_catch_adversarial_fields_values_and_prose():
    prompts = _import("experiments.llm_sim_v2.prompts")

    adversarial = [{"role": "user", "content": '{"target_option":"B", "hint":"confuses conjugate pairs"}'}]
    with pytest.raises(AssertionError, match="target_option|failure|leak"):
        prompts.assert_blind_no_leakage(
            adversarial,
            persona=_row(),
            item=_item(),
            frozen_leakage_lexicon=("confuses conjugate pairs",),
        )

    judge = prompts.render_judge_export(
        blind_messages=prompts.render_blind_prompt(_row(), _item()),
        model_output={"label": "unknown", "evidence": "insufficient_evidence"},
        persona=_row(),
        item=_item(),
        frozen_leakage_lexicon=(),
    )
    prompts.assert_judge_no_target_labels(judge, persona=_row(), item=_item())
    judge_text = json.dumps(judge, ensure_ascii=False)
    assert "unknown" in judge_text
    assert "insufficient_evidence" in judge_text
    with pytest.raises(AssertionError, match="target"):
        prompts.assert_judge_no_target_labels(
            [{"role": "user", "content": "target_option B"}],
            persona=_row(),
            item=_item(),
        )


def test_public_question_and_option_text_are_allowed_inside_nested_judge_export():
    prompts = _import("experiments.llm_sim_v2.prompts")
    item = _item()
    item["public_question"] = "Explain the failure_id term in this public question"
    item["options"]["B"] = "confuses conjugate pairs"
    blind = prompts.render_blind_prompt(_row(), item, frozen_leakage_lexicon=("confuses conjugate pairs",))
    judge = prompts.render_judge_export(
        blind_messages=blind,
        model_output={"label": "unknown", "evidence": "public text only"},
        persona=_row(),
        item=item,
        frozen_leakage_lexicon=("confuses conjugate pairs",),
    )
    prompts.assert_judge_no_target_labels(
        judge,
        persona=_row(),
        item=item,
        frozen_leakage_lexicon=("confuses conjugate pairs",),
    )


def test_same_public_phrase_is_still_rejected_when_repeated_outside_public_payload():
    prompts = _import("experiments.llm_sim_v2.prompts")
    item = _item()
    item["options"]["B"] = "confuses conjugate pairs"
    adversarial = [
        {
            "role": "user",
            "target_option": "B",
            "content": json.dumps(
                {
                    "public_question": item["public_question"],
                    "options": item["options"],
                    "hidden_hint": "confuses conjugate pairs",
                }
            ),
        }
    ]
    with pytest.raises(AssertionError, match="target_option|confuses conjugate pairs"):
        prompts.assert_blind_no_leakage(
            adversarial,
            persona=_row(),
            item=item,
            frozen_leakage_lexicon=("confuses conjugate pairs",),
        )


def test_judge_output_validation_accepts_unknown_and_insufficient_evidence():
    prompts = _import("experiments.llm_sim_v2.prompts")

    assert prompts.validate_judge_output({"label": "unknown"})["label"] == "unknown"
    assert prompts.validate_judge_output({"label": "insufficient_evidence"})["label"] == "insufficient_evidence"
    with pytest.raises(ValueError, match="label"):
        prompts.validate_judge_output({"label": "target_failure-00"})
    with pytest.raises((ValueError, AssertionError), match="target|leak"):
        prompts.validate_judge_output({"label": "unknown", "target_option": "B"})


def test_manuscript_qa_reports_blacklist_lines_and_ce_ai_claims_without_rewriting(tmp_path: Path):
    qa = _import("experiments.llm_sim_v2.manuscript_qa")
    text = (
        "The study used 600 learners and simulated a real student distribution.\n"
        "It is the first-ever four-state persona learning-trajectory simulation.\n"
        "Teacher gold validation established a human-validated result.\n"
        "C&E:AI is an SCIE/SSCI Q1 journal with impact factor 23.4.\n"
    )
    path = tmp_path / "manuscript.md"
    path.write_text(text, encoding="utf-8")

    findings = qa.scan_manuscript(path)
    terms = {finding.term for finding in findings}
    assert any("600" in term for term in terms)
    assert any("real student" in term.lower() for term in terms)
    assert any("teacher gold" in term.lower() for term in terms)
    assert any("learning-trajectory" in term.lower() for term in terms)
    assert any("first-ever" in term.lower() for term in terms)
    assert any("SCIE" in term or "SSCI" in term for term in terms)
    assert all(finding.path.endswith("manuscript.md") for finding in findings)
    assert all(finding.line >= 1 for finding in findings)

    clean = qa.scan_manuscript_text(
        "This is a simulated response-channel stress test with 50 persona clusters and text-only inputs."
    )
    assert clean == []


@pytest.mark.parametrize(
    "text",
    [
        "These personas are representative of the actual student population.",
        "This is a simulated learning trajectories study.",
        "The labels were treated as human ground truth.",
        "This is the first-of-a-kind method.",
        "Computers & Education: Artificial Intelligence is a Q1 journal.",
    ],
)
def test_manuscript_qa_flags_sensible_lexical_variants(text: str):
    qa = _import("experiments.llm_sim_v2.manuscript_qa")
    assert qa.scan_manuscript_text(text)


def test_manuscript_qa_accepts_long_in_memory_text_without_treating_it_as_a_path():
    qa = _import("experiments.llm_sim_v2.manuscript_qa")
    text = "simulated response-channel stress test. " * 500
    assert qa.scan_manuscript(text) == []
