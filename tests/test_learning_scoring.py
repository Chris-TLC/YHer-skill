"""Server-side scoring contracts for deterministic and LLM answer paths."""

from __future__ import annotations

import math

import pytest

from core.learning.item_catalog import CatalogItem
from core.learning.scoring import parse_scalar_answer, score_item


def _item(*, mode: str, answers=("A",), unit: str | None = None) -> CatalogItem:
    return CatalogItem(
        item_id="private-item",
        family_id="family-1",
        aligned_item_id="v3-meta-only",
        alignment_status="auto_inherited",
        node_ids=("氧化还原反应",),
        stem_blocks=({"para": [{"type": "text", "text": "question"}]},),
        stem_text="question",
        stem_hash="hash",
        stem_normalized="question",
        options={"A": "one", "B": "two"},
        difficulty=0.5,
        item_type="mcq" if mode == "mcq" else "free",
        scoring_mode=mode,
        answer_values=tuple(answers),
        numeric_unit=unit,
        source_label="fixture",
    )


def test_mcq_normalization_is_exact_not_substring_matching():
    item = _item(mode="mcq", answers=("B",))

    assert score_item(item, " b ").correct is True
    assert score_item(item, "选择：B").correct is True
    verbose = score_item(item, "B because it reacts")
    assert verbose.status == "invalid"
    assert verbose.correct is None
    assert verbose.update_allowed is False
    assert score_item(item, "A").correct is False


@pytest.mark.parametrize("submission", ["AA", "A A", "选择：AA"])
def test_repeated_mcq_letters_are_invalid_and_do_not_update_mastery(submission):
    result = score_item(_item(mode="mcq", answers=("A",)), submission)

    assert result.status == "invalid"
    assert result.correct is None
    assert result.update_allowed is False
    assert result.likelihood == (0.25, 0.25, 0.25, 0.25)
    assert result.error_code == "invalid_mcq_answer"


def test_numeric_scoring_requires_one_strict_scalar_and_compatible_unit():
    item = _item(mode="numeric", answers=("3 mol",), unit="mol")

    assert score_item(item, "3.000 mol").correct is True
    assert score_item(item, "3.0001 mol").correct is True
    assert score_item(item, "3 g").correct is False
    assert score_item(item, "3 mol and 4 mol").correct is False
    assert parse_scalar_answer("-1.2e-3 mol/L") is not None
    assert parse_scalar_answer("x=3") is None


def test_free_response_without_llm_is_deferred_and_zero_information():
    result = score_item(_item(mode="free_llm", answers=("long answer",)), "student work")

    assert result.status == "deferred"
    assert result.correct is None
    assert result.update_allowed is False
    assert result.likelihood == (0.25, 0.25, 0.25, 0.25)


def test_llm_likelihood_is_confidence_weighted_and_capped_at_three_to_one():
    def grader(_item, _submission):
        return {
            "correct": False,
            "error_code": "missing_conservation",
            "confidence": 1.0,
            "likelihood": [1000, 1, 1, 1],
        }

    result = score_item(_item(mode="free_llm", answers=("long answer",)), "work", grader)

    assert result.status == "scored"
    assert result.update_allowed is True
    assert result.error_code == "missing_conservation"
    assert math.isclose(sum(result.likelihood), 1.0)
    assert max(result.likelihood) / min(result.likelihood) <= 3.0 + 1e-9


def test_illegal_llm_likelihood_fails_closed_without_profile_update():
    def grader(_item, _submission):
        return {
            "correct": True,
            "error_code": "none",
            "confidence": 1.0,
            "likelihood": [float("nan"), -2, 0, 4],
        }

    result = score_item(_item(mode="free_llm", answers=("long answer",)), "work", grader)

    assert result.status == "deferred"
    assert result.correct is None
    assert result.update_allowed is False
    assert result.likelihood == (0.25, 0.25, 0.25, 0.25)


def test_llm_transport_failure_is_an_offline_deferred_result():
    def grader(_item, _submission):
        raise ConnectionError("offline")

    result = score_item(_item(mode="free_llm", answers=("long answer",)), "work", grader)

    assert result.status == "deferred"
    assert result.update_allowed is False


def test_non_finite_llm_confidence_is_deferred():
    def grader(_item, _submission):
        return {
            "correct": True,
            "confidence": float("nan"),
            "likelihood": [1, 1, 1, 1],
        }

    result = score_item(_item(mode="free_llm", answers=("long answer",)), "work", grader)

    assert result.status == "deferred"
    assert result.update_allowed is False
