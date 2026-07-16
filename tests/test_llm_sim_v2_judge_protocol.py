from __future__ import annotations

import pytest


def test_judge_protocol_defines_all_labels_and_fixed_error_categories() -> None:
    from experiments.llm_sim_v2 import judge_protocol

    protocol = judge_protocol.judge_protocol()
    assert set(protocol["label_definitions"]) == {
        "consistent",
        "inconsistent",
        "unknown",
        "insufficient_evidence",
    }
    assert all(protocol["label_definitions"].values())
    assert set(protocol["error_categories"]) == {
        "none",
        "answer_selection",
        "chemistry_reasoning",
        "internal_contradiction",
        "question_interpretation",
        "unsupported_abstention",
        "format_or_schema",
        "other",
        "not_applicable",
    }
    assert protocol["required_output_fields"] == [
        "error_category",
        "label",
        "rationale",
        "simulated",
    ]


@pytest.mark.parametrize(
    ("label", "error_category"),
    [
        ("consistent", "none"),
        ("inconsistent", "chemistry_reasoning"),
        ("unknown", "not_applicable"),
        ("insufficient_evidence", "not_applicable"),
    ],
)
def test_judge_output_validation_requires_rubric_complete_output(
    label: str, error_category: str
) -> None:
    from experiments.llm_sim_v2 import judge_protocol

    valid = {
        "label": label,
        "error_category": error_category,
        "rationale": "Brief evidence-based explanation.",
        "simulated": True,
    }
    assert judge_protocol.validate_judge_output(valid) == valid
    for missing in valid:
        incomplete = dict(valid)
        incomplete.pop(missing)
        with pytest.raises(ValueError, match="required|field|simulated"):
            judge_protocol.validate_judge_output(incomplete)

    incompatible = "none" if label == "inconsistent" else "chemistry_reasoning"
    with pytest.raises(ValueError, match="category|consistent"):
        judge_protocol.validate_judge_output(
            valid | {"error_category": incompatible}
        )
    with pytest.raises(ValueError, match="label"):
        judge_protocol.validate_judge_output(
            valid | {"label": "target_failure-00"}
        )
    with pytest.raises((ValueError, AssertionError), match="target|leak|field"):
        judge_protocol.validate_judge_output(valid | {"target_option": "B"})
