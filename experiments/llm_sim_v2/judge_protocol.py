"""Outcome-blind, post-collection adjudication protocol for Persona v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .keys import canonical_key
from .public import public_question_payload


JUDGE_LABELS = frozenset(
    {"consistent", "inconsistent", "unknown", "insufficient_evidence"}
)
JUDGE_PUBLIC_SCHEMA_KEYS = frozenset(
    {"kind", "options", "stem_blocks", "stem_text"}
)
_PROTOCOL_PATH = Path(__file__).with_name("judge_protocol_v1.json")
_OUTPUT_FIELDS = frozenset(
    {"error_category", "label", "rationale", "simulated"}
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer_values",
        "anchor_id",
        "authenticity",
        "authenticity_score",
        "correct_option",
        "deficit_condition",
        "failure_cause",
        "failure_id",
        "failure_symptom",
        "mapping_status",
        "misconception_id",
        "observable_error_policy",
        "pair_id",
        "persona_id",
        "provider",
        "realism",
        "realism_score",
        "row_id",
        "target_label",
        "target_node",
        "target_option",
        "truthfulness",
        "truthfulness_score",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _private_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = canonical_key(key)
            if normalized in _FORBIDDEN_KEYS:
                return normalized
            found = _private_key(child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _private_key(child)
            if found:
                return found
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[", '"')):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = value
            if decoded != value:
                return _private_key(decoded)
    return None


def judge_protocol() -> dict[str, Any]:
    """Load and validate the sole committed rubric used by both judges."""

    value = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != "yher.llm_sim_v2.judge_protocol.v1"
        or value.get("simulated") is not True
        or set(value.get("label_definitions") or {}) != JUDGE_LABELS
        or set(value.get("required_output_fields") or ()) != _OUTPUT_FIELDS
        or set(value.get("label_category_policy") or {}) != JUDGE_LABELS
        or set(value.get("question_field_whitelist") or ())
        != JUDGE_PUBLIC_SCHEMA_KEYS
    ):
        raise ValueError("committed judge protocol is invalid")
    categories = value.get("error_categories")
    policies = value.get("label_category_policy")
    if (
        not isinstance(categories, Mapping)
        or not all(isinstance(text, str) and text.strip() for text in categories.values())
        or not isinstance(policies, Mapping)
        or any(
            not isinstance(allowed, list)
            or not allowed
            or not set(allowed).issubset(categories)
            for allowed in policies.values()
        )
    ):
        raise ValueError("committed judge protocol taxonomy is invalid")
    return deepcopy(value)


def judge_public_question_payload(item: Any) -> dict[str, Any]:
    """Project a collection question onto the stricter judge-only whitelist."""

    public = public_question_payload(item)
    payload = {
        key: deepcopy(public[key])
        for key in sorted(JUDGE_PUBLIC_SCHEMA_KEYS)
        if key in public
    }
    if not isinstance(payload.get("options"), Mapping):
        raise ValueError("judge public question requires an option mapping")
    private_key = _private_key(payload)
    if private_key:
        raise ValueError(f"judge public question contains private field {private_key}")
    return payload


def render_judge_export(
    *,
    public_question: Mapping[str, Any],
    model_output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Render one opaque judge case from exact sanitized public inputs."""

    question = judge_public_question_payload(
        {"public_question": dict(public_question)}
    )
    output = deepcopy(dict(model_output))
    private_key = _private_key(output)
    if private_key:
        raise ValueError(f"candidate output contains private field {private_key}")
    protocol = judge_protocol()
    system = {
        "simulated": True,
        "actor": "simulated",
        "instruction": (
            "Apply the frozen rubric and return exactly one strict JSON object "
            "with every required output field and no markdown."
        ),
        "judge_protocol": protocol,
    }
    user = {
        "simulated": True,
        "actor": "simulated",
        "public_question": question,
        "candidate_output": output,
        "judge_protocol": protocol,
    }
    return [
        {
            "role": "system",
            "simulated": True,
            "content": _canonical_json(system),
        },
        {
            "role": "user",
            "simulated": True,
            "content": _canonical_json(user),
        },
    ]


def validate_judge_output(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one rating against the frozen label/category rubric."""

    if not isinstance(value, Mapping):
        raise ValueError("judge output must be an object")
    private_key = _private_key(value)
    if private_key:
        raise ValueError(
            f"judge output contains target-label leakage field: {private_key}"
        )
    if any(str(key) != canonical_key(key) for key in value):
        raise ValueError("judge output fields must use canonical names")
    if set(value) != _OUTPUT_FIELDS:
        raise ValueError("judge output must contain every required field exactly")
    label = value.get("label")
    if not isinstance(label, str) or label.strip().lower() not in JUDGE_LABELS:
        raise ValueError("judge output label is not allowed")
    normalized_label = label.strip().lower()
    category = value.get("error_category")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("judge error_category must be a non-empty string")
    normalized_category = category.strip()
    protocol = judge_protocol()
    if normalized_category not in protocol["label_category_policy"][normalized_label]:
        raise ValueError(
            "judge error category is incompatible with the selected label"
        )
    rationale = value.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("judge rationale must be a non-empty string")
    if value.get("simulated") is not True:
        raise ValueError("judge output simulated flag must be true")
    return {
        "label": normalized_label,
        "error_category": normalized_category,
        "rationale": rationale.strip(),
        "simulated": True,
    }


def normalized_target_terms(values: Sequence[Any]) -> tuple[str, ...]:
    """Normalize non-empty private target labels for exact per-case scanning."""

    return tuple(
        sorted(
            {
                normalized
                for value in values
                if value is not None
                and (normalized := str(value).strip().casefold())
            }
        )
    )


def target_label_hits(value: Any, terms: Sequence[str]) -> list[str]:
    """Return exact normalized target terms occurring in serialized case bytes."""

    serialized = _canonical_json(value).casefold()
    return sorted({term for term in terms if term and term in serialized})


__all__ = [
    "JUDGE_LABELS",
    "JUDGE_PUBLIC_SCHEMA_KEYS",
    "judge_protocol",
    "judge_public_question_payload",
    "normalized_target_terms",
    "render_judge_export",
    "target_label_hits",
    "validate_judge_output",
]
