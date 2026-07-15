"""Controlled/blind prompt renderers and offline leakage assertions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


JUDGE_LABELS = frozenset({"consistent", "inconsistent", "unknown", "insufficient_evidence"})
_FORBIDDEN_FIELD_NAMES = (
    "failure_id",
    "failure_cause",
    "failure_symptom",
    "target_option",
    "target_label",
    "target_failure",
    "observable_error_policy",
    "misconception_id",
)
_PUBLIC_KEYS = {"public_question", "question", "options", "public_text"}


def _dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, Mapping):
        return dict(value)
    fields = (
        "persona_id",
        "pair_id",
        "row_id",
        "curriculum_exposure",
        "deficit_condition",
        "local_skill_vector",
        "observable_error_policy",
        "noise_parameters",
        "modality_condition",
        "failure_id",
        "failure_cause",
        "failure_symptom",
        "target_label",
        "target_option",
        "public_question",
        "question",
        "stem_text",
        "options",
    )
    output = {field: getattr(value, field) for field in fields if hasattr(value, field)}
    if output:
        return output
    raise TypeError("prompt input must be a mapping or PersonaV2 record")


def _question(item: Any) -> str:
    value = _dict(item)
    return str(value.get("public_question", value.get("question", value.get("stem_text", ""))))


def _options(item: Any) -> dict[str, str]:
    raw = _dict(item).get("options", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("public item options must be a mapping")
    return {str(key): str(value) for key, value in raw.items()}


def _json_message(role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "simulated": True,
        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def render_controlled_prompt(persona: Any, item: Any) -> list[dict[str, Any]]:
    row = _dict(persona)
    payload = {
        "simulated": True,
        "actor": "simulated",
        "condition": "controlled",
        "curriculum_exposure": row.get("curriculum_exposure", []),
        "local_skill_vector": row.get("local_skill_vector", {}),
        "observable_error_policy": row.get("observable_error_policy", {}),
        "noise_parameters": row.get("noise_parameters", {}),
        "deficit_condition": row.get("deficit_condition"),
        "public_question": _question(item),
        "options": _options(item),
        "output_schema": {
            "simulated": "boolean",
            "answer": "option letter or null",
            "rationale": "short string",
            "manipulation_compliance": "number from 0 to 1",
        },
    }
    system = {
        "simulated": True,
        "actor": "simulated",
        "instruction": "Return exactly one strict JSON object and no markdown.",
        "metric": "manipulation_compliance",
    }
    return [_json_message("system", system), _json_message("user", payload)]


def _blind_payload(persona: Any, item: Any) -> dict[str, Any]:
    row = _dict(persona)
    return {
        "simulated": True,
        "actor": "simulated",
        "condition": "blind",
        "curriculum_exposure": row.get("curriculum_exposure", []),
        "local_skill_vector": row.get("local_skill_vector", {}),
        "noise_parameters": row.get("noise_parameters", {}),
        "modality_condition": "text_only",
        "public_question": _question(item),
        "options": _options(item),
        "output_schema": {
            "simulated": "boolean",
            "answer": "option letter or null",
            "rationale": "short string",
            "abstain": "boolean",
        },
    }


def render_blind_prompt(
    persona: Any,
    item: Any,
    *,
    frozen_leakage_lexicon: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Render only public question/options and general response context."""

    payload = _blind_payload(persona, item)
    system = {
        "simulated": True,
        "actor": "simulated",
        "instruction": "Return exactly one strict JSON object and no markdown.",
        "metric": "response_robustness",
    }
    messages = [_json_message("system", system), _json_message("user", payload)]
    assert_blind_no_leakage(
        messages,
        persona=persona,
        item=item,
        frozen_leakage_lexicon=frozen_leakage_lexicon,
    )
    return messages


def _without_public_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_public_payload(value_item)
            for key, value_item in value.items()
            if str(key).lower() not in _PUBLIC_KEYS
        }
    if isinstance(value, list):
        return [_without_public_payload(item) for item in value]
    return value


def _message_json(message: Any) -> Any:
    if isinstance(message, Mapping):
        output = dict(message)
        content = message.get("content")
        if isinstance(content, str):
            try:
                output["content"] = json.loads(content)
            except json.JSONDecodeError:
                output["content"] = content
        return output
    return message


def _decode_nested_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _decode_nested_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nested_json(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _decode_nested_json(json.loads(stripped))
            except json.JSONDecodeError:
                return value
    return value


def scan_blind_leakage(
    messages: Any,
    *,
    persona: Any | None = None,
    item: Any | None = None,
    frozen_leakage_lexicon: Sequence[str] = (),
) -> list[str]:
    parsed = [
        _decode_nested_json(_message_json(message))
        for message in (
            messages
            if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes))
            else [messages]
        )
    ]
    outside = json.dumps([_without_public_payload(payload) for payload in parsed], ensure_ascii=False, sort_keys=True).lower()
    violations: list[str] = []
    for field_name in _FORBIDDEN_FIELD_NAMES:
        if re.search(rf"(?<![a-z0-9]){re.escape(field_name)}(?![a-z0-9])", outside):
            violations.append(f"forbidden field: {field_name}")
    context_values: list[str] = []
    for source in (persona, item):
        if source is None:
            continue
        mapping = _dict(source)
        for key in ("failure_id", "failure_cause", "failure_symptom", "target_label"):
            value = mapping.get(key)
            if value is not None:
                context_values.append(str(value).strip().lower())
    for term in [*context_values, *(str(value).strip().lower() for value in frozen_leakage_lexicon)]:
        if term and term in outside:
            violations.append(f"forbidden leakage term: {term}")
    return sorted(set(violations))


def assert_blind_no_leakage(
    messages: Any,
    *,
    persona: Any | None = None,
    item: Any | None = None,
    frozen_leakage_lexicon: Sequence[str] = (),
) -> None:
    violations = scan_blind_leakage(
        messages,
        persona=persona,
        item=item,
        frozen_leakage_lexicon=frozen_leakage_lexicon,
    )
    if violations:
        raise AssertionError("blind prompt leakage: " + "; ".join(violations))


def render_judge_export(
    *,
    blind_messages: Any,
    model_output: Any,
) -> list[dict[str, Any]]:
    payload = {
        "simulated": True,
        "actor": "simulated",
        "blind_messages": blind_messages,
        "candidate_output": model_output,
        "allowed_labels": sorted(JUDGE_LABELS),
    }
    export = [
        _json_message(
            "system",
            {
                "simulated": True,
                "actor": "simulated",
                "instruction": "Return exactly one strict JSON object with one allowed label.",
                "allowed_labels": sorted(JUDGE_LABELS),
            },
        ),
        _json_message("user", payload),
    ]
    assert_judge_no_target_labels(export)
    return export


def assert_judge_no_target_labels(
    messages: Any,
    *,
    persona: Any | None = None,
    item: Any | None = None,
    frozen_leakage_lexicon: Sequence[str] = (),
) -> None:
    assert_blind_no_leakage(
        messages,
        persona=persona,
        item=item,
        frozen_leakage_lexicon=frozen_leakage_lexicon,
    )


def validate_judge_output(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("judge output must be an object")
    label = str(value.get("label") or "").strip().lower()
    if label not in JUDGE_LABELS:
        raise ValueError("judge output label is not allowed")
    output = dict(value)
    try:
        assert_judge_no_target_labels([_json_message("assistant", output)])
    except AssertionError as exc:
        raise ValueError("judge output contains target-label leakage") from exc
    output["label"] = label
    output["simulated"] = True
    return output


render_controlled = render_controlled_prompt
render_blind = render_blind_prompt
render_judge = render_judge_export
assert_no_blind_leakage = assert_blind_no_leakage
assert_no_judge_leakage = assert_judge_no_target_labels


__all__ = [
    "JUDGE_LABELS",
    "render_controlled_prompt",
    "render_blind_prompt",
    "render_judge_export",
    "scan_blind_leakage",
    "assert_blind_no_leakage",
    "assert_judge_no_target_labels",
    "validate_judge_output",
]
