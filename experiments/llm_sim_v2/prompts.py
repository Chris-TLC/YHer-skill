"""Controlled/blind prompt renderers and offline leakage assertions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .keys import canonical_key
from .public import public_question_payload


JUDGE_LABELS = frozenset({"consistent", "inconsistent", "unknown", "insufficient_evidence"})
_JUDGE_OUTPUT_FIELDS = frozenset({"label", "error_category", "rationale", "simulated"})
_JUDGE_AUTHENTICITY_FIELDS = frozenset(
    {"authenticity", "authenticity_score", "truthfulness", "truthfulness_score", "realism", "realism_score"}
)
_FORBIDDEN_FIELD_NAMES = (
    "persona_id",
    "pair_id",
    "row_id",
    "anchor_id",
    "target_node",
    "deficit_condition",
    "seed",
    "failure_id",
    "failure_cause",
    "failure_symptom",
    "target_option",
    "target_label",
    "target_failure",
    "observable_error_policy",
    "misconception_id",
)
_PUBLIC_KEY = "public_question"
_NO_OBSERVATION = object()


def _normalize_field_token(value: Any) -> str:
    return canonical_key(value)


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


def _json_message(role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "simulated": True,
        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def render_controlled_prompt(persona: Any, item: Any) -> list[dict[str, Any]]:
    row = _dict(persona)
    public_question = public_question_payload(item)
    payload = {
        "simulated": True,
        "actor": "simulated",
        "condition": "controlled",
        "curriculum_exposure": row.get("curriculum_exposure", []),
        "local_skill_vector": row.get("local_skill_vector", {}),
        "observable_error_policy": row.get("observable_error_policy", {}),
        "noise_parameters": row.get("noise_parameters", {}),
        "deficit_condition": row.get("deficit_condition"),
        "public_question": public_question,
        "output_schema": {
            "simulated": "boolean",
            "answer": "option letter or null",
            "rationale": "short string",
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
    public_question = public_question_payload(item)
    return {
        "simulated": True,
        "actor": "simulated",
        "condition": "blind",
        "curriculum_exposure": row.get("curriculum_exposure", []),
        "local_skill_vector": row.get("local_skill_vector", {}),
        "noise_parameters": row.get("noise_parameters", {}),
        "modality_condition": "text_only",
        "public_question": public_question,
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


def _decode_nested_json(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 20:
        return value
    if isinstance(value, Mapping):
        return {key: _decode_nested_json(item, _depth=_depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nested_json(item, _depth=_depth + 1) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[", '"')):
            try:
                decoded = json.loads(stripped)
                if decoded != value:
                    return _decode_nested_json(decoded, _depth=_depth + 1)
            except json.JSONDecodeError:
                return value
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _public_payload_matches(value: Any, expected: Mapping[str, Any] | None) -> bool:
    if expected is None or not isinstance(value, Mapping):
        return False
    try:
        return _canonical_json(value) == _canonical_json(expected)
    except (TypeError, ValueError):
        return False


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [leaf for child in value.values() for leaf in _string_leaves(child)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [leaf for child in value for leaf in _string_leaves(child)]
    if isinstance(value, str) and value.strip():
        return [value.strip().lower()]
    return []


def _scan_structure(
    value: Any,
    *,
    expected_public: Mapping[str, Any] | None,
    expected_observed: Any = _NO_OBSERVATION,
    context_terms: Sequence[str],
    violations: list[str],
    candidate_paths: list[str],
    path: str = "$",
    allow_observed_terms: bool = False,
) -> None:
    """Scan all rendered structure except one exact public-question subtree."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            key_normalized = _normalize_field_token(key_text)
            child_path = f"{path}.{key_text}"
            if key_normalized in _FORBIDDEN_FIELD_NAMES:
                violations.append(f"forbidden field: {key_normalized}")
            if key_normalized == _PUBLIC_KEY and key_text == _PUBLIC_KEY and _public_payload_matches(child, expected_public):
                continue
            candidate_matches = False
            if key_normalized == "candidate_output":
                if key_text != "candidate_output":
                    violations.append("candidate_output must use its canonical field name")
                if expected_observed is _NO_OBSERVATION:
                    violations.append("forbidden field: candidate_output")
                else:
                    candidate_paths.append(child_path)
                    try:
                        candidate_matches = _canonical_json(child) == _canonical_json(expected_observed)
                    except (TypeError, ValueError):
                        candidate_matches = False
                    if not candidate_matches:
                        violations.append("candidate output differs from supplied observation")
            _scan_structure(
                child,
                expected_public=expected_public,
                expected_observed=expected_observed,
                context_terms=context_terms,
                violations=violations,
                candidate_paths=candidate_paths,
                path=child_path,
                allow_observed_terms=allow_observed_terms or candidate_matches,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_structure(
                child,
                expected_public=expected_public,
                expected_observed=expected_observed,
                context_terms=context_terms,
                violations=violations,
                candidate_paths=candidate_paths,
                path=f"{path}[{index}]",
                allow_observed_terms=allow_observed_terms,
            )
        return
    if not isinstance(value, str) or allow_observed_terms:
        return
    lowered = value.lower()
    normalized_text = _normalize_field_token(lowered)
    for field_name in _FORBIDDEN_FIELD_NAMES:
        if re.search(rf"(?:^|_){re.escape(field_name)}(?:_|$)", normalized_text):
            violations.append(f"forbidden field text: {field_name}")
    for term in context_terms:
        if term and term in lowered:
            violations.append(f"forbidden leakage term: {term}")


def scan_blind_leakage(
    messages: Any,
    *,
    persona: Any | None = None,
    item: Any | None = None,
    frozen_leakage_lexicon: Sequence[str] = (),
    observed_output: Any = _NO_OBSERVATION,
) -> list[str]:
    parsed = [
        _decode_nested_json(_message_json(message))
        for message in (
            messages
            if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes))
            else [messages]
        )
    ]
    violations: list[str] = []
    context_values: list[str] = []
    for source in (persona, item):
        if source is None:
            continue
        mapping = _dict(source)
        for key in ("failure_id", "failure_cause", "failure_symptom", "target_label"):
            value = mapping.get(key)
            if value is not None:
                context_values.append(str(value).strip().lower())
        context_values.extend(_string_leaves(mapping.get("observable_error_policy")))
    expected_public = None
    if item is not None:
        expected_public = public_question_payload(item)
    expected_observed = (
        _NO_OBSERVATION if observed_output is _NO_OBSERVATION else _decode_nested_json(observed_output)
    )
    terms = [
        *context_values,
        *(str(value).strip().lower() for value in frozen_leakage_lexicon),
    ]
    candidate_paths: list[str] = []
    for payload in parsed:
        _scan_structure(
            payload,
            expected_public=expected_public,
            expected_observed=expected_observed,
            context_terms=tuple(term for term in terms if term),
            violations=violations,
            candidate_paths=candidate_paths,
        )
    if expected_observed is not _NO_OBSERVATION and len(candidate_paths) != 1:
        violations.append("judge export must contain exactly one candidate_output bound to the supplied observation")
    return sorted(set(violations))


def assert_blind_no_leakage(
    messages: Any,
    *,
    persona: Any | None = None,
    item: Any | None = None,
    frozen_leakage_lexicon: Sequence[str] = (),
    observed_output: Any = _NO_OBSERVATION,
) -> None:
    violations = scan_blind_leakage(
        messages,
        persona=persona,
        item=item,
        frozen_leakage_lexicon=frozen_leakage_lexicon,
        observed_output=observed_output,
    )
    if violations:
        raise AssertionError("blind prompt leakage: " + "; ".join(violations))


def render_judge_export(
    *,
    blind_messages: Any,
    model_output: Any,
    persona: Any,
    item: Any,
    frozen_leakage_lexicon: Sequence[str],
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
    assert_judge_no_target_labels(
        export,
        persona=persona,
        item=item,
        frozen_leakage_lexicon=frozen_leakage_lexicon,
        observed_output=model_output,
    )
    return export


def assert_judge_no_target_labels(
    messages: Any,
    *,
    persona: Any | None = None,
    item: Any | None = None,
    frozen_leakage_lexicon: Sequence[str] = (),
    observed_output: Any = _NO_OBSERVATION,
) -> None:
    assert_blind_no_leakage(
        messages,
        persona=persona,
        item=item,
        frozen_leakage_lexicon=frozen_leakage_lexicon,
        observed_output=observed_output,
    )


def validate_judge_output(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("judge output must be an object")
    label = str(value.get("label") or "").strip().lower()
    if label not in JUDGE_LABELS:
        raise ValueError("judge output label is not allowed")
    for key in value:
        normalized = canonical_key(key)
        if normalized in _FORBIDDEN_FIELD_NAMES:
            raise ValueError(f"judge output contains target-label leakage field: {normalized}")
        if normalized in _JUDGE_AUTHENTICITY_FIELDS:
            raise ValueError(f"judge output contains forbidden authenticity/truthfulness field: {normalized}")
        if str(key) != normalized or normalized not in _JUDGE_OUTPUT_FIELDS:
            raise ValueError("judge output permits only agreement and error-category fields")
    output = dict(value)
    if "error_category" in output:
        category = output["error_category"]
        if not isinstance(category, str) or not category.strip():
            raise ValueError("judge error_category must be a non-empty string")
        output["error_category"] = category.strip()
    if "rationale" in output and not isinstance(output["rationale"], str):
        raise ValueError("judge rationale must be a string")
    if "simulated" in output and output["simulated"] is not True:
        raise ValueError("judge output simulated flag must be true")
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
