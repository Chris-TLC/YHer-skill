"""Whitelist-preserving public question extraction for real catalog records."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


_PRIVATE_PUBLIC_KEYS = frozenset(
    {
        "answer_values",
        "authoritative_solution",
        "candidate_output",
        "correct_option",
        "failure_cause",
        "failure_id",
        "failure_symptom",
        "misconception_id",
        "observable_error_policy",
        "private_correct_option",
        "provider",
        "model",
        "model_id",
        "response",
        "outcome",
        "run_id",
        "rubric",
        "secret",
        "secret_token",
        "solution_steps",
        "target_failure",
        "target_label",
        "target",
        "target_node",
        "target_option",
        "candidate",
        "api_key",
        "apikey",
        "credentials",
    }
)
_PUBLIC_SCHEMA_KEYS = frozenset(
    {"kind", "stem_blocks", "stem_text", "options", "difficulty", "nodes", "source_label"}
)


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _private_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _PRIVATE_PUBLIC_KEYS:
                return normalized
            found = _private_key(child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _private_key(child)
            if found:
                return found
    return None


def public_question_payload(item: Any) -> dict[str, Any]:
    """Return the exact student-visible question structure without private data.

    Real catalog records expose ``public_question`` as a method.  Mapping
    fixtures may expose the same value directly.  The callable path is always
    invoked before copying, so a bound-method repr can never enter a prompt or
    review payload.
    """

    candidate = _value(item, "public_question", None)
    if callable(candidate):
        candidate = candidate()
    if isinstance(candidate, str):
        candidate = {
            "kind": _value(item, "item_type", _value(item, "scoring_mode", "mcq")),
            "stem_blocks": _value(item, "stem_blocks", ()),
            "stem_text": candidate,
            "options": _value(item, "options", {}),
            "difficulty": _value(item, "difficulty", 0.5),
            "nodes": _value(item, "node_ids", ()),
            "source_label": _value(item, "source_label", ""),
        }
    if candidate is None:
        candidate = {
            "kind": _value(item, "item_type", _value(item, "scoring_mode", "mcq")),
            "stem_blocks": _value(item, "stem_blocks", ()),
            "stem_text": _value(item, "stem_text", ""),
            "options": _value(item, "options", {}),
            "difficulty": _value(item, "difficulty", 0.5),
            "nodes": _value(item, "node_ids", ()),
            "source_label": _value(item, "source_label", ""),
        }
    if not isinstance(candidate, Mapping):
        raise ValueError("public_question must return a mapping")
    payload = deepcopy(dict(candidate))
    unknown_keys = {
        str(key).strip().lower().replace("-", "_")
        for key in payload
        if str(key).strip().lower().replace("-", "_") not in _PUBLIC_SCHEMA_KEYS
    }
    if unknown_keys:
        raise ValueError(f"public_question contains fields outside the public schema: {sorted(unknown_keys)}")
    if "options" not in payload:
        payload["options"] = deepcopy(_value(item, "options", {}))
    if not isinstance(payload.get("options"), Mapping):
        raise ValueError("public_question.options must be a mapping")
    private_key = _private_key(payload)
    if private_key:
        raise ValueError(f"public_question contains private field {private_key}")
    return payload


def public_question(item: Any) -> dict[str, Any]:
    return public_question_payload(item)


__all__ = ["public_question_payload", "public_question"]
