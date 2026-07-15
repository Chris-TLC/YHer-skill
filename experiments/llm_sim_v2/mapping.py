"""Normalized, pre-observation target-option mapping contracts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import Any

from .keys import canonical_key


SCHEMA_VERSION = "yher.llm_sim_v2.target_option_map.v1"
STATUSES = {"mapped", "excluded_ambiguous"}
REVIEWER_PROVENANCE_POLICY = (
    "Manual target-option alignment and exclusion reviewer fields cannot use codex_*; "
    "Codex code/test gate self-signing is a separate engineering review scope."
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _item_lookup(catalog: Any = None, items: Any = None) -> dict[str, Any]:
    source = items if items is not None else _value(catalog, "items", None)
    if source is None and isinstance(catalog, Mapping):
        source = catalog.values()
    if source is None and catalog is not None and hasattr(catalog, "all_items"):
        source = catalog.all_items()
    if source is None and isinstance(catalog, Iterable) and not isinstance(catalog, (str, bytes)):
        source = catalog
    if isinstance(source, Mapping):
        return {str(key): value for key, value in source.items()}
    if source is not None:
        output = {}
        for item in source:
            item_id = str(_value(item, "item_id", "") or "")
            if item_id:
                output[item_id] = item
        return output
    return {}


def _expected_pairs(value: Any) -> set[tuple[str, str]]:
    if value is None:
        return set()
    pairs = set()
    for row in value:
        if isinstance(row, Mapping):
            pairs.add((str(row.get("item_id") or ""), str(row.get("failure_id") or "")))
        else:
            item_id, failure_id = row
            pairs.add((str(item_id), str(failure_id)))
    return pairs


def _raw_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "rows" in value:
            value = value["rows"]
        elif "annotations" in value:
            value = value["annotations"]
        elif "items" in value:
            rows = []
            for item_id, failures in value["items"].items():
                for failure_id, option in failures.items():
                    if isinstance(option, Mapping):
                        row = dict(option)
                    else:
                        row = {"target_option": option}
                    row.update({"item_id": item_id, "failure_id": failure_id})
                    rows.append(row)
            value = rows
        else:
            value = [value]
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError("target-option map rows must be a list")
    rows = list(value)
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("target-option map rows must be objects")
    return rows


def _existing_rows(value: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    rows = value.get("rows", value)
    return _normalize_rows(rows)


def _normalize_rows(rows: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in _raw_rows(rows):
        if _contains_codex_reviewer(raw):
            raise ValueError("reviewer provenance cannot use a codex_* manual reviewer")
        item_id = str(raw.get("item_id") or "").strip()
        failure_id = str(raw.get("failure_id") or "").strip()
        if not item_id or not failure_id:
            raise ValueError("mapping rows require item_id and failure_id")
        status = str(raw.get("status") or ("mapped" if raw.get("target_option") else "excluded_ambiguous")).strip().lower()
        if status not in STATUSES:
            raise ValueError("mapping status must be mapped or excluded_ambiguous")
        option = raw.get("target_option")
        target_option = str(option).strip().upper() if option is not None and str(option).strip() else None
        reviewer = raw.get("reviewer_provenance", raw.get("reviewer", raw.get("provenance")))
        if not isinstance(reviewer, Mapping) or not reviewer:
            raise ValueError("mapping rows require non-empty structured reviewer provenance")
        if _contains_codex_reviewer(reviewer):
            raise ValueError("reviewer provenance cannot use a codex_* manual reviewer")
        row: dict[str, Any] = {
            "item_id": item_id,
            "failure_id": failure_id,
            "target_option": target_option,
            "status": status,
            "reviewer_provenance": deepcopy(dict(reviewer)),
        }
        if raw.get("ambiguity_reason") is not None:
            row["ambiguity_reason"] = str(raw["ambiguity_reason"])
        key = (item_id, failure_id)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError("mapping contains a duplicate entry")
        seen[key] = row
    output.extend(seen.values())
    output.sort(key=lambda row: (row["item_id"], row["failure_id"]))
    return output


def _is_signer_role(key: Any) -> bool:
    normalized = canonical_key(key)
    if normalized == "reviewer_provenance":
        return False
    role_tokens = {
        "reviewer",
        "reviewed",
        "signer",
        "signed",
        "crosscheck",
        "crosschecked",
        "crosschecker",
        "approved",
        "approver",
    }
    return bool(role_tokens.intersection(normalized.split("_")))


def _contains_codex_reviewer(value: Any, *, _signer_context: bool = False) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_signer_context = _signer_context or _is_signer_role(key)
            if _contains_codex_reviewer(item, _signer_context=child_signer_context):
                return True
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_codex_reviewer(item, _signer_context=_signer_context) for item in value)
    identity = canonical_key(value)
    return _signer_context and (identity == "codex" or identity.startswith("codex_"))


def _correct_option(item: Any) -> str | None:
    values = _value(item, "answer_values", ()) or ()
    direct = values[0] if values else None
    if direct is None:
        direct = _value(item, "correct_option", None) or _value(item, "answer_key", None)
    return str(direct).strip().upper() if direct is not None else None


def normalize_target_option_map(
    rows: Any,
    *,
    catalog: Any = None,
    items: Any = None,
    expected_rows: Iterable[Any] | None = None,
    existing: Mapping[str, Any] | None = None,
    observation_started: bool = False,
    post_observation: bool = False,
) -> dict[str, Any]:
    """Normalize and validate a target-option map before any observation."""

    existing_started = bool(
        isinstance(existing, Mapping)
        and (
            existing.get("observation_started")
            or existing.get("first_observation_at")
            or existing.get("observation_timestamp")
        )
    )
    mapping_locked = bool(observation_started or post_observation or existing_started)
    if mapping_locked:
        if existing is None:
            raise ValueError("post-observation mapping replacement is forbidden")
    normalized_rows = _normalize_rows(rows)
    if expected_rows is None:
        raise ValueError("a non-empty expected row set is required")
    expected_values = list(expected_rows)
    expected = _expected_pairs(expected_values)
    if not expected:
        raise ValueError("a non-empty expected row set is required")
    if len(expected) != len(expected_values) or any(not item_id or not failure_id for item_id, failure_id in expected):
        raise ValueError("expected rows must be unique non-empty item/failure pairs")
    actual = {(row["item_id"], row["failure_id"]) for row in normalized_rows}
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise ValueError(f"mapping rows are missing: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"mapping rows are unexpected extras: {sorted(unexpected)}")
    lookup = _item_lookup(catalog, items)
    if not lookup:
        raise ValueError("a non-empty catalog/items lookup is required")
    for row in normalized_rows:
        item = lookup.get(row["item_id"])
        if item is None:
            raise ValueError(f"mapping item is missing: {row['item_id']}")
        options = _value(item, "options", {}) or {}
        option_keys = {str(key).strip().upper() for key in options} if isinstance(options, Mapping) else set()
        answer = _correct_option(item)
        if not answer or answer not in option_keys:
            raise ValueError(f"mapping item has no known correct answer: {row['item_id']}")
        if row["status"] == "mapped":
            if row["target_option"] is None:
                raise ValueError("mapped rows require a target option")
            if row["target_option"] == answer:
                raise ValueError("target option cannot be the correct option")
            if row["target_option"] not in option_keys:
                raise ValueError("target option is absent from the item options")
        else:
            if row["target_option"] is not None:
                raise ValueError("excluded_ambiguous rows cannot target an option")
            if not str(row.get("ambiguity_reason") or "").strip():
                raise ValueError("excluded_ambiguous rows require ambiguity_reason")
    if existing is not None:
        if mapping_locked:
            missing_hashes = {
                key for key in ("mapping_sha256", "target_set_hash") if not existing.get(key)
            }
            if missing_hashes:
                raise ValueError(f"locked mapping is missing frozen hashes: {sorted(missing_hashes)}")
        mapping_sha256(existing)
        target_set_hash(existing)
        old_rows = _existing_rows(existing)
        if mapping_locked and old_rows != normalized_rows:
            raise ValueError("post-observation mapping replacement is forbidden")
    mapped_targets = [
        {
            "item_id": row["item_id"],
            "failure_id": row["failure_id"],
            "target_option": row["target_option"],
        }
        for row in normalized_rows
        if row["status"] == "mapped"
    ]
    mapping_hash = hashlib.sha256(_canonical(normalized_rows)).hexdigest()
    target_hash = hashlib.sha256(_canonical(mapped_targets)).hexdigest()
    if isinstance(rows, Mapping):
        advertised_mapping = rows.get("mapping_sha256")
        advertised_targets = rows.get("target_set_hash")
        if advertised_mapping is not None and str(advertised_mapping) != mapping_hash:
            raise ValueError("advertised mapping_sha256 does not match canonical rows")
        if advertised_targets is not None and str(advertised_targets) != target_hash:
            raise ValueError("advertised target_set_hash does not match canonical rows")
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "observation_started": mapping_locked,
        "rows": normalized_rows,
        "mapping_sha256": mapping_hash,
        "target_set_hash": target_hash,
    }


def validate_target_option_map(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return normalize_target_option_map(*args, **kwargs)


def mapping_sha256(value: Mapping[str, Any] | Any) -> str:
    rows = _normalize_rows(value)
    computed = hashlib.sha256(_canonical(rows)).hexdigest()
    if isinstance(value, Mapping) and value.get("mapping_sha256") is not None:
        if str(value["mapping_sha256"]) != computed:
            raise ValueError("advertised mapping_sha256 does not match canonical rows")
    return computed


def target_set_hash(value: Mapping[str, Any] | Any) -> str:
    rows = _normalize_rows(value)
    mapped_targets = [
        {
            "item_id": row["item_id"],
            "failure_id": row["failure_id"],
            "target_option": row["target_option"],
        }
        for row in rows
        if row["status"] == "mapped"
    ]
    computed = hashlib.sha256(_canonical(mapped_targets)).hexdigest()
    if isinstance(value, Mapping) and value.get("target_set_hash") is not None:
        if str(value["target_set_hash"]) != computed:
            raise ValueError("advertised target_set_hash does not match canonical rows")
    return computed


normalize_mapping = normalize_target_option_map
validate_mapping = validate_target_option_map


__all__ = [
    "SCHEMA_VERSION",
    "REVIEWER_PROVENANCE_POLICY",
    "normalize_target_option_map",
    "validate_target_option_map",
    "mapping_sha256",
    "target_set_hash",
]
