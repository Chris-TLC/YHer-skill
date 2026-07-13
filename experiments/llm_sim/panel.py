"""Pre-outcome manipulation panel construction and integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import Persona, StudyPanel


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def normalize_annotation_map(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an explicit item/failure/option map without text inference.

    Accepted JSON shapes are either ``{"items": {item_id: {failure_id:
    option}}}`` or ``{"annotations": [{"item_id": ..., "failure_id": ...,
    "target_option": ...}]}``.  A bare item mapping is also accepted for
    compatibility with the catalog-level ``distractor_map`` contract.
    """

    if not isinstance(value, Mapping):
        raise ValueError("annotation map must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version not in (None, "yher.llm_sim.annotation_map.v1"):
        raise ValueError("unsupported annotation map schema_version")
    if "items" in value and "annotations" in value:
        raise ValueError("annotation map must use either items or annotations")
    normalized_items: dict[str, dict[str, str]] = {}
    if "annotations" in value:
        rows = value["annotations"]
        if not isinstance(rows, list):
            raise ValueError("annotation map annotations must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("annotation map row must be an object")
            _add_annotation(
                normalized_items,
                item_id=row.get("item_id"),
                failure_id=row.get("failure_id"),
                option=row.get("target_option") or row.get("option"),
            )
    else:
        raw_items = value.get("items", value)
        if not isinstance(raw_items, Mapping):
            raise ValueError("annotation map items must be an object")
        for item_id, raw_failures in raw_items.items():
            if not isinstance(raw_failures, Mapping):
                raise ValueError("each annotation-map item must map failure ids to options")
            for failure_id, raw_option in raw_failures.items():
                option = raw_option
                if isinstance(raw_option, Mapping):
                    option = raw_option.get("target_option") or raw_option.get("option")
                _add_annotation(
                    normalized_items,
                    item_id=item_id,
                    failure_id=failure_id,
                    option=option,
                )
    return {
        "schema_version": "yher.llm_sim.annotation_map.v1",
        "items": {
            item_id: dict(sorted(failures.items()))
            for item_id, failures in sorted(normalized_items.items())
        },
    }


def _add_annotation(
    output: dict[str, dict[str, str]],
    *,
    item_id: Any,
    failure_id: Any,
    option: Any,
) -> None:
    item = str(item_id or "").strip()
    failure = str(failure_id or "").strip()
    target = str(option or "").strip().upper()
    if not item or not failure or target not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise ValueError(
            "annotation map entries require item_id, failure_id, and one-letter target_option"
        )
    existing = output.setdefault(item, {}).get(failure)
    if existing is not None and existing != target:
        raise ValueError("annotation map contains a conflicting duplicate entry")
    output[item][failure] = target


def annotation_map_hash(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return _sha(normalize_annotation_map(value))


def load_annotation_map(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("annotation map must be a JSON object")
    return normalize_annotation_map(raw)


def _persona(value: Persona | Mapping[str, Any], fallback_failure_id: str | None) -> Persona:
    if isinstance(value, Persona):
        return value
    normalized = dict(value)
    if not normalized.get("failure_id") and fallback_failure_id:
        normalized["failure_id"] = fallback_failure_id
    return Persona.from_mapping(normalized)


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _answer_key(item: Any) -> str | None:
    values = _item_value(item, "answer_values", ()) or ()
    if not values:
        return None
    return str(values[0]).strip().upper()


def _explicit_mapping(
    item: Any,
    failure_id: str,
    catalog: Any,
    annotation_map: Mapping[str, Any] | None,
) -> str | None:
    """Read only an explicit machine annotation; never infer from prose."""

    maps: list[Any] = []
    item_id = str(_item_value(item, "item_id", ""))
    if annotation_map is not None:
        map_items = annotation_map.get("items", {})
        if isinstance(map_items, Mapping):
            external_mapping = map_items.get(item_id)
            if external_mapping is not None:
                maps.append(external_mapping)
    for key in ("distractor_map", "manipulation_map", "misconception_map"):
        value = _item_value(item, key, None)
        if value is not None:
            maps.append(value)
    metadata = _item_value(item, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("distractor_map", "manipulation_map", "misconception_map"):
            if key in metadata:
                maps.append(metadata[key])
    catalog_map = _item_value(catalog, "distractor_map", None)
    if isinstance(catalog_map, Mapping):
        item_mapping = catalog_map.get(item_id)
        if item_mapping is not None:
            maps.append(item_mapping)
    candidates: set[str] = set()
    for mapping in maps:
        if not isinstance(mapping, Mapping):
            continue
        candidate = mapping.get(failure_id)
        if isinstance(candidate, Mapping):
            candidate = candidate.get("option") or candidate.get("target_option")
        if candidate is not None:
            candidates.add(str(candidate).strip().upper())
    if len(candidates) > 1:
        raise ValueError(
            f"conflicting explicit target-option annotations for item {item_id}"
        )
    return next(iter(candidates), None)


def _catalog_items(catalog: Any, node: str) -> list[Any]:
    if hasattr(catalog, "for_node"):
        return list(catalog.for_node(node, deterministic_only=True))
    items = _item_value(catalog, "items", {}) or {}
    return [
        item
        for item in (items.values() if isinstance(items, Mapping) else items)
        if node in tuple(_item_value(item, "node_ids", ()) or ())
    ]


def _annotation_for_persona(
    persona: Persona,
    catalog: Any,
    fallback_failure_id: str | None,
    annotation_map: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failure_id = persona.failure_id or fallback_failure_id or ""
    candidates: list[dict[str, Any]] = []
    for item in _catalog_items(catalog, persona.target_node):
        item_id = str(_item_value(item, "item_id", ""))
        family_id = str(_item_value(item, "family_id", ""))
        options = {
            str(k).strip().upper(): str(v)
            for k, v in (_item_value(item, "options", {}) or {}).items()
        }
        answer = _answer_key(item)
        scoring_mode = str(_item_value(item, "scoring_mode", ""))
        if (
            not item_id
            or not family_id
            or scoring_mode != "mcq"
            or not options
            or answer not in options
        ):
            continue
        target_option = _explicit_mapping(
            item,
            failure_id,
            catalog,
            annotation_map,
        )
        wrong_options = sorted(key for key in options if key != answer)
        mapping_valid = target_option in wrong_options
        candidates.append(
            {
                "item_id": item_id,
                "family_id": family_id,
                "answer_key": answer,
                "target_option": target_option if mapping_valid else None,
                "wrong_options": wrong_options,
                "random_wrong_option_baseline": (
                    1.0 / len(wrong_options) if wrong_options else None
                ),
                "mapping_status": (
                    "mapped" if mapping_valid else "excluded_pre_outcome"
                ),
                "mapping_exclusion_reason": (
                    None if mapping_valid else "no_mechanical_target_option_mapping"
                ),
            }
        )
    candidates.sort(key=lambda row: (row["family_id"], row["item_id"]))
    calibration_items = []
    selected_families: set[str] = set()
    for row in candidates:
        if row["family_id"] in selected_families:
            continue
        calibration_items.append(row)
        selected_families.add(row["family_id"])
        if len(calibration_items) == 4:
            break
    calibration_ready = len(calibration_items) == 4
    mapped_items = [
        row for row in calibration_items if row["mapping_status"] == "mapped"
    ]
    mapping_ready = calibration_ready and len(mapped_items) == 4
    if not calibration_ready:
        calibration_reason = "insufficient_family_distinct_calibration_mcq"
    else:
        calibration_reason = None
    if not mapping_ready:
        mapping_reason = (
            calibration_reason or "no_mechanical_target_option_mapping"
        )
    else:
        mapping_reason = None
    first_mapped = mapped_items[0] if mapped_items else None
    if not mapping_ready:
        return {
            "persona_id": persona.persona_id,
            "pair_id": persona.pair_id,
            "strength": persona.strength,
            "target_node": persona.target_node,
            "failure_id": failure_id,
            "mapping_status": "excluded_pre_outcome",
            "exclusion_reason": mapping_reason,
            "calibration_status": (
                "ready" if calibration_ready else "excluded_pre_outcome"
            ),
            "calibration_exclusion_reason": calibration_reason,
            "calibration_items": calibration_items,
            "target_item_id": first_mapped["item_id"] if first_mapped else None,
            "target_option": first_mapped["target_option"] if first_mapped else None,
            "wrong_options": first_mapped["wrong_options"] if first_mapped else [],
            "random_wrong_option_baseline": (
                first_mapped["random_wrong_option_baseline"]
                if first_mapped
                else None
            ),
            "annotation_source": persona.annotation_source,
        }
    first = calibration_items[0]
    return {
        "persona_id": persona.persona_id,
        "pair_id": persona.pair_id,
        "strength": persona.strength,
        "target_node": persona.target_node,
        "failure_id": failure_id,
        "mapping_status": "mapped",
        "exclusion_reason": None,
        "calibration_status": "ready",
        "calibration_exclusion_reason": None,
        "calibration_items": calibration_items,
        "target_item_id": first["item_id"],
        "target_option": first["target_option"],
        "wrong_options": first["wrong_options"],
        "random_wrong_option_baseline": first["random_wrong_option_baseline"],
        "annotation_source": persona.annotation_source,
    }


def derive_manipulation_panel(
    *,
    personas: Sequence[Persona | Mapping[str, Any]],
    catalog: Any,
    annotation_map: Mapping[str, Any] | None = None,
    annotation_map_source: str | Path | None = None,
    study_seed: int,
    failure_id: str | None = None,
) -> dict[str, Any]:
    """Mechanically derive the complete frozen panel without writing it.

    ``mapping_status=excluded_pre_outcome`` is intentionally a first-class
    result.  A missing distractor annotation is never repaired with a textual
    or semantic guess after seeing a provider response.
    """
    normalized = [_persona(value, failure_id) for value in personas]
    if not normalized:
        raise ValueError("at least one persona is required")
    normalized_annotation_map = (
        normalize_annotation_map(annotation_map)
        if annotation_map is not None
        else None
    )
    rows = [
        _annotation_for_persona(
            persona,
            catalog,
            failure_id,
            normalized_annotation_map,
        )
        for persona in sorted(normalized, key=lambda value: value.persona_id)
    ]
    persona_payload = [persona.to_dict() for persona in sorted(normalized, key=lambda value: value.persona_id)]
    core = {
        "simulated": True,
        "persona_id": "llm-sim-study:manipulation-panel",
        "provider": "study_design",
        "model_id": "no-provider-observation",
        "record_type": "llm_sim_manipulation_panel",
        "schema_version": "yher.llm_sim.manipulation_panel.v1",
        "frozen": True,
        "observation_started": False,
        "study_seed": int(study_seed),
        "personas_sha256": _sha(persona_payload),
        "annotation_map_sha256": (
            _sha(normalized_annotation_map)
            if normalized_annotation_map is not None
            else None
        ),
        "annotation_map_source": (
            str(Path(annotation_map_source).expanduser().resolve(strict=False))
            if annotation_map_source is not None
            else None
        ),
        "annotations": rows,
    }
    return {**core, "panel_sha256": _sha(core)}


def freeze_manipulation_panel(
    *,
    personas: Sequence[Persona | Mapping[str, Any]],
    catalog: Any,
    annotation_map: Mapping[str, Any] | None = None,
    annotation_map_source: str | Path | None = None,
    output_path: str | Path,
    study_seed: int,
    failure_id: str | None = None,
) -> dict[str, Any]:
    """Create an immutable panel before any provider observation."""

    path = Path(output_path).expanduser().resolve(strict=False)
    if any(part in {"local_store", "study_logs"} for part in path.parts):
        raise ValueError("manipulation panel cannot be written under local_store")
    record = derive_manipulation_panel(
        personas=personas,
        catalog=catalog,
        annotation_map=annotation_map,
        annotation_map_source=annotation_map_source,
        study_seed=study_seed,
        failure_id=failure_id,
    )
    if path.is_file():
        existing = load_frozen_panel(path)
        if existing == record:
            return existing
        raise FileExistsError(
            "a different manipulation panel is already frozen at this path"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return record


def load_frozen_panel(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    record = json.loads(candidate.read_text(encoding="utf-8"))
    required_envelope = {
        "simulated": True,
        "persona_id": "llm-sim-study:manipulation-panel",
        "provider": "study_design",
        "model_id": "no-provider-observation",
        "record_type": "llm_sim_manipulation_panel",
    }
    if any(record.get(key) != value for key, value in required_envelope.items()):
        raise ValueError("manipulation panel is missing the simulated-data envelope")
    if record.get("frozen") is not True:
        raise ValueError("manipulation panel is not frozen")
    if record.get("observation_started") is not False:
        raise ValueError("manipulation panel cannot be marked observation_started")
    supplied = str(record.get("panel_sha256") or "")
    core = {key: value for key, value in record.items() if key != "panel_sha256"}
    if supplied != _sha(core):
        raise ValueError("manipulation panel hash mismatch")
    if not isinstance(record.get("annotations"), list):
        raise ValueError("manipulation panel annotations must be a list")
    return record


def panel_hash(record: Mapping[str, Any]) -> str:
    core = {key: value for key, value in record.items() if key != "panel_sha256"}
    return _sha(core)
