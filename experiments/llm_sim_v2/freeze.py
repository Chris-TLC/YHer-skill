"""Pre-observation Persona v2 study configuration contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from .mapping import (
    SCHEMA_VERSION as MAPPING_SCHEMA_VERSION,
    mapping_sha256,
    target_set_hash,
)
from .store import RUN_ID


SCHEMA_VERSION = "yher.llm_sim_v2.study_config.v1"
MAPPING_MINIMUM_FRACTION = 0.60
PROVIDERS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "model": "deepseek-v4-pro",
        "concurrency": 4,
        "timeout_seconds": 90,
        "max_attempts": 4,
        "max_tokens": 1024,
        "retry_max_tokens": 2048,
    },
    "glm": {
        "model": "glm-4-plus",
        "concurrency": 4,
        "timeout_seconds": 60,
        "max_attempts": 3,
        "max_tokens": 512,
        "retry_max_tokens": 1024,
    },
    "kimi": {
        "model": "moonshot-v1-128k",
        "concurrency": 4,
        "timeout_seconds": 60,
        "max_attempts": 3,
        "max_tokens": 512,
        "retry_max_tokens": 1024,
    },
    "minimax": {
        "model": "abab6.5s-chat",
        "concurrency": 4,
        "timeout_seconds": 60,
        "max_attempts": 3,
        "max_tokens": 512,
        "retry_max_tokens": 1024,
    },
    "doubao": {
        "model": "doubao-seed-2-0-mini-260428",
        "concurrency": 2,
        "timeout_seconds": 120,
        "max_attempts": 4,
        "max_tokens": 1024,
        "retry_max_tokens": 2048,
    },
    "tongyi": {
        "model": "qwen-max",
        "concurrency": 4,
        "timeout_seconds": 60,
        "max_attempts": 3,
        "max_tokens": 512,
        "retry_max_tokens": 1024,
    },
}


def _row(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("study rows must be mappings or serializable records")


def _validate_timestamp(value: str) -> str:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("frozen_at_utc must be ISO-8601") from exc
    if not text.endswith("Z") or parsed.tzinfo is None:
        raise ValueError("frozen_at_utc must be UTC")
    return text


def _pilot_personas(persona_ids: Sequence[str]) -> list[str]:
    ranked = sorted(
        persona_ids,
        key=lambda value: (
            hashlib.sha256(f"pilot|20260715|{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    return ranked[:5]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_leakage_lexicon(anchors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze target-specific failure prose while leaving curriculum labels allowed."""

    values: dict[str, str] = {}
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise ValueError("leakage lexicon anchors must be mappings")
        for field in (
            "failure_id",
            "failure_cause",
            "failure_symptom",
            "diagnostic_question",
        ):
            text = str(anchor.get(field) or "").strip()
            if text:
                values.setdefault(text.casefold(), text)
    terms = sorted(values.values(), key=lambda value: (value.casefold(), value))
    if not terms:
        raise ValueError("leakage lexicon cannot be empty")
    return {
        "schema_version": "yher.llm_sim_v2.leakage_lexicon.v1",
        "terms": terms,
        "sha256": hashlib.sha256(_canonical(terms)).hexdigest(),
    }


def _validate_leakage_lexicon(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("a frozen leakage lexicon is required")
    terms = value.get("terms")
    if not isinstance(terms, list) or not terms or not all(
        isinstance(term, str) and term.strip() for term in terms
    ):
        raise ValueError("leakage lexicon terms must be non-empty strings")
    if len({term.casefold() for term in terms}) != len(terms):
        raise ValueError("leakage lexicon terms must be unique")
    computed = hashlib.sha256(_canonical(terms)).hexdigest()
    if value.get("sha256") != computed:
        raise ValueError("leakage lexicon sha256 mismatch")
    return computed


def validate_analysis_population(
    records: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> list[dict[str, Any]]:
    """Fail closed when an analysis loader crosses the pilot/main boundary."""

    expected = str(phase).strip().lower()
    if expected not in {"pilot", "main"}:
        raise ValueError("analysis phase must be pilot or main")
    output: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("analysis records must be objects")
        if (
            record.get("simulated") is not True
            or record.get("run_id") != RUN_ID
            or record.get("phase") != expected
            or record.get("analysis_population") != expected
        ):
            raise ValueError("analysis record phase/population does not match the requested population")
        output.append(dict(record))
    return output


def build_study_config(
    *,
    personas: Sequence[Any],
    mapping: Mapping[str, Any],
    blind_panel: Mapping[str, Any],
    leakage_lexicon: Mapping[str, Any],
    frozen_at_utc: str,
) -> dict[str, Any]:
    """Build the collection/analysis contract that is committed before observation."""

    persona_rows = [_row(value) for value in personas]
    persona_ids = sorted({str(row.get("persona_id") or "") for row in persona_rows})
    if len(persona_rows) != 100 or len(persona_ids) != 50 or not all(persona_ids):
        raise ValueError("v2 requires 100 paired rows in exactly 50 persona_id clusters")
    counts = Counter(str(row.get("persona_id") or "") for row in persona_rows)
    if set(counts.values()) != {2}:
        raise ValueError("each persona_id must have exactly two paired response rows")
    conditions: dict[str, set[str]] = {}
    for row in persona_rows:
        conditions.setdefault(str(row["persona_id"]), set()).add(
            str(row.get("deficit_condition") or "")
        )
    if any(value != {"deficit", "control"} for value in conditions.values()):
        raise ValueError("each persona cluster requires deficit and control rows")

    if (
        not isinstance(mapping, Mapping)
        or mapping.get("schema_version") != MAPPING_SCHEMA_VERSION
        or mapping.get("frozen") is not True
        or mapping.get("observation_started") is not False
    ):
        raise ValueError("mapping envelope is not a pre-observation frozen mapping")
    computed_mapping_sha = mapping_sha256(mapping)
    computed_target_hash = target_set_hash(mapping)
    consensus = mapping.get("consensus")
    if not isinstance(consensus, Mapping):
        raise ValueError("mapping consensus provenance is required")
    for field in ("draft_sha256", "crosscheck_sha256"):
        digest = str(consensus.get(field) or "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("mapping consensus digest is invalid")
    mapping_rows = mapping.get("rows")
    if not isinstance(mapping_rows, list) or len(mapping_rows) != 100:
        raise ValueError("the frozen target-option mapping must contain exactly 100 rows")
    allowed_statuses = {"mapped", "excluded_ambiguous"}
    if any(str(row.get("status")) not in allowed_statuses for row in mapping_rows):
        raise ValueError("mapping rows contain a non-frozen status")
    mapped_rows = sum(row.get("status") == "mapped" for row in mapping_rows)
    if (
        consensus.get("mapped_rows") != mapped_rows
        or consensus.get("excluded_ambiguous_rows") != len(mapping_rows) - mapped_rows
    ):
        raise ValueError("mapping consensus counts do not match mapping rows")
    mapped_fraction = mapped_rows / len(mapping_rows)
    mapping_passed = mapped_fraction >= MAPPING_MINIMUM_FRACTION

    lexicon_sha = _validate_leakage_lexicon(leakage_lexicon)

    panel_anchors = blind_panel.get("anchors")
    if not isinstance(panel_anchors, list) or len(panel_anchors) != 25:
        raise ValueError("blind panel must contain exactly 25 anchors")
    for anchor in panel_anchors:
        items = anchor.get("items") if isinstance(anchor, Mapping) else None
        if not isinstance(items, list) or not 4 <= len(items) <= 25:
            raise ValueError("each blind anchor must contain four to 25 items")
        item_ids = [str(item.get("item_id") or "") for item in items]
        family_ids = [str(item.get("family_id") or "") for item in items]
        if not all(item_ids) or len(set(item_ids)) != len(item_ids):
            raise ValueError("blind anchor item IDs must be non-empty and unique")
        if not all(family_ids) or len(set(family_ids)) != len(family_ids):
            raise ValueError("blind anchor family IDs must be non-empty and unique")
        calibration_ids = anchor.get("calibration_item_ids")
        if not isinstance(calibration_ids, list) or calibration_ids != item_ids[:4]:
            raise ValueError("blind panel must begin with its exact four calibration items")

    return {
        "schema_version": SCHEMA_VERSION,
        "simulated": True,
        "run_id": RUN_ID,
        "frozen_at_utc": _validate_timestamp(frozen_at_utc),
        "study_seed": 20260715,
        "modality_condition": "text_only",
        "prompt_revision": 0,
        "maximum_prompt_rewrites": 1,
        "rewrite_requires_pre_observation_commit": True,
        "cluster_unit": "persona_id",
        "cluster_count": 50,
        "paired_response_rows": 100,
        "response_arms": ["deficit", "control"],
        "repeated_measure_factors": ["provider", "response_arm"],
        "mapping_sha256": computed_mapping_sha,
        "target_set_hash": computed_target_hash,
        "leakage_lexicon_sha256": lexicon_sha,
        "mapping_gate": {
            "mapped_rows": mapped_rows,
            "total_rows": len(mapping_rows),
            "mapped_fraction": round(mapped_fraction, 8),
            "minimum_fraction": MAPPING_MINIMUM_FRACTION,
            "passed": mapping_passed,
            "confirmatory_target_misconception_hit_rate": mapping_passed,
            "sparse_descriptive_only": not mapping_passed,
        },
        "controlled": {
            "items_per_row": 4,
            "primary_outcomes": [
                "paired_correctness_difference",
                "paired_error_rate_difference",
                "valid_response_rate",
                "abstention_rate",
            ],
            "manipulation_compliance": (
                "confirmatory" if mapping_passed else "sparse_descriptive_only"
            ),
        },
        "blind": {
            "calibration_items_per_row": 4,
            "additional_diagnostic_items_maximum": 21,
            "maximum_items_per_row": 25,
            "primary_outcomes": [
                "terminal_response_consistency",
                "provider_pairwise_agreement",
                "failure_rate",
                "output_stability",
            ],
            "terminal_repeat_subset_persona_count": 10,
        },
        "pilot": {
            "providers": ["deepseek", "doubao"],
            "persona_ids": _pilot_personas(persona_ids),
            "excluded_from_main_analysis": True,
            "physical_phase": "pilot",
        },
        "main": {
            "providers": list(PROVIDERS),
            "persona_ids": persona_ids,
            "physical_phase": "main",
        },
        "providers": {name: dict(value) for name, value in PROVIDERS.items()},
        "provider_exclusion": {
            "blind_invalid_schema_fraction_strictly_above": 0.50,
            "minimum_complete_clusters_per_condition": 45,
            "model_drift": "exclude_and_disclose",
            "technical_failure": "retain_in_denominator_and_disclose",
        },
        "bootstrap": {
            "cluster_unit": "persona_id",
            "resamples": 10_000,
            "seed": 2026071503,
            "confidence_level": 0.95,
        },
        "judge": {
            "blind_to_target_labels": True,
            "labels": [
                "consistent",
                "inconsistent",
                "unknown",
                "insufficient_evidence",
            ],
            "outputs": ["pairwise_agreement", "error_category", "disagreement_examples"],
            "authenticity_score_forbidden": True,
        },
        "budget_yuan": {"soft_warning": 300.0, "hard_fuse": 450.0},
        "phase_isolation": {
            "pilot_population": "pilot",
            "main_population": "main",
            "cross_population_ingestion_forbidden": True,
        },
    }


__all__ = [
    "MAPPING_MINIMUM_FRACTION",
    "PROVIDERS",
    "SCHEMA_VERSION",
    "build_leakage_lexicon",
    "build_study_config",
    "validate_analysis_population",
]
