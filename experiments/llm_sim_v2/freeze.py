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
from .panel import select_blind_items, select_calibration_items
from .prompts import render_blind_prompt, render_controlled_prompt, render_judge_export
from .provenance import hash_declared_files
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


def _with_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(value)
    output[field] = hashlib.sha256(_canonical(output)).hexdigest()
    return output


def _digest(value: Any, *, name: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return digest


def build_population_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind separate pilot/main roots without globally excluding shared IDs."""

    if not isinstance(config, Mapping) or config.get("run_id") != RUN_ID:
        raise ValueError("population manifest requires the frozen v2 config")
    pilot = config.get("pilot")
    main = config.get("main")
    if not isinstance(pilot, Mapping) or not isinstance(main, Mapping):
        raise ValueError("population config requires pilot and main blocks")

    def block(name: str, source: Mapping[str, Any], *, include: bool) -> dict[str, Any]:
        ids = [str(value) for value in source.get("persona_ids", ())]
        providers = [str(value) for value in source.get("providers", ())]
        if not ids or len(set(ids)) != len(ids) or not providers or len(set(providers)) != len(providers):
            raise ValueError(f"{name} population IDs/providers must be unique and non-empty")
        value = {
            "phase": name,
            "analysis_population": name,
            "root_relative": f"data/sim_store/llm_personas/{RUN_ID}/{name}",
            "providers": providers,
            "persona_ids": ids,
            "response_arms": list(config.get("response_arms", ())),
            "conditions": ["controlled", "blind"],
            "include_in_main": include,
        }
        return _with_digest(value, "manifest_sha256")

    pilot_block = block("pilot", pilot, include=False)
    main_block = block("main", main, include=True)
    if not set(pilot_block["persona_ids"]).issubset(main_block["persona_ids"]):
        raise ValueError("pilot persona IDs must be a frozen subset of main persona IDs")
    manifest = {
        "schema_version": "yher.llm_sim_v2.population_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "selection_seed": int(config.get("study_seed", 0)),
        "pilot": pilot_block,
        "main": main_block,
        "ingestion_policy": {
            "accepted_phase": "main",
            "accepted_analysis_population": "main",
            "pilot_records_never_join_main": True,
            "same_cluster_recollection_allowed": True,
            "filesystem_glob_forbidden": True,
            "logical_key": [
                "phase",
                "provider",
                "persona_id",
                "response_arm",
                "condition",
                "prompt_revision",
                "item_id",
                "attempt_id",
            ],
        },
    }
    return _with_digest(manifest, "population_manifest_sha256")


def build_prompt_revision_ledger(
    repo_root: str | Any,
    *,
    prompt_paths: Sequence[str],
    rendered_contract_sha256: Mapping[str, str],
    leakage_lexicon_sha256: str,
    mapping_sha256: str,
    grid_sha256: str,
    panel_sha256: str,
    frozen_at_utc: str,
) -> dict[str, Any]:
    """Build immutable revision zero; later rewrites require a separate commit."""

    if set(rendered_contract_sha256) != {"controlled", "blind", "judge"}:
        raise ValueError("rendered prompt contracts must bind controlled, blind, and judge")
    rendered = {
        name: _digest(digest, name=f"rendered {name} contract")
        for name, digest in sorted(rendered_contract_sha256.items())
    }
    files = hash_declared_files(repo_root, {"prompt": list(prompt_paths)})["files"]
    revision = {
        "revision": 0,
        "parent_revision": None,
        "status": "pre_observation_frozen",
        "prompt_files": files,
        "rendered_contract_sha256": rendered,
        "blind_lexicon_sha256": _digest(
            leakage_lexicon_sha256, name="blind leakage lexicon"
        ),
        "mapping_sha256": _digest(mapping_sha256, name="mapping"),
        "grid_sha256": _digest(grid_sha256, name="grid"),
        "panel_sha256": _digest(panel_sha256, name="panel"),
        "committed_at_utc": _validate_timestamp(frozen_at_utc),
        "reason": "initial_pre_observation_freeze",
        "calibration_rewrite_required": False,
        "observed_row_count": 0,
    }
    revision["prompt_contract_sha256"] = hashlib.sha256(_canonical(revision)).hexdigest()
    ledger = {
        "schema_version": "yher.llm_sim_v2.prompt_revision_ledger.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "maximum_prompt_rewrites": 1,
        "current_revision": 0,
        "observation_started_at": None,
        "revisions": [revision],
        "transition_policy": {
            "only_transition": "0_to_1",
            "requires_calibration_rewrite_required": True,
            "requires_pre_observation_commit": True,
            "main_observation_forbids_transition": True,
            "retry_does_not_change_revision": True,
        },
    }
    return _with_digest(ledger, "prompt_ledger_sha256")


_FREEZE_SUMMARY_FIELDS = {
    "analysis_plan_sha256",
    "source_set_sha256",
    "official_inputs_sha256",
    "grid_sha256",
    "mapping_sha256",
    "target_set_hash",
    "prompt_ledger_sha256",
    "population_manifest_sha256",
}


def build_freeze_manifest(
    repo_root: str | Any,
    *,
    declared_paths: Mapping[str, Sequence[str]],
    frozen_at_utc: str,
    summary_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if set(summary_hashes) != _FREEZE_SUMMARY_FIELDS:
        raise ValueError("freeze summary hash set is incomplete")
    summary = {
        key: _digest(value, name=key) for key, value in sorted(summary_hashes.items())
    }
    provenance = hash_declared_files(repo_root, declared_paths)
    manifest = {
        "schema_version": "yher.llm_sim_v2.freeze_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "status": "pre_observation_frozen",
        "frozen_at_utc": _validate_timestamp(frozen_at_utc),
        "observation_started_at": None,
        **summary,
        "artifact_set_sha256": provenance["file_set_sha256"],
        "frozen_files": provenance["files"],
    }
    return _with_digest(manifest, "freeze_manifest_sha256")


def verify_freeze_manifest(repo_root: str | Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "yher.llm_sim_v2.freeze_manifest.v1":
        raise ValueError("unsupported freeze manifest")
    advertised = manifest.get("freeze_manifest_sha256")
    payload = dict(manifest)
    payload.pop("freeze_manifest_sha256", None)
    if advertised != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("freeze manifest digest mismatch")
    rows = manifest.get("frozen_files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("freeze manifest has no frozen files")
    declared: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("category"):
            raise ValueError("freeze manifest rows require categories")
        declared.setdefault(str(row["category"]), []).append(str(row.get("path") or ""))
    current = hash_declared_files(repo_root, declared)
    if current["files"] != rows or current["file_set_sha256"] != manifest.get("artifact_set_sha256"):
        raise ValueError("freeze artifact bytes drifted from the manifest")
    return {
        "schema_version": "yher.llm_sim_v2.freeze_manifest_proof.v1",
        "ok": True,
        "run_id": RUN_ID,
        "freeze_manifest_sha256": advertised,
        "artifact_set_sha256": current["file_set_sha256"],
    }


def build_blind_panel(anchors: Sequence[Mapping[str, Any]], catalog: Any) -> dict[str, Any]:
    """Build a globally family-distinct 4+up-to-21 item panel per anchor."""

    normalized = sorted(
        (dict(anchor) for anchor in anchors),
        key=lambda anchor: (str(anchor.get("anchor_id") or ""), str(anchor.get("target_node") or "")),
    )
    if len(normalized) != 25:
        raise ValueError("the frozen blind panel requires exactly 25 anchors")
    anchor_ids = [str(anchor.get("anchor_id") or "") for anchor in normalized]
    target_nodes = [str(anchor.get("target_node") or "") for anchor in normalized]
    if not all(anchor_ids) or len(set(anchor_ids)) != 25:
        raise ValueError("blind panel anchor IDs must be unique and non-empty")
    if not all(target_nodes) or len(set(target_nodes)) != 25:
        raise ValueError("blind panel target nodes must be unique and non-empty")

    selected: dict[str, list[dict[str, Any]]] = {}
    tails: dict[str, list[dict[str, Any]]] = {}
    used_items: set[str] = set()
    used_families: set[str] = set()
    for anchor in normalized:
        anchor_id = str(anchor["anchor_id"])
        calibration = select_calibration_items(anchor, catalog)
        full = select_blind_items(anchor, catalog)
        for item in calibration:
            if item["item_id"] in used_items or item["family_id"] in used_families:
                raise ValueError("calibration support overlaps across blind anchors")
            used_items.add(item["item_id"])
            used_families.add(item["family_id"])
        selected[anchor_id] = [dict(item) for item in calibration]
        calibration_ids = {item["item_id"] for item in calibration}
        tails[anchor_id] = [dict(item) for item in full if item["item_id"] not in calibration_ids]

    progress = True
    while progress:
        progress = False
        for anchor in normalized:
            anchor_id = str(anchor["anchor_id"])
            if len(selected[anchor_id]) >= 25:
                continue
            while tails[anchor_id]:
                candidate = tails[anchor_id].pop(0)
                if candidate["item_id"] in used_items or candidate["family_id"] in used_families:
                    continue
                selected[anchor_id].append(candidate)
                used_items.add(candidate["item_id"])
                used_families.add(candidate["family_id"])
                progress = True
                break

    rows: list[dict[str, Any]] = []
    counts: list[int] = []
    for anchor in normalized:
        anchor_id = str(anchor["anchor_id"])
        items = []
        for index, item in enumerate(selected[anchor_id]):
            items.append(
                {
                    **item,
                    "ordinal": index,
                    "role": "calibration" if index < 4 else "diagnostic",
                }
            )
        counts.append(len(items))
        rows.append(
            {
                "anchor_id": anchor_id,
                "target_node": str(anchor["target_node"]),
                "calibration_item_ids": [item["item_id"] for item in items[:4]],
                "items": items,
            }
        )
    panel_hash = hashlib.sha256(_canonical(rows)).hexdigest()
    return {
        "schema_version": "yher.llm_sim_v2.blind_panel.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "anchors": rows,
        "counts": {
            "anchors": len(rows),
            "total_items": sum(counts),
            "minimum_items_per_anchor": min(counts),
            "maximum_items_per_anchor": max(counts),
        },
        "panel_sha256": panel_hash,
    }


def build_rendered_prompt_contract_hashes(
    personas: Sequence[Any],
    blind_panel: Mapping[str, Any],
    leakage_lexicon: Mapping[str, Any],
) -> dict[str, str]:
    """Hash every frozen rendered prompt without persisting private prompt bodies."""

    terms = leakage_lexicon.get("terms") if isinstance(leakage_lexicon, Mapping) else None
    if not isinstance(terms, list) or not terms:
        raise ValueError("rendered prompt contracts require the frozen leakage lexicon")
    anchors = blind_panel.get("anchors") if isinstance(blind_panel, Mapping) else None
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("rendered prompt contracts require a blind panel")
    panel_index = {
        str(anchor.get("anchor_id") or ""): anchor
        for anchor in anchors
        if isinstance(anchor, Mapping)
    }
    controlled_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    for raw_persona in personas:
        persona = _row(raw_persona)
        row_id = str(persona.get("row_id") or persona.get("persona_id") or "")
        anchor_id = str(persona.get("anchor_id") or "")
        anchor = panel_index.get(anchor_id)
        if not row_id or anchor is None:
            raise ValueError("persona row does not bind to the frozen blind panel")
        items = anchor.get("items")
        if not isinstance(items, list) or len(items) < 4:
            raise ValueError("blind panel anchor is structurally incomplete")
        for item in items[:4]:
            controlled_rows.append(
                {
                    "row_id": row_id,
                    "item_id": item["item_id"],
                    "messages": render_controlled_prompt(persona, item),
                }
            )
        for item in items:
            messages = render_blind_prompt(
                persona,
                item,
                frozen_leakage_lexicon=terms,
            )
            blind_rows.append(
                {"row_id": row_id, "item_id": item["item_id"], "messages": messages}
            )
            options = item.get("options") or {}
            answer = sorted(str(option) for option in options)[0] if options else None
            observed = {
                "simulated": True,
                "answer": answer,
                "rationale": "synthetic contract fixture",
                "abstain": answer is None,
            }
            judge_rows.append(
                {
                    "row_id": row_id,
                    "item_id": item["item_id"],
                    "messages": render_judge_export(
                        blind_messages=messages,
                        model_output=observed,
                        persona=persona,
                        item=item,
                        frozen_leakage_lexicon=terms,
                    ),
                }
            )
    return {
        "controlled": hashlib.sha256(_canonical(controlled_rows)).hexdigest(),
        "blind": hashlib.sha256(_canonical(blind_rows)).hexdigest(),
        "judge": hashlib.sha256(_canonical(judge_rows)).hexdigest(),
    }


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
    "build_blind_panel",
    "build_freeze_manifest",
    "build_population_manifest",
    "build_prompt_revision_ledger",
    "build_rendered_prompt_contract_hashes",
    "build_study_config",
    "validate_analysis_population",
    "verify_freeze_manifest",
]
