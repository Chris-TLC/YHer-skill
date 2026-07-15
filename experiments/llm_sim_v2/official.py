"""Read-only official input adaptation and pre-observation mapping consensus."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.s0_census import normalize_kg_label

from .mapping import normalize_target_option_map
from .panel import is_calibration_candidate, select_calibration_items
from .store import RUN_ID


SCHEMA_VERSION = "yher.llm_sim_v2.official_inputs.v1"
SOURCE_SCHEMA_VERSION = "yher.llm_sim_v2.source_manifest.v1"
AUDIT_SAMPLE_SEED = 20260715


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _default_sources(root: Path) -> dict[str, Path]:
    return {
        "item_bank_v4": root / "data/item_bank/v4/chemistry_v4_1_3329.jsonl",
        "knowledge_graph": root / "data/knowledge_graph_150_enriched.jsonl",
        "r5_usability": root / "data/item_bank/v4/usability_r5_v1.jsonl",
        "service_exclusions": root / "data/item_bank/v4/service_exclusions.jsonl",
        "v3_catalog": root / "data/item_bank/chemistry_v3_6695.jsonl",
    }


def build_source_manifest(
    repo_root: str | Path,
    *,
    sources: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Bind both existing bytes and the intentional absence of an input."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    configured = dict(sources or _default_sources(root))
    if not configured:
        raise ValueError("official source manifest cannot be empty")
    rows: list[dict[str, Any]] = []
    for role, raw_path in configured.items():
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve(strict=False)
        if not path.is_relative_to(root):
            raise ValueError("official source paths must stay inside the repository")
        relative = path.relative_to(root).as_posix()
        exists = path.is_file()
        data = path.read_bytes() if exists else None
        rows.append(
            {
                "role": str(role),
                "path": relative,
                "exists": exists,
                "sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
                "size": len(data) if data is not None else None,
            }
        )
    rows.sort(key=lambda row: (row["role"], row["path"]))
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "files": rows,
        "source_set_sha256": _sha(rows),
    }


def verify_source_manifest(repo_root: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported official source manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("official source manifest requires files")
    advertised = manifest.get("source_set_sha256")
    if advertised != _sha(files):
        raise ValueError("official source manifest digest mismatch")
    sources = {
        str(row.get("role") or ""): Path(str(row.get("path") or ""))
        for row in files
        if isinstance(row, Mapping)
    }
    if len(sources) != len(files) or not all(sources):
        raise ValueError("official source manifest roles must be unique and non-empty")
    current = build_source_manifest(repo_root, sources=sources)
    if current["files"] != files or current["source_set_sha256"] != advertised:
        raise ValueError("official source state drifted after freeze")
    return {
        "schema_version": "yher.llm_sim_v2.source_proof.v1",
        "ok": True,
        "source_set_sha256": advertised,
        "files": deepcopy(files),
    }


def _failure_rows(
    kg_rows: Iterable[Any],
    eligible_nodes: set[str],
) -> list[tuple[str, int, Any, str]]:
    normalized_eligible: dict[str, list[str]] = defaultdict(list)
    for node in sorted(eligible_nodes):
        normalized_eligible[normalize_kg_label(node)].append(node)
    rows: list[tuple[str, int, Any, str]] = []
    for node in kg_rows:
        source_node = str(_value(node, "node_id", "") or "").strip()
        parent_node = str(_value(node, "parent_node", "") or "").strip()
        if not source_node:
            continue
        if source_node in eligible_nodes:
            target_node = source_node
        elif parent_node in eligible_nodes:
            target_node = parent_node
        else:
            target_node = ""
            for label in (source_node, parent_node):
                candidates = normalized_eligible.get(normalize_kg_label(label), ()) if label else ()
                if len(candidates) == 1:
                    target_node = candidates[0]
                    break
            if not target_node:
                continue
        for failure_index, failure in enumerate(_value(node, "common_failures", ()) or ()):
            rows.append((target_node, failure_index, failure, source_node))
    grouped: dict[str, list[tuple[str, int, Any, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)
    for group in grouped.values():
        group.sort(
            key=lambda row: (
                row[3],
                row[1],
                str(_value(row[2], "cause", "")),
                str(_value(row[2], "diagnostic_question", "")),
            )
        )
    interleaved: list[tuple[str, int, Any, str]] = []
    max_rows = max((len(group) for group in grouped.values()), default=0)
    for rank in range(max_rows):
        for target in sorted(grouped):
            if rank < len(grouped[target]):
                interleaved.append(grouped[target][rank])
    return interleaved


def derive_anchor_roster(
    kg_rows: Iterable[Any],
    *,
    eligible_nodes: set[str] | frozenset[str],
    pair_count: int = 25,
) -> list[dict[str, Any]]:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    rows = _failure_rows(kg_rows, {str(node) for node in eligible_nodes})
    if len(rows) < pair_count:
        raise ValueError(f"only {len(rows)} eligible failure anchors are available")
    anchors: list[dict[str, Any]] = []
    for index, (target, failure_index, failure, source_node) in enumerate(rows[:pair_count]):
        cause = str(_value(failure, "cause", "") or "").strip()
        symptom = str(_value(failure, "symptom", "") or "").strip()
        if not cause or not symptom:
            raise ValueError("official failure anchors require non-empty cause and symptom")
        anchors.append(
            {
                "anchor_id": f"anchor-v2:{index:02d}:{target}",
                "target_node": target,
                "failure_id": f"{source_node}#failure-{failure_index:02d}",
                "failure_cause": cause,
                "failure_symptom": symptom,
                "diagnostic_question": str(
                    _value(failure, "diagnostic_question", "") or ""
                ).strip(),
                "curriculum_exposure": [target],
            }
        )
    return anchors


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
    return rows


def _calibration_ready_targets(catalog: Any, open_targets: Iterable[str]) -> set[str]:
    ready: set[str] = set()
    for target in sorted({str(value) for value in open_targets}):
        families = {
            str(_value(item, "family_id", ""))
            for item in catalog.for_node(target, deterministic_only=True)
            if is_calibration_candidate(item)
        }
        if len(families) >= 4:
            ready.add(target)
    return ready


def build_official_study_inputs(repo_root: str | Path) -> dict[str, Any]:
    """Build deterministic in-memory derivatives without writing official data."""

    from core.learning.item_catalog import ItemCatalog

    root = Path(repo_root).expanduser().resolve(strict=True)
    source_manifest = build_source_manifest(root)
    catalog = ItemCatalog.from_default_data(
        v4_path=root / "data/item_bank/v4/chemistry_v4_1_3329.jsonl",
        r5_path=root / "data/item_bank/v4/usability_r5_v1.jsonl",
        v3_path=root / "data/item_bank/chemistry_v3_6695.jsonl",
        kg_path=root / "data/knowledge_graph_150_enriched.jsonl",
    )
    open_targets = set(catalog.open_nodes())
    ready_targets = _calibration_ready_targets(catalog, open_targets)
    kg_rows = _iter_jsonl(root / "data/knowledge_graph_150_enriched.jsonl")
    anchors = derive_anchor_roster(kg_rows, eligible_nodes=ready_targets, pair_count=25)
    candidates: list[dict[str, Any]] = []
    for anchor in anchors:
        for item in select_calibration_items(anchor, catalog):
            correct = str(item["correct_option"])
            options = dict(item["options"])
            candidates.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "target_node": anchor["target_node"],
                    "failure_id": anchor["failure_id"],
                    "failure_cause": anchor["failure_cause"],
                    "failure_symptom": anchor["failure_symptom"],
                    "item_id": item["item_id"],
                    "family_id": item["family_id"],
                    "public_question": item["public_question"],
                    "options": options,
                    "correct_option": correct,
                    "wrong_options": sorted(option for option in options if option != correct),
                }
            )
    candidates.sort(key=lambda row: (row["anchor_id"], row["family_id"], row["item_id"]))
    roster_rows = [
        {
            key: row[key]
            for key in (
                "anchor_id",
                "target_node",
                "failure_id",
                "item_id",
                "family_id",
                "correct_option",
            )
        }
        for row in candidates
    ]
    selected_targets = {str(anchor["target_node"]) for anchor in anchors}
    stats = catalog.stats()
    output = {
        "schema_version": SCHEMA_VERSION,
        "simulated": True,
        "run_id": RUN_ID,
        "modality_condition": "text_only",
        "source_manifest": source_manifest,
        "catalog_stats": {
            "r5_rows": int(stats.r5_rows),
            "trusted_items": int(stats.trusted_items),
            "rejected_items": int(stats.rejected_items),
            "families": int(stats.families),
        },
        "counts": {
            "open_targets": len(open_targets),
            "calibration_ready_targets": len(ready_targets),
            "selected_anchors": len(anchors),
            "calibration_candidates": len(candidates),
        },
        "excluded_open_targets": sorted(open_targets - ready_targets),
        "unselected_ready_targets": sorted(ready_targets - selected_targets),
        "anchors": anchors,
        "candidates": candidates,
        "roster_sha256": _sha(roster_rows),
    }
    output["inputs_sha256"] = _sha(output)
    return output


def select_mapping_audit_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = AUDIT_SAMPLE_SEED,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        anchor_id = str(row.get("anchor_id") or "")
        item_id = str(row.get("item_id") or "")
        failure_id = str(row.get("failure_id") or "")
        if not anchor_id or not item_id or not failure_id:
            raise ValueError("audit frame rows require anchor_id, item_id, and failure_id")
        grouped[anchor_id].append(row)
    if len(grouped) != 25 or any(len(group) != 4 for group in grouped.values()):
        raise ValueError("the frozen audit frame must contain 25 anchors with four rows each")

    def rank(*parts: str) -> str:
        return hashlib.sha256("|".join((str(seed), *parts)).encode("utf-8")).hexdigest()

    chosen_anchors = sorted(grouped, key=lambda anchor: (rank("anchor", anchor), anchor))[:20]
    selected: list[dict[str, Any]] = []
    for anchor in chosen_anchors:
        chosen = min(
            grouped[anchor],
            key=lambda row: (
                rank("row", anchor, str(row["item_id"]), str(row["failure_id"])),
                str(row["item_id"]),
            ),
        )
        selected.append(deepcopy(dict(chosen)))
    selected.sort(key=lambda row: (row["anchor_id"], row["item_id"]))
    return selected


def _decision_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected: set[tuple[str, str]],
    source: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        item_id = str(raw.get("item_id") or "")
        failure_id = str(raw.get("failure_id") or "")
        key = (item_id, failure_id)
        if key in output:
            raise ValueError(f"{source} decisions contain a duplicate row")
        status = str(raw.get("status") or "").strip().lower()
        if status not in {"mapped", "excluded_ambiguous"}:
            raise ValueError(f"{source} decision status is technical/schema invalid: {status}")
        target = raw.get("target_option")
        output[key] = {
            "status": status,
            "target_option": str(target).strip().upper() if target is not None else None,
        }
    if set(output) != expected:
        raise ValueError(f"{source} decisions do not exactly cover the frozen candidate frame")
    return output


def build_consensus_mapping(
    candidates: Sequence[Mapping[str, Any]],
    codex_draft: Sequence[Mapping[str, Any]],
    independent_crosscheck: Sequence[Mapping[str, Any]],
    *,
    items: Any,
) -> dict[str, Any]:
    candidate_index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in candidates:
        key = (str(row.get("item_id") or ""), str(row.get("failure_id") or ""))
        if not all(key) or key in candidate_index:
            raise ValueError("candidate frame contains an invalid or duplicate row")
        candidate_index[key] = row
    expected = set(candidate_index)
    if not expected:
        raise ValueError("candidate frame cannot be empty")
    drafted = _decision_index(codex_draft, expected=expected, source="codex draft")
    checked = _decision_index(
        independent_crosscheck,
        expected=expected,
        source="independent crosscheck",
    )
    provenance = {
        "method": "independent_dual_model_consensus",
        "drafted_by": "codex_gpt_5_6_sol_ultra",
        "crosschecked_by": "deepseek_chat",
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(expected):
        candidate = candidate_index[key]
        left, right = drafted[key], checked[key]
        wrong_options = {str(option) for option in candidate.get("wrong_options", ())}
        for source, decision in (("codex draft", left), ("independent crosscheck", right)):
            if decision["status"] == "mapped" and decision["target_option"] not in wrong_options:
                raise ValueError(f"{source} target option is not a frozen wrong option")
            if decision["status"] == "excluded_ambiguous" and decision["target_option"] is not None:
                raise ValueError(f"{source} ambiguous decision cannot carry a target option")
        agrees = (
            left["status"] == "mapped"
            and right["status"] == "mapped"
            and left["target_option"] == right["target_option"]
        )
        row = {
            "item_id": key[0],
            "failure_id": key[1],
            "target_option": left["target_option"] if agrees else None,
            "status": "mapped" if agrees else "excluded_ambiguous",
            "reviewer_provenance": provenance,
        }
        if not agrees:
            row["ambiguity_reason"] = "independent_mapping_disagreement"
        rows.append(row)
    normalized = normalize_target_option_map(
        rows,
        items=items,
        expected_rows=sorted(expected),
    )
    normalized["consensus"] = {
        "draft_sha256": _sha(list(codex_draft)),
        "crosscheck_sha256": _sha(list(independent_crosscheck)),
        "mapped_rows": sum(row["status"] == "mapped" for row in normalized["rows"]),
        "excluded_ambiguous_rows": sum(
            row["status"] == "excluded_ambiguous" for row in normalized["rows"]
        ),
    }
    return normalized


__all__ = [
    "AUDIT_SAMPLE_SEED",
    "SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "build_consensus_mapping",
    "build_official_study_inputs",
    "build_source_manifest",
    "derive_anchor_roster",
    "select_mapping_audit_sample",
    "verify_source_manifest",
]
