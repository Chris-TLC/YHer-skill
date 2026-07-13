"""Trusted R5 pools, S0 eligibility, held-out reservation, and capacity sets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.learning.item_catalog import ItemCatalog
from experiments.s0_census import require_simulated_event_envelope

from .allocation import precompute_common_support
from .config import REPO_ROOT, canonical_json_bytes
from .models import CatalogContext, EmpiricalItem, TargetPools


DEFAULT_CENSUS_SUMMARY = REPO_ROOT / "data" / "sim_store" / "census" / "summary.json"
DEFAULT_CENSUS_RECORDS = (
    REPO_ROOT / "data" / "sim_store" / "census" / "prerequisites.jsonl"
)


def load_catalog_context(
    config,
    *,
    census_summary_path: str | Path = DEFAULT_CENSUS_SUMMARY,
    census_records_path: str | Path = DEFAULT_CENSUS_RECORDS,
) -> CatalogContext:
    summary_path = Path(census_summary_path)
    records_path = Path(census_records_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_simulated_event_envelope(summary)
    records = tuple(_read_jsonl(records_path))
    for record in records:
        require_simulated_event_envelope(record)
    census = summary.get("census") or {}
    _validate_census_contract(census, records, config)
    input_sha256 = _validate_and_collect_inputs(
        summary,
        summary_path,
        records_path,
        config=config,
    )

    catalog = ItemCatalog.from_default_data()
    open_nodes = catalog.open_nodes()
    if tuple(sorted(open_nodes)) != tuple(sorted(row["target_node"] for row in records)):
        raise ValueError("S0 target set differs from the current trusted R5 catalog")
    eligible = tuple(str(value) for value in census["h1_h2_eligible_nodes"])
    excluded = tuple(str(value) for value in census["h1_h2_excluded_nodes"])
    record_by_target = {str(row["target_node"]): row for row in records}
    targets: dict[str, TargetPools] = {}
    for target in sorted(open_nodes):
        local_catalog_items = tuple(
            catalog.for_node(target, deterministic_only=True)
        )
        held_out, held_out_families = _reserve_held_out(
            target,
            local_catalog_items,
            master_seed=config.master_seed,
            count=int(config.raw["held_out_families"]),
        )
        local_items = tuple(
            _adapt(item, node_id=target, role="local")
            for item in local_catalog_items
            if str(item.family_id) not in held_out_families
        )
        prerequisite_items = _prerequisite_pool(
            catalog,
            record_by_target[target],
            held_out_families,
        )
        support = precompute_common_support(
            local_items,
            prerequisite_items,
            config.budgets,
            probe_interval=int(config.raw["c_probe_interval"]),
        )
        targets[target] = TargetPools(
            target_node=target,
            local_items=local_items,
            prerequisite_items=prerequisite_items,
            held_out_items=held_out,
            held_out_family_ids=held_out_families,
            h1_h2_eligible=target in eligible,
            common_support_no_repeat=support,
        )

    set_hashes: dict[int, str] = {}
    for budget in config.budgets:
        members = sorted(
            target
            for target, pools in targets.items()
            if pools.common_support_no_repeat[budget]
        )
        material = {
            "budget": budget,
            "targets": members,
            "config_sha256": config.sha256,
            "census_summary_sha256": input_sha256["census_summary"]["sha256"],
            "census_records_sha256": input_sha256["census_records"]["sha256"],
        }
        set_hashes[budget] = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    targets = {
        target: replace(
            pools,
            common_support_set_sha256=dict(set_hashes),
        )
        for target, pools in targets.items()
    }
    return CatalogContext(
        targets=targets,
        h1_h2_eligible_targets=eligible,
        h1_h2_excluded_targets=excluded,
        input_sha256=input_sha256,
    )


def _reserve_held_out(
    target: str,
    items: tuple[Any, ...],
    *,
    master_seed: int,
    count: int,
) -> tuple[tuple[EmpiricalItem, ...], frozenset[str]]:
    by_family: dict[str, list[Any]] = {}
    for item in items:
        by_family.setdefault(str(item.family_id), []).append(item)
    ordered_families = sorted(
        by_family,
        key=lambda family_id: (
            hashlib.sha256(
                f"yher-heldout-v1|{master_seed}|{target}|{family_id}".encode("utf-8")
            ).hexdigest(),
            family_id,
        ),
    )
    if len(ordered_families) < count:
        return (), frozenset()
    selected_families = frozenset(ordered_families[:count])
    held_out = tuple(
        _adapt(
            min(by_family[family_id], key=lambda item: str(item.item_id)),
            node_id=target,
            role="held_out",
        )
        for family_id in ordered_families[:count]
    )
    return held_out, selected_families


def _prerequisite_pool(
    catalog: ItemCatalog,
    census_record: Mapping[str, Any],
    held_out_families: frozenset[str],
) -> tuple[EmpiricalItem, ...]:
    by_item_id: dict[str, EmpiricalItem] = {}
    qualifying_labels = sorted(
        str(row["resolved_label"])
        for row in census_record.get("prerequisites", ())
        if row.get("has_two_independent_deterministic_families")
        and row.get("resolved_label")
    )
    for label in qualifying_labels:
        for item in catalog.for_node(label, deterministic_only=True):
            if str(item.family_id) in held_out_families:
                continue
            by_item_id.setdefault(
                str(item.item_id),
                _adapt(item, node_id=label, role="prereq"),
            )
    return tuple(
        sorted(
            by_item_id.values(),
            key=lambda item: (item.family_id, item.item_id, item.node_id),
        )
    )


def _adapt(item: Any, *, node_id: str, role: str) -> EmpiricalItem:
    item_type = "numeric" if str(item.item_type) == "numeric" else "mcq"
    return EmpiricalItem(
        item_id=str(item.item_id),
        family_id=str(item.family_id),
        node_id=str(node_id),
        difficulty=float(item.difficulty),
        item_type=item_type,
        role=role,
    )


def _validate_census_contract(
    census: Mapping[str, Any],
    records: tuple[Mapping[str, Any], ...],
    config,
) -> None:
    expected = {
        "open_node_count": 27,
        "h1_h2_eligible_count": 23,
        "structurally_prerequisite_free_count": 4,
    }
    for field, value in expected.items():
        if int(census.get(field, -1)) != value:
            raise ValueError(f"S0 census {field} must remain {value}")
    if len(records) != 27:
        raise ValueError("S0 prerequisite records must contain all 27 open nodes")
    if str(config.raw["census_analysis_plan_commit"]) != str(
        records[0].get("provenance", {}).get("analysis_plan_commit")
    ):
        raise ValueError("config census-plan commit differs from S0 provenance")


def _validate_and_collect_inputs(
    summary: Mapping[str, Any],
    summary_path: Path,
    records_path: Path,
    *,
    config,
) -> dict[str, dict[str, str]]:
    provenance = summary.get("provenance") or {}
    census_plan_commit = str(provenance.get("analysis_plan_commit", ""))
    if census_plan_commit != str(config.raw["census_analysis_plan_commit"]):
        raise ValueError("S0 summary census-plan commit differs from frozen config")
    output: dict[str, dict[str, str]] = {}
    for name, metadata in sorted((provenance.get("input_sha256") or {}).items()):
        path = Path(str(metadata["path"]))
        if not path.is_absolute():
            path = REPO_ROOT / path
        actual = (
            _git_blob_sha256(census_plan_commit, str(metadata["path"]))
            if name == "analysis_plan"
            else _sha256(path)
        )
        if actual != str(metadata["sha256"]):
            raise ValueError(f"confirmatory input hash changed: {name}")
        output[str(name)] = {
            "path": _display_path(path),
            "sha256": actual,
            **(
                {
                    "source_kind": "git_blob",
                    "source_commit": census_plan_commit,
                }
                if name == "analysis_plan"
                else {"source_kind": "working_tree"}
            ),
        }
    output["confirmatory_analysis_plan"] = {
        "path": "experiments/analysis_plan.md",
        "sha256": _sha256(REPO_ROOT / "experiments" / "analysis_plan.md"),
    }
    for name, path in (
        ("census_summary", summary_path),
        ("census_records", records_path),
        ("production_mastery", REPO_ROOT / "engine" / "mastery.py"),
        ("production_selector", REPO_ROOT / "engine" / "selector.py"),
    ):
        output[name] = {"path": _display_path(path), "sha256": _sha256(path)}
    return output


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("census JSONL rows must be objects")
                yield value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ValueError("historical input path must be repository-relative")
    completed = subprocess.run(
        ("git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative_path}"),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("cannot read historical S0 analysis-plan blob")
    return hashlib.sha256(completed.stdout).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    repository = REPO_ROOT.resolve(strict=False)
    if resolved == repository or resolved.is_relative_to(repository):
        return resolved.relative_to(repository).as_posix()
    return str(resolved)
