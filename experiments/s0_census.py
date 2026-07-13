"""Freeze-safe S0 isolation contracts and prerequisite census."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMP_ROOT = Path("/tmp/yher_sprint2")
CENSUS_SEED = 2026071300
MODEL_ID = "deterministic-census-v1"
PROVIDER = "local_programmatic"
BUDGETS = (9, 15, 25)
CONFIG = {
    "schema_version": "yher.simulation.prerequisite_census.v1",
    "census_seed": CENSUS_SEED,
    "minimum_prerequisite_families": 2,
    "held_out_local_families": 2,
    "budgets": list(BUDGETS),
    "label_normalization": {
        "\uff08": "(",
        "\uff09": ")",
        "/": "_",
    },
}


class ProtectedWriteError(RuntimeError):
    """Raised when a simulation changes protected real-student or cache state."""


def require_simulation_output_path(
    path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    temp_root: str | Path = DEFAULT_TEMP_ROOT,
) -> Path:
    """Return a resolved path only when it is inside an approved simulation root."""
    candidate = Path(path).expanduser().resolve(strict=False)
    repository = Path(repo_root).expanduser().resolve(strict=False)
    simulation_root_path = repository / "data" / "sim_store"
    simulation_root = simulation_root_path.resolve(strict=False)
    allowed_roots = [Path(temp_root).expanduser().resolve(strict=False)]
    if (
        not simulation_root_path.is_symlink()
        and simulation_root.is_relative_to(repository)
    ):
        allowed_roots.append(simulation_root)
    if not any(candidate == root or candidate.is_relative_to(root) for root in allowed_roots):
        raise ValueError(
            "simulation output path must be under data/sim_store or /tmp/yher_sprint2"
        )
    return candidate


def require_simulated_event_envelope(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require an explicit simulated marker and non-empty provenance identities."""
    valid = event.get("simulated") is True
    for field in ("persona_id", "provider", "model_id"):
        value = event.get(field)
        valid = valid and isinstance(value, str) and bool(value.strip())
    if not valid:
        raise ValueError(
            "simulated event envelope requires simulated:true, persona_id, provider, and model_id"
        )
    return event


def normalize_kg_label(value: str) -> str:
    """Apply only the mechanically authorized KG/R5 label substitutions."""
    return str(value).replace("\uff08", "(").replace("\uff09", ")").replace("/", "_")


def build_prerequisite_census(catalog: Any) -> dict[str, Any]:
    """Build a deterministic census over every open target in a trusted catalog."""
    open_nodes = catalog.open_nodes()
    kg_target_labels = sorted(
        str(label)
        for label in getattr(catalog, "_prerequisites", open_nodes).keys()
    )
    normalized_kg_targets: dict[str, list[str]] = {}
    for label in kg_target_labels:
        normalized_kg_targets.setdefault(normalize_kg_label(label), []).append(label)
    r5_node_labels = sorted(
        {
            str(node)
            for item in catalog.items.values()
            for node in getattr(item, "node_ids", ())
            if str(node)
        }
    )
    normalized_index: dict[str, list[str]] = {}
    for label in r5_node_labels:
        normalized_index.setdefault(normalize_kg_label(label), []).append(label)

    nodes: list[dict[str, Any]] = []
    for target_node in sorted(open_nodes):
        normalized_target = normalize_kg_label(target_node)
        kg_target_candidates = tuple(normalized_kg_targets.get(normalized_target, ()))
        if target_node in kg_target_candidates:
            resolved_kg_target = target_node
        elif normalized_target in kg_target_candidates:
            resolved_kg_target = normalized_target
        elif len(kg_target_candidates) == 1:
            resolved_kg_target = kg_target_candidates[0]
        else:
            resolved_kg_target = target_node
        local_items = tuple(
            catalog.for_node(target_node, deterministic_only=True)
        )
        local_families = {str(item.family_id) for item in local_items}
        local_capacity = max(0, len(local_families) - CONFIG["held_out_local_families"])
        prerequisites = [
            _census_prerequisite(raw_label, catalog, normalized_index)
            for raw_label in catalog.prerequisites_for(resolved_kg_target)
        ]
        structurally_free = not prerequisites
        eligible = any(
            row["has_two_independent_deterministic_families"]
            for row in prerequisites
        )
        if eligible:
            exclusion_reason = None
        elif structurally_free:
            exclusion_reason = "no_kg_prerequisites"
        else:
            exclusion_reason = "insufficient_prerequisite_family_coverage"
        nodes.append(
            {
                "target_node": str(target_node),
                "normalized_target_node": normalized_target,
                "resolved_kg_target_node": resolved_kg_target,
                "open_node_deterministic_family_count": int(open_nodes[target_node]),
                "local_deterministic_item_count": len(local_items),
                "local_independent_family_count": len(local_families),
                "no_repeat_local_capacity_after_holdout": local_capacity,
                "no_repeat_local_budget_eligible": {
                    str(budget): local_capacity >= budget for budget in BUDGETS
                },
                "prerequisites": prerequisites,
                "has_qualifying_prerequisite": eligible,
                "h1_h2_eligible": eligible,
                "structurally_prerequisite_free": structurally_free,
                "exclusion_reason": exclusion_reason,
            }
        )

    structurally_free_count = sum(
        bool(row["structurally_prerequisite_free"]) for row in nodes
    )
    insufficient_count = sum(
        row["exclusion_reason"] == "insufficient_prerequisite_family_coverage"
        for row in nodes
    )
    eligible_nodes = [row["target_node"] for row in nodes if row["h1_h2_eligible"]]
    return {
        "open_node_count": len(nodes),
        "h1_h2_eligible_count": len(eligible_nodes),
        "structurally_prerequisite_free_count": structurally_free_count,
        "insufficient_prerequisite_coverage_count": insufficient_count,
        "h1_h2_eligible_nodes": eligible_nodes,
        "h1_h2_excluded_nodes": [
            row["target_node"] for row in nodes if not row["h1_h2_eligible"]
        ],
        "nodes": nodes,
    }


def _census_prerequisite(
    raw_label: str,
    catalog: Any,
    normalized_index: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    raw = str(raw_label)
    normalized = normalize_kg_label(raw)
    candidates = tuple(normalized_index.get(normalized, ()))
    if raw in candidates:
        resolved = raw
    elif normalized in candidates:
        resolved = normalized
    elif len(candidates) == 1:
        resolved = candidates[0]
    else:
        resolved = None
    items = (
        tuple(catalog.for_node(resolved, deterministic_only=True))
        if resolved is not None
        else ()
    )
    family_count = len({str(item.family_id) for item in items})
    return {
        "raw_label": raw,
        "normalized_label": normalized,
        "resolved_label": resolved,
        "deterministic_r5_item_count": len(items),
        "independent_deterministic_family_count": family_count,
        "has_two_independent_deterministic_families": family_count >= 2,
    }


def protected_filesystem_fingerprint(
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Hash real logs, projections/snapshots, and repository cache paths."""
    repository = Path(repo_root).expanduser().resolve(strict=False)
    entries: dict[str, str] = {}
    protected_roots = (
        repository / "data" / "local_store",
        repository / "data" / "study_logs",
    )
    for root in protected_roots:
        _collect_path_entries(repository, root, entries)
    for path in _repository_cache_paths(repository):
        _collect_path_entries(repository, path, entries)
    for path in _snapshot_paths(repository):
        _collect_path_entries(repository, path, entries)

    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "entries": entries,
        "coverage": [
            "data/local_store",
            "data/local_store/students",
            "data/local_store/sessions",
            "data/local_store/events",
            "data/study_logs",
            "rec_served_profile_session_snapshots",
            "repository_cache_paths",
        ],
    }


@contextmanager
def guarded_simulation_run(
    repo_root: str | Path = REPO_ROOT,
) -> Iterator[dict[str, Any]]:
    """Assert that protected state is byte-identical across a simulation action."""
    before = protected_filesystem_fingerprint(repo_root)
    evidence = {"before": before, "after": None, "unchanged": False}
    try:
        yield evidence
    finally:
        after = protected_filesystem_fingerprint(repo_root)
        evidence["after"] = after
        evidence["unchanged"] = before == after
        if before != after:
            changed = sorted(
                key
                for key in set(before["entries"]) | set(after["entries"])
                if before["entries"].get(key) != after["entries"].get(key)
            )
            raise ProtectedWriteError(
                "protected simulation state changed: " + ", ".join(changed[:20])
            )


def write_census_artifacts(
    census: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    output_dir: str | Path,
    worklog_path: str | Path,
    repo_root: str | Path = REPO_ROOT,
    temp_root: str | Path = DEFAULT_TEMP_ROOT,
) -> dict[str, Path]:
    """Write envelope-validated census JSON and a concise worklog mirror."""
    output = require_simulation_output_path(
        output_dir, repo_root=repo_root, temp_root=temp_root
    )
    worklog = require_simulation_output_path(
        worklog_path, repo_root=repo_root, temp_root=temp_root
    )
    output.mkdir(parents=True, exist_ok=True)
    worklog.parent.mkdir(parents=True, exist_ok=True)

    base_envelope = {
        "simulated": True,
        "provider": PROVIDER,
        "model_id": MODEL_ID,
    }
    summary_counts = {key: value for key, value in census.items() if key != "nodes"}
    summary_record = {
        **base_envelope,
        "persona_id": "census:all-open-nodes",
        "record_type": "prerequisite_census_summary",
        "provenance": dict(provenance),
        "census": summary_counts,
    }
    require_simulated_event_envelope(summary_record)
    node_records = []
    for node in census.get("nodes", ()):
        target = str(node["target_node"])
        record = {
            **base_envelope,
            "persona_id": f"census:{target}",
            "record_type": "prerequisite_census_node",
            "provenance": dict(provenance),
            **dict(node),
        }
        require_simulated_event_envelope(record)
        node_records.append(record)

    summary_path = require_simulation_output_path(
        output / "summary.json", repo_root=repo_root, temp_root=temp_root
    )
    records_path = require_simulation_output_path(
        output / "prerequisites.jsonl", repo_root=repo_root, temp_root=temp_root
    )
    _write_text(
        summary_path,
        json.dumps(summary_record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _write_text(
        records_path,
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in node_records
        ),
    )
    worklog_text = (
        "\n## S0 prerequisite census (SIMULATED)\n"
        f"- eligible targets: {census['h1_h2_eligible_count']}/{census['open_node_count']}\n"
        "- structurally prerequisite-free exclusions: "
        f"{census['structurally_prerequisite_free_count']}\n"
        "- insufficient prerequisite-family exclusions: "
        f"{census['insufficient_prerequisite_coverage_count']}\n"
        f"- analysis-plan commit: {provenance['analysis_plan_commit']}\n"
        f"- config sha256: {provenance['config_sha256']}\n"
        f"- records: {records_path}\n"
    )
    with worklog.open("a", encoding="utf-8") as handle:
        handle.write(worklog_text)
    return {"summary": summary_path, "records": records_path, "worklog": worklog}


def build_provenance(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Bind the census to git state, frozen analysis plan, inputs, and baseline."""
    repository = Path(repo_root).expanduser().resolve(strict=False)
    from core.data.item_bank_v4 import V4_MANIFEST, V4_USABILITY_R5
    from core.learning.item_catalog import DEFAULT_KG_MANIFEST, DEFAULT_V3_MANIFEST

    analysis_plan = repository / "experiments" / "analysis_plan.md"
    input_paths = {
        "v4_manifest": Path(V4_MANIFEST),
        "r5_usability": Path(V4_USABILITY_R5),
        "trusted_v3_manifest": Path(DEFAULT_V3_MANIFEST),
        "kg_manifest": Path(DEFAULT_KG_MANIFEST),
        "analysis_plan": analysis_plan,
    }
    input_sha256 = {
        name: {
            "path": _display_path(repository, path),
            "sha256": _sha256_file(path),
        }
        for name, path in sorted(input_paths.items())
    }
    config_sha256 = hashlib.sha256(
        json.dumps(CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "repository_head": _git(repository, "rev-parse", "HEAD"),
        "analysis_plan_commit": _git(
            repository,
            "log",
            "-1",
            "--format=%H",
            "--",
            "experiments/analysis_plan.md",
        ),
        "implementation_sha256": _sha256_file(Path(__file__)),
        "input_sha256": input_sha256,
        "census_seed": CENSUS_SEED,
        "config_sha256": config_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pre_existing_baseline": {
            "root_engine_contracts": {
                "passed": 119,
                "total": 119,
                "status": "pass",
            },
            "repo_offline_suite": {
                "passed": 568,
                "total": 569,
                "status": "known_failure",
                "failing_test": (
                    "tests/test_ws2_transcripts.py::test_v3_repository_unaffected"
                ),
                "reason": (
                    "current dirty-tree ItemRepository count is 6440 versus "
                    "the test's hard-coded 6438"
                ),
            },
        },
    }


def run_census(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_dir: str | Path | None = None,
    worklog_path: str | Path | None = None,
    temp_root: str | Path = DEFAULT_TEMP_ROOT,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run the real census and prove protected state remains unchanged."""
    repository = Path(repo_root).expanduser().resolve(strict=False)
    destination = output_dir or repository / "data" / "sim_store" / "census"
    log_path = worklog_path or Path(temp_root) / "WORKLOG.md"
    require_simulation_output_path(
        destination, repo_root=repository, temp_root=temp_root
    )
    require_simulation_output_path(log_path, repo_root=repository, temp_root=temp_root)

    before = protected_filesystem_fingerprint(repository)
    from core.learning.item_catalog import ItemCatalog

    catalog = ItemCatalog.from_default_data()
    census = build_prerequisite_census(catalog)
    provenance = build_provenance(repository)
    after_compute = protected_filesystem_fingerprint(repository)
    _raise_if_changed(before, after_compute)
    provenance["protected_filesystem_assertion"] = {
        "before_sha256": before["digest"],
        "after_sha256": after_compute["digest"],
        "unchanged": True,
        "coverage": before["coverage"],
    }
    paths = write_census_artifacts(
        census,
        provenance,
        output_dir=destination,
        worklog_path=log_path,
        repo_root=repository,
        temp_root=temp_root,
    )
    after_write = protected_filesystem_fingerprint(repository)
    _raise_if_changed(before, after_write)
    return census, paths


def _raise_if_changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if before != after:
        changed = sorted(
            key
            for key in set(before["entries"]) | set(after["entries"])
            if before["entries"].get(key) != after["entries"].get(key)
        )
        raise ProtectedWriteError(
            "protected simulation state changed: " + ", ".join(changed[:20])
        )


def _repository_cache_paths(repository: Path) -> tuple[Path, ...]:
    matches: set[Path] = set()
    excluded_roots = {
        (repository / ".git").resolve(strict=False),
        (repository / "data" / "sim_store").resolve(strict=False),
    }
    for current, directory_names, file_names in os.walk(repository):
        current_path = Path(current)
        current_resolved = current_path.resolve(strict=False)
        if any(
            current_resolved == root or current_resolved.is_relative_to(root)
            for root in excluded_roots
        ):
            directory_names[:] = []
            continue
        directory_names[:] = [
            name
            for name in directory_names
            if name != ".git" and not name.startswith(".venv")
        ]
        for name in list(directory_names):
            if _is_cache_name(name):
                matches.add(current_path / name)
                directory_names.remove(name)
        for name in file_names:
            if _is_cache_name(name):
                matches.add(current_path / name)
    return tuple(sorted(matches, key=lambda path: path.as_posix()))


def _snapshot_paths(repository: Path) -> tuple[Path, ...]:
    data_root = repository / "data"
    sim_store = (data_root / "sim_store").resolve(strict=False)
    matches: set[Path] = set()
    if not data_root.exists():
        return ()
    for path in data_root.rglob("*"):
        resolved = path.resolve(strict=False)
        if resolved == sim_store or resolved.is_relative_to(sim_store):
            continue
        lowered = path.name.lower()
        if (
            "rec_served" in lowered
            or "profile_snapshot" in lowered
            or "session_snapshot" in lowered
        ):
            matches.add(path)
    return tuple(sorted(matches, key=lambda path: path.as_posix()))


def _is_cache_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in {
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".cache",
            "api_cache",
        }
        or lowered.endswith("_cache")
        or "_cache." in lowered
        or "_cache_" in lowered
        or lowered.startswith("cache_")
    )


def _collect_path_entries(repository: Path, path: Path, entries: dict[str, str]) -> None:
    relative = _display_path(repository, path)
    if path.is_symlink():
        entries[relative] = "symlink:" + os.readlink(path)
        return
    if not path.exists():
        entries[relative] = "missing"
        return
    if path.is_file():
        entries[relative] = "file:" + _sha256_file(path)
        return
    entries[relative] = "directory"
    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        child_relative = _display_path(repository, child)
        if child.is_symlink():
            entries[child_relative] = "symlink:" + os.readlink(child)
        elif child.is_file():
            entries[child_relative] = "file:" + _sha256_file(child)
        elif child.is_dir():
            entries[child_relative] = "directory"


def _display_path(repository: Path, path: Path) -> str:
    resolved_repository = repository.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if resolved_path == resolved_repository or resolved_path.is_relative_to(resolved_repository):
        return resolved_path.relative_to(resolved_repository).as_posix() or "."
    return str(resolved_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "sim_store" / "census",
    )
    parser.add_argument(
        "--worklog",
        type=Path,
        default=DEFAULT_TEMP_ROOT / "WORKLOG.md",
    )
    args = parser.parse_args(argv)
    census, paths = run_census(output_dir=args.output_dir, worklog_path=args.worklog)
    print(
        f"S0 census: {census['h1_h2_eligible_count']}/{census['open_node_count']} "
        f"eligible; summary={paths['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
