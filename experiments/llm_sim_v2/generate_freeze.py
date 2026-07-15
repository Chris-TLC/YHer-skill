"""Generate the immutable Persona v2 pre-observation freeze bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.learning.item_catalog import ItemCatalog

from .freeze import (
    build_blind_panel,
    build_freeze_manifest,
    build_leakage_lexicon,
    build_population_manifest,
    build_prompt_revision_ledger,
    build_rendered_prompt_contract_hashes,
    build_study_config,
    verify_freeze_manifest,
)
from .grid import build_persona_grid, grid_sha256
from .official import (
    AUDIT_SAMPLE_SEED,
    build_consensus_mapping,
    build_official_study_inputs,
    select_mapping_audit_sample,
    verify_source_manifest,
)
from .store import RUN_ID


FROZEN_DIR = Path("experiments/llm_sim_v2/frozen_v0")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"expected JSON object list: {path}")
    return [dict(row) for row in value]


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"frozen artifact already exists with different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _catalog(root: Path) -> ItemCatalog:
    return ItemCatalog.from_default_data(
        v4_path=root / "data/item_bank/v4/chemistry_v4_1_3329.jsonl",
        r5_path=root / "data/item_bank/v4/usability_r5_v1.jsonl",
        v3_path=root / "data/item_bank/chemistry_v3_6695.jsonl",
        kg_path=root / "data/knowledge_graph_150_enriched.jsonl",
    )


def _audit_package(
    candidates: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    selected = select_mapping_audit_sample(candidates, seed=AUDIT_SAMPLE_SEED)
    decisions = {
        (str(row["item_id"]), str(row["failure_id"])): row
        for row in mapping["rows"]
    }
    rows = []
    for candidate in selected:
        key = (str(candidate["item_id"]), str(candidate["failure_id"]))
        rows.append({**dict(candidate), "consensus_decision": dict(decisions[key])})
    output = {
        "schema_version": "yher.llm_sim_v2.mapping_audit_sample.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "seed": AUDIT_SAMPLE_SEED,
        "selection_is_outcome_independent": True,
        "rows": rows,
    }
    output["audit_sample_sha256"] = _sha(rows)
    return output


def generate_freeze_bundle(
    repo_root: str | Path,
    *,
    codex_parts: Sequence[str | Path],
    crosscheck_path: str | Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve(strict=True)
    output_root = root / FROZEN_DIR
    official_inputs = build_official_study_inputs(root)
    verify_source_manifest(root, official_inputs["source_manifest"])
    catalog = _catalog(root)

    codex_rows: list[dict[str, Any]] = []
    codex_sources = []
    for raw_path in codex_parts:
        path = Path(raw_path).expanduser().resolve(strict=True)
        rows = _read_list(path)
        codex_rows.extend(rows)
        codex_sources.append(
            {"path_hint": path.name, "sha256": _file_sha(path), "rows": len(rows)}
        )
    crosscheck_source = Path(crosscheck_path).expanduser().resolve(strict=True)
    crosscheck_raw = _read_object(crosscheck_source)
    crosscheck_rows = crosscheck_raw.get("decisions")
    if not isinstance(crosscheck_rows, list):
        raise ValueError("crosscheck artifact has no decisions list")

    mapping = build_consensus_mapping(
        official_inputs["candidates"],
        codex_rows,
        crosscheck_rows,
        items=catalog.items,
    )
    mapping["candidate_frame_sha256"] = _sha(official_inputs["candidates"])
    mapping["mapped_fraction"] = round(
        mapping["consensus"]["mapped_rows"] / len(mapping["rows"]), 8
    )
    mapping["confirmatory_target_misconception_hit_rate"] = (
        mapping["mapped_fraction"] >= 0.60
    )
    mapping["crosscheck_provenance"] = {
        "requested_model": crosscheck_raw.get("requested_model"),
        "returned_models": crosscheck_raw.get("returned_models"),
        "source_file_sha256": _file_sha(crosscheck_source),
        "content_sha256": crosscheck_raw.get("content_sha256"),
        "summary": crosscheck_raw.get("summary"),
        "technical_failures": (crosscheck_raw.get("summary") or {}).get(
            "technical_failures"
        ),
    }

    personas = build_persona_grid(official_inputs["anchors"], seed=20260715)
    persona_rows = [persona.to_dict() for persona in personas]
    persona_artifact = {
        "schema_version": "yher.llm_sim_v2.persona_grid.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "cluster_unit": "persona_id",
        "cluster_count": len({row["persona_id"] for row in persona_rows}),
        "paired_response_rows": len(persona_rows),
        "rows": persona_rows,
        "grid_sha256": grid_sha256(personas),
    }
    blind_panel = build_blind_panel(official_inputs["anchors"], catalog)
    lexicon = build_leakage_lexicon(official_inputs["anchors"])
    study_config = build_study_config(
        personas=personas,
        mapping=mapping,
        blind_panel=blind_panel,
        leakage_lexicon=lexicon,
        frozen_at_utc=frozen_at_utc,
    )
    population = build_population_manifest(study_config)
    rendered_hashes = build_rendered_prompt_contract_hashes(
        personas,
        blind_panel,
        lexicon,
    )
    prompt_paths = [
        "experiments/llm_sim_v2/keys.py",
        "experiments/llm_sim_v2/models.py",
        "experiments/llm_sim_v2/prompts.py",
        "experiments/llm_sim_v2/public.py",
    ]
    prompt_ledger = build_prompt_revision_ledger(
        root,
        prompt_paths=prompt_paths,
        rendered_contract_sha256=rendered_hashes,
        leakage_lexicon_sha256=lexicon["sha256"],
        mapping_sha256=mapping["mapping_sha256"],
        grid_sha256=persona_artifact["grid_sha256"],
        panel_sha256=blind_panel["panel_sha256"],
        frozen_at_utc=frozen_at_utc,
    )
    audit = _audit_package(official_inputs["candidates"], mapping)
    codex_artifact = {
        "schema_version": "yher.llm_sim_v2.codex_mapping_draft.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "sources": codex_sources,
        "rows": codex_rows,
        "decisions_sha256": _sha(codex_rows),
    }
    crosscheck_artifact = {
        "schema_version": "yher.llm_sim_v2.mapping_crosscheck.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "purpose": crosscheck_raw.get("purpose"),
        "requested_model": crosscheck_raw.get("requested_model"),
        "returned_models": crosscheck_raw.get("returned_models"),
        "roster_sha256": crosscheck_raw.get("roster_sha256"),
        "content_sha256": crosscheck_raw.get("content_sha256"),
        "source_file_sha256": _file_sha(crosscheck_source),
        "summary": crosscheck_raw.get("summary"),
        "decisions": crosscheck_rows,
        "decisions_sha256": _sha(crosscheck_rows),
    }

    artifacts: dict[str, Any] = {
        "official_inputs.json": official_inputs,
        "source_manifest.json": official_inputs["source_manifest"],
        "codex_mapping_draft.json": codex_artifact,
        "deepseek_mapping_crosscheck.json": crosscheck_artifact,
        "target_option_mapping.json": mapping,
        "mapping_audit_sample.json": audit,
        "persona_grid.json": persona_artifact,
        "blind_panel.json": blind_panel,
        "leakage_lexicon.json": lexicon,
        "study_config.json": study_config,
        "population_manifest.json": population,
        "prompt_revision_ledger.json": prompt_ledger,
    }
    for name, value in artifacts.items():
        _write_immutable(output_root / name, value)

    artifact_paths = [f"{FROZEN_DIR.as_posix()}/{name}" for name in artifacts]
    declared_paths = {
        "plan": ["experiments/h5v2_analysis_plan.md"],
        "code": [
            "experiments/llm_sim_v2/freeze.py",
            "experiments/llm_sim_v2/generate_freeze.py",
            "experiments/llm_sim_v2/grid.py",
            "experiments/llm_sim_v2/mapping.py",
            "experiments/llm_sim_v2/official.py",
            "experiments/llm_sim_v2/panel.py",
            "experiments/llm_sim_v2/provenance.py",
            "experiments/llm_sim_v2/store.py",
            "tests/test_llm_sim_v2_freeze.py",
        ],
        "prompt": prompt_paths,
        "mapping": [
            path
            for path in artifact_paths
            if Path(path).name
            in {
                "codex_mapping_draft.json",
                "deepseek_mapping_crosscheck.json",
                "mapping_audit_sample.json",
                "target_option_mapping.json",
            }
        ],
        "config": [
            path
            for path in artifact_paths
            if Path(path).name
            not in {
                "codex_mapping_draft.json",
                "deepseek_mapping_crosscheck.json",
                "mapping_audit_sample.json",
                "target_option_mapping.json",
            }
        ],
    }
    plan_path = root / "experiments/h5v2_analysis_plan.md"
    summary_hashes = {
        "analysis_plan_sha256": _file_sha(plan_path),
        "source_set_sha256": official_inputs["source_manifest"]["source_set_sha256"],
        "official_inputs_sha256": official_inputs["inputs_sha256"],
        "grid_sha256": persona_artifact["grid_sha256"],
        "mapping_sha256": mapping["mapping_sha256"],
        "target_set_hash": mapping["target_set_hash"],
        "prompt_ledger_sha256": prompt_ledger["prompt_ledger_sha256"],
        "population_manifest_sha256": population["population_manifest_sha256"],
    }
    freeze_manifest = build_freeze_manifest(
        root,
        declared_paths=declared_paths,
        frozen_at_utc=frozen_at_utc,
        summary_hashes=summary_hashes,
    )
    _write_immutable(output_root / "freeze_manifest.json", freeze_manifest)
    verify_freeze_manifest(root, freeze_manifest)
    return {
        "run_id": RUN_ID,
        "frozen_dir": FROZEN_DIR.as_posix(),
        "freeze_manifest_sha256": freeze_manifest["freeze_manifest_sha256"],
        "mapping_sha256": mapping["mapping_sha256"],
        "target_set_hash": mapping["target_set_hash"],
        "mapped_rows": mapping["consensus"]["mapped_rows"],
        "excluded_ambiguous_rows": mapping["consensus"]["excluded_ambiguous_rows"],
        "blind_panel_counts": blind_panel["counts"],
        "pilot_persona_ids": study_config["pilot"]["persona_ids"],
        "rendered_contract_sha256": rendered_hashes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--codex-part", action="append", required=True)
    parser.add_argument("--crosscheck", required=True)
    parser.add_argument("--frozen-at-utc", required=True)
    args = parser.parse_args(argv)
    result = generate_freeze_bundle(
        args.repo_root,
        codex_parts=args.codex_part,
        crosscheck_path=args.crosscheck,
        frozen_at_utc=args.frozen_at_utc,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
