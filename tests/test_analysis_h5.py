"""Contracts for the frozen H5 collection finalizer and analyzer."""

from __future__ import annotations

import copy
import math
import hashlib
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(autouse=True)
def _isolate_repository_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis_path = Path(__file__).parents[1] / "analysis/h5.py"
    analysis_sha256 = hashlib.sha256(analysis_path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        "analysis.h5._analysis_provenance",
        lambda: {
            "analysis_commit": "a" * 40,
            "analysis_code_sha256": analysis_sha256,
            "analysis_code_files": [
                {
                    "path": "analysis/h5.py",
                    "sha256": analysis_sha256,
                    "head_sha256": analysis_sha256,
                    "matches_head": True,
                }
            ],
        },
    )
    monkeypatch.setattr(
        "analysis.paper._validate_repository_provenance",
        lambda _source, _analysis: None,
    )
    monkeypatch.setattr(
        "analysis.paper._validate_programmatic_replay",
        lambda _payload, _artifact_root: None,
    )
    monkeypatch.setattr(
        "analysis.paper._validate_h5_replay",
        lambda _payload, _artifact_root: None,
    )


PROVIDERS = ("deepseek", "glm", "kimi", "minimax", "doubao", "tongyi")
H5_AMENDMENT_COMMIT = "289be3bc4634336a8598ad80c0de084afdeba51d"
H5_AMENDMENT_SHA256 = (
    "3ac258fe1d819cc857162588dead3d03e0ba414771269bf04f8ce9ec0ad99260"
)
H5_AMENDMENT_COMMITTED_AT_UTC = "2026-07-13T18:59:52Z"


@pytest.fixture(autouse=True)
def _isolate_h5_amendment_git_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    def verify(_repo_root: Path, **kwargs: object) -> dict[str, object]:
        expected = {
            "commit": H5_AMENDMENT_COMMIT,
            "relative_path": "experiments/h5_analysis_plan.md",
            "sha256": H5_AMENDMENT_SHA256,
            "committed_at_utc": H5_AMENDMENT_COMMITTED_AT_UTC,
        }
        head = kwargs.pop("head", None)
        if kwargs != expected or not isinstance(head, str) or len(head) != 40:
            raise RuntimeError("fixture amendment provenance mismatch")
        return {
            "commit": kwargs["commit"],
            "path": kwargs["relative_path"],
            "sha256": kwargs["sha256"],
            "committed_at_utc": kwargs["committed_at_utc"],
            "is_ancestor": True,
            "verified": True,
        }

    monkeypatch.setattr("analysis.h5.verify_frozen_document_commit", verify)


@pytest.fixture(autouse=True)
def _isolate_collection_lock(monkeypatch: pytest.MonkeyPatch):
    import analysis.h5 as h5

    verifier = getattr(h5, "_verify_committed_collection_lock", None)
    if verifier is None:
        yield None
        return
    monkeypatch.setattr(h5, "_verify_committed_collection_lock", lambda *_args: None)
    yield verifier


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _study_fixture(tmp_path: Path, *, mapped: bool = True) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    raw = tmp_path / "sim_store"
    config = {
        "schema_version": "yher.llm_sim.config.v1",
        "run_id": "llm-personas-v1",
        "analysis_plan_commit": "1" * 40,
        "h5_analysis_plan_commit": H5_AMENDMENT_COMMIT,
        "h5_analysis_plan_sha256": H5_AMENDMENT_SHA256,
        "h5_analysis_plan_committed_at_utc": H5_AMENDMENT_COMMITTED_AT_UTC,
        "persona_seed_derivation_version": "yher-llm-persona-v2",
        "prompt_version": "yher-llm-persona-prompt-v1",
        "frozen_pre_observation_utc": H5_AMENDMENT_COMMITTED_AT_UTC,
        "study_seed": 2026071302,
        "pair_count": 25,
        "persona_count": 50,
        "arms": ["A", "B"],
        "providers": list(PROVIDERS),
        "max_items": 15,
        "minimum_complete_per_cell": 45,
        "maximum_prompt_rewrites": 1,
        "accuracy_bands": {
            "weak_upper_exclusive": 0.4,
            "strong_lower_exclusive": 0.75,
        },
        "manipulation_bootstrap": {
            "seed": 2026071303,
            "resamples": 10_000,
            "confidence_level": 0.95,
            "cluster_unit": "persona_id",
        },
        "manipulation_mapping_policy": "explicit_machine_annotation_only",
        "provider_policy": {
            "max_attempts": 3,
            "failure_threshold": 3,
            "base_backoff_seconds": 1.0,
            "max_backoff_seconds": 30.0,
            "cooldown_seconds": 120.0,
        },
    }
    config_path = repo / "experiments/config/llm_sim_v1.json"
    _write_json(config_path, config)
    code_files = {
        "experiments/llm_sim/runner.py": "# frozen runner\n",
        "experiments/s0_census.py": "# frozen census\n",
        "engine/mastery.py": "# frozen mastery\n",
        "engine/selector.py": "# frozen selector\n",
        "core/data/item_bank_v4.py": "# frozen item bank loader\n",
        "core/data/knowledge_repository.py": "# frozen knowledge repository\n",
        "core/learning/scoring.py": "# frozen scoring\n",
        "core/learning/item_catalog.py": "# frozen item catalog\n",
    }
    for relative, content in code_files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    runner_path = repo / "experiments/llm_sim/runner.py"
    h5_plan = repo / "experiments/h5_analysis_plan.md"
    h5_plan.write_text("# frozen h5 plan\n", encoding="utf-8")
    official_path = repo / "data/official.jsonl"
    official_path.parent.mkdir(parents=True, exist_ok=True)
    official_path.write_text('{"item_id":"i1"}\n', encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "h5@example.invalid")
    _git(repo, "config", "user.name", "H5 Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "frozen S2 fixture")
    git_head = _git(repo, "rev-parse", "HEAD")

    personas = []
    annotations = []
    annotation_items: dict[str, dict[str, str]] = {}
    for pair in range(25):
        for strength in ("weak", "strong"):
            persona_id = f"pair-{pair:02d}:{strength}"
            personas.append(
                {
                    "persona_id": persona_id,
                    "pair_id": f"pair-{pair:02d}",
                    "strength": strength,
                    "target_node": f"node-{pair:02d}",
                    "failure_id": f"failure-{pair:02d}",
                    "failure_cause": f"cause-{pair:02d}",
                    "failure_symptom": f"symptom-{pair:02d}",
                    "diagnostic_question": f"question-{pair:02d}",
                    "annotation_source": "kg.common_failures",
                    "seed": pair,
                }
            )
            calibration_items = []
            for position in range(1, 5):
                item_id = f"{persona_id}:cal-{position}"
                calibration_items.append(
                    {
                        "item_id": item_id,
                        "family_id": f"{persona_id}:family-{position}",
                        "answer_key": "A",
                        "target_option": "B" if mapped else None,
                        "wrong_options": ["B", "C", "D"],
                        "random_wrong_option_baseline": 1 / 3,
                        "mapping_status": (
                            "mapped" if mapped else "excluded_pre_outcome"
                        ),
                        "mapping_exclusion_reason": (
                            None if mapped else "no_mechanical_target_option_mapping"
                        ),
                    }
                )
                if mapped:
                    annotation_items[item_id] = {f"failure-{pair:02d}": "B"}
            annotations.append(
                {
                    "persona_id": persona_id,
                    "pair_id": f"pair-{pair:02d}",
                    "strength": strength,
                    "target_node": f"node-{pair:02d}",
                    "failure_id": f"failure-{pair:02d}",
                    "mapping_status": "mapped" if mapped else "excluded_pre_outcome",
                    "exclusion_reason": None if mapped else "no_machine_annotation",
                    "calibration_status": "ready",
                    "calibration_items": calibration_items,
                }
            )
    personas.sort(key=lambda row: row["persona_id"])
    annotations.sort(key=lambda row: row["persona_id"])
    personas_sha = _canonical_sha(personas)
    annotation_map = {
        "schema_version": "yher.llm_sim.annotation_map.v1",
        "items": annotation_items,
    }
    annotation_sha = _canonical_sha(annotation_map) if mapped else None
    persona_core = {
        "simulated": True,
        "run_id": config["run_id"],
        "persona_id": "llm-sim-study:persona-panel",
        "provider": "study_design",
        "model_id": "no-provider-observation",
        "record_type": "llm_sim_persona_panel",
        "frozen": True,
        "observation_started": False,
        "frozen_pre_observation_utc": config["frozen_pre_observation_utc"],
        "persona_seed_derivation_version": config["persona_seed_derivation_version"],
        "prompt_version": config["prompt_version"],
        "personas_sha256": personas_sha,
        "canonical_personas_sha256": personas_sha,
        "canonical_match": True,
        "personas": personas,
        "manipulation_panel_sha256": "pending",
    }
    panel_core = {
        "simulated": True,
        "persona_id": "llm-sim-study:manipulation-panel",
        "provider": "study_design",
        "model_id": "no-provider-observation",
        "record_type": "llm_sim_manipulation_panel",
        "schema_version": "yher.llm_sim.manipulation_panel.v1",
        "frozen": True,
        "observation_started": False,
        "study_seed": config["study_seed"],
        "personas_sha256": personas_sha,
        "annotation_map_sha256": annotation_sha,
        "annotation_map_source": "fixture" if mapped else None,
        "annotations": annotations,
    }
    panel = {**panel_core, "panel_sha256": _canonical_sha(panel_core)}
    persona_core["manipulation_panel_sha256"] = panel["panel_sha256"]
    persona_panel = {
        **persona_core,
        "persona_panel_sha256": _canonical_sha(persona_core),
    }
    _write_json(raw / "manipulation_panel.json", panel)
    _write_json(raw / "persona_panel.json", persona_panel)
    if mapped:
        annotation_snapshot = {
            "simulated": True,
            "run_id": config["run_id"],
            "persona_id": "llm-sim-study:annotation-map",
            "provider": "study_design",
            "model_id": "no-provider-observation",
            "record_type": "llm_sim_annotation_map_snapshot",
            "annotation_map_sha256": annotation_sha,
            "source_path": "fixture",
            "annotation_map": annotation_map,
            "panel_sha256": panel["panel_sha256"],
        }
        _write_json(raw / "annotation_map_snapshot.json", annotation_snapshot)

    from experiments.llm_sim.provenance import CODE_PATTERNS

    frozen_paths: set[Path] = set()
    for pattern in CODE_PATTERNS:
        matches = {path for path in repo.glob(pattern) if path.is_file()}
        assert matches, pattern
        frozen_paths.update(matches)
    code_rows = []
    for path in sorted(frozen_paths):
        relative = path.relative_to(repo).as_posix()
        sha = _file_sha(path)
        code_rows.append(
            {
                "path": relative,
                "sha256": sha,
                "head_sha256": sha,
                "matches_head": True,
            }
        )
    code_sha = _canonical_sha(
        [{"path": row["path"], "sha256": row["sha256"]} for row in code_rows]
    )
    official_inputs = {
        "catalog_item_count": 1,
        "source_files": [
            {
                "path": str(official_path.resolve()),
                "roles": ["catalog_source"],
                "bytes": official_path.stat().st_size,
                "sha256": _file_sha(official_path),
            }
        ],
    }
    preparation = {
        "simulated": True,
        "run_id": config["run_id"],
        "persona_id": "llm-sim-study:pre-observation-panel",
        "provider": "study_design",
        "model_id": "no-provider-observation",
        "record_type": "llm_sim_preparation_manifest",
        "status": "panel_frozen",
        "persona_count": 50,
        "mapped_count": 50 if mapped else 0,
        "excluded_pre_outcome_count": 0 if mapped else 50,
        "provider_observations": 0,
        "git_head": git_head,
        "code_sha256": code_sha,
        "working_code_sha256": code_sha,
        "head_code_sha256": code_sha,
        "code_matches_head": True,
        "code_files": code_rows,
        "analysis_plan_commit": config["analysis_plan_commit"],
        "analysis_plan_is_ancestor": True,
        "h5_analysis_plan_commit": config["h5_analysis_plan_commit"],
        "h5_analysis_plan_sha256": config["h5_analysis_plan_sha256"],
        "h5_analysis_plan_committed_at_utc": config[
            "h5_analysis_plan_committed_at_utc"
        ],
        "h5_analysis_plan_verified": True,
        "persona_seed_derivation_version": config["persona_seed_derivation_version"],
        "prompt_version": config["prompt_version"],
        "official_input_sha256": _canonical_sha(official_inputs),
        "official_inputs": official_inputs,
        "panel_sha256": panel["panel_sha256"],
        "persona_panel_path": "persona_panel.json",
        "persona_panel_sha256": persona_panel["persona_panel_sha256"],
        "frozen_pre_observation_utc": config["frozen_pre_observation_utc"],
        "annotation_map_sha256": panel["annotation_map_sha256"],
        "annotation_map_snapshot": "annotation_map_snapshot.json" if mapped else None,
        "config_sha256": _canonical_sha(config),
        "study_seed": config["study_seed"],
    }
    _write_json(raw / "preparation_manifest.json", preparation)
    return repo, raw


def _provider_manifest(
    raw: Path,
    provider: str,
    *,
    prompt_revision: int = 0,
    artifacts: list[dict[str, object]] | None = None,
    status: str = "complete",
) -> dict[str, object]:
    preparation = json.loads((raw / "preparation_manifest.json").read_text())
    rows = list(artifacts or [])
    manifest = {
        "simulated": True,
        "run_id": preparation["run_id"],
        "persona_id": f"llm-sim-provider:{provider}",
        "provider": provider,
        "model_id": f"{provider}-model",
        "record_type": "llm_sim_provider_manifest",
        "status": status,
        "failure_category": None,
        "prompt_revision": prompt_revision,
        "prompt_version": "yher-llm-persona-prompt-v1",
        "persona_seed_derivation_version": "yher-llm-persona-v2",
        "run_started_at_utc": "2026-07-13T19:00:00Z",
        "panel_sha256": preparation["panel_sha256"],
        "persona_panel_sha256": preparation["persona_panel_sha256"],
        "config_sha256": preparation["config_sha256"],
        "study_seed": preparation["study_seed"],
        "git_head": preparation["git_head"],
        "code_sha256": preparation["code_sha256"],
        "analysis_plan_commit": preparation["analysis_plan_commit"],
        "h5_analysis_plan_commit": preparation["h5_analysis_plan_commit"],
        "h5_analysis_plan_sha256": preparation["h5_analysis_plan_sha256"],
        "h5_analysis_plan_committed_at_utc": preparation[
            "h5_analysis_plan_committed_at_utc"
        ],
        "analysis_plan_is_ancestor": True,
        "official_input_sha256": preparation["official_input_sha256"],
        "persona_count": 50,
        "arms": ["A", "B"],
        "max_items": 15,
        "artifacts": rows,
        "artifact_aggregate_sha256": _canonical_sha(rows),
        "provider_eligibility": {},
        "reportability": {
            "formal_design_match": True,
            "cell_completed": {"A": 0, "B": 0},
        },
        "accounting": {
            "requests": 0,
            "responses": 0,
            "retries": 0,
            "failed_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_yuan": 0.0,
            "model_drift_detected": False,
            "returned_model_ids": [],
        },
    }
    suffix = f"__prompt-v{prompt_revision}" if prompt_revision else ""
    _write_json(raw / "providers" / f"{provider}{suffix}.json", manifest)
    return manifest


def _rewrite_decision(raw: Path, provider: str) -> dict[str, object]:
    preparation = json.loads((raw / "preparation_manifest.json").read_text())
    artifacts: list[object] = []
    core = {
        "simulated": True,
        "run_id": preparation["run_id"],
        "persona_id": f"llm-sim-provider:{provider}:calibration-decision",
        "provider": provider,
        "model_id": f"{provider}-model",
        "record_type": "llm_sim_calibration_decision",
        "status": "calibration_rewrite_required",
        "prompt_revision": 0,
        "prompt_version": "yher-llm-persona-prompt-v1",
        "panel_sha256": preparation["panel_sha256"],
        "persona_panel_sha256": preparation["persona_panel_sha256"],
        "config_sha256": preparation["config_sha256"],
        "study_seed": preparation["study_seed"],
        "persona_seed_derivation_version": "yher-llm-persona-v2",
        "h5_analysis_plan_commit": preparation["h5_analysis_plan_commit"],
        "h5_analysis_plan_sha256": preparation["h5_analysis_plan_sha256"],
        "h5_analysis_plan_committed_at_utc": preparation[
            "h5_analysis_plan_committed_at_utc"
        ],
        "arms": ["A", "B"],
        "max_items": 15,
        "calibration_artifacts": artifacts,
        "calibration_artifact_aggregate_sha256": _canonical_sha(artifacts),
    }
    decision = {**core, "decision_sha256": _canonical_sha(core)}
    _write_json(raw / "calibration_decisions" / f"{provider}.json", decision)
    return decision


def _artifact_row(raw: Path, relative: str, record: dict[str, object]) -> dict[str, object]:
    path = raw / relative
    _write_json(path, record)
    return {
        "path": relative,
        "sha256": _file_sha(path),
        "record_type": record["record_type"],
        "persona_id": record["persona_id"],
        "status": record["status"],
        "arm": record.get("arm"),
    }


def _rehash_provider_manifest(raw: Path, provider: str) -> dict[str, object]:
    path = raw / "providers" / f"{provider}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    _write_json(path, manifest)
    return manifest


def _replace_bound_artifact(
    raw: Path,
    provider: str,
    relative: str,
    record: dict[str, object],
) -> None:
    path = raw / relative
    _write_json(path, record)
    manifest_path = raw / "providers" / f"{provider}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(row for row in manifest["artifacts"] if row["path"] == relative)
    row.update(
        {
            "sha256": _file_sha(path),
            "record_type": record["record_type"],
            "persona_id": record["persona_id"],
            "status": record["status"],
            "arm": record.get("arm"),
        }
    )
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    _write_json(manifest_path, manifest)


def _bind_v0_rewrite(raw: Path, provider: str) -> dict[str, object]:
    decision = _rewrite_decision(raw, provider)
    path = raw / "providers" / f"{provider}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "calibration_rewrite_required"
    manifest["calibration_decision_path"] = f"calibration_decisions/{provider}.json"
    manifest["calibration_decision_sha256"] = decision["decision_sha256"]
    _write_json(path, manifest)
    return manifest


def _replace_annotation_map(raw: Path, annotation_map: dict[str, object]) -> None:
    normalized = {
        "schema_version": "yher.llm_sim.annotation_map.v1",
        "items": annotation_map.get("items", {}),
    }
    annotation_sha = _canonical_sha(normalized)
    snapshot_path = raw / "annotation_map_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["annotation_map"] = annotation_map
    snapshot["annotation_map_sha256"] = annotation_sha

    panel_path = raw / "manipulation_panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_core = {key: value for key, value in panel.items() if key != "panel_sha256"}
    panel_core["annotation_map_sha256"] = annotation_sha
    panel = {**panel_core, "panel_sha256": _canonical_sha(panel_core)}
    snapshot["panel_sha256"] = panel["panel_sha256"]
    _write_json(snapshot_path, snapshot)
    _write_json(panel_path, panel)

    persona_path = raw / "persona_panel.json"
    persona = json.loads(persona_path.read_text(encoding="utf-8"))
    persona_core = {
        key: value for key, value in persona.items() if key != "persona_panel_sha256"
    }
    persona_core["manipulation_panel_sha256"] = panel["panel_sha256"]
    persona = {
        **persona_core,
        "persona_panel_sha256": _canonical_sha(persona_core),
    }
    _write_json(persona_path, persona)

    preparation_path = raw / "preparation_manifest.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["panel_sha256"] = panel["panel_sha256"]
    preparation["persona_panel_sha256"] = persona["persona_panel_sha256"]
    preparation["annotation_map_sha256"] = annotation_sha
    _write_json(preparation_path, preparation)


def _self_attest_preparation_code_path(
    repo: Path,
    raw: Path,
    relative: str,
) -> dict[str, object]:
    preparation_path = raw / "preparation_manifest.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    row = next(item for item in preparation["code_files"] if item["path"] == relative)
    digest = _file_sha(repo / relative)
    row.update(
        {
            "sha256": digest,
            "head_sha256": digest,
            "matches_head": True,
        }
    )
    aggregate = _canonical_sha(
        [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in preparation["code_files"]
        ]
    )
    preparation["code_sha256"] = aggregate
    preparation["working_code_sha256"] = aggregate
    preparation["head_code_sha256"] = aggregate
    preparation["code_matches_head"] = True
    _write_json(preparation_path, preparation)
    return preparation


def _populate_provider(
    raw: Path,
    provider: str,
    *,
    complete_a: int = 50,
    complete_b: int = 50,
    manipulation_pass: bool = True,
    category_offset: int = 0,
) -> None:
    preparation = json.loads((raw / "preparation_manifest.json").read_text())
    persona_panel = json.loads((raw / "persona_panel.json").read_text())
    manipulation_panel = json.loads((raw / "manipulation_panel.json").read_text())
    annotations = {
        row["persona_id"]: row for row in manipulation_panel["annotations"]
    }
    provider_index = PROVIDERS.index(provider)
    model = f"{provider}-model"
    artifacts = []
    response_count = 0
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    for persona in persona_panel["personas"]:
        weak = persona["strength"] == "weak"
        events = []
        calibration_items = annotations[persona["persona_id"]]["calibration_items"]
        for position, calibration_item in enumerate(calibration_items, start=1):
            hit = weak and manipulation_pass
            event = {
                "simulated": True,
                "run_id": preparation["run_id"],
                "persona_id": persona["persona_id"],
                "provider": provider,
                "model_id": model,
                "record_type": "llm_sim_calibration_attempt",
                "phase": "calibration",
                "strength": persona["strength"],
                "position": position,
                "item_id": calibration_item["item_id"],
                "correct": not weak,
                "target_misconception_hit": hit if weak else None,
                "random_wrong_option_baseline": 1 / 3 if weak else None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
                "cost_yuan": 0.001,
            }
            events.append(event)
            response_count += 1
            input_tokens += 2
            output_tokens += 1
            cost += 0.001
            attempt = {
                "simulated": True,
                "run_id": preparation["run_id"],
                "persona_id": persona["persona_id"],
                "provider": provider,
                "model_id": model,
                "record_type": "llm_sim_provider_attempt",
                "schema_version": "yher.llm_sim.provider_attempt.v1",
                "status": "response",
                "failure_category": None,
                "exclusion_type": None,
                "phase": "calibration",
                "arm": None,
                "position": position,
                "item_id": calibration_item["item_id"],
                "attempt_number": 1,
                "retry_number": 0,
                "requested_model_id": model,
                "returned_model_id": model,
                "response_received": True,
                "usage": {"input_tokens": 2, "output_tokens": 1},
                "cost_yuan": 0.001,
                "panel_sha256": preparation["panel_sha256"],
                "prompt_version": "yher-llm-persona-prompt-v1",
                "prompt_revision": 0,
                "run_started_at_utc": "2026-07-13T19:00:00Z",
            }
            artifacts.append(
                _artifact_row(
                    raw,
                    (
                        f"attempts/{provider}/{persona['persona_id']}"
                        f"__calibration-{position:03d}__attempt-001.json"
                    ),
                    attempt,
                )
            )
        record = {
            "simulated": True,
            "run_id": preparation["run_id"],
            "persona_id": persona["persona_id"],
            "provider": provider,
            "model_id": model,
            "record_type": "llm_sim_calibration",
            "status": "complete",
            "strength": persona["strength"],
            "events": events,
            "panel_sha256": preparation["panel_sha256"],
            "prompt_revision": 0,
        }
        artifacts.append(
            _artifact_row(
                raw,
                f"calibration/{provider}/{persona['persona_id']}.json",
                record,
            )
        )

    states = ("M", "P", "C", "U")
    beliefs = {
        "M": [0.7, 0.1, 0.1, 0.1],
        "P": [0.1, 0.7, 0.1, 0.1],
        "C": [0.1, 0.1, 0.7, 0.1],
        "U": [0.1, 0.1, 0.1, 0.7],
    }
    uniform = [0.25, 0.25, 0.25, 0.25]
    for persona_index, persona in enumerate(persona_panel["personas"]):
        for arm, complete_count in (("A", complete_a), ("B", complete_b)):
            is_complete = persona_index < complete_count
            category = states[
                (
                    persona_index
                    + (arm == "B")
                    + provider_index % 2
                    + category_offset
                )
                % 4
            ]
            event_count = 3 if is_complete else 2
            events = []
            for position in range(1, event_count + 1):
                terminal_update = is_complete and position == event_count
                item_id = f"{persona['persona_id']}:{arm}:journey-{position}"
                event = {
                    "simulated": True,
                    "run_id": preparation["run_id"],
                    "persona_id": persona["persona_id"],
                    "provider": provider,
                    "model_id": model,
                    "record_type": "llm_sim_event",
                    "pair_id": persona["pair_id"],
                    "strength": persona["strength"],
                    "target_node": persona["target_node"],
                    "failure_id": persona["failure_id"],
                    "arm": arm,
                    "position": position,
                    "item_id": item_id,
                    "role": "local",
                    "score_status": "scored",
                    "correct": True,
                    "update_applied": True,
                    "inference_likelihood": (
                        beliefs[category] if terminal_update else [1.0] * 4
                    ),
                    "prior_belief": uniform,
                    "posterior_belief": (
                        beliefs[category] if terminal_update else uniform
                    ),
                    "direct_answers_before": position - 1,
                    "direct_answers_after": position,
                    "panel_sha256": preparation["panel_sha256"],
                    "prompt_version": "yher-llm-persona-prompt-v1",
                    "prompt_revision": 0,
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                    "cost_yuan": 0.001,
                }
                events.append(event)
                response_count += 1
                input_tokens += 2
                output_tokens += 1
                cost += 0.001
                attempt = {
                    "simulated": True,
                    "run_id": preparation["run_id"],
                    "persona_id": persona["persona_id"],
                    "provider": provider,
                    "model_id": model,
                    "record_type": "llm_sim_provider_attempt",
                    "schema_version": "yher.llm_sim.provider_attempt.v1",
                    "status": "response",
                    "failure_category": None,
                    "exclusion_type": None,
                    "phase": "journey",
                    "arm": arm,
                    "position": position,
                    "item_id": item_id,
                    "attempt_number": 1,
                    "retry_number": 0,
                    "requested_model_id": model,
                    "returned_model_id": model,
                    "response_received": True,
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                    "cost_yuan": 0.001,
                    "panel_sha256": preparation["panel_sha256"],
                    "prompt_version": "yher-llm-persona-prompt-v1",
                    "prompt_revision": 0,
                    "run_started_at_utc": "2026-07-13T19:00:00Z",
                }
                artifacts.append(
                    _artifact_row(
                        raw,
                        (
                            f"attempts/{provider}/{persona['persona_id']}__{arm}"
                            f"__journey-{position:03d}__attempt-001.json"
                        ),
                        attempt,
                    )
                )
            record = {
                "simulated": True,
                "run_id": preparation["run_id"],
                "persona_id": persona["persona_id"],
                "provider": provider,
                "model_id": model,
                "record_type": "llm_sim_journey",
                "status": "complete" if is_complete else "incomplete",
                "pair_id": persona["pair_id"],
                "strength": persona["strength"],
                "target_node": persona["target_node"],
                "failure_id": persona["failure_id"],
                "annotation_source": persona["annotation_source"],
                "arm": arm,
                "max_items": 15,
                "actual_administered_count": len(events),
                "terminal_reason": (
                    "confidence"
                    if is_complete
                    else "structural_failure_item_pool"
                ),
                "events": events,
                "final_belief": events[-1]["posterior_belief"],
                "panel_sha256": preparation["panel_sha256"],
                "config_sha256": preparation["config_sha256"],
                "persona_panel_sha256": preparation["persona_panel_sha256"],
                "study_seed": preparation["study_seed"],
                "analysis_plan_commit": preparation["analysis_plan_commit"],
                "prompt_version": "yher-llm-persona-prompt-v1",
                "prompt_revision": 0,
            }
            artifacts.append(
                _artifact_row(
                    raw,
                    f"journeys/{provider}/{persona['persona_id']}__{arm}.json",
                    record,
                )
            )
    manifest = _provider_manifest(raw, provider, artifacts=artifacts)
    manifest["provider_eligibility"] = {
        "weak": {
            "eligible": manipulation_pass,
            "status": "eligible" if manipulation_pass else "excluded_post_calibration",
            "accuracy_gate": {"applicable": True, "pass": True, "status": "passed"},
            "target_gate": {
                "applicable": True,
                "pass": manipulation_pass,
                "status": "passed" if manipulation_pass else "failed",
            },
        },
        "strong": {
            "eligible": True,
            "status": "eligible",
            "accuracy_gate": {"applicable": True, "pass": True, "status": "passed"},
            "target_gate": {"applicable": False, "pass": None, "status": "not_applicable"},
        },
    }
    manifest["reportability"] = {
        "formal_design_match": True,
        "formal_design_failures": [],
        "canonical_persona_panel_match": True,
        "cell_completed": {"A": complete_a, "B": complete_b},
        "eligible_cell_completed": {
            "A": complete_a if manipulation_pass else 0,
            "B": complete_b if manipulation_pass else 0,
        },
    }
    manifest["accounting"] = {
        "requests": response_count,
        "responses": response_count,
        "retries": 0,
        "failed_requests": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_yuan": round(cost, 12),
        "model_drift_detected": False,
        "returned_model_ids": [model],
    }
    _write_json(raw / "providers" / f"{provider}.json", manifest)


def _s3_contract_fixture(
    tmp_path: Path,
    *,
    status: str = "PROGRAMMATIC_COMPLETE_H5_PENDING",
) -> tuple[Path, Path, dict[str, object]]:
    from analysis.paper import (
        ITEM_TYPE_DIAGNOSTIC_IDS,
        PROGRAMMATIC_IDS,
        derive_hypothesis_branch,
    )
    from analysis.results import _contract_metric_specs
    from analysis.paper import ANALYSIS_TIMESTAMP_POLICY
    from analysis.prepare import (
        FROZEN_ANALYSIS_PLAN_COMMIT,
        FROZEN_ANALYSIS_PLAN_SHA256,
        FROZEN_CONFIG_SHA256,
        FROZEN_EXPERIMENT_TAG,
        FROZEN_MANIFEST_SHA256,
        FROZEN_RUN_ID,
        FROZEN_RUNNER_COMMIT,
    )
    from analysis.hypotheses import (
        h1_branch_reason,
        h2_branch_reason,
        h3_branch_reason,
        h4_branch_reason,
    )

    artifacts = tmp_path / "s3-artifacts"
    artifacts.mkdir()
    from analysis.results import expected_programmatic_registry_ids

    registry_by_id: dict[str, dict[str, object]] = {}
    metrics = {}
    for index, result_id in enumerate(PROGRAMMATIC_IDS):
        metric_id = _contract_metric_specs()[result_id][0]
        row = {
            "metric_id": metric_id,
            "value": 0.5 + index / 1_000,
            "numerator": 50.0,
            "denominator": 100,
            "weighting": "fixture",
            "n_target": 25,
            "n_pair": 100,
            "raw_hash": "3" * 64,
            "ci_low": 0.4,
            "ci_high": 0.6,
        }
        if result_id in ITEM_TYPE_DIAGNOSTIC_IDS:
            row["weighting"] = (
                "equal_target_then_event; misspecified_event_level; "
                "target_stratified_paired_replicate_resample; "
                "journey_cluster_preserved; bootstrap_iterations=10000; "
                "diagnostic_only_not_item_type_H1_H2_estimand"
            )
        existing = registry_by_id.get(metric_id)
        if existing is None:
            registry_by_id[metric_id] = row
        elif existing != row:
            row = existing
    for metric_id in expected_programmatic_registry_ids({9: 9, 15: 4, 25: 1}):
        registry_by_id.setdefault(
            metric_id,
            {
                "metric_id": metric_id,
                "value": 0.5,
                "numerator": 50.0,
                "denominator": 100,
                "weighting": "fixture_generated_metric",
                "n_target": 25,
                "n_pair": 100,
                "raw_hash": "3" * 64,
                "ci_low": 0.4,
                "ci_high": 0.6,
            },
        )
    registry = [registry_by_id[key] for key in sorted(registry_by_id)]
    _write_json(artifacts / "metric_registry.json", registry)
    registry_sha = _file_sha(artifacts / "metric_registry.json")
    by_id = {row["metric_id"]: row for row in registry}
    for result_id in PROGRAMMATIC_IDS:
        metric_id = _contract_metric_specs()[result_id][0]
        row = by_id[metric_id]
        interval = result_id.endswith("_CI95")
        metrics[result_id] = {
            "registry_metric_id": metric_id,
            "value": [row["ci_low"], row["ci_high"]] if interval else row["value"],
            "ci95": [row["ci_low"], row["ci_high"]],
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "weighting": row["weighting"],
            "n_target": row["n_target"],
            "n_pair": row["n_pair"],
            "raw_hash": row["raw_hash"],
            "artifact": "metric_registry.json",
            "artifact_sha256": registry_sha,
        }
    metrics.update({result_id: None for result_id in (
        "H5_QUALIFYING_PROVIDER_COUNT",
        "H5_MINIMUM_COMPLETED_PERSONAS_PER_QUALIFYING_CELL",
        "H5_WEAK_ACCURACY_GATE",
        "H5_STRONG_ACCURACY_GATE",
        "H5_MISCONCEPTION_HIT_RATE_CONTRAST",
        "H5_MISCONCEPTION_HIT_RATE_CONTRAST_CI95",
    )})
    def value(result_id: str) -> float:
        raw = metrics[result_id]["value"]
        assert not isinstance(raw, list)
        return float(raw)

    def interval(result_id: str) -> list[float]:
        raw = metrics[result_id]["value"]
        assert isinstance(raw, list)
        return [float(raw[0]), float(raw[1])]

    predicate_inputs = {
        "H1": {
            "a_rate": value("H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"),
            "a_rate_threshold": 0.5,
            "rescue_point": value("H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"),
            "rescue_ci_low": interval("H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS_CI95")[0],
            "rescue_ci_strict_threshold": 0.0,
        },
        "H2": {
            "harm_point": value("H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS"),
            "harm_ci_low": interval("H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95")[0],
            "no_harm_point": value("H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS"),
            "no_harm_ci_high": interval("H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95")[1],
            "noninferiority_margin": 0.05,
        },
        "H3": {
            "accuracy_point": value("H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET"),
            "accuracy_ci_low": interval("H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET_CI95")[0],
            "accuracy_ci_threshold": 0.0,
            "median_a": value("H3_A_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET"),
            "median_b": value("H3_B_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET"),
            "nonconvergence_encoding": 16.0,
        },
        "H4": {
            "rescue_point": value("H4_H1_RESCUE_MISSPECIFIED_B15_ELIGIBLE_STRESS"),
            "harm_point": value("H4_H2_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS"),
            "strict_direction_threshold": 0.0,
        },
    }
    hypotheses = {}
    for hypothesis in ("H1", "H2", "H3", "H4"):
        branch = derive_hypothesis_branch(hypothesis, predicate_inputs[hypothesis])
        inputs = predicate_inputs[hypothesis]
        branch_reason = {
            "H1": lambda: h1_branch_reason(
                a_rate=inputs["a_rate"],
                rescue_point=inputs["rescue_point"],
                rescue_ci_low=inputs["rescue_ci_low"],
            ),
            "H2": lambda: h2_branch_reason(
                inputs["harm_point"],
                inputs["harm_ci_low"],
                inputs["no_harm_point"],
                inputs["no_harm_ci_high"],
            ),
            "H3": lambda: h3_branch_reason(
                inputs["accuracy_point"],
                inputs["accuracy_ci_low"],
                inputs["median_a"],
                inputs["median_b"],
            ),
            "H4": lambda: h4_branch_reason(
                inputs["rescue_point"],
                inputs["harm_point"],
            ),
        }[hypothesis]()
        hypotheses[hypothesis] = {
            "analysis_status": "complete",
            "decision": branch.decision,
            "branch_reason": branch_reason,
            "predicate_inputs": predicate_inputs[hypothesis],
        }
    hypotheses["H5"] = {
        "analysis_status": "pending_input",
        "decision": None,
        "branch_reason": "validated_S2_provider_panel_not_supplied",
        "predicate_inputs": {},
    }
    figures = {}
    names = {
        "FIG_P_RESCUE": "p_rescue",
        "FIG_C_PROBE_HARM": "c_probe_harm",
        "FIG_MATCHED_VS_MISSPECIFIED": "matched_misspecified",
        "FIG_CONFUSION_MATRICES": "confusions",
        "FIG_HELDOUT_BRIER": "brier",
        "FIG_CONVERGENCE_DISTRIBUTION": "convergence",
        "FIG_MISSPECIFICATION_BY_ITEM_TYPE": "misspecification_by_item_type",
    }
    for figure_id, name in names.items():
        path = artifacts / "figures" / f"{name}.png"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode("ascii"))
        figures[figure_id] = {
            "png": {
                "path": path.relative_to(artifacts).as_posix(),
                "sha256": _file_sha(path),
            }
        }
    figures["FIG_PROVIDER_AGREEMENT"] = None
    figures["FIG_MANIPULATION_CHECKS"] = None
    source_started = "2026-07-13T13:23:07Z"
    from analysis.h5 import _analysis_provenance

    h5_analysis_provenance = _analysis_provenance()
    analysis_committed = "2026-07-14T00:00:00Z"
    analysis_commit = h5_analysis_provenance["analysis_commit"]
    analysis_code_sha = "6" * 64
    analysis_code_files = {
        "analysis/h5.py": h5_analysis_provenance["analysis_code_sha256"],
        "experiments/analysis_plan.md": FROZEN_ANALYSIS_PLAN_SHA256,
    }
    source_provenance = {
        "run_id": FROZEN_RUN_ID,
        "run_started_at_utc": source_started,
        "runner_commit": FROZEN_RUNNER_COMMIT,
        "experiment_tag": FROZEN_EXPERIMENT_TAG,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "analysis_plan_commit": FROZEN_ANALYSIS_PLAN_COMMIT,
        "analysis_plan_sha256": FROZEN_ANALYSIS_PLAN_SHA256,
    }
    analysis_provenance = {
        "analysis_commit": analysis_commit,
        "analysis_code_committed_at_utc": analysis_committed,
        "analysis_code_sha256": analysis_code_sha,
        "analysis_code_files": analysis_code_files,
    }
    provenance_core = {
        "raw_hash": "3" * 64,
        "source_run_started_at_utc": source_started,
        "analysis_code_committed_at_utc": analysis_committed,
        "analysis_timestamp_policy": ANALYSIS_TIMESTAMP_POLICY,
        "source_provenance": source_provenance,
        "analysis_provenance": analysis_provenance,
    }
    # Bind the fixture to the independently reviewed policy required by the
    # current paper provenance contract.
    policy_path = artifacts / "static_audit_policy.json"
    policy_path.write_bytes(
        (Path(__file__).parents[1] / "analysis/static_audit_policy.json").read_bytes()
    )
    static_audit_policy = {
        "path": "static_audit_policy.json",
        "sha256": _file_sha(policy_path),
    }
    conditional_metric_audit: dict[str, object] = {}
    validation = {
        "full_target_count": 27,
        "h1_h2_eligible_target_count": 23,
        "common_support_target_count_b9": 9,
        "common_support_target_count_b15": 4,
        "common_support_target_count_b25": 1,
        "valid_journey_count": 32_400,
        "structural_failure_count": 0,
        "schema_invalid_count": 0,
        "schema_invalid_reasons": {},
        "intended_journey_count": 32_400,
        "estimand_excluded_journey_count": 0,
        "estimand_exclusion_reasons": {},
        "estimand_exclusion_arms": {},
        "estimand_exclusion_targets": [],
    }
    results_document = {
        **provenance_core,
        "numeric_source": "metric_registry.json",
        "registry_metric_ids": sorted(row["metric_id"] for row in registry),
        "validation": validation,
        "conditional_metric_audit": conditional_metric_audit,
        "static_audit_policy": static_audit_policy,
    }
    _write_json(artifacts / "results.json", results_document)
    results_sha = _file_sha(artifacts / "results.json")
    artifact_document = {
        **provenance_core,
        "registry_metric_ids": sorted(row["metric_id"] for row in registry),
        "files": {
            "metric_registry.json": registry_sha,
            "results.json": results_sha,
            "static_audit_policy.json": static_audit_policy["sha256"],
        },
        "conditional_metric_audit": conditional_metric_audit,
        "static_audit_policy": static_audit_policy,
    }
    _write_json(artifacts / "artifact_manifest.json", artifact_document)
    artifact_sha = _file_sha(artifacts / "artifact_manifest.json")
    payload = {
        "schema_version": "yher.paper-results.v1",
        "status": status,
        "run_id": FROZEN_RUN_ID,
        "runner_commit": FROZEN_RUNNER_COMMIT,
        "experiment_tag": FROZEN_EXPERIMENT_TAG,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "analysis_plan_commit": FROZEN_ANALYSIS_PLAN_COMMIT,
        "analysis_plan_sha256": FROZEN_ANALYSIS_PLAN_SHA256,
        "raw_hash": "3" * 64,
        "source_run_started_at_utc": source_started,
        "analysis_code_committed_at_utc": analysis_committed,
        "analysis_timestamp_policy": ANALYSIS_TIMESTAMP_POLICY,
        "analysis_artifact": "metric_registry.json",
        "analysis_artifact_sha256": registry_sha,
        "analysis_commit": analysis_commit,
        "analysis_code_sha256": analysis_code_sha,
        "analysis_code_files": analysis_code_files,
        "results_artifact": "results.json",
        "results_artifact_sha256": results_sha,
        "artifact_manifest": "artifact_manifest.json",
        "artifact_manifest_sha256": artifact_sha,
        "conditional_metric_audit": conditional_metric_audit,
        "static_audit_policy": static_audit_policy,
        "metrics": metrics,
        "decisions": {
            **{key: hypotheses[key]["decision"] for key in ("H1", "H2", "H3", "H4")},
            "H5": None,
        },
        "decision_details": hypotheses,
        "hypotheses": hypotheses,
        "denominators": {
            **validation,
            "excluded_provider_cells": None,
            "excluded_persona_cells": None,
        },
        "figures": figures,
    }
    contract = tmp_path / "results_contract.md"
    contract.write_text(
        "# Results\n\n<!-- BEGIN S3 GENERATED RESULTS -->\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n```\n\n<!-- END S3 GENERATED RESULTS -->\n\nTail.\n",
        encoding="utf-8",
    )
    return contract, artifacts, payload


def _ratings_from_counts(rows: list[list[int]]) -> list[list[str]]:
    labels = ("A", "B", "C", "D", "E")
    return [
        [label for label, count in zip(labels, row) for _ in range(count)]
        for row in rows
    ]


def test_terminal_category_uses_production_order_and_nc_for_incomplete() -> None:
    from analysis.h5 import terminal_category

    assert terminal_category(
        {
            "status": "complete",
            "terminal_reason": "confidence",
            "final_belief": [0.1, 0.7, 0.1, 0.1],
        }
    ) == "P"
    assert terminal_category(
        {
            "status": "complete",
            "terminal_reason": "budget_exhausted",
            "final_belief": [0.5, 0.5, 0.0, 0.0],
        }
    ) == "M"
    assert terminal_category(
        {
            "status": "incomplete",
            "terminal_reason": "structural_failure_item_pool",
            "final_belief": [0.0, 0.0, 0.0, 1.0],
        }
    ) == "NC"
    with pytest.raises(ValueError, match="final_belief"):
        terminal_category(
            {
                "status": "complete",
                "terminal_reason": "confidence",
                "final_belief": [0.5, math.nan, 0.2, 0.3],
            }
        )


def test_fleiss_kappa_matches_the_published_fleiss_example() -> None:
    from analysis.h5 import fleiss_kappa

    ratings = _ratings_from_counts(
        [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7],
        ]
    )
    assert fleiss_kappa(ratings, categories=("A", "B", "C", "D", "E")) == pytest.approx(
        0.2099307, abs=1e-6
    )


def test_pairwise_cohen_kappa_uses_sklearn_and_reports_subject_count() -> None:
    from analysis.h5 import pairwise_cohen_kappa

    result = pairwise_cohen_kappa(
        {
            "deepseek": {"p1|A": "M", "p2|A": "P", "p3|A": "NC"},
            "glm": {"p1|A": "M", "p2|A": "C", "p3|A": "NC"},
        }
    )
    cell = result["deepseek"]["glm"]
    assert cell["n_subject"] == 3
    assert cell["kappa"] == pytest.approx(4 / 7)


def test_persona_cluster_bootstrap_is_seeded_and_provider_weighted_equally() -> None:
    from analysis.h5 import persona_cluster_contrast_bootstrap

    observations = {
        "deepseek": {
            "p1": {"numerator": 1.0, "denominator": 1},
            "p2": {"numerator": 0.0, "denominator": 1},
        },
        "glm": {
            "p1": {"numerator": 0.5, "denominator": 1},
            "p2": {"numerator": 0.5, "denominator": 1},
        },
    }
    first = persona_cluster_contrast_bootstrap(
        observations,
        persona_ids=("p1", "p2"),
        iterations=10_000,
        seed=2026071303,
    )
    second = persona_cluster_contrast_bootstrap(
        observations,
        persona_ids=("p1", "p2"),
        iterations=10_000,
        seed=2026071303,
    )
    assert first == second
    assert first["point"] == pytest.approx(0.5)
    assert first["ci95"] == pytest.approx([0.25, 0.75])
    assert first["iterations"] == 10_000
    assert first["seed"] == 2026071303


def test_collection_is_exact_six_provider_immutable_and_ignores_unlisted_files(
    tmp_path: Path,
) -> None:
    from analysis.h5 import finalize_collection

    repo, raw = _study_fixture(tmp_path)
    for provider in PROVIDERS:
        _provider_manifest(raw, provider)
    _provider_manifest(raw, "favorable-extra")
    output = raw / "h5_collection_manifest.json"
    first = finalize_collection(raw, output, repo_root=repo)
    before = output.read_bytes()
    second = finalize_collection(raw, output, repo_root=repo)
    assert first == second
    assert output.read_bytes() == before
    assert [row["provider"] for row in first["providers"]] == list(PROVIDERS)
    assert {row["prompt_revision"] for row in first["providers"]} == {0}
    assert [row["manifest_path"] for row in first["providers"]] == [
        f"providers/{provider}.json" for provider in PROVIDERS
    ]
    assert first["collection_sha256"]
    assert "favorable-extra" not in output.read_text(encoding="utf-8")


def test_collection_rejects_qwen_as_a_tongyi_provider_alias(tmp_path: Path) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    for provider in PROVIDERS:
        _provider_manifest(raw, provider)
    tongyi_path = raw / "providers/tongyi.json"
    tongyi = json.loads(tongyi_path.read_text(encoding="utf-8"))
    tongyi["model_id"] = "qwen-max"
    _write_json(tongyi_path, tongyi)
    accepted = finalize_collection(
        raw,
        raw / "accepted_collection_manifest.json",
        repo_root=repo,
    )
    assert accepted["providers"][-1]["provider"] == "tongyi"
    assert accepted["providers"][-1]["model_id"] == "qwen-max"

    _provider_manifest(raw, "qwen")

    with pytest.raises(H5ContractError, match="qwen.*provider alias"):
        finalize_collection(
            raw,
            raw / "h5_collection_manifest.json",
            repo_root=repo,
        )


def test_collection_selects_v1_only_from_a_valid_immutable_v0_decision(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    for provider in PROVIDERS:
        _provider_manifest(raw, provider)
    _bind_v0_rewrite(raw, "deepseek")
    _provider_manifest(raw, "deepseek", prompt_revision=1)
    _provider_manifest(raw, "glm", prompt_revision=1)
    output = raw / "h5_collection_manifest.json"
    collection = finalize_collection(raw, output, repo_root=repo)
    selected = {row["provider"]: row for row in collection["providers"]}
    assert selected["deepseek"]["prompt_revision"] == 1
    assert selected["glm"]["prompt_revision"] == 0

    decision_path = raw / "calibration_decisions/deepseek.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["status"] = "complete"
    _write_json(decision_path, decision)
    output.unlink()
    with pytest.raises(H5ContractError, match="decision"):
        finalize_collection(raw, output, repo_root=repo)


def test_collection_rejects_decision_when_v0_did_not_request_a_rewrite(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    for provider in PROVIDERS:
        _provider_manifest(raw, provider)
    _rewrite_decision(raw, "deepseek")
    _provider_manifest(raw, "deepseek", prompt_revision=1)

    with pytest.raises(H5ContractError, match="v0.*rewrite|rewrite.*v0"):
        finalize_collection(raw, raw / "collection.json", repo_root=repo)


def test_collection_rejects_rewrite_decision_prompt_provenance_drift(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    for provider in PROVIDERS:
        _provider_manifest(raw, provider)
    _bind_v0_rewrite(raw, "deepseek")
    decision_path = raw / "calibration_decisions/deepseek.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["prompt_version"] = "post-outcome-prompt"
    core = {key: value for key, value in decision.items() if key != "decision_sha256"}
    decision["decision_sha256"] = _canonical_sha(core)
    _write_json(decision_path, decision)
    v0_path = raw / "providers/deepseek.json"
    v0 = json.loads(v0_path.read_text(encoding="utf-8"))
    v0["calibration_decision_sha256"] = decision["decision_sha256"]
    _write_json(v0_path, v0)

    with pytest.raises(H5ContractError, match="provenance"):
        finalize_collection(raw, raw / "collection.json", repo_root=repo)


@pytest.mark.parametrize("tamper", ("v0", "decision"))
def test_missing_required_revision_revalidates_bound_v0_and_decision(
    tmp_path: Path,
    tamper: str,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    for provider in PROVIDERS:
        _provider_manifest(raw, provider)
    _bind_v0_rewrite(raw, "deepseek")
    collection = raw / "collection.json"
    frozen = finalize_collection(raw, collection, repo_root=repo)
    assert frozen["providers"][0]["collection_status"] == "missing_required_revision"
    target = (
        raw / "providers/deepseek.json"
        if tamper == "v0"
        else raw / "calibration_decisions/deepseek.json"
    )
    value = json.loads(target.read_text(encoding="utf-8"))
    value["model_id"] = "tampered-model"
    if tamper == "decision":
        core = {key: child for key, child in value.items() if key != "decision_sha256"}
        value["decision_sha256"] = _canonical_sha(core)
    _write_json(target, value)

    with pytest.raises(H5ContractError, match="hash|model|rewrite"):
        analyze_collection(
            collection,
            tmp_path / "h5-output",
            raw_root=raw,
            repo_root=repo,
        )


@pytest.mark.parametrize("failure", ("escape", "hash", "envelope", "drift"))
def test_collection_rejects_artifact_escape_tamper_envelope_and_model_drift(
    tmp_path: Path,
    failure: str,
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    artifact = {
        "simulated": True,
        "run_id": "llm-personas-v1",
        "persona_id": "pair-00:weak",
        "provider": "deepseek",
        "model_id": "deepseek-model",
        "record_type": "llm_sim_journey",
        "status": "incomplete",
        "arm": "A",
        "max_items": 15,
        "actual_administered_count": 0,
        "terminal_reason": "structural_failure_no_items",
        "events": [],
        "final_belief": [0.25, 0.25, 0.25, 0.25],
    }
    artifact_path = raw / "journeys/deepseek/journey.json"
    if failure == "envelope":
        artifact["simulated"] = False
    if failure == "drift":
        artifact["model_id"] = "silent-other-model"
    _write_json(artifact_path, artifact)
    row = {
        "path": "journeys/deepseek/journey.json",
        "sha256": _file_sha(artifact_path),
        "record_type": "llm_sim_journey",
        "persona_id": "pair-00:weak",
        "status": "complete",
        "arm": "A",
    }
    if failure == "escape":
        row["path"] = "../outside.json"
        outside = raw.parent / "outside.json"
        outside.write_bytes(artifact_path.read_bytes())
    _provider_manifest(raw, "deepseek", artifacts=[row])
    for provider in PROVIDERS[1:]:
        _provider_manifest(raw, provider)
    if failure == "hash":
        artifact["terminal_reason"] = "tampered_without_rehash"
        _write_json(artifact_path, artifact)
    with pytest.raises(H5ContractError, match={
        "escape": "path",
        "hash": "hash",
        "envelope": "envelope",
        "drift": "drift",
    }[failure]):
        finalize_collection(
            raw,
            raw / "h5_collection_manifest.json",
            repo_root=repo,
        )


def test_collection_rejects_fully_rehashed_pre_amendment_config_and_panel(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    config_path = repo / "experiments/config/llm_sim_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["persona_seed_derivation_version"] = "yher-llm-persona-v1"
    config["frozen_pre_observation_utc"] = "2026-07-13T14:53:23Z"
    _write_json(config_path, config)

    persona_path = raw / "persona_panel.json"
    persona = json.loads(persona_path.read_text(encoding="utf-8"))
    persona["persona_seed_derivation_version"] = "yher-llm-persona-v1"
    persona["frozen_pre_observation_utc"] = config["frozen_pre_observation_utc"]
    persona_core = {
        key: value for key, value in persona.items() if key != "persona_panel_sha256"
    }
    persona["persona_panel_sha256"] = _canonical_sha(persona_core)
    _write_json(persona_path, persona)

    preparation = _self_attest_preparation_code_path(
        repo,
        raw,
        "experiments/config/llm_sim_v1.json",
    )
    preparation["persona_seed_derivation_version"] = "yher-llm-persona-v1"
    preparation["frozen_pre_observation_utc"] = config["frozen_pre_observation_utc"]
    preparation["config_sha256"] = _canonical_sha(config)
    preparation["persona_panel_sha256"] = persona["persona_panel_sha256"]
    _write_json(raw / "preparation_manifest.json", preparation)

    with pytest.raises(H5ContractError, match="frozen S2 config|frozen amendment"):
        finalize_collection(raw, raw / "collection.json", repo_root=repo)


def test_collection_rejects_self_attested_code_head_hash(tmp_path: Path) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    (repo / "experiments/llm_sim/runner.py").write_text(
        "# post-freeze forged runner\n",
        encoding="utf-8",
    )
    _self_attest_preparation_code_path(
        repo,
        raw,
        "experiments/llm_sim/runner.py",
    )

    with pytest.raises(H5ContractError, match="committed|git|HEAD"):
        finalize_collection(raw, raw / "collection.json", repo_root=repo)


def test_collection_rejects_incomplete_code_inventory(tmp_path: Path) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    preparation_path = raw / "preparation_manifest.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["code_files"] = [
        row
        for row in preparation["code_files"]
        if row["path"] != "engine/selector.py"
    ]
    preparation["code_sha256"] = _canonical_sha(
        [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in preparation["code_files"]
        ]
    )
    _write_json(preparation_path, preparation)

    with pytest.raises(H5ContractError, match="exact frozen scope"):
        finalize_collection(raw, raw / "collection.json", repo_root=repo)


@pytest.mark.parametrize(
    "run_started_at_utc",
    ("2026-07-13T18:59:51Z", H5_AMENDMENT_COMMITTED_AT_UTC),
)
def test_collection_rejects_provider_not_strictly_after_freeze(
    tmp_path: Path,
    run_started_at_utc: str,
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    manifest = _provider_manifest(raw, "deepseek")
    manifest["run_started_at_utc"] = run_started_at_utc
    _write_json(raw / "providers/deepseek.json", manifest)

    with pytest.raises(H5ContractError, match="run_started_at_utc|freeze|chronology"):
        finalize_collection(raw, raw / "collection.json", repo_root=repo)


def test_committed_collection_lock_rejects_self_consistent_outcome_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_collection_lock,
) -> None:
    import analysis.h5 as h5

    assert callable(_isolate_collection_lock), "committed collection lock verifier missing"
    monkeypatch.setattr(
        h5,
        "_verify_committed_collection_lock",
        _isolate_collection_lock,
    )
    repo, raw = _study_fixture(tmp_path)
    _populate_provider(raw, "deepseek", category_offset=0)
    lock_path = repo / "experiments/config/h5_collection_lock.json"
    h5.write_collection_lock(raw, lock_path, repo_root=repo)
    _git(repo, "add", lock_path.relative_to(repo).as_posix())
    _git(repo, "commit", "-qm", "lock H5 provider collection")
    collection_path = raw / "collection.json"
    first = h5.finalize_collection(raw, collection_path, repo_root=repo)

    collection_path.unlink()
    _populate_provider(raw, "deepseek", category_offset=1)
    forged = h5._build_collection(raw, repo)
    _write_json(lock_path, h5._collection_lock_value(forged))
    _git(repo, "add", lock_path.relative_to(repo).as_posix())
    _git(repo, "commit", "-qm", "attempt to replace locked H5 outcomes")
    with pytest.raises(h5.H5ContractError, match="collection lock"):
        h5.finalize_collection(raw, collection_path, repo_root=repo)
    assert first["collection_sha256"] != forged["collection_sha256"]


def test_collection_lock_must_be_committed_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolate_collection_lock,
) -> None:
    import analysis.h5 as h5

    assert callable(_isolate_collection_lock)
    monkeypatch.setattr(
        h5,
        "_verify_committed_collection_lock",
        _isolate_collection_lock,
    )
    repo, raw = _study_fixture(tmp_path)
    lock_path = repo / h5.COLLECTION_LOCK_RELATIVE
    h5.write_collection_lock(raw, lock_path, repo_root=repo)

    with pytest.raises(h5.H5ContractError, match="git provenance|collection lock"):
        h5.finalize_collection(raw, raw / "collection.json", repo_root=repo)


def test_analyzer_reverifies_supplied_provenance_and_embeds_h5_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=False)
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)
    supplied = {
        "analysis_commit": "b" * 40,
        "analysis_code_committed_at_utc": "2026-07-14T00:00:00Z",
        "analysis_code_sha256": "c" * 64,
        "analysis_code_files": {
            "analysis/h5.py": "d" * 64,
            "analysis/paper.py": "e" * 64,
        },
    }
    verified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "analysis.h5.verify_analysis_provenance",
        lambda _repo, value: verified.append(dict(value)),
    )

    result = analyze_collection(
        collection,
        tmp_path / "h5-output",
        raw_root=raw,
        repo_root=repo,
        verified_analysis_provenance=supplied,
    )

    assert verified == [supplied]
    assert result["provenance"]["analysis"] == {
        "analysis_commit": supplied["analysis_commit"],
        "analysis_code_sha256": supplied["analysis_code_files"]["analysis/h5.py"],
        "analysis_code_files": [
            {
                "path": "analysis/h5.py",
                "sha256": supplied["analysis_code_files"]["analysis/h5.py"],
                "head_sha256": supplied["analysis_code_files"]["analysis/h5.py"],
                "matches_head": True,
            }
        ],
    }


def test_analyzer_rejects_forged_supplied_analysis_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from analysis.dataset import DatasetContractError
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=False)
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)

    def reject(_repo: Path, _value: object) -> None:
        raise DatasetContractError("forged supplied provenance")

    monkeypatch.setattr("analysis.h5.verify_analysis_provenance", reject)
    with pytest.raises(H5ContractError, match="analysis provenance"):
        analyze_collection(
            collection,
            tmp_path / "h5-output",
            raw_root=raw,
            repo_root=repo,
            verified_analysis_provenance={
                "analysis_commit": "f" * 40,
                "analysis_code_committed_at_utc": "2026-07-14T00:00:00Z",
                "analysis_code_sha256": "e" * 64,
                "analysis_code_files": {"analysis/h5.py": "d" * 64},
            },
        )


def test_collection_rejects_preparation_code_or_official_input_mutation(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)
    for provider in PROVIDERS:
        _provider_manifest(raw, provider)
    (repo / "experiments/llm_sim/runner.py").write_text(
        "# mutated runner\n", encoding="utf-8"
    )
    with pytest.raises(H5ContractError, match="code.*hash"):
        finalize_collection(
            raw,
            raw / "h5_collection_manifest.json",
            repo_root=repo,
        )


def test_collection_rejects_unverified_h5_amendment_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path)

    def reject(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("git proof failed")

    monkeypatch.setattr("analysis.h5.verify_frozen_document_commit", reject)
    with pytest.raises(H5ContractError, match="H5 amendment provenance"):
        finalize_collection(
            raw,
            raw / "h5_collection_manifest.json",
            repo_root=repo,
        )


@pytest.mark.parametrize("tamper", ("mass", "transition", "final"))
def test_collection_rejects_rehashed_journey_belief_semantic_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    from engine import mastery
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    _populate_provider(raw, "deepseek")
    for provider in PROVIDERS[1:]:
        _provider_manifest(raw, provider)
    manifest_path = raw / "providers/deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(
        row
        for row in manifest["artifacts"]
        if row["record_type"] == "llm_sim_journey"
    )
    path = raw / row["path"]
    journey = json.loads(path.read_text(encoding="utf-8"))
    node = mastery.NodeBelief(mastery.UNIFORM.copy())
    prior = mastery.get_belief(node, 1.0).tolist()
    likelihood = [0.9, 0.2, 0.1, 0.1]
    mastery.observe(node, likelihood, 1.0, is_direct=True)
    posterior = mastery.get_belief(node, 1.0).tolist()
    journey["events"] = [
        {
            "role": "local",
            "prior_belief": prior,
            "inference_likelihood": likelihood,
            "posterior_belief": posterior,
            "update_applied": True,
            "direct_answers_before": 0,
            "direct_answers_after": 1,
        }
    ]
    journey["actual_administered_count"] = 1
    journey["final_belief"] = posterior
    if tamper == "mass":
        journey["final_belief"] = [2.0, 0.0, 0.0, 0.0]
    elif tamper == "transition":
        journey["events"][0]["prior_belief"] = [0.4, 0.2, 0.2, 0.2]
    else:
        journey["final_belief"] = prior
    _write_json(path, journey)
    row["sha256"] = _file_sha(path)
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    _write_json(manifest_path, manifest)

    with pytest.raises(H5ContractError, match="belief|transition|journey semantics"):
        finalize_collection(raw, raw / "collection.json", repo_root=repo)


@pytest.mark.parametrize(
    "tamper",
    (
        "delete_all_complete_events",
        "count_mismatch",
        "no_items_with_events",
        "item_pool_without_events",
        "budget_not_exhausted",
    ),
)
def test_collection_rejects_rehashed_journey_terminal_count_semantic_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    from analysis.h5 import H5ContractError, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    _populate_provider(raw, "deepseek")
    for provider in PROVIDERS[1:]:
        _provider_manifest(raw, provider)
    manifest_path = raw / "providers/deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(
        row
        for row in manifest["artifacts"]
        if row["record_type"] == "llm_sim_journey"
    )
    path = raw / row["path"]
    journey = json.loads(path.read_text(encoding="utf-8"))

    if tamper == "delete_all_complete_events":
        journey["events"] = []
        journey["actual_administered_count"] = 0
        journey["final_belief"] = [0.25, 0.25, 0.25, 0.25]
    elif tamper == "count_mismatch":
        journey["actual_administered_count"] = len(journey["events"]) - 1
    elif tamper == "no_items_with_events":
        journey["status"] = "incomplete"
        journey["terminal_reason"] = "structural_failure_no_items"
    elif tamper == "item_pool_without_events":
        journey["status"] = "incomplete"
        journey["terminal_reason"] = "structural_failure_item_pool"
        journey["events"] = []
        journey["actual_administered_count"] = 0
        journey["final_belief"] = [0.25, 0.25, 0.25, 0.25]
    else:
        journey["terminal_reason"] = "budget_exhausted"

    _write_json(path, journey)
    row["sha256"] = _file_sha(path)
    row["status"] = journey["status"]
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    _write_json(manifest_path, manifest)

    with pytest.raises(H5ContractError, match="journey.*semantics|terminal|count"):
        finalize_collection(raw, raw / "collection.json", repo_root=repo)


def test_analyzer_reports_pre_outcome_mapping_exclusion_and_still_draws_figures(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=False)
    collection_path = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection_path, repo_root=repo)
    output = tmp_path / "h5-results"
    first = analyze_collection(
        collection_path,
        output,
        raw_root=raw,
        repo_root=repo,
    )
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    second = analyze_collection(
        collection_path,
        output,
        raw_root=raw,
        repo_root=repo,
    )
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert first == second
    assert before == after
    assert first["hypothesis"]["analysis_status"] == "excluded_pre_outcome"
    assert first["hypothesis"]["decision"] is None
    assert first["hypothesis"]["predicate_inputs"]["annotation_map_sha256"] is None
    assert first["hypothesis"]["predicate_inputs"]["collection_sha256"]
    for name in ("provider_agreement", "manipulation_checks"):
        png = output / "figures" / f"{name}.png"
        svg = output / "figures" / f"{name}.svg"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert png.stat().st_size > 1_000
        assert b"<svg" in svg.read_bytes()[:500]
        assert svg.stat().st_size > 1_000


def test_annotation_snapshot_is_normalized_and_partial_coverage_is_pre_outcome(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    snapshot = json.loads(
        (raw / "annotation_map_snapshot.json").read_text(encoding="utf-8")
    )
    items = dict(snapshot["annotation_map"]["items"])
    items.pop(next(iter(items)))
    _replace_annotation_map(raw, {"items": items})
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)
    result = analyze_collection(
        collection, tmp_path / "h5-output", raw_root=raw, repo_root=repo
    )

    assert result["status"] == "excluded_pre_outcome"
    assert result["hypothesis"]["decision"] is None
    coverage = result["mapping_coverage"]
    assert coverage["declared_mapped_personas"] == 50
    assert coverage["required_entries"] == 200
    assert coverage["covered_entries"] == 199
    assert coverage["complete"] is False


def test_analyzer_rejects_nonallowlisted_stale_output_without_deleting_it(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=False)
    collection = tmp_path / "h5-output/collection.json"
    collection.parent.mkdir(parents=True)
    finalize_collection(raw, collection, repo_root=repo)
    stale = collection.parent / "stale.json"
    stale.write_text("{}\n", encoding="utf-8")

    with pytest.raises(H5ContractError, match="stale|unexpected"):
        analyze_collection(
            collection,
            collection.parent,
            raw_root=raw,
            repo_root=repo,
        )
    assert stale.is_file()


def test_analyzer_validates_full_grid_gates_agreement_bootstrap_and_ledger(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    attempts = []
    for provider in PROVIDERS:
        manifest = json.loads(
            (raw / f"providers/{provider}.json").read_text(encoding="utf-8")
        )
        attempts.extend(
            json.loads((raw / row["path"]).read_text(encoding="utf-8"))
            for row in manifest["artifacts"]
            if row["record_type"] == "llm_sim_provider_attempt"
        )
    manual_totals = {
        "requests": len(attempts),
        "responses": sum(row["response_received"] is True for row in attempts),
        "retries": sum(int(row["retry_number"]) > 0 for row in attempts),
        "input_tokens": sum(int(row["usage"]["input_tokens"]) for row in attempts),
        "output_tokens": sum(int(row["usage"]["output_tokens"]) for row in attempts),
        "cost_yuan": round(sum(float(row["cost_yuan"]) for row in attempts), 12),
    }
    collection_path = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection_path, repo_root=repo)
    result = analyze_collection(
        collection_path,
        tmp_path / "h5-results",
        raw_root=raw,
        repo_root=repo,
    )
    assert result["hypothesis"]["analysis_status"] == "complete"
    assert result["hypothesis"]["decision"] == "supported"
    assert result["metrics"]["H5_QUALIFYING_PROVIDER_COUNT"]["value"] == 6
    assert result["metrics"][
        "H5_MINIMUM_COMPLETED_PERSONAS_PER_QUALIFYING_CELL"
    ]["value"] == 50
    contrast = result["metrics"]["H5_MISCONCEPTION_HIT_RATE_CONTRAST"]
    assert contrast["value"] == pytest.approx(2 / 3)
    assert contrast["ci95"][0] > 0
    assert result["agreement"]["scope"] == "qualifying_providers"
    assert result["agreement"]["fleiss_kappa"] is not None
    assert result["agreement"]["pairwise"]["deepseek"]["glm"]["n_subject"] == 100
    expected_totals = {
        "requests": 3_000,
        "responses": 3_000,
        "retries": 0,
        "input_tokens": 6_000,
        "output_tokens": 3_000,
        "cost_yuan": 3.0,
    }
    assert manual_totals == expected_totals
    assert result["ledger"]["totals"] == expected_totals
    assert {row["model_id"] for row in result["ledger"]["providers"]} == {
        f"{provider}-model" for provider in PROVIDERS
    }


def test_analyzer_excludes_incomplete_provider_and_reports_structural_denominator(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    for provider in PROVIDERS:
        _populate_provider(
            raw,
            provider,
            complete_a=44 if provider == "tongyi" else 50,
        )
    collection_path = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection_path, repo_root=repo)
    result = analyze_collection(
        collection_path,
        tmp_path / "h5-results",
        raw_root=raw,
        repo_root=repo,
    )
    assert result["metrics"]["H5_QUALIFYING_PROVIDER_COUNT"]["value"] == 5
    tongyi = next(
        row for row in result["provider_matrix"] if row["provider"] == "tongyi"
    )
    assert tongyi["qualifies"] is False
    assert "completion" in tongyi["exclusion_reasons"]
    assert result["denominators"]["structural_incomplete_journeys"] == 6


def test_typed_model_drift_is_accounted_excluded_and_never_enters_agreement(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    provider = "deepseek"
    model = "deepseek-model"
    returned = "unexpected-model"
    attempt = {
        "simulated": True,
        "run_id": "llm-personas-v1",
        "persona_id": "pair-00:weak",
        "provider": provider,
        "model_id": returned,
        "record_type": "llm_sim_provider_attempt",
        "schema_version": "yher.llm_sim.provider_attempt.v1",
        "status": "model_drift",
        "failure_category": "model_id_drift",
        "exclusion_type": "model_id_drift",
        "phase": "calibration",
        "arm": None,
        "position": 5,
        "item_id": "drift-item",
        "attempt_number": 1,
        "retry_number": 0,
        "requested_model_id": model,
        "returned_model_id": returned,
        "response_received": True,
        "usage": {"input_tokens": 7, "output_tokens": 3},
        "cost_yuan": 0.02,
        "panel_sha256": json.loads(
            (raw / "preparation_manifest.json").read_text(encoding="utf-8")
        )["panel_sha256"],
        "prompt_version": "yher-llm-persona-prompt-v1",
        "prompt_revision": 0,
        "run_started_at_utc": "2026-07-13T19:00:00Z",
    }
    attempt_row = _artifact_row(raw, "attempts/deepseek/drift.json", attempt)
    exclusion = {
        **attempt,
        "record_type": "llm_sim_excluded_response_accounting",
        "schema_version": "yher.llm_sim.model_drift_exclusion.v1",
        "status": "excluded_response",
        "requested_model": model,
        "returned_model": returned,
        "source_attempt_number": 1,
    }
    exclusion.pop("response_received")
    exclusion.pop("attempt_number")
    exclusion.pop("retry_number")
    exclusion_row = _artifact_row(
        raw, "excluded_responses/deepseek/drift.json", exclusion
    )
    manifest_path = raw / "providers/deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].extend([attempt_row, exclusion_row])
    manifest["status"] = "excluded_model_drift"
    manifest["failure_category"] = "model_id_drift"
    expected_requests = int(manifest["accounting"]["requests"]) + 1
    expected_responses = int(manifest["accounting"]["responses"]) + 1
    manifest["accounting"].update(
        {
            "requests": expected_requests,
            "responses": expected_responses,
            "failed_requests": int(manifest["accounting"]["failed_requests"]) + 1,
            "input_tokens": int(manifest["accounting"]["input_tokens"]) + 7,
            "output_tokens": int(manifest["accounting"]["output_tokens"]) + 3,
            "cost_yuan": round(float(manifest["accounting"]["cost_yuan"]) + 0.02, 12),
            "model_drift_detected": True,
            "returned_model_ids": [model, returned],
        }
    )
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    _write_json(manifest_path, manifest)

    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)
    result = analyze_collection(
        collection, tmp_path / "h5-output", raw_root=raw, repo_root=repo
    )
    deepseek = next(
        row for row in result["provider_matrix"] if row["provider"] == "deepseek"
    )
    assert deepseek["drift_count"] == 1
    assert deepseek["raw_complete"] is False
    assert "model_drift" in deepseek["exclusion_reasons"]
    assert "deepseek" not in result["agreement"]["providers"]
    ledger = next(
        row for row in result["ledger"]["providers"] if row["provider"] == "deepseek"
    )
    assert ledger["requests"] == expected_requests
    assert ledger["responses"] == expected_responses
    assert ledger["drift_count"] == 1
    assert ledger["failure_reasons"] == {"model_id_drift": 1}


def test_analyzer_reconstructs_attempt_accounting_and_rejects_claimed_request_tamper(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    manifest_path = raw / "providers/deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["accounting"]["requests"] += 1
    _write_json(manifest_path, manifest)
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)

    with pytest.raises(H5ContractError, match="requests accounting mismatch"):
        analyze_collection(
            collection, tmp_path / "h5-output", raw_root=raw, repo_root=repo
        )


def test_analyzer_rejects_outcome_event_without_manifest_bound_response_attempt(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    manifest_path = raw / "providers/deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempt = next(
        row
        for row in manifest["artifacts"]
        if row["record_type"] == "llm_sim_provider_attempt"
    )
    attempt_record = json.loads((raw / attempt["path"]).read_text(encoding="utf-8"))
    manifest["artifacts"].remove(attempt)
    usage = attempt_record["usage"]
    manifest["accounting"].update(
        {
            "requests": int(manifest["accounting"]["requests"]) - 1,
            "responses": int(manifest["accounting"]["responses"]) - 1,
            "input_tokens": int(manifest["accounting"]["input_tokens"])
            - int(usage["input_tokens"]),
            "output_tokens": int(manifest["accounting"]["output_tokens"])
            - int(usage["output_tokens"]),
            "cost_yuan": round(
                float(manifest["accounting"]["cost_yuan"])
                - float(attempt_record["cost_yuan"]),
                12,
            ),
        }
    )
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    _write_json(manifest_path, manifest)
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)

    with pytest.raises(H5ContractError, match="attempt|outcome"):
        analyze_collection(
            collection, tmp_path / "h5-output", raw_root=raw, repo_root=repo
        )
    assert attempt_record["response_received"] is True


def test_protocol_invalid_received_response_is_counted_but_technically_excluded(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    provider = "deepseek"
    manifest_path = raw / "providers/deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_responses = manifest["accounting"]["responses"]
    row = next(
        row
        for row in manifest["artifacts"]
        if row["record_type"] == "llm_sim_provider_attempt"
    )
    attempt = json.loads((raw / row["path"]).read_text(encoding="utf-8"))
    attempt.update(
        {
            "status": "protocol_failure",
            "failure_category": "protocol",
            "model_id": "missing-provider-model-id",
            "returned_model_id": None,
            "response_received": True,
        }
    )
    _write_json(raw / row["path"], attempt)
    row["sha256"] = _file_sha(raw / row["path"])
    row["status"] = "protocol_failure"
    calibration_row = next(
        value
        for value in manifest["artifacts"]
        if value["record_type"] == "llm_sim_calibration"
        and value["persona_id"] == attempt["persona_id"]
    )
    calibration = json.loads(
        (raw / calibration_row["path"]).read_text(encoding="utf-8")
    )
    calibration["events"] = [
        event
        for event in calibration["events"]
        if event["item_id"] != attempt["item_id"]
    ]
    _write_json(raw / calibration_row["path"], calibration)
    calibration_row["sha256"] = _file_sha(raw / calibration_row["path"])
    manifest["accounting"]["failed_requests"] = 1
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    _write_json(manifest_path, manifest)
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)
    result = analyze_collection(
        collection, tmp_path / "h5-output", raw_root=raw, repo_root=repo
    )
    deepseek = next(
        value for value in result["provider_matrix"] if value["provider"] == provider
    )
    ledger = next(
        value for value in result["ledger"]["providers"] if value["provider"] == provider
    )
    assert ledger["responses"] == expected_responses
    assert ledger["failed_requests"] == 1
    assert ledger["failure_reasons"] == {"protocol": 1}
    assert deepseek["technical_valid"] is False
    assert deepseek["raw_complete"] is False
    assert provider not in result["agreement"]["providers"]


@pytest.mark.parametrize("defect", ("missing", "duplicate", "wrong_order", "extra"))
def test_calibration_grid_is_rebuilt_from_frozen_panel_and_structural_defects_exclude(
    tmp_path: Path,
    defect: str,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    provider = "deepseek"
    manifest_path = raw / "providers/deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration_rows = [
        row
        for row in manifest["artifacts"]
        if row["record_type"] == "llm_sim_calibration"
    ]
    first = calibration_rows[0]
    if defect == "missing":
        manifest["artifacts"].remove(first)
    elif defect == "duplicate":
        record = json.loads((raw / first["path"]).read_text(encoding="utf-8"))
        duplicate = _artifact_row(raw, "calibration/deepseek/duplicate.json", record)
        manifest["artifacts"].append(duplicate)
    elif defect == "wrong_order":
        record = json.loads((raw / first["path"]).read_text(encoding="utf-8"))
        record["events"][0]["item_id"], record["events"][1]["item_id"] = (
            record["events"][1]["item_id"],
            record["events"][0]["item_id"],
        )
        _write_json(raw / first["path"], record)
        first["sha256"] = _file_sha(raw / first["path"])
    else:
        record = json.loads((raw / first["path"]).read_text(encoding="utf-8"))
        record["persona_id"] = "unknown-persona"
        for event in record["events"]:
            event["persona_id"] = "unknown-persona"
        extra = _artifact_row(raw, "calibration/deepseek/extra.json", record)
        manifest["artifacts"].append(extra)
    manifest["artifact_aggregate_sha256"] = _canonical_sha(manifest["artifacts"])
    _write_json(manifest_path, manifest)
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)
    result = analyze_collection(
        collection, tmp_path / "h5-output", raw_root=raw, repo_root=repo
    )
    deepseek = next(
        row for row in result["provider_matrix"] if row["provider"] == provider
    )
    assert deepseek["calibration_grid_valid"] is False
    assert deepseek["raw_complete"] is False
    assert "calibration_structure" in deepseek["exclusion_reasons"]
    assert result["denominators"]["structural_calibration_personas"] >= 1


def test_analyzer_fails_if_collection_bound_raw_manifest_mutates(tmp_path: Path) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection

    repo, raw = _study_fixture(tmp_path, mapped=True)
    for provider in PROVIDERS:
        _provider_manifest(raw, provider)
    collection_path = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection_path, repo_root=repo)
    manifest_path = raw / "providers/deepseek.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "tampered"
    _write_json(manifest_path, manifest)
    with pytest.raises(H5ContractError, match="collection.*hash|manifest.*hash"):
        analyze_collection(
            collection_path,
            tmp_path / "h5-results",
            raw_root=raw,
            repo_root=repo,
        )


def test_merge_preserves_h1_h4_and_atomically_fills_only_h5_surfaces(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection, merge_h5_results
    from analysis.paper import (
        H5_IDS,
        PROGRAMMATIC_IDS,
        _validate_contract,
        load_results_contract,
    )

    contract, artifacts, original = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=False)
    collection = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    merge_h5_results(
        contract,
        h5_root / "h5_results.json",
        artifact_root=artifacts,
    )
    first_bytes = contract.read_bytes()
    merged = load_results_contract(contract)
    for key in PROGRAMMATIC_IDS:
        for field in (
            "registry_metric_id",
            "value",
            "ci95",
            "numerator",
            "denominator",
            "weighting",
            "n_target",
            "n_pair",
        ):
            assert merged["metrics"][key][field] == original["metrics"][key][field]
        assert merged["metrics"][key]["artifact"] == merged["analysis_artifact"]
        assert merged["metrics"][key].get("raw_hash") == original["metrics"][key].get(
            "raw_hash"
        )
    assert {key: merged["decisions"][key] for key in ("H1", "H2", "H3", "H4")} == {
        key: original["decisions"][key] for key in ("H1", "H2", "H3", "H4")
    }
    assert {key: merged["hypotheses"][key] for key in ("H1", "H2", "H3", "H4")} == {
        key: original["hypotheses"][key] for key in ("H1", "H2", "H3", "H4")
    }
    assert all(merged["figures"][key] is not None for key in merged["figures"])
    assert merged["hypotheses"]["H5"]["analysis_status"] == "excluded_pre_outcome"
    assert merged["status"] == "COMPLETE_H5_EXCLUDED_PRE_OUTCOME"
    assert merged["decisions"]["H5"] is None
    assert all(merged["metrics"][result_id] is None for result_id in H5_IDS)
    assert merged["h5_results_sha256"]
    assert merged["h5_artifact_manifest_sha256"]
    for relative in (
        "agreement_matrix.json",
        "manipulation_matrix.json",
        "provider_ledger.json",
    ):
        merged_path = artifacts / "h5" / relative
        source_path = h5_root / relative
        assert merged_path.read_bytes() == source_path.read_bytes()
    _validate_contract(merged, artifacts)
    repeated = merge_h5_results(
        contract,
        h5_root / "h5_results.json",
        artifact_root=artifacts,
    )
    assert repeated == merged
    assert contract.read_bytes() == first_bytes


def test_merge_evaluated_h5_builds_a_typed_combined_registry_and_valid_contract(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection, merge_h5_results
    from analysis.paper import H5_IDS, REQUIRED_FIGURE_IDS, _validate_contract

    contract, artifacts, _ = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    collection = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    result = analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    assert result["status"] == "complete"

    merged = merge_h5_results(
        contract,
        h5_root / "h5_results.json",
        artifact_root=artifacts,
    )

    assert merged["status"] == "COMPLETE_H5_EVALUATED"
    assert merged["analysis_artifact"] == "h5/merged_metric_registry.json"
    registry = json.loads(
        (artifacts / merged["analysis_artifact"]).read_text(encoding="utf-8")
    )
    registry_by_id = {row["metric_id"]: row for row in registry}
    assert isinstance(registry_by_id["h5.weak_accuracy_gate"]["value"], bool)
    assert isinstance(registry_by_id["h5.strong_accuracy_gate"]["value"], bool)
    assert all(
        display["artifact"] == merged["analysis_artifact"]
        and display["artifact_sha256"] == merged["analysis_artifact_sha256"]
        for display in merged["metrics"].values()
        if display is not None
    )
    assert all(
        merged["metrics"][result_id]["raw_hash"]
        == merged["h5_collection_manifest_sha256"]
        for result_id in H5_IDS
    )
    assert all(
        display["raw_hash"] == merged["raw_hash"]
        for result_id, display in merged["metrics"].items()
        if result_id not in H5_IDS and display is not None
    )
    assert all(
        row["raw_hash"]
        == (
            merged["h5_collection_manifest_sha256"]
            if str(row["metric_id"]).startswith("h5.")
            else merged["raw_hash"]
        )
        for row in registry
    )
    assert merged["results_artifact"] == "h5/merged_results.json"
    assert merged["artifact_manifest"] == "h5/merged_artifact_manifest.json"
    assert all(merged["metrics"][result_id] is not None for result_id in H5_IDS)
    predicates = merged["hypotheses"]["H5"]["predicate_inputs"]
    assert predicates["qualifying_provider_count"] == merged["metrics"][
        "H5_QUALIFYING_PROVIDER_COUNT"
    ]["value"]
    assert predicates["minimum_completed_personas_per_qualifying_cell"] == merged[
        "metrics"
    ]["H5_MINIMUM_COMPLETED_PERSONAS_PER_QUALIFYING_CELL"]["value"]
    assert predicates["weak_accuracy_gate"] is merged["metrics"][
        "H5_WEAK_ACCURACY_GATE"
    ]["value"]
    assert predicates["strong_accuracy_gate"] is merged["metrics"][
        "H5_STRONG_ACCURACY_GATE"
    ]["value"]
    assert REQUIRED_FIGURE_IDS == {
        figure_id
        for figure_id, references in merged["figures"].items()
        if references is not None
    }
    _validate_contract(merged, artifacts)


def test_h5_replay_authenticates_evaluated_collection_and_rejects_hash_forgery(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection, merge_h5_results
    from analysis.paper import (
        H5_IDS,
        PaperContractError,
        _validate_replayed_h5_surface,
    )

    contract, artifacts, _ = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    collection = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection, repo_root=repo)
    replay_root = tmp_path / "h5-output"
    analyze_collection(collection, replay_root, raw_root=raw, repo_root=repo)
    merged = merge_h5_results(
        contract,
        replay_root / "h5_results.json",
        artifact_root=artifacts,
    )

    _validate_replayed_h5_surface(merged, artifacts, replay_root)

    forged = copy.deepcopy(merged)
    forged["h5_collection_manifest_sha256"] = "f" * 64
    for result_id in H5_IDS:
        forged["metrics"][result_id]["raw_hash"] = "f" * 64
    with pytest.raises(PaperContractError, match="replayed H5"):
        _validate_replayed_h5_surface(forged, artifacts, replay_root)

    forged = copy.deepcopy(merged)
    forged["analysis_commit"] = "f" * 40
    with pytest.raises(PaperContractError, match="replayed H5 analysis provenance"):
        _validate_replayed_h5_surface(forged, artifacts, replay_root)

    forged = copy.deepcopy(merged)
    registry_path = artifacts / str(forged["analysis_artifact"])
    registry_rows = json.loads(registry_path.read_text(encoding="utf-8"))
    for row in registry_rows:
        if row["metric_id"] == "h5.weak_accuracy_gate":
            row["ci_low"] = 0.25
    registry_path.write_text(
        json.dumps(registry_rows, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    forged_registry_sha = _file_sha(registry_path)
    forged["analysis_artifact_sha256"] = forged_registry_sha
    for display in forged["metrics"].values():
        if display is not None:
            display["artifact_sha256"] = forged_registry_sha
    with pytest.raises(PaperContractError, match="replayed H5 registry"):
        _validate_replayed_h5_surface(forged, artifacts, replay_root)


def test_h5_replay_authenticates_pre_outcome_exclusion_evidence(
    tmp_path: Path,
) -> None:
    from analysis.h5 import analyze_collection, finalize_collection, merge_h5_results
    from analysis.paper import PaperContractError, _validate_replayed_h5_surface

    contract, artifacts, _ = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=False)
    collection = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection, repo_root=repo)
    replay_root = tmp_path / "h5-output"
    analyze_collection(collection, replay_root, raw_root=raw, repo_root=repo)
    merged = merge_h5_results(
        contract,
        replay_root / "h5_results.json",
        artifact_root=artifacts,
    )

    _validate_replayed_h5_surface(merged, artifacts, replay_root)

    evidence = artifacts / "h5/h5_results.json"
    evidence.write_text('{"status":"excluded_pre_outcome"}\n', encoding="utf-8")
    with pytest.raises(PaperContractError, match="replayed H5 artifact"):
        _validate_replayed_h5_surface(merged, artifacts, replay_root)


@pytest.mark.parametrize(
    "status",
    (
        "PROGRAMMATIC_" + "S3_" + "COMPLETE_H5_PENDING",
        "PROGRAMMATIC_" + "S3_" + "COMPLETE_H5_PENDING_" + "INPUT",
        "S3_H5_COMPLETE",
        "S3_" + "COMPLETE_H5_EXCLUDED_PRE_OUTCOME",
    ),
)
def test_merge_rejects_every_noncanonical_programmatic_input_lifecycle(
    tmp_path: Path,
    status: str,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection, merge_h5_results

    contract, artifacts, _ = _s3_contract_fixture(tmp_path, status=status)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=False)
    collection = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    before = contract.read_bytes()

    with pytest.raises(H5ContractError, match="lifecycle"):
        merge_h5_results(
            contract,
            h5_root / "h5_results.json",
            artifact_root=artifacts,
        )

    assert contract.read_bytes() == before


def test_merge_rejects_tampered_h5_decision_before_copying_any_artifact(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection, merge_h5_results

    contract, artifacts, _ = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=True)
    for provider in PROVIDERS:
        _populate_provider(raw, provider)
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    result_path = h5_root / "h5_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["hypothesis"]["decision"] = "not_supported"
    core = {key: value for key, value in result.items() if key != "h5_results_sha256"}
    result["h5_results_sha256"] = _canonical_sha(core)
    _write_json(result_path, result)
    before_contract = contract.read_bytes()
    before_tree = {
        path.relative_to(artifacts).as_posix(): path.read_bytes()
        for path in artifacts.rglob("*")
        if path.is_file()
    }

    with pytest.raises(H5ContractError, match="decision|predicate|branch"):
        merge_h5_results(contract, result_path, artifact_root=artifacts)
    assert contract.read_bytes() == before_contract
    assert {
        path.relative_to(artifacts).as_posix(): path.read_bytes()
        for path in artifacts.rglob("*")
        if path.is_file()
    } == before_tree


def test_merge_requires_every_programmatic_figure_before_any_change(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection, merge_h5_results

    contract, artifacts, payload = _s3_contract_fixture(tmp_path)
    payload["figures"]["FIG_MISSPECIFICATION_BY_ITEM_TYPE"] = None
    contract.write_text(
        "# Results\n\n<!-- BEGIN S3 GENERATED RESULTS -->\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n```\n\n<!-- END S3 GENERATED RESULTS -->\n\nTail.\n",
        encoding="utf-8",
    )
    repo, raw = _study_fixture(tmp_path / "h5", mapped=False)
    collection = raw / "collection.json"
    finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    before = contract.read_bytes()
    with pytest.raises(H5ContractError, match="FIG_MISSPECIFICATION_BY_ITEM_TYPE"):
        merge_h5_results(
            contract, h5_root / "h5_results.json", artifact_root=artifacts
        )
    assert contract.read_bytes() == before
    assert not (artifacts / "h5").exists()


def test_merge_rolls_back_artifact_tree_when_commit_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import analysis.h5 as h5

    contract, artifacts, _ = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=False)
    collection = raw / "collection.json"
    h5.finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    h5.analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    before_contract = contract.read_bytes()
    before_tree = {
        path.relative_to(artifacts).as_posix(): path.read_bytes()
        for path in artifacts.rglob("*")
        if path.is_file()
    }
    original_replace = h5.os.replace
    commits = 0

    def fail_second_artifact(source: object, destination: object) -> None:
        nonlocal commits
        target = Path(destination)
        if target.is_relative_to(artifacts / "h5"):
            commits += 1
            if commits == 2:
                raise OSError("injected artifact commit failure")
        original_replace(source, destination)

    monkeypatch.setattr(h5.os, "replace", fail_second_artifact)
    with pytest.raises((OSError, h5.H5ContractError), match="commit failure"):
        h5.merge_h5_results(
            contract, h5_root / "h5_results.json", artifact_root=artifacts
        )
    assert contract.read_bytes() == before_contract
    assert {
        path.relative_to(artifacts).as_posix(): path.read_bytes()
        for path in artifacts.rglob("*")
        if path.is_file()
    } == before_tree


def test_merge_rejects_pending_h5_without_changing_the_programmatic_contract(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection, merge_h5_results

    contract, artifacts, _ = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=True)
    collection = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    result = analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    assert result["status"] == "pending_input"
    before = contract.read_bytes()

    with pytest.raises(H5ContractError, match="pending_input"):
        merge_h5_results(
            contract,
            h5_root / "h5_results.json",
            artifact_root=artifacts,
        )

    assert contract.read_bytes() == before


def test_merge_rejects_s3_registry_tamper_before_marker_changes(tmp_path: Path) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection, merge_h5_results

    contract, artifacts, _ = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=False)
    collection = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    before = contract.read_bytes()
    (artifacts / "metric_registry.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(H5ContractError, match="S3.*hash|registry.*hash"):
        merge_h5_results(
            contract,
            h5_root / "h5_results.json",
            artifact_root=artifacts,
        )
    assert contract.read_bytes() == before


def test_merge_rejects_h5_path_escape_and_analyzer_outputs_no_secrets(
    tmp_path: Path,
) -> None:
    from analysis.h5 import H5ContractError, analyze_collection, finalize_collection, merge_h5_results

    contract, artifacts, _ = _s3_contract_fixture(tmp_path)
    repo, raw = _study_fixture(tmp_path / "h5", mapped=False)
    collection = raw / "h5_collection_manifest.json"
    finalize_collection(raw, collection, repo_root=repo)
    h5_root = tmp_path / "h5-output"
    analyze_collection(collection, h5_root, raw_root=raw, repo_root=repo)
    result_path = h5_root / "h5_results.json"
    text = result_path.read_text(encoding="utf-8").lower()
    assert "api_key" not in text
    assert '"content"' not in text
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["figures"]["FIG_PROVIDER_AGREEMENT"]["png"]["path"] = "../outside.png"
    core = {key: value for key, value in result.items() if key != "h5_results_sha256"}
    result["h5_results_sha256"] = _canonical_sha(core)
    _write_json(result_path, result)
    with pytest.raises(H5ContractError, match="path"):
        merge_h5_results(contract, result_path, artifact_root=artifacts)


def test_h5_module_cli_exposes_lock_finalize_analyze_merge_and_run() -> None:
    from analysis.h5 import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    for command in ("lock", "finalize", "analyze", "merge", "run"):
        assert command in help_text
