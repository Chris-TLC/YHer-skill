from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import shutil
from typing import Any

import pytest


RUN_ID = "llm-personas-v2-dual"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _self_hash(payload: dict[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _judge_case_manifest(tmp_path: Path, count: int = 3) -> Path:
    from experiments.llm_sim_v2.analyze import judge_input_bytes

    cases = [
        {
            "case_id": f"case-{index:03d}",
            "judge_messages": [
                {
                    "role": "user",
                    "simulated": True,
                    "content": f"Classify opaque response {index}.",
                }
            ],
        }
        for index in range(count)
    ]
    manifest: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.judge_case_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "analysis_population": "main",
        "cases": cases,
    }
    manifest["shared_input_sha256"] = hashlib.sha256(
        judge_input_bytes(manifest)
    ).hexdigest()
    manifest["case_manifest_sha256"] = _self_hash(
        manifest, "case_manifest_sha256"
    )
    path = tmp_path / "case_manifest.json"
    _write_json(path, manifest)
    return path


def _judge_rows(case_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "output": {
                "label": "consistent",
                "error_category": "chemistry_reasoning",
                "rationale": f"Opaque rationale for {case_id}",
                "simulated": True,
            },
        }
        for case_id in case_ids
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


def _case_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return [str(row["case_id"]) for row in value["cases"]]


def _binder_fixture_writer():
    namespace = runpy.run_path(
        str(Path(__file__).with_name("test_journal_binder.py"))
    )
    return namespace["_write_persona_v2_bundle"]


def _publication_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture = _binder_fixture_writer()(tmp_path / "binder-fixture")
    result_dir = tmp_path / "w3-final"
    result_dir.mkdir()
    shutil.copy2(fixture / "analysis_results.json", result_dir)
    shutil.copy2(fixture / "artifact_manifest.json", result_dir)
    table_path = result_dir / "tables/provider_summary.csv"
    table_path.parent.mkdir()
    table_path.write_bytes(b"provider,status\ndeepseek,complete\n")
    artifact_path = result_dir / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["artifacts"].append(
        {
            "path": "tables/provider_summary.csv",
            "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            "size": table_path.stat().st_size,
        }
    )
    artifact["artifact_set_sha256"] = hashlib.sha256(
        _canonical(artifact["artifacts"])
    ).hexdigest()
    _write_json(artifact_path, artifact)

    main_root = tmp_path / "epoch-20260716" / RUN_ID / "main"
    main_root.mkdir(parents=True)
    shutil.copy2(fixture / "phase_provenance.json", main_root)

    repo = tmp_path / "repo"
    runtime_parent = repo / "experiments/llm_sim_v2"
    mapping_parent = runtime_parent / "frozen_v0"
    mapping_parent.mkdir(parents=True)
    shutil.copy2(fixture / "runtime_task_manifest.json", runtime_parent)
    shutil.copy2(
        fixture / "mapping_manifest.json",
        mapping_parent / "target_option_mapping.json",
    )
    return result_dir, main_root, repo


def test_judge_manifest_builds_analyzer_compatible_self_hashed_result(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze, publication

    case_path = _judge_case_manifest(tmp_path)
    ids = _case_ids(case_path)
    raw_path = tmp_path / "claude-raw.jsonl"
    _write_jsonl(raw_path, _judge_rows(ids))
    output = tmp_path / "claude.json"

    manifest = publication.build_judge_result_manifest(
        case_manifest_path=case_path,
        raw_jsonl_path=raw_path,
        judge="claude",
        output_path=output,
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert manifest["schema_version"] == "yher.llm_sim_v2.judge_result_manifest.v1"
    assert manifest["judge"] == "claude"
    assert [row["case_id"] for row in manifest["results"]] == ids
    assert manifest["judge_result_manifest_sha256"] == _self_hash(
        manifest, "judge_result_manifest_sha256"
    )
    ingested = analyze.ingest_judge_results(
        json.loads(case_path.read_text(encoding="utf-8")),
        {"claude": manifest, "gpt": None},
    )
    assert ingested["status"] == "partial_missing_judge"


@pytest.mark.parametrize("judge", ["Claude", "openai", "deepseek", ""])
def test_judge_manifest_rejects_non_frozen_judge(
    tmp_path: Path, judge: str
) -> None:
    from experiments.llm_sim_v2 import publication

    case_path = _judge_case_manifest(tmp_path)
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, _judge_rows(_case_ids(case_path)))

    with pytest.raises(publication.PublicationAdapterError, match="judge"):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            raw_jsonl_path=raw_path,
            judge=judge,
            output_path=tmp_path / "result.json",
        )


@pytest.mark.parametrize("mutation", ["reverse", "missing", "extra", "duplicate"])
def test_judge_manifest_requires_exact_case_order_and_complete_coverage(
    tmp_path: Path, mutation: str
) -> None:
    from experiments.llm_sim_v2 import publication

    case_path = _judge_case_manifest(tmp_path)
    ids = _case_ids(case_path)
    if mutation == "reverse":
        ids.reverse()
    elif mutation == "missing":
        ids.pop()
    elif mutation == "extra":
        ids.append("case-not-in-manifest")
    elif mutation == "duplicate":
        ids[-1] = ids[0]
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, _judge_rows(ids))
    output = tmp_path / "result.json"

    with pytest.raises(publication.PublicationAdapterError, match="case|coverage|order"):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            raw_jsonl_path=raw_path,
            judge="gpt",
            output_path=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "raw_bytes",
    [
        b'{"case_id":"case-000","case_id":"case-000","output":{}}\n',
        b'{"case_id":"case-000","output":{},"extra":true}\n',
        b'{"case_id":"case-000","output":{"label":NaN}}\n',
        b"\n",
        b"not-json\n",
    ],
)
def test_judge_manifest_strictly_rejects_ambiguous_jsonl(
    tmp_path: Path, raw_bytes: bytes
) -> None:
    from experiments.llm_sim_v2 import publication

    case_path = _judge_case_manifest(tmp_path, count=1)
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_bytes(raw_bytes)
    output = tmp_path / "result.json"

    with pytest.raises(publication.PublicationAdapterError):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            raw_jsonl_path=raw_path,
            judge="claude",
            output_path=output,
        )
    assert not output.exists()


def test_judge_manifest_calls_frozen_output_validator_and_leaves_no_output(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    case_path = _judge_case_manifest(tmp_path, count=1)
    row = _judge_rows(["case-000"])[0]
    row["output"]["authenticity"] = 1.0
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, [row])
    output = tmp_path / "result.json"

    with pytest.raises(publication.PublicationAdapterError, match="output"):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            raw_jsonl_path=raw_path,
            judge="claude",
            output_path=output,
        )
    assert not output.exists()


def test_judge_manifest_refuses_existing_or_dangling_symlink_destination(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    case_path = _judge_case_manifest(tmp_path, count=1)
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, _judge_rows(["case-000"]))
    existing = tmp_path / "existing.json"
    existing.write_text("do not replace", encoding="utf-8")
    with pytest.raises(publication.PublicationAdapterError, match="exist"):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            raw_jsonl_path=raw_path,
            judge="claude",
            output_path=existing,
        )
    assert existing.read_text(encoding="utf-8") == "do not replace"

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target")
    with pytest.raises(publication.PublicationAdapterError, match="exist|symlink"):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            raw_jsonl_path=raw_path,
            judge="claude",
            output_path=dangling,
        )
    assert dangling.is_symlink()


def test_persona_bundle_atomically_builds_binder_accepted_snapshot(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(tmp_path)
    output = tmp_path / "persona-bundle"

    manifest = publication.build_persona_v2_bundle(
        result_dir=result_dir,
        main_phase_root=main_root,
        repo_root=repo,
        output_dir=output,
    )

    assert output.is_dir()
    assert json.loads((output / "binding_manifest.json").read_text()) == manifest
    assert set(manifest["files"]) == {
        "analysis_results",
        "analysis_artifact_manifest",
        "phase_provenance",
        "runtime_task_manifest",
        "mapping_manifest",
    }
    assert (output / "analysis/analysis_results.json").read_bytes() == (
        result_dir / "analysis_results.json"
    ).read_bytes()
    copied_table = output / "analysis/tables/provider_summary.csv"
    assert copied_table.read_bytes() == b"provider,status\ndeepseek,complete\n"
    table_path = result_dir / "tables/provider_summary.csv"
    table_path.write_bytes(b"mutated source\n")
    assert copied_table.read_bytes() == b"provider,status\ndeepseek,complete\n"
    assert copied_table.stat().st_ino != table_path.stat().st_ino
    bound = journal_binder.bind_persona_v2_artifacts(output)
    assert bound["status"] == "bound_formal_w3"
    assert bound["persona_cluster_count"] == 50


def test_persona_bundle_refuses_existing_and_symlink_destinations(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "marker"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(publication.PublicationAdapterError, match="exist"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=existing,
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(publication.PublicationAdapterError, match="exist|symlink"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=dangling,
        )
    assert dangling.is_symlink()


def test_persona_bundle_exclusive_install_does_not_replace_racing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(tmp_path)
    output = tmp_path / "racing-output"
    exclusive_install = publication._install_directory_no_replace
    racing_inode: list[int] = []

    def create_racer_then_install(staging: Path, destination: Path) -> None:
        destination.mkdir()
        racing_inode.append(destination.stat().st_ino)
        exclusive_install(staging, destination)

    monkeypatch.setattr(
        publication, "_install_directory_no_replace", create_racer_then_install
    )
    with pytest.raises(publication.PublicationAdapterError, match="exist"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=output,
        )
    assert output.stat().st_ino == racing_inode[0]
    assert list(output.iterdir()) == []


def test_persona_bundle_rejects_result_not_bound_by_artifact_manifest(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(tmp_path)
    result_path = result_dir / "analysis_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["claim_boundary"] += "; drift"
    _write_json(result_path, result)
    output = tmp_path / "persona-bundle"

    with pytest.raises(publication.PublicationAdapterError, match="artifact|drift|hash"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("source", ["phase", "runtime", "mapping"])
def test_persona_bundle_rejects_identity_drift_without_partial_output(
    tmp_path: Path, source: str
) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(tmp_path)
    if source == "phase":
        path = main_root / "phase_provenance.json"
        field = "phase_provenance_sha256"
        value = json.loads(path.read_text())
        value["task_roster"]["expected_task_count"] += 1
    elif source == "runtime":
        path = repo / "experiments/llm_sim_v2/runtime_task_manifest.json"
        field = "runtime_task_manifest_sha256"
        value = json.loads(path.read_text())
        value["runtime_commit"] = "f" * 40
    else:
        path = repo / "experiments/llm_sim_v2/frozen_v0/target_option_mapping.json"
        field = None
        value = json.loads(path.read_text())
        value["rows"][0]["target_option"] = "C"
        value["mapping_sha256"] = hashlib.sha256(
            _canonical(value["rows"])
        ).hexdigest()
        mapped_targets = [
            {
                "item_id": row["item_id"],
                "failure_id": row["failure_id"],
                "target_option": row["target_option"],
            }
            for row in value["rows"]
            if row["status"] == "mapped"
        ]
        value["target_set_hash"] = hashlib.sha256(
            _canonical(mapped_targets)
        ).hexdigest()
    if field is not None:
        value[field] = _self_hash(value, field)
    _write_json(path, value)
    output = tmp_path / "persona-bundle"

    with pytest.raises(publication.PublicationAdapterError, match="identity|bind|drift|phase|runtime|mapping"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=output,
        )
    assert not output.exists()


def test_persona_bundle_rejects_unsafe_artifact_path_and_symlink(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(tmp_path)
    artifact_path = result_dir / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["artifacts"].append(
        {"path": "../escape.json", "sha256": "0" * 64, "size": 0}
    )
    artifact["artifact_set_sha256"] = hashlib.sha256(
        _canonical(artifact["artifacts"])
    ).hexdigest()
    _write_json(artifact_path, artifact)
    with pytest.raises(publication.PublicationAdapterError, match="artifact|unsafe|escape"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=tmp_path / "escape-bundle",
        )

    result_dir, main_root, repo = _publication_sources(tmp_path / "symlink-case")
    result_path = result_dir / "analysis_results.json"
    original = tmp_path / "original-results.json"
    shutil.move(result_path, original)
    result_path.symlink_to(original)
    output = tmp_path / "symlink-bundle"
    with pytest.raises(publication.PublicationAdapterError, match="symlink|regular"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=output,
        )
    assert not output.exists()


def test_persona_bundle_requires_canonical_main_phase_root(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(tmp_path)
    wrong_root = tmp_path / "main"
    shutil.copytree(main_root, wrong_root)

    with pytest.raises(publication.PublicationAdapterError, match="main|run"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=wrong_root,
            repo_root=repo,
            output_dir=tmp_path / "persona-bundle",
        )


def test_persona_bundle_rejects_unmanifested_source_file(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(tmp_path)
    (result_dir / "unbound.txt").write_text("not in artifact manifest", encoding="utf-8")
    output = tmp_path / "persona-bundle"

    with pytest.raises(publication.PublicationAdapterError, match="unbound|artifact|file set"):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=output,
        )
    assert not output.exists()


def test_publication_cli_exposes_both_offline_subcommands() -> None:
    from experiments.llm_sim_v2 import publication

    parser = publication.build_parser()
    judge = parser.parse_args(
        [
            "judge-manifest",
            "--case-manifest",
            "cases.json",
            "--raw-jsonl",
            "raw.jsonl",
            "--judge",
            "gpt",
            "--output",
            "gpt.json",
        ]
    )
    persona = parser.parse_args(
        [
            "persona-bundle",
            "--result-dir",
            "results",
            "--main-phase-root",
            "main",
            "--repo-root",
            "repo",
            "--output-dir",
            "bundle",
        ]
    )
    assert judge.command == "judge-manifest"
    assert persona.command == "persona-bundle"
