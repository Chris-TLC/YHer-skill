from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
from typing import Any

import pytest


RUN_ID = "llm-personas-v2-dual"
BASE_PERSONA_BUNDLE_ROLES = {
    "analysis_results",
    "analysis_artifact_manifest",
    "analysis_input_artifact_manifest",
    "phase_provenance",
    "runtime_task_manifest",
    "mapping_manifest",
}
JUDGE_RUN_SNAPSHOT_ROLE = "judge_run_execution_snapshot_manifest"


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
    from experiments.llm_sim_v2.analyze import build_judge_case_manifest

    candidates = []
    for index in range(count):
        question = {
            "kind": "mcq",
            "stem_blocks": [],
            "stem_text": f"Public question {index}",
            "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
            "difficulty": 0.5,
            "nodes": ["private-node"],
            "source_label": "private-source",
        }
        candidates.append(
            {
                "candidate_identity": f"provider|task-{index}",
                "stratum": "agreement",
                "public_question": question,
                "model_output": {
                    "simulated": True,
                    "answer": "A",
                    "rationale": f"Candidate rationale {index}",
                    "abstain": False,
                },
                "persona": {
                    "persona_id": f"private-{index}",
                    "target_node": "private-node",
                },
                "item": {
                    "item_id": f"item-{index}",
                    "public_question": question,
                    "options": question["options"],
                },
            }
        )
    manifest = build_judge_case_manifest(
        candidates, frozen_leakage_lexicon=()
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
                "error_category": "none",
                "rationale": f"Opaque rationale for {case_id}",
                "simulated": True,
            },
        }
        for case_id in case_ids
    ]


def _case_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return [str(row["case_id"]) for row in value["cases"]]


def _mint_publication_test_budget_authority(
    output_root: Path, case_manifest: dict[str, Any], *, judge: str
) -> None:
    from experiments.llm_sim_v2 import judge_execution

    support = case_manifest["case_manifest_sha256"][:12]
    repo = output_root.parent / f".{output_root.name}-{judge}-{support}-authority-repo"
    anchor = (
        repo
        / "experiments/llm_sim_v2/evidence_anchors/"
        "main_phase_evidence_receipt.json"
    )
    anchor.parent.mkdir(parents=True)
    phase_receipt: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.phase_evidence_receipt.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "main",
        "authority": "post_invocation_phase_receipt",
        "phase_provenance_sha256": "a" * 64,
        "phase_provenance_file_sha256": "b" * 64,
        "store_snapshot": {"snapshot_sha256": "c" * 64},
        "providers": {},
    }
    phase_receipt["phase_evidence_receipt_sha256"] = hashlib.sha256(
        _canonical(phase_receipt)
    ).hexdigest()
    anchor.write_bytes(_canonical(phase_receipt) + b"\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "publication-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Publication Test"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "bind formal collection receipt"],
        cwd=repo,
        check=True,
    )
    ledger_path = repo.parent / f".{repo.name}-run-budget-ledger.json"
    ledger = {
        "schema_version": "yher.llm_sim_v2.run_budget_ledger.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "total_known_cost_yuan": 0.0,
        "total_unknown_reserve_yuan": 0.0,
        "total_accounted_cost_yuan": 0.0,
        "hard_fuse_yuan": 450.0,
        "updated_at_utc": "2026-07-16T00:00:00Z",
    }
    ledger_path.write_bytes(_canonical(ledger) + b"\n")
    judge_execution.bind_prepared_judge_case_manifest(
        case_manifest=case_manifest,
        output_root=output_root,
    )
    judge_execution.mint_judge_budget_authority(
        case_manifest=case_manifest,
        output_root=output_root,
        repo_root=repo,
        run_budget_ledger=ledger_path,
    )


def _judge_execution_receipt(
    tmp_path: Path, case_path: Path, *, judge: str
) -> Path:
    from experiments.llm_sim_v2.judge_execution import (
        ClaudeCLIJudgeTransport,
        CodexCLIJudgeTransport,
        canonical_json_bytes,
        execute_judge_pass,
    )

    case_manifest = json.loads(case_path.read_text(encoding="utf-8"))
    _mint_publication_test_budget_authority(
        tmp_path, case_manifest, judge=judge
    )
    if judge == "claude":
        from experiments.llm_sim_v2.judge_execution import (
            record_judge_family_disposition,
        )

        record_judge_family_disposition(
            case_manifest=case_manifest,
            output_root=tmp_path,
            judge_family="gpt",
            status="unavailable",
            reason_code="production_cli_unavailable",
        )
    model = f"{judge}-test-model"
    results = _judge_rows(_case_ids(case_path))
    binary_dir = case_path.parent / f".{judge}-publication-test-bin"
    binary_dir.mkdir()
    binary = binary_dir / judge.replace("gpt", "codex")
    version = "Claude Code 1.0.0" if judge == "claude" else "codex-cli 0.0.0"
    binary.write_text(
        f"#!/bin/sh\nprintf '%s\\n' '{version}'\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    if judge == "claude":
        stdout = canonical_json_bytes(
            {
                "result": json.dumps({"results": results}, ensure_ascii=False),
                "session_id": "claude-publication-test-session",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "modelUsage": {
                    model: {"inputTokens": 10, "outputTokens": 5}
                },
            }
        )
        transport = ClaudeCLIJudgeTransport(
            binary=str(binary),
        )
    else:
        stdout = b"".join(
            canonical_json_bytes(event) + b"\n"
            for event in [
                {
                    "type": "thread.started",
                    "thread_id": "codex-publication-test-thread",
                },
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": json.dumps({"results": results}, ensure_ascii=False),
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            ]
        )
        transport = CodexCLIJudgeTransport(
            binary=str(binary),
        )
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=b""
    )
    transport._run = lambda argv, prompt: completed  # type: ignore[method-assign]
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{binary_dir}{os.pathsep}{old_path}"
    return execute_judge_pass(
        case_manifest=case_manifest,
        output_root=tmp_path,
        judge_family=judge,
        exact_model=model,
        transport=transport,
    )


def _binder_fixture_namespace() -> dict[str, Any]:
    return runpy.run_path(
        str(Path(__file__).with_name("test_journal_binder.py"))
    )


def _binder_fixture_writer():
    return _binder_fixture_namespace()["_write_persona_v2_bundle"]


def _publication_sources(
    tmp_path: Path,
    *,
    judge_profile: str = "both_complete",
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if judge_profile not in {
        "both_complete",
        "gpt_only",
        "all_missing",
        "zero_cases",
    }:
        raise ValueError("unsupported publication fixture profile")
    fixture = _binder_fixture_writer()(
        tmp_path / "binder-fixture", judge_profile=judge_profile
    )
    result_dir = tmp_path / "w3-final"
    result_dir.mkdir()
    fixture_analysis = fixture / "analysis"
    fixture_manifest_path = fixture_analysis / "artifact_manifest.json"
    fixture_manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    for row in fixture_manifest["artifacts"]:
        relative = Path(str(row["path"]))
        destination = result_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture_analysis / relative, destination)
    fixture_manifest["artifact_set_sha256"] = hashlib.sha256(
        _canonical(fixture_manifest["artifacts"])
    ).hexdigest()
    shutil.copy2(fixture_manifest_path, result_dir)
    _write_json(result_dir / "artifact_manifest.json", fixture_manifest)
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
    evidence = fixture / "evidence"
    shutil.copy2(evidence / "phase_provenance.json", main_root)

    repo = tmp_path / "repo"
    runtime_parent = repo / "experiments/llm_sim_v2"
    mapping_parent = runtime_parent / "frozen_v0"
    mapping_parent.mkdir(parents=True)
    shutil.copy2(evidence / "runtime_task_manifest.json", runtime_parent)
    shutil.copy2(
        evidence / "target_option_mapping.json",
        mapping_parent / "target_option_mapping.json",
    )
    return result_dir, main_root, repo


def _add_manifested_artifact(
    result_dir: Path,
    relative: str,
    data: bytes = b"stray\n",
) -> None:
    path = result_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    artifact_path = result_dir / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["artifacts"].append(
        {
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
    )
    artifact["artifact_set_sha256"] = hashlib.sha256(
        _canonical(artifact["artifacts"])
    ).hexdigest()
    _write_json(artifact_path, artifact)


def _full_journal_binder(tmp_path: Path, *, persona_bundle: Path) -> dict[str, Any]:
    from experiments import journal_binder

    namespace = _binder_fixture_namespace()
    h1_h4 = namespace["_write_complete_h1_h4_bundle"](tmp_path / "h1-h4")
    p2 = namespace["_write_p2"](tmp_path / "p2", raw_manifest=h1_h4["raw"])
    return journal_binder.build_binder(
        raw_manifest_path=h1_h4["raw"],
        registry_path=h1_h4["registry"],
        results_path=h1_h4["results"],
        persona_v2_dir=persona_bundle,
        p2_dir=p2,
        require_complete=True,
        allow_fixture=True,
    )


def test_judge_manifest_builds_analyzer_compatible_self_hashed_result(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze, publication

    case_path = _judge_case_manifest(tmp_path / "case-source")
    ids = _case_ids(case_path)
    run_root = tmp_path / "judge-run"
    output = run_root / "claude.json"

    manifest = publication.build_judge_result_manifest(
        case_manifest_path=case_path,
        execution_receipt_path=_judge_execution_receipt(
            run_root, case_path, judge="claude"
        ),
        output_path=output,
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert manifest["schema_version"] == "yher.llm_sim_v2.judge_result_manifest.v2"
    assert manifest["judge"] == "claude"
    assert manifest["execution_receipt"]["identity"]["transport"] == "claude_cli"
    execution_id = manifest["execution_receipt"]["identity"]["execution_id"]
    assert manifest["execution_receipt_path"] == (
        f"executions/claude/{execution_id}/execution_receipt.json"
    )
    assert [row["case_id"] for row in manifest["results"]] == ids
    assert manifest["judge_result_manifest_sha256"] == _self_hash(
        manifest, "judge_result_manifest_sha256"
    )
    ingested = analyze.ingest_judge_results(
        json.loads(case_path.read_text(encoding="utf-8")),
        {"claude": manifest, "gpt": None},
    )
    assert ingested["status"] == "partial_missing_judge"


def test_judge_manifest_replays_and_rejects_tampered_execution_artifact(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    case_path = _judge_case_manifest(tmp_path / "case-source")
    run_root = tmp_path / "judge-run"
    receipt = _judge_execution_receipt(run_root, case_path, judge="gpt")
    execution = json.loads(receipt.read_text(encoding="utf-8"))
    raw_path = receipt.parent / execution["raw_artifacts"][0]["path"]
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    output = run_root / "result.json"

    with pytest.raises(
        publication.PublicationAdapterError,
        match="receipt|artifact|replay",
    ):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            execution_receipt_path=receipt,
            output_path=output,
        )
    assert not output.exists()


def test_judge_manifest_rejects_execution_bound_to_another_case_manifest(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    case_path = _judge_case_manifest(tmp_path / "case-source")
    run_root = tmp_path / "judge-run"
    receipt = _judge_execution_receipt(run_root, case_path, judge="gpt")
    other_root = tmp_path / "other"
    other_root.mkdir()
    other_case_path = _judge_case_manifest(other_root, count=1)
    output = run_root / "result.json"

    with pytest.raises(publication.PublicationAdapterError, match="receipt|replay|case"):
        publication.build_judge_result_manifest(
            case_manifest_path=other_case_path,
            execution_receipt_path=receipt,
            output_path=output,
        )
    assert not output.exists()


def test_judge_manifest_refuses_existing_or_dangling_symlink_destination(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    case_path = _judge_case_manifest(tmp_path / "case-source", count=1)
    run_root = tmp_path / "judge-run"
    receipt = _judge_execution_receipt(run_root, case_path, judge="claude")
    existing = run_root / "existing.json"
    existing.write_text("do not replace", encoding="utf-8")
    with pytest.raises(publication.PublicationAdapterError, match="exist"):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            execution_receipt_path=receipt,
            output_path=existing,
        )
    assert existing.read_text(encoding="utf-8") == "do not replace"

    dangling = run_root / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target")
    with pytest.raises(publication.PublicationAdapterError, match="exist|symlink"):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            execution_receipt_path=receipt,
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
        allow_fixture=True,
    )

    assert output.is_dir()
    assert json.loads((output / "binding_manifest.json").read_text()) == manifest
    assert set(manifest["files"]) == {
        "analysis_results",
        "analysis_artifact_manifest",
        "analysis_input_artifact_manifest",
        "phase_provenance",
        "runtime_task_manifest",
        "mapping_manifest",
        "judge_run_execution_snapshot_manifest",
        "claude_judge_result_manifest",
        "gpt_judge_result_manifest",
    }
    assert len(manifest["files"]) == 9
    assert (output / "analysis/analysis_results.json").read_bytes() == (
        result_dir / "analysis_results.json"
    ).read_bytes()
    copied_table = output / "analysis/tables/provider_summary.csv"
    assert copied_table.read_bytes() == b"provider,status\ndeepseek,complete\n"
    table_path = result_dir / "tables/provider_summary.csv"
    table_path.write_bytes(b"mutated source\n")
    assert copied_table.read_bytes() == b"provider,status\ndeepseek,complete\n"
    assert copied_table.stat().st_ino != table_path.stat().st_ino
    bound = journal_binder.bind_persona_v2_artifacts(output, allow_fixture=True)
    assert bound["status"] == "bound_formal_w3"
    assert bound["persona_cluster_count"] == 50


def test_persona_bundle_derives_exact_gpt_only_binding_roles(tmp_path: Path) -> None:
    from experiments import journal_binder
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(
        tmp_path, judge_profile="gpt_only"
    )
    output = tmp_path / "persona-bundle"

    manifest = publication.build_persona_v2_bundle(
        result_dir=result_dir,
        main_phase_root=main_root,
        repo_root=repo,
        output_dir=output,
        allow_fixture=True,
    )

    assert set(manifest["files"]) == BASE_PERSONA_BUNDLE_ROLES | {
        JUDGE_RUN_SNAPSHOT_ROLE,
        "gpt_judge_result_manifest",
    }
    assert len(manifest["files"]) == 8
    assert not (output / "analysis/judge-results/claude.json").exists()
    assert (output / "judge-snapshots/run/family_dispositions/claude.json").is_file()
    assert not (output / "judge-snapshots/claude").exists()
    bound = journal_binder.bind_persona_v2_artifacts(output, allow_fixture=True)
    assert bound["judge_adjudication"]["analysis"]["status"] == (
        "partial_missing_judge"
    )


def test_persona_bundle_preserves_failed_gpt_and_unavailable_claude(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(
        tmp_path,
        judge_profile="all_missing",
    )
    output = tmp_path / "persona-bundle"

    manifest = publication.build_persona_v2_bundle(
        result_dir=result_dir,
        main_phase_root=main_root,
        repo_root=repo,
        output_dir=output,
        allow_fixture=True,
    )

    assert set(manifest["files"]) == BASE_PERSONA_BUNDLE_ROLES | {
        JUDGE_RUN_SNAPSHOT_ROLE
    }
    assert not any(role.endswith("_judge_result_manifest") for role in manifest["files"])
    failed_receipts = list(
        output.glob(
            "judge-snapshots/run/executions/gpt/*/failed_execution_receipt.json"
        )
    )
    assert len(failed_receipts) == 1
    bound = journal_binder.bind_persona_v2_artifacts(output, allow_fixture=True)
    analysis = bound["judge_adjudication"]["analysis"]
    assert analysis["status"] == "missing_all_judges"
    assert analysis["available_judges"] == []
    assert analysis["missing_judges"] == ["claude", "gpt"]


@pytest.mark.parametrize(
    "relative",
    (
        "judge-results/claude.json",
        "judge-snapshots/claude/snapshot_manifest.json",
        "judge-snapshots/claude/raw_attempts/copied.json",
        "archive/claude-copy.json",
    ),
)
def test_persona_bundle_rejects_manifested_claude_artifact_in_gpt_only_profile(
    tmp_path: Path,
    relative: str,
) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(
        tmp_path, judge_profile="gpt_only"
    )
    _add_manifested_artifact(result_dir, relative)
    output = tmp_path / "persona-bundle"

    with pytest.raises(
        publication.PublicationAdapterError,
        match=(
            "stray.*Claude|Claude.*artifact|availability.*artifact|"
            "snapshot.*unbound|snapshot.*missing"
        ),
    ):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=output,
            allow_fixture=True,
        )
    assert not output.exists()


@pytest.mark.parametrize("judge_profile", ("both_complete", "gpt_only"))
def test_persona_bundle_rejects_copied_completed_judge_result(
    tmp_path: Path,
    judge_profile: str,
) -> None:
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(
        tmp_path, judge_profile=judge_profile
    )
    copied_result = (result_dir / "judge-snapshots/run/gpt.json").read_bytes()
    _add_manifested_artifact(result_dir, "archive/gpt-copy.json", copied_result)

    with pytest.raises(
        publication.PublicationAdapterError,
        match="judge.*canonical|stray.*judge|judge.*artifact",
    ):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=tmp_path / "persona-bundle",
            allow_fixture=True,
        )


def test_persona_bundle_zero_case_rejects_neutral_named_judge_result(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    completed_dir, _completed_main, _completed_repo = _publication_sources(
        tmp_path / "completed-source", judge_profile="gpt_only"
    )
    copied_result = (completed_dir / "judge-snapshots/run/gpt.json").read_bytes()
    result_dir, main_root, repo = _publication_sources(
        tmp_path / "zero-source", judge_profile="zero_cases"
    )
    _add_manifested_artifact(result_dir, "archive/copy.json", copied_result)

    with pytest.raises(
        publication.PublicationAdapterError,
        match="judge.*canonical|stray.*judge|judge.*artifact",
    ):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=tmp_path / "persona-bundle",
            allow_fixture=True,
        )


def test_persona_bundle_rejects_ambiguous_duplicate_key_judge_json(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import publication

    completed_dir, _completed_main, _completed_repo = _publication_sources(
        tmp_path / "completed-source", judge_profile="gpt_only"
    )
    copied_result = (completed_dir / "judge-snapshots/run/gpt.json").read_bytes()
    ambiguous_result = copied_result.replace(
        b"{",
        b'{"schema_version":"yher.llm_sim_v2.judge_result_manifest.v2",',
        1,
    )
    result_dir, main_root, repo = _publication_sources(
        tmp_path / "zero-source", judge_profile="zero_cases"
    )
    _add_manifested_artifact(
        result_dir, "archive/ambiguous.json", ambiguous_result
    )

    with pytest.raises(
        publication.PublicationAdapterError,
        match="duplicate JSON key|strict.*JSON|judge.*canonical",
    ):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=tmp_path / "persona-bundle",
            allow_fixture=True,
        )


@pytest.mark.parametrize("suffix", ["json", "bin"])
def test_persona_bundle_rejects_wrapped_judge_payload_with_neutral_name(
    tmp_path: Path,
    suffix: str,
) -> None:
    from experiments.llm_sim_v2 import publication

    completed_dir, _completed_main, _completed_repo = _publication_sources(
        tmp_path / "completed-source", judge_profile="gpt_only"
    )
    copied_result = json.loads(
        (completed_dir / "judge-snapshots/run/gpt.json").read_text(
            encoding="utf-8"
        )
    )
    wrapped = _canonical({"archived_payload": copied_result}) + b"\n"
    result_dir, main_root, repo = _publication_sources(
        tmp_path / "zero-source", judge_profile="zero_cases"
    )
    _add_manifested_artifact(
        result_dir, f"archive/neutral.{suffix}", wrapped
    )

    with pytest.raises(
        publication.PublicationAdapterError,
        match="judge.*canonical|judge.*payload|stray.*judge",
    ):
        publication.build_persona_v2_bundle(
            result_dir=result_dir,
            main_phase_root=main_root,
            repo_root=repo,
            output_dir=tmp_path / "persona-bundle",
            allow_fixture=True,
        )


def test_persona_bundle_zero_case_has_base_roles_and_no_judge_artifacts(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(
        tmp_path, judge_profile="zero_cases"
    )
    output = tmp_path / "persona-bundle"

    manifest = publication.build_persona_v2_bundle(
        result_dir=result_dir,
        main_phase_root=main_root,
        repo_root=repo,
        output_dir=output,
        allow_fixture=True,
    )

    assert set(manifest["files"]) == BASE_PERSONA_BUNDLE_ROLES | {
        JUDGE_RUN_SNAPSHOT_ROLE
    }
    assert len(manifest["files"]) == 7
    assert not (output / "analysis/judge-results").exists()
    assert (output / "judge-snapshots/snapshot_manifest.json").is_file()
    assert (output / "judge-snapshots/run/family_dispositions/claude.json").is_file()
    assert (output / "judge-snapshots/run/family_dispositions/gpt.json").is_file()
    bound = journal_binder.bind_persona_v2_artifacts(output, allow_fixture=True)
    analysis = bound["judge_adjudication"]["analysis"]
    assert analysis["status"] == "not_applicable_zero_cases"
    assert analysis["available_judges"] == []
    assert analysis["missing_judges"] == []


@pytest.mark.parametrize(
    "drift",
    (
        "selected_case",
        "selected_stratum",
        "available_judge",
        "missing_judge",
        "category_count",
        "pairwise_agreement",
        "result_manifest",
    ),
)
def test_zero_case_judge_profile_rejects_every_nonempty_execution_surface(
    tmp_path: Path,
    drift: str,
) -> None:
    from experiments import journal_binder

    fixture = _binder_fixture_writer()(
        tmp_path / "fixture", judge_profile="zero_cases"
    )
    result = json.loads(
        (fixture / "analysis/analysis_results.json").read_text(encoding="utf-8")
    )
    adjudication = result["judge_adjudication"]
    case_manifest = adjudication["case_manifest"]
    analysis = adjudication["analysis"]
    if drift == "selected_case":
        case_manifest["selected_count"] = 1
        case_manifest["cases"] = [{"case_id": "stray-case"}]
    elif drift == "selected_stratum":
        case_manifest["selected_stratum_counts"]["agreement"] = 1
    elif drift == "available_judge":
        analysis["available_judges"] = ["gpt"]
    elif drift == "missing_judge":
        analysis["missing_judges"] = ["claude"]
    elif drift == "category_count":
        analysis["category_counts"] = {"gpt": {"labels": {}}}
    elif drift == "pairwise_agreement":
        analysis["pairwise_label_agreement"] = {"denominator": 0}
    else:
        adjudication["result_manifests"]["gpt"] = {"stray": True}
    if drift in {"selected_case", "selected_stratum"}:
        case_manifest["case_manifest_sha256"] = _self_hash(
            case_manifest, "case_manifest_sha256"
        )
        analysis["case_manifest_sha256"] = case_manifest[
            "case_manifest_sha256"
        ]
    if drift == "selected_case":
        analysis["selected_count"] = 1
        analysis["cases"] = ["stray-case"]

    with pytest.raises(
        journal_binder.BinderError,
        match=(
            "(?:zero-case judge profile|judge availability profile|"
            "judge analysis/case hash) drifted|"
            "judge result manifest profile.*stray"
        ),
    ):
        journal_binder._validate_persona_judge(adjudication)


def test_real_gpt_only_adapter_bundle_finalizes_repository_template(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder, journal_manuscript
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(
        tmp_path / "publication", judge_profile="gpt_only"
    )
    persona_bundle = tmp_path / "persona-bundle"
    publication.build_persona_v2_bundle(
        result_dir=result_dir,
        main_phase_root=main_root,
        repo_root=repo,
        output_dir=persona_bundle,
        allow_fixture=True,
    )
    binder = _full_journal_binder(tmp_path / "journal-inputs", persona_bundle=persona_bundle)
    binder_root = tmp_path / "journal-binder"
    journal_binder.write_binder(binder, binder_root)
    binder_generation = (binder_root / "current").resolve()
    repository = Path(__file__).parents[1]
    template = repository / "docs/paper/journal_main.md"
    references = repository / "docs/paper/references.json"

    journal_manuscript.finalize_manuscript(
        template_path=template,
        binder_generation=binder_root / "current",
        references_path=references,
        output_dir=tmp_path / "final",
        expected_template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        expected_binder_generation_id=binder_generation.name,
    )

    final_generation = (tmp_path / "final/current").resolve()
    journal_manuscript.verify_finalized_generation(
        final_generation, references_path=references
    )
    text = (final_generation / "journal_main.md").read_text(encoding="utf-8")
    assert text.count("GPT-only exploratory coding") == 1
    assert text.count("Claude judge was unavailable") == 1
    assert text.count("pairwise judge agreement was not estimable") == 1
    assert text.count("GPT-only coding") == 1
    assert text.count("Claude unavailable") == 1
    assert text.count("pairwise agreement not estimable") == 1
    assert "Cross-model judge label agreement" not in text


def test_zero_case_bundle_renders_not_applicable_manuscript_disclosure(
    tmp_path: Path,
) -> None:
    from experiments import journal_binder
    from experiments.llm_sim_v2 import publication

    result_dir, main_root, repo = _publication_sources(
        tmp_path / "publication", judge_profile="zero_cases"
    )
    persona_bundle = tmp_path / "persona-bundle"
    publication.build_persona_v2_bundle(
        result_dir=result_dir,
        main_phase_root=main_root,
        repo_root=repo,
        output_dir=persona_bundle,
        allow_fixture=True,
    )
    binder = _full_journal_binder(tmp_path / "journal-inputs", persona_bundle=persona_bundle)

    slots = journal_binder.render_manuscript_slots(binder)
    for text in (
        slots["bound_abstract_results_markdown"],
        slots["persona_v2_markdown"],
    ):
        assert (
            "exploratory adjudication was not applicable because no outcome-blind "
            "cases survived structural exclusion"
        ) in text
        assert "GPT-only exploratory coding" not in text
        assert "Cross-model judge label agreement" not in text


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
            allow_fixture=True,
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
            allow_fixture=True,
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
            allow_fixture=True,
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
            allow_fixture=True,
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
            allow_fixture=True,
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
            allow_fixture=True,
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
            allow_fixture=True,
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
            allow_fixture=True,
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
            allow_fixture=True,
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
            "--execution-receipt",
            "executions/gpt/execution_receipt.json",
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
    assert judge.execution_receipt == Path(
        "executions/gpt/execution_receipt.json"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
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
    assert persona.command == "persona-bundle"


def _commit_judge_run_anchor(tmp_path: Path, internal_receipt: Path) -> Path:
    repo = tmp_path / "judge-anchor-repo"
    relative = Path(
        "experiments/llm_sim_v2/evidence_anchors/"
        "judge_run_evidence_receipt.json"
    )
    anchor = repo / relative
    anchor.parent.mkdir(parents=True)
    shutil.copyfile(internal_receipt, anchor)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "judge-loader@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Judge Loader"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", relative.as_posix()], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "bind judge run evidence"],
        cwd=repo,
        check=True,
    )
    return repo


def _judge_results_directory(tmp_path: Path) -> tuple[Path, Path]:
    staging = tmp_path / "judge-loader-staging"
    staging.mkdir()
    writer = _binder_fixture_namespace()["_write_persona_judge_fixture"]
    writer(staging, judge_profile="gpt_only")
    root = tmp_path / ".judge-loader-staging-judge-run-source"
    repo = _commit_judge_run_anchor(
        tmp_path, root / "judge_run_evidence_receipt.json"
    )
    return root, repo


def test_analysis_loader_binds_the_complete_judge_execution_tree(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze

    root, repo = _judge_results_directory(tmp_path)
    manifests, artifact_paths, artifact_roots, run_evidence = (
        analyze._load_judge_result_manifests(
            root, repo_root=repo, allow_fixture=True
        )
    )

    assert manifests["claude"] is None
    assert manifests["gpt"] is not None
    execution_id = manifests["gpt"]["execution_receipt"]["identity"][
        "execution_id"
    ]
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert {relative for relative, _path in artifact_paths} == expected_paths
    assert artifact_roots == {
        "gpt": str((root / "executions" / "gpt" / execution_id).resolve())
    }
    assert run_evidence["receipt"]["family_slots"]["claude"]["status"] == (
        "unavailable"
    )
    assert run_evidence["receipt"]["family_slots"]["gpt"]["status"] == "complete"


def test_analysis_loader_rejects_tampered_or_unbound_judge_artifacts(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze

    root, repo = _judge_results_directory(tmp_path)
    manifest = json.loads((root / "gpt.json").read_text(encoding="utf-8"))
    execution_id = manifest["execution_receipt"]["identity"]["execution_id"]
    raw_relative = manifest["execution_receipt"]["raw_artifacts"][0]["path"]
    raw_path = root / "executions" / "gpt" / execution_id / raw_relative
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b" ")
    with pytest.raises(analyze.AnalysisContractError, match="judge.*artifact|hash|drift"):
        analyze._load_judge_result_manifests(
            root, repo_root=repo, allow_fixture=True
        )

    raw_path.write_bytes(original)
    extra = root / "executions" / "gpt" / execution_id / "unbound.txt"
    extra.write_text("unbound", encoding="utf-8")
    with pytest.raises(analyze.AnalysisContractError, match="unbound|file set|artifact"):
        analyze._load_judge_result_manifests(
            root, repo_root=repo, allow_fixture=True
        )


def test_judge_costs_merge_into_the_cumulative_hard_fuse(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import analyze

    root, repo = _judge_results_directory(tmp_path)
    manifests, _paths, _roots, run_evidence = analyze._load_judge_result_manifests(
        root, repo_root=repo, allow_fixture=True
    )
    judge_cost = analyze._judge_cost_accounting(
        manifests, judge_run_evidence=run_evidence
    )
    assert judge_cost["total_known_cost_yuan"] == 0.125
    assert judge_cost["total_unknown_reserve_yuan"] == 0.0
    assert judge_cost["total_accounted_cost_yuan"] == 0.125
    assert judge_cost["rows"][0]["judge"] == "gpt"
    assert judge_cost["rows"][0]["status"] == "complete"
    assert judge_cost["rows"][0]["requested_model"] == "fixture-gpt-exact"
    assert judge_cost["rows"][0]["transport_reported_models"] == [
        "fixture-gpt-exact"
    ]

    collection_cost = {
        "total_known_cost_yuan": 24.0,
        "total_unknown_reserve_yuan": 0.5,
        "total_accounted_cost_yuan": 24.5,
        "needs_user": True,
        "needs_user_reasons": ["unknown_provider_billing_reserved"],
        "soft_warning_yuan": 300.0,
        "hard_fuse_yuan": 450.0,
    }
    merged = analyze._merge_judge_cost_accounting(collection_cost, judge_cost)
    assert merged["collection_total_accounted_cost_yuan"] == 24.5
    assert merged["total_known_cost_yuan"] == 24.125
    assert merged["total_unknown_reserve_yuan"] == 0.5
    assert merged["total_accounted_cost_yuan"] == 24.625
    assert merged["judge"] == judge_cost["rows"]
    assert merged["needs_user"] is True
    assert "unknown_judge_billing_reserved" not in merged["needs_user_reasons"]

    collection_cost["total_known_cost_yuan"] = 449.995
    collection_cost["total_unknown_reserve_yuan"] = 0.0
    collection_cost["total_accounted_cost_yuan"] = 449.995
    with pytest.raises(analyze.AnalysisContractError, match="hard fuse|450"):
        analyze._merge_judge_cost_accounting(collection_cost, judge_cost)


def _judge_input_manifest(
    artifact_paths: list[tuple[str, Path]],
) -> dict[str, Any]:
    files = [
        {
            "path": f"judge-results/{relative}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for relative, path in artifact_paths
    ]
    return {
        "schema_version": "yher.llm_sim_v2.analysis_input_artifact_manifest.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "analysis_population": "main",
        "files": files,
        "input_file_count": len(files),
        "record_file_count": 0,
        "input_file_set_sha256": hashlib.sha256(_canonical(files)).hexdigest(),
    }


def test_w3_writer_snapshots_every_bound_judge_artifact(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import analyze

    root, repo = _judge_results_directory(tmp_path)
    manifests, artifact_paths, _roots, _run_evidence = (
        analyze._load_judge_result_manifests(
            root, repo_root=repo, allow_fixture=True
        )
    )
    sources = {
        f"judge-results/{relative}": str(path.resolve())
        for relative, path in artifact_paths
    }
    staging = tmp_path / "w3-staging"
    staging.mkdir()

    snapshots = analyze._stage_judge_execution_snapshots(
        staging=staging,
        judge_result_manifests=manifests,
        judge_artifact_sources=sources,
        input_artifact_manifest=_judge_input_manifest(artifact_paths),
        allow_fixture=True,
    )

    assert snapshots["schema_version"] == (
        "yher.llm_sim_v2.judge_run_execution_snapshot_manifest.v1"
    )
    assert (staging / "judge-snapshots/run/gpt.json").read_bytes() == (
        root / "gpt.json"
    ).read_bytes()
    snapshot = snapshots
    assert snapshot["file_count"] == len(snapshot["files"])
    assert snapshot["file_set_sha256"] == hashlib.sha256(
        _canonical(snapshot["files"])
    ).hexdigest()
    for row in snapshot["files"]:
        path = staging / "judge-snapshots" / row["path"]
        assert path.is_file()
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert row["size"] == path.stat().st_size


def test_w3_writer_rejects_judge_source_drift_before_snapshot(tmp_path: Path) -> None:
    from experiments.llm_sim_v2 import analyze

    root, repo = _judge_results_directory(tmp_path)
    manifests, artifact_paths, _roots, _run_evidence = (
        analyze._load_judge_result_manifests(
            root, repo_root=repo, allow_fixture=True
        )
    )
    input_manifest = _judge_input_manifest(artifact_paths)
    sources = {
        f"judge-results/{relative}": str(path.resolve())
        for relative, path in artifact_paths
    }
    raw_relative, raw_path = next(
        (relative, path)
        for relative, path in artifact_paths
        if "/raw_attempts/" in relative
    )
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    staging = tmp_path / "w3-staging"
    staging.mkdir()

    with pytest.raises(analyze.AnalysisContractError, match="judge.*changed|hash|drift"):
        analyze._stage_judge_execution_snapshots(
            staging=staging,
            judge_result_manifests=manifests,
            judge_artifact_sources=sources,
            input_artifact_manifest=input_manifest,
            allow_fixture=True,
        )
    assert not (staging / "judge-snapshots").exists()
    assert f"judge-results/{raw_relative}" in sources
