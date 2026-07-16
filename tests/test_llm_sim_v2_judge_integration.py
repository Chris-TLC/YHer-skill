from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def _write_json(path: Path, value: object) -> None:
    from experiments.llm_sim_v2.judge_execution import canonical_json_bytes

    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _case_manifest(tmp_path: Path) -> tuple[dict[str, object], Path]:
    from experiments.llm_sim_v2.analyze import build_judge_case_manifest

    public_question = {
        "kind": "mcq",
        "stem_blocks": [],
        "stem_text": "Which option follows from the public evidence?",
        "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "difficulty": 0.5,
        "nodes": ["private-node"],
        "source_label": "private-source",
    }
    candidate = {
        "candidate_identity": "provider|task-001",
        "stratum": "agreement",
        "public_question": public_question,
        "model_output": {
            "simulated": True,
            "answer": "A",
            "rationale": "The public evidence supports A.",
            "abstain": False,
        },
        "persona": {
            "persona_id": "private-persona",
            "target_node": "private-node",
            "deficit_condition": "deficit",
        },
        "item": {
            "item_id": "item-001",
            "public_question": public_question,
            "options": public_question["options"],
        },
    }
    manifest = build_judge_case_manifest(
        [candidate], frozen_leakage_lexicon=()
    )
    path = tmp_path / "case_manifest.json"
    _write_json(path, manifest)
    return manifest, path


def _results(
    manifest: dict[str, object], *, category: str = "none"
) -> list[dict[str, object]]:
    cases = manifest["cases"]
    return [
        {
            "case_id": row["case_id"],
            "output": {
                "label": "consistent",
                "error_category": category,
                "rationale": "The answer and rationale are coherent.",
                "simulated": True,
            }
        }
        for row in cases  # type: ignore[union-attr]
    ]


def _execution(
    tmp_path: Path,
    manifest: dict[str, object],
    *,
    judge: str,
) -> Path:
    from experiments.llm_sim_v2.judge_execution import FixtureJudgeTransport, execute_judge_pass

    model = f"fixture-{judge}-exact"
    results = _results(manifest)
    transport = FixtureJudgeTransport(
        [
            {
                "schema_version": "yher.llm_sim_v2.judge_transport_response.v2",
                "simulated": True,
                "transport_reported_models": [model],
                "transport_reported_model_source": "fixture_response",
                "transport_request_id": f"fixture-{judge}-request",
                "results": results,
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "billing": {
                    "known_cost_yuan": 0.125,
                    "unknown_cost_reserve_yuan": 0,
                },
                "tool_calls": [],
            }
        ]
    )
    return execute_judge_pass(
        case_manifest=manifest,
        output_root=tmp_path,
        judge_family=judge,
        exact_model=model,
        transport=transport,
    )


def _result_manifest(
    root: Path,
    manifest: dict[str, object],
    *,
    judge: str,
    receipt_path: Path,
) -> dict[str, object]:
    from experiments.llm_sim_v2.judge_execution import canonical_sha256

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    value: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.judge_result_manifest.v2",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "judge": judge,
        "case_manifest_sha256": manifest["case_manifest_sha256"],
        "execution_receipt_path": receipt_path.relative_to(root).as_posix(),
        "execution_receipt": receipt,
        "results": _results(manifest),
    }
    value["judge_result_manifest_sha256"] = canonical_sha256(value)
    _write_json(root / f"{judge}.json", value)
    return value


def _fixture_response(*, model: str, results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "yher.llm_sim_v2.judge_transport_response.v2",
        "simulated": True,
        "transport_reported_models": [model],
        "transport_reported_model_source": "fixture_response",
        "transport_request_id": "fixture-request",
        "results": results,
        "usage": {"input_tokens": 11, "output_tokens": 7},
        "billing": {
            "known_cost_yuan": 0.125,
            "unknown_cost_reserve_yuan": 0,
        },
        "tool_calls": [],
    }


def _commit_judge_anchor(repo: Path, internal_receipt: Path) -> Path:
    relative = Path(
        "experiments/llm_sim_v2/evidence_anchors/judge_run_evidence_receipt.json"
    )
    anchor = repo / relative
    anchor.parent.mkdir(parents=True)
    shutil.copyfile(internal_receipt, anchor)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "judge-integration@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Judge Integration"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", relative.as_posix()], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "bind judge run evidence"],
        cwd=repo,
        check=True,
    )
    return anchor


def _rehash_result(value: dict[str, object]) -> None:
    from experiments.llm_sim_v2.judge_execution import canonical_sha256

    value.pop("judge_result_manifest_sha256", None)
    value["judge_result_manifest_sha256"] = canonical_sha256(value)


def test_analyzer_ingests_v2_result_from_replayed_execution_receipt(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze

    cases, case_path = _case_manifest(tmp_path)
    _execution(tmp_path, cases, judge="gpt")
    receipt_path = _execution(tmp_path, cases, judge="claude")
    result = _result_manifest(
        tmp_path, cases, judge="claude", receipt_path=receipt_path
    )

    assert result["schema_version"] == "yher.llm_sim_v2.judge_result_manifest.v2"
    assert result["judge"] == "claude"
    assert result["execution_receipt"]["identity"]["transport"] == "fixture"
    assert len(result["results"]) == 1
    ingested = analyze.ingest_judge_results(
        cases,
        {"claude": result, "gpt": None},
        judge_artifact_roots={"claude": str(receipt_path.parent)},
        allow_fixture=True,
    )
    assert ingested["status"] == "partial_missing_judge"
    assert ingested["execution_receipt_sha256"]["claude"]


def test_two_independent_execution_receipts_produce_complete_agreement(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze

    cases, case_path = _case_manifest(tmp_path)
    manifests = {}
    roots = {}
    for judge in ("gpt", "claude"):
        receipt = _execution(tmp_path, cases, judge=judge)
        manifests[judge] = _result_manifest(
            tmp_path, cases, judge=judge, receipt_path=receipt
        )
        roots[judge] = str(receipt.parent)

    result = analyze.ingest_judge_results(
        cases, manifests, judge_artifact_roots=roots, allow_fixture=True
    )

    assert result["status"] == "complete"
    assert result["pairwise_label_agreement"]["exact_agreement"] == 1.0
    assert len(set(result["execution_ids"].values())) == 2
    assert result["judge_families"] == {"claude": "claude", "gpt": "gpt"}


def test_ingestion_rejects_one_execution_relabelled_as_both_judges(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze

    cases, case_path = _case_manifest(tmp_path)
    _execution(tmp_path, cases, judge="gpt")
    receipt = _execution(tmp_path, cases, judge="claude")
    claude = _result_manifest(
        tmp_path, cases, judge="claude", receipt_path=receipt
    )
    relabelled = deepcopy(claude)
    relabelled["judge"] = "gpt"
    _rehash_result(relabelled)

    with pytest.raises(
        analyze.AnalysisContractError,
        match="execution|identity|judge|independent",
    ):
        analyze.ingest_judge_results(
            cases,
            {"claude": claude, "gpt": relabelled},
            judge_artifact_roots={
                "claude": str(receipt.parent),
                "gpt": str(receipt.parent),
            },
            allow_fixture=True,
        )


def test_failed_execution_receipt_cannot_be_published_as_a_judge_result(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import judge_execution, publication

    cases, case_path = _case_manifest(tmp_path)
    invalid = _fixture_response(model="fixture-gpt-exact", results=[])
    transport = judge_execution.FixtureJudgeTransport(
        [invalid, invalid]
    )
    with pytest.raises(judge_execution.JudgeExecutionError) as captured:
        judge_execution.execute_judge_pass(
            case_manifest=cases,
            output_root=tmp_path,
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=transport,
        )

    failed_path = captured.value.receipt_path  # type: ignore[attr-defined]
    with pytest.raises(
        publication.PublicationAdapterError,
        match="receipt|replay|failed|transport",
    ):
        publication.build_judge_result_manifest(
            case_manifest_path=case_path,
            execution_receipt_path=failed_path,
            output_path=tmp_path / "gpt.json",
        )


def test_zero_case_ingestion_is_not_reported_as_missing_judges() -> None:
    from experiments.llm_sim_v2 import analyze

    cases = analyze.build_judge_case_manifest([], frozen_leakage_lexicon=())
    result = analyze.ingest_judge_results(
        cases, {"claude": None, "gpt": None}
    )

    assert result["status"] == "not_applicable_zero_cases"
    assert result["selected_count"] == 0
    assert result["cases"] == []
    assert result["expected_judges"] == ["claude", "gpt"]
    assert result["available_judges"] == []
    assert result["missing_judges"] == []
    for field in (
        "result_manifest_sha256",
        "execution_receipt_sha256",
        "execution_ids",
        "judge_models",
        "judge_families",
        "judge_transports",
        "judge_accounting",
        "category_counts",
    ):
        assert result[field] == {}
    assert result["pairwise_label_agreement"] is None
    assert result["pairwise_error_category_agreement"] is None
    assert result["label_disagreement_examples"] == []
    assert result["error_category_disagreement_examples"] == []


def test_loader_keeps_failed_execution_visible_and_costed(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze, judge_execution

    manifest, _case_path = _case_manifest(tmp_path)
    root = tmp_path / "judge-run"
    invalid = _fixture_response(model="fixture-gpt-exact", results=[])
    with pytest.raises(judge_execution.JudgePassFailed):
        judge_execution.execute_judge_pass(
            case_manifest=manifest,
            output_root=root,
            judge_family="gpt",
            exact_model="fixture-gpt-exact",
            transport=judge_execution.FixtureJudgeTransport([invalid, invalid]),
        )
    judge_execution.record_judge_family_disposition(
        case_manifest=manifest,
        output_root=root,
        judge_family="claude",
        status="unavailable",
        reason_code="production_cli_unavailable",
    )
    internal = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=manifest,
        output_root=root,
        allow_fixture=True,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    anchor = _commit_judge_anchor(repo, internal)

    manifests, paths, roots, run_evidence = analyze._load_judge_result_manifests(
        root, repo_root=repo, allow_fixture=True
    )

    assert manifests == {"claude": None, "gpt": None}
    assert roots == {}
    assert run_evidence["receipt"]["family_slots"]["gpt"]["status"] == "failed"
    assert run_evidence["committed_anchor"]["sha256"] == hashlib.sha256(
        anchor.read_bytes()
    ).hexdigest()
    assert any(relative.endswith("failed_execution_receipt.json") for relative, _ in paths)
    costs = analyze._judge_cost_accounting(
        manifests, judge_run_evidence=run_evidence
    )
    assert costs["rows"][0]["judge"] == "gpt"
    assert costs["rows"][0]["status"] == "failed"
    assert costs["rows"][0]["known_cost_yuan"] == 0.25
    assert costs["total_accounted_cost_yuan"] == 0.25


def test_loader_accepts_one_complete_and_one_unavailable_finalized_root(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze, judge_execution

    manifest, _case_path = _case_manifest(tmp_path)
    root = tmp_path / "judge-run"
    receipt = _execution(root, manifest, judge="gpt")
    published = _result_manifest(
        root, manifest, judge="gpt", receipt_path=receipt
    )
    judge_execution.record_judge_family_disposition(
        case_manifest=manifest,
        output_root=root,
        judge_family="claude",
        status="unavailable",
        reason_code="production_cli_unavailable",
    )
    internal = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=manifest,
        output_root=root,
        allow_fixture=True,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _commit_judge_anchor(repo, internal)

    manifests, paths, roots, run_evidence = analyze._load_judge_result_manifests(
        root, repo_root=repo, allow_fixture=True
    )

    assert manifests == {"claude": None, "gpt": published}
    assert roots == {"gpt": str(receipt.parent.resolve())}
    assert {row["status"] for row in run_evidence["receipt"]["family_slots"].values()} == {
        "complete",
        "unavailable",
    }
    assert {relative for relative, _path in paths} == {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    ingested = analyze.ingest_judge_results(
        manifest,
        manifests,
        judge_artifact_roots=roots,
        judge_run_evidence=run_evidence,
        allow_fixture=True,
    )
    assert ingested["status"] == "partial_missing_judge"
    assert ingested["judge_models"] == {"gpt": "fixture-gpt-exact"}


def test_w3_snapshot_copies_and_validates_the_entire_finalized_run(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim_v2 import analyze, judge_execution

    manifest = analyze.build_judge_case_manifest([], frozen_leakage_lexicon=())
    root = tmp_path / "judge-run"
    for family in ("claude", "gpt"):
        judge_execution.record_judge_family_disposition(
            case_manifest=manifest,
            output_root=root,
            judge_family=family,
            status="not_applicable_zero_cases",
            reason_code="selected_case_count_zero",
        )
    internal = judge_execution.write_judge_run_evidence_receipt(
        case_manifest=manifest,
        output_root=root,
        allow_fixture=True,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _commit_judge_anchor(repo, internal)
    _manifests, paths, _roots, _run_evidence = (
        analyze._load_judge_result_manifests(
            root, repo_root=repo, allow_fixture=True
        )
    )
    input_files = [
        {
            "path": f"judge-results/{relative}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for relative, path in paths
    ]
    input_manifest = {
        "schema_version": "yher.llm_sim_v2.analysis_input_artifact_manifest.v1",
        "simulated": True,
        "run_id": "llm-personas-v2-dual",
        "analysis_population": "main",
        "files": input_files,
        "input_file_count": len(input_files),
        "record_file_count": 0,
        "input_file_set_sha256": judge_execution.canonical_sha256(input_files),
    }
    sources = {
        f"judge-results/{relative}": str(path.resolve())
        for relative, path in paths
    }
    staging = tmp_path / "w3"
    staging.mkdir()

    snapshot = analyze._stage_judge_execution_snapshots(
        staging=staging,
        judge_artifact_sources=sources,
        input_artifact_manifest=input_manifest,
        allow_fixture=True,
    )

    assert snapshot["schema_version"] == (
        "yher.llm_sim_v2.judge_run_execution_snapshot_manifest.v1"
    )
    assert (staging / "judge-snapshots/snapshot_manifest.json").is_file()
    assert not (staging / "judge-snapshots/claude").exists()
    assert not (staging / "judge-snapshots/gpt").exists()
    for source in root.rglob("*"):
        if source.is_file():
            relative = source.relative_to(root)
            assert (staging / "judge-snapshots/run" / relative).read_bytes() == (
                source.read_bytes()
            )
    validated = analyze.validate_judge_run_execution_snapshot(
        staging / "judge-snapshots/snapshot_manifest.json",
        snapshot_root=staging / "judge-snapshots",
        allow_fixture=True,
    )
    assert validated == snapshot

    disposition = staging / "judge-snapshots/run/family_dispositions/gpt.json"
    original = disposition.read_bytes()
    disposition.write_bytes(original + b" ")
    with pytest.raises(
        analyze.AnalysisContractError,
        match="snapshot|bytes|drift|replay",
    ):
        analyze.validate_judge_run_execution_snapshot(
            staging / "judge-snapshots/snapshot_manifest.json",
            snapshot_root=staging / "judge-snapshots",
            allow_fixture=True,
        )
    disposition.write_bytes(original)
    extra = staging / "judge-snapshots/run/unbound.txt"
    extra.write_text("unbound", encoding="utf-8")
    with pytest.raises(
        analyze.AnalysisContractError,
        match="unbound|missing|file|directory",
    ):
        analyze.validate_judge_run_execution_snapshot(
            staging / "judge-snapshots/snapshot_manifest.json",
            snapshot_root=staging / "judge-snapshots",
            allow_fixture=True,
        )
