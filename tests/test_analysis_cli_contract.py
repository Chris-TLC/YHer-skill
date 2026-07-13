from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from analysis.dataset import DatasetContractError
from analysis import runner as analysis_runner


ROOT = Path(__file__).resolve().parents[1]


def test_analysis_module_exposes_a_non_mutating_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "analysis", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--manifest" in result.stdout
    assert "--output" in result.stdout
    assert "--results-contract" in result.stdout
    assert "--bootstrap-iterations" not in result.stdout


def test_makefile_has_primary_and_compatibility_targets_with_no_bytecode() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "paper-results:" in makefile
    assert "figures: paper-all" in makefile
    assert "PYTHONDONTWRITEBYTECODE=1" in makefile
    assert "data/sim_store/confirmatory/confirmatory-v1/manifest.json" in makefile
    assert "/tmp/yher_sprint2/paper_results" in makefile
    assert "docs/paper/results_contract.md" in makefile
    assert "paper-h5-merge: paper-h5-analyze" in makefile
    assert "paper-h5-merge-existing:" in makefile
    assert "paper-bind:" in makefile
    assert "paper-check:" in makefile
    assert "paper-final:" in makefile
    assert "paper-all:" in makefile
    assert "paper-h5-lock:" in makefile
    assert "paper-h5-finalize:" in makefile
    assert "paper-h5-analyze: paper-h5-finalize" in makefile
    final_block = makefile.split("paper-final:", 1)[1]
    assert "paper-results" not in final_block
    assert "paper-h5-finalize" not in final_block
    assert "paper-h5-analyze" not in final_block
    assert "paper-h5-merge-existing" in final_block
    assert "analysis.h5 merge" in makefile
    assert "--h5-results" in makefile
    assert "analysis.paper" in makefile
    assert "--check" in makefile
    all_block = makefile.split("paper-all:", 1)[1].split("paper-final:", 1)[0]
    ordered_targets = (
        "paper-results",
        "paper-h5-merge",
        "paper-bind",
        "paper-check",
    )
    positions = [all_block.index(target) for target in ordered_targets]
    assert positions == sorted(positions)
    assert "data/sim_store/llm_personas/llm-personas-v1" in makefile
    for target in ("paper-h5-finalize", "paper-all", "paper-final"):
        block = makefile.split(f"{target}:", 1)[1].split("\n\n", 1)[0]
        assert "paper-h5-lock" not in block
    assert "analysis.h5 lock" in makefile


def test_publication_figures_have_an_explicit_matplotlib_dependency() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "matplotlib>=3.10.0" in requirements.splitlines()


def test_paper_contract_exposes_all_frozen_primary_endpoints() -> None:
    text = (ROOT / "docs/paper/results_contract.md").read_text(encoding="utf-8")
    payload = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])

    assert {
        "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS",
        "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT",
        "H2_C_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
        "H2_C_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
        "H2_C_C_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS",
        "H2_C_A_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT",
        "H2_C_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT",
        "H2_C_C_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT",
        "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT",
        "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95",
        "H4_H1_RESCUE_MISSPECIFIED_B15_COMMON_SUPPORT",
        "H4_H1_RESCUE_MISSPECIFIED_B15_COMMON_SUPPORT_CI95",
        "H4_H2_HARM_MISSPECIFIED_B9_COMMON_SUPPORT",
        "H4_H2_HARM_MISSPECIFIED_B9_COMMON_SUPPORT_CI95",
        "H4_H2_A_MINUS_B_MISDIAGNOSIS_MISSPECIFIED_B9_COMMON_SUPPORT",
        "H4_H2_A_MINUS_B_MISDIAGNOSIS_MISSPECIFIED_B9_COMMON_SUPPORT_CI95",
    } <= set(payload["metrics"])


def test_formal_outputs_and_results_contract_cannot_overlap_raw_collection(
    tmp_path: Path,
) -> None:
    check_overlap = getattr(
        analysis_runner,
        "_assert_output_paths_do_not_overlap_raw",
        lambda *_args: None,
    )
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest = raw_root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetContractError, match="overlap raw collection"):
        check_overlap(
            manifest,
            raw_root / "analysis-output",
            tmp_path / "results_contract.md",
        )
    with pytest.raises(DatasetContractError, match="overlap raw collection"):
        check_overlap(
            manifest,
            tmp_path / "analysis-output",
            raw_root / "results_contract.md",
        )


def test_paper_templates_have_generated_abstract_markers_and_neutral_outer_status() -> None:
    main = (ROOT / "docs/paper/main.md").read_text(encoding="utf-8")
    yau = (ROOT / "docs/paper/yau_award_4page.md").read_text(encoding="utf-8")
    contract = (ROOT / "docs/paper/results_contract.md").read_text(encoding="utf-8")
    defense = (ROOT / "docs/paper/defense_pack.md").read_text(encoding="utf-8")
    en_begin = "<!-- BEGIN PAPER GENERATED ABSTRACT FINDINGS EN -->"
    en_end = "<!-- END PAPER GENERATED ABSTRACT FINDINGS EN -->"
    zh_begin = "<!-- BEGIN PAPER GENERATED ABSTRACT FINDINGS ZH -->"
    zh_end = "<!-- END PAPER GENERATED ABSTRACT FINDINGS ZH -->"

    assert main.count(en_begin) == main.count(en_end) == 1
    assert zh_begin not in main and zh_end not in main
    assert yau.count(en_begin) == yau.count(en_end) == 1
    assert yau.count(zh_begin) == yau.count(zh_end) == 1
    assert "A Pre-Results Study" not in yau
    assert "This draft does not publish, push, mint a DOI, or claim a completed" not in main
    assert "PENDING_S3" not in defense
    assert "PENDING_S3" not in contract.split("<!-- BEGIN S3 GENERATED RESULTS -->", 1)[0]


def test_document_evidence_pointers_resolve_to_frozen_repo_sources_and_archive() -> None:
    defense_path = ROOT / "docs/paper/defense_pack.md"
    defense = defense_path.read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "data/sim_store/confirmatory/confirmatory-v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_hashes = {
        "mastery.py": manifest["input_sha256"]["production_mastery"]["sha256"],
        "selector.py": manifest["input_sha256"]["production_selector"]["sha256"],
    }
    for filename, expected_sha in expected_hashes.items():
        relative = Path("../../engine") / filename
        assert relative.as_posix() in defense
        resolved = (defense_path.parent / relative).resolve()
        assert resolved == ROOT / "engine" / filename
        assert hashlib.sha256(resolved.read_bytes()).hexdigest() == expected_sha
    planner = (defense_path.parent / "../../engine/planner.py").resolve()
    assert planner == ROOT / "engine/planner.py"
    assert "../../engine/planner.py" in defense
    assert "../../../engine/" not in defense

    decision_log = (ROOT / "docs/paper/decision_log.md").read_text(encoding="utf-8")
    archive_relative = "../../../PROJECT_HANDOFF/ledger_archive/ledger_2026-07.md"
    assert archive_relative in decision_log
    archive = (ROOT / "docs/paper" / archive_relative).resolve()
    heading = "### 2026-07-12 23:21 CST - YHer快速诚实结项与美本CS材料渠道"
    assert heading in archive.read_text(encoding="utf-8")
