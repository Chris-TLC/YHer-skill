from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from analysis.dataset import DatasetContractError
from analysis.metrics import AnalysisEvent, AnalysisRow
from analysis.no_repeat import NoRepeatSet
from analysis.results import (
    _FIGURE_RC,
    _plot_misspecification_by_item_type,
    build_results,
    replace_results_contract,
    write_results,
)
from analysis.runner import _complete_schema_invalid_pairs, run_formal_analysis


def _rows() -> tuple[AnalysisRow, ...]:
    output: list[AnalysisRow] = []
    for target in ("T1", "T2"):
        for truth in ("M", "P", "C", "U"):
            for condition in ("matched", "misspecified"):
                for replicate in range(2):
                    for arm in ("A", "B", "C"):
                        for budget in (9, 15, 25):
                            correct = True
                            converged = True
                            if truth == "P" and arm == "B":
                                correct = False
                                converged = False
                            if truth == "C" and arm == "C":
                                correct = False
                            output.append(
                                AnalysisRow(
                                    target=target,
                                    truth=truth,
                                    condition=condition,
                                    replicate=replicate,
                                    arm=arm,
                                    budget=budget,
                                    argmax=truth if correct else "U" if truth != "U" else "M",
                                    converged=converged,
                                    convergence_time=4 if converged else None,
                                    actual_administered_count=4 if converged else budget,
                                    held_out_brier=0.1 if arm == "A" else 0.2,
                                    exact_item_repeat_fraction=0.0,
                                    family_repeat_fraction=0.0,
                                    h1_h2_eligible=True,
                                    common_support_no_repeat=budget != 25,
                                    prerequisite_count=1 if arm == "A" else 0,
                                    prerequisite_share=(1 / 4) if arm == "A" else 0.0,
                                    direct_count=3 if arm == "A" else 4,
                                    unique_item_count=4 if converged else budget,
                                    unique_family_count=4 if converged else budget,
                                    valid=not (
                                        target == "T2"
                                        and truth == "M"
                                        and replicate == 1
                                        and arm == "C"
                                    ),
                                    exclusion_reason=(
                                        "structural_failure_item_pool"
                                        if target == "T2"
                                        and truth == "M"
                                        and replicate == 1
                                        and arm == "C"
                                        else None
                                    ),
                                )
                            )
    return tuple(output)


def _events() -> tuple[AnalysisEvent, ...]:
    output: list[AnalysisEvent] = []
    for target in ("T1", "T2"):
        for truth in ("M", "P", "C", "U"):
            for condition in ("matched", "misspecified"):
                for replicate in range(2):
                    for arm in ("A", "B", "C"):
                        invalid = (
                            target == "T2"
                            and truth == "M"
                            and replicate == 1
                            and arm == "C"
                        )
                        for position, item_type in enumerate(
                            ("mcq", "mcq", "numeric"), start=1
                        ):
                            production = 0.25 if item_type == "mcq" else 0.03
                            generator = (
                                production
                                if condition == "matched"
                                else 0.35 if item_type == "mcq" else 0.20
                            )
                            output.append(
                                AnalysisEvent(
                                    target=target,
                                    truth=truth,
                                    condition=condition,
                                    replicate=replicate,
                                    arm=arm,
                                    position=position,
                                    item_type=item_type,
                                    generator_probability=generator,
                                    production_probability=production,
                                    valid=not invalid,
                                    exclusion_reason=(
                                        "structural_failure_item_pool" if invalid else None
                                    ),
                                )
                            )
    return tuple(output)


def _no_repeat() -> dict[int, NoRepeatSet]:
    return {
        9: NoRepeatSet(9, ("T1", "T2"), 2, "hash-9"),
        15: NoRepeatSet(15, ("T1", "T2"), 2, "hash-15"),
        25: NoRepeatSet(25, (), 0, "hash-25"),
    }


def test_item_type_figure_footer_is_complete_and_inside_canvas() -> None:
    rows = (
        {
            "item_type": "mcq",
            "generator_minus_production": 0.10,
            "gap_ci_low": 0.08,
            "gap_ci_high": 0.12,
            "event_count": 123_456,
            "journey_count": 54_321,
            "n_target": 27,
        },
        {
            "item_type": "numeric",
            "generator_minus_production": 0.17,
            "gap_ci_low": 0.14,
            "gap_ci_high": 0.20,
            "event_count": 123_456,
            "journey_count": 54_321,
            "n_target": 27,
        },
    )

    with matplotlib.rc_context(_FIGURE_RC):
        figure = _plot_misspecification_by_item_type(rows)
        try:
            figure.canvas.draw()
            footer = figure.texts[-1]
            footer_lines = footer.get_text().splitlines()
            footer_box = footer.get_window_extent(figure.canvas.get_renderer())
            canvas_box = figure.bbox

            assert footer_box.x0 >= canvas_box.x0
            assert footer_box.x1 <= canvas_box.x1
            assert footer_box.y0 >= canvas_box.y0
            assert footer_box.y1 <= canvas_box.y1
            assert len(footer_lines) == 4
            assert footer_lines[2].startswith("MCQ: 123,456 events")
            assert footer_lines[3].startswith("Numeric: 123,456 events")
        finally:
            plt.close(figure)


def test_results_are_generated_from_registry_with_companion_data_and_stable_hashes(
    tmp_path: Path,
) -> None:
    source_provenance = {
        "run_id": "fixture-run",
        "run_started_at_utc": "2026-07-13T13:23:07Z",
        "runner_commit": "runner-commit",
        "experiment_tag": "experiment-freeze-20260713",
        "config_sha256": "config-sha",
        "source_manifest_sha256": "manifest-sha",
        "analysis_plan_commit": "plan-commit",
        "analysis_plan_sha256": "plan-sha",
    }
    analysis_provenance = {
        "analysis_commit": "analysis-commit",
        "analysis_code_committed_at_utc": "2026-07-14T00:00:00Z",
        "analysis_code_sha256": "analysis-code-sha",
        "analysis_code_files": {
            "analysis/results.py": "results-sha",
            "requirements.txt": "requirements-sha",
        },
    }
    bundle = build_results(
        _rows(),
        events=_events(),
        raw_hash="raw-hash",
        no_repeat_sets=_no_repeat(),
        bootstrap_iterations=40,
        source_provenance=source_provenance,
        analysis_provenance=analysis_provenance,
    )

    assert bundle.decisions == {
        "H1": "supported",
        "H2": "supported",
        "H3": "supported",
        "H4": "supported",
        "H5": None,
    }
    assert bundle.registry.records()
    assert all(row["raw_hash"] == "raw-hash" for row in bundle.registry.records())
    exploratory = [
        row
        for row in bundle.registry.records()
        if str(row["metric_id"]).startswith("exploratory_posthoc.prerequisite")
    ]
    assert exploratory
    assert all("exploratory_posthoc" in str(row["weighting"]) for row in exploratory)
    assert all(
        row["ci_low"] is not None and row["ci_high"] is not None
        for row in exploratory
    )
    assert any("direct_count" in str(row["metric_id"]) for row in exploratory)
    assert any("terminal_accuracy" in str(row["metric_id"]) for row in exploratory)
    assert any("correct_convergence" in str(row["metric_id"]) for row in exploratory)
    registry = {str(row["metric_id"]): row for row in bundle.registry.records()}
    for metric_id in (
        "h4.degradation.h1_rescue.matched_minus_misspecified",
        "h4.degradation.h2_harm.matched_minus_misspecified",
        "h4.degradation.h2_no_harm.matched_minus_misspecified",
    ):
        assert metric_id in registry
        assert registry[metric_id]["ci_low"] is not None
        assert registry[metric_id]["ci_high"] is not None
    assert "h4.misspecified.b9.no_harm_A_minus_B" in registry
    assert {
        "misspecification.item_type.mcq.generator_probability",
        "misspecification.item_type.mcq.production_probability",
        "misspecification.item_type.mcq.generator_minus_production",
        "misspecification.item_type.numeric.generator_probability",
        "misspecification.item_type.numeric.production_probability",
        "misspecification.item_type.numeric.generator_minus_production",
    } <= set(registry)
    for item_type in ("mcq", "numeric"):
        diagnostic = registry[
            f"misspecification.item_type.{item_type}.generator_minus_production"
        ]
        assert diagnostic["ci_low"] is not None
        assert diagnostic["ci_high"] is not None
        assert "bootstrap_iterations=10000" in str(diagnostic["weighting"])
        assert "journey_cluster_preserved" in str(diagnostic["weighting"])
        assert "diagnostic_only_not_item_type_H1_H2_estimand" in str(
            diagnostic["weighting"]
        )
        assert int(diagnostic["denominator"]) >= int(diagnostic["n_pair"])
    assert bundle.validation["estimand_excluded_journey_count"] == 2
    assert bundle.validation["estimand_exclusion_reasons"] == {
        "structural_failure_item_pool": 2
    }
    assert bundle.validation["estimand_exclusion_arms"] == {"C": 2}
    assert bundle.validation["estimand_exclusion_targets"] == ["T2"]
    assert bundle.conditional_metric_audit
    assert all(
        audit["attempted_iterations"] == 40
        and audit["defined_iterations"] <= 40
        and audit["redraw_count"] == 0
        for audit in bundle.conditional_metric_audit.values()
    )
    assert registry["h3.matched.b15.time_to_confidence.arm_A"]["ci_low"] is not None
    assert registry["h3.matched.b15.time_to_confidence.arm_B"]["ci_high"] is not None

    severe = [
        row
        for row in bundle.registry.records()
        if str(row["metric_id"]).startswith("severe.")
    ]
    assert len(severe) == 2 * 3 * 3 * 3
    assert all(row["ci_low"] is not None and row["ci_high"] is not None for row in severe)
    assert {
        str(row["metric_id"]).rsplit(".", 1)[-1]
        for row in severe
    } == {
        "all_journeys",
        "all_converged_journeys",
        "truth_M_or_U_all_journeys",
    }
    confusion = [
        row
        for row in bundle.registry.records()
        if str(row["metric_id"]).startswith("confusion.")
    ]
    assert len(confusion) == 2 * 3 * 3 * ((4 * 4) + (4 * 5))
    assert all(
        row["ci_low"] is not None and row["ci_high"] is not None
        for row in confusion
    )
    outcome_grid = [
        row
        for row in bundle.registry.records()
        if str(row["metric_id"]).startswith("outcome_by_view.")
    ]
    assert len(outcome_grid) == 2 * 3 * 3 * 5 * 4
    assert all(row["ci_low"] is not None and row["ci_high"] is not None for row in outcome_grid)
    brier_grid = [
        row
        for row in bundle.registry.records()
        if str(row["metric_id"]).startswith("held_out_brier.")
    ]
    assert len(brier_grid) == 2 * 3 * 3
    quality_grid = [
        row
        for row in bundle.registry.records()
        if str(row["metric_id"]).startswith("journey_quality.")
    ]
    assert len(quality_grid) == 2 * 3 * 3 * 5
    assert all(row["ci_low"] is not None and row["ci_high"] is not None for row in quality_grid)
    convergence_grid = [
        row
        for row in bundle.registry.records()
        if str(row["metric_id"]).startswith("convergence_time.")
    ]
    assert len(convergence_grid) == 2 * 3 * (10 + 16 + 26)
    no_repeat_h1_h2 = [
        row
        for row in bundle.registry.records()
        if str(row["metric_id"]).startswith(("h1.no_repeat.", "h2.no_repeat."))
    ]
    assert len(no_repeat_h1_h2) == 2 * (3 + 5)
    assert {
        str(row["metric_id"]).split(".")[2] for row in no_repeat_h1_h2
    } == {"matched", "misspecified"}

    first = tmp_path / "first"
    second = tmp_path / "second"
    write_results(bundle, first)
    write_results(bundle, second)

    required = {
        "metric_registry.json",
        "metric_registry.csv",
        "canonical_views.csv",
        "results.json",
        "results_fragment.md",
        "hypotheses.csv",
        "no_repeat_sets.json",
        "h5_status.json",
        "results_contract_block.md",
        "tables/outcomes_by_view.csv",
        "tables/journey_quality.csv",
        "tables/misspecification_by_item_type.csv",
        "figures/p_rescue.csv",
        "figures/p_rescue.svg",
        "figures/p_rescue.png",
        "figures/c_misdiagnosis.csv",
        "figures/c_misdiagnosis.svg",
        "figures/c_misdiagnosis.png",
        "figures/matched_vs_misspecified.csv",
        "figures/matched_vs_misspecified.svg",
        "figures/matched_vs_misspecified.png",
        "figures/misspecification_by_item_type.csv",
        "figures/misspecification_by_item_type.svg",
        "figures/misspecification_by_item_type.png",
        "figures/confusion_terminal.csv",
        "figures/confusion_terminal.svg",
        "figures/confusion_terminal.png",
        "figures/confusion_decision.csv",
        "figures/confusion_decision.svg",
        "figures/confusion_decision.png",
        "figures/held_out_brier.csv",
        "figures/held_out_brier.svg",
        "figures/held_out_brier.png",
        "figures/convergence_distribution.csv",
        "figures/convergence_distribution.svg",
        "figures/convergence_distribution.png",
        "artifact_manifest.json",
        "static_audit_policy.json",
    }
    assert required <= {
        str(path.relative_to(first)) for path in first.rglob("*") if path.is_file()
    }
    first_manifest = json.loads((first / "artifact_manifest.json").read_text())
    second_manifest = json.loads((second / "artifact_manifest.json").read_text())
    assert first_manifest == second_manifest
    assert first_manifest["analysis_commit"] == analysis_provenance["analysis_commit"]
    assert first_manifest["analysis_code_sha256"] == analysis_provenance[
        "analysis_code_sha256"
    ]
    assert first_manifest["analysis_code_files"] == analysis_provenance[
        "analysis_code_files"
    ]
    first_files = {
        str(path.relative_to(first)): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        str(path.relative_to(second)): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    assert "H1: supported" in (first / "results_fragment.md").read_text()
    h5_status = json.loads((first / "h5_status.json").read_text())
    assert h5_status["status"] == "PROGRAMMATIC_COMPLETE_H5_PENDING"
    assert h5_status["decision"] is None
    results = json.loads((first / "results.json").read_text())
    assert results["validation"]["canonical_view_count"] == len(_rows())
    assert results["source_run_started_at_utc"] == (
        source_provenance["run_started_at_utc"]
    )
    assert results["analysis_code_committed_at_utc"] == (
        analysis_provenance["analysis_code_committed_at_utc"]
    )
    assert results["analysis_timestamp_policy"] == (
        "analysis_code_commit_time_for_byte_determinism"
    )
    assert "generated_at_utc" not in results
    assert results["source_provenance"] == source_provenance
    assert results["analysis_provenance"] == analysis_provenance
    assert results["analysis_commit"] == analysis_provenance["analysis_commit"]
    assert results["conditional_metric_audit"] == bundle.conditional_metric_audit
    assert results["static_audit_policy"]["path"] == "static_audit_policy.json"
    assert results["bootstrap"]["item_type_diagnostic_iterations"] == 10_000
    assert results["bootstrap"]["item_type_diagnostic_unit"] == (
        "target-fixed; paired-replicate resample; whole-journey event clusters"
    )

    p_svg = (first / "figures/p_rescue.svg").read_text(encoding="utf-8")
    assert "YHer analysis (matplotlib)" in p_svg
    assert "Correct convergence (%)" in p_svg
    assert "Nominal item budget" in p_svg
    assert "95% CI" in p_svg
    assert "Matched" in p_svg and "Misspecified" in p_svg
    assert "n =" in p_svg

    degradation_svg = (first / "figures/matched_vs_misspecified.svg").read_text(
        encoding="utf-8"
    )
    assert "Matched to misspecified" in degradation_svg
    assert "H1 rescue" in degradation_svg
    assert "H2 harm" in degradation_svg
    assert "H2 no-harm" in degradation_svg
    assert "Contrast (percentage points)" in degradation_svg

    item_type_svg = (first / "figures/misspecification_by_item_type.svg").read_text(
        encoding="utf-8"
    )
    assert "Misspecified generator gap by administered item type" in item_type_svg
    assert "MCQ" in item_type_svg and "Numeric" in item_type_svg
    assert "95% CI" in item_type_svg
    assert "journey-cluster" in item_type_svg
    assert "diagnostic only" in item_type_svg
    assert "estimand. MCQ:" not in item_type_svg
    assert "targets, Numeric:" not in item_type_svg

    with (first / "tables/misspecification_by_item_type.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        item_type_rows = list(csv.DictReader(handle))
    assert {row["item_type"] for row in item_type_rows} == {"mcq", "numeric"}
    assert all(row["gap_ci_low"] and row["gap_ci_high"] for row in item_type_rows)
    assert all(int(row["bootstrap_iterations"]) == 10_000 for row in item_type_rows)
    assert all(int(row["event_count"]) >= int(row["journey_count"]) for row in item_type_rows)
    assert all(
        row["interpretation"] == "generator_diagnostic_not_item_type_H1_H2_estimand"
        for row in item_type_rows
    )

    terminal_svg = (first / "figures/confusion_terminal.svg").read_text(
        encoding="utf-8"
    )
    assert all(f"Arm {arm}" in terminal_svg for arm in ("A", "B", "C"))
    assert "Predicted terminal state" in terminal_svg
    assert "Truth state" in terminal_svg
    assert "95% CI" in terminal_svg

    with (first / "figures/confusion_terminal.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        terminal_csv = list(csv.DictReader(handle))
    assert terminal_csv
    assert all(row["ci_low"] and row["ci_high"] for row in terminal_csv)
    assert {row["condition"] for row in terminal_csv} == {
        "matched",
        "misspecified",
    }
    assert {int(row["budget"]) for row in terminal_csv} == {9, 15, 25}

    convergence_svg = (first / "figures/convergence_distribution.svg").read_text(
        encoding="utf-8"
    )
    assert "Convergence-time distribution" in convergence_svg
    assert "NC = 16" in convergence_svg
    assert "Journey share (%)" in convergence_svg
    assert "95% CI" in convergence_svg

    prerequisite_svg = (first / "figures/prerequisite_share.svg").read_text(
        encoding="utf-8"
    )
    assert "95% CI" in prerequisite_svg

    with (first / "figures/held_out_brier.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        brier_csv = list(csv.DictReader(handle))
    assert len(brier_csv) == 2 * 3 * 3
    assert {int(row["budget"]) for row in brier_csv} == {9, 15, 25}

    for png in (first / "figures").glob("*.png"):
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    contract_block = (first / "results_contract_block.md").read_text(encoding="utf-8")
    assert contract_block.startswith("<!-- BEGIN S3 GENERATED RESULTS -->\n")
    assert contract_block.endswith("<!-- END S3 GENERATED RESULTS -->\n")
    contract_payload = json.loads(contract_block.split("```json\n", 1)[1].split("\n```", 1)[0])
    template_text = (Path(__file__).parents[1] / "docs/paper/results_contract.md").read_text(
        encoding="utf-8"
    )
    template_payload = json.loads(
        template_text.split("```json\n", 1)[1].split("\n```", 1)[0]
    )
    assert set(contract_payload["metrics"]) == set(template_payload["metrics"])
    assert contract_payload["status"] == "PROGRAMMATIC_COMPLETE_H5_PENDING"
    assert contract_payload["source_run_started_at_utc"] == (
        source_provenance["run_started_at_utc"]
    )
    assert contract_payload["analysis_code_committed_at_utc"] == (
        analysis_provenance["analysis_code_committed_at_utc"]
    )
    assert contract_payload["analysis_timestamp_policy"] == (
        "analysis_code_commit_time_for_byte_determinism"
    )
    assert "generated_at_utc" not in contract_payload
    assert contract_payload["analysis_artifact"] == "metric_registry.json"
    assert contract_payload["conditional_metric_audit"] == (
        bundle.conditional_metric_audit
    )
    assert contract_payload["static_audit_policy"]["path"] == (
        "static_audit_policy.json"
    )
    assert contract_payload["analysis_plan_commit"] == "plan-commit"
    assert contract_payload["analysis_plan_sha256"] == "plan-sha"
    assert contract_payload["denominators"]["estimand_excluded_journey_count"] == 2
    assert contract_payload["denominators"]["estimand_exclusion_reasons"] == {
        "structural_failure_item_pool": 2
    }
    assert contract_payload["analysis_commit"] == analysis_provenance["analysis_commit"]
    assert contract_payload["analysis_code_sha256"] == analysis_provenance[
        "analysis_code_sha256"
    ]
    assert contract_payload["analysis_code_files"] == analysis_provenance[
        "analysis_code_files"
    ]
    assert len(contract_payload["analysis_artifact_sha256"]) == 64
    assert contract_payload["metrics"][
        "H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"
    ]["value"] == registry["p_rescue.full.matched.b15.arm_A"]["value"]
    assert contract_payload["metrics"][
        "H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS"
    ]["value"] == registry["p_rescue.full.matched.b15.arm_B"]["value"]
    assert contract_payload["metrics"][
        "H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT"
    ]["value"] == registry[
        "h2.no_repeat.matched.b9.no_harm_A_minus_B"
    ]["value"]
    assert contract_payload["metrics"]["H5_QUALIFYING_PROVIDER_COUNT"] is None
    assert contract_payload["decisions"]["H5"] is None
    assert contract_payload["decision_details"]["H1"]["analysis_status"] == "complete"
    assert contract_payload["decision_details"]["H2"]["branch_reason"] == (
        "harm_ci_strictly_positive_and_no_harm_ci_below_0_05"
    )
    assert contract_payload["decision_details"]["H5"] == {
        "analysis_status": "pending_input",
        "decision": None,
        "branch_reason": "validated_S2_provider_panel_not_supplied",
        "predicate_inputs": {},
    }
    assert contract_payload["hypotheses"] == contract_payload["decision_details"]
    assert contract_payload["metrics"][
        "H4_H1_RESCUE_MISSPECIFIED_B15_COMMON_SUPPORT"
    ]["registry_metric_id"] == "h1.no_repeat.misspecified.b15.rescue_A_minus_B"
    assert contract_payload["metrics"][
        "H4_H2_HARM_MISSPECIFIED_B9_COMMON_SUPPORT"
    ]["registry_metric_id"] == "h2.no_repeat.misspecified.b9.harm_C_minus_A"
    assert contract_payload["figures"]["FIG_CONFUSION_MATRICES"][
        "terminal_svg"
    ]["path"] == "figures/confusion_terminal.svg"


def test_schema_invalid_journey_gets_placeholders_and_excludes_its_whole_pair() -> None:
    rows = _rows()
    events = _events()
    invalid_key = ("T1", "M", "matched", 0, "A")
    rows_without_invalid = tuple(
        row
        for row in rows
        if (row.target, row.truth, row.condition, row.replicate, row.arm)
        != invalid_key
    )
    events_without_invalid = tuple(
        event
        for event in events
        if (event.target, event.truth, event.condition, event.replicate, event.arm)
        != invalid_key
    )

    completed_rows, completed_events = _complete_schema_invalid_pairs(
        rows_without_invalid,
        events_without_invalid,
        invalid_primary_keys=(invalid_key,),
    )

    pair_rows = [
        row
        for row in completed_rows
        if (row.target, row.truth, row.condition, row.replicate)
        == invalid_key[:4]
    ]
    assert len(pair_rows) == 3 * 3
    assert all(row.valid is False for row in pair_rows)
    assert all(row.exclusion_reason == "paired_schema_invalid" for row in pair_rows)
    assert len(completed_rows) == len(rows)
    assert all(
        event.valid is False
        for event in completed_events
        if (event.target, event.truth, event.condition, event.replicate)
        == invalid_key[:4]
    )


def test_results_validation_reports_nonzero_schema_quarantine_reasons() -> None:
    source_provenance = {
        "run_id": "fixture-run",
        "run_started_at_utc": "2026-07-13T13:23:07Z",
        "runner_commit": "runner-commit",
        "experiment_tag": "experiment-freeze-20260713",
        "config_sha256": "config-sha",
        "source_manifest_sha256": "manifest-sha",
        "analysis_plan_commit": "plan-commit",
        "analysis_plan_sha256": "plan-sha",
    }
    analysis_provenance = {
        "analysis_commit": "analysis-commit",
        "analysis_code_committed_at_utc": "2026-07-14T00:00:00Z",
        "analysis_code_sha256": "analysis-code-sha",
        "analysis_code_files": {"analysis/results.py": "results-sha"},
    }
    rows = tuple(
        replace(row, valid=False, exclusion_reason="paired_schema_invalid")
        if (row.target, row.truth, row.condition, row.replicate)
        == ("T1", "M", "matched", 0)
        else row
        for row in _rows()
    )
    events = tuple(
        replace(event, valid=False, exclusion_reason="paired_schema_invalid")
        if (event.target, event.truth, event.condition, event.replicate)
        == ("T1", "M", "matched", 0)
        else event
        for event in _events()
    )

    bundle = build_results(
        rows,
        events=events,
        raw_hash="raw-hash",
        no_repeat_sets=_no_repeat(),
        bootstrap_iterations=10,
        source_provenance=source_provenance,
        analysis_provenance=analysis_provenance,
        schema_invalid_count=1,
        schema_invalid_reasons={"posterior_schema_invalid": 1},
        intended_journey_count=len(rows) // 3,
    )

    assert bundle.validation["schema_invalid_count"] == 1
    assert bundle.validation["schema_invalid_reasons"] == {
        "posterior_schema_invalid": 1
    }
    assert bundle.validation["intended_journey_count"] == len(rows) // 3


def test_results_reject_no_repeat_targets_outside_h1_h2_eligibility() -> None:
    source_provenance = {
        "run_id": "fixture-run",
        "run_started_at_utc": "2026-07-13T13:23:07Z",
        "runner_commit": "runner-commit",
        "experiment_tag": "experiment-freeze-20260713",
        "config_sha256": "config-sha",
        "source_manifest_sha256": "manifest-sha",
        "analysis_plan_commit": "plan-commit",
        "analysis_plan_sha256": "plan-sha",
    }
    analysis_provenance = {
        "analysis_commit": "analysis-commit",
        "analysis_code_committed_at_utc": "2026-07-14T00:00:00Z",
        "analysis_code_sha256": "analysis-code-sha",
        "analysis_code_files": {"analysis/results.py": "results-sha"},
    }
    rows = tuple(
        replace(row, h1_h2_eligible=row.target == "T1") for row in _rows()
    )

    with pytest.raises(DatasetContractError, match="no-repeat.*eligible"):
        build_results(
            rows,
            events=_events(),
            raw_hash="raw-hash",
            no_repeat_sets=_no_repeat(),
            bootstrap_iterations=10,
            source_provenance=source_provenance,
            analysis_provenance=analysis_provenance,
        )


def test_formal_runner_reverifies_supplied_analysis_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "raw/manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("analysis.runner.validate_frozen_manifest", lambda *args, **kwargs: None)

    def reject(_root: Path, _provenance: object) -> None:
        raise DatasetContractError("forged supplied provenance")

    monkeypatch.setattr("analysis.runner.verify_analysis_provenance", reject)

    with pytest.raises(DatasetContractError, match="forged supplied provenance"):
        run_formal_analysis(
            manifest,
            tmp_path / "out",
            verified_analysis_provenance={
                "analysis_commit": "f" * 40,
                "analysis_code_committed_at_utc": "2026-07-14T00:00:00Z",
                "analysis_code_sha256": "e" * 64,
                "analysis_code_files": {"analysis/results.py": "d" * 64},
            },
        )


def test_results_contract_replacement_is_marker_scoped_and_idempotent(
    tmp_path: Path,
) -> None:
    target = tmp_path / "results_contract.md"
    target.write_text(
        "# Human-owned heading\n\n"
        "<!-- BEGIN S3 GENERATED RESULTS -->\nold\n"
        "<!-- END S3 GENERATED RESULTS -->\n\n"
        "## Human-owned interpretation\n",
        encoding="utf-8",
    )
    generated = (
        "<!-- BEGIN S3 GENERATED RESULTS -->\n"
        "```json\n{\"status\": \"generated\"}\n```\n"
        "<!-- END S3 GENERATED RESULTS -->\n"
    )

    replace_results_contract(target, generated)
    first = target.read_bytes()
    replace_results_contract(target, generated)

    assert target.read_bytes() == first
    text = first.decode("utf-8")
    assert text.startswith("# Human-owned heading\n\n")
    assert text.endswith("\n## Human-owned interpretation\n")
    assert generated.rstrip("\n") in text
