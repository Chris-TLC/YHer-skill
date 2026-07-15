"""Contract tests for the frozen P2 supply-bound illustration."""

from __future__ import annotations

from pathlib import Path
from decimal import Decimal
from collections.abc import Iterator, Mapping

import pytest


def _candidate(
    chunk_id: str,
    *,
    target: str = "基本操作",
    physical_key: str | None = None,
    seconds: int = 10,
    role: str = "drill",
):
    from experiments import p2_illustrative as p2

    return p2.Candidate(
        chunk_id=chunk_id,
        target=target,
        physical_key=physical_key or f"physical:{chunk_id}",
        charged_seconds=seconds,
        role=role,
        difficulty="T2",
        source_line=1,
        raw_row={"chunk_id": chunk_id},
    )


def test_p2_module_exists() -> None:
    module_path = Path(__file__).parents[1] / "experiments" / "p2_illustrative.py"
    assert module_path.is_file()


def test_hash_gate_rejects_bytes_before_json_parsing(tmp_path: Path) -> None:
    from experiments import p2_illustrative as p2

    drifted = tmp_path / "drifted.json"
    drifted.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(p2.InputDriftError, match="SHA-256 drift"):
        p2.verify_sha256(drifted, "0" * 64, label="fixture")


def test_frozen_candidate_fixture_is_eight_rows_and_three_physical_sources() -> None:
    from experiments import p2_illustrative as p2

    subset = p2.load_candidate_subset()

    assert subset.audit_declared_sha256 == p2.EXPECTED_SUBSET_AUDIT_SHA256
    assert subset.canonical_rows_sha256 == (
        "e080008a40e514bf57e95a5c9905c9ba469c1e9c976b0e1ec465e684d55ff34d"
    )
    assert len(subset.candidates) == 8
    assert {candidate.physical_key for candidate in subset.candidates} == {
        "BV1aComYMEms#P102",
        "BV18t4y1a7eD#P001",
        "BV1JT421C7WS#P001",
    }
    assert {
        candidate.chunk_id: (
            candidate.target,
            candidate.charged_seconds,
            candidate.role,
        )
        for candidate in subset.candidates
    } == p2.EXPECTED_CANDIDATE_FACTS


def test_selector_uses_full_posterior_and_binary_slot_saturation() -> None:
    from experiments import p2_illustrative as p2

    review = _candidate("review", seconds=10, role="review")
    drill_fast = _candidate("drill-fast", seconds=1, role="drill")
    drill_duplicate = _candidate("drill-duplicate", seconds=2, role="drill")
    result = p2.select_candidates(
        (review, drill_fast, drill_duplicate),
        {"基本操作": (Decimal("0.6"), Decimal("0"), Decimal("0.4"), Decimal("0"))},
        budget_seconds=20,
    )

    assert [candidate.chunk_id for candidate in result.selected] == [
        "drill-fast",
        "review",
    ]
    terminal = result.trace[-1]
    duplicate = next(
        row for row in terminal["evaluations"] if row["chunk_id"] == "drill-duplicate"
    )
    assert duplicate["marginal_utility"] == "0"
    assert duplicate["rejection_reason"] == "binary_saturation"


def test_selector_stable_order_and_physical_deduplication() -> None:
    from experiments import p2_illustrative as p2

    lower_gain = _candidate("z-lower-gain", seconds=10, role="drill")
    higher_gain = _candidate(
        "b-higher-gain",
        target="烷烃",
        seconds=20,
        role="review",
        physical_key="shared",
    )
    same_physical = _candidate(
        "a-same-physical",
        target="烷烃",
        seconds=10,
        role="drill",
        physical_key="shared",
    )
    result = p2.select_candidates(
        (lower_gain, higher_gain, same_physical),
        {
            "基本操作": (Decimal("0.5"), Decimal("0"), Decimal("0.5"), Decimal("0")),
            "烷烃": (Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0")),
        },
        budget_seconds=20,
    )

    # Equal ratios (0.05) are resolved by larger marginal utility first.
    assert result.selected[0].chunk_id == "b-higher-gain"
    final_same_source = next(
        row
        for row in result.trace[-1]["evaluations"]
        if row["chunk_id"] == "a-same-physical"
    )
    assert final_same_source["rejection_reason"] == "physical_source_reuse"

    lexicographic = p2.select_candidates(
        (
            _candidate("z-equal", physical_key="p-z"),
            _candidate("a-equal", physical_key="p-a"),
        ),
        {"基本操作": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0"))},
        budget_seconds=10,
    )
    assert lexicographic.selected[0].chunk_id == "a-equal"


def test_selector_enforces_six_hundred_second_budget_on_frozen_supply() -> None:
    from experiments import p2_illustrative as p2

    subset = p2.load_candidate_subset()
    result = p2.select_candidates(
        subset.candidates,
        {
            "基本操作": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0")),
            "烷烃": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0")),
        },
        budget_seconds=600,
    )

    assert [candidate.chunk_id for candidate in result.selected] == [
        "BV1JT421C7WS#P001#c004_b",
        "BV1aComYMEms#P102#c000",
    ]
    assert sum(candidate.charged_seconds for candidate in result.selected) == 524
    assert len({candidate.physical_key for candidate in result.selected}) == len(
        result.selected
    )
    assert all(
        sum(candidate.charged_seconds for candidate in result.selected)
        <= step["budget_seconds"]
        for step in result.trace
    )


def test_standalone_selector_rejects_non_normalized_belief() -> None:
    from experiments import p2_illustrative as p2

    with pytest.raises(p2.P2ContractError, match="not normalized"):
        p2.select_candidates(
            (_candidate("invalid-mass"),),
            {
                "基本操作": (
                    Decimal("0.4"),
                    Decimal("0.4"),
                    Decimal("0.4"),
                    Decimal("0"),
                )
            },
        )


class _PoisonInvalidView(Mapping[str, object]):
    """An invalid view whose stored belief must be unreachable."""

    _values = {
        "nominal_budget": 15,
        "valid": False,
        "incomplete": True,
        "terminal_reason": "structural_failure",
    }

    def __getitem__(self, key: str) -> object:
        if key == "belief":
            raise AssertionError("invalid belief was read")
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        yield from (*self._values, "belief")

    def __len__(self) -> int:
        return len(self._values) + 1


def test_arm_c_invalid_view_is_masked_without_reading_stored_belief() -> None:
    from experiments import p2_illustrative as p2

    belief, status = p2.posterior_from_b15(
        _PoisonInvalidView(),
        expected_valid=False,
        context="基本操作/C/0",
    )

    assert belief is None
    assert status == "structural_failure"


def test_h1_fixture_loads_matched_b15_product_margins_and_expected_failures() -> None:
    from experiments import p2_illustrative as p2

    dataset = p2.load_h1_b15_dataset()

    assert dataset.source_manifest_sha256 == p2.EXPECTED_SOURCE_SHA256[
        "h1_h4_raw_manifest"
    ]
    assert len(dataset.shard_sha256) == 8
    assert len(dataset.records) == 1_200
    assert len(
        {
            (record.target, record.truth, record.arm, record.replicate)
            for record in dataset.records
        }
    ) == 1_200
    failures = [record for record in dataset.records if record.belief is None]
    assert len(failures) == 200
    failure_kinds = {
        (record.target, record.arm, record.diagnostic_status) for record in failures
    }
    assert failure_kinds == {("基本操作", "C", "structural_failure")}
    assert all(record.belief is not None for record in dataset.records if record not in failures)


def test_decision_manifest_has_sixteen_product_truth_cells_not_a_sample() -> None:
    from experiments import p2_illustrative as p2

    manifest = p2.build_decision_instances_manifest()

    assert len(manifest["truth_cells"]) == 16
    assert {row["analytic_integration_terms"] for row in manifest["truth_cells"]} == {
        2_500
    }
    assert {row["truth_cell_weight"] for row in manifest["truth_cells"]} == {
        "0.0625"
    }
    assert {row["component_weight"] for row in manifest["truth_cells"]} == {
        "0.0004"
    }
    assert manifest["analytic_integration_terms_per_arm"] == 40_000
    assert manifest["arms"] == ["oracle", "A", "B", "C"]
    assert manifest["reporting_unit"] == (
        "two_fixed_target_strata_each_with_50_programmatic_replicate_clusters"
    )


def test_profile_metrics_use_truth_slot_equivalence_and_four_minute_fields() -> None:
    from experiments import p2_illustrative as p2

    subset = p2.load_candidate_subset()
    profile = p2.compute_profile(
        arm="A",
        truths={"基本操作": "C", "烷烃": "M"},
        beliefs_by_target={
            "基本操作": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0")),
            "烷烃": (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0")),
        },
        candidates=subset.candidates,
        replicate_basic=0,
        replicate_alkane=0,
    )
    metrics = profile.metrics

    assert metrics["selected_seconds"] == 524
    assert metrics["mismatched_selected_seconds"] == 28
    assert metrics["missed_available_seconds"] == 94
    assert metrics["unused_budget_seconds"] == 76
    assert metrics["mismatched_selected_minutes"] == pytest.approx(28 / 60)
    assert metrics["missed_available_supply_minutes"] == pytest.approx(94 / 60)
    assert metrics["unobtainable_supply_minutes"] is None
    assert metrics["unused_budget_minutes"] == pytest.approx(76 / 60)
    assert metrics["unobtainable_reason"] == "no_frozen_role_compatible_dose"
    assert metrics["unobtainable_truth_slots"] == 0
    assert metrics["unsupported_posterior_mass"] == "0"


def test_arm_c_masks_basic_node_and_attributes_missed_oracle_slot_to_failure() -> None:
    from experiments import p2_illustrative as p2

    subset = p2.load_candidate_subset()
    profile = p2.compute_profile(
        arm="C",
        truths={"基本操作": "C", "烷烃": "M"},
        beliefs_by_target={
            "烷烃": (Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0")),
        },
        candidates=subset.candidates,
        failed_nodes=frozenset({"基本操作"}),
        replicate_basic=0,
        replicate_alkane=0,
    )
    metrics = profile.metrics

    assert metrics["belief_basic"] is None
    assert metrics["belief_reason_by_node"]["基本操作"] == "structural_failure"
    assert metrics["selected_seconds"] == 94
    assert metrics["missed_available_seconds"] == 496
    assert metrics["missed_available_by_cause_seconds"] == {
        "diagnostic_structural_failure": 496,
        "posterior_selection": 0,
        "budget_constraint": 0,
    }
    assert metrics["structural_failure_node_fraction"] == 0.5
    assert metrics["prescription_status"] == "partial_structural_failure"
    assert metrics["unobtainable_supply_minutes"] is None


def test_input_manifest_binds_amended_spec_and_marks_old_digest_non_gate() -> None:
    from experiments import p2_illustrative as p2

    subset = p2.load_candidate_subset()
    dataset = p2.load_h1_b15_dataset()
    manifest = p2.build_input_manifest(subset, dataset)

    assert manifest["spec"]["commit"] == p2.P2_SPEC_COMMIT
    assert manifest["spec"]["sha256"] == p2.P2_SPEC_SHA256
    assert manifest["candidate_subset"]["canonical_sha256"] == (
        p2.EXPECTED_SUBSET_CANONICAL_ROWS_SHA256
    )
    assert manifest["candidate_subset"]["audit_declared_unreproduced"] == (
        p2.EXPECTED_SUBSET_AUDIT_SHA256
    )
    assert manifest["candidate_subset"]["audit_declared_digest_is_gate"] is False
    assert manifest["hash_gate_status"] == "pass"


def test_target_stratified_bootstrap_draws_ten_thousand_paired_cluster_samples() -> None:
    from experiments import p2_illustrative as p2

    basic, alkane = p2.bootstrap_replicate_indices()
    basic_again, alkane_again = p2.bootstrap_replicate_indices()

    assert basic.shape == (10_000, 50)
    assert alkane.shape == (10_000, 50)
    assert (basic == basic_again).all()
    assert (alkane == alkane_again).all()
    assert not (basic == alkane).all()
    assert int(basic.min()) == 0 and int(basic.max()) == 49
    assert int(alkane.min()) == 0 and int(alkane.max()) == 49


def test_arm_c_trace_hash_excludes_masked_basic_margin_identity() -> None:
    from experiments import p2_illustrative as p2

    subset = p2.load_candidate_subset()
    kwargs = {
        "arm": "C",
        "beliefs_by_target": {
            "烷烃": (Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0"))
        },
        "candidates": subset.candidates,
        "failed_nodes": frozenset({"基本操作"}),
        "replicate_alkane": 7,
    }
    first = p2.compute_profile(
        **kwargs,
        truths={"基本操作": "M", "烷烃": "M"},
        replicate_basic=0,
    )
    second = p2.compute_profile(
        **kwargs,
        truths={"基本操作": "C", "烷烃": "M"},
        replicate_basic=49,
    )

    assert first.metrics["selector_trace_hash"] == second.metrics["selector_trace_hash"]
    assert first.metrics["profile_id"] != second.metrics["profile_id"]


def test_bootstrap_engine_uses_all_ten_thousand_resamples_without_p_values() -> None:
    import numpy as np

    from experiments import p2_illustrative as p2

    cube = np.empty((4, 4, 4, 50, 50, 2), dtype=float)
    cube[..., 0] = 120.0
    cube[..., 1] = 0.5
    result = p2.bootstrap_metric_cube(cube, ("selected_seconds", "failure_fraction"))

    assert result["attempted_resamples"] == 10_000
    assert result["defined_resamples"] == 10_000
    assert result["seed"] == 2026071505
    assert len(result["overall"]) == 8
    assert all(row["point"] == row["ci95_low"] == row["ci95_high"] for row in result["overall"])
    assert "p_value" not in str(result)


def test_machine_output_contract_lists_every_required_artifact() -> None:
    from experiments import p2_illustrative as p2

    assert p2.REQUIRED_OUTPUT_FILES == (
        "input_manifest.json",
        "candidate_subset.jsonl",
        "decision_instances_manifest.json",
        "selector_trace.jsonl",
        "profile_metrics.jsonl",
        "summary.json",
        "bootstrap.json",
        "figure_data.json",
        "p2_supply_bound_illustration.png",
        "p2_supply_bound_illustration.svg",
        "output_manifest.json",
    )
    assert callable(p2.run_analysis)


def test_publication_figure_is_supply_bound_and_labels_arm_c_failure(
    tmp_path: Path,
) -> None:
    from experiments import p2_illustrative as p2

    summary = {
        "overall": [
            {
                "arm": "A",
                "mismatched_selected_minutes": 3.2985,
                "missed_available_supply_minutes": 0.8238,
                "structural_failure_node_fraction": 0.0,
            },
            {
                "arm": "B",
                "mismatched_selected_minutes": 4.0303,
                "missed_available_supply_minutes": 0.7563,
                "structural_failure_node_fraction": 0.0,
            },
            {
                "arm": "C",
                "mismatched_selected_minutes": 1.5250,
                "missed_available_supply_minutes": 2.0667,
                "structural_failure_node_fraction": 0.5,
            },
        ]
    }
    png_path, svg_path = p2.render_publication_figure(summary, tmp_path)

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = svg_path.read_text(encoding="utf-8")
    assert "Supply-bound illustrative prescription outputs" in svg
    assert "Mechanically mismatched selected minutes" in svg
    assert "Missed available-supply minutes" in svg
    for arm, mismatch, missed in (
        ("A", "3.2985", "0.8238"),
        ("B", "4.0303", "0.7563"),
        ("C", "1.5250", "2.0667"),
    ):
        assert f"Arm {arm}" in svg
        assert mismatch in svg
        assert missed in svg
    assert "Arm C structural-failure node fraction: 0.5" in svg
    forbidden = ("learning minutes", "wasted minutes", "saved minutes")
    assert all(term not in svg.lower() for term in forbidden)
