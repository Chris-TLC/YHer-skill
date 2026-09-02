from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from analysis.dataset import DatasetContractError
from analysis.metrics import MetricValue
from analysis.registry import MetricRegistry


def _metric(metric_id: str, value: float) -> MetricValue:
    return MetricValue(
        metric_id=metric_id,
        value=value,
        numerator=value * 10,
        denominator=10,
        weighting="equal_target_then_replicate",
        n_target=2,
        n_pair=10,
        raw_hash="raw",
        ci_low=value - 0.1,
        ci_high=value + 0.1,
    )


def test_registry_requires_complete_provenance_and_unique_metric_ids() -> None:
    registry = MetricRegistry(raw_hash="raw")
    registry.add(_metric("b", 0.2))
    registry.add(_metric("a", 0.1))

    assert [row["metric_id"] for row in registry.records()] == ["a", "b"]
    assert all(
        {
            "numerator",
            "denominator",
            "weighting",
            "n_target",
            "n_pair",
            "raw_hash",
        }
        <= set(row)
        for row in registry.records()
    )

    with pytest.raises(DatasetContractError, match="duplicate metric_id"):
        registry.add(_metric("a", 0.3))
    with pytest.raises(DatasetContractError, match="raw_hash"):
        registry.add(replace(_metric("c", 0.3), raw_hash="other"))
    with pytest.raises(DatasetContractError, match="finite"):
        registry.add(_metric("d", float("nan")))


def test_registry_json_and_csv_are_byte_deterministic(tmp_path: Path) -> None:
    registry = MetricRegistry(raw_hash="raw")
    registry.add(_metric("b", 0.2))
    registry.add(_metric("a", 0.1))

    first = tmp_path / "first"
    second = tmp_path / "second"
    registry.write(first)
    registry.write(second)

    assert (first / "metric_registry.json").read_bytes() == (
        second / "metric_registry.json"
    ).read_bytes()
    assert (first / "metric_registry.csv").read_bytes() == (
        second / "metric_registry.csv"
    ).read_bytes()
