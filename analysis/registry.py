"""Machine-auditable registry for every reported numeric result."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .dataset import DatasetContractError
from .metrics import MetricValue


FIELDS = (
    "metric_id",
    "value",
    "numerator",
    "denominator",
    "weighting",
    "n_target",
    "n_pair",
    "raw_hash",
    "ci_low",
    "ci_high",
)


class MetricRegistry:
    def __init__(self, *, raw_hash: str):
        if not raw_hash:
            raise DatasetContractError("registry raw_hash must be non-empty")
        self.raw_hash = raw_hash
        self._metrics: dict[str, MetricValue] = {}

    def add(self, metric: MetricValue) -> None:
        if metric.metric_id in self._metrics:
            raise DatasetContractError(f"duplicate metric_id: {metric.metric_id}")
        if metric.raw_hash != self.raw_hash:
            raise DatasetContractError(
                f"metric raw_hash {metric.raw_hash!r} does not match registry raw_hash"
            )
        numeric = (metric.value, metric.numerator)
        if metric.ci_low is not None:
            numeric += (metric.ci_low,)
        if metric.ci_high is not None:
            numeric += (metric.ci_high,)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise DatasetContractError(f"metric {metric.metric_id} is not finite")
        if metric.denominator <= 0 or metric.n_target <= 0 or metric.n_pair <= 0:
            raise DatasetContractError(
                f"metric {metric.metric_id} has a non-positive audit denominator"
            )
        if not metric.weighting:
            raise DatasetContractError(f"metric {metric.metric_id} lacks weighting")
        if (
            metric.ci_low is not None
            and metric.ci_high is not None
            and metric.ci_low > metric.ci_high
        ):
            raise DatasetContractError(f"metric {metric.metric_id} has reversed CI")
        self._metrics[metric.metric_id] = metric

    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {field: getattr(self._metrics[metric_id], field) for field in FIELDS}
            for metric_id in sorted(self._metrics)
        )

    def write(self, output_dir: Path | str) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        records = self.records()
        json_path = root / "metric_registry.json"
        json_path.write_text(
            json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        csv_path = root / "metric_registry.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
