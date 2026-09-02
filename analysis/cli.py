"""Command-line interface for deterministic paper result generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .runner import run_formal_analysis


DEFAULT_MANIFEST = Path(
    "data/sim_store/confirmatory/confirmatory-v1/manifest.json"
)
DEFAULT_OUTPUT = Path("/tmp/yher_sprint2/paper_results")
DEFAULT_RESULTS_CONTRACT = Path("docs/paper/results_contract.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen simulation and generate paper results."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--results-contract",
        type=Path,
        default=DEFAULT_RESULTS_CONTRACT,
        help="Marker-delimited paper results contract to update after validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_formal_analysis(
        args.manifest,
        args.output,
        results_contract_path=args.results_contract,
    )
    return 0
