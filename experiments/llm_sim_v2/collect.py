"""Explicit-live entry point for Persona v2 pilot/main collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.llm_sim.transport import HTTPProviderTransport

from .runner import (
    BudgetLedger,
    V2ProviderRunner,
    enumerate_tasks,
    load_runtime_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--secrets-root",
        help="read-only .env location for transport; never copied into the run store",
    )
    parser.add_argument("--phase", choices=("pilot", "main"), required=True)
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument(
        "--output-base",
        default="data/sim_store/llm_personas",
        help="v2-only base; phase roots are created below the frozen run ID",
    )
    parser.add_argument("--live", action="store_true", help="required to permit provider HTTP calls")
    parser.add_argument(
        "--limit",
        type=int,
        help="development-only task limit; forbidden for a formal phase unless --allow-partial is set",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--soft-warning", type=float, default=300.0)
    parser.add_argument("--hard-fuse", type=float, default=450.0)
    return parser


def run_collection(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.live:
        raise SystemExit("provider HTTP is disabled without --live")
    if args.limit is not None and not args.allow_partial:
        raise SystemExit("--limit requires --allow-partial and is not a formal collection")
    root = Path(args.repo_root).expanduser().resolve(strict=True)
    secrets_root = Path(args.secrets_root or args.repo_root).expanduser().resolve(strict=True)
    contract = load_runtime_contract(root)
    configured = tuple(args.providers or contract.config[args.phase]["providers"])
    unknown = sorted(set(configured) - set(contract.config["providers"]))
    if unknown:
        raise SystemExit("unknown frozen provider(s): " + ", ".join(unknown))
    tasks = enumerate_tasks(contract, phase=args.phase)
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
    results: list[dict[str, object]] = []
    for provider in configured:
        transport = HTTPProviderTransport.from_environment(
            provider,
            repo_root=secrets_root,
            version="v2",
        )
        runner = V2ProviderRunner(
            contract=contract,
            output_base=root / args.output_base,
            phase=args.phase,
            provider=provider,
            transport=transport,
            budget=BudgetLedger(
                soft_warning_yuan=float(args.soft_warning),
                hard_fuse_yuan=float(args.hard_fuse),
            ),
        )
        results.append(runner.run_tasks(tasks))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_collection(args)
    print(json.dumps({"providers": results}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
