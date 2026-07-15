"""Explicit-live entry point for Persona v2 pilot/main collection."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

from experiments.llm_sim.transport import HTTPProviderTransport

from .runner import (
    BudgetLedger,
    V2ProviderRunner,
    enumerate_tasks,
    load_runtime_contract,
    verify_runtime_task_manifest,
)
from .store import RUN_ID


def existing_phase_cost(output_base: str | Path, *, phase: str) -> float:
    phase_name = str(phase).strip().lower()
    if phase_name not in {"pilot", "main"}:
        raise ValueError("phase must be pilot or main")
    records_root = Path(output_base).expanduser().resolve(strict=False) / RUN_ID / phase_name / "records"
    if not records_root.exists():
        return 0.0
    total = 0.0
    seen: set[tuple[str, str]] = set()
    for path in sorted(records_root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid stored v2 cost record: {path}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid stored v2 cost record: {path}")
        provider = str(value.get("provider") or path.parent.name)
        task_id = str(value.get("task_id") or path.stem)
        key = (provider, task_id)
        try:
            cost = float(value["cost_yuan"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"stored v2 record has invalid cost: {path}") from exc
        if (
            value.get("run_id") != RUN_ID
            or value.get("phase") != phase_name
            or value.get("analysis_population") != phase_name
            or key in seen
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError(f"stored v2 record run/phase/cost identity is invalid: {path}")
        seen.add(key)
        total += cost
    return total


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
    if contract.runtime_manifest is None:
        raise SystemExit("runtime task manifest is not frozen; live collection is forbidden")
    verify_runtime_task_manifest(contract, contract.runtime_manifest, verify_git=True)
    configured = tuple(args.providers or contract.config[args.phase]["providers"])
    unknown = sorted(set(configured) - set(contract.config["providers"]))
    if unknown:
        raise SystemExit("unknown frozen provider(s): " + ", ".join(unknown))
    tasks = enumerate_tasks(contract, phase=args.phase)
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
    output_base = root / args.output_base
    budget = BudgetLedger(
        soft_warning_yuan=float(args.soft_warning),
        hard_fuse_yuan=float(args.hard_fuse),
        initial_cost_yuan=existing_phase_cost(output_base, phase=args.phase),
    )
    results: list[dict[str, object]] = []
    for provider in configured:
        transport = HTTPProviderTransport.from_environment(
            provider,
            repo_root=secrets_root,
            version="v2",
        )
        runner = V2ProviderRunner(
            contract=contract,
            output_base=output_base,
            phase=args.phase,
            provider=provider,
            transport=transport,
            budget=budget,
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
