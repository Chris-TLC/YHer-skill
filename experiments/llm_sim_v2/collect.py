"""Explicit-live entry point for Persona v2 pilot/main collection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

from experiments.llm_sim.transport import HTTPProviderTransport

from .runner import (
    BudgetLedger,
    V2ProviderRunner,
    build_phase_provenance,
    enumerate_tasks,
    load_runtime_contract,
    UNKNOWN_ATTEMPT_RESERVE_YUAN,
    validate_formal_phase_provenance,
    verify_prior_cost_ledger,
    verify_runtime_task_manifest,
    verify_phase_provenance_against_contract,
    write_phase_provenance,
)
from .store import RUN_ID


def _run_root(output_base: str | Path) -> Path:
    base = Path(output_base).expanduser().resolve(strict=False)
    return base if base.name == RUN_ID else base / RUN_ID


def _reconciled_record_cost(
    value: Mapping[str, object],
    *,
    path: Path,
    unknown_attempt_reserve_yuan: float,
) -> dict[str, float]:
    attempts = value.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"stored v2 record lacks an attempt ledger: {path}")
    known = 0.0
    reserve = 0.0
    unknown_count = 0
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError(f"stored v2 record attempt is invalid: {path}")
        if attempt.get("cost_known") is True:
            try:
                cost = float(attempt["cost_yuan"])
                attempt_reserve = float(attempt.get("cost_reserve_yuan", 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"stored v2 known attempt cost is invalid: {path}") from exc
            if (
                not math.isfinite(cost)
                or cost < 0
                or attempt_reserve != 0.0
                or attempt.get("billing_ambiguity") is not False
            ):
                raise ValueError(f"stored v2 known attempt cost is invalid: {path}")
            known += cost
        elif attempt.get("cost_known") is False:
            try:
                attempt_reserve = float(attempt["cost_reserve_yuan"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"stored v2 unknown attempt reserve is invalid: {path}") from exc
            if (
                attempt.get("cost_yuan") is not None
                or attempt.get("billing_ambiguity") is not True
                or attempt_reserve != float(unknown_attempt_reserve_yuan)
            ):
                raise ValueError(f"stored v2 unknown attempt reserve is invalid: {path}")
            reserve += attempt_reserve
            unknown_count += 1
        else:
            raise ValueError(f"stored v2 attempt lacks cost identity: {path}")
    known = round(known, 8)
    reserve = round(reserve, 8)
    accounted = round(known + reserve, 8)
    try:
        stored_known = float(value["known_cost_yuan"])
        stored_reserve = float(value["unknown_cost_reserve_yuan"])
        stored_accounted = float(value["cost_yuan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"stored v2 record cost reconciliation fields are invalid: {path}") from exc
    if (
        not math.isclose(stored_known, known, rel_tol=0.0, abs_tol=1e-8)
        or not math.isclose(stored_reserve, reserve, rel_tol=0.0, abs_tol=1e-8)
        or not math.isclose(stored_accounted, accounted, rel_tol=0.0, abs_tol=1e-8)
        or value.get("has_unknown_cost_attempts") is not bool(unknown_count)
        or value.get("needs_user") is not bool(unknown_count)
    ):
        raise ValueError(f"stored v2 record attempt costs do not reconcile: {path}")
    return {
        "known_cost_yuan": known,
        "unknown_cost_reserve_yuan": reserve,
        "accounted_cost_yuan": accounted,
        "unknown_attempt_count": float(unknown_count),
    }


def _stored_record_costs(
    output_base: str | Path,
    *,
    phases: Sequence[str],
    unknown_attempt_reserve_yuan: float = UNKNOWN_ATTEMPT_RESERVE_YUAN,
) -> dict[str, float]:
    run_root = _run_root(output_base)
    totals = {
        "known_cost_yuan": 0.0,
        "unknown_cost_reserve_yuan": 0.0,
        "accounted_cost_yuan": 0.0,
        "unknown_attempt_count": 0.0,
    }
    seen: set[tuple[str, str, str]] = set()
    for phase_name in phases:
        if phase_name not in {"pilot", "main"}:
            raise ValueError("phase must be pilot or main")
        records_root = run_root / phase_name / "records"
        if not records_root.exists():
            continue
        for path in sorted(records_root.rglob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid stored v2 cost record: {path}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"invalid stored v2 cost record: {path}")
            provider = str(value.get("provider") or path.parent.name)
            task_id = str(value.get("task_id") or path.stem)
            key = (phase_name, provider, task_id)
            if (
                value.get("run_id") != RUN_ID
                or value.get("phase") != phase_name
                or value.get("analysis_population") != phase_name
                or key in seen
            ):
                raise ValueError(
                    f"stored v2 record run/phase/cost identity is invalid: {path}"
                )
            seen.add(key)
            reconciled = _reconciled_record_cost(
                value,
                path=path,
                unknown_attempt_reserve_yuan=unknown_attempt_reserve_yuan,
            )
            for field in totals:
                totals[field] += reconciled[field]
    return {field: round(value, 8) for field, value in totals.items()}


def existing_phase_cost(output_base: str | Path, *, phase: str) -> float:
    phase_name = str(phase).strip().lower()
    return _stored_record_costs(output_base, phases=(phase_name,))[
        "accounted_cost_yuan"
    ]


def existing_run_cost(output_base: str | Path) -> float:
    return _stored_record_costs(output_base, phases=("pilot", "main"))[
        "accounted_cost_yuan"
    ]


def resolve_collection_scope(
    *,
    frozen_providers: Sequence[str],
    requested_providers: Sequence[str] | None,
    limit: int | None,
    allow_partial: bool,
) -> dict[str, object]:
    frozen = tuple(str(value).strip().lower() for value in frozen_providers)
    selected = tuple(
        str(value).strip().lower()
        for value in (requested_providers if requested_providers is not None else frozen)
    )
    if len(selected) != len(set(selected)):
        raise SystemExit("duplicate providers are forbidden")
    changed = selected != frozen or limit is not None
    if changed and not allow_partial:
        raise SystemExit("provider override or --limit requires --allow-partial")
    development_only = bool(allow_partial or changed)
    return {
        "collection_mode": "development_partial" if development_only else "formal",
        "development_only": development_only,
        "partial": development_only,
        "formal_analysis_eligible": not development_only,
        "frozen_providers": list(frozen),
        "selected_providers": list(selected),
        "task_limit": limit,
    }


def validate_budget_thresholds(
    *,
    soft_warning_yuan: float,
    hard_fuse_yuan: float,
    frozen_soft_warning_yuan: float,
    frozen_hard_fuse_yuan: float,
) -> None:
    soft = float(soft_warning_yuan)
    hard = float(hard_fuse_yuan)
    frozen_soft = float(frozen_soft_warning_yuan)
    frozen_hard = float(frozen_hard_fuse_yuan)
    if hard > 450.0 or soft != frozen_soft or hard != frozen_hard:
        raise SystemExit(
            f"formal budget thresholds are frozen at CNY {frozen_soft:g}/{frozen_hard:g}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def reconcile_run_budget_ledger(
    output_base: str | Path,
    *,
    prior_cost_ledger: Mapping[str, object],
    soft_warning_yuan: float,
    hard_fuse_yuan: float,
) -> dict[str, object]:
    verify_prior_cost_ledger(prior_cost_ledger)
    prior_known = float(prior_cost_ledger["known_cost_yuan"])
    prior_reserve = float(prior_cost_ledger["pre_run_ambiguity_reserve_yuan"])
    prior_total = float(prior_cost_ledger["pre_run_total_bound_yuan"])
    unknown_attempt_reserve = float(
        prior_cost_ledger["unknown_attempt_reserve_yuan"]
    )
    prior_digest = str(prior_cost_ledger["prior_cost_ledger_sha256"])
    validate_budget_thresholds(
        soft_warning_yuan=soft_warning_yuan,
        hard_fuse_yuan=hard_fuse_yuan,
        frozen_soft_warning_yuan=300.0,
        frozen_hard_fuse_yuan=450.0,
    )
    run_root = _run_root(output_base)
    path = run_root / "run_budget_ledger.json"
    previous: Mapping[str, object] | None = None
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("run budget ledger is invalid") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError("run budget ledger is invalid")
        previous = loaded
        if (
            loaded.get("schema_version") != "yher.llm_sim_v2.run_budget_ledger.v2"
            or loaded.get("run_id") != RUN_ID
            or loaded.get("prior_cost_ledger_sha256") != prior_digest
            or float(loaded.get("prior_documented_cost_yuan", -1)) != prior_total
            or float(loaded.get("soft_warning_yuan", -1)) != float(soft_warning_yuan)
            or float(loaded.get("hard_fuse_yuan", -1)) != float(hard_fuse_yuan)
        ):
            raise ValueError("machine prior cost or frozen budget identity changed")
    record_costs = _stored_record_costs(
        output_base,
        phases=("pilot", "main"),
        unknown_attempt_reserve_yuan=unknown_attempt_reserve,
    )
    record_cost = record_costs["accounted_cost_yuan"]
    if previous is not None and record_cost + 1e-12 < float(
        previous.get("immutable_record_cost_yuan", 0.0)
    ):
        raise ValueError("immutable run record cost decreased")
    needs_user = record_costs["unknown_attempt_count"] > 0
    ledger: dict[str, object] = {
        "schema_version": "yher.llm_sim_v2.run_budget_ledger.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "prior_cost_ledger_sha256": prior_digest,
        "prior_known_cost_yuan": round(prior_known, 8),
        "prior_ambiguity_reserve_yuan": round(prior_reserve, 8),
        "prior_documented_cost_yuan": round(prior_total, 8),
        "unknown_attempt_reserve_yuan": unknown_attempt_reserve,
        "immutable_record_known_cost_yuan": record_costs["known_cost_yuan"],
        "immutable_record_unknown_reserve_yuan": record_costs[
            "unknown_cost_reserve_yuan"
        ],
        "immutable_record_cost_yuan": round(record_cost, 8),
        "total_known_cost_yuan": round(
            prior_known + record_costs["known_cost_yuan"], 8
        ),
        "total_unknown_reserve_yuan": round(
            prior_reserve + record_costs["unknown_cost_reserve_yuan"], 8
        ),
        "total_accounted_cost_yuan": round(prior_total + record_cost, 8),
        "needs_user": needs_user,
        "needs_user_reasons": (
            ["unknown_provider_billing_reserved"] if needs_user else []
        ),
        "soft_warning_yuan": float(soft_warning_yuan),
        "hard_fuse_yuan": float(hard_fuse_yuan),
        "updated_at_utc": _utc_now(),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return ledger


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
        raise SystemExit("--limit requires --allow-partial and is not formal collection")
    root = Path(args.repo_root).expanduser().resolve(strict=True)
    secrets_root = Path(args.secrets_root or args.repo_root).expanduser().resolve(strict=True)
    contract = load_runtime_contract(root)
    if contract.runtime_manifest is None:
        raise SystemExit("runtime task manifest is not frozen; live collection is forbidden")
    runtime_proof = verify_runtime_task_manifest(
        contract, contract.runtime_manifest, verify_git=True
    )
    frozen_providers = tuple(contract.config[args.phase]["providers"])
    scope = resolve_collection_scope(
        frozen_providers=frozen_providers,
        requested_providers=args.providers,
        limit=args.limit,
        allow_partial=bool(args.allow_partial),
    )
    configured = tuple(scope["selected_providers"])
    unknown = sorted(set(configured) - set(contract.config["providers"]))
    if unknown:
        raise SystemExit("unknown frozen provider(s): " + ", ".join(unknown))
    tasks = enumerate_tasks(contract, phase=args.phase)
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
    output_base = root / args.output_base
    frozen_budget = contract.config["budget_yuan"]
    validate_budget_thresholds(
        soft_warning_yuan=float(args.soft_warning),
        hard_fuse_yuan=float(args.hard_fuse),
        frozen_soft_warning_yuan=float(frozen_budget["soft_warning"]),
        frozen_hard_fuse_yuan=float(frozen_budget["hard_fuse"]),
    )
    prior_cost_ledger = contract.prior_cost_ledger
    verify_prior_cost_ledger(prior_cost_ledger)
    run_budget = reconcile_run_budget_ledger(
        output_base,
        prior_cost_ledger=prior_cost_ledger,
        soft_warning_yuan=float(args.soft_warning),
        hard_fuse_yuan=float(args.hard_fuse),
    )
    phase_store_path = _run_root(output_base) / args.phase / "phase_provenance.json"
    if phase_store_path.is_file():
        loaded_phase = json.loads(phase_store_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_phase, Mapping):
            raise ValueError("stored phase provenance is invalid")
        phase_provenance = dict(loaded_phase)
        verify_phase_provenance_against_contract(
            phase_provenance,
            contract=contract,
            runtime_manifest=contract.runtime_manifest,
            runtime_proof=runtime_proof,
            tasks=tasks,
            collection_scope=scope,
        )
    else:
        phase_provenance = build_phase_provenance(
            contract,
            runtime_manifest=contract.runtime_manifest,
            runtime_proof=runtime_proof,
            phase=args.phase,
            tasks=tasks,
            collection_scope=scope,
            prior_cost_ledger=prior_cost_ledger,
            first_observation_at_utc=runtime_proof["git_proof"][
                "observation_timestamp"
            ],
        )
        write_phase_provenance(output_base, phase=phase_provenance)
    if not scope["development_only"]:
        validate_formal_phase_provenance(phase_provenance)
    budget = BudgetLedger(
        soft_warning_yuan=float(args.soft_warning),
        hard_fuse_yuan=float(args.hard_fuse),
        initial_cost_yuan=float(run_budget["total_accounted_cost_yuan"]),
    )
    results: list[dict[str, object]] = []
    try:
        for provider in configured:
            try:
                transport = HTTPProviderTransport.from_environment(
                    provider,
                    repo_root=secrets_root,
                    version="v2",
                )
            except Exception:
                unavailable = V2ProviderRunner(
                    contract=contract,
                    output_base=output_base,
                    phase=args.phase,
                    provider=provider,
                    transport=None,
                    budget=budget,
                    phase_provenance=phase_provenance,
                )
                results.append(
                    unavailable.write_unavailable_manifest(
                        tasks,
                        error_category=(
                            "transport_configuration_or_credential_unavailable"
                        ),
                    )
                )
                continue
            runner = V2ProviderRunner(
                contract=contract,
                output_base=output_base,
                phase=args.phase,
                provider=provider,
                transport=transport,
                budget=budget,
                phase_provenance=phase_provenance,
            )
            results.append(runner.run_tasks(tasks))
    finally:
        reconcile_run_budget_ledger(
            output_base,
            prior_cost_ledger=prior_cost_ledger,
            soft_warning_yuan=float(args.soft_warning),
            hard_fuse_yuan=float(args.hard_fuse),
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_collection(args)
    print(json.dumps({"providers": results}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
