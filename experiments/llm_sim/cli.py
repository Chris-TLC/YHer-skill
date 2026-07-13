"""Command-line entry point for the optional S2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import FROZEN_RUN_ID, load_frozen_config
from .panel import load_annotation_map
from .runner import LLMSimulationRunner
from .store import SimulationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YHer S2 simulated-persona evaluation")
    parser.add_argument(
        "--output-root",
        default=f"data/sim_store/llm_personas/{FROZEN_RUN_ID}",
        help="isolated simulation root; never a local_store path",
    )
    parser.add_argument("--prepare-only", action="store_true", help="freeze the panel without network calls")
    parser.add_argument("--live", action="store_true", help="explicitly allow official provider HTTP calls")
    parser.add_argument("--provider", dest="providers", action="append", help="provider (repeatable)")
    parser.add_argument("--model", action="append", default=[], help="provider=model override (repeatable)")
    parser.add_argument(
        "--annotation-map",
        help="explicit item/failure/target-option JSON frozen into the simulation store",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--prompt-revision",
        type=int,
        choices=(0, 1),
        default=0,
        help="frozen calibration prompt revision (at most one rewrite)",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_frozen_config()
    store = SimulationStore(Path(args.output_root))
    annotation_map = (
        load_annotation_map(args.annotation_map)
        if args.annotation_map
        else None
    )
    runner = LLMSimulationRunner(
        store=store,
        study_seed=int(config.raw["study_seed"]),
        annotation_map=annotation_map,
        annotation_map_source=args.annotation_map,
    )
    # Preparation is always the first operation.  A non-live invocation can
    # therefore be safely used in CI or by a reviewer with no credentials.
    preparation = runner.prepare()
    if args.prepare_only or not args.live:
        print(json.dumps(preparation, ensure_ascii=False, sort_keys=True))
        return 0
    providers = tuple(args.providers or config.providers)
    unknown = sorted(set(providers) - set(config.providers))
    if unknown:
        parser.error("unknown provider(s): " + ", ".join(unknown))
    model_overrides: dict[str, str] = {}
    for raw in args.model:
        if "=" not in raw:
            parser.error("--model must use provider=model")
        provider, model = raw.split("=", 1)
        if provider not in config.providers or not model.strip():
            parser.error("--model must name a configured provider and non-empty model")
        model_overrides[provider] = model.strip()
    results = []
    for provider in providers:
        results.append(
            runner.run_provider(
                provider,
                model=model_overrides.get(provider),
                max_items=args.max_items or config.max_items,
                resume=not args.no_resume,
                prompt_revision=args.prompt_revision,
            )
        )
    # Only aggregate metadata is printed; provider response bodies and keys are
    # intentionally absent from the CLI output.
    print(
        json.dumps(
            {
                "simulated": True,
                "run_id": config.run_id,
                "record_type": "llm_sim_cli_summary",
                "prompt_revision": args.prompt_revision,
                "providers": [result["provider"] for result in results],
                "status": {result["provider"]: result["status"] for result in results},
                "accounting": {result["provider"]: result["accounting"] for result in results},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
