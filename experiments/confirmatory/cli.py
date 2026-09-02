"""Command-line entry point for validate-only, smoke, and controller execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import REPO_ROOT, load_frozen_config
from .runner import execute, validate_definition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "sim_store" / "confirmatory",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--run-id", default="confirmatory-v1")
    parser.add_argument("--runner-commit")
    parser.add_argument("--experiment-tag")
    parser.add_argument("--run-started-at-utc")
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_frozen_config(args.config) if args.config else load_frozen_config()
    if args.validate_only:
        print(json.dumps(validate_definition(config), ensure_ascii=False, sort_keys=True))
        return 0
    if not args.runner_commit or not args.experiment_tag or not args.run_started_at_utc:
        raise SystemExit(
            "data execution requires --runner-commit, --experiment-tag, and "
            "--run-started-at-utc"
        )
    result = execute(
        config,
        output_root=args.output_root,
        run_id=args.run_id,
        workers=args.workers,
        resume=args.resume,
        limit_shards=args.limit_shards,
        runner_commit=args.runner_commit,
        experiment_tag=args.experiment_tag,
        run_started_at_utc=args.run_started_at_utc,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
