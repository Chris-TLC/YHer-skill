#!/usr/bin/env python3
"""Deduplicate dotenv keys and enforce owner-only permissions without logging values."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


_DEFINITION = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


class EnvHardeningError(RuntimeError):
    pass


def harden_env_file(path: Path) -> dict[str, Any]:
    """Keep each key's last effective definition and chmod the file to 0600."""
    path = Path(path)
    if path.is_symlink():
        raise EnvHardeningError(f"refusing symlink: {path}")
    if not path.is_file():
        raise EnvHardeningError(f"not a regular file: {path}")

    mode_before = os.stat(path).st_mode & 0o777
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    keys_by_index: dict[int, str] = {}
    counts: Counter[str] = Counter()
    last_index: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _DEFINITION.match(line)
        if not match:
            continue
        key = match.group(1)
        keys_by_index[index] = key
        counts[key] += 1
        last_index[key] = index

    duplicate_keys_removed = {
        key: count - 1 for key, count in sorted(counts.items()) if count > 1
    }
    if duplicate_keys_removed:
        retained = [
            line
            for index, line in enumerate(lines)
            if index not in keys_by_index or last_index[keys_by_index[index]] == index
        ]
        _atomic_replace(path, "".join(retained))
    else:
        path.chmod(0o600)

    return {
        "path": str(path),
        "duplicate_keys_removed": duplicate_keys_removed,
        "mode_before": f"{mode_before:04o}",
        "mode_after": f"{os.stat(path).st_mode & 0o777:04o}",
    }


def _atomic_replace(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [harden_env_file(path) for path in args.paths]
    print(json.dumps(reports, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
