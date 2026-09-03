#!/usr/bin/env bash
# Build a GitHub Release archive: git archive zip (no .venv; old data stays outside the package)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "snapshot")
OUT_DIR="$ROOT_DIR/dist"
mkdir -p "$OUT_DIR"

# Tracked files only; export-ignore drops the big JSONL (data lives in the repo / on HF)
git archive --format=zip -o "$OUT_DIR/yher-skill-$VERSION.zip" HEAD

echo "written: $OUT_DIR/yher-skill-$VERSION.zip"
ls -lh "$OUT_DIR"/*.zip | tail -1
