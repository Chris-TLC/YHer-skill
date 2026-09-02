#!/usr/bin/env bash
# 打 GitHub Release 打包件:git 归档压缩包(不含 .venv/旧数据外置)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "snapshot")
OUT_DIR="$ROOT_DIR/dist"
mkdir -p "$OUT_DIR"

# 只打已跟踪文件;按 .gitattributes export-ignore 排除大 JSONL(数据走仓库/HF)
git archive --format=zip -o "$OUT_DIR/yher-skill-$VERSION.zip" HEAD

echo "written: $OUT_DIR/yher-skill-$VERSION.zip"
ls -lh "$OUT_DIR"/*.zip | tail -1
