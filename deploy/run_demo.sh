#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEMO_VENV="${YHER_DEMO_VENV:-$ROOT_DIR/.venv-demo}"

select_bootstrap_python() {
  local candidate resolved
  for candidate in python3.14 python3.13 python3.12 python3.11 /opt/homebrew/bin/python3 python3; do
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -n "$resolved" ]] && "$resolved" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
      printf '%s\n' "$resolved"
      return 0
    fi
  done
  return 1
}

if [[ -n "${YHER_PYTHON:-}" ]]; then
  PYTHON_BIN="$YHER_PYTHON"
elif [[ -x "$DEMO_VENV/bin/python" ]]; then
  PYTHON_BIN="$DEMO_VENV/bin/python"
else
  BOOTSTRAP_PYTHON="$(select_bootstrap_python || true)"
  if [[ -z "$BOOTSTRAP_PYTHON" ]]; then
    echo "YHer Demo requires Python 3.11 or newer." >&2
    exit 1
  fi
  "$BOOTSTRAP_PYTHON" -m venv "$DEMO_VENV"
  PYTHON_BIN="$DEMO_VENV/bin/python"
fi

if ! "$PYTHON_BIN" -c 'import fastapi, numpy, openai, uvicorn, yaml' >/dev/null 2>&1; then
  if [[ -n "${YHER_PYTHON:-}" ]]; then
    echo "YHER_PYTHON is missing Demo dependencies; install requirements-demo.txt." >&2
    exit 1
  fi
  "$PYTHON_BIN" -m pip install --disable-pip-version-check -r "$ROOT_DIR/requirements-demo.txt"
fi

UVICORN_ARGS=()
if [[ -f "$ROOT_DIR/.env" ]]; then
  UVICORN_ARGS+=(--env-file "$ROOT_DIR/.env")
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export YHER_ENABLE_PAID_LLM="${YHER_ENABLE_PAID_LLM:-1}"

exec "$PYTHON_BIN" -m uvicorn apps.demo_api:app \
  --host 127.0.0.1 \
  --port "${YHER_DEMO_PORT:-8700}" \
  --workers 1 \
  "${UVICORN_ARGS[@]}"
