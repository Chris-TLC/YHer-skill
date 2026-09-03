#!/usr/bin/env bash
# Build the YHer engineering evidence report.
# Requires pdflatex + bibtex on PATH. On macOS with MacTeX:
#   export PATH="/Library/TeX/texbin:$PATH"
set -u
cd "$(dirname "$0")" || exit 1

# Accept an explicit TeX bin directory, then common installs, then PATH.
for candidate in "${TEXBIN:-}" "/Library/TeX/texbin" "/usr/local/texlive/2026/bin/universal-darwin" "/usr/local/texlive/2025/bin/universal-darwin"; do
  [ -n "$candidate" ] && [ -x "$candidate/pdflatex" ] && { PATH="$candidate:$PATH"; break; }
done

command -v pdflatex >/dev/null 2>&1 || { echo "pdflatex not found on PATH" >&2; exit 1; }
command -v bibtex >/dev/null 2>&1 || { echo "bibtex not found on PATH" >&2; exit 1; }

rm -f main.aux main.bbl main.blg main.log
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
bibtex main >/dev/null 2>&1 || { echo "BIBTEX FAILED" >&2; exit 1; }
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true

if [ -f main.pdf ]; then
  echo "OK: $(grep -c 'Output written' main.log) output, $(grep -c undefined main.log) undefined, $(grep -c 'Overfull' main.log) overfull"
else
  echo "NO PDF" >&2
  grep -n '^!' main.log | head -10 >&2
  exit 1
fi
