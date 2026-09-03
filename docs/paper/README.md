# Engineering Evidence Report

**File:** [`YHer_Engineering_Evidence_Report.pdf`](YHer_Engineering_Evidence_Report.pdf) (16 pages)
**Source:** [`source/`](source/) — LaTeX + BibTeX + figures, reproducible build
**Genre:** Engineering evidence report, not a peer-reviewed publication and not an arXiv preprint.

## What this document is

This report describes YHer Chemistry as a system of evidence, not of promises:

- **What was built** — the four-state diagnostic engine, EIG selection, signed recommendations, fail-closed browser contract.
- **What was measured** — the data funnel (6,083 raw slices → 1,202 serviceable items, 19.8% yield), the generation boundary (87% / 65% / 60%), and the architecture audit (4 keep / 10 modify / 5 replace / 2 downgrade across 21 components).
- **What was *not* validated** — no real-student trials, no learning-effect evidence. The limitations section is not an afterthought; it is the frame.

## Independent verification

The final draft was checked by an independent audit (2026-09-02): 23 references verified against Crossref/publisher records (16 exact, 6 metadata fixes, 1 replaced with the verified AIED 2013 record), every headline number re-derived from the data files in this repository, and a clean compile (0 undefined references, 0 TeX errors). The corrections it found are incorporated in this copy.

## Build

Requires a LaTeX toolchain (`pdflatex` + `bibtex`):

```bash
cd source
bash compile.sh
```

The PDF in this directory is the built output of that source.
