# Audit History Index

The audit archive in this public repo = the **final-verdict** reports of the project's three rounds of system-level audits. The full process records (356 daily session ledger entries, batch-level intermediate reviews) stay inside the project and are not in this public repo.

## Index

### Round 1: Project-level audit (2026-07-01 → 07-06)

| File | Topic |
|---|---|
| `YHER_FULL_PROJECT_AUDIT_2026-07-01.md` | Full-project audit: architecture, data pipeline, quality-gate design |
| `GOLD_QUESTIONS_MODEL_REVIEW_REPORT_2026-07-01.md` | Gold-question model review |
| `VISUAL_ITEM_QUALITY_GATE_EXECUTION_PLAN_2026-07-01.md` | Visual-item quality gate execution plan |
| `VISUAL_ITEM_QUALITY_GATE_EXECUTION_REPORT_2026-07-01.md` | Visual-item quality gate execution report |
| `STRATEGIC_REVIEW_2026-07-06.md` | Strategic re-review (project position and priorities) |

### QA marathon (BATCH6–16): usability audit of 2,526 items

| File | Topic |
|---|---|
| `BATCH10_AUDIT_2026-07-05.md` | Batch 10: LaTeX / ion / figure-asset repair audit |
| `BATCH14_AUDIT_2026-07-06.md` | Batch 14: calibration-gate failure and KaTeX root cause (audit infrastructure must be same-source as the product) |
| `BATCH16_AUDIT_2026-07-06.md` | Batch 16: node-aware review with zero false positives; reviewer precision-gold methodology |

### AI question generation: five-round verification (2026-07-06 → 07-08)

| File | Topic |
|---|---|
| `NEIHUA_P1_AUDIT_2026-07-06.md` | Round 1: bare-generation baseline (first-pass accuracy ≈45%) |
| `NEIHUA_P2_AUDIT_2026-07-06.md` | Round 2: gated generation (six-gate spec) |
| `NEIHUA_MVP_AUDIT_2026-07-07.md` | MVP three-route verification (text / rendering / figure-anchored) |
| `NEIHUA_R2_AUDIT_2026-07-07.md` | R2 three-route rerun (an execution-layer honest FAIL is not a route FAIL) |
| `NEIHUA_R3_AUDIT_2026-07-07.md` | R3 close-out (style-transfer direction) |
| `NEIHUA_R4R5_AUDIT_2026-07-08.md` | R4+R5 final: five-round verification closed (65% / 60% fair) |

### Round 2: CEO engineering audit (2026-07-10)

`YHER_CEO_ENGINEERING_AUDIT_2026-07-10.md` — dual-dimension re-review of engineering investment and product judgment (verdict: direction 70% right, priorities 40% right).

### Round 3: literature-level architecture audit (2026-08-05 → 08-13, 11 reports)

6 research lanes (lane1–6) + 3 red-team attack passes + independent verification round:

| File | Topic |
|---|---|
| `lane1_measurement.md` | Measurement: binarized mastery vs four-state construct |
| `lane2_selection_stopping.md` | Selection and stopping: gap>0.45 retired; P(top1)+min_length |
| `lane3_memory_review.md` | Memory and review: FSRS-4.5 damping replaces hand-set constants |
| `lane4_recommendation.md` | Recommendation: multiplicative scoring kept; efficacy Beta-Binomial only as tie-break |
| `lane5_verification_profile.md` | Verification and profile: held-out early-stop 3→6; n=2 binarization forbidden |
| `lane6_math_pipeline.md` | Math pipeline: full-page transcription + MFD + SymPy full syntax gate |
| `redteam1_measurement_selection.md` | Red team 1: four-state 12-item ceiling 54–69% independently recomputed |
| `redteam2_memory_recommendation.md` | Red team 2: S-inflation 4,608 days recomputed; efficacy has no academic position |
| `redteam3_verification_pipeline_product.md` | Red team 3: n=2 lower bound 22.4%; SymPy 7 false equivalences / 5 reproduced; product-layer consistency |
| `VERIFICATION_ROUND4.md` | Independent verification round: multiple overturns (old FSRS calculation wrong by 5×) |
| `MASTER_AUDIT_REPORT_2026-08-13.md` | Final verdict: 21 components 4 keep / 10 modify / 5 replace / 2 downgrade; math MVP blueprint v0′ |

## Why these are public

- Audit reports are direct evidence of engineering rigor: each contains **independent reproduction commands, input hashes, per-item verdicts, and evidence grades**;
- The evolution across the three rounds records "what conclusion was overturned and why" — the standard form of a research record;
- Verdicts in the reports (gap>0.45 retired, FSRS-4.5 damping) are already in, or entering, the production engine.

## What is not public

- `ledger_archive/` (356 daily session ledger entries): too fine-grained a process record; contains internal incidents and flips, not suitable as public archive;
- Batch-level intermediate reports (BATCH6/8/9/11/12/13/15, R1, etc.): process documents, not final-verdict level.
