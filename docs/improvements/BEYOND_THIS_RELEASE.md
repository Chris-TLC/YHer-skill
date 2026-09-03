# Improvements known but deliberately outside this release

> Snapshot: 2026-09-02. This list records improvements with audit evidence that were intentionally left out of this release (because they need real student data, a user decision, or cross a "no novelty" boundary). Each item carries its status and trigger condition, so a future restart can pick it up directly.

## 1. "Not implemented" remediation items from the 21-component architecture audit (final verdict 2026-08-13)

> Final verdict file: `MASTER_AUDIT_REPORT_2026-08-13.md`. Except for the three items that shipped with this release (see §3):

| Component | Audit recommendation | Status | Trigger |
|---|---|---|---|
| Four-state → binary mastery + prerequisite prior + error-code routing | Modify (inference layer) | **not implemented** | Needs real-student calibration; the four UI labels can stay |
| EIG → threshold-calibrated (random ≈ EIG already explained) | Keep | not implemented | Once item-bank parameters are ready |
| Budget 1.5× → lognormal RT conversion | Keep (fallback) | not implemented | Needs response-time data |
| FSRS revisited constants → 4.5 damping | Replace | ✅ **implemented (2026-09-02)** | — |
| M→C projection → after binarization "affects due only" | Keep direction | not implemented | Ships with binarization |
| efficacy≡1.0 → Beta-Binomial shrinkage | Replace | not implemented | Needs real watch → retest data |
| held_out 2 → 3/6 early-stop protocol | Modify | not implemented | Real-student data (or a next simulation round) |
| retained(d=7) as a gate | Modify | not implemented | Random |
| Event-provenance source field | Modify | ✅ **implemented (2026-09-02)** | — |
| SymPy L0–L4 tiered grading | Replace | not implemented | When the math line restarts |
| Math v0′ blueprint (fixed 6 items + binary + Beta) | Replace | not implemented (blueprint) | Math MVP start |

## 2. Known technical debt deliberately not addressed (from the internal backlog)

- 18 multi-step-reasoning hard nodes in the AI diagnostic bank cannot produce items (frozen at 119/137; three true holes: solution conservation laws / process flows / integrated experiment questions);
- options A/B/C in the original (manual gold into `gold_bank` / relaxed threshold / direct extraction from real items);
- figure-dependent nodes (crystal cells, apparatus, curves) have no AI-generated items and rely on authentic exam resources.

## 3. The three items that shipped with this release (tested; see git log)

1. `engine/mastery.py`: FSRS-4.5 damped formula replaces hand-set constants (fixes the 4608→73-day explosion; regression tests included);
2. `engine/selector.py`: stopping criterion gap>0.45 → P(top1)≥0.80 with min_length 4;
3. `engine/event_log.py`: event-provenance field (source∈{real,qa,synthetic} + schema_version=2).

## 4. Improvements gated on real student data (no data = not verifiable)

- Grader calibration (LLM grading 35–65% QWK, insufficient evidence);
- Video efficacy table (needs real watch → retest);
- Prerequisite-graph parameters;
- Profile timeline (needs real multi-session data).
