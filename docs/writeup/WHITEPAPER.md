# YHer:An Evidence-Bound Diagnostic Learning System for High-School Chemistry

**Non-archival engineering report · 2026-09-02**
Status: pre-alpha. This document describes a system and a set of engineering audits; it is **not** a report of controlled student trials. No claim herein rests on real-student data.

---

## 1. Abstract

We present YHer, a single-node learning loop for Shanghai high-school chemistry built around one central idea:**the diagnosis students receive must be evidence-bound** — every inferred weakness must be traceable to a specific item response, every standard answer to a verified source, every resource recommendation to a signed evidence trail. The system freezes three non-overlapping item families (diagnostic, practice, held-out) from a white-listed bank of 1,202 real exam items; administers an adaptive sequence under a four-state Bayesian belief model with expected-information-gain item selection; verifies the outcome on unseen item families; and writes a replayable student profile. The data pipeline that feeds it — 3,329 structured items recovered from Word documents with formulas, figures, sub/superscripts and answer keys intact — went through a ten-batch usability audit (2,526 items, 1,202 serviceable after R5 whitelisting) whose methodology (dual-direction gold labels, node-aware text review, auditor-product-sameness) generalizes to any exam-paper-to-structured-data conversion. A five-round blind evaluation of AI question generation produced a main finding the product honors today:generated questions are distinguishable from real exam items at 65% overall (60% under fair conditions) and are therefore confined to low-stakes practice slots, never to diagnosis. A third-round architecture audit (six literature lanes, three red-team passes, one independent verification round) returned 4 keep / 10 modify / 5 replace / 2 downgrade verdicts across twenty-one engine components, with the strongest finding being that all hand-set constants in the engine lack calibration evidence. Limitations are stated in §8; the most important is that every conclusion is simulation- or audit-derived, never student-derived.

## 2. Motivation

### 2.1 The problem we attack

In the Chinese gaokao system, the most valuable educational asset is not content — it is judgment. A student's question is rarely "what is the correct answer to item 12?" but "why did I keep failing at this exact kind of step, and what should I do in the next ten minutes?" Good teachers provide that judgment. Most students do not have access to them. Information equity means: give every student a machine that produces the *judgment*, with the evidence displayed, so the student can decide whether to trust it.

### 2.2 Why evidence-binding, not model-trust

The core design constraint is not "make the AI smarter" but **"make every answer a student sees traceable to something that cannot be invented"**:

- standard solutions displayed to students are projected only from `answer_verification=passed` records that came from official answer keys;
- the item bank is derived from original exam Word documents, not model-generated;
- recommendations carry a signed evidence trail (`track_map_v1.yaml` + `curriculum_runtime_v1.json`), and unverified resources stay neutral;
- the browser never receives answers, rubrics, or item IDs before the response is locked in (fail-closed).

The generative-AI layer does limited selection and language organization; it cannot create new chemistry facts.

## 3. System Design

### 3.1 Session flow

1. Freeze three disjoint item families (diagnostic / practice / held-out) from R5 pool;
2. server-side scoring, no answer leakage before submission;
3. adaptive selection under **four-state belief** (Mastered / Prerequisite-missing / Unstable reasoning / Unmastered) with EIG selection, prerequisite descent when beliefs compete;
4. learning checkpoint with explanations anchored to verified standard solutions;
5. signed video-resource recommendation with propensity snapshots and seen-segment tracking;
6. held-out verification on two unseen families → session report, FSRS stability, 7-day review hint.

### 3.2 Engine components (21 audited)

| Layer | Module | Role |
|---|---|---|
| Inference | `engine/mastery.py` | four-state belief + evidence update + FSRS decay projection |
| Selection | `engine/selector.py` | EIG item choice, prerequisite competition, stopping rule, seen-exclusion |
| Planning | `engine/planner.py` | 30/60/120/180-min budget tables, honest exhaustion |
| Recommendation | `engine/recommender.py` | signed tracks, budget, seen-segments, propensity snapshots (vector retrieval + reranking) |
| Memory | `engine/memory.py` | high-value events, restricted recall (expression-only injection; never into the diagnostic judgment path) |

The recommender is downstream of diagnosis, not an independent claim: its value is a reranked vector search whose quality is bounded by the diagnostic state that feeds it.

## 4. Data Pipeline

### 4.1 From exam Word documents to structured items

- 79% of the bank (4,522 docx + 573 doc) has a native Word source on disk;
- extraction: python-docx for text flow; dwml for OMML→LaTeX; olefile+WMF rendering; direct `word/media` asset extraction; 【答案】【解析】 anchors as natural cut points;
- block schema v4: `text / latex / chem / figure / table`; answer blocks separated from stem;
- result: 3,329 items, 3,329 with global-unique IDs; 60-item gold set built with 50 formal + 10 reserve.

### 4.2 WS2 asset transcription (10,102 rows)

Two-track scheme: formula images (3,299) → LaTeX/mhchem with KaTeX compile + double-run + render-back validation; illustrations (2,706) → structured transcripts (VL, double-run consistency, three-pool routing ai_seed / display_only / manual_queue). Gold set: 68 images (34 calibration + 34 blind), gold labels privately held by auditor, producer blind. Blind run caught a P0 systematic artifact: VL rewriting `=` as `⇌`/`→` under chemistry prior (574 cases) — double-run consistency is blind to "both runs made the same error"; the private blind gold set is the only catch.

### 4.3 QA marathon: 2,526 items to 1,202 serviceable

Ten batches (BATCH6, BATCH8–16). Three core methodological results:

1. **Rendering without crashing ≠ item usable.** The first acceptance attempt verified only "render pipeline doesn't throw"; the user review failed it (missing figures, empty answers, broken ions). The usable-item ledger became an on-line gate.
2. **The auditor must see what students see.** The screenshot pipeline ran without KaTeX for an entire batch; the VL channel judged a different product than the browser. Fix = same-source verification before auditing.
3. **Precision gold must accompany recall gold.** A new five-dimension reviewer was calibrated only on known-bad items and later mis-flagged 169 good items in the whitelist. Node-aware text scanning (sentinels for formula/media/omml) fixed it: 0 false positives on 1,207 whitelist, and R5 excludes were re-verified.

**Numbers**: 2,526 total; 1,202 serviceable after R5 (47.6%); the audit cost of the full-pool VL pass was ¥17.93; the hardest batch (batch13 dead-figure recovery) rescued 105 assets for ¥2.53.

## 5. Experiment 1 — AI Question Generation: Five Rounds, ¥13.87

Goal: internalize exam-paper generation patterns ("question DNA") instead of reusing the bank. Protocol: anchor items from R5 pool; human blind judgments on distinguishability under adversarial pairwise comparison; cost audit per round.

| Route | Result | Verdict |
|---|---|---|
| From-scratch generation | 87% distinguishable by blind expert | **dead** (two rounds identical; marginal gain proven zero; used only for low-stakes practice slots) |
| Style-transfer (transform real items) | 65% distinguishability / 60% fair-conditions | **alive** — usable in high-fidelity practice slots |
| Programmatic rendering (matplotlib + RDKit) | passes | scalable for figure-based items |
| Figure-anchored reskins | ~26/29 human-pass | **alive** |

Three meta-lessons: (a) adversarial distinguishability is not the same as student-scenario serviceability; (b) generators inherit the corpus's own transcription scars; (c) Goodhart is a property of any acceptance rule — anchor every acceptance line to non-forgeable concrete objects (numeric values / ids / hashes), never to semantic interpretation.

## 6. Experiment 2 — Three-Round Architecture Audit

### 6.1 Method

Round 1 (Aug 5) component-wise adjudication; Round 2 (Aug 6) independent re-review (47 claims: 30 REVISE / 5 REPLACE, overturning some prior numbers); Round 3 (Aug 13) six literature lanes (arXiv/Crossref/ERIC) + three red-team attack passes + independent verification round. Verdicts per component: **4 keep / 10 modify / 5 replace / 2 downgrade; none of the modifications has yet been implemented** — the engine is an audited design, not an audited implementation.

### 6.2 Quantified findings (the hardest numbers)

| Finding | Value | Source |
|---|---|---|
| Four-state classification ceiling (12 items) | 54–69% (two independent simulations) | redteam1 |
| P/U statistical non-identifiability | KL = 0.0247 | round 3 |
| Stopping rule gap>0.45 false-stop rate | 10.3% | redteam1 simulation |
| FSRS hand-set multipliers, 10 reviews at 3-day rhythm | S = 4,608 days (FSRS-4.5: 73) | redteam2 + verification |
| n=2 held-out binomial one-sided 95% CI lower bound | 22.4% | verification round |
| SymPy `simplify` false-equivalence counterexamples | 7 found / 5 reproduced | verification round |

### 6.3 Verdicts that will land in the product

- inference: binary mastery + prerequisite prior + error-code routing (UI four labels retained);
- stopping: P(top1) ≥ 0.8–0.9 with min_length 4–5; gap becomes display-only;
- memory: FSRS-4.5 damped formula replaces hand-set constants;
- efficacy: Beta-Binomial shrinkage, tie-break only (5pp real difference at n=20 still flips 30% of replicates);
- held-out: early-stop 3→6 protocol, never n=2 binary;
- events: provenance field `source∈{real,qa,synthetic}` (Caliper/xAPI alignment).

## 7. Engineering Governance Lessons

- 2026-06-28 agent-loop incident (1,418 tool calls, leaked plaintext key) → governance discipline;
- 2026-07-03 P0 answer-leakage incident (crops included printed answers) → OCR answer-leak gate as an on-line check;
- 2026-07-13 governance flip: contract-based authorization replaced by reversibility discipline (backup + manifest, single-command rollback, backups never deleted);
- 2026-07-17 dual-gate for research-type deliverables (honesty gate + contribution gate) after a paper delivery that was perfectly honest and zero contribution.

## 8. Limitations

1. **No real students.** All conclusions are simulation-, audit-, or engineer-derived. No retention, learning-effect, or usability evidence from actual learners.
2. **Quantified counterexamples rest on synthetic parameters and hand-set constants.** Directional conclusions are robust; point estimates are not calibrated.
3. **Chemistry line is frozen.** The engine is a design audited but not modified per verdicts — the 10 modify / 5 replace items are each documented with a specific fix, but not yet applied (except 2026-09 drop-in changes: FSRS-4.5 damping, P(top1) stopping, provenance field, each with tests).
4. **Math line has data but no system.** 10,103 structured items exist (400/400 papers, ¥104.28); knowledge graph = 0 nodes; diagnostic loop is blueprint-only. Published as archived notes, not released.
5. **Item-bank copyright boundary.** Items come from public exam papers; batch redistribution has provenance notes; license is CC BY-NC for data layers to make the boundary explicit.
6. **Chinese literature unreachable.** CNKI/Wanfang were blocked during the review; the Chinese assessment tradition is represented at metadata level only.
7. **Supplier survivor bias.** Of six LLM providers used in the persona study, three failed collection (DeepSeek/GLM/MiniMax excluded, documented); analysis full set is three houses.

## 9. Reproducibility

- Every audit report in `docs/audit-history/` lists the exact commands, input hashes, and verdict tables;
- data schema and R5 definition: `data/README.md`;
- session reconstruction: single-node deterministic path (`YHER_ENABLE_PAID_LLM=0`), synthetic replay corpus under `SYNTHETIC_DEMO` label, no cross-contamination with real or QA events;
- question-bank dataset mirror with a deterministic builder script: `scripts/make_hf_dataset.py`.

## 10. Acknowledgements

This project was designed, judged, and acceptance-gated by a student with AI assistants contributing design proposals, implementation, and audits under explicit signed-review governance (dual-agent workflow; reviewer=claude/codex with dates and evidence). The system-prompt persona derives teaching style from public educational content; the recommender surfaces that content as an external resources layer with provenance, and this report treats content attribution and persona replication as separate, auditable layers.

---

*All numbers in this document are from reproducible audits (BATCH6–16, NEIHUA P1–R5, CEO audit, three-round architecture audit); see docs/audit-history/ and docs/writeup/RESEARCH_REGISTRY.md.*
