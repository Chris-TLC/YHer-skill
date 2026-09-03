# Research Registry

> Every traceable number in the whitepaper (WHITEPAPER.md) opens its original evidence file here. All references are read-only.

## A. System facts

| Number | Value | Authoritative source |
|---|---|---|
| Structured question bank | 3,329 items | `data/item_bank/v4/chemistry_v4_3329.jsonl` (v4) + `v4_1` revision |
| R5 whitelist | 1,202 items (2,526 full pool) | `data/item_bank/v4/usability_r5_v1.jsonl` + loader `core/data/item_bank_v4.py` |
| Figure transcriptions | 10,102 rows | `data/item_bank/v4/ws2_asset_transcripts_v1.jsonl` |
| Media mappings | 13,171 rows | `data/item_bank/v4/ws2_media_ref_map_v1.jsonl` |
| Knowledge graph | 135 nodes | `data/knowledge_graph_150_enriched.jsonl` |
| Open nodes | 27 | `/api/demo/nodes` (snapshot, 2026-07-13) |
| Independent content families | 963 | `/health` (snapshot, 2026-07-13) |
| Deterministically gradable items | 400 | M6 catalog audit (snapshot, 2026-07-13) |

## B. Data pipeline and QA (BATCH6–16)

| Item | Value | Report |
|---|---|---|
| Full-pool VL audit cost | ¥17.93 / 2,526 items | `docs/audit-history/BATCH14_AUDIT_2026-07-06.md` |
| Final three pools | clean 799→1,295 (nominal)→1,203 (after exclusions) | BATCH10/14/16 reports |
| batch13 dead-figure recovery | 105 assets / ¥2.53 | BATCH16_AUDIT §batch13 |
| Answer-region placeholders | 249→45 (eliminated ~204 items) | BATCH16_AUDIT |
| Broken ions | 48 remaining (full pool), text_ion 4% | BATCH16_AUDIT |
| Reviewer precision gold | 236 rows, zero false positives (1,207 whitelist) | BATCH16_AUDIT §16b |

## C. AI question generation, five rounds (¥13.87)

| Round | Result | Report |
|---|---|---|
| P1 bare-generation baseline | first pass ≈45%; directly serviceable ≈40–45% | `NEIHUA_P1_AUDIT_2026-07-06.md` |
| P2 gated generation | final pass 75/100 | `NEIHUA_P2_AUDIT_2026-07-06.md` |
| MVP three routes | text 96.7% correct; Track B pipeline valid | `NEIHUA_MVP_AUDIT_2026-07-07.md` |
| R2 | figure-anchored 19/20; RDKit 12/14 | `NEIHUA_R2_AUDIT_2026-07-07.md` |
| R3 | style-transfer direction established | `NEIHUA_R3_AUDIT_2026-07-07.md` |
| R4+R5 final | distinguishability 65%/fair 60%; five rounds complete | `NEIHUA_R4R5_AUDIT_2026-07-08.md` |

## D. Three-round architecture audit (2026-08-05→13)

| Finding | Value | Report |
|---|---|---|
| Four-state classification ceiling | 54–69% (12 items) | `redteam1_measurement_selection.md` |
| P/U non-identifiability | KL=0.0247 | `lane1_measurement.md` |
| gap>0.45 early-stop false stops | 10.3% | `redteam1_measurement_selection.md` |
| Legacy FSRS S value | 4,608 days after ×10 reviews (FSRS-4.5: 73 days) | `redteam2_memory_recommendation.md` |
| n=2 lower bound | 22.4% | `VERIFICATION_ROUND4.md` |
| SymPy false equivalences | 7 counterexamples / 5 reproduced | `redteam3_verification_pipeline_product.md` |
| 21-component final verdict | 4 keep / 10 modify / 5 replace / 2 downgrade | `MASTER_AUDIT_REPORT_2026-08-13.md` |

## E. Cost ledger (aligned with the official data)

| Item | Value | Source |
|---|---|---|
| Chemistry full (incl. prerequisites) | ≈¥13,100 estimate (majority Opus development) | `各阶段Token成本估算_v1.md` (docs/history) |
| Math/physics per subject | ≈¥2,180/subject (engine reuse) | same |
| Math full-page transcription | ¥104.28 / 400 papers | `yihuier-math-skill/data/item_bank/split_progress.json` |
| Generation verification, five rounds | ¥13.87 | NEIHUA R4R5 |
| Demo full QA (2026-07-12 → 13) | ¥1.12 (205 events) | BATCH16 / overnight closeout |

*Note: the "chemistry full" line includes early exploratory development costs; it excludes the real-student validation phase (which has not happened).*
