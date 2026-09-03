# Data Assets

This directory contains the **public data** of YHer Chemistry: the question bank, the knowledge graph, and the asset transcription tables. The full data ships with the repository as JSONL; a readable 55-item sample lives in `samples/`, and a Hugging Face dataset mirror is published at [`Chris-TLC/yher-chemistry-question-bank`](https://huggingface.co/datasets/Chris-TLC/yher-chemistry-question-bank).

## File manifest

| File | Size | Description |
|---|---|---|
| `item_bank/v4/chemistry_v4_3329.jsonl` | 20 MB / 3,329 items | WS3 structured items (block schema v4), the single service source |
| `item_bank/v4/chemistry_v4_1_3329.jsonl` | 20 MB / 3,329 items | v4.1 revision (answer-verification semantics corrected, Batch 7 applied) |
| `item_bank/v4/usability_r5_v1.jsonl` | 660 KB / 2,526 rows | **R5 usability whitelist**: per-item service permission (1,202 items serviceable) |
| `item_bank/v4/service_exclusions.jsonl` | 4 KB | Permanent exclusion list for items unrecoverable at the source |
| `item_bank/v4/ws2_asset_transcripts_v1.jsonl` | 8.8 MB / 10,102 rows | Figure-asset transcriptions (formulas → LaTeX, illustrations → structured descriptions) |
| `item_bank/v4/ws2_media_ref_map_v1.jsonl` | 3.0 MB / 13,171 rows | Media references in items → asset hash mapping |
| `item_bank/v4/ws2_omml_latex_cache_v1.jsonl` | 196 KB / 1,561 entries | OMML→LaTeX pre-conversion cache (`katex_ok` on 1,559 / 99.9%) |
| `knowledge_graph_150.jsonl` + `_enriched.jsonl` | 360 KB + 680 KB | 135-node chemistry knowledge graph (prerequisites, question types, exam points, video recommendations) |
| `raw_papers/shanghai_all.jsonl` | 5.9 MB | Raw item-slicing output (6,083 items, noisy and uncleaned; provenance only) |
| `samples/` | 55 items | Deterministic readable sample from the R5 whitelist (with schema notes) |

## Schema (v4, block-level)

One item object per line:

```
{
  "item_id":        "sha1-derived; the downstream unique key",
  "group_key":      "paper identifier (e.g. 2023 上海高考化学卷)",
  "section_num":    "section number within the paper",
  "q_num":          "question number within the paper",
  "local_question_id": "local count within the group",
  "source_path":    "relative path of the source Word/PDF document",
  "answer_source_path": "source of the answer key (may differ from the stem source)",
  "schema_version": "ws3_schema_v4_candidate_1",
  "service_eligible": true/false,
  "answer_available": true/false,
  "analysis_blocks": [{"para":[{"type":"text","text":"solution..."}]}],
  "answer_blocks_effective": [{"para":[{"type":"text","text":"A"}]}],
  "quality_flags": [],
  "rubric": [{"point_id","desc","keywords","must_have","score","kg_node"}],
  "alignment": {...},        # alignment info inherited from v3
  "answer_verification": {...},  # answer-credibility verdict (e.g. 0.89; from the v3.4 pipeline)
  "kg_nodes": [...],         # linked knowledge-graph nodes
  "knowledge_points": [...]
}
```

**Block types** (`para[].type`): `text` / `latex` / `omml` (WMF images; the LaTeX lives in `ws2_omml_latex_cache`) / `image` (asset hash) / `table`.

## Service pool (R5) rules

```
load_service_pool() = item_bank_v4.loader with apply_r5=True by default
  → serves only rows where r5_serve=true in usability_r5_v1.jsonl
  → no ledger entry = not served
audit / preview / regression channels set apply_r5=False explicitly to see the full 2,526 pool
```

Currently **r5_serve=true = 1,202 items**.

## Copyright and licensing

- Items come from publicly released Shanghai gaokao and mock examinations; this repository contains a **mechanical structuring** of exam content (text extraction, layout repair, answer alignment), not original creation.
- Code and data-processing scripts: MIT. Question-bank data ships with the repository; the knowledge graph, transcription tables, and alignment information are project-built assets.
- Online mirror: Hugging Face `Chris-TLC/yher-chemistry-question-bank` (dataset card included).

## Building

```bash
# Readable sample (55 items)
python3 scripts/make_hf_dataset.py --sample-only

# Hugging Face dataset build (requires an HF token)
python3 scripts/make_hf_dataset.py --push <dataset-id>
```
