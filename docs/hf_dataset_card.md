---
license: cc-by-nc-4.0
task_categories:
- question-answering
- text-generation
language:
- zh
pretty_name: YHer Chemistry Question Bank
size_categories:
- 1K<n<10K
tags:
- chemistry
- education
- gaokao
- adaptive-testing
- knowledge-tracing
- question-bank
---

# YHer Chemistry Question Bank

The data layer of an evidence-bound diagnostic learning system for Shanghai high-school chemistry ([Chris-TLC/YHer-skill](https://github.com/Chris-TLC/YHer-skill)).

Every record in this dataset is derived from publicly released Shanghai gaokao and mock examination papers through deterministic mechanical structuring: text extraction, layout repair, and answer alignment. No content is model-generated.

## What's inside

The dataset ships in two configs:

| Config | Records | Content |
|---|---|---|
| `items_v4` | 3,329 | Structured exam items (block-level schema v4), each with stem, analysis, answer blocks, scoring rubric, knowledge-graph links, and answer-verification status |
| `knowledge_graph` | 135 | Chemistry knowledge-graph nodes with prerequisites, question types, exam points, and video-resource links |

Companion files in the source repository: an R5 serviceability whitelist (2,526 ledger rows, 1,202 serviceable), asset transcription tables (10,102 rows), and a media-reference map (13,171 rows).

## Record structure (items_v4)

Each item record contains:

- `item_id` — SHA-1-derived unique key
- `group_key`, `section_num`, `q_num` — paper and position identifiers
- `source_path`, `answer_source_path` — provenance of stem and answer key
- `analysis_blocks`, `answer_blocks_effective` — block-level text with types: `text`, `latex`, `omml`, `image`, `table`
- `rubric` — scoring rules with keywords and must-have flags
- `answer_verification` — credibility verdict from the source pipeline
- `kg_nodes` — links into the 135-node knowledge graph
- `r5_serve`, `r5_reason` — service whitelist status (1,202 of 3,329 items serviceable)

## Intended uses

- **Adaptive testing and diagnostic modeling** on real, structured exam items
- **Question-generation evaluation** — this bank served as the reference set for a five-round blind distinguishability study
- **Knowledge tracing** — items link into a prerequisite-structured knowledge graph
- **OCR and extraction benchmarking** on exam documents with complex layout and notation

## What this dataset is not

- **Not original creation.** Items are mechanically restructured from exam content, not manually re-authored. 926 of 3,329 items (27.8%) lack a separately sourced answer key path.
- **No student data.** No student response data is included. No claim in the companion system rests on learner data.
- **Non-commercial license.** The dataset is licensed under CC BY-NC 4.0. Code in the source repository is MIT-licensed.

## Building from source

```bash
python3 scripts/make_hf_dataset.py --sample-only          # local 55-item sample
python3 scripts/make_hf_dataset.py --push <dataset-id>    # build and push (needs HF token)
```

## Citation

```bibtex
@misc{yher-chemistry-question-bank,
  title = {YHer Chemistry Question Bank: Shanghai High-School Chemistry Exam Items with Knowledge-Graph Links},
  author = {Tu, Licheng},
  year = {2026},
  howpublished = {Hugging Face dataset},
  url = {https://huggingface.co/datasets/Chris-TLC/yher-chemistry-question-bank},
  license = {CC BY-NC 4.0}
}
```
