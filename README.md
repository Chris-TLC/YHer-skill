# YHer Chemistry — An Evidence-Bound Diagnostic Learning System

**Information equity, made verifiable.** Every diagnosis a student receives is traceable to the exact answer that produced it. Every standard solution comes from a verified source. Every video recommendation carries a signed evidence trail.

YHer is a single-node learning loop for Shanghai high-school chemistry. The core asset is not any one content source but the **diagnostic engine**: a four-state Bayesian belief model with expected-information-gain item selection, held-out verification on unseen question families, and a replayable student profile.

> **Status: pre-alpha, localhost demo.** This is not a deployed product. There is no real-student validation, retention, or learning-effect evidence — every conclusion in this repository is simulation-, audit-, or engineer-derived. Do not deploy for minors without adult consent processes.

This repository is [MIT-licensed](LICENSE) (code); the question bank, knowledge graph, and transcription tables are released alongside it, with a [Hugging Face dataset mirror](https://huggingface.co/datasets/Chris-TLC/yher-chemistry-question-bank) (see its dataset card for field semantics and license).

---

## Why this exists

In gaokao regions, the scarcest educational resource is not content — it is **judgment**. Good teachers, good methods, and good questions concentrate in a few well-resourced schools. YHer's answer to that inequality is threefold:

1. **Turn judgment into verifiable engineering.** Whether a student is stuck on a prerequisite or on method is *computed from evidence*, not guessed by a model.
2. **Turn "who teaches well" into a retrieval problem.** Quality-checked public video content is ranked by the student's current state, not by luck.
3. **Treat answer credibility as a discipline.** Every standard solution served to a student comes from verified official answer keys. AI organizes language; it does not invent chemistry facts.

The design constraint in one sentence: **every conclusion a student sees must be traceable to something that cannot be invented.**

## One canonical session

1. Freeze three disjoint item families (diagnostic / practice / held-out) from the R5 whitelist.
2. **Server-side scoring** — the browser never receives answers, rubrics, or item IDs before a response is locked in (fail-closed).
3. **Adaptive selection** under a four-state belief model (Mastered / Prerequisite-missing / Unstable-reasoning / Unmastered) with expected information gain (EIG); prerequisite descent when beliefs compete.
4. **Learning checkpoint** with explanations anchored to verified standard solutions.
5. **Signed video recommendations** with propensity snapshots and seen-segment tracking.
6. **Held-out verification** on two unseen families, producing a session report, FSRS stability estimate, and a 7-day review hint.

## Quick start

### Docker (recommended)

```bash
docker build -t yher-demo .
docker run -p 8700:8700 yher-demo            # default: paid LLM explanations if credentials exist
docker run -p 8700:8700 -e YHER_ENABLE_PAID_LLM=0 yher-demo   # zero-cost deterministic mode
```

Open [http://127.0.0.1:8700](http://127.0.0.1:8700).

### Local (Python 3.11+)

```bash
YHER_ENABLE_PAID_LLM=0 ./deploy/run_demo.sh   # bootstraps .venv-demo, serves 127.0.0.1:8700
```

Health check:

```bash
curl -fsS http://127.0.0.1:8700/health | python3 -m json.tool
```

Optional credentials (`DEEPSEEK_API_KEY` or another provider in `.env`) enable paid LLM explanations. Without credentials, with the paid channel explicitly disabled, or on timeout/malformed output, the system **keeps the deterministic path and degrades honestly**. Never write API keys into code, logs, screenshots, or reports.

## Architecture

```text
apps/web/index.html + app.js
            |
            | same-origin /api/demo/*
            v
apps/demo_api.py (FastAPI, localhost, one worker)
            |
            v
core/learning/session_service.py
   |          |          |          |
   |          |          |          +-- curriculum.py -> signed video map
   |          |          +------------- explanations.py / grading.py
   |          +------------------------ ItemCatalog -> v4 bank + R5 whitelist
   +----------------------------------- mastery / selector / planner
            |
            v
adapters/store/local_json.py
  append-only events + session snapshots + projected profile
```

The five engine modules, each audited against the literature:

| Engine | Responsibility |
|---|---|
| `engine/mastery.py` | Four-state belief, evidence updates, FSRS-4.5 decay projection |
| `engine/selector.py` | EIG item selection, prerequisite competition, stopping rule, seen-item exclusion |
| `engine/planner.py` | 30/60/120/180-minute budget tables with honest exhaustion |
| `engine/recommender.py` | Signed tracks, budget, seen segments, propensity snapshots |
| `engine/memory.py` | High-value event recall, restricted to expression-layer prompts |

**Diagnosis is the core; the recommender is its downstream.** The video layer is quality-check + vector retrieval + state-adapted reranking. Its value is bounded by the diagnostic state feeding it.

## Data and dataset

- Full data ships with the repo: 3,329 structured items, 1,202 R5 service-whitelisted, 10,102 figure/transcription rows, 13,171 media-reference rows, and a 135-node knowledge graph. Field semantics: [`data/README.md`](data/README.md).
- Readable 55-item sample: `data/samples/` (deterministic selection, reproducible).
- **Hugging Face mirror**: [`Chris-TLC/yher-chemistry-question-bank`](https://huggingface.co/datasets/Chris-TLC/yher-chemistry-question-bank) (configs: `items_v4` / `knowledge_graph`; dataset card included; builder: `scripts/make_hf_dataset.py`).

## Testing and verification

```bash
python3 -m venv .venv-pub
.venv-pub/bin/pip install -r requirements-dev.txt   # full test deps (faiss optional)
.venv-pub/bin/python -m pytest -q
```

- The offline suite and engine-contract tests define the baseline (see CI/local runs).
- QA evidence and synthetic scenarios (labeled `SYNTHETIC_DEMO`) are **engineering verification, not student evidence**.

## Honest boundaries

- This is a localhost pre-alpha: no login, tenant isolation, parental consent, deletion policy, or production operations.
- Browser journeys, API QA, and synthetic scenarios are engineering validation, not empirical research.
- Belief states are model states under current evidence — not scores, long-term mastery, or causal learning gains.
- R5 is a service whitelist, not a full manual gold standard. Critical diagnostic positions use only real exam items and verified standard solutions.
- AI-generated questions are a historical supply experiment; the canonical first diagnosis and held-out verification never use them.
- Video resources are hosted by their original platforms; link availability, copyright, and content changes are outside this repo's control.
- Public deployment, credential rotation, release processes, and minor-data flows are not yet implemented.

## Further reading

- **The engineering evidence report** — `docs/paper/` — a 16-page write-up of what was built, what was measured, and what remains unvalidated.
- [Whitepaper (release edition)](docs/writeup/WHITEPAPER.md)
- [Audit history](docs/audit-history/README.md) — three rounds of system-level audits, including the architecture verdicts
- [Two-minute demo walkthrough](docs/demo_walkthrough_script.md)
- [Data documentation](data/README.md)

Code is [MIT licensed](LICENSE). Exam questions, papers, subtitles, and external videos retain their original rights; the MIT license does not automatically cover content assets.
