# S2 LLM Simulated-Persona Runner

This package implements the secondary H5 study only. It never calls the 8700
application API and never creates an application session. Questions are read
from the production `ItemCatalog`; answers pass through production
`core.learning.scoring`, `engine.mastery`, and `engine.selector` in memory.
Artifacts are written only under an explicit simulation store.

## Pre-Observation Gate

Freeze and audit the 50 personas and manipulation panel without network I/O:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m experiments.llm_sim \
  --prepare-only \
  --output-root data/sim_store/llm_personas/llm-personas-v1
```

Before any provider observation, the `yher-llm-persona-v2` mechanical
amendment restricts persona targets to open catalog nodes with at least four
family-distinct valid MCQ calibration items. It uses the same item predicate as
panel construction and never consults annotation mappings or outcomes. The
ingestion protocol and `llm-personas-v1` run id remain v1; only the persona
derivation sub-version changes, with zero provider observations under v1.
The dated amendment is frozen at commit
`289be3bc4634336a8598ad80c0de084afdeba51d`; preparation and provider manifests
bind that commit, its H5-plan blob hash, and its UTC commit time.

The frozen panel accepts only an explicit machine mapping from a KG
`common_failure` identifier to an incorrect option. It never infers a mapping
from option text, symptoms, model rationales, or observed answers. With the
current R5 catalog, no such mapping exists, so all target-option manipulation
cells are honestly marked `excluded_pre_outcome`. Behavioral journeys may still
run, but target-misconception hit-rate claims are not reportable from those cells.

An independent, pre-reviewed machine annotation file can be supplied without
modifying official data:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m experiments.llm_sim \
  --prepare-only \
  --annotation-map /path/to/explicit_annotations.json \
  --output-root data/sim_store/llm_personas/llm-personas-v1
```

The accepted shape is
`{"items":{"ITEM_ID":{"KG_FAILURE_ID":"B"}}}` (or the documented row-list
equivalent). Preparation copies the normalized map and its source path into
`annotation_map_snapshot.json`; its content hash is part of the immutable panel
hash. A different map cannot replace an existing frozen panel.

## Live Provider Run

Live execution requires an explicit flag. Credentials are resolved in memory
from already-set process variables and then the repository `.env` for missing
values; they are never loaded during preparation and never serialized. Provider
model variables such as `DOUBAO_MODEL` use the same precedence:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m experiments.llm_sim \
  --live \
  --provider deepseek \
  --model deepseek=MODEL_ID \
  --output-root data/sim_store/llm_personas/llm-personas-v1
```

Supported provider names are `deepseek`, `glm`, `kimi`, `minimax`, `doubao`,
and `tongyi`. Each provider has independent checkpoints, retry/backoff state,
circuit breaking, model-id drift checks, and token/cost accounting. Keys are
read only inside the live transport and are not written to prompts, events,
manifests, logs, or CLI summaries.

Every journey and event carries `simulated:true`, `persona_id`, `provider`, and
the model id returned by the provider. A provider-arm cell is reportable only
after at least 45 of the frozen 50 personas complete; exclusions remain in the
provider manifest.

Preparation also freezes the full git HEAD, config hash, seed, and a hash of the
S2/production-engine code set. Live execution fails before resolving a key or
transport if HEAD, code, config, panel, or seed changed. If revision 0 misses a
frozen calibration band, its journeys remain raw observations but count as zero
eligible completions. At most one isolated retry is allowed with
`--prompt-revision 1`; a second miss is `excluded_post_calibration`.
