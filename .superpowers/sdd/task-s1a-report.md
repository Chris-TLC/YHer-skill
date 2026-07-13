# Task S1A Report: Confirmatory Runner Implementation And Verification

## Status

- S1A code, frozen configuration, CLI, and focused tests are implemented.
- No real confirmatory shard was run. The only execution smoke used an in-memory fake
  catalog and wrote below pytest's temporary `yher_sprint2` directory.
- No tag, push, publication, API call, official-data write, baseline fix, or generated
  `data/sim_store/confirmatory` output was created.
- The implementation commit is the commit containing this report; use `git log -1`
  for its SHA.

## Delivered Files

- `experiments/config/confirmatory_v1.json`: frozen machine-readable design.
- `experiments/confirmatory/config.py`: canonical config hashing and validation.
- `experiments/confirmatory/catalog.py`: real R5/S0 loading, two-family held-out
  reservation, prerequisite resolution, and pre-outcome common-support sets.
- `experiments/confirmatory/allocation.py`: seeded Arm A family epochs and exact fixed
  `(distance, family_id, item_id)` B/C allocation.
- `experiments/confirmatory/randomness.py`: exact SHA-256 first-128-bit seed derivation
  and paired streams.
- `experiments/confirmatory/response.py`: matched/misspecified generators and held-out
  posterior-predictive scoring.
- `experiments/confirmatory/simulation.py`: production `mastery.observe`,
  `selector.select_next`, and `selector.should_stop` loop.
- `experiments/confirmatory/metrics.py`: prefix/final repeat and severe-error metrics.
- `experiments/confirmatory/storage.py`: S0 path/envelope guards, deterministic
  JSONL, hash-valid resume, temporary-sibling validation, and atomic replacement.
- `experiments/confirmatory/runner.py`, `cli.py`, `__main__.py`: shard planning,
  protected-state execution, manifest binding, validate-only, resume, workers, and
  bounded smoke controls.
- `tests/test_confirmatory_runner.py`: focused S1A contract suite.

No production module is 500 lines or larger. The observation loop is kept separate
from response generation, metrics, catalog construction, and storage.

## TDD Evidence

First RED, before any confirmatory package code:

```text
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/test_confirmatory_runner.py::test_frozen_config_exposes_exact_confirmatory_grid
1 failed
Expected cause: ModuleNotFoundError: experiments.confirmatory
```

Minimal config GREEN:

```text
1 passed in 0.01s
```

Expanded contract RED, before allocation/simulation/storage/catalog/CLI modules:

```text
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_confirmatory_runner.py
12 failed, 1 passed
Expected causes: missing randomness, allocation, simulation, storage, catalog, and
CLI modules.
```

Incremental GREEN checkpoints:

```text
config/seed/allocation: 4 passed
production simulation contracts: 7 passed
atomic storage contract: 1 passed
first complete focused suite: 13 passed
final focused suite including bounded fake orchestration: 14 passed
```

The final tests dynamically spy on the real production functions and statically
confirm there is no local Bayesian updater or selector. Nested event envelopes are
validated independently of their parent journey.

## Contract Review

- Real catalog validation is exact: 27 open targets, 23 H1/H2 eligible, four
  prerequisite-free exclusions, four truths, three arms, two conditions, 50
  replicates, and 32,400 planned journeys.
- Held-out selection uses the frozen SHA-256 family order; the same two target-local
  families are removed from every administered local/prerequisite pool.
- Arm A calls production `selector.select_next` once per administered decision.
  B/C never call it. All arms call production `selector.should_stop` after every
  observation with `budget_items=26`.
- Every observation calls production `mastery.observe`. Local observations increment
  direct count once; prerequisite observations use `is_direct=False`.
- Production correct-probability and likelihood helpers are the only inference path.
  Misspecified slip/guess/ability values are stored separately and never passed into
  inference.
- B is local-only. C uses a prerequisite at every executed total position divisible
  by three. Fixed selection never uses seeded order to override the frozen tuple.
- Shared response/slip/guess/ability streams are arm-free; Arm A item-order streams
  are arm/purpose-derived. Held-out seeds exactly omit arm.
- Views at 9/15/25 preserve prefix repeat counts and actual administration count.
  Only confidence early stops carry forward. Item-25 non-confidence is labeled
  `budget_exhausted` and cannot be converged.
- Each view stores two family-level held-out atoms (`p_hat`, outcome, squared error),
  Brier, argmax, belief, convergence, severe errors, pre-outcome common-support flag,
  and common-support set hash.
- Shards are target x truth x condition and contain all replicate/arm pairs. Resume
  skips only envelope/schema/count/config/hash-valid shards. Worker count, shard
  ordering, and resume do not alter bytes.
- Raw events, journeys, shard manifests, and run manifests carry
  `simulated:true`, non-empty `persona_id`, `provider=programmatic`, and a model ID
  bound to production mastery/selector file hashes and the supplied runner commit.
  Execution provenance binds analysis-plan/runner commits, experiment tag, config,
  seeds, UTC artifact timestamp, and all catalog/census/engine input hashes.
- Execution is wrapped in the S0 protected-filesystem guard. All writes are restricted
  by the S0 path guard to `data/sim_store` or `/tmp/yher_sprint2` and use atomic
  replacement.

## Verification

Final evidence commands and outcomes:

```text
Focused S1A: 14 passed
py_compile: exit 0
Root engine contracts: 119 passed
Repository suite: 595 passed, 1 failed, 3 warnings
```

The sole repository failure is the required untouched baseline:

```text
tests/test_ws2_transcripts.py::test_v3_repository_unaffected
ItemRepository().count() == 6440, expected hard-coded 6438
```

Real validate-only output:

```text
open_nodes=27, h1_h2_eligible=23, h1_h2_excluded=4,
truth_states=4, arms=3, conditions=2, replicates=50,
expected_journeys=32400, common-support targets={9: 9, 15: 4, 25: 1}
```

The requested validate-only output path remained absent. No experimental outcome row
was created.

## Self-Review And Concerns

- Confirmed no Python `hash()`, parallel mastery/EIG/stopping implementation, key
  material, official-data writer, full-grid invocation, tag, or push path exists in
  S1A files.
- Confirmed only S1A paths are intended for exact-path staging; unrelated dirty-tree
  files remain untouched.
- The four mechanically excluded targets have no eligible prerequisite pool. Their
  Arm C record terminates as a counted `structural_failure` before total position 3,
  rather than violating the arm by substituting a local item. They remain present in
  the 32,400 planned record grid and must be disclosed/excluded by the predeclared
  analysis rules where appropriate.
- Capacity-explicit no-repeat sensitivity is narrow in the current trusted catalog:
  9 targets at budget 9, four at 15, and one at 25. These are pre-outcome results and
  must be co-reported without relaxation.
