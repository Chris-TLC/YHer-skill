# Pre-Computation P2 Supply-Bound Illustration Plan

Status: frozen-design candidate; no P2 result has been computed

Date: 2026-07-15

## Claim Boundary

P2 is a supply-bound algorithmic illustration. It measures trusted-video time
selected under an explicit role-compatibility model. It does not measure
learning benefit, remediation dose, prescription efficacy, a student-population
effect, or minutes of actual harm or benefit.

The wider validated library has 68 exact trusted chunks across 13 nodes. The
only exact overlap with the H1-H4 target set is `基本操作` and `烷烃`: eight
eligible rows collapsing to three physical `(bv,p)` choices. No fuzzy,
hierarchical, or semantic label expansion is permitted.

## Bound Inputs

| Input | SHA-256 |
|---|---|
| Trusted candidate JSONL | `9f14b8103eb191c7ffc5d2b1f1777e88b915082b94a3ac21f3c07f41a53f0406` |
| Signed runtime metadata | `6348b28805c75eddba73b39ef14c034f4c9aa0fd517a78b396440f865489dedd` |
| H1-H4 raw manifest | `2c68cada6c2229e6860d46fca4e4f65b3df674bfc4652b4a947934ba05e76dd3` |
| Reproducible canonical eight-row subset | `e080008a40e514bf57e95a5c9905c9ba469c1e9c976b0e1ec465e684d55ff34d` |
| Earlier audit-declared subset digest (retained, not a gate) | `b8ae2eaef4e047f75dbc2aa2a791188528115219660679b0fca70f530da7e2e2` |

The reproducible subset digest is SHA-256 over the UTF-8 bytes produced by:

```python
json.dumps(
    sorted(eight_full_source_rows, key=lambda row: row["chunk_id"]),
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
```

The earlier audit described its digest as full rows sorted by keys and
`chunk_id`, but did not retain the serialization algorithm. Exhaustive checks of
the ordinary JSON array and JSONL variants (ASCII/non-ASCII, compact/default,
with/without terminal newline) did not reproduce it. It is preserved as
`audit_declared_unreproduced` metadata and cannot make the run pass or fail.
Identity instead fails closed on the three full-source hashes, the exact eight
IDs and fields below, exactly three physical sources, and the reproducible
canonical digest.

The H1-H4 row filter is `condition=matched`, target in the two-node set,
`truth in {M,P,C,U}`, `arm in {A,B,C}`, replicate 0--49, and
`views.nominal_budget=15`. Posterior order is `[M,P,C,U]`. A posterior is
admissible only when that b15 view has `valid=true`; a stored belief in an
invalid view is never read. Oracle beliefs are deterministic one-hot truth
vectors.

## Candidate Universe

The frozen candidates and charged durations are:

| Target | Chunk ID | Seconds | Role | Physical source |
|---|---|---:|---|---|
| 基本操作 | `BV1aComYMEms#P102#c000` | 496 | drill | `BV1aComYMEms#P102` |
| 烷烃 | `BV18t4y1a7eD#P001#c000` | 110 | review | `BV18t4y1a7eD#P001` |
| 烷烃 | `BV18t4y1a7eD#P001#c003` | 99 | review | `BV18t4y1a7eD#P001` |
| 烷烃 | `BV18t4y1a7eD#P001#c004` | 94 | review | `BV18t4y1a7eD#P001` |
| 烷烃 | `BV1JT421C7WS#P001#c004_b` | 28 | drill | `BV1JT421C7WS#P001` |
| 烷烃 | `BV1JT421C7WS#P001#c011` | 134 | drill | `BV1JT421C7WS#P001` |
| 烷烃 | `BV1JT421C7WS#P001#c014_b` | 68 | drill | `BV1JT421C7WS#P001` |
| 烷烃 | `BV1JT421C7WS#P001#c016` | 121 | drill | `BV1JT421C7WS#P001` |

Every prescription preserves production physical deduplication: at most one
chunk per `(bv,p)`.

## Decision-Instance Distribution

Target-specific H1-H4 replicates are not joint students. P2 therefore integrates
over a fixed product-form factorial distribution rather than pairing equal
replicate indices and calling them learners.

For each of the 16 truth pairs
`(truth_basic, truth_alkane) in {M,P,C,U} x {M,P,C,U}`, cross the 50 retained
`基本操作` replicate margins with the 50 retained `烷烃` margins within an arm.
Each truth cell has design weight `1/16`; each component cross-product within a
cell has weight `1/2500`. The resulting terms are analytic integration points,
not independent observations or a real state-prevalence model.

Oracle, A, B, and C are always separate. There is no pooled B/C posterior or
estimate.

## Arm C Structural Failure

Every matched b15 Arm C `基本操作` view is structurally invalid. The primary ITT
policy is node-level fail-closed:

1. mark `基本操作` as `structural_failure`;
2. do not read or impute its stored belief;
3. mask every `基本操作` candidate from the Arm C selector;
4. retain the valid Arm C `烷烃` posterior; and
5. report `failed_node_fraction=0.5` for every Arm C joint instance.

Missed available oracle coverage on the masked node is attributed to
`diagnostic_structural_failure`. A separate appendix may show `烷烃`-only
available-case results. It cannot be averaged with A/B's two-node view.

## State-Role Compatibility

| State | Required slot | Available exact supply |
|---|---|---|
| M | same-target review | 烷烃 only |
| P | prerequisite, low-difficulty concept | none |
| C | same-target drill | both targets |
| U | same-target, low-difficulty concept | none |

This defines role compatibility only. It does not identify efficacy or required
dose. Report `unsupported_posterior_mass` and `unobtainable_truth_slots` as
probability mass/counts, never as minutes.

## Utility And Selector

For posterior `b` and selection `S`:

```text
U_b(S) = sum_target sum_state b[target,state]
         * I(S intersects eligible[target,state])
```

Coverage saturates after the first selected chunk fills a `(target,state)` role
slot. The oracle replaces `b` with one-hot truth. A failed Arm C node contributes
no utility. The selector uses one analytic budget of 600 charged seconds across
both targets, stops when no feasible positive marginal utility remains, and has
no segment-count cap.

At each step choose by this stable ordering:

1. larger marginal utility / charged second;
2. larger marginal utility;
3. smaller charged duration;
4. lexicographically smaller `chunk_id`.

Score comparison uses decimal arithmetic with at least 50 digits of precision.
Every feasibility and marginal-gain decision is retained in a selector trace.
No approximation guarantee, optimality claim, or exact-solver comparison is
made.

## Minute Fields

All metrics are computed in integer charged seconds and divided by 60 only for
display.

- `mismatched_selected_seconds`: selected duration whose target/role does not
  fill the programmatic truth slot.
- `missed_available_seconds`: duration of the deterministic oracle-selected
  representative for an available truth slot that the arm fails to fill. Slot
  equivalence, not exact chunk identity, determines a match.
- `unobtainable_supply_minutes`: always `null`, with reason
  `no_frozen_role_compatible_dose`. No P/U dose is identified.
- `unused_budget_seconds`: `600 - selected_seconds`; descriptive slack only.

Companion fields include selected seconds/count, structural-failure node
fraction, unobtainable truth slots, unsupported posterior mass, metric
denominators, and null-reason counts. No scalar composite combines them.

## Bootstrap

Use exactly 10,000 resamples with seed `2026071505`. Independently resample the
50 replicate IDs within each fixed target stratum, retaining all four truth
rows, three arms, validity state, and oracle truth for a sampled
`(target,replicate)` block. Recompute product-distribution and truth-cell means
inside each resample. Arms and truths stay paired.

Percentile 95% intervals describe simulator Monte Carlo variability under this
fixed design, not a human population. The two strata each contain 50
programmatic replicate clusters. Cross-product terms must never be reported as
sample size.

## Required Outputs And Gates

The implementation emits a hash-bound input manifest, canonical candidate
subset, decision-instance manifest, selector traces, profile metrics, summary,
bootstrap results, and figure data. Every result states `illustrative=true`,
`simulated=true`, and an explicit no-external-validity boundary.

Before computation, fixture tests must prove input hashes, physical
deduplication, binary saturation, the stable tie-break, 600-second feasibility,
Arm C node masking, invalid-belief non-access, and null unavailable minutes.
The implementation fails closed on drift, duplicates, missing b15 views,
non-finite/non-normalized beliefs, an unexpected validity pattern, physical
source reuse, or budget overflow.

If these gates fail, P2 degrades to the audited supply table only.

## Prohibited Claims

P2 cannot support real joint profiles or prevalence, a learner sample size,
causal learning effects, a numeric P/U demand, an effective/wasted/saved-second
interpretation, a valid Arm C `基本操作` posterior, a four-arm efficacy ranking,
generalization beyond two targets and three physical sources, or any greedy
guarantee. Paper wording must use `mechanically mismatched selected minutes`
and `missed available-supply minutes` under the frozen model.
