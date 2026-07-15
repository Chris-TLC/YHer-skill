# Pre-Observation Persona v2 Dual-Condition Analysis Plan

Status: frozen before every pilot or main provider observation

Run ID: `llm-personas-v2-dual`

## Claim Boundary

This study is an independent response-channel stress test using simulated
personas. It does not represent human participants, a real learner population,
learning trajectories, or educational efficacy. The programmatic H1-H4 study
remains the primary evidence and is not amended here. Every observation uses
`modality_condition=text_only`.

## Frozen Design And Unit Of Analysis

The independent cluster is persona_id (n=50). Each cluster contains paired
deficit and control response rows. Provider and response arm are repeated measurements.
Provider calls, item responses, retries, and model outputs are not
independent sample units.

The frozen grid contains 25 failure anchors crossed with low/high noise, giving
50 persona clusters and 100 paired rows per prompt condition. Controlled uses
exactly four calibration items. Blind reuses those four items and adds no more
than 21 family-distinct diagnostic items from the same frozen target panel.

## Pre-Observation Mapping Degradation

Independent Codex drafting and DeepSeek cross-checking produced exact consensus
for only 6 of 100 calibration rows. The remaining 94 rows were excluded as
semantically ambiguous before observation. Because 6% is below the frozen 60%
minimum, target-misconception hit rate is removed from confirmatory analysis.
The six mapped rows may appear only as a sparse descriptive manipulation check,
with their denominator and target-set hash shown. They cannot restore or change
a confirmatory decision.

## Confirmatory Outcomes

Controlled outcomes are paired deficit-control differences in correctness and
error rate, plus valid-response and abstention rates. These outcomes use all
frozen calibration rows and do not infer a specific misconception from an
unmapped option. Runner-computed `manipulation_compliance` is null when the
frozen item/failure/option map is unavailable and is sparse descriptive only.

Blind outcomes are terminal response consistency, provider-pair exact-answer
agreement, technical/schema failure rate, and output stability. The terminal
category is the frozen final-item answer in `A/B/C/D/ABSTAIN/NC`. A deterministic
10-persona subset repeats that exact terminal prompt once; answer agreement and
normalized-output equality on that repeat define output stability. Full-vector
and item-level agreement are exploratory.

No favorable-direction threshold is added after observation. Results are
estimates with two-sided intervals, denominators, missingness, and provider
lifecycle status.

## Pilot And Main Separation

The smoke pilot uses DeepSeek and Doubao mini on five frozen persona clusters,
both response arms and both prompt conditions. The rule is explicit: pilot data are physically isolated and excluded
from every main estimate, figure, bootstrap, and judge
export. Pilot approval checks transport, schema validity, leakage assertions,
model identity, resume behavior, and cost accounting only.

Main collection attempts the six frozen providers and all 50 clusters. A
provider with blind schema invalidity strictly above 50% has blind marked
excluded while controlled remains independently reportable. Every invalid,
interrupted, unavailable, model-drifted, or excluded lifecycle is retained in
the provider table; no provider is silently replaced.

## Prompt Revision And Exclusions

Prompt revision 0 is the default. At most exactly one prompt rewrite is
permitted, and revision 1 is admissible only if a committed pre-observation
pilot manifest records the engineering failure and the revised prompt bytes.
No rewrite may depend on a favorable main outcome. Missing public questions,
unknown answer keys, duplicate item families, model drift, and invalid schemas
are structural exclusions recorded before outcome aggregation.

Abstention is an observed response, not a technical failure. Judge labels include
`unknown` and `insufficient_evidence`; neither is coerced to agreement or error.

## Estimation

All main intervals use a persona-cluster bootstrap with 10,000 resamples, seed
`2026071503`, preserving every provider, response arm, condition, and item for a
drawn persona. Providers are equally weighted in aggregate summaries. Paired
provider agreement reports both numerator/denominator and pairwise Cohen kappa;
multi-provider agreement is descriptive and uses only rectangular common
support. NC remains a category rather than being dropped.

The sparse six-row mapping illustration is outside the confirmatory family.
Exploratory analyses and figures are marked exploratory. No call-count-based
sample size, human-behavior inference, or authenticity score is permitted.

## Blind LLM Adjudication

Two judge passes (Claude and GPT when available) receive only the frozen blind
public prompt and candidate output. Target node, deficit label, failure cause,
failure symptom, target option, correct option, and mapping status are absent by
assertion. Allowed outputs are agreement label, error category, short rationale,
and `simulated:true`. Pairwise judge agreement and disagreement examples are
reported; no consensus realism or authenticity score is constructed. A missing
judge is disclosed and is not silently substituted.

## Provenance, Cost, And Stopping

The committed freeze binds the plan, source manifest including explicit missing
files, official-input roster, candidate and blind panels, persona grid, mapping
decisions, consensus mapping, prompt bytes, provider/model configuration, and
audit sample. The freeze commit and byte hashes must precede each phase's first
observation. Source drift or protected-path drift stops collection.

Costs record requests, retries, returned token usage, timeout ambiguity, and CNY
by provider and phase. CNY 300 triggers a warning; CNY 450 is a hard fuse that
stops new calls and enters `needs_user`. Completed immutable responses are never
deleted or reissued merely to improve a result.
