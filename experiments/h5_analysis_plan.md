# Pre-Observation H5 Analysis Plan

Status: frozen before any live provider response

Date: 2026-07-13

This plan supplements only the LLM-persona H5 study. It does not amend H1-H4,
the programmatic simulation, or any observed result. No live S2 provider response
existed when this document was written.

## Scope And Claim Boundary

H5 asks whether the frozen simulated-persona protocol produces manipulation-valid,
cross-provider behavior under the specified prompts. It is secondary evidence about
simulated model behavior. It is not evidence about human students, learning,
educational efficacy, or programmatic diagnostic accuracy.

The absence of an explicit machine mapping from a KG misconception identifier to a
specific incorrect option is a pre-outcome measurement exclusion. In that case H5 is
`excluded_pre_outcome` and has no supported/partially-supported/not-supported
decision. It must not be converted into `not_supported`, because the misconception
contrast was not measurable.

## Frozen Inputs

The analysis reads only:

1. the immutable S2 preparation manifest and its listed persona, manipulation-panel,
   annotation-map, config, code, and official-input hashes;
2. an immutable collection manifest that names each frozen provider manifest by
   relative path and SHA-256; and
3. calibration and journey artifacts named and hashed by those provider manifests.

No filesystem glob, favorable-provider substitution, unlisted journey, response-text
reinterpretation, or semantic backfill of a missing misconception map is allowed.
Every consumed record must carry `simulated:true`, `persona_id`, `provider`, and the
provider-returned `model_id`. Artifact hashes are verified before any metric is
computed.

The frozen providers are DeepSeek, GLM, Kimi, MiniMax, Doubao, and Tongyi. The frozen
panel has 25 weak/strong pairs, 50 personas total, and Arms A and B. Prompt revision 1
may replace revision 0 only when an immutable revision-0 manifest records
`calibration_rewrite_required`; no other second attempt is admissible.

## Analysis Sets And Completion

- A raw provider-arm cell contains the 50 frozen personas, including structurally
  incomplete journeys and recorded technical failures.
- A provider-arm completion cell passes only with at least 45 immutable `complete`
  journeys under the exact frozen seed, panel, code, model, arm, and 15-item policy.
  Structural incompleteness remains in the denominator and is disclosed.
- A provider qualifies for H5 only if both arms pass completion, the formal 25-pair
  composition passes, the weak and strong accuracy-band gates pass, and the weak
  target-misconception gate is measurable and passes.
- Raw-complete but manipulation-ineligible providers may appear only in an explicitly
  exploratory agreement panel. They do not count toward the H5 decision.
- Provider, persona, arm, prompt-revision, structural, and technical exclusions are
  reported separately. No exclusion may depend on an observed favorable direction.

## Manipulation Metrics

The frozen accuracy bands are weak accuracy below 0.40 and strong accuracy above
0.75 on the frozen calibration items. The target-misconception contrast is the weak
persona target-option hit rate minus that item's random-wrong-option baseline.

When the target mapping is measurable, the combined contrast uses exactly 10,000
persona-cluster bootstrap resamples with seed `2026071303`. Each resample draws the
50 frozen persona IDs with replacement, preserves all included provider observations
for a drawn persona, computes a contrast within each qualifying provider, and then
averages providers equally. The reported interval is the two-sided percentile 95%
interval. A provider-specific manipulation matrix reports the accuracy bands, target
contrast, completion counts, revision, model IDs, and exclusion reason without a
post-hoc threshold.

## Cross-Provider Agreement

For each immutable journey, the terminal category is the argmax of the four-state
final belief in the production order `M`, `P`, `C`, `U`; a structurally incomplete or
missing terminal journey is `NC`. Agreement subjects are frozen
`(persona_id, arm)` pairs.

- Overall Fleiss kappa uses the rectangular set of subjects rated by every provider
  in the stated analysis set.
- The provider heatmap uses pairwise Cohen kappa on the pairwise-complete frozen
  subjects, with the subject count shown in every cell.
- Primary H5 agreement uses only qualifying providers. If H5 is excluded before
  outcome or fewer than two providers qualify, the same calculations may be shown for
  raw-complete providers only under the label `exploratory_unqualified`.
- Kappa is descriptive. No cutoff, significance gate, or human-behavior claim is
  introduced after observation.

## H5 Decision

The ordered branches in `experiments/analysis_plan.md` remain controlling when H5 is
measurable:

1. `supported` only when at least five providers qualify, both accuracy bands pass,
   and the combined target-misconception contrast 95% interval is strictly above zero;
2. `not_supported` only when fewer than four providers qualify or neither measurable
   manipulation gate passes; and
3. `partially_supported` otherwise.

If the target mapping is absent, the analysis status is `excluded_pre_outcome`, the
decision is null, and all collected behavior is descriptive. If a mapping exists but
no provider response is available, the status is `pending_input`, not a decision.

## Cost, Drift, And Provenance

The result package sums requests, responses, retries, input tokens, output tokens,
and yuan cost from manifest-bound artifacts. Requested and returned model IDs are
reported per provider; any unapproved model drift remains an exclusion. No key,
credential, raw environment value, or unredacted provider error body enters the
package.

The H5 package records the S2 preparation and collection-manifest hashes, provider
manifest hashes, config hash, panel hash, S2 code commit/hash, H5 analysis commit/hash,
bootstrap seed, prompt revision, and all output artifact hashes. Re-running against
the same immutable manifests must be byte-identical.

## Honest Stopping

All six configured providers are attempted when credentials and service access are
available. Missing access, rate limits, structural incompleteness, manipulation
failure, and model drift are reported rather than replaced. No extra personas,
providers, prompt revisions, annotation mapping, or calibration items may be added
after the first live response to improve H5.

## Amendment 2026-07-14: Pre-Observation Calibration Feasibility Gate

This amendment was written before a canonical S2 preparation and before every live
provider response. The draft `yher-llm-persona-v1` cohort had zero provider
observations. A read-only preflight found that its `同分异构体` pair had only three
family-distinct valid MCQ calibration items, while the frozen manipulation protocol
requires four. Leaving that pair in place would make both strength strata
structurally incomplete before any provider behavior was observed.

Persona derivation therefore advances to `yher-llm-persona-v2`. Before applying the
existing target-round-robin failure selection, it mechanically retains only open
catalog targets with at least four family-distinct calibration candidates. A
candidate must have a non-empty item ID and family ID, MCQ scoring, a non-empty option
mapping, and an answer key present in that mapping. This is the exact predicate used
later to construct the four calibration items. It does not inspect target-option
annotations, provider output, response quality, hypothesis direction, or any outcome.

Under the frozen production catalog this gate retains 26 of 27 open targets. It
removes only the draft pair anchored to
`同分异构体-手性碳原子判断#failure-00` and, through the unchanged deterministic
round-robin rule, adds `铁及其化合物-制备与转化#failure-00`. The resulting cohort remains
25 weak/strong pairs and 50 personas; every persona has exactly four distinct
calibration families. The absence of a mechanical target-option mapping is unchanged:
H5 remains `excluded_pre_outcome` with a null decision, while any collected behavior
may be reported only descriptively.

The implementation configuration and preparation manifest must name this amendment's
commit, content hash, and freeze time. They must also record the v2 persona derivation
version. No v1 panel, preparation artifact, provider response, or result may be
silently reused under v2.
