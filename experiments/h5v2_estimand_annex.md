# Persona v2 Estimand And Pilot-Approval Annex

Status: pre-main implementation annex

Date: 2026-07-15

Run ID: `llm-personas-v2-dual`

This annex removes implementation ambiguity left by the frozen
`h5v2_analysis_plan.md`. It does not change that plan, the persona population,
prompt bytes, task roster, mapping degradation, or any H1-H4 result. It was
written before the formal pilot and main observations.

## Analysis Identity

- The independent bootstrap cluster is `persona_id` (`n=50`).
- Deficit/control rows, prompt condition, item, and provider are repeated
  measurements inside a cluster.
- Main estimates use only records whose physical phase and
  `analysis_population` are both `main`. Pilot and development-partial roots are
  ineligible by assertion.
- The analyzer reconstructs the expected roster from the committed runtime task
  manifest. Missing tasks become disclosed missing/`NC` cells; filesystem glob
  discovery cannot define the denominator.
- Provider manifests, phase provenance, response identities, source hashes,
  prompt revision, runtime bytes, and record costs must reconcile before any
  metric is emitted.

## Controlled Condition

Each expected controlled task occupies exactly one mutually exclusive response
state:

1. `correct_answer`: complete, non-abstaining, correct response;
2. `incorrect_answer`: complete, non-abstaining, incorrect response;
3. `abstention`: complete response with `answer=null`;
4. `technical_or_schema_failure`: no complete parsed response.

The four-state composition uses every expected task as its denominator. In
addition, conditional answer accuracy uses only non-abstaining complete answers;
its denominator is always printed. No technical failure is silently relabeled
as a chemistry error.

Within each provider and persona cluster, outcomes are first averaged across the
four frozen calibration items in each response arm. The primary paired effect
orientations are:

- accuracy and correct-response yield: `control - deficit`;
- incorrect-response yield: `deficit - control`;
- abstention and technical/schema failure: `deficit - control`.

Provider-specific estimates are shown. The aggregate gives each eligible
provider equal weight, never each call equal weight. The six mapped
item/failure rows are sparse descriptive evidence only; target-option hit and
computed manipulation compliance remain outside the confirmatory family.

## Blind Condition

For each `(persona_id, response_arm, provider)`, the terminal response is the
primary task bound to the last item in that anchor's frozen blind panel. Its
category is `A/B/C/D/...`, `ABSTAIN`, or `NC` when the expected terminal record
is absent or not complete. `NC` stays in agreement denominators.

Pairwise provider output reports exact terminal agreement, numerator,
denominator, and Cohen kappa on the common frozen subjects including `NC`.
Multi-provider agreement is descriptive and uses the rectangular provider set
named in the corresponding table. A provider whose invalid-schema rate is
strictly above 50% of expected primary blind tasks is excluded from aggregate
blind estimates but remains in the lifecycle table; its controlled results are
considered separately.

Output stability is evaluated only on the frozen terminal-repeat subset. Answer
stability compares the primary and repeat categories. Normalized-output
stability compares canonical JSON encodings of the parsed output after the
runner's schema normalization. Retries are transport attempts, not stability
replicates.

Item-level/full-vector agreement, rationale length, noise-stratum interactions,
and disagreement topology are exploratory and must be labeled as such.

## Bootstrap And Intervals

- Seed: `2026071503`.
- Resamples: exactly 10,000.
- Each resample draws the 50 `persona_id` values with replacement and retains
  all repeated rows, providers, conditions, items, and stability repeats for a
  drawn cluster.
- Percentile 95% intervals use the 2.5th and 97.5th percentiles.
- Undefined resample statistics remain undefined; their count is disclosed and
  is never coerced to zero.
- Lifecycle counts and raw denominators are exact counts, not bootstrapped.

## Blind Adjudication Export

Adjudication is exploratory. A deterministic SHA-256 ordering with seed label
`2026071503` selects at most 120 complete terminal candidate records: up to 80
from cross-provider disagreement subjects and up to 40 from agreement subjects,
with unused capacity filled from the other stratum. Both judges receive the same
cases.

Each case contains only the public text-only question, public options, and the
candidate answer/rationale. It excludes target node, persona/failure labels,
correct option, target option, mapping status, private scoring fields, and
provider identity. Judges may return an error category, agreement label,
`unknown`/`insufficient_evidence`, and short rationale. Only pairwise agreement,
category counts, and disagreement examples are reported; no realism,
authenticity, or human-validation score is constructed.

## Formal Pilot Approval

Revision 0 is approved for main collection only when all of the following are
machine-audited for both frozen pilot providers:

1. exactly 128 expected tasks per provider are represented by a complete or
   disclosed terminal lifecycle state, with zero unexplained missing IDs;
2. returned model IDs contain no drift and every record binds the committed
   runtime/freeze/prompt provenance;
3. each provider-condition cell has at least 80% complete records and at most
   20% invalid-schema records;
4. blind leakage assertions, pilot/main physical isolation, and the text-only
   contract pass;
5. record, attempt, token, timeout-ambiguity, and CNY totals reconcile with the
   cumulative run ledger without opening the hard fuse; and
6. an immediate resume audit makes zero provider calls and preserves immutable
   response bytes.

A prompt revision is justified only by reproducible schema/instruction failure,
not by chemistry accuracy, provider disagreement, latency, rate limits,
truncation, or a favorable-result preference. If the engineering criteria fail
for a transport reason, the runner or lifecycle is repaired without changing
prompt content; any post-pilot runtime byte change receives a new committed
runtime manifest before main observation.

## Reported Boundaries

The paper reports simulated, independent response-channel behavior. It does not
report a learner sample size, human similarity, learning progression, education
efficacy, or authenticity. Provider and prompt-condition missingness is visible
in every relevant denominator.
