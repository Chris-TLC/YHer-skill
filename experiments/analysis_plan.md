# Pre-Data Analysis Plan: Budget-Limited Prerequisite-State Identifiability

Status: frozen before the prerequisite census and before confirmatory simulation output.
The commit containing this file is the analysis-plan freeze. Any later change to a
confirmatory rule requires a dated amendment that preserves this version and reports
results under both specifications; observed results will never be used to tune a rule.

## Question And Scope

This confirmatory study tests whether a four-state diagnosis can practically
distinguish a prerequisite gap (`P`) from mastered (`M`), chain-instability (`C`),
and unlearned (`U`) states under a binary response channel at budgets 9, 15, and 25,
and whether the timing policy for prerequisite probes changes that finite-budget
identifiability. It uses the trusted R5 catalog,
the production mastery likelihoods, and the production selector. Claims are limited
to one-session diagnosis in the 27 currently open Shanghai high-school chemistry
nodes. They are not claims about learning gains, longitudinal mastery, or humans.

Production local likelihoods differ between `P` and `U` by 0.10 (`gamma + DELTA_P`
versus `gamma`). With unlimited independent observations these states are
asymptotically identifiable. Consequently, this study must use the terms
**budget-limited weak identifiability** or **practical non-identifiability at the
pre-specified budgets**. It must not claim mathematical, structural, or asymptotic
non-identifiability. Likewise, the production selector's no-harm behavior is a
testable H2 hypothesis, not an implementation invariant.

The full grid is 27 targets x 4 truth states x 3 arms x 50 replicates x 2
generator conditions = 32,400 simulated journeys. One maximum-25-item trajectory
provides nominal-budget views at 9, 15, and 25 items. These are not all administered
checkpoints: a journey that satisfies the production convergence rule early stops,
and its terminal posterior and convergence time carry forward to later nominal-budget
views. Every view therefore reports both its nominal budget and the journey's actual
administered-item count. Budget exhaustion terminates a journey but does not count as
convergence unless the confidence-gap rule was already satisfied.

## Analysis Sets And Census Eligibility

The intention-to-simulate set contains all 27 targets, all four truth states, all
three arms, both generator conditions, and all 50 preassigned replicates. H3 and
descriptive results use this full set.

The confirmatory H1/H2 set is frozen mechanically by the pre-run census. A target is
eligible only if it is returned by `ItemCatalog.from_default_data().open_nodes()` and
at least one of its KG prerequisites resolves to a trusted R5 node with at least two
independent deterministic content families. Each `CatalogItem.family_id` is one
independent family. Label resolution may only replace Chinese parentheses `（` and
`）` with `(` and `)`, and `/` with `_`, on both sides before exact comparison.
The census will retain each raw label, normalized label, and resolved label. It will
not hand-map, fuzzy-match, or semantically equate unrelated labels.

Open targets with no KG prerequisites, or without a prerequisite meeting the
two-family threshold, remain in the full grid and are recording-only exclusions from
H1/H2. In addition to the required full-grid empirical with-replacement stress
estimand, H1 and H2 are always reported under a capacity-explicit no-repeat sensitivity
estimand. Before any responses are sampled, each nominal budget `b` receives a
three-arm common-support target set. A target enters that set only if, after the two
held-out families are reserved, it has at least `b` independent deterministic local
families, at least `floor(b/3)` prerequisite families for C's quota, and a deterministic
family/item matching that can fill all B and C positions without any repeat. This
capacity also leaves A a `b`-family local fallback after response-dependent EIG choices.
The sensitivity estimand uses this same precomputed target set for all three arms.
Capacity counts and all common-support exclusions are shown beside the primary result;
no arm-specific relaxation or imputation is allowed.

No KG, R5, or other official record will be changed. No journey is excluded
because of its responses, posterior, convergence, effect direction, or provider.
Schema-invalid or incomplete simulation records are excluded from estimands, counted
by reason, and never silently replaced. A deterministic implementation failure must
be fixed and the affected complete cell rerun from its original seed before analysis;
both failure and rerun remain in the audit log.

## Arms And Item Allocation

- **A, belief-triggered EIG:** call production `engine.selector.select_next`.
  Prerequisite candidates are enabled only for census-eligible targets. Candidate
  selection otherwise follows the production selector and production tie behavior.
- **B, fixed ladder without prerequisites:** request the cyclic difficulty ladder
  `(0.25, 0.50, 0.75, 1.00)` for administered positions 1 through 25 and select only
  target-local items.
- **C, fixed ladder with quota probes:** use the same ladder and replace every third
  administered item (positions 3, 6, 9, ..., 24) with a prerequisite item. Other
  positions use target-local items.

Fixed-arm selection chooses the nearest available empirical difficulty. Ties are
resolved by `(absolute_distance, seeded_stable_family_rank, item_id)`, where the family
rank is derived from the journey's SHA-256 item-order seed and is fixed within an
epoch. A missing difficulty is never synthesized. Within each node pool, two families
are reserved for held-out scoring as specified below. Remaining families are traversed
in seeded shuffled epochs: every
available family is used once before a family can recur, with one item per family per
epoch and stable `item_id` tie-breaking inside a family. A new seeded shuffle begins
only after an epoch is exhausted. Each journey records exact-item repeat fraction,
family repeat fraction, unique-item count, unique-family count, and actual administered
count. The required full-grid estimand is explicitly named **empirical with-replacement
stress**, not independent-item evidence. Sparse-pool repetition is disclosed rather
than presented as 25 unique production-grade items.

Arm A calls the production selector over the same non-held-out empirical pool. Seen
item IDs are excluded until no eligible unseen item remains; family-epoch traversal
governs replenishment. Arm C applies the same rule separately to its local and
prerequisite pools. All arms receive the same administered-item budget.

## Response Generators And Inference

The **matched** generator samples binary responses from the production
`engine.mastery` local or prerequisite correct-probability function for the true
state, empirical difficulty, and item type. The inference path uses the same
production functions and constants.

The **misspecified** generator draws per administered item `slip ~ U[0.05, 0.20]`
and `guess ~ U[0.15, 0.35]`. It also draws one replicate-level ability offset from
`Normal(0, 0.05)`, clipped to `[-0.10, 0.10]`, and adds it to that replicate's
per-item correct probability before clipping the final probability to
`[1e-6, 1 - 1e-6]`. Local probabilities otherwise retain the production functional
form and `DELTA_P`; prerequisite probabilities otherwise retain the production
state structure. Inference always retains the unmodified production constants.
The shared ability offset induces within-person response correlation without making
the inference model aware of it. The mandated `guess` range also applies to numeric
items despite production numeric guess being 0.03. This is a deliberate severe-stress
condition, not a realistic numeric-response model, and misspecified results are always
reported separately by item type as well as in aggregate.

All posterior updates must import production `engine.mastery`; Arm A and all stopping
decisions must import production `engine.selector`. A parallel Bayesian updater,
likelihood implementation, EIG implementation, or convergence implementation is not
permitted.

## Seeds And Reproducibility

The master simulation seed is `20260713`. Replicate randomness is derived independently
as the first 128 bits of SHA-256 over the UTF-8 string
`yher-confirmatory-v1|20260713|target|truth|condition|replicate`, where replicate is
zero-based `0..49`. Arms for the same target/truth/condition/replicate share the
response-noise stream prefix, while arm-specific item-order randomness is derived by
appending `|arm|purpose`. The bootstrap seed is `2026071301`; the census seed is
`2026071300` (the census itself has no stochastic decisions). Stable hash ordering,
not Python's process-randomized `hash()`, is used everywhere.

Every artifact records `simulated:true`, repository HEAD, analysis-plan commit,
configuration SHA-256, seed derivation version, UTC creation time, and SHA-256 hashes
of every R5/KG/catalog input. Simulated event records additionally require non-empty
`persona_id`, `provider`, and `model_id` fields. Programmatic runs use explicit
simulation identifiers rather than implying human participants.

## Stopping Rule And Checkpoints

After each administered item, call production `engine.selector.should_stop`.
Convergence is the production confidence condition: the top-two posterior gap is
greater than `0.45` and there are at least three direct local answers. At item 25,
production `should_stop` is called once with `budget_items=25` to terminate and again
with `budget_items=26` to distinguish confidence convergence from exhaustion. The
second call must be true for the terminal record to count as confidence-converged;
otherwise it is budget-exhausted and non-converged. Outcomes are
read at nominal-budget views 9, 15, and 25 from the single trajectory. If the journey
stopped earlier, its terminal posterior is carried forward and the lower actual
administered count remains visible. If it has not stopped, its posterior after exactly
the nominal number of administered items is used.

## Outcomes

At each nominal-budget view the study reports:

1. correct-convergence probability (converged and posterior argmax equals truth);
2. misdiagnosis probability (converged and posterior argmax differs from truth);
3. terminal argmax accuracy, whether converged or not;
4. four-by-four truth-versus-diagnosis confusion matrix;
5. severe misdiagnosis probability (`M` diagnosed as `U` or `U` as `M`) over all
   terminal outputs as the primary denominator, plus the converged-only denominator;
6. convergence-time distribution, including non-convergence at the budget;
7. held-out Brier score; and
8. actual administered-item count, unique-item/family counts, and exact-item/family
   repeat fractions.

H1's primary outcome is `P` correct convergence at 15. Its primary rescue contrast is
Arm A minus Arm B. H2's primary outcome is `C` misdiagnosis at 9. Its primary harm
contrast is Arm C minus Arm A. The Arm A minus Arm B `C`-misdiagnosis contrast is the
pre-specified no-harm comparison with a +0.05 non-inferiority margin. H3 is the
subordinate Arm A minus Arm B overall terminal-accuracy and convergence-time sanity
contrast. H4 repeats the H1/H2 estimands under misspecification and reports the
matched-to-misspecified degradation. H5 concerns only LLM-persona manipulation and
cross-provider behavioral consistency; it never contributes to programmatic accuracy
claims. Each H1/H2 point estimate and contrast is presented twice: (1) the required
eligible-target full-grid empirical with-replacement stress estimand and (2) the
budget-specific, three-arm-common-support no-repeat sensitivity estimand. Neither may
be hidden because it is less favorable.

## Held-Out Brier Protocol

For each target, deterministic local families are ordered by SHA-256 of
`yher-heldout-v1|20260713|target|family_id`; the first two families are permanently
held out from all arm candidate pools. The representative held-out item in each family
is the lexicographically first `item_id`. At each nominal-budget view, the current
posterior is mapped through the production local correct-probability function for each
held-out item to obtain a posterior-predictive probability of a correct response. One held-out
binary outcome per family and journey is sampled from that journey's assigned response
generator using the first 128 bits of SHA-256 over
`yher-heldout-outcome-v1|20260713|target|truth|condition|replicate|family_id`. The seed
deliberately contains no arm, so A/B/C receive the same paired held-out outcome for a
target/truth/condition/replicate/family. It never updates the posterior. The Brier
score is the mean squared error across the two held-out outcomes. Under the matched
generator this is an internal calibration sanity check only. The trusted R5 items
contribute item type, difficulty, family, and pool structure, not observed human
item-content behavior, so neither matched Brier nor any other programmatic result is
evidence of external validity. Targets unable to reserve two deterministic local
families are recorded as Brier-unavailable, without imputation; the open-node threshold
is expected to prevent this case.

## Interval Estimation

All reported effects include two-sided percentile bootstrap 95% confidence intervals
from exactly 10,000 resamples. The fixed target set is weighted equally: within each
target, resample the 50 replicate IDs with replacement, preserve the paired arms and
their matched replicate IDs, compute the target estimate, then average target
estimates. Contrasts are computed within each resample, never by subtracting endpoints
of separate intervals. Confusion matrices and unpaired descriptive rates use the same
target-stratified replicate resampling. No confirmatory claim depends on a naked
p-value. Exact numerators, denominators, point estimates, and interval endpoints are
reported.

## Hypothesis Decisions

- **H1 supported:** at checkpoint 15 under the matched generator, Arm A's `P`
  correct-convergence point estimate is at least 0.50 and the 95% CI for A minus B is
  strictly above 0. **Partially supported:** exactly one condition holds, or both point
  estimates are in the predicted direction but the rescue CI includes 0. **Not
  supported:** Arm A is below 0.50 and the rescue point estimate is non-positive.
  Arm B near zero is reported as observed and is never tuned into existence. The
  no-repeat sensitivity is co-reported; a non-positive rescue direction there
  downgrades an otherwise supported result to partially supported.
- **H2 supported:** at checkpoint 9 under the matched generator, the 95% CI for Arm C
  minus Arm A `C` misdiagnosis is strictly above 0 and the upper 95% bound for Arm A
  minus Arm B is below +0.05. **Partially supported:** only one of the harm and no-harm
  criteria holds. **Not supported:** the harm point estimate is non-positive or Arm A
  is inferior to B by at least 0.05 at the point estimate. The no-repeat sensitivity
  is co-reported; a non-positive harm direction there downgrades an otherwise
  supported result to partially supported.
- **H3 supported:** the matched-condition Arm A minus Arm B overall terminal-accuracy
  CI lower bound is at least 0 and Arm A's median convergence time is no longer than
  B's at checkpoint 15. **Partially supported:** one criterion holds. **Not supported:**
  both point directions favor B. This result is subordinate and not the headline.
- **H4 supported:** under misspecification, both H1 rescue and H2 harm point contrasts
  retain their predicted positive directions. **Partially supported:** one persists.
  **Not supported:** neither persists. Significance is not required for this direction
  check; degradation in every primary point estimate is reported with its bootstrap CI.
- **H5 supported:** at least five providers complete at least 45 of 50 personas in
  both arms, the predeclared weak/strong accuracy bands pass, and the target-misconception
  hit-rate contrast over each item's random-wrong-option baseline has a 95% CI strictly
  above 0 after at most one pre-observation prompt rewrite. **Partially supported:**
  four providers meet completion and manipulation gates, or only one manipulation gate
  passes. **Not supported:** fewer than four providers qualify or neither manipulation
  gate passes. Fleiss kappa is reported descriptively without a post hoc cutoff, and
  provider/persona exclusions are fully disclosed.

## Honest Reporting And Stopping

The confirmatory grid stops after all preassigned cells have either produced 50 valid
replicates or have a recorded structural/technical failure. There is no interim
efficacy look, precision-triggered extension, optional stopping, replacement of
unfavorable cells, or sample-size increase after viewing effects. Provider execution
may stop for the predeclared completion/circuit-breaker rules, never for effect size.

H1/H2 failures, reversals, wide intervals, sparse-pool repeats, unavailable Brier
cells, provider failures, and disagreement with the T0 pilot will be reported
unchanged. The pilot is hypothesis-generating; it will not be pooled with the
confirmatory estimates. Negative results trigger interpretation and limitations, not
parameter tuning or hidden exclusions.
