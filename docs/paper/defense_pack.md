# Defense Pack: Prerequisite-Probe Timing Under Item Budgets

**Status:** programmatic H1-H4 analysis is complete. H5 was excluded pre-outcome under
the frozen mapping gate; zero providers qualified and no H5 outcome decision was made.
Pilot observations may explain why hypotheses were chosen, but they are not
confirmatory evidence and must never be pooled with confirmatory results.

## Evidence Index

- **[E01] Claim boundary:** `../../experiments/analysis_plan.md`, headings
  `Question And Scope` and `Analysis Sets And Census Eligibility`.
- **[E02] Arms and allocation:** `../../experiments/analysis_plan.md`, heading
  `Arms And Item Allocation`.
- **[E03] Generators and provenance:** `../../experiments/analysis_plan.md`, headings
  `Response Generators And Inference` and `Seeds And Reproducibility`.
- **[E04] Stopping and outcomes:** `../../experiments/analysis_plan.md`, headings
  `Stopping Rule And Checkpoints` and `Outcomes`.
- **[E05] Held-out scoring:** `../../experiments/analysis_plan.md`, heading
  `Held-Out Brier Protocol`.
- **[E06] Statistics and decisions:** `../../experiments/analysis_plan.md`, headings
  `Interval Estimation` and `Hypothesis Decisions`.
- **[E07] Negative-result policy:** `../../experiments/analysis_plan.md`, headings
  `Honest Reporting And Stopping` and both dated amendments.
- **[E08] Pilot report:** `../../../scratchpad/T0_dosimetry_report.md`, headings
  `1. 执行摘要（TL;DR）`, `2. 方法与建模选择`, `3. 主结果：C 状态`,
  `4. P 状态是完全盲区`, and `5. 两个正交杠杆`.
- **[E09] Pilot raw evidence:** `../../../scratchpad/t0_results.json`, objects
  `_prereq_experiment` and `_meta`; and `../../../scratchpad/t0_montecarlo.py`,
  constants `SEED`, `BUDGETS`, `N_STUDENTS`, functions `simulate_group` and
  `summarize`.
- **[E10] Production likelihoods:** `../../engine/mastery.py`, constants `GAMMA`
  and `DELTA_P`, and functions `local_correct_probs`, `prereq_correct_probs`,
  `bayes_update`, and `observe`.
- **[E11] Production selection:** `../../engine/selector.py`, functions
  `item_eig`, `select_next`, and `should_stop`, plus constants `GAP_THRESHOLD` and
  `MIN_DIRECT_ANSWERS`.
- **[E12] Confirmatory implementation:** `../../experiments/confirmatory/response.py`,
  functions `generator_probability`, `misspecified_probability`, `sample_held_out`,
  and `score_held_out`; `../../experiments/confirmatory/simulation.py`, functions
  `run_paired_unit`, `run_journey`, and `production_confidence_stop`.
- **[E13] Pool and repeat controls:** `../../experiments/confirmatory/catalog.py`,
  functions `load_catalog_context` and `_reserve_held_out`;
  `../../experiments/confirmatory/allocation.py`, classes `FamilyEpoch` and
  `FixedLadderAllocator`, function `precompute_common_support`; and
  `../../analysis/no_repeat.py`, function `validate_no_repeat_sets`.
- **[E14] Raw-data and pairing checks:** `../../analysis/contracts.py`, function
  `validate_programmatic_grid`; `../../analysis/dataset.py`, function
  `load_manifest_dataset`; and `../../analysis/views.py`, functions
  `rebuild_views_from_events` and `validate_raw_views`.
- **[E15] Isolation and repository binding:**
  `../../experiments/confirmatory/provenance.py`, functions
  `verify_repository_binding`, `confirmatory_output_path`, and
  `aggregate_isolation_assertions`; and `../../experiments/confirmatory/storage.py`,
  functions `write_shards_atomic` and `validate_shard`.
- **[E16] LLM-persona gates:** `../../experiments/llm_sim/README.md`, headings
  `Pre-Observation Gate` and `Live Provider Run`;
  `../../experiments/config/llm_sim_v1.json`; and
  `../../experiments/llm_sim/panel.py`, function `freeze_manipulation_panel`.
- **[E17] Product and system boundary:** `../YHer_技术报告_v2_draft.md`, headings
  `2. 产品定义、用户旅程与当前能力边界`, `5. 五引擎`,
  `6. 真诊断、讲解信任边界与 held-out`, and `11. 工程教训、限制与下一步`.
- **[E18] Manuscript claim map:** [`main.md`](main.md), sections 1, 4-10.
- **[E19] Machine result boundary:** [`results_contract.md`](results_contract.md),
  headings `Replacement Protocol` and `Frozen Interpretation Rules`.
- **[E20] Verified prior work:** [`references.json`](references.json), all 17 records.
- **[E21] Responsibility evidence:** [`decision_log.md`](decision_log.md), entries
  D01-D13; and `../../../PROJECT_HANDOFF/codex_briefs/2026-07-13_仿真评估实验与论文总攻.md`,
  S4-S5 and hard gates H-A through H-H.
- **[E22] H5 lifecycle and accounting:** [`results_contract.md`](results_contract.md),
  JSON paths `hypotheses.H5`, `denominators.provider_lifecycle_counts`, and
  `h5_provider_exclusion_disclosure`; `/tmp/yher_sprint2/h5_results/h5_results.json`,
  JSON paths `status`, `hypothesis`, `denominators`, and
  `provider_exclusion_disclosure`; and
  `/tmp/yher_sprint2/h5_results/provider_ledger.json`, JSON path `totals`.

## 1. What is the exact research question?

**Answer.** The study asks whether a one-session, four-state chemistry diagnosis can
distinguish a prerequisite gap from mastered, chain-instability, and unlearned states
under a binary response channel and fixed item budgets, and whether the timing policy
for prerequisite probes changes that finite-budget distinguishability. It does not
test learning gains or long-term memory. Programmatic decisions are
`H1=partially_supported`, `H2=not_supported`, `H3=supported`, and `H4=supported`; H5
was excluded pre-outcome, zero providers qualified, and no H5 outcome decision was
made.

**Evidence:** [E01], [E18], [E19].

## 2. Why diagnose *why* a student is wrong instead of only predicting correctness?

**Answer.** The states imply different actions. A prerequisite gap calls for an
earlier concept, chain instability calls for a causal bridge, and unlearned target
knowledge calls for a direct introduction. A correct/incorrect score alone cannot
justify those different routes. The product loop uses this distinction for diagnosis,
resource routing, held-out verification, and profile updates, while the study isolates
only the diagnostic measurement question.

**Evidence:** [E17] sections 2, 5, and 6; [E18] section 1.

## 3. What do M, P, C, and U mean operationally?

**Answer.** M is mastered target knowledge; P is a missing prerequisite; C is an
unstable concept or causal chain despite relevant prerequisites; U is unlearned target
knowledge with prerequisites treated as intact by the model. They are latent
diagnostic states with different response likelihoods, not personality labels or
clinical categories. The confirmatory truth state remains fixed during one journey.

**Evidence:** [E01], [E10], [E18] sections 1.2 and 5.4.

## 4. Why use four states instead of IRT or Bayesian knowledge tracing?

**Answer.** BKT models changing skill acquisition over time, while this study needs a
small set of action-linked causes within one static diagnostic session. The four
states are not claimed to replace proficiency models or BKT; they are a
product-specific hypothesis whose recovery can fail and must be tested. Longitudinal
transitions are explicitly out of scope.

**Evidence:** [E01]; [E18] sections 2.2 and 9; [E20] records
`corbett-anderson1994` and `pelanek2017`.

## 5. Are P and U fundamentally impossible to distinguish?

**Answer.** No. Their production local correct-response probabilities differ by 0.10,
so unlimited independent observations identify them under the model. The permitted
claim is **budget-limited weak identifiability** or **practical non-identifiability at
the pre-specified budgets of 9, 15, and 25 items**. Any stronger mathematical or
asymptotic claim would contradict the frozen plan. H1 was only partially supported:
at 15 items, A achieved 148/1,150 = 12.87% P correct convergence versus B's 9/1,150 =
0.78%, a +12.09 percentage-point contrast (95% CI 10.35 to 13.83), but A remained far
below the frozen 50% absolute gate. The no-repeat sensitivity was +0.50 percentage
points (95% CI 0.00 to 1.50), so it does not justify a stronger claim.

**Evidence:** [E01] `Question And Scope`; [E10] `local_correct_probs`; [E18]
sections 1.2 and 9; [E19].

## 6. Why were budgets 9, 15, and 25 chosen?

**Answer.** They were fixed before confirmation to match the existing 30-minute,
1-hour, and 2-hour diagnostic planning views and to span shallow, intermediate, and
longer sessions. They are nominal views of one maximum-25-item trajectory, not three
separate opportunities to stop when a result looks favorable. Actual administered
counts remain visible after early convergence.

**Evidence:** [E04]; [E08] section 2.1; `../../engine/planner.py`, constant
`BUDGET_TABLE` and function `session_budget`.

## 7. Why study a binary response channel when wrong-option information could help?

**Answer.** The current trusted catalog does not provide a reliable misconception map
for the confirmatory channel, so correct/incorrect is the honest production condition.
Adding inferred distractor semantics would introduce unverified information. The
paper therefore states the binary and multiple-choice-heavy limitation rather than
claiming rich error-process identification.

**Evidence:** [E08] sections 1 and 2; [E17] sections 3 and 11; [E18] limitation 5.

## 8. Why expected information gain?

**Answer.** In this production implementation, EIG asks which eligible observation is
expected to reduce posterior uncertainty most. It gives prerequisite questions a
chance to compete only when their state-dependent likelihoods make them informative.
Adaptive item selection is established prior work; this study does not claim that its
particular entropy calculation is a literature contribution. It is the production
policy tested against timing controls.

**Evidence:** [E02]; [E11] `item_eig` and `select_next`; for the established adaptive
testing context, [E20] CAT records
`lord1971`, `weiss1982`, and `weiss-kingsbury1984`.

## 9. Why are three arms necessary?

**Answer.** A versus B compares adaptive belief-triggered selection with a local-only
fixed ladder. C versus A tests the additional harm from inserting prerequisite evidence
on a fixed schedule; A versus B separately tests whether belief-triggered selection
avoids material harm relative to asking local questions only. The results split those
claims. C-state misdiagnosis at 9 items was 37.39% in A, 27.65% in B, and 43.04% in C.
C minus A was +5.65 percentage points (95% CI 2.35 to 8.96), supporting fixed-quota
harm. But A minus B was +9.74 percentage points (95% CI 7.22 to 12.26), violating the
+5 percentage-point no-harm margin. Belief-triggered harm avoidance was therefore
falsified, and the frozen joint H2 decision is `not_supported`.

**Evidence:** [E02]; [E18] sections 1.3 and 5.3; [E19].

## 10. Why is the baseline a fixed difficulty ladder rather than random questions?

**Answer.** A deterministic ladder gives a reproducible, interpretable local-only
control and avoids an arbitrary random baseline. It cycles requested difficulties
0.25, 0.50, 0.75, and 1.00, chooses the nearest empirical item, and resolves ties by a
frozen tuple. Missing difficulties are not synthesized.

**Evidence:** [E02]; [E13] `FixedLadderAllocator`.

## 11. Why does Arm C insert a prerequisite item every third position?

**Answer.** That fixed quota reproduces the pilot stress policy: three prerequisite
questions within a nine-item session. It is deliberately simple and potentially
harmful, which makes it a useful timing-policy control. It is not presented as a
recommended teaching policy.

**Evidence:** [E02]; [E08] section 5.1; [E09] `simulate_group` parameter
`prereq_period` and `_prereq_experiment`.

## 12. Why is H1 evaluated at 15 items and H2 at 9?

**Answer.** H1 uses the predeclared intermediate 15-item checkpoint because the frozen
confirmatory design aligned it with the product's one-hour diagnostic threshold and
paired an absolute 50% gate with the A-versus-B contrast. T0 motivated testing
prerequisite evidence, but it did not establish that P required a longer budget: its
local-only P result remained 0.0% at both 9 and 15 items, while its fixed probes acted
within nine. H2 uses the shortest nine-item checkpoint because the fixed-quota C harm
appeared there and the frozen rule combines C-versus-A harm with A-versus-B no-harm.

**Evidence:** [E06] `Hypothesis Decisions`; [E08] sections 3-5; [E07];
`../../engine/planner.py`, `BUDGET_TABLE`.

## 13. How is this different from 40 years of computerized adaptive testing?

**Answer.** It does not claim that adaptive selection is new or generally better.
Classic CAT work established measurement efficiency and item-selection methods. This
study focuses on an action-linked P/U/C state distinction that needs cross-node
prerequisite evidence, then separates belief-triggered timing from a fixed quota.
H3, the generic adaptive-versus-fixed comparison, is explicitly subordinate. It was
supported: A minus B terminal accuracy was +10.04 percentage points (95% CI 8.89 to
11.19), and median convergence was 8 items for A versus 11 for B. These are a sanity
check under the simulated production model, not the paper's novelty claim.

**Evidence:** [E18] sections 1.3, 2.1, and 5.1; [E20] CAT records
`lord1971`, `weiss1982`, `weiss-kingsbury1984`, `barrada2010`, and `han2018`; [E19].

## 14. How do you know the pilot P result of 0.0% was not a calculation bug?

**Answer.** The 0.0% value is a pilot observation, not a confirmatory conclusion. It
can be traced from the fixed-seed source to the raw `_prereq_experiment` table and the
report. The script separately records convergence, correct convergence, and
misdiagnosis, applies the direct-answer gate, and used 4,000 simulated students per
cell. The production-bound study is intentionally independent enough to falsify that
pilot pattern; if it does, the confirmatory result wins.

**Evidence:** [E08] sections 2.3, 4, and 5.1; [E09]; [E07].

## 15. Why is T0 only hypothesis-generating?

**Answer.** T0 used synthetic difficulty grids and the then-specified likelihoods,
not the final empirical catalog allocation and full production runner. It revealed a
failure mode and motivated H1/H2, but reusing it as confirmation would select a claim
and test it on the same evidence. Its estimates remain in the pilot section and are
never pooled with S3.

**Evidence:** [E08] sections 1-2; [E18] section 4; [E19] opening paragraph and
`Frozen Interpretation Rules`.

## 16. Does the matched simulator make the study circular?

**Answer.** Partly. The matched generator and inference engine share production
probabilities, so matched results test internal recovery under the model rather than
human realism. The paper says this directly. Pairing, held-out scoring, and production
catalog structure improve precision and realism of the engineering test, but they do
not remove model dependence.

**Evidence:** [E03]; [E05]; [E18] sections 5.4 and 9.

## 17. Why use this particular misspecified generator?

**Answer.** It perturbs item-level slip and guess within frozen ranges and adds a
bounded person-level ability offset while leaving inference constants unchanged. This
creates wrong-model and within-person-correlation stress without tuning to observed
effects. Numeric items also receive the deliberately severe guess range, and results
must be separated by item type.

**Evidence:** [E03] `Response Generators And Inference`; [E12]
`misspecified_probability`.

## 18. Does success under misspecification prove the model is realistic?

**Answer.** No. H4 asks whether the H1/H2 point directions survive one declared
perturbation family and how much they degrade. That is a robustness check, not a
validation of human behavior. Failure is informative; success still leaves many
unmodeled response processes. H4 was supported under its directional rule: the
misspecified H1 rescue was +12.09 percentage points (95% CI 10.17 to 14.00), and the
misspecified H2 fixed-quota harm was +6.96 percentage points (95% CI 3.65 to 10.26).
Those effects establish robustness only to this frozen perturbation family.

**Evidence:** [E06] H4 decision rule; [E18] sections 7 and 9; [E19].

## 19. How do you know the confirmatory runner uses the real production engine?

**Answer.** The analysis contract forbids a parallel updater, likelihood, EIG rule,
or convergence rule. The runner imports the production mastery and selector paths,
and the model identifier binds hashes of those engine files. Code review and tests can
verify the call path instead of trusting prose.

**Evidence:** [E03]; [E10]; [E11]; [E12] `production_model_id`, `run_journey`, and
`production_confidence_stop`; [E21] hard gate H-E.

## 20. What if sparse item pools force repeated questions?

**Answer.** Repetition is not hidden. The required full-grid estimand is named an
empirical with-replacement stress estimand and reports exact-item and family repeat
fractions. A second pre-outcome sensitivity uses one budget-specific no-repeat
common-support target set for all three arms. Neither estimate can be suppressed for
being less favorable.

**Evidence:** [E01] `Analysis Sets And Census Eligibility`; [E02]; [E13].

## 21. How do you separate confidence convergence from merely running out of items?

**Answer.** After every item, the runner calls the production stop rule. Confidence
convergence requires a top-two posterior gap above 0.45 plus at least three direct
local answers. At the final budget, an additional check distinguishes confidence from
exhaustion. A non-converged terminal argmax can be reported descriptively but cannot
be mislabeled as correct convergence.

**Evidence:** [E04]; [E11] `should_stop`; [E12] `production_confidence_stop`.

## 22. Why use held-out Brier, and why is it not external validation?

**Answer.** Two deterministic local families are reserved before allocation. Their
paired outcomes are never used to update the posterior; Brier measures posterior-
predictive calibration on those reserved outcomes. Under the matched generator the
same model family generates them, so the score is an internal calibration check only.

**Evidence:** [E05]; [E12] `sample_held_out` and `score_held_out`; [E13]
`_reserve_held_out`.

## 23. Why bootstrap instead of reporting only p-values?

**Answer.** The target-stratified paired bootstrap preserves the fixed targets and
paired arms, resamples replicate IDs within target, and produces point-contrast
intervals directly. The paper reports numerators, denominators, effects, and 95%
intervals. Frozen decision branches use those intervals; no claim depends on a naked
p-value.

**Evidence:** [E06] `Interval Estimation` and `Hypothesis Decisions`.

## 24. Why report severe M-to-U and U-to-M errors separately?

**Answer.** Those errors cross the widest action boundary: they can turn a mastered
student into a direct-instruction recommendation or treat an unlearned student as
mastered. Reporting them over all terminal outputs and again over converged outputs
prevents a favorable denominator choice from hiding risk.

**Evidence:** [E04] `Outcomes`; `../../experiments/confirmatory/metrics.py`, function
`is_severe_misdiagnosis`; [E19] `Frozen Interpretation Rules`.

## 25. Why include LLM-simulated students at all?

**Answer.** They were intended to provide a secondary stress test of prompted behavior
across official providers. They do not contribute to programmatic accuracy claims and
do not stand in for humans. In this run H5 was excluded pre-outcome because the frozen
explicit misconception-to-option mapping gate was unavailable. No cross-provider
behavioral outcome was evaluated.

**Evidence:** [E06] H5 decision rule; [E16]; [E18] sections 2.4, 5.1, and 9; [E22].

## 26. How do you know a "weak" LLM persona is actually weak?

**Answer.** The label alone is not trusted. Weak and strong personas must pass
predeclared observed accuracy bands, and the target misconception requires a separate
wrong-option hit-rate contrast against a random-wrong-option baseline. A failed gate
permits at most one predeclared prompt rewrite; exclusions remain visible. Passing
these gates supports prompt manipulation, not human realism.

For this run the explicit option map was absent, so the required dual manipulation
check could not qualify any provider. Accuracy-only collection artifacts cannot replace
the frozen misconception-level gate.

**Evidence:** [E06] H5 decision rule; [E16] frozen config fields `accuracy_bands`,
`maximum_prompt_rewrites`, and `minimum_complete_per_cell`; [E22].

## 27. What happens if the catalog cannot map a misconception to an exact wrong option?

**Answer.** The system does not infer a mapping from option text, model rationale, or
observed responses. The cell is marked `excluded_pre_outcome`, and the misconception-
hit contrast is not reportable for it. The current runner documentation states that
the required explicit machine mapping is absent in the present catalog. The final
lifecycle therefore records H5 as excluded pre-outcome, zero qualifying providers, and
a null decision. This is not a negative H5 result. H5 status is read only from the
validated lifecycle in the results contract.

**Evidence:** [E16] `Pre-Observation Gate` and `freeze_manipulation_panel`; [E19];
[E22].

## 28. How do you prevent provider failures or exclusions from creating survivor bias?

**Answer.** Each provider has independent checkpoints, retries, circuit breaking,
model-ID drift checks, and accounting. A provider-arm cell needs at least 45 of 50
personas, and H5 support requires enough providers to qualify in both arms. Excluded
providers and personas remain in manifests and denominators rather than disappearing
from the narrative. The final machine lifecycle classified DeepSeek, GLM, and Kimi as
invalid calibration schema; Doubao as a network interruption; and MiniMax and Tongyi
as post-calibration exclusions. The immutable accounting ledger records 10,849
requests, 5,571,972 input tokens, 1,201,999 output tokens, and CNY 103.51977121. These
are collection and cost records, not H5 outcome evidence.

**Evidence:** [E06] H5 decision rule; [E16] `Live Provider Run` and frozen
`provider_policy`; `../../experiments/llm_sim/runner.py`, classes
`ProviderCallPolicy`, `CircuitOpenError`, and `ModelDriftError`; [E22].

## 29. What did AI do, and what did the student do?

**Answer.** The student owned direction, requirements, and pre-outcome quality and
honesty gates documented in D01-D13. AI systems proposed designs, implemented code and
automation, executed the simulation and analysis, drafted the current interpretation
and prose, and performed adversarial review under those gates. The audit layer -
commits, tag, tests, hashes, seeds, provenance, and isolation attestations - makes both
the assistance and the controls inspectable. Final interpretation and any submission
decision remain pending student review. This is not a claim that the student manually
authored AI-generated implementation.

**Evidence:** [E21] decision entries D01-D13 and S4-S5 contribution requirements;
[E18] section 8.

## 30. What happens if H1 or H2 fails, and how can anyone trust that answer?

**Answer.** The frozen branches were applied unchanged: `H1=partially_supported`,
`H2=not_supported`, `H3=supported`, and `H4=supported`; H5 was excluded pre-outcome
with a null decision and must not be described as negative. In particular, H2's
fixed-quota harm direction survived, but its belief-triggered no-harm requirement
failed. Neither that failure nor H1's missed 50% gate triggered parameter tuning,
hidden exclusions, extra replicates, or a new decision rule. Trust comes from the
pre-data plan, deterministic seeds, repository and annotated-tag binding, input/config
hashes, simulated-data envelopes, per-shard isolation attestations, raw-shard hashes,
and machine-owned result IDs. Publication, submission, push, and DOI creation remain
outside this draft task and require the student's later decision.

**Evidence:** [E07]; [E15]; [E19]; [E21] hard gates H-A through H-H and decision
entries D10-D12; [E22].

## AI Use Disclosure Template

> AI systems were used as research tools and engineering assistants for design
> proposals, implementation, automation, testing, analysis scaffolding, adversarial
> review, and drafting. The student set the research direction and requirements,
> established pre-outcome quality and honesty gates, and made go/no-go decisions.
> AI systems executed the simulation and analysis and drafted the current result
> interpretation. Final interpretation and any submission decision remain pending
> student review. Contributions are documented through dated decision anchors;
> technical work is auditable through commits, tests, hashes, seeds, provenance, and
> isolation records. Simulated learners are labeled as simulated and are not presented
> as human evidence.

Template basis: [E18] section 8 and [E21] D01-D13 plus S5.
