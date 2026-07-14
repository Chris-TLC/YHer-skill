# Diagnosing Why a Student Is Wrong Under Item Budgets

## A Confirmatory Study of Prerequisite-Probe Timing in High-School Chemistry

<!-- BEGIN PAPER GENERATED STATUS -->
**Manuscript status:** confirmatory result binding complete; H1-H4 were evaluated and H5 was excluded pre-outcome with machine-bound evidence. All result values and decisions in the generated sections come from the validated machine contract.
<!-- END PAPER GENERATED STATUS -->

## 中文摘要

多数自适应诊断系统回答学生“会不会”，本研究进一步追问：在有限题量内，
系统能否区分学生“为什么不会”？一化儿将单个化学知识点的状态表示为掌握
（M）、前置缺失（P）、思维链不稳（C）和目标知识未学（U）。生产模型中，
P 与 U 在本节点二元题上的正确率只差 0.10，因此本文不声称结构性、数学性或
渐近不可识别，而只研究预先规定的 9、15、25 题预算下的弱可识别性与实际不
可识别性。先导仿真提示：前置探针可显著改善 P 态诊断，但固定配额插入同类
探针可能增加 C 态误诊。为复证这一现象，确证实验在可信生产题库与未改写的
生产推断引擎上比较三种探针时机政策：信念触发的期望信息增益策略、无前置题
的固定难度阶梯，以及每三题固定插入一道前置题的阶梯。实验分别报告匹配与有
意错设定的反应生成条件；辅助 LLM 画像实验须同时通过正确率带和目标误概念
命中率操纵检验，且不作为真人学习证据。分析计划、种子、停止规则、负结果分
支与排除规则均在采集前冻结。确证数字仅由机器生成的结果合同回填；无论先导
结论是否复现，均如实报告。

<!-- BEGIN PAPER GENERATED ABSTRACT FINDINGS ZH -->
**确证结果。** H1=partially_supported (complete); H2=not_supported (complete); H3=supported (complete); H4=supported (complete); H5=not evaluated (excluded pre-outcome)。H1 A 臂收敛率=0.128696；救活效应 95% CI=[0.103478, 0.138261]。H2 伤害效应 95% CI=[0.023478, 0.089565]；无伤害对照 95% CI=[0.072174, 0.122609]。上述结果仅限于有限题量仿真，不证明真人学习效果或教育效能。
<!-- END PAPER GENERATED ABSTRACT FINDINGS ZH -->

## Abstract

Most adaptive diagnostic systems estimate whether a learner can answer a question.
This project asks a narrower and harder question: can a short diagnostic distinguish
*why* the learner is likely to fail? YHer represents one chemistry concept with four
states: mastered (M), prerequisite gap (P), chain instability (C), and unlearned
target knowledge (U). On target-local binary questions, the production model assigns
P and U correct-response probabilities that differ by only 0.10. The states are
asymptotically identifiable under the model; this project therefore makes no claim of
structural, mathematical, or asymptotic non-identifiability. It studies
**budget-limited weak identifiability**, or **practical non-identifiability at the
pre-specified budgets of 9, 15, and 25 items**.

A hypothesis-generating synthetic pilot suggested two simultaneous effects. Fixed
prerequisite probes substantially improved P-state diagnosis, but the same fixed
quota increased C-state misdiagnosis. The confirmatory protocol therefore compares
three timing policies using the trusted production question catalog and unmodified
production inference engine: belief-triggered expected information gain (A), a fixed
local-question ladder (B), and the same ladder with a prerequisite question every
third position (C). Matched and deliberately misspecified response generators are
reported separately. A secondary LLM-persona study is gated by accuracy and
misconception manipulation checks and is not evidence about human students. The
analysis plan, seeds, populations, stopping rules, hypothesis branches, and negative-
result policy were frozen before confirmatory responses were generated.

<!-- BEGIN PAPER GENERATED ABSTRACT FINDINGS EN -->
**Confirmatory findings.** H1=partially_supported (complete); H2=not_supported (complete); H3=supported (complete); H4=supported (complete); H5=not evaluated (excluded pre-outcome). H1 A rate=0.128696; rescue 95% CI=[0.103478, 0.138261]. H2 harm 95% CI=[0.023478, 0.089565]; no-harm 95% CI=[0.072174, 0.122609]. These are simulated finite-budget results, not evidence of human learning or educational efficacy.
<!-- END PAPER GENERATED ABSTRACT FINDINGS EN -->

## 1. Problem and Product Context

YHer is a pre-alpha, local system for a narrow Shanghai high-school chemistry loop:
diagnose one concept, route the learner to an existing teacher video or constrained
explanation, administer independent practice and held-out questions, and update an
event-reconstructable profile. It does not claim human validation, learning gains,
retention, or production readiness.

The diagnostic distinction matters operationally. A prerequisite gap calls for an
earlier concept; chain instability calls for a worked causal bridge; an unlearned
concept calls for a direct introduction. Treating all three as generic incorrectness
can route a learner to the wrong intervention.

For a target-local item with normalized difficulty \(d\), the production model uses

\[
P(Y=1\mid M,P,C,U)=(0.90,\ \gamma+0.10,\ 0.70-0.20d,\ \gamma),
\]

where \(\gamma=0.25\) for multiple-choice items and 0.03 for numeric items. A
prerequisite item uses

\[
P(Y_{pre}=1\mid M,P,C,U)=(0.90,\ \gamma,\ 0.80,\ 0.75).
\]

The 0.10 P/U local difference is nonzero, so unlimited independent observations can
identify the states. Within a short session, however, that evidence may be too weak
for correct confident convergence. A prerequisite response creates a larger P/U
contrast, but a true C learner is expected to answer prerequisite material correctly
with probability 0.80. Asking such probes indiscriminately can therefore move C
belief toward the wrong states. The scientific question is about *when* to probe,
not whether prerequisite questions are universally beneficial.

## 2. What Is New and What Is Prior Art

Computerized adaptive testing, adaptive item-selection rules, Bayesian knowledge
tracing, and spaced-repetition scheduling are established areas [@lord1971;
@weiss1982; @weiss-kingsbury1984; @barrada2010; @han2018;
@corbett-anderson1994; @pelanek2017; @tabibian2019; @ye2022]. The production policy
uses an expected-information calculation, but this particular entropy calculation is
not claimed as a literature contribution. Adaptive selection alone is not this
project's contribution.

The contribution is a production-bound test of a specific finite-budget failure mode:
whether four diagnostically actionable causes can be separated through a binary
response channel, and whether belief-triggered and fixed-quota prerequisite policies
produce different rescue and harm profiles. The design also makes model dependence
visible through a frozen misspecification condition and a no-repeat common-support
sensitivity analysis.

The product includes an **FSRS-inspired** memory projection. FSRS is treated here as
software and documentation, not as peer-reviewed validation of YHer's decay formula
[@fsrs-repository; @anki-fsrs-manual]. Memory scheduling is outside this one-session
confirmatory question.

## 3. Hypothesis-Generating Pilot

T0 used synthetic difficulty grids, a uniform prior, the then-specified production
likelihood constants, a fixed seed, and 4,000 simulated students per cell. It did not
use the real trusted-item distribution or the final production selector. Its values
are therefore pilot observations only and are never pooled with confirmatory
estimates.

Under binary local questions, pilot C correct convergence ranged from 13.9%-34.7% at
9 items, 45.1%-61.3% at 15, and 58.4%-70.1% at 25. At medium difficulty, P correct
convergence was 0.0% at both 9 and 15 items. Replacing three of nine local questions
with fixed-position prerequisite questions changed P correct convergence from 0.0%
to 73.9%, but changed C misdiagnosis from 26.8% to 46.1%. These observations generated
the confirmatory hypotheses; they do not support the final claims by themselves.

## 4. Confirmatory Protocol

### Hypotheses

- **H1, P rescue:** at budget 15 under the matched generator, Arm A must achieve at
  least 0.50 P correct convergence and the paired 95% interval for A minus B must be
  strictly above zero for full support.
- **H2, timing harm:** at budget 9, the paired 95% interval for C minus A C-state
  misdiagnosis must be above zero, while the upper bound for A minus B must stay below
  the +0.05 no-harm margin for full support.
- **H3, adaptive sanity check:** A versus B overall accuracy and convergence time are
  subordinate because adaptive-testing efficiency is prior art.
- **H4, misspecification:** the H1 rescue and H2 harm point directions are checked
  under the frozen perturbed generator, with degradation reported.
- **H5, LLM personas:** only predeclared provider completion, accuracy-band, and
  misconception-hit gates can make the secondary behavior analysis reportable.

The ordered supported, partially-supported, and not-supported branches are specified
in [`experiments/analysis_plan.md`](../../experiments/analysis_plan.md#hypothesis-decisions).

### Arms and grid

Arm A calls the production selector at every decision and allows eligible local and
prerequisite candidates to compete by expected information gain. Arm B cycles target
difficulties 0.25, 0.50, 0.75, and 1.00 and never asks prerequisite questions. Arm C
uses the same ladder but replaces positions 3, 6, 9, ..., 24 with prerequisite
questions. All observations update the posterior through the production mastery
module; no parallel Bayesian updater, EIG rule, or stopping implementation is
permitted.

The intention-to-simulate grid contains 27 targets x 4 truth states x 3 arms x 50
replicates x 2 generator conditions, or 32,400 planned journeys. One maximum-25-item
trajectory supplies nominal views at 9, 15, and 25. The H1/H2 target population is
fixed mechanically before responses by prerequisite-family availability. The primary
empirical with-replacement stress estimand discloses item and family repetition. A
budget-specific sensitivity analysis uses the same precomputed no-repeat
common-support target set for all three arms.

### Generator separation, outcomes, and stopping

The matched generator samples from production response probabilities. The
misspecified generator independently perturbs per-item slip and guess and adds a
bounded replicate-level ability offset; inference keeps the production constants.
This stress condition reduces but does not eliminate simulator-inference circularity.

At each nominal budget, the analysis reports correct convergence, converged
misdiagnosis, terminal argmax accuracy, four-state confusion matrices, severe M/U
errors, convergence time, actual item count, repeat fractions, and paired held-out
Brier. Matched held-out Brier is an internal calibration check, not external
validation. Confidence intervals use exactly 10,000 target-stratified percentile
bootstrap resamples while preserving paired arms.

The production stop rule requires a top-two posterior gap greater than 0.45 and at
least three direct local answers. Budget exhaustion is recorded separately from
confidence convergence. No interim efficacy look, favorable-cell replacement,
post-outcome sample increase, hidden exclusion, or parameter tuning is allowed.

## 5. Confirmatory Results

<!-- BEGIN PAPER GENERATED RESULTS -->
This compact table is generated from the same validated contract as the main paper. It reports only decision-driving primary evidence and the mandatory stress/common-support sensitivity evidence.

**Machine integrity summary.** 30,800 valid of 32,400 intended programmatic journeys; 1,600 predeclared estimand exclusions; H5 excluded pre-outcome with 0 qualifying providers. Provider lifecycle: invalid calibration schema=deepseek/glm/kimi; network interruption=doubao; post calibration exclusion=minimax/tongyi. 12 excluded provider cells and 464 excluded persona cells.

| Claim | Decision | Compact machine evidence |
|---|---|---|
| H1: A P convergence (complete) | partially_supported | A=12.87%; B=0.78%; A-B=+12.09 pp (95% CI 10.35 to 13.83 pp); no-repeat A-B=+0.50 pp (0.00 to 1.50 pp); stress n=1,150/arm; common-support n=200/arm |
| H2: C-state misdiagnosis (complete) | not_supported | A=37.39%; B=27.65%; C=43.04%; C-A=+5.65 pp (95% CI 2.35 to 8.96 pp); A-B=+9.74 pp (95% CI 7.22 to 12.26 pp); no-repeat C-A=+2.00 pp (95% CI -3.33 to 7.56 pp); A-B=+13.33 pp (95% CI 9.11 to 17.56 pp); stress n=1,150/arm; common-support n=450/arm |
| H3: A-B terminal accuracy (complete) | supported | +10.04 pp (95% CI 8.89 to 11.19 pp); median convergence A=8, B=11 items |
| H4: misspecified rescue and harm (complete) | supported | rescue=+12.09 pp (95% CI 10.17 to 14.00 pp); fixed-probe harm=+6.96 pp (95% CI 3.65 to 10.26 pp) |
| H5: LLM personas (excluded pre-outcome) | not evaluated | excluded pre-outcome; 0 qualifying providers; no outcome decision |

<!-- BEGIN YAU MACHINE AUDIT -->
lifecycle `YAU_VISIBLE_LIFECYCLE_DENOMINATORS`: intended_journeys=32,400; valid_journeys=30,800; estimand_excluded_journeys=1,600; frozen_providers=6; collected_providers=3; qualifying_providers=0; excluded_provider_cells=12; excluded_persona_cells=464; h5_results_path=`h5/h5_results.json`; h5_results_sha256=sha256:8f8368aa7dc32062d747c73031435dc9ae5ff7191f9a4f2077a1e0d918bc3b4f; source_artifact_path=`metric_registry.json`; source_artifact_sha256=sha256:3493b2687cecaa91b0ef7ae25f0dacdaa3ea05a5405c79a4321814d5fa092a7c
result `H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS` / registry `p_rescue.full.matched.b15.arm_A`: value=0.128696; ci95=[0.112174, 0.145217]; denominator=1150
result `H1_P_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS` / registry `p_rescue.full.matched.b15.arm_B`: value=0.007826; ci95=[0.003478, 0.013043]; denominator=1150
result `H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS` / registry `h1.primary.matched.b15.rescue_A_minus_B`: value=0.120870; ci95=[0.103478, 0.138261]; denominator=1150
result `H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS_CI95` / registry `h1.primary.matched.b15.rescue_A_minus_B`: value=[0.103478, 0.138261]; denominator=1150
result `H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT` / registry `h1.no_repeat.matched.b15.rescue_A_minus_B`: value=0.005000; ci95=[0, 0.015000]; denominator=200
result `H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT_CI95` / registry `h1.no_repeat.matched.b15.rescue_A_minus_B`: value=[0, 0.015000]; denominator=200
result `H2_C_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS` / registry `c_misdiagnosis.full.matched.b9.arm_A`: value=0.373913; ci95=[0.346087, 0.401739]; denominator=1150
result `H2_C_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS` / registry `c_misdiagnosis.full.matched.b9.arm_B`: value=0.276522; ci95=[0.250435, 0.302609]; denominator=1150
result `H2_C_C_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS` / registry `c_misdiagnosis.full.matched.b9.arm_C`: value=0.430435; ci95=[0.401739, 0.459130]; denominator=1150
result `H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS` / registry `h2.primary.matched.b9.harm_C_minus_A`: value=0.056522; ci95=[0.023478, 0.089565]; denominator=1150
result `H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95` / registry `h2.primary.matched.b9.harm_C_minus_A`: value=[0.023478, 0.089565]; denominator=1150
result `H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS` / registry `h2.primary.matched.b9.no_harm_A_minus_B`: value=0.097391; ci95=[0.072174, 0.122609]; denominator=1150
result `H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95` / registry `h2.primary.matched.b9.no_harm_A_minus_B`: value=[0.072174, 0.122609]; denominator=1150
result `H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT` / registry `h2.no_repeat.matched.b9.harm_C_minus_A`: value=0.020000; ci95=[-0.033333, 0.075556]; denominator=450
result `H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95` / registry `h2.no_repeat.matched.b9.harm_C_minus_A`: value=[-0.033333, 0.075556]; denominator=450
result `H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT` / registry `h2.no_repeat.matched.b9.no_harm_A_minus_B`: value=0.133333; ci95=[0.091111, 0.175556]; denominator=450
result `H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_COMMON_SUPPORT_CI95` / registry `h2.no_repeat.matched.b9.no_harm_A_minus_B`: value=[0.091111, 0.175556]; denominator=450
result `H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET` / registry `h3.matched.b15.terminal_accuracy_A_minus_B`: value=0.100370; ci95=[0.088889, 0.111852]; denominator=5400
result `H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET_CI95` / registry `h3.matched.b15.terminal_accuracy_A_minus_B`: value=[0.088889, 0.111852]; denominator=5400
result `H3_A_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET` / registry `h3.matched.b15.time_to_confidence.arm_A`: value=8; ci95=[8, 8]; denominator=5400
result `H3_B_MEDIAN_CONVERGENCE_ITEMS_MATCHED_B15_FULL_SET` / registry `h3.matched.b15.time_to_confidence.arm_B`: value=11; ci95=[10, 11]; denominator=5400
result `H4_H1_RESCUE_MISSPECIFIED_B15_ELIGIBLE_STRESS` / registry `h4.misspecified.b15.rescue_A_minus_B`: value=0.120870; ci95=[0.101739, 0.140000]; denominator=1150
result `H4_H2_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS` / registry `h4.misspecified.b9.harm_C_minus_A`: value=0.069565; ci95=[0.036522, 0.102609]; denominator=1150
result `H4_H2_NO_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS` / registry `h4.misspecified.b9.no_harm_A_minus_B`: value=0.080870; ci95=[0.054783, 0.106957]; denominator=1150
result `H4_H1_RESCUE_MISSPECIFIED_B15_COMMON_SUPPORT_CI95` / registry `h1.no_repeat.misspecified.b15.rescue_A_minus_B`: value=[0, 0.015000]; denominator=200
result `H4_H2_HARM_MISSPECIFIED_B9_COMMON_SUPPORT_CI95` / registry `h2.no_repeat.misspecified.b9.harm_C_minus_A`: value=[-0.022222, 0.080056]; denominator=450
result `H4_H2_A_MINUS_B_MISDIAGNOSIS_MISSPECIFIED_B9_COMMON_SUPPORT_CI95` / registry `h2.no_repeat.misspecified.b9.no_harm_A_minus_B`: value=[0.068889, 0.160000]; denominator=450
<!-- END YAU MACHINE AUDIT -->

Source artifact: `metric_registry.json` (`sha256:3493b2687cecaa91b0ef7ae25f0dacdaa3ea05a5405c79a4321814d5fa092a7c`). Publication PNG bytes were copied only after contract hash verification.

### Selected verified figures

::: {.figure-grid}

![C-state misdiagnosis across item budgets](generated/fig-c-probe-harm-png-e5e22d30fb2c.png)

![LLM-persona manipulation checks](generated/fig-manipulation-checks-png-240a29d8ba3e.png)

![Matched-to-misspecified contrast degradation](generated/fig-matched-vs-misspecified-png-39c4270e4169.png)

![P-state rescue across item budgets](generated/fig-p-rescue-png-c36a76849139.png)

:::
<!-- END PAPER GENERATED RESULTS -->

### Discussion

<!-- BEGIN PAPER GENERATED DISCUSSION -->
**H1.** The frozen P-state rescue criterion was partially supported. Some required evidence favored rescue, but the full conjunction did not pass; the paper therefore makes no full-support claim.

**H2.** The criterion was not supported. The fixed-quota harm contrast remained positive, but the belief-triggered arm failed the no-harm requirement relative to the local-only arm; this is not a reversal of the fixed-quota pattern.

**H3.** The subordinate adaptive sanity check was supported. It remains secondary because adaptive-testing efficiency is established prior art.

**H4.** Under the frozen misspecification stress, both predicted directions persisted. This result describes sensitivity to one declared perturbation family and cannot establish behavioral realism.

**H5.** The LLM-persona hypothesis was excluded pre-outcome because its machine mapping gate was unavailable. It was not evaluated and is not reported as a negative finding.

These findings remain limited to simulated, finite-budget diagnosis in the tested catalog. They do not establish structural non-identifiability, human validity, learning gains, or educational efficacy.
<!-- END PAPER GENERATED DISCUSSION -->

## 6. Responsibility, Limitations, and Reproducibility

The project uses three responsibility layers. The student set direction and
requirements and defined the pre-outcome quality and honesty gates documented in
[`decision_log.md`](decision_log.md). AI systems proposed designs, implemented
automation, executed the simulation and analysis, and drafted the current
interpretation and prose under those gates. Commits, an experiment tag, hashes, seeds,
tests, provenance envelopes, and isolation attestations form the audit layer. Final
interpretation and any submission decision remain pending student review.

The main limitations are explicit: simulated learners are not humans; matched
simulation is partly circular; the misspecification family is not exhaustive; the
truth state is static; evidence is binary and multiple-choice-heavy; only a subset of
the 135-node graph is open; sparse pools may repeat; held-out Brier is internal; and
the work comes from one project with extensive AI assistance. Prior
LLM-simulated-student work motivates explicit validation and limitation analysis
[@lu-wang2024; @liu2024; @jin2025; @wu2025; @scarlatos2025; @scarlatos2026]. In this
project, H5 is secondary and manipulation-gated; that status is an internal design fact.

The analysis plan, canonical config, seed derivation, source hashes, and isolation
rules are repository-bound. Raw confirmatory artifacts must remain in the isolated
simulation store and cannot update real profiles or logs. This draft does not publish,
push, submit, or mint a DOI.

## References

1. F. M. Lord, "Robbins-Monro Procedures for Tailored Testing," *Educational and
   Psychological Measurement*, 31(1), 3-31, 1971.
   [doi:10.1177/001316447103100101](https://doi.org/10.1177/001316447103100101).
2. D. J. Weiss, "Improving Measurement Quality and Efficiency with Adaptive
   Testing," *Applied Psychological Measurement*, 6(4), 473-492, 1982.
   [doi:10.1177/014662168200600408](https://doi.org/10.1177/014662168200600408).
3. D. J. Weiss and G. G. Kingsbury, "Application of Computerized Adaptive Testing
   to Educational Problems," *Journal of Educational Measurement*, 21(4),
   361-375, 1984. [doi:10.1111/j.1745-3984.1984.tb01040.x](https://doi.org/10.1111/j.1745-3984.1984.tb01040.x).
4. J. R. Barrada et al., "A Method for the Comparison of Item Selection Rules in
   Computerized Adaptive Testing," *Applied Psychological Measurement*, 34(6),
   438-452, 2010. [doi:10.1177/0146621610370152](https://doi.org/10.1177/0146621610370152).
5. K. C. T. Han, "Conducting simulation studies for computerized adaptive testing
   using SimulCAT: an instructional piece," *Journal of Educational Evaluation for
   Health Professions*, 15, 20, 2018. [doi:10.3352/jeehp.2018.15.20](https://doi.org/10.3352/jeehp.2018.15.20).
6. A. T. Corbett and J. R. Anderson, "Knowledge tracing: Modeling the acquisition
   of procedural knowledge," *User Modeling and User-Adapted Interaction*, 4,
   253-278, 1994. [doi:10.1007/BF01099821](https://doi.org/10.1007/BF01099821).
7. R. Pelanek, "Bayesian knowledge tracing, logistic models, and beyond," *User
   Modeling and User-Adapted Interaction*, 27, 313-350, 2017.
   [doi:10.1007/s11257-017-9193-2](https://doi.org/10.1007/s11257-017-9193-2).
8. B. Tabibian et al., "Enhancing human learning via spaced repetition
   optimization," *PNAS*, 116(10), 3988-3993, 2019.
   [doi:10.1073/pnas.1815156116](https://doi.org/10.1073/pnas.1815156116).
9. J. Ye, J. Su, and Y. Cao, "A Stochastic Shortest Path Algorithm for Optimizing
   Spaced Repetition Scheduling," *KDD '22*, 4381-4390, 2022.
   [doi:10.1145/3534678.3539081](https://doi.org/10.1145/3534678.3539081).
10. Open Spaced Repetition contributors, *Free Spaced Repetition Scheduler*,
    [software repository](https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler).
11. Anki contributors, *Anki Manual: FSRS*,
    [software documentation](https://docs.ankiweb.net/deck-options.html#fsrs).
12. X. Lu and X. Wang, "Generative Students," *ACM Learning at Scale*, 16-27,
    2024. [doi:10.1145/3657604.3662031](https://doi.org/10.1145/3657604.3662031).
13. Z. Liu et al., "Personality-aware Student Simulation for Conversational
    Intelligent Tutoring Systems," *EMNLP 2024*, 626-642.
    [ACL Anthology](https://aclanthology.org/2024.emnlp-main.37/).
14. H. Jin et al., "TeachTune," *CHI 2025*, 1-28.
    [doi:10.1145/3706598.3714054](https://doi.org/10.1145/3706598.3714054).
15. T. Wu et al., "Embracing Imperfection," *ACL 2025*, 9887-9908.
    [ACL Anthology](https://aclanthology.org/2025.acl-long.488/).
16. A. Scarlatos et al., "SMART," *EMNLP 2025*, 25071-25094.
    [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1274/).
17. A. Scarlatos et al., "Simulated Students in Tutoring Dialogues: Substance or
    Illusion?" *ACL 2026*, 42349-42385.
    [ACL Anthology](https://aclanthology.org/2026.acl-long.1960/).

The same machine-verifiable metadata and source-type labels are stored in
[`references.json`](references.json). No source outside that file is cited.
