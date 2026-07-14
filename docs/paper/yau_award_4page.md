# Diagnosing Why a Student Is Wrong Under Item Budgets

## A Confirmatory Study of Prerequisite-Probe Timing in High-School Chemistry

<!-- BEGIN PAPER GENERATED STATUS -->
**Submission-draft status:** compact four-page source; confirmatory results are not
yet available. Every confirmatory value and H1-H5 decision below is `PENDING` and may
be populated only from [`results_contract.md`](results_contract.md). This document has
not been submitted, published, or externally released.
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
**确证结果：**本区块仅由经验证的论文绑定器写入；程序分析与 H5 完成，
或 H5 依据机器绑定证据在结果前排除后，才会生成数值与判定。
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
**Confirmatory findings:** this block is owned by the validated paper binder and is
populated only after the programmatic and H5 lifecycle is complete or H5 is excluded
pre-outcome with machine-bound evidence.
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
**Status: PENDING.** The following rows are identifiers, not estimates. Their values
may enter this document only after S3 writes them to the delimited machine-generated
block in [`results_contract.md`](results_contract.md).

| Claim | Canonical result identifier | Value |
|---|---|---|
| H1 P convergence | `H1_P_A_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS` | PENDING |
| H1 A-B rescue interval | `H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_ELIGIBLE_STRESS_CI95` | PENDING |
| H1 no-repeat sensitivity | `H1_P_A_MINUS_B_CORRECT_CONVERGENCE_MATCHED_B15_COMMON_SUPPORT_CI95` | PENDING |
| H2 C-A fixed-probe harm | `H2_C_C_MINUS_A_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95` | PENDING |
| H2 A-B no-harm | `H2_C_A_MINUS_B_MISDIAGNOSIS_MATCHED_B9_ELIGIBLE_STRESS_CI95` | PENDING |
| H3 adaptive sanity | `H3_A_MINUS_B_TERMINAL_ACCURACY_MATCHED_B15_FULL_SET_CI95` | PENDING |
| H4 misspecified H1 direction | `H4_H1_RESCUE_MISSPECIFIED_B15_ELIGIBLE_STRESS` | PENDING |
| H4 misspecified H2 direction | `H4_H2_HARM_MISSPECIFIED_B9_ELIGIBLE_STRESS` | PENDING |
| H5 qualifying providers | `H5_QUALIFYING_PROVIDER_COUNT` | PENDING |
| H5 misconception contrast | `H5_MISCONCEPTION_HIT_RATE_CONTRAST_CI95` | PENDING |
<!-- END PAPER GENERATED RESULTS -->

### Discussion

<!-- BEGIN PAPER GENERATED DISCUSSION -->
If H1 or H2 fails, reverses, or has a wide interval, that outcome remains the result.
The production-bound confirmatory result overrides the pilot expectation. H3 cannot
be promoted to the headline, H4 cannot establish realism, and H5 cannot establish
human validity.
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
the work comes from one project with extensive AI assistance. LLM-simulated students
are secondary and manipulation-gated [@lu-wang2024; @liu2024; @jin2025; @wu2025;
@scarlatos2025; @scarlatos2026].

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
