<!-- JOURNAL TITLE OPTIONS
TITLE_CANDIDATE_1: When Accurate Terminal Decisions Do Not Converge: A Budget-Constrained Audit of Probe Selection in Chemistry Diagnosis
TITLE_CANDIDATE_2: From Diagnostic Evidence to Model-Defined Remediation: Auditing Probe Selection and Stopping Under Item Budgets
TITLE_CANDIDATE_3: Budget-Limited Confident Convergence in a Chemistry Diagnostic System: Programmatic Evidence and Response-Channel Stress Tests
SELECTED_TITLE: When Accurate Terminal Decisions Do Not Converge: A Budget-Constrained Audit of Probe Selection in Chemistry Diagnosis
-->

# When Accurate Terminal Decisions Do Not Converge: A Budget-Constrained Audit of Probe Selection in Chemistry Diagnosis

## Structured Abstract

**Background:** Adaptive diagnostic systems must do more than predict whether an
answer will be correct. They must distinguish among model-defined explanations that
lead to different remedial actions, while operating under short item budgets. A
system can rank the correct diagnostic state first without satisfying its own
confidence-based stopping rule, leaving terminal decisions and operational
convergence misaligned.

**Objective:** We audited whether prerequisite probes improve identification of a
model-defined prerequisite-gap state, whether fixed probe insertion creates a
different failure profile for a reasoning-chain state, and whether these effects
persist under a declared response-model perturbation. We also examined, post-hoc,
why terminal accuracy and confident convergence diverged.

**Methods:** The primary evidence was a simulation-only, production-bound experiment
covering 27 chemistry targets, four fixed diagnostic states, three item-selection
arms, 50 paired replicates, and matched and misspecified response generators. The
intention-to-simulate grid contained 32,400 journeys. Arms shared response-noise and
held-out-outcome streams. Confirmatory intervals used 10,000 target-stratified paired
bootstrap resamples. A separately frozen Persona-v2 protocol is defined as an
independent response-channel stress test with 50 persona clusters, two prompt
conditions, and provider as a repeated measure; no Persona-v2 outcome enters this
version. A resource-constrained prescription illustration is likewise reserved for
machine-bound insertion.

**Results:** At 15 items on the 23-target eligible support, the adaptive arm's
prerequisite-gap confident-convergence rate was 12.9% (148/1,150), compared with 0.8%
(9/1,150) for the local-only ladder; the paired difference was 12.1 percentage points
(95% CI 10.3 to 13.8). Because the adaptive rate remained below the frozen 50%
criterion, H1 was partially supported. At nine items, fixed probe insertion raised
reasoning-chain misdiagnosis relative to the adaptive arm by 5.7 points (95% CI 2.3
to 9.0), but the adaptive arm itself exceeded the local-only arm by 9.7 points (95%
CI 7.2 to 12.3), violating the frozen no-harm margin; H2 was not supported. The
adaptive sanity check and the declared misspecification direction check were
supported. On identical 27-target support, terminal prerequisite-state accuracy was
83.8% while correct confident convergence was 12.4%. On identical 23-target support,
the corresponding values were 90.7% and 12.9%. The selector-stopping mismatch is a
post-hoc interpretation and does not revise the frozen decisions.

**Conclusions:** Prerequisite probes improved one narrow convergence contrast but did
not make the model-defined state reliably identifiable at the frozen budget, and the
adaptive policy did not meet its no-harm requirement for the reasoning-chain state.
Terminal argmax accuracy substantially overstated operational convergence. The study
supports auditing selection and stopping as separate components before diagnostic
beliefs are routed to remediation. It does not estimate learning gains, does not
establish educational efficacy, and does not establish human behavioral validity.

## Keywords

adaptive diagnosis; computerized adaptive testing; chemistry education; Bayesian
learner model; simulation audit; stopping rule; prerequisite probe; model-defined
remediation

## 1. Introduction

Adaptive educational systems often compress a sequence of responses into a mastery
score and then use that score to choose the next item or learning resource. That loop
is useful when the only decision is whether to continue practice. It is less adequate
when the downstream action depends on *which model-defined condition* produced the
same observed error. A low score may route a learner to a prerequisite explanation,
a worked reasoning bridge, or a direct introduction to the target concept. If the
diagnostic distinction is weak, a confident-looking posterior can spend a limited
learning budget on the wrong type of resource.

This paper studies that measurement problem inside a working Shanghai high-school
chemistry diagnostic and recommendation system. The system represents a target
concept using four model-defined diagnostic states: mastered (M), prerequisite gap
(P), reasoning-chain instability (C), and unlearned target knowledge (U). The labels
are operational hypotheses in an engineered model. They are not clinical categories,
psychometric traits established in people, or observations of human participants.
Their value depends on whether the response channel can distinguish them within the
item budget used by the product and whether a state-conditioned downstream action is
less wasteful than a state-agnostic one.

The motivating tension is straightforward. For local target questions, the
production likelihoods assigned to P and U differ by only 0.10. With arbitrarily many
independent observations under the model, that gap can identify the states. With 9,
15, or 25 questions, the posterior may rank P first without crossing the production
confidence and gap thresholds. A prerequisite question creates a different response
channel that separates P from U more strongly, but the same channel can blur C with
other states. The design question is therefore not whether prerequisite questions
are informative in general. It is when a selector should expose that channel and
whether the stopping rule recognizes the evidence it accumulates.

The study was developed in response to known instrument limitations, including
multiple-choice-heavy evidence, sparse multi-step and media-dependent coverage, and
uncertain mapping from observed error to downstream remediation. These limitations
make a large efficacy claim premature. They also create a legitimate audit question:
under a fixed model, catalog, and response budget, where does state recovery fail and
how does that failure constrain later decisions? We call this a **budget-limited
confident-convergence failure**. The phrase is deliberately finite and operational.
It does not assert structural or asymptotic impossibility.

The primary programmatic experiment addresses four frozen hypotheses. H1 asks whether
belief-triggered expected information gain (EIG) improves correct confident
convergence for P relative to a local-only ladder while also reaching a minimum
absolute rate. H2 asks whether a fixed prerequisite quota harms C relative to the
adaptive policy and whether the adaptive policy stays within a no-harm margin relative
to the local-only ladder. H3 is a subordinate adaptive-selection sanity check. H4
tests whether the predicted H1 and H2 directions persist under one frozen family of
generator misspecification. The decisions, supports, exclusions, seeds, and interval
procedures were fixed before the confirmatory run.

Two later components are separated from that primary evidence. Persona v2 tests
whether multiple language-model providers follow controlled response policies and
whether blind responses are stable across providers. It is an independent
response-channel stress test, not an enlargement of the programmatic sample. The P2
component will illustrate how a diagnostic posterior can be converted into
model-defined mismatched or uncovered minutes within a small trusted resource library.
Neither component is allowed to alter H1-H4.

The research questions are:

1. Does belief-triggered prerequisite probing improve P-state correct confident
   convergence over a local-only fixed ladder at the frozen item budget?
2. Does fixed-quota prerequisite probing raise C-state misdiagnosis relative to the
   adaptive arm, while the adaptive arm remains within the frozen no-harm margin
   relative to the local-only arm?
3. Do adaptive terminal accuracy and convergence-time advantages appear on the full
   target set, and do the predicted directions persist under the declared generator
   perturbation?
4. On the same target support, how large is the gap between a correct terminal argmax
   and a correct confidence-based stop, and what selector behavior is associated with
   that gap?

The contributions are correspondingly bounded. First, we provide a production-bound,
pre-data comparison of a belief-triggered selector, a local-only ladder, and a fixed
prerequisite quota. Second, we preserve paired random streams and co-report
with-replacement stress and no-repeat common-support estimates rather than choosing
the more favorable view. Third, we separate terminal classification from confident
convergence and show that they can lead to very different assessments of readiness.
Fourth, we expose the dependence of the findings on one declared response generator
and on a narrow chemistry catalog. Finally, we define empty, machine-replaceable
result slots for the dual-condition Persona-v2 study and the illustrative prescription
analysis so that later evidence cannot be silently narrated into the manuscript.

The claim boundary is central: this is a simulation-only audit of model-defined
diagnostic states. It does not estimate learning gains, does not establish educational
efficacy, does not establish human behavioral validity, and does not claim that the
tested target set represents an entire chemistry curriculum.

## 2. Related Work

### 2.1 Adaptive testing and simulation

Computerized adaptive testing has a long history of using current evidence to select
more informative items. Robbins-Monro formulations, early work on measurement
efficiency, and educational applications established the basic logic well before the
present system [@lord1971; @weiss1982; @weiss-kingsbury1984]. Methodological work also
shows why simulations are useful for comparing item-selection rules when the latent
condition and response process are controlled [@barrada2010; @han2018]. The present
study does not offer EIG as a new adaptive-testing algorithm. It uses an existing
selector as an object of audit and asks whether the selector's probe choices align
with a separately implemented confidence-based stopping rule.

Simulation can make failure modes observable because the generating state is known.
It can also create circular evidence when the same likelihood family drives both the
generator and inference. We address, but cannot eliminate, that risk by separating a
matched generator from a pre-specified misspecified generator and by reporting the
change between them. This design tests sensitivity to one perturbation family; it
does not approximate the full range of response behavior found outside the model.

### 2.2 Learner models, confidence, and longitudinal scope

Bayesian knowledge tracing and related learner models estimate changing knowledge
from observed practice [@corbett-anderson1994; @pelanek2017]. Our primary experiment
uses Bayesian updating but freezes the generating diagnostic state within each
journey. It therefore evaluates state recovery during one diagnostic session, not
knowledge acquisition, forgetting, transfer, or retention. The distinction matters:
a model may update a posterior coherently without its state labels corresponding to
stable constructs outside the simulator.

The product also contains an FSRS-inspired scheduling projection. Research on spaced
practice and scheduling optimization addresses a different decision horizon
[@tabibian2019; @ye2022], while the operational implementation draws on openly
documented scheduling software [@fsrs-repository; @anki-fsrs-manual]. None of those
sources validates the diagnostic likelihoods or stopping thresholds audited here.
Memory scheduling is therefore outside the primary estimands.

### 2.3 Language-model response simulation

Recent studies use language models to instantiate diverse response profiles, review
question items, or test tutoring agents [@lu-wang2024; @liu2024; @jin2025; @wu2025;
@scarlatos2025]. This work demonstrates a potentially useful evaluation instrument,
but it also makes construct boundaries essential. Surface plausibility and prompt
compliance do not show that a response process matches human behavior. Direct tests
of simulated tutoring dialogue have reinforced the distinction between apparent
student-like language and substantive behavioral validity [@scarlatos2026].

Persona v2 therefore has two conditions with different purposes. Controlled
manipulation asks whether a provider can follow a stated error policy. Blind response
robustness removes target labels and tests cross-provider consistency, technical
failure, abstention, and repeat stability. Neither condition assigns independent
sample status to provider calls, and neither produces an authenticity score. Any
cross-model LLM adjudication is an automated descriptive coding procedure, not a
human reference standard.

### 2.4 From diagnostic output to remediation

A diagnostic posterior becomes educationally consequential only when it changes an
action. In the current system, P, C, and U would route to different resource roles.
This makes terminal state accuracy an incomplete endpoint: two policies with similar
argmax accuracy may differ in whether they stop, abstain, or allocate limited minutes
to a mismatched resource. The P2 illustration is designed to expose that translation
without claiming treatment effects. It maps posterior-conditioned demand onto a
fixed trusted segment library and reports model-defined mismatched and uncovered
minutes. Because the library is small and uneven, the illustration is a resource
audit rather than an optimizer benchmark or an efficacy experiment.

## 3. Methods

### 3.1 System boundary and trusted catalog

The audited system serves Shanghai high-school chemistry. Its knowledge graph has 135
nodes, but the deterministic confirmatory catalog opened only 27 targets that passed
service, scoring, mapping, media, and content-family gates. A mechanical prerequisite
census admitted 23 of those targets to H1 and H2. Four targets lacked the required
prerequisite-family structure and remained in the full descriptive and H3 grid but
not in H1 or H2.

Questions were separated at both item and content-family levels. Two local families
per target were reserved for held-out scoring and removed from all administered
pools. Held-out responses were never used for posterior updates. The programmatic
runner imported the production mastery updater and selector, so Bayesian updates,
EIG scores, eligibility rules, and stopping behavior were the same implementations
used by the local product. The experiment did not alter the official question bank or
the frozen H1-H4 decisions.

### 3.2 Model-defined diagnostic states and response channels

For a target-local item of difficulty \(d\), the production model assigns the
state-wise probability of a correct response as

\[
P(Y=1\mid M,P,C,U)=(0.90,\ \gamma+0.10,\ 0.70-0.20d,\ \gamma),
\]

where \(\gamma=0.25\) for multiple-choice items and \(0.03\) for numeric items. The
local P and U channels therefore differ by 0.10. A prerequisite item uses

\[
P(Y_{pre}=1\mid M,P,C,U)=(0.90,\ \gamma,\ 0.80,\ 0.75).
\]

This second vector creates more separation between P and U, while making a correct
prerequisite response compatible with several other states. The posterior starts
uniform within each journey and updates after each observation. A terminal decision
is the posterior argmax in the frozen M, P, C, U order. Correct confident convergence
requires both the correct argmax and the production confidence and posterior-gap
conditions. Exhausting the item budget is distinct from satisfying those conditions.

The prerequisite likelihood vector updates the target-state posterior directly. We
therefore call the comparison a *probe-channel comparison*. It is not evidence that
information propagated through the knowledge graph.

### 3.3 Programmatic design and item-selection arms

The intention-to-simulate grid crossed 27 targets, four fixed truth states, three
arms, 50 replicates, and two response-generator conditions, yielding 32,400 planned
journeys. A maximum 25-item run supplied frozen views at budgets 9, 15, and 25. When
confidence was reached before a nominal view, the posterior and convergence time were
carried forward without increasing the administered count.

The three arms were:

- **Arm A, belief-triggered EIG.** At each decision, all eligible local and
  prerequisite candidates competed under the production EIG selector. The production
  stopping rule was evaluated after every posterior update.
- **Arm B, local-only ladder.** Requested difficulties cycled through 0.25, 0.50,
  0.75, and 1.00, with only local items eligible.
- **Arm C, fixed prerequisite quota.** Arm C used the same ladder but replaced every
  third position with a prerequisite item through position 24.

Fixed-arm ties were resolved deterministically by absolute difficulty distance,
family identifier, and item identifier. After all unseen eligible families were
exhausted, the frozen family-epoch replenishment rule permitted reuse. Repetition was
measured and disclosed rather than treated as independent evidence.

The confirmatory hypotheses and decision branches were frozen before collection.
H1 required both an Arm-A correct-convergence rate of at least 0.50 at budget 15 and a
strictly positive lower confidence bound for A minus B. H2 required a strictly
positive lower bound for C minus A C-state misdiagnosis at budget 9 and an upper bound
below the +0.05 no-harm margin for A minus B. H3 required the adaptive arm not to lose
terminal accuracy and not to take longer to converge than B. H4 required the H1
rescue and H2 fixed-quota harm point directions to remain positive under the frozen
misspecified generator.

### 3.4 Response generators, pairing, and sparse-pool estimands

The matched generator sampled from the production probabilities. The misspecified
generator instead drew item-level slip from a uniform 0.05 to 0.20 distribution,
guess from a uniform 0.15 to 0.35 distribution, and one replicate-level
ability offset from a zero-centered normal distribution with standard deviation 0.05,
clipped to the interval -0.10 to 0.10. Inference retained its production constants in
both conditions. Generator parameters were not passed to the posterior updater.

Arms were paired within target, truth state, generator condition, and replicate. They
shared response-noise streams. The two held-out family outcomes were generated from
seeds without an arm component, ensuring identical held-out draws for A, B, and C.
This pairing reduced irrelevant Monte Carlo variation and permitted direct arm
contrasts.

H1 and H2 used two pre-declared population views. The primary empirical stress view
retained all 23 eligible targets and allowed the frozen family-replenishment behavior.
The sensitivity view restricted analysis to the budget-specific three-arm common
support on which every arm could complete without family repetition. That support
contained nine targets at budget 9, four at budget 15, and one at budget 25. Both
views were required regardless of direction.

Arm C could not construct the fixed prerequisite schedule for four targets. This
produced 1,600 structural failures in the intended grid. Those failures were retained
in the lifecycle and excluded uniformly from estimands under the frozen rule; they
were not reclassified after results were known.

### 3.5 Outcomes, support identity, and statistical analysis

At each nominal budget, outcomes included correct confident convergence, converged
misdiagnosis, terminal argmax accuracy, the four-by-four terminal and decision
confusion matrices, convergence time, held-out Brier score, administered count, exact
item repeats, and family repeats. Held-out Brier under the matched generator is an
internal calibration diagnostic only.

Programmatic intervals used exactly 10,000 target-stratified paired-replicate
percentile bootstrap resamples. Replicate identifiers were resampled with replacement
within each target, while all arm observations for a sampled replicate remained
paired. Contrasts were computed inside each resample. Reported quantities include the
numerator, denominator, target count, weighting rule, and 95% interval.

Support identity was checked before interpreting any pair. The 27-target full-set
terminal and convergence rates are compared only with each other. The 23-target
eligible-set rates are likewise paired only with the same eligible support. The
original artifact registry encodes support through metric names, target counts,
denominators, and predicates rather than a dedicated target-set hash; this is a
portability limitation, and a dedicated support hash is required before later
automated insertion.

The terminal-versus-convergence comparison and selector composition analysis were not
confirmatory hypotheses. They are labeled post-hoc throughout. They may suggest a
selector-stopping mismatch, but they cannot change the H1-H4 decisions.

### 3.6 Persona-v2 dual-condition protocol

Persona v2 is frozen separately from the programmatic study. Its independent cluster
is `persona_id`, represented in prose as 50 persona clusters. Each cluster contains a
paired deficit row and control row. Provider and response condition are repeated
observations; the analysis treats provider as a repeated measure. Calls, retries,
individual item answers, and model outputs are not independent sample units.

The grid crosses 25 failure anchors with low and high response-noise settings. The
controlled manipulation condition exposes a general observable error policy and uses
four calibration items. It estimates paired deficit-control differences in
correctness, error rate, valid-response rate, and abstention. The blind response
robustness condition removes target misconception labels and target options, reuses
the calibration items, and adds up to 21 family-distinct diagnostic items. It reports
terminal-answer agreement, technical and schema failure, abstention, and repeat
stability. Every response uses a text-only modality.

Before provider observation, exact item/failure/option consensus was achieved for
only 6 of 100 calibration rows. Because that coverage was below the frozen minimum,
target-misconception hit rate was removed from the confirmatory Persona-v2 analysis.
The six mapped rows may appear only as a sparse descriptive check with an explicit
denominator and support hash. This degradation is an input fact, not a provider
outcome.

The smoke pilot uses two providers and five clusters, is stored separately, and is
excluded from every main estimate. Main collection attempts six frozen providers.
Intervals use a 10,000-resample persona-cluster bootstrap that preserves all repeated
provider and condition observations for a sampled cluster. Cross-model LLM
adjudication receives blind public prompts and candidate outputs without target
labels, correct options, or mapping status. Allowed labels include unknown and
insufficient evidence. Agreement and disagreement examples are descriptive; no
realism or authenticity score is formed.

### 3.7 Illustrative prescription analysis

The prescription component is deliberately narrow. The audited default resource
library contains 13 nodes and 68 trusted segments, and each exposed node has only one
or two distinct physical video parts. This supply is a hard constraint, not a promise
of broad remediation coverage.

P2 will compare a truth-state oracle prescription with prescriptions opened from
diagnostic posteriors under a single budget setting. A greedy heuristic will select
from the fixed library. Planned outcomes are model-defined mismatched minutes and
uncovered minutes, reported separately. The exercise is illustrative: it does not
estimate a treatment effect, claim that a segment teaches the intended concept, or
benchmark exact optimization. Structural inability to construct a requested arm will
be reported rather than filled with invented values.

### 3.8 AI-assisted research workflow

Generative AI systems supported study design critique, software implementation and
testing, simulation execution, data analysis, figure generation, drafting and
revising prose, and adversarial manuscript review. AI-generated suggestions did not
enter the confirmatory result surface by prose alone. H1-H4 claims were checked
against frozen analysis rules, immutable hashes, machine-generated metric records,
and targeted regression tests. Provider outputs in Persona v2 are research objects
and are distinct from AI assistance used to conduct the work.

The human author set project direction and claim boundaries and retains responsibility
for source review, result interpretation, journal-policy compliance, and the final
text. AI systems were not treated as authors, participants, or independent reference
raters. No credential, private prompt, or machine-local location is part of the
manuscript evidence surface.

## 4. Results

### 4.1 Execution integrity and analysis populations

The programmatic grid contained 32,400 intended journeys. Of these, 30,800 were valid
for at least one declared analysis view. The remaining 1,600 were Arm-C structural
failures concentrated in four targets whose prerequisite pools could not satisfy the
fixed-quota schedule. The 27-target full set was used for H3 and descriptive views;
the mechanically eligible 23-target set was used for the H1/H2 empirical stress
estimands.

All results below are simulated finite-budget estimates. Matched and misspecified
conditions remain separate. H1-H4 decisions are reproduced from the frozen decision
rules rather than reassigned from the apparent favorability of a point estimate.

### 4.2 Prerequisite-gap rescue and reasoning-chain harm

At budget 15 on the 23-target eligible stress support, Arm A achieved P-state correct
confident convergence in 148 of 1,150 paired cases (12.9%, 95% CI 11.2% to 14.5%).
Arm B achieved 9 of 1,150 (0.8%, 95% CI 0.3% to 1.3%). The paired A-minus-B contrast
was 139 of 1,150, or 12.1 percentage points (95% CI 10.3 to 13.8). The direction and
interval favored adaptive probing, but the absolute Arm-A rate did not reach the
frozen 50% threshold. H1 was partially supported.

The no-repeat common-support sensitivity was much weaker. Across four targets and
200 paired cases at budget 15, Arm A converged correctly once (0.5%, 95% CI 0.0% to
1.5%) and Arm B never did; the paired contrast was 0.5 points (95% CI 0.0 to 1.5).
The effect was therefore not stable to the jointly changed no-repeat and common-support
estimand. Because that sensitivity also restricts the target set, its difference from
the broad estimate cannot be attributed to repetition alone.

At budget 9 on the 23-target support, C-state misdiagnosis was 37.4% for Arm A
(430/1,150), 27.7% for Arm B (318/1,150), and 43.0% for Arm C (495/1,150). The fixed
quota increased misdiagnosis relative to A by 5.7 points (65/1,150; 95% CI 2.3 to
9.0). However, A exceeded B by 9.7 points (112/1,150; 95% CI 7.2 to 12.3), and the
upper interval bound was well above the +5-point no-harm margin. H2 was not supported.

On the nine-target, no-repeat common support, the corresponding rates were 41.8% for
A (188/450), 28.4% for B (128/450), and 43.8% for C (197/450). C minus A was 2.0
points (95% CI -3.3 to 7.6), whereas A minus B was 13.3 points (95% CI 9.1 to 17.6).
Thus, the sensitivity analysis weakened the fixed-quota contrast but retained the
adaptive arm's disadvantage relative to the local ladder.

![Figure 1. P-state correct convergence across item budgets. The figure is reused from the verified programmatic artifact and does not include Persona-v2 evidence.](generated/fig-p-rescue-png-c36a76849139.png)

![Figure 2. C-state misdiagnosis across item budgets. Arm labels refer only to the programmatic study.](generated/fig-c-probe-harm-png-e5e22d30fb2c.png)

### 4.3 Adaptive sanity check and misspecification

On the full 27-target matched set at budget 15, Arm A exceeded Arm B in terminal
accuracy by 10.0 percentage points (542/5,400 paired contrast; 95% CI 8.9 to 11.2).
Median time to confidence was eight items for A (95% CI 8 to 8) and 11 items for B
(95% CI 10 to 11). Under the frozen branch, H3 was supported. This result is a sanity
check for the implementation, not evidence that adaptive testing is novel.

Under the misspecified generator on the 23-target eligible stress support, the P-state
A-minus-B rescue remained 12.1 points (139/1,150; 95% CI 10.2 to 14.0). The C-state
C-minus-A harm contrast was 7.0 points (80/1,150; 95% CI 3.7 to 10.3), and A minus B
was 8.1 points (93/1,150; 95% CI 5.5 to 10.7). The matched-minus-misspecified
degradation was 0.0 points for H1 (95% CI -2.6 to 2.5), -1.3 points for the fixed-quota
harm contrast (95% CI -6.0 to 3.5), and 1.7 points for A minus B (95% CI -2.1 to 5.3).
Both predicted directions therefore persisted under this perturbation, and H4 was
supported. The intervals do not show robustness to untested generator families.

![Figure 3. Matched and misspecified contrast estimates. The perturbation is one declared synthetic sensitivity condition.](generated/fig-matched-vs-misspecified-png-39c4270e4169.png)

### 4.4 Same-support terminal and convergence estimates

Terminal argmax accuracy and correct confident convergence answered different
questions. The former asked whether the correct state had the largest posterior at
the end of the budget. The latter additionally required the production confidence
and posterior-gap thresholds. Table 1 preserves support identity for both views.

**Table 1. Same-support terminal and confident-convergence estimates.**

| Support and view | Correct terminal P decision | Correct confident P convergence | Difference |
|---|---:|---:|---:|
| Full 27-target support, Arm A, budget 15 | 1,131/1,350 (83.8%) | 168/1,350 (12.4%) | 963/1,350 (71.3 points; arithmetic only) |
| Eligible 23-target support, Arm A, budget 15 | 1,043/1,150 (90.7%) | 148/1,150 (12.9%) | 895/1,150 (77.8 points; 95% CI 75.7 to 79.8) |

The full-set difference has no dedicated bootstrap interval in the retained registry
and is therefore reported as arithmetic only. The eligible-set difference has a
machine-bound paired interval. The invalid cross-support pairing of 83.8% with 12.9%
is not used.

The eligible-set selector composition provides a post-hoc mechanism clue. Arm A
administered a mean of 1.43 direct items and 11.95 prerequisite items by budget 15;
the mean prerequisite share was 83.8% (95% CI 82.6% to 84.9%). EIG was therefore
selecting heavily from the prerequisite channel. Yet a correct terminal ordering
usually did not cross the stopping rule's confidence and gap requirements. We refer
to this pattern as a **selector-stopping mismatch**. It is an exploratory
interpretation, not a randomized mechanism test, and it does not revise H1.

### 4.5 Persona-v2 result slot

<!-- BEGIN RESULT SLOT: PERSONA_V2_DUAL -->
No outcome estimate is reported in this slot. Insert only machine-bound main-analysis estimates after lifecycle, clustering, leakage, and pilot-exclusion checks pass.
<!-- END RESULT SLOT: PERSONA_V2_DUAL -->

The methods above define how controlled compliance, blind agreement, technical
failure, abstention, and repeat stability will be reported. Until a bound analysis
artifact is inserted, the manuscript makes no provider comparison and displays no
Persona-v2 figure.

### 4.6 Illustrative prescription result slot

<!-- BEGIN RESULT SLOT: P2_ILLUSTRATIVE -->
No outcome estimate is reported in this slot. Insert only machine-bound illustrative estimates with supply limits and structural failures preserved.
<!-- END RESULT SLOT: P2_ILLUSTRATIVE -->

The absence of an estimate is not a null finding. It prevents the known 13-node,
68-segment resource constraint from being converted into an unsupported statement
about mismatched or uncovered minutes.

## 5. Discussion

### 5.1 Main interpretation

The primary experiment produced a mixed but informative result. Belief-triggered
probing improved correct P convergence relative to a local-only ladder on the broad
eligible stress support, yet the absolute convergence rate remained only 12.9% at 15
items and fell to 0.5% on the small no-repeat common support. This is not a successful
identification result masked by a strict criterion. It is evidence that the tested
response channels and stopping thresholds did not reliably turn a directional
advantage into an operationally confident diagnosis.

The C-state result is equally important. Fixed prerequisite insertion was worse than
the adaptive policy on the primary support, matching the motivating direction. But
the adaptive policy was itself materially worse than the local-only ladder, and the
pre-specified no-harm condition failed. A narrative focused only on C minus A would
therefore be incomplete. The three-arm design shows that avoiding a rigid quota did
not imply that the adaptive policy was harmless.

H3 confirmed that the adaptive path improved aggregate terminal accuracy and reached
confidence sooner than the local ladder across the full state grid. That advantage
coexisted with poor P correct convergence. Aggregate adaptive efficiency can therefore
hide a state-specific failure that matters for remediation routing. H4 further showed
that the predicted rescue and fixed-quota harm directions survived one generator
perturbation, but the design provides no warrant to generalize beyond that family.

### 5.2 Why terminal accuracy is insufficient

The largest descriptive finding was not an arm contrast but the same-support gap
between terminal state ranking and confident convergence. On both the full and
eligible supports, Arm A usually placed P first by the end of budget 15. It rarely
satisfied the confidence-based stop. A product surface that displays only the argmax
could therefore present a decisive-looking label where the production rule itself
would continue gathering evidence or abstain.

The post-hoc selector composition makes this gap plausible. EIG repeatedly chose the
prerequisite response channel, so the issue was not simply that the selector ignored
prerequisite evidence. Selection optimized expected posterior information one item at
a time; stopping required an absolute confidence and separation condition. Those
objectives need not agree under sparse, overlapping likelihoods. The result motivates
a direct audit of selector and stopper compatibility, not a claim that EIG is
generally defective.

Operationally, three outputs should remain distinct: the terminal argmax, whether the
confidence rule passed, and the action taken when it did not. A remediation system can
use abstention, ask a different response type, or route to a low-risk general review
when confidence remains insufficient. The present study does not establish which
fallback improves outcomes, but it identifies the cases where a fallback is needed
under the model.

### 5.3 Implications for response-channel stress testing

Programmatic simulation supplies known generating states and exact pairing, which
makes it the strongest evidence in this paper for internal behavior. It cannot answer
whether language models or people produce the assumed response patterns. Persona v2
addresses only the narrower question of response-channel robustness: can different
providers follow a controlled behavior constraint, and do blind outputs remain stable
when target labels are hidden?

The two-condition separation prevents prompt compliance from being mistaken for
construct validity. Strong controlled differences may show that an instruction was
followed; they would not show that the induced errors resemble those of students.
Blind cross-provider agreement may reveal a robust model response pattern; it would
not establish that the pattern is educationally correct. Clustering by persona rather
than by call prevents retries and repeated provider observations from inflating the
sample size.

The pre-observation mapping degradation also changes what can be claimed. With exact
consensus for only six calibration rows, a target-specific misconception hit rate
would be dominated by ambiguous mappings. Removing it from the confirmatory family
before provider observation is more informative than forcing a noisy metric into the
paper. It leaves correctness, abstention, schema validity, terminal agreement, and
repeat stability as defensible response-channel outcomes.

### 5.4 From posterior error to prescription constraints

The downstream prescription illustration is intended to express diagnostic error in
the same currency as a limited learning session: minutes assigned to a role that does
not match the generating state and minutes of required coverage left unserved. This
translation is model-defined. It does not show that watching a selected segment
changes knowledge.

The resource supply sharply limits the exercise. Only 13 nodes and 68 trusted segments
are available in the audited default library, with one or two physical parts per
exposed node. A method that cannot open an Arm-C prescription for a target has a
structural failure, not a zero-regret outcome. Reporting that failure is scientifically
more useful than filling the requested budget with weakly matched content. The
illustrative P2 result slot remains empty until those rules and denominators are
machine bound.

### 5.5 Claim boundaries and use

The results support an internal audit conclusion: selection efficiency, terminal
accuracy, and confidence-based convergence are separable, and a policy that improves
one state contrast can still violate a no-harm condition for another state. They do
not support a claim that a learner was correctly diagnosed, that a remediation was
pedagogically appropriate, or that the system improved achievement.

The intended use of the evidence is engineering triage. It identifies where richer
response channels, better item-family coverage, altered stopping logic, or explicit
abstention deserve evaluation before a larger intervention study. It also supplies a
reproducible negative result: under the tested finite budgets, the P state remained
poorly converged despite an adaptive rescue contrast. This is the precise sense in
which the study documents budget-limited confident-convergence failure.

## 6. Limitations

1. **Simulation-only evidence.** No human participants contributed outcome data. The
   known generating state makes internal error analysis possible but cannot establish
   how people respond, learn, or interpret the diagnostic interface.

2. **Model circularity.** The matched generator uses the production likelihoods that
   inference assumes. The misspecified generator breaks that identity only through
   one frozen slip, guess, and ability-offset family, leaving many plausible response
   processes untested.

3. **Static diagnostic state.** The generating state is fixed within a journey. The
   experiment does not model knowledge acquisition, forgetting, transfer, or
   retention, and its results should not be read as longitudinal evidence.

4. **Text-only and multiple-choice-heavy channels.** The programmatic catalog is
   dominated by binary scored traces, and Persona v2 is explicitly text-only. The
   study does not test diagrams, experimental apparatus, open derivations, or the
   long-chain reasoning formats that motivated part of the instrument audit.

5. **Narrow curricular coverage.** Only 27 of 135 graph nodes enter the full
   programmatic grid, and only 23 satisfy the prerequisite-family gate for H1/H2.
   Important chemistry domains and difficult integrated problems remain outside the
   evidence base.

6. **Sparse pools and structural failures.** Family replenishment permits repeats in
   the broad stress estimand, while no-repeat common supports are small. Arm C also
   fails structurally for four targets, so comparisons are not uniformly available
   across the catalog.

7. **Stopping-rule specificity.** The terminal-versus-convergence gap depends on the
   current posterior confidence and separation thresholds. Other calibrated stopping
   rules may behave differently, and the post-hoc explanation has not been tested by
   a frozen stopping-rule intervention.

8. **No human reference labels for response coding.** Planned Persona-v2 coding uses
   blind cross-model LLM adjudication with unknown and insufficient-evidence options.
   Agreement between automated judges is descriptive and cannot establish semantic
   correctness or behavioral realism.

9. **Independent item responses.** Persona-v2 elicits each response through a frozen
   response-channel protocol rather than modeling persistent memory or learning
   across a session. The independent unit is 50 persona clusters, with providers and
   conditions repeated within cluster; provider calls are not additional units.

10. **Constrained prescription supply.** The illustrative remediation analysis is
    limited to 13 nodes and 68 trusted segments, with uneven role coverage and few
    distinct physical parts. Mismatched and uncovered minutes will describe this
    library and utility definition, not general remediation quality.

## 7. Conclusion

This study audited a chemistry diagnostic system at the point where answer evidence
becomes a model-defined state decision. Belief-triggered prerequisite probing produced
a positive P-state rescue contrast relative to a local-only ladder, but it did not
reach the frozen absolute convergence criterion. Fixed prerequisite insertion harmed
C relative to the adaptive arm, while the adaptive arm also failed its no-harm test
against the local ladder. Aggregate adaptive accuracy and speed improved, and the two
predicted directions persisted under one declared misspecification family.

Most importantly, correct terminal state ranking and correct confident convergence
were far apart on both legitimate supports. That gap, together with the post-hoc
selector composition, motivates treating item selection and stopping as separate
objects of evaluation. A system should not convert a terminal argmax into a precise
remediation label without disclosing whether its own confidence rule passed.

Persona-v2 response-channel results and the P2 prescription illustration are absent
by design until machine-bound artifacts satisfy their respective gates. Their later
insertion can extend the robustness and downstream-cost story, but it cannot revise
the frozen H1-H4 evidence or turn this internal simulation audit into a study of human
learning.

## Declarations

### Ethics and participant involvement

No human participants were enrolled and no personally identifying information was
collected for the experiments reported here. Programmatic responses and language-model
outputs are labeled as simulated evidence. Any future study involving students or
human raters requires a separate ethics, consent, and data-governance determination.

### Data and code availability

The reproducibility package is designed around frozen plans, versioned source,
hash-bound input manifests, deterministic seeds, raw simulated response envelopes,
metric registries, figure artifacts, and regression tests. Release contents and
licensing must be checked against question and video-source rights before any public
distribution. Credentials and private provider payloads are excluded.

### Funding

No external funding is declared for this study.

### Competing interests

The author declares no competing financial interest. The audited system was developed
within the same project, which creates an investigator-interest risk addressed through
pre-observation freezing, negative-result retention, and machine-bound reporting.

### Declaration of Generative AI and AI-assisted Technologies

Generative AI and AI-assisted technologies were used for study-design critique,
software implementation and testing, simulation execution, data analysis, figure
generation, drafting and revising prose, and adversarial manuscript review. Their
outputs were checked against frozen contracts, source artifacts, automated tests, and
the stated claim boundaries. AI systems were not treated as authors, human
participants, or independent reference raters. The human author retains responsibility
for the accuracy, integrity, interpretation, policy compliance, and final wording of
the work. Tool-specific disclosure will be conformed to the selected journal's current
author instructions without implying that one publisher policy applies universally.

## References
