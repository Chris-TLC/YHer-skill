# Decision Log and Authorship Evidence Index

**Status:** documentary index for the manuscript and defense preparation. Programmatic
H1-H4 analysis is complete; H5 was excluded pre-outcome with zero qualifying providers
and no outcome decision. D01-D13 document direction and pre-outcome decisions; they are
not a post-results student sign-off. Final interpretation and any submission decision
remain pending student review. This is not a claim that every implementation detail was
personally written by the student.

## Classification Rules

- **`verbatim`** means the quoted text is reproduced from the named source anchor.
  Only the delimited quotation is claimed to be verbatim.
- **`documented`** means a dated project record attributes the decision or requirement
  to the student/user, but the exact spoken wording is not available in that anchor.
  The summary is not presented as a quotation.
- **`synthesis`** means the entry is an interpretation assembled from two or more
  records. It is explicitly not a quotation and should not be cited as one.

Source anchors use a repository-relative path plus the exact Markdown heading and,
where needed, the numbered item or named clause. Newer sources supersede older ones.

## D01 - 2026-06-25 - Student Experience Requirements

- **Date:** 2026-06-25
- **Classification:** `documented`
- **Decision:** The student experience was constrained to restrained Apple-like UI,
  direct video viewing, dynamically sequenced diagnosis, purpose-specific strategies,
  persistent multi-year memory, and complete readable chemistry questions. These were
  requirements, not findings from the later simulation.
- **Exact source anchor:**
  `../../../PROJECT_HANDOFF/CURRENT_DECISIONS.md`, heading
  `## 2026-06-25 产品体验纠正`, items 1-6.

## D02 - 2026-06-26 to 2026-06-27 - Chemistry First and No Teacher-Imitation Core

- **Date:** 2026-06-26 to 2026-06-27
- **Classification:** `documented`
- **Decision:** The project moved away from a 500-question core and teacher catchphrase
  imitation. Chemistry became the first usable-product target; teacher videos became
  routing resources while AI handled diagnosis, recommendation, verification, profile
  updates, and question quality control. Math and physics expansion was deferred.
- **Exact source anchors:**
  `../../../PROJECT_HANDOFF/CURRENT_DECISIONS.md`, heading
  `## 2026-06-26 晚战略转向`, items 1-4;
  `../../../PROJECT_HANDOFF/CURRENT_DECISIONS.md`, heading
  `## 2026-06-27 当前纠正`, items 2-3; and
  `../../../PROJECT_HANDOFF/HISTORICAL_DETOURS.md`, headings
  `## 弯路 4：模仿老师口癖或金句` and
  `## 弯路 6：多学科同时推进过快`.

## D03 - 2026-07-03 - DOCX-Native Architecture Pivot

- **Date:** 2026-07-03
- **Classification:** `documented`
- **Decision:** The extraction architecture changed from image-first question crops to
  native Word structure where source DOCX existed, retaining PDF/vision processing as
  a fallback. The decision favored recoverable document objects, structured formulas,
  and original image assets over repeated raster inference.
- **Exact source anchor:**
  `../../../PROJECT_HANDOFF/ARCHITECTURE_PIVOT_DOCX_NATIVE_2026-07-03.md`, document
  title `# 架构转向决策:从"视觉裁片"翻转为"Word 原生结构化" 2026-07-03`,
  headings `## 一句话结论`, `## 新架构(五层)`, and `## 旧资产的去向`.

## D04 - 2026-07-06 - Stop Repairing Irrecoverable Source Items

- **Date:** 2026-07-06
- **Classification:** `verbatim`
- **Decision:** The recorded user approval was: **"接受放弃不可修"**. Items whose
  source answer was missing or irrecoverably misattributed were to be excluded rather
  than cosmetically repaired. The quality denominator became the repairable universe,
  with a fail-closed service whitelist protecting learners.
- **Exact source anchor:**
  `../../../PROJECT_HANDOFF/CURRENT_DECISIONS.md`, heading
  `## 2026-07-06 QA 攻坚目标口径与 R5 服务门（项目级）`, items 1-2.

## D05 - 2026-07-07 - Risk-Tiered Use of AI-Generated Questions

- **Date:** 2026-07-07
- **Classification:** `verbatim`
- **Decision:** The recorded choice was **"a+c:分级服务+style-transfer试点"**. Newly
  generated text questions remained in lower-risk practice positions; critical first
  diagnosis and held-out verification remained bound to trusted real questions.
  Style-transfer was tested as a separate, gated supply path.
- **Exact source anchor:**
  `../../../PROJECT_HANDOFF/CURRENT_DECISIONS.md`, heading
  `## 2026-07-07 内化生成三轮验证收口 + Track A 分级服务路线（用户拍板）`,
  item 1.

## D06 - 2026-07-12 - Rapid but Honest Project Closeout

- **Date:** 2026-07-12
- **Classification:** `documented`
- **Decision:** The student prioritized standardized-test and AP work and requested a
  faster engineering closeout. The accepted evidence boundary prohibited fabricated
  users, effects, or scale while permitting explicitly labeled synthetic demonstrations,
  founder QA, and later formative usability work. Publication and effect claims were
  not prerequisites for finishing the local engineering demonstration.
- **Exact source anchor:**
  `../../../PROJECT_HANDOFF/ledger_archive/ledger_2026-07.md`, ledger entry
  `### 2026-07-12 23:21 CST - YHer快速诚实结项与美本CS材料渠道`, fields
  `Requirements`, `Design`, and `Decisions/corrections`.

## D07 - 2026-07-13 - Six Experience Gates for the Demonstration

- **Date:** 2026-07-13
- **Classification:** `documented`
- **Decision:** The implementation brief recorded six student-specified experience
  gates: progressive diagnosis, clear time boxes and transitions, deep causal
  explanation, exam-like variants using real questions, analogies only when needed,
  and restrained UI without emoji or model branding. These requirements were used as
  QA criteria rather than post hoc marketing claims.
- **Exact source anchor:**
  `../../../PROJECT_HANDOFF/codex_briefs/2026-07-12_Demo过夜收官攻坚.md`, heading
  `## 1. 背景与前置必读（只读，不复述）`, clause `用户 6 条体感需求`.

## D08 - 2026-07-13 - Reversibility as the AI Governance Rule

- **Date:** 2026-07-13
- **Classification:** `verbatim`
- **Decision:** The stored user statement is:

  > 你可以暂且抛开之前的一些，你作为 claude code 与 Codex 的一些权限的边界，比如说审核非要你审，或者说什么入库必须由你或者说我来进行一个验证才能入库。我觉得这些你都可以交给 Codex 做……之前是做成这个 skill，好像是刻在工作流里面。你可以把这个先删掉。

  The resulting governance rule delegated reversible implementation and review work
  to AI agents, while retaining backups, line manifests, rollback commands, and later
  audit. Irreversible or external actions remained with the user. This superseded the
  older L0/L1 authorization split; it did not authorize publication or deletion.
- **Exact source anchors:**
  `../../../PROJECT_HANDOFF/CURRENT_DECISIONS.md`, heading
  `## 2026-07-13 治理翻转：L0/L1 废止，改可逆性纪律（用户拍板）`, items 1-4; and
  `../../../PROJECT_HANDOFF/codex_briefs/2026-07-12_Demo过夜收官攻坚.md`, heading
  `## 8. 授权级别（可逆性纪律，2026-07-13 用户拍板）`.

## D09 - 2026-07-13 - Simulated Personas Must Stay Simulated

- **Date:** 2026-07-13
- **Classification:** `documented`
- **Decision:** The student accepted the explicit honesty boundary that AI personas
  must be labeled simulated, may not be represented as humans, and may not be tuned
  toward a preferred result. Any later human participation was limited to formative
  usability evidence unless stronger consent and study procedures were established.
- **Exact source anchors:**
  `/Users/mac/.gstack/projects/Chris-TLC-YHer-skill/ceo-plans/2026-07-13-simulation-eval-paper.md`,
  heading `## 用户已拍板决策（2026-07-13）`, clause `诚实红线（用户已接受）`;
  and `../../../PROJECT_HANDOFF/YHER_PROJECT_OVERVIEW.md`, ledger entry
  `### 2026-07-13 17:11 CST - Codex session`, fields `Requirements` and
  `Decisions/corrections`.

## D10 - 2026-07-13 - Draft First, Publication Only After User Review

- **Date:** 2026-07-13
- **Classification:** `documented`
- **Decision:** The immediate deliverable was reduced to a publication-quality draft.
  Zenodo release and DOI creation were deferred until the student had cooled off,
  reviewed the work, and checked target-venue preprint policies. The timestamp side
  project was also removed from this sprint so that the research evaluation remained
  the priority.
- **Exact source anchors:**
  `../../../PROJECT_HANDOFF/YHER_PROJECT_OVERVIEW.md`, ledger entry
  `### 2026-07-13 17:11 CST - Codex session`, field `Decisions/corrections`, clauses
  `T1 DOI` and `T2 时间戳线`; and
  `../../../PROJECT_HANDOFF/codex_briefs/2026-07-13_仿真评估实验与论文总攻.md`, S4
  parenthetical `不发布不投稿` and hard gate `H-G` under `## 3. 硬门`.

## D11 - 2026-07-13 - Research Axis Shifted from Generic CAT to Probe Timing

- **Date:** 2026-07-13
- **Classification:** `synthesis`
- **Decision:** This is not a quotation. After the student requested a re-review with
  academic value as the priority, the decision layer proposed demoting generic
  adaptive-versus-fixed performance to H3 and centering the study on finite-budget
  P-state weakness and prerequisite-probe timing. The student commissioned the frozen
  v2 route recorded in the overview and brief. A later statistical review corrected
  the earlier "structural" wording: because P and U differ locally by 0.10, the valid
  claim is **budget-limited weak identifiability** or **practical non-identifiability at
  the pre-specified budgets of 9, 15, and 25 items**.
- **Exact source anchors:**
  `../../../PROJECT_HANDOFF/YHER_PROJECT_OVERVIEW.md`, ledger entry
  `### 2026-07-13 18:01 CST - Codex session`, fields `Work completed`, `Requirements`,
  and `Decisions/corrections`; `../../../PROJECT_HANDOFF/codex_briefs/2026-07-13_仿真评估实验与论文总攻.md`,
  headings `## 0. 本单与 v1 的三个实质差异（先看）` and `## 1. 背景与前置必读`;
  and `../../experiments/analysis_plan.md`, heading `## Question And Scope`, paragraph
  beginning `Production local likelihoods differ between P and U by 0.10`.

## D12 - 2026-07-13 - Freeze the Analysis and Report Negative Results

- **Date:** 2026-07-13
- **Classification:** `documented`
- **Decision:** Hypotheses, arms, analysis populations, seeds, stopping behavior,
  intervals, and ordered decision branches were frozen before confirmatory responses.
  Failures, reversals, wide intervals, sparse-pool repetition, provider exclusions,
  and disagreement with T0 must remain visible and may not trigger outcome-driven
  tuning or extra sampling.
- **Exact source anchors:**
  `../../experiments/analysis_plan.md`, opening status paragraph and headings
  `## Hypothesis Decisions` and `## Honest Reporting And Stopping`; and
  `../../../PROJECT_HANDOFF/codex_briefs/2026-07-13_仿真评估实验与论文总攻.md`, hard gate
  `H-H` in heading `## 3. 硬门`.

## D13 - 2026-07-13 - Three-Layer Contribution Disclosure

- **Date:** 2026-07-13
- **Classification:** `synthesis`
- **Decision:** This is not a quotation. The authorship disclosure separates three
  layers: student direction, requirements, quality gates, go/no-go decisions, and
  interpretation; AI proposals, implementation, automation, and drafting under review
  gates; and an audit layer of commits, tags, tests, hashes, seeds, provenance, and
  isolation attestations. The separation is designed to make both student leadership
  and extensive AI assistance inspectable without claiming manual authorship of AI
  work.
- **Exact source anchors:**
  `/Users/mac/.gstack/projects/Chris-TLC-YHer-skill/ceo-plans/2026-07-13-simulation-eval-paper.md`,
  heading `## 个人主导作用显化（申学第二优先级的落实件）`;
  `../../../PROJECT_HANDOFF/codex_briefs/2026-07-13_仿真评估实验与论文总攻.md`, S4 clause
  `Contribution Statement` and heading `### S5 · 答辩防御包 + 个人主导实录`;
  and [`main.md`](main.md), heading `## 8. Contribution Statement and AI Use`.
