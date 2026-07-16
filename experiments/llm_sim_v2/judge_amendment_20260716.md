# Outcome-Blind Judge Amendment

Date: 2026-07-16

This amendment was written after formal-main collection began but before the
first W3 aggregation, judge-case export, or judge call. It was triggered by a
source-schema audit, not by a response direction: no formal-main candidate
answer or rationale was inspected. It does not change the collection prompts,
task roster, mapping, provider responses, analysis population, sample selector,
estimands, bootstrap, or any confirmatory claim.

The collection-time `public_question` schema is broader than the adjudication
surface. In the frozen terminal roster, its `nodes` field is the exact target
node, so exporting that student-catalog metadata would violate T10 even though
the collection prompt itself remains valid. Judge case manifest v2 therefore
exports only `kind`, `options`, `stem_blocks`, and `stem_text`; it removes
`nodes`, `difficulty`, `source_label`, and every target, misconception, answer,
persona, provider, and mapping field. If an exact target-node label occurs
anywhere in the sanitized judge payload, including the indispensable stem,
options, or the provider's own answer/rationale, that candidate is structurally
excluded from exploratory adjudication rather than redacted or misreported as
blind. This full-payload rule was fixed before any formal-main response was
inspected.
The manifest binds the exact whitelist, protocol bytes and hash, shared input
bytes, opaque case IDs, both structural-exclusion scans, and false target-label
and target-metadata export flags.

The committed `judge_protocol_v1.json` is the sole rubric. It defines
`consistent`, `inconsistent`, `unknown`, and `insufficient_evidence`, nine fixed
error categories, four required output fields, and allowed label-category
pairs. A judge output missing any field, adding any field, using an unlisted
category, or violating a label-category pair is a schema failure rather than a
substantive rating.

Each available judge runs in a fresh isolated execution with no resumed
conversation, prior case context, tools, or external case data. GPT uses a
fresh Codex CLI execution; Claude uses a fresh Claude CLI execution only when
that independently authenticated transport is available. One family cannot be
run twice or relabelled to fill the other slot. Fixed batches contain 10 cases,
with at most two attempts per batch; only transport or output-schema failure is
retryable, never label content. GPT is executed before Claude so no Claude
output exists in the GPT environment. A missing independent transport remains
an explicitly missing judge.

Every pass writes immutable raw attempt artifacts, normalized JSONL, and a
self-hashed execution receipt binding exact model and transport, fresh
execution ID, ordered cases and attempts, raw and normalized hashes, timestamps,
isolation assertions, request/retry counts, returned tokens, known CNY cost,
unknown-billing reserve, and the shared case manifest. Result manifest v2 embeds
the validated receipt. Complete pairwise agreement is admissible only when the
two receipts prove different judge families and execution IDs plus disjoint
attempt and raw-artifact sets. Judge accounting is added to the phase and
hard-fuse totals. The W3 bundle snapshots both result/receipt/raw evidence trees;
hash-only references to unbound external bytes are insufficient.
