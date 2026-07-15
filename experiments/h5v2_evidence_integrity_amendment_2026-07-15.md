# Persona-v2 Pre-Main Evidence-Integrity Amendment

Status: must be committed with the evidence-v2 runtime before any replacement
pilot observation or main observation.

This amendment changes collection provenance only. It does not change the
persona grid, task roster, prompts, prompt revision, provider/model parameters,
mapping, estimands, thresholds, bootstrap, H1-H4 evidence, or scientific claim
boundary.

## Trigger

A post-pilot independent audit found two irrecoverable limits in the first
formal pilot store:

1. Its before/after resume snapshots cover response records, but do not retain
   a durable count of calls entering the provider transport. A call that ended
   before response-record persistence would not appear in those snapshots.
2. Response records retain normalized `parsed_output`, but not the exact
   provider content string consumed by the strict JSON parser. Serializing the
   normalized object cannot reproduce duplicate-key, wrapper-text, or original
   byte-level parsing behavior.

The earlier pilot GO is withdrawn as a release gate for main collection. Its
engineering observations remain legacy evidence only and are excluded from all
main estimates.

## Evidence-v2 Contract

- New observations use `yher.llm_sim_v2.response_record.v2`. Every transport
  response mapping stores the exact `response_content` string,
  `response_content_utf8_bytes`, and SHA-256 of its UTF-8 bytes. Complete
  records must replay through the strict parser to exactly the stored
  `parsed_output`.
- A response-record v1 file is not resumable under the evidence-v2 runner. This
  fail-closed rule prevents the legacy pilot from being silently upgraded or
  mixed with evidence-v2 records.
- Every provider manifest binds the exact persisted record files by relative
  path, byte length, SHA-256, attempt count, and response-attempt count, plus a
  deterministic aggregate record-set digest.
- Each provider/phase has an append-only event stream. Events bind their
  predecessor hash and implement an exact invocation state machine:
  `invocation_started -> zero or more provider_call_started ->
  invocation_finished`. A final receipt binds the actual transport-entry
  event delta and before/after phase-store hashes. An unmatched invocation or
  call blocks phase anchoring; caught interruptions close explicitly with
  `status=interrupted`.
- Transport-unavailable providers also emit a closed, zero-call invocation
  receipt. They are not represented by a silent absence of evidence events.
- A post-invocation phase receipt re-reads provider manifests, recomputes every
  record-set digest from disk, validates every provider event state machine,
  and binds provider-manifest hashes and evidence-chain heads. This phase
  receipt, not the pre-finish mutable provider manifest alone, is the
  authoritative artifact for a Git anchor.

The stored content is the exact model message content seen by the strict parser,
not HTTP headers, authorization material, or an unredacted wire envelope.

### Fresh-review closure before replacement observation

- A phase receipt enumerates the exact provider set independently from active
  phase provenance, provider-manifest filenames and content, provider-event
  directories, and record directories. Missing or orphaned providers, unbound
  files, filename/content mismatches, and unmatched invocations fail closed.
- Every anchored record must be `response_record.v2`. The receipt validates
  task identity, contiguous attempts, response-content bindings, billing and
  retry totals, status semantics, outcomes, and complete-record strict replay
  against the active frozen task. Manifest record sets are recomputed from the
  same bytes before anchoring.
- Every persisted attempt maps one-to-one to a hash-chained
  `provider_call_started` event on task ID, attempt number, requested model,
  requested token limit, and wire-message hash. Unavailable and complete
  zero-call resume invocations must contain zero call events. An unmatched call
  or attempt blocks anchoring.
- One OS-backed exclusive lock covers each provider transaction from reading
  existing records through invocation finish and manifest persistence. Event
  indices use exclusive immutable publication. Concurrent runners therefore
  serialize; the later runner re-reads the winner's records and enters the
  transport zero times.
- A caught transport-side interruption after `provider_call_started` is
  terminal for that task in the current epoch. Before propagating the original
  interruption, the runner persists a failed attempt with unknown billing, a
  CNY 10 reserve, and `needs_user=true`. A later invocation may therefore close
  as a zero-call resume and reconcile the phase one-to-one. If that terminal
  record cannot be persisted, the unresolved call blocks the epoch instead.
- A process death after `invocation_started` leaves immutable unmatched
  evidence. The next runner acquires the released OS lock but fails before any
  provider call. Recovery requires an explicit, reviewed new epoch; the event
  is never deleted, overwritten, or silently completed after the fact.

## Legacy Pilot And Cost

The existing pilot must not be edited, deleted, or treated as evidence-v2. A
separate carried-forward ledger will bind its retrospective evidence receipt
and retain:

- prior known cost: CNY 0.54466321;
- prior ambiguity reserve: CNY 0.11300000;
- legacy formal-pilot known cost: CNY 1.91386592;
- pre-recollection fuse-facing total: CNY 2.57152913.

The frozen `prior_cost_ledger.json` remains byte-unchanged. The separate
carried-forward ledger has its own digest and source-receipt digest; both its
digest and exact known/reserve/total values enter phase provenance and the run
budget. New-epoch response costs are then added. No ledger may relabel legacy
cost as pre-run frozen cost or reset the CNY 300/450 thresholds.

Formal pilot and main provenance additionally pin the reviewed carried-ledger
identity `87ff7a3d...be35`, legacy receipt `2ea161a0...2999`, aggregate legacy
record set `1490e7d3...8c0`, and CNY `1.91386592 / 0 / 1.91386592`. Omission or
a self-hashed substitute is invalid; the explicit source-record-set binding is
also copied into phase and response-record provenance.

## Required Replacement Pilot

The missing exact provider content cannot be reconstructed retrospectively, and
the evidence writer is new runtime code. Therefore the smallest compliant
release gate is a fresh full formal pilot with the same frozen 128 tasks for
each of DeepSeek and Doubao. A partial or synthetic-only rerun does not replace
the contracted formal pilot. The legacy and replacement pilots both remain
physically excluded from main analysis.

## Commit And Anchor Order

1. Commit evidence-v2 code, tests, this amendment, and deviation record.
2. Refresh and commit the runtime task manifest so it binds the exact evidence
   writer, runner, collector, store guard, and transport bytes.
3. Produce and commit the retrospective legacy-pilot receipt and separate
   carried-forward cost ledger. Verify the legacy pilot bytes remain unchanged.
4. Collect the full replacement pilot into a fresh epoch root using prompt
   revision 0 and the carried-forward ledger.
5. Export and commit the first post-collection phase receipt (anchor A).
6. From anchor A's descendant commit, run the complete-pilot resume. Require a
   closed receipt with provider-call delta zero for both providers, export the
   second phase receipt, and commit it as anchor B.
7. Run the evidence-v2 pilot auditor against anchor A -> anchor B. Main remains
   blocked until every evidence gate and the existing engineering gates pass.

## T1-T10 Effect

- T1/T2: mapping and blind prompt bytes do not change; rerun their assertions.
- T3: run ID stays `llm-personas-v2-dual`; v1 and official paths remain
  untouched. The replacement pilot uses a fresh storage epoch.
- T4/T5/T6: statistical unit, wording, and text-only modality do not change.
- T7: legacy and replacement pilot roots are excluded from main analysis.
- T8: prompt revision remains 0; this is a storage/provenance revision.
- T9: no push, publication, submission, DOI, official write, or credential
  persistence is authorized.
- T10: evidence receipts are not judge inputs. Judge export remains limited to
  the sanitized blind prompt and candidate output.
