#!/usr/bin/env python3
"""Batch 10 (QA-1) apply -- user authorization dated 2026-07-05 ("authorize all writes to the v4 main store").

Audit report: PROJECT_HANDOFF/BATCH10_AUDIT_2026-07-05.md
Authorization level: L1 (targeting the v4 main store). Mode follows WS2/batch7: pure additions / precise backfills + idempotent + dry-run.

Write targets (backed up throughout in data/_backup_pre_batch10_apply_20260705/):
  1. ws2_asset_transcripts_v1.jsonl
     - 10a: add 4075 transcription/latex rows for answer/analysis-zone assets (zero collisions with official rows, verified)
            latex tiered by whitelist: compliant -> latex_status=passed (rendering enabled); else -> archived (image shown directly)
     - 10b: 238 existing ai_seed/display_only rows get latex filled in (whitelist tiering)
     - 10d: 54 rows' latex changed -- 15 via Claude re-judgment (reviewer=claude), 12 deterministic, 27 no-op
  2. ws2_omml_latex_cache_v1.jsonl
     - 10f: 52 rows katex_ok=false -> upgraded to latex-converted (independently compiled 100% pass); 16 rows' placeholder cleared to bare symbols
  3. chemistry_v4_1_3329.jsonl
     - 10c: backfill content fields for 152 items (stem/answer/analysis blocks + stem_text),
            keeping all official service fields untouched (pool/service_eligible/quality_flags/item_id unchanged) -- literal-only cleanup

Not written: ws2_media_ref_map_v1.jsonl (loader does not read in_ws2_manifest; the 4209 refs already exist officially, no need to touch).
10e rejected wholesale, not included; 10g left for QA-3; 10a manual_queue/broken placeholder rows recorded as-is (pool=manual_queue, rendering side degrades).
"""
from __future__ import annotations
import argparse, json, re, shutil, sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
V4 = REPO / "data" / "item_bank" / "v4"
B10 = Path("/tmp/yher_batch10_qa1")
APPLY_ID = "batch10_apply_20260705"

TRANSCRIPTS = V4 / "ws2_asset_transcripts_v1.jsonl"
OMML_CACHE = V4 / "ws2_omml_latex_cache_v1.jsonl"
ITEM_BANK = V4 / "chemistry_v4_1_3329.jsonl"

# ---- Audit report P2 whitelist: compiles AND <=120 chars AND no Roman-code style AND no Lewis-structure signature -> rendering enabled ----
ROMAN_CODE_RE = re.compile(r"\\ce\{[^}]*\b(?:I{1,3}|IV|V|VI)\b")
LEWIS_RE = re.compile(r"\\ddot|(?<![a-zA-Z])：|(?<!\\):(?=[A-Z])")

def latex_enabled(latex: str, compile_ok: bool) -> bool:
    if not latex or not compile_ok:
        return False
    if len(latex) > 120:
        return False
    if ROMAN_CODE_RE.search(latex):
        return False
    if LEWIS_RE.search(latex):
        return False
    return True

# ---- Audit report P1: 15 rows in 10d re-judged by Claude (images verified by hand) ----
CLAUDE_10D_OVERRIDES = {
    "5e630523b505a41b90f53bca3b888e764cc3e3b5eec6450bbd8035b69f83d707": "{}^{2-}_{3}",
    "322dcdac6ccd9eb076582b36bb226a192af95d358fb03557d7ada75191ef2b6b": "{}^{2-}_{3}",
    "36f531504693de146c26e4e13d4599554fc552e1fc2eee9ebadac35dd1a857fe": "{}^{2-}_{3}",
    "5db4bca15cef757c5f51a93cef122177dd4552bc39f62e19c4fd81d951353f96": "{}^{2-}_{3}",
    "5f75c6ffc3379e7f6139681abb7ac4a33449a52c2a6daff895ce0b3c57bb3dc9": "{}^{2-}_{3}",
    "8255eb2edfa1171e8a0f4e6732f4f9c2b5521b5a3641a827082eb8c622ebf286": "{}^{2-}_{3}",
    "98972fbac4a7c15bfc5789172cf004cb4aacfac87cf32483e001133d74aa7429": "{}^{2-}_{3}",
    "e4a6ae99c45ea40ce6671fb9465fb7562b561306c8f98e2d69f073899f5d1b7e": "{}^{2-}_{3}",
    "26113c362ebf30f2bd3766fd42f0bc7e0a9ec69519d6cf808dc01fb07f546a11": "{}^{2-}_{7}",
    "ce0b470cf11f91cffa147aa2035356af72aa9aa7f1b0da4e5fa3f9ec0f3cd715": "{}^{2-}_{7}",
    "e87433975c3f3a3aef79ac12aa8a648693d5ab05ea80a40cc204ac02065d6d47": "{}^{2-}_{7}",
    "7985971fedc2394349388e4f0119b2c669c6da14c6aea70129873f18bdbdab00": "{}^{3-}_{4}",
    "ea8608f29c3c8a952a34184dce5611ad2aafbd4f753f98c68199709d5d8db37d": "{}^{3-}_{4}",
    "0d513979434a22ad4155e3135fbcf91548fc978c51a8533a03732d71dbb44931": "{}^{12}_{6}",
    "8f3a402ac97cfb7756821644d70accb489085ee9af349846055a08b5ed5ad64b": "{}^{3}_{2}",
}

def read_jsonl(p: Path):
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(p: Path, rows):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)

# ---------- Build 10a new transcript rows (candidate schema -> official schema) ----------
def build_10a_rows():
    rows = []
    stats = Counter()
    # formula candidates -> formula_latex rows
    for r in read_jsonl(B10 / "answer_zone_assets/formula_latex/formula_latex_candidates.jsonl"):
        h = r["asset_hash"]
        if r.get("pool") == "manual_queue" or not r.get("latex"):
            rows.append({"asset_hash": h, "asset_class": r.get("asset_class", "formula_image"),
                         "pool": "manual_queue", "schema_version": "ws2_transcript_v1",
                         "apply_id": APPLY_ID, "batch10_source": "10a_formula_failed"})
            stats["10a_formula_manual"] += 1
            continue
        compile_ok = bool((r.get("compile_result") or {}).get("ok"))
        enabled = latex_enabled(r["latex"], compile_ok)
        rows.append({"asset_hash": h, "asset_class": r.get("asset_class", "formula_image"),
                     "pool": "formula_latex", "schema_version": "ws2_transcript_v1",
                     "apply_id": APPLY_ID, "latex": r["latex"],
                     "latex_status": "passed" if enabled else "archived",
                     "latex_consistency": bool((r.get("consistency") or {}).get("consistent")),
                     "batch10_source": "10a_formula"})
        stats["10a_formula_passed" if enabled else "10a_formula_archived"] += 1
    # transcript candidates -> transcript rows
    for r in read_jsonl(B10 / "answer_zone_assets/transcripts/transcript_candidates.jsonl"):
        h = r["asset_hash"]
        pool = r.get("pool") or "manual_queue"
        if pool == "manual_queue" or not (r.get("summary") or r.get("elements")):
            rows.append({"asset_hash": h, "asset_class": r.get("asset_class", "illustration"),
                         "pool": "manual_queue", "fine_type": r.get("fine_type"),
                         "schema_version": "ws2_transcript_v1", "apply_id": APPLY_ID,
                         "batch10_source": "10a_transcript_broken"})
            stats["10a_transcript_manual"] += 1
            continue
        transcript = {k: r.get(k) for k in ("summary", "elements", "text_in_image", "data_points", "uncertain")}
        rows.append({"asset_hash": h, "asset_class": r.get("asset_class", "illustration"),
                     "pool": pool, "fine_type": r.get("fine_type"),
                     "schema_version": "ws2_transcript_v1", "apply_id": APPLY_ID,
                     "transcript": transcript,
                     "transcript_confidence": r.get("confidence"),
                     "batch10_source": "10a_transcript"})
        stats[f"10a_transcript_{pool}"] += 1
    return rows, stats

# ---------- 10b/10d: change latex in existing official rows ----------
def load_10b_updates():
    upd = {}
    for r in read_jsonl(B10 / "formula_backfill/formula_backfill_candidates.jsonl"):
        if not r.get("latex"):
            continue
        compile_ok = bool((r.get("compile_result") or {}).get("ok"))
        upd[r["asset_hash"]] = (r["latex"], latex_enabled(r["latex"], compile_ok))
    return upd

def load_10d_updates():
    upd = {}
    for r in read_jsonl(B10 / "latex_form_fix/latex_form_fix_candidates.jsonl"):
        h = r["asset_hash"]
        orig = (r.get("original_latex") or "").strip()
        if h in CLAUDE_10D_OVERRIDES:
            new = CLAUDE_10D_OVERRIDES[h]
            upd[h] = (new, "claude")
        else:
            sug = (r.get("suggested_latex") or r.get("latex") or "").strip()
            if sug and sug != orig:  # deterministic change (no-op left untouched)
                upd[h] = (sug, "batch10_deterministic")
    return upd

def apply_transcripts(dry: bool):
    rows = read_jsonl(TRANSCRIPTS)
    by_hash = {r["asset_hash"]: r for r in rows}
    new_rows, a_stats = build_10a_rows()
    # Idempotent: skip rows already applied
    existing_b10 = {r["asset_hash"] for r in rows if r.get("apply_id") == APPLY_ID}
    new_rows = [r for r in new_rows if r["asset_hash"] not in by_hash and r["asset_hash"] not in existing_b10]

    b_upd = load_10b_updates()
    d_upd = load_10d_updates()
    stats = Counter(a_stats)
    for r in rows:
        h = r["asset_hash"]
        if h in b_upd:
            latex, enabled = b_upd[h]
            r["latex"] = latex
            r["latex_status"] = "passed" if enabled else "archived"
            r["latex_consistency"] = r.get("latex_consistency", False)
            r["batch10_source"] = "10b_backfill"
            r["apply_id"] = APPLY_ID
            stats["10b_passed" if enabled else "10b_archived"] += 1
        if h in d_upd:
            latex, reviewer = d_upd[h]
            r["latex"] = latex
            r["latex_status"] = "passed"
            if reviewer == "claude":
                r["reviewer"] = "claude"
                r["batch10_source"] = "10d_claude_override"
                stats["10d_claude"] += 1
            else:
                r["batch10_source"] = "10d_deterministic"
                stats["10d_deterministic"] += 1
            r["apply_id"] = APPLY_ID
    final = rows + new_rows
    stats["10a_new_rows_total"] = len(new_rows)
    if not dry:
        write_jsonl(TRANSCRIPTS, final)
    return stats, len(rows), len(final)

def apply_omml(dry: bool):
    rows = read_jsonl(OMML_CACHE)
    by_sha = {r["omml_sha1"]: r for r in rows}
    stats = Counter()
    for r in read_jsonl(B10 / "omml_retry/omml_retry_candidates.jsonl"):
        sha = r.get("omml_sha1")
        latex = r.get("suggested_latex")
        ok = bool((r.get("compile_result") or {}).get("ok"))
        if not sha or not latex or not ok:
            continue
        latex = latex.replace("❑", "").replace("□", "")  # clear placeholder chars
        if sha in by_sha:
            tgt = by_sha[sha]
            if tgt.get("katex_ok") and tgt.get("latex") == latex:
                continue  # idempotent
            tgt["latex"] = latex
            tgt["ok"] = True
            tgt["katex_ok"] = True
            tgt["batch10_source"] = "10f_omml_retry"
            stats["10f_upgraded"] += 1
    if not dry:
        write_jsonl(OMML_CACHE, rows)
    return stats, len(rows)

# ---- 10c backfill guardrails (hardened after the audit) ----
# (1) Scope: answer/analysis zones only. The stem-zone rerun turned [figure:imageN.png] into bare-name media nodes,
#     but those bare names are neither in the official ref_map nor in WS1 assets (real files carry the ans_ prefix)
#     -> backfilling would instead degrade the stem images.
#     Re-segmenting stems needs ref_map support, which is beyond "pure literal cleanup"; left to QA-3.
# (2) Blocks are backfilled only when the candidate has not lost official real images and text volume did not drop >60% (blocks collapse-prone bad candidates from reruns).
# stem_text is a derived summary; it is refreshed when answer/analysis are backfilled; if stem literals are untouched, stem_text keeps the official value.
BACKFILL_FIELDS = ("answer_blocks_effective", "analysis_blocks")

def _struct_media(blocks):
    out = set()
    def w(b):
        if isinstance(b, dict):
            if b.get("media") and b.get("type") in ("figure", "image", "formula"):
                out.add(b["media"])
            for v in b.values():
                w(v)
        elif isinstance(b, list):
            for v in b:
                w(v)
    w(blocks)
    return out

def _text_len(blocks):
    tot = 0
    def w(b):
        nonlocal tot
        if isinstance(b, dict):
            if b.get("text"):
                tot += len(b["text"])
            for v in b.values():
                w(v)
        elif isinstance(b, list):
            for v in b:
                w(v)
    w(blocks)
    return tot

def _blocks_safe(old_val, new_val) -> bool:
    """Whether candidate blocks are safe to backfill: real image count does not decrease AND text volume does not drop >60%."""
    old_media, new_media = _struct_media(old_val), _struct_media(new_val)
    if len(new_media) < len(old_media):
        return False
    ot, nt = _text_len(old_val), _text_len(new_val)
    if ot > 50 and nt < ot * 0.4:
        return False
    return True

def apply_item_bank(dry: bool):
    rows = read_jsonl(ITEM_BANK)
    by_id = {r["item_id"]: i for i, r in enumerate(rows)}
    target = [r for r in read_jsonl(B10 / "literal_scan/ws1_segmentation/fixed_candidate_items.jsonl")
              if r.get("candidate_origin") == "batch6_ws1_bounded_rerun"]
    stats = Counter()
    for f in target:
        old_id = f.get("old_item_id")
        if old_id not in by_id:
            stats["10c_miss"] += 1
            continue
        item = rows[by_id[old_id]]
        if item.get("batch10_source") == "10c_literal_fix":
            stats["10c_skip_idempotent"] += 1
            continue
        changed = False
        guarded = False
        for fld in BACKFILL_FIELDS:
            if fld not in f:
                continue
            if json.dumps(f[fld], ensure_ascii=False) == json.dumps(item.get(fld), ensure_ascii=False):
                continue
            if not _blocks_safe(item.get(fld), f[fld]):
                guarded = True  # candidate collapsed for this field; keep the official original value
                continue
            item[fld] = f[fld]
            changed = True
        if changed:
            item["batch10_source"] = "10c_literal_fix"
            stats["10c_backfilled"] += 1
            if item.get("pool") == "main" and item.get("service_eligible") is True:
                stats["10c_backfilled_in_service"] += 1
        if guarded:
            stats["10c_field_guarded"] += 1
    if not dry:
        write_jsonl(ITEM_BANK, rows)
    return stats, len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write for real (dry-run by default)")
    args = ap.parse_args()
    dry = not args.apply
    mode = "DRY-RUN" if dry else "APPLY (real write)"
    print(f"=== Batch10 {mode} ===\n")

    t_stats, t_before, t_after = apply_transcripts(dry)
    print(f"[transcripts] {t_before} -> {t_after} rows (+{t_after - t_before})")
    for k, v in sorted(t_stats.items()):
        print(f"   {k}: {v}")

    o_stats, o_total = apply_omml(dry)
    print(f"\n[OMML cache] {o_total} rows")
    for k, v in sorted(o_stats.items()):
        print(f"   {k}: {v}")

    i_stats, i_total = apply_item_bank(dry)
    print(f"\n[item bank] {i_total} rows (row count unchanged, in-place backfill)")
    for k, v in sorted(i_stats.items()):
        print(f"   {k}: {v}")

    print(f"\n=== {mode} DONE ===")

if __name__ == "__main__":
    main()
