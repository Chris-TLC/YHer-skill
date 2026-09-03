#!/usr/bin/env python3
"""Batch13 (QA-3 visual crops) apply -- user authorization dated 2026-07-05 (bound to batch13's three goals).

Audit: 105 kept with zero misattributions (12 evidence chains + 4 manually verified images); 13d's 97 kept items with zero garbage/leaks; refmap's 28 rows compliant.
Writes:
  1. ws2_repaired_assets/ +105 crops, named by final_asset_hash
     (dead_asset 77 = original hash as-is; dead_ref 28 = crop_sha256 new hash)
  2. ws2_media_ref_map_v1.jsonl +28 rows (new mapping for dead references; in_ws2_manifest=false matches the live-service convention;
     the loader does not read this field; (group,media)->hash is sufficient for addressing)
  3. ws2_asset_transcripts_v1.jsonl: 13d's 97 rows written per the official schema
     - hash already official (76): promote to pool / fill fields (latex -> formula_latex pool; transcript -> per candidate pool)
     - new hash (24 = 28 dead_ref minus 4 without transcript): new rows
     - icon_or_noise never promotes to pool (batch12 lesson; none in 13d, but defensive filtering kept)
Idempotent: apply_id marker; skip if image exists; skip if ref_map already has (group,media). Dry-run by default.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
V4 = REPO / "data" / "item_bank" / "v4"
TRANSCRIPTS = V4 / "ws2_asset_transcripts_v1.jsonl"
REF_MAP = V4 / "ws2_media_ref_map_v1.jsonl"
REPAIRED = V4 / "ws2_repaired_assets"
B13 = Path("/tmp/yher_batch13_qa3")
APPLY_ID = "batch13_apply_20260705"

def read_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def write_jsonl(p, rows):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dry = not args.apply
    print(f"=== Batch13 {'APPLY (real write)' if not dry else 'DRY-RUN'} ===")
    stats = Counter()

    kept = read_jsonl(B13 / "crop_candidates.jsonl")
    # 1. Collect crops (named by final_asset_hash)
    for c in kept:
        fh = c["final_asset_hash"]
        src = Path(c["crop_path"])
        dst = REPAIRED / f"{fh}.png"
        if dst.exists():
            stats["png_exists_skip"] += 1
            continue
        if not src.exists():
            stats["png_src_missing"] += 1
            continue
        if not dry:
            shutil.copy2(src, dst)
        stats["png_collected"] += 1

    # 2. ref_map +28 (dead_ref)
    ref_rows = read_jsonl(REF_MAP)
    have_gm = {(r.get("group_key"), r.get("media")) for r in ref_rows}
    for r in read_jsonl(B13 / "refmap_new_rows.jsonl"):
        gm = (r["group_key"], r["media"])
        if gm in have_gm:
            stats["ref_exists_skip"] += 1
            continue
        # Take this media's zones from its kept row
        src_row = next((c for c in kept if c["group_key"] == r["group_key"] and c["media"] == r["media"]), None)
        zones = (src_row or {}).get("zones") or []
        ref_rows.append({"group_key": r["group_key"], "media": r["media"],
                         "asset_hash": r["asset_hash"], "in_ws2_manifest": False,
                         "zones": zones, "batch13_source": APPLY_ID})
        have_gm.add(gm)
        stats["ref_added"] += 1
    if not dry:
        write_jsonl(REF_MAP, ref_rows)

    # 3. Transcript table: 13d's 97 rows
    t_rows = read_jsonl(TRANSCRIPTS)
    by_hash = {r["asset_hash"]: r for r in t_rows}
    # formula
    for r in read_jsonl(B13 / "crop_transcripts/formula_latex/formula_latex_candidates.jsonl"):
        h = r["asset_hash"]
        latex = r.get("latex")
        ok = bool((r.get("compile_result") or {}).get("ok"))
        if not latex or not ok:
            stats["formula_skip_bad"] += 1
            continue
        tgt = by_hash.get(h)
        if tgt is None:
            tgt = {"asset_hash": h, "asset_class": r.get("asset_class", "formula_image"),
                   "schema_version": "ws2_transcript_v1"}
            t_rows.append(tgt)
            by_hash[h] = tgt
            stats["formula_new_row"] += 1
        elif tgt.get("apply_id") == APPLY_ID:
            stats["skip_idempotent"] += 1
            continue
        tgt["pool"] = "formula_latex"
        tgt["latex"] = latex
        tgt["latex_status"] = "passed"
        tgt["latex_consistency"] = bool((r.get("consistency") or {}).get("consistent"))
        tgt["apply_id"] = APPLY_ID
        tgt["batch13_source"] = "13d_crop_formula"
        stats["formula_applied"] += 1
    # transcript
    for r in read_jsonl(B13 / "crop_transcripts/transcripts/transcript_candidates.jsonl"):
        h = r["asset_hash"]
        pool = r.get("pool") or "display_only"
        if r.get("fine_type") == "icon_or_noise":
            stats["skip_icon"] += 1  # batch12 lesson: noise never promotes to pool
            continue
        tgt = by_hash.get(h)
        if tgt is None:
            tgt = {"asset_hash": h, "asset_class": r.get("asset_class", "illustration"),
                   "schema_version": "ws2_transcript_v1"}
            t_rows.append(tgt)
            by_hash[h] = tgt
            stats["transcript_new_row"] += 1
        elif tgt.get("apply_id") == APPLY_ID:
            stats["skip_idempotent"] += 1
            continue
        tgt["pool"] = pool
        tgt["fine_type"] = r.get("fine_type") or tgt.get("fine_type")
        tgt["transcript"] = {k: r.get(k) for k in ("summary", "elements", "text_in_image", "data_points", "uncertain")}
        tgt["transcript_confidence"] = r.get("confidence")
        tgt["apply_id"] = APPLY_ID
        tgt["batch13_source"] = "13d_crop_transcript"
        stats[f"transcript_applied_{pool}"] += 1
    if not dry:
        write_jsonl(TRANSCRIPTS, t_rows)

    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  transcript rows: {len(t_rows)}")
    print("=== DONE ===")

if __name__ == "__main__":
    main()
