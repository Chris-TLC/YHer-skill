#!/usr/bin/env python3
"""Batch12 (first rolling QA-3 batch) apply -- awaiting explicit user L1 authorization.

Audit scope (all verified by Claude, 2026-07-05):
  From 12a's 580 candidates, 19 element-list false splits (`C、N` style) + 50 empty option segments (`D.` no content) removed
      -> apply whitelist of 511 rows (88%), concatenation == original text independently re-verified 100%.
  12b: all 226 rescued PNGs pixel-scanned (1 4x1 noise removed -> 225); 94 kept rows with zero transcript garbage/leaks;
      OMML: 43 rows independently re-compiled with KaTeX, all passed.
  ref_map: 80 unresolvable rows manually verified as real (the entity files never existed in WS1); not applied; merged into batch 13 crop targets.

Write targets:
  1. chemistry_v4_1_3329.jsonl -- 12a's 511 splits (block level, reusing 10g semantics:
     in_block=reorder nodes within the block then split into multiple blocks; tight_text=split text into multiple blocks)
  2. ws2_repaired_assets/ -- collect the 225 rescued images
  3. ws2_asset_transcripts_v1.jsonl -- upgrade rows for 12b rescued assets:
     existing official pool=manual_queue rows -> 94 kept rows upgraded to ai_seed/display_only (with transcript);
     other rescued-but-weakly-transcribed rows stay manual_queue (image now displayable; row tagged batch12_rescued)
  4. ws2_omml_latex_cache_v1.jsonl -- 43 new rows (key omml_sha1, katex_ok=true)

Safety: splitting reuses the 10g-verified techniques (descending order prevents index shift / idempotent exact text match / exclusive vs non-exclusive split handling);
      transcript-row upgrades only modify fields of existing rows; everything idempotent. Dry-run by default.
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
V4 = REPO / "data" / "item_bank" / "v4"
ITEM_BANK = V4 / "chemistry_v4_1_3329.jsonl"
TRANSCRIPTS = V4 / "ws2_asset_transcripts_v1.jsonl"
OMML_CACHE = V4 / "ws2_omml_latex_cache_v1.jsonl"
REPAIRED_DIR = V4 / "ws2_repaired_assets"
B12 = Path("/tmp/yher_batch12_qa3")
APPLY_ID = "batch12_apply_20260705"

MARK = re.compile(r"^\s*([A-D])[.．、]")
ELEM = re.compile(r"^[A-D][、,]\s*[A-Z]")   # element-list false-split signature
EMPTY = re.compile(r"^[A-D][.．、]\s*$")     # empty option segment
PATH_RE = re.compile(r"(\w+)\[(\d+)\]\.para\[(\d+)\]")

def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def write_jsonl(p: Path, rows):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)

def _norm(s): return re.sub(r"\s+", "", s or "")

def clean_split_candidates():
    """Audit whitelist: drop element-list false splits + empty option segments."""
    cands = read_jsonl(B12 / "option_split_v2/option_split_candidates.jsonl")
    keep, rejected = [], Counter()
    for c in cands:
        segs = c["suggested_segments"]
        if any(ELEM.match(s.strip()) for s in segs):
            rejected["elem_list"] += 1
            continue
        if any(EMPTY.match(s.strip()) for s in segs):
            rejected["empty_seg"] += 1
            continue
        if _norm("".join(segs)) != _norm(c["original_text"]):
            rejected["join_mismatch"] += 1
            continue
        keep.append(c)
    return keep, rejected

def apply_split(dry: bool):
    rows = read_jsonl(ITEM_BANK)
    by_id = {r["item_id"]: r for r in rows}
    cands, rejected = clean_split_candidates()
    stats = Counter({f"reject_{k}": v for k, v in rejected.items()})
    by_item = defaultdict(list)
    for c in cands:
        by_item[c["item_id"]].append(c)
    touched = set()
    for iid, sites in by_item.items():
        item = by_id.get(iid)
        if item is None:
            stats["item_missing"] += len(sites)
            continue
        # Within the same field, process in descending block_idx (split later ones first so earlier indexes don't shift)
        def bi(c):
            m = PATH_RE.match(c["block_path"]); return int(m.group(2)) if m else -1
        sites.sort(key=lambda c: (c["field"] if "field" in c else "", -bi(c)))
        for c in sites:
            m = PATH_RE.match(c["block_path"])
            if not m:
                stats["path_bad"] += 1
                continue
            field, bidx, pidx = m.group(1), int(m.group(2)), int(m.group(3))
            blocks = item.get(field) or []
            if bidx >= len(blocks):
                stats["block_oob"] += 1
                continue
            para = blocks[bidx].get("para") if isinstance(blocks[bidx], dict) else None
            if not para or pidx >= len(para):
                stats["para_oob"] += 1
                continue
            node = para[pidx]
            if not (isinstance(node, dict) and node.get("type") == "text"):
                stats["not_text"] += 1
                continue
            if node.get("text") != c["original_text"]:
                stats["text_mismatch_or_done"] += 1
                continue
            segs = c["suggested_segments"]
            if len(para) == 1 and pidx == 0:
                # Exclusive block -> split into multiple blocks (same as 10g)
                blocks[bidx:bidx + 1] = [{"para": [{"type": "text", "text": s}]} for s in segs]
            else:
                # Non-exclusive: split node position into multiple text nodes + separate the option segments into their own block sequence
                # Conservative: first segment stays in place; remaining option segments become new blocks inserted right after this block
                node["text"] = segs[0]
                new_blocks = [{"para": [{"type": "text", "text": s}]} for s in segs[1:]]
                # If the node is the last node in the block, the new option blocks follow this block directly
                if pidx == len(para) - 1:
                    blocks[bidx + 1:bidx + 1] = new_blocks
                else:
                    # Content still follows the node (e.g. an image); keeping later nodes in this block while inserting new blocks after it would scramble the order
                    # -> fall back: don't split, record it (avoid image/text ordering errors)
                    node["text"] = c["original_text"]
                    stats["skip_mid_node"] += 1
                    continue
            stats["split_applied"] += 1
            touched.add(iid)
    for iid in touched:
        by_id[iid]["batch12_source_split"] = APPLY_ID
    stats["items_touched"] = len(touched)
    if not dry:
        write_jsonl(ITEM_BANK, rows)
    return stats

def apply_assets_and_transcripts(dry: bool):
    stats = Counter()
    # 1. Collect rescued images (drop 4x1 noise)
    from PIL import Image
    src = B12 / "asset_rerender/asset_repair/repaired"
    copied = 0
    for p in sorted(src.glob("*.png")):
        im = Image.open(p)
        if im.width < 6 or im.height < 6:
            stats["skip_tiny_png"] += 1
            continue
        dst = REPAIRED_DIR / p.name
        if not dst.exists():
            if not dry:
                shutil.copy2(p, dst)
            copied += 1
    stats["png_collected"] = copied

    # 2. Upgrade transcript rows (official manual_queue rows -> kept promote to pool / others flagged as image rescued)
    rows = read_jsonl(TRANSCRIPTS)
    by_hash = {r["asset_hash"]: r for r in rows}
    kept = read_jsonl(B12 / "asset_rerender/transcripts/transcript_candidates.jsonl")
    rescued_hashes = {p.stem for p in src.glob("*.png")}
    for r in kept:
        h = r["asset_hash"]
        tgt = by_hash.get(h)
        if tgt is None:
            stats["kept_hash_missing"] += 1
            continue
        if tgt.get("apply_id") == APPLY_ID:
            stats["skip_idempotent"] += 1
            continue
        pool = r.get("pool") or "display_only"
        tgt["pool"] = pool
        tgt["fine_type"] = r.get("fine_type") or tgt.get("fine_type")
        tgt["transcript"] = {k: r.get(k) for k in ("summary", "elements", "text_in_image", "data_points", "uncertain")}
        tgt["transcript_confidence"] = r.get("confidence")
        tgt["apply_id"] = APPLY_ID
        tgt["batch12_source"] = "12b_rescued_transcript"
        stats[f"transcript_upgraded_{pool}"] += 1
    # Rows still in manual_queue whose image is now rescued: tag them (rendering can display the image directly)
    for h in rescued_hashes:
        tgt = by_hash.get(h)
        if tgt and tgt.get("pool") == "manual_queue" and tgt.get("batch12_source") is None:
            tgt["batch12_rescued_png"] = True
            stats["manual_png_flagged"] += 1
    if not dry:
        write_jsonl(TRANSCRIPTS, rows)
    return stats

def apply_omml(dry: bool):
    rows = read_jsonl(OMML_CACHE)
    have = {r["omml_sha1"] for r in rows}
    stats = Counter()
    for r in read_jsonl(B12 / "omml_backfill/omml_backfill_candidates.jsonl"):
        sha = r.get("omml_sha1")
        lx = r.get("latex") or r.get("suggested_latex")
        if not sha or not lx:
            stats["bad_row"] += 1
            continue
        if sha in have:
            stats["skip_existing"] += 1
            continue
        rows.append({"omml_sha1": sha, "latex": lx, "ok": True, "katex_ok": True,
                     "batch12_source": "12b_omml_backfill"})
        have.add(sha)
        stats["omml_added"] += 1
    if not dry:
        write_jsonl(OMML_CACHE, rows)
    return stats, len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dry = not args.apply
    print(f"=== Batch12 {'APPLY (real write)' if not dry else 'DRY-RUN'} ===")
    s1 = apply_split(dry)
    print("[12a split]")
    for k, v in sorted(s1.items()): print(f"   {k}: {v}")
    s2 = apply_assets_and_transcripts(dry)
    print("[12b images + transcripts]")
    for k, v in sorted(s2.items()): print(f"   {k}: {v}")
    s3, total = apply_omml(dry)
    print(f"[12b OMML] cache rows -> {total}")
    for k, v in sorted(s3.items()): print(f"   {k}: {v}")
    print("=== DONE ===")

if __name__ == "__main__":
    main()
