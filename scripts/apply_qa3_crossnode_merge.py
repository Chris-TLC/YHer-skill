#!/usr/bin/env python3
"""QA-3 cross-node broken-ion apply -- user authorization dated 2026-07-05 ("apply QA-3 first"), targeting chemistry_v4_1_3329.jsonl.

Audit: Claude fully verified all 285 candidates by hand (17 groups of head+cluster->ion chemically self-consistent,
      SO3/SO4 disambiguated across different images, 4 PNGs manually checked, reversible rev_fail=0/285, right_node unchanged throughout).
      Codex's scripts/apply_qa3_crossnode_ions.py is the candidate generator (--apply hard-refused);
      the official write is executed by this script (Claude).

Writes: 285 cross-node sites in chemistry_v4_1_3329.jsonl. Per site:
      left text node (ending in an element-symbol head, e.g. "...c(SO") -> replaced_left("...c(SO₃²⁻");
      the adjacent formula node (opaque WMF subscript-charge image) is deleted; right node untouched.

Safety:
  - Location: candidate pins (field, left_block_idx, formula_para_idx), formula = left+1.
  - Misalignment guard: only edit when left.text exactly == original_left and formula node type==formula and
            media==candidate media. Any mismatch -> skip, count as mismatch.
  - Index-shift guard: within the same (item,field,block), process in descending formula_para_idx (delete later ones first,
                earlier site indexes unaffected) -- same descending-order technique as 10g.
  - Only left+formula are verified, not right (when sites share a node, right may already have been modified by an adjacent site).
  - Idempotent: left.text already == replaced_left -> skip.

Dry-run by default. --apply writes for real (user already granted L1).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
ITEM_BANK = REPO / "data" / "item_bank" / "v4" / "chemistry_v4_1_3329.jsonl"
CANDIDATES = Path("/tmp/yher_qa3_crossnode_candidates.jsonl")
APPLY_ID = "qa3_crossnode_apply_20260705"

def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def write_jsonl(p: Path, rows):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)

def apply_qa3(dry: bool):
    rows = read_jsonl(ITEM_BANK)
    by_id = {r["item_id"]: r for r in rows}
    cands = read_jsonl(CANDIDATES)
    stats = Counter()
    touched = set()

    # Group by item -> within a group, sort by (field, block, descending formula_para_idx)
    by_item = defaultdict(list)
    for c in cands:
        by_item[c["item_id"]].append(c)

    for iid, sites in by_item.items():
        item = by_id.get(iid)
        if item is None:
            stats["item_missing"] += len(sites)
            continue
        # Re-entrant idempotency: skip items already applied (marked) -- deleting nodes changes para indexes,
        # so reusing original indexes would misalign; an item-level marker is used instead of site-level exact matching.
        if item.get("batch_source_qa3") == APPLY_ID:
            stats["skip_applied_item"] += 1
            stats["skip_applied_sites"] += len(sites)
            continue
        # Descending order: within a block delete later formulas first, so earlier site indexes don't shift
        sites.sort(key=lambda s: (s["field"], s["left_block_idx"], -s["formula_para_idx"]))
        for s in sites:
            blocks = item.get(s["field"]) or []
            bi = s["left_block_idx"]
            if bi >= len(blocks):
                stats["block_oob"] += 1
                continue
            para = blocks[bi].get("para") if isinstance(blocks[bi], dict) else None
            if not para:
                stats["para_missing"] += 1
                continue
            li, fi = s["left_para_idx"], s["formula_para_idx"]
            if li >= len(para):
                stats["left_oob"] += 1
                continue
            left = para[li]
            # Idempotent: already applied
            if isinstance(left, dict) and left.get("text") == s["replaced_left"]:
                stats["skip_idempotent"] += 1
                continue
            # Misalignment guard: left must exactly match original_left
            if not (isinstance(left, dict) and left.get("type") == "text" and left.get("text") == s["original_left"]):
                stats["mismatch_left"] += 1
                continue
            # Misalignment guard: formula node must exist, type and media must match
            if fi >= len(para):
                stats["formula_oob"] += 1
                continue
            formula = para[fi]
            if not (isinstance(formula, dict) and formula.get("type") == "formula" and str(formula.get("media") or "") == str(s.get("media") or "")):
                stats["mismatch_formula"] += 1
                continue
            # Execute: backfill left + delete formula node
            left["text"] = s["replaced_left"]
            del para[fi]
            stats["applied"] += 1
            touched.add(iid)

    for iid in touched:
        by_id[iid]["batch_source_qa3"] = APPLY_ID
    stats["items_touched"] = len(touched)
    if not dry:
        write_jsonl(ITEM_BANK, rows)
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write for real (after explicit user L1 authorization)")
    args = ap.parse_args()
    print(f"=== QA-3 cross-node ions {'APPLY (real write)' if args.apply else 'DRY-RUN'} ===")
    st = apply_qa3(not args.apply)
    for k, v in sorted(st.items()):
        print(f"  {k}: {v}")
    print("=== DONE ===")

if __name__ == "__main__":
    main()
