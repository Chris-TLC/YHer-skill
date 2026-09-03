#!/usr/bin/env python3
"""10g option-split apply (first QA-3 item) -- 740 high-confidence candidates.

Audit: PROJECT_HANDOFF/BATCH10_AUDIT_2026-07-05.md P4.
Scope: only candidates in the service pool's stem_blocks whose suggested split is a valid strictly-increasing option sequence (A<B<C<D).
       The 82 in the answer area + 46 structurally suspicious + non-service-pool ones are all excluded (left for later QA-3 work).
Semantics: multi-options glued into a single text node (e.g. "A.淀粉B.二氧化硫") are located by block_path and
      split into multiple independent blocks in stem_blocks (one block per option = {"para":[{text}]}),
      frontend renders each block as one rir-para with line break + rir-option hanging indent.
      Pure text operation; does not touch media/ref_map (unrelated to the 10c stem backfill pitfalls).
Narrowed scope: only candidates whose glued node is the sole node in its block are processed (unambiguous split: 1 block -> N blocks);
          the 51 candidates whose node shares a block with other content are left for later (split position is ambiguous).
Safety: the text of the node located by block_path must exactly equal the candidate's original_text before splitting (prevents misalignment);
      the concatenation of the split segments (whitespace-stripped) must equal the original text (prevents dropped characters); zero matches are skipped. Idempotent: already-split candidates are skipped.

Authorization: requires explicit user L1 (item bank main-store write). Dry-run by default.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
ITEM_BANK = REPO / "data" / "item_bank" / "v4" / "chemistry_v4_1_3329.jsonl"
CANDIDATES = Path("/tmp/yher_10g_high_confidence.jsonl")
APPLY_ID = "batch10_10g_optsplit_20260705"

MARK = re.compile(r"^\s*([A-D])[.．、]")
PATH_RE = re.compile(r"(\w+)\[(\d+)\]\.para\[(\d+)\]")

def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def write_jsonl(p: Path, rows):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)

def locate(item, path):
    """Return (blocks_list, block_idx, para_idx) or None."""
    m = PATH_RE.match(path)
    if not m:
        return None
    field, bi, pi = m.group(1), int(m.group(2)), int(m.group(3))
    blocks = item.get(field)
    if not blocks or bi >= len(blocks):
        return None
    block = blocks[bi]
    para = block.get("para") if isinstance(block, dict) else None
    if not para or pi >= len(para):
        return None
    return blocks, bi, pi

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")

def seq_ok(segs):
    letters = []
    for s in segs:
        mm = MARK.match(s)
        letters.append(mm.group(1) if mm else None)
    tail = letters[1:] if letters and letters[0] is None else letters
    if not tail or None in tail:
        return False
    seq = "".join(tail)
    if any(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)):
        return False
    return seq[0] in "AB"

def apply_10g(dry: bool):
    rows = read_jsonl(ITEM_BANK)
    by_id = {r["item_id"]: r for r in rows}
    cands = read_jsonl(CANDIDATES)
    stats = Counter()
    touched_items = set()
    # Multiple candidates in one item -> split by descending block_idx so earlier block insertions don't shift later idx values
    def _bi(c):
        m = PATH_RE.match(c["block_path"])
        return int(m.group(2)) if m else -1
    cands_sorted = sorted(cands, key=lambda c: (c["item_id"], -_bi(c)))
    for c in cands_sorted:
        item = by_id.get(c["item_id"])
        if item is None:
            stats["item_missing"] += 1
            continue
        segs = c.get("suggested_segments") or []
        if len(segs) < 2 or not seq_ok(segs):
            stats["seq_reject"] += 1
            continue
        loc = locate(item, c["block_path"])
        if loc is None:
            stats["locate_fail"] += 1
            continue
        blocks, bi, pi = loc
        block = blocks[bi]
        para = block["para"]
        # Scope: only glued nodes that exclusively occupy their block (unambiguous split)
        if len(para) != 1 or pi != 0:
            stats["not_sole_node"] += 1
            continue
        node = para[0]
        if node.get("type") != "text" or "text" not in node:
            stats["not_text_or_done"] += 1
            continue
        if node["text"] != c["original_text"]:
            stats["text_mismatch_or_done"] += 1
            continue
        if _norm("".join(segs)) != _norm(c["original_text"]):
            stats["join_mismatch"] += 1
            continue
        # Execute: replace the original block with N independent blocks (one per option)
        new_blocks = [{"para": [{"type": "text", "text": s}]} for s in segs]
        blocks[bi:bi + 1] = new_blocks
        stats["split_applied"] += 1
        touched_items.add(c["item_id"])
    for iid in touched_items:
        by_id[iid]["batch10_source_10g"] = APPLY_ID
    stats["items_touched"] = len(touched_items)
    if not dry:
        write_jsonl(ITEM_BANK, rows)
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dry = not args.apply
    print(f"=== 10g option split {'APPLY' if not dry else 'DRY-RUN'} ===")
    st = apply_10g(dry)
    for k, v in sorted(st.items()):
        print(f"  {k}: {v}")
    print("=== DONE ===")

if __name__ == "__main__":
    main()
