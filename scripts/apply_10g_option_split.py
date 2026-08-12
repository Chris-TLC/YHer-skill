#!/usr/bin/env python3
"""10g 选项拆分 apply（QA-3 首件）—— 高置信子集 740 条。

审计: PROJECT_HANDOFF/BATCH10_AUDIT_2026-07-05.md P4。
范围: 仅服务池 stem_blocks 区、拆分段为合法严格递增选项序列(A<B<C<D)的候选。
      answer 区 82 条 + 结构可疑 46 条 + 非服务池,全部不入(留 QA-3 后续)。
语义: 把粘连成一个 text 节点的多选项(如 "A.淀粉B.二氧化硫"),按 block_path 定位后
      拆成 stem_blocks 里多个独立 block(每选项一个 block={"para":[{text}]}),
      前端每个 block = 一个 rir-para 各自换行 + rir-option 悬挂缩进。
      纯文本操作,不碰 media/ref_map(与 10c 题干回填的坑无关)。
范围收窄: 仅处理"粘连节点是其所在 block 唯一节点"的候选(拆法无歧义: 一 block 换 N block);
          节点与其他内容同 block 的 51 条留后续(拆点位置有歧义)。
安全: block_path 定位的节点 text 必须精确等于候选 original_text 才拆(防错位);
      拆分段拼接(去空白)必须等于原文(防吞字);零匹配则跳过。幂等: 已拆过跳过。

授权: 需用户点名 L1(题库主库写入)。默认 dry-run。
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
    """返回 (blocks_list, block_idx, para_idx) 或 None。"""
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
    # 同题多候选 → 按 block_idx 从大到小拆,避免前面插入 block 导致后面 idx 位移
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
        # 范围: 仅粘连节点独占 block(拆法无歧义)
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
        # 执行: 用 N 个独立 block(每选项一 block)替换原 block
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
    print(f"=== 10g 选项拆分 {'APPLY' if not dry else 'DRY-RUN'} ===")
    st = apply_10g(dry)
    for k, v in sorted(st.items()):
        print(f"  {k}: {v}")
    print("=== 完成 ===")

if __name__ == "__main__":
    main()
