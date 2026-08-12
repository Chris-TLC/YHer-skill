#!/usr/bin/env python3
"""QA-3 跨节点破损离子 apply —— 用户 2026-07-05「先 apply QA-3」(点名 chemistry_v4_1_3329.jsonl)。

审计: Claude 全实测审过 285 候选(化学自洽 17 组 head+cluster→ion 全对、
      SO3/SO4 不同图消歧、亲验 4 张 PNG、可逆 rev_fail=0/285、right_node 全不变)。
      Codex 的 scripts/apply_qa3_crossnode_ions.py 是候选生成器(--apply 硬拒绝),
      official 写入由本脚本(Claude)执行。

写入: chemistry_v4_1_3329.jsonl 的 285 个跨节点站点。每站点:
      left text 节点(以元素符号 head 结尾,如 "...c(SO") → replaced_left("...c(SO₃²⁻");
      紧跟的 formula 节点(不透明 WMF 下标电荷图)删除; right 节点不动。

安全:
  - 定位: 候选给死 (field, left_block_idx, formula_para_idx),formula=left+1。
  - 防错位: 只在 left.text 精确==original_left 且 formula 节点 type==formula 且
            media==候选 media 时才改。任一不符→跳过记 mismatch。
  - 防索引位移: 同一 (item,field,block) 内按 formula_para_idx 降序处理(先删靠后的,
                前面站点索引不受影响) —— 同 10g 降序手法。
  - 只验 left+formula,不验 right(跨站点共享节点时 right 可能已被相邻站点改)。
  - 幂等: left.text 已==replaced_left → 跳过。

默认 dry-run。--apply 真写(用户已 L1 授权)。
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

    # 按 item 分组 → 组内按 (field, block, formula_para_idx 降序)
    by_item = defaultdict(list)
    for c in cands:
        by_item[c["item_id"]].append(c)

    for iid, sites in by_item.items():
        item = by_id.get(iid)
        if item is None:
            stats["item_missing"] += len(sites)
            continue
        # 可重入幂等: 整个 item 已 apply(打标)则跳过 —— 因删节点会改 para 索引,
        # 二次用原始索引会错位,故用 item 级标记而非站点级精确匹配。
        if item.get("batch_source_qa3") == APPLY_ID:
            stats["skip_applied_item"] += 1
            stats["skip_applied_sites"] += len(sites)
            continue
        # 降序: 同 block 内先删靠后的 formula,前面站点索引不位移
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
            # 幂等: 已 apply
            if isinstance(left, dict) and left.get("text") == s["replaced_left"]:
                stats["skip_idempotent"] += 1
                continue
            # 防错位: left 必须精确匹配 original_left
            if not (isinstance(left, dict) and left.get("type") == "text" and left.get("text") == s["original_left"]):
                stats["mismatch_left"] += 1
                continue
            # 防错位: formula 节点必须存在、类型对、media 对
            if fi >= len(para):
                stats["formula_oob"] += 1
                continue
            formula = para[fi]
            if not (isinstance(formula, dict) and formula.get("type") == "formula" and str(formula.get("media") or "") == str(s.get("media") or "")):
                stats["mismatch_formula"] += 1
                continue
            # 执行: 回填 left + 删 formula 节点
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
    ap.add_argument("--apply", action="store_true", help="真写(用户 L1 授权后)")
    args = ap.parse_args()
    print(f"=== QA-3 跨节点离子 {'APPLY(真写)' if args.apply else 'DRY-RUN'} ===")
    st = apply_qa3(not args.apply)
    for k, v in sorted(st.items()):
        print(f"  {k}: {v}")
    print("=== 完成 ===")

if __name__ == "__main__":
    main()
