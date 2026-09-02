#!/usr/bin/env python3
"""Batch12 (QA-3 滚动第一批) apply —— 待用户点名 L1。

审计口径(Claude 全实测,2026-07-05):
  12a 580 候选中剔除 19 条元素列举误拆(`C、N` 类)+ 50 条空选项段(`D.` 无内容)
      → apply 白名单 511 条(88%),拼接=原文 100% 独立复验过。
  12b 救回 226 张 PNG 全量像素扫过(剔 1 张 4x1 噪声 → 225 张);94 kept 转写垃圾/泄漏 0;
      OMML 43 条独立 KaTeX 复编译全过。
  ref_map 80 条 unresolvable 亲验为真(实体文件 WS1 就没有),不 apply,并入批次13 裁片靶。

写入目标:
  1. chemistry_v4_1_3329.jsonl —— 12a 511 条拆分(block 级,复用 10g 语义:
     in_block=块内节点重排后拆成多 block;tight_text=文本切分成多 block)
  2. ws2_repaired_assets/ —— 225 张救回图收编
  3. ws2_asset_transcripts_v1.jsonl —— 12b 救回资产的行升级:
     官方已有 pool=manual_queue 行 → kept 94 条升级为 ai_seed/display_only(带 transcript);
     其余救回但转写弱的仍 manual_queue(图已可直显,行加 batch12_rescued 标记)
  4. ws2_omml_latex_cache_v1.jsonl —— 43 条新增(键 omml_sha1,katex_ok=true)

安全: 拆分沿用 10g 验证过的手法(降序防位移/幂等 text 精确匹配/独占与非独占分治);
      转写行升级只改既有行字段;全部幂等。默认 dry-run。
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
ELEM = re.compile(r"^[A-D][、,]\s*[A-Z]")   # 元素列举误拆特征
EMPTY = re.compile(r"^[A-D][.．、]\s*$")     # 空选项段
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
    """审计白名单: 剔元素列举误拆 + 空选项段。"""
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
        # 同 field 内按 block_idx 降序(先拆后面的,前面索引不位移)
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
                # 独占 block → 拆成多 block(10g 同款)
                blocks[bidx:bidx + 1] = [{"para": [{"type": "text", "text": s}]} for s in segs]
            else:
                # 非独占: 节点位置拆成多 text 节点 + 从原 block 分离成独立 block 序列
                # 保守: 前段留原位,后续选项段成新 block 插到本 block 之后
                node["text"] = segs[0]
                new_blocks = [{"para": [{"type": "text", "text": s}]} for s in segs[1:]]
                # 若节点是块内最后一个节点,选项新 block 紧跟本 block
                if pidx == len(para) - 1:
                    blocks[bidx + 1:bidx + 1] = new_blocks
                else:
                    # 节点后还有内容(如图),把后续节点留在原块,新 block 插在其后仍会乱序
                    # → 回退: 不拆,记录(避免图文错序)
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
    # 1. 收编救回图(剔 4x1 噪声)
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

    # 2. 转写行升级(官方 manual_queue 行 → kept 升池 / 其余标记已救图)
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
    # 仍 manual 但图已救回的行: 打标(渲染端图可直显)
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
    print(f"=== Batch12 {'APPLY(真写)' if not dry else 'DRY-RUN'} ===")
    s1 = apply_split(dry)
    print("[12a 拆分]")
    for k, v in sorted(s1.items()): print(f"   {k}: {v}")
    s2 = apply_assets_and_transcripts(dry)
    print("[12b 图+转写]")
    for k, v in sorted(s2.items()): print(f"   {k}: {v}")
    s3, total = apply_omml(dry)
    print(f"[12b OMML] cache 行数 -> {total}")
    for k, v in sorted(s3.items()): print(f"   {k}: {v}")
    print("=== 完成 ===")

if __name__ == "__main__":
    main()
