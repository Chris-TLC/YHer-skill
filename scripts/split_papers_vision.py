#!/usr/bin/env python3
"""
视觉版切题脚本（根治公式图片问题）。

把试卷每页转成图片，用多模态视觉模型"看"整页，
识别方程式/结构式/装置图/坐标图（转成文字/LaTeX 描述），切题对答案。

这是文本版 split_papers.py 的视觉增强版，专治"公式是图片提取不出"。

用法：
  # 需要先开通视觉模型 key（推荐通义 qwen-vl，填进 .env 的 DASHSCOPE_API_KEY）
  python3 scripts/split_papers_vision.py --file "xxx.pdf" --vision qwen-vl
  python3 scripts/split_papers_vision.py --dir "文件夹" --vision qwen-vl --limit 2
  # 只转图不调视觉(看转图效果)
  python3 scripts/split_papers_vision.py --file xxx.pdf --images-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv

load_dotenv(SKILL_DIR / ".env")

from scripts.page_to_image import paper_to_images
from scripts.split_papers import _extract_json, parse_meta_from_name

RAW_DIR = SKILL_DIR / "data" / "raw_papers"
PAPER_EXT = {".docx", ".pdf", ".doc"}


VISION_SYSTEM = """你是高考化学试卷识别专家。给你试卷的一页图片，你要：
1. 识别页面上所有题目：题号、题干、选项、小问。
2. 关键：准确读出图片里的化学方程式、结构式、电子式、装置图、坐标图、表格，
   转成文字或 LaTeX 描述（如装置图描述成"圆底烧瓶接冷凝管"，方程式写成 2SO2+O2⇌2SO3）。
3. 如果页面有【答案】【解析】，把答案对应到题。
4. 跨页的题（这页没结束），标 incomplete=true。
严格输出 JSON。"""


def build_vision_prompt(source, region, year, page_no):
    return f"""这是《{source}》（{region}，{year}年）第 {page_no} 页。

请识别本页所有题目，严格输出 JSON：
{{
  "items": [
    {{
      "source": "{source} T题号",
      "region": "{region}", "year": {year},
      "stem": "完整题干(含选项;方程式/结构式/装置图转成文字或LaTeX描述)",
      "sub_questions": [{{"sub_id":"(1)","stem":"小问","answer":"答案(无则空)","score":0}}],
      "has_visual": true,
      "incomplete": false
    }}
  ]
}}
特别注意：图片里的化学方程式、结构式、装置图必须读出来转成文字，这是最重要的。"""


def split_one_vision(path: Path, vision, dry_images=False, convert_lock=None):
    """convert_lock: 并发时传入，保证 LibreOffice 转图串行（soffice 不能并发）。"""
    source, region, year = parse_meta_from_name(path.name)
    try:
        if convert_lock is not None:
            with convert_lock:
                images = paper_to_images(path)
        else:
            images = paper_to_images(path)
    except Exception as e:
        return None, f"转图失败: {e}", 0.0
    if dry_images:
        return images, "images_only", 0.0

    all_items, cost = [], 0.0
    paper_start = time.time()
    PAPER_TIME_BUDGET = 900  # 单卷最多 15 分钟，超了跳过剩余页（防异常卷拖垮全量）
    for i, img in enumerate(images, 1):
        if time.time() - paper_start > PAPER_TIME_BUDGET:
            print(f"      ⏱ 单卷超 {PAPER_TIME_BUDGET//60} 分钟，跳过剩余 {len(images)-i+1} 页")
            break
        prompt = build_vision_prompt(source, region, year, i)
        try:
            resp = vision.read_page(img, VISION_SYSTEM, prompt)
            cost += resp["cost_yuan"]
            parsed = _extract_json(resp["content"])
            all_items.extend(parsed.get("items", []))
        except Exception as e:
            print(f"      第{i}页视觉识别失败: {e}")
    return dedup_items(all_items), "ok", cost


def _q_number(source: str) -> str:
    """从 source 提取题号 T1/T2... 作为去重键。"""
    m = re.search(r"[Tt题]\s*(\d+)", source or "")
    return m.group(1) if m else ""


def dedup_items(items: list) -> list:
    """
    按题号去重（跨页重复切题的修复）。
    同题号保留内容最全的那个（题干+答案总长最长）。无题号的全部保留。
    """
    by_num, no_num = {}, []
    for it in items:
        num = _q_number(it.get("source", ""))
        if not num:
            no_num.append(it)
            continue
        # 用题干+答案总长衡量"最全"
        def richness(x):
            ans = " ".join(sq.get("answer", "") for sq in x.get("sub_questions", []))
            return len(x.get("stem", "")) + len(ans)
        if num not in by_num or richness(it) > richness(by_num[num]):
            by_num[num] = it
    # 按题号排序
    ordered = [by_num[k] for k in sorted(by_num, key=lambda x: int(x))]
    return ordered + no_num


def _write_progress(path, batch, done_papers, total_papers, total_items, cost, current, status):
    """写进度状态文件，供看板实时读取。"""
    import datetime
    try:
        path.write_text(json.dumps({
            "batch": batch, "done_papers": done_papers, "total_papers": total_papers,
            "total_items": total_items, "cost": round(cost, 4),
            "current": current, "status": status,
            "updated": datetime.datetime.now().strftime("%H:%M:%S"),
        }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="视觉版切题")
    ap.add_argument("--dir")
    ap.add_argument("--file")
    ap.add_argument("--out", default="")
    ap.add_argument("--vision", default="qwen-vl",
                    help="视觉模型: qwen-vl/doubao-vision/glm-4v/gpt-4o")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images-only", action="store_true", help="只转图不调视觉")
    args = ap.parse_args()

    files = []
    if args.file:
        files = [Path(args.file)]
    elif args.dir:
        d = Path(args.dir)
        allp = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in PAPER_EXT]
        wa = [p for p in allp if "解析" in p.name or "答案" in p.name]
        files = wa if wa else allp
    else:
        ap.print_help(); return
    if args.limit:
        files = files[: args.limit]

    vision = None
    if not args.images_only:
        from adapters.vision_client import VISION_CONFIGS, VisionClient
        cfg = VISION_CONFIGS.get(args.vision)
        if not cfg:
            print(f"未知视觉模型 {args.vision}"); sys.exit(1)
        key = os.getenv(cfg["env_key"], "")
        if not key:
            print(f"⚠️  缺少视觉模型 key：请在 .env 填 {cfg['env_key']}")
            print(f"   开通地址：{cfg['key_link']}")
            print(f"   模型：{cfg['label']}")
            sys.exit(1)
        vision = VisionClient(args.vision, api_key=key)

    print(f"准备视觉切 {len(files)} 张卷（视觉模型: {args.vision}）")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_name = re.sub(r"[^\w一-鿿]", "_", args.out or (Path(args.dir).name if args.dir else "vision"))[:40]
    out_file = RAW_DIR / f"{out_name}.jsonl"

    total, cost, ok, skip = 0, 0.0, 0, 0
    progress_file = RAW_DIR.parent / "split_progress.json"
    _write_progress(progress_file, out_name, 0, len(files), 0, 0.0, "", "running")
    with open(out_file, "a", encoding="utf-8") as fout:
        for i, p in enumerate(files, 1):
            _write_progress(progress_file, out_name, i - 1, len(files), total, cost, p.name[:40], "running")
            items, status, c = split_one_vision(p, vision, dry_images=args.images_only)
            if args.images_only:
                print(f"  {i}. {len(items) if items else 0}页图 | {p.name[:40]}")
                continue
            if status != "ok":
                skip += 1
                print(f"  {i}. ⊘ {status}: {p.name[:40]}")
                continue
            for it in items:
                fout.write(json.dumps(it, ensure_ascii=False) + "\n")
            fout.flush()
            total += len(items); cost += c; ok += 1
            print(f"  {i}. ✓ {len(items)}题 ¥{c:.3f} | {p.name[:40]}")
            _write_progress(progress_file, out_name, i, len(files), total, cost, p.name[:40], "running")
            time.sleep(0.3)
    _write_progress(progress_file, out_name, len(files), len(files), total, cost, "", "done")

    if not args.images_only:
        print(f"\n完成：{ok}卷，{skip}跳过，{total}题，成本 ¥{cost:.2f}")
        print(f"输出：{out_file}")
        print(f"下一步：python3 scripts/process_papers.py --input {out_file} --topic {out_name}")


if __name__ == "__main__":
    main()
