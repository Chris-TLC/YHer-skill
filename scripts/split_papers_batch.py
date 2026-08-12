#!/usr/bin/env python3
"""
并发批量视觉切题（速度优化版）。

视觉切题慢主要是等 API 返回（网络往返），不是 CPU 忙。
所以多张卷并发跑（卷之间并行，每卷内部逐页串行），速度可快 3-5 倍。
单线程 24 小时 → 并发约 5-7 小时。

用法：
  python3 scripts/split_papers_batch.py --dir "文件夹" --out shanghai --workers 5
  python3 scripts/split_papers_batch.py --dir "文件夹" --out shanghai --limit 40 --workers 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv

load_dotenv(SKILL_DIR / ".env")

from scripts.split_papers_vision import split_one_vision

RAW_DIR = SKILL_DIR / "data" / "raw_papers"
PROGRESS_FILE = SKILL_DIR / "data" / "split_progress.json"
PAPER_EXT = {".docx", ".pdf", ".doc"}

_lock = threading.Lock()
_convert_lock = threading.Lock()  # LibreOffice 转图串行锁（soffice 不能并发）
_state = {"done": 0, "total": 0, "items": 0, "cost": 0.0, "batch": "", "current": []}


def _write_progress(status="running"):
    with _lock:
        try:
            PROGRESS_FILE.write_text(json.dumps({
                "batch": _state["batch"],
                "done_papers": _state["done"], "total_papers": _state["total"],
                "total_items": _state["items"], "cost": round(_state["cost"], 4),
                "current": " | ".join(_state["current"][-3:]), "status": status,
                "updated": datetime.datetime.now().strftime("%H:%M:%S"),
            }, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def collect_files(d: Path, limit: int = 0):
    allp = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in PAPER_EXT]
    wa = [p for p in allp if "解析" in p.name or "答案" in p.name]
    files = wa if wa else allp
    # 去重：同一份卷只取一个（优先解析版）
    seen, uniq = {}, []
    for p in files:
        key = re.sub(r"[（(](空白卷|解析卷|原卷版|解析版|考试版|参考答案|含解析)[）)]", "", p.stem)
        key = re.sub(r"\s+", "", key).strip()
        if key not in seen:
            seen[key] = p
            uniq.append(p)
    return uniq[:limit] if limit else uniq


def main():
    ap = argparse.ArgumentParser(description="并发批量视觉切题")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vision", default="qwen-vl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5, help="并发卷数(建议3-6)")
    args = ap.parse_args()

    files = collect_files(Path(args.dir), args.limit)
    print(f"准备并发切 {len(files)} 张卷，{args.workers} 并发")

    from adapters.vision_client import VISION_CONFIGS, VisionClient
    cfg = VISION_CONFIGS[args.vision]
    key = os.getenv(cfg["env_key"], "")
    if not key:
        print(f"缺 {cfg['env_key']}"); sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_name = re.sub(r"[^\w一-鿿]", "_", args.out)[:40]
    out_file = RAW_DIR / f"{out_name}.jsonl"
    write_lock = threading.Lock()

    _state["total"] = len(files)
    _state["batch"] = out_name
    _write_progress("running")

    def process(path: Path):
        # 每个线程独立 client（避免共享连接问题）
        vision = VisionClient(args.vision, api_key=key)
        with _lock:
            _state["current"].append(path.name[:30])
        items, status, cost = split_one_vision(path, vision, convert_lock=_convert_lock)
        if status == "ok" and items:
            with write_lock, open(out_file, "a", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        with _lock:
            _state["done"] += 1
            if status == "ok":
                _state["items"] += len(items)
                _state["cost"] += cost
            try:
                _state["current"].remove(path.name[:30])
            except ValueError:
                pass
        _write_progress("running")
        return path.name, status, len(items) if items else 0, cost

    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, p): p for p in files}
        for fut in as_completed(futures):
            try:
                name, status, n, cost = fut.result()
                tag = "✓" if status == "ok" else "⊘"
                print(f"  [{_state['done']}/{len(files)}] {tag} {n}题 ¥{cost:.3f} | {name[:38]}")
                if status != "ok":
                    fail += 1
            except Exception as e:
                fail += 1
                print(f"  ✗ 异常: {e}")

    _write_progress("done")
    print(f"\n完成：{len(files)-fail}成功，{fail}跳过/失败，共 {_state['items']} 题，成本 ¥{_state['cost']:.2f}")
    print(f"输出：{out_file}")
    print(f"下一步入库：python3 scripts/process_papers.py --input {out_file} --topic {out_name}")


if __name__ == "__main__":
    main()
