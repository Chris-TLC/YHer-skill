#!/usr/bin/env python3
"""
AI 自动切题脚本（题库入库的前置步骤）。

把整张试卷（docx/pdf/doc）→ AI 切成单题 + 对应答案 → 输出 raw_papers 格式，
再喂给 process_papers.py 入库。

流程：
  读整张卷文本 → AI 切题（题号/题干/小问/答案/分值）→ 写 data/raw_papers/{name}.jsonl

用法：
  # 切一个文件夹里所有卷子
  python3 scripts/split_papers.py --dir "/path/to/上海化学卷合集/05.2008-2026·（上海）化学高考真题"
  # 切单个文件
  python3 scripts/split_papers.py --file "xxx.docx"
  # 先试 N 个文件
  python3 scripts/split_papers.py --dir xxx --limit 2
  # 只看读取效果不调AI
  python3 scripts/split_papers.py --file xxx.docx --dry-run
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

from scripts.paper_reader import read_paper

RAW_DIR = SKILL_DIR / "data" / "raw_papers"
PAPER_EXT = {".docx", ".pdf", ".doc"}

# 一次喂给 AI 的最大字符。注意：输出比输入长（含答案结构化），
# 解析版1万字会让输出超8000token，所以分小块，每块输出不超限。
MAX_CHARS_PER_CALL = 5000


SPLIT_SYSTEM = """你是高考化学试卷结构化专家。给你一张化学试卷的纯文本（可能含题干和答案/解析），
你要把它切成一道道独立的题，并把答案对应到每道题。

规则：
1. 按题号切题（1. 2. 3. ... 选择题和非选择题都要）。
2. 选择题：题干+选项ABCD 合在 stem，答案放 answer。
3. 非选择大题：每个小问(1)(2)... 拆进 sub_questions，各自配答案和分值。
4. 化学式保持纯文本（如 N2+3H2⇌2NH3、SO4^2-）。
5. 如果原文只有题没有答案，answer 留空字符串。
6. 跳过完全无意义的乱码/页眉页脚。
严格输出 JSON，不要解释。"""


def build_split_prompt(paper_text: str, source: str, region: str, year: int) -> str:
    return f"""试卷来源：{source}（{region}，{year}年）

试卷纯文本：
\"\"\"
{paper_text}
\"\"\"

请切成单题，严格输出 JSON（不要任何解释文字）：
{{
  "items": [
    {{
      "source": "{source} T1",
      "region": "{region}",
      "year": {year},
      "stem": "题干（选择题含ABCD选项）",
      "sub_questions": [
        {{"sub_id": "(1)", "stem": "小问题干", "answer": "标准答案（无则空）", "score": 0}}
      ]
    }}
  ]
}}
说明：选择题 sub_questions 可只放一项 {{"sub_id":"", "stem":"", "answer":"正确选项如C", "score":2}}。
非选择大题按小问拆。answer 尽量从原文的【答案】【解析】部分对应提取。"""


def parse_meta_from_name(filename: str):
    """从文件名提取 来源/卷别/年份。"""
    name = filename
    year = 2024
    m = re.search(r"(20\d{2})", name)
    if m:
        year = int(m.group(1))
    region = "上海卷" if "上海" in name else "上海卷"
    source = re.sub(r"\.(docx|pdf|doc)$", "", name)
    source = re.sub(r"[（(](空白卷|解析卷|原卷版|解析版|考试版|参考答案|含解析)[）)]", "", source)
    source = re.sub(r"^精品解析[：:]\s*", "", source).strip()
    return source, region, year


def chunk_text(text: str, max_chars: int):
    """太长的卷分块（按题号边界尽量不切断题）。"""
    if len(text) <= max_chars:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > max_chars and cur:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur:
        chunks.append(cur)
    return chunks


def _extract_json(text: str) -> dict:
    """提取 JSON。若被 max_tokens 截断，抢救已完整的 items。"""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # 截断抢救：逐个抽出完整的 item 对象
    items = _salvage_items(text)
    return {"items": items} if items else {}


def _salvage_items(text: str) -> list:
    """从可能被截断的文本里，用括号配对抽出每个完整的 {...} item。"""
    items = []
    start = text.find('"items"')
    if start < 0:
        return items
    i = text.find("[", start)
    if i < 0:
        return items
    depth, obj_start = 0, -1
    for j in range(i, len(text)):
        ch = text[j]
        if ch == "{":
            if depth == 0:
                obj_start = j
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                frag = text[obj_start:j + 1]
                try:
                    items.append(json.loads(frag))
                except Exception:
                    pass
                obj_start = -1
    return items


def split_one_paper(path: Path, llm, dry_run=False):
    text, status = read_paper(path)
    if status != "ok":
        return None, status, 0.0

    source, region, year = parse_meta_from_name(path.name)
    if dry_run:
        print(f"    [{status}] {len(text)}字 | 来源={source} {region} {year}")
        print(f"    前200字: {text[:200]}")
        return [], "dry", 0.0

    all_items, cost = [], 0.0
    for chunk in chunk_text(text, MAX_CHARS_PER_CALL):
        prompt = build_split_prompt(chunk, source, region, year)
        resp = llm(SPLIT_SYSTEM, prompt)
        cost += resp.get("cost_yuan", 0)
        parsed = _extract_json(resp["content"])
        items = parsed.get("items", [])
        all_items.extend(items)
    return all_items, "ok", cost


def main():
    ap = argparse.ArgumentParser(description="AI 自动切题")
    ap.add_argument("--dir", help="处理整个文件夹")
    ap.add_argument("--file", help="处理单个文件")
    ap.add_argument("--out", default="", help="输出 jsonl 名(默认按来源)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--provider", default="deepseek")
    args = ap.parse_args()

    # 收集文件，优先解析版（含答案）
    files = []
    if args.file:
        files = [Path(args.file)]
    elif args.dir:
        d = Path(args.dir)
        all_papers = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in PAPER_EXT]
        # 优先解析版/答案版（含答案）；同一份卷只取一个
        with_answer = [p for p in all_papers if ("解析" in p.name or "答案" in p.name)]
        files = with_answer if with_answer else all_papers
    else:
        ap.print_help()
        return

    if args.limit:
        files = files[: args.limit]
    print(f"准备切 {len(files)} 张卷子")

    llm = None
    if not args.dry_run:
        from core.tutor.llm_bridge import make_llm_caller
        key = os.getenv(f"{args.provider.upper()}_API_KEY", "")
        if not key:
            print(f"缺 {args.provider.upper()}_API_KEY")
            sys.exit(1)
        llm = make_llm_caller(args.provider, api_key=key, max_tokens=8000)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.out or (Path(args.dir).name if args.dir else "papers")
    out_name = re.sub(r"[^\w一-鿿]", "_", out_name)[:40]
    out_file = RAW_DIR / f"{out_name}.jsonl"

    total_items, total_cost, ok, skip = 0, 0.0, 0, 0
    with open(out_file, "a", encoding="utf-8") as fout:
        for i, p in enumerate(files, 1):
            try:
                items, status, cost = split_one_paper(p, llm, dry_run=args.dry_run)
                if status in ("empty", "unreadable", "unsupported"):
                    skip += 1
                    print(f"  {i}. ⊘ 跳过({status}): {p.name[:45]}")
                    continue
                if args.dry_run:
                    continue
                for it in items:
                    fout.write(json.dumps(it, ensure_ascii=False) + "\n")
                fout.flush()
                total_items += len(items)
                total_cost += cost
                ok += 1
                print(f"  {i}. ✓ {len(items)}题 ¥{cost:.3f} | {p.name[:40]}")
            except Exception as e:
                skip += 1
                print(f"  {i}. ✗ {p.name[:40]}: {e}")
            time.sleep(0.3)

    if not args.dry_run:
        print(f"\n完成：{ok}卷成功，{skip}跳过，共 {total_items} 题，成本 ¥{total_cost:.2f}")
        print(f"输出：{out_file}")
        print(f"下一步：python3 scripts/process_papers.py --input {out_file} --topic {out_name}")


if __name__ == "__main__":
    main()
