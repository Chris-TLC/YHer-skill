#!/usr/bin/env python3
"""
题库入库处理管线（总蓝图 B-1 的批量版）。

把"卷子+答案"自动处理成 item_bank 里带 [标准解+rubric+KG+题型] 的结构化真题。

5 步管线（每题）：
  ① 切题对答案（由你在原始 jsonl 里配好，或后续接 OCR）
  ② AI 自己解题，写出解题思路（理解题目内核）
  ③ 从标准答案+AI解题提炼 rubric 得分点 + 出题人陷阱
  ④ 自动挂 KG 节点 + 题型（关键词匹配，可后续接向量）
  ⑤ 写入 item_bank/{topic}.jsonl

输入格式（原始题 jsonl，你搜集卷子后整理成这样，一行一题）：
  {
    "source": "2024上海普陀一模T19",
    "region": "上海卷",
    "year": 2024,
    "stem": "题干（化学式纯文本，图表用文字描述）",
    "sub_questions": [{"sub_id":"(1)","stem":"...","answer":"标准答案","score":4}],
    "raw_answer": "整题的标准答案/解析全文（如果没拆小问就放这）"
  }

用法：
  # 处理一个原始题文件，输出到 item_bank
  python3 scripts/process_papers.py --input data/raw_papers/shanghai_2024.jsonl --topic shanghai
  # 干跑（不调LLM，只看切题对不对）
  python3 scripts/process_papers.py --input xxx.jsonl --dry-run
  # 限制处理前 N 题（测试用）
  python3 scripts/process_papers.py --input xxx.jsonl --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv

load_dotenv(SKILL_DIR / ".env")

from core.data.knowledge_repository import get_knowledge_repository

ITEM_BANK_DIR = SKILL_DIR / "data" / "item_bank"
RAW_DIR = SKILL_DIR / "data" / "raw_papers"


# ── 处理单题的 LLM prompt ──────────────────────────────────
PROCESS_SYSTEM = """你是高考化学命题与阅卷专家。给你一道化学题和它的标准答案，你要：
1. 自己独立解一遍，理解这道题考什么、坑在哪。
2. 提炼出阅卷得分点（rubric），每个得分点配可匹配的关键词。
3. 标出哪些得分点是 must_have（不拿到就算没掌握核心）。
4. 识别出题人埋的陷阱。
严谨、应试、不啰嗦。输出严格 JSON。"""


def build_process_prompt(item: dict, kg_node_ids: list, pattern_ids: list) -> str:
    answer_text = item.get("raw_answer", "")
    if item.get("sub_questions"):
        answer_text = "\n".join(
            f"{sq.get('sub_id','')}: {sq.get('answer','')} ({sq.get('score','')}分)"
            for sq in item["sub_questions"]
        )
    return f"""题目来源：{item.get('source','')}
题干：
{item.get('stem','')}

标准答案/解析：
{answer_text}

可选知识点（挑最相关的1-3个）：{kg_node_ids}
可选题型（挑最相关的1个）：{pattern_ids}

请输出严格 JSON（不要任何解释文字）：
{{
  "kg_nodes": ["挂到的知识点"],
  "question_type": "挂到的题型id",
  "difficulty": "T1/T2/T3/T4",
  "ai_solution_thinking": "你自己解这题的核心思路（100-200字，提炼内核）",
  "standard_solution": {{
    "final_answers": ["每小问最终答案"],
    "solution_steps": [{{"step":1,"action":"做什么","result":"得到什么","formula":"用的公式/原理"}}],
    "key_insight": "这题最核心的得分关键/最大陷阱（一句话）"
  }},
  "rubric": [
    {{"point_id":"sp1","desc":"得分点描述","keywords":["可匹配关键词"],"score":2.0,"must_have":true,"kg_node":"对应知识点"}}
  ],
  "traps": ["出题人埋的陷阱"]
}}"""


def process_one(item: dict, llm_call, kg_node_ids, pattern_ids) -> dict:
    """处理一道题，返回完整 ExamItem。"""
    from core.tutor.prompts import extract_tagged_json  # noqa

    prompt = build_process_prompt(item, kg_node_ids, pattern_ids)
    resp = llm_call(PROCESS_SYSTEM, prompt)
    raw = resp["content"]
    # 提取 JSON（可能裹在 ```json 里）
    parsed = _extract_json(raw)
    if not parsed:
        return {"_error": "JSON解析失败", "_raw": raw[:200], "cost": resp.get("cost_yuan", 0)}

    # 组装成 item_bank 格式
    iid = _make_item_id(item, parsed)
    out = {
        "item_id": iid,
        "source": item.get("source", ""),
        "region": item.get("region", "上海卷"),
        "year": item.get("year", 2024),
        "difficulty": parsed.get("difficulty", "T2"),
        "question_type": parsed.get("question_type", ""),
        "kg_nodes": parsed.get("kg_nodes", []),
        "stem": item.get("stem", ""),
        "sub_questions": item.get("sub_questions", []),
        "standard_solution": parsed.get("standard_solution", {}),
        "rubric": parsed.get("rubric", []),
        "ai_solution_thinking": parsed.get("ai_solution_thinking", ""),
        "yihuier_chunk_ids": [],
        "videos": _videos_from_kg(parsed.get("kg_nodes", [])),
        "traps": parsed.get("traps", []),
    }
    out["_cost"] = resp.get("cost_yuan", 0)
    return out


def _videos_from_kg(node_ids: list) -> list:
    """从挂到的 KG 节点继承推荐视频（引流出口）。"""
    repo = get_knowledge_repository()
    vids = []
    for nid in node_ids:
        node = repo.find_node(nid)
        if node and node.videos:
            v = node.videos[0]
            vids.append({"bv": v.bv, "p_number": v.p_number,
                         "completion_criterion": v.completion_criterion})
    return vids[:2]


def _extract_json(text: str) -> dict:
    import re
    # 去掉 ```json ``` 包裹
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _make_item_id(item: dict, parsed: dict) -> str:
    import hashlib
    raw = f"{item.get('source','')}{item.get('stem','')[:50]}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:8]
    region = "sh" if "上海" in item.get("region", "") else "cn"
    return f"{region}-{item.get('year',2024)}-{h}"


def main():
    ap = argparse.ArgumentParser(description="题库入库处理管线")
    ap.add_argument("--input", required=True, help="原始题 jsonl 文件")
    ap.add_argument("--topic", default="", help="输出到 item_bank/{topic}.jsonl，默认按文件名")
    ap.add_argument("--dry-run", action="store_true", help="不调LLM，只看切题")
    ap.add_argument("--limit", type=int, default=0, help="只处理前N题(测试)")
    ap.add_argument("--provider", default="deepseek")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"输入文件不存在: {inp}")
        sys.exit(1)

    items = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]
    print(f"读取 {len(items)} 道原始题")

    repo = get_knowledge_repository()
    kg_ids = [n.node_id for n in repo.all_nodes()]
    pat_ids = [p.pattern_id for p in repo.all_patterns()]

    if args.dry_run:
        for it in items:
            print(f"  [{it.get('source','?')}] 题干前60字: {it.get('stem','')[:60]}")
        print("（dry-run：未调用 LLM）")
        return

    from core.tutor.llm_bridge import make_llm_caller
    key = os.getenv(f"{args.provider.upper()}_API_KEY", "")
    if not key:
        print(f"未找到 {args.provider.upper()}_API_KEY")
        sys.exit(1)
    llm = make_llm_caller(args.provider, api_key=key, max_tokens=2500)

    topic = args.topic or inp.stem
    ITEM_BANK_DIR.mkdir(parents=True, exist_ok=True)
    out_file = ITEM_BANK_DIR / f"{topic}.jsonl"

    import datetime
    progress_file = ITEM_BANK_DIR.parent / "ingest_progress.json"

    def _write_ingest(done, status="running"):
        try:
            progress_file.write_text(json.dumps({
                "done": done, "total": len(items), "cost": round(total_cost, 4),
                "status": status,
                "updated": datetime.datetime.now().strftime("%H:%M:%S"),
            }, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    total_cost, ok, fail = 0.0, 0, 0
    _write_ingest(0, "running")
    with open(out_file, "a", encoding="utf-8") as fout:
        for i, it in enumerate(items, 1):
            try:
                result = process_one(it, llm, kg_ids, pat_ids)
                if result.get("_error"):
                    fail += 1
                    print(f"  {i}. ✗ {it.get('source','?')}: {result['_error']}")
                    _write_ingest(i, "running")
                    continue
                total_cost += result.pop("_cost", 0)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
                print(f"  {i}. ✓ {result['item_id']} → 挂[{','.join(result['kg_nodes'][:2])}] "
                      f"{len(result['rubric'])}得分点 (¥{total_cost:.3f})")
            except Exception as e:
                fail += 1
                print(f"  {i}. ✗ {it.get('source','?')}: {e}")
            _write_ingest(i, "running")
            time.sleep(0.3)  # 轻微限速
    _write_ingest(len(items), "done")

    print(f"\n完成：{ok} 成功，{fail} 失败，成本 ¥{total_cost:.2f}")
    print(f"输出：{out_file}")
    print("下一步：python3 scripts/import_items.py --validate 校验格式")


if __name__ == "__main__":
    main()
