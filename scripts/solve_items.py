#!/usr/bin/env python3
"""化学题库 DeepSeek 解题脚本（一阶段架构验证核心）。
================================================================
对切出来的 6083 道题, 用 DeepSeek V4-Pro 逐题生成:
  - standard_answer  最终答案(补齐切题漏掉的2517题答案)
  - solution_steps   解题步骤(教学用)
  - key_insight      本题核心突破口(诊断+讲题用)
  - knowledge_points 知识点标签
  - difficulty       难度 1-5
  - question_type    题型

输入: data/raw_papers/shanghai_all.jsonl  (切题产出)
输出: data/item_bank/chemistry_solved.jsonl
断点续传: 按 source+stem 哈希去重, 已解的跳过。

用法:
  python3 scripts/solve_items.py --limit 50    # 先试50题看质量
  python3 scripts/solve_items.py               # 全量6083题
"""
from __future__ import annotations
import os, sys, json, asyncio, hashlib, argparse, time
from pathlib import Path

SKILL = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL))
DATA = SKILL / "data"
IN_FILE = DATA / "raw_papers" / "shanghai_all.jsonl"
OUT_FILE = DATA / "item_bank" / "chemistry_solved.jsonl"

try:
    import httpx
except ImportError:
    print("需要 httpx: pip install --break-system-packages httpx")
    sys.exit(1)

# ---- DeepSeek key ----
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    for envf in (SKILL / ".env", SKILL.parent / ".env"):
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    DEEPSEEK_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")

ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-pro"   # 注意: 旧的 deepseek-chat/reasoner 别名 2026-07-24 退役

CONCURRENT = 8
MAX_RETRIES = 3
RETRY_DELAYS = [5, 20, 60]

# V4-Pro 正式价(促销已过期), ¥/百万token, 按美元×7汇率粗估
PRICE_INPUT_PER_M = 3.0
PRICE_OUTPUT_PER_M = 6.0

SYSTEM_PROMPT = """你是上海高考化学命题与教学专家。给你一道化学题(可能含或不含标准答案),你要:
1. 给出准确的最终答案(若题目自带答案,核对并采纳;若没有,你来解出正确答案)。
2. 给出清晰的解题步骤(面向学生教学,讲清每一步为什么)。
3. 提炼本题的核心突破口(key_insight): 这道题最关键的那个条件分析/思维转折点是什么,
   一句话点破——这是"诊断+讲题"的灵魂,学生卡住往往就卡在这里。
4. 标注知识点(上海化学考纲术语,如"原电池""化学平衡移动""阿伏伽德罗常数")。
5. 标注难度 1-5 (1=送分,3=中等,5=压轴)。
6. 标注题型(单选题/不定项选择题/填空题/实验题/计算题/推断题)。
严格输出JSON,不解释。"""


def build_prompt(item):
    sub = ""
    if item.get("sub_questions"):
        sub = "\n小问:\n" + "\n".join(
            f"  {sq.get('sub_id','')} {sq.get('stem','')}" +
            (f" [原答案:{sq['answer']}]" if sq.get('answer') else "")
            for sq in item["sub_questions"])
    existing = item.get("answer", "")
    return f"""题目来源: {item.get('source','')}
题干: {item.get('stem','')}{sub}
{f'题目自带答案: {existing}' if existing else '(题目未提供答案,请你解出)'}

请严格输出JSON:
{{
  "standard_answer": "最终答案(选择题填字母;计算题填数值+单位;每小问分行)",
  "solution_steps": ["步骤1", "步骤2", "..."],
  "key_insight": "一句话点破本题核心突破口",
  "knowledge_points": ["知识点1", "知识点2"],
  "difficulty": 3,
  "question_type": "单选题"
}}"""


def parse_json(text):
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def item_hash(item):
    h = (item.get("source", "") + "|" + item.get("stem", "")[:100])
    return hashlib.md5(h.encode()).hexdigest()[:16]


async def solve_one(client, item, sem, stats):
    async with sem:
        prompt = build_prompt(item)
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    ENDPOINT,
                    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2000,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=httpx.Timeout(180.0, connect=30.0),
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = parse_json(content)
                if not parsed:
                    raise ValueError("JSON解析失败")
                usage = data.get("usage", {})
                stats["in_tok"] += usage.get("prompt_tokens", 0)
                stats["out_tok"] += usage.get("completion_tokens", 0)
                # 合并: 原题字段 + 解题产出
                out = dict(item)
                out["item_id"] = item_hash(item)
                out["solved"] = parsed
                return out
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAYS[attempt])
                    continue
                stats["fail"] += 1
                return {"item_id": item_hash(item), "source": item.get("source", ""),
                        "error": str(e)[:120], "_orig": item}


async def main_async(limit):
    items = [json.loads(l) for l in open(IN_FILE) if l.strip()]
    done = set()
    if OUT_FILE.exists():
        for l in open(OUT_FILE):
            try:
                done.add(json.loads(l).get("item_id"))
            except Exception:
                pass
    pending = [it for it in items if item_hash(it) not in done]
    if limit:
        pending = pending[:limit]

    print(f"总题: {len(items)}  已解: {len(done)}  本次待解: {len(pending)}")
    print(f"模型: {MODEL}  并发: {CONCURRENT}")
    if not pending:
        print("全部已解完!")
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    stats = {"in_tok": 0, "out_tok": 0, "fail": 0}
    sem = asyncio.Semaphore(CONCURRENT)
    t0 = time.time()
    ok = 0

    async with httpx.AsyncClient() as client:
        # 分批写, 避免中途崩了丢进度
        BATCH = 20
        with open(OUT_FILE, "a", encoding="utf-8") as fout:
            for i in range(0, len(pending), BATCH):
                batch = pending[i:i + BATCH]
                results = await asyncio.gather(*[solve_one(client, it, sem, stats) for it in batch])
                for r in results:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    if "error" not in r:
                        ok += 1
                fout.flush()
                cost = stats["in_tok"] / 1e6 * PRICE_INPUT_PER_M + stats["out_tok"] / 1e6 * PRICE_OUTPUT_PER_M
                el = time.time() - t0
                rate = (i + len(batch)) / el * 60
                print(f"  [{i+len(batch)}/{len(pending)}] 成功{ok} 失败{stats['fail']} "
                      f"¥{cost:.2f} {rate:.0f}题/分", flush=True)

    cost = stats["in_tok"] / 1e6 * PRICE_INPUT_PER_M + stats["out_tok"] / 1e6 * PRICE_OUTPUT_PER_M
    print(f"\n✅ 完成: 成功{ok} 失败{stats['fail']}")
    print(f"💰 成本: ¥{cost:.2f} (in {stats['in_tok']} + out {stats['out_tok']} token)")
    print(f"📁 输出: {OUT_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not DEEPSEEK_API_KEY:
        print("⚠️ 缺少 DEEPSEEK_API_KEY (.env 或环境变量)")
        sys.exit(1)
    asyncio.run(main_async(args.limit))


if __name__ == "__main__":
    main()
