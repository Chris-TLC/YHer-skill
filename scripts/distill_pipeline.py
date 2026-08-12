"""
Phase 0A: 知识蒸馏 Pipeline（修复版）
输出格式改为 JSONL（每题一行），避免化学符号破坏整批JSON
"""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from adapters.llm_client import LLMClient

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
QUESTIONS_FILE = Path(__file__).parent.parent / "data/raw_papers/shanghai_all.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data/distilled"
MAX_WORKERS = 10

# 改为 JSONL 格式（每题单独一行），避免化学符号破坏整批
DISTILL_PROMPT = """\
你是上海高考化学专家。分析以下{n}道选择题，每道题输出一行JSON（不要数组，每题单独一行）。

【题目】
{questions}

输出格式（每行一个JSON对象，注意：字符串中不要使用双引号，用「」代替引号）：
{{"idx":1,"id":"source字符串","kps":["知识点1","知识点2"],"diff":"T1","type":"概念辨析","path":[{{"s":1,"t":"看到X想到Y","k":"知识点"}}],"traps":["陷阱1"],"ans":"A"}}
{{"idx":2,...}}

规则：
- diff: T1基础/T2中档/T3拔高/T4压轴
- type: 概念辨析/计算推理/实验判断/综合推断
- path: 解题步骤，每步"看到X想到Y"格式，不超过3步
- 字符串中的化学式用文字描述（如"碳酸钠"而非"Na2CO3"），避免特殊字符"""


def load_clean_questions(filepath: str) -> list:
    data = []
    with open(filepath) as f:
        for line in f:
            q = json.loads(line)
            stem = q.get('stem', '')
            if (not q.get('incomplete')
                    and not q.get('has_visual')
                    and not q.get('sub_questions')
                    and len(stem) > 40
                    and any(f'{x}.' in stem or f'{x}．' in stem for x in 'ABCD')
                    and not stem.startswith(('【答案】', '根据', '故选'))):
                data.append(q)
    return data


def _parse_jsonl_lines(text: str) -> list:
    """逐行解析，跳过解析失败的行"""
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            # 尝试修复：把「」替换回双引号再试
            fixed = line.replace('「', '"').replace('」', '"')
            try:
                results.append(json.loads(fixed))
            except Exception:
                pass  # 跳过无法修复的行
    return results


def distill_batch(batch: list, client: LLMClient) -> tuple:
    questions_text = "\n\n".join(
        f"[{i+1}] source={q['source']} year={q['year']}\n{q['stem'][:350]}"
        for i, q in enumerate(batch)
    )
    messages = [{"role": "user",
                 "content": DISTILL_PROMPT.format(n=len(batch), questions=questions_text)}]

    result = client.chat(messages, max_tokens=3000, temperature=0.1)
    content = result['content'].strip()

    # 去掉 ``` 包裹
    if '```' in content:
        parts = content.split('```')
        content = '\n'.join(p.lstrip('json\n') for p in parts if '{' in p)

    parsed = _parse_jsonl_lines(content)

    # 为每个解析结果补充原题 stem
    src_map = {q['source']: q for q in batch}
    for item in parsed:
        orig = src_map.get(item.get('id', ''))
        if orig:
            item['year'] = orig['year']
            item['stem'] = orig['stem'][:250]

    return parsed, result['usage']


def run_full(batch_size: int = 15, max_workers: int = MAX_WORKERS):
    questions = load_clean_questions(QUESTIONS_FILE)
    batches = [questions[i:i+batch_size] for i in range(0, len(questions), batch_size)]

    print(f"\n{'='*55}")
    print(f"Phase 0A 蒸馏: {len(questions)} 题 / {len(batches)} 批")
    print(f"并发: {max_workers} 线程  批大小: {batch_size}")
    print('='*55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "all_distilled.jsonl"

    # 断点续跑
    done_ids = set()
    if out_file.exists():
        for line in out_file.open():
            r = json.loads(line)
            done_ids.add(r.get('id', ''))
        if done_ids:
            print(f"  续跑：跳过已完成 {len(done_ids)} 道")

    pending = [(i, b) for i, b in enumerate(batches)
               if not all(q['source'] in done_ids for q in b)]
    print(f"  待处理: {len(pending)} 批\n")

    def worker(args):
        idx, batch = args
        c = LLMClient(provider='deepseek', model='deepseek-chat', api_key=API_KEY)
        return idx, distill_batch(batch, c)

    done = 0
    total_saved = 0
    t0 = time.time()

    with open(out_file, 'a') as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(worker, a): a[0] for a in pending}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    _, (results, usage) = fut.result()
                    for r in results:
                        fout.write(json.dumps(r, ensure_ascii=False) + '\n')
                    fout.flush()
                    total_saved += len(results)
                    done += 1
                    elapsed = time.time() - t0
                    rate = done / elapsed * batch_size
                    eta = (len(pending) - done) * batch_size / max(rate, 0.1)
                    print(f"  [{done}/{len(pending)}] 批{idx:02d} "
                          f"保存{len(results)}条 | 合计{total_saved} | ETA {eta:.0f}s")
                except Exception as e:
                    done += 1
                    print(f"  [{done}/{len(pending)}] 批{idx:02d} ✗ {e}")

    print(f"\n完成！总保存: {total_saved} 道 | 耗时: {time.time()-t0:.0f}s")
    print(f"输出: {out_file}")


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_WORKERS
    run_full(max_workers=workers)
