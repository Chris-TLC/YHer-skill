"""
Phase 0A: Knowledge distillation pipeline (fixed version)
Output format changed to JSONL (one item per line) to prevent chemical symbols from breaking the entire batch JSON
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

# Changed to JSONL format (one item per line) to prevent chemical symbols from breaking the entire batch
DISTILL_PROMPT = """\
You are a Shanghai high school chemistry exam expert. Analyze the following {n} multiple-choice items; output one line of JSON per item (not an array, each item on its own line).

【Items】
{questions}

Output format (one JSON object per line; note: do not use double quotes inside strings, use 「」 instead):
{{"idx":1,"id":"source string","kps":["knowledge point 1","knowledge point 2"],"diff":"T1","type":"concept discrimination","path":[{{"s":1,"t":"seeing X think Y","k":"knowledge point"}}],"traps":["trap 1"],"ans":"A"}}
{{"idx":2,...}}

Rules:
- diff: T1 basic / T2 intermediate / T3 advanced / T4 capstone
- type: concept discrimination / calculation reasoning / experimental judgment / comprehensive inference
- path: solution steps, each step in "seeing X think Y" format, no more than 3 steps
- Describe chemical formulas in words (e.g. "sodium carbonate" rather than "Na2CO3") to avoid special characters"""


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
    """Parse line by line, skipping lines that fail to parse."""
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            # Attempt repair: replace 「」 back to double quotes and retry
            fixed = line.replace('「', '"').replace('」', '"')
            try:
                results.append(json.loads(fixed))
            except Exception:
                pass  # Skip lines that cannot be repaired
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

    # Strip ``` wrapper
    if '```' in content:
        parts = content.split('```')
        content = '\n'.join(p.lstrip('json\n') for p in parts if '{' in p)

    parsed = _parse_jsonl_lines(content)

    # Backfill the original item stem into each parsed result
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
    print(f"Phase 0A distillation: {len(questions)} items / {len(batches)} batches")
    print(f"Concurrency: {max_workers} threads  Batch size: {batch_size}")
    print('='*55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "all_distilled.jsonl"

    # Checkpoint resume
    done_ids = set()
    if out_file.exists():
        for line in out_file.open():
            r = json.loads(line)
            done_ids.add(r.get('id', ''))
        if done_ids:
            print(f"  Resume: skipping {len(done_ids)} already-completed items")

    pending = [(i, b) for i, b in enumerate(batches)
               if not all(q['source'] in done_ids for q in b)]
    print(f"  Pending: {len(pending)} batches\n")

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
                    print(f"  [{done}/{len(pending)}] batch {idx:02d} "
                          f"saved {len(results)} rows | total {total_saved} | ETA {eta:.0f}s")
                except Exception as e:
                    done += 1
                    print(f"  [{done}/{len(pending)}] batch {idx:02d} X {e}")

    print(f"\nComplete! Total saved: {total_saved} items | elapsed: {time.time()-t0:.0f}s")
    print(f"Output: {out_file}")


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_WORKERS
    run_full(max_workers=workers)
