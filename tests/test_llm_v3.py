#!/usr/bin/env python3
"""v3 LLM 集成测试 - 调用 DeepSeek API，验证缓存命中和成本"""

import sys
import os
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")

from core.retrieve import YihuierRetriever
from core.diagnose import diagnose_query
from core.format_answer import format_retrieval_for_prompt
from adapters.llm_client import LLMClient

print("=" * 60)
print("  v3 LLM 集成测试")
print("=" * 60)

retriever = YihuierRetriever(embeddings_dir=str(SKILL_DIR / "data" / "embeddings"))
print("✅ 检索引擎就绪")

api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key:
    print("❌ DEEPSEEK_API_KEY 未设置")
    sys.exit(1)

llm = LLMClient(provider="deepseek", model="deepseek-v4-pro", api_key=api_key)
print(f"✅ LLM 就绪 ({llm.provider} / {llm.model})")

system_prompt = (SKILL_DIR / "system_prompt.md").read_text()
print(f"✅ system_prompt 加载 ({len(system_prompt)} 字)")

test_queries = [
    "我离子浓度比较老错",
    "盖斯定律怎么用",
    "工艺流程产率怎么算",
    "氧化还原配平好难",
    "有机推断的同分异构体",
]

results = []

for i, q in enumerate(test_queries, 1):
    print(f"\n{'─' * 60}")
    print(f"  [{i}/5] {q}")
    print(f"{'─' * 60}")

    # 诊断
    diagnosis = diagnose_query(q, retriever)

    # 记忆（mock）
    memory_text = "[USER_PROFILE]\n（记忆功能未启用 - 测试模式）"

    # 检索
    retrieval_text = format_retrieval_for_prompt(diagnosis)

    user_msg = f"""{memory_text}

[RETRIEVAL_RESULTS]
{retrieval_text}

[USER_QUERY]
{q}"""

    # 调 LLM
    response = llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=2000,
    )

    content = response['content']
    usage = response['usage']
    cost = response['cost_yuan']

    # 显示结果
    print(f"\n🎯 杰哥回答（前 200 字）:")
    print(f"   {content[:200]}...")

    print(f"\n💰 成本: ¥{cost:.4f}")
    print(f"📊 输入: {usage['input_tokens']} tokens "
          f"({usage['cache_hit_tokens']} 缓存命中)")
    print(f"📊 输出: {usage['output_tokens']} tokens")
    print(f"🔍 模型返回: {response['model_returned']}")

    # 约束检查
    first_line = content.split('\n')[0].strip() if content else ''
    last_line = content.split('\n')[-1].strip() if content else ''
    print(f"📏 开场: {len(first_line)} 字 '{first_line[:30]}' {'✅' if len(first_line) <= 15 else '❌'}")
    print(f"📏 结尾: {len(last_line)} 字 '{last_line[:30]}' {'✅' if len(last_line) <= 20 else '❌'}")
    print(f"📏 总字数: {len(content)} {'✅' if len(content) <= 1500 else '⚠️'}")

    results.append({
        'query': q,
        'cost': cost,
        'cache_hit': usage['cache_hit_tokens'],
        'input': usage['input_tokens'],
        'output': usage['output_tokens'],
        'total_chars': len(content),
        'first_len': len(first_line),
        'last_len': len(last_line),
    })

# ── 汇总 ──
print(f"\n{'=' * 60}")
print(f"  📊 测试汇总")
print(f"{'=' * 60}")

print(f"\n{'#':<3} {'Query':<25} {'成本':>8} {'缓存命中':>8} {'输入':>8} {'输出':>8} {'字数':>6}")
print("-" * 80)
for i, r in enumerate(results, 1):
    print(f"{i:<3} {r['query']:<25} ¥{r['cost']:.4f} {r['cache_hit']:>6} {r['input']:>8} {r['output']:>8} {r['total_chars']:>6}")

avg_cost = sum(r['cost'] for r in results) / len(results)
total_cost = sum(r['cost'] for r in results)
avg_cache_hit = sum(r['cache_hit'] for r in results[1:]) / max(len(results) - 1, 1)
avg_input = sum(r['input'] for r in results) / len(results)
first_cache = results[0]['cache_hit']

print(f"\n📈 统计:")
print(f"  总成本: ¥{total_cost:.4f}")
print(f"  平均单次: ¥{avg_cost:.4f} (预期 < ¥0.05)")
print(f"  首次缓存命中: {first_cache} tokens")
print(f"  后续平均缓存命中: {avg_cache_hit:.0f} tokens (> 1500 预期)")
print(f"  平均输入: {avg_input:.0f} tokens")
print(f"  缓存命中率: {(avg_cache_hit / avg_input * 100) if avg_input > 0 else 0:.0f}%")
print(f"  模型返回: {results[0] if results else 'N/A'}")

# 约束检查
open_ok = all(r['first_len'] <= 15 for r in results)
close_ok = all(r['last_len'] <= 20 for r in results)
print(f"\n🔍 约束检查:")
print(f"  开场 ≤ 15 字: {'✅ 全部通过' if open_ok else '❌ 有违规'}")
print(f"  结尾 ≤ 20 字: {'✅ 全部通过' if close_ok else '❌ 有违规'}")

print(f"\n✅ v3 LLM 集成测试完成")
