#!/usr/bin/env python3
"""v3.1 缓存验证测试 - 3 连续查询，验证 cache_hit_tokens > 3000 + 结尾 ≤ 20 字"""

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
print("  v3.1 缓存友好验证测试")
print("=" * 60)

retriever = YihuierRetriever(embeddings_dir=str(SKILL_DIR / "data" / "embeddings"))
print("✅ 检索引擎就绪")

api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key:
    print("❌ DEEPSEEK_API_KEY 未设置")
    sys.exit(1)

llm = LLMClient(provider="deepseek", model="deepseek-v4-pro", api_key=api_key)
print(f"✅ LLM 就绪 ({llm.provider} / {llm.model})")

base_system_prompt = (SKILL_DIR / "system_prompt.md").read_text()
print(f"✅ base system_prompt 加载 ({len(base_system_prompt)} 字符)")

# v3.1: 模拟静态记忆（无 Supabase 时用 mock）
mock_static_memory = """[USER_PROFILE]
grade: 高二
weak_topics: [盐类水解, 离子浓度比较]
mastered_topics: [氧化还原配平, 化学计量]

[HISTORICAL_SUMMARIES（季度高保真档案）]
（首次使用，暂无历史档案）"""

enhanced_system_prompt = base_system_prompt + "\n\n## 用户长期档案\n\n" + mock_static_memory
print(f"✅ enhanced_system_prompt 就绪 ({len(enhanced_system_prompt)} 字符)")

# v3.1: 动态记忆（mock）
mock_dynamic_memory = "[RECENT_30_DAYS_HISTORY]\n（最近 30 天无记录）"

test_queries = [
    "盐类水解我搞不懂",
    "盖斯定律怎么用",
    "氧化还原配平",
]

results = []

for i, q in enumerate(test_queries, 1):
    print(f"\n{'─' * 60}")
    print(f"  [{i}/3] {q}")
    print(f"{'─' * 60}")

    diagnosis = diagnose_query(q, retriever)
    retrieval_text = format_retrieval_for_prompt(diagnosis)

    user_msg = f"""{mock_dynamic_memory}

[RETRIEVAL_RESULTS]
{retrieval_text}

[USER_QUERY]
{q}"""

    response = llm.chat(
        messages=[
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=2000,
    )

    content = response['content']
    usage = response['usage']
    cost = response['cost_yuan']

    first_line = content.split('\n')[0].strip() if content else ''
    last_line = content.split('\n')[-1].strip() if content else ''

    print(f"\n🎯 杰哥回答（前 150 字）:")
    print(f"   {content[:150]}...")

    print(f"\n💰 成本: ¥{cost:.4f}")
    print(f"📊 输入: {usage['input_tokens']} tokens "
          f"(缓存命中: {usage['cache_hit_tokens']})")
    print(f"📊 输出: {usage['output_tokens']} tokens")
    print(f"🔍 模型返回: {response['model_returned']}")

    print(f"📏 开场: {len(first_line)} 字 '{first_line[:30]}' "
          f"{'✅' if len(first_line) <= 15 else '❌'}")
    print(f"📏 结尾: {len(last_line)} 字 '{last_line[:30]}' "
          f"{'✅' if len(last_line) <= 20 else '❌'}")

    results.append({
        'query': q,
        'cost': cost,
        'cache_hit': usage['cache_hit_tokens'],
        'input': usage['input_tokens'],
        'output': usage['output_tokens'],
        'first_len': len(first_line),
        'last_len': len(last_line),
        'model_returned': response['model_returned'],
    })

# ── 汇总 ──
print(f"\n{'=' * 60}")
print(f"  📊 缓存验证汇总")
print(f"{'=' * 60}")

print(f"\n{'#':<3} {'Query':<22} {'成本':>8} {'缓存命中':>10} {'输入':>8} {'输出':>8}")
print("-" * 72)
for i, r in enumerate(results, 1):
    print(f"{i:<3} {r['query']:<22} ¥{r['cost']:.4f} {r['cache_hit']:>8} "
          f"{r['input']:>8} {r['output']:>8}")

# 关键指标
print(f"\n📈 关键指标:")
print(f"  总成本: ¥{sum(r['cost'] for r in results):.4f}")

cache_hit_2 = results[1]['cache_hit'] if len(results) > 1 else 0
cache_hit_3 = results[2]['cache_hit'] if len(results) > 2 else 0
print(f"  查询 1 缓存命中: {results[0]['cache_hit']} tokens (首次，预期为 0)")
print(f"  查询 2 缓存命中: {cache_hit_2} tokens {'✅' if cache_hit_2 > 3000 else '❌ 预期 > 3000'}")
print(f"  查询 3 缓存命中: {cache_hit_3} tokens {'✅' if cache_hit_3 > 3000 else '❌ 预期 > 3000'}")

all_open_ok = all(r['first_len'] <= 15 for r in results)
all_close_ok = all(r['last_len'] <= 20 for r in results)
all_model_ok = all('v4-pro' in r['model_returned'] for r in results)

print(f"\n🔍 约束检查:")
print(f"  开场 ≤ 15 字: {'✅ 全部通过' if all_open_ok else '❌ 有违规'}")
print(f"  结尾 ≤ 20 字: {'✅ 全部通过' if all_close_ok else '❌ 有违规'}")
print(f"  模型一致性: {'✅ 全部 v4-pro' if all_model_ok else '❌ 有降级'}")

if cache_hit_2 > 3000 and cache_hit_3 > 3000 and all_open_ok and all_close_ok and all_model_ok:
    print(f"\n✅✅✅ v3.1 缓存优化验证通过！")
else:
    print(f"\n⚠️ 部分指标未达标，需检查")

print(f"\n✅ v3.1 缓存验证测试完成")
