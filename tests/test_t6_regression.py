#!/usr/bin/env python3
"""T6 回归测试 - 3 题完整链路 + 记忆写入验证"""

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
from adapters.memory import YihuierMemory

print("=" * 60)
print("  T6 回归测试 - 3 题完整链路")
print("=" * 60)

# ── 初始化 ──
retriever = YihuierRetriever(embeddings_dir=str(SKILL_DIR / "data" / "embeddings"))
print("✅ 检索引擎就绪")

api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key:
    print("❌ DEEPSEEK_API_KEY 未设置")
    sys.exit(1)

llm = LLMClient(provider="deepseek", model="deepseek-v4-pro", api_key=api_key)
print(f"✅ LLM 就绪 ({llm.provider} / {llm.model})")

base_system_prompt = (SKILL_DIR / "system_prompt.md").read_text()

# ── 记忆 ──
supabase_url = os.environ.get('SUPABASE_URL', '')
supabase_key = os.environ.get('SUPABASE_KEY', '')
memory = None
if supabase_url and supabase_key:
    try:
        memory = YihuierMemory(supabase_url, supabase_key)
        user_id = "chris_2026"
        print(f"✅ 记忆系统就绪 (user: {user_id})")
        static_memory = memory.get_static_memory_section(user_id)
    except Exception as e:
        print(f"⚠️ 记忆不可用: {e}")
        memory = None

if memory is None:
    static_memory = "[USER_PROFILE]\n（记忆功能未启用）"
    user_id = "chris_2026"

enhanced_system_prompt = base_system_prompt + "\n\n## 用户长期档案\n\n" + static_memory
print(f"✅ system_prompt 就绪 ({len(enhanced_system_prompt)} 字符)")

# ── 测试题目 ──
test_queries = [
    "N₂(g) + 3H₂(g) ⇌ 2NH₃(g)，K=0.5，向容器中再充入 N₂，平衡如何移动？K 值如何变化？转化率如何变化？",
    "高一新生：什么是物质的量？为什么发明这个概念？",
    "一化儿在哪个 BV 视频里讲过\"量子化学计算 DFT 在高考压轴题中的应用\"？请告诉我具体的 BV 号和 P 数。",
]

results = []
session_cost = 0.0

for i, q in enumerate(test_queries, 1):
    print(f"\n{'─' * 60}")
    print(f"  [{i}/3] {q[:60]}...")
    print(f"{'─' * 60}")

    # 诊断
    diagnosis = diagnose_query(q, retriever)

    # 动态记忆
    if memory:
        dynamic_memory = memory.get_dynamic_memory_section(user_id)
    else:
        dynamic_memory = "[RECENT_30_DAYS_HISTORY]\n（记忆功能未启用）"

    retrieval_text = format_retrieval_for_prompt(diagnosis)

    user_msg = f"""{dynamic_memory}

[RETRIEVAL_RESULTS]
{retrieval_text}

[USER_QUERY]
{q}"""

    # 调 LLM
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
    session_cost += cost

    # 约束检查
    lines = [l for l in content.split('\n') if l.strip()]
    first_line = lines[0] if lines else ''
    last_line = lines[-1] if lines else ''

    print(f"\n🎯 杰哥回答（前 200 字）:")
    print(content[:200] + "...")

    # 视频引用格式检查
    has_collection = '【' in content and ('】' in content)
    has_url = 'bilibili.com/video/' in content
    has_bvbv = 'BVBV' in content

    print(f"\n📏 约束检查:")
    print(f"   开场: {len(first_line)} 字 '{first_line[:30]}' {'✅' if len(first_line) <= 15 else '❌'}")
    print(f"   结尾: {len(last_line)} 字 '{last_line[:30]}' {'✅' if len(last_line) <= 20 else '❌'}")
    print(f"   合集名: {'✅' if has_collection else '⚠️ 未见【】'}")
    print(f"   URL: {'✅' if has_url else '⚠️ 无链接'}")
    print(f"   BVBV: {'❌ BVBV!' if has_bvbv else '✅ 无重复前缀'}")

    print(f"\n💰 本次: ¥{cost:.4f} | 输入: {usage['input_tokens']} (缓存命中: {usage['cache_hit_tokens']}) | 输出: {usage['output_tokens']}")
    print(f"🔍 模型: {response['model_returned']}")

    results.append({
        'query': q[:60],
        'cost': cost,
        'cache_hit': usage['cache_hit_tokens'],
        'input': usage['input_tokens'],
        'output': usage['output_tokens'],
        'first_len': len(first_line),
        'last_len': len(last_line),
        'has_collection': has_collection,
        'has_url': has_url,
        'has_bvbv': has_bvbv,
        'model': response['model_returned'],
    })

    # 保存到记忆
    if memory:
        try:
            memory.save_query(
                user_id=user_id,
                query=q,
                diagnosis=diagnosis,
                response=content[:500],
                cost=cost,
            )
            print(f"💾 记忆写入: ✅")
        except Exception as e:
            print(f"💾 记忆写入: ❌ {e}")

    # 更新弱点
    if memory and diagnosis.get('missing_prereqs'):
        try:
            memory.update_weak_topics(user_id, diagnosis['missing_prereqs'])
            print(f"📝 弱点更新: {diagnosis['missing_prereqs'][:3]}")
        except Exception as e:
            print(f"📝 弱点更新: ❌ {e}")

    # 查本月成本
    if memory:
        from datetime import datetime
        month = datetime.now().strftime("%Y-%m")
        month_cost = memory.get_month_cost(user_id, month)
        print(f"📊 本月累计成本: ¥{month_cost:.4f}")

# ── Profile 验证 ──
print(f"\n{'=' * 60}")
print(f"  Profile 验证")
print(f"{'=' * 60}")
if memory:
    profile = memory.get_user_profile(user_id)
    print(f"  年级: {profile.get('grade')} {'✅' if profile.get('grade') == '高二' else '❌ 预期 高二'}")
    print(f"  学校: {profile.get('school', '未设置')}")
    print(f"  弱点: {profile.get('weak_topics', [])}")
    print(f"  已掌握: {profile.get('mastered_topics', [])}")
else:
    print("  ⚠️ 记忆未启用")

# ── 汇总 ──
print(f"\n{'=' * 60}")
print(f"  📊 3 题回归测试汇总")
print(f"{'=' * 60}")

print(f"\n{'#':<3} {'Query':<55} {'成本':>8} {'缓存':>6} {'开头':>4} {'结尾':>4} {'合集':>4} {'URL':>4} {'BVBV':>4}")
print("-" * 100)
for i, r in enumerate(results, 1):
    print(f"{i:<3} {r['query']:<55} ¥{r['cost']:.4f} {r['cache_hit']:>4} {r['first_len']:>4} {r['last_len']:>4} "
          f"{'✅' if r['has_collection'] else '⚠️':>4} {'✅' if r['has_url'] else '⚠️':>4} "
          f"{'❌' if r['has_bvbv'] else '✅':>4}")

total_cost = sum(r['cost'] for r in results)
print(f"\n📈 总成本: ¥{total_cost:.4f} ({len(results)} 题)")

all_open = all(r['first_len'] <= 15 for r in results)
all_close = all(r['last_len'] <= 20 for r in results)
all_v4 = all('v4-pro' in r['model'] for r in results)
all_no_bvbv = all(not r['has_bvbv'] for r in results)
cache_trend = [r['cache_hit'] for r in results]

print(f"\n🔍 全局检查:")
print(f"  开场 ≤ 15 字: {'✅ 全部通过' if all_open else '❌'}")
print(f"  结尾 ≤ 20 字: {'✅ 全部通过' if all_close else '❌'}")
print(f"  模型一致性: {'✅ 全部 v4-pro' if all_v4 else '❌'}")
print(f"  无 BVBV 重复: {'✅' if all_no_bvbv else '❌'}")
print(f"  缓存趋势: {cache_trend} {'✅ 递增' if cache_trend == sorted(cache_trend) else '⚠️'}")
print(f"  成本: ¥{total_cost:.4f} (预期 ¥0.025-0.040)")

print(f"\n✅ T6 回归测试完成")
print(f"\n📋 请 Chris 在 Supabase 后台执行:")
print(f"   SELECT user_id, query, cost_yuan, created_at FROM query_history ORDER BY created_at DESC;")
print(f"   → 预期至少 3 行（本次测试的 3 题）")
