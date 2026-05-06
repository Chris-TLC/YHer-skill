#!/usr/bin/env python3
"""
一化儿 AI 化学助手 - 命令行版
v3：完整记忆 always-on + 1M context 充分利用
"""

import sys
import os
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from core.retrieve import YihuierRetriever
from core.diagnose import diagnose_query
from core.format_answer import format_retrieval_for_prompt
from adapters.llm_client import LLMClient
from rich.console import Console
from rich.markdown import Markdown
import yaml
from dotenv import load_dotenv

# 加载 .env 文件（必须在 os.environ 读取之前）
load_dotenv(Path(__file__).parent.parent / ".env")

console = Console()


def load_config():
    config_path = SKILL_DIR / "config.yaml"
    if not config_path.exists():
        console.print("[red]❌ config.yaml 未找到，请从 config.example.yaml 复制[/red]")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def init_memory(config):
    """初始化记忆模块（可选，Supabase 未配置时返回 None）"""
    supabase_url = os.environ.get('SUPABASE_URL', '')
    supabase_key = os.environ.get('SUPABASE_KEY', '')

    if not supabase_url or not supabase_key:
        console.print("[yellow]⚠️  Supabase 未配置，记忆功能禁用[/yellow]")
        return None

    try:
        from adapters.memory import YihuierMemory
        memory = YihuierMemory(supabase_url, supabase_key)
        user_id = config.get('memory', {}).get('user_id', 'anonymous')
        console.print(f"✅ 记忆系统就绪 (user: {user_id})")
        return memory
    except Exception as e:
        console.print(f"[yellow]⚠️  记忆系统不可用: {e}[/yellow]")
        return None


def check_and_compress(memory, user_id, llm):
    """启动时检查是否需要压缩老记忆"""
    if memory is None:
        return
    try:
        status = memory.get_compression_status(user_id)
        if status.get('needs_compression'):
            console.print(f"[yellow]⚙️ 发现 {status['old_records']} 条 90+ 天老记忆，正在压缩...[/yellow]")
            result = memory.compress_old_memory(user_id, llm)
            if result.get('compressed', 0) > 0:
                console.print(
                    f"[green]✅ 压缩 {result['compressed']} 条 → "
                    f"季度档案 {result['periods']}，成本 ¥{result['cost']:.4f}[/green]"
                )
            else:
                console.print("[dim]无需压缩[/dim]")
        else:
            console.print(f"[dim]无需压缩（只有 {status.get('old_records', 0)} 条 90+ 天记录）[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠️  压缩检查失败: {e}[/yellow]")


def main():
    config = load_config()

    # ── 初始化 ──
    console.print("[bold blue]🧪 初始化中...[/bold blue]")

    retriever = YihuierRetriever(embeddings_dir=str(SKILL_DIR / "data" / "embeddings"))
    console.print("✅ 检索引擎就绪")

    api_key = os.environ.get(config['llm']['api_key_env'], '')
    if not api_key:
        console.print(f"[red]❌ 环境变量 {config['llm']['api_key_env']} 未设置[/red]")
        sys.exit(1)

    llm = LLMClient(
        provider=config['llm']['provider'],
        model=config['llm'].get('model'),
        api_key=api_key,
    )
    console.print(f"✅ LLM 就绪 ({config['llm']['provider']} / {llm.model})")

    memory = init_memory(config)
    user_id = config.get('memory', {}).get('user_id', 'anonymous')

    # 加载 system prompt
    sp_path = SKILL_DIR / "system_prompt.md"
    if not sp_path.exists():
        console.print("[red]❌ system_prompt.md 未找到[/red]")
        sys.exit(1)
    base_system_prompt = sp_path.read_text()

    # v3.1 缓存友好：静态记忆（USER_PROFILE + 季度摘要）拼入 system_prompt
    # DeepSeek prompt caching 按前缀匹配，system_prompt 稳定则缓存命中
    if memory:
        static_memory = memory.get_static_memory_section(user_id)
    else:
        static_memory = "[USER_PROFILE]\n（记忆功能未启用）"
    enhanced_system_prompt = base_system_prompt + "\n\n## 用户长期档案\n\n" + static_memory
    console.print(f"✅ system_prompt 就绪 ({len(enhanced_system_prompt)} 字符)")

    # 启动时检查压缩
    if memory and config.get('memory', {}).get('compression', {}).get('auto_run_on_startup', True):
        check_and_compress(memory, user_id, llm)

    # ── 主循环 ──
    console.print(f"\n[bold green]🎓 一化儿 AI 化学助手 v3[/bold green]")
    console.print(f"模型: {llm.model} | 完整记忆 always-on | {llm.config['context_window']:,} context")
    console.print("命令: 'exit' 退出 | 'profile' 查档案 | 'cost' 查月度成本\n")

    session_cost = 0.0
    session_queries = 0
    cache_hits = []

    while True:
        try:
            query = console.input("\n[cyan]问杰哥：[/cyan] ").strip()
            if not query:
                continue
            if query == 'exit':
                break

            if query == 'profile':
                if memory:
                    profile = memory.get_user_profile(user_id)
                    console.print(f"\n[bold]你的档案：[/bold]")
                    console.print(f"  年级: {profile.get('grade', '未设置')}")
                    console.print(f"  弱点: {profile.get('weak_topics', [])}")
                    console.print(f"  已掌握: {profile.get('mastered_topics', [])}")
                else:
                    console.print("[yellow]记忆功能未启用[/yellow]")
                continue

            if query == 'cost':
                from datetime import datetime
                month = datetime.now().strftime("%Y-%m")
                month_cost = memory.get_month_cost(user_id, month) if memory else 0
                console.print(f"\n[bold]💰 本月累计：¥{month_cost:.4f}[/bold]")
                console.print(f"本会话累计：¥{session_cost:.4f} ({session_queries} 次提问)")
                continue

            # ── 主流程 ──
            with console.status("[bold blue]杰哥思考中..."):
                # 1. 诊断
                diagnosis = diagnose_query(query, retriever)

                # 2. v3.1 动态记忆：仅最近 30 天历史（每次提问都可能变化）
                if memory:
                    dynamic_memory = memory.get_dynamic_memory_section(user_id)
                else:
                    dynamic_memory = "[RECENT_30_DAYS_HISTORY]\n（记忆功能未启用）"

                # 3. 格式化检索结果
                retrieval_text = format_retrieval_for_prompt(diagnosis)

                # 4. 构建 user message（动态内容，不破坏 system_prompt 缓存）
                user_msg = f"""{dynamic_memory}

[RETRIEVAL_RESULTS]
{retrieval_text}

[USER_QUERY]
{query}"""

                # 5. 调 LLM（system_prompt 含静态记忆，稳定 24h → 缓存命中）
                response = llm.chat(
                    messages=[
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=config['llm'].get('max_tokens', 2000),
                )

            # 6. 显示回答
            console.print("\n[bold green]🎯 杰哥：[/bold green]")
            console.print(Markdown(response['content']))

            # 7. 显示成本
            session_cost += response['cost_yuan']
            session_queries += 1
            usage = response['usage']
            cache_hits.append(usage['cache_hit_tokens'])
            console.print(
                f"\n[dim]💰 本次: ¥{response['cost_yuan']:.4f} | "
                f"会话累计: ¥{session_cost:.4f} | "
                f"输入: {usage['input_tokens']} ({usage['cache_hit_tokens']} 缓存命中) | "
                f"输出: {usage['output_tokens']}[/dim]"
            )

            # 8. 保存到记忆
            if memory:
                memory.save_query(
                    user_id=user_id,
                    query=query,
                    diagnosis=diagnosis,
                    response=response['content'][:500],
                    cost=response['cost_yuan'],
                )

                # 9. 自动更新弱点
                if diagnosis.get('missing_prereqs'):
                    memory.update_weak_topics(user_id, diagnosis['missing_prereqs'])

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]❌ 错误: {e}[/red]")
            import traceback
            traceback.print_exc()

    console.print(f"\n[green]再见，本会话总成本 ¥{session_cost:.4f} ({session_queries} 次提问)[/green]")
    if cache_hits:
        avg_hit = sum(cache_hits[1:]) / max(len(cache_hits) - 1, 1) if len(cache_hits) > 1 else 0
        console.print(f"[dim]缓存命中: 首次=0, 后续平均={avg_hit:.0f} tokens[/dim]")


if __name__ == "__main__":
    main()
