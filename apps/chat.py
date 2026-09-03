#!/usr/bin/env python3
"""
Yihuier AI chemistry assistant - command-line version
v3: full always-on memory + full use of the 1M context window
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

# Load the .env file (must happen before any os.environ reads)
load_dotenv(Path(__file__).parent.parent / ".env")

console = Console()


def load_config():
    config_path = SKILL_DIR / "config.yaml"
    if not config_path.exists():
        console.print("[red]❌ config.yaml not found; please copy it from config.example.yaml[/red]")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def init_memory(config):
    """Initialize the memory module (optional; returns None when Supabase isn't configured)."""
    supabase_url = os.environ.get('SUPABASE_URL', '')
    supabase_key = os.environ.get('SUPABASE_KEY', '')

    if not supabase_url or not supabase_key:
        console.print("[yellow]⚠️  Supabase not configured; memory features disabled[/yellow]")
        return None

    try:
        from adapters.memory import YihuierMemory
        memory = YihuierMemory(supabase_url, supabase_key)
        user_id = config.get('memory', {}).get('user_id', 'anonymous')
        console.print(f"✅ memory system ready (user: {user_id})")
        return memory
    except Exception as e:
        console.print(f"[yellow]⚠️  memory system unavailable: {e}[/yellow]")
        return None


def check_and_compress(memory, user_id, llm):
    """Check on startup whether old memories need compression."""
    if memory is None:
        return
    try:
        status = memory.get_compression_status(user_id)
        if status.get('needs_compression'):
            console.print(f"[yellow]⚙️ found {status['old_records']} records of 90+ day old memory, compressing...[/yellow]")
            result = memory.compress_old_memory(user_id, llm)
            if result.get('compressed', 0) > 0:
                console.print(
                    f"[green]✅ compressed {result['compressed']} records → "
                    f"quarterly profiles {result['periods']}, cost ¥{result['cost']:.4f}[/green]"
                )
            else:
                console.print("[dim]no compression needed[/dim]")
        else:
            console.print(f"[dim]no compression needed (only {status.get('old_records', 0)} records older than 90 days)[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠️  compression check failed: {e}[/yellow]")


def main():
    config = load_config()

    # ── Initialization ──
    console.print("[bold blue]🧪 initializing...[/bold blue]")

    retriever = YihuierRetriever(embeddings_dir=str(SKILL_DIR / "data" / "embeddings"))
    console.print("✅ retrieval engine ready")

    api_key = os.environ.get(config['llm']['api_key_env'], '')
    if not api_key:
        console.print(f"[red]❌ environment variable {config['llm']['api_key_env']} is not set[/red]")
        sys.exit(1)

    llm = LLMClient(
        provider=config['llm']['provider'],
        model=config['llm'].get('model'),
        api_key=api_key,
    )
    console.print(f"✅ LLM ready ({config['llm']['provider']} / {llm.model})")

    memory = init_memory(config)
    user_id = config.get('memory', {}).get('user_id', 'anonymous')

    # Load the system prompt
    sp_path = SKILL_DIR / "system_prompt.md"
    if not sp_path.exists():
        console.print("[red]❌ system_prompt.md not found[/red]")
        sys.exit(1)
    base_system_prompt = sp_path.read_text()

    # v3.1 cache-friendliness: static memory (USER_PROFILE + quarterly summaries) is appended to the system_prompt
    # DeepSeek prompt caching matches on prefix; a stable system_prompt means cache hits
    if memory:
        static_memory = memory.get_static_memory_section(user_id)
    else:
        static_memory = "[USER_PROFILE]\n（记忆功能未启用）"
    enhanced_system_prompt = base_system_prompt + "\n\n## 用户长期档案\n\n" + static_memory
    console.print(f"✅ system_prompt ready ({len(enhanced_system_prompt)} chars)")

    # Compression check on startup
    if memory and config.get('memory', {}).get('compression', {}).get('auto_run_on_startup', True):
        check_and_compress(memory, user_id, llm)

    # ── Main loop ──
    console.print(f"\n[bold green]🎓 Yihuier AI chemistry assistant v3[/bold green]")
    console.print(f"model: {llm.model} | full always-on memory | {llm.config['context_window']:,} context")
    console.print("commands: 'exit' quit | 'profile' view profile | 'cost' monthly cost\n")

    session_cost = 0.0
    session_queries = 0
    cache_hits = []

    while True:
        try:
            query = console.input("\n[cyan]Ask Jie-ge: [/cyan] ").strip()
            if not query:
                continue
            if query == 'exit':
                break

            if query == 'profile':
                if memory:
                    profile = memory.get_user_profile(user_id)
                    console.print(f"\n[bold]Your profile:[/bold]")
                    console.print(f"  grade: {profile.get('grade', 'not set')}")
                    console.print(f"  weak topics: {profile.get('weak_topics', [])}")
                    console.print(f"  mastered: {profile.get('mastered_topics', [])}")
                else:
                    console.print("[yellow]memory features disabled[/yellow]")
                continue

            if query == 'cost':
                from datetime import datetime
                month = datetime.now().strftime("%Y-%m")
                month_cost = memory.get_month_cost(user_id, month) if memory else 0
                console.print(f"\n[bold]💰 this month: ¥{month_cost:.4f}[/bold]")
                console.print(f"this session: ¥{session_cost:.4f} ({session_queries} questions)")
                continue

            # ── Main flow ──
            with console.status("[bold blue]Jie-ge is thinking..."):
                # 1. Diagnose
                diagnosis = diagnose_query(query, retriever)

                # 2. v3.1 dynamic memory: only the last 30 days of history (may change with every question)
                if memory:
                    dynamic_memory = memory.get_dynamic_memory_section(user_id)
                else:
                    dynamic_memory = "[RECENT_30_DAYS_HISTORY]\n（记忆功能未启用）"

                # 3. Format the retrieval results
                retrieval_text = format_retrieval_for_prompt(diagnosis)

                # 4. Build the user message (dynamic content, doesn't break system_prompt caching)
                user_msg = f"""{dynamic_memory}

[RETRIEVAL_RESULTS]
{retrieval_text}

[USER_QUERY]
{query}"""

                # 5. Call the LLM (system_prompt carries static memory; stable for 24h → cache hits)
                response = llm.chat(
                    messages=[
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=config['llm'].get('max_tokens', 2000),
                )

            # 6. Show the answer
            console.print("\n[bold green]🎯 Jie-ge: [/bold green]")
            console.print(Markdown(response['content']))

            # 7. Show the cost
            session_cost += response['cost_yuan']
            session_queries += 1
            usage = response['usage']
            cache_hits.append(usage['cache_hit_tokens'])
            console.print(
                f"\n[dim]💰 this turn: ¥{response['cost_yuan']:.4f} | "
                f"session total: ¥{session_cost:.4f} | "
                f"input: {usage['input_tokens']} ({usage['cache_hit_tokens']} cache hits) | "
                f"output: {usage['output_tokens']}[/dim]"
            )

            # 8. Save to memory
            if memory:
                memory.save_query(
                    user_id=user_id,
                    query=query,
                    diagnosis=diagnosis,
                    response=response['content'][:500],
                    cost=response['cost_yuan'],
                )

                # 9. Auto-update weak topics
                if diagnosis.get('missing_prereqs'):
                    memory.update_weak_topics(user_id, diagnosis['missing_prereqs'])

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]❌ error: {e}[/red]")
            import traceback
            traceback.print_exc()

    console.print(f"\n[green]Goodbye, total session cost ¥{session_cost:.4f} ({session_queries} questions)[/green]")
    if cache_hits:
        avg_hit = sum(cache_hits[1:]) / max(len(cache_hits) - 1, 1) if len(cache_hits) > 1 else 0
        console.print(f"[dim]cache hits: first=0, later average={avg_hit:.0f} tokens[/dim]")


if __name__ == "__main__":
    main()
