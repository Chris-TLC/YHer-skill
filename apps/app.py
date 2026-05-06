#!/usr/bin/env python3
"""
一化儿 AI 化学助手 - Web 版（Streamlit）
v3：ChatGPT 风格 + 完整记忆 + 多 API 切换
"""

import streamlit as st
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

st.set_page_config(
    page_title="一化儿 AI 化学助手",
    page_icon="🧪",
    layout="wide"
)


@st.cache_resource
def init_retriever():
    return YihuierRetriever(embeddings_dir=str(SKILL_DIR / "data" / "embeddings"))


@st.cache_resource
def load_system_prompt():
    return (SKILL_DIR / "system_prompt.md").read_text()


PROVIDER_LABELS = {
    "deepseek": "🐋 DeepSeek V4-Pro（推荐，1M context）",
    "kimi": "🌙 Kimi（128K context）",
    "tongyi": "👁️ 通义千问 Max",
    "zhipu": "🧠 智谱 GLM-4 Plus",
    "doubao": "🍞 豆包 Pro",
    "minimax": "🌀 MiniMax abab6.5s",
    "anthropic": "🤖 Claude Opus 4.7（最高质量）",
    "openai": "🟢 OpenAI GPT-4o",
}


def init_memory():
    """尝试初始化记忆模块"""
    supabase_url = os.environ.get('SUPABASE_URL', '')
    supabase_key = os.environ.get('SUPABASE_KEY', '')

    if not supabase_url or not supabase_key:
        return None

    try:
        from adapters.memory import YihuierMemory
        return YihuierMemory(supabase_url, supabase_key)
    except Exception:
        return None


def main():
    # ── 侧边栏 ──
    with st.sidebar:
        st.title("⚙️ 设置")

        st.subheader("LLM 提供商")
        provider = st.selectbox(
            "选择 API",
            list(PROVIDER_LABELS.keys()),
            format_func=lambda x: PROVIDER_LABELS[x]
        )

        api_key = st.text_input(
            f"{provider.upper()} API Key",
            type="password",
            help="不会上传到服务器"
        )

        st.divider()

        user_id = st.text_input(
            "用户 ID（用于长期记忆）",
            value="anonymous",
            help="同一 ID 跨设备共享记忆"
        )

        enable_memory = st.toggle("启用长期记忆", value=True,
                                  help="存储到 Supabase")

        st.divider()

        if st.button("🗑️ 清除当前对话"):
            st.session_state.messages = []
            st.rerun()

    # ── 主界面 ──
    st.title("🧪 一化儿 AI 化学助手")
    st.caption("模仿 B 站杰哥的高考化学应试 AI · v3 完整记忆版 · 支持 8 个主流 API")

    # 初始化对话
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "cost" in msg:
                st.caption(
                    f"💰 ¥{msg['cost']:.4f} | "
                    f"缓存命中: {msg.get('cache_hit', 0)}"
                )

    # 输入
    if query := st.chat_input("有什么不会的？"):
        if not api_key:
            st.error("⚠️ 请在侧边栏输入 API Key")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("杰哥思考中..."):
                try:
                    retriever = init_retriever()
                    system_prompt = load_system_prompt()

                    llm = LLMClient(provider=provider, api_key=api_key)

                    diagnosis = diagnose_query(query, retriever)

                    # 记忆 - v3.1 缓存友好：静态记忆拼入 system_prompt
                    memory_obj = init_memory() if enable_memory else None
                    if memory_obj:
                        static_memory = memory_obj.get_static_memory_section(user_id)
                        dynamic_memory = memory_obj.get_dynamic_memory_section(user_id)
                    else:
                        static_memory = "[USER_PROFILE]\n（记忆功能未启用）"
                        dynamic_memory = "[RECENT_30_DAYS_HISTORY]\n（记忆功能未启用）"

                    enhanced_system_prompt = system_prompt + "\n\n## 用户长期档案\n\n" + static_memory

                    retrieval_text = format_retrieval_for_prompt(diagnosis)

                    user_msg = f"""{dynamic_memory}

[RETRIEVAL_RESULTS]
{retrieval_text}

[USER_QUERY]
{query}"""

                    response = llm.chat(
                        messages=[
                            {"role": "system", "content": enhanced_system_prompt},
                            {"role": "user", "content": user_msg}
                        ],
                        max_tokens=2000,
                    )

                    st.markdown(response['content'])
                    st.caption(
                        f"💰 ¥{response['cost_yuan']:.4f} | "
                        f"输入 {response['usage']['input_tokens']} "
                        f"({response['usage']['cache_hit_tokens']} 缓存命中) | "
                        f"输出 {response['usage']['output_tokens']}"
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['content'],
                        "cost": response['cost_yuan'],
                        "cache_hit": response['usage']['cache_hit_tokens'],
                    })

                    # 保存到记忆
                    if memory_obj:
                        memory_obj.save_query(
                            user_id=user_id,
                            query=query,
                            diagnosis=diagnosis,
                            response=response['content'][:500],
                            cost=response['cost_yuan'],
                        )
                        if diagnosis.get('missing_prereqs'):
                            memory_obj.update_weak_topics(
                                user_id, diagnosis['missing_prereqs']
                            )

                except Exception as e:
                    st.error(f"❌ 出错了: {e}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
