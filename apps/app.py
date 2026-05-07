#!/usr/bin/env python3
"""
一化儿 AI 化学助手 - Web 版（Streamlit Cloud 公开部署）
阶段 10：BYOK + UUID 隔离 + 8 家 LLM + ModelScope 下载
"""

import streamlit as st
import sys
import os
import uuid
from pathlib import Path
from datetime import datetime

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")

from core.retrieve import YihuierRetriever
from core.diagnose import diagnose_query
from core.format_answer import format_retrieval_for_prompt
from adapters.llm_client import LLMClient

# ── 页面配置 ─────────────────────────────────────────
st.set_page_config(
    page_title="YHer-skill：杰哥 AI 化学私教",
    page_icon="🧪",
    layout="wide",
)

# ── Embeddings 下载（首次启动从 ModelScope 拉）────────
EMBEDDINGS_DIR = SKILL_DIR / "data" / "embeddings"


@st.cache_resource(show_spinner=False)
def ensure_embeddings():
    """确保 embeddings 存在，不存在则从 ModelScope 下载"""
    if (EMBEDDINGS_DIR / "chunks.faiss").exists():
        return str(EMBEDDINGS_DIR)

    # 检查软链接
    if EMBEDDINGS_DIR.is_symlink():
        real_path = EMBEDDINGS_DIR.resolve()
        if (real_path / "chunks.faiss").exists():
            return str(EMBEDDINGS_DIR)

    with st.spinner("首次启动：从 ModelScope 下载 embeddings（约 154MB，1-3 分钟）..."):
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            local_dir = snapshot_download(
                "ChrisTLC/YHer-skill-embeddings",
                repo_type="dataset",
                cache_dir=str(EMBEDDINGS_DIR.parent),
            )
            return local_dir
        except Exception as e:
            st.error(f"❌ embeddings 下载失败：{e}")
            st.info("请确保网络可访问 modelscope.cn")
            st.stop()


@st.cache_resource
def init_retriever():
    embeddings_path = ensure_embeddings()
    return YihuierRetriever(embeddings_dir=embeddings_path)


@st.cache_resource
def load_system_prompt():
    return (SKILL_DIR / "system_prompt.md").read_text(encoding="utf-8")


# ── 记忆初始化 ───────────────────────────────────────
def init_memory():
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")

    if not supabase_url or not supabase_key:
        return None

    try:
        from adapters.memory import YihuierMemory
        return YihuierMemory(supabase_url, supabase_key)
    except Exception as e:
        st.sidebar.warning(f"⚠️ 记忆系统不可用：{e}")
        return None


# ── Session 初始化 ───────────────────────────────────
def init_session():
    """初始化 session_state，优先从 query_params 恢复 user_id"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

        # 尝试从 URL query params 恢复 user_id
        qp = st.query_params
        if "uid" in qp:
            st.session_state.user_id = qp["uid"]
        else:
            st.session_state.user_id = f"anon_{uuid.uuid4().hex[:16]}"

        st.session_state.messages = []
        st.session_state.api_key = ""
        st.session_state.llm_provider = "deepseek"
        st.session_state.grade = ""
        st.session_state.name = ""

    # 如果 user_id 变了（用户手动导入），更新 query_params
    if "user_id" in st.session_state:
        st.query_params["uid"] = st.session_state.user_id


def is_ready():
    return (
        st.session_state.api_key.strip() != ""
        and st.session_state.grade != ""
    )


# ── Sidebar ──────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("⚙️ 设置")

    # LLM 提供商
    providers = LLMClient.PROVIDER_CONFIGS
    provider_keys = list(providers.keys())

    current_idx = provider_keys.index(st.session_state.llm_provider) \
        if st.session_state.llm_provider in provider_keys else 0

    provider = st.sidebar.selectbox(
        "LLM 提供商",
        options=provider_keys,
        index=current_idx,
        format_func=lambda p: providers[p]["label"],
        key="provider_select",
    )
    st.session_state.llm_provider = provider

    # API Key 链接
    key_link = providers[provider].get("key_link", "")
    if key_link:
        st.sidebar.markdown(f"[获取 API Key]({key_link})")

    # API Key 输入
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        placeholder="sk-xxxxxxxx",
        help="Key 仅在当前 session 内存中保留，刷新后需重新输入。不会上传到任何服务器。",
        key="api_key_input",
    )
    if api_key:
        st.session_state.api_key = api_key

    st.sidebar.divider()

    # 年级
    grade = st.sidebar.selectbox(
        "年级 *（必填，影响难度）",
        options=["", "高一", "高二", "高三"],
        index=0 if not st.session_state.grade
        else ["", "高一", "高二", "高三"].index(st.session_state.grade),
        key="grade_select",
    )
    if grade:
        st.session_state.grade = grade
        # 同步到 Supabase
        memory_obj = init_memory()
        if memory_obj:
            try:
                memory_obj.sync_user_info(
                    st.session_state.user_id, grade=grade)
            except Exception:
                pass

    # 姓名
    name = st.sidebar.text_input(
        "姓名（可选）",
        placeholder="留空则杰哥默认称呼为'同学'",
        max_chars=10,
        value=st.session_state.name,
        key="name_input",
    )
    if name != st.session_state.name:
        st.session_state.name = name
        memory_obj = init_memory()
        if memory_obj:
            try:
                memory_obj.sync_user_info(
                    st.session_state.user_id, name=name,
                    grade=st.session_state.grade)
            except Exception:
                pass

    st.sidebar.divider()

    # 用户 ID
    with st.sidebar.expander("📊 我的用户 ID（跨设备同步）"):
        uid = st.session_state.user_id
        st.code(uid)
        st.caption("这是你的唯一身份标识。换设备时复制此 ID，在另一设备粘贴以同步记忆。")

        st.button("📋 点击选中上方 ID（Ctrl+C 复制）", key="copy_uid",
                  on_click=lambda: None)

        new_id = st.text_input(
            "导入其他设备的 user_id",
            placeholder="anon_xxxxxxxxxxxxxxxx",
            key="import_uid",
        )
        if st.button("切换到此 ID") and new_id:
            if new_id.startswith("anon_") and len(new_id) == 21:
                st.session_state.user_id = new_id
                st.session_state.messages = []
                st.query_params["uid"] = new_id
                st.rerun()
            else:
                st.error("格式错误，应为 anon_xxxxxxxxxxxxxxxx（21 字符）")

    st.sidebar.divider()

    # 链接
    st.sidebar.markdown("🔗 **链接**")
    st.sidebar.markdown("[GitHub 仓库](https://github.com/Chris-TLC/YHer-skill)")
    st.sidebar.markdown("[一化儿 B 站主页](https://space.bilibili.com/277378387)")
    st.sidebar.markdown("[ModelScope Dataset](https://www.modelscope.cn/datasets/ChrisTLC/YHer-skill-embeddings)")

    st.sidebar.divider()

    # 清除对话
    if st.sidebar.button("🗑️ 清除当前对话"):
        st.session_state.messages = []
        st.rerun()


# ── 主区域 ───────────────────────────────────────────
def render_main():
    st.title("🧪 YHer-skill：杰哥 AI 化学私教")
    st.caption("模仿 B 站 [一化儿](https://space.bilibili.com/277378387)（杰哥）的高考化学应试 AI · BYOK · 长期记忆")

    # 未配置时的欢迎页
    if not is_ready():
        st.info("👋 欢迎！请在左侧侧边栏配置 **API Key** 和 **年级** 后开始对话。")
        st.markdown("""
        ### 快速开始（2 步）：
        1. **获取 API Key**：点击侧边栏的链接 → 注册/登录 → 复制 Key → 粘贴到输入框
        2. **选择年级**：影响杰哥讲题难度（高一基础版 / 高二进阶版 / 高三冲刺版）

        > 推荐选择 **DeepSeek**（国内最快，¥0.01/题，1M 上下文）
        """)
        return

    # 显示对话历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "cost" in msg:
                st.caption(
                    f"💰 ¥{msg['cost']:.4f} | "
                    f"输入 {msg.get('input_tokens', '?')} "
                    f"({msg.get('cache_hit', 0)} 缓存命中) | "
                    f"输出 {msg.get('output_tokens', '?')}"
                )

    # 输入框
    if query := st.chat_input("问杰哥一道题..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🎯 杰哥思考中..."):
                try:
                    _handle_query(query)
                except ValueError as e:
                    st.error(f"❌ {e}")
                except ConnectionError as e:
                    st.error(f"🌐 {e}")
                except Exception as e:
                    st.error(f"❌ 出错了：{e}")
                    import traceback
                    with st.expander("详细错误信息"):
                        st.code(traceback.format_exc())


def _handle_query(query: str):
    """核心对话流程"""
    retriever = init_retriever()
    system_prompt = load_system_prompt()

    llm = LLMClient(
        provider=st.session_state.llm_provider,
        api_key=st.session_state.api_key,
    )

    diagnosis = diagnose_query(query, retriever)

    # 记忆
    memory_obj = init_memory()
    if memory_obj:
        static_memory = memory_obj.get_static_memory_section(st.session_state.user_id)
        dynamic_memory = memory_obj.get_dynamic_memory_section(st.session_state.user_id)
    else:
        static_memory = "[USER_PROFILE]\n（记忆功能未启用）"
        dynamic_memory = "[RECENT_30_DAYS_HISTORY]\n（记忆功能未启用）"

    # 注入年级到 system_prompt
    grade_hint = f"\n\n[当前用户]\n年级: {st.session_state.grade}"
    name_hint = f"\n称呼: {st.session_state.name}" if st.session_state.name else ""
    enhanced_system_prompt = (
        system_prompt + grade_hint + name_hint +
        "\n\n## 用户长期档案\n\n" + static_memory
    )

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

    content = response["content"]
    usage = response["usage"]
    cost = response["cost_yuan"]

    st.markdown(content)
    st.caption(
        f"💰 ¥{cost:.4f} | "
        f"输入 {usage['input_tokens']} "
        f"({usage['cache_hit_tokens']} 缓存命中) | "
        f"输出 {usage['output_tokens']} | "
        f"模型 {response['model_returned']}"
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "cost": cost,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cache_hit": usage["cache_hit_tokens"],
    })

    # 保存到记忆
    if memory_obj:
        try:
            memory_obj.save_query(
                user_id=st.session_state.user_id,
                query=query,
                diagnosis=diagnosis,
                response=content[:500],
                cost=cost,
            )
        except Exception:
            pass

        if diagnosis.get("missing_prereqs"):
            try:
                memory_obj.update_weak_topics(
                    st.session_state.user_id,
                    diagnosis["missing_prereqs"],
                )
            except Exception:
                pass


# ── Main ─────────────────────────────────────────────
def main():
    init_session()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
