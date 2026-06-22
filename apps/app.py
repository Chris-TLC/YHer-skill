#!/usr/bin/env python3
"""
一化儿 AI 化学助手 - Web 版（Streamlit Cloud 公开部署）
阶段 10：BYOK + UUID 隔离 + 8 家 LLM + 对话搜索
"""

import streamlit as st
import sys
import os
import uuid
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")

from core.retrieve import YihuierRetriever
from core.diagnose import diagnose_query
from core.format_answer import build_response_prompt
from adapters.llm_client import LLMClient

# ── 页面配置 ─────────────────────────────────────────
st.set_page_config(
    page_title="YHer-skill：杰哥 AI 化学私教",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "模仿 B 站一化儿（杰哥）的高考化学应试 AI · BYOK · 长期记忆"
    }
)

# ── 隐藏 Streamlit 默认元素 ──────────────────────────
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    button[title="View fullscreen"] {visibility: hidden;}
    .st-keyboard-shortcuts {display: none !important;}
    [data-testid="stSidebarCollapseButton"] {display: none !important;}
    div[data-testid="collapsedControl"] {display: none !important;}
    .search-highlight {background-color: #FF6B3540; padding: 2px 4px; border-radius: 3px;}
    .st-key-jiege {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ── Embeddings 下载 ──────────────────────────────────
EMBEDDINGS_DIR = SKILL_DIR / "data" / "embeddings"


@st.cache_resource(show_spinner=False)
def ensure_embeddings():
    if (EMBEDDINGS_DIR / "chunks.faiss").exists():
        return str(EMBEDDINGS_DIR)
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
            st.error(f"embeddings 下载失败：{e}")
            st.stop()


@st.cache_resource
def init_retriever():
    return YihuierRetriever(embeddings_dir=ensure_embeddings())


@st.cache_resource
def load_system_prompt():
    return (SKILL_DIR / "system_prompt.md").read_text(encoding="utf-8")


def init_memory():
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        return None
    try:
        from adapters.memory import YihuierMemory
        return YihuierMemory(supabase_url, supabase_key)
    except Exception:
        return None


# ── Session 初始化 ───────────────────────────────────
def init_session():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

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
        st.session_state.show_settings = False
        st.session_state.search_query = ""
        st.session_state.search_results = []
        st.session_state.scroll_to_msg = None

    if "user_id" in st.session_state:
        st.query_params["uid"] = st.session_state.user_id


def is_ready():
    return (
        st.session_state.api_key.strip() != ""
        and st.session_state.grade != ""
    )


# ── Settings 按钮 ────────────────────────────────────
def render_top_bar():
    """顶部栏：标题 + 搜索 + 设置按钮"""
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.markdown("### 🧪 YHer-skill：杰哥 AI 化学私教")
    with col2:
        # 搜索框
        search_input = st.text_input(
            "🔍 搜索对话历史",
            placeholder="输入关键词搜索历史对话...",
            key="search_input",
            label_visibility="collapsed",
        )
        if search_input and search_input != st.session_state.get("_last_search", ""):
            st.session_state._last_search = search_input
            _do_search(search_input)

    with col3:
        # 设置按钮
        if st.button("⚙️ 设置", use_container_width=True, key="top_settings_btn"):
            st.session_state.show_settings = True
            st.rerun()


def _do_search(query: str):
    """搜索对话历史"""
    results = []
    msgs = st.session_state.messages
    # 对话配对：user 消息 + 紧随的 assistant 消息
    i = 0
    while i < len(msgs):
        if msgs[i]["role"] == "user" and query.lower() in msgs[i]["content"].lower():
            # 找紧随的 assistant 回复
            assistant_content = ""
            assistant_cost = None
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                assistant_content = msgs[i + 1]["content"]
                assistant_cost = msgs[i + 1].get("cost")
            results.append({
                "user_idx": i,
                "user_content": msgs[i]["content"],
                "assistant_content": assistant_content,
                "cost": assistant_cost,
            })
            i += 2 if (i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant") else 1
        else:
            i += 1

    st.session_state.search_results = results


# ── 顶部栏（紧凑版）──────────────────────────────────
def render_compact_topbar():
    """搜索栏 + 设置按钮（一行）"""
    c1, c2 = st.columns([5, 1])
    with c1:
        search = st.text_input(
            "🔍 搜索对话",
            placeholder="输入关键词...",
            key="compact_search",
            label_visibility="collapsed",
        )
        if search and search != st.session_state.get("_compact_search_last", ""):
            st.session_state._compact_search_last = search
            _do_search(search)
    with c2:
        if st.button("⚙️", help="设置", key="gear_btn", use_container_width=True):
            st.session_state.show_settings = True
            st.rerun()


# ── Settings 页面 ────────────────────────────────────
def render_settings_page():
    """设置页面（全屏覆盖）"""
    # 顶部：保存 + X 关闭
    sc1, sc2, sc3 = st.columns([4, 1, 1])
    with sc1:
        st.markdown("## ⚙️ 设置")
    with sc2:
        save_clicked = st.button("💾 保存", use_container_width=True, key="save_settings")
    with sc3:
        if st.button("✕ 关闭", use_container_width=True, key="close_settings"):
            st.session_state.show_settings = False
            st.rerun()

    st.divider()

    # LLM 提供商
    providers = LLMClient.PROVIDER_CONFIGS
    provider_keys = list(providers.keys())
    current_idx = provider_keys.index(st.session_state.llm_provider) \
        if st.session_state.llm_provider in provider_keys else 0

    provider = st.selectbox(
        "LLM 提供商",
        options=provider_keys,
        index=current_idx,
        format_func=lambda p: providers[p]["label"],
    )
    st.session_state.llm_provider = provider

    # API Key 链接
    key_link = providers[provider].get("key_link", "")
    st.caption(f"[获取 API Key]({key_link})（免费注册，按量付费 ~¥0.01/题）")

    # API Key 输入
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-xxxxxxxx",
        help="Key 仅在当前 session 内存保留，不存储。",
    )
    if api_key:
        st.session_state.api_key = api_key

    # 年级
    grade_opts = ["", "高一", "高二", "高三"]
    current_grade_idx = grade_opts.index(st.session_state.grade) \
        if st.session_state.grade in grade_opts else 0
    grade = st.selectbox(
        "年级 *（必填）",
        options=grade_opts,
        index=current_grade_idx,
        help="影响杰哥讲题难度和深度",
    )
    if grade:
        st.session_state.grade = grade

    # 姓名
    name = st.text_input(
        "你的名字（可选）",
        value=st.session_state.name,
        placeholder="杰哥会用它称呼你",
        max_chars=10,
    )
    st.session_state.name = name

    st.divider()

    # 用户 ID
    st.markdown("#### 📊 我的用户 ID")
    st.caption("跨设备同步记忆：复制此 ID → 在其他设备粘贴导入")
    st.code(st.session_state.user_id)

    import_id = st.text_input(
        "导入其他设备的 user_id",
        placeholder="anon_xxxxxxxxxxxxxxxx",
        key="import_uid_settings",
    )
    if st.button("切换到此 ID") and import_id:
        if import_id.startswith("anon_") and len(import_id) == 21:
            st.session_state.user_id = import_id
            st.session_state.messages = []
            st.query_params["uid"] = import_id
            st.success("已切换，记忆已同步！")
            st.rerun()
        else:
            st.error("格式错误，应为 anon_xxxxxxxxxxxxxxxx（21 字符）")

    # 保存按钮逻辑
    if save_clicked:
        if not st.session_state.api_key:
            st.error("请填写 API Key")
        elif not st.session_state.grade:
            st.error("请选择年级")
        else:
            memory_obj = init_memory()
            if memory_obj and st.session_state.grade:
                try:
                    memory_obj.sync_user_info(
                        st.session_state.user_id,
                        grade=st.session_state.grade,
                        name=st.session_state.name,
                    )
                except Exception:
                    pass
            st.session_state.show_settings = False
            st.success("设置已保存！")
            st.rerun()


# ── Main area ────────────────────────────────────────
def render_main():
    # 顶部栏
    render_compact_topbar()

    st.markdown(
        '<p style="color: #888; font-size: 0.85rem; margin-top: -10px;">'
        '模仿 B 站 <a href="https://space.bilibili.com/1526560679" target="_blank" '
        'style="color: #FF6B35;">一化儿（杰哥）</a> 的高考化学应试 AI · BYOK · 长期记忆 · '
        '<a href="https://github.com/Chris-TLC/YHer-skill" target="_blank" '
        'style="color: #888;">GitHub</a>'
        '</p>',
        unsafe_allow_html=True,
    )

    # 显示搜索结果
    if st.session_state.search_results:
        st.divider()
        st.caption(f"🔍 找到 {len(st.session_state.search_results)} 段相关对话：")
        for idx, result in enumerate(st.session_state.search_results):
            with st.container():
                c1, c2 = st.columns([8, 1])
                with c1:
                    user_preview = result["user_content"][:80].replace("\n", " ")
                    asst_preview = result["assistant_content"][:60].replace("\n", " ")
                    cost_str = f" · ¥{result['cost']:.4f}" if result["cost"] else ""
                    st.markdown(f"**Q**: {user_preview}...")
                    st.markdown(f"*A*: {asst_preview}...{cost_str}")
                with c2:
                    if st.button("📍", key=f"goto_{idx}", help="定位到这段对话"):
                        st.session_state.search_results = []
                        st.session_state.scroll_to_msg = result["user_idx"]
                        st.rerun()
                st.divider()

    # 历史消息（含滚动定位标记）
    target_idx = st.session_state.get("scroll_to_msg")
    for idx, msg in enumerate(st.session_state.messages):
        # 高亮目标消息
        highlight_attr = 'id="target-msg"' if idx == target_idx else ""

        avatar = "🧑‍🎓" if msg["role"] == "user" else "🧪"
        with st.chat_message(msg["role"], avatar=avatar):
            if highlight_attr:
                st.markdown(
                    f'<div {highlight_attr} style="border-left: 3px solid #FF6B35; '
                    f'padding-left: 12px;">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])

            if msg["role"] == "assistant" and "cost" in msg:
                st.caption(
                    f"💰 ¥{msg['cost']:.4f} | "
                    f"输入 {msg.get('input_tokens', '?')} "
                    f"({msg.get('cache_hit', 0)} 缓存命中) | "
                    f"输出 {msg.get('output_tokens', '?')}"
                )

    # 清除滚动目标
    if target_idx is not None:
        st.session_state.scroll_to_msg = None

    # 未配置提示
    if not is_ready():
        st.info("👋 欢迎！点击右上角 ⚙️ **设置**，配置 API Key 和年级后开始对话。")
        st.markdown("""
        **快速上手**（2 步）：
        1. 点 ⚙️ → 选择 LLM 提供商 → 点击"获取 API Key"链接 → 复制 Key → 粘贴
        2. 选择你的年级 → 点 💾 保存

        > 推荐 **DeepSeek**（国内最快，~¥0.01/题，1M 上下文）
        """)
        return

    # 输入框
    if query := st.chat_input("问杰哥一道题..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🧪"):
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
                    with st.expander("详情"):
                        st.code(traceback.format_exc())


def _handle_query(query: str):
    retriever = init_retriever()
    system_prompt = load_system_prompt()

    llm = LLMClient(
        provider=st.session_state.llm_provider,
        api_key=st.session_state.api_key,
    )

    diagnosis = diagnose_query(query, retriever)

    # 记忆（v3.1 缓存友好）
    memory_obj = init_memory()
    if memory_obj:
        static_memory = memory_obj.get_static_memory_section(st.session_state.user_id)
        dynamic_memory = memory_obj.get_dynamic_memory_section(st.session_state.user_id)
    else:
        static_memory = "[USER_PROFILE]\n（记忆功能未启用）"
        dynamic_memory = "[RECENT_30_DAYS_HISTORY]\n（暂无记录）"

    grade_hint = f"\n\n[当前用户]\n年级: {st.session_state.grade}"
    name_hint = f"\n称呼: {st.session_state.name}" if st.session_state.name else ""
    enhanced_system_prompt = (
        system_prompt + grade_hint + name_hint +
        "\n\n## 用户长期档案\n\n" + static_memory
    )

    response_prompt = build_response_prompt(query, diagnosis, style='auto')

    user_msg = f"""{dynamic_memory}

{response_prompt}"""

    response = llm.chat(
        messages=[
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=3000,
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


# ── Main entry ───────────────────────────────────────
def main():
    init_session()

    if st.session_state.show_settings:
        render_settings_page()
    else:
        render_main()


if __name__ == "__main__":
    main()
