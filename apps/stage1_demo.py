#!/usr/bin/env python3
"""
阶段一 Demo：化学单科 AI 私教学习舱（引擎层版）。

界面只负责渲染 + 收输入，所有业务逻辑在 core/tutor/ 引擎层。
苹果化克制风格：清爽、自然、不花哨。

运行：streamlit run apps/stage1_demo.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
load_dotenv(SKILL_DIR / ".env")

from adapters.llm_client import LLMClient
from core.data.knowledge_repository import get_knowledge_repository
from core.tutor.llm_bridge import make_llm_caller
from core.tutor.session_orchestrator import SessionOrchestrator, TutorSession

st.set_page_config(page_title="化学私教学习舱", page_icon="🧪",
                   layout="wide", initial_sidebar_state="expanded")

# ── 苹果化克制样式 ─────────────────────────────────────────
st.markdown("""
<style>
    :root{--ink:#1d1d1f;--muted:#86868b;--line:#e8e8ed;--accent:#0071e3;--soft:#f5f5f7;}
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding-top:2rem;max-width:1080px;}
    div[data-testid="stSidebar"]{background:#fbfbfd;border-right:1px solid var(--line);}
    h1,h2,h3,h4{letter-spacing:-.01em;color:var(--ink);}
    .hero{font-size:1.7rem;font-weight:680;color:var(--ink);margin-bottom:2px;}
    .hero-sub{color:var(--muted);font-size:.95rem;}
    .card{border:1px solid var(--line);border-radius:16px;padding:18px 20px;background:#fff;
          margin:10px 0;transition:all .2s ease;}
    .card:hover{box-shadow:0 6px 24px rgba(0,0,0,.05);}
    .metric{border:1px solid var(--line);border-radius:14px;padding:14px 16px;background:#fff;}
    .metric-l{color:var(--muted);font-size:.78rem;}
    .metric-v{color:var(--ink);font-size:1.1rem;font-weight:640;margin-top:6px;}
    .pill{display:inline-block;border:1px solid var(--line);border-radius:999px;
          padding:2px 10px;font-size:.74rem;color:var(--muted);margin-right:5px;}
    .qbox{border:1px solid #d9e7ff;background:#f5f9ff;border-radius:14px;padding:16px 18px;margin:8px 0;}
    .stButton>button{border-radius:10px;font-weight:560;}
    .stProgress>div>div>div{background:var(--accent);}
</style>
""", unsafe_allow_html=True)


# ── 状态 ──────────────────────────────────────────────────
def init_state():
    d = {
        "uid": f"demo_{uuid.uuid4().hex[:8]}", "session": None,
        "current_q": None, "diag_done": False,
        "started_at": None, "task_started_at": None,
        "provider": "deepseek", "api_key": "",
        "grade": "高二", "region": "全国卷", "node_id": "化学平衡",
        "goal": "我今天想补化学平衡，尤其是转化率和平衡移动", "time_budget": 180,
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)


def get_orchestrator() -> SessionOrchestrator:
    key = (st.session_state.api_key or "").strip()
    caller = make_llm_caller(st.session_state.provider, api_key=key) if key else None
    return SessionOrchestrator(llm_caller=caller)


def node_options():
    repo = get_knowledge_repository()
    # 高频高考节点放前面
    priority = ["化学平衡", "电化学", "氧化还原", "平衡常数", "盐类水解",
                "沉淀溶解平衡", "工艺流程", "弱电解质电离平衡"]
    all_nodes = [n.node_id for n in repo.all_nodes()]
    rest = [n for n in all_nodes if n not in priority]
    return [n for n in priority if n in all_nodes] + sorted(rest)


# ── 侧边栏 ─────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("### 设置")
        providers = list(LLMClient.PROVIDER_CONFIGS.keys())
        st.selectbox("模型", providers,
                     index=providers.index(st.session_state.provider) if st.session_state.provider in providers else 0,
                     key="provider", format_func=lambda p: LLMClient.PROVIDER_CONFIGS[p]["label"])
        st.text_input("API Key", type="password", key="api_key",
                      placeholder="留空走本地演示")
        st.divider()
        toc()


def toc():
    s = st.session_state.session
    st.markdown("### 目录")
    if not s:
        st.caption("创建学习舱后显示。")
        return
    for t in s.tasks:
        mark = "●" if t["task_id"] == s.current_task_id else (
            "✓" if t.get("status") == "done" else "○")
        st.markdown(f'<div style="padding:4px 0;font-size:.86rem;color:#1d1d1f">'
                    f'{mark} {t["task_id"]} {t["title"]}</div>', unsafe_allow_html=True)


# ── 创建学习舱 ─────────────────────────────────────────────
def tab_create():
    st.markdown("### 创建学习舱")
    with st.form("create"):
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.selectbox("知识点（全部 65 个化学考点）", node_options(), key="node_id")
            st.text_area("本次目标", key="goal", height=80)
        with c2:
            st.selectbox("年级", ["高一", "高二", "高三"], key="grade")
            st.selectbox("卷别", ["全国卷", "上海卷", "新高考"], key="region")
            st.number_input("托管时长（分钟）", 30, 360, step=15, key="time_budget")
        go = st.form_submit_button("生成学习舱", type="primary", use_container_width=True)

    if go:
        orch = SessionOrchestrator(llm_caller=None)  # 创建不需LLM
        s = orch.create_session(
            st.session_state.uid, st.session_state.goal, node_id=st.session_state.node_id,
            grade=st.session_state.grade, region=st.session_state.region,
            time_budget_min=int(st.session_state.time_budget),
        )
        st.session_state.session = s
        st.session_state.current_q = orch.first_question(s)
        st.session_state.diag_done = False
        st.session_state.started_at = time.time()
        st.session_state.task_started_at = time.time()
        st.rerun()

    s = st.session_state.session
    if s:
        overview(s)


def overview(s: TutorSession):
    repo = get_knowledge_repository()
    node = repo.find_node(s.node_id)
    total = sum(t["duration_min"] for t in s.tasks)
    cols = st.columns(4)
    metrics = [("知识点", s.node_id), ("年级/卷别", f"{s.grade}·{s.region}"),
               ("托管时长", f"{total} 分钟"),
               ("掌握判据", f"{len(node.mastery_rubric) if node else 0} 条")]
    for col, (l, v) in zip(cols, metrics):
        col.markdown(f'<div class="metric"><div class="metric-l">{l}</div>'
                     f'<div class="metric-v">{v}</div></div>', unsafe_allow_html=True)
    st.markdown("#### 任务队列")
    for t in s.tasks:
        active = t["task_id"] == s.current_task_id
        border = "border-color:#a8ccff;background:#f5f9ff;" if active else ""
        st.markdown(f'<div class="card" style="{border}">'
                    f'<b>{t["task_id"]}｜{t["title"]}</b> '
                    f'<span class="pill">{t["duration_min"]} 分钟</span>'
                    f'<span class="pill">闸门 {int(t["mastery_gate"]*100)}%</span></div>',
                    unsafe_allow_html=True)


# ── 逐层诊断 ───────────────────────────────────────────────
def tab_diagnose():
    s = st.session_state.session
    if not s:
        st.info("请先创建学习舱。")
        return
    st.markdown("### 逐层诊断")
    # 历史
    for h in s.diag_history:
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(f"**{h['question'].get('level','')}**：{h['answer']}")
        with st.chat_message("assistant", avatar="🧪"):
            st.caption(f"掌握度 {int(h.get('mastery',0)*100)}%")
            st.markdown(h.get("control", {}).get("reason", "") or "（已批改）")

    if st.session_state.diag_done:
        st.success("诊断收束，进入执行环节。")
        return

    q = st.session_state.current_q or {}
    st.markdown(f'<div class="qbox"><span class="pill">{q.get("level","诊断")}</span>'
                f'<h4 style="margin:8px 0">{q.get("prompt","")}</h4>'
                f'<div style="color:#86868b;font-size:.85rem">{q.get("look_for","")}</div></div>',
                unsafe_allow_html=True)
    ans = st.text_area("你的回答", key=f"ans_{len(s.diag_history)}", height=120,
                       placeholder="不会也可以写卡住的位置。")
    c1, c2 = st.columns([2, 1])
    if c1.button("提交，下一问", type="primary", use_container_width=True):
        if not ans.strip():
            st.warning("先写一点。")
        else:
            orch = get_orchestrator()
            with st.spinner("批改中…"):
                r = orch.run_diagnosis_turn(s, q, ans)
            if r["ready_for_execution"] or not r["next_question"]:
                st.session_state.diag_done = True
            else:
                st.session_state.current_q = r["next_question"]
            st.rerun()
    if c2.button("直接进入执行", use_container_width=True):
        st.session_state.diag_done = True
        s.current_task_id = "T2"
        st.session_state.task_started_at = time.time()
        st.rerun()


# ── 执行 ──────────────────────────────────────────────────
def tab_execute():
    s = st.session_state.session
    if not s:
        st.info("请先创建学习舱。")
        return
    if not st.session_state.diag_done:
        st.info("建议先完成诊断；急用可在诊断页直接进入执行。")

    cur = next((t for t in s.tasks if t["task_id"] == s.current_task_id), s.tasks[0])
    total = sum(t["duration_min"] for t in s.tasks)
    elapsed = (time.time() - (st.session_state.started_at or time.time())) / 60
    t_elapsed = (time.time() - (st.session_state.task_started_at or time.time())) / 60

    cols = st.columns(4)
    vals = [("总进度", f"{elapsed:.0f}/{total} 分"), ("当前", f"{cur['task_id']} {cur['title']}"),
            ("本阶段", f"{t_elapsed:.0f}/{cur['duration_min']} 分"),
            ("成本", f"¥{s.cost_yuan:.4f}")]
    for col, (l, v) in zip(cols, vals):
        col.markdown(f'<div class="metric"><div class="metric-l">{l}</div>'
                     f'<div class="metric-v">{v}</div></div>', unsafe_allow_html=True)
    st.progress(min(elapsed / total, 1.0) if total else 0)

    # 对话
    for ev in s.transcript:
        if ev["role"] == "student":
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(ev["content"])
        else:
            with st.chat_message("assistant", avatar="🧪"):
                if ev.get("mastery") is not None:
                    st.caption(f"{ev.get('at','')} · 掌握度 {int(ev.get('mastery',0)*100)}%")
                st.markdown(ev["content"])

    msg = st.chat_input("把你的解题过程、答案或卡住的位置发给私教…")
    if msg:
        orch = get_orchestrator()
        tc = {"task_elapsed_min": round(t_elapsed, 1), "task_budget_min": cur["duration_min"],
              "total_elapsed_min": round(elapsed, 1), "total_budget_min": total}
        with st.spinner("私教思考中…"):
            r = orch.run_execution_turn(s, msg, time_ctx=tc)
        if r["decision"]["to_task"] != cur["task_id"]:
            st.session_state.task_started_at = time.time()
        st.rerun()


# ── 复盘 ──────────────────────────────────────────────────
def tab_report():
    s = st.session_state.session
    if not s:
        st.info("请先创建学习舱。")
        return
    st.markdown("### 复盘报告")
    if st.button("生成复盘", type="primary", use_container_width=True):
        orch = get_orchestrator()
        with st.spinner("生成中…"):
            r = orch.run_report(s)
        st.markdown(r["visible"])
        if r.get("events"):
            with st.expander("学习事件（事件级记忆）"):
                st.json(r["events"])


def main():
    init_state()
    sidebar()
    st.markdown('<div class="hero">化学 AI 私教学习舱</div>'
                '<div class="hero-sub">把一次学习变成可诊断、可执行、可复盘的私教课。</div>',
                unsafe_allow_html=True)
    st.write("")
    tabs = st.tabs(["创建", "诊断", "执行", "复盘"])
    with tabs[0]:
        tab_create()
    with tabs[1]:
        tab_diagnose()
    with tabs[2]:
        tab_execute()
    with tabs[3]:
        tab_report()


if __name__ == "__main__":
    main()
