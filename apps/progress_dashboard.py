#!/usr/bin/env python3
"""
题库处理实时进度看板（图形化，含阶段总览）。

任何时候打开都能看到：
- 整个流程的阶段总览（切题→入库→完成，当前在哪一步）
- 当前阶段的实时进度条
- 累计题数、成本统计
每 3 秒自动刷新。

运行：streamlit run apps/progress_dashboard.py --server.port 8503
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

SKILL_DIR = Path(__file__).parent.parent
DATA_DIR = SKILL_DIR / "data"
SPLIT_PROGRESS = DATA_DIR / "split_progress.json"
INGEST_PROGRESS = DATA_DIR / "ingest_progress.json"
RAW_DIR = DATA_DIR / "raw_papers"
ITEM_BANK_DIR = DATA_DIR / "item_bank"

st.set_page_config(page_title="题库处理进度", page_icon="📊", layout="wide")

st.markdown("""
<style>
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding-top:1.6rem;max-width:920px;}
    h1,h2,h3{letter-spacing:-.01em;color:#1d1d1f;}
    .big{font-size:2.4rem;font-weight:700;color:#1d1d1f;}
    .sub{color:#86868b;font-size:.9rem;}
    .card{border:1px solid #e8e8ed;border-radius:18px;padding:20px 24px;background:#fff;
          box-shadow:0 4px 20px rgba(0,0,0,.04);margin:8px 0;}
    .stat{font-size:1.9rem;font-weight:680;color:#1d1d1f;}
    .stat-l{color:#86868b;font-size:.8rem;margin-bottom:4px;}
    .pulse{display:inline-block;width:9px;height:9px;border-radius:50%;
           background:#34c759;margin-right:6px;animation:p 1.4s infinite;}
    @keyframes p{0%{opacity:.3}50%{opacity:1}100%{opacity:.3}}
    .stProgress>div>div>div{background:linear-gradient(90deg,#0071e3,#34c759);}
    .stage{display:flex;gap:10px;margin:14px 0;}
    .stage-item{flex:1;border:1px solid #e8e8ed;border-radius:14px;padding:12px;text-align:center;background:#fff;}
    .stage-active{border-color:#0071e3;background:#f0f7ff;box-shadow:0 2px 12px rgba(0,113,227,.12);}
    .stage-done{border-color:#cfead8;background:#f2fbf5;}
    .stage-name{font-size:.85rem;font-weight:600;color:#1d1d1f;}
    .stage-detail{font-size:.72rem;color:#86868b;margin-top:3px;}
</style>
""", unsafe_allow_html=True)


def load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def count_lines(path):
    try:
        return sum(1 for _ in open(path, encoding="utf-8"))
    except Exception:
        return 0


def scan_bank():
    if not ITEM_BANK_DIR.exists():
        return 0
    return sum(count_lines(f) for f in ITEM_BANK_DIR.glob("*.jsonl")
               if not f.name.startswith("_"))


def scan_raw():
    if not RAW_DIR.exists():
        return 0
    return sum(count_lines(f) for f in RAW_DIR.glob("*.jsonl")
               if not f.name.startswith("_"))


st.markdown("# 📊 题库处理进度")

split = load_json(SPLIT_PROGRESS)
ingest = load_json(INGEST_PROGRESS)
raw_total = scan_raw()
bank_total = scan_bank()

# 判断当前阶段
split_running = split and split.get("status") == "running"
ingest_running = ingest and ingest.get("status") == "running"
if split_running:
    cur_stage = 1
elif ingest_running:
    cur_stage = 2
elif bank_total > 0:
    cur_stage = 3  # 有题库了
elif raw_total > 0:
    cur_stage = 2  # 切完了待入库
else:
    cur_stage = 0

# ── 阶段总览 ──
stages = [
    ("① 切题", "视觉读卷切单题", raw_total > 0 or split_running, cur_stage == 1),
    ("② 入库", "AI解题+提炼得分点", bank_total > 0 or ingest_running, cur_stage == 2),
    ("③ 可用", "题库接入私教", bank_total > 0 and not ingest_running, cur_stage == 3),
]
html = '<div class="stage">'
for name, detail, done, active in stages:
    cls = "stage-item stage-active" if active else ("stage-item stage-done" if done else "stage-item")
    mark = "●" if active else ("✓" if done else "○")
    html += f'<div class="{cls}"><div class="stage-name">{mark} {name}</div><div class="stage-detail">{detail}</div></div>'
html += '</div>'
st.markdown(html, unsafe_allow_html=True)

st.divider()

# ── 当前阶段细节 ──
if split_running:
    done = split.get("done_papers", 0)
    total = split.get("total_papers", 1) or 1
    pct = min(done / total, 1.0)
    st.markdown(f'<div class="sub"><span class="pulse"></span>正在切题…  更新于 {split.get("updated","")}</div>',
                unsafe_allow_html=True)
    st.markdown("### 切题进度")
    st.progress(pct)
    st.markdown(f'<div class="big">{done} / {total} 卷</div>'
                f'<div class="sub">{pct*100:.0f}%  ·  批次：{split.get("batch","")}</div>',
                unsafe_allow_html=True)
    if split.get("current"):
        st.markdown(f'<div class="sub">当前：{split.get("current")}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (l, v) in zip((c1, c2, c3), [
        ("已切出题数", f'{split.get("total_items",0)} 题'),
        ("切题累计成本", f'¥{split.get("cost",0):.3f}'),
        ("题库已入库", f'{bank_total} 题'),
    ]):
        col.markdown(f'<div class="card"><div class="stat-l">{l}</div><div class="stat">{v}</div></div>',
                     unsafe_allow_html=True)

elif ingest_running:
    done = ingest.get("done", 0)
    total = ingest.get("total", 1) or 1
    pct = min(done / total, 1.0)
    st.markdown(f'<div class="sub"><span class="pulse"></span>正在入库（AI解题+提炼得分点）…  更新于 {ingest.get("updated","")}</div>',
                unsafe_allow_html=True)
    st.markdown("### 入库进度")
    st.progress(pct)
    st.markdown(f'<div class="big">{done} / {total} 题</div><div class="sub">{pct*100:.0f}%</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, (l, v) in zip((c1, c2), [
        ("入库累计成本", f'¥{ingest.get("cost",0):.3f}'),
        ("题库总题数", f'{bank_total} 题'),
    ]):
        col.markdown(f'<div class="card"><div class="stat-l">{l}</div><div class="stat">{v}</div></div>',
                     unsafe_allow_html=True)

else:
    # 空闲：显示已有成果
    if bank_total > 0:
        st.success(f"✅ 当前题库已有 {bank_total} 道可用真题")
    elif raw_total > 0:
        st.info(f"已切出 {raw_total} 题，待入库")
    else:
        st.info("还没有任务在跑。开始后这里会实时显示进度。")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="card"><div class="stat-l">已切题(raw)</div><div class="stat">{raw_total} 题</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="stat-l">已入库(可用)</div><div class="stat">{bank_total} 题</div></div>',
                unsafe_allow_html=True)

st.caption("每 3 秒自动刷新 · 任何时候打开都能看到当前进度")
time.sleep(3)
st.rerun()
