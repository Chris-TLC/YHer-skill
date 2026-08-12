#!/usr/bin/env python3
"""
视觉提取管道 v3 实时进度看板

任何时候打开都能看到：
- 管道阶段（转录→切题→验证→评分）
- 实时进度条（卷数/题数/成本）
- 质量指标（置信度分布、验证覆盖率、意图偏差）
- 最近处理的试卷详情
每 5 秒自动刷新。

运行：streamlit run apps/pipeline_v3_dashboard.py --server.port 8504
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

SKILL_DIR = Path(__file__).parent.parent
DATA_DIR = SKILL_DIR / "data" / "from_pdf"
PROGRESS_FILE = DATA_DIR / "pipeline_v3_progress.json"
OUTPUT_FILE = DATA_DIR / "all_from_pdf_v3.jsonl"

st.set_page_config(
    page_title="管道进度",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Apple 克制风格 ──
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.2rem; max-width: 1080px; }
    h1, h2, h3 { letter-spacing: -.01em; color: #1d1d1f; font-weight: 600; }
    .big-num { font-size: 2.6rem; font-weight: 680; color: #1d1d1f; line-height: 1.1; }
    .big-label { font-size: .78rem; color: #86868b; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px; }
    .metric-card {
        border: 1px solid #e8e8ed; border-radius: 16px; padding: 16px 20px;
        background: #fff; box-shadow: 0 2px 12px rgba(0,0,0,.03);
        height: 100%;
    }
    .metric-card.green { border-left: 3px solid #34c759; }
    .metric-card.blue  { border-left: 3px solid #0071e3; }
    .metric-card.warn  { border-left: 3px solid #ff9500; }
    .paper-row {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 14px; border-bottom: 1px solid #f5f5f7;
        font-size: .85rem;
    }
    .paper-name { flex: 1; color: #1d1d1f; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .paper-stat { color: #86868b; font-size: .78rem; min-width: 48px; text-align: right; }
    .paper-badge {
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: .72rem; font-weight: 600;
    }
    .badge-pass { background: #e8f8ed; color: #248a3d; }
    .badge-warn { background: #fff3e0; color: #c25d00; }
    .badge-running { background: #e6f2ff; color: #0071e3; }
    .pulse {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #34c759; margin-right: 6px;
        animation: pulse-anim 1.4s infinite;
    }
    @keyframes pulse-anim { 0% { opacity: .3 } 50% { opacity: 1 } 100% { opacity: .3 } }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #0071e3, #34c759) !important;
    }
    .phase-bar { display: flex; gap: 0; margin: 16px 0; border-radius: 10px; overflow: hidden; height: 6px; }
    .phase-seg { height: 100%; }
    .phase-seg.transcribe { background: #0071e3; }
    .phase-seg.structure { background: #5856d6; }
    .phase-seg.verify { background: #34c759; }
    .phase-seg.pending { background: #e8e8ed; }
    .phase-labels { display: flex; gap: 0; font-size: .7rem; color: #86868b; margin-top: 4px; }
    .phase-labels span { flex: 1; text-align: center; }
    .intent-ok { color: #34c759; font-weight: 600; }
    .intent-bad { color: #ff3b30; font-weight: 600; }
    .error-tag { color: #ff3b30; font-size: .73rem; }
</style>
""", unsafe_allow_html=True)


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def count_output_lines() -> int:
    try:
        return sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8"))
    except Exception:
        return 0


def scan_output_quality():
    """快速扫描输出文件的质量分布"""
    if not OUTPUT_FILE.exists():
        return None
    confs = []
    vc_ok = 0
    intent_issues = 0
    total = 0
    try:
        for line in open(OUTPUT_FILE, encoding="utf-8"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                q = json.loads(line)
                total += 1
                c = q.get("confidence", 0)
                if c > 0:
                    confs.append(c)
                cov = q.get("verification_coverage", {})
                if cov and cov.get("vision_verified") and cov.get("chemistry_verified") and cov.get("intent_verified"):
                    vc_ok += 1
                if any("意图" in r for r in q.get("confidence_reasons", [])):
                    intent_issues += 1
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    if not confs:
        return None
    hi = sum(1 for c in confs if c >= 0.95)
    mi = sum(1 for c in confs if 0.85 <= c < 0.95)
    lo = sum(1 for c in confs if 0.60 <= c < 0.85)
    return {
        "total": total,
        "avg_conf": round(sum(confs) / len(confs), 3),
        "hi": hi, "mi": mi, "lo": lo,
        "vc_rate": round(vc_ok / max(total, 1), 3),
        "intent_issues": intent_issues,
    }


# ── Header ──
st.markdown("# 提取管道进度")

prog = load_json(PROGRESS_FILE)
q_scan = scan_output_quality()

# ── 阶段条 ──
total_files = prog.get("total_files", 237)
done_files = len(prog.get("done_files", []))
done_pct = done_files / max(total_files, 1)

running = done_files > 0 and (
    prog.get("last_updated", "") > "" and done_pct < 1.0
)
finished = done_pct >= 0.99 and done_files > 0

# 估算各阶段权重
# Phase1 转录: ~40% 时间, Phase2 切题: ~15%, Phase3a+b+c 验证: ~35%, Phase4 评分: ~10%
phases = [
    ("转录", done_pct * 0.40, "phase-seg transcribe"),
    ("切题", done_pct * 0.15, "phase-seg structure"),
    ("验证", done_pct * 0.35, "phase-seg verify"),
    ("评分", done_pct * 0.10, "phase-seg verify"),
]
# 这些阶段不是按顺序而是按比例，简单处理：每条管道的整体阶段
# 实际在整体进度中：转录和验证是视觉API瓶颈
overall_pct = min(done_pct, 1.0)

st.markdown(
    f'<div class="sub" style="margin-bottom:6px;">'
    f'{"<span class=\"pulse\"></span> 运行中" if running else ("✅ 已完成" if finished else "⏳ 待启动")}'
    f'{" · 更新于 " + prog.get("last_updated", "") if prog.get("last_updated") else ""}'
    f'</div>',
    unsafe_allow_html=True,
)
st.progress(overall_pct)
st.markdown(
    f'<div style="font-size:1.1rem;font-weight:600;color:#1d1d1f;margin-top:4px;">'
    f'{done_files} / {total_files} 卷'
    f'<span style="color:#86868b;font-weight:400;font-size:.85rem;"> · {overall_pct*100:.0f}%</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── 指标卡片 ──
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

agg = prog.get("aggregated", {})
total_qs = prog.get("total_qs", q_scan.get("total", 0) if q_scan else 0)
total_passed = agg.get("passed", 0)
total_review = agg.get("needs_review", 0)
total_rejected = agg.get("rejected", 0)
cost_v = prog.get("cost_v", 0)
cost_l = prog.get("cost_l", 0)
intent_issues = q_scan.get("intent_issues", agg.get("intent_issues", 0)) if q_scan else agg.get("intent_issues", 0)

pass_rate = round(total_passed / max(total_qs, 1), 3)
avg_conf = (q_scan.get("avg_conf", 0) if q_scan
            else round(agg.get("avg_confidence_sum", 0) / max(agg.get("count", 1), 1), 3))

with c1:
    st.markdown(
        f'<div class="metric-card blue">'
        f'<div class="big-label">已提取题目</div>'
        f'<div class="big-num">{total_qs}</div>'
        f'<div style="color:#86868b;font-size:.78rem;">通过 {total_passed} · 需审 {total_review}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c2:
    cls = "green" if pass_rate >= 0.75 else "warn"
    st.markdown(
        f'<div class="metric-card {cls}">'
        f'<div class="big-label">通过率</div>'
        f'<div class="big-num">{pass_rate*100:.0f}%</div>'
        f'<div style="color:#86868b;font-size:.78rem;">≥0.85 置信度</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="big-label">均置信度</div>'
        f'<div class="big-num">{avg_conf:.3f}</div>'
        f'<div style="color:#86868b;font-size:.78rem;">满分 1.000</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c4:
    intent_ok = intent_issues == 0
    st.markdown(
        f'<div class="metric-card {"green" if intent_ok else "warn"}">'
        f'<div class="big-label">意图偏差</div>'
        f'<div class="big-num {"intent-ok" if intent_ok else "intent-bad"}">{intent_issues}</div>'
        f'<div style="color:#86868b;font-size:.78rem;">{"✅ 零偏差" if intent_ok else "⚠️ 需检查"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="big-label">累计成本</div>'
        f'<div class="big-num">¥{cost_v + cost_l:.0f}</div>'
        f'<div style="color:#86868b;font-size:.78rem;">视觉 ¥{cost_v:.0f} · 文本 ¥{cost_l:.0f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── 质量分布 ──
st.markdown("<br>", unsafe_allow_html=True)
if q_scan and q_scan["total"] > 0:
    st.markdown("### 置信度分布")
    total = q_scan["total"]
    hi_pct = q_scan["hi"] / total * 100
    mi_pct = q_scan["mi"] / total * 100
    lo_pct = q_scan["lo"] / total * 100

    col_h, col_m, col_l = st.columns(3)
    with col_h:
        st.markdown(
            f'<div style="text-align:center;padding:12px;border-radius:12px;background:#e8f8ed;">'
            f'<div style="font-size:1.6rem;font-weight:680;color:#248a3d;">{q_scan["hi"]}</div>'
            f'<div style="font-size:.75rem;color:#86868b;">高 ≥0.95 ({hi_pct:.0f}%)</div></div>',
            unsafe_allow_html=True,
        )
    with col_m:
        st.markdown(
            f'<div style="text-align:center;padding:12px;border-radius:12px;background:#e6f2ff;">'
            f'<div style="font-size:1.6rem;font-weight:680;color:#0071e3;">{q_scan["mi"]}</div>'
            f'<div style="font-size:.75rem;color:#86868b;">中 0.85-0.95 ({mi_pct:.0f}%)</div></div>',
            unsafe_allow_html=True,
        )
    with col_l:
        st.markdown(
            f'<div style="text-align:center;padding:12px;border-radius:12px;background:#fff3e0;">'
            f'<div style="font-size:1.6rem;font-weight:680;color:#c25d00;">{q_scan["lo"]}</div>'
            f'<div style="font-size:.75rem;color:#86868b;">低 0.60-0.85 ({lo_pct:.0f}%)</div></div>',
            unsafe_allow_html=True,
        )

    vc_rate = q_scan["vc_rate"]
    vc_cls = "green" if vc_rate >= 0.95 else ("warn" if vc_rate >= 0.80 else "")
    st.markdown(
        f'<div style="margin-top:10px;font-size:.82rem;color:#86868b;">'
        f'三重验证全覆盖率: <span style="font-weight:600;color:#1d1d1f;">{vc_rate*100:.0f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── 最近处理 ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 最近处理")

recents = prog.get("recent_results", [])
if recents:
    # 表头
    st.markdown(
        '<div style="display:flex;padding:6px 14px;font-size:.72rem;color:#86868b;'
        'text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid #e8e8ed;">'
        '<span style="flex:2;">试卷</span>'
        '<span style="width:50px;text-align:right;">题</span>'
        '<span style="width:50px;text-align:right;">通</span>'
        '<span style="width:50px;text-align:right;">审</span>'
        '<span style="width:55px;text-align:right;">率</span>'
        '<span style="width:60px;text-align:right;">置信</span>'
        '<span style="width:50px;text-align:right;">意图</span>'
        '<span style="width:70px;text-align:right;">成本</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    for r in recents[:15]:
        pr = r.get("passed_rate", 0)
        badge = "badge-pass" if pr >= 0.85 else ("badge-warn" if pr >= 0.70 else "badge-running")
        pct_str = f"{pr*100:.0f}%"
        conf = r.get("avg_confidence", 0)
        conf_str = f"{conf:.3f}"
        i_issues = r.get("intent_issues", 0)
        i_cls = "color:#34c759;" if i_issues == 0 else "color:#ff3b30;"
        errs = r.get("errors", [])
        st.markdown(
            f'<div class="paper-row">'
            f'<span class="paper-name" title="{r["file"]}">{r["file"]}</span>'
            f'<span class="paper-stat" style="width:50px;">{r["questions"]}</span>'
            f'<span class="paper-stat" style="width:50px;">{r["passed"]}</span>'
            f'<span class="paper-stat" style="width:50px;">{r.get("needs_review",0)}</span>'
            f'<span class="paper-stat" style="width:55px;">'
            f'<span class="paper-badge {badge}">{pct_str}</span></span>'
            f'<span class="paper-stat" style="width:60px;">{conf_str}</span>'
            f'<span class="paper-stat" style="width:50px;{i_cls}">{"✓" if i_issues == 0 else i_issues}</span>'
            f'<span class="paper-stat" style="width:70px;">¥{r.get("cost_v",0)+r.get("cost_l",0):.2f}</span>'
            f'</div>'
            + (f'<div style="font-size:.7rem;color:#ff3b30;padding:0 14px 4px;">{" · ".join(errs)}</div>' if errs else ""),
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div style="color:#86868b;font-size:.85rem;padding:20px 0;text-align:center;">'
        '还没有完成任何试卷。管道启动后这里会实时显示。</div>',
        unsafe_allow_html=True,
    )

# ── ETA ──
if running and done_files > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    elapsed = time.time() - (prog.get("started_at_ts", time.time()))
    if done_files > 0 and done_pct > 0.01:
        eta_total = elapsed / done_pct
        eta_remaining = eta_total - elapsed
        eta_str = f"{eta_remaining/60:.0f} 分钟" if eta_remaining < 3600 else f"{eta_remaining/3600:.1f} 小时"
        st.markdown(
            f'<div style="color:#86868b;font-size:.78rem;text-align:center;">'
            f'预计剩余 {eta_str} · 总耗时约 {eta_total/3600:.1f}h · '
            f'完成 {done_files}/{total_files}'
            f'</div>',
            unsafe_allow_html=True,
        )

st.caption("每 5 秒自动刷新 · 页面保持打开即可")
time.sleep(5)
st.rerun()
