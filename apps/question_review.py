#!/usr/bin/env python3
"""
Phase 1 AI生成题 人工评审页 —— Chris逐题评审,对照≥95%合格标准

功能:
- 逐题卡片展示:题干/选项/答案(高亮)/解析/设计意图/AI对抗验证分
- 逐题打分:合格 / 不合格 / 存疑,可填备注
- 实时合格率统计(对照定稿≥95%进Phase2标准)
- 评审结果存 data/generated_questions/_review_results.json,随时可继续
- 筛选:全部/未评/合格/不合格/存疑

运行: streamlit run apps/question_review.py --server.port 8505
"""
from __future__ import annotations
import json
from pathlib import Path
import streamlit as st

SKILL_DIR = Path(__file__).parent.parent
GEN_FILE = SKILL_DIR / "data" / "generated_questions" / "generated_v1.jsonl"
REVIEW_FILE = SKILL_DIR / "data" / "generated_questions" / "_review_results.json"

st.set_page_config(page_title="AI出题评审", page_icon="✓", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.2rem; max-width: 920px; }
  h1,h2,h3 { letter-spacing:-.01em; color:#1d1d1f; font-weight:600; }
  .qcard { border:1px solid #e8e8ed; border-radius:16px; padding:22px 26px;
           background:#fff; box-shadow:0 2px 12px rgba(0,0,0,.03); margin-bottom:6px; }
  .qmeta { font-size:.75rem; color:#86868b; text-transform:uppercase; letter-spacing:.04em; }
  .stem { font-size:1.05rem; color:#1d1d1f; line-height:1.6; margin:10px 0 14px; font-weight:500; }
  .opt { padding:6px 12px; margin:4px 0; border-radius:8px; background:#f5f5f7; font-size:.95rem; }
  .opt.correct { background:#e8f8ed; color:#248a3d; font-weight:600; }
  .ans-box { background:#f0f7ff; border-left:3px solid #0071e3; padding:10px 14px;
             border-radius:8px; margin:10px 0; font-size:.92rem; }
  .expl { color:#424245; font-size:.9rem; line-height:1.55; margin-top:8px; }
  .design { color:#86868b; font-size:.82rem; font-style:italic; margin-top:8px;
            border-top:1px dashed #e8e8ed; padding-top:8px; }
  .score-pill { display:inline-block; padding:3px 10px; border-radius:12px;
                font-size:.78rem; font-weight:600; margin-right:6px; }
  .sc-good { background:#e8f8ed; color:#248a3d; }
  .sc-mid { background:#fff3e0; color:#c25d00; }
  .big-num { font-size:2.4rem; font-weight:680; color:#1d1d1f; }
  .big-label { font-size:.75rem; color:#86868b; text-transform:uppercase; }
  .metric-card { border:1px solid #e8e8ed; border-radius:14px; padding:14px 18px; background:#fff; }
</style>
""", unsafe_allow_html=True)

def load_questions():
    if not GEN_FILE.exists(): return []
    out = []
    for l in open(GEN_FILE, encoding='utf-8'):
        l = l.strip()
        if l:
            try: out.append(json.loads(l))
            except: pass
    return out

def load_reviews():
    if REVIEW_FILE.exists():
        try: return json.loads(REVIEW_FILE.read_text(encoding='utf-8'))
        except: return {}
    return {}

def save_reviews(r):
    REVIEW_FILE.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')

def qkey(q, i):
    return f"{i}_{q.get('_node','')[:10]}_{q.get('stem','')[:20]}"

questions = load_questions()
reviews = load_reviews()

st.markdown("# AI 生成题人工评审")
if not questions:
    st.info("还没有生成题。等 Phase 1 出题完成后这里会显示。")
    st.stop()

# ── 统计条 ──
total = len(questions)
verdicts = [reviews.get(qkey(q,i),{}).get("verdict") for i,q in enumerate(questions)]
n_pass = verdicts.count("合格")
n_fail = verdicts.count("不合格")
n_doubt = verdicts.count("存疑")
n_done = n_pass + n_fail + n_doubt
qualify_rate = n_pass / total if total else 0

c1,c2,c3,c4,c5 = st.columns(5)
for col,(lab,val,sub) in zip([c1,c2,c3,c4,c5],[
    ("总题数", total, f"已评 {n_done}"),
    ("合格", n_pass, f"{n_pass/max(total,1)*100:.0f}%"),
    ("不合格", n_fail, ""),
    ("存疑", n_doubt, ""),
    ("合格率", f"{qualify_rate*100:.0f}%", "标准≥95%"),
]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="big-label">{lab}</div>'
                    f'<div class="big-num">{val}</div>'
                    f'<div style="color:#86868b;font-size:.75rem;">{sub}</div></div>',
                    unsafe_allow_html=True)

if n_done == total and total > 0:
    if qualify_rate >= 0.95:
        st.success(f"✅ 合格率 {qualify_rate*100:.0f}% ≥ 95% — 达标,可进 Phase 2 诊断系统")
    else:
        st.warning(f"⚠️ 合格率 {qualify_rate*100:.0f}% < 95% — 未达标,需调出题器后重生")

# ── 筛选 ──
st.markdown("<br>", unsafe_allow_html=True)
flt = st.radio("筛选", ["全部","未评","合格","不合格","存疑"], horizontal=True, label_visibility="collapsed")

def show(i, q):
    rk = qkey(q, i)
    rv = reviews.get(rk, {})
    sc = q.get("_gen_scores", {})
    opts = q.get("options") or {}
    ans = str(q.get("answer",""))

    st.markdown('<div class="qcard">', unsafe_allow_html=True)
    st.markdown(f'<div class="qmeta">#{i+1} · {q.get("_node","")} · '
                f'{q.get("question_type","")}/{q.get("difficulty","")} · '
                f'AI验证{sc.get("overall","?")}分 · 重生{q.get("_regen_count",0)}次</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="stem">{q.get("stem","")}</div>', unsafe_allow_html=True)

    if isinstance(opts, dict) and opts:
        for k,v in opts.items():
            cls = "opt correct" if k.upper() in ans.upper() else "opt"
            st.markdown(f'<div class="{cls}">{k}. {v}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="ans-box"><b>答案:</b> {ans}</div>', unsafe_allow_html=True)
    with st.expander("解析 / 设计意图 / AI验证细节"):
        st.markdown(f'<div class="expl"><b>解析:</b> {q.get("explanation","")}</div>', unsafe_allow_html=True)
        if q.get("design_note"):
            st.markdown(f'<div class="design">设计意图: {q.get("design_note")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="design">AI验证: 化学{sc.get("chemistry")}·风格{sc.get("style")}·'
                    f'答案{sc.get("answer")}·新颖{sc.get("originality")} | 硬伤={sc.get("has_hard_defect")}</div>',
                    unsafe_allow_html=True)
        if sc.get("issues"):
            st.markdown(f'<div class="design">审稿残留: {"; ".join(sc["issues"][:3])}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 评审控件
    cc1, cc2 = st.columns([3,2])
    with cc1:
        cur = rv.get("verdict","未评")
        choice = st.radio("评定", ["未评","合格","不合格","存疑"],
                          index=["未评","合格","不合格","存疑"].index(cur) if cur in ["未评","合格","不合格","存疑"] else 0,
                          horizontal=True, key=f"v_{rk}")
    with cc2:
        note = st.text_input("备注", value=rv.get("note",""), key=f"n_{rk}", label_visibility="collapsed", placeholder="备注(可选)")
    if choice != "未评" or note:
        new = {"verdict": choice, "note": note}
        if reviews.get(rk) != new:
            reviews[rk] = new
            save_reviews(reviews)
    st.divider()

shown = 0
for i, q in enumerate(questions):
    rk = qkey(q, i); v = reviews.get(rk,{}).get("verdict","未评")
    if flt=="全部" or (flt=="未评" and v=="未评") or flt==v:
        show(i, q); shown += 1
if shown == 0:
    st.caption(f"「{flt}」筛选下没有题目")

st.caption("评审自动保存 · 逐题打分后顶部合格率实时更新 · 标准:合格率≥95%进Phase2")
