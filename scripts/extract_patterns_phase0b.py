#!/usr/bin/env python3
"""
Phase 0-B 出题规律提取 —— 每个节点喂全部真题给DeepSeek,提取"上海卷出题DNA"

策略(Chris拍板:质量优先 / 去金句库 / 数据驱动):
  - 对150节点体系里每个节点,把该节点全部真题喂DeepSeek,提取结构化出题规律
  - 字段严格对齐定稿 AI_INTERNALIZATION_STRATEGY.md:题型分布/难度分层/考查角度/
    常见陷阱/题干特征/选项设置规律/难度递进
  - 额外汇总 meta_patterns.json(上海卷整体DNA:风格/难度曲线/题型格式/语言风格)
  - ❌ 不提取金句(Chris拍板去金句库)
  - 断点续跑:已提取的节点跳过;输出独立到 question_generation_patterns/

数据源优先级:
  - 优先用 clustered_questions_150/(扩节点后的子节点题库)
  - 子节点没覆盖的题归入父节点;非过载节点用 clustered_questions/

用法:
  python3 scripts/extract_patterns_phase0b.py            # 全部节点
  python3 scripts/extract_patterns_phase0b.py --only 盐类水解  # 验证单节点
"""

import json, sys, os, time, argparse, glob
from pathlib import Path
from collections import Counter

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")
from adapters.llm_client import LLMClient

CLUSTER_DIR    = SKILL_DIR / "data" / "clustered_questions"
CLUSTER150_DIR = SKILL_DIR / "data" / "clustered_questions_150"
KG150_FILE     = SKILL_DIR / "data" / "knowledge_graph_150.jsonl"
OUT_DIR        = SKILL_DIR / "data" / "question_generation_patterns"
META_FILE      = OUT_DIR / "meta_patterns.json"
PROGRESS_F     = OUT_DIR / "_pattern_progress.json"

LLM_MODEL = "deepseek-chat"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
RETRY_MAX = 3
MAX_Q_PER_NODE = 50    # 每节点最多喂多少题(控token;按难度分层采样保证代表性)

def safe(name): return name.replace("/", "_").replace("（","(").replace("）",")")

PATTERN_SYS = """你是上海高考化学命题研究专家。给你某个知识节点名下的全部上海高考真题,你要深度分析这些真题,提取出"上海卷在这个知识点上的出题规律"——目的是让AI据此生成符合上海卷风格的新题,而不是记住这些题。

严格按以下JSON结构输出(不要markdown,不要解释):
{
  "knowledge_point": "<节点名>",
  "total_questions": <题数>,
  "question_type_distribution": {"选择题":0.x,"填空题":0.x,"计算题":0.x,"简答题":0.x},
  "difficulty_distribution": {"T1":0.x,"T2":0.x,"T3":0.x,"T4":0.x},
  "exam_focus": [
    {"angle":"<考查角度,如概念理解/计算应用/实验探究/综合推理>","weight":0.x,"typical_questions":["<典型问法>",...]}
  ],
  "common_traps": [
    {"trap":"<陷阱描述>","student_error":"<学生常犯错误>"}
  ],
  "stem_features": {
    "avg_length":"<平均字数区间>",
    "common_contexts":["<常用情境,如工业流程/实验装置/生活应用>"],
    "common_phrasings":["<常用表述>"]
  },
  "option_design": {
    "correct_option_traits":"<正确选项特点>",
    "distractor_patterns":["<干扰选项设置规律>"]
  },
  "difficulty_progression": {
    "easy":"<简单题考什么>","medium":"<中档增加什么>","hard":"<压轴综合点>"
  }
}
要求:规律必须从真题归纳,具体可操作(能指导出题),不要空泛套话。"""

def build_prompt(node, qs):
    # 按难度分层采样,保证各档都有代表
    by_diff = {}
    for q in qs:
        by_diff.setdefault(str(q.get("difficulty","T2")), []).append(q)
    sample = []
    per = max(1, MAX_Q_PER_NODE // max(len(by_diff),1))
    for d, lst in by_diff.items():
        sample.extend(lst[:per])
    sample = sample[:MAX_Q_PER_NODE]
    lines = [f"【知识节点】{node}", f"【该节点真题数】{len(qs)}（采样{len(sample)}道分析）", "", "【真题】"]
    for i, q in enumerate(sample):
        o = q.get("options")
        # options 统一为字符串:dict→"A.xx B.xx",list→拼接,其他→str
        if isinstance(o, dict):
            opts = "  ".join(f"{k}.{v}" for k, v in o.items())
        elif isinstance(o, list):
            opts = "  ".join(str(x) for x in o)
        else:
            opts = str(o) if o else ""
        ans = str(q.get("answer",""))[:40]
        lines.append(
            f'{i+1}. [{q.get("question_type","")}/{q.get("difficulty","")}] '
            f'{q.get("stem","")[:140]}'
            + (f'\n   选项:{opts[:120]}' if opts else '')
            + f'\n   答案:{ans}'
        )
    return "\n".join(lines) + "\n\n请提取这个知识节点的上海卷出题规律,严格按JSON结构输出。"

def load_node_index():
    """返回 {节点名: [题...]},优先用150子节点库,补上未过载的原节点"""
    idx = {}
    # 150子节点
    if CLUSTER150_DIR.exists():
        for fp in glob.glob(str(CLUSTER150_DIR/"*.jsonl")):
            bn = Path(fp).stem
            if bn.startswith("_"): continue
            idx[bn] = [json.loads(l) for l in open(fp,encoding='utf-8') if l.strip()]
    # 被拆的父节点(已在150里),不再重复加载原节点;只补"未过载"的原节点
    expanded_parents = set()
    subdef = CLUSTER150_DIR / "_subnode_definitions.json"
    if subdef.exists():
        expanded_parents = set(json.loads(subdef.read_text(encoding='utf-8')).keys())
    for fp in glob.glob(str(CLUSTER_DIR/"*.jsonl")):
        bn = Path(fp).stem
        if bn.startswith("_"): continue
        if bn in expanded_parents: continue   # 已拆,用子节点
        if bn in idx: continue
        idx[bn] = [json.loads(l) for l in open(fp,encoding='utf-8') if l.strip()]
    return idx

def extract_one(lc, node, qs):
    messages=[{"role":"system","content":PATTERN_SYS},
              {"role":"user","content":build_prompt(node, qs)}]
    for attempt in range(RETRY_MAX):
        try:
            r = lc.chat(messages, max_tokens=2000, temperature=0.2)
            cost = r.get("cost_yuan",0) or 0
            c = r["content"].strip()
            s=c.find("{"); e=c.rfind("}")
            if s>=0 and e>s:
                d = json.loads(c[s:e+1])
                if d.get("exam_focus"):   # 起码有考查角度才算成功
                    return d, cost
        except Exception as ex:
            es=str(ex).lower()
            if any(k in es for k in ['insufficient','余额','欠费','balance','arrears']):
                raise RuntimeError(f"BALANCE:{ex}")
            time.sleep(min(15,2**(attempt+1)))
    return None, 0

META_SYS = """你是上海高考化学命题研究专家。给你各知识节点的出题规律摘要,提炼出"上海卷的整体出题DNA"(跨知识点的元规律)。
严格按JSON输出(不要markdown):
{
  "shanghai_exam_characteristics":{"overall_style":"","difficulty_curve":"","calculation_emphasis":"","experiment_focus":"","organic_emphasis":""},
  "question_format_rules":{"multiple_choice":{"stem_length":"","option_count":4,"distractor_quality":""},"fill_in_blank":{"stem_length":"","blank_count":""}},
  "language_style":{"formality":"","typical_phrases":[],"avoid_phrases":[]}
}"""

def build_meta(patterns):
    # 汇总各节点题型/难度/情境给模型提炼
    type_agg = Counter(); ctx = Counter(); phrases = Counter()
    for p in patterns.values():
        for t,w in (p.get("question_type_distribution") or {}).items(): type_agg[t]+=w
        for c in (p.get("stem_features") or {}).get("common_contexts",[]): ctx[c]+=1
        for ph in (p.get("stem_features") or {}).get("common_phrasings",[]): phrases[ph]+=1
    summary = {
        "节点数": len(patterns),
        "题型总体倾向": dict(type_agg.most_common()),
        "高频情境": [c for c,_ in ctx.most_common(12)],
        "高频表述": [p for p,_ in phrases.most_common(15)],
    }
    return f"【各节点规律汇总】\n{json.dumps(summary,ensure_ascii=False,indent=2)}\n\n请提炼上海卷整体出题DNA。"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args=ap.parse_args()
    if not DEEPSEEK_KEY: print("❌ 缺 DEEPSEEK_API_KEY"); sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx = load_node_index()
    if args.only:
        idx = {k:v for k,v in idx.items() if k==args.only or k==safe(args.only)}
    print(f"待提取规律的节点: {len(idx)}个")

    # 断点续跑:已有规律文件的节点跳过
    done = set()
    for fp in glob.glob(str(OUT_DIR/"*.json")):
        bn=Path(fp).stem
        if not bn.startswith("_") and bn!="meta_patterns":
            done.add(bn)
    if done: print(f"已提取(跳过): {len(done)}个")

    lc = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=DEEPSEEK_KEY)
    cost=0.0; ok=0; fail=[]; t0=time.time()
    patterns={}
    # 先把已存的载入(供meta用)
    for bn in done:
        try: patterns[bn]=json.loads((OUT_DIR/f"{bn}.json").read_text(encoding='utf-8'))
        except: pass

    fail_streak=0
    try:
        for node, qs in sorted(idx.items(), key=lambda x:-len(x[1])):
            if safe(node) in done: continue
            p, c = extract_one(lc, node, qs)
            cost += c
            if p:
                p["total_questions"]=len(qs)
                (OUT_DIR/f"{safe(node)}.json").write_text(
                    json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
                patterns[safe(node)]=p; ok+=1; fail_streak=0
                print(f"  ✅ {node} ({len(qs)}题) ¥{cost:.3f}")
            else:
                fail.append(node); fail_streak+=1
                print(f"  ⚠️ {node} 规律提取失败")
                if fail_streak>=5:
                    print(f"\n⚠️ 连续5节点失败,停止。已完成{ok}个。"); break
            PROGRESS_F.write_text(json.dumps({"done":ok+len(done),"total":len(idx),
                "cost":round(cost,3),"updated":time.strftime("%H:%M:%S")},
                ensure_ascii=False),encoding='utf-8')
    except RuntimeError as e:
        if str(e).startswith("BALANCE"):
            print(f"\n❌ 余额不足立即停止: {e}\n已完成{ok}个,补钱后续跑"); sys.exit(2)
        raise

    # 生成 meta_patterns
    if len(patterns)>=10:
        print("\n提炼 meta_patterns(上海卷整体DNA)...")
        try:
            msgs=[{"role":"system","content":META_SYS},
                  {"role":"user","content":build_meta(patterns)}]
            r=lc.chat(msgs,max_tokens=1500,temperature=0.2); cost+=r.get("cost_yuan",0) or 0
            c=r["content"]; s=c.find("{"); e=c.rfind("}")
            if s>=0: META_FILE.write_text(json.dumps(json.loads(c[s:e+1]),
                ensure_ascii=False,indent=2),encoding='utf-8'); print("  ✅ meta_patterns.json")
        except Exception as ex: print(f"  ⚠️ meta提炼失败:{str(ex)[:80]}")

    print(f"\n{'='*55}")
    print(f"规律提取完成: {ok+len(done)}/{len(idx)}节点 | 成本¥{cost:.3f} | {time.time()-t0:.0f}s")
    if fail: print(f"失败{len(fail)}个: {fail[:10]}")
    print(f"规律库: {OUT_DIR}/<节点>.json")

if __name__=="__main__":
    main()
