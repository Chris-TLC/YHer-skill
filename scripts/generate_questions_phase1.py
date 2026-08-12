#!/usr/bin/env python3
"""
Phase 1 AI出题器 —— 基于内化规律库动态生成上海卷风格新题 + 双DeepSeek对抗验证

策略(Chris定稿:AI内化真题→动态出题 / 质量优先):
  生成闭环(每题):
    1. 加载目标子节点的出题规律(question_generation_patterns/<节点>.json)+ meta_patterns
    2. RAG基准:从该节点真题库随机抽3-5道作"上海卷风格锚点"(同节点真题即最精准基准,无需embedding)
    3. 生成器DeepSeek:据规律+基准生成1道新题(题干/选项/答案/解析/难度)
    4. 对抗验证DeepSeek(挑刺审稿人):4维度打分
         - 化学正确性(科学事实/方程式/计算)
         - 上海卷风格符合度(题型/语言/情境/难度)
         - 答案唯一性与正确性(答案确实对且唯一)
         - 与真题非雷同(不是抄真题,是新题)
       综合>7分留;否则带审稿反馈重生,最多 REGEN_MAX 次
    5. 留存:题 + 验证分 + 验证反馈,存 generated_questions/

  对抗验证是定稿"一个生成一个挑刺>7分重生"的兑现。质量优先:验证不通过宁可重生不将就。

用法:
  python3 scripts/generate_questions_phase1.py --sample 10        # 小样验证(跨节点选10题)
  python3 scripts/generate_questions_phase1.py --total 100        # 全量100题
  python3 scripts/generate_questions_phase1.py --node 盐类水解-水解规律与溶液酸碱性 --n 3
"""

import json, sys, os, time, argparse, glob, random
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")
from adapters.llm_client import LLMClient

PATTERN_DIR    = SKILL_DIR / "data" / "question_generation_patterns"
CLUSTER150_DIR = SKILL_DIR / "data" / "clustered_questions_150"
CLUSTER_DIR    = SKILL_DIR / "data" / "clustered_questions"
META_FILE      = PATTERN_DIR / "meta_patterns.json"
OUT_DIR        = SKILL_DIR / "data" / "generated_questions"
OUT_FILE       = OUT_DIR / "generated_v1.jsonl"
PROGRESS_F     = OUT_DIR / "_gen_progress.json"

LLM_MODEL = "deepseek-chat"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
RETRY_MAX = 3          # API重试
REGEN_MAX = 3          # 验证不过的重生次数(质量优先,多给机会改硬伤)
PASS_SCORE = 8.0       # 对抗验证通过线(>8,收紧:质量优先)
RAG_K = 4              # 抽几道真题作基准

# 代码层无图兜底关键词:生成题命中任一即打回重生(不依赖AI判断的硬保险)
IMG_KEYWORDS = ['如图','下图','上图','右图','左图','图所示','图示','装置图','示意图',
                '流程图','曲线图','坐标图','结构图','图中','图甲','图乙','图1','图2',
                '图一','图二','图(','图（','见图','如下图','如右图','晶胞如','结构如图']

def safe(name): return name.replace("/", "_").replace("（","(").replace("）",")")

def load_meta():
    return json.loads(META_FILE.read_text(encoding='utf-8')) if META_FILE.exists() else {}

def list_pattern_nodes():
    out = []
    for fp in glob.glob(str(PATTERN_DIR/"*.json")):
        bn = Path(fp).stem
        if bn.startswith("_") or bn == "meta_patterns": continue
        out.append(bn)
    return out

def load_pattern(node_safe):
    fp = PATTERN_DIR / f"{node_safe}.json"
    return json.loads(fp.read_text(encoding='utf-8')) if fp.exists() else None

def load_real_questions(node_safe):
    """该节点真题:优先150子节点库,fallback原65节点库"""
    for d in (CLUSTER150_DIR, CLUSTER_DIR):
        fp = d / f"{node_safe}.jsonl"
        if fp.exists():
            return [json.loads(l) for l in open(fp,encoding='utf-8') if l.strip()]
    return []

def fmt_real(q):
    o = q.get("options")
    if isinstance(o, dict): opts = "  ".join(f"{k}.{v}" for k,v in o.items())
    elif isinstance(o, list): opts = "  ".join(str(x) for x in o)
    else: opts = str(o) if o else ""
    return (f'题干:{q.get("stem","")[:200]}'
            + (f'\n选项:{opts[:200]}' if opts else '')
            + f'\n答案:{str(q.get("answer",""))[:80]}')

GEN_SYS = """你是上海高考化学命题专家。你已内化上海卷出题规律。任务:据给定的知识节点出题规律 + 上海卷整体DNA + 几道该节点真题(作风格基准),生成1道全新的上海卷风格题目。

要求:
1. 必须是新题,不能抄真题或简单改数字,要有独立的情境和考查设计。
2. 严格符合该节点的出题规律(题型倾向/难度/考查角度/常见陷阱设置)。
3. 符合上海卷整体风格(语言正式严谨,常用真实情境,正误判断辨析细节)。
4. 选择题必须4个选项且只有1个正确,干扰项要有迷惑性(踩在规律给的陷阱上)。
5. 答案必须科学正确且唯一,解析要讲清为什么。
6. 【最重要·无图自包含】你只能输出纯文字,无法生成任何图片。因此:
   - 绝对禁止出任何依赖图的题——不准出现"如图所示/下图/图中/装置图/晶胞图/示意图/流程图/曲线图/图甲图乙"等任何需要看图才能作答的表述。
   - 题目必须"自包含":作答所需的全部信息(数据、结构、条件、装置描述)都用文字写在题干里,用户不看任何图也能完整作答。
   - 若某考查角度本质上离不开图(如晶胞计算、图像分析、装置识别),就换一个该节点里不需要图的角度出题,或把图的关键信息改写成完整的文字描述(如"立方晶胞中,顶点为M原子、体心为N原子"),确保纯文字可解。

只输出JSON(不要markdown):
{"question_type":"选择题/填空题/计算题","difficulty":"T1/T2/T3/T4","stem":"题干","options":{"A":"","B":"","C":"","D":""},"answer":"","explanation":"解析","design_note":"这道题考查什么、设了什么陷阱"}
(填空/计算题options留空对象{})"""

def build_gen_prompt(node, pattern, meta, reals, feedback=None):
    parts = [f"【目标知识节点】{node}", "",
             "【该节点出题规律】", json.dumps(pattern, ensure_ascii=False, indent=1)[:2500], "",
             "【上海卷整体DNA】", json.dumps(meta.get("shanghai_exam_characteristics",{}), ensure_ascii=False)[:600], "",
             "【该节点真题(风格基准,模仿其风格但不要抄)】"]
    for i, q in enumerate(reals):
        parts.append(f"基准{i+1}. {fmt_real(q)}")
    if feedback:
        parts.append(f"\n【上一次生成被审稿打回,必须改进】\n{feedback}")
    parts.append("\n请生成1道全新的上海卷风格题目,严格按JSON输出。")
    return "\n".join(parts)

CRITIC_SYS = """你是极其严苛的上海高考化学审稿专家,任务是挑刺。给你一道AI生成的题,从4个维度打分(各0-10),并指出问题。

维度:
1. chemistry(化学正确性):科学事实/方程式/计算是否全对。
2. style(上海卷风格符合度):题型/语言/情境/难度是否像上海真题。
3. answer(答案唯一性与正确性):给出的答案是否确实正确且唯一,选择题干扰项是否真的错。
4. originality(新颖性):是否是新题而非抄真题改数字。

【硬伤判定——发现任一硬伤,verdict必须为fail且overall必须≤6】:
- 【无图依赖】题干/选项/解析出现"如图/下图/图中/图所示/装置图/晶胞图/示意图/流程图/曲线图/图甲图乙"等任何依赖图的表述,但题目是纯文字没有图——用户看不到图就无法作答,这是最严重的硬伤,必须fail。
- 【信息不完整】作答所需的关键信息(数据/结构/条件)没有完整写在题干文字里,用户凭题目文字无法独立作答。
- 题干/选项与解析自相矛盾(如选项写的数字与解析算出的不一致)
- 任何化学科学性错误(方程式不守恒、事实错误、概念错误)
- 答案错误,或答案不唯一(多个选项都对/都错)
- 关键表述有歧义、会引发争议或多种理解
- 解析逻辑错误或解释不能自洽

只输出JSON(不要markdown):
{"chemistry":0-10,"style":0-10,"answer":0-10,"originality":0-10,"overall":0-10,"issues":["问题1",...],"has_hard_defect":true/false,"verdict":"pass/fail"}
规则:overall取4维加权(化学正确性和答案最重要)。只要 has_hard_defect 为 true,overall必须≤6且verdict=fail。issues里列出的每一条都要在判分中体现——不要一边列出实质问题一边给pass。只有题目完全干净、无任何实质issue时才能 verdict=pass。"""

def parse_json(content):
    c = content.strip(); s = c.find("{"); e = c.rfind("}")
    if s>=0 and e>s:
        try: return json.loads(c[s:e+1])
        except: return None
    return None

def call(lc, sys_p, user_p, max_tokens=2000):
    for attempt in range(RETRY_MAX):
        try:
            r = lc.chat([{"role":"system","content":sys_p},{"role":"user","content":user_p}],
                        max_tokens=max_tokens, temperature=0.5)
            return r["content"], (r.get("cost_yuan",0) or 0)
        except Exception as ex:
            es=str(ex).lower()
            if any(k in es for k in ['insufficient','余额','欠费','balance','arrears']):
                raise RuntimeError(f"BALANCE:{ex}")
            time.sleep(min(15,2**(attempt+1)))
    return None, 0

def gen_one(lc, node, pattern, meta):
    """生成1题并对抗验证,返回(题dict或None, 成本, 尝试记录)"""
    reals = load_real_questions(node)
    cost = 0.0
    feedback = None
    attempts = []
    for regen in range(REGEN_MAX + 1):
        sample = random.sample(reals, min(RAG_K, len(reals))) if reals else []
        gp = build_gen_prompt(node, pattern, meta, sample, feedback)
        gen_c, c1 = call(lc, GEN_SYS, gp); cost += c1
        q = parse_json(gen_c) if gen_c else None
        if not q:
            attempts.append({"regen":regen,"err":"生成解析失败"}); continue
        # 代码层无图兜底:不依赖AI判断,关键词命中即打回重生(第三道保险)
        blob = (q.get("stem","") + json.dumps(q.get("options",{}),ensure_ascii=False)
                + q.get("explanation",""))
        img_hit = [kw for kw in IMG_KEYWORDS if kw in blob]
        if img_hit:
            attempts.append({"regen":regen,"score":0,"verdict":"fail",
                             "hard_defect":True,"issues":[f"无图依赖(代码拦截):{img_hit[:3]}"]})
            feedback = f"上一题出现了依赖图的表述{img_hit[:3]},但你无法生成图。必须改成纯文字自包含、不需要任何图就能作答的题。"
            continue
        # 对抗验证(JSON解析失败时,重试审稿一次,避免偶发解析失败误杀好题)
        crit_input = json.dumps({k:q.get(k) for k in
            ["question_type","difficulty","stem","options","answer","explanation"]},
            ensure_ascii=False)
        v = None
        for crit_try in range(2):
            crit_c, c2 = call(lc, CRITIC_SYS, f"【待审题目】\n{crit_input}\n\n请挑刺打分,严格按JSON输出。", max_tokens=900)
            cost += c2
            v = parse_json(crit_c) if crit_c else None
            if v: break
        score = float(v.get("overall",0)) if v else 0
        hard = bool(v.get("has_hard_defect", False)) if v else True
        attempts.append({"regen":regen,"score":score,
                         "verdict":v.get("verdict") if v else "?",
                         "hard_defect":hard,
                         "issues":v.get("issues",[]) if v else ["验证解析失败"]})
        # 质量优先:通过线>8 且 无硬伤 且 verdict=pass,三者全满足才留
        if v and score > PASS_SCORE and not hard and v.get("verdict")=="pass":
            q["_node"]=node; q["_gen_scores"]=v; q["_regen_count"]=regen
            return q, cost, attempts
        # 不过→带反馈重生
        feedback = f"得分{score}(需>{PASS_SCORE}且无硬伤), 必须修正的问题: {'; '.join(v.get('issues',[])[:4]) if v else '解析失败'}"
    return None, cost, attempts  # REGEN_MAX次仍不过→放弃该题

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="跨节点小样N题(验证)")
    ap.add_argument("--total", type=int, default=None, help="全量生成N题")
    ap.add_argument("--node", default=None, help="指定单节点")
    ap.add_argument("--n", type=int, default=3, help="单节点生成几题")
    args=ap.parse_args()
    if not DEEPSEEK_KEY: print("❌ 缺 DEEPSEEK_API_KEY"); sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    nodes = list_pattern_nodes()
    if not nodes: print("❌ 规律库为空,先跑Phase 0-B"); sys.exit(1)

    # 决定生成计划:[(node, 题数),...]
    plan = []
    if args.node:
        plan = [(safe(args.node), args.n)]
    elif args.sample:
        # 小样:从题量最多的节点里挑,每节点1题,跨节点
        ranked = sorted(nodes, key=lambda n: -(load_pattern(n) or {}).get("total_questions",0))
        plan = [(n, 1) for n in ranked[:args.sample]]
    else:
        total = args.total or 100
        # 全量:按节点题量加权分配(题多的节点多出几道),每节点至少0
        weights = {n:(load_pattern(n) or {}).get("total_questions",1) for n in nodes}
        tw = sum(weights.values())
        for n in sorted(nodes, key=lambda x:-weights[x]):
            share = max(1, round(total * weights[n]/tw))
            plan.append((n, share))
        # 截断到total
        acc=0; trimmed=[]
        for n,k in plan:
            if acc>=total: break
            k=min(k, total-acc); trimmed.append((n,k)); acc+=k
        plan=trimmed

    target = sum(k for _,k in plan)
    print(f"生成计划: {len(plan)}个节点, 目标{target}题, 通过线>{PASS_SCORE}, 重生上限{REGEN_MAX}")

    lc = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=DEEPSEEK_KEY)
    cost=0.0; passed=0; failed=0; t0=time.time()
    score_sum=0.0
    try:
        with open(OUT_FILE, 'a', encoding='utf-8') as fo:
            for node, k in plan:
                pattern = load_pattern(node)
                if not pattern: continue
                for j in range(k):
                    q, c, attempts = gen_one(lc, node, pattern, meta)
                    cost += c
                    if q:
                        fo.write(json.dumps(q, ensure_ascii=False)+"\n"); fo.flush()
                        passed += 1; score_sum += q["_gen_scores"]["overall"]
                        print(f"  ✅ [{passed}/{target}] {node[:24]} "
                              f"{q['question_type']}/{q['difficulty']} "
                              f"分{q['_gen_scores']['overall']} 重生{q['_regen_count']}次 ¥{cost:.3f}")
                    else:
                        failed += 1
                        last = attempts[-1] if attempts else {}
                        print(f"  ❌ {node[:24]} 验证未过(重生{REGEN_MAX}次) "
                              f"最后分{last.get('score','?')} {last.get('issues',[])[:1]}")
                    PROGRESS_F.write_text(json.dumps({"passed":passed,"failed":failed,
                        "target":target,"cost":round(cost,3),
                        "avg_score":round(score_sum/max(passed,1),2),
                        "updated":time.strftime("%H:%M:%S")},ensure_ascii=False),encoding='utf-8')
    except RuntimeError as e:
        if str(e).startswith("BALANCE"):
            print(f"\n❌ 余额不足停止: {e}\n已生成{passed}题,补钱后续跑"); sys.exit(2)
        raise

    print(f"\n{'='*55}")
    print(f"生成完成: 通过{passed}题 / 验证淘汰{failed}题 | 通过率{passed/max(passed+failed,1)*100:.0f}%")
    print(f"平均验证分: {score_sum/max(passed,1):.2f} | 成本¥{cost:.3f} | {time.time()-t0:.0f}s")
    print(f"生成题库: {OUT_FILE}")

if __name__=="__main__":
    main()
