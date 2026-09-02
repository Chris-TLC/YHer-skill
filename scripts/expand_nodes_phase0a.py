#!/usr/bin/env python3
"""
Phase 0-A+ 节点扩展(数据驱动)—— 把28个过载节点(≥80题)拆成子节点,凑到~150节点

策略(Chris拍板:数据驱动拆 / 技巧暂不加 / 质量优先):
  - 不凭空生成子节点;从该节点真实真题里让DeepSeek归纳实际子类型(每个子节点都有真题支撑)
  - 两步/节点:① 看真题样本→提出N个子节点定义(按题量动态:题多拆多)
               ② 该节点全部题二次归类到子节点
  - 37个非过载节点(<80题)原样保留
  - 输出:knowledge_graph_150.jsonl(扩展后节点树) + clustered_questions_150/<子节点>.jsonl
  - 断点续跑:已拆的节点跳过;绝不改原65节点图谱和原clustered_questions

用法:
  python3 scripts/expand_nodes_phase0a.py            # 全部28过载节点
  python3 scripts/expand_nodes_phase0a.py --only 氧化还原反应   # 只拆1个(验证)
"""

import json, sys, os, time, argparse, hashlib
from pathlib import Path
from collections import defaultdict

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")
from adapters.llm_client import LLMClient

CLUSTER_DIR  = SKILL_DIR / "data" / "clustered_questions"
KG65_FILE    = SKILL_DIR / "data" / "knowledge_graph_full.jsonl"
OUT_KG150    = SKILL_DIR / "data" / "knowledge_graph_150.jsonl"
OUT_DIR150   = SKILL_DIR / "data" / "clustered_questions_150"
SUBDEF_FILE  = OUT_DIR150 / "_subnode_definitions.json"   # 各过载节点的子节点定义(断点续跑)
PROGRESS_F   = OUT_DIR150 / "_expand_progress.json"

LLM_MODEL = "deepseek-chat"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OVERLOAD = 80          # ≥80题视为过载,需拆
RETRY_MAX = 3
SAMPLE_FOR_DEF = 60    # 提子节点定义时,最多取多少题样本喂模型(控token)
BATCH_RECLS = 8        # 二次归类批大小

def safe(name): return name.replace("/", "_").replace("（","(").replace("）",")")

def load_node_questions(node_safe):
    fp = CLUSTER_DIR / f"{node_safe}.jsonl"
    if not fp.exists(): return []
    return [json.loads(l) for l in open(fp, encoding='utf-8') if l.strip()]

# 按题量动态决定拆几个子节点
def target_sub_count(n):
    if n >= 400: return 6
    if n >= 250: return 5
    if n >= 150: return 4
    if n >= 110: return 3
    return 2

SUBDEF_SYS = """你是上海高考化学知识点体系专家。给你一个大知识节点和它名下的真实高考真题样本,你要从这些真题里归纳出这个大节点实际包含的若干子知识点。

规则:
1. 子节点必须从真题里真实归纳,不能凭空想象——每个子节点都要是样本里真实考过的方向。
2. 子节点数量严格等于要求的个数。
3. 子节点之间界限清晰、不重叠,合起来覆盖这个大节点的主要考查方向。
4. 命名用"大节点-具体方向",如"氧化还原反应-配平"、"化学平衡-平衡移动"。

只输出JSON,格式:
{"subnodes":[{"name":"<子节点名>","desc":"<一句话:这个子节点考什么>"},...]}
不要输出任何解释或markdown。"""

def gen_subnode_defs(lc, node, qs, k):
    """让DeepSeek从真题归纳k个子节点定义"""
    import random
    sample = qs if len(qs) <= SAMPLE_FOR_DEF else random.sample(qs, SAMPLE_FOR_DEF)
    lines = [f"【大节点】{node}", f"【要求拆成】{k}个子节点", "", "【真题样本(题干)】"]
    for i, q in enumerate(sample):
        lines.append(f"{i+1}. {q.get('stem','')[:90]}")
    prompt = "\n".join(lines) + f"\n\n请从以上真题归纳出恰好{k}个子知识点。"
    messages = [{"role":"system","content":SUBDEF_SYS},{"role":"user","content":prompt}]
    for attempt in range(RETRY_MAX):
        try:
            r = lc.chat(messages, max_tokens=800, temperature=0.2)
            cost = r.get("cost_yuan",0) or 0
            content = r["content"].strip()
            s = content.find("{"); e = content.rfind("}")
            if s>=0 and e>s:
                d = json.loads(content[s:e+1])
                subs = d.get("subnodes",[])
                if subs and all(x.get("name") for x in subs):
                    return subs, cost
        except Exception as ex:
            es = str(ex).lower()
            if any(kk in es for kk in ['insufficient','余额','欠费','balance','arrears']):
                raise RuntimeError(f"BALANCE:{ex}")
            time.sleep(min(15,2**(attempt+1)))
    return None, 0

RECLS_SYS = """你是上海高考化学归类专家。给你一道题和若干子节点,判断这道题属于哪个子节点。
只输出JSON每题一行:{"id":"<id>","sub":"<子节点名>"}。子节点必须从给定清单选,不要自创。"""

def reclassify(lc, qs, subdefs, node):
    """把节点下所有题二次归类到子节点"""
    valid = {s["name"] for s in subdefs}
    menu = "\n".join(f'- {s["name"]}: {s["desc"]}' for s in subdefs)
    assign = {}
    cost = 0.0
    for q in qs:
        q["_rid"] = hashlib.md5(f"{q.get('_source_file','')}|{q.get('q_num','')}|{q.get('stem','')[:40]}".encode()).hexdigest()[:16]
    for i in range(0, len(qs), BATCH_RECLS):
        batch = qs[i:i+BATCH_RECLS]
        lines = [f"【子节点清单(属于大节点{node})】", menu, "", "【待归类题】"]
        for q in batch:
            lines.append(f'id={q["_rid"]} 题干:{q.get("stem","")[:110]}')
        prompt = "\n".join(lines) + "\n\n为每题输出一行归类JSON。"
        messages=[{"role":"system","content":RECLS_SYS},{"role":"user","content":prompt}]
        for attempt in range(RETRY_MAX):
            try:
                r = lc.chat(messages, max_tokens=900, temperature=0.1)
                cost += r.get("cost_yuan",0) or 0
                for line in r["content"].splitlines():
                    line=line.strip()
                    if not line.startswith("{"): continue
                    try: d=json.loads(line)
                    except: continue
                    if d.get("id") and d.get("sub") in valid:
                        assign[d["id"]] = d["sub"]
                break
            except Exception as ex:
                es=str(ex).lower()
                if any(kk in es for kk in ['insufficient','余额','欠费','balance']):
                    raise RuntimeError(f"BALANCE:{ex}")
                time.sleep(min(15,2**(attempt+1)))
    return assign, cost

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只拆指定节点(验证用)")
    args = ap.parse_args()
    if not DEEPSEEK_KEY: print("❌ 缺 DEEPSEEK_API_KEY"); sys.exit(1)

    OUT_DIR150.mkdir(parents=True, exist_ok=True)

    # 加载原65节点
    kg65 = [json.loads(l) for l in open(KG65_FILE,encoding='utf-8') if l.strip()]
    kg65_by_id = {n["node_id"]: n for n in kg65}

    # 找过载节点
    import glob
    sizes = {}
    for fp in glob.glob(str(CLUSTER_DIR/"*.jsonl")):
        bn = Path(fp).stem
        if bn.startswith("_"): continue
        sizes[bn] = sum(1 for l in open(fp,encoding='utf-8') if l.strip())
    overload = {n: c for n, c in sizes.items() if c >= OVERLOAD}
    if args.only:
        only_safe = safe(args.only)
        overload = {k:v for k,v in overload.items() if k==only_safe or k==args.only}
    print(f"过载节点(≥{OVERLOAD}题): {len(overload)}个待拆")

    # 断点续跑:已有的子节点定义
    subdefs_all = {}
    if SUBDEF_FILE.exists():
        subdefs_all = json.loads(SUBDEF_FILE.read_text(encoding='utf-8'))
        print(f"已拆节点(跳过): {len(subdefs_all)}个")

    lc = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=DEEPSEEK_KEY)
    cost = 0.0
    sub_assignments = {}  # node_safe -> {rid: subname}

    try:
        for node_safe, cnt in sorted(overload.items(), key=lambda x:-x[1]):
            if node_safe in subdefs_all:
                continue  # 已拆,跳过(子节点定义已存)
            qs = load_node_questions(node_safe)
            k = target_sub_count(len(qs))
            print(f"\n[{node_safe}] {len(qs)}题 → 拆{k}个子节点...")
            subs, c1 = gen_subnode_defs(lc, node_safe, qs, k)
            cost += c1
            if not subs:
                print(f"  ⚠️ 子节点归纳失败,跳过(保持原节点不拆)")
                continue
            print("  子节点:", [s["name"] for s in subs])
            assign, c2 = reclassify(lc, qs, subs, node_safe)
            cost += c2
            covered = len(assign)
            print(f"  二次归类: {covered}/{len(qs)}题 ¥{c1+c2:.3f}")
            subdefs_all[node_safe] = subs
            sub_assignments[node_safe] = assign
            # 即时落盘(断点续跑)
            SUBDEF_FILE.write_text(json.dumps(subdefs_all,ensure_ascii=False,indent=2),encoding='utf-8')
            # 该节点的题写入子节点文件
            rid2sub = assign
            for q in qs:
                rid = hashlib.md5(f"{q.get('_source_file','')}|{q.get('q_num','')}|{q.get('stem','')[:40]}".encode()).hexdigest()[:16]
                sub = rid2sub.get(rid, f"{node_safe}-其他")  # 没归上的进"其他"
                fp = OUT_DIR150 / f"{safe(sub)}.jsonl"
                qo = {kk:vv for kk,vv in q.items() if not kk.startswith("_rid")}
                qo["_parent_node"] = node_safe; qo["_sub_node"] = sub
                with open(fp,'a',encoding='utf-8') as f:
                    f.write(json.dumps(qo,ensure_ascii=False)+"\n")
            PROGRESS_F.write_text(json.dumps({"expanded":len(subdefs_all),
                "total_overload":len(overload),"cost":round(cost,3),
                "updated":time.strftime("%H:%M:%S")},ensure_ascii=False),encoding='utf-8')
    except RuntimeError as e:
        if str(e).startswith("BALANCE"):
            print(f"\n❌ 余额不足立即停止: {e}")
            print(f"已拆{len(subdefs_all)}个节点,补钱后重跑续跑")
            sys.exit(2)
        raise

    # 生成扩展后的150节点图谱:保留<80题的37节点 + 拆出的子节点
    build_kg150(kg65, kg65_by_id, sizes, subdefs_all)
    print(f"\n本轮成本 ¥{cost:.3f}")

def build_kg150(kg65, kg65_by_id, sizes, subdefs_all):
    """合成150节点图谱:非过载节点原样 + 过载节点的子节点(继承父节点的category/prereq)"""
    out = []
    expanded_parents = set(subdefs_all.keys())
    # 非过载节点 + 拆失败的过载节点 → 原样保留
    for n in kg65:
        nid_safe = safe(n["node_id"])
        if nid_safe in expanded_parents:
            continue  # 被拆了,用子节点代替
        out.append(n)
    # 子节点:继承父节点的category等,标记parent
    for parent_safe, subs in subdefs_all.items():
        parent = kg65_by_id.get(parent_safe) or kg65_by_id.get(parent_safe.replace("_","/"))
        cat = parent.get("category","") if parent else ""
        for s in subs:
            out.append({
                "node_id": s["name"],
                "category": cat,
                "parent_node": parent_safe,
                "desc": s.get("desc",""),
                "difficulty": parent.get("difficulty","T2") if parent else "T2",
                "exam_weight": parent.get("exam_weight","") if parent else "",
                "prerequisites": parent.get("prerequisites",[]) if parent else [],
                "successors": parent.get("successors",[]) if parent else [],
                "_derived_from": parent_safe,
            })
    with open(OUT_KG150,'w',encoding='utf-8') as f:
        for n in out:
            f.write(json.dumps(n,ensure_ascii=False)+"\n")
    print(f"\n✅ 扩展后图谱: {len(out)}个节点 → {OUT_KG150}")
    print(f"   (保留{len(out)-sum(len(s) for s in subdefs_all.values())}原节点 + "
          f"{sum(len(s) for s in subdefs_all.values())}子节点)")

if __name__ == "__main__":
    main()
