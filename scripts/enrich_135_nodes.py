#!/usr/bin/env python3
"""
135节点判据细化 —— 给98个派生子节点补 judgment_criteria + common_failures(诊断命脉)

背景:扩节点时98个子节点只继承了父节点的判据(太泛)。诊断/能力画像需要细化到子节点的:
  - judgment_criteria_for_mastery: 掌握该子节点的客观判据(字符串数组)
  - common_failures: 该子节点的典型失误(cause/symptom/diagnostic_question对象数组)

做法:父节点判据 + 子节点真题 + 子节点描述 → DeepSeek生成细化判据。字段严格对齐65节点结构。
输出:knowledge_graph_150_enriched.jsonl(37原节点原样 + 98子节点补全判据)
断点续跑:已补的子节点跳过。
用法: python3 scripts/enrich_135_nodes.py
"""
import json, sys, os, time, glob, random
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")
from adapters.llm_client import LLMClient

KG150 = SKILL_DIR / "data" / "knowledge_graph_150.jsonl"
KG65  = SKILL_DIR / "data" / "knowledge_graph_full.jsonl"
CLUSTER150 = SKILL_DIR / "data" / "clustered_questions_150"
OUT = SKILL_DIR / "data" / "knowledge_graph_150_enriched.jsonl"
CACHE = SKILL_DIR / "data" / "_enrich_cache.json"   # 断点续跑:子节点→判据

LLM_MODEL = "deepseek-chat"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
RETRY_MAX = 3

def safe(n): return n.replace("/","_").replace("（","(").replace("）",")")

ENRICH_SYS = """你是上海高考化学教学诊断专家。给你一个细分知识子节点(及其父节点判据、几道该子节点真题),你要为这个子节点生成专属的"掌握判据"和"常见失误",用于诊断学生是否真掌握该子节点。

严格按JSON输出(不要markdown):
{
  "judgment_criteria_for_mastery": ["<判据1:能...>","<判据2>","<判据3>","<判据4>"],
  "common_failures": [
    {"cause":"<错误根因>","symptom":"<外在表现:学生会怎么错>","diagnostic_question":"<一句能3秒暴露这个错的诊断问题>"}
  ]
}
要求:
- judgment_criteria 4条左右,针对这个子节点的具体能力(不是泛泛的父节点判据),可客观检验。
- common_failures 至少4个,来自真题反映的真实易错点,diagnostic_question要能快速诊断。
- 紧扣这个子节点,不要写成父节点的泛泛内容。"""

def gen_enrich(lc, sub_name, parent_name, parent_crit, reals):
    sample = reals[:8]
    lines = [f"【子节点】{sub_name}", f"【父节点】{parent_name}",
             f"【父节点判据(太泛,需细化)】{json.dumps(parent_crit,ensure_ascii=False)[:400]}",
             "", "【该子节点真题】"]
    for i,q in enumerate(sample):
        lines.append(f"{i+1}. {q.get('stem','')[:100]}")
    prompt = "\n".join(lines) + f"\n\n请为子节点「{sub_name}」生成专属掌握判据和常见失误。"
    for attempt in range(RETRY_MAX):
        try:
            r=lc.chat([{"role":"system","content":ENRICH_SYS},{"role":"user","content":prompt}],
                      max_tokens=1500,temperature=0.3)
            cost=r.get("cost_yuan",0) or 0
            c=r["content"].strip(); s=c.find("{"); e=c.rfind("}")
            if s>=0:
                d=json.loads(c[s:e+1])
                if d.get("judgment_criteria_for_mastery") and d.get("common_failures"):
                    return d, cost
        except Exception as ex:
            es=str(ex).lower()
            if any(k in es for k in ['insufficient','余额','欠费','balance']):
                raise RuntimeError(f"BALANCE:{ex}")
            time.sleep(min(15,2**(attempt+1)))
    return None, 0

def load_reals(sub_safe):
    fp = CLUSTER150 / f"{sub_safe}.jsonl"
    if fp.exists():
        return [json.loads(l) for l in open(fp,encoding='utf-8') if l.strip()]
    return []

def main():
    if not DEEPSEEK_KEY: print("❌ 缺 DEEPSEEK_API_KEY"); sys.exit(1)
    nodes=[json.loads(l) for l in open(KG150,encoding='utf-8') if l.strip()]
    kg65={json.loads(l)["node_id"]:json.loads(l) for l in open(KG65,encoding='utf-8') if l.strip()}

    cache={}
    if CACHE.exists(): cache=json.loads(CACHE.read_text(encoding='utf-8'))

    subs=[n for n in nodes if n.get("_derived_from") or n.get("parent_node")]
    print(f"待补判据子节点: {len(subs)}个, 已补(跳过):{len([s for s in subs if s['node_id'] in cache])}")

    lc=LLMClient(provider='deepseek',model=LLM_MODEL,api_key=DEEPSEEK_KEY)
    cost=0.0; ok=0
    try:
        for n in subs:
            nid=n["node_id"]
            if nid in cache: continue
            parent=n.get("_derived_from") or n.get("parent_node","")
            pnode=kg65.get(parent) or kg65.get(parent.replace("_","/"),{})
            pcrit=pnode.get("judgment_criteria_for_mastery",[])
            reals=load_reals(safe(nid))
            d,c=gen_enrich(lc,nid,parent,pcrit,reals); cost+=c
            if d:
                cache[nid]=d; ok+=1
                CACHE.write_text(json.dumps(cache,ensure_ascii=False,indent=1),encoding='utf-8')
                if ok%10==0 or ok<=3:
                    print(f"  ✅ [{ok}] {nid[:30]} ¥{cost:.3f}")
            else:
                print(f"  ⚠️ {nid} 补判据失败,保留父节点判据")
    except RuntimeError as e:
        if str(e).startswith("BALANCE"):
            print(f"\n❌ 余额不足停止,已补{ok}个,补钱后续跑"); merge_and_write(nodes,kg65,cache); sys.exit(2)
        raise
    merge_and_write(nodes,kg65,cache)
    print(f"\n完成: 补判据{ok}个子节点 | 成本¥{cost:.3f}")

def merge_and_write(nodes,kg65,cache):
    """合并:37原节点原样 + 98子节点填入补全的判据,写enriched图谱"""
    out=[]
    for n in nodes:
        if n.get("_derived_from") or n.get("parent_node"):
            nid=n["node_id"]
            enr=cache.get(nid)
            if enr:
                n["judgment_criteria_for_mastery"]=enr["judgment_criteria_for_mastery"]
                n["common_failures"]=enr["common_failures"]
            else:
                # 没补成功的,继承父节点判据(兜底,不空)
                parent=n.get("_derived_from") or n.get("parent_node","")
                p=kg65.get(parent) or kg65.get(parent.replace("_","/"),{})
                n["judgment_criteria_for_mastery"]=p.get("judgment_criteria_for_mastery",[])
                n["common_failures"]=p.get("common_failures",[])
        out.append(n)
    with open(OUT,'w',encoding='utf-8') as f:
        for n in out: f.write(json.dumps(n,ensure_ascii=False)+"\n")
    print(f"✅ enriched图谱: {len(out)}节点 → {OUT.name}")

if __name__=="__main__":
    main()
