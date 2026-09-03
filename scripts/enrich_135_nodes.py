#!/usr/bin/env python3
"""
135-node criteria refinement -- backfill judgment_criteria + common_failures (diagnostic lifeblood) for the 98 derived child nodes.

Background: when expanding nodes, the 98 child nodes only inherited their parent node's criteria (too generic). Diagnosis/capability profiling needs child-node refinement of:
  - judgment_criteria_for_mastery: objective criteria for mastering this child node (string array)
  - common_failures: typical mistakes for this child node (cause/symptom/diagnostic_question object array)

Approach: parent-node criteria + child-node real exam items + child-node description -> DeepSeek generates refined criteria. Fields strictly aligned with the 65-node structure.
Output: knowledge_graph_150_enriched.jsonl (37 original nodes as-is + 98 child nodes with backfilled criteria)
Checkpoint resume: already-backfilled child nodes are skipped.
Usage: python3 scripts/enrich_135_nodes.py
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
CACHE = SKILL_DIR / "data" / "_enrich_cache.json"   # Checkpoint resume: child node -> criteria

LLM_MODEL = "deepseek-chat"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
RETRY_MAX = 3

def safe(n): return n.replace("/","_").replace("（","(").replace("）",")")

ENRICH_SYS = """You are a Shanghai high school chemistry teaching diagnostician. Given a fine-grained knowledge child node (along with its parent node's criteria and several real exam items for this child node), you must generate dedicated "mastery criteria" and "common failures" for this child node, for diagnosing whether a student truly masters the child node.

Output strict JSON (no markdown):
{
  "judgment_criteria_for_mastery": ["<criterion 1: can ...>","<criterion 2>","<criterion 3>","<criterion 4>"],
  "common_failures": [
    {"cause":"<root error cause>","symptom":"<outward manifestation: how the student will err>","diagnostic_question":"<one sentence that can expose this error in 3 seconds>"}
  ]
}
Requirements:
- judgment_criteria: about 4 criteria, targeting the specific abilities of this child node (not the generic parent-node criteria), objectively testable.
- common_failures: at least 4, drawn from real error points reflected in the real exam items; diagnostic_question must enable rapid diagnosis.
- Tightly focused on this child node; do not write generic parent-node content."""

def gen_enrich(lc, sub_name, parent_name, parent_crit, reals):
    sample = reals[:8]
    lines = [f"【Child node】{sub_name}", f"【Parent node】{parent_name}",
             f"【Parent node criteria (too generic, needs refinement)】{json.dumps(parent_crit,ensure_ascii=False)[:400]}",
             "", "【Real exam items for this child node】"]
    for i,q in enumerate(sample):
        lines.append(f"{i+1}. {q.get('stem','')[:100]}")
    prompt = "\n".join(lines) + f"\n\nPlease generate dedicated mastery criteria and common failures for child node「{sub_name}」."
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
            if any(k in es for k in ['insufficient','余额','欠费','balance']):  # Keep Chinese error strings as data
                raise RuntimeError(f"BALANCE:{ex}")
            time.sleep(min(15,2**(attempt+1)))
    return None, 0

def load_reals(sub_safe):
    fp = CLUSTER150 / f"{sub_safe}.jsonl"
    if fp.exists():
        return [json.loads(l) for l in open(fp,encoding='utf-8') if l.strip()]
    return []

def main():
    if not DEEPSEEK_KEY: print("ERROR: missing DEEPSEEK_API_KEY"); sys.exit(1)
    nodes=[json.loads(l) for l in open(KG150,encoding='utf-8') if l.strip()]
    kg65={json.loads(l)["node_id"]:json.loads(l) for l in open(KG65,encoding='utf-8') if l.strip()}

    cache={}
    if CACHE.exists(): cache=json.loads(CACHE.read_text(encoding='utf-8'))

    subs=[n for n in nodes if n.get("_derived_from") or n.get("parent_node")]
    print(f"Child nodes awaiting criteria backfill: {len(subs)}, already backfilled (skipping): {len([s for s in subs if s['node_id'] in cache])}")

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
                    print(f"  OK [{ok}] {nid[:30]} ¥{cost:.3f}")
            else:
                print(f"  WARNING: {nid} criteria backfill failed; keeping parent node criteria")
    except RuntimeError as e:
        if str(e).startswith("BALANCE"):
            print(f"\nERROR: insufficient balance; stopped after {ok} backfills; top up and resume"); merge_and_write(nodes,kg65,cache); sys.exit(2)
        raise
    merge_and_write(nodes,kg65,cache)
    print(f"\nComplete: backfilled {ok} child nodes | cost ¥{cost:.3f}")

def merge_and_write(nodes,kg65,cache):
    """Merge: 37 original nodes as-is + 98 child nodes with backfilled criteria; write enriched graph."""
    out=[]
    for n in nodes:
        if n.get("_derived_from") or n.get("parent_node"):
            nid=n["node_id"]
            enr=cache.get(nid)
            if enr:
                n["judgment_criteria_for_mastery"]=enr["judgment_criteria_for_mastery"]
                n["common_failures"]=enr["common_failures"]
            else:
                # Backfill unsuccessful; inherit parent node criteria (fallback, not left empty)
                parent=n.get("_derived_from") or n.get("parent_node","")
                p=kg65.get(parent) or kg65.get(parent.replace("_","/"),{})
                n["judgment_criteria_for_mastery"]=p.get("judgment_criteria_for_mastery",[])
                n["common_failures"]=p.get("common_failures",[])
        out.append(n)
    with open(OUT,'w',encoding='utf-8') as f:
        for n in out: f.write(json.dumps(n,ensure_ascii=False)+"\n")
    print(f"Enriched graph: {len(out)} nodes -> {OUT.name}")

if __name__=="__main__":
    main()
