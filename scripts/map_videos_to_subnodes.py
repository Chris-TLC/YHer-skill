#!/usr/bin/env python3
"""
视频映射精细化 —— 给135节点的98个子节点,从父节点视频里挑最匹配的(DeepSeek判断)

背景:65父节点都有视频(每个含多个视频P),但98子节点没单独映射。
做法(适合外包DeepSeek):父节点的若干视频P + 子节点主题描述 → DeepSeek判断哪几个P最匹配该子节点。
  - 父节点有视频的子节点(94个):DeepSeek从父节点视频里挑1-3个最贴子节点主题的
  - 父节点也没视频的(4个):标记空,前端显示"整理中"
输出:更新 knowledge_graph_150_enriched.jsonl,给子节点填 recommended_videos
断点续跑:已映射的子节点跳过。低成本(每子节点1次轻量调用)。

用法: python3 scripts/map_videos_to_subnodes.py
"""
import json, sys, os, time
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")
from adapters.llm_client import LLMClient

KG150 = SKILL_DIR / "data" / "knowledge_graph_150_enriched.jsonl"
KG65  = SKILL_DIR / "data" / "knowledge_graph_full.jsonl"
CACHE = SKILL_DIR / "data" / "_video_map_cache.json"

LLM_MODEL = "deepseek-chat"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
RETRY_MAX = 3

SYS = """你是上海化学教学视频匹配助手。给你一个细分知识子节点(及其主题),和它父节点下的几个一化儿教学视频(每个视频有编号和"讲什么"),你要判断哪几个视频最匹配这个子节点的主题。

只输出JSON(不要markdown):{"picks":[视频编号,...]}
规则:挑1-3个最贴合子节点主题的视频编号(从给定列表选,编号即列表序号从1开始)。子节点是细分主题,挑讲得最对口的;真没特别对口的就挑父节点最核心的1个。"""

def pick_videos(lc, sub_name, sub_desc, parent_videos):
    lines=[f"【子节点】{sub_name}", f"【子节点主题】{sub_desc[:100]}", "", "【父节点的一化儿视频】"]
    for i,v in enumerate(parent_videos):
        lines.append(f"{i+1}. {v.get('what_you_learn','')[:80]}")
    prompt="\n".join(lines)+f"\n\n请挑出最匹配「{sub_name}」的视频编号。"
    for attempt in range(RETRY_MAX):
        try:
            r=lc.chat([{"role":"system","content":SYS},{"role":"user","content":prompt}],
                      max_tokens=200,temperature=0.2)
            cost=r.get("cost_yuan",0) or 0
            c=r["content"].strip(); s=c.find("{"); e=c.rfind("}")
            if s>=0:
                d=json.loads(c[s:e+1])
                picks=[i for i in d.get("picks",[]) if isinstance(i,int) and 1<=i<=len(parent_videos)]
                if picks: return [parent_videos[i-1] for i in picks[:3]], cost
        except Exception as ex:
            es=str(ex).lower()
            if any(k in es for k in ['insufficient','余额','欠费','balance']):
                raise RuntimeError(f"BALANCE:{ex}")
            time.sleep(min(15,2**(attempt+1)))
    # 失败兜底:返回父节点前2个视频
    return parent_videos[:2], 0

def main():
    if not DEEPSEEK_KEY: print("❌ 缺 DEEPSEEK_API_KEY"); sys.exit(1)
    kg65={json.loads(l)["node_id"]:json.loads(l) for l in open(KG65,encoding='utf-8') if l.strip()}
    nodes=[json.loads(l) for l in open(KG150,encoding='utf-8') if l.strip()]
    cache=json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}

    subs=[n for n in nodes if n.get('_derived_from') or n.get('parent_node')]
    print(f"子节点: {len(subs)}个, 已映射(跳过):{len([s for s in subs if s['node_id'] in cache])}")

    lc=LLMClient(provider='deepseek',model=LLM_MODEL,api_key=DEEPSEEK_KEY)
    cost=0.0; ok=0; orphan=0
    try:
        for n in subs:
            nid=n["node_id"]
            if nid in cache: continue
            parent=n.get("_derived_from") or n.get("parent_node","")
            p=kg65.get(parent) or kg65.get(parent.replace("_","/"),{})
            pv=p.get("recommended_videos",[])
            if not pv:
                cache[nid]=[]; orphan+=1
                CACHE.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8'); continue
            picks,c=pick_videos(lc,nid,n.get("desc",""),pv); cost+=c
            cache[nid]=picks; ok+=1
            CACHE.write_text(json.dumps(cache,ensure_ascii=False),encoding='utf-8')
            if ok%15==0 or ok<=2: print(f"  ✅ [{ok}] {nid[:28]} → {len(picks)}个视频 ¥{cost:.3f}")
    except RuntimeError as e:
        if str(e).startswith("BALANCE"):
            print(f"\n❌ 余额不足,已映射{ok},补钱续跑"); write_back(nodes,cache); sys.exit(2)
        raise
    write_back(nodes,cache)
    print(f"\n完成: 映射{ok}子节点, 孤儿(父节点也无视频){orphan} | 成本¥{cost:.3f}")

def write_back(nodes,cache):
    """把映射的视频写回135图谱的子节点"""
    for n in nodes:
        nid=n.get("node_id")
        if nid in cache and cache[nid]:
            n["recommended_videos"]=cache[nid]
    with open(KG150,'w',encoding='utf-8') as f:
        for n in nodes: f.write(json.dumps(n,ensure_ascii=False)+"\n")
    # 统计
    have=sum(1 for n in nodes if n.get("recommended_videos"))
    print(f"✅ 写回完成: {have}/{len(nodes)}节点现在有视频")

if __name__=="__main__":
    main()
