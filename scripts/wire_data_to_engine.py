#!/usr/bin/env python3
"""
数据接线 —— 把6695题三重验证数据 + 135节点聚类结果,转成诊断引擎要的 item_bank 格式

零成本(不调DeepSeek):纯字段映射。
  新数据字段                → 引擎item字段
  q_num/_source_file        → item_id(稳定hash)
  _cluster_primary(135节点)  → kg_nodes[](诊断按节点筛题靠这个)
  question_type/difficulty  → 同名
  region                    → "上海卷"(全是上海卷)
  answer                    → standard_solution.standard_answer + final_answers
  explanation               → standard_solution.solution_steps + 转成 rubric 得分点
  confidence/_issues        → 保留供质量过滤

诊断引擎设计:rubric可空时退化到135节点的judgment_criteria,所以即使rubric粗,也能客观校验。

输出: data/item_bank/chemistry_v3_6695.jsonl(新),旧 chemistry_solved.jsonl 不动(备份)
用法: python3 scripts/wire_data_to_engine.py
"""
import json, sys, hashlib, re
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
SRC = SKILL_DIR / "data" / "from_pdf" / "all_from_pdf_v3.jsonl"
ASSIGN = SKILL_DIR / "data" / "clustered_questions" / "_assignments.jsonl"
SUBDEF = SKILL_DIR / "data" / "clustered_questions_150" / "_subnode_definitions.json"
OUT = SKILL_DIR / "data" / "item_bank" / "chemistry_v3_6695.jsonl"

def qid(q):
    raw=f"{q.get('_source_file','')}|{q.get('q_num','')}|{q.get('stem','')[:40]}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]

def explanation_to_rubric(expl, answer):
    """把解析拆成得分点。字段严格对齐引擎 RubricPoint:point_id/desc/keywords/score/must_have/kg_node"""
    rubric=[]
    # 答案本身作第一个must_have得分点
    if answer:
        rubric.append({
            "point_id": "ans",
            "desc": f"得出正确答案: {str(answer)[:60]}",
            "keywords": [str(answer)[:20]] if answer else [],
            "score": 2.0, "must_have": True, "kg_node": ""
        })
    if expl:
        parts = [p.strip() for p in re.split(r'[。;；\n]', expl) if len(p.strip())>=8]
        for i,p in enumerate(parts[:6]):
            # 抽关键词:句中的化学式/数字/关键名词(简单取2-8字的片段)
            kws = re.findall(r'[A-Za-z0-9₀-₉⁰-⁹]+|[一-龥]{2,4}', p)[:5]
            rubric.append({
                "point_id": f"p{i+1}",
                "desc": p[:120],
                "keywords": kws,
                "score": 1.0,
                "must_have": (i==0),  # 第一条(核心结论)设must_have
                "kg_node": ""
            })
    return rubric

def main():
    # 加载135节点归类(qid → primary子节点)
    node_map={}
    if ASSIGN.exists():
        for l in open(ASSIGN,encoding='utf-8'):
            if l.strip():
                d=json.loads(l); node_map[d["id"]]=d.get("primary","")
    # 加载子节点映射(过载父节点→子节点),用于把题挂到最细节点
    # 这里直接用聚类assignment的primary(已是65节点);若该节点被拆,引擎仍可用父节点名匹配
    print(f"加载节点归类: {len(node_map)}题")

    n=0; wrote=0; skip_lowconf=0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT,'w',encoding='utf-8') as fo:
        for l in open(SRC,encoding='utf-8'):
            l=l.strip()
            if not l: continue
            q=json.loads(l); n+=1
            iid=qid(q)
            # 质量过滤:confidence过低(<0.6)的不进诊断题库(避免脏题误导诊断)
            if q.get("confidence",1) < 0.6:
                skip_lowconf+=1; continue
            node = node_map.get(iid,"")  # 135/65节点
            kg_nodes=[node] if node else []
            # 原始knowledge_points也保留作辅助匹配
            kp=q.get("knowledge_points") or []
            ans=q.get("answer","")
            item={
                "item_id": iid,
                "source": q.get("_source_file",""),
                "region": "上海卷",
                "stem": q.get("stem",""),
                "options": q.get("options",{}),
                "question_type": q.get("question_type",""),
                "difficulty": q.get("difficulty",""),
                "kg_nodes": kg_nodes,
                "knowledge_points": kp,
                "standard_solution": {
                    "standard_answer": ans,
                    "final_answers": [ans] if ans else [],
                    "solution_steps": [s.strip() for s in re.split(r'[。\n]', q.get("explanation","")) if s.strip()][:8],
                    "key_insight": (q.get("explanation","")[:100]),
                },
                "rubric": explanation_to_rubric(q.get("explanation",""), ans),
                "confidence": q.get("confidence",0),
                "verification_status": q.get("verification_status",""),
                "_pipeline": "v3.4",
            }
            fo.write(json.dumps(item,ensure_ascii=False)+"\n"); wrote+=1
    print(f"源题:{n} | 写入诊断题库:{wrote} | 跳过低置信(<0.6):{skip_lowconf}")
    print(f"输出: {OUT}")
    # 抽查
    print("\n=== 抽查转换结果(1道) ===")
    first=json.loads(open(OUT,encoding='utf-8').readline())
    print(f"item_id:{first['item_id']} | 节点:{first['kg_nodes']} | 题型:{first['question_type']}/{first['difficulty']}")
    print(f"答案:{first['standard_solution']['standard_answer']}")
    print(f"rubric得分点数:{len(first['rubric'])}, must_have:{sum(1 for r in first['rubric'] if r['must_have'])}")
    print(f"首个得分点:{first['rubric'][0]['desc'][:60] if first['rubric'] else '无'}")

if __name__=="__main__":
    main()
