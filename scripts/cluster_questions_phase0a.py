#!/usr/bin/env python3
"""
Phase 0-A 知识点聚类 —— 把 6438 题按语义归类到 65 节点知识图谱

策略(Chris 拍板:质量优先,多用 DeepSeek):
  - DeepSeek 语义归类为主,不做高频标签字符串硬映射(避免粗暴归一丢精度)
  - 每题:从 65 标准节点选 1 主节点(必) + 0-1 副节点(可选,跨节点综合题用)+ 置信度
  - 批量(每批 BATCH 题)调用,降本但不牺牲单题判断
  - 断点续跑:已归类的题(by stable id)跳过;中断后重跑只补未完成
  - 绝不修改源 jsonl;输出独立到 clustered_questions/

用法:
  python3 scripts/cluster_questions_phase0a.py            # 全量
  python3 scripts/cluster_questions_phase0a.py --limit 40 # 冒烟测试
"""

import json, sys, time, os, hashlib, argparse
from pathlib import Path
from collections import Counter, defaultdict

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")
from adapters.llm_client import LLMClient

# ── 路径 ──
DATA_DIR     = SKILL_DIR / "data" / "from_pdf"
SRC_JSONL    = DATA_DIR / "all_from_pdf_v3.jsonl"
KG_FILE      = SKILL_DIR / "data" / "knowledge_graph_full.jsonl"
OUT_DIR      = SKILL_DIR / "data" / "clustered_questions"
ASSIGN_FILE  = OUT_DIR / "_assignments.jsonl"      # 每题归类结果(断点续跑依据)
PROGRESS_F   = OUT_DIR / "_cluster_progress.json"
REPORT_F     = OUT_DIR / "_cluster_report.json"

LLM_MODEL = "deepseek-chat"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BATCH = 8                  # 每批题数
RETRY_MAX = 3

# ── 题目稳定 id(源文件+题号+题干前40字)──
def qid(q: dict) -> str:
    raw = f"{q.get('_source_file','')}|{q.get('q_num','')}|{q.get('stem','')[:40]}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]

def load_nodes():
    nodes = []
    for l in open(KG_FILE, encoding='utf-8'):
        if l.strip():
            n = json.loads(l)
            nodes.append((n['category'], n['node_id']))
    return nodes

def build_node_menu(nodes):
    """按类别分组的节点清单,供 prompt 用"""
    by_cat = defaultdict(list)
    for cat, nid in nodes:
        by_cat[cat].append(nid)
    lines = []
    for cat in sorted(by_cat):
        lines.append(f"【{cat}】" + "、".join(by_cat[cat]))
    return "\n".join(lines)

CLUSTER_SYS = """你是上海高考化学知识点归类专家。给你一道题和标准知识节点清单,你要判断这道题主要考查哪个节点。

规则:
1. 必须从给定的标准节点清单里选,不能自创节点名。
2. primary(主节点):这道题最核心考查的1个节点,必填。
3. secondary(副节点):若题目明显跨节点综合(如"氧化还原+电化学"),给1个副节点;否则留空字符串""。
4. confidence:你对主节点判断的置信度 0.0-1.0。
5. 依据题目的题干、原有知识点标注、题型综合判断,但最终归类必须落到标准节点。

只输出 JSONL,每题一行,严格格式:
{"id":"<题的id>","primary":"<节点名>","secondary":"<节点名或空>","confidence":0.95}
不要输出任何解释、表头、markdown。"""

def build_user_prompt(batch, node_menu):
    parts = ["【标准知识节点清单】", node_menu, "", "【待归类题目】"]
    for q in batch:
        kp = q.get('knowledge_points') or []
        kp_str = "、".join(k for k in kp if isinstance(k, str))[:80]
        parts.append(
            f'id={q["_qid"]} | 题型:{q.get("question_type","")} | 原标注:{kp_str}\n'
            f'题干:{q.get("stem","")[:150]}'
        )
    parts.append("\n请为以上每题输出一行归类 JSON。")
    return "\n".join(parts)

def parse_assignments(content, valid_nodes):
    """解析 DeepSeek 返回的 JSONL,校验节点合法性"""
    out = {}
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid_ = d.get("id", "")
        prim = d.get("primary", "").strip()
        sec  = d.get("secondary", "").strip()
        # 校验:主节点必须在合法集合;非法则丢弃该题(留待重试)
        if qid_ and prim in valid_nodes:
            if sec and sec not in valid_nodes:
                sec = ""
            out[qid_] = {"primary": prim, "secondary": sec,
                         "confidence": float(d.get("confidence", 0.0))}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前N题(冒烟)")
    args = ap.parse_args()

    if not DEEPSEEK_KEY:
        print("❌ 缺 DEEPSEEK_API_KEY"); sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = load_nodes()
    valid_nodes = {nid for _, nid in nodes}
    node_menu = build_node_menu(nodes)
    print(f"标准节点: {len(valid_nodes)} 个")

    # 加载全部题
    all_q = []
    for l in open(SRC_JSONL, encoding='utf-8'):
        l = l.strip()
        if not l: continue
        q = json.loads(l)
        q["_qid"] = qid(q)
        all_q.append(q)
    print(f"源题目: {len(all_q)} 题")

    # 断点续跑:已归类的 id
    done = {}
    if ASSIGN_FILE.exists():
        for l in open(ASSIGN_FILE, encoding='utf-8'):
            l = l.strip()
            if l:
                d = json.loads(l); done[d["id"]] = d
        print(f"已归类(续跑跳过): {len(done)} 题")

    pending = [q for q in all_q if q["_qid"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"待归类: {len(pending)} 题\n")
    if not pending:
        print("✅ 全部已归类,直接生成报告")
        write_report(all_q, done, nodes); return

    lc = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=DEEPSEEK_KEY)
    cost = 0.0
    t0 = time.time()
    fail_batches = 0

    with open(ASSIGN_FILE, 'a', encoding='utf-8') as fo:
        for i in range(0, len(pending), BATCH):
            batch = pending[i:i+BATCH]
            prompt = build_user_prompt(batch, node_menu)
            messages = [{"role": "system", "content": CLUSTER_SYS},
                        {"role": "user", "content": prompt}]
            got = {}
            err = None
            for attempt in range(RETRY_MAX):
                try:
                    r = lc.chat(messages, max_tokens=1500, temperature=0.1)
                    cost += r.get("cost_yuan", 0) or 0
                    got = parse_assignments(r["content"], valid_nodes)
                    if got: break
                    err = "空解析"
                except Exception as e:
                    err = str(e)[:100]
                    es = err.lower()
                    if any(k in es for k in ['insufficient','余额','欠费','balance','arrears']):
                        print(f"\n❌ 余额不足,立即停止: {err}")
                        print(f"已归类 {len(done)} 题,可补钱后 --resume 续跑")
                        write_report(all_q, done, nodes); sys.exit(2)
                    time.sleep(min(20, 2**(attempt+1)))

            # 落盘本批(只写成功归类的;未归类的留待下次重跑补)
            wrote = 0
            for q in batch:
                a = got.get(q["_qid"])
                if a:
                    rec = {"id": q["_qid"], "q_num": q.get("q_num"),
                           "source": q.get("_source_file"),
                           "primary": a["primary"], "secondary": a["secondary"],
                           "confidence": a["confidence"]}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[q["_qid"]] = rec; wrote += 1
            fo.flush()

            n_done = len(done)
            # 熔断只看"整批失败"(wrote==0,真API问题);部分失败(如7/8)是单题归类问题,正常,单题留待补跑,不计入
            if wrote == 0:
                fail_batches += 1
            else:
                fail_batches = 0   # 只要有产出就清零,要求"连续"整批失败才熔断
            if (i//BATCH) % 10 == 0 or i+BATCH >= len(pending):
                el = time.time()-t0
                print(f"  [{n_done}/{len(all_q)}] 本批{wrote}/{len(batch)} "
                      f"¥{cost:.3f} {el:.0f}s" + (f" ⚠️{err}" if err and wrote<len(batch) else ""))
            # 连续失败保护:连续5个整批都0产出才停(真API挂了),单批空解析会自动靠下批补
            if fail_batches >= 5:
                print(f"\n⚠️ 连续{fail_batches}个整批归类失败,停止报告。已完成{n_done}题。最后错误:{err}")
                break

            # 进度落盘供监控
            PROGRESS_F.write_text(json.dumps({
                "done": len(done), "total": len(all_q),
                "cost": round(cost, 3),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }, ensure_ascii=False), encoding='utf-8')

    print(f"\n本轮成本 ¥{cost:.3f}, 累计归类 {len(done)}/{len(all_q)}")
    write_report(all_q, done, nodes)

def write_report(all_q, done, nodes):
    """生成聚类分布报告 + 按节点拆分题目到 clustered_questions/<节点>.jsonl"""
    qmap = {q["_qid"]: q for q in all_q}
    # 按主节点分组
    by_node = defaultdict(list)
    sec_count = Counter()
    low_conf = 0
    unclustered = []
    for q in all_q:
        a = done.get(q["_qid"])
        if not a:
            unclustered.append(q.get("_source_file","?"))
            continue
        by_node[a["primary"]].append(q)
        if a.get("secondary"): sec_count[a["secondary"]] += 1
        if a.get("confidence", 1) < 0.6: low_conf += 1

    # 拆分落盘(每节点一个文件,文件名安全化)
    for node, qs in by_node.items():
        safe = node.replace("/", "_").replace("（","(").replace("）",")")
        fp = OUT_DIR / f"{safe}.jsonl"
        with open(fp, 'w', encoding='utf-8') as f:
            for q in qs:
                qo = {k: v for k, v in q.items() if k != "_qid"}
                qo["_cluster_primary"] = node
                qo["_cluster_secondary"] = done[q["_qid"]].get("secondary","")
                qo["_cluster_confidence"] = done[q["_qid"]].get("confidence",0)
                f.write(json.dumps(qo, ensure_ascii=False) + "\n")

    # 节点覆盖核对:65节点里哪些有题/无题/过载
    all_node_names = {nid for _, nid in nodes}
    covered = set(by_node.keys())
    empty = sorted(all_node_names - covered)
    counts = {n: len(qs) for n, qs in by_node.items()}
    overload = sorted(counts.items(), key=lambda x: -x[1])[:10]
    underfilled = sorted([(n, counts.get(n,0)) for n in covered], key=lambda x: x[1])[:10]

    report = {
        "总题数": len(all_q),
        "已归类": len(done),
        "未归类": len(unclustered),
        "覆盖节点数": f"{len(covered)}/{len(all_node_names)}",
        "空节点(真题无覆盖)": empty,
        "过载Top10(可能需拆细)": overload,
        "稀疏Top10(题最少)": underfilled,
        "低置信(<0.6)题数": low_conf,
        "作为副节点出现Top10": sec_count.most_common(10),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    REPORT_F.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print("\n" + "="*60)
    print("聚类分布报告")
    print("="*60)
    print(f"已归类 {len(done)}/{len(all_q)} | 未归类 {len(unclustered)} | "
          f"覆盖节点 {len(covered)}/{len(all_node_names)} | 低置信 {low_conf}")
    print(f"\n过载Top10(题最多→可能需拆细):")
    for n, c in overload: print(f"  {c:>4}  {n}")
    if empty:
        print(f"\n⚠️ 空节点({len(empty)}个,真题无覆盖):")
        print("  " + "、".join(empty))
    print(f"\n完整报告: {REPORT_F}")
    print(f"分节点题库: {OUT_DIR}/<节点>.jsonl")

if __name__ == "__main__":
    main()
