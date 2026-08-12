#!/usr/bin/env python3
"""
AI 诊断题 Provider —— 让诊断的前3-5题用上 AI 内化真题生成的题(而非只靠KG兜底)

为什么预生成而非实时:出题器含对抗验证+重生,单题十几~几十秒;诊断时实时生成会让用户干等。
所以分两步:
  1. 预生成(离线批量,外包DeepSeek):为每个节点生成N道已过对抗验证的诊断题,存 ai_diagnostic_bank/
  2. 运行时读取(诊断时,秒取):诊断引擎调 get_ai_questions(node_id) 直接拿预生成的题

诊断题 vs 普通生成题的区别:诊断题要标 level(L1基础→L4拔高)和 axis(考查维度),
便于逐层诊断。这里复用出题器生成,再按难度标 level。

用法(预生成):
  python3 scripts/ai_diagnostic_provider.py --generate --per-node 3      # 为所有节点各生成3题
  python3 scripts/ai_diagnostic_provider.py --generate --node 盐类水解-水解规律与溶液酸碱性 --per-node 4
运行时(被诊断引擎import):
  from scripts.ai_diagnostic_provider import get_ai_questions
"""
import json, sys, os, argparse, time
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")

BANK_DIR = SKILL_DIR / "data" / "ai_diagnostic_bank"
DIFF_TO_LEVEL = {"T1":"L1 基础概念","T2":"L2 应用","T3":"L3 综合","T4":"L4 拔高"}

def safe(name): return name.replace("/", "_").replace("（","(").replace("）",")")

# ── 运行时:诊断引擎调这个,秒取预生成题 ──
def _read_bank(fp):
    """读一个题库文件为题列表。"""
    out = []
    for l in open(fp, encoding='utf-8'):
        l = l.strip()
        if l:
            out.append(json.loads(l))
    return out

def _sort_by_difficulty(qs):
    """按难度 T1→T4 排序,实现逐层递进。"""
    order = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    return sorted(qs, key=lambda q: order.get(q.get("difficulty", "T2"), 2))

def get_ai_questions(node_id: str, limit: int = 4):
    """读取某节点预生成的AI诊断题。无则返回[]。诊断引擎据此插入AI题。

    取题三层策略:
      1) 节点自身的题文件;
      2) 子节点取不到 → 回退父节点(子→父);
      3) 父节点没有自己的题 → 聚合其名下所有子节点的题(父→子聚合)。
         按子节点轮流各取,再按难度排序,保证多样且逐层递进。
    """
    safe_id = safe(node_id)
    fp = BANK_DIR / f"{safe_id}.jsonl"
    if fp.exists():
        return _sort_by_difficulty(_read_bank(fp))[:limit]

    # 子节点没有 → 试父节点
    if "-" in node_id:
        parent_fp = BANK_DIR / f"{safe(node_id.split('-')[0])}.jsonl"
        if parent_fp.exists():
            return _sort_by_difficulty(_read_bank(parent_fp))[:limit]
        # 父节点也没自己的题 → 落到下面聚合逻辑(用父节点名聚合兄弟子节点)
        safe_id = safe(node_id.split('-')[0])

    # 父节点(或无题的子节点退到父名)→ 聚合该父节点名下所有子节点的题
    prefix = safe_id + "-"
    sibling_files = sorted(
        f for f in BANK_DIR.glob("*.jsonl") if f.stem.startswith(prefix)
    )
    if not sibling_files:
        return []
    # 每个子节点轮流取题(round-robin),保证覆盖多个子主题而非堆在一个
    per_node = [_sort_by_difficulty(_read_bank(f)) for f in sibling_files]
    merged = []
    for i in range(max((len(p) for p in per_node), default=0)):
        for p in per_node:
            if i < len(p):
                merged.append(p[i])
    return _sort_by_difficulty(merged)[:limit]

def to_diagnostic_format(q: dict, idx: int) -> dict:
    """把生成题转成诊断引擎的 DiagnosticQuestion 字段格式。"""
    diff = q.get("difficulty","T2")
    opts = q.get("options",{})
    # 选择题:把选项拼进prompt
    stem = q.get("stem","")
    if isinstance(opts,dict) and opts:
        stem += "\n" + "\n".join(f"{k}. {v}" for k,v in opts.items())
    return {
        "id": f"ai-{idx}",
        "level": DIFF_TO_LEVEL.get(diff, "L2 应用"),
        "axis": "concept",
        "prompt": stem,
        "look_for": q.get("design_note","")[:80],
        "source": "ai_generated",
        # 诊断校验要用的(答案+解析):
        "standard_answer": q.get("answer",""),
        "explanation": q.get("explanation",""),
        "options": opts,
        "difficulty": diff,
    }

# ── 预生成:离线批量(外包DeepSeek) ──
def generate(per_node: int, only_node: str = None):
    # 复用出题器的函数
    import importlib.util
    spec = importlib.util.spec_from_file_location("genq", SKILL_DIR/"scripts"/"generate_questions_phase1.py")
    genq = importlib.util.module_from_spec(spec); spec.loader.exec_module(genq)

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    meta = genq.load_meta()
    nodes = genq.list_pattern_nodes()
    if only_node:
        nodes = [n for n in nodes if n==only_node or n==safe(only_node)]
    print(f"预生成AI诊断题: {len(nodes)}节点 × {per_node}题")

    from adapters.llm_client import LLMClient
    lc = LLMClient(provider='deepseek', model='deepseek-chat',
                   api_key=os.environ.get("DEEPSEEK_API_KEY",""))
    total_cost=0.0; total_q=0
    for node in nodes:
        fp = BANK_DIR / f"{node}.jsonl"
        if fp.exists():  # 断点续跑:已生成的跳过
            continue
        pattern = genq.load_pattern(node)
        if not pattern: continue
        got=[]
        for _ in range(per_node):
            try:
                q,c,_attempts = genq.gen_one(lc, node, pattern, meta)
                total_cost+=c
                if q: got.append(q)
            except RuntimeError as e:
                if str(e).startswith("BALANCE"):
                    print(f"\n❌ 余额不足,已生成{total_q}题,补钱续跑"); return
        if got:
            with open(fp,'w',encoding='utf-8') as f:
                for q in got: f.write(json.dumps(q,ensure_ascii=False)+"\n")
            total_q+=len(got)
            print(f"  ✅ {node[:30]}: {len(got)}题 ¥{total_cost:.2f}")
    print(f"\n完成: {total_q}题 | 成本¥{total_cost:.2f} | 库:{BANK_DIR}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--generate",action="store_true")
    ap.add_argument("--per-node",type=int,default=3)
    ap.add_argument("--node",default=None)
    args=ap.parse_args()
    if args.generate:
        generate(args.per_node, args.node)
    else:
        print("用法: --generate --per-node N [--node 节点名]")
