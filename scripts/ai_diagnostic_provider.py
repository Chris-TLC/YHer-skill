#!/usr/bin/env python3
"""
AI diagnostic item provider -- allows the first 3-5 diagnostic items to use AI-generated items from the internalized real-exam pipeline (rather than relying solely on KG fallback).

Why pre-generate instead of real-time: the item generator contains adversarial validation + regeneration, taking tens of seconds per item; real-time generation during diagnosis would make the user wait.
So two steps:
  1. Pre-generate (offline batch, outsourced to DeepSeek): generate N adversarially-validated diagnostic items per node, store in ai_diagnostic_bank/
  2. Runtime retrieval (instant during diagnosis): the diagnostic engine calls get_ai_questions(node_id) to fetch pre-generated items directly

Diagnostic items vs ordinary generated items: diagnostic items must be tagged with level (L1 basic -> L4 advanced) and axis (assessment dimension),
to enable layered diagnosis. This reuses the item generator and tags by difficulty level.

Usage (pre-generation):
  python3 scripts/ai_diagnostic_provider.py --generate --per-node 3      # generate 3 items per node across all nodes
  python3 scripts/ai_diagnostic_provider.py --generate --node 盐类水解-水解规律与溶液酸碱性 --per-node 4
Runtime (imported by the diagnostic engine):
  from scripts.ai_diagnostic_provider import get_ai_questions
"""
import json, sys, os, argparse, time
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))
from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")

BANK_DIR = SKILL_DIR / "data" / "ai_diagnostic_bank"
DIFF_TO_LEVEL = {"T1":"L1 basic concept","T2":"L2 application","T3":"L3 comprehensive","T4":"L4 advanced"}

def safe(name): return name.replace("/", "_").replace("（","(").replace("）",")")

# -- Runtime: diagnostic engine calls this for instant pre-generated item retrieval --
def _read_bank(fp):
    """Read one bank file into an item list."""
    out = []
    for l in open(fp, encoding='utf-8'):
        l = l.strip()
        if l:
            out.append(json.loads(l))
    return out

def _sort_by_difficulty(qs):
    """Sort by difficulty T1->T4 to implement layered progression."""
    order = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    return sorted(qs, key=lambda q: order.get(q.get("difficulty", "T2"), 2))

def get_ai_questions(node_id: str, limit: int = 4):
    """Retrieve pre-generated AI diagnostic items for a node. Returns [] if none exist. The diagnostic engine uses this to insert AI items.

    Three-tier retrieval strategy:
      1) The node's own item file;
      2) Child node has none -> fall back to parent node (child->parent);
      3) Parent node has no items of its own -> aggregate all child-node items under that parent (parent->child aggregate).
         Round-robin across child nodes, then sort by difficulty, ensuring diversity and layered progression.
    """
    safe_id = safe(node_id)
    fp = BANK_DIR / f"{safe_id}.jsonl"
    if fp.exists():
        return _sort_by_difficulty(_read_bank(fp))[:limit]

    # Child node has none -> try parent node
    if "-" in node_id:
        parent_fp = BANK_DIR / f"{safe(node_id.split('-')[0])}.jsonl"
        if parent_fp.exists():
            return _sort_by_difficulty(_read_bank(parent_fp))[:limit]
        # Parent node also has no items of its own -> fall through to aggregation logic below (aggregate sibling child nodes under the parent name)
        safe_id = safe(node_id.split('-')[0])

    # Parent node (or child node that fell back to the parent name) -> aggregate all child-node items under this parent
    prefix = safe_id + "-"
    sibling_files = sorted(
        f for f in BANK_DIR.glob("*.jsonl") if f.stem.startswith(prefix)
    )
    if not sibling_files:
        return []
    # Round-robin across child nodes, ensuring coverage of multiple sub-topics rather than piling onto one
    per_node = [_sort_by_difficulty(_read_bank(f)) for f in sibling_files]
    merged = []
    for i in range(max((len(p) for p in per_node), default=0)):
        for p in per_node:
            if i < len(p):
                merged.append(p[i])
    return _sort_by_difficulty(merged)[:limit]

def to_diagnostic_format(q: dict, idx: int) -> dict:
    """Convert a generated item to the diagnostic engine's DiagnosticQuestion field format."""
    diff = q.get("difficulty","T2")
    opts = q.get("options",{})
    # Choice item: concatenate options into the prompt
    stem = q.get("stem","")
    if isinstance(opts,dict) and opts:
        stem += "\n" + "\n".join(f"{k}. {v}" for k,v in opts.items())
    return {
        "id": f"ai-{idx}",
        "level": DIFF_TO_LEVEL.get(diff, "L2 application"),
        "axis": "concept",
        "prompt": stem,
        "look_for": q.get("design_note","")[:80],
        "source": "ai_generated",
        # For diagnostic validation (answer + explanation):
        "standard_answer": q.get("answer",""),
        "explanation": q.get("explanation",""),
        "options": opts,
        "difficulty": diff,
    }

# -- Pre-generation: offline batch (outsourced to DeepSeek) --
def generate(per_node: int, only_node: str = None):
    # Reuse the item-generator functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("genq", SKILL_DIR/"scripts"/"generate_questions_phase1.py")
    genq = importlib.util.module_from_spec(spec); spec.loader.exec_module(genq)

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    meta = genq.load_meta()
    nodes = genq.list_pattern_nodes()
    if only_node:
        nodes = [n for n in nodes if n==only_node or n==safe(only_node)]
    print(f"Pre-generating AI diagnostic items: {len(nodes)} nodes × {per_node} items")

    from adapters.llm_client import LLMClient
    lc = LLMClient(provider='deepseek', model='deepseek-chat',
                   api_key=os.environ.get("DEEPSEEK_API_KEY",""))
    total_cost=0.0; total_q=0
    for node in nodes:
        fp = BANK_DIR / f"{node}.jsonl"
        if fp.exists():  # Checkpoint resume: skip already-generated nodes
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
                    print(f"\nERROR: Insufficient balance; {total_q} items generated; top up and resume"); return
        if got:
            with open(fp,'w',encoding='utf-8') as f:
                for q in got: f.write(json.dumps(q,ensure_ascii=False)+"\n")
            total_q+=len(got)
            print(f"  OK {node[:30]}: {len(got)} items ¥{total_cost:.2f}")
    print(f"\nComplete: {total_q} items | cost ¥{total_cost:.2f} | bank: {BANK_DIR}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--generate",action="store_true")
    ap.add_argument("--per-node",type=int,default=3)
    ap.add_argument("--node",default=None)
    args=ap.parse_args()
    if args.generate:
        generate(args.per_node, args.node)
    else:
        print("Usage: --generate --per-node N [--node node_name]")
