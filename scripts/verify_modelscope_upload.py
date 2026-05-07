"""验证 ModelScope dataset 完整可读"""
import os
import sys
from modelscope.hub.api import HubApi

api = HubApi()

print("🔍 检查 ChrisTLC/YHer-skill-embeddings...")
try:
    files = api.get_dataset_files("ChrisTLC/YHer-skill-embeddings")
except Exception as e:
    print(f"❌ 无法访问 dataset: {e}")
    sys.exit(1)

print(f"   共 {len(files)} 个文件\n")

# 核心文件清单
expected_core = [
    "chunks.faiss",
    "chunks_meta.jsonl",
    "knowledge_graph.faiss",
    "knowledge_graph_meta.jsonl",
    "exam_patterns.faiss",
    "exam_patterns_meta.jsonl",
    "thinking_patterns.faiss",
    "thinking_patterns_meta.jsonl",
]

bm25_files = [
    "bm25/tfidf_matrix.npz",
    "bm25/chunk_ids.json",
    "bm25/vectorizer.pkl",
]

present = set()
for f in files:
    fpath = f.get('path', f.get('name', str(f)))
    size_mb = f.get('size', 0) / 1024 / 1024
    present.add(fpath)
    print(f"  {'✅' if size_mb > 0 else '⚠️'} {fpath:<45} {size_mb:>8.2f} MB")

# 核心文件检查
print(f"\n📋 核心文件检查:")
core_ok = True
for fname in expected_core:
    ok = any(fname in p for p in present)
    if ok:
        print(f"  ✅ {fname}")
    else:
        print(f"  ❌ {fname} 缺失!")
        core_ok = False

# BM25 文件检查
print(f"\n📋 BM25 文件检查:")
bm25_ok = True
for fname in bm25_files:
    ok = any(fname in p for p in present)
    if ok:
        print(f"  ✅ {fname}")
    else:
        print(f"  ❌ {fname} 缺失!")
        bm25_ok = False

print(f"\n{'=' * 50}")
if core_ok and bm25_ok:
    print("✅ ModelScope dataset 完整可读，可以开始 T0。")
else:
    print("❌ 文件不完整，需要重新上传!")
    sys.exit(1)
