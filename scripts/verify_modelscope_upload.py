"""Verify that the ModelScope dataset is complete and readable"""
import os
import sys
from modelscope.hub.api import HubApi

api = HubApi()

print("Checking ChrisTLC/YHer-skill-embeddings...")
try:
    files = api.get_dataset_files("ChrisTLC/YHer-skill-embeddings")
except Exception as e:
    print(f"ERROR: cannot access dataset: {e}")
    sys.exit(1)

print(f"   {len(files)} files total\n")

# Core file manifest
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
    print(f"  {'OK' if size_mb > 0 else 'WARN'} {fpath:<45} {size_mb:>8.2f} MB")

# Core file check
print(f"\nCore file check:")
core_ok = True
for fname in expected_core:
    ok = any(fname in p for p in present)
    if ok:
        print(f"  OK {fname}")
    else:
        print(f"  MISSING {fname}!")
        core_ok = False

# BM25 file check
print(f"\nBM25 file check:")
bm25_ok = True
for fname in bm25_files:
    ok = any(fname in p for p in present)
    if ok:
        print(f"  OK {fname}")
    else:
        print(f"  MISSING {fname}!")
        bm25_ok = False

print(f"\n{'=' * 50}")
if core_ok and bm25_ok:
    print("ModelScope dataset complete and readable; T0 can proceed.")
else:
    print("Files incomplete; re-upload required!")
    sys.exit(1)
