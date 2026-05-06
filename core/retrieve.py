#!/usr/bin/env python3
"""阶段 7：RAG 检索接口。SKILL 在阶段 8 使用的核心检索器。"""

import json
import time
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── 路径（调整到 SKILL 包的 data/ 目录） ──────
SKILL_DIR = Path(__file__).parent.parent
EMBED_DIR = SKILL_DIR / "data" / "embeddings"
BM25_DIR = EMBED_DIR / "bm25"

# ── 已加载的缓存 ─────────────────────────────────
_model = None
_indices = {}
_metas = {}
_bm25_searcher = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
    return _model


def _load_index(name):
    if name not in _indices:
        idx_path = EMBED_DIR / f"{name}.faiss"
        if idx_path.exists():
            _indices[name] = faiss.read_index(str(idx_path))
        else:
            raise FileNotFoundError(f"索引不存在: {idx_path}")
    return _indices[name]


def _load_meta(name):
    if name not in _metas:
        meta_path = EMBED_DIR / f"{name}_meta.jsonl"
        if meta_path.exists():
            data = []
            with open(meta_path) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            _metas[name] = data
        else:
            _metas[name] = []
    return _metas[name]


def _load_bm25():
    global _bm25_searcher
    if _bm25_searcher is None:
        try:
            from whoosh import index
            ix = index.open_dir(str(BM25_DIR))
            _bm25_searcher = ix.searcher()
        except Exception:
            _bm25_searcher = False
    return _bm25_searcher


# ── 编码 ─────────────────────────────────────────
def encode_query(query: str, prefix: str = ""):
    """编码查询向量（L2 归一化）"""
    model = _load_model()
    if prefix:
        query = f"{prefix}{query}"
    emb = model.encode([query], normalize_embeddings=True)
    return emb.astype(np.float32)


# ── 召回通道 ─────────────────────────────────────
def _vector_search(index_name: str, query_emb, top_k: int = 30) -> List[Dict]:
    """Channel A: 向量召回"""
    idx = _load_index(index_name)
    meta = _load_meta(index_name)
    D, I = idx.search(query_emb, min(top_k, idx.ntotal))

    results = []
    for rank, (meta_idx, score) in enumerate(zip(I[0], D[0])):
        if meta_idx >= 0 and meta_idx < len(meta):
            results.append({
                **meta[meta_idx],
                "score": float(score),
                "rank": rank + 1,
                "channel": "vector",
            })
    return results


def _tag_filter(query: str, top_k: int = 30) -> List[Dict]:
    """Channel B: 标签过滤召回"""
    meta = _load_meta("chunks")
    if not meta:
        return []

    kg_meta = _load_meta("knowledge_graph")
    ep_meta = _load_meta("exam_patterns")

    matched_nodes = set()
    for node in kg_meta:
        nid = node.get("node_id", "")
        if _contains_any(query, nid):
            matched_nodes.add(nid)
        for cf in node.get("common_failures", []):
            cause = cf.get("cause", "")
            symptom = cf.get("symptom", "")
            if _contains_any(query, cause) or _contains_any(query, symptom):
                matched_nodes.add(nid)

    matched_patterns = set()
    for ep in ep_meta:
        pid = ep.get("pattern_id", "")
        if _contains_any(query, pid):
            matched_patterns.add(pid)
        for v in ep.get("question_variants", []):
            if _contains_any(query, v.get("variant_name", "")):
                matched_patterns.add(pid)

    scored = []
    for i, c in enumerate(meta):
        score = 0.0
        kts = c.get("knowledge_topic", [])
        eps = c.get("exam_pattern", [])

        for kt in kts:
            if kt in matched_nodes:
                score += 0.5
        for ep in eps:
            if ep in matched_patterns:
                score += 0.3
        for kt in kts:
            if _contains_any(query, kt):
                score += 0.2
        for ep in eps:
            if _contains_any(query, ep):
                score += 0.15

        if score > 0:
            scored.append((i, score, c))

    scored.sort(key=lambda x: -x[1])
    results = []
    for rank, (meta_idx, score, c) in enumerate(scored[:top_k]):
        results.append({
            **c,
            "score": score,
            "rank": rank + 1,
            "channel": "tag_filter",
        })
    return results


def _bm25_search(query: str, top_k: int = 10) -> List[Dict]:
    """Channel C: BM25 关键词召回（sklearn TF-IDF + jieba 分词）"""
    global _bm25_searcher
    if _bm25_searcher is False:
        return []
    if _bm25_searcher is not None:
        vectorizer, tfidf_matrix, chunk_ids = _bm25_searcher
    else:
        try:
            import pickle
            from scipy.sparse import load_npz
            bm25_dir = EMBED_DIR / "bm25"
            with open(bm25_dir / "vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            tfidf_matrix = load_npz(str(bm25_dir / "tfidf_matrix.npz"))
            with open(bm25_dir / "chunk_ids.json") as f:
                chunk_ids = json.load(f)
            _bm25_searcher = (vectorizer, tfidf_matrix, chunk_ids)
        except Exception:
            _bm25_searcher = False
            return []

    try:
        q_vec = vectorizer.transform([query])
    except Exception:
        return []

    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    meta = _load_meta("chunks")
    id_to_chunk = {c.get("chunk_id", ""): c for c in meta}

    results = []
    for rank, idx in enumerate(top_idx):
        if sims[idx] <= 0:
            break
        cid = chunk_ids[idx]
        chunk = id_to_chunk.get(cid, {})
        results.append({
            **chunk,
            "score": float(sims[idx]),
            "rank": rank + 1,
            "channel": "bm25",
        })
    return results


def _contains_any(text: str, target: str) -> bool:
    """模糊匹配：target 的任意连续子串是否在 text 中"""
    if not target or not text:
        return False
    if len(target) >= 3 and target in text:
        return True
    if len(target) >= 4:
        matches = sum(1 for i in range(len(target) - 1) if target[i:i + 2] in text)
        if matches >= len(target) * 0.6:
            return True
    return False


# ── RRF 融合 ─────────────────────────────────────
def _rrf_merge(results_list: List[List[Dict]], weights=None, k=60) -> List[Dict]:
    """Reciprocal Rank Fusion"""
    if weights is None:
        weights = [0.5, 0.3, 0.2]

    id_scores = defaultdict(float)
    id_items = {}

    for channel_weight, channel_results in zip(weights, results_list):
        if not channel_results:
            continue
        for r in channel_results:
            cid = r.get("chunk_id", "")
            rank = r.get("rank", len(channel_results))
            rrf_score = channel_weight * (1.0 / (k + rank))
            id_scores[cid] += rrf_score
            if cid not in id_items:
                id_items[cid] = r

    merged = []
    for cid, score in sorted(id_scores.items(), key=lambda x: -x[1]):
        item = {**id_items[cid], "rrf_score": score}
        merged.append(item)

    return merged


# ── 重排序 ───────────────────────────────────────
def _rerank(results: List[Dict], query: str) -> List[Dict]:
    """一化儿独有信号加分"""
    for r in results:
        bonus = 0.0
        if r.get("has_jiehe_brand"):
            bonus += 0.05
        if r.get("has_warning"):
            bonus += 0.03
        kts = " ".join(r.get("knowledge_topic", []))
        eps = " ".join(r.get("exam_pattern", []))
        if _contains_any(query, r.get("text_preview", "")[:200]):
            bonus += 0.02
        if _contains_any(kts, query) or _contains_any(query, kts):
            bonus += 0.02
        r["rerank_bonus"] = bonus
        r["final_score"] = r.get("rrf_score", r.get("score", 0)) + bonus

    results.sort(key=lambda x: -x.get("final_score", 0))
    return results


# ── 主检索类 ─────────────────────────────────────
class YihuierRetriever:
    """一化儿 AI 化学助手 RAG 检索器"""

    def __init__(self, embeddings_dir: str = None):
        if embeddings_dir is None:
            embeddings_dir = str(EMBED_DIR)
        self.embeddings_dir = Path(embeddings_dir)
        self.model = _load_model()

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """混合检索主接口"""
        q_emb = encode_query(query)

        vector_results = _vector_search("chunks", q_emb, top_k=30)
        tag_results = _tag_filter(query, top_k=30)
        bm25_results = _bm25_search(query, top_k=10)

        merged = _rrf_merge([vector_results, tag_results, bm25_results])
        reranked = _rerank(merged, query)

        return reranked[:top_k]

    def retrieve_with_diagnosis(self, query: str) -> Dict:
        """带诊断功能的检索：知识点 + 题型 + 招式 + chunks"""
        t0 = time.time()
        q_emb = encode_query(query)

        # 1. 检索知识节点
        kg_results = _vector_search("knowledge_graph", q_emb, top_k=5)
        kg_meta = _load_meta("knowledge_graph")
        kg_map = {k["node_id"]: k for k in kg_meta}
        related_nodes = []
        prerequisites = []
        for r in kg_results[:3]:
            nid = r.get("node_id", "")
            node = kg_map.get(nid, {})
            related_nodes.append({
                "node_id": nid,
                "category": node.get("category", ""),
                "difficulty": node.get("difficulty", ""),
                "exam_weight": node.get("exam_weight", ""),
                "score": r.get("score", 0),
                "prerequisites": node.get("prerequisites", []),
            })
            for pre in node.get("prerequisites", []):
                if pre not in prerequisites:
                    prerequisites.append(pre)

        # 2. 检索题型
        ep_results = _vector_search("exam_patterns", q_emb, top_k=3)
        ep_meta = _load_meta("exam_patterns")
        ep_map = {e["pattern_id"]: e for e in ep_meta}
        related_patterns = []
        for r in ep_results[:2]:
            pid = r.get("pattern_id", "")
            pat = ep_map.get(pid, {})
            related_patterns.append({
                "pattern_id": pid,
                "category": pat.get("category", ""),
                "subcategory": pat.get("subcategory", ""),
                "score": r.get("score", 0),
            })

        # 3. 检索思维招式
        tp_results = _vector_search("thinking_patterns", q_emb, top_k=5)
        tp_meta = _load_meta("thinking_patterns")
        related_thinking = []
        for r in tp_results[:3]:
            tpid = r.get("id", "")
            for tp in tp_meta:
                if tp.get("id") == tpid:
                    related_thinking.append({
                        "id": tpid,
                        "name": tp.get("name", ""),
                        "desc": tp.get("desc", ""),
                        "score": r.get("score", 0),
                    })
                    break

        # 4. 拉取 chunks
        chunks = self.retrieve(query, top_k=8)

        # 5. 从 chunks 构建推荐视频列表（BV 去重，含完整视频信息）
        seen_bv = set()
        recommended_videos = []
        for c in chunks:
            bv = c.get('bv', '')
            pn = c.get('p_number', 1)
            key = f"{bv}#P{pn}"
            if key in seen_bv or not bv:
                continue
            seen_bv.add(key)
            recommended_videos.append({
                'bv': bv,
                'p_number': pn,
                'video_title': c.get('video_title', ''),
                'collection': c.get('collection', '其他视频'),
                'short_title': c.get('short_title', ''),
                'text_preview': c.get('text_preview', '')[:100],
                'score': c.get('final_score', c.get('score', 0)),
            })
            if len(recommended_videos) >= 5:
                break

        elapsed_ms = (time.time() - t0) * 1000

        return {
            "query": query,
            "related_nodes": related_nodes,
            "prerequisites": prerequisites[:5],
            "related_patterns": related_patterns,
            "related_thinking": related_thinking,
            "chunks": chunks,
            "recommended_videos": recommended_videos,
            "elapsed_ms": elapsed_ms,
        }


# ── 便捷函数 ─────────────────────────────────────
def retrieve(query: str, top_k: int = 10) -> List[Dict]:
    r = YihuierRetriever()
    return r.retrieve(query, top_k=top_k)


def diagnose(query: str) -> Dict:
    r = YihuierRetriever()
    return r.retrieve_with_diagnosis(query)
