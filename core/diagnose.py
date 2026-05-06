#!/usr/bin/env python3
"""
一化儿 SKILL 诊断引擎
功能：
1. 识别 query 的学段、复杂度
2. 通过 retrieve 找到相关知识点
3. 反查 prerequisite 链路
4. 推断可能的 missing prerequisites
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict

# ── 路径 ─────────────────────────────────────────
SKILL_DIR = Path(__file__).parent.parent
DATA_DIR = SKILL_DIR / "data"

# ── 学段识别关键词 ───────────────────────────────
GRADE_SIGNALS = {
    '高一': ['摩尔', '物质分类', '氧化还原', '化学计量', '配平',
             '必修一', '必修二', '阿伏伽德罗', '物质的量',
             '离子反应', '离子方程式', '元素周期律', '化学键'],
    '高二': ['化学平衡', '水解', '电化学', '热化学', '盖斯',
             '电离平衡', 'Ksp', '选修四', '反应原理', '沉淀溶解平衡',
             '电解质', '弱电解质', '盐类水解', '勒夏特列'],
    '高三': ['工艺流程', '压轴', '高考真题', '综合大题',
             '反应原理综合', '实验综合', '有机推断', '同分异构体',
             '滴定', '产率', '定量分析'],
}

COMPLEXITY_SIGNALS = {
    'simple': ['怎么写', '什么是', '快速判断', '秒杀', '定义'],
    'diagnostic': ['老错', '不会', '卡住', '搞不清', '为什么', '好难'],
    'complex': ['压轴', '综合', '系统', '原理', '所有'],
}

# ── 知识图谱加载 ─────────────────────────────────
_kg_data = None


def _load_kg() -> Dict:
    global _kg_data
    if _kg_data is None:
        _kg_data = {}
        kg_file = DATA_DIR / "knowledge_graph_full.jsonl"
        if kg_file.exists():
            with open(kg_file) as f:
                for line in f:
                    if line.strip():
                        node = json.loads(line)
                        _kg_data[node["node_id"]] = node
    return _kg_data


def detect_grade(query: str) -> str:
    """识别学段，返回 '高一'/'高二'/'高三'/'unknown'"""
    scores = {grade: 0 for grade in GRADE_SIGNALS}
    for grade, keywords in GRADE_SIGNALS.items():
        for kw in keywords:
            if kw in query:
                scores[grade] += 1

    max_grade = max(scores, key=scores.get)
    if scores[max_grade] == 0:
        return 'unknown'
    return max_grade


def detect_complexity(query: str) -> str:
    """识别复杂度"""
    # diagnostic 优先（用户表达了困难）
    if any(kw in query for kw in COMPLEXITY_SIGNALS['diagnostic']):
        return 'diagnostic'
    if any(kw in query for kw in COMPLEXITY_SIGNALS['complex']):
        return 'complex'
    if any(kw in query for kw in COMPLEXITY_SIGNALS['simple']):
        return 'simple'
    if len(query) < 20:
        return 'simple'
    return 'normal'


def trace_prerequisites(node_ids: List[str], depth: int = 3) -> List[str]:
    """
    反查 prerequisite 链路
    depth: 查多深
    返回所有前置节点 id（保序去重）
    """
    kg_data = _load_kg()

    visited = set()
    queue = list(node_ids)
    result = []

    for _ in range(depth):
        next_queue = []
        for node_id in queue:
            if node_id in visited:
                continue
            visited.add(node_id)

            node = kg_data.get(node_id)
            if not node:
                continue

            for prereq in node.get('prerequisites', []):
                if prereq not in visited:
                    result.append(prereq)
                    next_queue.append(prereq)

        queue = next_queue

    return list(OrderedDict.fromkeys(result))


def diagnose_query(query: str, retriever) -> Dict:
    """主诊断函数"""
    # 1. 调用 retrieve_with_diagnosis
    retrieval = retriever.retrieve_with_diagnosis(query)

    # 2. 学段识别
    grade = detect_grade(query)

    # 3. 复杂度识别
    complexity = detect_complexity(query)

    # 4. 反查前置节点
    related_node_ids = [n['node_id'] for n in retrieval['related_nodes']]
    depth_map = {'高一': 1, '高二': 2, '高三': 3, 'unknown': 2}
    depth = depth_map.get(grade, 2)
    prereqs = trace_prerequisites(related_node_ids, depth=depth)

    # 5. 推断 missing prerequisites
    n_missing = 2 if grade == '高一' else 3
    missing_prereqs = prereqs[:n_missing]

    # 6. 提取题型 + 招式
    exam_patterns = [p['pattern_id'] for p in retrieval['related_patterns']][:2]
    thinking_patterns = [t['id'] for t in retrieval['related_thinking']][:3]
    thinking_names = [t['name'] for t in retrieval['related_thinking']][:3]

    # 7. 提取推荐视频（从 chunks 中收集真实 BV+P，含完整视频信息）
    recommended_videos = []
    seen_bvp = set()
    for c in retrieval.get('chunks', []):
        bv = c.get('bv', '')
        pn = c.get('p_number', '')
        if bv and (bv, pn) not in seen_bvp:
            seen_bvp.add((bv, pn))
            recommended_videos.append({
                'bv': bv,
                'p_number': pn,
                'chunk_id': c.get('chunk_id', ''),
                'video_title': c.get('video_title', ''),
                'collection': c.get('collection', '其他视频'),
                'short_title': c.get('short_title', ''),
                'text_preview': c.get('text_preview', '')[:100],
            })
        if len(recommended_videos) >= 5:
            break

    return {
        'grade_signal': grade,
        'complexity': complexity,
        'related_nodes': related_node_ids,
        'missing_prereqs': missing_prereqs,
        'exam_patterns': exam_patterns,
        'thinking_patterns': thinking_patterns,
        'thinking_names': thinking_names,
        'chunks': retrieval['chunks'],
        'recommended_videos': recommended_videos,
        'prereq_chain_full': prereqs,
        'elapsed_ms': retrieval.get('elapsed_ms', 0),
    }
