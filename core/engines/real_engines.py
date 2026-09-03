#!/usr/bin/env python3
"""
Real engine layer - connects the knowledge graph, item bank, and LLM.
Used to replace the fake data in the demo.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# 1. Diagnosis engine: extract diagnostic questions from the knowledge graph
# ============================================================================

def generate_diagnostic_questions(
    node_id: str,
    kg_data: Dict[str, Any],
    count: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate diagnostic questions from the knowledge graph

    Args:
        node_id: knowledge point id (e.g. "盐类水解")
        kg_data: knowledge graph data (already loaded dict)
        count: number of questions to generate

    Returns:
        [
            {
                "question": "碳酸氢根与偏铝酸根反应是双水解吗？",
                "expected_answer": "不是，是酸碱中和",
                "root_cause": "混淆水解与电离",
                "symptom": "看到HCO3⁻与AlO2⁻反应写产物为CO2气体...",
                "thinking_patterns": ["root_cause_diagnosis", "comparison_memory"]
            }
        ]
    """
    node = kg_data.get(node_id, {})
    common_failures = node.get('common_failures', [])

    if not common_failures:
        return []

    # Take at most `count` questions
    selected = common_failures[:min(count, len(common_failures))]

    questions = []
    for fail in selected:
        questions.append({
            "question": fail.get('diagnostic_question', ''),
            "root_cause": fail.get('cause', ''),
            "symptom": fail.get('symptom', ''),
            "thinking_patterns": node.get('thinking_patterns_used', [])
        })

    return questions


# ============================================================================
# 2. Recommendation engine: intelligently match videos
# ============================================================================

def recommend_videos(
    node_id: str,
    kg_data: Dict[str, Any],
    scenario: str,
    grade: str,
    region: str = "上海",
    time_budget_hours: float = 2.0,
    mastery_level: str = "learning"
) -> List[Dict[str, Any]]:
    """
    Smart video recommendation (by scenario, grade, region, time budget, mastery)

    Args:
        node_id: knowledge point id
        scenario: learning scenario (preview / consolidate / pre-exam / breakthrough / redo)
        grade: grade (高一上/高一下/高二上/高二下/高三)
        region: region (default 上海)
        time_budget_hours: available time (hours)
        mastery_level: mastery (not_started/learning/partial/mastered)

    Returns:
        Recommended video list, sorted by priority
    """
    node = kg_data.get(node_id, {})
    all_videos = node.get('recommended_videos', [])

    if not all_videos:
        return []

    # Scenario → video type mapping
    scenario_to_type = {
        "预习新课": ["concept_intro"],
        "巩固复习": ["review_with_problems"],
        "考前冲刺": ["exam_problem_drill"],
        "知识突破": ["concept_intro", "review_with_problems"],
        "错题重做": ["exam_problem_drill"]
    }

    preferred_types = scenario_to_type.get(scenario, ["concept_intro"])

    # Mastery → difficulty mapping
    mastery_to_difficulty = {
        "not_started": ["T1", "T2"],
        "learning": ["T2"],
        "partial": ["T2", "T3"],
        "mastered": ["T3", "T4"]
    }

    preferred_difficulties = mastery_to_difficulty.get(mastery_level, ["T2"])

    # Filter + score
    scored_videos = []
    for video in all_videos:
        score = 0.0

        # Type match (40 points)
        if video.get('type') in preferred_types:
            score += 40

        # Difficulty match (30 points)
        if video.get('difficulty') in preferred_difficulties:
            score += 30

        # Region priority (20 points)
        if region in video.get('region_priority', []):
            score += 20
        elif "全国" in video.get('region_priority', []):
            score += 10

        # Suitable duration (10 points)
        duration = video.get('duration_min', 60)
        time_budget_min = time_budget_hours * 60
        if duration <= time_budget_min:
            score += 10

        scored_videos.append((score, video))

    # Sort by score
    scored_videos.sort(key=lambda x: x[0], reverse=True)

    # Return the top 3 (or all, if fewer than 3)
    return [v for _, v in scored_videos[:3]]


# ============================================================================
# 3. Verification engine: generate absorption-check questions
# ============================================================================

def generate_verification_question(
    node_id: str,
    kg_data: Dict[str, Any],
    video_watched: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate an absorption-check question (based on the video the student just watched)

    Args:
        node_id: knowledge point id
        kg_data: knowledge graph
        video_watched: info about the video the student just watched

    Returns:
        {
            "question": "verification question",
            "judgment_criteria": ["criterion 1", "criterion 2"],
            "rubric": "grading standard"
        }
    """
    node = kg_data.get(node_id, {})

    # Extract verification criteria from judgment_criteria_for_mastery
    criteria = node.get('judgment_criteria_for_mastery', [])

    if not criteria:
        return {
            "question": f"请简述{node_id}的核心要点。",
            "judgment_criteria": [],
            "rubric": "能说出3个以上关键点即为掌握"
        }

    # Randomly pick one criterion as the verification point
    criterion = random.choice(criteria)

    # Turn it into a verification question
    question = f"根据刚才的视频，{criterion.split('能')[1] if '能' in criterion else criterion}"

    return {
        "question": question,
        "judgment_criteria": criteria,
        "rubric": criterion
    }


# ============================================================================
# 4. Practice engine: smart item selection
# ============================================================================

def select_practice_questions(
    node_ids: List[str],
    question_bank_path: Path,
    count: int = 10,
    choice_ratio: float = 0.3,
    difficulty_range: List[int] = [1, 2, 3],
    region: str = "上海"
) -> List[Dict[str, Any]]:
    """
    Smart item selection from the item bank (adapted to the chemistry_solved.jsonl format)

    Args:
        node_ids: list of knowledge point ids (what the student is learning this session)
        question_bank_path: item bank file path (chemistry_solved.jsonl)
        count: total number of items
        choice_ratio: share of multiple-choice items (default 0.3 = 30%)
        difficulty_range: difficulty range (1=easy, 2=medium, 3=hard, 4=very hard)
        region: region (Shanghai papers preferred)

    Returns:
        List of items (already sorted by type: multiple-choice first, fill-in-the-blank after)
    """
    if not question_bank_path.exists():
        return []

    # Load the item bank
    all_questions = []
    with open(question_bank_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                q = json.loads(line.strip())
                # Only keep items that have the `solved` field
                if 'solved' in q:
                    all_questions.append(q)
            except:
                continue

    if not all_questions:
        return []

    # Filtering condition
    def match_question(q: Dict) -> bool:
        solved = q.get('solved', {})

        # Difficulty match
        q_diff = solved.get('difficulty', 2)
        if q_diff not in difficulty_range:
            return False

        return True  # For now filter on difficulty only; knowledge-point match is a bonus

    matched_questions = [q for q in all_questions if match_question(q)]

    # If knowledge points are given, prefer items matching them
    if node_ids and matched_questions:
        priority_questions = []
        for q in matched_questions:
            q_knowledge = q.get('solved', {}).get('knowledge_points', [])
            for node in node_ids:
                for kp in q_knowledge:
                    if node in kp or kp in node:
                        priority_questions.append(q)
                        break

        # Use the priority items if any; otherwise use all matched items
        if priority_questions:
            matched_questions = priority_questions

    if not matched_questions:
        # Nothing matched: relax the filter to region and difficulty only
        matched_questions = [q for q in all_questions
                            if q.get('solved', {}).get('difficulty', 2) in difficulty_range]

    if not matched_questions:
        # Relax further: pick randomly
        matched_questions = all_questions[:100]  # at most the first 100 items

    if not matched_questions:
        return []

    # Split multiple-choice vs non-multiple-choice
    choice_questions = [q for q in matched_questions
                       if '单选' in q.get('solved', {}).get('question_type', '') or
                          '多选' in q.get('solved', {}).get('question_type', '')]
    fillblank_questions = [q for q in matched_questions
                          if q not in choice_questions]

    # Work out the counts
    choice_count = int(count * choice_ratio)
    fillblank_count = count - choice_count

    # Random sample
    selected_choice = random.sample(choice_questions, min(choice_count, len(choice_questions)))
    selected_fillblank = random.sample(fillblank_questions, min(fillblank_count, len(fillblank_questions)))

    # Merge (multiple-choice first)
    selected = selected_choice + selected_fillblank

    # If still short of `count`, top up
    if len(selected) < count:
        remaining = [q for q in matched_questions if q not in selected]
        need = count - len(selected)
        if remaining:
            selected += random.sample(remaining, min(need, len(remaining)))

    return selected[:count]


# ============================================================================
# 5. Grading engine: auto-grade objective items + LLM-grade subjective items
# ============================================================================

def grade_answer(
    question: Dict[str, Any],
    user_answer: str,
    llm_caller: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Grade an answer (objective items by direct comparison, subjective items via LLM).
    Adapted to the chemistry_solved.jsonl format.

    Args:
        question: the item object
        user_answer: the student's answer
        llm_caller: LLM caller (used for grading subjective items)

    Returns:
        {
            "is_correct": True/False,
            "score": score,
            "explanation": "grading note",
            "correct_answer": "the correct answer"
        }
    """
    solved = question.get('solved', {})
    q_type = solved.get('question_type', '填空题')
    correct_answer = solved.get('standard_answer', '')

    # Objective items: direct comparison
    if '单选' in q_type or '多选' in q_type:
        is_correct = user_answer.strip().upper() == correct_answer.strip().upper()
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "explanation": "答案正确" if is_correct else f"正确答案是：{correct_answer}",
            "correct_answer": correct_answer
        }

    # Subjective items: simple comparison (a rubric engine could plug in later)
    user_clean = user_answer.replace(' ', '').replace('；', ';').replace('，', ',')
    correct_clean = correct_answer.replace(' ', '').replace('；', ';').replace('，', ',')

    is_correct = user_clean == correct_clean

    return {
        "is_correct": is_correct,
        "score": 1.0 if is_correct else 0.0,
        "explanation": "答案正确" if is_correct else f"正确答案是：{correct_answer}",
        "correct_answer": correct_answer
    }


# ============================================================================
# 6. Review engine: generate a review plan
# ============================================================================

def generate_review_plan(
    user_id: str,
    weak_nodes: List[str],
    kg_data: Dict[str, Any],
    scenario: str,
    grade: str
) -> Dict[str, Any]:
    """
    Generate a smart review plan

    Args:
        user_id: student id
        weak_nodes: list of weak knowledge points
        kg_data: knowledge graph
        scenario: this session's learning scenario
        grade: grade

    Returns:
        {
            "summary": "this session's summary",
            "weak_points": ["weak point 1", "weak point 2"],
            "recommended_videos": [video list],
            "next_review_date": "suggested next review date"
        }
    """
    # Build the summary
    summary = f"本次{scenario}环节已完成。"

    if not weak_nodes:
        summary += "所有知识点掌握良好，继续保持！"
        return {
            "summary": summary,
            "weak_points": [],
            "recommended_videos": [],
            "next_review_date": "3天后"
        }

    summary += f"发现{len(weak_nodes)}个薄弱点需要加强。"

    # Recommend videos for each weak point
    all_videos = []
    for node_id in weak_nodes[:3]:  # at most 3
        videos = recommend_videos(
            node_id=node_id,
            kg_data=kg_data,
            scenario="巩固复习",
            grade=grade,
            region="上海",
            time_budget_hours=1.0,
            mastery_level="learning"
        )
        for v in videos[:1]:  # at most 1 video per node
            v['node_id'] = node_id
            all_videos.append(v)

    return {
        "summary": summary,
        "weak_points": weak_nodes,
        "recommended_videos": all_videos,
        "next_review_date": "1天后"  # weak points need reviewing soon
    }


# ============================================================================
# 7. Helper: load the knowledge graph
# ============================================================================

def load_knowledge_graph(kg_path: Path) -> Dict[str, Any]:
    """
    Load a knowledge graph JSONL file into a dict

    Returns:
        {
            "node_id1": {node_data},
            "node_id2": {node_data},
            ...
        }
    """
    kg_dict = {}

    if not kg_path.exists():
        return kg_dict

    with open(kg_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                node = json.loads(line.strip())
                node_id = node.get('node_id')
                if node_id:
                    kg_dict[node_id] = node
            except:
                continue

    return kg_dict


# ============================================================================
# 8. LLM call wrappers (for diagnosis and supplementary teaching)
# ============================================================================

# System prompt templates
DIAGNOSIS_SYSTEM_PROMPT = """
你是一化儿化学教学团队的AI诊断助手，擅长通过学生的答案快速定位思维漏洞。

## 核心方法论：root_cause_diagnosis
你必须直击学生"为什么错"的根本原因，而不是只说"知识点没掌握"。

每次分析必须包含：
1. 根本原因（cause）：学生的思维漏洞在哪里？
2. 典型表现（symptom）：这个漏洞会导致什么具体错误？
3. 是否匹配已知的common_failures

## 判断标准
给你一个诊断问题及其对应的"常见错误根因"，你要判断学生的答案是否暴露了这个根因。

不要过度宽容：
- ❌ 学生答案模糊不清 → 判定为"有问题"
- ❌ 学生答案偏离主题 → 判定为"有问题"
- ✅ 学生答案准确、逻辑清晰 → 判定为"无问题"

## 输出格式
严格JSON格式：
{
    "has_issue": true/false,
    "analysis": "你的分析（50字以内，直接说根因）"
}
"""

DEEP_TEACHING_SYSTEM_PROMPT = """
你是一化儿化学教学团队的AI讲师，学生刚看完视频但验证答错了，需要你补充讲解。

## 你必须使用的4大教学思维

### 1. root_cause_diagnosis（根因诊断）
直击学生为什么错，不说表面话。
格式：
**你错的根本原因**：[具体思维漏洞]

### 2. fixed_procedure（固定流程）
给出"傻瓜式"步骤清单。
格式：
**解题固定流程**：
1. 第一步：...
2. 第二步：...
3. 第三步：...

### 3. setter_perspective（出题人视角）
拆解命题意图和陷阱。
格式：
**出题人挖的坑**：[这道题在哪里设陷阱]
**如何避开**：[识别信号+应对策略]

### 4. comparison_memory（对比记忆）
如果有易混淆点，并排对比。

## 上海考纲特征（必读）
你正在辅导上海地区学生，注意：
1. 实验题偏"探究设计"，不是背套路
2. 化学用语要精炼（"水浴加热"不是"加热"）
3. 胶体制备是高频点

## 输出要求
- 用Markdown格式
- 简洁直接，不超过300字
- 必须包含上述4大思维中的至少3个
- 不要啰嗦，不要说"同学你好"之类的客套话
"""

def call_llm_for_diagnosis(
    user_answer: str,
    diagnostic_question: Dict[str, Any],
    llm_caller: Any,
    model: str = "deepseek-v4-flash"
) -> Dict[str, Any]:
    """
    Use the LLM to analyze the student's diagnostic answer

    Returns:
        {
            "has_issue": True/False,
            "matched_root_cause": "root cause",
            "analysis": "analysis note"
        }
    """
    prompt = f"""诊断问题：{diagnostic_question['question']}
学生答案：{user_answer}

已知该问题对应的常见错误：
- 根因：{diagnostic_question['root_cause']}
- 典型表现：{diagnostic_question['symptom']}

请判断学生答案是否暴露了这个思维漏洞。
"""

    try:
        result = llm_caller(
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            user_message=prompt,
            model_override=model
        )

        # Parse the JSON
        content = result.get('content', '{}')
        parsed = json.loads(content)

        return {
            "has_issue": parsed.get('has_issue', False),
            "matched_root_cause": diagnostic_question['root_cause'] if parsed.get('has_issue') else None,
            "analysis": parsed.get('analysis', '')
        }
    except:
        # If the LLM call fails, return defaults
        return {
            "has_issue": False,
            "matched_root_cause": None,
            "analysis": "诊断分析失败，请重试"
        }


def call_llm_for_deep_teaching(
    node_id: str,
    kg_data: Dict[str, Any],
    user_wrong_answer: str,
    verification_question: str,
    llm_caller: Any,
    model: str = "deepseek-v4-pro"
) -> str:
    """
    Use the Pro model for deep supplementary teaching (when the student fails the verification)

    Returns:
        Supplementary teaching content (Markdown format)
    """
    node = kg_data.get(node_id, {})

    criteria_text = '\n'.join(['- ' + c for c in node.get('judgment_criteria_for_mastery', [])])
    failures_text = '\n'.join(['- ' + f['cause'] for f in node.get('common_failures', [])[:3]])

    prompt = f"""知识点：{node_id}

验证问题：{verification_question}
学生错误答案：{user_wrong_answer}

掌握标准：
{criteria_text}

该知识点的常见错误：
{failures_text}

请用一化儿的4大教学思维补充讲解。
"""

    try:
        result = llm_caller(
            system_prompt=DEEP_TEACHING_SYSTEM_PROMPT,
            user_message=prompt,
            model_override=model
        )

        return result.get('content', '补充讲解生成失败，请稍后重试。')
    except:
        return "补充讲解生成失败，请稍后重试。"
