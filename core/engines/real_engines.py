#!/usr/bin/env python3
"""
真实引擎层 - 连接知识图谱、题库、LLM
用于替换Demo中的假数据
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# 1. 诊断引擎：从知识图谱提取诊断问题
# ============================================================================

def generate_diagnostic_questions(
    node_id: str,
    kg_data: Dict[str, Any],
    count: int = 3
) -> List[Dict[str, Any]]:
    """
    从知识图谱生成诊断问题

    Args:
        node_id: 知识点ID（如"盐类水解"）
        kg_data: 知识图谱数据（已加载的dict）
        count: 生成问题数量

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

    # 最多取count个问题
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
# 2. 推荐引擎：智能匹配视频
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
    智能推荐视频（基于场景、年级、地区、时长、掌握度）

    Args:
        node_id: 知识点ID
        scenario: 学习场景（预习/巩固/考前/突破/重做）
        grade: 年级（高一上/高一下/高二上/高二下/高三）
        region: 地区（默认上海）
        time_budget_hours: 可用时长（小时）
        mastery_level: 掌握度（not_started/learning/partial/mastered）

    Returns:
        推荐视频列表，按优先级排序
    """
    node = kg_data.get(node_id, {})
    all_videos = node.get('recommended_videos', [])

    if not all_videos:
        return []

    # 场景 → 视频类型映射
    scenario_to_type = {
        "预习新课": ["concept_intro"],
        "巩固复习": ["review_with_problems"],
        "考前冲刺": ["exam_problem_drill"],
        "知识突破": ["concept_intro", "review_with_problems"],
        "错题重做": ["exam_problem_drill"]
    }

    preferred_types = scenario_to_type.get(scenario, ["concept_intro"])

    # 掌握度 → 难度映射
    mastery_to_difficulty = {
        "not_started": ["T1", "T2"],
        "learning": ["T2"],
        "partial": ["T2", "T3"],
        "mastered": ["T3", "T4"]
    }

    preferred_difficulties = mastery_to_difficulty.get(mastery_level, ["T2"])

    # 筛选+评分
    scored_videos = []
    for video in all_videos:
        score = 0.0

        # 类型匹配（40分）
        if video.get('type') in preferred_types:
            score += 40

        # 难度匹配（30分）
        if video.get('difficulty') in preferred_difficulties:
            score += 30

        # 地区优先级（20分）
        if region in video.get('region_priority', []):
            score += 20
        elif "全国" in video.get('region_priority', []):
            score += 10

        # 时长合适（10分）
        duration = video.get('duration_min', 60)
        time_budget_min = time_budget_hours * 60
        if duration <= time_budget_min:
            score += 10

        scored_videos.append((score, video))

    # 按分数排序
    scored_videos.sort(key=lambda x: x[0], reverse=True)

    # 返回前3个（或所有如果不足3个）
    return [v for _, v in scored_videos[:3]]


# ============================================================================
# 3. 验证引擎：生成吸收验证问题
# ============================================================================

def generate_verification_question(
    node_id: str,
    kg_data: Dict[str, Any],
    video_watched: Dict[str, Any]
) -> Dict[str, Any]:
    """
    生成吸收验证问题（基于学生刚看完的视频）

    Args:
        node_id: 知识点ID
        kg_data: 知识图谱
        video_watched: 学生刚看完的视频信息

    Returns:
        {
            "question": "验证问题",
            "judgment_criteria": ["标准1", "标准2"],
            "rubric": "评分标准"
        }
    """
    node = kg_data.get(node_id, {})

    # 从judgment_criteria_for_mastery提取验证标准
    criteria = node.get('judgment_criteria_for_mastery', [])

    if not criteria:
        return {
            "question": f"请简述{node_id}的核心要点。",
            "judgment_criteria": [],
            "rubric": "能说出3个以上关键点即为掌握"
        }

    # 随机选一个标准作为验证点
    criterion = random.choice(criteria)

    # 转化为验证问题
    question = f"根据刚才的视频，{criterion.split('能')[1] if '能' in criterion else criterion}"

    return {
        "question": question,
        "judgment_criteria": criteria,
        "rubric": criterion
    }


# ============================================================================
# 4. 练习引擎：智能选题
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
    从题库智能选题（适配chemistry_solved.jsonl格式）

    Args:
        node_ids: 知识点ID列表（学生本次学习的知识点）
        question_bank_path: 题库文件路径（chemistry_solved.jsonl）
        count: 总题数
        choice_ratio: 选择题占比（默认0.3 = 30%）
        difficulty_range: 难度范围（1=简单, 2=中等, 3=困难, 4=极难）
        region: 地区（优先选上海卷）

    Returns:
        题目列表（已按类型排序：选择题在前，填空题在后）
    """
    if not question_bank_path.exists():
        return []

    # 加载题库
    all_questions = []
    with open(question_bank_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                q = json.loads(line.strip())
                # 只取有solved字段的题目
                if 'solved' in q:
                    all_questions.append(q)
            except:
                continue

    if not all_questions:
        return []

    # 筛选条件
    def match_question(q: Dict) -> bool:
        solved = q.get('solved', {})

        # 难度匹配
        q_diff = solved.get('difficulty', 2)
        if q_diff not in difficulty_range:
            return False

        return True  # 先只看难度，知识点匹配留作加分项

    matched_questions = [q for q in all_questions if match_question(q)]

    # 如果有知识点要求，优先选知识点匹配的题
    if node_ids and matched_questions:
        priority_questions = []
        for q in matched_questions:
            q_knowledge = q.get('solved', {}).get('knowledge_points', [])
            for node in node_ids:
                for kp in q_knowledge:
                    if node in kp or kp in node:
                        priority_questions.append(q)
                        break

        # 如果有优先题目，用优先题目；否则用全部匹配题目
        if priority_questions:
            matched_questions = priority_questions

    if not matched_questions:
        # 如果没有匹配的，放宽条件：只看地区和难度
        matched_questions = [q for q in all_questions
                            if q.get('solved', {}).get('difficulty', 2) in difficulty_range]

    if not matched_questions:
        # 再放宽：随机选
        matched_questions = all_questions[:100]  # 最多取前100题

    if not matched_questions:
        return []

    # 分选择题和非选择题
    choice_questions = [q for q in matched_questions
                       if '单选' in q.get('solved', {}).get('question_type', '') or
                          '多选' in q.get('solved', {}).get('question_type', '')]
    fillblank_questions = [q for q in matched_questions
                          if q not in choice_questions]

    # 计算数量
    choice_count = int(count * choice_ratio)
    fillblank_count = count - choice_count

    # 随机抽取
    selected_choice = random.sample(choice_questions, min(choice_count, len(choice_questions)))
    selected_fillblank = random.sample(fillblank_questions, min(fillblank_count, len(fillblank_questions)))

    # 合并（选择题在前）
    selected = selected_choice + selected_fillblank

    # 如果不足count个，补充
    if len(selected) < count:
        remaining = [q for q in matched_questions if q not in selected]
        need = count - len(selected)
        if remaining:
            selected += random.sample(remaining, min(need, len(remaining)))

    return selected[:count]


# ============================================================================
# 5. 打分引擎：客观题自动判分 + 主观题LLM判分
# ============================================================================

def grade_answer(
    question: Dict[str, Any],
    user_answer: str,
    llm_caller: Optional[Any] = None
) -> Dict[str, Any]:
    """
    判分（客观题直接比对，主观题用LLM）
    适配chemistry_solved.jsonl格式

    Args:
        question: 题目对象
        user_answer: 学生答案
        llm_caller: LLM调用器（用于主观题判分）

    Returns:
        {
            "is_correct": True/False,
            "score": 分数,
            "explanation": "判分说明",
            "correct_answer": "正确答案"
        }
    """
    solved = question.get('solved', {})
    q_type = solved.get('question_type', '填空题')
    correct_answer = solved.get('standard_answer', '')

    # 客观题：直接比对
    if '单选' in q_type or '多选' in q_type:
        is_correct = user_answer.strip().upper() == correct_answer.strip().upper()
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "explanation": "答案正确" if is_correct else f"正确答案是：{correct_answer}",
            "correct_answer": correct_answer
        }

    # 主观题：简单比对（未来可接rubric引擎）
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
# 6. 复习引擎：生成复习计划
# ============================================================================

def generate_review_plan(
    user_id: str,
    weak_nodes: List[str],
    kg_data: Dict[str, Any],
    scenario: str,
    grade: str
) -> Dict[str, Any]:
    """
    生成智能复习计划

    Args:
        user_id: 学生ID
        weak_nodes: 薄弱知识点列表
        kg_data: 知识图谱
        scenario: 本次学习场景
        grade: 年级

    Returns:
        {
            "summary": "本次学习总结",
            "weak_points": ["薄弱点1", "薄弱点2"],
            "recommended_videos": [视频列表],
            "next_review_date": "建议复习日期"
        }
    """
    # 生成总结
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

    # 为每个薄弱点推荐视频
    all_videos = []
    for node_id in weak_nodes[:3]:  # 最多3个
        videos = recommend_videos(
            node_id=node_id,
            kg_data=kg_data,
            scenario="巩固复习",
            grade=grade,
            region="上海",
            time_budget_hours=1.0,
            mastery_level="learning"
        )
        for v in videos[:1]:  # 每个节点最多1个视频
            v['node_id'] = node_id
            all_videos.append(v)

    return {
        "summary": summary,
        "weak_points": weak_nodes,
        "recommended_videos": all_videos,
        "next_review_date": "1天后"  # 薄弱点需要尽快复习
    }


# ============================================================================
# 7. 辅助函数：加载知识图谱
# ============================================================================

def load_knowledge_graph(kg_path: Path) -> Dict[str, Any]:
    """
    加载知识图谱JSONL文件为dict

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
# 8. LLM调用封装（用于诊断和补充讲解）
# ============================================================================

# System Prompt模板
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
    用LLM分析学生的诊断答案

    Returns:
        {
            "has_issue": True/False,
            "matched_root_cause": "根因",
            "analysis": "分析说明"
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

        # 解析JSON
        content = result.get('content', '{}')
        parsed = json.loads(content)

        return {
            "has_issue": parsed.get('has_issue', False),
            "matched_root_cause": diagnostic_question['root_cause'] if parsed.get('has_issue') else None,
            "analysis": parsed.get('analysis', '')
        }
    except:
        # 如果LLM调用失败，返回默认值
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
    用Pro模型进行深度补充讲解（当学生验证答错时）

    Returns:
        补充讲解内容（Markdown格式）
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
