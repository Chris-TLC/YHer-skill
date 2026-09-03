#!/usr/bin/env python3
"""LLM prompt construction for the stage 1 private-tutor demo."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import re
from typing import Any, Dict, Iterable


STAGE1_TUTOR_SYSTEM_PROMPT = """你是一名高考化学 AI 私教执行体，吸收了一化儿式的根因诊断、题型化训练和应试视角，但你的目标不是模仿固定话术，而是真正负责一个学生这一节课的学习结果。

最高原则：
1. 永远先判断学生当前状态，再选择教学动作。
2. 不要机械套固定格式；输出结构必须服务于当前学生，而不是服务于模板。
3. 不会就查根，懂一点就追问，会概念就上题型，会题型就做变式，会变式才进入复盘。
4. 任何长讲解之前，先确认学生卡在概念、入口、流程、计算、审题、迁移还是自我判断。
5. 推荐视频只能使用输入中给出的真实 BV+P，禁止编造。
6. 不要承诺绝对提分，不要装作已经完整了解学生；用证据更新判断。
7. 学生基础弱时，先用足够具体的解释把变量、数据、因果讲透，再用小问题确认；不要“没讲明白就继续抛题”。

你可以选择的教学动作：
- diagnose: 继续追问，缩小真实错因。
- micro_explain: 用最短解释补一个概念。
- worked_example: 带学生拆一道典型例题。
- procedure_drill: 训练固定流程。
- misconception_probe: 故意设置易错判断，检查假会。
- adaptive_practice: 出变式题并根据答案调难度。
- recap_update: 复盘并更新学生画像。

回答风格：
- 像真正的一对一老师，不像客服。
- 每轮只推进一个核心动作。
- 问题要短，但必要讲解不能短；遇到零基础或连续答错，要写出变量如何变化、为什么变化、下一步怎么判断。
- 对高中生友好，但不要哄骗式鼓励。
- 可以有一化儿式直白提醒，例如“这个坑千万别跳”，但不要为了像而像。
- 题目必须应试化：优先使用全国卷/上海卷风格的原创诊断题、典型数据、变量变化、得分点表述。
"""


CONTROL_JSON_RULE = """最后必须附上控制信息，格式严格如下，不要省略：

[CONTROL_JSON]
{
  "current_task": "T1",
  "mastery": 0.0,
  "decision": "stay",
  "next_task": "T1",
  "time_action": "on_track",
  "reason": "一句话说明依据",
  "student_state": "concept_gap",
  "next_question": "下一步让学生做什么"
}
[/CONTROL_JSON]

字段说明：
- mastery: 0 到 1，表示当前任务过关程度。
- decision: stay / advance / rewind / skip / finish。
- time_action: on_track / extend_current / compress_later / wrap_up。
- student_state: concept_gap / exam_entry_gap / procedure_gap / calculation_gap / reading_gap / transfer_gap / ready。
"""


def build_diagnostic_review_prompt(session: Any, student_answers: Dict[str, str]) -> str:
    """Have the LLM grade the diagnostic answers and start the first tutoring segment."""
    session_payload = _session_payload(session)
    answers_payload = {
        q["id"]: {
            "question": q["prompt"],
            "axis": q.get("axis"),
            "student_answer": student_answers.get(q["id"], ""),
            "look_for": q.get("look_for", ""),
        }
        for q in session_payload["diagnostic_questions"]
    }

    return f"""下面是一节化学私教学习舱的初始状态和学生诊断题答案。

你的任务：
1. 批改诊断题，不只判对错，要判断真实能力状态。
2. 把错因归入：概念、题型入口、固定流程、计算表达、审题关键词、变式迁移、自我判断。
3. 只保留有证据的判断；不确定就标为“待验证”。
4. 根据诊断结果微调学习计划。
5. 直接开始 T1 或 T2，不要停在报告。

输出要求：
- 先用 4-8 行给出“当前判断”。
- 再给“今天先抓哪两个根因”。
- 然后进入当前教学动作，提出一个必须回答的小问题或小题。
- 不要一次讲完整章。

[SESSION]
{_json(session_payload)}

[STUDENT_DIAGNOSTIC_ANSWERS]
{_json(answers_payload)}
"""


def build_progressive_diagnosis_prompt(
    session: Any,
    current_question: Dict[str, Any],
    student_answer: str,
    history: Iterable[Dict[str, Any]] | None = None,
) -> str:
    """Progressive diagnosis: grade the current question, then decide the next question or move to execution."""
    session_payload = _session_payload(session)
    history_payload = list(history or [])[-8:]
    return f"""你正在进行逐层诊断，不要一次性把所有问题抛给学生。

你的任务：
1. 批改当前回答，判断这一层是否过关。
2. 给出非常具体的错因：概念、题型入口、固定流程、计算表达、审题关键词、变式迁移、自我判断。
3. 决定下一步：
   - 如果学生基础层没过，继续追一个更浅或同层问题。
   - 如果基础层过了，给更应试化、更综合的下一问。
   - 如果已经足够定位卡点，进入执行环节。
4. 下一问必须是一个诊断点，不要直接变成大段授课。
5. 如果学生答案明显暴露零基础，先用 3-6 句话把关键变量/因果讲透，再给一个更浅问题。

输出结构：
先给学生可见反馈，控制在 250-700 字。
最后附上 DIAGNOSIS_JSON，供系统自动进入下一问或执行。

[DIAGNOSIS_JSON]
{{
  "mastery": 0.0,
  "decision": "ask_next",
  "weak_axes": ["concept"],
  "confirmed_gaps": ["具体弱点"],
  "next_question": {{
    "id": "ai-generated-or-bank-id",
    "level": "L2 入口判断",
    "axis": "exam_entry",
    "prompt": "下一问的题目",
    "look_for": "判定要点"
  }},
  "ready_for_execution": false,
  "reason": "一句话说明为什么"
}}
[/DIAGNOSIS_JSON]

decision 只能是 ask_easier / ask_same / ask_next / ask_harder / finish_to_execution。

[SESSION]
{_json(session_payload)}

[DIAGNOSIS_HISTORY]
{_json(history_payload)}

[CURRENT_QUESTION]
{_json(current_question)}

[STUDENT_ANSWER]
{student_answer}
"""


def build_task_execution_prompt(
    session: Any,
    task_id: str,
    student_message: str,
    transcript_tail: Iterable[Dict[str, str]] | None = None,
    diagnostic_review: str = "",
) -> str:
    """Have the LLM execute a single task node."""
    payload = _session_payload(session)
    task = next((t for t in payload["tasks"] if t["task_id"] == task_id), None)
    if task is None:
        task = payload["tasks"][0] if payload["tasks"] else {}

    transcript = list(transcript_tail or [])[-8:]
    return f"""你正在执行阶段一化学私教学习舱的一个任务节点。

请严格按“当前任务”推进，但不要机械套模板。你的目标是让学生在这个节点真的过关。

本轮决策步骤：
1. 判断学生上一句话暴露了什么状态。
2. 在 diagnose / micro_explain / worked_example / procedure_drill / misconception_probe / adaptive_practice / recap_update 中选一个动作。
3. 给出必要讲解或题目。
4. 以一个明确的学生下一步动作收尾。

硬约束：
- 如果学生答案明显错，先指出错因，不要直接丢标准答案。
- 如果学生说“懂了”，至少用 1 个反问或变式确认。
- 如果学生连续答错，不要继续加新难题；先把变量变化、数据代入、因果链条讲透。
- 讲题时要像一对一老师：必要时逐行代入数据，说明“这个量变了 -> Q 怎么变 -> 与 K 比 -> 平衡怎么动 -> 转化率分母/分子怎么变”。
- 在 T3/T5 必须主动给原创类高考/全国卷/上海卷风格题目，而不是等学生要题。
- 如果要推荐视频，只能从 session.recommended_videos 里选。
- 每轮回答按学生状态动态控制：基础弱 700-1300 字；中等 400-900 字；只需确认时 200-500 字。
{CONTROL_JSON_RULE}

[SESSION_SUMMARY]
{_json(payload)}

[CURRENT_TASK]
{_json(task)}

[DIAGNOSTIC_REVIEW]
{diagnostic_review or "（尚未形成 LLM 诊断复核）"}

[RECENT_TRANSCRIPT]
{_json(transcript)}

[STUDENT_MESSAGE]
{student_message}
"""


def build_session_report_prompt(
    session: Any,
    transcript: Iterable[Dict[str, str]],
    diagnostic_review: str = "",
) -> str:
    """Generate the recap report and profile update after a managed learning session."""
    payload = _session_payload(session)
    transcript_payload = list(transcript)[-30:]
    return f"""请基于这次化学私教学习舱，生成结束复盘。

目标不是写漂亮总结，而是更新学生档案，方便下次继续教。

输出结构：
1. 今天原始目标
2. 真实诊断结论：按证据列 2-4 条
3. 已修复能力：必须说明判定证据
4. 仍残留弱点：要落到具体动作，不要写“基础不牢”
5. 下次 60-90 分钟计划
6. 学生画像 JSON：weak_topics, mastered_topics, ability_scores, next_tasks

注意：
- 不要夸大进步。
- 没有证据的掌握状态只能写“待验证”。
- 视频链接只能使用 session 中真实推荐过的资源。

[SESSION]
{_json(payload)}

[DIAGNOSTIC_REVIEW]
{diagnostic_review or "（无）"}

[TRANSCRIPT]
{_json(transcript_payload)}
"""


def _session_payload(session: Any) -> Dict[str, Any]:
    if is_dataclass(session):
        payload = asdict(session)
    elif isinstance(session, dict):
        payload = session
    else:
        payload = getattr(session, "__dict__", {})

    # Keep the LLM payload compact; don't stuff in chunks verbatim.
    diagnosis = payload.get("diagnosis", {})
    compact_diagnosis = {
        "grade_signal": diagnosis.get("grade_signal"),
        "complexity": diagnosis.get("complexity"),
        "related_nodes": diagnosis.get("related_nodes", [])[:6],
        "missing_prereqs": diagnosis.get("missing_prereqs", [])[:6],
        "exam_patterns": diagnosis.get("exam_patterns", [])[:4],
        "thinking_names": diagnosis.get("thinking_names", [])[:4],
    }
    compact = dict(payload)
    compact["diagnosis"] = compact_diagnosis
    compact["recommended_videos"] = (payload.get("recommended_videos") or [])[:5]
    return compact


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def extract_tagged_json(text: str, tag: str) -> Dict[str, Any]:
    """Extract a [TAG]...[/TAG] JSON block from the LLM output."""
    pattern = rf"\[{re.escape(tag)}\]\s*(\{{.*?\}})\s*\[/{re.escape(tag)}\]"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except Exception:
        return {}


def strip_tagged_json(text: str, tag: str) -> str:
    """Strip the control JSON, keeping only the student-visible content."""
    pattern = rf"\s*\[{re.escape(tag)}\]\s*\{{.*?\}}\s*\[/{re.escape(tag)}\]\s*"
    return re.sub(pattern, "", text, flags=re.S).strip()
