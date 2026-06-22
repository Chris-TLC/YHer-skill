#!/usr/bin/env python3
"""
会话编排层（总蓝图 A：engine 心脏）。

把诊断引擎/教学引擎/任务状态机/学生模型/题库/检索/LLM 串成一个完整私教会话。
纯逻辑，零界面依赖——FastAPI 后端和 Streamlit 壳调的是同一个 Orchestrator。

这是阶段三 9 个学科 AI 各自的实例化对象（subject-agnostic）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
from typing import Any, Callable, Dict, List, Optional

from core.data.item_repository import get_item_repository
from core.data.knowledge_repository import get_knowledge_repository
from core.tutor.diagnose_engine import DiagnoseEngine, RubricPoint
from core.tutor.prompts import (
    SYSTEM_PROMPT,
    build_diagnosis_prompt,
    build_report_prompt,
    build_teach_prompt,
    extract_tagged_json,
    strip_tagged_json,
)
from core.tutor.task_machine import TaskMachine, TaskSpec, TimeContext
from core.tutor.teach_engine import TeachEngine


# 默认任务队列（T1-T6），每个带闸门
DEFAULT_TASKS = [
    TaskSpec("T1", "诊断校准", 20, mastery_gate=0.60),
    TaskSpec("T2", "根因补洞", 40, mastery_gate=0.68, depends_on=["T1"]),
    TaskSpec("T3", "题型入口训练", 40, mastery_gate=0.72, depends_on=["T2"]),
    TaskSpec("T4", "视频定点巩固", 20, mastery_gate=0.65, depends_on=["T2"]),
    TaskSpec("T5", "变式迁移训练", 45, mastery_gate=0.78, depends_on=["T3", "T4"]),
    TaskSpec("T6", "复盘与下次计划", 15, mastery_gate=0.70, depends_on=["T5"]),
]


@dataclass
class TutorSession:
    session_id: str
    user_id: str
    subject: str
    node_id: str
    goal: str
    grade: str
    region: str
    time_budget_min: int
    created_at: str
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    current_task_id: str = "T1"
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    diag_history: List[Dict[str, Any]] = field(default_factory=list)
    used_angles: List[str] = field(default_factory=list)
    recent_masteries: List[float] = field(default_factory=list)
    last_tutor_action: str = ""
    cost_yuan: float = 0.0


# LLM 调用签名：(system, user) -> {"content":..., "cost_yuan":...}
LLMCaller = Callable[[str, str], Dict[str, Any]]


class SessionOrchestrator:
    def __init__(self, llm_caller: Optional[LLMCaller] = None, retriever=None):
        self.kg = get_knowledge_repository()
        self.items = get_item_repository()
        self.diagnose = DiagnoseEngine(repo=self.kg)
        self.teach = TeachEngine(repo=self.kg, retriever=retriever)
        self.task_machine = TaskMachine()
        self.llm = llm_caller  # 注入：None 时走本地 fallback

    # === 创建会话 ===
    def create_session(
        self, user_id: str, goal: str, node_id: str = "", grade: str = "高二",
        region: str = "全国卷", time_budget_min: int = 180, subject: str = "chemistry",
    ) -> TutorSession:
        if not node_id:
            node = self.kg.find_node(goal)
            node_id = node.node_id if node else "化学平衡"
        tasks = self._build_tasks(time_budget_min)
        sid = self._make_id(user_id, goal)
        return TutorSession(
            session_id=sid, user_id=user_id, subject=subject, node_id=node_id,
            goal=goal or "补化学薄弱点", grade=grade, region=region,
            time_budget_min=time_budget_min,
            created_at=datetime.now().isoformat(timespec="seconds"),
            tasks=[asdict(t) for t in tasks], current_task_id="T1",
        )

    def _build_tasks(self, total: int) -> List[TaskSpec]:
        ratios = [0.10, 0.20, 0.20, 0.12, 0.25, 0.13]
        total = max(45, min(total, 360))
        tasks = []
        for spec, r in zip(DEFAULT_TASKS, ratios):
            t = TaskSpec(spec.task_id, spec.title, max(5, round(total * r / 5) * 5),
                         spec.mastery_gate, spec.max_extend_min, list(spec.depends_on),
                         spec.priority)
            tasks.append(t)
        return tasks

    # === 诊断题（L1-L4）===
    def first_question(self, session: TutorSession) -> Dict[str, Any]:
        qs = self.diagnose.build_progressive_questions(session.node_id)
        return asdict(qs[0]) if qs else {}

    def diagnostic_questions(self, session: TutorSession) -> List[Dict[str, Any]]:
        return [asdict(q) for q in self.diagnose.build_progressive_questions(session.node_id)]

    # === 诊断一轮 ===
    def run_diagnosis_turn(
        self, session: TutorSession, question: Dict[str, Any], answer: str
    ) -> Dict[str, Any]:
        node = self.kg.find_node(session.node_id)
        criteria = node.mastery_rubric if node else []
        session.transcript.append(
            {"role": "student", "task_id": "T1",
             "content": f"诊断：{question.get('prompt','')}\n回答：{answer}",
             "at": datetime.now().strftime("%H:%M:%S")}
        )

        if self.llm:
            prompt = build_diagnosis_prompt(
                session.node_id, question, answer, kg_criteria=criteria,
                history=session.diag_history,
            )
            resp = self.llm(SYSTEM_PROMPT, prompt)
            raw = resp.get("content", "")
            control = extract_tagged_json(raw, "DIAGNOSIS_JSON")
            visible = strip_tagged_json(raw, "DIAGNOSIS_JSON")
            session.cost_yuan += resp.get("cost_yuan", 0.0)
        else:
            control, visible = self._fallback_diag(answer)

        mastery = _safe_float(control.get("mastery"), 0.5)
        session.diag_history.append({"question": question, "answer": answer,
                                     "mastery": mastery, "control": control})
        session.transcript.append(
            {"role": "tutor", "task_id": "T1", "content": visible,
             "at": datetime.now().strftime("%H:%M:%S"), "control": control}
        )

        ready = control.get("ready_for_execution") or control.get("decision") == "finish_to_execution"
        answered = len(session.diag_history)
        next_q = None
        if not ready and answered < 5:
            qs = self.diagnose.build_progressive_questions(session.node_id)
            if answered < len(qs):
                next_q = asdict(qs[answered])
        if next_q is None:
            ready = True
        if ready:
            session.current_task_id = "T2"
        return {"visible": visible, "control": control, "mastery": mastery,
                "next_question": next_q, "ready_for_execution": ready}

    # === 执行教学一轮 ===
    def run_execution_turn(
        self, session: TutorSession, student_message: str,
        time_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task = self._current_task(session)
        # 取一道关联真题做标准解锚点
        items = self.items.find_items(kg_node=session.node_id, limit=1)
        item = items[0] if items else None

        # 恶性循环检测
        vicious = self.teach.detect_vicious_cycle(
            session.node_id, session.recent_masteries[-3:], session.last_tutor_action
        )
        depth = self.teach.select_depth("ok", vicious.fail_streak)
        ctx = self.teach.build_teaching_context(
            session.node_id, item=item, used_angles=session.used_angles
        )

        session.transcript.append(
            {"role": "student", "task_id": task["task_id"], "content": student_message,
             "at": datetime.now().strftime("%H:%M:%S")}
        )

        ctx_dict = _ctx_to_dict(ctx)
        # 把完整 rubric（含 point_id）给 LLM，让它逐点判定真伪
        if item:
            ctx_dict["rubric_points"] = [
                {"point_id": p["point_id"], "desc": p["desc"], "must_have": p.get("must_have", False)}
                for p in item.get("rubric", [])
            ]

        if self.llm:
            prompt = build_teach_prompt(
                session.node_id, ctx_dict, task, student_message,
                depth=depth, vicious_flag=vicious.triggered, time_context=time_ctx,
                transcript_tail=session.transcript,
            )
            resp = self.llm(SYSTEM_PROMPT, prompt)
            raw = resp.get("content", "")
            control = extract_tagged_json(raw, "CONTROL_JSON")
            visible = strip_tagged_json(raw, "CONTROL_JSON")
            session.cost_yuan += resp.get("cost_yuan", 0.0)
        else:
            control, visible = self._fallback_exec(student_message, vicious.triggered)

        # 客观校验 mastery（有真题 rubric 时）
        # 关键：用 LLM 的 rubric_verdicts 判定每个得分点真伪，解决"关键词在但答错"误判
        llm_mastery = _safe_float(control.get("mastery"), 0.5)
        if item:
            rubric = [RubricPoint(p["point_id"], p["desc"], p["keywords"], p["score"],
                                  p.get("must_have", False), p.get("kg_node", ""))
                      for p in item.get("rubric", [])]
            verdicts = control.get("rubric_verdicts") if isinstance(control.get("rubric_verdicts"), dict) else None
            res = self.diagnose.check_against_rubric(student_message, rubric, point_verdicts=verdicts)
            est = self.diagnose.estimate_mastery(
                res, llm_mastery,
                has_numeric_answer=self.items.has_numeric_answer(item["item_id"]),
            )
            mastery = est.value
        else:
            mastery = llm_mastery

        session.recent_masteries.append(mastery)
        session.last_tutor_action = control.get("action", "")
        used_angle = control.get("used_angle")
        if used_angle and used_angle not in session.used_angles:
            session.used_angles.append(used_angle)

        # 状态机决策
        tasks = [TaskSpec(**{k: t[k] for k in ("task_id", "title", "duration_min",
                 "mastery_gate", "max_extend_min", "depends_on", "priority", "status")})
                 for t in session.tasks]
        cur = next(t for t in tasks if t.task_id == task["task_id"])
        tc = TimeContext(**(time_ctx or {})) if time_ctx else TimeContext(
            task_budget_min=task["duration_min"], total_budget_min=session.time_budget_min)
        decision = self.task_machine.decide(
            cur, mastery, tc, tasks, llm_decision=control.get("decision", "stay"),
            next_task_hint=control.get("next_task", ""), vicious_triggered=vicious.triggered,
        )
        if decision.to_task != task["task_id"]:
            session.current_task_id = decision.to_task

        if decision.reason and "未过闸门" in decision.reason:
            visible += f"\n\n> {decision.reason}"

        session.transcript.append(
            {"role": "tutor", "task_id": decision.to_task, "content": visible,
             "at": datetime.now().strftime("%H:%M:%S"), "control": control,
             "mastery": mastery, "decision": asdict(decision)}
        )
        return {"visible": visible, "control": control, "mastery": mastery,
                "decision": asdict(decision), "vicious": vicious.triggered,
                "depth": depth, "suggested_angle": ctx.suggested_angle}

    # === 复盘 ===
    def run_report(self, session: TutorSession) -> Dict[str, Any]:
        if not self.llm:
            return {"visible": "（本地模式无复盘，填入 API Key 后生成。）", "events": []}
        prompt = build_report_prompt(session.node_id, session.transcript)
        resp = self.llm(SYSTEM_PROMPT, prompt)
        raw = resp.get("content", "")
        events = extract_tagged_json(raw, "EVENTS_JSON")
        visible = strip_tagged_json(raw, "EVENTS_JSON")
        session.cost_yuan += resp.get("cost_yuan", 0.0)
        return {"visible": visible, "events": events.get("events", [])}

    # === fallback（无 LLM）===
    def _fallback_diag(self, answer: str):
        weak = any(t in answer for t in ["不会", "不懂", "不知道", "蒙", "忘"]) or len(answer.strip()) < 16
        score = 0.35 if weak else 0.7
        control = {"mastery": score, "decision": "ask_same" if weak else "ask_next",
                   "weak_axes": ["concept"], "ready_for_execution": False,
                   "reason": "本地启发式"}
        visible = f"先按 {score:.0%} 估计。填入 API Key 后会用 KG 判据严格批改。"
        return control, visible

    def _fallback_exec(self, msg: str, vicious: bool):
        weak = any(t in msg for t in ["不会", "不懂", "错", "卡"])
        mastery = 0.4 if weak else 0.7
        control = {"mastery": mastery, "decision": "stay" if weak else "advance",
                   "action": "micro_explain" if vicious else "diagnose",
                   "engagement": "med", "affect": "frustrated" if weak else "ok",
                   "reason": "本地启发式"}
        visible = "（本地模式）填入 API Key 后会用标准解锚点深讲。请把你的判断依据写完整。"
        return control, visible

    # === 工具 ===
    def _current_task(self, session: TutorSession) -> Dict[str, Any]:
        for t in session.tasks:
            if t["task_id"] == session.current_task_id:
                return t
        return session.tasks[0]

    def _make_id(self, user_id: str, goal: str) -> str:
        raw = f"{user_id}|{goal}|{datetime.now().isoformat()}"
        return "s1_" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def _ctx_to_dict(ctx) -> Dict[str, Any]:
    return {
        "node_id": ctx.node_id, "standard_solution": ctx.standard_solution,
        "rubric_desc": ctx.rubric_desc, "solving_skeleton": ctx.solving_skeleton,
        "setter_traps": ctx.setter_traps, "yihuier_chunks": ctx.yihuier_chunks,
        "recommended_video": ctx.recommended_video, "suggested_angle": ctx.suggested_angle,
    }


def _safe_float(v, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default
