#!/usr/bin/env python3
"""
任务状态机（总蓝图 B-4）。

比现状更可靠的三点：
1. 闸门只认客观 mastery（MasteryEstimate.value），不信 LLM 自报。
2. 时间盒三态硬规则：on_track / extend_current / wrap_up，闸门没过但时间到→记 unfinished_gap 不假装过关。
3. 动态重排：诊断后按 weak_axes 重排 T2-T5。
4. 恶性循环联动：vicious_flag 时 override 为 rewind 到最小概念。

设计为阶段三复用：reorder_queue 作用域从一节课放大到一周跨科，算法零改动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
    """一个任务的可重排要素（与 private_tutor.TutorTask 对齐，阶段三放大复用）。"""

    task_id: str
    title: str
    duration_min: int
    mastery_gate: float = 0.72
    max_extend_min: int = 15
    depends_on: List[str] = field(default_factory=list)
    priority: float = 0.5  # 越大越优先（由弱点严重度决定）
    status: str = "pending"  # pending|active|done|skipped


@dataclass
class TimeContext:
    task_elapsed_min: float = 0.0
    task_budget_min: float = 30.0
    total_elapsed_min: float = 0.0
    total_budget_min: float = 180.0


@dataclass
class TaskDecision:
    from_task: str = ""
    to_task: str = ""
    decision: str = "stay"  # stay|advance|rewind|skip|inject_video|wrap_up
    forced_action: str = ""  # 恶性循环时强制的教学动作
    time_action: str = "on_track"  # on_track|extend_current|compress_later|wrap_up
    gate_passed: bool = False
    mastery: float = 0.0
    unfinished_gap: bool = False
    reason: str = ""


class TaskMachine:
    # === 闸门：只认客观 mastery ===
    def gate_check(self, task: TaskSpec, mastery_value: float) -> bool:
        return mastery_value >= task.mastery_gate

    # === 时间盒三态 ===
    def time_action(self, tc: TimeContext, mastery_value: float, gate: float) -> str:
        ratio = tc.task_elapsed_min / max(tc.task_budget_min, 1)
        total_ratio = tc.total_elapsed_min / max(tc.total_budget_min, 1)
        if total_ratio >= 0.95:
            return "wrap_up"
        if ratio < 0.8:
            return "on_track"
        if ratio < 1.5:
            # 接近闸门值得延长，否则压缩后续
            return "extend_current" if mastery_value >= gate - 0.15 else "compress_later"
        return "wrap_up"

    # === 主决策 ===
    def decide(
        self,
        task: TaskSpec,
        mastery_value: float,
        time_ctx: TimeContext,
        tasks: List[TaskSpec],
        llm_decision: str = "stay",
        next_task_hint: str = "",
        vicious_triggered: bool = False,
    ) -> TaskDecision:
        gate_passed = self.gate_check(task, mastery_value)
        t_action = self.time_action(time_ctx, mastery_value, task.mastery_gate)
        idx = self._index(tasks, task.task_id)

        # 恶性循环优先 override：回到最小概念，禁止抛新题
        if vicious_triggered:
            return TaskDecision(
                from_task=task.task_id, to_task=task.task_id, decision="stay",
                forced_action="micro_explain", time_action=t_action,
                gate_passed=False, mastery=mastery_value,
                reason="恶性循环：强制回最小概念，禁止再抛新题。",
            )

        decision = llm_decision or "stay"
        # 闸门拦截：LLM 想 advance 但客观没过 → 留下
        if decision == "advance" and not gate_passed:
            decision = "stay"
            reason = f"掌握度 {mastery_value:.2f} 未过闸门 {task.mastery_gate:.2f}，留在当前阶段。"
        else:
            reason = ""

        # 时间到收束
        if t_action == "wrap_up" and decision == "stay":
            unfinished = not gate_passed
            to_task = tasks[-1].task_id if tasks else task.task_id
            return TaskDecision(
                from_task=task.task_id, to_task=to_task, decision="wrap_up",
                time_action=t_action, gate_passed=gate_passed, mastery=mastery_value,
                unfinished_gap=unfinished,
                reason="时间到，收束本次；闸门未过的记为待续，不假装过关。" if unfinished else "时间到，正常收束。",
            )

        to_task = task.task_id
        if decision == "advance":
            to_task = next_task_hint or (tasks[idx + 1].task_id if idx + 1 < len(tasks) else task.task_id)
        elif decision == "rewind":
            to_task = tasks[idx - 1].task_id if idx > 0 else task.task_id
        elif decision == "skip":
            to_task = tasks[idx + 1].task_id if idx + 1 < len(tasks) else task.task_id
        elif decision == "finish":
            to_task = tasks[-1].task_id if tasks else task.task_id

        return TaskDecision(
            from_task=task.task_id, to_task=to_task, decision=decision,
            time_action=t_action, gate_passed=gate_passed, mastery=mastery_value,
            reason=reason or f"决策 {decision}。",
        )

    # === 动态重排（阶段三复用核心）===
    def reorder_queue(
        self, tasks: List[TaskSpec], weak_axes: Optional[List[str]] = None,
        axis_to_task: Optional[Dict[str, str]] = None,
    ) -> List[TaskSpec]:
        """
        按弱点严重度重排。诊断出的 weak_axes 对应的任务优先级抬高。
        作用域从一节课的 5 个 task 放大到一周跨科 N 个 task，算法不变。
        """
        weak_axes = weak_axes or []
        axis_to_task = axis_to_task or {}
        # 抬高弱点相关任务优先级
        for axis in weak_axes:
            tid = axis_to_task.get(axis)
            for t in tasks:
                if t.task_id == tid:
                    t.priority = min(1.0, t.priority + 0.3)
        # 拓扑安全的优先级排序：尊重 depends_on，再按 priority 降序
        return self._priority_sort_with_deps(tasks)

    def _priority_sort_with_deps(self, tasks: List[TaskSpec]) -> List[TaskSpec]:
        by_id = {t.task_id: t for t in tasks}
        done: List[TaskSpec] = []
        placed = set()
        # 简单稳定排序：反复挑出依赖已满足里 priority 最高的
        remaining = list(tasks)
        while remaining:
            ready = [t for t in remaining if all(d in placed for d in t.depends_on)]
            if not ready:  # 循环依赖兜底：直接按原序放剩下的
                ready = remaining
            pick = max(ready, key=lambda t: t.priority)
            done.append(pick)
            placed.add(pick.task_id)
            remaining.remove(pick)
        return done

    def _index(self, tasks: List[TaskSpec], task_id: str) -> int:
        for i, t in enumerate(tasks):
            if t.task_id == task_id:
                return i
        return 0
