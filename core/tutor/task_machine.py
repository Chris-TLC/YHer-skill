#!/usr/bin/env python3
"""
Task state machine (master blueprint B-4).

Three ways it is more reliable than the status quo:
1. The gate only trusts objective mastery (MasteryEstimate.value), not the LLM's self-report.
2. Hard three-state timebox rule: on_track / extend_current / wrap_up. If time runs out before the gate is passed, record unfinished_gap instead of pretending it passed.
3. Dynamic reordering: after diagnosis, reorder T2-T5 by weak_axes.
4. Vicious-cycle interlock: when vicious_flag is set, override with a rewind to the minimal concept.

Designed for reuse in stage 3: reorder_queue scales from one lesson to a cross-subject week with zero algorithm changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
    """Reorderable elements of a task (aligned with private_tutor.TutorTask, scaled up for reuse in stage 3)."""

    task_id: str
    title: str
    duration_min: int
    mastery_gate: float = 0.72
    max_extend_min: int = 15
    depends_on: List[str] = field(default_factory=list)
    priority: float = 0.5  # higher runs first (driven by weakness severity)
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
    forced_action: str = ""  # teaching action forced on vicious-cycle trigger
    time_action: str = "on_track"  # on_track|extend_current|compress_later|wrap_up
    gate_passed: bool = False
    mastery: float = 0.0
    unfinished_gap: bool = False
    reason: str = ""


class TaskMachine:
    # === Gate: trusts objective mastery only ===
    def gate_check(self, task: TaskSpec, mastery_value: float) -> bool:
        return mastery_value >= task.mastery_gate

    # === Three-state timebox ===
    def time_action(self, tc: TimeContext, mastery_value: float, gate: float) -> str:
        ratio = tc.task_elapsed_min / max(tc.task_budget_min, 1)
        total_ratio = tc.total_elapsed_min / max(tc.total_budget_min, 1)
        if total_ratio >= 0.95:
            return "wrap_up"
        if ratio < 0.8:
            return "on_track"
        if ratio < 1.5:
            # close to the gate: worth extending, otherwise compress what follows
            return "extend_current" if mastery_value >= gate - 0.15 else "compress_later"
        return "wrap_up"

    # === Main decision ===
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

        # Vicious cycle overrides first: go back to the minimal concept, no new questions
        if vicious_triggered:
            return TaskDecision(
                from_task=task.task_id, to_task=task.task_id, decision="stay",
                forced_action="micro_explain", time_action=t_action,
                gate_passed=False, mastery=mastery_value,
                reason="恶性循环：强制回最小概念，禁止再抛新题。",
            )

        decision = llm_decision or "stay"
        # Gate interception: LLM wants to advance but objective mastery failed → stay
        if decision == "advance" and not gate_passed:
            decision = "stay"
            reason = f"掌握度 {mastery_value:.2f} 未过闸门 {task.mastery_gate:.2f}，留在当前阶段。"
        else:
            reason = ""

        # Time's up, wrap up
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

    # === Dynamic reordering (core for stage 3 reuse) ===
    def reorder_queue(
        self, tasks: List[TaskSpec], weak_axes: Optional[List[str]] = None,
        axis_to_task: Optional[Dict[str, str]] = None,
    ) -> List[TaskSpec]:
        """
        Reorder by weakness severity. Tasks mapped from the diagnosed
        weak_axes get a priority boost. The scope scales from one lesson's
        5 tasks to a week's N cross-subject tasks with the algorithm unchanged.
        """
        weak_axes = weak_axes or []
        axis_to_task = axis_to_task or {}
        # Boost the priority of tasks tied to weaknesses
        for axis in weak_axes:
            tid = axis_to_task.get(axis)
            for t in tasks:
                if t.task_id == tid:
                    t.priority = min(1.0, t.priority + 0.3)
        # Topologically safe priority sort: respect depends_on, then descending priority
        return self._priority_sort_with_deps(tasks)

    def _priority_sort_with_deps(self, tasks: List[TaskSpec]) -> List[TaskSpec]:
        by_id = {t.task_id: t for t in tasks}
        done: List[TaskSpec] = []
        placed = set()
        # Simple stable sort: repeatedly pick the highest-priority task whose dependencies are placed
        remaining = list(tasks)
        while remaining:
            ready = [t for t in remaining if all(d in placed for d in t.depends_on)]
            if not ready:  # cycle fallback: place the rest in original order
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
