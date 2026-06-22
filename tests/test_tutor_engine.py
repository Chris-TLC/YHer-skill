#!/usr/bin/env python3
"""
引擎层测试（不依赖 LLM / 不依赖 BGE 模型，纯逻辑验证）。

运行：python3 tests/test_tutor_engine.py
"""

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))


def test_knowledge_repository():
    from core.data.knowledge_repository import get_knowledge_repository
    repo = get_knowledge_repository()
    assert len(repo.all_nodes()) >= 60, "KG 节点应 >= 60"
    assert len(repo.all_patterns()) >= 10, "题型应 >= 10"
    node = repo.find_node("化学平衡")
    assert node is not None
    assert len(node.mastery_rubric) > 0, "应有掌握判据"
    assert len(node.diagnostic_questions()) > 0, "应有诊断问题"
    print("✅ test_knowledge_repository")


def test_item_repository():
    from core.data.item_repository import get_item_repository
    repo = get_item_repository()
    assert repo.count() >= 2, "应有样例题"
    item = repo.get_item("eq-sample-001")
    assert item is not None
    assert any(p.get("must_have") for p in item["rubric"]), "应有 must_have 得分点"
    print("✅ test_item_repository")


def test_rubric_must_have_cap():
    """核心：漏掉 must_have 得分点，mastery 被封顶，LLM 不能抬分。"""
    from core.tutor.diagnose_engine import DiagnoseEngine, RubricPoint
    eng = DiagnoseEngine()
    rubric = [
        RubricPoint("sp1", "三段式", ["三段式"], 2.0, must_have=True),
        RubricPoint("sp4", "分母陷阱", ["分母"], 2.0, must_have=True),
    ]
    # LLM 判定 sp4 没答对
    res = eng.check_against_rubric("我列了三段式", rubric, point_verdicts={"sp1": True, "sp4": False})
    assert "sp4" in res.missed_must_have
    est = eng.estimate_mastery(res, llm_self_score=0.95, has_numeric_answer=True)
    assert est.value <= 0.6, f"漏 must_have 应封顶 0.6，实际 {est.value}"
    print("✅ test_rubric_must_have_cap")


def test_rubric_verdict_overrides_keyword():
    """LLM 真伪判定优先于关键词命中（解决'转化率升高'误判）。"""
    from core.tutor.diagnose_engine import DiagnoseEngine, RubricPoint
    eng = DiagnoseEngine()
    rubric = [RubricPoint("sp4", "转化率降低", ["转化率"], 2.0, must_have=True)]
    # 学生说"转化率升高"（含关键词但答错）
    res_kw = eng.check_against_rubric("转化率升高", rubric)
    assert "sp4" in res_kw.hit_points, "纯关键词会误判命中"
    res_llm = eng.check_against_rubric("转化率升高", rubric, point_verdicts={"sp4": False})
    assert "sp4" in res_llm.missed_must_have, "LLM 判定应纠正误判"
    print("✅ test_rubric_verdict_overrides_keyword")


def test_multi_angle_teaching():
    """多讲法：卡住时换轨，不重复。"""
    from core.tutor.teach_engine import TeachEngine
    eng = TeachEngine()
    used = []
    angles = []
    for _ in range(4):
        a = eng.next_angle(used)
        angles.append(a)
        used.append(a)
    assert len(set(angles)) == 4, "前 4 次应换 4 种不同讲法"
    print("✅ test_multi_angle_teaching")


def test_vicious_cycle():
    """恶性循环检测：连错+抛新题→触发。"""
    from core.tutor.teach_engine import TeachEngine
    eng = TeachEngine()
    f = eng.detect_vicious_cycle("化学平衡", [0.4, 0.3], "adaptive_practice")
    assert f.triggered
    f2 = eng.detect_vicious_cycle("化学平衡", [0.4, 0.3], "micro_explain")
    assert not f2.triggered, "上轮是讲解不应触发"
    print("✅ test_vicious_cycle")


def test_gate_blocks_llm():
    """闸门拦截 LLM 自报：客观没过不放行。"""
    from core.tutor.task_machine import TaskMachine, TaskSpec, TimeContext
    tm = TaskMachine()
    tasks = [TaskSpec("T2", "补洞", 40, 0.68)]
    tc = TimeContext(task_elapsed_min=10, task_budget_min=40, total_elapsed_min=50, total_budget_min=180)
    d = tm.decide(tasks[0], 0.5, tc, tasks, llm_decision="advance")
    assert d.decision == "stay", "闸门应拦截 LLM 的 advance"
    print("✅ test_gate_blocks_llm")


def test_no_fake_pass():
    """时间到+闸门没过→记 unfinished_gap，不假装过关。"""
    from core.tutor.task_machine import TaskMachine, TaskSpec, TimeContext
    tm = TaskMachine()
    tasks = [TaskSpec("T2", "补洞", 40, 0.68), TaskSpec("T6", "复盘", 15, 0.7)]
    tc = TimeContext(task_elapsed_min=40, task_budget_min=40, total_elapsed_min=175, total_budget_min=180)
    d = tm.decide(tasks[0], 0.5, tc, tasks, llm_decision="stay")
    assert d.unfinished_gap, "时间到没过闸门应记 unfinished_gap"
    print("✅ test_no_fake_pass")


def test_student_model_rubric_protection():
    """rubric 来源的掌握度不被 LLM 自评覆盖。"""
    from core.tutor.profile_model import SubjectAbility, MasteryRecord
    sa = SubjectAbility()
    sa.update_node("化学平衡", MasteryRecord(value=0.45, source="rubric"))
    sa.update_node("化学平衡", MasteryRecord(value=0.9, source="llm"))
    assert sa.kg_mastery["化学平衡"].value == 0.45, "rubric 客观分不应被 LLM 覆盖"
    print("✅ test_student_model_rubric_protection")


def test_orchestrator_fallback():
    """无 LLM 时编排层走 fallback 不崩。"""
    from core.tutor.session_orchestrator import SessionOrchestrator
    orch = SessionOrchestrator(llm_caller=None)
    s = orch.create_session("u1", "补化学平衡", node_id="化学平衡")
    assert len(s.tasks) == 6
    q = orch.first_question(s)
    assert q.get("prompt")
    r = orch.run_diagnosis_turn(s, q, "不太会")
    assert "mastery" in r
    s.current_task_id = "T2"
    r2 = orch.run_execution_turn(s, "列了三段式算K=2.37")
    assert "mastery" in r2
    print("✅ test_orchestrator_fallback")


def test_retriever_port():
    """检索 Port 三模式可用。"""
    from adapters.retrieval import make_retriever
    r = make_retriever("null")
    assert r.retrieve_with_diagnosis("x")["related_nodes"] == []
    make_retriever("local")  # 懒加载不崩
    make_retriever("cloud")
    print("✅ test_retriever_port")


def test_store_port():
    """本地 JSON 存储读写。"""
    import tempfile
    from adapters.store import LocalJsonStore
    with tempfile.TemporaryDirectory() as d:
        s = LocalJsonStore(store_dir=d)
        s.save_student("u", {"grade": "高二"})
        assert s.load_student("u")["grade"] == "高二"
        s.append_event("u", {"node": "化学平衡"})
        assert len(s.recent_events("u")) == 1
    print("✅ test_store_port")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 测试通过")
    sys.exit(0 if passed == len(tests) else 1)
