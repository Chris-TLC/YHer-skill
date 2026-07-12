#!/usr/bin/env python3
"""练习舱 v1(study API + study.html)持久回归。

pytest 本机不可用时直接 `python3 tests/test_study_v1.py` 走 focused runner。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))


def test_study_pool_is_r5_whitelist_only():
    """学生口径红线:study 池必须 == R5 白名单(1202),与审计面(2526)不同池。"""
    from apps.api_v4_render import _study_pool, _pool
    sp = _study_pool()
    wl = set()
    for line in (SKILL_DIR / "data/item_bank/v4/usability_r5_v1.jsonl").open():
        row = json.loads(line)
        if row.get("r5_serve"):
            wl.add(row["item_id"])
    assert set(sp.keys()) == wl, "study 池偏离 R5 白名单"
    assert len(_pool()) > len(sp), "审计面应大于学生面(apply_r5=False)"


def test_study_next_filters_seen_and_node():
    import apps.api_v4_render as m
    import tempfile

    old_log_dir = m.STUDY_LOG_DIR
    with tempfile.TemporaryDirectory() as td:
        m.STUDY_LOG_DIR = Path(td)
        try:
            pool = m._study_pool()
            first = m.study_next({"seen_ids": []})
            assert first["done"] is False and first["item_id"] in pool
            # seen 过滤:把全部题标 seen → done
            alldone = m.study_next({"seen_ids": list(pool.keys())})
            assert alldone["done"] is True
            # 节点过滤:抽出的题必须挂该节点
            node = next(iter(pool.values())).get("kg_nodes", ["其他"])[0]
            r = m.study_next({"node": node, "seen_ids": []})
            assert r["done"] is False and node in (pool[r["item_id"]].get("kg_nodes") or ["其他"])
        finally:
            m.STUDY_LOG_DIR = old_log_dir


def test_study_event_and_report_write_pending_rows(tmp_path=None):
    import apps.api_v4_render as m
    import tempfile
    old = m.STUDY_LOG_DIR
    with tempfile.TemporaryDirectory() as td:
        m.STUDY_LOG_DIR = Path(td)
        try:
            m.study_event({"event": "item_shown", "session_id": "t1", "item_id": "x" * 40})
            r = m.study_report_bad({"item_id": "x" * 40, "session_id": "t1", "reason": "答案错误"})
            assert r["ok"]
            rows = [json.loads(l) for l in (Path(td) / "bad_reports.jsonl").open()]
            # 治理:报坏题行必须 pending + 空 reviewer(等 Claude/用户签,治理v2外部收敛器)
            assert rows[0]["status"] == "pending" and rows[0]["reviewer"] == ""
            ev_files = list(Path(td).glob("events_*.jsonl"))
            assert ev_files and json.loads(ev_files[0].read_text().splitlines()[0])["event"] == "item_shown"
        finally:
            m.STUDY_LOG_DIR = old


def test_study_html_contract():
    """前端契约:study.html 必须带报坏题按钮、埋点调用、正确的渲染器入口。"""
    html = (SKILL_DIR / "apps/web/study.html").read_text(encoding="utf-8")
    assert "YHerRirRenderer" in html and "renderRir" in html
    assert "/api/v4/study/next" in html and "/api/v4/study/event" in html
    assert "/api/v4/study/report_bad" in html
    assert "报坏题" in html


def _run_focused():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {e}")
    print("\nALL PASS" if not failures else f"\n{failures} FAILURES")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_focused() else 0)
