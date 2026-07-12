#!/usr/bin/env python3
"""WS2 转写官方表 loader 的持久回归(focused runner,pytest 不可用环境直跑)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.ws2_transcripts import (  # noqa: E402
    WS2_TRANSCRIPTS, load_transcripts, load_media_ref_map, renderable_latex,
    ai_citable_transcript, display_degraded, resolve_image_path, stats,
)


def test_applied_ws2_invariants():
    """对已 apply 的官方转写表的持久基线(2026-07-05 batch10 QA-1 并库)。"""
    if not WS2_TRANSCRIPTS.exists():
        return
    st = stats()
    assert st["total"] == 10102  # batch13: +22 新行(18 formula + 4 transcript, 死引用裁片新hash)
    assert st["pools"] == {
        # 2026-07-05 batch10: 10a 新增 4075 答案/解析区转写行(formula_latex/ai_seed/
        # display_only/manual_queue 各池增量),10b/10d 改现有行(不增行)。
        # 2026-07-05 batch12: 12b 救回资产 94 行升池(ai_seed +59, display_only +35)
        # + 132 行图救回升 display_only(broken 定性不成立): manual_queue 564→338,
        # display_only 1414→1573(icon 8 条保持静默不升), ai_seed 2567→2626。行数不变。
        # 2026-07-05 batch13: 裁片13d 97条升池/新增(formula+18,ai_seed+57升,display+22升)
        "formula_latex": 5552, "ai_seed": 2683, "display_only": 1595,
        "manual_queue": 271, "leak_rejected": 1,
    }
    assert st["renderable_latex"] == 5478
    assert st["with_transcript"] == 4308


def test_rule_r1_r2_r3_pool_discipline():
    rows = load_transcripts()
    for r in rows.values():
        cit = ai_citable_transcript(r)
        if cit is not None:
            assert r["pool"] == "ai_seed"
            assert r.get("fine_type") not in ("broken_image", "icon_or_noise")
        if r["pool"] in ("manual_queue", "leak_rejected"):
            assert ai_citable_transcript(r) is None
            assert display_degraded(r)


def test_rule_r4_latex_gate():
    rows = load_transcripts()
    n = 0
    for r in rows.values():
        lx = renderable_latex(r)
        if lx is not None:
            assert r["latex_status"] == "passed"
            n += 1
    assert n == 5478


def test_media_map_stem_coverage_and_no_dangling_stem():
    m = load_media_ref_map()
    # 13143:apply 当日修正 group 名规范化(补 ~/- 变体)后的基线(原 12790,覆盖 18715→19219)
    assert len(m) == 13171  # batch13: +28 死引用新映射
    rows = load_transcripts()
    from core.data.ws2_transcripts import _iter_jsonl, WS2_MEDIA_REF_MAP
    for r in _iter_jsonl(WS2_MEDIA_REF_MAP):
        if "stem" in (r.get("zones") or []):
            assert r["asset_hash"] in rows, f"题干区映射悬空: {r['group_key']}/{r['media']}"


def test_trap_fragment_literal():
    """陷阱基线:上下文依赖碎片必须按字面,禁止脑补完整离子。
    batch10(10d) 把碎片统一为渲染更稳的 prescript 形式({}^{-}_{3}),
    字面语义不变 —— 仍是孤立上下标碎片,未脑补成 \\ce{NO3-} 之类完整离子。"""
    rows = load_transcripts()
    e = next(r for h, r in rows.items() if h.startswith("e2c78343001a"))
    assert e["latex"] == r"{}^{-}_{3}"
    # 二锚点:另一条独立碎片同样保持字面(10d Claude 改判,图面依据见 BATCH10_AUDIT)
    e2 = next(r for h, r in rows.items() if h.startswith("577c78a5d28a"))
    assert e2["latex"] == r"{}^{-}_{3}"


def test_v3_repository_unaffected():
    from core.data.item_repository import ItemRepository
    assert ItemRepository().count() == 6438


def test_v4_loader_unaffected():
    from core.data.item_bank_v4 import service_pool_stats
    # R1-R4 数据口径 2526(R5 apply 不动题库);R5 学生端口径 1203(2026-07-06,batch14 台账)。
    st = service_pool_stats(apply_r5=False)
    assert st["total"] == 3329 and st["servable"] == 2526
    st_r5 = service_pool_stats()
    # 1202 = batch15 15c 的 1207，减去 Demo 2026-07-13 双金标裁决的 5 条坏题。
    assert st_r5["servable"] == 1202


def _run_focused():
    import inspect
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{failures} FAILURES" if failures else "\nALL PASS")
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_focused())
