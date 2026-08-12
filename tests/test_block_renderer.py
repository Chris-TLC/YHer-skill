#!/usr/bin/env python3
"""WS4 块渲染器回归(focused runner)。基线:2026-07-04 终版烟测。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.item_bank_v4 import iter_service_items  # noqa: E402

# 本文件所有遍历显式 apply_r5=False:渲染烟测的对象是"可渲染宇宙"(R1-R4 口径 2526,
# 含 fixable/blocked —— 预览/审计仍要渲它们);R5 白名单只缩学生端 serve,不缩渲染契约。
# (2026-07-06 R5 apply)
from core.render.block_renderer import item_to_rir, _Ctx  # noqa: E402


def test_full_service_pool_never_raises_and_baseline():
    """全服务池零异常 + 降级基线(题干区 308 题,全部为坏图/OMML 预期降级)。
    2026-07-05 batch10: 365→354 —— 10a 答案区资产转写/修复复用到题干、10d latex 修正,
    11 题题干资产复活;stem_blocks 本身零改动(10c 题干回填已因 ref_map 不配套砍掉,留 QA-3)。
    2026-07-05 batch12: 354→298; batch13 裁片: 298→238(105 裁片回原卷救活) —— 12b 重渲救回 226 资产(221 收编)+OMML 43 条补收
    + 132 行图救回升 display_only,合计题干脱离降级。"""
    n = deg_stem = 0
    reasons = {}
    for it in iter_service_items(apply_r5=False):
        n += 1
        rir = item_to_rir(it, zones=("stem",))
        if rir["degraded"]:
            deg_stem += 1
            for r in rir["degrade_reasons"]:
                k = r.split(":")[0]
                reasons[k] = reasons.get(k, 0) + 1
    assert n == 2526
    assert deg_stem == 238, f"题干降级题数漂移: {deg_stem}(基线 238,变化须解释)"
    assert set(reasons) <= {"asset_degraded", "omml_unconvertible", "media_unmapped"}
    assert reasons.get("media_unmapped", 0) <= 1


def test_rir_node_kinds_closed_set():
    kinds = set()
    for i, it in enumerate(iter_service_items(apply_r5=False)):
        rir = item_to_rir(it, zones=("stem", "answer"))
        for paras in rir["zones"].values():
            for para in paras:
                for node in para:
                    kinds.add(node["kind"])
                    if node["kind"] == "table":
                        for row in node["rows"]:
                            for cell in row:
                                for cn in cell:
                                    kinds.add(cn["kind"])
        if i > 400:
            break
    assert kinds <= {"text", "latex", "image", "table", "placeholder"}, kinds


def test_formula_asset_latex_priority():
    """含 ⇌ 高频公式图的题必须渲染为 latex 而非图片。"""
    ctx = _Ctx.get()
    target_hash_prefix = "d721f42eee5a"
    hit = 0
    for it in iter_service_items(apply_r5=False):
        rir = item_to_rir(it, zones=("stem",))
        for paras in rir["zones"].values():
            for para in paras:
                for node in para:
                    if node["kind"] == "latex" and node.get("source") == "formula_asset" \
                            and node["latex"] == r"\ce{<=>}":
                        hit += 1
        if hit:
            break
    assert hit > 0, "⇌ 公式图未走 latex 渲染"


def test_broken_asset_placeholder_and_flag():
    """171 broken 资产引用必须占位并打 degraded 标。"""
    import json
    broken = None
    for line in open(Path(__file__).parent.parent / "data/item_bank/v4/ws2_asset_transcripts_v1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("fine_type") == "broken_image":
            broken = r["asset_hash"]
            break
    assert broken
    ctx = _Ctx.get()
    gm = next(((g, m) for (g, m), h in ctx.ref_map.items() if h == broken), None)
    assert gm, "broken 资产无引用"
    fake_item = {"item_id": "t", "group_key": gm[0],
                 "stem_blocks": [{"para": [{"type": "figure", "media": gm[1]}]}]}
    rir = item_to_rir(fake_item, zones=("stem",))
    assert rir["degraded"]
    assert rir["zones"]["stem"][0][0]["kind"] == "placeholder"


def test_icon_noise_silent():
    """icon_or_noise 资产静默为 text,不降级。"""
    import json
    icon = None
    for line in open(Path(__file__).parent.parent / "data/item_bank/v4/ws2_asset_transcripts_v1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("fine_type") == "icon_or_noise":
            icon = r["asset_hash"]
            break
    ctx = _Ctx.get()
    gm = next(((g, m) for (g, m), h in ctx.ref_map.items() if h == icon), None)
    if gm is None:
        return  # 该资产无引用,跳过
    fake_item = {"item_id": "t", "group_key": gm[0],
                 "stem_blocks": [{"para": [{"type": "figure", "media": gm[1]}]}]}
    rir = item_to_rir(fake_item, zones=("stem",))
    assert not rir["degraded"]
    assert rir["zones"]["stem"][0][0]["kind"] == "text"


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
