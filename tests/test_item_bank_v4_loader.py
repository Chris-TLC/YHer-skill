#!/usr/bin/env python3
"""v4 loader(core/data/item_bank_v4.py)硬规则测试 + v3/v4 并存不变量。

pytest 在本机不可用时,直接 `python3 tests/test_item_bank_v4_loader.py` 走 focused runner。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from core.data.item_bank_v4 import (  # noqa: E402
    V4_MANIFEST,
    V4_SERVICE_EXCLUSIONS,
    effective_answer_blocks,
    has_effective_answer,
    iter_service_items,
    load_service_exclusions,
    load_usability_r5,
    service_blockers,
    service_pool_stats,
)


def _item(**kw):
    base = {
        "item_id": "a" * 40,
        "pool": "main",
        "service_eligible": True,
        "quality_flags": [],
        "answer_blocks_effective": [{"para": [{"type": "text", "text": "【答案】A"}]}],
        "standard_solution": {"final_answers": ["A"], "standard_answer": "A"},
    }
    base.update(kw)
    return base


def test_r1_only_main_and_eligible_serve():
    # apply_r5=False:本测试只验 R1 语义;R5 门有专属测试 test_r5_usability_gate。
    assert service_blockers(_item(), {}, apply_r5=False) == []
    assert "pool_not_main:legacy" in service_blockers(_item(pool="legacy", service_eligible=False), {}, apply_r5=False)
    assert "not_service_eligible" in service_blockers(_item(service_eligible=False), {}, apply_r5=False)


def test_r2_answer_type_mismatch_answers_treated_as_nonexistent():
    it = _item(quality_flags=["answer_type_mismatch"])
    # 即使数据里残留了答案块,规则层也视为不存在
    assert effective_answer_blocks(it) == []
    assert not has_effective_answer(it)
    blockers = service_blockers(it, {}, apply_r5=False)
    assert "answer_type_mismatch" in blockers
    assert "no_effective_answer" in blockers


def test_r3_no_answer_anywhere_blocks_service():
    it = _item(answer_blocks_effective=[], standard_solution={"final_answers": [], "standard_answer": ""})
    assert not has_effective_answer(it)
    assert "no_effective_answer" in service_blockers(it, {}, apply_r5=False)
    # 答案不在 abe 但 standard_solution 有 → 可服务(继承自 v3 核验)
    it2 = _item(answer_blocks_effective=[])
    assert has_effective_answer(it2)
    assert service_blockers(it2, {}, apply_r5=False) == []


def test_r4_service_exclusions_respected(tmp_path: Path):
    excl_path = tmp_path / "service_exclusions.jsonl"
    excl_path.write_text(
        json.dumps({"item_id": "a" * 40, "reason": "stem_is_analysis_text", "reviewer": "claude"}) + "\n"
    )
    exclusions = load_service_exclusions(excl_path)
    assert service_blockers(_item(), exclusions, apply_r5=False) == ["service_exclusion:stem_is_analysis_text"]

    manifest = tmp_path / "bank.jsonl"
    manifest.write_text(
        json.dumps(_item()) + "\n" + json.dumps(_item(item_id="b" * 40)) + "\n"
    )
    served = list(iter_service_items(manifest, excl_path, apply_r5=False))
    assert [it["item_id"] for it in served] == ["b" * 40]


def test_r5_usability_gate(tmp_path: Path):
    """R5(2026-07-06 apply):只 serve 台账 r5_serve==true;无记录/非 clean/排除类一律挡。"""
    ok = {"a" * 40: {"item_id": "a" * 40, "usability_pool": "clean", "r5_serve": True}}
    bad = {"a" * 40: {"item_id": "a" * 40, "usability_pool": "clean", "r5_serve": False,
                      "r5_block_reason": "exclusion:hollow_content"}}
    assert service_blockers(_item(), {}, usability=ok) == []
    assert service_blockers(_item(), {}, usability=bad) == [
        "usability_not_serveable:exclusion:hollow_content"
    ]
    assert service_blockers(_item(), {}, usability={}) == ["usability_missing"]

    # 文件级:iter_service_items 走 usability_path
    manifest = tmp_path / "bank.jsonl"
    manifest.write_text(json.dumps(_item()) + "\n" + json.dumps(_item(item_id="b" * 40)) + "\n")
    upath = tmp_path / "usability_r5.jsonl"
    upath.write_text(json.dumps({"item_id": "b" * 40, "usability_pool": "clean", "r5_serve": True}) + "\n")
    excl_path = tmp_path / "empty_excl.jsonl"
    excl_path.write_text("")
    served = list(iter_service_items(manifest, excl_path, usability_path=upath))
    assert [it["item_id"] for it in served] == ["b" * 40]


def test_v3_repository_never_sees_v4_subdir(tmp_path: Path):
    """并存不变量:v3 服务路径 glob(*.jsonl) 非递归,v4 子目录必须不可见。"""
    from core.data.item_repository import ItemRepository

    bank = tmp_path / "item_bank"
    (bank / "v4").mkdir(parents=True)
    (bank / "chemistry_v3.jsonl").write_text(json.dumps({"item_id": "v3-item"}) + "\n")
    (bank / "v4" / "chemistry_v4.jsonl").write_text(json.dumps({"item_id": "v4-item"}) + "\n")

    repo = ItemRepository(bank_dir=bank, quality_manifest_path=tmp_path / "no_manifest.jsonl")
    assert repo.count() == 1
    assert repo.get_item("v3-item") is not None
    assert repo.get_item("v4-item") is None


def test_applied_manifest_invariants():
    """对已 apply 的正式 v4 manifest 的持久回归(存在即跑)。"""
    if not V4_MANIFEST.exists():
        return  # apply 之前跳过
    # R1-R4 口径(apply_r5=False):数据层不变量,R5 apply 不动题库本身 —— 2526 恒成立。
    stats = service_pool_stats(apply_r5=False)
    assert stats["total"] == 3329
    # Batch 7 apply(2026-07-04)后的池基线:68 行转池(23 not_a_question + 9 bad_segmentation
    # + 36 answerless)+ 7 条 manual_inherited 答案继承复活。WS3 时基线为
    # {"main": 2677, "legacy": 230, "excluded_answerless": 422},servable 2530。
    assert stats["pools"] == {
        "main": 2609,
        "legacy": 230,
        "excluded_answerless": 458,
        "excluded_not_a_question": 23,
        "excluded_bad_segmentation": 9,
    }
    assert stats["exclusion_rows"] == 11
    # 服务池:2526 = 2530 + 7(答案继承复活)− 11(答案202501 垃圾题清出)
    assert stats["servable"] == 2526
    for it in iter_service_items(apply_r5=False):
        assert it["pool"] == "main" and it["service_eligible"] is True
        assert has_effective_answer(it)
        break  # 逐项断言在 stats 里已隐含,这里抽首行冒烟即可

    # R5 口径(默认,学生端所见):batch14 台账 1203 -> batch15 15c 终审 1207，
    # Demo 2026-07-13 双金标裁决 5 条积压坏题后为 1202。
    # (2026-07-06 R5v2 apply,用户 L1:re-admit 7 变式 keep + 移除 3 真内容漏;
    # 见 BATCH15_AUDIT_2026-07-06.md;签字 15c_variant_dispositions_signed.jsonl)。
    r5 = service_pool_stats()
    assert r5["servable"] == 1202
    assert r5["usability_rows"] == 2526
    # 2526 - 1202 = 1324 台账在册但不可 serve(fixable/blocked/排除类)
    assert r5["blocker_counts"]["usability_not_serveable"] == 1324
    usability = load_usability_r5()
    n_serve = sum(1 for row in usability.values() if row.get("r5_serve") is True)
    assert n_serve == 1202
    for it in iter_service_items():
        row = usability[it["item_id"]]
        assert row["r5_serve"] is True and row["usability_pool"] == "clean"
        break  # 全量一致性由 servable==1202 隐含,抽首行冒烟


def _run_focused():
    import inspect

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURES'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    _run_focused()
