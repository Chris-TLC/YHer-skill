#!/usr/bin/env python3
"""Schema v4 题库(WS3 apply,2026-07-03)的唯一合法读取入口。

v4 manifest 与 v3 并存:
- v3 服务路径(core/data/item_repository.py)glob 的是 data/item_bank/*.jsonl(非递归),
  永远看不到本目录 —— 切换到 v4 必须是显式改代码,不存在"悄悄混入"。
- 本模块是 v4 的规则级防线。任何学生端展示、练习、诊断、AI 内化种子,
  只允许通过 iter_service_items() / load_service_pool() 取题,不得绕过直接读 jsonl。

硬规则(交接文档 WS3 apply 条款 + 2026-07-03 apply 前审计结论):
  R1  只有 pool == "main" 且 service_eligible == true 的题可进服务池。
  R2  quality_flags 含 "answer_type_mismatch" 的题,答案视为不存在(即使数据里有也不采信),
      且不可服务 —— 这是对残留疑似错配的规则级兜底,优先于逐条排查。
  R3  全链路无答案的题不可服务:answer_blocks_effective 为空、standard_solution 也给不出
      答案的题挡在池外(apply 前审计实测 main 池有 138 道此类漏网题)。
  R4  service_exclusions.jsonl 显式排除清单必须尊重(7 道题干实为解析文本的错切题 +
      4 道答案片段/纯答案文档误抽题,均经人工逐条核验,泄漏类)。
      按治理纪律,该清单的新增只能由用户或 Claude 签字。
  R5  可用性白名单门(2026-07-06 R5 apply,用户 L1「授权,并且先上R5……直接开始授权入库」):
      只 serve usability_r5_v1.jsonl 中 r5_serve == true 的题。数量以台账动态读取为准；
      2026-07-13 坏题结案后为 1202。排除口径见
      PROJECT_HANDOFF/BATCH14_AUDIT_2026-07-06.md §五，含空心结构式、真源码、
      碎片字面量、部分答案和变式库挂起等 clean 漏报类。
      台账无记录的题一律不服务(宁缺勿滥)。审计/预览/回归需看全池时显式传
      apply_r5=False —— 学生端取题路径禁止关闭 R5。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

V4_BANK_DIR = Path(__file__).parent.parent.parent / "data" / "item_bank" / "v4"
# Batch 7 apply(2026-07-04,用户授权「授权batch7」):切到 v4.1(98 行处置,零增零删)。
# 回滚 = 指回 chemistry_v4_3329.jsonl(见 data/_backup_pre_v4_1_apply_20260704/ROLLBACK.md)。
V4_MANIFEST = V4_BANK_DIR / "chemistry_v4_1_3329.jsonl"
V4_SERVICE_EXCLUSIONS = V4_BANK_DIR / "service_exclusions.jsonl"
# R5 apply(2026-07-06,用户 L1「授权,并且先上R5……直接开始授权入库」):batch14 可用性台账。
# 回滚 = 删本文件引用/恢复备份(见 data/_backup_pre_r5_apply_20260706/ROLLBACK.md)。
V4_USABILITY_R5 = V4_BANK_DIR / "usability_r5_v1.jsonl"

ANSWER_TYPE_MISMATCH = "answer_type_mismatch"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_service_exclusions(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    return {
        row["item_id"]: row
        for row in _iter_jsonl(path or V4_SERVICE_EXCLUSIONS)
        if row.get("item_id")
    }


def load_usability_r5(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """R5:batch14 可用性台账(item_id → {usability_pool, r5_serve, r5_block_reason, ...})。"""
    return {
        row["item_id"]: row
        for row in _iter_jsonl(path or V4_USABILITY_R5)
        if row.get("item_id")
    }


def effective_answer_blocks(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """R2:错配题答案视为不存在 —— 不信任数据,规则再判一次。"""
    if ANSWER_TYPE_MISMATCH in (item.get("quality_flags") or []):
        return []
    return item.get("answer_blocks_effective") or []


def solution_answers(item: Dict[str, Any]) -> List[str]:
    if ANSWER_TYPE_MISMATCH in (item.get("quality_flags") or []):
        return []
    sol = item.get("standard_solution") or {}
    answers = [str(a).strip() for a in (sol.get("final_answers") or []) if str(a).strip()]
    standard = str(sol.get("standard_answer") or "").strip()
    if standard and standard not in answers:
        answers.append(standard)
    return answers


def has_effective_answer(item: Dict[str, Any]) -> bool:
    """R3:answer_blocks_effective 或 standard_solution 至少一处给得出答案。"""
    return bool(effective_answer_blocks(item)) or bool(solution_answers(item))


def service_blockers(
    item: Dict[str, Any],
    exclusions: Optional[Dict[str, Dict[str, Any]]] = None,
    usability: Optional[Dict[str, Dict[str, Any]]] = None,
    apply_r5: bool = True,
) -> List[str]:
    """返回该题不能进服务池的全部原因;空列表 = 可服务。

    apply_r5=False 仅供审计/预览/回归看全池(R1-R4 口径);学生端取题不得关闭。
    """
    if exclusions is None:
        exclusions = load_service_exclusions()
    blockers: List[str] = []
    if item.get("pool") != "main":
        blockers.append(f"pool_not_main:{item.get('pool')}")
    if item.get("service_eligible") is not True:
        blockers.append("not_service_eligible")
    if ANSWER_TYPE_MISMATCH in (item.get("quality_flags") or []):
        blockers.append("answer_type_mismatch")
    if not has_effective_answer(item):
        blockers.append("no_effective_answer")
    excl = exclusions.get(item.get("item_id", ""))
    if excl:
        blockers.append(f"service_exclusion:{excl.get('reason', 'listed')}")
    if apply_r5:
        if usability is None:
            usability = load_usability_r5()
        urow = usability.get(item.get("item_id", ""))
        if urow is None:
            # 台账无记录 = 未过可用性审计,宁缺勿滥。
            blockers.append("usability_missing")
        elif urow.get("r5_serve") is not True:
            blockers.append(
                f"usability_not_serveable:{urow.get('r5_block_reason') or urow.get('usability_pool')}"
            )
    return blockers


def iter_items(manifest_path: Optional[Path] = None) -> Iterator[Dict[str, Any]]:
    """全量遍历(含 legacy/excluded 池)。只用于审计、回归、数据工程,禁止用于服务。"""
    yield from _iter_jsonl(manifest_path or V4_MANIFEST)


def iter_service_items(
    manifest_path: Optional[Path] = None,
    exclusions_path: Optional[Path] = None,
    apply_r5: bool = True,
    usability_path: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """服务池遍历:学生端/练习/诊断/AI 内化种子的唯一取题入口(默认 R5 白名单口径)。

    apply_r5=False 仅供审计/预览/回归(R1-R4 口径,2526);学生端路径禁止使用。
    """
    exclusions = load_service_exclusions(exclusions_path)
    usability = load_usability_r5(usability_path) if apply_r5 else None
    for item in iter_items(manifest_path):
        if not service_blockers(item, exclusions, usability=usability, apply_r5=apply_r5):
            yield item


def load_service_pool(
    manifest_path: Optional[Path] = None,
    exclusions_path: Optional[Path] = None,
    apply_r5: bool = True,
) -> Dict[str, Dict[str, Any]]:
    return {
        it["item_id"]: it
        for it in iter_service_items(manifest_path, exclusions_path, apply_r5=apply_r5)
    }


def service_pool_stats(
    manifest_path: Optional[Path] = None,
    exclusions_path: Optional[Path] = None,
    apply_r5: bool = True,
) -> Dict[str, Any]:
    """审计用:全库分池 + 服务池规模 + 各 blocker 计数。"""
    exclusions = load_service_exclusions(exclusions_path)
    usability = load_usability_r5() if apply_r5 else None
    total = 0
    pools: Dict[str, int] = {}
    blocker_counts: Dict[str, int] = {}
    servable = 0
    for item in iter_items(manifest_path):
        total += 1
        pools[item.get("pool", "?")] = pools.get(item.get("pool", "?"), 0) + 1
        blockers = service_blockers(item, exclusions, usability=usability, apply_r5=apply_r5)
        if blockers:
            for b in blockers:
                key = b.split(":", 1)[0]
                blocker_counts[key] = blocker_counts.get(key, 0) + 1
        else:
            servable += 1
    return {
        "total": total,
        "pools": pools,
        "servable": servable,
        "blocker_counts": blocker_counts,
        "exclusion_rows": len(exclusions),
        "apply_r5": apply_r5,
        "usability_rows": len(usability) if usability is not None else None,
    }
