#!/usr/bin/env python3
"""
真题库仓库（总蓝图 B-1）。

这是"正确性层"：每道真题挂 [标准解 + 评分rubric + 关联一化儿讲法chunk + BV/P视频]。
教学引擎拿标准解当"讲什么对"的锚点，根治讲解太表面；
诊断引擎拿 rubric 当"客观打分依据"，根治诊断不准。

题库文件：data/item_bank/{topic}.jsonl，一行一题。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ITEM_BANK_DIR = Path(__file__).parent.parent.parent / "data" / "item_bank"


class ItemRepository:
    """读取真题库。允许题库为空（阶段一初期还没真题，引擎退化到 KG 判据）。"""

    def __init__(self, bank_dir: Optional[Path] = None):
        self.bank_dir = Path(bank_dir) if bank_dir else ITEM_BANK_DIR
        self._items: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.bank_dir.exists():
            for f in self.bank_dir.glob("*.jsonl"):
                with open(f, encoding="utf-8") as fp:
                    for line in fp:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if item.get("item_id"):
                            self._items[item["item_id"]] = item
        self._loaded = True

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        return self._items.get(item_id)

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._items)

    def find_items(
        self,
        kg_node: Optional[str] = None,
        question_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        region: Optional[str] = None,
        exclude_ids: Optional[Set[str]] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """按知识点/题型/难度/卷别筛选真题。"""
        self._ensure_loaded()
        exclude_ids = exclude_ids or set()
        out = []
        for iid, item in self._items.items():
            if iid in exclude_ids:
                continue
            if kg_node and kg_node not in item.get("kg_nodes", []):
                continue
            if question_type and item.get("question_type") != question_type:
                continue
            if difficulty and item.get("difficulty") != difficulty:
                continue
            if region and item.get("region") != region:
                continue
            out.append(item)
            if len(out) >= limit:
                break
        return out

    def get_rubric(self, item_id: str) -> List[Dict[str, Any]]:
        item = self.get_item(item_id)
        return item.get("rubric", []) if item else []

    def get_standard_solution(self, item_id: str) -> Optional[Dict[str, Any]]:
        item = self.get_item(item_id)
        return item.get("standard_solution") if item else None

    def has_numeric_answer(self, item_id: str) -> bool:
        """判断是否计算/有标准答案题（决定 mastery 客观权重）。"""
        item = self.get_item(item_id)
        if not item:
            return False
        sol = item.get("standard_solution", {})
        answers = " ".join(str(a) for a in sol.get("final_answers", []))
        return any(ch.isdigit() for ch in answers)


_repo: Optional[ItemRepository] = None


def get_item_repository() -> ItemRepository:
    global _repo
    if _repo is None:
        _repo = ItemRepository()
    return _repo
