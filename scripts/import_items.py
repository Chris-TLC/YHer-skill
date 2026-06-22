#!/usr/bin/env python3
"""
真题导入管线（总蓝图 B-1，离线运行，引擎层不依赖它）。

用途：Chris 搞到真题后，用这个脚本把原始题转成 item_bank/{topic}.jsonl。

Chris 需要提供的最小格式（每题 5 样，缺一不可）：
  1. 题干 stem（数据/图表文字化，化学式纯文本如 N2+3H2⇌2NH3）
  2. 来源 source（哪年哪卷第几题）
  3. 标准答案 final_answers（每小问）
  4. 标准解步骤 solution_steps
  5. 每小问分值 score
能多给更好：官方评分细则/得分点（= 现成 rubric，优先搞带细则的真题）。

管线步骤：
  解析 → 自动挂题型(向量) → 自动挂KG节点(向量) → LLM半自动拆rubric(★须人审)
       → 挂一化儿讲法chunk → 挂视频 → 写 jsonl

当前实现：提供"从结构化输入直接写库"的最小可用版（rubric 由人提供或 LLM 生成后人审）。
向量自动挂接、LLM 拆 rubric 在有真题批量导入需求时再接 retriever/llm_client。

用法示例：
    python3 scripts/import_items.py --validate   # 校验现有题库格式
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

ITEM_BANK_DIR = SKILL_DIR / "data" / "item_bank"

REQUIRED_FIELDS = ["item_id", "stem", "kg_nodes", "standard_solution", "rubric"]
REQUIRED_RUBRIC_FIELDS = ["point_id", "desc", "keywords", "score"]


def validate_item(item: dict) -> list:
    """校验一道题是否符合 schema，返回问题列表（空=合格）。"""
    problems = []
    for f in REQUIRED_FIELDS:
        if f not in item or item[f] in (None, "", [], {}):
            problems.append(f"缺字段: {f}")
    sol = item.get("standard_solution", {})
    if not sol.get("final_answers"):
        problems.append("standard_solution 缺 final_answers")
    if not sol.get("solution_steps"):
        problems.append("standard_solution 缺 solution_steps")
    for i, rp in enumerate(item.get("rubric", [])):
        for rf in REQUIRED_RUBRIC_FIELDS:
            if rf not in rp:
                problems.append(f"rubric[{i}] 缺 {rf}")
    if not any(rp.get("must_have") for rp in item.get("rubric", [])):
        problems.append("rubric 里没有任何 must_have 得分点（建议至少 1 个）")
    return problems


def validate_bank() -> int:
    """校验整个题库。返回不合格题数。"""
    if not ITEM_BANK_DIR.exists():
        print("题库目录不存在:", ITEM_BANK_DIR)
        return 0
    total, bad = 0, 0
    for f in sorted(ITEM_BANK_DIR.glob("*.jsonl")):
        print(f"\n=== {f.name} ===")
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            total += 1
            try:
                item = json.loads(line)
            except Exception as e:
                print(f"  行{ln} JSON解析失败: {e}")
                bad += 1
                continue
            problems = validate_item(item)
            if problems:
                bad += 1
                print(f"  ✗ {item.get('item_id','?')}: {'; '.join(problems)}")
            else:
                print(f"  ✓ {item.get('item_id')} ({len(item.get('rubric',[]))} 得分点)")
    print(f"\n合计 {total} 题，{bad} 题有问题，{total-bad} 题合格。")
    return bad


def main():
    parser = argparse.ArgumentParser(description="真题导入管线")
    parser.add_argument("--validate", action="store_true", help="校验现有题库格式")
    args = parser.parse_args()

    if args.validate:
        sys.exit(1 if validate_bank() else 0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
