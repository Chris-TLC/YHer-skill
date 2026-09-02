#!/usr/bin/env python3
"""G5 验收抽样器（2026-07-06，D2-A 决议 Track A①）。

从 R5 白名单（当前 1207）按 知识节点 × 题型 分层抽样 ~120 题，
生成给用户肉眼验收的清单（markdown，含预览深链）。

- 分层：每个覆盖节点至少 1 题；名额按节点池大小加权；choice/非choice 分桶。
- 确定性：固定 seed，重复运行同一清单（验收中途可续）。
- 只读官方数据；产物写 /tmp/yher_g5/。

用法：python3 scripts/make_g5_sample.py [--n 120] [--out-dir /tmp/yher_g5]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from core.data.item_bank_v4 import iter_service_items  # noqa: E402

PREVIEW = "http://127.0.0.1:8822/v4_preview.html?v=g5"


def item_kind(it: dict) -> str:
    txt = json.dumps(it.get("stem_blocks") or [], ensure_ascii=False)
    return "choice" if ('"A.' in txt or '"A．' in txt or "A. " in txt) else "free"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out-dir", default="/tmp/yher_g5")
    args = ap.parse_args()

    items = list(iter_service_items())  # R5 默认口径 = 学生所见
    by_node: dict[str, list[dict]] = collections.defaultdict(list)
    for it in items:
        for n in (it.get("kg_nodes") or ["(无节点)"]):
            by_node[n].append(it)

    rng = random.Random(20260706)
    total_pool = sum(len(v) for v in by_node.values())
    picked: dict[str, dict] = {}
    # 第一轮：每节点保底 1 题
    for node, pool in sorted(by_node.items()):
        it = rng.choice(pool)
        picked.setdefault(it["item_id"], it)
    # 第二轮：按节点权重补齐到 n，choice/free 尽量均衡
    weights = sorted(by_node.items(), key=lambda kv: -len(kv[1]))
    wi = 0
    while len(picked) < args.n and wi < 10000:
        node, pool = weights[wi % len(weights)]
        kinds = collections.Counter(item_kind(x) for x in picked.values())
        want = "free" if kinds["free"] <= kinds["choice"] else "choice"
        cands = [x for x in pool if x["item_id"] not in picked and item_kind(x) == want] or \
                [x for x in pool if x["item_id"] not in picked]
        if cands:
            it = rng.choice(cands)
            picked[it["item_id"]] = it
        wi += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(picked.values(), key=lambda x: (sorted(x.get("kg_nodes") or ["~"])[0], x["item_id"]))

    md = [
        "# G5 验收清单（R5 白名单分层抽样）",
        "",
        f"共 {len(rows)} 题 / 覆盖 {len(by_node)} 节点（白名单 {len(items)} 题池）。"
        "先起预览：launch.json 的 yher-api-v4（8822 端口），再逐条点链接。",
        "",
        "**判定口径（对每题勾一格）**：`OK` = 题面完整可作答 + 答案对得上题目且完整；"
        "`BAD-题面` / `BAD-答案` / `BAD-渲染` = 任一硬伤（记 item_id 前 12 位即可）；`?` = 拿不准。",
        "",
        "| # | 节点 | 型 | 预览 | 判定 |",
        "|---|---|---|---|---|",
    ]
    for i, it in enumerate(rows, 1):
        node = "、".join((it.get("kg_nodes") or ["-"])[:2])
        kind = "选" if item_kind(it) == "choice" else "答"
        iid = it["item_id"]
        md.append(f"| {i} | {node[:14]} | {kind} | [{iid[:12]}]({PREVIEW}#{iid}) | ☐ |")
    md += [
        "",
        "> 放行判据（D2-A 已批）：抽样零硬伤漏报即 G5 过；发现 BAD 记 id 交 Claude 归因，",
        "> 不足以推翻整门（R5 宁缺勿滥仍在线上兜底），除非 BAD 率 >3%。",
    ]
    (out / "G5_SAMPLE.md").write_text("\n".join(md), encoding="utf-8")
    (out / "g5_sample_ids.json").write_text(
        json.dumps([r["item_id"] for r in rows], indent=1), encoding="utf-8")

    kinds = collections.Counter(item_kind(x) for x in rows)
    print(json.dumps({
        "sampled": len(rows), "nodes_covered": len({n for r in rows for n in (r.get("kg_nodes") or [])}),
        "choice": kinds["choice"], "free": kinds["free"],
        "out": str(out / "G5_SAMPLE.md"),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
