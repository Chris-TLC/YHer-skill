#!/usr/bin/env python3
"""G5 acceptance sampler (2026-07-06, D2-A decision Track A).

Stratified-samples ~120 items from the R5 whitelist (currently 1207) by knowledge node x item type,
and generates a checklist for the user's eyeball acceptance (markdown with deep preview links).

- Stratification: each covered node gets at least 1 item; remaining slots weighted by node pool size; choice/free bucketed.
- Deterministic: fixed seed, same checklist on re-run (acceptance can be resumed mid-way).
- Read-only on official data; output written to /tmp/yher_g5/.

Usage: python3 scripts/make_g5_sample.py [--n 120] [--out-dir /tmp/yher_g5]
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

    items = list(iter_service_items())  # R5 default scope = what students see
    by_node: dict[str, list[dict]] = collections.defaultdict(list)
    for it in items:
        for n in (it.get("kg_nodes") or ["(no node)"]):
            by_node[n].append(it)

    rng = random.Random(20260706)
    total_pool = sum(len(v) for v in by_node.values())
    picked: dict[str, dict] = {}
    # Round 1: at least 1 item per node
    for node, pool in sorted(by_node.items()):
        it = rng.choice(pool)
        picked.setdefault(it["item_id"], it)
    # Round 2: fill up to n by node weight, balancing choice/free as much as possible
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
        "# G5 Acceptance Checklist (R5 whitelist stratified sample)",
        "",
        f"{len(rows)} items / {len(by_node)} nodes covered (whitelist pool: {len(items)} items)."
        "Start the preview first: launch.json's yher-api-v4 (port 8822), then click through the links.",
        "",
        "**Judgment rubric (tick one box per item)**: `OK` = stem complete and answerable + answer matches the stem and is complete;"
        "`BAD-stem` / `BAD-answer` / `BAD-render` = any hard defect (just note the first 12 chars of item_id); `?` = unsure.",
        "",
        "| # | Node | Type | Preview | Verdict |",
        "|---|---|---|---|---|",
    ]
    for i, it in enumerate(rows, 1):
        node = "、".join((it.get("kg_nodes") or ["-"])[:2])
        kind = "choice" if item_kind(it) == "choice" else "free"
        iid = it["item_id"]
        md.append(f"| {i} | {node[:14]} | {kind} | [{iid[:12]}]({PREVIEW}#{iid}) | ☐ |")
    md += [
        "",
        "> Release criterion (approved in D2-A): G5 passes if the sample reports zero missed hard defects;"
        "> on BAD, record the id and hand it to Claude for root-cause; one defect does not overturn the whole set (R5's 'rather fewer than faulty' line stays online as fallback), unless the BAD rate exceeds 3%.",
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
