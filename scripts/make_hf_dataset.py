#!/usr/bin/env python3
"""构建公开数据集(HF 镜像 + 仓库内样例),确定性 seed 可复现。

用法:
  python3 scripts/make_hf_dataset.py --sample-only            # 只生成 samples/(55 题)
  python3 scripts/make_hf_dataset.py --build samples          # 生成 samples/ + hf_export/
  python3 scripts/make_hf_dataset.py --push Chris-TLC/yher-chemistry-question-bank   # 需要 HF token

字段口径与 data/README.md 一致。样例抽取规则:
  sha256(item_id) 前 4 位为偶数 → 进样例(约 50%),再加 R5 serve 过滤,直到 55 题。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V4 = REPO / "data" / "item_bank" / "v4" / "chemistry_v4_3329.jsonl"
R5 = REPO / "data" / "item_bank" / "v4" / "usability_r5_v1.jsonl"
KG = REPO / "data" / "knowledge_graph_150_enriched.jsonl"
SAMPLE_DIR = REPO / "data" / "samples"
HF_EXPORT = REPO / "data" / "hf_export"
SAMPLE_TARGET = 55
SAMPLE_SEED_TAG = "yher-public-v1"          # 变更任何一项→解析该值变化,样例即变,不要改


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _hash(item_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED_TAG}|{item_id}".encode()).hexdigest()


def r5_lookup() -> dict[str, dict]:
    return {row["item_id"]: row for row in load_jsonl(R5)}


def iter_v4():
    yield from load_jsonl(V4)


def build_records(r5: dict[str, dict]):
    """生成记录字典 (item_id → {题对象 + r5 状态})。"""
    for row in iter_v4():
        rec = dict(row)
        rec["r5"] = r5.get(row["item_id"], {})
        yield rec


def pick_sample(records, r5: dict[str, dict]) -> list[dict]:
    serve = [r for r in records if r["r5"].get("r5_serve")]
    chosen = [r for r in serve if int(_hash(r["item_id"])[:2], 16) % 2 == 0][:SAMPLE_TARGET]
    if len(chosen) < SAMPLE_TARGET:                      # 不足则放宽条件补足
        rest = [r for r in serve if r not in chosen]
        chosen += rest[: SAMPLE_TARGET - len(chosen)]
    return chosen


def write_samples():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    r5 = r5_lookup()
    records = list(build_records(r5))
    sample = pick_sample(records, r5)
    with (SAMPLE_DIR / "sample_55.jsonl").open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (SAMPLE_DIR / "README.md").open("w", encoding="utf-8") as f:
        f.write(f"""# Sample (55 items)

从 R5 白名单(1202 题)抽取的可读样例,每个 item 完整包含题面/解析/答案/评分细则/KG 节点关联。
抽取规则固定:`yher-public-v1|item_id` 的 sha256 前两位十六进制为偶数;50% 命中率直到满 55。

字段口径:data/README.md。全量数据见本仓库 `data/item_bank/` 与 Hugging Face 数据集镜像。
""")
    print(f"samples written: {len(sample)} -> {SAMPLE_DIR}")


def build_export():
    HF_EXPORT.mkdir(parents=True, exist_ok=True)
    r5 = r5_lookup()
    kg = list(load_jsonl(KG))
    out = []
    for r in build_records(r5):
        # 平整化一点:保留原始对象,附加 r5_serve/r5_reason
        r["r5_serve"] = r["r5"].get("r5_serve", False)
        r["r5_reason"] = r["r5"].get("r5_block_reason", "")
        del r["r5"]
        out.append(r)
    with (HF_EXPORT / "chemistry_items_v4.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (HF_EXPORT / "knowledge_graph.jsonl").open("w", encoding="utf-8") as f:
        for r in kg:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"export written: {len(out)} items + {len(kg)} KG nodes -> {HF_EXPORT}")


def push_hf(dataset_id: str):
    # Push as two separate configs (items_v4 / knowledge_graph), NOT as two splits of
    # one config: the two sources have different schemas, and the HF viewer cannot
    # render mixed-schema splits (it fails with CastError).
    from datasets import Dataset
    items = [json.loads(line) for line in (HF_EXPORT / "chemistry_items_v4.jsonl").open(encoding="utf-8")]
    kg = [json.loads(line) for line in (HF_EXPORT / "knowledge_graph.jsonl").open(encoding="utf-8")]
    ds = Dataset.from_list(items)
    ds.push_to_hub(dataset_id, config_name="items_v4")
    kds = Dataset.from_list(kg)
    kds.push_to_hub(dataset_id, config_name="knowledge_graph")
    print(f"pushed to hf: {dataset_id} (configs: items_v4, knowledge_graph)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", metavar="dataset-id")
    args = ap.parse_args()
    write_samples()
    if args.push:
        build_export()
        push_hf(args.push)
    elif args.build or args.sample_only:
        if args.build:
            build_export()


if __name__ == "__main__":
    main()
