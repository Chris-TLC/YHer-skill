#!/usr/bin/env python3
"""WS2 图形转写官方表的唯一合法读取入口(2026-07-04 apply,用户授权「授权ws2并库」)。

数据文件(与 v3/v4 题库并存,零修改既有文件):
- ws2_asset_transcripts_v1.jsonl : 6005 资产 × (latex | 结构化 transcript) + 分池
- ws2_media_ref_map_v1.jsonl     : (group_key, media) → asset_hash(题干区全覆盖;
                                   答案/解析区 4070 条映射存在但无转写,zones 字段标明)
- ws2_repaired_assets/           : 52 张回源重渲的 PNG(其余图源在
                                   data/ws2_assets_v1_candidate_20260703/normalized/,冻结)

硬规则(Batch 8/8.1 审计 + G2 门结论):
  R1  pool=="ai_seed" 才可作为 AI 内化/诊断引用的 transcript 来源。
  R2  pool=="display_only" 只允许前端展示原图,AI 不得引用其 transcript 细节。
  R3  pool in {"manual_queue","leak_rejected"} 一律不得服务(含 171 张 broken 源空资产,
      渲染端对其引用题走"图缺失降级")。
  R4  latex 仅在 latex_status=="passed" 时可交 KaTeX 渲染;latex_consistency 为 False 的
      410 条在 AI 种子使用时降权(渲染不受限)。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

_V4_DIR = Path(__file__).parent.parent.parent / "data" / "item_bank" / "v4"
WS2_TRANSCRIPTS = _V4_DIR / "ws2_asset_transcripts_v1.jsonl"
WS2_MEDIA_REF_MAP = _V4_DIR / "ws2_media_ref_map_v1.jsonl"
WS2_REPAIRED_DIR = _V4_DIR / "ws2_repaired_assets"
WS2_IMAGE_BASE = Path(__file__).parent.parent.parent / "data" / "ws2_assets_v1_candidate_20260703" / "normalized"

AI_SEED = "ai_seed"
DISPLAY_ONLY = "display_only"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


def load_transcripts() -> Dict[str, Dict[str, Any]]:
    """全量资产转写表,按 asset_hash 索引。审计/工程用;服务侧用下面的口径函数。"""
    return {r["asset_hash"]: r for r in _iter_jsonl(WS2_TRANSCRIPTS) if r.get("asset_hash")}


def load_media_ref_map() -> Dict[tuple, str]:
    """(group_key, media) → asset_hash。含答案/解析区映射(无转写,查 zones)。"""
    return {
        (r["group_key"], r["media"]): r["asset_hash"]
        for r in _iter_jsonl(WS2_MEDIA_REF_MAP)
        if r.get("group_key") and r.get("media")
    }


def renderable_latex(row: Dict[str, Any]) -> Optional[str]:
    """R4:可渲染 latex,否则 None(渲染端回退图片)。"""
    if row.get("latex") and row.get("latex_status") == "passed":
        return row["latex"]
    return None


def ai_citable_transcript(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """R1/R2/R3:只有 ai_seed 池返回 transcript,其余 None。"""
    if row.get("pool") == AI_SEED and row.get("transcript"):
        return row["transcript"]
    return None


def display_degraded(row: Dict[str, Any]) -> bool:
    """R3:该资产是否必须走'图缺失降级'(broken 源空/泄漏/人工队列)。"""
    return row.get("pool") in ("manual_queue", "leak_rejected")


def resolve_image_path(asset_hash: str) -> Optional[Path]:
    """图源解析:修复件优先,其次冻结的 normalized 库。"""
    for base in (WS2_REPAIRED_DIR, WS2_IMAGE_BASE):
        p = base / f"{asset_hash}.png"
        if p.exists():
            return p
    return None


def stats() -> Dict[str, Any]:
    pools: Dict[str, int] = {}
    latex_ok = trans = 0
    total = 0
    for r in _iter_jsonl(WS2_TRANSCRIPTS):
        total += 1
        pools[r.get("pool", "?")] = pools.get(r.get("pool", "?"), 0) + 1
        if renderable_latex(r):
            latex_ok += 1
        if r.get("transcript"):
            trans += 1
    return {"total": total, "pools": pools, "renderable_latex": latex_ok, "with_transcript": trans}
