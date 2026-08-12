#!/usr/bin/env python3
"""WS4 块渲染器:v4 item → RIR(渲染中间表示)。纯函数层,渲染永不抛异常。

RIR 节点类型(前端按 kind 分派):
  {"kind":"text",   "text": str}
  {"kind":"latex",  "latex": str, "source": "math_omml"|"formula_asset"}
  {"kind":"image",  "url": str, "asset_hash": str|None, "alt": str|None}
  {"kind":"table",  "rows": [[ [node,...], ... ]]}          # cell = 节点列表
  {"kind":"placeholder", "reason": str}                     # 降级占位(图暂缺等)

规则(WS4_RENDER_PLAN_2026-07-04.md §4a-1):
  math_omml  → OMML 预转缓存(katex_ok 才发 latex)→ 否则 omml 原文占位(前端 MathML 兜底)
  formula    → ref_map → 转写表 latex(R4)→ 否则资产图 → 否则占位
  figure     → 资产图 + alt=transcript.summary(R2:display_only 只给概要)→ 降级占位(R3)
  table      → 结构化 cell 递归
任何异常都折叠为 placeholder 节点并计入 item 级 degrade_reasons —— 渲染层永不 500。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.data.item_bank_v4 import solution_answers
from core.data.ws2_transcripts import (
    WS2_TRANSCRIPTS,  # noqa: F401  (路径存在性由测试保证)
    load_transcripts, load_media_ref_map, renderable_latex, display_degraded,
    resolve_image_path,
)

_V4_DIR = Path(__file__).parent.parent.parent / "data" / "item_bank" / "v4"
OMML_LATEX_CACHE = _V4_DIR / "ws2_omml_latex_cache_v1.jsonl"
_ASSET_MANIFEST = Path(__file__).parent.parent.parent / "data" / "ws2_assets_v1_candidate_20260703" / "asset_manifest.jsonl"
_WS1_ASSET_ROOT = Path(__file__).parent.parent.parent / "data" / "ws1_batch_v4_20260703"

ASSET_URL_PREFIX = "/api/v4/assets/"       # normalized/修复件图源(按 asset_hash)
RAW_ASSET_URL_PREFIX = "/api/v4/raw_assets"  # 原始 png/jpeg 直显(按 group+media)

_RAW_DIRECT_EXT = {".png", ".jpeg", ".jpg"}


def _norm_group_name(s: str) -> str:
    return re.sub(r"[,，+＋\s_、()（）~～\-—–]", "", s)

_ZONES = (("stem", "stem_blocks"), ("answer", "answer_blocks_effective"), ("analysis", "analysis_blocks"))


class _Ctx:
    """进程级只读缓存(懒加载一次)。"""

    _inst: Optional["_Ctx"] = None

    def __init__(self) -> None:
        self.transcripts = load_transcripts()
        self.ref_map = load_media_ref_map()
        self.omml_cache: Dict[str, Dict[str, Any]] = {}
        if OMML_LATEX_CACHE.exists():
            with OMML_LATEX_CACHE.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.omml_cache[r.get("omml_sha1", "")] = r
        # group_key(规范化) → ws1 assets 目录(apply 时实证过的模糊匹配口径)
        self.group_dirs: Dict[str, Path] = {}
        if _WS1_ASSET_ROOT.exists():
            for gdir in _WS1_ASSET_ROOT.iterdir():
                assets = gdir / "assets"
                if assets.is_dir():
                    base = re.sub(r"_[0-9a-f]{10}$", "", gdir.name)
                    self.group_dirs[_norm_group_name(base)] = assets
        # 资产尺寸(QA-0:image 节点带 w/h,前端禁放大;formula 类小图行内化)
        self.asset_dims: Dict[str, tuple] = {}
        if _ASSET_MANIFEST.exists():
            with _ASSET_MANIFEST.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    d = r.get("dimensions")
                    if isinstance(d, list) and len(d) == 2:
                        self.asset_dims[r.get("asset_hash", "")] = (int(d[0]), int(d[1]), r.get("asset_class", ""))

    def raw_asset_path(self, group_key: str, media: str) -> Optional[Path]:
        """原始资产文件解析(png/jpeg 直显回退用),路径安全:只在 ws1 资产目录内。"""
        assets = self.group_dirs.get(_norm_group_name(group_key or ""))
        if assets is None or "/" in media or ".." in media:
            return None
        p = assets / media
        return p if p.exists() else None

    @classmethod
    def get(cls) -> "_Ctx":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst


def _text_node(t: Any) -> Dict[str, Any]:
    return {"kind": "text", "text": str(t)}


def _placeholder(reason: str, degrade_reasons: List[str]) -> Dict[str, Any]:
    degrade_reasons.append(reason)
    return {"kind": "placeholder", "reason": reason}


def _omml_node(seg: Dict[str, Any], ctx: _Ctx, degrade_reasons: List[str]) -> Dict[str, Any]:
    omml = seg.get("omml") or ""
    if seg.get("latex"):  # batch 7 起附带 latex 的块直接用
        return {"kind": "latex", "latex": seg["latex"], "source": "math_omml"}
    key = hashlib.sha1(omml.encode()).hexdigest()
    hit = ctx.omml_cache.get(key)
    if hit and hit.get("katex_ok") and hit.get("latex"):
        return {"kind": "latex", "latex": hit["latex"], "source": "math_omml"}
    # 转换失败/编译失败:占位(前端可尝试 MathML,数据无损保留在 v4 行内)
    return _placeholder("omml_unconvertible", degrade_reasons)


def _image_node(url: str, asset_hash: Optional[str], alt: Optional[str], ctx: _Ctx,
                seg_type: str = "figure") -> Dict[str, Any]:
    """image 节点统一构造:带原始尺寸(前端禁放大)与行内标记(formula 小图不独占行)。"""
    node: Dict[str, Any] = {"kind": "image", "url": url, "asset_hash": asset_hash, "alt": alt}
    dims = ctx.asset_dims.get(asset_hash or "")
    if dims:
        node["w"], node["h"] = dims[0], dims[1]
        node["inline"] = seg_type == "formula" or (dims[1] <= 60 and dims[0] <= 300)
    else:
        node["inline"] = seg_type == "formula"
    return node


def _raw_fallback_node(group_key: str, media: str, ctx: _Ctx,
                       degrade_reasons: List[str], reason: str,
                       seg_type: str = "figure") -> Dict[str, Any]:
    """无转写/无 normalized 资产的兜底:原始 png/jpeg 直显,其余占位。"""
    from urllib.parse import quote
    p = ctx.raw_asset_path(group_key, media)
    if p is not None and p.suffix.lower() in _RAW_DIRECT_EXT:
        url = f"{RAW_ASSET_URL_PREFIX}?group_key={quote(group_key)}&media={quote(media)}"
        return {"kind": "image", "url": url, "asset_hash": None, "alt": None,
                "inline": seg_type == "formula"}
    return _placeholder(reason, degrade_reasons)


def _asset_node(seg: Dict[str, Any], group_key: str, ctx: _Ctx,
                degrade_reasons: List[str]) -> Dict[str, Any]:
    media = seg.get("media") or ""
    seg_type = seg.get("type") or "figure"
    h = ctx.ref_map.get((group_key, media))
    if h is None and media.startswith("ans_"):
        h = ctx.ref_map.get((group_key, re.sub(r"^ans_[0-9a-f]{8}_", "", media)))
    if h is None:
        return _raw_fallback_node(group_key, media, ctx, degrade_reasons, f"media_unmapped:{media}", seg_type)
    row = ctx.transcripts.get(h)
    if row is None:
        # 有映射无转写(答案/解析区资产):normalized → 原始 png/jpeg → 占位
        p = resolve_image_path(h)
        if p is not None:
            return _image_node(f"{ASSET_URL_PREFIX}{h}.png", h, None, ctx, seg_type)
        return _raw_fallback_node(group_key, media, ctx, degrade_reasons, f"asset_no_transcript:{media}", seg_type)
    if seg.get("type") == "formula":
        latex = renderable_latex(row)
        if latex:
            return {"kind": "latex", "latex": latex, "source": "formula_asset"}
    if display_degraded(row):
        ft = row.get("fine_type")
        if ft == "icon_or_noise":
            return _text_node("")  # 噪声资产(空文本框/标点碎片):静默,不占位不降级
        if ft == "broken_image" or row.get("pool") == "leak_rejected":
            return _placeholder(f"asset_degraded:{row.get('pool')}", degrade_reasons)
        # manual_queue 的普通图:transcript 不可信但图本身可看,直显原图(无 alt)
        p = resolve_image_path(h)
        if p is not None:
            return _image_node(f"{ASSET_URL_PREFIX}{h}.png", h, None, ctx, seg_type)
        return _placeholder(f"asset_degraded:{row.get('pool')}", degrade_reasons)
    p = resolve_image_path(h)
    if p is None:
        return _placeholder(f"asset_image_missing:{media}", degrade_reasons)
    alt = None
    tr = row.get("transcript") or {}
    if tr.get("summary"):
        alt = str(tr["summary"])[:200]  # R2:任何池都只把概要作为 alt,细节不经渲染层外发
    return _image_node(f"{ASSET_URL_PREFIX}{h}.png", h, alt, ctx, seg_type)


def _cell_nodes(cell: Any, group_key: str, ctx: _Ctx, degrade_reasons: List[str]) -> List[Dict[str, Any]]:
    if isinstance(cell, str):
        return [_text_node(cell)]
    if isinstance(cell, dict):
        return [_seg_node(cell, group_key, ctx, degrade_reasons)]
    if isinstance(cell, list):
        out: List[Dict[str, Any]] = []
        for c in cell:
            out.extend(_cell_nodes(c, group_key, ctx, degrade_reasons))
        return out
    return [_text_node(cell)]


def _seg_node(seg: Dict[str, Any], group_key: str, ctx: _Ctx, degrade_reasons: List[str]) -> Dict[str, Any]:
    try:
        t = seg.get("type")
        if t == "text":
            return _text_node(seg.get("text", ""))
        if t == "math_omml":
            return _omml_node(seg, ctx, degrade_reasons)
        if t in ("formula", "figure"):
            return _asset_node(seg, group_key, ctx, degrade_reasons)
        if t == "table":
            rows = []
            for row in seg.get("rows") or []:
                rows.append([_cell_nodes(c, group_key, ctx, degrade_reasons) for c in row])
            return {"kind": "table", "rows": rows}
        return _placeholder(f"unknown_block_type:{t}", degrade_reasons)
    except Exception as e:  # 永不 500
        return _placeholder(f"render_error:{type(e).__name__}", degrade_reasons)


def item_to_rir(item: Dict[str, Any], zones: tuple = ("stem", "answer")) -> Dict[str, Any]:
    """v4 item → RIR。zones 默认题干+答案(analysis 按需)。"""
    ctx = _Ctx.get()
    gk = item.get("group_key") or ""
    degrade_reasons: List[str] = []
    out_zones: Dict[str, List[List[Dict[str, Any]]]] = {}
    for zone, field in _ZONES:
        if zone not in zones:
            continue
        paras: List[List[Dict[str, Any]]] = []
        for para in item.get(field) or []:
            segs = para.get("para", []) if isinstance(para, dict) else para
            paras.append([_seg_node(s, gk, ctx, degrade_reasons) for s in segs if isinstance(s, dict)])
        # QA-0:答案区渲染为空但 standard_solution 有答案 → 回退注入(接 loader 口径,225 题复活)
        if zone == "answer" and not _zone_has_visible(paras):
            answers = solution_answers(item)
            if answers:
                paras = [[{"kind": "text", "text": "、".join(answers), "source": "standard_solution"}]]
        out_zones[zone] = paras
    return {
        "item_id": item.get("item_id"),
        "zones": out_zones,
        "degraded": bool(degrade_reasons),
        "degrade_reasons": degrade_reasons,
        "rir_version": "ws4_rir_v1",
    }


def _zone_has_visible(paras: List[List[Dict[str, Any]]]) -> bool:
    """区内是否有可见内容(非空 text / 图 / 公式 / 表格;占位不算)。"""
    for para in paras:
        for n in para:
            k = n.get("kind")
            if k == "text" and str(n.get("text", "")).strip():
                return True
            if k in ("latex", "image", "table"):
                return True
    return False
