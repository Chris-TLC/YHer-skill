#!/usr/bin/env python3
"""Batch 13 QA-3 visual crop rescue candidate package.

This runner is L0-only. It reads official v4 data and WS1 source metadata, then
writes all deliverables under /tmp/yher_batch13_qa3 by default. It never applies
anything to official item-bank, ref-map, transcript, or repaired-asset files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TOOLS_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_batch8_ws2 import (  # noqa: E402
    build_vision_client,
    image_extrema_min,
    is_blank_image,
    parse_json_response,
    run_illustration_track,
    write_json,
    write_jsonl,
)
from scripts.run_batch10_qa1 import (  # noqa: E402
    build_media_occurrence_map,
    run_formula_rows,
    sum_cost,
    write_transcription_outputs,
)
from core.data.item_bank_v4 import iter_items, iter_service_items  # noqa: E402

OUT_ROOT = Path("/tmp/yher_batch13_qa3")
TARGETS_PATH = Path("/tmp/yher_b13_crop_targets.jsonl")
WS1_ROOT = REPO_ROOT / "data" / "ws1_batch_v4_20260703"
SOFFICE = Path("/opt/homebrew/bin/soffice")
SCHEMA_VERSION = "qa3_batch13_crop_v1"
PDF_DPI = 200

PAGE_NOT_FOUND = "page_not_found"
PAGE_AMBIGUOUS = "page_ambiguous"
LOW_CONFIDENCE = "vl_low_confidence"
READBACK_MISMATCH = "readback_mismatch"
SOURCE_NOT_FOUND = "source_not_found"
RENDER_FAILED = "render_failed"


@dataclass(frozen=True)
class SourceSelection:
    group_key: str
    role: str
    source: dict[str, Any] | None
    path: Path | None
    reason: str
    score: int = 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(*parts: Any, length: int = 16) -> str:
    text = "\n".join(str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def target_key(row: dict[str, Any]) -> str:
    return stable_id(row.get("group_key"), row.get("media"), row.get("kind"), row.get("asset_hash"))


def role_for_media(media: str) -> str:
    return "analysis" if str(media).startswith("ans_") else "question_source"


def resolve_source_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    raw = Path(raw_path)
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            (TOOLS_ROOT / raw).resolve(),
            (REPO_ROOT / raw).resolve(),
            (Path("/tmp/yher_ws1_batch_v4/_converted_docx") / raw.name).resolve(),
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_sources_by_group(root: Path = WS1_ROOT) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return out
    for path in root.glob("*/sources.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        group_key = row.get("group_key")
        if group_key:
            out[str(group_key)] = row
    return out


def _source_score(source: dict[str, Any], wanted_role: str) -> int:
    name = " ".join(str(source.get(k) or "") for k in ("name", "file_name", "stem"))
    role = str(source.get("role") or "")
    markers = int(source.get("answer_marker_count") or 0)
    score = 0
    if wanted_role == "analysis":
        if role == "analysis":
            score += 100
        if role in {"answer", "answer_key"}:
            score += 75
        if any(word in name for word in ("解析", "答案", "参考答案")):
            score += 60
        score += min(40, markers)
        if any(word in name for word in ("空白卷", "原卷版", "考试版")) and "解析" not in name:
            score -= 30
    else:
        if role == "question_source":
            score += 100
        if any(word in name for word in ("空白卷", "原卷版", "考试版")):
            score += 60
        if role == "unknown":
            score += 20
        if markers == 0:
            score += 10
        if any(word in name for word in ("解析", "答案", "参考答案")):
            score -= 90
        if role in {"analysis", "answer", "answer_key"}:
            score -= 40
    return score


def select_source_for_media(group_sources: dict[str, Any] | None, group_key: str, media: str) -> SourceSelection:
    wanted = role_for_media(media)
    if not group_sources:
        return SourceSelection(group_key, wanted, None, None, "sources_json_missing")
    sources = [s for s in group_sources.get("unique_sources") or [] if s.get("status") == "ok"]
    if not sources:
        return SourceSelection(group_key, wanted, None, None, "no_ok_sources")
    ranked = sorted(((_source_score(src, wanted), src) for src in sources), key=lambda pair: pair[0], reverse=True)
    best_score, best = ranked[0]
    path = resolve_source_path(best.get("path"))
    if not path:
        return SourceSelection(group_key, wanted, best, None, "source_path_missing", best_score)
    if wanted == "analysis" and best_score < 75:
        return SourceSelection(group_key, wanted, best, path, "analysis_role_low_confidence", best_score)
    if wanted == "question_source" and best_score < 0:
        return SourceSelection(group_key, wanted, best, path, "question_role_low_confidence", best_score)
    return SourceSelection(group_key, wanted, best, path, "ok", best_score)


def source_render_id(selection: SourceSelection) -> str:
    src = selection.source or {}
    return stable_id(selection.group_key, selection.role, src.get("sha1"), selection.path, length=16)


def norm_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or ""), flags=re.UNICODE)


def short_anchor(text: str, limit: int = 30, tail: bool = False) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned[-limit:] if tail else cleaned[:limit]


def iter_media_contexts_for_item(item: dict[str, Any], zone: str, field: str) -> Iterable[dict[str, Any]]:
    blocks = item.get(field) or []
    for block_idx, block in enumerate(blocks):
        para = block.get("para") if isinstance(block, dict) else None
        if not isinstance(para, list):
            continue
        for seg_idx, seg in enumerate(para):
            if not isinstance(seg, dict):
                continue
            if seg.get("type") not in {"formula", "figure"} or not seg.get("media"):
                continue
            left = "".join(str(s.get("text") or "") for s in para[:seg_idx] if isinstance(s, dict) and s.get("type") == "text")
            right = "".join(str(s.get("text") or "") for s in para[seg_idx + 1 :] if isinstance(s, dict) and s.get("type") == "text")
            before = short_anchor(left, tail=True)
            after = short_anchor(right)
            q_num = item.get("q_num")
            section_num = item.get("section_num")
            label = " ".join(str(v) for v in (section_num, q_num) if v not in (None, ""))
            anchor_text = " ".join(part for part in [label, before, after] if part).strip()
            if not anchor_text:
                anchor_text = short_anchor(item.get("stem_text") or "", limit=60)
            yield {
                "group_key": item.get("group_key"),
                "media": seg.get("media"),
                "block_type": seg.get("type"),
                "zone": zone,
                "item_id": item.get("item_id"),
                "q_num": q_num,
                "section_num": section_num,
                "block_path": f"{field}[{block_idx}].para[{seg_idx}]",
                "anchor_before": before,
                "anchor_after": after,
                "anchor_text": anchor_text,
                "stem_head": short_anchor(item.get("stem_text") or "", limit=80),
            }


def build_context_map(items: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    contexts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        for zone, field in (
            ("stem", "stem_blocks"),
            ("answer", "answer_blocks_effective"),
            ("analysis", "analysis_blocks"),
        ):
            for ctx in iter_media_contexts_for_item(item, zone, field):
                contexts[(str(ctx.get("group_key") or ""), str(ctx.get("media") or ""))].append(ctx)
    return contexts


def contexts_for_target(row: dict[str, Any], context_map: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    contexts = list(context_map.get((str(row.get("group_key") or ""), str(row.get("media") or "")), []))
    zones = set(row.get("zones") or [])
    if zones:
        contexts = [ctx for ctx in contexts if ctx.get("zone") in zones]
    item_ids = set(row.get("item_ids") or [])
    if item_ids:
        contexts = [ctx for ctx in contexts if ctx.get("item_id") in item_ids]
    return contexts


def context_search_terms(ctx: dict[str, Any]) -> list[str]:
    before = str(ctx.get("anchor_before") or "")
    after = str(ctx.get("anchor_after") or "")
    stem = str(ctx.get("stem_head") or "")
    raw_terms = [
        before + after,
        before,
        after,
        stem[:60],
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        normalized = norm_text(term)
        if len(normalized) >= 6 and normalized not in seen:
            seen.add(normalized)
            terms.append(normalized)
    return terms


def match_pages(page_texts: dict[int, str], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    if not contexts:
        return {"status": PAGE_NOT_FOUND, "page": None, "matches": []}
    matches: list[dict[str, Any]] = []
    for ctx_idx, ctx in enumerate(contexts):
        terms = context_search_terms(ctx)
        if not terms:
            continue
        for page_num, text in page_texts.items():
            normalized = norm_text(text)
            hit_terms = [term for term in terms if term and term in normalized]
            if not hit_terms:
                continue
            score = sum(3 if i == 0 else 1 for i, term in enumerate(terms) if term in hit_terms)
            matches.append(
                {
                    "context_index": ctx_idx,
                    "page": page_num,
                    "score": score,
                    "hit_terms": hit_terms[:4],
                    "anchor_text": ctx.get("anchor_text"),
                    "block_path": ctx.get("block_path"),
                }
            )
    if not matches:
        return {"status": PAGE_NOT_FOUND, "page": None, "matches": []}
    best_score = max(int(m["score"]) for m in matches)
    best_pages = sorted({int(m["page"]) for m in matches if int(m["score"]) == best_score})
    all_pages = sorted({int(m["page"]) for m in matches})
    if len(best_pages) == 1 and (len(all_pages) == 1 or best_score >= 3):
        return {"status": "ok", "page": best_pages[0], "matches": matches, "best_score": best_score}
    return {"status": PAGE_AMBIGUOUS, "page": None, "matches": matches, "best_score": best_score, "pages": all_pages}


def run_command(args: list[str], timeout: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or "timeout"
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def render_source(selection: SourceSelection, out_root: Path) -> dict[str, Any]:
    doc_id = source_render_id(selection)
    page_dir = out_root / "pages" / doc_id
    text_dir = out_root / "page_text" / doc_id
    pdf_dir = out_root / "pdfs" / doc_id
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_id,
        "group_key": selection.group_key,
        "source_role": selection.role,
        "selection_reason": selection.reason,
        "selection_score": selection.score,
        "source_path": str(selection.path) if selection.path else "",
        "status": "pending",
        "page_dir": str(page_dir),
        "pdf_dir": str(pdf_dir),
    }
    if not selection.path or selection.reason.endswith("missing"):
        ledger.update({"status": "source_missing", "reason": selection.reason})
        return ledger
    if not SOFFICE.exists():
        ledger.update({"status": "render_failed", "reason": f"soffice_missing:{SOFFICE}"})
        return ledger
    pdf_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    existing_pages = sorted(page_dir.glob("page-*.png"))
    existing_pdf = next(iter(pdf_dir.glob("*.pdf")), None)
    if existing_pages and existing_pdf:
        ledger.update(
            {
                "status": "ok",
                "pdf_path": str(existing_pdf),
                "page_count": len(existing_pages),
                "cached": True,
            }
        )
        ensure_page_texts(existing_pdf, len(existing_pages), text_dir)
        return ledger

    code, stdout, stderr = run_command(
        [str(SOFFICE), "--headless", "--convert-to", "pdf", "--outdir", str(pdf_dir), str(selection.path)],
        timeout=240,
    )
    pdf = pdf_dir / f"{selection.path.stem}.pdf"
    if code != 0 or not pdf.exists():
        detail = " ".join((stdout, stderr)).strip().replace("\n", " ")[:500]
        ledger.update({"status": "render_failed", "stage": "docx_to_pdf", "returncode": code, "reason": detail})
        return ledger
    code, stdout, stderr = run_command(["pdftoppm", "-r", str(PDF_DPI), "-png", str(pdf), str(page_dir / "page")], timeout=300)
    pages = sorted(page_dir.glob("page-*.png"))
    if code != 0 or not pages:
        detail = " ".join((stdout, stderr)).strip().replace("\n", " ")[:500]
        ledger.update({"status": "render_failed", "stage": "pdf_to_png", "returncode": code, "reason": detail, "pdf_path": str(pdf)})
        return ledger
    ensure_page_texts(pdf, len(pages), text_dir)
    ledger.update({"status": "ok", "pdf_path": str(pdf), "page_count": len(pages), "cached": False})
    return ledger


def ensure_page_texts(pdf: Path, page_count: int, text_dir: Path) -> None:
    text_dir.mkdir(parents=True, exist_ok=True)
    for page_num in range(1, page_count + 1):
        target = text_dir / f"page-{page_num}.txt"
        if target.exists():
            continue
        code, stdout, stderr = run_command(
            ["pdftotext", "-f", str(page_num), "-l", str(page_num), "-layout", str(pdf), "-"],
            timeout=30,
        )
        target.write_text(stdout if code == 0 else f"__PDFTOTEXT_ERROR__ {stderr}", encoding="utf-8")


def load_page_texts(render_row: dict[str, Any]) -> dict[int, str]:
    text_dir = Path(render_row["pdf_dir"]).parent.parent / "page_text" / render_row["doc_id"]
    page_count = int(render_row.get("page_count") or 0)
    out: dict[int, str] = {}
    for page_num in range(1, page_count + 1):
        path = text_dir / f"page-{page_num}.txt"
        out[page_num] = path.read_text(encoding="utf-8") if path.exists() else ""
    return out


def page_image_path(render_row: dict[str, Any], page_num: int) -> Path:
    page_dir = Path(render_row["page_dir"])
    candidates = [
        page_dir / f"page-{page_num}.png",
        page_dir / f"page-{page_num:02d}.png",
        page_dir / f"page-{page_num:03d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(page_dir.glob(f"page-*{page_num}.png"))
    if matches:
        return matches[0]
    return candidates[0]


def strip_code_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_loose(text: str) -> dict[str, Any]:
    parsed = parse_json_response(text)
    if parsed.get("parse_error") or "raw_text" in parsed:
        cleaned = strip_code_fence(text)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
    return parsed


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isnan(out):
        return default
    return out


def normalize_bbox(raw_bbox: Any, image_size: tuple[int, int], units: str | None = None) -> tuple[int, int, int, int] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        vals = [float(v) for v in raw_bbox]
    except Exception:
        return None
    width, height = image_size
    x0, y0, x1, y1 = vals
    units_text = str(units or "").lower()
    if "norm" in units_text or "1000" in units_text or (max(vals) <= 1000 and (width > 1200 or height > 1200)):
        x0, x1 = x0 / 1000.0 * width, x1 / 1000.0 * width
        y0, y1 = y0 / 1000.0 * height, y1 / 1000.0 * height
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = max(0, min(width, int(round(x0))))
    x1 = max(0, min(width, int(round(x1))))
    y0 = max(0, min(height, int(round(y0))))
    y1 = max(0, min(height, int(round(y1))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_plausible(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[bool, str, float]:
    x0, y0, x1, y1 = bbox
    width, height = image_size
    bw, bh = x1 - x0, y1 - y0
    frac = (bw * bh) / float(width * height)
    if bw < 18 or bh < 18:
        return False, "bbox_too_small", frac
    if frac > 0.82:
        return False, "bbox_too_large", frac
    return True, "", frac


def build_bbox_prompt(row: dict[str, Any], contexts: list[dict[str, Any]], image_size: tuple[int, int]) -> tuple[str, str]:
    ctx_lines = []
    for idx, ctx in enumerate(contexts[:5], 1):
        ctx_lines.append(
            f"{idx}. zone={ctx.get('zone')} q={ctx.get('q_num')} before=『{ctx.get('anchor_before','')}』 "
            f"after=『{ctx.get('anchor_after','')}』 block={ctx.get('block_path')}"
        )
    width, height = image_size
    system_prompt = (
        "你是严谨的高中化学试卷图像定位员。只根据图片和锚文本定位目标图/公式。"
        "如果有多个可能目标、锚文本无法对应、或不确定,必须返回 found=false。"
        "不要猜测,不要框答案解析外的其他图。只返回 JSON。"
    )
    user_prompt = (
        f"整页图片像素尺寸: width={width}, height={height}。\n"
        f"目标 media={row.get('media')} kind={row.get('kind')} zones={row.get('zones')}。\n"
        "请定位紧跟/夹在这些锚文本附近的目标图或公式:\n"
        + "\n".join(ctx_lines)
        + "\n\n返回 JSON,字段固定为:"
        '{"found":true,"confidence":0.0,"bbox":[x0,y0,x1,y1],"bbox_units":"normalized_1000",'
        '"description":"一句话描述图/公式内容","evidence":"你依据的锚文本","uncertain":[]}'
        "\n要求 bbox 必须使用 0-1000 归一化坐标:左上角=(0,0),右下角=(1000,1000),bbox_units 固定写 normalized_1000。"
    )
    return system_prompt, user_prompt


def build_readback_prompt(anchor_text: str, expected_description: str) -> tuple[str, str]:
    system_prompt = (
        "你是裁片回读核验员。只描述裁片内容并判断它是否与给定锚文本/预期描述对应。"
        "锚文本通常位于裁片相邻位置,不一定出现在裁片内部;不要因为裁片里没有锚文本原文就判不匹配。"
        "主要核验裁片内容是否与预期描述一致,并且没有明显张冠李戴。"
        "不确定或只有弱相关时 matches_expected=false。只返回 JSON。"
    )
    user_prompt = (
        f"锚文本: 『{anchor_text}』\n"
        f"预期描述: 『{expected_description}』\n"
        "请回读裁片内容并核验是否一致。返回 JSON:"
        '{"description":"裁片内容","confidence":0.0,"matches_expected":true,'
        '"mismatch_reason":"","uncertain":[]}'
    )
    return system_prompt, user_prompt


def cached_vision_json(
    client: Any,
    image_path: Path,
    cache_path: Path,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    started = time.time()
    raw = client.read_page(image_path, system_prompt, user_prompt, max_tokens=max_tokens, timeout=120.0, temperature=temperature)
    payload = {
        "content": raw.get("content", ""),
        "cost_yuan": raw.get("cost_yuan", 0.0),
        "usage": raw.get("usage", {}),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(cache_path, payload)
    return payload


def crop_bbox(page_image: Path, bbox: tuple[int, int, int, int], crop_path: Path, pad: int = 6) -> dict[str, Any]:
    from PIL import Image

    with Image.open(page_image) as image:
        width, height = image.size
        x0, y0, x1, y1 = bbox
        box = (max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad))
        crop = image.convert("RGB").crop(box)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)
    extrema_min = image_extrema_min(crop_path)
    return {
        "bbox_padded": list(box),
        "crop_width": box[2] - box[0],
        "crop_height": box[3] - box[1],
        "extrema_min": extrema_min,
        "is_blank": bool(extrema_min is not None and extrema_min >= 250),
        "sha256": sha256_file(crop_path),
    }


def pending_row(row: dict[str, Any], **fields: Any) -> dict[str, Any]:
    out = {
        "schema_version": SCHEMA_VERSION,
        **row,
        **fields,
        "review_status": "pending_user_or_claude",
        "reviewer": "",
    }
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    return value


def source_selection_payload(selection: SourceSelection) -> dict[str, Any]:
    return {
        "group_key": selection.group_key,
        "role": selection.role,
        "reason": selection.reason,
        "score": selection.score,
        "path": str(selection.path) if selection.path else "",
        "source": selection.source or {},
    }


def manual_row(row: dict[str, Any], reason: str, **fields: Any) -> dict[str, Any]:
    return pending_row(row, status="manual", reason=reason, **fields)


def crop_name_for(row: dict[str, Any], crop_sha: str | None = None) -> str:
    if row.get("kind") == "dead_ref" and crop_sha:
        return f"{crop_sha}.png"
    asset_hash = str(row.get("asset_hash") or "")
    if asset_hash:
        return f"{asset_hash}__{target_key(row)[:10]}.png"
    safe_media = re.sub(r"[^0-9A-Za-z._-]+", "_", str(row.get("media") or "media"))
    return f"{stable_id(row.get('group_key'), row.get('media'), length=12)}__{safe_media}.png"


def process_one_target(
    idx: int,
    row: dict[str, Any],
    out_root: Path,
    contexts: list[dict[str, Any]],
    render_row: dict[str, Any] | None,
    client: Any | None,
    skip_vision: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not render_row or render_row.get("status") != "ok":
        return None, manual_row(row, RENDER_FAILED, target_index=idx, contexts=contexts[:5], render_status=render_row or {})
    page_texts = load_page_texts(render_row)
    page_match = match_pages(page_texts, contexts)
    if page_match.get("status") != "ok":
        return None, manual_row(row, str(page_match.get("status") or PAGE_NOT_FOUND), target_index=idx, contexts=contexts[:5], page_match=page_match)
    page_num = int(page_match["page"])
    image_path = page_image_path(render_row, page_num)
    if not image_path.exists():
        return None, manual_row(row, "page_image_missing", target_index=idx, page_number=page_num, page_image_path=str(image_path))
    if skip_vision or client is None:
        return None, manual_row(row, "vision_skipped", target_index=idx, page_number=page_num, contexts=contexts[:5], page_match=page_match)

    from PIL import Image

    with Image.open(image_path) as image:
        image_size = image.size
    cache_dir = out_root / "api_cache"
    bbox_cache = cache_dir / "bbox_v2_normalized" / f"{idx:04d}_{target_key(row)}.json"
    system_prompt, user_prompt = build_bbox_prompt(row, contexts, image_size)
    try:
        bbox_raw = cached_vision_json(client, image_path, bbox_cache, system_prompt, user_prompt, max_tokens=900)
        parsed_bbox = parse_json_loose(str(bbox_raw.get("content") or ""))
    except Exception as exc:
        return None, manual_row(row, "vl_bbox_error", target_index=idx, page_number=page_num, error=f"{type(exc).__name__}: {exc}"[:300])
    confidence = coerce_float(parsed_bbox.get("confidence"), 0.0)
    if not parsed_bbox.get("found") or confidence < 0.72:
        return None, manual_row(row, LOW_CONFIDENCE, target_index=idx, page_number=page_num, contexts=contexts[:5], page_match=page_match, vl_bbox=parsed_bbox, vl_raw=bbox_raw)
    bbox = normalize_bbox(parsed_bbox.get("bbox") or parsed_bbox.get("box_2d"), image_size, parsed_bbox.get("bbox_units") or parsed_bbox.get("coordinate_space"))
    if bbox is None:
        return None, manual_row(row, "bbox_invalid", target_index=idx, page_number=page_num, vl_bbox=parsed_bbox)
    ok, reason, frac = bbox_plausible(bbox, image_size)
    if not ok:
        return None, manual_row(row, reason, target_index=idx, page_number=page_num, bbox=list(bbox), bbox_frac=round(frac, 6), vl_bbox=parsed_bbox)

    prelim = out_root / "crops" / f"{idx:04d}_{target_key(row)}_pre.png"
    crop_meta = crop_bbox(image_path, bbox, prelim)
    if crop_meta["is_blank"]:
        return None, manual_row(row, "crop_blank", target_index=idx, page_number=page_num, bbox=list(bbox), crop_meta=crop_meta, vl_bbox=parsed_bbox)
    crop_path = out_root / "crops" / crop_name_for(row, crop_meta["sha256"])
    if crop_path != prelim:
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(prelim), str(crop_path))
    crop_meta["path"] = str(crop_path)

    anchor_text = " | ".join(str(ctx.get("anchor_text") or "") for ctx in contexts[:3] if ctx.get("anchor_text"))
    description = str(parsed_bbox.get("description") or parsed_bbox.get("note") or "")
    read_cache = cache_dir / "readback_v3_anchor_context" / f"{idx:04d}_{target_key(row)}.json"
    rb_system, rb_user = build_readback_prompt(anchor_text, description)
    try:
        read_raw = cached_vision_json(client, crop_path, read_cache, rb_system, rb_user, max_tokens=700)
        readback = parse_json_loose(str(read_raw.get("content") or ""))
    except Exception as exc:
        return None, manual_row(row, "readback_error", target_index=idx, page_number=page_num, bbox=list(bbox), crop_path=str(crop_path), error=f"{type(exc).__name__}: {exc}"[:300])
    read_conf = coerce_float(readback.get("confidence"), 0.0)
    if read_conf < 0.65 or readback.get("matches_expected") is not True:
        return None, manual_row(
            row,
            READBACK_MISMATCH,
            target_index=idx,
            page_number=page_num,
            bbox=list(bbox),
            crop_path=str(crop_path),
            anchor_text=anchor_text,
            vl_description=description,
            vl_bbox=parsed_bbox,
            vl_bbox_raw=bbox_raw,
            crop_meta=crop_meta,
            page_match=page_match,
            contexts=contexts[:5],
            readback=readback,
            readback_raw=read_raw,
        )
    final_asset_hash = crop_meta["sha256"] if row.get("kind") == "dead_ref" else row.get("asset_hash")
    candidate = pending_row(
        row,
        status="kept",
        target_index=idx,
        final_asset_hash=final_asset_hash,
        crop_sha256=crop_meta["sha256"],
        crop_path=str(crop_path),
        page_number=page_num,
        page_image_path=str(image_path),
        page_match=page_match,
        bbox=list(bbox),
        bbox_frac=round(frac, 6),
        crop_meta=crop_meta,
        anchor_text=anchor_text,
        contexts=contexts[:5],
        selected_source_role=render_row.get("source_role"),
        source_docx_path=render_row.get("source_path"),
        source_pdf_path=render_row.get("pdf_path"),
        render_doc_id=render_row.get("doc_id"),
        vl_description=description,
        vl_bbox=parsed_bbox,
        vl_bbox_raw=bbox_raw,
        readback=readback,
        readback_raw=read_raw,
    )
    return candidate, None


def build_refmap_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in candidates:
        if row.get("kind") != "dead_ref":
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_kind": "refmap_new_row",
                "group_key": row.get("group_key"),
                "media": row.get("media"),
                "asset_hash": row.get("final_asset_hash"),
                "zones": row.get("zones") or [],
                "source_crop_sha256": row.get("crop_sha256"),
                "crop_path": row.get("crop_path"),
                "page_number": row.get("page_number"),
                "bbox": row.get("bbox"),
                "review_status": "pending_user_or_claude",
                "reviewer": "",
            }
        )
    return rows


def asset_class_for_candidate(row: dict[str, Any]) -> str:
    block_types = {str(ctx.get("block_type") or "") for ctx in row.get("contexts") or []}
    if "formula" in block_types:
        return "formula_image"
    return "illustration"


def prepare_transcript_manifest(candidates: list[dict[str, Any]], out_root: Path) -> list[dict[str, Any]]:
    transcript_root = out_root / "crop_transcripts"
    repaired_dir = transcript_root / "asset_repair" / "repaired"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    by_hash: dict[str, dict[str, Any]] = {}
    refs_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        asset_hash = str(row.get("final_asset_hash") or row.get("asset_hash") or "")
        if not asset_hash:
            continue
        refs_by_hash[asset_hash].append(
            {
                "group_key": row.get("group_key"),
                "media": row.get("media"),
                "question_ids": row.get("item_ids") or [ctx.get("item_id") for ctx in row.get("contexts") or [] if ctx.get("item_id")],
                "zones": row.get("zones") or [],
                "block_types": sorted({ctx.get("block_type") for ctx in row.get("contexts") or [] if ctx.get("block_type")}),
                "crop_path": row.get("crop_path"),
                "page_number": row.get("page_number"),
                "bbox": row.get("bbox"),
            }
        )
        if asset_hash not in by_hash:
            by_hash[asset_hash] = row
            src = Path(str(row.get("crop_path") or ""))
            if src.exists():
                dst = repaired_dir / f"{asset_hash}.png"
                if not dst.exists():
                    shutil.copy2(src, dst)
    manifest_rows: list[dict[str, Any]] = []
    for asset_hash, row in sorted(by_hash.items()):
        refs = refs_by_hash[asset_hash]
        manifest_rows.append(
            {
                "asset_hash": asset_hash,
                "asset_class": asset_class_for_candidate(row),
                "sample_refs": refs,
                "ref_count": len(refs),
                "question_count": len({qid for ref in refs for qid in ref.get("question_ids") or []}),
                "source_scope": "batch13_crop_rescue",
                "crop_sha256": row.get("crop_sha256"),
                "crop_path": row.get("crop_path"),
                "zones": sorted({z for ref in refs for z in ref.get("zones") or []}),
                "block_types": sorted({bt for ref in refs for bt in ref.get("block_types") or []}),
            }
        )
    write_jsonl(transcript_root / "crop_asset_manifest.jsonl", manifest_rows)
    return manifest_rows


def run_transcripts(candidates: list[dict[str, Any]], out_root: Path, client: Any | None, workers: int, skip_vision: bool) -> dict[str, Any]:
    transcript_root = out_root / "crop_transcripts"
    manifest_rows = prepare_transcript_manifest(candidates, out_root)
    formula_rows = [row for row in manifest_rows if row.get("asset_class") == "formula_image"]
    illustration_rows = [row for row in manifest_rows if row.get("asset_class") == "illustration"]
    cache_dir = transcript_root / "api_cache"
    raw_formula = run_formula_rows(formula_rows, transcript_root, cache_dir, client, workers, skip_vision, "batch13_crop_formula")
    formula_summary = write_transcription_outputs(
        raw_formula,
        transcript_root / "formula_latex" / "formula_latex_candidates.jsonl",
        "13d_crop_formula",
        "formula",
    )
    raw_transcripts = run_illustration_track(
        illustration_rows,
        transcript_root,
        client,
        cache_dir,
        workers,
        None,
        out_path=transcript_root / "transcripts" / "transcript_candidates_raw.jsonl",
        skip_vision=skip_vision,
    )
    transcript_summary = write_transcription_outputs(
        raw_transcripts,
        transcript_root / "transcripts" / "transcript_candidates.jsonl",
        "13d_crop_transcript",
        "transcript",
    )
    summary = {
        "manifest_rows": len(manifest_rows),
        "formula_rows": len(formula_rows),
        "illustration_rows": len(illustration_rows),
        "formula": formula_summary,
        "transcripts": transcript_summary,
        "cost_yuan": round(sum_cost(raw_formula) + sum_cost(raw_transcripts), 4),
    }
    write_json(transcript_root / "transcript_summary.json", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Batch 13 QA-3 Visual Crop Report",
        "",
        "## Counts",
        "",
        f"- Targets: {summary['targets']}",
        f"- Kept: {summary['kept']}",
        f"- Rescue rate: {summary['kept_rate_pct']}%",
        f"- Kept dead_asset/dead_ref: {summary['dead_asset_kept']} / {summary['dead_ref_kept']}",
        f"- Manual: {summary['manual']}",
        f"- Render-fail refs: {summary['render_fail_refs']}",
        f"- Accounting kept + manual_non_render + render_fail = {summary['accounting_total']}",
        f"- Dead-ref refmap rows: {summary['refmap_rows']}",
        f"- Unique rendered documents: {summary['render_docs']}",
        f"- Render failed documents: {summary['render_failed_docs']}",
        f"- Estimated vision cost yuan: {summary['cost_yuan']}",
        "",
        "## Manual Reasons",
        "",
    ]
    for reason, count in summary["manual_reasons"]:
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            "## 13d Transcripts",
            "",
            json.dumps(summary.get("transcripts") or {}, ensure_ascii=False, indent=2),
            "",
            "## Discipline",
            "",
            "- L0 candidate package only; official data was not modified.",
            "- All candidate/manual/rejection rows keep reviewer empty and review_status pending_user_or_claude.",
            "- Source selection follows media prefix: ans_* uses analysis/answer-like source; non-ans uses question/original-like source.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_delivery(out_root: Path, targets: int) -> dict[str, Any]:
    candidates = load_jsonl(out_root / "crop_candidates.jsonl")
    manual = load_jsonl(out_root / "crop_manual.jsonl")
    refmap = load_jsonl(out_root / "refmap_new_rows.jsonl")
    render_fail_refs = [row for row in manual if row.get("reason") == RENDER_FAILED]
    manual_non_render = [row for row in manual if row.get("reason") != RENDER_FAILED]
    candidate_bad_review = [row for row in candidates + manual + refmap if row.get("reviewer") or row.get("review_status") != "pending_user_or_claude"]
    codex_hits = [row for row in candidates + manual + refmap if "codex_" in json.dumps(row, ensure_ascii=False).lower()]
    blank_kept = []
    for row in candidates:
        path = Path(str(row.get("crop_path") or ""))
        if not path.exists() or is_blank_image(path):
            blank_kept.append(row.get("crop_path"))
    return {
        "targets": targets,
        "kept": len(candidates),
        "manual": len(manual),
        "render_fail_refs": len(render_fail_refs),
        "manual_non_render": len(manual_non_render),
        "accounting_total": len(candidates) + len(manual_non_render) + len(render_fail_refs),
        "refmap_rows": len(refmap),
        "bad_review_rows": len(candidate_bad_review),
        "codex_reviewer_hits": len(codex_hits),
        "blank_kept": blank_kept,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=TARGETS_PATH)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-transcripts", action="store_true")
    parser.add_argument("--transcript-workers", type=int, default=1)
    args = parser.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    targets = load_jsonl(args.targets)
    if args.limit:
        targets = targets[: args.limit]
    sources_by_group = load_sources_by_group()
    items = list(iter_items())
    service_ids = {row["item_id"] for row in iter_service_items()}
    context_map = build_context_map(items)

    selections: dict[tuple[str, str], SourceSelection] = {}
    for row in targets:
        key = (str(row.get("group_key") or ""), role_for_media(str(row.get("media") or "")))
        if key not in selections:
            selections[key] = select_source_for_media(sources_by_group.get(key[0]), key[0], str(row.get("media") or ""))
    render_rows: dict[str, dict[str, Any]] = {}
    render_failures: list[dict[str, Any]] = []
    for selection in selections.values():
        rendered = render_source(selection, out_root)
        render_rows[source_render_id(selection)] = rendered
        if rendered.get("status") != "ok":
            render_failures.append(rendered)
    write_jsonl(out_root / "render_ledger.jsonl", render_rows.values())
    write_jsonl(out_root / "render_failures.jsonl", render_failures)

    client = None
    if not args.skip_vision:
        client = build_vision_client()

    candidates: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    for idx, row in enumerate(targets, 1):
        row_contexts = contexts_for_target(row, context_map)
        for ctx in row_contexts:
            ctx["_is_service_item"] = ctx.get("item_id") in service_ids
        selection = select_source_for_media(sources_by_group.get(str(row.get("group_key") or "")), str(row.get("group_key") or ""), str(row.get("media") or ""))
        rendered = render_rows.get(source_render_id(selection))
        if selection.reason not in {"ok"}:
            manual.append(manual_row(row, SOURCE_NOT_FOUND, target_index=idx, source_selection=source_selection_payload(selection), contexts=row_contexts[:5]))
            continue
        candidate, miss = process_one_target(idx, row, out_root, row_contexts, rendered, client, args.skip_vision)
        if candidate:
            candidates.append(candidate)
        if miss:
            manual.append(miss)
        if idx % 25 == 0:
            print(f"[progress] processed {idx}/{len(targets)} kept={len(candidates)} manual={len(manual)}")
    candidates = [json_safe(row) for row in candidates]
    manual = [json_safe(row) for row in manual]
    write_jsonl(out_root / "crop_candidates.jsonl", candidates)
    write_jsonl(out_root / "crop_manual.jsonl", manual)
    refmap_rows = build_refmap_rows(candidates)
    write_jsonl(out_root / "refmap_new_rows.jsonl", refmap_rows)

    transcript_summary: dict[str, Any] = {}
    if not args.skip_transcripts and candidates:
        transcript_summary = run_transcripts(candidates, out_root, client, args.transcript_workers, args.skip_vision)

    validation = validate_delivery(out_root, len(targets))
    reason_counts = Counter(row.get("reason") for row in manual)
    total_cost = 0.0
    for row in candidates + manual:
        for key in ("vl_bbox_raw", "readback_raw"):
            raw = row.get(key) or {}
            total_cost += float(raw.get("cost_yuan") or 0.0)
    total_cost += float((transcript_summary or {}).get("cost_yuan") or 0.0)
    summary = {
        **validation,
        "kept_rate_pct": round(validation["kept"] * 100.0 / len(targets), 2) if targets else 0.0,
        "dead_asset_kept": sum(1 for row in candidates if row.get("kind") == "dead_asset"),
        "dead_ref_kept": sum(1 for row in candidates if row.get("kind") == "dead_ref"),
        "manual_reasons": reason_counts.most_common(),
        "render_docs": len(render_rows),
        "render_failed_docs": len(render_failures),
        "cost_yuan": round(total_cost, 4),
        "transcripts": transcript_summary,
    }
    write_json(out_root / "batch13_summary.json", summary)
    write_report(out_root / "BATCH13_REPORT.md", summary)
    if validation["accounting_total"] != len(targets) or validation["bad_review_rows"] or validation["codex_reviewer_hits"] or validation["blank_kept"]:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
