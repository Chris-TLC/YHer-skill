#!/usr/bin/env python3
"""Batch 8 WS2 asset repair and two-track transcription.

This is an L0 candidate pipeline: data artifacts are written under
/tmp/yher_batch8_ws2 by default. Official item-bank and WS2 candidate asset
directories are read-only inputs.
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
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ASSET_ROOT = REPO_ROOT / "data" / "ws2_assets_v1_candidate_20260703"
MANIFEST_PATH = ASSET_ROOT / "asset_manifest.jsonl"
BATCH8_CANDIDATE_ROOT = REPO_ROOT / "data" / "batch8_ws2_candidate_20260704"
BATCH8P1_OUT_ROOT = Path("/tmp/yher_batch8p1_ws2")
GOLD_LIST_PATH = REPO_ROOT / "data" / "ws2_gold_set_20260704" / "gold_asset_list.jsonl"
SERVICE_MAP_PATH = REPO_ROOT / "data" / "ws2_gold_set_20260704" / "ws2_asset_service_map.json"
CLAUDE_BLANK_PATH = REPO_ROOT / "data" / "ws2_gold_set_20260704" / "ws2_blank_assets.json"
TMP_BLANK_PATH = Path("/tmp/ws2_blank_assets.json")
OUT_ROOT = Path("/tmp/yher_batch8_ws2")
SOFFICE = Path("/opt/homebrew/bin/soffice")
SCHEMA_VERSION = "ws2_transcript_v1"
RELATION_SYMBOL_TRANSCRIPTION_RULE = "化学式之间的关系符号(=、→、⇌、↑、↓)必须严格按图面逐字转写。禁止依据化学习惯改写:图面是等号就写 =,即使你认为该反应可逆或应写箭头。"
BATCH8P1_SINGLE_RETRY_PREFIX = "042b96f70d36eea1"
LEAK_RE = re.compile(r"【\s*(?:试题|题目)?\s*解\s*析\s*】|故选|正确答案")
FINE_TYPES = {
    "plot_curve",
    "apparatus_diagram",
    "flow_chart",
    "molecular_structure",
    "chemical_equation_image",
    "data_table_image",
    "formula_fragment",
    "text_snippet",
    "diagram_other",
    "icon_or_noise",
    "broken_image",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_png_for(row: dict[str, Any], asset_root: Path = ASSET_ROOT) -> Path:
    return asset_root / "normalized" / f"{row['asset_hash']}.png"


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    return load_jsonl(path)


def png_extrema_min_fallback(path: Path) -> int:
    from scripts.build_ws2_asset_manifest import png_decode_rows

    width, _height, color_type, bpp, rows = png_decode_rows(path.read_bytes())
    min_value = 255
    for row in rows:
        for x in range(width):
            px = row[x * bpp : (x + 1) * bpp]
            if color_type == 0:
                min_value = min(min_value, px[0])
            elif color_type == 2:
                min_value = min(min_value, px[0], px[1], px[2])
            elif color_type == 4:
                if px[1] > 0:
                    min_value = min(min_value, px[0])
            elif color_type == 6:
                alpha = px[3]
                if alpha > 0:
                    min_value = min(min_value, px[0], px[1], px[2])
    return min_value


def image_extrema_min(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.mode in {"RGBA", "LA"} or ("transparency" in image.info):
                bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
                bg.alpha_composite(image.convert("RGBA"))
                image = bg.convert("RGB")
            else:
                image = image.convert("RGB")
            extrema = image.getextrema()
            return min(channel_min for channel_min, _channel_max in extrema)
    except ImportError:
        if path.suffix.lower() == ".png":
            return png_extrema_min_fallback(path)
        raise
    except Exception:
        if path.suffix.lower() == ".png":
            try:
                return png_extrema_min_fallback(path)
            except Exception:
                return None
        return None


def is_blank_image(path: Path) -> bool:
    extrema_min = image_extrema_min(path)
    return extrema_min is not None and extrema_min >= 250


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_bad_assets(manifest_rows: list[dict[str, Any]], asset_root: Path = ASSET_ROOT) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    blank: list[dict[str, Any]] = []
    checked = 0
    for row in manifest_rows:
        asset_hash = row["asset_hash"]
        png = normalized_png_for(row, asset_root)
        if not png.exists():
            missing.append({"asset_hash": asset_hash, "reason": "normalized_png_missing", "asset_class": row.get("asset_class")})
            continue
        checked += 1
        extrema_min = image_extrema_min(png)
        if extrema_min is not None and extrema_min >= 250:
            blank.append(
                {
                    "asset_hash": asset_hash,
                    "reason": "pixel_extrema_min_ge_250",
                    "extrema_min": extrema_min,
                    "asset_class": row.get("asset_class"),
                    "dimensions": row.get("dimensions"),
                }
            )
    bad_hashes = sorted({r["asset_hash"] for r in missing + blank})
    return {"missing": missing, "blank": blank, "bad_hashes": bad_hashes, "checked_existing_png": checked}


def load_service_items() -> dict[str, dict[str, Any]]:
    from core.data.item_bank_v4 import iter_items, iter_service_items

    service_ids = {item["item_id"] for item in iter_service_items()}
    by_id: dict[str, dict[str, Any]] = {}
    for item in iter_items():
        copied = dict(item)
        copied["_is_service_item"] = item.get("item_id") in service_ids
        by_id[item["item_id"]] = copied
    return by_id


def service_metadata_for_asset(row: dict[str, Any], items_by_id: dict[str, dict[str, Any]], service_map: dict[str, Any]) -> dict[str, Any]:
    refs = row.get("sample_refs") or []
    ref_items = [items_by_id.get(ref.get("question_id", "")) for ref in refs]
    ref_items = [item for item in ref_items if item]
    service_items = [item for item in ref_items if item.get("_is_service_item")]
    chosen = (service_items or ref_items or [{}])[0]
    svc = service_map.get(row.get("asset_hash", ""), {}) if isinstance(service_map, dict) else {}
    return {
        "context_item_id": chosen.get("item_id"),
        "context_is_service_item": bool(chosen.get("_is_service_item")),
        "context_stem_text": str(chosen.get("stem_text") or ""),
        "knowledge_points": chosen.get("knowledge_points") or [],
        "service_item_count": svc.get("n_items", 0),
        "top_kg": svc.get("top_kg", []) if isinstance(svc, dict) else [],
    }


def candidate_source_paths(row: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for ref in row.get("sample_refs") or []:
        raw = str(ref.get("asset_path") or "")
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists() and p not in paths:
            paths.append(p)
    original = ASSET_ROOT / "originals" / f"{row['asset_hash']}{row.get('original_ext') or ''}"
    if original.exists() and original not in paths:
        paths.append(original)
    return paths


def composite_png_on_white(src: Path, dst: Path) -> None:
    try:
        from PIL import Image

        with Image.open(src) as image:
            if image.mode in {"RGBA", "LA"} or ("transparency" in image.info):
                bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
                bg.alpha_composite(image.convert("RGBA"))
                bg.convert("RGB").save(dst)
            else:
                image.convert("RGB").save(dst)
    except ImportError:
        shutil.copy2(src, dst)


def crop_png(src: Path, dst: Path) -> dict[str, Any]:
    from scripts.build_ws2_asset_manifest import crop_png_whitespace

    return crop_png_whitespace(src, dst)


def run_soffice_convert(src: Path, out_dir: Path, target: str, timeout: int = 120) -> tuple[Path | None, str]:
    if not SOFFICE.exists():
        return None, f"soffice_missing:{SOFFICE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [str(SOFFICE), "--headless", "--convert-to", target, "--outdir", str(out_dir), str(src)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"soffice_timeout_{target}"
    produced = out_dir / f"{src.stem}.{target.split(':', 1)[0].split(',', 1)[0]}"
    if produced.exists():
        return produced, "ok"
    details = " ".join((proc.stdout or "", proc.stderr or "")).strip().replace("\n", " ")[:240]
    return None, f"soffice_no_output_{target}:{details}"


def try_repair_asset(row: dict[str, Any], out_root: Path, items_by_id: dict[str, dict[str, Any]], service_map: dict[str, Any]) -> dict[str, Any]:
    asset_hash = row["asset_hash"]
    repair_dir = out_root / "asset_repair"
    repaired_dir = repair_dir / "repaired"
    work_dir = repair_dir / "work" / asset_hash
    repaired_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    before_png = normalized_png_for(row)
    before_min = image_extrema_min(before_png) if before_png.exists() else None
    meta = service_metadata_for_asset(row, items_by_id, service_map)
    source_paths = candidate_source_paths(row)
    attempts: list[dict[str, Any]] = []
    final_png = repaired_dir / f"{asset_hash}.png"

    for source_idx, source in enumerate(source_paths):
        ext = source.suffix.lower()
        local_source = work_dir / f"{asset_hash}_{source_idx}{ext}"
        shutil.copy2(source, local_source)
        if ext in {".png", ".jpg", ".jpeg"}:
            candidate = work_dir / f"{asset_hash}_{source_idx}_white.png"
            composite_png_on_white(local_source, candidate)
            cropped = work_dir / f"{asset_hash}_{source_idx}_cropped.png"
            try:
                crop_png(candidate, cropped)
                candidate = cropped
            except Exception:
                pass
            after_min = image_extrema_min(candidate)
            attempts.append({"converter": "source_raster_copy", "source": str(source), "extrema_min": after_min})
            if after_min is not None and after_min < 250:
                shutil.copy2(candidate, final_png)
                return {
                    "asset_hash": asset_hash,
                    "status": "repaired",
                    "source_path": str(source),
                    "converter": "source_raster_copy",
                    "before_extrema_min": before_min,
                    "after_extrema_min": after_min,
                    "repaired_png": str(final_png),
                    "service_item_count": meta["service_item_count"],
                    "top_kg": meta["top_kg"],
                    "attempts": attempts,
                }
        if ext in {".wmf", ".emf"}:
            raw_png, reason = run_soffice_convert(local_source, work_dir / "soffice_png", "png")
            if raw_png:
                candidate = work_dir / f"{asset_hash}_{source_idx}_soffice_cropped.png"
                try:
                    crop_png(raw_png, candidate)
                except Exception:
                    candidate = raw_png
                after_min = image_extrema_min(candidate)
                attempts.append({"converter": "libreoffice_png", "source": str(source), "extrema_min": after_min})
                if after_min is not None and after_min < 250:
                    shutil.copy2(candidate, final_png)
                    return {
                        "asset_hash": asset_hash,
                        "status": "repaired",
                        "source_path": str(source),
                        "converter": "libreoffice_png",
                        "before_extrema_min": before_min,
                        "after_extrema_min": after_min,
                        "repaired_png": str(final_png),
                        "service_item_count": meta["service_item_count"],
                        "top_kg": meta["top_kg"],
                        "attempts": attempts,
                    }
            else:
                attempts.append({"converter": "libreoffice_png", "source": str(source), "failure": reason})
            raw_svg, svg_reason = run_soffice_convert(local_source, work_dir / "soffice_svg", "svg")
            attempts.append({"converter": "libreoffice_svg", "source": str(source), "status": "ok" if raw_svg else svg_reason})

    return {
        "asset_hash": asset_hash,
        "status": "unrepairable",
        "source_paths": [str(p) for p in source_paths],
        "failure_reason": "all_sources_missing_or_blank_after_render" if source_paths else "no_source_asset_found",
        "before_extrema_min": before_min,
        "service_item_count": meta["service_item_count"],
        "top_kg": meta["top_kg"],
        "attempts": attempts,
    }


def run_soffice_batch(paths: list[Path], out_dir: Path, target: str, chunk_size: int = 80, timeout: int = 300) -> None:
    if not paths or not SOFFICE.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start : start + chunk_size]
        try:
            subprocess.run(
                [str(SOFFICE), "--headless", "--convert-to", target, "--outdir", str(out_dir), *map(str, chunk)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            continue


def repair_bad_assets_batch(
    bad_rows: list[dict[str, Any]],
    out_root: Path,
    items_by_id: dict[str, dict[str, Any]],
    service_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    repair_dir = out_root / "asset_repair"
    source_dir = repair_dir / "source_stage"
    raw_png_dir = repair_dir / "raw_png"
    raw_svg_dir = repair_dir / "raw_svg"
    repaired_dir = repair_dir / "repaired"
    for directory in [source_dir, raw_png_dir, raw_svg_dir, repaired_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    stage_rows: list[dict[str, Any]] = []
    vector_inputs: list[Path] = []
    for row in bad_rows:
        source_paths = candidate_source_paths(row)
        chosen = source_paths[0] if source_paths else None
        staged: Path | None = None
        if chosen:
            staged = source_dir / f"{row['asset_hash']}{chosen.suffix.lower()}"
            if not staged.exists():
                shutil.copy2(chosen, staged)
            if chosen.suffix.lower() in {".wmf", ".emf"}:
                vector_inputs.append(staged)
        stage_rows.append({"row": row, "source_paths": source_paths, "chosen": chosen, "staged": staged})

    run_soffice_batch(vector_inputs, raw_png_dir, "png")
    run_soffice_batch(vector_inputs, raw_svg_dir, "svg")

    repaired: list[dict[str, Any]] = []
    unrepairable: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for entry in stage_rows:
        row = entry["row"]
        asset_hash = row["asset_hash"]
        meta = service_metadata_for_asset(row, items_by_id, service_map)
        before_png = normalized_png_for(row)
        before_min = image_extrema_min(before_png) if before_png.exists() else None
        attempts: list[dict[str, Any]] = []
        staged: Path | None = entry["staged"]
        final_png = repaired_dir / f"{asset_hash}.png"
        result: dict[str, Any] | None = None
        if staged and staged.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            candidate = repair_dir / "work" / f"{asset_hash}_source_png.png"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            composite_png_on_white(staged, candidate)
            after_min = image_extrema_min(candidate)
            attempts.append({"converter": "source_raster_copy", "source": str(entry["chosen"]), "extrema_min": after_min})
            if after_min is not None and after_min < 250:
                shutil.copy2(candidate, final_png)
                result = {
                    "asset_hash": asset_hash,
                    "status": "repaired",
                    "source_path": str(entry["chosen"]),
                    "converter": "source_raster_copy",
                    "before_extrema_min": before_min,
                    "after_extrema_min": after_min,
                    "repaired_png": str(final_png),
                    "service_item_count": meta["service_item_count"],
                    "top_kg": meta["top_kg"],
                    "attempts": attempts,
                }
        if result is None and staged and staged.suffix.lower() in {".wmf", ".emf"}:
            raw_png = raw_png_dir / f"{staged.stem}.png"
            raw_svg = raw_svg_dir / f"{staged.stem}.svg"
            if raw_svg.exists():
                attempts.append({"converter": "libreoffice_svg", "source": str(entry["chosen"]), "status": "ok", "svg": str(raw_svg)})
            else:
                attempts.append({"converter": "libreoffice_svg", "source": str(entry["chosen"]), "failure": "no_svg_output"})
            if raw_png.exists():
                candidate = repair_dir / "work" / f"{asset_hash}_soffice_cropped.png"
                candidate.parent.mkdir(parents=True, exist_ok=True)
                try:
                    crop_png(raw_png, candidate)
                except Exception:
                    candidate = raw_png
                after_min = image_extrema_min(candidate)
                attempts.append({"converter": "libreoffice_png", "source": str(entry["chosen"]), "extrema_min": after_min})
                if after_min is not None and after_min < 250:
                    shutil.copy2(candidate, final_png)
                    result = {
                        "asset_hash": asset_hash,
                        "status": "repaired",
                        "source_path": str(entry["chosen"]),
                        "converter": "libreoffice_png",
                        "before_extrema_min": before_min,
                        "after_extrema_min": after_min,
                        "repaired_png": str(final_png),
                        "service_item_count": meta["service_item_count"],
                        "top_kg": meta["top_kg"],
                        "attempts": attempts,
                    }
            else:
                attempts.append({"converter": "libreoffice_png", "source": str(entry["chosen"]), "failure": "no_png_output"})
        if result is None:
            result = {
                "asset_hash": asset_hash,
                "status": "unrepairable",
                "source_paths": [str(p) for p in entry["source_paths"]],
                "failure_reason": "all_sources_missing_or_blank_after_render" if entry["source_paths"] else "no_source_asset_found",
                "before_extrema_min": before_min,
                "service_item_count": meta["service_item_count"],
                "top_kg": meta["top_kg"],
                "attempts": attempts,
            }
        all_results.append(result)
        if result["status"] == "repaired":
            repaired.append(result)
        else:
            unrepairable.append(result)
    return repaired, unrepairable, all_results


def repair_bad_assets(manifest_rows: list[dict[str, Any]], out_root: Path) -> dict[str, Any]:
    repair_dir = out_root / "asset_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    items_by_id = load_service_items()
    service_map = json.loads(SERVICE_MAP_PATH.read_text(encoding="utf-8")) if SERVICE_MAP_PATH.exists() else {}
    detected = detect_bad_assets(manifest_rows)
    by_hash = {row["asset_hash"]: row for row in manifest_rows}
    bad_rows = [by_hash[h] for h in detected["bad_hashes"] if h in by_hash]
    repaired, unrepairable, all_results = repair_bad_assets_batch(bad_rows, out_root, items_by_id, service_map)
    claude_blank = load_json_or_empty(CLAUDE_BLANK_PATH)
    tmp_blank = load_json_or_empty(TMP_BLANK_PATH)
    claude_blank_hashes = sorted(extract_hashes(claude_blank))
    tmp_blank_hashes = sorted(extract_hashes(tmp_blank))
    detected_blank_hashes = sorted(row["asset_hash"] for row in detected["blank"])
    detected_missing_hashes = sorted(row["asset_hash"] for row in detected["missing"])
    summary = {
        "detected_blank_count": len(detected_blank_hashes),
        "detected_missing_count": len(detected_missing_hashes),
        "detected_bad_total": len(detected["bad_hashes"]),
        "repaired_count": len(repaired),
        "unrepairable_count": len(unrepairable),
        "claude_blank_count": len(claude_blank_hashes),
        "tmp_blank_count": len(tmp_blank_hashes),
        "blank_diff_vs_claude": {
            "detected_minus_claude": sorted(set(detected_blank_hashes) - set(claude_blank_hashes)),
            "claude_minus_detected": sorted(set(claude_blank_hashes) - set(detected_blank_hashes)),
        },
        "blank_diff_vs_tmp": {
            "detected_minus_tmp": sorted(set(detected_blank_hashes) - set(tmp_blank_hashes)),
            "tmp_minus_detected": sorted(set(tmp_blank_hashes) - set(detected_blank_hashes)),
        },
    }
    write_json(repair_dir / "bad_asset_detection_summary.json", summary)
    write_jsonl(repair_dir / "detected_blank_assets.jsonl", detected["blank"])
    write_jsonl(repair_dir / "detected_missing_assets.jsonl", detected["missing"])
    write_jsonl(repair_dir / "repair_attempts.jsonl", all_results)
    write_jsonl(repair_dir / "repaired_assets.jsonl", repaired)
    write_jsonl(repair_dir / "unrepairable.jsonl", unrepairable)
    write_asset_repair_report(repair_dir / "asset_repair_report.md", summary, repaired, unrepairable)
    return {"summary": summary, "repaired": repaired, "unrepairable": unrepairable, "detected": detected}


def load_json_or_empty(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def extract_hashes(data: Any) -> set[str]:
    hashes: set[str] = set()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                hashes.add(item)
            elif isinstance(item, dict):
                value = item.get("asset_hash") or item.get("hash")
                if value:
                    hashes.add(str(value))
    elif isinstance(data, dict):
        for key, value in data.items():
            if re.fullmatch(r"[0-9a-f]{64}", str(key)):
                hashes.add(str(key))
            if isinstance(value, dict):
                nested = value.get("asset_hash") or value.get("hash")
                if nested:
                    hashes.add(str(nested))
    return hashes


def write_asset_repair_report(path: Path, summary: dict[str, Any], repaired: list[dict[str, Any]], unrepairable: list[dict[str, Any]]) -> None:
    lines = [
        "# Batch 8 WS2 Asset Repair Report",
        "",
        "## Counts",
        "",
        f"- Detected blank assets: {summary['detected_blank_count']}",
        f"- Detected missing assets: {summary['detected_missing_count']}",
        f"- Bad assets total: {summary['detected_bad_total']}",
        f"- Repaired assets: {summary['repaired_count']}",
        f"- Unrepairable assets: {summary['unrepairable_count']}",
        "",
        "## Diff Against Frozen Lists",
        "",
        f"- Claude blank list count: {summary['claude_blank_count']}",
        f"- /tmp blank list count: {summary['tmp_blank_count']}",
        f"- Detected minus Claude blank: {len(summary['blank_diff_vs_claude']['detected_minus_claude'])}",
        f"- Claude blank minus detected: {len(summary['blank_diff_vs_claude']['claude_minus_detected'])}",
        f"- Detected minus /tmp blank: {len(summary['blank_diff_vs_tmp']['detected_minus_tmp'])}",
        f"- /tmp blank minus detected: {len(summary['blank_diff_vs_tmp']['tmp_minus_detected'])}",
        "",
        "## Repaired Samples",
        "",
    ]
    for row in repaired[:30]:
        lines.append(
            f"- `{row['asset_hash'][:12]}` converter={row.get('converter')} before={row.get('before_extrema_min')} "
            f"after={row.get('after_extrema_min')} svc_items={row.get('service_item_count')}"
        )
    lines.extend(["", "## Unrepairable Samples", ""])
    for row in unrepairable[:50]:
        lines.append(
            f"- `{row['asset_hash'][:12]}` reason={row.get('failure_reason')} before={row.get('before_extrema_min')} "
            f"svc_items={row.get('service_item_count')} kg={','.join(map(str, row.get('top_kg') or []))}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def effective_png_for(row: dict[str, Any], out_root: Path) -> Path | None:
    repaired_paths = [
        out_root / "asset_repair" / "repaired" / f"{row['asset_hash']}.png",
        BATCH8_CANDIDATE_ROOT / "asset_repair" / "repaired" / f"{row['asset_hash']}.png",
    ]
    for repaired in repaired_paths:
        if repaired.exists():
            return repaired
    normalized = normalized_png_for(row)
    return normalized if normalized.exists() else None


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def build_vision_client() -> Any:
    load_env_file(REPO_ROOT / ".env")
    env_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not env_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not available in environment or .env")
    from adapters.vision_client import VisionClient

    return VisionClient(provider="qwen-vl", api_key=env_key)


def build_formula_prompts() -> tuple[str, str]:
    system_prompt = (
        "你是上海高中化学公式图片转写器。只把图片中的公式、方程式、结构式文字转成 LaTeX。"
        "优先使用 mhchem 的 \\ce{} 语法。不要解题、不要补充图片外信息。"
        f"{RELATION_SYMBOL_TRANSCRIPTION_RULE}"
        "只返回 JSON: {\"latex\":\"...\",\"confidence\":0.0,\"uncertain\":[\"...\"]}。"
    )
    user_prompt = (
        "请转写这张 formula_image。要求: 可逆号用 \\rightleftharpoons 或 \\ce{<=>}; "
        "气体/沉淀/条件/电荷/上下标必须保留; MathType 伪影如 NH3gH2O 中 g 可能是 ·。"
    )
    return system_prompt, user_prompt


def build_illustration_prompts(_fine_type_hint: str, item: dict[str, Any]) -> tuple[str, str]:
    stem_context = str(item.get("stem_text") or "")[:300]
    system_prompt = (
        "只描述、不分析、不推断、不解题;图内每个标注/数字/箭头/结构都要写;"
        "不确定就进 uncertain,禁止编造。"
        "必须只返回 JSON,字段为 asset_hash 可省略、fine_type、summary、elements、text_in_image、data_points、uncertain、confidence。"
    )
    user_prompt = (
        "请按 WS2 图形转写规范描述这张图。fine_type 必须是以下之一: "
        + ", ".join(sorted(FINE_TYPES))
        + "。\n分型要求: plot_curve 记录轴名单位刻度曲线交点极值平台; apparatus_diagram 记录仪器连接和流向;"
        " flow_chart 记录节点、箭头试剂条件、分支循环; molecular_structure 记录骨架官能团取代基电荷;"
        " chemical_equation_image 记录反应物产物和箭头条件; data_table_image 转 Markdown 表格。"
        " 还原图中文字的 GBK mojibake,不要照抄乱码。\n"
        f"题干上下文前 300 字: {stem_context}"
    )
    return system_prompt, user_prompt


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                return parsed if isinstance(parsed, dict) else {"raw": parsed}
            except Exception:
                pass
    return {"raw_text": text, "parse_error": True}


def fix_mojibake_text(value: Any) -> Any:
    if isinstance(value, str):
        return fix_mojibake_string(value)
    if isinstance(value, list):
        return [fix_mojibake_text(v) for v in value]
    if isinstance(value, dict):
        return {k: fix_mojibake_text(v) for k, v in value.items()}
    return value


def fix_mojibake_string(text: str) -> str:
    # Common WMF path: GBK bytes were decoded as latin-1-like text.
    if not re.search(r"[Ê¼ÁÓ¦·´ÏÔÐÑ¼]", text):
        return text
    try:
        repaired = text.encode("latin-1", errors="ignore").decode("gbk", errors="ignore")
        if repaired and sum("\u4e00" <= ch <= "\u9fff" for ch in repaired) > sum("\u4e00" <= ch <= "\u9fff" for ch in text):
            return repaired
    except Exception:
        return text
    return text


def normalize_formula_result(parsed: dict[str, Any]) -> dict[str, Any]:
    latex = str(parsed.get("latex") or parsed.get("LaTeX") or parsed.get("result") or "").strip()
    confidence = coerce_confidence(parsed.get("confidence"), default=0.0)
    uncertain = parsed.get("uncertain") if isinstance(parsed.get("uncertain"), list) else []
    return {"latex": latex, "confidence": confidence, "uncertain": [str(x) for x in uncertain], "raw": parsed}


def normalize_transcript_result(parsed: dict[str, Any]) -> dict[str, Any]:
    parsed = fix_mojibake_text(parsed)
    fine_type = str(parsed.get("fine_type") or parsed.get("figure_type") or "diagram_other").strip()
    if fine_type not in FINE_TYPES:
        fine_type = "diagram_other"
    return {
        "fine_type": fine_type,
        "summary": str(parsed.get("summary") or ""),
        "elements": normalize_str_list(parsed.get("elements")),
        "text_in_image": normalize_str_list(parsed.get("text_in_image") or parsed.get("labels")),
        "data_points": normalize_str_list(parsed.get("data_points")),
        "uncertain": normalize_str_list(parsed.get("uncertain")),
        "confidence": coerce_confidence(parsed.get("confidence"), default=0.0),
        "raw": parsed,
    }


def normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def coerce_confidence(value: Any, default: float) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    if math.isnan(f):
        return default
    return max(0.0, min(1.0, f))


def validate_latex_static(latex: str) -> dict[str, Any]:
    if not latex.strip():
        return {"ok": False, "engine": "static", "error": "empty_latex"}
    balance = 0
    for ch in latex:
        if ch == "{":
            balance += 1
        elif ch == "}":
            balance -= 1
        if balance < 0:
            return {"ok": False, "engine": "static", "error": "unbalanced_braces"}
    if balance != 0:
        return {"ok": False, "engine": "static", "error": "unbalanced_braces"}
    return {"ok": True, "engine": "static", "error": ""}


def validate_latex_katex(latex: str, node_path: Path | None = None, node_modules: Path | None = None) -> dict[str, Any]:
    latex = strip_math_delimiters(latex)
    static = validate_latex_static(latex)
    if not static["ok"]:
        return static
    if node_modules is None:
        node_modules = default_node_modules(OUT_ROOT)
    node = node_path or Path(os.environ.get("YHER_NODE_BIN", str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")))
    if not node.exists():
        node = Path(shutil.which("node") or "")
    if not node or not node.exists():
        static["engine"] = "static_no_node"
        return static
    script = (
        "const fs=require('fs');"
        "const katex=require('katex');"
        "try{require('katex/contrib/mhchem');}catch(e){}"
        "const s=fs.readFileSync(0,'utf8');"
        "try{katex.renderToString(s,{throwOnError:true,strict:false});console.log('OK');}"
        "catch(e){console.error(e.message);process.exit(2);}"
    )
    env = os.environ.copy()
    if node_modules and node_modules.exists():
        env["NODE_PATH"] = str(node_modules)
    try:
        proc = subprocess.run([str(node), "-e", script], input=latex, capture_output=True, text=True, timeout=10, env=env)
    except Exception as exc:
        return {"ok": static["ok"], "engine": "static_katex_unavailable", "error": str(exc)[:160]}
    if proc.returncode == 0:
        return {"ok": True, "engine": "katex_mhchem", "error": ""}
    error = (proc.stderr or proc.stdout).strip()
    if "Cannot find module 'katex'" in error or 'Cannot find module "katex"' in error:
        return {"ok": static["ok"], "engine": "static_katex_unavailable", "error": "katex_module_not_installed"}
    return {"ok": False, "engine": "katex_mhchem", "error": error[:240]}


def strip_math_delimiters(latex: str) -> str:
    text = str(latex or "").strip()
    wrappers = [(r"\(", r"\)"), (r"\[", r"\]"), ("$", "$"), ("$$", "$$")]
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
    return text


def latex_consistency(low: dict[str, Any], high: dict[str, Any]) -> dict[str, Any]:
    a = re.sub(r"\s+", "", low.get("latex") or "")
    b = re.sub(r"\s+", "", high.get("latex") or "")
    return {"consistent": bool(a and b and a == b), "low_latex_len": len(a), "high_latex_len": len(b)}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = {normalize_compare_text(x) for x in a if normalize_compare_text(x)}
    sb = {normalize_compare_text(x) for x in b if normalize_compare_text(x)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalize_compare_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).strip()


def numeric_tokens(values: Iterable[str]) -> set[str]:
    nums: set[str] = set()
    for value in values:
        nums.update(re.findall(r"[-+]?\d+(?:\.\d+)?", str(value)))
    return nums


def transcript_consistency(low: dict[str, Any], high: dict[str, Any]) -> dict[str, Any]:
    same_type = low.get("fine_type") == high.get("fine_type")
    text_jaccard = jaccard(low.get("text_in_image") or [], high.get("text_in_image") or [])
    low_nums = numeric_tokens(low.get("data_points") or [])
    high_nums = numeric_tokens(high.get("data_points") or [])
    numeric_conflict = bool(low_nums and high_nums and not (low_nums <= high_nums or high_nums <= low_nums))
    consistent = bool(same_type and text_jaccard >= 0.85 and not numeric_conflict)
    diff_summary = []
    if not same_type:
        diff_summary.append(f"fine_type {low.get('fine_type')} vs {high.get('fine_type')}")
    if text_jaccard < 0.85:
        diff_summary.append(f"text_jaccard={text_jaccard:.2f}")
    if numeric_conflict:
        diff_summary.append("data_points_numeric_conflict")
    return {
        "consistent": consistent,
        "fine_type_same": same_type,
        "text_jaccard": text_jaccard,
        "numeric_conflict": numeric_conflict,
        "diff_summary": "; ".join(diff_summary),
    }


def merge_transcript(low: dict[str, Any], high: dict[str, Any]) -> dict[str, Any]:
    merged = dict(low)
    for key in ["elements", "text_in_image", "data_points", "uncertain"]:
        values: list[str] = []
        seen: set[str] = set()
        for source in [low, high]:
            for value in source.get(key) or []:
                norm = normalize_compare_text(value)
                if norm not in seen:
                    seen.add(norm)
                    values.append(str(value))
        merged[key] = values
    merged["confidence"] = min(coerce_confidence(low.get("confidence"), 0.0), coerce_confidence(high.get("confidence"), 0.0))
    return merged


def find_leak_hits(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return sorted(set(match.group(0) for match in LEAK_RE.finditer(text)))


def assign_pool(consistent: bool, min_confidence: float, fine_type: str, diff_summary: str) -> dict[str, str]:
    if fine_type in {"broken_image", "icon_or_noise"}:
        return {"pool": "manual_queue", "reason": f"fine_type:{fine_type}"}
    if consistent and min_confidence >= 0.75:
        return {"pool": "ai_seed", "reason": "consistent_and_confident"}
    if fine_type and fine_type not in {"broken_image", "icon_or_noise"}:
        return {"pool": "display_only", "reason": diff_summary or "minor_diff_or_low_confidence"}
    return {"pool": "manual_queue", "reason": diff_summary or "failed"}


def make_transcript_row(
    asset_hash: str,
    fine_type: str,
    merged: dict[str, Any],
    run_low: dict[str, Any],
    run_high: dict[str, Any],
    consistency: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    min_conf = min(coerce_confidence(run_low.get("confidence"), 0.0), coerce_confidence(run_high.get("confidence"), 0.0))
    pool = assign_pool(bool(consistency.get("consistent")), min_conf, fine_type, str(consistency.get("diff_summary") or ""))
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_hash": asset_hash,
        "asset_class": "illustration",
        "fine_type": fine_type,
        "summary": merged.get("summary", ""),
        "elements": merged.get("elements", []),
        "text_in_image": merged.get("text_in_image", []),
        "data_points": merged.get("data_points", []),
        "uncertain": merged.get("uncertain", []),
        "confidence": merged.get("confidence", 0.0),
        "runs": {"temperature_0_1": run_low, "temperature_0_4": run_high},
        "consistency": consistency,
        "pool": pool["pool"],
        "pool_reason": pool["reason"],
        "metadata": metadata,
    }


def make_formula_row(
    asset_hash: str,
    low: dict[str, Any],
    high: dict[str, Any],
    consistency: dict[str, Any],
    compile_result: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    chosen = low if low.get("latex") else high
    latex = strip_math_delimiters(str(chosen.get("latex", "")))
    min_conf = min(coerce_confidence(low.get("confidence"), 0.0), coerce_confidence(high.get("confidence"), 0.0))
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_hash": asset_hash,
        "asset_class": "formula_image",
        "latex": latex,
        "runs": {"temperature_0_1": low, "temperature_0_4": high},
        "consistency": consistency,
        "compile_result": compile_result,
        "confidence": min_conf,
        "latex_status": "passed" if compile_result.get("ok") else "failed",
        "metadata": metadata,
    }


def image_phash(path: Path) -> int | None:
    try:
        from PIL import Image
        import numpy as np

        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=float)
            n = 32
            k = np.arange(8).reshape((8, 1))
            i = np.arange(n).reshape((1, n))
            basis = np.cos((math.pi / n) * (i + 0.5) * k)
            coeffs = basis @ pixels @ basis.T
            low = coeffs[:8, :8].flatten()
            median = float(np.median(low[1:]))
            value = 0
            for idx, coeff in enumerate(low):
                if idx == 0:
                    continue
                if coeff >= median:
                    value |= 1 << idx
            return value
    except ImportError:
        try:
            from PIL import Image

            with Image.open(path) as image:
                image = image.convert("L").resize((8, 8))
                pixels = list(image.getdata())
                avg = sum(pixels) / len(pixels)
                value = 0
                for idx, pixel in enumerate(pixels):
                    if pixel >= avg:
                        value |= 1 << idx
                return value
        except Exception:
            if path.suffix.lower() != ".png":
                return None
            try:
                from scripts.build_ws2_asset_manifest import png_decode_rows

                width, height, color_type, bpp, rows = png_decode_rows(path.read_bytes())
                if width <= 0 or height <= 0:
                    return None
                pixels: list[int] = []
                for y in range(8):
                    src_y = min(height - 1, int((y + 0.5) * height / 8))
                    row = rows[src_y]
                    for x in range(8):
                        src_x = min(width - 1, int((x + 0.5) * width / 8))
                        px = row[src_x * bpp : (src_x + 1) * bpp]
                        if color_type == 0:
                            gray = px[0]
                        elif color_type == 2:
                            gray = (299 * px[0] + 587 * px[1] + 114 * px[2]) // 1000
                        elif color_type == 4:
                            alpha = px[1]
                            gray = int(px[0] * alpha / 255 + 255 * (1 - alpha / 255))
                        elif color_type == 6:
                            alpha = px[3]
                            rgb_gray = (299 * px[0] + 587 * px[1] + 114 * px[2]) // 1000
                            gray = int(rgb_gray * alpha / 255 + 255 * (1 - alpha / 255))
                        else:
                            return None
                        pixels.append(gray)
                avg = sum(pixels) / len(pixels)
                value = 0
                for idx, pixel in enumerate(pixels):
                    if pixel >= avg:
                        value |= 1 << idx
                return value
            except Exception:
                return None
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_near_duplicate_groups(rows: list[dict[str, Any]], out_root: Path) -> dict[str, str]:
    reps: list[tuple[str, int]] = []
    mapping: dict[str, str] = {}
    for row in rows:
        path = effective_png_for(row, out_root)
        if not path or not path.exists() or is_blank_image(path):
            continue
        ph = image_phash(path)
        if ph is None:
            continue
        assigned = None
        for rep_hash, rep_ph in reps:
            if hamming(ph, rep_ph) <= 4:
                assigned = rep_hash
                break
        if assigned is None:
            assigned = row["asset_hash"]
            reps.append((assigned, ph))
        mapping[row["asset_hash"]] = assigned
    return mapping


def call_vision(client: Any, image_path: Path, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    started = time.time()
    result = client.read_page(
        image_path=image_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        timeout=120.0,
        temperature=temperature,
    )
    result["elapsed_sec"] = round(time.time() - started, 3)
    return result


def cache_path(cache_dir: Path, asset_hash: str, track: str, temperature: float) -> Path:
    return cache_dir / track / f"{asset_hash}_t{str(temperature).replace('.', '_')}.json"


def run_cached_vision(
    client: Any,
    cache_dir: Path,
    row: dict[str, Any],
    image_path: Path,
    track: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    path = cache_path(cache_dir, row["asset_hash"], track, temperature)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raw = call_vision(client, image_path, system_prompt, user_prompt, temperature, max_tokens)
    data = {
        "asset_hash": row["asset_hash"],
        "temperature": temperature,
        "content": raw.get("content", ""),
        "cost_yuan": raw.get("cost_yuan", 0.0),
        "usage": raw.get("usage", {}),
        "elapsed_sec": raw.get("elapsed_sec"),
    }
    write_json(path, data)
    return data


def transcribe_formula_asset(
    row: dict[str, Any],
    out_root: Path,
    client: Any | None,
    cache_dir: Path,
    node_modules: Path | None,
    skip_vision: bool = False,
    cache_track: str = "formula",
) -> dict[str, Any]:
    asset_hash = row["asset_hash"]
    image_path = effective_png_for(row, out_root)
    metadata = {"ref_count": row.get("ref_count"), "question_count": row.get("question_count"), "image_path": str(image_path) if image_path else None}
    if not image_path or not image_path.exists() or is_blank_image(image_path):
        empty = {"latex": "", "confidence": 0.0, "uncertain": ["image_missing_or_blank"], "raw": {}}
        return make_formula_row(asset_hash, empty, empty, {"consistent": False, "reason": "image_missing_or_blank"}, {"ok": False, "engine": "none", "error": "image_missing_or_blank"}, metadata)
    if skip_vision:
        empty = {"latex": "", "confidence": 0.0, "uncertain": ["vision_skipped"], "raw": {}}
        return make_formula_row(asset_hash, empty, empty, {"consistent": False, "reason": "vision_skipped"}, {"ok": False, "engine": "none", "error": "vision_skipped"}, metadata)
    assert client is not None
    system_prompt, user_prompt = build_formula_prompts()
    raw_low = run_cached_vision(client, cache_dir, row, image_path, cache_track, system_prompt, user_prompt, 0.1, 1200)
    raw_high = run_cached_vision(client, cache_dir, row, image_path, cache_track, system_prompt, user_prompt, 0.4, 1200)
    low = normalize_formula_result(parse_json_response(str(raw_low.get("content") or "")))
    high = normalize_formula_result(parse_json_response(str(raw_high.get("content") or "")))
    consistency = latex_consistency(low, high)
    compile_result = validate_latex_katex(low.get("latex") or high.get("latex") or "", node_modules=node_modules)
    metadata["cost_yuan"] = float(raw_low.get("cost_yuan") or 0.0) + float(raw_high.get("cost_yuan") or 0.0)
    metadata["usage"] = {"temperature_0_1": raw_low.get("usage", {}), "temperature_0_4": raw_high.get("usage", {})}
    return make_formula_row(asset_hash, low, high, consistency, compile_result, metadata)


def transcribe_illustration_asset(
    row: dict[str, Any],
    out_root: Path,
    client: Any | None,
    cache_dir: Path,
    items_by_id: dict[str, dict[str, Any]],
    service_map: dict[str, Any],
    duplicate_rep: str | None = None,
    rep_result: dict[str, Any] | None = None,
    skip_vision: bool = False,
    cache_track: str = "illustration",
) -> dict[str, Any]:
    asset_hash = row["asset_hash"]
    image_path = effective_png_for(row, out_root)
    meta = service_metadata_for_asset(row, items_by_id, service_map)
    metadata = {
        "ref_count": row.get("ref_count"),
        "question_count": row.get("question_count"),
        "service_item_count": meta.get("service_item_count"),
        "top_kg": meta.get("top_kg"),
        "context_item_id": meta.get("context_item_id"),
        "context_stem_chars_used": min(300, len(meta.get("context_stem_text") or "")),
        "image_path": str(image_path) if image_path else None,
        "duplicate_representative": duplicate_rep,
    }
    if rep_result is not None and duplicate_rep and duplicate_rep != asset_hash:
        copied = json.loads(json.dumps(rep_result, ensure_ascii=False))
        copied["asset_hash"] = asset_hash
        copied["metadata"] = {**copied.get("metadata", {}), **metadata, "cache_source_asset_hash": duplicate_rep}
        return copied
    if not image_path or not image_path.exists() or is_blank_image(image_path):
        run = {"fine_type": "broken_image", "summary": "", "elements": [], "text_in_image": [], "data_points": [], "uncertain": ["image_missing_or_blank"], "confidence": 0.0}
        return make_transcript_row(asset_hash, "broken_image", run, run, run, {"consistent": False, "diff_summary": "image_missing_or_blank"}, metadata)
    if skip_vision:
        run = {"fine_type": "diagram_other", "summary": "", "elements": [], "text_in_image": [], "data_points": [], "uncertain": ["vision_skipped"], "confidence": 0.0}
        return make_transcript_row(asset_hash, "diagram_other", run, run, run, {"consistent": False, "diff_summary": "vision_skipped"}, metadata)
    assert client is not None
    context_item = {"stem_text": meta.get("context_stem_text") or ""}
    system_prompt, user_prompt = build_illustration_prompts("", context_item)
    raw_low = run_cached_vision(client, cache_dir, row, image_path, cache_track, system_prompt, user_prompt, 0.1, 2200)
    raw_high = run_cached_vision(client, cache_dir, row, image_path, cache_track, system_prompt, user_prompt, 0.4, 2200)
    low = normalize_transcript_result(parse_json_response(str(raw_low.get("content") or "")))
    high = normalize_transcript_result(parse_json_response(str(raw_high.get("content") or "")))
    consistency = transcript_consistency(low, high)
    merged = merge_transcript(low, high)
    metadata["cost_yuan"] = float(raw_low.get("cost_yuan") or 0.0) + float(raw_high.get("cost_yuan") or 0.0)
    metadata["usage"] = {"temperature_0_1": raw_low.get("usage", {}), "temperature_0_4": raw_high.get("usage", {})}
    return make_transcript_row(asset_hash, str(merged.get("fine_type") or "diagram_other"), merged, low, high, consistency, metadata)


def run_parallel(items: list[dict[str, Any]], fn: Any, workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [fn(row) for row in items]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, row): row for row in items}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "asset_hash": row.get("asset_hash"),
                        "error": str(exc),
                        "pool": "manual_queue",
                        "pool_reason": "pipeline_exception",
                    }
                )
    return sorted(results, key=lambda r: str(r.get("asset_hash") or ""))


def select_rows(rows: list[dict[str, Any]], asset_hashes: set[str] | None = None, asset_class: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if asset_hashes is not None and row.get("asset_hash") not in asset_hashes:
            continue
        if asset_class is not None and row.get("asset_class") != asset_class:
            continue
        selected.append(row)
        if limit and len(selected) >= limit:
            break
    return selected


def run_formula_track(
    manifest_rows: list[dict[str, Any]],
    out_root: Path,
    client: Any | None,
    cache_dir: Path,
    workers: int,
    limit: int | None,
    asset_hashes: set[str] | None = None,
    out_path: Path | None = None,
    skip_vision: bool = False,
) -> list[dict[str, Any]]:
    rows = select_rows(manifest_rows, asset_hashes, "formula_image", limit)
    node_modules = default_node_modules(out_root)
    fn = lambda row: transcribe_formula_asset(row, out_root, client, cache_dir, node_modules, skip_vision=skip_vision)
    results = run_parallel(rows, fn, workers)
    target = out_path or out_root / "formula_latex" / "formula_latex_candidates.jsonl"
    write_jsonl(target, results)
    failed = [row for row in results if row.get("latex_status") != "passed"]
    write_jsonl(target.parent / "formula_latex_failures.jsonl", failed)
    write_formula_preview(target.parent / "formula_latex_preview_top200.html", results[:200])
    return results


def run_illustration_track(
    manifest_rows: list[dict[str, Any]],
    out_root: Path,
    client: Any | None,
    cache_dir: Path,
    workers: int,
    limit: int | None,
    asset_hashes: set[str] | None = None,
    out_path: Path | None = None,
    skip_vision: bool = False,
) -> list[dict[str, Any]]:
    rows = select_rows(manifest_rows, asset_hashes, "illustration", limit)
    items_by_id = load_service_items()
    service_map = json.loads(SERVICE_MAP_PATH.read_text(encoding="utf-8")) if SERVICE_MAP_PATH.exists() else {}
    duplicate_map = build_near_duplicate_groups(rows, out_root)
    rep_results: dict[str, dict[str, Any]] = {}
    ordered = sorted(rows, key=lambda row: (duplicate_map.get(row["asset_hash"], row["asset_hash"]) != row["asset_hash"], row["asset_hash"]))

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        rep = duplicate_map.get(row["asset_hash"])
        if rep and rep != row["asset_hash"] and rep in rep_results:
            return transcribe_illustration_asset(row, out_root, client, cache_dir, items_by_id, service_map, rep, rep_results[rep], skip_vision=skip_vision)
        result = transcribe_illustration_asset(row, out_root, client, cache_dir, items_by_id, service_map, rep, None, skip_vision=skip_vision)
        if rep == row["asset_hash"] or not rep:
            rep_results[row["asset_hash"]] = result
        return result

    # Duplicate reuse is stateful, so keep deterministic sequential order.
    if workers > 1 and not skip_vision:
        representative_rows = [row for row in ordered if duplicate_map.get(row["asset_hash"], row["asset_hash"]) == row["asset_hash"]]
        rep_fn = lambda row: transcribe_illustration_asset(row, out_root, client, cache_dir, items_by_id, service_map, duplicate_map.get(row["asset_hash"]), None, skip_vision=skip_vision)
        rep_list = run_parallel(representative_rows, rep_fn, workers)
        rep_results.update({row["asset_hash"]: row for row in rep_list})
        results = []
        for row in ordered:
            rep = duplicate_map.get(row["asset_hash"], row["asset_hash"])
            if rep != row["asset_hash"] and rep in rep_results:
                results.append(transcribe_illustration_asset(row, out_root, client, cache_dir, items_by_id, service_map, rep, rep_results[rep], skip_vision=skip_vision))
            else:
                results.append(rep_results.get(row["asset_hash"]) or run_one(row))
    else:
        results = [run_one(row) for row in ordered]
    results = sorted(results, key=lambda r: str(r.get("asset_hash") or ""))
    target = out_path or out_root / "transcripts" / "transcript_candidates.jsonl"
    write_jsonl(target, results)
    dup_rows = [{"asset_hash": k, "representative_asset_hash": v} for k, v in sorted(duplicate_map.items()) if k != v]
    write_jsonl(target.parent / "near_duplicate_groups.jsonl", dup_rows)
    return results


def write_formula_preview(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "<!doctype html><meta charset='utf-8'><title>Batch8 Formula Preview</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px;vertical-align:top} code{white-space:pre-wrap}</style>",
        "<h1>Top Formula Candidates</h1><table><tr><th>asset</th><th>status</th><th>latex</th><th>confidence</th></tr>",
    ]
    for row in rows:
        lines.append(
            f"<tr><td>{row.get('asset_hash','')[:12]}</td><td>{row.get('latex_status')}</td>"
            f"<td><code>{html_escape(str(row.get('latex') or ''))}</code></td><td>{row.get('confidence')}</td></tr>"
        )
    lines.append("</table>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pool_and_leak(formula_rows: list[dict[str, Any]], transcript_rows: list[dict[str, Any]], out_root: Path) -> dict[str, Any]:
    pooling_dir = out_root / "pooling"
    assignments: list[dict[str, Any]] = []
    leak_rejected: list[dict[str, Any]] = []
    kept_transcripts: list[dict[str, Any]] = []
    for row in transcript_rows:
        hits = find_leak_hits({k: row.get(k) for k in ["summary", "elements", "text_in_image", "data_points", "uncertain"]})
        if hits:
            leak_row = {"asset_hash": row.get("asset_hash"), "leak_hits": hits, "row": row}
            leak_rejected.append(leak_row)
            assignments.append({"asset_hash": row.get("asset_hash"), "asset_class": row.get("asset_class"), "pool": "leak_rejected", "reason": "leak_pattern_rejected"})
        else:
            kept_transcripts.append(row)
            assignments.append(
                {
                    "asset_hash": row.get("asset_hash"),
                    "asset_class": row.get("asset_class"),
                    "pool": row.get("pool"),
                    "reason": row.get("pool_reason"),
                    "diff_summary": (row.get("consistency") or {}).get("diff_summary"),
                    "top_kg": (row.get("metadata") or {}).get("top_kg"),
                    "service_item_count": (row.get("metadata") or {}).get("service_item_count"),
                }
            )
    if leak_rejected:
        transcript_path = out_root / "transcripts" / "transcript_candidates.jsonl"
        write_jsonl(transcript_path, kept_transcripts)
    for row in formula_rows:
        pool = "formula_latex" if row.get("latex_status") == "passed" else "manual_queue"
        assignments.append(
            {
                "asset_hash": row.get("asset_hash"),
                "asset_class": row.get("asset_class"),
                "pool": pool,
                "reason": "compile_passed" if pool == "formula_latex" else (row.get("compile_result") or {}).get("error"),
                "service_item_count": (row.get("metadata") or {}).get("service_item_count"),
            }
        )
    write_jsonl(pooling_dir / "pool_assignment.jsonl", assignments)
    write_jsonl(pooling_dir / "leak_rejected.jsonl", leak_rejected)
    manual = [row for row in assignments if row.get("pool") in {"manual_queue", "leak_rejected"}]
    manual.sort(key=lambda r: (kg_priority(r.get("top_kg") or []), -int(r.get("service_item_count") or 0), str(r.get("asset_hash"))))
    batches = []
    for idx in range(0, len(manual), 50):
        for row in manual[idx : idx + 50]:
            enriched = dict(row)
            enriched["review_status"] = "pending_user_or_claude"
            enriched["reviewer"] = ""
            enriched["batch_index"] = idx // 50 + 1
            batches.append(enriched)
    write_jsonl(pooling_dir / "manual_queue_batches.jsonl", batches)
    summary = {"pool_counts": dict(Counter(row.get("pool") for row in assignments)), "leak_rejected_count": len(leak_rejected), "manual_queue_count": len(manual)}
    write_json(pooling_dir / "pooling_summary.json", summary)
    return summary


def kg_priority(top_kg: list[str]) -> int:
    joined = " ".join(map(str, top_kg))
    priorities = ["化学平衡", "氧化还原", "速率", "工艺流程", "实验", "溶液"]
    for idx, needle in enumerate(priorities):
        if needle in joined:
            return idx
    return len(priorities)


def run_gold_blind(
    manifest_rows: list[dict[str, Any]],
    out_root: Path,
    client: Any | None,
    cache_dir: Path,
    workers: int,
    skip_vision: bool,
) -> list[dict[str, Any]]:
    gold_rows = load_jsonl(GOLD_LIST_PATH)
    gold_hashes = {row["asset_hash"] for row in gold_rows}
    formula_out = out_root / "gold_blind" / "gold_formula_outputs.jsonl"
    illustration_out = out_root / "gold_blind" / "gold_illustration_outputs.jsonl"
    formula_rows = run_formula_track(manifest_rows, out_root, client, cache_dir, workers, None, gold_hashes, formula_out, skip_vision)
    transcript_rows = run_illustration_track(manifest_rows, out_root, client, cache_dir, workers, None, gold_hashes, illustration_out, skip_vision)
    combined = sorted(formula_rows + transcript_rows, key=lambda r: str(r.get("asset_hash") or ""))
    write_jsonl(out_root / "gold_blind" / "gold_blind_outputs.jsonl", combined)
    missing = sorted(gold_hashes - {row.get("asset_hash") for row in combined})
    write_json(out_root / "gold_blind" / "gold_blind_summary.json", {"gold_count": len(gold_hashes), "outputs": len(combined), "missing": missing})
    return combined


def default_node_modules(out_root: Path) -> Path:
    for candidate in [out_root / "node_modules", OUT_ROOT / "node_modules", BATCH8_CANDIDATE_ROOT / "node_modules"]:
        if candidate.exists():
            return candidate
    return out_root / "node_modules"


def clone_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def has_batch8p1_relation_operator(latex: str) -> bool:
    text = str(latex or "")
    return "<=>" in text or "->" in text


def select_batch8p1_relation_targets(formula_candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in formula_candidate_rows if has_batch8p1_relation_operator(str(row.get("latex") or ""))]


def select_batch8p1_track_i_fallback_targets(
    formula_candidate_rows: list[dict[str, Any]], gold_failure_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in formula_candidate_rows:
        if row.get("latex_status") == "failed":
            targets.append(
                {
                    "asset_hash": row.get("asset_hash"),
                    "source_scope": "formula_latex_candidates_latex_status_failed",
                    "source_latex_status": row.get("latex_status"),
                    "source_latex": row.get("latex"),
                }
            )
    for row in gold_failure_rows:
        targets.append(
            {
                "asset_hash": row.get("asset_hash"),
                "source_scope": "gold_blind_formula_latex_failures_named",
                "source_latex_status": row.get("latex_status"),
                "source_latex": row.get("latex"),
            }
        )
    return targets


def load_batch8p1_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    formula_rows = load_jsonl(BATCH8_CANDIDATE_ROOT / "formula_latex" / "formula_latex_candidates.jsonl")
    gold_failure_rows = load_jsonl(BATCH8_CANDIDATE_ROOT / "gold_blind" / "formula_latex_failures.jsonl")
    manifest_rows = load_manifest()
    return manifest_rows, formula_rows, gold_failure_rows


def batch8p1_dirs(out_root: Path) -> None:
    for name in ["relation_fix", "trackI_fallback", "single_retry", "api_cache"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)


def transcript_payload_for_leak_scan(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in ["summary", "elements", "text_in_image", "data_points", "uncertain"]}


def filter_transcript_leaks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        hits = find_leak_hits(transcript_payload_for_leak_scan(row))
        if hits:
            rejected.append({"schema_version": SCHEMA_VERSION, "asset_hash": row.get("asset_hash"), "leak_hits": hits, "row": row})
        else:
            kept.append(row)
    return kept, rejected


def image_loading_diagnostic(row: dict[str, Any], out_root: Path) -> dict[str, Any]:
    path = effective_png_for(row, out_root)
    diagnostic: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "asset_hash": row.get("asset_hash"),
        "manifest_dimensions": row.get("dimensions"),
        "image_path": str(path) if path else None,
        "exists": bool(path and path.exists()),
    }
    if not path or not path.exists():
        return diagnostic
    try:
        diagnostic["size_bytes"] = path.stat().st_size
        diagnostic["sha256"] = sha256_file(path)
        diagnostic["extrema_min"] = image_extrema_min(path)
        diagnostic["is_blank_image"] = is_blank_image(path)
        try:
            from PIL import Image

            with Image.open(path) as image:
                diagnostic["decoded_dimensions"] = list(image.size)
                diagnostic["mode"] = image.mode
        except Exception as exc:
            diagnostic["pil_decode_error"] = str(exc)[:240]
    except Exception as exc:
        diagnostic["diagnostic_error"] = str(exc)[:240]
    return diagnostic


def run_batch8p1_relation_fix(
    manifest_rows: list[dict[str, Any]],
    formula_candidate_rows: list[dict[str, Any]],
    out_root: Path,
    client: Any | None,
    workers: int,
    skip_vision: bool,
) -> list[dict[str, Any]]:
    targets = select_batch8p1_relation_targets(formula_candidate_rows)
    if len(targets) != 574:
        raise RuntimeError(f"Batch8.1 relation target count must be 574, got {len(targets)}")
    manifest_by_hash = {row["asset_hash"]: row for row in manifest_rows}
    node_modules = default_node_modules(out_root)
    cache_dir = out_root / "api_cache" / "relation_fix"

    def run_one(source_row: dict[str, Any]) -> dict[str, Any]:
        asset_hash = source_row["asset_hash"]
        manifest_row = manifest_by_hash.get(asset_hash)
        if manifest_row is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "asset_hash": asset_hash,
                "batch8p1_task": "U_relation_fix",
                "original_latex": source_row.get("latex"),
                "error": "asset_hash_missing_from_manifest",
            }
        new_row = transcribe_formula_asset(
            manifest_row,
            out_root,
            client,
            cache_dir,
            node_modules,
            skip_vision=skip_vision,
            cache_track="formula_relation_refix_v1",
        )
        new_latex = str(new_row.get("latex") or "")
        original_latex = str(source_row.get("latex") or "")
        relation_after = has_batch8p1_relation_operator(new_latex)
        return {
            "schema_version": SCHEMA_VERSION,
            "asset_hash": asset_hash,
            "asset_class": "formula_image",
            "batch8p1_task": "U_relation_fix",
            "original_latex": original_latex,
            "new_latex": new_latex,
            "changed": new_latex != original_latex,
            "original_latex_status": source_row.get("latex_status"),
            "new_latex_status": new_row.get("latex_status"),
            "compile_result": new_row.get("compile_result"),
            "confidence": new_row.get("confidence"),
            "consistency": new_row.get("consistency"),
            "relation_operator_before": True,
            "relation_operator_after": relation_after,
            "relation_rewritten_to_equal": bool(not relation_after and "=" in new_latex),
            "relation_verified_by_refix": relation_after,
            "runs": new_row.get("runs"),
            "metadata": {**(new_row.get("metadata") or {}), "source": "batch8_formula_latex_candidates"},
        }

    results = run_parallel(targets, run_one, workers)
    write_jsonl(out_root / "relation_fix" / "relation_refix.jsonl", results)
    write_relation_diff_report(out_root / "relation_fix" / "relation_diff_report.md", results)
    return results


def write_relation_diff_report(path: Path, rows: list[dict[str, Any]]) -> None:
    after_relation = sum(1 for row in rows if row.get("relation_operator_after"))
    rewritten_to_equal = sum(1 for row in rows if row.get("relation_rewritten_to_equal"))
    changed = sum(1 for row in rows if row.get("changed"))
    passed = sum(1 for row in rows if row.get("new_latex_status") == "passed")
    lines = [
        "# Batch 8.1 Relation Refix Diff Report",
        "",
        f"- target rows: {len(rows)}",
        f"- before relation-token rows (`<=>` or `->`): {len(rows)}",
        f"- after relation-token rows (`<=>` or `->`): {after_relation}",
        f"- rows rewritten back to `=`: {rewritten_to_equal}",
        f"- changed rows: {changed}",
        f"- new compile passed: {passed}",
        f"- before rewrite rate: 100.00%",
        f"- after relation-token rate: {(after_relation / len(rows) * 100) if rows else 0:.2f}%",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch8p1_track_i_fallback(
    manifest_rows: list[dict[str, Any]],
    formula_candidate_rows: list[dict[str, Any]],
    gold_failure_rows: list[dict[str, Any]],
    out_root: Path,
    client: Any | None,
    workers: int,
    skip_vision: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targets = select_batch8p1_track_i_fallback_targets(formula_candidate_rows, gold_failure_rows)
    if len(targets) != 23:
        raise RuntimeError(f"Batch8.1 Track-I fallback source row count must be 23, got {len(targets)}")
    manifest_by_hash = {row["asset_hash"]: row for row in manifest_rows}
    cache_dir = out_root / "api_cache" / "trackI_fallback"
    items_by_id = load_service_items()
    service_map = json.loads(SERVICE_MAP_PATH.read_text(encoding="utf-8")) if SERVICE_MAP_PATH.exists() else {}
    unique_targets: dict[str, dict[str, Any]] = {}
    for target in targets:
        unique_targets.setdefault(str(target.get("asset_hash")), target)

    def run_unique(target: dict[str, Any]) -> dict[str, Any]:
        asset_hash = str(target.get("asset_hash"))
        manifest_row = manifest_by_hash.get(asset_hash)
        if manifest_row is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "asset_hash": asset_hash,
                "batch8p1_task": "V_trackI_fallback",
                "error": "asset_hash_missing_from_manifest",
                "metadata": {"source_scope": target.get("source_scope")},
            }
        result = transcribe_illustration_asset(
            manifest_row,
            out_root,
            client,
            cache_dir,
            items_by_id,
            service_map,
            skip_vision=skip_vision,
            cache_track="trackI_fallback_v1",
        )
        result["batch8p1_task"] = "V_trackI_fallback"
        result["metadata"] = {**(result.get("metadata") or {}), "source_scope": target.get("source_scope"), "source_latex_status": target.get("source_latex_status")}
        return result

    unique_results = run_parallel(list(unique_targets.values()), run_unique, workers)
    unique_by_hash = {row.get("asset_hash"): row for row in unique_results}
    emitted: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    for index, target in enumerate(targets, start=1):
        asset_hash = str(target.get("asset_hash"))
        base = clone_jsonable(unique_by_hash.get(asset_hash) or {"schema_version": SCHEMA_VERSION, "asset_hash": asset_hash, "error": "missing_unique_result"})
        meta = dict(base.get("metadata") or {})
        if asset_hash in seen_assets:
            meta["cost_yuan_original_reused"] = meta.get("cost_yuan", 0.0)
            meta["cost_yuan"] = 0.0
            meta["usage"] = {}
            meta["reused_from_same_asset_fallback"] = True
        seen_assets.add(asset_hash)
        meta.update({"batch8p1_target_index": index, "source_scope": target.get("source_scope"), "source_latex_status": target.get("source_latex_status")})
        base["metadata"] = meta
        base["batch8p1_task"] = "V_trackI_fallback"
        base["batch8p1_source"] = {k: target.get(k) for k in ["source_scope", "source_latex_status", "source_latex"]}
        emitted.append(base)

    kept, rejected = filter_transcript_leaks(emitted)
    write_jsonl(out_root / "trackI_fallback" / "fallback_targets.jsonl", targets)
    write_jsonl(out_root / "trackI_fallback" / "fallback_transcripts.jsonl", kept)
    write_jsonl(out_root / "trackI_fallback" / "leak_rejected.jsonl", rejected)
    return kept, rejected, targets


def run_batch8p1_single_retry(
    manifest_rows: list[dict[str, Any]],
    out_root: Path,
    client: Any | None,
    skip_vision: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    matches = [row for row in manifest_rows if str(row.get("asset_hash") or "").startswith(BATCH8P1_SINGLE_RETRY_PREFIX)]
    if len(matches) != 1:
        raise RuntimeError(f"Batch8.1 single retry target must resolve to 1 row, got {len(matches)}")
    row = matches[0]
    cache_dir = out_root / "api_cache" / "single_retry"
    items_by_id = load_service_items()
    service_map = json.loads(SERVICE_MAP_PATH.read_text(encoding="utf-8")) if SERVICE_MAP_PATH.exists() else {}
    diagnostic = image_loading_diagnostic(row, out_root)
    try:
        result = transcribe_illustration_asset(
            row,
            out_root,
            client,
            cache_dir,
            items_by_id,
            service_map,
            skip_vision=skip_vision,
            cache_track="single_retry_v1",
        )
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "asset_hash": row.get("asset_hash"),
            "batch8p1_task": "W_single_retry",
            "fine_type": "broken_image",
            "summary": "",
            "elements": [],
            "text_in_image": [],
            "data_points": [],
            "uncertain": ["pipeline_exception"],
            "confidence": 0.0,
            "error": str(exc),
            "metadata": {},
        }
    result["batch8p1_task"] = "W_single_retry"
    result["metadata"] = {**(result.get("metadata") or {}), "single_retry_prefix": BATCH8P1_SINGLE_RETRY_PREFIX}
    if result.get("fine_type") == "broken_image" or result.get("error"):
        result["loading_diagnostic"] = diagnostic
    kept, rejected = filter_transcript_leaks([result])
    write_jsonl(out_root / "single_retry" / "single_retry.jsonl", kept)
    write_jsonl(out_root / "single_retry" / "leak_rejected.jsonl", rejected)
    write_json(out_root / "single_retry" / "loading_diagnostic.json", diagnostic)
    return kept, rejected, diagnostic


def run_batch8p1(out_root: Path, workers: int, skip_vision: bool) -> dict[str, Any]:
    started = time.time()
    batch8p1_dirs(out_root)
    manifest_rows, formula_rows, gold_failure_rows = load_batch8p1_inputs()
    client = None if skip_vision else build_vision_client()
    relation_rows = run_batch8p1_relation_fix(manifest_rows, formula_rows, out_root, client, workers, skip_vision)
    fallback_rows, fallback_leaks, fallback_targets = run_batch8p1_track_i_fallback(
        manifest_rows, formula_rows, gold_failure_rows, out_root, client, workers, skip_vision
    )
    single_rows, single_leaks, single_diagnostic = run_batch8p1_single_retry(manifest_rows, out_root, client, skip_vision)
    report = write_batch8p1_report(
        out_root,
        relation_rows,
        fallback_rows,
        fallback_leaks,
        fallback_targets,
        single_rows,
        single_leaks,
        single_diagnostic,
        started,
        skip_vision,
    )
    return report


def write_batch8p1_report(
    out_root: Path,
    relation_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    fallback_leaks: list[dict[str, Any]],
    fallback_targets: list[dict[str, Any]],
    single_rows: list[dict[str, Any]],
    single_leaks: list[dict[str, Any]],
    single_diagnostic: dict[str, Any],
    started_at: float,
    skip_vision: bool,
) -> dict[str, Any]:
    after_relation = sum(1 for row in relation_rows if row.get("relation_operator_after"))
    rewritten_to_equal = sum(1 for row in relation_rows if row.get("relation_rewritten_to_equal"))
    relation_changed = sum(1 for row in relation_rows if row.get("changed"))
    relation_passed = sum(1 for row in relation_rows if row.get("new_latex_status") == "passed")
    fallback_unique_assets = len({row.get("asset_hash") for row in fallback_targets})
    fallback_pool_counts = Counter(row.get("pool") for row in fallback_rows)
    single_fine_type = (single_rows[0].get("fine_type") if single_rows else "leak_rejected") if not single_leaks else "leak_rejected"
    total_cost = round(sum_cost(relation_rows) + sum_cost(fallback_rows) + sum_cost(single_rows), 4)
    report = {
        "relation_target_rows": len(relation_rows),
        "relation_after_operator_rows": after_relation,
        "relation_rewritten_to_equal_rows": rewritten_to_equal,
        "relation_changed_rows": relation_changed,
        "relation_compile_passed": relation_passed,
        "fallback_source_rows": len(fallback_targets),
        "fallback_unique_assets": fallback_unique_assets,
        "fallback_output_rows": len(fallback_rows),
        "fallback_leak_rejected": len(fallback_leaks),
        "single_retry_rows": len(single_rows),
        "single_retry_leak_rejected": len(single_leaks),
        "single_retry_fine_type": single_fine_type,
        "total_measured_cost_yuan": total_cost,
    }
    lines = [
        "# Batch 8.1 WS2 Patch Report",
        "",
        "## Run Mode",
        "",
        f"- output_root: `{out_root}`",
        f"- skip_vision: {skip_vision}",
        f"- elapsed_sec: {round(time.time() - started_at, 1)}",
        "",
        "## Task U - Relation Symbol Refix",
        "",
        f"- target rows: {len(relation_rows)}",
        f"- before relation-token rows (`<=>` or `->`): {len(relation_rows)}",
        f"- after relation-token rows (`<=>` or `->`): {after_relation}",
        f"- rows rewritten back to `=`: {rewritten_to_equal}",
        f"- changed rows: {relation_changed}",
        f"- new compile passed: {relation_passed}",
        f"- before relation-token rate: 100.00%",
        f"- after relation-token rate: {(after_relation / len(relation_rows) * 100) if relation_rows else 0:.2f}%",
        "",
        "## Task V - Track-I Fallback",
        "",
        f"- source target rows: {len(fallback_targets)}",
        f"- unique assets: {fallback_unique_assets}",
        f"- fallback transcript rows written: {len(fallback_rows)}",
        f"- leak rejected rows: {len(fallback_leaks)}",
    ]
    for pool, count in sorted(fallback_pool_counts.items()):
        lines.append(f"- fallback pool `{pool}`: {count}")
    lines.extend(
        [
            "",
            "## Task W - Single Retry",
            "",
            f"- retry rows written: {len(single_rows)}",
            f"- leak rejected rows: {len(single_leaks)}",
            f"- fine_type: {single_fine_type}",
            f"- diagnostic exists: {bool(single_diagnostic.get('exists'))}",
            f"- diagnostic dimensions: {single_diagnostic.get('decoded_dimensions') or single_diagnostic.get('manifest_dimensions')}",
            "",
            "## Cost",
            "",
            f"- relation cost yuan: {sum_cost(relation_rows)}",
            f"- fallback cost yuan: {sum_cost(fallback_rows)}",
            f"- single retry cost yuan: {sum_cost(single_rows)}",
            f"- total measured cost yuan: {total_cost}",
            "",
            "## Hard Gates",
            "",
            "- schema_version target: `ws2_transcript_v1`",
            "- reviewer fields are not written by Batch 8.1 runner",
            "- candidate and official data directories are read-only inputs",
            "- leak gate files: `trackI_fallback/leak_rejected.jsonl`, `single_retry/leak_rejected.jsonl`",
        ]
    )
    (out_root / "BATCH8P1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_root / "BATCH8P1_SUMMARY.json", report)
    return report


def sum_cost(rows: Iterable[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        meta = row.get("metadata") or {}
        try:
            total += float(meta.get("cost_yuan") or 0.0)
        except Exception:
            pass
    return round(total, 4)


def write_batch_report(
    out_root: Path,
    manifest_rows: list[dict[str, Any]],
    repair_summary: dict[str, Any] | None,
    formula_rows: list[dict[str, Any]],
    transcript_rows: list[dict[str, Any]],
    pooling_summary: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    started_at: float,
    skip_vision: bool,
) -> None:
    if repair_summary is None:
        existing_repair_summary = out_root / "asset_repair" / "bad_asset_detection_summary.json"
        if existing_repair_summary.exists():
            repair_summary = {"summary": json.loads(existing_repair_summary.read_text(encoding="utf-8"))}
    formula_compile_ok = sum(1 for row in formula_rows if row.get("latex_status") == "passed")
    transcript_pool_counts = Counter(row.get("pool") for row in transcript_rows)
    consistency_counts = Counter(bool((row.get("consistency") or {}).get("consistent")) for row in transcript_rows)
    lines = [
        "# Batch 8 WS2 Report",
        "",
        "## Run Mode",
        "",
        f"- skip_vision: {skip_vision}",
        f"- elapsed_sec: {round(time.time() - started_at, 1)}",
        f"- output_root: `{out_root}`",
        "",
        "## Inputs",
        "",
        f"- manifest rows: {len(manifest_rows)}",
        f"- formula_image rows: {sum(1 for row in manifest_rows if row.get('asset_class') == 'formula_image')}",
        f"- illustration rows: {sum(1 for row in manifest_rows if row.get('asset_class') == 'illustration')}",
        "",
        "## O Asset Repair",
        "",
    ]
    if repair_summary:
        s = repair_summary["summary"]
        lines.extend(
            [
                f"- detected blank: {s['detected_blank_count']}",
                f"- detected missing: {s['detected_missing_count']}",
                f"- bad total: {s['detected_bad_total']}",
                f"- repaired: {s['repaired_count']}",
                f"- unrepairable: {s['unrepairable_count']}",
            ]
        )
    else:
        lines.append("- not run in this invocation")
    lines.extend(
        [
            "",
            "## P Formula Track",
            "",
            f"- outputs: {len(formula_rows)}",
            f"- compile/static pass: {formula_compile_ok}",
            f"- pass rate: {(formula_compile_ok / len(formula_rows) * 100) if formula_rows else 0:.2f}%",
            f"- double-run exact consistency: {sum(1 for row in formula_rows if (row.get('consistency') or {}).get('consistent'))}",
            "",
            "## Q Illustration Track",
            "",
            f"- outputs: {len(transcript_rows)}",
            f"- consistency true: {consistency_counts.get(True, 0)}",
            f"- consistency false: {consistency_counts.get(False, 0)}",
        ]
    )
    for pool, count in sorted(transcript_pool_counts.items()):
        lines.append(f"- transcript pool `{pool}`: {count}")
    lines.extend(["", "## R Pooling And Leak Gate", ""])
    for pool, count in sorted((pooling_summary.get("pool_counts") or {}).items()):
        lines.append(f"- `{pool}`: {count}")
    lines.extend(
        [
            f"- leak rejected: {pooling_summary.get('leak_rejected_count', 0)}",
            f"- manual queue rows: {pooling_summary.get('manual_queue_count', 0)}",
            "",
            "## S Gold Blind",
            "",
            f"- gold outputs: {len(gold_rows)}",
            "",
            "## Cost",
            "",
            f"- formula cost yuan: {sum_cost(formula_rows)}",
            f"- illustration cost yuan: {sum_cost(transcript_rows)}",
            f"- gold cost yuan: {sum_cost(gold_rows)}",
            f"- total measured cost yuan: {round(sum_cost(formula_rows) + sum_cost(transcript_rows) + sum_cost(gold_rows), 4)}",
            "",
            "## Residuals",
            "",
            "- Manual/rejected rows are in `pooling/manual_queue_batches.jsonl` with empty reviewer and `pending_user_or_claude` status.",
            "- Repair failures are in `asset_repair/unrepairable.jsonl`.",
            "- Gold blind outputs use the same formula/illustration runner functions as full assets; no gold-specific prompt branch is present.",
        ]
    )
    (out_root / "BATCH8_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_dirs(out_root: Path) -> None:
    for name in ["asset_repair", "formula_latex", "transcripts", "pooling", "gold_blind", "api_cache"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repair-only", action="store_true")
    parser.add_argument("--transcribe-only", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--batch8p1", action="store_true", help="Run the limited Batch 8.1 WS2 patch tasks U/V/W.")
    args = parser.parse_args(argv)

    started = time.time()
    out_root = BATCH8P1_OUT_ROOT if args.batch8p1 and args.out_root == OUT_ROOT else args.out_root
    if args.batch8p1:
        run_batch8p1(out_root, args.workers, args.skip_vision)
        return 0
    ensure_dirs(out_root)
    manifest_rows = load_manifest()
    repair_summary = None
    if not args.transcribe_only:
        repair_summary = repair_bad_assets(manifest_rows, out_root)
    if args.repair_only:
        write_batch_report(out_root, manifest_rows, repair_summary, [], [], {"pool_counts": {}, "leak_rejected_count": 0, "manual_queue_count": 0}, [], started, args.skip_vision)
        return 0

    client = None if args.skip_vision else build_vision_client()
    cache_dir = out_root / "api_cache"
    formula_rows = run_formula_track(manifest_rows, out_root, client, cache_dir, args.workers, args.limit, skip_vision=args.skip_vision)
    transcript_rows = run_illustration_track(manifest_rows, out_root, client, cache_dir, args.workers, args.limit, skip_vision=args.skip_vision)
    pooling_summary = pool_and_leak(formula_rows, transcript_rows, out_root)
    gold_rows: list[dict[str, Any]] = []
    if not args.skip_gold:
        gold_rows = run_gold_blind(manifest_rows, out_root, client, cache_dir, args.workers, args.skip_vision)
    write_batch_report(out_root, manifest_rows, repair_summary, formula_rows, transcript_rows, pooling_summary, gold_rows, started, args.skip_vision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
