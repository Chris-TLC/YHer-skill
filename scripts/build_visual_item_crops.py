#!/usr/bin/env python3
"""Build conservative item-level crop candidates for visual eval items.

Default mode is dry-run. Crops are evidence candidates only; they do not upgrade
an item to strong profile evidence by themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

SKILL_DIR = Path(__file__).parent.parent
TOOLS_DIR = SKILL_DIR.parent
DEFAULT_EVAL_SET = SKILL_DIR / "data" / "evals" / "visual_item_eval_set.jsonl"
DEFAULT_OUT = Path("/tmp/yher_visual_item_crops.jsonl")
DEFAULT_CROP_DIR = Path("/tmp/yher_visual_item_crops")
DEFAULT_DOC_CACHE = SKILL_DIR / "data" / ".doc_to_pdf_cache"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。、“”‘’：；？！,.!?;:()（）\[\]【】{}<>《》_—\-]+", "", text)
    return text


def source_pdf_path(row: dict[str, Any], doc_cache_dir: Path = DEFAULT_DOC_CACHE) -> Path | None:
    source = Path(str(row.get("source_path") or ""))
    if not source.exists():
        return None
    if source.suffix.lower() == ".pdf":
        return source
    if source.suffix.lower() in {".doc", ".docx"}:
        digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
        pdf = doc_cache_dir / f"{digest}.pdf"
        return pdf if pdf.exists() else None
    return None


def import_fitz() -> Any:
    try:
        import fitz  # type: ignore

        return fitz
    except ModuleNotFoundError as exc:
        raise RuntimeError("missing_pymupdf") from exc


def stem_anchors(row: dict[str, Any]) -> list[str]:
    stem = normalize_text(row.get("stem", ""))
    anchors: list[str] = []
    for width in (42, 30, 20, 12):
        if len(stem) >= width:
            anchors.append(stem[:width])
    for anchor in row.get("visible_anchors") or []:
        norm = normalize_text(anchor)
        if len(norm) >= 8:
            anchors.append(norm[: min(30, len(norm))])
    # Preserve order while deduping.
    seen: set[str] = set()
    out: list[str] = []
    for anchor in anchors:
        if anchor and anchor not in seen:
            seen.add(anchor)
            out.append(anchor)
    return out


def find_anchor_rect(page: Any, row: dict[str, Any]) -> tuple[Any | None, str]:
    for anchor in stem_anchors(row):
        # Exact text search handles PDF text layer when the anchor survived
        # conversion. This intentionally avoids OCR guesses.
        rects = page.search_for(anchor)
        if rects:
            return rects[0], anchor

    # Fallback: short CJK anchors often fail exact search because formulas and
    # spacing differ. Use word coordinates and a normalized page string only to
    # locate an approximate vertical band.
    words = page.get_text("words")
    if not words:
        return None, ""
    ordered = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    page_norm = normalize_text("".join(str(w[4]) for w in ordered))
    for anchor in stem_anchors(row):
        probe = anchor[:12]
        if len(probe) < 6:
            continue
        idx = page_norm.find(probe)
        if idx < 0:
            continue
        consumed = 0
        for word in ordered:
            consumed += len(normalize_text(word[4]))
            if consumed >= idx:
                fitz = import_fitz()
                return fitz.Rect(word[:4]), probe
    return None, ""


def next_item_y(page: Any, current_row: dict[str, Any], page_rows: list[dict[str, Any]]) -> float | None:
    current_id = current_row.get("item_id")
    current_rect, _ = find_anchor_rect(page, current_row)
    if not current_rect:
        return None
    candidates: list[float] = []
    for row in page_rows:
        if row.get("item_id") == current_id:
            continue
        rect, _ = find_anchor_rect(page, row)
        if rect and rect.y0 > current_rect.y0 + 8:
            candidates.append(float(rect.y0))
    return min(candidates) if candidates else None


def fallback_crop_fraction(row: dict[str, Any]) -> float:
    category = row.get("category")
    if category in {"chart_curve", "process_flow", "experiment_device", "organic_structure", "crystal_cell", "electrochem_device"}:
        return 0.62
    return 0.34


def context_padding_for(row: dict[str, Any]) -> dict[str, Any]:
    stem = str(row.get("stem") or "")
    category = row.get("category")
    references_figure = any(marker in stem for marker in ["上图", "下图", "如下图", "如图", "图所示"])
    if category == "chart_curve" and references_figure:
        return {"top": 190.0, "bottom": 300.0, "blockers": ["referenced_figure_context"]}
    return {"top": 26.0, "bottom": 220.0, "blockers": []}


def crop_rect_for(page: Any, row: dict[str, Any], page_rows: list[dict[str, Any]]) -> tuple[Any | None, str, list[str]]:
    fitz = import_fitz()
    page_rect = page.rect
    anchor_rect, anchor = find_anchor_rect(page, row)
    blockers: list[str] = []
    if not anchor_rect:
        return None, "", ["anchor_not_found_in_pdf_text"]

    next_y = next_item_y(page, row, page_rows)
    padding = context_padding_for(row)
    blockers.extend(padding["blockers"])
    top = max(0.0, float(anchor_rect.y0) - float(padding["top"]))
    if next_y is not None:
        bottom = min(float(page_rect.y1), max(float(anchor_rect.y1) + 140.0, next_y - 8.0))
    else:
        # Conservative last-item crop: include enough context without pretending
        # we located a precise lower boundary.
        bottom = min(float(page_rect.y1), max(float(anchor_rect.y1) + float(padding["bottom"]), top + float(page_rect.height) * fallback_crop_fraction(row)))
        blockers.append("bottom_boundary_estimated")

    if bottom - top < 90:
        bottom = min(float(page_rect.y1), top + 180.0)
        blockers.append("crop_height_expanded")
    if bottom - top > float(page_rect.height) * 0.72:
        blockers.append("crop_too_large")

    rect = fitz.Rect(0.0, top, float(page_rect.x1), bottom)
    return rect, anchor, blockers


def render_crop(pdf_path: Path, page_index: int, rect: Any, out_path: Path) -> None:
    fitz = import_fitz()
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out_path)
    finally:
        doc.close()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def build_crops(
    eval_set_path: Path = DEFAULT_EVAL_SET,
    crop_dir: Path = DEFAULT_CROP_DIR,
    write_images: bool = False,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(Path(eval_set_path))
    if limit is not None:
        rows = rows[:limit]

    by_pdf_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    pdf_by_id: dict[str, Path] = {}
    for row in rows:
        pdf = source_pdf_path(row)
        if not pdf:
            continue
        page = int(row.get("page") or 0)
        if page <= 0:
            continue
        pdf_by_id[str(row.get("item_id"))] = pdf
        by_pdf_page.setdefault((str(pdf), page), []).append(row)

    fitz = import_fitz()
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        item_id = str(row.get("item_id") or "")
        pdf = pdf_by_id.get(item_id)
        page_num = int(row.get("page") or 0)
        blockers: list[str] = []
        crop_path = ""
        crop_hash = ""
        crop_tier = "missing"
        bbox = None
        anchor = ""
        if not pdf:
            blockers.append("source_pdf_unresolved")
        elif not row.get("page_image_path"):
            blockers.append("missing_page_image")
        elif page_num <= 0:
            blockers.append("missing_page_number")
        else:
            doc = fitz.open(pdf)
            try:
                if page_num > len(doc):
                    blockers.append("page_out_of_range")
                else:
                    page = doc[page_num - 1]
                    page_rows = by_pdf_page.get((str(pdf), page_num), [row])
                    rect, anchor, rect_blockers = crop_rect_for(page, row, page_rows)
                    blockers.extend(rect_blockers)
                    if rect:
                        bbox = [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]
                        out_file = crop_dir / f"{item_id}.png"
                        crop_path = str(out_file)
                        crop_tier = "item_crop_candidate"
                        if "crop_too_large" in blockers:
                            crop_tier = "page_region_candidate"
                        if write_images:
                            render_crop(pdf, page_num - 1, rect, out_file)
                            crop_hash = sha256_file(out_file)
            finally:
                doc.close()

        out_rows.append(
            {
                "item_id": item_id,
                "source_file": row.get("source_file", ""),
                "page": page_num or None,
                "category": row.get("category", ""),
                "page_image_path": row.get("page_image_path", ""),
                "crop_path": crop_path if write_images else "",
                "crop_hash": crop_hash,
                "crop_tier": crop_tier,
                "bbox_pdf_points": bbox,
                "anchor_used": anchor,
                "blocker_reasons": sorted(set(blockers)),
                "profile_evidence_allowed": False,
            }
        )

    summary = {
        "eval_items": len(rows),
        "write_images": write_images,
        "crop_candidates": sum(1 for row in out_rows if row["crop_tier"] in {"item_crop_candidate", "page_region_candidate"}),
        "item_crop_candidate": sum(1 for row in out_rows if row["crop_tier"] == "item_crop_candidate"),
        "page_region_candidate": sum(1 for row in out_rows if row["crop_tier"] == "page_region_candidate"),
        "missing": sum(1 for row in out_rows if row["crop_tier"] == "missing"),
        "with_blockers": sum(1 for row in out_rows if row["blocker_reasons"]),
        "by_category": {},
    }
    by_category: dict[str, int] = {}
    for row in out_rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
    summary["by_category"] = by_category
    return out_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build conservative visual item crop candidates.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write", action="store_true", help="Write crop images and JSONL output. Default is dry-run.")
    args = parser.parse_args()

    rows, summary = build_crops(
        eval_set_path=args.eval_set,
        crop_dir=args.crop_dir,
        write_images=args.write,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        write_jsonl(args.out, rows)
        print(f"WROTE {args.out}")
        print(f"WROTE {args.crop_dir}")
    else:
        print("DRY RUN: pass --write to write /tmp crop outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
