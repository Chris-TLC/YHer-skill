#!/usr/bin/env python3
"""
Build a conservative visual asset manifest for v3 PDF-derived chemistry items.

This script is read-only by default. It reads the v3 item/source outputs and
writes manifest files only when --write is passed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

SKILL_DIR = Path(__file__).parent.parent
TOOLS_DIR = SKILL_DIR.parent
DEFAULT_PDF_ITEMS = SKILL_DIR / "data" / "from_pdf" / "all_from_pdf_v3.jsonl"
DEFAULT_ITEM_BANK = SKILL_DIR / "data" / "item_bank" / "chemistry_v3_6695.jsonl"
DEFAULT_TRANSCRIPTS = SKILL_DIR / "data" / "from_pdf" / "full_markdown_v3"
DEFAULT_PAGE_IMAGES = SKILL_DIR / "data" / "page_images_v3"
DEFAULT_DOC_CACHE = SKILL_DIR / "data" / ".doc_to_pdf_cache"
DEFAULT_QUALITY_DIR = SKILL_DIR / "data" / "quality"
DEFAULT_SOURCE_ROOTS = [
    TOOLS_DIR / "上海化学卷合集",
    TOOLS_DIR / "_papers_archive",
    SKILL_DIR / "data",
]

IMAGE_KEYWORDS = [
    "如图",
    "下图",
    "上图",
    "图示",
    "图中",
    "所示",
    "示意图",
    "装置图",
    "流程图",
    "曲线图",
    "坐标图",
    "结构图",
    "晶胞",
    "合成路线",
    "转化关系",
    "能量变化图",
    "滴定曲线",
    "实验装置",
    "工艺流程",
    "图像",
    "图象",
    "图表",
    "[图示",
    "【图",
]

CATEGORY_RULES = [
    ("organic_structure", ["结构式", "有机", "官能团", "同分异构", "合成路线", "键线式", "烃", "酯", "醇", "醛", "酮", "羧酸", "苯环"]),
    ("experiment_device", ["装置", "实验", "仪器", "烧瓶", "分液漏斗", "冷凝管", "洗气瓶", "气体发生"]),
    ("process_flow", ["流程", "工艺", "工业", "流程图", "转化流程", "制备流程", "生产流程"]),
    ("chart_curve", ["曲线", "坐标", "图像", "图象", "滴定曲线", "能量变化", "速率", "随时间", "pH"]),
    ("crystal_cell", ["晶胞", "晶体", "配位数", "晶格", "立方", "晶体结构"]),
    ("electrochem_device", ["原电池", "电解池", "电极", "阴极", "阳极", "电池", "电化学", "离子交换膜"]),
]

STOP_CHARS = set("的了和是中在与及或为由对将可能应下列有关如下分别其中该某一个一种____()（）[]【】,.，。；;:：、\n\t ")


@dataclass(frozen=True)
class SourceResolution:
    candidates: list[Path]
    source_path: Path | None
    pdf_path: Path | None
    ambiguous: bool


def qid_from_pdf_item(item: dict[str, Any]) -> str:
    raw = f"{item.get('_source_file','')}|{item.get('q_num','')}|{item.get('stem','')[:40]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def hash_path_12(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(r"[，。、“”‘’：；？！,.!?;:()（）\[\]【】{}<>《》_—\-]+", "", s)
    return s


def tokenize(value: str) -> set[str]:
    text = normalize_text(value)
    tokens = set(re.findall(r"[A-Za-z]{1,3}\d*|\d+\.?\d*|[\u4e00-\u9fff]{2,6}", text))
    for i in range(max(0, len(text) - 2)):
        tri = text[i : i + 3]
        if any(ch in STOP_CHARS for ch in tri):
            continue
        tokens.add(tri)
    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def longest_common_substring_len(a: str, b: str, limit: int = 5000) -> int:
    a = normalize_text(a)[:limit]
    b = normalize_text(b)[:limit]
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    if a in b:
        return len(a)
    # This score is a quality heuristic run for many item/page pairs. Exact
    # dynamic-programming LCS is too slow on 5k-character transcript pages, so
    # use bounded substring probes plus n-gram overlap.
    for width in (120, 80, 50, 30, 20, 12):
        if len(a) < width:
            continue
        for start in range(0, len(a) - width + 1, max(1, width // 2)):
            if a[start : start + width] in b:
                return width
    if len(a) < 6 or len(b) < 6:
        return 0
    b_grams = {b[i : i + 6] for i in range(len(b) - 5)}
    best = 0
    for i in range(len(a) - 5):
        if a[i : i + 6] in b_grams:
            best = 6
            break
    return best


def item_text(item: dict[str, Any]) -> str:
    options = item.get("options") or {}
    if isinstance(options, dict):
        option_text = " ".join(f"{key}.{value}" for key, value in options.items())
    else:
        option_text = str(options)
    return "\n".join(
        [
            str(item.get("q_num", "")),
            str(item.get("stem", "")),
            option_text,
            str(item.get("diagram_description", "")),
        ]
    )


def is_image_like(item: dict[str, Any]) -> bool:
    text = item_text(item)
    return bool(item.get("diagram_description")) or any(keyword in text for keyword in IMAGE_KEYWORDS)


def item_category(item: dict[str, Any]) -> str:
    text = item_text(item)
    for category, keywords in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def transcript_path_for(source_file: str, transcript_dir: Path) -> Path | None:
    stem = Path(source_file).stem[:60]
    direct = transcript_dir / f"{stem}_transcript.md"
    if direct.exists():
        return direct
    candidates = sorted(transcript_dir.glob(f"{stem[:40]}*_transcript.md"))
    if len(candidates) == 1:
        return candidates[0]
    return direct if direct.exists() else None


def split_transcript_pages(markdown: str) -> dict[int, str]:
    parts = re.split(r"---\s*第(\d+)页\s*---", markdown)
    pages: dict[int, str] = {}
    if len(parts) == 1:
        return pages
    for i in range(1, len(parts) - 1, 2):
        try:
            pages[int(parts[i])] = parts[i + 1]
        except ValueError:
            continue
    return pages


def score_page(item: dict[str, Any], page_text: str) -> dict[str, Any]:
    qtext = item_text(item)
    qnum = str(item.get("q_num") or "")
    main_q = re.sub(r"[（(]\d+[)）]$", "", qnum).strip()
    norm_page = normalize_text(page_text)
    stem = normalize_text(item.get("stem", ""))
    exact = 0
    for width in (80, 50, 30, 20, 12):
        if len(stem) >= width and stem[:width] in norm_page:
            exact = width
            break
    qnum_pattern = False
    if main_q:
        qnum_pattern = bool(re.search(rf"(^|[^\d]){re.escape(main_q)}\s*[.．、]", page_text))
    token_score = jaccard(tokenize(qtext), tokenize(page_text))
    lcs = longest_common_substring_len(qtext, page_text)
    score = min(
        1.0,
        (exact / 80.0) * 0.58
        + min(lcs, 120) / 120 * 0.25
        + min(token_score * 3.0, 1.0) * 0.17
        + (0.08 if qnum_pattern else 0.0),
    )
    return {
        "score": round(score, 4),
        "exact_prefix": exact,
        "lcs": lcs,
        "token_jaccard": round(token_score, 4),
        "qnum_pattern": qnum_pattern,
    }


def text_tier(score: dict[str, Any]) -> str:
    if score["exact_prefix"] >= 30 or score["score"] >= 0.48:
        return "strong"
    if score["exact_prefix"] >= 12 or score["score"] >= 0.28 or score["qnum_pattern"]:
        return "weak"
    return "reject"


def build_source_index(source_roots: Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in source_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".pdf", ".doc", ".docx"}:
                index.setdefault(path.name, []).append(path)
    return index


def resolve_source(source_file: str, source_index: dict[str, list[Path]], doc_cache_dir: Path) -> SourceResolution:
    candidates = sorted(source_index.get(source_file, []), key=lambda p: (len(str(p)), str(p)))
    source_path = candidates[0] if candidates else None
    pdf_path = source_path
    if source_path and source_path.suffix.lower() in {".doc", ".docx"}:
        pdf_path = doc_cache_dir / f"{hash_path_12(source_path)}.pdf"
    return SourceResolution(
        candidates=candidates,
        source_path=source_path,
        pdf_path=pdf_path,
        ambiguous=len(candidates) > 1,
    )


def expected_page_image(pdf_path: Path | None, page: int | None, page_images_dir: Path) -> Path | None:
    if not pdf_path or not page:
        return None
    return page_images_dir / f"{hash_path_12(pdf_path)}_p{int(page):03d}.jpg"


def visible_anchors(item: dict[str, Any], page_text: str) -> list[str]:
    anchors: list[str] = []
    qnum = str(item.get("q_num") or "").strip()
    if qnum and qnum in page_text:
        anchors.append(qnum)
    stem_norm = normalize_text(item.get("stem", ""))
    page_norm = normalize_text(page_text)
    for width in (30, 20, 12):
        if len(stem_norm) >= width and stem_norm[:width] in page_norm:
            anchors.append(stem_norm[:width])
            break
    options = item.get("options") or {}
    if isinstance(options, dict):
        for key, value in options.items():
            value_norm = normalize_text(value)
            if value_norm and value_norm[: min(12, len(value_norm))] in page_norm:
                anchors.append(f"{key}.{value_norm[:12]}")
            if len(anchors) >= 5:
                break
    return anchors[:5]


def build_manifest(
    pdf_items_path: Path = DEFAULT_PDF_ITEMS,
    item_bank_path: Path = DEFAULT_ITEM_BANK,
    page_images_dir: Path = DEFAULT_PAGE_IMAGES,
    transcript_dir: Path = DEFAULT_TRANSCRIPTS,
    source_roots: Iterable[Path] = DEFAULT_SOURCE_ROOTS,
    doc_cache_dir: Path = DEFAULT_DOC_CACHE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdf_items = load_jsonl(Path(pdf_items_path))
    item_bank = {item.get("item_id"): item for item in load_jsonl(Path(item_bank_path)) if item.get("item_id")}
    source_index = build_source_index([Path(p) for p in source_roots])
    transcript_cache: dict[Path, dict[int, str]] = {}

    manifest: list[dict[str, Any]] = []
    for item in pdf_items:
        if not is_image_like(item):
            continue

        item_id = qid_from_pdf_item(item)
        bank_item = item_bank.get(item_id, {})
        source_file = item.get("_source_file") or bank_item.get("source") or ""
        declared_page = item.get("_page")
        try:
            declared_page = int(declared_page) if declared_page else None
        except (TypeError, ValueError):
            declared_page = None

        transcript_path = transcript_path_for(source_file, transcript_dir)
        pages: dict[int, str] = {}
        if transcript_path:
            if transcript_path not in transcript_cache:
                transcript_cache[transcript_path] = split_transcript_pages(
                    transcript_path.read_text(encoding="utf-8", errors="ignore")
                )
            pages = transcript_cache[transcript_path]

        declared_text = pages.get(declared_page or -1, "")
        declared_score = score_page(item, declared_text) if declared_text else {
            "score": 0.0,
            "exact_prefix": 0,
            "lcs": 0,
            "token_jaccard": 0.0,
            "qnum_pattern": False,
        }
        best_text_page = None
        best_score = {
            "score": 0.0,
            "exact_prefix": 0,
            "lcs": 0,
            "token_jaccard": 0.0,
            "qnum_pattern": False,
        }
        for page_num, text in pages.items():
            candidate = score_page(item, text)
            if candidate["score"] > best_score["score"]:
                best_score = candidate
                best_text_page = page_num

        source = resolve_source(source_file, source_index, doc_cache_dir)
        page_image = expected_page_image(source.pdf_path, declared_page, page_images_dir)
        image_exists = bool(page_image and page_image.exists() and page_image.stat().st_size > 1000)
        page_mismatch = bool(
            declared_page
            and best_text_page
            and int(declared_page) != int(best_text_page)
            and text_tier(best_score) != "reject"
        )

        blockers: list[str] = []
        if not source.source_path:
            blockers.append("source_unresolved")
        if source.ambiguous:
            blockers.append("source_ambiguous")
        if not transcript_path or not pages:
            blockers.append("missing_transcript")
        if declared_page is None:
            blockers.append("missing_declared_page")
        elif declared_page not in pages:
            blockers.append("declared_page_not_in_transcript")
        if page_mismatch:
            blockers.append("page_mismatch")
        if not image_exists:
            blockers.append("missing_page_image")

        declared_tier = text_tier(declared_score)
        if image_exists and not source.ambiguous and not page_mismatch and declared_tier == "strong":
            match_tier = "strong"
        elif image_exists and declared_tier in {"strong", "weak"}:
            match_tier = "weak"
        else:
            match_tier = "reject"

        if match_tier == "reject" and "visual_asset_rejected" not in blockers:
            blockers.append("visual_asset_rejected")

        page_hash = sha256_file(page_image) if image_exists and page_image else ""
        row = {
            "item_id": item_id,
            "source_file": source_file,
            "source_path": str(source.source_path) if source.source_path else "",
            "source_candidates": len(source.candidates),
            "source_ambiguous": source.ambiguous,
            "declared_page": declared_page,
            "best_text_page": best_text_page,
            "page_image_path": str(page_image) if image_exists and page_image else "",
            "page_image_hash": page_hash,
            "crop_path": None,
            "crop_hash": None,
            "visible_anchors": visible_anchors(item, declared_text),
            "declared_match_score": declared_score["score"],
            "best_match_score": best_score["score"],
            "declared_text_tier": declared_tier,
            "best_text_tier": text_tier(best_score),
            "match_tier": match_tier,
            "crop_tier": "page_only" if image_exists else "missing",
            "needs_image": True,
            "category": item_category(item),
            "question_type": item.get("question_type") or bank_item.get("question_type") or "",
            "difficulty": item.get("difficulty") or bank_item.get("difficulty") or "",
            "answer": item.get("answer") or (bank_item.get("standard_solution") or {}).get("standard_answer", ""),
            "blocker_reasons": blockers,
        }
        manifest.append(row)

    tier_counts = Counter(row["match_tier"] for row in manifest)
    category_counts = Counter(row["category"] for row in manifest)
    summary = {
        "total_items": len(pdf_items),
        "image_like_items": len(manifest),
        "manifest_rows": len(manifest),
        "match_tiers": dict(tier_counts),
        "missing_page_image": sum(1 for row in manifest if "missing_page_image" in row["blocker_reasons"]),
        "page_mismatch": sum(1 for row in manifest if "page_mismatch" in row["blocker_reasons"]),
        "source_unresolved": sum(1 for row in manifest if "source_unresolved" in row["blocker_reasons"]),
        "source_ambiguous": sum(1 for row in manifest if row["source_ambiguous"]),
        "by_category": dict(category_counts),
    }
    for tier in ("strong", "weak", "reject"):
        summary["match_tiers"].setdefault(tier, 0)
    return manifest, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build visual asset manifest for v3 item bank.")
    parser.add_argument("--pdf-items", type=Path, default=DEFAULT_PDF_ITEMS)
    parser.add_argument("--item-bank", type=Path, default=DEFAULT_ITEM_BANK)
    parser.add_argument("--page-images", type=Path, default=DEFAULT_PAGE_IMAGES)
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--doc-cache", type=Path, default=DEFAULT_DOC_CACHE)
    parser.add_argument("--source-root", action="append", type=Path, dest="source_roots")
    parser.add_argument("--out", type=Path, default=DEFAULT_QUALITY_DIR / "visual_asset_manifest.jsonl")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_QUALITY_DIR / "visual_asset_manifest_summary.json")
    parser.add_argument("--write", action="store_true", help="Write manifest and summary files. Default is dry-run.")
    args = parser.parse_args()

    source_roots = args.source_roots if args.source_roots else DEFAULT_SOURCE_ROOTS
    manifest, summary = build_manifest(
        pdf_items_path=args.pdf_items,
        item_bank_path=args.item_bank,
        page_images_dir=args.page_images,
        transcript_dir=args.transcripts,
        source_roots=source_roots,
        doc_cache_dir=args.doc_cache,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        write_jsonl(args.out, manifest)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out}")
        print(f"WROTE {args.summary_out}")
    else:
        print("DRY RUN: pass --write to write data/quality outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
