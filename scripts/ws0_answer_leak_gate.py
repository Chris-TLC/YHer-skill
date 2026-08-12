#!/usr/bin/env python3
"""WS0: answer-leak gate over all student-facing display images.

Collects every display image (item crops + page images referenced by the
visual asset manifest), OCRs them with the local Vision CLI
(scripts/ws0_vision_ocr, free/offline), applies leak keyword rules, and writes:

  /tmp/yher_ws0_answer_leak/ocr_raw.jsonl        full OCR text per image
  /tmp/yher_ws0_answer_leak/leak_report.json     summary + per-image hits
  /tmp/yher_ws0_answer_leak/quarantine_list.txt  image paths that MUST NOT be displayed

Read-only with respect to official data: this gate reports; it does not
modify manifests. Quarantine enforcement is a separate reviewed step.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
MANIFEST = SKILL / "data" / "quality" / "visual_asset_manifest.jsonl"
CROPS_ROOT = SKILL / "data" / "quality" / "visual_crops"
OCR_BIN = SKILL / "scripts" / "ws0_vision_ocr"
OUT_DIR = Path("/tmp/yher_ws0_answer_leak")

# Answer/analysis markers = printed answers visible to students.
# Ad/watermark markers = third-party contamination on display images.
LEAK_KEYWORDS = {
    "answer": ["【答案】", "【解析】", "【详解】", "【点评】", "【分析】", "故选", "答案：", "解析："],
    "ad": ["jiajiao", "+V:", "+v:", "家教", "微信号", "公众号"],
}


def collect_display_images() -> dict[str, list[str]]:
    """path -> [contexts] (crop:item_id / page:item_id / crops_dir)."""
    images: dict[str, list[str]] = {}

    def add(path: str | None, ctx: str) -> None:
        if not path:
            return
        p = str(path)
        if Path(p).exists():
            images.setdefault(p, []).append(ctx)

    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        iid = row.get("item_id", "?")
        add(row.get("crop_path"), f"crop:{iid}")
        add(row.get("page_image_path"), f"page:{iid}")

    # All approved crop dirs (covers crops not referenced by current manifest rows)
    for img in CROPS_ROOT.rglob("*.png"):
        add(str(img), f"crops_dir:{img.parent.name}")
    for img in CROPS_ROOT.rglob("*.jpg"):
        add(str(img), f"crops_dir:{img.parent.name}")
    return images


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = collect_display_images()
    crop_paths = [p for p, ctx in images.items() if any(c.startswith(("crop:", "crops_dir:")) for c in ctx)]
    page_paths = [p for p, ctx in images.items() if any(c.startswith("page:") for c in ctx)]
    print(f"[info] display images: {len(images)} total | crops {len(crop_paths)} | pages {len(page_paths)}")

    list_file = OUT_DIR / "image_list.txt"
    list_file.write_text("\n".join(sorted(images)), encoding="utf-8")

    raw_path = OUT_DIR / "ocr_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen([str(OCR_BIN), str(list_file)], stdout=out, stderr=subprocess.PIPE, text=True)
        _, err = proc.communicate()
        if proc.returncode != 0:
            print(f"[fatal] ocr bin failed: {err[:400]}", file=sys.stderr)
            return 2

    leaks = []
    errors = []
    kw_counter: Counter[str] = Counter()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        path = r["path"]
        if not r.get("ok"):
            errors.append({"path": path, "error": r.get("error")})
            continue
        text = r.get("text", "")
        hits = {cat: [k for k in kws if k in text] for cat, kws in LEAK_KEYWORDS.items()}
        hits = {cat: ks for cat, ks in hits.items() if ks}
        if hits:
            for ks in hits.values():
                kw_counter.update(ks)
            is_crop = any(c.startswith(("crop:", "crops_dir:")) for c in images.get(path, []))
            leaks.append({
                "path": path,
                "kind": "crop" if is_crop else "page",
                "contexts": images.get(path, [])[:6],
                "hits": hits,
            })

    leaks.sort(key=lambda x: (x["kind"] != "crop", x["path"]))
    crop_leaks = [l for l in leaks if l["kind"] == "crop"]
    page_leaks = [l for l in leaks if l["kind"] == "page"]

    report = {
        "scanned": len(images),
        "ocr_errors": len(errors),
        "leak_total": len(leaks),
        "crop_leaks": len(crop_leaks),
        "crop_total": len(crop_paths),
        "page_leaks": len(page_leaks),
        "page_total": len(page_paths),
        "keyword_counts": dict(kw_counter.most_common()),
        "leaks": leaks,
        "errors": errors[:50],
    }
    (OUT_DIR / "leak_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "quarantine_list.txt").write_text("\n".join(l["path"] for l in leaks), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k not in ("leaks", "errors")}, ensure_ascii=False, indent=2))
    print(f"[done] report: {OUT_DIR / 'leak_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
