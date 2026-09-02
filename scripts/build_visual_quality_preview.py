#!/usr/bin/env python3
"""Generate a static visual quality preview page for browser smoke testing."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_EVAL_SET = SKILL_DIR / "data" / "evals" / "visual_item_eval_set.jsonl"
DEFAULT_QUALITY = SKILL_DIR / "data" / "quality" / "item_quality_manifest.jsonl"
DEFAULT_OUT = SKILL_DIR / "apps" / "web" / "visual_quality_preview.html"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rel_to_html(path_value: str, html_path: Path) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    return os.path.relpath(path, html_path.parent)


def asset_name_for(item_id: str, source_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source_path.stem).strip("-") or "image"
    suffix = source_path.suffix.lower() or ".png"
    return f"{item_id}_{stem}{suffix}"


def package_image_for_html(path_value: str, item_id: str, html_path: Path) -> str:
    if not path_value:
        return ""
    source = Path(path_value)
    if not source.exists():
        return ""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = html_path.parent / "visual_quality_preview_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_path = asset_dir / asset_name_for(item_id, source)
    if not asset_path.exists() or asset_path.stat().st_size != source.stat().st_size:
        shutil.copy2(source, asset_path)
    return os.path.relpath(asset_path, html_path.parent)


def build_html(eval_set_path: Path = DEFAULT_EVAL_SET, quality_path: Path = DEFAULT_QUALITY, out_path: Path = DEFAULT_OUT, limit: int = 12) -> str:
    eval_rows = load_jsonl(eval_set_path)[:limit]
    quality_by_id = {row.get("item_id"): row for row in load_jsonl(quality_path)}
    cards = []
    for row in eval_rows:
        q = quality_by_id.get(row["item_id"], {})
        student_readable = bool(q.get("student_readable", False))
        strong = bool(q.get("strong", False))
        blocked = not student_readable
        display_image = q.get("display_image_path") or row.get("crop_path") or row.get("page_image_path", "")
        image_src = package_image_for_html(display_image, str(row.get("item_id", "")), out_path)
        image_html = (
            f'<img src="{html.escape(image_src)}" alt="Original exam page crop for {html.escape(row.get("item_id",""))}">'
            if image_src
            else '<div class="missing-image">image evidence missing</div>'
        )
        options = row.get("options") or {}
        option_html = "".join(
            f"<div class='option'><span>{html.escape(str(k))}</span>{html.escape(str(v))}</div>"
            for k, v in options.items()
        )
        blocker_html = "".join(
            f"<span>{html.escape(str(reason))}</span>" for reason in q.get("blocker_reasons", [])
        ) or "<span>no hard blockers</span>"
        cards.append(
            f"""
      <article class="item-card {'blocked' if blocked else 'allowed'}">
        <header>
          <div>
            <p class="eyebrow">{html.escape(row.get('category',''))} · {html.escape(row.get('difficulty',''))} · {html.escape(row.get('question_type',''))}</p>
            <h2>{html.escape(row.get('item_id',''))}</h2>
          </div>
          <strong>{'not-student-readable' if blocked else ('diagnosis-strong' if strong else 'student-readable')}</strong>
        </header>
        <section class="question">
          <p>{html.escape(row.get('stem',''))}</p>
          <div class="options">{option_html}</div>
        </section>
        <figure>
          {image_html}
          <figcaption>{html.escape(row.get('source_file',''))} · page {html.escape(str(row.get('page','')))}</figcaption>
        </figure>
        <footer>
          <div><b>stage</b> {html.escape(str(q.get('visual_pipeline_stage','')))}</div>
          <div><b>visual</b> {html.escape(str(q.get('visual_asset_status','')))}</div>
          <div><b>readability</b> {html.escape(str(q.get('readability_status','')))}</div>
          <div><b>llm</b> {html.escape(str(q.get('llm_understanding_status','')))}</div>
          <div class="blockers">{blocker_html}</div>
        </footer>
      </article>
"""
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>YHer Visual Quality Preview</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif; background: #0b0c0d; color: #f5f5f7; }}
main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 48px; }}
h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: 0; }}
.sub {{ color: #a1a1aa; margin: 0 0 24px; line-height: 1.5; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
.item-card {{ border: 1px solid #303236; border-radius: 8px; background: #17181a; overflow: hidden; }}
.item-card header {{ display: flex; justify-content: space-between; gap: 16px; padding: 16px 16px 10px; border-bottom: 1px solid #303236; }}
.item-card h2 {{ font-size: 16px; margin: 2px 0 0; letter-spacing: 0; overflow-wrap: anywhere; }}
.eyebrow {{ color: #a1a1aa; font-size: 12px; margin: 0; }}
.item-card strong {{ align-self: start; border: 1px solid #5b5f66; border-radius: 999px; padding: 4px 10px; font-size: 12px; color: #d4d4d8; }}
.item-card.allowed strong {{ border-color: #2f8f5b; color: #67e8a6; }}
.item-card.blocked strong {{ border-color: #9b6a2f; color: #f2b464; }}
.question {{ padding: 14px 16px; color: #e4e4e7; line-height: 1.65; }}
.question p {{ margin: 0 0 10px; white-space: pre-wrap; overflow-wrap: anywhere; }}
.option {{ border-top: 1px solid #303236; padding: 8px 0; display: grid; grid-template-columns: 28px 1fr; gap: 8px; }}
.option span {{ color: #a1a1aa; }}
figure {{ margin: 0; border-top: 1px solid #303236; background: #0b0c0d; }}
img {{ display: block; width: 100%; height: auto; max-height: 620px; object-fit: contain; background: #fff; }}
.missing-image {{ min-height: 220px; display: grid; place-items: center; background: #1f2023; color: #f2b464; border-top: 1px solid #303236; }}
figcaption {{ color: #a1a1aa; font-size: 12px; line-height: 1.4; padding: 10px 16px; overflow-wrap: anywhere; }}
footer {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; padding: 12px 16px 16px; color: #d4d4d8; font-size: 13px; border-top: 1px solid #303236; }}
footer b {{ color: #a1a1aa; font-weight: 500; display: block; margin-bottom: 2px; }}
.blockers {{ grid-column: 1 / -1; display: flex; gap: 6px; flex-wrap: wrap; }}
.blockers span {{ border: 1px solid #3f4248; border-radius: 999px; padding: 4px 8px; color: #c7c7cc; }}
@media (max-width: 760px) {{
  main {{ width: min(100% - 20px, 560px); padding-top: 18px; }}
  .grid {{ grid-template-columns: 1fr; gap: 12px; }}
  h1 {{ font-size: 22px; }}
  footer {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <h1>Visual Quality Preview</h1>
  <p class="sub">Shows structured question text, original page image, source, and current quality gate status. Blocked visual items must not enter diagnosis/profile.</p>
  <section class="grid">
    {''.join(cards)}
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static visual quality preview HTML.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    html_text = build_html(args.eval_set, args.quality, args.out, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
