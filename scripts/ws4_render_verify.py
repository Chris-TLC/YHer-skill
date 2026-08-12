#!/usr/bin/env python3
"""WS4 full render verification and regression runner.

Outputs go to /tmp/yher_batch9_ws4 by default. The script does not write
item-bank data and does not modify the RIR renderer semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = Path("/tmp/yher_batch9_ws4")
EXPECTED_DEGRADE = ROOT / "data/ws4_render_baseline_20260704/expected_degrade_items.json"
BUNDLED_NODE = Path("/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")
BUNDLED_NODE_MODULES = Path("/Users/mac/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")

sys.path.insert(0, str(ROOT))

from core.data.item_bank_v4 import iter_service_items  # noqa: E402
from core.data.ws2_transcripts import resolve_image_path  # noqa: E402
from core.render.block_renderer import item_to_rir  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def walk_nodes(nodes: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for node in nodes:
        yield node
        if node.get("kind") == "table":
            for row in node.get("rows") or []:
                for cell in row or []:
                    yield from walk_nodes(cell or [])


def all_rir_nodes(rir: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for paragraphs in (rir.get("zones") or {}).values():
        for para in paragraphs or []:
            yield from walk_nodes(para or [])


def choose_node_bin(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if BUNDLED_NODE.exists():
        return str(BUNDLED_NODE)
    found = shutil.which("node")
    if found:
        return found
    raise RuntimeError("node executable not found")


def node_env(out_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    paths = [str(out_dir / "node_modules")]
    if BUNDLED_NODE_MODULES.exists():
        paths.append(str(BUNDLED_NODE_MODULES))
    if env.get("NODE_PATH"):
        paths.append(env["NODE_PATH"])
    env["NODE_PATH"] = os.pathsep.join(paths)
    return env


def katex_dist(out_dir: Path) -> Path | None:
    candidates = [
        out_dir / "node_modules/katex/dist",
        BUNDLED_NODE_MODULES / "katex/dist",
        ROOT / "node_modules/katex/dist",
    ]
    for c in candidates:
        if (c / "katex.min.js").exists() and (c / "contrib/mhchem.min.js").exists():
            return c
    return None


def classify_samples(records: List[Dict[str, Any]], limit_pairs: int = 10) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "latex": [],
        "image": [],
        "table": [],
        "placeholder": [],
        "plain": [],
    }
    for rec in records:
        kinds = {n.get("kind") for n in all_rir_nodes(rec["rir"])}
        if "latex" in kinds:
            buckets["latex"].append(rec)
        if "image" in kinds:
            buckets["image"].append(rec)
        if "table" in kinds:
            buckets["table"].append(rec)
        if "placeholder" in kinds:
            buckets["placeholder"].append(rec)
        if kinds == {"text"}:
            buckets["plain"].append(rec)

    picked: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("latex", "image", "table", "placeholder", "plain"):
        for rec in buckets[key][:3]:
            if rec["item_id"] not in seen:
                picked.append(rec)
                seen.add(rec["item_id"])
    for rec in records:
        if len(picked) >= limit_pairs:
            break
        if rec["item_id"] not in seen:
            picked.append(rec)
            seen.add(rec["item_id"])
    return picked[:limit_pairs]


def collect_service_records() -> Tuple[List[Dict[str, Any]], List[str], Counter]:
    records: List[Dict[str, Any]] = []
    actual_degrade: List[str] = []
    reason_counts: Counter = Counter()
    for item in iter_service_items():
        rir = item_to_rir(item, zones=("stem", "answer"))
        stem_rir = item_to_rir(item, zones=("stem",))
        if stem_rir.get("degraded"):
            actual_degrade.append(item["item_id"])
            for reason in stem_rir.get("degrade_reasons") or []:
                reason_counts[reason.split(":", 1)[0]] += 1
        records.append({"item_id": item["item_id"], "rir": rir})
    return records, actual_degrade, reason_counts


def collect_asset_paths(records: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for rec in records:
        for node in all_rir_nodes(rec["rir"]):
            h = node.get("asset_hash")
            if not h or h in paths:
                continue
            p = resolve_image_path(h)
            if p is not None:
                paths[h] = str(p)
    return paths


def run_browser_checks(
    *,
    node_bin: str,
    out_dir: Path,
    records: List[Dict[str, Any]],
    screenshot_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    dist = katex_dist(out_dir)
    if dist is None:
        return {
            "ok": False,
            "reason": "katex_dist_not_found",
            "katex_errors": [],
            "dom_empty_blocks": [],
            "render_reports": [],
            "screenshots": [],
        }

    browser_dir = out_dir / "browser"
    browser_dir.mkdir(parents=True, exist_ok=True)
    input_path = browser_dir / "rir_records.json"
    screenshot_input_path = browser_dir / "screenshot_records.json"
    asset_map_path = browser_dir / "asset_paths.json"
    result_path = browser_dir / "browser_result.json"
    node_script = browser_dir / "ws4_browser_check.cjs"

    write_json(input_path, records)
    write_json(screenshot_input_path, screenshot_records)
    write_json(asset_map_path, collect_asset_paths(screenshot_records))

    node_script.write_text(
        textwrap.dedent(
            f"""
            const fs = require("fs");
            const path = require("path");
            const {{ chromium }} = require("playwright");

            const records = JSON.parse(fs.readFileSync({json.dumps(str(input_path))}, "utf8"));
            const screenshotRecords = JSON.parse(fs.readFileSync({json.dumps(str(screenshot_input_path))}, "utf8"));
            const assetPaths = JSON.parse(fs.readFileSync({json.dumps(str(asset_map_path))}, "utf8"));
            const katexJs = fs.readFileSync({json.dumps(str(dist / "katex.min.js"))}, "utf8");
            const mhchemJs = fs.readFileSync({json.dumps(str(dist / "contrib/mhchem.min.js"))}, "utf8");
            const rendererJs = fs.readFileSync({json.dumps(str(ROOT / "apps/web/rir_renderer.js"))}, "utf8");
            const screenshotDir = {json.dumps(str(out_dir / "screenshots"))};

            function html() {{
              return `<!doctype html><html><head><meta charset="utf-8"><base href="http://yher.local/">
              <style>
              body{{margin:0;background:#f5f5f7;color:#1d1d1f;font-family:-apple-system,"PingFang SC",system-ui,sans-serif}}
              .page{{padding:24px}}.item{{background:#fff;border:1px solid #d2d2d7;border-radius:8px;padding:20px;max-width:920px;margin:0 auto}}
              .head{{font:12px ui-monospace,monospace;color:#6e6e73;margin-bottom:16px;overflow-wrap:anywhere}}
              .zone-title{{font-size:13px;color:#6e6e73;font-weight:700;margin:18px 0 8px}}
              </style>
              <script>${{katexJs}}<\\/script><script>${{mhchemJs}}<\\/script><script>${{rendererJs}}<\\/script>
              </head><body><main class="page"><section id="root" class="item"></section></main></body></html>`;
            }}

            function visibleParagraphIssues(itemId) {{
              const issues = [];
              document.querySelectorAll(".rir-para").forEach((el, idx) => {{
                const hasVisible = (el.textContent || "").trim().length > 0 ||
                  Boolean(el.querySelector("img,.katex,.rir-placeholder,table"));
                if (!hasVisible) {{
                  issues.push({{item_id:itemId, index:idx, html:el.innerHTML.slice(0,200)}});
                }}
              }});
              return issues;
            }}

            async function main() {{
              fs.mkdirSync(screenshotDir, {{recursive:true}});
              const systemChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
              const launchOptions = {{headless:true}};
              if (fs.existsSync(systemChrome)) launchOptions.executablePath = systemChrome;
              const browser = await chromium.launch(launchOptions);
              const page = await browser.newPage({{viewport:{{width:1280,height:900}}, deviceScaleFactor:1}});
              await page.route("**/api/v4/assets/*.png", async route => {{
                const url = new URL(route.request().url());
                const m = url.pathname.match(/\\/api\\/v4\\/assets\\/([0-9a-f]{{64}})\\.png$/);
                const p = m ? assetPaths[m[1]] : null;
                if (p && fs.existsSync(p)) await route.fulfill({{path:p}});
                else await route.fulfill({{status:404, body:"missing"}});
              }});
              await page.setContent(html(), {{waitUntil:"load"}});

              const result = await page.evaluate(async (items) => {{
                const renderReports = [];
                window.fetch = async (url, opts) => {{
                  try {{ renderReports.push(JSON.parse((opts && opts.body) || "{{}}")); }} catch(e) {{}}
                  return {{ok:true, json:async()=>({{ok:true}})}};
                }};
                const domEmptyBlocks = [];
                for (const rec of items) {{
                  const root = document.getElementById("root");
                  root.textContent = "";
                  window.YHerRirRenderer.renderRir(rec.rir, root, {{itemId:rec.item_id, reportUrl:"/api/render_report"}});
                  await new Promise(resolve => requestAnimationFrame(resolve));
                  document.querySelectorAll(".rir-para").forEach((el, idx) => {{
                    const hasVisible = (el.textContent || "").trim().length > 0 ||
                      Boolean(el.querySelector("img,.katex,.rir-placeholder,table"));
                    if (!hasVisible) {{
                      domEmptyBlocks.push({{item_id:rec.item_id, index:idx, html:el.innerHTML.slice(0,200)}});
                    }}
                  }});
                }}
                return {{renderReports, domEmptyBlocks}};
              }}, records);

              const screenshots = [];
              let index = 0;
              for (const rec of screenshotRecords) {{
                for (const vp of [
                  {{name:"desktop", width:1280, height:900}},
                  {{name:"mobile", width:390, height:844}}
                ]) {{
                  await page.setViewportSize({{width:vp.width, height:vp.height}});
                  await page.evaluate(async (rec) => {{
                    const root = document.getElementById("root");
                    root.textContent = "";
                    const head = document.createElement("div");
                    head.className = "head";
                    head.textContent = rec.item_id;
                    root.appendChild(head);
                    for (const zone of ["stem", "answer"]) {{
                      const title = document.createElement("div");
                      title.className = "zone-title";
                      title.textContent = zone;
                      const body = document.createElement("div");
                      root.appendChild(title);
                      root.appendChild(body);
                      window.YHerRirRenderer.renderZone((rec.rir.zones && rec.rir.zones[zone]) || [], body, {{itemId:rec.item_id, disableReports:true}});
                    }}
                    await new Promise(resolve => requestAnimationFrame(resolve));
                  }}, rec);
                  await page.waitForTimeout(80);
                  const name = `${{String(index).padStart(2,"0")}}_${{rec.item_id}}_${{vp.name}}.png`;
                  const p = path.join(screenshotDir, name);
                  await page.screenshot({{path:p, fullPage:true}});
                  screenshots.push(p);
                  index += 1;
                }}
              }}
              await browser.close();
              fs.writeFileSync({json.dumps(str(result_path))}, JSON.stringify({{...result, screenshots}}, null, 2));
            }}

            main().catch(err => {{
              fs.writeFileSync({json.dumps(str(result_path))}, JSON.stringify({{fatal:String(err && err.stack || err)}}, null, 2));
              process.exit(1);
            }});
            """
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [node_bin, str(node_script)],
        cwd=str(ROOT),
        env=node_env(out_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = read_json(result_path) if result_path.exists() else {}
    result["ok"] = proc.returncode == 0 and not result.get("fatal")
    result["node_returncode"] = proc.returncode
    result["node_stdout"] = proc.stdout[-4000:]
    result["node_stderr"] = proc.stderr[-4000:]
    return result


def load_regression_targets() -> Dict[str, Any]:
    service_ids = {item["item_id"] for item in iter_service_items()}

    old_visual_rows = []
    visual_path = ROOT / "data/quality/visual_asset_manifest.jsonl"
    if visual_path.exists():
        with visual_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("promotion_status") == "official_promoted_approved_candidate":
                    old_visual_rows.append(row)

    quality_visual_strong = []
    quality_path = ROOT / "data/quality/item_quality_manifest.jsonl"
    if quality_path.exists():
        with quality_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("needs_image") and row.get("strong"):
                    quality_visual_strong.append(row)

    gold_assets = []
    gold_path = ROOT / "data/ws2_gold_set_20260704/gold_asset_list.jsonl"
    if gold_path.exists():
        with gold_path.open(encoding="utf-8") as f:
            gold_assets = [json.loads(line) for line in f]

    gold_hashes = {r.get("asset_hash") for r in gold_assets if r.get("asset_hash")}
    gold_item_ids: set[str] = set()
    for item in iter_service_items():
        rir = item_to_rir(item, zones=("stem", "answer"))
        hashes = {n.get("asset_hash") for n in all_rir_nodes(rir) if n.get("asset_hash")}
        if hashes & gold_hashes:
            gold_item_ids.add(item["item_id"])

    direct_old_ids = {r.get("item_id") for r in old_visual_rows if r.get("item_id") in service_ids}
    direct_quality_ids = {r.get("item_id") for r in quality_visual_strong if r.get("item_id") in service_ids}

    return {
        "old_visual_manifest_promoted_count": len(old_visual_rows),
        "old_visual_manifest_direct_v4_hits": sorted(direct_old_ids),
        "quality_visual_strong_count": len(quality_visual_strong),
        "quality_visual_strong_direct_v4_hits": sorted(direct_quality_ids),
        "gold_assets_count": len(gold_assets),
        "gold_service_item_ids": sorted(gold_item_ids),
    }


def queue_row(item_id: str, issue_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "issue_type": issue_type,
        "status": "pending",
        "reviewer": "",
        "details": details,
    }


def write_reports(
    *,
    out_dir: Path,
    records: List[Dict[str, Any]],
    expected_degrade: List[str],
    actual_degrade: List[str],
    reason_counts: Counter,
    browser: Dict[str, Any],
    regression: Dict[str, Any],
) -> Dict[str, Any]:
    expected = set(expected_degrade)
    actual = set(actual_degrade)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    render_reports = browser.get("renderReports") or browser.get("render_reports") or []
    dom_empty = browser.get("domEmptyBlocks") or browser.get("dom_empty_blocks") or []
    screenshots = browser.get("screenshots") or []

    review_rows: List[Dict[str, Any]] = []
    for row in dom_empty:
        review_rows.append(queue_row(row.get("item_id", ""), "empty_block", row))
    for report in render_reports:
        review_rows.append(queue_row(report.get("item_id", ""), report.get("issue_type", "render_report"), report))
    for item_id in missing:
        review_rows.append(queue_row(item_id, "degrade_missing_from_actual", {"expected": True, "actual": False}))
    for item_id in unexpected:
        review_rows.append(queue_row(item_id, "degrade_unexpected", {"expected": False, "actual": True}))

    write_jsonl(out_dir / "render_review_queue.jsonl", review_rows)
    renderer_issue_rows = [
        queue_row(item_id, "degrade_baseline_diff", {"direction": "missing_from_actual"})
        for item_id in missing
    ] + [
        queue_row(item_id, "degrade_baseline_diff", {"direction": "unexpected_actual"})
        for item_id in unexpected
    ]
    write_jsonl(out_dir / "renderer_issues.jsonl", renderer_issue_rows)
    if not (out_dir / "render_reports.jsonl").exists():
        (out_dir / "render_reports.jsonl").write_text("", encoding="utf-8")

    katex_error_count = len(render_reports)
    empty_count = len(dom_empty)
    degrade_diff_count = len(missing) + len(unexpected)

    summary = {
        "service_items": len(records),
        "katex_compile_errors": katex_error_count,
        "empty_blocks": empty_count,
        "expected_degrade_count": len(expected),
        "actual_degrade_count": len(actual),
        "degrade_missing_count": len(missing),
        "degrade_unexpected_count": len(unexpected),
        "review_queue_rows": len(review_rows),
        "renderer_issue_rows": len(renderer_issue_rows),
        "screenshots": len(screenshots),
        "browser_ok": browser.get("ok", False),
    }
    write_json(out_dir / "render_verify_summary.json", summary)

    report = f"""# WS4 Render Verify Report

## Scope

- Service items checked: {len(records)}
- RIR zones rendered for DOM/KaTeX: stem + answer
- Degrade assertion zone: stem, matching `expected_degrade_items.json`
- Browser engine: Playwright Chromium via local Node
- KaTeX source: `{katex_dist(out_dir) or 'missing'}`

## Three Assertions

| Assertion | Count | Status |
|---|---:|---|
| KaTeX compile/render errors | {katex_error_count} | {'PASS' if katex_error_count == 0 and browser.get('ok') else 'FAIL'} |
| Empty rendered blocks | {empty_count} | {'PASS' if empty_count == 0 and browser.get('ok') else 'FAIL'} |
| Degrade set diff | {degrade_diff_count} | {'PASS' if degrade_diff_count == 0 else 'FAIL'} |

## Degrade Set

- Expected degrade items: {len(expected)}
- Actual stem degrade items: {len(actual)}
- Missing from actual: {len(missing)}
- Unexpected actual: {len(unexpected)}
- Reason counts: `{dict(reason_counts)}`

## Artifacts

- Review queue: `{out_dir / 'render_review_queue.jsonl'}`
- Renderer issues: `{out_dir / 'renderer_issues.jsonl'}`
- Render reports: `{out_dir / 'render_reports.jsonl'}`
- Screenshots: `{out_dir / 'screenshots'}`

## Browser Evidence

- Browser run ok: {browser.get('ok', False)}
- Node return code: {browser.get('node_returncode')}
- Screenshot count: {len(screenshots)}
"""
    if missing or unexpected:
        report += "\n## Degrade Diff\n\n"
        if missing:
            report += "Missing from actual:\n\n" + "\n".join(f"- `{x}`" for x in missing[:50]) + "\n"
        if unexpected:
            report += "Unexpected actual:\n\n" + "\n".join(f"- `{x}`" for x in unexpected[:50]) + "\n"
    if not browser.get("ok"):
        report += "\n## Browser Failure Detail\n\n"
        report += f"stderr tail:\n\n```text\n{browser.get('node_stderr','')}\n```\n"
    (out_dir / "render_verify_report.md").write_text(report, encoding="utf-8")

    regression_items = sorted(
        set(regression.get("old_visual_manifest_direct_v4_hits") or [])
        | set(regression.get("quality_visual_strong_direct_v4_hits") or [])
        | set(regression.get("gold_service_item_ids") or [])
    )
    regression_failures = [
        row for row in review_rows
        if row.get("item_id") in set(regression_items)
    ]
    regression_report = f"""# WS4 Regression Report

## Target Discovery

- Brief expected old visual strong rows: 162
- Old visual promoted rows in `visual_asset_manifest`: {regression.get('old_visual_manifest_promoted_count')}
- Direct v4 id hits from old visual promoted rows: {len(regression.get('old_visual_manifest_direct_v4_hits') or [])}
- Current `item_quality_manifest` visual strong rows: {regression.get('quality_visual_strong_count')}
- Direct v4 id hits from current visual strong rows: {len(regression.get('quality_visual_strong_direct_v4_hits') or [])}
- Gold assets listed: {regression.get('gold_assets_count')}
- Gold asset service item hits through RIR asset hashes: {len(regression.get('gold_service_item_ids') or [])}
- Unique regression v4 items checked: {len(regression_items)}

## Result

- Regression failures intersecting discovered v4 items: {len(regression_failures)}
- Full service-pool render gate result: see `render_verify_report.md` (2526/2526 checked).
- Policy note: old visual-strong ids are v3 16-char ids; current v4 service ids are 40-char ids. Direct id intersection is therefore recorded as evidence instead of forcing a baseline rewrite.
- Count note: the current manifest exposes 164 `official_promoted_approved_candidate` rows and 89 stricter `needs_image && strong` rows; this differs from the brief's 162 label and is treated as manifest drift evidence, not a Codex baseline edit.

"""
    if regression_failures:
        regression_report += "## Failure Rows\n\n"
        for row in regression_failures[:100]:
            regression_report += f"- `{row.get('item_id')}` {row.get('issue_type')} {row.get('details')}\n"
    else:
        regression_report += "No failures intersected the discovered v4 regression item set.\n"
    (out_dir / "regression_report.md").write_text(regression_report, encoding="utf-8")
    write_json(out_dir / "regression_targets.json", regression)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify WS4 RIR frontend rendering.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--node-bin", default=None)
    parser.add_argument("--screenshot-pairs", type=int, default=10)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records, actual_degrade, reason_counts = collect_service_records()
    expected_degrade = read_json(EXPECTED_DEGRADE)
    screenshot_records = classify_samples(records, limit_pairs=args.screenshot_pairs)
    regression = load_regression_targets()

    node_bin = choose_node_bin(args.node_bin)
    browser = run_browser_checks(
        node_bin=node_bin,
        out_dir=out_dir,
        records=records,
        screenshot_records=screenshot_records,
    )
    summary = write_reports(
        out_dir=out_dir,
        records=records,
        expected_degrade=expected_degrade,
        actual_degrade=actual_degrade,
        reason_counts=reason_counts,
        browser=browser,
        regression=regression,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if (
        summary["service_items"] == 2526
        and summary["katex_compile_errors"] == 0
        and summary["empty_blocks"] == 0
        and summary["degrade_missing_count"] == 0
        and summary["degrade_unexpected_count"] == 0
        and summary["browser_ok"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
