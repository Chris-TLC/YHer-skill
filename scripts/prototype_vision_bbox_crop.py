#!/usr/bin/env python3
"""Prototype: crop page_only chemistry items by asking a vision model for the
question bounding box, instead of relying on the PDF text layer.

Background
----------
`build_visual_item_crops.py` locates each question by searching the PDF text
layer for its stem ("anchor"). That fails on scanned / image PDFs and on stems
that contain formulas, so only ~164 / 1815 image items ever got an item-level
crop. Every downstream visual-strong promotion is gated on having a crop, so
the whole 80% target is stuck behind crop coverage.

This prototype tests a different route: send the *page image* plus the known
stem anchors to a grounding-capable vision model (Qwen3-VL by default, GPT-4o
as fallback) and ask for the bounding box of that one question. If the model
returns a usable box we crop the full-resolution page to it.

It is deliberately a PROTOTYPE:
- reads only the official manifest, never writes it;
- writes crops + a side-by-side preview + a summary into a /tmp run dir;
- bounded to a small sample so the paid vision cost stays tiny.

Usage
-----
    python scripts/prototype_vision_bbox_crop.py --limit 24
    python scripts/prototype_vision_bbox_crop.py --provider openai --model gpt-4o --limit 12
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "data" / "quality" / "visual_asset_manifest.jsonl"
ENV_FILE = SKILL_ROOT / ".env"

PROVIDERS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "base_url_env": ["DASHSCOPE_CHAT_COMPLETIONS_URL", "DASHSCOPE_BASE_URL"],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "base_url_env": ["OPENAI_CHAT_COMPLETIONS_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"],
    },
}


def load_env(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines from .env without exporting anything noisy."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_endpoint(config: dict[str, Any], env: dict[str, str]) -> str:
    base_url = ""
    for env_key in config.get("base_url_env", []):
        if env.get(env_key):
            base_url = env[env_key].strip()
            break
    if not base_url:
        return str(config["base_url"])
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/openai"):
        return f"{base_url}/chat/completions"
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    if base_url.endswith("/v1beta"):
        return f"{base_url}/openai/chat/completions"
    return f"{base_url}/v1/chat/completions"


def clean_page_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("crop_tier") != "page_only":
            continue
        if not r.get("page_image_path"):
            continue
        if "page_mismatch" in (r.get("blocker_reasons") or []):
            continue
        if not Path(r["page_image_path"]).exists():
            continue
        out.append(r)
    return out


def diverse_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin over categories so a small sample still spans question types."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(r.get("category") or "unknown", []).append(r)
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    picked: list[dict[str, Any]] = []
    idx = 0
    while len(picked) < min(limit, len(rows)):
        cat = order[idx % len(order)]
        if buckets[cat]:
            picked.append(buckets[cat].pop(0))
        idx += 1
        if idx > len(rows) * 4:
            break
    return picked


def image_data_url(image_path: Path, max_side: int = 1568) -> tuple[str, int, int]:
    """Return (data_url, full_width, full_height). Downscale only for transport."""
    from PIL import Image

    with Image.open(image_path) as image:
        full_w, full_h = image.size
        work = image.convert("RGB")
        work.thumbnail((max_side, max_side))
        buffer = BytesIO()
        work.save(buffer, format="JPEG", quality=85)
        data = buffer.getvalue()
    url = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    return url, full_w, full_h


def build_prompt(anchors: list[str], qtype: str) -> str:
    anchor_text = "\n".join(f"- {a}" for a in anchors[:6] if a)
    return (
        "你面前是一张高中化学试卷的整页扫描图。这张图很可能来自【解析版】,\n"
        "页面上除了题目,还可能印着【答案】【正确答案】【解析】【试题解析】【详解】等内容。\n\n"
        "请只定位下面这一道题的【题目本身】(题型:" + (qtype or "未知") + "):\n"
        f"{anchor_text}\n\n"
        "框选规则(务必严格遵守):\n"
        "1. 只框题目部分:题号、题干、全部选项(到最后一个选项 D 或更后)、以及属于这道题的图/表/坐标系。\n"
        "2. 绝对不要把【答案】【正确答案】【解析】【试题解析】【详解】以及任何解题过程框进去。\n"
        "   如果这些内容在题目下方,框的下边界必须停在它们【上方】。\n"
        "3. 不要包含上一题或下一题的内容。\n"
        "4. 宁可在题目四周留一点空白,也不要切掉题干、选项或题目的图。\n"
        "5. 坐标用左上角为原点、范围 0-1000 的整数归一化坐标。\n\n"
        "只输出 JSON:\n"
        '{"found": true, "is_pure_question": true, "saw_answer_block": false, '
        '"last_option_visible": true, "box_2d": [x0,y0,x1,y1], "note": "简述依据"}\n'
        "字段说明:\n"
        "- is_pure_question: 框内是否只有题目、没有混入答案/解析。\n"
        "- saw_answer_block: 该页是否出现了答案或解析(不影响框,只是报告)。\n"
        "- last_option_visible: 最后一个选项/最后一个填空是否完整可见(若题目在页面底部被截断则为 false)。\n"
        '找不到这道题时输出 {"found": false, "is_pure_question": false, "saw_answer_block": false, '
        '"last_option_visible": false, "box_2d": [0,0,0,0], "note": "原因"}'
    )


def call_vision(endpoint: str, api_key: str, model: str, prompt: str, data_url: str,
                timeout: int = 90) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 400,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    latency = time.time() - started
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    usage = data.get("usage", {})
    return {"content": content, "usage": usage, "latency_s": round(latency, 2)}


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        raise ValueError("no_json_in_response")
    return json.loads(match.group(0))


def denorm_box(box: list[float], full_w: int, full_h: int, pad_frac: float = 0.015) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    px0 = max(0.0, x0 / 1000.0 - pad_frac) * full_w
    py0 = max(0.0, y0 / 1000.0 - pad_frac) * full_h
    px1 = min(1.0, x1 / 1000.0 + pad_frac) * full_w
    py1 = min(1.0, y1 / 1000.0 + pad_frac) * full_h
    return int(px0), int(py0), int(px1), int(py1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=list(PROVIDERS), default="qwen")
    parser.add_argument("--model", default=None, help="override model name")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    env = load_env(ENV_FILE)
    config = PROVIDERS[args.provider]
    model = args.model or config["model"]
    api_key = env.get(config["env_key"]) or os.environ.get(config["env_key"], "")
    if not api_key:
        print(f"[fatal] missing {config['env_key']} in {ENV_FILE}")
        return 2
    endpoint = resolve_endpoint(config, env)

    tag = f"{args.provider}_{model}".replace("/", "_")
    run_dir = Path(args.run_dir) if args.run_dir else Path(
        "/tmp/yher_vision_bbox_crop_prototype"
    ) / tag
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    rows = load_jsonl(MANIFEST)
    pool = clean_page_only(rows)
    sample = diverse_sample(pool, args.limit)
    print(f"[info] endpoint host: {endpoint.split('/')[2]}")
    print(f"[info] provider={args.provider} model={model}")
    print(f"[info] clean page_only pool={len(pool)}  sampling={len(sample)}")

    results: list[dict[str, Any]] = []
    ok = 0
    tokens = 0
    for i, row in enumerate(sample, 1):
        item_id = row.get("item_id")
        page_path = Path(row["page_image_path"])
        anchors = [a for a in (row.get("visible_anchors") or []) if a]
        rec: dict[str, Any] = {
            "item_id": item_id,
            "category": row.get("category"),
            "question_type": row.get("question_type"),
            "page_image_path": str(page_path),
        }
        try:
            data_url, full_w, full_h = image_data_url(page_path)
            prompt = build_prompt(anchors, row.get("question_type") or "")
            resp = call_vision(endpoint, api_key, model, prompt, data_url)
            tokens += int(resp["usage"].get("total_tokens", 0) or 0)
            parsed = extract_json(resp["content"])
            rec["latency_s"] = resp["latency_s"]
            rec["model_note"] = parsed.get("note", "")[:120]
            rec["is_pure_question"] = parsed.get("is_pure_question")
            rec["saw_answer_block"] = parsed.get("saw_answer_block")
            rec["last_option_visible"] = parsed.get("last_option_visible")
            if not parsed.get("found") or not parsed.get("box_2d"):
                rec["status"] = "not_found"
                results.append(rec)
                print(f"  [{i}/{len(sample)}] {item_id} not_found")
                continue
            box = [float(v) for v in parsed["box_2d"]]
            x0, y0, x1, y1 = denorm_box(box, full_w, full_h)
            w, h = x1 - x0, y1 - y0
            # sanity: a real single-question crop is neither the whole page nor a sliver
            frac = (w * h) / float(full_w * full_h)
            if w < 40 or h < 40 or frac > 0.96 or frac < 0.004:
                rec["status"] = "implausible_box"
                rec["box_frac"] = round(frac, 4)
                results.append(rec)
                print(f"  [{i}/{len(sample)}] {item_id} implausible_box frac={frac:.3f}")
                continue
            with Image.open(page_path) as image:
                crop = image.convert("RGB").crop((x0, y0, x1, y1))
                crop_path = crops_dir / f"{item_id}.jpg"
                crop.save(crop_path, format="JPEG", quality=88)
            rec["box_2d"] = box
            rec["box_frac"] = round(frac, 4)
            rec["crop_path"] = str(crop_path)
            rec["crop_size"] = [w, h]
            # quality routing: crop is saved either way so it can be eyeballed,
            # but only a clean pure-question crop counts as a pass.
            if parsed.get("is_pure_question") is False:
                rec["status"] = "answer_contaminated"
            elif parsed.get("last_option_visible") is False:
                rec["status"] = "maybe_crosspage"
            else:
                rec["status"] = "cropped"
                ok += 1
            results.append(rec)
            print(f"  [{i}/{len(sample)}] {item_id} {rec['status']} frac={frac:.3f} {w}x{h}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:200]
            rec["status"] = "http_error"
            rec["error"] = f"{exc.code}: {detail}"
            results.append(rec)
            print(f"  [{i}/{len(sample)}] {item_id} HTTP {exc.code} {detail[:80]}")
        except Exception as exc:  # noqa: BLE001 - prototype: record and continue
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            results.append(rec)
            print(f"  [{i}/{len(sample)}] {item_id} error {type(exc).__name__}: {exc}")

    summary = {
        "provider": args.provider,
        "model": model,
        "endpoint_host": endpoint.split("/")[2],
        "clean_page_only_pool": len(pool),
        "sampled": len(sample),
        "cropped_ok": ok,
        "success_rate": round(ok / len(sample), 3) if sample else 0.0,
        "total_tokens": tokens,
        "status_counts": _counts(results),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8"
    )
    _write_preview(run_dir, results)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\npreview: {run_dir / 'preview.html'}")
    return 0


def _counts(results: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r.get("status", "?")] = out.get(r.get("status", "?"), 0) + 1
    return out


def _write_preview(run_dir: Path, results: list[dict[str, Any]]) -> None:
    cards = []
    for r in results:
        status = r.get("status")
        crop = r.get("crop_path")
        page = r.get("page_image_path")
        crop_img = (
            f'<img src="file://{crop}" style="max-width:420px;border:2px solid #2a7">'
            if crop else f'<div style="color:#a22">[{status}] {r.get("error","")}</div>'
        )
        cards.append(
            f'<div style="border:1px solid #ccc;margin:10px;padding:10px;display:inline-block;vertical-align:top;width:460px">'
            f'<div><b>{r.get("item_id")}</b> · {r.get("category")} · {r.get("question_type")} · '
            f'<span style="color:{"#2a7" if status=="cropped" else "#a22"}">{status}</span></div>'
            f'<div style="font-size:12px;color:#666">{r.get("model_note","")}</div>'
            f'<div style="margin-top:6px">{crop_img}</div>'
            f'<details><summary style="font-size:12px;color:#888">整页原图</summary>'
            f'<img src="file://{page}" style="max-width:440px"></details>'
            f'</div>'
        )
    html = (
        '<!doctype html><meta charset="utf-8"><title>vision bbox crop prototype</title>'
        '<body style="font-family:system-ui">'
        f'<h2>视觉框题裁片原型 · {len([r for r in results if r.get("status")=="cropped"])}/{len(results)} 成功</h2>'
        '<p>绿框=裁出的单题图,点"整页原图"可核对框得准不准。</p>'
        + "".join(cards)
        + "</body>"
    )
    (run_dir / "preview.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
