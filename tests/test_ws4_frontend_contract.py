#!/usr/bin/env python3
"""WS4 frontend/API contract checks.

Kept dependency-light because the local execution environment may not have
FastAPI installed; these tests verify the static contract Codex owns in Batch 9.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_render_report_endpoint_path_and_pending_queue_contract():
    src = _read("apps/api_v4_render.py")
    assert '@router.post("/api/render_report")' in src
    assert '"status": "pending"' in src
    assert '"reviewer": ""' in src
    assert "/tmp/yher_batch9_ws4" in src
    assert "/api/v4/items/{item_id}/rir" in src


def test_rir_renderer_handles_closed_node_set_and_reports_katex_errors():
    js = _read("apps/web/rir_renderer.js")
    for kind in ("text", "latex", "image", "table", "placeholder"):
        assert f'"{kind}"' in js
    assert "window.katex.render" in js
    assert "throwOnError: false" in js
    assert "/api/render_report" in js
    assert "loading = \"lazy\"" in js
    assert "srcset" in js
    assert "图暂缺" in js
    assert "normalizeLatexForKatex" in js
    assert "{}^{" in js


def test_preview_page_renders_stem_answer_and_uses_full_service_id_list():
    html = _read("apps/web/v4_preview.html")
    assert "rir_renderer.js" in html
    assert "zones=stem,answer" in html
    assert "随机服务题" in html
    ids = json.loads(_read("apps/web/v4_service_item_ids.json"))
    # 预览页是审计面:随机按钮要能抽到 fixable/blocked 题,故保持 R1-R4 全池 2526,
    # 不随 R5 白名单缩窄(2026-07-06 R5 apply)。
    assert len(ids) == 2526
    assert all(re.fullmatch(r"[0-9a-f]{40}", x) for x in ids)
    assert len(ids) == len(set(ids))


def _run_focused() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{failures} FAILURES" if failures else "\nALL PASS")
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_focused())
