#!/usr/bin/env python3
"""Review semantic equivalence between model and standard chemistry answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from scripts.evaluate_visual_understanding import (
    PROVIDERS,
    chat_completions_url,
    extract_json_object,
    load_env_values,
    load_jsonl,
    answers_match,
)
DEFAULT_ENV = SKILL_DIR / ".env"

PROMPT = """你是严谨的上海高中化学答案等价审核员。判断模型答案和标准答案是否语义等价。
不要因为表达更长或顺序不同就判错；但如果选择题字母不同、关键物质/方向/数值/原因不同，必须判 not_equivalent。

题目：
{stem}

标准答案：
{standard_answer}

模型答案：
{model_answer}

严格输出 JSON：
{{
  "decision": "equivalent|not_equivalent|uncertain",
  "confidence": 0到1,
  "reason": "...",
  "critical_difference": "..." 或 null
}}"""


def call_review(row: dict[str, Any], provider: str, api_key: str, model: str, base_url: str, timeout_s: int) -> tuple[dict[str, Any], dict[str, Any], float]:
    prompt = PROMPT.format(
        stem=row.get("stem", ""),
        standard_answer=row.get("standard_answer", ""),
        model_answer=row.get("model_answer", ""),
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 700,
        "messages": [
            {"role": "system", "content": "输出 JSON，不要 Markdown。"},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = response_payload["choices"][0]["message"]["content"] or ""
    return extract_json_object(content), response_payload.get("usage", {}), round(time.time() - started, 2)


def review_rows(
    rows: list[dict[str, Any]],
    provider: str,
    model: str | None,
    env_path: Path = DEFAULT_ENV,
    timeout_s: int = 60,
    force_model_review: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = PROVIDERS[provider]
    env_values = load_env_values(env_path)
    api_key = env_values.get(config["env_key"], "")
    model_name = model or config["model"]
    base_url = chat_completions_url(config, env_values)

    results: list[dict[str, Any]] = []
    for row in rows:
        if answers_match(row.get("model_answer"), row.get("standard_answer")) and not force_model_review:
            result = {"decision": "equivalent", "confidence": 1.0, "reason": "local_exact_or_rule_match", "critical_difference": None}
            usage: dict[str, Any] = {}
            latency_s = 0.0
        elif not api_key:
            result = {"decision": "error", "confidence": 0, "reason": "missing_api_key", "critical_difference": None}
            usage = {}
            latency_s = 0.0
        else:
            try:
                result, usage, latency_s = call_review(row, provider, api_key, model_name, base_url, timeout_s)
            except Exception as exc:  # noqa: BLE001
                result = {"decision": "error", "confidence": 0, "reason": type(exc).__name__, "critical_difference": str(exc)[:200]}
                usage = {}
                latency_s = 0.0
        results.append(
            {
                **row,
                "decision": result.get("decision"),
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "critical_difference": result.get("critical_difference"),
                "review_model": model_name,
                "usage": usage,
                "latency_s": latency_s,
                **({"error": result.get("reason")} if result.get("decision") == "error" else {}),
            }
        )
        time.sleep(0.3)

    summary = {
        "total": len(results),
        "decisions": {},
        "provider": provider,
        "model": model_name,
    }
    for row in results:
        summary["decisions"][row.get("decision")] = summary["decisions"].get(row.get("decision"), 0) + 1
    if summary["decisions"].get("error"):
        summary["error"] = "missing_api_key" if not api_key else "review_errors"
    return results, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Review semantic answer equivalence.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--force-model-review", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    results, summary = review_rows(
        rows,
        provider=args.provider,
        model=args.model,
        timeout_s=args.timeout_s,
        force_model_review=args.force_model_review,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    write_jsonl(args.out, results)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
