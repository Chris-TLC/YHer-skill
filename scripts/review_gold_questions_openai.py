#!/usr/bin/env python3
"""Review gold diagnostic question candidates with an OpenAI-compatible model."""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_ENV = SKILL_DIR / ".env"
DEFAULT_IN = SKILL_DIR / "data" / "quality" / "gold_question_candidates" / "anthropic_opus48" / "combined.jsonl"
DEFAULT_OUT = SKILL_DIR / "data" / "quality" / "gold_question_candidates" / "anthropic_opus48_gpt55_review.jsonl"
DEFAULT_SUMMARY = SKILL_DIR / "data" / "quality" / "gold_question_candidates" / "anthropic_opus48_gpt55_review_summary.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_env(path: Path = DEFAULT_ENV) -> dict[str, str]:
    allowed = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_CHAT_COMPLETIONS_URL"}
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in allowed:
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key in allowed})
    return values


def chat_url(env: dict[str, str]) -> str:
    explicit = env.get("OPENAI_CHAT_COMPLETIONS_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    base = (env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE") or "https://api.openai.com").strip().rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def review_prompt(rows: list[dict[str, Any]]) -> str:
    compact = []
    for row in rows:
        compact.append(
            {
                "gold_id": row.get("gold_id"),
                "hard_hole": row.get("hard_hole"),
                "kg_node": row.get("kg_node"),
                "diagnostic_axis": row.get("diagnostic_axis"),
                "difficulty": row.get("difficulty"),
                "answer_type": row.get("answer_type"),
                "prompt": row.get("prompt"),
                "options": row.get("options"),
                "standard_answer": row.get("standard_answer"),
                "rubric": row.get("rubric"),
                "misconceptions": row.get("misconceptions"),
            }
        )
    return f"""你是上海高中化学题库审核员。请审查下面 gold diagnostic question candidates 是否适合作为 silver/gold 候选。

审核重点：
1. 化学事实、方程式、电荷守恒、浓度关系、实验现象和流程逻辑是否正确。
2. 标准答案是否与题干一致。
3. rubric 是否能判分，must_have 是否覆盖关键点。
4. misconceptions 是否真正对应常见错因。
5. 是否存在外部视觉依赖、表述歧义、超纲严重、证据不足或题目不可作答。

请只输出 JSON 数组。每个输入题输出一个对象：
{{
  "gold_id": "...",
  "review_decision": "approve|revise|reject",
  "severity": "none|minor|major|critical",
  "reasons": ["..."],
  "chemistry_risk": "...",
  "suggested_fix": "..."
}}

如果题目基本可用但有小措辞问题，用 revise/minor；如果答案或核心化学错误，用 reject/critical。

输入题目：
{json.dumps(compact, ensure_ascii=False, indent=2)}
"""


def extract_json_array(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", stripped)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("review_response_not_array")
    return data


def call_review(prompt: str, api_key: str, url: str, model: str, timeout_s: int) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 3500,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"] or ""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Review gold candidates with GPT/OpenAI-compatible model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    env = load_env()
    api_key = env.get("OPENAI_API_KEY", "")
    if not api_key:
        print(json.dumps({"ok": False, "error": "missing_openai_api_key"}, ensure_ascii=False))
        return 2

    rows = load_jsonl(args.input)
    url = chat_url(env)
    reviews: list[dict[str, Any]] = []
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("", encoding="utf-8")
    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        parsed: list[dict[str, Any]] | None = None
        for attempt in range(1, args.retries + 2):
            print(f"REVIEWING rows {batch_start + 1}-{batch_start + len(batch)} attempt {attempt}", flush=True)
            try:
                text = call_review(review_prompt(batch), api_key=api_key, url=url, model=args.model, timeout_s=args.timeout_s)
                parsed = extract_json_array(text)
                break
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                ConnectionResetError,
                http.client.HTTPException,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                if attempt >= args.retries + 1:
                    raise RuntimeError(
                        f"review rows {batch_start + 1}-{batch_start + len(batch)} failed after {attempt} attempts: {type(exc).__name__}: {exc}"
                    ) from exc
                print(f"RETRY {type(exc).__name__}: {str(exc)[:180]}", flush=True)
                time.sleep(2 * attempt)
        if parsed is None:
            raise RuntimeError(f"review rows {batch_start + 1}-{batch_start + len(batch)} produced no parsed output")
        reviews.extend(parsed)
        if args.write:
            with args.out.open("a", encoding="utf-8") as f:
                for row in parsed:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        time.sleep(0.5)

    summary = {
        "ok": True,
        "model": args.model,
        "input": str(args.input),
        "total": len(reviews),
        "decisions": {},
        "severities": {},
    }
    for row in reviews:
        decision = str(row.get("review_decision") or "")
        severity = str(row.get("severity") or "")
        summary["decisions"][decision] = summary["decisions"].get(decision, 0) + 1
        summary["severities"][severity] = summary["severities"].get(severity, 0) + 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out}")
        print(f"WROTE {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
