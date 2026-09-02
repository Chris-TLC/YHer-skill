#!/usr/bin/env python3
"""Generate hard-hole gold diagnostic question candidates with Anthropic.

The script writes model drafts only as silver candidates. They must still pass
the local router and review before any production use.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
DEFAULT_OUT_DIR = SKILL_DIR / "data" / "quality" / "gold_question_candidates" / "anthropic_opus48"

MODEL_DEFAULT = "claude-opus-4-8"
PROFILE_AXES = ["基础概念", "审题入口", "步骤执行", "应用迁移", "综合推理"]
FORBIDDEN_PROMPT_PATTERNS = ["如图", "见图", "下图", "上图", "图中", "所示装置图", "根据装置图", "见表", "如下表", "截图"]

ITEM_PLANS: dict[str, list[dict[str, str]]] = {
    "solution_three_balances": [
        {"axis": "entry", "difficulty": "T1", "answer_type": "single_choice", "node": "守恒类型识别"},
        {"axis": "entry", "difficulty": "T1", "answer_type": "single_choice", "node": "电荷守恒入口"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "single_choice", "node": "物料守恒"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "single_choice", "node": "质子守恒"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "free_text", "node": "守恒式适用边界"},
        {"axis": "procedure", "difficulty": "T2", "answer_type": "free_text", "node": "电荷守恒列式"},
        {"axis": "procedure", "difficulty": "T3", "answer_type": "free_text", "node": "物料守恒列式"},
        {"axis": "procedure", "difficulty": "T3", "answer_type": "free_text", "node": "质子守恒列式"},
        {"axis": "transfer", "difficulty": "T3", "answer_type": "multi_step", "node": "混合溶液守恒迁移"},
        {"axis": "transfer", "difficulty": "T3", "answer_type": "multi_step", "node": "滴定曲线中的守恒判断"},
        {"axis": "integrated", "difficulty": "T4", "answer_type": "single_choice", "node": "离子浓度排序综合"},
        {"axis": "integrated", "difficulty": "T4", "answer_type": "multi_step", "node": "三大守恒综合诊断"},
    ],
    "process_flow": [
        {"axis": "entry", "difficulty": "T1", "answer_type": "single_choice", "node": "流程原料产物识别"},
        {"axis": "entry", "difficulty": "T1", "answer_type": "single_choice", "node": "流程步骤顺序识别"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "single_choice", "node": "氧化还原角色判断"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "single_choice", "node": "沉淀与转化原理"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "free_text", "node": "循环物质与副产物判断"},
        {"axis": "procedure", "difficulty": "T2", "answer_type": "free_text", "node": "条件控制解释"},
        {"axis": "procedure", "difficulty": "T3", "answer_type": "multi_step", "node": "分离提纯步骤解释"},
        {"axis": "procedure", "difficulty": "T3", "answer_type": "multi_step", "node": "流程方程式配平"},
        {"axis": "transfer", "difficulty": "T3", "answer_type": "multi_step", "node": "陌生流程类比迁移"},
        {"axis": "transfer", "difficulty": "T3", "answer_type": "multi_step", "node": "试剂选择迁移"},
        {"axis": "integrated", "difficulty": "T4", "answer_type": "multi_step", "node": "流程定量与守恒综合"},
        {"axis": "integrated", "difficulty": "T4", "answer_type": "multi_step", "node": "多步氧化还原流程综合"},
    ],
    "integrated_experiment": [
        {"axis": "entry", "difficulty": "T1", "answer_type": "single_choice", "node": "安全与尾气处理"},
        {"axis": "entry", "difficulty": "T1", "answer_type": "single_choice", "node": "实验目的识别"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "single_choice", "node": "干燥剂选择"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "single_choice", "node": "检验试剂选择"},
        {"axis": "concept", "difficulty": "T2", "answer_type": "free_text", "node": "现象与离子反应"},
        {"axis": "procedure", "difficulty": "T2", "answer_type": "free_text", "node": "操作顺序与防干扰"},
        {"axis": "procedure", "difficulty": "T3", "answer_type": "multi_step", "node": "变量控制"},
        {"axis": "procedure", "difficulty": "T3", "answer_type": "multi_step", "node": "误差分析"},
        {"axis": "transfer", "difficulty": "T3", "answer_type": "multi_step", "node": "陌生实验迁移"},
        {"axis": "transfer", "difficulty": "T3", "answer_type": "multi_step", "node": "方案评价与改进"},
        {"axis": "integrated", "difficulty": "T4", "answer_type": "multi_step", "node": "综合实验设计"},
        {"axis": "integrated", "difficulty": "T4", "answer_type": "multi_step", "node": "现象证据链综合推理"},
    ],
}


def load_env(path: Path = DEFAULT_ENV) -> dict[str, str]:
    values: dict[str, str] = {}
    allowed = {"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE", "ANTHROPIC_MESSAGES_URL"}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in allowed:
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key in allowed})
    return values


def messages_url(env: dict[str, str]) -> str:
    explicit = env.get("ANTHROPIC_MESSAGES_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    base = (env.get("ANTHROPIC_BASE_URL") or env.get("ANTHROPIC_API_BASE") or "https://api.anthropic.com").strip().rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def call_anthropic(
    prompt: str,
    api_key: str,
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    return "".join(
        block.get("text", "")
        for block in response_payload.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


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
        raise ValueError("response_json_is_not_array")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError("response_array_contains_non_object")
    return data


def prompt_for(hard_hole: str, batch: list[dict[str, str]], start_index: int) -> str:
    item_specs = []
    for offset, spec in enumerate(batch, start=start_index):
        item_specs.append({**spec, "gold_id": f"anthropic_{hard_hole}_{offset:03d}"})
    return f"""你是上海高中化学诊断题库总编辑。请生成严格 JSON 数组，不要 Markdown，不要解释。

目标：为 YHer 化学诊断补齐 hard_hole={hard_hole} 的 silver candidate gold questions。

每个对象必须完整包含这些字段：
- gold_id
- hard_hole
- kg_node
- diagnostic_axis: entry/concept/procedure/transfer/integrated 之一
- difficulty: T1/T2/T3/T4 之一
- answer_type: single_choice/free_text/multi_step 之一
- prompt
- options: single_choice 必须是 A/B/C/D 对象；其他题为 null
- standard_answer
- rubric: 至少 3 个对象，每个对象必须有 point_id, desc, must_have, score, accept_patterns, reject_patterns；至少 1 个 must_have=true
- misconceptions: 至少 2 个对象，每个对象必须有 wrong_pattern, reveals, profile_update, recommended_remediation
- profile_update 必须是 {{axis, direction, tag}}，axis 只能从 {PROFILE_AXES} 中选，direction 必须是 weaken
- profile_evidence_rule: {{can_update_profile:true, max_weight:"medium", mastery_signal, weakness_signal}}
- verification_use: 必须严格等于 ["diagnosis","post_video_verification"]
- source_type: 必须严格等于 "model_candidate"
- review_status: 必须严格等于 "silver_candidate"
- risk_notes: 数组，可以为空

硬约束：
1. 只能生成文本题，prompt 中禁止出现这些外部依赖词：{FORBIDDEN_PROMPT_PATTERNS}。
2. 不要依赖图片、表格截图或外部材料；必要数据必须写在题干文本内。
3. 题目必须能诊断一个具体薄弱点，而不是泛泛考知识。
4. 不得把模型候选标为 production 可用；不要输出 production_profile_evidence_allowed 字段。
5. max_weight 不得为 high。
6. 化学答案必须自洽；守恒式要包含 H+ / OH- 时不能漏；电荷数系数要正确；实验结论不能超出现象证据；流程题要明确每步物质和电子/离子变化。

本批次必须生成 exactly {len(item_specs)} 个对象，按以下规格逐一生成：
{json.dumps(item_specs, ensure_ascii=False, indent=2)}

输出格式：只输出 JSON 数组。"""


def normalize_row(row: dict[str, Any], hard_hole: str, expected: dict[str, str]) -> dict[str, Any]:
    row = dict(row)
    row["hard_hole"] = hard_hole
    row["gold_id"] = expected["gold_id"]
    row["diagnostic_axis"] = expected["axis"]
    row["difficulty"] = expected["difficulty"]
    row["answer_type"] = expected["answer_type"]
    row.setdefault("kg_node", expected["node"])
    row["verification_use"] = ["diagnosis", "post_video_verification"]
    row["source_type"] = "model_candidate"
    row["review_status"] = "silver_candidate"
    row.setdefault("risk_notes", [])
    rule = row.get("profile_evidence_rule")
    if not isinstance(rule, dict):
        rule = {}
    rule["can_update_profile"] = True
    if rule.get("max_weight") not in {"low", "medium"}:
        rule["max_weight"] = "medium"
    row["profile_evidence_rule"] = rule
    if row["answer_type"] == "single_choice":
        row.setdefault("options", {})
    else:
        row["options"] = None
    if isinstance(row.get("rubric"), list):
        for point in row["rubric"]:
            if isinstance(point, dict):
                point.setdefault("accept_patterns", [])
                point.setdefault("reject_patterns", [])
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Anthropic gold question candidates.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--hard-hole", choices=sorted(ITEM_PLANS), default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=9000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    env = load_env()
    api_key = env.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(json.dumps({"ok": False, "error": "missing_anthropic_api_key"}, ensure_ascii=False))
        return 2
    url = messages_url(env)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    holes = [args.hard_hole] if args.hard_hole else list(ITEM_PLANS)
    for hard_hole in holes:
        rows: list[dict[str, Any]] = []
        plan = ITEM_PLANS[hard_hole]
        for batch_start in range(0, len(plan), args.batch_size):
            batch = plan[batch_start : batch_start + args.batch_size]
            expected_specs = [
                {**spec, "gold_id": f"anthropic_{hard_hole}_{index:03d}"}
                for index, spec in enumerate(batch, start=batch_start + 1)
            ]
            prompt = prompt_for(hard_hole, batch, batch_start + 1)
            parsed: list[dict[str, Any]] | None = None
            for attempt in range(1, args.retries + 2):
                print(
                    f"GENERATING {hard_hole} rows {batch_start + 1}-{batch_start + len(batch)} attempt {attempt}",
                    flush=True,
                )
                try:
                    text = call_anthropic(
                        prompt=prompt,
                        api_key=api_key,
                        url=url,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout_s=args.timeout_s,
                    )
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
                            f"{hard_hole} batch {batch_start + 1}-{batch_start + len(batch)} failed after {attempt} attempts: {type(exc).__name__}: {exc}"
                        ) from exc
                    print(f"RETRY {type(exc).__name__}: {str(exc)[:180]}", flush=True)
                    time.sleep(2 * attempt)
            if parsed is None:
                raise RuntimeError(f"{hard_hole} batch {batch_start + 1}: no parsed response")
            if len(parsed) != len(batch):
                raise ValueError(f"{hard_hole} batch {batch_start + 1}: expected {len(batch)} rows, got {len(parsed)}")
            for row, expected in zip(parsed, expected_specs):
                rows.append(normalize_row(row, hard_hole, expected))
            time.sleep(0.5)
        write_jsonl(args.out_dir / f"{hard_hole}.jsonl", rows)
        all_rows.extend(rows)
    write_jsonl(args.out_dir / "combined.jsonl", all_rows)
    summary = {
        "ok": True,
        "model": args.model,
        "out_dir": str(args.out_dir),
        "rows": len(all_rows),
        "by_hard_hole": dict(Counter(row["hard_hole"] for row in all_rows)),
        "by_axis": dict(Counter(row["diagnostic_axis"] for row in all_rows)),
        "by_difficulty": dict(Counter(row["difficulty"] for row in all_rows)),
    }
    (args.out_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
