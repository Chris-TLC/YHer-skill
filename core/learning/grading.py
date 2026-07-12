"""Evidence-bound LLM rubric grading behind the server trust boundary."""

from __future__ import annotations

import json
import math
from typing import Any

from .item_catalog import CatalogItem


class ChatRubricGrader:
    """Adapt the provider-neutral chat client to ``LLMGrader``."""

    def __init__(self, client):
        self.client = client

    def __call__(self, item: CatalogItem, submission: str) -> dict[str, Any]:
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": "你只返回符合指定 schema 的 JSON，不附加 Markdown 围栏。",
                },
                {"role": "user", "content": build_grading_prompt(item, submission)},
            ],
            max_tokens=900,
            temperature=0.0,
        )
        content = response.get("content") if isinstance(response, dict) else response
        if isinstance(content, str):
            content = json.loads(_strip_json_fence(content))
        result = _validated_result(content)
        usage = response.get("usage") if isinstance(response, dict) else {}
        result["usage"] = {
            "input_tokens": max(0, int((usage or {}).get("input_tokens") or 0)),
            "output_tokens": max(0, int((usage or {}).get("output_tokens") or 0)),
        }
        try:
            result["cost_yuan"] = max(0.0, float(response.get("cost_yuan") or 0.0))
        except (AttributeError, TypeError, ValueError):
            result["cost_yuan"] = 0.0
        return result


def build_grading_prompt(item: CatalogItem, submission: str) -> str:
    evidence = {
        "question": item.stem_text,
        "options": dict(item.options),
        "expected_response": list(item.answer_values),
        "rubric": [dict(point) for point in item.rubric],
        "source": item.source_label,
        "difficulty": item.difficulty,
        "student_response": str(submission),
    }
    return (
        "你是上海高中化学自由作答判分器。只能依据服务端题面、标准答案和 rubric 判定，"
        "不得因表达风格扣分。输出 correct(布尔)、error_code(简短错因码)、confidence(0到1)、"
        "likelihood(长度4的非负数组，依次对应 mastered/prerequisite_gap/"
        "concept_confusion/uncertain)。证据不充分时降低 confidence，不得虚构题意。\n"
        f"服务端证据：{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}"
    )


def _validated_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("correct"), bool):
        raise ValueError("grader result must contain a boolean correct field")
    confidence = float(value.get("confidence"))
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("grader confidence must be finite and within [0, 1]")
    likelihood = value.get("likelihood")
    if not isinstance(likelihood, (list, tuple)) or len(likelihood) != 4:
        raise ValueError("grader likelihood must contain four values")
    clean = [float(entry) for entry in likelihood]
    if not all(math.isfinite(entry) and entry >= 0.0 for entry in clean) or sum(clean) <= 0:
        raise ValueError("grader likelihood must be finite and nonnegative")
    return {
        "correct": value["correct"],
        "error_code": str(value.get("error_code") or "")[:120] or None,
        "confidence": confidence,
        "likelihood": clean,
    }


def _strip_json_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text
