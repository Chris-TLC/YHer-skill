"""Evidence-bound explanation generation with a deterministic offline fallback."""

from __future__ import annotations

import json
import os
from typing import Any


_PUBLIC_FIELDS = (
    "title",
    "diagnosis",
    "worked_example",
    "causal_chain",
    "exam_strategy",
    "analogy_used",
)


class ChatExplanationProvider:
    """Adapter for the repository's provider-neutral ``LLMClient.chat`` API."""

    def __init__(self, client):
        self.client = client

    def generate(self, *, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        return self.client.chat(
            [
                {
                    "role": "system",
                    "content": "你只返回符合指定 schema 的 JSON，不附加 Markdown 围栏。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=_dynamic_max_tokens(context),
            temperature=0.2,
        )


def environment_explanation_provider():
    """Opt-in paid provider; missing key/dependency safely selects offline mode."""
    return environment_learning_adapters()[0]


def environment_learning_adapters():
    """Build explanation and grading adapters around one opt-in paid client."""
    enabled = str(os.environ.get("YHER_ENABLE_PAID_LLM") or "").strip().lower()
    api_key = str(os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if enabled not in {"1", "true", "yes"} or not api_key:
        return None, None
    try:
        from adapters.llm_client import LLMClient
        from core.learning.grading import ChatRubricGrader

        client = LLMClient(provider="deepseek", api_key=api_key)
        return ChatExplanationProvider(client), ChatRubricGrader(client)
    except Exception:  # noqa: BLE001 - startup must retain the offline capability
        return None, None


def build_explanation_prompt(context: dict[str, Any]) -> str:
    """Build a private provider prompt from already submitted, server-held evidence."""
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
    return (
        "你是上海高中化学讲解员。只能使用下方服务端证据，不得补写未提供的题目事实。\n"
        "讲解要求：\n"
        "1. 必须做题干数据代入，不能只说抽象定义。\n"
        "2. 明确列出变量，再写从条件到结论的因果链。\n"
        "3. 按证据中的实际难度动态调整长度：低难度仍要零基础可跟，高难度要拆出中间步骤。\n"
        "4. 只在必要时使用比喻；不需要比喻时 analogy_used=false。\n"
        "5. 绑定来源、判据与本次实际作答结果，不得声称长期学习效果。\n"
        "只返回 JSON 对象，键为 title, diagnosis, worked_example, "
        "causal_chain(字符串数组), exam_strategy(字符串数组), analogy_used(布尔)。\n"
        f"服务端证据：{payload}"
    )


def generate_explanation(
    provider,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a public explanation and a provider-neutral internal audit payload."""
    prompt = build_explanation_prompt(context)
    try:
        if provider is None:
            raise RuntimeError("offline")
        method = getattr(provider, "generate", provider)
        response = method(prompt=prompt, context=context)
        content = response.get("content") if isinstance(response, dict) else response
        if isinstance(content, str):
            content = json.loads(_strip_json_fence(content))
        explanation = _public_projection(content)
        explanation["status"] = "generated"
        usage = _usage_projection(response.get("usage") if isinstance(response, dict) else None)
        cost = _nonnegative_float(response.get("cost_yuan") if isinstance(response, dict) else 0.0)
        audit = {
            "generation_status": "generated",
            "usage": usage,
            "cost_yuan": cost,
            "public_explanation": explanation,
        }
        return explanation, audit
    except Exception:  # noqa: BLE001 - the learning loop must remain available offline
        explanation = offline_fallback(context)
        return explanation, {
            "generation_status": "offline_fallback",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "cost_yuan": 0.0,
            "public_explanation": explanation,
        }


def offline_fallback(context: dict[str, Any]) -> dict[str, Any]:
    evidence = list(context.get("evidence") or [])
    incorrect = sum(row.get("result") == "incorrect" for row in evidence)
    source = next((str(row.get("source")) for row in evidence if row.get("source")), "本次题组")
    question = next(
        (str(row.get("question")) for row in evidence if row.get("question")),
        "本次诊断题",
    )
    criterion = next(
        (
            str(point.get("desc"))
            for row in evidence
            for point in (row.get("criteria") or [])
            if isinstance(point, dict) and point.get("desc")
        ),
        "题目的已知条件与待求结论必须逐项对应",
    )
    return {
        "status": "offline_fallback",
        "title": f"{context.get('node') or '本考点'}诊断复盘",
        "diagnosis": (
            f"本次 {len(evidence)} 条有效作答中有 {incorrect} 条未通过确定性判据；"
            f"以 {source} 的考法为锚点复盘。"
        ),
        "worked_example": (
            f"回到题干「{question[:120]}」：先列已知条件和待求量，把每个数值或现象放到对应变量旁；"
            f"再按「{criterion[:160]}」逐步推到结论。"
        ),
        "causal_chain": ["列出已知条件与待求结论", "确定条件改变影响的中间量", "沿中间量推到结论", "用题目判据回查"],
        "exam_strategy": ["先圈条件与限定词", "再写中间关系", "最后对照判据验算"],
        "analogy_used": False,
    }


def _public_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("explanation content must be an object")
    projected = {
        "title": _text(value.get("title"), 120),
        "diagnosis": _text(value.get("diagnosis"), 2_000),
        "worked_example": _text(value.get("worked_example"), 4_000),
        "causal_chain": _text_list(value.get("causal_chain"), 8, 500),
        "exam_strategy": _text_list(value.get("exam_strategy"), 8, 500),
        "analogy_used": bool(value.get("analogy_used", False)),
    }
    if not all(projected[key] for key in _PUBLIC_FIELDS[:5]):
        raise ValueError("explanation content is incomplete")
    return projected


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _text_list(value: Any, max_items: int, item_limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_text(item, item_limit) for item in value[:max_items] if _text(item, item_limit)]


def _usage_projection(value: Any) -> dict[str, int]:
    value = value if isinstance(value, dict) else {}
    return {
        "input_tokens": max(0, int(value.get("input_tokens") or 0)),
        "output_tokens": max(0, int(value.get("output_tokens") or 0)),
    }


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _dynamic_max_tokens(context: dict[str, Any]) -> int:
    difficulties: list[float] = []
    for row in context.get("evidence") or []:
        try:
            difficulties.append(float(row.get("difficulty")))
        except (AttributeError, TypeError, ValueError):
            continue
    maximum = max(difficulties, default=0.5)
    if maximum <= 0.35:
        return 900
    if maximum <= 0.7:
        return 1_400
    return 2_000


def _strip_json_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text
