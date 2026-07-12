"""Evidence-bound explanation generation with a deterministic offline fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any


_PUBLIC_FIELDS = (
    "title",
    "diagnosis",
    "worked_example",
    "causal_chain",
    "exam_strategy",
    "analogy_used",
)

_SAFE_EXAM_STRATEGIES = (
    "先列出题干给定量和待求量，再逐步对应标准解中的关系。",
    "每完成一步就核对单位、原子数、电荷或题目给定判据，不跳过矛盾。",
    "最后把结论代回题干，并与标准答案逐项核对。",
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

    def validate(
        self, *, explanation: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": "你只返回符合指定 schema 的 JSON，不附加 Markdown 围栏。",
                },
                {
                    "role": "user",
                    "content": build_explanation_review_prompt(explanation, context),
                },
            ],
            max_tokens=900,
            temperature=0.0,
        )
        content = response.get("content") if isinstance(response, dict) else response
        if isinstance(content, str):
            content = json.loads(_strip_json_fence(content))
        result = _review_projection(content)
        result["usage"] = _usage_projection(
            response.get("usage") if isinstance(response, dict) else None
        )
        result["cost_yuan"] = _nonnegative_float(
            response.get("cost_yuan") if isinstance(response, dict) else 0.0
        )
        return result


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

        explanation_model = str(
            os.environ.get("YHER_EXPLANATION_MODEL") or "deepseek-chat"
        ).strip()
        explanation_client = LLMClient(
            provider="deepseek", model=explanation_model, api_key=api_key
        )
        grader_client = LLMClient(provider="deepseek", api_key=api_key)
        return (
            ChatExplanationProvider(explanation_client),
            ChatRubricGrader(grader_client),
        )
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
        "5. 绑定来源、标准解与本次实际作答结果，不得声称长期学习效果。\n"
        "6. expected_response 是权威答案；不得引入权威答案中没有的产物、数值或结论。\n"
        "7. 涉及方程式、计量或氧化还原时，逐项复核原子守恒、电荷守恒、电子得失守恒。\n"
        "8. 输出前逐句检查 worked_example 与 causal_chain；任何一步不自洽就不得输出。\n"
        "9. result_summary 是唯一作答归因；全对时禁止声称答错、困难或薄弱。\n"
        "10. causal_chain 只能逐字复制 solution_steps；exam_strategy 只能从以下安全策略中选择："
        f"{json.dumps(_SAFE_EXAM_STRATEGIES, ensure_ascii=False)}。\n"
        "注意：服务端会把公开事实重新投影到已核验标准解，任何自补事实都不会展示。\n"
        "只返回 JSON 对象，键为 title, diagnosis, worked_example, "
        "causal_chain(字符串数组), exam_strategy(字符串数组), analogy_used(布尔)。\n"
        f"服务端证据：{payload}"
    )


def build_explanation_review_prompt(
    explanation: dict[str, Any], context: dict[str, Any]
) -> str:
    evidence = json.dumps(context, ensure_ascii=False, sort_keys=True)
    candidate = json.dumps(explanation, ensure_ascii=False, sort_keys=True)
    return (
        "你是高中化学讲解的发布前审校员。逐句对照服务端 evidence，"
        "其中 expected_response 是权威答案。重新计算所有原子守恒、电荷守恒、"
        "电子得失守恒、化学式下标和物质的量关系。任何产物、价态、电子数、"
        "原子数或结论与 evidence 冲突，或候选内部前后矛盾，都必须 valid=false。"
        "不能因为最终答案碰巧正确而放过中间错误。只返回 valid(布尔) 和 errors(字符串数组)。\n"
        f"服务端 evidence：{evidence}\n候选讲解：{candidate}"
    )


def generate_explanation(
    provider,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a public explanation and a provider-neutral internal audit payload."""
    if not _verified_evidence(context):
        explanation = evidence_fallback(context)
        return explanation, {
            "generation_status": "evidence_fallback",
            "grounding_status": "no_verified_solution",
            "quality_status": "not_run",
            "quality_failures": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "cost_yuan": 0.0,
            "public_explanation": explanation,
        }
    prompt = build_explanation_prompt(context)
    response: Any = None
    quality_failures: list[str] = []
    try:
        if provider is None:
            raise RuntimeError("offline")
        method = getattr(provider, "generate", provider)
        response = method(prompt=prompt, context=context)
        content = response.get("content") if isinstance(response, dict) else response
        if isinstance(content, str):
            content = json.loads(_strip_json_fence(content))
        proposed = _public_projection(content)
        quality_failures = _deterministic_quality_failures(proposed, context)
        explanation = _authoritative_projection(proposed, context)
        explanation["status"] = "generated"
        usage = _combined_usage(response)
        cost = _combined_cost(response)
        audit = {
            "generation_status": "generated",
            "grounding_status": "authoritative_projection",
            "quality_status": "projected",
            "quality_failures": quality_failures,
            "usage": usage,
            "cost_yuan": cost,
            "public_explanation": explanation,
        }
        return explanation, audit
    except Exception:  # noqa: BLE001 - the learning loop must remain available offline
        explanation = _authoritative_projection({}, context)
        explanation["status"] = "offline_fallback"
        usage = _combined_usage(response)
        cost = _combined_cost(response)
        return explanation, {
            "generation_status": "offline_fallback",
            "grounding_status": "authoritative_projection",
            "quality_status": "projected" if quality_failures else "not_run",
            "quality_failures": quality_failures,
            "usage": usage,
            "cost_yuan": cost,
            "public_explanation": explanation,
        }


def offline_fallback(context: dict[str, Any]) -> dict[str, Any]:
    evidence = list(context.get("evidence") or [])
    summary = _result_summary(context)
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
        "diagnosis": _diagnosis_text(summary, source),
        "worked_example": (
            f"回到题干「{question[:120]}」：先列已知条件和待求量，把每个数值或现象放到对应变量旁；"
            f"再按「{criterion[:160]}」逐步推到结论。"
        ),
        "causal_chain": ["列出已知条件与待求结论", "确定条件改变影响的中间量", "沿中间量推到结论", "用题目判据回查"],
        "exam_strategy": ["先圈条件与限定词", "再写中间关系", "最后对照判据验算"],
        "analogy_used": False,
    }


def evidence_fallback(context: dict[str, Any]) -> dict[str, Any]:
    """Explain the evidence boundary without asking a model to complete missing facts."""
    summary = _result_summary(context)
    return {
        "status": "evidence_fallback",
        "title": f"{context.get('node') or '本考点'}诊断复盘",
        "diagnosis": _diagnosis_text(summary, "本次题组"),
        "worked_example": (
            "本轮题目的库内标准解尚未通过讲解证据门，因此这里不补写反应产物、"
            "中间数值或推导步骤。请保留本次作答记录，改用已核验例题继续学习。"
        ),
        "causal_chain": ["保留本次判分结果", "不扩写未核验答案", "切换到已核验例题"],
        "exam_strategy": list(_SAFE_EXAM_STRATEGIES),
        "analogy_used": False,
    }


def _verified_evidence(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in (context.get("evidence") or [])
        if isinstance(row, dict) and row.get("solution_steps")
    ]


def _authoritative_projection(
    proposed: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Project only reviewed facts; the model may select but cannot invent content."""
    primary = sorted(
        _verified_evidence(context),
        key=lambda row: (
            {"incorrect": 0, "deferred": 1, "correct": 2}.get(str(row.get("result")), 3),
            -_float_or_zero(row.get("difficulty")),
            str(row.get("source") or ""),
            str(row.get("question") or ""),
        ),
    )[0]
    steps = [str(step).strip() for step in primary["solution_steps"] if str(step).strip()]
    selected_steps = [step for step in proposed.get("causal_chain", []) if step in steps]
    selected_strategies = [
        strategy
        for strategy in proposed.get("exam_strategy", [])
        if strategy in _SAFE_EXAM_STRATEGIES
    ]
    answers = "；".join(str(value) for value in (primary.get("expected_response") or []))
    source = str(primary.get("source") or "本次题组")
    question = str(primary.get("question") or "本次诊断题")
    anchor_node = str(primary.get("node") or context.get("node") or "本考点")
    key_insight = str(primary.get("key_insight") or "").strip()
    summary = _result_summary(context)
    needs_layered_scaffold = (
        _float_or_zero(primary.get("difficulty")) >= 0.75
        or summary["incorrect"] >= 2
    )
    worked_parts = [
        f"零基础起点：{key_insight or '先明确题目要求、已知条件和待求结论。'}",
        f"例题（{source}）：{question}",
    ]
    if needs_layered_scaffold:
        worked_parts.append(
            "分层支架：先读清题目要求与已知条件，再逐条执行已核验步骤；"
            "每一步只使用题干与标准解给出的关系。"
        )
    worked_parts.extend(["已核验步骤：", *steps])
    if needs_layered_scaffold:
        worked_parts.append(
            "验算闭环：逐项对照题干、已核验步骤和标准答案；若三者不能对应，"
            "停在当前步骤重新核对。"
        )
    if answers:
        worked_parts.append(f"标准答案：{answers}")
    return {
        "title": f"{anchor_node}标准解复盘",
        "diagnosis": _diagnosis_text(_result_summary(context), source),
        "worked_example": "\n".join(worked_parts),
        "causal_chain": selected_steps or steps,
        "exam_strategy": selected_strategies or list(_SAFE_EXAM_STRATEGIES),
        "analogy_used": False,
    }


def _result_summary(context: dict[str, Any]) -> dict[str, int]:
    raw = context.get("result_summary") or {}
    if raw:
        return {
            key: max(0, int(raw.get(key) or 0))
            for key in ("total", "correct", "incorrect", "deferred")
        }
    evidence = list(context.get("evidence") or [])
    return {
        "total": len(evidence),
        "correct": sum(row.get("result") == "correct" for row in evidence),
        "incorrect": sum(row.get("result") == "incorrect" for row in evidence),
        "deferred": sum(row.get("result") == "deferred" for row in evidence),
    }


def _diagnosis_text(summary: dict[str, int], source: str) -> str:
    total = summary["total"]
    correct = summary["correct"]
    if total and correct == total:
        result = f"本次诊断 {correct}/{total} 全部通过，没有错误项。"
    else:
        result = (
            f"本次诊断共 {total} 题：{correct} 题通过、{summary['incorrect']} 题未通过、"
            f"{summary['deferred']} 题待判。"
        )
    return f"{result}以 {source} 的已核验标准解为锚点复盘。"


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def _review_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("valid"), bool):
        raise ValueError("explanation review must contain a boolean valid field")
    errors = _text_list(value.get("errors"), 8, 300)
    if value["valid"] is False and not errors:
        errors = ["semantic_validation_failed"]
    return {"valid": value["valid"], "errors": errors}


def _deterministic_quality_failures(
    explanation: dict[str, Any], context: dict[str, Any]
) -> list[str]:
    rendered = " ".join(
        [
            str(explanation.get("title") or ""),
            str(explanation.get("diagnosis") or ""),
            str(explanation.get("worked_example") or ""),
            *[str(value) for value in explanation.get("causal_chain") or []],
            *[str(value) for value in explanation.get("exam_strategy") or []],
        ]
    )
    normalized = _normalize_chemistry_text(rendered)
    compact = re.sub(r"\s+", "", normalized)
    evidence = _normalize_chemistry_text(
        json.dumps(context, ensure_ascii=False, sort_keys=True)
    )
    failures: list[str] = []

    if "naclo3" in evidence:
        chlorate_patterns = (
            r"naclo3.{0,80}(?:\+5|5价).{0,30}(?:降|到|->).{0,10}(?:0价|0)",
            r"naclo3.{0,100}(?:得到|得)5(?:个)?电子",
            r"naclo3中无cl",
            r"0价或-1价",
            r"通常生成cl2",
        )
        if any(re.search(pattern, compact, re.IGNORECASE) for pattern in chlorate_patterns):
            failures.append("chlorate_reduction_contradicts_expected_product")
    if "6fecl2" in evidence and re.search(
        r"(?:6fecl2中(?:有)?6(?:个)?cl|左侧有6fe、6cl)", compact, re.IGNORECASE
    ):
        failures.append("fecl2_chlorine_atom_count_is_inconsistent")
    if "o2f2" in evidence and "1×4=4" in compact:
        failures.append("o2f2_electron_count_ignores_formula_subscript")
    if "电子得失不相等" in compact:
        failures.append("explanation_admits_unresolved_electron_imbalance")
    return list(dict.fromkeys(failures))


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


def _combined_usage(*values: Any) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0}
    for value in values:
        usage = _usage_projection(value.get("usage") if isinstance(value, dict) else None)
        totals["input_tokens"] += usage["input_tokens"]
        totals["output_tokens"] += usage["output_tokens"]
    return totals


def _combined_cost(*values: Any) -> float:
    return sum(
        _nonnegative_float(value.get("cost_yuan") if isinstance(value, dict) else 0.0)
        for value in values
    )


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _normalize_chemistry_text(value: str) -> str:
    return str(value).translate(
        str.maketrans("₀₁₂₃₄₅₆₇₈₉→−", "0123456789>-",)
    ).lower()


def _dynamic_max_tokens(context: dict[str, Any]) -> int:
    difficulties: list[float] = []
    for row in context.get("evidence") or []:
        try:
            difficulties.append(float(row.get("difficulty")))
        except (AttributeError, TypeError, ValueError):
            continue
    maximum = max(difficulties, default=0.5)
    if maximum <= 0.35:
        return 3_200
    if maximum <= 0.7:
        return 4_200
    return 5_200


def _strip_json_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text
