#!/usr/bin/env python3
"""
Evaluate multimodal understanding records for the visual eval set.

Default mode is offline: it does not call any paid model. It can import existing
pilot results and marks the remaining eval items as not_evaluated.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable
import urllib.error
import urllib.request
from io import BytesIO

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_EVAL_SET = SKILL_DIR / "data" / "evals" / "visual_item_eval_set.jsonl"
DEFAULT_RESULTS = SKILL_DIR / "data" / "evals" / "visual_understanding_results.jsonl"
DEFAULT_SUMMARY = SKILL_DIR / "data" / "evals" / "visual_understanding_summary.json"
DEFAULT_PILOT = Path("/tmp/yher_multimodal_pilot.json")
DEFAULT_PDF_ITEMS = SKILL_DIR / "data" / "from_pdf" / "all_from_pdf_v3.jsonl"
DEFAULT_ENV = SKILL_DIR / ".env"

PROVIDERS = {
    "qwen-vl": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen3-vl-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "base_url_env": ["DASHSCOPE_CHAT_COMPLETIONS_URL", "DASHSCOPE_BASE_URL"],
    },
    "gpt-4o": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "base_url_env": ["OPENAI_CHAT_COMPLETIONS_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"],
    },
    "gpt-5.5": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-5.5",
        "env_key": "OPENAI_API_KEY",
        "base_url_env": ["OPENAI_CHAT_COMPLETIONS_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-3.1-pro-preview",
        "env_key": "GEMINI_API_KEY",
        "base_url_env": ["GEMINI_CHAT_COMPLETIONS_URL", "GEMINI_BASE_URL", "GEMINI_API_BASE"],
    },
}

SYSTEM_PROMPT = """你是严谨的上海高中化学视觉题质检员。你只能依据图片中可见内容和用户给出的题号/题干/选项作答。
不要猜。若图片中文字或图示看不清，必须写入 missing_or_uncertain。输出必须是 JSON，不要 Markdown。"""

USER_TEMPLATE = """请对这张试卷页面上的指定题目做质检和作答。指定题目结构如下（不含标准答案）：
{item_json}

任务：
1. 定位题目，确认图片中题号、题干、选项或图示是否支持这是同一道题。
2. 提取图中关键视觉元素、变量、数据、箭头、坐标、装置、物质或结构。
3. 判断是否可作答；如果图片或文字不清楚，必须降低 confidence 并说明不确定项。
4. 给出答案和简短解题链路；不要长篇散文。

严格输出 JSON，字段为：
{{
  "question_restatement": "...",
  "visual_elements": ["..."],
  "data_points": ["..."],
  "missing_or_uncertain": ["..."],
  "answerability": "answerable|uncertain|not_answerable",
  "solution_outline": "...",
  "final_answer": "..." 或 null,
  "confidence": 0到1,
  "evidence_citations": ["图片中可见锚点..."]
}}"""

TEXT_ASSISTED_USER_TEMPLATE = """请对指定化学题做质检和作答。你必须同时使用：
1. 结构化题干中的图像描述、选项、流程/曲线/装置/结构文字；
2. 输入图片中的可见证据。

指定题目结构如下（不含标准答案）：
{item_json}

任务：
1. 先依据结构化题干中的图像描述定位题目所需信息。
2. 再用图片互证题号、图表、装置、曲线或结构是否一致。
3. 如果结构化题干已经完整表达了图像信息，且图片没有明显矛盾，可以判定 answerable。
4. 给出答案和简短解题链路；不要长篇散文。

严格输出 JSON，字段为：
{{
  "question_restatement": "...",
  "visual_elements": ["..."],
  "data_points": ["..."],
  "missing_or_uncertain": ["..."],
  "answerability": "answerable|uncertain|not_answerable",
  "solution_outline": "...",
  "final_answer": "..." 或 null,
  "confidence": 0到1,
  "evidence_citations": ["结构化题干或图片中可见锚点..."]
}}"""

VISIBILITY_REVIEW_TEMPLATE = """请只做视觉证据二审，不要重新解题。指定题目结构如下：
{item_json}

上一轮模型答案（已由本地答案等价器和标准答案比对过）：{previous_answer}

任务：
1. 判断图片中是否能明确看到同一道题，或题干已完整给出且图片只承担辅助证据。
2. 判断作答所需的关键图像/表格/曲线/装置/结构信息是否可见。
3. 如果缺关键图像信息，必须写入 missing_key_information。
4. 只评估视觉证据是否足以支持上一轮答案，不要另给新答案。

严格输出 JSON，字段为：
{{
  "same_question_visible": true/false,
  "key_information_visible": true/false,
  "missing_key_information": ["..."],
  "answer_supported_by_visible_evidence": true/false,
  "evidence_citations": ["图片或题干中可见锚点..."],
  "confidence": 0到1
}}"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "电解": "通电",
        "去除": "除去",
        "除掉": "除去",
        "降低温度": "降温",
        "冷却": "降温",
        "酮羰基": "羰基",
        "氢氧化钠": "NaOH",
        "水溶液": "溶液",
        "或": "",
        "（": "(",
        "）": ")",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉⁺⁻＞＜＝−", "0123456789+-><=-"))
    text = re.sub(r"\\xrightarrow\{\\text\{[^{}]+\}\}", "=", text)
    text = re.sub(r"\\xrightarrow\{[^{}]+\}", "=", text)
    text = re.sub(r"\\(?:long)?rightarrow|\\to|→", "=", text)
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"[\s，。、“”‘’：；？！,.!?;:()（）\[\]【】{}《》_—\-]+", "", text)
    return text.upper()


def answer_alternatives(value: Any) -> list[str]:
    text = str(value or "")
    pieces = re.split(r"(?:或|/|；|;)", text)
    out = [normalize_answer(text)]
    out.extend(normalize_answer(piece) for piece in pieces)
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def equation_side_terms(value: str) -> tuple[set[str], set[str]] | None:
    normalized = normalize_answer(value)
    if "=" not in normalized:
        return None
    left, right = normalized.split("=", 1)
    if not left or not right:
        return None

    def split_terms(side: str) -> set[str]:
        terms = set()
        for term in side.split("+"):
            cleaned = re.sub(r"^[0-9]+", "", term)
            if cleaned:
                terms.add(cleaned)
        return terms

    return split_terms(left), split_terms(right)


def multi_blank_answer_match(predicted: Any, standard: Any) -> bool | None:
    pred_text = str(predicted or "")
    gold_text = str(standard or "")
    if not any(sep in gold_text for sep in ["；", ";"]):
        return None
    pred_parts = [part for part in re.split(r"[；;]+", pred_text) if normalize_answer(part)]
    gold_parts = [part for part in re.split(r"[；;]+", gold_text) if normalize_answer(part)]
    if len(gold_parts) <= 1:
        return None
    if len(pred_parts) != len(gold_parts):
        return False
    return all(answers_match(pred, gold) for pred, gold in zip(pred_parts, gold_parts))


def equations_equivalent(predicted: Any, standard: Any) -> bool:
    pred = equation_side_terms(str(predicted or ""))
    gold = equation_side_terms(str(standard or ""))
    if not pred or not gold:
        return False
    return pred == gold


def unordered_list_answer_match(predicted: Any, standard: Any) -> bool:
    pred_text = str(predicted or "")
    gold_text = str(standard or "")
    if any(symbol in pred_text + gold_text for symbol in [">", "<", "＞", "＜", "=", "＝", "→", "\\xrightarrow"]):
        return False
    if not any(sep in pred_text + gold_text for sep in ["、", "，", ",", "；", ";"]):
        return False
    splitter = r"[、，,；;]+"
    pred_parts = {normalize_answer(part) for part in re.split(splitter, pred_text) if normalize_answer(part)}
    gold_parts = {normalize_answer(part) for part in re.split(splitter, gold_text) if normalize_answer(part)}
    return bool(pred_parts and pred_parts == gold_parts)


def answers_match(predicted: Any, standard: Any) -> bool:
    multi_blank = multi_blank_answer_match(predicted, standard)
    if multi_blank is not None:
        return multi_blank
    pred_alts = answer_alternatives(predicted)
    gold_alts = answer_alternatives(standard)
    pred = pred_alts[0] if pred_alts else ""
    gold = gold_alts[0] if gold_alts else ""
    if not pred or not gold:
        return False
    if equations_equivalent(predicted, standard):
        return True
    if unordered_list_answer_match(predicted, standard):
        return True
    if re.fullmatch(r"[A-D]+", gold):
        predicted_options = set(re.findall(r"[A-D]", pred))
        gold_options = set(gold)
        if len(gold_options) > 1:
            return predicted_options == gold_options
        first = re.search(r"[A-D]", pred)
        return bool(first and first.group(0) in gold_options)
    if any(p == g or p in g or g in p for p in pred_alts for g in gold_alts):
        return True
    return gold in pred or pred in gold


def input_image_path_for_model(eval_item: dict[str, Any]) -> Path:
    crop_path = eval_item.get("crop_path")
    if crop_path and Path(crop_path).exists():
        return Path(crop_path)
    return Path(eval_item["page_image_path"])


def image_paths_for_model(eval_item: dict[str, Any]) -> list[Path]:
    paths = [input_image_path_for_model(eval_item)]
    seen = {str(paths[0])}
    for value in eval_item.get("additional_image_paths") or []:
        path = Path(str(value))
        if not path.exists():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def image_content_parts(eval_item: dict[str, Any], image_max_side: int) -> list[dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": image_data_url(path, max_side=image_max_side)}}
        for path in image_paths_for_model(eval_item)
    ]


def pilot_key(row: dict[str, Any]) -> tuple[str, int | None, str]:
    return (
        str(row.get("source_file", "")),
        int(row["page"]) if row.get("page") is not None else None,
        str(row.get("q_num") or row.get("item_id") or ""),
    )


def qid_from_pdf_item(item: dict[str, Any]) -> str:
    raw = f"{item.get('_source_file','')}|{item.get('q_num','')}|{item.get('stem','')[:40]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def load_pilot_results(path: Path) -> dict[tuple[str, int | None, str], dict[str, Any]]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, int | None, str], dict[str, Any]] = {}
    for row in obj.get("results", []):
        out[pilot_key(row)] = row
    return out


def load_pilot_results_by_item_id(path: Path, pdf_items_path: Path = DEFAULT_PDF_ITEMS) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    pdf_by_key = {
        (item.get("_source_file"), str(item.get("q_num")), item.get("_page")): item
        for item in load_jsonl(pdf_items_path)
    }
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in obj.get("results", []):
        item = pdf_by_key.get((row.get("source_file"), str(row.get("q_num")), row.get("page")))
        if item:
            out[qid_from_pdf_item(item)] = row
    return out


def result_from_pilot(eval_item: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    parsed = pilot.get("parsed") or {}
    answer = parsed.get("answer", pilot.get("answer", ""))
    standard = eval_item.get("standard_answer", pilot.get("gold_answer", ""))
    answer_match = answers_match(answer, standard)
    confidence = parsed.get("answer_confidence")
    try:
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        confidence_float = 0.0
    visible_pass = bool(pilot.get("visible_pass"))
    understanding_pass = bool(visible_pass and answer_match and confidence_float >= 0.8 and not parsed.get("uncertainty_reasons"))
    error_types: list[str] = []
    if pilot.get("error"):
        error_types.append("model_output_parse_error")
    if not visible_pass:
        error_types.append("question_not_visible")
    if not answer_match:
        error_types.append("answer_mismatch")
    if confidence_float >= 0.8 and not answer_match:
        error_types.append("high_confidence_error")
    return {
        "item_id": eval_item["item_id"],
        "category": eval_item.get("category"),
        "model": "qwen3-vl-plus",
        "input_image_path": str(input_image_path_for_model(eval_item)),
        "input_text": eval_item.get("stem"),
        "standard_answer": standard,
        "model_answer": answer,
        "confidence": confidence_float,
        "visible_pass": visible_pass,
        "answer_match": answer_match,
        "understanding_pass": understanding_pass,
        "profile_evidence_allowed": understanding_pass,
        "error_types": error_types,
        "evidence_citations": parsed.get("visible_anchors", []),
        "raw_source": "pilot_import",
    }


def result_from_model_payload(
    eval_item: dict[str, Any],
    payload: dict[str, Any],
    model: str,
    raw_source: str,
    usage: dict[str, Any] | None = None,
    latency_s: float | None = None,
) -> dict[str, Any]:
    standard = eval_item.get("standard_answer", "")
    final_answer = payload.get("final_answer")
    answer_match = answers_match(final_answer, standard)
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    missing = payload.get("missing_or_uncertain") or []
    citations = payload.get("evidence_citations") or []
    visual_elements = payload.get("visual_elements") or []
    answerability = str(payload.get("answerability") or "").lower()
    visible_pass = bool(len(citations) >= 2 and len(visual_elements) >= 1 and answerability != "not_answerable")
    understanding_pass = bool(
        visible_pass
        and answer_match
        and confidence >= 0.8
        and not missing
        and answerability == "answerable"
    )
    error_types: list[str] = []
    if not visible_pass:
        error_types.append("question_not_visible_or_weak_evidence")
    if missing:
        error_types.append("has_missing_or_uncertain")
    if answerability != "answerable":
        error_types.append("not_answerable")
    if not answer_match:
        error_types.append("answer_mismatch")
    if confidence >= 0.8 and not answer_match:
        error_types.append("high_confidence_error")

    return {
        "item_id": eval_item["item_id"],
        "category": eval_item.get("category"),
        "model": model,
        "input_image_path": str(input_image_path_for_model(eval_item)),
        "input_text": eval_item.get("stem"),
        "standard_answer": standard,
        "model_answer": final_answer,
        "confidence": confidence,
        "visible_pass": visible_pass,
        "answer_match": answer_match,
        "understanding_pass": understanding_pass,
        "profile_evidence_allowed": understanding_pass,
        "error_types": sorted(set(error_types)),
        "question_restatement": payload.get("question_restatement", ""),
        "visual_elements": visual_elements,
        "data_points": payload.get("data_points") or [],
        "missing_or_uncertain": missing,
        "solution_outline": payload.get("solution_outline", ""),
        "evidence_citations": citations,
        "usage": usage or {},
        "latency_s": latency_s,
        "raw_source": raw_source,
    }


def result_from_visibility_review_payload(
    eval_item: dict[str, Any],
    payload: dict[str, Any],
    model: str,
    raw_source: str,
    usage: dict[str, Any] | None = None,
    latency_s: float | None = None,
) -> dict[str, Any]:
    standard = eval_item.get("standard_answer", "")
    previous_answer = eval_item.get("previous_model_answer")
    answer_match = answers_match(previous_answer, standard)
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    missing = payload.get("missing_key_information") or []
    citations = payload.get("evidence_citations") or []
    visible_pass = bool(
        payload.get("same_question_visible")
        and payload.get("key_information_visible")
        and payload.get("answer_supported_by_visible_evidence")
        and len(citations) >= 2
    )
    understanding_pass = bool(visible_pass and answer_match and confidence >= 0.8 and not missing)
    error_types: list[str] = []
    if not answer_match:
        error_types.append("answer_mismatch")
    if not visible_pass:
        error_types.append("visibility_review_not_pass")
    if missing:
        error_types.append("has_missing_key_information")
    if confidence < 0.8:
        error_types.append("low_visibility_review_confidence")

    return {
        "item_id": eval_item["item_id"],
        "category": eval_item.get("category"),
        "model": model,
        "input_image_path": str(input_image_path_for_model(eval_item)),
        "input_text": eval_item.get("stem"),
        "standard_answer": standard,
        "model_answer": previous_answer,
        "confidence": confidence,
        "visible_pass": visible_pass,
        "answer_match": answer_match,
        "understanding_pass": understanding_pass,
        "profile_evidence_allowed": understanding_pass,
        "error_types": sorted(set(error_types)),
        "question_restatement": "",
        "visual_elements": [],
        "data_points": [],
        "missing_or_uncertain": missing,
        "solution_outline": "",
        "evidence_citations": citations,
        "usage": usage or {},
        "latency_s": latency_s,
        "raw_source": raw_source,
    }


def result_from_answer_review_payload(
    eval_item: dict[str, Any],
    payload: dict[str, Any],
    model: str,
    raw_source: str,
    usage: dict[str, Any] | None = None,
    latency_s: float | None = None,
) -> dict[str, Any]:
    standard = eval_item.get("standard_answer", "")
    model_answer = eval_item.get("model_answer")
    review_equivalent = str(payload.get("decision") or "").lower() == "equivalent"
    local_match = answers_match(model_answer, standard)
    equivalent = bool(review_equivalent or local_match)
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    visible_pass = bool(eval_item.get("visible_pass", True))
    answer_match = bool(equivalent)
    understanding_pass = bool(visible_pass and answer_match and confidence >= 0.8)
    error_types: list[str] = []
    if not answer_match:
        error_types.append("answer_mismatch")
    if confidence < 0.8:
        error_types.append("low_answer_review_confidence")

    return {
        "item_id": eval_item["item_id"],
        "category": eval_item.get("category"),
        "model": model,
        "input_image_path": eval_item.get("input_image_path") or eval_item.get("page_image_path"),
        "input_text": eval_item.get("stem"),
        "standard_answer": standard,
        "model_answer": model_answer,
        "confidence": confidence,
        "visible_pass": visible_pass,
        "answer_match": answer_match,
        "understanding_pass": understanding_pass,
        "profile_evidence_allowed": understanding_pass,
        "error_types": sorted(set(error_types)),
        "question_restatement": "",
        "visual_elements": [],
        "data_points": [],
        "missing_or_uncertain": [],
        "solution_outline": payload.get("reason", ""),
        "evidence_citations": [],
        "usage": usage or {},
        "latency_s": latency_s,
        "raw_source": raw_source,
        "semantic_review": True,
    }


def transcript_supported_strong(row: dict[str, Any]) -> bool:
    try:
        confidence = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.9:
        return False
    if not row.get("visible_pass"):
        return False
    if not answers_match(row.get("model_answer"), row.get("standard_answer")):
        return False
    citations = [str(item) for item in row.get("evidence_citations") or []]
    missing = [str(item) for item in row.get("missing_or_uncertain") or []]
    data_points = [str(item) for item in row.get("data_points") or []]
    visual_elements = [str(item) for item in row.get("visual_elements") or []]
    input_text = str(row.get("input_text") or row.get("stem") or "")
    solution_outline = str(row.get("solution_outline") or "")
    combined = "\n".join(citations + missing + data_points + visual_elements + [input_text, solution_outline])
    has_structured_transcript = "结构化题干" in combined and (
        "图像描述" in combined
        or "图像文字描述" in combined
        or "图示描述" in combined
        or "图表的描述" in combined
        or "结构化题干描述" in combined
    )
    has_explicit_diagram_transcript = (
        any(
            marker in input_text
            for marker in ["（图示：", "(图示:", "图示：", "图示内容：", "（图示内容：", "(图示内容:", "（图：", "(图:", "图："]
        )
        and any(marker in input_text for marker in ["横轴", "纵轴", "坐标系", "曲线", "箭头", "标注", "装置", "流程", "结构"])
    )
    says_complete = "完整" in combined or "可完整作答" in combined or "已完整提供" in combined
    no_direct_contradiction = not any(
        marker in combined
        for marker in ["图文矛盾", "图片矛盾", "与题干矛盾", "答案矛盾", "证据矛盾", "标准答案错误", "模型答案错误", "高置信错误"]
    )
    missing_only_source_image = any(
        marker in combined
        for marker in ["作答依赖于结构化题干", "题干提供的图像文字描述", "结构化题干给出的图像描述"]
    )
    structured_transcript_complete = bool(
        has_structured_transcript
        and (says_complete or has_explicit_diagram_transcript or missing_only_source_image)
        and no_direct_contradiction
    )

    relies_on_general_knowledge = any(marker in solution_outline for marker in ["基础知识", "排除法", "尽管题目缺失"])
    missing_required_visual = any(
        marker in combined
        for marker in ["原卷存在漏图", "图表缺失", "缺失题干提到的晶胞图", "缺失完整的题干和晶胞结构图"]
    )
    visible_data_complete = bool(
        "可完整作答" in combined
        and not relies_on_general_knowledge
        and not missing_required_visual
        and sum(1 for item in citations + data_points if re.search(r"[0-9ΔH]|反应|方程", item)) >= 2
    )
    visual_detail_terms = (
        "横轴",
        "纵轴",
        "坐标",
        "曲线",
        "箭头",
        "标注",
        "装置",
        "流程",
        "结构",
        "数据",
        "物质",
        "反应",
    )
    explicit_stem_transcript_complete = bool(
        has_explicit_diagram_transcript
        and (says_complete or sum(1 for item in [input_text] + visual_elements + data_points if any(term in item for term in visual_detail_terms)) >= 3)
        and not relies_on_general_knowledge
        and not missing_required_visual
        and no_direct_contradiction
    )

    return bool(structured_transcript_complete or visible_data_complete or explicit_stem_transcript_complete)


def placeholder_result(eval_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": eval_item["item_id"],
        "category": eval_item.get("category"),
        "model": "not_called",
        "input_image_path": eval_item.get("page_image_path"),
        "input_text": eval_item.get("stem"),
        "standard_answer": eval_item.get("standard_answer", ""),
        "model_answer": "",
        "confidence": 0.0,
        "visible_pass": False,
        "answer_match": False,
        "understanding_pass": False,
        "profile_evidence_allowed": False,
        "error_types": ["not_evaluated_offline"],
        "evidence_citations": [],
        "raw_source": "offline_placeholder",
    }


def load_env_values(path: Path = DEFAULT_ENV) -> dict[str, str]:
    values: dict[str, str] = {}
    allowed_env_keys = {
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_CHAT_COMPLETIONS_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_COMPLETIONS_URL",
        "GEMINI_API_KEY",
        "GEMINI_API_BASE",
        "GEMINI_BASE_URL",
        "GEMINI_CHAT_COMPLETIONS_URL",
    }
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in allowed_env_keys:
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key in allowed_env_keys})
    return values


def chat_completions_url(config: dict[str, Any], env_values: dict[str, str]) -> str:
    base_url = ""
    for env_key in config.get("base_url_env", []):
        if env_values.get(env_key):
            base_url = env_values[env_key].strip()
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


def image_data_url(image_path: Path, max_side: int = 1200) -> str:
    path = Path(image_path)
    data = path.read_bytes()
    ext = path.suffix.lower().lstrip(".") or "jpg"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.thumbnail((max_side, max_side))
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=82)
            data = buffer.getvalue()
            mime = "jpeg"
    except Exception:
        pass
    return f"data:image/{mime};base64," + base64.b64encode(data).decode("ascii")


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        raise ValueError("model_response_missing_json")
    return json.loads(match.group(0))


def strip_eval_item_for_model(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item.get("item_id"),
        "category": item.get("category"),
        "question_type": item.get("question_type"),
        "difficulty": item.get("difficulty"),
        "stem": item.get("stem"),
        "options": item.get("options") or {},
        "source_file": item.get("source_file"),
        "page": item.get("page"),
    }


def call_openai_compatible(
    eval_item: dict[str, Any],
    provider: str,
    api_key: str,
    model: str | None = None,
    timeout_s: int = 45,
    image_max_side: int = 1200,
    base_url: str | None = None,
    review_mode: str = "answer",
) -> tuple[dict[str, Any], dict[str, Any], float, str]:
    config = PROVIDERS[provider]
    model_name = model or config["model"]
    template = TEXT_ASSISTED_USER_TEMPLATE if review_mode == "text_assisted_answer" else USER_TEMPLATE
    prompt = template.format(
        item_json=json.dumps(strip_eval_item_for_model(eval_item), ensure_ascii=False)
    )
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 1400,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_content_parts(eval_item, image_max_side),
                ],
            },
        ],
    }
    request = urllib.request.Request(
        base_url or str(config["base_url"]),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    latency = round(time.time() - started, 2)
    content = response_payload["choices"][0]["message"]["content"] or ""
    parsed = extract_json_object(content)
    return parsed, response_payload.get("usage", {}), latency, model_name


def call_visibility_review(
    eval_item: dict[str, Any],
    provider: str,
    api_key: str,
    model: str | None = None,
    timeout_s: int = 45,
    image_max_side: int = 1200,
    base_url: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float, str]:
    config = PROVIDERS[provider]
    model_name = model or config["model"]
    prompt = VISIBILITY_REVIEW_TEMPLATE.format(
        item_json=json.dumps(strip_eval_item_for_model(eval_item), ensure_ascii=False),
        previous_answer=json.dumps(eval_item.get("previous_model_answer"), ensure_ascii=False),
    )
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_content_parts(eval_item, image_max_side),
                ],
            },
        ],
    }
    request = urllib.request.Request(
        base_url or str(config["base_url"]),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    latency = round(time.time() - started, 2)
    content = response_payload["choices"][0]["message"]["content"] or ""
    parsed = extract_json_object(content)
    return parsed, response_payload.get("usage", {}), latency, model_name


def evaluate_offline(
    eval_set_path: Path = DEFAULT_EVAL_SET,
    pilot_path: Path = DEFAULT_PILOT,
    pdf_items_path: Path = DEFAULT_PDF_ITEMS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_items = load_jsonl(Path(eval_set_path))
    pilot_by_key = load_pilot_results(Path(pilot_path))
    pilot_by_item_id = load_pilot_results_by_item_id(Path(pilot_path), Path(pdf_items_path))
    results: list[dict[str, Any]] = []
    imported = 0
    for item in eval_items:
        if item.get("item_id") in pilot_by_item_id:
            results.append(result_from_pilot(item, pilot_by_item_id[item["item_id"]]))
            imported += 1
            continue
        # Pilot rows do not know item_id, so match by source/page when possible.
        candidates = [
            row
            for key, row in pilot_by_key.items()
            if key[0] == item.get("source_file") and key[1] == item.get("page")
        ]
        if candidates:
            results.append(result_from_pilot(item, candidates[0]))
            imported += 1
        else:
            results.append(placeholder_result(item))

    error_counter = Counter(error for row in results for error in row.get("error_types", []))
    summary = {
        "eval_items": len(eval_items),
        "model_called": False,
        "pilot_imported": imported,
        "not_evaluated": sum(1 for row in results if "not_evaluated_offline" in row["error_types"]),
        "visible_pass": sum(1 for row in results if row["visible_pass"]),
        "answer_match": sum(1 for row in results if row["answer_match"]),
        "understanding_pass": sum(1 for row in results if row["understanding_pass"]),
        "profile_evidence_allowed": sum(1 for row in results if row["profile_evidence_allowed"]),
        "high_confidence_errors": error_counter.get("high_confidence_error", 0),
        "error_types": dict(error_counter),
        "category_counts": dict(Counter(row.get("category") for row in results)),
    }
    return results, summary


def evaluate_with_model(
    eval_set_path: Path = DEFAULT_EVAL_SET,
    provider: str = "qwen-vl",
    model: str | None = None,
    limit: int = 6,
    env_path: Path = DEFAULT_ENV,
    timeout_s: int = 45,
    image_max_side: int = 1200,
    progress_out: Path | None = None,
    progress: bool = False,
    review_mode: str = "answer",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_items = load_jsonl(Path(eval_set_path))[:limit]
    config = PROVIDERS[provider]
    env_values = load_env_values(env_path)
    api_key = env_values.get(config["env_key"], "")
    base_url = chat_completions_url(config, env_values)
    if not api_key:
        results = [placeholder_result(item) for item in eval_items]
        for row in results:
            row["error_types"] = ["missing_api_key"]
            row["raw_source"] = "model_not_called"
        summary = summarize_results(results, model_called=False, provider=provider, model=model or config["model"])
        return results, summary

    results: list[dict[str, Any]] = []
    if progress_out:
        progress_out.parent.mkdir(parents=True, exist_ok=True)
        progress_out.write_text("", encoding="utf-8")
    for index, item in enumerate(eval_items, start=1):
        if progress:
            print(f"[{index}/{len(eval_items)}] evaluating {item.get('item_id')} {item.get('category')}", flush=True)
        try:
            if review_mode == "visibility":
                payload, usage, latency, model_name = call_visibility_review(
                    item,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    timeout_s=timeout_s,
                    image_max_side=image_max_side,
                    base_url=base_url,
                )
                results.append(
                    result_from_visibility_review_payload(
                        item,
                        payload,
                        model=model_name,
                        raw_source="visibility_review_model_call",
                        usage=usage,
                        latency_s=latency,
                    )
                )
            else:
                payload, usage, latency, model_name = call_openai_compatible(
                    item,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    timeout_s=timeout_s,
                    image_max_side=image_max_side,
                    base_url=base_url,
                    review_mode=review_mode,
                )
                results.append(
                    result_from_model_payload(
                        item,
                        payload,
                        model=model_name,
                        raw_source="model_call",
                        usage=usage,
                        latency_s=latency,
                    )
                )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
            row = placeholder_result(item)
            row["model"] = model or config["model"]
            row["raw_source"] = "model_error"
            row["error_types"] = [type(exc).__name__]
            row["model_error"] = str(exc)[:240]
            results.append(row)
        if progress_out:
            with progress_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")
        time.sleep(0.5)
    return results, summarize_results(results, model_called=True, provider=provider, model=model or config["model"])


def summarize_results(
    results: list[dict[str, Any]],
    model_called: bool,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    error_counter = Counter(error for row in results for error in row.get("error_types", []))
    return {
        "eval_items": len(results),
        "model_called": model_called,
        "provider": provider or "",
        "model": model or "",
        "not_evaluated": sum(1 for row in results if "not_evaluated_offline" in row.get("error_types", [])),
        "visible_pass": sum(1 for row in results if row["visible_pass"]),
        "answer_match": sum(1 for row in results if row["answer_match"]),
        "understanding_pass": sum(1 for row in results if row["understanding_pass"]),
        "profile_evidence_allowed": sum(1 for row in results if row["profile_evidence_allowed"]),
        "high_confidence_errors": error_counter.get("high_confidence_error", 0),
        "error_types": dict(error_counter),
        "category_counts": dict(Counter(row.get("category") for row in results)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate visual understanding results.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--pdf-items", type=Path, default=DEFAULT_PDF_ITEMS)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--mode", choices=["offline", "model"], default="offline")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="qwen-vl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--image-max-side", type=int, default=1200)
    parser.add_argument("--progress-out", type=Path, default=None, help="Append each model result as it completes.")
    parser.add_argument("--progress", action="store_true", help="Print one progress line per evaluated item.")
    parser.add_argument("--review-mode", choices=["answer", "visibility", "text_assisted_answer"], default="answer")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.mode == "model":
        results, summary = evaluate_with_model(
            eval_set_path=args.eval_set,
            provider=args.provider,
            model=args.model,
            limit=args.limit,
            timeout_s=args.timeout_s,
            image_max_side=args.image_max_side,
            progress_out=args.progress_out,
            progress=args.progress,
            review_mode=args.review_mode,
        )
    else:
        results, summary = evaluate_offline(
            eval_set_path=args.eval_set,
            pilot_path=args.pilot,
            pdf_items_path=args.pdf_items,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        write_jsonl(args.out, results)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out}")
        print(f"WROTE {args.summary_out}")
    else:
        print("DRY RUN: pass --write to write data/evals outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
