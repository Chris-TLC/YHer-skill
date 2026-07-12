#!/usr/bin/env python3
"""Tests for visual understanding evaluation result normalization."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

OPENAI_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_CHAT_COMPLETIONS_URL",
)
GEMINI_ENV_KEYS = (
    "GEMINI_API_KEY",
    "GEMINI_API_BASE",
    "GEMINI_BASE_URL",
    "GEMINI_CHAT_COMPLETIONS_URL",
)


@contextmanager
def without_env(*keys: str):
    saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def test_parse_model_payload_records_high_confidence_error():
    from scripts.evaluate_visual_understanding import result_from_model_payload

    eval_item = {
        "item_id": "i1",
        "category": "experiment_device",
        "page_image_path": "/tmp/page.jpg",
        "stem": "如图所示实验装置题",
        "standard_answer": "B",
    }
    payload = {
        "question_restatement": "装置选择题",
        "visual_elements": ["冷凝管", "温度计"],
        "data_points": [],
        "missing_or_uncertain": [],
        "answerability": "answerable",
        "solution_outline": "依据装置判断",
        "final_answer": "C",
        "confidence": 0.92,
        "evidence_citations": ["题号9", "选项C"],
    }

    row = result_from_model_payload(eval_item, payload, model="unit-model", raw_source="unit")

    assert row["model"] == "unit-model"
    assert row["visible_pass"]
    assert not row["answer_match"]
    assert not row["understanding_pass"]
    assert not row["profile_evidence_allowed"]
    assert "high_confidence_error" in row["error_types"]


def test_parse_model_payload_marks_strong_understanding_when_answered_with_evidence():
    from scripts.evaluate_visual_understanding import result_from_model_payload

    eval_item = {
        "item_id": "i2",
        "category": "organic_structure",
        "page_image_path": "/tmp/page.jpg",
        "stem": "如图所示有机结构题",
        "standard_answer": "A",
    }
    payload = {
        "question_restatement": "结构判断题",
        "visual_elements": ["苯环", "羟基"],
        "data_points": ["选项A"],
        "missing_or_uncertain": [],
        "answerability": "answerable",
        "solution_outline": "图中官能团支持A",
        "final_answer": "A",
        "confidence": 0.86,
        "evidence_citations": ["题号3", "苯环", "选项A"],
    }

    row = result_from_model_payload(eval_item, payload, model="unit-model", raw_source="unit")

    assert row["answer_match"]
    assert row["visible_pass"]
    assert row["understanding_pass"]
    assert row["profile_evidence_allowed"]
    assert row["error_types"] == []


def test_imported_pilot_understanding_pass_allows_profile_evidence():
    from scripts.evaluate_visual_understanding import result_from_pilot

    eval_item = {
        "item_id": "i3",
        "category": "chart_curve",
        "page_image_path": "/tmp/page.jpg",
        "stem": "如图所示曲线题",
        "standard_answer": "A",
    }
    pilot = {
        "parsed": {
            "question_visible": True,
            "match_confidence": 1.0,
            "visible_anchors": ["题号32", "曲线c1", "曲线c2"],
            "answer": "A",
            "answer_confidence": 0.95,
            "uncertainty_reasons": [],
        },
        "visible_pass": True,
    }

    row = result_from_pilot(eval_item, pilot)

    assert row["understanding_pass"]
    assert row["profile_evidence_allowed"]


def test_multi_choice_answer_match_requires_all_gold_options():
    from scripts.evaluate_visual_understanding import answers_match

    assert answers_match("AC", "AC")
    assert answers_match("A、C", "AC")
    assert not answers_match("A", "AC")
    assert not answers_match("ABC", "AC")


def test_answer_match_handles_symbol_format_equivalence_conservatively():
    from scripts.evaluate_visual_understanding import answers_match

    assert answers_match("C > N > O", "C＞N＞O")
    assert answers_match("p₁ < p₂", "p₁＜p₂")
    assert answers_match("烧杯、量筒", "量筒、烧杯")
    assert not answers_match("C > O > N", "C＞N＞O")


def test_answer_match_handles_common_chemistry_wording_equivalence():
    from scripts.evaluate_visual_understanding import answers_match

    assert answers_match("冷却结晶（或降温结晶）", "冷却(降温)结晶")
    assert answers_match("去除废铁屑表面的油污", "除去废铁屑表面的油污")
    assert answers_match(
        "2Cl⁻ + 2H₂O \\xrightarrow{\\text{电解}} 2OH⁻ + H₂↑ + Cl₂↑",
        "2Cl⁻ + 2H₂O \\xrightarrow{\\text{通电}} Cl₂↑ + H₂↑ + 2OH⁻",
    )
    assert answers_match("K = [CO]", "K=c(CO)或K=[CO]")
    assert answers_match("羟基、羰基（或酮羰基）", "羟基、酮羰基")
    assert answers_match("氢氧化钠水溶液、加热（或稀硫酸、加热）", "NaOH溶液、加热")
    assert answers_match("2NaI + Cl_{2} \\rightarrow 2NaCl + I_{2}", "Cl₂ + 2NaI = 2NaCl + I₂")


def test_answer_match_does_not_accept_partial_multi_blank_answer():
    from scripts.evaluate_visual_understanding import answers_match

    assert not answers_match("1s²2s²2p⁶3s²3p⁴；5；Na", "1s²2s²2p⁶3s²3p⁴；6；Na")


def test_model_call_prefers_crop_path_when_present(tmp_path: Path):
    from scripts import evaluate_visual_understanding as evu

    used_paths: list[Path] = []

    def fake_image_data_url(path: Path, max_side: int = 1200) -> str:
        used_paths.append(path)
        return "data:image/jpeg;base64,unit"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "question_restatement": "题目",
                                        "visual_elements": ["图"],
                                        "data_points": [],
                                        "missing_or_uncertain": [],
                                        "answerability": "answerable",
                                        "solution_outline": "依据图",
                                        "final_answer": "A",
                                        "confidence": 1,
                                        "evidence_citations": ["题号", "选项A"],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    crop = tmp_path / "crop.png"
    page = tmp_path / "page.png"
    crop.write_bytes(b"crop")
    page.write_bytes(b"page")

    original_image_data_url = evu.image_data_url
    original_urlopen = evu.urllib.request.urlopen
    evu.image_data_url = fake_image_data_url
    evu.urllib.request.urlopen = lambda request, timeout: FakeResponse()
    try:
        evu.call_openai_compatible(
            {
                "item_id": "i_crop",
                "crop_path": str(crop),
                "page_image_path": str(page),
                "stem": "如图",
                "standard_answer": "A",
            },
            provider="gemini",
            api_key="unit-key",
            base_url="https://example.com/v1/chat/completions",
        )
    finally:
        evu.image_data_url = original_image_data_url
        evu.urllib.request.urlopen = original_urlopen

    assert used_paths == [crop]


def test_model_call_sends_additional_image_paths_after_primary_crop(tmp_path: Path):
    from scripts import evaluate_visual_understanding as evu

    used_paths: list[Path] = []
    captured_body: dict[str, object] = {}

    def fake_image_data_url(path: Path, max_side: int = 1200) -> str:
        used_paths.append(path)
        return f"data:image/jpeg;base64,{path.stem}"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "question_restatement": "跨页曲线题",
                                        "visual_elements": ["题干", "下一页曲线"],
                                        "data_points": ["曲线读数"],
                                        "missing_or_uncertain": [],
                                        "answerability": "answerable",
                                        "solution_outline": "依据两张图片",
                                        "final_answer": "A",
                                        "confidence": 1,
                                        "evidence_citations": ["主裁片题干", "下一页图表"],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured_body.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    crop = tmp_path / "crop.png"
    page = tmp_path / "page.png"
    next_page = tmp_path / "next_page.png"
    crop.write_bytes(b"crop")
    page.write_bytes(b"page")
    next_page.write_bytes(b"next")

    original_image_data_url = evu.image_data_url
    original_urlopen = evu.urllib.request.urlopen
    evu.image_data_url = fake_image_data_url
    evu.urllib.request.urlopen = fake_urlopen
    try:
        evu.call_openai_compatible(
            {
                "item_id": "i_cross_page",
                "crop_path": str(crop),
                "page_image_path": str(page),
                "additional_image_paths": [str(next_page), str(next_page)],
                "stem": "如下图所示，图表在下一页",
                "standard_answer": "A",
            },
            provider="gemini",
            api_key="unit-key",
            base_url="https://example.com/v1/chat/completions",
        )
    finally:
        evu.image_data_url = original_image_data_url
        evu.urllib.request.urlopen = original_urlopen

    content = captured_body["messages"][1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert used_paths == [crop, next_page]
    assert len(image_parts) == 2


def test_visibility_review_sends_additional_image_paths_after_primary_crop(tmp_path: Path):
    from scripts import evaluate_visual_understanding as evu

    used_paths: list[Path] = []

    def fake_image_data_url(path: Path, max_side: int = 1200) -> str:
        used_paths.append(path)
        return f"data:image/jpeg;base64,{path.stem}"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "same_question_visible": True,
                                        "key_information_visible": True,
                                        "missing_key_information": [],
                                        "answer_supported_by_visible_evidence": True,
                                        "evidence_citations": ["主裁片", "下一页图表"],
                                        "confidence": 0.95,
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    crop = tmp_path / "crop.png"
    next_page = tmp_path / "next_page.png"
    crop.write_bytes(b"crop")
    next_page.write_bytes(b"next")

    original_image_data_url = evu.image_data_url
    original_urlopen = evu.urllib.request.urlopen
    evu.image_data_url = fake_image_data_url
    evu.urllib.request.urlopen = lambda request, timeout: FakeResponse()
    try:
        evu.call_visibility_review(
            {
                "item_id": "i_cross_page_visibility",
                "crop_path": str(crop),
                "page_image_path": str(crop),
                "additional_image_paths": [str(next_page)],
                "stem": "如下图所示，图表在下一页",
                "standard_answer": "A",
                "previous_model_answer": "A",
            },
            provider="gemini",
            api_key="unit-key",
            base_url="https://example.com/v1/chat/completions",
        )
    finally:
        evu.image_data_url = original_image_data_url
        evu.urllib.request.urlopen = original_urlopen

    assert used_paths == [crop, next_page]


def test_model_payload_records_effective_crop_input_path(tmp_path: Path):
    from scripts.evaluate_visual_understanding import result_from_model_payload

    crop = tmp_path / "crop.png"
    page = tmp_path / "page.png"
    crop.write_bytes(b"crop")
    page.write_bytes(b"page")

    row = result_from_model_payload(
        {
            "item_id": "i_crop_record",
            "category": "chart_curve",
            "crop_path": str(crop),
            "page_image_path": str(page),
            "stem": "如图所示曲线题",
            "standard_answer": "A",
        },
        {
            "question_restatement": "曲线判断题",
            "visual_elements": ["曲线"],
            "data_points": ["选项A"],
            "missing_or_uncertain": [],
            "answerability": "answerable",
            "solution_outline": "依据裁片中的曲线判断",
            "final_answer": "A",
            "confidence": 0.9,
            "evidence_citations": ["裁片题号", "裁片选项A"],
        },
        model="unit-model",
        raw_source="unit",
    )

    assert row["input_image_path"] == str(crop)


def test_visibility_review_payload_can_upgrade_answer_verified_item():
    from scripts.evaluate_visual_understanding import result_from_visibility_review_payload

    eval_item = {
        "item_id": "i_review",
        "category": "chart_curve",
        "page_image_path": "/tmp/page.jpg",
        "stem": "如图所示曲线题",
        "standard_answer": "D",
        "previous_model_answer": "D",
        "previous_confidence": 0.95,
    }
    payload = {
        "same_question_visible": True,
        "key_information_visible": True,
        "missing_key_information": [],
        "evidence_citations": ["题号", "曲线", "选项D"],
        "answer_supported_by_visible_evidence": True,
        "confidence": 0.9,
    }

    row = result_from_visibility_review_payload(eval_item, payload, model="unit-model", raw_source="unit")

    assert row["answer_match"]
    assert row["visible_pass"]
    assert row["understanding_pass"]
    assert row["profile_evidence_allowed"]
    assert row["error_types"] == []


def test_transcript_supported_strong_requires_answer_visible_and_complete_transcript():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "D",
        "model_answer": "D",
        "confidence": 0.95,
        "visible_pass": True,
        "missing_or_uncertain": ["图片中未显示原图，但结构化题干中已完整提供了图像信息的文字描述及选项内容。"],
        "evidence_citations": ["结构化题干中的图示描述", "图片可见题号"],
    }

    assert transcript_supported_strong(row)


def test_transcript_supported_strong_rejects_answer_mismatch():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "B",
        "model_answer": "C",
        "confidence": 1.0,
        "visible_pass": True,
        "missing_or_uncertain": ["结构化题干中已完整提供图像描述"],
        "evidence_citations": ["结构化题干"],
    }

    assert not transcript_supported_strong(row)


def test_transcript_supported_strong_accepts_structured_graph_transcript_when_page_image_is_cropped():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "p₁＜p₂",
        "model_answer": "p₁<p₂",
        "confidence": 0.95,
        "visible_pass": True,
        "input_text": "（图示：坐标系横轴为“温度”，纵轴为“CH₄的平衡转化率”；两条上升曲线，上方曲线标注“p₁”，下方曲线标注“p₂”）",
        "missing_or_uncertain": ["图片在题目21第(2)问处被截断，缺失第(3)小题及题干描述的p₁、p₂曲线图示"],
        "evidence_citations": [
            "结构化题干图像描述：纵轴为'CH₄的平衡转化率'，上方曲线标注'p₁'，下方曲线标注'p₂'",
            "图片中题目21的反应①方程式",
        ],
        "solution_outline": "根据结构化题干给出的图像描述，在相同温度下，p₁对应的CH₄平衡转化率高于p₂。",
    }

    assert transcript_supported_strong(row)


def test_transcript_supported_strong_accepts_complete_explicit_diagram_transcript_in_stem():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "p₁＜p₂",
        "model_answer": "p₁ < p₂",
        "confidence": 1.0,
        "visible_pass": True,
        "input_text": "（图示：坐标系横轴为“温度”，纵轴为“CH₄的平衡转化率”；两条上升曲线，上方曲线标注“p₁”，下方曲线标注“p₂”。）",
        "missing_or_uncertain": [],
        "visual_elements": [
            "坐标系横轴标注为温度",
            "坐标系纵轴标注为CH₄的平衡转化率",
            "上方曲线标注p₁，下方曲线标注p₂",
        ],
        "data_points": [
            "同一温度下，p₁对应的CH₄平衡转化率大于p₂对应的CH₄平衡转化率"
        ],
        "evidence_citations": [
            "题干图示完整描述了p₁、p₂两条曲线的位置关系",
            "反应①气体分子数由4增至9，压强越小越有利于转化率增大",
        ],
        "solution_outline": "根据题干图示，p₁曲线在p₂上方；反应气体分子数增大，低压更有利，故p₁<p₂。",
    }

    assert transcript_supported_strong(row)


def test_transcript_supported_strong_accepts_explicit_tushineirong_marker():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "p₁＜p₂",
        "model_answer": "p₁ < p₂",
        "confidence": 1.0,
        "visible_pass": True,
        "input_text": "（图示内容：坐标系横轴为“温度”，纵轴为“CH₄的平衡转化率”；两条上升曲线，上方曲线标注为p₁，下方曲线标注为p₂，p₁始终高于p₂）",
        "visual_elements": [
            "横轴为温度",
            "纵轴为CH₄的平衡转化率",
            "上方曲线标注p₁，下方曲线标注p₂",
        ],
        "data_points": ["同一温度下p₁曲线高于p₂曲线"],
        "evidence_citations": ["题干图示内容完整描述了p₁、p₂曲线位置"],
        "solution_outline": "正反应气体分子数增大，相同温度下p₁曲线更高，说明p₁较小。",
    }

    assert transcript_supported_strong(row)


def test_transcript_supported_strong_does_not_treat_option_error_wording_as_contradiction():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "AC",
        "model_answer": "AC",
        "confidence": 1.0,
        "visible_pass": True,
        "input_text": "（图：坐标系，横轴为t/min，纵轴为c(AsO₄³⁻)，曲线上升后趋近水平虚线y；曲线上标有点m和点n。）A. pH不再变化 B. v(I⁻)=2v(AsO₃³⁻) C. 浓度比不再变化 D. c(I⁻)=y",
        "visual_elements": [
            "横轴为t/min",
            "纵轴为c(AsO₄³⁻)",
            "曲线最终趋近水平虚线y",
        ],
        "data_points": ["平衡时c(AsO₄³⁻)=y"],
        "evidence_citations": ["题干图示完整描述了曲线和水平虚线y"],
        "solution_outline": "A正确；B错误，因为只体现计量关系；C正确；D错误，因为I⁻为2y。",
    }

    assert transcript_supported_strong(row)


def test_transcript_supported_strong_accepts_structured_catalyst_curve_transcript():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "Cu/Al₂O₃",
        "model_answer": "Cu/Al₂O₃效果较好。理由：在相同的CO物质的量分数下，使用 Cu/Al₂O₃ 作催化剂时的反应速率始终大于使用 Cu/ZnO-cp 作催化剂时的反应速率。",
        "confidence": 0.95,
        "visible_pass": True,
        "input_text": "（图：坐标系，纵轴为“反应速率”，横轴为“CO物质的量分数”；图中两条曲线：上方为平缓下降后趋于水平的曲线，标注“Cu/Al₂O₃”；下方为先缓慢上升后急剧下降的曲线，标注“Cu/ZnO-cp”）",
        "missing_or_uncertain": ["图片中实际缺失了题目所述的曲线图，作答依赖于结构化题干提供的图像文字描述。"],
        "evidence_citations": [
            "结构化题干描述：纵轴为“反应速率”，横轴为“CO物质的量分数”",
            "结构化题干描述：上方曲线标注“Cu/Al₂O₃”",
        ],
        "solution_outline": "根据结构化题干中对图表的描述，Cu/Al₂O₃ 曲线始终位于 Cu/ZnO-cp 上方。",
    }

    assert transcript_supported_strong(row)


def test_transcript_supported_strong_accepts_complete_visible_data_when_subquestion_label_is_missing():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "−90 kJ·mol⁻¹",
        "model_answer": "-90 kJ·mol⁻¹",
        "confidence": 0.95,
        "visible_pass": True,
        "missing_or_uncertain": ["图片中未直接显示标号为（3）的题目，但根据图中提供的反应①、反应②及总反应的数据可完整作答该问。"],
        "evidence_citations": [
            "反应①：CO₂(g)+H₂(g)⇌CO(g)+H₂O(g) ΔH₁ = +41kJ·mol⁻¹",
            "已知总反应：CO₂(g)+3H₂(g)⇌CH₃OH(g)+H₂O(g) ΔH总 = -49kJ·mol⁻¹",
        ],
        "data_points": ["反应①的焓变 ΔH₁ = +41 kJ·mol⁻¹", "总反应的焓变 ΔH总 = -49 kJ·mol⁻¹"],
        "solution_outline": "根据盖斯定律，总反应 = 反应① + 反应②，因此 ΔH₂ = -49 - 41。",
    }

    assert transcript_supported_strong(row)


def test_transcript_supported_strong_rejects_common_knowledge_answer_when_required_graph_is_missing():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "D",
        "model_answer": "D",
        "confidence": 0.95,
        "visible_pass": True,
        "input_text": "其能量随反应进程的变化如下图所示。下列说法正确的是（　　）",
        "missing_or_uncertain": ["题干中明确写有‘如下图所示’，但图片中第8题的题干下方未提供‘能量随反应进程的变化’图表，原卷存在漏图现象。"],
        "evidence_citations": ["题干中的文本描述", "图片中选项A、B、C、D的文字表述"],
        "solution_outline": "尽管题目缺失了能量变化图，但可以通过化学基础知识进行排除法作答。",
    }

    assert not transcript_supported_strong(row)


def test_transcript_supported_strong_rejects_crystal_structure_name_without_diagram_transcript():
    from scripts.evaluate_visual_understanding import transcript_supported_strong

    row = {
        "standard_answer": "4",
        "model_answer": "4",
        "confidence": 1.0,
        "visible_pass": True,
        "input_text": "白锡和灰锡是单质Sn常见同素异形体。二者晶胞如图：白锡具有体心四方结构；灰锡具有立方金刚石结构。",
        "missing_or_uncertain": ["图片中缺失完整的题干和晶胞结构图，但结构化题干已提供解答所需的结构名称。"],
        "evidence_citations": ["结构化题干：灰锡具有立方金刚石结构"],
        "solution_outline": "根据结构化题干给出的信息，灰锡具有立方金刚石结构。",
    }

    assert not transcript_supported_strong(row)


def test_text_assisted_prompt_includes_diagram_transcript_context(tmp_path: Path):
    from scripts import evaluate_visual_understanding as evu

    captured_body: dict[str, object] = {}

    def fake_image_data_url(path: Path, max_side: int = 1200) -> str:
        return "data:image/jpeg;base64,unit"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "question_restatement": "题目",
                                        "visual_elements": ["图像描述中的曲线"],
                                        "data_points": [],
                                        "missing_or_uncertain": [],
                                        "answerability": "answerable",
                                        "solution_outline": "依据结构化图像描述",
                                        "final_answer": "A",
                                        "confidence": 1,
                                        "evidence_citations": ["结构化图像描述", "图片题号"],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured_body.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    page = tmp_path / "page.png"
    page.write_bytes(b"page")
    original_image_data_url = evu.image_data_url
    original_urlopen = evu.urllib.request.urlopen
    evu.image_data_url = fake_image_data_url
    evu.urllib.request.urlopen = fake_urlopen
    try:
        evu.call_openai_compatible(
            {
                "item_id": "i_text_assisted",
                "page_image_path": str(page),
                "stem": "如图所示曲线题（图示：上方曲线标注p1，下方曲线标注p2）",
                "standard_answer": "A",
            },
            provider="gemini",
            api_key="unit-key",
            base_url="https://example.com/v1/chat/completions",
            review_mode="text_assisted_answer",
        )
    finally:
        evu.image_data_url = original_image_data_url
        evu.urllib.request.urlopen = original_urlopen

    text_part = captured_body["messages"][1]["content"][0]["text"]
    assert "结构化题干中的图像描述" in text_part
    assert "图片互证" in text_part


def test_openai_provider_missing_key_returns_missing_api_key_without_calling_model(tmp_path: Path):
    import os
    from scripts.evaluate_visual_understanding import evaluate_with_model

    eval_set = tmp_path / "visual_item_eval_set.jsonl"
    eval_set.write_text(
        json.dumps(
            {
                "item_id": "i_missing_key",
                "category": "chart_curve",
                "page_image_path": "/tmp/nonexistent.jpg",
                "stem": "如图所示曲线题",
                "standard_answer": "A",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        results, summary = evaluate_with_model(
            eval_set_path=eval_set,
            provider="gpt-4o",
            env_path=tmp_path / ".env",
            limit=1,
        )
    finally:
        if old_key is not None:
            os.environ["OPENAI_API_KEY"] = old_key

    assert summary["model_called"] is False
    assert summary["provider"] == "gpt-4o"
    assert summary["error_types"] == {"missing_api_key": 1}
    assert results[0]["raw_source"] == "model_not_called"
    assert results[0]["error_types"] == ["missing_api_key"]


def test_openai_provider_reads_relay_base_url_from_env_file(tmp_path: Path):
    from scripts.evaluate_visual_understanding import PROVIDERS, chat_completions_url, load_env_values

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-relay\n"
        "OPENAI_BASE_URL=https://relay.example.com/v1\n",
        encoding="utf-8",
    )

    with without_env(*OPENAI_ENV_KEYS):
        values = load_env_values(env_file)

    assert values["OPENAI_API_KEY"] == "sk-relay"
    assert chat_completions_url(PROVIDERS["gpt-4o"], values) == "https://relay.example.com/v1/chat/completions"


def test_openai_provider_accepts_full_chat_completions_url(tmp_path: Path):
    from scripts.evaluate_visual_understanding import PROVIDERS, chat_completions_url, load_env_values

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_CHAT_COMPLETIONS_URL=https://relay.example.com/openai/v1/chat/completions\n",
        encoding="utf-8",
    )

    with without_env(*OPENAI_ENV_KEYS):
        values = load_env_values(env_file)

    assert chat_completions_url(PROVIDERS["gpt-4o"], values) == "https://relay.example.com/openai/v1/chat/completions"


def test_openai_provider_adds_v1_chat_completions_to_bare_base_url(tmp_path: Path):
    from scripts.evaluate_visual_understanding import PROVIDERS, chat_completions_url, load_env_values

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_BASE=https://relay.example.com\n",
        encoding="utf-8",
    )

    with without_env(*OPENAI_ENV_KEYS):
        values = load_env_values(env_file)

    assert chat_completions_url(PROVIDERS["gpt-4o"], values) == "https://relay.example.com/v1/chat/completions"


def test_gemini_provider_reads_relay_base_url_from_env_file(tmp_path: Path):
    from scripts.evaluate_visual_understanding import PROVIDERS, chat_completions_url, load_env_values

    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=gemini-relay\n"
        "GEMINI_BASE_URL=https://api.ooapi.cc\n",
        encoding="utf-8",
    )

    with without_env(*GEMINI_ENV_KEYS):
        values = load_env_values(env_file)

    assert values["GEMINI_API_KEY"] == "gemini-relay"
    assert chat_completions_url(PROVIDERS["gemini"], values) == "https://api.ooapi.cc/v1/chat/completions"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    import tempfile

    for test in tests:
        try:
            if "tmp_path" in test.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    test(Path(d))
            else:
                test()
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 测试通过")
    sys.exit(0 if passed == len(tests) else 1)
