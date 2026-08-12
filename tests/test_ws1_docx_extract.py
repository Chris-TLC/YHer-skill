#!/usr/bin/env python3
"""Tests for WS1 docx-native extraction helpers."""

from __future__ import annotations

import zipfile
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET


def text_para(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def test_para_blocks_preserves_omml_as_math_block():
    from scripts.ws1_docx_extract_prototype import para_blocks

    xml = """
    <w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
         xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
      <w:r><w:t>已知</w:t></w:r>
      <m:oMath><m:r><m:t>n</m:t></m:r><m:sup><m:e><m:r><m:t>2</m:t></m:r></m:e></m:sup></m:oMath>
      <w:r><w:t>的值</w:t></w:r>
    </w:p>
    """
    blocks = para_blocks(ET.fromstring(xml), {})

    assert [block["type"] for block in blocks] == ["text", "math_omml", "text"]
    assert blocks[0]["text"] == "已知"
    assert "<m:oMath" in blocks[1]["omml"]
    assert blocks[2]["text"] == "的值"


def test_segment_uses_section_scoped_compound_question_ids():
    from scripts.ws1_docx_extract_prototype import segment

    questions = segment(
        [
            text_para("一、选择题"),
            text_para("1. 第一题"),
            text_para("2. 第二题"),
            text_para("二、综合题"),
            text_para("1. 综合第一题"),
        ]
    )

    assert [q["q_num"] for q in questions] == [1, 2, 1]
    assert [q["section_num"] for q in questions] == [1, 1, 2]
    assert [q["question_id"] for q in questions] == ["1-1", "1-2", "2-1"]


def test_segment_ignores_numbered_exam_instructions_before_first_question():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("1.试卷满分100分，考试时间60分钟。"),
            text_para("2.答题前，考生务必将学校、姓名、准考证号填写清楚，并将条形码贴在规定位置。"),
            text_para("3.本考试分设试卷和答题纸，作答必须涂或写在答题纸上。"),
            text_para("一、综合题"),
            text_para("1. 大气污染物主要来自燃煤和机动车尾气，下列物质能吸收SO2的是"),
            text_para("【答案】C"),
        ]
    )

    assert len(questions) == 1
    assert questions[0]["question_id"] == "1-1"
    stem = blocks_text([b for para in questions[0]["stem_blocks"] for b in para["para"]])
    assert "试卷满分" not in stem
    assert "答题纸" not in stem
    assert "大气污染物" in stem


def test_segment_remaps_trailing_answer_block_by_question_number():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("一、选择题"),
            text_para("1. 第一题"),
            text_para("2. 第二题"),
            text_para("【答案】1. A 2. B"),
        ]
    )

    assert len(questions) == 2
    assert blocks_text([b for para in questions[0]["answer_blocks"] for b in para["para"]]) == "【答案】A"
    assert blocks_text([b for para in questions[1]["answer_blocks"] for b in para["para"]]) == "【答案】B"


def test_segment_recognizes_plain_answer_and_analysis_markers():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("1．第一题"),
            text_para("答案：A"),
            text_para("解析：这是解析"),
            text_para("2．第二题"),
            text_para("答案：B"),
        ]
    )

    assert len(questions) == 2
    assert blocks_text([b for para in questions[0]["answer_blocks"] for b in para["para"]]) == "答案：A"
    assert blocks_text([b for para in questions[0]["analysis_blocks"] for b in para["para"]]) == "解析：这是解析"
    assert blocks_text([b for para in questions[1]["answer_blocks"] for b in para["para"]]) == "答案：B"


def test_segment_recognizes_reference_and_final_answer_markers():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("1．第一题"),
            text_para("【参考答案】A"),
            text_para("2．第二题"),
            text_para("故选：B。"),
            text_para("3．第三题"),
            text_para("故答案为：2mol"),
        ]
    )

    assert len(questions) == 3
    assert blocks_text([b for para in questions[0]["answer_blocks"] for b in para["para"]]) == "【参考答案】A"
    assert blocks_text([b for para in questions[1]["answer_blocks"] for b in para["para"]]) == "故选：B。"
    assert blocks_text([b for para in questions[2]["answer_blocks"] for b in para["para"]]) == "故答案为：2mol"


def test_probe_docx_format_counts_native_structures_and_answer_position(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import probe_docx_format

    docx = tmp_path / "probe.docx"
    document_xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:v="urn:schemas-microsoft-com:vml"
                xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
      <w:body>
        <w:p><w:r><w:t>1. 第一题</w:t></w:r></w:p>
        <w:p><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath></w:p>
        <w:p><w:r><w:pict><v:imagedata r:id="rId1"/></w:pict></w:r></w:p>
        <w:p><w:r><w:drawing><a:blip r:embed="rId2"/></w:drawing></w:r></w:p>
        <w:p><w:r><w:t>【答案】1. A</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """
    rels_xml = """
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship Id="rId1" Type="image" Target="media/image1.wmf"/>
      <Relationship Id="rId2" Type="image" Target="media/image2.png"/>
    </Relationships>
    """
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", rels_xml)
        zf.writestr("word/media/image1.wmf", b"wmf")
        zf.writestr("word/media/image2.png", b"png")

    row = probe_docx_format(docx)

    assert row["omml_count"] == 1
    assert row["ole_count"] == 1
    assert row["drawing_count"] == 1
    assert row["media_count"] == 2
    assert row["answer_marker_count"] == 1
    assert row["answer_position"] == "trailing"


def test_table_blocks_handles_omml_cells_without_media_key():
    from scripts.ws1_docx_extract_prototype import table_blocks

    xml = """
    <w:tbl xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
           xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
      <w:tr>
        <w:tc>
          <w:p>
            <w:r><w:t>K=</w:t></w:r>
            <m:oMath><m:r><m:t>c</m:t></m:r></m:oMath>
          </w:p>
        </w:tc>
      </w:tr>
    </w:tbl>
    """

    table = table_blocks(ET.fromstring(xml), {})

    cell = table["rows"][0][0]
    assert isinstance(cell, list)
    assert cell[0] == {"type": "text", "text": "K="}
    assert cell[1]["type"] == "math_omml"
    assert "<m:oMath" in cell[1]["omml"]
    assert "[OMML]" not in str(table["rows"])


def test_blocks_text_extracts_structured_table_cells_without_omml_literal():
    from scripts.ws1_docx_extract_prototype import blocks_text

    text = blocks_text([
        {
            "type": "table",
            "rows": [[
                [
                    {"type": "text", "text": "K="},
                    {"type": "math_omml", "omml": "<m:oMath><m:r><m:t>c</m:t></m:r></m:oMath>", "latex": "c"},
                ],
                "plain",
            ]],
        }
    ])

    assert "K=" in text
    assert "c" in text
    assert "plain" in text
    assert "[OMML]" not in text


def test_parse_answer_table_rows_handles_structured_cells():
    from scripts.ws1_docx_extract_prototype import parse_answer_table_rows

    rows = [
        ["题号", [{"type": "text", "text": "1"}]],
        ["答案", [{"type": "text", "text": "A"}, {"type": "math_omml", "omml": "<m:oMath/>", "latex": "x"}]],
    ]

    assert parse_answer_table_rows(rows) == {1: "Ax"}


def test_render_preview_handles_structured_table_cells(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import render_preview

    questions = [
        {
            "q_num": 1,
            "section": "一、综合题",
            "stem_blocks": [
                {
                    "para": [
                        {
                            "type": "table",
                            "rows": [[[
                                {"type": "text", "text": "K="},
                                {"type": "math_omml", "omml": "<m:oMath><m:r><m:t>c</m:t></m:r></m:oMath>", "latex": "c"},
                            ]]],
                        }
                    ]
                }
            ],
            "answer_blocks": [],
        }
    ]

    render_preview(questions, tmp_path, tmp_path / "preview.html")

    html = (tmp_path / "preview.html").read_text(encoding="utf-8")
    assert "K=" in html
    assert "OMML" in html


def test_normalize_group_key_and_role_collapses_original_analysis_reference_and_duplicates():
    from scripts.ws1_docx_extract_prototype import classify_source_name

    original = classify_source_name("精品解析：2024届上海市徐汇区高三下学期二模化学试题（原卷版）.docx")
    analysis = classify_source_name("2024届上海市徐汇区高三下学期二模化学试题（解析版）.docx")
    duplicate = classify_source_name("2024届上海市徐汇区高三下学期二模化学试题（原卷版）(1).docx")
    answer_key = classify_source_name("2025届高考模拟测试题模拟卷01（上海专用）（参考答案） .docx")
    exam = classify_source_name("2025届高考模拟测试题模拟卷01（上海专用）（考试版） .docx")

    assert original["group_key"] == analysis["group_key"] == duplicate["group_key"]
    assert original["role"] == "question_source"
    assert duplicate["is_duplicate_name"]
    assert analysis["role"] == "analysis"
    assert answer_key["group_key"] == exam["group_key"]
    assert answer_key["role"] == "answer_key"
    assert exam["role"] == "question_source"


def test_classify_source_name_marks_pure_answer_doc_as_answer_only():
    from scripts.ws1_docx_extract_prototype import classify_source_name

    row = classify_source_name("答案202501.docx")

    assert row["role"] == "answer_only"
    assert row["role_inferred_reason"] == "filename_answer_only"


def test_answer_only_source_is_not_question_source_but_is_answer_source():
    from scripts.ws1_docx_extract_prototype import choose_group_source, select_answer_sources

    question_source = {
        "role": "question_source",
        "status": "ok",
        "route": "docx_native",
        "file_name": "试卷.docx",
        "path": "paper.docx",
        "text_char_count": 5000,
    }
    answer_only = {
        "role": "answer_only",
        "status": "ok",
        "route": "docx_native",
        "file_name": "答案202501.docx",
        "path": "answer.docx",
        "text_char_count": 800,
    }
    group = {"unique_sources": [answer_only, question_source]}

    assert choose_group_source(group, {"analysis", "unknown"}) is None
    assert choose_group_source(group, {"question_source"}) is question_source
    assert select_answer_sources(group, question_source) == [answer_only]


def test_answer_only_only_group_is_not_extractable_question_group():
    from scripts.ws1_docx_extract_prototype import has_extractable_question_source

    group = {
        "unique_sources": [
            {"role": "answer_only", "status": "ok", "route": "docx_native"},
        ]
    }

    assert has_extractable_question_source(group) is False


def test_refine_source_role_records_content_evidence_for_answer_only_doc():
    from scripts.ws1_docx_extract_prototype import refine_source_role_with_probe

    record = refine_source_role_with_probe(
        {
            "role": "answer_only",
            "role_inferred_reason": "filename_answer_only",
            "text_char_count": 866,
            "question_prompt_count": 0,
            "answer_fragment_line_count": 12,
            "q_start_count": 12,
            "option_marker_count": 0,
        }
    )

    assert record["role"] == "answer_only"
    assert "content_answer_dominant" in record["role_inferred_reason"]
    assert "answer_only_evidence" in record


def test_segment_does_not_open_new_question_for_answer_enumeration_lines():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("一、选择题"),
            text_para("1. 第一题"),
            text_para("【答案】A"),
            text_para("【解析】"),
            text_para("2. B"),
            text_para("2. 第二题"),
        ]
    )

    assert len(questions) == 2
    assert [q["q_num"] for q in questions] == [1, 2]
    second_stem = blocks_text([b for para in questions[1]["stem_blocks"] for b in para["para"]])
    assert "2. B" not in second_stem
    assert "2. 第二题" in second_stem


def test_segment_keeps_short_scored_answer_fragment_with_previous_question():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("一、填空题"),
            text_para("4．由煤和石油燃烧导致的酸雨样品，放置一段时间后pH会     。（填“变大”或“变小”）"),
            text_para("4．变小（2分）"),
            text_para("5．下列说法正确的是"),
        ]
    )

    assert [q["q_num"] for q in questions] == [4, 5]
    first_answer = blocks_text([b for para in questions[0]["answer_blocks"] for b in para["para"]])
    second_stem = blocks_text([b for para in questions[1]["stem_blocks"] for b in para["para"]])
    assert "变小" in first_answer
    assert "变小" not in second_stem


def test_segment_plain_answer_marker_prevents_answer_key_sections_opening_questions():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("一、填空题"),
            text_para("4．由煤和石油燃烧导致的酸雨样品，放置一段时间后pH会     。（填“变大”或“变小”）"),
            text_para("5．下列说法正确的是"),
            text_para("答案"),
            text_para("一、氮元素的循环（本题18分）"),
            text_para("4．变小（2分）"),
            text_para("5．C（1分），D（1分）"),
            text_para("二、碳酸钙的制备（本题22分）"),
            text_para("1．AD（2分）"),
            text_para("四、光聚合的光引发剂（本题22分）"),
            text_para("10．"),
            text_para("图（4分）"),
        ]
    )

    assert [q["q_num"] for q in questions] == [4, 5]
    stems = [
        blocks_text([b for para in q["stem_blocks"] for b in para["para"]])
        for q in questions
    ]
    assert all("变小（2分）" not in stem for stem in stems)
    assert all("图（4分）" not in stem for stem in stems)


def test_segment_inline_answer_marker_does_not_start_global_answer_key_mode():
    from scripts.ws1_docx_extract_prototype import segment

    questions = segment(
        [
            text_para("一、第一大题"),
            text_para("1．第一题"),
            text_para("【答案】A"),
            text_para("【解析】第一题解析"),
            text_para("二、第二大题"),
            text_para("2．第二题"),
        ]
    )

    assert [(q["section_num"], q["q_num"]) for q in questions] == [(1, 1), (2, 2)]


def test_segment_does_not_rewind_to_smaller_question_number_inside_analysis_zone():
    from scripts.ws1_docx_extract_prototype import segment

    questions = segment(
        [
            text_para("一、选择题"),
            text_para("18. 第十八题"),
            text_para("【答案】A"),
            text_para("【解析】"),
            text_para("19. 第十九题"),
            text_para("【答案】C"),
            text_para("【解析】"),
            text_para("13. 某步解析编号"),
            text_para("20. 第二十题"),
        ]
    )

    assert [q["q_num"] for q in questions] == [18, 19, 20]


def test_segment_treats_trial_analysis_marker_as_analysis_not_new_question():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("1. 第一题"),
            text_para("【答案】A"),
            text_para("2. 【试题解析】这是第一题解析编号,不是第二题题干"),
            text_para("2. 下列物质说法正确的是"),
            text_para("【答案】B"),
        ]
    )

    assert [q["q_num"] for q in questions] == [1, 2]
    first_analysis = blocks_text([b for para in questions[0]["analysis_blocks"] for b in para["para"]])
    second_stem = blocks_text([b for para in questions[1]["stem_blocks"] for b in para["para"]])
    assert "试题解析" in first_analysis
    assert "试题解析" not in second_stem


def test_segment_keeps_numbered_answer_fragment_with_trial_analysis_in_analysis_zone():
    from scripts.ws1_docx_extract_prototype import blocks_text, segment

    questions = segment(
        [
            text_para("3. 第三题"),
            text_para("【答案】A"),
            text_para("【解析】"),
            text_para("4、分子结构相似，相对分子质量BF3＜BCl3【试题解析】这是上一题解析"),
            text_para("4. 下列说法正确的是"),
            text_para("【答案】B"),
        ]
    )

    assert [q["q_num"] for q in questions] == [3, 4]
    first_analysis = blocks_text([b for para in questions[0]["analysis_blocks"] for b in para["para"]])
    second_stem = blocks_text([b for para in questions[1]["stem_blocks"] for b in para["para"]])
    assert "分子结构相似" in first_analysis
    assert "试题解析" in first_analysis
    assert "试题解析" not in second_stem


def test_quality_gate_flags_trial_analysis_marker_in_stem():
    from scripts.ws1_docx_extract_prototype import apply_quality_gates

    questions = [
        {
            "stem_blocks": [{"para": [{"type": "text", "text": "【试题解析】故选A"}]}],
            "answer_blocks": [],
            "analysis_blocks": [],
            "quality_flags": [],
        }
    ]

    report = apply_quality_gates(questions)

    assert report["stem_contaminated"] == 1
    assert "stem_contaminated" in questions[0]["quality_flags"]


def test_quality_gate_flags_trial_analysis_marker_with_spaces_in_stem():
    from scripts.ws1_docx_extract_prototype import apply_quality_gates

    questions = [
        {
            "stem_blocks": [{"para": [{"type": "text", "text": "【 试题 解析 】故选A"}]}],
            "answer_blocks": [],
            "analysis_blocks": [],
            "quality_flags": [],
        }
    ]

    report = apply_quality_gates(questions)

    assert report["stem_contaminated"] == 1
    assert "stem_contaminated" in questions[0]["quality_flags"]


def test_extract_section_answer_key_maps_answers_to_question_sequence():
    from scripts.ws1_docx_extract_prototype import extract_section_answer_key_map

    paragraphs = [
        text_para("一、超级钢的组成与性质"),
        text_para("【答案】(1)N (2) 24NA"),
        text_para("二、铜与铜的化合物"),
        text_para("【答案】(1) Cu+ (2) 蓝色变黄色"),
    ]

    answer_map = extract_section_answer_key_map(paragraphs)

    assert answer_map[1] == "【答案】(1)N (2) 24NA"
    assert answer_map[2] == "【答案】(1) Cu+ (2) 蓝色变黄色"


def test_extract_reference_answer_table_maps_numbered_columns():
    from scripts.ws1_docx_extract_prototype import extract_reference_answer_table_map

    paragraphs = [
        text_para("参考答案"),
        [
            {
                "type": "table",
                "rows": [
                    ["题号", "1", "2", "3"],
                    ["答案", "A", "B", "C"],
                ],
            }
        ],
    ]

    answer_map = extract_reference_answer_table_map(paragraphs)

    assert answer_map == {1: "A", 2: "B", 3: "C"}


def test_merge_answers_into_questions_uses_reference_answer_table(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import blocks_text, merge_answers_into_questions

    answer_docx = tmp_path / "answer_key.docx"
    document_xml = """
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>参考答案</w:t></w:r></w:p>
        <w:tbl>
          <w:tr>
            <w:tc><w:p><w:r><w:t>题号</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>2</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>3</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc><w:p><w:r><w:t>答案</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>C</w:t></w:r></w:p></w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>
    """
    rels_xml = """
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
    """
    with zipfile.ZipFile(answer_docx, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", rels_xml)

    targets = [
        {"q_num": 1, "section_num": 1, "question_id": "1-1", "answer_blocks": [], "analysis_blocks": []},
        {"q_num": 2, "section_num": 1, "question_id": "1-2", "answer_blocks": [], "analysis_blocks": []},
        {"q_num": 3, "section_num": 1, "question_id": "1-3", "answer_blocks": [], "analysis_blocks": []},
    ]

    result = merge_answers_into_questions(
        targets,
        [{"path": str(answer_docx), "original_path": str(answer_docx), "role": "answer_key"}],
    )

    assert result["assigned"] == 3
    assert [blocks_text([b for para in q["answer_blocks"] for b in para["para"]]) for q in targets] == [
        "【答案】A",
        "【答案】B",
        "【答案】C",
    ]


def test_is_scan_fallback_probe_row_requires_low_text_and_large_page_media():
    from scripts.ws1_docx_extract_prototype import is_scan_fallback_probe_row

    assert is_scan_fallback_probe_row(
        {
            "text_char_count": 0,
            "media_count": 17,
            "large_page_media_count": 17,
            "non_page_media_count": 0,
        }
    )
    assert not is_scan_fallback_probe_row(
        {
            "text_char_count": 320,
            "media_count": 17,
            "large_page_media_count": 17,
            "non_page_media_count": 0,
        }
    )
    assert not is_scan_fallback_probe_row(
        {
            "text_char_count": 0,
            "media_count": 17,
            "large_page_media_count": 16,
            "non_page_media_count": 1,
        }
    )


def test_group_source_records_dedupes_duplicate_hashes_and_keeps_roles():
    from scripts.ws1_docx_extract_prototype import group_source_records

    groups = group_source_records(
        [
            {
                "path": "/papers/x_orig.docx",
                "group_key": "x",
                "role": "question_source",
                "sha1": "h1",
                "file_name": "x（原卷版）.docx",
            },
            {
                "path": "/papers/x_orig(1).docx",
                "group_key": "x",
                "role": "question_source",
                "sha1": "h1",
                "file_name": "x（原卷版）(1).docx",
            },
            {
                "path": "/papers/x_analysis.docx",
                "group_key": "x",
                "role": "analysis",
                "sha1": "h2",
                "file_name": "x（解析版）.docx",
            },
            {
                "path": "/papers/x_answer.docx",
                "group_key": "x",
                "role": "answer_key",
                "sha1": "h3",
                "file_name": "x（参考答案）.docx",
            },
        ]
    )

    assert len(groups) == 1
    group = groups[0]
    assert group["group_key"] == "x"
    assert len(group["unique_sources"]) == 3
    assert len(group["duplicate_sources"]) == 1
    assert [row["role"] for row in group["unique_sources"]] == ["question_source", "analysis", "answer_key"]


def test_group_route_skips_when_question_source_is_scan_fallback_without_native_question_source():
    from scripts.ws1_docx_extract_prototype import group_should_scan_fallback

    group = {
        "unique_sources": [
            {"role": "question_source", "route": "scan_fallback", "status": "ok"},
            {"role": "analysis", "route": "docx_native", "status": "ok"},
        ]
    }

    assert group_should_scan_fallback(group)

    group["unique_sources"].append({"role": "question_source", "route": "docx_native", "status": "ok"})
    assert not group_should_scan_fallback(group)


def test_merge_answers_into_questions_matches_compound_question_id():
    from scripts.ws1_docx_extract_prototype import merge_answer_questions_into_targets

    targets = [
        {
            "q_num": 1,
            "section_num": 1,
            "question_id": "1-1",
            "answer_blocks": [],
            "analysis_blocks": [],
        },
        {
            "q_num": 1,
            "section_num": 2,
            "question_id": "2-1",
            "answer_blocks": [],
            "analysis_blocks": [],
        },
    ]
    answer_questions = [
        {
            "q_num": 1,
            "section_num": 1,
            "question_id": "1-1",
            "answer_blocks": [text_para("【答案】A")],
            "analysis_blocks": [],
        },
        {
            "q_num": 1,
            "section_num": 2,
            "question_id": "2-1",
            "answer_blocks": [text_para("【答案】B")],
            "analysis_blocks": [],
        },
    ]

    assigned = merge_answer_questions_into_targets(targets, answer_questions, "analysis.docx", "analysis")

    assert assigned == 2
    assert targets[0]["answer_blocks"] == [text_para("【答案】A")]
    assert targets[1]["answer_blocks"] == [text_para("【答案】B")]


def test_merge_answers_into_questions_continues_after_partial_section_key(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import merge_answers_into_questions

    def write_docx(path: Path, paragraphs: list[str]) -> None:
        body = "\n".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
        document_xml = f"""
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>{body}</w:body>
        </w:document>
        """
        rels_xml = """
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
        """
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("word/document.xml", document_xml)
            zf.writestr("word/_rels/document.xml.rels", rels_xml)

    answer_docx = tmp_path / "analysis.docx"
    write_docx(
        answer_docx,
        [
            "一、选择题",
            "1. 第一题",
            "【答案】A",
            "【解析】略",
            "2. 第二题",
            "【答案】B",
            "【解析】略",
            "二、综合题",
            "1. 综合题",
            "【答案】C",
            "【解析】略",
        ],
    )
    targets = [
        {"q_num": 1, "section_num": 1, "question_id": "1-1", "answer_blocks": [], "analysis_blocks": []},
        {"q_num": 2, "section_num": 1, "question_id": "1-2", "answer_blocks": [], "analysis_blocks": []},
        {"q_num": 1, "section_num": 2, "question_id": "2-1", "answer_blocks": [], "analysis_blocks": []},
    ]

    result = merge_answers_into_questions(
        targets,
        [{"path": str(answer_docx), "original_path": str(answer_docx), "role": "analysis"}],
    )

    assert result["assigned"] == 3
    assert [bool(q["answer_blocks"]) for q in targets] == [True, True, True]


def test_golden_candidate_classifier_requires_question_source_answer_and_visual_block():
    from scripts.ws1_docx_extract_prototype import (
        classify_golden_candidate_categories,
        is_golden_candidate_eligible,
    )

    question = {
        "source_role": "question_source",
        "answer_blocks": [text_para("【答案】A")],
        "stem_blocks": [
            {
                "para": [
                    {"type": "text", "text": "如图所示实验装置用于制取气体，写出反应方程式。"},
                    {"type": "figure", "media": "image1.png"},
                ]
            }
        ],
    }

    assert is_golden_candidate_eligible(question)
    categories = classify_golden_candidate_categories(question)
    assert "device" in categories
    assert "equation" in categories

    question["source_role"] = "analysis"
    assert not is_golden_candidate_eligible(question)


def test_select_golden_candidates_exports_question_and_assets(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import select_golden_candidates

    batch_root = tmp_path / "batch"
    paper_dir = batch_root / "paper"
    assets_dir = paper_dir / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "image1.png").write_bytes(b"png")
    question = {
        "q_num": 1,
        "section_num": 1,
        "question_id": "1-1",
        "section": "一、实验题",
        "source_role": "question_source",
        "source_path": "original.docx",
        "group_key": "测试卷",
        "stem_blocks": [
            {
                "para": [
                    {"type": "text", "text": "如图所示实验装置用于制取气体。"},
                    {"type": "figure", "media": "image1.png"},
                ]
            }
        ],
        "answer_blocks": [text_para("【答案】A")],
        "analysis_blocks": [],
    }
    (paper_dir / "questions.jsonl").write_text(json.dumps(question, ensure_ascii=False) + "\n", encoding="utf-8")
    (paper_dir / "summary.json").write_text(
        json.dumps({"group_key": "测试卷", "out_dir": str(paper_dir)}, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = select_golden_candidates(batch_root, batch_root / "golden_candidates", quotas={"device": 1})

    assert summary["selected_total"] == 1
    candidate_dir = batch_root / "golden_candidates" / "device_001"
    assert (candidate_dir / "question.json").exists()
    assert (candidate_dir / "assets" / "image1.png").read_bytes() == b"png"
    index = (batch_root / "golden_candidates" / "candidates_index.md").read_text(encoding="utf-8")
    assert "device_001" in index


def test_select_golden_candidates_rejects_missing_answer_assets(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import select_golden_candidates

    batch_root = tmp_path / "batch"
    paper_dir = batch_root / "paper"
    assets_dir = paper_dir / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "image1.png").write_bytes(b"stem")
    question = {
        "q_num": 1,
        "section_num": 1,
        "question_id": "1-1",
        "source_role": "question_source",
        "source_path": "original.docx",
        "group_key": "测试卷",
        "stem_blocks": [
            {
                "para": [
                    {"type": "text", "text": "如图所示实验装置用于制取气体。"},
                    {"type": "figure", "media": "image1.png"},
                ]
            }
        ],
        "answer_blocks": [
            {"para": [{"type": "text", "text": "【答案】A"}, {"type": "formula", "media": "answer_missing.wmf"}]}
        ],
        "analysis_blocks": [],
    }
    (paper_dir / "questions.jsonl").write_text(json.dumps(question, ensure_ascii=False) + "\n", encoding="utf-8")
    (paper_dir / "summary.json").write_text(
        json.dumps({"group_key": "测试卷", "out_dir": str(paper_dir)}, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = select_golden_candidates(batch_root, batch_root / "golden_candidates", quotas={"device": 1})

    assert summary["selected_total"] == 0
    assert summary["missing_asset_rejected"] == 1


def test_group_quality_gate_defers_choice_answer_type_to_final_dataset_gate():
    from scripts.ws1_docx_extract_prototype import apply_final_dataset_answer_type_gate, apply_quality_gates

    question = {
        "group_key": "2013年高考化学试卷上海",
        "q_num": 2,
        "section": "一、选择题",
        "stem_blocks": [
            {
                "para": [
                    {"type": "text", "text": "2．下列说法正确的是（   ）A．甲 B．乙 C．丙 D．丁"}
                ]
            }
        ],
        "answer_blocks": [text_para("【答案】36L（标准状况）二氧化硫")],
        "analysis_blocks": [],
    }

    report = apply_quality_gates([question])

    assert report["answer_type_mismatch"] == 0
    assert question["answer_blocks"]
    assert "answer_type_mismatch" not in question.get("quality_flags", [])

    final_report = apply_final_dataset_answer_type_gate([question])

    assert final_report["cleared_choice_answer_mismatches"] == 1
    assert question["answer_blocks"] == []
    assert "answer_type_mismatch" in question["quality_flags"]


def test_final_dataset_choice_gate_strips_prefix_and_clears_only_strict_choice_mismatches():
    from scripts.ws1_docx_extract_prototype import apply_final_dataset_answer_type_gate, answer_para

    questions = [
        {
            "group_key": "严格选择样例",
            "q_num": 1,
            "section": "",
            "stem_blocks": [
                {
                    "para": [
                        {"type": "text", "text": "1. 下列说法正确的是A. 甲B. 乙C. 丙D. 丁"}
                    ]
                }
            ],
            "answer_blocks": [text_para("答案： B。")],
            "analysis_blocks": [],
        },
        {
            "group_key": "严格选择样例",
            "q_num": 2,
            "section": "",
            "stem_blocks": [
                {
                    "para": [
                        {"type": "text", "text": "2. 下列说法正确的是A. 甲B. 乙C. 丙D. 丁"}
                    ]
                }
            ],
            "answer_blocks": [text_para("【答案】36L（标准状况）")],
            "analysis_blocks": [],
        },
        {
            "group_key": "嵌套小问样例",
            "q_num": 24,
            "section": "",
            "stem_blocks": [
                {
                    "para": [
                        {
                            "type": "text",
                            "text": "24. 完成下列填空：（1）判断沉淀完全的操作是____。（2）下列说法正确的是A. 甲B. 乙C. 丙D. 丁（3）写出方程式____。",
                        }
                    ]
                }
            ],
            "answer_blocks": [text_para("（1）滴加试剂（2）36L（3）方程式")],
            "analysis_blocks": [],
        },
    ]

    report = apply_final_dataset_answer_type_gate(questions)

    assert report["strict_choice_questions"] == 2
    assert report["normalized_choice_answers"] == 1
    assert report["cleared_choice_answer_mismatches"] == 1
    assert report["already_flagged_choice_mismatches"] == 0
    assert questions[0]["answer_blocks"] == [answer_para("B")]
    assert questions[1]["answer_blocks"] == []
    assert "answer_type_mismatch" in questions[1]["quality_flags"]
    assert questions[2]["answer_blocks"]
    assert "answer_type_mismatch" not in questions[2].get("quality_flags", [])


def test_stem_contamination_gate_marks_numbered_and_reference_answers():
    from scripts.ws1_docx_extract_prototype import apply_quality_gates

    questions = [
        {
            "group_key": "污染样例",
            "q_num": 1,
            "section": "",
            "stem_blocks": [{"para": [{"type": "text", "text": "【33题答案】A 这不是题干"}]}],
            "answer_blocks": [],
            "analysis_blocks": [],
        },
        {
            "group_key": "污染样例",
            "q_num": 2,
            "section": "",
            "stem_blocks": [{"para": [{"type": "text", "text": "有机合成题。参考答案 乙烷制乙烯"}]}],
            "answer_blocks": [],
            "analysis_blocks": [],
        },
    ]

    report = apply_quality_gates(questions)

    assert report["stem_contaminated"] == 2
    assert all("stem_contaminated" in q["quality_flags"] for q in questions)


def test_assign_stable_question_ids_preserves_local_id_and_uses_sha1():
    from scripts.ws1_docx_extract_prototype import assign_stable_question_ids

    questions = [
        {
            "question_id": "1-1",
            "q_num": 1,
            "section": "一、选择题",
            "stem_blocks": [{"para": [{"type": "text", "text": "1. 这是一道用于生成稳定ID的题干。"}]}],
        }
    ]

    assign_stable_question_ids(questions, "测试卷")

    assert questions[0]["local_question_id"] == "1-1"
    assert len(questions[0]["question_id"]) == 40
    assert questions[0]["question_id"] != "1-1"


def test_export_golden_candidate_copies_answer_and_analysis_assets(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import export_golden_candidate

    paper_dir = tmp_path / "paper"
    assets_dir = paper_dir / "assets"
    assets_dir.mkdir(parents=True)
    for name in ("stem.wmf", "answer.wmf", "analysis.wmf"):
        (assets_dir / name).write_bytes(name.encode("ascii"))
    question = {
        "stem_blocks": [{"para": [{"type": "formula", "media": "stem.wmf"}]}],
        "answer_blocks": [{"para": [{"type": "formula", "media": "answer.wmf"}]}],
        "analysis_blocks": [{"para": [{"type": "formula", "media": "analysis.wmf"}]}],
    }
    row = {
        "question": question,
        "paper_dir": str(paper_dir),
        "questions_path": str(paper_dir / "questions.jsonl"),
        "line_num": 1,
        "category": "equation",
        "contains": ["formula"],
        "asset_refs": ["stem.wmf", "answer.wmf", "analysis.wmf"],
    }

    metadata = export_golden_candidate(row, tmp_path / "candidate", "equation_001")

    assert metadata["missing_assets"] == []
    assert sorted(metadata["asset_files"]) == ["analysis.wmf", "answer.wmf", "stem.wmf"]
    assert (tmp_path / "candidate" / "assets" / "answer.wmf").exists()


def test_build_golden_round2_preserves_formal_and_backup_category_quotas(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import (
        GOLDEN_CANDIDATE_QUOTAS,
        GOLDEN_ROUND2_BACKUP_QUOTAS,
        GOLDEN_ROUND2_FORMAL_QUOTAS,
        GOLDEN_ROUND2_RETIRED_IDS,
        build_golden_round2,
    )

    def write_candidate(out_dir: Path, category: str, idx: int) -> None:
        candidate_id = f"{category}_{idx:03d}"
        candidate_dir = out_dir / candidate_id
        assets_dir = candidate_dir / "assets"
        assets_dir.mkdir(parents=True)
        asset = "figure.png"
        (assets_dir / asset).write_bytes(b"x" * (25 * 1024))
        question = {
            "group_key": f"{category}-paper",
            "local_question_id": f"1-{idx}",
            "stem_blocks": [{"para": [{"type": "figure", "media": asset}]}],
            "answer_blocks": [{"para": [{"type": "text", "text": "A"}]}],
            "analysis_blocks": [],
            "golden_candidate": {
                "candidate_id": candidate_id,
                "category": category,
                "asset_files": [asset],
                "missing_assets": [],
                "content_key": f"{category}-{idx}",
            },
        }
        (candidate_dir / "question.json").write_text(json.dumps(question, ensure_ascii=False), encoding="utf-8")

    def fake_select(batch_root: Path, out_dir: Path, quotas: dict[str, int]) -> dict:
        for category, quota in quotas.items():
            for idx in range(1, quota + 1):
                write_candidate(out_dir, category, idx)
        return {"selected_total": sum(quotas.values()), "quotas": quotas}

    with patch("scripts.ws1_docx_extract_prototype.select_golden_candidates", side_effect=fake_select), patch(
        "scripts.ws1_docx_extract_prototype.retired_content_keys_from_dir",
        return_value=set(),
    ):
        summary = build_golden_round2(tmp_path / "batch", tmp_path / "source", tmp_path / "round2")

    assert summary["machine_check_pass"] is True
    assert summary["selected_total"] == 60
    assert summary["formal_count"] == 50
    assert summary["backup_count"] == 10
    assert summary["category_counts"] == GOLDEN_CANDIDATE_QUOTAS
    assert summary["formal_category_counts"] == GOLDEN_ROUND2_FORMAL_QUOTAS
    assert summary["backup_category_counts"] == GOLDEN_ROUND2_BACKUP_QUOTAS

    remaining_ids = {path.parent.name for path in (tmp_path / "round2").glob("*/question.json")}
    assert not (remaining_ids & GOLDEN_ROUND2_RETIRED_IDS)

    roles = Counter()
    for question_path in (tmp_path / "round2").glob("*/question.json"):
        question = json.loads(question_path.read_text(encoding="utf-8"))
        roles[(question["golden_candidate"]["category"], question["golden_candidate"]["set_role"])] += 1
    assert roles[("table", "formal")] == 3
    assert roles[("table", "backup")] == 2


def test_v3_batch_report_includes_answer_coverage_distribution(tmp_path: Path):
    from scripts.ws1_docx_extract_prototype import write_v3_batch_report

    write_v3_batch_report(
        tmp_path,
        records=[],
        groups=[],
        summaries=[
            {
                "group_key": "a",
                "questions": 10,
                "answer_coverage": 1.0,
                "route": "paired_question_source",
                "source_roles": {"question_source": 1, "analysis": 1},
                "answer_sources": ["a_analysis.docx"],
                "with_answer": 10,
                "out_dir": "a",
                "large_number_jumps": [],
            },
            {
                "group_key": "b",
                "questions": 10,
                "answer_coverage": 0.5,
                "route": "single_document",
                "source_roles": {"unknown": 1},
                "answer_sources": [],
                "with_answer": 5,
                "out_dir": "b",
                "large_number_jumps": [],
            },
        ],
        failures=[],
        scan_rows=[],
        preview_count=0,
    )

    report = (tmp_path / "BATCH_REPORT.md").read_text(encoding="utf-8")
    assert "Coverage buckets" in report
    assert "Answer-capable papers with >=80% answer coverage" in report
    assert "Papers with answer sources used" in report
