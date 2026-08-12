#!/usr/bin/env python3
"""Focused tests for Batch 11 QA-2 machine audit and merge rules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def clean_rir(item_id: str = "clean"):
    return {
        "item_id": item_id,
        "zones": {
            "stem": [
                [{"kind": "text", "text": "1. 下列说法正确的是"}],
                [{"kind": "text", "text": "A．甲"}],
                [{"kind": "text", "text": "B．乙"}],
                [{"kind": "text", "text": "C．丙"}],
                [{"kind": "text", "text": "D．丁"}],
            ],
            "answer": [[{"kind": "text", "text": "B"}]],
        },
    }


def clean_item(item_id: str = "clean"):
    return {
        "item_id": item_id,
        "group_key": "unit",
        "q_num": 1,
        "section_num": 1,
        "stem_blocks": [
            {"para": [{"type": "text", "text": "1. 下列说法正确的是"}]},
            {"para": [{"type": "text", "text": "A．甲"}]},
            {"para": [{"type": "text", "text": "B．乙"}]},
            {"para": [{"type": "text", "text": "C．丙"}]},
            {"para": [{"type": "text", "text": "D．丁"}]},
        ],
        "answer_blocks_effective": [{"para": [{"type": "text", "text": "B"}]}],
        "analysis_blocks": [],
        "standard_solution": {"final_answers": ["B"], "standard_answer": "B"},
    }


def test_clean_item_passes_all_machine_dimensions():
    from scripts.qa_item_auditor import DIMENSIONS, machine_audit_item

    row = machine_audit_item(clean_item(), rir=clean_rir(), latex_compile_map={})

    assert len(DIMENSIONS) == 20
    assert row["machine_pass_count"] == 20
    assert row["machine_failed_dimensions"] == []
    assert all(row["dimensions"][name] is True for name in DIMENSIONS)
    assert row["reviewer"] == ""
    assert row["review_status"] == "pending"


def test_machine_audit_catches_each_negative_dimension():
    from scripts.qa_item_auditor import DIMENSIONS, machine_audit_item

    cases = []

    def add(name, item, rir, latex_compile_map=None):
        row = machine_audit_item(item, rir=rir, latex_compile_map=latex_compile_map or {})
        cases.append((name, row))

    bad_options = clean_item("bad_options")
    bad_options["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 下列说法正确的是 A．甲B．乙"}]}]
    bad_options_rir = {"item_id": "bad_options", "zones": {"stem": [[{"kind": "text", "text": "1. 下列说法正确的是 A．甲B．乙"}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("options", bad_options, bad_options_rir)

    truncated = clean_item("truncated")
    truncated["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 下列说法正确的是，"}]}]
    truncated_rir = {"item_id": "truncated", "zones": {"stem": [[{"kind": "text", "text": "1. 下列说法正确的是，"}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("truncated", truncated, truncated_rir)

    crossed = clean_item("crossed")
    crossed["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 下列说法正确的是\n【答案】上一题解析\n2．下一题"}]}]
    crossed_rir = {"item_id": "crossed", "zones": {"stem": [[{"kind": "text", "text": "1. 下列说法正确的是\n【答案】上一题解析\n2．下一题"}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("crossed", crossed, crossed_rir)

    fragment = clean_item("fragment")
    fragment["stem_blocks"] = [{"para": [{"type": "text", "text": "13. 写出X的结构简式"}]}]
    fragment_rir = {"item_id": "fragment", "zones": {"stem": [[{"kind": "text", "text": "13. 写出X的结构简式"}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("fragment", fragment, fragment_rir)

    no_answer = clean_item("no_answer")
    no_answer["answer_blocks_effective"] = []
    no_answer["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    no_answer_rir = {"item_id": "no_answer", "zones": {"stem": clean_rir()["zones"]["stem"], "answer": [[{"kind": "text", "text": ""}]]}}
    add("no_answer", no_answer, no_answer_rir)

    bad_answer = clean_item("bad_answer")
    bad_answer["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "BB"}]}]
    bad_answer["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    bad_answer_rir = {"item_id": "bad_answer", "zones": {"stem": clean_rir()["zones"]["stem"], "answer": [[{"kind": "text", "text": "BB"}]]}}
    add("bad_answer", bad_answer, bad_answer_rir)

    missing_subanswer = clean_item("missing_subanswer")
    missing_subanswer["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 回答（1）甲、（2）乙、（3）丙。"}]}]
    missing_subanswer["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "（1）A"}]}]
    missing_subanswer["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    missing_subanswer_rir = {"item_id": "missing_subanswer", "zones": {"stem": [[{"kind": "text", "text": "1. 回答（1）甲、（2）乙、（3）丙。"}]], "answer": [[{"kind": "text", "text": "（1）A"}]]}}
    add("missing_subanswer", missing_subanswer, missing_subanswer_rir)

    literal = clean_item("literal")
    literal["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 表格含 [formula:image16.wmf]。"}]}]
    literal_rir = {"item_id": "literal", "zones": {"stem": [[{"kind": "text", "text": "1. 表格含 [formula:image16.wmf]。"}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("literal", literal, literal_rir)

    placeholder = clean_item("placeholder")
    placeholder_rir = {"item_id": "placeholder", "zones": {"stem": [[{"kind": "placeholder", "reason": "asset_degraded:manual_queue"}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("placeholder", placeholder, placeholder_rir)

    latex = clean_item("latex")
    latex_rir = {"item_id": "latex", "zones": {"stem": [[{"kind": "latex", "latex": r"\frac{2}{3}"}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("latex", latex, latex_rir, {r"\frac{2}{3}": {"ok": False, "error": "bad"}})

    ion = clean_item("ion")
    ion["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 溶液中含 NH+4。"}]}]
    ion_rir = {"item_id": "ion", "zones": {"stem": [[{"kind": "text", "text": "1. 溶液中含 NH+4。"}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("ion", ion, ion_rir)

    table = clean_item("table")
    table["analysis_blocks"] = [{"para": [{"type": "table", "rows": [[""], [""]]}]}]
    add("table", table, clean_rir("table"))

    image = clean_item("image")
    image_rir = {"item_id": "image", "zones": {"stem": [[{"kind": "image", "url": "/api/v4/raw_assets?group_key=g&media=x.png", "asset_hash": None}]], "answer": [[{"kind": "text", "text": "B"}]]}}
    add("image", image, image_rir)

    hollow_subanswer = clean_item("hollow_subanswer")
    hollow_subanswer["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "（1）A；（2）（3）B；或。"}]}]
    hollow_subanswer["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    hollow_subanswer_rir = {
        "item_id": "hollow_subanswer",
        "zones": {"stem": clean_rir()["zones"]["stem"], "answer": [[{"kind": "text", "text": "（1）A；（2）（3）B；或。"}]]},
    }
    add("hollow_subanswer", hollow_subanswer, hollow_subanswer_rir)

    hollow_mention = clean_item("hollow_mention")
    hollow_mention["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "【答案】结构简式为。"}]}]
    hollow_mention["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    hollow_mention_rir = {
        "item_id": "hollow_mention",
        "zones": {"stem": clean_rir()["zones"]["stem"], "answer": [[{"kind": "text", "text": "【答案】结构简式为。"}]]},
    }
    add("hollow_mention", hollow_mention, hollow_mention_rir)

    fragment_literal = clean_item("fragment_literal")
    fragment_literal["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "答案首段残留 wmf]，其后为空。"}]}]
    fragment_literal["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    fragment_literal_rir = {
        "item_id": "fragment_literal",
        "zones": {"stem": clean_rir()["zones"]["stem"], "answer": [[{"kind": "text", "text": "答案首段残留 wmf]，其后为空。"}]]},
    }
    add("fragment_literal", fragment_literal, fragment_literal_rir)

    partial_answer = clean_item("partial_answer")
    partial_answer["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 回答下列问题：\n（1）现象。\n（2）原因。"}]}]
    partial_answer["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "不能"}]}]
    partial_answer["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    partial_answer_rir = {
        "item_id": "partial_answer",
        "zones": {"stem": [[{"kind": "text", "text": "1. 回答下列问题：\n（1）现象。\n（2）原因。"}]], "answer": [[{"kind": "text", "text": "不能"}]]},
    }
    add("partial_answer", partial_answer, partial_answer_rir)

    answer_mismatch = clean_item("answer_mismatch")
    answer_mismatch["q_num"] = 5
    answer_mismatch["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "【正确答案】1、A；2、B"}]}]
    answer_mismatch["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    answer_mismatch_rir = {
        "item_id": "answer_mismatch",
        "zones": {
            "stem": clean_rir()["zones"]["stem"],
            "answer": [[{"kind": "text", "text": "【正确答案】1、A；2、B"}]],
        },
    }
    add("answer_mismatch", answer_mismatch, answer_mismatch_rir)

    failed_union = set()
    for _name, row in cases:
        failed_union.update(row["machine_failed_dimensions"])

    assert len(DIMENSIONS) == 20
    assert failed_union == set(DIMENSIONS)


def test_merge_rules_clean_fixable_blocked_and_pending():
    from scripts.qa_item_auditor import merge_audits

    machine_rows = [
        {
            "item_id": "clean",
            "group_key": "g",
            "reviewer": "",
            "review_status": "pending",
            "machine_pass_count": 15,
            "machine_failed_dimensions": [],
            "evidence": {},
        },
        {
            "item_id": "fixable",
            "group_key": "g",
            "reviewer": "",
            "review_status": "pending",
            "machine_pass_count": 14,
            "machine_failed_dimensions": ["option_no_sticky"],
            "evidence": {"option_no_sticky": [{"type": "tight_option_join"}]},
        },
        {
            "item_id": "blocked",
            "group_key": "g",
            "reviewer": "",
            "review_status": "pending",
            "machine_pass_count": 14,
            "machine_failed_dimensions": ["answer_nonempty"],
            "evidence": {"answer_nonempty": [{"answer_blocks": 0}]},
        },
    ]
    vl_rows = [
        {"item_id": "clean", "verdict": "usable", "reason": "ok"},
        {"item_id": "fixable", "verdict": "minor_issue", "reason": "layout"},
        {"item_id": "blocked", "verdict": "usable", "reason": "ok"},
    ]

    rows = merge_audits(machine_rows, vl_rows)
    by_id = {row["item_id"]: row for row in rows}

    assert by_id["clean"]["pool"] == "clean"
    assert by_id["fixable"]["pool"] == "fixable"
    assert by_id["blocked"]["pool"] == "blocked"
    assert by_id["fixable"]["bucket"] == ["切分", "排版"]
    assert all(row["reviewer"] == "" for row in rows)
    assert all(row["review_status"] == "pending" for row in rows)


def test_vl_normalization_defaults_unclear_output_to_minor_issue():
    from scripts.qa_item_auditor import normalize_vl

    row = normalize_vl({"q1": "yes", "reason": "无法确定移动端小图是否完整"})

    assert row["verdict"] == "minor_issue"
    assert row["q1_stem_complete"] == "yes"


def test_answer_duplicate_choice_does_not_cross_answer_sources():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("answer_source_dupe")
    item["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "【答案】A"}]}]
    item["standard_solution"] = {"final_answers": ["A"], "standard_answer": "A"}
    rir = clean_rir("answer_source_dupe")
    rir["zones"]["answer"] = [[{"kind": "text", "text": "【答案】A"}]]

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["answer_format_normal"] is True
    assert "answer_format_normal" not in row["machine_failed_dimensions"]


def test_answer_duplicate_choice_still_catches_same_source_bb():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("true_bb")
    item["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "【答案】BB"}]}]
    item["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    rir = clean_rir("true_bb")
    rir["zones"]["answer"] = [[{"kind": "text", "text": "【答案】BB"}]]

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["answer_format_normal"] is False
    assert row["evidence"]["answer_format_normal"][0]["type"] == "duplicate_choice_letter"


def test_answer_duplicate_choice_ignores_analysis_prose_repeated_letters():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("analysis_repeated_letters")
    item["answer_blocks_effective"] = [
        {
            "para": [
                {
                    "type": "text",
                    "text": "【正确答案】 B\n【试题解析】 A正确；C正确；CCl4可作萃取剂；D正确。",
                }
            ]
        }
    ]
    item["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    rir = clean_rir("analysis_repeated_letters")
    rir["zones"]["answer"] = [[{"kind": "text", "text": item["answer_blocks_effective"][0]["para"][0]["text"]}]]

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["answer_format_normal"] is True


def test_fill_blank_with_sentence_punctuation_is_not_truncated():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("blank_sentence")
    item["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 下列物质中呈酸性的是________。"}]}]
    rir = {
        "item_id": "blank_sentence",
        "zones": {
            "stem": [[{"kind": "text", "text": "1. 下列物质中呈酸性的是________。"}]],
            "answer": [[{"kind": "text", "text": "B"}]],
        },
    }

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["stem_not_truncated"] is True


def test_true_dangling_terminal_still_counts_as_truncated():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("true_truncated")
    item["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 该反应生成"}]}]
    rir = {
        "item_id": "true_truncated",
        "zones": {
            "stem": [[{"kind": "text", "text": "1. 该反应生成"}]],
            "answer": [[{"kind": "text", "text": "B"}]],
        },
    }

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["stem_not_truncated"] is False
    assert row["evidence"]["stem_not_truncated"][0]["type"] == "dangling_terminal"


def test_sticky_options_do_not_cascade_into_option_complete_failure():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("sticky_complete")
    text = "1. 选择正确数据 A. 0.1B. 0.05C. 0.01D. 0.005"
    item["stem_blocks"] = [{"para": [{"type": "text", "text": text}]}]
    rir = {
        "item_id": "sticky_complete",
        "zones": {
            "stem": [[{"kind": "text", "text": text}]],
            "answer": [[{"kind": "text", "text": "A"}]],
        },
    }

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["option_complete"] is True
    assert row["dimensions"]["option_no_sticky"] is False


def test_letter_only_sticky_options_are_complete_but_sticky():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("letter_sticky_complete")
    text = "1. 下列排序正确的是 A. AB. BC. CD. D"
    item["stem_blocks"] = [{"para": [{"type": "text", "text": text}]}]
    rir = {
        "item_id": "letter_sticky_complete",
        "zones": {
            "stem": [[{"kind": "text", "text": text}]],
            "answer": [[{"kind": "text", "text": "A"}]],
        },
    }

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["option_complete"] is True
    assert row["dimensions"]["option_no_sticky"] is False


def test_line_start_option_without_punctuation_counts_as_present():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("line_start_a_option")
    text = "1. 下列有热量放出的过程是\nA 浓硫酸稀释B. 冰融化\nC. 石灰石高温分解D. 断开H-H键"
    item["stem_blocks"] = [{"para": [{"type": "text", "text": text}]}]
    rir = {
        "item_id": "line_start_a_option",
        "zones": {
            "stem": [[{"kind": "text", "text": text}]],
            "answer": [[{"kind": "text", "text": "A"}]],
        },
    }

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["option_complete"] is True
    assert row["dimensions"]["option_no_sticky"] is False


def test_merge_uses_only_true_machine_problem_dimensions():
    from scripts.qa_item_auditor import merge_audits

    machine_rows = [
        {
            "item_id": "non_true_only",
            "group_key": "g",
            "reviewer": "",
            "review_status": "pending",
            "machine_pass_count": 12,
            "machine_failed_dimensions": ["stem_length_reasonable", "subanswer_complete", "table_complete"],
            "evidence": {"stem_length_reasonable": [], "subanswer_complete": [], "table_complete": []},
        },
        {
            "item_id": "true_sticky",
            "group_key": "g",
            "reviewer": "",
            "review_status": "pending",
            "machine_pass_count": 14,
            "machine_failed_dimensions": ["option_no_sticky"],
            "evidence": {"option_no_sticky": [{"type": "tight_option_join"}]},
        },
    ]
    vl_rows = [
        {"item_id": "non_true_only", "verdict": "usable", "reason": "ok"},
        {"item_id": "true_sticky", "verdict": "usable", "reason": "ok"},
    ]

    rows = merge_audits(machine_rows, vl_rows)
    by_id = {row["item_id"]: row for row in rows}

    assert by_id["non_true_only"]["pool"] == "clean"
    assert by_id["non_true_only"]["machine_issue_dimensions"] == []
    assert by_id["true_sticky"]["pool"] == "fixable"
    assert by_id["true_sticky"]["machine_issue_dimensions"] == ["option_no_sticky"]


def test_vl_prompt_forbids_inferring_unrendered_latex_source():
    from scripts.qa_item_auditor import build_vl_prompts

    system, user = build_vl_prompts()

    assert "只依据截图文字内容" in system + user
    assert "不臆测未渲染的 latex 源码" in system + user


def test_vl_prompt_treats_narrow_word_wrap_as_normal_layout():
    from scripts.qa_item_auditor import build_vl_prompts

    system, user = build_vl_prompts()
    prompt = system + user

    assert "窄屏下文本/选项自然换行" in prompt
    assert "不是缺陷" in prompt
    assert "不得因此判 minor_issue 或 broken" in prompt
    assert "verdict 必须是 usable" in prompt
    assert "内容被截断丢失、重叠遮挡、错序" in prompt


def test_screenshot_script_waits_for_fonts_and_stable_katex_then_checks_latex_residue():
    import tempfile

    import scripts.qa_item_auditor as qa

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    old_choose = qa.choose_node_bin
    old_run = qa.subprocess.run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result_path = out_dir / "browser" / "screenshot_result.json"

            def fake_run(*args, **kwargs):
                qa.write_json(result_path, {"ok": True, "screenshots": {}})
                return FakeProc()

            qa.choose_node_bin = lambda explicit=None: "/usr/bin/node"
            qa.subprocess.run = fake_run
            qa.render_screenshots(out_dir=out_dir, records=[], node_bin=None)
            script = (out_dir / "browser" / "qa2_render_screenshots.cjs").read_text(encoding="utf-8")
    finally:
        qa.choose_node_bin = old_choose
        qa.subprocess.run = old_run

    assert "document.fonts.ready" in script
    assert "waitForKatexStable" in script
    assert ".katex" in script
    assert ".rir-image{max-width:100%!important" in script
    assert "constrainOversizedImages" in script
    assert "constrainOversizedLatex" in script
    assert "img.naturalWidth" in script
    assert "setProperty(\"width\"" in script
    assert "page.waitForTimeout(300)" in script
    assert "findLatexSourceResidue" in script
    assert "\\\\frac|\\\\ce\\{|\\\\text\\{" in script
    assert "screenshot_fail" in script


def test_vl_and_merge_rows_keep_screenshot_fail_separate_from_verdict():
    from scripts.qa_item_auditor import merge_audits

    machine_rows = [
        {
            "item_id": "late",
            "group_key": "g",
            "reviewer": "",
            "review_status": "pending",
            "machine_pass_count": 15,
            "machine_failed_dimensions": [],
            "evidence": {},
        }
    ]
    vl_rows = [
        {
            "item_id": "late",
            "verdict": "usable",
            "reason": "截图内容完整",
            "screenshot_fail": {"mobile": [{"match": "\\frac{1}{2}"}]},
        }
    ]

    merged = merge_audits(machine_rows, vl_rows)

    assert merged[0]["pool"] == "clean"
    assert merged[0]["screenshot_fail"] == {"mobile": [{"match": "\\frac{1}{2}"}]}


def test_batch16_subanswer_hollow_treats_visual_node_as_content():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("visual_subanswer")
    item["answer_blocks_effective"] = [
        {
            "para": [
                {"type": "text", "text": "【答案】（1）"},
                {"type": "formula", "media": "ans_abcd1234_image1.wmf"},
                {"type": "text", "text": "（2）B"},
            ]
        }
    ]
    item["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    rir = clean_rir("visual_subanswer")
    rir["zones"]["answer"] = [
        [
            {"kind": "text", "text": "【答案】（1）"},
            {"kind": "image", "asset_hash": "a" * 64, "w": 80, "h": 40},
            {"kind": "text", "text": "（2）B"},
        ]
    ]

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["subanswer_not_hollow"] is True


def test_batch16_hollow_mention_does_not_scan_normal_stem_prompt():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("stem_hollow_prompt")
    item["stem_blocks"] = [
        {"para": [{"type": "text", "text": "1. 有机物X的结构简式为。"}]},
        {"para": [{"type": "text", "text": "A．甲"}]},
        {"para": [{"type": "text", "text": "B．乙"}]},
        {"para": [{"type": "text", "text": "C．丙"}]},
        {"para": [{"type": "text", "text": "D．丁"}]},
    ]
    rir = {
        "item_id": "stem_hollow_prompt",
        "zones": {
            "stem": [[{"kind": "text", "text": "1. 有机物X的结构简式为。"}]],
            "answer": [[{"kind": "text", "text": "A"}]],
        },
    }

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["no_hollow_mention"] is True


def test_batch16_hollow_mention_treats_answer_visual_node_as_content():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("answer_visual_mention")
    item["answer_blocks_effective"] = [
        {
            "para": [
                {"type": "text", "text": "【答案】结构简式为"},
                {"type": "figure", "media": "ans_abcd1234_image2.png"},
                {"type": "text", "text": "。"},
            ]
        }
    ]
    item["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    rir = clean_rir("answer_visual_mention")
    rir["zones"]["answer"] = [
        [
            {"kind": "text", "text": "【答案】结构简式为"},
            {"kind": "image", "asset_hash": "b" * 64, "w": 100, "h": 60},
            {"kind": "text", "text": "。"},
        ]
    ]

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["no_hollow_mention"] is True


def test_batch16_fragment_literal_ignores_omml_nodes_but_catches_visible_text():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("omml_not_literal")
    item["answer_blocks_effective"] = [
        {
            "para": [
                {"type": "text", "text": "【答案】"},
                {"type": "math_omml", "omml": "<m:oMath><m:r><m:t>Na</m:t></m:r></m:oMath>"},
            ]
        }
    ]
    item["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    rir = clean_rir("omml_not_literal")
    rir["zones"]["answer"] = [[{"kind": "latex", "latex": "Na"}]]

    row = machine_audit_item(item, rir=rir, latex_compile_map={"Na": {"ok": True}})

    assert row["dimensions"]["no_fragment_literal"] is True

    item["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "【答案】残留 wmf]"}]}]
    rir["zones"]["answer"] = [[{"kind": "text", "text": "【答案】残留 wmf]"}]]
    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["no_fragment_literal"] is False


def test_batch16_fragment_literal_treats_visual_inside_c_parentheses_as_content():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("c_visual")
    item["answer_blocks_effective"] = [
        {
            "para": [
                {"type": "text", "text": "【答案】c("},
                {"type": "math_omml", "omml": "<m:oMath><m:r><m:t>CO3</m:t></m:r></m:oMath>"},
                {"type": "text", "text": ")较大"},
            ]
        }
    ]
    item["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    rir = clean_rir("c_visual")
    rir["zones"]["answer"] = [
        [
            {"kind": "text", "text": "【答案】c("},
            {"kind": "latex", "latex": "CO_3^{2-}"},
            {"kind": "text", "text": ")较大"},
        ]
    ]

    row = machine_audit_item(item, rir=rir, latex_compile_map={"CO_3^{2-}": {"ok": True}})

    assert row["dimensions"]["no_fragment_literal"] is True


def test_batch16_sub_coverage_exempts_single_choice_and_ignores_stem_step_numbers():
    from scripts.qa_item_auditor import machine_audit_item

    item = clean_item("choice_with_step_nums")
    item["stem_blocks"] = [
        {"para": [{"type": "text", "text": "1. 容器（1）（2）中变化如下，下列说法正确的是"}]},
        {"para": [{"type": "text", "text": "A．甲"}]},
        {"para": [{"type": "text", "text": "B．乙"}]},
        {"para": [{"type": "text", "text": "C．丙"}]},
        {"para": [{"type": "text", "text": "D．丁"}]},
    ]
    item["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "【答案】A"}]}]
    item["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    rir = clean_rir("choice_with_step_nums")

    row = machine_audit_item(item, rir=rir, latex_compile_map={})

    assert row["dimensions"]["stem_answer_sub_coverage"] is True


def test_batch16_answer_stem_match_flags_only_structural_misalignment():
    from scripts.qa_item_auditor import machine_audit_item

    normal = clean_item("normal_equation_answer")
    normal["stem_blocks"] = [{"para": [{"type": "text", "text": "1. 写出该反应的化学方程式。"}]}]
    normal["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "【答案】2H2+O2=2H2O"}]}]
    normal["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    normal_rir = {
        "item_id": "normal_equation_answer",
        "zones": {
            "stem": [[{"kind": "text", "text": "1. 写出该反应的化学方程式。"}]],
            "answer": [[{"kind": "text", "text": "【答案】2H2+O2=2H2O"}]],
        },
    }

    row = machine_audit_item(normal, rir=normal_rir, latex_compile_map={})

    assert row["dimensions"]["answer_stem_match_flag"] is True

    wrong_head = clean_item("wrong_head")
    wrong_head["q_num"] = 5
    wrong_head["answer_blocks_effective"] = [{"para": [{"type": "text", "text": "【正确答案】1、A；2、B"}]}]
    wrong_head["standard_solution"] = {"final_answers": [], "standard_answer": ""}
    wrong_head_rir = clean_rir("wrong_head")
    wrong_head_rir["zones"]["answer"] = [[{"kind": "text", "text": "【正确答案】1、A；2、B"}]]

    row = machine_audit_item(wrong_head, rir=wrong_head_rir, latex_compile_map={})

    assert row["dimensions"]["answer_stem_match_flag"] is False
    assert row["evidence"]["answer_stem_match_flag"][0]["type"] == "answer_key_starts_at_1_for_later_question"

    fragment_stem = clean_item("fragment_stem")
    fragment_stem["stem_blocks"] = [{"para": [{"type": "text", "text": "、继续反应后溶液变蓝。"}]}]
    fragment_rir = {
        "item_id": "fragment_stem",
        "zones": {
            "stem": [[{"kind": "text", "text": "、继续反应后溶液变蓝。"}]],
            "answer": [[{"kind": "text", "text": "B"}]],
        },
    }

    row = machine_audit_item(fragment_stem, rir=fragment_rir, latex_compile_map={})

    assert row["dimensions"]["answer_stem_match_flag"] is False
    assert row["evidence"]["answer_stem_match_flag"][0]["type"] == "stem_starts_with_continuation_punctuation"


def _run_focused() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{failures} FAILURES" if failures else "\nALL PASS")
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_focused())
