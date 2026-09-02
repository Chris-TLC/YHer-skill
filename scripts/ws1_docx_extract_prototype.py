#!/usr/bin/env python3
"""WS1 prototype: extract structured questions from a native docx exam paper.

Proves the four core mechanics on real Shanghai papers:
  1. reading-order traversal (paragraphs + tables)
  2. inline formula (MathType OLE -> WMF preview) placement inside text
  3. figure/media extraction as independent assets
  4. question segmentation via numbering + 【答案】/【解析】 anchors,
     with answer/analysis split into separate fields (never shown to students)

Outputs per paper under /tmp/yher_ws1_proto/<stem>/:
  questions.jsonl   one structured question per line (blocks model, Schema v4 direction)
  assets/           extracted media referenced by the questions
  preview.html      first questions rendered with inline formula images (print-grade check)
  summary.json      counts + segmentation quality signals
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from struct import unpack
from pathlib import Path
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W, "r": R, "a": A, "m": M}
ET.register_namespace("w", W)
ET.register_namespace("m", M)

SOFFICE = "/opt/homebrew/bin/soffice"

Q_START = re.compile(r"^\s*(\d{1,3})[.、．]\s*")
SECTION = re.compile(r"^([一二三四五六七八九十]+)、")
ANSWER = re.compile(r"^(?:【(?:参考|正确)?答案】|参考答案\s*$|答案\s*$|参考答案[:：]|答案[:：]|故选[:：]|故答案为[:：])")
PLAIN_ANSWER_KEY_MARKER = re.compile(r"^(?:参考答案|答案)\s*$")
ANALYSIS = re.compile(r"^(?:【(试题解析|题目解析|解析|详解|分析|点评|导语)】|(?:试题解析|题目解析|解析|详解|分析|点评|导语)[:：])")
ANALYSIS_MARKER_ANY = re.compile(r"【\s*(?:试题\s*解析|题目\s*解析|解析|详解|分析|点评|导语)\s*】|(?:试题\s*解析|题目\s*解析|解析|详解|分析|点评|导语)[:：]")
OPTION = re.compile(r"(?:^|\s)([A-D])[.、．]\s*")
NUMBERED_ANSWER = re.compile(r"(?<!\d)(\d{1,3})[.、．]\s*")
NUMBERED_ANSWER_LINE = re.compile(r"^\s*\d{1,3}[.、．]\s*(?:【(?:参考|正确)?答案】|答案[:：]|故选[:：]|故答案为[:：])")
NUMBERED_ANALYSIS_LINE = re.compile(r"^\s*\d{1,3}[.、．]\s*(?:【(?:试题解析|题目解析|解析|详解|分析|点评|导语)】|(?:试题解析|题目解析|解析|详解|分析|点评|导语)[:：])")
ANSWER_MARKER_ANY = re.compile(r"【(?:参考|正确)?答案】|【\d+题?答案】|答案[:：]|故选[:：]|故答案为[:：]")
INLINE_ANSWER_MARKER = re.compile(r"(【(?:参考|正确)?答案】|【\d+题?答案】|答案[:：]|故选[:：]|故答案为[:：]|参考答案)")
SCORE_MARKER = re.compile(r"[（(]\s*\d+(?:\.\d+)?\s*分\s*[)）]")
STEM_CONTAMINATION_PATTERNS = (
    re.compile(r"【\d+题?答案】"),
    re.compile(r"参考答案\s*\S+"),
    re.compile(r"【(?:参考|正确)?答案】"),
    ANALYSIS_MARKER_ANY,
)
CHOICE_ANSWER_RE = re.compile(r"^[A-D]{1,4}$")
QUESTION_CUE_WORDS = ("下列", "说法", "判断", "写出", "比较", "如图", "已知", "原因", "实验", "关系", "有关")
EXAM_INSTRUCTION_WORDS = (
    "试卷满分",
    "考试时间",
    "答题前",
    "答题纸",
    "答题卡",
    "条形码",
    "不得分",
    "不能错位",
    "正确选项",
    "相对原子质量",
    "可能用到",
)
QUESTION_SOURCE_MARKERS = ("原卷版", "空白卷", "考试版", "原卷")
ANALYSIS_MARKERS = ("解析版", "解析卷", "全解全析", "含解析")
ANSWER_KEY_MARKERS = ("参考答案", "答题卡")
ANSWER_ONLY_NAME_RE = re.compile(r"^(?:答案|参考答案)[\w\u4e00-\u9fff.-]*$")
GROUP_STRIP_MARKERS = QUESTION_SOURCE_MARKERS + ANALYSIS_MARKERS + ANSWER_KEY_MARKERS + ("精品解析：", "精品解析", "（含解析）", "(含解析)")

CN_NUM = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def load_rels(zf: zipfile.ZipFile) -> dict[str, str]:
    xml = zf.read("word/_rels/document.xml.rels").decode("utf-8")
    return dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', xml))


def safe_paper_dir_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    clean = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", path.stem).strip("_")
    return f"{clean[:60]}_{digest}"


def normalize_group_key(name: str) -> str:
    stem = Path(name).stem.strip()
    stem = re.sub(r"^\s*精品解析[:：]\s*", "", stem)
    stem = re.sub(r"\(\d+\)\s*$", "", stem).strip()
    stem = re.sub(r"（\d+）\s*$", "", stem).strip()
    for marker in GROUP_STRIP_MARKERS:
        stem = stem.replace(marker, "")
    stem = re.sub(r"[()（）]", "", stem)
    stem = re.sub(r"\s+", "", stem)
    stem = stem.replace("．", ".").replace("：", "").replace(":", "")
    stem = stem.strip("._- ")
    return stem


def classify_source_name(name: str) -> dict[str, object]:
    stem = Path(name).stem.strip()
    role = "unknown"
    role_inferred_reason = ""
    if ANSWER_ONLY_NAME_RE.match(stem) and not any(marker in stem for marker in ("试题", "试卷", "原卷", "解析")):
        role = "answer_only"
        role_inferred_reason = "filename_answer_only"
    elif any(marker in stem for marker in ANSWER_KEY_MARKERS):
        role = "answer_key"
    elif any(marker in stem for marker in ANALYSIS_MARKERS):
        role = "analysis"
    elif any(marker in stem for marker in QUESTION_SOURCE_MARKERS):
        role = "question_source"
    row = {
        "name": name,
        "stem": stem,
        "group_key": normalize_group_key(name),
        "role": role,
        "is_duplicate_name": bool(re.search(r"(\(\d+\)|（\d+）)\s*$", stem)),
    }
    if role_inferred_reason:
        row["role_inferred_reason"] = role_inferred_reason
    return row


def refine_source_role_with_probe(record: dict[str, object]) -> dict[str, object]:
    if record.get("role") != "answer_only":
        return record
    evidence: list[str] = []
    text_chars = int(record.get("text_char_count") or 0)
    q_prompts = int(record.get("question_prompt_count") or 0)
    answer_fragments = int(record.get("answer_fragment_line_count") or 0)
    q_starts = int(record.get("q_start_count") or 0)
    option_markers = int(record.get("option_marker_count") or 0)
    if text_chars and text_chars <= 3000:
        evidence.append("short_document")
    if q_prompts <= 1:
        evidence.append("low_question_prompt_count")
    if q_starts and answer_fragments >= max(2, q_starts // 2):
        evidence.append("content_answer_dominant")
    elif answer_fragments >= 3:
        evidence.append("numbered_answer_fragments_present")
    if option_markers < 4:
        evidence.append("low_option_marker_count")
    if evidence:
        record["answer_only_evidence"] = evidence
        reason = str(record.get("role_inferred_reason") or "answer_only")
        for item in evidence:
            if item not in reason:
                reason += f"+{item}"
        record["role_inferred_reason"] = reason
    return record


def iter_source_files(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".docx", ".doc"} and not path.name.startswith("~$")
        ],
        key=lambda p: str(p),
    )


def convert_doc_to_docx(doc: Path, out_dir: Path) -> tuple[Path | None, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / f"{doc.stem}.docx"
    if expected.exists():
        return expected, ""
    try:
        subprocess.run(
            [SOFFICE, "--headless", "--convert-to", "docx", str(doc), "--outdir", str(out_dir)],
            capture_output=True,
            timeout=120,
            check=True,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if expected.exists():
        return expected, ""
    matches = sorted(out_dir.glob(f"{doc.stem}*.docx"))
    if matches:
        return matches[0], ""
    return None, "converted docx not found"


def chinese_section_num(value: str) -> int | None:
    if value in CN_NUM:
        return CN_NUM[value]
    if value.startswith("十") and len(value) == 2 and value[1] in CN_NUM:
        return 10 + CN_NUM[value[1]]
    if value.endswith("十") and len(value) == 2 and value[0] in CN_NUM:
        return CN_NUM[value[0]] * 10
    if "十" in value and len(value) == 3 and value[0] in CN_NUM and value[2] in CN_NUM:
        return CN_NUM[value[0]] * 10 + CN_NUM[value[2]]
    return None


def para_blocks(p: ET.Element, rels: dict[str, str]) -> list[dict]:
    """Paragraph -> ordered blocks: text | math_omml | formula(media) | figure(media)."""
    blocks: list[dict] = []

    def push_text(s: str) -> None:
        if not s:
            return
        if blocks and blocks[-1]["type"] == "text":
            blocks[-1]["text"] += s
        else:
            blocks.append({"type": "text", "text": s})

    def visit(node: ET.Element) -> None:
        for child in list(node):
            tag = child.tag.split("}")[-1]
            if tag == "t" and child.text:
                push_text(child.text)
                continue
            if tag == "oMath":
                blocks.append({"type": "math_omml", "omml": ET.tostring(child, encoding="unicode")})
                continue
            if tag == "imagedata":  # VML preview inside w:object / w:pict -> inline formula
                rid = child.get(f"{{{R}}}id")
                target = rels.get(rid, "")
                if target:
                    blocks.append({"type": "formula", "media": target.split("/")[-1]})
                continue
            if tag == "blip":  # DrawingML -> standalone figure
                rid = child.get(f"{{{R}}}embed")
                target = rels.get(rid, "")
                if target:
                    blocks.append({"type": "figure", "media": target.split("/")[-1]})
                continue
            visit(child)

    visit(p)
    return blocks


def table_blocks(tbl: ET.Element, rels: dict[str, str]) -> dict:
    rows = []
    for tr in tbl.findall("w:tr", NS):
        cells = []
        for tc in tr.findall("w:tc", NS):
            cell_blocks: list[dict] = []
            for p in tc.findall(".//w:p", NS):
                for b in para_blocks(p, rels):
                    if b["type"] == "text":
                        if cell_blocks and cell_blocks[-1]["type"] == "text":
                            cell_blocks[-1]["text"] += b["text"]
                        else:
                            cell_blocks.append({"type": "text", "text": b["text"]})
                    elif b["type"] == "math_omml":
                        cell_blocks.append(b)
                    else:
                        cell_blocks.append(b)
            if not cell_blocks:
                cells.append("")
            elif all(block.get("type") == "text" for block in cell_blocks):
                cells.append(" ".join(str(block.get("text") or "") for block in cell_blocks).strip())
            else:
                cells.append(cell_blocks)
        rows.append(cells)
    return {"type": "table", "rows": rows}


def blocks_text(blocks: list[dict]) -> str:
    out = []

    def cell_text(cell: object) -> str:
        if isinstance(cell, str):
            return cell
        if isinstance(cell, list):
            return blocks_text([block for block in cell if isinstance(block, dict)])
        if isinstance(cell, dict):
            return blocks_text([cell])
        return str(cell)

    for b in blocks:
        if b["type"] == "text":
            out.append(b["text"])
        elif b["type"] == "table":
            out.append(" ".join(cell_text(c) for row in b["rows"] for c in row))
        elif b["type"] == "math_omml":
            out.append(str(b.get("latex") or b.get("text") or ""))
        elif b["type"] in {"formula", "figure"}:
            out.append("图")
    return "".join(out)


def image_size_from_bytes(name: str, data: bytes) -> tuple[int, int] | None:
    lower = name.lower()
    try:
        if lower.endswith(".png") and data[:8] == b"\x89PNG\r\n\x1a\n":
            return unpack(">II", data[16:24])
        if lower.endswith(".gif") and data[:6] in {b"GIF87a", b"GIF89a"}:
            return unpack("<HH", data[6:10])
        if lower.endswith((".jpg", ".jpeg")) and data[:2] == b"\xff\xd8":
            idx = 2
            while idx + 9 < len(data):
                if data[idx] != 0xFF:
                    idx += 1
                    continue
                marker = data[idx + 1]
                if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                    height, width = unpack(">HH", data[idx + 5:idx + 9])
                    return width, height
                if marker in {0xD8, 0xD9}:
                    idx += 2
                    continue
                seg_len = unpack(">H", data[idx + 2:idx + 4])[0]
                idx += 2 + seg_len
    except Exception:
        return None
    return None


def probe_docx_media_stats(zf: zipfile.ZipFile) -> dict[str, int]:
    media_names = [name for name in zf.namelist() if name.startswith("word/media/") and not name.endswith("/")]
    large = 0
    non_page = 0
    for name in media_names:
        size = image_size_from_bytes(name, zf.read(name))
        if not size:
            non_page += 1
            continue
        _, height = size
        if height > 1400:
            large += 1
        else:
            non_page += 1
    return {
        "media_count": len(media_names),
        "large_page_media_count": large,
        "non_page_media_count": non_page,
    }


def is_scan_fallback_probe_row(row: dict[str, object]) -> bool:
    return bool(
        int(row.get("text_char_count") or 0) < 200
        and int(row.get("media_count") or 0) > 0
        and int(row.get("large_page_media_count") or 0) == int(row.get("media_count") or 0)
        and int(row.get("non_page_media_count") or 0) == 0
    )


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_source_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_group: dict[str, dict[str, object]] = {}
    role_order = {"question_source": 0, "analysis": 1, "answer_key": 2, "answer_only": 3, "unknown": 4}
    for record in records:
        group_key = str(record.get("group_key") or "")
        group = by_group.setdefault(
            group_key,
            {"group_key": group_key, "unique_sources": [], "duplicate_sources": [], "_hashes": set()},
        )
        sha1 = str(record.get("sha1") or "")
        if sha1 and sha1 in group["_hashes"]:
            group["duplicate_sources"].append(record)
            continue
        if sha1:
            group["_hashes"].add(sha1)
        group["unique_sources"].append(record)
    groups = []
    for group in by_group.values():
        unique_sources = group["unique_sources"]
        has_plain_unknown = any(
            row.get("role") == "unknown" and not row.get("is_duplicate_name") for row in unique_sources
        )
        if has_plain_unknown:
            for row in unique_sources:
                if row.get("role") == "unknown" and row.get("is_duplicate_name"):
                    row["role"] = "answer_key"
                    row["role_inferred_reason"] = "duplicate_unknown_not_used_as_question_source"
        group["unique_sources"].sort(key=lambda row: (role_order.get(str(row.get("role")), 99), str(row.get("file_name", ""))))
        group["duplicate_sources"].sort(key=lambda row: str(row.get("file_name", "")))
        group.pop("_hashes", None)
        groups.append(group)
    groups.sort(key=lambda row: str(row.get("group_key", "")))
    return groups


def paragraph_block_sequence(zf: zipfile.ZipFile, rels: dict[str, str]) -> list[list[dict]]:
    root = ET.fromstring(zf.read("word/document.xml").decode("utf-8"))
    body = root.find("w:body", NS)
    if body is None:
        return []

    paragraphs: list[list[dict]] = []
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            paragraphs.append(para_blocks(child, rels))
        elif tag == "tbl":
            paragraphs.append([table_blocks(child, rels)])
    return paragraphs


def probe_docx_format(docx: Path) -> dict:
    row = {
        "path": str(docx),
        "file_name": docx.name,
        "suffix": docx.suffix.lower(),
        "status": "ok",
        "omml_count": 0,
        "ole_count": 0,
        "drawing_count": 0,
        "media_count": 0,
        "answer_marker_count": 0,
        "answer_position": "none",
        "error": "",
    }
    try:
        with zipfile.ZipFile(docx) as zf:
            document_xml = zf.read("word/document.xml").decode("utf-8")
            root = ET.fromstring(document_xml)
            row["omml_count"] = sum(1 for node in root.iter() if node.tag.split("}")[-1] == "oMath")
            row["ole_count"] = sum(1 for node in root.iter() if node.tag.split("}")[-1] == "imagedata")
            row["drawing_count"] = sum(1 for node in root.iter() if node.tag.split("}")[-1] == "drawing")
            row.update(probe_docx_media_stats(zf))
            rels = load_rels(zf)
            paragraphs = paragraph_block_sequence(zf, rels)
            texts = [blocks_text(blocks).strip() for blocks in paragraphs]
            row["text_char_count"] = sum(len(text) for text in texts)
            answer_indices = [idx for idx, text in enumerate(texts) if ANSWER.search(text)]
            q_indices = [idx for idx, text in enumerate(texts) if Q_START.match(text)]
            row["q_start_count"] = len(q_indices)
            row["question_prompt_count"] = sum(1 for text in texts if looks_like_question_prompt(text))
            row["answer_fragment_line_count"] = sum(1 for text in texts if looks_like_answer_enumeration(text))
            row["option_marker_count"] = sum(len(OPTION.findall(text)) for text in texts)
            row["answer_marker_count"] = sum(1 for text in texts if ANSWER.search(text))
            if answer_indices:
                first_answer = answer_indices[0]
                row["answer_position"] = "inline" if any(idx > first_answer for idx in q_indices) else "trailing"
            row["route"] = "scan_fallback" if is_scan_fallback_probe_row(row) else "docx_native"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def parse_numbered_answers(text: str) -> dict[int, str]:
    text = ANSWER.sub("", text).strip()
    matches = list(NUMBERED_ANSWER.finditer(text))
    if len(matches) < 2:
        return {}
    answers: dict[int, str] = {}
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = text[match.end():end].strip(" \t\r\n，,；;。")
        if value:
            answers[int(match.group(1))] = value
    return answers


def answer_para(value: str) -> dict:
    return {"para": [{"type": "text", "text": f"【答案】{value}"}]}


def normalize_answer_value(text: str) -> str:
    text = ANSWER_MARKER_ANY.sub("", text or "")
    text = re.sub(r"^[\s:：]+", "", text)
    text = text.strip(" \t\r\n。；;，,")
    return text


def normalize_choice_answer(text: str) -> str:
    value = normalize_answer_value(text).upper()
    value = re.sub(r"^(?:故选|选|答案为)\s*[:：]?", "", value)
    value = re.sub(r"[\s、,，;；.。()（）]+", "", value)
    return value


def split_blocks_at_inline_answer(blocks: list[dict]) -> tuple[list[dict], list[dict], bool]:
    before: list[dict] = []
    after: list[dict] = []
    found = False
    for block in blocks:
        if not found and block.get("type") == "text":
            text = block.get("text", "")
            match = INLINE_ANSWER_MARKER.search(text)
            if match:
                prefix = text[: match.start()]
                suffix = text[match.start():]
                if prefix:
                    before.append({**block, "text": prefix})
                if suffix:
                    after.append({**block, "text": suffix})
                found = True
                continue
        (after if found else before).append(block)
    return before, after, found


def table_cell_plain_text(cell: object) -> str:
    if isinstance(cell, str):
        return cell.strip()
    if isinstance(cell, list):
        return blocks_text([block for block in cell if isinstance(block, dict)]).strip()
    if isinstance(cell, dict):
        return blocks_text([cell]).strip()
    return str(cell).strip()


def parse_answer_table_rows(rows: list[list[object]]) -> dict[int, str]:
    answer_map: dict[int, str] = {}
    if len(rows) < 2:
        return answer_map

    text_rows = [[table_cell_plain_text(cell) for cell in row] for row in rows]
    q_row: list[str] | None = None
    a_row: list[str] | None = None
    for row in text_rows:
        if not row:
            continue
        first = row[0].strip()
        if "题号" in first:
            q_row = row
        elif "答案" in first:
            a_row = row

    if q_row and a_row:
        limit = min(len(q_row), len(a_row))
        for idx in range(1, limit):
            q_match = re.fullmatch(r"\d{1,3}", q_row[idx].strip())
            answer = a_row[idx].strip()
            if q_match and answer:
                answer_map[int(q_match.group(0))] = answer
        return answer_map

    if text_rows[0] and len(text_rows[0]) >= 2 and "题号" in text_rows[0][0] and "答案" in text_rows[0][1]:
        for row in text_rows[1:]:
            if len(row) < 2:
                continue
            q_match = re.fullmatch(r"\d{1,3}", row[0].strip())
            answer = row[1].strip()
            if q_match and answer:
                answer_map[int(q_match.group(0))] = answer
    return answer_map


def extract_reference_answer_table_map(paragraphs: list[list[dict]]) -> dict[int, str]:
    answer_map: dict[int, str] = {}
    seen_reference_answer = False
    for blocks in paragraphs:
        text = blocks_text(blocks).strip()
        if text and ("参考答案" in text or ANSWER.match(text)):
            seen_reference_answer = True
        for block in blocks:
            if block.get("type") != "table":
                continue
            table_map = parse_answer_table_rows(block.get("rows", []))
            if table_map and seen_reference_answer:
                answer_map.update(table_map)
    return answer_map


def extract_section_answer_key_map(paragraphs: list[list[dict]]) -> dict[int, str]:
    section_index = 0
    current_section: int | None = None
    collecting = False
    chunks: list[str] = []
    answer_map: dict[int, str] = {}

    def flush() -> None:
        nonlocal chunks, collecting
        if current_section is not None and chunks:
            answer_map[current_section] = "\n".join(chunk for chunk in chunks if chunk).strip()
        chunks = []
        collecting = False

    for blocks in paragraphs:
        text = blocks_text(blocks).strip()
        if not text:
            continue
        if SECTION.match(text):
            flush()
            section_index += 1
            current_section = section_index
            continue
        if ANSWER.match(text):
            flush()
            collecting = True
            chunks = [text]
            if current_section is None:
                section_index += 1
                current_section = section_index
            continue
        if collecting:
            chunks.append(text)
    flush()
    return answer_map


def looks_like_answer_enumeration(text: str) -> bool:
    match = Q_START.match(text)
    if not match:
        return False
    rest = text[match.end():].strip()
    if not rest:
        return False
    if any(word in rest for word in QUESTION_CUE_WORDS):
        return False
    if rest.endswith("题") and len(rest) <= 6:
        return False
    if re.fullmatch(r"[\sA-Da-d0-9一二三四五六七八九十.、．,，;；:：()（）+\-*/=<>\[\]【】NACOHSKBrClFeMnCuZnMgAlSiPbPTe△→⇌↓↑·℃%]+", rest):
        return True
    if SCORE_MARKER.search(rest) and len(rest) <= 24:
        return True
    if len(rest) <= 16 and sum("\u4e00" <= ch <= "\u9fff" for ch in rest) <= 2:
        return True
    return False


def looks_like_numbered_answer_line(text: str) -> bool:
    return bool(NUMBERED_ANSWER_LINE.match(text))


def looks_like_numbered_analysis_line(text: str) -> bool:
    return bool(NUMBERED_ANALYSIS_LINE.match(text))


def looks_like_numbered_line_with_analysis_marker(text: str) -> bool:
    match = Q_START.match(text)
    if not match:
        return False
    return bool(ANALYSIS_MARKER_ANY.search(text[match.end():]))


def looks_like_question_prompt(text: str) -> bool:
    match = Q_START.match(text)
    if not match:
        return False
    rest = text[match.end():].strip()
    if not rest:
        return False
    if "____" in rest or "______" in rest or "（   ）" in rest or "(   )" in rest:
        return True
    if any(word in rest for word in QUESTION_CUE_WORDS):
        return True
    chinese_count = sum("\u4e00" <= ch <= "\u9fff" for ch in rest)
    return chinese_count >= 8 and len(rest) >= 18 and not ANSWER_MARKER_ANY.search(rest)


def looks_like_short_question_label(text: str) -> bool:
    match = Q_START.match(text)
    if not match:
        return False
    rest = text[match.end():].strip()
    return bool(rest.endswith("题") and len(rest) <= 8)


def looks_like_exam_instruction(text: str) -> bool:
    match = Q_START.match(text)
    if not match:
        return False
    rest = text[match.end():].strip()
    return any(word in rest for word in EXAM_INSTRUCTION_WORDS)


def apply_trailing_answer_remap(questions: list[dict]) -> bool:
    by_num: dict[int, list[dict]] = {}
    for q in questions:
        by_num.setdefault(q["q_num"], []).append(q)

    changed = False
    for q in questions:
        answer_text = blocks_text([b for para in q["answer_blocks"] for b in para["para"]])
        mapping = parse_numbered_answers(answer_text)
        if not mapping:
            continue
        for q_num, value in mapping.items():
            targets = by_num.get(q_num, [])
            if len(targets) == 1:
                targets[0]["answer_blocks"] = [answer_para(value)]
                changed = True
    return changed


def segment(paragraphs: list[list[dict]]) -> list[dict]:
    """Split document paragraphs into questions with stem/answer/analysis zones."""
    questions: list[dict] = []
    cur: dict | None = None
    zone = "stem"
    answer_key_mode = False
    section_title = ""
    section_num: int | None = None

    for blocks in paragraphs:
        text = blocks_text(blocks).strip()
        if not text and not any(b["type"] in ("formula", "figure") for b in blocks):
            continue
        section_match = SECTION.match(text)
        if section_match:
            if answer_key_mode and cur is not None:
                cur[f"{zone}_blocks"].append({"para": blocks})
                continue
            section_title = text[:60]
            section_num = chinese_section_num(section_match.group(1))
            continue
        m = Q_START.match(text)
        if cur is None and section_num is None and looks_like_exam_instruction(text):
            continue
        candidate_q_num = int(m.group(1)) if m else None
        numbered_answer_line = looks_like_numbered_answer_line(text)
        numbered_analysis_line = looks_like_numbered_analysis_line(text)
        numbered_analysis_fragment = looks_like_numbered_line_with_analysis_marker(text)
        suppress_in_answer_zone = bool(
            m
            and (zone in {"answer", "analysis"} or answer_key_mode)
            and (
                looks_like_answer_enumeration(text)
                or numbered_analysis_line
                or numbered_analysis_fragment
                or not text[m.end():].strip()
                or (
                    cur is not None
                    and candidate_q_num is not None
                    and candidate_q_num <= cur.get("q_num", -1)
                    and not looks_like_question_prompt(text)
                )
            )
        )
        suppress_weak_answer_fragment = bool(
            m and cur is not None and zone == "stem" and looks_like_answer_enumeration(text)
        )
        should_start_question = bool(
            m
            and not numbered_answer_line
            and not numbered_analysis_line
            and not suppress_in_answer_zone
            and not suppress_weak_answer_fragment
            and not (answer_key_mode and not (looks_like_question_prompt(text) or looks_like_short_question_label(text)))
            and (zone != "stem" or cur is None or candidate_q_num != cur.get("q_num"))
        )
        if should_start_question:
            # a new question starts
            if cur:
                questions.append(cur)
            q_num = candidate_q_num
            cur = {
                "q_num": q_num,
                "section_num": section_num,
                "question_id": f"{section_num or 0}-{q_num}",
                "section": section_title,
                "stem_blocks": [],
                "answer_blocks": [],
                "analysis_blocks": [],
            }
            zone = "stem"
            answer_key_mode = False
        if cur is None:
            continue  # front matter before question 1
        if numbered_answer_line:
            zone = "answer"
            cur["answer_blocks"].append({"para": blocks})
            continue
        if numbered_analysis_line:
            zone = "analysis"
            cur["analysis_blocks"].append({"para": blocks})
            continue
        if suppress_weak_answer_fragment:
            zone = "answer"
            cur["answer_blocks"].append({"para": blocks})
            continue
        if ANSWER.match(text):
            zone = "answer"
            if PLAIN_ANSWER_KEY_MARKER.match(text):
                answer_key_mode = True
        elif ANALYSIS.match(text):
            zone = "analysis"
        else:
            stem_part, answer_part, has_inline_answer = split_blocks_at_inline_answer(blocks)
            if has_inline_answer:
                if stem_part:
                    cur[f"{zone}_blocks"].append({"para": stem_part})
                if answer_part:
                    zone = "answer"
                    cur["answer_blocks"].append({"para": answer_part})
                continue
        cur[f"{zone}_blocks"].append({"para": blocks})
    if cur:
        questions.append(cur)
    apply_trailing_answer_remap(questions)
    return questions


def wmf_to_png(wmf: Path, out_dir: Path) -> Path | None:
    """Convert one WMF to autocropped PNG for preview."""
    try:
        subprocess.run(
            [SOFFICE, "--headless", "--convert-to", "png", str(wmf), "--outdir", str(out_dir)],
            capture_output=True, timeout=60, check=True,
        )
        png = out_dir / (wmf.stem + ".png")
        if not png.exists():
            return None
        from PIL import Image, ImageChops
        with Image.open(png) as im:
            rgb = im.convert("RGB")
            bg = Image.new("RGB", rgb.size, (255, 255, 255))
            diff = ImageChops.difference(rgb, bg)
            bbox = diff.getbbox()
            if bbox:
                pad = 6
                bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                        min(rgb.width, bbox[2] + pad), min(rgb.height, bbox[3] + pad))
                rgb.crop(bbox).save(png)
        return png
    except Exception:
        return None


def render_preview(questions: list[dict], assets: Path, out: Path, limit: int = 8) -> None:
    converted: dict[str, str] = {}

    def block_html(b: dict) -> str:
        if b["type"] == "text":
            return b["text"]
        if b["type"] == "table":
            def cell_html(cell: object) -> str:
                if isinstance(cell, str):
                    return html.escape(cell)
                if isinstance(cell, list):
                    return "".join(block_html(block) for block in cell if isinstance(block, dict))
                if isinstance(cell, dict):
                    return block_html(cell)
                return html.escape(str(cell))

            rows = "".join(
                "<tr>" + "".join(f"<td style='border:1px solid #999;padding:3px 8px'>{cell_html(c)}</td>" for c in row) + "</tr>"
                for row in b["rows"])
            return f"<table style='border-collapse:collapse;margin:6px 0'>{rows}</table>"
        if b["type"] == "math_omml":
            return f"<span style='font-family:monospace;background:#f5f5f5;padding:1px 4px'>OMML:{html.escape(b['omml'][:80])}</span>"
        media = b["media"]
        src = assets / media
        if media.lower().endswith(".wmf"):
            if media not in converted:
                png = wmf_to_png(src, assets)
                converted[media] = png.name if png else ""
            name = converted[media]
            if not name:
                return f"<span style='color:#a22'>[公式转换失败:{media}]</span>"
            style = "vertical-align:middle;max-height:34px" if b["type"] == "formula" else "max-width:520px;display:block;margin:8px 0"
            return f"<img src='assets/{name}' style='{style}'>"
        style = "vertical-align:middle;max-height:34px" if b["type"] == "formula" else "max-width:520px;display:block;margin:8px 0"
        return f"<img src='assets/{media}' style='{style}'>"

    cards = []
    for q in questions[:limit]:
        stem_html = ""
        for para in q["stem_blocks"]:
            stem_html += "<p style='margin:7px 0'>" + "".join(block_html(b) for b in para["para"]) + "</p>"
        ans_text = blocks_text([b for para in q["answer_blocks"] for b in para["para"]])[:60]
        cards.append(
            f"<div style='border:1px solid #ddd;border-radius:8px;padding:14px 18px;margin:14px 0;max-width:760px'>"
            f"<div style='color:#888;font-size:12px'>第{q['q_num']}题 · {q['section']}</div>"
            f"{stem_html}"
            f"<div style='color:#2a7;font-size:12px;margin-top:6px'>[答案已分离,不展示给学生: {ans_text}...]</div>"
            f"</div>")
    preview_html = ("<!doctype html><meta charset='utf-8'><title>WS1 抽取预览</title>"
            "<body style='font-family:-apple-system,PingFang SC,serif;font-size:16px;line-height:1.7;padding:24px'>"
            "<h2>WS1 docx 原生抽取 · 印刷级预览(公式为原生矢量转图,位置=原文位置)</h2>"
            + "".join(cards) + "</body>")
    out.write_text(preview_html, encoding="utf-8")


def parse_docx_model(docx: Path) -> dict[str, object]:
    with zipfile.ZipFile(docx) as zf:
        rels = load_rels(zf)
        paragraphs = paragraph_block_sequence(zf, rels)
        questions = segment(paragraphs)
        referenced = {
            b["media"]
            for q in questions
            for zone in ("stem_blocks", "answer_blocks", "analysis_blocks")
            for para in q[zone]
            for b in para["para"]
            if b["type"] in ("formula", "figure") and b.get("media")
        }
    return {
        "paragraphs": paragraphs,
        "questions": questions,
        "referenced_media": sorted(referenced),
    }


def copy_docx_media(docx: Path, media_names: list[str], assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx) as zf:
        for name in media_names:
            try:
                data = zf.read(f"word/media/{name}")
            except KeyError:
                continue
            (assets_dir / name).write_bytes(data)


def copy_docx_media_with_prefix(docx: Path, media_names: list[str], assets_dir: Path, prefix: str = "") -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx) as zf:
        for name in media_names:
            try:
                data = zf.read(f"word/media/{name}")
            except KeyError:
                continue
            (assets_dir / f"{prefix}{name}").write_bytes(data)


def source_media_prefix(source_path: str) -> str:
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8]
    return f"ans_{digest}_"


def clone_blocks_with_media_prefix(paragraphs: list[dict], prefix: str = "") -> list[dict]:
    cloned = json.loads(json.dumps(paragraphs, ensure_ascii=False))
    if not prefix:
        return cloned
    for para in cloned:
        for block in para.get("para", []):
            if block.get("type") in {"formula", "figure"} and block.get("media"):
                block["media"] = f"{prefix}{block['media']}"
    return cloned


def q_summary_from_questions(docx: Path, out_dir: Path, questions: list[dict], referenced_media: list[str]) -> dict:
    q_with_answer = sum(1 for q in questions if q["answer_blocks"])
    q_with_figure = sum(1 for q in questions if any(
        b["type"] == "figure" for para in q["stem_blocks"] for b in para["para"]))
    formula_refs = sum(1 for q in questions for para in q["stem_blocks"] for b in para["para"] if b["type"] == "formula")
    omml_refs = sum(1 for q in questions for zone in ("stem_blocks", "answer_blocks", "analysis_blocks")
                    for para in q[zone] for b in para["para"] if b["type"] == "math_omml")
    nums = [q["q_num"] for q in questions]
    large_jumps = [
        {"from": nums[idx - 1], "to": nums[idx], "index": idx}
        for idx in range(1, len(nums))
        if abs(nums[idx] - nums[idx - 1]) > 5 and nums[idx] != 1
    ]
    return {
        "paper": docx.name,
        "docx_path": str(docx),
        "out_dir": str(out_dir),
        "questions": len(questions),
        "q_nums": nums,
        "with_answer": q_with_answer,
        "with_figure_in_stem": q_with_figure,
        "formula_refs_in_stems": formula_refs,
        "omml_blocks": omml_refs,
        "media_extracted": len(referenced_media),
        "numbering_monotonic": nums == sorted(nums),
        "large_number_jumps": large_jumps,
        "compound_question_ids": [q["question_id"] for q in questions],
    }


def extract_docx(docx: Path, out_root: Path, render_preview_file: bool = True) -> dict:
    out_dir = out_root / safe_paper_dir_name(docx)
    assets = out_dir / "assets"
    model = parse_docx_model(docx)
    questions = model["questions"]
    referenced = model["referenced_media"]
    copy_docx_media(docx, referenced, assets)

    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    summary = q_summary_from_questions(docx, out_dir, questions, referenced)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if render_preview_file:
        render_preview(questions, assets, out_dir / "preview.html")
    return summary


def run_format_probe(root: Path, out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    converted_dir = out_root / "_converted_docx"
    probe_path = out_root / "format_probe.jsonl"
    rows: list[dict] = []
    for source in iter_source_files(root):
        row = {"original_path": str(source), "converted_from_doc": source.suffix.lower() == ".doc"}
        docx = source
        if source.suffix.lower() == ".doc":
            converted, error = convert_doc_to_docx(source, converted_dir)
            row["conversion_error"] = error
            if not converted:
                row.update(
                    {
                        "path": "",
                        "file_name": source.name,
                        "suffix": ".doc",
                        "status": "conversion_error",
                        "omml_count": 0,
                        "ole_count": 0,
                        "drawing_count": 0,
                        "media_count": 0,
                        "answer_marker_count": 0,
                        "answer_position": "none",
                        "error": error,
                    }
                )
                rows.append(row)
                continue
            docx = converted
            row["converted_docx_path"] = str(converted)
        row.update(probe_docx_format(docx))
        rows.append(row)
    with probe_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source_root": str(root),
        "out_root": str(out_root),
        "total_sources": len(rows),
        "ok": sum(1 for row in rows if row.get("status") == "ok"),
        "errors": sum(1 for row in rows if row.get("status") != "ok"),
        "doc_sources": sum(1 for row in rows if row.get("converted_from_doc")),
        "docx_sources": sum(1 for row in rows if not row.get("converted_from_doc")),
        "omml_total": sum(int(row.get("omml_count") or 0) for row in rows),
        "ole_total": sum(int(row.get("ole_count") or 0) for row in rows),
        "drawing_total": sum(int(row.get("drawing_count") or 0) for row in rows),
        "media_total": sum(int(row.get("media_count") or 0) for row in rows),
        "answer_position": dict(Counter(str(row.get("answer_position")) for row in rows)),
        "format_probe": str(probe_path),
    }
    (out_root / "format_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_batch_report(out_root: Path, probe_summary: dict | None, summaries: list[dict], failures: list[dict]) -> Path:
    report = out_root / "BATCH_REPORT.md"
    total_sources = len(summaries) + len(failures)
    success_rate = (len(summaries) / total_sources * 100) if total_sources else 0
    q_counts = [int(summary.get("questions", 0)) for summary in summaries]
    anomalies = [
        summary
        for summary in summaries
        if int(summary.get("questions", 0)) < 5 or summary.get("large_number_jumps")
    ]
    total_questions = sum(q_counts)
    total_omml = sum(int(summary.get("omml_blocks", 0)) for summary in summaries)
    total_formula = sum(int(summary.get("formula_refs_in_stems", 0)) for summary in summaries)
    total_figures = sum(int(summary.get("with_figure_in_stem", 0)) for summary in summaries)
    lines = [
        "# WS1 Batch Report",
        "",
        "## Summary",
        "",
        f"- Total sources attempted: {total_sources}",
        f"- Successful question outputs: {len(summaries)} ({success_rate:.1f}%)",
        f"- Failures: {len(failures)}",
        f"- Total extracted questions: {total_questions}",
        f"- Question count min/median/max: {min(q_counts) if q_counts else 0}/{sorted(q_counts)[len(q_counts)//2] if q_counts else 0}/{max(q_counts) if q_counts else 0}",
        f"- Papers with <5 questions or large numbering jumps: {len(anomalies)}",
        "",
        "## Formula And Figure Stats",
        "",
        f"- OMML blocks preserved: {total_omml}",
        f"- OLE/WMF formula refs in stems: {total_formula}",
        f"- Questions with figure in stem: {total_figures}",
    ]
    if probe_summary:
        lines.extend(
            [
                f"- Probe OMML total: {probe_summary.get('omml_total')}",
                f"- Probe OLE total: {probe_summary.get('ole_total')}",
                f"- Probe drawing total: {probe_summary.get('drawing_total')}",
                f"- Probe media total: {probe_summary.get('media_total')}",
                f"- Probe answer positions: {json.dumps(probe_summary.get('answer_position', {}), ensure_ascii=False)}",
            ]
        )
    lines.extend(["", "## Per-Paper Question Counts", ""])
    for summary in sorted(summaries, key=lambda row: row.get("paper", "")):
        lines.append(f"- {summary['paper']}: {summary['questions']} questions, answers {summary['with_answer']}, OMML {summary['omml_blocks']}, media {summary['media_extracted']}")
    lines.extend(["", "## Anomaly Review Queue (first 20)", ""])
    for summary in anomalies[:20]:
        reason = []
        if int(summary.get("questions", 0)) < 5:
            reason.append("questions<5")
        if summary.get("large_number_jumps"):
            reason.append("large_number_jump")
        lines.append(f"- {summary['paper']}: {summary['questions']} questions; {', '.join(reason)}; out `{summary['out_dir']}`")
    lines.extend(["", "## Failures", ""])
    for failure in failures[:50]:
        lines.append(f"- {failure['source']}: {failure['error']}")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_batch_extract(root: Path, out_root: Path, preview_limit: int = 5) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    converted_dir = out_root / "_converted_docx"
    summaries: list[dict] = []
    failures: list[dict] = []
    previews = 0
    for source in iter_source_files(root):
        docx = source
        if source.suffix.lower() == ".doc":
            converted, error = convert_doc_to_docx(source, converted_dir)
            if not converted:
                failures.append({"source": str(source), "error": error})
                continue
            docx = converted
        try:
            render_preview_file = previews < preview_limit
            summary = extract_docx(docx, out_root, render_preview_file=render_preview_file)
            summary["original_path"] = str(source)
            if render_preview_file:
                previews += 1
            summaries.append(summary)
        except Exception as exc:
            failures.append({"source": str(source), "error": f"{type(exc).__name__}: {exc}"})
    probe_summary_path = out_root / "format_probe_summary.json"
    probe_summary = None
    if probe_summary_path.exists():
        probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    report = write_batch_report(out_root, probe_summary, summaries, failures)
    run_summary = {
        "total_sources": len(summaries) + len(failures),
        "success": len(summaries),
        "failures": len(failures),
        "success_rate": (len(summaries) / (len(summaries) + len(failures)) if summaries or failures else 0),
        "preview_count": previews,
        "batch_report": str(report),
    }
    (out_root / "batch_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_summary


def build_source_records(
    root: Path,
    out_root: Path,
    source_paths: list[Path] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    converted_dir = out_root / "_converted_docx"
    records: list[dict[str, object]] = []
    conversion_failures: list[dict[str, object]] = []
    sources = source_paths if source_paths is not None else iter_source_files(root)
    for source in sources:
        docx = source
        converted_from_doc = source.suffix.lower() == ".doc"
        conversion_error = ""
        if converted_from_doc:
            converted, conversion_error = convert_doc_to_docx(source, converted_dir)
            if not converted:
                conversion_failures.append({"source": str(source), "error": conversion_error})
                continue
            docx = converted
        role_info = classify_source_name(source.name)
        probe = probe_docx_format(docx)
        record = {
            **role_info,
            **probe,
            "path": str(docx),
            "original_path": str(source),
            "converted_from_doc": converted_from_doc,
            "converted_docx_path": str(docx) if converted_from_doc else "",
            "conversion_error": conversion_error,
            "sha1": file_sha1(docx),
        }
        refine_source_role_with_probe(record)
        records.append(record)
    return records, conversion_failures


def choose_group_source(group: dict[str, object], roles: set[str]) -> dict[str, object] | None:
    sources = [
        row
        for row in group.get("unique_sources", [])
        if row.get("role") in roles and row.get("status") == "ok" and row.get("route") != "scan_fallback"
    ]
    if not sources:
        return None
    role_rank = {"question_source": 0, "analysis": 1, "answer_key": 2, "unknown": 3}
    return sorted(
        sources,
        key=lambda row: (
            role_rank.get(str(row.get("role")), 9),
            -int(row.get("text_char_count") or 0),
            str(row.get("file_name") or ""),
        ),
    )[0]


def select_answer_sources(group: dict[str, object], question_source: dict[str, object]) -> list[dict[str, object]]:
    return [
        row
        for row in group.get("unique_sources", [])
        if row is not question_source
        and row.get("role") in {"analysis", "answer_key", "answer_only"}
        and row.get("status") == "ok"
        and row.get("route") != "scan_fallback"
    ]


def has_extractable_question_source(group: dict[str, object]) -> bool:
    return any(
        row.get("role") in {"question_source", "analysis", "unknown"}
        and row.get("status") == "ok"
        and row.get("route") != "scan_fallback"
        for row in group.get("unique_sources", [])
    )


def group_should_scan_fallback(group: dict[str, object]) -> bool:
    question_sources = [
        row
        for row in group.get("unique_sources", [])
        if row.get("role") == "question_source" and row.get("status") == "ok"
    ]
    if not question_sources:
        return False
    return all(row.get("route") == "scan_fallback" for row in question_sources)


def clone_question_with_source(q: dict, source_role: str, source_path: str, group_key: str) -> dict:
    cloned = json.loads(json.dumps(q, ensure_ascii=False))
    cloned["source_role"] = source_role
    cloned["source_path"] = source_path
    cloned["group_key"] = group_key
    return cloned


def merge_answer_questions_into_targets(
    targets: list[dict],
    answer_questions: list[dict],
    source_path: str,
    source_role: str,
    media_prefix: str = "",
) -> int:
    by_question_id: dict[str, list[dict]] = {}
    by_q_num: dict[int, list[dict]] = {}
    for target in targets:
        question_id = str(target.get("question_id") or "")
        if question_id:
            by_question_id.setdefault(question_id, []).append(target)
        q_num = target.get("q_num")
        if q_num is not None:
            by_q_num.setdefault(int(q_num), []).append(target)

    assigned = 0
    for answer_q in answer_questions:
        if not answer_q.get("answer_blocks"):
            continue

        target = None
        question_id = str(answer_q.get("question_id") or "")
        if question_id and len(by_question_id.get(question_id, [])) == 1:
            target = by_question_id[question_id][0]
        else:
            q_num = answer_q.get("q_num")
            if q_num is None:
                continue
            q_num_targets = by_q_num.get(int(q_num), [])
            if len(q_num_targets) == 1:
                target = q_num_targets[0]

        if target is None or target.get("answer_blocks"):
            continue
        target["answer_blocks"] = clone_blocks_with_media_prefix(answer_q["answer_blocks"], media_prefix)
        target["analysis_blocks"] = clone_blocks_with_media_prefix(answer_q.get("analysis_blocks", []), media_prefix)
        target["answer_source_path"] = source_path
        target["answer_source_role"] = source_role
        assigned += 1
    return assigned


def merge_answers_into_questions(
    questions: list[dict],
    answer_sources: list[dict[str, object]],
    assets_dir: Path | None = None,
) -> dict[str, object]:
    by_section_num: dict[int, list[dict]] = {}
    by_q_num: dict[int, list[dict]] = {}
    for q in questions:
        if q.get("section_num") is not None:
            by_section_num.setdefault(int(q["section_num"]), []).append(q)
        if q.get("q_num") is not None:
            by_q_num.setdefault(int(q["q_num"]), []).append(q)

    assigned = 0
    sources_used: list[str] = []
    for source in answer_sources:
        try:
            model = parse_docx_model(Path(str(source["path"])))
        except Exception:
            continue
        source_used = False
        source_path = str(source.get("original_path") or source.get("path"))
        source_role = str(source.get("role"))
        media_prefix = source_media_prefix(source_path)
        section_map = extract_section_answer_key_map(model["paragraphs"])
        if section_map:
            for section_num, answer_text in section_map.items():
                targets = by_section_num.get(section_num, [])
                if len(targets) == 1 and not targets[0]["answer_blocks"]:
                    targets[0]["answer_blocks"] = [answer_para(answer_text.replace("【答案】", "", 1).strip())]
                    targets[0]["answer_source_path"] = source_path
                    targets[0]["answer_source_role"] = source_role
                    assigned += 1
                    source_used = True

        table_map = extract_reference_answer_table_map(model["paragraphs"])
        if table_map:
            for q_num, answer_text in table_map.items():
                targets = by_q_num.get(q_num, [])
                if len(targets) == 1 and not targets[0]["answer_blocks"]:
                    targets[0]["answer_blocks"] = [answer_para(answer_text)]
                    targets[0]["answer_source_path"] = source_path
                    targets[0]["answer_source_role"] = source_role
                    assigned += 1
                    source_used = True

        source_assigned = merge_answer_questions_into_targets(
            questions,
            model["questions"],
            source_path,
            source_role,
            media_prefix,
        )
        assigned += source_assigned
        source_used = source_used or source_assigned > 0
        if source_used:
            sources_used.append(source_path)
            if assets_dir is not None:
                copy_docx_media_with_prefix(
                    Path(str(source["path"])),
                    list(model.get("referenced_media", [])),
                    assets_dir,
                    media_prefix,
                )
    return {"assigned": assigned, "sources_used": sources_used}


def looks_like_choice_question(question: dict) -> bool:
    text = question_text(question)
    section = str(question.get("section") or "")
    return bool("选择题" in section or "（   ）" in text or "(   )" in text or len(set(OPTION.findall(text))) >= 3)


STRICT_OPTION = re.compile(r"(?<![A-Za-z0-9])([A-D])\s*[.．、]")
NESTED_SUBQUESTION = re.compile(r"[（(]\s*\d{1,2}\s*[)）]")


def is_multi_part_nested_question(question: dict) -> bool:
    text = question_text(question, ("stem_blocks",))
    markers = NESTED_SUBQUESTION.findall(text)
    return len(markers) >= 2


def is_strict_choice_question(question: dict) -> bool:
    if is_multi_part_nested_question(question):
        return False
    text = question_text(question, ("stem_blocks",))
    options = STRICT_OPTION.findall(text)
    return set(options) >= {"A", "B", "C", "D"}


def answer_text(question: dict) -> str:
    blocks = []
    for para in question.get("answer_blocks", []):
        if isinstance(para, dict):
            blocks.extend(para.get("para", []))
        elif isinstance(para, list):
            blocks.extend(para)
    return blocks_text(blocks)


def answer_has_non_text_content(question: dict) -> bool:
    for para in question.get("answer_blocks", []):
        blocks = para.get("para", []) if isinstance(para, dict) else para
        for block in blocks:
            if block.get("type") in {"formula", "figure", "math_omml", "table"}:
                return True
    return False


def clear_answer(question: dict) -> None:
    question["answer_blocks"] = []


def add_quality_flag(question: dict, flag: str) -> None:
    flags = question.setdefault("quality_flags", [])
    if flag not in flags:
        flags.append(flag)


def apply_quality_gates(questions: list[dict]) -> dict[str, int]:
    report = {
        "answer_type_mismatch": 0,
        "empty_answer": 0,
        "stem_contaminated": 0,
    }
    for question in questions:
        stem = question_text(question, ("stem_blocks",))
        if any(pattern.search(stem) for pattern in STEM_CONTAMINATION_PATTERNS):
            add_quality_flag(question, "stem_contaminated")
            report["stem_contaminated"] += 1

        if question.get("answer_blocks"):
            normalized_raw = normalize_answer_value(answer_text(question))
            if not normalized_raw and not answer_has_non_text_content(question):
                add_quality_flag(question, "empty_answer")
                clear_answer(question)
                report["empty_answer"] += 1
                continue
    return report


def apply_final_dataset_answer_type_gate(questions: list[dict]) -> dict[str, int]:
    report = {
        "strict_choice_questions": 0,
        "normalized_choice_answers": 0,
        "cleared_choice_answer_mismatches": 0,
        "flagged_empty_choice_mismatches": 0,
        "already_flagged_choice_mismatches": 0,
    }
    for question in questions:
        if not is_strict_choice_question(question):
            continue
        report["strict_choice_questions"] += 1
        had_answer = bool(question.get("answer_blocks"))
        flags = set(question.get("quality_flags") or [])
        if "answer_type_mismatch" in flags and not question.get("answer_blocks"):
            report["already_flagged_choice_mismatches"] += 1
            continue
        normalized_choice = normalize_choice_answer(answer_text(question))
        if CHOICE_ANSWER_RE.fullmatch(normalized_choice):
            if question.get("answer_blocks") != [answer_para(normalized_choice)]:
                question["answer_blocks"] = [answer_para(normalized_choice)]
                report["normalized_choice_answers"] += 1
        else:
            if "answer_type_mismatch" in flags and not question.get("answer_blocks"):
                report["already_flagged_choice_mismatches"] += 1
            else:
                add_quality_flag(question, "answer_type_mismatch")
                clear_answer(question)
                if had_answer:
                    report["cleared_choice_answer_mismatches"] += 1
                else:
                    report["flagged_empty_choice_mismatches"] += 1
    return report


def stem_prefix(question: dict, length: int = 80) -> str:
    text = question_text(question, ("stem_blocks",))
    text = re.sub(r"\s+", "", text)
    return text[:length]


def assign_stable_question_ids(questions: list[dict], group_key: str) -> None:
    for question in questions:
        local_id = str(question.get("question_id") or "")
        if local_id and "local_question_id" not in question:
            question["local_question_id"] = local_id
        raw = "|".join(
            [
                str(group_key),
                str(question.get("section") or question.get("section_num") or ""),
                str(question.get("q_num") or ""),
                stem_prefix(question),
            ]
        )
        question["question_id_base"] = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        question["question_id"] = question["question_id_base"]
    ensure_unique_question_ids(questions)


def ensure_unique_question_ids(questions: list[dict]) -> dict[str, int]:
    seen: dict[str, int] = {}
    collisions = 0
    for idx, question in enumerate(questions):
        question_id = str(question.get("question_id") or "")
        occurrence = seen.get(question_id, 0)
        seen[question_id] = occurrence + 1
        if occurrence == 0:
            continue
        collisions += 1
        raw = "|".join(
            [
                question_id,
                str(occurrence),
                str(idx),
                str(question.get("local_question_id") or ""),
                hashlib.sha1(question_text(question, ("stem_blocks",)).encode("utf-8")).hexdigest(),
            ]
        )
        question["question_id_collision_disambiguated"] = True
        question["question_id"] = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        while question["question_id"] in seen:
            raw += "|retry"
            question["question_id"] = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        seen[question["question_id"]] = 1
    return {"question_id_disambiguated": collisions}


def dedupe_questions_by_stem(questions: list[dict]) -> tuple[list[dict], dict[str, int]]:
    seen: dict[str, dict] = {}
    deduped: list[dict] = []
    duplicate_count = 0
    for question in questions:
        stem = question_text(question, ("stem_blocks",))
        normalized = re.sub(r"\s+", "", stem)
        key = hashlib.sha1(normalized.encode("utf-8")).hexdigest() if len(normalized) >= 30 else ""
        if key and key in seen:
            primary = seen[key]
            source_refs = primary.setdefault("source_refs", [])
            source_refs.append(
                {
                    "group_key": question.get("group_key"),
                    "source_path": question.get("source_path"),
                    "local_question_id": question.get("local_question_id") or question.get("question_id"),
                    "q_num": question.get("q_num"),
                }
            )
            add_quality_flag(question, "duplicate_stem_merged")
            duplicate_count += 1
            continue
        if key:
            question["stem_hash"] = key
            seen[key] = question
        deduped.append(question)
    return deduped, {"duplicate_stem_merged": duplicate_count}


def extract_group_v3(group: dict[str, object], out_root: Path, render_preview_file: bool) -> dict[str, object]:
    group_key = str(group["group_key"])
    question_source = choose_group_source(group, {"question_source"})
    route = "paired_question_source" if question_source else "single_document"
    if not question_source:
        question_source = choose_group_source(group, {"analysis", "unknown"})
    if not question_source:
        raise ValueError("no usable docx_native source in group")

    docx = Path(str(question_source["path"]))
    out_dir = out_root / safe_paper_dir_name(Path(group_key + ".docx"))
    assets = out_dir / "assets"
    model = parse_docx_model(docx)
    questions = [
        clone_question_with_source(
            q,
            str(question_source.get("role")),
            str(question_source.get("original_path") or question_source.get("path")),
            group_key,
        )
        for q in model["questions"]
    ]
    referenced = list(model["referenced_media"])

    answer_sources = select_answer_sources(group, question_source)
    copy_docx_media(docx, referenced, assets)
    merge_result = merge_answers_into_questions(questions, answer_sources, assets)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    summary = q_summary_from_questions(docx, out_dir, questions, referenced)
    summary.update(
        {
            "group_key": group_key,
            "route": route,
            "question_source": str(question_source.get("original_path") or question_source.get("path")),
            "answer_sources": merge_result["sources_used"],
            "answer_assigned_from_sources": merge_result["assigned"],
            "answer_coverage": (summary["with_answer"] / summary["questions"]) if summary["questions"] else 0,
            "unique_source_count": len(group.get("unique_sources", [])),
            "duplicate_source_count": len(group.get("duplicate_sources", [])),
            "source_roles": dict(Counter(str(row.get("role")) for row in group.get("unique_sources", []))),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "sources.json").write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding="utf-8")
    if render_preview_file:
        render_preview(questions, assets, out_dir / "preview.html")
    return summary


def load_questions_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_questions_jsonl(path: Path, questions: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")


def recompute_summary_counts(summary: dict[str, object], questions: list[dict]) -> None:
    summary["questions"] = len(questions)
    summary["q_nums"] = [q.get("q_num") for q in questions]
    summary["with_answer"] = sum(1 for q in questions if q.get("answer_blocks"))
    summary["answer_coverage"] = (summary["with_answer"] / summary["questions"]) if summary["questions"] else 0
    summary["quality_flags"] = dict(
        Counter(flag for q in questions for flag in q.get("quality_flags", []))
    )
    ids = [str(q.get("question_id") or "") for q in questions]
    summary["question_id_collisions"] = len(ids) - len(set(ids))


def postprocess_group_v4(out_dir: Path, summary: dict[str, object]) -> dict[str, int]:
    questions_path = out_dir / "questions.jsonl"
    questions = load_questions_jsonl(questions_path)
    assign_stable_question_ids(questions, str(summary.get("group_key") or summary.get("paper") or ""))
    gate_report = apply_quality_gates(questions)
    write_questions_jsonl(questions_path, questions)
    recompute_summary_counts(summary, questions)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate_report


def write_global_dedup_outputs(out_root: Path, summaries: list[dict[str, object]]) -> dict[str, object]:
    all_rows: list[dict] = []
    for summary in summaries:
        out_dir = Path(str(summary.get("out_dir") or ""))
        questions_path = out_dir / "questions.jsonl"
        for question in load_questions_jsonl(questions_path):
            all_rows.append(question)
    deduped, dedupe_report = dedupe_questions_by_stem(all_rows)
    global_id_report = ensure_unique_question_ids(deduped)
    write_questions_jsonl(out_root / "questions_deduped.jsonl", deduped)
    ids = [str(q.get("question_id") or "") for q in all_rows]
    deduped_ids = [str(q.get("question_id") or "") for q in deduped]
    report = {
        "total_questions_before_dedupe": len(all_rows),
        "total_questions_after_dedupe": len(deduped),
        "duplicate_stem_merged": dedupe_report["duplicate_stem_merged"],
        "question_id_disambiguated_after_dedupe": global_id_report["question_id_disambiguated"],
        "question_id_collisions_before_dedupe": len(ids) - len(set(ids)),
        "question_id_collisions_after_dedupe": len(deduped_ids) - len(set(deduped_ids)),
        "deduped_questions_path": str(out_root / "questions_deduped.jsonl"),
    }
    (out_root / "global_dedup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def final_answer_type_gate_file_stats(questions: list[dict]) -> dict[str, int]:
    strict_choice_questions = sum(1 for question in questions if is_strict_choice_question(question))
    flagged_before = sum(1 for question in questions if "answer_type_mismatch" in set(question.get("quality_flags") or []))
    flagged_empty_before = sum(
        1
        for question in questions
        if "answer_type_mismatch" in set(question.get("quality_flags") or []) and not question.get("answer_blocks")
    )
    with_answer_before = sum(1 for question in questions if question.get("answer_blocks"))
    gate_report = apply_final_dataset_answer_type_gate(questions)
    flagged_after = sum(1 for question in questions if "answer_type_mismatch" in set(question.get("quality_flags") or []))
    flagged_empty_after = sum(
        1
        for question in questions
        if "answer_type_mismatch" in set(question.get("quality_flags") or []) and not question.get("answer_blocks")
    )
    with_answer_after = sum(1 for question in questions if question.get("answer_blocks"))
    stats = {
        "rows": len(questions),
        "strict_choice_questions_before": strict_choice_questions,
        "with_answer_before": with_answer_before,
        "with_answer_after": with_answer_after,
        "answer_type_mismatch_before": flagged_before,
        "answer_type_mismatch_after": flagged_after,
        "answer_type_mismatch_empty_before": flagged_empty_before,
        "answer_type_mismatch_empty_after": flagged_empty_after,
        **gate_report,
    }
    stats["answers_cleared_by_final_gate"] = stats["cleared_choice_answer_mismatches"]
    return stats


def write_final_answer_type_gate_report(batch_root: Path, report: dict[str, object]) -> None:
    (batch_root / "final_answer_type_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    batch_summary_path = batch_root / "batch_summary.json"
    if batch_summary_path.exists():
        batch_summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
        batch_summary["final_answer_type_gate"] = report
        batch_summary_path.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_final_dataset_answer_type_gate_to_batch(
    batch_root: Path,
    include_group_files: bool = True,
    include_deduped: bool = True,
    write_report: bool = True,
) -> dict[str, object]:
    batch_root = Path(batch_root)
    file_reports: list[dict[str, object]] = []
    totals: Counter[str] = Counter()
    if include_group_files:
        for questions_path in sorted(batch_root.glob("*/questions.jsonl")):
            if any(part.startswith("golden_") for part in questions_path.parts):
                continue
            questions = load_questions_jsonl(questions_path)
            stats = final_answer_type_gate_file_stats(questions)
            if stats["rows"] == 0:
                continue
            write_questions_jsonl(questions_path, questions)
            summary_path = questions_path.parent / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                recompute_summary_counts(summary, questions)
                summary["final_answer_type_gate"] = stats
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            file_report = {
                "path": str(questions_path),
                "paper_dir": questions_path.parent.name,
                **stats,
            }
            file_reports.append(file_report)
            totals.update(stats)

    deduped_path = batch_root / "questions_deduped.jsonl"
    deduped_report = None
    if include_deduped and deduped_path.exists():
        deduped_questions = load_questions_jsonl(deduped_path)
        deduped_report = final_answer_type_gate_file_stats(deduped_questions)
        write_questions_jsonl(deduped_path, deduped_questions)
        totals.update({f"deduped_{key}": value for key, value in deduped_report.items()})

    report = {
        "batch_root": str(batch_root),
        "questions_files_processed": len(file_reports),
        "totals": dict(totals),
        "deduped_report": deduped_report,
        "files_with_new_clears": [
            row
            for row in file_reports
            if int(row.get("answers_cleared_by_final_gate") or 0) > 0
        ],
        "file_reports": file_reports,
    }
    if write_report:
        write_final_answer_type_gate_report(batch_root, report)
    return report


def write_v3_batch_report(
    out_root: Path,
    records: list[dict[str, object]],
    groups: list[dict[str, object]],
    summaries: list[dict[str, object]],
    failures: list[dict[str, object]],
    scan_rows: list[dict[str, object]],
    preview_count: int,
) -> Path:
    report = out_root / "BATCH_REPORT.md"
    q_counts = [int(s.get("questions") or 0) for s in summaries]
    answer_coverage = [float(s.get("answer_coverage") or 0) for s in summaries if int(s.get("questions") or 0) > 0]
    answer_80 = sum(1 for value in answer_coverage if value >= 0.8)
    answer_capable = [
        s
        for s in summaries
        if int(s.get("questions") or 0) > 0
        and any(role in (s.get("source_roles") or {}) for role in ("analysis", "answer_key", "answer_only"))
    ]
    answer_capable_80 = sum(1 for s in answer_capable if float(s.get("answer_coverage") or 0) >= 0.8)
    answer_source_used = [s for s in summaries if int(s.get("questions") or 0) > 0 and s.get("answer_sources")]
    answer_source_used_80 = sum(1 for s in answer_source_used if float(s.get("answer_coverage") or 0) >= 0.8)
    buckets = {
        "0": 0,
        "0-20%": 0,
        "20-50%": 0,
        "50-80%": 0,
        ">=80%": 0,
    }
    for value in answer_coverage:
        if value == 0:
            buckets["0"] += 1
        elif value < 0.2:
            buckets["0-20%"] += 1
        elif value < 0.5:
            buckets["20-50%"] += 1
        elif value < 0.8:
            buckets["50-80%"] += 1
        else:
            buckets[">=80%"] += 1
    anomalies = [
        s for s in summaries if int(s.get("questions") or 0) < 5 or s.get("large_number_jumps")
    ]
    paired = [s for s in summaries if s.get("route") == "paired_question_source"]
    role_counts = Counter(str(r.get("role")) for r in records)
    lines = [
        "# WS1 v3 Batch Report",
        "",
        "## Summary",
        "",
        f"- Source files scanned: {len(records)}",
        f"- Source roles: {json.dumps(dict(role_counts), ensure_ascii=False)}",
        f"- Unique paper groups after filename clustering: {len(groups)}",
        f"- Groups extracted: {len(summaries)}",
        f"- Paired original/analysis strategy groups: {len(paired)}",
        f"- Scan fallback sources: {len(scan_rows)}",
        f"- Failures: {len(failures)}",
        f"- Preview files generated: {preview_count}",
        f"- Independent extracted questions: {sum(q_counts)}",
        f"- Question count min/median/max: {min(q_counts) if q_counts else 0}/{sorted(q_counts)[len(q_counts)//2] if q_counts else 0}/{max(q_counts) if q_counts else 0}",
        f"- Papers with <5 questions or large numbering jumps: {len(anomalies)}",
        "",
        "## Answer Coverage",
        "",
        f"- Papers with questions: {len(answer_coverage)}",
        f"- Papers with >=80% answer coverage: {answer_80}/{len(answer_coverage)} ({(answer_80 / len(answer_coverage) * 100) if answer_coverage else 0:.1f}%)",
        f"- Answer-capable papers with >=80% answer coverage: {answer_capable_80}/{len(answer_capable)} ({(answer_capable_80 / len(answer_capable) * 100) if answer_capable else 0:.1f}%)",
        f"- Papers with answer sources used: {len(answer_source_used)}",
        f"- Papers with answer sources used and >=80% answer coverage: {answer_source_used_80}/{len(answer_source_used)} ({(answer_source_used_80 / len(answer_source_used) * 100) if answer_source_used else 0:.1f}%)",
        f"- Coverage min/median/max: {min(answer_coverage) if answer_coverage else 0:.2f}/{sorted(answer_coverage)[len(answer_coverage)//2] if answer_coverage else 0:.2f}/{max(answer_coverage) if answer_coverage else 0:.2f}",
        f"- Coverage buckets: {json.dumps(buckets, ensure_ascii=False)}",
        "",
        "## Formula And Figure Stats",
        "",
        f"- OMML blocks preserved: {sum(int(s.get('omml_blocks') or 0) for s in summaries)}",
        f"- OLE/WMF formula refs in stems: {sum(int(s.get('formula_refs_in_stems') or 0) for s in summaries)}",
        f"- Questions with figure in stem: {sum(int(s.get('with_figure_in_stem') or 0) for s in summaries)}",
        "",
        "## Anomaly Review Queue (first 20)",
        "",
    ]
    for summary in anomalies[:20]:
        reasons = []
        if int(summary.get("questions") or 0) < 5:
            reasons.append("questions<5")
        if summary.get("large_number_jumps"):
            reasons.append("large_number_jump")
        lines.append(f"- {summary['group_key']}: {summary['questions']} questions, coverage {summary['answer_coverage']:.2f}; {', '.join(reasons)}; out `{summary['out_dir']}`")
    low_answer_capable = [
        s
        for s in answer_capable
        if float(s.get("answer_coverage") or 0) < 0.8
    ]
    low_answer_capable.sort(key=lambda s: (float(s.get("answer_coverage") or 0), str(s.get("group_key") or "")))
    lines.extend(["", "## Low Answer Coverage Review Queue (first 30)", ""])
    for summary in low_answer_capable[:30]:
        lines.append(
            f"- {summary['group_key']}: {summary['questions']} questions, answers {summary['with_answer']}, "
            f"coverage {summary['answer_coverage']:.2f}, roles {json.dumps(summary.get('source_roles', {}), ensure_ascii=False)}; "
            f"out `{summary['out_dir']}`"
        )
    lines.extend(["", "## Scan Fallback Sources", ""])
    for row in scan_rows[:80]:
        lines.append(f"- {row.get('file_name')}: text_chars={row.get('text_char_count')} media={row.get('media_count')} path=`{row.get('original_path')}`")
    lines.extend(["", "## Failures", ""])
    for failure in failures[:80]:
        lines.append(f"- {failure.get('group_key') or failure.get('source')}: {failure.get('error')}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def append_v4_report_section(
    report: Path,
    gate_totals: dict[str, int],
    dedupe_report: dict[str, object],
    missing_group_dirs: list[str],
) -> None:
    lines = report.read_text(encoding="utf-8").splitlines()
    lines.extend(
        [
            "",
            "## v4 Audit Gates",
            "",
            f"- answer_type_mismatch cleared: {gate_totals.get('answer_type_mismatch', 0)}",
            f"- empty_answer cleared: {gate_totals.get('empty_answer', 0)}",
            f"- stem_contaminated marked: {gate_totals.get('stem_contaminated', 0)}",
            f"- Global question_id collisions before dedupe: {dedupe_report.get('question_id_collisions_before_dedupe', 0)}",
            f"- Global question_id collisions after dedupe: {dedupe_report.get('question_id_collisions_after_dedupe', 0)}",
            f"- Duplicate stems merged: {dedupe_report.get('duplicate_stem_merged', 0)}",
            f"- Deduped questions: {dedupe_report.get('total_questions_after_dedupe', 0)} / {dedupe_report.get('total_questions_before_dedupe', 0)}",
            f"- Group dirs without questions.jsonl: {len(missing_group_dirs)}",
            "",
            "### Missing questions.jsonl Group Dirs",
            "",
        ]
    )
    if missing_group_dirs:
        lines.extend(f"- {name}" for name in missing_group_dirs[:50])
    else:
        lines.append("- none")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch_extract_v3(root: Path, out_root: Path, preview_limit: int = 5) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    records, conversion_failures = build_source_records(root, out_root)
    groups = group_source_records(records)
    scan_rows = [row for row in records if row.get("route") == "scan_fallback"]
    scan_group_keys = {str(group.get("group_key")) for group in groups if group_should_scan_fallback(group)}
    with (out_root / "scan_fallback_list.jsonl").open("w", encoding="utf-8") as f:
        for row in scan_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_root / "source_groups.json").open("w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    with (out_root / "source_records.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = list(conversion_failures)
    previews = 0
    for group in groups:
        if str(group.get("group_key")) in scan_group_keys:
            continue
        usable = [
            row
            for row in group.get("unique_sources", [])
            if row.get("status") == "ok" and row.get("route") != "scan_fallback"
        ]
        if not usable or not has_extractable_question_source(group):
            continue
        try:
            render_preview_file = previews < preview_limit
            summary = extract_group_v3(group, out_root, render_preview_file=render_preview_file)
            if render_preview_file:
                previews += 1
            summaries.append(summary)
        except Exception as exc:
            failures.append({"group_key": group.get("group_key"), "error": f"{type(exc).__name__}: {exc}"})

    report = write_v3_batch_report(out_root, records, groups, summaries, failures, scan_rows, previews)
    run_summary = {
        "source_files": len(records),
        "groups": len(groups),
        "extracted_groups": len(summaries),
        "scan_fallback_sources": len(scan_rows),
        "failures": len(failures),
        "preview_count": previews,
        "independent_questions": sum(int(s.get("questions") or 0) for s in summaries),
        "answer_coverage_ge_80_count": sum(
            1 for s in summaries if int(s.get("questions") or 0) > 0 and float(s.get("answer_coverage") or 0) >= 0.8
        ),
        "batch_report": str(report),
    }
    (out_root / "batch_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_summary


def read_source_list(path: Path | None, source_root: Path | None = None) -> list[Path] | None:
    if path is None:
        return None
    sources = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            candidate = Path(value)
            if not candidate.is_absolute() and not candidate.exists():
                alternatives = []
                if source_root is not None:
                    alternatives.append(source_root / candidate)
                alternatives.append(Path.cwd().parent / candidate)
                alternatives.append(path.parent / candidate)
                for alternative in alternatives:
                    if alternative.exists():
                        candidate = alternative
                        break
            sources.append(candidate)
    return sources


def run_batch_extract_v4(
    root: Path,
    out_root: Path,
    preview_limit: int = 5,
    source_paths: list[Path] | None = None,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    records, conversion_failures = build_source_records(root, out_root, source_paths)
    groups = group_source_records(records)
    scan_rows = [row for row in records if row.get("route") == "scan_fallback"]
    scan_group_keys = {str(group.get("group_key")) for group in groups if group_should_scan_fallback(group)}
    with (out_root / "scan_fallback_list.jsonl").open("w", encoding="utf-8") as f:
        for row in scan_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_root / "source_groups.json").open("w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    with (out_root / "source_records.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = list(conversion_failures)
    previews = 0
    gate_totals: Counter[str] = Counter()
    for group in groups:
        if str(group.get("group_key")) in scan_group_keys:
            continue
        usable = [
            row
            for row in group.get("unique_sources", [])
            if row.get("status") == "ok" and row.get("route") != "scan_fallback"
        ]
        if not usable or not has_extractable_question_source(group):
            continue
        try:
            render_preview_file = previews < preview_limit
            summary = extract_group_v3(group, out_root, render_preview_file=render_preview_file)
            group_out_dir = Path(str(summary["out_dir"]))
            gate_totals.update(postprocess_group_v4(group_out_dir, summary))
            if render_preview_file:
                render_preview(load_questions_jsonl(group_out_dir / "questions.jsonl"), group_out_dir / "assets", group_out_dir / "preview.html")
                previews += 1
            summaries.append(summary)
        except Exception as exc:
            failures.append({"group_key": group.get("group_key"), "error": f"{type(exc).__name__}: {exc}"})

    final_gate_report = apply_final_dataset_answer_type_gate_to_batch(
        out_root,
        include_group_files=True,
        include_deduped=False,
        write_report=False,
    )
    dedupe_report = write_global_dedup_outputs(out_root, summaries)
    deduped_gate_report = apply_final_dataset_answer_type_gate_to_batch(
        out_root,
        include_group_files=False,
        include_deduped=True,
        write_report=False,
    )
    if deduped_gate_report.get("deduped_report") is not None:
        final_gate_report["deduped_report"] = deduped_gate_report["deduped_report"]
        final_gate_report["totals"].update(
            {
                f"deduped_{key}": value
                for key, value in deduped_gate_report["deduped_report"].items()
            }
        )
    write_final_answer_type_gate_report(out_root, final_gate_report)
    missing_group_dirs = sorted(
        path.name
        for path in out_root.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and path.name != "golden_candidates"
        and not (path / "questions.jsonl").exists()
    )
    report = write_v3_batch_report(out_root, records, groups, summaries, failures, scan_rows, previews)
    append_v4_report_section(report, dict(gate_totals), dedupe_report, missing_group_dirs)
    run_summary = {
        "source_files": len(records),
        "groups": len(groups),
        "extracted_groups": len(summaries),
        "scan_fallback_sources": len(scan_rows),
        "failures": len(failures),
        "preview_count": previews,
        "independent_questions": sum(int(s.get("questions") or 0) for s in summaries),
        "answer_coverage_ge_80_count": sum(
            1 for s in summaries if int(s.get("questions") or 0) > 0 and float(s.get("answer_coverage") or 0) >= 0.8
        ),
        "gate_totals": dict(gate_totals),
        "final_answer_type_gate": final_gate_report,
        "global_dedup": dedupe_report,
        "missing_questions_jsonl_group_dirs": missing_group_dirs,
        "batch_report": str(report),
    }
    (out_root / "batch_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_summary


GOLDEN_CANDIDATE_QUOTAS = {
    "equation": 15,
    "electron_structure": 10,
    "table": 5,
    "device": 10,
    "curve": 10,
    "process": 5,
    "crystal": 5,
}

GOLDEN_CATEGORY_LABELS = {
    "equation": "化学方程式题",
    "electron_structure": "电子式/结构式题",
    "table": "表格题",
    "device": "装置图题",
    "curve": "曲线图题",
    "process": "工艺流程题",
    "crystal": "晶胞题",
}

GOLDEN_CATEGORY_KEYWORDS = {
    "equation": ("方程式", "离子方程式", "化学方程式", "反应方程式", "反应式", "配平"),
    "electron_structure": ("电子式", "结构式", "结构简式", "键线式", "同分异构", "官能团", "有机物"),
    "table": ("如表", "下表", "表中", "表格", "数据见表", "见表"),
    "device": ("装置", "仪器", "烧瓶", "分液漏斗", "冷凝管", "洗气瓶", "导管", "干燥管", "滴定管"),
    "curve": ("曲线", "图像", "图象", "坐标", "滴定曲线", "pH", "速率", "平衡常数", "随时间"),
    "process": ("流程", "工艺", "工业", "制备", "生产", "转化流程", "合成路线"),
    "crystal": ("晶胞", "晶体", "晶格", "配位数", "立方晶胞", "晶体结构"),
}

GOLDEN_ROUND2_RETIRED_IDS = {
    "table_001",
    "curve_007",
    "equation_013",
    "equation_009",
    "electron_structure_010",
    "process_001",
    "table_004",
    "crystal_001",
    "crystal_002",
    "curve_008",
}

GOLDEN_ROUND2_RETAINED_NOTES = {
    "crystal_003": "needs_manual_transcription",
    "device_006": "needs_manual_transcription",
}

GOLDEN_ROUND2_BACKUP_QUOTAS = {
    "equation": 2,
    "table": 2,
    "curve": 2,
    "crystal": 2,
    "process": 1,
    "electron_structure": 1,
}

GOLDEN_ROUND2_FORMAL_QUOTAS = {
    category: GOLDEN_CANDIDATE_QUOTAS[category] - GOLDEN_ROUND2_BACKUP_QUOTAS.get(category, 0)
    for category in GOLDEN_CANDIDATE_QUOTAS
}


def iter_question_blocks(question: dict, zones: tuple[str, ...] = ("stem_blocks", "answer_blocks", "analysis_blocks")):
    for zone in zones:
        for para in question.get(zone, []):
            blocks = para.get("para", []) if isinstance(para, dict) else para
            for block in blocks:
                yield block


def question_text(question: dict, zones: tuple[str, ...] = ("stem_blocks",)) -> str:
    return blocks_text(list(iter_question_blocks(question, zones)))


def question_media_refs(question: dict, zones: tuple[str, ...] = ("stem_blocks",)) -> list[str]:
    refs = {
        str(block.get("media"))
        for block in iter_question_blocks(question, zones)
        if block.get("type") in {"figure", "formula"} and block.get("media")
    }
    return sorted(refs)


def all_question_media_refs(question: dict) -> list[str]:
    return question_media_refs(question, ("stem_blocks", "answer_blocks", "analysis_blocks"))


def question_contains(question: dict, zones: tuple[str, ...] = ("stem_blocks",)) -> list[str]:
    contains = set()
    for block in iter_question_blocks(question, zones):
        block_type = block.get("type")
        if block_type == "figure":
            contains.add("figure")
        elif block_type == "formula":
            contains.add("formula")
        elif block_type == "math_omml":
            contains.add("omml")
        elif block_type == "table":
            contains.add("table")
    return sorted(contains)


def has_stem_formula_or_figure(question: dict) -> bool:
    return any(
        block.get("type") in {"figure", "formula", "math_omml"}
        for block in iter_question_blocks(question, ("stem_blocks",))
    )


def has_answer_blocks(question: dict) -> bool:
    return bool(question.get("answer_blocks") and normalize_answer_value(answer_text(question)))


def has_table_block(question: dict) -> bool:
    return any(block.get("type") == "table" for block in iter_question_blocks(question, ("stem_blocks",)))


def is_golden_candidate_eligible(question: dict) -> bool:
    flags = set(question.get("quality_flags") or [])
    return bool(
        question.get("source_role") == "question_source"
        and has_answer_blocks(question)
        and has_stem_formula_or_figure(question)
        and not (flags & {"answer_type_mismatch", "empty_answer", "stem_contaminated"})
    )


def classify_golden_candidate_categories(question: dict) -> list[str]:
    text = question_text(question)
    categories: list[str] = []
    for category in GOLDEN_CANDIDATE_QUOTAS:
        keywords = GOLDEN_CATEGORY_KEYWORDS[category]
        if category == "table" and has_table_block(question):
            categories.append(category)
            continue
        if any(keyword in text for keyword in keywords):
            categories.append(category)
    return categories


def golden_candidate_key(summary: dict, question: dict) -> str:
    raw = "|".join(
        [
            str(summary.get("group_key") or question.get("group_key") or ""),
            str(question.get("source_path") or ""),
            str(question.get("question_id") or ""),
            str(question.get("q_num") or ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def golden_candidate_content_key(summary: dict, question: dict) -> str:
    raw = "|".join(
        [
            str(summary.get("group_key") or question.get("group_key") or ""),
            str(question.get("local_question_id") or question.get("question_id") or ""),
            stem_prefix(question, 120),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def golden_candidate_score(question: dict, category: str) -> tuple[int, int, int, str]:
    contains = set(question_contains(question))
    text = question_text(question)
    score = 0
    if "figure" in contains:
        score -= 30
    if "formula" in contains or "omml" in contains:
        score -= 15
    if category == "table" and has_table_block(question):
        score -= 40
    if category in {"device", "curve", "process", "crystal"} and "figure" in contains:
        score -= 20
    return (score, len(text), int(question.get("q_num") or 0), str(question.get("question_id") or ""))


def load_golden_candidate_pool(batch_root: Path) -> dict[str, list[dict[str, object]]]:
    pools: dict[str, list[dict[str, object]]] = {category: [] for category in GOLDEN_CANDIDATE_QUOTAS}
    for summary_path in sorted(batch_root.glob("*/summary.json")):
        paper_dir = summary_path.parent
        if paper_dir.name == "golden_candidates":
            continue
        questions_path = paper_dir / "questions.jsonl"
        if not questions_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for line_num, line in enumerate(questions_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            question = json.loads(line)
            if not is_golden_candidate_eligible(question):
                continue
            categories = classify_golden_candidate_categories(question)
            if not categories:
                continue
            key = golden_candidate_key(summary, question)
            row = {
                "key": key,
                "content_key": golden_candidate_content_key(summary, question),
                "summary": summary,
                "question": question,
                "paper_dir": str(paper_dir),
                "questions_path": str(questions_path),
                "line_num": line_num,
                "contains": question_contains(question),
                "asset_refs": all_question_media_refs(question),
            }
            for category in categories:
                pools[category].append({**row, "category": category, "score": golden_candidate_score(question, category)})
    for category, rows in pools.items():
        rows.sort(
            key=lambda row: (
                row["score"],
                str(row["summary"].get("group_key") or ""),
                int(row["question"].get("q_num") or 0),
                str(row["question"].get("question_id") or ""),
            )
        )
    return pools


def missing_assets_for_row(row: dict[str, object]) -> list[str]:
    source_assets = Path(str(row["paper_dir"])) / "assets"
    missing = []
    for asset in row.get("asset_refs", []):
        if not (source_assets / str(asset)).exists():
            missing.append(str(asset))
    return missing


def largest_figure_asset_size(row: dict[str, object]) -> int:
    source_assets = Path(str(row["paper_dir"])) / "assets"
    question = row["question"]
    sizes = []
    for asset in question_media_refs(question, ("stem_blocks",)):
        path = source_assets / asset
        if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            sizes.append(path.stat().st_size)
    return max(sizes) if sizes else 0


def row_passes_golden_hard_gates(row: dict[str, object]) -> tuple[bool, list[str]]:
    blockers = []
    if missing_assets_for_row(row):
        blockers.append("missing_assets")
    if row.get("category") == "curve" and largest_figure_asset_size(row) <= 20 * 1024:
        blockers.append("curve_figure_too_small")
    return not blockers, blockers


def export_golden_candidate(row: dict[str, object], candidate_dir: Path, candidate_id: str) -> dict[str, object]:
    question = json.loads(json.dumps(row["question"], ensure_ascii=False))
    assets_dir = candidate_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    source_assets = Path(str(row["paper_dir"])) / "assets"
    copied_assets: list[str] = []
    missing_assets: list[str] = []
    for asset in row["asset_refs"]:
        source = source_assets / str(asset)
        if source.exists():
            shutil.copy2(source, assets_dir / source.name)
            copied_assets.append(source.name)
        else:
            missing_assets.append(str(asset))
    question["golden_candidate"] = {
        "candidate_id": candidate_id,
        "category": row["category"],
        "category_label": GOLDEN_CATEGORY_LABELS[str(row["category"])],
        "source_out_dir": row["paper_dir"],
        "source_questions_file": row["questions_path"],
        "source_line_num": row["line_num"],
        "contains": row["contains"],
        "asset_files": copied_assets,
        "missing_assets": missing_assets,
        "content_key": row.get("content_key"),
    }
    (candidate_dir / "question.json").write_text(json.dumps(question, ensure_ascii=False, indent=2), encoding="utf-8")
    return question["golden_candidate"]


def select_golden_candidates(
    batch_root: Path,
    out_dir: Path | None = None,
    quotas: dict[str, int] | None = None,
) -> dict[str, object]:
    quotas = quotas or dict(GOLDEN_CANDIDATE_QUOTAS)
    out_dir = out_dir or (batch_root / "golden_candidates")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = load_golden_candidate_pool(batch_root)
    used_keys: set[str] = set()
    group_counts: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    missing_asset_candidates = 0
    missing_asset_rejected = 0
    hard_gate_rejections: Counter[str] = Counter()

    for category, quota in quotas.items():
        picked = 0
        while picked < quota:
            available = [row for row in pools.get(category, []) if str(row["key"]) not in used_keys]
            if not available:
                break
            available.sort(
                key=lambda row: (
                    group_counts[str(row["summary"].get("group_key") or "")],
                    row["score"],
                    str(row["summary"].get("group_key") or ""),
                    int(row["question"].get("q_num") or 0),
                    str(row["question"].get("question_id") or ""),
                )
            )
            row = None
            for candidate in available:
                passed, blockers = row_passes_golden_hard_gates(candidate)
                if passed:
                    row = candidate
                    break
                if "missing_assets" in blockers:
                    missing_asset_rejected += 1
                hard_gate_rejections.update(blockers)
                used_keys.add(str(candidate["key"]))
            if row is None:
                break
            picked += 1
            category_counts[category] += 1
            used_keys.add(str(row["key"]))
            group_key = str(row["summary"].get("group_key") or row["question"].get("group_key") or "")
            group_counts[group_key] += 1
            candidate_id = f"{category}_{picked:03d}"
            candidate_dir = out_dir / candidate_id
            metadata = export_golden_candidate(row, candidate_dir, candidate_id)
            if metadata["missing_assets"]:
                missing_asset_candidates += 1
            selected.append({**row, "candidate_id": candidate_id, "metadata": metadata})

    lines = [
        "# WS1 v3 Golden Candidates",
        "",
        "| 编号 | 来源卷 | 题型 | 题号 | 含什么块 | 资产 | question.json |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in selected:
        metadata = row["metadata"]
        question = row["question"]
        summary = row["summary"]
        candidate_id = str(row["candidate_id"])
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate_id,
                    str(summary.get("group_key") or question.get("group_key") or "").replace("|", "/"),
                    GOLDEN_CATEGORY_LABELS[str(row["category"])],
                    str(question.get("question_id") or question.get("q_num") or ""),
                    ", ".join(metadata["contains"]) or "-",
                    ", ".join(metadata["asset_files"]) or "-",
                    f"`{candidate_id}/question.json`",
                ]
            )
            + " |"
        )
    (out_dir / "candidates_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "batch_root": str(batch_root),
        "out_dir": str(out_dir),
        "selected_total": len(selected),
        "quotas": quotas,
        "category_counts": dict(category_counts),
        "pool_counts": {category: len(rows) for category, rows in pools.items()},
        "missing_asset_candidates": missing_asset_candidates,
        "missing_asset_rejected": missing_asset_rejected,
        "hard_gate_rejections": dict(hard_gate_rejections),
        "index": str(out_dir / "candidates_index.md"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_candidate_question(candidate_dir: Path) -> dict | None:
    path = candidate_dir / "question.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def retired_content_keys_from_dir(golden_dir: Path) -> set[str]:
    retired_keys = set()
    for candidate_id in GOLDEN_ROUND2_RETIRED_IDS:
        question = load_candidate_question(golden_dir / candidate_id)
        if not question:
            continue
        metadata = question.get("golden_candidate", {})
        if metadata.get("content_key"):
            retired_keys.add(str(metadata["content_key"]))
            continue
        retired_keys.add(
            golden_candidate_content_key(
                {"group_key": question.get("group_key")},
                question,
            )
        )
    return retired_keys


def validate_golden_question(candidate_dir: Path, require_curve_large: bool = True) -> list[str]:
    blockers = []
    question = load_candidate_question(candidate_dir)
    if not question:
        return ["missing_question_json"]
    metadata = question.get("golden_candidate", {})
    if not has_answer_blocks(question):
        blockers.append("answer_empty")
    for asset in metadata.get("asset_files", []):
        if not (candidate_dir / "assets" / asset).exists():
            blockers.append(f"asset_missing:{asset}")
    if metadata.get("missing_assets"):
        blockers.append("metadata_missing_assets")
    if metadata.get("category") == "curve" and require_curve_large:
        sizes = [
            (candidate_dir / "assets" / asset).stat().st_size
            for asset in metadata.get("asset_files", [])
            if (candidate_dir / "assets" / asset).exists()
            and Path(asset).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        ]
        if not sizes or max(sizes) <= 20 * 1024:
            blockers.append("curve_figure_too_small")
    if set(question.get("quality_flags") or []) & {"answer_type_mismatch", "empty_answer", "stem_contaminated"}:
        blockers.append("quality_flagged")
    return blockers


def rename_round2_candidate_dirs(final_dirs: list[Path]) -> list[Path]:
    renamed_dirs: list[Path] = []
    for idx, candidate_dir in enumerate(final_dirs, start=1):
        question = load_candidate_question(candidate_dir)
        if not question:
            continue
        metadata = question.setdefault("golden_candidate", {})
        category = str(metadata.get("category") or "candidate")
        original_candidate_id = str(metadata.get("candidate_id") or candidate_dir.name)
        metadata["source_candidate_id"] = original_candidate_id
        metadata["candidate_id"] = f"round2_{idx:03d}_{category}"
        target_dir = candidate_dir.parent / metadata["candidate_id"]
        (candidate_dir / "question.json").write_text(
            json.dumps(question, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if target_dir != candidate_dir:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            candidate_dir.rename(target_dir)
        renamed_dirs.append(target_dir)
    return renamed_dirs


def build_golden_round2(
    batch_root: Path,
    source_golden_dir: Path,
    out_dir: Path | None = None,
) -> dict[str, object]:
    out_dir = out_dir or (batch_root / "golden_round2")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retired_keys = retired_content_keys_from_dir(source_golden_dir)
    source_summary = select_golden_candidates(
        batch_root,
        out_dir,
        quotas={
            "equation": 17,
            "electron_structure": 11,
            "table": 7,
            "device": 12,
            "curve": 12,
            "process": 6,
            "crystal": 7,
        },
    )

    candidate_dirs_by_category: dict[str, list[Path]] = {category: [] for category in GOLDEN_CANDIDATE_QUOTAS}
    retired_replaced = 0
    for candidate_dir in sorted(out_dir.iterdir()):
        if not candidate_dir.is_dir():
            continue
        question = load_candidate_question(candidate_dir)
        if not question:
            continue
        content_key = str(question.get("golden_candidate", {}).get("content_key") or "")
        blockers = validate_golden_question(candidate_dir)
        if content_key in retired_keys:
            retired_replaced += 1
            shutil.rmtree(candidate_dir)
            continue
        if blockers:
            shutil.rmtree(candidate_dir)
            continue
        category = str(question.get("golden_candidate", {}).get("category") or "")
        candidate_dirs_by_category.setdefault(category, []).append(candidate_dir)

    final_formal_dirs: list[Path] = []
    final_backup_dirs: list[Path] = []
    underfilled_quotas: dict[str, dict[str, int]] = {}
    for category in GOLDEN_CANDIDATE_QUOTAS:
        category_dirs = sorted(candidate_dirs_by_category.get(category, []), key=lambda p: p.name)
        formal_quota = GOLDEN_ROUND2_FORMAL_QUOTAS[category]
        backup_quota = GOLDEN_ROUND2_BACKUP_QUOTAS.get(category, 0)
        formal_dirs = category_dirs[:formal_quota]
        backup_dirs = category_dirs[formal_quota : formal_quota + backup_quota]
        final_formal_dirs.extend(formal_dirs)
        final_backup_dirs.extend(backup_dirs)
        if len(formal_dirs) < formal_quota or len(backup_dirs) < backup_quota:
            underfilled_quotas[category] = {
                "formal_expected": formal_quota,
                "formal_selected": len(formal_dirs),
                "backup_expected": backup_quota,
                "backup_selected": len(backup_dirs),
            }

    final_formal_dirs.sort(key=lambda p: p.name)
    final_backup_dirs.sort(key=lambda p: p.name)
    final_dirs = final_formal_dirs + final_backup_dirs
    keep_dirs = set(final_dirs)
    for extra_dir in [p for rows in candidate_dirs_by_category.values() for p in rows if p not in keep_dirs]:
        shutil.rmtree(extra_dir)

    final_dirs = rename_round2_candidate_dirs(final_dirs)
    final_formal_dir_names = {path.name for path in final_dirs[: len(final_formal_dirs)]}
    for idx, candidate_dir in enumerate(final_dirs, start=1):
        question = load_candidate_question(candidate_dir)
        if not question:
            continue
        metadata = question.setdefault("golden_candidate", {})
        metadata["round2_rank"] = idx
        metadata["set_role"] = "formal" if candidate_dir.name in final_formal_dir_names else "backup"
        source_candidate_id = str(metadata.get("source_candidate_id") or candidate_dir.name)
        if source_candidate_id in GOLDEN_ROUND2_RETAINED_NOTES:
            metadata["manual_status"] = GOLDEN_ROUND2_RETAINED_NOTES[source_candidate_id]
        (candidate_dir / "question.json").write_text(
            json.dumps(question, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    lines = [
        "# WS1 v4 Golden Round2",
        "",
        "| 编号 | set | 来源卷 | 题型 | 题号 | 资产数 | question.json |",
        "|---|---|---|---|---|---:|---|",
    ]
    category_counts: Counter[str] = Counter()
    formal_counts: Counter[str] = Counter()
    backup_counts: Counter[str] = Counter()
    validation_failures: dict[str, list[str]] = {}
    for candidate_dir in final_dirs:
        question = load_candidate_question(candidate_dir)
        if not question:
            continue
        metadata = question["golden_candidate"]
        blockers = validate_golden_question(candidate_dir)
        if blockers:
            validation_failures[candidate_dir.name] = blockers
        category = str(metadata.get("category"))
        category_counts[category] += 1
        if metadata.get("set_role") == "formal":
            formal_counts[category] += 1
        else:
            backup_counts[category] += 1
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate_dir.name,
                    str(metadata.get("set_role")),
                    str(question.get("group_key") or "").replace("|", "/"),
                    GOLDEN_CATEGORY_LABELS.get(category, category),
                    str(question.get("local_question_id") or question.get("question_id") or ""),
                    str(len(metadata.get("asset_files", []))),
                    f"`{candidate_dir.name}/question.json`",
                ]
            )
            + " |"
        )
    (out_dir / "candidates_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "batch_root": str(batch_root),
        "source_golden_dir": str(source_golden_dir),
        "out_dir": str(out_dir),
        "selected_total": len(final_dirs),
        "formal_count": sum(formal_counts.values()),
        "backup_count": sum(backup_counts.values()),
        "category_counts": dict(category_counts),
        "formal_category_counts": dict(formal_counts),
        "backup_category_counts": dict(backup_counts),
        "retired_ids": sorted(GOLDEN_ROUND2_RETIRED_IDS),
        "retired_content_keys": len(retired_keys),
        "retired_replaced": retired_replaced,
        "target_category_counts": dict(GOLDEN_CANDIDATE_QUOTAS),
        "target_formal_category_counts": dict(GOLDEN_ROUND2_FORMAL_QUOTAS),
        "target_backup_category_counts": dict(GOLDEN_ROUND2_BACKUP_QUOTAS),
        "underfilled_quotas": underfilled_quotas,
        "source_selection_summary": source_summary,
        "validation_failures": validation_failures,
        "machine_check_pass": (
            len(final_dirs) == 60
            and sum(formal_counts.values()) == 50
            and sum(backup_counts.values()) == 10
            and dict(category_counts) == GOLDEN_CANDIDATE_QUOTAS
            and dict(formal_counts) == GOLDEN_ROUND2_FORMAL_QUOTAS
            and dict(backup_counts) == GOLDEN_ROUND2_BACKUP_QUOTAS
            and not underfilled_quotas
            and not validation_failures
        ),
        "index": str(out_dir / "candidates_index.md"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    commands = {
        "extract",
        "probe-batch",
        "batch-extract",
        "batch-extract-v3",
        "batch-extract-v4",
        "select-golden-candidates",
        "build-golden-round2",
        "final-answer-type-gate",
    }
    if len(sys.argv) > 1 and sys.argv[1] not in commands:
        sys.argv.insert(1, "extract")

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    extract_ap = sub.add_parser("extract")
    extract_ap.add_argument("docx", type=Path)
    extract_ap.add_argument("--out-root", type=Path, default=Path("/tmp/yher_ws1_proto"))
    probe_ap = sub.add_parser("probe-batch")
    probe_ap.add_argument("--source-root", type=Path, default=Path("../上海化学卷合集"))
    probe_ap.add_argument("--out-root", type=Path, default=Path("/tmp/yher_ws1_batch"))
    batch_ap = sub.add_parser("batch-extract")
    batch_ap.add_argument("--source-root", type=Path, default=Path("../上海化学卷合集"))
    batch_ap.add_argument("--out-root", type=Path, default=Path("/tmp/yher_ws1_batch"))
    batch_ap.add_argument("--preview-limit", type=int, default=5)
    batch_v3_ap = sub.add_parser("batch-extract-v3")
    batch_v3_ap.add_argument("--source-root", type=Path, default=Path("../上海化学卷合集"))
    batch_v3_ap.add_argument("--out-root", type=Path, default=Path("/tmp/yher_ws1_batch_v3"))
    batch_v3_ap.add_argument("--preview-limit", type=int, default=5)
    batch_v4_ap = sub.add_parser("batch-extract-v4")
    batch_v4_ap.add_argument("--source-root", type=Path, default=Path("../上海化学卷合集"))
    batch_v4_ap.add_argument("--out-root", type=Path, default=Path("/tmp/yher_ws1_batch_v4"))
    batch_v4_ap.add_argument("--preview-limit", type=int, default=5)
    batch_v4_ap.add_argument("--source-list", type=Path, default=None)
    golden_ap = sub.add_parser("select-golden-candidates")
    golden_ap.add_argument("--batch-root", type=Path, default=Path("/tmp/yher_ws1_batch_v3"))
    golden_ap.add_argument("--out-dir", type=Path, default=None)
    round2_ap = sub.add_parser("build-golden-round2")
    round2_ap.add_argument("--batch-root", type=Path, default=Path("/tmp/yher_ws1_batch_v4"))
    round2_ap.add_argument("--source-golden-dir", type=Path, default=Path("/tmp/yher_ws1_batch_v3/golden_candidates"))
    round2_ap.add_argument("--out-dir", type=Path, default=None)
    final_gate_ap = sub.add_parser("final-answer-type-gate")
    final_gate_ap.add_argument("--batch-root", type=Path, default=Path("/tmp/yher_ws1_batch_v4"))
    args = ap.parse_args()

    if args.command == "probe-batch":
        summary = run_format_probe(args.source_root, args.out_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "batch-extract":
        summary = run_batch_extract(args.source_root, args.out_root, preview_limit=args.preview_limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "batch-extract-v3":
        summary = run_batch_extract_v3(args.source_root, args.out_root, preview_limit=args.preview_limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "batch-extract-v4":
        summary = run_batch_extract_v4(
            args.source_root,
            args.out_root,
            preview_limit=args.preview_limit,
            source_paths=read_source_list(args.source_list, args.source_root),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "select-golden-candidates":
        summary = select_golden_candidates(args.batch_root, args.out_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-golden-round2":
        summary = build_golden_round2(args.batch_root, args.source_golden_dir, args.out_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "final-answer-type-gate":
        summary = apply_final_dataset_answer_type_gate_to_batch(args.batch_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    summary = extract_docx(args.docx, args.out_root, render_preview_file=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] {summary['out_dir']}/preview.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
