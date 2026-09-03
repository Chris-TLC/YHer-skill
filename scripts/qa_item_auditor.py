#!/usr/bin/env python3
"""Batch 11 QA-2 per-item usability audit runner.

Default mode runs the 120-item calibration package only:

    python3 scripts/qa_item_auditor.py

Outputs are written under /tmp/yher_batch11_qa2/calibration_120 by default.
The script reads official v4 service data through iter_service_items(), writes no
official data, and keeps all review rows unsigned for Claude/user adjudication.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("/tmp/yher_batch11_qa2")
CALIBRATION_INPUT = OUT_ROOT / "calibration_120" / "calibration_120.jsonl"
BUNDLED_NODE = Path(os.environ.get("YHER_NODE_BIN", str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node")))
BUNDLED_NODE_MODULES = Path(os.environ.get("YHER_NODE_MODULES", str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")))

sys.path.insert(0, str(ROOT))

from adapters.vision_client import VISION_CONFIGS  # noqa: E402
from core.data.item_bank_v4 import iter_service_items, load_service_pool, solution_answers  # noqa: E402
from core.data.ws2_transcripts import resolve_image_path  # noqa: E402
from core.render.block_renderer import RAW_ASSET_URL_PREFIX, _Ctx, item_to_rir  # noqa: E402


DIMENSIONS = [
    "option_complete",
    "option_no_sticky",
    "stem_not_truncated",
    "stem_no_cross_item",
    "stem_not_fragment",
    "stem_length_reasonable",
    "answer_nonempty",
    "answer_format_normal",
    "subanswer_complete",
    "no_asset_literal_residue",
    "no_degrade_placeholder",
    "latex_hygiene",
    "text_ion_format_normal",
    "table_complete",
    "image_dimensions_normal",
    "subanswer_not_hollow",
    "no_hollow_mention",
    "no_fragment_literal",
    "stem_answer_sub_coverage",
    "answer_stem_match_flag",
]

BLOCKING_DIMENSIONS = {
    "stem_not_truncated",
    "stem_no_cross_item",
    "stem_not_fragment",
    "answer_nonempty",
}

MACHINE_ISSUE_DIMENSIONS = {
    "option_complete",
    "option_no_sticky",
    "stem_not_truncated",
    "stem_no_cross_item",
    "stem_not_fragment",
    "answer_nonempty",
    "answer_format_normal",
    "no_asset_literal_residue",
    "no_degrade_placeholder",
    "latex_hygiene",
    "text_ion_format_normal",
    "image_dimensions_normal",
    "subanswer_not_hollow",
    "no_hollow_mention",
    "no_fragment_literal",
    "stem_answer_sub_coverage",
}

BUCKET_BY_DIMENSION = {
    "option_complete": "切分",
    "option_no_sticky": "切分",
    "stem_not_truncated": "切分",
    "stem_no_cross_item": "切分",
    "stem_not_fragment": "切分",
    "stem_length_reasonable": "切分",
    "answer_nonempty": "答案",
    "answer_format_normal": "答案",
    "subanswer_complete": "答案",
    "no_asset_literal_residue": "资产",
    "no_degrade_placeholder": "资产",
    "latex_hygiene": "latex",
    "text_ion_format_normal": "文本格式",
    "table_complete": "资产",
    "image_dimensions_normal": "资产",
    "subanswer_not_hollow": "答案",
    "no_hollow_mention": "答案",
    "no_fragment_literal": "资产",
    "stem_answer_sub_coverage": "答案",
    "answer_stem_match_flag": "答案",
}

FAILURE_PROMPT_RE = re.compile(
    r"(无法识别|不能识别|图片.*模糊|看不清|无法读取|cannot\s+(?:read|recognize)|sorry)",
    re.I,
)
ASSET_LITERAL_RE = re.compile(r"\[(?:formula|figure|image|OMML|asset)[^\]]*\]|\b(?:formula|figure):[A-Za-z0-9_.-]+", re.I)
FRAGMENT_LITERAL_RE = re.compile(
    r"(?:\b(?:wmf|emf|png|jpe?g|gif|svg)\]|[A-Za-z0-9_-]+\.(?:wmf|emf|png|jpe?g|gif|svg)\]|<m:oMath\b|\bc\s*[（(]\s*[）)])",
    re.I,
)
ION_BROKEN_RE = re.compile(
    r"(?:\b(?:NH|SO|CO|NO|PO|HCO|HSO|MnO|CrO)\+[\d]+|\b(?:CO|SO|NO|PO)\d+/\d+|\b[A-Z][a-z]?\+\d)",
    re.I,
)
OPTION_LABEL_RE = re.compile(r"([A-D])\s*[.．、]")
TIGHT_OPTION_RE = re.compile(r"[A-D]\s*[.．、][^\n]*?[^\s　]([B-D])\s*[.．、]")
SUBQUESTION_RE = re.compile(r"[（(]\s*([1-9])\s*[）)]")
CIRCLED_SUBQUESTION = "①②③④⑤⑥⑦⑧⑨⑩"
CIRCLED_SUBQUESTION_RE = re.compile(f"([{CIRCLED_SUBQUESTION}])")
SUBQUESTION_MARKER_RE = re.compile(rf"[（(]\s*[1-9]\s*[）)]|[{CIRCLED_SUBQUESTION}]\s*[.．、]")
QUESTION_NUM_RE = re.compile(r"^\s*\d+\s*[.．、]")
LATEX_PSEUDO_FRAC_RE = re.compile(r"\\frac\{\s*\d{1,2}\s*\}\{\s*\d{1,2}\s*\}")
VISUAL_SENTINEL = "⟪V⟫"


def load_env_file(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def choose_node_bin(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if BUNDLED_NODE.exists():
        return str(BUNDLED_NODE)
    found = shutil.which("node")
    if found:
        return found
    raise RuntimeError("node executable not found")


def node_env(out_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    paths = [str(out_dir / "node_modules")]
    if BUNDLED_NODE_MODULES.exists():
        paths.append(str(BUNDLED_NODE_MODULES))
    if env.get("NODE_PATH"):
        paths.append(env["NODE_PATH"])
    env["NODE_PATH"] = os.pathsep.join(paths)
    return env


def katex_dist(out_dir: Path) -> Optional[Path]:
    candidates = [
        out_dir / "node_modules/katex/dist",
        BUNDLED_NODE_MODULES / "katex/dist",
        ROOT / "node_modules/katex/dist",
        # batch14 root-cause fix (Claude 2026-07-05): previously all three candidates
        # above missed on this machine → dist=None → the screenshot page's katexJs was
        # empty → window.katex undefined → rir_renderer fell back to textContent=source
        # for every kind=latex node → the VL reported "latex not rendered" en masse
        # (same-source distortion as the batch11 ledger).
        # The preserved KaTeX has always lived in the batch8 directory (see the
        # "KaTeX offline build" item in the handoff doc).
        Path("/tmp/yher_batch8_ws2/node_modules/katex/dist"),
    ]
    for c in candidates:
        if (c / "katex.min.js").exists() and (c / "contrib/mhchem.min.js").exists():
            return c
    return None


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                return parsed if isinstance(parsed, dict) else {"raw": parsed}
            except Exception:
                pass
    return {"raw_text": text}


def text_len_zhish(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def segment_text(seg: Any) -> str:
    if isinstance(seg, str):
        return seg
    if not isinstance(seg, dict):
        return ""
    t = seg.get("type")
    if t == "text":
        return str(seg.get("text") or "")
    if t == "math_omml":
        return " " + str(seg.get("latex") or seg.get("omml") or "") + " "
    if t in ("formula", "figure"):
        return ""
    if t == "table":
        cells: List[str] = []
        for row in seg.get("rows") or []:
            for cell in row or []:
                cells.append(cell_text(cell))
        return "\n".join(cells)
    return ""


def cell_text(cell: Any) -> str:
    if isinstance(cell, str):
        return cell
    if isinstance(cell, dict):
        return segment_text(cell)
    if isinstance(cell, list):
        return "".join(cell_text(x) for x in cell)
    return ""


def para_segments(para: Any) -> List[Dict[str, Any]]:
    if isinstance(para, dict):
        segs = para.get("para") or []
    elif isinstance(para, list):
        segs = para
    else:
        segs = []
    return [seg for seg in segs if isinstance(seg, dict)]


def field_paragraph_texts(item: Dict[str, Any], field: str) -> List[str]:
    texts: List[str] = []
    for para in item.get(field) or []:
        texts.append("".join(segment_text(seg) for seg in para_segments(para)))
    return texts


def zone_text(item: Dict[str, Any], fields: Sequence[str]) -> str:
    pieces: List[str] = []
    for field in fields:
        pieces.extend(field_paragraph_texts(item, field))
    return "\n".join(pieces)


def iter_segments(item: Dict[str, Any], fields: Sequence[str]) -> Iterator[Tuple[str, int, int, Dict[str, Any]]]:
    for field in fields:
        for para_idx, para in enumerate(item.get(field) or []):
            for seg_idx, seg in enumerate(para_segments(para)):
                yield field, para_idx, seg_idx, seg


def walk_nodes(nodes: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for node in nodes:
        yield node
        if node.get("kind") == "table":
            for row in node.get("rows") or []:
                for cell in row or []:
                    yield from walk_nodes(cell or [])


def zone_nodes(rir: Dict[str, Any], zone: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for para in ((rir.get("zones") or {}).get(zone) or []):
        out.extend(walk_nodes(para or []))
    return out


def all_rir_nodes(rir: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    for paragraphs in (rir.get("zones") or {}).values():
        for para in paragraphs or []:
            yield from walk_nodes(para or [])


def visible_zone(rir: Dict[str, Any], zone: str) -> bool:
    for node in zone_nodes(rir, zone):
        kind = node.get("kind")
        if kind == "text" and str(node.get("text") or "").strip():
            return True
        if kind in {"latex", "image", "table"}:
            return True
    return False


def has_stem_visual(rir: Dict[str, Any]) -> bool:
    return any(n.get("kind") in {"image", "table"} for n in zone_nodes(rir, "stem"))


def check(ok: bool, evidence: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {"ok": bool(ok), "evidence": evidence or []}


def short(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def option_labels(stem_paras: List[str]) -> List[str]:
    labels: List[str] = []
    for para in stem_paras:
        labels.extend(m.group(1) for m in OPTION_LABEL_RE.finditer(para))
        labels.extend(m.group(1) for m in re.finditer(r"(?m)(?:^|\n)\s*([A-D])\s+(?=\S)", para))
    return labels


def sticky_option_evidence(stem_paras: List[str]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for idx, para in enumerate(stem_paras):
        labels = [m.group(1) for m in OPTION_LABEL_RE.finditer(para)]
        if len(labels) > 1:
            issues.append({"type": "multi_option_para", "para_index": idx, "labels": labels, "text": short(para)})
        for m in TIGHT_OPTION_RE.finditer(para):
            issues.append({"type": "tight_option_join", "para_index": idx, "next_label": m.group(1), "text": short(para)})
    return issues


def unclosed_bracket_evidence(text: str) -> List[Dict[str, Any]]:
    pairs = [("（", "）"), ("(", ")"), ("【", "】"), ("[", "]"), ("{", "}")]
    issues = []
    for left, right in pairs:
        diff = text.count(left) - text.count(right)
        if diff > 0:
            issues.append({"type": "unclosed_bracket", "left": left, "right": right, "diff": diff})
    return issues


def stem_truncation_evidence(stem_text: str) -> List[Dict[str, Any]]:
    stripped = stem_text.strip()
    issues = unclosed_bracket_evidence(stripped)

    if re.search(r"[，,、；;:：]\s*$", stripped):
        issues.append({"type": "dangling_terminal", "text": short(stripped[-120:])})
    elif not re.search(r"[。．.!！？?]\s*$", stripped):
        terminal_patterns = [
            r"(?:和|与|或|及|为|是|属于|分别为|可得|生成|如下|如图所示)$",
            r"(?:第|选|填|写出)$",
        ]
        for pat in terminal_patterns:
            if re.search(pat, stripped):
                issues.append({"type": "dangling_terminal", "text": short(stripped[-120:])})
                break

    normal_fill_blank_end = re.search(r"(?:_{3,}| {8,}|[\u3000]{3,})\s*[。．.!！？?]\s*$", stripped)
    if not normal_fill_blank_end and re.search(r"(?:_{3,}| {8,}|[\u3000]{3,})\s*$", stripped):
        issues.append({"type": "orphan_blank_at_end", "text": short(stripped[-120:])})
    return issues


def cross_item_evidence(stem_text: str, q_num: Any) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if re.search(r"【?\s*答案\s*】?|【?\s*解析\s*】?", stem_text):
        issues.append({"type": "answer_or_analysis_marker_in_stem", "text": short(stem_text)})
    try:
        q = int(q_num)
    except Exception:
        q = None
    if q is not None:
        for nxt in (q + 1, q + 2):
            pat = re.compile(rf"(?:^|\n)\s*{nxt}\s*[.．、](?!\d)")
            m = pat.search(stem_text[40:])
            if m:
                issues.append({"type": "next_question_marker", "marker": str(nxt), "text": short(stem_text[m.start() : m.start() + 120])})
                break
    return issues


def fragment_evidence(stem_text: str, rir: Dict[str, Any]) -> List[Dict[str, Any]]:
    body = QUESTION_NUM_RE.sub("", stem_text.strip())
    compact_len = text_len_zhish(body)
    if has_stem_visual(rir):
        return []
    if compact_len < 18:
        return [{"type": "too_short_fragment", "length": compact_len, "text": short(body)}]
    if compact_len < 38 and re.search(r"(写出|指出|填写|回答).{0,16}(结构简式|方程式|化学式|名称|离子方程式)", body):
        return [{"type": "imperative_without_context", "length": compact_len, "text": short(body)}]
    return []


def answer_format_evidence(answer_text: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    compact = re.sub(r"\s+", "", answer_text or "")
    marker_duplicate = re.search(r"(?:【(?:答案|正确答案)】|答案[:：]?|正确答案[:：]?)\s*([A-D])\1(?![A-Z])", compact)
    bare_duplicate = re.fullmatch(r"(?:【(?:答案|正确答案)】)?([A-D])\1", compact)
    if marker_duplicate or bare_duplicate:
        issues.append({"type": "duplicate_choice_letter", "text": short(answer_text)})
    if re.search(r"【?\s*答案\s*】?\s*\d+\s*[.．]\s*$", compact):
        issues.append({"type": "empty_answer_reference", "text": short(answer_text)})
    if FAILURE_PROMPT_RE.search(answer_text or ""):
        issues.append({"type": "failure_prompt_text", "text": short(answer_text)})
    return issues


def subquestion_evidence(stem_text: str, answer_text: str) -> List[Dict[str, Any]]:
    stem_nums = {int(x) for x in SUBQUESTION_RE.findall(stem_text or "")}
    if len(stem_nums) < 2:
        return []
    answer_nums = {int(x) for x in SUBQUESTION_RE.findall(answer_text or "")}
    missing = sorted(stem_nums - answer_nums)
    return [] if len(missing) <= 1 else [{"type": "subquestion_answer_mismatch", "stem": sorted(stem_nums), "answer": sorted(answer_nums), "missing": missing}]


def subquestion_markers(text: str) -> List[str]:
    markers: List[str] = []
    for m in SUBQUESTION_RE.finditer(text or ""):
        markers.append(m.group(1))
    for m in CIRCLED_SUBQUESTION_RE.finditer(text or ""):
        markers.append(str(CIRCLED_SUBQUESTION.index(m.group(1)) + 1))
    return markers


def subquestion_marker_level(marker: str) -> str:
    return "circled" if any(ch in CIRCLED_SUBQUESTION for ch in marker) else "paren"


def subquestion_marker_num(marker: str) -> Optional[int]:
    m = SUBQUESTION_RE.search(marker)
    if m:
        return int(m.group(1))
    c = CIRCLED_SUBQUESTION_RE.search(marker)
    if c:
        return CIRCLED_SUBQUESTION.index(c.group(1)) + 1
    return None


def subanswer_hollow_evidence(text: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    text = re.split(r"【\s*(?:解析|分析|小问\d*详解)\s*】|【小问\d+详解】", text or "", maxsplit=1)[0]
    markers = list(SUBQUESTION_MARKER_RE.finditer(text or ""))
    body_start_match = re.search(r"(?:【?\s*(?:答案|正确答案)\s*】?|答案[:：]?|正确答案[:：]?)", text or "")
    body_start = body_start_match.end() if body_start_match else 0
    for left, right in zip(markers, markers[1:]):
        if subquestion_marker_level(left.group(0)) != subquestion_marker_level(right.group(0)):
            continue
        between = text[left.end() : right.start()]
        meaningful = re.sub(r"[\s　,，.。;；:：、\-—]+", "", between)
        left_num = subquestion_marker_num(left.group(0))
        right_num = subquestion_marker_num(right.group(0))
        prefix = re.sub(r"[\s　,，.。;；:：、\-—]+", "", text[body_start : left.start()])
        leading_empty = not prefix and left.start() - body_start <= 8
        if not meaningful and left_num is not None and right_num is not None:
            issues.append(
                {
                    "type": "leading_empty_subanswer" if leading_empty else "adjacent_empty_subanswer",
                    "left": left.group(0),
                    "right": right.group(0),
                    "text": short(text[max(0, left.start() - 40) : min(len(text), right.end() + 80)]),
                }
            )
    if markers:
        last = markers[-1]
        tail = re.sub(r"[\s　,，.。;；:：、\-—]+", "", text[last.end() :])
        if not tail:
            issues.append({"type": "terminal_empty_subanswer", "marker": last.group(0), "text": short(text[max(0, last.start() - 80) :])})
    return issues[:20]


def segment_text_with_visual_marker(seg: Any) -> str:
    if isinstance(seg, str):
        return seg
    if not isinstance(seg, dict):
        return ""
    t = seg.get("type")
    if t == "text":
        return str(seg.get("text") or "")
    if t in {"math_omml", "formula", "figure", "image", "latex", "media"}:
        return VISUAL_SENTINEL
    if t == "table":
        return VISUAL_SENTINEL
    return ""


def field_node_streams(item: Dict[str, Any], fields: Sequence[str]) -> List[Dict[str, Any]]:
    streams: List[Dict[str, Any]] = []
    for field in fields:
        for para_idx, para in enumerate(item.get(field) or []):
            stream = "".join(segment_text_with_visual_marker(seg) for seg in para_segments(para))
            if stream:
                streams.append({"field": field, "para": para_idx, "text": stream})
    return streams


def field_node_stream_text(item: Dict[str, Any], fields: Sequence[str]) -> str:
    return "\n".join(row["text"] for row in field_node_streams(item, fields))


def visible_text_node_streams(item: Dict[str, Any], fields: Sequence[str]) -> List[Dict[str, Any]]:
    streams: List[Dict[str, Any]] = []
    for field in fields:
        for para_idx, para in enumerate(item.get(field) or []):
            pieces: List[str] = []
            for seg in para_segments(para):
                if seg.get("type") == "text":
                    pieces.append(str(seg.get("text") or ""))
            text = "".join(pieces)
            if text:
                streams.append({"field": field, "para": para_idx, "text": text})
    return streams


def hollow_mention_evidence(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    mention_re = re.compile(
        r"(结构简式|化学式|电子式|电极反应式|离子方程式|反应方程式|化学方程式|方程式)\s*(?:为|是|[:：])?"
    )
    for field in ("answer_blocks_effective", "analysis_blocks"):
        combined = field_node_stream_text(item, (field,))
        if not combined:
            continue
        for m in mention_re.finditer(combined):
            tail = combined[m.end() : m.end() + 8]
            tail_no_space = re.sub(r"\s+", "", tail)
            if VISUAL_SENTINEL in tail_no_space[: len(VISUAL_SENTINEL) + 2]:
                continue
            if not tail_no_space or re.match(r"^[:：]?[，,。；;、:：）)]*$", tail_no_space):
                issues.append(
                    {
                        "type": "hollow_mention",
                        "field": field,
                        "mention": m.group(0),
                        "text": short(combined[max(0, m.start() - 50) : m.end() + 80]),
                    }
                )
    return issues[:20]


VISIBLE_FRAGMENT_LITERAL_RE = re.compile(r"(?:\b(?:wmf|emf)\]|\[(?:formula|figure):[^\]]+\])", re.I)
EMPTY_C_PARENS_RE = re.compile(r"\bc\s*[（(]\s*[）)]", re.I)


def fragment_literal_evidence(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for row in visible_text_node_streams(item, ("stem_blocks", "answer_blocks_effective", "analysis_blocks")):
        text = row["text"]
        for m in VISIBLE_FRAGMENT_LITERAL_RE.finditer(text or ""):
            issues.append(
                {
                    "type": "fragment_literal",
                    "field": row["field"],
                    "para": row["para"],
                    "literal": m.group(0),
                    "text": short(text[max(0, m.start() - 50) : m.end() + 50]),
                }
            )
    for row in field_node_streams(item, ("stem_blocks", "answer_blocks_effective", "analysis_blocks")):
        text = row["text"]
        for m in EMPTY_C_PARENS_RE.finditer(text or ""):
            issues.append(
                {
                    "type": "empty_c_parentheses",
                    "field": row["field"],
                    "para": row["para"],
                    "literal": m.group(0),
                    "text": short(text[max(0, m.start() - 50) : m.end() + 50]),
                }
            )
    return issues[:20]


def compact_answer_body(text: str) -> str:
    compact = re.sub(r"(【?\s*(?:答案|正确答案|解析)\s*】?|答案[:：]?|正确答案[:：]?)", "", text or "")
    return re.sub(r"[\s　,，.。;；:：、（）()\[\]【】]+", "", compact)


SEMANTIC_SUBQUESTION_RE = re.compile(r"(?:^|\n|[。；;：:])\s*（\s*([1-9]\d?)\s*）|小问\s*([1-9]\d?)")
ANSWER_SEMANTIC_SUBQUESTION_RE = re.compile(r"[（(]\s*([1-9]\d?)\s*[）)]|小问\s*([1-9]\d?)")


def semantic_subquestion_nums(text: str) -> List[int]:
    nums: List[int] = []
    for m in SEMANTIC_SUBQUESTION_RE.finditer(text or ""):
        raw = m.group(1) or m.group(2)
        if raw:
            nums.append(int(raw))
    return nums


def answer_semantic_subquestion_nums(text: str) -> List[int]:
    nums: List[int] = []
    for m in ANSWER_SEMANTIC_SUBQUESTION_RE.finditer(text or ""):
        raw = m.group(1) or m.group(2)
        if raw:
            nums.append(int(raw))
    return nums


def is_single_choice_stem(stem_text: str) -> bool:
    labels = set(m.group(1) for m in OPTION_LABEL_RE.finditer(stem_text or ""))
    labels.update(m.group(1) for m in re.finditer(r"(?m)(?:^|\n)\s*([A-D])\s+(?=\S)", stem_text or ""))
    return labels >= {"A", "B", "C", "D"}


def stem_answer_sub_coverage_evidence(stem_text: str, answer_text: str, solution_answer_text: str = "") -> List[Dict[str, Any]]:
    if is_single_choice_stem(stem_text):
        return []
    stem_nums = sorted(set(semantic_subquestion_nums(stem_text)))
    if len(stem_nums) < 2:
        return []
    answer_nums = sorted(set(answer_semantic_subquestion_nums(answer_text)))
    solution_nums = sorted(set(answer_semantic_subquestion_nums(solution_answer_text)))
    compact = compact_answer_body(answer_text)
    threshold = 12 * len(stem_nums)
    if not answer_nums and len(compact) < threshold:
        return [
            {
                "type": "low_answer_subquestion_coverage",
                "stem_subquestions": stem_nums,
                "answer_subquestions": answer_nums,
                "answer_compact_length": len(compact),
                "threshold": threshold,
                "answer_preview": short(answer_text),
            }
        ]
    return []


ANSWER_STEM_TERMS = [
    "结构简式",
    "化学式",
    "分子式",
    "电子式",
    "离子方程式",
    "反应方程式",
    "方程式",
    "电极反应式",
    "同分异构体",
    "官能团",
    "键角",
    "配平",
    "氧化还原",
    "共价键",
    "离子键",
    "氢键",
    "晶胞",
    "配位",
    "沉淀",
    "溶液",
    "滴定",
    "水解",
    "电离",
    "浓度",
    "物质的量",
    "电极",
    "阴极",
    "阳极",
    "原电池",
    "电解池",
    "焓变",
    "平衡常数",
    "转化率",
    "羟基",
    "羧基",
    "醛基",
    "酯基",
    "酚羟基",
]
CHEM_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}(?:[+-])?\b|[A-Z][₀₁₂₃₄₅₆₇₈₉0-9]+")


def answer_stem_terms(text: str) -> set[str]:
    terms = set()
    source = text or ""
    for m in CHEM_ENTITY_RE.finditer(source):
        terms.add(m.group(0))
    for term in ANSWER_STEM_TERMS:
        if term in source:
            terms.add(term)
    if any(group in source for group in ("羟基", "羧基", "醛基", "酯基", "酚羟基")):
        terms.add("官能团")
    return terms


def answer_stem_match_flag_evidence(stem_text: str, answer_text: str, q_num: Any = None) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    stem_head = re.sub(r"\s+", "", stem_text or "")[:20]
    if re.match(r"^[＞>、，,；;:：)]", stem_head):
        issues.append({"type": "stem_starts_with_continuation_punctuation", "stem_head": short(stem_head)})

    try:
        q = int(q_num)
    except Exception:
        q = None

    raw_answer = re.sub(r"\s+", "", answer_text or "")
    if q is not None and q > 1 and re.match(r"^【正确答案】1[、.．]", raw_answer):
        issues.append(
            {
                "type": "answer_key_starts_at_1_for_later_question",
                "q_num": q,
                "answer_head": short(raw_answer[:80]),
            }
        )

    if re.match(r"^【?答案】?[①②③④⑤⑥⑦⑧⑨⑩]{2,}(?:相加|相减|代入|联立)", raw_answer):
        issues.append(
            {
                "type": "answer_starts_with_unresolved_circled_reference",
                "answer_head": short(raw_answer[:80]),
            }
        )

    stripped = re.sub(r"^【?(?:正确)?答案】?|^答案[:：]?|^正确答案[:：]?", "", raw_answer)
    first_num_match = re.match(r"^(?:[（(]\s*)?([1-9]\d?)(?:\s*[）)])?[、.．]", stripped)
    if q is not None and first_num_match:
        first = int(first_num_match.group(1))
        if raw_answer.startswith("【正确答案】") and first <= 12 and first != 1 and first != q:
            issues.append(
                {
                    "type": "answer_first_number_not_1_or_qnum",
                    "q_num": q,
                    "first_answer_number": first,
                    "answer_head": short(raw_answer[:80]),
                }
            )
    return issues[:20]


def table_evidence(item: Dict[str, Any], rir: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for field, para_idx, seg_idx, seg in iter_segments(item, ("stem_blocks", "answer_blocks_effective", "analysis_blocks")):
        if seg.get("type") != "table":
            continue
        rows = seg.get("rows") or []
        if not rows:
            issues.append({"type": "empty_table", "field": field, "para": para_idx, "seg": seg_idx})
            continue
        widths = [len(r or []) for r in rows]
        if len(rows) > 1 and max(widths or [0]) <= 1:
            issues.append({"type": "single_column_table", "field": field, "para": para_idx, "rows": len(rows)})
        empty_cells = 0
        total_cells = 0
        for row in rows:
            for cell in row or []:
                total_cells += 1
                if not cell_text(cell).strip():
                    empty_cells += 1
        if total_cells and empty_cells / total_cells > 0.5:
            issues.append({"type": "mostly_empty_table", "field": field, "empty": empty_cells, "total": total_cells})
    for node in all_rir_nodes(rir):
        if node.get("kind") != "table":
            continue
        rows = node.get("rows") or []
        if not rows:
            issues.append({"type": "empty_rir_table"})
    return issues


def image_dimension_evidence(rir: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for node in all_rir_nodes(rir):
        if node.get("kind") != "image":
            continue
        w = node.get("w")
        h = node.get("h")
        if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
            issues.append({"type": "missing_or_zero_dimensions", "asset_hash": node.get("asset_hash"), "url": node.get("url")})
            continue
        ratio = w / h if h else 999
        if ratio > 40 or ratio < 0.025:
            issues.append({"type": "extreme_ratio", "asset_hash": node.get("asset_hash"), "w": w, "h": h, "ratio": ratio})
    return issues


def latex_values(item: Dict[str, Any], rir: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for node in all_rir_nodes(rir):
        if node.get("kind") == "latex" and str(node.get("latex") or "").strip():
            values.append(str(node.get("latex")))
    for _field, _para, _seg, seg in iter_segments(item, ("stem_blocks", "answer_blocks_effective", "analysis_blocks")):
        if seg.get("latex"):
            values.append(str(seg.get("latex")))
    return list(dict.fromkeys(values))


def latex_hygiene_evidence(values: List[str], compile_map: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for latex in values:
        if FAILURE_PROMPT_RE.search(latex):
            issues.append({"type": "failure_prompt_latex", "latex": short(latex)})
        if LATEX_PSEUDO_FRAC_RE.search(latex):
            issues.append({"type": "pseudo_fraction_script", "latex": short(latex)})
        if compile_map is not None and latex in compile_map and not compile_map[latex].get("ok"):
            issues.append({"type": "katex_compile_error", "latex": short(latex), "error": short(str(compile_map[latex].get("error") or ""))})
    return issues


def validate_latex_batch(values: Sequence[str], out_dir: Path, node_bin: Optional[str]) -> Dict[str, Dict[str, Any]]:
    unique = [v for v in dict.fromkeys(values) if v.strip()]
    if not unique:
        return {}
    try:
        node = choose_node_bin(node_bin)
    except RuntimeError:
        return {}
    dist = katex_dist(out_dir)
    if dist is None:
        return {}

    work = out_dir / "machine" / "katex"
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / "latex_inputs.json"
    result_path = work / "latex_results.json"
    script_path = work / "validate_katex.cjs"
    write_json(input_path, unique)
    script_path.write_text(
        textwrap.dedent(
            f"""
            const fs = require("fs");
            const katex = require("katex");
            require("katex/contrib/mhchem");
            const inputs = JSON.parse(fs.readFileSync({json.dumps(str(input_path))}, "utf8"));
            const out = {{}};
            for (const latex of inputs) {{
              try {{
                katex.renderToString(latex, {{throwOnError:true, strict:"ignore", trust:false}});
                out[latex] = {{ok:true}};
              }} catch (err) {{
                out[latex] = {{ok:false, error:String(err && err.message || err)}};
              }}
            }}
            fs.writeFileSync({json.dumps(str(result_path))}, JSON.stringify(out, null, 2));
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [node, str(script_path)],
        cwd=str(ROOT),
        env=node_env(out_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not result_path.exists():
        return {}
    return read_json(result_path)


def machine_audit_item(
    item: Dict[str, Any],
    *,
    stratum: str = "",
    rir: Optional[Dict[str, Any]] = None,
    latex_compile_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rir = rir or item_to_rir(item, zones=("stem", "answer"))
    stem_paras = field_paragraph_texts(item, "stem_blocks")
    stem_text = "\n".join(stem_paras) or str(item.get("stem_text") or "")
    answer_block_text = zone_text(item, ("answer_blocks_effective",))
    analysis_text = zone_text(item, ("analysis_blocks",))
    solution_answer_text = "\n".join(solution_answers(item))
    answer_text = "\n".join(x for x in (answer_block_text, solution_answer_text) if x.strip())
    all_text = "\n".join(
        [
            zone_text(item, ("stem_blocks", "answer_blocks_effective", "analysis_blocks")),
            json.dumps(item.get("standard_solution") or {}, ensure_ascii=False),
        ]
    )

    labels = option_labels(stem_paras)
    option_needed = bool(labels)
    option_complete_ev = [] if not option_needed or set(labels) >= {"A", "B", "C", "D"} else [{"labels_found": sorted(set(labels)), "missing": sorted({"A", "B", "C", "D"} - set(labels))}]

    sticky_ev = sticky_option_evidence(stem_paras)
    trunc_ev = stem_truncation_evidence(stem_text)
    cross_ev = cross_item_evidence(stem_text, item.get("q_num"))
    fragment_ev = fragment_evidence(stem_text, rir)
    length_ok = text_len_zhish(stem_text) > 20 or has_stem_visual(rir)
    answer_nonempty_ok = bool(answer_text.strip()) or visible_zone(rir, "answer")
    answer_format_ev = []
    for source, source_text in (("answer_blocks_effective", answer_block_text), ("standard_solution", solution_answer_text)):
        for ev in answer_format_evidence(source_text):
            answer_format_ev.append({"source": source, **ev})
    subanswer_ev = subquestion_evidence(stem_text, answer_text)
    literal_matches = [m.group(0) for m in ASSET_LITERAL_RE.finditer(all_text)]
    literal_ev = [{"literal": x} for x in sorted(set(literal_matches))[:20]]
    placeholder_ev = [
        {"zone": zone, "reason": node.get("reason", ""), "kind": "placeholder"}
        for zone in ("stem", "answer")
        for node in zone_nodes(rir, zone)
        if node.get("kind") == "placeholder"
    ]
    ltx_values = latex_values(item, rir)
    latex_ev = latex_hygiene_evidence(ltx_values, latex_compile_map)
    ion_ev = [{"match": m.group(0), "text": short(all_text[max(0, m.start() - 50) : m.end() + 50])} for m in ION_BROKEN_RE.finditer(all_text)]
    table_ev = table_evidence(item, rir)
    image_ev = image_dimension_evidence(rir)
    answer_node_text = field_node_stream_text(item, ("answer_blocks_effective",))
    analysis_node_text = field_node_stream_text(item, ("analysis_blocks",))
    hollow_subanswer_ev = subanswer_hollow_evidence(answer_node_text) + subanswer_hollow_evidence(analysis_node_text)
    hollow_mention_ev = hollow_mention_evidence(item)
    fragment_literal_ev = fragment_literal_evidence(item)
    sub_coverage_ev = stem_answer_sub_coverage_evidence(stem_text, answer_text, solution_answer_text)
    answer_stem_match_ev = answer_stem_match_flag_evidence(stem_text, answer_text, item.get("q_num"))

    checks = {
        "option_complete": check(not option_complete_ev, option_complete_ev),
        "option_no_sticky": check(not sticky_ev, sticky_ev[:20]),
        "stem_not_truncated": check(not trunc_ev, trunc_ev),
        "stem_no_cross_item": check(not cross_ev, cross_ev),
        "stem_not_fragment": check(not fragment_ev, fragment_ev),
        "stem_length_reasonable": check(length_ok, [] if length_ok else [{"length": text_len_zhish(stem_text), "text": short(stem_text)}]),
        "answer_nonempty": check(answer_nonempty_ok, [] if answer_nonempty_ok else [{"answer_blocks": len(item.get("answer_blocks_effective") or []), "solution_answers": len(solution_answers(item))}]),
        "answer_format_normal": check(not answer_format_ev, answer_format_ev),
        "subanswer_complete": check(not subanswer_ev, subanswer_ev),
        "no_asset_literal_residue": check(not literal_ev, literal_ev),
        "no_degrade_placeholder": check(not placeholder_ev, placeholder_ev),
        "latex_hygiene": check(not latex_ev, latex_ev[:20]),
        "text_ion_format_normal": check(not ion_ev, ion_ev[:20]),
        "table_complete": check(not table_ev, table_ev[:20]),
        "image_dimensions_normal": check(not image_ev, image_ev[:20]),
        "subanswer_not_hollow": check(not hollow_subanswer_ev, hollow_subanswer_ev),
        "no_hollow_mention": check(not hollow_mention_ev, hollow_mention_ev),
        "no_fragment_literal": check(not fragment_literal_ev, fragment_literal_ev),
        "stem_answer_sub_coverage": check(not sub_coverage_ev, sub_coverage_ev),
        "answer_stem_match_flag": check(not answer_stem_match_ev, answer_stem_match_ev),
    }

    dimensions = {name: checks[name]["ok"] for name in DIMENSIONS}
    failed = [name for name, ok in dimensions.items() if not ok]
    row: Dict[str, Any] = {
        "schema_version": "qa2_machine_audit_v1",
        "item_id": item.get("item_id", ""),
        "group_key": item.get("group_key", ""),
        "section_num": item.get("section_num"),
        "q_num": item.get("q_num"),
        "stratum": stratum,
        "reviewer": "",
        "review_status": "pending",
        "dimensions": dimensions,
        "evidence": {name: checks[name]["evidence"] for name in DIMENSIONS},
        "machine_pass_count": sum(1 for ok in dimensions.values() if ok),
        "machine_failed_dimensions": failed,
    }
    for name, ok in dimensions.items():
        row[name] = ok
    return row


def load_target_items(input_jsonl: Optional[Path], all_service: bool) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    # The usability audit must see the full R1-R4 pool (2526, including fixable/blocked)
    # and must not be masked by the R5 whitelist — otherwise the ledger would only audit
    # whitelisted items and fix batches could never be re-checked. Hence the explicit
    # apply_r5=False (2026-07-06 R5 apply).
    if all_service:
        items = list(iter_service_items(apply_r5=False))
        return items, {it["item_id"]: "" for it in items}
    if input_jsonl is None:
        input_jsonl = CALIBRATION_INPUT
    rows = read_jsonl(input_jsonl)
    service = load_service_pool(apply_r5=False)
    items: List[Dict[str, Any]] = []
    strata: Dict[str, str] = {}
    missing: List[str] = []
    for row in rows:
        item_id = row.get("item_id", "")
        strata[item_id] = str(row.get("stratum") or "")
        item = service.get(item_id)
        if item is None:
            missing.append(item_id)
        else:
            items.append(item)
    if missing:
        raise RuntimeError(f"{len(missing)} target item_ids are not in service pool: {missing[:5]}")
    return items, strata


def build_rir_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in items:
        rir = item_to_rir(item, zones=("stem", "answer"))
        records.append({"item_id": item["item_id"], "rir": rir})
    return records


def run_machine_audit(
    items: List[Dict[str, Any]],
    strata: Dict[str, str],
    records: List[Dict[str, Any]],
    out_dir: Path,
    node_bin: Optional[str],
) -> List[Dict[str, Any]]:
    rirs = {rec["item_id"]: rec["rir"] for rec in records}
    all_latex: List[str] = []
    for item in items:
        all_latex.extend(latex_values(item, rirs[item["item_id"]]))
    compile_map = validate_latex_batch(all_latex, out_dir, node_bin)
    rows = [
        machine_audit_item(
            item,
            stratum=strata.get(item["item_id"], ""),
            rir=rirs[item["item_id"]],
            latex_compile_map=compile_map,
        )
        for item in items
    ]
    write_jsonl(out_dir / "machine_audit.jsonl", rows)
    return rows


def collect_image_paths(records: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    asset_paths: Dict[str, str] = {}
    raw_paths: Dict[str, str] = {}
    ctx = _Ctx.get()
    for rec in records:
        for node in all_rir_nodes(rec["rir"]):
            if node.get("kind") != "image":
                continue
            h = node.get("asset_hash")
            if h and h not in asset_paths:
                p = resolve_image_path(h)
                if p is not None:
                    asset_paths[h] = str(p)
            url = str(node.get("url") or "")
            if url.startswith(RAW_ASSET_URL_PREFIX):
                parsed = urllib.parse.urlparse(url)
                qs = urllib.parse.parse_qs(parsed.query)
                group_key = (qs.get("group_key") or [""])[0]
                media = (qs.get("media") or [""])[0]
                p = ctx.raw_asset_path(group_key, media)
                if p is not None:
                    raw_paths[f"{group_key}||{media}"] = str(p)
    return asset_paths, raw_paths


def render_screenshots(
    *,
    out_dir: Path,
    records: List[Dict[str, Any]],
    node_bin: Optional[str],
) -> Dict[str, Dict[str, str]]:
    node = choose_node_bin(node_bin)
    dist = katex_dist(out_dir)

    browser_dir = out_dir / "browser"
    screenshot_dir = out_dir / "screenshots"
    browser_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    input_path = browser_dir / "rir_records.json"
    asset_path = browser_dir / "asset_paths.json"
    raw_path = browser_dir / "raw_asset_paths.json"
    result_path = browser_dir / "screenshot_result.json"
    script_path = browser_dir / "qa2_render_screenshots.cjs"

    write_json(input_path, records)
    asset_paths, raw_paths = collect_image_paths(records)
    write_json(asset_path, asset_paths)
    write_json(raw_path, raw_paths)

    script_path.write_text(
        textwrap.dedent(
            f"""
            const fs = require("fs");
            const path = require("path");
            const {{ chromium }} = require("playwright");

            const records = JSON.parse(fs.readFileSync({json.dumps(str(input_path))}, "utf8"));
            const assetPaths = JSON.parse(fs.readFileSync({json.dumps(str(asset_path))}, "utf8"));
            const rawAssetPaths = JSON.parse(fs.readFileSync({json.dumps(str(raw_path))}, "utf8"));
            const katexPath = {json.dumps(str(dist / "katex.min.js") if dist else None)};
            const mhchemPath = {json.dumps(str(dist / "contrib/mhchem.min.js") if dist else None)};
            const katexJs = katexPath ? fs.readFileSync(katexPath, "utf8") : "";
            const mhchemJs = mhchemPath ? fs.readFileSync(mhchemPath, "utf8") : "";
            const rendererJs = fs.readFileSync({json.dumps(str(ROOT / "apps/web/rir_renderer.js"))}, "utf8");
            const screenshotDir = {json.dumps(str(screenshot_dir))};

            function html() {{
              return `<!doctype html><html><head><meta charset="utf-8"><base href="http://yher.local/">
              <style>
              body{{margin:0;background:#f5f5f7;color:#1d1d1f;font-family:-apple-system,"PingFang SC",system-ui,sans-serif}}
              .page{{box-sizing:border-box;min-height:100vh;padding:24px}}
              .item{{background:#fff;border:1px solid #d2d2d7;border-radius:8px;padding:20px;max-width:920px;margin:0 auto}}
              .head{{font:12px ui-monospace,monospace;color:#6e6e73;margin-bottom:14px;overflow-wrap:anywhere}}
              .zone-title{{font-size:13px;color:#6e6e73;font-weight:700;margin:16px 0 8px}}
              .rir-image{{max-width:100%!important;width:auto!important}}
              .rir-img-inline{{max-width:none!important;width:auto!important}}
              @media(max-width:640px){{.page{{padding:12px}}.item{{padding:14px;border-radius:6px}}}}
              </style>
              <script>${{katexJs}}<\\/script><script>${{mhchemJs}}<\\/script><script>${{rendererJs}}<\\/script>
              </head><body><main class="page"><section id="root" class="item"></section></main></body></html>`;
            }}

            async function waitForImages(page) {{
              await page.evaluate(async () => {{
                const imgs = Array.from(document.images);
                imgs.forEach(img => {{
                  img.loading = "eager";
                  img.decoding = "sync";
                }});
                await Promise.all(imgs.map(img => img.complete ? Promise.resolve() : Promise.race([
                  new Promise(resolve => {{
                    img.onload = resolve;
                    img.onerror = resolve;
                  }}),
                  new Promise(resolve => setTimeout(resolve, 1200))
                ])));
              }});
            }}

            async function constrainOversizedImages(page) {{
              await page.evaluate(() => {{
                const viewportWidth = window.innerWidth || 1280;
                document.querySelectorAll("img.rir-image:not(.rir-img-inline)").forEach(img => {{
                  const item = img.closest(".item") || img.parentElement || document.body;
                  const maxWidth = Math.max(1, Math.min(item.clientWidth || viewportWidth, viewportWidth));
                  if ((img.naturalWidth || 0) > maxWidth || img.getBoundingClientRect().width > maxWidth) {{
                    img.style.setProperty("width", `${{maxWidth}}px`, "important");
                    img.style.setProperty("max-width", "100%", "important");
                    img.style.setProperty("height", "auto", "important");
                  }}
                }});
              }});
            }}

            async function constrainOversizedLatex(page) {{
              await page.evaluate(() => {{
                const viewportWidth = window.innerWidth || 1280;
                document.querySelectorAll(".rir-latex").forEach(el => {{
                  const para = el.closest(".rir-para") || el.parentElement || document.body;
                  const maxWidth = Math.max(1, Math.min(para.clientWidth || viewportWidth, viewportWidth));
                  if (el.getBoundingClientRect().width > maxWidth || el.scrollWidth > maxWidth) {{
                    el.style.setProperty("display", "inline-block", "important");
                    el.style.setProperty("max-width", "100%", "important");
                    el.style.setProperty("overflow-x", "auto", "important");
                    el.style.setProperty("overflow-y", "hidden", "important");
                    el.style.setProperty("vertical-align", "middle", "important");
                  }}
                }});
              }});
            }}

            async function waitForKatexStable(page) {{
              await page.evaluate(async () => {{
                function snapshotKatex() {{
                  return Array.from(document.querySelectorAll(".katex")).map(el => {{
                    const rect = el.getBoundingClientRect();
                    return [
                      Math.round(rect.left * 10) / 10,
                      Math.round(rect.top * 10) / 10,
                      Math.round(rect.width * 10) / 10,
                      Math.round(rect.height * 10) / 10,
                      el.textContent || ""
                    ].join("|");
                  }}).join("\\n");
                }}
                let previous = snapshotKatex();
                let stableFrames = 0;
                for (let i = 0; i < 30; i += 1) {{
                  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                  const next = snapshotKatex();
                  if (next === previous) {{
                    stableFrames += 1;
                    if (stableFrames >= 2) return;
                  }} else {{
                    previous = next;
                    stableFrames = 0;
                  }}
                }}
              }});
            }}

            async function waitForRenderedContent(page) {{
              await page.evaluate(async () => {{
                if (document.fonts && document.fonts.ready) {{
                  try {{
                    await document.fonts.ready;
                  }} catch (_err) {{}}
                }}
              }});
              await waitForKatexStable(page);
              await waitForImages(page);
              await constrainOversizedImages(page);
              await constrainOversizedLatex(page);
              await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
              await page.waitForTimeout(300);
            }}

            async function findLatexSourceResidue(page) {{
              return await page.evaluate(() => {{
                const root = document.getElementById("root") || document.body;
                const text = root ? (root.innerText || "") : "";
                const re = /\\\\frac|\\\\ce\\{{|\\\\text\\{{/g;
                const matches = [];
                let match;
                while ((match = re.exec(text)) && matches.length < 20) {{
                  matches.push({{
                    match: match[0],
                    offset: match.index,
                    context: text.slice(Math.max(0, match.index - 50), match.index + 90)
                  }});
                }}
                return matches;
              }});
            }}

            async function main() {{
              fs.mkdirSync(screenshotDir, {{recursive:true}});
              const systemChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
              const launchOptions = {{headless:true}};
              if (fs.existsSync(systemChrome)) launchOptions.executablePath = systemChrome;
              const browser = await chromium.launch(launchOptions);
              const page = await browser.newPage({{viewport:{{width:1280,height:900}}, deviceScaleFactor:1}});
              await page.route("**/api/v4/assets/*.png", async route => {{
                const url = new URL(route.request().url());
                const m = url.pathname.match(/\\/api\\/v4\\/assets\\/([0-9a-f]{{64}})\\.png$/);
                const p = m ? assetPaths[m[1]] : null;
                if (p && fs.existsSync(p)) await route.fulfill({{path:p}});
                else await route.fulfill({{status:404, body:"missing"}});
              }});
              await page.route("**/api/v4/raw_assets**", async route => {{
                const url = new URL(route.request().url());
                const key = `${{url.searchParams.get("group_key") || ""}}||${{url.searchParams.get("media") || ""}}`;
                const p = rawAssetPaths[key];
                if (p && fs.existsSync(p)) await route.fulfill({{path:p}});
                else await route.fulfill({{status:404, body:"missing"}});
              }});
              await page.setContent(html(), {{waitUntil:"load"}});
              const outputs = {{}};
              for (const rec of records) {{
                outputs[rec.item_id] = {{}};
                for (const vp of [
                  {{name:"desktop", width:1280, height:900}},
                  {{name:"mobile", width:375, height:844}}
                ]) {{
                  await page.setViewportSize({{width:vp.width, height:vp.height}});
                  await page.evaluate(async (rec) => {{
                    const root = document.getElementById("root");
                    root.textContent = "";
                    const head = document.createElement("div");
                    head.className = "head";
                    head.textContent = rec.item_id;
                    root.appendChild(head);
                    for (const zone of ["stem", "answer"]) {{
                      const title = document.createElement("div");
                      title.className = "zone-title";
                      title.textContent = zone;
                      const body = document.createElement("div");
                      root.appendChild(title);
                      root.appendChild(body);
                      window.YHerRirRenderer.renderZone((rec.rir.zones && rec.rir.zones[zone]) || [], body, {{itemId:rec.item_id, disableReports:true}});
                    }}
                    await new Promise(resolve => requestAnimationFrame(resolve));
                  }}, rec);
                  const filename = `${{rec.item_id}}_${{vp.name}}.png`;
                  const p = path.join(screenshotDir, filename);
                  let residue = [];
                  for (let attempt = 0; attempt < 3; attempt += 1) {{
                    await waitForRenderedContent(page);
                    await page.screenshot({{path:p, fullPage:true}});
                    residue = await findLatexSourceResidue(page);
                    if (!residue.length) break;
                    if (attempt < 2) await page.waitForTimeout(300);
                  }}
                  outputs[rec.item_id][vp.name] = p;
                  if (residue.length) {{
                    outputs[rec.item_id].screenshot_fail = outputs[rec.item_id].screenshot_fail || {{}};
                    outputs[rec.item_id].screenshot_fail[vp.name] = residue;
                  }}
                }}
                if ((Object.keys(outputs).length % 10) === 0) {{
                  console.log(`rendered ${{Object.keys(outputs).length}}/${{records.length}} items`);
                }}
              }}
              await browser.close();
              fs.writeFileSync({json.dumps(str(result_path))}, JSON.stringify({{ok:true, screenshots:outputs}}, null, 2));
            }}

            main().catch(err => {{
              fs.writeFileSync({json.dumps(str(result_path))}, JSON.stringify({{ok:false, fatal:String(err && err.stack || err)}}, null, 2));
              process.exit(1);
            }});
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [node, str(script_path)],
        cwd=str(ROOT),
        env=node_env(out_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = read_json(result_path) if result_path.exists() else {}
    if proc.returncode != 0 or not result.get("ok"):
        raise RuntimeError(f"screenshot_render_failed rc={proc.returncode} stderr={proc.stderr[-1000:]} fatal={result.get('fatal')}")
    return result.get("screenshots") or {}


def image_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    return f"data:image/{mime};base64,{b64}"


def vision_keys(provider: str) -> List[str]:
    load_env_file()
    config = VISION_CONFIGS[provider]
    raw = os.environ.get(config["env_key"], "")
    keys = [x.strip() for x in re.split(r"[,\s]+", raw) if x.strip()]
    if not keys:
        raise RuntimeError(f"{config['env_key']} is not available in environment or .env")
    return keys


def build_vl_prompts() -> Tuple[str, str]:
    system = (
        "你是上海高中化学题面可用性审查员。只基于截图可见内容判断,禁止解题,禁止补图外信息。"
        "只依据截图文字内容判定,不臆测未渲染的 latex 源码。"
        "窄屏下文本/选项自然换行(word-wrap)是正常排版,不是缺陷;"
        "若内容未被截断丢失、未重叠遮挡、未错序,不得因此判 minor_issue 或 broken;"
        "如果唯一问题是窄屏换行/跨行/分行但内容仍完整可识别,verdict 必须是 usable;"
        "只有内容被截断丢失、重叠遮挡、错序才算排版问题。"
        "不确定时判 minor_issue 并说明不确定点。只返回 JSON。"
    )
    user = (
        "请同时查看 desktop 与 mobile 两张截图,判断学生是否能完整作答并核对答案区是否完整。"
        "按以下字段返回 JSON: "
        "{\"q1_stem_complete\":\"yes/no/minor\","
        "\"q2_visuals_clear\":\"yes/no/minor/not_applicable\","
        "\"q3_options_complete\":\"yes/no/minor/not_applicable\","
        "\"q4_answer_complete\":\"yes/no/minor\","
        "\"q5_layout_readable\":\"yes/no/minor\","
        "\"verdict\":\"usable/minor_issue/broken\","
        "\"reason\":\"一句中文依据\"}。"
        "verdict 规则: 无法作答、题干/图表/答案关键缺失、严重重叠乱码为 broken; "
        "可作答但有小排版或轻微信息问题为 minor_issue; 完整清楚为 usable。"
    )
    return system, user


def call_vl_two_images(
    *,
    provider: str,
    model: Optional[str],
    api_key: str,
    desktop: Path,
    mobile: Path,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    config = VISION_CONFIGS[provider]
    model_name = model or config["model"]
    system, user = build_vl_prompts()
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "text", "text": "desktop screenshot:"},
                    {"type": "image_url", "image_url": {"url": image_data_url(desktop)}},
                    {"type": "text", "text": "mobile screenshot:"},
                    {"type": "image_url", "image_url": {"url": image_data_url(mobile)}},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        config["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=150) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vision_http_error status={exc.code} body={body[:500]}") from exc
    content = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    usage = raw.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    pricing = config["pricing"]
    cost = (input_tokens * pricing["input"] / 1e6) + (output_tokens * pricing["output"] / 1e6)
    return {
        "provider": provider,
        "model": model_name,
        "content": content,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "cost_yuan": cost,
        "elapsed_sec": round(time.time() - started, 3),
    }


def normalize_vl(parsed: Dict[str, Any]) -> Dict[str, Any]:
    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in {"usable", "minor_issue", "broken"}:
        raw = json.dumps(parsed, ensure_ascii=False).lower()
        if "无法确定" in raw or "不确定" in raw:
            verdict = "minor_issue"
        elif "broken" in raw or "无法" in raw or "缺失" in raw:
            verdict = "broken"
        elif "minor" in raw or "不确定" in raw or "轻微" in raw:
            verdict = "minor_issue"
        else:
            verdict = "minor_issue"
    reason = str(parsed.get("reason") or parsed.get("依据") or parsed.get("raw_text") or "")[:500]
    return {
        "q1_stem_complete": str(parsed.get("q1_stem_complete") or parsed.get("q1") or "").strip(),
        "q2_visuals_clear": str(parsed.get("q2_visuals_clear") or parsed.get("q2") or "").strip(),
        "q3_options_complete": str(parsed.get("q3_options_complete") or parsed.get("q3") or "").strip(),
        "q4_answer_complete": str(parsed.get("q4_answer_complete") or parsed.get("q4") or "").strip(),
        "q5_layout_readable": str(parsed.get("q5_layout_readable") or parsed.get("q5") or "").strip(),
        "verdict": verdict,
        "reason": reason,
    }


def run_vl_audit(
    *,
    items: List[Dict[str, Any]],
    strata: Dict[str, str],
    screenshots: Dict[str, Dict[str, str]],
    out_dir: Path,
    provider: str,
    model: Optional[str],
    refresh: bool,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    keys = vision_keys(provider)
    cache_dir = out_dir / "vl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        item_id = item["item_id"]
        paths = screenshots.get(item_id) or {}
        desktop = Path(paths.get("desktop") or "")
        mobile = Path(paths.get("mobile") or "")
        if not desktop.exists() or not mobile.exists():
            raise RuntimeError(f"missing screenshots for {item_id}")
        cache_path = cache_dir / f"{item_id}.json"
        if cache_path.exists() and not refresh:
            raw = read_json(cache_path)
        else:
            raw = call_vl_two_images(
                provider=provider,
                model=model,
                api_key=keys[idx % len(keys)],
                desktop=desktop,
                mobile=mobile,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            write_json(cache_path, raw)
        parsed = parse_json_response(str(raw.get("content") or ""))
        normalized = normalize_vl(parsed)
        rows.append(
            {
                "schema_version": "qa2_vl_audit_v1",
                "item_id": item_id,
                "group_key": item.get("group_key", ""),
                "section_num": item.get("section_num"),
                "q_num": item.get("q_num"),
                "stratum": strata.get(item_id, ""),
                "reviewer": "",
                "review_status": "pending",
                "screenshot_desktop": str(desktop),
                "screenshot_mobile": str(mobile),
                "screenshot_fail": paths.get("screenshot_fail") or {},
                "provider": raw.get("provider", provider),
                "model": raw.get("model", model or VISION_CONFIGS[provider]["model"]),
                "usage": raw.get("usage") or {},
                "cost_yuan": float(raw.get("cost_yuan") or 0.0),
                "elapsed_sec": raw.get("elapsed_sec"),
                "raw_response": raw.get("content", ""),
                **normalized,
            }
        )
        if (idx + 1) % 10 == 0:
            print(f"VL audited {idx + 1}/{len(items)}", flush=True)
    write_jsonl(out_dir / "vl_audit.jsonl", rows)
    return rows


def merge_audits(machine_rows: List[Dict[str, Any]], vl_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    vl_by_id = {row["item_id"]: row for row in vl_rows}
    merged: List[Dict[str, Any]] = []
    for row in machine_rows:
        item_id = row["item_id"]
        vl = vl_by_id.get(item_id)
        if vl is None:
            raise RuntimeError(f"missing vl row for {item_id}")
        failed = list(row.get("machine_failed_dimensions") or [])
        issue_failed = [dim for dim in failed if dim in MACHINE_ISSUE_DIMENSIONS]
        verdict = vl.get("verdict")
        if not issue_failed and verdict == "usable":
            pool = "clean"
        elif verdict == "broken" or any(dim in BLOCKING_DIMENSIONS for dim in issue_failed):
            pool = "blocked"
        else:
            pool = "fixable"
        buckets = sorted({BUCKET_BY_DIMENSION.get(dim, "排版") for dim in issue_failed})
        if verdict == "minor_issue":
            buckets.append("排版")
        if verdict == "broken" and "排版" not in buckets:
            buckets.append("排版")
        merged.append(
            {
                "schema_version": "qa2_usability_audit_v1",
                "item_id": item_id,
                "group_key": row.get("group_key", ""),
                "section_num": row.get("section_num"),
                "q_num": row.get("q_num"),
                "stratum": row.get("stratum", ""),
                "reviewer": "",
                "review_status": "pending",
                "pool": pool,
                "machine_pass_count": row.get("machine_pass_count"),
                "machine_failed_dimensions": failed,
                "machine_issue_dimensions": issue_failed,
                "vl_verdict": verdict,
                "vl_reason": vl.get("reason", ""),
                "screenshot_fail": vl.get("screenshot_fail") or {},
                "bucket": sorted(set(buckets)),
                "evidence": {
                    "machine": {dim: (row.get("evidence") or {}).get(dim, []) for dim in issue_failed},
                    "vl": {
                        "q1_stem_complete": vl.get("q1_stem_complete", ""),
                        "q2_visuals_clear": vl.get("q2_visuals_clear", ""),
                        "q3_options_complete": vl.get("q3_options_complete", ""),
                        "q4_answer_complete": vl.get("q4_answer_complete", ""),
                        "q5_layout_readable": vl.get("q5_layout_readable", ""),
                        "reason": vl.get("reason", ""),
                    },
                },
            }
        )
    return merged


def write_summary(
    out_dir: Path,
    items: List[Dict[str, Any]],
    machine_rows: List[Dict[str, Any]],
    vl_rows: List[Dict[str, Any]],
    usability_rows: List[Dict[str, Any]],
    elapsed_sec: float,
    *,
    all_service: bool = False,
) -> None:
    pool_counts = Counter(row["pool"] for row in usability_rows)
    bucket_counts: Counter = Counter()
    failed_counts: Counter = Counter()
    group_counts: Counter = Counter()
    strata_counts = Counter(row.get("stratum", "") for row in usability_rows)
    for row in usability_rows:
        if row["pool"] != "clean":
            group_counts[row.get("group_key", "")] += 1
        for b in row.get("bucket") or []:
            bucket_counts[b] += 1
        for dim in row.get("machine_failed_dimensions") or []:
            failed_counts[dim] += 1
    vl_counts = Counter(row.get("verdict") for row in vl_rows)
    cost = round(sum(float(row.get("cost_yuan") or 0.0) for row in vl_rows), 4)
    screenshot_count = len(list((out_dir / "screenshots").glob("*.png"))) if (out_dir / "screenshots").exists() else 0
    screenshot_fail_rows = sum(1 for row in vl_rows if row.get("screenshot_fail"))
    reviewer_bad = sum(1 for row in [*machine_rows, *vl_rows, *usability_rows] if row.get("reviewer") not in ("", None))
    status_bad = sum(1 for row in [*machine_rows, *vl_rows, *usability_rows] if row.get("review_status") != "pending")
    run_scope = "full 2526 service pool" if all_service else "120-item calibration set"
    full_pool_status = "completed after Claude calibration approval" if all_service else "not run; waiting for Claude threshold approval after this calibration package"
    strata_line = "`not applicable`" if all_service else f"`{dict(strata_counts)}`"

    summary = f"""# Batch 11 QA-2 Usability Summary

## Scope

- Items audited: {len(usability_rows)}
- Service-pool source: `core/data/item_bank_v4.py:iter_service_items()`
- Run scope: {run_scope}
- Calibration strata: {strata_line}
- Full pool status: {full_pool_status}

## Three Pools

| Pool | Count |
|---|---:|
| clean | {pool_counts.get('clean', 0)} |
| fixable | {pool_counts.get('fixable', 0)} |
| blocked | {pool_counts.get('blocked', 0)} |

## Machine Audit

- Dimensions per item: {len(DIMENSIONS)}
- Failed dimension counts: `{dict(failed_counts)}`

## VL Audit

- Verdict counts: `{dict(vl_counts)}`
- Screenshot files retained: {screenshot_count}
- Screenshot fail rows: {screenshot_fail_rows}
- Measured VL cost yuan: {cost}

## Bucket Distribution

`{dict(bucket_counts)}`

## Top Problem Groups

"""
    for group, count in group_counts.most_common(20):
        summary += f"- `{group}`: {count}\n"
    summary += f"""
## Governance Checks

- reviewer nonempty rows: {reviewer_bad}
- review_status not pending rows: {status_bad}
- output root: `{out_dir}`
"""
    (out_dir / "USABILITY_SUMMARY.md").write_text(summary, encoding="utf-8")

    report_title = "Batch 11 QA-2 Full Pool Report" if all_service else "Batch 11 QA-2 Calibration Report"
    result_line = "Completed 11a + 11b + 11c for the full service pool." if all_service else "Completed 11a + 11b + 11c for the 120-item calibration set only."
    gate_line = "Full 2526 pool was run after Claude/user approval." if all_service else "Full 2526 pool was not run; it is gated on Claude's calibration threshold approval."
    report = f"""# {report_title}

## Result

{result_line}

- machine_audit rows: {len(machine_rows)}
- vl_audit rows: {len(vl_rows)}
- usability_audit rows: {len(usability_rows)}
- pool counts: `{dict(pool_counts)}`
- VL cost yuan: {cost}
- screenshot_fail rows: {screenshot_fail_rows}
- elapsed seconds: {round(elapsed_sec, 1)}

## Discipline

- Official data was read-only.
- No `core/render` or `apps/web` files were modified.
- All ledger rows have `reviewer=""` and `review_status="pending"`.
- {gate_line}

## Artifacts

- `{out_dir / 'machine_audit.jsonl'}`
- `{out_dir / 'vl_audit.jsonl'}`
- `{out_dir / 'usability_audit.jsonl'}`
- `{out_dir / 'USABILITY_SUMMARY.md'}`
- `{out_dir / 'screenshots'}`
- `{out_dir / 'vl_cache'}`
"""
    (out_dir / "BATCH11_REPORT.md").write_text(report, encoding="utf-8")


def validate_governance(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    return {
        "rows": len(rows),
        "reviewer_nonempty": sum(1 for row in rows if row.get("reviewer") not in ("", None)),
        "review_status_not_pending": sum(1 for row in rows if row.get("review_status") != "pending"),
        "codex_reviewer_hits": sum(1 for row in rows if "codex" in str(row.get("reviewer") or "").lower()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Batch 11 QA-2 usability audit.")
    parser.add_argument("--input-jsonl", type=Path, default=CALIBRATION_INPUT)
    parser.add_argument("--all-service", action="store_true", help="Run all 2526 service items. Use only after Claude approves calibration thresholds.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--provider", default="qwen-vl", choices=sorted(VISION_CONFIGS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--node-bin", default=None)
    parser.add_argument("--refresh-vl", action="store_true")
    parser.add_argument("--skip-vl", action="store_true", help="Developer-only: write machine audit without VL/merge.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=700)
    args = parser.parse_args()

    started = time.time()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = OUT_ROOT / ("full_pool" if args.all_service else "calibration_120")
    out_dir.mkdir(parents=True, exist_ok=True)

    items, strata = load_target_items(args.input_jsonl if not args.all_service else None, args.all_service)
    records = build_rir_records(items)
    write_json(out_dir / "rir_records.json", records)

    machine_rows = run_machine_audit(items, strata, records, out_dir, args.node_bin)
    if args.skip_vl:
        print(json.dumps({"machine_rows": len(machine_rows), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
        return 0

    screenshots = render_screenshots(out_dir=out_dir, records=records, node_bin=args.node_bin)
    write_json(out_dir / "screenshot_manifest.json", screenshots)
    vl_rows = run_vl_audit(
        items=items,
        strata=strata,
        screenshots=screenshots,
        out_dir=out_dir,
        provider=args.provider,
        model=args.model,
        refresh=args.refresh_vl,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    usability_rows = merge_audits(machine_rows, vl_rows)
    write_jsonl(out_dir / "usability_audit.jsonl", usability_rows)
    write_summary(out_dir, items, machine_rows, vl_rows, usability_rows, time.time() - started, all_service=args.all_service)
    governance = validate_governance([*machine_rows, *vl_rows, *usability_rows])
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "items": len(items),
                "machine_rows": len(machine_rows),
                "vl_rows": len(vl_rows),
                "usability_rows": len(usability_rows),
                "pool_counts": dict(Counter(row["pool"] for row in usability_rows)),
                "vl_cost_yuan": round(sum(float(row.get("cost_yuan") or 0.0) for row in vl_rows), 4),
                "governance": governance,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if governance["reviewer_nonempty"] == 0 and governance["review_status_not_pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
