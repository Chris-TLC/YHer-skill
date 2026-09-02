#!/usr/bin/env python3
"""Apply the signed 2026-07-13 Demo bad-report resolutions exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable


REVIEWER = "codex_sol_20260713"
RESOLVED_ON = "2026-07-13"

EXPECTED_HASHES = {
    "data/study_logs/bad_reports.jsonl": "e784ddbafde30067494107cd52a911ae4eec9e8e3c10578f8302d4ae512aa85b",
    "data/item_bank/v4/usability_r5_v1.jsonl": "1cfd7cef1b99f6cf77d8006640888dd5eb15e560bedc719f768a85789ea2ca96",
    "data/item_bank/v4/chemistry_v4_1_3329.jsonl": "a09ad7154f9fc35d8278c06ac4a5ca6e4a132c1604683eb280bb281eb5b4536d",
}

REPORT_DECISIONS = {
    "92e023f3ab0d3a87cc2492ae13717a20b87222ca": (
        "false_positive_closed",
        "误报：RIR 无降级，Ni 题面与 3d⁸4s² 答案、解析一致；原 note 明示端到端测试。",
    ),
    "51709f8e4716934d54069af9365674b824d9b551": (
        "false_positive_closed",
        "误报：完整单选题不依赖图片，答案 C 与解析一致。",
    ),
    "3615815efe1c9ed277fa17dc0231815e64580679": (
        "false_positive_closed",
        "误报：Cl₂、MnO₂ 公式资产均可转为 LaTeX，RIR 无降级，答案 A 与解析一致。",
    ),
    "a30cff9246adf4f720454d3120f83773b7f549fa": (
        "r5_excluded",
        "排除：孤立子问“连续操作”缺少前文流程，answer_blocks 还错配 NaNO₃，不能独立作答。",
    ),
    "9fd28aff8ee44c7e813781e00e41ac032994a83d": (
        "r5_excluded",
        "排除：题面引用结构 c 但缺少结构或母题上下文，且 standard_solution 缺失。",
    ),
    "10eb15f4b9466a829751dd59f9ef34398e065b56": (
        "r5_excluded",
        "排除：题面引用滤渣2、除杂2但缺流程上下文，answer_blocks 仅余 Fe(OH) 残片。",
    ),
    "d544a7850e7c0c40382b16f486d281294f9d9da8": (
        "r5_excluded",
        "排除：题面引用系列操作②但缺流程上下文，不能独立作答。",
    ),
    "216abbd2c77fd2bb7d0b0d6b228be3125b45066f": (
        "r5_excluded",
        "排除：题目要求在方程式上标电子转移，当前文本输入无法忠实作答和确定性判分。",
    ),
    "201be664b516b614e40969e0f77cc4320bda8bc4": (
        "official_repaired",
        "修复：按原 DOCX 中三个 w:u=single 空白 run 恢复可见下划线。",
    ),
    "4cdec6c6cdd5574b228a85c3548235af272adf1a": (
        "official_repaired",
        "修复：answer_blocks 恢复为 Cu⁺；孤电子对；配位，与解析和 standard_solution 一致。",
    ),
}

R5_EXCLUSIONS = {
    "a30cff9246adf4f720454d3120f83773b7f549fa": "exclusion:hollow_content",
    "9fd28aff8ee44c7e813781e00e41ac032994a83d": "exclusion:hollow_content",
    "10eb15f4b9466a829751dd59f9ef34398e065b56": "exclusion:partial_answer",
    "d544a7850e7c0c40382b16f486d281294f9d9da8": "exclusion:hollow_content",
    "216abbd2c77fd2bb7d0b0d6b228be3125b45066f": "exclusion:non_text_response_required",
}

BLANK_ITEM_ID = "201be664b516b614e40969e0f77cc4320bda8bc4"
ANSWER_ITEM_ID = "4cdec6c6cdd5574b228a85c3548235af272adf1a"
EXPECTED_BLANK_STEM = (
    "28.若只考虑①②反应，生成物为Fe和FeO，向0.1g Fe和FeO的混合物中加入过量稀盐酸配成溶液B，"
    "放出气体22.4mL（标准状况），求混合物FeO的质量分数为             。隔绝空气，向溶液B中加入"
    "过量NaOH溶液，有沉淀生成，沉淀颜色为            。写出在空气中该沉淀的反应方程式："
    "                                              。"
)
REPAIRED_BLANK_STEM = (
    "28.若只考虑①②反应，生成物为Fe和FeO，向0.1g Fe和FeO的混合物中加入过量稀盐酸配成溶液B，"
    "放出气体22.4mL（标准状况），求混合物FeO的质量分数为_____________。隔绝空气，向溶液B中加入"
    "过量NaOH溶液，有沉淀生成，沉淀颜色为____________。写出在空气中该沉淀的反应方程式："
    "______________________________________________。"
)
EXACT_COORDINATION_ANSWER = "Cu⁺；孤电子对；配位"
SOURCE_SUFFIX = (
    "; bad_report_resolution 2026-07-13 "
    "(codex_sol_20260713; double-gold precision 0/236, recall 55/55)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_jsonl(
    path: Path,
    targets: set[str],
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[bytes, set[str]]:
    output: list[bytes] = []
    seen: set[str] = set()
    for raw in path.read_bytes().splitlines(keepends=True):
        row = json.loads(raw)
        item_id = str(row.get("item_id") or "")
        if item_id not in targets:
            output.append(raw)
            continue
        if item_id in seen:
            raise RuntimeError(f"duplicate target row: {item_id}")
        seen.add(item_id)
        changed = transform(dict(row))
        output.append(
            (json.dumps(changed, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        )
    if seen != targets:
        raise RuntimeError(f"target mismatch for {path}: missing={sorted(targets - seen)}")
    return b"".join(output), seen


def _resolve_report(row: dict[str, Any]) -> dict[str, Any]:
    item_id = row["item_id"]
    if row.get("status") != "pending" or row.get("reviewer") not in ("", None):
        raise RuntimeError(f"report is not pending and unsigned: {item_id}")
    decision, resolution = REPORT_DECISIONS[item_id]
    row.update(
        {
            "decision": decision,
            "resolution": resolution,
            "resolved_on": RESOLVED_ON,
            "reviewer": REVIEWER,
            "status": "resolved",
        }
    )
    return row


def _exclude_r5(row: dict[str, Any]) -> dict[str, Any]:
    item_id = row["item_id"]
    if row.get("r5_serve") is not True or row.get("r5_block_reason") not in (None, ""):
        raise RuntimeError(f"R5 row is not currently serveable: {item_id}")
    row.update(
        {
            "r5_block_reason": R5_EXCLUSIONS[item_id],
            "r5_serve": False,
            "review_status": "approved",
            "reviewer": REVIEWER,
            "source": str(row.get("source") or "") + SOURCE_SUFFIX,
        }
    )
    return row


def _repair_item(row: dict[str, Any]) -> dict[str, Any]:
    item_id = row["item_id"]
    if item_id == BLANK_ITEM_ID:
        if row.get("stem_text") != EXPECTED_BLANK_STEM:
            raise RuntimeError("blank-line item stem_text no longer matches audited input")
        blocks = row.get("stem_blocks") or []
        if blocks[0]["para"][0].get("text") != EXPECTED_BLANK_STEM:
            raise RuntimeError("blank-line item stem_blocks no longer match audited input")
        row["stem_text"] = REPAIRED_BLANK_STEM
        blocks[0]["para"][0]["text"] = REPAIRED_BLANK_STEM
        row["stem_blocks"] = blocks
        return row
    if item_id == ANSWER_ITEM_ID:
        standard = (row.get("standard_solution") or {}).get("standard_answer")
        if standard != EXACT_COORDINATION_ANSWER:
            raise RuntimeError("coordination standard answer no longer matches audited evidence")
        if row.get("answer_blocks_effective") != [
            {"para": [{"text": "【答案】000g=", "type": "text"}]}
        ]:
            raise RuntimeError("coordination answer block no longer matches audited input")
        row["answer_blocks_effective"] = [
            {"para": [{"text": EXACT_COORDINATION_ANSWER, "type": "text"}]}
        ]
        return row
    raise RuntimeError(f"unexpected repair target: {item_id}")


def apply(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    paths = {relative: workspace / relative for relative in EXPECTED_HASHES}
    observed = {relative: _sha256(path) for relative, path in paths.items()}
    if observed != EXPECTED_HASHES:
        raise RuntimeError(f"official input hash mismatch: {observed}")

    rendered = {
        "data/study_logs/bad_reports.jsonl": _render_jsonl(
            paths["data/study_logs/bad_reports.jsonl"],
            set(REPORT_DECISIONS),
            _resolve_report,
        )[0],
        "data/item_bank/v4/usability_r5_v1.jsonl": _render_jsonl(
            paths["data/item_bank/v4/usability_r5_v1.jsonl"],
            set(R5_EXCLUSIONS),
            _exclude_r5,
        )[0],
        "data/item_bank/v4/chemistry_v4_1_3329.jsonl": _render_jsonl(
            paths["data/item_bank/v4/chemistry_v4_1_3329.jsonl"],
            {BLANK_ITEM_ID, ANSWER_ITEM_ID},
            _repair_item,
        )[0],
    }

    temporary_paths: dict[str, Path] = {}
    try:
        for relative, payload in rendered.items():
            target = paths[relative]
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
            temporary_paths[relative] = temporary
        for relative, temporary in temporary_paths.items():
            temporary.replace(paths[relative])
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)

    return {
        "reports_resolved": len(REPORT_DECISIONS),
        "r5_excluded": len(R5_EXCLUSIONS),
        "items_repaired": 2,
        "after_sha256": {relative: _sha256(path) for relative, path in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.workspace), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
