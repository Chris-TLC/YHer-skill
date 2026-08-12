#!/usr/bin/env python3
"""Validate and route model-written gold diagnostic question candidates.

The router keeps model drafts out of production diagnosis/profile flows. It
splits candidates into approved, revise, and reject JSONL files with explicit
blocker reasons.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_IN = Path("/tmp/yher_gold_question_candidates.jsonl")
DEFAULT_OUT_DIR = SKILL_DIR / "data" / "quality" / "gold_question_candidates"

HARD_HOLES = {"solution_three_balances", "process_flow", "integrated_experiment"}
AXES = {"entry", "concept", "procedure", "transfer", "integrated"}
DIFFICULTIES = {"T1", "T2", "T3", "T4"}
ANSWER_TYPES = {"single_choice", "free_text", "multi_step"}
PROFILE_AXES = {"基础概念", "审题入口", "步骤执行", "应用迁移", "综合推理"}
MAX_WEIGHTS = {"low", "medium", "high"}
REQUIRED_FIELDS = {
    "gold_id",
    "hard_hole",
    "kg_node",
    "diagnostic_axis",
    "difficulty",
    "answer_type",
    "prompt",
    "options",
    "standard_answer",
    "rubric",
    "misconceptions",
    "profile_evidence_rule",
    "verification_use",
    "source_type",
    "review_status",
    "risk_notes",
}
FORBIDDEN_PROMPT_PATTERNS = [
    "如图",
    "见图",
    "下图",
    "上图",
    "图中",
    "所示装置图",
    "根据装置图",
    "见表",
    "如下表",
    "截图",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"_invalid_json": True, "_line": line_no, "_error": str(exc)})
                continue
            row.setdefault("_line", line_no)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_forbidden_prompt(prompt: str) -> bool:
    return any(pattern in prompt for pattern in FORBIDDEN_PROMPT_PATTERNS)


def validate_options(row: dict[str, Any], blockers: list[str], warnings: list[str]) -> None:
    answer_type = row.get("answer_type")
    options = row.get("options")
    if answer_type == "single_choice":
        if not isinstance(options, dict) or set(options.keys()) != {"A", "B", "C", "D"}:
            blockers.append("single_choice_requires_A_B_C_D_options")
        standard = str(row.get("standard_answer") or "").strip().upper()
        if not re.fullmatch(r"[A-D]", standard):
            blockers.append("single_choice_standard_answer_must_be_one_option")
    else:
        if options not in (None, {}) and not isinstance(options, dict):
            blockers.append("options_must_be_null_or_object")
        if options:
            warnings.append("non_choice_question_has_options")


def validate_rubric(row: dict[str, Any], blockers: list[str]) -> None:
    rubric = row.get("rubric")
    if not isinstance(rubric, list) or not rubric:
        blockers.append("rubric_missing")
        return
    if not any(point.get("must_have") is True for point in rubric if isinstance(point, dict)):
        blockers.append("rubric_missing_must_have_point")
    for index, point in enumerate(rubric):
        if not isinstance(point, dict):
            blockers.append(f"rubric_{index}_not_object")
            continue
        for field in {"point_id", "desc", "must_have", "score", "accept_patterns", "reject_patterns"}:
            if field not in point:
                blockers.append(f"rubric_{index}_missing_{field}")
        if not isinstance(point.get("accept_patterns"), list):
            blockers.append(f"rubric_{index}_accept_patterns_not_list")
        if not isinstance(point.get("reject_patterns"), list):
            blockers.append(f"rubric_{index}_reject_patterns_not_list")


def validate_misconceptions(row: dict[str, Any], blockers: list[str]) -> None:
    misconceptions = row.get("misconceptions")
    if not isinstance(misconceptions, list) or not misconceptions:
        blockers.append("misconceptions_missing")
        return
    for index, item in enumerate(misconceptions):
        if not isinstance(item, dict):
            blockers.append(f"misconception_{index}_not_object")
            continue
        for field in {"wrong_pattern", "reveals", "profile_update", "recommended_remediation"}:
            if not item.get(field):
                blockers.append(f"misconception_{index}_missing_{field}")
        profile_update = item.get("profile_update") or {}
        if not isinstance(profile_update, dict):
            blockers.append(f"misconception_{index}_profile_update_not_object")
            continue
        if profile_update.get("axis") not in PROFILE_AXES:
            blockers.append(f"misconception_{index}_profile_axis_invalid")
        if profile_update.get("direction") != "weaken":
            blockers.append(f"misconception_{index}_profile_direction_must_weaken")


def validate_profile_rule(row: dict[str, Any], blockers: list[str], warnings: list[str]) -> None:
    rule = row.get("profile_evidence_rule")
    if not isinstance(rule, dict):
        blockers.append("profile_evidence_rule_missing")
        return
    if rule.get("can_update_profile") is not True:
        blockers.append("profile_rule_can_update_profile_must_be_true_for_gold_candidate")
    if rule.get("max_weight") not in MAX_WEIGHTS:
        blockers.append("profile_rule_max_weight_invalid")
    elif rule.get("max_weight") == "high" and row.get("source_type") == "model_candidate":
        warnings.append("model_candidate_high_profile_weight_needs_human_or_empirical_review")
    for field in {"mastery_signal", "weakness_signal"}:
        if not str(rule.get(field) or "").strip():
            blockers.append(f"profile_rule_missing_{field}")


def validate_candidate(row: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []

    if row.get("_invalid_json"):
        return "reject", ["invalid_json"], []

    missing = sorted(REQUIRED_FIELDS - set(row.keys()))
    blockers.extend(f"missing_field:{field}" for field in missing)
    if blockers:
        return "reject", blockers, warnings

    if row.get("hard_hole") not in HARD_HOLES:
        blockers.append("hard_hole_invalid")
    if row.get("diagnostic_axis") not in AXES:
        blockers.append("diagnostic_axis_invalid")
    if row.get("difficulty") not in DIFFICULTIES:
        blockers.append("difficulty_invalid")
    if row.get("answer_type") not in ANSWER_TYPES:
        blockers.append("answer_type_invalid")
    if row.get("source_type") != "model_candidate":
        blockers.append("source_type_must_be_model_candidate")
    if row.get("review_status") != "silver_candidate":
        blockers.append("review_status_must_be_silver_candidate")
    if not isinstance(row.get("risk_notes"), list):
        blockers.append("risk_notes_must_be_list")

    prompt = str(row.get("prompt") or "").strip()
    if len(prompt) < 12:
        blockers.append("prompt_too_short")
    if has_forbidden_prompt(prompt):
        blockers.append("prompt_has_external_visual_dependency")
    if not str(row.get("standard_answer") or "").strip():
        blockers.append("standard_answer_missing")
    if row.get("verification_use") != ["diagnosis", "post_video_verification"]:
        blockers.append("verification_use_must_be_diagnosis_and_post_video_verification")

    validate_options(row, blockers, warnings)
    validate_rubric(row, blockers)
    validate_misconceptions(row, blockers)
    validate_profile_rule(row, blockers, warnings)

    if blockers:
        if any(
            blocker.startswith("missing_field:")
            or blocker in {
                "invalid_json",
                "hard_hole_invalid",
                "prompt_has_external_visual_dependency",
                "rubric_missing",
                "misconceptions_missing",
            }
            for blocker in blockers
        ):
            return "reject", sorted(set(blockers)), sorted(set(warnings))
        return "revise", sorted(set(blockers)), sorted(set(warnings))
    if warnings:
        return "revise", [], sorted(set(warnings))
    return "approved", [], []


def route_candidates(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    routed = {"approved": [], "revise": [], "reject": []}
    for row in rows:
        status, blockers, warnings = validate_candidate(row)
        routed[status].append(
            {
                **row,
                "route_status": status,
                "route_blockers": blockers,
                "route_warnings": warnings,
                "production_profile_evidence_allowed": False,
                "approved_label": "gold_v1_model_reviewed" if status == "approved" else "",
            }
        )

    summary = {
        "total": len(rows),
        "approved": len(routed["approved"]),
        "revise": len(routed["revise"]),
        "reject": len(routed["reject"]),
        "by_hard_hole": dict(Counter(str(row.get("hard_hole", "")) for row in rows if row.get("hard_hole"))),
        "top_blockers": dict(Counter(blocker for group in routed.values() for row in group for blocker in row["route_blockers"]).most_common(20)),
    }
    return routed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Route gold diagnostic question candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--write", action="store_true", help="Write approved/revise/reject JSONL files. Default is dry-run.")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    routed, summary = route_candidates(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, group in routed.items():
            write_jsonl(args.out_dir / f"{name}.jsonl", group)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out_dir}")
    else:
        print("DRY RUN: pass --write to write routed candidate files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
