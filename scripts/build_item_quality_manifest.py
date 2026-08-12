#!/usr/bin/env python3
"""Build item_quality_manifest.jsonl from item bank and visual asset manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_ITEM_BANK = SKILL_DIR / "data" / "item_bank" / "chemistry_v3_6695.jsonl"
DEFAULT_VISUAL_MANIFEST = SKILL_DIR / "data" / "quality" / "visual_asset_manifest.jsonl"
DEFAULT_UNDERSTANDING_RESULTS = SKILL_DIR / "data" / "evals" / "visual_understanding_results.jsonl"
DEFAULT_ANSWER_VERIFICATION_OVERRIDES = SKILL_DIR / "data" / "quality" / "answer_verification_overrides.jsonl"
DEFAULT_QUALITY_DIR = SKILL_DIR / "data" / "quality"
DEFAULT_DISPLAY_ANSWER_LEAK_QUARANTINE = (
    DEFAULT_QUALITY_DIR / "answer_leak_gate_20260703" / "quarantine_list.txt"
)

IMAGE_KEYWORDS = [
    "如图",
    "下图",
    "上图",
    "图示",
    "图中",
    "所示",
    "示意图",
    "装置图",
    "流程图",
    "曲线图",
    "坐标图",
    "结构图",
    "晶胞",
    "合成路线",
    "转化关系",
    "能量变化图",
    "滴定曲线",
    "实验装置",
    "工艺流程",
    "[图示",
    "【图",
]

ANSWER_VERIFIED_OVERRIDE_DECISIONS = {
    "standard_answer_verified",
    "model_answer_equivalent_or_better",
    "answer_equivalent_verified",
}


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


def item_needs_image(item: dict[str, Any], visual_status: dict[str, Any] | None) -> bool:
    if visual_status:
        return True
    text = "\n".join(
        [
            str(item.get("stem", "")),
            json.dumps(item.get("options", {}), ensure_ascii=False),
        ]
    )
    return any(keyword in text for keyword in IMAGE_KEYWORDS)


def is_valid_answer_verification_override(row: dict[str, Any]) -> bool:
    return bool(
        row.get("item_id")
        and row.get("answer_status") == "verified"
        and row.get("review_decision") in ANSWER_VERIFIED_OVERRIDE_DECISIONS
        and row.get("review_source")
    )


def load_answer_verification_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(Path(path)):
        if is_valid_answer_verification_override(row):
            overrides[str(row["item_id"])] = row
    return overrides


def normalized_path_keys(path_value: str | Path | None) -> set[str]:
    if not path_value:
        return set()
    raw = str(path_value)
    path = Path(raw).expanduser()
    keys = {raw, str(path)}
    if not path.is_absolute():
        keys.add(str((SKILL_DIR / path).resolve(strict=False)))
    keys.add(str(path.resolve(strict=False)))
    return keys


def load_display_answer_leak_quarantine(path: Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    quarantined: set[str] = set()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            quarantined.update(normalized_path_keys(value))
    return quarantined


def display_has_answer_leak(visual: dict[str, Any] | None, quarantined_paths: set[str]) -> bool:
    if not visual or not quarantined_paths:
        return False
    return bool(normalized_path_keys(display_image_path(visual)) & quarantined_paths)


def answer_status(item: dict[str, Any], answer_override: dict[str, Any] | None = None) -> str:
    solution = item.get("standard_solution") or {}
    if solution.get("standard_answer") or solution.get("final_answers"):
        if item.get("verification_status", "passed") == "passed":
            return "verified"
        return "verified" if answer_override else "suspect"
    return "missing"


def rubric_status(item: dict[str, Any]) -> str:
    rubric = item.get("rubric") or []
    if not rubric:
        return "missing"
    if any(point.get("must_have") for point in rubric):
        return "complete"
    return "partial"


def has_image_evidence(visual: dict[str, Any] | None) -> bool:
    if not visual:
        return False
    return bool(
        (visual.get("crop_path") and visual.get("crop_hash"))
        or (visual.get("page_image_path") and visual.get("page_image_hash"))
    )


def has_crop_evidence_for_strong(visual: dict[str, Any] | None) -> bool:
    if not visual:
        return False
    return bool(
        visual.get("crop_path")
        and visual.get("crop_hash")
        and visual.get("crop_tier") in {"item_crop", "item_crop_candidate"}
    )


def has_transcript_evidence_for_strong(understanding: dict[str, Any] | None) -> bool:
    if not understanding:
        return False
    if understanding.get("transcript_supported_strong"):
        return True
    return understanding.get("visual_evidence_mode") == "structured_transcript"


def visual_evidence_mode(visual: dict[str, Any] | None, understanding: dict[str, Any] | None) -> str:
    if has_crop_evidence_for_strong(visual):
        return "crop"
    if has_transcript_evidence_for_strong(understanding):
        return "structured_transcript"
    if visual and visual.get("page_image_path") and visual.get("page_image_hash"):
        return "page"
    return "missing"


def display_image_path(visual: dict[str, Any] | None) -> str:
    if not visual:
        return ""
    return str(visual.get("crop_path") or visual.get("page_image_path") or "")


def display_image_hash(visual: dict[str, Any] | None) -> str:
    if not visual:
        return ""
    return str(visual.get("crop_hash") or visual.get("page_image_hash") or "")


def understanding_status(understanding: dict[str, Any] | None, visual_asset_status: str) -> str:
    if understanding and understanding.get("understanding_pass") and not understanding.get("error_types"):
        return "strong"
    if understanding and understanding.get("understanding_pass"):
        return "strong"
    if understanding:
        error_types = set(understanding.get("error_types") or [])
        if error_types and error_types != {"not_evaluated_offline"}:
            return "reject"
        return "weak"
    return "weak" if visual_asset_status in {"strong", "weak"} else "reject"


def visual_pipeline_stage(
    needs_image: bool,
    visual: dict[str, Any] | None,
    ans_status: str,
    rub_status: str,
    readability_status: str,
    student_readable: bool,
    strong: bool,
    understanding: dict[str, Any] | None,
) -> str:
    if not needs_image:
        return "text_ready" if ans_status == "verified" and rub_status in {"complete", "partial"} else "text_review"
    if strong:
        return "strong"
    if understanding and student_readable:
        if understanding.get("visible_pass"):
            if understanding.get("answer_match"):
                return "strong_candidate"
            return "vl_answerable"
        return "student_readable"
    if student_readable:
        return "student_readable"
    if visual and readability_status == "pass" and has_image_evidence(visual):
        return "crop_readable"
    if visual and visual.get("page_image_path") and visual.get("page_image_hash"):
        return "asset_linked"
    return "raw_visual_item"


def review_queue_for(
    needs_image: bool,
    student_readable: bool,
    strong: bool,
    blockers: list[str],
    llm_status: str,
    ans_status: str,
) -> str:
    if not needs_image:
        return "none" if not blockers else "manual_review"
    if strong:
        return "none"
    hard_quarantine = {
        "missing_visual_asset_manifest",
        "missing_page_image",
        "visual_asset_rejected",
        "source_unresolved",
        "answer_missing",
    }
    if hard_quarantine & set(blockers) or ans_status == "missing":
        return "quarantine"
    if student_readable:
        return "strong_review"
    return "manual_review"


def strong_blockers_for(
    needs_image: bool,
    visual: dict[str, Any] | None,
    understanding: dict[str, Any] | None,
    visual_asset_status: str,
    readability_status: str,
    llm_status: str,
    ans_status: str,
    rub_status: str,
    blockers: list[str],
) -> list[str]:
    if not needs_image:
        reasons: list[str] = []
        if ans_status != "verified":
            reasons.append("answer_not_verified")
        if rub_status == "missing":
            reasons.append("rubric_missing")
        if blockers:
            reasons.append("blocker_reasons_present")
        return sorted(set(reasons))

    reasons = []
    if visual_asset_status != "strong":
        reasons.append("visual_asset_not_strong")
    if not (visual or {}).get("source_file"):
        reasons.append("missing_source_file")
    if (visual or {}).get("declared_page") is None:
        reasons.append("missing_source_page")
    if not (visual or {}).get("page_image_hash"):
        reasons.append("missing_page_image_hash")
    if readability_status != "pass":
        reasons.append("readability_not_pass")
    if not has_crop_evidence_for_strong(visual):
        if not has_transcript_evidence_for_strong(understanding):
            reasons.append("missing_crop_evidence_for_strong")
    if llm_status != "strong":
        reasons.append("llm_understanding_not_strong")
    if not understanding or not understanding.get("model"):
        reasons.append("missing_vl_result")
    if ans_status != "verified":
        reasons.append("answer_not_verified")
    if rub_status == "missing":
        reasons.append("rubric_missing")
    if blockers:
        reasons.append("blocker_reasons_present")
    return sorted(set(reasons))


def quality_evidence_id(
    item_id: str,
    visual: dict[str, Any] | None,
    understanding: dict[str, Any] | None,
    stage: str,
) -> str:
    payload = {
        "item_id": item_id,
        "stage": stage,
        "page_image_hash": (visual or {}).get("page_image_hash", ""),
        "crop_hash": (visual or {}).get("crop_hash", ""),
        "vl_model": (understanding or {}).get("model", ""),
        "vl_answer_match": (understanding or {}).get("answer_match"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"iq:{digest}"


def build_quality_manifest(
    item_bank_path: Path = DEFAULT_ITEM_BANK,
    visual_manifest_path: Path = DEFAULT_VISUAL_MANIFEST,
    understanding_results_path: Path = DEFAULT_UNDERSTANDING_RESULTS,
    answer_verification_overrides_path: Path | None = DEFAULT_ANSWER_VERIFICATION_OVERRIDES,
    display_answer_leak_quarantine_path: Path | None = DEFAULT_DISPLAY_ANSWER_LEAK_QUARANTINE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = load_jsonl(Path(item_bank_path))
    visual_by_id = {
        row.get("item_id"): row
        for row in load_jsonl(Path(visual_manifest_path))
        if row.get("item_id")
    }
    understanding_by_id = {
        row.get("item_id"): row
        for row in load_jsonl(Path(understanding_results_path))
        if row.get("item_id")
    }
    answer_overrides_by_id = load_answer_verification_overrides(answer_verification_overrides_path)
    display_answer_leak_quarantine = load_display_answer_leak_quarantine(display_answer_leak_quarantine_path)

    rows: list[dict[str, Any]] = []
    display_answer_leak_by_kg_node: Counter[str] = Counter()
    for item in items:
        item_id = item.get("item_id", "")
        visual = visual_by_id.get(item_id)
        needs_image = item_needs_image(item, visual)
        answer_override = answer_overrides_by_id.get(str(item_id))

        blockers: list[str] = []
        ans_status = answer_status(item, answer_override=answer_override)
        rub_status = rubric_status(item)
        if ans_status == "missing":
            blockers.append("answer_missing")
        elif ans_status == "suspect":
            blockers.append("answer_suspect")
        if rub_status == "missing":
            blockers.append("rubric_missing")

        if needs_image:
            understanding = understanding_by_id.get(item_id)
            if visual:
                visual_asset_status = visual.get("match_tier", "reject")
                blockers.extend(visual.get("blocker_reasons") or [])
                if display_has_answer_leak(visual, display_answer_leak_quarantine):
                    blockers.append("display_answer_leak")
                    kg_nodes = item.get("kg_nodes") or ["(missing_kg_node)"]
                    for kg_node in kg_nodes:
                        display_answer_leak_by_kg_node[str(kg_node)] += 1
                has_page_image = bool(visual.get("page_image_path") and visual.get("page_image_hash"))
                readability_status = "pass" if visual_asset_status == "strong" and has_page_image else "manual_review"
            else:
                visual_asset_status = "reject"
                readability_status = "reject"
                blockers.append("missing_visual_asset_manifest")

            llm_status = understanding_status(understanding, visual_asset_status)
        else:
            understanding = None
            visual_asset_status = "not_required"
            readability_status = "pass"
            llm_status = "not_required"

        blockers = sorted(set(blockers))
        base_verified = ans_status == "verified" and rub_status in {"complete", "partial"} and not blockers
        visual_display_ready = bool(
            needs_image
            and visual
            and visual_asset_status == "strong"
            and readability_status == "pass"
            and has_image_evidence(visual)
            and ans_status == "verified"
            and rub_status in {"complete", "partial"}
            and not blockers
        )
        strong_blockers = strong_blockers_for(
            needs_image=needs_image,
            visual=visual,
            understanding=understanding,
            visual_asset_status=visual_asset_status,
            readability_status=readability_status,
            llm_status=llm_status,
            ans_status=ans_status,
            rub_status=rub_status,
            blockers=blockers,
        )
        visual_ready_for_profile = bool(visual_display_ready and not strong_blockers)
        text_ready_for_profile = not needs_image and base_verified and rub_status == "complete"
        student_readable = bool(text_ready_for_profile or visual_display_ready)
        strong = bool((text_ready_for_profile and not strong_blockers) or visual_ready_for_profile)
        stage = visual_pipeline_stage(
            needs_image=needs_image,
            visual=visual,
            ans_status=ans_status,
            rub_status=rub_status,
            readability_status=readability_status,
            student_readable=student_readable,
            strong=strong,
            understanding=understanding,
        )
        review_queue = review_queue_for(
            needs_image=needs_image,
            student_readable=student_readable,
            strong=strong,
            blockers=blockers,
            llm_status=llm_status,
            ans_status=ans_status,
        )
        source_page = visual.get("declared_page") if visual else item.get("page")
        evidence_id = quality_evidence_id(item_id, visual, understanding, stage)

        row = {
            "item_id": item_id,
            "needs_image": needs_image,
            "student_readable": student_readable,
            "strong": strong,
            "visual_pipeline_stage": stage,
            "review_queue": review_queue,
            "quality_evidence_id": evidence_id,
            "visual_asset_status": visual_asset_status,
            "readability_status": readability_status,
            "llm_understanding_status": llm_status,
            "answer_status": ans_status,
            "rubric_status": rub_status,
            "usable_for_diagnosis": strong,
            "usable_for_practice": student_readable,
            "usable_for_teaching": student_readable,
            "usable_for_profile_evidence": strong,
            "source_file": visual.get("source_file", "") if visual else item.get("source", ""),
            "source_path": visual.get("source_path", "") if visual else "",
            "page": source_page,
            "page_image_path": visual.get("page_image_path", "") if visual else "",
            "page_image_hash": visual.get("page_image_hash", "") if visual else "",
            "crop_path": visual.get("crop_path") or "" if visual else "",
            "crop_hash": visual.get("crop_hash") or "" if visual else "",
            "crop_tier": visual.get("crop_tier", "") if visual else "",
            "display_image_path": display_image_path(visual),
            "display_image_hash": display_image_hash(visual),
            "display_answer_leak": "display_answer_leak" in blockers,
            "visual_evidence_mode": visual_evidence_mode(visual, understanding),
            "category": (visual.get("category") if visual else item.get("category")) or "",
            "vl_model": (understanding or {}).get("model", ""),
            "vl_visible_pass": bool((understanding or {}).get("visible_pass", False)),
            "vl_answer_match": bool((understanding or {}).get("answer_match", False)),
            "vl_error_types": list((understanding or {}).get("error_types") or []),
            "vl_raw_source": (understanding or {}).get("raw_source", ""),
            "answer_review_source": (answer_override or {}).get("review_source", ""),
            "answer_review_decision": (answer_override or {}).get("review_decision", ""),
            "blocker_reasons": blockers,
            "strong_blocker_reasons": strong_blockers,
            "reviewer": "script",
            "updated_at": "2026-07-01T00:00:00+08:00",
        }
        rows.append(row)

    blocker_reasons = Counter(reason for row in rows for reason in row["blocker_reasons"])
    strong_blocker_reasons = Counter(reason for row in rows for reason in row["strong_blocker_reasons"])
    summary = {
        "total_items": len(items),
        "needs_image": sum(1 for row in rows if row["needs_image"]),
        "visual_asset_status": dict(Counter(row["visual_asset_status"] for row in rows)),
        "readability_status": dict(Counter(row["readability_status"] for row in rows)),
        "llm_understanding_status": dict(Counter(row["llm_understanding_status"] for row in rows)),
        "answer_status": dict(Counter(row["answer_status"] for row in rows)),
        "rubric_status": dict(Counter(row["rubric_status"] for row in rows)),
        "student_readable": sum(1 for row in rows if row["student_readable"]),
        "strong": sum(1 for row in rows if row["strong"]),
        "visual_pipeline_stage": dict(Counter(row["visual_pipeline_stage"] for row in rows)),
        "review_queue": dict(Counter(row["review_queue"] for row in rows)),
        "usable_for_diagnosis": sum(1 for row in rows if row["usable_for_diagnosis"]),
        "usable_for_practice": sum(1 for row in rows if row["usable_for_practice"]),
        "usable_for_teaching": sum(1 for row in rows if row["usable_for_teaching"]),
        "usable_for_profile_evidence": sum(1 for row in rows if row["usable_for_profile_evidence"]),
        "blocked_items": sum(1 for row in rows if row["blocker_reasons"]),
        "display_answer_leak": blocker_reasons.get("display_answer_leak", 0),
        "display_answer_leak_by_kg_node": dict(display_answer_leak_by_kg_node),
        "blocker_reasons": dict(blocker_reasons),
        "strong_blocker_reasons": dict(strong_blocker_reasons),
        "answer_verification_overrides": len(answer_overrides_by_id),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build item quality manifest from item bank and visual manifest.")
    parser.add_argument("--item-bank", type=Path, default=DEFAULT_ITEM_BANK)
    parser.add_argument("--visual-manifest", type=Path, default=DEFAULT_VISUAL_MANIFEST)
    parser.add_argument("--understanding-results", type=Path, default=DEFAULT_UNDERSTANDING_RESULTS)
    parser.add_argument("--answer-verification-overrides", type=Path, default=DEFAULT_ANSWER_VERIFICATION_OVERRIDES)
    parser.add_argument("--display-answer-leak-quarantine", type=Path, default=DEFAULT_DISPLAY_ANSWER_LEAK_QUARANTINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_QUALITY_DIR / "item_quality_manifest.jsonl")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_QUALITY_DIR / "item_quality_summary.json")
    parser.add_argument("--write", action="store_true", help="Write output files. Default is dry-run.")
    args = parser.parse_args()

    rows, summary = build_quality_manifest(
        item_bank_path=args.item_bank,
        visual_manifest_path=args.visual_manifest,
        understanding_results_path=args.understanding_results,
        answer_verification_overrides_path=args.answer_verification_overrides,
        display_answer_leak_quarantine_path=args.display_answer_leak_quarantine,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.write:
        write_jsonl(args.out, rows)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {args.out}")
        print(f"WROTE {args.summary_out}")
    else:
        print("DRY RUN: pass --write to write data/quality outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
