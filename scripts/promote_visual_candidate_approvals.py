#!/usr/bin/env python3
"""Apply approved visual candidate promotions into explicit output manifests.

This script is intentionally narrow:
- only rows with status=approved are promoted;
- manual_review/reject rows are skipped;
- crop files are copied from temporary run directories into a stable project path;
- official manifests are never mutated unless the caller explicitly points the
  output paths at official files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

SKILL_DIR = Path(__file__).parent.parent
DEFAULT_OFFICIAL_VISUAL = SKILL_DIR / "data" / "quality" / "visual_asset_manifest.jsonl"
DEFAULT_OFFICIAL_UNDERSTANDING = SKILL_DIR / "data" / "evals" / "visual_understanding_results.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def by_item_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("item_id")): row for row in rows if row.get("item_id")}


def sha256_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def safe_filename(item_id: str, crop_hash: str, suffix: str) -> str:
    digest = crop_hash.replace("sha256:", "")[:16] or "unknownhash"
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{item_id}_{digest}{suffix}"


def validate_approved_row(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("status") != "approved":
        return False, "non_approved_status"
    if row.get("review_reasons") or row.get("reject_reasons"):
        return False, "approved_row_has_review_or_reject_reasons"
    if row.get("additional_image_paths"):
        return False, "approved_row_has_cross_page_evidence"
    if row.get("candidate_repair"):
        return False, "approved_row_has_candidate_repair"
    return True, ""


def is_structured_transcript_promotion(
    approval: dict[str, Any],
    candidate_understanding: dict[str, Any] | None,
) -> bool:
    if approval.get("promotion_scope") == "understanding_structured_transcript":
        return True
    if not candidate_understanding:
        return False
    return bool(
        candidate_understanding.get("transcript_supported_strong")
        or candidate_understanding.get("visual_evidence_mode") == "structured_transcript"
    )


def validate_candidate_evidence(
    item_id: str,
    approval: dict[str, Any],
    candidate_visual: dict[str, Any] | None,
    candidate_understanding: dict[str, Any] | None,
) -> str:
    if not candidate_visual:
        raise ValueError(f"{item_id}:missing_candidate_visual_row")
    if not candidate_understanding:
        raise ValueError(f"{item_id}:missing_candidate_understanding_row")

    if candidate_visual.get("additional_image_paths") or candidate_understanding.get("additional_image_paths"):
        raise ValueError(f"{item_id}:cross_page_evidence_not_allowed_for_approved_apply")
    if candidate_understanding.get("candidate_repair"):
        raise ValueError(f"{item_id}:candidate_repair_not_allowed_for_approved_apply")
    if candidate_understanding.get("error_types"):
        raise ValueError(f"{item_id}:candidate_understanding_has_error_types")
    if not (
        candidate_understanding.get("visible_pass")
        and candidate_understanding.get("answer_match")
        and candidate_understanding.get("understanding_pass")
    ):
        raise ValueError(f"{item_id}:candidate_understanding_not_strong")
    if is_structured_transcript_promotion(approval, candidate_understanding):
        if candidate_understanding.get("visual_evidence_mode") != "structured_transcript":
            raise ValueError(f"{item_id}:missing_structured_transcript_evidence_mode")
        if not candidate_understanding.get("transcript_supported_strong"):
            raise ValueError(f"{item_id}:missing_transcript_supported_strong_flag")
        if not (candidate_visual.get("page_image_path") and candidate_visual.get("page_image_hash")):
            raise ValueError(f"{item_id}:missing_page_image_evidence_for_transcript_promotion")
        return "structured_transcript"

    crop_path = Path(str(approval.get("candidate_crop_path") or candidate_visual.get("crop_path") or ""))
    crop_hash = str(approval.get("candidate_crop_hash") or candidate_visual.get("crop_hash") or "")
    if not crop_path.exists():
        raise ValueError(f"{item_id}:missing_candidate_crop_file:{crop_path}")
    if not crop_hash.startswith("sha256:"):
        raise ValueError(f"{item_id}:missing_candidate_crop_hash")
    actual_hash = sha256_digest(crop_path)
    if actual_hash != crop_hash:
        raise ValueError(f"{item_id}:candidate_crop_hash_mismatch:{actual_hash}!={crop_hash}")
    return "crop"


def copy_stable_crop(item_id: str, crop_path: Path, crop_hash: str, stable_crop_dir: Path) -> Path:
    stable_crop_dir.mkdir(parents=True, exist_ok=True)
    suffix = crop_path.suffix or ".png"
    destination = stable_crop_dir / safe_filename(item_id, crop_hash, suffix)
    if not destination.exists() or sha256_digest(destination) != crop_hash:
        shutil.copyfile(crop_path, destination)
    if sha256_digest(destination) != crop_hash:
        raise ValueError(f"{item_id}:stable_crop_hash_mismatch:{destination}")
    return destination


def apply_approved_promotions(
    approved_plan_path: Path,
    candidate_visual_path: Path,
    candidate_understanding_path: Path,
    official_visual_path: Path,
    official_understanding_path: Path,
    out_visual_path: Path,
    out_understanding_path: Path,
    stable_crop_dir: Path,
    summary_out: Path | None = None,
    patch_out: Path | None = None,
) -> dict[str, Any]:
    plan_rows = load_jsonl(Path(approved_plan_path))
    candidate_visual_by_id = by_item_id(load_jsonl(Path(candidate_visual_path)))
    candidate_understanding_by_id = by_item_id(load_jsonl(Path(candidate_understanding_path)))
    official_visual_rows = load_jsonl(Path(official_visual_path))
    official_understanding_rows = load_jsonl(Path(official_understanding_path))

    approved_rows: list[dict[str, Any]] = []
    skipped_non_approved = 0
    for row in plan_rows:
        is_valid, reason = validate_approved_row(row)
        if is_valid:
            approved_rows.append(row)
        elif row.get("status") == "approved":
            raise ValueError(f"{row.get('item_id')}:invalid_approved_row:{reason}")
        else:
            skipped_non_approved += 1

    approved_ids = {str(row.get("item_id")) for row in approved_rows}
    patch_rows: list[dict[str, Any]] = []
    stable_crop_by_id: dict[str, Path] = {}
    promotion_mode_by_id: dict[str, str] = {}

    for approval in approved_rows:
        item_id = str(approval.get("item_id"))
        candidate_visual = candidate_visual_by_id.get(item_id)
        candidate_understanding = candidate_understanding_by_id.get(item_id)
        mode = validate_candidate_evidence(item_id, approval, candidate_visual, candidate_understanding)
        promotion_mode_by_id[item_id] = mode
        if mode == "crop":
            crop_path = Path(str(approval.get("candidate_crop_path") or candidate_visual.get("crop_path")))
            crop_hash = str(approval.get("candidate_crop_hash") or candidate_visual.get("crop_hash"))
            stable_crop_path = copy_stable_crop(item_id, crop_path, crop_hash, stable_crop_dir)
            stable_crop_by_id[item_id] = stable_crop_path

    promoted_visual_rows: list[dict[str, Any]] = []
    for row in official_visual_rows:
        item_id = str(row.get("item_id"))
        if item_id not in approved_ids:
            promoted_visual_rows.append(row)
            continue
        if promotion_mode_by_id[item_id] == "structured_transcript":
            promoted_visual_rows.append(row)
            patch_rows.append(
                {
                    "item_id": item_id,
                    "visual_field_changes": {},
                    "understanding_field_changes": {
                        "visual_evidence_mode": {"to": "structured_transcript"},
                        "transcript_supported_strong": {"to": True},
                    },
                }
            )
            continue
        candidate_visual = candidate_visual_by_id[item_id]
        stable_crop_path = stable_crop_by_id[item_id]
        crop_hash = str(candidate_visual.get("crop_hash"))
        new_row = dict(row)
        before = {
            "crop_path": row.get("crop_path"),
            "crop_hash": row.get("crop_hash"),
            "crop_tier": row.get("crop_tier"),
        }
        new_row.update(
            {
                "crop_path": str(stable_crop_path),
                "crop_hash": crop_hash,
                "crop_tier": candidate_visual.get("crop_tier") or "item_crop_candidate",
                "candidate_crop_source": "approved_candidate_promotion",
                "promotion_status": "official_promoted_approved_candidate",
            }
        )
        promoted_visual_rows.append(new_row)
        patch_rows.append(
            {
                "item_id": item_id,
                "visual_field_changes": {
                    "crop_path": {"from": before["crop_path"], "to": new_row["crop_path"]},
                    "crop_hash": {"from": before["crop_hash"], "to": new_row["crop_hash"]},
                    "crop_tier": {"from": before["crop_tier"], "to": new_row["crop_tier"]},
                },
            }
        )

    official_visual_ids = {str(row.get("item_id")) for row in official_visual_rows if row.get("item_id")}
    missing_visual_ids = sorted(approved_ids - official_visual_ids)
    if missing_visual_ids:
        raise ValueError(f"approved_ids_missing_from_official_visual:{missing_visual_ids}")

    promoted_understanding_by_id = by_item_id(official_understanding_rows)
    for item_id in approved_ids:
        candidate_understanding = dict(candidate_understanding_by_id[item_id])
        if promotion_mode_by_id[item_id] == "structured_transcript":
            candidate_understanding["visual_evidence_mode"] = "structured_transcript"
            candidate_understanding["transcript_supported_strong"] = True
            candidate_understanding["promotion_status"] = "official_promoted_structured_transcript"
        else:
            candidate_understanding["input_image_path"] = str(stable_crop_by_id[item_id])
            candidate_understanding["promotion_status"] = "official_promoted_approved_candidate"
        promoted_understanding_by_id[item_id] = candidate_understanding

    original_understanding_order = [
        str(row.get("item_id")) for row in official_understanding_rows if row.get("item_id") in promoted_understanding_by_id
    ]
    appended_ids = sorted(approved_ids - set(original_understanding_order))
    final_understanding_rows = [promoted_understanding_by_id[item_id] for item_id in original_understanding_order]
    final_understanding_rows.extend(promoted_understanding_by_id[item_id] for item_id in appended_ids)

    write_jsonl(Path(out_visual_path), promoted_visual_rows)
    write_jsonl(Path(out_understanding_path), final_understanding_rows)

    summary = {
        "approved_count": len(approved_rows),
        "promoted_count": len(approved_rows),
        "skipped_non_approved_count": skipped_non_approved,
        "official_visual_rows": len(official_visual_rows),
        "official_understanding_before": len(official_understanding_rows),
        "official_understanding_after": len(final_understanding_rows),
        "stable_crop_dir": str(stable_crop_dir),
        "promoted_ids": sorted(approved_ids),
        "crop_promoted_count": sum(1 for mode in promotion_mode_by_id.values() if mode == "crop"),
        "transcript_promoted_count": sum(
            1 for mode in promotion_mode_by_id.values() if mode == "structured_transcript"
        ),
        "output_visual": str(out_visual_path),
        "output_understanding": str(out_understanding_path),
    }
    if summary_out:
        write_json(Path(summary_out), summary)
    if patch_out:
        write_json(Path(patch_out), {"scope": "approved_visual_candidate_promotion", "patch": patch_rows})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply approved visual candidate promotions to output manifests.")
    parser.add_argument("--approved-plan", type=Path, required=True)
    parser.add_argument("--candidate-visual", type=Path, required=True)
    parser.add_argument("--candidate-understanding", type=Path, required=True)
    parser.add_argument("--official-visual", type=Path, default=DEFAULT_OFFICIAL_VISUAL)
    parser.add_argument("--official-understanding", type=Path, default=DEFAULT_OFFICIAL_UNDERSTANDING)
    parser.add_argument("--out-visual", type=Path, required=True)
    parser.add_argument("--out-understanding", type=Path, required=True)
    parser.add_argument("--stable-crop-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--patch-out", type=Path)
    args = parser.parse_args()

    summary = apply_approved_promotions(
        approved_plan_path=args.approved_plan,
        candidate_visual_path=args.candidate_visual,
        candidate_understanding_path=args.candidate_understanding,
        official_visual_path=args.official_visual,
        official_understanding_path=args.official_understanding,
        out_visual_path=args.out_visual,
        out_understanding_path=args.out_understanding,
        stable_crop_dir=args.stable_crop_dir,
        summary_out=args.summary_out,
        patch_out=args.patch_out,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
