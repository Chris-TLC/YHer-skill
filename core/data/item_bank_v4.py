#!/usr/bin/env python3
"""The only legitimate read entry point for the schema-v4 item bank (WS3 apply, 2026-07-03).

The v4 manifest coexists with v3:
- The v3 service path (core/data/item_repository.py) globs data/item_bank/*.jsonl
  (non-recursive), so it can never see this directory — switching to v4 must be an
  explicit code change; there is no way for v4 to "quietly sneak in".
- This module is v4's rule-level line of defense. Any student-facing display,
  practice, diagnosis, or AI internalization seed may only obtain items through
  iter_service_items() / load_service_pool(), and must never bypass them by
  reading the jsonl directly.

Hard rules (WS3 apply clauses in the handoff doc + pre-apply audit conclusions of 2026-07-03):
  R1  Only items with pool == "main" and service_eligible == true may enter the service pool.
  R2  Items whose quality_flags contains "answer_type_mismatch" are treated as having no
      answer (not trusted even if present in the data) and are not serviceable — a rule-level
      backstop against residual suspected mismatches, taking priority over case-by-case triage.
  R3  Items with no answer anywhere in the chain are not serviceable: items whose
      answer_blocks_effective is empty and whose standard_solution also yields no answer are
      kept out of the pool (the pre-apply audit measured 138 such leaked-through items in the
      main pool).
  R4  The explicit exclusion list in service_exclusions.jsonl must be respected (7 mis-sliced
      items whose stems are actually solution text + 4 items wrongly extracted from answer
      fragments / answer-only documents; all verified one by one by hand, leak class).
      Per governance discipline, additions to this list may only be signed off by the user
      or Claude.
  R5  Usability whitelist gate (2026-07-06 R5 apply; user L1: "authorized, and R5 goes first...
      proceed directly with the authorized ingest"): only serve items with r5_serve == true in
      usability_r5_v1.jsonl. The count is read dynamically from the ledger; after the
      bad-item closeout of 2026-07-13 it stands at 1202. See
      PROJECT_HANDOFF/BATCH14_AUDIT_2026-07-06.md §5 for the exclusion criteria, including
      hollow structural formulas, verbatim source code, fragment literals, partial answers,
      and held variant libraries — the clean false-negative classes.
      Items absent from the ledger are never served (better to omit than to serve junk).
      Audits/previews/regressions that need the full pool must pass apply_r5=False explicitly —
      student-facing item-retrieval paths must not disable R5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

V4_BANK_DIR = Path(__file__).parent.parent.parent / "data" / "item_bank" / "v4"
# Batch 7 apply (2026-07-04, user authorization "authorize batch7"): switch to v4.1
# (98 rows handled, zero additions, zero deletions).
# Rollback = point back to chemistry_v4_3329.jsonl (see data/_backup_pre_v4_1_apply_20260704/ROLLBACK.md).
V4_MANIFEST = V4_BANK_DIR / "chemistry_v4_1_3329.jsonl"
V4_SERVICE_EXCLUSIONS = V4_BANK_DIR / "service_exclusions.jsonl"
# R5 apply (2026-07-06, user L1 "authorized, and R5 goes first... proceed directly with the
# authorized ingest"): the batch14 usability ledger.
# Rollback = remove this file's reference / restore the backup (see data/_backup_pre_r5_apply_20260706/ROLLBACK.md).
V4_USABILITY_R5 = V4_BANK_DIR / "usability_r5_v1.jsonl"

ANSWER_TYPE_MISMATCH = "answer_type_mismatch"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_service_exclusions(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    return {
        row["item_id"]: row
        for row in _iter_jsonl(path or V4_SERVICE_EXCLUSIONS)
        if row.get("item_id")
    }


def load_usability_r5(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """R5: the batch14 usability ledger (item_id → {usability_pool, r5_serve, r5_block_reason, ...})."""
    return {
        row["item_id"]: row
        for row in _iter_jsonl(path or V4_USABILITY_R5)
        if row.get("item_id")
    }


def effective_answer_blocks(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """R2: mismatched items are treated as having no answer — don't trust the data, re-judge at the rule level."""
    if ANSWER_TYPE_MISMATCH in (item.get("quality_flags") or []):
        return []
    return item.get("answer_blocks_effective") or []


def solution_answers(item: Dict[str, Any]) -> List[str]:
    if ANSWER_TYPE_MISMATCH in (item.get("quality_flags") or []):
        return []
    sol = item.get("standard_solution") or {}
    answers = [str(a).strip() for a in (sol.get("final_answers") or []) if str(a).strip()]
    standard = str(sol.get("standard_answer") or "").strip()
    if standard and standard not in answers:
        answers.append(standard)
    return answers


def has_effective_answer(item: Dict[str, Any]) -> bool:
    """R3: at least one of answer_blocks_effective or standard_solution must yield an answer."""
    return bool(effective_answer_blocks(item)) or bool(solution_answers(item))


def service_blockers(
    item: Dict[str, Any],
    exclusions: Optional[Dict[str, Dict[str, Any]]] = None,
    usability: Optional[Dict[str, Dict[str, Any]]] = None,
    apply_r5: bool = True,
) -> List[str]:
    """Return every reason this item cannot enter the service pool; an empty list = serviceable.

    apply_r5=False is only for audits/previews/regressions that need the full pool
    (R1-R4 scope); student-facing item retrieval must not disable it.
    """
    if exclusions is None:
        exclusions = load_service_exclusions()
    blockers: List[str] = []
    if item.get("pool") != "main":
        blockers.append(f"pool_not_main:{item.get('pool')}")
    if item.get("service_eligible") is not True:
        blockers.append("not_service_eligible")
    if ANSWER_TYPE_MISMATCH in (item.get("quality_flags") or []):
        blockers.append("answer_type_mismatch")
    if not has_effective_answer(item):
        blockers.append("no_effective_answer")
    excl = exclusions.get(item.get("item_id", ""))
    if excl:
        blockers.append(f"service_exclusion:{excl.get('reason', 'listed')}")
    if apply_r5:
        if usability is None:
            usability = load_usability_r5()
        urow = usability.get(item.get("item_id", ""))
        if urow is None:
            # No ledger entry = never passed the usability audit; better to omit than to serve junk.
            blockers.append("usability_missing")
        elif urow.get("r5_serve") is not True:
            blockers.append(
                f"usability_not_serveable:{urow.get('r5_block_reason') or urow.get('usability_pool')}"
            )
    return blockers


def iter_items(manifest_path: Optional[Path] = None) -> Iterator[Dict[str, Any]]:
    """Full traversal (including the legacy/excluded pools). For audits, regression, and data engineering only — never for serving."""
    yield from _iter_jsonl(manifest_path or V4_MANIFEST)


def iter_service_items(
    manifest_path: Optional[Path] = None,
    exclusions_path: Optional[Path] = None,
    apply_r5: bool = True,
    usability_path: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Service-pool traversal: the only item-retrieval entry for student-facing features / practice / diagnosis / AI internalization seeds (defaults to the R5 whitelist scope).

    apply_r5=False is only for audits/previews/regression (R1-R4 scope, 2526);
    student-facing paths must not use it.
    """
    exclusions = load_service_exclusions(exclusions_path)
    usability = load_usability_r5(usability_path) if apply_r5 else None
    for item in iter_items(manifest_path):
        if not service_blockers(item, exclusions, usability=usability, apply_r5=apply_r5):
            yield item


def load_service_pool(
    manifest_path: Optional[Path] = None,
    exclusions_path: Optional[Path] = None,
    apply_r5: bool = True,
) -> Dict[str, Dict[str, Any]]:
    return {
        it["item_id"]: it
        for it in iter_service_items(manifest_path, exclusions_path, apply_r5=apply_r5)
    }


def service_pool_stats(
    manifest_path: Optional[Path] = None,
    exclusions_path: Optional[Path] = None,
    apply_r5: bool = True,
) -> Dict[str, Any]:
    """For audits: per-pool split of the whole bank + service-pool size + per-blocker counts."""
    exclusions = load_service_exclusions(exclusions_path)
    usability = load_usability_r5() if apply_r5 else None
    total = 0
    pools: Dict[str, int] = {}
    blocker_counts: Dict[str, int] = {}
    servable = 0
    for item in iter_items(manifest_path):
        total += 1
        pools[item.get("pool", "?")] = pools.get(item.get("pool", "?"), 0) + 1
        blockers = service_blockers(item, exclusions, usability=usability, apply_r5=apply_r5)
        if blockers:
            for b in blockers:
                key = b.split(":", 1)[0]
                blocker_counts[key] = blocker_counts.get(key, 0) + 1
        else:
            servable += 1
    return {
        "total": total,
        "pools": pools,
        "servable": servable,
        "blocker_counts": blocker_counts,
        "exclusion_rows": len(exclusions),
        "apply_r5": apply_r5,
        "usability_rows": len(usability) if usability is not None else None,
    }
