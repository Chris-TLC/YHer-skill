#!/usr/bin/env python3
"""QA-3 cross-node broken ion merge candidates.

L0 only: this script reads the v4 service pool and writes candidate/manual review
ledgers to /tmp. It never rewrites official item-bank data; --apply is a hard
refusal until a separate L1 authorization exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.apply_10e_unicode_ions import SUB, SUP, WHITELIST

ITEM_BANK = REPO / "data" / "item_bank" / "v4" / "chemistry_v4_1_3329.jsonl"
MEDIA_REF = REPO / "data" / "item_bank" / "v4" / "ws2_media_ref_map_v1.jsonl"
PNG_DIR = REPO / "data" / "ws2_assets_v1_candidate_20260703" / "converted_raw_png"

CANDIDATES_OUT = Path("/tmp/yher_qa3_crossnode_candidates.jsonl")
MANUAL_OUT = Path("/tmp/yher_qa3_manual_tail.jsonl")
REPORT_OUT = Path("/tmp/yher_qa3_report.md")

FIELDS = ("stem_blocks", "answer_blocks_effective", "analysis_blocks")
HEADS = [
    "HCO",
    "HSO",
    "HPO",
    "ClO",
    "AlO",
    "MnO",
    "CrO",
    "SiO",
    "BrO",
    "NH",
    "SO",
    "NO",
    "CO",
    "PO",
    "IO",
    "Cl",
    "S",
    "O",
]
HEADS_SORTED = sorted(HEADS, key=len, reverse=True)
BOUNDARY_CHARS = {" ", "+", "（", "(", "＝", "=", "→", "\t"}
IDEMPOTENT_TAIL_CHARS = set("₀₁₂₃₄₅₆₇₈₉0123456789²³⁺⁻")

# Frozen from PROJECT_HANDOFF/codex_briefs/2026-07-05_qa3_crossnode_ion_merge.md.
# Keys are hash prefixes and are intentionally matched with startswith().
VERIFIED_CLUSTERS: Dict[str, Optional[Tuple[str, str]]] = {
    "e6b92e82": ("4", "+"),
    "66bbc00e": ("4", "+"),
    "03550e07": ("4", "+"),
    "e2c78343": ("3", "-"),
    "e8654e31": ("3", "-"),
    "f380f2be": ("3", "-"),
    "320d611391": ("3", "-"),
    "d94b9136": ("3", "-"),
    "577c78a5": ("3", "-"),
    "e4a6ae99": ("3", "2-"),
    "5f75c6ff": ("3", "2-"),
    "322dcdac": ("3", "2-"),
    "5e630523": ("3", "2-"),
    "98972fba": ("3", "2-"),
    "5db4bca1": ("3", "2-"),
    "36f53150": ("3", "2-"),
    "965f5aab": ("4", "2-"),
    "46d9c9b1": ("4", "2-"),
    "d71c5f52": ("4", "2-"),
    "60de8b59": ("4", "2-"),
    "86f40293": ("4", "2-"),
    "39f20a05": ("4", "2-"),
    "0735755f": ("2", "-"),
    "7a5dfabc": ("2", "-"),
    "6d201cd0": ("4", "-"),
    "ea8608f2": ("4", "3-"),
    "fa32cc3f": ("2", "2-"),
    "d627e6db": None,
}

EXTENSION = {
    "IO3-": "IO" + SUB["3"] + SUP["-"],
    "BrO3-": "BrO" + SUB["3"] + SUP["-"],
    "ClO3-": "ClO" + SUB["3"] + SUP["-"],
    "ClO4-": "ClO" + SUB["4"] + SUP["-"],
}

REV_SUP = {v: k for k, v in SUP.items()}
REV_SUB = {v: k for k, v in SUB.items()}
REV_UNICODE = {**REV_SUP, **REV_SUB}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_media_lookup(path: Path = MEDIA_REF) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    for row in read_jsonl(path):
        group_key = row.get("group_key")
        media = row.get("media")
        asset_hash = row.get("asset_hash")
        if group_key and media and asset_hash:
            lookup[(str(group_key), str(media))] = str(asset_hash)
    return lookup


def make_target(head: str, digit: str, charge: str) -> str:
    return head + SUB[digit] + "".join(SUP[c] for c in charge)


def cluster_text(digit: str, charge: str) -> str:
    return SUB[digit] + "".join(SUP[c] for c in charge)


def unwind(text: str) -> str:
    return "".join(REV_UNICODE.get(ch, ch) for ch in text)


def head_at_end(text: str) -> Optional[Tuple[str, int, int]]:
    trimmed = (text or "").rstrip()
    if not trimmed or trimmed[-1] in IDEMPOTENT_TAIL_CHARS:
        return None
    for head in HEADS_SORTED:
        if not trimmed.endswith(head):
            continue
        start = len(trimmed) - len(head)
        if start == 0 or trimmed[start - 1] in BOUNDARY_CHARS:
            return head, start, len(trimmed)
    return None


def verified_cluster(asset_hash: Optional[str]) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    if not asset_hash:
        return None, None
    for prefix, value in VERIFIED_CLUSTERS.items():
        if asset_hash.startswith(prefix):
            return prefix, value
    return None, None


def png_path_for(asset_hash: str) -> Path:
    return PNG_DIR / f"{asset_hash}.png"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _site_base(
    item: Mapping[str, Any],
    field: str,
    block_idx: int,
    left_para_idx: int,
    formula_para_idx: int,
    head: str,
    left: Mapping[str, Any],
    formula: Mapping[str, Any],
    right: Mapping[str, Any],
    asset_hash: Optional[str],
    png_path: Optional[Path],
) -> Dict[str, Any]:
    return {
        "item_id": item.get("item_id", ""),
        "group_key": item.get("group_key", ""),
        "section_num": item.get("section_num"),
        "q_num": item.get("q_num"),
        "field": field,
        "left_block_idx": block_idx,
        "left_para_idx": left_para_idx,
        "formula_para_idx": formula_para_idx,
        "right_para_idx": formula_para_idx + 1,
        "asset_hash": asset_hash or "",
        "media": formula.get("media", ""),
        "head": head,
        "original_left": left.get("text", ""),
        "original_right": right.get("text", ""),
        "png_path": rel(png_path) if png_path else "",
        "review_status": "pending_user_or_claude",
        "reviewer": "",
        "schema_version": "qa3_candidate_v1",
        "candidate_kind": "crossnode_ion_merge",
    }


def _manual_row(
    reason: str,
    item: Mapping[str, Any],
    field: str,
    block_idx: int,
    left_para_idx: int,
    formula_para_idx: int,
    head: str,
    left: Mapping[str, Any],
    formula: Mapping[str, Any],
    right: Mapping[str, Any],
    asset_hash: Optional[str],
    png_path: Optional[Path],
    cluster: str = "",
    ion: str = "",
) -> Dict[str, Any]:
    row = _site_base(
        item,
        field,
        block_idx,
        left_para_idx,
        formula_para_idx,
        head,
        left,
        formula,
        right,
        asset_hash,
        png_path,
    )
    row.update(
        {
            "reason": reason,
            "cluster": cluster,
            "ion": ion,
            "review_note": "manual_tail_for_user_or_claude",
        }
    )
    return row


def _candidate_row(
    item: Mapping[str, Any],
    field: str,
    block_idx: int,
    left_para_idx: int,
    formula_para_idx: int,
    head: str,
    left: Mapping[str, Any],
    formula: Mapping[str, Any],
    right: Mapping[str, Any],
    asset_hash: str,
    digit: str,
    charge: str,
    ion: str,
    ion_group: str,
    target: str,
    replaced_left: str,
) -> Dict[str, Any]:
    png_path = png_path_for(asset_hash)
    row = _site_base(
        item,
        field,
        block_idx,
        left_para_idx,
        formula_para_idx,
        head,
        left,
        formula,
        right,
        asset_hash,
        png_path,
    )
    row.update(
        {
            "cluster": cluster_text(digit, charge),
            "ion": ion,
            "ion_group": ion_group,
            "replaced_left": replaced_left,
            "delete_formula_node": True,
            "right_node_unchanged": True,
        }
    )
    return row


def iter_formula_sandwiches(
    item: Mapping[str, Any],
) -> Iterator[Tuple[str, int, int, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    for field in FIELDS:
        blocks = item.get(field) or []
        for block_idx, block in enumerate(blocks):
            para = block.get("para") if isinstance(block, dict) else None
            if not para:
                continue
            for formula_idx in range(1, len(para) - 1):
                left = para[formula_idx - 1]
                formula = para[formula_idx]
                right = para[formula_idx + 1]
                if not (
                    isinstance(left, dict)
                    and isinstance(formula, dict)
                    and isinstance(right, dict)
                    and left.get("type") == "text"
                    and formula.get("type") == "formula"
                    and right.get("type") == "text"
                ):
                    continue
                yield field, block_idx, formula_idx - 1, para, left, formula, right


def build_crossnode_rows(
    items: Iterable[Mapping[str, Any]],
    media_lookup: Mapping[Tuple[str, str], str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    candidates: List[Dict[str, Any]] = []
    manual: List[Dict[str, Any]] = []
    stats: Counter = Counter()

    for item in items:
        stats["service_items_seen"] += 1
        for field, block_idx, left_idx, _para, left, formula, right in iter_formula_sandwiches(item):
            text = str(left.get("text") or "")
            head_info = head_at_end(text)
            if not head_info:
                continue
            head, head_start, head_end = head_info
            formula_idx = left_idx + 1
            stats["head_sites"] += 1

            media = str(formula.get("media") or "")
            asset_hash = media_lookup.get((str(item.get("group_key") or ""), media))
            if not asset_hash:
                manual.append(
                    _manual_row(
                        "unresolved_media",
                        item,
                        field,
                        block_idx,
                        left_idx,
                        formula_idx,
                        head,
                        left,
                        formula,
                        right,
                        None,
                        None,
                    )
                )
                stats["manual_unresolved_media"] += 1
                continue

            png_path = png_path_for(asset_hash)
            prefix, cluster = verified_cluster(asset_hash)
            if prefix is None:
                manual.append(
                    _manual_row(
                        "no_png",
                        item,
                        field,
                        block_idx,
                        left_idx,
                        formula_idx,
                        head,
                        left,
                        formula,
                        right,
                        asset_hash,
                        png_path if png_path.exists() else None,
                    )
                )
                stats["manual_no_png"] += 1
                continue
            if cluster is None:
                manual.append(
                    _manual_row(
                        "arrow_excluded",
                        item,
                        field,
                        block_idx,
                        left_idx,
                        formula_idx,
                        head,
                        left,
                        formula,
                        right,
                        asset_hash,
                        png_path if png_path.exists() else None,
                    )
                )
                stats["manual_arrow_excluded"] += 1
                continue
            if not png_path.exists():
                manual.append(
                    _manual_row(
                        "no_png",
                        item,
                        field,
                        block_idx,
                        left_idx,
                        formula_idx,
                        head,
                        left,
                        formula,
                        right,
                        asset_hash,
                        None,
                    )
                )
                stats["manual_no_png"] += 1
                continue

            digit, charge = cluster
            ion = f"{head}{digit}{charge}"
            if ion in WHITELIST:
                target = WHITELIST[ion]
                ion_group = "whitelist"
            elif ion in EXTENSION:
                target = EXTENSION[ion]
                ion_group = "extension"
            else:
                manual.append(
                    _manual_row(
                        "not_whitelist",
                        item,
                        field,
                        block_idx,
                        left_idx,
                        formula_idx,
                        head,
                        left,
                        formula,
                        right,
                        asset_hash,
                        png_path,
                        cluster_text(digit, charge),
                        ion,
                    )
                )
                stats["manual_not_whitelist"] += 1
                continue

            replaced_left = text[:head_start] + target + text[head_end:]
            ascii_cluster = digit + charge
            if unwind(replaced_left) != unwind(text) + ascii_cluster or unwind(target) != ion:
                stats["rev_fail"] += 1
                continue

            candidates.append(
                _candidate_row(
                    item,
                    field,
                    block_idx,
                    left_idx,
                    formula_idx,
                    head,
                    left,
                    formula,
                    right,
                    asset_hash,
                    digit,
                    charge,
                    ion,
                    ion_group,
                    target,
                    replaced_left,
                )
            )
            stats["candidates"] += 1
            stats[f"candidate_{ion_group}"] += 1

    stats["manual"] = len(manual)
    stats["candidate_items"] = len({row["item_id"] for row in candidates})
    stats["manual_items"] = len({row["item_id"] for row in manual})
    return candidates, manual, stats


def validate_rows(candidates: List[Mapping[str, Any]], manual: List[Mapping[str, Any]]) -> Dict[str, int]:
    allowed_manual = {"no_png", "unresolved_media", "arrow_excluded", "not_whitelist"}
    return {
        "candidate_reviewer_bad": sum(
            1
            for row in candidates
            if row.get("review_status") != "pending_user_or_claude" or row.get("reviewer") != ""
        ),
        "manual_reviewer_bad": sum(
            1
            for row in manual
            if row.get("review_status") != "pending_user_or_claude" or row.get("reviewer") != ""
        ),
        "candidate_codex_mentions": sum(
            1 for row in candidates if "codex_" in json.dumps(row, ensure_ascii=False).lower()
        ),
        "manual_codex_mentions": sum(
            1 for row in manual if "codex_" in json.dumps(row, ensure_ascii=False).lower()
        ),
        "candidate_asset_not_verified": sum(
            1
            for row in candidates
            if verified_cluster(str(row.get("asset_hash") or ""))[1] is None
        ),
        "manual_reason_bad": sum(1 for row in manual if row.get("reason") not in allowed_manual),
    }


def selfcheck(candidates: List[Mapping[str, Any]], manual: List[Mapping[str, Any]], stats: Counter) -> None:
    prefixes = list(VERIFIED_CLUSTERS)
    assert len(prefixes) == 28
    for i, a in enumerate(prefixes):
        for b in prefixes[i + 1 :]:
            assert not a.startswith(b) and not b.startswith(a), (a, b)
    for head in HEADS:
        for cluster in VERIFIED_CLUSTERS.values():
            if cluster is None:
                continue
            digit, charge = cluster
            ion = f"{head}{digit}{charge}"
            if ion in WHITELIST:
                assert unwind(WHITELIST[ion]) == ion, ion
            if ion in EXTENSION:
                assert unwind(EXTENSION[ion]) == ion, ion
                assert unwind(make_target(head, digit, charge)) == ion, ion
    assert 275 <= len(candidates) <= 295, len(candidates)
    assert len(manual) == 16, len(manual)
    assert stats["rev_fail"] == 0, stats["rev_fail"]
    validation = validate_rows(candidates, manual)
    assert all(v == 0 for v in validation.values()), validation


def sample_rows(candidates: List[Mapping[str, Any]], limit: int = 15) -> List[Mapping[str, Any]]:
    if len(candidates) <= limit:
        return candidates
    step = max(1, len(candidates) // limit)
    rows = candidates[::step][:limit]
    if len(rows) < limit:
        rows.extend(candidates[len(rows) : limit])
    return rows[:limit]


def render_report(
    path: Path,
    candidates: List[Mapping[str, Any]],
    manual: List[Mapping[str, Any]],
    stats: Counter,
    validation: Mapping[str, int],
    md5_before: str,
    md5_after: str,
    service_count: int,
) -> None:
    fields = Counter(row["field"] for row in candidates)
    ions = Counter(row["ion"] for row in candidates)
    manual_reasons = Counter(row["reason"] for row in manual)
    lines: List[str] = []
    lines.append("# QA-3 Cross-node Ion Merge L0 Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- service_pool_count: {service_count}")
    lines.append(f"- candidate_rows: {len(candidates)}")
    lines.append(f"- candidate_items: {stats['candidate_items']}")
    lines.append(f"- manual_tail_rows: {len(manual)}")
    lines.append(f"- manual_tail_items: {stats['manual_items']}")
    lines.append(f"- head_sites: {stats['head_sites']}")
    lines.append(f"- rev_fail: {stats['rev_fail']}")
    lines.append(f"- official_md5_before: `{md5_before}`")
    lines.append(f"- official_md5_after: `{md5_after}`")
    lines.append("")
    lines.append("## Field Counts")
    for key, val in sorted(fields.items()):
        lines.append(f"- {key}: {val}")
    lines.append("")
    lines.append("## Ion Counts")
    for key, val in ions.most_common():
        lines.append(f"- {key}: {val}")
    lines.append("")
    lines.append("## Manual Tail Reasons")
    for key, val in sorted(manual_reasons.items()):
        lines.append(f"- {key}: {val}")
    lines.append("")
    lines.append("## Section 4 Gate Reconciliation")
    lines.append(f"- zero_official_write_md5_match: {md5_before == md5_after}")
    lines.append(f"- service_pool_2526: {service_count == 2526} ({service_count})")
    lines.append(f"- reversible_failures_zero: {stats['rev_fail'] == 0} ({stats['rev_fail']})")
    lines.append(
        f"- candidate_assets_all_verified_non_arrow: {validation['candidate_asset_not_verified'] == 0} "
        f"({validation['candidate_asset_not_verified']} bad)"
    )
    lines.append(
        "- reviewer_empty_pending: "
        f"{validation['candidate_reviewer_bad'] == 0 and validation['manual_reviewer_bad'] == 0} "
        f"(candidate_bad={validation['candidate_reviewer_bad']}, manual_bad={validation['manual_reviewer_bad']})"
    )
    lines.append(
        "- codex_reviewer_mentions_zero: "
        f"{validation['candidate_codex_mentions'] == 0 and validation['manual_codex_mentions'] == 0} "
        f"(candidate={validation['candidate_codex_mentions']}, manual={validation['manual_codex_mentions']})"
    )
    lines.append(
        f"- expected_scale: candidates={len(candidates)} (~285), items={stats['candidate_items']} (~82), "
        f"manual={len(manual)} (16), total_sites={len(candidates) + len(manual)} (301)"
    )
    lines.append(
        "- scale_breakdown: "
        f"whitelist={stats['candidate_whitelist']}, extension={stats['candidate_extension']}, "
        f"manual_no_png={stats['manual_no_png']}, "
        f"manual_unresolved_media={stats['manual_unresolved_media']}, "
        f"manual_not_whitelist={stats['manual_not_whitelist']}, "
        f"manual_arrow_excluded={stats['manual_arrow_excluded']}"
    )
    lines.append(f"- manual_reason_schema_bad: {validation['manual_reason_bad']}")
    lines.append("")
    lines.append("## 15 Candidate Samples")
    for idx, row in enumerate(sample_rows(candidates), start=1):
        before = f"{row['original_left']} [formula:{row['media']}] {row['original_right']}"
        after = f"{row['replaced_left']} {row['original_right']}"
        lines.append(f"{idx}. `{row['item_id']}` {row['field']}[{row['left_block_idx']}].para[{row['left_para_idx']}]")
        lines.append(f"   - before: {before}")
        lines.append(f"   - after: {after}")
        lines.append(f"   - ion: {row['ion']} ({row['ion_group']}), asset: `{row['asset_hash']}`")
    lines.append("")
    lines.append("## Artifact Paths")
    lines.append(f"- candidates: `{CANDIDATES_OUT}`")
    lines.append(f"- manual_tail: `{MANUAL_OUT}`")
    lines.append(f"- report: `{REPORT_OUT}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(do_selfcheck: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    from core.data.item_bank_v4 import iter_service_items

    md5_before = md5_file(ITEM_BANK)
    media_lookup = load_media_lookup()
    service_items = list(iter_service_items())
    candidates, manual, stats = build_crossnode_rows(service_items, media_lookup)
    validation = validate_rows(candidates, manual)
    if do_selfcheck:
        selfcheck(candidates, manual, stats)
    write_jsonl(CANDIDATES_OUT, candidates)
    write_jsonl(MANUAL_OUT, manual)
    md5_after = md5_file(ITEM_BANK)
    render_report(REPORT_OUT, candidates, manual, stats, validation, md5_before, md5_after, len(service_items))
    return candidates, manual, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Refused for L0; official writes require future L1.")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.apply:
        print("L1 未授权，拒绝写 official")
        return 2

    candidates, manual, stats = run(do_selfcheck=args.selfcheck)
    print("=== QA-3 cross-node ion merge L0 candidates ===")
    print(f"candidate_rows: {len(candidates)}")
    print(f"candidate_items: {stats['candidate_items']}")
    print(f"manual_tail_rows: {len(manual)}")
    print(f"manual_tail_items: {stats['manual_items']}")
    print(f"head_sites: {stats['head_sites']}")
    print(f"rev_fail: {stats['rev_fail']}")
    print(f"candidates_out: {CANDIDATES_OUT}")
    print(f"manual_out: {MANUAL_OUT}")
    print(f"report_out: {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
