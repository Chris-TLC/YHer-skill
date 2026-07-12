#!/usr/bin/env python3
"""Build an evidence-backed signed curriculum map from the read-only draft."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


AUTHORIZED_REVIEWER = "codex_sol_20260713"


class SigningError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_index(rows: Iterable[dict[str, Any]]) -> dict[str, tuple[int, dict[str, Any]]]:
    indexed: dict[str, tuple[int, dict[str, Any]]] = {}
    for line_number, row in enumerate(rows, start=1):
        entity_id = str(row.get("entity_id") or "").strip()
        if not entity_id:
            raise SigningError(f"catalog line {line_number} missing entity_id")
        if entity_id in indexed:
            raise SigningError(f"duplicate catalog entity: {entity_id}")
        indexed[entity_id] = (line_number, row)
    return indexed


def _evidence(
    *,
    catalog_source: str,
    line_number: int,
    entity_id: str,
    catalog_name: str,
    match: str,
    reviewed_at: str,
) -> dict[str, Any]:
    return {
        "source_file": catalog_source,
        "source_line": line_number,
        "catalog_entity_id": entity_id,
        "catalog_name": catalog_name,
        "match": match,
        "reviewed_at": reviewed_at,
    }


def build_signed_map(
    draft: dict[str, Any],
    catalog_rows: Iterable[dict[str, Any]],
    *,
    reviewer: str,
    reviewed_at: str,
    catalog_source: str,
    catalog_sha256: str,
    draft_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if reviewer != AUTHORIZED_REVIEWER:
        raise SigningError(f"unauthorized reviewer: {reviewer}")
    entities = list(draft.get("entities") or [])
    if not entities:
        raise SigningError("draft has no entities")
    catalog = _catalog_index(catalog_rows)
    seen: set[str] = set()
    signed_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for sequence, source_row in enumerate(entities, start=1):
        row = copy.deepcopy(source_row)
        entity_id = str(row.get("entity") or "").strip()
        if not entity_id:
            raise SigningError(f"draft entity {sequence} missing id")
        if entity_id in seen:
            raise SigningError(f"duplicate draft entity: {entity_id}")
        seen.add(entity_id)
        confidence = str(row.get("confidence") or "").strip().lower()
        if confidence not in {"high", "med", "low"}:
            raise SigningError(f"invalid confidence for {entity_id}")
        name_hint = str(row.get("name_hint") or "")
        catalog_entry = catalog.get(entity_id)
        catalog_line = catalog_entry[0] if catalog_entry else None
        catalog_row = catalog_entry[1] if catalog_entry else {}
        catalog_name = str(catalog_row.get("name") or "")
        subject_ok = str(catalog_row.get("subject") or "") == str(draft.get("subject") or "")
        title_ok = bool(catalog_entry) and catalog_name == name_hint

        if confidence != "high":
            decision, reason = "neutral", "confidence_not_high"
        elif catalog_entry is None:
            decision, reason = "neutral", "catalog_entity_missing"
        elif not subject_ok:
            decision, reason = "neutral", "catalog_subject_mismatch"
        elif not title_ok:
            decision, reason = "neutral", "catalog_title_mismatch"
        else:
            decision, reason = "signed", "exact_high_confidence_evidence"

        match = "exact_entity_id_and_title" if title_ok else (
            "entity_id_only_title_mismatch" if catalog_entry else "catalog_entity_missing"
        )
        evidence = (
            _evidence(
                catalog_source=catalog_source,
                line_number=int(catalog_line),
                entity_id=entity_id,
                catalog_name=catalog_name,
                match=match,
                reviewed_at=reviewed_at,
            )
            if catalog_entry
            else None
        )
        if decision == "signed":
            row["needs_human"] = False
            row["reviewer"] = reviewer
            row["evidence"] = evidence
            row.pop("neutral_reason", None)
        else:
            row["needs_human"] = True
            row["reviewer"] = ""
            row["neutral_reason"] = reason
            if evidence is not None:
                row["evidence"] = evidence
        signed_rows.append(row)
        manifest.append(
            {
                "sequence": sequence,
                "entity": entity_id,
                "name_hint": name_hint,
                "proposed_track": str(row.get("track") or ""),
                "confidence": confidence,
                "decision": decision,
                "reason": reason,
                "reviewer": reviewer if decision == "signed" else "",
                "catalog_source_line": catalog_line,
                "catalog_entity_id": str(catalog_row.get("entity_id") or "") or None,
                "catalog_name": catalog_name or None,
                "exact_entity_match": catalog_entry is not None,
                "exact_title_match": title_ok,
            }
        )

    date_token = reviewed_at.replace("-", "")
    output = copy.deepcopy(draft)
    output["version"] = f"curriculum_v1_signed_{date_token}"
    output["reviewer"] = reviewer
    output["reviewed_at"] = reviewed_at
    output["provenance"] = {
        "draft_sha256": draft_sha256,
        "catalog_source": catalog_source,
        "catalog_sha256": catalog_sha256,
        "signing_rule": "high confidence plus exact entity_id and exact catalog title",
    }
    output["entities"] = signed_rows
    return output, manifest


def write_artifacts(
    *,
    signed_map: dict[str, Any],
    evidence_manifest: list[dict[str, Any]],
    config_path: Path,
    manifest_path: Path,
    rollback_path: Path,
) -> None:
    config_path, manifest_path, rollback_path = map(
        Path, (config_path, manifest_path, rollback_path)
    )
    for path in (config_path, manifest_path, rollback_path):
        if path.exists():
            raise SigningError(f"refusing to overwrite artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(signed_map, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    manifest_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in evidence_manifest
        ),
        encoding="utf-8",
    )
    rollback_path.write_text(
        "# Curriculum Track Signing Rollback\n\n"
        "The source draft and catalog are read-only and were not modified. To disable the new signed config:\n\n"
        "```bash\n"
        "rm config/curriculum/track_map_v1.yaml\n"
        "```\n\n"
        "Retain this evidence manifest after rollback.\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise SigningError(f"{path}:{line_number} must be an object")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollback", type=Path, required=True)
    parser.add_argument("--reviewer", default=AUTHORIZED_REVIEWER)
    parser.add_argument("--reviewed-at", default="2026-07-13")
    args = parser.parse_args()
    draft_bytes = args.draft.read_bytes()
    draft = yaml.safe_load(draft_bytes)
    if not isinstance(draft, dict):
        raise SigningError("draft must be a mapping")
    signed_map, manifest = build_signed_map(
        draft,
        _read_jsonl(args.catalog),
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
        catalog_source=str(args.catalog),
        catalog_sha256=_sha256(args.catalog),
        draft_sha256=hashlib.sha256(draft_bytes).hexdigest(),
    )
    write_artifacts(
        signed_map=signed_map,
        evidence_manifest=manifest,
        config_path=args.config,
        manifest_path=args.manifest,
        rollback_path=args.rollback,
    )
    signed_count = sum(row["decision"] == "signed" for row in manifest)
    print(json.dumps({"entities": len(manifest), "signed": signed_count, "neutral": len(manifest) - signed_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
