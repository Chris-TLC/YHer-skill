"""Frozen code provenance for the S2 simulated-persona experiment."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_PATTERNS = (
    "experiments/llm_sim/*.py",
    "experiments/s0_census.py",
    "engine/mastery.py",
    "engine/selector.py",
    "core/learning/scoring.py",
    "core/learning/item_catalog.py",
)


def collect_code_provenance(repo_root: str | Path) -> dict[str, Any]:
    """Hash every executable contract used by S2 and bind it to git HEAD."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    files: set[Path] = set()
    for pattern in CODE_PATTERNS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise RuntimeError("S2 provenance code set is empty")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(files)
    ]
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve git HEAD for S2 provenance") from exc
    git_head = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", git_head):
        raise RuntimeError("S2 git HEAD is not a full lowercase commit SHA")
    return {
        "git_head": git_head,
        "code_sha256": hashlib.sha256(payload).hexdigest(),
        "code_files": rows,
    }


def collect_official_input_provenance(
    *,
    catalog: Any,
    kg: Any,
    personas: Sequence[Any],
) -> dict[str, Any]:
    """Bind loaded questions, scoring contracts, KG personas, and source files.

    Private answer values participate in a digest but are never returned.  The
    manifest therefore proves which scoring contract was used without exposing
    an answer key.
    """

    raw_items = getattr(catalog, "items", {}) or {}
    items = list(raw_items.values() if isinstance(raw_items, Mapping) else raw_items)
    items.sort(key=lambda item: str(_value(item, "item_id", "")))
    public_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    for item in items:
        public_method = getattr(item, "public_question", None)
        if callable(public_method):
            public = dict(public_method())
        else:
            public = {
                "kind": str(_value(item, "item_type", "mcq")),
                "stem_blocks": list(_value(item, "stem_blocks", ()) or ()),
                "stem_text": str(_value(item, "stem_text", "")),
                "options": dict(_value(item, "options", {}) or {}),
                "difficulty": float(_value(item, "difficulty", 0.5)),
                "nodes": list(_value(item, "node_ids", ()) or ()),
                "source_label": str(_value(item, "source_label", "")),
            }
        item_id = str(_value(item, "item_id", ""))
        public_rows.append(
            {
                "item_id": item_id,
                "family_id": str(_value(item, "family_id", "")),
                "public_question": public,
            }
        )
        scoring_rows.append(
            {
                "item_id": item_id,
                "scoring_mode": str(_value(item, "scoring_mode", "")),
                "answer_contract": list(_value(item, "answer_values", ()) or ()),
                "answer_verification_status": str(
                    _value(item, "answer_verification_status", "")
                ),
                "numeric_unit": _value(item, "numeric_unit", None),
            }
        )
    prerequisites = getattr(catalog, "_prerequisites", {}) or {}
    prerequisite_rows = {
        str(node): sorted(str(value) for value in (values or ()))
        for node, values in sorted(prerequisites.items())
    }
    persona_rows = [
        persona.to_dict() if callable(getattr(persona, "to_dict", None)) else dict(persona)
        for persona in personas
    ]
    persona_rows.sort(key=lambda row: str(row.get("persona_id", "")))
    source_files = _official_source_files(catalog, kg)
    summary = {
        "catalog_item_count": len(items),
        "catalog_public_question_sha256": _sha256_json(public_rows),
        "catalog_scoring_contract_sha256": _sha256_json(scoring_rows),
        "catalog_prerequisites_sha256": _sha256_json(prerequisite_rows),
        "kg_common_failures_sha256": _sha256_json(persona_rows),
        "persona_count": len(persona_rows),
        "source_files": source_files,
    }
    return {
        "official_input_sha256": _sha256_json(summary),
        "official_inputs": summary,
    }


def _official_source_files(catalog: Any, kg: Any) -> list[dict[str, Any]]:
    roles_by_path: dict[Path, set[str]] = {}
    for raw in tuple(getattr(catalog, "_metadata_paths", ()) or ()):
        path = Path(raw).expanduser().resolve(strict=False)
        roles_by_path.setdefault(path, set()).add("catalog_source")
    kg_path = getattr(kg, "_kg_file", None)
    if kg_path is not None:
        path = Path(kg_path).expanduser().resolve(strict=False)
        roles_by_path.setdefault(path, set()).add("kg_common_failures_source")
    rows = []
    for path, roles in sorted(roles_by_path.items(), key=lambda pair: str(pair[0])):
        if not path.is_file():
            raise RuntimeError(f"S2 official input source is missing: {path}")
        payload = path.read_bytes()
        rows.append(
            {
                "path": str(path),
                "roles": sorted(roles),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def _value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
