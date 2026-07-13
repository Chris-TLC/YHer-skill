"""Deterministic persona construction from KG common-failure annotations."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.s0_census import normalize_kg_label

from .models import Persona


def _value(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _nodes(kg: Any) -> list[Any]:
    kg_path = getattr(kg, "_kg_file", None)
    if kg_path is not None and Path(kg_path).is_file():
        rows = []
        with Path(kg_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        rows.append(value)
        return rows
    if hasattr(kg, "all_nodes"):
        return list(kg.all_nodes())
    if isinstance(kg, Iterable):
        return list(kg)
    raise TypeError("kg must expose all_nodes() or be iterable")


def _failure_rows(
    kg: Any,
    eligible_nodes: set[str] | None = None,
) -> list[tuple[str, int, Any, str]]:
    rows: list[tuple[str, int, Any, str]] = []
    normalized_eligible: dict[str, list[str]] = defaultdict(list)
    for node in sorted(eligible_nodes or ()):
        normalized_eligible[normalize_kg_label(node)].append(node)
    for node in _nodes(kg):
        source_node_id = str(_value(node, "node_id", "")).strip()
        parent_node = str(_value(node, "parent_node", "")).strip()
        if not source_node_id:
            continue
        target_node = source_node_id
        if eligible_nodes is not None:
            if source_node_id in eligible_nodes:
                target_node = source_node_id
            elif parent_node in eligible_nodes:
                target_node = parent_node
            else:
                target_node = ""
                for label in (source_node_id, parent_node):
                    if not label:
                        continue
                    candidates = normalized_eligible.get(normalize_kg_label(label), ())
                    if len(candidates) == 1:
                        target_node = candidates[0]
                        break
                if not target_node:
                    continue
        failures = list(_value(node, "common_failures", ()) or ())
        for index, failure in enumerate(failures):
            # The row order is part of the frozen study design.  No response
            # data is consulted here.
            rows.append((target_node, index, failure, source_node_id))
    grouped: dict[str, list[tuple[str, int, Any, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)
    for group in grouped.values():
        group.sort(
            key=lambda row: (
                row[3],
                row[1],
                str(_value(row[2], "cause", "")),
                str(_value(row[2], "diagnostic_question", "")),
            )
        )
    # Round-robin targets before taking a second failure from any target.  The
    # 25 base personas therefore cover the breadth of the serviceable KG rather
    # than being dominated by alphabetically early child nodes.
    interleaved: list[tuple[str, int, Any, str]] = []
    max_rows = max((len(group) for group in grouped.values()), default=0)
    for rank in range(max_rows):
        for target in sorted(grouped):
            if rank < len(grouped[target]):
                interleaved.append(grouped[target][rank])
    return interleaved


def build_personas(
    kg: Any,
    *,
    pair_count: int = 25,
    seed: int = 2026071302,
    eligible_nodes: set[str] | frozenset[str] | None = None,
) -> list[Persona]:
    """Build exactly ``pair_count * 2`` pre-observation persona rows.

    A pair is anchored to one concrete KG ``common_failure``.  If the KG does
    not contain enough annotations, the function fails closed instead of
    inventing or duplicating a failure to reach the requested sample size.
    """

    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    eligible = {str(node) for node in eligible_nodes} if eligible_nodes is not None else None
    rows = _failure_rows(kg, eligible)
    if len(rows) < pair_count:
        raise ValueError(
            f"KG provides only {len(rows)} common failures; "
            f"{pair_count} persona pairs are required"
        )
    personas: list[Persona] = []
    for pair_index, (node_id, failure_index, failure, source_node_id) in enumerate(rows[:pair_count]):
        failure_id = f"{source_node_id}#failure-{failure_index:02d}"
        material = (
            f"yher-llm-persona-v1|{seed}|{pair_index}|{node_id}|"
            f"{source_node_id}|{failure_index}"
        )
        pair_seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
        fields = {
            "failure_cause": str(_value(failure, "cause", "")),
            "failure_symptom": str(_value(failure, "symptom", "")),
            "diagnostic_question": str(_value(failure, "diagnostic_question", "")),
        }
        pair_id = f"llm-pair:{pair_index:02d}:{node_id}"
        for strength in ("weak", "strong"):
            personas.append(
                Persona(
                    persona_id=f"{pair_id}:{strength}",
                    pair_id=pair_id,
                    strength=strength,
                    target_node=node_id,
                    failure_id=failure_id,
                    annotation_source="kg.common_failures",
                    seed=pair_seed,
                    **fields,
                )
            )
    return personas
