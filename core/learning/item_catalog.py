"""R5-only item catalog with trusted v3 metadata and content-family deduplication.

The v4 record owns every student-visible stem and every answer key.  A trusted
v3 alignment may supply only difficulty, question type, and display options.
Unaligned R5 rows stay measurable in ``CatalogStats`` but can never be served.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core.data.item_bank_v4 import V4_MANIFEST, V4_USABILITY_R5, iter_service_items


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3_MANIFEST = REPO_ROOT / "data/item_bank/chemistry_v3_6695.jsonl"
DEFAULT_KG_MANIFEST = REPO_ROOT / "data/knowledge_graph_150_enriched.jsonl"
TRUSTED_ALIGNMENT_STATUSES = frozenset({"auto_inherited", "manual_inherited"})
DIFFICULTY_MAP = {"T1": 0.25, "T2": 0.5, "T3": 0.75, "T4": 1.0}
CHOICE_TYPES = frozenset({"选择题", "单选题", "多选题"})


def map_difficulty(value: Any) -> float:
    """Map the only accepted v3 difficulty vocabulary to [0, 1]."""
    try:
        return DIFFICULTY_MAP[str(value).strip().upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported difficulty: {value!r}") from exc


def map_item_type(value: Any) -> str:
    """Map v3 display type. Non-choice rows need answer-shape classification."""
    normalized = str(value).strip()
    return "mcq" if normalized in CHOICE_TYPES or "选择" in normalized else "free"


@dataclass(frozen=True)
class CatalogItem:
    """Internal item contract. ``answer_values`` must never cross the API boundary."""

    item_id: str
    family_id: str
    aligned_item_id: str
    alignment_status: str
    node_ids: tuple[str, ...]
    stem_blocks: tuple[dict[str, Any], ...]
    stem_text: str
    stem_hash: str
    stem_normalized: str
    options: Mapping[str, str]
    difficulty: float
    item_type: str
    scoring_mode: str
    answer_values: tuple[str, ...]
    rubric: tuple[dict[str, Any], ...] = ()
    has_media: bool = False
    numeric_unit: str | None = None
    source_label: str = ""

    @property
    def deterministic(self) -> bool:
        return self.scoring_mode in {"mcq", "numeric"} and not self.has_media

    def public_question(self) -> dict[str, Any]:
        """Build a whitelist-only pre-submit view; no private dict is copied."""
        return {
            "kind": self.item_type,
            "stem_blocks": list(self.stem_blocks),
            "stem_text": self.stem_text,
            "options": dict(self.options),
            "difficulty": self.difficulty,
            "nodes": list(self.node_ids),
            "source_label": self.source_label,
        }


@dataclass(frozen=True)
class CatalogStats:
    r5_rows: int
    trusted_items: int
    rejected_items: int
    families: int


class _UnionFind:
    def __init__(self, keys: Iterable[str]):
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            key, self.parent[key] = self.parent[key], root
        return root

    def union(self, left: str, right: str) -> None:
        lroot, rroot = self.find(left), self.find(right)
        if lroot != rroot:
            self.parent[max(lroot, rroot)] = min(lroot, rroot)


class ItemCatalog:
    """Immutable, process-local view of questions eligible for the Demo loop."""

    def __init__(
        self,
        items: Sequence[CatalogItem],
        *,
        r5_rows: int,
        rejected: Mapping[str, str] | None = None,
        metadata_paths: Sequence[Path] = (),
        prerequisites: Mapping[str, Sequence[str]] | None = None,
    ):
        self.items = {item.item_id: item for item in items}
        self.rejected = dict(rejected or {})
        self._r5_rows = int(r5_rows)
        self._metadata_paths = tuple(Path(path) for path in metadata_paths)
        self._prerequisites = {
            str(node): tuple(sorted({str(value) for value in values if str(value)}))
            for node, values in (prerequisites or {}).items()
        }
        by_node: dict[str, list[CatalogItem]] = defaultdict(list)
        for item in items:
            for node in item.node_ids:
                by_node[node].append(item)
        self._by_node = {node: tuple(rows) for node, rows in by_node.items()}

    @classmethod
    def from_items(cls, items: Sequence[CatalogItem]) -> "ItemCatalog":
        return cls(tuple(items), r5_rows=len(items))

    @classmethod
    def from_default_data(
        cls,
        *,
        v4_path: Path = V4_MANIFEST,
        r5_path: Path = V4_USABILITY_R5,
        v3_path: Path = DEFAULT_V3_MANIFEST,
        kg_path: Path = DEFAULT_KG_MANIFEST,
    ) -> "ItemCatalog":
        v4_path, r5_path, v3_path, kg_path = (
            Path(v4_path),
            Path(r5_path),
            Path(v3_path),
            Path(kg_path),
        )
        # Use the existing R1-R5 loader, including the allow-list's fail-closed rule.
        r5_pool = {
            item["item_id"]: item
            for item in iter_service_items(
                manifest_path=v4_path,
                usability_path=r5_path,
                apply_r5=True,
            )
        }
        v3_index = {
            row["item_id"]: row
            for row in _iter_jsonl(v3_path)
            if isinstance(row.get("item_id"), str)
        }

        trusted_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        rejected: dict[str, str] = {}
        for item_id, v4 in r5_pool.items():
            alignment = v4.get("alignment") or {}
            status = str(alignment.get("status") or "")
            aligned_id = str(alignment.get("aligned_item_id") or "")
            if status not in TRUSTED_ALIGNMENT_STATUSES:
                rejected[item_id] = "untrusted_alignment_status"
                continue
            v3 = v3_index.get(aligned_id)
            if not v3:
                rejected[item_id] = "trusted_alignment_target_missing"
                continue
            trusted_rows.append((v4, v3))

        family_ids = _content_family_ids([v4 for v4, _ in trusted_rows])
        items = [
            _adapt_item(v4, v3, family_ids[v4["item_id"]])
            for v4, v3 in trusted_rows
        ]
        prerequisite_sets: dict[str, set[str]] = defaultdict(set)
        for row in _iter_jsonl(kg_path):
            node_id = str(row.get("node_id") or "")
            parent = str(row.get("parent_node") or "")
            values = {str(value) for value in (row.get("prerequisites") or []) if str(value)}
            if node_id:
                prerequisite_sets[node_id].update(values)
            if parent:
                prerequisite_sets[parent].update(values)
        return cls(
            items,
            r5_rows=len(r5_pool),
            rejected=rejected,
            metadata_paths=(v4_path, r5_path, v3_path, kg_path),
            prerequisites=prerequisite_sets,
        )

    def stats(self) -> CatalogStats:
        return CatalogStats(
            r5_rows=self._r5_rows,
            trusted_items=len(self.items),
            rejected_items=len(self.rejected),
            families=len({item.family_id for item in self.items.values()}),
        )

    def for_node(self, node: str, *, deterministic_only: bool = False) -> tuple[CatalogItem, ...]:
        items = self._by_node.get(node, ())
        if deterministic_only:
            return tuple(item for item in items if item.deterministic)
        return items

    def open_nodes(self, minimum_families: int = 5) -> dict[str, int]:
        opened: dict[str, int] = {}
        for node, items in self._by_node.items():
            count = len({item.family_id for item in items if item.deterministic})
            if count >= minimum_families:
                opened[node] = count
        return dict(sorted(opened.items()))

    def data_metadata(self) -> list[dict[str, Any]]:
        return [_file_metadata(path) for path in self._metadata_paths]

    def prerequisites_for(self, node: str) -> tuple[str, ...]:
        return self._prerequisites.get(str(node), ())


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _content_family_ids(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    ids = [str(row["item_id"]) for row in rows]
    union = _UnionFind(ids)
    identifier_owner: dict[tuple[str, str], str] = {}
    for row in rows:
        item_id = str(row["item_id"])
        alignment = row.get("alignment") or {}
        identifiers = (
            ("aligned", str(alignment.get("aligned_item_id") or "")),
            ("hash", str(row.get("stem_hash") or "")),
            ("normalized", str(row.get("stem_normalized") or "")),
        )
        for key in identifiers:
            if not key[1]:
                continue
            previous = identifier_owner.setdefault(key, item_id)
            union.union(item_id, previous)

    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        members[union.find(str(row["item_id"]))].append(row)
    output: dict[str, str] = {}
    for group in members.values():
        aligned_ids = sorted(
            str((row.get("alignment") or {}).get("aligned_item_id") or "")
            for row in group
            if (row.get("alignment") or {}).get("aligned_item_id")
        )
        if aligned_ids:
            family_id = f"aligned:{aligned_ids[0]}"
        else:  # Defensive fallback; trusted catalog rows currently always align.
            material = "|".join(sorted(str(row["item_id"]) for row in group))
            family_id = f"stem:{hashlib.sha256(material.encode()).hexdigest()[:20]}"
        for row in group:
            output[str(row["item_id"])] = family_id
    return output


def _adapt_item(v4: dict[str, Any], v3: dict[str, Any], family_id: str) -> CatalogItem:
    from core.learning.scoring import normalize_mcq, parse_scalar_answer

    solution = v4.get("standard_solution") or {}
    answer_values = tuple(
        str(value).strip()
        for value in (solution.get("final_answers") or [])
        if str(value).strip()
    )
    if not answer_values and str(solution.get("standard_answer") or "").strip():
        answer_values = (str(solution["standard_answer"]).strip(),)
    display_type = map_item_type(v3.get("question_type"))
    parsed = parse_scalar_answer(answer_values[0]) if len(answer_values) == 1 else None
    mcq_key = normalize_mcq(answer_values[0]) if len(answer_values) == 1 else None
    has_options = bool(v3.get("options"))
    if display_type == "mcq" and mcq_key is not None and has_options:
        item_type, scoring_mode, numeric_unit = "mcq", "mcq", None
    elif display_type == "mcq":
        item_type, scoring_mode, numeric_unit = "mcq", "free_llm", None
    elif parsed is not None:
        item_type, scoring_mode, numeric_unit = "numeric", "numeric", parsed.unit
    else:
        item_type, scoring_mode, numeric_unit = "free", "free_llm", None
    source = str(v4.get("group_key") or Path(str(v4.get("source_path") or "")).stem)
    return CatalogItem(
        item_id=str(v4["item_id"]),
        family_id=family_id,
        aligned_item_id=str((v4.get("alignment") or {})["aligned_item_id"]),
        alignment_status=str((v4.get("alignment") or {})["status"]),
        node_ids=tuple(str(node) for node in (v4.get("kg_nodes") or []) if str(node)),
        stem_blocks=tuple(v4.get("stem_blocks") or ()),
        stem_text=str(v4.get("stem_text") or ""),
        stem_hash=str(v4.get("stem_hash") or ""),
        stem_normalized=str(v4.get("stem_normalized") or ""),
        options={str(k): str(value) for k, value in (v3.get("options") or {}).items()},
        difficulty=map_difficulty(v3.get("difficulty")),
        item_type=item_type,
        scoring_mode=scoring_mode,
        answer_values=answer_values,
        rubric=tuple(v4.get("rubric") or ()),
        has_media=_contains_media(v4.get("stem_blocks") or ()),
        numeric_unit=numeric_unit,
        source_label=source,
    )


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.md5()  # nosec B324 - content fingerprint, not a security primitive
    lines = 0
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            lines += 1
    return {"name": path.name, "path": str(path), "lines": lines, "md5": digest.hexdigest()}


def _contains_media(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("media"):
            return True
        return any(_contains_media(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_media(child) for child in value)
    return False
