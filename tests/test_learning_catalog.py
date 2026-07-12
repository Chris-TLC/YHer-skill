"""Contracts for the canonical R5 learning-item catalog."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from core.learning.item_catalog import (
    TRUSTED_ALIGNMENT_STATUSES,
    ItemCatalog,
    map_difficulty,
    map_item_type,
)


REPO = Path(__file__).resolve().parents[1]
V4 = REPO / "data/item_bank/v4/chemistry_v4_1_3329.jsonl"
V3 = REPO / "data/item_bank/chemistry_v3_6695.jsonl"
KG = REPO / "data/knowledge_graph_150_enriched.jsonl"


def _row(path: Path, item_id: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("item_id") == item_id:
                return row
    raise AssertionError(f"missing fixture row {item_id}")


@pytest.fixture(scope="module")
def catalog() -> ItemCatalog:
    # This is deliberately read-only. No study-log or official-data writer is used.
    return ItemCatalog.from_default_data()


def test_actual_r5_pool_rejects_all_untrusted_crosswalk_rows(catalog: ItemCatalog):
    stats = catalog.stats()

    assert stats.r5_rows == 1202
    assert stats.trusted_items == 973
    assert stats.rejected_items == 229
    assert stats.families == 963
    assert len(catalog.items) == 973
    assert all(i.alignment_status in TRUSTED_ALIGNMENT_STATUSES for i in catalog.items.values())


def test_v4_owns_stem_and_answer_while_v3_only_supplies_metadata(catalog: ItemCatalog):
    item = next(iter(catalog.items.values()))
    v4 = _row(V4, item.item_id)
    v3 = _row(V3, item.aligned_item_id)

    assert item.stem_blocks == tuple(v4["stem_blocks"])
    assert item.answer_values == tuple(v4["standard_solution"]["final_answers"])
    assert item.difficulty == map_difficulty(v3["difficulty"])
    assert item.item_type == map_item_type(v3["question_type"])
    assert item.options == v3["options"]
    assert item.rubric == tuple(v4.get("rubric") or ())
    assert "rubric" not in item.public_question()
    assert item.answer_values != tuple(
        (v3.get("standard_solution") or {}).get("final_answers") or []
    ) or item.answer_values == tuple(v4["standard_solution"]["final_answers"])


def test_adapter_maps_are_explicit_and_fail_closed():
    assert [map_difficulty(x) for x in ("T1", "T2", "T3", "T4")] == [
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert map_item_type("选择题") == "mcq"
    assert map_item_type("单选题") == "mcq"
    assert map_item_type("多项选择") == "mcq"
    assert map_item_type("计算题") == "free"
    with pytest.raises(ValueError):
        map_difficulty("T5")


def test_custom_r5_path_is_the_actual_allowlist(tmp_path):
    v4_path = tmp_path / "v4.jsonl"
    r5_path = tmp_path / "r5.jsonl"
    v3_path = tmp_path / "v3.jsonl"
    v4_row = {
        "item_id": "custom-v4",
        "pool": "main",
        "service_eligible": True,
        "quality_flags": [],
        "answer_blocks_effective": [{"para": [{"type": "text", "text": "A"}]}],
        "standard_solution": {"final_answers": ["A"], "standard_answer": "A"},
        "alignment": {"status": "auto_inherited", "aligned_item_id": "custom-v3"},
        "kg_nodes": ["氧化还原反应"],
        "stem_blocks": [{"para": [{"type": "text", "text": "custom"}]}],
        "stem_text": "custom",
        "stem_hash": "custom-hash",
        "stem_normalized": "custom",
    }
    v3_row = {
        "item_id": "custom-v3",
        "difficulty": "T1",
        "question_type": "选择题",
        "options": {"A": "yes", "B": "no"},
    }
    v4_path.write_text(json.dumps(v4_row, ensure_ascii=False) + "\n", encoding="utf-8")
    r5_path.write_text(
        json.dumps({"item_id": "custom-v4", "r5_serve": True}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    v3_path.write_text(json.dumps(v3_row, ensure_ascii=False) + "\n", encoding="utf-8")

    catalog = ItemCatalog.from_default_data(v4_path=v4_path, r5_path=r5_path, v3_path=v3_path)

    assert set(catalog.items) == {"custom-v4"}


def test_content_family_unions_alignment_hash_and_normalized_duplicates(catalog: ItemCatalog):
    by_alignment: dict[str, list] = defaultdict(list)
    by_hash: dict[str, list] = defaultdict(list)
    by_normalized: dict[str, list] = defaultdict(list)
    for item in catalog.items.values():
        by_alignment[item.aligned_item_id].append(item)
        by_hash[item.stem_hash].append(item)
        by_normalized[item.stem_normalized].append(item)

    duplicate_groups = [
        group
        for index in (by_alignment, by_hash, by_normalized)
        for group in index.values()
        if len(group) > 1
    ]
    assert duplicate_groups, "actual R5 data must exercise family de-duplication"
    for group in duplicate_groups:
        assert len({item.family_id for item in group}) == 1


def test_open_nodes_have_five_independent_deterministic_families(catalog: ItemCatalog):
    opened = catalog.open_nodes()

    assert "氧化还原反应" in opened
    assert opened["氧化还原反应"] >= 5
    assert opened
    assert all(count >= 5 for count in opened.values())


def test_choice_without_a_normalizable_v4_key_is_not_deterministic(catalog: ItemCatalog):
    item = catalog.items["3c5f05431e0f95041ca531768ee2db659f5732ed"]

    assert item.options
    assert item.answer_values == ()
    assert item.scoring_mode == "free_llm"
    assert item.deterministic is False


def test_media_questions_fail_closed_until_a_safe_asset_route_exists(catalog: ItemCatalog):
    media_scored = [
        item
        for item in catalog.items.values()
        if item.has_media and item.scoring_mode in {"mcq", "numeric"}
    ]
    deterministic = [item for item in catalog.items.values() if item.deterministic]

    assert media_scored
    assert all(item.deterministic is False for item in media_scored)
    assert len(deterministic) == 409
    assert len(catalog.open_nodes()) == 28
    assert catalog.open_nodes()["氧化还原反应"] == 33


def test_health_metadata_has_exact_read_only_data_fingerprints(catalog: ItemCatalog):
    metadata = {entry["name"]: entry for entry in catalog.data_metadata()}

    assert metadata[V4.name]["lines"] == 3329
    assert metadata["usability_r5_v1.jsonl"]["lines"] >= 1207
    assert metadata[V3.name]["lines"] >= 6438
    assert metadata[KG.name]["lines"] == 135
    assert all(len(entry["md5"]) == 32 for entry in metadata.values())


def test_parent_node_aggregates_prerequisites_from_kg_children(catalog: ItemCatalog):
    assert set(catalog.prerequisites_for("氧化还原反应")) == {
        "物质分类",
        "化学计量（摩尔/阿伏伽德罗）",
    }
