from __future__ import annotations

import hashlib
import json

import pytest

from analysis.dataset import DatasetContractError
from analysis.no_repeat import validate_no_repeat_sets


def _set_hash(budget: int, targets: list[str]) -> str:
    material = {
        "budget": budget,
        "targets": targets,
        "config_sha256": "config-sha",
        "census_summary_sha256": "summary-sha",
        "census_records_sha256": "records-sha",
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest() -> dict[str, object]:
    return {
        "config_sha256": "config-sha",
        "input_sha256": {
            "census_summary": {"sha256": "summary-sha"},
            "census_records": {"sha256": "records-sha"},
        },
    }


def _journey(target: str, member_budgets: set[int]) -> dict[str, object]:
    members = {
        9: ["T-a", "T-b"],
        15: ["T-a"],
        25: [],
    }
    unique_through = 15 if target == "T-a" else 9
    events = [
        {
            "item_id": f"item-{position if position <= unique_through else 1}",
            "family_id": f"family-{position if position <= unique_through else 1}",
        }
        for position in range(1, 26)
    ]
    return {
        "target_node": target,
        "events": events,
        "views": [
            {
                "nominal_budget": budget,
                "common_support_no_repeat": budget in member_budgets,
                "common_support_set_sha256": _set_hash(budget, members[budget]),
            }
            for budget in (9, 15, 25)
        ],
    }


def test_no_repeat_sets_expose_target_list_count_and_recomputed_hash() -> None:
    rows = [
        _journey("T-a", {9, 15}),
        _journey("T-a", {9, 15}),
        _journey("T-b", {9}),
    ]

    result = validate_no_repeat_sets(rows, _manifest())

    assert result[9].targets == ("T-a", "T-b")
    assert result[9].n_target == 2
    assert result[9].sha256 == _set_hash(9, ["T-a", "T-b"])
    assert result[15].targets == ("T-a",)
    assert result[25].targets == ()


def test_no_repeat_sets_reject_flag_or_hash_drift() -> None:
    rows = [_journey("T-a", {9, 15}), _journey("T-a", {9, 15})]
    rows[1]["views"][0]["common_support_no_repeat"] = False  # type: ignore[index]
    with pytest.raises(DatasetContractError, match="no-repeat flag drift"):
        validate_no_repeat_sets(rows, _manifest())

    rows = [_journey("T-a", {9, 15})]
    rows[0]["views"][0]["common_support_set_sha256"] = "bad"  # type: ignore[index]
    with pytest.raises(DatasetContractError, match="no-repeat set hash"):
        validate_no_repeat_sets(rows, _manifest())


def test_no_repeat_membership_is_verified_from_event_prefixes() -> None:
    row = _journey("T-a", {9, 15})
    row["events"][4]["item_id"] = "item-1"  # type: ignore[index]

    with pytest.raises(DatasetContractError, match="declared no-repeat member"):
        validate_no_repeat_sets([row], _manifest())
