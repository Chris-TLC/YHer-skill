from __future__ import annotations

from copy import deepcopy

import pytest

from analysis.contracts import validate_programmatic_grid
from analysis.dataset import DatasetContractError


def _journey(replicate: int, arm: str) -> dict[str, object]:
    noises = [0.1, 0.2] if arm != "B" else [0.1]
    persona = f"confirmatory:T:M:matched:{replicate}:{arm}"
    return {
        "record_type": "confirmatory_journey",
        "target_node": "T",
        "truth": "M",
        "condition": "matched",
        "replicate": replicate,
        "arm": arm,
        "persona_id": persona,
        "held_out_outcomes": {"h1": True, "h2": False},
        "held_out_pairs": (("held-family-1", "h1"), ("held-family-2", "h2")),
        "events": [
            {
                "target_node": "T",
                "truth": "M",
                "condition": "matched",
                "replicate": replicate,
                "arm": arm,
                "persona_id": persona,
                "position": position,
                "response_noise": noise,
            }
            for position, noise in enumerate(noises, start=1)
        ],
        "views": [],
    }


def _grid() -> list[dict[str, object]]:
    return [
        _journey(replicate, arm)
        for replicate in range(2)
        for arm in ("A", "B", "C")
    ]


def test_grid_contract_validates_primary_keys_replicates_and_paired_arms() -> None:
    result = validate_programmatic_grid(
        _grid(),
        expected_journeys=6,
        expected_replicates=2,
    )

    assert result.journey_count == 6
    assert result.primary_key_count == 6
    assert result.pair_count == 2
    assert result.target_count == 1


def test_grid_contract_rejects_duplicate_programmatic_primary_key() -> None:
    rows = _grid()
    rows.append(deepcopy(rows[0]))

    with pytest.raises(DatasetContractError, match="duplicate programmatic primary key"):
        validate_programmatic_grid(rows, expected_journeys=7, expected_replicates=2)


def test_grid_contract_rejects_an_incomplete_three_arm_pair() -> None:
    rows = _grid()
    rows.pop()

    with pytest.raises(DatasetContractError, match="paired arms"):
        validate_programmatic_grid(rows, expected_journeys=5, expected_replicates=2)


def test_grid_contract_accounts_for_a_quarantined_intention_key() -> None:
    rows = _grid()
    invalid = rows.pop(0)
    invalid_key = tuple(
        invalid[field]
        for field in ("target_node", "truth", "condition", "replicate", "arm")
    )

    result = validate_programmatic_grid(
        rows,
        expected_journeys=6,
        expected_replicates=2,
        invalid_primary_keys=(invalid_key,),
    )

    assert result.journey_count == 6
    assert result.primary_key_count == 6
    assert result.pair_count == 2


def test_grid_contract_rejects_noncontiguous_replicates() -> None:
    rows = _grid()
    for row in rows[3:]:
        row["replicate"] = 2
        for event in row["events"]:  # type: ignore[index]
            event["replicate"] = 2

    with pytest.raises(DatasetContractError, match="replicate IDs"):
        validate_programmatic_grid(rows, expected_journeys=6, expected_replicates=2)


def test_grid_contract_rejects_event_metadata_or_position_drift() -> None:
    rows = _grid()
    rows[0]["events"][1]["position"] = 3  # type: ignore[index]

    with pytest.raises(DatasetContractError, match="event prefix"):
        validate_programmatic_grid(rows, expected_journeys=6, expected_replicates=2)


def test_grid_contract_rejects_cross_arm_noise_or_heldout_pairing_drift() -> None:
    rows = _grid()
    rows[1]["events"][0]["response_noise"] = 0.9  # type: ignore[index]

    with pytest.raises(DatasetContractError, match="response-noise prefix"):
        validate_programmatic_grid(rows, expected_journeys=6, expected_replicates=2)

    rows = _grid()
    rows[2]["held_out_outcomes"] = {"h1": False, "h2": False}
    with pytest.raises(DatasetContractError, match="held-out outcomes"):
        validate_programmatic_grid(rows, expected_journeys=6, expected_replicates=2)

    rows = _grid()
    rows[2]["held_out_pairs"] = (
        ("held-family-1", "h1"),
        ("held-family-other", "h2"),
    )
    with pytest.raises(DatasetContractError, match="held-out family/item pairs"):
        validate_programmatic_grid(rows, expected_journeys=6, expected_replicates=2)
