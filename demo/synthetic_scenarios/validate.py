"""Validate the labeled synthetic scenario matrix against the live catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from core.learning.item_catalog import ItemCatalog


BUDGETS = ("30min", "1h", "2h")
OUTCOMES = ("verified", "needs_reinforcement", "partial", "paused")
SCHEMA_VERSION = "yher.synthetic_scenario.v1"


def load_suite(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    scenarios: list[dict[str, Any]] = []
    for row in manifest.get("cases") or []:
        relative = Path(str(row.get("file") or ""))
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("scenario path escapes the synthetic suite")
        scenario = json.loads(path.read_text(encoding="utf-8"))
        if scenario.get("scenario_id") != row.get("scenario_id"):
            raise ValueError(f"manifest id mismatch for {relative}")
        scenarios.append(scenario)
    return manifest, scenarios


def validate_suite(
    manifest: dict[str, Any],
    scenarios: list[dict[str, Any]],
    *,
    catalog: ItemCatalog | None = None,
) -> dict[str, Any]:
    _require(manifest.get("synthetic") is True, "manifest must be synthetic:true")
    _require(len(scenarios) == 24, "suite must contain exactly 24 scenarios")
    ids = [str(row.get("scenario_id") or "") for row in scenarios]
    _require(all(ids) and len(set(ids)) == len(ids), "scenario ids must be unique")

    matrix: Counter[str] = Counter()
    planned_nodes: set[str] = set()
    episode_count = 0
    session_counts: Counter[int] = Counter()
    for scenario in scenarios:
        _validate_scenario(scenario)
        budget = str(scenario["budget_tier"])
        outcome = str(scenario["outcome"])
        matrix[f"{budget}:{outcome}"] += 1
        episodes = scenario["sessions"]
        session_counts[len(episodes)] += 1
        episode_count += len(episodes)
        planned_nodes.update(str(row["node"]) for row in episodes)

    expected_matrix = {
        f"{budget}:{outcome}": 2 for budget in BUDGETS for outcome in OUTCOMES
    }
    _require(dict(sorted(matrix.items())) == dict(sorted(expected_matrix.items())),
             "budget/outcome matrix must contain every combination exactly twice")
    _require(session_counts == Counter({1: 16, 2: 8}),
             "suite must contain 16 single-session and 8 double-session scenarios")
    _require(episode_count == 32, "suite must contain exactly 32 episodes")
    _require(len(planned_nodes) == 28, "suite plan must cover exactly 28 distinct nodes")

    actual_catalog = catalog or ItemCatalog.from_default_data()
    open_nodes = set(actual_catalog.open_nodes())
    closed_planned = sorted(planned_nodes - open_nodes)
    extra_open = sorted(open_nodes - planned_nodes)
    _require(not extra_open, f"open nodes missing from scenario plan: {extra_open}")
    for scenario in scenarios:
        for episode in scenario["sessions"]:
            expected = "open" if episode["node"] in open_nodes else "closed"
            _require(episode["expected_availability"] == expected,
                     f"availability mismatch for {episode['episode_id']}")

    return {
        "synthetic": True,
        "scenario_count": len(scenarios),
        "episode_count": episode_count,
        "single_session_scenarios": session_counts[1],
        "double_session_scenarios": session_counts[2],
        "planned_node_count": len(planned_nodes),
        "current_open_node_count": len(open_nodes),
        "closed_planned_nodes": closed_planned,
        "budget_outcome_counts": dict(sorted(matrix.items())),
    }


def _validate_scenario(scenario: dict[str, Any]) -> None:
    _require(scenario.get("schema_version") == SCHEMA_VERSION, "invalid schema version")
    _require(scenario.get("synthetic") is True, "every scenario must be synthetic:true")
    _require(scenario.get("budget_tier") in BUDGETS, "invalid budget tier")
    _require(scenario.get("outcome") in OUTCOMES, "invalid outcome")
    student = scenario.get("student") or {}
    _require(str(student.get("user_id") or "").startswith("SYNTHETIC_DEMO_"),
             "synthetic user id must use the reserved prefix")
    sessions = scenario.get("sessions")
    _require(isinstance(sessions, list) and len(sessions) in {1, 2},
             "each scenario must contain one or two sessions")
    episode_ids: set[str] = set()
    for episode in sessions:
        _require(episode.get("synthetic") is True, "every episode must be synthetic:true")
        episode_id = str(episode.get("episode_id") or "")
        _require(episode_id and episode_id not in episode_ids, "episode ids must be unique")
        episode_ids.add(episode_id)
        _require(bool(str(episode.get("node") or "").strip()), "episode node is required")
        _require(episode.get("expected_availability") in {"open", "closed"},
                 "invalid expected availability")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    manifest, scenarios = load_suite(args.root)
    print(json.dumps(validate_suite(manifest, scenarios), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
