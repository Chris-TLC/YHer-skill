"""Replay labeled scenarios without network calls or writes to real student logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from adapters.store.local_json import LocalJsonStore
from core.learning.curriculum import CurriculumRuntime
from core.learning.item_catalog import CatalogItem, ItemCatalog
from core.learning.session_service import SessionError, SessionService

from demo.synthetic_scenarios.validate import load_suite, validate_suite


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/tmp/yher_synthetic_demo_replays")
PROTECTED_ROOTS = (
    REPO_ROOT / "data" / "local_store",
    REPO_ROOT / "data" / "study_logs",
)
TOTAL_MINUTES = {"30min": 30, "1h": 60, "2h": 120}


class ReplayClock:
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class SyntheticJsonStore(LocalJsonStore):
    """Local store whose projections also retain an explicit synthetic marker."""

    def save_student(self, user_id: str, model: dict[str, Any]) -> None:
        super().save_student(user_id, {**model, "synthetic": True})


class SyntheticExplanationProvider:
    def generate(self, *, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        del prompt
        node = str(context.get("node") or "本考点")
        return {
            "content": {
                "title": f"{node}合成演示复盘",
                "diagnosis": "这是隔离的 SYNTHETIC_DEMO 讲解，不代表真人学习结果。",
                "worked_example": "按服务端已验证步骤列出条件、变量和结论，再逐项回查。",
                "causal_chain": ["读取条件", "确定变量", "应用关系", "回查结论"],
                "exam_strategy": ["圈限定词", "写中间关系", "核对单位与结论"],
                "analogy_used": False,
            },
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "cost_yuan": 0.0,
        }


def synthetic_grader(_item: CatalogItem, answer: str) -> dict[str, Any]:
    correct = answer == "SYNTHETIC_CORRECT"
    return {
        "correct": correct,
        "error_code": None if correct else "synthetic_incorrect",
        "confidence": 1.0,
        "likelihood": [0.75, 0.08, 0.09, 0.08] if correct else [0.08, 0.16, 0.56, 0.20],
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "cost_yuan": 0.0,
    }


def assert_isolated_output(output_dir: Path) -> Path:
    resolved = Path(output_dir).expanduser().resolve()
    if resolved == REPO_ROOT or resolved.is_relative_to(REPO_ROOT):
        raise ValueError("synthetic replay output must remain outside the repository")
    for protected in PROTECTED_ROOTS:
        protected = protected.resolve()
        if resolved == protected or resolved.is_relative_to(protected):
            raise ValueError("synthetic replay output cannot target real student logs")
    repo_data = (REPO_ROOT / "data").resolve()
    if resolved == repo_data or resolved.is_relative_to(repo_data):
        raise ValueError("synthetic replay output cannot target repository data")
    return resolved


def replay_suite(scenario_root: Path, output_dir: Path) -> dict[str, Any]:
    root = Path(scenario_root).resolve()
    output = assert_isolated_output(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("synthetic replay output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    manifest, scenarios = load_suite(root)
    catalog = ItemCatalog.from_default_data()
    matrix = validate_suite(manifest, scenarios, catalog=catalog)
    curriculum = CurriculumRuntime.from_default_asset()
    store = SyntheticJsonStore(output / "store")
    transcripts: list[dict[str, Any]] = []
    expected_closed = 0
    failures: list[str] = []

    for scenario_index, scenario in enumerate(scenarios, start=1):
        transcript = _replay_scenario(
            scenario,
            scenario_index=scenario_index,
            catalog=catalog,
            curriculum=curriculum,
            store=store,
        )
        transcripts.append(transcript)
        expected_closed += sum(
            episode["status"] == "expected_closed" for episode in transcript["episodes"]
        )
        failures.extend(transcript["unexpected_failures"])
        _write_json(output / "transcripts" / f"{scenario['scenario_id']}.json", transcript)

    persisted_synthetic = _persisted_rows_are_synthetic(output / "store")
    normalized = {
        "synthetic": True,
        "matrix": matrix,
        "transcripts": transcripts,
    }
    digest = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "synthetic": True,
        "scenario_count": len(scenarios),
        "episode_count": sum(len(row["episodes"]) for row in transcripts),
        "expected_closed_episodes": expected_closed,
        "unexpected_failures": failures,
        "all_persisted_rows_synthetic": persisted_synthetic,
        "digest": digest,
        "matrix": matrix,
    }
    _write_json(output / "replay_summary.json", summary)
    return summary


def _replay_scenario(
    scenario: dict[str, Any],
    *,
    scenario_index: int,
    catalog: ItemCatalog,
    curriculum: CurriculumRuntime,
    store: SyntheticJsonStore,
) -> dict[str, Any]:
    clock = ReplayClock(1_783_900_000 + scenario_index * 10_000)
    id_factory = _id_factory(str(scenario["scenario_id"]))
    service = SessionService(
        catalog,
        store,
        clock=clock,
        id_factory=id_factory,
        llm_grader=synthetic_grader,
        explanation_provider=SyntheticExplanationProvider(),
        curriculum=curriculum,
        synthetic=True,
    )
    episodes: list[dict[str, Any]] = []
    failures: list[str] = []
    for episode in scenario["sessions"]:
        result = _replay_episode(service, store, catalog, clock, scenario, episode)
        episodes.append(result)
        failures.extend(result.get("unexpected_failures") or [])
        clock.advance(3_600)
    return {
        "schema_version": "yher.synthetic_replay.v1",
        "synthetic": True,
        "scenario_id": scenario["scenario_id"],
        "user_id": scenario["student"]["user_id"],
        "budget_tier": scenario["budget_tier"],
        "outcome": scenario["outcome"],
        "episodes": episodes,
        "unexpected_failures": failures,
    }


def _replay_episode(
    service: SessionService,
    store: SyntheticJsonStore,
    catalog: ItemCatalog,
    clock: ReplayClock,
    scenario: dict[str, Any],
    episode: dict[str, Any],
) -> dict[str, Any]:
    expected_availability = episode["expected_availability"]
    try:
        started = service.start_session(
            scenario["student"]["user_id"],
            episode["node"],
            scenario["budget_tier"],
            grade=scenario["student"]["grade"],
            learning_purpose=scenario["student"]["learning_purpose"],
        )
    except SessionError as exc:
        if expected_availability == "closed":
            return {
                "synthetic": True,
                "episode_id": episode["episode_id"],
                "node": episode["node"],
                "status": "expected_closed",
                "reason": str(exc),
                "unexpected_failures": [],
            }
        return _failed_episode(episode, f"unexpected start rejection: {exc}")
    if expected_availability == "closed":
        return _failed_episode(episode, "closed episode unexpectedly started")

    session_id = started["session_id"]
    outcome = scenario["outcome"]
    submissions = 0
    budget_forced = False
    step = service.next_assignment(session_id)
    for _ in range(100):
        if (
            outcome == "partial"
            and step.get("budget_exhausted") is True
            and step.get("next_action") == "resume"
        ):
            service.resume_session(session_id)
            step = service.next_assignment(session_id)
            continue
        if step.get("done") is True:
            report = step.get("report")
            if report is None:
                try:
                    report = service.report(session_id)
                except SessionError:
                    step = service.next_assignment(session_id)
                    continue
            expected = outcome if outcome != "paused" else None
            actual = report.get("outcome")
            failures = [] if actual == expected else [f"{episode['episode_id']}: expected {expected}, got {actual}"]
            return {
                "synthetic": True,
                "episode_id": episode["episode_id"],
                "node": episode["node"],
                "session_id": session_id,
                "status": "completed",
                "outcome": actual,
                "submissions": submissions,
                "unexpected_failures": failures,
            }
        if step.get("phase") == "complete":
            step = service.next_assignment(session_id)
            continue
        if step.get("phase") == "paused":
            if outcome == "partial" and step.get("budget_exhausted") is True:
                service.resume_session(session_id)
                step = service.next_assignment(session_id)
                continue
            return _failed_episode(episode, "unexpected paused checkpoint", session_id)
        if step.get("phase") == "learning":
            recommendations = step.get("recommendations") or []
            if recommendations and episode.get("watch") == "complete":
                rec = recommendations[0]
                service.record_watch(
                    session_id,
                    rec["rec_id"],
                    _stable_hex(f"watch:{episode['episode_id']}")[:32],
                    watched_seconds=float(rec.get("duration_seconds") or 0),
                    completed=True,
                )
            service.ack_learning(session_id, step["action_id"])
            step = service.next_assignment(session_id)
            continue

        assignment_id = str(step.get("assignment_id") or "")
        if not assignment_id:
            return _failed_episode(episode, "assignment id missing", session_id)
        submissions += 1
        correct = _should_answer_correct(outcome, step["phase"], submissions)
        answer = _answer_for_assignment(store, catalog, session_id, assignment_id, correct)
        service.submit(
            session_id,
            assignment_id,
            _stable_hex(f"submission:{episode['episode_id']}:{submissions}")[:32],
            answer,
        )
        clock.advance(45)

        if outcome == "paused" and submissions >= 1:
            paused = service.pause_session(session_id)
            failures = [] if paused.get("status") == "paused" else [f"{episode['episode_id']}: pause failed"]
            return {
                "synthetic": True,
                "episode_id": episode["episode_id"],
                "node": episode["node"],
                "session_id": session_id,
                "status": "paused",
                "outcome": "paused",
                "submissions": submissions,
                "unexpected_failures": failures,
            }
        if outcome == "partial" and not budget_forced:
            clock.advance((TOTAL_MINUTES[scenario["budget_tier"]] + 1) * 60)
            budget_forced = True
        step = service.next_assignment(session_id)
    return _failed_episode(episode, "episode exceeded replay step limit", session_id)


def _should_answer_correct(outcome: str, phase: str, submissions: int) -> bool:
    if outcome == "verified":
        return True
    if outcome == "needs_reinforcement":
        return phase != "held_out" and submissions % 2 == 0
    if outcome == "partial":
        return submissions % 2 == 1
    return submissions % 2 == 1


def _answer_for_assignment(
    store: SyntheticJsonStore,
    catalog: ItemCatalog,
    session_id: str,
    assignment_id: str,
    correct: bool,
) -> str:
    session = store.load_session(session_id) or {}
    item_id = (session.get("assignments") or {}).get(assignment_id, {}).get("item_id")
    item = catalog.items[str(item_id)]
    if item.scoring_mode == "free_llm":
        return "SYNTHETIC_CORRECT" if correct else "SYNTHETIC_INCORRECT"
    if correct:
        return str(item.answer_values[0])
    if item.scoring_mode == "mcq":
        expected = set(str(item.answer_values[0]).upper())
        return next((key for key in sorted(item.options) if key.upper() not in expected), "Z")
    expected = str(item.answer_values[0])
    try:
        numeric = float(expected)
        return str(numeric + max(1.0, abs(numeric)))
    except ValueError:
        return "0" if expected.strip() != "0" else "1"


def _failed_episode(
    episode: dict[str, Any], message: str, session_id: str | None = None
) -> dict[str, Any]:
    return {
        "synthetic": True,
        "episode_id": episode["episode_id"],
        "node": episode["node"],
        "session_id": session_id,
        "status": "failed",
        "unexpected_failures": [f"{episode['episode_id']}: {message}"],
    }


def _id_factory(seed: str):
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return _stable_hex(f"{seed}:{counter}")[:32]

    return next_id


def _stable_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _persisted_rows_are_synthetic(store_root: Path) -> bool:
    for path in sorted(store_root.rglob("*.json")):
        if json.loads(path.read_text(encoding="utf-8")).get("synthetic") is not True:
            return False
    for path in sorted(store_root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("synthetic") is not True:
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = replay_suite(args.root, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["unexpected_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
