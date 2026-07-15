"""Frozen P2 supply-bound illustrative analysis."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
from pathlib import Path
from typing import Any, Mapping


TARGETS = ("基本操作", "烷烃")
STATES = ("M", "P", "C", "U")

CANDIDATE_SOURCE = Path(
    "/tmp/yher-video-recommender-20260714/data/video_chunks/chemistry_chunks_v1.jsonl"
)
RUNTIME_SOURCE = Path(
    "/tmp/yher-video-recommender-20260714/core/learning/assets/"
    "curriculum_runtime_v1.json"
)
H1_MANIFEST_SOURCE = Path(
    "/Users/mac/Desktop/项目文件夹/Tools/yihuier-chemistry-skill/data/"
    "sim_store/confirmatory/confirmatory-v1/manifest.json"
)

EXPECTED_SOURCE_SHA256 = {
    "trusted_candidate_jsonl": (
        "9f14b8103eb191c7ffc5d2b1f1777e88b915082b94a3ac21f3c07f41a53f0406"
    ),
    "signed_runtime_metadata": (
        "6348b28805c75eddba73b39ef14c034f4c9aa0fd517a78b396440f865489dedd"
    ),
    "h1_h4_raw_manifest": (
        "2c68cada6c2229e6860d46fca4e4f65b3df674bfc4652b4a947934ba05e76dd3"
    ),
}
EXPECTED_SUBSET_AUDIT_SHA256 = (
    "b8ae2eaef4e047f75dbc2aa2a791188528115219660679b0fca70f530da7e2e2"
)
EXPECTED_SUBSET_CANONICAL_ROWS_SHA256 = (
    "e080008a40e514bf57e95a5c9905c9ba469c1e9c976b0e1ec465e684d55ff34d"
)
P2_SPEC_COMMIT = "d506fad14259f134d5a051e029dd885bdb693796"
P2_SPEC_SHA256 = "259cb64e3211a9f4bb309cebe7c415b9d7296d112b42f32dc2bdd373c9a61b83"
BOOTSTRAP_SEED = 2026071505
BOOTSTRAP_RESAMPLES = 10_000
BUDGET_SECONDS = 600
CLAIM_BOUNDARY = (
    "supply_bound_algorithmic_illustration_not_learning_benefit_or_external_validation"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
P2_SPEC_PATH = REPO_ROOT / "experiments" / "p2_illustrative_analysis_plan.md"
DEFAULT_OUTPUT_DIR = Path("/tmp/yher_h5v2/p2")
REQUIRED_OUTPUT_FILES = (
    "input_manifest.json",
    "candidate_subset.jsonl",
    "decision_instances_manifest.json",
    "selector_trace.jsonl",
    "profile_metrics.jsonl",
    "summary.json",
    "bootstrap.json",
    "figure_data.json",
    "p2_supply_bound_illustration.png",
    "p2_supply_bound_illustration.svg",
    "output_manifest.json",
)
ARM_NAMES = ("oracle", "A", "B", "C")
BOOTSTRAP_METRICS = (
    "selected_seconds",
    "selected_segment_count",
    "mismatched_selected_seconds",
    "missed_available_seconds",
    "unused_budget_seconds",
    "unobtainable_truth_slots",
    "unsupported_posterior_mass",
    "structural_failure_node_fraction",
    "missed_diagnostic_structural_failure_seconds",
    "missed_posterior_selection_seconds",
    "missed_budget_constraint_seconds",
)

EXPECTED_CANDIDATE_FACTS = {
    "BV1aComYMEms#P102#c000": ("基本操作", 496, "drill"),
    "BV18t4y1a7eD#P001#c000": ("烷烃", 110, "review"),
    "BV18t4y1a7eD#P001#c003": ("烷烃", 99, "review"),
    "BV18t4y1a7eD#P001#c004": ("烷烃", 94, "review"),
    "BV1JT421C7WS#P001#c004_b": ("烷烃", 28, "drill"),
    "BV1JT421C7WS#P001#c011": ("烷烃", 134, "drill"),
    "BV1JT421C7WS#P001#c014_b": ("烷烃", 68, "drill"),
    "BV1JT421C7WS#P001#c016": ("烷烃", 121, "drill"),
}


class P2ContractError(ValueError):
    """Raised when a frozen P2 contract is violated."""


class InputDriftError(P2ContractError):
    """Raised before parsing when source bytes differ from the frozen input."""


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    target: str
    physical_key: str
    charged_seconds: int
    role: str
    difficulty: str
    source_line: int
    raw_row: Mapping[str, Any]

    @property
    def slot_state(self) -> str:
        try:
            return {"review": "M", "drill": "C"}[self.role]
        except KeyError as exc:
            raise P2ContractError(f"unsupported candidate role: {self.role}") from exc


@dataclass(frozen=True)
class CandidateSubset:
    candidates: tuple[Candidate, ...]
    raw_rows: tuple[Mapping[str, Any], ...]
    audit_declared_sha256: str
    canonical_rows_sha256: str
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[Candidate, ...]
    total_seconds: int
    utility: Decimal
    trace: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class DiagnosticRecord:
    target: str
    truth: str
    arm: str
    replicate: int
    belief: tuple[Decimal, Decimal, Decimal, Decimal] | None
    diagnostic_status: str


@dataclass(frozen=True)
class DiagnosticDataset:
    records: tuple[DiagnosticRecord, ...]
    source_manifest_sha256: str
    shard_sha256: Mapping[str, str]


@dataclass(frozen=True)
class ProfileComputation:
    metrics: Mapping[str, Any]
    trace_record: Mapping[str, Any]
    selection: SelectionResult
    oracle_selection: SelectionResult


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise InputDriftError(
            f"{label} SHA-256 drift: expected {expected}, observed {actual}"
        )
    return actual


def _runtime_index(runtime: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    segments_by_node = runtime.get("segments_by_node")
    if not isinstance(segments_by_node, Mapping):
        raise P2ContractError("runtime segments_by_node is missing")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for target in TARGETS:
        rows = segments_by_node.get(target)
        if not isinstance(rows, list):
            raise P2ContractError(f"runtime target is missing: {target}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise P2ContractError(f"runtime row is not an object: {target}")
            physical_key = row.get("segment_id")
            if not isinstance(physical_key, str) or not physical_key:
                raise P2ContractError(f"runtime physical key is invalid: {target}")
            key = (target, physical_key)
            if key in result:
                raise P2ContractError(f"duplicate runtime physical key: {key}")
            result[key] = row
    return result


def load_candidate_subset(
    candidate_path: Path = CANDIDATE_SOURCE,
    runtime_path: Path = RUNTIME_SOURCE,
) -> CandidateSubset:
    source_hashes = {
        "trusted_candidate_jsonl": verify_sha256(
            candidate_path,
            EXPECTED_SOURCE_SHA256["trusted_candidate_jsonl"],
            label="trusted candidate JSONL",
        ),
        "signed_runtime_metadata": verify_sha256(
            runtime_path,
            EXPECTED_SOURCE_SHA256["signed_runtime_metadata"],
            label="signed runtime metadata",
        ),
    }
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(runtime, Mapping):
        raise P2ContractError("runtime root is not an object")
    runtime_index = _runtime_index(runtime)

    raw_rows: list[Mapping[str, Any]] = []
    candidates: list[Candidate] = []
    seen_chunk_ids: set[str] = set()
    with candidate_path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise P2ContractError(f"candidate line {source_line} is not an object")
            topics = row.get("knowledge_topic")
            if not isinstance(topics, list):
                continue
            exact_targets = [target for target in TARGETS if target in topics]
            if not exact_targets or row.get("needs_human") is not False:
                continue
            if len(exact_targets) != 1:
                raise P2ContractError(
                    f"candidate line {source_line} overlaps multiple frozen targets"
                )
            target = exact_targets[0]
            bv = row.get("bv")
            p_number = row.get("p_number")
            if not isinstance(bv, str) or not isinstance(p_number, int):
                raise P2ContractError(f"candidate line {source_line} has invalid source")
            physical_key = f"{bv}#P{p_number:03d}"
            runtime_row = runtime_index.get((target, physical_key))
            if runtime_row is None:
                continue
            if not isinstance(runtime_row.get("signed_entity"), str) or not runtime_row.get(
                "signed_entity"
            ):
                raise P2ContractError(f"runtime entity is unsigned: {physical_key}")
            chunk_id = row.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise P2ContractError(f"candidate line {source_line} has invalid chunk_id")
            if chunk_id in seen_chunk_ids:
                raise P2ContractError(f"duplicate candidate chunk_id: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            try:
                start = Decimal(str(row["start_sec"]))
                end = Decimal(str(row["end_sec"]))
            except (KeyError, ArithmeticError) as exc:
                raise P2ContractError(
                    f"candidate line {source_line} has invalid bounds"
                ) from exc
            if not start.is_finite() or not end.is_finite() or start < 0 or end <= start:
                raise P2ContractError(f"candidate line {source_line} has invalid bounds")
            charged_seconds = int((end - start).to_integral_value(rounding=ROUND_CEILING))
            role = runtime_row.get("seg_type")
            difficulty = runtime_row.get("difficulty")
            if not isinstance(role, str) or not isinstance(difficulty, str):
                raise P2ContractError(f"runtime role metadata is invalid: {physical_key}")
            raw_rows.append(dict(row))
            candidates.append(
                Candidate(
                    chunk_id=chunk_id,
                    target=target,
                    physical_key=physical_key,
                    charged_seconds=charged_seconds,
                    role=role,
                    difficulty=difficulty,
                    source_line=source_line,
                    raw_row=dict(row),
                )
            )

    candidates.sort(key=lambda candidate: candidate.chunk_id)
    raw_rows.sort(key=lambda row: str(row["chunk_id"]))
    observed_facts = {
        candidate.chunk_id: (
            candidate.target,
            candidate.charged_seconds,
            candidate.role,
        )
        for candidate in candidates
    }
    if observed_facts != EXPECTED_CANDIDATE_FACTS:
        raise P2ContractError("canonical eight-row candidate set drift")
    if len({candidate.physical_key for candidate in candidates}) != 3:
        raise P2ContractError("canonical candidate set does not have three physical sources")
    canonical_rows_sha256 = hashlib.sha256(canonical_json_bytes(raw_rows)).hexdigest()
    if canonical_rows_sha256 != EXPECTED_SUBSET_CANONICAL_ROWS_SHA256:
        raise P2ContractError("canonical candidate row serialization drift")
    return CandidateSubset(
        candidates=tuple(candidates),
        raw_rows=tuple(raw_rows),
        audit_declared_sha256=EXPECTED_SUBSET_AUDIT_SHA256,
        canonical_rows_sha256=canonical_rows_sha256,
        source_hashes=source_hashes,
    )


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value, "f")


def select_candidates(
    candidates: tuple[Candidate, ...] | list[Candidate],
    beliefs_by_target: Mapping[str, tuple[Decimal, Decimal, Decimal, Decimal]],
    *,
    failed_nodes: frozenset[str] = frozenset(),
    budget_seconds: int = 600,
) -> SelectionResult:
    if not isinstance(budget_seconds, int) or budget_seconds <= 0:
        raise P2ContractError("budget_seconds must be a positive integer")
    ordered_candidates = tuple(sorted(candidates, key=lambda value: value.chunk_id))
    if len({candidate.chunk_id for candidate in ordered_candidates}) != len(
        ordered_candidates
    ):
        raise P2ContractError("selector candidate chunk_id values must be unique")
    for target, belief in beliefs_by_target.items():
        if target in failed_nodes:
            raise P2ContractError(f"failed node must not carry a belief: {target}")
        if len(belief) != len(STATES):
            raise P2ContractError(f"belief must have four states: {target}")
        if any(not value.is_finite() or value < 0 or value > 1 for value in belief):
            raise P2ContractError(f"belief contains an invalid probability: {target}")
        if abs(sum(belief, Decimal(0)) - Decimal(1)) > Decimal("1e-12"):
            raise P2ContractError(f"belief is not normalized: {target}")

    selected: list[Candidate] = []
    used_physical: set[str] = set()
    covered_slots: set[tuple[str, str]] = set()
    total_seconds = 0
    total_utility = Decimal(0)
    trace: list[Mapping[str, Any]] = []

    with localcontext() as context:
        context.prec = 60
        while True:
            remaining_seconds = budget_seconds - total_seconds
            evaluations: list[dict[str, Any]] = []
            selectable: list[tuple[Candidate, Decimal, Decimal]] = []
            for candidate in ordered_candidates:
                slot = (candidate.target, candidate.slot_state)
                belief = beliefs_by_target.get(candidate.target)
                if belief is None or candidate.target in failed_nodes:
                    marginal = Decimal(0)
                elif slot in covered_slots:
                    marginal = Decimal(0)
                else:
                    marginal = belief[STATES.index(candidate.slot_state)]
                ratio = marginal / Decimal(candidate.charged_seconds)
                rejection_reason: str | None = None
                feasible = True
                if candidate.target in failed_nodes or belief is None:
                    feasible = False
                    rejection_reason = "structural_failure_mask"
                elif candidate.physical_key in used_physical:
                    feasible = False
                    rejection_reason = "physical_source_reuse"
                elif slot in covered_slots:
                    feasible = False
                    rejection_reason = "binary_saturation"
                elif candidate.charged_seconds > remaining_seconds:
                    feasible = False
                    rejection_reason = "budget"
                elif marginal <= 0:
                    rejection_reason = "zero_marginal_utility"
                else:
                    selectable.append((candidate, marginal, ratio))
                evaluations.append(
                    {
                        "chunk_id": candidate.chunk_id,
                        "physical_key": candidate.physical_key,
                        "target": candidate.target,
                        "slot_state": candidate.slot_state,
                        "charged_seconds": candidate.charged_seconds,
                        "marginal_utility": _decimal_text(marginal),
                        "marginal_utility_per_second": _decimal_text(ratio),
                        "feasible": feasible,
                        "rejection_reason": rejection_reason,
                        "selected": False,
                    }
                )

            chosen: tuple[Candidate, Decimal, Decimal] | None = None
            if selectable:
                chosen = sorted(
                    selectable,
                    key=lambda value: (
                        -value[2],
                        -value[1],
                        value[0].charged_seconds,
                        value[0].chunk_id,
                    ),
                )[0]
                chosen_id = chosen[0].chunk_id
                for evaluation in evaluations:
                    if evaluation["chunk_id"] == chosen_id:
                        evaluation["selected"] = True
                        evaluation["rejection_reason"] = None
                    elif evaluation["feasible"] and evaluation["rejection_reason"] is None:
                        evaluation["rejection_reason"] = "lower_priority_this_step"

            trace.append(
                {
                    "step": len(trace),
                    "budget_seconds": budget_seconds,
                    "remaining_seconds_before": remaining_seconds,
                    "selected_chunk_id": chosen[0].chunk_id if chosen else None,
                    "evaluations": evaluations,
                }
            )
            if chosen is None:
                break
            candidate, marginal, _ = chosen
            selected.append(candidate)
            used_physical.add(candidate.physical_key)
            covered_slots.add((candidate.target, candidate.slot_state))
            total_seconds += candidate.charged_seconds
            total_utility += marginal
            if total_seconds > budget_seconds:
                raise P2ContractError("selector exceeded the frozen budget")

    return SelectionResult(
        selected=tuple(selected),
        total_seconds=total_seconds,
        utility=total_utility,
        trace=tuple(trace),
    )


def posterior_from_b15(
    view: Mapping[str, Any],
    *,
    expected_valid: bool,
    context: str,
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal] | None, str]:
    if view.get("nominal_budget") != 15:
        raise P2ContractError(f"{context}: view is not the frozen b15 checkpoint")
    valid = view.get("valid")
    if not isinstance(valid, bool) or valid is not expected_valid:
        raise P2ContractError(f"{context}: unexpected b15 validity pattern")
    if not valid:
        if view.get("incomplete") is not True or view.get("terminal_reason") != (
            "structural_failure"
        ):
            raise P2ContractError(f"{context}: invalid b15 is not structural failure")
        # Deliberately return before the stored belief key can be accessed.
        return None, "structural_failure"

    belief_raw = view["belief"]
    if not isinstance(belief_raw, list) or len(belief_raw) != len(STATES):
        raise P2ContractError(f"{context}: b15 belief must contain four probabilities")
    try:
        belief = tuple(Decimal(str(value)) for value in belief_raw)
    except ArithmeticError as exc:
        raise P2ContractError(f"{context}: b15 belief is not decimal") from exc
    if any(
        not value.is_finite() or value < 0 or value > 1 for value in belief
    ):
        raise P2ContractError(f"{context}: b15 belief has invalid probability")
    if abs(sum(belief, Decimal(0)) - Decimal(1)) > Decimal("1e-12"):
        raise P2ContractError(f"{context}: b15 belief is not normalized")
    return belief, "valid"


def load_h1_b15_dataset(
    manifest_path: Path = H1_MANIFEST_SOURCE,
) -> DiagnosticDataset:
    manifest_hash = verify_sha256(
        manifest_path,
        EXPECTED_SOURCE_SHA256["h1_h4_raw_manifest"],
        label="H1-H4 raw manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("run_id") != "confirmatory-v1":
        raise P2ContractError("H1-H4 manifest run_id drift")
    shard_rows = manifest.get("shards")
    if not isinstance(shard_rows, list):
        raise P2ContractError("H1-H4 manifest shards are missing")
    expected_shard_ids = {
        f"target={target}|truth={truth}|condition=matched"
        for target in TARGETS
        for truth in STATES
    }
    selected_entries: dict[str, Mapping[str, Any]] = {}
    for entry in shard_rows:
        if not isinstance(entry, Mapping):
            raise P2ContractError("H1-H4 manifest shard entry is invalid")
        shard_id = entry.get("shard_id")
        if shard_id in expected_shard_ids:
            if shard_id in selected_entries:
                raise P2ContractError(f"duplicate H1-H4 shard entry: {shard_id}")
            selected_entries[str(shard_id)] = entry
    if set(selected_entries) != expected_shard_ids:
        raise P2ContractError("H1-H4 matched target shard set is incomplete")

    records: list[DiagnosticRecord] = []
    shard_hashes: dict[str, str] = {}
    seen_keys: set[tuple[str, str, str, int]] = set()
    for shard_id in sorted(selected_entries):
        entry = selected_entries[shard_id]
        filename = entry.get("filename")
        expected_hash = entry.get("sha256")
        if not isinstance(filename, str) or not isinstance(expected_hash, str):
            raise P2ContractError(f"H1-H4 shard binding is invalid: {shard_id}")
        shard_path = manifest_path.parent / filename
        shard_hashes[filename] = verify_sha256(
            shard_path,
            expected_hash,
            label=f"H1-H4 shard {filename}",
        )
        parsed_rows: list[Mapping[str, Any]] = []
        with shard_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parsed = json.loads(line, parse_float=Decimal)
                if not isinstance(parsed, Mapping):
                    raise P2ContractError(
                        f"{filename}:{line_number}: shard row is not an object"
                    )
                parsed_rows.append(parsed)
        if not parsed_rows or parsed_rows[0].get("record_type") != (
            "confirmatory_shard_manifest"
        ):
            raise P2ContractError(f"{filename}: shard manifest row is missing")
        if parsed_rows[0].get("shard_id") != shard_id:
            raise P2ContractError(f"{filename}: internal shard_id drift")
        journey_rows = parsed_rows[1:]
        if parsed_rows[0].get("record_count") != len(journey_rows):
            raise P2ContractError(f"{filename}: shard record_count drift")
        if len(journey_rows) != 150:
            raise P2ContractError(f"{filename}: expected exactly 150 journeys")

        shard_parts = dict(part.split("=", 1) for part in shard_id.split("|"))
        shard_target = shard_parts["target"]
        shard_truth = shard_parts["truth"]
        for row in journey_rows:
            if any(
                (
                    row.get("record_type") != "confirmatory_journey",
                    row.get("simulated") is not True,
                    row.get("target_node") != shard_target,
                    row.get("truth") != shard_truth,
                    row.get("condition") != "matched",
                    row.get("arm") not in {"A", "B", "C"},
                    not isinstance(row.get("replicate"), int),
                    row.get("replicate") not in range(50),
                )
            ):
                raise P2ContractError(f"{filename}: journey identity drift")
            arm = str(row["arm"])
            replicate = int(row["replicate"])
            key = (shard_target, shard_truth, arm, replicate)
            if key in seen_keys:
                raise P2ContractError(f"duplicate H1-H4 journey key: {key}")
            seen_keys.add(key)
            views = row.get("views")
            if not isinstance(views, list):
                raise P2ContractError(f"{key}: journey views are missing")
            b15_views = [view for view in views if view.get("nominal_budget") == 15]
            if len(b15_views) != 1 or not isinstance(b15_views[0], Mapping):
                raise P2ContractError(f"{key}: expected exactly one b15 view")
            expected_valid = not (shard_target == "基本操作" and arm == "C")
            belief, status = posterior_from_b15(
                b15_views[0],
                expected_valid=expected_valid,
                context="/".join(map(str, key)),
            )
            records.append(
                DiagnosticRecord(
                    target=shard_target,
                    truth=shard_truth,
                    arm=arm,
                    replicate=replicate,
                    belief=belief,
                    diagnostic_status=status,
                )
            )

    expected_keys = {
        (target, truth, arm, replicate)
        for target in TARGETS
        for truth in STATES
        for arm in ("A", "B", "C")
        for replicate in range(50)
    }
    if seen_keys != expected_keys:
        raise P2ContractError("H1-H4 b15 grid is incomplete")
    records.sort(
        key=lambda row: (
            TARGETS.index(row.target),
            STATES.index(row.truth),
            row.arm,
            row.replicate,
        )
    )
    return DiagnosticDataset(
        records=tuple(records),
        source_manifest_sha256=manifest_hash,
        shard_sha256=shard_hashes,
    )


def _result_envelope() -> dict[str, Any]:
    return {
        "illustrative": True,
        "simulated": True,
        "external_validity": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_decision_instances_manifest() -> dict[str, Any]:
    truth_cells = [
        {
            "truth_basic": truth_basic,
            "truth_alkane": truth_alkane,
            "truth_cell_weight": "0.0625",
            "component_weight": "0.0004",
            "analytic_integration_terms": 2_500,
        }
        for truth_basic in STATES
        for truth_alkane in STATES
    ]
    return {
        **_result_envelope(),
        "schema_version": "yher.p2.decision_instances.v1",
        "arms": ["oracle", "A", "B", "C"],
        "target_strata": list(TARGETS),
        "replicate_clusters_per_target_stratum": 50,
        "truth_cells": truth_cells,
        "analytic_integration_terms_per_arm": 40_000,
        "reporting_unit": (
            "two_fixed_target_strata_each_with_50_programmatic_replicate_clusters"
        ),
        "terms_are_not_independent_observations": True,
        "fixed_design_not_real_state_prevalence": True,
    }


def _one_hot(state: str) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    if state not in STATES:
        raise P2ContractError(f"invalid truth state: {state}")
    return tuple(
        Decimal(1) if candidate_state == state else Decimal(0)
        for candidate_state in STATES
    )  # type: ignore[return-value]


def _available_slots(candidates: tuple[Candidate, ...]) -> set[tuple[str, str]]:
    return {(candidate.target, candidate.slot_state) for candidate in candidates}


def _belief_json(
    belief: tuple[Decimal, Decimal, Decimal, Decimal] | None,
) -> list[str] | None:
    if belief is None:
        return None
    return [_decimal_text(value) for value in belief]


def _minutes(seconds: int) -> float:
    return round(seconds / 60.0, 10)


def compute_profile(
    *,
    arm: str,
    truths: Mapping[str, str],
    beliefs_by_target: Mapping[
        str, tuple[Decimal, Decimal, Decimal, Decimal]
    ],
    candidates: tuple[Candidate, ...],
    replicate_basic: int,
    replicate_alkane: int,
    failed_nodes: frozenset[str] = frozenset(),
    oracle_selection: SelectionResult | None = None,
) -> ProfileComputation:
    if arm not in {"oracle", "A", "B", "C"}:
        raise P2ContractError(f"invalid profile arm: {arm}")
    if set(truths) != set(TARGETS) or any(truths[target] not in STATES for target in TARGETS):
        raise P2ContractError("profile truths must bind both frozen targets")
    if replicate_basic not in range(50) or replicate_alkane not in range(50):
        raise P2ContractError("profile replicate must be in 0..49")
    if arm == "C":
        if failed_nodes != frozenset({"基本操作"}):
            raise P2ContractError("primary Arm C must mask exactly the Basic node")
    elif failed_nodes:
        raise P2ContractError(f"{arm} cannot contain a structural failure node")

    oracle_beliefs = {target: _one_hot(truths[target]) for target in TARGETS}
    if oracle_selection is None:
        oracle_selection = select_candidates(
            candidates,
            oracle_beliefs,
            budget_seconds=BUDGET_SECONDS,
        )
    if arm == "oracle":
        if failed_nodes:
            raise P2ContractError("oracle cannot have failed nodes")
        active_beliefs = oracle_beliefs
        selection = oracle_selection
    else:
        expected_belief_nodes = set(TARGETS) - set(failed_nodes)
        if set(beliefs_by_target) != expected_belief_nodes:
            raise P2ContractError("profile belief nodes do not match diagnostic status")
        for target, belief in beliefs_by_target.items():
            if len(belief) != 4 or any(
                not value.is_finite() or value < 0 or value > 1 for value in belief
            ):
                raise P2ContractError(f"profile belief is invalid: {target}")
            if abs(sum(belief, Decimal(0)) - Decimal(1)) > Decimal("1e-12"):
                raise P2ContractError(f"profile belief is not normalized: {target}")
        active_beliefs = dict(beliefs_by_target)
        selection = select_candidates(
            candidates,
            active_beliefs,
            failed_nodes=failed_nodes,
            budget_seconds=BUDGET_SECONDS,
        )

    available_slots = _available_slots(candidates)
    selected_slots = {
        (candidate.target, candidate.slot_state) for candidate in selection.selected
    }
    mismatched_seconds = sum(
        candidate.charged_seconds
        for candidate in selection.selected
        if candidate.slot_state != truths[candidate.target]
    )
    missed_by_cause = {
        "diagnostic_structural_failure": 0,
        "posterior_selection": 0,
        "budget_constraint": 0,
    }
    for oracle_candidate in oracle_selection.selected:
        slot = (oracle_candidate.target, oracle_candidate.slot_state)
        if slot in selected_slots:
            continue
        if oracle_candidate.target in failed_nodes:
            cause = "diagnostic_structural_failure"
        else:
            saw_budget_rejection = any(
                evaluation["target"] == oracle_candidate.target
                and evaluation["slot_state"] == oracle_candidate.slot_state
                and evaluation["rejection_reason"] == "budget"
                for step in selection.trace
                for evaluation in step["evaluations"]
            )
            cause = "budget_constraint" if saw_budget_rejection else "posterior_selection"
        missed_by_cause[cause] += oracle_candidate.charged_seconds
    missed_available_seconds = sum(missed_by_cause.values())

    unsupported_by_node: dict[str, str | None] = {}
    unsupported_total = Decimal(0)
    for target in TARGETS:
        belief = active_beliefs.get(target)
        if belief is None:
            unsupported_by_node[target] = None
            continue
        unsupported = sum(
            (
                belief[index]
                for index, state in enumerate(STATES)
                if (target, state) not in available_slots
            ),
            Decimal(0),
        )
        unsupported_by_node[target] = _decimal_text(unsupported)
        unsupported_total += unsupported
    unobtainable_truth_slots = sum(
        (target, truths[target]) not in available_slots for target in TARGETS
    )
    selected_seconds = selection.total_seconds
    unused_budget_seconds = BUDGET_SECONDS - selected_seconds
    if unused_budget_seconds < 0:
        raise P2ContractError("profile selection exceeds frozen budget")

    trace_payload = {
        **_result_envelope(),
        "arm": arm,
        "beliefs_by_node": {
            target: _belief_json(active_beliefs.get(target)) for target in TARGETS
        },
        "failed_nodes": sorted(failed_nodes),
        "trace": list(selection.trace),
    }
    trace_hash = hashlib.sha256(canonical_json_bytes(trace_payload)).hexdigest()
    profile_key = {
        "arm": arm,
        "truth_basic": truths["基本操作"],
        "truth_alkane": truths["烷烃"],
        "replicate_basic": replicate_basic,
        "replicate_alkane": replicate_alkane,
    }
    profile_id = hashlib.sha256(canonical_json_bytes(profile_key)).hexdigest()[:24]
    diagnostic_status = {
        target: ("structural_failure" if target in failed_nodes else "valid")
        for target in TARGETS
    }
    belief_reason = {
        target: ("structural_failure" if target in failed_nodes else None)
        for target in TARGETS
    }
    metrics: dict[str, Any] = {
        **_result_envelope(),
        "schema_version": "yher.p2.profile_metrics.v1",
        "spec_hash": P2_SPEC_SHA256,
        "source_manifest_hash": EXPECTED_SOURCE_SHA256["h1_h4_raw_manifest"],
        "candidate_subset_hash": EXPECTED_SUBSET_CANONICAL_ROWS_SHA256,
        "audit_declared_unreproduced": EXPECTED_SUBSET_AUDIT_SHA256,
        "profile_id": profile_id,
        **profile_key,
        "belief_basic": _belief_json(active_beliefs.get("基本操作")),
        "belief_alkane": _belief_json(active_beliefs.get("烷烃")),
        "belief_reason_by_node": belief_reason,
        "diagnostic_status_by_node": diagnostic_status,
        "prescription_status": (
            "partial_structural_failure" if failed_nodes else "complete"
        ),
        "selected_chunk_ids": [candidate.chunk_id for candidate in selection.selected],
        "selected_physical_keys": [
            candidate.physical_key for candidate in selection.selected
        ],
        "selected_seconds": selected_seconds,
        "selected_minutes": _minutes(selected_seconds),
        "selected_segment_count": len(selection.selected),
        "mismatched_selected_seconds": mismatched_seconds,
        "mismatched_selected_minutes": _minutes(mismatched_seconds),
        "missed_available_seconds": missed_available_seconds,
        "missed_available_supply_minutes": _minutes(missed_available_seconds),
        "missed_available_minutes": _minutes(missed_available_seconds),
        "missed_available_by_cause_seconds": missed_by_cause,
        "unobtainable_supply_minutes": None,
        "unobtainable_reason": "no_frozen_role_compatible_dose",
        "unused_budget_seconds": unused_budget_seconds,
        "unused_budget_minutes": _minutes(unused_budget_seconds),
        "unobtainable_truth_slots": unobtainable_truth_slots,
        "unsupported_posterior_mass": _decimal_text(unsupported_total),
        "unsupported_posterior_mass_by_node": unsupported_by_node,
        "structural_failure_node_fraction": len(failed_nodes) / len(TARGETS),
        "metric_denominators": {
            "fixed_target_strata": 2,
            "programmatic_replicate_clusters_per_stratum": 50,
            "truth_cell_weight": "0.0625",
            "component_weight_within_truth_cell": "0.0004",
            "valid_posterior_node_count": len(TARGETS) - len(failed_nodes),
        },
        "selector_trace_hash": trace_hash,
    }
    trace_record = {**trace_payload, "selector_trace_hash": trace_hash}
    return ProfileComputation(
        metrics=metrics,
        trace_record=trace_record,
        selection=selection,
        oracle_selection=oracle_selection,
    )


def build_input_manifest(
    subset: CandidateSubset,
    dataset: DiagnosticDataset,
) -> dict[str, Any]:
    spec_hash = verify_sha256(P2_SPEC_PATH, P2_SPEC_SHA256, label="P2 analysis plan")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", P2_SPEC_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise InputDriftError("P2 specification commit is not an ancestor of HEAD")
    committed = subprocess.run(
        ["git", "show", f"{P2_SPEC_COMMIT}:experiments/p2_illustrative_analysis_plan.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    committed_hash = hashlib.sha256(committed).hexdigest()
    if committed_hash != P2_SPEC_SHA256:
        raise InputDriftError("committed P2 specification bytes drift")
    if subset.canonical_rows_sha256 != EXPECTED_SUBSET_CANONICAL_ROWS_SHA256:
        raise InputDriftError("candidate subset canonical digest drift")
    if subset.source_hashes != {
        "trusted_candidate_jsonl": EXPECTED_SOURCE_SHA256["trusted_candidate_jsonl"],
        "signed_runtime_metadata": EXPECTED_SOURCE_SHA256["signed_runtime_metadata"],
    }:
        raise InputDriftError("candidate source binding drift")
    if dataset.source_manifest_sha256 != EXPECTED_SOURCE_SHA256["h1_h4_raw_manifest"]:
        raise InputDriftError("H1-H4 source manifest binding drift")
    if len(dataset.records) != 1_200 or len(dataset.shard_sha256) != 8:
        raise InputDriftError("H1-H4 product-margin fixture drift")
    return {
        **_result_envelope(),
        "schema_version": "yher.p2.input_manifest.v1",
        "hash_gate_status": "pass",
        "spec": {
            "commit": P2_SPEC_COMMIT,
            "sha256": spec_hash,
            "committed_bytes_sha256": committed_hash,
            "precedes_outcome_computation": True,
        },
        "source_files": {
            "trusted_candidate_jsonl": {
                "path": str(CANDIDATE_SOURCE),
                "sha256": subset.source_hashes["trusted_candidate_jsonl"],
            },
            "signed_runtime_metadata": {
                "path": str(RUNTIME_SOURCE),
                "sha256": subset.source_hashes["signed_runtime_metadata"],
            },
            "h1_h4_raw_manifest": {
                "path": str(H1_MANIFEST_SOURCE),
                "sha256": dataset.source_manifest_sha256,
            },
        },
        "candidate_subset": {
            "row_count": len(subset.candidates),
            "physical_source_count": len(
                {candidate.physical_key for candidate in subset.candidates}
            ),
            "canonical_sha256": subset.canonical_rows_sha256,
            "canonical_serialization": (
                "json.dumps(sorted(rows,key=chunk_id),ensure_ascii=False,"
                "sort_keys=True,separators=(',',':')).encode('utf-8')"
            ),
            "audit_declared_unreproduced": subset.audit_declared_sha256,
            "audit_declared_digest_is_gate": False,
            "exact_chunk_ids": [
                candidate.chunk_id for candidate in subset.candidates
            ],
        },
        "h1_h4_product_margins": {
            "condition": "matched",
            "checkpoint_nominal_budget": 15,
            "posterior_order": list(STATES),
            "journey_count": len(dataset.records),
            "matched_shard_count": len(dataset.shard_sha256),
            "matched_shard_sha256": dict(sorted(dataset.shard_sha256.items())),
            "invalid_belief_read_policy": "never_read_fail_closed",
        },
        "selector": {
            "budget_seconds": BUDGET_SECONDS,
            "decimal_precision_digits": 60,
            "binary_role_slot_saturation": True,
            "physical_source_no_repeat": True,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng": "numpy.PCG64",
            "fixed_target_strata": list(TARGETS),
            "cluster_count_per_stratum": 50,
            "arms_and_truths_paired_within_target_replicate": True,
        },
    }


def bootstrap_replicate_indices() -> tuple[Any, Any]:
    import numpy as np

    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    basic = generator.integers(
        0,
        50,
        size=(BOOTSTRAP_RESAMPLES, 50),
        dtype=np.int16,
    )
    alkane = generator.integers(
        0,
        50,
        size=(BOOTSTRAP_RESAMPLES, 50),
        dtype=np.int16,
    )
    return basic, alkane


def _interval(values: Any, point: float) -> dict[str, float]:
    import numpy as np

    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "point": float(point),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
    }


def bootstrap_metric_cube(metric_cube: Any, metric_names: tuple[str, ...]) -> dict[str, Any]:
    import numpy as np

    cube = np.asarray(metric_cube, dtype=np.float64)
    expected_shape = (4, 4, 4, 50, 50, len(metric_names))
    if cube.shape != expected_shape:
        raise P2ContractError(
            f"bootstrap metric cube shape drift: expected {expected_shape}, got {cube.shape}"
        )
    if not np.isfinite(cube).all():
        raise P2ContractError("bootstrap metric cube contains a non-finite value")
    basic_indices, alkane_indices = bootstrap_replicate_indices()
    cell_draws = np.empty(
        (BOOTSTRAP_RESAMPLES, 4, 4, 4, len(metric_names)),
        dtype=np.float64,
    )

    for arm_index in range(4):
        for basic_truth_index in range(4):
            for alkane_truth_index in range(4):
                matrix = cube[
                    arm_index,
                    basic_truth_index,
                    alkane_truth_index,
                ]
                nonadditive: list[int] = []
                for metric_index in range(len(metric_names)):
                    surface = matrix[:, :, metric_index]
                    if np.all(surface == surface[0, 0]):
                        cell_draws[
                            :, arm_index, basic_truth_index, alkane_truth_index, metric_index
                        ] = surface[0, 0]
                        continue
                    additive_surface = (
                        surface[:, [0]] + surface[[0], :] - surface[0, 0]
                    )
                    if np.allclose(surface, additive_surface, rtol=0.0, atol=1e-11):
                        basic_component = surface[:, 0]
                        alkane_component = surface[0, :] - surface[0, 0]
                        cell_draws[
                            :, arm_index, basic_truth_index, alkane_truth_index, metric_index
                        ] = (
                            basic_component[basic_indices].mean(axis=1)
                            + alkane_component[alkane_indices].mean(axis=1)
                        )
                    else:
                        nonadditive.append(metric_index)

                if not nonadditive:
                    continue
                flattened = matrix[:, :, nonadditive].reshape(2_500, len(nonadditive))
                categories, inverse = np.unique(
                    flattened,
                    axis=0,
                    return_inverse=True,
                )
                category_grid = inverse.reshape(50, 50)
                category_count = len(categories)
                batch_size = 500
                for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
                    stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
                    batch_categories = category_grid[
                        basic_indices[start:stop, :, None],
                        alkane_indices[start:stop, None, :],
                    ]
                    offsets = (
                        np.arange(stop - start, dtype=np.int64)[:, None, None]
                        * category_count
                    )
                    counts = np.bincount(
                        (batch_categories + offsets).ravel(),
                        minlength=(stop - start) * category_count,
                    ).reshape(stop - start, category_count)
                    means = counts @ categories / 2_500.0
                    for local_index, metric_index in enumerate(nonadditive):
                        cell_draws[
                            start:stop,
                            arm_index,
                            basic_truth_index,
                            alkane_truth_index,
                            metric_index,
                        ] = means[:, local_index]

    point_cells = cube.mean(axis=(3, 4))
    point_overall = point_cells.mean(axis=(1, 2))
    overall_draws = cell_draws.mean(axis=(2, 3))
    overall_rows: list[dict[str, Any]] = []
    truth_cell_rows: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(ARM_NAMES):
        for metric_index, metric in enumerate(metric_names):
            overall_rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    **_interval(
                        overall_draws[:, arm_index, metric_index],
                        point_overall[arm_index, metric_index],
                    ),
                    "defined_resamples": BOOTSTRAP_RESAMPLES,
                }
            )
        for basic_truth_index, truth_basic in enumerate(STATES):
            for alkane_truth_index, truth_alkane in enumerate(STATES):
                for metric_index, metric in enumerate(metric_names):
                    truth_cell_rows.append(
                        {
                            "arm": arm,
                            "truth_basic": truth_basic,
                            "truth_alkane": truth_alkane,
                            "metric": metric,
                            **_interval(
                                cell_draws[
                                    :,
                                    arm_index,
                                    basic_truth_index,
                                    alkane_truth_index,
                                    metric_index,
                                ],
                                point_cells[
                                    arm_index,
                                    basic_truth_index,
                                    alkane_truth_index,
                                    metric_index,
                                ],
                            ),
                            "defined_resamples": BOOTSTRAP_RESAMPLES,
                        }
                    )

    contrast_definitions = (
        ("A_minus_oracle", 1, 0),
        ("B_minus_oracle", 2, 0),
        ("A_minus_B", 1, 2),
        ("C_minus_oracle_ITT", 3, 0),
    )
    contrast_rows: list[dict[str, Any]] = []
    for contrast, left, right in contrast_definitions:
        for metric_index, metric in enumerate(metric_names):
            values = overall_draws[:, left, metric_index] - overall_draws[
                :, right, metric_index
            ]
            point = point_overall[left, metric_index] - point_overall[
                right, metric_index
            ]
            contrast_rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    **_interval(values, point),
                    "defined_resamples": BOOTSTRAP_RESAMPLES,
                    "arm_c_structural_failure_annotation": (
                        "failed_node_fraction=0.5"
                        if contrast.startswith("C_minus")
                        else None
                    ),
                }
            )
    return {
        **_result_envelope(),
        "schema_version": "yher.p2.bootstrap.v1",
        "seed": BOOTSTRAP_SEED,
        "rng": "numpy.PCG64",
        "attempted_resamples": BOOTSTRAP_RESAMPLES,
        "defined_resamples": BOOTSTRAP_RESAMPLES,
        "interval": "two_sided_percentile_95",
        "uncertainty_scope": (
            "simulator_Monte_Carlo_variability_under_fixed_design_not_human_population"
        ),
        "cluster_unit": "target_node_x_programmatic_replicate",
        "fixed_target_strata": list(TARGETS),
        "clusters_per_stratum": 50,
        "overall": overall_rows,
        "truth_cells": truth_cell_rows,
        "contrasts": contrast_rows,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_line(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _profile_metric_vector(metrics: Mapping[str, Any]) -> list[float]:
    causes = metrics["missed_available_by_cause_seconds"]
    return [
        float(metrics["selected_seconds"]),
        float(metrics["selected_segment_count"]),
        float(metrics["mismatched_selected_seconds"]),
        float(metrics["missed_available_seconds"]),
        float(metrics["unused_budget_seconds"]),
        float(metrics["unobtainable_truth_slots"]),
        float(metrics["unsupported_posterior_mass"]),
        float(metrics["structural_failure_node_fraction"]),
        float(causes["diagnostic_structural_failure"]),
        float(causes["posterior_selection"]),
        float(causes["budget_constraint"]),
    ]


def _summary_from_cube(metric_cube: Any, bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    point_cells = np.asarray(metric_cube).mean(axis=(3, 4))
    point_overall = point_cells.mean(axis=(1, 2))
    overall_rows = []
    for arm_index, arm in enumerate(ARM_NAMES):
        values = {
            metric: float(point_overall[arm_index, metric_index])
            for metric_index, metric in enumerate(BOOTSTRAP_METRICS)
        }
        overall_rows.append(
            {
                "arm": arm,
                **values,
                "selected_minutes": values["selected_seconds"] / 60.0,
                "mismatched_selected_minutes": (
                    values["mismatched_selected_seconds"] / 60.0
                ),
                "missed_available_supply_minutes": (
                    values["missed_available_seconds"] / 60.0
                ),
                "unobtainable_supply_minutes": None,
                "unobtainable_reason": "no_frozen_role_compatible_dose",
                "unused_budget_minutes": values["unused_budget_seconds"] / 60.0,
                "analytic_integration_terms": 40_000,
                "analytic_terms_are_not_sample_size": True,
            }
        )
    truth_cells = []
    for arm_index, arm in enumerate(ARM_NAMES):
        for basic_index, truth_basic in enumerate(STATES):
            for alkane_index, truth_alkane in enumerate(STATES):
                truth_cells.append(
                    {
                        "arm": arm,
                        "truth_basic": truth_basic,
                        "truth_alkane": truth_alkane,
                        **{
                            metric: float(
                                point_cells[
                                    arm_index,
                                    basic_index,
                                    alkane_index,
                                    metric_index,
                                ]
                            )
                            for metric_index, metric in enumerate(BOOTSTRAP_METRICS)
                        },
                    }
                )
    return {
        **_result_envelope(),
        "schema_version": "yher.p2.summary.v1",
        "spec_hash": P2_SPEC_SHA256,
        "budget_seconds": BUDGET_SECONDS,
        "candidate_row_count": 8,
        "physical_source_count": 3,
        "exact_overlap_targets": list(TARGETS),
        "overall": overall_rows,
        "truth_cells": truth_cells,
        "bootstrap_overall": bootstrap["overall"],
        "bootstrap_contrasts": bootstrap["contrasts"],
        "reporting_unit": (
            "two_fixed_target_strata_each_with_50_programmatic_replicate_clusters"
        ),
        "unavailable_minute_field_policy": {
            "value": None,
            "reason": "no_frozen_role_compatible_dose",
        },
        "scalar_composite_computed": False,
    }


def _figure_data(
    summary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    subset: CandidateSubset,
) -> dict[str, Any]:
    interval_index = {
        (row["arm"], row["metric"]): row for row in bootstrap["overall"]
    }
    overall_minutes = []
    seconds_metrics = (
        ("selected_seconds", "selected_minutes"),
        ("mismatched_selected_seconds", "mismatched_selected_minutes"),
        ("missed_available_seconds", "missed_available_supply_minutes"),
        ("unused_budget_seconds", "unused_budget_minutes"),
    )
    for row in summary["overall"]:
        for seconds_metric, minute_metric in seconds_metrics:
            interval = interval_index[(row["arm"], seconds_metric)]
            overall_minutes.append(
                {
                    "arm": row["arm"],
                    "metric": minute_metric,
                    "point": interval["point"] / 60.0,
                    "ci95_low": interval["ci95_low"] / 60.0,
                    "ci95_high": interval["ci95_high"] / 60.0,
                }
            )
    truth_cell_minutes = []
    for row in summary["truth_cells"]:
        for metric in ("mismatched_selected_seconds", "missed_available_seconds"):
            truth_cell_minutes.append(
                {
                    "arm": row["arm"],
                    "truth_basic": row["truth_basic"],
                    "truth_alkane": row["truth_alkane"],
                    "metric": metric.replace("_seconds", "_minutes"),
                    "value": row[metric] / 60.0,
                }
            )
    return {
        **_result_envelope(),
        "schema_version": "yher.p2.figure_data.v1",
        "figure_title": "Supply-bound prescription illustration under a 600-second budget",
        "caption_boundary": (
            "Intervals reflect programmatic simulator Monte Carlo variability; product-form "
            "terms are analytic integration points, not independent observations."
        ),
        "overall_minute_panel": overall_minutes,
        "truth_cell_minute_panels": truth_cell_minutes,
        "structural_failure_panel": [
            {
                "arm": row["arm"],
                "failed_node_fraction": row["structural_failure_node_fraction"],
            }
            for row in summary["overall"]
        ],
        "supply_panel": [
            {
                "chunk_id": candidate.chunk_id,
                "target": candidate.target,
                "role": candidate.role,
                "charged_seconds": candidate.charged_seconds,
                "physical_key": candidate.physical_key,
            }
            for candidate in subset.candidates
        ],
        "unobtainable_supply_minutes": None,
        "unobtainable_reason": "no_frozen_role_compatible_dose",
    }


def _publication_figure_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    overall = summary.get("overall")
    if not isinstance(overall, list):
        raise P2ContractError("publication figure summary overall rows are missing")
    by_arm = {
        row.get("arm"): row
        for row in overall
        if isinstance(row, Mapping) and row.get("arm") in {"A", "B", "C"}
    }
    if set(by_arm) != {"A", "B", "C"}:
        raise P2ContractError("publication figure requires separate A/B/C rows")
    rows: list[dict[str, Any]] = []
    for arm in ("A", "B", "C"):
        source = by_arm[arm]
        try:
            mismatch = float(source["mismatched_selected_minutes"])
            missed = float(source["missed_available_supply_minutes"])
            failed_fraction = float(source["structural_failure_node_fraction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise P2ContractError(f"publication figure row is invalid: Arm {arm}") from exc
        if any(not math.isfinite(value) or value < 0 for value in (mismatch, missed)):
            raise P2ContractError(f"publication figure minutes are invalid: Arm {arm}")
        expected_failure = 0.5 if arm == "C" else 0.0
        if failed_fraction != expected_failure:
            raise P2ContractError(
                f"publication figure structural-failure fraction drift: Arm {arm}"
            )
        rows.append(
            {
                "arm": arm,
                "mismatch": mismatch,
                "missed": missed,
                "failed_fraction": failed_fraction,
            }
        )
    return rows


def _figure_font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for name in names:
        path = Path(name)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_publication_figure(
    summary: Mapping[str, Any],
    output_dir: Path,
    *,
    figure_data: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    if figure_data is not None and any(
        (
            figure_data.get("illustrative") is not True,
            figure_data.get("simulated") is not True,
            figure_data.get("external_validity") is not False,
        )
    ):
        raise P2ContractError("publication figure data claim boundary drift")
    rows = _publication_figure_rows(summary)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "p2_supply_bound_illustration.png"
    svg_path = output_dir / "p2_supply_bound_illustration.svg"

    width, height = 1600, 1000
    plot_left, plot_right = 330, 1450
    plot_top, group_gap = 330, 190
    bar_height, bar_gap = 38, 18
    maximum = max(max(row["mismatch"], row["missed"]) for row in rows)
    axis_max = max(1.0, math.ceil(maximum * 1.1 * 2) / 2)
    mismatch_color = "#246B72"
    missed_color = "#C65A3A"
    text_color = "#1F2933"
    muted_color = "#5E6C76"
    grid_color = "#D8DEE3"
    background = "#FAFBFC"

    def x_position(value: float) -> float:
        return plot_left + value / axis_max * (plot_right - plot_left)

    ticks = [float(value) for value in range(math.floor(axis_max) + 1)]
    if not math.isclose(ticks[-1], axis_max):
        ticks.append(axis_max)

    svg: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" '
        'viewBox="0 0 1600 1000" role="img" '
        'aria-labelledby="figure-title figure-desc">',
        '<title id="figure-title">Supply-bound illustrative prescription outputs</title>',
        '<desc id="figure-desc">A/B/C mechanically mismatched selected minutes and '
        "missed available-supply minutes under the frozen 600-second model; "
        "Arm C has structural-failure node fraction 0.5.</desc>",
        f'<rect width="{width}" height="{height}" fill="{background}"/>',
        f'<text x="90" y="92" font-family="Arial, sans-serif" font-size="46" '
        f'font-weight="700" fill="{text_color}">Supply-bound illustrative prescription outputs</text>',
        f'<text x="90" y="140" font-family="Arial, sans-serif" font-size="24" '
        f'fill="{muted_color}">Mechanical duration fields under the frozen available-supply model</text>',
        f'<rect x="90" y="188" width="28" height="20" fill="{mismatch_color}"/>',
        f'<text x="132" y="207" font-family="Arial, sans-serif" font-size="22" '
        f'fill="{text_color}">Mechanically mismatched selected minutes</text>',
        f'<rect x="650" y="188" width="28" height="20" fill="{missed_color}"/>',
        f'<text x="692" y="207" font-family="Arial, sans-serif" font-size="22" '
        f'fill="{text_color}">Missed available-supply minutes</text>',
    ]
    for tick in ticks:
        x = x_position(tick)
        svg.extend(
            [
                f'<line x1="{x:.2f}" y1="285" x2="{x:.2f}" y2="820" '
                f'stroke="{grid_color}" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="855" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="20" fill="{muted_color}">'
                f"{tick:g}</text>",
            ]
        )
    svg.append(
        f'<text x="{(plot_left + plot_right) / 2:.2f}" y="895" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="22" fill="{text_color}">Charged minutes</text>'
    )
    for index, row in enumerate(rows):
        base_y = plot_top + index * group_gap
        mismatch_width = x_position(row["mismatch"]) - plot_left
        missed_y = base_y + bar_height + bar_gap
        missed_width = x_position(row["missed"]) - plot_left
        arm = html.escape(row["arm"])
        svg.extend(
            [
                f'<text x="90" y="{base_y + 28}" font-family="Arial, sans-serif" '
                f'font-size="30" font-weight="700" fill="{text_color}">Arm {arm}</text>',
                f'<rect x="{plot_left}" y="{base_y}" width="{mismatch_width:.2f}" '
                f'height="{bar_height}" fill="{mismatch_color}"/>',
                f'<text x="{x_position(row["mismatch"]) + 12:.2f}" y="{base_y + 28}" '
                f'font-family="Arial, sans-serif" font-size="22" fill="{text_color}">'
                f'{row["mismatch"]:.4f}</text>',
                f'<rect x="{plot_left}" y="{missed_y}" width="{missed_width:.2f}" '
                f'height="{bar_height}" fill="{missed_color}"/>',
                f'<text x="{x_position(row["missed"]) + 12:.2f}" y="{missed_y + 28}" '
                f'font-family="Arial, sans-serif" font-size="22" fill="{text_color}">'
                f'{row["missed"]:.4f}</text>',
            ]
        )
    svg.extend(
        [
            f'<line x1="90" y1="915" x2="1510" y2="915" stroke="{grid_color}"/>',
            f'<text x="90" y="955" font-family="Arial, sans-serif" font-size="21" '
            f'font-weight="700" fill="{missed_color}">Arm C structural-failure node fraction: 0.5</text>',
            f'<text x="770" y="955" font-family="Arial, sans-serif" font-size="19" '
            f'fill="{muted_color}">Simulated fixed-design illustration; no external validity.</text>',
            "</svg>",
        ]
    )
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")

    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    title_font = _figure_font(46, bold=True)
    subtitle_font = _figure_font(24)
    legend_font = _figure_font(22)
    arm_font = _figure_font(30, bold=True)
    value_font = _figure_font(22)
    note_font = _figure_font(21, bold=True)
    small_font = _figure_font(19)
    draw.text(
        (90, 48),
        "Supply-bound illustrative prescription outputs",
        font=title_font,
        fill=text_color,
    )
    draw.text(
        (90, 112),
        "Mechanical duration fields under the frozen available-supply model",
        font=subtitle_font,
        fill=muted_color,
    )
    draw.rectangle((90, 188, 118, 208), fill=mismatch_color)
    draw.text(
        (132, 183),
        "Mechanically mismatched selected minutes",
        font=legend_font,
        fill=text_color,
    )
    draw.rectangle((650, 188, 678, 208), fill=missed_color)
    draw.text(
        (692, 183),
        "Missed available-supply minutes",
        font=legend_font,
        fill=text_color,
    )
    for tick in ticks:
        x = x_position(tick)
        draw.line((x, 285, x, 820), fill=grid_color, width=1)
        label = f"{tick:g}"
        box = draw.textbbox((0, 0), label, font=value_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, 828),
            label,
            font=value_font,
            fill=muted_color,
        )
    axis_label = "Charged minutes"
    box = draw.textbbox((0, 0), axis_label, font=legend_font)
    draw.text(
        (((plot_left + plot_right) - (box[2] - box[0])) / 2, 875),
        axis_label,
        font=legend_font,
        fill=text_color,
    )
    for index, row in enumerate(rows):
        base_y = plot_top + index * group_gap
        missed_y = base_y + bar_height + bar_gap
        draw.text((90, base_y), f"Arm {row['arm']}", font=arm_font, fill=text_color)
        draw.rectangle(
            (plot_left, base_y, x_position(row["mismatch"]), base_y + bar_height),
            fill=mismatch_color,
        )
        draw.text(
            (x_position(row["mismatch"]) + 12, base_y + 5),
            f'{row["mismatch"]:.4f}',
            font=value_font,
            fill=text_color,
        )
        draw.rectangle(
            (plot_left, missed_y, x_position(row["missed"]), missed_y + bar_height),
            fill=missed_color,
        )
        draw.text(
            (x_position(row["missed"]) + 12, missed_y + 5),
            f'{row["missed"]:.4f}',
            font=value_font,
            fill=text_color,
        )
    draw.line((90, 915, 1510, 915), fill=grid_color, width=1)
    draw.text(
        (90, 932),
        "Arm C structural-failure node fraction: 0.5",
        font=note_font,
        fill=missed_color,
    )
    draw.text(
        (770, 934),
        "Simulated fixed-design illustration; no external validity.",
        font=small_font,
        fill=muted_color,
    )
    image.save(png_path, format="PNG", optimize=True, dpi=(200, 200))
    return png_path, svg_path


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# P2 Supply-Bound Illustration Report",
        "",
        "Status: COMPUTED; ILLUSTRATIVE; SIMULATED; NO EXTERNAL VALIDITY",
        "",
        "The analysis uses eight exact trusted chunks collapsing to three physical sources, "
        "two fixed target strata, and one 600-second analytic budget. Product-form terms are "
        "integration points, not independent observations.",
        "",
        "| Arm | Selected min | Mechanically mismatched min | Missed available-supply min | Unused min | Failed node fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["overall"]:
        lines.append(
            f"| {row['arm']} | {row['selected_minutes']:.4f} | "
            f"{row['mismatched_selected_minutes']:.4f} | "
            f"{row['missed_available_supply_minutes']:.4f} | "
            f"{row['unused_budget_minutes']:.4f} | "
            f"{row['structural_failure_node_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`unobtainable_supply_minutes` is null for every profile because no frozen "
            "role-compatible P/U dose exists.",
            "",
            "These durations are mechanical outputs under the frozen supply model. They are "
            "not learning benefit, efficacy, wasted time, saved time, or population effects.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def refresh_output_manifest(output_dir: Path) -> Mapping[str, Any]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise P2ContractError("cannot refresh P2 output manifest without summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile_count = summary.get("profile_row_count")
    trace_count = summary.get("unique_selector_trace_count")
    if not isinstance(profile_count, int) or not isinstance(trace_count, int):
        raise P2ContractError("P2 summary output counts are invalid")
    artifacts = []
    for path in sorted(output_dir.iterdir(), key=lambda value: value.name):
        if path.name == "output_manifest.json" or not path.is_file():
            continue
        artifacts.append(
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    output_manifest = {
        **_result_envelope(),
        "schema_version": "yher.p2.output_manifest.v1",
        "profile_row_count": profile_count,
        "unique_selector_trace_count": trace_count,
        "artifacts": artifacts,
        "manifest_self_hash_policy": "output_manifest_excluded_from_its_own_artifact_list",
    }
    _write_json(output_dir / "output_manifest.json", output_manifest)
    return output_manifest


def run_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Mapping[str, Any]:
    import numpy as np

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise P2ContractError(f"refusing to overwrite existing P2 output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent)
    )

    subset = load_candidate_subset()
    dataset = load_h1_b15_dataset()
    input_manifest = build_input_manifest(subset, dataset)
    _write_json(staging / "input_manifest.json", input_manifest)
    with (staging / "candidate_subset.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in subset.candidates:
            handle.write(
                _json_line(
                    {
                        **_result_envelope(),
                        "schema_version": "yher.p2.candidate.v1",
                        "candidate_subset_hash": subset.canonical_rows_sha256,
                        "chunk_id": candidate.chunk_id,
                        "target": candidate.target,
                        "physical_key": candidate.physical_key,
                        "charged_seconds": candidate.charged_seconds,
                        "role": candidate.role,
                        "difficulty": candidate.difficulty,
                        "source_line": candidate.source_line,
                        "source_row": candidate.raw_row,
                    }
                )
            )
    _write_json(
        staging / "decision_instances_manifest.json",
        build_decision_instances_manifest(),
    )

    record_index = {
        (record.target, record.truth, record.arm, record.replicate): record
        for record in dataset.records
    }
    metric_cube = np.empty((4, 4, 4, 50, 50, len(BOOTSTRAP_METRICS)))
    trace_hashes: set[str] = set()
    trace_count = 0
    profile_count = 0
    oracle_cache: dict[tuple[str, str], SelectionResult] = {}
    with (staging / "selector_trace.jsonl").open(
        "w", encoding="utf-8", buffering=1024 * 1024
    ) as trace_handle, (staging / "profile_metrics.jsonl").open(
        "w", encoding="utf-8", buffering=1024 * 1024
    ) as profile_handle:
        for arm_index, arm in enumerate(ARM_NAMES):
            for basic_truth_index, truth_basic in enumerate(STATES):
                for alkane_truth_index, truth_alkane in enumerate(STATES):
                    truths = {"基本操作": truth_basic, "烷烃": truth_alkane}
                    oracle_key = (truth_basic, truth_alkane)
                    if oracle_key not in oracle_cache:
                        oracle_cache[oracle_key] = select_candidates(
                            subset.candidates,
                            {target: _one_hot(truths[target]) for target in TARGETS},
                            budget_seconds=BUDGET_SECONDS,
                        )
                    for replicate_basic in range(50):
                        for replicate_alkane in range(50):
                            failed_nodes = frozenset()
                            beliefs: dict[
                                str, tuple[Decimal, Decimal, Decimal, Decimal]
                            ] = {}
                            if arm != "oracle":
                                basic_record = record_index[
                                    ("基本操作", truth_basic, arm, replicate_basic)
                                ]
                                alkane_record = record_index[
                                    ("烷烃", truth_alkane, arm, replicate_alkane)
                                ]
                                if arm == "C":
                                    if basic_record.belief is not None or (
                                        basic_record.diagnostic_status
                                        != "structural_failure"
                                    ):
                                        raise P2ContractError(
                                            "Arm C Basic invalid-belief contract drift"
                                        )
                                    failed_nodes = frozenset({"基本操作"})
                                elif basic_record.belief is None:
                                    raise P2ContractError(f"{arm} Basic posterior is missing")
                                else:
                                    beliefs["基本操作"] = basic_record.belief
                                if alkane_record.belief is None:
                                    raise P2ContractError(f"{arm} Alkane posterior is missing")
                                beliefs["烷烃"] = alkane_record.belief
                            profile = compute_profile(
                                arm=arm,
                                truths=truths,
                                beliefs_by_target=beliefs,
                                candidates=subset.candidates,
                                failed_nodes=failed_nodes,
                                replicate_basic=replicate_basic,
                                replicate_alkane=replicate_alkane,
                                oracle_selection=oracle_cache[oracle_key],
                            )
                            trace_hash = str(profile.metrics["selector_trace_hash"])
                            if trace_hash not in trace_hashes:
                                trace_hashes.add(trace_hash)
                                trace_handle.write(_json_line(profile.trace_record))
                                trace_count += 1
                            profile_handle.write(_json_line(profile.metrics))
                            metric_cube[
                                arm_index,
                                basic_truth_index,
                                alkane_truth_index,
                                replicate_basic,
                                replicate_alkane,
                            ] = _profile_metric_vector(profile.metrics)
                            profile_count += 1
    if profile_count != 160_000:
        raise P2ContractError(f"profile output count drift: {profile_count}")

    bootstrap = bootstrap_metric_cube(metric_cube, BOOTSTRAP_METRICS)
    summary = _summary_from_cube(metric_cube, bootstrap)
    summary["profile_row_count"] = profile_count
    summary["unique_selector_trace_count"] = trace_count
    summary["selector_trace_deduplication"] = (
        "profile rows reference exact trace hashes; identical selector inputs share one trace"
    )
    figure_data = _figure_data(summary, bootstrap, subset)
    _write_json(staging / "bootstrap.json", bootstrap)
    _write_json(staging / "summary.json", summary)
    _write_json(staging / "figure_data.json", figure_data)
    render_publication_figure(summary, staging, figure_data=figure_data)
    _write_report(staging / "P2_REPORT.md", summary)

    output_manifest = refresh_output_manifest(staging)
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (staging / name).is_file()]
    if missing:
        raise P2ContractError(f"required P2 output files are missing: {missing}")
    os.replace(staging, output_dir)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "output_manifest": output_manifest,
    }


if __name__ == "__main__":
    result = run_analysis()
    print(json.dumps(result["output_manifest"], ensure_ascii=False, sort_keys=True))
