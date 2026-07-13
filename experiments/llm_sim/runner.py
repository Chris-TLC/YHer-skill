"""Provider-local execution, resume, and production-engine integration for S2."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from core.learning import scoring
from engine import mastery, selector
from experiments.s0_census import normalize_kg_label, protected_filesystem_fingerprint

from .models import Persona
from .config import FROZEN_RUN_ID, load_frozen_config
from .panel import (
    annotation_map_hash,
    freeze_manipulation_panel,
    load_frozen_panel,
    normalize_annotation_map,
)
from .personas import build_personas
from .provenance import (
    collect_code_provenance,
    collect_official_input_provenance,
)
from .store import SimulationStore
from .transport import (
    HTTPProviderTransport,
    ProviderConfigurationError,
    ProviderTransport,
    load_live_environment,
    model_from_environment,
    provider_spec,
)


DEFAULT_STUDY_SEED = 2026071302
DEFAULT_MAX_ITEMS = 15
FIXED_LADDER = (0.25, 0.50, 0.75, 1.00)
CALIBRATION_ITEMS_PER_PERSONA = 4
WEAK_ACCURACY_THRESHOLD = 0.40
STRONG_ACCURACY_THRESHOLD = 0.75


class CircuitOpenError(RuntimeError):
    pass


class ModelDriftError(RuntimeError):
    pass


@dataclass
class ProviderCallPolicy:
    """Retry and circuit-breaker state owned by exactly one provider."""

    max_attempts: int = 3
    failure_threshold: int = 3
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    cooldown_seconds: float = 120.0
    sleeper: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    consecutive_failures: int = 0
    opened_until: float = 0.0
    retries: int = 0
    attempts: int = 0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.failure_threshold < 1:
            raise ValueError("retry and circuit thresholds must be positive")

    def call(self, operation: Callable[[], Mapping[str, Any]]) -> Mapping[str, Any]:
        if self.clock() < self.opened_until:
            raise CircuitOpenError("provider circuit is open")
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                self.attempts += 1
                result = operation()
                self.consecutive_failures = 0
                return result
            except Exception as exc:  # provider SDK/network boundaries are heterogeneous
                last_error = exc
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.failure_threshold:
                    self.opened_until = self.clock() + self.cooldown_seconds
                    break
                if attempt + 1 >= self.max_attempts or not _retryable(exc):
                    break
                self.retries += 1
                delay = min(
                    self.max_backoff_seconds,
                    self.base_backoff_seconds * (2**attempt),
                )
                self.sleeper(delay)
        assert last_error is not None
        raise last_error


class LLMSimulationRunner:
    """Run two-arm simulated-persona journeys without application storage."""

    def __init__(
        self,
        *,
        catalog: Any | None = None,
        kg: Any | None = None,
        store: SimulationStore,
        transport: ProviderTransport | Mapping[str, ProviderTransport] | Callable[[str], ProviderTransport] | None = None,
        personas: Sequence[Persona | Mapping[str, Any]] | None = None,
        annotation_map: Mapping[str, Any] | None = None,
        annotation_map_source: str | Path | None = None,
        panel_path: str | Path | None = None,
        study_seed: int = DEFAULT_STUDY_SEED,
        repo_root: str | Path | None = None,
        policy_factory: Callable[[], ProviderCallPolicy] = ProviderCallPolicy,
    ):
        if catalog is None:
            from core.learning.item_catalog import ItemCatalog

            catalog = ItemCatalog.from_default_data()
        if kg is None:
            from core.data.knowledge_repository import KnowledgeRepository

            kg = KnowledgeRepository()
        self.catalog = catalog
        self.kg = kg
        self.store = store
        self.annotation_map = annotation_map
        self.annotation_map_source = (
            str(Path(annotation_map_source).expanduser().resolve(strict=False))
            if annotation_map_source is not None
            else None
        )
        self.transport = transport
        self.config = load_frozen_config()
        self.study_seed = int(study_seed)
        self.repo_root = Path(repo_root).resolve(strict=False) if repo_root else Path(__file__).resolve().parents[2]
        self.code_root = Path(__file__).resolve().parents[2]
        self.personas = tuple(
            value if isinstance(value, Persona) else Persona.from_mapping(value)
            for value in (
                personas
                if personas is not None
                else build_personas(
                    kg,
                    eligible_nodes=set(catalog.open_nodes()),
                    seed=self.study_seed,
                )
            )
        )
        if not self.personas:
            raise ValueError("at least one simulated persona is required")
        self.panel_path = Path(panel_path or (store.root / "manipulation_panel.json")).resolve(strict=False)
        if not (
            self.panel_path == store.root
            or self.panel_path.is_relative_to(store.root)
        ):
            raise ValueError("manipulation panel must be inside the simulation store")
        self.policy_factory = policy_factory
        self._policies: dict[str, ProviderCallPolicy] = {}
        self._panel: dict[str, Any] | None = None
        self._preparation_manifest: dict[str, Any] | None = None

    def prepare(self) -> dict[str, Any]:
        """Freeze the study panel and provenance without resolving a provider."""

        provenance_before = collect_code_provenance(self.code_root)
        official_inputs_before = collect_official_input_provenance(
            catalog=self.catalog,
            kg=self.kg,
            personas=self.personas,
        )
        before = protected_filesystem_fingerprint(self.repo_root)
        panel = self._ensure_panel()
        after = protected_filesystem_fingerprint(self.repo_root)
        if before != after:
            raise RuntimeError("protected local_store/study state changed while freezing S2")
        provenance_after = collect_code_provenance(self.code_root)
        if provenance_before != provenance_after:
            raise RuntimeError("S2 code or git HEAD changed while freezing preparation")
        official_inputs_after = collect_official_input_provenance(
            catalog=self.catalog,
            kg=self.kg,
            personas=self.personas,
        )
        if official_inputs_before != official_inputs_after:
            raise RuntimeError("S2 official input changed while freezing preparation")
        config = load_frozen_config()
        if config.sha256 != self.config.sha256:
            raise RuntimeError("S2 config changed while freezing preparation")
        mapped = sum(
            row.get("mapping_status") == "mapped" for row in panel["annotations"]
        )
        summary = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": "llm-sim-study:pre-observation-panel",
            "provider": "study_design",
            "model_id": "no-provider-observation",
            "record_type": "llm_sim_preparation_manifest",
            "status": "panel_frozen",
            "persona_count": len(self.personas),
            "mapped_count": mapped,
            "excluded_pre_outcome_count": len(self.personas) - mapped,
            "provider_observations": 0,
            "git_head": provenance_before["git_head"],
            "code_sha256": provenance_before["code_sha256"],
            "code_files": provenance_before["code_files"],
            "official_input_sha256": official_inputs_before["official_input_sha256"],
            "official_inputs": official_inputs_before["official_inputs"],
            "panel_sha256": panel["panel_sha256"],
            "annotation_map_sha256": panel.get("annotation_map_sha256"),
            "annotation_map_source": panel.get("annotation_map_source"),
            "annotation_map_snapshot": (
                "annotation_map_snapshot.json"
                if panel.get("annotation_map_sha256")
                else None
            ),
            "config_sha256": config.sha256,
            "study_seed": self.study_seed,
            "protected_filesystem_assertion": {
                "before_sha256": before["digest"],
                "after_sha256": after["digest"],
                "unchanged": True,
                "coverage": before["coverage"],
            },
        }
        existing = self.store.read_json("preparation_manifest.json")
        if existing is not None:
            frozen_fields = (
                "run_id",
                "git_head",
                "code_sha256",
                "code_files",
                "official_input_sha256",
                "official_inputs",
                "panel_sha256",
                "annotation_map_sha256",
                "config_sha256",
                "study_seed",
                "persona_count",
            )
            if any(existing.get(key) != summary.get(key) for key in frozen_fields):
                raise RuntimeError("S2 preparation is already frozen to different provenance")
            self._preparation_manifest = existing
            return existing
        self.store.write_json("preparation_manifest.json", summary)
        self._preparation_manifest = summary
        return summary

    def run_provider(
        self,
        provider: str,
        *,
        model: str | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        arms: Sequence[str] = ("A", "B"),
        resume: bool = True,
        timeout_seconds: float = 30.0,
        prompt_revision: int = 0,
    ) -> dict[str, Any]:
        """Run one provider after the pre-outcome panel is durably frozen."""

        spec = provider_spec(provider)
        provider = spec.name
        if max_items < 1:
            raise ValueError("max_items must be positive")
        arms = tuple(str(arm) for arm in arms)
        if not arms or set(arms) - {"A", "B"}:
            raise ValueError("S2 arms must be a non-empty subset of A and B")
        if not 0 <= int(prompt_revision) <= self.config.maximum_prompt_rewrites:
            raise ValueError("prompt_revision exceeds the frozen rewrite allowance")
        prompt_revision = int(prompt_revision)

        # Hard ordering gate: no transport object or dotenv lookup is resolved
        # until this file is present and its hash has been verified.
        panel = self._ensure_panel()
        panel_sha = str(panel["panel_sha256"])
        self._assert_frozen_preparation(panel)
        live_environment = (
            load_live_environment(repo_root=self.repo_root)
            if self.transport is None
            else None
        )
        requested_model = str(
            model
            or model_from_environment(
                provider,
                spec.model_default,
                environment=live_environment,
            )
        )
        accounting = {
            "requests": 0,
            "responses": 0,
            "retries": 0,
            "skipped": 0,
            "completed": 0,
            "failed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_yuan": 0.0,
            "completed_by_arm": {arm: 0 for arm in arms},
        }
        journeys: list[dict[str, Any]] = []
        observations_started = False
        before = protected_filesystem_fingerprint(self.repo_root)
        try:
            transport = self._transport_for(provider, environment=live_environment)
        except ProviderConfigurationError:
            after = protected_filesystem_fingerprint(self.repo_root)
            manifest = self._write_provider_manifest(
                provider=provider,
                requested_model=requested_model,
                panel_sha=panel_sha,
                max_items=max_items,
                arms=arms,
                accounting=accounting,
                status="excluded_pre_outcome",
                exclusion_reason="provider_configuration_unavailable",
                prompt_revision=prompt_revision,
                protected_before=before,
                protected_after=after,
            )
            return {
                "simulated": True,
                "run_id": self.config.run_id,
                "provider": provider,
                "model_id": requested_model,
                "config_sha256": self.config.sha256,
                "panel_sha256": panel_sha,
                "prompt_revision": prompt_revision,
                "observation_started": False,
                "status": "excluded_pre_outcome",
                "exclusion_reason": "provider_configuration_unavailable",
                "calibration_attempts": [],
                "provider_eligibility": {},
                "journeys": [],
                "accounting": accounting,
                "manifest": manifest,
            }
        policy = self._policies.setdefault(provider, self.policy_factory())
        annotation_index = {
            str(row["persona_id"]): row for row in panel["annotations"]
        }
        retries_before = policy.retries
        attempts_before = policy.attempts

        try:
            calibration_attempts, provider_eligibility, calibration_responses = (
                self._run_calibration(
                    provider=provider,
                    model=requested_model,
                    transport=transport,
                    policy=policy,
                    annotation_index=annotation_index,
                    resume=resume,
                    timeout_seconds=timeout_seconds,
                    prompt_revision=prompt_revision,
                )
            )
        except Exception:
            accounting["requests"] = policy.attempts - attempts_before
            accounting["retries"] = policy.retries - retries_before
            accounting["circuit_open"] = policy.clock() < policy.opened_until
            accounting["consecutive_failures"] = policy.consecutive_failures
            accounting["failed"] += 1
            self._write_provider_manifest(
                provider=provider,
                requested_model=requested_model,
                panel_sha=panel_sha,
                max_items=max_items,
                arms=arms,
                accounting=accounting,
                status="interrupted_calibration",
                prompt_revision=prompt_revision,
                provider_eligibility={},
                calibration_attempt_count=0,
                protected_before=before,
                protected_after=protected_filesystem_fingerprint(self.repo_root),
            )
            raise
        accounting["responses"] += calibration_responses

        for persona in sorted(self.personas, key=lambda value: value.persona_id):
            annotation = annotation_index.get(persona.persona_id)
            if annotation is None:
                raise ValueError("frozen panel is missing a persona annotation")
            for arm in arms:
                relative = self.store.journey_relative_path(
                    provider, persona.persona_id, arm, prompt_revision
                )
                existing = self.store.read_json(relative)
                if resume and _complete_existing(
                    existing,
                    provider=provider,
                    requested_model=requested_model,
                    persona=persona,
                    arm=arm,
                    max_items=max_items,
                    panel_sha=panel_sha,
                    prompt_revision=prompt_revision,
                ):
                    journeys.append(existing)
                    accounting["skipped"] += 1
                    if existing.get("status") == "complete":
                        accounting["completed_by_arm"][arm] += 1
                    continue
                try:
                    journey, request_count = self._run_journey(
                        persona=persona,
                        annotation=annotation,
                        provider=provider,
                        model=requested_model,
                        arm=arm,
                        max_items=max_items,
                        timeout_seconds=timeout_seconds,
                        transport=transport,
                        policy=policy,
                        panel_sha=panel_sha,
                        resume=resume,
                        prompt_revision=prompt_revision,
                    )
                    observations_started |= request_count > 0
                    journeys.append(journey)
                    accounting["responses"] += request_count
                    accounting["completed"] += journey["status"] == "complete"
                    accounting["failed"] += journey["status"] != "complete"
                    if journey["status"] == "complete":
                        accounting["completed_by_arm"][arm] += 1
                except Exception:
                    accounting["requests"] = policy.attempts - attempts_before
                    accounting["retries"] = policy.retries - retries_before
                    accounting["circuit_open"] = policy.clock() < policy.opened_until
                    accounting["consecutive_failures"] = policy.consecutive_failures
                    accounting["failed"] += 1
                    self._write_provider_manifest(
                        provider=provider,
                        requested_model=requested_model,
                        panel_sha=panel_sha,
                        max_items=max_items,
                        arms=arms,
                        accounting=accounting,
                        status="interrupted",
                        prompt_revision=prompt_revision,
                        provider_eligibility=provider_eligibility,
                        calibration_attempt_count=len(calibration_attempts),
                        protected_before=before,
                        protected_after=protected_filesystem_fingerprint(self.repo_root),
                    )
                    raise

        accounting["requests"] = policy.attempts - attempts_before
        accounting["retries"] = policy.retries - retries_before
        accounting["circuit_open"] = policy.clock() < policy.opened_until
        accounting["consecutive_failures"] = policy.consecutive_failures
        accounting["completed"] = sum(accounting["completed_by_arm"].values())
        for journey in journeys:
            for event in journey["events"]:
                usage = event.get("usage") or {}
                accounting["input_tokens"] += int(usage.get("input_tokens") or 0)
                accounting["output_tokens"] += int(usage.get("output_tokens") or 0)
                accounting["cost_yuan"] += float(event.get("cost_yuan") or 0.0)
        for event in calibration_attempts:
            usage = event.get("usage") or {}
            accounting["input_tokens"] += int(usage.get("input_tokens") or 0)
            accounting["output_tokens"] += int(usage.get("output_tokens") or 0)
            accounting["cost_yuan"] += float(event.get("cost_yuan") or 0.0)
        accounting["cost_yuan"] = round(accounting["cost_yuan"], 12)
        observations_started = bool(calibration_attempts) or any(
            journey.get("events") for journey in journeys
        )
        after = protected_filesystem_fingerprint(self.repo_root)
        if before != after:
            raise RuntimeError("protected local_store/study state changed during S2 run")
        journey_status = "complete" if accounting["failed"] == 0 else "partial"
        status = _provider_run_status(journey_status, provider_eligibility)
        manifest = self._write_provider_manifest(
            provider=provider,
            requested_model=requested_model,
            panel_sha=panel_sha,
            max_items=max_items,
            arms=arms,
            accounting=accounting,
            status=status,
            prompt_revision=prompt_revision,
            provider_eligibility=provider_eligibility,
            calibration_attempt_count=len(calibration_attempts),
            protected_before=before,
            protected_after=after,
        )
        return {
            "simulated": True,
            "run_id": self.config.run_id,
            "provider": provider,
            "model_id": requested_model,
            "config_sha256": self.config.sha256,
            "panel_sha256": panel_sha,
            "prompt_revision": prompt_revision,
            "observation_started": observations_started,
            "status": status,
            "calibration_attempts": calibration_attempts,
            "provider_eligibility": provider_eligibility,
            "journeys": journeys,
            "accounting": accounting,
            "manifest": manifest,
        }

    def _run_calibration(
        self,
        *,
        provider: str,
        model: str,
        transport: ProviderTransport,
        policy: ProviderCallPolicy,
        annotation_index: Mapping[str, Mapping[str, Any]],
        resume: bool,
        timeout_seconds: float,
        prompt_revision: int,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int]:
        """Administer the frozen four-item panel before any A/B journey.

        Calibration has no mastery side effects and is persisted separately from
        journeys.  Its accuracy denominator is therefore auditable as
        ``personas_in_strength * 4`` (100 for the frozen 25-pair study).
        """

        attempts: list[dict[str, Any]] = []
        responses = 0
        eligibility: dict[str, dict[str, Any]] = {}
        for strength in ("weak", "strong"):
            personas = [
                persona
                for persona in sorted(self.personas, key=lambda value: value.persona_id)
                if persona.strength == strength
            ]
            strength_attempts: list[dict[str, Any]] = []
            calibration_ready = True
            for persona in personas:
                annotation = annotation_index[persona.persona_id]
                if annotation.get("calibration_status") != "ready":
                    calibration_ready = False
                    continue
                relative = self.store.calibration_relative_path(
                    provider, persona.persona_id, prompt_revision
                )
                existing = self.store.read_json(relative)
                if _complete_calibration(
                    existing,
                    provider=provider,
                    model=model,
                    persona=persona,
                    panel_sha=str(self._panel["panel_sha256"]),
                    prompt_revision=prompt_revision,
                ):
                    rows = [dict(event) for event in existing.get("events", ())]
                    strength_attempts.extend(rows)
                    continue
                checkpoint_relative = self.store.calibration_checkpoint_relative_path(
                    provider, persona.persona_id, prompt_revision
                )
                checkpoint = self.store.read_json(checkpoint_relative) if resume else None
                events = _resume_calibration_events(
                    checkpoint,
                    provider=provider,
                    model=model,
                    persona=persona,
                    panel_sha=str(self._panel["panel_sha256"]),
                    prompt_revision=prompt_revision,
                    calibration_items=annotation.get("calibration_items", ()),
                )
                calibration_items = list(annotation.get("calibration_items", ()))
                for item_row in calibration_items[len(events) :]:
                    item = _catalog_item_by_id(self.catalog, item_row["item_id"])
                    if item is None:
                        calibration_ready = False
                        break
                    raw = policy.call(
                        lambda item=item: transport.complete(
                            provider=provider,
                            model=model,
                            messages=_messages(
                                persona,
                                item,
                                phase="calibration",
                                prompt_revision=prompt_revision,
                            ),
                            timeout_seconds=timeout_seconds,
                        )
                    )
                    responses += 1
                    returned_model = str(raw.get("model_returned") or "").strip()
                    if returned_model != model:
                        raise ModelDriftError(
                            f"provider {provider} returned a model id different from the frozen request"
                        )
                    answer = _parse_answer(
                        str(raw.get("content") or ""),
                        response_kind=str(getattr(item, "scoring_mode", "mcq")),
                    )
                    score = scoring.score_item(item, answer or "")
                    usage_raw = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
                    usage = {
                        "input_tokens": max(0, int(usage_raw.get("input_tokens") or 0)),
                        "output_tokens": max(0, int(usage_raw.get("output_tokens") or 0)),
                    }
                    target_hit = (
                        item_row.get("mapping_status") == "mapped"
                        and answer == item_row.get("target_option")
                    )
                    event = {
                        "simulated": True,
                        "run_id": self.config.run_id,
                        "persona_id": persona.persona_id,
                        "provider": provider,
                        "model_id": returned_model,
                        "record_type": "llm_sim_calibration_attempt",
                        "phase": "calibration",
                        "pair_id": persona.pair_id,
                        "strength": persona.strength,
                        "target_node": persona.target_node,
                        "failure_id": persona.failure_id,
                        "position": len(events) + 1,
                        "item_id": str(item_row["item_id"]),
                        "family_id": str(item_row["family_id"]),
                        "answer": answer,
                        "correct": score.correct,
                        "score_status": score.status,
                        "target_misconception_hit": bool(target_hit)
                        if item_row.get("mapping_status") == "mapped"
                        else None,
                        "random_wrong_option_baseline": item_row.get(
                            "random_wrong_option_baseline"
                        ),
                        "usage": usage,
                        "cost_yuan": max(0.0, float(raw.get("cost_yuan") or 0.0)),
                        "panel_sha256": self._panel["panel_sha256"],
                        "prompt_revision": prompt_revision,
                    }
                    events.append(event)
                    checkpoint = {
                        "simulated": True,
                        "run_id": self.config.run_id,
                        "persona_id": persona.persona_id,
                        "provider": provider,
                        "model_id": returned_model,
                        "record_type": "llm_sim_calibration",
                        "status": "in_progress",
                        "events": events,
                        "panel_sha256": self._panel["panel_sha256"],
                        "prompt_revision": prompt_revision,
                    }
                    self.store.write_json(
                        checkpoint_relative,
                        checkpoint,
                    )
                record = {
                    "simulated": True,
                    "run_id": self.config.run_id,
                    "persona_id": persona.persona_id,
                    "provider": provider,
                    "model_id": model,
                    "record_type": "llm_sim_calibration",
                    "status": "complete" if len(events) == CALIBRATION_ITEMS_PER_PERSONA else "incomplete",
                    "strength": persona.strength,
                    "events": events,
                    "panel_sha256": self._panel["panel_sha256"],
                    "prompt_revision": prompt_revision,
                }
                self.store.write_json(relative, record)
                partial_path = self.store.root / checkpoint_relative
                try:
                    partial_path.unlink()
                except FileNotFoundError:
                    pass
                strength_attempts.extend(events)
            attempts.extend(strength_attempts)
            eligibility[strength] = _calibration_eligibility(
                strength=strength,
                personas=personas,
                attempts=strength_attempts,
                panel_rows=[annotation_index[p.persona_id] for p in personas],
                calibration_ready=calibration_ready,
                minimum_complete=self.config.minimum_complete_per_cell,
                formal_persona_count=self.config.persona_count // 2,
                prompt_revision=prompt_revision,
                maximum_prompt_rewrites=self.config.maximum_prompt_rewrites,
            )
        return attempts, eligibility, responses

    def _run_journey(
        self,
        *,
        persona: Persona,
        annotation: Mapping[str, Any],
        provider: str,
        model: str,
        arm: str,
        max_items: int,
        timeout_seconds: float,
        transport: ProviderTransport,
        policy: ProviderCallPolicy,
        panel_sha: str,
        resume: bool,
        prompt_revision: int,
    ) -> tuple[dict[str, Any], int]:
        local_items = list(
            self.catalog.for_node(persona.target_node, deterministic_only=True)
        )
        prerequisite_items = (
            _prerequisite_items(self.catalog, persona.target_node)
            if arm == "A"
            else []
        )
        local_ids = {str(getattr(item, "item_id", "")) for item in local_items}
        prerequisite_items = [
            item
            for item in prerequisite_items
            if str(getattr(item, "item_id", "")) not in local_ids
        ]
        items = local_items + prerequisite_items
        role_by_item_id = {
            **{str(getattr(item, "item_id", "")): "local" for item in local_items},
            **{
                str(getattr(item, "item_id", "")): "prereq"
                for item in prerequisite_items
            },
        }
        items.sort(key=lambda item: (str(getattr(item, "family_id", "")), str(getattr(item, "item_id", ""))))
        if not items:
            journey = self._journey_record(
                persona=persona,
                provider=provider,
                model=model,
                arm=arm,
                max_items=max_items,
                panel_sha=panel_sha,
                events=[],
                terminal_reason="structural_failure_no_items",
                prompt_revision=prompt_revision,
            )
            self.store.write_json(
                self.store.journey_relative_path(
                    provider, persona.persona_id, arm, prompt_revision
                ),
                journey,
            )
            return journey, 0

        checkpoint_rel = self.store.checkpoint_relative_path(
            provider, persona.persona_id, arm, prompt_revision
        )
        checkpoint = self.store.read_json(checkpoint_rel) if resume else None
        events = _resume_events(
            checkpoint,
            provider=provider,
            model=model,
            persona=persona,
            arm=arm,
            max_items=max_items,
            panel_sha=panel_sha,
            prompt_revision=prompt_revision,
        )
        node = mastery.NodeBelief(mastery.UNIFORM.copy())
        for event in events:
            mastery.observe(
                node,
                np.asarray(event["inference_likelihood"], dtype=float),
                float(event["position"]),
                is_direct=event.get("role") == "local",
            )
        seen_ids = {str(event["item_id"]) for event in events}
        request_count = 0
        terminal_reason = "budget_exhausted"
        returned_model = model
        for position in range(len(events) + 1, max_items + 1):
            belief_before = mastery.get_belief(node, float(position))
            item = _choose_item(
                items,
                arm=arm,
                position=position,
                target_node=persona.target_node,
                belief=belief_before,
                direct_answers=node.direct_answers,
                seen_ids=seen_ids,
                role_by_item_id=role_by_item_id,
            )
            if item is None:
                terminal_reason = "structural_failure_item_pool"
                break
            messages = _messages(persona, item, prompt_revision=prompt_revision)
            raw = policy.call(
                lambda: transport.complete(
                    provider=provider,
                    model=model,
                    messages=messages,
                    timeout_seconds=timeout_seconds,
                )
            )
            request_count += 1
            returned_model = str(raw.get("model_returned") or "").strip()
            if returned_model != model:
                raise ModelDriftError(
                    f"provider {provider} returned a model id different from the frozen request"
                )
            answer = _parse_answer(
                str(raw.get("content") or ""),
                response_kind=str(getattr(item, "scoring_mode", "mcq")),
            )
            score = scoring.score_item(item, answer or "")
            item_id = str(getattr(item, "item_id", ""))
            role = role_by_item_id[item_id]
            if role == "prereq" and score.update_allowed and score.correct is not None:
                engine_type = (
                    "numeric"
                    if getattr(item, "scoring_mode", "") == "numeric"
                    else "mcq"
                )
                probabilities = mastery.prereq_correct_probs(item_type=engine_type)
                likelihood = (
                    mastery.likelihood_correct(probabilities)
                    if score.correct
                    else mastery.likelihood_wrong_binary(probabilities)
                )
            else:
                likelihood = np.asarray(score.likelihood, dtype=float)
            prior = mastery.get_belief(node, float(position))
            direct_before = node.direct_answers
            if score.update_allowed:
                mastery.observe(
                    node,
                    likelihood,
                    float(position),
                    is_direct=role == "local",
                )
            posterior = mastery.get_belief(node, float(position))
            direct_after = node.direct_answers
            usage_raw = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
            usage = {
                "input_tokens": max(0, int(usage_raw.get("input_tokens") or 0)),
                "output_tokens": max(0, int(usage_raw.get("output_tokens") or 0)),
            }
            target_mapping_applicable = (
                annotation.get("mapping_status") == "mapped"
                and annotation.get("target_item_id") == item_id
            )
            target_hit = target_mapping_applicable and answer == annotation.get("target_option")
            envelope = {
                "simulated": True,
                "run_id": self.config.run_id,
                "persona_id": persona.persona_id,
                "provider": provider,
                "model_id": returned_model,
            }
            event = {
                **envelope,
                "record_type": "llm_sim_event",
                "pair_id": persona.pair_id,
                "strength": persona.strength,
                "target_node": persona.target_node,
                "failure_id": persona.failure_id,
                "arm": arm,
                "position": position,
                "role": role,
                "item_id": item_id,
                "family_id": str(getattr(item, "family_id", "")),
                "difficulty": float(getattr(item, "difficulty", 0.5)),
                "answer": answer,
                "score_status": score.status,
                "correct": score.correct,
                "update_applied": bool(score.update_allowed),
                "inference_likelihood": likelihood.tolist(),
                "prior_belief": prior.tolist(),
                "posterior_belief": posterior.tolist(),
                "direct_answers_before": direct_before,
                "direct_answers_after": direct_after,
                "target_mapping_status": annotation.get("mapping_status"),
                "target_mapping_applicable": bool(target_mapping_applicable),
                "target_misconception_hit": bool(target_hit)
                if annotation.get("mapping_status") == "mapped"
                else None,
                "random_wrong_option_baseline": annotation.get("random_wrong_option_baseline"),
                "usage": usage,
                "cost_yuan": max(0.0, float(raw.get("cost_yuan") or 0.0)),
                "panel_sha256": panel_sha,
            }
            events.append(event)
            seen_ids.add(item_id)
            checkpoint_record = self._journey_record(
                persona=persona,
                provider=provider,
                model=returned_model,
                arm=arm,
                max_items=max_items,
                panel_sha=panel_sha,
                events=events,
                terminal_reason="in_progress",
                prompt_revision=prompt_revision,
            )
            checkpoint_record["status"] = "in_progress"
            self.store.write_json(checkpoint_rel, checkpoint_record)
            should_stop = selector.should_stop(
                {persona.target_node: posterior},
                [persona.target_node],
                direct_answers={persona.target_node: node.direct_answers},
                budget_items=max_items + 1,
                asked=position,
            )
            if should_stop:
                terminal_reason = "confidence"
                break

        journey = self._journey_record(
            persona=persona,
            provider=provider,
            model=returned_model,
            arm=arm,
            max_items=max_items,
            panel_sha=panel_sha,
            events=events,
            terminal_reason=terminal_reason,
            prompt_revision=prompt_revision,
        )
        self.store.write_json(
            self.store.journey_relative_path(
                provider, persona.persona_id, arm, prompt_revision
            ),
            journey,
        )
        checkpoint_path = self.store.root / checkpoint_rel
        try:
            checkpoint_path.unlink()
        except FileNotFoundError:
            pass
        return journey, request_count

    def _journey_record(
        self,
        *,
        persona: Persona,
        provider: str,
        model: str,
        arm: str,
        max_items: int,
        panel_sha: str,
        events: Sequence[Mapping[str, Any]],
        terminal_reason: str,
        prompt_revision: int,
    ) -> dict[str, Any]:
        beliefs = list(events[-1]["posterior_belief"]) if events else mastery.UNIFORM.tolist()
        mapped_events = [
            event for event in events if event.get("target_mapping_applicable") is True
        ]
        wrong_events = [event for event in mapped_events if event.get("correct") is False]
        target_hits = sum(event.get("target_misconception_hit") is True for event in wrong_events)
        correct_count = sum(event.get("correct") is True for event in events)
        status = "complete" if terminal_reason in {"confidence", "budget_exhausted"} else "incomplete"
        return {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": persona.persona_id,
            "provider": provider,
            "model_id": model,
            "record_type": "llm_sim_journey",
            "status": status,
            "pair_id": persona.pair_id,
            "strength": persona.strength,
            "target_node": persona.target_node,
            "failure_id": persona.failure_id,
            "annotation_source": persona.annotation_source,
            "arm": arm,
            "max_items": int(max_items),
            "actual_administered_count": len(events),
            "terminal_reason": terminal_reason,
            "events": [dict(event) for event in events],
            "final_belief": beliefs,
            "accuracy": correct_count / len(events) if events else None,
            "target_misconception_hit_count": target_hits,
            "target_misconception_wrong_denominator": len(wrong_events),
            "panel_sha256": panel_sha,
            "config_sha256": self.config.sha256,
            "prompt_revision": prompt_revision,
        }

    def _ensure_panel(self) -> dict[str, Any]:
        if self._panel is not None:
            return self._panel
        if self.panel_path.is_file():
            panel = load_frozen_panel(self.panel_path)
            expected_annotation_hash = annotation_map_hash(self.annotation_map)
            actual_annotation_hash = panel.get("annotation_map_sha256")
            if (
                self.annotation_map is not None
                and actual_annotation_hash != expected_annotation_hash
            ):
                raise ValueError(
                    "frozen manipulation panel annotation map does not match input"
                )
        else:
            panel = freeze_manipulation_panel(
                personas=self.personas,
                catalog=self.catalog,
                annotation_map=self.annotation_map,
                annotation_map_source=self.annotation_map_source,
                output_path=self.panel_path,
                study_seed=self.study_seed,
            )
        expected_ids = sorted(persona.persona_id for persona in self.personas)
        actual_ids = sorted(str(row.get("persona_id")) for row in panel["annotations"])
        if actual_ids != expected_ids:
            raise ValueError("frozen manipulation panel does not match configured personas")
        self._ensure_annotation_map_snapshot(panel)
        self._panel = panel
        return panel

    def _ensure_annotation_map_snapshot(self, panel: Mapping[str, Any]) -> None:
        if self.annotation_map is None:
            expected_hash = panel.get("annotation_map_sha256")
            if expected_hash is None:
                return
            existing = self.store.read_json("annotation_map_snapshot.json")
            if not existing or any(
                (
                    existing.get("annotation_map_sha256") != expected_hash,
                    existing.get("panel_sha256") != panel.get("panel_sha256"),
                    annotation_map_hash(existing.get("annotation_map"))
                    != expected_hash,
                )
            ):
                raise ValueError("frozen annotation map snapshot is missing or invalid")
            return
        normalized = normalize_annotation_map(self.annotation_map)
        expected_hash = annotation_map_hash(normalized)
        if panel.get("annotation_map_sha256") != expected_hash:
            raise ValueError("frozen panel does not bind the supplied annotation map")
        record = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": "llm-sim-study:annotation-map",
            "provider": "study_design",
            "model_id": "no-provider-observation",
            "record_type": "llm_sim_annotation_map_snapshot",
            "annotation_map_sha256": expected_hash,
            "source_path": self.annotation_map_source or "inline_mapping",
            "annotation_map": normalized,
            "panel_sha256": panel["panel_sha256"],
        }
        existing = self.store.read_json("annotation_map_snapshot.json")
        if existing is not None and existing != record:
            raise ValueError("a different annotation map snapshot already exists")
        if existing is None:
            self.store.write_json("annotation_map_snapshot.json", record)

    def _assert_frozen_preparation(self, panel: Mapping[str, Any]) -> None:
        preparation = self.store.read_json("preparation_manifest.json")
        if preparation is None:
            preparation = self.prepare()
        current = collect_code_provenance(self.code_root)
        current_config = load_frozen_config()
        current_official_inputs = collect_official_input_provenance(
            catalog=self.catalog,
            kg=self.kg,
            personas=self.personas,
        )
        expected = {
            "git_head": preparation.get("git_head"),
            "code_sha256": preparation.get("code_sha256"),
            "code_files": preparation.get("code_files"),
        }
        if current != expected:
            raise RuntimeError("code or git HEAD changed after S2 preparation")
        expected_official_inputs = {
            "official_input_sha256": preparation.get("official_input_sha256"),
            "official_inputs": preparation.get("official_inputs"),
        }
        if current_official_inputs != expected_official_inputs:
            raise RuntimeError("official input changed after S2 preparation")
        if any(
            (
                preparation.get("panel_sha256") != panel.get("panel_sha256"),
                preparation.get("run_id") != self.config.run_id,
                preparation.get("config_sha256") != self.config.sha256,
                preparation.get("config_sha256") != current_config.sha256,
                int(preparation.get("study_seed", -1)) != self.study_seed,
            )
        ):
            raise RuntimeError("S2 frozen preparation does not match the live run")
        self._preparation_manifest = preparation

    def _transport_for(
        self,
        provider: str,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> ProviderTransport:
        source = self.transport
        if source is None:
            return HTTPProviderTransport.from_environment(
                provider,
                environment=environment,
                repo_root=self.repo_root,
            )
        if isinstance(source, Mapping):
            try:
                return source[provider]
            except KeyError as exc:
                raise ValueError(f"no transport configured for provider {provider}") from exc
        if callable(source) and not hasattr(source, "complete"):
            return source(provider)
        return source  # type: ignore[return-value]

    def _write_provider_manifest(
        self,
        *,
        provider: str,
        requested_model: str,
        panel_sha: str,
        max_items: int,
        arms: Sequence[str],
        accounting: Mapping[str, Any],
        status: str,
        protected_before: Mapping[str, Any],
        protected_after: Mapping[str, Any],
        exclusion_reason: str | None = None,
        prompt_revision: int = 0,
        provider_eligibility: Mapping[str, Any] | None = None,
        calibration_attempt_count: int = 0,
    ) -> dict[str, Any]:
        panel_rows = list((self._panel or {}).get("annotations") or ())
        mapped_personas = sum(
            row.get("mapping_status") == "mapped" for row in panel_rows
        )
        calibration_reportable = all(
            bool((provider_eligibility or {}).get(strength, {}).get("eligible"))
            for strength in ("weak", "strong")
        )
        raw_completed = dict(accounting.get("completed_by_arm") or {})
        eligible_completed = {
            arm: int(raw_completed.get(arm, 0)) if calibration_reportable else 0
            for arm in arms
        }
        completion_reportable = (
            len(self.personas) == self.config.persona_count
            and tuple(arms) == self.config.arms
            and all(
                eligible_completed[arm] >= self.config.minimum_complete_per_cell
                for arm in arms
            )
        )
        manifest = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": f"llm-sim-provider:{provider}",
            "provider": provider,
            "model_id": requested_model,
            "record_type": "llm_sim_provider_manifest",
            "status": status,
            "exclusion_reason": exclusion_reason,
            "prompt_revision": prompt_revision,
            "calibration_attempt_count": calibration_attempt_count,
            "provider_eligibility": dict(provider_eligibility or {}),
            "panel_sha256": panel_sha,
            "config_sha256": self.config.sha256,
            "study_seed": self.study_seed,
            "git_head": (self._preparation_manifest or {}).get("git_head"),
            "code_sha256": (self._preparation_manifest or {}).get("code_sha256"),
            "official_input_sha256": (self._preparation_manifest or {}).get(
                "official_input_sha256"
            ),
            "official_inputs": (self._preparation_manifest or {}).get(
                "official_inputs"
            ),
            "persona_count": len(self.personas),
            "arms": list(arms),
            "max_items": max_items,
            "minimum_complete_per_cell": self.config.minimum_complete_per_cell,
            "reportability": {
                "minimum_complete_per_cell": self.config.minimum_complete_per_cell,
                "cell_completed": raw_completed,
                "eligible_cell_completed": eligible_completed,
                "completion_reportable": completion_reportable,
                "mechanically_mapped_personas": mapped_personas,
                "pre_outcome_mapping_exclusions": len(panel_rows) - mapped_personas,
                "manipulation_metric_reportable": (
                    completion_reportable and calibration_reportable and mapped_personas > 0
                ),
                "calibration_reportable": calibration_reportable,
                "reportable": (
                    completion_reportable
                    and calibration_reportable
                    and mapped_personas > 0
                ),
            },
            "accounting": dict(accounting),
            "protected_filesystem_assertion": {
                "before_sha256": protected_before["digest"],
                "after_sha256": protected_after["digest"],
                "unchanged": protected_before == protected_after,
                "coverage": protected_before["coverage"],
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        suffix = f"__prompt-v{prompt_revision}" if prompt_revision else ""
        self.store.write_json(
            Path("providers") / f"{provider}{suffix}.json", manifest
        )
        return manifest


def _provider_run_status(
    journey_status: str,
    provider_eligibility: Mapping[str, Mapping[str, Any]],
) -> str:
    statuses = {
        str(row.get("status"))
        for row in provider_eligibility.values()
        if isinstance(row, Mapping)
    }
    if "prompt_rewrite_available" in statuses:
        return "calibration_rewrite_required"
    if "excluded_post_calibration" in statuses:
        return "excluded_post_calibration"
    if "excluded_pre_outcome" in statuses:
        return "excluded_pre_outcome"
    if "incomplete_calibration" in statuses:
        return "incomplete_calibration"
    return journey_status


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    if status == 429 or (isinstance(status, int) and status >= 500):
        return True
    text = str(exc).lower()
    return bool(re.search(r"\b5\d\d\b", text)) or any(
        token in text
        for token in ("429", "rate limit", "timeout", "network", "connection", "http 5")
    )


def _prerequisite_items(catalog: Any, target_node: str) -> list[Any]:
    """Resolve prerequisite pools with the same mechanical labels as S0."""

    if not hasattr(catalog, "prerequisites_for"):
        return []
    raw_labels = tuple(str(value) for value in catalog.prerequisites_for(target_node))
    catalog_items = getattr(catalog, "items", {})
    values = catalog_items.values() if isinstance(catalog_items, Mapping) else catalog_items
    node_labels = sorted(
        {
            str(node)
            for item in values
            for node in tuple(getattr(item, "node_ids", ()) or ())
            if str(node)
        }
    )
    normalized: dict[str, list[str]] = {}
    for node in node_labels:
        normalized.setdefault(normalize_kg_label(node), []).append(node)
    resolved: list[str] = []
    for raw in raw_labels:
        if raw in node_labels:
            selected = raw
        else:
            candidates = normalized.get(normalize_kg_label(raw), ())
            selected = candidates[0] if len(candidates) == 1 else ""
        if selected and selected not in resolved:
            resolved.append(selected)
    output: list[Any] = []
    seen_ids: set[str] = set()
    for node in resolved:
        for item in catalog.for_node(node, deterministic_only=True):
            item_id = str(getattr(item, "item_id", ""))
            if item_id and item_id not in seen_ids:
                output.append(item)
                seen_ids.add(item_id)
    return output


def _candidate_row(item: Any, target_node: str, role: str) -> dict[str, Any]:
    return {
        "item_id": str(getattr(item, "item_id", "")),
        "target_node": target_node,
        "node": str(next(iter(getattr(item, "node_ids", (target_node,))), target_node)),
        "difficulty": float(getattr(item, "difficulty", 0.5)),
        "item_type": "numeric" if getattr(item, "scoring_mode", "") == "numeric" else "mcq",
        "role": role,
        "holdout": False,
    }


def _choose_item(
    items: Sequence[Any],
    *,
    arm: str,
    position: int,
    target_node: str,
    belief: np.ndarray,
    direct_answers: int,
    seen_ids: set[str],
    role_by_item_id: Mapping[str, str],
) -> Any | None:
    unseen = [item for item in items if str(getattr(item, "item_id", "")) not in seen_ids]
    pool = unseen or list(items)
    if arm == "A":
        rows = [
            _candidate_row(
                item,
                target_node,
                role_by_item_id[str(getattr(item, "item_id", ""))],
            )
            for item in pool
        ]
        chosen = selector.select_next(
            rows,
            {target_node: belief},
            [target_node],
            seen_ids=set(),
            prereq_available=any(row["role"] == "prereq" for row in rows),
            asked_per_node={target_node: direct_answers},
        )
        if chosen is None:
            return None
        chosen_id = str(chosen.get("item_id"))
        return next((item for item in pool if str(getattr(item, "item_id", "")) == chosen_id), None)
    requested = FIXED_LADDER[(position - 1) % len(FIXED_LADDER)]
    return min(
        pool,
        key=lambda item: (
            abs(float(getattr(item, "difficulty", 0.5)) - requested),
            str(getattr(item, "family_id", "")),
            str(getattr(item, "item_id", "")),
        ),
        default=None,
    )


def _public_question(item: Any) -> dict[str, Any]:
    public_method = getattr(item, "public_question", None)
    if callable(public_method):
        return dict(public_method())
    return {
        "kind": str(getattr(item, "item_type", "mcq")),
        "stem_text": str(getattr(item, "stem_text", "")),
        "options": dict(getattr(item, "options", {}) or {}),
        "difficulty": float(getattr(item, "difficulty", 0.5)),
        "nodes": list(getattr(item, "node_ids", ()) or ()),
    }


def _messages(
    persona: Persona,
    item: Any,
    *,
    phase: str = "journey",
    prompt_revision: int = 0,
) -> list[dict[str, str]]:
    question = _public_question(item)
    response_kind = str(
        question.get("kind") or getattr(item, "scoring_mode", "mcq")
    ).strip().lower()
    if response_kind == "numeric":
        answer_contract = {"answer": "<numeric string>", "rationale": "<short reason>"}
        answer_instruction = "The answer field must be one numeric value, with a unit only if the question requires it. "
    else:
        answer_contract = {"answer": "<option key>", "rationale": "<short reason>"}
        answer_instruction = "The answer field must be exactly one option key shown in question.options. "
    strength_instruction = (
        "Use a weak, misconception-prone high-school chemistry understanding. "
        "Remain consistent with the stated misconception; do not deliberately randomize."
        if persona.strength == "weak"
        else "Use a strong high-school chemistry understanding and solve the item carefully."
    )
    revision_instruction = (
        (
            " Apply the stated misconception consistently even when you recognize a more advanced solution; "
            "do not answer as an expert."
            if persona.strength == "weak"
            else " Recheck the chemistry and calculation once before returning the answer."
        )
        if prompt_revision == 1
        else ""
    )
    system = (
        "You are a SIMULATED student for an evaluation, not a real participant. "
        + answer_instruction
        + "Return strict JSON only: "
        + json.dumps(answer_contract, ensure_ascii=False)
        + ". "
        + strength_instruction
        + revision_instruction
        + f" This is the predeclared {phase} phase (prompt revision {prompt_revision})."
    )
    persona_context = {
        "strength": persona.strength,
        "target_node": persona.target_node,
        "known_failure_cause": persona.failure_cause if persona.strength == "weak" else "",
        "known_failure_symptom": persona.failure_symptom if persona.strength == "weak" else "",
        "question": question,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(persona_context, ensure_ascii=False, sort_keys=True)},
    ]


def _catalog_item_by_id(catalog: Any, item_id: str) -> Any | None:
    items = getattr(catalog, "items", {})
    if isinstance(items, Mapping) and item_id in items:
        return items[item_id]
    for node in getattr(catalog, "open_nodes", lambda: {})():
        for item in catalog.for_node(node, deterministic_only=True):
            if str(getattr(item, "item_id", "")) == str(item_id):
                return item
    return None


def _complete_calibration(
    record: Mapping[str, Any] | None,
    *,
    provider: str,
    model: str,
    persona: Persona,
    panel_sha: str,
    prompt_revision: int,
) -> bool:
    if not record or record.get("status") != "complete":
        return False
    return all(
        (
            record.get("simulated") is True,
            record.get("run_id") == FROZEN_RUN_ID,
            record.get("record_type") == "llm_sim_calibration",
            record.get("provider") == provider,
            record.get("model_id") == model,
            record.get("persona_id") == persona.persona_id,
            record.get("panel_sha256") == panel_sha,
            int(record.get("prompt_revision", -1)) == prompt_revision,
            len(record.get("events") or ()) == CALIBRATION_ITEMS_PER_PERSONA,
        )
    )


def _resume_calibration_events(
    record: Mapping[str, Any] | None,
    *,
    provider: str,
    model: str,
    persona: Persona,
    panel_sha: str,
    prompt_revision: int,
    calibration_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not record:
        return []
    if not all(
        (
            record.get("simulated") is True,
            record.get("run_id") == FROZEN_RUN_ID,
            record.get("record_type") == "llm_sim_calibration",
            record.get("status") == "in_progress",
            record.get("provider") == provider,
            record.get("model_id") == model,
            record.get("persona_id") == persona.persona_id,
            record.get("panel_sha256") == panel_sha,
            int(record.get("prompt_revision", -1)) == prompt_revision,
        )
    ):
        return []
    events = [dict(event) for event in record.get("events", ())]
    expected_ids = [str(row["item_id"]) for row in calibration_items]
    if len(events) > CALIBRATION_ITEMS_PER_PERSONA or any(
        event.get("record_type") != "llm_sim_calibration_attempt"
        or event.get("model_id") != model
        or int(event.get("position", -1)) != index
        or str(event.get("item_id")) != expected_ids[index - 1]
        for index, event in enumerate(events, start=1)
    ):
        return []
    return events


def _calibration_eligibility(
    *,
    strength: str,
    personas: Sequence[Persona],
    attempts: Sequence[Mapping[str, Any]],
    panel_rows: Sequence[Mapping[str, Any]],
    calibration_ready: bool,
    minimum_complete: int,
    formal_persona_count: int,
    prompt_revision: int,
    maximum_prompt_rewrites: int,
) -> dict[str, Any]:
    denominator = len(attempts)
    correct = sum(row.get("correct") is True for row in attempts)
    accuracy = correct / denominator if denominator else None
    if strength == "weak":
        threshold = "<0.40"
        band_pass = accuracy is not None and accuracy < WEAK_ACCURACY_THRESHOLD
    else:
        threshold = ">0.75"
        band_pass = accuracy is not None and accuracy > STRONG_ACCURACY_THRESHOLD
    expected_n = len(personas) * CALIBRATION_ITEMS_PER_PERSONA
    formal_expected_n = formal_persona_count * CALIBRATION_ITEMS_PER_PERSONA
    target_panel_ready = bool(panel_rows) and all(
        row.get("mapping_status") == "mapped" for row in panel_rows
    )
    mapped_wrong = [
        row
        for row in attempts
        if row.get("target_misconception_hit") is not None
        and row.get("correct") is False
    ]
    target_hits = sum(
        row.get("target_misconception_hit") is True for row in mapped_wrong
    )
    target_hit_rate = target_hits / len(mapped_wrong) if mapped_wrong else None
    baselines = [
        float(row["random_wrong_option_baseline"])
        for row in mapped_wrong
        if row.get("random_wrong_option_baseline") is not None
    ]
    if not calibration_ready:
        status = "excluded_pre_outcome"
        reason = "insufficient_family_distinct_calibration_mcq"
    elif not target_panel_ready:
        status = "excluded_pre_outcome"
        reason = "no_mechanical_target_option_mapping"
    elif denominator != expected_n:
        status = "incomplete_calibration"
        reason = "calibration_denominator_incomplete"
    elif not band_pass:
        status = "prompt_rewrite_available" if prompt_revision < maximum_prompt_rewrites else "excluded_post_calibration"
        reason = "accuracy_band_failed"
    else:
        status = "eligible"
        reason = None
    return {
        "strength": strength,
        "n": denominator,
        "accuracy_denominator": denominator,
        "expected_n_for_configured_personas": expected_n,
        "formal_expected_n": formal_expected_n,
        "accuracy": accuracy,
        "correct": correct,
        "threshold": threshold,
        "accuracy_band_pass": bool(band_pass),
        "target_panel_ready": target_panel_ready,
        "target_misconception_hit_count": target_hits,
        "target_misconception_wrong_denominator": len(mapped_wrong),
        "target_misconception_hit_rate": target_hit_rate,
        "random_wrong_option_baseline_mean": (
            sum(baselines) / len(baselines) if baselines else None
        ),
        "calibration_item_count": CALIBRATION_ITEMS_PER_PERSONA if calibration_ready else 0,
        "eligible": status == "eligible",
        "status": status,
        "excluded_reason": reason,
        "minimum_complete_per_cell": minimum_complete,
        "personas": len(personas),
        "prompt_revision": prompt_revision,
    }


def _parse_answer(content: str, *, response_kind: str = "mcq") -> str | None:
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        if isinstance(value, Mapping):
            candidate = value.get("answer")
            if candidate is not None:
                if str(response_kind).strip().lower() == "numeric":
                    normalized_numeric = str(candidate).strip()
                    return normalized_numeric or None
                normalized = scoring.normalize_mcq(str(candidate))
                if normalized is not None:
                    return normalized
    except json.JSONDecodeError:
        pass
    if str(response_kind).strip().lower() == "numeric":
        return text if scoring.parse_scalar_answer(text) is not None else None
    exact = scoring.normalize_mcq(text)
    if exact is not None:
        return exact
    match = re.search(r"(?:answer|答案|选择)\s*[:：]?\s*([A-F])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _complete_existing(
    record: Mapping[str, Any] | None,
    *,
    provider: str,
    requested_model: str,
    persona: Persona,
    arm: str,
    max_items: int,
    panel_sha: str,
    prompt_revision: int,
) -> bool:
    if not record or record.get("status") != "complete":
        return False
    return all(
        (
            record.get("simulated") is True,
            record.get("run_id") == FROZEN_RUN_ID,
            record.get("provider") == provider,
            record.get("model_id") == requested_model,
            record.get("persona_id") == persona.persona_id,
            record.get("arm") == arm,
            int(record.get("max_items", -1)) == max_items,
            record.get("panel_sha256") == panel_sha,
            int(record.get("prompt_revision", -1)) == prompt_revision,
            len(record.get("events") or ()) == int(record.get("actual_administered_count", -1)),
        )
    )


def _resume_events(
    record: Mapping[str, Any] | None,
    *,
    provider: str,
    model: str,
    persona: Persona,
    arm: str,
    max_items: int,
    panel_sha: str,
    prompt_revision: int,
) -> list[dict[str, Any]]:
    if not record:
        return []
    if not all(
        (
            record.get("simulated") is True,
            record.get("run_id") == FROZEN_RUN_ID,
            record.get("provider") == provider,
            record.get("model_id") == model,
            record.get("persona_id") == persona.persona_id,
            record.get("arm") == arm,
            int(record.get("max_items", -1)) == max_items,
            record.get("panel_sha256") == panel_sha,
            int(record.get("prompt_revision", -1)) == prompt_revision,
        )
    ):
        return []
    events = list(record.get("events") or ())
    if any(
        not isinstance(event, Mapping)
        or event.get("model_id") != model
        or int(event.get("position", -1)) != index
        for index, event in enumerate(events, start=1)
    ):
        return []
    return [dict(event) for event in events]
