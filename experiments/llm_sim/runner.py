"""Provider-local execution, resume, and production-engine integration for S2."""

from __future__ import annotations

import json
import hashlib
import re
import sys
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
from .config import FROZEN_RUN_ID, LLMSimConfig, load_frozen_config
from .panel import (
    annotation_map_hash,
    derive_manipulation_panel,
    freeze_manipulation_panel,
    load_frozen_panel,
    normalize_annotation_map,
)
from .personas import build_personas
from .provenance import (
    analysis_plan_is_ancestor,
    collect_code_provenance,
    collect_official_input_provenance,
)
from .store import SimulationStore
from .transport import (
    HTTPProviderTransport,
    ProviderConfigurationError,
    ProviderHTTPError,
    ProviderNetworkError,
    ProviderProtocolError,
    ProviderTransport,
    load_live_environment,
    model_from_environment,
    provider_spec,
)


FIXED_LADDER = (0.25, 0.50, 0.75, 1.00)
CALIBRATION_ITEMS_PER_PERSONA = 4


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
        study_seed: int | None = None,
        repo_root: str | Path | None = None,
        policy_factory: Callable[[], ProviderCallPolicy] | None = None,
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
        self.study_seed = int(
            self.config.study_seed if study_seed is None else study_seed
        )
        self.repo_root = Path(repo_root).resolve(strict=False) if repo_root else Path(__file__).resolve().parents[2]
        self.code_root = Path(__file__).resolve().parents[2]
        persona_build_kwargs = {
            "pair_count": self.config.pair_count,
            "eligible_nodes": set(catalog.open_nodes()),
            "seed_derivation_version": self.config.persona_seed_derivation_version,
        }
        if personas is None:
            generated = tuple(
                build_personas(kg, seed=self.study_seed, **persona_build_kwargs)
            )
            self._canonical_personas = (
                generated
                if self.study_seed == self.config.study_seed
                else tuple(
                    build_personas(
                        kg,
                        seed=self.config.study_seed,
                        **persona_build_kwargs,
                    )
                )
            )
            self.personas = generated
        else:
            self.personas = tuple(
                value if isinstance(value, Persona) else Persona.from_mapping(value)
                for value in personas
            )
            try:
                self._canonical_personas = tuple(
                    build_personas(
                        kg,
                        seed=self.config.study_seed,
                        **persona_build_kwargs,
                    )
                )
            except (TypeError, ValueError):
                self._canonical_personas = ()
        if not self.personas:
            raise ValueError("at least one simulated persona is required")
        self.panel_path = Path(panel_path or (store.root / "manipulation_panel.json")).resolve(strict=False)
        if not (
            self.panel_path == store.root
            or self.panel_path.is_relative_to(store.root)
        ):
            raise ValueError("manipulation panel must be inside the simulation store")
        self.policy_factory = policy_factory or self._policy_from_config
        self._policies: dict[str, ProviderCallPolicy] = {}
        self._panel: dict[str, Any] | None = None
        self._preparation_manifest: dict[str, Any] | None = None
        self._provider_run_started: dict[tuple[str, int], str] = {}

    def _policy_from_config(self) -> ProviderCallPolicy:
        return ProviderCallPolicy(**self.config.provider_policy)

    def _provider_run_started_at_utc(
        self, provider: str, prompt_revision: int
    ) -> str:
        key = (provider, int(prompt_revision))
        if key in self._provider_run_started:
            return self._provider_run_started[key]
        candidates: set[str] = set()
        manifest = self.store.read_json(
            _provider_manifest_relative(provider, prompt_revision)
        )
        if manifest:
            value = manifest.get("run_started_at_utc") or manifest.get(
                "created_at_utc"
            )
            if value:
                candidates.add(str(value))
        root = self.store.root / "attempts"
        for absolute in root.rglob("*.json") if root.is_dir() else ():
            record = self.store.read_json(absolute.relative_to(self.store.root))
            if (
                record
                and record.get("provider") == provider
                and int(record.get("prompt_revision", -1)) == prompt_revision
                and record.get("run_started_at_utc")
            ):
                candidates.add(str(record["run_started_at_utc"]))
        if len(candidates) > 1:
            raise ValueError("provider run_started_at_utc provenance conflicts")
        started = (
            next(iter(candidates))
            if candidates
            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        try:
            parsed = datetime.fromisoformat(started.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("provider run_started_at_utc is invalid") from exc
        if not started.endswith("Z") or parsed.tzinfo is None:
            raise ValueError("provider run_started_at_utc must be UTC")
        self._provider_run_started[key] = started
        return started

    def prepare(self) -> dict[str, Any]:
        """Freeze the study panel and provenance without resolving a provider."""

        self._assert_frozen_singletons_or_fresh()
        provenance_before = collect_code_provenance(self.code_root)
        official_inputs_before = collect_official_input_provenance(
            catalog=self.catalog,
            kg=self.kg,
            personas=self.personas,
        )
        before = protected_filesystem_fingerprint(self.repo_root)
        panel = self._ensure_panel()
        persona_panel = self._ensure_persona_panel_snapshot(panel)
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
        plan_is_ancestor = analysis_plan_is_ancestor(
            self.code_root,
            self.config.analysis_plan_commit,
            provenance_before["git_head"],
        )
        if not plan_is_ancestor:
            raise RuntimeError("frozen analysis-plan commit is not an ancestor of S2 HEAD")
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
            "working_code_sha256": provenance_before["working_code_sha256"],
            "head_code_sha256": provenance_before["head_code_sha256"],
            "code_matches_head": provenance_before["code_matches_head"],
            "code_files": provenance_before["code_files"],
            "analysis_plan_commit": self.config.analysis_plan_commit,
            "analysis_plan_is_ancestor": plan_is_ancestor,
            "persona_seed_derivation_version": (
                self.config.persona_seed_derivation_version
            ),
            "prompt_version": self.config.prompt_version,
            "official_input_sha256": official_inputs_before["official_input_sha256"],
            "official_inputs": official_inputs_before["official_inputs"],
            "panel_sha256": panel["panel_sha256"],
            "persona_panel_path": "persona_panel.json",
            "persona_panel_sha256": persona_panel["persona_panel_sha256"],
            "frozen_pre_observation_utc": self.config.frozen_pre_observation_utc,
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
                "working_code_sha256",
                "head_code_sha256",
                "code_matches_head",
                "code_files",
                "analysis_plan_commit",
                "analysis_plan_is_ancestor",
                "persona_seed_derivation_version",
                "prompt_version",
                "official_input_sha256",
                "official_inputs",
                "panel_sha256",
                "persona_panel_sha256",
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
        max_items: int | None = None,
        arms: Sequence[str] | None = None,
        resume: bool = True,
        timeout_seconds: float = 30.0,
        prompt_revision: int = 0,
    ) -> dict[str, Any]:
        """Run one provider after the pre-outcome panel is durably frozen."""

        spec = provider_spec(provider)
        provider = spec.name
        max_items = self.config.max_items if max_items is None else int(max_items)
        if max_items < 1:
            raise ValueError("max_items must be positive")
        arms = tuple(str(arm) for arm in (self.config.arms if arms is None else arms))
        if not arms or set(arms) - {"A", "B"}:
            raise ValueError("S2 arms must be a non-empty subset of A and B")
        if not 0 <= int(prompt_revision) <= self.config.maximum_prompt_rewrites:
            raise ValueError("prompt_revision exceeds the frozen rewrite allowance")
        prompt_revision = int(prompt_revision)

        # Hard ordering gate: no transport object or dotenv lookup is resolved
        # until this file is present and its hash has been verified.
        self._assert_frozen_singletons_or_fresh()
        if not self.store.exists("preparation_manifest.json"):
            self.prepare()
        panel = self._ensure_panel()
        panel_sha = str(panel["panel_sha256"])
        self._assert_frozen_preparation(panel)
        self._preflight_existing_finals(
            provider=provider,
            arms=arms,
            prompt_revision=prompt_revision,
            resume=resume,
        )
        self._verify_existing_provider_artifacts(provider, prompt_revision)
        official_ready_checked = False
        if self.transport is None:
            self._assert_official_live_ready()
            self._assert_bytecode_disabled_for_http()
            self._assert_canonical_http_design(max_items=max_items, arms=arms)
            official_ready_checked = True
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
        self._assert_prompt_revision_allowed(
            provider=provider,
            requested_model=requested_model,
            max_items=max_items,
            arms=arms,
            prompt_revision=prompt_revision,
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
            if isinstance(transport, HTTPProviderTransport):
                self._assert_bytecode_disabled_for_http()
                if not official_ready_checked:
                    self._assert_official_live_ready()
                self._assert_canonical_http_design(max_items=max_items, arms=arms)
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
            if prompt_revision == 0:
                self._write_calibration_decision(
                    provider=provider,
                    requested_model=requested_model,
                    panel_sha=panel_sha,
                    max_items=max_items,
                    arms=arms,
                    status=_provider_run_status(
                        "complete", provider_eligibility
                    ),
                )
        except Exception as exc:
            accounting["requests"] = policy.attempts - attempts_before
            accounting["retries"] = policy.retries - retries_before
            accounting["circuit_open"] = policy.clock() < policy.opened_until
            accounting["consecutive_failures"] = policy.consecutive_failures
            accounting["failed"] += 1
            accounting = self._persisted_accounting(
                provider=provider,
                arms=arms,
                prompt_revision=prompt_revision,
                accounting=accounting,
            )
            accounting["failed"] = max(1, int(accounting.get("failed") or 0))
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
                failure_category=_failure_category(exc),
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
                    config_sha256=self.config.sha256,
                    persona_panel_sha256=(self._preparation_manifest or {}).get(
                        "persona_panel_sha256"
                    ),
                    study_seed=self.study_seed,
                    analysis_plan_commit=self.config.analysis_plan_commit,
                    prompt_version=self.config.prompt_version,
                ):
                    journeys.append(existing)
                    accounting["skipped"] += 1
                    if existing.get("status") == "complete":
                        accounting["completed_by_arm"][arm] += 1
                    else:
                        accounting["failed"] += 1
                    continue
                if existing is not None:
                    raise FileExistsError(
                        "immutable completed journey artifact does not match the frozen run"
                    )
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
                except Exception as exc:
                    accounting["requests"] = policy.attempts - attempts_before
                    accounting["retries"] = policy.retries - retries_before
                    accounting["circuit_open"] = policy.clock() < policy.opened_until
                    accounting["consecutive_failures"] = policy.consecutive_failures
                    accounting["failed"] += 1
                    accounting = self._persisted_accounting(
                        provider=provider,
                        arms=arms,
                        prompt_revision=prompt_revision,
                        accounting=accounting,
                    )
                    accounting["failed"] = max(
                        1, int(accounting.get("failed") or 0)
                    )
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
                        failure_category=_failure_category(exc),
                    )
                    raise

        accounting["circuit_open"] = policy.clock() < policy.opened_until
        accounting["consecutive_failures"] = policy.consecutive_failures
        accounting = self._persisted_accounting(
            provider=provider,
            arms=arms,
            prompt_revision=prompt_revision,
            accounting=accounting,
        )
        observations_started = int(accounting.get("requests") or 0) > 0
        after = protected_filesystem_fingerprint(self.repo_root)
        if before != after:
            raise RuntimeError("protected local_store/study state changed during S2 run")
        journey_status = "complete" if accounting["failed"] == 0 else "partial"
        status = _provider_run_status(journey_status, provider_eligibility)
        if accounting.get("model_drift_detected") is True:
            status = "excluded_model_drift"
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
            if not personas:
                continue
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
                calibration_items = list(annotation.get("calibration_items", ()))
                existing = self.store.read_json(relative)
                if _complete_calibration(
                    existing,
                    provider=provider,
                    model=model,
                    persona=persona,
                    panel_sha=str(self._panel["panel_sha256"]),
                    prompt_revision=prompt_revision,
                    prompt_version=self.config.prompt_version,
                    calibration_items=calibration_items,
                ):
                    rows = [dict(event) for event in existing.get("events", ())]
                    strength_attempts.extend(rows)
                    if existing.get("status") == "structural_failure":
                        calibration_ready = False
                    continue
                if existing is not None:
                    raise FileExistsError(
                        "immutable completed calibration artifact does not match the frozen run"
                    )
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
                    prompt_version=self.config.prompt_version,
                    calibration_items=annotation.get("calibration_items", ()),
                )
                missing_item_id: str | None = None
                for item_row in calibration_items[len(events) :]:
                    item = _catalog_item_by_id(self.catalog, item_row["item_id"])
                    if item is None:
                        calibration_ready = False
                        missing_item_id = str(item_row["item_id"])
                        break
                    raw = self._provider_call(
                        provider=provider,
                        requested_model=model,
                        persona=persona,
                        phase="calibration",
                        position=len(events) + 1,
                        item_id=str(item_row["item_id"]),
                        arm=None,
                        prompt_revision=prompt_revision,
                        messages=_messages(
                            persona,
                            item,
                            phase="calibration",
                            prompt_revision=prompt_revision,
                            prompt_version=self.config.prompt_version,
                        ),
                        timeout_seconds=timeout_seconds,
                        transport=transport,
                        policy=policy,
                    )
                    responses += 1
                    returned_model = str(raw.get("model_returned") or "").strip()
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
                        "prompt_version": self.config.prompt_version,
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
                        "prompt_version": self.config.prompt_version,
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
                    "status": (
                        "structural_failure" if missing_item_id else "complete"
                    ),
                    "strength": persona.strength,
                    "events": events,
                    "panel_sha256": self._panel["panel_sha256"],
                    "prompt_version": self.config.prompt_version,
                    "prompt_revision": prompt_revision,
                }
                if missing_item_id:
                    record.update(
                        {
                            "failure_category": "catalog_item_missing",
                            "terminal_reason": "frozen_calibration_item_unavailable",
                            "expected_item_count": CALIBRATION_ITEMS_PER_PERSONA,
                            "actual_administered_count": len(events),
                            "missing_item_id": missing_item_id,
                        }
                    )
                self.store.write_json(relative, record, immutable=True)
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
                config=self.config,
                prompt_revision=prompt_revision,
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
                immutable=True,
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
            prompt_version=self.config.prompt_version,
        )
        node = mastery.NodeBelief(mastery.UNIFORM.copy())
        for event in events:
            if event.get("update_applied") is True:
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
            messages = _messages(
                persona,
                item,
                prompt_revision=prompt_revision,
                prompt_version=self.config.prompt_version,
            )
            raw = self._provider_call(
                provider=provider,
                requested_model=model,
                persona=persona,
                phase="journey",
                position=position,
                item_id=str(getattr(item, "item_id", "")),
                arm=arm,
                prompt_revision=prompt_revision,
                messages=messages,
                timeout_seconds=timeout_seconds,
                transport=transport,
                policy=policy,
            )
            request_count += 1
            returned_model = str(raw.get("model_returned") or "").strip()
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
                "prompt_version": self.config.prompt_version,
                "prompt_revision": prompt_revision,
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
            immutable=True,
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
            "persona_panel_sha256": (self._preparation_manifest or {}).get(
                "persona_panel_sha256"
            ),
            "study_seed": self.study_seed,
            "analysis_plan_commit": self.config.analysis_plan_commit,
            "prompt_version": self.config.prompt_version,
            "prompt_revision": prompt_revision,
        }

    def _ensure_panel(self) -> dict[str, Any]:
        if self._panel is not None:
            return self._panel
        if self.panel_path.is_file():
            panel = load_frozen_panel(self.panel_path)
            if (
                self.annotation_map is None
                and panel.get("annotation_map_sha256") is not None
            ):
                snapshot = self.store.read_json("annotation_map_snapshot.json")
                if not snapshot:
                    raise ValueError("frozen annotation map snapshot is missing")
                normalized_snapshot = normalize_annotation_map(
                    snapshot.get("annotation_map") or {}
                )
                if any(
                    (
                        snapshot.get("annotation_map_sha256")
                        != panel.get("annotation_map_sha256"),
                        snapshot.get("panel_sha256") != panel.get("panel_sha256"),
                        annotation_map_hash(normalized_snapshot)
                        != panel.get("annotation_map_sha256"),
                    )
                ):
                    raise ValueError("frozen annotation map snapshot is invalid")
                self.annotation_map = normalized_snapshot
                self.annotation_map_source = str(
                    panel.get("annotation_map_source")
                    or snapshot.get("source_path")
                    or "inline_mapping"
                )
            expected = derive_manipulation_panel(
                personas=self.personas,
                catalog=self.catalog,
                annotation_map=self.annotation_map,
                annotation_map_source=self.annotation_map_source,
                study_seed=self.study_seed,
            )
            if panel != expected:
                raise ValueError(
                    "frozen manipulation panel does not match the mechanically rederived manipulation panel"
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
        self._ensure_persona_panel_snapshot(panel)
        self._panel = panel
        return panel

    def _assert_frozen_singletons_or_fresh(self) -> None:
        singleton_paths = {
            "preparation_manifest.json": self.store.root / "preparation_manifest.json",
            self.panel_path.name: self.panel_path,
            "persona_panel.json": self.store.root / "persona_panel.json",
        }
        present = {name for name, path in singleton_paths.items() if path.is_file()}
        observation_roots = (
            "providers",
            "calibration_decisions",
            "calibration",
            "journeys",
            "excluded_responses",
            "attempts",
        )
        has_observations = any(
            any(root.rglob("*.json"))
            for name in observation_roots
            if (root := self.store.root / name).is_dir()
        )
        if not present and not has_observations:
            return
        missing = sorted(set(singleton_paths) - present)
        if missing:
            raise FileNotFoundError(
                "frozen provenance singleton missing: " + ", ".join(missing)
            )
        preparation = self.store.read_json("preparation_manifest.json") or {}
        snapshot_path = self.store.root / "annotation_map_snapshot.json"
        snapshot_required = preparation.get("annotation_map_sha256") is not None
        if snapshot_required != snapshot_path.is_file():
            state = "missing" if snapshot_required else "unexpected"
            raise FileNotFoundError(
                f"frozen annotation-map singleton {state}: annotation_map_snapshot.json"
            )

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

    def _ensure_persona_panel_snapshot(
        self,
        panel: Mapping[str, Any],
    ) -> dict[str, Any]:
        personas = [
            persona.to_dict()
            for persona in sorted(self.personas, key=lambda row: row.persona_id)
        ]
        personas_sha = _canonical_sha256(personas)
        canonical_personas = [
            persona.to_dict()
            for persona in sorted(
                self._canonical_personas, key=lambda row: row.persona_id
            )
        ]
        canonical_personas_sha = (
            _canonical_sha256(canonical_personas) if canonical_personas else None
        )
        canonical_match = canonical_personas_sha == personas_sha
        core = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": "llm-sim-study:persona-panel",
            "provider": "study_design",
            "model_id": "no-provider-observation",
            "record_type": "llm_sim_persona_panel",
            "frozen": True,
            "observation_started": False,
            "frozen_pre_observation_utc": self.config.frozen_pre_observation_utc,
            "persona_seed_derivation_version": (
                self.config.persona_seed_derivation_version
            ),
            "prompt_version": self.config.prompt_version,
            "personas_sha256": personas_sha,
            "canonical_personas_sha256": canonical_personas_sha,
            "canonical_match": canonical_match,
            "personas": personas,
            "manipulation_panel_sha256": panel["panel_sha256"],
        }
        record = {**core, "persona_panel_sha256": _canonical_sha256(core)}
        existing = self.store.read_json("persona_panel.json")
        if existing is None:
            self.store.write_json(
                "persona_panel.json",
                record,
                immutable=True,
            )
            existing = record
        supplied_sha = str(existing.get("persona_panel_sha256") or "")
        existing_core = {
            key: value
            for key, value in existing.items()
            if key != "persona_panel_sha256"
        }
        if any(
            (
                supplied_sha != _canonical_sha256(existing_core),
                existing != record,
                panel.get("personas_sha256") != personas_sha,
            )
        ):
            raise ValueError("frozen persona panel is missing, changed, or invalid")
        return existing

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
        expected = {key: preparation.get(key) for key in current}
        if current != expected:
            raise RuntimeError("code or git HEAD changed after S2 preparation")
        expected_official_inputs = {
            "official_input_sha256": preparation.get("official_input_sha256"),
            "official_inputs": preparation.get("official_inputs"),
        }
        if current_official_inputs != expected_official_inputs:
            raise RuntimeError("official input changed after S2 preparation")
        persona_panel = self._ensure_persona_panel_snapshot(panel)
        plan_is_ancestor = analysis_plan_is_ancestor(
            self.code_root,
            self.config.analysis_plan_commit,
            current["git_head"],
        )
        if any(
            (
                preparation.get("panel_sha256") != panel.get("panel_sha256"),
                preparation.get("run_id") != self.config.run_id,
                preparation.get("config_sha256") != self.config.sha256,
                preparation.get("config_sha256") != current_config.sha256,
                int(preparation.get("study_seed", -1)) != self.study_seed,
                preparation.get("persona_panel_path") != "persona_panel.json",
                preparation.get("persona_panel_sha256")
                != persona_panel.get("persona_panel_sha256"),
                preparation.get("analysis_plan_commit")
                != self.config.analysis_plan_commit,
                preparation.get("analysis_plan_is_ancestor") is not True,
                not plan_is_ancestor,
            )
        ):
            raise RuntimeError("S2 frozen preparation does not match the live run")
        self._preparation_manifest = preparation

    def _assert_official_live_ready(self) -> None:
        preparation = self._preparation_manifest or {}
        if preparation.get("code_matches_head") is not True:
            raise RuntimeError(
                "S2 code dependencies must be byte-equal to recorded HEAD before live transport"
            )
        if preparation.get("analysis_plan_is_ancestor") is not True:
            raise RuntimeError(
                "frozen analysis-plan commit must be an ancestor before live transport"
            )

    @staticmethod
    def _assert_bytecode_disabled_for_http() -> None:
        if not sys.dont_write_bytecode:
            raise RuntimeError(
                "official HTTP transport requires Python -B or PYTHONDONTWRITEBYTECODE=1 from process start"
            )

    def _assert_canonical_http_design(
        self, *, max_items: int, arms: Sequence[str]
    ) -> None:
        if int(max_items) != self.config.max_items:
            raise ValueError(
                f"official HTTP max_items must equal frozen config {self.config.max_items}"
            )
        if tuple(arms) != self.config.arms:
            raise ValueError("official HTTP arms must be the frozen A+B design")

    def _preflight_existing_finals(
        self,
        *,
        provider: str,
        arms: Sequence[str],
        prompt_revision: int,
        resume: bool,
    ) -> None:
        if resume:
            return
        artifacts: list[Path] = []
        for persona in self.personas:
            artifacts.extend(
                (
                    self.store.calibration_relative_path(
                        provider, persona.persona_id, prompt_revision
                    ),
                    self.store.calibration_checkpoint_relative_path(
                        provider, persona.persona_id, prompt_revision
                    ),
                )
            )
            for arm in arms:
                artifacts.extend(
                    (
                        self.store.journey_relative_path(
                            provider, persona.persona_id, arm, prompt_revision
                        ),
                        self.store.checkpoint_relative_path(
                            provider, persona.persona_id, arm, prompt_revision
                        ),
                    )
                )
        artifacts.extend(
            (
                _provider_manifest_relative(provider, prompt_revision),
                _calibration_decision_relative(provider),
            )
        )
        for directory in ("excluded_responses", "attempts"):
            root = self.store.root / directory
            for absolute in root.rglob("*.json") if root.is_dir() else ():
                relative = absolute.relative_to(self.store.root)
                record = self.store.read_json(relative)
                if (
                    record is not None
                    and record.get("provider") == provider
                    and int(record.get("prompt_revision", 0)) == prompt_revision
                ):
                    artifacts.append(relative)
        existing = [path for path in artifacts if self.store.exists(path)]
        if existing:
            raise FileExistsError(
                "--no-resume refuses every existing provider observation artifact"
            )

    def _verify_existing_provider_artifacts(
        self,
        provider: str,
        prompt_revision: int,
    ) -> None:
        manifest = self.store.read_json(
            _provider_manifest_relative(provider, prompt_revision)
        )
        if manifest is None:
            return
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("provider manifest artifact list is missing")
        model = str(manifest.get("model_id") or "")
        arms = tuple(str(value) for value in (manifest.get("arms") or ()))
        max_items = int(manifest.get("max_items", -1))
        run_started_at_utc = str(manifest.get("run_started_at_utc") or "")
        preparation = self._preparation_manifest or {}
        if not all(
            (
                manifest.get("simulated") is True,
                manifest.get("run_id") == self.config.run_id,
                manifest.get("record_type") == "llm_sim_provider_manifest",
                manifest.get("provider") == provider,
                bool(model),
                int(manifest.get("prompt_revision", -1)) == prompt_revision,
                manifest.get("panel_sha256") == (self._panel or {}).get("panel_sha256"),
                manifest.get("persona_panel_sha256")
                == preparation.get("persona_panel_sha256"),
                manifest.get("config_sha256") == self.config.sha256,
                int(manifest.get("study_seed", -1)) == self.study_seed,
                bool(arms),
                not (set(arms) - {"A", "B"}),
                max_items > 0,
                run_started_at_utc.endswith("Z"),
            )
        ):
            raise ValueError("provider manifest frozen provenance is invalid")
        personas = {persona.persona_id: persona for persona in self.personas}
        annotations = {
            str(row.get("persona_id")): row
            for row in ((self._panel or {}).get("annotations") or ())
            if isinstance(row, Mapping)
        }
        seen_paths: set[str] = set()
        for row in artifacts:
            if not isinstance(row, Mapping):
                raise ValueError("provider manifest artifact row is invalid")
            path = str(row.get("path") or "")
            if (
                not path
                or path in seen_paths
                or self.store.file_sha256(path) != row.get("sha256")
            ):
                raise ValueError("provider manifest artifact hash mismatch")
            seen_paths.add(path)
            record = self.store.read_json(path)
            if record is None or any(
                (
                    row.get("record_type") != record.get("record_type"),
                    row.get("persona_id") != record.get("persona_id"),
                    row.get("status") != record.get("status"),
                    row.get("arm") != record.get("arm"),
                )
            ):
                raise ValueError("provider manifest artifact metadata is invalid")
            persona = personas.get(str(record.get("persona_id") or ""))
            record_type = record.get("record_type")
            try:
                if record_type == "llm_sim_calibration":
                    if persona is None:
                        raise ValueError("unknown calibration persona")
                    expected_path = (
                        self.store.calibration_checkpoint_relative_path(
                            provider, persona.persona_id, prompt_revision
                        )
                        if record.get("status") == "in_progress"
                        else self.store.calibration_relative_path(
                            provider, persona.persona_id, prompt_revision
                        )
                    )
                    if path != expected_path.as_posix():
                        raise ValueError("unexpected calibration path")
                    calibration_items = list(
                        (annotations.get(persona.persona_id) or {}).get(
                            "calibration_items", ()
                        )
                    )
                    if record.get("status") in {"complete", "structural_failure"}:
                        valid = _complete_calibration(
                            record,
                            provider=provider,
                            model=model,
                            persona=persona,
                            panel_sha=str((self._panel or {}).get("panel_sha256") or ""),
                            prompt_revision=prompt_revision,
                            prompt_version=self.config.prompt_version,
                            calibration_items=calibration_items,
                        )
                        if not valid:
                            raise ValueError("invalid terminal calibration")
                    elif record.get("status") == "in_progress":
                        _resume_calibration_events(
                            record,
                            provider=provider,
                            model=model,
                            persona=persona,
                            panel_sha=str((self._panel or {}).get("panel_sha256") or ""),
                            prompt_revision=prompt_revision,
                            prompt_version=self.config.prompt_version,
                            calibration_items=calibration_items,
                        )
                    else:
                        raise ValueError("invalid calibration status")
                elif record_type == "llm_sim_journey":
                    if persona is None:
                        raise ValueError("unknown journey persona")
                    arm = str(record.get("arm") or "")
                    if arm not in arms:
                        raise ValueError("unknown journey arm")
                    expected_path = (
                        self.store.checkpoint_relative_path(
                            provider, persona.persona_id, arm, prompt_revision
                        )
                        if record.get("status") == "in_progress"
                        else self.store.journey_relative_path(
                            provider, persona.persona_id, arm, prompt_revision
                        )
                    )
                    if path != expected_path.as_posix():
                        raise ValueError("unexpected journey path")
                    if record.get("status") in {"complete", "incomplete"}:
                        valid = _complete_existing(
                            record,
                            provider=provider,
                            requested_model=model,
                            persona=persona,
                            arm=arm,
                            max_items=max_items,
                            panel_sha=str((self._panel or {}).get("panel_sha256") or ""),
                            prompt_revision=prompt_revision,
                            config_sha256=self.config.sha256,
                            persona_panel_sha256=preparation.get("persona_panel_sha256"),
                            study_seed=self.study_seed,
                            analysis_plan_commit=self.config.analysis_plan_commit,
                            prompt_version=self.config.prompt_version,
                        )
                        if not valid:
                            raise ValueError("invalid final journey")
                    elif record.get("status") == "in_progress":
                        _resume_events(
                            record,
                            provider=provider,
                            model=model,
                            persona=persona,
                            arm=arm,
                            max_items=max_items,
                            panel_sha=str((self._panel or {}).get("panel_sha256") or ""),
                            prompt_revision=prompt_revision,
                            prompt_version=self.config.prompt_version,
                        )
                    else:
                        raise ValueError("invalid journey status")
                elif record_type == "llm_sim_provider_attempt":
                    if persona is None or not _provider_attempt_is_valid(
                        record,
                        provider=provider,
                        model=model,
                        persona=persona,
                        panel_sha=str((self._panel or {}).get("panel_sha256") or ""),
                        prompt_revision=prompt_revision,
                        prompt_version=self.config.prompt_version,
                        run_started_at_utc=run_started_at_utc,
                    ):
                        raise ValueError("invalid provider attempt")
                    expected_path = self.store.attempt_relative_path(
                        provider,
                        persona.persona_id,
                        phase=str(record.get("phase") or ""),
                        position=int(record.get("position", -1)),
                        attempt_number=int(record.get("attempt_number", -1)),
                        prompt_revision=prompt_revision,
                        arm=record.get("arm"),
                    )
                    if path != expected_path.as_posix():
                        raise ValueError("unexpected attempt path")
                elif record_type == "llm_sim_excluded_response_accounting":
                    if persona is None or not all(
                        (
                            record.get("simulated") is True,
                            record.get("run_id") == self.config.run_id,
                            record.get("provider") == provider,
                            record.get("persona_id") == persona.persona_id,
                            record.get("failure_category") == "model_id_drift",
                            record.get("schema_version")
                            == "yher.llm_sim.model_drift_exclusion.v1",
                            record.get("exclusion_type") == "model_id_drift",
                            record.get("requested_model_id") == model,
                            bool(str(record.get("returned_model_id") or "")),
                            int(record.get("source_attempt_number", -1)) >= 1,
                            record.get("run_started_at_utc")
                            == run_started_at_utc,
                            int(record.get("prompt_revision", -1)) == prompt_revision,
                        )
                    ):
                        raise ValueError("invalid excluded response")
                    expected_path = self.store.excluded_response_relative_path(
                        provider,
                        persona.persona_id,
                        phase=str(record.get("phase") or ""),
                        position=int(record.get("position", -1)),
                        prompt_revision=prompt_revision,
                        arm=record.get("arm"),
                        sequence=int(record.get("source_attempt_number", -1)),
                    )
                    if path != expected_path.as_posix():
                        raise ValueError("unexpected excluded-response path")
                else:
                    raise ValueError("unexpected artifact record type")
            except (TypeError, ValueError) as exc:
                label = (
                    "calibration"
                    if record_type == "llm_sim_calibration"
                    else "journey"
                    if record_type == "llm_sim_journey"
                    else "attempt"
                )
                raise ValueError(
                    f"provider manifest artifact {label} is invalid"
                ) from exc
        if _canonical_sha256(artifacts) != manifest.get("artifact_aggregate_sha256"):
            raise ValueError("provider manifest aggregate artifact hash mismatch")
        if artifacts != self._provider_artifacts(provider, arms, prompt_revision):
            raise ValueError("provider manifest artifact set is incomplete or stale")

    def _assert_prompt_revision_allowed(
        self,
        *,
        provider: str,
        requested_model: str,
        max_items: int,
        arms: Sequence[str],
        prompt_revision: int,
    ) -> None:
        if prompt_revision == 0:
            return
        decision = self._validated_calibration_decision(provider)
        preparation = self._preparation_manifest or {}
        required = {
            "status": "calibration_rewrite_required",
            "prompt_revision": 0,
            "provider": provider,
            "model_id": requested_model,
            "panel_sha256": (self._panel or {}).get("panel_sha256"),
            "persona_panel_sha256": preparation.get("persona_panel_sha256"),
            "config_sha256": self.config.sha256,
            "study_seed": self.study_seed,
            "arms": list(arms),
            "max_items": max_items,
        }
        if not decision or any(
            decision.get(key) != value for key, value in required.items()
        ):
            raise ValueError(
                "prompt revision 1 requires a matching immutable v0 calibration_rewrite_required decision"
            )

    def _calibration_artifacts(
        self,
        provider: str,
        prompt_revision: int = 0,
    ) -> list[dict[str, Any]]:
        rows = []
        for persona in sorted(self.personas, key=lambda row: row.persona_id):
            relative = self.store.calibration_relative_path(
                provider, persona.persona_id, prompt_revision
            )
            record = self.store.read_json(relative)
            if record is None:
                continue
            rows.append(
                {
                    "path": relative.as_posix(),
                    "sha256": self.store.file_sha256(relative),
                    "persona_id": persona.persona_id,
                    "status": str(record.get("status") or ""),
                }
            )
        return rows

    def _provider_call(
        self,
        *,
        provider: str,
        requested_model: str,
        persona: Persona,
        phase: str,
        position: int,
        item_id: str,
        arm: str | None,
        prompt_revision: int,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        transport: ProviderTransport,
        policy: ProviderCallPolicy,
    ) -> Mapping[str, Any]:
        retry_number = -1

        def operation() -> Mapping[str, Any]:
            nonlocal retry_number
            retry_number += 1
            try:
                raw = transport.complete(
                    provider=provider,
                    model=requested_model,
                    messages=messages,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                self._persist_provider_attempt(
                    provider=provider,
                    requested_model=requested_model,
                    returned_model=None,
                    persona=persona,
                    phase=phase,
                    position=position,
                    item_id=item_id,
                    arm=arm,
                    prompt_revision=prompt_revision,
                    retry_number=retry_number,
                    raw=None,
                    status="failed",
                    failure_category=_failure_category(exc),
                )
                raise
            if not isinstance(raw, Mapping):
                self._persist_provider_attempt(
                    provider=provider,
                    requested_model=requested_model,
                    returned_model=None,
                    persona=persona,
                    phase=phase,
                    position=position,
                    item_id=item_id,
                    arm=arm,
                    prompt_revision=prompt_revision,
                    retry_number=retry_number,
                    raw=None,
                    status="protocol_failure",
                    failure_category="protocol",
                )
                raise ProviderProtocolError()
            returned_model = str(raw.get("model_returned") or "").strip()
            if not returned_model:
                self._persist_provider_attempt(
                    provider=provider,
                    requested_model=requested_model,
                    returned_model=None,
                    persona=persona,
                    phase=phase,
                    position=position,
                    item_id=item_id,
                    arm=arm,
                    prompt_revision=prompt_revision,
                    retry_number=retry_number,
                    raw=raw,
                    status="protocol_failure",
                    failure_category="protocol",
                )
                raise ProviderProtocolError()
            if returned_model != requested_model:
                drift_attempt = self._persist_provider_attempt(
                    provider=provider,
                    requested_model=requested_model,
                    returned_model=returned_model,
                    persona=persona,
                    phase=phase,
                    position=position,
                    item_id=item_id,
                    arm=arm,
                    prompt_revision=prompt_revision,
                    retry_number=retry_number,
                    raw=raw,
                    status="model_drift",
                    failure_category="model_id_drift",
                )
                self._persist_model_drift_response(
                    provider=provider,
                    requested_model=requested_model,
                    returned_model=returned_model,
                    persona=persona,
                    phase=phase,
                    position=position,
                    item_id=item_id,
                    raw=raw,
                    prompt_revision=prompt_revision,
                    arm=arm,
                    source_attempt_number=int(drift_attempt["attempt_number"]),
                )
                raise ModelDriftError(
                    f"provider {provider} returned a model id different from the frozen request"
                )
            self._persist_provider_attempt(
                provider=provider,
                requested_model=requested_model,
                returned_model=returned_model,
                persona=persona,
                phase=phase,
                position=position,
                item_id=item_id,
                arm=arm,
                prompt_revision=prompt_revision,
                retry_number=retry_number,
                raw=raw,
                status="response",
                failure_category=None,
            )
            return raw

        return policy.call(operation)

    def _persist_provider_attempt(
        self,
        *,
        provider: str,
        requested_model: str,
        returned_model: str | None,
        persona: Persona,
        phase: str,
        position: int,
        item_id: str,
        arm: str | None,
        prompt_revision: int,
        retry_number: int,
        raw: Mapping[str, Any] | None,
        status: str,
        failure_category: str | None,
    ) -> dict[str, Any]:
        attempt_number = 1
        while self.store.exists(
            self.store.attempt_relative_path(
                provider,
                persona.persona_id,
                phase=phase,
                position=position,
                attempt_number=attempt_number,
                prompt_revision=prompt_revision,
                arm=arm,
            )
        ):
            attempt_number += 1
        usage_raw = (
            raw.get("usage")
            if isinstance(raw, Mapping) and isinstance(raw.get("usage"), Mapping)
            else {}
        )
        usage = {
            "input_tokens": max(0, int(usage_raw.get("input_tokens") or 0)),
            "output_tokens": max(0, int(usage_raw.get("output_tokens") or 0)),
        }
        record = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": persona.persona_id,
            "provider": provider,
            "model_id": (
                returned_model
                if returned_model
                else (
                    "missing-provider-model-id"
                    if status == "protocol_failure"
                    else requested_model
                )
            ),
            "record_type": "llm_sim_provider_attempt",
            "schema_version": "yher.llm_sim.provider_attempt.v1",
            "status": status,
            "failure_category": failure_category,
            "exclusion_type": (
                "model_id_drift" if failure_category == "model_id_drift" else None
            ),
            "phase": phase,
            "arm": arm,
            "position": int(position),
            "item_id": item_id,
            "attempt_number": attempt_number,
            "retry_number": int(retry_number),
            "requested_model_id": requested_model,
            "returned_model_id": returned_model,
            "response_received": isinstance(raw, Mapping),
            "usage": usage,
            "cost_yuan": max(
                0.0,
                float(raw.get("cost_yuan") or 0.0)
                if isinstance(raw, Mapping)
                else 0.0,
            ),
            "panel_sha256": (self._panel or {}).get("panel_sha256"),
            "prompt_version": self.config.prompt_version,
            "prompt_revision": int(prompt_revision),
            "run_started_at_utc": self._provider_run_started_at_utc(
                provider, prompt_revision
            ),
        }
        relative = self.store.attempt_relative_path(
            provider,
            persona.persona_id,
            phase=phase,
            position=position,
            attempt_number=attempt_number,
            prompt_revision=prompt_revision,
            arm=arm,
        )
        self.store.write_json(relative, record, immutable=True)
        return record

    def _persist_model_drift_response(
        self,
        *,
        provider: str,
        requested_model: str,
        returned_model: str,
        persona: Persona,
        phase: str,
        position: int,
        item_id: str,
        raw: Mapping[str, Any],
        prompt_revision: int,
        arm: str | None = None,
        source_attempt_number: int,
    ) -> dict[str, Any]:
        usage_raw = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
        usage = {
            "input_tokens": max(0, int(usage_raw.get("input_tokens") or 0)),
            "output_tokens": max(0, int(usage_raw.get("output_tokens") or 0)),
        }
        record = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": persona.persona_id,
            "provider": provider,
            "model_id": returned_model or "missing-provider-model-id",
            "record_type": "llm_sim_excluded_response_accounting",
            "schema_version": "yher.llm_sim.model_drift_exclusion.v1",
            "status": "excluded_response",
            "failure_category": "model_id_drift",
            "exclusion_type": "model_id_drift",
            "phase": phase,
            "arm": arm,
            "position": int(position),
            "item_id": item_id,
            "requested_model": requested_model,
            "returned_model": returned_model,
            "requested_model_id": requested_model,
            "returned_model_id": returned_model,
            "source_attempt_number": int(source_attempt_number),
            "usage": usage,
            "cost_yuan": max(0.0, float(raw.get("cost_yuan") or 0.0)),
            "panel_sha256": (self._panel or {}).get("panel_sha256"),
            "prompt_version": self.config.prompt_version,
            "prompt_revision": int(prompt_revision),
            "run_started_at_utc": self._provider_run_started_at_utc(
                provider, prompt_revision
            ),
        }
        relative = self.store.excluded_response_relative_path(
            provider,
            persona.persona_id,
            phase=phase,
            position=position,
            prompt_revision=prompt_revision,
            arm=arm,
            sequence=source_attempt_number,
        )
        self.store.write_json(relative, record, immutable=True)
        return record

    def _write_calibration_decision(
        self,
        *,
        provider: str,
        requested_model: str,
        panel_sha: str,
        max_items: int,
        arms: Sequence[str],
        status: str,
    ) -> dict[str, Any]:
        preparation = self._preparation_manifest or {}
        calibration_artifacts = self._calibration_artifacts(provider, 0)
        core = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": f"llm-sim-provider:{provider}:calibration-decision",
            "provider": provider,
            "model_id": requested_model,
            "record_type": "llm_sim_calibration_decision",
            "status": status,
            "prompt_revision": 0,
            "prompt_version": self.config.prompt_version,
            "panel_sha256": panel_sha,
            "persona_panel_sha256": preparation.get("persona_panel_sha256"),
            "config_sha256": self.config.sha256,
            "study_seed": self.study_seed,
            "persona_seed_derivation_version": (
                self.config.persona_seed_derivation_version
            ),
            "arms": list(arms),
            "max_items": int(max_items),
            "calibration_artifacts": calibration_artifacts,
            "calibration_artifact_aggregate_sha256": _canonical_sha256(
                calibration_artifacts
            ),
        }
        decision = {**core, "decision_sha256": _canonical_sha256(core)}
        self.store.write_json(
            _calibration_decision_relative(provider),
            decision,
            immutable=True,
        )
        return decision

    def _validated_calibration_decision(
        self,
        provider: str,
    ) -> dict[str, Any] | None:
        relative = _calibration_decision_relative(provider)
        decision = self.store.read_json(relative)
        if decision is None:
            return None
        core = {
            key: value
            for key, value in decision.items()
            if key != "decision_sha256"
        }
        artifacts = decision.get("calibration_artifacts")
        preparation = self._preparation_manifest or {}
        required = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": f"llm-sim-provider:{provider}:calibration-decision",
            "provider": provider,
            "record_type": "llm_sim_calibration_decision",
            "prompt_revision": 0,
            "prompt_version": self.config.prompt_version,
            "panel_sha256": (self._panel or {}).get("panel_sha256"),
            "persona_panel_sha256": preparation.get("persona_panel_sha256"),
            "config_sha256": self.config.sha256,
            "study_seed": self.study_seed,
            "persona_seed_derivation_version": (
                self.config.persona_seed_derivation_version
            ),
        }
        if not isinstance(artifacts, list) or any(
            (
                any(decision.get(key) != value for key, value in required.items()),
                not isinstance(decision.get("model_id"), str),
                not str(decision.get("model_id") or "").strip(),
                decision.get("decision_sha256") != _canonical_sha256(core),
                decision.get("calibration_artifact_aggregate_sha256")
                != _canonical_sha256(artifacts),
                artifacts != self._calibration_artifacts(provider, 0),
            )
        ):
            raise ValueError("immutable calibration decision is missing or invalid")
        return decision

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

    def _provider_artifacts(
        self,
        provider: str,
        arms: Sequence[str],
        prompt_revision: int,
    ) -> list[dict[str, Any]]:
        candidates: set[Path] = set()
        for directory in ("excluded_responses", "attempts"):
            artifact_root = self.store.root / directory
            for absolute in artifact_root.rglob("*.json") if artifact_root.is_dir() else ():
                relative = absolute.relative_to(self.store.root)
                record = self.store.read_json(relative)
                if (
                    record is not None
                    and record.get("provider") == provider
                    and int(record.get("prompt_revision", -1)) == prompt_revision
                ):
                    candidates.add(relative)
        for persona in self.personas:
            calibration = self.store.calibration_relative_path(
                provider, persona.persona_id, prompt_revision
            )
            calibration_partial = self.store.calibration_checkpoint_relative_path(
                provider, persona.persona_id, prompt_revision
            )
            candidates.add(
                calibration if self.store.exists(calibration) else calibration_partial
            )
            for arm in arms:
                journey = self.store.journey_relative_path(
                    provider, persona.persona_id, arm, prompt_revision
                )
                journey_partial = self.store.checkpoint_relative_path(
                    provider, persona.persona_id, arm, prompt_revision
                )
                candidates.add(journey if self.store.exists(journey) else journey_partial)
        rows = []
        for path in sorted(candidates, key=lambda value: value.as_posix()):
            record = self.store.read_json(path)
            if record is None:
                continue
            rows.append(
                {
                    "path": path.as_posix(),
                    "sha256": self.store.file_sha256(path),
                    "record_type": str(record.get("record_type") or ""),
                    "persona_id": str(record.get("persona_id") or ""),
                    "status": str(record.get("status") or ""),
                    "arm": record.get("arm"),
                }
            )
        return rows

    def _persisted_accounting(
        self,
        *,
        provider: str,
        arms: Sequence[str],
        prompt_revision: int,
        accounting: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(accounting)
        attempts: list[Mapping[str, Any]] = []
        completed_by_arm = {arm: 0 for arm in arms}
        incomplete_by_arm = {arm: 0 for arm in arms}
        for artifact in self._provider_artifacts(provider, arms, prompt_revision):
            record = self.store.read_json(artifact["path"])
            if record is None:
                continue
            if record.get("record_type") == "llm_sim_provider_attempt":
                attempts.append(record)
            arm = record.get("arm")
            if (
                record.get("record_type") == "llm_sim_journey"
                and record.get("status") == "complete"
                and arm in completed_by_arm
            ):
                completed_by_arm[str(arm)] += 1
            elif (
                record.get("record_type") == "llm_sim_journey"
                and record.get("status") == "incomplete"
                and arm in incomplete_by_arm
            ):
                incomplete_by_arm[str(arm)] += 1
        if attempts:
            result["requests"] = len(attempts)
            result["responses"] = sum(
                record.get("response_received") is True for record in attempts
            )
            result["retries"] = sum(
                int(record.get("retry_number", -1)) > 0 for record in attempts
            )
            result["failed_requests"] = sum(
                record.get("status") != "response" for record in attempts
            )
            result["input_tokens"] = sum(
                int((record.get("usage") or {}).get("input_tokens") or 0)
                for record in attempts
            )
            result["output_tokens"] = sum(
                int((record.get("usage") or {}).get("output_tokens") or 0)
                for record in attempts
            )
            result["cost_yuan"] = round(
                sum(float(record.get("cost_yuan") or 0.0) for record in attempts),
                12,
            )
            result["model_drift_detected"] = any(
                record.get("failure_category") == "model_id_drift"
                for record in attempts
            )
            result["returned_model_ids"] = sorted(
                {
                    str(record["returned_model_id"])
                    for record in attempts
                    if record.get("returned_model_id")
                }
            )
        result["completed_by_arm"] = completed_by_arm
        result["completed"] = sum(completed_by_arm.values())
        result["incomplete_by_arm"] = incomplete_by_arm
        result["structural_incomplete"] = sum(incomplete_by_arm.values())
        return result

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
        failure_category: str | None = None,
    ) -> dict[str, Any]:
        panel_rows = list((self._panel or {}).get("annotations") or ())
        mapped_personas = sum(
            row.get("mapping_status") == "mapped" for row in panel_rows
        )
        calibration_reportable = all(
            bool((provider_eligibility or {}).get(strength, {}).get("eligible"))
            for strength in ("weak", "strong")
        )
        model_drift_detected = accounting.get("model_drift_detected") is True
        if model_drift_detected:
            status = "excluded_model_drift"
            failure_category = failure_category or "model_id_drift"
        raw_completed = dict(accounting.get("completed_by_arm") or {})
        eligible_completed = {
            arm: (
                int(raw_completed.get(arm, 0))
                if calibration_reportable and not model_drift_detected
                else 0
            )
            for arm in arms
        }
        formal_design = _formal_design_check(
            personas=self.personas,
            study_seed=self.study_seed,
            max_items=max_items,
            arms=arms,
            config=self.config,
            canonical_personas_sha256=(
                self.store.read_json("persona_panel.json") or {}
            ).get("canonical_personas_sha256"),
            persona_panel_personas_sha256=(
                self.store.read_json("persona_panel.json") or {}
            ).get("personas_sha256"),
        )
        completion_reportable = (
            formal_design["match"]
            and all(
                eligible_completed[arm] >= self.config.minimum_complete_per_cell
                for arm in arms
            )
        )
        artifacts = self._provider_artifacts(provider, arms, prompt_revision)
        existing_manifest = self.store.read_json(
            _provider_manifest_relative(provider, prompt_revision)
        )
        calibration_decision = self._validated_calibration_decision(provider)
        manifest = {
            "simulated": True,
            "run_id": self.config.run_id,
            "persona_id": f"llm-sim-provider:{provider}",
            "provider": provider,
            "model_id": requested_model,
            "record_type": "llm_sim_provider_manifest",
            "run_started_at_utc": self._provider_run_started_at_utc(
                provider, prompt_revision
            ),
            "status": status,
            "exclusion_reason": exclusion_reason,
            "failure_category": failure_category,
            "prompt_revision": prompt_revision,
            "calibration_decision_path": (
                _calibration_decision_relative(provider).as_posix()
                if calibration_decision
                else None
            ),
            "calibration_decision_sha256": (
                calibration_decision.get("decision_sha256")
                if calibration_decision
                else None
            ),
            "calibration_attempt_count": calibration_attempt_count,
            "provider_eligibility": dict(provider_eligibility or {}),
            "panel_sha256": panel_sha,
            "persona_panel_path": "persona_panel.json",
            "persona_panel_sha256": (self._preparation_manifest or {}).get(
                "persona_panel_sha256"
            ),
            "preparation_persona_panel_sha256": (
                self._preparation_manifest or {}
            ).get("persona_panel_sha256"),
            "frozen_pre_observation_utc": self.config.frozen_pre_observation_utc,
            "persona_seed_derivation_version": (
                self.config.persona_seed_derivation_version
            ),
            "prompt_version": self.config.prompt_version,
            "config_sha256": self.config.sha256,
            "study_seed": self.study_seed,
            "git_head": (self._preparation_manifest or {}).get("git_head"),
            "code_sha256": (self._preparation_manifest or {}).get("code_sha256"),
            "working_code_sha256": (self._preparation_manifest or {}).get(
                "working_code_sha256"
            ),
            "head_code_sha256": (self._preparation_manifest or {}).get(
                "head_code_sha256"
            ),
            "code_matches_head": (self._preparation_manifest or {}).get(
                "code_matches_head"
            ),
            "analysis_plan_commit": self.config.analysis_plan_commit,
            "analysis_plan_is_ancestor": (self._preparation_manifest or {}).get(
                "analysis_plan_is_ancestor"
            ),
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
            "artifacts": artifacts,
            "artifact_aggregate_sha256": _canonical_sha256(artifacts),
            "reportability": {
                "minimum_complete_per_cell": self.config.minimum_complete_per_cell,
                "cell_completed": raw_completed,
                "eligible_cell_completed": eligible_completed,
                "completion_reportable": completion_reportable,
                "formal_design_match": formal_design["match"],
                "formal_design_failures": formal_design["failures"],
                "persona_composition": formal_design["composition"],
                "canonical_persona_panel_match": (
                    "canonical_persona_panel"
                    not in formal_design["failures"]
                ),
                "mechanically_mapped_personas": mapped_personas,
                "pre_outcome_mapping_exclusions": len(panel_rows) - mapped_personas,
                "model_drift_detected": model_drift_detected,
                "manipulation_metric_reportable": (
                    completion_reportable
                    and calibration_reportable
                    and mapped_personas > 0
                    and not model_drift_detected
                ),
                "calibration_reportable": calibration_reportable,
                "reportable": (
                    completion_reportable
                    and calibration_reportable
                    and mapped_personas > 0
                    and not model_drift_detected
                ),
            },
            "accounting": dict(accounting),
            "protected_filesystem_assertion": {
                "before_sha256": protected_before["digest"],
                "after_sha256": protected_after["digest"],
                "unchanged": protected_before == protected_after,
                "coverage": protected_before["coverage"],
            },
            "created_at_utc": (
                existing_manifest.get("created_at_utc")
                if existing_manifest
                else self._provider_run_started_at_utc(provider, prompt_revision)
            ),
        }
        self.store.write_json(
            _provider_manifest_relative(provider, prompt_revision), manifest
        )
        return manifest


def _formal_design_check(
    *,
    personas: Sequence[Persona],
    study_seed: int,
    max_items: int,
    arms: Sequence[str],
    config: LLMSimConfig,
    canonical_personas_sha256: str | None = None,
    persona_panel_personas_sha256: str | None = None,
) -> dict[str, Any]:
    composition = {
        strength: sum(persona.strength == strength for persona in personas)
        for strength in ("weak", "strong")
    }
    pair_rows: dict[str, list[Persona]] = {}
    for persona in personas:
        pair_rows.setdefault(persona.pair_id, []).append(persona)
    failures = []
    if composition != {"weak": config.pair_count, "strong": config.pair_count}:
        failures.append("persona_composition")
    if len({persona.persona_id for persona in personas}) != len(personas):
        failures.append("persona_id_uniqueness")
    if len(pair_rows) != config.pair_count or any(
        len(rows) != 2
        or sorted(persona.strength for persona in rows) != ["strong", "weak"]
        for rows in pair_rows.values()
    ):
        failures.append("pair_composition")
    canonical_pairs = [rows for rows in pair_rows.values() if len(rows) == 2]
    if any(len({persona.target_node for persona in rows}) != 1 for rows in canonical_pairs):
        failures.append("pair_target_node")
    if any(len({persona.failure_id for persona in rows}) != 1 for rows in canonical_pairs):
        failures.append("pair_failure_id")
    if any(len({persona.seed for persona in rows}) != 1 for rows in canonical_pairs):
        failures.append("pair_seed")
    definition_fields = (
        "failure_cause",
        "failure_symptom",
        "diagnostic_question",
        "annotation_source",
    )
    if any(
        any(len({getattr(persona, field) for persona in rows}) != 1 for field in definition_fields)
        for rows in canonical_pairs
    ):
        failures.append("pair_failure_definition")
    if int(study_seed) != config.study_seed:
        failures.append("study_seed")
    if int(max_items) != config.max_items:
        failures.append("max_items")
    if tuple(arms) != config.arms:
        failures.append("arms")
    if (
        canonical_personas_sha256 is not None
        or persona_panel_personas_sha256 is not None
    ) and canonical_personas_sha256 != persona_panel_personas_sha256:
        failures.append("canonical_persona_panel")
    return {
        "match": not failures,
        "failures": failures,
        "composition": composition,
        "pair_count": len(pair_rows),
    }


def _provider_manifest_relative(provider: str, prompt_revision: int) -> Path:
    suffix = f"__prompt-v{prompt_revision}" if prompt_revision else ""
    return Path("providers") / f"{provider}{suffix}.json"


def _calibration_decision_relative(provider: str) -> Path:
    return Path("calibration_decisions") / f"{provider}.json"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _failure_category(exc: Exception) -> str:
    if isinstance(exc, ModelDriftError):
        return "model_id_drift"
    if isinstance(exc, ProviderHTTPError):
        return "http_status"
    if isinstance(exc, ProviderNetworkError):
        return "network"
    if isinstance(exc, ProviderProtocolError):
        return "protocol"
    if isinstance(exc, CircuitOpenError):
        return "circuit"
    return "unexpected"


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
    if not unseen:
        return None
    pool = unseen
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
    prompt_version: str = "yher-llm-persona-prompt-v1",
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
        + (
            f" This is the predeclared {phase} phase "
            f"(prompt version {prompt_version}; revision {prompt_revision})."
        )
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
    prompt_version: str,
    calibration_items: Sequence[Mapping[str, Any]],
) -> bool:
    if not record or record.get("status") not in {
        "complete",
        "structural_failure",
    }:
        return False
    status = str(record["status"])
    events = record.get("events")
    if not isinstance(events, list):
        return False
    expected_items = list(calibration_items)
    if len(expected_items) != CALIBRATION_ITEMS_PER_PERSONA:
        return False
    if status == "complete" and len(events) != CALIBRATION_ITEMS_PER_PERSONA:
        return False
    if status == "structural_failure" and not 0 <= len(events) < CALIBRATION_ITEMS_PER_PERSONA:
        return False
    record_valid = all(
        (
            record.get("simulated") is True,
            record.get("run_id") == FROZEN_RUN_ID,
            record.get("record_type") == "llm_sim_calibration",
            record.get("provider") == provider,
            record.get("model_id") == model,
            record.get("persona_id") == persona.persona_id,
            record.get("strength") == persona.strength,
            record.get("panel_sha256") == panel_sha,
            record.get("prompt_version") == prompt_version,
            int(record.get("prompt_revision", -1)) == prompt_revision,
        )
    )
    if not record_valid:
        return False
    seen_items: set[str] = set()
    seen_families: set[str] = set()
    for index, (event, expected) in enumerate(zip(events, expected_items), start=1):
        if not isinstance(event, Mapping):
            return False
        item_id = str(expected.get("item_id") or "")
        family_id = str(expected.get("family_id") or "")
        if not item_id or not family_id:
            return False
        if item_id in seen_items or family_id in seen_families:
            return False
        seen_items.add(item_id)
        seen_families.add(family_id)
        if not all(
            (
                event.get("simulated") is True,
                event.get("run_id") == FROZEN_RUN_ID,
                event.get("record_type") == "llm_sim_calibration_attempt",
                event.get("phase") == "calibration",
                event.get("provider") == provider,
                event.get("model_id") == model,
                event.get("persona_id") == persona.persona_id,
                event.get("pair_id") == persona.pair_id,
                event.get("strength") == persona.strength,
                event.get("target_node") == persona.target_node,
                event.get("failure_id") == persona.failure_id,
                int(event.get("position", -1)) == index,
                str(event.get("item_id") or "") == item_id,
                str(event.get("family_id") or "") == family_id,
                isinstance(event.get("correct"), bool),
                event.get("panel_sha256") == panel_sha,
                event.get("prompt_version") == prompt_version,
                int(event.get("prompt_revision", -1)) == prompt_revision,
            )
        ):
            return False
    if status == "structural_failure":
        missing_index = len(events)
        return all(
            (
                record.get("failure_category") == "catalog_item_missing",
                record.get("terminal_reason")
                == "frozen_calibration_item_unavailable",
                int(record.get("expected_item_count", -1))
                == CALIBRATION_ITEMS_PER_PERSONA,
                int(record.get("actual_administered_count", -1)) == len(events),
                record.get("missing_item_id")
                == str(expected_items[missing_index].get("item_id") or ""),
            )
        )
    return True


def _resume_calibration_events(
    record: Mapping[str, Any] | None,
    *,
    provider: str,
    model: str,
    persona: Persona,
    panel_sha: str,
    prompt_revision: int,
    prompt_version: str,
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
            record.get("prompt_version") == prompt_version,
            int(record.get("prompt_revision", -1)) == prompt_revision,
        )
    ):
        raise ValueError("calibration checkpoint envelope is invalid")
    raw_events = record.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("calibration checkpoint events are invalid")
    events = [dict(event) for event in raw_events if isinstance(event, Mapping)]
    if len(events) != len(raw_events):
        raise ValueError("calibration checkpoint event is invalid")
    expected_ids = [str(row["item_id"]) for row in calibration_items]
    expected_families = [str(row.get("family_id") or "") for row in calibration_items]
    if (
        len(calibration_items) != CALIBRATION_ITEMS_PER_PERSONA
        or len(events) > CALIBRATION_ITEMS_PER_PERSONA
        or len({str(event.get("item_id") or "") for event in events}) != len(events)
        or any(
            not all(
                (
                    event.get("simulated") is True,
                    event.get("run_id") == FROZEN_RUN_ID,
                    event.get("record_type") == "llm_sim_calibration_attempt",
                    event.get("phase") == "calibration",
                    event.get("provider") == provider,
                    event.get("model_id") == model,
                    event.get("persona_id") == persona.persona_id,
                    event.get("pair_id") == persona.pair_id,
                    event.get("strength") == persona.strength,
                    event.get("target_node") == persona.target_node,
                    event.get("failure_id") == persona.failure_id,
                    int(event.get("position", -1)) == index,
                    str(event.get("item_id") or "") == expected_ids[index - 1],
                    str(event.get("family_id") or "")
                    == expected_families[index - 1],
                    event.get("panel_sha256") == panel_sha,
                    event.get("prompt_version") == prompt_version,
                    int(event.get("prompt_revision", -1)) == prompt_revision,
                )
            )
            for index, event in enumerate(events, start=1)
        )
    ):
        raise ValueError("calibration checkpoint does not match the frozen item order")
    return events


def _calibration_eligibility(
    *,
    strength: str,
    personas: Sequence[Persona],
    attempts: Sequence[Mapping[str, Any]],
    panel_rows: Sequence[Mapping[str, Any]],
    calibration_ready: bool,
    config: LLMSimConfig,
    prompt_revision: int,
) -> dict[str, Any]:
    denominator = len(attempts)
    correct = sum(row.get("correct") is True for row in attempts)
    accuracy = correct / denominator if denominator else None
    if strength == "weak":
        threshold_value = config.weak_accuracy_upper
        threshold = f"<{threshold_value:g}"
        band_pass = accuracy is not None and accuracy < threshold_value
    else:
        threshold_value = config.strong_accuracy_lower
        threshold = f">{threshold_value:g}"
        band_pass = accuracy is not None and accuracy > threshold_value
    expected_n = len(personas) * CALIBRATION_ITEMS_PER_PERSONA
    formal_expected_n = config.pair_count * CALIBRATION_ITEMS_PER_PERSONA
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
    if strength == "weak" and target_panel_ready:
        contrast, contrast_lower = _persona_cluster_contrast_bootstrap(
            personas=personas,
            attempts=mapped_wrong,
            seed=config.manipulation_bootstrap_seed,
            resamples=config.manipulation_bootstrap_resamples,
        )
        target_gate_pass = contrast_lower is not None and contrast_lower > 0.0
        target_gate = {
            "applicable": True,
            "status": "passed" if target_gate_pass else "failed",
            "pass": bool(target_gate_pass),
            "contrast": contrast,
            "contrast_ci95_lower": contrast_lower,
            "bootstrap_seed": config.manipulation_bootstrap_seed,
            "bootstrap_resamples": config.manipulation_bootstrap_resamples,
            "cluster_unit": "persona_id",
        }
    elif strength == "weak":
        target_gate_pass = False
        target_gate = {
            "applicable": True,
            "status": "excluded_pre_outcome",
            "pass": False,
            "contrast": None,
            "contrast_ci95_lower": None,
            "bootstrap_seed": config.manipulation_bootstrap_seed,
            "bootstrap_resamples": config.manipulation_bootstrap_resamples,
            "cluster_unit": "persona_id",
        }
    else:
        target_gate_pass = True
        target_gate = {
            "applicable": False,
            "status": "not_applicable",
            "pass": None,
            "contrast": None,
            "contrast_ci95_lower": None,
            "bootstrap_seed": config.manipulation_bootstrap_seed,
            "bootstrap_resamples": config.manipulation_bootstrap_resamples,
            "cluster_unit": "persona_id",
        }
    accuracy_gate = {
        "applicable": True,
        "status": "passed" if band_pass else "failed",
        "pass": bool(band_pass),
        "threshold": threshold,
        "accuracy": accuracy,
    }
    if not calibration_ready:
        status = "excluded_pre_outcome"
        reason = "insufficient_family_distinct_calibration_mcq"
    elif denominator != expected_n:
        status = "incomplete_calibration"
        reason = "calibration_denominator_incomplete"
    elif not band_pass:
        status = (
            "prompt_rewrite_available"
            if prompt_revision < config.maximum_prompt_rewrites
            else "excluded_post_calibration"
        )
        reason = "accuracy_band_failed"
    elif strength == "weak" and not target_panel_ready:
        status = "excluded_pre_outcome"
        reason = "no_mechanical_target_option_mapping"
    elif strength == "weak" and not target_gate_pass:
        status = (
            "prompt_rewrite_available"
            if prompt_revision < config.maximum_prompt_rewrites
            else "excluded_post_calibration"
        )
        reason = "target_contrast_failed"
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
        "accuracy_gate": accuracy_gate,
        "target_gate": target_gate,
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
        "minimum_complete_per_cell": config.minimum_complete_per_cell,
        "personas": len(personas),
        "prompt_revision": prompt_revision,
    }


def _persona_cluster_contrast_bootstrap(
    *,
    personas: Sequence[Persona],
    attempts: Sequence[Mapping[str, Any]],
    seed: int,
    resamples: int,
) -> tuple[float | None, float | None]:
    persona_ids = [persona.persona_id for persona in personas]
    if not persona_ids:
        return None, None
    index = {persona_id: offset for offset, persona_id in enumerate(persona_ids)}
    numerators = np.zeros(len(persona_ids), dtype=float)
    denominators = np.zeros(len(persona_ids), dtype=float)
    for row in attempts:
        persona_index = index.get(str(row.get("persona_id") or ""))
        baseline = row.get("random_wrong_option_baseline")
        hit = row.get("target_misconception_hit")
        if persona_index is None or baseline is None or not isinstance(hit, bool):
            continue
        numerators[persona_index] += float(hit) - float(baseline)
        denominators[persona_index] += 1.0
    total_denominator = float(denominators.sum())
    if total_denominator <= 0:
        return None, None
    contrast = float(numerators.sum() / total_denominator)
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(
        0,
        len(persona_ids),
        size=(int(resamples), len(persona_ids)),
    )
    sampled_numerators = numerators[sampled].sum(axis=1)
    sampled_denominators = denominators[sampled].sum(axis=1)
    valid = sampled_denominators > 0
    if not np.any(valid):
        return contrast, None
    estimates = sampled_numerators[valid] / sampled_denominators[valid]
    lower = float(np.quantile(estimates, 0.025))
    return contrast, lower


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
    config_sha256: str,
    persona_panel_sha256: str | None,
    study_seed: int,
    analysis_plan_commit: str,
    prompt_version: str,
) -> bool:
    if not record:
        return False
    status = record.get("status")
    terminal_reason = record.get("terminal_reason")
    if status == "complete":
        terminal_valid = terminal_reason in {"confidence", "budget_exhausted"}
    elif status == "incomplete":
        terminal_valid = terminal_reason in {
            "structural_failure_no_items",
            "structural_failure_item_pool",
        }
    else:
        terminal_valid = False
    events = record.get("events")
    if not terminal_valid or not isinstance(events, list):
        return False
    if not 0 <= len(events) <= max_items:
        return False
    if terminal_reason == "budget_exhausted" and len(events) != max_items:
        return False
    if len(events) != int(record.get("actual_administered_count", -1)):
        return False
    record_valid = all(
        (
            record.get("simulated") is True,
            record.get("run_id") == FROZEN_RUN_ID,
            record.get("record_type") == "llm_sim_journey",
            record.get("provider") == provider,
            record.get("model_id") == requested_model,
            record.get("persona_id") == persona.persona_id,
            record.get("pair_id") == persona.pair_id,
            record.get("strength") == persona.strength,
            record.get("target_node") == persona.target_node,
            record.get("failure_id") == persona.failure_id,
            record.get("arm") == arm,
            int(record.get("max_items", -1)) == max_items,
            record.get("panel_sha256") == panel_sha,
            record.get("config_sha256") == config_sha256,
            record.get("persona_panel_sha256") == persona_panel_sha256,
            int(record.get("study_seed", -1)) == study_seed,
            record.get("analysis_plan_commit") == analysis_plan_commit,
            record.get("prompt_version") == prompt_version,
            int(record.get("prompt_revision", -1)) == prompt_revision,
        )
    )
    if not record_valid:
        return False
    item_ids = [str(event.get("item_id") or "") for event in events if isinstance(event, Mapping)]
    if len(item_ids) != len(events) or not all(item_ids) or len(set(item_ids)) != len(item_ids):
        return False
    if any(
        not _journey_event_is_valid(
            event,
            provider=provider,
            model=requested_model,
            persona=persona,
            arm=arm,
            panel_sha=panel_sha,
            prompt_revision=prompt_revision,
            prompt_version=prompt_version,
            position=index,
        )
        for index, event in enumerate(events, start=1)
    ):
        return False
    if not _journey_transitions_are_valid(events):
        return False
    final_belief = record.get("final_belief")
    if not _belief_vector(final_belief):
        return False
    expected_final = events[-1].get("posterior_belief") if events else mastery.UNIFORM.tolist()
    if list(final_belief) != list(expected_final):
        return False

    confidence_stop_position = next(
        (
            index
            for index, event in enumerate(events, start=1)
            if selector.should_stop(
                {
                    persona.target_node: np.asarray(
                        event["posterior_belief"], dtype=float
                    )
                },
                [persona.target_node],
                direct_answers={
                    persona.target_node: int(event["direct_answers_after"])
                },
                budget_items=max_items + 1,
                asked=index,
            )
        ),
        None,
    )
    event_count = len(events)
    if terminal_reason == "confidence":
        return event_count > 0 and confidence_stop_position == event_count
    if terminal_reason == "budget_exhausted":
        return (
            event_count > 0
            and event_count == max_items
            and confidence_stop_position is None
        )
    if terminal_reason == "structural_failure_no_items":
        return event_count == 0
    return (
        terminal_reason == "structural_failure_item_pool"
        and 0 < event_count < max_items
        and confidence_stop_position is None
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
    prompt_version: str,
) -> list[dict[str, Any]]:
    if not record:
        return []
    if not all(
        (
            record.get("simulated") is True,
            record.get("run_id") == FROZEN_RUN_ID,
            record.get("record_type") == "llm_sim_journey",
            record.get("status") == "in_progress",
            record.get("provider") == provider,
            record.get("model_id") == model,
            record.get("persona_id") == persona.persona_id,
            record.get("pair_id") == persona.pair_id,
            record.get("strength") == persona.strength,
            record.get("target_node") == persona.target_node,
            record.get("failure_id") == persona.failure_id,
            record.get("arm") == arm,
            int(record.get("max_items", -1)) == max_items,
            record.get("panel_sha256") == panel_sha,
            record.get("prompt_version") == prompt_version,
            int(record.get("prompt_revision", -1)) == prompt_revision,
        )
    ):
        raise ValueError("journey checkpoint envelope is invalid")
    raw_events = record.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("journey checkpoint events are invalid")
    events = list(raw_events)
    item_ids = [str(event.get("item_id") or "") for event in events if isinstance(event, Mapping)]
    if (
        len(events) > max_items
        or len(item_ids) != len(events)
        or not all(item_ids)
        or len(set(item_ids)) != len(item_ids)
        or any(
            not _journey_event_is_valid(
                event,
                provider=provider,
                model=model,
                persona=persona,
                arm=arm,
                panel_sha=panel_sha,
                prompt_revision=prompt_revision,
                prompt_version=prompt_version,
                position=index,
            )
            for index, event in enumerate(events, start=1)
        )
        or not _journey_transitions_are_valid(events)
    ):
        raise ValueError("journey checkpoint contains invalid or repeated events")
    return [dict(event) for event in events]


def _belief_vector(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        numbers = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        np.all(np.isfinite(numbers))
        and np.all(numbers >= 0.0)
        and np.isclose(numbers.sum(), 1.0, rtol=0.0, atol=1e-9)
    )


def _likelihood_vector(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        numbers = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        np.all(np.isfinite(numbers))
        and np.all(numbers >= 0.0)
        and numbers.sum() > 0.0
    )


def _journey_event_is_valid(
    event: Any,
    *,
    provider: str,
    model: str,
    persona: Persona,
    arm: str,
    panel_sha: str,
    prompt_revision: int,
    prompt_version: str,
    position: int,
) -> bool:
    if not isinstance(event, Mapping):
        return False
    update_applied = event.get("update_applied")
    correct = event.get("correct")
    try:
        direct_before = int(event.get("direct_answers_before", -1))
        direct_after = int(event.get("direct_answers_after", -1))
    except (TypeError, ValueError):
        return False
    return all(
        (
            event.get("simulated") is True,
            event.get("run_id") == FROZEN_RUN_ID,
            event.get("record_type") == "llm_sim_event",
            event.get("provider") == provider,
            event.get("model_id") == model,
            event.get("persona_id") == persona.persona_id,
            event.get("pair_id") == persona.pair_id,
            event.get("strength") == persona.strength,
            event.get("target_node") == persona.target_node,
            event.get("failure_id") == persona.failure_id,
            event.get("arm") == arm,
            int(event.get("position", -1)) == position,
            str(event.get("item_id") or "") != "",
            event.get("role") in {"local", "prereq"},
            isinstance(event.get("score_status"), str),
            bool(str(event.get("score_status") or "")),
            correct is None or isinstance(correct, bool),
            isinstance(update_applied, bool),
            not update_applied or isinstance(correct, bool),
            direct_before >= 0,
            direct_after >= 0,
            event.get("panel_sha256") == panel_sha,
            event.get("prompt_version") == prompt_version,
            int(event.get("prompt_revision", -1)) == prompt_revision,
            _likelihood_vector(event.get("inference_likelihood")),
            _belief_vector(event.get("prior_belief")),
            _belief_vector(event.get("posterior_belief")),
        )
    )


def _journey_transitions_are_valid(events: Sequence[Mapping[str, Any]]) -> bool:
    node = mastery.NodeBelief(mastery.UNIFORM.copy())
    for index, event in enumerate(events, start=1):
        now = float(index)
        expected_prior = mastery.get_belief(node, now)
        if not np.allclose(
            expected_prior,
            np.asarray(event.get("prior_belief"), dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            return False
        if int(event.get("direct_answers_before", -1)) != node.direct_answers:
            return False
        if event.get("update_applied") is True:
            mastery.observe(
                node,
                np.asarray(event.get("inference_likelihood"), dtype=float),
                now,
                is_direct=event.get("role") == "local",
            )
        expected_posterior = mastery.get_belief(node, now)
        if not np.allclose(
            expected_posterior,
            np.asarray(event.get("posterior_belief"), dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            return False
        if int(event.get("direct_answers_after", -1)) != node.direct_answers:
            return False
    return True


def _provider_attempt_is_valid(
    record: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    persona: Persona,
    panel_sha: str,
    prompt_revision: int,
    prompt_version: str,
    run_started_at_utc: str,
) -> bool:
    status = record.get("status")
    returned = record.get("returned_model_id")
    response_received = record.get("response_received")
    if status == "response":
        outcome_valid = all(
            (
                returned == model,
                record.get("model_id") == model,
                response_received is True,
                record.get("failure_category") is None,
            )
        )
    elif status == "model_drift":
        outcome_valid = all(
            (
                isinstance(returned, str),
                bool(str(returned or "")),
                returned != model,
                record.get("model_id") == returned,
                response_received is True,
                record.get("failure_category") == "model_id_drift",
            )
        )
    elif status == "protocol_failure":
        outcome_valid = all(
            (
                returned is None,
                record.get("model_id") == "missing-provider-model-id",
                record.get("failure_category") == "protocol",
            )
        )
    elif status == "failed":
        outcome_valid = all(
            (
                returned is None,
                record.get("model_id") == model,
                response_received is False,
                isinstance(record.get("failure_category"), str),
            )
        )
    else:
        return False
    usage = record.get("usage")
    try:
        usage_valid = (
            isinstance(usage, Mapping)
            and int(usage.get("input_tokens", -1)) >= 0
            and int(usage.get("output_tokens", -1)) >= 0
            and np.isfinite(float(record.get("cost_yuan", -1)))
            and float(record.get("cost_yuan", -1)) >= 0.0
        )
    except (TypeError, ValueError):
        usage_valid = False
    return bool(
        outcome_valid
        and usage_valid
        and all(
            (
                record.get("simulated") is True,
                record.get("run_id") == FROZEN_RUN_ID,
                record.get("record_type") == "llm_sim_provider_attempt",
                record.get("schema_version")
                == "yher.llm_sim.provider_attempt.v1",
                record.get("provider") == provider,
                record.get("persona_id") == persona.persona_id,
                record.get("requested_model_id") == model,
                record.get("run_started_at_utc") == run_started_at_utc,
                record.get("exclusion_type")
                == ("model_id_drift" if status == "model_drift" else None),
                record.get("phase") in {"calibration", "journey"},
                record.get("arm") in {None, "A", "B"},
                int(record.get("position", -1)) >= 1,
                int(record.get("attempt_number", -1)) >= 1,
                int(record.get("retry_number", -1)) >= 0,
                bool(str(record.get("item_id") or "")),
                record.get("panel_sha256") == panel_sha,
                record.get("prompt_version") == prompt_version,
                int(record.get("prompt_revision", -1)) == prompt_revision,
                not ({"content", "raw", "error", "error_body"} & record.keys()),
            )
        )
    )
