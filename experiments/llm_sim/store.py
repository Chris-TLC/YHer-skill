"""Filesystem-isolated storage for simulated LLM journeys."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class SimulationStore:
    """A small atomic JSON store that cannot target production state.

    The constructor accepts a temporary directory or ``data/sim_store/*``.
    Paths containing ``local_store``, study logs, or official manifests are
    rejected before any directory is created.
    """

    def __init__(self, root: str | Path):
        candidate = Path(root).expanduser().resolve(strict=False)
        lowered = {part.lower() for part in candidate.parts}
        if "local_store" in lowered:
            raise ValueError("simulation store cannot be local_store")
        if "study_logs" in lowered:
            raise ValueError("simulation store cannot be study_logs")
        if any(part.lower() in {"item_bank", "knowledge_graph", "official"} for part in candidate.parts):
            raise ValueError("simulation store cannot be an official data path")
        # A final ``sim_store`` component is required for repository paths;
        # arbitrary /tmp roots remain useful for tests and isolated runs.
        if str(candidate).startswith("/") and "tmp" not in lowered and candidate.name != "sim_store":
            # Permit a repository's data/sim_store subtree, including named
            # provider/run descendants.
            if not any(part == "sim_store" for part in candidate.parts):
                raise ValueError("simulation store must be under an isolated sim_store root")
        self.root = candidate

    def _path(self, relative: str | Path) -> Path:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("simulation path must be relative and cannot escape store")
        path = (self.root / rel).resolve(strict=False)
        if not (path == self.root or path.is_relative_to(self.root)):
            raise ValueError("simulation path escapes store")
        if any(part.lower() in {"local_store", "study_logs"} for part in path.parts):
            raise ValueError("simulation path cannot target local_store or study_logs")
        return path

    def write_json(
        self,
        relative: str | Path,
        record: Mapping[str, Any],
        *,
        immutable: bool = False,
    ) -> Path:
        self._validate_envelope(record)
        path = self._path(relative)
        payload = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        if immutable and path.exists():
            if path.read_bytes() == payload:
                return path
            raise FileExistsError(f"immutable simulation artifact already exists: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return path

    def read_json(self, relative: str | Path) -> dict[str, Any] | None:
        path = self._path(relative)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("simulation record must be a JSON object")
        return value

    def exists(self, relative: str | Path) -> bool:
        return self._path(relative).is_file()

    def file_sha256(self, relative: str | Path) -> str:
        path = self._path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def journey_relative_path(
        self,
        provider: str,
        persona_id: str,
        arm: str,
        prompt_revision: int = 0,
    ) -> Path:
        safe_provider = _safe_component(provider)
        safe_persona = _safe_component(persona_id)
        safe_arm = _safe_component(arm)
        revision = f"__prompt-v{int(prompt_revision)}" if prompt_revision else ""
        return Path("journeys") / safe_provider / f"{safe_persona}__arm-{safe_arm}{revision}.json"

    def checkpoint_relative_path(
        self,
        provider: str,
        persona_id: str,
        arm: str,
        prompt_revision: int = 0,
    ) -> Path:
        journey = self.journey_relative_path(provider, persona_id, arm, prompt_revision)
        return journey.with_suffix(".partial.json")

    def calibration_relative_path(
        self,
        provider: str,
        persona_id: str,
        prompt_revision: int = 0,
    ) -> Path:
        revision = f"__prompt-v{int(prompt_revision)}" if prompt_revision else ""
        return Path("calibration") / _safe_component(provider) / f"{_safe_component(persona_id)}{revision}.json"

    def calibration_checkpoint_relative_path(
        self,
        provider: str,
        persona_id: str,
        prompt_revision: int = 0,
    ) -> Path:
        return self.calibration_relative_path(
            provider, persona_id, prompt_revision
        ).with_suffix(".partial.json")

    def excluded_response_relative_path(
        self,
        provider: str,
        persona_id: str,
        *,
        phase: str,
        position: int,
        prompt_revision: int = 0,
        arm: str | None = None,
        sequence: int | None = None,
    ) -> Path:
        if int(position) < 1:
            raise ValueError("excluded response position must be positive")
        revision = f"__prompt-v{int(prompt_revision)}" if prompt_revision else ""
        arm_suffix = f"__arm-{_safe_component(arm)}" if arm else ""
        sequence_suffix = (
            f"__attempt-{int(sequence):03d}" if sequence is not None else ""
        )
        if sequence is not None and int(sequence) < 1:
            raise ValueError("excluded response sequence must be positive")
        filename = (
            f"{_safe_component(persona_id)}__phase-{_safe_component(phase)}"
            f"{arm_suffix}__position-{int(position):03d}{sequence_suffix}{revision}.json"
        )
        return Path("excluded_responses") / _safe_component(provider) / filename

    def attempt_relative_path(
        self,
        provider: str,
        persona_id: str,
        *,
        phase: str,
        position: int,
        attempt_number: int,
        prompt_revision: int = 0,
        arm: str | None = None,
    ) -> Path:
        if int(position) < 1 or int(attempt_number) < 1:
            raise ValueError("attempt position and number must be positive")
        revision = f"__prompt-v{int(prompt_revision)}" if prompt_revision else ""
        arm_suffix = f"__arm-{_safe_component(arm)}" if arm else ""
        filename = (
            f"{_safe_component(persona_id)}__phase-{_safe_component(phase)}"
            f"{arm_suffix}__position-{int(position):03d}"
            f"__attempt-{int(attempt_number):03d}{revision}.json"
        )
        return Path("attempts") / _safe_component(provider) / filename

    @staticmethod
    def _validate_envelope(record: Mapping[str, Any]) -> None:
        if record.get("simulated") is not True:
            raise ValueError("simulation record requires simulated:true")
        for field in ("persona_id", "provider", "model_id"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "simulated event envelope requires simulated:true, persona_id, provider, and model_id"
                )


def _safe_component(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("unsafe simulation path component")
    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")
    return "id-" + encoded
