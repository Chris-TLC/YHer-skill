"""Filesystem guard for physically separate Persona v2 pilot/main runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .keys import canonical_key


RUN_ID = "llm-personas-v2-dual"
_FORBIDDEN_PATH_MARKERS = {
    "official",
    "local_store",
    "study_log",
    "study_logs",
    "study-log",
    "study-logs",
}


def _reject_path(path: Path) -> None:
    lowered = [part.lower() for part in path.parts]
    for part in lowered:
        normalized = re.sub(r"[^a-z0-9]+", "_", part).strip("_")
        if (
            part in _FORBIDDEN_PATH_MARKERS
            or normalized == "official"
            or normalized.startswith("official_")
            or normalized.endswith("_official")
            or re.search(r"(?:^|_)local_store(?:_|$)", normalized)
            or re.search(r"(?:^|_)study_log(?:_|$)", normalized)
            or re.search(r"(?:^|_)studylog(?:_|$)", normalized)
            or re.search(r"(?:^|_)item_bank(?:_|$)", normalized)
            or re.search(r"(?:^|_)knowledge_graph(?:_|$)", normalized)
            or "llm-personas-v1" in part
            or "llm_sim_v1" in part
        ):
            raise ValueError("v1, official, local-store, and study-log paths are forbidden")
        if re.search(r"(?:^|_)v1(?:_|$)", normalized):
            raise ValueError("v1 roots are forbidden")
        if part.startswith("llm-personas-") and part != RUN_ID:
            raise ValueError(f"the only permitted persona run name is {RUN_ID}")


def _validate_record(record: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("v2 records must be objects")
    if record.get("simulated") is not True:
        raise ValueError("v2 records require simulated:true")
    if record.get("run_id") != RUN_ID:
        raise ValueError(f"v2 record run_id must be exactly {RUN_ID}")
    if record.get("phase") != phase:
        raise ValueError("v2 record phase does not match its store")
    if record.get("analysis_population") != phase:
        raise ValueError("v2 record analysis_population does not match its store")
    if _contains_v1_run_envelope(record):
        raise ValueError("v1 envelopes are forbidden in the v2 store")
    schema_version = str(record.get("schema_version", "")).lower()
    if "v1" in schema_version and not schema_version.startswith("yher.llm_sim_v2."):
        raise ValueError("v1 envelopes are forbidden in the v2 store")
    return dict(record)


def _contains_v1_run_envelope(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = canonical_key(key)
            if normalized == "run_id" or normalized.endswith("_run_id"):
                run_value = str(child).casefold()
                if "llm-personas-v1" in run_value or "llm_sim_v1" in run_value:
                    return True
            if _contains_v1_run_envelope(child):
                return True
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_v1_run_envelope(child) for child in value)
    return False


class V2Store:
    """A run-scoped store that cannot alias v1 or application state."""

    def __init__(
        self,
        base_root: str | Path,
        *,
        run_id: str = RUN_ID,
        phase: str = "pilot",
    ) -> None:
        if run_id != RUN_ID:
            raise ValueError(f"run_id must be exactly {RUN_ID}")
        phase_name = str(phase).strip().lower()
        if phase_name not in {"pilot", "main"}:
            raise ValueError("phase must be pilot or main")
        base = Path(base_root).expanduser().resolve(strict=False)
        _reject_path(base)
        if base.name == RUN_ID:
            run_root = base
        elif base.name in {"pilot", "main"} and base.parent.name == RUN_ID:
            run_root = base.parent
        else:
            run_root = base / RUN_ID
        _reject_path(run_root)
        self.run_id = RUN_ID
        self.phase = phase_name
        self.root = run_root / phase_name
        _reject_path(self.root)

    def _relative(self, relative: str | Path) -> Path:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("v2 store path must be relative")
        path = (self.root / rel).resolve(strict=False)
        if not path.is_relative_to(self.root):
            raise ValueError("v2 store path escapes phase root")
        _reject_path(path)
        return path

    def write_json(
        self,
        relative: str | Path,
        record: Mapping[str, Any],
        *,
        immutable: bool = False,
    ) -> Path:
        validated = _validate_record(record, phase=self.phase)
        path = self._relative(relative)
        payload = json.dumps(validated, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        if immutable and path.exists():
            if path.read_bytes() == payload:
                return path
            raise FileExistsError("immutable v2 artifact differs from existing bytes")
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
        path = self._relative(relative)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return _validate_record(value, phase=self.phase)

    def path(self, relative: str | Path) -> Path:
        return self._relative(relative)


SimulationV2Store = V2Store
V2RunStore = V2Store


def validate_v2_root(base_root: str | Path, *, run_id: str = RUN_ID, phase: str = "pilot") -> Path:
    return V2Store(base_root, run_id=run_id, phase=phase).root


__all__ = ["RUN_ID", "V2Store", "SimulationV2Store", "V2RunStore", "validate_v2_root"]
