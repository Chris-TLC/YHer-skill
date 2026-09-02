"""Envelope validation, hash-valid resume, and atomic deterministic shards."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.s0_census import (
    require_simulated_event_envelope,
    require_simulation_output_path,
)

from .config import REPO_ROOT, canonical_json_bytes


def write_shards_atomic(
    shards: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_dir: str | Path,
    config_sha256: str,
    workers: int,
    resume: bool,
    repo_root: str | Path = REPO_ROOT,
    temp_root: str | Path = "/tmp/yher_sprint2",
    manifest_metadata: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    output = require_simulation_output_path(
        output_dir,
        repo_root=repo_root,
        temp_root=temp_root,
    )
    if workers < 1:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(manifest_metadata or {})

    def write_one(entry: tuple[str, Sequence[Mapping[str, Any]]]) -> str:
        shard_id, rows = entry
        model_ids = sorted({str(record["model_id"]) for record in rows})
        expected_manifest = {
            **metadata,
            "shard_id": shard_id,
            "model_id": ";".join(model_ids)
            if model_ids
            else "programmatic-empty-shard",
        }
        path = require_simulation_output_path(
            output / _shard_filename(shard_id),
            repo_root=repo_root,
            temp_root=temp_root,
        )
        if resume and validate_shard(
            path,
            config_sha256=config_sha256,
            expected_manifest=expected_manifest,
        ):
            return "skipped"
        content = build_shard_bytes(
            shard_id,
            rows,
            config_sha256=config_sha256,
            manifest_metadata=metadata,
        )
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        try:
            with temporary.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if not validate_shard(
                temporary,
                config_sha256=config_sha256,
                expected_manifest=expected_manifest,
            ):
                raise ValueError(f"temporary shard validation failed: {shard_id}")
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return "written"

    ordered = sorted(shards.items(), key=lambda entry: entry[0])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(write_one, ordered))
    return {
        "written": sum(result == "written" for result in results),
        "skipped": sum(result == "skipped" for result in results),
    }


def build_shard_bytes(
    shard_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    config_sha256: str,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> bytes:
    sorted_records = sorted((dict(record) for record in records), key=_record_key)
    for record in sorted_records:
        _validate_record_envelopes(record)
    payload = b"".join(canonical_json_bytes(record) + b"\n" for record in sorted_records)
    model_ids = sorted({str(record["model_id"]) for record in sorted_records})
    manifest = {
        **dict(manifest_metadata or {}),
        "simulated": True,
        "persona_id": f"confirmatory-shard:{shard_id}",
        "provider": "programmatic",
        "model_id": ";".join(model_ids) if model_ids else "programmatic-empty-shard",
        "record_type": "confirmatory_shard_manifest",
        "shard_id": shard_id,
        "config_sha256": config_sha256,
        "record_count": len(sorted_records),
        "records_sha256": hashlib.sha256(payload).hexdigest(),
    }
    require_simulated_event_envelope(manifest)
    return canonical_json_bytes(manifest) + b"\n" + payload


def validate_shard(
    path: str | Path,
    *,
    config_sha256: str,
    expected_manifest: Mapping[str, Any] | None = None,
    require_isolation_attestation: bool = False,
) -> bool:
    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        lines = candidate.read_bytes().splitlines(keepends=True)
        if not lines:
            return False
        manifest = json.loads(lines[0])
        require_simulated_event_envelope(manifest)
        if manifest.get("record_type") != "confirmatory_shard_manifest":
            return False
        if manifest.get("config_sha256") != config_sha256:
            return False
        for key, expected in (expected_manifest or {}).items():
            if manifest.get(key) != expected:
                return False
        if require_isolation_attestation:
            assertion = manifest.get("protected_filesystem_assertion")
            if not isinstance(assertion, Mapping):
                return False
            if assertion.get("unchanged") is not True:
                return False
            if assertion.get("before_sha256") != assertion.get("after_sha256"):
                return False
        payload = b"".join(lines[1:])
        if manifest.get("records_sha256") != hashlib.sha256(payload).hexdigest():
            return False
        records = [json.loads(line) for line in lines[1:] if line.strip()]
        if int(manifest.get("record_count", -1)) != len(records):
            return False
        for record in records:
            _validate_record_envelopes(record)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def read_shard_records(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[1:] if line.strip()]


def read_shard_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.loads(handle.readline())
    if not isinstance(value, dict):
        raise ValueError("shard manifest must be an object")
    return value


def shard_file_path(output_dir: str | Path, shard_id: str) -> Path:
    return Path(output_dir) / _shard_filename(shard_id)


def write_run_manifest_atomic(
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    repo_root: str | Path = REPO_ROOT,
    temp_root: str | Path = "/tmp/yher_sprint2",
) -> Path:
    output = require_simulation_output_path(
        output_dir,
        repo_root=repo_root,
        temp_root=temp_root,
    )
    record = dict(manifest)
    require_simulated_event_envelope(record)
    output.mkdir(parents=True, exist_ok=True)
    path = require_simulation_output_path(
        output / "manifest.json",
        repo_root=repo_root,
        temp_root=temp_root,
    )
    content = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _validate_record_envelopes(record: Mapping[str, Any]) -> None:
    require_simulated_event_envelope(record)
    for event in record.get("events", ()):
        if not isinstance(event, Mapping):
            raise ValueError("journey events must be mappings")
        require_simulated_event_envelope(event)


def _record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("record_type", "")),
        str(record.get("target_node", "")),
        str(record.get("truth", "")),
        str(record.get("condition", "")),
        int(record.get("replicate", -1)),
        str(record.get("arm", "")),
        int(record.get("position", -1)),
        canonical_json_bytes(record),
    )


def _shard_filename(shard_id: str) -> str:
    digest = hashlib.sha256(shard_id.encode("utf-8")).hexdigest()[:20]
    return f"shard-{digest}.jsonl"
