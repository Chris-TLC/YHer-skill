"""Manifest-bound JSONL ingestion with byte-level provenance checks."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Generic, Mapping, TypeVar


class DatasetContractError(ValueError):
    """Raised when raw simulation artifacts violate their frozen contract."""


T = TypeVar("T")


@dataclass(frozen=True)
class ManifestDataset(Generic[T]):
    manifest: Mapping[str, Any]
    journeys: tuple[T, ...]
    raw_hash: str
    shard_paths: tuple[Path, ...]
    intended_journey_count: int
    invalid_record_count: int
    invalid_reasons: Mapping[str, int]
    invalid_primary_keys: tuple[tuple[object, ...], ...]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_manifest_dataset(
    manifest_path: Path | str,
    *,
    projector: Callable[[dict[str, Any]], T] | None = None,
    manifest_bytes: bytes | None = None,
) -> ManifestDataset[T | dict[str, Any]]:
    """Read exactly the shards named by ``manifest_path`` and verify their bytes."""

    path = Path(manifest_path).resolve()
    source_bytes = path.read_bytes() if manifest_bytes is None else manifest_bytes
    try:
        manifest = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise DatasetContractError(f"invalid manifest JSON: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("record_type") != (
        "confirmatory_run_manifest"
    ):
        raise DatasetContractError("manifest record_type is not confirmatory_run_manifest")
    _require_simulated_envelope(manifest, "run manifest")
    shard_specs = manifest.get("shards")
    if not isinstance(shard_specs, list) or not shard_specs:
        raise DatasetContractError("manifest must contain a non-empty shards list")

    convert: Callable[[dict[str, Any]], Any] = projector or (lambda row: row)
    root = path.parent
    seen_names: set[str] = set()
    shard_paths: list[Path] = []
    journeys: list[Any] = []
    invalid_reasons: Counter[str] = Counter()
    invalid_primary_keys: list[tuple[object, ...]] = []
    intended_journey_count = 0
    actual_shards: list[dict[str, str]] = []

    for spec in shard_specs:
        if not isinstance(spec, dict):
            raise DatasetContractError("each shard specification must be an object")
        filename = spec.get("filename")
        declared_sha = spec.get("sha256")
        declared_id = spec.get("shard_id")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".jsonl")
        ):
            raise DatasetContractError(f"unsafe shard filename: {filename!r}")
        if filename in seen_names:
            raise DatasetContractError(f"duplicate shard filename: {filename}")
        if not isinstance(declared_sha, str) or len(declared_sha) != 64:
            raise DatasetContractError(f"invalid shard SHA-256 for {filename}")
        if not isinstance(declared_id, str) or not declared_id:
            raise DatasetContractError(f"invalid shard_id for {filename}")
        seen_names.add(filename)
        shard_path = (root / filename).resolve()
        if shard_path.parent != root:
            raise DatasetContractError(f"shard escapes manifest directory: {filename}")

        digest = hashlib.sha256()
        records_digest = hashlib.sha256()
        parsed: list[dict[str, Any] | None] = []
        parse_reasons: dict[int, str] = {}
        with shard_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                if line_number > 1:
                    records_digest.update(line)
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed.append(None)
                    parse_reasons[line_number] = "invalid_json"
                    continue
                if not isinstance(value, dict):
                    parsed.append(None)
                    parse_reasons[line_number] = "non_object_record"
                    continue
                parsed.append(value)

        actual_sha = digest.hexdigest()
        if actual_sha != declared_sha:
            raise DatasetContractError(
                f"SHA-256 mismatch for {filename}: {actual_sha} != {declared_sha}"
            )
        if not parsed or parsed[0] is None:
            raise DatasetContractError(f"invalid shard manifest JSON in {filename}")
        if parsed[0].get("record_type") != "confirmatory_shard_manifest":
            raise DatasetContractError(f"missing shard manifest in {filename}")
        shard_manifest = parsed[0]
        records = parsed[1:]
        intended_journey_count += len(records)
        _require_simulated_envelope(shard_manifest, f"shard manifest {filename}")
        if shard_manifest.get("shard_id") != declared_id:
            raise DatasetContractError(f"shard_id mismatch for {filename}")
        if shard_manifest.get("record_count") != len(records):
            raise DatasetContractError(f"record_count mismatch for {filename}")
        if shard_manifest.get("records_sha256") != records_digest.hexdigest():
            raise DatasetContractError(f"records_sha256 mismatch for {filename}")
        for ordinal, record in enumerate(records):
            if record is None:
                invalid_reasons[parse_reasons[ordinal + 2]] += 1
                invalid_primary_keys.append(
                    _inferred_primary_key(declared_id, ordinal)
                )
                continue
            try:
                if record.get("record_type") != "confirmatory_journey":
                    raise DatasetContractError(f"unexpected record_type in {filename}")
                _require_simulated_envelope(record, f"journey in {filename}")
                events = record.get("events")
                if not isinstance(events, list):
                    raise DatasetContractError(
                        f"journey events are not a list in {filename}"
                    )
                for event in events:
                    if not isinstance(event, Mapping):
                        raise DatasetContractError(
                            f"journey event is not an object in {filename}"
                        )
                    _require_simulated_envelope(event, f"event in {filename}")
                journeys.append(convert(record))
            except DatasetContractError as exc:
                invalid_reasons[_quarantine_reason(exc)] += 1
                key = _primary_key(record)
                invalid_primary_keys.append(
                    key if key is not None else _inferred_primary_key(declared_id, ordinal)
                )
        shard_paths.append(shard_path)
        actual_shards.append({"filename": filename, "sha256": actual_sha})

    expected_count = manifest.get("expected_journey_count")
    if expected_count != intended_journey_count:
        raise DatasetContractError(
            f"journey intention count mismatch: {intended_journey_count} != {expected_count}"
        )
    raw_binding = {
        "manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "shards": actual_shards,
    }
    raw_hash = hashlib.sha256(_canonical_json_bytes(raw_binding)).hexdigest()
    return ManifestDataset(
        manifest=manifest,
        journeys=tuple(journeys),
        raw_hash=raw_hash,
        shard_paths=tuple(shard_paths),
        intended_journey_count=intended_journey_count,
        invalid_record_count=sum(invalid_reasons.values()),
        invalid_reasons=dict(sorted(invalid_reasons.items())),
        invalid_primary_keys=tuple(invalid_primary_keys),
    )


def _primary_key(record: Mapping[str, Any]) -> tuple[object, ...] | None:
    fields = ("target_node", "truth", "condition", "replicate", "arm")
    if any(field not in record for field in fields):
        return None
    return tuple(record[field] for field in fields)


def _inferred_primary_key(shard_id: str, ordinal: int) -> tuple[object, ...]:
    try:
        fields = dict(part.split("=", 1) for part in shard_id.split("|"))
        target = fields["target"]
        truth = fields["truth"]
        condition = fields["condition"]
    except (KeyError, ValueError) as exc:
        raise DatasetContractError(
            f"cannot infer invalid record key from shard_id {shard_id!r}"
        ) from exc
    arms = ("A", "B", "C")
    return (target, truth, condition, ordinal // len(arms), arms[ordinal % len(arms)])


def _quarantine_reason(exc: DatasetContractError) -> str:
    message = str(exc).lower()
    if "simulated-data envelope" in message:
        return "simulated_data_envelope"
    normalized = re.sub(r"[^a-z0-9]+", "_", message).strip("_")
    return normalized or "schema_invalid"


def _require_simulated_envelope(record: Mapping[str, Any], label: str) -> None:
    if record.get("simulated") is not True:
        raise DatasetContractError(f"{label} lacks the simulated-data envelope")
    for field in ("persona_id", "provider", "model_id"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise DatasetContractError(
                f"{label} lacks the simulated-data envelope field {field}"
            )
