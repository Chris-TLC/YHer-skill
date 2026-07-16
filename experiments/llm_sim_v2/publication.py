"""Offline publication adapters for Persona-v2 formal W3 artifacts."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


RUN_ID = "llm-personas-v2-dual"
_JUDGES = ("claude", "gpt")
_RUNTIME_RELATIVE = Path("experiments/llm_sim_v2/runtime_task_manifest.json")
_MAPPING_RELATIVE = Path(
    "experiments/llm_sim_v2/frozen_v0/target_option_mapping.json"
)


class PublicationAdapterError(ValueError):
    """Raised when an offline publication input cannot be proven safe and bound."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PublicationAdapterError("value is not canonical finite JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_value_bytes(data: bytes, *, label: str) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in rows:
            if key in value:
                raise PublicationAdapterError(
                    f"{label} contains a duplicate JSON key: {key}"
                )
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise PublicationAdapterError(
            f"{label} contains a non-finite JSON constant: {value}"
        )

    try:
        decoded = data.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except PublicationAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationAdapterError(f"{label} is not strict UTF-8 JSON") from exc
    return value


def _strict_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    value = _strict_json_value_bytes(data, label=label)
    if not isinstance(value, dict):
        raise PublicationAdapterError(f"{label} must be a JSON object")
    return value


def _validated_relative(value: str, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PublicationAdapterError(f"{label} path is empty")
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise PublicationAdapterError(f"{label} path is unsafe or non-canonical")
    return relative


def _read_descriptor(descriptor: int, *, label: str) -> bytes:
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise PublicationAdapterError(f"{label} must be a regular file")
            data = stream.read()
            after = os.fstat(stream.fileno())
    except PublicationAdapterError:
        raise
    except OSError as exc:
        raise PublicationAdapterError(f"cannot read {label}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(data) != after.st_size:
        raise PublicationAdapterError(f"{label} changed while it was read")
    return data


def _read_open_file(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicationAdapterError(
            f"{label} is missing, unsafe, or cannot be opened: {path}"
        ) from exc
    return _read_descriptor(descriptor, label=label)


def _resolve_directory(path: str | Path, *, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise PublicationAdapterError(f"{label} is missing or cannot resolve") from exc
    if not resolved.is_dir():
        raise PublicationAdapterError(f"{label} is not a directory")
    return resolved


def _read_relative(root: Path, relative: Path, *, label: str) -> bytes:
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise PublicationAdapterError(f"{label} path is unsafe")
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    file_flags = os.O_RDONLY
    file_flags |= getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise PublicationAdapterError(f"{label} root is unsafe") from exc
    try:
        for part in relative.parts[:-1]:
            try:
                next_descriptor = os.open(
                    part, directory_flags, dir_fd=directory_descriptor
                )
            except OSError as exc:
                raise PublicationAdapterError(
                    f"{label} contains a symlink or unsafe parent: {relative}"
                ) from exc
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        try:
            file_descriptor = os.open(
                relative.parts[-1], file_flags, dir_fd=directory_descriptor
            )
        except OSError as exc:
            raise PublicationAdapterError(
                f"{label} is missing, symlinked, or unsafe: {relative}"
            ) from exc
        return _read_descriptor(file_descriptor, label=label)
    finally:
        os.close(directory_descriptor)


def _read_path(path: str | Path, *, label: str) -> bytes:
    expanded = Path(path).expanduser()
    if expanded.name in {"", ".", ".."}:
        raise PublicationAdapterError(f"{label} path is invalid")
    try:
        parent = expanded.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicationAdapterError(f"{label} parent is missing") from exc
    return _read_open_file(parent / expanded.name, label=label)


def _new_destination(path: str | Path, *, label: str) -> Path:
    expanded = Path(path).expanduser()
    if ".." in expanded.parts or expanded.name in {"", ".", ".."}:
        raise PublicationAdapterError(f"{label} path is unsafe")
    try:
        expanded.parent.mkdir(parents=True, exist_ok=True)
        parent = expanded.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicationAdapterError(f"cannot prepare {label} parent") from exc
    destination = parent / expanded.name
    if os.path.lexists(destination):
        raise PublicationAdapterError(f"{label} already exists")
    return destination


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Directory fsync is not supported by every local filesystem.
        return


def _write_new_file(destination: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise PublicationAdapterError("output path already exists") from exc
        except OSError as exc:
            raise PublicationAdapterError("cannot atomically install output file") from exc
        _fsync_directory(destination.parent)
    except BaseException:
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, label: str
) -> str:
    advertised = payload.get(field)
    value = dict(payload)
    value.pop(field, None)
    computed = _canonical_sha256(value)
    if not isinstance(advertised, str) or advertised != computed:
        raise PublicationAdapterError(f"{label} self-hash drift")
    return computed


def _judge_case_ids(case_manifest: Mapping[str, Any]) -> list[str]:
    from .analyze import judge_input_bytes
    from .judge_protocol import JUDGE_PUBLIC_SCHEMA_KEYS, judge_protocol

    protocol = judge_protocol()
    amendment_bytes = Path(__file__).with_name(
        "judge_amendment_20260716.md"
    ).read_bytes()
    amendment_binding = {
        "path": "experiments/llm_sim_v2/judge_amendment_20260716.md",
        "sha256": hashlib.sha256(amendment_bytes).hexdigest(),
        "size": len(amendment_bytes),
    }
    if (
        case_manifest.get("schema_version")
        != "yher.llm_sim_v2.judge_case_manifest.v2"
        or case_manifest.get("simulated") is not True
        or case_manifest.get("run_id") != RUN_ID
        or case_manifest.get("analysis_population") != "main"
        or case_manifest.get("judge_protocol") != protocol
        or case_manifest.get("judge_protocol_sha256")
        != _canonical_sha256(protocol)
        or case_manifest.get("judge_amendment") != amendment_binding
        or case_manifest.get("question_field_whitelist")
        != sorted(JUDGE_PUBLIC_SCHEMA_KEYS)
        or case_manifest.get("target_metadata_exported") is not False
        or case_manifest.get("target_labels_exported") is not False
    ):
        raise PublicationAdapterError("judge case manifest envelope is invalid")
    case_hash = _verify_self_hash(
        case_manifest,
        field="case_manifest_sha256",
        label="judge case manifest",
    )
    del case_hash
    try:
        shared_input = judge_input_bytes(case_manifest)
    except (TypeError, ValueError) as exc:
        raise PublicationAdapterError("judge case manifest cases are invalid") from exc
    if case_manifest.get("shared_input_sha256") != hashlib.sha256(
        shared_input
    ).hexdigest():
        raise PublicationAdapterError("judge case manifest shared input hash drift")
    cases = case_manifest.get("cases")
    if not isinstance(cases, list):
        raise PublicationAdapterError("judge case manifest has no case list")
    case_ids: list[str] = []
    for row in cases:
        if not isinstance(row, Mapping):
            raise PublicationAdapterError("judge case row is not an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise PublicationAdapterError("judge case ID is invalid")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise PublicationAdapterError("judge case IDs are duplicated")
    return case_ids


def _strict_jsonl(data: bytes, *, label: str) -> list[dict[str, Any]]:
    if not data:
        return []
    lines = data.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise PublicationAdapterError(f"{label} contains a blank JSONL line")
    return [
        _strict_json_bytes(line, label=f"{label} line {index}")
        for index, line in enumerate(lines, start=1)
    ]


def build_judge_result_manifest(
    *,
    case_manifest_path: str | Path,
    execution_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Replay one isolated judge execution and publish its bound result."""

    destination = _new_destination(output_path, label="judge manifest output")
    case_manifest = _strict_json_bytes(
        _read_path(case_manifest_path, label="judge case manifest"),
        label="judge case manifest",
    )
    case_ids = _judge_case_ids(case_manifest)
    receipt_path = Path(execution_receipt_path).expanduser()
    receipt = _strict_json_bytes(
        _read_path(receipt_path, label="judge execution receipt"),
        label="judge execution receipt",
    )
    identity = receipt.get("identity")
    if not isinstance(identity, Mapping):
        raise PublicationAdapterError("judge execution identity is invalid")
    judge = identity.get("judge_family")
    transport = identity.get("transport")
    expected_transport = {"claude": "claude_cli", "gpt": "codex_cli"}
    if judge not in _JUDGES or transport != expected_transport[judge]:
        raise PublicationAdapterError(
            "judge execution family or production transport is invalid"
        )
    from .judge_execution import JudgeExecutionError, validate_execution_receipt

    try:
        receipt = validate_execution_receipt(
            receipt_path,
            case_manifest,
            str(judge),
        )
    except JudgeExecutionError as exc:
        raise PublicationAdapterError(
            "judge execution receipt or artifact replay failed"
        ) from exc
    receipt_identity = receipt.get("identity")
    if not isinstance(receipt_identity, Mapping):
        raise PublicationAdapterError("judge execution identity is invalid")
    execution_id = str(receipt_identity["execution_id"])
    expected_receipt_path = (
        destination.parent
        / "executions"
        / str(judge)
        / execution_id
        / "execution_receipt.json"
    )
    try:
        actual_receipt_path = receipt_path.resolve(strict=True)
    except OSError as exc:
        raise PublicationAdapterError("judge execution receipt cannot resolve") from exc
    if actual_receipt_path != expected_receipt_path.resolve(strict=False):
        raise PublicationAdapterError(
            "judge execution receipt must use executions/<judge>/<execution_id>/execution_receipt.json beside the result manifest"
        )
    normalized = receipt.get("normalized_results")
    if not isinstance(normalized, Mapping):
        raise PublicationAdapterError("judge normalized result binding is invalid")
    normalized_relative = _validated_relative(
        str(normalized.get("path") or ""),
        label="judge normalized result",
    )
    normalized_path = receipt_path.resolve().parent / normalized_relative
    raw_rows = _strict_jsonl(
        _read_path(normalized_path, label=f"{judge} normalized judge JSONL"),
        label=f"{judge} normalized judge JSONL",
    )
    raw_ids: list[str] = []
    results: list[dict[str, Any]] = []
    from .judge_protocol import validate_judge_output

    for index, row in enumerate(raw_rows, start=1):
        if set(row) != {"case_id", "output"}:
            raise PublicationAdapterError(
                f"{judge} normalized judge row {index} must contain exact case_id and output"
            )
        case_id = row.get("case_id")
        output = row.get("output")
        if not isinstance(case_id, str) or not case_id:
            raise PublicationAdapterError(f"{judge} raw judge case ID is invalid")
        if not isinstance(output, Mapping):
            raise PublicationAdapterError(f"{judge} raw judge output is invalid")
        try:
            normalized = validate_judge_output(output)
        except (TypeError, ValueError) as exc:
            raise PublicationAdapterError(f"{judge} judge output is invalid") from exc
        raw_ids.append(case_id)
        results.append({"case_id": case_id, "output": normalized})
    if raw_ids != case_ids:
        raise PublicationAdapterError(
            f"{judge} judge case coverage or order differs from case manifest"
        )
    manifest: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.judge_result_manifest.v2",
        "simulated": True,
        "run_id": RUN_ID,
        "judge": judge,
        "case_manifest_sha256": case_manifest["case_manifest_sha256"],
        "execution_receipt_path": (
            Path("executions")
            / str(judge)
            / execution_id
            / "execution_receipt.json"
        ).as_posix(),
        "execution_receipt": receipt,
        "results": results,
    }
    manifest["judge_result_manifest_sha256"] = _canonical_sha256(manifest)

    from .analyze import AnalysisContractError, ingest_judge_results

    slots: dict[str, Mapping[str, Any] | None] = {"claude": None, "gpt": None}
    slots[judge] = manifest
    try:
        ingest_judge_results(
            case_manifest,
            slots,
            judge_artifact_roots={str(judge): str(receipt_path.resolve().parent)},
        )
    except AnalysisContractError as exc:
        raise PublicationAdapterError(
            "judge result manifest is incompatible with analyzer ingestion"
        ) from exc
    _write_new_file(destination, _canonical_bytes(manifest) + b"\n")
    return manifest


def _result_tree_snapshot(result_root: Path) -> tuple[
    dict[str, Any], dict[str, Any], bytes, dict[str, bytes]
]:
    manifest_relative = Path("artifact_manifest.json")
    manifest_bytes = _read_relative(
        result_root, manifest_relative, label="analysis artifact manifest"
    )
    manifest = _strict_json_bytes(
        manifest_bytes, label="analysis artifact manifest"
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PublicationAdapterError("analysis artifact manifest is empty")
    if manifest.get("artifact_set_sha256") != _canonical_sha256(artifacts):
        raise PublicationAdapterError("analysis artifact manifest set hash drift")
    expected: dict[str, Mapping[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size"}:
            raise PublicationAdapterError("analysis artifact row is invalid")
        path_value = row.get("path")
        if not isinstance(path_value, str):
            raise PublicationAdapterError("analysis artifact path is invalid")
        relative = _validated_relative(path_value, label="artifact")
        name = relative.as_posix()
        if name == manifest_relative.as_posix() or name in expected:
            raise PublicationAdapterError("analysis artifact path is duplicated or unsafe")
        expected[name] = row

    actual: set[str] = set()
    for current_value, directory_names, file_names in os.walk(
        result_root, topdown=True, followlinks=False
    ):
        current = Path(current_value)
        for directory_name in directory_names:
            directory = current / directory_name
            if stat.S_ISLNK(os.lstat(directory).st_mode):
                raise PublicationAdapterError("analysis artifact tree contains a symlink")
        for file_name in file_names:
            path = current / file_name
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode):
                raise PublicationAdapterError("analysis artifact tree contains a symlink")
            if not stat.S_ISREG(metadata.st_mode):
                raise PublicationAdapterError(
                    "analysis artifact tree contains a non-regular file"
                )
            actual.add(path.relative_to(result_root).as_posix())
    permitted = set(expected) | {manifest_relative.as_posix()}
    if actual != permitted:
        raise PublicationAdapterError(
            "analysis result file set contains missing or unbound artifacts"
        )

    snapshots: dict[str, bytes] = {}
    for name, row in expected.items():
        data = _read_relative(
            result_root, Path(name), label=f"analysis artifact {name}"
        )
        if (
            row.get("sha256") != hashlib.sha256(data).hexdigest()
            or row.get("size") != len(data)
        ):
            raise PublicationAdapterError(f"analysis artifact hash drift: {name}")
        snapshots[name] = data
    if "analysis_results.json" not in snapshots:
        raise PublicationAdapterError(
            "analysis results are not bound by the artifact manifest"
        )
    results = _strict_json_bytes(
        snapshots["analysis_results.json"], label="analysis results"
    )
    if _read_relative(
        result_root, manifest_relative, label="analysis artifact manifest"
    ) != manifest_bytes:
        raise PublicationAdapterError("analysis artifact manifest changed during snapshot")
    for name, data in snapshots.items():
        if _read_relative(
            result_root, Path(name), label=f"analysis artifact {name}"
        ) != data:
            raise PublicationAdapterError(
                f"analysis artifact changed during snapshot: {name}"
            )
    return results, manifest, manifest_bytes, snapshots


def _mapping_identity(mapping: Mapping[str, Any]) -> tuple[str, str]:
    rows = mapping.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise PublicationAdapterError("mapping manifest rows are invalid")
    mapping_hash = _canonical_sha256(rows)
    mapped_targets = [
        {
            "item_id": row.get("item_id"),
            "failure_id": row.get("failure_id"),
            "target_option": row.get("target_option"),
        }
        for row in rows
        if row.get("status") == "mapped"
    ]
    target_hash = _canonical_sha256(mapped_targets)
    if (
        mapping.get("mapping_sha256") != mapping_hash
        or mapping.get("target_set_hash") != target_hash
    ):
        raise PublicationAdapterError("mapping manifest identity drift")
    return mapping_hash, target_hash


def _verify_bundle_identities(
    *,
    results: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    phase: Mapping[str, Any],
    runtime: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> None:
    phase_hash = _verify_self_hash(
        phase, field="phase_provenance_sha256", label="main phase provenance"
    )
    runtime_hash = _verify_self_hash(
        runtime,
        field="runtime_task_manifest_sha256",
        label="runtime task manifest",
    )
    mapping_hash, target_hash = _mapping_identity(mapping)
    input_proof = results.get("input_proof")
    phase_runtime = phase.get("runtime")
    phase_target = phase.get("target")
    sparse = results.get("sparse_mapping_descriptive")
    if not all(
        isinstance(value, Mapping)
        for value in (input_proof, phase_runtime, phase_target, sparse)
    ):
        raise PublicationAdapterError("Persona-v2 identity bindings are incomplete")
    if (
        input_proof.get("runtime_task_manifest_sha256") != runtime_hash
        or input_proof.get("phase_provenance_sha256") != phase_hash
        or artifact_manifest.get("runtime_task_manifest_sha256") != runtime_hash
        or artifact_manifest.get("phase_provenance_sha256") != phase_hash
        or artifact_manifest.get("target_set_hash") != target_hash
        or phase_runtime.get("runtime_task_manifest_sha256") != runtime_hash
        or phase_target.get("mapping_sha256") != mapping_hash
        or phase_target.get("target_set_hash") != target_hash
        or sparse.get("mapping_sha256") != mapping_hash
        or sparse.get("target_set_hash") != target_hash
    ):
        raise PublicationAdapterError(
            "analysis, main phase, runtime, or mapping identity drift"
        )


def _reject_noncanonical_judge_payloads(
    artifacts: Mapping[str, bytes],
    *,
    canonical_snapshot_artifacts: set[str],
) -> None:
    allowed_judge_objects = {
        "judge/case_manifest.json": "yher.llm_sim_v2.judge_case_manifest.v2",
        "judge/judge_analysis.json": "yher.llm_sim_v2.judge_analysis.v2",
    }
    snapshot_hashes: dict[str, set[str]] = {}
    for name in canonical_snapshot_artifacts:
        data = artifacts.get(name)
        if data is None:
            continue
        snapshot_hashes.setdefault(hashlib.sha256(data).hexdigest(), set()).add(name)

    def contains_judge_payload(value: Any) -> bool:
        if isinstance(value, Mapping):
            schema = value.get("schema_version")
            if isinstance(schema, str) and schema.startswith(
                "yher.llm_sim_v2.judge_"
            ):
                return True
            return any(contains_judge_payload(child) for child in value.values())
        if isinstance(value, list):
            return any(contains_judge_payload(child) for child in value)
        return False

    for name, data in artifacts.items():
        if name in canonical_snapshot_artifacts:
            continue
        matching_snapshot_paths = snapshot_hashes.get(
            hashlib.sha256(data).hexdigest(), set()
        )
        intentional_case_mirror = (
            name == "judge/case_manifest.json"
            and matching_snapshot_paths
            == {"judge-snapshots/run/case_manifest.json"}
        )
        if matching_snapshot_paths and not intentional_case_mirror:
            raise PublicationAdapterError(
                f"judge artifact must use its canonical snapshot path: {name}"
            )
        try:
            payload = _strict_json_value_bytes(
                data, label=f"analysis artifact {name}"
            )
        except PublicationAdapterError:
            if Path(name).suffix.lower() == ".json":
                raise
            continue
        if name == "analysis_results.json":
            continue
        schema = payload.get("schema_version") if isinstance(payload, Mapping) else None
        if name in allowed_judge_objects and allowed_judge_objects[name] == schema:
            continue
        if contains_judge_payload(payload):
            raise PublicationAdapterError(
                f"judge payload must use its canonical snapshot path: {name}"
            )


def _bundle_judge_roster(
    results: Mapping[str, Any], artifacts: Mapping[str, bytes]
) -> tuple[str, ...]:
    adjudication = results.get("judge_adjudication")
    analysis = adjudication.get("analysis") if isinstance(adjudication, Mapping) else None
    if not isinstance(analysis, Mapping) or analysis.get("expected_judges") != [
        "claude",
        "gpt",
    ]:
        raise PublicationAdapterError("Persona-v2 judge availability profile is invalid")
    profile = (
        analysis.get("status"),
        analysis.get("available_judges"),
        analysis.get("missing_judges"),
    )
    rosters = {
        ("complete", ("claude", "gpt"), ()): ("claude", "gpt"),
        ("partial_missing_judge", ("gpt",), ("claude",)): ("gpt",),
        ("not_applicable_zero_cases", (), ()): (),
    }
    try:
        normalized_profile = (
            profile[0],
            tuple(profile[1]) if isinstance(profile[1], list) else None,
            tuple(profile[2]) if isinstance(profile[2], list) else None,
        )
        judges = rosters[normalized_profile]
    except (KeyError, TypeError) as exc:
        raise PublicationAdapterError(
            "Persona-v2 judge availability profile is unsupported"
        ) from exc

    legacy_results = {
        name for name in artifacts if name.startswith("judge-results/")
    }
    if legacy_results:
        name = sorted(legacy_results)[0]
        family = Path(name).stem
        raise PublicationAdapterError(
            f"stray {family.title()} judge result artifact contradicts availability"
        )

    snapshot_manifest_name = "judge-snapshots/snapshot_manifest.json"
    snapshot_bytes = artifacts.get(snapshot_manifest_name)
    if snapshot_bytes is None:
        raise PublicationAdapterError(
            "canonical judge run snapshot manifest is missing"
        )
    snapshot = _strict_json_bytes(
        snapshot_bytes, label="judge run execution snapshot manifest"
    )
    family_slots = snapshot.get("family_slots")
    expected_slot_statuses = {
        ("claude", "gpt"): {"claude": "complete", "gpt": "complete"},
        ("gpt",): {"claude": "unavailable", "gpt": "complete"},
        (): {
            "claude": "not_applicable_zero_cases",
            "gpt": "not_applicable_zero_cases",
        },
    }[judges]
    if (
        snapshot.get("schema_version")
        != "yher.llm_sim_v2.judge_run_execution_snapshot_manifest.v1"
        or not isinstance(family_slots, Mapping)
        or set(family_slots) != set(_JUDGES)
        or {
            family: slot.get("status") if isinstance(slot, Mapping) else None
            for family, slot in family_slots.items()
        }
        != expected_slot_statuses
    ):
        raise PublicationAdapterError(
            "judge run snapshot family slots differ from the declared availability profile"
        )
    snapshot_files = snapshot.get("files")
    if not isinstance(snapshot_files, list):
        raise PublicationAdapterError("judge run snapshot file roster is invalid")
    expected_snapshot_artifacts = {snapshot_manifest_name}
    for row in snapshot_files:
        relative = row.get("path") if isinstance(row, Mapping) else None
        if not isinstance(relative, str) or not relative.startswith("run/"):
            raise PublicationAdapterError("judge run snapshot file roster is invalid")
        expected_snapshot_artifacts.add(f"judge-snapshots/{relative}")
    actual_snapshot_artifacts = {
        name for name in artifacts if name.startswith("judge-snapshots/")
    }
    if actual_snapshot_artifacts != expected_snapshot_artifacts:
        raise PublicationAdapterError(
            "judge run snapshot contains an unbound or missing artifact"
        )
    expected_results = {
        f"judge-snapshots/run/{judge}.json" for judge in judges
    }
    actual_results = {
        name
        for name in actual_snapshot_artifacts
        if name in {
            "judge-snapshots/run/claude.json",
            "judge-snapshots/run/gpt.json",
        }
    }
    if actual_results != expected_results:
        raise PublicationAdapterError(
            "judge result artifacts differ from the declared availability profile"
        )

    approved_family_artifacts = {
        f"judge/{family}_input.jsonl" for family in _JUDGES
    } | expected_results | expected_snapshot_artifacts
    for family in set(_JUDGES) - set(judges):
        family_pattern = re.compile(
            rf"(?:^|[/_.-]){re.escape(family)}(?:$|[/_.-])",
            flags=re.IGNORECASE,
        )
        for name in artifacts:
            if name in approved_family_artifacts:
                continue
            if family_pattern.search(name):
                raise PublicationAdapterError(
                    f"stray {family.title()} judge artifact contradicts availability"
                )
    _reject_noncanonical_judge_payloads(
        artifacts,
        canonical_snapshot_artifacts=expected_snapshot_artifacts,
    )
    return judges


def _write_staged_file(root: Path, relative: Path, data: bytes) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.path.lexists(destination):
        raise PublicationAdapterError("staged bundle path collision")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _install_directory_no_replace(staging: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing any existing destination."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        try:
            rename_exclusive = library.renamex_np
        except AttributeError as exc:
            raise PublicationAdapterError(
                "platform lacks exclusive atomic directory rename"
            ) from exc
        rename_exclusive.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        status = rename_exclusive(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        try:
            rename_exclusive = library.renameat2
        except AttributeError as exc:
            raise PublicationAdapterError(
                "platform lacks exclusive atomic directory rename"
            ) from exc
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        status = rename_exclusive(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
    else:
        raise PublicationAdapterError(
            "platform lacks exclusive atomic directory rename"
        )
    if status == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PublicationAdapterError("Persona-v2 bundle output already exists")
    raise PublicationAdapterError(
        "exclusive atomic directory rename failed: "
        f"{os.strerror(error_number)}"
    )


def build_persona_v2_bundle(
    *,
    result_dir: str | Path,
    main_phase_root: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Atomically snapshot formal W3 artifacts into a journal-binder bundle."""

    result_root = _resolve_directory(result_dir, label="final W3 result directory")
    main_root = _resolve_directory(main_phase_root, label="main phase root")
    repo = _resolve_directory(repo_root, label="repository root")
    if main_root.name != "main" or main_root.parent.name != RUN_ID:
        raise PublicationAdapterError(
            "main phase root must be the main directory under the Persona-v2 run"
        )
    destination = _new_destination(output_dir, label="Persona-v2 bundle output")
    if any(
        destination == root or destination.is_relative_to(root)
        for root in (result_root, main_root, repo)
    ):
        raise PublicationAdapterError("bundle output must not overlap an input root")

    results, artifact_manifest, artifact_manifest_bytes, artifacts = (
        _result_tree_snapshot(result_root)
    )
    judges = _bundle_judge_roster(results, artifacts)
    required_analysis_artifacts = {
        "analysis_results.json",
        "input_artifact_manifest.json",
        "judge-snapshots/snapshot_manifest.json",
        *(f"judge-snapshots/run/{judge}.json" for judge in judges),
    }
    if not required_analysis_artifacts.issubset(artifacts):
        raise PublicationAdapterError(
            "formal W3 artifacts do not bind the input manifest and declared judge profile"
        )
    phase_bytes = _read_relative(
        main_root, Path("phase_provenance.json"), label="main phase provenance"
    )
    runtime_bytes = _read_relative(
        repo, _RUNTIME_RELATIVE, label="runtime task manifest"
    )
    mapping_bytes = _read_relative(
        repo, _MAPPING_RELATIVE, label="target-option mapping manifest"
    )
    phase = _strict_json_bytes(phase_bytes, label="main phase provenance")
    runtime = _strict_json_bytes(runtime_bytes, label="runtime task manifest")
    mapping = _strict_json_bytes(mapping_bytes, label="target-option mapping manifest")
    _verify_bundle_identities(
        results=results,
        artifact_manifest=artifact_manifest,
        phase=phase,
        runtime=runtime,
        mapping=mapping,
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    os.chmod(staging, 0o700)
    installed = False
    try:
        for name, data in artifacts.items():
            _write_staged_file(staging, Path("analysis") / name, data)
            if name.startswith("judge-snapshots/"):
                _write_staged_file(staging, Path(name), data)
        analysis_manifest_path = _write_staged_file(
            staging,
            Path("analysis/artifact_manifest.json"),
            artifact_manifest_bytes,
        )
        phase_path = _write_staged_file(
            staging, Path("evidence/phase_provenance.json"), phase_bytes
        )
        runtime_path = _write_staged_file(
            staging, Path("evidence/runtime_task_manifest.json"), runtime_bytes
        )
        mapping_path = _write_staged_file(
            staging, Path("evidence/target_option_mapping.json"), mapping_bytes
        )
        role_paths = {
            "analysis_results": staging / "analysis/analysis_results.json",
            "analysis_artifact_manifest": analysis_manifest_path,
            "analysis_input_artifact_manifest": (
                staging / "analysis/input_artifact_manifest.json"
            ),
            "phase_provenance": phase_path,
            "runtime_task_manifest": runtime_path,
            "mapping_manifest": mapping_path,
            **{
                f"{judge}_judge_result_manifest": (
                    staging / f"judge-snapshots/run/{judge}.json"
                )
                for judge in judges
            },
            **{
                "judge_run_execution_snapshot_manifest": (
                    staging / "judge-snapshots/snapshot_manifest.json"
                )
            },
        }
        binding_manifest = {
            "schema_version": "yher.journal_binder.persona_v2_bundle.v1",
            "simulated": True,
            "run_id": RUN_ID,
            "analysis_population": "main",
            "files": {
                role: {
                    "path": path.relative_to(staging).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for role, path in sorted(role_paths.items())
            },
        }
        _write_staged_file(
            staging,
            Path("binding_manifest.json"),
            _canonical_bytes(binding_manifest) + b"\n",
        )

        from experiments import journal_binder

        try:
            journal_binder.bind_persona_v2_artifacts(
                staging, allow_fixture=allow_fixture
            )
        except journal_binder.BinderError as exc:
            raise PublicationAdapterError(
                f"Persona-v2 bundle failed journal binder validation: {exc}"
            ) from exc

        source_checks = (
            (
                _read_relative(
                    main_root,
                    Path("phase_provenance.json"),
                    label="main phase provenance",
                ),
                phase_bytes,
            ),
            (
                _read_relative(repo, _RUNTIME_RELATIVE, label="runtime task manifest"),
                runtime_bytes,
            ),
            (
                _read_relative(
                    repo, _MAPPING_RELATIVE, label="target-option mapping manifest"
                ),
                mapping_bytes,
            ),
        )
        if any(current != frozen for current, frozen in source_checks):
            raise PublicationAdapterError(
                "main phase, runtime, or mapping changed during bundle construction"
            )
        if os.path.lexists(destination):
            raise PublicationAdapterError("Persona-v2 bundle output already exists")
        _fsync_directory(staging)
        _install_directory_no_replace(staging, destination)
        installed = True
        _fsync_directory(destination.parent)
        return binding_manifest
    except PublicationAdapterError:
        raise
    except OSError as exc:
        raise PublicationAdapterError("cannot atomically publish Persona-v2 bundle") from exc
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build offline Persona-v2 W3 publication artifacts."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    judge = commands.add_parser(
        "judge-manifest", help="Bind one fully replayed frozen-judge execution."
    )
    judge.add_argument("--case-manifest", required=True, type=Path)
    judge.add_argument("--execution-receipt", required=True, type=Path)
    judge.add_argument("--output", required=True, type=Path)
    persona = commands.add_parser(
        "persona-bundle", help="Build one formal W3 journal-binder bundle."
    )
    persona.add_argument("--result-dir", required=True, type=Path)
    persona.add_argument("--main-phase-root", required=True, type=Path)
    persona.add_argument("--repo-root", required=True, type=Path)
    persona.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "judge-manifest":
            manifest = build_judge_result_manifest(
                case_manifest_path=args.case_manifest,
                execution_receipt_path=args.execution_receipt,
                output_path=args.output,
            )
            summary = {
                "command": args.command,
                "judge": manifest["judge"],
                "result_count": len(manifest["results"]),
                "output": str(args.output),
            }
        else:
            manifest = build_persona_v2_bundle(
                result_dir=args.result_dir,
                main_phase_root=args.main_phase_root,
                repo_root=args.repo_root,
                output_dir=args.output_dir,
            )
            summary = {
                "command": args.command,
                "role_count": len(manifest["files"]),
                "output_dir": str(args.output_dir),
            }
    except PublicationAdapterError as exc:
        parser.exit(2, f"publication adapter error: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PublicationAdapterError",
    "build_judge_result_manifest",
    "build_persona_v2_bundle",
    "build_parser",
    "main",
]
