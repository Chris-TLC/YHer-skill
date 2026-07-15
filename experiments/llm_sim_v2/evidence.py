"""Append-only evidence receipts for Persona-v2 provider collection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any
import uuid


RUN_ID = "llm-personas-v2-dual"
_PHASES = {"pilot", "main"}
_EVENT_SCHEMA = "yher.llm_sim_v2.provider_evidence_event.v1"
REVIEWED_CARRIED_LEDGER_SHA256 = (
    "87ff7a3d08df64d67df2372188d6f707ddb4deb295a85300aae0b6190d48be35"
)
REVIEWED_LEGACY_RECEIPT_SHA256 = (
    "2ea161a0f29e3a8bac1eeee38cb238f7cb722a38b809997172b933e063e82999"
)
REVIEWED_LEGACY_RECORD_SET_SHA256 = (
    "1490e7d35410717614ba9fd2d37e1eda12d824fa247bff29ae29ff9a0b6c58c0"
)
REVIEWED_LEGACY_KNOWN_COST_YUAN = 1.91386592


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_response_content(content: str) -> dict[str, Any]:
    """Bind the exact UTF-8 content string consumed by the strict parser."""

    if not isinstance(content, str):
        raise TypeError("provider response content must be a string")
    payload = content.encode("utf-8")
    return {
        "provider_response_received": True,
        "response_content": content,
        "response_content_utf8_bytes": len(payload),
        "response_content_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_response_content_binding(attempt: Mapping[str, Any]) -> None:
    if attempt.get("provider_response_received") is not True:
        raise ValueError("attempt does not declare a provider response")
    content = attempt.get("response_content")
    if not isinstance(content, str):
        raise ValueError("response attempt lacks exact provider content")
    expected = bind_response_content(content)
    if any(attempt.get(key) != value for key, value in expected.items()):
        raise ValueError("response attempt content digest or byte count drifted")


def validate_response_attempt_replay(
    attempt: Mapping[str, Any],
    *,
    condition: str,
    option_keys: set[str],
    requested_model: str,
) -> dict[str, Any] | None:
    """Replay one response-bearing attempt under the runner's strict parser."""

    validate_response_content_binding(attempt)
    from .runner import InvalidProviderOutput, parse_provider_output

    status = attempt.get("status")
    error_category = attempt.get("error_category")
    returned_model = str(attempt.get("model_returned") or "")
    if status == "failed" and error_category == "invalid_schema":
        if returned_model != requested_model:
            raise ValueError("invalid-schema attempt contains model drift")
        try:
            parse_provider_output(
                str(attempt["response_content"]),
                condition=condition,
                option_keys=option_keys,
            )
        except InvalidProviderOutput:
            return None
        raise ValueError("invalid-schema response passed the strict parser")
    if status == "response":
        if returned_model != requested_model:
            return None
        try:
            return parse_provider_output(
                str(attempt["response_content"]),
                condition=condition,
                option_keys=option_keys,
            )
        except InvalidProviderOutput as exc:
            raise ValueError("successful response failed the strict parser") from exc
    raise ValueError("response attempt status semantics do not reconcile")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_phase_root(phase_root: Path | str) -> tuple[Path, str]:
    root = Path(phase_root).expanduser().resolve(strict=False)
    phase = root.name
    if phase not in _PHASES or root.parent.name != RUN_ID:
        raise ValueError("evidence root must be the frozen run's pilot or main root")
    return root, phase


def _atomic_json(path: Path, value: Mapping[str, Any], *, immutable: bool) -> Path:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() == payload:
            return path
        raise FileExistsError(f"immutable evidence artifact differs: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        temporary.write_bytes(payload)
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise FileExistsError(
                        f"immutable evidence artifact differs: {path}"
                    )
        else:
            os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def snapshot_phase_store(phase_root: Path | str) -> dict[str, Any]:
    """Hash every extant phase file except derived phase-receipt copies."""

    root, phase = _validate_phase_root(phase_root)
    rows: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("phase store evidence cannot contain symlinks")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative.startswith("evidence/phase_receipts/"):
                continue
            if path.name.startswith(".") and ".tmp-" in path.name:
                continue
            rows.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": "yher.llm_sim_v2.phase_store_snapshot.v1",
        "run_id": RUN_ID,
        "phase": phase,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "file_set_sha256": canonical_sha256(rows),
    }


def build_provider_record_set(
    phase_root: Path | str,
    *,
    provider: str,
    expected_task_ids: Sequence[str],
) -> dict[str, Any]:
    root, phase = _validate_phase_root(phase_root)
    provider_name = str(provider).strip().lower()
    expected = [str(task_id) for task_id in expected_task_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("provider record-set roster contains duplicate task IDs")
    record_root = root / "records" / provider_name
    if record_root.is_dir():
        entries = list(record_root.iterdir())
        if any(
            path.is_symlink() or not path.is_file() or path.suffix != ".json"
            for path in entries
        ):
            raise ValueError("provider record root contains an unbound entry")
        paths = sorted(entries)
    else:
        paths = []
    rows: list[dict[str, Any]] = []
    observed: list[str] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"provider record is not valid JSON: {path}") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"provider record is not an object: {path}")
        task_id = str(record.get("task_id") or path.stem)
        if task_id != path.stem:
            raise ValueError("provider record filename and task_id differ")
        attempts = record.get("attempts")
        attempt_rows = attempts if isinstance(attempts, list) else []
        observed.append(task_id)
        rows.append(
            {
                "task_id": task_id,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "attempt_count": len(attempt_rows),
                "response_attempt_count": sum(
                    isinstance(attempt, Mapping)
                    and attempt.get("provider_response_received") is True
                    for attempt in attempt_rows
                ),
            }
        )
    unexpected = sorted(set(observed) - set(expected))
    missing = [task_id for task_id in expected if task_id not in set(observed)]
    payload = {
        "schema_version": "yher.llm_sim_v2.provider_record_set.v1",
        "run_id": RUN_ID,
        "phase": phase,
        "provider": provider_name,
        "expected_task_count": len(expected),
        "expected_task_ids_sha256": canonical_sha256(expected),
        "record_count": len(rows),
        "missing_task_ids": missing,
        "unexpected_task_ids": unexpected,
        "records": rows,
    }
    payload["record_set_sha256"] = canonical_sha256(payload)
    return payload


class ProviderEvidenceLedger:
    """One append-only, hash-chained event stream per provider and phase."""

    def __init__(
        self,
        phase_root: Path | str,
        *,
        run_id: str,
        phase: str,
        provider: str,
    ) -> None:
        root, phase_name = _validate_phase_root(phase_root)
        if run_id != RUN_ID or str(phase).strip().lower() != phase_name:
            raise ValueError("provider evidence scope differs from the frozen run")
        self.phase_root = root
        self.phase = phase_name
        self.provider = str(provider).strip().lower()
        if not self.provider:
            raise ValueError("provider evidence requires a provider")
        self.event_root = root / "evidence" / "provider_events" / self.provider
        self._lock = threading.Lock()
        self._process_lock_guard = threading.RLock()
        self._process_lock_fd: int | None = None
        self._process_lock_depth = 0
        self._active_invocation_id: str | None = None

    def _enter_provider_lock(self) -> None:
        self._process_lock_guard.acquire()
        try:
            if self._process_lock_depth == 0:
                lock_root = self.phase_root / "evidence" / "provider_locks"
                lock_root.mkdir(parents=True, exist_ok=True)
                lock_path = lock_root / f"{self.provider}.lock"
                if lock_path.is_symlink():
                    raise ValueError("provider evidence lock cannot be a symlink")
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                except BaseException:
                    os.close(descriptor)
                    raise
                self._process_lock_fd = descriptor
            self._process_lock_depth += 1
        except BaseException:
            self._process_lock_guard.release()
            raise

    def _exit_provider_lock(self) -> None:
        try:
            if self._process_lock_depth <= 0:
                raise RuntimeError("provider evidence lock is not held")
            self._process_lock_depth -= 1
            if self._process_lock_depth == 0:
                descriptor = self._process_lock_fd
                self._process_lock_fd = None
                if descriptor is not None:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(descriptor)
        finally:
            self._process_lock_guard.release()

    @contextmanager
    def provider_lock(self):
        """Hold the OS-backed provider lock across one store transaction."""

        self._enter_provider_lock()
        try:
            yield self
        finally:
            self._exit_provider_lock()

    def _read_events(self, *, allow_open_invocation: bool) -> list[dict[str, Any]]:
        if self.event_root.is_dir():
            entries = list(self.event_root.iterdir())
            if any(
                path.is_symlink() or not path.is_file() or path.suffix != ".json"
                for path in entries
            ):
                raise ValueError("provider evidence stream contains an unbound entry")
            paths = sorted(entries)
        else:
            paths = []
        events: list[dict[str, Any]] = []
        previous: str | None = None
        for index, path in enumerate(paths):
            try:
                event = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"provider evidence event is invalid: {path}") from exc
            if not isinstance(event, Mapping):
                raise ValueError("provider evidence event must be an object")
            payload = dict(event)
            advertised = payload.pop("event_sha256", None)
            if (
                event.get("schema_version") != _EVENT_SCHEMA
                or event.get("simulated") is not True
                or event.get("run_id") != RUN_ID
                or event.get("phase") != self.phase
                or event.get("provider") != self.provider
                or event.get("event_index") != index
                or event.get("previous_event_sha256") != previous
                or advertised != canonical_sha256(payload)
                or path.name != f"{index:06d}-{advertised}.json"
            ):
                raise ValueError("provider evidence hash chain is invalid")
            events.append(dict(event))
            previous = str(advertised)
        active_id: str | None = None
        active_start: Mapping[str, Any] | None = None
        active_calls: list[Mapping[str, Any]] = []
        for event in events:
            event_type = event.get("event_type")
            invocation_id = str(event.get("invocation_id") or "")
            if event_type == "invocation_started":
                if active_id is not None or not invocation_id:
                    raise ValueError("provider evidence invocation state is invalid")
                active_id = invocation_id
                active_start = event
                active_calls = []
                continue
            if event_type == "provider_call_started":
                if active_id is None or invocation_id != active_id:
                    raise ValueError("provider call event lacks its active invocation")
                active_calls.append(event)
                continue
            if event_type == "invocation_finished":
                call_hashes = [row["event_sha256"] for row in active_calls]
                valid_finish = (
                    active_id is not None
                    and active_start is not None
                    and invocation_id == active_id
                    and event.get("start_event_sha256")
                    == active_start.get("event_sha256")
                    and event.get("status")
                    in {"complete", "interrupted", "unavailable", "failed"}
                    and event.get("provider_call_count") == len(active_calls)
                    and event.get("provider_call_event_sha256s") == call_hashes
                    and event.get("provider_call_count_before")
                    == active_start.get("provider_call_count_before")
                    and event.get("provider_call_count_after")
                    == int(active_start.get("provider_call_count_before") or 0)
                    + len(active_calls)
                    and event.get("before_store") == active_start.get("before_store")
                )
                if not valid_finish:
                    raise ValueError("provider evidence final receipt is invalid")
                active_id = None
                active_start = None
                active_calls = []
                continue
            raise ValueError("provider evidence event type is invalid")
        if active_id is not None and not allow_open_invocation:
            raise ValueError("provider evidence history has an unmatched invocation")
        return events

    def read_events(self) -> list[dict[str, Any]]:
        return self._read_events(allow_open_invocation=False)

    def _assert_no_unresolved_provider_calls(
        self,
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        call_counts = Counter(
            (
                str(event.get("task_id") or ""),
                event.get("attempt"),
                event.get("model"),
                event.get("request_max_tokens"),
                event.get("wire_message_sha256"),
            )
            for event in events
            if event.get("event_type") == "provider_call_started"
        )
        if not call_counts:
            return
        attempt_counts: Counter[tuple[Any, ...]] = Counter()
        record_root = self.phase_root / "records" / self.provider
        if record_root.is_dir():
            entries = list(record_root.iterdir())
            if any(
                path.is_symlink() or not path.is_file() or path.suffix != ".json"
                for path in entries
            ):
                raise ValueError("provider record store contains an unbound entry")
            for path in sorted(entries):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError("provider record store is invalid") from exc
                if not isinstance(record, Mapping):
                    raise ValueError("provider record store is invalid")
                attempts = record.get("attempts")
                if not isinstance(attempts, list):
                    raise ValueError("provider record lacks attempt evidence")
                for attempt in attempts:
                    if not isinstance(attempt, Mapping):
                        raise ValueError("provider record attempt is invalid")
                    attempt_counts[
                        (
                            str(record.get("task_id") or ""),
                            attempt.get("attempt"),
                            record.get("requested_model"),
                            attempt.get("request_max_tokens"),
                            record.get("wire_message_sha256"),
                        )
                    ] += 1
        unresolved = call_counts - attempt_counts
        if unresolved:
            raise ValueError(
                "unresolved provider call blocks same-epoch resume; "
                "a reviewed new evidence epoch is required"
            )

    def _append(self, event_type: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            events = self._read_events(allow_open_invocation=True)
            previous = events[-1]["event_sha256"] if events else None
            event: dict[str, Any] = {
                "schema_version": _EVENT_SCHEMA,
                "simulated": True,
                "run_id": RUN_ID,
                "phase": self.phase,
                "provider": self.provider,
                "event_index": len(events),
                "previous_event_sha256": previous,
                "event_type": str(event_type),
                "recorded_at_utc": _utc_now(),
                **dict(fields),
            }
            event["event_sha256"] = canonical_sha256(event)
            path = self.event_root / (
                f"{len(events):06d}-{event['event_sha256']}.json"
            )
            _atomic_json(path, event, immutable=True)
            return event

    def begin_invocation(
        self,
        *,
        expected_task_ids: Sequence[str],
        resumed_task_ids: Sequence[str],
    ) -> dict[str, Any]:
        owns_provider_lock = self._process_lock_depth == 0
        if owns_provider_lock:
            self._enter_provider_lock()
        try:
            expected = [str(value) for value in expected_task_ids]
            resumed = [str(value) for value in resumed_task_ids]
            if len(expected) != len(set(expected)) or not set(resumed) <= set(expected):
                raise ValueError("invocation task roster or resumed subset is invalid")
            invocation_id = uuid.uuid4().hex
            before_events = self.read_events()
            self._assert_no_unresolved_provider_calls(before_events)
            before_call_count = sum(
                event.get("event_type") == "provider_call_started"
                for event in before_events
            )
            before_store = snapshot_phase_store(self.phase_root)
            started = self._append(
                "invocation_started",
                {
                    "invocation_id": invocation_id,
                    "expected_task_count": len(expected),
                    "expected_task_ids_sha256": canonical_sha256(expected),
                    "resumed_record_count": len(resumed),
                    "resumed_task_ids_sha256": canonical_sha256(resumed),
                    "provider_call_count_before": before_call_count,
                    "before_store": before_store,
                },
            )
            self._active_invocation_id = invocation_id
            return {
                "invocation_id": invocation_id,
                "start_event_sha256": started["event_sha256"],
                "start_event_index": started["event_index"],
                "expected_task_count": len(expected),
                "resumed_record_count": len(resumed),
                "provider_call_count_before": before_call_count,
                "before_store": before_store,
                "owns_provider_lock": owns_provider_lock,
            }
        except BaseException:
            if owns_provider_lock:
                self._exit_provider_lock()
            raise

    def record_provider_call_started(
        self,
        *,
        task_id: str,
        attempt: int,
        model: str,
        request_max_tokens: int,
        wire_message_sha256: str,
    ) -> dict[str, Any]:
        if self._active_invocation_id is None:
            raise RuntimeError("provider call occurred outside an evidence invocation")
        return self._append(
            "provider_call_started",
            {
                "invocation_id": self._active_invocation_id,
                "task_id": str(task_id),
                "attempt": int(attempt),
                "model": str(model),
                "request_max_tokens": int(request_max_tokens),
                "wire_message_sha256": str(wire_message_sha256),
            },
        )

    def finish_invocation(
        self,
        invocation: Mapping[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        try:
            invocation_id = str(invocation.get("invocation_id") or "")
            if not invocation_id or invocation_id != self._active_invocation_id:
                raise ValueError("provider evidence invocation token is not active")
            events = self._read_events(allow_open_invocation=True)
            calls = [
                event
                for event in events
                if event.get("event_type") == "provider_call_started"
                and event.get("invocation_id") == invocation_id
            ]
            expected_count = int(invocation["expected_task_count"])
            resumed_count = int(invocation["resumed_record_count"])
            if expected_count and resumed_count == expected_count:
                invocation_kind = "resume"
            elif resumed_count:
                invocation_kind = "mixed_resume"
            else:
                invocation_kind = "initial_collection"
            after_store = snapshot_phase_store(self.phase_root)
            finished = self._append(
                "invocation_finished",
                {
                    "invocation_id": invocation_id,
                    "start_event_sha256": invocation["start_event_sha256"],
                    "status": str(status),
                    "invocation_kind": invocation_kind,
                    "expected_task_count": expected_count,
                    "resumed_record_count": resumed_count,
                    "provider_call_count": len(calls),
                    "provider_call_count_before": int(
                        invocation["provider_call_count_before"]
                    ),
                    "provider_call_count_after": int(
                        invocation["provider_call_count_before"]
                    )
                    + len(calls),
                    "provider_call_event_sha256s": [
                        event["event_sha256"] for event in calls
                    ],
                    "before_store": dict(invocation["before_store"]),
                    "after_store": after_store,
                },
            )
            self._active_invocation_id = None
            return finished
        finally:
            if invocation.get("owns_provider_lock") is True:
                self._exit_provider_lock()


def _load_bound_phase_provenance(
    root: Path,
    *,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = root / "phase_provenance.json"
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("phase receipt requires valid phase provenance") from exc
    if not isinstance(artifact, Mapping):
        raise ValueError("phase receipt requires object phase provenance")
    payload = dict(artifact)
    advertised = payload.pop("phase_provenance_sha256", None)
    if (
        artifact.get("schema_version")
        != "yher.llm_sim_v2.phase_provenance.v1"
        or artifact.get("simulated") is not True
        or artifact.get("run_id") != RUN_ID
        or artifact.get("phase") != root.name
        or artifact.get("analysis_population") != root.name
        or advertised != canonical_sha256(payload)
    ):
        raise ValueError("phase provenance identity is invalid")
    if supplied is not None and dict(supplied) != dict(artifact):
        raise ValueError("stored phase provenance differs from the active verified value")
    if artifact.get("collection_mode") == "formal":
        budget = artifact.get("budget")
        if not isinstance(budget, Mapping) or not (
            budget.get("carried_forward_cost_ledger_sha256")
            == REVIEWED_CARRIED_LEDGER_SHA256
            and budget.get("source_phase_receipt_sha256")
            == REVIEWED_LEGACY_RECEIPT_SHA256
            and budget.get("source_record_set_sha256")
            == REVIEWED_LEGACY_RECORD_SET_SHA256
            and _same_amount(
                budget.get("carried_forward_known_cost_yuan"),
                REVIEWED_LEGACY_KNOWN_COST_YUAN,
            )
            and _same_amount(
                budget.get("carried_forward_unknown_reserve_yuan"), 0.0
            )
            and _same_amount(
                budget.get("carried_forward_total_accounted_cost_yuan"),
                REVIEWED_LEGACY_KNOWN_COST_YUAN,
            )
        ):
            raise ValueError("formal phase provenance lacks reviewed carried cost")
    return dict(artifact)


def _provider_directory_set(root: Path, relative: str) -> set[str]:
    directory = root / relative
    if not directory.exists():
        return set()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"phase {relative} provider root is invalid")
    providers: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"phase {relative} provider entry is invalid")
        provider = path.name.strip().lower()
        if not provider or provider != path.name or provider in providers:
            raise ValueError(f"phase {relative} provider identity is invalid")
        providers.add(provider)
    return providers


def _manifest_provider_paths(root: Path) -> dict[str, Path]:
    manifest_root = root / "provider_manifests"
    if not manifest_root.is_dir() or manifest_root.is_symlink():
        return {}
    output: dict[str, Path] = {}
    for path in manifest_root.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise ValueError("phase provider manifest entry is invalid")
        provider = path.stem.strip().lower()
        if not provider or provider != path.stem or provider in output:
            raise ValueError("phase provider manifest filename is invalid")
        output[provider] = path
    return output


_MISSING = object()
_NO_DEFAULT = object()


def _task_value(task: Any, field: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(task, Mapping):
        if field in task:
            return task[field]
    elif hasattr(task, field):
        return getattr(task, field)
    if default is _NO_DEFAULT:
        raise ValueError(f"phase task contract lacks {field}")
    return default


def _task_option_keys(task: Any) -> set[str]:
    explicit = _task_value(task, "option_keys", None)
    if explicit is not None:
        return {str(value).strip().upper() for value in explicit}
    item = _task_value(task, "item")
    if not isinstance(item, Mapping) or not isinstance(item.get("options"), Mapping):
        raise ValueError("phase task contract lacks option keys")
    return {str(value).strip().upper() for value in item["options"]}


def validate_v2_response_record(
    record: Mapping[str, Any],
    *,
    provider: str,
    requested_model: str,
    phase: str,
    task: Any,
    expected_provenance: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if (
        record.get("schema_version")
        != "yher.llm_sim_v2.response_record.v2"
        or record.get("simulated") is not True
        or record.get("run_id") != RUN_ID
        or record.get("phase") != phase
        or record.get("analysis_population") != phase
        or record.get("provider") != provider
        or record.get("requested_model") != requested_model
    ):
        raise ValueError("phase anchor requires response_record.v2 evidence")
    if expected_provenance is not None and record.get("provenance") != dict(
        expected_provenance
    ):
        raise ValueError("phase record provenance differs from active phase")
    if expected_provenance is not None:
        for field in (
            "collection_mode",
            "development_only",
            "partial",
            "formal_analysis_eligible",
        ):
            if field in expected_provenance and record.get(field) != expected_provenance[
                field
            ]:
                raise ValueError("phase record eligibility differs from provenance")
    for field in (
        "task_id",
        "logical_key",
        "phase",
        "analysis_population",
        "condition",
        "response_arm",
        "message_sha256",
        "wire_message_sha256",
        "persona_id",
        "pair_id",
        "row_id",
        "anchor_id",
        "target_node",
        "item_id",
        "family_id",
        "is_stability_repeat",
        "attempt_id",
        "prompt_revision",
        "prompt_contract_sha256",
    ):
        expected = _task_value(task, field, _MISSING)
        if expected is not _MISSING and record.get(field) != expected:
            raise ValueError(f"phase record differs from task contract: {field}")
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("phase v2 record lacks attempt evidence")
    known_cost = 0.0
    reserve = 0.0
    attempt_identities: list[dict[str, Any]] = []
    response_replays: dict[int, dict[str, Any] | None] = {}
    option_keys = _task_option_keys(task)
    condition = str(_task_value(task, "condition"))
    for index, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt") != index
            or attempt.get("status") not in {"response", "failed"}
        ):
            raise ValueError("phase v2 attempt sequence or status is invalid")
        request_max_tokens = attempt.get("request_max_tokens")
        if not isinstance(request_max_tokens, int) or request_max_tokens <= 0:
            raise ValueError("phase v2 attempt lacks request token identity")
        received = attempt.get("provider_response_received")
        if received is True:
            response_replays[index] = validate_response_attempt_replay(
                attempt,
                condition=condition,
                option_keys=option_keys,
                requested_model=requested_model,
            )
            if attempt.get("status") == "response" and index != len(attempts):
                raise ValueError("successful response must be final")
        elif received is not False:
            raise ValueError("phase v2 attempt response state is invalid")
        elif any(
            field in attempt
            for field in (
                "response_content",
                "response_content_utf8_bytes",
                "response_content_sha256",
            )
        ):
            raise ValueError("non-response attempt retains raw content evidence")
        elif (
            attempt.get("status") != "failed"
            or not isinstance(attempt.get("error_category"), str)
            or not str(attempt.get("error_category")).strip()
            or attempt.get("error_category") == "invalid_schema"
        ):
            raise ValueError("non-response attempt status semantics do not reconcile")
        if attempt.get("cost_known") is True:
            cost = _finite_nonnegative(
                attempt.get("cost_yuan"), label="phase v2 known attempt cost"
            )
            if (
                not _same_amount(attempt.get("cost_reserve_yuan", 0.0), 0.0)
                or attempt.get("billing_ambiguity") is not False
            ):
                raise ValueError("phase v2 known attempt billing is invalid")
            known_cost += cost
        elif attempt.get("cost_known") is False:
            attempt_reserve = _finite_nonnegative(
                attempt.get("cost_reserve_yuan"),
                label="phase v2 unknown attempt reserve",
            )
            if (
                attempt.get("cost_yuan") is not None
                or attempt.get("billing_ambiguity") is not True
                or not _same_amount(attempt_reserve, 10.0)
            ):
                raise ValueError("phase v2 unknown attempt billing is invalid")
            reserve += attempt_reserve
        else:
            raise ValueError("phase v2 attempt cost identity is invalid")
        attempt_identities.append(
            {
                "task_id": str(record["task_id"]),
                "attempt": index,
                "model": requested_model,
                "request_max_tokens": request_max_tokens,
                "wire_message_sha256": str(record["wire_message_sha256"]),
            }
        )
    known_cost = round(known_cost, 8)
    reserve = round(reserve, 8)
    if not (
        record.get("retry_count") == len(attempts) - 1
        and _same_amount(record.get("known_cost_yuan"), known_cost)
        and _same_amount(record.get("unknown_cost_reserve_yuan"), reserve)
        and _same_amount(record.get("cost_yuan"), known_cost + reserve)
        and record.get("has_unknown_cost_attempts") is (reserve > 0.0)
        and record.get("needs_user") is (reserve > 0.0)
        and record.get("needs_user_reasons")
        == (["unknown_provider_billing_reserved"] if reserve > 0.0 else [])
    ):
        raise ValueError("phase v2 record attempt costs do not reconcile")

    status = record.get("status")
    if status == "complete":
        response_attempts = [
            row for row in attempts if row.get("provider_response_received") is True
        ]
        if not response_attempts or attempts[-1].get("status") != "response":
            raise ValueError("complete phase v2 record lacks final raw response")
        from .runner import compute_outcomes

        replayed = response_replays[len(attempts)]
        if replayed is None:
            raise ValueError("complete phase record lacks a valid strict response")
        if replayed != record.get("parsed_output"):
            raise ValueError("phase record differs from raw strict replay")
        expected_outcomes = compute_outcomes(
            condition=str(_task_value(task, "condition")),
            response_arm=str(_task_value(task, "response_arm")),
            answer=replayed.get("answer"),
            abstain=bool(replayed.get("abstain")),
            correct_option=str(_task_value(task, "correct_option")),
            target_option=_task_value(task, "target_option", None),
        )
        if (
            record.get("outcomes") != expected_outcomes
            or record.get("error") is not None
            or record.get("model_id") != requested_model
        ):
            raise ValueError("phase record outcomes differ from strict replay")
    elif status in {
        "excluded_schema",
        "technical_failure",
        "excluded_model_drift",
    }:
        null_outcomes = {
            "is_correct": None,
            "target_option_hit": None,
            "manipulation_compliance": None,
        }
        final_attempt = attempts[-1]
        valid = (
            record.get("parsed_output") is None
            and record.get("outcomes") == null_outcomes
            and isinstance(record.get("error"), str)
            and bool(str(record.get("error")).strip())
        )
        if status == "excluded_schema":
            valid = (
                valid
                and final_attempt.get("status") == "failed"
                and final_attempt.get("error_category") == "invalid_schema"
                and final_attempt.get("provider_response_received") is True
                and response_replays.get(len(attempts), _MISSING) is None
            )
        elif status == "technical_failure":
            valid = (
                valid
                and final_attempt.get("status") == "failed"
                and final_attempt.get("provider_response_received") is False
                and final_attempt.get("error_category") != "invalid_schema"
                and not (
                    str(final_attempt.get("model_returned") or "")
                    and str(final_attempt.get("model_returned")) != requested_model
                )
            )
        else:
            final_returned_model = str(final_attempt.get("model_returned") or "")
            valid = (
                valid
                and bool(final_returned_model)
                and final_returned_model != requested_model
                and record.get("model_id") == final_returned_model
                and record.get("error") == "returned_model_drift"
            )
        if not valid:
            raise ValueError("phase record status semantics do not reconcile")
    else:
        raise ValueError("phase record status semantics do not reconcile")
    return attempt_identities


def _reconcile_provider_calls(
    *,
    provider: str,
    events: Sequence[Mapping[str, Any]],
    attempt_identities: Sequence[Mapping[str, Any]],
) -> None:
    call_rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") == "invocation_finished":
            if event.get("status") == "unavailable" and event.get(
                "provider_call_count"
            ) != 0:
                raise ValueError("unavailable provider evidence contains calls")
            if (
                event.get("invocation_kind") == "resume"
                and event.get("resumed_record_count")
                == event.get("expected_task_count")
                and event.get("provider_call_count") != 0
            ):
                raise ValueError("complete resume evidence contains provider calls")
        if event.get("event_type") != "provider_call_started":
            continue
        call_rows.append(
            {
                "task_id": str(event.get("task_id") or ""),
                "attempt": event.get("attempt"),
                "model": event.get("model"),
                "request_max_tokens": event.get("request_max_tokens"),
                "wire_message_sha256": event.get("wire_message_sha256"),
            }
        )
    key = lambda row: (str(row["task_id"]), int(row["attempt"]))
    try:
        calls = sorted((dict(row) for row in call_rows), key=key)
        attempts = sorted((dict(row) for row in attempt_identities), key=key)
    except (TypeError, ValueError) as exc:
        raise ValueError("provider call-attempt identity is invalid") from exc
    if len({key(row) for row in calls}) != len(calls) or calls != attempts:
        raise ValueError(f"provider {provider} call-attempt evidence does not reconcile")


def build_phase_evidence_receipt(
    phase_root: Path | str,
    *,
    phase_provenance: Mapping[str, Any] | None = None,
    tasks: Sequence[Any] = (),
) -> dict[str, Any]:
    root, phase = _validate_phase_root(phase_root)
    bound_phase = _load_bound_phase_provenance(root, supplied=phase_provenance)
    selected = bound_phase.get("selected_providers")
    if (
        not isinstance(selected, list)
        or len(selected) != len(set(selected))
        or any(str(value).strip().lower() != value for value in selected)
    ):
        raise ValueError("phase provenance selected provider set is invalid")
    expected_providers = set(selected)
    manifest_paths = _manifest_provider_paths(root)
    provider_sets = {
        "phase provenance": expected_providers,
        "manifests": set(manifest_paths),
        "events": _provider_directory_set(root, "evidence/provider_events"),
        "records": _provider_directory_set(root, "records"),
    }
    if any(providers != expected_providers for providers in provider_sets.values()):
        raise ValueError(f"phase provider set drifted across roots: {provider_sets}")

    roster = bound_phase.get("task_roster")
    expected_task_ids = (
        roster.get("expected_task_ids") if isinstance(roster, Mapping) else None
    )
    if not isinstance(expected_task_ids, list):
        raise ValueError("phase provenance lacks its task roster")
    task_map = {str(_task_value(task, "task_id")): task for task in tasks}
    if len(task_map) != len(tasks) or set(task_map) != set(expected_task_ids):
        raise ValueError("phase receipt task contract differs from phase provenance")
    expected_record_provenance: Mapping[str, Any] | None = None
    try:
        from .runner import phase_provenance_binding

        expected_record_provenance = phase_provenance_binding(bound_phase)
    except (KeyError, TypeError, ValueError):
        if bound_phase.get("collection_mode") == "formal":
            raise ValueError("formal phase provenance cannot bind response records")

    providers: dict[str, Any] = {}
    for provider in selected:
        path = manifest_paths[provider]
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"provider manifest is invalid: {path}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("provider manifest must be an object")
        if manifest.get("provider") != provider or manifest.get("phase") not in {
            None,
            phase,
        }:
            raise ValueError("provider manifest filename/content identity differs")
        record_set = manifest.get("record_set")
        if not isinstance(record_set, Mapping):
            raise ValueError("provider manifest lacks its record-set binding")
        lifecycle = manifest.get("lifecycle")
        manifest_task_ids = (
            lifecycle.get("expected_task_ids")
            if isinstance(lifecycle, Mapping)
            else None
        )
        if manifest_task_ids != expected_task_ids:
            raise ValueError("provider manifest task roster differs from phase provenance")
        recomputed_record_set = build_provider_record_set(
            root,
            provider=provider,
            expected_task_ids=expected_task_ids,
        )
        if dict(record_set) != recomputed_record_set:
            raise ValueError("provider record-set binding differs from disk bytes")
        requested_model = str(manifest.get("requested_model") or "")
        if not requested_model:
            raise ValueError("provider manifest lacks requested model identity")
        if expected_record_provenance is not None and manifest.get(
            "provenance"
        ) != dict(expected_record_provenance):
            raise ValueError("provider manifest provenance differs from active phase")
        attempt_identities: list[dict[str, Any]] = []
        for record_row in recomputed_record_set["records"]:
            record_path = root / str(record_row["path"])
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("phase provider record is invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError("phase provider record must be an object")
            attempt_identities.extend(
                validate_v2_response_record(
                    record,
                    provider=provider,
                    requested_model=requested_model,
                    phase=phase,
                    task=task_map[str(record_row["task_id"])],
                    expected_provenance=expected_record_provenance,
                )
            )
        ledger = ProviderEvidenceLedger(
            root,
            run_id=RUN_ID,
            phase=phase,
            provider=provider,
        )
        events = ledger.read_events()
        _reconcile_provider_calls(
            provider=provider,
            events=events,
            attempt_identities=attempt_identities,
        )
        providers[provider] = {
            "provider_manifest_path": path.relative_to(root).as_posix(),
            "provider_manifest_sha256": sha256_file(path),
            "record_set_sha256": recomputed_record_set["record_set_sha256"],
            "record_count": recomputed_record_set["record_count"],
            "validated_attempt_count": len(attempt_identities),
            "evidence_event_count": len(events),
            "evidence_chain_head_sha256": (
                events[-1]["event_sha256"] if events else None
            ),
            "provider_call_count": sum(
                event.get("event_type") == "provider_call_started"
                for event in events
            ),
        }
    receipt: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.phase_evidence_receipt.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": phase,
        "authority": "post_invocation_phase_receipt",
        "store_snapshot": snapshot_phase_store(root),
        "providers": providers,
    }
    receipt["phase_provenance_sha256"] = bound_phase[
        "phase_provenance_sha256"
    ]
    receipt["phase_provenance_file_sha256"] = sha256_file(
        root / "phase_provenance.json"
    )
    receipt["phase_evidence_receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def write_phase_evidence_receipt(
    phase_root: Path | str,
    *,
    output: Path | str | None = None,
    phase_provenance: Mapping[str, Any] | None = None,
    tasks: Sequence[Any] = (),
) -> dict[str, Any]:
    root, _ = _validate_phase_root(phase_root)
    receipt = build_phase_evidence_receipt(
        root,
        phase_provenance=phase_provenance,
        tasks=tasks,
    )
    destination = (
        Path(output).expanduser().resolve(strict=False)
        if output is not None
        else root
        / "evidence"
        / "phase_receipts"
        / f"{receipt['phase_evidence_receipt_sha256']}.json"
    )
    _atomic_json(destination, receipt, immutable=True)
    return receipt


def build_phase_source_file_set(phase_root: Path | str) -> dict[str, Any]:
    """Bind every existing byte in a phase without writing to that phase."""

    supplied_root = Path(phase_root).expanduser()
    if supplied_root.is_symlink():
        raise ValueError("phase source root cannot be a symlink")
    root, phase = _validate_phase_root(supplied_root)
    if not root.is_dir():
        raise ValueError("phase source root does not exist")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("phase source file set cannot contain symlinks")
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "yher.llm_sim_v2.phase_source_file_set.v1",
        "run_id": RUN_ID,
        "phase": phase,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
        "file_set_sha256": canonical_sha256(rows),
    }


def _finite_nonnegative(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _same_amount(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-8
        )
    except (TypeError, ValueError):
        return False


def _audit_digest(audit_report: Mapping[str, Any]) -> str:
    payload = dict(audit_report)
    advertised = payload.pop("pilot_audit_sha256", None)
    if not isinstance(advertised, str) or advertised != canonical_sha256(payload):
        raise ValueError("legacy pilot audit digest is invalid")
    return advertised


def _legacy_record_accounting(
    record: Mapping[str, Any],
    *,
    provider: str,
    task_id: str,
) -> dict[str, Any]:
    if record.get("schema_version") != "yher.llm_sim_v2.response_record.v1":
        raise ValueError("retrospective receipt accepts only legacy v1 records")
    if str(record.get("task_id") or "") != task_id:
        raise ValueError("legacy record filename and task_id differ")
    if record.get("provider") not in {None, provider}:
        raise ValueError("legacy record provider differs from its directory")
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("legacy record lacks its attempt ledger")

    known = 0.0
    reserve = 0.0
    unknown_attempt_count = 0
    response_attempt_count = 0
    raw_content_present_count = 0
    raw_content_bound_count = 0
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping) or attempt.get("attempt") != index:
            raise ValueError("legacy attempt sequence is invalid")
        if attempt.get("status") == "response":
            response_attempt_count += 1
        if isinstance(attempt.get("response_content"), str):
            raw_content_present_count += 1
        if attempt.get("provider_response_received") is True:
            validate_response_content_binding(attempt)
            raw_content_bound_count += 1
        if attempt.get("cost_known") is True:
            known += _finite_nonnegative(
                attempt.get("cost_yuan"), label="legacy known attempt cost"
            )
            if not _same_amount(attempt.get("cost_reserve_yuan", 0.0), 0.0):
                raise ValueError("known legacy attempt carries a reserve")
        elif attempt.get("cost_known") is False:
            if attempt.get("cost_yuan") is not None:
                raise ValueError("unknown legacy attempt claims a known cost")
            reserve += _finite_nonnegative(
                attempt.get("cost_reserve_yuan"),
                label="legacy unknown attempt reserve",
            )
            unknown_attempt_count += 1
        else:
            raise ValueError("legacy attempt cost identity is missing")

    known = round(known, 8)
    reserve = round(reserve, 8)
    accounted = round(known + reserve, 8)
    if not (
        _same_amount(record.get("known_cost_yuan"), known)
        and _same_amount(record.get("unknown_cost_reserve_yuan"), reserve)
        and _same_amount(record.get("cost_yuan"), accounted)
    ):
        raise ValueError("legacy record cost does not reconcile with attempts")
    return {
        "attempt_count": len(attempts),
        "unknown_attempt_count": unknown_attempt_count,
        "response_attempt_count": response_attempt_count,
        "raw_content_present_attempt_count": raw_content_present_count,
        "raw_content_bound_response_attempt_count": raw_content_bound_count,
        "known_cost_yuan": known,
        "unknown_cost_reserve_yuan": reserve,
        "accounted_cost_yuan": accounted,
    }


def _legacy_provider_binding(
    root: Path,
    *,
    provider: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    audit_provider: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lifecycle = manifest.get("lifecycle")
    expected_task_ids = (
        lifecycle.get("expected_task_ids")
        if isinstance(lifecycle, Mapping)
        else None
    )
    if not isinstance(expected_task_ids, list):
        raise ValueError("legacy provider manifest lacks its expected roster")
    expected = [str(task_id) for task_id in expected_task_ids]
    record_set = build_provider_record_set(
        root, provider=provider, expected_task_ids=expected
    )
    if record_set["missing_task_ids"] or record_set["unexpected_task_ids"]:
        raise ValueError("legacy provider record set differs from its roster")

    accounting_rows: list[dict[str, Any]] = []
    audit_style_rows: list[dict[str, Any]] = []
    for row in record_set["records"]:
        path = root / str(row["path"])
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("legacy provider record is invalid JSON") from exc
        if not isinstance(record, Mapping):
            raise ValueError("legacy provider record must be an object")
        metrics = _legacy_record_accounting(
            record,
            provider=provider,
            task_id=str(row["task_id"]),
        )
        accounting_rows.append(metrics)
        audit_style_rows.append(
            {
                "task_id": row["task_id"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "attempt_count": row["attempt_count"],
            }
        )

    accounting = {
        key: (
            round(sum(float(row[key]) for row in accounting_rows), 8)
            if key.endswith("_yuan")
            else sum(int(row[key]) for row in accounting_rows)
        )
        for key in (
            "attempt_count",
            "unknown_attempt_count",
            "response_attempt_count",
            "raw_content_present_attempt_count",
            "raw_content_bound_response_attempt_count",
            "known_cost_yuan",
            "unknown_cost_reserve_yuan",
            "accounted_cost_yuan",
        )
    }
    budget = manifest.get("budget")
    if isinstance(budget, Mapping) and not (
        _same_amount(
            budget.get("provider_record_known_cost_yuan"),
            accounting["known_cost_yuan"],
        )
        and _same_amount(
            budget.get("provider_record_unknown_reserve_yuan"),
            accounting["unknown_cost_reserve_yuan"],
        )
        and _same_amount(
            budget.get("provider_record_accounted_cost_yuan"),
            accounting["accounted_cost_yuan"],
        )
    ):
        raise ValueError("legacy provider manifest cost differs from records")

    audit_record_set_sha256 = canonical_sha256(audit_style_rows)
    if audit_provider is not None:
        audit_accounting = audit_provider.get("accounting")
        if not isinstance(audit_accounting, Mapping) or not all(
            _same_amount(audit_accounting.get(key), accounting[key])
            for key in (
                "attempt_count",
                "unknown_attempt_count",
                "known_cost_yuan",
                "unknown_cost_reserve_yuan",
                "accounted_cost_yuan",
            )
        ):
            raise ValueError("legacy audit provider accounting differs from records")
        if audit_provider.get("record_set_sha256") != audit_record_set_sha256:
            raise ValueError("legacy audit provider record-set digest differs")

    return {
        **record_set,
        "provider_manifest_path": manifest_path.relative_to(root).as_posix(),
        "provider_manifest_sha256": sha256_file(manifest_path),
        "legacy_audit_record_set_sha256": audit_record_set_sha256,
        "accounting": accounting,
    }


def build_retrospective_legacy_receipt(
    phase_root: Path | str,
    *,
    audit_report: Mapping[str, Any],
    expected_audit_sha256: str,
    expected_known_cost_yuan: float,
    expected_unknown_reserve_yuan: float,
) -> dict[str, Any]:
    """Create a deterministic, non-upgrading receipt for the legacy pilot."""

    root, phase = _validate_phase_root(phase_root)
    if phase != "pilot":
        raise ValueError("retrospective legacy receipts are pilot-only")
    audit_sha256 = _audit_digest(audit_report)
    if audit_sha256 != expected_audit_sha256:
        raise ValueError("legacy pilot audit differs from the expected digest")
    source_set = build_phase_source_file_set(root)
    source_binding = audit_report.get("source_binding")
    audit_files = (
        source_binding.get("pilot_source_files")
        if isinstance(source_binding, Mapping)
        else None
    )
    if not (
        isinstance(audit_files, list)
        and audit_files == source_set["files"]
        and source_binding.get("pilot_source_file_count")
        == source_set["file_count"]
        and source_binding.get("pilot_source_set_sha256")
        == source_set["file_set_sha256"]
    ):
        raise ValueError("legacy pilot audit source binding differs from disk")

    audit_providers = audit_report.get("providers")
    audit_provider_map = audit_providers if isinstance(audit_providers, Mapping) else {}
    providers: dict[str, Any] = {}
    manifest_root = root / "provider_manifests"
    manifest_paths = sorted(manifest_root.glob("*.json")) if manifest_root.is_dir() else []
    if not manifest_paths:
        raise ValueError("legacy pilot has no provider manifests")
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("legacy provider manifest is invalid JSON") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("legacy provider manifest must be an object")
        provider = str(manifest.get("provider") or manifest_path.stem).strip().lower()
        if not provider or provider != manifest_path.stem or provider in providers:
            raise ValueError("legacy provider manifest identity is invalid")
        if manifest.get("run_id") not in {None, RUN_ID} or manifest.get("phase") not in {
            None,
            "pilot",
        }:
            raise ValueError("legacy provider manifest scope differs")
        audit_provider = audit_provider_map.get(provider)
        if audit_provider is not None and not isinstance(audit_provider, Mapping):
            raise ValueError("legacy audit provider entry is invalid")
        providers[provider] = _legacy_provider_binding(
            root,
            provider=provider,
            manifest_path=manifest_path,
            manifest=manifest,
            audit_provider=audit_provider,
        )

    record_provider_names = {
        path.parent.name
        for path in (root / "records").glob("*/*.json")
        if path.is_file()
    }
    if record_provider_names != set(providers):
        raise ValueError("legacy record providers and manifests differ")
    if audit_provider_map and set(audit_provider_map) != set(providers):
        raise ValueError("legacy audit provider set differs from manifests")

    accounting = {
        key: (
            round(sum(float(row["accounting"][key]) for row in providers.values()), 8)
            if key.endswith("_yuan")
            else sum(int(row["accounting"][key]) for row in providers.values())
        )
        for key in (
            "attempt_count",
            "unknown_attempt_count",
            "response_attempt_count",
            "raw_content_present_attempt_count",
            "raw_content_bound_response_attempt_count",
            "known_cost_yuan",
            "unknown_cost_reserve_yuan",
            "accounted_cost_yuan",
        )
    }
    expected_known = round(
        _finite_nonnegative(expected_known_cost_yuan, label="expected legacy known cost"),
        8,
    )
    expected_reserve = round(
        _finite_nonnegative(
            expected_unknown_reserve_yuan, label="expected legacy reserve"
        ),
        8,
    )
    audit_accounting = audit_report.get("accounting")
    if not isinstance(audit_accounting, Mapping) or not (
        _same_amount(audit_accounting.get("known_cost_yuan"), expected_known)
        and _same_amount(
            audit_accounting.get("unknown_cost_reserve_yuan"), expected_reserve
        )
        and _same_amount(
            audit_accounting.get("accounted_cost_yuan"),
            round(expected_known + expected_reserve, 8),
        )
        and _same_amount(accounting["known_cost_yuan"], expected_known)
        and _same_amount(accounting["unknown_cost_reserve_yuan"], expected_reserve)
    ):
        raise ValueError("legacy pilot costs do not reconcile")
    if accounting["raw_content_bound_response_attempt_count"] != 0:
        raise ValueError("raw-bound responses are not legacy parsed-only evidence")

    provider_record_sets_sha256 = canonical_sha256(providers)
    receipt: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.retrospective_legacy_receipt.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "phase": "pilot",
        "authority": "retrospective_legacy_evidence_receipt",
        "evidence_quality": "legacy_parsed_only_no_raw_content",
        "formal_release_gate_eligible": False,
        "pilot_audit_sha256": audit_sha256,
        "audit_source_binding_sha256": canonical_sha256(source_binding),
        "source_file_set": source_set,
        "provider_record_sets": providers,
        "provider_record_sets_sha256": provider_record_sets_sha256,
        "record_count": sum(int(row["record_count"]) for row in providers.values()),
        **accounting,
        "total_accounted_cost_yuan": round(
            float(accounting["known_cost_yuan"])
            + float(accounting["unknown_cost_reserve_yuan"]),
            8,
        ),
        "limitations": [
            "exact provider response content was not retained",
            "strict parsing cannot be replayed from provider bytes",
            "this receipt cannot approve a formal release gate",
        ],
    }
    receipt["retrospective_legacy_receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _validate_legacy_receipt_identity(receipt: Mapping[str, Any]) -> None:
    payload = dict(receipt)
    advertised = payload.pop("retrospective_legacy_receipt_sha256", None)
    if not (
        receipt.get("schema_version")
        == "yher.llm_sim_v2.retrospective_legacy_receipt.v1"
        and receipt.get("simulated") is True
        and receipt.get("run_id") == RUN_ID
        and receipt.get("phase") == "pilot"
        and receipt.get("evidence_quality")
        == "legacy_parsed_only_no_raw_content"
        and receipt.get("formal_release_gate_eligible") is False
        and receipt.get("raw_content_bound_response_attempt_count") == 0
        and isinstance(advertised, str)
        and advertised == canonical_sha256(payload)
    ):
        raise ValueError("retrospective legacy receipt identity is invalid")


def validate_retrospective_legacy_receipt(
    receipt: Mapping[str, Any],
    *,
    phase_root: Path | str,
    audit_report: Mapping[str, Any],
    expected_audit_sha256: str,
    expected_known_cost_yuan: float,
    expected_unknown_reserve_yuan: float,
) -> dict[str, Any]:
    _validate_legacy_receipt_identity(receipt)
    rebuilt = build_retrospective_legacy_receipt(
        phase_root,
        audit_report=audit_report,
        expected_audit_sha256=expected_audit_sha256,
        expected_known_cost_yuan=expected_known_cost_yuan,
        expected_unknown_reserve_yuan=expected_unknown_reserve_yuan,
    )
    if dict(receipt) != rebuilt:
        raise ValueError("retrospective legacy receipt differs from bound sources")
    return {
        "ok": True,
        "retrospective_legacy_receipt_sha256": receipt[
            "retrospective_legacy_receipt_sha256"
        ],
        "file_set_sha256": receipt["source_file_set"]["file_set_sha256"],
        "record_count": receipt["record_count"],
        "known_cost_yuan": receipt["known_cost_yuan"],
        "unknown_cost_reserve_yuan": receipt["unknown_cost_reserve_yuan"],
    }


def build_carried_forward_cost_ledger(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_legacy_receipt_identity(receipt)
    known = round(
        _finite_nonnegative(receipt.get("known_cost_yuan"), label="carried known cost"),
        8,
    )
    reserve = round(
        _finite_nonnegative(
            receipt.get("unknown_cost_reserve_yuan"), label="carried reserve"
        ),
        8,
    )
    if not _same_amount(receipt.get("total_accounted_cost_yuan"), known + reserve):
        raise ValueError("retrospective receipt costs do not reconcile")
    source_record_set = str(receipt.get("provider_record_sets_sha256") or "")
    source_receipt = str(receipt.get("retrospective_legacy_receipt_sha256") or "")
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in (source_record_set, source_receipt)
    ):
        raise ValueError("retrospective receipt source digests are invalid")
    ledger: dict[str, Any] = {
        "schema_version": "yher.llm_sim_v2.carried_forward_cost_ledger.v1",
        "simulated": True,
        "run_id": RUN_ID,
        "source_phase": "pilot",
        "source_record_set_sha256": source_record_set,
        "source_phase_receipt_sha256": source_receipt,
        "known_cost_yuan": known,
        "unknown_cost_reserve_yuan": reserve,
        "total_accounted_cost_yuan": round(known + reserve, 8),
    }
    ledger["carried_forward_cost_ledger_sha256"] = canonical_sha256(ledger)
    return ledger


def _reject_phase_output(output: Path | str) -> Path:
    destination = Path(output).expanduser().resolve(strict=False)
    for parent in (destination, *destination.parents):
        if parent.name in _PHASES and parent.parent.name == RUN_ID:
            raise ValueError("evidence anchors cannot be written into a phase source")
    return destination


def write_retrospective_legacy_receipt(
    receipt: Mapping[str, Any],
    *,
    output: Path | str,
) -> Path:
    _validate_legacy_receipt_identity(receipt)
    destination = _reject_phase_output(output)
    return _atomic_json(destination, receipt, immutable=True)


def write_carried_forward_cost_ledger(
    ledger: Mapping[str, Any],
    *,
    output: Path | str,
) -> Path:
    payload = dict(ledger)
    advertised = payload.pop("carried_forward_cost_ledger_sha256", None)
    if not (
        ledger.get("schema_version")
        == "yher.llm_sim_v2.carried_forward_cost_ledger.v1"
        and ledger.get("simulated") is True
        and ledger.get("run_id") == RUN_ID
        and isinstance(advertised, str)
        and advertised == canonical_sha256(payload)
    ):
        raise ValueError("carried-forward cost ledger identity is invalid")
    destination = _reject_phase_output(output)
    return _atomic_json(destination, ledger, immutable=True)


__all__ = [
    "ProviderEvidenceLedger",
    "bind_response_content",
    "build_carried_forward_cost_ledger",
    "build_phase_evidence_receipt",
    "build_phase_source_file_set",
    "build_provider_record_set",
    "build_retrospective_legacy_receipt",
    "canonical_sha256",
    "sha256_file",
    "snapshot_phase_store",
    "validate_retrospective_legacy_receipt",
    "validate_response_content_binding",
    "validate_response_attempt_replay",
    "validate_v2_response_record",
    "write_carried_forward_cost_ledger",
    "write_phase_evidence_receipt",
    "write_retrospective_legacy_receipt",
]
