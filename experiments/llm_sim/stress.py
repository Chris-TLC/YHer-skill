"""Resumable provider transport stress test confined to temporary storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .transport import (
    HTTPProviderTransport,
    PROVIDER_SPECS,
    ProviderHTTPError,
    ProviderNetworkError,
    ProviderRuntimePolicy,
    ProviderTruncatedResponseError,
    ProviderTransport,
    load_live_environment,
    model_from_environment,
    provider_spec,
    transport_policy,
)


DEFAULT_OUTPUT_ROOT = Path("/tmp/yher_h5_stress")
DEFAULT_CALL_COUNT = 50
CALLS_PER_V2_JOURNEY = 19


def validate_stress_output_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve(strict=False)
    allowed_roots = {
        Path("/tmp").resolve(strict=False),
        Path(tempfile.gettempdir()).resolve(strict=False),
    }
    if not any(candidate == root or candidate.is_relative_to(root) for root in allowed_roots):
        raise ValueError("stress output must remain under /tmp")
    if any(part.lower() == "sim_store" for part in candidate.parts):
        raise ValueError("stress output must never use sim_store")
    return candidate


def conservative_cost_projection(
    *,
    average_call_cost_yuan: float,
    calls_per_journey: int = CALLS_PER_V2_JOURNEY,
    journeys: int = 600,
    safety_factor: float = 1.3,
) -> float:
    if min(average_call_cost_yuan, calls_per_journey, journeys, safety_factor) < 0:
        raise ValueError("cost projection inputs must be non-negative")
    return float(average_call_cost_yuan) * calls_per_journey * journeys * safety_factor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object, *, immutable: bool = False) -> None:
    payload = _canonical_bytes(value)
    if path.exists() and immutable:
        if path.read_bytes() == payload:
            return
        raise ValueError(f"immutable stress record differs: {path}")
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


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid stress artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid stress artifact: {path}")
    return value


def _messages(call_id: int) -> list[dict[str, str]]:
    personas = (
        "You are a weak student who confuses oxidation with reduction.",
        "You are a student with a missing prerequisite and low confidence.",
        "You are a strong student who checks chemistry carefully.",
        "You are a student whose reasoning chain is unstable under multi-step questions.",
    )
    questions = (
        "在反应 Zn + CuSO4 = ZnSO4 + Cu 中，作还原剂的是？ A.Zn B.CuSO4 C.ZnSO4 D.Cu",
        "下列微粒中，与Na+电子总数相同的是？ A.Ne B.F- C.Mg2+ D.以上都是",
        "增大反应物浓度时，化学反应速率通常如何变化？ A.减小 B.增大 C.不变 D.必为零",
        "原电池工作时，电子从哪一极经外电路流出？ A.正极 B.负极 C.盐桥 D.电解质",
        "25摄氏度时，pH=3的溶液中氢离子浓度为？ A.10^-3 B.3 C.10^3 D.0",
    )
    system = (
        personas[call_id % len(personas)]
        + " Simulate the persona instead of answering as an expert. Return exactly one "
        'JSON object: {"answer":"A|B|C|D","rationale":"one short Chinese sentence"}. '
        "Do not use Markdown."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": questions[call_id % len(questions)]},
    ]


def _messages_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(
        list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shape_valid(content: str) -> bool:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(value, Mapping)
        and str(value.get("answer") or "").strip().upper() in {"A", "B", "C", "D"}
        and bool(str(value.get("rationale") or "").strip())
    )


def _error_category(exc: Exception) -> str:
    if isinstance(exc, ProviderTruncatedResponseError):
        return "truncated_length"
    if isinstance(exc, ProviderHTTPError):
        return f"http_{exc.status}"
    if isinstance(exc, ProviderNetworkError):
        return "network_timeout"
    return "protocol_or_unexpected"


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (ProviderTruncatedResponseError, ProviderNetworkError)):
        return True
    return isinstance(exc, ProviderHTTPError) and (
        exc.status == 429 or exc.status >= 500
    )


def _backoff_delay(
    policy: ProviderRuntimePolicy,
    retry_index: int,
    *,
    random_value: Callable[[], float],
    retry_after_seconds: float | None = None,
) -> float:
    base = min(
        policy.max_backoff_seconds,
        policy.base_backoff_seconds * (2**retry_index),
    )
    jitter = 1.0 + policy.jitter_fraction * (2.0 * random_value() - 1.0)
    return max(float(retry_after_seconds or 0.0), max(0.0, base * jitter))


def _execute_call(
    *,
    provider: str,
    call_id: int,
    requested_model: str,
    transport: ProviderTransport,
    policy: ProviderRuntimePolicy,
    sleep: Callable[[float], None],
    random_value: Callable[[], float],
) -> dict[str, Any]:
    messages = _messages(call_id)
    attempts: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    request_max_tokens = policy.max_tokens
    final_response: Mapping[str, Any] | None = None
    final_error: Exception | None = None
    started_at = _utc_now()

    for attempt_index in range(policy.max_attempts):
        attempt_started = time.monotonic()
        try:
            response = transport.complete(
                provider=provider,
                model=requested_model,
                messages=messages,
                timeout_seconds=policy.timeout_seconds,
                max_tokens=request_max_tokens,
            )
            latency_ms = float(
                response.get("latency_ms")
                or (time.monotonic() - attempt_started) * 1000.0
            )
            usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
            input_tokens = max(0, int(usage.get("input_tokens") or 0))
            output_tokens = max(0, int(usage.get("output_tokens") or 0))
            cost = max(0.0, float(response.get("cost_yuan") or 0.0))
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cost += cost
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "success",
                    "request_max_tokens": request_max_tokens,
                    "latency_ms": round(latency_ms, 3),
                    "finish_reason": str(response.get("finish_reason") or ""),
                    "reasoning_tokens": max(0, int(response.get("reasoning_tokens") or 0)),
                    "http_status": int(response.get("http_status") or 200),
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    "cost_yuan": cost,
                }
            )
            final_response = response
            break
        except Exception as exc:
            final_error = exc
            latency_ms = (time.monotonic() - attempt_started) * 1000.0
            if isinstance(exc, ProviderTruncatedResponseError):
                input_tokens = max(0, int(exc.usage.get("input_tokens") or 0))
                output_tokens = max(0, int(exc.usage.get("output_tokens") or 0))
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += exc.cost_yuan
                latency_ms = exc.latency_ms or latency_ms
            else:
                input_tokens = output_tokens = 0
            category = _error_category(exc)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "status": "truncated" if category == "truncated_length" else "failed",
                    "error_category": category,
                    "request_max_tokens": request_max_tokens,
                    "latency_ms": round(latency_ms, 3),
                    "finish_reason": getattr(exc, "finish_reason", None),
                    "reasoning_tokens": int(getattr(exc, "reasoning_tokens", 0)),
                    "http_status": getattr(exc, "status", None),
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    "cost_yuan": float(getattr(exc, "cost_yuan", 0.0)),
                }
            )
            if attempt_index + 1 >= policy.max_attempts or not _retryable(exc):
                break
            if isinstance(exc, ProviderTruncatedResponseError):
                request_max_tokens = policy.retry_max_tokens
            delay = _backoff_delay(
                policy,
                attempt_index,
                random_value=random_value,
                retry_after_seconds=getattr(exc, "retry_after_seconds", None),
            )
            sleep(delay)

    error_counts = Counter(
        str(row["error_category"])
        for row in attempts
        if row.get("error_category")
    )
    content = str(final_response.get("content") or "") if final_response else ""
    returned_model = (
        str(final_response.get("model_returned") or "") if final_response else None
    )
    return {
        "record_type": "yher.h5_transport_stress_call.v1",
        "provider": provider,
        "call_id": call_id,
        "requested_model": requested_model,
        "returned_model": returned_model,
        "model_drift": bool(returned_model and returned_model != requested_model),
        "messages_sha256": _messages_sha256(messages),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "success": final_response is not None,
        "shape_valid": _shape_valid(content) if final_response else False,
        "content_length": len(content),
        "finish_reason": str(final_response.get("finish_reason") or "") if final_response else None,
        "reasoning_tokens": max(0, int(final_response.get("reasoning_tokens") or 0)) if final_response else int(getattr(final_error, "reasoning_tokens", 0)),
        "request_max_tokens": int(final_response.get("request_max_tokens") or request_max_tokens) if final_response else request_max_tokens,
        "latency_ms": round(sum(float(row["latency_ms"]) for row in attempts), 3),
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "cost_yuan": total_cost,
        "retry_count": max(0, len(attempts) - 1),
        "error_counts": dict(sorted(error_counts.items())),
        "attempts": attempts,
    }


def _existing_records(provider_dir: Path, provider: str, call_count: int) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for path in sorted((provider_dir / "records").glob("call-*.json")):
        row = _read_json(path)
        call_id = row.get("call_id")
        if (
            row.get("provider") != provider
            or not isinstance(call_id, int)
            or not 0 <= call_id < call_count
            or call_id in records
        ):
            raise ValueError("stress checkpoint contains duplicate or invalid records")
        records[call_id] = row
    return records


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[rank]


def _summary(
    provider: str,
    requested_model: str,
    call_count: int,
    records: Mapping[int, Mapping[str, Any]],
    *,
    status: str,
    policy: ProviderRuntimePolicy,
) -> dict[str, Any]:
    rows = [records[key] for key in sorted(records)]
    successful = [row for row in rows if row.get("success") is True]
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    error_counts: Counter[str] = Counter()
    for row in rows:
        error_counts.update(
            {str(key): int(value) for key, value in (row.get("error_counts") or {}).items()}
        )
    returned_models = sorted(
        {str(row["returned_model"]) for row in successful if row.get("returned_model")}
    )
    total_cost = sum(float(row.get("cost_yuan") or 0.0) for row in rows)
    total_input = sum(int((row.get("usage") or {}).get("input_tokens") or 0) for row in rows)
    total_output = sum(int((row.get("usage") or {}).get("output_tokens") or 0) for row in rows)
    return {
        "record_type": "yher.h5_transport_stress_summary.v1",
        "provider": provider,
        "requested_model": requested_model,
        "returned_models": returned_models,
        "model_drift_count": sum(bool(row.get("model_drift")) for row in rows),
        "status": status,
        "target_calls": call_count,
        "completed": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "shape_valid_rate": sum(bool(row.get("shape_valid")) for row in rows) / len(rows) if rows else 0.0,
        "empty_content_rate": sum(
            bool((row.get("error_counts") or {}).get("truncated_length")) for row in rows
        ) / len(rows) if rows else 0.0,
        "error_counts": dict(sorted(error_counts.items())),
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "usage": {"input_tokens": total_input, "output_tokens": total_output},
        "cost_yuan": total_cost,
        "average_call_cost_yuan": total_cost / len(rows) if rows else 0.0,
        "retry_count": sum(int(row.get("retry_count") or 0) for row in rows),
        "policy": {
            key: getattr(policy, key)
            for key in (
                "max_tokens",
                "retry_max_tokens",
                "timeout_seconds",
                "concurrency",
                "max_attempts",
                "failure_threshold",
                "base_backoff_seconds",
                "max_backoff_seconds",
                "cooldown_seconds",
                "jitter_fraction",
            )
        },
    }


def _write_checkpoint(provider_dir: Path, provider: str, records: Mapping[int, Mapping[str, Any]]) -> Path:
    rows = [
        {
            "call_id": call_id,
            "path": f"records/call-{call_id:03d}.json",
            "sha256": _sha256(provider_dir / f"records/call-{call_id:03d}.json"),
        }
        for call_id in sorted(records)
    ]
    path = provider_dir / "checkpoint.json"
    _write_json(
        path,
        {
            "record_type": "yher.h5_transport_stress_checkpoint.v1",
            "provider": provider,
            "completed_call_ids": [row["call_id"] for row in rows],
            "records": rows,
        },
    )
    return path


def run_provider_stress(
    provider: str,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    call_count: int = DEFAULT_CALL_COUNT,
    transport: ProviderTransport | None = None,
    environment: Mapping[str, str] | None = None,
    policy: ProviderRuntimePolicy | None = None,
    checkpoint_stop_after: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
) -> dict[str, Any]:
    if call_count < 1:
        raise ValueError("stress call count must be positive")
    spec = provider_spec(provider)
    provider = spec.name
    root = validate_stress_output_root(output_root)
    provider_dir = root / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    runtime = policy or transport_policy(provider, version="v2")
    live_environment = dict(environment or {})
    if transport is None:
        live_environment = dict(
            environment or load_live_environment(repo_root=Path(__file__).resolve().parents[2])
        )
        transport = HTTPProviderTransport.from_environment(
            provider,
            environment=live_environment,
            version="v2",
        )
    requested_model = model_from_environment(
        provider, spec.model_default, environment=live_environment
    )
    records = _existing_records(provider_dir, provider, call_count)
    initial_record_ids = set(records)
    kill_ready_path = provider_dir / "kill_ready.json"
    previous_kill_ready = _read_json(kill_ready_path) if kill_ready_path.is_file() else None
    consecutive_failures = 0
    status = "complete" if len(records) == call_count else "running"

    while len(records) < call_count:
        if checkpoint_stop_after is not None and len(records) >= checkpoint_stop_after:
            status = "checkpoint_ready"
            break
        remaining = [call_id for call_id in range(call_count) if call_id not in records]
        batch_size = min(runtime.concurrency, len(remaining))
        if checkpoint_stop_after is not None:
            batch_size = min(batch_size, checkpoint_stop_after - len(records))
        batch = remaining[:batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(
                    _execute_call,
                    provider=provider,
                    call_id=call_id,
                    requested_model=requested_model,
                    transport=transport,
                    policy=runtime,
                    sleep=sleep,
                    random_value=random_value,
                ): call_id
                for call_id in batch
            }
            batch_records: dict[int, Mapping[str, Any]] = {}
            for future in as_completed(futures):
                call_id = futures[future]
                row = future.result()
                record_path = provider_dir / f"records/call-{call_id:03d}.json"
                _write_json(record_path, row, immutable=True)
                records[call_id] = row
                batch_records[call_id] = row
                _write_checkpoint(provider_dir, provider, records)
        for call_id in sorted(batch_records):
            if batch_records[call_id].get("success") is True:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= runtime.failure_threshold:
                    status = "circuit_open"
                    break
        if status == "circuit_open":
            break

    if len(records) == call_count:
        status = "complete"
    checkpoint = _write_checkpoint(provider_dir, provider, records)
    summary = _summary(
        provider,
        requested_model,
        call_count,
        records,
        status=status,
        policy=runtime,
    )
    _write_json(provider_dir / "summary.json", summary)
    if status == "checkpoint_ready":
        _write_json(
            kill_ready_path,
            {
                "record_type": "yher.h5_transport_kill_ready.v1",
                "provider": provider,
                "pid": os.getpid(),
                "ready_at_utc": _utc_now(),
                "completed_call_ids": sorted(records),
                "checkpoint_sha256": _sha256(checkpoint),
            },
        )
    elif status == "complete" and previous_kill_ready is not None:
        resumed_call_ids = sorted(set(records) - initial_record_ids)
        _write_json(
            provider_dir / "resume_evidence.json",
            {
                "record_type": "yher.h5_transport_resume_evidence.v1",
                "provider": provider,
                "killed_pid": previous_kill_ready.get("pid"),
                "kill_ready_at_utc": previous_kill_ready.get("ready_at_utc"),
                "pre_kill_completed_call_ids": previous_kill_ready.get(
                    "completed_call_ids"
                ),
                "pre_resume_record_count": len(initial_record_ids),
                "resumed_call_ids": resumed_call_ids,
                "new_record_count": len(resumed_call_ids),
                "post_resume_completed_call_ids": sorted(records),
                "duplicate_record_count": len(initial_record_ids.intersection(resumed_call_ids)),
                "resumed_at_utc": _utc_now(),
                "final_checkpoint_sha256": _sha256(checkpoint),
            },
        )
    return summary


def summarize_stress_root(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = validate_stress_output_root(output_root)
    providers: dict[str, Mapping[str, Any]] = {}
    for provider in PROVIDER_SPECS:
        path = root / provider / "summary.json"
        if path.is_file():
            providers[provider] = _read_json(path)
    return {
        "record_type": "yher.h5_transport_stress_run.v1",
        "generated_at_utc": _utc_now(),
        "providers": providers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--provider", action="append", dest="providers")
    parser.add_argument("--calls", type=int, default=DEFAULT_CALL_COUNT)
    parser.add_argument("--kill-ready-after", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = validate_stress_output_root(args.output_root)
    if args.summarize_only:
        summary = summarize_stress_root(root)
        _write_json(root / "stress_run_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    providers = tuple(args.providers or PROVIDER_SPECS)
    if args.kill_ready_after is not None and len(providers) != 1:
        raise SystemExit("--kill-ready-after requires exactly one provider")
    results = []
    for provider in providers:
        result = run_provider_stress(
            provider,
            output_root=root,
            call_count=args.calls,
            checkpoint_stop_after=args.kill_ready_after,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if result["status"] == "checkpoint_ready":
            while True:
                time.sleep(60)
    summary = summarize_stress_root(root)
    _write_json(root / "stress_run_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
