from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_v2_transport_policy_is_provider_specific_and_keeps_v1_traceable() -> None:
    from experiments.llm_sim.transport import transport_policy

    v1 = transport_policy("deepseek", version="v1")
    deepseek = transport_policy("deepseek", version="v2")
    doubao = transport_policy("doubao", version="v2")
    glm = transport_policy("glm", version="v2")

    assert v1.max_tokens == 256
    assert deepseek.max_tokens >= 1024
    assert deepseek.retry_max_tokens > deepseek.max_tokens
    assert doubao.max_tokens >= 1024
    assert doubao.timeout_seconds >= 120
    assert 2 <= doubao.concurrency <= 4
    assert glm.max_tokens >= 256
    assert glm.timeout_seconds >= 60


class _Response:
    def __init__(self, payload: dict[str, object], *, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_http_transport_records_finish_reason_reasoning_and_request_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.llm_sim.transport import HTTPProviderTransport, provider_spec

    captured: dict[str, object] = {}

    def urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(
            {
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"answer":"B","rationale":"x"}'},
                    }
                ],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 90,
                    "completion_tokens_details": {"reasoning_tokens": 70},
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    transport = HTTPProviderTransport.from_environment(
        "deepseek",
        environment={"DEEPSEEK_API_KEY": "secret-not-for-artifacts"},
        version="v2",
    )
    result = transport.complete(
        provider="deepseek",
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "question"}],
        timeout_seconds=5,
        max_tokens=1536,
    )

    assert captured["body"]["max_tokens"] == 1536
    assert captured["timeout"] >= 60
    assert result["finish_reason"] == "stop"
    assert result["reasoning_tokens"] == 70
    assert result["request_max_tokens"] == 1536
    assert result["usage"] == {"input_tokens": 40, "output_tokens": 90}
    assert "secret-not-for-artifacts" not in json.dumps(result)


def test_empty_length_response_is_retryable_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.llm_sim.transport import (
        HTTPProviderTransport,
        ProviderTruncatedResponseError,
    )

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _Response(
            {
                "model": "deepseek-v4-pro",
                "choices": [
                    {"finish_reason": "length", "message": {"content": ""}}
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 1024,
                    "completion_tokens_details": {"reasoning_tokens": 1024},
                },
            }
        ),
    )
    transport = HTTPProviderTransport.from_environment(
        "deepseek",
        environment={"DEEPSEEK_API_KEY": "secret"},
        version="v2",
    )

    with pytest.raises(ProviderTruncatedResponseError) as caught:
        transport.complete(
            provider="deepseek",
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "question"}],
            timeout_seconds=60,
        )

    assert caught.value.finish_reason == "length"
    assert caught.value.reasoning_tokens == 1024
    assert caught.value.request_max_tokens == 1024


def test_malformed_empty_message_is_classified_as_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.llm_sim.transport import (
        HTTPProviderTransport,
        ProviderProtocolError,
    )

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _Response(
            {
                "model": "deepseek-v4-pro",
                "choices": [{"finish_reason": "stop", "message": None}],
                "usage": {},
            }
        ),
    )
    transport = HTTPProviderTransport.from_environment(
        "deepseek",
        environment={"DEEPSEEK_API_KEY": "secret"},
        version="v2",
    )

    with pytest.raises(ProviderProtocolError):
        transport.complete(
            provider="deepseek",
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "question"}],
            timeout_seconds=60,
        )


class _SequenceTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, **kwargs):
        self.calls.append(dict(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return dict(response)


def _success(model: str = "deepseek-v4-pro") -> dict[str, object]:
    return {
        "content": '{"answer":"B","rationale":"persona-shaped"}',
        "model_returned": model,
        "finish_reason": "stop",
        "reasoning_tokens": 12,
        "request_max_tokens": 1024,
        "latency_ms": 25.0,
        "usage": {"input_tokens": 80, "output_tokens": 30},
        "cost_yuan": 0.001,
    }


def test_stress_call_retries_truncation_with_larger_budget_and_records_attempts(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim.stress import run_provider_stress
    from experiments.llm_sim.transport import ProviderTruncatedResponseError

    transport = _SequenceTransport(
        [
            ProviderTruncatedResponseError(
                finish_reason="length",
                request_max_tokens=1024,
                reasoning_tokens=1024,
            ),
            _success(),
        ]
    )
    result = run_provider_stress(
        "deepseek",
        output_root=tmp_path,
        call_count=1,
        transport=transport,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert result["completed"] == 1
    assert result["successful"] == 1
    assert [call["max_tokens"] for call in transport.calls] == [1024, 2048]
    record = json.loads((tmp_path / "deepseek/records/call-000.json").read_text())
    assert record["retry_count"] == 1
    assert [attempt["status"] for attempt in record["attempts"]] == [
        "truncated",
        "success",
    ]
    assert record["requested_model"] == "deepseek-v4-pro"
    assert record["returned_model"] == "deepseek-v4-pro"
    assert record["finish_reason"] == "stop"
    assert "messages" not in record
    assert "api_key" not in record


def test_checkpoint_resume_skips_completed_calls_and_has_no_duplicate_records(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim.stress import run_provider_stress

    first_transport = _SequenceTransport([_success() for _ in range(2)])
    first = run_provider_stress(
        "deepseek",
        output_root=tmp_path,
        call_count=5,
        transport=first_transport,
        checkpoint_stop_after=2,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )
    assert first["status"] == "checkpoint_ready"
    assert first["completed"] == 2

    second_transport = _SequenceTransport([_success() for _ in range(3)])
    resumed = run_provider_stress(
        "deepseek",
        output_root=tmp_path,
        call_count=5,
        transport=second_transport,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert resumed["status"] == "complete"
    assert resumed["completed"] == 5
    assert len(second_transport.calls) == 3
    records = sorted((tmp_path / "deepseek/records").glob("call-*.json"))
    assert len(records) == 5
    call_ids = [json.loads(path.read_text())["call_id"] for path in records]
    assert call_ids == list(range(5))
    checkpoint = json.loads((tmp_path / "deepseek/checkpoint.json").read_text())
    assert checkpoint["completed_call_ids"] == list(range(5))
    evidence = json.loads((tmp_path / "deepseek/resume_evidence.json").read_text())
    assert evidence["pre_kill_completed_call_ids"] == [0, 1]
    assert evidence["resumed_call_ids"] == [2, 3, 4]
    assert evidence["duplicate_record_count"] == 0


def test_429_uses_exponential_backoff_and_persists_retry_accounting(
    tmp_path: Path,
) -> None:
    from experiments.llm_sim.stress import run_provider_stress
    from experiments.llm_sim.transport import ProviderHTTPError

    transport = _SequenceTransport(
        [ProviderHTTPError(429), ProviderHTTPError(429), _success()]
    )
    delays: list[float] = []
    result = run_provider_stress(
        "deepseek",
        output_root=tmp_path,
        call_count=1,
        transport=transport,
        sleep=delays.append,
        random_value=lambda: 0.5,
    )

    assert result["successful"] == 1
    assert delays == [1.0, 2.0]
    record = json.loads((tmp_path / "deepseek/records/call-000.json").read_text())
    assert record["retry_count"] == 2
    assert record["error_counts"] == {"http_429": 2}


def test_consecutive_logical_failures_open_only_the_provider_circuit(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from experiments.llm_sim.stress import run_provider_stress
    from experiments.llm_sim.transport import ProviderNetworkError, transport_policy

    policy = replace(
        transport_policy("deepseek", version="v2"),
        concurrency=1,
        max_attempts=1,
        failure_threshold=3,
    )
    transport = _SequenceTransport([ProviderNetworkError() for _ in range(3)])
    result = run_provider_stress(
        "deepseek",
        output_root=tmp_path,
        call_count=10,
        transport=transport,
        policy=policy,
        sleep=lambda _: None,
        random_value=lambda: 0.5,
    )

    assert result["status"] == "circuit_open"
    assert result["completed"] == 3
    assert result["failed"] == 3
    assert len(transport.calls) == 3


def test_stress_output_rejects_sim_store_and_non_tmp_paths(tmp_path: Path) -> None:
    from experiments.llm_sim.stress import validate_stress_output_root

    with pytest.raises(ValueError, match="/tmp"):
        validate_stress_output_root(Path("data/sim_store/stress"))
    with pytest.raises(ValueError, match="sim_store"):
        validate_stress_output_root(Path("/tmp/sim_store/stress"))
    assert validate_stress_output_root(tmp_path).is_absolute()


def test_conservative_cost_projection_uses_600_journeys_and_safety_factor() -> None:
    from experiments.llm_sim.stress import conservative_cost_projection

    projection = conservative_cost_projection(
        average_call_cost_yuan=0.01,
        calls_per_journey=19,
        journeys=600,
        safety_factor=1.3,
    )

    assert projection == pytest.approx(148.2)
