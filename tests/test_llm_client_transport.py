"""Offline transport-contract tests for the shared LLM client."""

from __future__ import annotations


def test_openai_compatible_client_disables_sdk_retries_for_wall_clock_deadline(
    monkeypatch,
) -> None:
    import openai

    from adapters.llm_client import LLMClient

    captured: dict[str, object] = {}

    class CapturingOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", CapturingOpenAI)

    LLMClient(provider="deepseek", api_key="offline-test-key")

    assert captured["max_retries"] == 0
