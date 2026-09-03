#!/usr/bin/env python3
"""
LLM bridge: wraps adapters/llm_client.LLMClient into the
(system, user) -> dict signature the orchestration layer needs.

The orchestration layer only knows this simple signature and never depends on
a concrete LLM SDK directly, which makes swapping models and mocking tests easy.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def make_llm_caller(
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    model: Optional[str] = None,  # ✨ new: supports specifying a model
    max_tokens: int = 3000,
    temperature: float = 0.3,
) -> Callable[[str, str, Optional[str]], Dict[str, Any]]:
    """
    Return a (system_prompt, user_prompt, model_override?) -> {content, cost_yuan, usage} function.

    Args:
        provider: API provider
        api_key: API key
        model: default model (can be overridden per call via model_override)
        max_tokens: max output tokens
        temperature: temperature

    Returns:
        The call function, which supports specifying a model at runtime
    """
    from adapters.llm_client import LLMClient

    client = LLMClient(provider=provider, model=model, api_key=api_key)

    def _call(system_prompt: str, user_prompt: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Call the LLM, with runtime model override support.

        Args:
            system_prompt: system prompt
            user_prompt: user prompt
            model_override: model specified at runtime (e.g. "deepseek-v4-flash"), overriding the default
        """
        # Create a temporary client for the different model (when specified)
        if model_override:
            temp_client = LLMClient(provider=provider, model=model_override, api_key=api_key)
            resp = temp_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            resp = client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return {
            "content": resp["content"],
            "cost_yuan": resp.get("cost_yuan", 0.0),
            "usage": resp.get("usage", {}),
            "model_used": resp.get("model_returned", model_override or model or "unknown"),
        }

    return _call
