#!/usr/bin/env python3
"""
LLM 桥接：把 adapters/llm_client.LLMClient 包成编排层需要的 (system, user) -> dict 签名。

编排层只认这个简单签名，不直接依赖具体 LLM SDK，便于换模型/测试 mock。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def make_llm_caller(
    provider: str = "deepseek", api_key: Optional[str] = None,
    max_tokens: int = 3000, temperature: float = 0.3,
) -> Callable[[str, str], Dict[str, Any]]:
    """返回一个 (system_prompt, user_prompt) -> {content, cost_yuan, usage} 的函数。"""
    from adapters.llm_client import LLMClient

    client = LLMClient(provider=provider, api_key=api_key)

    def _call(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
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
        }

    return _call
