#!/usr/bin/env python3
"""
LLM 桥接：把 adapters/llm_client.LLMClient 包成编排层需要的 (system, user) -> dict 签名。

编排层只认这个简单签名，不直接依赖具体 LLM SDK，便于换模型/测试 mock。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def make_llm_caller(
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    model: Optional[str] = None,  # ✨ 新增：支持指定模型
    max_tokens: int = 3000,
    temperature: float = 0.3,
) -> Callable[[str, str, Optional[str]], Dict[str, Any]]:
    """
    返回一个 (system_prompt, user_prompt, model_override?) -> {content, cost_yuan, usage} 的函数。

    Args:
        provider: API提供商
        api_key: API密钥
        model: 默认模型（可被每次调用的model_override覆盖）
        max_tokens: 最大输出token数
        temperature: 温度

    Returns:
        调用函数，支持运行时指定模型
    """
    from adapters.llm_client import LLMClient

    client = LLMClient(provider=provider, model=model, api_key=api_key)

    def _call(system_prompt: str, user_prompt: str, model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        调用LLM，支持运行时覆盖模型。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            model_override: 运行时指定的模型（如"deepseek-v4-flash"），覆盖默认值
        """
        # 临时创建新client以使用不同模型（如果指定了）
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
