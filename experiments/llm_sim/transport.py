"""Official-provider HTTP transport used by the optional live S2 run."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from dotenv import dotenv_values

from .models import ProviderSpec


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        "deepseek", "https://api.deepseek.com/v1", "deepseek-v4-pro", "DEEPSEEK_API_KEY",
        {"input": 3.13, "output": 6.26},
    ),
    "glm": ProviderSpec(
        "glm", "https://open.bigmodel.cn/api/paas/v4", "glm-4-plus", "GLM_API_KEY",
        {"input": 50.0, "output": 50.0},
    ),
    "kimi": ProviderSpec(
        "kimi", "https://api.moonshot.cn/v1", "moonshot-v1-128k", "KIMI_API_KEY",
        {"input": 12.0, "output": 12.0},
    ),
    "minimax": ProviderSpec(
        "minimax", "https://api.minimax.chat/v1", "abab6.5s-chat", "MINIMAX_API_KEY",
        {"input": 1.0, "output": 1.0},
    ),
    "doubao": ProviderSpec(
        "doubao", "https://ark.cn-beijing.volces.com/api/v3", "doubao-pro-32k", "DOUBAO_API_KEY",
        {"input": 0.8, "output": 2.0},
    ),
    "tongyi": ProviderSpec(
        "tongyi", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-max", "TONGYI_API_KEY",
        {"input": 20.0, "output": 60.0},
    ),
}

# Existing project .env files historically call GLM ``ZHIPU_API_KEY``.  The
# alias is resolved internally and is never written to artifacts.
PROVIDER_KEY_ALIASES = {
    "glm": ("GLM_API_KEY", "ZHIPU_API_KEY"),
    "zhipu": ("GLM_API_KEY", "ZHIPU_API_KEY"),
    "tongyi": ("TONGYI_API_KEY", "DASHSCOPE_API_KEY"),
}

# Model aliases are deliberately separate from key aliases.  A model override
# is ordinary run configuration, while credentials must remain an opaque value
# that never enters a simulation record.
PROVIDER_MODEL_ALIASES = {
    "glm": ("GLM_MODEL", "ZHIPU_MODEL"),
    "tongyi": ("TONGYI_MODEL", "DASHSCOPE_MODEL"),
    "doubao": ("DOUBAO_MODEL",),
}


def load_live_environment(
    *,
    repo_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return live-only configuration without mutating ``os.environ``.

    The project keeps provider credentials in ``.env``.  Reading that file is
    intentionally explicit and is only called by the live transport path;
    prepare/offline runs never invoke this function.  Process environment
    values win over the file, matching normal dotenv semantics.  The returned
    mapping is held in memory and is never serialized by this package.
    """

    values: dict[str, str] = {
        str(key): str(value)
        for key, value in (dotenv_values(_dotenv_path(repo_root)) or {}).items()
        if value is not None
    }
    source = os.environ if environ is None else environ
    for key, value in source.items():
        if value is not None:
            values[str(key)] = str(value)
    return values


def _dotenv_path(repo_root: str | Path | None = None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[2] / ".env"
    return Path(repo_root).expanduser().resolve(strict=False) / ".env"


class ProviderTransport(Protocol):
    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class ProviderConfigurationError(RuntimeError):
    pass


class HTTPProviderTransport:
    """Minimal OpenAI-compatible HTTP client; it does not contact 8700."""

    def __init__(self, spec: ProviderSpec, api_key: str, *, user_agent: str = "yher-llm-sim/1"):
        if not api_key or not api_key.strip():
            raise ProviderConfigurationError(f"missing API key for provider {spec.name}")
        self.spec = spec
        self._api_key = api_key
        self.user_agent = user_agent

    @classmethod
    def from_environment(
        cls,
        provider: str,
        *,
        environment: Mapping[str, str] | None = None,
        repo_root: str | Path | None = None,
    ) -> "HTTPProviderTransport":
        spec = provider_spec(provider)
        env = dict(environment or load_live_environment(repo_root=repo_root))
        env_names = PROVIDER_KEY_ALIASES.get(provider, (spec.key_env,))
        key = next((env.get(name, "") for name in env_names if env.get(name)), "")
        return cls(spec, key)

    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if provider != self.spec.name:
            raise ProviderConfigurationError("transport/provider mismatch")
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": 0, "max_tokens": 256},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.spec.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                body = response.read()
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            # Do not relay response bodies: providers occasionally echo request
            # metadata, and a key must never appear in logs or artifacts.
            status = int(exc.code)
            raise ProviderHTTPError(status) from None
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise ProviderNetworkError() from exc
        if status >= 400:
            raise ProviderHTTPError(status)
        try:
            raw = json.loads(body.decode("utf-8"))
            choice = raw["choices"][0]
            content = choice.get("message", {}).get("content", "")
            returned_model = str(raw.get("model") or "").strip()
            if not returned_model:
                raise ProviderProtocolError()
            usage = raw.get("usage") or {}
            input_tokens = max(0, int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0))
            output_tokens = max(0, int(usage.get("completion_tokens") or usage.get("output_tokens") or 0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderProtocolError() from exc
        cost = (
            input_tokens * float(self.spec.pricing.get("input") or 0.0) / 1_000_000
            + output_tokens * float(self.spec.pricing.get("output") or 0.0) / 1_000_000
        )
        return {
            "content": str(content),
            "model_returned": returned_model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "cost_yuan": cost,
        }


class ProviderHTTPError(RuntimeError):
    def __init__(self, status: int):
        self.status = int(status)
        super().__init__(f"provider HTTP {self.status}")


class ProviderNetworkError(RuntimeError):
    def __init__(self):
        super().__init__("provider network or timeout error")


class ProviderProtocolError(RuntimeError):
    def __init__(self):
        super().__init__("provider returned an invalid chat response")


def provider_spec(provider: str) -> ProviderSpec:
    key = str(provider).strip().lower()
    if key == "zhipu":
        key = "glm"
    try:
        return PROVIDER_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"unknown S2 provider: {provider}") from exc


def model_from_environment(
    provider: str,
    default: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Honor an explicitly configured endpoint model without exposing secrets."""

    env = os.environ if environment is None else environment
    aliases = PROVIDER_MODEL_ALIASES.get(
        str(provider).strip().lower(),
        (str(provider).strip().upper() + "_MODEL",),
    )
    value = next((str(env.get(key, "")).strip() for key in aliases if env.get(key)), "")
    return value or default
