"""Official-provider HTTP transport used by the optional live S2 run."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
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

TRANSPORT_V2_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "llm_transport_v2.json"
)


@dataclass(frozen=True)
class ProviderRuntimePolicy:
    max_tokens: int
    retry_max_tokens: int
    timeout_seconds: float
    concurrency: int
    max_attempts: int
    failure_threshold: int
    base_backoff_seconds: float
    max_backoff_seconds: float
    cooldown_seconds: float
    jitter_fraction: float

    def __post_init__(self) -> None:
        if self.max_tokens < 1 or self.retry_max_tokens < self.max_tokens:
            raise ValueError("transport token budgets are invalid")
        if self.timeout_seconds < 0 or self.concurrency < 1:
            raise ValueError("transport timeout/concurrency are invalid")
        if self.max_attempts < 1 or self.failure_threshold < 1:
            raise ValueError("transport retry/circuit thresholds are invalid")
        if self.base_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("transport backoff is invalid")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("transport jitter fraction is invalid")


_V1_POLICY = ProviderRuntimePolicy(
    max_tokens=256,
    retry_max_tokens=256,
    timeout_seconds=0,
    concurrency=1,
    max_attempts=3,
    failure_threshold=3,
    base_backoff_seconds=1.0,
    max_backoff_seconds=30.0,
    cooldown_seconds=120.0,
    jitter_fraction=0.0,
)


@lru_cache(maxsize=1)
def _v2_policies() -> dict[str, ProviderRuntimePolicy]:
    try:
        payload = json.loads(TRANSPORT_V2_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderConfigurationError("invalid transport v2 configuration") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "yher.llm_sim.transport.v2"
        or payload.get("version") != "v2"
        or not isinstance(payload.get("providers"), Mapping)
    ):
        raise ProviderConfigurationError("invalid transport v2 configuration")
    rows: dict[str, ProviderRuntimePolicy] = {}
    for provider in PROVIDER_SPECS:
        raw = payload["providers"].get(provider)
        if not isinstance(raw, Mapping):
            raise ProviderConfigurationError(
                f"transport v2 policy missing provider {provider}"
            )
        try:
            rows[provider] = ProviderRuntimePolicy(
                max_tokens=int(raw["max_tokens"]),
                retry_max_tokens=int(raw["retry_max_tokens"]),
                timeout_seconds=float(raw["timeout_seconds"]),
                concurrency=int(raw["concurrency"]),
                max_attempts=int(raw["max_attempts"]),
                failure_threshold=int(raw["failure_threshold"]),
                base_backoff_seconds=float(raw["base_backoff_seconds"]),
                max_backoff_seconds=float(raw["max_backoff_seconds"]),
                cooldown_seconds=float(raw["cooldown_seconds"]),
                jitter_fraction=float(raw["jitter_fraction"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderConfigurationError(
                f"transport v2 policy invalid for provider {provider}"
            ) from exc
    if set(payload["providers"]) != set(PROVIDER_SPECS):
        raise ProviderConfigurationError("transport v2 provider set differs")
    return rows


def transport_policy(provider: str, *, version: str = "v1") -> ProviderRuntimePolicy:
    name = provider_spec(provider).name
    if version == "v1":
        return _V1_POLICY
    if version == "v2":
        return _v2_policies()[name]
    raise ValueError(f"unknown transport version: {version}")


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
        max_tokens: int | None = None,
    ) -> Mapping[str, Any]: ...


class ProviderConfigurationError(RuntimeError):
    pass


class HTTPProviderTransport:
    """Minimal OpenAI-compatible HTTP client; it does not contact 8700."""

    def __init__(
        self,
        spec: ProviderSpec,
        api_key: str,
        *,
        runtime_policy: ProviderRuntimePolicy | None = None,
        user_agent: str = "yher-llm-sim/1",
    ):
        if not api_key or not api_key.strip():
            raise ProviderConfigurationError(f"missing API key for provider {spec.name}")
        self.spec = spec
        self._api_key = api_key
        self.runtime_policy = runtime_policy or _V1_POLICY
        self.user_agent = user_agent

    @classmethod
    def from_environment(
        cls,
        provider: str,
        *,
        environment: Mapping[str, str] | None = None,
        repo_root: str | Path | None = None,
        version: str = "v1",
    ) -> "HTTPProviderTransport":
        spec = provider_spec(provider)
        env = dict(environment or load_live_environment(repo_root=repo_root))
        env_names = PROVIDER_KEY_ALIASES.get(provider, (spec.key_env,))
        key = next((env.get(name, "") for name in env_names if env.get(name)), "")
        return cls(spec, key, runtime_policy=transport_policy(provider, version=version))

    def complete(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if provider != self.spec.name:
            raise ProviderConfigurationError("transport/provider mismatch")
        request_max_tokens = int(max_tokens or self.runtime_policy.max_tokens)
        if request_max_tokens < 1:
            raise ProviderConfigurationError("max_tokens must be positive")
        effective_timeout = max(
            float(timeout_seconds), float(self.runtime_policy.timeout_seconds)
        )
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": request_max_tokens,
            },
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
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=effective_timeout) as response:
                body = response.read()
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            # Do not relay response bodies: providers occasionally echo request
            # metadata, and a key must never appear in logs or artifacts.
            status = int(exc.code)
            retry_after = _retry_after_seconds(exc.headers)
            raise ProviderHTTPError(status, retry_after_seconds=retry_after) from None
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise ProviderNetworkError() from exc
        latency_ms = (time.monotonic() - started) * 1000.0
        if status >= 400:
            raise ProviderHTTPError(status)
        try:
            raw = json.loads(body.decode("utf-8"))
            if not isinstance(raw, Mapping):
                raise TypeError("response is not an object")
            choices = raw["choices"]
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise TypeError("choice is not an object")
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise TypeError("message is not an object")
            content = str(message.get("content") or "")
            finish_reason = str(choice.get("finish_reason") or "").strip()
            returned_model = str(raw.get("model") or "").strip()
            if not returned_model:
                raise ProviderProtocolError()
            usage = raw.get("usage") or {}
            if not isinstance(usage, Mapping):
                raise TypeError("usage is not an object")
            input_tokens = max(0, int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0))
            output_tokens = max(0, int(usage.get("completion_tokens") or usage.get("output_tokens") or 0))
            reasoning_tokens = _reasoning_tokens(usage)
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderProtocolError() from exc
        cost = (
            input_tokens * float(self.spec.pricing.get("input") or 0.0) / 1_000_000
            + output_tokens * float(self.spec.pricing.get("output") or 0.0) / 1_000_000
        )
        if not content.strip():
            if finish_reason == "length":
                raise ProviderTruncatedResponseError(
                    finish_reason=finish_reason,
                    request_max_tokens=request_max_tokens,
                    reasoning_tokens=reasoning_tokens,
                    returned_model=returned_model,
                    usage={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    cost_yuan=cost,
                    latency_ms=latency_ms,
                )
            raise ProviderProtocolError("provider returned empty content")
        return {
            "content": content,
            "model_returned": returned_model,
            "finish_reason": finish_reason,
            "reasoning_tokens": reasoning_tokens,
            "request_max_tokens": request_max_tokens,
            "latency_ms": latency_ms,
            "http_status": status,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "cost_yuan": cost,
        }


class ProviderHTTPError(RuntimeError):
    def __init__(self, status: int, *, retry_after_seconds: float | None = None):
        self.status = int(status)
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"provider HTTP {self.status}")


class ProviderNetworkError(RuntimeError):
    def __init__(self):
        super().__init__("provider network or timeout error")


class ProviderProtocolError(RuntimeError):
    def __init__(self, message: str = "provider returned an invalid chat response"):
        super().__init__(message)


class ProviderTruncatedResponseError(ProviderProtocolError):
    def __init__(
        self,
        *,
        finish_reason: str,
        request_max_tokens: int,
        reasoning_tokens: int,
        returned_model: str | None = None,
        usage: Mapping[str, int] | None = None,
        cost_yuan: float = 0.0,
        latency_ms: float = 0.0,
    ):
        self.finish_reason = finish_reason
        self.request_max_tokens = int(request_max_tokens)
        self.reasoning_tokens = int(reasoning_tokens)
        self.returned_model = returned_model
        self.usage = dict(usage or {})
        self.cost_yuan = max(0.0, float(cost_yuan))
        self.latency_ms = max(0.0, float(latency_ms))
        super().__init__("provider exhausted output budget before producing content")


def _reasoning_tokens(usage: Mapping[str, Any]) -> int:
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        if isinstance(details, Mapping) and details.get("reasoning_tokens") is not None:
            return max(0, int(details.get("reasoning_tokens") or 0))
    return max(0, int(usage.get("reasoning_tokens") or 0))


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
        return max(0.0, float(value)) if value not in (None, "") else None
    except (AttributeError, TypeError, ValueError):
        return None


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
