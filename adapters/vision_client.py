#!/usr/bin/env python3
"""
视觉模型客户端：读整页试卷图片，识别方程式/结构式/装置图等。

根治"公式图片提取不出"问题（总蓝图第4章 L2）。
支持国内便宜的多模态模型（OpenAI 兼容接口），统一图片+文本输入。

多 Key 轮换: 支持传入多个 API Key，按页轮换，避免限流。

支持的视觉模型（都走 OpenAI 兼容接口）：
- 通义 qwen-vl-max（阿里，国内快，便宜，推荐）
- 豆包 doubao-vision（字节）
- 智谱 glm-4v
- OpenAI gpt-4o（海外）
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

VISION_CONFIGS = {
    "qwen-vl": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "label": "通义千问 qwen3-vl-plus（阿里，新一代，文档/公式识别强）",
        "key_link": "https://dashscope.console.aliyun.com/apiKey",
        # ¥/百万token (估，视觉按token计)
        "pricing": {"input": 3.0, "output": 9.0},
    },
    "doubao-vision": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-1.5-vision-pro",
        "env_key": "DOUBAO_API_KEY",
        "label": "豆包 vision（字节）",
        "key_link": "https://console.volcengine.com/ark",
        "pricing": {"input": 3.0, "output": 9.0},
    },
    "glm-4v": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-plus",
        "env_key": "ZHIPU_API_KEY",
        "label": "智谱 GLM-4V",
        "key_link": "https://bigmodel.cn/usercenter/apikeys",
        "pricing": {"input": 10.0, "output": 10.0},
    },
    "gpt-4o": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "label": "OpenAI GPT-4o（海外，需翻墙）",
        "key_link": "https://platform.openai.com/api-keys",
        "pricing": {"input": 17.5, "output": 70.0},
    },
}


class VisionClient:
    """统一视觉模型接口。支持多Key轮换。"""

    def __init__(self, provider: str = "qwen-vl", api_key: Optional[str] = None,
                 model: Optional[str] = None, api_keys: Optional[List[str]] = None):
        if provider not in VISION_CONFIGS:
            raise ValueError(f"未知视觉 provider: {provider}，支持 {list(VISION_CONFIGS)}")
        self.provider = provider
        self.config = VISION_CONFIGS[provider]
        self.model = model or self.config["model"]
        self.api_key = api_key or ""
        # 多Key轮换支持
        self.api_keys = api_keys or ([self.api_key] if self.api_key else [])
        if not self.api_keys:
            raise ValueError("未提供任何API Key")
        self._key_idx = 0
        self._lock = threading.Lock()
        self._openai_sdk_available = True
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            self._openai_sdk_available = False
            self.client = None
        else:
            # 默认client用第一个key初始化
            self.client = OpenAI(api_key=self.api_keys[0], base_url=self.config["base_url"],
                                 timeout=90.0, max_retries=2)

    def _get_next_key(self) -> str:
        """轮换获取下一个API Key（线程安全）"""
        with self._lock:
            key = self.api_keys[self._key_idx % len(self.api_keys)]
            self._key_idx += 1
            return key

    def _get_client_for_key(self, key: str):
        """为指定Key创建一个OpenAI客户端（缓存避免重复创建）"""
        if not hasattr(self, '_key_clients'):
            self._key_clients = {}
        if key not in self._key_clients:
            if not self._openai_sdk_available:
                self._key_clients[key] = None
                return None
            from openai import OpenAI
            self._key_clients[key] = OpenAI(
                api_key=key, base_url=self.config["base_url"],
                timeout=90.0, max_retries=2
            )
        return self._key_clients[key]

    @staticmethod
    def encode_image(image_path: Path) -> str:
        """图片转 base64 data URL。"""
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode()
        ext = Path(image_path).suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{b64}"

    def read_page(self, image_path: Path, system_prompt: str, user_prompt: str,
                  max_tokens: int = 4000, timeout: float = 90.0,
                  temperature: float = 0.1) -> Dict[str, Any]:
        """看一页图片，返回识别文本。多Key模式下自动轮换。timeout可覆盖。"""
        image_url = self.encode_image(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        # 多Key模式：使用轮换的Key对应的client
        if len(self.api_keys) > 1:
            key = self._get_next_key()
            client = self._get_client_for_key(key)
        else:
            key = self.api_keys[0]
            client = self.client

        if client is None:
            data = self._chat_completion_http(key, messages, max_tokens, temperature, timeout)
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            usage_data = data.get("usage") or {}
            prompt_tokens = int(usage_data.get("prompt_tokens") or usage_data.get("input_tokens") or 0)
            completion_tokens = int(usage_data.get("completion_tokens") or usage_data.get("output_tokens") or 0)
        else:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                timeout=timeout,
            )
            content = resp.choices[0].message.content
            usage = resp.usage
            prompt_tokens = int(usage.prompt_tokens or 0)
            completion_tokens = int(usage.completion_tokens or 0)
        pricing = self.config["pricing"]
        cost = (prompt_tokens * pricing["input"] / 1e6 +
                completion_tokens * pricing["output"] / 1e6)
        return {
            "content": content,
            "cost_yuan": cost,
            "usage": {"input_tokens": prompt_tokens,
                      "output_tokens": completion_tokens},
        }

    def _chat_completion_http(self, api_key: str, messages: List[Dict[str, Any]],
                              max_tokens: int, temperature: float, timeout: float) -> Dict[str, Any]:
        """OpenAI-compatible chat completion using only the standard library."""
        url = self.config["base_url"].rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vision_http_error status={exc.code} body={body[:500]}") from exc
        return json.loads(body)
