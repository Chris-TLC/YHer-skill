#!/usr/bin/env python3
"""
一化儿 AI 化学助手 - LLM 抽象层
v3：支持 8 个主流 API 提供商，统一 OpenAI 格式接口
"""

from typing import List, Dict, Optional


class LLMClient:
    """统一 LLM 接口，支持 8 个主流提供商"""

    PROVIDER_CONFIGS = {
        'deepseek': {
            'base_url': 'https://api.deepseek.com/v1',
            'model_default': 'deepseek-v4-pro',
            'context_window': 1_000_000,
            'sdk': 'openai',
            'label': 'DeepSeek（推荐，国内最快，最便宜）',
            'key_link': 'https://platform.deepseek.com/api_keys',
            'pricing': {
                'input_miss': 3.13,
                'input_hit': 0.026,
                'output': 6.26,
            }
        },
        'kimi': {
            'base_url': 'https://api.moonshot.cn/v1',
            'model_default': 'moonshot-v1-128k',
            'context_window': 128_000,
            'sdk': 'openai',
            'label': 'Kimi（月之暗面，128K 上下文）',
            'key_link': 'https://platform.moonshot.cn/console/api-keys',
            'pricing': {'input_miss': 12, 'input_hit': 1.2, 'output': 12}
        },
        'tongyi': {
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'model_default': 'qwen-max',
            'context_window': 32_000,
            'sdk': 'openai',
            'label': '通义千问（阿里）',
            'key_link': 'https://dashscope.console.aliyun.com/apiKey',
            'pricing': {'input_miss': 20, 'input_hit': None, 'output': 60}
        },
        'zhipu': {
            'base_url': 'https://open.bigmodel.cn/api/paas/v4',
            'model_default': 'glm-4-plus',
            'context_window': 128_000,
            'sdk': 'openai',
            'label': '智谱 GLM',
            'key_link': 'https://bigmodel.cn/usercenter/apikeys',
            'pricing': {'input_miss': 50, 'input_hit': None, 'output': 50}
        },
        'doubao': {
            'base_url': 'https://ark.cn-beijing.volces.com/api/v3',
            'model_default': 'doubao-pro-32k',
            'context_window': 32_000,
            'sdk': 'openai',
            'label': '豆包（字节）',
            'key_link': 'https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey',
            'pricing': {'input_miss': 0.8, 'input_hit': None, 'output': 2}
        },
        'minimax': {
            'base_url': 'https://api.minimax.chat/v1',
            'model_default': 'abab6.5s-chat',
            'context_window': 245_000,
            'sdk': 'openai_compat',
            'label': 'MiniMax',
            'key_link': 'https://platform.minimax.chat/user-center/basic-information/interface-key',
            'pricing': {'input_miss': 1, 'input_hit': None, 'output': 1}
        },
        'anthropic': {
            'base_url': 'https://api.anthropic.com/v1',
            'model_default': 'claude-sonnet-4-6',
            'context_window': 200_000,
            'sdk': 'anthropic',
            'label': 'Anthropic Claude（海外，需翻墙）',
            'key_link': 'https://console.anthropic.com/settings/keys',
            'pricing': {'input_miss': 105, 'input_hit': 10.5, 'output': 525}
        },
        'openai': {
            'base_url': 'https://api.openai.com/v1',
            'model_default': 'gpt-4o',
            'context_window': 128_000,
            'sdk': 'openai',
            'label': 'OpenAI GPT（海外，需翻墙）',
            'key_link': 'https://platform.openai.com/api-keys',
            'pricing': {'input_miss': 17.5, 'input_hit': 8.75, 'output': 70}
        },
    }

    def __init__(self, provider: str, model: str = None,
                 api_key: str = None, base_url: str = None):
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(
                f"未知 provider: {provider}。"
                f"支持: {list(self.PROVIDER_CONFIGS.keys())}"
            )

        self.provider = provider
        self.config = self.PROVIDER_CONFIGS[provider]
        self.model = model or self.config['model_default']
        self.api_key = api_key
        self.base_url = base_url or self.config['base_url']

        sdk = self.config['sdk']
        if sdk in ('openai', 'openai_compat'):
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif sdk == 'anthropic':
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"未知 SDK 类型: {sdk}")

    def chat(self, messages: list, max_tokens: int = 2000,
             temperature: float = 0.3, enable_caching: bool = True) -> dict:
        """
        统一 chat 接口

        Args:
            messages: OpenAI 格式 [{"role": "system", "content": "..."}, ...]
            max_tokens: 最大输出 tokens
            temperature: 温度参数
            enable_caching: 是否启用 prompt caching

        Returns:
            {
                "content": str,
                "usage": {
                    "input_tokens": int,
                    "output_tokens": int,
                    "cache_hit_tokens": int
                },
                "cost_yuan": float,
                "model_returned": str
            }
        """
        sdk = self.config['sdk']
        if sdk in ('openai', 'openai_compat'):
            return self._chat_openai(messages, max_tokens, temperature)
        elif sdk == 'anthropic':
            return self._chat_anthropic(messages, max_tokens, temperature,
                                        enable_caching)
        else:
            raise ValueError(f"未知 SDK: {sdk}")

    # ── OpenAI 兼容接口（7 家国产 + OpenAI）─────────

    def _chat_openai(self, messages, max_tokens, temperature):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            error_str = str(e).lower()
            if '401' in error_str or 'unauthorized' in error_str or 'invalid api key' in error_str:
                raise ValueError(f"API Key 无效，请检查是否输入正确。获取地址：{self.config.get('key_link', '')}") from e
            if '402' in error_str or 'insufficient' in error_str or 'balance' in error_str or '余额' in error_str:
                raise ValueError("账户余额不足，请前往充值。") from e
            if '429' in error_str or 'rate' in error_str:
                raise ValueError("请求过于频繁，请稍后重试。") from e
            if 'timeout' in error_str or 'connection' in error_str or 'network' in error_str:
                raise ConnectionError("网络异常，请检查网络连接后重试。") from e
            raise

        returned_model = response.model
        self._validate_model(returned_model)

        usage = response.usage
        cache_hit = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
        cache_miss = usage.prompt_tokens - cache_hit

        pricing = self.config['pricing']
        hit_price = pricing.get('input_hit')
        if hit_price is None:
            hit_price = pricing['input_miss']

        cost = (
            cache_hit * hit_price / 1_000_000 +
            cache_miss * pricing['input_miss'] / 1_000_000 +
            usage.completion_tokens * pricing['output'] / 1_000_000
        )

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "cache_hit_tokens": cache_hit,
            },
            "cost_yuan": cost,
            "model_returned": returned_model,
        }

    # ── Anthropic 接口（独立 SDK）───────────────────

    def _chat_anthropic(self, messages, max_tokens, temperature,
                        enable_caching):
        system_msg = next((m for m in messages if m['role'] == 'system'), None)
        user_msgs = [m for m in messages if m['role'] != 'system']

        if system_msg and enable_caching:
            system_param = [{
                "type": "text",
                "text": system_msg['content'],
                "cache_control": {"type": "ephemeral"}
            }]
        else:
            system_param = system_msg['content'] if system_msg else ""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_param,
            messages=user_msgs,
        )

        usage = response.usage
        cache_hit = getattr(usage, 'cache_read_input_tokens', 0) or 0
        cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        cache_miss = usage.input_tokens - cache_hit - cache_creation

        pricing = self.config['pricing']
        cost = (
            cache_hit * pricing['input_hit'] / 1_000_000 +
            cache_miss * pricing['input_miss'] / 1_000_000 +
            usage.output_tokens * pricing['output'] / 1_000_000
        )

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_hit_tokens": cache_hit,
            },
            "cost_yuan": cost,
            "model_returned": response.model,
        }

    # ── 模型校验 ────────────────────────────────────

    def _validate_model(self, returned_model: str):
        """DeepSeek 静默降级保护"""
        if self.provider == 'deepseek':
            if 'v4-pro' not in returned_model.lower():
                raise ValueError(
                    f"模型降级！请求 deepseek-v4-pro，"
                    f"实际返回 {returned_model}"
                )

    # ── 便捷方法 ────────────────────────────────────

    def get_context_window(self) -> int:
        return self.config['context_window']

    def get_pricing(self) -> dict:
        return self.config['pricing']
