#!/usr/bin/env python3
"""
RetrieverPort 的三个实现：
- LocalBgeRetriever：包装现有 core/retrieve.py 的 YihuierRetriever（本地 2GB BGE-M3）。
- NullRetriever：零配置/无模型时的安全空实现，让 Demo 也能跑。
- CloudVectorRetriever：云端向量服务占位（阶段二实接，App 后端用，零模型依赖）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import EMPTY_DIAGNOSIS

SKILL_DIR = Path(__file__).parent.parent.parent


class NullRetriever:
    """无检索能力的安全实现：不依赖任何模型，永远返回空诊断。

    用途：本地零配置跑、单元测试、检索模型未就绪时。引擎层据此走"内置 KG 题库"路径。
    """

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return []

    def retrieve_with_diagnosis(self, query: str) -> Dict[str, Any]:
        return dict(EMPTY_DIAGNOSIS)


class LocalBgeRetriever:
    """包装现有 YihuierRetriever（本地 BGE-M3 + FAISS）。

    懒加载：只有第一次真正检索时才加载 2GB 模型，构造本身不触发。
    加载失败时自动降级为 NullRetriever 行为，绝不让 Demo 崩。
    """

    def __init__(self, embeddings_dir: Optional[str] = None):
        self._embeddings_dir = embeddings_dir or str(SKILL_DIR / "data" / "embeddings")
        self._impl = None
        self._failed = False

    def _ensure(self):
        if self._impl is not None or self._failed:
            return
        try:
            from core.retrieve import YihuierRetriever
            self._impl = YihuierRetriever(embeddings_dir=self._embeddings_dir)
        except Exception:
            self._failed = True
            self._impl = None

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        self._ensure()
        if self._impl is None:
            return []
        try:
            return self._impl.retrieve(query, top_k=top_k)
        except Exception:
            return []

    def retrieve_with_diagnosis(self, query: str) -> Dict[str, Any]:
        self._ensure()
        if self._impl is None:
            return dict(EMPTY_DIAGNOSIS)
        try:
            return self._impl.retrieve_with_diagnosis(query)
        except Exception:
            return dict(EMPTY_DIAGNOSIS)


class CloudVectorRetriever:
    """云端向量服务占位（阶段二实接）。

    目标：App 后端无需装 2GB 模型，调用阿里 DashVector / 腾讯向量服务。
    阶段一只留接口，调用时退化为空诊断，不阻塞开发。
    """

    def __init__(self, endpoint: str = "", api_key: str = ""):
        self.endpoint = endpoint
        self.api_key = api_key

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # TODO(阶段二): 调云端向量服务。
        return []

    def retrieve_with_diagnosis(self, query: str) -> Dict[str, Any]:
        # TODO(阶段二): 调云端向量服务 + 云端 KG 索引。
        return dict(EMPTY_DIAGNOSIS)
