#!/usr/bin/env python3
"""检索 Port 工厂。引擎层只调 make_retriever()，由 mode 决定后端。"""

from __future__ import annotations

from .base import EMPTY_DIAGNOSIS, RetrieverPort
from .implementations import CloudVectorRetriever, LocalBgeRetriever, NullRetriever

__all__ = [
    "RetrieverPort",
    "EMPTY_DIAGNOSIS",
    "LocalBgeRetriever",
    "CloudVectorRetriever",
    "NullRetriever",
    "make_retriever",
]


def make_retriever(mode: str = "local", **kwargs) -> RetrieverPort:
    """
    mode:
      - "local"：本地 BGE-M3（开发/Chris 本地用）。加载失败自动退化为空。
      - "cloud"：云端向量服务（阶段二，App 后端用）。
      - "null" ：无检索（零配置/测试），走内置 KG 题库路径。
    """
    if mode == "null":
        return NullRetriever()
    if mode == "cloud":
        return CloudVectorRetriever(**kwargs)
    return LocalBgeRetriever(**kwargs)
