#!/usr/bin/env python3
"""
Three implementations of RetrieverPort:
- LocalBgeRetriever: wraps the existing YihuierRetriever from core/retrieve.py (local 2GB BGE-M3).
- NullRetriever: a safe empty implementation for zero-config / no-model setups, so the demo can still run.
- CloudVectorRetriever: placeholder for a cloud vector service (to be wired up in stage 2, for the app backend, zero model dependencies).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import EMPTY_DIAGNOSIS

SKILL_DIR = Path(__file__).parent.parent.parent


class NullRetriever:
    """A safe no-retrieval implementation: depends on no model and always returns an empty diagnosis.

    Use cases: running locally with zero config, unit tests, or when the retrieval
    model isn't ready. The engine layer then falls back to the "built-in KG + item
    bank" path.
    """

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return []

    def retrieve_with_diagnosis(self, query: str) -> Dict[str, Any]:
        return dict(EMPTY_DIAGNOSIS)


class LocalBgeRetriever:
    """Wraps the existing YihuierRetriever (local BGE-M3 + FAISS).

    Lazy loading: the 2GB model is only loaded on the first actual retrieval;
    construction alone doesn't trigger it. On load failure it silently degrades
    to NullRetriever behavior — the demo must never crash.
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
    """Placeholder for a cloud vector service (to be wired up in stage 2).

    Goal: the app backend doesn't need the 2GB model installed; it will call
    Alibaba DashVector / Tencent's vector service. In stage 1 only the interface
    exists — calls degrade to an empty diagnosis so development isn't blocked.
    """

    def __init__(self, endpoint: str = "", api_key: str = ""):
        self.endpoint = endpoint
        self.api_key = api_key

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        # TODO(stage 2): call the cloud vector service.
        return []

    def retrieve_with_diagnosis(self, query: str) -> Dict[str, Any]:
        # TODO(stage 2): call the cloud vector service + cloud KG index.
        return dict(EMPTY_DIAGNOSIS)
