#!/usr/bin/env python3
"""存储 Port 工厂。引擎层只调 make_store()，由 mode 决定后端。"""

from __future__ import annotations

from .base import StorePort
from .local_json import LocalJsonStore

__all__ = ["StorePort", "LocalJsonStore", "make_store"]


def make_store(mode: str = "local", **kwargs) -> StorePort:
    """
    mode:
      - "local"：本地 JSON（零配置，开发/测试）。
      - "supabase"：云端（阶段二/上线，待实现）。
    """
    if mode == "supabase":
        # TODO(上线): 返回 SupabaseStore(...)，复用 adapters/memory.py。
        raise NotImplementedError("SupabaseStore 待上线阶段实现")
    return LocalJsonStore(**kwargs)
