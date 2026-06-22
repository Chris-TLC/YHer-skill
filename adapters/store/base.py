#!/usr/bin/env python3
"""
存储 Port 抽象（总蓝图 B-6）。

引擎层只认 StorePort，本地用 JSON（零配置），云端用 Supabase。
存的是：学生模型、会话状态、学习事件（活人感③ 事件级记忆的落点）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class StorePort(Protocol):
    """存储后端必须满足的接口。"""

    # 学生模型（跨会话稳定层）
    def load_student(self, user_id: str) -> Optional[Dict[str, Any]]: ...
    def save_student(self, user_id: str, model: Dict[str, Any]) -> None: ...

    # 会话（一次托管）
    def save_session(self, session_id: str, session: Dict[str, Any]) -> None: ...
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    # 学习事件（活人感③：记"怎么错的+哪个讲法点通了"）
    def append_event(self, user_id: str, event: Dict[str, Any]) -> None: ...
    def recent_events(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]: ...
