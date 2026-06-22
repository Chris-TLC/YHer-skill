#!/usr/bin/env python3
"""
检索 Port 抽象（总蓝图第 2/B-5 章的关键边界）。

为什么要这层：
- 本地开发用 BGE-M3（2GB 模型），但手机端 App 后端塞不下 2GB 模型。
- 引擎层只认 RetrieverPort 协议，永远不知道背后是本地 BGE 还是云端向量服务。
- 切换检索后端 = 换一个实现类，引擎层零改动。
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class RetrieverPort(Protocol):
    """所有检索实现必须满足的接口。"""

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """返回一组 chunk（每个含 bv/p_number/text_preview/collection 等）。"""
        ...

    def retrieve_with_diagnosis(self, query: str) -> Dict[str, Any]:
        """返回带诊断的检索结果（related_nodes/missing_prereqs/chunks/recommended_videos…）。"""
        ...


# 引擎层使用的统一空诊断结构（无检索器时的安全默认）
EMPTY_DIAGNOSIS: Dict[str, Any] = {
    "grade_signal": "unknown",
    "complexity": "normal",
    "related_nodes": [],
    "missing_prereqs": [],
    "exam_patterns": [],
    "thinking_names": [],
    "chunks": [],
    "recommended_videos": [],
    "elapsed_ms": 0,
}
