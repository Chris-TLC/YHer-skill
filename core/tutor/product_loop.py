"""Product-facing contracts for the narrow chemistry learning loop."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from core.data.item_repository import get_item_repository
from core.data.knowledge_repository import KGNode, RecommendedVideo, get_knowledge_repository


def build_nodes_contract() -> dict[str, Any]:
    """Return KG parent nodes with nested children for the home page."""

    repo = get_knowledge_repository()
    nodes = repo.all_nodes()
    by_id = {n.node_id: n for n in nodes}
    children_by_parent: dict[str, list[KGNode]] = {}
    parent_by_id: dict[str, KGNode] = {}

    for node in nodes:
        if "-" in node.node_id:
            parent_id = node.node_id.split("-", 1)[0]
            children_by_parent.setdefault(parent_id, []).append(node)
            if parent_id not in by_id and parent_id not in parent_by_id:
                parent_by_id[parent_id] = replace(node, node_id=parent_id)
            continue
        parent_by_id[node.node_id] = node

    for parent_id in children_by_parent:
        if parent_id in by_id:
            parent_by_id[parent_id] = by_id[parent_id]

    return {
        "total": len(nodes),
        "parents": [
            _node_contract(parent, children_by_parent.get(parent.node_id, []))
            for parent in parent_by_id.values()
        ],
    }


def _node_contract(node: KGNode, children: list[KGNode]) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "label": node.node_id,
        "category": node.category,
        "difficulty": node.difficulty,
        "exam_weight": node.exam_weight,
        "has_video": bool(node.videos),
        "children": [
            {
                "node_id": child.node_id,
                "label": child.node_id.split("-", 1)[1] if "-" in child.node_id else child.node_id,
                "category": child.category,
                "difficulty": child.difficulty,
                "exam_weight": child.exam_weight,
                "has_video": bool(child.videos),
            }
            for child in children
        ],
    }


def build_video_recommendation(node_id: str) -> dict[str, Any]:
    """Return student-facing video data without exposing internal BV fields."""

    repo = get_knowledge_repository()
    node = repo.get_node(node_id)
    videos = list(getattr(node, "videos", []) or []) if node else []
    used_node = node_id

    if not videos and node_id and "-" in node_id:
        parent_id = node_id.split("-", 1)[0]
        parent = repo.get_node(parent_id)
        if parent and parent.videos:
            videos = list(parent.videos)
            used_node = parent_id
    if not videos and node_id and "-" not in node_id:
        child = next(
            (
                candidate
                for candidate in repo.all_nodes()
                if candidate.node_id.startswith(f"{node_id}-") and candidate.videos
            ),
            None,
        )
        if child:
            videos = list(child.videos)
            used_node = child.node_id

    if not videos:
        return {
            "node_id": node_id,
            "video_source_node": used_node,
            "status": "organizing",
            "videos": [],
            "fallback": {
                "reason": "video_metadata_missing",
                "message": "这个知识点的视频正在整理中，可以先选择其他知识点或进入低风险练习。",
            },
        }

    return {
        "node_id": node_id,
        "video_source_node": used_node,
        "status": "ready",
        "videos": [_student_video(v, used_node) for v in videos],
        "fallback": None,
    }


def _student_video(video: RecommendedVideo | dict[str, Any], source_node: str) -> dict[str, Any]:
    def field(key: str, default: Any = "") -> Any:
        if isinstance(video, dict):
            return video.get(key, default)
        return getattr(video, key, default)

    bv = str(field("bv", ""))
    p_number = field("p_number", 1) or 1
    url = getattr(video, "url", None) if not isinstance(video, dict) else video.get("url")
    if not isinstance(url, str) or not url:
        url = f"https://www.bilibili.com/video/{bv}?p={p_number}"

    title = (
        field("title", "")
        or field("video_title", "")
        or field("short_title", "")
        or f"{source_node}核心讲解"
    )
    return {
        "title": title,
        "duration_min": field("duration_min", 0) or 0,
        "what_you_learn": field("what_you_learn", "") or f"系统学习{source_node}的核心方法",
        "completion_criterion": field("completion_criterion", "") or "能独立完成一道基础验证题",
        "url": url,
        "source": "metadata_cache" if field("title", "") or field("video_title", "") else "curated_fallback",
    }


def build_formal_question_contract(item: dict[str, Any], status: dict[str, Any] | None = None) -> dict[str, Any]:
    needs_image = bool((status or {}).get("needs_image"))
    return {
        "question_id": item.get("item_id", ""),
        "display_role": "formal_diagnosis",
        "quality_tier": "strong",
        "profile_allowed": True,
        "counts_toward_formal_total": True,
        "level": _level_for_item(item),
        "axis": _axis_for_item(item),
        "prompt": item.get("stem", ""),
        "options": item.get("options") or {},
        "source": "item_bank",
        "needs_image": needs_image,
        "image": _image_contract(status) if needs_image else None,
    }


def build_warmup_contract(node_id: str) -> dict[str, Any]:
    return {
        "question_id": f"warmup:{node_id}",
        "display_role": "warmup_only",
        "quality_tier": "warmup",
        "profile_allowed": False,
        "counts_toward_formal_total": False,
        "level": "L0 自我定位",
        "axis": "metacognition",
        "prompt": "开场了解：你最近做这个知识点时，最容易卡在概念、题型入口、步骤还是计算？",
        "options": {},
        "source": "warmup",
        "needs_image": False,
        "image": None,
    }


def select_formal_questions(node_id: str, limit: int = 5) -> list[dict[str, Any]]:
    repo = get_item_repository()
    items = repo.find_items(kg_node=node_id, limit=limit, purpose="diagnosis")
    questions: list[dict[str, Any]] = []
    for item in items:
        status = repo.quality_gate.status_for(item.get("item_id", ""))
        questions.append(build_formal_question_contract(item, status))
    return questions


def build_verification_item(node_id: str, item: dict[str, Any], index: int) -> dict[str, Any]:
    status = get_item_repository().quality_gate.status_for(item.get("item_id", ""))
    return {
        "verification_id": f"verify_{node_id}_{index + 1}",
        "question": build_formal_question_contract(item, status),
        "quality_tier": "strong",
        "profile_allowed": True,
        "dimension": _dimension_for_node(node_id),
        "pass_threshold": 0.7,
    }


def build_next_plan(node_id: str, *, verification_passed: bool) -> dict[str, Any]:
    if verification_passed:
        steps = [
            f"做 2 道{node_id}同类题，保持刚通过的解题路径。",
            "把错因和关键步骤各写一句，作为下次复习入口。",
            "再做 1 道验证题，确认能独立完成。",
        ]
    else:
        steps = [
            f"重看推荐讲解中与{node_id}相关的核心片段。",
            "补做 1 道基础题，只检查题型入口和关键条件。",
            "重新完成验证题，通过后再更新能力画像。",
        ]
    return {
        "structure_source": "rules",
        "polish_source": "fallback",
        "steps": steps,
    }


def _level_for_item(item: dict[str, Any]) -> str:
    difficulty = item.get("difficulty", "")
    if difficulty == "T1":
        return "L1 基础概念"
    if difficulty == "T3":
        return "L3 综合应用"
    if difficulty == "T4":
        return "L4 拔高迁移"
    return "L2 应用迁移"


def _axis_for_item(item: dict[str, Any]) -> str:
    qtype = item.get("question_type") or ""
    if "选择" in qtype:
        return "审题入口"
    if "填空" in qtype:
        return "关键表达"
    return "综合推理"


def _dimension_for_node(node_id: str) -> str:
    node = get_knowledge_repository().find_node(node_id)
    category = (node.category if node else "") or node_id
    if "实验" in category or "实验" in node_id:
        return "实验分析"
    if "有机" in category or "有机" in node_id:
        return "有机推断"
    if "电" in category or "电" in node_id:
        return "电化学分析"
    return "基础概念"


def _image_contract(status: dict[str, Any] | None) -> dict[str, Any] | None:
    if not status:
        return None
    path = status.get("display_image_path") or status.get("crop_path") or status.get("page_image_path")
    if not path:
        return None
    return {
        "path": path,
        "hash": status.get("display_image_hash") or status.get("crop_hash") or status.get("page_image_hash"),
        "source_page": status.get("page"),
    }
