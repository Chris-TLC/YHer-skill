#!/usr/bin/env python3
"""
一化儿 SKILL 回答格式化器
根据诊断结果和复杂度，生成符合产品魂的回答 prompt
"""

import random
from typing import Dict, List


# ── 杰哥开场库（≤ 15 字）─────────────────────────
OPENING_PHRASES = {
    'simple': ['嗯，这题', '明白', '这考的是', '直接说'],
    'normal': ['这题杰哥讲过', '老经典考点了', '看到这题先想'],
    'complex': ['压轴题别慌', '这题坑很多', '先稳住'],
    'diagnostic': ['卡这是吧', '错点不在题', '常见漏洞', '来诊断下'],
}

# ── 杰哥互动库（≤ 20 字）─────────────────────────
CLOSING_PHRASES = [
    '搞清楚了吗？',
    '懂了告诉杰哥一声',
    '哪步不通再问',
    '试一道再来反馈',
    '看完视频再做题',
    '还是这步卡住了？',
]


def _format_chunks_for_prompt(chunks: list) -> str:
    """把 chunks 格式化成 prompt 可读的形式"""
    formatted = []
    for c in chunks[:5]:
        cid = c.get('chunk_id', '')[:40]
        text = c.get('text_preview', '')[:150]
        formatted.append(f"  - [{cid}]: {text}...")
    return '\n'.join(formatted) if formatted else '(无匹配 chunks)'


def _format_videos_for_prompt(videos: list) -> str:
    """格式化推荐视频列表"""
    if not videos:
        return '(无推荐视频)'
    lines = []
    for v in videos[:3]:
        bv = v.get('bv', '')
        pn = v.get('p_number', 1)
        collection = v.get('collection', '其他视频')
        short_title = v.get('short_title') or v.get('video_title', '')[:50]
        preview = v.get('text_preview', '')[:60]
        # bv 已自带 BV 前缀，不要再拼 BV
        url = f"https://www.bilibili.com/video/{bv}?p={pn}"
        lines.append(
            f"  - 【{collection}】P{pn}：{short_title}\n"
            f"    {url}\n"
            f"    片段参考：{preview}..."
        )
    return '\n'.join(lines)


def build_response_prompt(query: str, diagnosis: Dict,
                          style: str = 'auto') -> str:
    """
    生成给 Claude 的指令，让它按一化儿风格回答

    style:
      - 'concise': 三段式（开场 + 直答 + 互动）
      - 'full': 五段式（开场 + 诊断 + 双轨 + 路径 + 互动）
      - 'auto': 根据 complexity 自动选
    """
    if style == 'auto':
        if diagnosis['complexity'] in ('simple',):
            style = 'concise'
        else:
            style = 'full'

    # 提取诊断材料
    chunks_summary = _format_chunks_for_prompt(diagnosis['chunks'])
    videos_str = _format_videos_for_prompt(diagnosis.get('recommended_videos', []))
    missing_prereqs_str = ', '.join(diagnosis['missing_prereqs']) or '无明显缺漏'
    exam_patterns_str = ', '.join(diagnosis['exam_patterns']) or '通用'
    thinking_str = ', '.join(diagnosis.get('thinking_names', []))

    if style == 'concise':
        opening = random.choice(OPENING_PHRASES['simple'])
        closing = random.choice(CLOSING_PHRASES[:2])
        return f"""用杰哥（一化儿）的风格回答以下问题。

【用户问题】{query}

【已识别题型】{exam_patterns_str}
【推荐招式】{thinking_str}

【相关 chunks（杰哥原话片段，参考勿照抄）】
{chunks_summary}

【格式要求 - 三段式】
1. 开场：≤ 15 字（参考：{opening}）
2. 直答：用杰哥口吻给答案 + 1 个考点提醒（出题人想坑你什么）
3. 互动：≤ 20 字（参考：{closing}）

【硬约束】
- 开场 ≤ 15 字
- 结尾 ≤ 20 字
- 总字数 200-400 字
- 不要话痨，不要客套
- 自称用"杰哥"（约 30% 概率）
- 不要照抄 chunks，用杰哥自己的话重新组织"""

    # full 模式
    opening = random.choice(OPENING_PHRASES.get(
        diagnosis['complexity'], OPENING_PHRASES['normal']))
    closing = random.choice(CLOSING_PHRASES)

    return f"""用杰哥（一化儿）的风格回答以下问题。

【用户问题】{query}

【诊断结果】
- 学段：{diagnosis['grade_signal']}
- 涉及知识点：{', '.join(diagnosis['related_nodes'])}
- 推断缺漏前置：{missing_prereqs_str}
- 题型：{exam_patterns_str}
- 推荐招式：{thinking_str}

【相关 chunks（杰哥原话片段，参考勿照抄）】
{chunks_summary}

【推荐视频（真实 BV+P，勿编造）】
{videos_str}

【格式要求 - 五段式】

Section 1（开场，≤ 15 字）:
参考：{opening}

Section 2（根因诊断）:
- 抛出 1-2 个诊断问题让用户自查
- 例如："先问你两个：[问题1]？[问题2]？这两个不熟，根子在 {missing_prereqs_str}"

Section 3（双轨答题）:
- 知识层面：[化学原理]
- 考点层面：[出题人想坑你什么 + 用什么招式破]

Section 4（学习路径）:
- 推荐 2-3 个视频，必须严格按以下完整格式（4 行一组）：
    - 【合集名】P×：视频简短标题
      https://www.bilibili.com/video/BVxxx?p=×
      为什么看这个：xxx（一句话给出完成标准）
- 视频信息只能来自上方"推荐视频"列表，禁止编造或猜测
- BV 号已自带 BV 前缀，写成 BV1xxx 即可，不要写 BVBV

Section 5（互动，≤ 20 字）:
参考：{closing}

【硬约束】
- 不要在 Section 1 写超过 15 字
- 不要在 Section 5 写超过 20 字
- 不要 verbatim 引用 chunks，要用杰哥自己的话重新组织
- 推荐视频必须从上方的推荐视频列表里选（真实存在的 BV+P）
- 总字数 600-1200 字
- 自称用"杰哥"而非"我"（约 30% 概率）
- 不要用英文"""


def format_retrieval_for_prompt(diagnosis: Dict) -> str:
    """v3 新增：把 diagnosis 格式化成 [RETRIEVAL_RESULTS] 部分"""
    parts = []

    nodes = ', '.join(diagnosis.get('related_nodes', [])[:5])
    if nodes:
        parts.append(f"related_nodes: [{nodes}]")

    patterns = ', '.join(diagnosis.get('exam_patterns', [])[:3])
    if patterns:
        parts.append(f"related_patterns: [{patterns}]")

    thinking = ', '.join(diagnosis.get('thinking_names', [])[:3])
    if thinking:
        parts.append(f"recommended_thinking: [{thinking}]")

    prereqs = ', '.join(diagnosis.get('missing_prereqs', [])[:5])
    if prereqs:
        parts.append(f"prerequisites_to_check: [{prereqs}]")

    # chunks
    chunks = diagnosis.get('chunks', [])[:5]
    if chunks:
        chunk_lines = []
        for c in chunks:
            cid = c.get('chunk_id', '')[:40]
            bv = c.get('bv', '')
            pn = c.get('p_number', '')
            text = c.get('text_preview', '')[:150]
            chunk_lines.append(f"  - [{bv}#P{pn}] {text}...")
        parts.append(f"reference_chunks:\n" + '\n'.join(chunk_lines))

    # videos（v3.1 视频引用规范：合集名 + 标题 + URL）
    videos = diagnosis.get('recommended_videos', [])[:3]
    if videos:
        vid_lines = []
        for v in videos:
            bv = v.get('bv', '?')
            pn = v.get('p_number', '?')
            collection = v.get('collection', '其他视频')
            short_title = v.get('short_title') or v.get('video_title', '')[:40]
            url = f"https://www.bilibili.com/video/{bv}?p={pn}"
            vid_lines.append(
                f"  - 【{collection}】P{pn}：{short_title}\n"
                f"    {url}"
            )
        parts.append(f"recommended_videos:\n" + '\n'.join(vid_lines))

    return '\n'.join(parts)


def validate_answer_constraints(answer: str, style: str = 'full') -> Dict:
    """验证回答是否满足硬约束"""
    lines = [l for l in answer.split('\n') if l.strip()]
    issues = []

    # 检查开场（第一句非空行）
    first_line = lines[0] if lines else ''
    if len(first_line) > 15:
        issues.append(f"开场 {len(first_line)} 字 > 15: '{first_line[:30]}...'")

    # 检查结尾（最后一句话）
    last_line = lines[-1] if lines else ''
    if len(last_line) > 20:
        issues.append(f"结尾 {len(last_line)} 字 > 20: '{last_line[:30]}...'")

    # 检查总字数
    total_chars = len(answer)
    if total_chars > 1500:
        issues.append(f"总字数 {total_chars} > 1500")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_chars': total_chars,
        'first_line_len': len(first_line),
        'last_line_len': len(last_line),
    }
