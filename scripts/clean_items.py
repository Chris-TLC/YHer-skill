#!/usr/bin/env python3
"""
切题产物清洗（入库前的兜底，根治视觉切题的 source 归属乱象）。

视觉模型在 prompt 里自由填 source="{卷名} T题号"，输出不受控，会产生四类脏数据：
  1. 答案/解析碎片被切成"孤儿题"（stem 本身是【解析】/【答案】，没挂回母题）
  2. 中文大题号（T七/八/九）逃过原去重正则（只认阿拉伯数字）
  3. 小问被提升为独立题（T(4)/T(5)）
  4. 题号区间块（T33-36）——合法但 source 形态不一

清洗策略（保守，宁可保留不误杀）：
  · 真题：规范化 source 的题号写法（中文→阿拉伯、去空格、统一 T 前缀）。
  · 答案碎片：先尝试把答案文本挂回同年同题号的母题（写进 raw_answer / 对应 sub_question.answer），
    挂上了就删碎片；挂不上（找不到母题）就保留碎片但打标 _orphan_answer=true，绝不静默丢答案。
  · 判断"是不是答案碎片"看 stem 内容（有无【解析】【答案】等标记 + 是否像题目），不只看 source 标签，
    避免把"3．硼的最高价含氧酸…A. B. C."这种真选择题误删。

用法：
  python3 scripts/clean_items.py --input data/raw_papers/shanghai_all.jsonl
  python3 scripts/clean_items.py --input xxx.jsonl --out xxx.clean.jsonl   # 指定输出
  python3 scripts/clean_items.py --input xxx.jsonl --report               # 只看报告不写文件

清洗不会改原文件，默认写到 <input>.clean.jsonl。入库时把 --input 指向 .clean.jsonl 即可。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ── 题号正则 ───────────────────────────────────────────────────────────
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
           "七": 7, "八": 8, "九": 9, "十": 10}

# stem 本身是答案/解析的标记词
ANS_MARK = re.compile(
    r"【解析】|【答案】|试题分析|解题思路|名师点睛|考点定位|考查点|"
    r"^答案[:：]|^解析[:：]|前一题|前题答案"
)
# 真题特征词（出现说明这是题目而非答案）
Q_MARK = re.compile(
    r"下列|是否|为什么|写出|完成下列|计算|判断|的是[（(]?\s*[）)]?$|可制得|"
    r"结构简式|不正确|正确的是|错误的是|属于"
)


def cn_to_int(s: str) -> int | None:
    """十一→11, 七→7（只覆盖 1-19，够用）。"""
    s = s.strip()
    if s in _CN_NUM:
        return _CN_NUM[s]
    if s.startswith("十") and len(s) == 2 and s[1] in _CN_NUM:  # 十一..十九
        return 10 + _CN_NUM[s[1]]
    if len(s) == 2 and s[0] in _CN_NUM and s[1] == "十":        # 二十..九十（罕见）
        return _CN_NUM[s[0]] * 10
    return None


def parse_qno(source: str):
    """
    从 source 末尾解析题号，返回 (kind, lo, hi, raw_suffix)。
    kind: 'num'(单题) / 'range'(区间) / 'subq'(小问) / 'cn'(中文) / 'none'。
    """
    m = re.search(r"T\s*(.+)$", source or "")
    suf = m.group(1).strip() if m else ""
    if not suf:
        return ("none", None, None, "")
    # 区间 33-36 / 33—36
    mr = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", suf)
    if mr:
        return ("range", int(mr.group(1)), int(mr.group(2)), suf)
    # 纯数字
    if re.fullmatch(r"\d+", suf):
        n = int(suf)
        return ("num", n, n, suf)
    # 小问 (4) / （4）
    ms = re.fullmatch(r"[（(]\s*(\d+)\s*[）)]", suf)
    if ms:
        n = int(ms.group(1))
        return ("subq", n, n, suf)
    # 中文 七 / 十一
    cn = cn_to_int(suf)
    if cn is not None:
        return ("cn", cn, cn, suf)
    # 脏后缀但开头是题号，如 '28补充说明' / '16（续）' → 当作单题 28/16（用于挂答案，
    # 但 kind 标 'messy'，normalize_source 不据此改写 source，避免把不确定的归属落地成干净题号）
    mm = re.match(r"(\d+)", suf)
    if mm:
        n = int(mm.group(1))
        return ("messy", n, n, suf)
    return ("none", None, None, suf)


def leading_qno(stem: str):
    """答案碎片的开头题号，如 '46. 甲苯…' → 46。"""
    m = re.match(r"^\s*(\d+)\s*[.．、]", stem or "")
    return int(m.group(1)) if m else None


def looks_like_question(it: dict) -> bool:
    """像题目而非答案：有小问 / 有选项 / 有真题特征词。"""
    stem = it.get("stem", "") or ""
    if it.get("sub_questions"):
        return True
    if re.search(r"[A-D][．.、]\s*\S", stem):   # A. B. C. D. 选项
        return True
    if Q_MARK.search(stem):
        return True
    return False


def is_answer_noise(it: dict) -> bool:
    """是纯答案/解析碎片：stem 是答案标记 或 source 标了解析，且不像题目。"""
    if looks_like_question(it):
        return False
    src = it.get("source", "")
    stem = (it.get("stem", "") or "").strip()
    src_says = bool(re.search(r"解析|答案|补充|点睛|前题|未显|未完整|名师|考点|非独立|非题号", src))
    stem_says = bool(ANS_MARK.search(stem))
    return stem_says or src_says


def normalize_source(it: dict) -> str:
    """把题号规范成 'T<阿拉伯>' 或 'T<lo>-<hi>'，去空格/中文/括号。卷名部分保持原样。"""
    src = it.get("source", "")
    base = re.sub(r"\s*T\s*.+$", "", src).strip()  # 去掉 T... 后缀
    kind, lo, hi, _ = parse_qno(src)
    if kind == "range":
        return f"{base} T{lo}-{hi}"
    if kind in ("num", "subq", "cn") and lo is not None:
        return f"{base} T{lo}"
    return src  # 无法解析，原样返回（会带 _qno_unparsed 标记）


def build_parent_index(items):
    """按 (year, qno) 建母题索引：题号 → 母题 item（区间块覆盖多个题号）。"""
    idx = {}
    for it in items:
        if is_answer_noise(it):
            continue
        yr = it.get("year")
        kind, lo, hi, _ = parse_qno(it.get("source", ""))
        if lo is None:
            continue
        for n in range(lo, hi + 1):
            idx.setdefault((yr, n), it)
    return idx


def attach_answer(parent: dict, qno: int, answer_text: str) -> bool:
    """把答案文本挂到母题。优先写进对应 sub_id 的 answer（若空），否则追加进 raw_answer。返回是否挂上。"""
    answer_text = answer_text.strip()
    if not answer_text:
        return False
    # 1) 找 sub_id 匹配 qno 且 answer 为空的小问
    for sq in parent.get("sub_questions", []):
        sid = re.search(r"(\d+)", str(sq.get("sub_id", "")))
        if sid and int(sid.group(1)) == qno and not (sq.get("answer") or "").strip():
            sq["answer"] = answer_text
            return True
    # 2) 母题已有该题号的答案 → 视为冗余，算"挂上了"（直接删碎片即可）
    for sq in parent.get("sub_questions", []):
        sid = re.search(r"(\d+)", str(sq.get("sub_id", "")))
        if sid and int(sid.group(1)) == qno and (sq.get("answer") or "").strip():
            return True
    # 3) 没有 sub_questions 结构 → 追加进 raw_answer（入库时会用）
    existing = parent.get("raw_answer", "") or ""
    tag = f"{qno}. " if not answer_text.lstrip().startswith(str(qno)) else ""
    parent["raw_answer"] = (existing + "\n" + tag + answer_text).strip()
    return True


def clean(items):
    stats = Counter()
    parent_idx = build_parent_index(items)
    out = []

    for it in items:
        if is_answer_noise(it):
            stats["answer_noise"] += 1
            yr = it.get("year")
            # 题号：优先 source 末尾数字，否则 stem 开头数字
            kind, lo, hi, _ = parse_qno(it.get("source", ""))
            qno = lo if lo is not None else leading_qno(it.get("stem", ""))
            parent = parent_idx.get((yr, qno)) if qno is not None else None
            if parent is not None and parent is not it:
                if attach_answer(parent, qno, it.get("stem", "")):
                    stats["answer_merged"] += 1
                    continue  # 挂回成功 → 丢弃碎片
            # 挂不回：保留碎片但打标，绝不静默丢答案
            it["_orphan_answer"] = True
            stats["answer_orphan_kept"] += 1
            out.append(it)
            continue

        # 真题：规范化 source
        kind, lo, hi, suf = parse_qno(it.get("source", ""))
        old = it.get("source", "")
        new = normalize_source(it)
        if new != old:
            it["source"] = new
            stats[f"normalized_{kind}"] += 1
        if kind == "none" and suf:
            it["_qno_unparsed"] = suf  # 题号没解析出来，留标记供人工看
            stats["qno_unparsed"] += 1
        stats["kept_real"] += 1
        out.append(it)

    return out, stats


def main():
    ap = argparse.ArgumentParser(description="切题产物清洗（入库前兜底）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--report", action="store_true", help="只打印报告，不写文件")
    args = ap.parse_args()

    inp = Path(args.input)
    items = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    before = len(items)

    cleaned, stats = clean(items)

    print(f"\n清洗报告  {inp.name}")
    print(f"  输入: {before} 条")
    print(f"  ├ 真题保留:        {stats['kept_real']}")
    print(f"  ├ 答案碎片识别:    {stats['answer_noise']}")
    print(f"  │   ├ 挂回母题(删): {stats['answer_merged']}")
    print(f"  │   └ 找不到母题(留+打标): {stats['answer_orphan_kept']}")
    norm = {k: v for k, v in stats.items() if k.startswith("normalized_")}
    if norm:
        print(f"  └ source 规范化:   {sum(norm.values())}  " +
              " ".join(f"{k.split('_')[1]}={v}" for k, v in norm.items()))
    if stats["qno_unparsed"]:
        print(f"  ⚠ 题号仍无法解析(打 _qno_unparsed): {stats['qno_unparsed']}")
    print(f"  输出: {len(cleaned)} 条  (净减 {before - len(cleaned)})")

    if args.report:
        return

    out = Path(args.out) if args.out else inp.with_suffix(".clean.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in cleaned:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  已写入: {out}")
    print(f"  入库改用: python3 scripts/process_papers.py --input {out} --topic <topic>")


if __name__ == "__main__":
    main()
