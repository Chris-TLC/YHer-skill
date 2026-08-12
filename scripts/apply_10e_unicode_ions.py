#!/usr/bin/env python3
"""10e-B 文本破损离子 → Unicode 上下标 候选生成 + apply（QA-3）。

审计 P3: 原 10e 走 \\ce{} latex 被否(渲染层无内联支持)。走 B: 直接替换成
Unicode 上下标字符(Fe2+ → Fe²⁺, CO32- → CO₃²⁻),text 节点原样显示,零渲染层改动。

安全设计(全库扫已验的陷阱):
  - 白名单确定式: 只转由真实元素符号构成的已知无机离子; 每条写死"主体+下标+电荷"的
    Unicode 目标。字母代号离子(R/T/A/HR2-)、材料名(T-碳)一律不碰。
  - 电荷符号归一化(＋/−/－ → +/-)后匹配。
  - 边界: 匹配前不可是字母/数字/右括号(防吞相邻); 电荷符号后不可是 =/元素/数字
    (防误伤方程式 Cl2+H2O、NH3+NaCl 里的加号)。系数前缀(2Fe3+ 的 2)保留。
  - 幂等: 目标已是 Unicode 形态则跳过。

默认出候选(dry-run 打印统计)。--apply 需用户点名 L1(题库主库写入)。
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
ITEM_BANK = REPO / "data" / "item_bank" / "v4" / "chemistry_v4_1_3329.jsonl"
CAND_OUT = Path("/tmp/yher_10e_unicode_candidates.jsonl")
APPLY_ID = "batch10_10e_unicode_20260705"

SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵",
       "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻"}
SUB = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅",
       "6": "₆", "7": "₇", "8": "₈", "9": "₉"}

def sup(s: str) -> str:
    return "".join(SUP[c] for c in s)

def sub(s: str) -> str:
    return "".join(SUB[c] for c in s)

# ---- 确定式离子白名单 ----
# 每条: 破损写法(电荷符号已归一为 +/-) → Unicode 目标。
# 单原子/铵: 主体原样 + 电荷上标。多原子含下标: 逐条写死下标位置。
# 只列全库扫到的真实无机离子(≥5 次或化学确定),字母代号一律不入。
def _cation(body, charge):  # body 原样 + 上标电荷(如 Fe + 3+ → Fe³⁺, Na + + → Na⁺)
    return body + (sup(charge) if charge and charge != "1" else "") + SUP["+"]

def _anion(body, charge):
    return body + (sup(charge) if charge and charge != "1" else "") + SUP["-"]

# (正则主体, 目标) —— 主体已含下标转换
WHITELIST = {}
# 单原子阳离子(元素+电荷数+)
for elem, charges in {
    "H": ["1"], "Na": ["1"], "K": ["1"], "Li": ["1"], "Ag": ["1"], "Cu": ["1", "2"],
    "Fe": ["2", "3"], "Al": ["3"], "Ca": ["2"], "Mg": ["2"], "Ba": ["2"], "Zn": ["2"],
    "Mn": ["2"], "Ni": ["2"], "Cd": ["2"], "Pb": ["2"], "Co": ["2"], "Cr": ["3"],
    "Ce": ["3", "4"], "La": ["3"], "Ti": ["4"], "Sn": ["2", "4"], "Hg": ["2"], "Sr": ["2"],
}.items():
    for ch in charges:
        raw = f"{elem}{'' if ch == '1' else ch}+"
        WHITELIST[raw] = elem + (sup(ch) if ch != "1" else "") + SUP["+"]
# 单原子阴离子
for elem, charges in {
    "Cl": ["1"], "Br": ["1"], "I": ["1"], "F": ["1"], "S": ["2"], "O": ["2"], "N": ["3"],
}.items():
    for ch in charges:
        raw = f"{elem}{'' if ch == '1' else ch}-"
        WHITELIST[raw] = elem + (sup(ch) if ch != "1" else "") + SUP["-"]
# 铵/氢氧根/常见多原子阴离子(下标位置逐条写死)
WHITELIST.update({
    "NH4+": "NH" + SUB["4"] + SUP["+"],
    "OH-": "OH" + SUP["-"],
    "ClO-": "ClO" + SUP["-"],
    "CN-": "CN" + SUP["-"],
    "SCN-": "SCN" + SUP["-"],
    "HS-": "HS" + SUP["-"],
    "HCO3-": "HCO" + SUB["3"] + SUP["-"],
    "HSO3-": "HSO" + SUB["3"] + SUP["-"],
    "HSO4-": "HSO" + SUB["4"] + SUP["-"],
    "HCOO-": "HCOO" + SUP["-"],
    "CH3COO-": "CH" + SUB["3"] + "COO" + SUP["-"],
    "NO3-": "NO" + SUB["3"] + SUP["-"],
    "NO2-": "NO" + SUB["2"] + SUP["-"],
    "MnO4-": "MnO" + SUB["4"] + SUP["-"],
    "AlO2-": "AlO" + SUB["2"] + SUP["-"],
    "HSO3-": "HSO" + SUB["3"] + SUP["-"],
    "CO32-": "CO" + SUB["3"] + SUP["2"] + SUP["-"],
    "SO42-": "SO" + SUB["4"] + SUP["2"] + SUP["-"],
    "SO32-": "SO" + SUB["3"] + SUP["2"] + SUP["-"],
    "SiO32-": "SiO" + SUB["3"] + SUP["2"] + SUP["-"],
    "S2O32-": "S" + SUB["2"] + "O" + SUB["3"] + SUP["2"] + SUP["-"],
    "Cr2O72-": "Cr" + SUB["2"] + "O" + SUB["7"] + SUP["2"] + SUP["-"],
    "CrO42-": "CrO" + SUB["4"] + SUP["2"] + SUP["-"],
    "PO43-": "PO" + SUB["4"] + SUP["3"] + SUP["-"],
    "HPO42-": "HPO" + SUB["4"] + SUP["2"] + SUP["-"],
    "H2PO4-": "H" + SUB["2"] + "PO" + SUB["4"] + SUP["-"],
    "S2-": "S" + SUP["2"] + SUP["-"],
    "O2-": "O" + SUP["2"] + SUP["-"],
    "N3-": "N" + SUP["3"] + SUP["-"],
})

# 归一化电荷符号
CHARGE_NORM = str.maketrans({"＋": "+", "−": "-", "－": "-", "—": "-"})

# 按 raw 长度降序(先匹配长的,防 SO42- 被 S2- 抢)
WL_SORTED = sorted(WHITELIST.items(), key=lambda kv: -len(kv[0]))

def replace_ions(text: str):
    """返回 (新文本, 替换列表[(raw, target)])。仅边界安全处替换。"""
    norm = text.translate(CHARGE_NORM)
    changes = []
    result = []
    i = 0
    n = len(norm)
    while i < n:
        matched = False
        for raw, target in WL_SORTED:
            if norm.startswith(raw, i):
                # 左边界: 前一字符不可是拉丁字母/右括号(防吞元素符号如 [FeCl4]、防接续式);
                # ASCII 数字放行(系数,如 2Fe3+→2Fe³⁺);中文、标点、空格放行
                # (isalpha 对中文为 True,故只判 ASCII 字母)。
                left_ok = True
                if i > 0:
                    p = norm[i - 1]
                    if (p.isascii() and p.isalpha()) or p == ")":
                        left_ok = False
                # 右边界: raw 末尾是 +/-,其后不可是 =/拉丁字母/ASCII数字
                # (防方程式 Cl2+H2O 的加号、防吞相邻元素/数字);中文、标点、状态符 ( 放行。
                j = i + len(raw)
                right_ok = True
                if j < n:
                    q = norm[j]
                    if q == "=" or (q.isascii() and (q.isalpha() or q.isdigit())):
                        right_ok = False
                if left_ok and right_ok:
                    result.append(target)
                    changes.append((raw, target))
                    i = j
                    matched = True
                    break
        if not matched:
            result.append(norm[i])
            i += 1
    return "".join(result), changes

def iter_text_nodes(item):
    """yield (field, block_idx, para_idx, node) for text nodes."""
    for field in ("stem_blocks", "answer_blocks_effective", "analysis_blocks"):
        blocks = item.get(field) or []
        for bi, block in enumerate(blocks):
            para = block.get("para") if isinstance(block, dict) else None
            if not para:
                continue
            for pi, node in enumerate(para):
                if isinstance(node, dict) and node.get("type") == "text" and node.get("text"):
                    yield field, bi, pi, node

def read_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def write_jsonl(p, rows):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)

def run(apply: bool):
    import sys
    sys.path.insert(0, str(REPO))
    from core.data.item_bank_v4 import iter_service_items
    svc_ids = {it["item_id"] for it in iter_service_items()}

    rows = read_jsonl(ITEM_BANK)
    stats = Counter()
    ion_freq = Counter()
    candidates = []
    touched_items = set()
    for item in rows:
        if item["item_id"] not in svc_ids:
            continue
        for field, bi, pi, node in iter_text_nodes(item):
            new_text, changes = replace_ions(node["text"])
            if not changes:
                continue
            stats["nodes_changed"] += 1
            for raw, tgt in changes:
                ion_freq[raw] += 1
            candidates.append({
                "item_id": item["item_id"], "field": field,
                "block_idx": bi, "para_idx": pi,
                "original": node["text"], "replaced": new_text,
                "ions": [r for r, _ in changes],
                "review_status": "pending_user_or_claude", "reviewer": "",
                "schema_version": "qa1_candidate_v1", "candidate_kind": "text_ion_unicode",
            })
            touched_items.add(item["item_id"])
            if apply:
                node["text"] = new_text
    stats["items_touched"] = len(touched_items)
    stats["total_ion_replacements"] = sum(ion_freq.values())
    if apply:
        for item in rows:
            if item["item_id"] in touched_items:
                item["batch10_source_10e"] = APPLY_ID
        write_jsonl(ITEM_BANK, rows)
    else:
        write_jsonl(CAND_OUT, candidates)
    return stats, ion_freq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print(f"=== 10e-B Unicode 离子 {'APPLY' if args.apply else 'DRY-RUN(候选)'} ===")
    stats, ion_freq = run(args.apply)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print("  Top 离子替换:")
    for ion, c in ion_freq.most_common(15):
        print(f"    {ion}: {c}")
    if not args.apply:
        print(f"  候选落: {CAND_OUT}")
    print("=== 完成 ===")

if __name__ == "__main__":
    main()
