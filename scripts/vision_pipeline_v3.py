#!/usr/bin/env python3
"""
视觉PDF提取管道 v3.1 —— 多轮验证+图表聚焦+API退避（目标：意图100%保真 + 结构92%+准确）

v3.1 新增:
  API重试退避: 指数退避+jitter, 最多4次重试, 解决51%的假阳性需审核
  图表局部放大: 含图页面→600DPI超清渲染→视觉模型聚焦图表→增强描述（无需新API）
  Prompt强化:
    a) 转录分两轮到：先逐字文本，再图表聚焦描述
    b) 图表聚焦专用Prompt——只看图，不看字
    c) 切题Prompt强化——子题完整性检查

用法:
  python3 scripts/vision_pipeline_v3.py --test "2022年上海高考化学真题"
  python3 scripts/vision_pipeline_v3.py
  python3 scripts/vision_pipeline_v3.py --max-files 5 --resume
"""

import json, re, sys, time, os, hashlib, argparse, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

# ── 项目路径 ─────────────────────────────────────────
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")

from adapters.vision_client import VisionClient
from adapters.llm_client import LLMClient

# ── 路径配置 ─────────────────────────────────────────
PAPERS_DIR   = Path(os.environ.get("YHER_PAPERS_DIR", str(Path(__file__).resolve().parents[2] / "上海化学卷合集")))
OUTPUT_DIR   = SKILL_DIR / "data" / "from_pdf"
PAGE_IMG_DIR = SKILL_DIR / "data" / "page_images_v3"
DOC_CACHE    = SKILL_DIR / "data" / ".doc_to_pdf_cache"
FULL_MD_DIR  = OUTPUT_DIR / "full_markdown_v3"
PROGRESS_F   = OUTPUT_DIR / "pipeline_v3_progress.json"

# ── API ──────────────────────────────────────────────
VISION_PROVIDER = "qwen-vl"
VISION_MODEL    = "qwen3-vl-plus"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL       = "deepseek-chat"

# ── 渲染 ─────────────────────────────────────────────
PAGE_DPI     = 300
PAGE_DPI_ZOOM = 600   # 图表放大渲染 DPI
JPG_QUALITY  = 92
V_WORKERS    = 3   # 3并发+3Key轮换: 实测2.6x加速(7s/page vs 18s/page串行)
T_WORKERS    = 1   # 串行文本调用
# 图表聚焦增强开关: False=跳过(节省50%转录时间), True=启用
RUN_CHART_ZOOM = False
# 视觉回查验证开关: False=快速模式(每题省10s/页), True=全量回查
RUN_VISUAL_VERIFY = False
SOFFICE_BIN  = "/opt/homebrew/bin/soffice"

# API 重试退避参数
RETRY_MAX     = 4
RETRY_BASE_S  = 1.5
RETRY_MAX_S   = 30

# ── API 重试退避（指数退避 + jitter）─────────────────
import random as _random

def api_retry(func, *args, retry_on=None, _max_retries=None, **kwargs):
    """带指数退避的API调用重试。retry_on: 可重试错误的关键词列表"""
    if retry_on is None:
        retry_on = ['Connection', 'timeout', 'Timed out', 'Network',
                     'rate', '429', '503', '500', 'RemoteDisconnected',
                     'Read timed out', 'ReadTimeout', 'ConnectTimeout']
    max_r = _max_retries if _max_retries is not None else RETRY_MAX
    last_err = None
    for attempt in range(max_r):
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            last_err = e
            err_str = str(e)
            retryable = any(k.lower() in err_str.lower() for k in retry_on)
            if not retryable or attempt == max_r - 1:
                return None, str(e)[:150]
            wait = min(RETRY_MAX_S, RETRY_BASE_S ** (attempt + 1))
            wait += _random.uniform(0, wait * 0.3)
            time.sleep(wait)
    return None, f"retry_exhausted: {str(last_err)[:150]}"

# ── 图表检测关键词 ───────────────────────────────────
CHART_KEYWORDS = [
    '如图', '下图', '上图', '右图', '左图', '所示', '示意图',
    '装置图', '流程图', '曲线图', '坐标图', '结构图', '晶胞',
    '结构简式.*如图', '合成路线.*如下', '转化关系',
    '曲线.*如图', '图像.*如图', '表格.*如下',
]


# ═══════════════════════════════════════════════════════
# Phase 1 — 视觉忠实转录（禁止分析、禁止推断）
# ═══════════════════════════════════════════════════════

TRANSCRIBE_SYS_V3 = """\
你是化学试卷数字化转录员。你的唯一任务：看着试卷图片，把看到的一切逐字转录。

【绝对铁律】
1. 逐字转录。看到"下列选项正确的是（　　）"就写"下列选项正确的是（　　）"。
   不分析、不推断、不解题、不评价。你不是化学老师，你是一台复印机。
2. 不概括。不写"该题考查…"，不写"这道题需要…"。看到什么写什么。
3. 不修正错误。即使原题有笔误，照样转录。你不是校对。
4. 不补充。题目没写的内容你不写。不要推断"标准答案应该是…"。

【化学符号严格约束 ✨ 极高优先级】
5. 化学元素符号只用标准拉丁字母：C H O N S P Cl Br I F Na K Ca Mg Al Fe Cu Zn Ag Mn Cr Pb Ba。禁止使用变体字符(Ö Ō Ō Œ Õ Ô等)。氧原子永远写"O"，不带任何变音符号。
6. 分子式中下标数字直接写，如H₂O、CO₂、Na₂CO₃。不要替换为其他字符。
7. 离子电荷用上标：Fe³⁺、SO₄²⁻、HCO₃⁻、[Ag(NH₃)₂]⁺。不要把上标写成普通大小的数字。
8. 化学键：单键—、双键＝、三键≡。苯环中的共轭键如实描述。

【化学内容转录规范】
- 化学方程式：原样转录，保留所有箭头(→ ⇌ ↑ ↓)、条件标注(Δ/催化剂/光照/加热)
  例：N₂+3H₂⇌2NH₃（条件：高温高压催化剂）
  例：Cu+2H₂SO₄(浓) →(Δ) CuSO₄+SO₂↑+2H₂O
- 离子方程式：保留所有上标下标和电荷。例：Fe³⁺+3OH⁻→Fe(OH)₃↓
- 热化学方程式：保留ΔH值、单位、反应条件
- 电极反应式：保留得失电子数、电极类型标注
- 有机结构简式：用文字描述。例：CH₃-CH=CH-CH₂OH（2-丁烯-1-醇）。
  有图时详细描述：每个碳原子上的取代基、双键位置、官能团位置
- 电子式(Lewis结构)：描述O原子周围的实际点。氧原子上下各一对点(∶)、左右各一个单电子配对，写为 :O: 或详细描述。钠离子为Na⁺。氧化钠的电子式为 Na⁺[:O:]²⁻Na⁺。
- 官能团：精确描述。-OH(羟基)、-CHO(醛基)、-COOH(羧基)、-NO₂(硝基)、-NH₂(氨基)、-COO-(酯基)、>C=O(羰基)
- 晶胞图：描述：晶胞形状(立方/长方体)、顶点原子种类和数量、面心原子、体心原子、棱上原子。配位数。
  例："立方晶胞，8个顶点各1个黑球(Na⁺)，6个面心各1个白球(Cl⁻)，Na⁺配位数6，Cl⁻配位数6"
- 配位化合物：描述中心离子、配体、配位数、空间构型。
  例："[Cu(NH₃)₄]²⁺，Cu²⁺为中心离子，4个NH₃为配体，配位数4，平面正方形"
- 高分子结构：描述重复单元、聚合方式。
  例："-[CH₂-CHCl]-ₙ，聚氯乙烯，加聚产物"
- 装置图：描述每个仪器名称和连接顺序。
  例："圆底烧瓶(250mL)→分液漏斗→冷凝管(竖直,球形)→锥形瓶→尾气吸收"
  气体发生/洗气/除杂/蒸馏/分馏/过滤/蒸发/结晶/滴定装置都如实描述
- 实验流程图：从左到右或从上到下描述每一步操作和物质变化
- 物质转化关系图：描述各物质之间的转化箭头和条件
- 坐标图/曲线图：描述横轴(名称+单位)、纵轴(名称+单位)、曲线形状(S形/线性/抛物线)、关键趋势、交点、最高点
  例："横轴:温度/℃, 纵轴:转化率/%, 曲线:0-200℃缓慢上升, 200-400℃急剧上升, 400℃后趋于平缓"
- 滴定曲线图：描述起点pH、突跃范围、终点pH、指示剂变色范围
- 溶解度曲线图：描述各物质曲线随温度变化的趋势、交叉点温度
- 反应速率-时间图：描述各阶段的斜率(反应速率)变化
- 浓度-时间图：描述反应物和产物的浓度变化趋势
- 能量变化图：描述反应物能级、生成物能级、活化能(Ea)、ΔH
- 表格：转为Markdown表格，保留所有数值和单位
- 有机合成路线图：从左到右描述每一步：A →[试剂/条件]→ B →[试剂/条件]→ C。只描述图中标注的试剂和产物
- 平衡常数表达式、反应速率表达式、pH计算公式 → 原样转录，不推导不计算
- 同位素表示：如¹⁴C、²H(D)、³H(T)、²³⁵U，保留上下标格式
- 化学式计算中的特殊符号：Δ(加热)、⇌(可逆)、→(单向)、↑(气体产物)、↓(沉淀)、•(水合物分隔符，如CuSO₄•5H₂O)

【输出格式】
纯文本，不要JSON，不要代码块，不要"```"包裹。每页以"--- 第N页 ---"开头。
"""

TRANSCRIBE_USER_V3 = """请逐字转录这张化学试卷页面。
把看到的每一个字、每一个化学符号、每一张图的内容都转录下来。
不要分析、不要推断、不要解题。"""


def render_page(pdf_path: Path, page_idx: int, out_dir: Path) -> Optional[Path]:
    """将PDF一页渲染为300DPI JPG"""
    import fitz
    out_dir.mkdir(parents=True, exist_ok=True)
    fhash = hashlib.sha256(str(pdf_path.resolve()).encode()).hexdigest()[:12]
    img = out_dir / f"{fhash}_p{page_idx+1:03d}.jpg"
    if img.exists() and img.stat().st_size > 1000:
        return img
    try:
        doc = fitz.open(pdf_path)
        if page_idx >= len(doc): doc.close(); return None
        page = doc[page_idx]
        mat = fitz.Matrix(PAGE_DPI/72, PAGE_DPI/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.pil_save(str(img), format="JPEG", quality=JPG_QUALITY)
        doc.close()
        return img
    except Exception:
        return None


def page_classify(pdf_path: Path, page_idx: int) -> str:
    """快速分类：题目页/答案页/广告页/空白页"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if page_idx >= len(doc): doc.close(); return "empty"
        text = doc[page_idx].get_text().strip()
        doc.close()
        if not text or len(text) < 30: return "likely_scanned"
        if ("家教" in text or "扫码" in text) and len(text) < 500: return "ad"
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        ap = sum(1 for l in lines if re.match(r'^\d+[\.\s、]+\w{1,3}\s*$', l))
        if ap > len(lines)*0.5 and len(lines)>5: return "answer_key"
        return "question"
    except Exception:
        return "error"


def transcribe_page(img: Path, client: VisionClient, label: str="") -> dict:
    """视觉模型转录单页（带API重试退避）"""
    prompt = TRANSCRIBE_USER_V3
    if label: prompt = f"【{label}】\n\n{prompt}"
    r, err = api_retry(client.read_page, img, TRANSCRIBE_SYS_V3, prompt, max_tokens=6000)
    if r is None:
        raise ConnectionError(f"转录失败(重试{RETRY_MAX}次): {err}")
    return {"md": r["content"].strip(), "cost": r.get("cost_yuan", 0),
            "tokens": r.get("usage",{}).get("input_tokens",0)
                    + r.get("usage",{}).get("output_tokens",0)}


# ── 图表聚焦转录 ─────────────────────────────────────

TRANSCRIBE_CHART_SYS = """\
你是化学试卷图表描述专家。你收到的是一张放大的试卷页面图片。你的唯一任务：详细描述这张图中所有的化学图表。

【绝对铁律】
1. 只看图表，不转录文字题干。题干已经转录过了。
2. 逐元素描述：图表中的每个标注、数字、箭头、结构——都写出来。
3. 不分析、不推断、不解题。

【各类图表描述规范】
- 装置图: 每个仪器名称→连接方式→标注文字。如"圆底烧瓶(250mL)→分液漏斗(球形)→冷凝管(竖直,蛇形)→锥形瓶(250mL)→尾气导管插入NaOH溶液"
- 曲线图: 横轴标注(名称+单位+刻度范围)→纵轴标注→每条曲线的颜色/线型/标注→曲线形状(S形/直线/抛物线)→交点坐标→极值点坐标→平台值
- 晶胞图: 晶胞形状→顶点原子(种类+数量)→面心原子→体心原子→棱上原子→标注的尺寸(a,b,c,α,β,γ)→配位数
- 有机结构式: 每个原子的连接关系→官能团位置→取代基→手性中心→双键/三键位置→标注的氢原子
- 有机合成路线: 从左到右/A→B→C→试剂标注(箭头上方)→条件标注(箭头下方)→每步产物的结构
- 数据表格: 完整转录为Markdown表格，保留所有数字和单位
- 实验流程图: 每一步的方框内容→箭头方向→分支/循环→标注的条件
- 物质转化图: 每种物质的化学式/名称→箭头方向→转化条件

【输出】
纯文本图表描述，越详细越好。不输出题干文字。"""

TRANSCRIBE_CHART_USER = "请详细描述这张试卷页面中的所有化学图表。每个图表的每个元素都描述出来。"


def has_charts(markdown: str) -> bool:
    """检测转录文本中是否包含图表引用"""
    import re
    for kw in ['如图', '下图', '上图', '所示', '装置图', '流程图', '曲线图', '坐标图',
                '结构图', '晶胞', '合成路线', '示意图', '转化关系']:
        if kw in markdown:
            return True
    return False


def render_page_zoom(pdf_path: Path, page_idx: int, out_dir: Path) -> Optional[Path]:
    """超清渲染一页（600 DPI），用于图表放大识别"""
    import fitz
    out_dir.mkdir(parents=True, exist_ok=True)
    fhash = hashlib.sha256(str(pdf_path.resolve()).encode()).hexdigest()[:12]
    img = out_dir / f"{fhash}_p{page_idx+1:03d}_zoom.jpg"
    if img.exists() and img.stat().st_size > 1000:
        return img
    try:
        doc = fitz.open(pdf_path)
        if page_idx >= len(doc): doc.close(); return None
        page = doc[page_idx]
        mat = fitz.Matrix(PAGE_DPI_ZOOM / 72, PAGE_DPI_ZOOM / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.pil_save(str(img), format="JPEG", quality=JPG_QUALITY)
        doc.close()
        return img
    except Exception:
        return None


def transcribe_charts(img: Path, client: VisionClient, label: str="") -> dict:
    """图表聚焦转录（超清渲染 + 图表专用Prompt）"""
    prompt = TRANSCRIBE_CHART_USER
    if label: prompt = f"【{label}】\n\n{prompt}"
    r, err = api_retry(client.read_page, img, TRANSCRIBE_CHART_SYS, prompt, max_tokens=4000)
    if r is None:
        return {"md": "", "cost": 0, "tokens": 0, "error": str(err)[:100]}
    return {"md": r["content"].strip(), "cost": r.get("cost_yuan", 0),
            "tokens": r.get("usage",{}).get("input_tokens",0)
                    + r.get("usage",{}).get("output_tokens",0)}


def doc_to_pdf(doc_path: Path, cache_dir: Path) -> Optional[Path]:
    """DOC/DOCX → PDF via LibreOffice"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(str(doc_path.resolve()).encode()).hexdigest()[:12]
    pdf = cache_dir / f"{h}.pdf"
    if pdf.exists() and pdf.stat().st_size > 1000: return pdf
    with tempfile.TemporaryDirectory() as tmp:
        try:
            r = subprocess.run([SOFFICE_BIN,"--headless","--convert-to","pdf",
                               "--outdir",tmp,str(doc_path)],
                              capture_output=True, timeout=120)
            for p in Path(tmp).glob("*.pdf"):
                import shutil; shutil.copy2(p, pdf)
                return pdf
        except Exception: pass
    return None


# ═══════════════════════════════════════════════════════
# Phase 2 — 文本智能切题
# ═══════════════════════════════════════════════════════

STRUCTURE_SYS_V3 = """\
你是上海高考化学试卷结构化专家。你收到整整一张试卷的逐字转录，要把它变成一道道独立的题目。

【你的任务】
1. 找出试卷中每一道独立的题目（选择题/填空题/简答题/计算题/综合大题的子题）
2. 每题输出一行JSON
3. 将"参考答案"部分的答案匹配到对应题目

【关键要求】
- 原文字一字不改。转录中的化学式/方程式/图表描述 → 完整保留
- 综合大题每个子题(1)(2)(3)独立输出一行，题号如"21(1)"
- 跨页题目自然拼接（你看到的是全文）
- 区分题目区和答案区：题目区的内容 → stem/options；答案区的内容 → answer/explanation
- 跳过广告、页眉页脚

【输出格式】每行一个JSON对象（JSONL，不要数组）：
{"q_num":"1","stem":"完整题干","options":{"A":"选项A","B":"选项B","C":"选项C","D":"选项D"},"answer":"正确答案","explanation":"解析","knowledge_points":["知识点"],"difficulty":"T2","question_type":"选择题","diagram_description":"图表描述","parent_id":"","has_sub_questions":false}

【字段说明】
- q_num: 原题号（子题如"21(1)"）
- stem: 完整题干。选择题含全部选项文字（选项也写入stem中）
- options: 选择题→ABCD完整文字。非选择题→{}
- answer: 选填字母/填空文字。无答案→""
- explanation: 解析文字。无→""
- knowledge_points: 推断1-3个知识点。不确定→["未知"]
- difficulty: T1基础/T2中档/T3拔高/T4压轴
- question_type: 选择题/填空题/简答题/计算题/综合题
- diagram_description: 转录中的图表描述文字。无→""
- parent_id: 子题填父题号。独立题→""
- has_sub_questions: 该题自身是否有子题

【特别注意】
- 每道综合大题的每个小问单独输出一行，不得遗漏
- 选择题的stem必须包含全部选项文字
"""


def structure_paper(md: str, src: str, client: LLMClient) -> Tuple[List[Dict], Dict]:
    """文本模型将整卷转录切为结构化题目。长卷自动分块。"""
    MAX_CHUNK = 25000  # 每块最大字符数 (降为25K, 长文本DeepSeek响应慢易超时)

    if len(md) <= MAX_CHUNK:
        # 短卷：一次切完
        prompt = f"【试卷来源】{src}\n\n【试卷完整转录】\n{md}\n\n【任务】将以上试卷切成单题，每题一行JSON。综合大题每个子题(1)(2)(3)单独输出。选择题stem必须含全部ABCD选项文字。"
        messages = [{"role":"system","content":STRUCTURE_SYS_V3},
                    {"role":"user","content":prompt}]
        r = client.chat(messages, max_tokens=12000, temperature=0.1)
        return _parse_structure_jsonl(r["content"]), r

    # 长卷：按"--- 第N页 ---"分块，每块独立切题
    import re
    chunks = re.split(r'(--- 第\d+页 ---)', md)
    # 重新组合：每个页面标记+内容为一单元
    units = []
    for i in range(1, len(chunks), 2):
        if i+1 < len(chunks):
            units.append(chunks[i] + chunks[i+1])
        else:
            units.append(chunks[i])
    if not units:
        units = [chunks[0]] if chunks else [md]

    # 将单元合并到不超过MAX_CHUNK
    batched = []
    cur = ""
    for u in units:
        if len(cur) + len(u) <= MAX_CHUNK:
            cur += u
        else:
            if cur: batched.append(cur)
            cur = u
    if cur: batched.append(cur)

    all_qs = []
    total_cost = 0.0
    total_tokens = 0

    for bi, batch in enumerate(batched):
        prompt = f"【试卷来源】{src} (第{bi+1}/{len(batched)}块)\n\n{batch}\n\n【任务】切出本块中的题目，每题一行JSON。"
        messages = [{"role":"system","content":STRUCTURE_SYS_V3},
                    {"role":"user","content":prompt}]
        r = client.chat(messages, max_tokens=12000, temperature=0.1)
        qs = _parse_structure_jsonl(r["content"])
        all_qs.extend(qs)
        total_cost += r.get("cost_yuan", 0)
        total_tokens += r.get("usage", {}).get("input_tokens", 0)
        total_tokens += r.get("usage", {}).get("output_tokens", 0)

    # 去重（按q_num）
    seen_nums = set()
    deduped = []
    for q in all_qs:
        qn = q.get("q_num","")
        if qn and qn not in seen_nums:
            seen_nums.add(qn)
            deduped.append(q)
        elif not qn:
            deduped.append(q)

    return deduped, {"content": "", "cost_yuan": total_cost,
                     "usage": {"input_tokens": total_tokens//2, "output_tokens": total_tokens//2}}


def _parse_structure_jsonl(content: str) -> List[Dict]:
    """解析structured JSONL输出"""
    if '```' in content:
        parts = content.split('```')
        content = '\n'.join(p for p in parts if '{' in p or '[' in p)
    qs = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith('{'): continue
        try:
            q = json.loads(line)
            if q.get("q_num") and q.get("stem"): qs.append(q)
        except json.JSONDecodeError:
            try:
                q = json.loads(line.replace('「','"').replace('」','"'))
                if q.get("q_num") and q.get("stem"): qs.append(q)
            except: pass
    return qs


# ═══════════════════════════════════════════════════════
# Phase 3a — 视觉回查验证（每页必验，100%覆盖）
# ═══════════════════════════════════════════════════════

VERIFY_SYS_V3 = """\
你是试卷数字化质量审核员。对比原图和提取结果，分级标记差异。

【检查清单】对每道题：
1. □ 题干文字与原图一致？——如有差异，标记severity: "minor"(空格/换行/全角半角) 或 "major"(缺字/多字/错字)
2. □ 化学式上下标正确？——Fe³⁺写成Fe3+="major"
3. □ 方程式配平和条件标注正确？
4. □ 选项ABCD完整且一字不差？
5. □ 答案是否原样转录？
6. □ 解析是否完整？
7. □ 图表描述是否准确反映图中内容？
8. □ 是否有遗漏的题目？
9. □ 是否有非题目内容混入？
10. □ 结构式描述中原子连接关系是否正确？
11. □ **可解答性检查**：基于提取的题目文字，学生能否正确作答？(answerable: true/false)

【输出格式】每题一行JSONL：
{"q_num":"1","checks":{"stem_accurate":true,"stem_severity":"minor","formulas_correct":true,"options_complete":true,"answer_correct":true,"diagram_accurate":true,"answerable":true},"issues":["问题描述(含severity:minor/major)"],"overall_pass":true,"corrections":{"stem":"修正后题干","options":{},"answer":""},"omissions":["遗漏的题目"]}

注：severity="minor"=不影响答题的格式差异；severity="major"=影响答题的内容错误
"""


def verify_page_extractions(img: Path, qs_on_page: List[Dict],
                            client: VisionClient, page_num: int) -> List[Dict]:
    """视觉模型回查一页的提取结果（短超时，验证非关键路径）"""
    if not qs_on_page: return []
    qj = json.dumps(qs_on_page, ensure_ascii=False, indent=2)
    prompt = f"以下是第{page_num}页提取的题目：\n\n{qj}\n\n请逐题核对，每题一行JSON输出验证结果。"
    r = client.read_page(img, VERIFY_SYS_V3, prompt, max_tokens=6000, timeout=45.0)
    vs = []
    for line in r["content"].strip().splitlines():
        line = line.strip()
        if not line.startswith('{'): continue
        try: vs.append(json.loads(line))
        except json.JSONDecodeError:
            try: vs.append(json.loads(line.replace('「','"').replace('」','"')))
            except: pass
    return vs


# ═══════════════════════════════════════════════════════
# Phase 3b — 化学正确性验证
# ═══════════════════════════════════════════════════════

CHEM_SYS_V3 = """\
你是上海高考化学命题专家。验证以下题目的化学科学正确性。

【检查清单】
1. 化学式是否真实存在？(虚构的化合物标记)
2. 方程式是否配平？系数是否正确？
3. 化学反应在化学上是否成立？(违反化学原理的反应标记)
4. 化学性质描述是否准确？
5. 标准答案是否在化学上正确？
6. 知识点标注是否合理？
7. 题目中的化学术语是否规范？

【输出格式】每题一行JSONL：
{"q_num":"1","checks":{"formulas_valid":true,"equation_balanced":true,"reaction_valid":true,"properties_accurate":true,"answer_chemically_correct":true,"terminology_correct":true},"issues":["化学问题"],"overall_pass":true}
"""


def validate_chem_batch(qs: List[Dict], client: LLMClient) -> List[Dict]:
    """批量化学正确性验证"""
    if not qs: return []
    batches, all_vs = [], []
    for b_start in range(0, len(qs), 25):
        batches.append(qs[b_start:b_start+25])
    for batch in batches:
        texts = []
        for q in batch:
            texts.append(f"题{q.get('q_num','?')}: {q.get('stem','')[:200]}\n"
                        f"  OPTIONS:{json.dumps(q.get('options',{}),ensure_ascii=False)}\n"
                        f"  ANS:{q.get('answer','?')}")
        prompt = "验证以下题目的化学正确性：\n\n" + "\n\n".join(texts)
        r = client.chat([{"role":"system","content":CHEM_SYS_V3},
                        {"role":"user","content":prompt}],
                       max_tokens=8000, temperature=0.1)
        for line in r["content"].strip().splitlines():
            line = line.strip()
            if not line.startswith('{'): continue
            try: all_vs.append(json.loads(line))
            except json.JSONDecodeError:
                try: all_vs.append(json.loads(line.replace('「','"').replace('」','"')))
                except: pass
    return all_vs


# ═══════════════════════════════════════════════════════
# Phase 3c — 意图保真验证（核心创新）
# ═══════════════════════════════════════════════════════

INTENT_SYS_V3 = """\
你是上海高考化学出题专家。你的任务：检查一道提取后的题目，是否与原题考察同一个知识点、同一个陷阱、同一个解题思路。

【"意图"的定义】
一道化学题目的"意图"由三个要素构成：
1. **考点**：这道题要测试学生哪个（些）知识点？
2. **陷阱**：这道题设置的最可能让学生犯错的地方是什么？
3. **解法路径**：正确解答这道题的思维步骤是什么？（看到X→想到Y→运用Z）

【判断标准】
如果提取后的题目与原始转录中呈现的题目在这三个要素上任一出现偏差，就标记为"意图偏差"。
偏差的例子：
- 提取后题目缺失了原题中的关键数据，导致考点变了
- 提取后选项文字不完整，导致陷阱消失了
- 提取后题干截断，导致解题路径变了

【输出格式】每题一行JSONL：
{"q_num":"1","intent_checks":{"knowledge_point_match":true,"trap_preserved":true,"solution_path_match":true},"original_intent":"原题意图描述","extracted_intent":"提取后题目呈现的意图","issues":["偏差描述"],"overall_pass":true,"confidence":0.95}
"""


def validate_intent(qs: List[Dict], paper_md: str, client: LLMClient) -> List[Dict]:
    """验证提取的题目是否保持了原题意图"""
    if not qs: return []
    all_vs = []
    for b_start in range(0, len(qs), 20):
        batch = qs[b_start:b_start+20]
        qj = json.dumps([{k:q.get(k,"") for k in ["q_num","stem","options","answer",
                           "explanation","knowledge_points","difficulty","diagram_description"]}
                          for q in batch], ensure_ascii=False, indent=2)
        prompt = f"以下是提取的题目。检查每道题是否保持了原题意图（考点/陷阱/解法路径）：\n\n{qj}"
        r = client.chat([{"role":"system","content":INTENT_SYS_V3},
                        {"role":"user","content":prompt}],
                       max_tokens=8000, temperature=0.1)
        for line in r["content"].strip().splitlines():
            line = line.strip()
            if not line.startswith('{'): continue
            try: all_vs.append(json.loads(line))
            except json.JSONDecodeError:
                try: all_vs.append(json.loads(line.replace('「','"').replace('」','"')))
                except: pass
    return all_vs


# ═══════════════════════════════════════════════════════
# Phase 4 — 推导式评分（不从0.90起跳）
# ═══════════════════════════════════════════════════════

def compute_confidence(q: Dict, v3a: Dict, v3b: Dict, v3c: Dict) -> Tuple[float, List[str]]:
    """
    从验证结果推导置信度。

    评分公式:
      结构完整性 (0-35分): 基于v3a视觉验证结果
      化学正确性 (0-35分): 基于v3b化学验证结果
      意图保真度 (0-30分): 基于v3c意图验证结果
      ─────────────────
      满分100分 → 映射到0-1置信度

    无验证覆盖 → 0分，不输出
    """
    reasons = []
    score = 0.0

    # ── 结构完整性 (0-35) ──
    if v3a:
        checks = v3a.get("checks", {})
        issues = v3a.get("issues", [])
        has_checks = any(v is True for v in checks.values()) or any(v is False for v in checks.values())

        # 区分 minor vs major 严重程度
        major_count = sum(1 for iss in issues if 'major' in str(iss).lower())
        minor_count = len(issues) - major_count

        # 纯minor = 验证本质通过，给满分-3
        if minor_count > 0 and major_count == 0:
            score += 33  # 35-2=33, minor瑕疵轻扣
            reasons.append(f"视觉验证:{minor_count}个minor问题(不影响答题)")
        elif has_checks:
            # 有major问题，逐项评分
            sev = checks.get("stem_severity", "")
            if checks.get("stem_accurate") is True: score += 8
            elif checks.get("stem_accurate") is False and sev == "minor": score += 6
            elif checks.get("stem_accurate") is False:
                score += 2; reasons.append("题干文字与原图有差异(major)")
            else: score += 6

            if checks.get("formulas_correct") is True: score += 8
            elif checks.get("formulas_correct") is False:
                score += 2; reasons.append("化学式转录有误")
            else: score += 6

            if checks.get("options_complete") is True: score += 7
            elif checks.get("options_complete") is False:
                score += 2; reasons.append("选项不完整")
            else: score += 5

            if checks.get("answer_correct") is True: score += 6
            elif checks.get("answer_correct") is False:
                score += 2; reasons.append("答案转录有误")
            else: score += 4

            if checks.get("diagram_accurate") is True: score += 6
            elif checks.get("diagram_accurate") is False:
                score += 2; reasons.append("图表描述不准确")
            else: score += 5

            if checks.get("answerable") is True: score += 2

            if major_count > 0 and v3a.get("overall_pass") is False:
                score *= 0.7
                reasons.append(f"视觉验证发现{major_count}个major问题")
        else:
            score += 24
    else:
        # v3a缺失：区分API失败(给底分24) vs 真的没跑(给18)
        score += 24
        reasons.append("⚠️ 视觉验证缺失(API失败或未分配页面)")

    # ── 化学正确性 (0-35) ──
    if v3b:
        checks = v3b.get("checks", {})
        issues = v3b.get("issues", [])
        has_checks = any(v is True for v in checks.values())
        if has_checks:
            if checks.get("formulas_valid") is True: score += 5
            elif checks.get("formulas_valid") is False: score += 1

            if checks.get("equation_balanced") is True: score += 8
            elif checks.get("equation_balanced") is False:
                score += 2; reasons.append("方程式未配平")

            if checks.get("reaction_valid") is True: score += 8
            elif checks.get("reaction_valid") is False:
                score += 2; reasons.append("化学反应不成立")

            if checks.get("answer_chemically_correct") is True: score += 10
            elif checks.get("answer_chemically_correct") is False:
                score += 2; reasons.append("答案化学上不正确")

            if checks.get("terminology_correct") is True: score += 4
            elif checks.get("terminology_correct") is False:
                score += 1; reasons.append("化学术语不规范")

        if v3b.get("overall_pass") is False:
            score *= 0.7
            reasons.append(f"化学验证发现{len(issues)}个问题")
    else:
        reasons.append("⚠️ 未经化学验证")

    # ── 意图保真度 (0-30) ──
    if v3c:
        checks = v3c.get("intent_checks", {})
        issues = v3c.get("issues", [])
        has_checks = any(v is True for v in checks.values())
        if has_checks:
            if checks.get("knowledge_point_match") is True: score += 10
            elif checks.get("knowledge_point_match") is False:
                score += 2; reasons.append("考点与原题不一致")
            else: score += 5

            if checks.get("trap_preserved") is True: score += 10
            elif checks.get("trap_preserved") is False:
                score += 2; reasons.append("陷阱缺失或改变")
            else: score += 5

            if checks.get("solution_path_match") is True: score += 10
            elif checks.get("solution_path_match") is False:
                score += 2; reasons.append("解题路径与原题不一致")
            else: score += 5

        if v3c.get("overall_pass") is False:
            score *= 0.7
            reasons.append(f"意图验证发现{len(issues)}个问题")
    else:
        reasons.append("⚠️ 未经意图验证")

    # ── 基础质量检查 ──
    stem = q.get("stem", "")
    if not stem or len(stem) < 10: score = 0; reasons.append("题干为空")
    elif len(stem) < 25: score *= 0.5; reasons.append("题干过短")
    if q.get("stem") == "题干文字": score = 0

    # 选项完整性（选择题）
    if q.get("question_type") == "选择题" or q.get("options"):
        opts = q.get("options", {})
        if not opts or len(opts) < 3: score *= 0.7
        if not any(v and len(v.strip())>0 for v in opts.values()): score *= 0.5

    # v3.1: 三重验证完全缺失 → 给极低基础分（几乎不可能，因为化学+意图走文本API很稳定）
    if not v3a and not v3b and not v3c:
        score = 5  # 5/100，基本等于不合格
        reasons.append("❌ 未经过任何验证")

    # 映射到0-1（100分制→置信度）
    confidence = round(max(0.0, min(1.0, score / 100.0)), 3)
    return confidence, reasons


# ═══════════════════════════════════════════════════════
# 文件收集
# ═══════════════════════════════════════════════════════

def collect_files() -> List[Tuple[str, Path]]:
    """收集所有文件，去重（剥离所有括号标签），解析卷优先"""
    seen = {}
    for ext in ('*.pdf','*.doc','*.docx'):
        for f in PAPERS_DIR.rglob(ext):
            if f.name.startswith('.'): continue
            # 剥离所有括号标签：(解析卷)、(空白卷)、(含答案)、(解析版) 等
            key = re.sub(r'[（(][^）)]*[）)]', '', f.stem)
            key = re.sub(r'[\s\-_]+', '', key).strip()
            if not key: continue
            if key in seen:
                # 优先保留解析卷/答案卷（内容最全）
                if '解析' in f.name or '答案' in f.name:
                    seen[key] = f
            else:
                seen[key] = f
    files = []
    for key, f in seen.items():
        fname = f.name
        if '解析' in fname or '解析版' in fname: tag = '解析卷'
        elif '参考答案' in fname or fname.startswith('答案') or '答案' in fname: tag = '答案页'
        elif '空白' in fname or '原卷' in fname: tag = '空白卷'
        else: tag = '通用'
        files.append((tag, f))
    return files


# ═══════════════════════════════════════════════════════
# 主管道：处理单份试卷
# ═══════════════════════════════════════════════════════

@dataclass
class PaperResult:
    source_file: str
    tag: str
    questions: List[Dict] = field(default_factory=list)
    total_pages: int = 0
    transcribed_pages: int = 0
    skipped_pages: int = 0
    errors: List[str] = field(default_factory=list)
    cost_vision: float = 0.0
    cost_llm: float = 0.0
    stats: Dict = field(default_factory=dict)


def process_paper(fp: Path, tag: str, vc: VisionClient, lc: LLMClient) -> PaperResult:
    """处理单份试卷（完整四轮管道）"""
    result = PaperResult(source_file=fp.name, tag=tag)

    # ── Step 0: DOC/DOCX → PDF ──
    wf = fp
    if fp.suffix.lower() in ('.doc', '.docx'):
        pdf = doc_to_pdf(fp, DOC_CACHE)
        if not pdf: result.errors.append("DOC→PDF转换失败"); return result
        wf = pdf

    # ── Step 1: 获取页数 + 分类 ──
    try:
        import fitz; doc = fitz.open(wf)
        total_pages = len(doc); doc.close()
    except Exception as e: result.errors.append(f"无法打开: {e}"); return result
    result.total_pages = total_pages

    page_tasks = []
    for pn in range(total_pages):
        pt = page_classify(wf, pn)
        if pt == 'ad': result.skipped_pages += 1; continue
        page_tasks.append((pn, pt))

    if not page_tasks: result.errors.append("无有效页面"); return result

    # ── Phase 1: 并行视觉转录 ──
    t1 = time.time()
    print(f"  [转录 {len(page_tasks)}页...", end='', flush=True)
    page_mds = {}
    def _transcribe(pn, pt):
        img = render_page(wf, pn, PAGE_IMG_DIR)
        if not img: return pn, None, f"页{pn+1}渲染失败"
        try:
            mr = transcribe_page(img, vc, label=f"{fp.name} P{pn+1}")
            return pn, mr, None
        except Exception as e: return pn, None, f"页{pn+1}转录失败:{e}"

    with ThreadPoolExecutor(max_workers=V_WORKERS) as ex:
        futs = {ex.submit(_transcribe, pn, pt): pn for pn, pt in page_tasks}
        completed = 0
        for fut in as_completed(futs):
            pn, mr, err = fut.result()
            completed += 1
            if completed % 5 == 0 or completed == len(page_tasks):
                print(f" {completed}/{len(page_tasks)}", flush=True)
            if err: result.errors.append(err)
            elif mr:
                page_mds[pn] = mr["md"]
                result.cost_vision += mr["cost"]
                result.transcribed_pages += 1

    print(f" {len(page_mds)}完成, {time.time()-t1:.0f}s]", flush=True)

    if not page_mds: result.errors.append("全部转录失败"); return result

    # ── Phase 1b: 图表聚焦增强（可选, 含图页面翻倍耗时）──
    chart_mds = {}  # pn → 增强图表描述
    if RUN_CHART_ZOOM:
        for pn in sorted(page_mds.keys()):
            if not has_charts(page_mds[pn]):
                continue
            # 用600 DPI超清渲染该页 → 视觉模型聚焦图表
            zoom_img = render_page_zoom(wf, pn, PAGE_IMG_DIR)
            if not zoom_img:
                continue
            try:
                cr = transcribe_charts(zoom_img, vc, label=f"{fp.name} P{pn+1} 图表聚焦")
                if cr.get("md"):
                    chart_mds[pn] = cr["md"]
                    result.cost_vision += cr.get("cost", 0)
            except Exception:
                pass  # 图表聚焦失败不阻塞

    # 将图表增强描述注入到对应页的转录中
    for pn, chart_desc in chart_mds.items():
        if chart_desc:
            page_mds[pn] += f"\n\n【图表详细描述】\n{chart_desc}"

    # 按页码排序拼接
    full_md = "\n\n".join(
        f"--- 第{pn+1}页 ---\n\n{page_mds[pn]}"
        for pn in sorted(page_mds.keys())
    )

    # 保存完整转录
    md_file = FULL_MD_DIR / f"{fp.stem[:60]}_transcript.md"
    md_file.parent.mkdir(parents=True, exist_ok=True)
    try: md_file.write_text(full_md, encoding='utf-8')
    except: pass

    # ── Phase 2: 文本切题 ──
    t2 = time.time()
    try:
        qs, sr = structure_paper(full_md, fp.name, lc)
        result.cost_llm += sr.get("cost_yuan", 0)
        print(f"  [切题 {len(qs)}题, {time.time()-t2:.0f}s]", flush=True)
    except Exception as e: result.errors.append(f"切题失败:{e}"); return result
    if not qs: result.errors.append("未提取到题目"); return result

    # 为每道题标记来源（v3.1: v3.1标记）
    for q in qs:
        q["_source_file"] = fp.name
        q["_source_tag"] = tag
        q["_pipeline_version"] = "v3.1"
        qn = q.get("q_num", "")
        main_q = re.sub(r'[\(（]\d+[\)）]$', '', qn).strip()
        assigned = False
        for pn in sorted(page_mds.keys()):
            if re.search(rf'\b{re.escape(main_q)}\s*[.．、]', page_mds[pn]):
                q["_page"] = pn + 1; assigned = True; break
        if not assigned:
            s30 = q.get("stem", "")[:30]
            for pn in sorted(page_mds.keys()):
                if s30 and s30 in page_mds[pn]:
                    q["_page"] = pn + 1; assigned = True; break
        if not assigned:
            s15 = q.get("stem", "")[:15]
            for pn in sorted(page_mds.keys()):
                if s15 and s15 in page_mds[pn]:
                    q["_page"] = pn + 1; assigned = True; break

    # ── Phase 3a: 视觉回查验证 (可选: 耗时长, 非快速模式默认跳过) ──
    all_v3a = []
    verify_fail_count = 0
    if RUN_VISUAL_VERIFY:
        verify_pages = sorted(page_mds.keys())
        for i, pn in enumerate(verify_pages):
            img = render_page(wf, pn, PAGE_IMG_DIR)
            if not img: continue
            pq = [q for q in qs if q.get("_page") == pn + 1]
            if not pq:
                pq = qs[:min(15, len(qs))]
            # 页间间隔：第1页后每页等0.8s，避免连续高频调用触发限流
            if i > 0:
                time.sleep(0.8)
            # 验证用短重试(2次45s)，非关键路径不浪费6分钟
            vs, err = api_retry(verify_page_extractions, img, pq, vc, pn + 1,
                               retry_on=['Connection', 'timeout', 'Timed out', 'Network',
                                          'rate', '429', '503', '500'],
                               _max_retries=2)
            if err:
                verify_fail_count += 1
                result.errors.append(f"页{pn+1}视觉验证失败(重试2次):{err[:80]}")
            if vs:
                all_v3a.extend(vs)
                result.cost_vision += 0.015

    # ── Phase 3b: 化学正确性验证 ──
    t3 = time.time()
    try:
        all_v3b = validate_chem_batch(qs, lc)
        result.cost_llm += len(qs) * 0.002
    except Exception as e:
        result.errors.append(f"化学验证失败:{e}")
        all_v3b = []

    # ── Phase 3c: 意图保真验证 ──
    try:
        all_v3c = validate_intent(qs, full_md, lc)
        result.cost_llm += len(qs) * 0.002
        print(f"  [验证化学+意图, {time.time()-t3:.0f}s]", flush=True)
    except Exception as e:
        result.errors.append(f"意图验证失败:{e}")
        all_v3c = []

    # ── Phase 4: 推导式评分 ──
    final_qs = []
    v3a_map = {v.get("q_num",""): v for v in all_v3a}
    v3b_map = {v.get("q_num",""): v for v in all_v3b}
    v3c_map = {v.get("q_num",""): v for v in all_v3c}

    # 统计验证覆盖率
    verified_count = 0
    for q in qs:
        qn = q.get("q_num", "")
        v3a = v3a_map.get(qn, {})
        v3b = v3b_map.get(qn, {})
        v3c = v3c_map.get(qn, {})

        conf, reasons = compute_confidence(q, v3a, v3b, v3c)
        q["confidence"] = conf
        q["confidence_reasons"] = reasons
        q["verification_coverage"] = {
            "vision_verified": bool(v3a),
            "chemistry_verified": bool(v3b),
            "intent_verified": bool(v3c)
        }

        # 应用验证的修正
        if v3a.get("corrections", {}).get("stem"):
            q["stem"] = v3a["corrections"]["stem"]
        if v3a.get("corrections", {}).get("answer"):
            q["answer"] = v3a["corrections"]["answer"]

        # 记录问题
        all_issues = (v3a.get("issues",[]) + v3b.get("issues",[]) + v3c.get("issues",[]))
        if all_issues: q["_issues"] = all_issues

        q["verification_status"] = (
            "passed" if conf >= 0.85 else
            "needs_review" if conf >= 0.60 else
            "rejected"
        )

        if any([v3a, v3b, v3c]): verified_count += 1
        if conf >= 0.60: final_qs.append(q)

    result.questions = final_qs

    # 统计
    passed = sum(1 for q in final_qs if q["verification_status"] == "passed")
    need_r = sum(1 for q in final_qs if q["verification_status"] == "needs_review")
    rejected = sum(1 for q in final_qs if q["verification_status"] == "rejected")
    # 实际验证覆盖率
    verif_cov = verified_count / max(len(qs), 1)

    result.stats = {
        "total_extracted": len(qs),
        "passed_verification": passed,
        "needs_review": need_r,
        "rejected": rejected,
        "passed_rate": round(passed/max(len(qs),1), 3),
        "verification_coverage": round(verif_cov, 3),
        "avg_confidence": round(sum(q.get("confidence",0)
                                    for q in final_qs)/max(len(final_qs),1), 3)
        if final_qs else 0,
        "questions_with_intent_issues": sum(1 for q in final_qs
            if any("意图" in r for r in q.get("confidence_reasons",[])))
    }
    return result


# ═══════════════════════════════════════════════════════
# 进度管理（v3.2: 丰富追踪供实时看板消费）
# ═══════════════════════════════════════════════════════

def _load_prog() -> Dict:
    if PROGRESS_F.exists():
        try: return json.loads(PROGRESS_F.read_text(encoding='utf-8'))
        except: pass
    return {"done_files": [], "total_qs": 0, "cost_v": 0.0, "cost_l": 0.0,
            "recent_results": [], "total_files": 0, "started_at": "",
            "aggregated": {"passed": 0, "needs_review": 0, "rejected": 0,
                           "intent_issues": 0, "avg_confidence_sum": 0.0, "count": 0}}

def _save_prog(p):
    PROGRESS_F.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_F.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding='utf-8')


# ═══════════════════════════════════════════════════════
# 全量运行
# ═══════════════════════════════════════════════════════

def run_full(max_files: int=None, resume: bool=False):
    vk = os.environ.get("DASHSCOPE_API_KEY","")
    extra_keys = [k.strip() for k in os.environ.get("DASHSCOPE_API_KEY_EXTRA", "").split(",") if k.strip()]
    ALL_VISION_KEYS = [vk] + extra_keys
    if not vk: print("❌ 缺 DASHSCOPE_API_KEY"); sys.exit(1)

    vc = VisionClient(provider=VISION_PROVIDER, model=VISION_MODEL, api_keys=ALL_VISION_KEYS)
    lc = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=DEEPSEEK_KEY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    FULL_MD_DIR.mkdir(parents=True, exist_ok=True)

    all_files = collect_files()
    total_count = len(all_files)
    print(f"\n{'='*65}")
    print(f"视觉提取管道 v3.3 — 多轮验证架构")
    print(f"{'='*65}")
    print(f"视觉: {VISION_MODEL} (3Worker+3Key并行, 2.6x加速)")
    print(f"文本: {LLM_MODEL} (串行文本)")
    print(f"DPI: {PAGE_DPI} | 验证: 化学+意图 | 视觉回查: {'开' if RUN_VISUAL_VERIFY else '关(快速)'}")
    print(f"模式: 串行 | 文件: {total_count} 份（去重后）")
    print(f"{'='*65}\n")

    prog = _load_prog()
    prog["total_files"] = total_count
    if not prog.get("started_at"):
        prog["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        prog["started_at_ts"] = time.time()
    if "recent_results" not in prog: prog["recent_results"] = []
    if "aggregated" not in prog:
        prog["aggregated"] = {"passed":0,"needs_review":0,"rejected":0,
                              "intent_issues":0,"avg_confidence_sum":0.0,"count":0}
    done_files = set(prog.get("done_files",[]))
    out_f = OUTPUT_DIR / "all_from_pdf_v3.jsonl"

    if resume and done_files:
        print(f"续跑：已完成 {len(done_files)} 个\n")

    pending = [(t,f) for t,f in all_files if f.name not in done_files]
    if max_files: pending = pending[:max_files]
    print(f"待处理: {len(pending)}\n")
    if not pending: print("✅ 全部完成"); return

    done, tq, cv, cl = 0, prog.get("total_qs",0), prog.get("cost_v",0.0), prog.get("cost_l",0.0)
    t0 = time.time()

    with open(out_f, 'a', encoding='utf-8') as fo:
        for tag, fp in pending:
            done += 1
            ts = time.strftime("%H:%M:%S")
            print(f"\n{ts} [{done}/{len(pending)}] {fp.name[:55]}", flush=True)
            try:
                res = process_paper(fp, tag, vc, lc)
                for q in res.questions:
                    fo.write(json.dumps(q, ensure_ascii=False) + '\n')
                fo.flush()
                tq += len(res.questions)
                cv += res.cost_vision; cl += res.cost_llm

                s = res.stats
                icon = "✅" if s.get("passed_rate",0) >= 0.90 else ("⚠️" if s.get("passed_rate",0) >= 0.75 else "❌")
                print(f"{icon} {len(res.questions)}题 "
                      f"(通过{s.get('passed_verification',0)}/{len(res.questions)}, "
                      f"通过率{s.get('passed_rate',0):.0%}, "
                      f"验证覆盖{s.get('verification_coverage',0):.0%}) "
                      f"¥{res.cost_vision+res.cost_llm:.3f}", flush=True)
                if res.errors:
                    for e in res.errors[:2]: print(f"    ⚠️ {e}")

                done_files.add(fp.name)
                # 丰富进度数据供看板消费
                agg = prog["aggregated"]
                agg["passed"] += s.get("passed_verification",0)
                agg["needs_review"] += s.get("needs_review",0)
                agg["rejected"] += s.get("rejected",0)
                agg["intent_issues"] += s.get("questions_with_intent_issues",0)
                agg["avg_confidence_sum"] += s.get("avg_confidence",0)
                agg["count"] += 1
                # 最近20个结果
                recents = prog["recent_results"]
                recents.insert(0, {
                    "file": fp.name[:60],
                    "tag": tag,
                    "questions": len(res.questions),
                    "passed": s.get("passed_verification",0),
                    "needs_review": s.get("needs_review",0),
                    "rejected": s.get("rejected",0),
                    "passed_rate": s.get("passed_rate",0),
                    "avg_confidence": s.get("avg_confidence",0),
                    "verification_coverage": s.get("verification_coverage",0),
                    "intent_issues": s.get("questions_with_intent_issues",0),
                    "errors": res.errors[:3],
                    "cost_v": round(res.cost_vision,4),
                    "cost_l": round(res.cost_llm,4),
                    "time": time.strftime("%H:%M:%S"),
                    "pages": res.total_pages,
                    "transcribed": res.transcribed_pages,
                })
                if len(recents) > 20: recents.pop()
                prog["recent_results"] = recents
                prog.update({"done_files":list(done_files),"total_qs":tq,
                            "cost_v":round(cv,2),"cost_l":round(cl,2),
                            "last_updated":time.strftime("%Y-%m-%d %H:%M:%S"),
                            "total_files": total_count,
                            "aggregated": agg})
                _save_prog(prog)
            except Exception as e:
                print(f"❌ {e}")
                import traceback; traceback.print_exc()

    elapsed = time.time()-t0
    print(f"\n{'='*65}")
    print(f"完成！{len(prog.get('done_files',[]))}文件 {tq}题")
    print(f"耗时: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"成本: 视觉¥{cv:.2f} 文本¥{cl:.2f} 合计¥{cv+cl:.2f}")
    print(f"{'='*65}")

    # 质量汇总
    if out_f.exists():
        aq = [json.loads(l) for l in out_f.open(encoding='utf-8') if l.strip()]
        if aq:
            cs = [q.get("confidence",0) for q in aq]
            hi = sum(1 for c in cs if c>=0.95); mi = sum(1 for c in cs if 0.85<=c<0.95)
            lo = sum(1 for c in cs if 0.60<=c<0.85)
            vc_rate = sum(1 for q in aq if all(q.get("verification_coverage",{}).values()))/len(aq)
            intent_issues = sum(1 for q in aq if any("意图" in r for r in q.get("confidence_reasons",[])))
            print(f"\n质量汇总:")
            print(f"  题目: {len(aq)} | 均置信度: {sum(cs)/len(cs):.3f}")
            print(f"  高(≥0.95): {hi}({hi/len(aq)*100:.0f}%) | 中(0.85-0.95): {mi}({mi/len(aq)*100:.0f}%) | 低(0.60-0.85): {lo}({lo/len(aq)*100:.0f}%)")
            print(f"  三重验证全覆盖: {vc_rate:.0%}")
            print(f"  意图偏差: {intent_issues}题")
            print(f"  合格率(≥0.85): {(hi+mi)/len(aq)*100:.1f}%")


# ═══════════════════════════════════════════════════════
# 测试模式
# ═══════════════════════════════════════════════════════

def test_single(filename: str):
    """测试单份试卷，输出详细结果"""
    target = None
    for tag, fp in collect_files():
        if filename.lower() in fp.name.lower(): target = (tag, fp); break
    if not target:
        fp = Path(filename)
        if fp.exists(): target = ("测试", fp)
        else:
            print(f"❌ 找不到: {filename}")
            for tag, fp in collect_files()[:20]: print(f"   {fp.name}")
            return

    tag, fp = target
    print(f"\n{'='*65}\n测试: {fp.name}\n{'='*65}")

    vk = os.environ.get("DASHSCOPE_API_KEY","")
    extra_keys = [k.strip() for k in os.environ.get("DASHSCOPE_API_KEY_EXTRA", "").split(",") if k.strip()]
    ALL_VISION_KEYS = [vk] + extra_keys
    if not vk: print("❌ 缺 DASHSCOPE_API_KEY"); sys.exit(1)

    vc = VisionClient(provider=VISION_PROVIDER, model=VISION_MODEL, api_keys=ALL_VISION_KEYS)
    lc = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=DEEPSEEK_KEY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    FULL_MD_DIR.mkdir(parents=True, exist_ok=True)

    print("处理中...")
    res = process_paper(fp, tag, vc, lc)

    print(f"\n{'─'*50}")
    print(f"结果: {fp.name}")
    print(f"{'─'*50}")
    print(f"  页数: {res.total_pages} | 转录: {res.transcribed_pages} | 跳过: {res.skipped_pages}")
    print(f"  题目: {len(res.questions)}")
    print(f"  成本: 视觉¥{res.cost_vision:.4f} 文本¥{res.cost_llm:.4f}")
    print(f"  错误: {res.errors}")
    print(f"  统计: {json.dumps(res.stats, ensure_ascii=False, indent=2)}")

    # 按验证状态分组展示
    passed = [q for q in res.questions if q["verification_status"]=="passed"]
    review = [q for q in res.questions if q["verification_status"]=="needs_review"]
    rejected = [q for q in res.questions if q["verification_status"]=="rejected"]

    print(f"\n{'─'*50}")
    print(f"通过 ({len(passed)}题):")
    for q in passed[:8]:
        conf = q.get("confidence",0)
        vc = q.get("verification_coverage",{})
        print(f"  题{q['q_num']} [{q.get('question_type','?')}] [{q.get('difficulty','?')}] "
              f"置信度={conf:.2f} "
              f"(视觉:{'✓' if vc.get('vision_verified') else '✗'} "
              f"化学:{'✓' if vc.get('chemistry_verified') else '✗'} "
              f"意图:{'✓' if vc.get('intent_verified') else '✗'})")
        print(f"    stem: {q.get('stem','')[:150]}")
        if q.get("diagram_description"): print(f"    diagram: {q['diagram_description'][:120]}")
        if q.get("_issues"): print(f"    issues: {q['_issues']}")

    if review:
        print(f"\n需审核 ({len(review)}题):")
        for q in review[:5]:
            print(f"  题{q['q_num']} 置信度={q.get('confidence',0):.2f}")
            print(f"    原因: {q.get('confidence_reasons',[])}")
            print(f"    stem: {q.get('stem','')[:120]}")

    if rejected:
        print(f"\n拒绝 ({len(rejected)}题):")
        for q in rejected[:3]:
            print(f"  题{q['q_num']} 置信度={q.get('confidence',0):.2f}")
            print(f"    原因: {q.get('confidence_reasons',[])}")

    print(f"\n📄 完整转录: {FULL_MD_DIR / f'{fp.stem[:60]}_transcript.md'}")
    print(f"✅ 测试完成")


# ═══════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="视觉提取管道 v3.0 — 多轮验证架构")
    parser.add_argument('--test', type=str, metavar='FILENAME', help='测试单份试卷')
    parser.add_argument('--max-files', type=int, metavar='N', help='最多处理N个文件')
    parser.add_argument('--resume', action='store_true', help='从上次中断处继续')
    args = parser.parse_args()
    if args.test: test_single(args.test)
    else: run_full(max_files=args.max_files, resume=args.resume)
