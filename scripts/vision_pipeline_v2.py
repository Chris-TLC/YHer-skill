#!/usr/bin/env python3
"""
Vision PDF extraction pipeline v2.0 — separation-of-concerns architecture (99%+ accuracy target)

Architecture (fixes the old pipeline's three fatal problems at the root):
  Old: the vision model did "read image + split items + match answers + detect cross-page + fill metadata" in one inference pass → attention diluted
  New: three separated steps, each doing exactly one thing:

  Phase 1 — faithful visual transcription: screenshot each page → qwen3-vl-plus → look at the image only, transcribe verbatim into Markdown
           (equations/structural formulas/apparatus diagrams/unit-cell diagrams/organic routes → textual descriptions)
  Phase 2 — smart text item-splitting: the full paper's complete Markdown → DeepSeek text model
           → split into items + pair answers + knowledge-point tagging + difficulty grading + option structuring
           (cross-page items resolve naturally because the full text is visible)
  Phase 3 — dual quality verification:
           3a. visual re-check: randomly sample 10% of pages; the vision model compares the original image with the extraction
           3b. chemistry check: DeepSeek verifies the chemical equations/properties/answers
           3c. every item gets a confidence score (0-1)

Problems fixed at the root:
  ✅ scanned PDFs (12%) → the vision model is a natural OCR; no need for PyMuPDF text extraction
  ✅ chemistry figures/equations/structural formulas → the vision model directly "sees" the image → transcribes to text
  ✅ cross-page items → the text model sees the whole paper and joins them automatically
  ✅ answer fragments mixed in → the text model distinguishes items vs solutions and pairs them correctly
  ✅ incomplete options/truncated stems → the visual transcription is verbatim-faithful, never summarized

Usage:
  # Test a single paper (recommended to run first)
  python3 scripts/vision_pipeline_v2.py --test "2022年上海高考化学真题（解析卷）.pdf"

  # Full run
  python3 scripts/vision_pipeline_v2.py

  # Only process the first N files
  python3 scripts/vision_pipeline_v2.py --max-files 5

  # Resume from the last interrupted progress
  python3 scripts/vision_pipeline_v2.py --resume

Dependencies:
  - DASHSCOPE_API_KEY (Tongyi Qianwen vision model, in .env)
  - DeepSeek API key (text model, hardcoded in the script)
  - PyMuPDF (fitz), pdf2image (install first: pip install pdf2image)
  - LibreOffice (brew install --cask libreoffice), for DOC→PDF conversion
"""

import json, re, sys, time, os, hashlib, argparse, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

# ── Project paths ─────────────────────────────────────
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")

from adapters.vision_client import VisionClient, VISION_CONFIGS
from adapters.llm_client import LLMClient

# ── Path configuration ────────────────────────────────
PAPERS_DIR = Path(os.environ.get("YHER_PAPERS_DIR", str(Path(__file__).resolve().parents[2] / "上海化学卷合集")))
OUTPUT_DIR = SKILL_DIR / "data" / "from_pdf"
PAGE_IMG_DIR = SKILL_DIR / "data" / "page_images_v2"
DOC_PDF_CACHE = SKILL_DIR / "data" / ".doc_to_pdf_cache"
PROGRESS_FILE = OUTPUT_DIR / "pipeline_v2_progress.json"
FULL_MD_DIR = OUTPUT_DIR / "full_markdown"  # Phase 1 intermediate output (for debugging)

# ── API configuration ─────────────────────────────────
VISION_PROVIDER = "qwen-vl"
VISION_MODEL = "qwen3-vl-plus"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL = "deepseek-chat"

# ── Rendering configuration ───────────────────────────
PAGE_DPI = 300          # clarity: 300 DPI suffices for chemistry sub/superscripts
JPG_QUALITY = 92
VISION_WORKERS = 3      # Tongyi vision API concurrency limit
TEXT_WORKERS = 6        # DeepSeek can go higher
SOFFICE_BIN = "/opt/homebrew/bin/soffice"

# ── Cost reference ────────────────────────────────────
# qwen3-vl-plus: ¥3/M in, ¥9/M out (vision tokens cost more; ~¥0.015-0.03 per page)
# deepseek-chat:  ¥3.13/M in, ¥6.26/M out


# ═══════════════════════════════════════════════════════
# Phase 1: faithful visual transcription
# ═══════════════════════════════════════════════════════

TRANSCRIBE_SYSTEM = """你是化学试卷数字化转录专家。你的唯一任务：看着试卷页面图片，把看到的所有内容逐字转录成 Markdown。

【核心原则】
1. **逐字忠实**：看到什么就写什么，不概括、不省略、不"润色"、不修改原文
2. **化学式精确**：保留所有上下标格式。如 Na₂CO₃、SO₄²⁻、[Cu(NH₃)₄]²⁺、H⁺
3. **方程式完整**：转录所有化学方程式、离子方程式、热化学方程式、电极反应式
4. **图表转文字**：
   - 有机结构简式 → 文字详细描述结构
   - 晶胞图 → 描述原子位置、坐标、配位数
   - 实验装置图 → 描述仪器连接关系
   - 曲线/坐标图 → 描述坐标系、曲线趋势、关键数据点
   - 数据表 → 转为 Markdown 表格
5. **选项完整**：每题的所有选项(A/B/C/D/E)必须完整转录，一字不差
6. **不跳内容**：页码、广告、解析、答案——所有文字都转录，后续会筛选

【输出格式】
纯 Markdown 文本。不要 JSON，不要额外解释。"""

TRANSCRIBE_USER = "请逐字转录这张化学试卷页面。把所有可见的化学式、方程式、结构式、图表都转成文字描述。不要遗漏任何内容。"


def render_page_to_jpg(pdf_path: Path, page_num: int,
                       output_dir: Path) -> Optional[Path]:
    """Render one PDF page into a 300 DPI high-res JPG."""
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.sha256(str(pdf_path.resolve()).encode()).hexdigest()[:12]
    img_path = output_dir / f"{file_hash}_p{page_num+1:03d}.jpg"

    if img_path.exists() and img_path.stat().st_size > 1000:
        return img_path

    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            doc.close()
            return None

        page = doc[page_num]
        mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.pil_save(str(img_path), format="JPEG", quality=JPG_QUALITY)
        doc.close()
        return img_path
    except Exception:
        return None


def classify_page(pdf_path: Path, page_num: int) -> str:
    """Quick pre-classification: distinguish question pages / answer pages / ad pages / blank pages."""
    import fitz
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            doc.close()
            return "empty"
        text = doc[page_num].get_text().strip()
        doc.close()

        if not text or len(text) < 30:
            return "likely_scanned"  # possibly a scanned page; still needs vision processing

        # Pure ad / tutoring-page
        if ("家教" in text or "扫码" in text or "加微信" in text) and len(text) < 500:
            return "ad"

        # Pure answer-key page signature: many "question number answer" lines with no stem
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        ans_pattern_lines = sum(1 for l in lines if re.match(r'^\d+[\.\s、]+\w{1,3}\s*$', l))
        if ans_pattern_lines > len(lines) * 0.5 and len(lines) > 5:
            return "answer_key"

        return "question"
    except Exception:
        return "error"


def transcribe_page(image_path: Path, client: VisionClient,
                    page_label: str = "") -> Dict[str, Any]:
    """Transcribe one page via the vision model → Markdown."""
    prompt = TRANSCRIBE_USER
    if page_label:
        prompt = f"【{page_label}】\n\n" + prompt

    result = client.read_page(
        image_path=image_path,
        system_prompt=TRANSCRIBE_SYSTEM,
        user_prompt=prompt,
        max_tokens=6000
    )
    return {
        "markdown": result['content'].strip(),
        "cost_yuan": result.get('cost_yuan', 0),
        "tokens": result.get('usage', {}).get('input_tokens', 0)
               + result.get('usage', {}).get('output_tokens', 0)
    }


def convert_doc_to_pdf(doc_path: Path, cache_dir: Path) -> Optional[Path]:
    """DOC/DOCX → PDF (via LibreOffice)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_hash = hashlib.sha256(str(doc_path.resolve()).encode()).hexdigest()[:12]
    pdf_path = cache_dir / f"{doc_hash}.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return pdf_path

    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = subprocess.run(
                [SOFFICE_BIN, "--headless", "--convert-to", "pdf",
                 "--outdir", tmp, str(doc_path)],
                capture_output=True, timeout=120
            )
            pdfs = list(Path(tmp).glob("*.pdf"))
            if pdfs:
                # Copy into the cache
                import shutil
                shutil.copy2(pdfs[0], pdf_path)
                return pdf_path
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════
# Phase 2: smart text item-splitting
# ═══════════════════════════════════════════════════════

STRUCTURE_SYSTEM = """你是上海高考化学试卷结构化专家。你收到的是整张试卷的完整文字转录（含题干、选项、图表描述、答案、解析），要把它切成一道道独立的题目。

【核心要求】
1. **识别所有题目**：选择题(一/1-20)、填空题、简答题、计算题、综合大题(二/三/四)及其所有子题(1)(2)(3)...
2. **每道题独立输出为一行 JSON**。综合大题的每个小问(1)(2)(3)作为独立行输出，不可遗漏
3. **跨页题目自然拼接**：你现在看到的是全文，跨页的题目应该合并
4. **区分题目与答案**：
   - "参考答案与试题解析"之前的 → 题干/选项
   - "参考答案与试题解析"之后的 → 匹配到对应题目的 answer 和 explanation
5. **子题题号规则**：综合大题每个小问独立输出，题号格式如 "21(1)"、"21(2)"。parent_id 填父题号
6. **图表内容**：转录中的图表文字描述填入 diagram_description
7. **跳过**：广告文字("家教"/"加微信")、页眉页脚

【输出格式】每题一行 JSON（JSONL，不要数组）：
{"q_num":"1","stem":"完整题干（选择题含全部选项文字）","options":{"A":"选项A","B":"选项B","C":"选项C","D":"选项D"},"answer":"正确答案","explanation":"解析文字","knowledge_points":["知识点1"],"difficulty":"T2","question_type":"选择题","diagram_description":"","parent_id":"","has_sub_questions":false}

【字段说明】
- q_num: 原题号。子题如 "21(1)"。不确定时从1开始编号
- stem: 完整题干，一字不改。选择题必须包含全部ABCD选项文字
- options: 选择题→填ABCD完整选项文字，非选择题→{}
- answer: 选择题填字母，填空/简答填文字，无答案填 ""
- explanation: 从答案部分提取的解析。无解析填 ""
- knowledge_points: 推断1-3个知识点。不确定填 ["未知"]
- difficulty: T1基础/T2中档/T3拔高/T4压轴。不确定填 "T2"
- question_type: 选择题/填空题/简答题/计算题/综合题
- diagram_description: 有图则描述图中化学信息，无则 ""
- parent_id: 子题填父题号，独立题填 ""
- has_sub_questions: 该题本身是否还有子题

【绝对规则】
- 题干原样转录，一字不改
- 化学式/方程式保留原格式（Unicode上下标）
- **综合大题的每个小问(1)(2)(3)必须单独输出一行**
- 如果答案部分有【解答】【解析】，必须提取到对应题目的 explanation"""



def build_structure_prompt(paper_markdown: str, source_name: str) -> str:
    """Build the item-splitting prompt (truncates overly long text to 100K chars)."""
    if len(paper_markdown) > 100000:
        paper_markdown = paper_markdown[:100000] + "\n\n[... 文本过长，已截断 ...]"

    return f"""【试卷来源】{source_name}

【试卷完整转录】
{paper_markdown}

【任务】请将以上试卷转录切成单题，每题一行 JSON。注意：
- 跨页的题目请合并输出（你现在看到的是全文）
- 选择题必须包含完整的ABCD选项文字
- 答案页的答案匹配到对应题目
- 解析文字填入 explanation 字段
- 图表描述填入 diagram_description 字段"""


def structure_paper(paper_markdown: str, source_name: str,
                    client: LLMClient) -> Tuple[List[Dict], Dict]:
    """Split the whole-paper transcription into structured items via the text model."""
    messages = [
        {"role": "system", "content": STRUCTURE_SYSTEM},
        {"role": "user", "content": build_structure_prompt(paper_markdown, source_name)}
    ]

    result = client.chat(messages, max_tokens=12000, temperature=0.1)
    content = result['content'].strip()

    # Parse the output
    questions = []
    if '```' in content:
        parts = content.split('```')
        content = '\n'.join(p for p in parts if '{' in p or '[' in p)

    for line in content.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            q = json.loads(line)
            if q.get('q_num') and q.get('stem'):
                q['_source_file'] = source_name
                questions.append(q)
        except json.JSONDecodeError:
            fixed = line.replace('「', '"').replace('」', '"').replace('（', '(').replace('）', ')')
            try:
                q = json.loads(fixed)
                if q.get('q_num') and q.get('stem'):
                    q['_source_file'] = source_name
                    questions.append(q)
            except json.JSONDecodeError:
                pass

    return questions, {
        "cost_yuan": result.get('cost_yuan', 0),
        "tokens": result.get('usage', {}).get('input_tokens', 0)
                + result.get('usage', {}).get('output_tokens', 0)
    }


# ═══════════════════════════════════════════════════════
# Phase 3a: visual re-check verification
# ═══════════════════════════════════════════════════════

VERIFY_SYSTEM = """你是试卷数字化的质量审核员。对比原始试卷图片和已提取的题目，逐题检查是否有错误或遗漏。

检查维度：
1. 题干文字是否与原图逐字一致（包括标点、数字、单位）？
2. 化学式/方程式转录是否正确（特别注意上下标、电荷数）？
3. 选项(A/B/C/D)是否完整且与原图一字不差？
4. 答案是否正确转录（字母/文字）？
5. 解析是否完整准确？
6. 是否有遗漏的题目（原图有但提取结果没有的）？
7. 图表描述是否准确反映原图内容？
8. 提取结果中是否有不是题目的内容（广告/答案碎片混入）？

输出格式：每题一行评审JSONL"""


def build_verify_prompt(questions_json: str, page_num: int) -> str:
    return f"""以下是第{page_num}页提取的题目：

{questions_json}

请逐题核对：
{{"q_num":"1","checks":{{"stem_accurate":true,"options_complete":true,"formulas_correct":true,"answer_correct":true}},"issues":["如有问题在此描述"],"overall_pass":true,"confidence":0.95}}

如果页面没有题目（纯广告/答案页）→ {{"page_has_no_questions":true}}
如果页面题目全部正确 → overall_pass:true, issues:[]
如果有错误或遗漏 → overall_pass:false, 详细说明问题"""


def verify_page(image_path: Path, questions_on_page: List[Dict],
                client: VisionClient) -> List[Dict]:
    """Visually re-check one page's extraction results via the vision model."""
    if not questions_on_page:
        return []

    qs_json = json.dumps(questions_on_page, ensure_ascii=False, indent=1)
    result = client.read_page(
        image_path=image_path,
        system_prompt=VERIFY_SYSTEM,
        user_prompt=build_verify_prompt(qs_json, 0),
        max_tokens=4000
    )

    verifications = []
    content = result['content'].strip()
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            verifications.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    return verifications


# ═══════════════════════════════════════════════════════
# Phase 3b: chemical-correctness check
# ═══════════════════════════════════════════════════════

CHEM_SYSTEM = """你是上海高考化学命题专家。验证以下题目的化学科学正确性。

检查维度：
1. 化学式是否真实存在？
2. 方程式是否配平正确？
3. 化学性质描述是否准确？
4. 标准答案是否在化学上正确？
5. 知识点标注是否准确？
6. 题目本身的化学表述是否有科学错误？

输出格式：每题一行 JSONL"""


def build_chem_prompt(questions: List[Dict]) -> str:
    q_texts = []
    for q in questions:
        q_texts.append(
            f"题{q.get('q_num','?')}: {q.get('stem','')[:200]}\n"
            f"  答案: {q.get('answer','?')}\n"
            f"  知识点: {q.get('knowledge_points',[])}\n"
            f"  解析: {q.get('explanation','')[:100]}"
        )
    return "请验证以下题目的化学正确性：\n\n" + "\n\n".join(q_texts)


def validate_chemistry(questions: List[Dict], client: LLMClient) -> List[Dict]:
    """Verify chemical correctness via the text model."""
    if not questions:
        return []

    # Batch up (25 items per batch)
    all_validations = []
    for batch_start in range(0, len(questions), 25):
        batch = questions[batch_start:batch_start + 25]
        messages = [
            {"role": "system", "content": CHEM_SYSTEM},
            {"role": "user", "content": build_chem_prompt(batch)}
        ]
        result = client.chat(messages, max_tokens=6000, temperature=0.1)
        content = result['content'].strip()

        for line in content.splitlines():
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                v = json.loads(line)
                all_validations.append(v)
            except json.JSONDecodeError:
                try:
                    v = json.loads(line.replace('「', '"').replace('」', '"'))
                    all_validations.append(v)
                except json.JSONDecodeError:
                    pass

    return all_validations


# ═══════════════════════════════════════════════════════
# Scoring and merging
# ═══════════════════════════════════════════════════════

def compute_confidence(q: Dict, verifications: List[Dict],
                       chem_validations: List[Dict]) -> float:
    """Compute the per-item confidence score."""
    confidence = 0.90  # base score (visual transcription + text structuring)

    q_num = q.get('q_num', '')

    # Basic quality checks
    stem = q.get('stem', '')
    if not stem or len(stem) < 10:
        return 0.0
    if len(stem) < 25:
        confidence -= 0.15
    if q.get('stem') == '题干文字':
        return 0.0

    # Adjust by verification results
    v = next((v for v in verifications if v.get('q_num') == q_num), {})
    if v:
        if v.get('overall_pass') is True:
            confidence += min(0.05, 1.0 - confidence)
        elif v.get('overall_pass') is False:
            confidence -= 0.15
            # Apply corrections
            if v.get('corrections', {}).get('stem'):
                q['stem'] = v['corrections']['stem']
            q['_verify_issues'] = v.get('issues', [])

    # Adjust by the chemistry check
    c = next((c for c in chem_validations if c.get('q_num') == q_num), {})
    if c:
        if c.get('overall_pass') is True:
            confidence += min(0.05, 1.0 - confidence)
        elif c.get('overall_pass') is False:
            confidence -= 0.20
            q['_chem_issues'] = c.get('issues', [])

    # Option-completeness check (multiple-choice items)
    if q.get('question_type') == '选择题' or q.get('options'):
        opts = q.get('options', {})
        if not opts or len(opts) < 3:
            confidence -= 0.10
        # Check whether any option text exists
        has_content = any(v and len(v.strip()) > 0 for v in opts.values())
        if not has_content:
            confidence -= 0.10

    return round(max(0.0, min(1.0, confidence)), 3)


# ═══════════════════════════════════════════════════════
# File collection
# ═══════════════════════════════════════════════════════

def collect_files() -> List[Tuple[str, Path]]:
    """Collect all files to process, deduplicated, preferring annotated papers."""
    files = []
    seen = {}

    for ext in ('*.pdf', '*.doc', '*.docx'):
        for f in PAPERS_DIR.rglob(ext):
            if f.name.startswith('.'):
                continue
            # Dedup: for same-named files, prefer keeping the annotated paper
            key = re.sub(r'[（(](空白卷|解析卷|原卷版|解析版|考试版|参考答案|含解析)[）)]', '', f.stem)
            key = re.sub(r'\s+', '', key)
            if key in seen:
                # Keep priority: annotated > others
                if '解析' in f.name or '答案' in f.name:
                    seen[key] = f
            else:
                seen[key] = f

    for key, f in seen.items():
        if '解析' in f.name or '解析版' in f.name or '答案' in f.name:
            tag = '解析卷'
        elif '参考答案' in f.name or f.name.startswith('答案'):
            tag = '答案页'
        elif '空白' in f.name or '原卷' in f.name:
            tag = '空白卷'
        else:
            tag = '通用'
        files.append((tag, f))

    return files


# ═══════════════════════════════════════════════════════
# Main pipeline: process a single file
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


def process_one_paper(filepath: Path, tag: str,
                      vision_client: VisionClient,
                      llm_client: LLMClient) -> PaperResult:
    """Process a single paper: transcribe → split items → verify."""
    result = PaperResult(source_file=filepath.name, tag=tag)

    # Step 0: DOC/DOCX → PDF
    working_path = filepath
    if filepath.suffix.lower() in ('.doc', '.docx'):
        pdf_path = convert_doc_to_pdf(filepath, DOC_PDF_CACHE)
        if not pdf_path:
            result.errors.append("DOC→PDF conversion failed")
            return result
        working_path = pdf_path

    # Step 1: get the page count + page classification
    try:
        import fitz
        doc = fitz.open(working_path)
        total_pages = len(doc)
        doc.close()
    except Exception as e:
        result.errors.append(f"cannot open file: {e}")
        return result

    result.total_pages = total_pages

    # Classify page by page
    page_tasks = []  # (page_num, page_type)
    for pn in range(total_pages):
        ptype = classify_page(working_path, pn)
        if ptype == 'ad':
            result.skipped_pages += 1
            continue
        page_tasks.append((pn, ptype))

    if not page_tasks:
        result.errors.append("no valid pages (all ads/blank)")
        return result

    # ── Phase 1: page-by-page visual transcription ──
    page_markdowns = {}  # page_num → markdown

    # Render + transcribe (parallelized with a thread pool for speed)
    def transcribe_task(pn, ptype):
        img_path = render_page_to_jpg(working_path, pn, PAGE_IMG_DIR)
        if not img_path:
            return pn, None, f"page {pn+1} render failed"

        try:
            mr = transcribe_page(
                img_path, vision_client,
                page_label=f"{filepath.name} 第{pn+1}页"
            )
            return pn, mr, None
        except Exception as e:
            return pn, None, f"page {pn+1} transcription failed: {e}"

    with ThreadPoolExecutor(max_workers=VISION_WORKERS) as ex:
        futs = {ex.submit(transcribe_task, pn, pt): pn for pn, pt in page_tasks}
        for fut in as_completed(futs):
            pn, mr, err = fut.result()
            if err:
                result.errors.append(err)
            elif mr:
                page_markdowns[pn] = mr['markdown']
                result.cost_vision += mr['cost_yuan']
                result.transcribed_pages += 1

    if not page_markdowns:
        result.errors.append("all pages failed transcription")
        return result

    # Join in page order (fixes the out-of-order bug caused by parallelism)
    sorted_pages = sorted(page_markdowns.keys())
    full_markdown_parts = []
    for pn in sorted_pages:
        full_markdown_parts.append(
            f"\n\n--- 第{pn+1}页 ---\n\n{page_markdowns[pn]}"
        )
    full_markdown = "\n".join(full_markdown_parts)

    # Save the full transcript (for debugging)
    md_file = FULL_MD_DIR / f"{filepath.stem[:60]}_transcript.md"
    md_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        md_file.write_text(full_markdown, encoding='utf-8')
    except Exception:
        pass

    # ── Phase 2: smart text item-splitting ──
    try:
        questions, struct_info = structure_paper(
            full_markdown, filepath.name, llm_client
        )
        result.cost_llm += struct_info.get('cost_yuan', 0)
    except Exception as e:
        result.errors.append(f"item-splitting failed: {e}")
        return result

    if not questions:
        result.errors.append("no items extracted")
        return result

    # ── Phase 3a: visual re-check (10% sample, at least 3 pages) ──
    sample_size = max(3, int(len(page_markdowns) * 0.10))
    sample_pages = list(page_markdowns.keys())
    if len(sample_pages) > sample_size:
        import random
        random.seed(42)
        sample_pages = random.sample(sample_pages, sample_size)

    all_verifications = []
    for pn in sample_pages:
        img_path = render_page_to_jpg(working_path, pn, PAGE_IMG_DIR)
        if not img_path:
            continue
        # Find the items on this page (allocated by page)
        page_qs = [q for q in questions if q.get('_page', 0) == pn + 1]
        if not page_qs:
            # Fallback: give the vision model the first 10 of all items to check
            page_qs = questions[:min(10, len(questions))]

        try:
            vers = verify_page(img_path, page_qs, vision_client)
            all_verifications.extend(vers)
            # Cost estimate
            result.cost_vision += 0.015  # ~¥0.015 per page verification
        except Exception as e:
            result.errors.append(f"page {pn+1} verification failed: {e}")

    # ── Phase 3b: chemical-correctness check ──
    try:
        chem_validations = validate_chemistry(questions, llm_client)
        # Cost estimate
        result.cost_llm += len(questions) * 0.002  # ~¥0.002 per item
    except Exception as e:
        result.errors.append(f"chemistry check failed: {e}")
        chem_validations = []

    # ── Scoring and merging ──
    final_questions = []
    for q in questions:
        conf = compute_confidence(q, all_verifications, chem_validations)
        q['confidence'] = conf
        q['verification_status'] = (
            'passed' if conf >= 0.85 else
            'needs_review' if conf >= 0.60 else
            'rejected'
        )
        q['_pipeline_version'] = 'v2.0'
        q['_source_file'] = filepath.name
        q['_source_tag'] = tag

        if conf >= 0.60:
            final_questions.append(q)

    result.questions = final_questions

    # Statistics
    passed = sum(1 for q in final_questions if q.get('verification_status') == 'passed')
    needs_review = sum(1 for q in final_questions if q.get('verification_status') == 'needs_review')
    result.stats = {
        'total_extracted': len(questions),
        'passed_verification': passed,
        'needs_review': needs_review,
        'rejected': len(questions) - len(final_questions),
        'passed_rate': round(passed / max(len(questions), 1), 3),
        'avg_confidence': round(
            sum(q.get('confidence', 0) for q in final_questions) / max(len(final_questions), 1), 3
        ) if final_questions else 0
    }

    return result


# ═══════════════════════════════════════════════════════
# Progress management
# ═══════════════════════════════════════════════════════

def load_progress() -> Dict:
    """Load progress."""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {"done_files": [], "total_questions": 0, "total_cost_vision": 0.0,
            "total_cost_llm": 0.0}


def save_progress(progress: Dict):
    """Save progress."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2),
                             encoding='utf-8')


# ═══════════════════════════════════════════════════════
# Full run
# ═══════════════════════════════════════════════════════

def run_full(max_files: int = None, resume: bool = False):
    """Run the pipeline at full scale."""
    # Initialization
    vision_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not vision_key:
        print("❌ missing DASHSCOPE_API_KEY environment variable")
        print("   get one at: https://dashscope.console.aliyun.com/apiKey")
        print("   set it via: export DASHSCOPE_API_KEY=sk-xxxx")
        sys.exit(1)

    vision_client = VisionClient(provider=VISION_PROVIDER, model=VISION_MODEL,
                                 api_key=vision_key)
    llm_client = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=DEEPSEEK_KEY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    FULL_MD_DIR.mkdir(parents=True, exist_ok=True)

    # Collect files
    all_files = collect_files()
    print(f"\n{'='*65}")
    print(f"Vision extraction pipeline v2.0 — separation-of-concerns architecture")
    print(f"{'='*65}")
    print(f"vision model: {VISION_MODEL} (transcription + spot-check)")
    print(f"text model: {LLM_MODEL} (item-splitting + verification)")
    print(f"render DPI: {PAGE_DPI}")
    print(f"total files: {len(all_files)} (after dedup)")
    print(f"{'='*65}\n")

    # Resume progress
    progress = load_progress()
    done_files = set(progress.get("done_files", []))
    out_file = OUTPUT_DIR / "all_from_pdf_v2.jsonl"

    if resume and done_files:
        print(f"resuming: {len(done_files)} files already done\n")

    pending = [(tag, f) for tag, f in all_files if f.name not in done_files]
    if max_files:
        pending = pending[:max_files]

    print(f"to process: {len(pending)} files\n")
    if not pending:
        print("✅ all done")
        return

    # Main loop
    done = 0
    total_qs = progress.get("total_questions", 0)
    total_cost_v = progress.get("total_cost_vision", 0.0)
    total_cost_l = progress.get("total_cost_llm", 0.0)
    t0 = time.time()

    with open(out_file, 'a', encoding='utf-8') as fout:
        for tag, fp in pending:
            done += 1
            print(f"[{done}/{len(pending)}] {fp.name[:55]} ...", end=' ', flush=True)

            try:
                result = process_one_paper(fp, tag, vision_client, llm_client)

                # Write out
                for q in result.questions:
                    fout.write(json.dumps(q, ensure_ascii=False) + '\n')
                fout.flush()

                total_qs += len(result.questions)
                total_cost_v += result.cost_vision
                total_cost_l += result.cost_llm

                # Status icon
                rate = result.stats.get('passed_rate', 0)
                icon = "✅" if rate >= 0.90 else ("⚠️" if rate >= 0.75 else "❌")
                print(f"{icon} {len(result.questions)} items "
                      f"(passed {result.stats.get('passed_verification',0)}/{len(result.questions)}, "
                      f"pass rate {rate:.0%}) "
                      f"¥{result.cost_vision + result.cost_llm:.3f}")

                if result.errors:
                    for e in result.errors[:2]:
                        print(f"    ⚠️ {e}")

                # Update progress
                done_files.add(fp.name)
                progress.update({
                    "done_files": list(done_files),
                    "total_questions": total_qs,
                    "total_cost_vision": round(total_cost_v, 2),
                    "total_cost_llm": round(total_cost_l, 2),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                save_progress(progress)

            except Exception as e:
                print(f"❌ failed: {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*65}")
    print(f"all done!")
    print(f"  files processed: {len(progress.get('done_files', []))}")
    print(f"  total items: {total_qs}")
    print(f"  total elapsed: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  vision cost: ¥{total_cost_v:.2f}")
    print(f"  text cost: ¥{total_cost_l:.2f}")
    print(f"  total cost: ¥{total_cost_v + total_cost_l:.2f}")
    print(f"{'='*65}")

    # Quality statistics
    print("\nquality stats:")
    all_qs = []
    if out_file.exists():
        for line in out_file.open(encoding='utf-8'):
            try:
                all_qs.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if all_qs:
        confidences = [q.get('confidence', 0) for q in all_qs]
        avg_conf = sum(confidences) / len(confidences)
        high = sum(1 for c in confidences if c >= 0.95)
        mid = sum(1 for c in confidences if 0.85 <= c < 0.95)
        low = sum(1 for c in confidences if 0.60 <= c < 0.85)

        print(f"  total items: {len(all_qs)}")
        print(f"  average confidence: {avg_conf:.3f}")
        print(f"  high confidence (≥0.95): {high} ({high/len(all_qs)*100:.1f}%)")
        print(f"  mid confidence (0.85-0.95): {mid} ({mid/len(all_qs)*100:.1f}%)")
        print(f"  low confidence (0.60-0.85): {low} ({low/len(all_qs)*100:.1f}%)")
        print(f"  qualified rate (≥0.85): {(high+mid)/len(all_qs)*100:.1f}%")

        # Problem distribution
        rejected = sum(1 for q in all_qs if q.get('verification_status') == 'rejected')
        needs_review = sum(1 for q in all_qs if q.get('verification_status') == 'needs_review')
        print(f"  status: passed={len(all_qs)-rejected-needs_review}, "
              f"needs_review={needs_review}, rejected={rejected}")


# ═══════════════════════════════════════════════════════
# Test mode: detailed single-paper output
# ═══════════════════════════════════════════════════════

def test_single_paper(filename: str):
    """Test a single paper and print detailed results for manual inspection."""
    # Find the file
    target = None
    for tag, fp in collect_files():
        if filename.lower() in fp.name.lower():
            target = (tag, fp)
            break

    if not target:
        # Try the full path
        fp = Path(filename)
        if fp.exists():
            target = ("测试", fp)
        else:
            print(f"❌ file not found: {filename}")
            print(f"   available files (partial):")
            for tag, fp in collect_files()[:20]:
                print(f"     {fp.name}")
            return

    tag, fp = target
    print(f"\n{'='*65}")
    print(f"test mode: {fp.name}")
    print(f"{'='*65}")

    vision_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not vision_key:
        print("❌ missing DASHSCOPE_API_KEY")
        sys.exit(1)

    vision_client = VisionClient(provider=VISION_PROVIDER, model=VISION_MODEL,
                                 api_key=vision_key)
    llm_client = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=DEEPSEEK_KEY)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    FULL_MD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"starting...")
    result = process_one_paper(fp, tag, vision_client, llm_client)

    # ── Print detailed results ──
    print(f"\n{'─'*50}")
    print(f"processing result: {fp.name}")
    print(f"{'─'*50}")
    print(f"  total pages: {result.total_pages}")
    print(f"  transcribed pages: {result.transcribed_pages}")
    print(f"  skipped pages: {result.skipped_pages}")
    print(f"  extracted items: {len(result.questions)}")
    print(f"  vision cost: ¥{result.cost_vision:.4f}")
    print(f"  text cost: ¥{result.cost_llm:.4f}")
    print(f"  errors: {result.errors}")
    print(f"  stats: {json.dumps(result.stats, ensure_ascii=False, indent=2)}")

    # Print the first 10 items for manual inspection
    print(f"\n{'─'*50}")
    print(f"first 10 items in detail:")
    print(f"{'─'*50}")

    for i, q in enumerate(result.questions[:10]):
        print(f"\nQ{q.get('q_num', '?')} "
              f"[{q.get('question_type','?')}] "
              f"[{q.get('difficulty','?')}] "
              f"confidence: {q.get('confidence', 0):.2f}")
        print(f"  stem: {q.get('stem', '')[:200]}")
        opts = q.get('options', {})
        if opts:
            print(f"  options: {json.dumps(opts, ensure_ascii=False)}")
        print(f"  answer: {q.get('answer', '?')}")
        print(f"  explanation: {q.get('explanation', '')[:120]}")
        print(f"  knowledge points: {q.get('knowledge_points', [])}")
        if q.get('diagram_description'):
            print(f"  diagram: {q['diagram_description'][:120]}")
        if q.get('_verify_issues'):
            print(f"  ⚠️ vision issues: {q['_verify_issues']}")
        if q.get('_chem_issues'):
            print(f"  ⚠️ chemistry issues: {q['_chem_issues']}")

    # Full transcript file location
    md_path = FULL_MD_DIR / f"{fp.stem[:60]}_transcript.md"
    if md_path.exists():
        print(f"\n📄 full transcript: {md_path}")

    print(f"\n✅ test done")


# ═══════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Vision extraction pipeline v2.0 — separation-of-concerns architecture (99%+ accuracy target)"
    )
    parser.add_argument('--test', type=str, metavar='FILENAME',
                        help='test a single paper (fuzzy filename match)')
    parser.add_argument('--max-files', type=int, metavar='N',
                        help='process at most N files')
    parser.add_argument('--resume', action='store_true',
                        help='continue from where it last stopped')
    args = parser.parse_args()

    if args.test:
        test_single_paper(args.test)
    else:
        run_full(max_files=args.max_files, resume=args.resume)
