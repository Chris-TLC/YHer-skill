"""
PDF/DOC 原始卷子提取管道
策略：用 LibreOffice 把 .doc/.docx 转文本，PDF直接读，
      对整张解析卷送 LLM 一次性提取所有题目+答案+解析
"""
import json, os, re, sys, subprocess, time, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from adapters.llm_client import LLMClient

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
PAPERS_DIR = Path("/Users/mac/Desktop/项目文件夹/Tools/上海化学卷合集")
OUTPUT_DIR = Path(__file__).parent.parent / "data/from_pdf"
MAX_WORKERS = 6
SOFFICE = "/opt/homebrew/bin/soffice"

# ── 文本提取 ─────────────────────────────────────────

def extract_text_doc(filepath: Path) -> str:
    """用 LibreOffice 把 .doc/.docx 转为纯文本"""
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [SOFFICE, "--headless", "--convert-to", "txt:Text",
             "--outdir", tmp, str(filepath)],
            capture_output=True, timeout=60
        )
        out_files = list(Path(tmp).glob("*.txt"))
        if not out_files:
            raise RuntimeError(f"LibreOffice 转换失败: {result.stderr.decode()[:200]}")
        return out_files[0].read_text(encoding="utf-8", errors="ignore")


def extract_text_pdf(filepath: Path) -> str:
    """用 PyMuPDF 提取 PDF 文本"""
    try:
        import fitz
        doc = fitz.open(filepath)
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        raise RuntimeError("PyMuPDF 未安装: pip install pymupdf")


def extract_text(filepath: Path) -> str:
    ext = filepath.suffix.lower()
    if ext in ('.doc', '.docx'):
        return extract_text_doc(filepath)
    elif ext == '.pdf':
        return extract_text_pdf(filepath)
    else:
        raise ValueError(f"不支持的格式: {ext}")


# ── LLM 提取题目 ─────────────────────────────────────

EXTRACT_PROMPT = """\
以下是一张上海化学解析卷的全文。请提取所有题目，输出JSONL格式（每题一行）。

【卷子全文】
{text}

每题输出一行JSON：
{{"q_num":"1","stem":"题干文字","options":{{"A":"选项A","B":"","C":"","D":""}},"answer":"C","explanation":"解析文字","knowledge_points":["知识点1"],"difficulty":"T1"}}

规则：
- 选择题填options，填空/简答题options留空 {{}}
- answer只填字母（选择题）或关键答案（其他题）
- difficulty: T1基础/T2中档/T3拔高/T4压轴
- 如果题目有化学方程式或结构式，用文字描述（"碳酸钠溶液与盐酸反应"）
- 只输出JSONL，不要其他文字"""


def extract_questions_from_text(text: str, source_name: str,
                                 client: LLMClient) -> list:
    # 截断过长文本（保留前80K字符）
    if len(text) > 80000:
        text = text[:80000]

    result = client.chat(
        [{"role": "user",
          "content": EXTRACT_PROMPT.format(text=text)}],
        max_tokens=8000, temperature=0.1
    )
    content = result['content'].strip()
    if '```' in content:
        parts = content.split('```')
        content = '\n'.join(p.lstrip('json\n') for p in parts if '{' in p)

    questions = []
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            q = json.loads(line)
            q['source_file'] = source_name
            questions.append(q)
        except Exception:
            pass

    return questions, result['usage']


# ── 主流程 ────────────────────────────────────────────

def collect_exam_files() -> list:
    """收集所有解析卷文件（含答案的版本优先）"""
    files = []
    for ext in ('*.pdf', '*.doc', '*.docx'):
        for f in PAPERS_DIR.rglob(ext):
            name = f.name
            # 优先处理解析卷（含答案+解析）
            if '解析' in name or '解析版' in name or '答案' in name:
                files.append(('解析卷', f))
    # 去重（同名文件保留一个）
    seen = set()
    unique = []
    for tag, f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append((tag, f))
    return unique


def process_one_file(args) -> tuple:
    tag, filepath = args
    client = LLMClient(provider='deepseek', model='deepseek-chat', api_key=API_KEY)

    text = extract_text(filepath)
    if len(text.strip()) < 200:
        raise ValueError("文本过短，可能是扫描版或提取失败")

    questions, usage = extract_questions_from_text(
        text, filepath.name, client
    )
    return filepath.name, questions, usage


def run_full(max_workers: int = MAX_WORKERS):
    files = collect_exam_files()
    print(f"\n{'='*55}")
    print(f"PDF管道: {len(files)} 份解析卷")
    print(f"并发: {max_workers} 线程")
    print('='*55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "all_from_pdf.jsonl"

    # 断点续跑
    done_files = set()
    if out_file.exists():
        for line in out_file.open():
            r = json.loads(line)
            done_files.add(r.get('source_file', ''))
        print(f"  续跑：跳过 {len(done_files)} 份")

    pending = [(t, f) for t, f in files if f.name not in done_files]
    print(f"  待处理: {len(pending)} 份\n")

    done = 0
    total_q = 0
    t0 = time.time()

    with open(out_file, 'a') as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(process_one_file, a): a[1].name for a in pending}
            for fut in as_completed(futs):
                fname = futs[fut]
                done += 1
                try:
                    name, questions, usage = fut.result()
                    for q in questions:
                        fout.write(json.dumps(q, ensure_ascii=False) + '\n')
                    fout.flush()
                    total_q += len(questions)
                    print(f"  [{done}/{len(pending)}] {name[:40]} "
                          f"→ {len(questions)}题 | 合计{total_q}")
                except Exception as e:
                    print(f"  [{done}/{len(pending)}] {fname[:40]} ✗ {e}")

    elapsed = time.time() - t0
    print(f"\n完成! 总题数: {total_q} | 耗时: {elapsed:.0f}s")
    print(f"输出: {out_file}")


if __name__ == '__main__':
    test_mode = '--test' in sys.argv
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    workers = int(nums[0]) if nums else MAX_WORKERS
    if test_mode:
        files = collect_exam_files()
        print(f"找到 {len(files)} 份解析卷，测试第一份：{files[0][1].name}")
        name, qs, usage = process_one_file(files[0])
        print(f"提取 {len(qs)} 题")
        for q in qs[:3]:
            print(f"  T{q.get('difficulty','?')} [{q.get('q_num','?')}] "
                  f"{q.get('stem','')[:60]}...")
    else:
        run_full(max_workers=workers)
