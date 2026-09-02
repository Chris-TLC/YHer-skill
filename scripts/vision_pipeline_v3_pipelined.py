#!/usr/bin/env python3
"""
视觉PDF提取管道 v3.4 —— 两阶段流水线重叠版（生产者-消费者）

目标：在不改变任何 prompt / 打分 / 验证逻辑的前提下，把"文件间纯串行"
重构为两阶段重叠流水线，墙钟时间 ≈ max(Σ阿里, ΣDeepSeek) 而非两者相加。

架构:
  阶段A (Producer, 阿里视觉): 连续转录每份卷 → markdown 丢进队列 → 立刻转下一份
  阶段B (Consumer, DeepSeek文本): 从队列取卷 → 切题+化学验证+意图验证+评分 → 落盘

复用 vision_pipeline_v3.py 的全部函数：转录/切题/验证/打分/重试 一字不改。
所有数据文件与 v3 共用，断点续跑，追加写入，绝不删除/覆盖。

纪律（来自交接）:
  - 每题意图验证全量保留，不为提速跳过任何一道
  - 单份失败→标记继续；连续2份失败→停止报告不重启
  - 余额不足立即停
  - key 只从 .env 读，绝不进代码/日志
  - 不轮询、不死循环

用法:
  python3 scripts/vision_pipeline_v3_pipelined.py --resume
  python3 scripts/vision_pipeline_v3_pipelined.py --resume --max-files 3   # 小批冒烟测试
"""

import json, sys, time, os, argparse, threading, queue, traceback
from pathlib import Path
from typing import Optional, Dict, List, Tuple

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from dotenv import load_dotenv
load_dotenv(SKILL_DIR / ".env")

# ── 复用 v3 主脚本的全部逻辑（prompt/打分/验证一字不改）──
import scripts.vision_pipeline_v3 as v3
from scripts.vision_pipeline_v3 import (
    VisionClient, LLMClient, PaperResult,
    collect_files, doc_to_pdf, page_classify, render_page,
    transcribe_page, render_page_zoom, transcribe_charts, has_charts,
    structure_paper, validate_chem_batch, validate_intent, compute_confidence,
    api_retry,
    PAGE_IMG_DIR, FULL_MD_DIR, DOC_CACHE,
    RUN_CHART_ZOOM, RUN_VISUAL_VERIFY,
    VISION_PROVIDER, VISION_MODEL, LLM_MODEL,
)

# 数据文件与 v3 完全共用（绝不另起炉灶）
OUTPUT_DIR = v3.OUTPUT_DIR
PROGRESS_F = v3.PROGRESS_F
OUT_JSONL  = OUTPUT_DIR / "all_from_pdf_v3.jsonl"

# 并发配置
V_WORKERS = v3.V_WORKERS          # 阿里单卷内页并发（沿用 3）
QUEUE_MAX = 4                     # 转录完待切题的卷缓冲上限（防阿里跑太快堆内存）
MAX_CONSECUTIVE_FAIL = 2          # 连续失败阈值（交接纪律）

# 进度写入锁（生产者/消费者都要更新 progress）
_prog_lock = threading.Lock()


# ═══════════════════════════════════════════════════════
# 阶段A：阿里视觉转录（producer 侧，单卷）
# 等价于原 process_paper 的 Step0 ~ Phase1b，产出 full_md
# ═══════════════════════════════════════════════════════

def phase_vision(fp: Path, tag: str, vc: VisionClient) -> Tuple[Optional[PaperResult], Optional[str], Optional[Dict]]:
    """
    返回 (result, full_md, page_mds)
      - 成功: result 带 cost_vision/transcribed_pages, full_md 为拼接转录, page_mds 供切题定位
      - 失败(无有效页/全转录失败): result.errors 已填, full_md=None
    逻辑与 vision_pipeline_v3.process_paper 的视觉段一字不差。
    """
    result = PaperResult(source_file=fp.name, tag=tag)

    # Step 0: DOC/DOCX → PDF
    wf = fp
    if fp.suffix.lower() in ('.doc', '.docx'):
        pdf = doc_to_pdf(fp, DOC_CACHE)
        if not pdf:
            result.errors.append("DOC→PDF转换失败"); return result, None, None
        wf = pdf

    # Step 1: 页数 + 分类
    try:
        import fitz; doc = fitz.open(wf)
        total_pages = len(doc); doc.close()
    except Exception as e:
        result.errors.append(f"无法打开: {e}"); return result, None, None
    result.total_pages = total_pages

    page_tasks = []
    for pn in range(total_pages):
        pt = page_classify(wf, pn)
        if pt == 'ad':
            result.skipped_pages += 1; continue
        page_tasks.append((pn, pt))
    if not page_tasks:
        result.errors.append("无有效页面"); return result, None, None

    # Phase 1: 并行视觉转录（3Worker+3Key）
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t1 = time.time()
    print(f"  [阿里] {fp.name[:45]} 转录 {len(page_tasks)}页...", flush=True)
    page_mds = {}

    def _transcribe(pn, pt):
        img = render_page(wf, pn, PAGE_IMG_DIR)
        if not img:
            return pn, None, f"页{pn+1}渲染失败"
        try:
            mr = transcribe_page(img, vc, label=f"{fp.name} P{pn+1}")
            return pn, mr, None
        except Exception as e:
            return pn, None, f"页{pn+1}转录失败:{e}"

    with ThreadPoolExecutor(max_workers=V_WORKERS) as ex:
        futs = {ex.submit(_transcribe, pn, pt): pn for pn, pt in page_tasks}
        for fut in as_completed(futs):
            pn, mr, err = fut.result()
            if err:
                result.errors.append(err)
            elif mr:
                page_mds[pn] = mr["md"]
                result.cost_vision += mr["cost"]
                result.transcribed_pages += 1

    if not page_mds:
        result.errors.append("全部转录失败"); return result, None, None

    # Phase 1b: 图表聚焦增强（默认关）
    if RUN_CHART_ZOOM:
        chart_mds = {}
        for pn in sorted(page_mds.keys()):
            if not has_charts(page_mds[pn]):
                continue
            zoom_img = render_page_zoom(wf, pn, PAGE_IMG_DIR)
            if not zoom_img:
                continue
            try:
                cr = transcribe_charts(zoom_img, vc, label=f"{fp.name} P{pn+1} 图表聚焦")
                if cr.get("md"):
                    chart_mds[pn] = cr["md"]
                    result.cost_vision += cr.get("cost", 0)
            except Exception:
                pass
        for pn, chart_desc in chart_mds.items():
            if chart_desc:
                page_mds[pn] += f"\n\n【图表详细描述】\n{chart_desc}"

    # 拼接 + 保存转录（与 v3 同路径同格式）
    full_md = "\n\n".join(
        f"--- 第{pn+1}页 ---\n\n{page_mds[pn]}"
        for pn in sorted(page_mds.keys())
    )
    md_file = FULL_MD_DIR / f"{fp.stem[:60]}_transcript.md"
    md_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        md_file.write_text(full_md, encoding='utf-8')
    except Exception:
        pass

    print(f"  [阿里] {fp.name[:45]} ✓ {len(page_mds)}页 {time.time()-t1:.0f}s ¥{result.cost_vision:.3f}", flush=True)
    return result, full_md, page_mds


# ═══════════════════════════════════════════════════════
# 阶段B：DeepSeek 文本切题+验证+评分（consumer 侧，单卷）
# 等价于原 process_paper 的 Phase2 ~ Phase4，输入 full_md/page_mds
# ═══════════════════════════════════════════════════════

def phase_llm(fp: Path, tag: str, lc: LLMClient,
              result: PaperResult, full_md: str, page_mds: Dict) -> PaperResult:
    """逻辑与 vision_pipeline_v3.process_paper 的文本段一字不差。"""
    import re

    # Phase 2: 文本切题
    t2 = time.time()
    try:
        qs, sr = structure_paper(full_md, fp.name, lc)
        result.cost_llm += sr.get("cost_yuan", 0)
    except Exception as e:
        result.errors.append(f"切题失败:{e}"); return result
    if not qs:
        result.errors.append("未提取到题目"); return result

    # 题目来源标记 + 页码定位（与 v3 同）
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

    # Phase 3a: 视觉回查（默认关，两阶段下不在文本端做）
    all_v3a = []

    # Phase 3b: 化学正确性验证
    t3 = time.time()
    try:
        all_v3b = validate_chem_batch(qs, lc)
        result.cost_llm += len(qs) * 0.002
    except Exception as e:
        result.errors.append(f"化学验证失败:{e}"); all_v3b = []

    # Phase 3c: 意图保真验证（核心，绝不跳过）
    try:
        all_v3c = validate_intent(qs, full_md, lc)
        result.cost_llm += len(qs) * 0.002
    except Exception as e:
        result.errors.append(f"意图验证失败:{e}"); all_v3c = []
    print(f"  [DeepSeek] {fp.name[:42]} 切{len(qs)}题+验证 {time.time()-t2:.0f}s", flush=True)

    # Phase 4: 推导式评分（与 v3 完全一致）
    final_qs = []
    v3a_map = {v.get("q_num",""): v for v in all_v3a}
    v3b_map = {v.get("q_num",""): v for v in all_v3b}
    v3c_map = {v.get("q_num",""): v for v in all_v3c}

    verified_count = 0
    for q in qs:
        qn = q.get("q_num", "")
        v3a = v3a_map.get(qn, {}); v3b = v3b_map.get(qn, {}); v3c = v3c_map.get(qn, {})
        conf, reasons = compute_confidence(q, v3a, v3b, v3c)
        q["confidence"] = conf
        q["confidence_reasons"] = reasons
        q["verification_coverage"] = {
            "vision_verified": bool(v3a),
            "chemistry_verified": bool(v3b),
            "intent_verified": bool(v3c),
        }
        if v3a.get("corrections", {}).get("stem"):
            q["stem"] = v3a["corrections"]["stem"]
        if v3a.get("corrections", {}).get("answer"):
            q["answer"] = v3a["corrections"]["answer"]
        all_issues = (v3a.get("issues",[]) + v3b.get("issues",[]) + v3c.get("issues",[]))
        if all_issues:
            q["_issues"] = all_issues
        q["verification_status"] = (
            "passed" if conf >= 0.85 else
            "needs_review" if conf >= 0.60 else
            "rejected"
        )
        if any([v3a, v3b, v3c]):
            verified_count += 1
        if conf >= 0.60:
            final_qs.append(q)

    result.questions = final_qs
    passed = sum(1 for q in final_qs if q["verification_status"] == "passed")
    need_r = sum(1 for q in final_qs if q["verification_status"] == "needs_review")
    rejected = sum(1 for q in final_qs if q["verification_status"] == "rejected")
    verif_cov = verified_count / max(len(qs), 1)
    result.stats = {
        "total_extracted": len(qs),
        "passed_verification": passed,
        "needs_review": need_r,
        "rejected": rejected,
        "passed_rate": round(passed/max(len(qs),1), 3),
        "verification_coverage": round(verif_cov, 3),
        "avg_confidence": round(sum(q.get("confidence",0) for q in final_qs)/max(len(final_qs),1), 3)
            if final_qs else 0,
        "questions_with_intent_issues": sum(1 for q in final_qs
            if any("意图" in r for r in q.get("confidence_reasons",[]))),
    }
    return result


# ═══════════════════════════════════════════════════════
# 进度 I/O（复用 v3 格式，加锁，绝不覆盖已有）
# ═══════════════════════════════════════════════════════

def load_prog() -> Dict:
    return v3._load_prog()

def save_prog_for_paper(fp: Path, tag: str, res: PaperResult):
    """单份完成后更新 progress.json（线程安全，沿用 v3 看板字段）"""
    with _prog_lock:
        prog = v3._load_prog()
        done_files = set(prog.get("done_files", []))
        done_files.add(fp.name)
        s = res.stats
        agg = prog.setdefault("aggregated",
            {"passed":0,"needs_review":0,"rejected":0,"intent_issues":0,"avg_confidence_sum":0.0,"count":0})
        agg["passed"] += s.get("passed_verification", 0)
        agg["needs_review"] += s.get("needs_review", 0)
        agg["rejected"] += s.get("rejected", 0)
        agg["intent_issues"] += s.get("questions_with_intent_issues", 0)
        agg["avg_confidence_sum"] += s.get("avg_confidence", 0)
        agg["count"] += 1
        recents = prog.setdefault("recent_results", [])
        recents.insert(0, {
            "file": fp.name[:60], "tag": tag,
            "questions": len(res.questions),
            "passed": s.get("passed_verification", 0),
            "needs_review": s.get("needs_review", 0),
            "rejected": s.get("rejected", 0),
            "passed_rate": s.get("passed_rate", 0),
            "avg_confidence": s.get("avg_confidence", 0),
            "verification_coverage": s.get("verification_coverage", 0),
            "intent_issues": s.get("questions_with_intent_issues", 0),
            "errors": res.errors[:3],
            "cost_v": round(res.cost_vision, 4),
            "cost_l": round(res.cost_llm, 4),
            "time": time.strftime("%H:%M:%S"),
            "pages": res.total_pages,
            "transcribed": res.transcribed_pages,
        })
        if len(recents) > 20:
            recents.pop()
        prog["done_files"] = list(done_files)
        prog["total_qs"] = prog.get("total_qs", 0) + len(res.questions)
        prog["cost_v"] = round(prog.get("cost_v", 0.0) + res.cost_vision, 2)
        prog["cost_l"] = round(prog.get("cost_l", 0.0) + res.cost_llm, 2)
        prog["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        v3._save_prog(prog)


# ═══════════════════════════════════════════════════════
# 主流程：两阶段重叠
# ═══════════════════════════════════════════════════════

# 余额不足 / 鉴权类错误关键词 → 立即停（交接纪律）
FATAL_KEYWORDS = ['余额', 'insufficient', 'balance', 'Insufficient Balance',
                  'quota', 'arrearage', '欠费', '401', 'invalid_api_key',
                  'Unauthorized', 'authentication']

class FatalStop(Exception):
    pass

def _is_fatal(err: str) -> bool:
    return any(k.lower() in (err or "").lower() for k in FATAL_KEYWORDS)


def run_pipelined(max_files: Optional[int] = None):
    # ── key 从 .env 读，3key 并行 ──
    vk = os.environ.get("DASHSCOPE_API_KEY", "")
    extra = [k.strip() for k in os.environ.get("DASHSCOPE_API_KEY_EXTRA", "").split(",") if k.strip()]
    ALL_VISION_KEYS = ([vk] if vk else []) + extra
    dk = os.environ.get("DEEPSEEK_API_KEY", "")
    if not vk:
        print("❌ 缺 DASHSCOPE_API_KEY"); sys.exit(1)
    if not dk:
        print("❌ 缺 DEEPSEEK_API_KEY"); sys.exit(1)

    vc = VisionClient(provider=VISION_PROVIDER, model=VISION_MODEL, api_keys=ALL_VISION_KEYS)
    lc = LLMClient(provider='deepseek', model=LLM_MODEL, api_key=dk)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    FULL_MD_DIR.mkdir(parents=True, exist_ok=True)

    all_files = collect_files()
    total_count = len(all_files)
    prog = v3._load_prog()
    prog["total_files"] = total_count
    if not prog.get("started_at"):
        prog["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        prog["started_at_ts"] = time.time()
    v3._save_prog(prog)

    done_files = set(prog.get("done_files", []))
    pending = [(t, f) for t, f in all_files if f.name not in done_files]
    if max_files:
        pending = pending[:max_files]

    print(f"\n{'='*68}")
    print(f"视觉提取管道 v3.4 — 两阶段流水线重叠")
    print(f"{'='*68}")
    print(f"视觉: {VISION_MODEL} ({len(ALL_VISION_KEYS)}key×{V_WORKERS}worker)")
    print(f"文本: {LLM_MODEL} (流水线消费)")
    print(f"已完成: {len(done_files)}/{total_count} | 待处理: {len(pending)}")
    print(f"队列缓冲: {QUEUE_MAX} | 数据: 追加写入,断点续跑")
    print(f"{'='*68}\n")
    if not pending:
        print("✅ 全部完成"); return

    # ── 两阶段：阿里 producer 线程 + DeepSeek 主线程 consumer ──
    work_q: "queue.Queue" = queue.Queue(maxsize=QUEUE_MAX)
    SENTINEL = object()
    producer_state = {"fatal": None}

    def producer():
        """阿里：连续转录，丢队列。"""
        for tag, fp in pending:
            try:
                res, full_md, page_mds = phase_vision(fp, tag, vc)
            except Exception as e:
                err = f"视觉异常:{e}"
                if _is_fatal(err):
                    producer_state["fatal"] = err
                    work_q.put(SENTINEL); return
                # 非致命：包成失败结果传下去，让 consumer 统一记账
                res = PaperResult(source_file=fp.name, tag=tag)
                res.errors.append(err); full_md, page_mds = None, None
            # 视觉端致命错误（余额/鉴权）→ 立即停
            if res and res.errors:
                fatal_hit = next((e for e in res.errors if _is_fatal(e)), None)
                if fatal_hit:
                    producer_state["fatal"] = fatal_hit
                    work_q.put((tag, fp, res, full_md, page_mds))  # 这份仍记一笔失败
                    work_q.put(SENTINEL); return
            work_q.put((tag, fp, res, full_md, page_mds))
        work_q.put(SENTINEL)

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    # consumer：DeepSeek 切题+验证
    done = 0
    consecutive_fail = 0
    fatal_stop = None
    n_pending = len(pending)
    t0 = time.time()

    with open(OUT_JSONL, 'a', encoding='utf-8') as fo:
        while True:
            item = work_q.get()
            if item is SENTINEL:
                break
            tag, fp, res, full_md, page_mds = item
            done += 1
            ts = time.strftime("%H:%M:%S")

            # 视觉段已失败（无 full_md）→ 直接记一笔失败，标记继续
            if full_md is None or not page_mds:
                err0 = res.errors[0] if res.errors else "视觉段失败"
                print(f"{ts} [{done}/{n_pending}] ❌ {fp.name[:50]} — {err0[:60]}", flush=True)
                save_prog_for_paper(fp, tag, res)  # done_files 标记，避免重复
                # 致命？
                if producer_state["fatal"] or _is_fatal(err0):
                    fatal_stop = producer_state["fatal"] or err0
                    break
                # "无有效页面" / "DOC转换失败" 属预期失败，不计入连续失败熔断
                if any(k in err0 for k in ("无有效页面", "DOC→PDF", "无法打开")):
                    consecutive_fail = 0
                else:
                    consecutive_fail += 1
                if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                    print(f"\n⛔ 连续{consecutive_fail}份失败，按纪律停止报告（不自动重启）", flush=True)
                    break
                continue

            # 文本段
            try:
                res = phase_llm(fp, tag, lc, res, full_md, page_mds)
            except Exception as e:
                res.errors.append(f"文本段异常:{e}")

            # 落盘题目（追加）
            for q in res.questions:
                fo.write(json.dumps(q, ensure_ascii=False) + '\n')
            fo.flush()

            s = res.stats
            n_q = len(res.questions)
            pr = s.get("passed_rate", 0)
            icon = "✅" if pr >= 0.90 else ("⚠️" if pr >= 0.75 else "❌")
            print(f"{ts} [{done}/{n_pending}] {icon} {fp.name[:48]} "
                  f"{n_q}题 通过率{pr:.0%} 意图偏差{s.get('questions_with_intent_issues',0)} "
                  f"¥{res.cost_vision+res.cost_llm:.3f}", flush=True)
            if res.errors:
                for e in res.errors[:2]:
                    print(f"      ⚠️ {e[:80]}")

            save_prog_for_paper(fp, tag, res)

            # 连续失败熔断：0题且有非预期错误才算失败
            llm_fatal = next((e for e in res.errors if _is_fatal(e)), None)
            if llm_fatal:
                fatal_stop = llm_fatal; break
            if n_q == 0 and res.errors and not any(
                k in (res.errors[0]) for k in ("答案页", "无有效", "DOC→PDF")):
                consecutive_fail += 1
            else:
                consecutive_fail = 0
            if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                print(f"\n⛔ 连续{consecutive_fail}份失败，按纪律停止报告（不自动重启）", flush=True)
                break

            # producer 端致命
            if producer_state["fatal"]:
                fatal_stop = producer_state["fatal"]
                # 把队列里剩余的排空成"未处理"，不强行跑
                break

    elapsed = time.time() - t0
    final_prog = v3._load_prog()
    print(f"\n{'='*68}")
    if fatal_stop:
        print(f"⛔ 因致命错误停止: {fatal_stop[:100]}")
        print(f"   已安全保存进度，可修复后 --resume 断点续跑。")
    print(f"本轮处理 {done} 份 | 累计完成 {len(final_prog.get('done_files',[]))}/{total_count}")
    print(f"耗时: {elapsed:.0f}s ({elapsed/3600:.2f}h)")
    print(f"累计成本: 视觉¥{final_prog.get('cost_v',0):.2f} 文本¥{final_prog.get('cost_l',0):.2f} "
          f"合计¥{final_prog.get('cost_v',0)+final_prog.get('cost_l',0):.2f}")
    print(f"累计题数: {final_prog.get('total_qs',0)}")
    print(f"{'='*68}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="断点续跑（跳过已完成）")
    ap.add_argument("--max-files", type=int, default=None, help="本轮最多处理N份（冒烟测试用）")
    args = ap.parse_args()
    # resume 是默认且唯一安全模式；这里始终按 resume 语义跑（跳过 done_files）
    run_pipelined(max_files=args.max_files)
