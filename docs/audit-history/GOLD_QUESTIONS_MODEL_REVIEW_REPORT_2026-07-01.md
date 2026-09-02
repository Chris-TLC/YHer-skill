# Gold Questions Model Review Report

更新时间：2026-07-01 CST

## 结论

本轮完成三类硬空洞 gold question candidates 的本地路由、Anthropic Opus 4.8 重新生成、GPT-5.5 审核与保守子集输出。

关键结论：

- DeepSeek 修正版 36 题已经通过本地 router：36 approved / 0 revise / 0 reject。
- Anthropic `claude-opus-4-8` 重新生成版也通过本地 router：36 approved / 0 revise / 0 reject。
- 但 GPT-5.5 学科审核显示两版都不能直接当生产 gold：
  - DeepSeek 版：10 approve / 19 revise / 7 reject。
  - Opus 4.8 版：7 approve / 15 revise / 14 reject。
- 因此不要删除 DeepSeek 原始三文件；Opus 版保留为实验产物和返修素材。
- 当前最保守可继续推进的子集是 DeepSeek 版中 GPT-5.5 审核 `approve/none` 的 10 题：
  - `yihuier-chemistry-skill/data/quality/gold_question_candidates/approved_gpt55_clean.jsonl`

## API 连通性

`.env` 中已配置非密钥 base URL：

- `OPENAI_BASE_URL=https://api.ooapi.cc`
- `ANTHROPIC_BASE_URL=https://api.ooapi.cc`

密钥值不记录在本文档中。

连通性结果：

- OpenAI-compatible `/v1/models` 可用，返回 `gpt-5.4`、`gpt-5.4-mini`、`gpt-5.5`。
- Anthropic-compatible `/v1/models` 可用，包含 `claude-opus-4-8`。
- Anthropic `claude-opus-4-8` 文本调用可用。
- OpenAI-compatible `gpt-5.5` 文本调用可用。
- `gpt-5.5` 视觉单题调用仍出现 `ConnectionResetError`，暂不能作为稳定视觉复核路径。

## DeepSeek 修正版

输入文件：

- `solution_charge_gold_candidates.jsonl`
- `process_flow_gold_candidates.jsonl`
- `integrated_experiment_gold_questions.jsonl`

本地 router 输出：

- `yihuier-chemistry-skill/data/quality/gold_question_candidates/approved.jsonl`
- `yihuier-chemistry-skill/data/quality/gold_question_candidates/revise.jsonl`
- `yihuier-chemistry-skill/data/quality/gold_question_candidates/reject.jsonl`
- `yihuier-chemistry-skill/data/quality/gold_question_candidates/summary.json`

Router summary：

```json
{
  "total": 36,
  "approved": 36,
  "revise": 0,
  "reject": 0
}
```

GPT-5.5 审核输出：

- `yihuier-chemistry-skill/data/quality/gold_question_candidates/deepseek_gpt55_review.jsonl`
- `yihuier-chemistry-skill/data/quality/gold_question_candidates/deepseek_gpt55_review_summary.json`

GPT-5.5 summary：

```json
{
  "total": 36,
  "decisions": {
    "approve": 10,
    "revise": 19,
    "reject": 7
  },
  "severities": {
    "none": 10,
    "minor": 15,
    "major": 4,
    "critical": 7
  }
}
```

保守可用子集：

- `yihuier-chemistry-skill/data/quality/gold_question_candidates/approved_gpt55_clean.jsonl`
- 10 行，分布为：solution_three_balances 6，process_flow 2，integrated_experiment 2。

## Opus 4.8 重新生成版

生成脚本：

- `yihuier-chemistry-skill/scripts/generate_gold_questions_anthropic.py`

生成输出：

- `yihuier-chemistry-skill/data/quality/gold_question_candidates/anthropic_opus48/combined.jsonl`
- `yihuier-chemistry-skill/data/quality/gold_question_candidates/anthropic_opus48/generation_summary.json`

Router 输出：

- `yihuier-chemistry-skill/data/quality/gold_question_candidates/anthropic_opus48_routed/summary.json`

Router summary：

```json
{
  "total": 36,
  "approved": 36,
  "revise": 0,
  "reject": 0
}
```

GPT-5.5 审核输出：

- `yihuier-chemistry-skill/data/quality/gold_question_candidates/anthropic_opus48_gpt55_review.jsonl`
- `yihuier-chemistry-skill/data/quality/gold_question_candidates/anthropic_opus48_gpt55_review_summary.json`

GPT-5.5 summary：

```json
{
  "total": 36,
  "decisions": {
    "approve": 7,
    "revise": 15,
    "reject": 14
  },
  "severities": {
    "none": 7,
    "minor": 12,
    "critical": 14,
    "major": 3
  }
}
```

主要问题类型：

- 单选题多答案。
- 题干物质写错，例如 NaHCO3 / Na2CO3 混用。
- 标准答案与题干数据自相矛盾。
- 实验流程缺关键氧化/除杂步骤。
- 计量数据导致质量分数超过 100%。
- rubric 把错误化学解释设为 must_have。

结论：Opus 4.8 生成能力可用，但不能替代审核；本轮 Opus 版整体低于 DeepSeek 修正版。

## 新增/更新脚本

- `scripts/generate_gold_questions_anthropic.py`
  - 读取 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_BASE_URL`。
  - 默认模型 `claude-opus-4-8`。
  - 逐题/小批生成，自动重试坏 JSON 或连接重置。
  - 输出仍为 `silver_candidate`，不直接进入生产 gold。

- `scripts/review_gold_questions_openai.py`
  - 读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。
  - 默认模型 `gpt-5.5`。
  - 只做审核，不直接修改题。
  - 输出 `approve/revise/reject`、严重等级、原因、建议修复。

- `scripts/evaluate_visual_understanding.py`
  - 支持 `OPENAI_BASE_URL` / `OPENAI_CHAT_COMPLETIONS_URL` / `OPENAI_API_BASE`。
  - 支持中转站 OpenAI-compatible 调用。

## 验证命令

已通过：

```bash
python3 tests/test_visual_understanding_eval.py
python3 tests/test_gold_question_routing.py
python3 -m py_compile scripts/generate_gold_questions_anthropic.py scripts/review_gold_questions_openai.py scripts/evaluate_visual_understanding.py scripts/route_gold_question_candidates.py
python3 scripts/route_gold_question_candidates.py --input data/quality/gold_question_candidates/approved_gpt55_clean.jsonl
```

关键输出：

- visual understanding tests：9/9 通过。
- gold routing tests：3/3 通过。
- `approved_gpt55_clean.jsonl` router dry-run：10 approved / 0 revise / 0 reject。

## 下一步

1. 不删 DeepSeek 三个原始 JSONL 文件。
2. 不把 `approved.jsonl` 的 36 题直接当 gold。
3. 以 `approved_gpt55_clean.jsonl` 的 10 题作为当前最保守候选。
4. 对 `deepseek_gpt55_review.jsonl` 中 19 条 revise 和 7 条 reject 按原因返修。
5. 返修后重新跑：
   - router。
   - GPT-5.5 review。
   - 人工抽查重点：三大守恒、工艺流程、实验综合中所有 major/critical。
6. 只有 router approved 且 GPT-5.5 approve/none，才可进入下一层人工验收或标为更高等级候选。
