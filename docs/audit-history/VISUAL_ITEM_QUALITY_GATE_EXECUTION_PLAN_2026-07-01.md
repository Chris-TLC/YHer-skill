# YHer 视觉题资产与质量门执行计划

更新时间：2026-07-01 CST

## 0. 本文件用途

下一轮对话进入执行验证阶段时，先读：

- `PROJECT_HANDOFF/YHER_PROJECT_OVERVIEW.md`
- 本文件
- `PROJECT_HANDOFF/CURRENT_DECISIONS.md`
- `PROJECT_HANDOFF/HISTORICAL_DETOURS.md`
- `PROJECT_HANDOFF/INCIDENT_2026-06-28_AGENT_LOOP.md`
- `PROJECT_HANDOFF/YHER_FULL_PROJECT_AUDIT_2026-07-01.md`

目标不是继续讨论方向，而是把图片题和复杂题的质量问题落成可运行的验证与质量门。

## 1. 绝对交付标准

### 1.1 学生端 100% 可读可理解的含义

这里的 100% 不是说 6438 道历史题全部天然完美，而是：

> 任何一题只要展示给学生，就必须 100% 可读、可作答、题图一致、无缺失信息；达不到就不展示。

如果题目缺图、题图匹配不确定、裁片不清、选项缺失、公式乱码、答案或解析疑似错，就必须被 `quality_gate` 拦下，进入 `quarantined` 或 `manual_review`，不能进入诊断、画像或学生主流程。

### 1.2 大模型内化原数据合格的含义

大模型不能被当作唯一真相。合格的“内化原数据”必须同时有：

- 原始来源：`source_file`、`source_path`、`page`、`page_image_path`、`render_hash`
- 视觉证据：题目级裁片或页级裁片、bbox、可见锚点
- 结构化转录：题干、选项、图像描述、单位、变量、表格/曲线/装置/结构信息
- 答案校验：标准答案、可解释的解题链路、rubric 或评分点
- 质量等级：`strong` / `weak` / `reject`
- 使用权限：是否允许 `diagnosis`、`practice`、`teaching`、`profile_evidence`

只有 `strong` 且通过答案/图像/可读性校验的题，才能更新学生画像。`weak` 最多用于练习或人工复核，不得写入高权重画像。`reject` 一律隔离。

### 1.3 “完美质量”的工程定义

模型识图不可能数学意义上永不出错。可达到的产品级完美是：

1. 对学生：不合格题永远不出现。
2. 对画像：不合格证据永远不入库。
3. 对开发：每个被放行的 item 都能追溯到原图、裁片、结构化文本、标准答案和闸门记录。
4. 对失败：系统宁可拒绝、降级、人工复核，也不让错误题污染学生信任。

## 2. 当前证据

最近小批量试点结果：

- 图依赖题总量：1848 / 6438。
- 抽样 56 道图依赖题：
  - source/page/transcript/page image 都能回连：56/56。
  - declared page 强匹配：38/56，弱匹配：18/56。
  - best text page 强匹配：51/56，弱匹配：5/56。
  - best page 与 declared page 不一致：17/56。
- 多模态理解试点 6 道：
  - visible_pass：4/6。
  - answer_match：2/6。
  - understanding_pass：2/6。
  - 实验装置、晶胞、流程题出现高置信错误或标准答案疑似问题。

结论：

- 题图回连有基础，但不能直接信任 declared page。
- 页级图像可用于学生展示，但题目级裁片和锚点还没产品化。
- 当前多模态模型可以辅助转录和描述，不能单独承担诊断画像。
- gold diagnostic 不能取消，只能缩小为校准/eval 和高风险样本标尺。

证据文件：

- `/tmp/yher_visual_matching_pilot2.json`
- `/tmp/yher_visual_asset_manifest_check.json`
- `/tmp/yher_multimodal_pilot.json`
- `/tmp/yher_vision_provider_probe.json`

## 3. 四类验证的改进方案

### 3.1 验证一：题图匹配与视觉资产链路

要解决的问题：

- “如图”题到底对应哪张图。
- item 里的 declared page 是否可信。
- 题目裁片是否包含完整题干、图、选项和必要表格/曲线。

执行步骤：

1. 新增只读 manifest 生成器：
   - 建议文件：`yihuier-chemistry-skill/scripts/build_visual_asset_manifest.py`
   - 输入：`data/from_pdf/all_from_pdf_v3.jsonl`、`data/page_images_v3/`、`data/item_bank/chemistry_v3_6695.jsonl`
   - 输出：`data/quality/visual_asset_manifest.jsonl`

2. 每道图依赖题生成证据链：
   - `item_id`
   - `source_file`
   - `source_path`
   - `declared_page`
   - `best_text_page`
   - `page_image_path`
   - `page_image_hash`
   - `declared_match_score`
   - `best_match_score`
   - `visible_anchors`
   - `match_tier`
   - `blocker_reasons`

3. 匹配规则：
   - 先用 `_source_file + _page` 找 declared page。
   - 再用题干关键词、题号、选项、答案锚点搜索 transcript，得到 best text page。
   - 如果 declared page 与 best text page 不一致，不直接报错，先标记 `page_mismatch`，转入人工/视觉复核。
   - 不允许只凭模型一句“应该是这一页”放行。

硬性验收：

- 所有进入学生端候选集的图依赖题，必须有存在的 `page_image_path` 和稳定 `page_image_hash`。
- 所有 `match_tier=strong` 的题，至少满足：
  - source 唯一解析。
  - page image 存在。
  - 题干关键词或选项锚点可在 transcript 中命中。
  - declared page 与 best page 一致，或有人工确认记录。
- `match_tier=weak` 不得进入画像诊断，只能进入人工复核或低风险练习候选。
- `match_tier=reject` 不得被任何学生流程读取。

### 3.2 验证二：学生展示可读性

要解决的问题：

- 学生看到的是不是完整题。
- 图、表、曲线、装置、结构式是否清晰。
- 前端是否会因为长题、公式、选项、移动端宽度导致看不懂。

执行步骤：

1. 生成题目级裁片：
   - 优先用 transcript 题号和页面图定位。
   - 不能可靠定位时，用页级裁片展示，但必须标注 `crop_tier=page_only`。
   - 裁片不得只截到题干或只截到图。

2. 建立展示预览页或静态验收 HTML：
   - 每道候选题展示：结构化题干 + 原始裁片 + source/page + gate 状态。
   - 不允许只看 JSON 判定可读。

3. 用 Playwright 做浏览器检查：
   - 桌面视口。
   - 手机视口。
   - 长题、图题、选项题、流程题、晶胞题、实验装置题。

硬性验收：

- 学生端候选题 100% 无以下问题：
  - “如图”但没有图。
  - 图片裁片缺关键图或关键数据。
  - 选项缺失或黏在一行难读。
  - 公式、上下标、离子、电荷严重乱码。
  - 题干/选项/图互相矛盾。
  - 前端移动端遮挡、溢出、按钮覆盖题目。
- 任一题出现上述问题，处理不是“容忍”，而是：
  - 修裁片/结构化文本；或
  - 标记 `manual_review`；或
  - 标记 `reject` 并禁止展示。

### 3.3 验证三：大模型视觉理解与内化

要解决的问题：

- 模型能不能看懂图。
- 模型是否把题干、图、选项、数据和标准答案正确绑定。
- 模型是否会高置信错读，并污染推荐/画像。

执行步骤：

1. 为每道视觉候选题生成 `vision_transcript`：
   - 题目讲什么。
   - 图中有哪些对象、变量、数据、箭头、坐标、装置、物质。
   - 哪些信息来自图，哪些来自文字。
   - 哪些信息不确定。

2. 要求模型输出结构化 JSON，而不是散文：
   - `question_restatement`
   - `visual_elements`
   - `data_points`
   - `missing_or_uncertain`
   - `answerability`
   - `solution_outline`
   - `final_answer`
   - `confidence`
   - `evidence_citations`

3. 做答案与标准解交叉校验：
   - final answer 与题库答案一致。
   - solution outline 覆盖关键 rubric。
   - 图中关键数据没有漏读或错读。
   - 对不确定图像必须降置信，而不是硬答。

4. 对高风险题型建立小 gold/eval：
   - 晶胞。
   - 实验装置。
   - 工艺流程。
   - 曲线/图像分析。
   - 有机结构式。
   - 滴定/守恒综合。

硬性验收：

- 多模态理解不能以“模型答对几道”作为唯一标准。
- 放行到画像的视觉题必须满足：
  - 视觉转录没有关键事实错误。
  - 答案与标准解一致，或标准答案被人工确认修正。
  - 关键推理链路能引用图中具体证据。
  - 没有 `missing_or_uncertain` 里列出的关键缺口。
- 模型高置信但答案错，必须被记录为 P0 failure mode，并调低该类题型自动放行权限。

### 3.4 验证四：中心质量门

要解决的问题：

- 低质量题不能靠前端或人工记忆兜底。
- `ItemRepository` 当前会加载所有 item，缺中心过滤。
- 诊断、练习、教学、画像必须按不同风险级别读题。

执行步骤：

1. 新增质量 manifest：
   - 建议文件：`data/quality/item_quality_manifest.jsonl`
   - 每行一个 item 的闸门状态。

2. 新增质量评估模块：
   - 建议文件：`core/data/item_quality.py`
   - 负责读取 manifest、判断用途、返回 blocker。

3. 修改 `core/data/item_repository.py`：
   - 默认只返回可用于当前用途的 item。
   - `find_items(..., purpose="diagnosis")`
   - `purpose` 可选：`diagnosis`、`practice`、`teaching`、`profile_evidence`、`debug_all`
   - `debug_all` 只允许脚本和测试使用，学生/API 默认不能用。

4. API 和前端联动：
   - 诊断题接口只返回 `usable_for_diagnosis=true`。
   - 画像更新只吃 `usable_for_profile_evidence=true`。
   - 前端展示原始裁片和结构化题干，缺任一项则显示“暂不可用”，不进入答题。

硬性验收：

- 没有 manifest 的题，默认不允许进入学生诊断。
- 有 `needs_image=true` 但没有合格 image asset 的题，禁止展示。
- `match_tier != strong` 的图依赖题，禁止写画像。
- `llm_understanding_tier != strong` 的视觉题，禁止写画像。
- 任何 `blocker_reasons` 非空的题，除非 blocker 被明确列入允许列表，否则禁止进入学生端。

## 4. 建议数据结构

### 4.1 visual_asset_manifest.jsonl

```json
{
  "item_id": "string",
  "source_file": "string",
  "source_path": "string",
  "declared_page": 1,
  "best_text_page": 1,
  "page_image_path": "string",
  "page_image_hash": "sha256:string",
  "crop_path": "string|null",
  "crop_hash": "sha256:string|null",
  "visible_anchors": ["题号", "选项A", "关键图示词"],
  "declared_match_score": 0.0,
  "best_match_score": 0.0,
  "match_tier": "strong|weak|reject",
  "crop_tier": "item_crop|page_crop|page_only|missing",
  "blocker_reasons": []
}
```

### 4.2 item_quality_manifest.jsonl

```json
{
  "item_id": "string",
  "needs_image": true,
  "visual_asset_status": "strong|weak|reject|not_required",
  "readability_status": "pass|manual_review|reject",
  "llm_understanding_status": "strong|weak|reject|not_required",
  "answer_status": "verified|suspect|missing",
  "rubric_status": "complete|partial|missing",
  "usable_for_diagnosis": false,
  "usable_for_practice": true,
  "usable_for_teaching": true,
  "usable_for_profile_evidence": false,
  "blocker_reasons": ["page_mismatch", "llm_answer_mismatch"],
  "reviewer": "script|human|model_crosscheck",
  "updated_at": "2026-07-01T00:00:00+08:00"
}
```

## 5. 执行顺序

### P0：只读复核上下文

必须先读本文件和总览。确认：

- 不重跑旧视觉管道。
- 不清理工作树。
- 不写 API key。
- 不高频轮询。
- 本轮只构建 manifest、质量门、试点报告和必要测试。

### P1：实现视觉资产 manifest 生成器

交付物：

- `scripts/build_visual_asset_manifest.py`
- `data/quality/visual_asset_manifest.jsonl`
- `data/quality/visual_asset_manifest_summary.json`

验收：

- 脚本可重复运行。
- 默认 dry-run 或显式 `--write`。
- 不覆盖源数据。
- 输出统计总量、强/弱/reject、缺图、page mismatch、按题型分类。

### P2：实现 item quality manifest 与中心质量门

交付物：

- `core/data/item_quality.py`
- `data/quality/item_quality_manifest.jsonl`
- `data/quality/item_quality_summary.json`
- `core/data/item_repository.py` 支持 `purpose`

验收：

- 没有 manifest 的题默认不能进入诊断。
- 现有纯文本合格题不能被误伤到 demo 完全不可用。
- 图依赖弱匹配题不能进入画像。
- 单元测试覆盖 allow/block。

### P3：构建 20-50 道视觉评测集

交付物：

- `data/evals/visual_item_eval_set.jsonl`
- 覆盖：晶胞、实验装置、工艺流程、曲线图、有机结构、电化学装置、其他图依赖。

验收：

- 每类至少 3 道，优先从已有 56 样本扩展。
- 每道有原图、结构化题干、标准答案、人工/脚本可查证字段。
- 不能只选模型容易做的题。

### P4：多模态理解评估脚本

交付物：

- `scripts/evaluate_visual_understanding.py`
- `data/evals/visual_understanding_results.jsonl`
- `data/evals/visual_understanding_summary.json`

验收：

- 输出结构化 JSON。
- 明确记录模型名称、输入图、输入文本、答案、置信度、错误类型。
- 模型答错或不确定时不会被算作可画像证据。
- 对高置信错误单独统计。

### P5：学生端展示 smoke

交付物：

- 浏览器截图或 Playwright 报告。
- 至少覆盖桌面和手机视口。
- 至少覆盖 10 道视觉题预览，包括复杂类别。

验收：

- 题干、选项、裁片、来源、不可用状态都可见。
- 长题不溢出。
- 图像不缺失。
- 质量门拦下的题不会进入正式答题流程。

### P6：最终报告

交付物：

- `PROJECT_HANDOFF/VISUAL_ITEM_QUALITY_GATE_EXECUTION_REPORT_YYYY-MM-DD.md`

报告必须包含：

- 总题量与图依赖题量。
- manifest 覆盖率。
- strong/weak/reject 数量。
- page mismatch 数量。
- 学生端可展示候选数量。
- 画像可用证据数量。
- 多模态理解通过率和高置信错误。
- 被拦截样例。
- 下一步人工 gold 清单。

## 6. 测试要求

最低测试：

- `python3 tests/test_tutor_engine.py`
- 新增 `tests/test_item_quality.py`
- 新增 `tests/test_visual_asset_manifest.py`
- API smoke：`/health`、诊断取题、profile。
- 前端 smoke：首页、诊断至少两题、视觉题预览或拦截状态。

禁止说“已完成”，除非：

- 测试命令跑过并记录结果。
- 生成文件路径明确。
- 抽样截图或报告路径明确。
- 失败项被列出，不被掩盖。

## 7. Gold diagnostic 的边界

LLM 可以参与 gold diagnostic 流程，但不能单独完成 gold。

可接受流程：

1. LLM 从真实题或知识点生成候选诊断题。
2. 另一个模型或同模型不同 prompt 做挑刺。
3. 脚本做格式、答案、无图依赖、rubric 完整性检查。
4. 高风险题型必须有人审或用真实标准答案/权威解析锁定。
5. 通过后才标 `gold`；否则最多标 `silver_candidate`。

原因：

- gold 的作用是评测模型，不能让被评测对象自己当唯一裁判。
- 对学生画像来说，错的 gold 比没有 gold 更危险。
- LLM 可以把人工成本从“从零写”降到“审候选”，但不能取消校准集。

## 8. 下一轮对话可直接使用的 prompt

```text
[$superpowers:using-superpowers] [$autoplan] [$yher-project-memory]

按 YHer 项目记忆规范开始。先读：
1. /Users/mac/Desktop/项目文件夹/Tools/PROJECT_HANDOFF/YHER_PROJECT_OVERVIEW.md
2. /Users/mac/Desktop/项目文件夹/Tools/PROJECT_HANDOFF/VISUAL_ITEM_QUALITY_GATE_EXECUTION_PLAN_2026-07-01.md
3. 总览 Must-Read 中与本任务相关的文件

本轮进入执行验证阶段，目标是实现并验证视觉题资产 manifest、学生端可读性验证、多模态理解评估、中心质量门。质量优先，宁可拦截，不许错题展示给学生或污染画像。

执行要求：
- 不重跑旧 v3 视觉管道。
- 不清理、不 reset 工作树。
- 不写入或展示 API key。
- 不高频轮询。
- 使用 apply_patch 做代码/文档编辑。
- 先实现只读/可重复脚本，再接中心质量门。
- 所有输出写到 data/quality、data/evals 或 PROJECT_HANDOFF，不覆盖原始题库。

硬性验收：
- 展示给学生的题目必须 100% 可读可理解；达不到就被 quality gate 拦截。
- 大模型内化用的数据必须有 source/page/image/crop/transcript/answer/rubric/quality tier。
- 图依赖题没有合格视觉资产，不得进入 diagnosis/profile_evidence。
- match_tier 或 llm_understanding_tier 不是 strong 的图题，不得更新画像。
- 最后必须运行相关测试和 demo smoke，并写执行报告。
- 收尾必须按 yher-project-memory 追加 PROJECT_HANDOFF/YHER_PROJECT_OVERVIEW.md ledger。

请从 P1 开始：实现 visual_asset_manifest 生成器和摘要；然后实现 item_quality gate；最后跑 20-50 道小批量验证和当前 demo smoke。
```

## 9. 不在下一轮范围

- 不做数学/物理扩科。
- 不做大规模付费部署。
- 不重写整个 v3 管道。
- 不取消 gold diagnostic。
- 不把所有图题一次性强行放进学生端。
- 不让 LLM 视觉理解直接决定画像。

