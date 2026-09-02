# YHer 视觉题资产与质量门执行报告

日期：2026-07-01 CST

## 1. 本轮结论

已把视觉题质量方案落成可重复脚本、数据 manifest、中心质量门、35 题视觉 eval set、多模态理解评估结果和浏览器预览 smoke。本轮复核时又修补了三个交付风险：`ItemRepository.find_items()` 默认不再绕过质量门，`build_visual_asset_manifest.py` 全量 dry-run 从无界慢执行修到约 22 秒可完成，主前端诊断结果返回后不再残留可见的禁用 `分析中...` 按钮。

关键质量策略：视觉题即使题图匹配为 `strong`，没有 P4 多模态理解强证据也不得进入 `diagnosis` 或 `profile_evidence`。当前只有 2 道视觉题有真实 qwen3-vl-plus 试点强证据，其余视觉题继续被拦截或保留在低风险练习/人工复核候选。当前学生诊断/画像有 4263 道可用，demo 不会被清空。

## 2. 交付物

- `yihuier-chemistry-skill/scripts/build_visual_asset_manifest.py`
- `yihuier-chemistry-skill/scripts/build_item_quality_manifest.py`
- `yihuier-chemistry-skill/scripts/build_visual_eval_set.py`
- `yihuier-chemistry-skill/scripts/evaluate_visual_understanding.py`
- `yihuier-chemistry-skill/scripts/build_visual_quality_preview.py`
- `yihuier-chemistry-skill/core/data/item_quality.py`
- `yihuier-chemistry-skill/core/data/item_repository.py`
- `yihuier-chemistry-skill/data/quality/visual_asset_manifest.jsonl`
- `yihuier-chemistry-skill/data/quality/visual_asset_manifest_summary.json`
- `yihuier-chemistry-skill/data/quality/item_quality_manifest.jsonl`
- `yihuier-chemistry-skill/data/quality/item_quality_summary.json`
- `yihuier-chemistry-skill/data/evals/visual_item_eval_set.jsonl`
- `yihuier-chemistry-skill/data/evals/visual_item_eval_set_summary.json`
- `yihuier-chemistry-skill/data/evals/visual_understanding_results.jsonl`
- `yihuier-chemistry-skill/data/evals/visual_understanding_summary.json`
- `yihuier-chemistry-skill/apps/web/visual_quality_preview.html`
- `yihuier-chemistry-skill/tests/test_visual_asset_manifest.py`
- `yihuier-chemistry-skill/tests/test_item_quality.py`
- `yihuier-chemistry-skill/tests/test_visual_understanding_eval.py`

## 3. Manifest 结果

总题量：6438。

图依赖题量：1815。

视觉资产 manifest 覆盖：

- `strong`: 1066
- `weak`: 705
- `reject`: 44
- 缺页图：44
- page mismatch：559
- source unresolved：0
- source ambiguous：37

按类别：

- organic_structure：515
- experiment_device：429
- chart_curve：340
- other：269
- process_flow：126
- crystal_cell：104
- electrochem_device：32

## 4. 中心质量门结果

`item_quality_manifest.jsonl` 覆盖 6438 道。

- needs_image：1815
- readability pass：5689
- readability manual_review：749
- answer verified：5909
- answer missing：170
- answer suspect：359
- rubric complete：6288
- rubric missing：150
- llm understanding strong：2
- llm understanding weak：1765
- llm understanding reject：48
- usable_for_diagnosis：4263
- usable_for_practice：5347
- usable_for_teaching：5347
- usable_for_profile_evidence：4263
- blocked_items：1091

中心规则：

- 没有 manifest 的题默认不能进入诊断。
- `ItemRepository.find_items()` 默认用途为 `diagnosis`，调用方忘记传 `purpose` 时不会绕过质量门。
- `debug_all` 只供脚本/测试显式查看全部题。
- 图依赖题必须有 strong visual asset、pass readability、strong LLM understanding 后才能进入 diagnosis/profile_evidence。
- 只有 P4 `understanding_pass=true` 的视觉题能进入 diagnosis/profile_evidence；当前为 2 道。

## 5. 视觉评测集与多模态评估

`visual_item_eval_set.jsonl`：35 题，21 个来源文件，其中 6 题来自已有 qwen3-vl-plus 小批量多模态试点并能映射到当前 item_id。

覆盖：

- crystal_cell：3
- experiment_device：3
- process_flow：3
- chart_curve：17
- organic_structure：3
- electrochem_device：3
- other：3

`visual_understanding_results.jsonl`：

- eval_items：35
- model_called：false（当前结果为导入已有 qwen3-vl-plus 试点 + 未评估项占位）
- pilot_imported：6
- not_evaluated：29
- visible_pass：4
- answer_match：2
- understanding_pass：2
- profile_evidence_allowed：2
- high_confidence_errors：2

说明：本轮尝试直接调用 qwen-vl 进行小批量评估；最新 dry-run 为 2/2 timeout，没有写入结果。当前 P4 采用已有 `/tmp/yher_multimodal_pilot.json` 的 6 条真实 qwen3-vl-plus 结果作为模型证据，并把其余 29 题显式标记 `not_evaluated_offline`。2 个高置信错误已计入，不会放入画像。

## 6. 学生端展示 Smoke

生成静态预览页：

- `apps/web/visual_quality_preview.html`

Browser + Playwright 检查：

- 桌面视口 1280 宽：12 张卡片、12/12 图片加载、无横向溢出。
- 手机视口 390x1100：12 张卡片、12/12 图片加载、无横向溢出。
- 控制台 warn/error：0。

截图：

- Browser 截图已在本轮 smoke 中采集；旧截图路径 `/tmp/yher_visual_quality_preview_desktop.png`、`/tmp/yher_visual_quality_preview_mobile.png` 可作为上一轮参考。

API smoke：

- `GET /health` 返回 `{"status":"ok","kg_nodes":135,"item_bank":6438}`
- 创建 session 成功：`化学平衡`，6 个 task
- `first_question` 成功
- 连续两轮 `diagnose` 成功，均返回 `next_question`
- `profile` 返回五维画像：基础概念、应用迁移、综合推理、审题入口、整体掌握

主前端 smoke：

- `index.html` 首页加载，知识点列表可见，控制台 warn/error：0。
- 实际点击 `等效平衡` -> `开始诊断` -> 提交两题 -> `查看能力画像` 成功。
- 画像页显示五维画像 SVG 和百分比分布，无横向溢出。
- 复测结果：自由回答题分析返回后旧 `分析中...` 按钮 `display:none`，页面只显示下一步按钮；无标准答案的自由回答显示中性 `已记录`，不再误写 `答对了`。

## 7. 测试

已运行：

- `python3 tests/test_visual_asset_manifest.py`：3/3 通过
- `python3 tests/test_item_quality.py`：6/6 通过
- `python3 tests/test_visual_understanding_eval.py`：3/3 通过
- `python3 tests/test_tutor_engine.py`：12/12 通过
- `python3 -m py_compile ...`：通过
- 数据审计：visual 1815 行、quality 6438 行、eval 35 行、understanding 35 行，0 坏 JSON、0 必填字段缺失、0 视觉诊断/画像泄漏、eval 图片 35/35 存在。
- `python3 scripts/build_visual_asset_manifest.py` dry-run：约 22 秒完成并输出摘要。

## 8. 被拦截样例类型

质量门会拦截：

- 缺页图的视觉题。
- source ambiguous 的题。
- declared page 与 best text page 不一致的题。
- answer missing / answer suspect 的题。
- rubric missing 的诊断/画像候选题。
- 未经过强多模态理解评估的视觉题。
- 多模态高置信但答案错的视觉题。

## 9. 下一步人工 Gold / Eval 清单

优先补人工或强校准样本：

1. 晶胞：配位数、密度、空间结构题。
2. 实验装置：装置连接、除杂、尾气吸收、制备装置。
3. 工艺流程：物质流向、循环物、条件控制、答案结构。
4. 曲线/图像分析：pH、滴定、速率、转化率、溶解度曲线。
5. 有机结构式：官能团、同分异构、合成路线。
6. 电化学装置：电极反应、离子迁移、隔膜、电解/原电池判断。
7. 三大硬空洞：溶液三大守恒、工艺流程、实验综合大题。

## 10. 未完成与风险

- 直接实时模型调用在本地环境中出现长时间无响应；当前 P4 使用已有真实 6 题试点结果，剩余 29 题仍需后续在稳定调用环境中补跑。
- 未生成题目级 bbox/crop，只使用 `page_only` 预览；因此学生端正式视觉题仍应走预览/人工复核后再开放。
- 未评估或失败视觉题不得进入 diagnosis/profile_evidence；当前只有 2 道视觉题有 `llm_understanding_status=strong`。
- 本轮没有清理脏工作树、没有重跑旧 v3 管道、没有写入或展示 API key。
