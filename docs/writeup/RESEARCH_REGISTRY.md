# Research Registry / 研究索引

> 白皮书(WHITEPAPER.md)中每个可溯源的数字,在此可点开原始证据文件。全部为只读引用。

## A. 系统事实

| 数字 | 值 | 权威来源 |
|---|---|---|
| 结构化题库 | 3,329 条 | `data/item_bank/v4/chemistry_v4_3329.jsonl`(v4)+ `v4_1` 修正版 |
| R5 白名单 | 1,202 题(2,526 全池) | `data/item_bank/v4/usability_r5_v1.jsonl` + loader `core/data/item_bank_v4.py` |
| 图形转写 | 6,005 行 | `data/item_bank/v4/ws2_asset_transcripts_v1.jsonl` |
| 媒体映射 | 12,790 行 | `data/item_bank/v4/ws2_media_ref_map_v1.jsonl` |
| 知识图谱 | 135 节点 | `data/knowledge_graph_150_enriched.jsonl` |
| 开放节点 | 27 | `/api/demo/nodes`(快照,2026-07-13) |
| 独立内容题族 | 963 | `/health`(快照,2026-07-13) |
| 确定判分题 | 400 | M6 catalog audit(快照,2026-07-13) |

## B. 数据管线与 QA(BATCH6–16)

| 项 | 值 | 报告 |
|---|---|---|
| QA 全池 VL 审 | ¥17.93 / 2,526 题 | `docs/audit-history/BATCH14_AUDIT_2026-07-06.md` |
| 最终三池 | clean 799→1,295(名义)→1,203(排除后) | BATCH10/14/16 报告 |
| batch13 真死图回收 | 105 资产 / ¥2.53 | BATCH16_AUDIT §batch13 |
| 答案区占位 | 249→45(消 ~204 题) | BATCH16_AUDIT |
| 离子破损 | 残留 48 题(全池),text_ion 4% | BATCH16_AUDIT |
| 审查器精度金标 | 236 行,零误伤(1,207 白名单) | BATCH16_AUDIT §16b |

## C. AI 出题五轮(¥13.87)

| 轮 | 结果 | 报告 |
|---|---|---|
| P1 裸生成基线 | 首过 ≈45%;直接可服务 ≈40-45% | `NEIHUA_P1_AUDIT_2026-07-06.md` |
| P2 门控生成 | 终过 75/100 | `NEIHUA_P2_AUDIT_2026-07-06.md` |
| MVP 三路 | 文字 96.7% 正确;Track B 管线成立 | `NEIHUA_MVP_AUDIT_2026-07-07.md` |
| R2 | 图锚 19/20;RDKit 12/14 | `NEIHUA_R2_AUDIT_2026-07-07.md` |
| R3 | 风格转移方向确立 | `NEIHUA_R3_AUDIT_2026-07-07.md` |
| R4+R5 终审 | 区分率 65%/公平 60%;五轮完结 | `NEIHUA_R4R5_AUDIT_2026-07-08.md` |

## D. 三轮架构审计(2026-08-05→13)

| 发现 | 值 | 报告 |
|---|---|---|
| 四状态分类上限 | 54–69%(12 题) | `redteam1_measurement_selection.md` |
| P/U 不可辨识 | KL=0.0247 | `lane1_measurement.md` |
| gap>0.45 早停误判 | 10.3% | `redteam1_measurement_selection.md` |
| FSRS 旧式 S 值 | ×10 次后 4,608 天(4.5:73 天) | `redteam2_memory_recommendation.md` |
| n=2 下界 | 22.4% | `VERIFICATION_ROUND4.md` |
| SymPy 假等价 | 7 反例 / 5 复现 | `redteam3_verification_pipeline_product.md` |
| 21 组件终裁 | 4 保留 / 10 改造 / 5 替换 / 2 降级 | `MASTER_AUDIT_REPORT_2026-08-13.md` |

## E. 成本台账(与官方数据一致)

| 项 | 值 | 来源 |
|---|---|---|
| 化学全量(含前置) | ≈¥13,100 估算(Opus 开发占大头) | `各阶段Token成本估算_v1.md`(docs/history) |
| 数学/物理单科 | ≈¥2,180/科(引擎复用) | 同上 |
| 数学整页转写 | ¥104.28 / 400 卷 | `yihuier-math-skill/data/item_bank/split_progress.json` |
| 生成验证五轮 | ¥13.87 | NEIHUA R4R5 |
| Demo 全部 QA(2026-07-12 至 13) | ¥1.12(205 事件) | BATCH16/过夜收官 |

*注:E3 的"化学全量"包含早期探索性开发历史成本;不含真人验证阶段(未发生)。*
