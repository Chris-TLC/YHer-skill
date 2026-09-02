# Audit History Index

公开仓内的审计档案 = 项目三轮系统级审计的**终审级**报告。完整的过程记录(356 条逐日会话账本、批次内审记录)保留在项目内部,不在本公开仓。

## 目录

### 第一轮:项目级审计(2026-07-01 → 07-06)

| 文件 | 主题 |
|---|---|
| `YHER_FULL_PROJECT_AUDIT_2026-07-01.md` | 全项目审计:架构、数据管线、质量门设计 |
| `GOLD_QUESTIONS_MODEL_REVIEW_REPORT_2026-07-01.md` | 黄金题模型评审 |
| `VISUAL_ITEM_QUALITY_GATE_EXECUTION_PLAN_2026-07-01.md` | 视觉题质量门执行计划 |
| `VISUAL_ITEM_QUALITY_GATE_EXECUTION_REPORT_2026-07-01.md` | 视觉题质量门执行报告 |
| `STRATEGIC_REVIEW_2026-07-06.md` | 战略复审(项目定位与优先级) |

### QA 长征(BATCH6–16):2526 题可用性审计

| 文件 | 主题 |
|---|---|
| `BATCH10_AUDIT_2026-07-05.md` | 批次10:latex/离子/图资产修复审计 |
| `BATCH14_AUDIT_2026-07-06.md` | 批次14:校准门失败与 KaTeX 根因(审计基础设施与真产品同源教训) |
| `BATCH16_AUDIT_2026-07-06.md` | 批次16:节点感知审查零误伤;审查器精度金标方法论 |

### AI 出题五轮验证(2026-07-06 → 07-08)

| 文件 | 主题 |
|---|---|
| `NEIHUA_P1_AUDIT_2026-07-06.md` | 第一轮:裸生成基线(首过正确率≈45%) |
| `NEIHUA_P2_AUDIT_2026-07-06.md` | 第二轮:门控生成(六门规格) |
| `NEIHUA_MVP_AUDIT_2026-07-07.md` | MVP 三路验证(文字/渲染/图锚) |
| `NEIHUA_R2_AUDIT_2026-07-07.md` | R2 三路重跑(执行层诚实 FAIL 不等于路线 FAIL) |
| `NEIHUA_R3_AUDIT_2026-07-07.md` | R3 收口(风格转移方向) |
| `NEIHUA_R4R5_AUDIT_2026-07-08.md` | R4+R5 终审:五轮验证收口(区分率 65%/公平 60%) |

### 第二轮:CEO 工程审计(2026-07-10)

`YHER_CEO_ENGINEERING_AUDIT_2026-07-10.md` — 工程投入与产品判断双维度复审(结论:方向 70% 正确、优先级 40% 正确)。

### 第三轮:文献级架构审计(2026-08-05 → 08-13,11 份报告)

6 条研究车道(lane1-6)+ 3 条红队攻防(redteam1-3)+ 独立复核轮:

| 文件 | 主题 |
|---|---|
| `lane1_measurement.md` | 测量:二元化掌握度 vs 四状态构念 |
| `lane2_selection_stopping.md` | 选题与停止规则:gap>0.45 废止;P(top1)+min_length |
| `lane3_memory_review.md` | 记忆与复习:FSRS-4.5 阻尼替换手设常数 |
| `lane4_recommendation.md` | 推荐:乘法评分保留;efficacy Beta-Binomial 只作平局键 |
| `lane5_verification_profile.md` | 验证与画像:held-out 早停 3→6;n=2 二值化禁止 |
| `lane6_math_pipeline.md` | 数学管线:整页转写 + MFD + SymPy 全量语法门 |
| `redteam1_measurement_selection.md` | 红队1:四状态 12 题上限 54–69% 独立复算 |
| `redteam2_memory_recommendation.md` | 红队2:S 膨胀 4608 天复算;efficacy 无学术位置 |
| `redteam3_verification_pipeline_product.md` | 红队3:n=2 下界 22.4%;SymPy 7 假等价 5 复现;产品层一致性 |
| `VERIFICATION_ROUND4.md` | 独立复核轮:多重推翻(FSRS 旧计算错 5 倍) |
| `MASTER_AUDIT_REPORT_2026-08-13.md` | 终裁:21 组件 4 保留/10 改造/5 替换/2 降级;数学 MVP 蓝图 v0′ |

## 为什么公开这些

- 审计报告是"工程严谨性"的直接证据:每份都含**独立复现命令、输入哈希、逐条判定与证据等级**;
- 三轮审计的演进本身记录了"什么结论被推翻、为什么推翻"——这是科研记录的标准形态;
- 报告中的结论(如 gap>0.45 废止、FSRS-4.5 阻尼)已经或正在进入生产引擎。

## 未公开部分

- `ledger_archive/`(逐日会话账本 356 条):流程颗粒度过细,含内部事故与翻车记录,不作为公开档案;
- 批次级中间报告(BATCH6/8/9/11/12/13/15、R1 等):过程文档,非终审级。
