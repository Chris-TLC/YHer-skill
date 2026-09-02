# Lane 6：数学题库数据管线 / 知识图谱 / 符号判分 — 文献检索与最优解判定

生成时间：2026-08-13 CST ｜ 性质：只读研究，未改任何项目代码
证据等级约定：[全文] > [摘要] > [题名] > [记忆回溯]；数字一律来自本次抓取内容；API 不可达标"检索受限"。

---

## 1. 检索记录

| 渠道 | 用途 | 状态 |
|---|---|---|
| arXiv export API（https + -L） | uniMERNet、先修关系、判分、出题等 40+ 查询 | 初段可用；中后段被并行会话挤占，大量 429，部分改用 arxiv.org 站内搜索补足 |
| Semantic Scholar API | 引用数、摘要核验 | **全程 429，检索受限**（共享出口 IP 限流） |
| Crossref | Sangwin 2013 书、FormulaNet、distractor、Qiu 2019 等 | 正常，命中率中等 |
| DBLP / OpenAlex | 先修关系、难度预测、KT 粒度 | 可用（OpenAlex 后期配额耗尽） |
| webfetch | stack-assessment.org 文档、UniMERNet GitHub、教育部课标原文、沪教版目录页、上海卷结构页面 | 正常；百度百科/知乎/知网/OECD 403 或反爬 |
| 检索受限记录 | Semantic Scholar 全量、arXiv 部分查询（MTMM、MFR benchmark、MATH-Verify 精确定位）、CNKI 全文、中文 MFR 基准 | 记入未决问题 |

---

## 2. 六个问题的文献回答

### Q1 数学公式 OCR 现状（整页转写 vs 区域裁剪）

- **uniMERNet**（arXiv:2404.15254，2024，OpenDataLab/上海AI实验室）：UniMER-1M 训练集 **1,061,791** 对 LaTeX-图像，UniMER-Test 23,757 样本；官方 README 的准确率具体数字在图片中，本次未能读出，不编造。[摘要+README]
- 指标：ExpRate = 预测与真值完全一致比例（引 Yuan et al. 2022）；**CDM**（arXiv:2409.03643，CVPR2025）把 LaTeX 渲染成图后做字符级检测匹配，解决"同式异写"评测不公平问题，2025-06 起支持中文公式评测。[摘要+README]
- **PP-FormulaNet**（百度，arXiv:2503.18382，2025）宣称 L 版准确率超 uniMERNet 6%、S 版快 16 倍。[摘要]（厂商自报，需自测复核）
- **中文/高考数学 MFR 基准：不存在**（arXiv/OpenAlex/Crossref 均无专门数据；ICDAR 2017 有一篇中文文档鲁棒公式识别，10.1109/icdar.2017.27）。[题名]——这是本项目最大的测量真空。
- LaTeX 错误检测/修复 SOTA：EMNLP 2023 "LM 用于手写公式识别纠错"（10.18653/v1/2023.emnlp-main.247）；Applied Sciences 2023 LaTeX 语法约束+主动学习（10.3390/app132212503）。sympy parse/LaTeX AST 校验**无专门论文**，属工程实践；最完整的可复现范式是 **Minerva**（arXiv:2206.14858）附录：`sympy.parsing.latex.parse_latex` 解析 → simplify 判零 → 5s 超时。[全文]
- 整页 vs 两阶段：公式密集页整页端到端模型（Nougat 2308.13418、Docling 2408.09869）在公式混排处质量不稳；**两阶段（公式检测→区域裁剪→逐式识别）是当前开源主流**（Pix2Text mfd+mfr、uniMERNet 官方教程用 PDF-Extract-Kit 的 MFD 模型先检测）。[摘要+README]

**结论**：无 DOCX 源时，视觉路线最优管线 = **两阶段补充识别**：保留整页转写作初稿，对公式密集/判分关键题用 MFD 检测裁剪 → uniMERNet/PP-FormulaNet 重识别 → sympy parse 作语法门 → 渲染回图比对（CDM 思路）做二次校验。整页转写单通道不可靠，但作为"题干上下文+结构"的初稿保留价值明确。

### Q2 题库质量保证

- **CAS 验证答案**：Minerva（arXiv:2206.14858）[全文] 是符号验证模式的最佳文档：解析答案 → 两 SymPy 对象相减 → simplify 判零 → 5s 超时，并警告归一化不当会致假阴性。xVerify（arXiv:2504.10481，2025）答案等价性判定模型，全变体 F1>95%。[摘要]
- **LLM 一致性检查**：Self-Consistency（arXiv:2203.11171，ICLR2023）多样本投票（S2 引用 7412 次，检索受限未复核）。**最对口**：ValiMath「Let's Verify Math Questions Step by Step」（KDD 2026，DOI:10.1145/3770854.3785700）——2,147 道人验数学题，首次把验证对象从答案转向**题目本身正确性**。[摘要] "MATH-Verify" 精确论文未检索到（受限）。
- **干扰项质量**：Feng et al.（NAACL Findings 2024）：LLM 干扰项"数学上有效但难复现学生真实误解"；DiVERT（EMNLP 2024）1434 道数学 MCQ 上 7B 模型以可解释误差表示超 GPT-4o；**Schmucker & Moore**（arXiv:2503.10533，2025）：19 项 IWF（题目编写缺陷）准则可自动标注 7,126 道 STEM 选择题，"implausible distractors"是关键缺陷项，IWF 数与 IRT 难度/区分度显著相关。[摘要]
- **难度预测**：Qiu et al.（CIKM 2019）DAN 文本特征预测；SMART（EMNLP 2025，arXiv:2507.05129）模拟学生+DPO+IRT 拟合开放题难度；警示：**"Hard or Just Unreached?"**（arXiv:2606.19636）——pass@k 难度信号有盲区，pass@k=0 的题 10.3–22.9% 实际可解。[摘要]

### Q3 知识图谱构建

- **先修关系**：Liang et al. AAAI 2017「Recovering Concept Prerequisite Relations」（DOI 10.1609/aaai.v31i1.10550，课程依赖→概念先修+数据集）、AAAI 2018 主动学习版；**Sayyadiharikandeh et al. 2019 实为 WWW'19 Companion**（DOI 10.1145/3308560.3316753，Wikipedia Clickstream，非 WSDM2019——纠正）；Pan et al. ACL 2017 MOOC 先修（10.18653/v1/p17-1133）；"How Much is 131 Million Dollars?" 实为 Chaganty & Liang ACL 2016（数字描述任务，**与 MOOC 先修无关，用户记忆有误**）。LLM 路线：EAI 2024 prompt 抽取先修（DOI 10.4108/eai.24-11-2023.2343577）、arXiv:2507.18479（2025）「LLMs Predict Prerequisite Skills? 零样本 vs 专家 KC」。[题名为主]
- **数学 KG 资源**：Math-KG 实为 **arXiv:2205.03772，2022，单人作者，非微软**（纠正）；AutoMathKG（arXiv:2505.13406，2025，LLM+向量库）；MOOCube、AlgoMath、MathKnowTopic **未能验证**，不引用。[题名]
- **沪教版/考纲**（抓到原文/页面）：《普通高中数学课程标准（2017 版 2020 修订）》（教育部官网）——四条主线（函数、几何与代数、概率与统计、数学建模与探究）+必修 5 主题/选必 4 主题；沪教版 7 册目录（教习网+中学课本网交叉佐证）：必修一（集合与逻辑/等式不等式/幂指对/函数）、必修二（三角/向量/复数）、必修三（立体几何/概率统计）、必修四（建模）、选必一（直线/圆锥曲线/空间向量/数列）、选必二（导数/计数原理/概率统计续）、选必三（建模）。**上海 2023 年起不发布公开考纲**，命题依据新课标；春考=必修+选必 1-6 章，秋考=全部。[题名/页面全文，二手来源]
- **KC 粒度**：细粒度 KC 提升诊断精度但增加标注成本与数据稀疏性；多粒度集成 + 自动 KC 生成是 2023-2025 趋势（KBS 2025 10.1016/j.knosys.2024.112834、Findings ACL 2026 自动 KC 生成、Koedinger EDM 2012 Automated Student Model Improvement 为先河）。[题名为主]
- **注意**：Liang/Sayyadiharikandeh 路线依赖 MOOC 点击流/课程依赖数据，YHer 没有；**LLM 抽取+专家审核才是冷启动可行路线**。

### Q4 符号判分

- **STACK**（Sangwin）：判分树 = Answer Tests 分层 + 潜在响应树（PRT，含惩罚机制）。测试严格度递增：CasEqual（解析树相等）→ EqualComAss（交换/结合）→ **AlgEquiv（simplify(ex1−ex2)=0）** → SubstEquiv（变量重命名）→ SysEquiv（Gröbner 基判方程组）。**已文档化的坑**（stack-assessment.org 文档 [全文]）：方程等价按根集+重数判（num(x1)/num(x2)），绕开解方程不可判定性；Cardano 例 ³√(√108+10)−³√(√108−10)=2 AlgEquiv 判不出，需补数值探针；浮点数禁用 AlgEquiv（452 vs 4.52×10² 会 fail）；(a^x)^y vs a^(xy) 刻意判不等价（定义域坑，需 assume）；radcan 不默认（√2/√3 vs −2/√6 判不等）；**数值采样是补充探针、非充分判据**（Sangwin《Computer Aided Assessment of Mathematics》OUP 2013，DOI:10.1093/acprof:oso/9780199660353.001.0001）。[摘要+全文]
- **三路线可靠性**：CAS 确定性、可审计、需按题型配测试 [全文]；LLM 判分实证**严重分化**——正面：AMMORE（arXiv:2409.17904，2024）CoT 判边界案例 92%；Henkel（arXiv:2510.05538，2025）MLLM 判手写算术 95%、κ=0.90，**但图形题 κ=0.20**；SciEx（EMNLP 2024）与专家 r=0.948。**负面**：Jade et al.（arXiv:2505.04645，2025）ChatGPT-4o 判医学简答 ICC1=0.086、κ=−0.08、>60% 超误差界，**作者明确警示高风险场景禁用**。[摘要]
- **过程判分替代**：Math-Shepherd（arXiv:2312.08935，2023）步骤级 PRM 无人工标注；应用层：MathEDU（EACL 2026）正确性分类/错误定位/反馈三任务，**反馈质量与教师差距大**；SteLLA（arXiv:2501.09092）RAG+结构化 rubric 与人工"substantial 一致"；**CHiL(L)Grader（arXiv:2603.11957，2026）置信度路由：仅 35–65% 响应可自动判到 QWK≥0.80**。[摘要] 各研究用 κ/QWK/Pearson 不一，无统一基准，不可直接横向比。

### Q5 题目分类与 LLM 出题

- **分类**：通用文本分类直接迁移到数学题效果不佳，需标签注意力（LABS，arXiv:2208.09867，中文 K12 数学题多标签知识点标注 F1 超 BiLSTM）；LLM 抽取知识点可行（MathScale 用 GPT-3.5 建概念图，arXiv:2403.02884）；**中文带知识点标注的现成数据集**：CMMaTH 23k 中文 K12 多模态题（知识点/题型/标准解）、MDK12-Bench 140K 题/6,827 知识点、MM-MATH 5,929 题。嵌入聚类在中文数学题上**无证据**。Ape210K（210K 中文题）**作者撤回、数据不公开**。[摘要]
- **出题/变式**：MathGenie（ACL 2024，arXiv:2402.16352）解→题回译+代码解验证，保真靠解验证；MetaMath（ICLR 2024）只做表面改写、**无结构保真验证**（反例）；SDE-GPG（ACL 2025，arXiv:2506.02565）符号演绎引擎生成几何题，**硬保证**可解+知识点/难度可控；We-Math 2.0 每道题 7 个渐进难度变式+491 知识点体系；难度可控：DQG（IJCAI 2019）→ DCQG（ACL 2021，难度=推理步数）→ CoDiQ（2026，44K 竞赛题 82%+ 可解）。"DiversityMath" 精确名未检索到，最接近为 ControlMath（EMNLP 2024）。[摘要]
- **警示**：自动评测指标与专家评测不一致（AIED 2024，arXiv:2408.04394）——LLM 出题必须先人工验收再入练习供给。

### Q6 中文数学教育测量传统

- **难度模型**：官方口径 = "较高的信度、效度，必要的区分度，适当的难度"（任子朝等《高考试题难度预估研究》，《数学教育学报》2018,27(5):13-16；知网全文受限，[摘要]级经公众号转载核）。
- **素养框架**：2017 课标六大核心素养（数学抽象、逻辑推理、数学建模、直观想象、数学运算、数据分析）+ 学业质量三水平（水平二=高考要求）；喻平三水平（知识理解/迁移/创新）；PISA 2022 以数学推理为核心+三环节+8 项 21 世纪技能（8 项名称[记忆回溯]，OECD 官网 403 未能直读）。[摘要]
- **上海卷结构（多源交叉核实）**：2017–2022：填空 12 题 54 分（1-6 各 4，7-12 各 5）+ 单选 4 题 20 分 + 解答 5 题 76 分（14/14/14/16/18）= 21 题 150 分，无多选题；**2024 起微调**：选择 18 分（13-14 各 4，15-16 各 5）、解答 78 分（17-19 各 14，20-21 各 18），压轴为第 21 题。[题名/页面，2025-2026 仅单一来源待核实]
- **对诊断设计的影响**：填空/选择 = 结果可判分（与 SymPy 判分器天然匹配，占 72-74 分）；解答题 = 过程分 76-78 分是诊断信号富矿但判分难，与"过程题不判分"直接冲突（见裁决）。

---

## 3. 方案对比表

### 3.1 题库 QC 管线

| 方案 | 成本 | 可靠性 | 冷启动 |
|---|---|---|---|
| A. sympy parse 语法门 + 渲染回图 CDM 比对（Minerva 式） | 低（开源、无 API） | 语法级可靠；语义错误漏检 | 立即可用，每道题 <1s |
| B. LLM 三解一致性（ValiMath 式题目正确性验证） | 中（API 费，每道题 3-6 次采样） | 中（一致性高但共模错误存在） | 需先建提示词+人验 200 题金标 |
| C. 人工复核（教师/志愿者） | 高（人力） | 高 | 慢，10103 题不可行 |
| D. 纯规则去重/完整性（现有 1732 筛查） | 低 | 高查准低查全 | 已存在 |
| **推荐：D 保底 + A 全量 + B 对拟服务题定向 + C 抽检 5%** | 中 | 分层兜底 | 分阶段上线 |

### 3.2 KG 构建路线

| 方案 | 成本 | 可靠性 | 冷启动 |
|---|---|---|---|
| A. 数据驱动先修学习（Liang 2017/Sayyadiharikandeh 2019） | 高（需 MOOC 点击流/课程依赖数据） | 高（有数据时） | **不可行：YHer 无此类数据** |
| B. 课标+沪教版骨架手动构建（课标四主线→7 册章→节） | 中（1-2 人周） | 高（权威、与考纲对齐） | 立即可做，已抓到目录 |
| C. LLM 抽取先修边 + 专家审核（EAI 2024 模式） | 低-中 | 中高（LLM 先修预测 vs 专家有差距，需审核） | 骨架就绪后即可叠加 |
| D. 借用现成数学 KG（Math-KG/AutoMathKG） | 低 | 低-中（非 K12 考纲对齐） | 快但需大量对齐工作 |
| **推荐：B 骨架 + C 边 + D 仅作概念名参考** | 中 | 高 | 2 周出 v1 |

### 3.3 判分器路线

| 方案 | 成本 | 可靠性 | 冷启动 |
|---|---|---|---|
| A. SymPy 单一 simplify 判零 | 低 | 中：假阴性/假阳性（化简标准、定义域、浮点、多值函数坑，STACK 文档实证） | 1 天可建 |
| B. STACK 式分层 Answer Test（CasEqual→AlgEquiv→数值探针→题型化） | 中 | 高：坑有文档化规避；确定性可审计 | 2-3 周可建 v1 |
| C. LLM 判分（结果或过程） | 中 | 分化严重：算术类 κ=0.90，图形/开放类 κ=0.20；医学简答 κ=−0.08 被作者警示禁用 | 快但不可审计 |
| D. 数值采样判分 | 低 | 中：伪等价风险（Sangwin：探针非充分判据） | 快 |
| **推荐：B（A 是其子集）+ D 作 B 的探针层；C 仅用于"过程自查辅助"（rubric+置信度路由，只给反馈不给分数）** | 中 | 高 | 分层渐进 |

---

## 4. 裁决

| 现行设计 | 裁决 | 理由与改法 |
|---|---|---|
| 整页转写（qwen-vl 视觉 OCR 路线） | **保留但改造** | 视觉路线本身不否定（无 DOCX 源），但需补两阶段管道：MFD 检测→裁剪→uniMERNet/PP-FormulaNet 重识别公式区；sympy parse 作全量语法门（Minerva 模式）；CDM 渲染回图比对作输出校验。整页转写降级为"题干结构初稿+判分题重识别源"。 |
| 1732 题保守筛查 | **保留为冷启动种子，扩容协议改造** | 不推翻筛查逻辑，但 1732 不是终点：定义"可服务题"= 过 parse 门 + 答案键经 sympy 验证 + 有知识点标签 + LLM 一致性验证通过；每 500 题出一份验收报告（解析通过率/验证通过率/人工抽检一致率≥95%）。 |
| SymPy 等价比对判分（方案已定、代码未建） | **保留（方向正确），按 STACK 分层实现** | 单一 simplify 判零不可直接用。分层：L0 文本归一（全角/空白/裸上标修复）→ L1 parse 失败入人工队列 → L2 simplify 判零 → L3 数值采样 20 点探针 → L4 题型化等价（方程根集+重数、区间、角度弧度、分母不为零）。LLM 判分不进主链路（Jade 2025 警示）。 |
| KG=0 从头搭 | **不替换，但"借壳"而非从零学习** | 放弃数据驱动先修学习（无 MOOC 数据）；用课标四主线+沪教版 7 册目录做骨架（已抓取），LLM 抽取先修边+专家审核；KC 粒度走粗→细渐进（先章级 40-60 节点，再 20 个高频知识点试点细粒度）。 |
| 过程题不判分（对照标准解法自查） | **部分保留：不自动打分，但升级为 rubric 辅助自查** | 证据支持不给分数（LLM 过程判分反馈质量与教师差距大、仅 35-65% 可自动判到 QWK≥0.80）。但"对照解法自查"可产品化为：SteLLA 式结构化 rubric + CHiL(L)Grader 式置信度路由，只输出错因反馈与自查清单，不输出分数。 |

**数学 MVP 数据基座最小可行方案**：
1. **KG v1（2 周）**：节点 = 课标四主线→沪教版 7 册→章→节（约 150-250 节点）；边 = 先修关系（LLM 抽取+人工审核 100 组以内）+ 考纲范围（春/秋考标记，2024 后结构）；首期只精化 20 个高频知识点。
2. **判分器分层（3 周）**：L0-L4 如裁决所述；先只覆盖填空/选择（72-74 分题型），解答题最终小问答案（如压轴题第 (1) 问的数值答案）纳入 L4 题型化等价。
3. **1732→可服务验收协议**：① 每题过 L0-L2 与 3 次 LLM 求解一致性；② 知识点标签人机双标（冲突仲裁）；③ 随机抽 5% 人工复核，一致率≥95% 方可上诊断闭环；④ 每 500 题出验收报告。
4. **中文 MFR 评测集**：自建 200 道题的人工金标（含 2024-2025 上海卷），用于验收重识别管道——文献不存在现成基准。

---

## 5. 未决问题

1. Semantic Scholar 全程 429：引用数、部分摘要未能核验；"MATH-Verify"（2024）精确定位未完成。
2. 中文 MFR 无现成基准——需自建金标评测集；uniMERNet 官方准确率数字（图片中）未读出。
3. 上海考纲为二手转载（官方不发布）、沪教版目录来自教辅网站——建议以纸质教材目录二次核对；2025-2026 卷结构仅单一来源。
4. MOOCube / AlgoMath / Math-KG 细节未能验证；"MTMM"、"MFR benchmark"、"FormulaNet CVPR2019" 未获文献证实（疑似项目内引用失真）。
5. LLM 判分研究指标不统一（κ/QWK/Pearson），跨研究不可比；纯数学符号答案场景的 LLM 判分实证缺失。
6. 任子朝 2018 难度预估公式细节（知网全文受限）未取得；PISA 2022 官方框架原文未直读（OECD 403）。

---

## 6. 参考文献表

### OCR / LaTeX
| 标题 | 年 | ID/DOI | 证据 |
|---|---|---|---|
| UniMERNet: A Universal Network for Real-World Mathematical Expression Recognition | 2024 | arXiv:2404.15254 | 摘要+README |
| CDM: A Reliable Metric for Fair and Accurate Formula Recognition Evaluation | 2024 | arXiv:2409.03643 (CVPR2025) | 摘要+README |
| PP-FormulaNet: A Small Fast Baseline for Printed Math Expression Recognition | 2025 | arXiv:2503.18382 | 摘要 |
| MathWriting: The Largest Handwritten Mathematical Expression Dataset | 2024 | arXiv:2404.10690 | 摘要 |
| Nougat: Neural Optical Understanding for Academic Documents | 2023 | arXiv:2308.13418 | 摘要 |
| Docling Technical Report | 2024 | arXiv:2408.09869 | 摘要 |
| IBEM: A New Dataset for Printed Scientific Document Formula Detection | 2023 | 10.1016/j.patrec.2023.05.033 | 摘要 |
| 1st Place Solution for ICDAR2021 MFD | 2021 | arXiv:2107.05534 | 摘要 |
| Robust Math Formula Recognition in Degraded Chinese Document Images | 2017 | 10.1109/icdar.2017.27 | 题名 |
| LM-based correction for handwritten math expression recognition | 2023 | 10.18653/v1/2023.emnlp-main.247 | 摘要 |
| Advancing OCR image-to-LaTeX via syntax-constrained augmentation | 2023 | 10.3390/app132212503 | 摘要 |
| Pix2Text / PDF-Extract-Kit | 持续 | GitHub（breezedeus/pix2text；opendatalab/PDF-Extract-Kit） | README |

### 题库 QC / 难度
| 标题 | 年 | ID/DOI | 证据 |
|---|---|---|---|
| Solving Quantitative Reasoning Problems with Language Models (Minerva) | 2022 | arXiv:2206.14858 | 全文（sympy 验证附录） |
| Self-Consistency Improves Chain of Thought Reasoning | 2023 | arXiv:2203.11171 (ICLR) | 摘要 |
| Let's Verify Math Questions Step by Step (ValiMath) | 2026 | 10.1145/3770854.3785700 (KDD) | 摘要 |
| Let's Verify Step by Step (PRM800K) | 2023 | arXiv:2305.20050 | 摘要 |
| xVerify | 2025 | arXiv:2504.10481 | 摘要 |
| Exploring Automated Distractor Generation for Math MCQs | 2024 | 10.18653/v1/2024.findings-naacl.193 | 摘要 |
| DiVERT | 2024 | 10.18653/v1/2024.emnlp-main.512 | 摘要 |
| Impact of Item-Writing Flaws on IRT | 2025 | arXiv:2503.10533 | 摘要 |
| Question Difficulty Prediction for MCQ in Medical Exams (DAN) | 2019 | 10.1145/3357384.3358013 (CIKM) | 摘要 |
| SMART: Student Modeling for Difficulty Estimation | 2025 | arXiv:2507.05129 (EMNLP) | 摘要 |
| Hard or Just Unreached? | 2026 | arXiv:2606.19636 | 摘要 |

### KG / 先修 / KT
| 标题 | 年 | ID/DOI | 证据 |
|---|---|---|---|
| Recovering Concept Prerequisite Relations from University Course Dependencies | 2017 | 10.1609/aaai.v31i1.10550 (AAAI) | 题名 |
| Investigating Active Learning for Concept Prerequisite Learning | 2018 | 10.1609/aaai.v32i1.11396 (AAAI) | 题名 |
| Finding Prerequisite Relations using the Wikipedia Clickstream | 2019 | 10.1145/3308560.3316753 (WWW'19 Comp.) | 题名 |
| Prerequisite Relation Learning for Concepts in MOOCs | 2017 | 10.18653/v1/p17-1133 (ACL) | 题名 |
| A Prompt-based Approach for Discovering Prerequisite Relations Among Concepts | 2024 | 10.4108/eai.24-11-2023.2343577 | 题名 |
| How Well Do LLMs Predict Prerequisite Skills? | 2025 | arXiv:2507.18479 | 题名 |
| Math-KG: Construction and Applications | 2022 | arXiv:2205.03772（非微软，纠正） | 题名 |
| AutoMathKG | 2025 | arXiv:2505.13406 | 题名 |
| Automated Student Model Improvement | 2012 | Koedinger et al., EDM | 题名 |
| Learning Factors Analysis | 2006 | 10.1007/11774303_17 (ITS) | 题名 |
| Multi-Granularity Ensemble Interaction Graph Modeling for KT | 2025 | 10.1016/j.knosys.2024.112834 (KBS) | 题名 |
| 普通高中数学课程标准（2017 年版 2020 年修订） | 2020 | moe.gov.cn/srcsite/A26/s8001/202006/t20200603_462199.html | 全文 |
| 沪教版高中数学教材目录（必修 1-4、选必 1-3） | 2020 | zxkeben.szxuexiao.com / 51jiaoxi.com | 页面全文 |

### 判分
| 标题 | 年 | ID/DOI | 证据 |
|---|---|---|---|
| STACK 官方文档（Answer Tests / PRT / Equivalence） | 2025 | docs.stack-assessment.org | 全文 |
| Sangwin: Computer Aided Assessment of Mathematics | 2013 | 10.1093/acprof:oso/9780199660353.001.0001 (OUP) | 摘要+记忆 |
| AMMORE: Learning to Love Edge Cases | 2024 | arXiv:2409.17904 | 摘要 |
| Seeing the Big Picture（MLLM 手写判分） | 2025 | arXiv:2510.05538 | 摘要 |
| SciEx: Benchmarking LLMs on Scientific Exams | 2024 | arXiv:2406.10421 (EMNLP) | 摘要 |
| ChatGPT for automated grading SAQs | 2025 | arXiv:2505.04645 | 摘要 |
| Math-Shepherd | 2023 | arXiv:2312.08935 | 摘要 |
| AlphaMath Almost Zero | 2024 | arXiv:2405.03553 (NeurIPS) | 摘要 |
| MathEDU | 2026 | arXiv:2505.18056 (EACL) | 摘要 |
| SteLLA | 2025 | arXiv:2501.09092 | 摘要 |
| CHiL(L)Grader | 2026 | arXiv:2603.11957 | 摘要 |

### 分类 / 出题
| 标题 | 年 | ID/DOI | 证据 |
|---|---|---|---|
| LABS: 中文 K12 数学题知识点多标签标注 | 2022 | arXiv:2208.09867 | 摘要 |
| CMMaTH | 2024 | arXiv:2407.12023 | 摘要 |
| MDK12-Bench | 2025 | arXiv:2504.05782 | 摘要 |
| MM-MATH | 2024 | arXiv:2404.05091 | 摘要 |
| Ape210K（作者撤回、不公开） | 2020 | arXiv:2009.11506 | 摘要 |
| MathGenie: Question Back-translation | 2024 | arXiv:2402.16352 (ACL) | 摘要 |
| ControlMath | 2024 | arXiv:2409.15376 (EMNLP) | 摘要 |
| MetaMath | 2023 | arXiv:2309.12284 (ICLR) | 摘要 |
| SDE-GPG: 符号演绎生成几何题 | 2025 | arXiv:2506.02565 (ACL) | 摘要 |
| We-Math 2.0 | 2025 | arXiv:2508.10433 | 摘要 |
| DCQG: Step-by-Step Rewriting | 2021 | arXiv:2105.11698 (ACL) | 摘要 |
| CoDiQ | 2026 | arXiv:2602.01660 | 摘要 |
| 自动评测与专家评测不一致 | 2024 | arXiv:2408.04394 (AIED) | 摘要 |

### 测量传统
| 标题 | 年 | 出处 | 证据 |
|---|---|---|---|
| 任子朝等《高考试题难度预估研究》 | 2018 | 数学教育学报 27(5):13-16 | 摘要（转载） |
| 高考数学核心素养测评研究 | 2018-2020 | 数学教育学报/课程·教材·教法（知网受限） | 摘要 |
| PISA 2022 Mathematics Framework | 2023 | OECD（官网 403 未直读） | 摘要+记忆 |
| 上海高考数学卷结构（2017-2022 / 2024-） | 2017-2025 | creditsailing.com 等多源交叉 | 题名/页面 |

---

*本报告由 lane6 研究代理生成；所有"纠正"标注（如 Sayyadiharikandeh 出处、Math-KG 非微软、FormulaNet CVPR2019 不存在、"131 Million Dollars" 与 MOOC 无关）均基于本次抓取的 API 返回与页面，与项目既有记忆冲突处以此为准或列入未决。*
