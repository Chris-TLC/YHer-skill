# Lane 1: 知识状态测量模型文献检索与最优解判定（YHer 数学 MVP）

- 日期: 2026-08-13
- 任务: 只读研究。为 YHer 数学线「单知识点诊断」裁定测量模型：当前化学线四状态 Bayes 是否保留/改造/替换。
- 结论速览: **改造**。保留「每 KC 一个二元 mastery 概率 + 显式不确定度」的 BKT 骨架；废除 M/P/C/U 互斥四状态；前置关系改为图边（KST surmise 结构）；"概念/解题链不稳"改为错误码表；证据不足改为后验不确定度元标签。冷启动路径 = BKT 默认参数 + KST-lite 前置图 + 半分裂选题，真人数据出现后再升级 CD-CAT（MI 选题）与参数拟合。

---

## 1. 方法与检索记录

### 1.1 检索渠道与可达性

| 渠道 | 状态 | 使用情况 |
|---|---|---|
| arXiv API (export.arxiv.org) | 前期可用，中段 429/空响应约 20 分钟，后恢复 | 约 10 次查询（BKT/KST/DINA/CD-CAT/前置/KT 深度学习） |
| Semantic Scholar API | 全程重度 429（无 key），仅按 DOI 单条查询偶尔成功 | 约 5 次单条 DOI 查询成功 |
| OpenAlex API | 中期高效（约 25 次 works 查询），后期触发 429 | 主力检索源 |
| Crossref API | 全程可用 | 约 8 次书目核实 |
| webfetch（arXiv abs 页） | 可用 | 核实 AKT/SAINT+/simpleKT/QIKT 的 arXiv ID（纠正 1 处错误 ID 记忆） |
| webfetch（ar5iv 全文） | 可用 | 读取 Doignon & Falmagne KST 章节全文 |
| curl（Zenodo） | 可用 | 下载 van de Sande 2013 全文 PDF |
| webfetch（Springer 页面） | 可用 | 获取 Templin & Bradshaw 2013 完整摘要 |
| Google Scholar / 百度学术 | 不可达（transport error / 403） | 记录为检索受限 |
| CNKI / 万方 | 未尝试直连（已知需机构权限）；中文文献仅通过 Crossref/OpenAlex 索引间接覆盖 | 记录为检索受限 |

### 1.2 全文阅读（证据等级 [全文] 的来源）

1. van de Sande, B. (2013). Properties of the Bayesian Knowledge Tracing Model. JEDM 5(2). —— PDF 全文（Zenodo 3554630）。
2. Doignon, J.-P., & Falmagne, J.-C. (2015). Knowledge Spaces and Learning Spaces.（New Handbook of Mathematical Psychology 章节）arXiv:1511.06757 —— ar5iv HTML 全文。

其余论文证据等级为 [摘要]（读到了出版方摘要）或 [题名]（书目信息核实无误但未见摘要）。

---

## 2. 六个对抗性问题的文献回答

### Q1. 短预算 + 可解释 + 冷启动 + 零真人数据下，什么测量模型最优？

**逐族裁决（证据支撑见括号）：**

1. **深度 KT 族（DKT Piech 2015 / DKVMN Zhang 2017 / AKT Ghosh 2020 / SAINT Choi 2020 / SAINT+ Shin 2021 / simpleKT Liu 2023 / CL4KT Lee 2022 / QIKT Chen 2023）—— 全部排除。**
   - 它们预测"下一题对错"（AUC 竞赛），不输出可解释诊断标签；需要万级交互序列训练，冷启动（0 真人数据）直接不可行 [摘要，各 arXiv 页核实]。
   - Khajah et al. 2016（EDM，arXiv:1604.02416）实证：在真实数据集上 BKT 与 DKT 表现相当 [摘要]。simpleKT 论文自述 DKT 在 ASSISTments 上报告 AUC 从 0.721 到 0.821 波动，基线不可比 [摘要/全文摘要]。另有 2026 预印本（arXiv:2603.02830）发现专用小模型 KT 优于 LLM，且需训练数据——同样不适用于冷启动 [摘要]。
   - 保留价值：Zhou et al. 2024（ICLR, arXiv:2403.13179）与 QIKT 的"结构化 + 可解释"路线提示：把心理测量结构（Rasch/前置图）显式编码进模型比黑箱好，这支持我们用结构化小模型而非深度模型 [摘要]。

2. **IRT 2PL/3PL + CAT —— 排除为主，保留 θ 作为辅助校验。**
   - 需要已校准题库（item 参数需大量真人作答校准），冷启动不可行；单维 θ 不可解释到知识点 [摘要/记忆回溯：IRT 基础教科书中既定事实，等级标 [记忆回溯] 仅限"需校准"这一常识性命题]。
   - 分类目的下文献明确推荐 SPRT/CI 终止规则（Eggen 1999, APM, DOI 10.1177/01466219922031365 [题名+书目核实]）；DCM 分类比 IRT 在同长度下可靠性更高（Templin & Bradshaw 2013 [摘要]）。

3. **CDM 族（DINA de la Torre 2009 / G-DINA 2011 / LCDM / DINO）—— 中期目标，冷启动不可行。**
   - 输出二元属性 mastery 向量，可解释性符合产品需求；DINA 是最简特例（每题 2 个 slip/guess 参数）[摘要：de la Torre 2009 JEBS 34(1):115-130, Crossref 核实]。
   - 硬约束：需要 Q 矩阵（专家构念）+ 已校准题目参数 + 数百人量级作答矩阵做 EM/MCMC；0 真人数据无法上线。Gu & Xu 2017 给出 DINA 可辨识的充要条件（arXiv:1711.03174）[摘要]，进一步说明参数不可随意手设。
   - 若未来题库完成校准，CD-CAT 是文献上"短测 + 分类"的黄金路线：Wang 2013（EPM，MI 选题，专门针对 short test length）[摘要]、Zheng & Chang 2016（APM，短长度 CD-CAT 高效选题）[摘要]。

4. **KST（Doignon & Falmagne 1999 书 / 1985 创始文 / ALEKS）—— 冷启动构念与选题逻辑的模板。**
   - 唯一不需要数值参数校准的家族：知识结构是组合结构（learning space / surmise system），评估是"确定学生在哪个知识状态" [全文：Doignon & Falmagne 2015 章节]。
   - 实证：ALEKS 对 650 题的领域用 25–35 题完成全领域状态评估；状态外缘（outer fringe）的"可学性"预测成功率约 0.93；状态对未考题目作答的预测 phi≈0.43–0.58、tetrachoric≈0.68–0.80（125,786 次评估）[全文，该章节 12 节援引 Cosyn et al. 2013]。
   - 成本：知识空间构建"极其昂贵"（专家 + 海量数据）[全文]。**YHer 可行的近似 = KST-lite：只建"前置关系边"（surmise/atom 的最简形式），不建完整状态格。**
   - 选题规则可直接借用：half-split 提问规则（每次选当前后验下答对概率最接近 0.5 的题）[全文]。

5. **BKT（Corbett & Anderson 1995）—— 最接近 YHer 冷启动约束的模型骨架。**
   - 每 KC 四个参数 L0/G/S/T + 每次作答 Bayes 更新 P(master)，输出可解释的掌握概率 [全文：van de Sande 2013 复述其定义]。
   - 冷启动：文献中 BKT 参数可用默认值先行，再用数据拟合（EM/RSS），且有上下文化 G/S 的改进（Baker et al. 2008 [摘要]）。这正是化学线引擎的实际形态——但化学线多加了三个互斥状态。

**Q1 结论**：文献最优解不存在单一模型，而是**分阶段组合**：
- 冷启动期（当前）：**BKT 式二元 mastery + KST 式前置边 + 错误码表**；选题用 half-split；不确定度显式展示。
- 校准期（有真人数据后）：拟合 G/S/T、上下文化参数（Baker 2008 路线）；引入 SPRT 式可变长度终止（Eggen 1999）。
- 成熟期（题库校准后）：CD-CAT（DINA/G-DINA + MI 选题，Wang 2013 / Zheng & Chang 2016）。

### Q2. 手设似然常数的四状态 Bayes 在文献中的近亲？BKT guess/slip 真实估计范围？gamma_mcq=0.25 是否合理？

**近亲检索结果：**

- 用户提到的 "Smith & Shute 1996 Bayesian student model" **未能检索到**（Crossref/OpenAlex 均无对应条目）。已核实的最近亲：
  - Shute, V. J. (1995). SMART: Student modeling approach for responsive tutoring. UMUAI 5(3):349-383, DOI 10.1007/BF01101800 [题名核实]。
  - Mislevy & Gitomer (1996). The role of probability-based inference in an intelligent tutoring system. UMUAI 6(1):11-46, DOI 10.1007/BF01126112 —— 离散状态上的概率推断（HYDRIVE），是"多状态 + 手设观测模型"的真正先例 [题名核实；内容为记忆回溯：多状态概率更新框架]。
  - Conati, Gertner, VanLehn (1997). On-line student modeling for coached problem solving using Bayesian networks (UM'97) [题名核实]。
  - Villano (1992). Probabilistic student models: BBN + KST [题名核实]。
  - Millán et al. (2010). Bayesian networks for student model engineering. Computers & Education 55(3):1011-1022 [题名核实]。
  - Wang & Beck (2013). Class vs. Student in a Bayesian Network Student Model [题名核实]。
  - 高序 DINA：de la Torre & Douglas (2004). Higher-order latent trait models for cognitive diagnosis. Psychometrika 69:333-353 —— 用高阶连续潜变量约束属性间的相关/层级 [摘要（OpenAlex 条目）+ 记忆回溯]。
- 直接结论：**文献中不存在"四状态互斥 + 全手设似然"的成熟模型**。BKT 是二元 HMM；BBN 学生模型是多元布尔变量（非互斥标签）；DINA 族是二元属性向量。手设似然的做法在 BKT 的"默认参数"传统里有对应，但 BKT 文献同时反复警告手设参数的陷阱：
  - Beck & Chang (2007, UM'07): 可辨识性问题——不同参数组合拟合相同数据 [题名核实 + van de Sande 全文引证]。
  - Baker, Corbett & Aleven (2008, ITS): 模型退化（empirical degeneracy）——参数违反构念含义时模型行为颠倒 [摘要 + van de Sande 全文引证]。
  - van de Sande (2013, JEDM): 形式化约束 **P(G)+P(S)<1；且 Baker et al. 2008 进一步要求 P(S)<1/2, P(G)<1/2**；违反则 HMM 行为反转（答对反而降低掌握概率）[全文]。

**guess/slip 的实证范围：**
- van de Sande (2013) 图例参数 P(S)=0.05, P(G)=0.3 [全文]。
- 文献常规拟合值区间：slip 约 0.05–0.2、guess 约 0.1–0.4 [记忆回溯：多篇 BKT 拟合报告的常见区间，未逐篇核实——已标注]。
- Baker et al. 2008 主张按题目/情境上下文估计 G/S 而非常数 [摘要]。

**gamma_mcq=0.25 判定：**
- 0.25 恰为四选一随机猜测概率，等于 BKT 里 guess 的"无信息下界"，且满足 P(G)<1/2 约束 [全文约束 + 算术]。
- 落在文献常规 guess 拟合区间（0.1–0.4）中部偏保守端 [记忆回溯区间]。
- **结论：0.25 作为 MCQ 的默认先验合理**；gamma_numeric=0.03 对填空题保守（可能过低——低于随机水平无意义，但数值题猜中率本就≈0，保守无害，只会让后验收敛偏慢）。真正的问题不在 gamma 取值，而在：(a) 常数而非上下文估计（Baker 2008 路线可改进）；(b) 四状态里多个似然常数同时手设导致构念退化（见 Q4）。

### Q3. 分类可靠性需要多少题？

文献不给出单一公式，但证据链清晰：

1. **反面证据（短测危险）**：Emons, Sijtsma & Meijer (2007, Psychological Methods 12(1):105-116) 系统论证短量表对**个体分类的一致性很差** [摘要核实，DOI 10.1037/1082-989X.12.1.105]。这与 YHer 两轮审计发现（P/U 在 3-25 题内 KL=0.0247 不可辨识）方向一致：**状态越多、每题似然对比度越弱，小样本下的分类就越接近抛硬币**。
2. **正面证据（分类比打分省题）**：Templin & Bradshaw (2013, J. Classification 30:251-275)：DCM 在同长度下比 IRT 提供更高的被试估计可靠性，因此**可以缩短测试**或获得更可靠的多维测量 [完整摘要]。其可靠性定义（基于四分类相关/分类准确性指标）可直接移植为 YHer 的上线验收指标。
3. **计算工具**：Cui, Gierl & Chang (2012, JEM 41(1):19-46) 给出认知诊断分类一致性/准确度的渐近指标，可用已校准题库先验估计每题属性的分类质量 [摘要]；Wang et al. (2015, JEM 52:457-476) 属性级与模式级指标 [书目核实]；Johnson & Sinharay (2018, JEM 55:635-664) 一致性度量 [书目核实]。
4. **实测参考点**：ALEKS 对**整个领域**（数百题）用 25–35 题完成状态定位，且状态预测未考题目作答 phi≈0.43–0.58 [全文]。注意这是"全领域粗定位"而非"单 KC 高置信分类"。
5. **Q 矩阵设计**：Madison & Bradshaw (2014, EPM) 显示 Q 矩阵设计显著影响分类准确度与收敛 [摘要]——单 KC 多题的"每 KC 3-15 题"预算下，题目的诊断质量（对 KC 的区分度）比题数本身更关键。
6. **可变长度终止**：SPRT/置信区间法在分类测试中可变长度终止（Eggen 1999 [书目核实]；SPRT 模拟研究 2015 [题名]）。二元 mastery 场景下，YHer 可直接采用"后验概率阈值 + 最小/最大题数"的 SPRT 式规则。

**实用结论**：对**二元** mastery 标签、强先验（板块层次先验）、高质量题目，文献经验区间约为**每题 4–8 题可达 0.85–0.95 级后验置信**（此为文献综述后的工程推断，非单一出处，标注 [记忆回溯+外推]）；但**四状态互斥标签在 3–15 题内无解**（本身上述 Emons/可辨识性文献预示，YHer 审计已证）。

### Q4. 四状态互斥 vs 二元 mastery + 前置边，哪个构念有文献支持？

**二元 mastery + 前置结构有压倒性文献支持：**
- KST：知识状态 = 已掌握题目类型集合；前置关系由 surmise 系统/atom 刻画（掌握 b 必须掌握 a 等）[全文]。
- AHM：Leighton, Gierl & Hunka (2004, JEM 41:205-237) 属性层级方法——属性间的层级前置约束 [书目核实，DOI 10.1111/j.1745-3984.2004.tb01163.x]。
- HO-DINA：高序潜变量约束属性相关（de la Torre & Douglas 2004 [摘要]）。
- 深度学习线同样吸收前置图：Prerequisite Attention Model (CIKM 2022) [摘要]、RPKT (arXiv:2508.11892) [摘要]、Zhou et al. 2024 [摘要]——前置结构是跨模型族的共识构念。
- "前置缺口"作为**诊断动作**有心理学依据：KST 的外缘（outer fringe）预测"可学性"成功率 0.93 [全文]；AHM 在数学（Gierl, Alves & Taylor Majeau 2010, IJT [题名]）与代数（Gierl et al. 2008 [题名]）上落地。**但文献中"前置缺口"是前置 KC 的 mastery 为 0/低的推论，不是与"掌握"并列的独立状态**。

**四状态互斥标签（M/P/C/U）无文献支持：**
- CDM 输出二元属性向量（多 KC 各一维），KST 输出状态集合，BKT 输出每个 KC 的掌握概率——没有家族把"掌握/前置缺口/解题链不稳/证据不足"作为单一 KC 上互斥的四个状态 [各模型定义层面，[全文]/[摘要] 综合]。
- "证据不足"（U）在测量文献中是**不确定性度量**（后验方差、分类一致性/准确度、SPRT 的 continue-testing 区），不是与被测构念并列的状态 [SPRT 文献 + Cui 2012 摘要综合]。
- "概念/解题链不稳"（C）没有直接对应物；最近的替代构念是**错误类型/错误码**（见 Q5）和 BKT 里的 slip 高估——即"掌握但答错"与"未掌握但蒙对"的混合。
- 审计已证实 P/U 不可辨识（KL=0.0247）——这与 Beck & Chang 2007 的可辨识性论证、van de Sande 2013 的约束分析一致：状态空间膨胀 + 手设似然 = 后验由先验和常数决定，而非数据。

### Q5. 数学学科特定的测量传统

- **错误分析传统（强，可复用）**：
  - Cox, L. S. (1975). Systematic errors in the four vertical algorithms in normal and handicapped populations. JRME 6(4):202-220 —— 系统错误模式（借位/进位算法错误）的经典实证 [摘要核实，DOI 10.5951/jresematheduc.6.4.0202]。
  - Ashlock, R. B. *Error Patterns in Computation*（1976 半程序化版；1993/2001/2006 修订版）—— 计算错误模式诊断法，直接对应 YHer 的"错误码表" [题名核实多版本]。
  - Radatz, H. (1979). Error analysis in mathematics education. JRME 10(3):163-172 —— 基于信息加工机制的错误分类（语义、结构、算法等成因）[摘要核实]。
- **认知诊断在数学的落地**：AHM 用于小学数学（Gierl et al. 2010, IJT）与 SAT 代数（Gierl et al. 2008）[题名]；Tatsuoka 分数减法数据集是 DCM 的标准基准（Cui et al. 2012 即用该数据）[摘要]。
- **认知导师传统**：Ritter et al. (2007) Cognitive Tutor 数学应用 [van de Sande 参考文献确认存在]。
- **中文文献（检索受限，诚实记录）**：CNKI/万方不可达；通过 Crossref 核实到涂冬波团队 2023 年发布 flexCDMs 认知诊断数据分析平台（Chinese/English Journal of Educational Measurement and Evaluation, DOI 10.59863/vtip9358）[题名核实]；涂冬波、蔡艳、丁树良《认知诊断理论、方法与应用》及其中文 CD-CAT 论文系列**未能直接检索到原文**，不作引用。

### Q6. 小样本下哪种模型参数可估计性最好？

1. **BKT**：4 参数/技能。可辨识性受限（Beck & Chang 2007 [题名+van de Sande 引证]）；EM 拟合易退化（Baker et al. 2008 [摘要]）；有解析约束（van de Sande 2013 [全文]）；EM 收敛路径需人工导航（Pardos & Heffernan 2010 [van de Sande 参考文献确认]）；个体化 BKT（Yudelson et al. 2013 [题名]）。**小样本下 EM 不稳，但"默认参数 + 在线 Bayes 更新"不需要拟合，是唯一零数据可行方案**。
2. **DINA/G-DINA**：DINA 每题 2 参数；可辨识性有充要条件（Gu & Xu 2017 [摘要]）；小样本专门研究：DINA-BAG（Arthur & Chang 2023, JEBS [摘要]）与神经网络估计（2013, J. Classification [题名]）都说明**数百人以下样本 DINA 参数估计不稳**；Q 矩阵设计影响收敛（Madison & Bradshaw 2014 [摘要]）。
3. **KST**：组合结构无数值参数，冷启动成本在结构构建而非估计 [全文]。参数最省的家族。
4. **深度 KT**：需要万级序列 [各论文实验设置，[摘要] 综合]。

**排序（冷启动可估计性）**：KST-lite > BKT 默认参数 Bayes > DINA（需校准）> IRT（需校准）> 深度 KT（需大数据）。

---

## 3. 候选模型对比表

| 模型 | 数据需求 | 可解释性 | 短测可靠性证据 | 冷启动（0 真人数据） | 对 YHer 数学 MVP 推荐度 |
|---|---|---|---|---|---|
| 当前四状态 Bayes | 无（手设） | 表面高（四标签），构念效度无 | 负证据：审计 KL=0.0247 不可辨识 | 能跑但构念无支撑 | 不保留（改造） |
| BKT（Corbett & Anderson 1995） | 默认参数可用；拟合需每技能数百次作答 | 高（每 KC 掌握概率） | 依赖 G/S 设置与题质；与 DKT 表现相当（Khajah 2016 [摘要]） | **可行**：L0=层级先验、G/S=科目默认 | **核心骨架，采纳** |
| KST/ALEKS（Doignon & Falmagne） | 结构构建昂贵；无数值校准 | 最高（知识状态 + fringe） | 25–35 题定位全领域状态；外缘可学性 0.93 [全文] | **可行**：只建前置边（KST-lite） | **前置图与选题逻辑，采纳** |
| DINA（de la Torre 2009） | Q 矩阵 + 校准（数百人量级） | 高（属性 mastery 向量） | DCM 比 IRT 同长度更可靠（Templin & Bradshaw 2013 [摘要]） | 不可行（无校准） | 成熟期目标 |
| G-DINA/LCDM/DINO | 更高（更多参数） | 高 | 同上 | 不可行 | 成熟期备选（LCDM 为 G-DINA 特例） |
| IRT-2PL/3PL + SPRT | 题库校准 | 中（θ 不可解释到 KC） | SPRT 可变长度（Eggen 1999 [题名]） | 不可行 | 辅助校验用 θ |
| DKT（Piech 2015） | 万级序列 | 低 | — | 不可行 | 排除 |
| DKVMN（Zhang 2017） | 万级序列 | 低 | — | 不可行 | 排除 |
| AKT（Ghosh 2020） | 万级序列 | 中（注意力可读，仍非诊断） | — | 不可行 | 排除 |
| SAINT/SAINT+ | 万级序列 | 低 | — | 不可行 | 排除 |
| simpleKT / CL4KT / QIKT | 万级序列 | 低-中（QIKT 最好） | — | 不可行 | 排除（QIKT 的 IRT 预测层思想可借鉴） |

---

## 4. 对当前四状态设计的裁决与最小改造方案

### 裁决：改造（保留引擎骨架，重构状态空间）

依据：四状态互斥标签在文献中无近亲（Q2/Q4）、不可辨识（审计 + Beck & Chang 2007 传统）、且浪费了文献明确支持的两个构念（前置结构、错误类型）。

### 最小改造方案（按优先级）

1. **状态空间：四标签 → 每 KC 二元 P(master) ∈ [0,1] + 不确定度元标签**
   - 保留现有 Bayes 更新管线，但每个 KC 只跟踪 P(master)。答对似然：P(correct|master)=1-slip（slip 默认 0.1）；P(correct|¬master)=guess（MCQ 0.25 / 数值 0.03 保留，标注为待校准默认值）。
   - 满足 van de Sande 约束检验：G+S<1、G<0.5、S<0.5 自动成立 [全文]。
   - "证据不足"（U）改为**后验不确定度元标签**：当 P(master) 落在接近先验的带状区间（如 |P−0.5|<ε 或后验区间宽度 > 阈值）且题数 < 下限时，输出"不确定，需再测"，触发 SPRT 式继续测试（Eggen 1999 [题名]）——U 不再参与似然竞争。
2. **前置缺口（P）→ 前置图边 + 前置 KC 的 mastery**
   - 建每 KC 的前置边表（surmise/atom 最简形式 [全文]）。诊断结果 = 目标 KC 的 P(master) + 其前置 KC 的 P(master) 列表。"前置缺口"作为诊断输出保留（路由前置视频/练习），但它由前置 KC 自己的概率推导，不占用目标 KC 的状态空间。
3. **概念/解题链不稳（C）→ 错误码表 + 模式指标**
   - 数学线可用学科传统（Cox 1975 系统错误 [摘要]、Ashlock 错误模式 [题名]、Radatz 1979 错误分类 [摘要]）建立每题级错误码表；自由题 LLM 判分时同时标注错误码。
   - "不稳"信号 = 同 KC 内答对/答错交替 + slip 后验估计高 + 错误码一致性差，作为推荐"看视频回炉"的触发条件，而非状态标签。
4. **选题：half-split**
   - 每次选当前后验下 P(correct)≈0.5 的题（KST 提问规则 [全文]），替代纯难度投影。这是 KST/ALEKS 与 CD-CAT MI 法的共同直觉（信息量最大化）。
5. **参数路线**：冷启动用手设默认（现状）+ 层级先验接线（板块 0.6 + 全局 0.4，已实现未接线——本方案明确建议接线）；有真人数据后：(a) 按 Baker et al. 2008 上下文化 G/S [摘要]；(b) 用 Cui et al. 2012 分类一致性/准确度指标做上线验收 [摘要]；(c) 题库校准完成后升级 CD-CAT（Wang 2013 MI 选题 [摘要]）。
6. **保留件**：deferred/均匀似然不更新的处理（与"不确定度元标签"合并即可）；四标签在 UI 层的展示可以保留（M=高概率掌握、P=前置低概率、C=错误码/模式触发、U=不确定元标签）——**改的是推断层，不是展示层**。

---

## 5. 未决问题清单（留待真人数据）

1. slip/guess 的科目级真实值：需按 Baker et al. 2008 方法用真人作答拟合；gamma_mcq=0.25、gamma_numeric=0.03 仅是默认值。
2. 每 KC 达到目标后验置信（如 0.9）所需题数分布：上线后按 Cui et al. 2012 指标估计属性级分类准确度。
3. 前置边表的构念效度：由谁构建（教师/LLM+教师复核）、多少个前置关系、是否允许 AND/OR 结构（surmise 一般形式 [全文]）。
4. 错误码表的覆盖率与判分者间一致性（LLM 判分错误码的可靠性，需双评实验）。
5. 层级先验权重（0.6/0.4）的敏感性。
6. "不确定"区间的阈值设定（SPRT 型 α/β 与 indifference zone 宽度）。
7. DINA/LCDM 升级路径的 Q 矩阵来源（现题库是否支持每题标注所需 KC 集）。
8. 中文 CD-CAT 文献补检（CNKI 检索受限，涂冬波团队的中文方法论需后续人工补读）。
9. 自由题 LLM 判分错误对后验的污染（判分错误≠作答错误，需判分可靠性校正项）。
10. 时间/时延数据是否并入（SAINT+ 表明时序特征有用 [摘要]，但需要真实作答日志）。

---

## 6. 参考文献表（均已通过检索核实；等级标注）

**[全文]**
1. van de Sande, B. (2013). Properties of the Bayesian Knowledge Tracing Model. Journal of Educational Data Mining, 5(2), 1–10. DOI 10.5281/zenodo.3554629.
2. Doignon, J.-P., & Falmagne, J.-C. (2015). Knowledge Spaces and Learning Spaces. Chapter in The New Handbook of Mathematical Psychology. arXiv:1511.06757.

**[摘要]**
3. Baker, R. S., Corbett, A. T., & Aleven, V. (2008). More Accurate Student Modeling through Contextual Estimation of Slip and Guess Probabilities in Bayesian Knowledge Tracing. ITS 2008, LNCS 5091. DOI 10.1007/978-3-540-69132-7_44.
4. Templin, J., & Bradshaw, L. (2013). Measuring the Reliability of Diagnostic Classification Model Examinee Estimates. Journal of Classification, 30(2), 251–275. DOI 10.1007/s00357-013-9129-4.
5. Cui, Y., Gierl, M. J., & Chang, H.-H. (2012). Estimating Classification Consistency and Accuracy for Cognitive Diagnostic Assessment. Journal of Educational Measurement, 49(1), 19–38. DOI 10.1111/j.1745-3984.2011.00158.x.
6. Wang, C. (2013). Mutual Information Item Selection Method in Cognitive Diagnostic Computerized Adaptive Testing With Short Test Length. Educational and Psychological Measurement, 73(6), 1017–1035. DOI 10.1177/0013164413498256.
7. Zheng, C., & Chang, H.-H. (2016). High-Efficiency Response Distribution–Based Item Selection Algorithms for Short-Length CD-CAT. Applied Psychological Measurement, 40(8), 608–624. DOI 10.1177/0146621616665196.
8. Arthur, D., & Chang, H.-H. (2023). DINA-BAG: A Bagging Algorithm for DINA Model Parameter Estimation in Small Samples. Journal of Educational and Behavioral Statistics. DOI 10.3102/10769986231188442.
9. Madison, M. J., & Bradshaw, L. (2014). The Effects of Q-Matrix Design on Classification Accuracy in the Log-Linear Cognitive Diagnosis Model. Educational and Psychological Measurement, 75(3), 491–511. DOI 10.1177/0013164414539162.
10. Gu, Y., & Xu, G. (2017). The Sufficient and Necessary Condition for the Identifiability and Estimability of the DINA Model. arXiv:1711.03174 (另见 Psychometrika 2020 版).
11. Khajah, M., Lindsey, R. V., & Mozer, M. C. (2016). How Deep is Knowledge Tracing? EDM 2016. arXiv:1604.02416.
12. de la Torre, J., & Douglas, J. A. (2004). Higher-Order Latent Trait Models for Cognitive Diagnosis. Psychometrika, 69(3), 333–353. DOI 10.1007/BF02295640.
13. Cox, L. S. (1975). Systematic Errors in the Four Vertical Algorithms in Normal and Handicapped Populations. Journal for Research in Mathematics Education, 6(4), 202–220. DOI 10.5951/jresematheduc.6.4.0202.
14. Radatz, H. (1979). Error Analysis in Mathematics Education. Journal for Research in Mathematics Education, 10(3), 163–172. DOI 10.5951/jresematheduc.10.3.0163.
15. Emons, W. H. M., Sijtsma, K., & Meijer, R. R. (2007). On the Consistency of Individual Classification Using Short Scales. Psychological Methods, 12(1), 105–116. DOI 10.1037/1082-989X.12.1.105.
16. Lee, W., Chun, J., Lee, Y., Park, K., & Park, S. (2022). Contrastive Learning for Knowledge Tracing (CL4KT). WWW 2022. DOI 10.1145/3485447.3512105.
17. Liu, Z., Liu, Q., Chen, J., Huang, S., & Luo, W. (2023). simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing. ICLR 2023. arXiv:2302.06881.
18. Chen, J., Liu, Z., Huang, S., Liu, Q., & Luo, W. (2023). Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations (QIKT). AAAI 2023. arXiv:2302.06885.
19. Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-Aware Attentive Knowledge Tracing (AKT). KDD 2020. arXiv:2007.12324.
20. Zhou, H., Bamler, R., Wu, C. M., & Tejero-Cantero, Á. (2024). Predictive, Scalable and Interpretable Knowledge Tracing on Structured Domains. ICLR 2024. arXiv:2403.13179.

**[题名核实（书目信息确认，未见摘要或仅部分信息）]**
21. Corbett, A. T., & Anderson, J. R. (1995). Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge. UMUAI, 4(4), 253–278. DOI 10.1007/BF01099821.
22. Piech, C., et al. (2015). Deep Knowledge Tracing. NeurIPS 2015. arXiv:1506.05908.
23. Zhang, J., Shi, X., King, I., & Yeung, D.-Y. (2017). Dynamic Key-Value Memory Networks for Knowledge Tracing. WWW 2017. DOI 10.1145/3038912.3052580.
24. Choi, Y., et al. (2020). Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing (SAINT). L@S 2020. DOI 10.1145/3386527.3405945.
25. Shin, D., Shim, Y., Yu, H., Lee, S., Kim, B., & Choi, Y. (2021). SAINT+: Integrating Temporal Features for EdNet Correctness Prediction. LAK 2021. DOI 10.1145/3448139.3448188.
26. de la Torre, J. (2009). DINA Model and Parameter Estimation: A Didactic. JEBS, 34(1), 115–130. DOI 10.3102/1076998607309474.
27. de la Torre, J. (2011). The Generalized DINA Model Framework. Psychometrika, 76(2), 179–199. DOI 10.1007/s11336-011-9207-7.
28. Doignon, J.-P., & Falmagne, J.-C. (1985). Spaces for the Assessment of Knowledge. International Journal of Man-Machine Studies, 23(2), 175–196. DOI 10.1016/S0020-7373(85)80031-6.
29. Falmagne, J.-C., & Doignon, J.-P. (1999). Knowledge Spaces. Springer.
30. Leighton, J. P., Gierl, M. J., & Hunka, S. M. (2004). The Attribute Hierarchy Method for Cognitive Assessment: A Variation on Tatsuoka's Rule-Space Approach. JEM, 41(3), 205–237. DOI 10.1111/j.1745-3984.2004.tb01163.x.
31. Gierl, M. J., Alves, C., & Taylor Majeau, R. (2010). Using the Attribute Hierarchy Method to Make Diagnostic Inferences about Examinees' Knowledge and Skills in Mathematics. International Journal of Testing, 10(4), 318–341. DOI 10.1080/15305058.2010.509554.
32. Wang, W., Song, L., Chen, P., Meng, Y., & Ding, S. (2015). Attribute-Level and Pattern-Level Classification Consistency and Accuracy Indices for Cognitive Diagnostic Assessment. JEM, 52(4), 457–476. DOI 10.1111/jedm.12096.
33. Johnson, M. S., & Sinharay, S. (2018). Measures of Agreement to Assess Attribute-Level Classification Accuracy and Consistency for Cognitive Diagnostic Assessments. JEM, 55(4), 635–664. DOI 10.1111/jedm.12196.
34. Eggen, T. J. H. M. (1999). Item Selection in Adaptive Testing with the Sequential Probability Ratio Test. APM, 23(3), 286–305. DOI 10.1177/01466219922031365.
35. Mislevy, R. J., & Gitomer, D. H. (1996). The Role of Probability-Based Inference in an Intelligent Tutoring System. UMUAI, 6(1), 11–46. DOI 10.1007/BF01126112.
36. Shute, V. J. (1995). SMART: Student Modeling Approach for Responsive Tutoring. UMUAI, 5(3), 349–383. DOI 10.1007/BF01101800.
37. Villano, M. (1992). Probabilistic Student Models: Bayesian Belief Networks and Knowledge Space Theory. UM'92, LNCS 608. DOI 10.1007/3-540-55606-0_58.
38. Millán, E., Loboda, T., & Pérez-de-la-Cruz, J. L. (2010). Bayesian Networks for Student Model Engineering. Computers & Education, 55(3), 1011–1022. DOI 10.1016/j.compedu.2010.07.010.
39. Beck, J. E., & Chang, K. (2007). Identifiability: A Fundamental Problem of Student Modeling. UM 2007, LNCS 4511. DOI 10.1007/978-3-540-73078-1_17.（卷号经 van de Sande 参考文献确认，DOI 未逐条核验）
40. Pardos, Z. A., & Heffernan, N. T. (2010). Navigating the Parameter Space of BKT Models: Visualizations of the Convergence of the EM Algorithm. EDM 2010.
41. Yudelson, M. V., Koedinger, K. R., & Gordon, G. J. (2013). Individualized Bayesian Knowledge Tracing Models. AIED 2013, LNCS 7926. DOI 10.1007/978-3-642-39112-5_18.
42. Ashlock, R. B. (1976 初版；1993/2001/2006 修订). Error Patterns in Computation: Using Error Patterns to Improve Instruction. Merrill/Prentice Hall.（无 DOI；多版本经 OpenAlex 条目核实）
43. Ritter, S., Anderson, J. R., Koedinger, K. R., & Corbett, A. (2007). Cognitive Tutor: Applied Research in Mathematics Education. Psychonomic Bulletin & Review, 14(2), 249–255.
44. Wang, Y., & Beck, J. (2013). Class vs. Student in a Bayesian Network Student Model. AIED 2013. DOI 10.1007/978-3-642-39112-5_16.
45. Tu, D., & Gao, X. (2023). flexCDMs：认知诊断数据分析平台. Chinese/English Journal of Educational Measurement and Evaluation. DOI 10.59863/vtip9358.
46. Tang, J., et al. (2025). RPKT: Learning What You Don't — Recursive Prerequisite Knowledge Tracing in Conversational AI Tutors. arXiv:2508.11892.
47. Deng, Y., et al. (2025). Adaptive Knowledge Transfer for Cross-Disciplinary Cold-Start Knowledge Tracing. arXiv:2511.20009.
48. Bhattacharyya, P., et al. (2026). Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs. arXiv:2603.02830.

**[未核实，仅记录]**
- "Smith & Shute (1996) Bayesian student model"：用户引用，多源检索未找到，不纳入论证。
- 涂冬波团队中文 CD-CAT 论文与《认知诊断理论、方法与应用》专著：存在性可信（团队有英文与平台产出），但 CNKI 不可达，本轮未引用。
- 常规 BKT 拟合区间（slip 0.05–0.2，guess 0.1–0.4）与"每属性 4–6 题"经验法则：多篇文献常见但未逐篇核验，标记为 [记忆回溯]。

---

## 检索局限声明

- Semantic Scholar API 全程 429 限流，仅零星按 DOI 查询成功；检索覆盖率低于计划。
- OpenAlex 后期触发 429，个别补充检索（如 BKT 小样本专项、中文 CD-CAT 专项）未完成，结论基于已获取文献。
- CNKI/万方/百度学术/Google Scholar 均不可达，中文文献仅覆盖 Crossref/OpenAlex 索引的条目，存在明显缺口。
- arXiv API 中段出现约 20 分钟空响应窗口，该窗口内查询未计入证据。
- 部分经典书（Knowledge Spaces 1999、Error Patterns in Computation、Rupp/Templin/Henson 2010）仅书目核实，未读原文。
