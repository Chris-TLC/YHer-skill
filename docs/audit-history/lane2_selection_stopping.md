# Lane 2：自适应选题与停止规则 —— 文献检索与最优解判定

> 日期：2026-08-13 ｜ 任务性质：只读研究（未改任何项目代码）
> 上游输入：YHer 数学单节点诊断现设计（EIG 选题 + gap>0.45 & direct>=3 停止 + 四档预算 + 1.5× 诚实暂停）、前两轮审计（random≈EIG、PWKL 同族、gap=0.45 无据、direct>=3 为主要结果混杂来源）
> 与本报告互补的既有文件：`PROJECT_HANDOFF/CAT_SELECTION_STOPPING_LITERATURE_REVIEW_2026-08-05.md`（其 227 行条目级清单是本报告的引用底账，本报告不再重复其全部条目，只补充新检索与裁决）

## 1. 检索记录

| 通道 | 状态 | 备注 |
|---|---|---|
| arXiv API (export.arxiv.org) | 可用（https） | 无鉴权；部分复杂 query 语法报 Invalid；结果噪声大，需 title/author 限定 |
| Crossref REST (api.crossref.org) | 可用，成功率最高 | 主通道：按 DOI 取全文摘要，共验证 60+ DOI |
| Semantic Scholar API | 部分可用 | 带 UA 时 DOI 端点可达；search 端点几乎全部 429，仅 1/10 查询成功（Rafferty 2019 经此确认） |
| OpenAlex | 检索受限（全程 429） | 未获得任何结果 |
| ERIC API (api.ies.ed.gov) | 可用 | 补齐 Thompson 系列、He&Reckase 2014、Patton 2013、Demir 2022 等条目的出版信息 |
| Google Scholar / DuckDuckGo / Bing | 检索受限 | 均被反爬拦截或返回噪声 |
| CNKI / 万方 | 检索受限 | 中文文献经 Crossref（心理学报、心理科学进展、CEJME 有 DOI 收录）间接验证；CNKI 原文未读 |

新检索获得的关键新条目（相对 08-05 综述的增量）：
- **Thompson, N.A. (2008/2009). Item Selection in Computerized Classification Testing. EPM, 68(6).** DOI 10.1177/0013164408324460 —— 解释 random≈EIG 现象的核心文献（见问题 1）。
- **van der Linden (2011). Setting Time Limits on Tests. APM.** DOI 10.1177/0146621610391648 —— 30 分钟预算设计的直接方法论文（问题 6）。
- **van der Linden & Xiong (2013). Speededness and Adaptive Testing. JEBS.** DOI 10.3102/1076998612466143；van der Linden (2008) Predictive Control of Speededness in AT；van der Linden, Scrams & Schnipke (1999) RT 约束选题 —— 时间侧约束的完整文献线（问题 6）。
- **Thompson, N.A. (2011). Termination Criteria for Computerized Classification Testing. PAR/E, 16(4)**（ERIC EJ933698，题名级）；Thompson & Weiss (2011) A Framework for the Development of CAT（PAR/E 16(3)）。
- **REN, HUANG & CHEN (2022). 计算机化分类测验终止规则的类型、特点与应用. 心理科学进展.** DOI 10.3724/sp.j.1042.2022.01168 —— 中文综述，题名级（Crossref 无摘要，CNKI 受限）。
- **李、郑 (2024). 基于二分搜索的非参数 CD-CAT 选题及终止规则（NDBI）. CEJME.** DOI 10.59863/yqmx8617 —— 中文摘要已读：变长测验中可调临界值换判准率。
- **Rafferty, Ying & Williams (2019). Statistical Consequences of using Multi-armed Bandits to Conduct Adaptive Educational Experiments. JEDM 11(1).** DOI 10.5281/zenodo.3554749（S2 确认；引 38 次）。
- **Clement, Roy, Oudeyer & Lopes. MAB for ITS**：arXiv:1310.3174（摘要已读）；JEDM 2015 7(2):20-48 出版版为[记忆回溯]。
- **Demir & Atar (2021). CACT 模拟研究. EPOD.** DOI 10.21031/epod.787865（SPRT vs CI × MFI 选题，48 条件模拟）。
- **Eggen & Straetmans (2000). CAT for Classifying Examinees into Three Categories. EPM**（摘要已读）；Eggen (2009) Three-Category ACT 章节。
- **Liu & Weiss (2026). Interactions Between Termination Criteria and Ability Estimators in CAT. EPM.** DOI 10.1177/00131644261453945（固定长度/SEM/MI/Δθ × MLE/WLE/MAP/EAP × 题库形状交互）。
- **He & Reckase (2014). Item Pool Design for an Operational Variable-Length CAT**（ERIC EJ1026121，题名级）；**Demir (2022). The Effect of Item Pool and Selection Algorithms on CCT Performance**（ERIC EJ1352160，题名级）；**Patton, Cheng, Yuan & Diao (2013). Item Calibration Error on Variable-Length CAT**（ERIC EJ1006846，题名级）。
- **Chen & Braeken (2026 preprint). Progressive tests with stopping rules from CAT**（osf e4y5h_v1，摘要已读）—— 固定蓝图+错误数停止 vs CAT 停止规则的直接对比，横跨问题 1/2/6。
- **Guo & Zheng (2019). Termination Rules for VL CD-CAT from the Information Theory Perspective. Frontiers in Psychology.** DOI 10.3389/fpsyg.2019.01122（题名级，Crossref 无摘要）。

## 2. 六个对抗性问题的文献回答

### 问题 1：四状态短预算下，信息量选题是否真的优于固定蓝图/分层随机？random≈EIG 与文献一致吗？

**结论：random≈EIG 与文献一致，但这不代表 EIG 无用；文献的排序是 MI(EIG) ≥ 其他信息量准则 > FI > 随机，且差距大小取决于题库异质性与停止规则。**

- **"各准则无实质差异"的直接证据**：Thompson (2008, EPM) 明确写道——CCT 中多种 IRT 选题算法"no conclusive evidence on the substantial superiority of a single method"，原因是各准则"assess items very similarly through different calculations and will usually select the same item"；且"the efficiency of item selection approaches depend on the termination criteria that are used"。[摘要] 这解释了合成模拟里 random≈EIG：若题库各题对 4 状态的可区分度近似同质，EIG 的排序收益趋零。
- **准则间排序仍是 MI 最优**：Weissman (2007) 四分类模拟：MI > 后验加权 FI > FI，MI 分类正确率最高且测验最短。[摘要] Wang (2013) 短测 CD-CAT：MI 与 KL/PWKL/Shannon 比较，MI 恢复率最高，但准则间差距为"过半数条件"级别，非碾压级。[摘要] Kaplan et al. (2014)：MPWKL≈GDI，"perform very similarly"。[摘要] Hsu & Wang (2022)：除特殊画像外 MER/MPWKL/PWCDI/SHE 表现"similar"。[摘要] Veerkamp & Berger (1997，见 08-05 综述)：多准则表现相近。
- **何时差异才大**：理论上是渐近最优（Tatsuoka 2002/2003 定理；Cheng 2009 的 PWKL 即其启发式实现[摘要]），但有限长度下收益打折；Liu, Ying & Zhang (2015) 的 rate-function 选题"even for moderate length tests"有效[摘要]——注意其卖点是渐近误分类率最优，也侧面说明有限样本下普通信息量准则优势有限。差距放大条件：题库质量异质（He & Reckase 2014 强调为变长 CAT 设计题库[题名]）、终止规则匹配（Thompson 2008）、以及带约束的真实组卷场景。
- **自适应 vs 固定长度的证据**：变长 CAT 的公认卖点是"固定长度对不同考生精度不均、变长保证等精度"（Hsu, Wang & Chen 2013[摘要]；Weiss 1982[记忆回溯]）。固定蓝图+事后分类的问题是精度不均衡与超支（对低能力者无效题多）；分层随机/固定蓝图在四状态诊断下等同于"覆盖率有保证的无信息选择"。**裁决：四状态诊断里，纯 EIG 相对随机蓝图的中位收益本来就小（每道二元题对 4 状态的上界 ≤1 bit，且单节点题库内同质性强）；EIG 的真正价值在配合覆盖约束（问题 4）与异构题库时体现。不应因 E2 实验废弃 EIG，也不应期待它在同质合成库里打赢随机。**

### 问题 2：可变长度停止的最优实践；"top1−top2 gap" 的文献对应物

**结论：文献主流是"序贯置信判据 + 最短/最长长度约束"，判据用后验概率阈值（sequential Bayes）或似然比界（SPRT/mGLR）。gap 形式无已发表校准，但数学上是后验支配判据的线性变换；固定长度+事后分类是次优形态。**

- SPRT 线：Wald 1945[记忆回溯/经典]；Eggen 1999（SPRT 与选题一体）[摘要]；Spray 1993 多类 SPRT[题名]；Spray & Reckase 1996——**SPRT 比 sequential Bayes 用更少题达到同错误率**（两类）[摘要]；Stefan et al. 2022——SPRT 与序贯 BF 数学同构[摘要]；多类推广：Wang, Chen & Huebner 2020 提出 mGLR + 随机截尾，"shorter average test length without sacrificing classification accuracy"[摘要]；Bartroff, Finkelman & Lai 2008/2011 序贯 GLR 渐近最优[摘要]。
- 后验阈值线（YHer gap 所在家族）：Spray & Reckase 1996 的 sequential Bayes；变长 CD-CAT 实现 Hsu et al. 2013（固定精度终止）[摘要]、Huebner, Finkelman & Weissman 2018（分类精度×平均长度权衡因素）[题名]、Guo & Zheng 2019[题名]、李&郑 2024 NDBI[摘要]。
- **gap 判据的定位**：gap>0.45 ⇔ P(top1)≥P(top2)+0.45；若把 top2 近似为 1−P(top1)，等价于 P(top1)≥0.725。文献的阈值形式是 max 后验 ≥ τ（常见 τ≈0.7–0.9，见 08-05 综述 3.7/3.9 条目），或 Bayes factor 界（Kass & Raftery 1995：2lnBF>6 强证据）。**gap 形式没有文献直接用，但同族；0.45 数值无依据，须模拟校准。**
- 稳定类判据的警示：Wang, Weiss & Shang (2019)——CT（θ 变化）规则单独使用"不稳定、会过早终止"[摘要]；Babcock & Weiss 2012 比较固定长度 vs 各类变长准则[题名，内容经 Wang et al. 2019 摘要转述]；Liu & Weiss 2026——终止准则与估计方法、题库形状存在交互，最优配置条件依赖[摘要]。**推论：把 gap 判据改写成 P(top1)≥τ（或 BF 界）后仍需模拟校准 τ；不要引入"连续若干题 gap 稳定"类规则。**
- 固定长度+事后分类 vs 阈值+置信：文献一致认为变长/序贯在等精度、平均长度与决策保证上占优（Spray & Reckase 1996；Hsu 2013；Babcock & Weiss 2012 题名级结论"efficient and effective"）。Chen & Braeken 2026（preprint）把 CAT 停止规则移植到渐进式蓝图测试，反证"固定蓝图+事后判类"被当作需要被 CAT 规则改进的对象[摘要]。**YHer 保持变长是对的；但应把判据换成有理论保证的 τ 或 mGLR。**

### 问题 3：最小题数约束（direct>=3）的文献依据与校准

**结论：最小长度约束是变长测验的标准组件，文献依据充分；但其数值一律由模拟校准，无理论定值。3 题对四状态诊断偏小。**

- 标准形态：Babcock & Weiss 2012（最短/最长长度约束）[题名，经 Wang et al. 2019 转述]；Thompson 2011 终止准则综述[题名]；Hsu et al. 2013[摘要]；Sie et al. 2014（置信区间停止 + 截尾，提前停的判据是"继续测改变结论的概率低"）[摘要]。
- 为什么需要最小题数（过早停止的证据）：Wang, Weiss & Shang 2019（CT 规则过早终止）[摘要]；Sun, Liu, Xin & Song 2020（**校准误差→平均测验长度变短、效率被高估**）[摘要]；Patton et al. 2013（同主题，变长 CAT）[题名]；Nájera et al. 2025（小样本/低质量题下可靠性被高估）[摘要]。
- **数值依据**：信息论下界——4 状态完全确定需要 log₂4=2 bit；单道二元题在 4 状态下的最大互信息约 0.5–0.8 bit（依 q 向量），现实题约 0.2–0.4 bit → 理论下界 3 题、实际 5–8 题。文献无统一常值（不同研究用 5、10 等，未形成标准）——**标注[记忆回溯]，无法引单篇**；Kruyen et al. 2012 用"决策质量拐点"定最短可接受长度[题名]。**裁决：保留最小题数机制，数值改为模拟校准（下节协议），预期落在 4–6。**
- **校准协议（可执行，伪代码级）**：
  ```
  输入：题库参数（滑/猜或 IRT）、四状态模型、候选最小题数 m∈{2..8}、后验阈值 τ
  for m in 2..8:
    for τ in 0.60..0.95:
      for 每个模拟考生（10k 次，含模型正确/参数污染/先验污染三场景）:
        逐题作答（含滑/猜噪声）→ 更新后验
        若题数≥m 且 P(top1)≥τ → 停止并判类；否则继续
        记录：判类对错、实际长度、是否恰在第 m 题时判错但后验已"自信"
      指标：整体判准率、ATL、早停误判率（在第 m 题即触发停止且判错的占比）
    反事实对照：同场景下若在第 m 题强制停止（事后分类），判准率是多少 → 画"m vs 判准率"曲线
  选定 m*：早停误判率<1% 且 m* 位于判准率曲线的拐点（Kruyen 2012 决策质量拐点法）
  ```
  注意：早停误判率必须按"先验污染"场景评估（direct 机制就是为防先验污染而设，Sun 2020 已证污染会缩短测验）。

### 问题 4：覆盖约束 vs 纯信息量——对单节点场景的启示

**结论：单节点诊断的"覆盖"= 节点内知识点面/题型面的覆盖；文献证据强烈支持覆盖约束，且 CD-CAT 有专门的平衡准则，shadow test 框架是通用解法。YHer 现有"每个目标节点最低覆盖优先"方向正确，但实现应从"硬优先"改为"约束/加权"。**

- CD-CAT 覆盖线：Cheng 2010 MMGDI——覆盖平衡"improves the validity of the test scores"与画像恢复率[摘要]；Sun, Andersson & Xin 2021 RTA 指标——精度与覆盖兼得[摘要]；Wang, Sun, Chong & Xin 2020（短测属性覆盖）[题名]；Cheng 2009 PWKL 本身即在属性空间穷举[摘要]。
- 变长下的覆盖：Li, Zhang & Chang 2019 look-ahead content balancing——变长 CCT 中"fewer constraint violations and higher classification accuracy"[摘要]；Huo 2009（a-stratified + 内容平衡的变长 CAT）[题名]；Lin 2011（带实际约束的 CCT 选题）[题名]。
- Shadow test 框架：van der Linden 2000/2009/2021——把内容/曝光/时间约束并入每步最优组卷，实时最优且满足规格[摘要（2021 综述）]；vdl & Xiong 2013 用同一框架控制 speededness（±10 秒保证）[摘要]。**对单节点：shadow test 的"整卷组装"思路可简化为"约束集 + 每步 0-1 线性规划"，节点内约束 = 每个 facet 至少 k 题、题族上限、预计总时长上限。**
- 裁决："每族至少一题优先于 EIG"合理（对单节点即"每个知识点面/题型至少一题"），文献形态是 MMGDI 式加权（λ·coverage + (1−λ)·EIG）或约束集，比硬优先级更稳（硬优先会牺牲判准效率，Sun 2021 显示纯 ABI 平衡牺牲精度）。

### 问题 5：诊断与学习交织时的选题——共用选择器吗？

**结论：不共用。诊断阶段用信息量/覆盖率准则（测量模型固定）；学习/练习阶段用学习目标准则（进度/掌握增长/时间预算）；两者共用画像状态，但选择器目标函数不同。文献三线证据：**

- 教学规划线：Rafferty, Brunskill, Griffiths & Shafto (2015/2016) POMDP 教学——最优教学动作是序列级规划而非单步信息量贪心；学生模型假设改变会改变最优教学策略[摘要]。**启示：练习排序用 POMDP/长程目标，诊断用单步信息量，两者逻辑本就不同。**
- Bandit/ITS 线：Clement et al. MAB ITS（arXiv 1310.3174，JEDM 2015）——MAB 选"使学生进步最快的活动"，目标是学习增益而非测量[摘要]；Rafferty, Ying & Williams (2019)——**用 bandit 收集的数据做统计推断有偏差风险**（自适应分配使观测非 iid，估计与检验失真）[题名+摘要级确认出版信息，内容[记忆回溯]]。**这是"诊断不可用 bandit 数据"的关键证据：诊断数据需要干净的测量设计，学习数据天然被选择策略污染，两者混用互相伤害。**
- 诊断式学习系统线：Eggen (2014) "Item Selection in Computerized Adaptive Learning Systems" 章节——CAT 选题在自适应学习系统中的应用[题名]；van Buuren & Eggen (2017) 潜在类别选题用于 progress test（阶段性复测）[题名]；Wang (2020 preprint) interim CD-CAT 学习情境设计[摘要]。共同模式：**评估环节与教学环节分离，评估用测量准则，教学用学习准则，画像统一。**
- 对 YHer：诊断（信息量+覆盖+后验停止）→ 推荐（视频/练习）→ 验证（短重测）的窄闭环本身与文献一致；练习排序可用 bandit/POMDP，但绝不能把练习阶段的行为数据直接喂给诊断模型当先验（Rafferty 2019 的偏差问题 + Sun 2020 的先验污染问题叠加）。

### 问题 6：预算编排——30 分钟档的文献设计与单题耗时建模

**结论：30/60/120/180 四档本身无文献对应值；正确做法是用响应时间模型把分钟预算换算成"概率性题数上限"，并在选题约束中引入时长（van der Linden 线的完整方法）。诊断题与练习题节奏差异有 RT 文献支撑（题目的 time-intensity 参数）。**

- 时间上限设计：**van der Linden 2011 "Setting Time Limits on Tests"（APM）——用 lognormal RT 模型推导"在给定时限前完成测验的概率曲线"，把时限设为先验可接受的超时概率**（这正是 30 分钟档的文献级做法）[摘要]。模型基础：vdl 2006 lognormal RT 模型（考生速度 + 题目时间强度）[摘要]；vdl 2007 速度-精度层次框架[摘要]。
- 自适应测验中的时间约束：vdl, Scrams & Schnipke 1999——RT 预测作 0-1 LP 约束控制差异化 speededness[摘要]；vdl 2008 预测式 speededness 控制[摘要]；vdl & Xiong 2013——shadow test 加 RT 约束，测验时长与参考测验差 <10 秒[摘要]；Finkelman, Kim, Weissman & Cook 2014——**CD-CAT 中把 RT 纳入选题的两个新准则**（诊断+时间双目标）[题名]；Tang 2023（PISA on-the-fly MST 纳入 RT）[题名]。
- 短测验/quiz 设计：Kruyen et al. 2012（以决策质量定最短长度）[题名]；Seo, Choi & Kim 2024（真实考试数据上 SEM=0.3/0.25 停止的比较）[摘要]；Crotts et al. 2013（缩短 MST 的精度损失评估，见 08-05 综述）。
- 诊断 vs 练习节奏：RT 模型将"题目的时间强度"作为题目参数、考生速度作为人的参数[摘要（vdl 2006）]；诊断题（多步推理、需读题时间）time-intensity 高且方差大，练习题（熟练度驱动）时间强度低。**YHer 应按题型分别标定 time-intensity，用 vdl 2011 公式把 30 分钟换成"该档可容纳题数的分布"，再据此设最长题数上限；1.5× 预算诚实暂停在文献中无直接对应，最接近的是 max-length 截断（Babcock & Weiss 2012）与截尾思想（Finkelman 2008），作为工程护栏合理但应作为"极端兜底"而非常态路径。**

## 3. 对比表

| 停止规则方案 | 代表文献 | 所需样本（模拟） | 理论依据 | 对 YHer 适用度 |
|---|---|---|---|---|
| 固定长度 + 事后分类 | Liu & Weiss 2026 基线 | 无需 | 无停止决策风险 | 低：精度不均、无预算弹性 |
| P(top1)≥τ（sequential Bayes） | Spray & Reckase 1996；Hsu et al. 2013 | 校准 τ（模拟 10k 级） | 后验最优决策 | **高（推荐）**：可解释状态输出天然契合 |
| gap=top1−top2>g | 无直接文献（同族 τ） | 同 τ | τ 判据线性变换 | 中：保留显示、不作判据 |
| SPRT/mGLR（似然比界） | Wald 1945；Eggen 1999；Wang, Chen & Huebner 2020 | 定 α/β（无需大样本） | α/β 误差保证 | 中高：多类版 mGLR 可直接用；需连续状态似然 |
| 后验阈值+最小题数 m | Babcock & Weiss 2012；Thompson 2011 | 校准 m | 防过早停止 | **高（推荐）**：替换 direct>=3 |
| 预算截断（max-length） | Babcock & Weiss 2012 | 由 RT 模型换算 | 工程约束 | 高：保留，改为 RT 换算 |
| 截尾/随机截尾 | Finkelman 2008/2009；Sie et al. 2014 | 中 | "继续测改变结论概率低" | 中：可作加速选项 |

| 选题方案 | 代表文献 | 理论依据 | 对 YHer 适用度 |
|---|---|---|---|
| MI/EIG（现方案） | Weissman 2007；Wang 2013 | 期望熵降=互信息，分类导向 | **保留**：分类场景文献首选 |
| PWKL | Cheng 2009 | 后验加权 KL | 同族替代，收益相近 |
| FI（cut-score） | Thompson 2008 | 局部信息 | 不推荐：四状态多峰下劣于 MI |
| 随机 | 基线 | 无 | 仅作基线 |
| EIG+覆盖约束（MMGDI 式） | Cheng 2010；Sun et al. 2021 | 覆盖保障效度 | **改造采用**：加权/约束而非硬优先 |
| Shadow test 约束组卷 | vdl 2000/2021 | 0-1 LP 最优+约束 | 可选：节点内约束多时用 |
| Look-ahead 内容平衡 | Li, Zhang & Chang 2019 | 变长下的约束前瞻 | 可选：变长形态需要 |
| RT 纳入选题 | Finkelman et al. 2014 | 时间-信息双目标 | 可选：预算紧张时 |
| POMDP/RL 选题 | Rafferty 2015；BOBCAT；Deep CAT | 序列级最优 | 暂不：学习阶段排序可探索，诊断阶段不必 |

## 4. 裁决（针对被质疑的四条设计）

| 现行设计 | 裁决 | 依据与操作 |
|---|---|---|
| **gap>0.45** | **改造** | 判据改为 P(top1)≥τ（文献家族 Spray&Reckase 1996 / Hsu 2013），gap 保留为 UI 显示量；τ 与 m 联合模拟校准（协议见下）。0.45 数值废弃。 |
| **direct>=3** | **保留并校准** | 机制 = 文献标准最小长度约束（Babcock & Weiss 2012）；数值 3 偏小，按校准协议改（预期 4–6）；名称改为 min_length。这是论文主要结果混杂源 → 论文中应报告"阈值×最小题数×覆盖率"全网格，不再把单一配置当主结果。 |
| **EIG + 覆盖优先** | **保留并改造** | EIG=MI 是分类场景文献推荐（Weissman 2007）；random≈EIG 与 Thompson 2008 的解释一致（同质题库下准则选同一题），不构成废弃理由；覆盖优先改为 MMGDI 式加权 λ·EIG+(1−λ)·coverage 或约束集，λ 纳入校准网格。 |
| **预算 1.5× 暂停** | **保留（降级为兜底）** | 文献对应 max-length 截断（Babcock & Weiss 2012）；无 1.5× 的直接文献；改用 vdl 2011 RT 方法把 30/60/120/180 分钟换算为概率性题数上限作为主约束，1.5× 仅作极端兜底且必须保留诚实提示。 |

### 模拟校准协议（伪代码级，可执行）

```
# 目标：产出 (τ*, m*, λ*, N_max(预算档)) 一组操作参数
输入:
  pool   = 该节点题库（滑/猜参数或 IRT 参数 + q 向量/题型面 + RT 参数[time-intensity]）
  states = 4 状态先验分布
  budgets= {30,60,120,180} 分钟
  N_max(b) = RT 模型(vdl 2011)解出的"第 95 百分位考生在 b 分钟内完成的题数"

场景（关键：停止规则必须跨场景稳健）:
  S0 = 模型正确
  S1 = 题目参数污染（校准误差 σ_e ∈ {0.05, 0.15}）   # Sun 2020: 污染→过早停
  S2 = 先验污染（错分学生画像的概率 p_init ∈ {0.1,0.3}） # direct>=3 的防御对象
  S3 = 时间压力（考生速度参数整体右移）

候选:
  停止:  T1=P(top1)≥τ, τ∈{0.60,0.65,...,0.95}
         T2=T1 + min_length m∈{2..8}
         T3=mGLR(α=0.05,β=0.05)   # 对照
  选题:  A1=random(基线)  A2=blueprint(题型面轮转, 基线)  A3=EIG  A4=λ·EIG+(1−λ)·coverage, λ∈{0.3,0.5,0.7,1.0}  A5=PWKL(对照)

for s in {S0..S3}:
  for (T,A) in 候选网格:
    simulate(10_000 考生):
      循环: 选下一题 → 按状态×滑猜生成作答 → 更新后验 → 判停
    记录: 画像判准率 acc_profile(s,T,A)
          单状态判准率 acc_state
          ATL 均值 / p95
          早停误判率 = P(在第 m 题触发停止 ∧ 判错)
          预算超支率 = P(实际时长 > 档位)
          覆盖率 = 每题型面出现 ≥1 次的考生占比

选参:
  (τ*,m*) = argmax acc_profile
            s.t. 早停误判率<1% (全部场景), 预算超支率<1% (对应档位)
  λ*     = argmax acc_profile(S0) 且不损害 S1/S2 稳健性（差距<2pt）
输出: 各档位下的 (τ*,m*,λ*,N_max) 及四条曲线（判准率×ATL 拐点图, 按 Kruyen 2012 决策质量法复核）
```

执行提醒：校准必须在**真实题库参数**上跑（合成参数会重演 E2 的 random≈EIG 假象）；报告必须包含 S1/S2 场景，否则停止规则在校准误差与先验污染下的表现不可知。

## 5. 未决问题

1. **无真人数据**：所有校准依赖题库参数估计与模拟模型假设；滑/猜与 RT 参数本身未标定。上线后需用真数据回验（Seo et al. 2024 是"真实数据复核停止规则"的范本）。
2. **τ 与 m 的联合校准未见文献给过四状态场景的推荐值**——需自行跑协议；文献阈值（0.7–0.9）来自二类/连续分类场景。
3. mGLR 需要连续状态空间内的似然构造；YHer 的四状态离散模型下 mGLR 与 P(top1)≥τ 的等价性需要推导或模拟确认（Wang, Chen & Huebner 2020 的 mGLR 是连续 θ 多切点）。
4. **中文文献（涂冬波团队 CD-CAT 停止规则）未取得原文**：CNKI 受限，仅有 Crossref 元数据（REN 2021/2022、GUO 2015、李&郑 2024 摘要）——若需逐条引用请开放 CNKI 通道。
5. Babcock & Weiss 2012、Kingsbury & Weiss 1983、Thompson 2007(PAR/E 12(1)) 全文/摘要未取得（JCAT/书籍章节），关键结论经二手摘要转述，证据等级已相应降级。
6. Rafferty 2019 的具体结论（bandit 数据偏误的量化）仅确认出版信息与主题，内容为[记忆回溯]，引用前建议核读 JEDM 全文。
7. "练习阶段是否也用 bandit 排序"超出本 lane 范围，若进入推荐器设计需另开 lane（涉及 Rafferty 2015 POMDP vs bandit vs 规则式 mastery learning 的取舍）。

## 6. 参考文献表（本报告新增/重核条目；其余见 08-05 综述）

### 选题与准则
1. Weissman, A. (2007). Mutual Information Item Selection in Adaptive Classification Testing. *EPM*, 67(2). DOI:10.1177/0013164406288164 [摘要]
2. Thompson, N. A. (2008). Item Selection in Computerized Classification Testing. *EPM*, 68(6). DOI:10.1177/0013164408324460 [摘要]
3. Wang, C. (2013). MI Item Selection in CD-CAT With Short Test Length. *EPM*, 73(6). DOI:10.1177/0013164413498256 [摘要]
4. Cheng, Y. (2009). When Cognitive Diagnosis Meets CAT: CD-CAT. *Psychometrika*, 74. DOI:10.1007/s11336-009-9123-2 [摘要]
5. Cheng, Y. (2010). MMGDI: Balancing Attribute Coverage in CD-CAT. *EPM*, 70. DOI:10.1177/0013164410366693 [摘要]
6. Sun, Andersson & Xin (2021). Balance Measurement Accuracy and Attribute Coverage in CD-CAT. *APM*. DOI:10.1177/01466216211040489 [摘要]
7. Kaplan, de la Torre & Barrada (2014). New Item Selection Methods for CD-CAT. *APM*. DOI:10.1177/0146621614554650 [摘要]
8. Liu, Ying & Zhang (2015). A Rate Function Approach to CAT for Cognitive Diagnosis. *Psychometrika*. DOI:10.1007/s11336-013-9395-4 [摘要]
9. Hsu & Wang (2022). Reducing the Misclassification Costs of CD-CAT: MER. *APM*. DOI:10.1177/01466216211066610 [摘要]
10. Eggen, T. J. H. M. (1999). Item Selection in Adaptive Testing with the SPRT. *APM*, 23. DOI:10.1177/01466219922031365 [摘要]
11. Eggen & Straetmans (2000). CAT for Classifying Examinees into Three Categories. *EPM*. DOI:10.1177/00131640021970862 [摘要]
12. He & Reckase (2014). Item Pool Design for an Operational Variable-Length CAT. (ERIC EJ1026121) [题名]
13. Demir, S. (2022). The Effect of Item Pool and Selection Algorithms on CCT Performance. (ERIC EJ1352160) [题名]

### 停止规则
14. Wald, A. (1945). Sequential Tests of Statistical Hypotheses. *Ann. Math. Stat.* [记忆回溯/经典]
15. Spray & Reckase (1996). Comparison of SPRT and Sequential Bayes Procedures... *JEBS*. DOI:10.3102/10769986021004405 [摘要]
16. Wang, Chen & Huebner (2020). Stopping Rules for Multi-Category CCT. *BJMSP*. DOI:10.1111/bmsp.12202 [摘要]
17. Wang, Weiss & Shang (2019). Variable-Length Stopping Rules for Multidimensional CAT. *Psychometrika*. DOI:10.1007/s11336-018-9644-7 [摘要]
18. Babcock & Weiss (2012). Termination Criteria in CAT. *JCAT*. DOI:10.7333/1212-0101001 [题名]
19. Thompson, N. A. (2011). Termination Criteria for CCT. *PAR/E*, 16(4). (ERIC EJ933698) [题名]
20. Hsu, Wang & Chen (2013). Variable-Length CAT Based on CDMs. *APM*. DOI:10.1177/0146621613488642 [摘要]
21. Finkelman (2008). Stochastic Curtailment to Shorten the SPRT. *JEBS*. DOI:10.3102/1076998607308573 [摘要]；Finkelman (2009). *APM*. DOI:10.1177/0146621609336113 [摘要]
22. Sie, Finkelman, Bartroff & Thompson (2014). Stochastic Curtailment in Adaptive Mastery Testing. *APM*. DOI:10.1177/0146621614561314 [摘要]
23. Bartroff, Finkelman & Lai (2008). Modern Sequential Analysis and Its Applications to CAT. *Psychometrika*, 73; arXiv:1106.2559 [摘要]
24. Choi, Grady & Dodd (2010). A New Stopping Rule for CAT (PSER). *EPM*. DOI:10.1177/0013164410387338 [摘要]
25. Luo, Kim & Dickison (2017). Projection-Based Stopping Rules for CAT in Licensure Testing. *APM*. DOI:10.1177/0146621617726790 [摘要]
26. Liu & Weiss (2026). Interactions Between Termination Criteria and Ability Estimators in CAT. *EPM*. DOI:10.1177/00131644261453945 [摘要]
27. Sun, Liu, Xin & Song (2020). Impact of Item Calibration Error on Variable-Length CD-CAT. *Front. Psychol.*. DOI:10.3389/fpsyg.2020.575141 [摘要]
28. Nájera, Sorrel, Chiu & Abad (2025). Variable-Length CD-CAT in Small-Scale Assessments. *JEBS*. DOI:10.3102/10769986251366581 [摘要]
29. Demir & Atar (2021). CACT Simulation Study (SPRT vs CI × MFI). *EPOD*. DOI:10.21031/epod.787865 [摘要]
30. Stefan et al. (2022). SPRT vs Sequential Bayes Factor Test. *BRM*. DOI:10.3758/s13428-021-01754-8 [摘要]
31. Chen & Braeken (2026 preprint). Progressive Tests With Stopping Rules From CAT. osf:e4y5h_v1 [摘要]
32. REN, HUANG & CHEN (2022). 计算机化分类测验终止规则的类型、特点与应用. 心理科学进展. DOI:10.3724/sp.j.1042.2022.01168 [题名]
33. 李 & 郑 (2024). 基于二分搜索的非参数 CD-CAT 选题策略及终止规则. CEJME. DOI:10.59863/yqmx8617 [摘要]

### 覆盖/约束
34. van der Linden (2021). Review of the Shadow-Test Approach. *Behaviormetrika*. DOI:10.1007/s41237-021-00150-y [摘要]
35. Li, Zhang & Chang (2019). Look-Ahead Content Balancing in Variable-Length CCT. *BJMSP*. DOI:10.1111/bmsp.12165 [摘要]

### 教学/学习选择器
36. Rafferty, Brunskill, Griffiths & Shafto (2015). Faster Teaching via POMDP Planning. *Cognitive Science*. DOI:10.1111/cogs.12290 [摘要]
37. Rafferty, Ying & Williams (2019). Statistical Consequences of Using MABs to Conduct Adaptive Educational Experiments. *JEDM*, 11(1). DOI:10.5281/zenodo.3554749 [题名+出版信息]
38. Clement et al. (2015). MAB for Intelligent Tutoring Systems. *JEDM*, 7(2).（arXiv:1310.3174 [摘要]；期刊版信息[记忆回溯]）

### 时间/预算
39. van der Linden (2011). Setting Time Limits on Tests. *APM*. DOI:10.1177/0146621610391648 [摘要]
40. van der Linden (2006). A Lognormal Model for Response Times on Test Items. *JEBS*. DOI:10.3102/10769986031002181 [摘要]
41. van der Linden (2007). A Hierarchical Framework for Modeling Speed and Accuracy. *Psychometrika*. DOI:10.1007/s11336-006-1478-z [摘要]
42. van der Linden, Scrams & Schnipke (1999). Using RT Constraints to Control Differential Speededness in CAT. *APM*. DOI:10.1177/01466219922031329 [摘要]
43. van der Linden (2008). Predictive Control of Speededness in Adaptive Testing. *APM*. DOI:10.1177/0146621607314042 [摘要]
44. van der Linden & Xiong (2013). Speededness and Adaptive Testing. *JEBS*. DOI:10.3102/1076998612466143 [摘要]
45. Finkelman, Kim, Weissman & Cook (2014). CDMs and CAT: Two New Item-Selection Methods That Incorporate Response Times. *JCAT*. DOI:10.7333/1412-0204059 [题名]
46. Kruyen, Emons & Sijtsma (2012). Test Length and Decision Quality. *IJT*. DOI:10.1080/15305058.2011.643517 [题名]

---
*证据等级说明：[全文]=读到正文；[摘要]=API/出版社提供摘要或网页全文摘要；[题名]=仅确认题名与出版信息；[记忆回溯]=经典知识或未取到原文的表述，引用前需复核。本报告无[全文]级条目（未订阅全文），全部为[摘要]及以下。*
