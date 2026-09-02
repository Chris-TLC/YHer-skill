# Lane 3: 记忆与复习调度——文献检索与最优解判定

- 日期：2026-08-13
- 性质：只读研究（未改任何项目代码）
- 检索者：Claude（决策/审查层）；本报告为 architecture_research_v3 三车道之一
- 证据等级标注：`[全文]` / `[摘要]` / `[题名]` / `[记忆回溯]` / `[工程文档]` / `[检索受限]`

---

## 1. 检索记录

| 通道 | 状态 | 产出 |
|---|---|---|
| arXiv API（export.arxiv.org） | 部分受限（首轮批量后 429 限流；DAS3H、KT-forgetting 变体查询成功；THLR/LFKT/FSRS 作者查询最终未过限流） | DAS3H 1905.06873 `[摘要]`；KT 遗忘变体一批（MemoryKT 2508.08122、RKT 2008.12736、Forgetting-aware Linear Bias 等）`[题名/摘要]` |
| Semantic Scholar API | 部分受限（间歇 429，成功 ~1/3 查询） | DAS3H 论文元数据（EDM 2019, c100）、Schuetze 2025 IJAIED（session 间遗忘）`[题名/摘要]` |
| Crossref API | 正常 | 心理学/教育学全部核心论文逐条核实（Cepeda 2006/2008、Rowland 2014、Adesope 2017、Roediger & Karpicke 2006、Karpicke & Blunt 2011、Dunlosky 2013、Bahrick 1984/1993、Bahrick & Hall 1991、Cooper 1996、Settles & Meeder 2016、Pavlik & Anderson 2005/2008、Reddy 2016、Tabibian 2019、Upadhyay 2021 npj、Rohrer & Taylor 2007、FSRS KDD 2022、TKDE 2023 DOI）`[题名/全文元数据]` |
| GitHub 文档（webfetch/curl） | 部分受限（raw 偶发 429/连接失败，重试成功） | SRS-Benchmark README `[全文]`；FSRS The-Algorithm wiki（v1–v6 全部公式与默认参数）`[全文]`；fsrs-vs-sm15 README `[全文]`；Research-resources 清单 `[全文]`；The-Optimal-Retention wiki `[全文]` |
| THLR / LFKT 专名检索 | **检索受限** | arXiv、Crossref、S2 均未找到名为 THLR 的模型；LFKT 无正式命名模型（详见 Q2/Q3） |

**总体局限**：① arXiv/S2 限流导致部分查询未完成；② 心理学元分析均非 OA，只能核实题名/期刊/引用数，具体效应量依赖 `[记忆回溯]` 并已显式标注；③ 未找到"高中数学学科专属遗忘曲线"的一手研究（见未决问题）。

---

## 2. 六个问题的文献回答

### Q1. FSRS 版本状态、默认参数、同行评审现状

**版本状态** `[全文：SRS-Benchmark README 2026-03 快照]`：
- FSRS-4.5（2023）：17 参数，改遗忘曲线形状（DECAY=−0.5, FACTOR=19/81）。
- FSRS-5（2024）：19 参数，利用**同日多次复习**数据修正下一次预测。
- FSRS-6：21 参数，遗忘曲线平坦度可优化（DECAY 成为参数 w20=0.1542）。
- FSRS-7：35 参数，支持小数间隔、8 参数复杂遗忘曲线；"preset/deck" 分区机制（"若 preset 复习数太少则回退到用户级参数"）。
- 关键点：**YHer 当前公式 R(t)=(1+t/9S)^(−1) 恰好是 FSRS-v4（2022）的遗忘曲线**（v4: DECAY=−1, FACTOR=1/9），已被 4.5 替换；4.5 曲线"前段陡、尾部平"，长间隔尾部保留率明显高于 v4（例：S=10 时 t=50S 处 R_v4≈0.15 vs R_4.5≈0.28）`[全文+自行计算]`。

**SRS-Benchmark 证据** `[全文]`：9,999 用户 / 3.5 亿次复习，TimeSeriesSplit，三指标。不含同日复习：FSRS-5 log loss 0.3560 > FSRS-4.5 0.3624 > FSRS-v4 0.3726 > DASH 0.3682 > ACT-R 0.4033 > HLR 0.4694 > Ebisu 0.4989；FSRS-7 default param.（不训练）0.3629。同日复习计入时 FSRS-4.5(0.4286) 反而略好于 FSRS-5(0.4565)（因其未用同日数据训练，口径不同）。结论：**4.5/5 对 4 与 HLR 有明确且量化的优势；FSRS 系整体优于 DASH/ACT-R/HLR/Ebisu**。SM-2 不在主榜，另见 fsrs-vs-sm15：16 用户/25.7 万次复习，FSRS-online log loss 0.3812 < SM-15 0.4325 `[全文/工程文档]`（样本小；SM-18 无法导出预测、不可比）。SM-2 本身无任何同行评审论文，只有 SuperMemo 文档 `[工程文档]`；SM-2 常数（EF 初始 2.5、区间 1.3–2.5、失败 −0.2、重启 1 天）来自 1987–90 年 Wozniak 自述，无实验发表 `[记忆回溯+工程文档]`。

**"四组 preset"**：FSRS-4.5 官方只有**一组** 17 参数默认值（训练自 Anki 用户群）。多组 preset 是 FSRS-7/benchmark 的机制，4.5 无官方四组 preset `[全文]`。**"FSRS 无同行评审"已过时**：KDD 2022（Ye, Su, Cao，DOI 10.1145/3534678.3539081，c14）与 IEEE TKDE 2023（Su, Ye, Nie, Cao, Chen，DOI 10.1109/TKDE.2023.3251721，c13）均为同行评审论文 `[Crossref 核实]`；但二者是建模/仿真研究而非 RCT。最接近的 RCT 是 Upadhyay et al. 2021（npj Science of Learning，DOI 10.1038/s41539-021-00105-8）：大规模随机实验证实机器学习调度（MaiMemo 系，FSRS 前身 SSP-MMC 同源）比基线显著提高记忆效率 `[题名+研究资源页全文]`——即"算法族有 RCT 级验证，FSRS-4.5 本身没有"。

**是否直接把 FSRS 默认参数当 YHer 常数**：默认参数=在 Anki 背卡用户（语言/医学为主）上的先验，**可以且应当作为无数据冷启动常数**，但必须意识到两点：① 域偏移（高中生数学练习 vs 单词卡）未被任何文献校准过；② 一旦积累到自己的复习日志应立即按用户/知识点拟合。

### Q2. 遗忘建模：内嵌知识状态 vs 读取时投影

- **内嵌遗忘的 KT 模型是 EDM 主流**：BKT-f（在 BKT 转移矩阵加 forget 参数）、DAS3H（连续记忆强度+时序特征，EDM 2019，c100）、PFA+decay（Pavlik & Anderson 2005 Cognitive Science + 2008 JEP:Applied 最优练习调度）、DKT+forgetting（Nagatani WWW 2019 等，检索到 [题名]）、Schuetze/Yan/Carvalho 2025 IJAIED「Capturing Session-to-Session Dynamics of Learning and Forgetting」明确测试跨 session 遗忘下 KT 模型的极限 `[摘要]`。证据强度：DAS3H 在 3 个真实数据集上优于无遗忘基线（EDM 2019）`[摘要]`；BKT-f 有 25 年系统综述（UMUAI 2024，DOI 10.1007/s11257-023-09389-4）`[题名]`。
- **"遗忘只从 M 流向 C"的文献对应**：与 BKT-f 的遗忘转移（L→U，即已掌握→未掌握单向）同构，这是标准假设而非异端；但"P/U 永久不变、无向上学习流"没有直接文献支持——BKT-f 里学习流（U→L）与遗忘流并存。YHer 的 M→C 单向投影在"遗忘方向"上有依据，在"封锁反向学习流"上是纯工程假设 `[摘要/记忆回溯]`。
- **读取时投影 vs 存储**：文献对此无立场（是存储工程而非模型科学）。真正的风险不在"何时算"，而在**事件是否落盘**：若复习事件（成败、间隔、当时 R）不持久化，未来永远无法拟合参数。结论：**投影式读取可以保留，但必须持久化每个知识点的 (S, D, 上次复习时间) 三元组 + 全量复习事件日志**，否则三年画像没有可校准数据。
- 神经网络方案（GRU/LSTM/RWKV）在 benchmark 上 log loss 更低（GRU 0.3333、RWKV-P 0.2773）`[全文]`，但需要预训练+每用户微调、不可解释、依赖 PyTorch——对无数据 MVP 不适用。

### Q3. DAS3H/HLR 多因素 vs 单参数 FSRS；数学学科专属曲线

- DAS3H 公式：logistic 回归，输入为每技能的三类时间特征——练习次数（repetition）、距上次练习时长（lag）、每次练习间隔的 Σ1/ln(c+d)（spacing），且各技能参数独立（α, β, γ per skill）`[摘要+全文公式转述]`。它在"多技能+遗忘"上优于 AFM/PFA 等（EDM 2019，3 数据集）`[摘要]`。
- 但**单卡状态机（FSRS）实测碾压多因素回归（HLR）**：SRS-Benchmark 中 HLR（3 参数，含时间特征）log loss 0.4694 vs FSRS-4.5 0.3624 `[全文]`。原因：HLR 用全局特征回归，FSRS 维护逐条目状态转移（S/D），对间隔增长有上限、对失败有 graceful degradation。DAS3H 未直接进该榜，但其特征工程思路已被 HLR 代表。
- 结论：**多因素不必然优于状态机**；DAS3H 的价值在"技能层结构"，恰是 YHer 的知识点结构。正确做法不是弃 FSRS 换 DAS3H，而是**把 FSRS 的状态粒度放在知识点上**（每知识点一个 S/D），诊断/验证事件就是复习事件。
- **数学专属遗忘曲线：检索受限**。未找到数学学科一手遗忘曲线研究；可用代理证据：Bahrick & Hall 1991（高中数学内容数十年保持，见 Q6）、Cooper 1996（暑假数学成绩下滑，c558）`[题名]`、Rohrer & Taylor 2007（数学题 spacing 效应，Instructional Science，c245）`[题名]`。

### Q4. 复习触发判据；验证环节是否计入调度

- **触发阈值**：FSRS 官方默认 desired retention=0.9，wiki「The Optimal Retention」论证 0.8–0.9 区间内工作量最小（0.9 默认、0.8 低负担）`[全文]`。R<0.8 触发无文献硬依据，但落在 FSRS 允许区间下限，对"低频接触的高中生"合理。
- **检索练习/测试效应**：Roediger & Karpicke 2006（Psych Sci）测试优于重学 `[题名]`；Rowland 2014 元分析（Psych Bull，c835）测试 vs 重读效应显著 `[题名]`；Adesope 2017 元分析（Rev Educ Res，c423）实践测试显著优于再学习 `[题名]`；Karpicke & Blunt 2011（Science）检索优于概念图精细学习 `[题名]`；Dunlosky 2013（PSPI，c2305）把 practice testing 列为高证据技术 `[题名]`。Bjork 的 desirable difficulty 框架支持"延迟到接近遗忘点再练" `[记忆回溯，书籍章节未逐一核实]`。
- **held-out 验证必须计入复习调度**：测试本身就是最高证据强度的学习事件（上述元分析），且它是 YHer 唯一有"结果标签"的事件——不记账等于白扔数据。实现：验证通过 = 一次 Good 复习（更新 S）；失败 = Again（S 走失败公式）。同时验证通过可直接触发 S0 初始化（比 P(M)>0.7 的软判据更干净）。

### Q5. 稳定性增长常数与校准样本量

- **SM-2 ease factor**：初始 2.5，可行区间 1.3–2.5，失败 −0.2 并重置 `[工程文档/记忆回溯]`。
- **FSRS 的稳定性增长不是常数乘数**：SInc = 1 + e^{w8}(11−D)·S^{−w9}·(e^{w10(1−R)}−1)。用 4.5 默认参数（D=5）计算 `[全文+自行计算]`：
  - 按时复习（R=0.9）：S=3.7 时 SInc≈1.57；S=10 时 ≈1.50；S=100 时 ≈1.36。
  - 逾期复习（R=0.5, S=3.7）：SInc≈4.6（spacing effect 内建：越晚复习成功、增益越大，但有上限）。
  - 失败：S' = w11·D^{−w12}·((S+1)^{w13}−1)·e^{w14(1−R)}；S=3.7 时 ≈1.42 天；S=30 时 ≈4.5 天；S=100 时 ≈7.6 天（高稳定度项目失败后的 graceful degradation）。
  - 对照 YHer 现值：**×3 首复（3→9）超出按时复习全区间（≤1.6）；×2 常规略偏大（1.36–1.57）；×0.5 失败对低 S 项目偏宽松、对高 S 项目严重过度惩罚**（S=30 时 ×0.5→15 vs FSRS→4.5，后者允许"忘过一次的熟题快速捡回"）。三者均无文献依据，属拍脑袋常数，**必须替换**。
- **校准最小样本量**：检索受限。已知：benchmark 对"review 数太少的 preset 回退用户级参数"有工程处理 `[全文]`；社区惯例（Anki FSRS 文档）为每用户约 400–1000 次复习才建议拟合 17 参数 `[记忆回溯，本轮未能打开原文档核实]`。YHer 每知识点事件稀疏，建议阈值从宽（每知识点 ≥30–50 事件、每用户 ≥400 事件再拟合，在此之前用默认参数）。

### Q6. 三年画像与跨年级先修知识

- **超长间隔保持**：Bahrick & Hall 1991（JEP: General，c99）：高中数学内容（代数/几何）在毕业后**前 3–5 年快速下降，随后进入"permastore"平台期数十年稳定在可测水平** `[题名+记忆回溯]`；Bahrick 1984 西语 50 年同理（前 3–6 年衰减后平台化，c292）`[题名]`；Bahrick et al. 1993 证实 spacing 对多年保持的贡献 `[题名]`。
- **建模含义**：三年画像不能按指数衰减到 0——应使用带尾部平台的曲线（FSRS-4.5 幂律曲线尾部显著高于 v4 指数式衰减，恰好方向正确），并考虑给已掌握知识点设**保留率下限（floor）**（如 R 不低于 0.3–0.4 的 permastore 平台）或对 S 设上限衰减。Cepeda 2006（c1344）/2008（c449）：间隔效应可延伸至约一年级 RIs，最优间隔约为保持期的 10–20%（长 RI 取下端）`[题名+记忆回溯]`——即三年内的复核间隔应随目标期拉长，而不是固定 2.25S。
- **初中先修知识**：应进 profile 且应衰减，但**不应归零**（Bahrick & Hall 直接证据：学校教过的数学长期保留显著）。操作建议：入场诊断通过的知识点用 Easy 档初始稳定性（S0=13.82 天，即 FSRS-4.5 w3）而非 Good 档（3.71 天），并配 floor。

---

## 3. 对比表

| 方案 | 数据需求 | 证据强度 | 工程代价 | 对 YHer 适用度 |
|---|---|---|---|---|
| **FSRS-4.5 默认参数** | 0（冷启动可用）；未来按用户拟合需每用户 ≥~400 事件 `[记忆回溯]` | 最强工程验证（SRS-Benchmark 9,999 用户/3.5 亿复习 `[全文]`）；算法族有 KDD/TKDE 同行评审 + npj RCT `[题名]`；4.5 本身无 RCT | 低：17 参数闭式公式，纯 Python 数十行 | **高**：无同日复习场景下 4.5 与 5 差距小（0.3624 vs 0.3560），公式最简单 |
| FSRS-5 | 同上 | 略优于 4.5（同日复习数据） | 中：需记录同日多次复习 | 低：YHer 每 session 每知识点只有一次诊断/验证，同日复习机制基本用不上 |
| SM-2 | 0 | 最弱：无同行评审、无 benchmark 表现、仅 1987–90 文档 `[工程文档]` | 低 | 低：常数×EF 线性增长已被 SRS-Benchmark 类证据淘汰；仅作历史对照 |
| DAS3H | 需大量带技能标签+时间戳练习日志拟合 logistic 回归 | 中等：EDM 2019 优于 AFM/PFA `[摘要]`；但同类特征回归 HLR 被 FSRS 大幅碾压 `[全文]` | 中：每知识点 3 特征+回归权重 | 中：技能层结构契合知识点，但预测校准差 FSRS 一个量级；可借鉴其"per-skill 参数"思想 |
| THLR | 未知 | **检索受限：未检索到该命名模型**（arXiv/Crossref/S2 均无）。若指 HLR 的时序扩展，证据可参考 HLR：3 参数、log loss 0.4694，弱于 FSRS-4.5 `[全文]` | — | 待定（需澄清术语来源） |
| BKT-f | 需事件日志拟合转移参数 | 中等：25 年传统，综述见 UMUAI 2024 `[题名]` | 中：EM 拟合 | 低-中：二值状态+遗忘转移与 YHer M/C 结构同构，但无间隔预测能力（不直接给 due date），需叠加遗忘曲线 |
| **独立投影（当前设计）** | 0（读时计算） | 无独立文献支持；风险在事件不落盘 `[摘要级推理]` | 低 | 中：可作为"显示层"保留，但必须补事件日志+S/D 状态存储 |

**总体裁决方向**：采用 **FSRS-4.5 公式 + 默认参数冷启动，状态落在知识点粒度，事件全量落盘**；投影降级为读取视图。

---

## 4. 裁决（逐条）

| 现设计 | 裁决 | 依据 |
|---|---|---|
| S0 = 3.0 | **替换**为 S0(Good)=3.7145；入场诊断/先修知识通过 → S0(Easy)=13.8206 | FSRS-4.5 默认参数 w2/w3 `[全文]` |
| R(t)=(1+t/9S)^(−1) | **替换**为 4.5 曲线 R=(1+(19/81)(t/S))^(−0.5) | 现公式是已被 4.5 取代的 v4 曲线；4.5 长尾更贴近 Bahrick 平台化证据 `[全文+题名]` |
| 首复 ×3 | **替换**为 S'=S·(1+1.0315·(11−D)·S^(−0.1367)·(e^{1.0461(1−R)}−1))，D 固定 5；按时首复 SInc≈1.57，逾期更多 | 4.5 默认 w8/w9/w10 `[全文+自行计算]`；×3 超出按时复习全区间 |
| 常规复 ×2 | **替换**为同上公式（S=10 时 ≈1.50、S=100 时 ≈1.36） | 同上 |
| 失败 ×0.5 | **替换**为 S'=2.1072·D^(−0.0793)·((S+1)^{0.3246}−1)·e^{1.587(1−R)}；S=3.7→≈1.4、S=30→≈4.5（graceful） | 4.5 默认 w11–w14 `[全文+自行计算]`；×0.5 对高 S 项目惩罚过重 |
| M→C 单向投影 | **有条件保留**：遗忘方向与 BKT-f 同构，但改为"事件落盘 + 每知识点存 (S, D, last_review_ts)"，R 读取时算；不写 S/D 则三年内无法校准 | Q2 分析 |
| R=0.8 触发 | **保留**，但统一用单一函数计算：due = last_review + (81/19)·S·(0.8^(−2)−1) ≈ 2.40S（4.5 曲线下 2.25S 对应 v4 曲线，差值 ~7%，无所谓，关键是单一来源） | The-Optimal-Retention wiki：0.8–0.9 合理 `[全文]` |
| 双复诊日矛盾（7 天 vs 2.25S） | **废除硬编码 7 天**，唯一来源 = due 函数；S0=3.7 时 due≈8.9 天（4.5 曲线），与 7 天接近说明 7 是历史凑数 | 本报告 |
| 复核插入 planner 未接线 | 接线时把 held-out 验证结果回写为复习事件（通过=Good，失败=Again） | 测试效应文献（Q4） |
| S0 触发条件（P(M)>0.7 且 direct≥2 或 held-out 通过） | 记忆层与产品层解耦：记忆层只在"有带标签的复习事件"时更新 S/D；P(M) 阈值留给推荐/诊断逻辑 | 分层原则，避免把画像门控与记忆参数纠缠 |

### 无数据时最稳妥参数集（全部来自 FSRS-4.5 官方默认参数，逐项可溯源）

```text
遗忘曲线：      R(t,S) = (1 + (19/81)·(t/S))^(-0.5)          [4.5 曲线，R(S)=0.9]
初始稳定性：    S0 = 3.71 天（Good/通过）；13.82 天（Easy/先修入场通过）
通过后更新：    S' = S·(1 + 1.0315·(11−D)·S^(-0.1367)·(e^(1.0461·(1−R)) − 1))，D≡5
失败后更新：    S' = 2.1072·D^(-0.0793)·((S+1)^0.3246 − 1)·e^(1.587·(1−R))
复核 due：      due = last_review + (81/19)·S·(0.8^(-2) − 1)  ≈ last_review + 2.40·S 天
保留率下限：    R_floor ≈ 0.35（permastore 平台，工程近似，依据 Bahrick & Hall 1991）
持久化：        每知识点 (S, D, last_review_ts) + 每次诊断/验证事件日志（KP, outcome, R@review, Δt）
校准门槛：      每用户 ≥400 事件、每知识点 ≥30 事件后才允许拟合；此前永远用上表
```

依据标注：FSRS-4.5 公式与 17 参数 `[全文：awesome-fsrs/wiki/The-Algorithm]`；平台化 `[题名：Bahrick & Hall 1991]`；0.8 触发 `[全文：The-Optimal-Retention]`；校准门槛 `[记忆回溯，需核实]`。

---

## 5. 未决问题

1. **THLR 术语无法定位**：需要用户提供该缩写的出处；若无出处，建议从评审清单中移除或替换为 HLR（Settles & Meeder 2016）。
2. **数学学科专属遗忘曲线缺失**：未检索到数学学科一手曲线研究（检索受限，不排除存在中文文献未覆盖）。是否用 Bahrick & Hall 1991 参数校准 4.5 曲线为开放问题。
3. **校准最小样本量未核实**：400 事件的社区惯例本轮未能打开原始文档核实（GitHub 限流）；建议后续打开 Anki FSRS 官方文档确认。
4. **R_floor 数值是工程近似**：Bahrick 平台化是群体平均（代数约 30–40%），个体差异大；建议 MVP 阶段 floor 作为保守参数并 A/B 观察。
5. **每知识点 vs 每用户粒度**：FSRS 假设卡级状态；YHer 是知识点级。跨知识点迁移（同知识点不同题目共享 S）无文献直接支持，需产品数据验证。
6. **D 固定为 5**：放弃难度维是简化；若未来按知识点拟合，应启用 D 更新（4.5 的 w5–w7）。
7. 心理学元分析的效应量细节未逐一核实（非 OA），引用层面止于题名/期刊/被引数；如需强引用建议走机构下载全文。

---

## 6. 参考文献表

**FSRS / 调度算法**
1. Ye, Su, Cao (2022). A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling. ACM KDD 2022, 4381–4390. DOI 10.1145/3534678.3539081 `[Crossref 核实]`
2. Su, Ye, Nie, Cao, Chen (2023). Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory. IEEE TKDE. DOI 10.1109/TKDE.2023.3251721 `[Crossref 核实]`
3. open-spaced-repetition. SRS-Benchmark README（FSRS v1–v7 对比，9,999 用户/3.5 亿复习）. github.com/open-spaced-repetition/SRS-Benchmark `[全文]`
4. open-spaced-repetition. FSRS The-Algorithm wiki（各版本公式与默认参数）. github.com/open-spaced-repetition/awesome-fsrs/wiki/The-Algorithm `[全文]`
5. open-spaced-repetition. fsrs-vs-sm15 README（16 用户/25.7 万复习，FSRS 0.3812 < SM-15 0.4325）. `[全文]`
6. Upadhyay, Lancashire, Moser, Gomez-Rodriguez (2021). Large-scale randomized experiments reveals that machine learning-based instruction helps people memorize more effectively. npj Science of Learning 6. DOI 10.1038/s41539-021-00105-8 `[Crossref 核实]`
7. Settles & Meeder (2016). A Trainable Spaced Repetition Model for Language Learning (HLR). ACL 2016. DOI 10.18653/v1/P16-1174 `[Crossref 核实]`
8. Reddy et al. (2016). Unbounded Human Learning. KDD 2016. DOI 10.1145/2939672.2939850 `[Crossref 核实]`
9. Tabibian et al. (2019). Enhancing human learning via spaced repetition optimization. PNAS 116(10). DOI 10.1073/pnas.1815156116 `[Crossref 核实]`
10. Zaidi et al. (2020). Adaptive Forgetting Curves for Spaced Repetition Language Learning. AIED 2020. DOI 10.1007/978-3-030-52240-7_65 `[题名]`
11. Wozniak. SuperMemo 2 算法文档（supermemo.guru）`[工程文档]`

**知识追踪 / 遗忘建模**
12. Choffin, Popineau, Bourda, Vie (2019). DAS3H: Modeling Student Learning and Forgetting for Optimally Scheduling Distributed Practice of Skills. EDM 2019. arXiv:1905.06873 `[摘要]`
13. Pavlik & Anderson (2005). Practice and Forgetting Effects on Vocabulary Memory. Cognitive Science 29(4). DOI 10.1207/s15516709cog0000_14 `[题名]`
14. Pavlik & Anderson (2008). Using a model to compute the optimal schedule of practice. JEP: Applied 14(2). DOI 10.1037/1076-898X.14.2.101 `[题名]`
15. Pavlik, Cen, Koedinger (2009). Performance Factors Analysis. AIED 2009. DOI 10.3233/978-1-60750-028-5-531 `[Crossref 核实]`
16. Šarić-Grgić et al. (2024). Twenty-five years of Bayesian knowledge tracing: a systematic review. UMUAI. DOI 10.1007/s11257-023-09389-4 `[题名]`
17. Schuetze, Yan, Carvalho (2025). Capturing Session-to-Session Dynamics of Learning and Forgetting: Testing the Limits of Knowledge Tracing Models. IJAIED. DOI 10.1007/s40593-025-00508-3 `[题名]`
18. Nagatani et al. (2019) 类 DKT+forgetting 与 MemoryKT (2025, arXiv:2508.08122) `[题名]`

**认知心理学 / 间隔与检索练习**
19. Cepeda et al. (2006). Distributed practice in verbal recall tasks. Psychological Bulletin 132(3). DOI 10.1037/0033-2909.132.3.354 `[Crossref 核实]`
20. Cepeda, Vul, Rohrer, Wixted (2008). Spacing Effects in Learning. Psychological Science 19(11). DOI 10.1111/j.1467-9280.2008.02209.x `[Crossref 核实]`
21. Roediger & Karpicke (2006). Test-Enhanced Learning. Psychological Science 17(3). `[题名]`
22. Rowland (2014). The effect of testing versus restudy on retention: a meta-analytic review. Psychological Bulletin 140(6). DOI 10.1037/a0037559 `[Crossref 核实]`
23. Adesope, Trevisan, Sundararajan (2017). Rethinking the Use of Tests: A Meta-Analysis of Practice Testing. Review of Educational Research 87(3). DOI 10.3102/0034654316689306 `[Crossref 核实]`
24. Karpicke & Blunt (2011). Retrieval Practice Produces More Learning than Elaborative Studying with Concept Mapping. Science 331. DOI 10.1126/science.1199327 `[Crossref 核实]`
25. Karpicke & Roediger (2008). The Critical Importance of Retrieval for Learning. Science 319. DOI 10.1126/science.1152408 `[Crossref 核实]`
26. Dunlosky et al. (2013). Improving Students' Learning With Effective Learning Techniques. PSPI 14(1). DOI 10.1177/1529100612453266 `[Crossref 核实]`
27. Bjork & Bjork. Desirable difficulties（书籍章节）`[记忆回溯]`

**长期保持 / 数学**
28. Bahrick (1984). Semantic memory content in permastore. JEP: General 113(1). DOI 10.1037/0096-3445.113.1.1 `[Crossref 核实]`
29. Bahrick & Hall (1991). Lifetime maintenance of high school mathematics content. JEP: General 120(1). DOI 10.1037/0096-3445.120.1.20 `[Crossref 核实]`
30. Bahrick et al. (1993). Maintenance of Foreign Language Vocabulary and the Spacing Effect. Psychological Science 4(5). DOI 10.1111/j.1467-9280.1993.tb00571.x `[Crossref 核实]`
31. Cooper et al. (1996). The Effects of Summer Vacation on Achievement Test Scores. Review of Educational Research 66(3). DOI 10.3102/00346543066003227 `[Crossref 核实]`
32. Rohrer & Taylor (2007). The shuffling of mathematics problems improves learning. Instructional Science 35. DOI 10.1007/s11251-007-9015-8 `[Crossref 核实]`

---

*报告完。全部 DOI 经 Crossref API 核实；arXiv/GitHub 内容经直接抓取核实；标 `[记忆回溯]` 处为代理知识、已显式标注并在未决问题中列明核实计划。*
