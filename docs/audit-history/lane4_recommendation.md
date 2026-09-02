# Lane 4：教育资源推荐架构文献检索与最优解判定

> 日期：2026-08-13 | 对象：YHer 数学 MVP 推荐器（Score = W_track × topic_match × type_fit × difficulty_fit × efficacy；TOPK=3；BV,p,segment 粒度；rec_served 审计）
> 性质：只读研究，未改任何产品代码。承接 `architecture_research/w3_recommendation.md`（2026-08-05）与 `MASTER_AUDIT_REPORT_2026-08-05.md` 的 M5 裁决。
> 证据等级约定：[全文]=通读全文；[摘要]=API/页面返回摘要；[题名]=仅核实元数据存在；[记忆回溯]=未取得一手材料。

---

## 1. 检索记录

- **通道**：arXiv API（`https://export.arxiv.org/api/query`，本机 curl 验证 HTTP 200；子代理期间偶发 429 已退避重试，部分改用 arxiv.org HTML 搜索页）；Crossref / ERIC / DBLP API 补位；webfetch arXiv abs 页抽验。
- **Semantic Scholar API：全程 429（三次退避无效），判定「检索受限」**，未采用其 citationCount。
- 关键词组：educational recommender survey、MOOC recommendation、knowledge-aware/KG recommendation（RippleNet/KGCN/KPRN）、sequential/session-based（SASRec/GRU4Rec）、LightGCN、cold start、small catalog、constraint-based recommender、learning to rank、contextual bandit education、off-policy evaluation、IPS/doubly robust、Thompson sampling、deconfounded recommendation、hierarchical Bayesian、lecture segment retrieval、video timestamp、mastery learning、learning path generation、knowledge tracing recommendation。
- 主代理复核：17 个关键 arXiv ID 经 `id_list` 批量核验标题全部一致（1502.02362、1604.00923、1808.06581、1808.09781、1907.06902、2002.02126、1803.03467、2207.14003、2103.12198、2212.06679、1904.11738、2005.09683、1602.05352、1103.4601、1405.7544、2002.00467、1907.09623 等）。
- 未检索到（如实记录）：用户点名的 **"Huang 2019" MOOC 推荐综述在所有可用源未找到**；"small catalog recommendation" 无直接文献（结论由冷启动+小数据经验评估文献合成）；"lecture segment retrieval" 精确短语 arXiv 0 命中（由 TAGV/moment retrieval/videoQA 邻近任务承载）；**Shaikh 2019 exploration 单独论文未核实**（Shaikh 为 Williams 2021、Zavaleta-Bernuy 2022 合著者）；ZEST（Mandel 2014）仅 AAMAS 官方 TOC 确认存在（题名级）。

---

## 2. 六个对抗性问题的文献回答

### Q1. 教育视频/讲座推荐 SOTA 在「30 视频 + 冷启动 + 强约束」下是否劣于规则/内容混合？

**回答：是，且证据方向一致（含模型作者自述）。**

- KG 推荐（RippleNet [摘要] arXiv:1803.03467、KGCN [摘要] arXiv:1904.12575、KPRN [摘要] AAAI 2019）验证集均为 MovieLens-1M/Book-Crossing/Last.FM 级（10⁵–10⁶ 交互、上万 item）；KGCN 摘要明言 "operate on large datasets" 是前提。YHer KG=0 时无信号可传播。
- 序列推荐：SASRec [摘要] arXiv:1808.09781 原文 "MC-based methods perform best in extremely sparse datasets, where model parsimony is critical"——作者本人确认稀疏下省俭模型优先；GRU4Rec [摘要] arXiv:1511.06939 动机场景（小型无登录站点）最接近 YHer，但训练数据是 YOOCHOOSE 百万级会话。
- LightGCN [摘要] arXiv:2002.02126 消融证明 GCN 有效成分只剩线性邻居聚合（非线性甚至有害）——无图规模即无增益。
- 经验反证（最强证据）：Dacrema et al. 2019 [摘要] arXiv:1907.06902：18 个顶会神经 top-N 模型仅 7 个可复现，6 个被简单启发式反超；Ludewig et al. 2020 UMUAI **[全文]**：12 算法×8 数据集（百万级会话），结论 "simple heuristic methods based on nearest-neighbors schemes are preferable"，且 "<5 次交互的 item 被剔除"——30 视频全目录不达训练门槛；Rendle et al. 2020 [摘要] arXiv:2005.09683：调好的点积 MF 优于 MLP（"dot products might be a better default choice"）。
- 教育侧综述：Uddin 2021（IEEE Access）[摘要]、Urdaneta-Ponte 2021 [摘要]、da Silva 2022 [题名]——所收录方法无一小目录/百人级验证；Gulzar 2018（用户所指 Gulzar）[题名, Crossref] 的课程推荐实践路线即"内容+CF+规则混合"，非纯学习模型。冷启动综述（Yuan & Hernandez 2023 [摘要]、Schein 2002 [题名]、Fernández-Tobías 2019 [题名]）一致指向：冷启动解=元数据/内容/规则，非交互学习。

### Q2. 约束型推荐 vs 数据驱动；乘法评分公式的文献近亲

- 近亲排序：① Felfernig & Burke 2008（ICEC）[题名, Crossref] 约束型/知识型推荐——显式规则+权重、零交互历史，YHer 的正统家族；② Burke 2002（UMUAI）[题名] 混合推荐分类学（特征加权+加权混合）；③ MCDA 乘性效用/加权积模型 WPM（Keeney & Raiffa [题名]）。
- 乘法 vs 加性 vs LTR 的小样本取舍（无专门直接对照实验，由理论+经验文献合成）：
  - **乘性 = 非补偿性（AND 门）**：Triantaphyllou & Mann 1989（Decision Support Systems）[题名] 记录 WSM/WPM 排名反转行为——乘法天然实现"difficulty_fit 或 topic_match 过低即淘汰"的强约束语义，与 YHer 教学约束完全契合；
  - 加性 logit 是补偿性的，需逐维边际校准，小样本下同样只能靠专家先验；
  - LTR（RankNet [题名]、ListNet [题名]、LambdaMART [摘要, MSR]）假设大规模标注/三元组采样；BPR [题名, DBLP] 需海量 (u,i,j) 三元组。且 Dacrema 2019 / Ludewig 2020 / Rendle 2020 表明即使在大数据上 LTR 也常不敌调好的简单基线。
  - **小样本稳健性排序：乘法 WPM > 加性 logit > LTR。**

### Q3. 稀疏反馈下的推荐优化：bandit + OPE 如何适配审计事件

- 教育 bandit 实证：Clément et al. 2015（JEDM）[题名] arXiv:1310.3174 MAB 用于 ITS 活动序列；Belfer et al. 2022（AIED）[摘要] arXiv:2207.14003 数千学生轨迹 + RCT 验证，教育 CB 可行但规模远大于 YHer；De Kerpel et al. 2026 [题名] arXiv:2602.04347 教育推荐器+情境 TS+技能增益优化（与 YHer 同构的最新工作，仅题名级）。
- 探索：Thompson sampling 有后悔界与渐近最优性（Agrawal & Goyal [题名] arXiv:1111.1797、1209.3352），ε-greedy 无此类保证；冷启动 CB（Nguyen 2014 [题名] arXiv:1405.7544、Young & Leith 2023 [题名]）；高失败成本需安全探索（Jagerman et al. 2020 [题名] arXiv:2002.00467）。
- **审计事件 → logged bandit data 的最小字段集**：`(user_id, t, context x_t, action a_t, reward r_t, propensity p_t)`；propensity = 当时策略输出该 action 的概率（含探索参数与模型版本快照），**缺 propensity 即不能 IPS**（Swaminathan & Joachims 2015 [摘要] arXiv:1502.02362）；TOPK 多曝光需记录 slate 结构/位置（Swaminathan et al. 2017 [题名] arXiv:1605.04812）。
- OPE 估计量：IPS（1502.02362）、DR（Dudík et al. 2011 [题名] arXiv:1103.4601）、DR+shrinkage（Su et al. 2019 [题名] arXiv:1907.09623）、**CAPE（Thomas & Brunskill 2016 [摘要] arXiv:1604.00923，明确为"部署坏策略代价高昂"设计，MSE 常比 IPS/DR 低数个量级）**；自适应数据下最优 OPE（Wang et al. 2017 [题名] arXiv:1612.01205）；MNAR 去偏（Schnabel et al. 2016 [题名] arXiv:1602.05352）；真实数据大规模校验（Lefortier 2016 [题名] arXiv:1612.00367）。
- **败笔警示（设计硬约束）**：Rafferty et al. 2019（JEDM 11(1)）[题名，JEDM 官网核验] 自适应分配破坏随机化，标准检验失效；Williams et al. 2021 [摘要] arXiv:2103.12198 三个真实课堂 TS 实验 FPR/FNR 最高翻倍；Sun et al. 2023 [题名] arXiv:2311.14110 明确指出 **OPE 价值受数据量制约，<100 用户时结论需非常谨慎**；解药谱系：统计-奖励折中框架（Li et al. 2021 [题名] arXiv:2112.08507、SRO 2026 [题名] arXiv:2603.11267）、稀疏自适应实验（Song et al. 2025 [题名] arXiv:2501.03999）。

### Q4. efficacy 参数化：Beta+收缩 vs 分层 vs 因果调整

**结论：三者正交、按阶段叠加，不是互斥。** [均为摘要级]

- 朴素 per-video Beta 后验在少计数下方差大且先验强度无依据 → **最低成本 = Beta-Binomial + 经验收缩先验**（Cheng, Ho & Schorfheide 2025 [摘要] arXiv:2506.21987 教师增值的经验贝叶斯收缩是成熟模板；SABDB [摘要] arXiv:2608.07708 警告标准分层模型在小子群会区间过窄失校准——收缩方式影响校准质量）。
- 数据增长后升级分层贝叶斯：StanBKT [摘要] arXiv:2605.23048（standard/grouped/hierarchical BKT，ASSISTments 上精度不损失且获不确定性量化）；Sun 2025 [摘要] arXiv:2506.00057（同时估计技能难度与学生能力）。
- 因果调整第二阶段叠加：deconfounded 两阶段（Wang, Liang, Charlin, Blei arXiv:1808.06581 [摘要]——"曝光模型+结果模型"正是"看视频→通过"的混杂结构）；DecRS（KDD 2021 [摘要] arXiv:2105.10648——直接对历史 success 计数建模会复现选择偏差）；iDCF [摘要] arXiv:2302.05052；因果推荐综述（Luo et al. 2024 [摘要] arXiv:2303.11666）。
- reward 混杂控制的标准手段：题目难度+学生能力协变量分层/回归调整（Deep-IRT [摘要] arXiv:1904.11738 把"学生基础×题目难度"与资源效应显式分离）；"看视频后通过"作为信号有可行性证据（Otto et al. 2022 AIED [摘要] arXiv:2212.06679 多模态特征预测 knowledge gain）；小样本下 IPW 高方差脆弱（Saito 2020 [摘要] arXiv:1910.01444），IPW 宜作稳健性检验而非主估计；效果判定用离线反事实/interleaving 而非全量 A/B（Sato 2021 [摘要] arXiv:2107.06630）；BKT 在概念漂移下最稳（Lee et al. 2025 [摘要] arXiv:2511.00704）→ 支持"简单计数+收缩"作稳健基线。

### Q5. 视频段级推荐（BV,p,segment）

- 段级推荐存在性证据：MOOC-Rec（Zhu, Hauff, Yang, EDM 2022）[题名, DBLP]——论坛问题→视频 clip 推荐，少见的 clip 级工作；Zoom 课个性化段摘要（Lee et al. 2021 [摘要] arXiv:2101.06328）；CogGen [摘要] arXiv:2506.20600（按学习目标分割+BKT 模型，ASR+LLM 粗定位路线模板）。
- 代价：需要时间锚与定位模型；TAGV 已成熟为基准任务（CACR [摘要] arXiv:2606.08436 六基准 SOTA；DA-MIVQA [摘要] arXiv:2607.06618 人工难度标注成本高）；**E-VQA [摘要] arXiv:2607.11862 警示：QA/定位模型可能"答对但给错段"，段锚需人工校验**。
- **无时间锚给"整视频+知识点标签"是教育场景主流默认**（PEEK [摘要] arXiv:2109.03154、VLE [摘要] arXiv:2207.01504、TED 转录推荐 [摘要] arXiv:1809.05350），搜索成本转给学生但避免错误锚点风险；知识点（p）粒度优于整课粒度（ACKRec [摘要] arXiv:2006.13257、HinCRec-RL [摘要] arXiv:2203.11011）。

### Q6. 处方映射：mastery learning 传统与学习路径生成

- 处方传统背书充分：Bloom 1968 Learning for Mastery [摘要, ERIC]；Bloom 1984 2-Sigma [题名, Crossref]；**Kulik & Kulik 1990（RER）108 项对照元分析 [摘要, ERIC] mastery 对考试显著正效应**；Guskey & Pigott 1988 [摘要]；Kulik & Kulik 1987 [摘要]（达标标准严格度敏感——支撑 mastery gate 设计）。效果量数字（Hattie 2018 聚合，[全文] visible-learning.org）：mastery d=0.57、ITS d=0.48、feedback d=0.70、RTI d=1.29；VanLehn 2011 [摘要, ERIC] step-based ITS d=0.76 ≈ 人辅导 d=0.79。
- **矫正性教学规范（Guskey 2010, Educational Leadership [全文]）可逐条翻译为 M/P/C/U→资源类型规则**：矫正须与初教定性不同（同伴辅导/合作组/助教替代形式，非重讲）；单位时间 +10–20%；随后二次平行形成性评价验证；已掌握者 enrichment；补救前补先修缺口。另有：提前调离未掌握学生降低后测（Israni 2018 [摘要] arXiv:1802.08616）；过度练习可减 1/3（Xia et al. 2025 [摘要] arXiv:2506.17577）。
- 路径生成 vs 手工处方表：小目录+强约束下深层 LPR 数据需求不满足（KnowLP 2025 [摘要] arXiv:2506.22303 明言先修标注昂贵；Vassoyan 2024 [摘要] arXiv:2411.11520 多数方法需大规模交互数据或专家标注）。轻量替代均为"有日志后"选项：预训练图排序模型（Vassoyan 2024）、从轨迹学先修结构（Annabi 2023 [摘要] arXiv:2402.01672）、语义化 KT（ExRec [摘要] arXiv:2507.11060）。**UniER 2026 [摘要] arXiv:2605.16750（9 数据集 18 方法）路径级处方系统性优于单资源推荐**——数据充分后应迁移，MVP 冷启动期手工处方表是 mastery 传统直接支持的基线。注意 Pelkola 2017 [摘要] arXiv:1712.07848 提醒 LFM 落地真实数学课效果"modest but positive"，勿高估。

---

## 3. 对比表：五种路线在「小目录+冷启动+强约束+可审计」下

| 路线 | 冷启动可用性 | 强教学约束表达 | 可审计/确定性 | 数据需求 | 探索/学习能力 | 裁决 |
|---|---|---|---|---|---|---|
| **规则混合（现状）** | 即开即用 | 天然（乘法=AND 门） | 确定性可复现，最强 | 零（除 efficacy 计数） | 无（权重不可学习） | **保留为骨架**，权重改数据校准 |
| **约束推荐（Felfernig & Burke）** | 即开即用 | 最强（约束→过滤→可解释） | 强（规则可交互） | 零交互 | 无 | **吸收其分层**：约束=过滤层，评分=排序层 |
| **内容 CF / 最近邻** | 可用（元数据/诊断特征） | 弱 | 可解释 | 低 | 弱（近邻可在线更新，Jannach 2017 [摘要]） | 其"特征加权"思想已在公式中；不单独立项 |
| **学习排序（BPR/LambdaMART）** | 不可用 | 弱 | 黑盒 | 10⁴+ 交互、三元组采样 | 有 | **不引入**（Dacrema 2019/Ludewig 2020/Rendle 2020 反证） |
| **bandit + OPE** | 依赖先验 | 弱（需安全约束包装） | 审计事件即天然 logged data | 低（可零数据启动，靠先验） | 唯一在线学习路线 | **引入但只做"探索+日志"**；效果判定靠隔离臂，非自适应数据推断（Rafferty 2019/Williams 2021/Sun 2023） |

---

## 4. 裁决：逐条保留/改造/替换 + 最小可行架构

### 4.1 逐条裁决

1. **Score = W_track × topic_match × type_fit × difficulty_fit × efficacy（乘法形式）：保留结构**。文献位置 = Felfernig & Burke 2008 约束推荐 + MCDA 乘性效用（WPM）；乘法非补偿性天然实现强约束。但需做两处改造：(a) 语义拆层——过滤层（unlock/先修/mastery gate 硬约束，参考 Cheng 2025 先修阻塞、Israni 2018）+ 排序层（乘法评分）；(b) 权重不再纯拍脑袋，走"处方先验 + held-out 数据校准"（承接 MASTER_AUDIT T1 模式）。
2. **efficacy=1.0：替换**。升级路径（承接 w3 裁决并具体化）：Beta-Binomial + 知识点/教师层经验收缩先验（Cheng 2025 VA；不直接用朴素 per-video Beta）→ 分层贝叶斯（StanBKT 类）→ 数据足够后叠 deconfounded/IPW 稳健性检验。reward 定义改为 **gain（后测−前测）按题目难度分层**，纯 pass/fail 不作因果语言（MASTER_AUDIT T3）。
3. **TOPK 分桶平局（round(score/0.05)）：改造**。分桶无文献依据且制造伪并列；**删除 round 分桶，直接用连续分数排序**；平局键保留确定性（发布日期/播放量/合集序）但**首键改为 efficacy 后验均值降序**（有统计含义且随数据更新）。TOPK=3 与预算时长截断保留。
4. **W_track 手工权重表：保留但降级为"处方先验"**——负责过滤与类型路由，不负责微调排序；处方表本身受 mastery learning 元分析（Kulik 1990）与 corrective instruction 规范（Guskey 2010）背书。按 MASTER_AUDIT V1 的 3 簇运营（{M,C,{P,U}}）重新映射；数据积累后按 UniER 证据迁移路径级处方 + KT 驱动。
5. **段级推荐（BV,p,segment）：MVP 保留"视频级 + p/知识点标签"**（PEEK 模式，教育主流默认）；segment 作为后续增强：有字幕时 ASR+LLM 粗定位（CogGen 路线）+ 人工校验（E-VQA 警示），无锚不给 t= 是正确取舍，不设为首版硬要求。
6. **rec_served 审计：扩展字段**（见 4.3），使审计事件同时是 logged bandit data。

### 4.2 最小可行架构（数学 MVP）

```
诊断(3簇: M / C / P·U) ──▶ 处方先验 W_track(过滤层: unlock/mastery gate/类型路由)
                          │
                          ▼
              候选集(~30视频 × p粒度) ──▶ 评分层: score = W_track × topic_match × type_fit × difficulty_fit × efficacy_posterior
                          │                     efficacy_posterior = Beta(α+success, β+fail) 均值, 收缩至知识点层先验
                          ▼
              排序层: 连续分数 desc → 平局键(efficacy 均值, 发布日期) → TOPK=3 → 预算时长截断 → 跨 session seen 去重
                          │
                          ▼
              服务层: 确定性服务 90-95% / 探索臂 5-10%(Thompson, 先验=处方分数归一化)
                          │
                          ▼
              rec_served 事件(含 propensity) → 看后验证(gain) → Beta 计数回流 + 周度 OPE 报告
```

探索设计（高失败成本 + <100 用户）：不用 ε-greedy 逐用户乱探；用**"确定性服务 + 计划内小比例探索 + 隔离评估臂"**三件套——需要统计结论的实验保持统一随机分配（Rafferty 2019 教训），日常探索用保守 TS（先验注入，Jagerman 2020 安全约束精神）。

### 4.3 OPE 评估协议

- **logging policy**：`π_b = 0.95·argmax(score) + 0.05·softmax(score/τ)`（显式、可复算）；所有推荐（含确定性臂）都写 propensity = π_b(a|x)。
- **rec_served 事件最小字段集**（在现有审计之上）：`user_id, ts, context 快照{诊断簇, 知识点, 前测基线, 题目难度分层}, action{BV,p,segment|null}, slate{TOPK 列表与位置}, propensity, reward{后测通过/失败, gain}, 策略版本`。
- **IPS 计算要点**：w = π_e(a|x)/π_b(a|x)；小样本必用 **clipping + 自归一化（SNIPS）**；DR 作稳健性检验（回归模型 = 用 Deep-IRT 式难度×能力调整）；**CAPE 作小样本首选估计量**（MSE 低数个量级）。
- **解释边界**：<100 用户时 OPE 只作离线筛查/方向判断（Sun 2023），最终效果判定靠隔离臂 RCT；报告语言用"描述性关联"，禁止因果断言（MASTER_AUDIT T3）。

---

## 5. 未决问题

1. S2S 全程 429：citationCount 全部缺失，无法区分高引与冷门，**报告中的引用权重是缺失维度**。
2. "Huang 2019" MOOC 推荐综述未找到（可能为书籍章节/非英文或用户记忆有误），Q1 结论不依赖它。
3. Shaikh 2019 exploration、ZEST 原典（Mandel 2014）仅题名级；Chapelle & Li 2011 TS 实证评估未取得。
4. 乘法 vs 加性 vs LTR 的小样本直接对照实验文献不存在，稳健性排序为合成推断。
5. 2026 年新论文（De Kerpel 2026、UniER、SRO 等）多为预印本 [摘要] 级，未同行评议。
6. 处方表 3 簇映射、探索臂比例 5-10%、收缩先验强度等具体数值需数据回流后校准（本报告只给结构）。
7. 化学侧 135 节点 KG 与数学新 KG 的映射资产如何复用，超出本 lane 范围。

---

## 6. 参考文献表

（等级标于每条末尾；arXiv 条目均经主代理 id_list 复核或子代理 abs 页核验）

### 推荐方法/小数据经验评估
1. Dacrema, Ferrari, Cremonesi, Jannach. Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. RecSys 2019. arXiv:1907.06902 [摘要]
2. Ludewig, Mauro, Latifi, Jannach. Empirical Analysis of Session-Based Recommendation Algorithms. UMUAI 31:149–181, 2020. Springer OA [全文]
3. Rendle, Krichene, Zhang, Anderson. Neural Collaborative Filtering vs. Matrix Factorization Revisited. RecSys 2020. arXiv:2005.09683 [摘要]
4. Kang & McAuley. Self-Attentive Sequential Recommendation. ICDM 2018. arXiv:1808.09781 [摘要]
5. Hidasi et al. Session-based Recommendations with RNNs (GRU4Rec). ICLR 2016. arXiv:1511.06939 [摘要]
6. He et al. LightGCN. SIGIR 2020. arXiv:2002.02126 [摘要]
7. Wang et al. RippleNet. CIKM 2018. arXiv:1803.03467 [摘要]
8. Wang et al. KGCN. WWW 2019. arXiv:1904.12575 [摘要]
9. Wang et al. Explainable Reasoning over KGs for Recommendation (KPRN). AAAI 2019 [摘要]
10. Jannach & Ludewig. When RNNs meet the Neighborhood for Session-Based Recommendation. RecSys 2017 [摘要]
11. Uddin et al. A Systematic Mapping Review on MOOC Recommender Systems. IEEE Access 2021 [摘要]
12. Urdaneta-Ponte et al. Recommendation Systems for Education: Systematic Review. Electronics 2021 [摘要]
13. da Silva et al. SLR on Educational Recommender Systems for Teaching and Learning. Educ. Inf. Technol. 2022 [题名]
14. Yuan & Hernandez. User Cold Start Problem in RS: A Systematic Review. IEEE Access 2023 [摘要]
15. Schein et al. Methods and Metrics for Cold-Start Recommendations. SIGIR 2002 [题名]
16. Fernández-Tobías et al. Addressing User Cold Start with Cross-Domain CF. UMUAI 2019 [题名]
17. Gulzar, Leema, Deepak. PCRS: Personalized Course Recommender System Based on Hybrid Approach. Procedia CS 125:518–524, 2018 [题名]

### 约束/知识型推荐与评分形式
18. Felfernig & Burke. Constraint-Based Recommender Systems. ICEC 2008. DOI 10.1145/1409540.1409544 [题名]
19. Felfernig, Friedrich, Jannach, Zanker. Constraint-Based Recommender Systems. Recommender Systems Handbook 2015 [题名]
20. Burke. Hybrid Recommender Systems: Survey and Experiments. UMUAI 2002 [题名]
21. Keeney & Raiffa. Decisions with Multiple Objectives. Cambridge UP [题名]
22. Triantaphyllou & Mann. An Examination of the Effectiveness of Multi-Dimensional Decision-Making Methods. Decision Support Systems 1989 [题名]
23. Smyth. Case-Based Recommendation. The Adaptive Web, LNCS 4321, 2007 [题名]
24. Pazzani & Billsus. Content-Based Recommendation Systems. The Adaptive Web, LNCS 4321, 2007 [题名]

### 排序学习
25. Rendle et al. BPR: Bayesian Personalized Ranking. UAI 2009 [题名]
26. Burges et al. Learning to Rank Using Gradient Descent (RankNet). ICML 2005 [题名]
27. Cao et al. Learning to Rank: Pairwise to Listwise (ListNet). ICML 2007 [题名]
28. Burges. From RankNet to LambdaRank to LambdaMART. MSR-TR-2010-82 [摘要]
29. Karatzoglou, Baltrunas, Shi. Learning to Rank for Recommender Systems. RecSys 2013 [题名]

### Bandit / OPE / 自适应实验
30. Li, Chu, Langford, Schapire. A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010. arXiv:1003.0146 [题名]
31. Clément, Roy, Oudeyer, Lopes. Multi-Armed Bandits for ITS. JEDM 2015. arXiv:1310.3174 [题名]
32. Belfer, Kochmar, Serban. Raising Student Completion Rates with Adaptive Curriculum and Contextual Bandits. AIED 2022. arXiv:2207.14003 [摘要]
33. De Kerpel et al. A Bandit-Based Approach to Educational RS: Contextual TS for Learner Skill Gain. 2026. arXiv:2602.04347 [题名]
34. Agrawal & Goyal. Analysis of TS for MAB / TS for Contextual Bandits. arXiv:1111.1797, arXiv:1209.3352 [题名]
35. Russo et al. A Tutorial on Thompson Sampling. arXiv:1707.02038 [题名]
36. Nguyen, Mary, Preux. Cold-start Problems in RS via Contextual-bandit Algorithms. arXiv:1405.7544 [题名]
37. Young & Leith. High Accuracy and Low Regret for User-Cold-Start Using Latent Bandits. arXiv:2305.18305 [题名]
38. Jagerman, Markov, de Rijke. Safe Exploration for Optimizing Contextual Bandits. arXiv:2002.00467 [题名]
39. Swaminathan & Joachims. Counterfactual Risk Minimization: Learning from Logged Bandit Feedback. ICML/JMLR 2015. arXiv:1502.02362 [摘要]
40. Dudík, Langford, Li. Doubly Robust Policy Evaluation and Learning. ICML 2011. arXiv:1103.4601 [题名]
41. Thomas & Brunskill. Data-Efficient Off-Policy Policy Evaluation for RL (CAPE). ICML 2016. arXiv:1604.00923 [摘要]
42. Su et al. Doubly Robust OPE with Shrinkage. arXiv:1907.09623 [题名]
43. Wang, Agarwal, Dudík. Optimal and Adaptive OPE in Contextual Bandits. arXiv:1612.01205 [题名]
44. Swaminathan et al. Off-policy Evaluation for Slate Recommendation. arXiv:1605.04812 [题名]
45. Schnabel et al. Recommendations as Treatments. ICML 2016. arXiv:1602.05352 [题名]
46. Lefortier et al. Large-scale Validation of Counterfactual Learning Methods: A Test-Bed. arXiv:1612.00367 [题名]
47. Sun et al. When is OPE Useful in Contextual Bandits? A Data-Centric Perspective. arXiv:2311.14110 [题名]
48. Rafferty, Ying, Williams. Statistical Consequences of using MABs to Conduct Adaptive Educational Experiments. JEDM 11(1):47–79, 2019 [题名，JEDM 官网核验]
49. Williams, Nogas, Deliu, Shaikh, Villar, Durand, Rafferty. Challenges in Statistical Analysis of Data Collected by a Bandit Algorithm. arXiv:2103.12198 [摘要]
50. Li et al. Algorithms for Adaptive Experiments that Trade-off Statistical Analysis with Reward. arXiv:2112.08507 [题名]
51. Li, Mandel, Phillips, Rafferty et al. A Statistically Reliable Optimization Framework for Bandit Experiments (SRO). 2026. arXiv:2603.11267 [题名]
52. Song, Musabirov, Bhattacharjee, Durand, Franklin, Rafferty. Adaptive Experiments Under Data Sparse Settings. arXiv:2501.03999 [题名]
53. Zavaleta-Bernuy, Zheng, Shaikh, Nogas, Rafferty, Petersen, Williams. Using Adaptive Experiments to Rapidly Help Students. arXiv:2208.05092 [题名]
54. Girard et al. Counterfactual learning of new adaptive instructional policies using logged data. 2026. arXiv:2606.23015 [题名]
55. Mandel, Liu, Levine, Brunskill, Popović. Offline Policy Evaluation Across Representations with Applications to Educational Games. AAMAS 2014 [题名，AAMAS 官方 TOC 核验]

### efficacy / 因果 / 分层建模
56. Wang, Liang, Charlin, Blei. The Deconfounded Recommender: A Causal Inference Approach. arXiv:1808.06581 [摘要]
57. Wang, Feng, He, Wang, Chua. Deconfounded Recommendation for Alleviating Bias Amplification. KDD 2021. arXiv:2105.10648 [摘要]
58. Zhang et al. Debiasing Recommendation by Learning Identifiable Latent Confounders (iDCF). arXiv:2302.05052 [摘要]
59. Luo et al. A Survey on Causal Inference for Recommendation. The Innovation 5(2), 2024. arXiv:2303.11666 [摘要]
60. Pradhan et al. StanBKT: Rethinking Parameter Estimation in BKT. 2026. arXiv:2605.23048 [摘要]
61. Sun. Hierarchical Bayesian Knowledge Tracing in Undergraduate Engineering Education. 2025. arXiv:2506.00057 [摘要]
62. Yavuz & Kaplan. Small Area Bayesian Dynamic Borrowing (SABDB). 2026. arXiv:2608.07708 [摘要]
63. Cheng, Ho, Schorfheide. Optimal Estimation of Two-Way Effects under Limited Mobility. 2025. arXiv:2506.21987 [摘要]
64. Yeung. Deep-IRT. arXiv:1904.11738 [摘要]
65. Otto et al. Predicting Knowledge Gain for MOOC Video Consumption. AIED 2022, LNCS 13356. arXiv:2212.06679 [摘要]
66. Oganisian, Mitra, Roy. Hierarchical Bayesian Bootstrap for HTE Estimation. Int. J. Biostatistics 2022. arXiv:2009.10839 [摘要]
67. Witty et al. Causal Inference using Gaussian Processes with Structured Latent Confounders. ICML 2020. arXiv:2007.07127 [摘要]
68. Saito. Asymmetric Tri-training for Debiasing MNAR Explicit Feedback. SIGIR 2020. arXiv:1910.01444 [摘要]
69. Raja & Vats. Counterfactual Risk Minimization with IPS-Weighted BPR... RecSys 2025 CONSEQUENCES. arXiv:2509.00333 [摘要]
70. Sato. Online Evaluation Methods for the Causal Effect of Recommendations. RecSys 2021. arXiv:2107.06630 [摘要]
71. Lee et al. Investigating the Robustness of KT Models in the Presence of Student Concept Drift. 2025. arXiv:2511.00704 [摘要]
72. Zhou et al. An efficient adaptive dimension selection algorithm for multidimensional probit graded response models. 2026. arXiv:2607.17654 [摘要]

### 段级推荐 / 视频
73. Zhu, Hauff, Yang. MOOC-Rec: Instructional Video Clip Recommendation for MOOC Forum Questions. EDM 2022 [题名，DBLP conf/edm/ZhuH022]
74. Lee et al. Attention Based Video Summaries of Live Online Zoom Classes. AAAI-2021 TIPCE. arXiv:2101.06328 [摘要]
75. Li, Pea, Haber, Subramonyam. CogGen: A Learner-Centered Generative AI Architecture for ITS with Programming Video. 2025. arXiv:2506.20600 [摘要]
76. Qi et al. CACR: Reinforcing Temporal Answer Grounding in Instructional Video. 2026. arXiv:2606.08436 [摘要]
77. Liu et al. Overview of NLPCC 2026 Shared Task 1 (DA-MIVQA). arXiv:2607.06618 [摘要]
78. Wang et al. Evidence-Backed Video Question Answering (E-VQA). ECCV 2026. arXiv:2607.11862 [摘要]
79. Bulathwela et al. PEEK: A Large Dataset of Learner Engagement with Educational Videos. RecSys 2021 ORSUM. arXiv:2109.03154 [摘要]
80. Zhao, Wang, Sahebi. Modeling Knowledge Acquisition from Multiple Learning Resource Types (MVKM). arXiv:2006.13390 [摘要]
81. Wang, Gong, Wang et al. ACKRec: Attentional GCN for Knowledge Concept Recommendation in MOOCs. arXiv:2006.13257 [摘要]
82. Gong et al. HinCRec-RL: Reinforced MOOCs Concept Recommendation in HINs. arXiv:2203.11011 [摘要]
83. Bulathwela et al. Can Population-based Engagement Improve Personalisation? (VLE). EDM 2022. arXiv:2207.01504 [摘要]
84. Oh et al. TED Talk Recommender Using Speech Transcripts. ASONAM 2018. arXiv:1809.05350 [摘要]

### 处方 / mastery / 学习路径
85. Bloom. Learning for Mastery. Evaluation Comment 1(2), 1968. ERIC ED053419 [摘要]
86. Bloom. The 2 Sigma Problem. Educational Researcher 13(6):4–16, 1984 [题名]
87. Kulik, Kulik, Bangert-Drowns. Effectiveness of Mastery Learning Programs: A Meta-Analysis. RER 60(2):265–299, 1990 [摘要]
88. Guskey & Pigott. Research on Group-Based Mastery Learning Programs: A Meta-Analysis. J. Educ. Research 81(4), 1988 [摘要]
89. Kulik & Kulik. Mastery Testing and Student Learning: A Meta-Analysis. J. Educ. Tech. Systems 1987 [摘要]
90. Guskey. Lessons of Mastery Learning. Educational Leadership 68(2), 2010 [全文]
91. VanLehn. The Relative Effectiveness of Human Tutoring, ITS, and Other Tutoring Systems. Educational Psychologist 46(4):197–221, 2011 [摘要]
92. Israni, Sales, Pane. Mastery Learning in Practice... arXiv:1802.08616 [摘要]
93. Sales & Pane. The Role of Mastery Learning in ITS... arXiv:1707.09308 [摘要]
94. Nabizadeh et al. Learning Path Personalization and Recommendation Methods: A Survey. ESWA 159:113596, 2020 [题名]
95. Shen et al. A Survey of Knowledge Tracing. IEEE TLT 17, 2024. arXiv:2105.15106 [摘要]
96. Abdelrahman, Wang, Nunes. Knowledge Tracing: A Survey. arXiv:2201.06953 [摘要]
97. Annabi & Nguyen. Prerequisite Structure Discovery in ITS. ICDL 2023. arXiv:2402.01672 [摘要]
98. Cheng et al. GraphRAG-Induced Dual Knowledge Structure Graphs for LPR (KnowLP). 2025. arXiv:2506.22303 [摘要]
99. Ozyurt et al. Personalized Exercise Recommendation with Semantically-Grounded KT (ExRec). 2025. arXiv:2507.11060 [摘要]
100. Vassoyan et al. A Pre-Trained Graph-Based Model for Adaptive Sequencing of Educational Documents. NeurIPS 2024 FM-Assess. arXiv:2411.11520 [摘要]
101. Vassoyan, Vie, Lemberger. Towards Scalable Adaptive Learning with GNNs and RL. arXiv:2305.06398 [摘要]
102. Chen, Saeedvand, Lai. Adaptive Learning Path Navigation Based on KT and RL. arXiv:2305.04475 [摘要]
103. Chen et al. Set-to-Sequence Concept-aware LPR (SRC). arXiv:2306.04234 [摘要]
104. Cheng et al. UniER: A Unified Benchmark for Item-level and Path-level Exercise Recommendation. 2026. arXiv:2605.16750 [摘要]
105. Pelkola, Rasila, Sangwin. Blended Mastery Learning in Mathematics. arXiv:1712.07848 [摘要]
106. Xia et al. Optimizing Mastery Learning by Fast-Forwarding Over-Practice Steps. EC-TEL 2025. arXiv:2506.17577 [摘要]
107. Noh et al. Simulating Learners' Task-Selection Strategies and System Constraints in Mastery Learning. EDM 2026. arXiv:2605.21613 [摘要]
108. Hattie. Visible Learning 252 Influences (聚合效果量表). visible-learning.org [全文]
