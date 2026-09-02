# Lane5：验证设计 / 掌握判定 / 学习者画像 / 事件记录标准 — 文献检索与最优解判定

> 日期：2026-08-13 | 任务：architecture_research_v3 车道 5（只读研究，不改项目代码）
> 对象：上海高中数学 MVP「单知识点诊断 → 视频/练习 → 看后验证 → 更新画像」窄闭环
> 与既有研究的关系：本文是 `architecture_research/MASTER_AUDIT_REPORT_2026-08-05.md`（M7 验证闭环 / M8 事件溯源画像）与 `w4_memory.md`（M6 记忆）的深化车道，冲突时以本文为准。

## 1. 检索记录

| 渠道 | 状态 | 用途 |
|---|---|---|
| arXiv API（export.arxiv.org） | ✅ 可达（http 需 -L 走 https） | OLM/LA/privacy 近期文献（命中见下） |
| Semantic Scholar API | ❌ 检索受限（持续 429 Too Many Requests，未配 key） | 引用数、摘要字段未获得 |
| Crossref REST API | ✅ 可达 | 经典文献书目核验（标题/作者/年份/卷期/DOI） |
| webfetch：xAPI.com overview | ✅ 全文 | xAPI/IEEE 9274.1.1-2023 地位 |
| webfetch：IMS Caliper 1.2 spec | ✅ 全文 | 事件 schema 规范细节 |
| webfetch：IMS CASE v1.0 spec | ✅ 全文 | 能力/课标交换标准 |
| webfetch：DigiChina PIPL 英译全文 | ✅ 全文 | PIPL 第 28/29/31/47/55 条 |
| webfetch：gdpr-info.eu Art.8 | ✅ 全文 | GDPR 儿童同意年龄 |
| webfetch：FTC COPPA Rule 页 | ✅ 全文（规则摘要+2025 修订通告） | COPPA 适用与数据最小化 |
| webfetch：OECD（Learning Compass 2030 / PISA 2022 框架） | ❌ 检索受限（403） | PISA 框架内容标 [记忆回溯] |

证据等级约定：**[全文]**=本次实际读到正文；**[摘要]**=本次读到摘要；**[题名]**=Crossref/官方目录核验书目（标题/出处/DOI 属实，正文内容来自领域公认结论，个别数值标 [记忆回溯]）；**[记忆回溯]**=未获一手来源，依赖研究者知识，且未编造——已尽量用可复算的数学推导替代不可靠的记忆数值。

## 2. 七个问题的文献回答

### Q1 掌握判定标准：mastery threshold 与 held-out n=2 的二项噪声

**传统阈值**：Bloom（1968, *Learning for Mastery*, UCLA-CSEIP 灰文献）主张形成性测验 90% 达标为掌握线；Block & Burns（1976, RRE, 10.3102/0091732X004001003）[题名] 综述当时实践：掌握标准普遍取 **80–90%**；Guskey（2012 百科条目 / 2010 "Lessons of Mastery Learning"）[题名] 归纳同一区间；Kulik & Kulik（1990, RER, 10.3102/00346543060002265）[题名] 元分析：掌握学习对期末成绩平均 +0.5 SD，但对 K-12 证据弱于高等教育（Slavin 1987 批评与 Bloom 1987 回应 [题名]）。"连续 3 次正确"在文献中没有独立理论地位，属实践启发式；若按"全对才过"计，n=3 时 95% CI 下界 = 0.05^(1/3) ≈ **36.8%**，同样无辩护空间。

**"mastery decision 需要多少题"有专门文献**（criterion-referenced mastery testing 谱系）：
- Millman（1973, RER, 10.3102/00346543043002205）[题名]：passing score × 题目长度 → 分类决策误差（误判 master/non-master 概率）的解析框架；
- Emrick（1971, JEM, 10.1111/j.1745-3984.1971.tb00946.x）[题名]：决策理论式 mastery testing 模型（"连续多题全对"式规则在其中被证明需要大样本才可控）；Wilcox（1976, JES, 10.3102/10769986001004359）[题名] 与 van der Linden（1982, Evaluation in Education, 10.1016/0191-765x(82)90015-5）[题名] 给出长度-分数线设计表；
- Kingsbury & Weiss（1983）[题名]：顺序掌握测验（SPRT 风格）与 IRT 自适应掌握测验的对比。

**可复算的结论**（推导，非引用）：二项模型下"n 题全对"的 95% Clopper-Pearson CI 下界 = 0.05^(1/n)。n=2 → **15.8%**（与质疑值一致）；n=4 → 47.3%；n=6 → 60.7%；n=10 → 74.1%；**n=14 → 80.7%**（下界才够到 0.8）。即：若坚持"全对"规则且要 95% 置信下界 ≥ 0.8，需要 ≥14 题。若采用"≥80% 正确"规则并控制误判率 α=β=0.05~0.10，Millman/van der Linden 框架给出的长度在 **15–25 题**量级。**结论：held-out n=2 做 pass/fail 掌握判定在文献与统计学上均不可辩护**；它最多是一个带极大不确定性的弱信号，必须以 CI 形式报告，不得二值化。

**delayed post-test（retention testing）**：d≥7 天是记忆研究的通行约定（Roediger & Karpicke 2006 [题名] 用 1 周保留间隔；Rowland 2014 元分析 [题名] 显示测试效应在延时保持上更大；Adesope et al. 2017, RER, 10.3102/0034654316689306 [题名] 报告延时测验效应量高于即时）。即：即时测验测"看过没有"，延时测验（≥7 天）才测"记住了没有"。

### Q2 迁移验证：near vs far，单知识点闭环的天花板

Barnett & Ceci（2002, Psych. Bulletin, 10.1037/0033-2909.128.4.612）[题名]：迁移距离是 content×context 的多维连续体（知识域、物理环境、时间、功能、社会等）。YHer 的 held-out 同知识点不同题族 = 内容近、情境近的 **near transfer**；上海高考综合题（跨知识点、新情境、多步推理）= **far transfer**。文献结论：近迁移不代表远迁移，两者相关弱（同一 taxonomy 的核心主张）。**单知识点闭环的验证天花板 = 该知识点内的 near transfer + 前置链上的可诊断性；far transfer（综合题）从结构上无法由单知识点 held-out 推断**。所以"verified"若指向综合题能力，是逻辑越权；项目现有红线（verified 只描述本 session 未见题族表现）恰与文献一致，是正确自限。

### Q3 开放学习者模型 OLM

- Bull & Kay（2007, IJAIED, 10.3233/IRG-2007-17(2)02 "SMILI:() Framework"）[题名] 与 Bull & Kay（2016, IJAIED, 10.1007/s40593-015-0090-8 "SMILI☺"）[题名]：OLM = 把学习者模型对学生可见、可理解（scrutability）、可协商（negotiation）——支持元认知与信任。
- Long & Aleven（2016, UMUAI, 10.1007/s11257-016-9186-6）[题名]：在方程求解 ITS 上 OLM+自我调节提示带来学习增益（小学/中学场景）。
- Bodily & Verbert（2017, IEEE TLT, 10.1109/TLT.2017.2740172）[题名] 与 Matcha et al.（2020, IEEE TLT, 10.1109/TLT.2019.2916802）[题名] 系统综述：学生面向 LA 面板对成绩增益证据有限、对动机/元认知有正面但中等且不均的效果；效果偏向已有高自我调节能力的学生；Valle et al.（2021, Computers & Education, 10.1016/j.compedu.2021.104288）[题名]：面板+任务价值脚手架比裸面板有效。
- **裁决依据**：OLM 文献支持"可见+可解释"，不支持"只给结论不给证据"。对未成年用户，最小可见、证据可追溯（scrutability）是合规与教育学双赢。

### Q4 事件记录标准：xAPI / Caliper vs 自研 JSONL

- xAPI（Experience API）= **IEEE 9274.1.1-2023** 正式标准 [全文：xAPI.com overview 确认 IEEE 编号与"noun-verb-object statement + LRS"模型]；statement 核心字段（spec 1.0.x）[记忆回溯]：`actor`/`verb`/`object`/`result`/`context`/`timestamp`/`stored`/`id`（UUID）/`authority`（**声明来源=provenance 的官方位置**）/`version`。LRS 按 id 幂等去重；支持 learner 自带 "personal data locker"（个人数据柜，恰好对应无登录三年画像的可携带方案）[全文：overview 页 "personal data lockers"]。
- IMS Caliper 1.2 [全文]：每 Event **必须** `id`（UUID v4）+ `type` + `actor` + `action` + `object` + `eventTime`（ISO8601 毫秒 UTC，MUST）；可选 `edApp`/`session`/`group`/`extensions`；Media/Assessment/Session Profiles 直接覆盖 YHer 的 watch_proxy/seen_segments/pause/resume/answer_scored 事件类型。
- **迁移价值判定**：YHer 的事件类型集合已是 Caliper 子集。把自研 JSONL schema 对齐到 xAPI/Caliper 的字段形状（保留内部名），未来接 LRS 只是一次字段映射，不重写重放逻辑；反之自由 schema 会在三年期付出重放器重写的代价。**当前阶段不需要真实 LRS**（无登录、单机、成本为零）；"xAPI 形状的本地 JSONL"是成本最低的期权。

### Q5 长期画像建模：三年画像 schema

- IMS CASE v1.0 [全文]：能力/课标条目的机器可读交换（URI 标识、层级、关联/先决关系），价值 = 未来对齐上海考纲时 KG 节点应有稳定 URI 标识，而非仅为"标签"。
- PISA 2022 数学素养框架 [记忆回溯，OECD 主站 403]：内容维度（数量、不确定性与数据、变化与关系、空间与形状）+ 过程维度（表述/运用/阐释与评估 + 数学推理）。用途：给画像的"能力面"提供通用语言，但 YHer 是单知识点粒度，不需要全框架。
- OECD Learning Compass 2030 [记忆回溯，403]：能力 = 知识+技能+态度与价值观，强调 agency（学生能动性）——支持"画像归学生所有、可携带、可删除"的架构立场。
- **schema 建议**：三层。①事件日志（append-only，可重放，唯一真相，本地）；②派生快照（per-KP：belief/stability/retention、上次验证、n 与 CI、review_due_at——可随时从①重算）；③用户自报元数据（年级/目标/计划，自愿可改）。forgetting 由 FSRS（快照层）承担，不由事件层承担。原始秒级事件保留策略：本地 90 天，之后聚合为日级统计+删除原始（同时满足 PIPL 第 19 条最短保存期）。

### Q6 未成年人数据合规的硬约束

- **中国 PIPL** [全文，DigiChina 英译]：第 28 条——不满 14 周岁未成年人个人信息属**敏感个人信息**；第 31 条——处理须取得**父母/监护人同意**，并**制定专门的个人信息处理规则**；第 29 条——敏感信息需单独同意；第 17/19 条——告知保存期限、保存期取最短必要；第 47 条——目的实现/撤回同意/停止服务时**主动删除**；第 55 条——处理敏感信息与自动化决策前须做**个人信息保护影响评估（PIPIA）**；第 24 条——自动化决策（含画像推送）须透明、可拒绝。→ YHer 的画像与推荐属于"自动化决策+敏感信息"双重触发，PIPIA 与专门规则是硬义务，不是可选。
- **GDPR-K** [全文]：Art. 8——信息社会服务面向儿童时，默认 16 岁为同意年龄、成员国可降至 13 岁；Art. 17 删除权；Art. 22 画像自动化决策限制；Art. 35 DPIA。→ 若未来面向欧盟用户，同样要求监护人同意或 16+ 自证。
- **美国 COPPA** [全文，FTC 规则页]：面向 13 岁以下的服务收集个人信息须**可验证父母同意**、数据最小化、仅保留必要期限（2025-04 修订版进一步收紧保留与删除）[全文页含 2025 Federal Register 通告链接]。
- **FERPA** [记忆回溯]：只约束受资助教育机构的教育记录（20 U.S.C. §1232g）；YHer 公益产品不直接适用，但若与学校合作导入成绩/名单即触发，须在架构上把"校方数据"与"个人画像"物理隔离。
- **"localStorage 无登录"与"三年画像"的矛盾**：文献与法规共同指向的解法 = **本地优先（local-first）+ 导出/导入 + 删除**。事件源 JSONL 本身就是可携带档案（xAPI "personal data locker" 同构 [全文]）；三年连续性由学生自己携带导出文件跨设备延续，而非服务端账号。服务端若未来存在：只存伪匿名 ID，画像永不离开设备；提供一键删除（本地删除+云端 ID 注销）。这是唯一在"无登录+未成年+三年愿景"约束下成立且文献可辩护的架构。

### Q7 闭环有效性：检索实践（testing effect）

- Roediger & Karpicke（2006, Psych. Science, 10.1111/j.1467-9280.2006.01693.x）[题名]：测试优于重读，1 周后保持差距扩大。
- Rowland（2014, Psych. Bulletin, 10.1037/a0037559）[题名] 元分析：testing effect g≈0.50；Adesope et al.（2017, RER, 10.3102/0034654316689306）[题名]：练习测试 vs 重读 g≈0.61，延时测验更大；Yang et al.（2021, Psych. Bulletin, 10.1037/bul0000309）[题名]：课堂测验类 meta，正效应稳健。
- 间隔安排：Karpicke & Bauernschmidt（2011, JEP:LMC, 10.1037/a0023436）[题名]：绝对间隔越大保持越好；Karpicke & Roediger（2007, JEP:LMC, 10.1037/0278-7393.33.4.704）[题名]：等间隔优于膨胀间隔（长时）；Latimier et al.（2020, Educational Psychology Review, 10.1007/s10648-020-09572-8）[题名]：间隔的检索练习对保持有稳定正效应 meta。
- **设计含义**："看视频后立即测试"（零间隔检索）有即时收益但测不到保持；"间隔后测试"（≥7 天）才是 retention 证据。两者都要：即时 2-3 题作检索环节（testing effect），延时 3-5 题作 delayed check（retention）。

## 3. 方案对比表

### 3.1 验证设计

| 方案 | 题族数×题数 | 判定规则 | near/far | immediate/delayed | 证据地位 |
|---|---|---|---|---|---|
| A 现行 | 2 族×1 题 | 2/2=verified | near | immediate only | 95%CI 下界 15.8%，文献不可辩护 |
| B 扩大 held-out | 2 族×3 题（n=6） | 后验 p̂+95%CI，p̂≥0.8 且下界≥0.5 → session_verified | near | immediate | CI 下界 60.7%，可作弱信号+诚实标签 |
| C 顺序掌握测验 | 变长，SPRT/Emrick | 误判率 α=β 可控 | near | immediate | 文献标准（Millman/van der Linden），需 ~15-25 题量级才能 α,β≤0.1 |
| D BKT/IRT 后验 | 变长 | P(master)≥0.8 | near | immediate | 与 lane M1 改造合并，冷启动参数不足 |
| E 建议组合（MVP） | B + 7 天后 3-5 题复习检查 | session_verified（near，CI 必报）+ retained(d=7)（更新 S） | near + 弱 far（前置链） | immediate + delayed(≥7d) | 文献充分（Q1/Q7），成本低 |

### 3.2 画像可见性

| 方案 | 可见内容 | 实证依据 |
|---|---|---|
| 隐藏 | 无 | —（无依据） |
| 静态结论视图 | 状态/待复习 | 面板类证据弱（Bodily & Verbert 2017） |
| **Scrutable OLM（推荐）** | 状态 + p̂ + n + 证据链（点击看"为什么"）+ 下次复习 | Bull & Kay 2007/2016、Long & Aleven 2016 |
| Negotiated OLM（+1 步） | 上者 + 学生可质疑/标记"我会了" | Bull & Kay 2013（驱动元认知）；未成年用户保守上线 |

### 3.3 事件标准

| 方案 | 成本 | 互操作 | 重放审计 | 迁移 LRS |
|---|---|---|---|---|
| 自研自由 JSONL（现行） | 零 | 无 | ✅（项目最扎实资产） | 需重写映射+可能改重放器 |
| **xAPI 形状 JSONL（推荐）** | 零+半天改 schema | 高（字段对齐 Caliper/xAPI） | ✅ | 一次映射器 |
| 完整 xAPI LRS | 引入服务/存储 | 最高 | ✅ | 直接 |
| Caliper Sensor | 引入端点 | 最高 | ✅ | 直接 |

## 4. 裁决（逐条）

| 现行设计 | 裁决 | 依据与动作 |
|---|---|---|
| held_out=2（2 族×1 题，2/2=verified） | **改造** | 二项 CI 下界 15.8%，mastery-testing 文献要求 15-25 题才能控制误判率；MVP 改 n=6（2 族×3 题），输出 p̂+95%CI；"verified"仅当 p̂≥0.8 且下界≥0.5；预算耗尽=partial 保留但永远带 n 与 CI |
| verified 语义（=session 内未见题族表现） | **保留（红线正确）** | 与 Barnett & Ceci 2002 一致：near-transfer 局部信号≠长期掌握；改名建议：`session_verified(near)`，新增 `retained(d=7)` 标签由 delayed check 颁发，"mastered"仅由跨 session 累积证据颁发 |
| 事件源 append-only JSONL 全量重放 | **保留** | 事件溯源+可重放审计是项目最扎实贡献（此前审计共识），与 xAPI LRS 模型同构 |
| 事件 schema（无 id/无 provenance） | **改造** | 对齐 Caliper 1.2 必填字段：`id`(UUIDv4)/`actor`/`action`/`object`/`eventTime`(ISO8601 ms UTC)；加 `schema_version`、`session`(registration 组)、`source∈{real,qa,synthetic}`、`agent_version`（=xAPI authority 角色）；幂等=按 id 去重 |
| review_due_at 双日期（7 天 vs 2.25S） | **替换** | 单一来源=FSRS 反解间隔。数学事实：R(t)=(1+t/9S)^-1 在 t=S 时=0.9，t=2.25S 时=0.8，t=7 天且 S=3 时≈0.79——三者是"同一公式在不同 desired retention 下"的伪装冲突。声明 desired retention r=0.9 → I=S；删除硬编码 7 天 |
| 无 provenance（真人/QA/合成混流） | **改造** | 无 provenance 的画像在 PIPL 第 8 条（数据质量）与可复现研究意义上都站不住；所有事件加 source 字段，默认投影只消费 real，QA/synthetic 仅用于回放测试与校准 |
| 无登录+localStorage+三年画像 | **保留（架构正确）+改造** | PIPL 28/31 使服务端账号+未成年画像成为 PIPIA 级高成本路径；local-first+导出/导入（事件源即档案）+一键删除是合规最优解，且有 xAPI personal data locker 先例；改造=补导出/导入/删除三键+专规文案+PIPIA 模板 |
| watch 完成非硬门 | **保留现状** | 即时验证本身承担把关（testing effect 环节），把 watch 当协变量而非资格门，与检索文献一致 |
| 即时验证=零间隔测试 | **补充** | 增加 delayed check（7 天，3-5 题），据结果用 FSRS 更新 S（Q1/Q7 文献） |

**MVP 验证协议建议（最小可辩护）**：
1. 视频后即时嵌入 2-3 道检索题（不进 held-out，testing effect 环节）。
2. 结束环节 held-out n=6（2 族×3），汇报 p̂+95%CI；≥4/6 且下界≥0.5 → `session_verified(near)`。
3. 7 天后 delayed check 3-5 题（覆盖本次学习知识点）→ `retained(d=7)`，通过者按 FSRS 更新 S。
4. OLM：学生可见 per-KP 卡片 {状态, p̂, n, 下次复习日期}；点开看证据链（最近事件）；预留"标记我会了"（negotiated）灰测。
5. 样本量口径：单 session n=6 只够 near-transfer 弱声明（CI 下界 60.7%）；"掌握"声明需 ≥14 题一次测或跨 session 累积（画像层，FSRS/信念合并），MVP 内禁止 n=2 二值化判定。

## 5. 未决问题

1. Semantic Scholar 全程 429，全部论文的引用数与摘要字段未获一手核验；文中依赖题名核验+领域公认结论，关键效应量（Kulik +0.5SD、Rowland g≈0.50、Adesope g≈0.61）数值标 [记忆回溯]，建议后续配 S2 key 或 Crossref citation 二次确认。
2. OECD 主站 403：PISA 2022 数学框架与 Learning Compass 2030 的框架细节未获一手全文。
3. 未成年 OLM 的动机证据基本来自欧美初等/中等样本；中国上海高中生的 scrutable OLM 效果无直接实证——建议 MVP 内做小型 A/B 与访谈，而非直接外推。
4. 监护人同意在"无登录"产品中的实现形式（本地弹窗声明+导出记录？）没有文献先例，属产品创新风险，需法务复核。
5. far transfer（综合题）验证是否/何时纳入闭环：文献说单知识点闭环做不到，产品要给出明确的天花板声明或引入"跨知识点综合诊断"第二层。
6. 合成数据用于 FSRS/信念参数校准时，若合成数据源自真实学生数据派生的分布，是否构成 PIPL 敏感数据处理（去标识≠匿名化，第 73 条），未决。

## 6. 参考文献表

**掌握学习 / mastery testing**
- Bloom, B. S. (1968). Learning for mastery. *Evaluation Comment*, 1(2). UCLA-CSEIP. [记忆回溯（灰文献）]
- Block, J. H., & Burns, R. B. (1976). Mastery learning. *Review of Research in Education*, 4, 3–49. DOI 10.3102/0091732X004001003 [题名]
- Kulik, C.-L. C., Kulik, J. A., & Bangert-Drowns, R. L. (1990). Effectiveness of mastery learning programs: A meta-analysis. *Review of Educational Research*, 60(2), 265–299. DOI 10.3102/00346543060002265 [题名]
- Guskey, T. R. (2012). Mastery learning. *Encyclopedia of the Sciences of Learning*. DOI 10.1007/978-1-4419-1428-6_1553 [题名]
- Millman, J. (1973). Passing scores and test lengths for domain-referenced measures. *Review of Educational Research*, 43(2), 205–216. DOI 10.3102/00346543043002205 [题名]
- Emrick, J. A. (1971). An evaluation model for mastery testing. *Journal of Educational Measurement*, 8(4), 321–326. DOI 10.1111/j.1745-3984.1971.tb00946.x [题名]
- Wilcox, R. R. (1976). A note on the length and passing score of a mastery test. *Journal of Educational Statistics*, 1(4), 359–364. DOI 10.3102/10769986001004359 [题名]
- van der Linden, W. J. (1982). Passing score and length of a mastery test. *Evaluation in Education*, 5(2), 149–153. DOI 10.1016/0191-765X(82)90015-5 [题名]
- Kingsbury, G. G., & Weiss, D. J. (1983). A comparison of IRT-based adaptive mastery testing and a sequential mastery testing procedure. In *New Horizons in Testing*. DOI 10.1016/B978-0-12-742780-5.50024-X [题名]

**迁移**
- Barnett, S. M., & Ceci, S. J. (2002). When and where do we apply what we learn? A taxonomy for far transfer. *Psychological Bulletin*, 128(4), 612–637. DOI 10.1037/0033-2909.128.4.612 [题名]

**OLM / 学习分析面板**
- Bull, S., & Kay, J. (2007). Student models that invite the learner in: The SMILI:() open learner modelling framework. *IJAIED*, 17(2), 89–120. DOI 10.3233/IRG-2007-17(2)02 [题名]
- Bull, S., & Kay, J. (2016). SMILI☺: A framework for interfaces to learning data in open learner models, learning analytics and related fields. *IJAIED*, 26, 293–331. DOI 10.1007/s40593-015-0090-8 [题名]
- Long, Y., & Aleven, V. (2016). Enhancing learning outcomes through self-regulated learning support with an open learner model. *UMUAI*, 27, 55–88. DOI 10.1007/s11257-016-9186-6 [题名]
- Bodily, R., & Verbert, K. (2017). Review of research on student-facing learning analytics dashboards and educational recommender systems. *IEEE TLT*, 10(4), 405–418. DOI 10.1109/TLT.2017.2740172 [题名]
- Matcha, W., Uzir, N. A., Gasevic, D., & Pardo, A. (2020). A systematic review of empirical studies on learning analytics dashboards: A self-regulated learning perspective. *IEEE TLT*, 13(2), 226–245. DOI 10.1109/TLT.2019.2916802 [题名]
- Valle, N., et al. (2021). The influence of task-value scaffolding in a predictive learning analytics dashboard on learners' statistics anxiety, motivation, and performance. *Computers & Education*, 173, 104288. DOI 10.1016/j.compedu.2021.104288 [题名]

**事件标准**
- IEEE 9274.1.1-2023, Experience API (xAPI). [全文：xAPI.com overview + IEEE 页确认编号]
- IMS Global. (2020). Caliper Analytics 1.2 Final Release. [全文]
- IMS Global. (2017). Competencies and Academic Standards Exchange (CASE) Service v1.0. [全文]

**检索实践 / 间隔**
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning. *Psychological Science*, 17(3), 249–255. DOI 10.1111/j.1467-9280.2006.01693.x [题名]
- Rowland, C. A. (2014). The effect of testing versus restudy on retention: A meta-analytic review of the testing effect. *Psychological Bulletin*, 140(6), 1432–1463. DOI 10.1037/a0037559 [题名]
- Adesope, O. O., Trevisan, D. A., & Sundararajan, N. (2017). Rethinking the use of tests: A meta-analysis of practice testing. *Review of Educational Research*, 87(3), 659–701. DOI 10.3102/0034654316689306 [题名]
- Yang, C., Luo, L., et al. (2021). Testing (quizzing) boosts classroom learning: A systematic and meta-analytic review. *Psychological Bulletin*, 147(4), 399–435. DOI 10.1037/bul0000309 [题名]
- Karpicke, J. D., & Bauernschmidt, A. (2011). Spaced retrieval: Absolute spacing enhances learning regardless of relative spacing. *JEP:LMC*, 37(5), 1250–1257. DOI 10.1037/a0023436 [题名]
- Karpicke, J. D., & Roediger, H. L. (2007). Expanding retrieval practice promotes short-term retention, but equally spaced retrieval enhances long-term retention. *JEP:LMC*, 33(4), 704–719. DOI 10.1037/0278-7393.33.4.704 [题名]
- Latimier, A., Peyre, H., & Ramus, F. (2020). A meta-analytic review of the benefit of spacing out retrieval practice episodes on retention. *Educational Psychology Review*. DOI 10.1007/s10648-020-09572-8 [题名]
- Dunlosky, J., Rawson, K. A., et al. (2013). Improving students' learning with effective learning techniques. *Psychological Science in the Public Interest*, 14(1), 4–58. DOI 10.1177/1529100612453266 [题名]

**长期画像 / 记忆（沿用 w4_memory.md 已核验条目）**
- Ye, J., Su, J., Cao, Y. (2022). A stochastic shortest path algorithm for optimizing spaced repetition scheduling (FSRS). *KDD 2022*. [题名]
- Su et al. (2023). Optimizing spaced repetition schedule by capturing the dynamics of memory. *TKDE*. [题名]

**合规**
- 中华人民共和国个人信息保护法（2021-11-01 施行），第 28/29/31/47/55 条。 [全文，DigiChina/Stanford 英译]
- GDPR Art. 8（儿童同意，默认 16 岁，可降至 13）。 [全文]
- FTC. Children's Online Privacy Protection Rule (16 CFR Part 312)，2025-04 修订版。 [全文，规则页+Federal Register 链接]
- FERPA, 20 U.S.C. §1232g。 [记忆回溯]

**标准框架（一手受限）**
- OECD. PISA 2022 Assessment and Analytical Framework（数学素养框架）。 [记忆回溯，主站 403]
- OECD. Learning Compass 2030。 [记忆回溯，主站 403]
