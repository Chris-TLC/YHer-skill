# YHer 技术报告 v2（全文初稿）

版本：2026-07-13 收官分支快照
代码基线：`demo-overnight-20260712`，已审计实现 SHA `d62aad4`（后续纯文档提交不改变运行时事实）
状态：pre-alpha、localhost、工程 Demo，不是效果研究结论

## 1. 摘要：从供给资产收敛为一个可证伪闭环

YHer 的当前问题不是“如何做一个万能 AI 老师”，而是更窄的一件事：在上海高中化学的一个知识点上，能否根据学生的真实作答逐步诊断，推荐与证据相符的讲解和视频，再用没有见过的题族独立验证，并把结果写回可重算的画像。

本轮把此前分散的三条链收敛为一个学生入口、一个 FastAPI 应用、一套 R5/v4 题源和一个事件日志。canonical 路径不再使用浏览器本地判分、固定 `0.78/0.42` mastery、首题复测或固定 `+0.16` 提升。服务端冻结诊断、练习和 held-out 三个集合，用四状态 belief、EIG 选择器、时间预算、签字视频路由和 FSRS 复习状态完成闭环。

这一结果仍只能称为 pre-alpha。自动化浏览器门已覆盖桌面和手机 12/12 条旅程且 `failures=[]`，离线测试为 569/569，根五引擎契约为 119/119。post-fix 真实 UI 已关闭首题计数、前置标题、100 字残句和错误层级支架四个内容问题；但合同另要求的 final `computer-use` 人工矩阵仍在收口。公开讲解是 DeepSeek 辅助的、以 verified standard solution 为权威来源的投影，不是模型自由生成化学推理。

## 2. 产品定义、用户旅程与当前能力边界

### 2.1 目标用户旅程

学生在首页选择年级、学习目的、知识点和 30/60/120 分钟时间档。服务端创建 session 后返回一个不透明 assignment；浏览器只拿到题面、选项、难度、来源标签和本 session 进度。学生提交后，服务端从私有 item key 判分，更新 belief，再根据 EIG 和前置竞争选择下一题。

诊断满足最小直接证据或预算条件后，状态机进入 learning checkpoint。checkpoint 绑定本 session 的已判作答、已验证标准解和推荐证据；学生必须显式确认后才能进入练习或 held-out。最终报告只读取 session 开始前基线和本 session 截止事件，展示四状态分布、session delta、FSRS 稳定度、7 日复习日期、失败原因和补强计划。

### 2.2 当前已经成立的能力

- 单一 same-origin Web/API 路径，默认 `127.0.0.1:8700`、单 worker。
- R5/v4 为唯一 canonical 学生题源，旧 v3 学生路由 fail-closed。
- 提交前不返回答案、rubric、item ID 或 family ID。
- MCQ、数值题走服务端确定性判分；自由作答在 provider 不可用时 deferred，不更新画像。
- 诊断、练习和 held-out 在 item 与内容题族两个层面强制不相交。
- 推荐、观看代理、seen segments、session ID 和画像投影都进入 append-only 事件链。
- 同一用户第二次进入可读取前次画像并避开已见视频段。

### 2.3 尚未成立的能力

- 没有真实学生留存、付费、提分或长期学习效果证据。
- 没有公网身份认证、监护人同意、数据保留/删除和生产监控。
- 已核 post-fix 样本满足完整起点、难度/错题数支架和 verified 标准解回连；这仍不等于模型可自由、稳定地生成化学事实或完成任意动态深讲。
- 27 个开放节点不等于 135 个 KG 节点全部可诊断。
- 合成演示和 founder QA 只证明工程行为，不证明教育效果。

## 3. 数据谱系：从 PDF 视觉抽取转向 DOCX-native

早期 v3 题库共有 6,438 行，其中历史源分布为 4,522 道 DOCX、573 道 DOC、1,343 道 PDF。视觉资产管线建立了题目级 manifest、质量门、35 题视觉 eval 和多模态理解结果，但也暴露出一个根本问题：从整页截图反推题目边界、公式和图文关系会累积裁切、上下文和对象丢失风险。

2026-07-03 的架构转向把 Word 原生结构作为优先源。DOCX 内部已经保存段落顺序、表格、OMML、DrawingML、嵌入图片和部分 MathType/WMF 对象；因此新管线按对象类型提取，保留无法无损转换的媒体引用，再通过渲染回验确认学生是否能够理解和作答。PDF 视觉路线保留为无 Word 源时的兜底，而不是默认入口。

当前 v4 主库 `chemistry_v4_1_3329.jsonl` 为 3,329 行。R5 台账 `usability_r5_v1.jsonl` 对 R1-R4 服务宇宙的 2,526 行逐项记录 `r5_serve`、阻断原因、reviewer 和来源。当前 `r5_serve=true` 为 1,202；canonical catalog 进一步要求可信 v3 映射、答案可判、`answer_verification=passed` 和无媒体依赖，最终得到 973 个 trusted items、400 个确定性可判 item、963 个独立内容题族和 27 个开放节点。

历史 v3 的 6,438 行仍作为来源和审计资产存在，但不再进入 canonical 学生题源。v3 与当前 v4 R5 的 item ID 不能直接跨版本等同；去重和分区使用 v4 内部的 content family，而 official 数据比对继续使用 `(group_key, section_num, q_num)` 复合键。

主要证据路径：

- `data/item_bank/v4/chemistry_v4_1_3329.jsonl`
- `data/item_bank/v4/usability_r5_v1.jsonl`
- `data/item_bank/chemistry_v3_6695.jsonl`
- `data/knowledge_graph_150_enriched.jsonl`
- `PROJECT_HANDOFF/ARCHITECTURE_PIVOT_DOCX_NATIVE_2026-07-03.md`
- `PROJECT_HANDOFF/VISUAL_ITEM_QUALITY_GATE_EXECUTION_REPORT_2026-07-01.md`

## 4. QA 6-16、R5 白名单与双向金标

题库治理最大的工程教训是：召回已知坏题不代表审查器安全。Batch 15 的新文本规则虽然命中了坏题金标，却在当时的服务好题上误伤 169 条。根因包括直接扫裸 JSON、把 formula/media/OMML/LaTeX 节点中的合法内容当普通文本，以及只验证召回、不验证精度。

Batch 16 将结构化节点先替换为哨兵，再运行文本规则，并建立 236 条已知好题精度金标。该版本对当时服务白名单达到 0/236 误伤。此后任何会改变 R5 的规则都必须同时满足：

1. 已知坏题召回全命中；
2. 已知好题精度金标零误伤；
3. 结构化节点感知；
4. 变更前备份、逐行 manifest 和单命令回滚；
5. 变更后与回滚恢复后重复验证。

M5 按该纪律处理 10 条坏题报告：5 条从 R5 排除，2 条按 DOCX/标准解做机械修复，3 条判为误报并关闭。双向金门结果为 precision `0/236`、recall `55/55`，pre-write、post-write 和 post-restore 均有记录。R5 可服务数因此从历史快照 1,207 诚实下降到 1,202。

变更备份永久保留在 `data/_backup_pre_demo_bad_reports_20260713/`；逐行 manifest 和回滚演练位于 `/tmp/yher_demo_overnight/manifest/demo_bad_reports/`。这些数字表示质量治理结果，不表示题库 1,202 条都适合当前确定性诊断；canonical 的答案信任门还会继续收缩到 400 条。

## 5. 五引擎：从算法契约到生产接线

### 5.1 Mastery

`engine/mastery.py` 不把掌握度压成一个固定分数，而是维护四状态 belief：M（已掌握）、P（前置缺口）、C（概念/因果链混淆）、U（证据不足）。本地题型似然和自由判分似然都经过合法化，单条 LLM 证据的似然比上限为 3:1。held-out 证据可初始化或更新 FSRS stability，读取画像时按时间投影衰减。

### 5.2 Selector

`engine/selector.py` 对候选题计算信息增益，排除 held-out、已见 item 和已见 family。P/U 竞争可触发前置题，C 占优时继续考目标概念；至少取得三条目标节点直接证据前不能因为表面收敛过早停止。

### 5.3 Planner

`engine/planner.py` 定义 30min、1h、2h、3h+ 四档预算。30 分钟只承诺浅诊断，不承诺病因完全收敛；预算耗尽会保存 checkpoint 并标记 partial，而不是伪造完整验证。M7 合成矩阵只使用面向 Demo 的 30/60/120 分钟三档。

### 5.4 Recommender

`engine/recommender.py` 接收 belief、年级、学习目的、预算、签字轨道和 seen segments。它拒绝重复 ID、非法 audience/efficacy、首段超预算和同一 `(bv,p)` 跨节点重复；每次推荐写入 served 与 unserved top-k 的 propensity 快照。

### 5.5 Memory

`engine/memory.py` 只允许突破、belief 翻转、顽固错误、高信息追问和高价值观看等事件进入高层记忆。画像可完全从 append-only events 重算；L3 叙事不会进入诊断追问等不应受故事影响的触点。

五引擎原始根级契约为 119/119。生产实现通过 `core/learning/session_service.py`、`item_catalog.py`、`events.py` 和 store adapters 接入同一状态机，而不是复制一套“Demo 算法”。

## 6. 真诊断、讲解信任边界与 held-out

SessionService 在开始时按 family 排序并先冻结两个 held-out family，再从剩余题族构造诊断和练习分区；前置题只能追加新的 family。assignment ID 由服务端生成，浏览器不能指定 item。运行时和持久化加载时都检查三分区交集，损坏状态会 fail-closed。

MCQ 使用规范化选项精确判分，重复字母如 `AA` 无效且提供零信息。数值题同时检查数值容差与单位。自由回答只有在 item 含 reviewed answer 与 rubric、且 provider 返回合法 confidence/likelihood/correct 时才更新 belief；否则返回 deferred。

M6 内容审计发现，早期 provider 自由讲解会产生 `NaClO3`、`6FeCl2`、`O2F2` 等硬化学错误，并把全对 session 误说成薄弱。修复后的信任边界是：

- 只有 `answer_verification=passed` 的题可进入确定性诊断；
- 公开 worked steps 只取 verified `standard_solution.solution_steps`；
- DeepSeek 可以选择原文步骤和有限安全策略，但不能把自写化学事实投影到学生端；
- 全对/错题数量由服务端 result summary 决定；
- provider、模型名、prompt、usage 和 cost 只留内部事件。

这保证“讲解不越过标准解”。`d62aad4` 进一步避免截断 key insight，把难度支架与真实错题数支架分开；post-fix 真实 UI 的 j01 已看到完整起点、难度支架和两错额外支架，j03 已完成 strong 3/3、practice 与两个 held-out 并得到 verified 结局。该证据支持受约束 authoritative projection 的当前范围，不支持把系统宣传为已经可靠掌握自由化学事实生成。

held-out 完成后，报告从 session 前基线和本 session 截止事件重算 belief。通过时 outcome 为 `verified`；任一独立验证失败时为 `needs_reinforcement`；经历预算耗尽再完成时为 `partial`。失败报告列出错误码计数、不同题族补强步骤和下一次验证约束，不使用“提分”“治愈”或长期疗效话术。

## 7. 视频格局层与可审计推荐

频道目录覆盖 551 个 BV、1,136 个分 P，现有语料中的 842 个 `(bv,p)` 均有 catalog 记录。轨道草稿包含 43 个实体；本轮只对“实体 ID 与 catalog 标题足以支持轨道判断”的 30 个签字启用，reviewer 为 `codex_sol_20260713`，其余 13 个保持 neutral。

签字配置经过 `scripts/build_curriculum_runtime.py` 与 KG 绑定，得到 148 个路由键、345 个候选段。148 是 KG 名称、别名和路由键的运行时集合，不是把知识图谱从 135 节点扩成 148 节点。只有有机字幕块中 8 个段具有 `needs_human=false` 的真实时间锚，覆盖 6 个路由键；其他节点返回视频级链接，并明确从视频开头按本次预算观看，不生成假的 `t=` 参数。

每次推荐都产生不透明 `rec_id` 和私有 binding。watch API 只接受本 session 已发出的 `rec_id`，seen segment 以 `(bv,p)` 投影到画像，并在后续 session 去重。主要证据：

- `config/curriculum/track_map_v1.yaml`
- `core/learning/assets/curriculum_runtime_v1.json`
- `core/learning/curriculum.py`
- `scratchpad/organic_chunks_timestamped.jsonl`（工作区来源，641 行）

## 8. 内化生成五轮：成立的是分层供给，不是替代真题

内化生成实验回答的是“AI 能否在受控门下提供新练习”，不是“AI 题能否替代真题诊断”。五轮形成三条供给车道：

1. 从头生成文字题：化学正确率约 95%-97%，但专家对抗盲判中约 87% 可区分，因此只适合低风险练习。
2. Style-transfer：R5 盲判 40 题总区分率 65%，公平口径 60%，转化题化学正确 19/20，适合经母本和二次 QA 后进入高保真练习位。
3. 程序渲染与图锚换皮：曲线、装置、流程、有机路线和晶胞均得到真样本，但必须使用图必需门、VLM solver、答案唯一性和同图同考点查重。

五轮总成本为人民币 13.87 元，盲判 153 题，人工亲验图题 70 余道。实验同时暴露模板塌缩、条件句对冲、配额改名、seed backfill、宽容 solver 和超纲 solver 等 Goodhart 路径。最终 R6 门要求逐干扰项判假、考纲适定、母本资格、格式复查、item/hash 锚定配额和内容级查重。

这些结果是历史受控实验的供给证据，不是生产学习效果。当前 canonical 首测与 held-out 仍只使用 R5 真题；AI 诊断库的 119 文件、200 题也没有进入关键判定位。

## 9. 本地安全、可逆 official 写入与复现

Demo 只监听 `127.0.0.1`，CORS 收紧到 same-origin，并使用请求体大小限制、滑动窗口速率限制和有界 limiter state。上传接口清洗 user ID，校验 MIME、扩展名和文件签名，流式写临时文件并在失败时清理。匿名 append 类端点同样受到体积和频率门约束。

`.env*` 权限为 0600，重复 key 已去除；代码、报告、截图和 manifest 不记录 key 值。凭据轮换、git push、公网部署和对外发布仍属于用户动作，本轮没有执行。

official 修缮遵守可逆性纪律：先创建 `_backup_pre_*_20260713`，再生成逐行 before/after、整文件 SHA-256 和 rollback command。真实回滚演练后恢复到 post-write 状态，备份不删除。

本地启动入口 `deploy/run_demo.sh` 使用 Python 3.11+、repo-local `.venv-demo`、固定 `requirements-demo.txt` 和单 worker。离线复现可显式设置 `YHER_ENABLE_PAID_LLM=0`；缺 key 或 provider 失败不会阻塞确定性诊断，但自由作答不会伪装为已判分。

## 10. M6 工程验证与 M7 合成重放

M6 的机械验证结果：

- 仓库离线套件 569/569；
- 根五引擎 119/119；
- 自动化桌面 1280×800 与手机 390×844 共 12 条浏览器旅程，12/12，`failures=[]`；合同点名的 final `computer-use` 人工矩阵仍在收口，不能用自动化数量替代；
- 覆盖全对、全错、交替、中途暂停、断点恢复、错误态和连续 session；
- console 无 error，未出现意外 4xx/5xx、水平溢出或公式资源缺失；
- API fresh run `20260712T224150Z`（服务 SHA `ce0700f`）中，确定性样本 p95 为 6.704 ms，低于 500 ms 门；LLM 全文最大 16.497 s，低于 20 s 门；端点为非流式 JSON，因此不声称首 token <5 s；
- 提交前响应的私有字段、模型标识和凭据扫描为零命中；
- item/family 三分区交集为空，服务题属于当前 R5；
- 同用户两次 session 的画像和 seen segment 连续性成立。

机械门不等于内容门。post-fix 真实 UI 已关闭首题计数、前置标题、100 字残句和错误层级支架问题；这里签的是有 verified standard solution 边界的 authoritative projection，不是自由化学事实生成。`failures=[]` 也不覆盖尚在收口的 final `computer-use` 人工矩阵。

M7 另外建立 24 个 `SYNTHETIC_DEMO` 场景。场景层严格覆盖 30/60/120 分钟 × `verified/needs_reinforcement/partial/paused` × 每组合 2 个；8 个场景含两个 session、16 个含一个 session，共 32 episodes。计划覆盖 28 个节点，但当前开放节点只有 27 个；“化学反应速率”因答案信任门后不足五个确定题族，被重放为预期关闭。

重放器直接使用正式 ItemCatalog 和 SessionService，注入 `synthetic=True`、确定性时钟/ID、零 token grader 和离线讲解 provider。输出只能进入独立临时 store，明确拒绝 `data/local_store`、`data/study_logs` 和仓库 `data/`。每个 case、session、event、profile projection 和 transcript 都有 `synthetic:true`；两次重放 digest 相同，真实日志树 hash 前后不变。

## 11. 工程教训、限制与下一步

第一，能力边界必须由可验证的数据门决定。答案信任门使 deterministic item 减少到 400、开放节点降到 27，这比维持漂亮的 28 更可信。第二，测试通过不等于内容正确；4 条浏览器讲解硬错误只有真实读内容才会被发现。第三，LLM 应处在受约束的选择/组织边界内，标准答案、判分和画像更新必须由服务端证据控制。

第四，任何质量规则都要同时测 precision 和 recall；Batch 15 的 169 条误伤证明单向金标会把安全门变成数据破坏器。第五，演示、QA、合成和真人数据必须在身份、目录和叙述上同时隔离，否则很容易把工程完成度误写成教育证据。

当前最合理的后续顺序不是扩科或继续扩大题库，而是：

1. 收口 final `computer-use` 人工矩阵，并保留逐旅程截图、console/network 与结局证据；
2. 由可信化学教师抽查开放节点、标准解投影和失败补强建议；
3. 建立未成年人同意、最小化、保留与删除规则后，再做小规模真实可用性测试；
4. 把 synthetic、QA 和 founder data 从所有真实效果分析中永久排除；
5. 只有得到独立 held-out 和跨日证据后，才讨论学习效果或商业化。

这份 v2 是工程事实初稿。它刻意保留 pre-alpha、authoritative projection、人工矩阵未收口和 27 节点边界，因为一个可复查的有限系统比一个无法证伪的“AI 私教”叙事更有价值。
