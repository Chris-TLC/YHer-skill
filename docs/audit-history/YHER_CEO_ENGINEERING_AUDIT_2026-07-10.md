# YHer 当前阶段 CEO + 工程审计

日期：2026-07-10  
范围：`/Users/mac/Desktop/项目文件夹/Tools` 全工作区  
性质：只读审计与建议；不改变现行 manual/official/serve 签字权，不授权任何数据 apply 或付费管道恢复。

## 0. 结论先行

YHer 的核心方向是对的：先做上海高中化学，在一个具体知识点上完成“诊断 -> 视频/练习推荐 -> 独立验证 -> 画像更新”。真正的问题不是缺愿景、缺数据或缺算法，而是过去两周的工作没有把这些资产收敛成一个真实、可测、可收费的产品闭环。

当前阶段应定义为：**pre-alpha、pre-PMF、数据和算法资产较强，但产品集成与商业证据很弱**。

最准确的一句话评价是：

> 方向约 70% 正确，近期优先级约 40% 正确；供给侧和治理侧做得远超同阶段，需求侧、集成侧和真实学生验证明显落后。

未来 7 天不应继续增加新引擎、扩科、长尾洗题或恢复全量 ASR。唯一主线应是：选一个知识点，使用 v4 R5 真题、两段已人工签字的视频、真正的 held-out 复测和持久化画像，做出一条不能靠演示常量伪造的闭环。

## 1. 审计口径与边界

### 1.1 覆盖范围

- 当前工作区有 97,458 个普通文件、141 个符号链接，总大小 21.77 GiB（不含嵌套 Git 对象）。
- 主要资产：47,158 PNG、18,525 WMF、11,776 JSON、7,261 SVG、5,153 JPG、1,874 SRT、848 PDF、810 Python、809 DOCX、613 JSONL、370 Markdown。
- 逐路径纳入元数据清点；所有第一方 JSON/JSONL/Python/UTF-8 文本做解析检查；DOCX/PDF/PNG/JPEG/WMF/音频做容器或解码检查。
- 对 canonical memory、当前决策、历史弯路、事故、战略/治理、视觉 QA、生成题五轮、最新 briefs/handoffs、生产关键代码、测试和运行态做人工语义审读。

### 1.2 “完整阅读”的诚实定义

不能声称人工逐像素理解了 7GB 图片或逐题判断了 32 万行结构化数据的化学正确性。本审计采用两层证据：

1. 全覆盖机器检查：每个路径、文本语法、结构化数据和二进制容器均进入检查。
2. 风险导向人工审读：会改变产品、商业、数据口径、签字权或付费行为的文件逐条阅读。

因此，“文件没坏”不等于“题目化学正确”，“单测通过”也不等于“学生闭环有效”。

## 2. 当前评分

| 维度 | 当前分 | 判断 |
|---|---:|---|
| 产品方向 | 7/10 | 化学窄闭环正确；全科 OS 已被正确后置 |
| 近期优先级 | 4/10 | 评审、门、引擎和批处理挤压了真实闭环 |
| 外部需求证据 | 1/10 | 没有可信真实学生留存或付费证据 |
| 产品闭环 | 2/10 | 页面可展示，但诊断/复测/画像存在演示逻辑和断链 |
| 数据资产 | 7/10 | R5、v4、视觉治理和生成验证扎实，但版本割裂 |
| 架构设计 | 6/10 | 新五引擎设计有质量，模块边界基本清晰 |
| 生产集成 | 2/10 | 根引擎、API、两套前端、v3/v4 尚未形成单链 |
| 测试工程 | 5/10 | 断言很多，但默认 pytest 会联网、付费、写远端 |
| 安全/隐私 | 1.5/10 | 可做本地原型，不能直接公网给未成年人使用 |
| DX/可复现性 | 2.5/10 | 无锁文件、无 CI、入口过多、README 与现状冲突 |
| 商业化成熟度 | 1/10 | 有合理假设，没有付款、复用或结果数据 |

## 3. 当前硬事实

### 3.1 数据与质量治理

- KG：135 个节点。
- v3 item bank：6,438 题；现有 quality gate 允许诊断/画像 4,350 题、练习/教学 4,586 题。
- v4：3,329 行；R1-R4 服务宇宙 2,526；R5 白名单实际 1,207 题，覆盖 61 个节点，54 个节点至少 3 题。
- v3 与当前 v4 R5 的 `item_id` 交集为 0，不能跨版本直接做已见题排除、证据追踪或结果合并。
- AI 诊断题库：119 节点、200 题；仍记录有三个 gold diagnostic 硬空洞。
- 生成题五轮已证明：化学正确率约 95%-97%；从头生成风格仍明显；style-transfer 更接近真题；图锚与程序渲染可行。正确生产分层应继续是“真题用于关键诊断，转化题用于高保真位，从头生成用于低风险练习”。

### 3.2 新引擎

根目录已有：

- `engine/mastery.py`
- `engine/selector.py`
- `engine/planner.py`
- `engine/recommender.py`
- `engine/memory.py`

六组根级契约测试共 119/119 通过。它们是有价值的算法资产，但目前只被根级测试引用；FastAPI 生产代码没有 import 它们。

### 3.3 产品覆盖

- UI 提供 65 个父级知识点选择，其中 59 个在旧链路上至少有 3 道 formal 题、视频和一题 verification。
- 用户还可以选择 135 个细分节点；按实际细分节点算，只有 36/135 同时满足 formal>=3、视频和 verification，98 个细分节点 formal=0。
- 当前“看起来覆盖广”主要来自父节点回退，不是细粒度诊断真正打通。

### 3.4 运行态

- FastAPI 进程从 2026-07-01 启动，监听 `127.0.0.1:8600`，未使用 reload。
- 运行中 OpenAPI 只有 12 条旧路由，缺少 `diagnosis_prep`、verification、next plan 和全部 v4 study 路由；根路径返回 404。
- 磁盘最新代码可导入 22 条路由并能提供 `/`、`study.html` 和 v4 API，证明“磁盘代码”和“当前服务”已经漂移。
- Streamlit `8504` 是 6 月 28 日启动的管道 dashboard，不是当前学生产品。

### 3.5 真实使用证据

- `data/study_logs` 共 314 行；只有 2 个非空 session。
- 92 次 self mark 的中位思考时间约 1.65 秒、中位总耗时约 3.30 秒；一个 session 连做 91 题，明显是 QA 点测，不是学生学习。
- 学生持久化目录：0 students、53 local sessions、0 events。
- `yher_memory.db` 六张业务表全部 0 行。
- 10 条坏题报告全部 pending：2 条答案错误、2 条题面缺失、6 条未选原因；无人签字处理。

结论：当前不能声称有真实留存、学习效果或用户画像数据。

### 3.6 在途任务

**B 站频道目录**

- 551 BV、1,136 分 P；842 个语料对全部有 catalog 行。
- 43 个轨道实体全部 `needs_human=true`、reviewer 为空；尚未签字。
- 当前 `load_track_map()` 会接受 reviewer 为空和 `needs_human=true` 的实体，这是治理漏洞，不能直接接生产。

**豆包 2.0 ASR**

- 停在 P1 57/188，raw 共 62 文件（P0 5 + P1 57）。
- `BV1Qi4y1R7tW` 存在系统 page offset；P003/P007/P009 三条连续 0 对齐。
- 当前合并覆盖约 27.74%；原始 priority scope 理论上限约 54.36%，与 70% all-chunks 验收门冲突。
- preflight 已正确阻止原样恢复；final/asr2/LEDGER 均未生成。

结论：继续暂停。一个节点 MVP 不需要先完成全库 ASR。

## 4. P0 工程问题

### P0-1 诊断和复测目前不具备测量效度

证据：

- `core/tutor/product_loop.py` 把 `standard_answer` 直接放入 formal question JSON。
- `apps/web/index.html` 不调用 `/session/{sid}/diagnose`；浏览器本地比较答案，并写死“答对 0.78、答错 0.42”。
- 初诊结果没有落入服务端 session，API profile 也没有被前端调用。
- `build_post_video_verification()` 每次重新取该节点第一道诊断题，不排除首测题。对所有有题的 37 个实际 KG 节点，首个 formal item 与 verification item 100% 相同。
- verification 只有 1 题，答案又随 GET 发给客户端；POST 使用精确字符串比较。
- 通过后画像固定 `+0.16`，最低抬到 0.58，不是 mastery 引擎的后验更新。

影响：任何“视频后提升”数字都不可作为疗效或商业证据。

### P0-2 两个学生产品、两套题库、三代引擎互不相认

当前实际结构：

```text
index.html -> api_server -> core.tutor -> v3 + quality manifest
study.html -> api_v4_render -> v4 R5 -> random choice
root engine/* -> tests only
```

这会造成：

- R5 已签服务门没有约束主诊断页；
- v4 练习事件不能更新 v3 会话画像；
- 新 selector/recommender/memory 无生产输入；
- Docker 从子仓库构建，根目录 `engine/` 根本不在镜像上下文。

必须收敛为一个仓库、一个 API、一个题库入口、一个事件日志和一个学生页。

### P0-3 凭据与公网安全不合格

- ASR brief 的 curl 示例含明文凭据；必须轮换并脱敏，不能只删除文本。
- `.env` 和 `.env.bak.*` 权限为 `644`；`.env` 还有重复的豆包 key 定义。
- 默认 pytest 的失败 diff 曾把进程环境中的真实 key 打进测试输出；密钥可能进入终端或 agent transcript。
- API 接受客户端传入 `api_key`；公网产品不应让学生浏览器携带供应商 key。
- FastAPI CORS 为 `*`；无认证、无速率限制、无请求体上限。
- `/upload/homework` 把未清洗的 `user_id` 拼入路径，可目录穿越；文件整块读入内存，且不验证 MIME/扩展/大小。
- render report、study event、bad report 都可匿名无限 append，存在磁盘耗尽风险。
- `start_server.sh` 对外监听并关闭 Streamlit XSRF；不能用于公网。
- 学生是未成年人，现有日志没有同意、保留期、删除和最小化规则。

### P0-4 近期成果没有版本保护

- 唯一 Git 仓库是 `yihuier-chemistry-skill`；工作区根不是仓库。
- 主仓库只有 64 个 tracked 文件、4 次提交。
- 审计测试前有 69,345 个非忽略文件未跟踪；测试新建/追加 `data/study_logs/propensity_20260710.jsonl` 后当前为 69,346 个、约 5.23 GiB，其中含 59 scripts、29 tests、12 apps、8 core 和大量数据。
- 当前 `main` 与 `origin/main` 同为 `5be25aa`，只能证明远端保存了旧状态，不能保护最近一周成果。
- 根 `engine/`、根 `tests/`、`PROJECT_HANDOFF` 也未受 Git 保护。

不要执行 `git add .`。先把代码/测试/文档与 5GB 数据拆开：代码进入可审查提交，大数据用 hash manifest + 对象存储/备份策略。

### P0-5 没有真实需求和付费证据

当前最大风险已经不是“题还不够好”，而是“没有学生证明这个流程比自己找视频、刷题或问老师更好”。继续内部评审不能替代这条证据。

## 5. P1 工程问题

### P1-1 数据 schema 已建，但生产事件没有接线

- study 前端调用 `/study/next` 时不传 `session_id`；本轮产生的 propensity 行 session 为空。
- study 前端不调用 `watch_proxy` 或 `followup_triage`。
- mastery state、efficacy、L2、L3 只有 schema/纯函数，没有生产 writer。
- `rec_served` 从未在产品推荐路径写入。
- `StorePort.save_student/load_student/append_event` 仅在测试里被调用，生产代码没有持久化 StudentModel。

### P1-2 测试默认行为可能联网、花钱和改远端

- `test_llm_v3.py`、`test_cache_v3.py`、`test_t6_regression.py` 在模块导入时读取真实 `.env` 并执行集成脚本。
- T6 可能调用 3 次付费 LLM，并读写 Supabase。
- 默认 `pytest` 会递归收集 `data/_backup_*` 中同名测试，产生 import mismatch。
- 测试收集曾触发 BGE/Hugging Face 网络读取并占约 4GB 内存。
- 离线限定套件结果为 252/253；唯一稳定失败是“有 Node、无 KaTeX 包”时不正确降级。另三条失败是测试顺序污染，单文件运行 28/28。

需要 `pytest.ini`/`pyproject.toml` 锁定 `testpaths=tests`、`norecursedirs=data`，付费测试加显式 marker 和 `RUN_PAID_TESTS=1`，默认 CI 禁网。

### P1-3 部署和入口不可复现

- `requirements.txt` 只有宽泛下限，无 lock；pytest 不在依赖中；无 CI。
- Docker 默认跑旧 `apps/app.py`，不是当前窄闭环。
- `index.html` 默认 API 是 `http://127.0.0.1:8600`；远程学生打开后会请求学生自己电脑的 localhost。
- apps 目录同时保留多个 Streamlit、HTML、dashboard 和历史入口，没有 active/deprecated 标记。
- `/health` 不返回 git SHA、数据 manifest、引擎版本或启动时间，无法发现陈旧进程。

### P1-4 新引擎接线前还有边界问题

- selector 期待 numeric difficulty 和 `mcq|numeric`，实际题库常用 `T1-T4` 和中文题型；缺 adapter 会 TypeError/KeyError。
- `NodeBelief` 不校验四维概率、非负和归一化。
- memory 对 numpy vector 使用 `if not a` 会报 ambiguous truth；不同维向量被 zip 静默截断。
- memory 的 `max_refs` 参数可绕过“每 session 最多 1 条”硬门。
- recommender 接受 unsigned track entities；不校验 track 引用、重复 ID、audience 范围或 efficacy 范围。
- recommender 允许第一段视频即使超过整个预算；同一 `(bv,p)` 可在多节点内重复推荐。
- `append_rec_served` 只有内存重试队列，无持久化、上限和幂等写语义。

这些不否定算法方向，但说明接线任务必须包含 adapter、schema validation 和集成测试，不能只 import 后宣告上线。

## 6. 历史路线逐项评价

| 路线/转折 | 评价 | 应保留 | 应停止或修正 |
|---|---|---|---|
| 初始 RAG 化学私教 | 问题真实，作为检索基线有价值 | KG、真视频来源、标准解锚点 | 不把 RAG 问答当完整私教 |
| 化学/数学/物理并行 | 战略时机错误 | 数学物理资产冷存 | 化学留存前不继续扩科 |
| 纯文本 PDF 抽取 | 合理低成本基线，不适合图题主路线 | 答案页/纯文本对照 | 不再作为化学主抽取 |
| 单次视觉模型一把切题 | 明确失败 | 局部视觉转录 | 禁止多任务同 prompt 一步完成 |
| 500 黄金题作为核心供给 | 作为供给太小，作为评测正确 | 私持 gold/eval | 不恢复“500 题撑产品”叙事 |
| 模仿老师口癖/金句 | 被正确废弃 | 视频元数据、适用人群、教学目标 | 旧 language fingerprint 文档标 historical |
| v3 多阶段视觉管线 | 相比一步视觉是正确进步 | 三重验证、停止条件、失败台账 | 不继续重跑已失败长尾 |
| DOCX-native 架构翻转 | 很正确，利用原生结构降低视觉负担 | 原生结构优先、资产一等公民 | 不回到全页视觉重做 |
| R5 白名单与 QA 6-16 | 项目最强资产之一 | 白名单、双向金标、回滚、复合键 | 长尾修复已边际坍缩，转后台 |
| AI 内化/生成五轮 | 终于完成关键假设验证，价值高 | 三车道分层、R6 v4、不可自签 | 不再追求从头生成“像真题” |
| 前端 v1/练习舱 | 是必要 forcing function，但实现仍是假闭环 | 报坏题、RIR、事件理念 | 两个学生入口必须合并 |
| 五个诊断引擎 | 技术质量好，顺序偏早 | mastery/selector/recommender 核心 | memory/L3 在真实数据前不继续扩建 |
| B 站全频道目录 | 正确，解决视频格局盲区 | 551 BV/1136 P catalog | 先签 MVP 两段，不等全表 |
| 豆包全量 ASR | P0 小试合理，P1 scope/门冲突 | page offset 事故与 preflight 经验 | 当前暂停，不为完成率恢复 |
| Claude 大脑 + Codex 发包 | 高风险数据 apply 很有效 | brief、权限、独立审查、原始证据 | 不让每个普通开发任务都走重型发包 |
| 多模型“共识” | 可发现不同错误模式 | maker/reviewer 异构 | 模型一致不等于用户证据或签字 |

总体上，项目的转弯能力是优势。问题不是转弯太多，而是每次正确转弯后又容易把新方法扩成大工程，直到真实学生再次被后置。

## 7. 商业判断

### 7.1 最早可卖的不是“AI 教育操作系统”

最早可测试的产品应是：

> 上海高中化学单专题 45 分钟补弱处方：3-5 题定位错因，精确到 B 站分 P/片段，学后用不同真题验证，给出下一步。

推荐第一专题：**氧化还原反应中的电子转移与配平**。原因：v4 R5 供给多、判分相对客观、视觉依赖较低、学生痛点普遍、闭环实现风险小。第二候选是盐类水解；有机推断价值高但视觉和自由答案复杂度更高，不适合第一条工程验收链。

### 7.2 首轮变现实验

1. 先给 5 名同学免费，但必须完成一次完整 session 和 10 分钟访谈。
2. 下一批提供 `19 元 / 7 天 / 1 个化学专题`，而不是月订阅或复杂额度。
3. 不以收入为首要目标，以“有人愿意付款而不是继续自己找视频”为证据。
4. 暂不卖给陌生公众；先解决未成年人隐私、内容版权和公网安全。

### 7.3 真正的壁垒

题库数量不是主要壁垒。长期壁垒应是：

- 诊断题选择与错因后验；
- 推荐候选、propensity 和 `rec_served`；
- 视频后 held-out 结果；
- 哪类学生被哪类片段真正帮助；
- 跨 session 的可信画像和复习时机；
- 真实学生反馈形成的 product taste。

如果没有这些结果数据，5GB 文件只是供应资产，不是商业护城河。

### 7.4 30-session 停止条件

完成 30 个可信 session 后复盘：

- 少于 60% 完成诊断；
- 少于 50% 完成视频后验证；
- 少于 30% 在 7 天内主动复用；
- 没有人愿意支付 19 元；
- 多数学生认为“不如自己找视频快”；

若同时出现两项，不扩节点、不扩科，先重新访谈和改 offer。不要用更多架构掩盖需求问题。

## 8. 推荐的目标架构

```text
一个学生 Web（same-origin）
        |
一个 FastAPI 应用
        |
Session Service
  |-- v4 R5 ItemRepository（唯一题库入口）
  |-- Mastery + Selector + Planner
  |-- Signed Curriculum + Recommender
  |-- Append-only Event Store
        |
Profile Projector（可从事件重算）
```

### 8.1 第一阶段只接这几件事

- 把根 `engine/` 移入主仓库正式 package，并纳入锁定依赖和 CI。
- 只使用 v4 R5；为 MVP 人工钉死 3 诊断题、2 练习题、2 held-out 题，集合必须不相交。
- answer key 永不下发客户端；服务端记录 item assignment 和回答。
- MCQ/数值题确定性判分；自由回答才调用低成本模型并受 rubric/likelihood cap 约束。
- 签两段具体视频；不等 43 个实体全签。
- 每个推荐、点击、返回、复测和 mastery update 写事件。
- 画像从事件重算，禁止固定 `+0.16` 或 UI 常量。

### 8.2 暂缓

- L3 叙事蒸馏；至少有 10 次真实累计作答再上屏。
- 大规模 efficacy 学习；先确保日志可关联。
- 全频道 ASR；MVP 两段可先人工标时间。
- Supabase 复杂云架构；5 人 pilot 用 SQLite WAL + migration 足够。

## 9. 路线图

### 未来 7 个工程日

**Day 0：止血**

- 轮换 brief/.env 涉及的凭据；brief 替换成环境变量占位。
- 权限收紧到 600；付费测试默认禁用。
- 只提交代码/测试/文档的恢复点，不提交 5GB 数据。

**Day 1-2：收敛**

- 确定唯一 repo/package/app；根引擎进入主仓库。
- v4 R5 成为唯一学生题库入口。
- 只开放一个专题，其他节点隐藏而不是“整理中”。

**Day 3-4：真闭环**

- 服务端选题、判分、belief 更新。
- signed 两段视频推荐并写 `rec_served`。
- held-out ID 与所有首测/练习 ID 强制不相交。
- 画像跨 session 持久化。

**Day 5：验证**

- API contract + offline unit/integration + Playwright 手机/桌面。
- `/health` 输出 git SHA、engine/data/track version、started_at。
- 默认测试零网络、零付费、零 official/远端写。

**Day 6-7：真实试用**

- 用户本人跑 3 条完整链。
- 阻断问题清零后邀请 3-5 名同学。
- 不新增功能，只观察完成率、信任点和推荐是否省事。

### 30 天

- 5-10 名学生，至少 20 个可信 session。
- 只新增学生真实选择最多的 2-3 个专题。
- 做第一轮 19 元/7 天付费实验。
- 指标至少包括：诊断完成、time-to-value、推荐打开、复测完成、held-out 变化、7 日复用、坏题、成本/延迟。
- 关闭 10 条历史 pending 坏题，建立 48 小时处理 SLA。

### 90 天

- 目标 20-30 名学生、100 个可信 session、5-10 个真实付费者。
- 只有在数据量足够后才启用 efficacy 个性化和复习规划。
- 用最热视频做对照，证明推荐器优于“大家都看这个”。
- 有稳定导流与结果数据后，再和老师讨论合作，而不是先谈宏大合作。

### 1 年（条件式，不是承诺）

- 化学先形成 20-30 个高质量 gold diagnostic nodes 和稳定复用。
- 若留存/付费成立，再选择“扩上海化学覆盖”或“一门新学科/一个新省”之一，不能同时扩。
- 新省第一步永远是源格式和版权普查，不是直接复用整条化学补丁链。

## 10. Claude、Codex、GPT-5.6、DeepSeek、Fable 工作流

### 10.1 不再使用“一个永远的大脑 + 一个永远的手”

更好的分工是按任务所有权和风险切换：

| 任务 | Owner | 独立 reviewer | 禁止事项 |
|---|---|---|---|
| 产品方向、offer、范围、治理 | Opus/Claude | Sol 做工程反证 | 不由模型替用户签元决策 |
| 多文件实现、集成、调试 | GPT-5.6 Sol | Terra 或 fresh Opus/Sol | 作者不得自写自测自签 |
| 只读全库扫描、证据复算 | Terra | Sol 汇总 | 不写 official |
| 大批量机械候选 | DeepSeek/Luna | 确定性门 + Sol 抽查 | 不签 manual/serve/apply |
| 高代价战略挑战、超长视觉审计 | Fable 5 | Opus/Sol 对照 | 不作为日常常驻模型 |

### 10.2 按你给出的价格，推荐的成本策略

- GPT-5.6 只比 DeepSeek 贵约 20%：凡是跨文件、要理解状态、可能返工的任务，默认用 Sol。省 20% token 却多一次返工通常是假省钱。
- DeepSeek/Luna 用在可逆、可机器验收的 L0 批处理，例如 census、格式转换、候选生成。
- Opus 在你的渠道价格很低：保留为产品决策、需求追问和 brief reviewer，不需要每个普通 bug 都先经过 Opus。
- Fable 5 是 Opus 的 3-4 倍：只在重大转向、一次性超长材料或高价值对抗审计使用；每个里程碑最多一轮，不常驻。
- 生产运行时不要用 Opus/Fable/Sol 做所有题。确定性选题/判分免费，低置信自由回答才升级模型。
- 每个 paid job 必须记录模型、prompt version、token、实际费用、重试数和停止原因。

### 10.3 关于“5.6 更强”的证据边界

- 本机 model registry 把 Sol 描述为 frontier、Terra 为 balanced、Luna 为 fast/affordable，注册 context window 为 372K；先前桌面任务实际窗口约 353.4K。
- 官方 OpenAI 文档在本轮抓取返回 403；本地 fallback 仍只列到 5.5。因此本报告不把 5.6 的市场宣传、API 1M 上下文或价格当作已独立验证事实。
- 中转站的模型名也不应直接信任。要求保存 `model_returned`，并用同 brief、同验收集、盲评结果决定角色。

### 10.4 三模型公平试跑

用 6 个真实任务评估 Opus/Sol/Fable：

1. 找出一份 brief 的规格矛盾；
2. 设计一个跨模块闭环；
3. 修一个带隐藏测试的 bug；
4. 审一个数据 apply 候选；
5. 读浏览器结果找产品阻断；
6. 做一次成本/停止条件决策。

统一评分：漏问关键前提、越权、隐藏缺陷召回、一次通过率、返工、总成本、真实结果。不要用“感觉谁更聪明”选大脑。

## 11. 单机协作规则

一台电脑不适合多个写入 agent 同时操作同一脏工作树。推荐硬规则：

1. 同一时刻最多 1 个 writer + 1 个只读 reviewer。
2. writer 必须独立 Git worktree/branch；数据目录只读挂载或显式共享。
3. CPU/磁盘重任务同时最多 1 个；模型云推理可以并行，但本地管道不并行。
4. 每个服务用固定端口、PID、started_at、git SHA；任务结束即停，禁止十天陈旧进程。
5. 每个 paid pipeline 有 lock、预算、最多 2 次连续失败、单调 checkpoint；不轮询、不自动重启。
6. `/tmp` 只是 L0 候选区；重要候选必须有 hash manifest 和备份，不能让 71MB catalog 或 1.6GB ASR 只活在临时目录。

## 12. Handoff 和上下文工作流

不要再“等 100 万上下文用满才 handoff”。handoff 应在事件边界发生：

- 原子目标完成；
- 开始新阶段；
- 更换 owner/reviewer；
- 发生 compaction；
- 上下文约用到 60%-70%，且剩余工作是新的子问题。

每次 handoff 只需要：

1. 当前目标和非目标；
2. 用户已签决定与未决决定；
3. commit/branch/dirty state；
4. 原始证据文件和测试命令；
5. 已知失败与停止条件；
6. 下一步第一个可执行动作。

`YHER_PROJECT_OVERVIEW` 应只保留当前状态和短 ledger；历史继续轮转归档。聊天上下文不是数据库，summary 也不是证据原件。

### 推荐的新日常流程

```text
用户目标
 -> 1 页 decision card（只问不可从仓库发现的问题）
 -> active brief（success/non-goals/risk/stop/evidence）
 -> owner 端到端实现
 -> 独立 reviewer 读 diff 和原始证据
 -> 离线测试 + 真浏览器 + 必要的人验
 -> commit/PR
 -> overview ledger
```

普通任务只需要 owner + reviewer。只有 R3 高风险任务才需要 CEO/eng/design/security 多轮评审；不要每轮跑完整 autoplan。

## 13. 立即停止、继续、开始

### 停止

- 全量 ASR 原样恢复；
- 新增诊断/记忆架构；
- 题库长尾百分比冲刺；
- 数学/物理扩张；
- 复杂订阅档位、App、AI 班主任；
- 多模型重复审同一份材料但没有新证据；
- 默认 pytest 付费/联网。

### 继续

- R5 白名单和不可自签纪律；
- 真题关键位、style-transfer 高保真位、生成题低风险位的分级；
- candidate -> audit -> sign -> apply；
- 私持 gold、双向金标、浏览器真验；
- B 站目录，但先做最小签字子集。

### 开始

- 凭据轮换；
- 代码恢复点与 repo 收敛；
- 单专题真实闭环；
- 默认离线 CI；
- 3-5 名真实同学试用；
- 19 元/7 天付费实验；
- 版权、未成年人隐私和中转站数据边界清单。

## 14. 完整性检查结果

- 11,775 JSON 全部可解析。
- 613 JSONL、319,387 行全部可解析。
- 189 个第一方 Python 文件 AST 全部可解析。
- 9,554 个第一方 UTF-8 文本无编码失败。
- 808 个非链接 DOCX 容器与核心 XML 全部正常。
- 848 PDF 中 1 个物理目录 PDF 为 0 字节；化学 PDF 无容器失败。
- 52,609 PNG/JPEG 无损坏；3 张化学 JPG 为 12250x17275，超过 Pillow 默认防炸弹阈值但可验证，属于性能风险。
- 18,525 WMF：18,078 placeable WMF；447 标准 WMF 头，Pillow 无尺寸但 magic 合法。
- 7,261 SVG 和 18 XML 全部可解析。
- 679 音频中 3 个数学 M4A 缺 `moov`；化学音频未发现容器失败。
- 当前磁盘余量约 164 GiB；`/tmp/yher_*` 中有多个 1.6GB 副本，暂不删除。

## 15. 测试与审计副作用说明

- 根引擎：119/119。
- 主仓库离线限定套件：252/253；唯一稳定失败为 KaTeX 缺包降级。
- 默认全量 pytest 被主动中止：它会递归收集 backups、联网加载模型，并可能执行付费/远端写测试。
- 本轮直接运行 `test_study_v1.py` 和 pytest 各一次，测试中的 `study_next()` 未隔离日志目录，共向 `data/study_logs/propensity_20260710.jsonl` 追加 4 条 `session_id=""` 的审计污染行。未擅自删除，后续分析必须排除这 4 行。
- 为查官方 Codex/模型资料，按 `openai-docs` 规则注册了全局 `openaiDeveloperDocs` MCP；当前线程未热加载，需新任务/重启后才可能调用。

## 16. 最终 CEO 决策建议

继续做 YHer，但立刻改变成功定义：

> 下一阶段成功不再是“又完成一个引擎/批次/评审”，而是“一个真实学生在不同题上证明推荐有效，并愿意下一次继续用或付 19 元”。

数据地基已经足够支撑这个实验。现在继续往地基加材料，商业风险不会下降；把闭环接通、让学生使用，才会产生下一条真正改变路线的信息。
