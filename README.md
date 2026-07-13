# YHer Chemistry Demo

YHer 是一个面向上海高中化学的证据约束学习闭环：用真题做动态诊断，按本次作答证据提供讲解和视频资源，再用未见题族独立验证，并把结果写回可重放的学生画像。

当前状态是 **pre-alpha 单机 Demo**。它不是已上线产品，没有真实学生留存、付费或学习效果证据，也不应直接部署给未成年人使用。

## 当前能做什么

一次 canonical session 由同一套服务端状态机完成：

1. 从当前 R5 白名单和可信映射中冻结诊断、练习、held-out 三个互不重叠的题族集合。
2. 服务端判分；提交前不向浏览器下发答案、rubric、item ID 或 family ID。
3. 用四状态 belief 和 EIG 递进选题，必要时下探前置知识。
4. 诊断结束后进入显式 learning checkpoint，展示与真实作答和已验证标准解绑定的讲解。
5. 推荐有证据签字的视频资源；记录推荐、观看代理和已见视频段。
6. 用两个未见题族做 held-out 验证，生成只针对本 session 的报告、FSRS 稳定度和 7 日复习提示。

首页默认推荐“氧化还原反应”。当前只有满足“至少 5 个独立、可确定判分、答案验证通过的题族”的节点才开放。

## 本地运行

要求 Python 3.11 或更新版本。启动脚本会在仓库内自举 `.venv-demo`，安装 [requirements-demo.txt](requirements-demo.txt) 中固定版本的最小依赖，并以单 worker 监听本机回环地址。

`run_demo.sh` 默认允许已配置的付费 LLM 通道。要做零 provider LLM 调用、零付费的确定性复现，必须显式关闭：

```bash
cd yihuier-chemistry-skill
YHER_ENABLE_PAID_LLM=0 ./deploy/run_demo.sh
```

首次启动若缺少依赖，脚本可能通过 `pip` 联网安装；只有已准备好 `.venv-demo` 或本地依赖缓存时，上述运行流程才同时满足外网隔离。

打开 [http://127.0.0.1:8700](http://127.0.0.1:8700)。健康检查：

```bash
curl -fsS http://127.0.0.1:8700/health | python3 -m json.tool
```

默认启动在存在 DeepSeek 凭据时，讲解和少量自由作答可使用 provider；凭据缺失、显式设置 `YHER_ENABLE_PAID_LLM=0`、超时或输出不合格时，系统保留确定性主链并诚实降级。不要把密钥写入代码、日志、截图或报告。

## 架构

```text
apps/web/index.html + app.js
            |
            | same-origin /api/demo/*
            v
apps/demo_api.py (FastAPI, localhost, one worker)
            |
            v
core/learning/session_service.py
   |          |          |          |
   |          |          |          +-- curriculum.py -> signed video map
   |          |          +------------- explanations.py / grading.py
   |          +------------------------ ItemCatalog -> v4 + R5
   +----------------------------------- mastery / selector / planner
            |
            v
adapters/store/local_json.py
  append-only events + session snapshots + projected profile
```

五个引擎的职责边界：

- `engine/mastery.py`：M/P/C/U 四状态 belief、证据更新和 FSRS 衰减。
- `engine/selector.py`：EIG 选题、前置竞争、收敛和已见题排除。
- `engine/planner.py`：30/60/120/180 分钟预算与诚实耗尽策略。
- `engine/recommender.py`：签字轨道、预算、已见段和 propensity 快照。
- `engine/memory.py`：可进入画像的高价值事件和受限召回。

浏览器不直接信任模型输出。公开讲解中的化学步骤只投影自 `answer_verification=passed` 的服务端标准解；LLM 可以做有限的选择和组织，不能创造新的标准答案事实。

## 2026-07-13 数据快照

以下数字来自已审计实现 SHA `d62aad4` 的 `/health`、正式数据文件和签字配置。它们描述的是本机快照，不是用户规模；后续纯文档提交不改变这组运行时事实。

| 资产/门 | 当前事实 | 权威路径 |
|---|---:|---|
| v4 主库 | 3,329 行 | `data/item_bank/v4/chemistry_v4_1_3329.jsonl` |
| R5 决策台账 | 2,526 行 | `data/item_bank/v4/usability_r5_v1.jsonl` |
| `r5_serve=true` | 1,202 | `/health` + R5 台账 |
| canonical trusted | 973 | `/health` |
| canonical rejected | 229 | `/health` |
| 独立内容题族 | 963 | `/health` |
| 可确定判分题 | 400 | M6 catalog audit |
| 当前开放节点 | 27 | `/api/demo/nodes` |
| 知识图谱 | 135 行 | `data/knowledge_graph_150_enriched.jsonl` |
| 历史 v3 题库 | 6,438 行，不进入 canonical 学生题源 | `data/item_bank/chemistry_v3_6695.jsonl` |

答案信任门在 M6 将 9 条 `needs_review` 的“看似可判”记录改为 fail-closed，因此开放节点从先前快照的 28 降为 27；“化学反应速率”当前低于五题族开放门。这是正确的能力收缩，不用旧数字掩盖。

## 视频资源

频道目录包含 551 个 BV、1,136 个分 P，其中 842 个 `(bv, p)` 与现有语料相交。43 个轨道实体经过证据核对：

- 30 个实体签字启用；
- 13 个因标题证据不足保持 neutral；
- runtime 含 148 个路由键和 345 个候选段，这里的 148 不是新增 KG 节点；
- 只有 8 个有机段带真实时间锚，覆盖 6 个路由键；其他资源只给视频级链接，不编造时间戳。

配置和可审计证据在 `config/curriculum/track_map_v1.yaml` 与 `core/learning/assets/curriculum_runtime_v1.json`。

## 测试与 QA

本次收官快照的 fresh 证据：

- 仓库离线套件：569/569；
- 根五引擎契约：119/119；
- 自动化桌面 1280×800 与手机 390×844 浏览器旅程：12/12，`failures=[]`；合同另要求的 final `computer-use` 人工矩阵仍在收口，不能用自动化数量替代；
- API fresh run `20260712T224150Z`（服务 SHA `ce0700f`）中，确定性接口 p95 为 6.704 ms，低于 500 ms 门；LLM 全文最大 16.497 s，低于 20 s 门；
- 提交前响应泄漏扫描未发现答案、模型名或凭据字段；
- 同一 session 的诊断/练习/held-out item 与 family 均不相交。

这里的 `failures=[]` 只表示对应自动化浏览器机械门通过。后续实现已关闭首题计数、前置标题、100 字残句和错误层级不驱动支架四个问题；post-fix 真实 UI 已核到完整零基础起点、难度支架、两错额外支架，以及 strong 3/3 后完成 practice + 2 held-out 的 verified 旅程。公开讲解仍严格定位为 DeepSeek 辅助的 verified standard-solution authoritative projection，不宣称模型可自由生成可靠化学事实。

创建同一套 runtime/dev 虚拟环境后可运行：

```bash
python3 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements-demo.txt -r requirements-dev.txt
.venv-demo/bin/python -m pytest tests -q
.venv-demo/bin/python -m pytest tests/test_synthetic_scenarios.py -q
```

完整 QA 证据保存在本机交付目录 `/tmp/yher_demo_overnight/`，该目录不是仓库运行依赖。

## SYNTHETIC_DEMO

`demo/synthetic_scenarios/` 包含 24 个明确标注的合成学生场景，形成 32 个 episode。它们在 30/60/120 分钟与四种结局上做平衡覆盖，只用于重放、回归和录屏准备。

场景计划覆盖 28 个专题，但当前可实际启动的只有上述 27 个；“化学反应速率”被重放器验证为预期关闭。合成重放使用独立 `/tmp` store、零网络、零付费调用，绝不与真实或 QA 事件混写。

```bash
.venv-demo/bin/python -m demo.synthetic_scenarios.validate
.venv-demo/bin/python -m demo.synthetic_scenarios.replay \
  --output /tmp/yher_synthetic_demo_replays/readme_run
```

任何带 `SYNTHETIC_DEMO` 标识的会话都不是学生证据。

## 诚实边界

- 这是 localhost pre-alpha，没有登录、租户隔离、监护人同意、删除策略或生产运维。
- 现有 12 条浏览器旅程、API QA 和 24 个合成场景都是工程验证，不是实证研究。
- belief 表示当前证据下的模型状态，不等于分数、长期掌握或因果学习增益。
- R5 是服务白名单，不等于全量人工 gold；关键判定位只使用真题和可信标准解。
- AI 生成题验证是历史供给实验；canonical 首测和 held-out 不使用生成题。
- 外部视频仍由原平台托管，链接可用性、版权和内容变更不受本仓库控制。
- 公网部署、凭据轮换、发布和未成年人数据流程均未完成。

## 进一步阅读

- [技术报告 v2 初稿](docs/YHer_技术报告_v2_draft.md)
- [两分钟 Demo 分镜](docs/demo_walkthrough_script.md)
- `PROJECT_HANDOFF/` 中的数据治理、视觉管线和内化验证审计属于工作区项目记录，不是本仓库 README 的上线声明。

代码采用 [MIT License](LICENSE)。题目、试卷、字幕和外部视频各自保留原权利归属；MIT 许可不自动覆盖这些内容资产。
