# YHer Chemistry:Evidence-Bound Diagnostic Learning System

**信息平权:让每个学生都能得到"看得见依据"的诊断与辅导。**
Educational information equity: every student deserves a diagnosis whose evidence they can inspect.

YHer 是一套面向上海高中化学的**证据约束学习闭环**——用真题做动态诊断,按本次作答证据提供讲解与资源推荐,再用未见题族做独立验证,并把结果写回可重放的学生画像。整条链路的创新锚点是**诊断引擎**(belief 推断 + 信息量选题 + 独立验证),不是任何单一内容来源。

当前状态是 **pre-alpha 单机 Demo**:不是已上线产品,没有真实学生留存、付费或学习效果证据,也不应直接部署给未成年人使用。

> 本仓库开源计划:代码 MIT;题库/知识图谱/转写表随仓库公开,另有 [Hugging Face 数据镜像](https://huggingface.co/datasets/ChrisTLC/yher-chemistry-question-bank)(Dataset Card 及字段口径)。

---

## 为什么存在(定位)

高考地区的教育信息分布不均:好老师、好方法、好题目,往往集中在少数资源充足的学校,而绝大多数学生只能依赖自己的摸索。YHer 的目标不是"再造一个老师",而是:

1. **把最值钱的判断做成可验证的工程**——学生到底卡在哪个知识点、是该补前置还是该练方法,用证据计算出来,而不是碰运气;
2. **把"谁讲得好"从经验变成检索问题**——把经过质量核验的公开教学内容按学生状态排序,而不是让学生在一个个视频里自己碰;
3. **把答案的可信度做成纪律**——所有下发给学生的标准答案都来自已核验的官方解析与标准解,AI 只能组织语言,不能创造化学事实。

这三点合起来是"教育信息平权"的一种工程实现:不是把所有信息免费堆给学生,而是**让每个学生都能得到信息筛选与诊断判断的平等机会**。

---

## 一次 canonical session(完整闭环)

1. 从 R5 白名单和可信映射中冻结诊断、练习、held-out **三个互不重叠的题族集合**;
2. **服务端判分**;提交前不向浏览器下发答案、rubric、item ID 或 family ID(fail-closed);
3. 用**四状态 belief(已掌握/前置缺失/思路不稳/未掌握)**与**期望信息增益(EIG)**递进选题,必要时下探前置知识;
4. 诊断结束后进入显式 learning checkpoint:展示与真实作答、已验证标准解绑定的讲解;
5. **推荐已签字视频资源**(证据签字的轨道),记录推荐、观看代理与已见视频段;
6. 用两个**未见题族**做 held-out 独立验证,生成只针对本 session 的报告、FSRS 稳定度与 7 日复习提示。

首页默认推荐"氧化还原反应";当前只有满足"至少 5 个独立、可确定判分、答案验证通过的题族"的节点才开放。

## 快速开始

### 方式一:Docker(推荐)

```bash
docker build -t yher-demo .
docker run -p 8700:8700 yher-demo            # 默认:有凭据走付费 LLM 讲解
docker run -p 8700:8700 -e YHER_ENABLE_PAID_LLM=0 yher-demo   # 零付费确定性模式
```

打开 [http://127.0.0.1:8700](http://127.0.0.1:8700)。

### 方式二:本地自举(要求 Python 3.11+)

```bash
cd yihuier-chemistry-skill
YHER_ENABLE_PAID_LLM=0 ./deploy/run_demo.sh   # 自举 .venv-demo,监听 127.0.0.1:8700
```

健康检查:

```bash
curl -fsS http://127.0.0.1:8700/health | python3 -m json.tool
```

凭据可选项:`DEEPSEEK_API_KEY`(默认)或其他 provider key 配置在 `.env`。缺少凭据、显式关闭付费通道、超时或输出不合格时,系统**保留确定性主链并诚实降级**。不要把密钥写入代码、日志、截图或报告。

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

五个引擎的职责边界:

| 引擎 | 职责 |
|---|---|
| `engine/mastery.py` | M/P/C/U 四状态 belief、证据更新、FSRS 衰减 |
| `engine/selector.py` | EIG 选题、前置竞争、收敛判据、已见题排除 |
| `engine/planner.py` | 30/60/120/180 分钟预算与诚实耗尽策略 |
| `engine/recommender.py` | 签字轨道、预算、已见段与 propensity 快照(向量检索/重排层) |
| `engine/memory.py` | 高价值事件与受限召回(画像只进表达类触点) |

**诊断是本项目的核心创新,推荐器只是它的下游**:视频资源层本质是"内容质量核验 + 向量检索 + 状态适配重排",其价值由诊断质量决定,不作为本项目独立主张。

浏览器不直接信任模型输出:公开讲解中的化学步骤只投影自 `answer_verification=passed` 的服务端标准解;LLM 只能有限选择与组织,不能创造新的标准答案事实。

## 数据与数据集

- 全量数据(3,329 条结构化题目、1,202 条 R5 服务白名单、6,005 条图形转写、135 节点知识图谱)随仓库发布,字段口径见 [`data/README.md`](data/README.md);
- 55 题可读样例:`data/samples/`(确定性抽取,可复现);
- **Hugging Face 数据集镜像**:`ChrisTLC/yher-chemistry-question-bank`(分片:题目 `items_v4` / 图谱 `knowledge_graph`,含 Dataset Card 与构建脚本 `scripts/make_hf_dataset.py`)。

## 测试与验证

```bash
python3 -m venv .venv-pub
.venv-pub/bin/pip install -r requirements-dev.txt   # 完整测试依赖(faiss 可选)
.venv-pub/bin/python -m pytest -q
```

- 仓库离线套件与引擎契约测试全绿基线见 CI/本地运行;
- QA 证据与合成场景(`SYNTHETIC_DEMO` 标识)是工程验证,**不是学生证据**。

## 诚实边界

- 这是 localhost pre-alpha:无登录、租户隔离、监护人同意、删除策略或生产运维;
- 现有浏览器旅程 / API QA / 合成场景都是**工程验证,不是实证研究**;
- belief 表示当前证据下的模型状态,不等于分数、长期掌握或因果学习增益;
- R5 是服务白名单,不等于全量人工 gold;关键判定位只使用真题与可信标准解;
- AI 生成题验证是历史供给实验;canonical 首测与 held-out 不使用生成题;
- 视频资源由原平台托管,链接可用性、版权与内容变更不受本仓库控制;
- 公网部署、凭据轮换、发布与未成年人数据流程均未完成。

## 进一步阅读

- [白皮书(英文,发布版)](docs/writeup/WHITEPAPER.md)
- [审计档案](docs/audit-history/README.md)(三轮系统级审计,含架构终裁)
- [技术报告 v2 初稿](docs/YHer_技术报告_v2_draft.md)
- [两分钟 Demo 分镜](docs/demo_walkthrough_script.md)
- [数据说明](data/README.md)

代码采用 [MIT License](LICENSE)。题目、试卷、字幕与外部视频各自保留原权利归属;MIT 许可不自动覆盖内容资产。
