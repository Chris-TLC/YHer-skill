# YHer-skill：基于 RAG 的应试 AI 化学私教

> 模仿 B 站化学 UP 主"一化儿"（杰哥）的应试教学体系，构建以高考化学复习为目标的 RAG-based AI 私教。
>
> **状态**：阶段 8.6 已部署 ✅ | **下一步**：30 题盲测 + 研究性学习论文

---

## ✨ 这个项目在做什么

- 🎯 **高考化学私教**：从 B 站化学 UP 主"一化儿"的 525 个公开教学视频出发，用 RAG 技术构建一个会"模仿杰哥讲题套路"的 AI 助教
- 🧠 **杰哥风格**：五段式回答（开场诊断 → 思维招式 → 知识双轨 → 学习路径 → 互动鼓励）+ 八大思维招式 + 应试 meta 视角
- 🎓 **学段适配**：自动判断高一/高二/高三难度，从基础概念到压轴题对应不同讲法
- 📊 **私教级长期记忆**：完整记忆 always-on，季度高保真压缩（4000 tokens 档案），跨设备 Supabase 云端同步
- 🌐 **8 家 LLM 兼容**：DeepSeek、Kimi、通义、智谱、豆包、MiniMax、Anthropic、OpenAI

---

## 🎬 演示

```
你：2024年高考某题：N₂(g) + 3H₂(g) ⇌ 2NH₃(g)，K=0.5。
    向容器中再充入 N₂，平衡如何移动？K 值如何变化？

🎯 杰哥：
嗯，这题卡 K 和转化率了是吧。

先诊断下：能秒判出来 K 只跟温度有关、跟浓度无关吗？...

【一招讲透：K 和转化率的关系】

第一轨：知识层面（懂原理）
化学平衡常数 K，本质是"反应完成度的量尺"...

第二轨：考点层面（怎么考的，怎么破）
出题人挖的坑就一个：把"平衡正向移动"直接等价于"转化率升高"...

【学习路径】
- 【高考系统课】P56：化学平衡常数 K 的应用与三段式
  https://www.bilibili.com/video/BV1aComYMEms?p=56
  通关标准：随便给转化率数据，30 秒内列出三段式

试一道再反馈。

💰 本次：¥0.0100 | 输入：2264 (1280 缓存命中) | 输出：1481
```

---

## 🏗️ 技术架构

```
            ┌─── 525 BV / 1090 SRT (B 站公开字幕)
            ↓
[阶段 1-3] 数据采集 → ASR 错词纠正 (98.1%) → MinHash 去重 → Unicode 规范化
            ↓
[阶段 4]   LLM 标签化 (23,843 chunks)
            ↓
[阶段 5-6] 知识图谱 (65 节点) + 题型库 (13 父 + 19 子)
            ↓
[阶段 7]   BGE-M3 向量索引 + BM25 + RRF 融合 (FAISS)
            ↓
[阶段 8]   杰哥 system_prompt (五段式 + 八大招式)
            ↓
[阶段 8.5] 多 API 抽象层 + 季度压缩长期记忆 (Supabase)
            ↓
[阶段 8.6] 部署 polish (本仓库当前状态) ✅
            ↓
[阶段 9]   30 题盲测 + 论文 (进行中)
            ↓
[阶段 11]  iOS App (高考后)
```

**关键技术栈**：
- 检索：BGE-M3 (1024 维) + FAISS IndexFlatIP + BM25 三通道 + RRF 融合
- LLM：DeepSeek V4-Pro (1M context, 充分利用 13%)
- 数据库：Supabase (PostgreSQL with RLS)
- 框架：Python 3.10+, sentence-transformers, faiss-cpu, supabase-py

---

## 🚀 快速上手

### 前置依赖

- Python 3.10+
- 至少 8GB 内存（BGE-M3 模型加载需要）
- Supabase 账号（免费）
- DeepSeek API Key（或其他 7 家之一）

### 1. 克隆 + 安装

```bash
git clone https://github.com/YOUR-USERNAME/YHer-skill.git
cd YHer-skill

pip install -r requirements.txt
```

### 2. 下载 embeddings 数据

embeddings 索引文件（约 154 MB）托管在 HuggingFace Datasets，方便研究复现：

```bash
# 方式 A：huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download YOUR-HF-USERNAME/YHer-skill-embeddings \
    --local-dir ./data/embeddings --repo-type dataset

# 方式 B：手动下载（如果网络受限可用 hf-mirror.com）
# 详见 data/embeddings/README.md
```

### 3. 配置 .env

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

```bash
# .env 内容
DEEPSEEK_API_KEY=sk-xxxxxxxx
SUPABASE_URL=https://xxxxxxxx.supabase.co
SUPABASE_KEY=eyJxxxxxxxx
```

### 4. 初始化 Supabase

在 Supabase 后台 SQL Editor 跑 `deploy/init_supabase.sql`（仓库已附）。

### 5. 开跑

```bash
# 命令行版
python3 apps/chat.py

# Web 版（Streamlit）
streamlit run apps/app.py
```

---

## 📁 目录结构

```
YHer-skill/
├── README.md                  # 本文件
├── LICENSE                    # MIT
├── requirements.txt
├── config.example.yaml
├── .env.example
│
├── system_prompt.md           # 杰哥人设 + 五段式规则 (~3K 字符)
│
├── core/                      # 核心引擎
│   ├── retrieve.py            # RAG 检索（向量+BM25+RRF）
│   ├── diagnose.py            # 知识点诊断
│   └── format_answer.py       # 回答格式化器
│
├── adapters/                  # 抽象层
│   ├── llm_client.py          # 8 家 LLM 兼容
│   └── memory.py              # 长期记忆（Supabase）
│
├── apps/
│   ├── chat.py                # 命令行版
│   ├── app.py                 # Streamlit Web 版
│   └── api_server.py          # FastAPI（占位）
│
├── tests/                     # 测试
├── deploy/
│   ├── init_supabase.sql      # 表 + RLS policy
│   ├── Dockerfile
│   ├── install.sh
│   └── huggingface_spaces_guide.md
└── docs/                      # 设计文档
```

---

## 🎓 核心创新点（论文方向）

1. **教学者认知 vs 考纲分类的颗粒度选择**
   - 题型库做了 13 父 + 19 子的二级颗粒度，不是滥竽充数地缩减，而是细分后让 LLM 具体情况具体分析时更全面从容

2. **缓存友好的 prompt 结构（前缀稳定 → 后缀变化）**
   - USER_PROFILE 移到 system_prompt 末尾（每天更新 1 次），DeepSeek prompt cache 命中率从 0 升至 1280-1664 tokens 稳定区间

3. **LLM hallucination 的特征签名与防御**
   - 通过 `source_chunk_count` 字段检测知识图谱的"无数据虚构"现象
   - RAG + system_prompt 双层防御：知识边界外的问题会被自动拒绝并重定向到真实存在的内容

4. **多 API 统一抽象（OpenAI 兼容 + Anthropic 独立 SDK）**
   - 7 家用 OpenAI SDK，1 家（Anthropic）用独立 SDK
   - 模型降级检测（防 DeepSeek API 把 reasoner 路由到 chat 的静默降级）

5. **充分利用 1M context 的私教级长期记忆**
   - 完整记忆 always-on，24 个月堆积只用 1M context 的 13%
   - 季度压缩（90 天/4000 tokens 高保真档案）

---

## 💰 成本

- **开发期总成本**：¥219.24（数据处理 + 知识图谱构建）
- **运行期 24 月预估**：¥124（按每天 1.5 题 × ¥0.01）
- **prompt caching 节省**：120 倍（input miss ¥3.13/M vs hit ¥0.026/M）

---

## ⚖️ 版权与免责声明

### 关于源数据

本项目使用的训练数据来自 B 站 UP 主"一化儿"的**公开发布教学视频字幕**。
- 项目用途：**非营利的教育研究 + 个人学习工具 + 研究性学习论文**
- 不发行视频原内容，不替代原视频访问
- 推荐视频时附带原视频 URL，引导用户回到 B 站观看

如果一化儿（杰哥）本人或法定代理人对本项目有任何意见，请通过 Issues 联系，我会立即响应。

### 关于代码 License

代码部分采用 **MIT License**（见 LICENSE 文件）。

但请注意：
- ✅ 你可以自由 fork、修改、商用代码框架
- ❌ 不要直接复用本项目里的 system_prompt.md 或杰哥风格 prompt 用于商业产品（这是一化儿教学体系的衍生表达）
- ✅ 你可以借鉴架构，自己构建针对其他教育内容创作者的类似工具

### 关于 AI 生成内容

本项目的 AI 回答由 LLM 生成，可能包含错误。**不能替代专业教师**。请把它当作"复习辅助工具"而非"权威答案"。高考前的备考请以学校教师指导为准。

---

## 🤝 贡献

欢迎 Issues 和 PR：
- 数据增强：发现 ASR 识别错的化学术语
- prompt 工程：改进杰哥语调或思维招式
- 多模型适配：测试新的 LLM 提供商
- 部署文档：iOS App、HuggingFace Spaces、Docker

---

## 📚 致谢

- **一化儿（杰哥）**：高质量的化学应试教学体系是本项目的源泉
- **DeepSeek 团队**：V4-Pro 1M context + prompt caching 让长期记忆经济可行
- **BGE 团队（北京智源）**：BGE-M3 中文 embedding
- **Anthropic**：Claude（架构设计、prompt 工程指导）
- **Supabase**：免费 PostgreSQL + Auth 生态

---

## 👤 作者

- **Chris**（高二学生 / 化学教育 AI 爱好者）
- 联系：通过 GitHub Issues

---

## 📜 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{yher2026,
  author = {Chris},
  title = {YHer-skill: A RAG-based AI Chemistry Tutor Mimicking a Real Educator's Teaching Methodology},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/YOUR-USERNAME/YHer-skill}}
}
```

研究性学习论文（中文版）：完成后链接添加到此处。

---

**本仓库于 2026-05-06 首次发布。** 阶段 1-8.6 完成，阶段 9（评估 + 论文）进行中。
