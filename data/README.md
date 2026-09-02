# Data Assets

本目录包含 YHer 化学题库与知识图谱的**公开数据**。全量数据以 JSONL 形式随仓库发布;另有 55 题可读样例(`samples/`)与 Hugging Face 数据集镜像(见下)。

## 文件清单

| 文件 | 规模 | 说明 |
|---|---|---|
| `item_bank/v4/chemistry_v4_3329.jsonl` | 20 MB / 3,329 题 | WS3 结构化题目(块级 schema v4),唯一服务来源 |
| `item_bank/v4/chemistry_v4_1_3329.jsonl` | 20 MB / 3,329 题 | v4.1 修正版(答案判定口径修正,Batch7 apply) |
| `item_bank/v4/usability_r5_v1.jsonl` | 660 KB / 2,526 行 | **R5 可用性白名单**:逐题服务许可(1202 题可服务) |
| `item_bank/v4/service_exclusions.jsonl` | 4 KB | 源头不可修的永久排除清单 |
| `item_bank/v4/ws2_asset_transcripts_v1.jsonl` | 8.8 MB / 6,005 行 | 图形资产转写表(公式→LaTeX/插图→结构化描述) |
| `item_bank/v4/ws2_media_ref_map_v1.jsonl` | 3.0 MB / 12,790 行 | 题目中媒体引用 → 资产 hash 映射 |
| `item_bank/v4/ws2_omml_latex_cache_v1.jsonl` | 196 KB / 1,518 条 | OMML→LaTeX 预转缓存(katex_ok 96.4%) |
| `knowledge_graph_150.jsonl` + `_enriched.jsonl` | 360 KB + 680 KB | 135 节点化学知识图谱(含前置/题型/考点/视频推荐) |
| `raw_papers/shanghai_all.jsonl` | 5.9 MB | 原始切题输出(6083 题,含噪声未清洗,仅供溯源) |
| `samples/` | 55 题 | R5 白名单抽样可读样例(含 schema 说明) |

## Schema(v4,块级)

每行一个题目对象:

```
{
  "item_id":        "sha1 派生,下游唯一键",
  "group_key":      试卷标识(如 "2023年高考化学上海卷"),
  "section_num":    试卷内大题号,
  "q_num":          试卷内题号,
  "local_question_id": "组内局部计数",
  "source_path":    源 Word/PDF 相对路径,
  "answer_source_path": 答案解析源(可与题面源不同),
  "schema_version": "ws3_schema_v4_candidate_1",
  "service_eligible": true/false,
  "answer_available": true/false,
  "analysis_blocks": [{"para":[{"type":"text","text":"解析..."}]}],
  "answer_blocks_effective": [{"para":[{"type":"text","text":"【答案】A"}]}],
  "quality_flags": [],
  "rubric": [{"point_id","desc","keywords","must_have","score","kg_node"}],
  "alignment": {...},        # 与 v3 题的对齐信息(继承)
  "answer_verification": {...},  # 答案可信度(0.89 等,来源 v3.4 管线)
  "kg_nodes": [...],         # 关联知识图谱节点
  "knowledge_points": [...]
}
```

**块类型**(`para[].type`):`text` / `latex` / `omml`(WMF 图,latex 在 ws2_omml_latex_cache 中) / `image`(asset hash) / `table`。

## 服务池(R5)规则

```
load_service_pool() = item_bank_v4.loader 默认 apply_r5=True
  → 只 serve usability_r5_v1.jsonl 中 r5_serve=true 的行
  → 台账无记录 = 不服务
审计/预览/回归通道显式 apply_r5=False 才看全池 2526
```

当前 **r5_serve=true = 1202 题**。

## 版权与许可

- 题目来源于公开的高考/模拟试卷(上海卷),为**试卷内容的机械性结构化**(文本抽取、排版修复、答案对齐),非原创创作;
- 本仓库代码与数据处理脚本:MIT;题库数据:随仓库公开,**知识图谱/转写表/对齐信息为项目自建资产**;
- 在线数据镜像:Hugging Face `ChrisTLC/yher-chemistry-question-bank`(含 Dataset Card)。

## 构建

```bash
# 样例(55 题)
python3 scripts/make_hf_dataset.py --sample-only

# HF 数据集构建(需要 HF token)
python3 scripts/make_hf_dataset.py --push  <dataset-id>
```
