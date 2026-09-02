# Testing / 测试说明

## 主套件(公开仓,CI 用)

```bash
python3 -m venv .venv-pub
.venv-pub/bin/pip install -r requirements-dev.txt
.venv-pub/bin/python -m pytest            # pytest.ini 默认标记/配置
```

- Green baseline:**734+ 通过**(随运行机器与网络环境浮动 1-3);
- 无外部付费调用(主套件默认 `-m "not paid"`,跳过带 `paid` 标记的联网/付费测试);
- faiss 无 py3.14 wheel:retrieve 模块已将 faiss 设为可选(`FAISS_AVAILABLE` 降级),向量检索相关测试在无 faiss 环境自动跳过;
- 若运行环境为 Python 3.11/3.12,faiss-cpu 可正常安装,测试覆盖更全。

## 论文产物类测试(公开仓默认排除)

19 个 `paper` 类测试(`test_analysis_paper` / `test_paper_pdf` / `test_paper_references` 等)依赖内部论文产物(`docs/paper` 与 /tmp 生成器),为 2026-07 论文科研线的原始验证套件。公开仓默认排除;相关产物归档在项目工作区 `_papers_archive/`。该类测试文件保留以维持可追溯性。

## 引擎契约(不依赖大数据)

```bash
.venv-pub/bin/python -m pytest tests/test_mastery.py tests/test_selector.py \
  tests/test_planner.py tests/test_recommender.py tests/test_memory.py \
  tests/test_event_log.py -q
```

## 全量含论文类(需要内部产物,公开仓不可复现)

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pub/bin/python -B -m pytest -q --timeout=120 \
  --override-ini="addopts="
```
