FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目文件
COPY . .

# BGE 模型预下载（可选，减少首次启动时间）
# RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

EXPOSE 8501

# 默认运行 Streamlit Web 版
CMD ["streamlit", "run", "apps/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
