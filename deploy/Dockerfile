FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 最小 demo 依赖（服务器判分/主链路,不含重向量检索）
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# 项目文件与数据
COPY . .

# 默认:证据闭环 demo(8700)。零付费确定性模式:
#   docker run -p 8700:8700 -e YHER_ENABLE_PAID_LLM=0 yher-demo
EXPOSE 8700
CMD ["bash", "deploy/run_demo.sh"]
