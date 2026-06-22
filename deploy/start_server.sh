#!/bin/bash
# YHer 化学私教 - 服务器一键启动脚本
# 用法: bash deploy/start_server.sh
# 同时启动 FastAPI 后端(8600) + Streamlit 前端(8502)

set -e
cd "$(dirname "$0")/.."

echo "===== YHer 化学私教启动中 ====="

# 检查 .env
if [ ! -f .env ]; then
    echo "⚠️  缺少 .env，请先 cp .env.example .env 并填入 DEEPSEEK_API_KEY"
    exit 1
fi

# 启动 FastAPI 后端（后台）
echo "启动后端 API (端口 8600)..."
nohup python3 -m uvicorn apps.api_server:app --host 0.0.0.0 --port 8600 > logs_api.log 2>&1 &
echo $! > .api.pid
sleep 3

# 启动 Streamlit 前端（前台，监听所有网卡让外网可访问）
echo "启动前端 Demo (端口 8502)..."
streamlit run apps/stage1_demo.py \
    --server.address 0.0.0.0 \
    --server.port 8502 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false

# 退出时清理后端
trap "kill \$(cat .api.pid) 2>/dev/null; rm -f .api.pid" EXIT
