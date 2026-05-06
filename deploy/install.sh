#!/bin/bash
# 一化儿 AI 化学助手 - 一键安装脚本 v3

set -e

echo "🧪 一化儿 AI 化学助手 v3 安装"
echo "================================"

# 检查 Python
python3 --version || {
    echo "❌ 需要 Python 3.10+"
    exit 1
}

# 装依赖
echo ""
echo "📦 安装 Python 依赖..."
pip3 install -r requirements.txt --break-system-packages 2>/dev/null || \
pip3 install -r requirements.txt

# 检查 BGE 模型
echo ""
echo "🔍 检查 BGE-M3 embedding 模型..."
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-m3', local_files_only=False)
print('✅ BGE-M3 模型就绪')
" 2>/dev/null || {
    echo "⚠️  BGE-M3 模型首次下载中（约 2GB），请稍候..."
    python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-m3')
print('✅ BGE-M3 模型下载完成')
"
}

# 复制配置
if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml
    echo "✅ 已创建 config.yaml（请编辑 API provider 和 model）"
else
    echo "✅ config.yaml 已存在"
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ 已创建 .env（请填入你的 API Keys）"
else
    echo "✅ .env 已存在"
fi

echo ""
echo "================================"
echo "✅ 安装完成！"
echo ""
echo "快速开始："
echo "  1. 编辑 .env 填入你的 API Key"
echo "  2. 编辑 config.yaml 选择 provider"
echo "  3. 运行: python apps/chat.py"
echo ""
echo "  或: streamlit run apps/app.py"
echo "================================"
