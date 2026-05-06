# HuggingFace Spaces 部署指南

## Step 1: 注册

https://huggingface.co/join

## Step 2: 创建 Space

1. 头像 → New Space
2. Space name: `yihuier-chemistry-assistant`
3. License: MIT
4. SDK: **Streamlit**
5. Hardware: **CPU basic（免费）**
6. Visibility: Public

## Step 3: 上传文件

```bash
git clone https://huggingface.co/spaces/<username>/yihuier-chemistry-assistant
cd yihuier-chemistry-assistant

# 复制项目（embeddings 153MB，HF 单文件 < 5GB OK）
cp -r ~/Desktop/Tools/yihuier-chemistry-skill/* .

git add .
git commit -m "Initial deploy v3"
git push
```

## Step 4: 配置 Secrets

Settings → Repository secrets：
- `SUPABASE_URL`
- `SUPABASE_KEY`

（可选）
- `DEEPSEEK_API_KEY`

## Step 5: 验证

5-10 分钟后访问：
`https://huggingface.co/spaces/<username>/yihuier-chemistry-assistant`
