"""
更新 GitHub README 里的占位符链接：
- YOUR-GITHUB-USERNAME → Chris-TLC
- YOUR-HF-USERNAME → ChrisTLC（同时把下载方式从 HuggingFace 改为 ModelScope）

用法：python3 scripts/update_readme_links.py
"""
from pathlib import Path

GITHUB_USERNAME = "Chris-TLC"
MS_USERNAME = "ChrisTLC"

readme = Path('README.md')
content = readme.read_text(encoding='utf-8')

# 替换所有 GitHub 用户名占位符
content = content.replace('YOUR-GITHUB-USERNAME', GITHUB_USERNAME)
content = content.replace('YOUR-USERNAME', GITHUB_USERNAME)

# 把 HuggingFace 描述文字替换
content = content.replace(
    "embeddings 索引文件（约 154 MB）托管在 HuggingFace Datasets，方便研究复现：",
    "embeddings 索引文件（约 154 MB）托管在 ModelScope（国内访问最快）："
)

# 替换 HuggingFace 下载块为 ModelScope
old_hf_block = """# 方式 A：huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download YOUR-HF-USERNAME/YHer-skill-embeddings \\
    --local-dir ./data/embeddings --repo-type dataset

# 方式 B：手动下载（如果网络受限可用 hf-mirror.com）
# 详见 data/embeddings/README.md"""

new_block = f"""# 方式 A：ModelScope CLI（推荐，国内访问最快）
pip install modelscope
modelscope download --dataset {MS_USERNAME}/YHer-skill-embeddings --local_dir ./data/embeddings

# 方式 B：使用 modelscope SDK
python3 -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('{MS_USERNAME}/YHer-skill-embeddings', repo_type='dataset', cache_dir='./data/embeddings')"

# Dataset 主页：
# https://www.modelscope.cn/datasets/{MS_USERNAME}/YHer-skill-embeddings"""

content = content.replace(old_hf_block, new_block)

# 验证有没有残留 placeholder
remaining = []
for ph in ['YOUR-GITHUB-USERNAME', 'YOUR-HF-USERNAME', 'YOUR-USERNAME']:
    if ph in content:
        remaining.append(ph)

readme.write_text(content, encoding='utf-8')

if remaining:
    print(f"⚠️ 还有 placeholder 没替换：{remaining}")
    raise SystemExit(1)
else:
    print("✅ README placeholder 全部替换完成")

print(f"   文件大小：{len(content)} 字符")
print(f"   ModelScope 用户名：{MS_USERNAME}")
print(f"   GitHub 用户名：{GITHUB_USERNAME}")
