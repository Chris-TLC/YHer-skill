"""
上传 embeddings 到 ModelScope dataset。
注意：SDK Token 通过环境变量传入，不要硬编码到脚本里。

用法：
    export MS_TOKEN="ms-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    python3 scripts/upload_to_modelscope.py
"""
import os
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi


# ─── 配置 ───
USERNAME = "ChrisTLC"
DATASET_NAME = "YHer-skill-embeddings"
LOCAL_EMBEDDINGS_PATH = "/Users/mac/Desktop/Tools/embeddings"
COMMIT_MESSAGE = "Initial upload: FAISS + BM25 indices (~154MB) for YHer-skill RAG"


def main():
    # 检查 token
    token = os.environ.get("MS_TOKEN")
    if not token:
        print("❌ 错误：环境变量 MS_TOKEN 未设置")
        print("   请先跑：export MS_TOKEN=\"你的SDK Token\"")
        sys.exit(1)

    # 检查本地目录
    embeddings_path = Path(LOCAL_EMBEDDINGS_PATH)
    if not embeddings_path.is_dir():
        print(f"❌ 错误：embeddings 目录不存在：{LOCAL_EMBEDDINGS_PATH}")
        sys.exit(1)

    # 估算总大小
    total_bytes = sum(
        f.stat().st_size for f in embeddings_path.rglob("*") if f.is_file()
    )
    total_mb = total_bytes / 1024 / 1024
    print(f"📦 待上传：{LOCAL_EMBEDDINGS_PATH}")
    print(f"   总大小：{total_mb:.1f} MB")

    # 登录
    api = HubApi()
    api.login(token)
    print("✅ 登录 ModelScope 成功")

    # 上传
    repo_id = f"{USERNAME}/{DATASET_NAME}"
    print(f"📤 开始上传 → {repo_id}")
    print("   预计耗时 5-15 分钟，请耐心等待...")

    start = time.time()
    api.upload_folder(
        repo_id=repo_id,
        folder_path=LOCAL_EMBEDDINGS_PATH,
        commit_message=COMMIT_MESSAGE,
        repo_type="dataset",
    )
    elapsed = time.time() - start

    print(f"✅ 上传完成（耗时 {elapsed:.1f} 秒）")
    print(f"🌐 访问：https://www.modelscope.cn/datasets/{USERNAME}/{DATASET_NAME}/files")


if __name__ == "__main__":
    main()
