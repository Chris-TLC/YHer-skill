"""
Upload embeddings to the ModelScope dataset.
Note: the SDK token is passed via an environment variable; never hardcode it in the script.

Usage:
    export MS_TOKEN="ms-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    python3 scripts/upload_to_modelscope.py
"""
import os
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi


# --- Configuration ---
USERNAME = "ChrisTLC"
DATASET_NAME = "YHer-skill-embeddings"
LOCAL_EMBEDDINGS_PATH = str(Path(__file__).resolve().parents[1] / "embeddings")
COMMIT_MESSAGE = "Initial upload: FAISS + BM25 indices (~154MB) for YHer-skill RAG"


def main():
    # Check token
    token = os.environ.get("MS_TOKEN")
    if not token:
        print("ERROR: environment variable MS_TOKEN is not set")
        print("   Run first: export MS_TOKEN=\"your SDK Token\"")
        sys.exit(1)

    # Check local directory
    embeddings_path = Path(LOCAL_EMBEDDINGS_PATH)
    if not embeddings_path.is_dir():
        print(f"ERROR: embeddings directory does not exist: {LOCAL_EMBEDDINGS_PATH}")
        sys.exit(1)

    # Estimate total size
    total_bytes = sum(
        f.stat().st_size for f in embeddings_path.rglob("*") if f.is_file()
    )
    total_mb = total_bytes / 1024 / 1024
    print(f"Ready to upload: {LOCAL_EMBEDDINGS_PATH}")
    print(f"   Total size: {total_mb:.1f} MB")

    # Log in
    api = HubApi()
    api.login(token)
    print("Logged in to ModelScope successfully")

    # Upload
    repo_id = f"{USERNAME}/{DATASET_NAME}"
    print(f"Uploading -> {repo_id}")
    print("   Estimated 5-15 minutes; please wait...")

    start = time.time()
    api.upload_folder(
        repo_id=repo_id,
        folder_path=LOCAL_EMBEDDINGS_PATH,
        commit_message=COMMIT_MESSAGE,
        repo_type="dataset",
    )
    elapsed = time.time() - start

    print(f"Upload complete (took {elapsed:.1f} seconds)")
    print(f"Visit: https://www.modelscope.cn/datasets/{USERNAME}/{DATASET_NAME}/files")


if __name__ == "__main__":
    main()
