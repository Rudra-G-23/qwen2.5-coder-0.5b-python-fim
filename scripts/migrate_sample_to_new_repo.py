#!/usr/bin/env python
"""
scripts/migrate_sample_to_new_repo.py
One-time migration: copies Stage 1's already-collected 10k-file curated
sample from the old flat repo (Rudra-G-23/qwen-coder-python-fim-data) into
the new structured repo (Rudra-G-23/the-stack-v3-python-fim-data), under
sample_filtered_data_10000/{data,checkpoints}/ (data-stage-2.md §1-2).

Idempotent — safe to re-run: skips any file that already exists at its
destination path in the new repo. Creates the new repo if it doesn't exist
yet. Ends by writing sample_filtered_data_10000/metadata.json from the
migrated checkpoint history.

Not run automatically by any notebook or CI — run manually, once, with a
Hugging Face WRITE token:

    HF_TOKEN=... python scripts/migrate_sample_to_new_repo.py

Usage:
    python scripts/migrate_sample_to_new_repo.py
    python scripts/migrate_sample_to_new_repo.py --old-repo Rudra-G-23/qwen-coder-python-fim-data \\
        --new-repo Rudra-G-23/the-stack-v3-python-fim-data --path-prefix sample_filtered_data_10000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

from huggingface_hub import HfApi, hf_hub_download

from src.curation import checkpoint as ckpt

CHECKPOINT_PREFIX = "checkpoint_"
CHECKPOINT_SUFFIX = ".json"


def _classify(old_path: str) -> str:
    """Old flat repo has no folders — files are either checkpoint JSONs or
    chunk parquet files sitting at repo root."""
    if old_path.startswith(CHECKPOINT_PREFIX) and old_path.endswith(CHECKPOINT_SUFFIX):
        return "checkpoints"
    return "data"


def migrate(old_repo: str, new_repo: str, path_prefix: str, token: str) -> None:
    api = HfApi(token=token)
    api.create_repo(new_repo, repo_type="dataset", exist_ok=True)

    old_files = api.list_repo_files(old_repo, repo_type="dataset")
    new_files = set(api.list_repo_files(new_repo, repo_type="dataset"))

    copied, skipped = 0, 0
    for old_path in old_files:
        if old_path in (".gitattributes", "README.md"):
            continue
        subdir = _classify(old_path)
        new_path = f"{path_prefix}/{subdir}/{old_path}"

        if new_path in new_files:
            skipped += 1
            continue

        local_path = hf_hub_download(
            repo_id=old_repo, filename=old_path, repo_type="dataset", token=token
        )
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=new_path,
            repo_id=new_repo,
            repo_type="dataset",
            token=token,
        )
        copied += 1
        print(f"  copied {old_path} → {new_path}")

    print(f"\n✓ Migration done: {copied} files copied, {skipped} already present.")

    meta_path = ckpt.write_metadata_json(
        new_repo,
        path_prefix,
        source={"dataset": "HuggingFaceCode/stack-v3-train", "split": "train"},
        token=token,
    )
    print(f"✓ metadata.json written → {meta_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-repo", default="Rudra-G-23/qwen-coder-python-fim-data")
    parser.add_argument("--new-repo", default="Rudra-G-23/the-stack-v3-python-fim-data")
    parser.add_argument("--path-prefix", default="sample_filtered_data_10000")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN must be set (write access) to migrate data between HF repos.")
    migrate(args.old_repo, args.new_repo, args.path_prefix, hf_token)
