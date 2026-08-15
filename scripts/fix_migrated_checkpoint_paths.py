#!/usr/bin/env python
"""
scripts/fix_migrated_checkpoint_paths.py
One-time repair for checkpoint JSONs already uploaded to
Rudra-G-23/the-stack-v3-python-fim-data by an earlier, buggy run of
migrate_sample_to_new_repo.py: their `chunk_file` field still holds the old
flat-repo filename (e.g. "chunk_shard0000_row0-1673.parquet") instead of the
new nested path (e.g. "sample_filtered_data_10000/data/chunk_shard0000_row0-
1673.parquet"), which makes rebuild_seen_hashes 404 on resume.

migrate_sample_to_new_repo.py itself is now fixed to write the correct path
on future migrations — this script only patches the checkpoints that were
already uploaded before that fix landed.

Idempotent — skips any checkpoint whose chunk_file is already prefixed.

Usage:
    HF_TOKEN=... python scripts/fix_migrated_checkpoint_paths.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

from huggingface_hub import HfApi, hf_hub_download

from src.curation import checkpoint as ckpt


def fix(repo_id: str, path_prefix: str, token: str) -> None:
    api = HfApi(token=token)
    filenames = ckpt.list_checkpoint_files(repo_id, token=token, path_prefix=path_prefix)

    fixed, skipped = 0, 0
    for filename in filenames:
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", token=token
        )
        with open(local_path, encoding="utf-8") as f:
            data = json.load(f)

        if data["chunk_file"].startswith(f"{path_prefix}/data/"):
            skipped += 1
            continue

        data["chunk_file"] = f"{path_prefix}/data/{data['chunk_file']}"
        api.upload_file(
            path_or_fileobj=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        fixed += 1
        print(f"  fixed {filename} -> chunk_file={data['chunk_file']}")

    print(f"\n✓ Repair done: {fixed} checkpoints fixed, {skipped} already correct.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Rudra-G-23/the-stack-v3-python-fim-data")
    parser.add_argument("--path-prefix", default="sample_filtered_data_10000")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN must be set (write access) to patch checkpoints.")
    fix(args.repo, args.path_prefix, hf_token)
