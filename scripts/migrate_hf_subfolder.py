#!/usr/bin/env python
"""
scripts/migrate_hf_subfolder.py
One-time, server-side move of every file under `--old-prefix` to
`--new-prefix` in an HF model repo, in a single atomic commit. No local
download/re-upload of file contents — uses CommitOperationCopy (server-side,
byte-for-byte, works for LFS files too) followed by CommitOperationDelete on
the old paths.

Written for migrating scripts/run_pilot_experiment.py's pre-per-seed-subfolder
pushes (e.g. "experiment/random_model/*" -> "experiment/random_model/seed42/*")
after that fix landed, but takes arbitrary prefixes so it's reusable for any
future rename of the same shape.

Usage:
    python scripts/migrate_hf_subfolder.py \
        --repo Rudra-G-23/qwen2.5-coder-0.5b-python-fim \
        --old-prefix experiment/random_model \
        --new-prefix experiment/random_model/seed42

Requires WANDB_API_KEY / HF_TOKEN in the environment (e.g.
`set -a; source .env; set +a` — see .env.example, same convention as every
other script here). Needs a write token.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi


def migrate_subfolder(repo: str, old_prefix: str, new_prefix: str, token: str | None) -> int:
    """Moves every file under old_prefix/ to new_prefix/ in one commit.
    Returns the number of files moved. Raises SystemExit if old_prefix has
    no files (nothing to migrate — likely already done, or a typo'd path)."""
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(repo, repo_type="model") if f.startswith(f"{old_prefix}/")]
    if not files:
        raise SystemExit(f"No files found under {old_prefix!r} in {repo} — nothing to migrate.")

    ops = []
    for f in files:
        new_path = new_prefix + f[len(old_prefix):]
        ops.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=new_path))
        print(f"  {f}  →  {new_path}")
    for f in files:
        ops.append(CommitOperationDelete(path_in_repo=f))

    api.create_commit(
        repo_id=repo,
        repo_type="model",
        operations=ops,
        commit_message=f"Migrate {old_prefix} -> {new_prefix} (per-seed subfolder fix)",
    )
    print(f"\n✓ Migrated {len(files)} file(s): {old_prefix}/ → {new_prefix}/")
    return len(files)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Server-side move of an HF model repo's subfolder.")
    parser.add_argument("--repo", required=True, help="HF model repo id, e.g. entity/model-name")
    parser.add_argument("--old-prefix", required=True, help="Existing path prefix to move from")
    parser.add_argument("--new-prefix", required=True, help="New path prefix to move to")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    token = os.environ.get("HF_TOKEN", "").strip() or None
    if not token:
        raise SystemExit("HF_TOKEN not set — export it first (e.g. `set -a; source .env; set +a`).")
    migrate_subfolder(args.repo, args.old_prefix, args.new_prefix, token)


if __name__ == "__main__":
    main()
