#!/usr/bin/env python
"""
scripts/build_sample.py
Stage 1 entrypoint: stream HuggingFaceCode/stack-v3-train, filter, checkpoint,
and push curated Python files to the HF dataset repo in configs/data/stack_v3_filter.yaml.

Usage:
    python scripts/build_sample.py --max-gb 2.0
    python scripts/build_sample.py --report-only   # regenerate reports/stage1_filter_report.md
                                                     # from existing checkpoints, no new streaming

Never downloads the full corpus — always streams, and stops at whichever
comes first: configs/data/stack_v3_filter.yaml's `target.files`, the
`--max-gb` session budget, or the upstream stream running dry.

Implementation note: the checkpoint schema's `shard_index` is used here as an
incrementing chunk sequence number, not a literal HF-Hub parquet shard id —
plain streaming iteration over `datasets.load_dataset(..., streaming=True)`
doesn't expose the physical shard boundary without reaching into `datasets`
internals, and a monotonic per-run counter serves the same resume/ordering
purpose the spec's example schema needs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi

from src.curation import checkpoint as ckpt
from src.curation.stream_source import collect_chunk, iter_stack_v3_repos
from src.dedup.exact_dedup import ExactDedup

BYTES_PER_GB = 1024**3


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resume_state(hf_repo: str, token: str | None) -> tuple[int, int, int, ExactDedup]:
    """Returns (start_shard_index, start_row_offset, cumulative_collected, dedup)."""
    checkpoints = ckpt.load_all_checkpoints(hf_repo, token=token)
    resume = ckpt.pick_resume_checkpoint(checkpoints)
    if resume is None:
        return 0, 0, 0, ExactDedup()

    seen_hashes = ckpt.rebuild_seen_hashes(hf_repo, checkpoints, resume, token=token)
    return (
        resume["shard_index"] + 1,
        resume["row_offset"],
        resume["cumulative_files_collected"],
        ExactDedup(seen_hashes),
    )


def _write_and_upload_chunk(
    chunk,
    hf_repo: str,
    chunk_id: str,
    chunk_filename: str,
    cumulative_files_collected: int,
    token: str | None,
) -> None:
    table = pa.Table.from_pylist(chunk.records)
    local_path = f"/tmp/{chunk_filename}"
    pq.write_table(table, local_path)

    api = HfApi(token=token)
    if not ckpt.chunk_exists(hf_repo, chunk_filename, token=token):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=chunk_filename,
            repo_id=hf_repo,
            repo_type="dataset",
            token=token,
        )
    os.remove(local_path)

    checkpoint = ckpt.build_checkpoint(
        chunk_id=chunk_id,
        chunk_file=chunk_filename,
        shard_index=chunk.shard_index,
        row_offset=chunk.row_offset,
        files_collected_this_chunk=len(chunk.records),
        raw_files_scanned_this_chunk=chunk.raw_files_scanned,
        filter_stats=chunk.filter_stats,
        cumulative_files_collected=cumulative_files_collected,
    )
    ckpt.upload_checkpoint(hf_repo, checkpoint, token=token)


def run_session(config: dict, max_gb: float, token: str | None) -> None:
    hf_repo = config["checkpoint"]["hf_dataset_repo"]
    target_files = config["target"]["files"]
    chunk_size = config["target"]["chunk_size"]
    max_bytes = max_gb * BYTES_PER_GB

    shard_index, row_offset, cumulative_collected, dedup = _resume_state(hf_repo, token)
    print(f"Resuming at shard_index={shard_index}, row_offset={row_offset}, "
          f"cumulative_files_collected={cumulative_collected}")

    if cumulative_collected >= target_files:
        print(f"Target of {target_files} files already reached. Nothing to do.")
        ckpt.write_filter_report(hf_repo, token=token)
        return

    repo_iterator = iter_stack_v3_repos(
        config["source"]["dataset"], config["source"]["split"], row_offset=row_offset
    )

    bytes_used_this_session = 0.0
    exhausted = False

    while cumulative_collected < target_files and bytes_used_this_session < max_bytes and not exhausted:
        remaining_files = target_files - cumulative_collected
        remaining_bytes = max_bytes - bytes_used_this_session
        row_start = row_offset
        chunk, exhausted = collect_chunk(
            repo_iterator=repo_iterator,
            config=config,
            dedup=dedup,
            chunk_size=min(chunk_size, remaining_files),
            start_row_offset=row_offset,
            shard_index=shard_index,
            max_bytes_remaining=remaining_bytes,
        )
        if chunk is None:
            break

        cumulative_collected += len(chunk.records)
        bytes_used_this_session += chunk.bytes_collected
        row_offset = chunk.row_offset

        chunk_id = f"chunk_{shard_index:04d}"
        chunk_filename = config["checkpoint"]["chunk_filename_template"].format(
            shard_index=shard_index, row_start=row_start, row_end=row_offset
        )
        _write_and_upload_chunk(
            chunk, hf_repo, chunk_id, chunk_filename, cumulative_collected, token
        )

        print(
            f"  chunk {chunk_id}: +{len(chunk.records)} files "
            f"(scanned {chunk.raw_files_scanned}, cumulative {cumulative_collected}/{target_files}), "
            f"session bytes {bytes_used_this_session / BYTES_PER_GB:.3f} GB"
        )
        shard_index += 1

    ckpt.write_filter_report(hf_repo, token=token)

    if cumulative_collected >= target_files:
        print(f"\n✓ Target reached: {cumulative_collected} files collected.")
    elif exhausted:
        print(f"\n⚠  Upstream stream exhausted at {cumulative_collected} files (target {target_files}).")
    else:
        print(f"\n✓ Session budget ({max_gb} GB) reached at {cumulative_collected} files. Resume later.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream + filter + checkpoint stack-v3-train Python files to HF."
    )
    parser.add_argument(
        "--max-gb", type=float, default=None,
        help="Session budget: stop once this many GB of collected content is reached.",
    )
    parser.add_argument(
        "--config", default="configs/data/stack_v3_filter.yaml",
        help="Path to the filter config (default: configs/data/stack_v3_filter.yaml)",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Regenerate reports/stage1_filter_report.md from existing checkpoints; no streaming.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    hf_token = os.environ.get("HF_TOKEN")

    if args.report_only:
        path = ckpt.write_filter_report(cfg["checkpoint"]["hf_dataset_repo"], token=hf_token)
        print(f"✓ Report written → {path}")
    else:
        if args.max_gb is None:
            raise SystemExit("--max-gb is required unless --report-only is set.")
        if not hf_token:
            raise SystemExit("HF_TOKEN must be set to stream + push chunks to Hugging Face.")
        run_session(cfg, max_gb=args.max_gb, token=hf_token)
