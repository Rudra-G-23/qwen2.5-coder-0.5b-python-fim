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
from src.curation import wandb_logger
from src.curation.stream_source import collect_chunk, iter_stack_v3_repos
from src.dedup.exact_dedup import ExactDedup

BYTES_PER_GB = 1024**3


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resume_state(
    hf_repo: str, token: str | None, path_prefix: str, continue_from: str | None = None
) -> tuple[int, int, int, ExactDedup, tuple[int, int] | None]:
    """Returns (start_shard_index, start_row_offset, cumulative_collected,
    dedup, latest_checkpoint_key). latest_checkpoint_key is
    (shard_index, row_offset) of the most recent checkpoint this dedup set
    is known to fully account for, or None if there's no checkpoint history
    yet — run_session threads it through to save_seen_hashes at the end.

    `continue_from`, if given, only matters on the "no checkpoint history
    yet" branch below (this path_prefix's very first-ever session). It makes
    this pool a strict *continuation* of a different, already-complete
    prefix's stream:
      1. its dedup set is seeded from that prefix's final seen_hashes.json
         (src.curation.checkpoint.seed_dedup_from_prefix), and
      2. streaming starts at that prefix's highest checkpoint position
         (shard_index + 1, row_offset) instead of (0, 0),
    so every file collected here is strictly newer than anything the source
    prefix scanned AND hash-disjoint from it — disjoint by construction, not
    by a downstream filter, and without re-streaming rows the source prefix
    already processed. `cumulative_collected` still starts at 0 — that
    counts THIS pool's own files, not the source prefix's.

    Every later resume takes the branch below unchanged and ignores
    `continue_from` — this prefix's own checkpoint history (which by then
    already includes the seeded hashes, since save_seen_hashes persists the
    full dedup set) is authoritative from then on."""
    checkpoints = ckpt.load_all_checkpoints(hf_repo, token=token, path_prefix=path_prefix)
    resume = ckpt.pick_resume_checkpoint(checkpoints)
    if resume is None:
        seed_hashes: set[str] = set()
        start_shard_index, start_row_offset = 0, 0
        if continue_from:
            seed_hashes = ckpt.seed_dedup_from_prefix(hf_repo, continue_from, token=token)
            source_resume = ckpt.pick_resume_checkpoint(
                ckpt.load_all_checkpoints(hf_repo, token=token, path_prefix=continue_from)
            )
            if source_resume is not None:
                start_shard_index = source_resume["shard_index"] + 1
                start_row_offset = source_resume["row_offset"]
            print(
                f"Continuing from {continue_from!r} (first run for this prefix only): "
                f"seeded dedup set with {len(seed_hashes)} hashes, streaming from "
                f"shard_index={start_shard_index}, row_offset={start_row_offset}."
            )
        return start_shard_index, start_row_offset, 0, ExactDedup(seed_hashes), None

    seen_hashes = ckpt.load_seen_hashes(
        hf_repo, checkpoints, resume, path_prefix=path_prefix, token=token
    )
    return (
        resume["shard_index"] + 1,
        resume["row_offset"],
        resume["cumulative_files_collected"],
        ExactDedup(seen_hashes),
        (resume["shard_index"], resume["row_offset"]),
    )


def _write_and_upload_chunk(
    chunk,
    hf_repo: str,
    chunk_id: str,
    chunk_filename: str,
    path_prefix: str,
    cumulative_files_collected: int,
    token: str | None,
    wandb_run,
    existing_files: set[str],
) -> None:
    # Full in-repo path — stored as-is in the checkpoint's chunk_file so
    # rebuild_seen_hashes/chunk_exists don't need their own path_prefix param.
    path_in_repo = f"{path_prefix}/data/{chunk_filename}" if path_prefix else chunk_filename

    table = pa.Table.from_pylist(chunk.records)
    local_path = f"/tmp/{chunk_filename}"
    pq.write_table(table, local_path)

    # existing_files is fetched once per session (run_session) instead of
    # re-listing the whole repo on every chunk (ckpt.chunk_exists) — at
    # thousands of chunks, one list_repo_files call per chunk makes each
    # chunk slower than the last as the repo grows.
    api = HfApi(token=token)
    if path_in_repo not in existing_files:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=hf_repo,
            repo_type="dataset",
            token=token,
        )
        existing_files.add(path_in_repo)
    os.remove(local_path)

    checkpoint = ckpt.build_checkpoint(
        chunk_id=chunk_id,
        chunk_file=path_in_repo,
        shard_index=chunk.shard_index,
        row_offset=chunk.row_offset,
        files_collected_this_chunk=len(chunk.records),
        raw_files_scanned_this_chunk=chunk.raw_files_scanned,
        filter_stats=chunk.filter_stats,
        cumulative_files_collected=cumulative_files_collected,
        quality_reject_by_reason=chunk.quality_reject_by_reason,
    )
    ckpt.upload_checkpoint(hf_repo, checkpoint, token=token, path_prefix=path_prefix)

    wandb_logger.log_chunk(
        wandb_run,
        step=chunk.shard_index,
        filter_stats=chunk.filter_stats,
        quality_reject_by_reason=chunk.quality_reject_by_reason,
        raw_files_scanned_this_chunk=chunk.raw_files_scanned,
        cumulative_files_collected=cumulative_files_collected,
        cumulative_bytes_collected=chunk.bytes_collected,
    )


def run_session(config: dict, max_gb: float, token: str | None) -> None:
    hf_repo = config["checkpoint"]["hf_dataset_repo"]
    path_prefix = config["checkpoint"].get("path_prefix", "")
    target_files = config["target"]["files"]
    chunk_size = config["target"]["chunk_size"]
    max_bytes = max_gb * BYTES_PER_GB
    source_meta = {"dataset": config["source"]["dataset"], "split": config["source"]["split"]}

    continue_from = config["checkpoint"].get("continue_from_path_prefix")
    shard_index, row_offset, cumulative_collected, dedup, latest_key = _resume_state(
        hf_repo, token, path_prefix, continue_from=continue_from
    )
    print(f"Resuming at shard_index={shard_index}, row_offset={row_offset}, "
          f"cumulative_files_collected={cumulative_collected}")

    if cumulative_collected >= target_files:
        print(f"Target of {target_files} files already reached. Nothing to do.")
        ckpt.write_filter_report(hf_repo, token=token, path_prefix=path_prefix)
        ckpt.write_metadata_json(hf_repo, path_prefix, source_meta, token=token)
        if latest_key is not None:
            ckpt.save_seen_hashes(hf_repo, path_prefix, dedup.seen_hashes, *latest_key, token=token)
        return

    wandb_run = wandb_logger.init_curation_run(
        project=config.get("wandb", {}).get(
            "project", "521er1007-national-institute-of-technology-rourkela/stack-v3-python-fim-data"
        ),
        run_id=f"curation-{hf_repo.split('/')[-1]}-{path_prefix or 'root'}",
        group=path_prefix or hf_repo,
        tags=["data-curation", path_prefix] if path_prefix else ["data-curation"],
    )

    repo_iterator = iter_stack_v3_repos(
        config["source"]["dataset"], config["source"]["split"], row_offset=row_offset
    )

    # Listed once per session instead of once per chunk (see
    # _write_and_upload_chunk) — a per-chunk list_repo_files call makes each
    # successive chunk slower to upload as the repo grows into the
    # thousands of chunks a multi-million-file corpus implies.
    existing_files = set(HfApi(token=token).list_repo_files(hf_repo, repo_type="dataset"))

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
            chunk, hf_repo, chunk_id, chunk_filename, path_prefix, cumulative_collected,
            token, wandb_run, existing_files,
        )
        latest_key = (chunk.shard_index, chunk.row_offset)

        print(
            f"  chunk {chunk_id}: +{len(chunk.records)} files "
            f"(scanned {chunk.raw_files_scanned}, cumulative {cumulative_collected}/{target_files}), "
            f"session bytes {bytes_used_this_session / BYTES_PER_GB:.3f} GB"
        )
        shard_index += 1

    ckpt.write_filter_report(hf_repo, token=token, path_prefix=path_prefix)
    ckpt.write_metadata_json(hf_repo, path_prefix, source_meta, token=token)
    if latest_key is not None:
        ckpt.save_seen_hashes(hf_repo, path_prefix, dedup.seen_hashes, *latest_key, token=token)

    if wandb_run is not None:
        wandb_run.finish()

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
        report_repo = cfg["checkpoint"]["hf_dataset_repo"]
        report_prefix = cfg["checkpoint"].get("path_prefix", "")
        path = ckpt.write_filter_report(report_repo, token=hf_token, path_prefix=report_prefix)
        print(f"✓ Report written → {path}")
        meta_source = {"dataset": cfg["source"]["dataset"], "split": cfg["source"]["split"]}
        meta_path = ckpt.write_metadata_json(report_repo, report_prefix, meta_source, token=hf_token)
        print(f"✓ metadata.json written → {meta_path}")
    else:
        if args.max_gb is None:
            raise SystemExit("--max-gb is required unless --report-only is set.")
        if not hf_token:
            raise SystemExit("HF_TOKEN must be set to stream + push chunks to Hugging Face.")
        run_session(cfg, max_gb=args.max_gb, token=hf_token)
