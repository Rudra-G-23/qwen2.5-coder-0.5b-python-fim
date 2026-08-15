"""
src/curation/checkpoint.py
Stage 1 §4: HF-based resume/checkpoint logic for scripts/build_sample.py.

Checkpoint JSONs are stored alongside the data chunks in the same HF dataset
repo (no separate state store). On start, the highest shard_index/row_offset
checkpoint determines where to resume; the exact-dedup hash set is rebuilt
from previously uploaded chunk parquet files (checkpoint JSONs don't carry
per-file hashes) so cross-session duplicates aren't missed.

All Hugging Face Hub calls go through module-level `HfApi`/`hf_hub_download`
references so tests can monkeypatch them without any real network access.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

CHECKPOINT_PREFIX = "checkpoint_"
CHECKPOINT_SUFFIX = ".json"


# ── Checkpoint construction ───────────────────────────────────────────────────


def build_checkpoint(
    chunk_id: str,
    chunk_file: str,
    shard_index: int,
    row_offset: int,
    files_collected_this_chunk: int,
    raw_files_scanned_this_chunk: int,
    filter_stats: dict[str, int],
    cumulative_files_collected: int,
    quality_reject_by_reason: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a checkpoint dict matching .claude/data-v1.md §4's schema, plus
    `chunk_file` — the exact parquet filename this checkpoint corresponds to,
    so resume doesn't have to re-derive it from row ranges. `quality_reject_by_reason`
    (data-stage-2.md §3) is optional and defaults to empty — historical
    checkpoints written before this field existed simply omit it."""
    return {
        "chunk_id": chunk_id,
        "chunk_file": chunk_file,
        "shard_index": shard_index,
        "row_offset": row_offset,
        "files_collected_this_chunk": files_collected_this_chunk,
        "raw_files_scanned_this_chunk": raw_files_scanned_this_chunk,
        "filter_stats": dict(filter_stats),
        "quality_reject_by_reason": dict(quality_reject_by_reason or {}),
        "cumulative_files_collected": cumulative_files_collected,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Path-prefix helper ─────────────────────────────────────────────────────────


def _checkpoint_dir(path_prefix: str) -> str:
    """Where checkpoint_*.json files live in the repo. Empty prefix (the
    default) preserves the original flat-repo-root layout unchanged."""
    return f"{path_prefix}/checkpoints" if path_prefix else ""


def _with_dir(dirname: str, filename: str) -> str:
    return f"{dirname}/{filename}" if dirname else filename


# ── Listing + resume ──────────────────────────────────────────────────────────


def list_checkpoint_files(
    repo_id: str, token: str | None = None, repo_type: str = "dataset", path_prefix: str = ""
) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type=repo_type)
    checkpoint_dir = _checkpoint_dir(path_prefix)
    prefix = f"{checkpoint_dir}/{CHECKPOINT_PREFIX}" if checkpoint_dir else CHECKPOINT_PREFIX
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(CHECKPOINT_SUFFIX))


def load_all_checkpoints(
    repo_id: str, token: str | None = None, repo_type: str = "dataset", path_prefix: str = ""
) -> list[dict[str, Any]]:
    checkpoints = []
    for filename in list_checkpoint_files(
        repo_id, token=token, repo_type=repo_type, path_prefix=path_prefix
    ):
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type=repo_type, token=token
        )
        with open(local_path, encoding="utf-8") as f:
            checkpoints.append(json.load(f))
    return checkpoints


def pick_resume_checkpoint(checkpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the checkpoint with the highest (shard_index, row_offset) — the
    furthest point already collected."""
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda c: (c["shard_index"], c["row_offset"]))


def _download_hashes(
    repo_id: str, checkpoints: list[dict[str, Any]], token: str | None, repo_type: str
) -> set[str]:
    """Download each checkpoint's chunk parquet file and pull its content
    hashes out of curation_metadata. Shared by rebuild_seen_hashes (full
    history) and load_seen_hashes's delta path (only the chunks written
    since the last saved seen_hashes.json) — the expensive part either way
    is this per-chunk download+read, so keeping it in one place matters."""
    seen: set[str] = set()
    for checkpoint in checkpoints:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint["chunk_file"],
            repo_type=repo_type,
            token=token,
        )
        table = pq.read_table(local_path, columns=["curation_metadata"])
        for row in table.column("curation_metadata").to_pylist():
            seen.add(row["content_hash_sha256"])
    return seen


def rebuild_seen_hashes(
    repo_id: str,
    checkpoints: list[dict[str, Any]],
    resume_checkpoint: dict[str, Any],
    token: str | None = None,
    repo_type: str = "dataset",
) -> set[str]:
    """
    Rebuild the exact-dedup hash set from every chunk parquet file at or
    before the resume point, so a resumed session can't re-admit a duplicate
    a prior session already collected.

    This re-downloads and re-reads *every* historical chunk every time it's
    called — fine at a handful of chunks, but becomes the dominant cost of
    starting a new session once the corpus spans thousands of chunks. Prefer
    load_seen_hashes, which caches this result and only falls back to a full
    rebuild here when no cache exists yet.
    """
    resume_key = (resume_checkpoint["shard_index"], resume_checkpoint["row_offset"])
    relevant = [
        c for c in checkpoints if (c["shard_index"], c["row_offset"]) <= resume_key
    ]
    return _download_hashes(repo_id, relevant, token, repo_type)


# ── seen_hashes persistence (data-stage-2.md §1's "seen_hashes persistence
# timing" item) ──────────────────────────────────────────────────────────
#
# rebuild_seen_hashes's full re-download-every-resume cost grows with total
# corpus size, not session size — at millions of files it can end up eating
# most of a session's time budget before any new streaming even starts.
# seen_hashes.json caches the set as of a known checkpoint position so a
# resume only has to re-derive hashes for chunks written *since* that save
# — normally zero, or the handful from a session that crashed before it
# could save.

SEEN_HASHES_FILENAME = "seen_hashes.json"


def _seen_hashes_path(path_prefix: str) -> str:
    return _with_dir(_checkpoint_dir(path_prefix), SEEN_HASHES_FILENAME)


def save_seen_hashes(
    repo_id: str,
    path_prefix: str,
    hashes: set[str] | frozenset[str],
    as_of_shard_index: int,
    as_of_row_offset: int,
    token: str | None = None,
    repo_type: str = "dataset",
) -> None:
    """Persist the full seen-hash set as of (as_of_shard_index,
    as_of_row_offset) — the identity of the last checkpoint these hashes
    fully account for. Called once per session (not once per chunk) from
    scripts/build_sample.py's run_session, mirroring where
    write_filter_report/write_metadata_json are already called."""
    api = HfApi(token=token)
    payload = {
        "as_of_shard_index": as_of_shard_index,
        "as_of_row_offset": as_of_row_offset,
        "hashes": sorted(hashes),
    }
    api.upload_file(
        path_or_fileobj=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        path_in_repo=_seen_hashes_path(path_prefix),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )


def load_seen_hashes(
    repo_id: str,
    checkpoints: list[dict[str, Any]],
    resume_checkpoint: dict[str, Any],
    path_prefix: str = "",
    token: str | None = None,
    repo_type: str = "dataset",
) -> set[str]:
    """Fast-path resume: read the cached seen_hashes.json instead of
    re-downloading every historical chunk. Falls back to a full
    rebuild_seen_hashes() when no cache exists yet (fresh repo, or history
    that predates this file — e.g. the original Stage 1 checkpoints). If the
    cache is behind the resume point (a prior session checkpointed a chunk
    but crashed before it could save), only the missing tail of chunks is
    re-downloaded and merged in — never the full history."""
    resume_key = (resume_checkpoint["shard_index"], resume_checkpoint["row_offset"])

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=_seen_hashes_path(path_prefix),
            repo_type=repo_type,
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return rebuild_seen_hashes(
            repo_id, checkpoints, resume_checkpoint, token=token, repo_type=repo_type
        )

    with open(local_path, encoding="utf-8") as f:
        cached = json.load(f)
    as_of_key = (cached["as_of_shard_index"], cached["as_of_row_offset"])
    hashes = set(cached["hashes"])

    if as_of_key >= resume_key:
        return hashes

    missing = [
        c
        for c in checkpoints
        if as_of_key < (c["shard_index"], c["row_offset"]) <= resume_key
    ]
    hashes |= _download_hashes(repo_id, missing, token, repo_type)
    return hashes


# ── Upload (idempotent) ───────────────────────────────────────────────────────


def chunk_exists(
    repo_id: str, chunk_filename: str, token: str | None = None, repo_type: str = "dataset"
) -> bool:
    api = HfApi(token=token)
    return chunk_filename in api.list_repo_files(repo_id, repo_type=repo_type)


def upload_checkpoint(
    repo_id: str,
    checkpoint: dict[str, Any],
    token: str | None = None,
    repo_type: str = "dataset",
    path_prefix: str = "",
) -> None:
    api = HfApi(token=token)
    filename = _with_dir(
        _checkpoint_dir(path_prefix), f"{CHECKPOINT_PREFIX}{checkpoint['chunk_id']}{CHECKPOINT_SUFFIX}"
    )
    api.upload_file(
        path_or_fileobj=json.dumps(checkpoint, indent=2, ensure_ascii=False).encode("utf-8"),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )


# ── Filter report ─────────────────────────────────────────────────────────────

_STAGE_LABELS = {
    "language_reject": "Language filter (non-Python)",
    "vendor_reject": "Vendor filter",
    "license_reject": "License policy (ambiguous)",
    "quality_reject": "Quality filters (AST/size/binary/minified)",
    "secret_reject": "Secret detection",
    "exact_dup_reject": "Exact deduplication",
}


def render_filter_report(checkpoints: list[dict[str, Any]]) -> str:
    """Render the aggregated per-stage counts across all checkpoints as markdown."""
    totals: dict[str, int] = dict.fromkeys(_STAGE_LABELS, 0)
    raw_scanned = 0
    collected = 0

    for checkpoint in checkpoints:
        raw_scanned += checkpoint.get("raw_files_scanned_this_chunk", 0)
        collected = max(collected, checkpoint.get("cumulative_files_collected", 0))
        for stage, count in checkpoint.get("filter_stats", {}).items():
            totals[stage] = totals.get(stage, 0) + count

    lines = [
        "# Stage 1 Filter Report",
        "",
        f"- Chunks processed: {len(checkpoints)}",
        f"- Raw files scanned: {raw_scanned}",
        f"- Files collected (curated sample): {collected}",
        f"- Overall reject rate: {(1 - collected / raw_scanned) * 100:.2f}%"
        if raw_scanned
        else "- Overall reject rate: n/a (no files scanned yet)",
        "",
        "| Stage | Rejected |",
        "|---|---|",
    ]
    for stage, label in _STAGE_LABELS.items():
        lines.append(f"| {label} | {totals.get(stage, 0)} |")

    return "\n".join(lines) + "\n"


def write_filter_report(
    repo_id: str,
    out_path: str = "reports/stage1_filter_report.md",
    token: str | None = None,
    path_prefix: str = "",
) -> Path:
    checkpoints = load_all_checkpoints(repo_id, token=token, path_prefix=path_prefix)
    report = render_filter_report(checkpoints)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


# ── metadata.json (data-stage-2.md §3) ──────────────────────────────────────
#
# One aggregate metadata.json per folder — never a per-row dump. Per-row
# provenance (content hash, source shard/row, ...) already lives in every
# curated row's `curation_metadata` inside the parquet files themselves
# (see stream_source.build_curated_record); this file only explains *why*
# the folder's counts look the way they do, generated on demand from
# checkpoint history the same way render_filter_report is.


def render_metadata(
    checkpoints: list[dict[str, Any]], folder: str, source: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate the same checkpoint history render_filter_report uses into
    the data-stage-2.md §3 metadata.json shape. `quality_reject_by_reason`
    only reflects chunks collected after that field was added to the
    checkpoint schema — earlier (e.g. original Stage 1) checkpoints simply
    contribute nothing to it, so the total may undercount older history;
    `reject_by_stage` has no such gap since it was tracked from the start."""
    reject_by_stage: dict[str, int] = dict.fromkeys(_STAGE_LABELS, 0)
    quality_reject_by_reason: dict[str, int] = {}
    raw_scanned = 0
    collected = 0

    for checkpoint in checkpoints:
        raw_scanned += checkpoint.get("raw_files_scanned_this_chunk", 0)
        collected = max(collected, checkpoint.get("cumulative_files_collected", 0))
        for stage, count in checkpoint.get("filter_stats", {}).items():
            reject_by_stage[stage] = reject_by_stage.get(stage, 0) + count
        for reason, count in checkpoint.get("quality_reject_by_reason", {}).items():
            quality_reject_by_reason[reason] = quality_reject_by_reason.get(reason, 0) + count

    return {
        "folder": folder,
        "stage": "sample filter (10k pilot)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "counts": {
            "raw_files_scanned": raw_scanned,
            "files_kept": collected,
            "reject_by_stage": reject_by_stage,
            "quality_reject_by_reason": quality_reject_by_reason,
        },
    }


def write_metadata_json(
    repo_id: str,
    path_prefix: str,
    source: dict[str, Any],
    token: str | None = None,
    out_path: str | None = None,
) -> Path:
    checkpoints = load_all_checkpoints(repo_id, token=token, path_prefix=path_prefix)
    folder = path_prefix or repo_id
    metadata = render_metadata(checkpoints, folder=folder, source=source)
    path = Path(out_path or f"reports/{folder.replace('/', '_')}_metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=_with_dir(path_prefix, "metadata.json"),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    return path
