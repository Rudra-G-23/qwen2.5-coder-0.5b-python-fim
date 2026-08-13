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
) -> dict[str, Any]:
    """Build a checkpoint dict matching .claude/data-v1.md §4's schema, plus
    `chunk_file` — the exact parquet filename this checkpoint corresponds to,
    so resume doesn't have to re-derive it from row ranges."""
    return {
        "chunk_id": chunk_id,
        "chunk_file": chunk_file,
        "shard_index": shard_index,
        "row_offset": row_offset,
        "files_collected_this_chunk": files_collected_this_chunk,
        "raw_files_scanned_this_chunk": raw_files_scanned_this_chunk,
        "filter_stats": dict(filter_stats),
        "cumulative_files_collected": cumulative_files_collected,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Listing + resume ──────────────────────────────────────────────────────────


def list_checkpoint_files(
    repo_id: str, token: str | None = None, repo_type: str = "dataset"
) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type=repo_type)
    return sorted(
        f for f in files if f.startswith(CHECKPOINT_PREFIX) and f.endswith(CHECKPOINT_SUFFIX)
    )


def load_all_checkpoints(
    repo_id: str, token: str | None = None, repo_type: str = "dataset"
) -> list[dict[str, Any]]:
    checkpoints = []
    for filename in list_checkpoint_files(repo_id, token=token, repo_type=repo_type):
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
    """
    resume_key = (resume_checkpoint["shard_index"], resume_checkpoint["row_offset"])
    seen: set[str] = set()

    for checkpoint in checkpoints:
        if (checkpoint["shard_index"], checkpoint["row_offset"]) > resume_key:
            continue  # chunk collected after the resume point — not part of history yet

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
) -> None:
    api = HfApi(token=token)
    filename = f"{CHECKPOINT_PREFIX}{checkpoint['chunk_id']}{CHECKPOINT_SUFFIX}"
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
    repo_id: str, out_path: str = "reports/stage1_filter_report.md", token: str | None = None
) -> Path:
    checkpoints = load_all_checkpoints(repo_id, token=token)
    report = render_filter_report(checkpoints)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
