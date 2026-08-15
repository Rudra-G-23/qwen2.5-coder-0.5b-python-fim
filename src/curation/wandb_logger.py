"""
src/curation/wandb_logger.py
data-stage-2.md §7: thin W&B wrapper for scripts/build_sample.py's per-chunk
curation loop. Mirrors src/training/train_lora.py's init_wandb — same
single project (`qwen-coder-python-fim`), same graceful no-op if
WANDB_API_KEY isn't set, distinguished from training runs by
`job_type="data-curation"`.

A full-corpus run spans many Kaggle 12-hour sessions, so this uses a fixed
run id + resume="allow": every session's chunk logs land on one continuous
time series instead of fragmenting into one run per session.
"""

from __future__ import annotations

import os
from typing import Any


def init_curation_run(project: str, run_id: str, group: str, tags: list[str] | None = None) -> Any | None:
    """Returns the active wandb.Run, or None if WANDB_API_KEY isn't set —
    curation must work with zero W&B setup, tracking is opt-in."""
    if not os.environ.get("WANDB_API_KEY"):
        print("⚠  WANDB_API_KEY not set — skipping W&B curation tracking.")
        return None

    import wandb

    return wandb.init(
        project=project,
        id=run_id,
        resume="allow",
        job_type="data-curation",
        group=group,
        tags=tags or [],
    )


def log_chunk(
    run: Any | None,
    step: int,
    filter_stats: dict[str, int],
    quality_reject_by_reason: dict[str, int],
    raw_files_scanned_this_chunk: int,
    cumulative_files_collected: int,
    cumulative_bytes_collected: float,
) -> None:
    """Log one chunk's curation progress. No-op if `run` is None (W&B disabled)."""
    if run is None:
        return

    dup_rejects = filter_stats.get("exact_dup_reject", 0)
    dedup_rate_this_chunk = (
        dup_rejects / raw_files_scanned_this_chunk if raw_files_scanned_this_chunk else 0.0
    )

    run.log(
        {
            **{f"filter_funnel/{stage}": count for stage, count in filter_stats.items()},
            **{f"quality_reject_reason/{reason}": count for reason, count in quality_reject_by_reason.items()},
            "curation/cumulative_files_collected": cumulative_files_collected,
            "curation/cumulative_bytes_collected": cumulative_bytes_collected,
            "curation/dedup_rate_this_chunk": dedup_rate_this_chunk,
        },
        step=step,
    )
