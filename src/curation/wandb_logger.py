"""
src/curation/wandb_logger.py
data-stage-2.md §7: thin W&B + Weave wrapper for scripts/build_sample.py's
per-chunk curation loop. Mirrors src/training/train_lora.py's init_wandb —
same graceful no-op if WANDB_API_KEY isn't set, same "entity/project"
config-string convention, same joint Weave init inside the W&B run —
distinguished from training runs by `job_type="data-curation"` and by
using a separate project (`stack-v3-python-fim-data`, all data-curation
and FIM-generation monitoring — see configs/data/*.yaml's `wandb.project`)
instead of the training project.

A full-corpus run spans many Kaggle 12-hour sessions, so this uses a fixed
run id + resume="allow": every session's chunk logs land on one continuous
time series instead of fragmenting into one run per session.
"""

from __future__ import annotations

import os
from typing import Any


def init_curation_run(project: str, run_id: str, group: str, tags: list[str] | None = None) -> Any | None:
    """Returns the active wandb.Run, or None if WANDB_API_KEY isn't set or
    wandb/weave aren't installed — curation must work with zero W&B setup,
    tracking is opt-in. `project` may be "entity/project" (recommended,
    matches train_lora.py's init_wandb) or a bare project name."""
    if not os.environ.get("WANDB_API_KEY"):
        print("⚠  WANDB_API_KEY not set — skipping W&B curation tracking.")
        return None

    try:
        import wandb
        import weave  # noqa: F401  (imported for side-effect: patch tracing)
    except ImportError as exc:
        print(f"⚠  W&B / Weave not installed ({exc}) — skipping curation tracking.")
        return None

    full_project = project
    if "/" in full_project:
        entity, bare_project = full_project.split("/", 1)
    else:
        entity, bare_project = None, full_project

    run = wandb.init(
        entity=entity,
        project=bare_project,
        id=run_id,
        resume="allow",
        job_type="data-curation",
        group=group,
        tags=tags or [],
    )
    weave.init(full_project)
    return run


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
