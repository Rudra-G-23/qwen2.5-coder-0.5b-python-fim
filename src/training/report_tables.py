"""
src/training/report_tables.py
Markdown table generators for the paper's "Experimental Setup" and
reproducibility sections — pure functions over already-loaded config dicts
/ run-summary dicts so they're testable without a GPU or network access,
same pattern as src.training.safim_eval's aggregation functions.

Usage:
    from src.training.report_tables import (
        render_hyperparameter_table,
        render_compute_cost_table,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_hyperparameter_table(cfg: dict[str, Any]) -> str:
    """
    Markdown table of every LoRA/training hyperparameter in a loaded
    configs/training/lora.yaml (optionally deep-merged with a pilot
    override) — the standard "Experimental Setup" table. Reads whatever
    keys are present rather than hardcoding a fixed schema, so it stays
    correct if lora.yaml gains/loses fields.
    """
    model = cfg.get("model", {})
    lora = cfg.get("lora", {})
    training = cfg.get("training", {})

    rows: list[tuple[str, Any]] = [
        ("Base model", model.get("name")),
        ("Max sequence length", model.get("max_seq_length")),
        ("LoRA rank (r)", lora.get("r")),
        ("LoRA alpha", lora.get("alpha")),
        ("LoRA dropout", lora.get("dropout")),
        ("LoRA target modules", ", ".join(lora.get("target_modules", []))),
        ("Epochs", training.get("num_epochs")),
        ("Per-device train batch size", training.get("per_device_train_batch_size")),
        ("Gradient accumulation steps", training.get("gradient_accumulation_steps")),
        ("Learning rate", training.get("learning_rate")),
        ("LR scheduler", training.get("lr_scheduler_type")),
        ("Warmup steps", training.get("warmup_steps")),
        ("Optimizer", training.get("optim")),
        ("Seed", training.get("seed")),
        ("fp16", training.get("fp16")),
    ]

    lines = ["| Hyperparameter | Value |", "|---|---|"]
    for name, value in rows:
        if value is not None:
            lines.append(f"| {name} | {value} |")
    return "\n".join(lines) + "\n"


def write_hyperparameter_table(
    cfg: dict[str, Any], out_path: str = "reports/hyperparameter_table.md"
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_hyperparameter_table(cfg), encoding="utf-8")
    print(f"✓ Hyperparameter table → {path}")
    return path


def render_compute_cost_table(runs: list[dict[str, Any]]) -> str:
    """
    Markdown compute/cost table from a list of already-collected run
    summaries — one row per training run:
        {"run_id": str, "gpu_type": str, "duration_hours": float,
         "hourly_rate_usd": float (optional, default 0.0 for free Kaggle GPU
         quota — set it to a cloud-equivalent rate if the paper wants an
         opportunity-cost estimate instead)}

    Deliberately takes plain dicts rather than fetching from the W&B API
    itself: run duration lives in each run's summary._runtime, but GPU type
    and $/hr aren't something W&B knows on Kaggle's free tier, so collecting
    them is a manual/notebook-side step — this function only renders what's
    already been collected, same boundary as
    safim_eval.aggregate_pilot_results reading pre-written CSVs rather than
    calling HF/W&B itself.
    """
    lines = [
        "| Run | GPU | Duration (h) | Rate ($/h) | Est. cost ($) |",
        "|---|---|---|---|---|",
    ]
    total_hours = 0.0
    total_cost = 0.0
    for run in runs:
        duration = float(run["duration_hours"])
        rate = float(run.get("hourly_rate_usd", 0.0))
        cost = duration * rate
        total_hours += duration
        total_cost += cost
        lines.append(
            f"| {run['run_id']} | {run.get('gpu_type', '—')} | {duration:.2f} | "
            f"{rate:.2f} | {cost:.2f} |"
        )
    lines.append(f"| **Total** | | **{total_hours:.2f}** | | **{total_cost:.2f}** |")
    return "\n".join(lines) + "\n"


def write_compute_cost_table(
    runs: list[dict[str, Any]], out_path: str = "reports/compute_cost_table.md"
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_compute_cost_table(runs), encoding="utf-8")
    print(f"✓ Compute/cost table → {path}")
    return path


def log_plots_artifact(
    wandb_run: Any | None,
    plot_paths: list[str],
    *csv_paths: str,
    name: str = "eval-plots",
) -> None:
    """
    Bundle already-saved plot PNGs (+ optional metrics CSVs) into a single
    versioned wandb.Artifact and log it. wandb_run.log({...: wandb.Image(...)})
    (done at each plot_*() call site) only puts images in the run's Media
    tab — it isn't a downloadable, versioned artifact, so a report/paper
    step reading "the exact figures from run X" has nothing to pull. This
    is the versioned counterpart, keyed by run_id like the adapter artifact
    train_lora.py already logs.
    No-ops (with a skip message) when wandb_run is None or nothing to bundle.
    """
    if wandb_run is None:
        print("⚠  No active W&B run — skipping plots artifact.")
        return
    if not plot_paths and not csv_paths:
        print("⚠  No plots/CSVs generated — skipping plots artifact.")
        return

    import wandb

    artifact = wandb.Artifact(
        name=f"{wandb_run.id}-{name}",
        type="eval-plots",
        description=f"Evaluation plots and metrics CSVs for run {wandb_run.name}",
    )
    for path in [*plot_paths, *csv_paths]:
        if path and Path(path).exists():
            artifact.add_file(path)
    wandb_run.log_artifact(artifact)
    print(f"✓ Plots artifact logged → {artifact.name}")


def fetch_wandb_run_durations(
    project: str, entity: str | None = None, tags: list[str] | None = None
) -> list[dict[str, Any]]:
    """
    Thin W&B API wrapper: pulls run_id/duration_hours/gpu_type for runs in
    `project` (optionally filtered by `tags`, e.g. ["pilot"]), ready to feed
    into render_compute_cost_table. Requires WANDB_API_KEY — not covered by
    local_tests since it needs network + an authenticated W&B project,
    same boundary as run_safim_evaluation's own model-loading calls.
    """
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"tags": {"$in": tags}} if tags else None)

    collected: list[dict[str, Any]] = []
    for run in runs:
        duration_hours = run.summary.get("_runtime", 0) / 3600.0
        collected.append(
            {
                "run_id": run.name,
                "gpu_type": run.metadata.get("gpu", "—") if run.metadata else "—",
                "duration_hours": duration_hours,
            }
        )
    return collected
