#!/usr/bin/env python
"""
scripts/run_safim_eval_1000.py
Phase B entrypoint: evaluate Base, Random-LoRA, and Distributed-LoRA on the
fixed `experiment/safim_eval_1000` FIM task set (built by
scripts/generate_safim_eval_1000.py), broken down by fim_type bucket.

This is the long-running, GPU, interruption-prone step (3 models x up to
1000 unbatched greedy-decode generations) — src.training.evaluate's
evaluate_model_by_type resumes per-model from a local + HF-mirrored
checkpoint JSONL, so a crash or Kaggle session restart loses at most
`checkpoint_every_n_records` records for whichever model was in flight, not
the whole run.

Usage:
    python scripts/run_safim_eval_1000.py
    python scripts/run_safim_eval_1000.py --config configs/data/safim_eval_1000.yaml
    python scripts/run_safim_eval_1000.py --max-samples 50   # fast calibration run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

# This pipeline loads exactly one 0.5B model at a time (Base, then each
# LoRA) — it never wants model parallelism. On a multi-GPU box (e.g. Kaggle
# "GPU T4 x2") unsloth otherwise shards even this tiny model across
# cuda:0/cuda:1, and generation then dies with "Expected all tensors to be
# on the same device" once embed_tokens lands on a different GPU than the
# tokenised inputs. Pin to one visible GPU before torch is imported (which
# happens transitively via src.training.evaluate below). Respect an
# explicit override if the caller already set it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import yaml
from huggingface_hub import hf_hub_download

from src.training.evaluate import (
    download_adapter_snapshot,
    load_fim_eval_records,
    run_evaluation_nway_by_type,
)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_wandb_run(project: str, name: str, job_type: str, group: str, tags: list[str]):
    """`project` may be "entity/project" (recommended — data-curation, FIM
    generation and this eval all share the `stack-v3-python-fim-data`
    project, kept separate from training's `qwen-coder-python-fim` one) or a
    bare project name. Mirrors scripts/generate_fim_variants.py's
    init_wandb_run / src/curation/wandb_logger.py's init_curation_run — same
    graceful no-op if WANDB_API_KEY isn't set, same entity/project split,
    same joint Weave init inside the W&B run. Returns the active wandb.Run or
    None so the eval always runs even with zero W&B setup."""
    if not os.environ.get("WANDB_API_KEY"):
        print("⚠  WANDB_API_KEY not set — skipping W&B eval tracking (graphs stay local only).")
        return None

    try:
        import wandb
        import weave  # noqa: F401  (imported for side-effect: patch tracing)
    except ImportError as exc:
        print(f"⚠  W&B / Weave not installed ({exc}) — skipping eval tracking.")
        return None

    full_project = project
    if "/" in full_project:
        entity, bare_project = full_project.split("/", 1)
    else:
        entity, bare_project = None, full_project

    run = wandb.init(
        entity=entity, project=bare_project, name=name, job_type=job_type, group=group, tags=tags
    )
    weave.init(full_project)
    print(f"✓ W&B eval run : {run.url}")
    return run


def _resolve_adapter_path(model_cfg: dict, model_hf_repo: str, token: str | None) -> str | None:
    subfolder = model_cfg.get("adapter_subfolder")
    if not subfolder:
        return None
    return download_adapter_snapshot(model_hf_repo, subfolder, token=token)


def run(config: dict, token: str | None, max_samples_override: int | None) -> None:
    ev = config["evaluation"]
    output = config["output"]

    wandb_project = config.get("wandb", {}).get(
        "project", "stack-v3-python-fim-data"
    )
    wandb_run = init_wandb_run(
        wandb_project,
        name="safim_eval_1000",
        job_type="evaluation",
        # clusters this eval next to the pilot's training/generation runs on
        # one W&B comparison panel (seed42 = the pilot seed both LoRA
        # variants were trained on).
        group=f"experiment-{config.get('sampling', {}).get('seed', 42)}",
        tags=["safim_eval_1000", "base-vs-random-vs-distributed"],
    )

    local_parquet = hf_hub_download(
        repo_id=output["hf_dataset_repo"],
        filename=f"{output['folder']}/data/fim_{output['config_name']}.parquet",
        repo_type="dataset",
        token=token,
    )
    test_records = load_fim_eval_records(local_parquet)
    print(f"Loaded {len(test_records)} fixed eval tasks from {output['folder']}")

    models = [
        {"label": m["label"], "adapter_path": _resolve_adapter_path(m, ev["model_hf_repo"], token)}
        for m in ev["models"]
    ]

    max_samples = max_samples_override if max_samples_override is not None else ev.get("max_samples")

    try:
        run_evaluation_nway_by_type(
            base_model_name=ev["base_model_name"],
            models=models,
            test_records=test_records,
            output_dir=ev["output_dir"],
            max_samples=max_samples,
            hf_checkpoint_repo=output["hf_dataset_repo"],
            hf_checkpoint_folder=output["results_folder"],
            checkpoint_every_n_records=ev["checkpoint_every_n_records"],
            token=token,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Base/Random-LoRA/Distributed-LoRA on the fixed safim_eval_1000 set."
    )
    parser.add_argument(
        "--config", default="configs/data/safim_eval_1000.yaml",
        help="Path to the safim_eval_1000 config.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Evaluate only the first N tasks per model instead of the full set — for a fast "
             "calibration run to measure real per-sample throughput. Overrides the config's "
             "evaluation.max_samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN must be set to load the eval set and adapters.")
    run(cfg, token=hf_token, max_samples_override=args.max_samples)
