#!/usr/bin/env python
"""
scripts/run_pilot_experiment.py
Stage 1 §7 entrypoint: trains both FIM variants (random, planned) across the
seeds in configs/training/pilot.yaml. Every run starts fresh from the base
model with the shared hyperparameters in configs/training/lora.yaml
(untouched) — only training data and seed vary between runs, via
src.training.train_lora.train()'s `overrides` param.

Run on Kaggle (GPU required) — see notebooks/kaggle_wandb.ipynb for the
pattern this reuses; running locally without a GPU will only get as far as
train()'s own model-loading step.

Usage:
    python scripts/run_pilot_experiment.py
    python scripts/run_pilot_experiment.py --pilot-config configs/training/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download

from src.data_prep import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
from src.training.train_lora import train


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def materialize_variant_jsonl(
    dataset_hf_repo: str,
    dataset_config: str,
    token: str | None,
    cache_dir: str = "data/pilot",
) -> str:
    """
    train_lora.load_fim_dataset() reads a local JSONL with a "text" field —
    keep that loading path untouched and do the HF-parquet → local-JSONL
    conversion here instead, since it's sweep-specific glue, not core
    training logic.
    """
    local_parquet = hf_hub_download(
        repo_id=dataset_hf_repo,
        filename=f"{dataset_config}/fim_{dataset_config}.parquet",
        repo_type="dataset",
        token=token,
    )
    table = pq.read_table(local_parquet)

    out_path = Path(cache_dir) / f"{dataset_config}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in table.to_pylist():
            text = f"{FIM_PREFIX}{row['prefix']}{FIM_SUFFIX}{row['suffix']}{FIM_MIDDLE}{row['middle']}"
            f.write(json.dumps({**row, "text": text}, ensure_ascii=False) + "\n")

    return str(out_path)


def run(pilot_cfg: dict, token: str | None) -> None:
    base_config = pilot_cfg["base_config"]
    wandb_cfg = pilot_cfg.get("wandb", {})
    variant_jsonl_cache: dict[str, str] = {}

    for variant_name, variant in pilot_cfg["variants"].items():
        if variant_name not in variant_jsonl_cache:
            variant_jsonl_cache[variant_name] = materialize_variant_jsonl(
                variant["dataset_hf_repo"], variant["dataset_config"], token
            )
        data_path = variant_jsonl_cache[variant_name]

        for seed in pilot_cfg["seeds"]:
            overrides = {
                "data": {"path": data_path},
                "training": {"seed": seed},
                "output": {"hf_repo": variant["output_hf_repo"]},
                "wandb": {
                    "project": wandb_cfg["project"],
                    "job_type": wandb_cfg.get("job_type", "train"),
                    # reuses build_run_id's existing attempt/dataset_version
                    # fields to get a distinct, deterministic run id per
                    # (variant, seed) without new run-id logic
                    "attempt": seed,
                    "dataset_version": variant["dataset_config"],
                    "tags": [variant_name, f"seed{seed}"],
                },
            }
            print(f"\n{'=' * 60}\nVariant: {variant_name}   Seed: {seed}\n{'=' * 60}")
            train(config_path=base_config, overrides=overrides)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 pilot: random vs. planned FIM, seed-controlled."
    )
    parser.add_argument(
        "--pilot-config", default="configs/training/pilot.yaml",
        help="Path to the pilot sweep config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.pilot_config)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN must be set to load FIM variants and push adapters.")
    run(cfg, token=hf_token)
