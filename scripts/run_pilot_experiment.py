#!/usr/bin/env python
"""
scripts/run_pilot_experiment.py
Stage 1 §7-8 entrypoint: trains both FIM variants (random, planned) across the
seeds in configs/training/pilot.yaml, then SAFIM-evaluates each resulting
adapter so the pilot comparison in scripts/analyze_pilot_results.py has real
per-(variant, seed) numbers to aggregate. Every run starts fresh from the base
model with the shared hyperparameters in configs/training/lora.yaml
(untouched) — only training data and seed vary between runs, via
src.training.train_lora.train()'s `overrides` param.

Both variants push into ONE shared HF model repo
(configs/training/pilot.yaml's output_hf_repo), separated by subfolder
(output_subfolder, e.g. "experiment/random_model") rather than by repo — see
data-stage-1.md §7. Future fine-tune approaches on whichever variant wins
(LoRA, QLoRA, ...) land in the same repo too, under their own subfolders.

Known limitation: each *variant*'s subfolder is shared across all its seeds
(not one subfolder per seed) — each seed's push lands as a new commit to
that subfolder, so only the last seed's adapter is on the default branch;
earlier seeds are only recoverable by HF commit SHA. This doesn't affect the
pilot comparison itself (SAFIM eval runs immediately on each seed's local
adapter, right after that seed's training, before the next seed's push can
touch it) — it only affects re-fetching a specific past seed's adapter from
HF later. Fixing it means deciding a new per-seed subfolder naming scheme,
which is a naming decision for you to make, not something to silently invent
here (data-stage-2.md §1 reserves naming decisions the same way).

Run on Kaggle (GPU required) — see notebooks/kaggle_wandb.ipynb for the
pattern this reuses; running locally without a GPU will only get as far as
train()'s own model-loading step.

Usage:
    python scripts/run_pilot_experiment.py
    python scripts/run_pilot_experiment.py --pilot-config configs/training/pilot.yaml
    python scripts/run_pilot_experiment.py --variant random
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
from src.training.safim_eval import run_safim_evaluation
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


def run(
    pilot_cfg: dict,
    token: str | None,
    only_variant: str | None = None,
    safim_max_samples: int = 100,
) -> None:
    base_config = pilot_cfg["base_config"]
    base_cfg = load_config(base_config)
    wandb_cfg = pilot_cfg.get("wandb", {})
    variant_jsonl_cache: dict[str, str] = {}

    variants = pilot_cfg["variants"]
    if only_variant is not None:
        if only_variant not in variants:
            raise SystemExit(f"--variant {only_variant!r} not in pilot config variants: {list(variants)}")
        variants = {only_variant: variants[only_variant]}

    adapter_path = f"{base_cfg['output']['dir']}/lora_adapter"

    for variant_name, variant in variants.items():
        if variant_name not in variant_jsonl_cache:
            variant_jsonl_cache[variant_name] = materialize_variant_jsonl(
                variant["dataset_hf_repo"], variant["dataset_config"], token
            )
        data_path = variant_jsonl_cache[variant_name]

        for seed in pilot_cfg["seeds"]:
            overrides = {
                "data": {"path": data_path},
                "training": {"seed": seed},
                "output": {
                    "hf_repo": pilot_cfg["output_hf_repo"],
                    "subfolder": variant["output_subfolder"],
                },
                "wandb": {
                    "project": wandb_cfg["project"],
                    "job_type": wandb_cfg.get("job_type", "train"),
                    # reuses build_run_id's existing attempt/dataset_version
                    # fields to get a distinct, deterministic run id per
                    # (variant, seed) without new run-id logic
                    "attempt": seed,
                    "dataset_version": variant["dataset_config"],
                    # Live filtering during a run: search/filter by any of
                    # these in the W&B UI. "pilot" separates this phase from
                    # later fine-tune-approach runs (LoRA/QLoRA on the
                    # winning variant) landing in the same project.
                    "tags": ["pilot", variant_name, f"seed{seed}"],
                    # Groups all of this variant's seeds onto one comparison
                    # panel in the W&B UI instead of 3 separate ungrouped
                    # runs — data-stage-2.md §7's flagged monitoring gap.
                    "group": variant_name,
                },
            }
            print(f"\n{'=' * 60}\nVariant: {variant_name}   Seed: {seed}\n{'=' * 60}")
            # finish_wandb=False: keep this run open so SAFIM eval below logs
            # live progress into the SAME run instead of a finished one —
            # wandb silently drops .log() calls made after .finish(), so
            # this has to be set before train() returns, not patched after.
            wandb_run = train(config_path=base_config, overrides=overrides, finish_wandb=False)

            # SAFIM eval runs on this seed's adapter immediately, before the
            # next seed's train() call overwrites the shared local
            # output.dir — see the module docstring's "known limitation"
            # note on why this ordering matters.
            print(f"\n▶  SAFIM eval — variant={variant_name} seed={seed}")
            run_safim_evaluation(
                base_model_name=base_cfg["model"]["name"],
                adapter_path=adapter_path,
                output_dir=f"results/pilot/{variant_name}_seed{seed}",
                max_samples=safim_max_samples,
                seed=seed,
                wandb_run=wandb_run,
            )

            if wandb_run is not None:
                wandb_run.finish()
                print(f"✓ W&B run finished → {wandb_run.url}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 pilot: random vs. planned FIM, seed-controlled."
    )
    parser.add_argument(
        "--pilot-config", default="configs/training/pilot.yaml",
        help="Path to the pilot sweep config.",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Run only this variant (e.g. 'random' or 'planned') instead of all "
             "variants in the pilot config — lets one Kaggle session cover one "
             "variant's seeds within the 12h session cap instead of all variants.",
    )
    parser.add_argument(
        "--safim-max-samples", type=int, default=100,
        help="SAFIM sample count per model (Base + LoRA-FT) for each seed's "
             "post-training eval (default: 100).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.pilot_config)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN must be set to load FIM variants and push adapters.")
    run(cfg, token=hf_token, only_variant=args.variant, safim_max_samples=args.safim_max_samples)
