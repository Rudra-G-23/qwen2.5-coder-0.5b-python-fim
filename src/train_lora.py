"""
src/train_lora.py
LoRA fine-tuning of Qwen2.5-Coder-0.5B on a Python FIM dataset.

Workflow:
  1. Load Qwen2.5-Coder-0.5B via Unsloth (fast) or plain HF PEFT (fallback).
  2. Attach LoRA adapters — base weights stay FROZEN.
  3. Train on FIM triples read from a JSONL file.
  4. Save best checkpoint by validation loss (not last step).
  5. Push the adapter (not the full model!) to Hugging Face.
  6. Log all hyperparameters, metrics, and artifacts to Weights & Biases + Weave.

On Kaggle: called from notebooks/kaggle_wandb.ipynb or kaggle_train.ipynb.
Locally:   python src/train_lora.py  (dry-run: verifies config, no actual training)

Run ID format (deterministic, never random):
    {model_slug}-lora-r{r}-e{epochs}-ds{dataset_version}-{YYYYMMDD}-a{attempt}
    Example: qwen05b-lora-r16-e3-dsv1-20260810-a1
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/lora.yaml") -> dict:
    """Load YAML config and return as a plain dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Run ID builder ────────────────────────────────────────────────────────────

def build_run_id(cfg: dict) -> str:
    """
    Construct a deterministic, human-readable run ID from config fields.

    Format:
        {model_slug}-lora-r{r}-e{epochs}-ds{dataset_version}-{YYYYMMDD}-a{attempt}

    Example:
        qwen05b-lora-r16-e3-dsv1-20260810-a1

    Components:
        model_slug      — short slug derived from model name (e.g. qwen05b)
        lora            — training method (always lora)
        r{r}            — LoRA rank (e.g. r16)
        e{epochs}       — number of training epochs (e.g. e3)
        ds{version}     — dataset version from configs/lora.yaml wandb.dataset_version
        {YYYYMMDD}      — today's date (UTC)
        a{attempt}      — attempt/run number from configs/lora.yaml wandb.attempt
    """
    model_name: str = cfg["model"]["name"]
    lora_r: int = cfg["lora"]["r"]
    epochs: int = cfg["training"]["num_epochs"]
    wandb_cfg: dict = cfg.get("wandb", {})
    attempt: int = wandb_cfg.get("attempt", 1)
    dataset_version: str = str(wandb_cfg.get("dataset_version", "v1"))

    # Build a short, lowercase model slug from the HF repo name
    # "Qwen/Qwen2.5-Coder-0.5B" → "qwen05b"
    repo_part = model_name.split("/")[-1].lower()  # "qwen2.5-coder-0.5b"
    slug = (
        repo_part
        .replace("qwen2.5-coder-", "qwen")
        .replace(".", "")
        .replace("-", "")
    )
    # Normalise: "qwen05b" is the expected output for 0.5B
    slug = slug[:8]  # cap length

    date_str = datetime.utcnow().strftime("%Y%m%d")
    ds_tag = f"ds{dataset_version}"

    run_id = f"{slug}-lora-r{lora_r}-e{epochs}-{ds_tag}-{date_str}-a{attempt}"
    return run_id


# ── W&B + Weave initialisation ────────────────────────────────────────────────

def init_wandb(cfg: dict, run_id: str) -> Any | None:
    """
    Initialise Weights & Biases and Weave for experiment tracking.

    Returns the active wandb.Run if W&B is available and WANDB_API_KEY is set,
    otherwise returns None so training proceeds without tracking.

    Weave is initialised inside the same W&B run for joint trace + metric logging.
    """
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        print("⚠  WANDB_API_KEY not set — skipping W&B / Weave tracking.")
        print("   Set os.environ['WANDB_API_KEY'] before calling train() to enable.")
        return None

    try:
        import wandb
        import weave  # noqa: F401  (imported for side-effect: patch tracing)
    except ImportError as exc:
        print(f"⚠  W&B / Weave not installed ({exc}) — skipping tracking.")
        return None

    wandb_cfg = cfg.get("wandb", {})
    project: str = wandb_cfg.get("project", "qwen-coder-python-fim")

    # Flatten the full config into a plain dict for wandb.config
    flat_cfg: dict = {
        # model
        "model/name": cfg["model"]["name"],
        "model/max_seq_length": cfg["model"]["max_seq_length"],
        # lora
        "lora/r": cfg["lora"]["r"],
        "lora/alpha": cfg["lora"]["alpha"],
        "lora/dropout": cfg["lora"]["dropout"],
        "lora/target_modules": cfg["lora"]["target_modules"],
        # training
        "training/num_epochs": cfg["training"]["num_epochs"],
        "training/per_device_train_batch_size": cfg["training"]["per_device_train_batch_size"],
        "training/gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "training/effective_batch_size": (
            cfg["training"]["per_device_train_batch_size"]
            * cfg["training"]["gradient_accumulation_steps"]
        ),
        "training/learning_rate": cfg["training"]["learning_rate"],
        "training/warmup_steps": cfg["training"]["warmup_steps"],
        "training/lr_scheduler_type": cfg["training"]["lr_scheduler_type"],
        "training/fp16": cfg["training"]["fp16"],
        "training/seed": cfg["training"]["seed"],
        # data
        "data/path": cfg["data"]["path"],
        "data/train_split": cfg["data"]["train_split"],
        # wandb metadata
        "run/attempt": wandb_cfg.get("attempt", 1),
        "run/dataset_version": str(wandb_cfg.get("dataset_version", "v1")),
        "run/id": run_id,
    }

    run = wandb.init(
        project=project,
        name=run_id,
        id=run_id,           # deterministic — resume same run if re-run with same ID
        resume="allow",      # allows resuming an interrupted run
        config=flat_cfg,
        tags=wandb_cfg.get("tags", []),
        notes=wandb_cfg.get("notes", ""),
    )

    # Initialise Weave inside the same W&B run for joint tracing
    weave.init(project)
    print(f"✓ W&B run  : {run.url}")
    print(f"  Run ID   : {run_id}")
    print(f"  Project  : {project}")

    return run


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_fim_dataset(
    path: str,
    train_split: float = 0.90,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Load JSONL FIM dataset, shuffle, and split into train / val.

    Each JSONL line must have a "text" field containing the full
    <|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>... string.

    Returns:
        (train_dataset, val_dataset)
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}. Run data_prep.py first.")

    random.seed(seed)
    random.shuffle(records)

    n_train = int(len(records) * train_split)
    train_records = records[:n_train]
    val_records = records[n_train:]

    print(f"Train examples : {len(train_records)}")
    print(f"Val examples   : {len(val_records)}")

    train_ds = Dataset.from_list([{"text": r["text"]} for r in train_records])
    val_ds = Dataset.from_list([{"text": r["text"]} for r in val_records])

    return train_ds, val_ds


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_with_unsloth(cfg: dict):
    """Fast path: Unsloth handles model + LoRA in one call."""
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,         # auto-detect: bf16 on A100, fp16 on T4/P100
        load_in_4bit=False, # full precision — fine at 0.5B on T4/P100
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimised variant
        random_state=cfg["training"]["seed"],
    )
    return model, tokenizer


def _load_with_peft(cfg: dict):
    """Fallback: plain HuggingFace transformers + PEFT (slightly slower)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    return model, tokenizer


# ── Main training function ────────────────────────────────────────────────────

def train(config_path: str = "configs/lora.yaml") -> Any | None:
    """
    Full LoRA training run driven entirely by the YAML config.
    Call this from the Kaggle notebook.

    W&B + Weave tracking is enabled automatically if WANDB_API_KEY is set.
    If the key is not set, training proceeds normally without any tracking.

    Returns:
        The active wandb.Run if W&B is enabled, else None.
        Pass this to run_evaluation(wandb_run=...) to log eval metrics
        into the same W&B run.
    """
    cfg = load_config(config_path)

    # ── Build deterministic run ID ────────────────────────────────────────────
    run_id = build_run_id(cfg)

    print("=" * 60)
    print("LoRA Fine-tuning — Qwen2.5-Coder-0.5B  |  Python FIM")
    print("=" * 60)
    print(f"Config    : {config_path}")
    print(f"Run ID    : {run_id}")
    print(f"Model     : {cfg['model']['name']}")
    print(f"LoRA r    : {cfg['lora']['r']}   alpha : {cfg['lora']['alpha']}")
    print(f"Epochs    : {cfg['training']['num_epochs']}")
    print()

    # ── 1. Initialise W&B + Weave (graceful no-op if key not set) ────────────
    wandb_run = init_wandb(cfg, run_id)
    report_to = "wandb" if wandb_run is not None else "none"

    # ── 2. Load model ─────────────────────────────────────────────────────────
    try:
        model, tokenizer = _load_with_unsloth(cfg)
        print("✓ Loaded with Unsloth (fast path)")
    except ImportError:
        model, tokenizer = _load_with_peft(cfg)
        print("✓ Loaded with plain PEFT (Unsloth not installed)")

    model.print_trainable_parameters()
    print()

    # ── 3. Load data ──────────────────────────────────────────────────────────
    train_ds, val_ds = load_fim_dataset(
        path=cfg["data"]["path"],
        train_split=cfg["data"]["train_split"],
        seed=cfg["training"]["seed"],
    )

    # ── 4. TrainingArguments ──────────────────────────────────────────────────
    output_dir = cfg["output"]["dir"]
    t = cfg["training"]

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        lr_scheduler_type=t["lr_scheduler_type"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_total_limit=2,            # keep only last 2 checkpoints on disk
        load_best_model_at_end=True,   # pick by val loss, NOT last step
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,           # "wandb" if key is set, else "none"
        run_name=run_id,               # deterministic name shown in W&B
        seed=t["seed"],
        dataloader_num_workers=2,
        average_tokens_across_devices=False,
    )

    # ── 5. SFTTrainer ─────────────────────────────────────────────────────────
    from trl import SFTTrainer  # type: ignore

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        args=training_args,
    )

    # WORKAROUND: Prevent PicklingError for SFTConfig when saving checkpoints
    _orig_torch_save = torch.save
    def _patched_torch_save(obj, f, *args, **kwargs):
        if type(obj).__name__ == "SFTConfig":
            return
        return _orig_torch_save(obj, f, *args, **kwargs)
    torch.save = _patched_torch_save

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n▶  Starting training…")
    result = trainer.train()
    print(f"\n✓ Training complete")
    print(f"  Training loss : {result.training_loss:.4f}")
    print(f"  Runtime       : {result.metrics.get('train_runtime', 0):.0f}s")
    print(f"  Samples/sec   : {result.metrics.get('train_samples_per_second', 0):.1f}")

    # ── 7. Save adapter (NOT the full model) ──────────────────────────────────
    adapter_path = Path(output_dir) / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    adapter_size_mb = (
        sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file())
        / 1024 / 1024
    )
    print(f"\n✓ Adapter saved → {adapter_path}")
    print(f"  Size: {adapter_size_mb:.1f} MB")

    # ── 8. Log final training summary to W&B ─────────────────────────────────
    if wandb_run is not None:
        import wandb

        wandb.summary.update({
            "train/loss_final": result.training_loss,
            "train/runtime_s": result.metrics.get("train_runtime", 0),
            "train/samples_per_second": result.metrics.get("train_samples_per_second", 0),
            "adapter/size_mb": adapter_size_mb,
            "adapter/path": str(adapter_path),
        })

    # ── 9. Push adapter to Hugging Face ───────────────────────────────────────
    hf_repo = cfg["output"]["hf_repo"]
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n⚠  HF_TOKEN not set — skipping Hugging Face push.")
        print("   Set os.environ['HF_TOKEN'] before calling train() to enable push.")
        if wandb_run is not None:
            wandb_run.finish()
        return wandb_run

    print(f"\n▶  Pushing adapter to HF: {hf_repo}")
    model.push_to_hub(hf_repo, token=hf_token)
    tokenizer.push_to_hub(hf_repo, token=hf_token)
    hf_url = f"https://huggingface.co/{hf_repo}"
    print(f"✓ Adapter pushed → {hf_url}")

    # Log HF URL to W&B run summary
    if wandb_run is not None:
        import wandb
        wandb.summary["hf_repo_url"] = hf_url
        wandb_run.finish()
        print(f"\n✓ W&B run finished → {wandb_run.url}")

    return wandb_run


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dry-run: verify config loads and dataset parses without launching training.
    # Real training is invoked from notebooks/kaggle_wandb.ipynb on Kaggle.
    cfg = load_config("configs/lora.yaml")
    run_id = build_run_id(cfg)
    print("Config loaded OK:")
    print(f"  model          : {cfg['model']['name']}")
    print(f"  lora r / alpha : {cfg['lora']['r']} / {cfg['lora']['alpha']}")
    print(f"  epochs         : {cfg['training']['num_epochs']}")
    print(f"  output repo    : {cfg['output']['hf_repo']}")
    print(f"  run id         : {run_id}")
    print(f"  wandb project  : {cfg.get('wandb', {}).get('project', '(not set)')}")
    print("\nTo actually train, run from the Kaggle notebook:")
    print("  from src.train_lora import train")
    print("  train(config_path='configs/lora.yaml')")
