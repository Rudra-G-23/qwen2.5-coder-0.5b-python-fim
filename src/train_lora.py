"""
src/train_lora.py
LoRA fine-tuning of Qwen2.5-Coder-0.5B on a Python FIM dataset.

Workflow:
  1. Load Qwen2.5-Coder-0.5B via Unsloth (fast) or plain HF PEFT (fallback).
  2. Attach LoRA adapters — base weights stay FROZEN.
  3. Train on FIM triples read from a JSONL file.
  4. Save best checkpoint by validation loss (not last step).
  5. Push the adapter (not the full model!) to Hugging Face.

On Kaggle: called from notebooks/kaggle_train.ipynb — do NOT run this directly.
Locally:   python src/train_lora.py  (for debugging config, not actual training)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

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

    # HuggingFace Dataset needs list-of-dicts with the correct field
    train_ds = Dataset.from_list([{"text": r["text"]} for r in train_records])
    val_ds = Dataset.from_list([{"text": r["text"]} for r in val_records])

    return train_ds, val_ds


# ── Model loader ──────────────────────────────────────────────────────────────

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

def train(config_path: str = "configs/lora.yaml") -> None:
    """
    Full LoRA training run driven entirely by the YAML config.
    Call this from the Kaggle notebook.
    """
    cfg = load_config(config_path)

    print("=" * 60)
    print("LoRA Fine-tuning — Qwen2.5-Coder-0.5B  |  Python FIM")
    print("=" * 60)
    print(f"Config : {config_path}")
    print(f"Model  : {cfg['model']['name']}")
    print(f"LoRA r : {cfg['lora']['r']}   alpha : {cfg['lora']['alpha']}")
    print(f"Epochs : {cfg['training']['num_epochs']}")
    print()

    # ── 1. Load model ────────────────────────────────────────────────────────
    try:
        model, tokenizer = _load_with_unsloth(cfg)
        print("✓ Loaded with Unsloth (fast path)")
    except ImportError:
        model, tokenizer = _load_with_peft(cfg)
        print("✓ Loaded with plain PEFT (Unsloth not installed)")

    model.print_trainable_parameters()
    print()

    # ── 2. Load data ─────────────────────────────────────────────────────────
    train_ds, val_ds = load_fim_dataset(
        path=cfg["data"]["path"],
        train_split=cfg["data"]["train_split"],
        seed=cfg["training"]["seed"],
    )

    # ── 3. TrainingArguments ─────────────────────────────────────────────────
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
        report_to="none",              # change to "wandb" if you want W&B
        seed=t["seed"],
        dataloader_num_workers=2,
        average_tokens_across_devices=False,
    )

    # ── 4. SFTTrainer ────────────────────────────────────────────────────────
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

    # ── 5. Train ─────────────────────────────────────────────────────────────
    print("\n▶  Starting training…")
    result = trainer.train()
    print(f"\n✓ Training complete")
    print(f"  Training loss : {result.training_loss:.4f}")
    print(f"  Runtime       : {result.metrics.get('train_runtime', 0):.0f}s")
    print(f"  Samples/sec   : {result.metrics.get('train_samples_per_second', 0):.1f}")

    # ── 6. Save adapter (NOT the full model) ─────────────────────────────────
    adapter_path = Path(output_dir) / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✓ Adapter saved → {adapter_path}")
    print(f"  Size: {sum(f.stat().st_size for f in adapter_path.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")

    # ── 7. Push adapter to Hugging Face ──────────────────────────────────────
    hf_repo = cfg["output"]["hf_repo"]
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n⚠  HF_TOKEN not set — skipping Hugging Face push.")
        print("   Set os.environ['HF_TOKEN'] before calling train() to enable push.")
        return

    print(f"\n▶  Pushing adapter to HF: {hf_repo}")
    model.push_to_hub(hf_repo, token=hf_token)
    tokenizer.push_to_hub(hf_repo, token=hf_token)
    print(f"✓ Adapter pushed → https://huggingface.co/{hf_repo}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dry-run: verify config loads and dataset parses without launching training.
    # Real training is invoked from notebooks/kaggle_train.ipynb on Kaggle.
    cfg = load_config("configs/lora.yaml")
    print("Config loaded OK:")
    print(f"  model          : {cfg['model']['name']}")
    print(f"  lora r / alpha : {cfg['lora']['r']} / {cfg['lora']['alpha']}")
    print(f"  epochs         : {cfg['training']['num_epochs']}")
    print(f"  output repo    : {cfg['output']['hf_repo']}")
    print("\nTo actually train, run from the Kaggle notebook:")
    print("  from src.train_lora import train")
    print("  train(config_path='configs/lora.yaml')")
