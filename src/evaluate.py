"""
src/evaluate.py
Compare the base Qwen2.5-Coder-0.5B against the LoRA fine-tuned adapter
on a held-out FIM dataset.

Outputs:
    results/metrics.csv           — per-model aggregated scores
    results/plots/loss_curve.png  — training loss curve (reads trainer log)
    results/plots/comparison.png  — grouped bar chart: Base vs LoRA-FT

Metrics used:
    exact_match     — strict character equality after stripping whitespace
    edit_similarity — 1 − normalised Levenshtein distance  (0..1, higher = better)

Usage (from Kaggle notebook, after training):
    from src.evaluate import run_evaluation
    run_evaluation(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B",
        adapter_path="/kaggle/working/checkpoints/lora_adapter",
        test_data_path="data/fim_dataset.jsonl",
        output_dir="/kaggle/working/results",
        max_samples=200,
        wandb_run=wandb_run,    # optional — pass the active run from train()
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Qwen2.5-Coder FIM tokens ──────────────────────────────────────────────────
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(
    base_model_name: str,
    adapter_path: str | None = None,
):
    """
    Load the base model, optionally wrapped with a LoRA adapter.

    Args:
        base_model_name: HF model identifier, e.g. "Qwen/Qwen2.5-Coder-0.5B"
        adapter_path:    Local path or HF repo of the LoRA adapter.
                         Pass None to evaluate the raw base model.
    Returns:
        (model, tokenizer)
    """
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=False,
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"  ✓ Adapter loaded from: {adapter_path}")
        FastLanguageModel.for_inference(model)
    except ImportError:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"  ✓ Adapter loaded from: {adapter_path}")
        model.eval()

    return model, tokenizer


def generate_completion(
    model,
    tokenizer,
    prefix: str,
    suffix: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Run greedy FIM inference for one (prefix, suffix) pair.
    Returns the model's predicted middle (decoded, no special tokens).
    """
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                        # greedy → deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt tokens, decode only new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def exact_match(predicted: str, reference: str) -> float:
    """1.0 if stripped strings match exactly, else 0.0."""
    return float(predicted.strip() == reference.strip())


def edit_similarity(predicted: str, reference: str) -> float:
    """
    Character-level edit similarity: 1 − (Levenshtein / max_len).
    Returns a value in [0, 1]; higher is better.
    """
    p = predicted.strip()
    r = reference.strip()
    if not r:
        return 1.0 if not p else 0.0

    m, n = len(p), len(r)
    # Space-optimised single-row DP
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if p[i - 1] == r[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return 1.0 - dp[n] / max(m, n)


# ── Per-model evaluation loop ─────────────────────────────────────────────────

def evaluate_model(
    model,
    tokenizer,
    test_records: list[dict],
    model_label: str,
    max_samples: int = 200,
) -> list[dict]:
    """
    Run inference + metric computation on up to max_samples records.

    Returns a list of per-example result dicts.
    """
    results: list[dict] = []
    n = min(max_samples, len(test_records))

    for i, rec in enumerate(test_records[:n]):
        pred = generate_completion(model, tokenizer, rec["prefix"], rec["suffix"])
        em = exact_match(pred, rec["middle"])
        es = edit_similarity(pred, rec["middle"])
        results.append(
            {
                "model": model_label,
                "exact_match": em,
                "edit_similarity": es,
            }
        )
        if (i + 1) % 20 == 0 or (i + 1) == n:
            avg_em = sum(r["exact_match"] for r in results) / len(results)
            avg_es = sum(r["edit_similarity"] for r in results) / len(results)
            print(f"  [{i + 1:3d}/{n}]  EM = {avg_em:.3f}   ES = {avg_es:.3f}")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(
    df: pd.DataFrame,
    output_dir: str = "results/plots",
    wandb_run: Any | None = None,
) -> str:
    """
    Grouped bar chart: Exact Match and Edit Similarity side-by-side
    for each model variant (Base, LoRA-FT).

    Args:
        df:          DataFrame with columns [model, exact_match, edit_similarity].
        output_dir:  Directory to save the PNG.
        wandb_run:   Active wandb.Run — if provided, logs the plot as a W&B Image.

    Returns:
        Absolute path to the saved PNG.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Base Model vs LoRA Fine-Tuned — Python FIM Evaluation", fontsize=14)

    for ax, metric, title in [
        (axes[0], "exact_match", "Exact Match"),
        (axes[1], "edit_similarity", "Edit Similarity"),
    ]:
        means = df.groupby("model")[metric].mean().reset_index()
        sns.barplot(
            data=means, x="model", y=metric, ax=ax, palette="viridis", width=0.5
        )
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_ylabel(title)
        ax.set_xlabel("")
        # Annotate bar heights
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.01),
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    plt.tight_layout()
    out_path = f"{output_dir}/comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Comparison plot → {out_path}")

    # Log to W&B
    if wandb_run is not None:
        try:
            import wandb
            wandb_run.log({"eval/comparison_chart": wandb.Image(out_path)})
        except Exception as exc:
            print(f"⚠  W&B image log failed: {exc}")

    return out_path


def plot_loss_curve(
    trainer_log_path: str,
    output_dir: str = "results/plots",
    wandb_run: Any | None = None,
) -> str | None:
    """
    Read the trainer_state.json log written by HuggingFace Trainer
    and plot training + validation loss curves.

    Args:
        trainer_log_path: Path to trainer_state.json (e.g. /kaggle/working/checkpoints/trainer_state.json)
        output_dir:       Directory to save the PNG.
        wandb_run:        Active wandb.Run — if provided, logs the plot as a W&B Image.

    Returns:
        Absolute path to the saved PNG, or None if no data.
    """
    log_path = Path(trainer_log_path)
    if not log_path.exists():
        print(f"⚠  Trainer log not found at {trainer_log_path} — skipping loss curve.")
        return None

    with open(log_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    if not train_losses:
        print("⚠  No loss data in trainer log — skipping loss curve.")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_losses, label="Train loss", linewidth=2)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Val loss", linewidth=2, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("LoRA Training — Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = f"{output_dir}/loss_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Loss curve → {out_path}")

    # Log to W&B
    if wandb_run is not None:
        try:
            import wandb
            wandb_run.log({"train/loss_curve": wandb.Image(out_path)})
        except Exception as exc:
            print(f"⚠  W&B image log failed: {exc}")

    return out_path


# ── Top-level orchestrator ────────────────────────────────────────────────────

def run_evaluation(
    base_model_name: str,
    adapter_path: str,
    test_data_path: str,
    output_dir: str = "results",
    max_samples: int = 200,
    trainer_log_path: str | None = None,
    wandb_run: Any | None = None,
) -> pd.DataFrame:
    """
    Full evaluation pipeline:
      1. Load test data.
      2. Evaluate base model.
      3. Evaluate LoRA fine-tuned model.
      4. Save metrics CSV.
      5. Generate comparison bar chart.
      6. (Optional) Plot loss curve from trainer log.
      7. (Optional) Log all results to W&B.

    Args:
        base_model_name:  HF model id for the base model.
        adapter_path:     Local path or HF repo id of the LoRA adapter.
        test_data_path:   Path to the JSONL test file.
        output_dir:       Root directory for outputs (CSV + plots).
        max_samples:      Number of examples to evaluate per model.
        trainer_log_path: Path to trainer_state.json for loss curve plot.
        wandb_run:        Active wandb.Run to log metrics and plots into.
                          Pass None (default) to skip W&B logging.

    Returns:
        Summary DataFrame with aggregated metrics per model.
    """
    # ── Load test records ─────────────────────────────────────────────────────
    test_records: list[dict] = []
    with open(test_data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))

    print(f"Test records available : {len(test_records)}")
    print(f"Evaluating up to       : {max_samples} per model")
    print()

    all_results: list[dict] = []

    # ── Evaluate base model ───────────────────────────────────────────────────
    print("▶  Base model (no adapter)")
    base_model, tokenizer = load_model(base_model_name)
    all_results += evaluate_model(base_model, tokenizer, test_records, "Base", max_samples)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()

    # ── Evaluate LoRA fine-tuned model ────────────────────────────────────────
    print("▶  LoRA fine-tuned model")
    lora_model, tokenizer = load_model(base_model_name, adapter_path)
    all_results += evaluate_model(lora_model, tokenizer, test_records, "LoRA-FT", max_samples)
    del lora_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Aggregate and save metrics ────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    summary = df.groupby("model")[["exact_match", "edit_similarity"]].mean()

    print("\n" + "=" * 50)
    print("📊  Results Summary")
    print("=" * 50)
    print(summary.to_string())
    print("=" * 50)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = f"{output_dir}/metrics.csv"
    summary.to_csv(csv_path)
    print(f"\n✓ Metrics CSV → {csv_path}")

    # Log scalar metrics to W&B
    if wandb_run is not None:
        try:
            import wandb

            for model_label, row in summary.iterrows():
                prefix = "eval/base" if model_label == "Base" else "eval/lora_ft"
                wandb_run.log({
                    f"{prefix}/exact_match": row["exact_match"],
                    f"{prefix}/edit_similarity": row["edit_similarity"],
                })

            # Log the full per-example results as a W&B Table for rich analysis
            results_table = wandb.Table(dataframe=df)
            wandb_run.log({"eval/results_table": results_table})
        except Exception as exc:
            print(f"⚠  W&B metrics log failed: {exc}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots_dir = f"{output_dir}/plots"
    plot_comparison(df, output_dir=plots_dir, wandb_run=wandb_run)
    if trainer_log_path:
        plot_loss_curve(trainer_log_path, output_dir=plots_dir, wandb_run=wandb_run)

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example — adjust paths before running on Kaggle
    run_evaluation(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B",
        adapter_path="/kaggle/working/checkpoints/lora_adapter",
        test_data_path="data/fim_dataset.jsonl",
        output_dir="/kaggle/working/results",
        max_samples=200,
        trainer_log_path="/kaggle/working/checkpoints/trainer_state.json",
        wandb_run=None,  # pass active run to enable W&B logging
    )
