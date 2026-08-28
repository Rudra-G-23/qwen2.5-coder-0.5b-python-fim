"""
src/training/evaluate.py
Compare the base Qwen2.5-Coder-0.5B against the LoRA fine-tuned adapter
on a held-out FIM dataset.

Outputs:
    results/metrics.csv                 — per-model aggregated scores
    results/plots/loss_curve.png        — training loss curve (reads trainer log)
    results/plots/lr_curve.png          — learning-rate schedule (reads trainer log)
    results/plots/perplexity_curve.png  — exp(loss) curve (reads trainer log)
    results/plots/comparison.png        — grouped bar chart: Base vs LoRA-FT

Metrics used:
    exact_match     — strict character equality after stripping whitespace
    edit_similarity — 1 − normalised Levenshtein distance  (0..1, higher = better)

Usage (from Kaggle notebook, after training):
    from src.training.evaluate import run_evaluation
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
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.report_tables import log_plots_artifact

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
            do_sample=False,  # greedy → deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt tokens, decode only new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
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
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
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


def plot_lr_curve(
    trainer_log_path: str,
    output_dir: str = "results/plots",
    wandb_run: Any | None = None,
) -> str | None:
    """
    Read the trainer_state.json log written by HuggingFace Trainer and plot
    the learning-rate schedule actually used during training. HF's own
    Trainer logs this to W&B automatically when report_to="wandb", but it
    was never saved as a repo artifact / paper-ready figure — this makes it
    one, mirroring plot_loss_curve's shape.
    """
    log_path = Path(trainer_log_path)
    if not log_path.exists():
        print(f"⚠  Trainer log not found at {trainer_log_path} — skipping LR curve.")
        return None

    with open(log_path) as f:
        state = json.load(f)

    steps, lrs = [], []
    for entry in state.get("log_history", []):
        if "learning_rate" in entry:
            steps.append(entry["step"])
            lrs.append(entry["learning_rate"])

    if not lrs:
        print("⚠  No learning_rate data in trainer log — skipping LR curve.")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(steps, lrs, linewidth=2, color="#C44E52")
    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.title("LoRA Training — Learning-Rate Schedule")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = f"{output_dir}/lr_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ LR curve → {out_path}")

    if wandb_run is not None:
        try:
            import wandb

            wandb_run.log({"train/lr_curve": wandb.Image(out_path)})
        except Exception as exc:
            print(f"⚠  W&B image log failed: {exc}")

    return out_path


def plot_perplexity_curve(
    trainer_log_path: str,
    output_dir: str = "results/plots",
    wandb_run: Any | None = None,
) -> str | None:
    """
    Perplexity = exp(loss), derived from the same trainer_state.json log
    plot_loss_curve reads. Conventional to report alongside loss for LM
    fine-tuning papers even though it carries no information loss doesn't.
    """
    import math

    log_path = Path(trainer_log_path)
    if not log_path.exists():
        print(f"⚠  Trainer log not found at {trainer_log_path} — skipping perplexity curve.")
        return None

    with open(log_path) as f:
        state = json.load(f)

    train_steps, train_ppl = [], []
    eval_steps, eval_ppl = [], []
    for entry in state.get("log_history", []):
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_ppl.append(math.exp(entry["loss"]))
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_ppl.append(math.exp(entry["eval_loss"]))

    if not train_ppl:
        print("⚠  No loss data in trainer log — skipping perplexity curve.")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_ppl, label="Train perplexity", linewidth=2)
    if eval_ppl:
        plt.plot(eval_steps, eval_ppl, label="Val perplexity", linewidth=2, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.title("LoRA Training — Perplexity Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = f"{output_dir}/perplexity_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Perplexity curve → {out_path}")

    if wandb_run is not None:
        try:
            import wandb

            wandb_run.log({"train/perplexity_curve": wandb.Image(out_path)})
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
    all_results += evaluate_model(
        base_model, tokenizer, test_records, "Base", max_samples
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()

    # ── Evaluate LoRA fine-tuned model ────────────────────────────────────────
    print("▶  LoRA fine-tuned model")
    lora_model, tokenizer = load_model(base_model_name, adapter_path)
    all_results += evaluate_model(
        lora_model, tokenizer, test_records, "LoRA-FT", max_samples
    )
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
                wandb_run.log(
                    {
                        f"{prefix}/exact_match": row["exact_match"],
                        f"{prefix}/edit_similarity": row["edit_similarity"],
                    }
                )

            # Log the full per-example results as a W&B Table for rich analysis
            results_table = wandb.Table(dataframe=df)
            wandb_run.log({"eval/results_table": results_table})
        except Exception as exc:
            print(f"⚠  W&B metrics log failed: {exc}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots_dir = f"{output_dir}/plots"
    plot_paths = [plot_comparison(df, output_dir=plots_dir, wandb_run=wandb_run)]
    if trainer_log_path:
        plot_paths.append(plot_loss_curve(trainer_log_path, output_dir=plots_dir, wandb_run=wandb_run))
        plot_paths.append(plot_lr_curve(trainer_log_path, output_dir=plots_dir, wandb_run=wandb_run))
        plot_paths.append(
            plot_perplexity_curve(trainer_log_path, output_dir=plots_dir, wandb_run=wandb_run)
        )

    log_plots_artifact(wandb_run, [p for p in plot_paths if p], csv_path, name="eval-plots")

    return summary


# ── N-way, per-bucket, resumable evaluation (safim_eval_1000) ─────────────────
#
# Below extends the Base-vs-one-LoRA flow above to an arbitrary N models
# (Base, Random-LoRA, Distributed-LoRA, ...) on a fixed FIM eval-set parquet
# (experiment/safim_eval_1000/), broken down by fim_type bucket, with genuine
# crash-resumability — 3 models x 1000 unbatched greedy-decode generations is
# the actual long-running, interruption-prone step in that pipeline, unlike
# Phase A's fast in-memory dataset build. None of run_evaluation/
# evaluate_model/plot_comparison above are modified; other callers of those
# are unaffected.


def download_adapter_snapshot(hf_repo: str, subfolder: str, token: str | None = None) -> str:
    """snapshot_download the given adapter subfolder from an HF *model* repo
    to a local dir, returning the local path load_model()'s adapter_path
    already expects (it only accepts a local directory today). Same
    snapshot_download+subfolder pattern scripts/run_pilot_experiment.py
    already uses once for its own resume path."""
    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(
        repo_id=hf_repo, repo_type="model", allow_patterns=f"{subfolder}/*", token=token
    )
    return str(Path(snapshot_dir) / subfolder)


def load_fim_eval_records(parquet_path: str) -> list[dict]:
    """Read a FIM eval-set parquet (e.g. experiment/safim_eval_1000/data/
    fim_safim_eval_1000.parquet) into a list of record dicts, keeping every
    column — including fim_type/task_id, which evaluate_model_by_type needs
    alongside prefix/suffix/middle. Distinct from run_evaluation's JSONL
    loader above (that format has no fim_type/task_id and stays untouched)."""
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path).to_pylist()


def _read_results_jsonl(results_path: str) -> list[dict]:
    path = Path(results_path)
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_already_scored(
    results_path: str,
    model_label: str,
    hf_checkpoint_repo: str | None = None,
    hf_checkpoint_folder: str = "experiment/safim_eval_1000_results",
    token: str | None = None,
) -> set[str]:
    """Already-scored task_ids for `model_label`. Reads `results_path`
    directly if it already exists locally (mid-session resume after a
    transient error); otherwise, if `hf_checkpoint_repo` is given, tries to
    restore the HF-mirrored checkpoint into `results_path` first (the
    cross-session case — e.g. a whole Kaggle session died and the local disk
    didn't survive). Any failure to find a checkpoint anywhere (first-ever
    run for this model) is treated as "nothing scored yet", not an error —
    mirrors the fail-open convention already used for existence checks
    elsewhere in this repo (src.fim.experiment_io.folder_metadata_exists,
    scripts.run_pilot_experiment._subfolder_already_pushed)."""
    path = Path(results_path)
    if not path.exists() and hf_checkpoint_repo:
        try:
            from huggingface_hub import hf_hub_download

            remote_path = hf_hub_download(
                repo_id=hf_checkpoint_repo,
                filename=f"{hf_checkpoint_folder}/{model_label}.jsonl",
                repo_type="dataset",
                token=token,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(remote_path, path)
            print(f"  ↻ Restored {model_label}'s checkpoint from {hf_checkpoint_repo}/{hf_checkpoint_folder}")
        except Exception as exc:
            print(f"  … no existing checkpoint for {model_label} ({exc}) — starting fresh.")

    records = _read_results_jsonl(str(path))
    return {r["task_id"] for r in records if r.get("model") == model_label}


def _append_result(results_path: str, result: dict) -> None:
    """Append one JSON line immediately — never buffer results in memory
    until the end, so a mid-run crash loses at most the current record."""
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _mirror_checkpoint_to_hf(
    results_path: str, hf_repo: str, hf_folder: str, model_label: str, token: str | None
) -> None:
    """Upload the current local results file to `{hf_folder}/{model_label}.jsonl`
    in hf_repo — the durable, cross-session checkpoint. `hf_folder` is always
    a sibling of the frozen experiment/safim_eval_1000/ folder, never written
    inside it, so eval results can never touch that folder's idempotency
    guarantee (src.fim.experiment_io.folder_metadata_exists)."""
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        path_or_fileobj=results_path,
        path_in_repo=f"{hf_folder}/{model_label}.jsonl",
        repo_id=hf_repo,
        repo_type="dataset",
        token=token,
    )


def evaluate_model_by_type(
    model,
    tokenizer,
    test_records: list[dict],
    model_label: str,
    results_path: str,
    hf_checkpoint_repo: str | None = None,
    hf_checkpoint_folder: str = "experiment/safim_eval_1000_results",
    checkpoint_every_n_records: int = 50,
    max_samples: int | None = None,
    token: str | None = None,
    wandb_run: Any | None = None,
) -> list[dict]:
    """
    Like evaluate_model, but: carries task_id/fim_type through each result,
    resumes from already-scored task_ids (local file, falling back to the
    HF-mirrored checkpoint) instead of re-generating them, appends each new
    result to `results_path` immediately, and re-uploads `results_path` to
    `hf_checkpoint_repo` every `checkpoint_every_n_records` (and always at
    the end) so a whole-session crash loses at most that many records for
    this model, not the model's entire progress.

    `test_records` each need: prefix, suffix, middle, fim_type, task_id
    (src.training.evaluate.load_fim_eval_records's output shape).
    `max_samples=None` runs every record; otherwise runs only the first
    `max_samples` records (by test_records order) — useful for a fast
    calibration pass to measure real per-sample throughput before committing
    to a full run.

    Returns every scored result for this model (already-scored + newly-
    scored), read back from `results_path`.
    """
    already_scored = _load_already_scored(
        results_path, model_label, hf_checkpoint_repo, hf_checkpoint_folder, token
    )
    if already_scored:
        print(f"  ↻ Resuming {model_label}: {len(already_scored)} record(s) already scored — skipping them.")

    candidate_records = test_records if max_samples is None else test_records[:max_samples]
    records_to_run = [r for r in candidate_records if r["task_id"] not in already_scored]

    n = len(records_to_run)
    start = time.monotonic()
    since_last_mirror = 0

    for i, rec in enumerate(records_to_run):
        completion = generate_completion(model, tokenizer, rec["prefix"], rec["suffix"])
        result = {
            "model": model_label,
            "task_id": rec["task_id"],
            "fim_type": rec["fim_type"],
            "exact_match": exact_match(completion, rec["middle"]),
            "edit_similarity": edit_similarity(completion, rec["middle"]),
        }
        _append_result(results_path, result)
        since_last_mirror += 1

        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.monotonic() - start
            per_sample = elapsed / (i + 1)
            eta_s = per_sample * (n - (i + 1))
            print(
                f"  [{i + 1:4d}/{n}]  {model_label}  "
                f"({per_sample:.1f}s/sample, ETA {eta_s / 60:.1f} min)"
            )
            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            f"eval/safim_eval_1000/{model_label}/completed": len(already_scored) + i + 1,
                            f"eval/safim_eval_1000/{model_label}/sec_per_sample": per_sample,
                        }
                    )
                except Exception as exc:
                    print(f"  ⚠ W&B live progress log failed: {exc}")

        if hf_checkpoint_repo and since_last_mirror >= checkpoint_every_n_records:
            _mirror_checkpoint_to_hf(results_path, hf_checkpoint_repo, hf_checkpoint_folder, model_label, token)
            since_last_mirror = 0

    if hf_checkpoint_repo and since_last_mirror > 0:
        _mirror_checkpoint_to_hf(results_path, hf_checkpoint_repo, hf_checkpoint_folder, model_label, token)

    return [r for r in _read_results_jsonl(results_path) if r.get("model") == model_label]


def plot_comparison_by_bucket(
    summary_df: pd.DataFrame,
    metric: str,
    output_dir: str = "results/plots",
    wandb_run: Any | None = None,
) -> str:
    """
    Grouped bar chart: `metric` ("exact_match" or "edit_similarity"), one
    bar-group per fim_type bucket, one bar per model within each group —
    structurally mirrors src.training.safim_eval.plot_safim_by_type's
    grouped-bar layout, adapted to this module's metrics and fim_type
    grouping key instead of SAFIM's pass@1/config.

    `summary_df` needs columns: fim_type, model, <metric>.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fim_types = sorted(summary_df["fim_type"].unique())
    models = sorted(summary_df["model"].unique())
    x = range(len(fim_types))
    width = 0.8 / max(len(models), 1)

    _fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_label in enumerate(models):
        vals = [
            summary_df[
                (summary_df["fim_type"] == fim_type) & (summary_df["model"] == model_label)
            ][metric].sum()
            for fim_type in fim_types
        ]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width, label=model_label)

    ax.set_xticks(list(x))
    ax.set_xticklabels(fim_types, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric.replace('_', ' ').title()} by FIM bucket — safim_eval_1000", fontsize=12)
    ax.legend()
    plt.tight_layout()

    out_path = f"{output_dir}/{metric}_by_bucket.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {metric} by-bucket plot → {out_path}")

    if wandb_run is not None:
        try:
            import wandb

            wandb_run.log({f"eval/safim_eval_1000/{metric}_by_bucket_chart": wandb.Image(out_path)})
        except Exception as exc:
            print(f"⚠  W&B image log failed: {exc}")

    return out_path


def run_evaluation_nway_by_type(
    base_model_name: str,
    models: list[dict[str, Any]],
    test_records: list[dict],
    output_dir: str = "results/safim_eval_1000",
    max_samples: int | None = None,
    hf_checkpoint_repo: str | None = None,
    hf_checkpoint_folder: str = "experiment/safim_eval_1000_results",
    checkpoint_every_n_records: int = 50,
    token: str | None = None,
    wandb_run: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    N-way exact_match/edit_similarity comparison on a fixed test_records set
    (here: Base, Random-LoRA, Distributed-LoRA on safim_eval_1000), broken
    down by fim_type bucket.

    `models`: list of {"label": str, "adapter_path": str | None} — already-
    resolved LOCAL adapter directories (or None for the base model with no
    adapter). Resolving an HF repo + subfolder to a local dir is the
    caller's job (see download_adapter_snapshot / scripts/run_safim_eval_1000.py)
    — keeps this function testable without network access.

    For each model: load_model() once, evaluate_model_by_type() (resumable
    per-model), then del model + empty_cache before the next — same memory
    pattern as run_evaluation above.

    Returns (aggregate_summary, by_bucket_summary):
      - aggregate_summary: groupby("model") mean exact_match/edit_similarity.
        Plotted via plot_comparison() UNCHANGED — it already groups by
        "model" generically, no new code needed there.
      - by_bucket_summary: groupby(["fim_type","model"]) means, written to
        {output_dir}/metrics_by_bucket.csv and plotted via
        plot_comparison_by_bucket() once per metric.
    """
    all_results: list[dict] = []

    for model_cfg in models:
        model_label = model_cfg["label"]
        adapter_path = model_cfg.get("adapter_path")
        print(f"\n▶  {model_label}" + (f"  (adapter: {adapter_path})" if adapter_path else "  (no adapter)"))

        model, tokenizer = load_model(base_model_name, adapter_path)
        results_path = f"{output_dir}/{model_label}_results.jsonl"
        all_results += evaluate_model_by_type(
            model,
            tokenizer,
            test_records,
            model_label,
            results_path,
            hf_checkpoint_repo=hf_checkpoint_repo,
            hf_checkpoint_folder=hf_checkpoint_folder,
            checkpoint_every_n_records=checkpoint_every_n_records,
            max_samples=max_samples,
            token=token,
            wandb_run=wandb_run,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    aggregate_summary = df.groupby("model")[["exact_match", "edit_similarity"]].mean()
    by_bucket_summary = (
        df.groupby(["fim_type", "model"])[["exact_match", "edit_similarity"]].mean().reset_index()
    )

    print("\n" + "=" * 50)
    print("📊  safim_eval_1000 Results Summary")
    print("=" * 50)
    print(aggregate_summary.to_string())
    print("=" * 50)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_csv = f"{output_dir}/metrics.csv"
    by_bucket_csv = f"{output_dir}/metrics_by_bucket.csv"
    aggregate_summary.to_csv(metrics_csv)
    by_bucket_summary.to_csv(by_bucket_csv, index=False)
    print(f"\n✓ Metrics CSVs → {metrics_csv}, {by_bucket_csv}")

    if wandb_run is not None:
        try:
            import wandb

            for model_label, row in aggregate_summary.iterrows():
                wandb_run.log(
                    {
                        f"eval/safim_eval_1000/{model_label}/exact_match": row["exact_match"],
                        f"eval/safim_eval_1000/{model_label}/edit_similarity": row["edit_similarity"],
                    }
                )
            wandb_run.log({"eval/safim_eval_1000/by_bucket_table": wandb.Table(dataframe=by_bucket_summary)})
        except Exception as exc:
            print(f"⚠  W&B metrics log failed: {exc}")

    plots_dir = f"{output_dir}/plots"
    plot_paths = [
        plot_comparison(df, output_dir=plots_dir, wandb_run=wandb_run),
        plot_comparison_by_bucket(by_bucket_summary, "exact_match", output_dir=plots_dir, wandb_run=wandb_run),
        plot_comparison_by_bucket(by_bucket_summary, "edit_similarity", output_dir=plots_dir, wandb_run=wandb_run),
    ]
    log_plots_artifact(wandb_run, plot_paths, metrics_csv, by_bucket_csv, name="safim-eval-1000-plots")

    return aggregate_summary, by_bucket_summary


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
