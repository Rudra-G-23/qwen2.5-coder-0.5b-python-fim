"""
src/training/safim_eval.py
Execution-based pass@1 evaluation on the Python subset of SAFIM
(Syntax-Aware Fill-in-the-Middle: https://huggingface.co/datasets/gonglinyuan/safim).

Unlike src/evaluate.py's exact_match / edit_similarity (string-level metrics),
this actually runs the completed program against SAFIM's unit tests and checks
stdout, which is what "pass@1" means for this benchmark.

Safety note: this executes model-generated code as a subprocess. Only run it
inside an ephemeral, isolated environment (a fresh Kaggle session, a container,
a throwaway VM) — never on a host with data or credentials worth protecting.
Each run is sandboxed with a wall-clock timeout and CPU/memory rlimits, but
that is not a substitute for process isolation.

Usage (from a Kaggle notebook, after training):
    from src.training.safim_eval import run_safim_evaluation
    run_safim_evaluation(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B",
        adapter_path="/kaggle/working/checkpoints/lora_adapter",
        output_dir="/kaggle/working/results",
        max_samples=100,
        wandb_run=wandb_run,
    )
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.training.evaluate import generate_completion, load_model

MASK_TOKEN = "{{completion}}"


# ── Dataset loading ───────────────────────────────────────────────────────────


def load_safim_python(
    config: str = "block",
    max_samples: int = 100,
    seed: int = 42,
) -> list[dict]:
    """
    Load the Python subset of a SAFIM config and split each eval_prompt into
    (prefix, suffix) around the {{completion}} mask, ready for FIM generation.

    Returns a list of dicts: {task_id, prefix, suffix, eval_prompt, unit_tests}
    """
    from datasets import load_dataset

    ds = load_dataset("gonglinyuan/safim", config, split="test")
    ds = ds.filter(lambda r: r["lang"] == "python")
    ds = ds.shuffle(seed=seed)

    n = min(max_samples, len(ds))
    records: list[dict] = []
    for row in ds.select(range(n)):
        eval_prompt = row["eval_prompt"]
        if MASK_TOKEN not in eval_prompt:
            continue
        prefix, suffix = eval_prompt.split(MASK_TOKEN, 1)
        unit_tests = row["unit_tests"]
        if isinstance(unit_tests, str):
            unit_tests = json.loads(unit_tests)
        records.append(
            {
                "task_id": row["task_id"],
                "prefix": prefix,
                "suffix": suffix,
                "eval_prompt": eval_prompt,
                "unit_tests": unit_tests,
            }
        )
    return records


# ── Sandboxed execution ───────────────────────────────────────────────────────


def _limit_resources() -> None:
    """preexec_fn: cap CPU time and address space for the child process."""
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))  # 1 GB


def _run_program(program: str, stdin_input: str, timeout: float = 5.0) -> str | None:
    """Run `program` as a fresh Python subprocess, feeding stdin_input on stdin.

    Returns stripped stdout, or None on timeout / crash / resource-limit trip.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limit_resources,
        )
    except (subprocess.TimeoutExpired, Exception):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def passes_unit_tests(
    eval_prompt: str, completion: str, unit_tests: list[dict]
) -> bool:
    """A task passes iff the completed program's stdout matches an accepted
    output for every unit test."""
    program = eval_prompt.replace(MASK_TOKEN, completion, 1)
    if not unit_tests:
        return False
    for test in unit_tests:
        stdout = _run_program(program, test["input"])
        if stdout is None:
            return False
        accepted = [o.strip() for o in test["output"]]
        if stdout not in accepted:
            return False
    return True


# ── Per-model pass@1 loop ─────────────────────────────────────────────────────


def evaluate_pass_at_1(
    model,
    tokenizer,
    records: list[dict],
    model_label: str,
) -> list[dict]:
    """Greedy-decode a completion per task and score it against SAFIM's unit tests."""
    results: list[dict] = []
    n = len(records)

    for i, rec in enumerate(records):
        completion = generate_completion(model, tokenizer, rec["prefix"], rec["suffix"])
        passed = passes_unit_tests(rec["eval_prompt"], completion, rec["unit_tests"])
        results.append(
            {
                "model": model_label,
                "task_id": rec["task_id"],
                "pass@1": float(passed),
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == n:
            avg = sum(r["pass@1"] for r in results) / len(results)
            print(f"  [{i + 1:3d}/{n}]  pass@1 = {avg:.3f}")

    return results


# ── Top-level orchestrator ────────────────────────────────────────────────────


def run_safim_evaluation(
    base_model_name: str,
    adapter_path: str,
    output_dir: str = "results",
    config: str = "block",
    max_samples: int = 100,
    seed: int = 42,
    wandb_run: Any | None = None,
) -> pd.DataFrame:
    """
    Execution-based pass@1: Base vs LoRA-FT on SAFIM's Python subset.

    Logs to the same W&B run as src.training.evaluate.run_evaluation, under the
    eval/safim/* namespace, so it shows up side by side with the exact-match /
    edit-similarity metrics from your own held-out FIM dataset.
    """
    records = load_safim_python(config=config, max_samples=max_samples, seed=seed)
    print(f"SAFIM Python tasks ({config}) : {len(records)}")
    print()

    all_results: list[dict] = []

    print("▶  Base model (no adapter)")
    base_model, tokenizer = load_model(base_model_name)
    all_results += evaluate_pass_at_1(base_model, tokenizer, records, "Base")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()

    print("▶  LoRA fine-tuned model")
    lora_model, tokenizer = load_model(base_model_name, adapter_path)
    all_results += evaluate_pass_at_1(lora_model, tokenizer, records, "LoRA-FT")
    del lora_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    summary = df.groupby("model")[["pass@1"]].mean()

    print("\n" + "=" * 50)
    print(f"📊  SAFIM ({config}) pass@1 Summary")
    print("=" * 50)
    print(summary.to_string())
    print("=" * 50)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = f"{output_dir}/safim_metrics.csv"
    summary.to_csv(csv_path)
    print(f"\n✓ SAFIM metrics CSV → {csv_path}")

    if wandb_run is not None:
        try:
            import wandb

            for model_label, row in summary.iterrows():
                prefix = (
                    "eval/safim/base" if model_label == "Base" else "eval/safim/lora_ft"
                )
                wandb_run.log({f"{prefix}/pass_at_1": row["pass@1"]})

            wandb_run.log({"eval/safim/results_table": wandb.Table(dataframe=df)})
        except Exception as exc:
            print(f"⚠  W&B SAFIM metrics log failed: {exc}")

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_safim_evaluation(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B",
        adapter_path="/kaggle/working/checkpoints/lora_adapter",
        output_dir="/kaggle/working/results",
        max_samples=100,
        wandb_run=None,
    )
