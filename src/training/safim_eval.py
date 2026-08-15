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
import statistics
import subprocess
import sys
import time
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
    wandb_run: Any | None = None,
    log_every: int = 10,
) -> list[dict]:
    """Greedy-decode a completion per task and score it against SAFIM's unit
    tests. Logs running pass@1 to stdout every `log_every` samples, and to
    the same W&B run (if given) so progress is visible on the dashboard
    without watching the notebook — useful since a 100-sample run per model
    can take a while and previously gave no live signal beyond stdout."""
    results: list[dict] = []
    n = len(records)
    start = time.monotonic()

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
        if (i + 1) % log_every == 0 or (i + 1) == n:
            avg = sum(r["pass@1"] for r in results) / len(results)
            elapsed = time.monotonic() - start
            per_sample = elapsed / (i + 1)
            eta_s = per_sample * (n - (i + 1))
            print(
                f"  [{i + 1:3d}/{n}]  pass@1 = {avg:.3f}  "
                f"({per_sample:.1f}s/sample, ETA {eta_s / 60:.1f} min)"
            )
            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            f"eval/safim/{model_label}/running_pass_at_1": avg,
                            f"eval/safim/{model_label}/completed": i + 1,
                            f"eval/safim/{model_label}/sec_per_sample": per_sample,
                        }
                    )
                except Exception as exc:
                    print(f"  ⚠ W&B live progress log failed: {exc}")

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
    all_results += evaluate_pass_at_1(base_model, tokenizer, records, "Base", wandb_run=wandb_run)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print()

    print("▶  LoRA fine-tuned model")
    lora_model, tokenizer = load_model(base_model_name, adapter_path)
    all_results += evaluate_pass_at_1(lora_model, tokenizer, records, "LoRA-FT", wandb_run=wandb_run)
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


# ── Pilot comparison aggregation ────────────────────────────────────────────
# Stage 1 §8: aggregate the per-(variant, seed) pass@1 numbers written by
# run_safim_evaluation (via scripts/run_pilot_experiment.py) into a
# random-vs-planned comparison. Kept here, not in scripts/, so it's testable
# without a GPU — the same split as write_filter_report living in
# src/curation/checkpoint.py rather than scripts/build_sample.py.


def load_seed_pass_at_1(
    output_dir: str, model_label: str = "LoRA-FT"
) -> float | None:
    """Read one seed's safim_metrics.csv (written by run_safim_evaluation)
    and return that model label's pass@1, or None if the run hasn't
    completed / produced a file yet — a missing run is reported, not
    treated as a zero score."""
    csv_path = Path(output_dir) / "safim_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col="model")
    if model_label not in df.index:
        return None
    return float(df.loc[model_label, "pass@1"])


def aggregate_pilot_results(
    pilot_cfg: dict[str, Any], results_dir: str = "results/pilot"
) -> dict[str, Any]:
    """
    Collect LoRA-FT pass@1 across every (variant, seed) in pilot_cfg,
    per-variant mean/min/max, and a plain-language verdict on whether the
    random-vs-planned gap looks bigger than seed-to-seed spread — the bar
    data-stage-1.md §8 sets ("a consistent, non-noise gap, not a single-run
    difference"). With only 2-3 seeds this is deliberately NOT a formal
    significance test (too few samples for one to mean anything) — it's a
    range-overlap check, reported as directional evidence only.
    """
    seeds = pilot_cfg["seeds"]
    per_variant: dict[str, Any] = {}

    for variant_name in pilot_cfg["variants"]:
        scores: dict[int, float | None] = {}
        for seed in seeds:
            scores[seed] = load_seed_pass_at_1(f"{results_dir}/{variant_name}_seed{seed}")

        found = [v for v in scores.values() if v is not None]
        missing = [seed for seed, v in scores.items() if v is None]
        per_variant[variant_name] = {
            "scores_by_seed": scores,
            "missing_seeds": missing,
            "mean": statistics.mean(found) if found else None,
            "min": min(found) if found else None,
            "max": max(found) if found else None,
        }

    complete = {
        name: stats for name, stats in per_variant.items() if not stats["missing_seeds"]
    }
    if len(complete) < len(per_variant):
        verdict = "incomplete: not every variant has all seeds' results yet — run the missing pilot notebooks first."
    elif len(complete) < 2:
        verdict = "not enough variants with complete results to compare yet (need at least 2)."
    else:
        ranges = {name: (stats["min"], stats["max"]) for name, stats in complete.items()}
        (name_a, (min_a, max_a)), (name_b, (min_b, max_b)) = list(ranges.items())[:2]
        overlaps = max_a >= min_b and max_b >= min_a
        if overlaps:
            verdict = (
                f"'{name_a}' and '{name_b}' seed-score ranges overlap — the gap is not "
                "clearly bigger than seed-to-seed noise yet. Treat as inconclusive, per "
                "data-stage-1.md §8; more seeds or more data would be needed before "
                "picking a winner."
            )
        else:
            leader = name_a if min_a > max_b else name_b
            verdict = (
                f"'{leader}' seed-score range does not overlap with the other variant's — "
                "a real gap, not just seed noise, on this pilot sample. Still preliminary "
                "(data-stage-1.md §8) until re-checked at full-corpus scale (data-stage-2.md)."
            )

    return {"per_variant": per_variant, "verdict": verdict}


def write_pilot_comparison_report(
    pilot_cfg: dict[str, Any],
    results_dir: str = "results/pilot",
    out_path: str = "reports/pilot_comparison.json",
) -> Path:
    report = aggregate_pilot_results(pilot_cfg, results_dir)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_safim_evaluation(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B",
        adapter_path="/kaggle/working/checkpoints/lora_adapter",
        output_dir="/kaggle/working/results",
        max_samples=100,
        wandb_run=None,
    )
