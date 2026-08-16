# Graphs, W&B tracking, and experiment-setup reporting — audit

Answers the three questions asked on 2026-08-16. Status reflects the repo
after this pass's fixes (see "Fixed in this pass" at the bottom).

## 1. What graphs exist

| Graph | Function | Output file | Called from |
|---|---|---|---|
| Base vs LoRA-FT metrics bar chart | `plot_comparison` (`src/training/evaluate.py`) | `results/plots/comparison.png` | `run_evaluation` |
| Training loss curve | `plot_loss_curve` (`evaluate.py`) | `results/plots/loss_curve.png` | `run_evaluation` (if `trainer_log_path`) |
| Learning-rate schedule | `plot_lr_curve` (`evaluate.py`) | `results/plots/lr_curve.png` | `run_evaluation` (if `trainer_log_path`) |
| Perplexity curve (train/val) | `plot_perplexity_curve` (`evaluate.py`) | `results/plots/perplexity_curve.png` | `run_evaluation` (if `trainer_log_path`) |
| SAFIM pass@1, Base vs LoRA-FT | `plot_safim_comparison` (`src/training/safim_eval.py`) | `results/plots/safim_comparison.png` | `run_safim_evaluation` |
| SAFIM pass@1 by task type (control/api/block) | `plot_safim_by_type` (`safim_eval.py`) | `results/plots/safim_by_type.png` | `run_safim_evaluation_by_type` — **not wired into `scripts/run_pilot_experiment.py` or any notebook**; call it manually if you want the span-type breakdown (it re-loads both models per config, ~3x eval cost) |
| Pilot variant comparison (random vs. planned, seed error bars) | `plot_pilot_comparison` (`safim_eval.py`) | `reports/pilot_comparison.png` | `write_pilot_comparison_report`, called from `scripts/analyze_pilot_results.py` (offline, post-hoc — no live W&B run at that point) |

Non-graph reporting artifacts in the same family:
- `render_hyperparameter_table` / `write_hyperparameter_table` (`src/training/report_tables.py`) — Markdown "Experimental Setup" table.
- `render_compute_cost_table` / `write_compute_cost_table` (`report_tables.py`) — Markdown run-cost table, fed by `fetch_wandb_run_durations`.
- `render_seed_variance_table` / `write_seed_variance_table` (`safim_eval.py`) — mean±std/min/max pilot seed table.

## 2. Shown in notebook, stored in W&B as an artifact?

**Before this pass: no, on both counts, for the main training notebook.**

- `notebooks/kaggle_train.ipynb` Cell 4 called `train(...)` without capturing
  its return value, and Cell 5 called `run_evaluation(...)` without passing
  `wandb_run`. Every plot function silently skipped its W&B logging branch
  — the four training-eval plots existed only as ephemeral PNGs in
  `/kaggle/working/results`, gone when the Kaggle session ends.
- Every plot function that *did* have `wandb_run` (pilot path, via
  `scripts/run_pilot_experiment.py`) only called `wandb_run.log({...:
  wandb.Image(path)})` — that puts the image in the run's **Media tab**,
  which is not a versioned, downloadable `wandb.Artifact`. Nothing in the
  repo called `wandb.Artifact(...)` for plots before this pass (only
  `train_lora.py`'s adapter metadata artifact existed).

**Fixed in this pass:**
- `notebooks/kaggle_train.ipynb` — Cell 4 now does
  `wandb_run = train(config_path=..., finish_wandb=False)`; Cell 5 passes
  `wandb_run=wandb_run` into `run_evaluation` and finishes the run itself.
  All four training-eval plots + the hyperparameter table now land in the
  same W&B run as training.
- Added `log_plots_artifact()` (`src/training/report_tables.py`) — bundles
  a run's generated plot PNGs + metrics CSV into one versioned
  `wandb.Artifact(type="eval-plots")` and logs it. Wired into
  `run_evaluation` (`evaluate.py`) and both `run_safim_evaluation` /
  `run_safim_evaluation_by_type` (`safim_eval.py`). `wandb.Image` media
  logging is left in place too (good for the live Charts/Media view) —
  the artifact is the versioned, downloadable counterpart.
- `plot_pilot_comparison` (analyze_pilot_results.py path) is unchanged —
  it runs after training, offline, with no live run to attach an artifact
  to. Left as a local PNG; out of scope unless you want a dedicated
  "pilot report" W&B run to log it into.

## 3. Is experiment-setup reporting properly implemented?

**Before this pass: no.** `render_hyperparameter_table` /
`write_hyperparameter_table` (`report_tables.py`) existed and were unit
tested (`local_tests/test_report_tables.py`), but nothing in
`train_lora.py`, `evaluate.py`, or any notebook ever called them — dead
code exercised only by tests.

**Fixed in this pass:** `train_lora.py`'s `train()` now, right after
writing `metadata.json`:
- writes `hyperparameter_table.md` alongside the adapter,
- logs it as a `wandb.Table` under `experiment_setup/hyperparameter_table`
  (visible in the run's Tables tab),
- adds the same file to the existing `{run_id}-adapter` W&B artifact, so
  the exact hyperparameters travel with the adapter weights' provenance
  record.

`render_compute_cost_table` is **still not wired in** — by design, per its
own docstring: GPU type / $-per-hour aren't knowable from inside a live
Kaggle run, so it deliberately takes pre-collected dicts rather than
querying W&B itself. Use `fetch_wandb_run_durations()` +
`write_compute_cost_table()` as a manual/notebook step after a batch of
runs, the same way `analyze_pilot_results.py` aggregates post-hoc. Say the
word if you want that turned into a script now — it wasn't in scope of
"is it wired into training," since it structurally can't be.
