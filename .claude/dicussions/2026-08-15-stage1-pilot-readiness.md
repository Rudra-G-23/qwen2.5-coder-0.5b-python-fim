# 2026-08-15 — Stage 1 pilot readiness: citation fix, live monitoring, execution wiring

Session goal: answer "how were the FIM bucketing percentages actually
decided," assess whether the project is on track, and get Stage 1 (the
10k-file pilot) from "fully coded, never run" to "ready to execute on
Kaggle." Full plan: `/home/rudra/.claude/plans/home-rudra-open-qwen2-5-coder-0-5b-pyth-polymorphic-shamir.md`.

This doc exists because most of what changed here is in code and notebooks,
not in `.claude/stages/data/data-stage-1.md` or `data-stage-2.md` — those
only picked up one small citation fix (the repo-tree comment). Everything
else below is new, not reflected in those spec docs.

## 1. The bucketing-percentage citation was phantom

`src/fim/distribution_planned.py`, `configs/data/fim_distribution.yaml`,
`src/fim/ast_bucketer.py`, and `data-stage-1.md` all attributed the
20/15/20/15/15/10/3/2% span-type split to "research plan §8.3." No such
section exists anywhere in the repo (checked the full project plan, the
archived write-up, the paper reading list). Fixed the citation everywhere
except `data-stage-1.md`'s own §6 prose, which stays as the historical
record of what Stage 1 actually shipped with.

## 2. Natural span-type distribution — measure, don't just assert

Added `measure_natural_distribution` / `render_natural_distribution_report` /
`write_natural_distribution_report` to `src/fim/ast_bucketer.py`, wired into
`scripts/generate_fim_variants.py`. When that script runs, it now also
counts how span-types naturally occur in the curated corpus (unweighted) and
writes `reports/natural_span_distribution.json` next to the random/planned
variants — so the hand-set planned percentages can be compared against
reality, not just asserted. Tested in `local_tests/test_ast_bucketer.py`.

## 3. Pilot execution notebooks — were empty stubs, now real

`notebooks/experiments/random.ipynb` and `planned.ipynb` were 0-byte files.
Built out as thin Kaggle controllers (clone → install GPU deps → HF+W&B auth
→ run → next-steps), one per variant so each fits a 12h Kaggle session
without needing all 6 pilot runs (2 variants × 3 seeds) in one sitting.

Also cleared a stale `CalledProcessError` traceback that had gotten
committed into `curate_and_checkpoint.ipynb` — it was from testing a
Kaggle-only path (`/kaggle/working/...`) locally, not a real bug in
`build_sample.py`.

## 4. `run_pilot_experiment.py` — added `--variant` and automatic SAFIM eval

- `--variant random|planned`: runs only that variant's seeds, matching the
  one-notebook-per-variant split above.
- SAFIM eval now runs automatically after each seed's training, instead of
  being a separate manual step — results land in
  `results/pilot/{variant}_seed{seed}/safim_metrics.csv`.
- **Known limitation, documented in the script's own docstring, not fixed
  here**: `pilot.yaml` gives each *variant* one shared HF adapter repo across
  all 3 seeds, not one repo per seed — each seed's push is a new commit in
  that repo, so only the last seed's adapter is on the default branch.
  Doesn't affect the pilot comparison (eval runs on the local adapter
  immediately after training, before the next seed's push), only later
  re-fetching a specific past seed's adapter from HF. Fixing it means
  picking a new per-seed repo naming scheme — a naming decision left to
  Rudra, same as `data-stage-2.md §1` already reserves naming decisions.

## 5. `scripts/analyze_pilot_results.py` — new

Aggregates the 6 (variant × seed) `safim_metrics.csv` files into a
random-vs-planned comparison: per-variant mean/min/max pass@1, and a
plain-language verdict on whether the two variants' seed-score *ranges*
overlap (inconclusive) or not (real gap) — deliberately not a formal
significance test, since n=3 seeds is too few for one to mean anything.
Aggregation logic lives in `src/training/safim_eval.py`
(`aggregate_pilot_results`, tested in `local_tests/test_safim_eval.py`) so
it's testable without a GPU; the script itself is a thin caller.

## 6. Found and fixed: W&B run was being finished before eval could log to it

`train_lora.train()` always called `wandb_run.finish()` before returning.
Every existing caller that then tried to log more into that same run
(`run_evaluation`, `run_safim_evaluation` in both `kaggle_wandb.ipynb` and
the new pilot flow) was calling `.log()` on an already-finished run — the
W&B SDK drops those calls silently, no error, so this would have looked like
it worked while quietly not logging anything.

Fix: `train()` now takes `finish_wandb: bool = True`. Both
`run_pilot_experiment.py` and `kaggle_wandb.ipynb` now pass
`finish_wandb=False`, do their eval logging into the still-open run, and
call `wandb_run.finish()` themselves afterward.

Also added live progress logging to `evaluate_pass_at_1`
(`src/training/safim_eval.py`): every 10 samples (configurable via
`log_every`), it now logs `running_pass_at_1`, `completed`, and
`sec_per_sample` to the W&B run (in addition to the existing stdout print),
and prints a rough ETA — so a 100-sample SAFIM run's progress is visible on
the W&B dashboard live, not just in Kaggle notebook stdout you have to be
watching.

## State after this session

113/113 tests pass (101 original + 12 new). Nothing here required a GPU —
all of it is code/notebook wiring.

**Correction, checked directly against the live HF repo right after this
session's commit**: curation is further along than assumed above. The
`Rudra-G-23/qwen-coder-python-fim-data` repo already has all 10 checkpoint
chunks and `cumulative_files_collected: 10000` as of 2026-08-14 — curation
is done, not pending. `python scripts/build_sample.py --report-only`
confirms: 144,934 raw files scanned, 10,000 collected (93.10% reject rate,
dominated by the language filter as expected for a multi-language corpus).
`generate_fim_variants.py` has NOT been run yet (no `random`/`planned` files
in the repo), and neither pilot output repo
(`qwen-coder-python-fim-{random,planned}`) exists yet. So the actual next
step is `scripts/generate_fim_variants.py` (CPU-only, runs locally, no
Kaggle needed), not another curation session.
