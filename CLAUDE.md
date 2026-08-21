# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tunes Qwen2.5-Coder-0.5B for Python fill-in-the-middle (FIM) autocomplete via LoRA on Kaggle
GPUs, evaluated with execution-based benchmarks (SAFIM), and deployed locally through Ollama. See
`README.md` for the full project overview and `logs/python-fim-0.5b-project-plan.md` for the
original design rationale (why LoRA over QLoRA, distillation plan, naming conventions).

**Core rule that shapes the whole repo:** notebooks contain no real logic. `notebooks/kaggle_train.ipynb`
and `notebooks/kaggle/curate_and_checkpoint.ipynb` are thin controllers that `pip install` the repo
and call into `src/`/`scripts/`. All logic is version-controlled Python, testable locally without a GPU.

## Commands

```bash
# Environment: uv-managed (uv.lock present), Python >= 3.13
uv sync                                    # install deps
# or: python3 -m venv .venv && pip install <deps from pyproject.toml>

# Tests — all data-pipeline/FIM logic is GPU-free and unit-tested
pytest local_tests/ -v
pytest local_tests/test_ast_bucketer.py -v          # single file
pytest local_tests/test_ast_bucketer.py::test_name -v   # single test

# Lint
ruff check .

# Build a local FIM dataset from this repo's own source (smoke-tests the pipeline)
python src/data_prep.py --source-dirs src local_tests --output data/fim_dataset.jsonl

# Verify training config loads without a GPU
python src/training/train_lora.py

# Stage 1 data curation (streams HF stack-v3-train, never materializes the full corpus)
python scripts/build_sample.py --max-gb 2.0
python scripts/build_sample.py --report-only       # regenerate reports/stage1_filter_report.md only

# Stage 1 FIM variant generation (random vs. planned span sampling)
python scripts/generate_fim_variants.py             # both variants
python scripts/generate_fim_variants.py --variant random

# Pilot experiment: trains random vs. planned variants across seeds, then SAFIM-evaluates each
python scripts/run_pilot_experiment.py
python scripts/analyze_pilot_results.py

# GGUF conversion for Ollama deployment (merge adapter -> GGUF -> Modelfile)
python src/convert_gguf.py --base-model Qwen/Qwen2.5-Coder-0.5B --adapter <path>
```

Secrets (`HF_TOKEN`, `WANDB_API_KEY`) load from `.env` locally (see `.env.example`); on Kaggle they
come from Kaggle secrets, never hardcoded. W&B/Weave tracking is opt-in everywhere — every init
function no-ops with a warning if `WANDB_API_KEY` isn't set, so all local/test workflows run with
zero W&B setup.

## Architecture

### Two-stage pipeline

- **Stage 1** (`src/curation/`, `src/dedup/`, `scripts/build_sample.py`): streams
  `HuggingFaceCode/stack-v3-train`, runs six ordered filter stages (language reject → license →
  quality/AST-validity/size → secrets → exact dedup), and checkpoints curated Python files to an HF
  dataset repo. Never calls `load_dataset(..., streaming=False)` — the corpus must stay streamed.
  Design spec: `.claude/stages/data_stages/data-stage-1.md`.
- **Stage 2** (`src/fim/`, `scripts/generate_fim_variants.py`): buckets curated files' AST spans by
  type (line/expression/statement/block/function-body/method-body/class-level/api-call —
  `src/fim/ast_bucketer.py`) and samples two FIM dataset variants from the same pool:
  `distribution_random.py` (uniform, baseline) vs. `distribution_planned.py` (fixed percentages from
  `configs/data/fim_distribution.yaml`, a hypothesis the pilot is meant to validate or falsify).
  Design spec: `.claude/stages/data_stages/data-stage-2.md`.

### Config-driven training — no hardcoded hyperparameters

- `configs/training/lora.yaml` holds every LoRA/training hyperparameter and the W&B run-ID fields.
  Change hyperparameters there, never in `src/training/train_lora.py`.
- `configs/training/pilot.yaml` layers only what varies per pilot run (seed, dataset variant, output
  subfolder) on top of `lora.yaml` via a deep merge in `train_lora.train()`'s `overrides` param —
  shared hyperparameters stay untouched in one place.
- W&B run IDs are deterministic, never random: built by `build_run_id()` in `train_lora.py` as
  `{model_slug}-lora-r{r}-e{epochs}-ds{dataset_version}-{YYYYMMDD}-a{attempt}`, so re-running the same
  config resumes the same run instead of forking a new one.

### One shared HF model repo, subfolder-separated

Every experiment variant and every fine-tune approach (LoRA, QLoRA, distilled, ...) pushes adapters
into **one** HF repo (`Rudra-G-23/qwen2.5-coder-0.5b-python-fim`), separated by `subfolder`
(e.g. `experiment/random_model`) rather than by separate repos — keeps W&B/eval comparisons and HF
browsing under one roof. `lora.yaml`'s `output.subfolder` is the default; `pilot.yaml`'s
`run_pilot_experiment.py` composes a per-seed path from it (`{output_subfolder}/seed{seed}`, e.g.
`experiment/random_model/seed42`), so every seed's adapter lands on its own subfolder in the
default branch instead of overwriting the variant's shared folder.

### Data-curation vs. training W&B/Weave tracking are separate projects

`src/curation/wandb_logger.py` (data-curation/FIM-generation runs) and `src/training/train_lora.py`'s
`init_wandb` (training runs) share the same graceful no-op-if-unset pattern and "entity/project"
config-string convention, but log to **different** W&B projects — curation uses
`configs/data/*.yaml`'s `wandb.project`, training uses `configs/training/*.yaml`'s. Don't merge them.

### Checkpointing (Stage 1 resume)

`src/curation/checkpoint.py` stores checkpoint JSONs alongside data chunks in the same HF dataset
repo — no separate state store. On resume, the highest `shard_index`/`row_offset` checkpoint
determines where streaming continues, and the dedup hash set is rebuilt from previously uploaded
chunk parquet files (checkpoint JSONs don't carry per-file hashes). All HF Hub calls go through
module-level `HfApi`/`hf_hub_download` references specifically so tests can monkeypatch them without
network access.

### Evaluation

`src/training/evaluate.py` does base-vs-LoRA comparison (exact match, edit similarity, plots).
`src/training/safim_eval.py` runs execution-based pass@1 against SAFIM's Python subset and exposes
`aggregate_pilot_results`, kept in `src/` (not `scripts/`) specifically so pilot aggregation is
testable without a GPU.

## Repo-specific docs worth reading before large changes

- `.claude/stages/data_stages/data-stage-1.md`, `data-stage-2.md` — authoritative specs for the
  curation and FIM-generation stages; module docstrings throughout `src/curation/`, `src/dedup/`,
  `src/fim/` reference specific sections (e.g. "Stage 1 §6") — check the spec before changing behavior
  a docstring attributes to it.
- `.claude/dicussions/` — dated design-decision records (model repo consolidation, scaling-risk
  fixes, pilot readiness, sample migration) for context on why current structure looks the way it does.
