# 2026-08-15 — Consolidate pilot model repos into one HF model repo

Session goal: resolve `data-stage-2.md` §1's implicit "two model repos"
assumption (inherited from `data-stage-1.md` §7's original pilot spec) into
a single shared model repo, matching the data side's existing convention of
one dataset repo with subfolders per variant.

## Decision

**One HF model repo**: `Rudra-G-23/qwen2.5-coder-0.5b-python-fim`, holding
both pilot experiment adapters under `experiment/random_model/` and
`experiment/distributed_model/`, instead of the previously-planned
`qwen-coder-python-fim-random` / `qwen-coder-python-fim-planned` two-repo
split. Each subfolder carries its own `metadata.json` (run id, base model,
LoRA hyperparams, seed, dataset version, final train loss) alongside the
adapter weights, since there's no per-repo README to lean on anymore.

Rudra's framing, applied directly: this repo holds the two Base-vs-LoRA-FT
pilot comparisons now, and will hold every later fine-tune approach
(LoRA, QLoRA, ...) tried on whichever dataset variant wins the pilot —
same repo, same reasoning, no reason to fork a new repo per approach either.
That later phase's folder naming (e.g. `finetune/<approach>_model/`) is
intentionally left undecided here — it starts only once the pilot winner is
picked, so naming it now would be inventing ahead of need, the same
restraint `data-stage-2.md` §1 already applies to
`python_random_distributed_data/`'s eventual rename.

## What changed

1. `configs/training/lora.yaml` — `output.hf_repo` →
   `Rudra-G-23/qwen2.5-coder-0.5b-python-fim` (was
   `...-lora`); added `output.subfolder: ""` (root, for standalone
   non-pilot runs).
2. `configs/training/pilot.yaml` — replaced each variant's
   `output_hf_repo` with a single top-level `output_hf_repo` plus a
   per-variant `output_subfolder` (`experiment/random_model` /
   `experiment/distributed_model`).
3. `src/training/train_lora.py`'s `train()` — replaced
   `model.push_to_hub()` / `tokenizer.push_to_hub()` (which always target a
   repo root) with `HfApi.create_repo(..., exist_ok=True)` +
   `HfApi.upload_folder(..., path_in_repo=subfolder)`, since `push_to_hub`
   has no clean subfolder-targeting path. Also writes a `metadata.json`
   into the adapter folder before upload.
4. `scripts/run_pilot_experiment.py` — overrides now pass both
   `output.hf_repo` (shared) and `output.subfolder` (per variant); docstring
   updated to describe the shared-repo/subfolder model instead of
   per-variant repos. The known per-seed-overwrite limitation is unchanged
   in substance, just reworded (subfolder instead of repo).
5. `notebooks/kaggle_train.ipynb`, `notebooks/kaggle_wandb.ipynb` — the
   "Resuming next week" `PeftModel.from_pretrained(...)` example now points
   at the new repo name (root, since these are standalone non-pilot runs
   using `output.subfolder: ""`).
6. `data-stage-1.md` §7, `data-stage-2.md` §1 — documented the resolved
   repo name/structure and the future fine-tune-approach discussion.

Not touched: `logs/archive/*` (historical record of a prior state, left
as-is on purpose) and the data-curation HF repo
(`Rudra-G-23/the-stack-v3-python-fim-data`), which is a separate repo for
curated/FIM *data*, not trained models.
