# 2026-08-15 — Stage 1 sample migrated into `the-stack-v3-python-fim-data`

Session goal: finalize `data-stage-2.md` §1's open "new repo name" item and
build out the slice of its repo tree that covers the existing 10k-file
Stage 1 sample — `sample_filtered_data_10000/` and
`experiment/{random_data,distributed_data}/` — plus the two FIM-generation
notebooks that were speced (§6) but never built. Full plan:
`/home/rudra/.claude/plans/nifty-singing-llama.md`.

Code/config/notebooks only this session — no HF repo created, no data
migrated, no notebook run. Rudra runs
`scripts/migrate_sample_to_new_repo.py` and the three
`notebooks/stack_v3_data_pull/*.ipynb` notebooks himself, with his own
`HF_TOKEN`.

## Decisions made

1. **Repo name resolved**: `Rudra-G-23/the-stack-v3-python-fim-data`
   (namespace matches every other repo in this project — `configs/`,
   `README.md`). Resolves `data-stage-2.md` §1's "new repo name" item. The
   other three §1 items (pilot winner, full-corpus-vs-Kaggle scale,
   `seen_hashes` timing) are untouched and still open.

2. **`metadata.json`, not `.jsonl`**: Rudra's request said `.jsonl`, but
   `data-stage-2.md` §3 had already deliberately specced a single aggregate
   `metadata.json` per folder specifically to avoid duplicating per-row
   provenance that already lives in every parquet row's `curation_metadata`
   (content hash, source shard/row, span/bucket). A per-row `.jsonl` would
   be a second, redundant copy of the same data at real scale. Went with
   the already-decided spec — `src/curation/checkpoint.py`'s new
   `write_metadata_json`/`render_metadata` and `src/fim/ast_bucketer.py`'s
   new `write_experiment_metadata_json`/`render_experiment_metadata` both
   produce aggregate counts + a bounded config block, never a per-row dump.

3. **`experiment/*/checkpoints/` are empty on purpose**:
   `scripts/generate_fim_variants.py` loads the whole 10k-file sample into
   memory and bucket-samples both variants in one pass (see runtime estimate
   below) — there's nothing to resume. Only
   `sample_filtered_data_10000/checkpoints/` holds real per-chunk resume
   state, migrated from the old flat repo's `checkpoint_*.json` files.
   Each experiment folder's `metadata.json` states this explicitly
   (`"checkpoints": "empty — ... nothing to resume"`) instead of shipping
   an unexplained empty folder.

4. **Naming split — internal code vs. HF folder names**: the codebase keeps
   calling the two FIM-sampling strategies `random` / `planned`
   (`src/fim/distribution_random.py` / `distribution_planned.py`, existing
   tests, `data-stage-1.md`). The HF repo folder names and CLI-facing labels
   use `random_data` / `distributed_data`, matching Rudra's requested layout
   and `data-stage-2.md`'s tree. The mapping is made in exactly one place —
   `scripts/generate_fim_variants.py`'s `FOLDER_NAME` dict — not renamed
   throughout the codebase.

5. **`quality_reject_by_reason` — the "small, real fix" `data-stage-2.md`
   §3 flagged**: `evaluate_quality()` in `src/curation/quality_filters.py`
   already computed a specific reason (`ast_invalid` / `size_out_of_bounds`
   / `binary` / `minified_or_generated`), but
   `stream_source.apply_filter_pipeline` discarded it, keeping only the
   coarse `quality_reject` boolean. `FilterResult` now carries
   `quality_reason`, and `collect_chunk`/checkpoints aggregate it into
   `quality_reject_by_reason`. **This can't be retrofitted onto the
   already-existing Stage 1 checkpoints** — they predate this field, so
   `sample_filtered_data_10000/metadata.json`'s
   `quality_reject_by_reason` will read empty/undercounted against the
   historical 10k run specifically. `reject_by_stage` has no such gap; it
   was tracked from the start. `license_reject_by_type` and
   `rejected_examples_sample` (also part of §3's full schema) were
   deliberately **not** built this pass — no current need beyond what
   `reject_by_stage` + `quality_reject_by_reason` cover.

## What got built

- **Path-prefix support** in `src/curation/checkpoint.py` — checkpoints and
  chunk data now live under `{path_prefix}/checkpoints/` and
  `{path_prefix}/data/` instead of flat repo root. Default `path_prefix=""`
  preserves the old flat-repo behavior exactly (existing tests pass
  unchanged).
- **`scripts/migrate_sample_to_new_repo.py`** (new) — one-time, idempotent
  copy of the old flat repo's chunk parquet + checkpoint JSONs into the new
  repo's `sample_filtered_data_10000/{data,checkpoints}/`.
- **`--variant {random,planned}` flag** on `scripts/generate_fim_variants.py`
  (previously always generated both in one run) — lets
  `random_data_10000.ipynb` and `distributed_data_10000.ipynb` run
  independently, mirroring the existing pattern in
  `scripts/run_pilot_experiment.py`.
- **Cross-run comparison chart**: since each generation notebook only
  produces one variant, the random-vs-distributed bar chart
  (`plot_bucket_distribution` in `src/fim/ast_bucketer.py`) can't be built
  from a single run. `generate_fim_variants.py` now checks the *other*
  variant's already-pushed `metadata.json` on HF and completes the chart
  automatically once both exist — whichever notebook runs second finishes
  it, regardless of order.
- **W&B logging** — `src/curation/wandb_logger.py` (new, curation side,
  `job_type="data-curation"`, fixed run id + `resume="allow"` for multi-session
  continuity per `data-stage-2.md` §7) and inline logging in
  `generate_fim_variants.py` (`job_type="experiment-random"` /
  `"experiment-distributed"`). Both are graceful no-ops if `WANDB_API_KEY`
  isn't set, matching `src/training/train_lora.py`'s existing pattern. The
  bucket-distribution chart is also displayed inline in the Kaggle notebook
  output via `IPython.display.Image`, not just pushed to W&B.
- **Notebook decision note**: every generation notebook's final cell prints
  that the generation-stage chart is *not* a winner decision — the actual
  random-vs-distributed call is made after training + SAFIM eval comparison
  (`data-stage-1.md` §7-8, `scripts/analyze_pilot_results.py`), not from
  bucket counts alone.
- `notebooks/stack_v3_data_pull/sample_filtered_data_10000.ipynb` was a
  0-byte invalid-JSON stub — same failure mode
  `2026-08-15-stage1-pilot-readiness.md` found and fixed in
  `notebooks/experiments/{random,planned}.ipynb`. Built out for real.

## Estimated experiment runtime

Both `random_data_10000.ipynb` and `distributed_data_10000.ipynb` are
CPU-only, single in-memory pass over the already-curated 10k-file
(135B–13KB each) pool — no GPU, no streaming:

- Clone + pip install (light deps: `datasets`, `pyarrow`, `huggingface_hub`,
  `pyyaml`, `matplotlib`, `wandb`): ~1–2 min
- `load_dataset()` of the 10k-row sample from HF: ~1–3 min, network-bound
- AST bucketing all 10k files + sampling ~20,000 FIM examples for one
  variant: well under a minute of CPU — `ast.parse` on files this small is
  fast, no training step here
- Parquet write + upload: well under a minute

**Roughly 5–10 minutes per notebook**, comfortably inside Kaggle's 12-hour
session cap — no session-splitting needed, unlike the GPU training
notebooks. First run may take a little longer due to cold HF dataset
caching.

## Known gap this session doesn't close

`configs/training/pilot.yaml` (`run_pilot_experiment.py`'s config) still
points `dataset_hf_repo` at the old flat repo
(`Rudra-G-23/qwen-coder-python-fim-data`) with `dataset_config: "random"` /
`"planned"`. `materialize_variant_jsonl` in `run_pilot_experiment.py`
downloads `f"{dataset_config}/fim_{dataset_config}.parquet"` — a flat,
one-level path that doesn't match the new repo's nested
`experiment/random_data/data/fim_random.parquet` layout. Repointing pilot
training at the newly-generated data needs a small code change in
`materialize_variant_jsonl` (separate `folder`/`config_name` values instead
of one `dataset_config` doing both jobs), not just a config edit — left
alone this session since neither pilot model repo exists yet and pilot
training hasn't started (confirmed in project memory as of 2026-08-15).
Flagging it here so it isn't silently missed once training actually starts.

## Explicitly out of scope

Full-corpus filtering (`python_filtered_data/`, uncapped `target.files`),
top-level `checkpoint/seen_hashes.*` persistence, `model_data/` splitter,
and the pilot-winner decision — all still open per `data-stage-2.md` §1,
none of it touched here.
