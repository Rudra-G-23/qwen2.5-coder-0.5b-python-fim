# Stage 1 Implementation Plan — Data Curation Pipeline + Pilot FIM Experiment

This document is the complete spec for an implementation agent. It covers pulling
Python data from `HuggingFaceCode/stack-v3-train`, filtering/curating it, checkpointing
progress to Hugging Face, generating FIM examples in two span-distributions (random
vs. planned), and running a controlled pilot LoRA comparison on Qwen2.5-Coder-0.5B.

Do not scale to the full corpus in this stage. Target: a **10,000-file curated sample**,
used to run a small, fair, seed-controlled comparison before any large-scale run.

---

## 0. Non-negotiable ground rules

- **Never download the full `stack-v3-train` corpus.** Always use
  `load_dataset(..., streaming=True)`. Stop once 10,000 qualifying Python files
  have been collected (or a per-session GB budget is hit — see §4).
- **Every filter-stage count must be logged.** No silent drops.
- **License fields are saved unconditionally**, for every file, including
  `license_type == "no_license"` and empty `detected_licenses` lists. Never omit
  the field — filtering by license happens at *train-set construction time*, not
  at storage time.
- **No hardcoded hyperparameters or paths in scripts.** Everything driven by
  YAML configs under `configs/`.
- **Base model is never re-trained on itself.** Both pilot runs (random vs.
  planned distribution) start fresh from `Qwen/Qwen2.5-Coder-0.5B`, same LoRA
  config, same seed(s), same hyperparameters — the *only* variable that changes
  is the FIM span distribution.
- **HF write token already exists** (`Rudra-G-23` account). No new token/scope
  needed for public dataset + model repos.

---

## 1. Repository structure to create/extend

```
qwen2.5-coder-0.5b-python-fim/
├── configs/
│   ├── data/
│   │   ├── stack_v3_filter.yaml
│   │   └── fim_distribution.yaml
│   └── training/
│       └── lora.yaml                 # already provided by user — keep as-is
├── src/
│   ├── curation/
│   │   ├── stream_source.py          # streams stack-v3-train, applies filters
│   │   ├── license_policy.py
│   │   ├── quality_filters.py        # AST validity, size bounds, secrets
│   │   └── checkpoint.py             # HF-based resume/checkpoint logic
│   ├── dedup/
│   │   └── exact_dedup.py            # SHA-256 based
│   ├── fim/
│   │   ├── ast_bucketer.py           # tags spans: line/expr/stmt/block/func/...
│   │   ├── distribution_random.py
│   │   └── distribution_planned.py   # 20/15/20/15/15/10/3/2% from research plan §8.3
│   └── training/
│       └── train_lora.py
├── scripts/
│   ├── build_sample.py               # entrypoint: stream + filter + checkpoint + push
│   ├── generate_fim_variants.py      # builds random + planned FIM datasets from sample
│   └── run_pilot_experiment.py       # trains both variants, logs to W&B
├── notebooks/
│   └── kaggle/
│       └── curate_and_checkpoint.ipynb   # thin — calls scripts/build_sample.py
└── reports/
    └── stage1_filter_report.md       # auto-generated per-stage counts
```

---

## 2. Data source

- Source: `HuggingFaceCode/stack-v3-train` (repo-grouped, license-filtered already
  at `permissive` / `no_license` level, inline file content).
- Load with:
  ```python
  from datasets import load_dataset
  ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True)
  ```
- Each row = one repository. Iterate `repo["files"]`, filter to `language == "Python"`.
- Do **not** re-implement license detection — `license_type` and `detected_licenses`
  are already provided per file. Just record them.

---

## 3. Filter stages (apply in this exact order, count rejects at each stage)

1. **Language filter** — keep only `language == "Python"`.
2. **Vendor filter** — drop `is_vendor == true`.
3. **License policy** (`src/curation/license_policy.py`) — per the research plan:
   keep permissive + no_license; flag (don't silently drop) anything ambiguous
   for later manual review. Always store `license_type` + `detected_licenses`
   regardless of decision.
4. **Quality filters** (`src/curation/quality_filters.py`):
   - Parse with Python's `ast` module — reject files that fail to parse.
   - Size bounds — configurable min/max in `stack_v3_filter.yaml` (use the
     v3 stats' p5/p95 file-size-bytes as sane defaults: ~135–13,457 bytes).
   - Reject binary/minified/generated-looking files.
5. **Secret detection** — reject files containing likely API keys/tokens/private
   keys/passwords (simple regex pass is sufficient for this stage; do not overbuild).
6. **Exact deduplication** (`src/dedup/exact_dedup.py`) — SHA-256 of file content,
   keep first occurrence, log duplicate count.

Stop collecting once **10,000 files survive all six stages**. Log the total number
of raw files scanned to reach that count (this is the "reject rate" metric for the report).

---

## 4. Streaming, chunking, and checkpointing to Hugging Face

- **Chunk format: Parquet.** Chosen because chunks will be re-fetched for training
  via `load_dataset()` — Parquet is faster to load and smaller on disk than JSONL.
- **Per-chunk checkpoint JSON**, stored in the *same* HF dataset repo, one per chunk,
  containing **both** resume position and filter-stage stats:
  ```json
  {
    "chunk_id": "chunk_0007",
    "shard_index": 42,
    "row_offset": 15032,
    "files_collected_this_chunk": 1000,
    "raw_files_scanned_this_chunk": 15310,
    "filter_stats": {
      "language_reject": 8210,
      "vendor_reject": 240,
      "license_reject": 130,
      "quality_reject": 900,
      "secret_reject": 12,
      "exact_dup_reject": 818
    },
    "cumulative_files_collected": 7000,
    "timestamp": "2026-08-13T12:00:00Z"
  }
  ```
- **Auto-resume logic** (`src/curation/checkpoint.py`):
  1. On script start, list all `checkpoint_*.json` files in the HF dataset repo.
  2. Pick the one with the highest `shard_index`/`row_offset`.
  3. Call `.skip(row_offset)` on the streaming dataset to resume exactly there.
  4. No manual copy-pasting of checkpoint state between sessions — this is automatic.
- **GB-budget control per session:** the user passes a single argument,
  `--max-gb`, to `scripts/build_sample.py`. The script tracks cumulative bytes of
  *collected* (post-filter) content, stops once `max_gb` is hit, uploads the
  current chunk + checkpoint, and exits cleanly. This gives session-by-session
  manual control without manual checkpoint handling.
- **Idempotent uploads:** name chunk files by shard/row range, e.g.
  `chunk_shard0042_row10000-15000.parquet`. If a file with that name already
  exists in the HF repo, skip re-uploading it.
- **No GPU required for this stage** — pure CPU/I/O. Preserve Kaggle GPU quota
  for the pilot training runs in §7.

---

## 5. Per-row schema (mirrors `stack-v3-train`, do not drop fields)

Save every field listed below for every file, unconditionally:

```
repo_path, repo_id, commit_id,
github_metadata: { branch, commit_count, repo_created_at, is_fork, is_org_owned,
                    forked_from, stars, forks, issues, pull_requests },
num_files,
files[]: { content_id, content, size_bytes, file_path, file_timestamp,
           language, is_vendor, license_type, detected_licenses }
```

Add project-specific fields on top (not replacing any of the above):
```
curation_metadata: {
  content_hash_sha256,
  passed_quality_filter: bool,
  ast_parseable: bool,
  secret_scan_flag: bool,
  source_shard_index,
  source_row_offset,
  collected_at_timestamp
}
```

---

## 6. FIM generation — two variants from the same 10,000-file sample

Build the AST bucketer first (`src/fim/ast_bucketer.py`) — this is required by
*both* variants, not optional infrastructure:

- Parse each file's AST.
- Tag every valid span with type: line / expression / statement / block /
  function-body / method-body / class-level / API-call, plus metadata
  (node type, nesting depth, enclosing function/class, structural position).

Then generate two FIM datasets from the same bucketed pool:

- **Variant A — Random** (`src/fim/distribution_random.py`): sample spans
  uniformly at random from all valid spans, ignoring type balance.
- **Variant B — Planned** (`src/fim/distribution_planned.py`): sample according
  to the research plan's starting distribution — 20% line, 15% expression,
  20% statement, 15% block, 15% function-body, 10% method-body, 3% class-level,
  2% API-call.

Store both as separate HF dataset configs/folders in the **same** dataset repo
(e.g. `qwen-coder-python-fim-data/random/` and `.../planned/`) since they share
the same underlying source content.

Each FIM example record must carry: `fim_type`, `source content_id` (traceability
back to the raw file), `prefix`, `suffix`, `middle`, and structural metadata from
the bucketer (ast_depth, nesting_depth, etc. — reuse the hardness-record shape
from the research plan §9.2, even if hardness scoring itself is deferred).

---

## 7. Pilot training comparison

Two separate HF model repos (not branches — cleaner for W&B linking and eval comparison):
- `Rudra-G-23/qwen-coder-python-fim-random`
- `Rudra-G-23/qwen-coder-python-fim-planned`

Both runs:
- Start fresh from `Qwen/Qwen2.5-Coder-0.5B` (no continuation from one another).
- Use the **same** `configs/training/lora.yaml` (rank 16, alpha 32, same target
  modules, same batch/accum/lr/scheduler).
- Use the same seed(s). Given the small dataset size, run **2–3 seeds per
  variant** if compute allows, not a single run each — a 10k-file result is a
  pilot signal, not a conclusive one, and seed variance must be distinguishable
  from distribution effect.
- Only the training data (`random/` vs `planned/` FIM dataset) differs.
- Data loading: `load_dataset()` directly from the HF dataset repo — **no Kaggle
  dataset-import feature**. This replaces the old Kaggle-dataset-attachment workflow.

W&B tracking:
- Single project: `qwen-coder-python-fim` (not a separate project per the
  decision above).
- Use `job_type` to distinguish curation runs (`job_type="data-curation"`) from
  training runs (`job_type="train"`), so both live in one history without
  fragmenting it.
- Deterministic run IDs per the existing `lora.yaml` convention:
  `{model_slug}-lora-r{r}-e{epochs}-ds{dataset_version}-{YYYYMMDD}-a{attempt}`.
- Log dataset provenance in each training run's config: which chunk range /
  dataset config (`random` or `planned`) was used, and a link/commit hash to
  the exact HF dataset revision — this is the reproducibility trail.

---

## 8. Evaluation of the pilot

- Use SAFIM as the primary benchmark (already in project references).
- Compare random vs. planned across all seeds — look for a consistent, non-noise
  gap, not a single-run difference.
- Treat results as **preliminary**. Do not scale the corpus or commit to one
  distribution until the gap is consistent across seeds. If small/inconsistent,
  more data is needed before concluding anything.

---

## 9. Explicit non-goals for this stage

- Do not build near-deduplication yet (exact dedup only, per the phased plan).
- Do not build repository-aware / multi-file context yet (Dataset C — later phase).
- Do not implement hardness scoring beyond capturing the metadata fields needed
  to compute it later.
- Do not scale past 10,000 files until the pilot experiment result is reviewed.
