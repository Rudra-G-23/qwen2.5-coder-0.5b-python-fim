# Stage 2 — Full-Corpus Curation, Experiment Re-Run & Model Data

## 0. Context

Stage 1 streamed `stack-v3-train`, filtered down to a 10,000-file Python
sample, and ran a pilot LoRA comparison between two FIM-span sampling
strategies — `random` (type-blind uniform) and `planned` (fixed
percentage-per-span-type) — on that small sample. It lives in
`scripts/build_sample.py`, `src/curation/`, `src/fim/`, and
`scripts/run_pilot_experiment.py`, driven by
`configs/data/stack_v3_filter.yaml` and `configs/data/fim_distribution.yaml`,
pushing flat parquet + `checkpoint_*.json` files into one HF dataset repo
(`Rudra-G-23/qwen-coder-python-fim-data`).

Stage 2 scales this to the **full** stack-v3-train Python corpus — no
`target.files` cap — and restructures the output into an explicit,
documented HF dataset repo instead of a flat one. The output backs a
research paper and portfolio work, so every curation and sampling decision
must be traceable back to *why* a row was kept, cut, or bucketed — not just
how many.

Prior code comments reference `.claude/data-v1.md §N` as Stage 1's spec of
record. That file now lives at `.claude/stages/data/data-stage-1.md`
(relocated from `.claude/prompts/data-stage-v1.md`) — it *is* checked into
this repo, correcting an earlier wrong claim in this doc that it wasn't.
This document is `.claude/stages/data/data-stage-2.md` and should be cited
as `data-stage-2.md §N` in new code comments, matching that naming.

Stage 1's spec (§7) already made one decision directly relevant here:
**W&B tracking uses a single project (`qwen-coder-python-fim`), not a
separate one per stage — data-curation runs and training runs are
distinguished by `job_type` instead.** That convention was never actually
implemented for curation (only `src/training/train_lora.py` and
`safim_eval.py` call `wandb.init`/`wandb.log` today — `build_sample.py`
and `src/curation/*` have zero W&B instrumentation despite the spec calling
for `job_type="data-curation"`). §7 below covers what Stage 2 needs to
actually build to make curation/experiment progress visible.

## 1. Open decisions (not yet resolved — do not treat as settled)

- **Pilot winner (random vs. planned/"distributed")**: not chosen yet. The
  10k-sample pilot result doesn't automatically carry over — `experiment/`
  re-runs the comparison at full scale before `python_random_distributed_data/`
  is generated from whichever variant wins.
- **New repo name — RESOLVED (2026-08-15)**: `Rudra-G-23/the-stack-v3-python-fim-data`.
  `sample_filtered_data_10000/` and `experiment/{random_data,distributed_data}/`
  (this doc's §2 tree, minus the full-corpus folders) are implemented against
  this repo — `configs/data/stack_v3_filter.yaml`,
  `configs/data/fim_distribution.yaml`, `scripts/build_sample.py`,
  `scripts/generate_fim_variants.py`, `scripts/migrate_sample_to_new_repo.py`,
  and `notebooks/stack_v3_data_pull/*.ipynb`. Migrating the actual data and
  running the two generation notebooks is still manual (Rudra runs them with
  his own `HF_TOKEN`) — see
  `.claude/dicussions/2026-08-15-stage2-sample-migration.md`. The other three
  items below remain open; none of this touches them.
- **`python_random_distributed_data/` naming**: this folder will only ever
  hold one variant's data (whichever wins the pilot decision above), but its
  name says "random_distributed" regardless of which one wins. Rename it to
  match the winner (e.g. `python_planned_data/`) once chosen, or keep the
  generic name but state the actual variant prominently in its
  `metadata.json` — needs a decision, not silently left ambiguous.
- **"No cap" full-corpus scale vs. Kaggle's session model — needs explicit
  reconsideration before §5.1 starts, not after.** The Stack v3's Python
  subset is very likely tens of millions of files. Streamed with no cap,
  resumed manually across Kaggle's 12-hour ephemeral sessions, §5.1 alone
  could take many weeks-to-months of restarts before `python_filtered_data/`
  is even finished — before any GPU time is spent on the actual experiment.
  That's the real time/compute-waste risk, with three concrete options:
  1. Keep it uncapped, but move the filter+checkpoint stage (§5.1) off
     Kaggle notebooks onto a persistent long-running machine, since it's
     CPU-only anyway — Kaggle's session model fit the 10k pilot, not an
     open-ended run.
  2. Set a large-but-finite ceiling (`target.files` or `--max-gb`) sized
     backward from how many FIM examples the final model actually needs,
     rather than "as much as exists."
  3. Explicitly accept a multi-week, many-session Kaggle resume cycle —
     valid only if that timeline is genuinely acceptable.
  This document does not pick one for you — resolve it before §5.1 work
  starts.
- **`seen_hashes` persistence timing**: whether this ships before the first
  full-corpus streaming run starts, or is retrofitted after. Likely resolved
  together with the Kaggle-vs-persistent-machine decision above.

## 2. Repo / folder structure

```
the-stack-v3-python-fim-data/            (HF dataset repo — Rudra-G-23/the-stack-v3-python-fim-data)
├── sample_filtered_data_10000/          # Stage 1 pilot sample — IMPLEMENTED 2026-08-15
│   ├── data/                            #   chunk_*.parquet, migrated from the old flat repo
│   ├── checkpoints/                     #   checkpoint_*.json (added to this tree — the original
│   │                                    #   sketch omitted it; this is where real resume state lives)
│   └── metadata.json
├── checkpoint/                          # full-corpus streaming resume state — NOT YET BUILT (§1)
│   ├── checkpoint_*.json                # per-chunk resume markers (same shape as Stage 1)
│   └── seen_hashes.*                    # persisted exact-dedup hash set (new, see §3)
├── experiment/                          # 10k-sample-scale random-vs-planned generation —
│   │                                    # IMPLEMENTED 2026-08-15 (full-scale re-run over
│   │                                    # python_filtered_data/ is still §1/§5.3, NOT this)
│   ├── random_data/
│   │   ├── data/
│   │   ├── checkpoints/                 #   empty — single-pass generation, nothing to resume
│   │   └── metadata.json
│   └── distributed_data/                # "distributed" = the planned/fixed-percentage variant
│       ├── data/
│       ├── checkpoints/                 #   empty, same reason as above
│       └── metadata.json
├── python_filtered_data/                # full-corpus curated pool (no 10k cap) — NOT YET BUILT
│   ├── data/
│   └── metadata.json
├── python_random_distributed_data/      # winning variant's FIM examples, scaled to full pool
│   ├── data/                            # winner TBD — see §1 — NOT YET BUILT
│   └── metadata.json
└── model_data/                          # final train-ready split — NOT YET BUILT
    ├── train/
    ├── test/
    ├── validate/
    └── metadata.json
```

**Per-folder description:**

- `sample_filtered_data_10000/` — Stage 1's original 10k pilot output,
  copied in as a historical reference point; not regenerated by Stage 2.
- `checkpoint/` — resumable state for the full-corpus streaming run.
  Replaces Stage 1's flat `checkpoint_*.json`-at-repo-root layout. Produced
  and consumed by `src/curation/checkpoint.py` (extended, see §3).
- `experiment/random_data/`, `experiment/distributed_data/` — full-scale
  re-run of Stage 1's pilot comparison, produced by
  `src/fim/distribution_random.py` / `distribution_planned.py` against the
  full `python_filtered_data/` pool instead of the 10k sample.
- `python_filtered_data/` — the full curated Python file pool: every file
  that survived all six filter stages, no cap. Produced by
  `scripts/build_sample.py` once its `target.files` cap is removed.
- `python_random_distributed_data/` — the FIM training examples generated
  by whichever variant wins the `experiment/` comparison, run at full scale
  over `python_filtered_data/`. Not populated until §1's winner decision is
  made.
- `model_data/` — final `train/`/`test/`/`validate/` split of
  `python_random_distributed_data/`, ready for `src/training/train_lora.py`.
  No splitter exists yet — this is new code (§5).

## 3. `metadata.json` schema (one per folder, not per-row, not `.jsonl`)

Each folder's `metadata.json` explains *why* that folder's data looks the
way it does — an aggregate reasoning document, not a second copy of the
data. Per-row provenance (content hash, source shard/row, bucket/span-type)
already lives on every row in `curation_metadata` inside the parquet files
themselves (see `src/curation/stream_source.py`'s `build_curated_record`)
— `metadata.json` must not duplicate that per-row, even bounded to "kept
rows only." At real scale, "kept rows" for `python_filtered_data/` is still
easily hundreds of thousands of entries; repeating them in a JSON array
defeats the point of keeping this file light and turns it back into a
second dataset copy. `metadata.json` stays aggregate-only:

```jsonc
{
  "folder": "python_filtered_data",
  "stage": "full-corpus filter",
  "generated_at": "<ISO-8601 timestamp>",
  "source": { "dataset": "HuggingFaceCode/stack-v3-train", "split": "train" },

  "counts": {
    "raw_files_scanned": 0,
    "files_kept": 0,
    "reject_by_stage": {
      "language_reject": 0,
      "vendor_reject": 0,
      "license_reject": 0,
      "quality_reject": 0,
      "secret_reject": 0,
      "exact_dup_reject": 0
    },
    // one level deeper where cheap:
    // license_type is already threaded through on reject in FilterResult
    // (src/curation/stream_source.py) — free.
    "license_reject_by_type": { "copyleft": 0, "unknown": 0 },
    // quality_filters.py's evaluate_quality() already computes a specific
    // reason (ast_invalid / size_out_of_bounds / binary /
    // minified_or_generated), but apply_filter_pipeline() currently
    // discards it, only keeping the boolean — needs a small, real fix to
    // thread QualityDecision.reason into FilterResult before this is free.
    "quality_reject_by_reason": { "ast_invalid": 0, "size_out_of_bounds": 0, "binary": 0, "minified_or_generated": 0 }
  },

  // aggregate distribution over kept rows — counts per bucket/span-type,
  // NOT one entry per row (that provenance already lives in each row's own
  // curation_metadata / bucket assignment inside the data files)
  "kept_row_distribution": {
    "by_bucket_or_span_type": { "line": 0, "expression": 0, "statement": 0 }
  },

  // small illustrative sample per reject reason — qualitative spot-check /
  // paper appendix material, deliberately bounded, NOT an exhaustive log
  "rejected_examples_sample": {
    "quality_reject": [ { "content_id": "...", "reason": "minified_or_generated", "snippet": "..." } ]
  }
}
```

Fields adapt per folder (`experiment/*` metadata describes span-type bucket
composition; `model_data/metadata.json` describes split ratios, seed, and
row counts per split). The shared principle: aggregate counts and
distributions plus a bounded rejected-sample — never an unbounded per-row
listing, since that data already exists in the parquet rows themselves.

**Generation mechanics**: this must extend the existing incremental
aggregation pattern in `src/curation/checkpoint.py`
(`render_filter_report`/`write_filter_report`, which already aggregates
`filter_stats` cheaply across all checkpoints on demand), not invent a new
mechanism. Given the multi-session, resumable nature of §5.1, `metadata.json`
should be regenerated the same way `reports/stage1_filter_report.md` is
today — on demand from checkpoint history — so it stays currently accurate
without needing a single end-of-run computation.

## 4. Scale risk: dedup-set rebuild cost

`src/dedup/exact_dedup.py`'s `ExactDedup` is an in-memory `set()` of hex
SHA-256 strings. `src/curation/checkpoint.py`'s `rebuild_seen_hashes()`
reconstructs it on every resumed session by re-downloading and re-reading
**every previously uploaded chunk parquet file** from HF. At 10k files this
is instant; at full-corpus scale, spread across many Kaggle 12-hour
sessions, this rebuild cost grows linearly with total history collected so
far and will eventually dominate session startup time.

**Fix direction**: persist the hash set itself as an artifact in
`checkpoint/seen_hashes.*`, updated **per chunk** — the same granularity
`checkpoint_*.json` already uses, right after each chunk's parquet is
durably uploaded — not once per session. If it only flushed at session end,
a session killed by Kaggle's time limit or a crash would silently lose that
session's hash updates, reopening the door to re-admitted duplicates on the
next resume even though the corresponding chunk data was already uploaded.
Exact format (flat binary of digests vs. a compact set-serialization) and
whether this ships before or after the first full-corpus run starts is an
open decision (§1).

## 5. Pipeline stages

- **§5.1 — Full-corpus filter + checkpoint.** Extends
  `scripts/build_sample.py` / `src/curation/checkpoint.py`: remove the
  `target.files` cap, write to `python_filtered_data/` + `checkpoint/` in
  the new repo layout instead of flat repo root.
- **§5.2 — Dedup-set persistence.** New: `checkpoint/seen_hashes.*`,
  incremental update instead of full rebuild (§4).
- **§5.3 — Experiment re-run at full scale.** Extends
  `src/fim/distribution_random.py` / `distribution_planned.py` and
  `scripts/run_pilot_experiment.py` to run against `python_filtered_data/`
  and write to `experiment/random_data/` / `experiment/distributed_data/`.
- **§5.4 — Winner selection → `python_random_distributed_data/`.** Once §1's
  open decision is resolved, generate the winning variant's full FIM
  example set over the whole curated pool.
- **§5.5 — `model_data/` split.** New: train/test/validate splitter — no
  equivalent exists today in `src/training/` or `scripts/`. Needs a seed,
  split ratios, and a `metadata.json` documenting both plus row counts per
  split.

## 6. Notebook plan

One thin Kaggle notebook per stage, matching
`notebooks/kaggle/curate_and_checkpoint.ipynb`'s existing pattern (clone
repo → install CPU-only deps → HF token from Kaggle Secrets → call into
`scripts/` → display the generated report). All logic stays in `scripts/`
and `src/`; notebooks are controllers only.

1. **10k-sample filtering/migration — IMPLEMENTED**:
   `notebooks/stack_v3_data_pull/sample_filtered_data_10000.ipynb` (migrates
   the old flat-repo sample, then confirms it against the new repo/prefix via
   `scripts/build_sample.py`). Full-scale filtering (uncapped `target.files`
   over the whole corpus, extending `curate_and_checkpoint.ipynb`) is
   separate, still not built, still gated behind §1's scale decision.
2. **Random-variant experiment — IMPLEMENTED** (10k-sample scale):
   `notebooks/stack_v3_data_pull/random_data_10000.ipynb`, calls
   `scripts/generate_fim_variants.py --variant random`. The full-scale
   §5.3 re-run over `python_filtered_data/` is separate future work.
3. **Distributed-variant experiment — IMPLEMENTED** (10k-sample scale):
   `notebooks/stack_v3_data_pull/distributed_data_10000.ipynb`, calls
   `scripts/generate_fim_variants.py --variant planned`. Same full-scale
   caveat as #2.
4. **Model data split** — new notebook, calls the §5.5 splitter once §1's
   winner decision is made. Not started.

## 7. Monitoring — decided

**Single W&B project** (`qwen-coder-python-fim`), confirming Stage 1's
original decision (`data-stage-1.md` §7) rather than splitting curation and
training into separate projects. The reason to keep them together: the
whole point of tracking is correlating *which curated dataset* produced
*which training result* — split projects break that link into two UIs you'd
cross-reference by hand. Sliced instead by:
- `job_type`: `"data-curation"` | `"experiment-random"` | `"experiment-distributed"` | `"train"` | `"eval"`
- `group`: pipeline-run / dataset-version id, so W&B's UI clusters every
  run belonging to one execution together
- `tags`: variant name, seed — matches the existing convention in
  `run_pilot_experiment.py`'s `overrides["wandb"]["tags"]`

**§5.1 (full-corpus filter) must be one resumable run, not one run per
Kaggle session or per chunk.** A multi-week job resumed across dozens of
12-hour sessions needs `wandb.init(id=<fixed-id>, resume="allow")` so every
session's chunk logs land on the same continuous time series instead of
fragmenting into dozens of disconnected runs cluttering the project view.
Log, per chunk, at `step=cumulative_chunk_index`:
- Filter funnel: reject counts per stage (bar chart — clearer for
  comparing several reject reasons at a glance than a pie chart, same
  reasoning applies to §5.3's bucket distribution below)
- Cumulative files kept / bytes collected (line chart — pace + time-to-completion)
- Dedup rate: `duplicate_count / raw_files_scanned` this chunk, and cumulative

This is new code: neither `build_sample.py` nor `src/curation/*` call W&B
today (only `train_lora.py`/`safim_eval.py` do) despite the original spec
calling for `job_type="data-curation"`. Implementation should live as a
thin wrapper (e.g. `src/curation/wandb_logger.py`), called from
`build_sample.py`'s per-chunk loop right where `_write_and_upload_chunk`
already runs — instrumentation only, no filter logic changes.

**§5.3 (experiment re-run)** — logs both angles you asked for:
- Span-type bucket distribution per variant (bar chart, `random` vs
  `distributed` side by side) — logged once sampling finishes for each
  variant, from `distribution_random.py`/`distribution_planned.py`'s output.
- Training curves per variant — already implemented (`train_lora.py`,
  `safim_eval.py` log loss/SAFIM pass@1 to W&B); the only gap is consistent
  `group` tagging across the random and distributed runs so W&B overlays
  them on the same comparison panel automatically instead of you having to
  build that view by hand.

**Nothing else needs deciding before implementation** — the open items
left for this stage are §1's ones (pilot winner, repo name, scale-vs-Kaggle
decision), not monitoring shape.
