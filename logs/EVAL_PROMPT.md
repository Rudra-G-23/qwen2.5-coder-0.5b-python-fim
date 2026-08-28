# EVAL_PROMPT.md — Evaluation Stage: task prompt + repo audit + plan

> **How to use this file.** When your held-out evaluation dataset is ready, fill in
> **§1 (DATA — fill this in)** below, then paste this whole file back into Claude Code
> as the prompt. Everything else (audit, plan, constraints, phase order) is already
> resolved here so the execution session does **not** need to re-audit the repo.
> Start execution at **PHASE 2 (Plan confirmation)**.

---

## TOPIC

Build **one resumable, config-driven evaluation harness** that scores three models on
**one canonical fixed unseen evaluation dataset**, with a **single inference pass per
(model, task)**, and derives three analyses from one per-example results table:

1. **Overall pass@1** per model.
2. **pass@1 by `fim_type`** (the canonical AST span-type set — see §3).
3. **pass@1 by target-length bucket** (token count of `middle` under the base
   Qwen tokenizer; bucket edges from config).

Plus: Wilson 95% CIs on every pass@1, W&B monitoring with the repo's existing
deterministic run-ID convention, three W&B charts, and artifacts pushed to **both**
W&B and Hugging Face. Resume must be idempotent at the `(model_variant, task_id)`
granularity because Kaggle sessions terminate.

**This is evaluation only. Do NOT retrain, change LoRA settings, regenerate training
data, or rename FIM categories.**

---

## 1. DATA — FILL THIS IN before using this file as a prompt

### 1a. The canonical evaluation dataset (already exists on HF — provide its coordinates)

| Field | Value |
|---|---|
| HF dataset repo | `<FILL IN e.g. Rudra-G-23/the-stack-v3-python-fim-data>` |
| Subfolder / path to parquet(s) | `<FILL IN e.g. experiment/eval_data/data/*.parquet>` |
| Revision / commit SHA | `<FILL IN — pin it; do not use a moving "main">` |
| Config name / filename | `<FILL IN e.g. fim_eval.parquet>` |
| Number of examples | `<FILL IN e.g. ~1000>` |
| Split name (if a `datasets` split) | `<FILL IN or "n/a — raw parquet">` |

### 1b. Confirm the schema (tick what is actually present)

Expected per-example fields (from your training FIM schema — see §3):

- [ ] `fim_type` (string; canonical set in §3)
- [ ] `content_id` (string; source-file identifier for leakage detection)
- [ ] `prefix` (string)
- [ ] `suffix` (string)
- [ ] `middle` (string; the target span)
- [ ] `ast_depth` (int)
- [ ] `nesting_depth` (int)
- [ ] `node_type` (string)
- [ ] `enclosing_function` (string | null)
- [ ] `enclosing_class` (string | null)
- [ ] `structural_position` (string)
- [ ] `span_line_count` (int)
- [ ] a stable per-row id — if none, the harness derives `task_id = sha1(content_id|prefix|middle|suffix)[:16]`
- [ ] **unit tests / executable harness?**  `<yes / no>`  ← decides the metric, see §2

### 1c. Leakage-check inputs (training data this eval set must NOT overlap)

The eval `content_id` set must have **zero intersection** with the training
`content_id` sets. Known training datasets:

| Variant | HF repo | Path | Approx rows |
|---|---|---|---|
| random (seed42 training data) | `Rudra-G-23/the-stack-v3-python-fim-data` | `experiment/random_data/data/fim_random.parquet` | 20,000 |
| distributed (seed42 training data) | `Rudra-G-23/the-stack-v3-python-fim-data` | `experiment/distributed_data/data/fim_distributed.parquet` | 20,000 |
| (both sampled from) curated pool | `Rudra-G-23/the-stack-v3-python-fim-data` | `sample_filtered_data_10000/data/*.parquet` | ~10,000 files |

- Confirm the eval set was built from files **excluded** from the 10k curated pool
  used for the two training variants, OR at minimum that its `content_id`s are
  disjoint from both training parquets. `<confirm / explain>`
- If the eval set was drawn from the *same* 10k pool, say so — the harness must
  then report the overlap loudly and NOT claim "unseen".

### 1d. Metric decision (REQUIRED — see §2 for why this is a real fork)

Choose one:

- [ ] **(A)** Canonical eval runs **only** on this held-out set. `pass_at_1` :=
      exact-match of generated `middle` vs. reference `middle` (whitespace-stripped);
      `edit_similarity` kept as a secondary column. Real SAFIM
      (`gonglinyuan/safim`) is **not** part of this workflow.
- [ ] **(C)** Keep running **real SAFIM execution-based pass@1** as the headline
      number via the existing `src/training/safim_eval.py` (unchanged), **and**
      additionally run this held-out set for the `fim_type` / target-length
      breakdowns + leakage check. Two datasets, one inference pass each.
- [ ] **(B — only if 1b says the eval set is executable)** `pass_at_1` := SAFIM-style
      execution pass@1 against the eval set's own bundled tests, reusing
      `passes_unit_tests` / `_run_program` from `safim_eval.py`.

**Default if left unticked:** (A).

### 1e. Models to evaluate (already trained — confirm paths)

| Variant key | Base model | Adapter (HF repo + subfolder) | Training seed |
|---|---|---|---|
| `base` | `Qwen/Qwen2.5-Coder-0.5B` | *(none)* | — |
| `random_seed42` | `Qwen/Qwen2.5-Coder-0.5B` | `Rudra-G-23/qwen2.5-coder-0.5b-python-fim` @ `experiment/random_model/seed42` | 42 |
| `distributed_seed42` | `Qwen/Qwen2.5-Coder-0.5B` | `Rudra-G-23/qwen2.5-coder-0.5b-python-fim` @ `experiment/distributed_model/seed42` | 42 |

Adapter revision/commit to pin: `<FILL IN or "main">`.
Future models (`random_seed1`, `distributed_seed123`, …) must be addable **by
config only**, no code change.

### 1f. W&B / HF destinations (confirm)

| Purpose | Value |
|---|---|
| W&B project (training/eval share it) | `521er1007-national-institute-of-technology-rourkela/qwen-coder-python-fim` |
| W&B `job_type` for this work | `eval` |
| HF repo for eval artifacts | `<FILL IN — default: Rudra-G-23/qwen2.5-coder-0.5b-python-fim, subfolder `evaluation/<eval_run_id>/`>` |
| Eval dataset version tag (for run naming / provenance) | `<FILL IN e.g. evalv1>` |

---

## 2. Why the metric is a real fork (context for whoever executes this)

`src/training/safim_eval.py`'s pass@1 is **execution-based**: it substitutes the
model completion into `eval_prompt`, runs it as a sandboxed subprocess, and checks
stdout against `unit_tests` shipped by `gonglinyuan/safim`. That benchmark has
**no `content_id`** and its span types are SAFIM's own `control` / `api` / `block`
configs — **not** `line` / `expression` / `statement` / … .

A held-out slice of *your* FIM dataset has `content_id`, the canonical `fim_type`
set, and `middle`, but **no unit tests**, so execution pass@1 cannot run on it.
Hence: option (A) redefines `pass_at_1` as exact-match on that set; option (C)
keeps real SAFIM as a separate headline number and adds your set for the
breakdowns. Pick in §1d.

---

## 3. Canonical `fim_type` set — DO NOT rename or invent

From `src/fim/ast_bucketer.py` and `configs/data/fim_distribution.yaml`
(`planned_distribution` keys). Hyphenated, exactly:

```
line
expression
statement
block
function-body
method-body
class-level
api-call
```

`src/fim/ast_bucketer.py` emits these as `Span.span_type` and
`build_fim_record()` writes them to the `fim_type` field. `structural_position`
is a *different* field with values `module_level` / `class_body` /
`function_body` (underscored) — do not confuse the two.
The eval harness must **discover the actual category set from the eval dataset at
runtime** and group by whatever is present; the list above is the expected set,
not a hardcode. Empty buckets must render as a row with `total = 0`, not be dropped.

---

## 4. PHASE 1 — Repository audit (DONE — findings below, no need to redo)

### 4a. Evaluation code that already exists

| File | What it does | Reuse for this task |
|---|---|---|
| `src/training/evaluate.py` | Base-vs-LoRA on a held-out **FIM JSONL** (`prefix`/`suffix`/`middle`). Metrics: `exact_match`, `edit_similarity` (pure functions). `load_model(base, adapter_path=None)` → unsloth fast-path else HF+PEFT fallback. `generate_completion(model, tok, prefix, suffix, max_new_tokens=128)` → greedy FIM decode with `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`. `evaluate_model()` per-example loop. `run_evaluation()` orchestrator → `results/metrics.csv`, plots, optional `wandb_run`. `plot_comparison()` grouped bar. | **Reuse verbatim:** `load_model`, `generate_completion`, `exact_match`, `edit_similarity`. These are the inference + scoring primitives. Do **not** fork them. |
| `src/training/safim_eval.py` | Execution-based pass@1 on `gonglinyuan/safim`. `load_safim_python(config, max_samples, seed)` splits `eval_prompt` on `{{completion}}`. `_limit_resources` / `_run_program` / `passes_unit_tests` = sandboxed exec. `evaluate_pass_at_1(model, tok, records, label, wandb_run, log_every)` → `[{model, task_id, "pass@1"}]` + live W&B progress logging. `run_safim_evaluation()` (Base vs one LoRA-FT), `run_safim_evaluation_by_type()` (by SAFIM config, loads each model once, loops configs). `plot_safim_comparison`, `plot_safim_by_type` (grouped bar). Pilot aggregation: `load_seed_pass_at_1`, `aggregate_pilot_results` (range-overlap verdict, mean/std/min/max), `write_pilot_comparison_report`, `render_seed_variance_table`, `plot_pilot_comparison`. | **Reuse if metric = (C) or (B):** `passes_unit_tests`, `_run_program`, `_limit_resources`, `evaluate_pass_at_1`'s loop shape, `load_safim_python`. **Model-loop pattern to copy:** load each model once, run all tasks, `del model` + `torch.cuda.empty_cache()`. **Do not modify** `run_safim_evaluation*` if (C) — they stay the untouched SAFIM headline path. |
| `src/training/report_tables.py` | `render_hyperparameter_table`, `render_compute_cost_table`, **`log_plots_artifact(wandb_run, plot_paths, *csv_paths, name=)`** — bundles PNGs+CSVs into one versioned `wandb.Artifact(type="eval-plots")`. `fetch_wandb_run_durations`. Pure functions, unit-tested, no GPU. | **Reuse:** `log_plots_artifact` for the W&B artifact bundle. Add eval summary-table renderers here (same module, same pure-function style). |
| `scripts/run_pilot_experiment.py` | Trains each `(variant, seed)` then calls `run_safim_evaluation` on its adapter. `materialize_variant_jsonl` (HF parquet → local JSONL with `text`). `_subfolder_already_pushed(hf_repo, subfolder, token)` = whole-run resume via `HfApi.file_exists`. `--resume-run-id`. | **Reference for:** HF parquet download pattern, module-level `HfApi`/`hf_hub_download`/`snapshot_download` refs (monkeypatchable), `(variant, seed)`-level resume idea → generalise to `(model_variant, task_id)`. |
| `scripts/analyze_pilot_results.py` | Thin CLI over `write_pilot_comparison_report`. | Pattern for the offline aggregation CLI. |

### 4b. W&B / run-ID / config infrastructure

- **Deterministic run IDs** live in `src/training/train_lora.py`:
  - `build_run_id(cfg)` → `{model_slug}-lora-r{r}-e{epochs}-ds{dataset_version}-{YYYYMMDD}-a{attempt}`
    (e.g. `qwen05b-lora-r16-e1-dsrandom-20260828-a42`).
  - `_stable_wandb_id(run_id)` → same string with the `YYYYMMDD` component dropped;
    used as the W&B `id` so a resume on a later UTC day reattaches to the SAME run
    (`resume="allow"`, or `"must"` for an explicit paste).
  - `init_wandb(cfg, run_id, explicit_resume=False)` → graceful `None` if
    `WANDB_API_KEY` unset; splits `"entity/project"`; flattens config; sets
    `tags`, `group`, `notes`; `weave.init()` inside the run.
  - `_deep_merge(base, overrides)` → recursive dict merge (pilot layering).
- **The eval harness must reuse this exact convention.** Build an eval run ID like
  `qwen05b-eval-ds{eval_dataset_version}-{YYYYMMDD}-a{attempt}` via a small
  `build_eval_run_id()` that mirrors `build_run_id` (or extend `build_run_id` to
  take a `method`/`stage` arg — prefer the smallest change). Reuse
  `_stable_wandb_id` verbatim for resume. **One logical W&B run for the whole eval
  sweep**, resumed across sessions, not one run per model or per session.
- **Config convention:** every hyperparameter/path in YAML under `configs/`; scripts
  read config, never hardcode. `configs/training/lora.yaml` (base),
  `configs/training/pilot.yaml` (layers overrides via `_deep_merge`),
  `configs/data/fim_distribution.yaml`. W&B project string is
  `"entity/project"` inside the YAML.

### 4c. Resume / checkpoint infrastructure (patterns to mirror, none reusable as-is)

- `src/training/train_lora.py`: `HFCheckpointCallback` mirrors Trainer checkpoints
  to `checkpoints/{run_id}/` in the HF **model** repo; `find_resume_checkpoint`
  (date-tolerant), `download_resume_checkpoint`, `_cleanup_hf_checkpoints`,
  `_latest_pointer_path` (a `latest_checkpoint.json` pointer file).
- `scripts/run_pilot_experiment.py`: `_subfolder_already_pushed` — resume unit is a
  whole `(variant, seed)` run, checked by file existence on HF default branch.
- **Gap:** there is **no per-example / `(model, task_id)` eval resume anywhere.**
  This task must add it. Follow the same shape: a append-only `results.jsonl`
  (or parquet + progress json) that is the source of truth; a `latest` pointer;
  periodic flush to W&B artifact + (less often) HF; on restart, read completed
  `(model_variant, task_id)` pairs and skip them.

### 4d. Datasets / models on Hugging Face (from configs)

- **Data repo:** `Rudra-G-23/the-stack-v3-python-fim-data`
  - `sample_filtered_data_10000/data/*.parquet` — curated Python file pool
    (`content_id`, `content`), source for both training variants.
  - `experiment/random_data/data/fim_random.parquet` — 20k, random sampling.
  - `experiment/distributed_data/data/fim_distributed.parquet` — 20k, planned
    ("distributed") fixed-percentage sampling.
  - per-folder `metadata.json` with `counts.by_fim_type`.
- **Model repo:** `Rudra-G-23/qwen2.5-coder-0.5b-python-fim`
  - `experiment/random_model/seed42/` — LoRA adapter + `metadata.json` + `hyperparameter_table.md`.
  - `experiment/distributed_model/seed42/` — same.
  - each subfolder self-describes (`metadata.json`: run_id, base_model, lora r/alpha/dropout, seed, epochs, dataset_version, train_loss_final, trained_at_utc).
- Training FIM-type counts (context only — **NOT** an eval distribution, do not
  hardcode): random ≈ {line 8534, statement 4835, expression 3561, api-call 2128,
  block 347, method-body 226, function-body 201, class-level 168}; distributed ≈
  {line 4000, statement 4000, expression 3000, block 3000, function-body 3000,
  method-body 2000, class-level 600, api-call 400}.

### 4e. Notebooks (thin controllers — the rule: "notebooks contain no real logic")

- `notebooks/experiments/random.ipynb`, `planned.ipynb` — pilot train controllers:
  clone repo at `BRANCH`/`COMMIT` → GPU check → `pip install` → HF+W&B secrets from
  `kaggle_secrets.UserSecretsClient` → `subprocess.run([... run_pilot_experiment.py ...])`.
- `notebooks/kaggle_wandb.ipynb` — full train+eval: patches `lora.yaml`
  `attempt`/`dataset_version`, `train(finish_wandb=False)`, `run_evaluation(wandb_run=...)`,
  `run_safim_evaluation(wandb_run=...)`, finishes run, displays plots.
- `notebooks/kaggle/curate_and_checkpoint.ipynb` — Stage-1 curation controller.
- **This task adds ONE new thin notebook:** `notebooks/kaggle_eval.ipynb`
  (clone → GPU check → install → secrets → `python scripts/run_evaluation_suite.py`
  → display final tables/charts). No model paths edited in cells — all from config
  + env. Must be safe to rerun after a Kaggle disconnect (resume).

### 4f. Tests convention

- `local_tests/`, `pytest`, `ruff check .`. `sys.path.insert(0, <repo root>)` at top
  of each test. **No GPU, no network.** HF Hub calls monkeypatched (see
  `test_run_pilot_experiment.py`'s `FakeHfApi`, `test_train_lora_resume.py`).
  `test_safim_eval.py` covers only the CSV aggregation, never `run_safim_evaluation`
  (needs a model). Mock model inference the same way here.
- `pyproject.toml`: Python ≥ 3.13, deps incl. `datasets`, `transformers`, `trl`,
  `peft`, `wandb`, `weave`, `pandas`, `pyarrow`(implied), `matplotlib`, `seaborn`,
  `pyyaml`, `pytest`, `ruff`.

### 4g. Docs worth reading before large changes

- `.claude/stages/data_stages/data-stage-1.md` §6 (span-type set), §7 (one shared
  repo, subfolders, W&B job_type/group/tags, deterministic run IDs), §8 (pilot eval:
  SAFIM primary, seed-vs-noise, treat as preliminary).
- `.claude/stages/data_stages/data-stage-2.md` §7 (single W&B project;
  `job_type` ∈ {data-curation, experiment-random, experiment-distributed, train, **eval**};
  `group` = pipeline-run id; bar charts over pie).
- `CLAUDE.md` (core rule: notebooks are thin; config-driven; W&B no-ops without key).
- `reports/graphs_and_experiment_setup.md` (what plots/artifacts already exist).

---

## 5. PHASE 2 — Plan (proposed; confirm or adjust at execution time)

### 5a. Reuse verbatim (no edits)
- `evaluate.load_model`, `evaluate.generate_completion`, `evaluate.exact_match`,
  `evaluate.edit_similarity`.
- `report_tables.log_plots_artifact`.
- `train_lora._deep_merge`, `train_lora._stable_wandb_id`, `train_lora.init_wandb`
  (call with an eval cfg), `train_lora._file_sha256`, `_resolve_model_revision`.
- If metric (C): `safim_eval.run_safim_evaluation` untouched as the headline path.

### 5b. New files
| File | Purpose |
|---|---|
| `configs/eval/eval.yaml` | Single source of truth: eval dataset (repo/subfolder/revision/config), `max_examples`, `models:` list (variant key → base + adapter repo/subfolder/revision + training_seed), `target_length_buckets: [[1,16],[17,32],[33,64],[65,128],[129,null]]`, `metrics:` toggles, `generation:` (max_new_tokens, greedy), `checkpoint_every: 25`, `hf_push_every` / major-checkpoint policy, `wandb:` (project, job_type: eval, tags, group), `huggingface:` (repo, subfolder template), `eval_dataset_version`, `attempt`. |
| `src/training/eval_harness.py` | Core, GPU-touching but import-safe. `load_eval_dataset(cfg)` (HF parquet/`datasets`, pinned revision, → list of dicts). `assign_length_bucket(n_tokens, buckets)`. `count_target_tokens(tokenizer, middle)` = `len(tok.encode(middle, add_special_tokens=False))`. `score_example(pred, ref, unit_tests=None)` → dict with `pass_at_1` (+ `edit_similarity`). `run_model_over_tasks(model, tok, tasks, variant, seed, sink, log_every, wandb_run)` — single pass, writes each result through `sink` immediately. `build_eval_run_id(cfg)` / reuse `_stable_wandb_id`. Leakage check `check_leakage(eval_ids, train_id_sets)`. |
| `src/training/eval_aggregate.py` | Pure, no GPU. From the per-example table (parquet/DataFrame): `overall_pass_at_1(df)`, `pass_at_1_by(df, key)` for `key ∈ {fim_type, target_length_bucket}`, `wilson_ci(passed, total, z=1.96)`, `summary_json(df, cfg)`, markdown renderers (`render_overall_table`, `render_by_fim_type_table`, `render_by_length_table`) — mirror `report_tables.py` style. Bucket order fixed by config, empty buckets kept. |
| `src/training/eval_persistence.py` | Pure-ish, no GPU. `ResultSink` writing append-only `results.jsonl`; `load_completed_pairs(path)` → `set[(variant, task_id)]`; dedup rule (keep first occurrence deterministically); `to_parquet(jsonl) -> results.parquet`; `write_resolved_config(cfg)`; local↔W&B-artifact↔HF sync helpers with **try/except that never drops local progress** (log failure, continue). |
| `scripts/run_evaluation_suite.py` | Thin CLI orchestrator. Load `configs/eval/eval.yaml` (+ CLI overrides via `_deep_merge`). Resolve models. `init_wandb(eval_cfg, eval_run_id)` — one run, `resume="allow"`. Discover prior progress (local, else pull latest W&B/HF artifact). Leakage check → abort/loudly-warn per §1c. For each model: `load_model` once → `run_model_over_tasks` skipping completed pairs → `del` + `empty_cache`. Every `checkpoint_every`: flush local + `wandb_run.log` running metrics + update W&B artifact. At major checkpoints / end: push `results.parquet` + `results.csv` + `summary.json` + `resolved_eval_config.yaml` + charts to W&B **and** HF. Build 3 charts (`wandb.plot.bar` / grouped bar PNGs). `--dry-run N` limits tasks; `--models a,b`. |
| `scripts/analyze_evaluation_results.py` | Offline: read `results.parquet`, print/write the 3 tables + Wilson CIs + `summary.json`, regenerate charts. Mirrors `analyze_pilot_results.py`. |
| `notebooks/kaggle_eval.ipynb` | Thin controller (see §4e). |
| `local_tests/test_eval_harness.py`, `test_eval_aggregate.py`, `test_eval_persistence.py` | See PHASE 4. |

### 5c. Minimal edits to existing files
- `src/training/report_tables.py`: add eval summary-table renderers *here* (keep one
  home for pure table functions) **or** put them in `eval_aggregate.py` — pick one,
  don't split. If `log_plots_artifact` needs a `type=` other than `"eval-plots"`,
  parameterise rather than fork.
- `src/training/train_lora.py`: only if `build_run_id` is extended to take a
  `stage`/`method` param — smallest possible change, keep `build_run_id(cfg)` calls
  working unchanged (default `method="lora"`).
- `README.md` / `CLAUDE.md` "Commands": add the eval command + notebook.
- **No other files change.** No training code paths touched.

### 5d. Single-inference-pass architecture (hard requirement)
```
configs/eval/eval.yaml
        │
        ▼
scripts/run_evaluation_suite.py ──► init_wandb (one resumable run)
        │
        ├─ load_eval_dataset (pinned revision)         ─┐
        ├─ leakage check vs training content_id sets    │  once
        ▼                                               ─┘
for variant in models:                    ← skip (variant,task_id) already in results.jsonl
    load_model(base, adapter) once
    for task in eval_tasks:
        pred = generate_completion(...)            ← ONE generation per (model,task)
        row  = { task_id, model_variant, training_seed, fim_type,
                 target_tokens, target_length_bucket, pass_at_1,
                 edit_similarity, generated_output, reference_middle, content_id }
        sink.write(row)                            ← persisted immediately (append)
    del model; torch.cuda.empty_cache()
        │
        ▼
results.jsonl ──► results.parquet ──┬─► overall pass@1  (+ Wilson CI)
                                    ├─► pass@1 by fim_type  (+ Wilson CI)   [group-by, no re-inference]
                                    └─► pass@1 by target_length_bucket (+ Wilson CI)
```

### 5e. W&B charts (exactly three, no more)
1. **Overall pass@1 by model** — bar. Series: Base / Random-LoRA / Distributed-LoRA.
2. **pass@1 by FIM type** — grouped bar. X = `fim_type` (canonical set order), series = 3 models.
3. **pass@1 by target length** — grouped bar (or ordered line). X = `1-16,17-32,33-64,65-128,129+`, series = 3 models.
Log config, git SHA (`git rev-parse HEAD`), model/adapter source + HF revision,
dataset source + revision, `eval_dataset_version`, training seed per model, eval
run ID, generation config, the 3 aggregate metric groups, and `completed/total`
progress. W&B auto-name must not be the only identifier — set `name=eval_run_id`.

### 5f. Artifacts (BOTH W&B and HF)
`results.parquet` (preferred), `results.csv` (optional), `summary.json`,
`resolved_eval_config.yaml`, chart PNGs + underlying data. W&B: versioned
`wandb.Artifact` via `log_plots_artifact` + a dedicated results artifact. HF:
`evaluation/<eval_run_id>/` subfolder in the model repo (or the repo named in §1f).

### 5g. Resume (idempotent, `(model_variant, task_id)` unit)
- `results.jsonl` append-only = source of truth. Flush interval `checkpoint_every`
  (config, default 25): save local → `wandb_run.log(running metrics)` → update W&B
  progress artifact. Less often / major checkpoints: push to HF.
- On restart: load config → find prior eval run (same `_stable_wandb_id`) →
  download latest progress artifact if local missing → read completed pairs → skip
  them → continue → reuse the same W&B run (`resume="allow"`).
- Duplicate rows for the same `(variant, task_id)`: resolve deterministically
  (keep first by file order; log the collision).
- W&B upload failure ⇒ keep local + keep going. HF upload failure ⇒ keep local +
  W&B + keep going. Telemetry loss never costs inference results. Log every failure.

---

## 6. PHASE 3 — Implement
Smallest coherent implementation of §5. Match surrounding code style (module
docstring citing the relevant spec §, `from __future__ import annotations`, pure
functions kept GPU-free, module-level HF refs for monkeypatching). No unrelated
refactors. Do not touch training code paths. Do not add tokens/sec, GPU-util,
latency, throughput, or memory benchmarking.

## 7. PHASE 4 — Test (`pytest local_tests/ -v` + `ruff check .`, no GPU/network)
At minimum:
1. `count_target_tokens` — exact token count for known strings (mock tokenizer with
   a deterministic `encode`).
2. `assign_length_bucket` — boundaries (16/17, 32/33, 64/65, 128/129, open top),
   config-driven edges, `null` upper bound.
3. `pass_at_1_by(df, "fim_type")` — grouping + counts on a fabricated table incl. a
   `fim_type` with zero rows (empty-bucket row with `total=0`, not dropped).
4. `overall_pass_at_1` + `wilson_ci` — known passed/total → known CI (check against
   a hand value, e.g. 7/10 → ~[0.397, 0.892]).
5. Resume: `load_completed_pairs` skips done `(variant, task_id)`; a second harness
   pass over a partially-filled `results.jsonl` runs only the remainder (mock
   inference; assert call count).
6. Duplicate prevention: two rows same `(variant, task_id)` → aggregation counts it
   once, deterministically.
7. Partial-recovery: truncated/half-written last JSONL line tolerated (skip, warn).
8. Same tasks across models: assert the task_id set evaluated for each model is
   identical.
9. Leakage check: overlapping `content_id` → returns non-empty intersection and the
   orchestrator refuses to label the run "unseen".
10. `summary_json` shape/keys stable; `write_resolved_config` round-trips YAML.
Mock model inference (a fake object whose `generate` returns canned ids, or
monkeypatch `generate_completion`). Mock `HfApi` / `wandb` like existing tests.

## 8. PHASE 5 — Resume test (simulated, no GPU)
Drive `scripts/run_evaluation_suite.py` with mocked inference + mocked HF/W&B:
run 5 tasks → kill → restart → assert those 5 are skipped and the rest run → final
`results.parquet` has exactly `n_models × n_tasks` unique rows, no dupes. Document
the real Kaggle interruption drill (run, stop the session, rerun the notebook,
confirm skip) as the manual acceptance step.

## 9. PHASE 6 — Review (self-review the diff before final report)
Check for: duplicated logic vs. `evaluate.py` / `safim_eval.py`; any second
`fim_type` definition; hardcoded paths / model IDs / bucket edges (must be config);
leakage-check actually wired in and blocking; the three analyses all derived from
ONE results table (no re-inference); persistence writes before telemetry; notebook
has zero real logic; Kaggle-compatible (no local-only paths, `kaggle_secrets`,
`subprocess` into `scripts/`); W&B run is one resumable run with the deterministic
ID; no perf-benchmarking crept in; `ruff` clean; all tests green.

## 10. PHASE 7 — Final report (deliver to the user)
1. What already existed (reference §4). 2. What was added (files + why each).
3. What files changed (diff summary). 4. Exact command / notebook flow to run the
eval (Kaggle: which secrets, which notebook, which cell; local dry-run command).
5. Where W&B artifacts appear (project URL, run name, artifact names, the 3 charts).
6. Where HF artifacts appear (repo + `evaluation/<eval_run_id>/` path).
7. How to resume after Kaggle termination (rerun the same notebook; what it skips;
which W&B run it reattaches to). 8. Remaining limitations (e.g. metric caveat from
§2, CI interpretation with small n, seed coverage = 1).
**State honestly what was and wasn't executed** (no GPU/tokens in the dev env ⇒
unit + simulated-resume tests run; real 1000-example run + real interruption drill
are the user's Kaggle step).

---

## 11. Multi-agent execution (requested)

Orchestrator = main session, owns integration, the diff, and the final report.
Do **not** let agents create competing implementations or edit the same file
concurrently. Suggested split (adjust to what's available):

- **Agent A — audit refresh / dataset+schema verification.** Confirm §4 still holds
  at the pinned commit; inspect the eval dataset from §1 (schema, row count,
  `content_id` presence); verify the two training parquets' `content_id` sets for
  the leakage baseline. Read-only. Reports facts, edits nothing.
- **Agent B — evaluation design review.** Verify exactly how the chosen metric
  (§1d) is computed; confirm ANALYSIS 2/3 can be pure group-bys of one table;
  check for leakage and eval consistency (same task set per model); sanity-check
  Wilson CI math. Read-only.
- **Agent C — implementation + tests.** Writes `configs/eval/eval.yaml`,
  `src/training/eval_harness.py`, `eval_aggregate.py`, `eval_persistence.py`,
  `scripts/run_evaluation_suite.py`, `scripts/analyze_evaluation_results.py`,
  `notebooks/kaggle_eval.ipynb`, and the three test files. Runs `ruff` + `pytest`.
  Owns all edits.
- Orchestrator integrates C's work using A's and B's findings, runs PHASE 5/6,
  writes PHASE 7.

If isolated worktrees/branches are available, give Agent C its own; otherwise run
A and B first (read-only, parallel), then C alone.

---

## 12. Non-goals (do not do these)
Retrain / change LoRA settings / redesign training / regenerate training data /
invent or rename FIM categories / hardcode the random-vs-distributed training
counts as an eval distribution / build a separate eval set per model / run separate
inference per analysis / replace working SAFIM logic without cause / add
tokens-per-sec, samples-per-sec, steps-per-sec, GPU-util, latency, memory, or
throughput benchmarking / refactor unrelated code / create a second W&B system or
a second run-ID system.
