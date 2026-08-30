# `safim_eval_1000` — Code Reference (Base vs Random-LoRA vs Distributed-LoRA)

Extracted verbatim from the repo so the eval logic can be reviewed in one place. Covers:

1. [The evaluation function that computes `exact_match` and `edit_similarity`](#1-evaluation-metrics--per-model-loop)
2. [The generation / inference configuration](#2-generation--inference-configuration)
3. [~10–20 comparison rows (`fim_type`, `middle`, Base / Random / Distributed output)](#3-per-example-comparison-rows)
4. [The code that constructs the Qwen FIM prompt](#4-qwen-fim-prompt-construction)

All line numbers are against the working tree at the time of writing (branch `feat/data`).

---

## 1. Evaluation metrics + per-model loop

### 1a. Metric definitions — `src/training/evaluate.py:126-154`

```python
# ── Metrics ───────────────────────────────────────────────────────────────────


def exact_match(predicted: str, reference: str) -> float:
    """1.0 if stripped strings match exactly, else 0.0."""
    return float(predicted.strip() == reference.strip())


def edit_similarity(predicted: str, reference: str) -> float:
    """
    Character-level edit similarity: 1 − (Levenshtein / max_len).
    Returns a value in [0, 1]; higher is better.
    """
    p = predicted.strip()
    r = reference.strip()
    if not r:
        return 1.0 if not p else 0.0

    m, n = len(p), len(r)
    # Space-optimised single-row DP
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if p[i - 1] == r[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return 1.0 - dp[n] / max(m, n)
```

- **`exact_match`** — strict character equality after `.strip()` on both sides. Returns `1.0` / `0.0`.
- **`edit_similarity`** — `1 − normalisedLevenshtein`, where the Levenshtein distance is computed
  with a space-optimised single-row DP and normalised by `max(len(pred), len(ref))`. Range `[0, 1]`,
  higher is better. Empty-reference edge case: `1.0` if the prediction is also empty, else `0.0`.

### 1b. Per-example scoring loop (used by `safim_eval_1000`) — `src/training/evaluate.py:678-761`

This is the loop that actually produces the numbers for the three-way comparison. It is resumable
(local JSONL + HF-mirrored checkpoint) and carries `task_id` / `fim_type` through every result.

```python
def evaluate_model_by_type(
    model,
    tokenizer,
    test_records: list[dict],
    model_label: str,
    results_path: str,
    hf_checkpoint_repo: str | None = None,
    hf_checkpoint_folder: str = "experiment/safim_eval_1000_results",
    checkpoint_every_n_records: int = 50,
    max_samples: int | None = None,
    token: str | None = None,
    wandb_run: Any | None = None,
) -> list[dict]:
    """
    Like evaluate_model, but: carries task_id/fim_type through each result,
    resumes from already-scored task_ids (local file, falling back to the
    HF-mirrored checkpoint) instead of re-generating them, appends each new
    result to `results_path` immediately, and re-uploads `results_path` to
    `hf_checkpoint_repo` every `checkpoint_every_n_records` (and always at
    the end) so a whole-session crash loses at most that many records for
    this model, not the model's entire progress.

    `test_records` each need: prefix, suffix, middle, fim_type, task_id
    (src.training.evaluate.load_fim_eval_records's output shape).
    `max_samples=None` runs every record; otherwise runs only the first
    `max_samples` records (by test_records order) — useful for a fast
    calibration pass to measure real per-sample throughput before committing
    to a full run.

    Returns every scored result for this model (already-scored + newly-
    scored), read back from `results_path`.
    """
    already_scored = _load_already_scored(
        results_path, model_label, hf_checkpoint_repo, hf_checkpoint_folder, token
    )
    if already_scored:
        print(f"  ↻ Resuming {model_label}: {len(already_scored)} record(s) already scored — skipping them.")

    candidate_records = test_records if max_samples is None else test_records[:max_samples]
    records_to_run = [r for r in candidate_records if r["task_id"] not in already_scored]

    n = len(records_to_run)
    start = time.monotonic()
    since_last_mirror = 0

    for i, rec in enumerate(records_to_run):
        completion = generate_completion(model, tokenizer, rec["prefix"], rec["suffix"])
        result = {
            "model": model_label,
            "task_id": rec["task_id"],
            "fim_type": rec["fim_type"],
            "exact_match": exact_match(completion, rec["middle"]),
            "edit_similarity": edit_similarity(completion, rec["middle"]),
        }
        _append_result(results_path, result)
        since_last_mirror += 1

        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.monotonic() - start
            per_sample = elapsed / (i + 1)
            eta_s = per_sample * (n - (i + 1))
            print(
                f"  [{i + 1:4d}/{n}]  {model_label}  "
                f"({per_sample:.1f}s/sample, ETA {eta_s / 60:.1f} min)"
            )
            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            f"eval/safim_eval_1000/{model_label}/completed": len(already_scored) + i + 1,
                            f"eval/safim_eval_1000/{model_label}/sec_per_sample": per_sample,
                        }
                    )
                except Exception as exc:
                    print(f"  ⚠ W&B live progress log failed: {exc}")

        if hf_checkpoint_repo and since_last_mirror >= checkpoint_every_n_records:
            _mirror_checkpoint_to_hf(results_path, hf_checkpoint_repo, hf_checkpoint_folder, model_label, token)
            since_last_mirror = 0

    if hf_checkpoint_repo and since_last_mirror > 0:
        _mirror_checkpoint_to_hf(results_path, hf_checkpoint_repo, hf_checkpoint_folder, model_label, token)

    return [r for r in _read_results_jsonl(results_path) if r.get("model") == model_label]
```

### 1c. Aggregation — `src/training/evaluate.py:882-886`

```python
df = pd.DataFrame(all_results)
aggregate_summary = df.groupby("model")[["exact_match", "edit_similarity"]].mean()
by_bucket_summary = (
    df.groupby(["fim_type", "model"])[["exact_match", "edit_similarity"]].mean().reset_index()
)
```

- **Aggregate** = mean `exact_match` / `edit_similarity` per model → `results/safim_eval_1000/metrics.csv`
- **By bucket** = mean per `(fim_type, model)` → `results/safim_eval_1000/metrics_by_bucket.csv`

> The simpler two-way flow (`Base` vs a single `LoRA-FT`) lives in `evaluate_model` /
> `run_evaluation` (`src/training/evaluate.py:160-191`, `447-560`) and uses the same two metric
> functions; `safim_eval_1000` uses the N-way `evaluate_model_by_type` / `run_evaluation_nway_by_type`
> path above.

---

## 2. Generation / inference configuration

### 2a. Decode call — `src/training/evaluate.py:97-120`

```python
def generate_completion(
    model,
    tokenizer,
    prefix: str,
    suffix: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Run greedy FIM inference for one (prefix, suffix) pair.
    Returns the model's predicted middle (decoded, no special tokens).
    """
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy → deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt tokens, decode only new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

| Setting | Value | Notes |
|---|---|---|
| Decoding | Greedy (`do_sample=False`) | Deterministic — required for apples-to-apples model comparison |
| `max_new_tokens` | `128` | Hard cap on the predicted middle |
| `pad_token_id` / `eos_token_id` | `tokenizer.eos_token_id` | |
| Batching | None | One `(prefix, suffix)` pair per `generate` call |
| Precision | `torch.float16` | Set at model-load time (see 2c) |
| Output post-processing | decode `output_ids[prompt_len:]` with `skip_special_tokens=True` | Only the newly generated tokens are returned |
| `no_grad` | Yes | |

### 2b. Model loading — `src/training/evaluate.py:55-94`

```python
def load_model(
    base_model_name: str,
    adapter_path: str | None = None,
):
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=False,
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"  ✓ Adapter loaded from: {adapter_path}")
        FastLanguageModel.for_inference(model)
    except ImportError:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"  ✓ Adapter loaded from: {adapter_path}")
        model.eval()

    return model, tokenizer
```

- `max_seq_length = 2048`, `dtype = float16`, `load_in_4bit = False`.
- unsloth `FastLanguageModel` when available; otherwise plain `transformers`
  (`AutoModelForCausalLM` + `device_map="auto"`) + `PeftModel` for the adapter.
- One 0.5B model is loaded at a time; `del model` + `torch.cuda.empty_cache()` between models
  (`run_evaluation_nway_by_type`, `src/training/evaluate.py:878-880`).

### 2c. Single-GPU pin — `scripts/run_safim_eval_1000.py:30-38`

```python
# This pipeline loads exactly one 0.5B model at a time (Base, then each
# LoRA) — it never wants model parallelism. On a multi-GPU box (e.g. Kaggle
# "GPU T4 x2") unsloth otherwise shards even this tiny model across
# cuda:0/cuda:1, and generation then dies with "Expected all tensors to be
# on the same device" once embed_tokens lands on a different GPU than the
# tokenised inputs. Pin to one visible GPU before torch is imported (which
# happens transitively via src.training.evaluate below). Respect an
# explicit override if the caller already set it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
```

### 2d. Run config — `configs/data/safim_eval_1000.yaml` (`evaluation` block)

```yaml
evaluation:
  base_model_name: "Qwen/Qwen2.5-Coder-0.5B"
  model_hf_repo: "Rudra-G-23/qwen2.5-coder-0.5b-python-fim"
  models:
    - label: "Base"
      adapter_subfolder: null
    - label: "Random-LoRA"
      adapter_subfolder: "experiment/random_model/seed42"
    - label: "Distributed-LoRA"
      adapter_subfolder: "experiment/distributed_model/seed42"
  max_samples: null            # null = all 1000; set lower for a fast calibration run
  checkpoint_every_n_records: 50
  output_dir: "results/safim_eval_1000"
```

Eval set (fixed 1000 FIM tasks, `sampling.seed: 42`) and its bucket quota:

```yaml
sampling:
  source_file_count: 750
  max_spans_per_file: 6
  seed: 42

bucket_quota:            # literal counts, sum = 1000
  line: 150
  statement: 150
  expression: 150
  block: 120
  function-body: 120
  method-body: 100
  api-call: 120
  class-level: 90
```

---

## 3. Per-example comparison rows

> **NOTE — code gap.** As written, `evaluate_model_by_type`
> (`src/training/evaluate.py:725-731`) persists **only** `model`, `task_id`, `fim_type`,
> `exact_match`, `edit_similarity` to `{output_dir}/{model_label}_results.jsonl`. The generated
> completion text and the reference `middle` are **not** saved, and there is no `results/`
> directory in the repo yet (the eval has not been run locally). To fill the table below with
> real generations you must first extend the result dict, e.g.:
>
> ```python
> completion = generate_completion(model, tokenizer, rec["prefix"], rec["suffix"])
> result = {
>     "model": model_label,
>     "task_id": rec["task_id"],
>     "fim_type": rec["fim_type"],
>     "middle": rec["middle"],          # add
>     "completion": completion,          # add
>     "exact_match": exact_match(completion, rec["middle"]),
>     "edit_similarity": edit_similarity(completion, rec["middle"]),
> }
> ```
>
> Then join the three per-model JSONLs on `task_id` and take 10–20 rows spanning the eight
> `fim_type` buckets (`line`, `statement`, `expression`, `block`, `function-body`,
> `method-body`, `api-call`, `class-level`).

Fill after running the eval:

| # | task_id | fim_type | middle (reference) | Base output | Random-LoRA output | Distributed-LoRA output | EM B/R/D | ES B/R/D |
|---|---------|----------|--------------------|-------------|--------------------|-------------------------|----------|----------|
| 1 |  | line |  |  |  |  |  |  |
| 2 |  | line |  |  |  |  |  |  |  |
| 3 |  | statement |  |  |  |  |  |  |  |
| 4 |  | statement |  |  |  |  |  |  |  |
| 5 |  | expression |  |  |  |  |  |  |  |
| 6 |  | expression |  |  |  |  |  |  |  |
| 7 |  | block |  |  |  |  |  |  |  |
| 8 |  | block |  |  |  |  |  |  |  |
| 9 |  | function-body |  |  |  |  |  |  |  |
| 10 |  | function-body |  |  |  |  |  |  |  |
| 11 |  | method-body |  |  |  |  |  |  |  |
| 12 |  | method-body |  |  |  |  |  |  |  |
| 13 |  | api-call |  |  |  |  |  |  |  |
| 14 |  | api-call |  |  |  |  |  |  |  |
| 15 |  | class-level |  |  |  |  |  |  |  |
| 16 |  | class-level |  |  |  |  |  |  |  |

(EM = `exact_match`, ES = `edit_similarity`; B/R/D = Base / Random-LoRA / Distributed-LoRA.)

---

## 4. Qwen FIM prompt construction

Qwen2.5-Coder's native FIM sentinel tokens, used identically everywhere:

```python
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
```

### 4a. Inference prompt (no `middle`) — `src/training/evaluate.py:108`

```python
prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
```

Layout: `<|fim_prefix|>` + code before the hole + `<|fim_suffix|>` + code after the hole +
`<|fim_middle|>` → the model then generates the middle. This is the **PSM** (prefix–suffix–middle)
ordering.

### 4b. Training text (with `middle` appended) — `src/data_prep.py:78`

```python
"text": f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}",
```

Same layout as 4a, with the gold `middle` concatenated after `<|fim_middle|>` so the model
learns to produce it. Tokens defined at `src/data_prep.py:17-19`.

### 4c. Pilot experiment training text — `scripts/run_pilot_experiment.py:120`

```python
text = f"{FIM_PREFIX}{row['prefix']}{FIM_SUFFIX}{row['suffix']}{FIM_MIDDLE}{row['middle']}"
```

Imports the tokens from `src.data_prep` (`scripts/run_pilot_experiment.py:79`) — identical format
to 4b.

### 4d. Ollama Modelfile template (deployment) — `src/convert_gguf.py:145-150`

```
TEMPLATE """<|fim_prefix|>{{ .Prompt }}<|fim_suffix|><|fim_middle|>"""

PARAMETER stop "<|fim_middle|>"
PARAMETER stop "<|fim_prefix|>"
PARAMETER stop "<|fim_suffix|>"
```

Same PSM ordering at serve time; the three sentinels are also registered as stop strings.

### Consistency check

| Site | File:line | Order | Includes `middle`? |
|---|---|---|---|
| Inference / eval | `src/training/evaluate.py:108` | `prefix, suffix, middle` (PSM) | No — model generates it |
| Local FIM data build | `src/data_prep.py:78` | PSM | Yes (training target) |
| Pilot training data | `scripts/run_pilot_experiment.py:120` | PSM | Yes |
| Ollama serve template | `src/convert_gguf.py:145` | PSM | No — `{{ .Prompt }}` = prefix, stops before middle |

All four use the identical `<|fim_prefix|> … <|fim_suffix|> … <|fim_middle|>` PSM layout, so the
eval prompt matches what the adapters were trained on.
