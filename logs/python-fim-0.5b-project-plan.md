# Python FIM Completion Model — Project Plan
### Fine-tuning a Small Code Model (Qwen2.5-Coder-0.5B) for Python Autocomplete

For the latest changes you refer [@UPDATES.md](/logs/UPDATES.md)

---

## 0. Starting Point: Correcting the Original Assumption

The project began from an observed pattern: autocomplete suggestions from a locally-run Qwen2.5 model (via Ollama + the Continue VS Code extension) seemed to improve with daily use, leading to the hypothesis that the model was being trained/updated by usage.

**This is not what happens.** Ollama-served inference is stateless — no gradient updates, no weight changes, no memory across sessions. The apparent improvement is explained by:
- Confirmation bias (noticing good completions, forgetting bad ones)
- The user's own code becoming more self-similar/predictable over a session (repeated naming conventions, structure)
- Continue's context window feeding more of the open file/session context into the prompt as a session progresses

The real, buildable version of the underlying idea — training a small model that genuinely specializes to Python — is what this plan covers.

---

## 1. Project Framing

**Goal:** Fine-tune a 0.5B parameter code model to be a fast, locally-deployable, Python-specialized fill-in-the-middle (FIM) autocomplete model, and rigorously evaluate whether the fine-tuning actually helps — using standard, execution-based benchmarks rather than subjective impressions.

**Honest scope statement (important for the paper's framing):**
A 0.5B model will never beat 7B/32B models on raw capability. The value proposition is **latency and local deployability** — millisecond-scale tab completion on a laptop CPU/small GPU. This project should be framed as *"best Python FIM accuracy achievable within this size/latency class"*, not *"a better completion model"* in general. That is a defensible, evidence-backed claim; the broader claim is not.

---

## 2. Base Model & Licensing

- **Base model:** `Qwen2.5-Coder-0.5B` (or `-0.5B-Instruct`), not plain Qwen2.5 — it is already code-pretrained.
- **License:** Apache 2.0. Free for commercial use, fine-tuning, and redistribution of derivative models. No permission needed, no royalty. Keep the original license notice in the derivative repo.
- Confirmed: Qwen2.5-Coder 0.5B / 1.5B / 7B / 14B / 32B are Apache 2.0 (only the 3B variant uses a separate Qwen-Research license).

---

## 3. Data: Building FIM Training Data

Qwen2.5-Coder's native FIM template:
```
<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}
```

**How to construct examples (no manual labeling needed):**
1. Take a Python source file.
2. Pick a random contiguous span (a line, a block, a function body).
3. Everything above the span = prefix. Everything below = suffix. The removed span = the middle (training label).
4. Script this over a full corpus to generate unlimited FIM triples automatically.

**Span variety matters** — don't only mask single lines. Mix line-level, block-level, and function-level spans to match realistic FIM training distribution (per Bavarian et al.).

**Data sources, ranked:**
1. Your own repositories — most valuable, genuinely personalizes the model, no license ambiguity.
2. A permissively-licensed public corpus (e.g. a license-filtered Python subset of "the-stack") to bulk up volume.

**Data quality steps:**
- Deduplicate examples.
- Filter to syntactically valid Python only (`ast.parse` compile-check) — a small clean dataset beats a large noisy one at this model scale.

**Kaggle storage limits (confirmed current, not a constraint for this project):**
- Private datasets: 200GB total quota.
- Public dataset/model uploads: up to 200GB per item.
- Your FIM dataset will realistically be tens to low hundreds of MB — nowhere near any limit.

---

## 4. Training Methods — What Each One Actually Is

**Clarification (this was a recurring point of confusion):** LoRA, QLoRA, and distillation are not sequential stages stacked on top of each other. LoRA and QLoRA are **two alternative methods for the same fine-tuning step** — you use one or the other (or run both deliberately as a comparison). Distillation is a **fully separate, independent training run**, not something layered on top of a LoRA checkpoint.

### 4.1 LoRA (primary method)
- Freezes the base model, trains small low-rank adapter matrices.
- At 0.5B scale, the full model + LoRA adapter fits comfortably in a T4/P100's 16GB even at fp16 — there is no memory pressure forcing QLoRA here.
- This should be the primary method for the project.

### 4.2 QLoRA (ablation, not a necessity)
- Same idea as LoRA, but the frozen base weights are kept quantized (4-bit NF4) during training to save memory.
- At this model size it isn't *needed* — frame it explicitly as a controlled ablation answering: "does the memory-saving quantized-base approach cost accuracy at small scale?" This is a genuinely useful comparison for readers deciding between the two on smaller hardware.

### 4.3 Knowledge Distillation
- Tractable approach on Kaggle: **sequence-level distillation**, done as two separate sessions (not simultaneous teacher+student loading, which is memory-heavy):
  1. **Session A:** Load `Qwen2.5-Coder-7B` in 4-bit (bitsandbytes). Run it over your FIM prompts. Save its generated completions as a new "teacher pseudo-label" dataset.
  2. **Session B:** Ordinary SFT — fine-tune the 0.5B student on (prefix+suffix → teacher's completion) pairs, same setup as the LoRA run but with teacher-generated targets instead of ground-truth code.
- This is standard practice in the distillation literature and realistic within a free-tier compute budget.

### 4.4 Training tool
- **Unsloth** — native Qwen2.5-Coder support, runs QLoRA/LoRA on Kaggle's T4/P100 with no local GPU required, easiest on-ramp for a first fine-tuning project.

---

## 5. What NOT to Implement (and why) — MoE and GRPO

### Mixture of Experts (MoE)
Not something you can "add" to an existing dense 0.5B checkpoint as a fine-tuning step. MoE is an architectural decision made at pretraining time (router networks, load-balancing losses, sparse layer structure), and only pays off with pretraining-scale compute. "Sparse upcycling" (converting dense checkpoints to MoE) exists as a technique but itself requires a full continued-pretraining run — outside this project's budget. **Verdict: list as Future Work in the paper. Do not attempt.**

### GRPO
Conceptually applicable — code completion has a verifiable reward (tests pass or not), which is exactly the kind of signal GRPO uses. But practically: it requires multiple rollouts per prompt, a reference model for the KL penalty, a safe code-execution reward function, and typically fast serving infrastructure (e.g. vLLM) that a standard Kaggle notebook isn't set up for. **Verdict: legitimate second-phase idea after a working SFT baseline and more compute. List as Future Work, do not fake-implement.**

Listing both honestly as future work is a stronger choice than attempting a broken implementation — reviewers respect a clearly stated compute boundary.

---

## 6. Kaggle Workflow

- **Free GPU quota:** 30 GPU-hours/week (T4x2 or P100), 12-hour session cap, resets weekly.
- **Reality check:** a 0.5B model is small. A full LoRA run on a few thousand examples will likely finish in a couple of hours, not require the full 20+ hour budget per week. The quota matters more for running multiple experiments/ablations than for needing to wait a week to continue any single run.
- **Session mechanics:** Each Kaggle session starts a fresh container. Nothing persists automatically except what's attached as input datasets/models, or explicitly pushed out before the session ends.
- **Workflow per session:**
  1. Attach base model (download from HF, or attach as a Kaggle "model" input) and your FIM dataset (Kaggle dataset input).
  2. Train, writing periodic checkpoints to `/kaggle/working`.
  3. Before the session ends, authenticate to Hugging Face using a token stored as a **Kaggle secret** (never hardcoded), and push the LoRA adapter to your own HF model repo.
- **Resuming next week:** new notebook → download the same frozen base model fresh → load your saved adapter from HF on top of it → continue training or run evaluation. The base model never changes; only the adapter accumulates updates. This is what makes weekly resumption practical.
- **Output limits:** Kaggle has raised notebook output size limits over time; a LoRA adapter (a few hundred MB) is far below any relevant cap.
- **Reusing public Kaggle notebooks/repos while learning:** normal and fine — just credit anything substantially adapted, in the README.
- **Not needed for this project:** LangChain (an orchestration framework for RAG/agents) — unrelated to fine-tuning a completion model, would be scope creep.

---

## 7. Repository Structure

The core rule: **the training notebook contains no real logic.** All logic lives once, under version control, in a proper repo. The notebook is a thin remote control that installs the repo and calls into it.

```
python-fim-0.5b/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── data_prep.py       # builds FIM prefix/suffix/middle triples
│   ├── train_lora.py
│   ├── train_qlora.py
│   ├── distill.py
│   ├── evaluate.py        # runs SAFIM / RepoEval-style harness against a checkpoint
│   └── convert_gguf.py
├── configs/
│   ├── lora.yaml
│   ├── qlora.yaml
│   └── distill.yaml
├── notebooks/
│   └── kaggle_train.ipynb # thin — pip installs the repo, calls src/ functions
└── results/
    ├── metrics.csv
    └── plots/
```

**Virtual environment:** only useful locally for the parts that don't need a GPU — developing/testing `data_prep.py`, unit-testing FIM-splitting logic, dry-running `evaluate.py` on a small local sample before running for real on Kaggle. `python -m venv .venv` + a pinned `requirements.txt`. On Kaggle, venv is irrelevant — each session is already an isolated container; just `pip install` extra deps (`peft`, `bitsandbytes`, `unsloth`) at the top of the notebook.

---

## 8. Model Naming & Hugging Face Organization

**Naming convention:**
```
qwen2.5-coder-0.5b-python-fim-{variant}
```
where `variant` ∈ `{base, lora, qlora, distilled, distilled-q5km}`.

**Actual variant set (parallel, not stacked):**
- Base (untouched — reference numbers)
- LoRA-SFT on your Python FIM data (primary method)
- QLoRA-SFT on the same data (ablation vs. LoRA)
- Distilled from Qwen2.5-Coder-7B on Python completions
- Each of the above, quantized to GGUF (Q4_K_M / Q5_K_M / Q8_0) — for the "does deployment quantization hurt accuracy" comparison

**Organization:** Use Hugging Face **Collections** (free) to group all repos under a single page, rather than leaving them scattered as unrelated uploads.

---

## 9. Deployment Pipeline: HF Checkpoint → Ollama

The exact chain, mechanically:
1. Trained model exists in HF format (`safetensors` + config + tokenizer). If using a LoRA/QLoRA adapter, **merge the adapter into the base weights first** — GGUF/llama.cpp doesn't juggle separate adapters at inference the way HF `peft` does.
2. Run `llama.cpp`'s `convert_hf_to_gguf.py` to repack the merged HF weights into GGUF format.
3. Run llama.cpp's `quantize` binary against the full-precision GGUF to produce smaller variants (Q4_K_M, Q5_K_M, Q8_0, etc.) — each is a different bit-width/packing tradeoff between size and accuracy.
4. Write an Ollama `Modelfile` pointing at the GGUF path, including the FIM prompt template.
5. `ollama create` — now usable inside Continue exactly like the original pulled model.

**KV cache note (context, not something to modify):** this is a serving-time mechanism that avoids recomputing attention over the whole prefix on every new token — it's why local autocomplete latency behaves the way it does. Worth a paragraph in the write-up, not something to touch.

---

## 10. Evaluation Methodology

Token-overlap metrics (BLEU, raw exact-match) are known to correlate weakly with whether completions are actually *useful*. The field-standard fix is **execution-based evaluation**: run completions against real test cases and report **pass@1**.

**Benchmarks to use (all standard, citable, execution- or similarity-based):**
- **SAFIM** (Syntax-Aware Fill-in-the-Middle) — evaluates FIM specifically across three subtasks (algorithmic block completion, control-flow expression completion, API function call completion), covers Python among four languages, uses execution-based pass@1 scoring. Open-source toolkit and public leaderboard available.
- **RepoEval** — repository-level Python completion benchmark (line/API/function completion tasks from real Python repos), scored with exact match and edit similarity.
- **CrossCodeEval** / **CrossCodeLongEval** — repository-level, cross-file Python completion benchmarks built from real repositories, also scored with exact match and edit similarity.

**Your evaluation setup:**
- Run baseline Qwen2.5-Coder-0.5B **and** each fine-tuned variant through SAFIM's Python subset — report pass@1 for each.
- Run edit-similarity on a held-out slice of your own repos for a personalization-specific number.
- Report accuracy retention **after** GGUF quantization (pass@1 before vs. after quantizing) — most hobby fine-tunes skip this, and it directly matters for a "deployable locally" claim.

**Referencing other papers' published numbers:**
Legitimate *if and only if* the benchmark, metric definition, and evaluation protocol (prompt template, postprocessing) match, and your own measured numbers are clearly labeled separately from cited ones (e.g. "measured, this work" vs. "reported, [Author et al.]"). Standard practice in papers. To strengthen this: run the **base** Qwen2.5-Coder-0.5B yourself through your own harness (trivial extra cost since the harness already exists) rather than relying solely on someone else's published number for it — gives at least one fully apples-to-apples column generated end-to-end.

---

## 11. What to Log During Training

Per run, record:
- **Hyperparameters:** LoRA rank, alpha, dropout, target modules, learning rate, batch size, gradient accumulation steps, epochs, optimizer, warmup steps, seed.
- **Data:** total examples, train/val/test split sizes, source mix (% own repos vs. % external corpus), average/median sequence length.
- **Compute:** GPU type, wall-clock time, total GPU-hours used, peak VRAM.
- **Curves:** training loss and validation loss per logging step (not just the final number) — this becomes your loss-curve plot.
- **Reproducibility:** exact pinned library versions (`transformers`, `peft`, `unsloth`, `torch`) printed at run start and stored in `requirements.txt`.
- **Post-run:** checkpoint size on disk, size per quantization level, inference latency (tokens/sec — ideally measured on CPU, since that's the realistic local-deployment target for a 0.5B model).
- **Checkpoint selection:** pick by validation loss, not simply "last checkpoint" — avoids reporting an overfit run as the headline number.

**Tooling:** MLflow (self-hosted, free, can log from the Kaggle notebook) or Weights & Biases free tier — either auto-captures loss curves as plots, avoiding manual reconstruction later.

---

## 12. Comparing Results & Building Plots

1. Evaluation script runs each model variant against each benchmark, outputs one row per `(model, benchmark, metric)` to a CSV:
   `model_name, benchmark, pass@1, exact_match, edit_similarity, params, quantization`
2. **Grouped bar chart** — benchmark on x-axis, one bar cluster per model variant. Standard "who wins where" view (matplotlib/seaborn).
3. **Accuracy vs. deployment cost plot** — x-axis = model size/quantization level, y-axis = pass@1. This is the plot that actually supports a "best small Python completion model for local use" claim — it shows the tradeoff curve, not just a leaderboard ranking. Likely the most interesting single result in the project.
4. The paper/README table is the same CSV pivoted: rows = model variant, columns = benchmark, cells = pass@1 (plus a second table for edit-similarity).

---

## 13. Reading List (in order)

1. *Efficient Training of Language Models to Fill in the Middle* (Bavarian et al., 2022) — the actual FIM method the base model was trained with. Read first.
2. *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021) — what rank/alpha/target-modules actually configure.
3. *QLoRA* (Dettmers et al., 2023) — what 4-bit NF4 quantized-base training changes vs. plain LoRA.
4. SAFIM paper (*Evaluation of LLMs on Syntax-Aware Code Fill-in-the-Middle Tasks*, Gong et al., 2024) — read the evaluation methodology closely; you'll be reusing their harness.
5. A sequence-level knowledge distillation paper for LLMs — the practical version of distillation actually being used here.
6. Hugging Face `peft` and `transformers` docs, plus Unsloth's own documentation — written directly for this hardware situation (free-tier GPU, small models).

---

## 14. Further Optimization (beyond the base plan)

- **Data quality over quantity** — dedupe, syntax-validate (`ast.parse`), a small clean dataset beats a large noisy one at 0.5B scale.
- **FIM span variety** — mix line/block/function-level spans rather than one granularity.
- **LR schedule** — warmup + cosine decay; state this explicitly in the methods section, reviewers look for it.
- **Checkpoint selection by validation loss**, not "last checkpoint."
- **The quantization sweep as the headline optimization result** — report Q4_K_M vs. Q5_K_M vs. Q8_0 vs. full precision pass@1, not a single quant level. This tradeoff curve is more interesting than any single accuracy number and is what makes an applied study feel complete.

---

## 15. Repo & Paper Naming

- **GitHub repo name:** `python-fim-0.5b` or `qwen-coder-python-fim` — literal and searchable over anything cute.
- **Paper title pattern:**
  *"Adaptation of a 0.5B Code Model for Python Fill-in-the-Middle Completion: A Comparative Study of LoRA, QLoRA, and Distillation Under Deployment Quantization"*
  — tells a reviewer exactly what's inside before opening it.

---

## 16. Summary Pipeline (end to end)

```
Base model (Apache 2.0, no license issue)
        │
        ▼
Self-generated FIM data (own repos + licensed Python corpus)
        │
        ├──► LoRA SFT ─────┐
        ├──► QLoRA SFT ────┤  (parallel variants, not stacked)
        └──► Distillation ─┘  (from Qwen2.5-Coder-7B, sequence-level)
        │
        ▼
Checkpointed as adapters/models on Hugging Face (weekly, via Kaggle sessions)
        │
        ▼
SAFIM / RepoEval / CrossCodeEval evaluation — pass@1, exact match, edit similarity
        │
        ▼
GGUF conversion + quantization sweep (Q4_K_M / Q5_K_M / Q8_0) — re-evaluate for accuracy retention
        │
        ▼
Deployed back into Ollama + Continue (closes the loop to daily use)
        │
        ▼
Results table + plots → GitHub repo + HF Collection + paper write-up
```

**Framing to hold onto throughout:** this is an applied, comparative engineering study with real, execution-based benchmark numbers — not a single fine-tune, and not a claim of general superiority. That distinction is what separates it from a hobby project.

---

## 17. Explicitly Out of Scope (Future Work, not implemented)

- **Mixture of Experts (MoE)** — architectural, pretraining-scale decision; sparse upcycling requires continued pretraining beyond this budget.
- **GRPO** — conceptually applicable (verifiable code-execution reward), but requires multi-rollout sampling, a reference model, and fast serving infra (e.g. vLLM) not available on free-tier Kaggle. Legitimate next phase after a working SFT baseline and more compute.
