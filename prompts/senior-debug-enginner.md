You are acting as a senior ML research engineer and code-audit agent.

Your task is NOT to retrain the model and NOT to blindly modify the repository.

Your task is to deeply inspect my current Qwen2.5-Coder-0.5B Python FIM fine-tuning/evaluation codebase and determine whether my disappointing evaluation results are caused by:

1. the fine-tuned models genuinely not improving,
2. incorrect/incomplete evaluation,
3. generation/stopping behavior,
4. dataset/evaluation design,
5. training target construction,
6. LoRA/model loading,
7. tokenizer/FIM formatting,
8. or another implementation issue.

After inspection, create ONE comprehensive Markdown report that I can paste into ChatGPT for further analysis.

==================================================
PROJECT CONTEXT
==================================================

Base model:
Qwen/Qwen2.5-Coder-0.5B

Task:
Python Fill-in-the-Middle (FIM) code autocomplete.

Fine-tuning:
LoRA.

Models being compared:
- Base
- Random-LoRA seed42
- Distributed-LoRA seed42

Training experiment:
- Random training distribution: 20,000 FIM samples
- Distributed/bucket-balanced training distribution: 20,000 FIM samples
- pilot based on ~10,000 source files
- evaluation later created from unseen files

Main current evaluation:
approximately 1,000 FIM examples.

Current aggregate results:

Base:
exact_match = 0.000
edit_similarity ≈ 0.135630

Distributed-LoRA:
exact_match = 0.000
edit_similarity ≈ 0.145897

Random-LoRA:
exact_match = 0.000
edit_similarity ≈ 0.142690

Current bucket-level edit similarity:

api-call:
Base 0.105411
Distributed 0.114008
Random 0.112317

block:
Base 0.216795
Distributed 0.226130
Random 0.218875

class-level:
Base 0.094470
Distributed 0.116643
Random 0.116142

expression:
Base 0.088208
Distributed 0.095461
Random 0.095115

function-body:
Base 0.218148
Distributed 0.240265
Random 0.226810

line:
Base 0.073593
Distributed 0.076497
Random 0.082148

method-body:
Base 0.226834
Distributed 0.224357
Random 0.218351

statement:
Base 0.102212
Distributed 0.116807
Random 0.112351

IMPORTANT HISTORICAL RESULT:

An earlier ~100-task SAFIM-style evaluation reportedly produced approximately:

Base: 0/100
Random-LoRA: 9/100
Distributed-LoRA: 2/100

The new 1,000-task evaluation reports exact_match = 0 for all models.

One of the most important things you must investigate is WHY these two results differ.

Do not assume "pass@1", "exact match", and functional SAFIM correctness are the same metric.

==================================================
OPERATING RULES
==================================================

1. INSPECT BEFORE CHANGING ANYTHING.

2. Do NOT start training.

3. Do NOT launch a full 1,000-example evaluation.

4. Do NOT run expensive GPU jobs unless absolutely required to verify a specific hypothesis.

5. Do NOT modify model weights, datasets, adapters, Hugging Face artifacts, W&B artifacts, or existing results.

6. Do NOT delete or overwrite existing experiment outputs.

7. Prefer static code inspection first.

8. You may run:
   - unit tests,
   - small CPU checks,
   - tokenizer checks,
   - tiny 1–5 sample inference checks if a usable model environment already exists,
   - dataset statistics scripts,
   - non-destructive diagnostic scripts.

9. If something cannot be verified without expensive execution, mark it:
   "REQUIRES RUNTIME VERIFICATION"

10. Do not speculate as fact.

For every conclusion give:
- evidence,
- file path,
- relevant function/class,
- line numbers when possible,
- confidence: HIGH / MEDIUM / LOW.

11. Search the whole repository before concluding something is missing.

12. Check current implementation AND relevant git history/configs if useful.

13. Do not implement fixes during this task unless a tiny temporary diagnostic is required.

The final deliverable is primarily an AUDIT REPORT.

==================================================
PHASE 1 — MAP THE ACTUAL PIPELINE
==================================================

First reconstruct the actual end-to-end experiment.

Find and document:

Training dataset
    ↓
FIM extraction
    ↓
training text construction
    ↓
tokenization
    ↓
labels / masking
    ↓
LoRA configuration
    ↓
Trainer/SFTTrainer
    ↓
checkpoint/adapter
    ↓
model loading
    ↓
evaluation dataset
    ↓
FIM prompt
    ↓
generation
    ↓
post-processing
    ↓
metrics
    ↓
aggregation
    ↓
W&B / Hugging Face output

Identify the exact files/functions responsible for every stage.

Do not rely on README claims when code can be inspected directly.

==================================================
PHASE 2 — VERIFY FIM FORMAT
==================================================

Find every occurrence of:

<|fim_prefix|>
<|fim_suffix|>
<|fim_middle|>

Determine whether training and inference use exactly the same ordering.

Expected current format appears to be:

Training:

<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}

Inference:

<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>

Verify this from code.

Check:

- accidental spaces/newlines around sentinel tokens
- duplicated tokens
- missing tokens
- different PSM/SPM ordering somewhere
- tokenizer special-token handling
- whether sentinel tokens already exist in Qwen tokenizer
- whether they are accidentally added again
- whether token IDs match expectations
- whether preprocessing strips meaningful whitespace
- whether train/eval formatting differs in any path
- whether Ollama/deployment formatting differs from training/eval

Create a table:

Location | File | Function | FIM format | Consistent? | Notes

==================================================
PHASE 3 — INVESTIGATE TRAINING TARGET AND EOS
==================================================

This is CRITICAL.

Locate the complete training pipeline including:

- tokenizer call
- max_seq_length
- truncation
- padding
- labels
- attention mask
- DataCollator
- SFTTrainer/Trainer configuration
- dataset_text_field if applicable
- packing
- add_special_tokens
- EOS behavior
- loss masking

Answer explicitly:

1. Is EOS appended after the gold `middle`?

For example, does training effectively contain:

<|fim_middle|>{middle}<EOS>

or does it end immediately after `{middle}` with no EOS?

2. Does the tokenizer automatically append EOS?

Do not guess. Verify.

3. Are labels equal to all input IDs?

4. Is loss being calculated on:

A. the entire prefix + suffix + middle sequence,

or

B. only the target middle?

5. Is that intentional?

6. Could the training setup be teaching the model to continue generating rather than stopping after the missing span?

7. Is padding properly masked with -100?

8. Are FIM sentinel tokens incorrectly included/excluded from loss?

9. Is truncation potentially chopping off the `middle` or EOS?

10. What percentage of training samples are truncated?

If easily calculable, report statistics.

11. Check whether Random and Distributed datasets have different:
- average total token length
- average target token length
- truncation rate
- source-file diversity

These could make "20,000 examples vs 20,000 examples" misleading.

==================================================
PHASE 4 — AUDIT GENERATION
==================================================

Find the exact inference code.

Inspect at least:

- max_new_tokens
- min_new_tokens
- do_sample
- temperature
- top_p
- top_k
- repetition_penalty
- eos_token_id
- pad_token_id
- stopping criteria
- stop strings
- output slicing
- skip_special_tokens
- model.eval()
- inference mode/no_grad
- tokenizer padding/truncation
- prompt length

Current suspected behavior is approximately:

max_new_tokens=128
do_sample=False
eos_token_id=tokenizer.eos_token_id

Verify it.

Then answer:

1. What makes generation stop?

2. Is there any FIM-specific stopping rule?

3. Is there syntax-aware truncation?

4. Is there task-dependent truncation for:
- line
- expression
- statement
- block
- function-body
- method-body
- class-level
- api-call?

5. Could the model generate the correct middle first and then continue producing additional code?

6. Would such an output receive exact_match=0?

7. Could this explain the current result?

8. Does generation frequently hit exactly max_new_tokens=128?

Search existing result/log artifacts for evidence.

9. Are special FIM tokens accidentally removed before they can be used as stop signals?

10. Compare evaluation generation behavior with Ollama's Modelfile stop configuration.

Determine whether deployment currently has better stopping rules than evaluation.

==================================================
PHASE 5 — AUDIT max_new_tokens=128
==================================================

This is CRITICAL.

Compute target token lengths for the fixed evaluation dataset using the SAME Qwen tokenizer.

Provide:

count
mean
median
p75
p90
p95
p99
max

Then buckets:

1–16
17–32
33–64
65–128
129+

Also calculate the same statistics per `fim_type`.

Most importantly answer:

How many evaluation references contain >128 target tokens?

Report:

count
percentage

If target length >128, exact character reproduction is impossible under max_new_tokens=128.

State exactly how many tasks have this issue.

Also inspect whether input context + output could hit context-window limits.

==================================================
PHASE 6 — AUDIT EXACT MATCH
==================================================

Find exact_match implementation.

Current suspected implementation:

predicted.strip() == reference.strip()

Verify.

Explain exactly what transformations occur before comparison.

Test representative examples WITHOUT changing production code:

A:
reference = "return x"
prediction = "return x"

B:
reference = "return x"
prediction = "return  x"

C:
reference = "return x"
prediction = "return x\n"

D:
reference = "return x"
prediction = "return x\n# extra"

E:
semantically identical Python with different formatting.

Report whether each passes current EM.

Determine whether exact_match is appropriate as:
- primary metric,
- secondary metric,
- diagnostic metric.

Do NOT silently redefine it.

==================================================
PHASE 7 — AUDIT EDIT SIMILARITY
==================================================

Verify the Levenshtein implementation mathematically.

Check:

- normalization denominator
- empty strings
- symmetry
- range
- Unicode handling if relevant
- whether stripping changes results
- whether long over-generation heavily penalizes results

Run a few tiny known examples and ensure the implementation is correct.

Determine what an edit similarity around 0.14 actually means under this implementation.

Do NOT interpret it as "14% accuracy."

==================================================
PHASE 8 — OFFICIAL SAFIM VS CUSTOM EVALUATION
==================================================

Determine whether the repository's `safim_eval_1000` is actually the official SAFIM benchmark or a custom evaluation dataset inspired by SAFIM.

Inspect:

- dataset origin
- dataset generation
- bucket taxonomy
- evaluator
- post-processing
- functional correctness
- benchmark task definitions

Current custom buckets seem to be:

line
statement
expression
block
function-body
method-body
class-level
api-call

If this is NOT official SAFIM, clearly state so.

Recommend a clearer name such as:

python_fim_eval_1000

only as a recommendation; do not rename anything yet.

Then locate the earlier 100-task evaluation that produced:

Base 0
Random 9
Distributed 2

Determine:

- exact script
- dataset
- metric
- post-processing
- evaluation semantics
- model checkpoints
- generation config

Create a side-by-side table:

Property | Earlier 100-task eval | Current 1000-task eval

Include:

Dataset
Dataset source
Number examples
Metric
Functional execution?
Syntax-aware?
Exact match?
Generation settings
Stopping
Post-processing
Model revision
Adapter revision
Tokenizer
Prompt format
Task categories

Then give your best evidence-backed explanation for:

WHY 9/100 previously became 0/1000 exact-match now.

==================================================
PHASE 9 — CHECK OFFICIAL SAFIM EVALUATION LOGIC
==================================================

If internet access is available, consult ONLY authoritative sources first:

- official SAFIM paper
- official SAFIM GitHub repository
- Qwen/Qwen2.5-Coder official model/tokenizer resources

Do not copy large amounts of code.

Determine whether official SAFIM uses:

- syntax-aware truncation
- task-specific post-processing
- execution
- AST comparison
- exact match
- CodeBLEU
- other functional criteria

Record links/sources in the report.

Clearly distinguish:

REPOSITORY FACT
vs
EXTERNAL BENCHMARK FACT

If internet is unavailable, state that and continue with repository inspection.

==================================================
PHASE 10 — CHECK MODEL / LoRA LOADING
==================================================

Verify that adapters are genuinely being applied.

For each model:

Base
Random-LoRA
Distributed-LoRA

Identify:

- base model revision
- adapter path/revision
- PEFT config
- LoRA rank
- alpha
- dropout
- target modules
- trainable parameter count if available

Check for possible failures such as:

- adapter path incorrect
- silently loading Base instead
- adapter not active
- wrong model revision
- wrong tokenizer revision
- adapters accidentally merged incorrectly
- incompatible PEFT setup
- FastLanguageModel / PeftModel interaction problem
- adapter loaded but inference mode not enabled correctly

Where feasible, perform a small non-destructive check such as:

- inspect active adapters
- inspect a known LoRA parameter
- compare logits Base vs LoRA for the same prompt

Do NOT run a full benchmark.

Report whether Base and LoRA models actually produce different logits/output.

==================================================
PHASE 11 — DATA LEAKAGE / EVAL INTEGRITY
==================================================

Verify the fixed 1000-example evaluation set.

Check:

- task IDs
- content IDs
- source file IDs
- training/eval intersection
- duplicate tasks
- duplicate source files
- duplicate prefix/suffix/middle triples
- duplicate middle snippets if useful

Explicitly test:

training source IDs ∩ evaluation source IDs

Expected:
0

Report actual count.

Also determine whether Random and Distributed training datasets overlap with each other.

That overlap is not inherently wrong, but report it.

Confirm all 3 models receive exactly the same 1000 task IDs in exactly the same evaluation set.

==================================================
PHASE 12 — BUCKET DISTRIBUTION
==================================================

Verify actual evaluation counts by fim_type.

Expected approximate quotas:

line: 150
statement: 150
expression: 150
block: 120
function-body: 120
method-body: 100
api-call: 120
class-level: 90

Do not trust config alone.

Count the actual dataset.

Compare actual vs configured.

Also report average target token count by FIM bucket.

This is important because a bucket may appear worse simply because its targets are substantially longer.

==================================================
PHASE 13 — OUTPUT PERSISTENCE PROBLEM
==================================================

Inspect what each evaluation JSONL currently saves.

Determine whether it saves:

task_id
content_id
model
fim_type
prefix
suffix
reference middle
generated completion
target_tokens
completion_tokens
exact_match
edit_similarity
generation length
whether generation hit max_new_tokens

Current suspicion is that only something like this is stored:

model
task_id
fim_type
exact_match
edit_similarity

If so, mark it as an important observability/debugging gap.

Explain why aggregate metrics cannot diagnose the model without predictions.

Recommend a future canonical result schema.

DO NOT modify the production result format yet.

==================================================
PHASE 14 — EXISTING RESULTS / LOGS
==================================================

Search the repository and available artifacts for:

- JSONL evaluation outputs
- metrics.csv
- metrics_by_bucket.csv
- W&B logs/references
- Hugging Face paths
- saved generations
- earlier pilot results
- notebooks
- screenshots or reports
- previous evaluation scripts

Use existing evidence wherever possible.

Do not assume results do not exist just because they are not in `results/`.

==================================================
PHASE 15 — SMALL DIAGNOSTIC SAMPLE DESIGN
==================================================

Do NOT run it unless inexpensive and environment-ready.

Design a diagnostic run of approximately:

5 examples × 8 FIM buckets = 40 tasks

for:

Base
Random-LoRA
Distributed-LoRA

= approximately 120 generations.

For each task, future output should include:

task_id
content_id
fim_type
prefix
suffix
reference_middle
base_completion
random_completion
distributed_completion
target_tokens
completion_tokens
exact_match
edit_similarity
hit_generation_limit

Then manually/heuristically classify completion behavior:

A = wrong immediately
B = correct/near-correct beginning but over-generated
C = mostly correct but formatting/syntactic differences
D = correct completion
E = unclear

Explain how this diagnostic would separate:
model failure
from
evaluation failure.

Again: DESIGN first. Do not automatically run expensive inference.

==================================================
PHASE 16 — STATISTICAL INTERPRETATION
==================================================

Using existing 1000-task scores, calculate simple descriptive differences:

Distributed vs Base
Random vs Base
Distributed vs Random

overall and by FIM bucket.

Report:
- absolute difference
- relative difference where meaningful

Do NOT claim statistical significance from these means alone.

Determine what information would be needed for paired testing.

Because all models run the same task IDs, future evaluation should use paired comparisons.

Suggest appropriate analysis such as:
- paired bootstrap confidence intervals
- per-task deltas
- sign/win/tie analysis

For binary functional correctness, mention an appropriate paired test such as McNemar if applicable.

Do not overcomplicate or implement statistical testing unless the per-example data already supports it.

==================================================
PHASE 17 — FIND ANY OTHER PROBLEMS
==================================================

Do not restrict yourself to the issues listed above.

Search for anything else that could invalidate the experiment, including:

- wrong dataset split
- tokenizer mismatch
- silent truncation
- seed misuse
- resume bugs
- duplicated result rows
- stale cached files
- wrong HF adapter revision
- mismatch between config and code
- random vs distributed using different training steps
- different token counts
- different learning-rate schedules
- checkpoint-selection differences
- data formatting differences
- accidental train/eval contamination
- result aggregation bugs
- model/device bugs
- inconsistent precision
- incomplete adapter loading
- generation output slicing bugs
- multiple EOS token IDs
- bad stop token assumptions
- suffix/context truncation
- FIM holes positioned differently between datasets
- preprocessing that normalizes one dataset but not another
- W&B/HF resume mixing results from old runs
- task_id collisions
- model labels accidentally sharing result files

Anything suspicious should be included.

==================================================
FINAL OUTPUT
==================================================

Create a file named:

FIM_EVALUATION_AUDIT.md

The report MUST be understandable by another AI that has never seen the repository.

Use this exact structure:

# Python FIM Evaluation Audit

## 1. Executive Summary

Answer in plain language:

- Is the current evaluation trustworthy?
- Can we conclude fine-tuning failed?
- What are the 3–5 most important problems?
- Should we train more models right now?
- Overall confidence.

## 2. Repository / Experiment Snapshot

Include:

- branch
- HEAD commit
- dirty working tree status
- base model
- model/adapters
- relevant datasets
- training seed
- data seed
- key configs

## 3. End-to-End Pipeline

Provide a Mermaid or text diagram of the real pipeline.

## 4. Confirmed Correct Components

For each:
- component
- evidence
- file/path
- reason

## 5. Confirmed Problems

Use a table:

| Severity | Problem | Evidence | Impact | Confidence |

Severity:
CRITICAL / HIGH / MEDIUM / LOW

Then expand each problem below the table.

## 6. Suspected Problems Requiring Runtime Verification

Use:

| Hypothesis | Evidence | What would verify it | Cost |

## 7. FIM Prompt Consistency Audit

Include every relevant code location.

## 8. Training / EOS / Label Audit

Explicitly answer every question about EOS, labels, truncation and loss masking.

## 9. Generation & Stopping Audit

Explicitly describe when and why generation stops.

## 10. Target-Length Analysis

Include overall and per-bucket statistics.

Important:
state count and % of targets >128 tokens.

## 11. Metric Audit

### Exact Match
### Edit Similarity
### Functional/Syntax-Aware Metrics

Explain exactly what each does and does not measure.

## 12. Earlier 100-Task vs Current 1000-Task Evaluation

Provide the side-by-side comparison and explain the discrepancy.

## 13. Official SAFIM vs Our Evaluation

Clearly state whether current evaluation is official SAFIM.

## 14. Model/Adapter Loading Verification

Include evidence that the two LoRA adapters are actually active and distinct.

## 15. Dataset Integrity / Leakage

Include exact overlap numbers.

## 16. Current Results Interpretation

Use the currently available results.

Do NOT call ~0.14 edit similarity "14% accuracy."

Explain what can and cannot be inferred.

## 17. Missing Observability

Explain whether predictions/reference middles are currently persisted.

Give recommended canonical result schema.

## 18. Recommended Diagnostic Evaluation

Describe the 40-task × 3-model diagnostic.

## 19. Recommended Fixes — PRIORITIZED

Use:

P0 = required before trusting evaluation
P1 = strongly recommended
P2 = research-quality improvement

For every recommendation include:

- what to change
- why
- file/function likely involved
- whether existing results need rerunning

Do NOT implement them during this audit.

## 20. Go / No-Go Decision Tree

Create something like:

Current evaluation
    ↓
Fix P0 issues
    ↓
Run diagnostic generations
    ↓
If outputs are correct but over-generated
        → fix post-processing/stopping
If outputs are immediately wrong
        → investigate training
If corrected functional metric improves
        → continue experiment
If no meaningful improvement
        → stop/pivot before more training

Make this specific to findings.

## 21. Exact Files That Would Need Modification Later

List them only.

No modifications yet.

## 22. Questions Still Unanswered

Anything genuinely unresolved.

## 23. Evidence Appendix

For every major finding include concise relevant code snippets with:

file path
function
line number/range

Avoid dumping huge files.

## 24. Final Verdict

End with exactly these classifications:

Evaluation reliability:
TRUSTWORTHY / PARTIALLY TRUSTWORTHY / NOT YET TRUSTWORTHY

Evidence that LoRA changes behavior:
STRONG / MODERATE / WEAK / NONE

Evidence that LoRA materially improves autocomplete:
STRONG / MODERATE / WEAK / INCONCLUSIVE / NONE

Should train another seed now:
YES / NO

Should publish current model as superior autocomplete model:
YES / NO

Recommended next action:
<one sentence>

==================================================
IMPORTANT FINAL INSTRUCTION
==================================================

Do not give me a generic ML explanation.

I need an evidence-backed audit of MY repository.

Search the repository thoroughly.

For every finding:
show where it came from.

Do not fix issues silently.

Do not launch new training.

Do not publish anything.

Do not overwrite results.

When finished:

1. save the full report as `FIM_EVALUATION_AUDIT.md`
2. print the complete Markdown report in your final response as well
3. make it self-contained so I can copy/paste it directly into another AI conversation.
