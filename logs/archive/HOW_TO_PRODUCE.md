# 🚀 How to Produce: Qwen2.5-Coder-0.5B Python FIM Fine-Tuning
### A Complete Step-by-Step Guide — From Zero to Deployed Model

> **Your Goal (from OPNINON.md):** Understand the full end-to-end flow confidently, reduce training time, save the model, and then produce industry-standard code with small data → eval → graphs. Only LoRA. Workflow: write code in VS Code → push to GitHub → clone on Kaggle → train.

---

## 📋 Table of Contents

1. [Mental Model First — Understand the Big Picture](#1-mental-model-first)
2. [One-Time Local Setup](#2-one-time-local-setup)
3. [Build Your FIM Dataset](#3-build-your-fim-dataset)
4. [Set Up Your Repository Structure](#4-set-up-your-repository-structure)
5. [Write the LoRA Training Code](#5-write-the-lora-training-code)
6. [Write the Kaggle Notebook (Thin Controller)](#6-write-the-kaggle-notebook)
7. [Push to GitHub](#7-push-to-github)
8. [Run on Kaggle](#8-run-on-kaggle)
9. [Save the Model to Hugging Face](#9-save-the-model-to-hugging-face)
10. [Evaluate — Pass@1 and Plots](#10-evaluate-and-plots)
11. [Deploy to Ollama](#11-deploy-to-ollama)
12. [Weekly Resume Flow](#12-weekly-resume-flow)

---

## 1. Mental Model First

Before writing a single line of code, burn this into your head:

```
Your Python files
      │
      ▼
[data_prep.py] — Slice files into FIM triples (prefix, suffix, middle)
      │
      ▼
[train_lora.py] — Freeze Qwen2.5-Coder-0.5B, train only the LoRA adapter
      │
      ▼
Adapter saved to Hugging Face (not the full model — just the small diff)
      │
      ▼
[evaluate.py] — Run benchmark, get pass@1 / edit-similarity numbers
      │
      ▼
[convert_gguf.py] — Merge adapter + base → GGUF → Ollama
      │
      ▼
Continue extension in VS Code uses your own fine-tuned model
```

**Key facts you must never forget:**
- LoRA does NOT change the base model. It adds tiny adapter matrices. The base model weights stay frozen.
- The adapter is small (a few hundred MB). You push this to Hugging Face, not the full model.
- Kaggle gives you 30 GPU-hours/week. A 0.5B LoRA run on a small dataset takes ~1–2 hours. You have plenty of budget.
- Each Kaggle session is a fresh container. Everything lives in GitHub and Hugging Face — nothing in Kaggle itself.

---

## 2. One-Time Local Setup

> Do this once on your Ubuntu/WSL machine. You won't do training here — this is just for developing and testing code before pushing.

### 2.1 Check Python version

```bash
python3 --version
# Should be 3.13+ (your pyproject.toml requires >=3.13)
```

### 2.2 Create and activate virtual environment

```bash
cd ~/open/qwen2.5-coder-0.5b-python-fim
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Install local dev dependencies (no GPU needed locally)

```bash
pip install transformers datasets tokenizers peft
pip install matplotlib seaborn pandas pyyaml
pip install pytest
```

### 2.4 Create requirements.txt

```bash
pip freeze > requirements.txt
```

> **On Kaggle:** You will NOT use this venv. Each Kaggle session is its own container. You'll pip install things at the top of the notebook. The requirements.txt is for reproducibility records.

---

## 3. Build Your FIM Dataset

This is the most important and easiest-to-misunderstand step.

### 3.1 What is a FIM triple?

Take any Python file. Pick a random span (a line, a function body, a block). That gives you:

```
PREFIX: everything ABOVE the span
SUFFIX: everything BELOW the span
MIDDLE: the span itself (this is what the model learns to predict)
```

Formatted for Qwen2.5-Coder's native FIM template:
```
<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}
```

### 3.2 Create `src/data_prep.py`

```python
"""
src/data_prep.py
Build FIM (fill-in-the-middle) training triples from Python source files.
"""

import ast
import json
import random
from pathlib import Path


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def is_valid_python(source: str) -> bool:
    """Return True only if source is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def make_fim_triples(
    source: str,
    n_samples: int = 5,
    min_middle_lines: int = 1,
    max_middle_lines: int = 15,
) -> list[dict]:
    """
    Given a Python source string, produce n_samples FIM triples
    by randomly masking contiguous line spans.
    """
    lines = source.splitlines()
    if len(lines) < 10:
        return []  # too short to be useful

    triples = []
    for _ in range(n_samples * 3):  # try 3x to get n_samples valid ones
        if len(triples) >= n_samples:
            break

        span_len = random.randint(min_middle_lines, min(max_middle_lines, len(lines) - 2))
        start = random.randint(1, len(lines) - span_len - 1)
        end = start + span_len

        prefix = "\n".join(lines[:start])
        middle = "\n".join(lines[start:end])
        suffix = "\n".join(lines[end:])

        if not middle.strip():
            continue

        triples.append({
            "prefix": prefix,
            "suffix": suffix,
            "middle": middle,
            "text": f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}",
        })

    return triples


def build_dataset(
    source_dirs: list[str],
    output_path: str,
    max_files: int = 5000,
    samples_per_file: int = 5,
    seed: int = 42,
) -> None:
    """
    Walk source_dirs, collect Python files, generate FIM triples,
    write to a JSONL file at output_path.
    """
    random.seed(seed)
    python_files = []

    for src_dir in source_dirs:
        python_files.extend(Path(src_dir).rglob("*.py"))

    random.shuffle(python_files)
    python_files = python_files[:max_files]

    records = []
    skipped = 0

    for fpath in python_files:
        try:
            source = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            skipped += 1
            continue

        if not is_valid_python(source):
            skipped += 1
            continue

        triples = make_fim_triples(source, n_samples=samples_per_file)
        records.extend(triples)

    print(f"Files processed: {len(python_files) - skipped}")
    print(f"Files skipped (invalid/unreadable): {skipped}")
    print(f"Total FIM triples: {len(records)}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Dataset saved to: {output}")


if __name__ == "__main__":
    # Quick local test: run on THIS repo itself
    build_dataset(
        source_dirs=["src"],
        output_path="data/fim_dataset.jsonl",
        max_files=100,
        samples_per_file=5,
    )
```

### 3.3 Test locally (no GPU needed)

```bash
python src/data_prep.py
# Should create data/fim_dataset.jsonl with your FIM triples
```

---

## 4. Set Up Your Repository Structure

This is the exact structure that keeps your notebook "thin" and all logic in version control.

```
qwen2.5-coder-0.5b-python-fim/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── data_prep.py         ← (built in Step 3)
│   ├── train_lora.py        ← (built in Step 5)
│   └── evaluate.py          ← (built in Step 10)
├── configs/
│   └── lora.yaml            ← all hyperparameters live here
├── notebooks/
│   └── kaggle_train.ipynb   ← thin controller, calls src/
├── data/
│   └── fim_dataset.jsonl    ← generated locally, uploaded to Kaggle Dataset
├── results/
│   ├── metrics.csv
│   └── plots/
└── logs/
    ├── OPNINON.md
    ├── UPDATES.md
    ├── HOW_TO_PRODUCE.md    ← this file
    └── python-fim-0.5b-project-plan.md
```

### 4.1 Create `src/__init__.py`

```bash
touch src/__init__.py
```

### 4.2 Create `configs/lora.yaml`

```yaml
# configs/lora.yaml
# All hyperparameters for the LoRA run. Change here, not in the code.

model:
  name: "Qwen/Qwen2.5-Coder-0.5B"        # base model from HF
  max_seq_length: 2048

lora:
  r: 16                                    # LoRA rank — start here
  alpha: 32                                # lora_alpha = 2 * r is a common rule
  dropout: 0.05
  target_modules:                          # modules to attach LoRA to
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  num_epochs: 1                            # start with 1 epoch to verify pipeline
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4           # effective batch = 4 * 4 = 16
  learning_rate: 2.0e-4
  warmup_steps: 50
  lr_scheduler_type: "cosine"
  logging_steps: 10
  save_steps: 100
  eval_steps: 100
  seed: 42
  fp16: true                               # T4/P100 supports fp16

data:
  path: "data/fim_dataset.jsonl"
  train_split: 0.90
  val_split: 0.10

output:
  dir: "/kaggle/working/checkpoints"
  hf_repo: "Rudra-G-23/qwen2.5-coder-0.5b-python-fim-lora"
```

---

## 5. Write the LoRA Training Code

### 5.1 Create `src/train_lora.py`

```python
"""
src/train_lora.py
LoRA fine-tuning of Qwen2.5-Coder-0.5B on FIM Python data.
Uses Unsloth for fast, memory-efficient training on Kaggle T4/P100.
"""

import json
import random
from pathlib import Path
import yaml

import torch
from datasets import Dataset
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


def load_config(config_path: str = "configs/lora.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(
    path: str, train_split: float = 0.90, seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Load JSONL FIM dataset and split into train/val."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.seed(seed)
    random.shuffle(records)

    n_train = int(len(records) * train_split)
    train_records = records[:n_train]
    val_records = records[n_train:]

    print(f"Train examples: {len(train_records)}")
    print(f"Val examples:   {len(val_records)}")

    train_ds = Dataset.from_list([{"text": r["text"]} for r in train_records])
    val_ds = Dataset.from_list([{"text": r["text"]} for r in val_records])

    return train_ds, val_ds


def train(config_path: str = "configs/lora.yaml") -> None:
    cfg = load_config(config_path)

    print("=" * 60)
    print("LoRA Fine-tuning — Qwen2.5-Coder-0.5B Python FIM")
    print("=" * 60)

    # ── 1. Load model & tokenizer via Unsloth ──
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model"]["name"],
            max_seq_length=cfg["model"]["max_seq_length"],
            dtype=None,
            load_in_4bit=False,  # full fp16 — fine at 0.5B on T4/P100
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora"]["r"],
            target_modules=cfg["lora"]["target_modules"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg["training"]["seed"],
        )
        print("✓ Loaded model with Unsloth")
    except ImportError:
        # Fallback: plain HuggingFace PEFT
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["name"],
            torch_dtype=torch.float16,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            target_modules=cfg["lora"]["target_modules"],
            lora_dropout=cfg["lora"]["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_config)
        print("✓ Loaded model with plain PEFT (no Unsloth)")

    model.print_trainable_parameters()

    # ── 2. Load data ──
    train_ds, val_ds = load_dataset_from_jsonl(
        cfg["data"]["path"],
        train_split=cfg["data"]["train_split"],
        seed=cfg["training"]["seed"],
    )

    # ── 3. Training arguments ──
    output_dir = cfg["output"]["dir"]
    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        lr_scheduler_type=t["lr_scheduler_type"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,   # pick best by val loss, not last
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=t["seed"],
    )

    # ── 4. Trainer ──
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        args=training_args,
    )

    # ── 5. Train ──
    print("\n▶ Starting training...")
    trainer_output = trainer.train()
    print(f"\n✓ Training complete. Loss: {trainer_output.training_loss:.4f}")

    # ── 6. Save the adapter (NOT the full model) ──
    adapter_path = Path(output_dir) / "lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"✓ Adapter saved to: {adapter_path}")

    # ── 7. Push to Hugging Face ──
    hf_repo = cfg["output"]["hf_repo"]
    print(f"\n▶ Pushing adapter to HF: {hf_repo}")
    model.push_to_hub(hf_repo, token=True)    # uses HF_TOKEN env var
    tokenizer.push_to_hub(hf_repo, token=True)
    print(f"✓ Adapter pushed to: https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    train()
```

---

## 6. Write the Kaggle Notebook (Thin Controller)

The notebook does NOTHING except:
1. Clone your GitHub repo
2. Install dependencies
3. Call your `src/` functions

### 6.1 Kaggle Notebook cells

**Cell 1 — Clone repo and install deps:**
```python
import subprocess, os

GITHUB_REPO = "https://github.com/Rudra-G-23/qwen2.5-coder-0.5b-python-fim.git"
subprocess.run(["git", "clone", GITHUB_REPO, "/kaggle/working/repo"], check=True)

subprocess.run([
    "pip", "install", "-q",
    "unsloth[colab-new]",
    "peft", "trl", "transformers", "datasets", "pyyaml", "huggingface_hub",
], check=True)
print("✓ Setup complete")
```

**Cell 2 — Authenticate to Hugging Face:**
```python
# HF_TOKEN stored as a Kaggle Secret — NEVER hardcode your token!
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
print("✓ HF_TOKEN set")
```

**Cell 3 — Copy dataset from Kaggle Dataset input:**
```python
import shutil

# Attach your Kaggle Dataset named "fim-python-dataset" in the sidebar first
shutil.copy("/kaggle/input/fim-python-dataset/fim_dataset.jsonl",
            "/kaggle/working/repo/data/fim_dataset.jsonl")
print("✓ Dataset copied")
```

**Cell 4 — Run training:**
```python
import sys
sys.path.insert(0, "/kaggle/working/repo")
os.chdir("/kaggle/working/repo")

from src.train_lora import train
train(config_path="configs/lora.yaml")
```

---

## 7. Push to GitHub

```bash
# From your local WSL terminal
cd ~/open/qwen2.5-coder-0.5b-python-fim

# Update .gitignore
cat >> .gitignore << 'EOF'
.venv/
data/
results/
__pycache__/
*.pyc
*.pyo
EOF

git add .
git commit -m "feat: add data_prep, train_lora, lora config, kaggle notebook"
git push origin main
```

---

## 8. Run on Kaggle

### 8.1 Upload your dataset to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **New Dataset**
3. Name it `fim-python-dataset`
4. Upload `data/fim_dataset.jsonl`
5. Click **Create**

### 8.2 Add your HF token as a Kaggle Secret

1. Go to your Kaggle account → **Settings** → **API**
2. Under **Secrets**, add:
   - Name: `HF_TOKEN`
   - Value: your Hugging Face access token (from [hf.co/settings/tokens](https://huggingface.co/settings/tokens))

### 8.3 Create the Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **File** → **Import Notebook** → upload `notebooks/kaggle_train.ipynb`
3. Right panel settings:
   - **Accelerator:** GPU T4 x2 (free)
   - **Persistence:** Files only
   - **Internet:** On (needed for GitHub clone + HF push)
4. **Add Data** → search `fim-python-dataset` → add it
5. Click **Run All**

### 8.4 What happens during the run (~1–2 hrs total)

| Phase | Time |
|---|---|
| Package install | ~3–5 min |
| Model download (Qwen 0.5B) | ~2–3 min |
| Training (1 epoch, small data) | ~30–90 min |
| Push to Hugging Face | ~2–5 min |

---

## 9. Save the Model to Hugging Face

This happens **automatically** at the end of `train_lora.py`. Verify it worked:

1. Go to `https://huggingface.co/Rudra-G-23`
2. You should see `qwen2.5-coder-0.5b-python-fim-lora`
3. The repo should contain:
   - `adapter_config.json`
   - `adapter_model.safetensors`
   - `tokenizer.json`, `tokenizer_config.json`, etc.

> ✅ The base model is NOT uploaded — only the adapter (~50–200MB). That's the whole point of LoRA.

---

## 10. Evaluate and Plots

### 10.1 Create `src/evaluate.py`

```python
"""
src/evaluate.py
Evaluate a LoRA adapter vs. base model on held-out FIM data.
Outputs: results/metrics.csv and results/plots/model_comparison.png
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def load_model(base_model_name: str, adapter_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_completion(model, tokenizer, prefix: str, suffix: str, max_new_tokens: int = 128) -> str:
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def exact_match(predicted: str, reference: str) -> float:
    return float(predicted.strip() == reference.strip())


def edit_similarity(predicted: str, reference: str) -> float:
    """Character-level edit similarity (0..1)."""
    p, r = predicted.strip(), reference.strip()
    if not r:
        return 1.0 if not p else 0.0
    m, n = len(p), len(r)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if p[i-1] == r[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return 1.0 - dp[n] / max(m, n)


def evaluate_model(model, tokenizer, test_records, model_label, max_samples=200):
    results = []
    for i, rec in enumerate(test_records[:max_samples]):
        pred = generate_completion(model, tokenizer, rec["prefix"], rec["suffix"])
        results.append({
            "model": model_label,
            "exact_match": exact_match(pred, rec["middle"]),
            "edit_similarity": edit_similarity(pred, rec["middle"]),
        })
        if (i + 1) % 20 == 0:
            avg_em = sum(r["exact_match"] for r in results) / len(results)
            avg_es = sum(r["edit_similarity"] for r in results) / len(results)
            print(f"  [{i+1}/{min(max_samples, len(test_records))}] EM={avg_em:.3f}  ES={avg_es:.3f}")
    return results


def plot_results(df: pd.DataFrame, output_dir: str = "results/plots") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(
        axes,
        ["exact_match", "edit_similarity"],
        ["Exact Match", "Edit Similarity"],
    ):
        means = df.groupby("model")[metric].mean().reset_index()
        sns.barplot(data=means, x="model", y=metric, ax=ax, palette="viridis")
        ax.set_title(f"{label} by Model", fontsize=13)
        ax.set_ylim(0, 1)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved: {output_dir}/model_comparison.png")


def run_evaluation(
    base_model_name: str,
    adapter_path: str,
    test_data_path: str,
    output_dir: str = "results",
    max_samples: int = 200,
) -> None:
    test_records = []
    with open(test_data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))
    print(f"Test examples: {len(test_records)}")

    all_results = []

    print("\n▶ Evaluating BASE model...")
    base_model, tokenizer = load_model(base_model_name)
    all_results += evaluate_model(base_model, tokenizer, test_records, "Base", max_samples)
    del base_model
    torch.cuda.empty_cache()

    print("\n▶ Evaluating LoRA model...")
    lora_model, tokenizer = load_model(base_model_name, adapter_path)
    all_results += evaluate_model(lora_model, tokenizer, test_records, "LoRA-FT", max_samples)
    del lora_model

    df = pd.DataFrame(all_results)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.groupby("model")[["exact_match", "edit_similarity"]].mean().to_csv(f"{output_dir}/metrics.csv")
    print(f"✓ Metrics saved: {output_dir}/metrics.csv")
    print("\n📊 Results:")
    print(df.groupby("model")[["exact_match", "edit_similarity"]].mean().to_string())

    plot_results(df, output_dir=f"{output_dir}/plots")


if __name__ == "__main__":
    run_evaluation(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B",
        adapter_path="/kaggle/working/checkpoints/lora_adapter",
        test_data_path="data/fim_dataset_test.jsonl",
        output_dir="/kaggle/working/results",
        max_samples=200,
    )
```

### 10.2 Add an evaluation cell to your Kaggle notebook

```python
# Cell 5 — Evaluate (add after training cell)
from src.evaluate import run_evaluation

run_evaluation(
    base_model_name="Qwen/Qwen2.5-Coder-0.5B",
    adapter_path="/kaggle/working/checkpoints/lora_adapter",
    test_data_path="/kaggle/working/repo/data/fim_dataset.jsonl",  # reuse same data (or separate test split)
    output_dir="/kaggle/working/results",
    max_samples=100,  # 100 samples is enough to see the trend
)
```

---

## 11. Deploy to Ollama

> Do this ONCE after you're confident in the model.

### 11.1 Merge adapter into base weights

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "Rudra-G-23/qwen2.5-coder-0.5b-python-fim-lora")
merged = model.merge_and_unload()   # folds adapter into base weights

merged.save_pretrained("merged_model")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B").save_pretrained("merged_model")
print("✓ Merged model saved")
```

### 11.2 Convert to GGUF

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make

python convert_hf_to_gguf.py /path/to/merged_model --outfile qwen-fim.gguf
```

### 11.3 Create Ollama Modelfile and load

```
# Modelfile
FROM ./qwen-fim.gguf
TEMPLATE """<|fim_prefix|>{{ .Prompt }}<|fim_suffix|><|fim_middle|>"""
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|fim_middle|>"
```

```bash
ollama create qwen-fim -f Modelfile
ollama run qwen-fim
```

---

## 12. Weekly Resume Flow

Because each Kaggle session is a fresh container, resuming is simple:

```
Week 2+:
1. New Kaggle notebook → clone same GitHub repo
2. Attach same Kaggle Dataset + same secrets
3. Load YOUR adapter from HF on top of fresh base:
   model = PeftModel.from_pretrained(base, "Rudra-G-23/qwen2.5-coder-0.5b-python-fim-lora")
4. Continue training OR just evaluate
5. Push updated adapter back to HF at session end
```

The base model never changes. Only your adapter accumulates updates over weeks.

---

## 📌 Quick Cheat Sheet

| Task | Where |
|---|---|
| Write / edit code | VS Code on WSL |
| Push changes | GitHub (`git push`) |
| Generate FIM data | Local (`python src/data_prep.py`) |
| Upload dataset | Kaggle Datasets (one-time) |
| Train LoRA | Kaggle Notebook (GPU T4) |
| Save adapter | Hugging Face (automatic) |
| Run evaluation | Kaggle Notebook (after training) |
| Deploy locally | Ollama via GGUF merge |

---

## ⚠️ Common Mistakes to Avoid

1. **Don't put logic in the notebook.** Logic goes in `src/`. Notebook only clones + calls.
2. **Don't hardcode your HF token.** Use Kaggle Secrets → `os.environ["HF_TOKEN"]`.
3. **Don't save the full model.** `model.save_pretrained()` on a PeftModel saves just the adapter.
4. **Don't train without a validation split.** You need val loss to pick the best checkpoint.
5. **Don't report only training loss.** Val loss is what matters and what reviewers check.
6. **Don't skip `is_valid_python()`.** Clean data beats large noisy data at 0.5B scale.
7. **Don't use `load_best_model_at_end=False`.** Always let the trainer pick the best checkpoint by val loss.

---

## 🎯 Minimum Path to First Working Run (30 min)

1. Run `data_prep.py` on this repo's `src/` folder → ~50 FIM examples
2. Upload that tiny JSONL to Kaggle Dataset
3. Run Kaggle notebook with `num_epochs: 1`
4. Verify the adapter appears on Hugging Face
5. Run `evaluate.py` with `max_samples=20`

**This validates the entire pipeline in under 30 minutes. Then scale up data and epochs.**

---

*Last updated: 2026-08-10 | Repo: https://github.com/Rudra-G-23/qwen2.5-coder-0.5b-python-fim | Kaggle: https://www.kaggle.com/rudraprasadbhuyan | HF: https://huggingface.co/Rudra-G-23*
