"""
src/training/train_lora.py
LoRA fine-tuning of Qwen2.5-Coder-0.5B on a Python FIM dataset.

Workflow:
  1. Load Qwen2.5-Coder-0.5B via Unsloth (fast) or plain HF PEFT (fallback).
  2. Attach LoRA adapters — base weights stay FROZEN.
  3. Train on FIM triples read from a JSONL file.
  4. Save best checkpoint by validation loss (not last step).
  5. Push the adapter (not the full model!) to Hugging Face.
  6. Log all hyperparameters, metrics, and artifacts to Weights & Biases + Weave.

On Kaggle: called from notebooks/kaggle_wandb.ipynb or kaggle_train.ipynb.
Locally:   python src/training/train_lora.py  (dry-run: verifies config, no actual training)

Run ID format (deterministic, never random):
    {model_slug}-lora-r{r}-e{epochs}-ds{dataset_version}-{YYYYMMDD}-a{attempt}
    Example: qwen05b-lora-r16-e3-dsv1-20260810-a1
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainerCallback
from trl import SFTConfig

# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/training/lora.yaml") -> dict:
    """Load YAML config and return as a plain dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` onto a copy of `base`. Nested dicts are
    merged key-by-key; any other value type is replaced outright."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ── Run ID builder ────────────────────────────────────────────────────────────

def build_run_id(cfg: dict) -> str:
    """
    Construct a deterministic, human-readable run ID from config fields.

    Format:
        {model_slug}-lora-r{r}-e{epochs}-ds{dataset_version}-{YYYYMMDD}-a{attempt}

    Example:
        qwen05b-lora-r16-e3-dsv1-20260810-a1

    Components:
        model_slug      — short slug derived from model name (e.g. qwen05b)
        lora            — training method (always lora)
        r{r}            — LoRA rank (e.g. r16)
        e{epochs}       — number of training epochs (e.g. e3)
        ds{version}     — dataset version from configs/training/lora.yaml wandb.dataset_version
        {YYYYMMDD}      — today's date (UTC)
        a{attempt}      — attempt/run number from configs/training/lora.yaml wandb.attempt
    """
    model_name: str = cfg["model"]["name"]
    lora_r: int = cfg["lora"]["r"]
    epochs: int = cfg["training"]["num_epochs"]
    wandb_cfg: dict = cfg.get("wandb", {})
    attempt: int = wandb_cfg.get("attempt", 1)
    dataset_version: str = str(wandb_cfg.get("dataset_version", "v1"))

    # Build a short, lowercase model slug from the HF repo name
    # "Qwen/Qwen2.5-Coder-0.5B" → "qwen05b"
    repo_part = model_name.split("/")[-1].lower()  # "qwen2.5-coder-0.5b"
    slug = (
        repo_part
        .replace("qwen2.5-coder-", "qwen")
        .replace(".", "")
        .replace("-", "")
    )
    # Normalise: "qwen05b" is the expected output for 0.5B
    slug = slug[:8]  # cap length

    date_str = datetime.utcnow().strftime("%Y%m%d")
    ds_tag = f"ds{dataset_version}"

    run_id = f"{slug}-lora-r{lora_r}-e{epochs}-{ds_tag}-{date_str}-a{attempt}"
    return run_id


# ── Small helpers ────────────────────────────────────────────────────────────

def _file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Hash a file's bytes so a W&B run can be tied to the exact data it trained on."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _resolve_model_revision(model_name: str) -> str:
    """Best-effort exact commit SHA for the base model on the HF Hub."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_name).sha or "unknown"
    except Exception:
        return "unknown"


def _gpu_type() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


# ── W&B + Weave initialisation ────────────────────────────────────────────────

def init_wandb(cfg: dict, run_id: str) -> Any | None:
    """
    Initialise Weights & Biases and Weave for experiment tracking.

    Returns the active wandb.Run if W&B is available and WANDB_API_KEY is set,
    otherwise returns None so training proceeds without tracking.

    Weave is initialised inside the same W&B run for joint trace + metric logging.
    """
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        print("⚠  WANDB_API_KEY not set — skipping W&B / Weave tracking.")
        print("   Set os.environ['WANDB_API_KEY'] before calling train() to enable.")
        return None

    try:
        import wandb
        import weave
    except ImportError as exc:
        print(f"⚠  W&B / Weave not installed ({exc}) — skipping tracking.")
        return None

    wandb_cfg = cfg.get("wandb", {})
    full_project: str = wandb_cfg.get("project", "qwen-coder-python-fim")

    if "/" in full_project:
        entity, project = full_project.split("/", 1)
    else:
        entity = None
        project = full_project

    # Flatten the full config into a plain dict for wandb.config
    flat_cfg: dict = {
        # model
        "model/name": cfg["model"]["name"],
        "model/max_seq_length": cfg["model"]["max_seq_length"],
        # lora
        "lora/r": cfg["lora"]["r"],
        "lora/alpha": cfg["lora"]["alpha"],
        "lora/dropout": cfg["lora"]["dropout"],
        "lora/target_modules": cfg["lora"]["target_modules"],
        # training
        "training/num_epochs": cfg["training"]["num_epochs"],
        "training/per_device_train_batch_size": cfg["training"]["per_device_train_batch_size"],
        "training/gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "training/effective_batch_size": (
            cfg["training"]["per_device_train_batch_size"]
            * cfg["training"]["gradient_accumulation_steps"]
        ),
        "training/learning_rate": cfg["training"]["learning_rate"],
        "training/warmup_steps": cfg["training"]["warmup_steps"],
        "training/lr_scheduler_type": cfg["training"]["lr_scheduler_type"],
        "training/fp16": cfg["training"]["fp16"],
        "training/seed": cfg["training"]["seed"],
        "training/optim": cfg["training"].get("optim", "adamw_torch"),
        # data
        "data/path": cfg["data"]["path"],
        "data/train_split": cfg["data"]["train_split"],
        "data/sha256": (
            _file_sha256(cfg["data"]["path"]) if os.path.exists(cfg["data"]["path"]) else "unknown"
        ),
        # exact model provenance — pins the run to the precise HF revision trained on
        "model/revision": _resolve_model_revision(cfg["model"]["name"]),
        # compute
        "compute/gpu_type": _gpu_type(),
        # wandb metadata
        "run/attempt": wandb_cfg.get("attempt", 1),
        "run/dataset_version": str(wandb_cfg.get("dataset_version", "v1")),
        "run/id": run_id,
    }

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_id,
        id=run_id,           # deterministic — resume same run if re-run with same ID
        resume="allow",      # allows resuming an interrupted run
        config=flat_cfg,
        tags=wandb_cfg.get("tags", []),
        notes=wandb_cfg.get("notes", ""),
        # Clusters related runs (e.g. one pilot variant's seeds) onto the
        # same W&B comparison panel automatically — data-stage-2.md §7's
        # one identified monitoring gap. None (default) leaves W&B's
        # ungrouped view, same as before this was added.
        group=wandb_cfg.get("group"),
    )

    # Initialise Weave inside the same W&B run for joint tracing
    weave.init(full_project)
    print(f"✓ W&B run  : {run.url}")
    print(f"  Run ID   : {run_id}")
    print(f"  Project  : {full_project}")

    return run


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_fim_dataset(
    path: str,
    train_split: float = 0.90,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Load JSONL FIM dataset, shuffle, and split into train / val.

    Each JSONL line must have a "text" field containing the full
    <|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>... string.

    Returns:
        (train_dataset, val_dataset)
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}. Run data_prep.py first.")

    random.seed(seed)
    random.shuffle(records)

    n_train = int(len(records) * train_split)
    train_records = records[:n_train]
    val_records = records[n_train:]

    print(f"Train examples : {len(train_records)}")
    print(f"Val examples   : {len(val_records)}")

    train_ds = Dataset.from_list([{"text": r["text"]} for r in train_records])
    val_ds = Dataset.from_list([{"text": r["text"]} for r in val_records])

    return train_ds, val_ds


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_with_unsloth(cfg: dict):
    """Fast path: Unsloth handles model + LoRA in one call."""
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,         # auto-detect: bf16 on A100, fp16 on T4/P100
        load_in_4bit=False, # full precision — fine at 0.5B on T4/P100
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimised variant
        random_state=cfg["training"]["seed"],
    )
    return model, tokenizer


def _load_with_peft(cfg: dict):
    """Fallback: plain HuggingFace transformers + PEFT (slightly slower)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    return model, tokenizer


# ── Mid-training HF checkpoint mirror (resume across a lost session) ──────────
#
# Kaggle sessions can drop (12h cap, disconnect, OOM) mid-training, and
# /kaggle/working does not survive past that session. Without this, a lost
# session meant re-training the whole run from the base model (see
# scripts/run_pilot_experiment.py's former "known limitation" note — the
# unit of resume was a whole (variant, seed) run, not a training step).
# HFCheckpointCallback mirrors each local Trainer checkpoint to
# `checkpoints/{run_id}/checkpoint-{step}` in the same HF model repo the
# final adapter is pushed to, and find_resume_checkpoint/
# download_resume_checkpoint let a fresh session pick up from the latest one.
# All HF Hub calls go through the module-level HfApi/snapshot_download
# references above so tests can monkeypatch them without network access.

def _checkpoint_prefix(run_id: str) -> str:
    return f"checkpoints/{run_id}"


def _latest_pointer_path(run_id: str) -> str:
    return f"{_checkpoint_prefix(run_id)}/latest_checkpoint.json"


class HFCheckpointCallback(TrainerCallback):
    """Mirrors each local Trainer checkpoint to HF as it's saved, so a new
    session can resume from the latest step instead of the base model.
    Keeps only the `keep_last` most recent checkpoints on HF (mirroring
    Trainer's own `save_total_limit` for local disk) to bound storage."""

    def __init__(self, hf_repo: str, run_id: str, hf_token: str, keep_last: int = 2):
        self.hf_repo = hf_repo
        self.run_id = run_id
        self.hf_token = hf_token
        self.keep_last = keep_last

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        local_ckpt = Path(args.output_dir) / f"checkpoint-{step}"
        if not local_ckpt.exists():
            return control

        api = HfApi(token=self.hf_token)
        repo_ckpt_path = f"{_checkpoint_prefix(self.run_id)}/checkpoint-{step}"
        print(f"\n▶  Mirroring checkpoint-{step} to HF: {self.hf_repo}/{repo_ckpt_path}")
        api.create_repo(self.hf_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(local_ckpt),
            repo_id=self.hf_repo,
            path_in_repo=repo_ckpt_path,
            repo_type="model",
            commit_message=f"{self.run_id}: checkpoint at step {step}",
        )
        pointer = {
            "run_id": self.run_id,
            "latest_step": step,
            "path": repo_ckpt_path,
            "updated_at_utc": datetime.utcnow().isoformat() + "Z",
        }
        api.upload_file(
            path_or_fileobj=json.dumps(pointer, indent=2).encode("utf-8"),
            path_in_repo=_latest_pointer_path(self.run_id),
            repo_id=self.hf_repo,
            repo_type="model",
            commit_message=f"{self.run_id}: latest checkpoint -> step {step}",
        )
        self._prune_old(api, current_step=step)
        return control

    def _prune_old(self, api: HfApi, current_step: int) -> None:
        try:
            files = api.list_repo_files(self.hf_repo, repo_type="model")
        except Exception as exc:
            print(f"⚠  Could not list HF repo files to prune old checkpoints: {exc}")
            return

        prefix = f"{_checkpoint_prefix(self.run_id)}/checkpoint-"
        steps: set[int] = set()
        for f in files:
            if f.startswith(prefix):
                step_str = f[len(prefix):].split("/")[0]
                if step_str.isdigit():
                    steps.add(int(step_str))

        old_steps = sorted(s for s in steps if s != current_step)
        excess = len(old_steps) + 1 - self.keep_last
        for s in old_steps[:max(0, excess)]:
            try:
                api.delete_folder(
                    path_in_repo=f"{_checkpoint_prefix(self.run_id)}/checkpoint-{s}",
                    repo_id=self.hf_repo,
                    repo_type="model",
                    commit_message=f"{self.run_id}: prune checkpoint step {s}",
                )
            except Exception as exc:
                print(f"⚠  Could not prune old checkpoint step {s}: {exc}")


def find_resume_checkpoint(hf_repo: str, run_id: str, hf_token: str) -> dict[str, Any] | None:
    """Look up the latest mid-training checkpoint pushed for this exact
    run_id. Returns the pointer dict (see HFCheckpointCallback.on_save) or
    None if this run has never checkpointed to HF (fresh run, or a repo/HF
    hiccup — either way, falling through to training from scratch is the
    safe default)."""
    api = HfApi(token=hf_token)
    try:
        files = api.list_repo_files(hf_repo, repo_type="model")
    except Exception:
        return None
    if _latest_pointer_path(run_id) not in files:
        return None

    from huggingface_hub import hf_hub_download

    local_pointer = hf_hub_download(
        repo_id=hf_repo, filename=_latest_pointer_path(run_id), repo_type="model", token=hf_token,
    )
    with open(local_pointer, encoding="utf-8") as f:
        return json.load(f)


def download_resume_checkpoint(
    hf_repo: str, pointer: dict[str, Any], hf_token: str, local_dir: str,
) -> str:
    """Download the checkpoint folder `pointer` points at and return its
    local path, suitable for Trainer.train(resume_from_checkpoint=...)."""
    repo_ckpt_path = pointer["path"]
    snapshot_dir = snapshot_download(
        repo_id=hf_repo,
        repo_type="model",
        allow_patterns=f"{repo_ckpt_path}/*",
        token=hf_token,
        local_dir=local_dir,
    )
    return str(Path(snapshot_dir) / repo_ckpt_path)


def _cleanup_hf_checkpoints(hf_repo: str, run_id: str, hf_token: str) -> None:
    """Delete this run's mid-training checkpoints from HF once the run has
    finished and its final adapter is pushed — they've served their purpose
    (resuming an interrupted session) and just cost storage after that."""
    api = HfApi(token=hf_token)
    try:
        api.delete_folder(
            path_in_repo=_checkpoint_prefix(run_id),
            repo_id=hf_repo,
            repo_type="model",
            commit_message=f"{run_id}: training complete — drop mid-training checkpoints",
        )
    except Exception as exc:
        print(f"⚠  Could not clean up HF mid-training checkpoints: {exc}")


# ── Main training function ────────────────────────────────────────────────────

def train(
    config_path: str = "configs/training/lora.yaml",
    overrides: dict | None = None,
    finish_wandb: bool = True,
) -> Any | None:
    """
    Full LoRA training run driven entirely by the YAML config.
    Call this from the Kaggle notebook.

    W&B + Weave tracking is enabled automatically if WANDB_API_KEY is set.
    If the key is not set, training proceeds normally without any tracking.

    Args:
        config_path: Path to the base YAML config (unchanged hyperparameters).
        overrides:   Optional dict deep-merged onto the loaded config before
                     use — e.g. {"data": {"path": ...}, "training": {"seed": ...},
                     "output": {"hf_repo": ...}}. Used by
                     scripts/run_pilot_experiment.py to sweep dataset variant /
                     seed / output repo without duplicating lora.yaml per run.
        finish_wandb: Whether to call wandb_run.finish() before returning.
                     Default True (existing behavior). Pass False when a
                     caller needs to log more into this same run afterward
                     (e.g. run_pilot_experiment.py logging SAFIM eval results)
                     — logging into an already-finished run is silently
                     dropped by the W&B SDK, not an error, so this must be
                     set BEFORE calling run_evaluation()/run_safim_evaluation()
                     with this run, not worked around after the fact. The
                     caller then owns calling wandb_run.finish() itself once
                     all logging for this run is done.

    Returns:
        The active wandb.Run if W&B is enabled, else None.
        Pass this to run_evaluation(wandb_run=...) to log eval metrics
        into the same W&B run.
    """
    cfg = load_config(config_path)
    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # ── Build deterministic run ID ────────────────────────────────────────────
    run_id = build_run_id(cfg)

    print("=" * 60)
    print("LoRA Fine-tuning — Qwen2.5-Coder-0.5B  |  Python FIM")
    print("=" * 60)
    print(f"Config    : {config_path}")
    print(f"Run ID    : {run_id}")
    print(f"Model     : {cfg['model']['name']}")
    print(f"LoRA r    : {cfg['lora']['r']}   alpha : {cfg['lora']['alpha']}")
    print(f"Epochs    : {cfg['training']['num_epochs']}")
    print()

    # ── 1. Initialise W&B + Weave (graceful no-op if key not set) ────────────
    wandb_run = init_wandb(cfg, run_id)
    report_to = "wandb" if wandb_run is not None else "none"

    # ── 1b. Resolve HF push destination + look for a mid-training checkpoint
    # to resume from (survives a lost Kaggle session — see
    # HFCheckpointCallback above). Computed up front since both the resume
    # lookup and the final adapter push need it.
    output_dir = cfg["output"]["dir"]
    hf_repo = cfg["output"]["hf_repo"]
    hf_token = os.environ.get("HF_TOKEN")

    resume_from_checkpoint: str | None = None
    if hf_token and cfg["training"].get("resume_from_hf", True):
        pointer = find_resume_checkpoint(hf_repo, run_id, hf_token)
        if pointer is not None:
            resume_from_checkpoint = download_resume_checkpoint(
                hf_repo, pointer, hf_token, output_dir
            )
            print(
                f"↻ Resuming {run_id} from HF checkpoint step {pointer['latest_step']}"
                f" → {resume_from_checkpoint}"
            )
        else:
            print(f"  No prior HF checkpoint for {run_id} — starting fresh.")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    try:
        model, tokenizer = _load_with_unsloth(cfg)
        print("✓ Loaded with Unsloth (fast path)")
    except ImportError:
        model, tokenizer = _load_with_peft(cfg)
        print("✓ Loaded with plain PEFT (Unsloth not installed)")

    model.print_trainable_parameters()
    print()

    # ── 3. Load data ──────────────────────────────────────────────────────────
    train_ds, val_ds = load_fim_dataset(
        path=cfg["data"]["path"],
        train_split=cfg["data"]["train_split"],
        seed=cfg["training"]["seed"],
    )

    # ── 4. TrainingArguments ──────────────────────────────────────────────────
    t = cfg["training"]

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        lr_scheduler_type=t["lr_scheduler_type"],
        fp16=t["fp16"],
        optim=t.get("optim", "adamw_torch"),
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_total_limit=2,            # keep only last 2 checkpoints on disk
        load_best_model_at_end=True,   # pick by val loss, NOT last step
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,           # "wandb" if key is set, else "none"
        run_name=run_id,               # deterministic name shown in W&B
        seed=t["seed"],
        dataloader_num_workers=2,
        average_tokens_across_devices=False,
    )

    # ── 5. SFTTrainer ─────────────────────────────────────────────────────────
    from trl import SFTTrainer  # type: ignore

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        args=training_args,
    )

    if hf_token:
        trainer.add_callback(HFCheckpointCallback(hf_repo=hf_repo, run_id=run_id, hf_token=hf_token))

    # WORKAROUND: Prevent PicklingError for SFTConfig when saving checkpoints
    _orig_torch_save = torch.save
    def _patched_torch_save(obj, f, *args, **kwargs):
        if type(obj).__name__ == "SFTConfig":
            return
        return _orig_torch_save(obj, f, *args, **kwargs)
    torch.save = _patched_torch_save

    # ── 6. Train ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n▶  Starting training…")
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("\n✓ Training complete")
    print(f"  Training loss : {result.training_loss:.4f}")
    print(f"  Runtime       : {result.metrics.get('train_runtime', 0):.0f}s")
    print(f"  Samples/sec   : {result.metrics.get('train_samples_per_second', 0):.1f}")

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    )

    # ── 7. Save adapter (NOT the full model) ──────────────────────────────────
    adapter_path = Path(output_dir) / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    adapter_size_mb = (
        sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file())
        / 1024 / 1024
    )
    print(f"\n✓ Adapter saved → {adapter_path}")
    print(f"  Size: {adapter_size_mb:.1f} MB")

    # ── 8. Log final training summary to W&B ─────────────────────────────────
    if wandb_run is not None:
        import wandb

        wandb.summary.update({
            "train/loss_final": result.training_loss,
            "train/runtime_s": result.metrics.get("train_runtime", 0),
            "train/samples_per_second": result.metrics.get("train_samples_per_second", 0),
            "adapter/size_mb": adapter_size_mb,
            "adapter/path": str(adapter_path),
            "compute/wall_clock_s": result.metrics.get("train_runtime", 0),
            "compute/peak_vram_mb": peak_vram_mb,
            "compute/gpu_type": _gpu_type(),
        })

    # ── 9. Push adapter to Hugging Face ───────────────────────────────────────
    # subfolder places this run inside a single shared repo (e.g.
    # "experiment/random_model") instead of giving every variant/approach
    # its own repo — see data-stage-1.md §7. "" pushes to the repo root.
    subfolder = cfg["output"].get("subfolder", "")
    if not hf_token:
        print("\n⚠  HF_TOKEN not set — skipping Hugging Face push.")
        print("   Set os.environ['HF_TOKEN'] before calling train() to enable push.")
        if wandb_run is not None and finish_wandb:
            wandb_run.finish()
        return wandb_run

    # Self-describing folder: a metadata.json alongside the adapter weights,
    # since a shared repo with multiple subfolders needs each one to carry
    # its own provenance rather than relying on a repo-level README.
    metadata = {
        "run_id": run_id,
        "base_model": cfg["model"]["name"],
        "lora": {
            "r": cfg["lora"]["r"],
            "alpha": cfg["lora"]["alpha"],
            "dropout": cfg["lora"]["dropout"],
        },
        "training": {"seed": cfg["training"]["seed"], "epochs": cfg["training"]["num_epochs"]},
        "dataset_version": str(cfg.get("wandb", {}).get("dataset_version", "v1")),
        "data_path": cfg["data"]["path"],
        "train_loss_final": result.training_loss,
        "adapter_size_mb": adapter_size_mb,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(adapter_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    from src.training.report_tables import render_hyperparameter_table

    hyperparameter_table_path = adapter_path / "hyperparameter_table.md"
    hyperparameter_table_path.write_text(render_hyperparameter_table(cfg), encoding="utf-8")
    if wandb_run is not None:
        import wandb

        wandb_run.log({"experiment_setup/hyperparameter_table": wandb.Table(
            columns=["Hyperparameter", "Value"],
            data=[
                [cell.strip() for cell in line.split("|")[1:3]]
                for line in render_hyperparameter_table(cfg).splitlines()[2:]
            ],
        )})

    dest = f"{hf_repo}/{subfolder}" if subfolder else hf_repo
    print(f"\n▶  Pushing adapter to HF: {dest}")
    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=hf_repo,
        path_in_repo=subfolder,
        repo_type="model",
        commit_message=f"{run_id}: push adapter" + (f" → {subfolder}" if subfolder else ""),
    )
    hf_url = f"https://huggingface.co/{hf_repo}" + (f"/tree/main/{subfolder}" if subfolder else "")
    print(f"✓ Adapter pushed → {hf_url}")

    # Run finished and the final adapter is safely on HF — the mid-training
    # checkpoints that existed only to survive a lost session are no longer
    # needed. Best-effort: a failure here doesn't affect the run's success.
    _cleanup_hf_checkpoints(hf_repo, run_id, hf_token)

    # Log HF URL + a lightweight model artifact to W&B.
    # The adapter weights themselves live on HF (see hf_repo above) — the W&B
    # artifact only tracks metadata + a reference, never the weight files.
    if wandb_run is not None:
        import wandb

        wandb.summary["hf_repo_url"] = hf_url

        artifact = wandb.Artifact(
            name=f"{run_id}-adapter",
            type="model",
            description=f"LoRA adapter for {cfg['model']['name']} — Python FIM (weights on HF, not W&B)",
            metadata={
                "hf_repo": hf_repo,
                "hf_url": hf_url,
                "adapter_size_mb": adapter_size_mb,
                "base_model": cfg["model"]["name"],
                "lora_r": cfg["lora"]["r"],
                "lora_alpha": cfg["lora"]["alpha"],
                "lora_dropout": cfg["lora"]["dropout"],
                "target_modules": cfg["lora"]["target_modules"],
                "epochs": cfg["training"]["num_epochs"],
                "train_loss_final": result.training_loss,
            },
        )
        artifact.add_reference(f"https://huggingface.co/{hf_repo}", name="hf_repo")
        artifact.add_file(str(hyperparameter_table_path))
        wandb_run.log_artifact(artifact)

        if finish_wandb:
            wandb_run.finish()
            print(f"\n✓ W&B run finished → {wandb_run.url}")
        else:
            print(f"\n✓ W&B run still open (caller will finish it) → {wandb_run.url}")

    return wandb_run


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dry-run: verify config loads and dataset parses without launching training.
    # Real training is invoked from notebooks/kaggle_wandb.ipynb on Kaggle.
    cfg = load_config("configs/training/lora.yaml")
    run_id = build_run_id(cfg)
    print("Config loaded OK:")
    print(f"  model          : {cfg['model']['name']}")
    print(f"  lora r / alpha : {cfg['lora']['r']} / {cfg['lora']['alpha']}")
    print(f"  epochs         : {cfg['training']['num_epochs']}")
    print(f"  output repo    : {cfg['output']['hf_repo']}")
    print(f"  run id         : {run_id}")
    print(f"  wandb project  : {cfg.get('wandb', {}).get('project', '(not set)')}")
    print("\nTo actually train, run from the Kaggle notebook:")
    print("  from src.training.train_lora import train")
    print("  train(config_path='configs/training/lora.yaml')")
