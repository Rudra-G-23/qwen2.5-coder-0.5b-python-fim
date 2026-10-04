"""
src/convert_gguf.py
Merge a LoRA adapter into the base model weights, then convert to GGUF
so the result can be loaded by Ollama / llama.cpp for local deployment.

Pipeline:
    Base model (HF safetensors)
          +
    LoRA adapter (HF peft)
          │
          ▼  merge_and_unload()   — folds adapter into base, returns a plain model
    Merged HF model  ──►  saved to disk
          │
          ▼  llama.cpp convert_hf_to_gguf.py
    Full-precision GGUF  ──►  used as-is or quantized further

Usage:
    python src/convert_gguf.py \
        --base-model  Qwen/Qwen2.5-Coder-0.5B \
        --adapter     /kaggle/working/checkpoints/lora_adapter \
        --merged-dir  /kaggle/working/merged_model \
        --gguf-out    /kaggle/working/qwen-fim.gguf \
        --llamacpp    ~/llama.cpp

Note: Run on Kaggle (GPU) or locally on CPU — 0.5B is small enough for CPU.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Step 1: Merge adapter into base weights ───────────────────────────────────

def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    merged_dir: str,
) -> None:
    """
    Load base model + LoRA adapter and merge them into a single set of weights.
    Saves the merged model (safetensors + config + tokenizer) to merged_dir.

    The merged model is a plain AutoModelForCausalLM with no PEFT dependency —
    ready for llama.cpp conversion.
    """
    print("=" * 60)
    print("Step 1 — Merge LoRA adapter into base weights")
    print("=" * 60)
    print(f"Base model   : {base_model_name}")
    print(f"Adapter path : {adapter_path}")
    print(f"Output dir   : {merged_dir}")
    print()

    print("Loading base model…")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Attaching LoRA adapter…")
    peft_model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging (merge_and_unload)…")
    merged = peft_model.merge_and_unload()  # adapter weights folded into base

    out = Path(merged_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model → {merged_dir}")
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"\n✓ Merged model saved ({size_mb:.0f} MB)")


# ── Step 2: Convert merged HF model → GGUF ───────────────────────────────────

def convert_to_gguf(
    merged_dir: str,
    gguf_out: str,
    llamacpp_dir: str,
) -> None:
    """
    Run llama.cpp's convert_hf_to_gguf.py to convert the merged HF model to GGUF.

    Args:
        merged_dir:   Directory containing the merged model (from merge_adapter).
        gguf_out:     Output path for the GGUF file.
        llamacpp_dir: Root directory of a cloned llama.cpp repo (must be built).
    """
    print()
    print("=" * 60)
    print("Step 2 — Convert HF model → GGUF")
    print("=" * 60)

    convert_script = Path(llamacpp_dir) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}\n"
            "Clone and build llama.cpp first:\n"
            "  git clone https://github.com/ggml-org/llama.cpp\n"
            "  cd llama.cpp && make"
        )

    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_out),
        "--outtype", "f16",   # full-precision GGUF (quantize separately if needed)
    ]
    print("Running:", " ".join(cmd))
    print()
    subprocess.run(cmd, check=True)
    print(f"\n✓ GGUF file → {gguf_out}")

    gguf_size = Path(gguf_out).stat().st_size / 1024 / 1024
    print(f"  Size: {gguf_size:.0f} MB")


# ── Step 3: Print Ollama Modelfile ────────────────────────────────────────────

def print_modelfile(gguf_path: str) -> None:
    """Print the Ollama Modelfile to stdout. Save it and run `ollama create`."""
    print()
    print("=" * 60)
    print("Step 3 — Ollama Modelfile")
    print("=" * 60)
    modelfile = f"""\
# Modelfile for Qwen2.5-Coder-0.5B Python FIM
FROM {gguf_path}

# Qwen2.5-Coder FIM template
TEMPLATE \"\"\"<|fim_prefix|>{{{{ .Prompt }}}}<|fim_suffix|><|fim_middle|>\"\"\"

PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|fim_middle|>"
PARAMETER stop "<|fim_prefix|>"
PARAMETER stop "<|fim_suffix|>"
"""
    print(modelfile)
    print("─" * 60)
    print("To deploy:")
    print(f"  ollama create qwen-fim -f Modelfile")
    print(f"  ollama run qwen-fim")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and convert to GGUF for Ollama."
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="HF model name or local path for the base model",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the LoRA adapter (local dir or HF repo id)",
    )
    parser.add_argument(
        "--merged-dir",
        default="merged_model",
        help="Output directory for the merged HF model (default: merged_model)",
    )
    parser.add_argument(
        "--gguf-out",
        default="qwen-fim.gguf",
        help="Output path for the GGUF file (default: qwen-fim.gguf)",
    )
    parser.add_argument(
        "--llamacpp",
        default="~/llama.cpp",
        help="Root directory of a cloned + built llama.cpp repo",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the merge step (merged model already exists at --merged-dir)",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip the GGUF conversion step (just merge and print Modelfile)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.skip_merge:
        merge_adapter(
            base_model_name=args.base_model,
            adapter_path=args.adapter,
            merged_dir=args.merged_dir,
        )

    if not args.skip_gguf:
        convert_to_gguf(
            merged_dir=args.merged_dir,
            gguf_out=args.gguf_out,
            llamacpp_dir=str(Path(args.llamacpp).expanduser()),
        )

    print_modelfile(gguf_path=str(Path(args.gguf_out).resolve()))
