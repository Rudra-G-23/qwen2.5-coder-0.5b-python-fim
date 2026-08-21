#!/usr/bin/env python
"""
scripts/download_wandb_media.py
Downloads everything W&B has for a run to a local folder for offline review:
- every file under the run's Files tab (config.yaml, output.log,
  requirements.txt, wandb-metadata.json, wandb-summary.json, artifact
  manifests, media/images/, media/table/, ...) — pass --prefix "" (default)
  to get all of them, or narrow to e.g. --prefix media/ for just images/tables.
- the full train/eval metrics history (run.history(): train/loss,
  eval/safim/*, learning rate, etc. — one row per logged step) as
  history.csv.
- the full system metrics history (run.history(stream="events"): per-step
  CPU/GPU/disk/memory utilization) as system_metrics.csv.

Reusable for any run: pass either --run-path (entity/project/run_id) or the
full run URL via --run-url, plus --output-dir.

Usage:
    python scripts/download_wandb_media.py \
        --run-path 521er1007-national-institute-of-technology-rourkela/qwen-coder-python-fim/qwen05b-lora-r16-e1-dsrandom-20260816-a42 \
        --output-dir artifacts/experiments/random

    # or paste the run URL directly:
    python scripts/download_wandb_media.py \
        --run-url https://wandb.ai/521er1007-national-institute-of-technology-rourkela/qwen-coder-python-fim/runs/qwen05b-lora-r16-e1-dsrandom-20260816-a42/files/media \
        --output-dir artifacts/experiments/random

Requires WANDB_API_KEY in the environment (e.g. `set -a; source .env; set +a`
before running — see .env.example, same convention as every other script
here).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

_URL_RE = re.compile(r"wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[^/?#]+)")


def _resolve_run_path(run_path: str | None, run_url: str | None) -> str:
    if run_path:
        return run_path
    if run_url:
        m = _URL_RE.search(run_url)
        if not m:
            raise SystemExit(f"Could not parse entity/project/run_id out of --run-url: {run_url!r}")
        return f"{m['entity']}/{m['project']}/{m['run_id']}"
    raise SystemExit("Pass --run-path entity/project/run_id, or --run-url with the full wandb.ai run URL.")


def download_run_media(run_path: str, output_dir: str, prefix: str = "media/") -> int:
    """Downloads every run file whose name starts with `prefix` (default
    "media/", which covers both media/images/ and media/table/) into
    output_dir, preserving the run's internal folder structure so the local
    layout matches what you see under <run-url>/files/media on wandb.ai.
    Returns the count of files downloaded."""
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not set — export it first (e.g. `set -a; source .env; set +a`).")

    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for f in run.files():
        if not f.name.startswith(prefix):
            continue
        f.download(root=str(out), replace=True, exist_ok=True)
        count += 1
        print(f"  ↓ {f.name}")

    print(f"\n✓ Downloaded {count} file(s) from {run_path} → {out}/{prefix}")
    return count


def download_run_metrics(run_path: str, output_dir: str) -> None:
    """Saves the run's full logged-metrics history (train/loss, eval/safim/*,
    learning_rate, etc.) and full system-metrics history (per-step CPU/GPU/
    disk/memory) as CSVs — these aren't part of run.files(), they come from a
    separate history API, so download_run_media() alone won't fetch them."""
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    history = run.history()
    history_path = out / "history.csv"
    history.to_csv(history_path, index=False)
    print(f"  ↓ history.csv  ({history.shape[0]} rows x {history.shape[1]} cols — train/eval metrics)")

    system_metrics = run.history(stream="events")
    system_path = out / "system_metrics.csv"
    system_metrics.to_csv(system_path, index=False)
    print(f"  ↓ system_metrics.csv  ({system_metrics.shape[0]} rows x {system_metrics.shape[1]} cols — CPU/GPU/disk)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download everything W&B has for a run (files + metrics history) for local offline review."
    )
    parser.add_argument("--run-path", help="entity/project/run_id")
    parser.add_argument("--run-url", help="Full https://wandb.ai/.../runs/<run_id> URL — parsed into --run-path.")
    parser.add_argument("--output-dir", required=True, help="Local folder to save downloaded files into.")
    parser.add_argument(
        "--prefix", default="",
        help="Only download run files whose name starts with this prefix (default: '' = every file).",
    )
    parser.add_argument(
        "--files-only", action="store_true",
        help="Skip history.csv / system_metrics.csv, only download files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_path = _resolve_run_path(args.run_path, args.run_url)
    download_run_media(run_path, args.output_dir, args.prefix)
    if not args.files_only:
        download_run_metrics(run_path, args.output_dir)


if __name__ == "__main__":
    main()
