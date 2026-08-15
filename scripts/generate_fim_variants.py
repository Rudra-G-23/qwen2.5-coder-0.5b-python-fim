#!/usr/bin/env python
"""
scripts/generate_fim_variants.py
Stage 1 §6 entrypoint: bucket the curated 10k-file sample and sample both the
random and planned FIM variants from it, pushing each as a separate config
folder in the same HF dataset repo (configs/data/fim_distribution.yaml).

Usage:
    python scripts/generate_fim_variants.py
    python scripts/generate_fim_variants.py --config configs/data/fim_distribution.yaml \\
        --source-repo Rudra-G-23/qwen-coder-python-fim-data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi

from src.fim.ast_bucketer import write_natural_distribution_report
from src.fim.distribution_planned import sample_planned
from src.fim.distribution_random import sample_random


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_curated_files(hf_repo: str, token: str | None) -> list[dict[str, str]]:
    """Load the curated sample's content_id/content columns from every chunk
    parquet file in the dataset repo. Non-streaming is fine here — the
    curated sample is capped at 10k files by construction (§0's streaming
    rule applies to the raw stack-v3-train pull, not this already-curated,
    size-bounded output)."""
    ds = load_dataset(hf_repo, split="train", token=token)
    return [{"content_id": row["content_id"], "content": row["content"]} for row in ds]


def push_variant(
    records: list[dict], hf_repo: str, config_name: str, token: str | None
) -> None:
    table = pa.Table.from_pylist(records)
    local_path = f"/tmp/fim_{config_name}.parquet"
    pq.write_table(table, local_path)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{config_name}/fim_{config_name}.parquet",
        repo_id=hf_repo,
        repo_type="dataset",
        token=token,
    )
    os.remove(local_path)


def report_natural_distribution(files: list[dict[str, str]], planned_distribution: dict[str, float]) -> None:
    """Print the corpus's natural span-type frequency next to the hand-set
    planned percentages, so the gap between them is visible before training —
    not just asserted. Counting/rendering logic lives in
    src.fim.ast_bucketer.write_natural_distribution_report (mirrors how
    src.curation.checkpoint.write_filter_report keeps report logic in src/,
    not scripts/)."""
    report_path = write_natural_distribution_report(files, planned_distribution)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    natural_counts, natural_pct = report["natural_counts"], report["natural_percent"]

    print("\nNatural span-type distribution (unweighted, before sampling):")
    for span_type in sorted(natural_counts, key=natural_counts.get, reverse=True):
        planned_pct = planned_distribution.get(span_type)
        planned_note = f" (planned target: {planned_pct}%)" if planned_pct is not None else " (not in planned_distribution)"
        print(f"  {span_type:<15} {natural_counts[span_type]:>8} spans  {natural_pct[span_type]:>5.2f}%{planned_note}")
    print(f"  → written to {report_path}")


def run(config: dict, source_repo: str | None, token: str | None) -> None:
    hf_source_repo = source_repo or config["source"]["hf_dataset_repo"]
    files = load_curated_files(hf_source_repo, token)
    print(f"Loaded {len(files)} curated files from {hf_source_repo}")

    report_natural_distribution(files, config["planned_distribution"])

    sampling = config["sampling"]

    random_records = sample_random(
        files,
        total_examples=sampling["total_examples"],
        max_spans_per_file=sampling["max_spans_per_file"],
        seed=sampling["seed"],
    )
    print(f"Random variant   : {len(random_records)} examples")

    planned_records, shortfall = sample_planned(
        files,
        total_examples=sampling["total_examples"],
        max_spans_per_file=sampling["max_spans_per_file"],
        planned_distribution=config["planned_distribution"],
        seed=sampling["seed"],
    )
    print(f"Planned variant  : {len(planned_records)} examples")
    if shortfall:
        print(f"  ⚠ shortfall vs. planned percentages: {shortfall}")

    output_repo = config["output"]["hf_dataset_repo"]
    push_variant(random_records, output_repo, config["output"]["configs"]["random"], token)
    push_variant(planned_records, output_repo, config["output"]["configs"]["planned"], token)
    print(f"\n✓ Both variants pushed → {output_repo}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random + planned FIM variants from the curated sample."
    )
    parser.add_argument(
        "--config", default="configs/data/fim_distribution.yaml",
        help="Path to the FIM distribution config.",
    )
    parser.add_argument(
        "--source-repo", default=None,
        help="Override the curated-sample HF dataset repo (default: from config).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN must be set to load the curated sample and push FIM variants.")
    run(cfg, source_repo=args.source_repo, token=hf_token)
