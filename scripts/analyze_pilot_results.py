#!/usr/bin/env python
"""
scripts/analyze_pilot_results.py
Stage 1 §8 entrypoint: aggregate the per-(variant, seed) SAFIM pass@1 numbers
written by scripts/run_pilot_experiment.py into a random-vs-planned
comparison. Aggregation logic lives in
src.training.safim_eval.aggregate_pilot_results (kept in src/ so it's
testable without a GPU) — this script just loads the pilot config, calls it,
and prints/writes the result.

Usage:
    python scripts/analyze_pilot_results.py
    python scripts/analyze_pilot_results.py --pilot-config configs/training/pilot.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

import yaml

from src.training.safim_eval import write_pilot_comparison_report


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_report(report_path: Path) -> None:
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))

    print("Stage 1 pilot — SAFIM pass@1 by variant\n")
    for variant_name, stats in report["per_variant"].items():
        print(f"  {variant_name}")
        for seed, score in stats["scores_by_seed"].items():
            note = "MISSING" if score is None else f"{score:.4f}"
            print(f"    seed {seed}: {note}")
        if stats["mean"] is not None:
            print(f"    mean={stats['mean']:.4f}  range=[{stats['min']:.4f}, {stats['max']:.4f}]")
        print()

    print(f"Verdict: {report['verdict']}")
    print(f"\n→ full report written to {report_path}")
    print(f"→ bar chart written to {report_path.with_suffix('.png')}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Stage 1 pilot SAFIM results: random vs. planned."
    )
    parser.add_argument(
        "--pilot-config", default="configs/training/pilot.yaml",
        help="Path to the pilot sweep config.",
    )
    parser.add_argument(
        "--results-dir", default="results/pilot",
        help="Directory containing each seed's safim_metrics.csv (default: results/pilot).",
    )
    parser.add_argument(
        "--out-path", default="reports/pilot_comparison.json",
        help="Where to write the aggregated comparison report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pilot_cfg = load_config(args.pilot_config)
    report_path = write_pilot_comparison_report(
        pilot_cfg, results_dir=args.results_dir, out_path=args.out_path
    )
    _print_report(report_path)
