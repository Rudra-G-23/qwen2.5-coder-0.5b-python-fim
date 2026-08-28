#!/usr/bin/env python
"""
scripts/generate_fim_variants.py
Stage 1 §6 / data-stage-2.md entrypoint: bucket the curated 10k-file sample
and sample the random and/or planned("distributed") FIM variants from it,
pushing each to its own experiment/{random_data,distributed_data}/ folder
in the HF dataset repo (configs/data/fim_distribution.yaml).

Usage:
    python scripts/generate_fim_variants.py                    # both variants
    python scripts/generate_fim_variants.py --variant random   # one variant —
        lets notebooks/stack_v3_data_pull/random_data_10000.ipynb and
        distributed_data_10000.ipynb each cover one variant independently.
    python scripts/generate_fim_variants.py --config configs/data/fim_distribution.yaml \\
        --source-repo Rudra-G-23/the-stack-v3-python-fim-data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from src.fim.ast_bucketer import (
    count_by_fim_type,
    plot_bucket_distribution,
    write_experiment_metadata_json,
    write_natural_distribution_report,
)
from src.fim.distribution_planned import sample_planned
from src.fim.distribution_random import sample_random
from src.fim.experiment_io import load_curated_files, push_metadata, push_variant

# random/planned (internal, code-facing) -> HF folder name (data-stage-2.md's
# repo tree, matches Rudra's requested layout) — the one place this mapping
# is made; the rest of the codebase keeps calling it "planned".
FOLDER_NAME = {"random": "random_data", "planned": "distributed_data"}

DECISION_NOTE = (
    "This is a generation-time span-distribution check, not a winner decision. "
    "The actual random-vs-distributed winner is chosen only after training + "
    "SAFIM eval comparison on both variants (data-stage-1.md §7-8) — "
    "scripts/analyze_pilot_results.py is what makes that call, not this "
    "notebook's bucket counts."
)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def push_chart(hf_repo: str, chart_path: Path, token: str | None) -> None:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(chart_path),
        path_in_repo=f"experiment/{chart_path.name}",
        repo_id=hf_repo,
        repo_type="dataset",
        token=token,
    )


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


def fetch_other_variant_counts(
    hf_repo: str, variant: str, token: str | None
) -> dict[str, int] | None:
    """Look up an already-pushed sibling variant's by_fim_type counts from
    its metadata.json in the HF repo, so a single-variant notebook run
    (--variant random or --variant planned) can still produce the full
    random-vs-distributed comparison chart once *both* variants exist,
    however many separate runs that took. Returns None if the sibling
    hasn't been generated yet — not an error, just "not comparable yet".
    Only treats a genuine 404 (file/repo not found) as "not generated yet" —
    a transient network/auth/rate-limit error propagates instead of being
    silently swallowed, which previously made the comparison chart appear
    or "defer" inconsistently across reruns of the exact same completed
    state."""
    folder = f"experiment/{FOLDER_NAME[variant]}"
    try:
        local_path = hf_hub_download(
            repo_id=hf_repo, filename=f"{folder}/metadata.json", repo_type="dataset", token=token
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    metadata = json.loads(Path(local_path).read_text(encoding="utf-8"))
    return metadata.get("counts", {}).get("by_fim_type")


def init_wandb_run(project: str, name: str, job_type: str, group: str, tags: list[str]):
    """`project` may be "entity/project" (recommended — data-curation and
    FIM-generation share the `stack-v3-python-fim-data` project, kept
    separate from training's `qwen-coder-python-fim` one) or a bare project
    name. Mirrors src/curation/wandb_logger.py's init_curation_run and
    src/training/train_lora.py's init_wandb — same entity/project split,
    same joint Weave init inside the W&B run."""
    if not os.environ.get("WANDB_API_KEY"):
        print("⚠  WANDB_API_KEY not set — skipping W&B experiment tracking.")
        return None

    try:
        import wandb
        import weave  # noqa: F401  (imported for side-effect: patch tracing)
    except ImportError as exc:
        print(f"⚠  W&B / Weave not installed ({exc}) — skipping experiment tracking.")
        return None

    full_project = project
    if "/" in full_project:
        entity, bare_project = full_project.split("/", 1)
    else:
        entity, bare_project = None, full_project

    run = wandb.init(
        entity=entity, project=bare_project, name=name, job_type=job_type, group=group, tags=tags
    )
    weave.init(full_project)
    return run


def run(config: dict, source_repo: str | None, only_variant: str | None, token: str | None) -> None:
    hf_source_repo = source_repo or config["source"]["hf_dataset_repo"]
    source_prefix = config["source"].get("path_prefix", "")
    files = load_curated_files(hf_source_repo, source_prefix, token)
    print(f"Loaded {len(files)} curated files from {hf_source_repo}/{source_prefix}")

    report_natural_distribution(files, config["planned_distribution"])

    sampling = config["sampling"]
    output_repo = config["output"]["hf_dataset_repo"]
    wandb_project = config.get("wandb", {}).get(
        "project", "521er1007-national-institute-of-technology-rourkela/stack-v3-python-fim-data"
    )

    variants_to_run = ["random", "planned"] if only_variant is None else [only_variant]

    generated: dict[str, list[dict]] = {}

    for variant in variants_to_run:
        folder = f"experiment/{FOLDER_NAME[variant]}"
        wandb_run = init_wandb_run(
            wandb_project,
            name=f"{FOLDER_NAME[variant]}-seed{sampling['seed']}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
            job_type=f"experiment-{FOLDER_NAME[variant].replace('_data', '')}",
            group=f"experiment-{sampling['seed']}",
            tags=[variant, FOLDER_NAME[variant]],
        )

        if variant == "random":
            records = sample_random(
                files,
                total_examples=sampling["total_examples"],
                max_spans_per_file=sampling["max_spans_per_file"],
                seed=sampling["seed"],
            )
            shortfall: dict[str, int] = {}
        else:
            records, shortfall = sample_planned(
                files,
                total_examples=sampling["total_examples"],
                max_spans_per_file=sampling["max_spans_per_file"],
                planned_distribution=config["planned_distribution"],
                seed=sampling["seed"],
            )
        print(f"{variant.capitalize()} variant ({FOLDER_NAME[variant]}): {len(records)} examples")
        if shortfall:
            print(f"  ⚠ shortfall vs. planned percentages: {shortfall}")

        generated[variant] = records

        push_variant(records, output_repo, folder, config["output"]["configs"][variant], token)
        meta_path = write_experiment_metadata_json(
            FOLDER_NAME[variant], records, sampling, shortfall
        )
        push_metadata(output_repo, folder, meta_path, token)

        if wandb_run is not None:
            wandb_run.log(
                {f"bucket_distribution/{k}": v for k, v in count_by_fim_type(records).items()}
            )
            wandb_run.finish()

    # The random-vs-distributed comparison chart needs both variants' counts.
    # A single `--variant` run only has one in `generated` — look up the
    # other from its already-pushed metadata.json (may be from an earlier,
    # separate notebook run) before deciding whether the chart is possible.
    counts_by_variant = {v: count_by_fim_type(records) for v, records in generated.items()}
    for variant in ("random", "planned"):
        if variant not in counts_by_variant:
            other_counts = fetch_other_variant_counts(output_repo, variant, token)
            if other_counts is not None:
                counts_by_variant[variant] = other_counts

    if "random" in counts_by_variant and "planned" in counts_by_variant:
        chart_path = plot_bucket_distribution(
            counts_by_variant["random"], counts_by_variant["planned"]
        )
        push_chart(output_repo, chart_path, token)
        print(f"\n✓ Bucket-distribution chart (random vs distributed) → {chart_path}")
    else:
        missing = FOLDER_NAME["random" if "random" not in counts_by_variant else "planned"]
        print(
            f"\n… Comparison chart deferred: {missing} hasn't been generated yet "
            f"(run the other variant's notebook to complete it)."
        )

    print(f"\n✓ {'/'.join(FOLDER_NAME[v] for v in variants_to_run)} pushed → {output_repo}")
    print(f"\n{'=' * 70}\n{DECISION_NOTE}\n{'=' * 70}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random and/or distributed(planned) FIM variants from the curated sample."
    )
    parser.add_argument(
        "--config", default="configs/data/fim_distribution.yaml",
        help="Path to the FIM distribution config.",
    )
    parser.add_argument(
        "--source-repo", default=None,
        help="Override the curated-sample HF dataset repo (default: from config).",
    )
    parser.add_argument(
        "--variant", default=None, choices=["random", "planned"],
        help="Generate only this variant instead of both — lets each of "
             "notebooks/stack_v3_data_pull/{random,distributed}_data_10000.ipynb "
             "cover one variant independently.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN must be set to load the curated sample and push FIM variants.")
    run(cfg, source_repo=args.source_repo, only_variant=args.variant, token=hf_token)
