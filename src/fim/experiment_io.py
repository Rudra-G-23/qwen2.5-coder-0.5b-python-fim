"""
src/fim/experiment_io.py
HF dataset-repo I/O for `experiment/*` folders — loading a curated source
pool and pushing a sampled FIM variant's data + metadata. Moved out of
scripts/generate_fim_variants.py (where this logic originally lived, against
CLAUDE.md's "logic lives in src/, not scripts/" rule) once
scripts/generate_safim_eval_1000.py became a second real caller needing the
exact same three operations.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

# Above this many source chunk files, load_curated_files's "load everything
# into one Python list" approach risks OOM on a CPU-only Kaggle session —
# see the scaling-risk audit (.claude/dicussions/). 10k-file pilot at
# chunk_size=1000 is 10 chunks; this threshold gives headroom before
# warning, not a hard limit — full-corpus generation needs a streaming/
# reservoir-sampling rewrite of this module, not just a bigger threshold.
_LARGE_SOURCE_CHUNK_WARNING_THRESHOLD = 50


def load_curated_files(
    hf_repo: str, path_prefix: str, token: str | None
) -> list[dict[str, str]]:
    """Load the curated sample's content_id/content columns from every chunk
    parquet file under `{path_prefix}/data/` in the dataset repo. Non-streaming
    is fine here — every source pool this is pointed at is capped by
    construction (10k pilot sample, or the ~2k stack_v3_python_only_data_without_fim
    continuation pool — §0's streaming rule applies to the raw stack-v3-train
    pull, not this already-curated, size-bounded output). This assumption is NOT
    enforced in code: pointing this at a full-corpus-scale folder will try
    to hold every file's content in RAM at once and can OOM. See
    _LARGE_SOURCE_CHUNK_WARNING_THRESHOLD below for the loud-warning
    tripwire this function prints instead of failing silently.

    Reads each chunk file directly with pyarrow instead of
    `datasets.load_dataset("parquet", ...)`: each chunk was written
    independently (build_sample.py), so PyArrow infers that chunk's schema
    from its own rows alone — a nullable column (e.g. github_metadata) that's
    all-None in one chunk comes out typed `null` there but `string` in a
    chunk with real values. `load_dataset` unifies all files under one
    schema and errors trying to cast between those two types. Reading only
    the two columns this needs, per file, sidesteps that entirely — same
    approach checkpoint.rebuild_seen_hashes already uses for this reason."""
    api = HfApi(token=token)
    dir_prefix = f"{path_prefix}/data/" if path_prefix else ""
    parquet_files = sorted(
        f for f in api.list_repo_files(hf_repo, repo_type="dataset")
        if f.startswith(dir_prefix) and f.endswith(".parquet")
    )
    if len(parquet_files) > _LARGE_SOURCE_CHUNK_WARNING_THRESHOLD:
        print(
            f"⚠  {len(parquet_files)} source chunk files under {dir_prefix or '(repo root)'} — "
            "well beyond the 10k-file pilot's ~10 chunks. load_curated_files loads every "
            "file's content into memory at once and can OOM at this scale; if this crashes, "
            "that's why — see the scaling-risk audit before re-running with a bigger --max-gb."
        )

    records: list[dict[str, str]] = []
    for filename in parquet_files:
        local_path = hf_hub_download(
            repo_id=hf_repo, filename=filename, repo_type="dataset", token=token
        )
        table = pq.read_table(local_path, columns=["content_id", "content"])
        records.extend(table.to_pylist())
    return records


def push_variant(
    records: list[dict], hf_repo: str, folder: str, config_name: str, token: str | None
) -> None:
    table = pa.Table.from_pylist(records)
    local_path = f"/tmp/fim_{config_name}.parquet"
    pq.write_table(table, local_path)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{folder}/data/fim_{config_name}.parquet",
        repo_id=hf_repo,
        repo_type="dataset",
        token=token,
    )
    os.remove(local_path)


def push_metadata(hf_repo: str, folder: str, local_path: Path, token: str | None) -> None:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=f"{folder}/metadata.json",
        repo_id=hf_repo,
        repo_type="dataset",
        token=token,
    )


def folder_metadata_exists(
    hf_repo: str, folder: str, token: str | None, repo_type: str = "dataset"
) -> bool:
    """Idempotency check: does `{folder}/metadata.json` already exist on the
    default branch? Mirrors scripts/run_pilot_experiment.py's
    _subfolder_already_pushed (HfApi.file_exists), used by
    scripts/generate_safim_eval_1000.py to guarantee the fixed eval set is
    never silently regenerated on a re-run. Fails open to False (i.e.
    "not generated yet") on any exception — a repo/HF hiccup should fall
    through to a normal first-run generation, not a false "already exists"."""
    api = HfApi(token=token)
    try:
        return api.file_exists(repo_id=hf_repo, filename=f"{folder}/metadata.json", repo_type=repo_type)
    except Exception:
        return False
