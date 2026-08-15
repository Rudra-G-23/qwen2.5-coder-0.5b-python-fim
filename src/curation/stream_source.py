"""
src/curation/stream_source.py
Stage 1 §2-§4: streams HuggingFaceCode/stack-v3-train, applies the six filter
stages from §3 in order, buffers survivors into chunks, and hands finished
chunks back to the caller (scripts/build_sample.py) for parquet + checkpoint
upload.

Schema note: the spec's §5 schema is written per-repo-with-nested-files[], but
this pipeline collects at file granularity (only some files in a repo survive
the filters) — so each output record is one FLATTENED file row: the repo-level
fields (repo_path, repo_id, commit_id, github_metadata, num_files) are copied
onto every file row alongside that file's own fields and curation_metadata.
Every field §5 lists is still present, just flattened rather than nested.

Never call `load_dataset(..., streaming=False)` or anything that would
materialize the full corpus — see .claude/data-v1.md §0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from src.curation.license_policy import decide_license
from src.curation.quality_filters import evaluate_quality, scan_for_secrets
from src.dedup.exact_dedup import ExactDedup

STAGE_ORDER = [
    "language_reject",
    "vendor_reject",
    "license_reject",
    "quality_reject",
    "secret_reject",
    "exact_dup_reject",
]


# ── Per-file filter pipeline (pure, no I/O — the unit-testable core) ─────────


@dataclass
class FilterResult:
    keep: bool
    reject_stage: str | None  # one of STAGE_ORDER, or None if kept
    content_hash_sha256: str | None = None
    license_type: str | None = None
    detected_licenses: list[str] = field(default_factory=list)
    quality_reason: str | None = None  # QualityDecision.reason, set only on quality_reject


def apply_filter_pipeline(
    file_record: dict[str, Any],
    config: dict[str, Any],
    dedup: ExactDedup,
) -> FilterResult:
    """Run one `files[]` entry through all six §3 stages, in order."""
    if file_record.get("language") != config["language"]["keep"]:
        return FilterResult(keep=False, reject_stage="language_reject")

    if file_record.get("is_vendor", False):
        return FilterResult(keep=False, reject_stage="vendor_reject")

    license_decision = decide_license(
        file_record.get("license_type"),
        file_record.get("detected_licenses"),
        config["license"]["keep_types"],
    )
    if not license_decision.keep:
        return FilterResult(
            keep=False,
            reject_stage="license_reject",
            license_type=license_decision.license_type,
            detected_licenses=license_decision.detected_licenses,
        )

    content = file_record.get("content", "")
    quality_cfg = config["quality"]
    quality = evaluate_quality(
        content=content,
        size_bytes=file_record.get("size_bytes", len(content.encode("utf-8"))),
        size_min=quality_cfg["size_bytes"]["min"],
        size_max=quality_cfg["size_bytes"]["max"],
        max_line_length=quality_cfg["max_line_length"],
        min_avg_line_length_for_minified_check=quality_cfg[
            "min_avg_line_length_for_minified_check"
        ],
        generated_markers=quality_cfg["generated_markers"],
    )
    if not quality.keep:
        return FilterResult(
            keep=False,
            reject_stage="quality_reject",
            license_type=license_decision.license_type,
            detected_licenses=license_decision.detected_licenses,
            quality_reason=quality.reason,
        )

    if scan_for_secrets(content, config["secrets"]["patterns"]):
        return FilterResult(
            keep=False,
            reject_stage="secret_reject",
            license_type=license_decision.license_type,
            detected_licenses=license_decision.detected_licenses,
        )

    content_hash, is_dup = dedup.add(content)
    if is_dup:
        return FilterResult(
            keep=False,
            reject_stage="exact_dup_reject",
            content_hash_sha256=content_hash,
            license_type=license_decision.license_type,
            detected_licenses=license_decision.detected_licenses,
        )

    return FilterResult(
        keep=True,
        reject_stage=None,
        content_hash_sha256=content_hash,
        license_type=license_decision.license_type,
        detected_licenses=license_decision.detected_licenses,
    )


def build_curated_record(
    repo: dict[str, Any],
    file_record: dict[str, Any],
    filter_result: FilterResult,
    shard_index: int,
    row_offset: int,
) -> dict[str, Any]:
    """Flatten one kept file into the §5 schema (see module docstring)."""
    return {
        "repo_path": repo.get("repo_path"),
        "repo_id": repo.get("repo_id"),
        "commit_id": repo.get("commit_id"),
        "github_metadata": repo.get("github_metadata"),
        "num_files": repo.get("num_files"),
        "content_id": file_record.get("content_id"),
        "content": file_record.get("content"),
        "size_bytes": file_record.get("size_bytes"),
        "file_path": file_record.get("file_path"),
        "file_timestamp": file_record.get("file_timestamp"),
        "language": file_record.get("language"),
        "is_vendor": file_record.get("is_vendor", False),
        "license_type": filter_result.license_type,
        "detected_licenses": filter_result.detected_licenses,
        "curation_metadata": {
            "content_hash_sha256": filter_result.content_hash_sha256,
            "passed_quality_filter": True,
            "ast_parseable": True,
            "secret_scan_flag": False,
            "source_shard_index": shard_index,
            "source_row_offset": row_offset,
            "collected_at_timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


# ── Chunk collection loop ─────────────────────────────────────────────────────


@dataclass
class ChunkResult:
    shard_index: int
    row_offset: int  # cumulative repo rows consumed once this chunk closed
    records: list[dict[str, Any]]
    raw_files_scanned: int
    filter_stats: dict[str, int]
    quality_reject_by_reason: dict[str, int]
    bytes_collected: int


def collect_chunk(
    repo_iterator: Iterator[dict[str, Any]],
    config: dict[str, Any],
    dedup: ExactDedup,
    chunk_size: int,
    start_row_offset: int,
    shard_index: int,
    max_bytes_remaining: float | None = None,
) -> tuple[ChunkResult | None, bool]:
    """
    Pull repos from `repo_iterator` (real: a streaming `datasets` iterator;
    tests: a plain list iterator — this function does no I/O of its own)
    until `chunk_size` files have survived all filters, the iterator is
    exhausted, or `max_bytes_remaining` (the --max-gb budget) is used up.

    Returns (chunk, exhausted). `chunk` is None only when the iterator was
    already exhausted with nothing collected this call.
    """
    records: list[dict[str, Any]] = []
    raw_scanned = 0
    stats = dict.fromkeys(STAGE_ORDER, 0)
    quality_reject_by_reason: dict[str, int] = {}
    row_offset = start_row_offset
    bytes_collected = 0
    exhausted = False

    while len(records) < chunk_size:
        try:
            repo = next(repo_iterator)
        except StopIteration:
            exhausted = True
            break
        row_offset += 1

        for file_record in repo.get("files", []):
            raw_scanned += 1
            result = apply_filter_pipeline(file_record, config, dedup)
            if not result.keep:
                stats[result.reject_stage] += 1
                if result.reject_stage == "quality_reject" and result.quality_reason:
                    quality_reject_by_reason[result.quality_reason] = (
                        quality_reject_by_reason.get(result.quality_reason, 0) + 1
                    )
                continue

            record = build_curated_record(repo, file_record, result, shard_index, row_offset)
            records.append(record)
            bytes_collected += file_record.get("size_bytes", 0) or 0

            if len(records) >= chunk_size:
                break
            if max_bytes_remaining is not None and bytes_collected >= max_bytes_remaining:
                break

        if max_bytes_remaining is not None and bytes_collected >= max_bytes_remaining:
            break

    if not records and exhausted:
        return None, True

    return (
        ChunkResult(
            shard_index=shard_index,
            row_offset=row_offset,
            records=records,
            raw_files_scanned=raw_scanned,
            filter_stats=stats,
            quality_reject_by_reason=quality_reject_by_reason,
            bytes_collected=bytes_collected,
        ),
        exhausted,
    )


# ── Real streaming source (network — not exercised by local_tests) ──────────


def iter_stack_v3_repos(
    dataset_name: str, split: str, row_offset: int = 0
) -> Iterator[dict[str, Any]]:
    """Never non-streaming — see .claude/data-v1.md §0."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    if row_offset:
        ds = ds.skip(row_offset)
    return iter(ds)
