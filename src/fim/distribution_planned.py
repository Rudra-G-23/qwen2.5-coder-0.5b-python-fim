"""
src/fim/distribution_planned.py
Stage 1 §6 Variant B: sample FIM spans per the fixed planned distribution
(20% line / 15% expression / 20% statement / 15% block / 15% function-body /
10% method-body / 3% class-level / 2% api-call, defined in
configs/data/fim_distribution.yaml), from the same bucketed pool as
distribution_random.py.

These percentages are a hand-set starting hypothesis, not a value derived
from measurement or literature — the pilot experiment (data-stage-1.md §7-8)
is what's meant to validate or falsify it, by comparing this variant's
downstream SAFIM result against both distribution_random.py's uniform
baseline and the corpus's own natural span-type frequency (see
scripts/generate_fim_variants.py's natural-distribution report).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from src.fim.ast_bucketer import (
    Span,
    build_fim_record,
    iter_bucketed_files,
    span_to_prefix_suffix_middle,
)


def _build_shuffled_pool(
    files: Iterable[dict[str, str]], seed: int
) -> dict[str, list[tuple[str, str, Span]]]:
    rng = random.Random(seed)
    pool_by_type: dict[str, list[tuple[str, str, Span]]] = {}
    for content_id, content, span in iter_bucketed_files(files):
        pool_by_type.setdefault(span.span_type, []).append((content_id, content, span))
    for candidates in pool_by_type.values():
        rng.shuffle(candidates)
    return pool_by_type


def _collect_by_type_targets(
    pool_by_type: dict[str, list[tuple[str, str, Span]]],
    targets: dict[str, int],
    max_spans_per_file: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Shared greedy-collection core: for each span_type, take up to
    `targets[span_type]` spans from its (already-shuffled) pool, skipping
    files already at `max_spans_per_file`, reporting a shortfall instead of
    fabricating spans when the pool runs dry. Used by both sample_planned
    (percentage-derived targets) and sample_fixed_bucket_counts (literal
    integer targets) — everything except how the per-type target number is
    computed is identical between the two."""
    records: list[dict[str, Any]] = []
    per_file_count: dict[str, int] = {}
    shortfall: dict[str, int] = {}

    for span_type, target in targets.items():
        collected = 0

        for content_id, content, span in pool_by_type.get(span_type, []):
            if collected >= target:
                break
            if per_file_count.get(content_id, 0) >= max_spans_per_file:
                continue

            sliced = span_to_prefix_suffix_middle(content, span)
            if sliced is None:
                continue
            prefix, suffix, middle = sliced

            records.append(build_fim_record(content_id, span, prefix, suffix, middle))
            per_file_count[content_id] = per_file_count.get(content_id, 0) + 1
            collected += 1

        if collected < target:
            shortfall[span_type] = target - collected

    return records, shortfall


def sample_planned(
    files: Iterable[dict[str, str]],
    total_examples: int,
    max_spans_per_file: int,
    planned_distribution: dict[str, float],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Sample per `planned_distribution` (span_type -> percent of total_examples;
    values should sum to ~100). Bounded by actual pool availability per type —
    if a rare bucket (e.g. class-level, api-call) can't fill its planned share
    at this sample size, the shortfall is reported, not backfilled with
    fabricated spans.

    Returns (records, shortfall_by_type) — shortfall_by_type only contains
    entries for types that came up short.
    """
    pool_by_type = _build_shuffled_pool(files, seed)
    targets = {
        span_type: round(total_examples * pct / 100)
        for span_type, pct in planned_distribution.items()
    }
    return _collect_by_type_targets(pool_by_type, targets, max_spans_per_file)


def sample_fixed_bucket_counts(
    files: Iterable[dict[str, str]],
    bucket_counts: dict[str, int],
    max_spans_per_file: int,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Sample per `bucket_counts` (span_type -> LITERAL integer count, e.g. the
    fixed 1000-task eval quota: line=150/statement=150/expression=150/
    block=120/function-body=120/method-body=100/api-call=120/class-level=90)
    — unlike sample_planned's percentage-of-total targets, these are exact
    counts, not shares of some larger total. Same pool-building, per-file
    cap, and honest-shortfall-reporting behavior as sample_planned (see
    _collect_by_type_targets); used to build a fixed, never-regenerated
    evaluation set rather than a training-data variant.

    Returns (records, shortfall_by_type) — shortfall_by_type only contains
    entries for types that came up short.
    """
    pool_by_type = _build_shuffled_pool(files, seed)
    return _collect_by_type_targets(pool_by_type, bucket_counts, max_spans_per_file)
