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
    rng = random.Random(seed)

    pool_by_type: dict[str, list[tuple[str, str, Span]]] = {}
    for content_id, content, span in iter_bucketed_files(files):
        pool_by_type.setdefault(span.span_type, []).append((content_id, content, span))
    for candidates in pool_by_type.values():
        rng.shuffle(candidates)

    records: list[dict[str, Any]] = []
    per_file_count: dict[str, int] = {}
    shortfall: dict[str, int] = {}

    for span_type, pct in planned_distribution.items():
        target = round(total_examples * pct / 100)
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
