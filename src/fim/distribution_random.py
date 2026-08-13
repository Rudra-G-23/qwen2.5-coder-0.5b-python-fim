"""
src/fim/distribution_random.py
Stage 1 §6 Variant A: sample FIM spans uniformly at random across the whole
bucketed pool, ignoring span type entirely. This is the baseline comparison
arm against distribution_planned.py's fixed-percentage sampling — same
source files, same bucketer, only the sampling strategy differs.
"""

from __future__ import annotations

import random
from typing import Any, Iterable

from src.fim.ast_bucketer import Span, build_fim_record, iter_bucketed_files, span_to_prefix_suffix_middle


def sample_random(
    files: Iterable[dict[str, str]],
    total_examples: int,
    max_spans_per_file: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Uniformly sample up to `total_examples` FIM records across every file's
    bucketed span pool, type-blind, capped at `max_spans_per_file` per file
    so no single large file dominates the variant.
    """
    rng = random.Random(seed)

    candidates: list[tuple[str, str, Span]] = list(iter_bucketed_files(files))
    rng.shuffle(candidates)

    records: list[dict[str, Any]] = []
    per_file_count: dict[str, int] = {}

    for content_id, content, span in candidates:
        if len(records) >= total_examples:
            break
        if per_file_count.get(content_id, 0) >= max_spans_per_file:
            continue

        sliced = span_to_prefix_suffix_middle(content, span)
        if sliced is None:
            continue
        prefix, suffix, middle = sliced

        records.append(build_fim_record(content_id, span, prefix, suffix, middle))
        per_file_count[content_id] = per_file_count.get(content_id, 0) + 1

    return records
