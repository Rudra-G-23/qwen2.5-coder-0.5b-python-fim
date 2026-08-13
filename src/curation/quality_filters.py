"""
src/curation/quality_filters.py
Stage 1 filter stage 4 (AST validity, size bounds, binary/minified/generated
heuristics) and stage 5 (secret detection).

Both stages live in this one module (matching .claude/data-v1.md §1's own tree
comment "AST validity, size bounds, secrets"), but callers must count them as
two distinct filter_stats counters — quality_reject and secret_reject — since
the checkpoint schema in §4 tracks them separately.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityDecision:
    keep: bool
    reason: str  # "ok" | "ast_invalid" | "size_out_of_bounds" | "binary" | "minified_or_generated"


def is_ast_parseable(content: str) -> bool:
    try:
        ast.parse(content)
        return True
    except (SyntaxError, ValueError):
        return False


def is_within_size_bounds(size_bytes: int, min_bytes: int, max_bytes: int) -> bool:
    return min_bytes <= size_bytes <= max_bytes


def looks_binary(content: str) -> bool:
    """Null bytes are the cheap, reliable binary-content signal."""
    return "\x00" in content


def looks_minified_or_generated(
    content: str,
    max_line_length: int,
    min_avg_line_length_for_minified_check: int,
    generated_markers: list[str],
) -> bool:
    """
    Reject files that are either:
      - generated (a known marker comment appears anywhere in the content), or
      - minified (at least one very long line AND a high average line length —
        avoids false-positiving on files with one long string literal in an
        otherwise normally-formatted file).
    """
    lower = content.lower()
    if any(marker in lower for marker in generated_markers):
        return True

    lines = content.splitlines()
    if not lines:
        return False

    longest = max(len(line) for line in lines)
    avg_len = sum(len(line) for line in lines) / len(lines)

    return longest > max_line_length and avg_len > min_avg_line_length_for_minified_check


def evaluate_quality(
    content: str,
    size_bytes: int,
    size_min: int,
    size_max: int,
    max_line_length: int,
    min_avg_line_length_for_minified_check: int,
    generated_markers: list[str],
) -> QualityDecision:
    """Run the stage-4 quality checks in a fixed, cheapest-first order."""
    if not is_within_size_bounds(size_bytes, size_min, size_max):
        return QualityDecision(keep=False, reason="size_out_of_bounds")

    if looks_binary(content):
        return QualityDecision(keep=False, reason="binary")

    if not is_ast_parseable(content):
        return QualityDecision(keep=False, reason="ast_invalid")

    if looks_minified_or_generated(
        content, max_line_length, min_avg_line_length_for_minified_check, generated_markers
    ):
        return QualityDecision(keep=False, reason="minified_or_generated")

    return QualityDecision(keep=True, reason="ok")


def scan_for_secrets(content: str, patterns: list[str]) -> bool:
    """Stage 5: return True if any configured secret-token pattern matches."""
    return any(re.search(pattern, content) for pattern in patterns)
