"""
src/dedup/exact_dedup.py
Stage 1 filter stage 6: exact (SHA-256) deduplication. Keeps the first
occurrence of any given file content, logs every later duplicate.

This is intentionally not near-deduplication — see .claude/data-v1.md §9
("Do not build near-deduplication yet").
"""

from __future__ import annotations

import hashlib


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ExactDedup:
    """
    Tracks content hashes seen so far across the whole collection run.

    Must be seeded with hashes from any previously collected chunks before a
    resumed session starts streaming again — see
    src.curation.checkpoint.rebuild_seen_hashes, which reads them back out of
    already-uploaded chunk parquet files. An in-memory-only set would only
    catch duplicates within a single session and silently miss cross-session
    duplicates.
    """

    def __init__(self, seen_hashes: set[str] | None = None) -> None:
        self._seen: set[str] = set(seen_hashes) if seen_hashes else set()
        self.duplicate_count: int = 0

    def __len__(self) -> int:
        return len(self._seen)

    def is_duplicate(self, content_hash: str) -> bool:
        return content_hash in self._seen

    def add(self, content: str) -> tuple[str, bool]:
        """
        Register `content`. Returns (content_hash, is_duplicate).

        On a duplicate, the hash is NOT re-added (already present) and
        duplicate_count is incremented; on a first occurrence, the hash is
        recorded and the caller should keep the file.
        """
        content_hash = content_sha256(content)
        if content_hash in self._seen:
            self.duplicate_count += 1
            return content_hash, True

        self._seen.add(content_hash)
        return content_hash, False
