"""
local_tests/test_exact_dedup.py
Unit tests for src/dedup/exact_dedup.py.

Usage:
    pytest local_tests/test_exact_dedup.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dedup.exact_dedup import ExactDedup, content_sha256


class TestContentSha256:
    def test_deterministic(self):
        assert content_sha256("x = 1") == content_sha256("x = 1")

    def test_different_content_different_hash(self):
        assert content_sha256("x = 1") != content_sha256("x = 2")

    def test_returns_hex_digest(self):
        h = content_sha256("x = 1")
        assert len(h) == 64
        int(h, 16)  # raises ValueError if not valid hex


class TestExactDedup:
    def test_first_occurrence_is_not_duplicate(self):
        dedup = ExactDedup()
        _, is_dup = dedup.add("x = 1")
        assert is_dup is False
        assert dedup.duplicate_count == 0

    def test_second_occurrence_is_duplicate(self):
        dedup = ExactDedup()
        dedup.add("x = 1")
        _, is_dup = dedup.add("x = 1")
        assert is_dup is True
        assert dedup.duplicate_count == 1

    def test_different_content_not_duplicate(self):
        dedup = ExactDedup()
        dedup.add("x = 1")
        _, is_dup = dedup.add("x = 2")
        assert is_dup is False
        assert dedup.duplicate_count == 0

    def test_len_reflects_unique_content_seen(self):
        dedup = ExactDedup()
        dedup.add("x = 1")
        dedup.add("x = 2")
        dedup.add("x = 1")  # duplicate, shouldn't grow the set
        assert len(dedup) == 2

    def test_seeded_from_existing_hashes_catches_cross_session_duplicate(self):
        """Resume must rebuild seen_hashes from prior chunks — see
        src.curation.checkpoint.rebuild_seen_hashes — otherwise a resumed
        session would silently re-admit a file collected in a prior one."""
        prior_hash = content_sha256("x = 1")
        dedup = ExactDedup(seen_hashes={prior_hash})
        _, is_dup = dedup.add("x = 1")
        assert is_dup is True

    def test_is_duplicate_does_not_mutate_state(self):
        dedup = ExactDedup()
        dedup.add("x = 1")
        h = content_sha256("x = 1")
        assert dedup.is_duplicate(h) is True
        assert dedup.duplicate_count == 0  # is_duplicate() alone shouldn't count anything
