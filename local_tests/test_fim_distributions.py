"""
local_tests/test_fim_distributions.py
Unit tests for src/fim/distribution_random.py and distribution_planned.py
against a synthetic multi-file bucketed pool.

Usage:
    pytest local_tests/test_fim_distributions.py -v
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fim.distribution_planned import sample_planned
from src.fim.distribution_random import sample_random

_TEMPLATE = '''
import requests

class Foo{n}:
    x = 1

    def bar(self, y):
        if y > 0:
            z = y + 1
            requests.get("http://x")
            return z
        else:
            return 0

def top_level_{n}(a, b):
    total = a + b
    for i in range(10):
        total += i
        print(i)
    return total
'''

PLANNED_DISTRIBUTION = {
    "line": 20,
    "expression": 15,
    "statement": 20,
    "block": 15,
    "function-body": 15,
    "method-body": 10,
    "class-level": 3,
    "api-call": 2,
}


def _files(n: int) -> list[dict]:
    return [{"content_id": f"file{i}", "content": _TEMPLATE.format(n=i)} for i in range(n)]


class TestSampleRandom:
    def test_respects_total_examples_cap(self):
        records = sample_random(_files(30), total_examples=50, max_spans_per_file=10, seed=1)
        assert len(records) == 50

    def test_respects_max_spans_per_file(self):
        records = sample_random(_files(5), total_examples=1000, max_spans_per_file=3, seed=1)
        counts = Counter(r["content_id"] for r in records)
        assert all(c <= 3 for c in counts.values())

    def test_deterministic_given_seed(self):
        a = sample_random(_files(10), total_examples=20, max_spans_per_file=5, seed=7)
        b = sample_random(_files(10), total_examples=20, max_spans_per_file=5, seed=7)
        assert a == b

    def test_different_seeds_can_differ(self):
        a = sample_random(_files(10), total_examples=20, max_spans_per_file=5, seed=1)
        b = sample_random(_files(10), total_examples=20, max_spans_per_file=5, seed=2)
        assert a != b

    def test_record_shape(self):
        records = sample_random(_files(5), total_examples=5, max_spans_per_file=5, seed=1)
        required_keys = {
            "fim_type", "content_id", "prefix", "suffix", "middle",
            "ast_depth", "nesting_depth", "node_type",
            "enclosing_function", "enclosing_class", "structural_position",
        }
        for record in records:
            assert required_keys <= record.keys()
            assert record["prefix"] or record["suffix"]  # at least one side non-empty

    def test_is_type_blind_not_uniform_but_covers_many_types(self):
        """Random shouldn't be restricted to a subset of bucket types the way
        the planned variant's percentages are — it should touch most/all of
        them given enough draws from a rich-enough pool."""
        records = sample_random(_files(30), total_examples=200, max_spans_per_file=20, seed=3)
        types = {r["fim_type"] for r in records}
        assert len(types) >= 5


class TestSamplePlanned:
    def test_respects_planned_percentages_within_tolerance(self):
        """Each type should land at (or, if pool-bound, below) its own planned
        target count — checked per-type against its own target rather than as
        a share of the overall total, since a shortfall in one type shifts
        the total and would otherwise skew every other type's share."""
        total_examples = 500
        records, shortfall = sample_planned(
            _files(100), total_examples=total_examples, max_spans_per_file=50,
            planned_distribution=PLANNED_DISTRIBUTION, seed=1,
        )
        counts = Counter(r["fim_type"] for r in records)
        assert len(records) > 0

        for span_type, pct in PLANNED_DISTRIBUTION.items():
            target = round(total_examples * pct / 100)
            if span_type in shortfall:
                assert counts.get(span_type, 0) < target
            else:
                assert counts.get(span_type, 0) == target

    def test_shortfall_reported_not_fabricated(self):
        """With very few files, rare buckets (class-level, api-call: 1 per
        file) can't fill their planned share — must be reported, not padded."""
        records, shortfall = sample_planned(
            _files(3), total_examples=500, max_spans_per_file=50,
            planned_distribution=PLANNED_DISTRIBUTION, seed=1,
        )
        assert "class-level" in shortfall or "api-call" in shortfall
        # every returned record must still be a genuine span from the pool
        for record in records:
            assert record["middle"].strip() != ""

    def test_respects_max_spans_per_file(self):
        _, _ = sample_planned(
            _files(5), total_examples=1000, max_spans_per_file=4,
            planned_distribution=PLANNED_DISTRIBUTION, seed=1,
        )
        records, _ = sample_planned(
            _files(5), total_examples=1000, max_spans_per_file=4,
            planned_distribution=PLANNED_DISTRIBUTION, seed=1,
        )
        counts = Counter(r["content_id"] for r in records)
        assert all(c <= 4 for c in counts.values())

    def test_deterministic_given_seed(self):
        a, _ = sample_planned(
            _files(10), total_examples=50, max_spans_per_file=10,
            planned_distribution=PLANNED_DISTRIBUTION, seed=5,
        )
        b, _ = sample_planned(
            _files(10), total_examples=50, max_spans_per_file=10,
            planned_distribution=PLANNED_DISTRIBUTION, seed=5,
        )
        assert a == b
