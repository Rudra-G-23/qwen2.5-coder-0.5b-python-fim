"""
local_tests/test_ast_bucketer.py
Unit tests for src/fim/ast_bucketer.py — span tagging across all eight bucket
types, including the api-call heuristic's two branches, and no-duplicate
regression coverage for the nested-statement recursion bug fixed during
development (see the docstring on _walk_expr).

Usage:
    pytest local_tests/test_ast_bucketer.py -v
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fim.ast_bucketer import bucket_spans, span_to_prefix_suffix_middle

SAMPLE_SOURCE = '''
import requests

class Foo:
    x = 1

    def bar(self, y):
        if y > 0:
            z = y + 1
            requests.get("http://x")
            return z
        else:
            return 0

def top_level(a, b):
    total = a + b
    for i in range(10):
        total += i
        print(i)
    return total
'''


class TestBucketSpans:
    def test_all_eight_bucket_types_present(self):
        spans = bucket_spans(SAMPLE_SOURCE)
        types = {s.span_type for s in spans}
        expected = {
            "line",
            "expression",
            "statement",
            "block",
            "function-body",
            "method-body",
            "class-level",
            "api-call",
        }
        assert expected <= types

    def test_api_call_heuristic_attribute_call(self):
        """requests.get(...) — Attribute-based call → api-call."""
        spans = bucket_spans(SAMPLE_SOURCE)
        api_calls = [s for s in spans if s.span_type == "api-call"]
        assert len(api_calls) == 1
        assert api_calls[0].node_type == "Call"

    def test_api_call_heuristic_bare_name_call_is_expression(self):
        """range(10) and print(i) — bare-name calls → expression, not api-call."""
        spans = bucket_spans(SAMPLE_SOURCE)
        expressions = [s for s in spans if s.span_type == "expression"]
        expr_lines = {s.start_line for s in expressions}
        # line 17 = "for i in range(10):", line 19 = "print(i)"
        assert 17 in expr_lines
        assert 19 in expr_lines

    def test_no_duplicate_spans(self):
        """Regression test for the _walk_expr recursion bug: every (type, node_type,
        start_line, end_line, enclosing_function) combination must be unique —
        the buggy version double-tagged expressions inside nested statement bodies."""
        spans = bucket_spans(SAMPLE_SOURCE)
        keys = [
            (s.span_type, s.node_type, s.start_line, s.end_line, s.enclosing_function)
            for s in spans
        ]
        assert len(keys) == len(set(keys))

    def test_class_level_span_excludes_methods(self):
        spans = bucket_spans(SAMPLE_SOURCE)
        class_level = [s for s in spans if s.span_type == "class-level"]
        assert len(class_level) == 1
        assert class_level[0].node_type == "Assign"
        assert class_level[0].enclosing_class == "Foo"

    def test_method_body_vs_function_body(self):
        spans = bucket_spans(SAMPLE_SOURCE)
        method_bodies = [s for s in spans if s.span_type == "method-body"]
        function_bodies = [s for s in spans if s.span_type == "function-body"]
        assert len(method_bodies) == 1
        assert method_bodies[0].enclosing_class == "Foo"
        assert len(function_bodies) == 1
        assert function_bodies[0].enclosing_class is None

    def test_block_requires_multiple_statements(self):
        spans = bucket_spans(SAMPLE_SOURCE)
        blocks = [s for s in spans if s.span_type == "block"]
        assert len(blocks) == 2  # the if-body (z=, requests.get, return z) and the for-body

    def test_nesting_depth_increases_inside_function(self):
        spans = bucket_spans(SAMPLE_SOURCE)
        module_import = next(s for s in spans if s.node_type == "Import")
        method_return = next(
            s for s in spans if s.node_type == "Return" and s.enclosing_function == "bar"
        )
        assert module_import.nesting_depth == 0
        assert method_return.nesting_depth > module_import.nesting_depth

    def test_line_spans_need_prefix_and_suffix_context(self):
        spans = bucket_spans(SAMPLE_SOURCE)
        line_spans = [s for s in spans if s.span_type == "line"]
        n_lines = len(SAMPLE_SOURCE.splitlines())
        assert all(1 < s.start_line < n_lines for s in line_spans)

    def test_syntax_error_raises(self):
        import pytest

        with pytest.raises(SyntaxError):
            bucket_spans("def foo(:\n    pass")

    def test_span_type_counts_are_stable(self):
        """Documents the exact tag counts for the fixture — guards against
        silent regressions in the walker."""
        spans = bucket_spans(SAMPLE_SOURCE)
        counts = Counter(s.span_type for s in spans)
        assert counts["api-call"] == 1
        assert counts["class-level"] == 1
        assert counts["method-body"] == 1
        assert counts["function-body"] == 1
        assert counts["block"] == 2


class TestSpanToPrefixSuffixMiddle:
    def test_slices_correctly(self):
        spans = bucket_spans(SAMPLE_SOURCE)
        method_body = next(s for s in spans if s.span_type == "method-body")
        result = span_to_prefix_suffix_middle(SAMPLE_SOURCE, method_body)
        assert result is not None
        prefix, suffix, middle = result
        lines = SAMPLE_SOURCE.splitlines()
        assert middle == "\n".join(lines[method_body.start_line - 1 : method_body.end_line])
        assert prefix == "\n".join(lines[: method_body.start_line - 1])
        assert suffix == "\n".join(lines[method_body.end_line :])

    def test_returns_none_without_prefix_context(self):
        from src.fim.ast_bucketer import Span

        span = Span(
            span_type="line",
            start_line=1,
            end_line=1,
            node_type="line",
            nesting_depth=0,
            enclosing_function=None,
            enclosing_class=None,
            structural_position="module_level",
        )
        assert span_to_prefix_suffix_middle("x = 1\ny = 2\nz = 3\n", span) is None

    def test_returns_none_without_suffix_context(self):
        from src.fim.ast_bucketer import Span

        source = "x = 1\ny = 2\nz = 3"
        n_lines = len(source.splitlines())
        span = Span(
            span_type="line",
            start_line=n_lines,
            end_line=n_lines,
            node_type="line",
            nesting_depth=0,
            enclosing_function=None,
            enclosing_class=None,
            structural_position="module_level",
        )
        assert span_to_prefix_suffix_middle(source, span) is None
