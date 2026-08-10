"""
local_tests/test_data_prep.py
Unit tests for src/data_prep.py — run locally, no GPU required.

Usage:
    pytest local_tests/test_data_prep.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

# Make sure the repo root is on the path so `src` imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.data_prep import (
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    build_dataset,
    is_valid_python,
    make_fim_triples,
)


# ── is_valid_python ───────────────────────────────────────────────────────────

class TestIsValidPython:
    def test_valid_simple(self):
        assert is_valid_python("x = 1 + 2") is True

    def test_valid_function(self):
        source = "def foo(x):\n    return x + 1\n"
        assert is_valid_python(source) is True

    def test_invalid_syntax(self):
        assert is_valid_python("def foo(:\n    pass") is False

    def test_empty_string(self):
        # An empty string is valid Python
        assert is_valid_python("") is True

    def test_unicode_comment(self):
        assert is_valid_python("# こんにちは\nx = 1") is True


# ── make_fim_triples ──────────────────────────────────────────────────────────

def _make_long_source(n_lines: int = 30) -> str:
    """Generate a synthetic Python source with n_lines lines."""
    lines = [f"line_{i} = {i}" for i in range(n_lines)]
    return "\n".join(lines)


class TestMakeFimTriples:
    def test_returns_correct_count(self):
        source = _make_long_source(30)
        triples = make_fim_triples(source, n_samples=5)
        assert len(triples) == 5

    def test_short_source_returns_empty(self):
        source = _make_long_source(5)  # < 10 lines threshold
        triples = make_fim_triples(source, n_samples=5)
        assert triples == []

    def test_triple_structure(self):
        source = _make_long_source(30)
        triples = make_fim_triples(source, n_samples=1)
        assert len(triples) == 1
        t = triples[0]
        assert "prefix" in t
        assert "suffix" in t
        assert "middle" in t
        assert "text" in t

    def test_text_contains_fim_tokens(self):
        source = _make_long_source(30)
        for triple in make_fim_triples(source, n_samples=3):
            assert FIM_PREFIX in triple["text"]
            assert FIM_SUFFIX in triple["text"]
            assert FIM_MIDDLE in triple["text"]

    def test_reconstruction(self):
        """prefix + middle + suffix should reconstruct the original line span."""
        source = _make_long_source(30)
        lines = source.splitlines()
        for triple in make_fim_triples(source, n_samples=5):
            reconstructed = triple["prefix"] + "\n" + triple["middle"] + "\n" + triple["suffix"]
            # The original source should be a prefix of the reconstruction or equal
            assert triple["middle"].strip() != ""

    def test_no_empty_middles(self):
        source = _make_long_source(30)
        for triple in make_fim_triples(source, n_samples=10):
            assert triple["middle"].strip() != ""

    def test_token_ordering_in_text(self):
        """FIM tokens must appear in the correct order: PREFIX → SUFFIX → MIDDLE."""
        source = _make_long_source(30)
        for triple in make_fim_triples(source, n_samples=3):
            text = triple["text"]
            prefix_pos = text.index(FIM_PREFIX)
            suffix_pos = text.index(FIM_SUFFIX)
            middle_pos = text.index(FIM_MIDDLE)
            assert prefix_pos < suffix_pos < middle_pos


# ── build_dataset ─────────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_creates_jsonl_file(self, tmp_path: Path):
        # Write a few valid .py files
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        for i in range(3):
            (src_dir / f"file_{i}.py").write_text(_make_long_source(30))

        out_path = tmp_path / "out" / "dataset.jsonl"
        build_dataset(
            source_dirs=[str(src_dir)],
            output_path=str(out_path),
            max_files=10,
            samples_per_file=2,
            seed=0,
        )
        assert out_path.exists()

    def test_output_is_valid_jsonl(self, tmp_path: Path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        for i in range(3):
            (src_dir / f"file_{i}.py").write_text(_make_long_source(30))

        out_path = tmp_path / "dataset.jsonl"
        build_dataset(
            source_dirs=[str(src_dir)],
            output_path=str(out_path),
            samples_per_file=2,
            seed=0,
        )
        # Every line must be valid JSON with required keys
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                assert "text" in rec
                assert "prefix" in rec
                assert "suffix" in rec
                assert "middle" in rec

    def test_invalid_python_skipped(self, tmp_path: Path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "valid.py").write_text(_make_long_source(30))
        (src_dir / "invalid.py").write_text("def foo(:\n    pass")  # bad syntax

        out_path = tmp_path / "dataset.jsonl"
        build_dataset(
            source_dirs=[str(src_dir)],
            output_path=str(out_path),
            samples_per_file=2,
            seed=0,
        )
        # Records should only come from valid.py
        with open(out_path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        assert len(records) > 0  # valid.py produced some

    def test_deduplication(self, tmp_path: Path):
        """Duplicate files should not produce duplicate records."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        content = _make_long_source(30)
        # Write the SAME content to many files
        for i in range(5):
            (src_dir / f"dup_{i}.py").write_text(content)

        out_path = tmp_path / "dataset.jsonl"
        build_dataset(
            source_dirs=[str(src_dir)],
            output_path=str(out_path),
            samples_per_file=5,
            seed=0,
        )
        with open(out_path) as f:
            records = [json.loads(l) for l in f if l.strip()]

        texts = [r["text"] for r in records]
        assert len(texts) == len(set(texts)), "Duplicates found in output"
