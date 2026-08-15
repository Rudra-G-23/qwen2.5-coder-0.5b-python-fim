"""
local_tests/test_stream_source.py
Unit tests for src/curation/stream_source.py — exercises the composed §3
filter pipeline and the chunk-collection loop directly, with a synthetic
in-memory repo iterator (no network, no real load_dataset call).

Usage:
    pytest local_tests/test_stream_source.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.curation.stream_source import apply_filter_pipeline, build_curated_record, collect_chunk
from src.dedup.exact_dedup import ExactDedup

CONFIG = {
    "language": {"keep": "Python"},
    "license": {"keep_types": ["permissive", "no_license"]},
    "quality": {
        "size_bytes": {"min": 10, "max": 100000},
        "max_line_length": 1000,
        "min_avg_line_length_for_minified_check": 300,
        "generated_markers": ["@generated"],
    },
    "secrets": {"patterns": [r"(?i)password\s*=\s*\S{8,}"]},
}


def _good_python_file(**overrides) -> dict:
    base = {
        "content_id": "abc123",
        "content": "def foo(x):\n    return x + 1\n",
        "size_bytes": 32,
        "file_path": "foo.py",
        "file_timestamp": 0,
        "language": "Python",
        "is_vendor": False,
        "license_type": "permissive",
        "detected_licenses": ["MIT"],
    }
    base.update(overrides)
    return base


class TestApplyFilterPipeline:
    def test_good_file_kept(self):
        result = apply_filter_pipeline(_good_python_file(), CONFIG, ExactDedup())
        assert result.keep is True
        assert result.reject_stage is None
        assert result.content_hash_sha256 is not None

    def test_non_python_rejected_at_language_stage(self):
        result = apply_filter_pipeline(
            _good_python_file(language="JavaScript"), CONFIG, ExactDedup()
        )
        assert result.keep is False
        assert result.reject_stage == "language_reject"

    def test_vendor_file_rejected(self):
        result = apply_filter_pipeline(_good_python_file(is_vendor=True), CONFIG, ExactDedup())
        assert result.keep is False
        assert result.reject_stage == "vendor_reject"

    def test_ambiguous_license_rejected(self):
        result = apply_filter_pipeline(
            _good_python_file(license_type="restricted"), CONFIG, ExactDedup()
        )
        assert result.keep is False
        assert result.reject_stage == "license_reject"

    def test_invalid_syntax_rejected_at_quality_stage(self):
        result = apply_filter_pipeline(
            _good_python_file(content="def foo(:\n    pass\n" * 5, size_bytes=100),
            CONFIG,
            ExactDedup(),
        )
        assert result.keep is False
        assert result.reject_stage == "quality_reject"
        assert result.quality_reason == "ast_invalid"

    def test_secret_rejected(self):
        result = apply_filter_pipeline(
            _good_python_file(content='password = "supersecret123"\nx = 1\n'),
            CONFIG,
            ExactDedup(),
        )
        assert result.keep is False
        assert result.reject_stage == "secret_reject"

    def test_duplicate_rejected_on_second_occurrence(self):
        dedup = ExactDedup()
        first = apply_filter_pipeline(_good_python_file(), CONFIG, dedup)
        second = apply_filter_pipeline(_good_python_file(), CONFIG, dedup)
        assert first.keep is True
        assert second.keep is False
        assert second.reject_stage == "exact_dup_reject"

    def test_stage_order_language_before_vendor(self):
        """A non-Python vendor file must be rejected at the language stage,
        not double-counted at the vendor stage."""
        result = apply_filter_pipeline(
            _good_python_file(language="JavaScript", is_vendor=True), CONFIG, ExactDedup()
        )
        assert result.reject_stage == "language_reject"


class TestBuildCuratedRecord:
    def test_flattens_repo_and_file_fields(self):
        repo = {
            "repo_path": "octocat/hello",
            "repo_id": 1,
            "commit_id": "deadbeef",
            "github_metadata": {"stars": 5},
            "num_files": 3,
        }
        file_record = _good_python_file()
        from src.curation.stream_source import apply_filter_pipeline

        result = apply_filter_pipeline(file_record, CONFIG, ExactDedup())
        record = build_curated_record(repo, file_record, result, shard_index=0, row_offset=1)

        assert record["repo_path"] == "octocat/hello"
        assert record["content"] == file_record["content"]
        assert record["curation_metadata"]["content_hash_sha256"] == result.content_hash_sha256
        assert record["curation_metadata"]["source_shard_index"] == 0
        assert record["license_type"] == "permissive"


class TestCollectChunk:
    def _repo(self, n_files: int, prefix: str) -> dict:
        return {
            "repo_path": prefix,
            "repo_id": 1,
            "commit_id": "x",
            "github_metadata": {},
            "num_files": n_files,
            "files": [
                _good_python_file(
                    content_id=f"{prefix}_{i}",
                    content=f"def {prefix}_f{i}(x):\n    return x + {i}\n",
                )
                for i in range(n_files)
            ],
        }

    def test_collects_up_to_chunk_size(self):
        repos = iter([self._repo(2, "r1"), self._repo(2, "r2"), self._repo(2, "r3")])
        chunk, exhausted = collect_chunk(
            repo_iterator=repos,
            config=CONFIG,
            dedup=ExactDedup(),
            chunk_size=3,
            start_row_offset=0,
            shard_index=0,
        )
        assert chunk is not None
        assert len(chunk.records) == 3
        assert exhausted is False

    def test_exhausted_iterator_returns_none_when_nothing_collected(self):
        chunk, exhausted = collect_chunk(
            repo_iterator=iter([]),
            config=CONFIG,
            dedup=ExactDedup(),
            chunk_size=10,
            start_row_offset=0,
            shard_index=0,
        )
        assert chunk is None
        assert exhausted is True

    def test_exhausted_iterator_returns_partial_chunk(self):
        repos = iter([self._repo(1, "r1")])
        chunk, exhausted = collect_chunk(
            repo_iterator=repos,
            config=CONFIG,
            dedup=ExactDedup(),
            chunk_size=10,
            start_row_offset=0,
            shard_index=0,
        )
        assert chunk is not None
        assert len(chunk.records) == 1
        assert exhausted is True

    def test_filter_stats_counted(self):
        repo = self._repo(1, "r1")
        repo["files"].append(_good_python_file(content_id="bad", language="JavaScript"))
        chunk, _ = collect_chunk(
            repo_iterator=iter([repo]),
            config=CONFIG,
            dedup=ExactDedup(),
            chunk_size=10,
            start_row_offset=0,
            shard_index=0,
        )
        assert chunk.filter_stats["language_reject"] == 1
        assert chunk.raw_files_scanned == 2

    def test_quality_reject_by_reason_counted(self):
        repo = self._repo(1, "r1")
        repo["files"].append(
            _good_python_file(content_id="bad", content="def foo(:\n    pass\n" * 5, size_bytes=100)
        )
        chunk, _ = collect_chunk(
            repo_iterator=iter([repo]),
            config=CONFIG,
            dedup=ExactDedup(),
            chunk_size=10,
            start_row_offset=0,
            shard_index=0,
        )
        assert chunk.filter_stats["quality_reject"] == 1
        assert chunk.quality_reject_by_reason == {"ast_invalid": 1}

    def test_max_bytes_remaining_stops_collection(self):
        repo = self._repo(5, "r1")
        chunk, _ = collect_chunk(
            repo_iterator=iter([repo]),
            config=CONFIG,
            dedup=ExactDedup(),
            chunk_size=10,
            start_row_offset=0,
            shard_index=0,
            max_bytes_remaining=32,  # one file's worth
        )
        assert len(chunk.records) == 1
