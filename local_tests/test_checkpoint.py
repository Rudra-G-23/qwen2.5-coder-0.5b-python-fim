"""
local_tests/test_checkpoint.py
Unit tests for src/curation/checkpoint.py — resume-position selection, the
dedup-hash-rebuild-on-resume logic, and filter-report rendering. All Hugging
Face Hub calls are monkeypatched; no real network access.

Usage:
    pytest local_tests/test_checkpoint.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.curation.checkpoint as ckpt


class FakeHfApi:
    """Stand-in for huggingface_hub.HfApi — records nothing, just serves a
    fixed file listing from the fake repo."""

    def __init__(self, files: list[str], token: str | None = None) -> None:
        self._files = files

    def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
        return list(self._files)

    def upload_file(self, **kwargs) -> None:
        pass


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Builds two chunk parquet files + two checkpoint JSONs on disk, and
    monkeypatches HfApi/hf_hub_download to serve them without any network
    call — mirrors what a resumed session would see in the real HF repo."""
    chunk_0_records = [
        {"curation_metadata": {"content_hash_sha256": "hash_a"}},
        {"curation_metadata": {"content_hash_sha256": "hash_b"}},
    ]
    chunk_1_records = [
        {"curation_metadata": {"content_hash_sha256": "hash_c"}},
    ]

    chunk_0_path = tmp_path / "chunk_shard0000_row0-2.parquet"
    chunk_1_path = tmp_path / "chunk_shard0001_row2-3.parquet"
    pq.write_table(pa.Table.from_pylist(chunk_0_records), chunk_0_path)
    pq.write_table(pa.Table.from_pylist(chunk_1_records), chunk_1_path)

    checkpoint_0 = ckpt.build_checkpoint(
        chunk_id="chunk_0000",
        chunk_file="chunk_shard0000_row0-2.parquet",
        shard_index=0,
        row_offset=2,
        files_collected_this_chunk=2,
        raw_files_scanned_this_chunk=5,
        filter_stats={"language_reject": 3},
        cumulative_files_collected=2,
    )
    checkpoint_1 = ckpt.build_checkpoint(
        chunk_id="chunk_0001",
        chunk_file="chunk_shard0001_row2-3.parquet",
        shard_index=1,
        row_offset=3,
        files_collected_this_chunk=1,
        raw_files_scanned_this_chunk=4,
        filter_stats={"language_reject": 2, "exact_dup_reject": 1},
        cumulative_files_collected=3,
    )
    checkpoint_0_path = tmp_path / "checkpoint_chunk_0000.json"
    checkpoint_1_path = tmp_path / "checkpoint_chunk_0001.json"
    checkpoint_0_path.write_text(json.dumps(checkpoint_0))
    checkpoint_1_path.write_text(json.dumps(checkpoint_1))

    files_in_repo = [
        "checkpoint_chunk_0000.json",
        "checkpoint_chunk_0001.json",
        "chunk_shard0000_row0-2.parquet",
        "chunk_shard0001_row2-3.parquet",
    ]
    local_paths = {
        "checkpoint_chunk_0000.json": checkpoint_0_path,
        "checkpoint_chunk_0001.json": checkpoint_1_path,
        "chunk_shard0000_row0-2.parquet": chunk_0_path,
        "chunk_shard0001_row2-3.parquet": chunk_1_path,
    }

    def fake_hf_hub_download(repo_id, filename, repo_type="dataset", token=None):
        return str(local_paths[filename])

    monkeypatch.setattr(ckpt, "HfApi", lambda token=None: FakeHfApi(files_in_repo))
    monkeypatch.setattr(ckpt, "hf_hub_download", fake_hf_hub_download)

    return {"checkpoint_0": checkpoint_0, "checkpoint_1": checkpoint_1}


class TestListAndLoadCheckpoints:
    def test_list_checkpoint_files_filters_to_checkpoints_only(self, fake_hub):
        files = ckpt.list_checkpoint_files("fake/repo")
        assert files == ["checkpoint_chunk_0000.json", "checkpoint_chunk_0001.json"]

    def test_load_all_checkpoints_parses_json(self, fake_hub):
        checkpoints = ckpt.load_all_checkpoints("fake/repo")
        assert len(checkpoints) == 2
        assert {c["chunk_id"] for c in checkpoints} == {"chunk_0000", "chunk_0001"}


class TestPickResumeCheckpoint:
    def test_empty_list_returns_none(self):
        assert ckpt.pick_resume_checkpoint([]) is None

    def test_picks_highest_shard_and_row_offset(self, fake_hub):
        checkpoints = ckpt.load_all_checkpoints("fake/repo")
        resume = ckpt.pick_resume_checkpoint(checkpoints)
        assert resume["chunk_id"] == "chunk_0001"
        assert resume["shard_index"] == 1


class TestRebuildSeenHashes:
    def test_rebuilds_hashes_from_chunks_up_to_resume_point(self, fake_hub):
        checkpoints = ckpt.load_all_checkpoints("fake/repo")
        resume = ckpt.pick_resume_checkpoint(checkpoints)
        seen = ckpt.rebuild_seen_hashes("fake/repo", checkpoints, resume)
        assert seen == {"hash_a", "hash_b", "hash_c"}

    def test_resuming_from_earlier_checkpoint_excludes_later_chunks(self, fake_hub):
        checkpoints = ckpt.load_all_checkpoints("fake/repo")
        earlier = fake_hub["checkpoint_0"]
        seen = ckpt.rebuild_seen_hashes("fake/repo", checkpoints, earlier)
        assert seen == {"hash_a", "hash_b"}


class TestChunkExists:
    def test_existing_chunk_detected(self, fake_hub):
        assert ckpt.chunk_exists("fake/repo", "chunk_shard0000_row0-2.parquet") is True

    def test_missing_chunk_not_detected(self, fake_hub):
        assert ckpt.chunk_exists("fake/repo", "chunk_shard9999_row0-2.parquet") is False


class TestRenderFilterReport:
    def test_aggregates_counts_and_cumulative(self, fake_hub):
        checkpoints = ckpt.load_all_checkpoints("fake/repo")
        report = ckpt.render_filter_report(checkpoints)
        assert "Chunks processed: 2" in report
        assert "Raw files scanned: 9" in report
        assert "Files collected (curated sample): 3" in report

    def test_empty_checkpoints_no_division_by_zero(self):
        report = ckpt.render_filter_report([])
        assert "n/a" in report
