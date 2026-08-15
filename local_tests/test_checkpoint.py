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
from huggingface_hub.utils import EntryNotFoundError

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


@pytest.fixture
def fake_hub_prefixed(tmp_path, monkeypatch):
    """Same shape as `fake_hub`, but checkpoint/data files live under
    myprefix/checkpoints/ and myprefix/data/ — mirrors the new
    sample_filtered_data_10000/ layout (data-stage-2.md §2)."""
    checkpoint_0 = ckpt.build_checkpoint(
        chunk_id="chunk_0000",
        chunk_file="myprefix/data/chunk_shard0000_row0-2.parquet",
        shard_index=0,
        row_offset=2,
        files_collected_this_chunk=2,
        raw_files_scanned_this_chunk=5,
        filter_stats={"language_reject": 3},
        cumulative_files_collected=2,
        quality_reject_by_reason={"ast_invalid": 1},
    )
    checkpoint_0_path = tmp_path / "checkpoint_chunk_0000.json"
    checkpoint_0_path.write_text(json.dumps(checkpoint_0))

    files_in_repo = [
        "myprefix/checkpoints/checkpoint_chunk_0000.json",
        "checkpoint_unrelated_at_root.json",  # must NOT be picked up when path_prefix="myprefix"
    ]
    local_paths = {"myprefix/checkpoints/checkpoint_chunk_0000.json": checkpoint_0_path}

    def fake_hf_hub_download(repo_id, filename, repo_type="dataset", token=None):
        return str(local_paths[filename])

    monkeypatch.setattr(ckpt, "HfApi", lambda token=None: FakeHfApi(files_in_repo))
    monkeypatch.setattr(ckpt, "hf_hub_download", fake_hf_hub_download)

    return {"checkpoint_0": checkpoint_0}


class TestPathPrefix:
    def test_list_checkpoint_files_scoped_to_prefix(self, fake_hub_prefixed):
        files = ckpt.list_checkpoint_files("fake/repo", path_prefix="myprefix")
        assert files == ["myprefix/checkpoints/checkpoint_chunk_0000.json"]

    def test_load_all_checkpoints_scoped_to_prefix(self, fake_hub_prefixed):
        checkpoints = ckpt.load_all_checkpoints("fake/repo", path_prefix="myprefix")
        assert len(checkpoints) == 1
        assert checkpoints[0]["chunk_id"] == "chunk_0000"

    def test_default_prefix_is_flat_repo_root(self, fake_hub):
        # fake_hub's files sit at repo root with no prefix — path_prefix=""
        # (the default) must still find them, preserving old behavior.
        files = ckpt.list_checkpoint_files("fake/repo")
        assert files == ["checkpoint_chunk_0000.json", "checkpoint_chunk_0001.json"]


class RecordingFakeHfApi(FakeHfApi):
    """Like FakeHfApi, but records upload_file calls so save_seen_hashes's
    output can be inspected without a real network call."""

    def __init__(self, files: list[str], token: str | None = None) -> None:
        super().__init__(files, token)
        self.uploads: list[dict] = []

    def upload_file(self, **kwargs) -> None:
        self.uploads.append(kwargs)


@pytest.fixture
def fake_hub_seen_hashes(tmp_path, monkeypatch):
    """Same two-chunk history as `fake_hub`, plus control over whether a
    seen_hashes.json cache exists — exercises load_seen_hashes's three paths
    (no cache yet / cache fully up to date / cache stale and needs a delta
    merge)."""
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
        chunk_id="chunk_0000", chunk_file="chunk_shard0000_row0-2.parquet",
        shard_index=0, row_offset=2, files_collected_this_chunk=2,
        raw_files_scanned_this_chunk=5, filter_stats={}, cumulative_files_collected=2,
    )
    checkpoint_1 = ckpt.build_checkpoint(
        chunk_id="chunk_0001", chunk_file="chunk_shard0001_row2-3.parquet",
        shard_index=1, row_offset=3, files_collected_this_chunk=1,
        raw_files_scanned_this_chunk=4, filter_stats={}, cumulative_files_collected=3,
    )
    checkpoints = [checkpoint_0, checkpoint_1]
    resume = checkpoint_1

    local_paths = {
        "chunk_shard0000_row0-2.parquet": chunk_0_path,
        "chunk_shard0001_row2-3.parquet": chunk_1_path,
    }
    state = {"seen_hashes_json": None}  # tests set this to inject a cache

    def fake_hf_hub_download(repo_id, filename, repo_type="dataset", token=None):
        if filename == "seen_hashes.json":
            if state["seen_hashes_json"] is None:
                raise EntryNotFoundError("no seen_hashes.json yet")
            path = tmp_path / "seen_hashes.json"
            path.write_text(json.dumps(state["seen_hashes_json"]))
            return str(path)
        return str(local_paths[filename])  # KeyError if a path this test shouldn't touch is requested

    api = RecordingFakeHfApi([])
    monkeypatch.setattr(ckpt, "HfApi", lambda token=None: api)
    monkeypatch.setattr(ckpt, "hf_hub_download", fake_hf_hub_download)

    return {"checkpoints": checkpoints, "resume": resume, "state": state, "api": api}


class TestSeenHashesPersistence:
    def test_no_cache_falls_back_to_full_rebuild(self, fake_hub_seen_hashes):
        f = fake_hub_seen_hashes
        seen = ckpt.load_seen_hashes("fake/repo", f["checkpoints"], f["resume"])
        assert seen == {"hash_a", "hash_b", "hash_c"}

    def test_up_to_date_cache_skips_downloading_any_chunk(self, fake_hub_seen_hashes):
        f = fake_hub_seen_hashes
        f["state"]["seen_hashes_json"] = {
            "as_of_shard_index": 1, "as_of_row_offset": 3,
            "hashes": ["hash_a", "hash_b", "hash_c"],
        }
        # If load_seen_hashes tried to download a chunk it shouldn't need on
        # the fast path, fake_hf_hub_download's local_paths lookup would
        # KeyError — reaching an assertion at all proves it didn't.
        seen = ckpt.load_seen_hashes("fake/repo", f["checkpoints"], f["resume"])
        assert seen == {"hash_a", "hash_b", "hash_c"}

    def test_stale_cache_merges_only_the_missing_tail(self, fake_hub_seen_hashes):
        f = fake_hub_seen_hashes
        # Cache only accounts for chunk_0000 — chunk_0001 must be filled in
        # via a delta download, not a full rebuild.
        f["state"]["seen_hashes_json"] = {
            "as_of_shard_index": 0, "as_of_row_offset": 2,
            "hashes": ["hash_a", "hash_b"],
        }
        seen = ckpt.load_seen_hashes("fake/repo", f["checkpoints"], f["resume"])
        assert seen == {"hash_a", "hash_b", "hash_c"}

    def test_save_writes_as_of_position_and_sorted_hashes(self, fake_hub_seen_hashes):
        f = fake_hub_seen_hashes
        ckpt.save_seen_hashes("fake/repo", "", {"hash_c", "hash_a", "hash_b"}, 1, 3)
        upload = f["api"].uploads[-1]
        assert upload["path_in_repo"] == "seen_hashes.json"
        payload = json.loads(upload["path_or_fileobj"])
        assert payload == {
            "as_of_shard_index": 1,
            "as_of_row_offset": 3,
            "hashes": ["hash_a", "hash_b", "hash_c"],
        }

    def test_save_scopes_path_to_prefix(self, fake_hub_seen_hashes):
        f = fake_hub_seen_hashes
        ckpt.save_seen_hashes("fake/repo", "myprefix", {"hash_a"}, 0, 2)
        upload = f["api"].uploads[-1]
        assert upload["path_in_repo"] == "myprefix/checkpoints/seen_hashes.json"


class TestRenderMetadata:
    def test_aggregates_reject_by_stage_and_quality_reasons(self, fake_hub_prefixed):
        checkpoints = ckpt.load_all_checkpoints("fake/repo", path_prefix="myprefix")
        metadata = ckpt.render_metadata(
            checkpoints,
            folder="myprefix",
            source={"dataset": "HuggingFaceCode/stack-v3-train", "split": "train"},
        )
        assert metadata["folder"] == "myprefix"
        assert metadata["counts"]["raw_files_scanned"] == 5
        assert metadata["counts"]["files_kept"] == 2
        assert metadata["counts"]["reject_by_stage"]["language_reject"] == 3
        assert metadata["counts"]["quality_reject_by_reason"] == {"ast_invalid": 1}

    def test_empty_checkpoints_produce_zeroed_counts(self):
        metadata = ckpt.render_metadata([], folder="empty", source={})
        assert metadata["counts"]["raw_files_scanned"] == 0
        assert metadata["counts"]["files_kept"] == 0
        assert metadata["counts"]["quality_reject_by_reason"] == {}
