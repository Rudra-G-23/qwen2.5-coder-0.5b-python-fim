"""
local_tests/test_evaluate_resume.py
Unit tests for src/training/evaluate.py's GPU-free resumability helpers —
the append-only JSONL checkpoint read/write and the FIM eval-set parquet
loader used by evaluate_model_by_type / run_evaluation_nway_by_type. Model
inference itself stays untested here, matching the existing project pattern
for evaluate_model/run_evaluation (no GPU in local_tests).

Usage:
    pytest local_tests/test_evaluate_resume.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq

import src.training.evaluate as ev


class TestReadResultsJsonl:
    def test_missing_file_returns_empty_list(self, tmp_path):
        assert ev._read_results_jsonl(str(tmp_path / "nope.jsonl")) == []

    def test_reads_back_appended_lines(self, tmp_path):
        path = tmp_path / "results.jsonl"
        ev._append_result(str(path), {"model": "Base", "task_id": "t0"})
        ev._append_result(str(path), {"model": "Base", "task_id": "t1"})
        records = ev._read_results_jsonl(str(path))
        assert [r["task_id"] for r in records] == ["t0", "t1"]

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "results.jsonl"
        ev._append_result(str(path), {"model": "Base", "task_id": "t0"})
        assert path.exists()


class TestLoadAlreadyScored:
    def test_local_file_present_no_hf_lookup_attempted(self, tmp_path, monkeypatch):
        path = tmp_path / "Base_results.jsonl"
        ev._append_result(str(path), {"model": "Base", "task_id": "t0"})
        ev._append_result(str(path), {"model": "Base", "task_id": "t1"})

        def _unexpected_download(*a, **k):
            raise AssertionError("must not hit HF when a local file already exists")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _unexpected_download)

        scored = ev._load_already_scored(str(path), "Base")
        assert scored == {"t0", "t1"}

    def test_filters_to_requested_model_label_only(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        ev._append_result(str(path), {"model": "Base", "task_id": "t0"})
        ev._append_result(str(path), {"model": "Random-LoRA", "task_id": "t1"})
        assert ev._load_already_scored(str(path), "Base") == {"t0"}
        assert ev._load_already_scored(str(path), "Random-LoRA") == {"t1"}

    def test_no_local_file_and_no_hf_repo_starts_empty(self, tmp_path):
        path = tmp_path / "nope.jsonl"
        assert ev._load_already_scored(str(path), "Base", hf_checkpoint_repo=None) == set()

    def test_restores_from_hf_checkpoint_when_local_missing(self, tmp_path, monkeypatch):
        remote_path = tmp_path / "remote" / "Base.jsonl"
        remote_path.parent.mkdir(parents=True)
        remote_path.write_text(
            json.dumps({"model": "Base", "task_id": "t0"}) + "\n"
            + json.dumps({"model": "Base", "task_id": "t1"}) + "\n"
        )

        def fake_hf_hub_download(repo_id, filename, repo_type="dataset", token=None):
            assert filename == "experiment/safim_eval_1000_results/Base.jsonl"
            return str(remote_path)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

        local_path = tmp_path / "local" / "Base_results.jsonl"
        scored = ev._load_already_scored(
            str(local_path), "Base", hf_checkpoint_repo="fake/repo", token=None
        )
        assert scored == {"t0", "t1"}
        assert local_path.exists()  # restored locally for subsequent appends

    def test_hf_lookup_failure_starts_empty_not_an_error(self, tmp_path, monkeypatch):
        def fake_hf_hub_download(repo_id, filename, repo_type="dataset", token=None):
            raise RuntimeError("404 not found")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

        local_path = tmp_path / "Base_results.jsonl"
        scored = ev._load_already_scored(str(local_path), "Base", hf_checkpoint_repo="fake/repo")
        assert scored == set()


class TestMirrorCheckpointToHf:
    def test_uploads_to_sibling_results_folder(self, tmp_path, monkeypatch):
        path = tmp_path / "Base_results.jsonl"
        ev._append_result(str(path), {"model": "Base", "task_id": "t0"})

        uploads = []

        class FakeHfApi:
            def __init__(self, token=None):
                pass

            def upload_file(self, **kwargs):
                uploads.append(kwargs)

        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

        ev._mirror_checkpoint_to_hf(
            str(path), "fake/repo", "experiment/safim_eval_1000_results", "Base", token=None
        )
        assert len(uploads) == 1
        assert uploads[0]["path_in_repo"] == "experiment/safim_eval_1000_results/Base.jsonl"
        assert uploads[0]["repo_id"] == "fake/repo"
        assert uploads[0]["repo_type"] == "dataset"


class TestLoadFimEvalRecords:
    def test_reads_every_column_including_fim_type_and_task_id(self, tmp_path):
        records = [
            {
                "fim_type": "line", "content_id": "c0", "prefix": "a", "suffix": "b",
                "middle": "c", "task_id": "safim_eval_0000",
            },
            {
                "fim_type": "block", "content_id": "c1", "prefix": "d", "suffix": "e",
                "middle": "f", "task_id": "safim_eval_0001",
            },
        ]
        path = tmp_path / "eval.parquet"
        pq.write_table(pa.Table.from_pylist(records), path)

        loaded = ev.load_fim_eval_records(str(path))
        assert len(loaded) == 2
        assert {r["task_id"] for r in loaded} == {"safim_eval_0000", "safim_eval_0001"}
        assert {r["fim_type"] for r in loaded} == {"line", "block"}
