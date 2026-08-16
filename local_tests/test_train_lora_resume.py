"""
local_tests/test_train_lora_resume.py
Unit tests for src/training/train_lora.py's mid-training HF checkpoint
mirror + resume logic (HFCheckpointCallback, find_resume_checkpoint,
download_resume_checkpoint, _cleanup_hf_checkpoints). All Hugging Face Hub
calls are monkeypatched — no network access.

Run:
    pytest local_tests/test_train_lora_resume.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.training import train_lora


class FakeHfApi:
    """Records every call so tests can assert on what was pushed/pruned,
    and backs list_repo_files/upload_folder/upload_file/delete_folder with
    an in-memory "repo" dict shared across instances via `state`."""

    def __init__(self, token=None, state=None):
        self.token = token
        self.state = state if state is not None else {"files": set(), "uploaded": []}

    def create_repo(self, repo_id, repo_type=None, exist_ok=True):
        return None

    def list_repo_files(self, repo_id, repo_type=None):
        return sorted(self.state["files"])

    def upload_folder(self, folder_path, repo_id, path_in_repo, repo_type=None, commit_message=None):
        self.state["files"].add(f"{path_in_repo}/adapter_model.bin")
        self.state["uploaded"].append(("folder", path_in_repo, commit_message))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
        self.state["files"].add(path_in_repo)
        self.state[f"content:{path_in_repo}"] = path_or_fileobj
        self.state["uploaded"].append(("file", path_in_repo, commit_message))

    def delete_folder(self, path_in_repo, repo_id, repo_type=None, commit_message=None):
        removed = {f for f in self.state["files"] if f.startswith(path_in_repo)}
        if not removed:
            raise Exception("nothing to delete")
        self.state["files"] -= removed
        self.state["uploaded"].append(("delete", path_in_repo, commit_message))


@pytest.fixture
def fake_state():
    return {"files": set(), "uploaded": []}


@pytest.fixture
def patch_hfapi(monkeypatch, fake_state):
    monkeypatch.setattr(
        train_lora, "HfApi", lambda token=None: FakeHfApi(token=token, state=fake_state)
    )
    return fake_state


class TestHFCheckpointCallback:
    def test_on_save_skips_missing_local_checkpoint(self, tmp_path, patch_hfapi):
        cb = train_lora.HFCheckpointCallback(hf_repo="org/repo", run_id="run1", hf_token="tok")
        args = SimpleNamespace(output_dir=str(tmp_path))
        state = SimpleNamespace(global_step=100)
        cb.on_save(args, state, control=None)
        assert patch_hfapi["uploaded"] == []

    def test_on_save_uploads_checkpoint_and_pointer(self, tmp_path, patch_hfapi):
        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()
        (ckpt_dir / "trainer_state.json").write_text("{}")

        cb = train_lora.HFCheckpointCallback(hf_repo="org/repo", run_id="run1", hf_token="tok")
        args = SimpleNamespace(output_dir=str(tmp_path))
        state = SimpleNamespace(global_step=100)
        cb.on_save(args, state, control="ctrl")

        assert "checkpoints/run1/checkpoint-100/adapter_model.bin" in patch_hfapi["files"]
        pointer_raw = patch_hfapi["content:checkpoints/run1/latest_checkpoint.json"]
        pointer = json.loads(pointer_raw)
        assert pointer["latest_step"] == 100
        assert pointer["path"] == "checkpoints/run1/checkpoint-100"

    def test_prunes_old_checkpoints_beyond_keep_last(self, tmp_path, patch_hfapi):
        # Simulate two prior checkpoints already on HF for this run.
        patch_hfapi["files"] |= {
            "checkpoints/run1/checkpoint-100/adapter_model.bin",
            "checkpoints/run1/checkpoint-200/adapter_model.bin",
        }

        ckpt_dir = tmp_path / "checkpoint-300"
        ckpt_dir.mkdir()
        (ckpt_dir / "trainer_state.json").write_text("{}")

        cb = train_lora.HFCheckpointCallback(
            hf_repo="org/repo", run_id="run1", hf_token="tok", keep_last=2
        )
        args = SimpleNamespace(output_dir=str(tmp_path))
        state = SimpleNamespace(global_step=300)
        cb.on_save(args, state, control=None)

        remaining_steps = {
            f.split("/")[2] for f in patch_hfapi["files"] if f.startswith("checkpoints/run1/checkpoint-")
        }
        assert remaining_steps == {"checkpoint-200", "checkpoint-300"}


class TestFindResumeCheckpoint:
    def test_returns_none_when_no_pointer(self, patch_hfapi):
        assert train_lora.find_resume_checkpoint("org/repo", "run1", "tok") is None

    def test_returns_pointer_when_present(self, monkeypatch, patch_hfapi, tmp_path):
        pointer = {"run_id": "run1", "latest_step": 100, "path": "checkpoints/run1/checkpoint-100"}
        patch_hfapi["files"].add("checkpoints/run1/latest_checkpoint.json")

        local_file = tmp_path / "latest_checkpoint.json"
        local_file.write_text(json.dumps(pointer))
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", lambda **kwargs: str(local_file)
        )

        result = train_lora.find_resume_checkpoint("org/repo", "run1", "tok")
        assert result == pointer


class TestDownloadResumeCheckpoint:
    def test_builds_local_path_from_snapshot(self, monkeypatch, tmp_path):
        pointer = {"path": "checkpoints/run1/checkpoint-100"}
        monkeypatch.setattr(train_lora, "snapshot_download", lambda **kwargs: str(tmp_path))

        result = train_lora.download_resume_checkpoint("org/repo", pointer, "tok", str(tmp_path))
        assert result == str(tmp_path / "checkpoints" / "run1" / "checkpoint-100")


class TestCleanupHfCheckpoints:
    def test_deletes_run_checkpoint_prefix(self, patch_hfapi):
        patch_hfapi["files"] |= {
            "checkpoints/run1/checkpoint-100/adapter_model.bin",
            "checkpoints/run1/latest_checkpoint.json",
            "experiment/random_model/adapter_model.bin",
        }
        train_lora._cleanup_hf_checkpoints("org/repo", "run1", "tok")
        assert patch_hfapi["files"] == {"experiment/random_model/adapter_model.bin"}

    def test_missing_prefix_does_not_raise(self, patch_hfapi):
        train_lora._cleanup_hf_checkpoints("org/repo", "run1", "tok")
