"""
local_tests/test_run_pilot_experiment.py
Unit tests for scripts/run_pilot_experiment.py's `_subfolder_already_pushed` —
the HF-file-existence check that lets a restarted Kaggle session skip
re-training a (variant, seed) already pushed in a prior session. All
huggingface_hub calls are monkeypatched; no real network access.

Usage:
    pytest local_tests/test_run_pilot_experiment.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.run_pilot_experiment as rpe


class FakeHfApi:
    def __init__(self, existing_files: set[str], raise_on_check: bool = False) -> None:
        self._existing_files = existing_files
        self._raise_on_check = raise_on_check

    def file_exists(self, repo_id: str, filename: str, repo_type: str = "model") -> bool:
        if self._raise_on_check:
            raise RuntimeError("repo not found")
        return filename in self._existing_files


def test_finds_existing_subfolder(monkeypatch):
    monkeypatch.setattr(
        rpe, "HfApi",
        lambda token=None: FakeHfApi({"experiment/random_model/seed42/adapter_model.safetensors"}),
    )

    result = rpe._subfolder_already_pushed(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "experiment/random_model/seed42",
        token=None,
    )
    assert result is True


def test_missing_subfolder_returns_false(monkeypatch):
    monkeypatch.setattr(
        rpe, "HfApi",
        lambda token=None: FakeHfApi({"experiment/random_model/seed123/adapter_model.safetensors"}),
    )

    result = rpe._subfolder_already_pushed(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "experiment/random_model/seed42",
        token=None,
    )
    assert result is False


def test_empty_repo_returns_false(monkeypatch):
    monkeypatch.setattr(rpe, "HfApi", lambda token=None: FakeHfApi(set()))

    result = rpe._subfolder_already_pushed(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "experiment/random_model/seed42",
        token=None,
    )
    assert result is False


def test_repo_lookup_failure_falls_back_to_false(monkeypatch):
    """A brand-new repo (first pilot run ever) or a transient HF hiccup must
    fall through to a normal training run, not crash the sweep."""
    monkeypatch.setattr(rpe, "HfApi", lambda token=None: FakeHfApi(set(), raise_on_check=True))

    result = rpe._subfolder_already_pushed(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "experiment/random_model/seed42",
        token=None,
    )
    assert result is False


def test_does_not_false_positive_on_different_seed(monkeypatch):
    """Seed 42's check must not be satisfied by seed 423's file existing —
    subfolder names are exact paths, not prefixes, so no accidental overlap
    is possible the way commit-title-prefix matching used to risk."""
    monkeypatch.setattr(
        rpe, "HfApi",
        lambda token=None: FakeHfApi({"experiment/random_model/seed423/adapter_model.safetensors"}),
    )

    result = rpe._subfolder_already_pushed(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "experiment/random_model/seed42",
        token=None,
    )
    assert result is False
