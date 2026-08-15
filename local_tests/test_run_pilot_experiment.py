"""
local_tests/test_run_pilot_experiment.py
Unit tests for scripts/run_pilot_experiment.py's `_find_existing_run_commit` —
the HF-commit-history lookup that lets a restarted Kaggle session skip
re-training a (variant, seed) already pushed in a prior session. All
huggingface_hub calls are monkeypatched; no real network access.

Usage:
    pytest local_tests/test_run_pilot_experiment.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.run_pilot_experiment as rpe


class FakeCommit(SimpleNamespace):
    pass


class FakeHfApi:
    def __init__(self, commits: list[FakeCommit], raise_on_list: bool = False) -> None:
        self._commits = commits
        self._raise_on_list = raise_on_list

    def list_repo_commits(self, repo_id: str, repo_type: str = "model") -> list[FakeCommit]:
        if self._raise_on_list:
            raise RuntimeError("repo not found")
        return self._commits


def test_finds_matching_commit(monkeypatch):
    commits = [
        FakeCommit(title="qwen05b-lora-r16-e1-dsrandom-20260815-a42: push adapter", commit_id="sha-a"),
        FakeCommit(title="qwen05b-lora-r16-e1-dsrandom-20260815-a123: push adapter", commit_id="sha-b"),
    ]
    monkeypatch.setattr(rpe, "HfApi", lambda token=None: FakeHfApi(commits))

    result = rpe._find_existing_run_commit(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "qwen05b-lora-r16-e1-dsrandom-20260815-a42",
        token=None,
    )
    assert result == "sha-a"


def test_no_matching_commit_returns_none(monkeypatch):
    commits = [
        FakeCommit(title="qwen05b-lora-r16-e1-dsrandom-20260815-a123: push adapter", commit_id="sha-b"),
    ]
    monkeypatch.setattr(rpe, "HfApi", lambda token=None: FakeHfApi(commits))

    result = rpe._find_existing_run_commit(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "qwen05b-lora-r16-e1-dsrandom-20260815-a42",
        token=None,
    )
    assert result is None


def test_empty_history_returns_none(monkeypatch):
    monkeypatch.setattr(rpe, "HfApi", lambda token=None: FakeHfApi([]))

    result = rpe._find_existing_run_commit(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "qwen05b-lora-r16-e1-dsrandom-20260815-a42",
        token=None,
    )
    assert result is None


def test_repo_lookup_failure_falls_back_to_none(monkeypatch):
    """A brand-new repo (first pilot run ever) or a transient HF hiccup must
    fall through to a normal training run, not crash the sweep."""
    monkeypatch.setattr(rpe, "HfApi", lambda token=None: FakeHfApi([], raise_on_list=True))

    result = rpe._find_existing_run_commit(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "qwen05b-lora-r16-e1-dsrandom-20260815-a42",
        token=None,
    )
    assert result is None


def test_prefix_match_does_not_false_positive_on_different_seed(monkeypatch):
    """run_id encodes the seed via the `a{attempt}` suffix — a commit for
    seed 123 must not match a lookup for seed 42, even though both share the
    same model/rank/epoch/dataset-version prefix."""
    commits = [
        FakeCommit(title="qwen05b-lora-r16-e1-dsrandom-20260815-a423: push adapter", commit_id="sha-a"),
    ]
    monkeypatch.setattr(rpe, "HfApi", lambda token=None: FakeHfApi(commits))

    result = rpe._find_existing_run_commit(
        "Rudra-G-23/qwen2.5-coder-0.5b-python-fim",
        "qwen05b-lora-r16-e1-dsrandom-20260815-a42",
        token=None,
    )
    assert result is None
