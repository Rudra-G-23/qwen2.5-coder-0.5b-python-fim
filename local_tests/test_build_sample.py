"""
local_tests/test_build_sample.py
Unit tests for scripts/build_sample.py's `_resume_state` — specifically the
cross-prefix stream-continuation behavior added for
configs/data/stack_v3_eval_pool_no_fim.yaml's continue_from_path_prefix
(seed the dedup set from another prefix AND start streaming at its highest
checkpoint position, so the new pool is a strict continuation of that
prefix's stream). All src.curation.checkpoint calls are monkeypatched; no
real network access.

Usage:
    pytest local_tests/test_build_sample.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.build_sample as bs


def _checkpoints_by_prefix(monkeypatch, mapping):
    """Stub ckpt.load_all_checkpoints so it returns a different checkpoint
    list per `path_prefix` kwarg — _resume_state calls it once for the pool's
    own prefix and (when continue_from is set) once for the source prefix."""
    def fake_load_all_checkpoints(hf_repo, token=None, repo_type="dataset", path_prefix=""):
        return mapping.get(path_prefix, [])

    monkeypatch.setattr(bs.ckpt, "load_all_checkpoints", fake_load_all_checkpoints)
    monkeypatch.setattr(bs.ckpt, "pick_resume_checkpoint", _real_pick_resume)


def _real_pick_resume(checkpoints):
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda c: (c["shard_index"], c["row_offset"]))


def test_continues_from_source_prefix_on_first_ever_session(monkeypatch):
    """No checkpoint history yet for this prefix + continue_from given ->
    dedup contains exactly the seeded hashes AND streaming starts at the
    source prefix's highest checkpoint (shard_index + 1, row_offset)."""
    source_ckpt = {
        "shard_index": 9,
        "row_offset": 13500,
        "cumulative_files_collected": 10000,
    }
    _checkpoints_by_prefix(
        monkeypatch,
        {"new_prefix": [], "sample_filtered_data_10000": [source_ckpt]},
    )
    monkeypatch.setattr(
        bs.ckpt, "seed_dedup_from_prefix",
        lambda hf_repo, source_path_prefix, token=None: {"seed_a", "seed_b"},
    )

    shard_index, row_offset, cumulative, dedup, latest_key = bs._resume_state(
        "fake/repo", None, "new_prefix", continue_from="sample_filtered_data_10000"
    )

    assert shard_index == 10          # 9 + 1
    assert row_offset == 13500
    assert cumulative == 0            # this pool counts from 0, not the pilot's 10k
    assert latest_key is None
    assert dedup.seen_hashes == frozenset({"seed_a", "seed_b"})


def test_no_continuation_without_continue_from(monkeypatch):
    """First-ever session but no continue_from configured -> dedup starts
    empty and streaming starts at (0, 0), matching the pre-existing behavior."""
    _checkpoints_by_prefix(monkeypatch, {"new_prefix": []})

    def _unexpected_seed_call(*a, **k):
        raise AssertionError("seed_dedup_from_prefix must not be called when continue_from is None")

    monkeypatch.setattr(bs.ckpt, "seed_dedup_from_prefix", _unexpected_seed_call)

    shard_index, row_offset, _, dedup, latest_key = bs._resume_state(
        "fake/repo", None, "new_prefix", continue_from=None
    )
    assert (shard_index, row_offset) == (0, 0)
    assert latest_key is None
    assert dedup.seen_hashes == frozenset()


def test_continue_from_ignored_once_prefix_has_own_history(monkeypatch):
    """This prefix already has checkpoint history -> continue_from is
    ignored: seed_dedup_from_prefix must NOT be called, and resume comes
    from this prefix's own highest checkpoint. Its own load_seen_hashes
    cache (already including any seeded hashes) is authoritative from the
    second session onward."""
    own_ckpt = {"shard_index": 12, "row_offset": 18000, "cumulative_files_collected": 1500}
    _checkpoints_by_prefix(
        monkeypatch,
        {"new_prefix": [own_ckpt], "sample_filtered_data_10000": [{"shard_index": 9, "row_offset": 13500}]},
    )
    monkeypatch.setattr(bs.ckpt, "load_seen_hashes", lambda *a, **k: {"hash_x"})

    def _unexpected_seed_call(*a, **k):
        raise AssertionError("seed_dedup_from_prefix must not be called on a resumed session")

    monkeypatch.setattr(bs.ckpt, "seed_dedup_from_prefix", _unexpected_seed_call)

    shard_index, row_offset, cumulative, dedup, latest_key = bs._resume_state(
        "fake/repo", None, "new_prefix", continue_from="sample_filtered_data_10000"
    )

    assert shard_index == 13          # 12 + 1, from this prefix's own history
    assert row_offset == 18000
    assert cumulative == 1500
    assert latest_key == (12, 18000)
    assert dedup.seen_hashes == frozenset({"hash_x"})
