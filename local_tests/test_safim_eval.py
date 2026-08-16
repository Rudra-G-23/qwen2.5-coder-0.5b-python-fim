"""
local_tests/test_safim_eval.py
Unit tests for the pilot-comparison aggregation in src/training/safim_eval.py
(load_seed_pass_at_1, aggregate_pilot_results, write_pilot_comparison_report).
Deliberately does NOT touch run_safim_evaluation itself, which loads models
and needs a GPU — these three functions only read/aggregate the CSVs it
writes, so they're testable with fabricated CSVs and no GPU.

Usage:
    pytest local_tests/test_safim_eval.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.safim_eval import (
    aggregate_pilot_results,
    load_seed_pass_at_1,
    render_seed_variance_table,
    write_pilot_comparison_report,
)


def _write_metrics_csv(dir_path: Path, base_pass1: float, lora_pass1: float) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "safim_metrics.csv").write_text(
        f"model,pass@1\nBase,{base_pass1}\nLoRA-FT,{lora_pass1}\n", encoding="utf-8"
    )


class TestLoadSeedPassAt1:
    def test_reads_lora_ft_score(self, tmp_path):
        _write_metrics_csv(tmp_path / "run", base_pass1=0.10, lora_pass1=0.25)
        assert load_seed_pass_at_1(str(tmp_path / "run")) == 0.25

    def test_reads_other_model_label(self, tmp_path):
        _write_metrics_csv(tmp_path / "run", base_pass1=0.10, lora_pass1=0.25)
        assert load_seed_pass_at_1(str(tmp_path / "run"), model_label="Base") == 0.10

    def test_missing_file_returns_none(self, tmp_path):
        assert load_seed_pass_at_1(str(tmp_path / "nonexistent")) is None

    def test_missing_model_label_returns_none(self, tmp_path):
        _write_metrics_csv(tmp_path / "run", base_pass1=0.10, lora_pass1=0.25)
        assert load_seed_pass_at_1(str(tmp_path / "run"), model_label="QLoRA-FT") is None


class TestAggregatePilotResults:
    PILOT_CFG = {
        "seeds": [42, 123, 7],
        "variants": {"random": {}, "planned": {}},
    }

    def test_missing_seeds_reported_not_silently_dropped(self, tmp_path):
        _write_metrics_csv(tmp_path / "random_seed42", 0.1, 0.2)
        # random_seed123, random_seed7, and every planned_seed* left missing.
        report = aggregate_pilot_results(self.PILOT_CFG, results_dir=str(tmp_path))
        assert report["per_variant"]["random"]["missing_seeds"] == [123, 7]
        assert report["per_variant"]["planned"]["missing_seeds"] == [42, 123, 7]
        assert "incomplete" in report["verdict"]

    def test_non_overlapping_ranges_declare_a_leader(self, tmp_path):
        for seed, score in zip(self.PILOT_CFG["seeds"], [0.10, 0.11, 0.12]):
            _write_metrics_csv(tmp_path / f"random_seed{seed}", 0.05, score)
        for seed, score in zip(self.PILOT_CFG["seeds"], [0.30, 0.31, 0.32]):
            _write_metrics_csv(tmp_path / f"planned_seed{seed}", 0.05, score)

        report = aggregate_pilot_results(self.PILOT_CFG, results_dir=str(tmp_path))
        assert report["per_variant"]["random"]["mean"] == (0.10 + 0.11 + 0.12) / 3
        assert report["per_variant"]["planned"]["mean"] == (0.30 + 0.31 + 0.32) / 3
        assert "planned" in report["verdict"]
        assert "not just seed noise" in report["verdict"]

    def test_overlapping_ranges_declared_inconclusive(self, tmp_path):
        for seed, score in zip(self.PILOT_CFG["seeds"], [0.10, 0.20, 0.30]):
            _write_metrics_csv(tmp_path / f"random_seed{seed}", 0.05, score)
        for seed, score in zip(self.PILOT_CFG["seeds"], [0.15, 0.25, 0.35]):
            _write_metrics_csv(tmp_path / f"planned_seed{seed}", 0.05, score)

        report = aggregate_pilot_results(self.PILOT_CFG, results_dir=str(tmp_path))
        assert "overlap" in report["verdict"]
        assert "inconclusive" in report["verdict"]

    def test_std_is_none_for_single_seed(self, tmp_path):
        pilot_cfg = {"seeds": [1], "variants": {"random": {}}}
        _write_metrics_csv(tmp_path / "random_seed1", 0.1, 0.2)
        report = aggregate_pilot_results(pilot_cfg, results_dir=str(tmp_path))
        assert report["per_variant"]["random"]["std"] is None

    def test_std_computed_across_seeds(self, tmp_path):
        for seed, score in zip(self.PILOT_CFG["seeds"], [0.10, 0.20, 0.30]):
            _write_metrics_csv(tmp_path / f"random_seed{seed}", 0.05, score)
        report = aggregate_pilot_results(
            {"seeds": self.PILOT_CFG["seeds"], "variants": {"random": {}}},
            results_dir=str(tmp_path),
        )
        import statistics

        assert report["per_variant"]["random"]["std"] == statistics.stdev([0.10, 0.20, 0.30])


class TestRenderSeedVarianceTable:
    def test_includes_mean_std_and_verdict(self, tmp_path):
        pilot_cfg = {"seeds": [42, 123], "variants": {"random": {}, "planned": {}}}
        for seed, score in zip(pilot_cfg["seeds"], [0.10, 0.20]):
            _write_metrics_csv(tmp_path / f"random_seed{seed}", 0.05, score)
        for seed, score in zip(pilot_cfg["seeds"], [0.30, 0.32]):
            _write_metrics_csv(tmp_path / f"planned_seed{seed}", 0.05, score)

        report = aggregate_pilot_results(pilot_cfg, results_dir=str(tmp_path))
        table = render_seed_variance_table(report)
        assert "| random |" in table
        assert "| planned |" in table
        assert "**Verdict:**" in table
        assert report["verdict"] in table


class TestWritePilotComparisonReport:
    def test_writes_valid_json_matching_aggregate_result(self, tmp_path):
        pilot_cfg = {"seeds": [1], "variants": {"random": {}}}
        _write_metrics_csv(tmp_path / "results" / "random_seed1", 0.1, 0.2)

        out_path = write_pilot_comparison_report(
            pilot_cfg,
            results_dir=str(tmp_path / "results"),
            out_path=str(tmp_path / "reports" / "pilot_comparison.json"),
        )
        assert out_path.exists()
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        # JSON round-trips int dict keys (seeds) as strings — compare
        # everything except that one key-typing artifact.
        assert "not enough variants" in saved["verdict"]
        assert saved["per_variant"]["random"]["mean"] == 0.2
        assert saved["per_variant"]["random"]["scores_by_seed"] == {"1": 0.2}
