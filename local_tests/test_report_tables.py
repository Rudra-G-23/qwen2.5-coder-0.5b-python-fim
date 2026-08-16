"""
local_tests/test_report_tables.py
Unit tests for src/training/report_tables.py's pure markdown-table
generators (render_hyperparameter_table, render_compute_cost_table).

Usage:
    pytest local_tests/test_report_tables.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.report_tables import (
    render_compute_cost_table,
    render_hyperparameter_table,
    write_compute_cost_table,
    write_hyperparameter_table,
)


class TestRenderHyperparameterTable:
    CFG = {
        "model": {"name": "Qwen/Qwen2.5-Coder-0.5B", "max_seq_length": 2048},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "training": {
            "num_epochs": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2.0e-4,
            "lr_scheduler_type": "cosine",
            "warmup_steps": 50,
            "optim": "adamw_torch",
            "seed": 42,
            "fp16": True,
        },
    }

    def test_includes_key_hyperparameters(self):
        table = render_hyperparameter_table(self.CFG)
        assert "Qwen/Qwen2.5-Coder-0.5B" in table
        assert "16" in table  # r
        assert "q_proj, v_proj" in table
        assert "cosine" in table

    def test_omits_missing_fields(self):
        table = render_hyperparameter_table({"model": {"name": "x"}})
        assert "LoRA rank" not in table
        assert "x" in table

    def test_write_creates_file(self, tmp_path):
        out = tmp_path / "table.md"
        write_hyperparameter_table(self.CFG, out_path=str(out))
        assert out.exists()
        assert "Qwen" in out.read_text()


class TestRenderComputeCostTable:
    def test_computes_cost_and_totals(self):
        runs = [
            {"run_id": "random-seed42", "gpu_type": "T4", "duration_hours": 2.0, "hourly_rate_usd": 0.5},
            {"run_id": "planned-seed42", "gpu_type": "T4", "duration_hours": 1.5, "hourly_rate_usd": 0.5},
        ]
        table = render_compute_cost_table(runs)
        assert "random-seed42" in table
        assert "1.00" in table  # 2.0 * 0.5
        assert "0.75" in table  # 1.5 * 0.5
        assert "3.50" in table  # total hours
        assert "1.75" in table  # total cost

    def test_defaults_free_rate_to_zero(self):
        runs = [{"run_id": "kaggle-run", "gpu_type": "T4", "duration_hours": 3.0}]
        table = render_compute_cost_table(runs)
        assert "0.00" in table

    def test_write_creates_file(self, tmp_path):
        out = tmp_path / "cost.md"
        write_compute_cost_table(
            [{"run_id": "r1", "gpu_type": "T4", "duration_hours": 1.0}], out_path=str(out)
        )
        assert out.exists()
