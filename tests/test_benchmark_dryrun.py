# Copyright 2025 The HuggingFace Team. All rights reserved.

import json
import tempfile
import unittest
from pathlib import Path

from open_r1.benchmarks.dryrun import run_benchmark_dryrun


class TestBenchmarkDryRun(unittest.TestCase):
    def test_run_benchmark_dryrun(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_benchmark_dryrun(
                datasets=["gsm8k", "amc23"],
                num_questions=4,
                num_generations=4,
                output_dir=tmp_dir,
                seed=42,
                runtime_profile="local-dryrun",
                method="vanilla",
                offline_only=True,
            )

            self.assertIn("datasets", summary)
            self.assertIn("gsm8k", summary["datasets"])
            self.assertIn("amc23", summary["datasets"])

            summary_path = Path(summary["summary_path"])
            self.assertTrue(summary_path.exists())

            data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(data["runtime_profile"], "local-dryrun")

    def test_run_benchmark_dryrun_amsb_method(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_benchmark_dryrun(
                datasets=["math500"],
                num_questions=3,
                num_generations=2,
                output_dir=tmp_dir,
                seed=7,
                runtime_profile="local-dryrun",
                method="amsb",
                offline_only=True,
            )

            summary_path = Path(summary["summary_path"])
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(data["method"], "amsb")


if __name__ == "__main__":
    unittest.main()
