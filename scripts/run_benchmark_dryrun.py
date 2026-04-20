#!/usr/bin/env python3
"""Run deterministic benchmark dry-run to validate pipeline wiring and output contracts."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def _ensure_src_on_path():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark dry-run pipeline")
    parser.add_argument(
        "--datasets",
        default="math500,gsm8k,aime2025,olympiadbench",
        help="Comma-separated dataset list",
    )
    parser.add_argument(
        "--num-questions", type=int, default=16, help="Number of questions per dataset"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Number of generations per question",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--runtime-profile", default="local-dryrun", help="Runtime profile metadata"
    )
    parser.add_argument("--method", default="vanilla", help="Method metadata")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Do not attempt HF downloads, use synthetic samples only",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark-dryrun",
        help="Output directory for JSONL artifacts and summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_src_on_path()

    run_benchmark_dryrun = importlib.import_module(
        "open_r1.benchmarks"
    ).run_benchmark_dryrun

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        raise ValueError("At least one dataset must be specified")

    os.makedirs(args.output_dir, exist_ok=True)
    summary = run_benchmark_dryrun(
        datasets=datasets,
        num_questions=args.num_questions,
        num_generations=args.num_generations,
        output_dir=args.output_dir,
        seed=args.seed,
        runtime_profile=args.runtime_profile,
        method=args.method,
        offline_only=args.offline_only,
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Benchmark dry-run failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
