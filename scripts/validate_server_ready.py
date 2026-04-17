#!/usr/bin/env python3
"""Validate local-dryrun and H100 profile parity for no-code-change deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REQUIRED_KEYS = [
    "model_name_or_path",
    "dataset_name",
    "dataset_prompt_column",
    "runtime_profile",
    "method",
    "dry_run",
    "reward_funcs",
    "output_dir",
]

PARITY_KEYS = [
    "dataset_name",
    "dataset_prompt_column",
    "method",
    "reward_funcs",
    "reward_weights",
    "system_prompt",
]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def validate_required_keys(config: dict, path: Path):
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")


def validate_profiles(local_cfg: dict, h100_cfg: dict):
    if local_cfg.get("runtime_profile") != "local-dryrun":
        raise ValueError("local config runtime_profile must be 'local-dryrun'")
    if not local_cfg.get("dry_run", False):
        raise ValueError("local config must set dry_run: true")

    if h100_cfg.get("runtime_profile") != "h100-prod":
        raise ValueError("h100 config runtime_profile must be 'h100-prod'")
    if h100_cfg.get("dry_run", True):
        raise ValueError("h100 config must set dry_run: false")


def validate_parity(local_cfg: dict, h100_cfg: dict):
    mismatches = []
    for key in PARITY_KEYS:
        if local_cfg.get(key) != h100_cfg.get(key):
            mismatches.append(key)

    if mismatches:
        raise ValueError(
            "Profile parity failed. These keys differ but should match for benchmark fairness: "
            f"{mismatches}"
        )


def print_commands(local_cfg_path: Path, h100_cfg_path: Path):
    commands = {
        "local_dryrun": (
            "python -m accelerate.commands.launch --config_file recipes/accelerate_configs/cpu.yaml "
            f"src/open_r1/grpo.py --config {local_cfg_path.as_posix()}"
        ),
        "benchmark_dryrun": (
            "python scripts/run_benchmark_dryrun.py --offline-only "
            "--runtime-profile local-dryrun --method vanilla"
        ),
        "validate_datasets": "python scripts/validate_datasets.py",
        "h100_prod": (
            "ACCELERATE_LOG_LEVEL=info python -m accelerate.commands.launch --config_file recipes/accelerate_configs/zero3.yaml "
            f"src/open_r1/grpo.py --config {h100_cfg_path.as_posix()}"
        ),
    }
    print(json.dumps(commands, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate local and H100 profile readiness"
    )
    parser.add_argument(
        "--local-config",
        default="recipes/Qwen2.5-Math-7B/grpo/config_local_dryrun.yaml",
        help="Path to local dry-run config",
    )
    parser.add_argument(
        "--h100-config",
        default="recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml",
        help="Path to H100 production config",
    )
    args = parser.parse_args()

    local_path = Path(args.local_config)
    h100_path = Path(args.h100_config)

    local_cfg = load_yaml(local_path)
    h100_cfg = load_yaml(h100_path)

    validate_required_keys(local_cfg, local_path)
    validate_required_keys(h100_cfg, h100_path)
    validate_profiles(local_cfg, h100_cfg)
    validate_parity(local_cfg, h100_cfg)

    print("Server readiness profile validation passed.")
    print_commands(local_path, h100_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
