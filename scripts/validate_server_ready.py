#!/usr/bin/env python3
"""Validate local-dryrun and H100 profile parity for no-code-change deployment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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

SOLUTION_REQUIRED_REWARD_FUNCS = {
    "accuracy",
    "cosine",
    "length",
}


def resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / path).resolve()


def infer_local_config_path(h100_path: Path) -> Path | None:
    if "_h100_prod" not in h100_path.name:
        return None
    candidate = h100_path.with_name(h100_path.name.replace("_h100_prod", "_local_dryrun"))
    return candidate if candidate.exists() else None


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


def validate_accelerate_config(accelerate_config: Path):
    if not accelerate_config.exists():
        raise FileNotFoundError(f"Missing accelerate config: {accelerate_config}")


def validate_python_dependencies(h100_cfg: dict) -> dict:
    required_modules = [
        "torch",
        "accelerate",
        "transformers",
        "datasets",
        "trl",
        "yaml",
    ]

    if h100_cfg.get("use_vllm", False):
        required_modules.append("vllm")
    if h100_cfg.get("attn_implementation") == "flash_attention_2":
        required_modules.append("flash_attn")

    missing = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    if missing:
        raise RuntimeError(f"Missing Python dependencies for H100 profile: {missing}")

    return {"required_modules": required_modules, "missing_modules": []}


def validate_gpu_environment(allow_non_h100: bool) -> dict:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is not installed; cannot verify CUDA device state")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; H100 training cannot start")

    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices detected")

    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    if not allow_non_h100 and not any("H100" in name for name in device_names):
        raise RuntimeError(
            "CUDA devices detected but no H100 found. "
            f"Detected devices: {device_names}"
        )

    return {
        "device_count": device_count,
        "device_names": device_names,
    }


def validate_dataset_access(
    cfg: dict,
    repo_root: Path,
    strict_dataset: bool,
) -> dict:
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from open_r1.utils.data import get_dataset

    dataset_name = cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("dataset_name is required for dataset validation")

    args = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_config=cfg.get("dataset_config"),
        dataset_mixture=None,
    )

    dataset = get_dataset(args)
    train_split = cfg.get("dataset_train_split", "train")
    test_split = cfg.get("dataset_test_split", "test")
    prompt_column = cfg.get("dataset_prompt_column", "prompt")

    if train_split not in dataset:
        raise RuntimeError(
            f"Train split '{train_split}' not found. Available splits: {list(dataset.keys())}"
        )

    if strict_dataset and test_split not in dataset:
        raise RuntimeError(
            f"Test split '{test_split}' not found. Available splits: {list(dataset.keys())}"
        )

    train_columns = set(dataset[train_split].column_names)
    if prompt_column not in train_columns:
        raise RuntimeError(
            f"Prompt column '{prompt_column}' missing from train split columns: {sorted(train_columns)}"
        )

    reward_funcs = cfg.get("reward_funcs", []) or []
    if any(func in SOLUTION_REQUIRED_REWARD_FUNCS for func in reward_funcs):
        if "solution" not in train_columns:
            raise RuntimeError(
                "Reward functions require 'solution' column, but it is missing from train split"
            )

    test_rows = len(dataset[test_split]) if test_split in dataset else None
    if len(dataset[train_split]) < 1:
        raise RuntimeError("Train split is empty")
    if strict_dataset and (test_rows is None or test_rows < 1):
        raise RuntimeError("Test split is empty or missing under strict dataset validation")

    return {
        "dataset_name": dataset_name,
        "available_splits": list(dataset.keys()),
        "train_split": train_split,
        "test_split": test_split if test_split in dataset else None,
        "train_rows": len(dataset[train_split]),
        "test_rows": test_rows,
        "prompt_column": prompt_column,
    }


def print_commands(local_cfg_path: Path, h100_cfg_path: Path):
    commands = {
        "local_dryrun": (
            "python -m accelerate.commands.launch --config_file recipes/accelerate_configs/cpu.yaml "
            f"src/open_r1/grpo.py --config {local_cfg_path.as_posix()}"
        ),
        "benchmark_dryrun": (
            "python scripts/run_benchmark_dryrun.py "
            "--datasets math500,gsm8k,aime2025,olympiadbench "
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
        default=None,
        help="Path to local dry-run config",
    )
    parser.add_argument(
        "--h100-config",
        default="recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml",
        help="Path to H100 production config",
    )
    parser.add_argument(
        "--accelerate-config",
        default="recipes/accelerate_configs/zero3.yaml",
        help="Path to accelerate config used for H100 training",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip local-vs-H100 parity checks",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Validate required Python dependencies",
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Validate CUDA/H100 availability",
    )
    parser.add_argument(
        "--allow-non-h100",
        action="store_true",
        help="Allow non-H100 CUDA devices when --check-gpu is enabled",
    )
    parser.add_argument(
        "--strict-dataset",
        action="store_true",
        help="Require both train/test splits and non-empty rows",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print detailed validation report as JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    h100_path = resolve_path(args.h100_config, repo_root)
    accelerate_path = resolve_path(args.accelerate_config, repo_root)

    local_path: Path | None = None
    if args.local_config:
        local_path = resolve_path(args.local_config, repo_root)
    elif not args.skip_parity:
        local_path = infer_local_config_path(h100_path)

    report: dict[str, object] = {
        "h100_config": h100_path.as_posix(),
        "accelerate_config": accelerate_path.as_posix(),
    }

    h100_cfg = load_yaml(h100_path)
    validate_required_keys(h100_cfg, h100_path)
    validate_profiles(
        {"runtime_profile": "local-dryrun", "dry_run": True},
        h100_cfg,
    )

    if local_path is not None and not args.skip_parity:
        local_cfg = load_yaml(local_path)
        validate_required_keys(local_cfg, local_path)
        validate_profiles(local_cfg, h100_cfg)
        validate_parity(local_cfg, h100_cfg)
        report["local_config"] = local_path.as_posix()
        report["parity"] = "passed"
    elif args.skip_parity:
        report["parity"] = "skipped"
    else:
        report["parity"] = "not_checked_no_local_config"

    validate_accelerate_config(accelerate_path)
    report["dataset"] = validate_dataset_access(
        h100_cfg,
        repo_root=repo_root,
        strict_dataset=args.strict_dataset,
    )

    if args.check_deps:
        report["dependencies"] = validate_python_dependencies(h100_cfg)
    else:
        report["dependencies"] = "skipped"

    if args.check_gpu:
        report["gpu"] = validate_gpu_environment(allow_non_h100=args.allow_non_h100)
    else:
        report["gpu"] = "skipped"

    print("Server readiness profile validation passed.")
    if local_path is not None:
        print_commands(local_path, h100_path)
    if args.print_json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
