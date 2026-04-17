#!/usr/bin/env python3
"""Validate dataset manifest and lock consistency and optional local parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ValidationResult:
    dataset_id: str
    split: str
    status: str
    details: str = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return table.num_rows
    except Exception as exc:
        raise RuntimeError(f"Failed reading parquet '{path}': {exc}") from exc


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return data


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Lock must be a mapping: {path}")
    return data


def validate_manifest_lock(manifest: dict, lock: dict) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    manifest_ids = [item["id"] for item in manifest.get("datasets", [])]
    locked_datasets = lock.get("locked_datasets", {})

    for dataset_id in manifest_ids:
        if dataset_id not in locked_datasets:
            results.append(
                ValidationResult(
                    dataset_id, "-", "MISSING_LOCK", "Dataset not found in lock file"
                )
            )
            continue

        manifest_entry = next(
            item for item in manifest["datasets"] if item["id"] == dataset_id
        )
        split_specs = manifest_entry.get("splits", [])
        locked_splits = locked_datasets[dataset_id].get("splits", {})

        for split_spec in split_specs:
            split_name = split_spec["name"]
            if split_name not in locked_splits:
                results.append(
                    ValidationResult(
                        dataset_id,
                        split_name,
                        "MISSING_LOCK_SPLIT",
                        "Split not found in lock file",
                    )
                )
                continue

            expected_rows = split_spec.get("size")
            locked_rows = locked_splits[split_name].get("row_count")
            if expected_rows != locked_rows:
                results.append(
                    ValidationResult(
                        dataset_id,
                        split_name,
                        "ROW_COUNT_MISMATCH",
                        f"manifest={expected_rows} lock={locked_rows}",
                    )
                )
            else:
                results.append(ValidationResult(dataset_id, split_name, "LOCK_OK"))

    return results


def validate_local_files(lock: dict, repo_root: Path) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    for dataset_id, dataset in lock.get("locked_datasets", {}).items():
        for split_name, split_info in dataset.get("splits", {}).items():
            local_path_value = split_info.get("local_path")
            if not local_path_value:
                results.append(
                    ValidationResult(
                        dataset_id, split_name, "SKIPPED", "No local_path configured"
                    )
                )
                continue

            parquet_path = (repo_root / local_path_value).resolve()
            if not parquet_path.exists():
                results.append(
                    ValidationResult(
                        dataset_id, split_name, "MISSING_FILE", str(parquet_path)
                    )
                )
                continue

            try:
                row_count = _read_parquet_row_count(parquet_path)
            except Exception as exc:
                results.append(
                    ValidationResult(dataset_id, split_name, "PARQUET_ERROR", str(exc))
                )
                continue

            expected_rows = split_info.get("row_count")
            if expected_rows is not None and row_count != expected_rows:
                results.append(
                    ValidationResult(
                        dataset_id,
                        split_name,
                        "ROW_COUNT_MISMATCH",
                        f"expected={expected_rows} actual={row_count}",
                    )
                )
                continue

            expected_hash = split_info.get("sha256")
            actual_hash = _sha256_file(parquet_path)
            if expected_hash and expected_hash != actual_hash:
                results.append(
                    ValidationResult(
                        dataset_id,
                        split_name,
                        "HASH_MISMATCH",
                        f"expected={expected_hash} actual={actual_hash}",
                    )
                )
            else:
                status = "FILE_OK" if expected_hash else "FILE_OK_NO_HASH"
                results.append(
                    ValidationResult(dataset_id, split_name, status, actual_hash)
                )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset manifest and lock")
    parser.add_argument(
        "--manifest",
        default="datasets/datasets.manifest.yaml",
        help="Path to dataset manifest YAML",
    )
    parser.add_argument(
        "--lock",
        default="datasets/datasets.lock.json",
        help="Path to dataset lock JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing local files and skipped entries",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    lock_path = (repo_root / args.lock).resolve()

    if not manifest_path.exists():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        return 1
    if not lock_path.exists():
        print(f"Missing lock file: {lock_path}", file=sys.stderr)
        return 1

    manifest = _load_yaml(manifest_path)
    lock = _load_json(lock_path)

    schema_results = validate_manifest_lock(manifest, lock)
    file_results = validate_local_files(lock, repo_root)
    all_results = schema_results + file_results

    failures = []
    for result in all_results:
        line = f"[{result.status}] {result.dataset_id}/{result.split}"
        if result.details:
            line += f" :: {result.details}"
        print(line)

        if result.status in {
            "MISSING_LOCK",
            "MISSING_LOCK_SPLIT",
            "ROW_COUNT_MISMATCH",
            "HASH_MISMATCH",
            "PARQUET_ERROR",
        }:
            failures.append(result)
        if args.strict and result.status in {"MISSING_FILE", "SKIPPED"}:
            failures.append(result)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
