import logging
from pathlib import Path
from typing import cast

import datasets
from datasets import DatasetDict, concatenate_datasets

from ..configs import DatasetMixtureConfig, ScriptArguments


logger = logging.getLogger(__name__)


def _resolve_dataset_path_candidates(dataset_name: str) -> list[Path]:
    raw_path = Path(dataset_name).expanduser()
    if raw_path.is_absolute():
        return [raw_path.resolve()]

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        (Path.cwd() / raw_path).resolve(),
        (repo_root / raw_path).resolve(),
    ]

    unique_candidates: list[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def _load_local_dataset(dataset_name: str) -> DatasetDict | None:
    """Load a local parquet or prepared dataset directory when available."""
    for dataset_path in _resolve_dataset_path_candidates(dataset_name):
        if dataset_path.is_file() and dataset_path.suffix == ".parquet":
            return datasets.load_dataset(
                "parquet", data_files={"train": str(dataset_path)}
            )

        if not dataset_path.is_dir():
            continue

        train_candidates = [
            dataset_path / "train" / "combined_train.parquet",
            dataset_path / "combined_train.parquet",
        ]
        test_candidates = [
            dataset_path / "test" / "combined_test.parquet",
            dataset_path / "combined_test.parquet",
        ]

        data_files: dict[str, str] = {}
        for train_path in train_candidates:
            if train_path.exists():
                data_files["train"] = str(train_path)
                break

        for test_path in test_candidates:
            if test_path.exists():
                data_files["test"] = str(test_path)
                break

        if data_files:
            return datasets.load_dataset("parquet", data_files=data_files)

        # Fallback for datasets saved with datasets.save_to_disk.
        if (dataset_path / "dataset_info.json").exists() or (
            dataset_path / "state.json"
        ).exists():
            loaded = datasets.load_from_disk(str(dataset_path))
            if isinstance(loaded, DatasetDict):
                return loaded
            return DatasetDict({"train": loaded})

    return None


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        local_dataset = _load_local_dataset(args.dataset_name)
        if local_dataset is not None:
            logger.info(f"Loaded local dataset from path: {args.dataset_name}")
            return local_dataset
        return datasets.load_dataset(args.dataset_name, args.dataset_config)
    elif args.dataset_mixture:
        mixture_config = cast(DatasetMixtureConfig, args.dataset_mixture)
        logger.info(
            f"Creating dataset mixture with {len(mixture_config.datasets)} datasets"
        )
        seed = mixture_config.seed
        datasets_list = []

        for dataset_config in mixture_config.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if mixture_config.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=mixture_config.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {mixture_config.test_split_size}"
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
