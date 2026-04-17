"""Dataset adapters for dry-run benchmark orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable


logger = logging.getLogger(__name__)

SUPPORTED_DRYRUN_DATASETS = ("math_500", "gsm8k", "aime24", "amc23")


@dataclass
class BenchmarkSample:
    question_id: int
    question: str
    answer: str


def _synthetic_samples(dataset_name: str, max_questions: int) -> list[BenchmarkSample]:
    templates = {
        "math_500": (
            "Solve: What is {a} + {b}?",
            lambda i: str(i + (i + 1)),
        ),
        "gsm8k": (
            "A box has {a} apples and gets {b} more. How many apples now?",
            lambda i: str(i + (i + 2)),
        ),
        "aime24": (
            "Compute the remainder when {a}^2 is divided by 10.",
            lambda i: str((i * i) % 10),
        ),
        "amc23": (
            "If x={a} and y={b}, compute x+y.",
            lambda i: str(i + (i + 3)),
        ),
    }
    pattern, answer_fn = templates[dataset_name]

    samples: list[BenchmarkSample] = []
    for idx in range(max_questions):
        a = idx + 1
        b = idx + 2 if dataset_name != "amc23" else idx + 3
        question = pattern.format(a=a, b=b)
        samples.append(
            BenchmarkSample(question_id=idx, question=question, answer=answer_fn(a))
        )

    return samples


def _try_load_hf_dataset(
    dataset_name: str, max_questions: int
) -> list[BenchmarkSample]:
    """Best-effort HF loading for dry-run. Falls back to synthetic on any error."""
    dataset_map = {
        "gsm8k": ("openai/gsm8k", "main", "test", "question", "answer"),
        "amc23": ("TianHongZXY/amc23", None, "test", "problem", "answer"),
    }
    if dataset_name not in dataset_map:
        return []

    hf_name, hf_config, split, question_col, answer_col = dataset_map[dataset_name]

    try:
        import datasets

        ds = datasets.load_dataset(hf_name, hf_config, split=split)
        sample_count = min(max_questions, len(ds))
        ds = ds.select(range(sample_count))
        samples: list[BenchmarkSample] = []
        for idx, row in enumerate(ds):
            samples.append(
                BenchmarkSample(
                    question_id=idx,
                    question=str(row.get(question_col, "")),
                    answer=str(row.get(answer_col, "")),
                )
            )
        return samples
    except Exception as exc:  # pragma: no cover - network and env dependent
        logger.warning(
            "Falling back to synthetic data for dataset '%s' due to HF load error: %s",
            dataset_name,
            exc,
        )
        return []


def load_benchmark_samples(
    dataset_name: str,
    max_questions: int,
    offline_only: bool,
) -> list[BenchmarkSample]:
    if dataset_name not in SUPPORTED_DRYRUN_DATASETS:
        raise ValueError(
            f"Unsupported dry-run dataset '{dataset_name}'. Supported: {SUPPORTED_DRYRUN_DATASETS}"
        )
    if max_questions < 1:
        raise ValueError("max_questions must be >= 1")

    if not offline_only:
        hf_samples = _try_load_hf_dataset(dataset_name, max_questions)
        if hf_samples:
            return hf_samples

    return _synthetic_samples(dataset_name, max_questions)


def iter_prompts(samples: Iterable[BenchmarkSample]) -> Iterable[tuple[int, str, str]]:
    for sample in samples:
        yield sample.question_id, sample.question, sample.answer
