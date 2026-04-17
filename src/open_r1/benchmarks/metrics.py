"""Metrics helpers for benchmark dry-run."""

from __future__ import annotations

from collections import defaultdict


def unbiased_pass_at_k(records: list[dict], k: int, n: int) -> float:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        qid = int(record["question_id"])
        if len(grouped[qid]) < n:
            grouped[qid].append(record)

    total_prob = 0.0
    total_questions = 0

    for examples in grouped.values():
        if len(examples) < n:
            continue
        correct_count = sum(1 for item in examples[:n] if bool(item["label"]))

        if n - correct_count < k:
            prob = 1.0
        else:
            prod = 1.0
            for i in range(n - correct_count + 1, n + 1):
                prod *= 1.0 - (k / i)
            prob = 1.0 - prod

        total_prob += prob
        total_questions += 1

    if total_questions == 0:
        return 0.0
    return total_prob / total_questions


def compute_pass_metrics(
    records: list[dict], n: int, ks: list[int]
) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in ks:
        if k > n:
            continue
        output[f"pass@{k}"] = unbiased_pass_at_k(records, k=k, n=n)
    return output
