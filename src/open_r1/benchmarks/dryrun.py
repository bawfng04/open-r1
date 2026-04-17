"""Dry-run benchmark orchestration with deterministic mock generations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .adapters import iter_prompts, load_benchmark_samples
from .grading import extract_predicted_answer, grade_prediction
from .metrics import compute_pass_metrics


def _mock_answer(question: str, gold_answer: str, seed: int, generation_id: int) -> str:
    digest = hashlib.sha256(
        f"{question}|{seed}|{generation_id}".encode("utf-8")
    ).hexdigest()
    selector = int(digest[:2], 16) / 255.0

    # Keep deterministic but non-trivial accuracy for dry-run statistics.
    if selector < 0.6:
        answer = gold_answer
    elif selector < 0.8:
        try:
            answer = str(float(gold_answer) + 1.0)
        except Exception:
            answer = f"{gold_answer}_wrong"
    else:
        answer = "0"

    return (
        "<think>\n"
        "Mock dry-run reasoning path.\n"
        "</think>\n"
        "<answer>\n"
        f"{answer}\n"
        "</answer>"
    )


def run_benchmark_dryrun(
    datasets: list[str],
    num_questions: int,
    num_generations: int,
    output_dir: str,
    seed: int,
    runtime_profile: str,
    method: str,
    offline_only: bool,
) -> dict:
    if num_generations < 1:
        raise ValueError("num_generations must be >= 1")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary = {
        "runtime_profile": runtime_profile,
        "method": method,
        "num_generations": num_generations,
        "num_questions": num_questions,
        "datasets": {},
    }

    for dataset_name in datasets:
        samples = load_benchmark_samples(
            dataset_name, num_questions, offline_only=offline_only
        )

        records: list[dict] = []
        for question_id, question, gold_answer in iter_prompts(samples):
            for generation_id in range(num_generations):
                response = _mock_answer(question, gold_answer, seed, generation_id)
                pred_answer = extract_predicted_answer(response)
                label = grade_prediction(pred_answer, gold_answer, dataset_name)
                records.append(
                    {
                        "dataset": dataset_name,
                        "question_id": question_id,
                        "generation_id": generation_id,
                        "prompt": question,
                        "question": question,
                        "answer": gold_answer,
                        "gold_answer": gold_answer,
                        "response": response,
                        "pred_answer": pred_answer,
                        "label": bool(label),
                        "runtime_profile": runtime_profile,
                        "method": method,
                    }
                )

        dataset_path = output_root / f"{dataset_name}.jsonl"
        with dataset_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        metrics = compute_pass_metrics(records, n=num_generations, ks=[1, 2, 4, 8, 16])
        dataset_summary = {
            "records": len(records),
            "questions": len(samples),
            "metrics": metrics,
            "jsonl": str(dataset_path.as_posix()),
        }
        all_summary["datasets"][dataset_name] = dataset_summary

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    all_summary["summary_path"] = str(summary_path.as_posix())
    return all_summary
