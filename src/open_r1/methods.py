# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Method helpers for MGRPO and SEED-style workflows."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_final_answer(text: str) -> str:
    """Extract a stable answer surface form from model output."""
    if not text:
        return ""

    tag_matches = ANSWER_TAG_PATTERN.findall(text)
    if tag_matches:
        return tag_matches[-1].strip()

    boxed_matches = BOXED_PATTERN.findall(text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    number_matches = NUMBER_PATTERN.findall(text)
    if number_matches:
        return number_matches[-1].strip()

    return text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer strings for semantic clustering without heavy dependencies."""
    normalized = answer.strip().lower()
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\s+", "", normalized)
    if normalized.endswith("."):
        normalized = normalized[:-1]
    return normalized


def cluster_answers(answers: Sequence[str]) -> dict[str, list[int]]:
    """Cluster answers by normalized answer string."""
    clusters: dict[str, list[int]] = defaultdict(list)
    for idx, answer in enumerate(answers):
        key = normalize_answer(extract_final_answer(answer))
        clusters[key].append(idx)
    return dict(clusters)


def compute_semantic_entropy(
    answers: Sequence[str],
    normalize_entropy: bool = True,
) -> float:
    """Compute semantic entropy over normalized answer clusters."""
    if not answers:
        return 0.0

    clusters = cluster_answers(answers)
    total = sum(len(indices) for indices in clusters.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for indices in clusters.values():
        p = len(indices) / total
        entropy -= p * math.log(max(p, 1e-12))

    if normalize_entropy and len(clusters) > 1:
        entropy /= math.log(len(clusters))

    return float(entropy)


def modulation_factor(entropy: float, alpha: float, mode: str) -> float:
    """Return SEED modulation factor from semantic entropy."""
    clamped_entropy = max(0.0, entropy)
    if mode == "linear":
        factor = 1.0 + alpha * clamped_entropy
    elif mode == "exp":
        factor = math.exp(alpha * clamped_entropy)
    elif mode == "focal":
        factor = 1.0 + alpha * (clamped_entropy**2)
    else:
        raise ValueError(f"Unknown entropy modulation mode: {mode}")

    return float(max(factor, 1e-6))


def modulate_advantages(
    advantages: Iterable[float],
    entropy: float,
    alpha: float,
    mode: str,
) -> list[float]:
    """Scale advantages by a semantic-entropy modulation factor."""
    factor = modulation_factor(entropy, alpha, mode)
    return [float(adv) * factor for adv in advantages]


def build_mgrpo_layer2_prompt(
    original_prompt: str,
    layer1_completion: str,
    guiding_phrase: str,
) -> str:
    """Build the MGRPO layer-2 correction prompt."""
    return (
        f"{original_prompt}\n\n"
        "First attempt:\n"
        f"{layer1_completion}\n\n"
        f"{guiding_phrase}\n"
        "Return your revised reasoning and final answer in <think>/<answer> format."
    )


def compute_mgrpo_transition_metrics(
    turn1_correct: Sequence[bool],
    turn2_correct: Sequence[bool],
) -> dict[str, float]:
    """Compute MGRPO transition metrics from turn-1 and turn-2 correctness."""
    if len(turn1_correct) != len(turn2_correct):
        raise ValueError("turn1_correct and turn2_correct must have the same length")

    total = len(turn1_correct)
    if total == 0:
        return {
            "acc_t1": 0.0,
            "acc_t2": 0.0,
            "delta_i_to_c": 0.0,
            "delta_c_to_i": 0.0,
        }

    t1_hits = sum(1 for x in turn1_correct if x)
    t2_hits = sum(1 for x in turn2_correct if x)
    i_to_c = sum(1 for a, b in zip(turn1_correct, turn2_correct) if (not a) and b)
    c_to_i = sum(1 for a, b in zip(turn1_correct, turn2_correct) if a and (not b))

    return {
        "acc_t1": t1_hits / total,
        "acc_t2": t2_hits / total,
        "delta_i_to_c": i_to_c / total,
        "delta_c_to_i": c_to_i / total,
    }


def select_top_error_clusters(
    error_answers: Sequence[str],
    max_clusters: int,
) -> list[list[int]]:
    """Return indices for the largest semantic error clusters."""
    if max_clusters < 1:
        raise ValueError("max_clusters must be >= 1")

    clusters = cluster_answers(error_answers)
    ranked = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    return [indices for _, indices in ranked[:max_clusters]]


def sample_balanced_indices(
    correct_indices: Sequence[int],
    top_error_clusters: Sequence[Sequence[int]],
    group_size: int,
    seed: int,
) -> dict[str, list[int]]:
    """Sample a static 50/50 correct-vs-error index group with replacement."""
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    if group_size % 2 != 0:
        raise ValueError("group_size must be even for 50/50 balancing")

    rng = random.Random(seed)
    correct_pool = list(correct_indices)
    error_pool = [idx for cluster in top_error_clusters for idx in cluster]

    if not correct_pool:
        correct_pool = error_pool[:] if error_pool else [0]
    if not error_pool:
        error_pool = correct_pool[:]

    half = group_size // 2
    selected_correct = rng.choices(correct_pool, k=half)
    selected_error = rng.choices(error_pool, k=half)
    return {
        "correct": selected_correct,
        "error": selected_error,
        "all": selected_correct + selected_error,
    }


def entropy_scale_factor(entropy: float, mode: str, temperature: float) -> float:
    """Convert entropy to a continuous loss scale used by A-MSB-GRPO."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    normalized_entropy = max(entropy, 0.0) / temperature
    if mode == "exp_decay":
        scale = math.exp(-normalized_entropy)
    elif mode == "inverse":
        scale = 1.0 / (1.0 + normalized_entropy)
    elif mode == "linear":
        scale = max(0.0, 1.0 - normalized_entropy)
    else:
        raise ValueError(f"Unknown entropy scale mode: {mode}")

    return float(max(scale, 1e-6))


def build_amsb_plan(
    candidate_completions: Sequence[str],
    candidate_correctness: Sequence[bool] | None,
    balanced_group_size: int,
    max_error_clusters: int,
    entropy_scale_mode: str,
    entropy_temperature: float,
    seed: int,
) -> dict[str, Any]:
    """Build A-MSB diagnostics from candidate completions."""
    completions = [str(value) for value in candidate_completions] or [""]

    correctness: list[bool] | None = None
    if candidate_correctness is not None and len(candidate_correctness) == len(
        completions
    ):
        correctness = [bool(value) for value in candidate_correctness]

    if correctness is None:
        correct_indices: list[int] = []
        error_indices = list(range(len(completions)))
    else:
        correct_indices = [idx for idx, value in enumerate(correctness) if value]
        error_indices = [idx for idx, value in enumerate(correctness) if not value]

    if not error_indices:
        error_indices = list(range(len(completions)))

    error_answers = [completions[idx] for idx in error_indices]
    all_error_clusters = cluster_answers(error_answers)
    top_error_clusters_relative = select_top_error_clusters(
        error_answers,
        max_clusters=max_error_clusters,
    )
    top_error_clusters = [
        [error_indices[inner_idx] for inner_idx in cluster]
        for cluster in top_error_clusters_relative
    ]

    sampled_group = sample_balanced_indices(
        correct_indices=correct_indices,
        top_error_clusters=top_error_clusters,
        group_size=balanced_group_size,
        seed=seed,
    )

    entropy = compute_semantic_entropy(error_answers, normalize_entropy=True)
    scale = entropy_scale_factor(
        entropy,
        mode=entropy_scale_mode,
        temperature=entropy_temperature,
    )

    return {
        "entropy": entropy,
        "scale": scale,
        "correct_pool_size": len(correct_indices),
        "error_pool_size": len(error_indices),
        "error_cluster_count": len(all_error_clusters),
        "top_error_cluster_count": len(top_error_clusters),
        "balanced_group_size": balanced_group_size,
        "balanced_correct_count": len(sampled_group["correct"]),
        "balanced_error_count": len(sampled_group["error"]),
    }
