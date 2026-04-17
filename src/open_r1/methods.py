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
import re
from collections import defaultdict
from typing import Iterable, Sequence


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
