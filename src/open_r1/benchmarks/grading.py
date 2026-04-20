"""Answer extraction and grading helpers for dry-run benchmark outputs."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _try_import_ngrpo_math_equal():
    repo_root = Path(__file__).resolve().parents[4]
    ngrpo_dir = repo_root.parent / "NGRPO"
    if not ngrpo_dir.exists():
        return None

    if str(ngrpo_dir) not in sys.path:
        sys.path.insert(0, str(ngrpo_dir))

    try:
        from grader import math_equal  # type: ignore

        return math_equal
    except Exception:
        return None


def extract_predicted_answer(response: str) -> str:
    if not response:
        return ""

    tag_matches = ANSWER_TAG_PATTERN.findall(response)
    if tag_matches:
        return tag_matches[-1].strip()

    boxed_matches = BOXED_PATTERN.findall(response)
    if boxed_matches:
        return boxed_matches[-1].strip()

    number_matches = NUMBER_PATTERN.findall(response)
    if number_matches:
        return number_matches[-1].strip()

    return response.strip()


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    if text.endswith("."):
        text = text[:-1]
    return text


def _try_float(text: str):
    try:
        return float(text)
    except Exception:
        return None


def grade_prediction(predicted: str, reference: str, dataset_name: str) -> bool:
    pred = _normalize(predicted)
    ref = _normalize(reference)

    aliases = {
        "math_500": "math500",
        "aime24": "aime2025",
    }
    canonical_dataset = aliases.get(dataset_name, dataset_name)

    math_equal = _try_import_ngrpo_math_equal()
    if math_equal is not None and canonical_dataset in {
        "math500",
        "gsm8k",
        "aime2025",
        "olympiadbench",
        "amc23",
    }:
        try:
            return bool(math_equal(pred, ref, timeout=False))
        except Exception:
            pass

    if pred == ref:
        return True

    pred_val = _try_float(pred)
    ref_val = _try_float(ref)
    if pred_val is not None and ref_val is not None:
        return math.isclose(pred_val, ref_val, rel_tol=1e-9, abs_tol=1e-9)

    return False
