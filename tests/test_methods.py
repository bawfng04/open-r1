# Copyright 2025 The HuggingFace Team. All rights reserved.

import math
import unittest

from open_r1.methods import (
    build_amsb_plan,
    build_mgrpo_layer2_prompt,
    cluster_answers,
    compute_mgrpo_transition_metrics,
    compute_semantic_entropy,
    entropy_scale_factor,
    modulate_advantages,
    select_top_error_clusters,
    sample_balanced_indices,
)


class TestMethods(unittest.TestCase):
    def test_cluster_answers(self):
        clusters = cluster_answers(
            [
                "<answer>\n3\n</answer>",
                "3",
                "<answer>\n4\n</answer>",
            ]
        )
        self.assertEqual(len(clusters), 2)

    def test_semantic_entropy_normalized(self):
        entropy = compute_semantic_entropy(["3", "3", "4"], normalize_entropy=True)
        self.assertGreater(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)

    def test_modulate_advantages_linear(self):
        base = [1.0, -0.5]
        scaled = modulate_advantages(base, entropy=0.5, alpha=1.0, mode="linear")
        self.assertEqual(len(scaled), 2)
        self.assertGreater(scaled[0], base[0])

    def test_mgrpo_prompt_builder(self):
        prompt = build_mgrpo_layer2_prompt("Q", "A1", "Please revise")
        self.assertIn("Please revise", prompt)
        self.assertIn("First attempt", prompt)

    def test_mgrpo_transition_metrics(self):
        metrics = compute_mgrpo_transition_metrics(
            [False, True, False, True],
            [True, True, False, False],
        )
        self.assertTrue(math.isclose(metrics["acc_t1"], 0.5))
        self.assertTrue(math.isclose(metrics["acc_t2"], 0.5))
        self.assertTrue(math.isclose(metrics["delta_i_to_c"], 0.25))
        self.assertTrue(math.isclose(metrics["delta_c_to_i"], 0.25))

    def test_select_top_error_clusters(self):
        error_answers = ["3", "3", "4", "5", "5", "5"]
        top_clusters = select_top_error_clusters(error_answers, max_clusters=2)
        self.assertEqual(len(top_clusters), 2)
        self.assertEqual(len(top_clusters[0]), 3)

    def test_sample_balanced_indices(self):
        sampled = sample_balanced_indices(
            correct_indices=[0, 1, 2],
            top_error_clusters=[[3, 4], [5]],
            group_size=8,
            seed=7,
        )
        self.assertEqual(len(sampled["correct"]), 4)
        self.assertEqual(len(sampled["error"]), 4)
        self.assertEqual(len(sampled["all"]), 8)

    def test_entropy_scale_factor(self):
        scale = entropy_scale_factor(0.7, mode="exp_decay", temperature=1.0)
        self.assertGreater(scale, 0.0)
        self.assertLessEqual(scale, 1.0)

    def test_build_amsb_plan(self):
        plan = build_amsb_plan(
            candidate_completions=["3", "3", "4", "5"],
            candidate_correctness=[True, False, False, True],
            balanced_group_size=6,
            max_error_clusters=2,
            entropy_scale_mode="exp_decay",
            entropy_temperature=1.0,
            seed=42,
        )
        self.assertEqual(plan["balanced_group_size"], 6)
        self.assertEqual(plan["balanced_correct_count"], 3)
        self.assertEqual(plan["balanced_error_count"], 3)
        self.assertGreaterEqual(plan["error_cluster_count"], 1)


if __name__ == "__main__":
    unittest.main()
