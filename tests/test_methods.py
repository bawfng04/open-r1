# Copyright 2025 The HuggingFace Team. All rights reserved.

import math
import unittest

from open_r1.methods import (
    build_mgrpo_layer2_prompt,
    cluster_answers,
    compute_mgrpo_transition_metrics,
    compute_semantic_entropy,
    modulate_advantages,
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


if __name__ == "__main__":
    unittest.main()
