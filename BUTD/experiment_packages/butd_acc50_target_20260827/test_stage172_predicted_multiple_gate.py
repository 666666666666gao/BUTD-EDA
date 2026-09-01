#!/usr/bin/env python3
import unittest

import numpy as np

import stage172_predicted_multiple_gate as multiple


class PredictedMultipleGateTest(unittest.TestCase):
    def test_predicted_multiple_mask_requires_two_target_class_detections(self):
        counts = np.asarray([0.0, 1.0, 1.999, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(
            multiple.predicted_multiple_mask(counts),
            np.asarray([False, False, False, True, True]),
        )

    def test_feature_index_requires_exact_unique_count_feature(self):
        names = [
            "compact__text_target_detector_match_ratio",
            multiple.MULTIPLICITY_FEATURE,
            "meta__stage154_score",
        ]
        self.assertEqual(multiple.multiplicity_feature_index(names), 1)
        with self.assertRaises(AssertionError):
            multiple.multiplicity_feature_index(names + [
                multiple.MULTIPLICITY_FEATURE
            ])

    def test_gate_scores_abstains_outside_predicted_multiple_samples(self):
        scores = np.asarray([-1.0, 0.5, 2.0], dtype=np.float32)
        counts = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
        gated = multiple.gate_scores(scores, counts)
        self.assertEqual(float(gated[0]), -1.0)
        self.assertTrue(np.isneginf(gated[1]))
        self.assertEqual(float(gated[2]), 2.0)


if __name__ == "__main__":
    unittest.main()
