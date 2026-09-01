#!/usr/bin/env python3
import math
import unittest

import numpy as np

import stage153_train_source_selector as selector


class SourceSelectorTest(unittest.TestCase):
    def test_source_feature_allowlist_blocks_training_labels(self):
        row = {
            "fused_margin": 0.4,
            "same_query": 1.0,
            "candidate_iou": 0.9,
            "oracle_source_id": 2,
            "threshold_utility": 3.0,
            "gt_center_x": 1.5,
            "base_top_query": 17,
            "absolute_center_x": 2.0,
            "normalized_center_delta_x": 0.2,
            "metadata": "not numeric",
        }
        names = selector.choose_source_feature_names(row)
        self.assertEqual(
            names,
            ["fused_margin", "normalized_center_delta_x", "same_query"],
        )

    def test_threshold_lock_prefers_high_precision_fixes(self):
        old = np.full(100, 0.30, dtype=np.float32)
        new = old.copy()
        old[:10] = 0.30
        new[:10] = 0.60
        old[10:12] = 0.60
        new[10:12] = 0.30
        scores = np.full(100, 0.10, dtype=np.float32)
        scores[:10] = 0.90
        scores[10:12] = 0.70

        locked = selector.choose_threshold(scores, old, new)
        self.assertEqual(locked["fix_break"]["fix_050"], 10)
        self.assertEqual(locked["fix_break"]["break_050"], 0)
        self.assertEqual(locked["fix_break"]["net_025"], 0)
        self.assertLessEqual(locked["changed_ratio"], 0.20)
        self.assertTrue(locked["precision_gate"])

    def test_threshold_lock_abstains_with_json_safe_finite_value(self):
        old = np.full(100, 0.60, dtype=np.float32)
        new = np.full(100, 0.30, dtype=np.float32)
        scores = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
        locked = selector.choose_threshold(scores, old, new)
        self.assertTrue(math.isfinite(locked["threshold"]))
        self.assertEqual(locked["fix_break"]["changed"], 0)
        self.assertTrue(locked["no_positive_feasible_threshold"])

    def test_feature_dict_is_finite_and_does_not_admit_source_labels(self):
        source_row = {
            "fused_margin": 0.4,
            "candidate_iou": 0.9,
            "oracle_source_id": 2,
            "base_top_query": 2,
            "fused_top_query": 3,
            "quality_top_query": 3,
            "contrastive_base_top_query": 1,
            "acd_top_query": 4,
            "target_detector_top_query": 2,
            "target_detector_logit_top_query": 3,
        }
        source_names = selector.choose_source_feature_names(source_row)
        compact = {
            "adapter_score_at_candidate": [0.1, 0.8],
            "adapter_iou_at_candidate": [0.2, 0.6],
            "adapter_box_at_candidate": [
                [0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1]
            ],
            "adapter_candidate_query": [2, 3],
        }
        raw = {
            "decomposition_status": "ok",
            "spacy_augmentation_bucket": "spacy_aug_none",
            "spacy_profile_bucket": "spacy_profile_none",
        }
        features = selector.feature_dict(
            source_row,
            source_names,
            compact,
            raw,
            np.zeros(len(selector.option_ranker.FEATURE_NAMES), dtype=np.float32),
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32),
            2,
            {"safe_gap": 0.1},
        )
        self.assertIn("source__fused_margin", features)
        self.assertNotIn("source__candidate_iou", features)
        self.assertNotIn("source__oracle_source_id", features)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))


if __name__ == "__main__":
    unittest.main()
