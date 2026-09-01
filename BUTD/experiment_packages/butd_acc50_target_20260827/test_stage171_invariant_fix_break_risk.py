#!/usr/bin/env python3
import unittest

import numpy as np

import stage171_invariant_fix_break_risk as invariant


class InvariantFixBreakRiskTest(unittest.TestCase):
    def test_invariant_feature_policy_keeps_scores_and_drops_scene_geometry(self):
        cases = {
            "meta__stage154_margin": True,
            "compact__adapter_score_at_candidate__entropy": True,
            "cross__adapter_hit50_logit_at_candidate__new_minus_old": True,
            "cross__box_iou": True,
            "stage142_context__safe_margin": True,
            "stage142_option__z_minus_baseline_hit50": True,
            "source__query_features_17": False,
            "compact__selected_log_size_x": False,
            "compact__detector_conf50_count": False,
            "compact__adapter_score_at_candidate__count": False,
            "cross__normalized_center_delta_x": False,
            "cross__log_size_ratio_z": False,
            "stage142_context__group_size_norm": False,
            "stage142_option__detector_count_norm": False,
            "stage142_option__pred_log_volume": False,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    invariant.is_invariant_feature_name(name), expected
                )

    def test_select_invariant_features_preserves_order_and_matrix_alignment(self):
        names = [
            "source__raw_0",
            "meta__stage154_score",
            "cross__normalized_center_delta_x",
            "stage142_context__safe_margin",
            "compact__adapter_fused_at_candidate__std",
        ]
        matrix = np.arange(15, dtype=np.float32).reshape(3, 5)
        selected, selected_names, selected_indices = (
            invariant.select_invariant_features(matrix, names)
        )
        self.assertEqual(
            selected_names,
            [
                "meta__stage154_score",
                "stage142_context__safe_margin",
                "compact__adapter_fused_at_candidate__std",
            ],
        )
        self.assertEqual(selected_indices, [1, 3, 4])
        np.testing.assert_array_equal(selected, matrix[:, [1, 3, 4]])


if __name__ == "__main__":
    unittest.main()
