#!/usr/bin/env python3
import unittest

import stage160_invariant_feature_selector as invariant


class InvariantFeatureSelectorTest(unittest.TestCase):
    def test_feature_groups_remove_absolute_geometry_without_losing_relative_geometry(self):
        names = [
            "stage142_context__safe_margin",
            "stage142_option__center_delta_norm_x",
            "stage142_option__option_log_size_x",
            "cross__adapter_hit50_logit_at_candidate__new_minus_old",
            "cross__normalized_center_delta_x",
            "cross__log_size_ratio_x",
            "compact__selected_log_size_x",
            "compact__quality_topk_score__margin",
            "source__base_top_size_x",
            "source__base_top_center_x",
            "source__source_pair_base_quality_top_size_l1_delta",
            "source__base_rapf_gate",
        ]
        groups = invariant.feature_groups(names)
        self.assertEqual(set(groups), {
            "score_only", "relative_geometry", "compact_relative_geometry"
        })
        for group in groups.values():
            self.assertNotIn("stage142_option__option_log_size_x", group)
            self.assertNotIn("compact__selected_log_size_x", group)
            self.assertNotIn("source__base_top_size_x", group)
            self.assertNotIn("source__base_top_center_x", group)
        self.assertNotIn("cross__normalized_center_delta_x", groups["score_only"])
        self.assertIn("cross__normalized_center_delta_x", groups["relative_geometry"])
        self.assertIn(
            "source__source_pair_base_quality_top_size_l1_delta",
            groups["relative_geometry"],
        )
        self.assertNotIn("source__base_rapf_gate", groups["compact_relative_geometry"])
        self.assertIn("compact__quality_topk_score__margin", groups["compact_relative_geometry"])


if __name__ == "__main__":
    unittest.main()
