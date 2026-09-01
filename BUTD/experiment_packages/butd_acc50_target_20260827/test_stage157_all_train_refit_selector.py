#!/usr/bin/env python3
import unittest

import stage157_all_train_refit_selector as refit


class AllTrainRefitSelectorTest(unittest.TestCase):
    def test_refit_requires_prior_scene_test_authorization(self):
        lock = {
            "stage": "154_train_only_scene_oof_stage142_stage150_source_selector",
            "validation_labels_used_for_selection": False,
            "internal_gate_pass": True,
            "validation_evaluation_authorized": True,
            "selected_candidate": "fix_vs_break_classifier",
            "selected_oof": {"threshold": 0.48868875622749336},
        }
        identity = refit.validate_upstream_lock(lock)
        self.assertEqual(identity["selected_candidate"], "fix_vs_break_classifier")
        self.assertEqual(identity["threshold"], 0.48868875622749336)
        lock["internal_gate_pass"] = False
        with self.assertRaises(AssertionError):
            refit.validate_upstream_lock(lock)


if __name__ == "__main__":
    unittest.main()
