#!/usr/bin/env python3
import unittest

import numpy as np

import stage168_risk_capped_meta_selector as stage168


class RiskCappedThresholdTest(unittest.TestCase):
    def test_selects_high_precision_top_decile(self):
        count = 1000
        scores = np.linspace(0.0, 1.0, count, dtype=np.float32)
        old = np.full(count, 0.2, dtype=np.float32)
        new = old.copy()
        new[-80:] = 0.8
        old[-20:] = 0.8
        new[-20:] = 0.2
        result = stage168.choose_risk_capped_threshold(scores, old, new)
        self.assertLessEqual(result["changed_ratio"], 0.10 + 1e-12)
        self.assertTrue(result["high_precision_gate"])
        self.assertGreaterEqual(
            result["fix_break"]["fix_050"],
            1.75 * result["fix_break"]["break_050"],
        )
        self.assertGreater(result["fix_break"]["net_050"], 0)

    def test_abstains_without_high_precision_region(self):
        scores = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        old = np.full(100, 0.8, dtype=np.float32)
        new = np.full(100, 0.2, dtype=np.float32)
        result = stage168.choose_risk_capped_threshold(scores, old, new)
        self.assertEqual(result["changed_ratio"], 0.0)
        self.assertTrue(result["no_positive_feasible_threshold"])
        self.assertFalse(result["high_precision_gate"])


if __name__ == "__main__":
    unittest.main()
