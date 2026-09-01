#!/usr/bin/env python3
import math
import unittest

import numpy as np

import stage170_explicit_fix_break_risk as risk


class ExplicitFixBreakRiskTest(unittest.TestCase):
    def test_log_odds_ratio_prefers_fix_and_penalizes_break(self):
        scores = risk.log_odds_ratio(
            np.asarray([0.8, 0.2], dtype=np.float32),
            np.asarray([0.2, 0.8], dtype=np.float32),
        )
        expected = 2.0 * math.log(4.0)
        self.assertAlmostEqual(float(scores[0]), expected, places=5)
        self.assertAlmostEqual(float(scores[1]), -expected, places=5)

    def test_log_odds_ratio_is_finite_at_probability_boundaries(self):
        scores = risk.log_odds_ratio(
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertTrue(np.isfinite(scores).all())
        self.assertLess(float(scores[0]), 0.0)
        self.assertGreater(float(scores[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
