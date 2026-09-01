#!/usr/bin/env python3
import unittest

import stage158_regularized_selector as regularized


class RegularizedSelectorTest(unittest.TestCase):
    def test_candidate_specs_are_low_capacity_and_distinct(self):
        specs = regularized.candidate_specs()
        self.assertEqual(len({item["variant"] for item in specs}), len(specs))
        self.assertGreaterEqual(len(specs), 3)
        for item in specs:
            self.assertLessEqual(item["max_depth"], 3)
            self.assertLessEqual(item["num_leaves"], 7)
            self.assertLessEqual(item["n_estimators"], 200)
            self.assertGreaterEqual(item["min_child_samples"], 240)


if __name__ == "__main__":
    unittest.main()
