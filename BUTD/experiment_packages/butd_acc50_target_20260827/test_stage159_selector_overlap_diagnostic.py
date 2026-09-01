#!/usr/bin/env python3
import unittest

import numpy as np

import stage159_selector_overlap_diagnostic as diagnostic


class SelectorOverlapDiagnosticTest(unittest.TestCase):
    def test_masks_partition_union_and_neither(self):
        masks = diagnostic.policy_masks(
            np.asarray([0.9, 0.8, 0.1, 0.2]), 0.5,
            np.asarray([0.9, 0.1, 0.8, 0.2]), 0.5,
        )
        np.testing.assert_array_equal(
            masks["intersection"], [True, False, False, False]
        )
        np.testing.assert_array_equal(
            masks["stage154_only"], [False, True, False, False]
        )
        np.testing.assert_array_equal(
            masks["stage158_only"], [False, False, True, False]
        )
        np.testing.assert_array_equal(
            masks["neither"], [False, False, False, True]
        )
        self.assertTrue(np.all(masks["union"] | masks["neither"]))
        self.assertFalse(np.any(masks["union"] & masks["neither"]))


if __name__ == "__main__":
    unittest.main()
