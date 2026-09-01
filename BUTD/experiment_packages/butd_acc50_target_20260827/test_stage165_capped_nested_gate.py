#!/usr/bin/env python3
import unittest

import numpy as np

import stage165_capped_nested_gate as capped


class CappedNestedGateTest(unittest.TestCase):
    def test_dev_gate_never_exceeds_fixed_change_cap(self):
        groups = np.asarray([2] * 10, dtype=np.int32)
        baselines = np.asarray([0] * 10, dtype=np.int32)
        scores = np.asarray([
            value for index in range(10)
            for value in (0.0, 1.0 - 0.05 * index)
        ], dtype=np.float32)
        ious = np.asarray([
            value for index in range(10)
            for value in (0.2, 0.6 if index < 8 else 0.1)
        ], dtype=np.float32)
        gate = capped.choose_capped_gate(scores, ious, groups, baselines)
        self.assertLessEqual(
            gate["dev"]["changed_ratio"], capped.MAX_CHANGED_RATIO
        )
        self.assertGreaterEqual(
            gate["dev"]["selected"]["hits025"],
            gate["dev"]["baseline"]["hits025"],
        )

    def test_cap_is_fixed_at_point_nine(self):
        self.assertEqual(capped.MAX_CHANGED_RATIO, 0.90)


if __name__ == "__main__":
    unittest.main()
