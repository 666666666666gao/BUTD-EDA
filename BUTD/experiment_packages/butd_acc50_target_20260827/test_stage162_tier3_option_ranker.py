#!/usr/bin/env python3
import unittest
import json
import os
import tempfile

import numpy as np

import stage162_tier3_option_ranker as tier3


class Tier3OptionRankerTest(unittest.TestCase):
    def test_atomic_json_writes_valid_payload(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "receipt.json")
            tier3.atomic_json(path, {"ok": True, "value": 3})
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"ok": True, "value": 3})

    def test_tier3_labels_follow_acc025_and_acc050_boundaries(self):
        labels = tier3.tier3_labels(np.asarray([
            0.0, 0.2499, 0.25, 0.4999, 0.50, 0.75
        ], dtype=np.float32))
        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])

    def test_internal_gate_requires_acc050_gain_and_acc025_safety(self):
        self.assertTrue(tier3.internal_gate_pass({
            "baseline": {"acc025": 0.80, "acc050": 0.70},
            "selected": {"acc025": 0.799, "acc050": 0.711},
            "changed_ratio": 0.8,
        }))
        self.assertFalse(tier3.internal_gate_pass({
            "baseline": {"acc025": 0.80, "acc050": 0.70},
            "selected": {"acc025": 0.799, "acc050": 0.709},
            "changed_ratio": 0.8,
        }))
        self.assertFalse(tier3.internal_gate_pass({
            "baseline": {"acc025": 0.80, "acc050": 0.70},
            "selected": {"acc025": 0.797, "acc050": 0.72},
            "changed_ratio": 0.8,
        }))


if __name__ == "__main__":
    unittest.main()
