#!/usr/bin/env python3
import unittest

import numpy as np

from stage167_stage154_stage165_meta_selector import (
    META_FEATURE_NAMES,
    augment_features,
)


class MetaSelectorFeatureTest(unittest.TestCase):
    def test_six_inference_safe_meta_features(self):
        matrix = np.zeros((3, 4), dtype=np.float32)
        result = augment_features(
            matrix,
            np.asarray([0.1, 0.4, 0.8], dtype=np.float32), 0.5,
            np.asarray([0.2, 0.6, 0.9], dtype=np.float32), 0.7,
        )
        self.assertEqual(result.shape, (3, 4 + len(META_FEATURE_NAMES)))
        self.assertEqual(result[0, -4], 0.0)
        self.assertEqual(result[2, -4], 1.0)
        self.assertEqual(result[1, -1], 0.0)
        self.assertEqual(result[2, -1], 1.0)


if __name__ == "__main__":
    unittest.main()
