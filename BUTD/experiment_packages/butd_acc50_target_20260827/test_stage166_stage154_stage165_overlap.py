#!/usr/bin/env python3
import unittest

import numpy as np

from stage166_stage154_stage165_overlap import pair_overlap


class Stage154Stage165OverlapTest(unittest.TestCase):
    def test_pair_oracle_and_partitions(self):
        first = np.asarray([0.6, 0.4, 0.2, 0.1], dtype=np.float32)
        second = np.asarray([0.4, 0.6, 0.3, 0.1], dtype=np.float32)
        result = pair_overlap(first, second)
        self.assertEqual(result["pair_oracle"]["hits050"], 2)
        self.assertEqual(
            result["threshold_partitions"]["050"]["stage154_only_hit"], 1
        )
        self.assertEqual(
            result["threshold_partitions"]["050"]["stage165_only_hit"], 1
        )


if __name__ == "__main__":
    unittest.main()
