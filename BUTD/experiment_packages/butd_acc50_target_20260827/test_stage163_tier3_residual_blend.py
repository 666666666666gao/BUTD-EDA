#!/usr/bin/env python3
import unittest

import numpy as np

import stage163_tier3_residual_blend as residual


class Tier3ResidualBlendTest(unittest.TestCase):
    def test_fixed_residual_blend_is_point_one(self):
        old = np.asarray([1.0, -1.0], dtype=np.float32)
        new = np.asarray([0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(
            residual.fixed_blend(old, new), [0.9, -0.8], atol=1e-7
        )

    def test_internal_gate_compares_against_stage142(self):
        stage142 = {"hits025": 4700, "hits050": 4400, "count": 5647}
        selected = {
            "selected": {"hits025": 4695, "hits050": 4405, "count": 5647},
            "changed_ratio": 0.9,
        }
        self.assertTrue(residual.internal_gate_pass(selected, stage142))
        selected["selected"]["hits050"] = 4404
        self.assertFalse(residual.internal_gate_pass(selected, stage142))
        selected["selected"]["hits050"] = 4406
        selected["selected"]["hits025"] = 4680
        self.assertFalse(residual.internal_gate_pass(selected, stage142))


if __name__ == "__main__":
    unittest.main()
