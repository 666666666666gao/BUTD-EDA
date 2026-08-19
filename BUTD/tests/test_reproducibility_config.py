import unittest
from unittest import mock

import torch

import train_dist_mod


class ReproducibilityConfigTest(unittest.TestCase):
    def test_configure_reproducibility_seeds_all_rngs_and_disables_benchmark(self):
        old_benchmark = torch.backends.cudnn.benchmark
        old_deterministic = torch.backends.cudnn.deterministic
        old_enabled = torch.backends.cudnn.enabled
        try:
            with mock.patch.object(train_dist_mod.random, "seed") as random_seed:
                with mock.patch.object(train_dist_mod.np.random, "seed") as numpy_seed:
                    with mock.patch.object(train_dist_mod.torch, "manual_seed") as torch_seed:
                        with mock.patch.object(
                            train_dist_mod.torch.cuda, "manual_seed_all"
                        ) as cuda_seed:
                            train_dist_mod.configure_reproducibility(123)

            random_seed.assert_called_once_with(123)
            numpy_seed.assert_called_once_with(123)
            torch_seed.assert_called_once_with(123)
            cuda_seed.assert_called_once_with(123)
            self.assertTrue(torch.backends.cudnn.enabled)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
        finally:
            torch.backends.cudnn.benchmark = old_benchmark
            torch.backends.cudnn.deterministic = old_deterministic
            torch.backends.cudnn.enabled = old_enabled


if __name__ == "__main__":
    unittest.main()
