#!/usr/bin/env python3
import unittest

import numpy as np

import stage156_fold_ensemble_selector as ensemble


class FoldEnsembleSelectorTest(unittest.TestCase):
    def test_ensemble_predictions_average_all_locked_fold_models(self):
        predictors = {fold: float(10 + fold) for fold in range(5)}
        features = np.arange(8, dtype=np.float32).reshape(4, 2)

        def predict(value, fold_features):
            return np.full(len(fold_features), value, dtype=np.float32)

        scores = ensemble.ensemble_predictions(
            predictors, features, predict
        )
        self.assertEqual(scores.tolist(), [12.0, 12.0, 12.0, 12.0])


if __name__ == "__main__":
    unittest.main()
