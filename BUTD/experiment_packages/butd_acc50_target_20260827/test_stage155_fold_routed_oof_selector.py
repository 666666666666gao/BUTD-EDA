#!/usr/bin/env python3
import unittest

import numpy as np

import stage155_fold_routed_oof_selector as routed


class Meta:
    def __init__(self, scene_id):
        self.scene_id = scene_id


class FoldRoutedSelectorTest(unittest.TestCase):
    def test_routed_predictions_use_one_fold_model_per_scene(self):
        metas = [
            Meta("scene0000_00"), Meta("scene0000_00"),
            Meta("scene0001_00"), Meta("scene0002_00"),
        ]
        features = np.arange(8, dtype=np.float32).reshape(4, 2)
        predictors = {fold: float(10 + fold) for fold in range(5)}

        def predict(value, fold_features):
            return np.full(len(fold_features), value, dtype=np.float32)

        scores, folds = routed.routed_predictions(
            predictors, features, metas, predict
        )
        self.assertEqual(folds.tolist(), [3, 3, 2, 3])
        self.assertEqual(scores.tolist(), [13.0, 13.0, 12.0, 13.0])

    def test_group_models_requires_all_five_folds_and_preserves_order(self):
        items = []
        for fold in range(5):
            items.append({"fold": fold, "fold_model_index": 1, "path": "b"})
            items.append({"fold": fold, "fold_model_index": 0, "path": "a"})
        grouped = routed.group_model_items(items, fold_count=5)
        self.assertEqual(
            [[item["fold_model_index"] for item in group] for group in grouped],
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        )
        with self.assertRaises(AssertionError):
            routed.group_model_items(items[:-2], fold_count=5)


if __name__ == "__main__":
    unittest.main()
