#!/usr/bin/env python3
import unittest

import stage154_oof_source_selector as oof


class Meta:
    def __init__(self, scene_id):
        self.scene_id = scene_id


class OofSelectorTest(unittest.TestCase):
    def test_oof_folds_are_scene_disjoint_and_cover_development(self):
        metas = []
        for scene_index in range(80):
            metas.extend([Meta("scene_{:03d}".format(scene_index))] * 3)
        development = list(range(len(metas)))
        folds = oof.oof_folds(metas, development, fold_count=5)
        covered = []
        for item in folds:
            fit_scenes = {metas[index].scene_id for index in item["fit"]}
            held_scenes = {metas[index].scene_id for index in item["heldout"]}
            self.assertTrue(fit_scenes.isdisjoint(held_scenes))
            covered.extend(item["heldout"].tolist())
        self.assertEqual(sorted(covered), development)

    def test_candidate_lookup_is_exact(self):
        candidates = [
            {"name": "a", "models": [1]},
            {"name": "b", "models": [2]},
        ]
        self.assertEqual(oof.candidate_by_name(candidates, "b")["models"], [2])
        with self.assertRaises(AssertionError):
            oof.candidate_by_name(candidates, "missing")


if __name__ == "__main__":
    unittest.main()
