#!/usr/bin/env python3
import unittest

import stage155_enrich_validation_scene_ids as enrich


class EnrichValidationSceneIdsTest(unittest.TestCase):
    def test_filtered_scene_ids_follow_dataset_reader_order(self):
        annotations = [
            {"scene_id": "scene_b"},
            {"scene_id": "scene_skip"},
            {"scene_id": "scene_a"},
            {"scene_id": "scene_b"},
        ]
        self.assertEqual(
            enrich.filtered_scene_ids(
                annotations, {"scene_a", "scene_b"}
            ),
            ["scene_b", "scene_a", "scene_b"],
        )


if __name__ == "__main__":
    unittest.main()
