import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_spacy_refined import audit_path


class TestSpacyRefinedAudit(unittest.TestCase):
    def test_audit_counts_json_and_csv_refined_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scanrefer = root / "ScanRefer_filtered_train_spacy_refined.json"
            nr3d = root / "nr3d_spacy_refined.csv"

            scanrefer.write_text(json.dumps([
                {
                    "description": "the top pillow from the right",
                    "rel_slots": [],
                    "attr_slot": {
                        "items": [{"text": "top", "type": "spatial_attribute"}]
                    },
                    "coverage_stats": {"candidate_relation_count": 1},
                    "decomp_train_global_only_mask": True,
                    "decomp_train_weak_generic_mask": True,
                },
                {
                    "description": "the chair left of the table",
                    "rel_slots": [{"text": "left of"}],
                    "attr_slot": {"items": []},
                    "coverage_stats": {"refined_relation_count": 1},
                },
            ]))

            with nr3d.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "utterance",
                        "rel_slots",
                        "attr_slot",
                        "coverage_stats",
                        "decomp_train_global_only_mask",
                        "decomp_train_weak_generic_mask",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "utterance": "the box closest to the north wall",
                    "rel_slots": json.dumps([{"text": "closest to the north"}]),
                    "attr_slot": json.dumps({"items": []}),
                    "coverage_stats": json.dumps({"refined_relation_count": 1}),
                    "decomp_train_global_only_mask": "true",
                    "decomp_train_weak_generic_mask": "true",
                })

            stats = audit_path(root)

        self.assertEqual(stats["rows"], 3)
        self.assertEqual(stats["relation_free_rows"], 1)
        self.assertEqual(stats["relation_rows"], 2)
        self.assertEqual(stats["raw_view_rows"], 2)
        self.assertEqual(stats["compass_rows"], 1)
        self.assertEqual(stats["spatial_attribute_rows"], 1)
        self.assertEqual(stats["weak_parse_rows"], 1)
        self.assertEqual(stats["weak_raw_view_rows"], 1)
        self.assertEqual(stats["train_global_only_rows"], 2)
        self.assertEqual(stats["train_weak_generic_rows"], 2)


if __name__ == "__main__":
    unittest.main()
