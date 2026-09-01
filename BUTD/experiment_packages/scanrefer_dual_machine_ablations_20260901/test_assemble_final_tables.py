import json
import os
import tempfile
import unittest

from assemble_final_tables import assemble


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)
        handle.write("\n")


class AssembleFinalTablesTest(unittest.TestCase):
    def make_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        manifest = {
            "baseline": {"overall_025": 50.42, "overall_050": 38.6},
            "main_rows": {
                "M1": {"overall_025": 50.8309, "overall_050": 37.4632},
                "M2": {"overall_025": 53.5654, "overall_050": 40.1662},
                "M3": {"overall_025": 54.9011, "overall_050": 42.3538},
            },
            "matched_internal_full": {
                "overall_025": 53.7547,
                "overall_050": 39.7770,
                "multiple_025": 48.1024,
                "multiple_050": 35.1959,
            },
            "reused_internal_rows": {
                "S1": {
                    "overall_025": 53.2814,
                    "overall_050": 39.7665,
                    "multiple_025": 47.8057,
                    "multiple_050": 35.2207,
                }
            },
        }
        manifest_path = os.path.join(temporary.name, "manifest.json")
        write_json(manifest_path, manifest)
        rows = []
        for index, row in enumerate(("S0", "S2", "S3", "R0", "R1", "R2", "R3")):
            payload = {
                "row": row,
                "best_epoch": 20 + index * 5,
                "checkpoint_sha256": "{:064x}".format(index + 1),
                "counts": {"unique": 1419, "multiple": 8089, "total": 9508},
                "metrics": {
                    "overall_025": 0.51 + index * 0.001,
                    "overall_050": 0.39 + index * 0.001,
                    "unique_025": 0.8,
                    "unique_050": 0.6,
                    "multiple_025": 0.46 + index * 0.001,
                    "multiple_050": 0.35 + index * 0.001,
                },
                "percent": {
                    "overall_025": 51.0 + index * 0.1,
                    "overall_050": 39.0 + index * 0.1,
                    "unique_025": 80.0,
                    "unique_050": 60.0,
                    "multiple_025": 46.0 + index * 0.1,
                    "multiple_050": 35.0 + index * 0.1,
                },
                "reload_parity": True,
                "training_protocol": {
                    "rng_seed": 0,
                    "batch_size": 24,
                    "max_epoch": 65,
                    "val_freq": 5,
                    "lr": 0.0001,
                    "lr_backbone": 0.001,
                    "lr_decay_epochs": [55, 60],
                },
            }
            path = os.path.join(temporary.name, "{}.json".format(row))
            write_json(path, payload)
            rows.append(path)
        return temporary, manifest_path, rows

    def test_assembles_all_three_tables_from_exactly_seven_rows(self):
        temporary, manifest_path, rows = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        output_json = os.path.join(temporary.name, "final.json")
        output_md = os.path.join(temporary.name, "final.md")

        result = assemble(manifest_path, rows, output_json, output_md)

        self.assertEqual(result["new_row_count"], 7)
        self.assertEqual(result["sacr_internal"]["S2"]["overall_025"], 51.1)
        self.assertEqual(result["rapf_internal"]["R3"]["multiple_050"], 35.6)
        self.assertEqual(result["main_modules"]["M3"]["overall_050"], 42.3538)
        with open(output_md, "r") as handle:
            markdown = handle.read()
        self.assertIn("## Table 3. Main modules", markdown)
        self.assertIn("## Table 4. SACR internal design", markdown)
        self.assertIn("## Table 5. RAPF internal design", markdown)
        self.assertTrue(os.path.isfile(output_json))

    def test_rejects_cross_row_protocol_drift(self):
        temporary, manifest_path, rows = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        with open(rows[-1], "r") as handle:
            payload = json.load(handle)
        payload["training_protocol"]["batch_size"] = 16
        write_json(rows[-1], payload)

        with self.assertRaisesRegex(ValueError, "training protocol differs"):
            assemble(
                manifest_path,
                rows,
                os.path.join(temporary.name, "final.json"),
                os.path.join(temporary.name, "final.md"),
            )

    def test_rejects_duplicate_checkpoint_identity(self):
        temporary, manifest_path, rows = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        with open(rows[0], "r") as handle:
            first = json.load(handle)
        with open(rows[1], "r") as handle:
            second = json.load(handle)
        second["checkpoint_sha256"] = first["checkpoint_sha256"]
        write_json(rows[1], second)

        with self.assertRaisesRegex(ValueError, "duplicate checkpoint SHA256"):
            assemble(
                manifest_path,
                rows,
                os.path.join(temporary.name, "final.json"),
                os.path.join(temporary.name, "final.md"),
            )


if __name__ == "__main__":
    unittest.main()
