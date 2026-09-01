import json
import os
import shutil
import subprocess
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
            "baseline": {
                "source": "published_BUTD_DETR_not_retrained",
                "protocol_note": "published external baseline",
                "unique_025": 82.88,
                "unique_050": 64.98,
                "multiple_025": 44.73,
                "multiple_050": 33.97,
                "overall_025": 50.42,
                "overall_050": 38.6,
            },
            "main_rows": {
                "M1": {
                    "status": "reuse_audited",
                    "best_epoch": 65,
                    "checkpoint_sha256": "a" * 64,
                    "unique_025": 81.5363,
                    "unique_050": 62.0155,
                    "multiple_025": 45.4444,
                    "multiple_050": 33.1561,
                    "overall_025": 50.8309,
                    "overall_050": 37.4632,
                },
                "M2": {
                    "status": "reuse_audited_strict_best_E60",
                    "best_epoch": 60,
                    "checkpoint_sha256": "b" * 64,
                    "unique_025": 84.2847,
                    "unique_050": 65.1868,
                    "multiple_025": 48.1765,
                    "multiple_050": 35.7770,
                    "overall_025": 53.5654,
                    "overall_050": 40.1662,
                },
                "M3": {
                    "status": "reuse_stage154_without_retraining",
                    "label": "full_plus_fixed_source_selection_calibration",
                    "single_checkpoint": False,
                    "validation_result_sha256": "c" * 64,
                    "subgroup_report_sha256": "d" * 64,
                    "subgroup_source_manifest_sha256": "e" * 64,
                    "unique_025": 87.3855,
                    "unique_050": 70.2607,
                    "multiple_025": 49.2026,
                    "multiple_050": 37.4583,
                    "overall_025": 54.9011,
                    "overall_050": 42.3538,
                },
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
        self.assertTrue(result["main_monotonicity"]["overall_025_non_decreasing"])
        self.assertFalse(result["main_monotonicity"]["overall_050_non_decreasing"])
        self.assertAlmostEqual(
            result["internal_effects"]["sacr"]["S1"]["overall_025_full_minus_variant"],
            0.4733,
        )
        self.assertTrue(
            result["internal_effects"]["sacr"]["S1"]["overall_025_full_not_lower"]
        )
        self.assertFalse(
            result["internal_effects"]["rapf"]["R3"]["multiple_050_full_not_lower"]
        )
        self.assertTrue(
            result["internal_claim_summary"][
                "sacr_all_variants_full_not_lower_overall_025"
            ]
        )
        self.assertTrue(
            result["internal_claim_summary"][
                "rapf_all_variants_full_not_lower_overall_025"
            ]
        )
        with open(output_md, "r") as handle:
            markdown = handle.read()
        self.assertIn("## Table 3. Main modules", markdown)
        self.assertIn("| Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 |", markdown)
        self.assertIn("## Table 4. SACR internal design", markdown)
        self.assertIn("## Table 5. RAPF internal design", markdown)
        self.assertIn("## Evidence checks", markdown)
        self.assertIn("Main Overall@0.50 non-decreasing: no", markdown)
        self.assertIn("RAPF Full not lower than every variant on Multiple@0.50: no", markdown)
        self.assertTrue(os.path.isfile(output_json))

    def test_writes_three_paper_ready_latex_tables_with_provenance_note(self):
        temporary, manifest_path, rows = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        output_json = os.path.join(temporary.name, "final.json")
        output_md = os.path.join(temporary.name, "final.md")
        latex_dir = os.path.join(temporary.name, "latex")

        assemble(
            manifest_path,
            rows,
            output_json,
            output_md,
            output_latex_dir=latex_dir,
        )

        expected = (
            "TABLE_3_main_modules.tex",
            "TABLE_4_sacr_internal.tex",
            "TABLE_5_rapf_internal.tex",
            "TABLES_scanrefer_ablations_all.tex",
        )
        for name in expected:
            self.assertTrue(os.path.isfile(os.path.join(latex_dir, name)), name)

        with open(os.path.join(latex_dir, expected[0]), "r") as handle:
            main_table = handle.read()
        self.assertIn("\\begin{table*}[t]", main_table)
        self.assertIn("SACR & RAPF & QAHNL", main_table)
        self.assertIn("fixed source-selection calibration", main_table)
        self.assertIn("Stage154", main_table)
        self.assertIn("not a single-checkpoint result", main_table)
        self.assertIn("published BUTD-DETR result", main_table)
        self.assertIn("\\label{tab:scanrefer_main_ablation}", main_table)

        with open(os.path.join(latex_dir, expected[1]), "r") as handle:
            sacr_table = handle.read()
        self.assertIn("Multiple@0.25", sacr_table)
        self.assertIn("\\label{tab:scanrefer_sacr_ablation}", sacr_table)

        with open(os.path.join(latex_dir, expected[2]), "r") as handle:
            rapf_table = handle.read()
        self.assertIn("fixed fusion", rapf_table)
        self.assertIn("\\label{tab:scanrefer_rapf_ablation}", rapf_table)

    def test_rejects_stage154_if_mislabeled_as_single_checkpoint(self):
        temporary, manifest_path, rows = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        with open(manifest_path, "r") as handle:
            manifest = json.load(handle)
        manifest["main_rows"]["M3"]["single_checkpoint"] = True
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "M3 must be the audited Stage154"):
            assemble(
                manifest_path,
                rows,
                os.path.join(temporary.name, "final.json"),
                os.path.join(temporary.name, "final.md"),
            )

    def test_rejects_missing_published_baseline_provenance(self):
        temporary, manifest_path, rows = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        with open(manifest_path, "r") as handle:
            manifest = json.load(handle)
        manifest["baseline"]["source"] = "unknown"
        write_json(manifest_path, manifest)

        with self.assertRaisesRegex(ValueError, "M0 baseline provenance"):
            assemble(
                manifest_path,
                rows,
                os.path.join(temporary.name, "final.json"),
                os.path.join(temporary.name, "final.md"),
            )

    @unittest.skipUnless(shutil.which("pdflatex"), "pdflatex is unavailable")
    def test_generated_latex_compiles_without_overfull_boxes(self):
        temporary, manifest_path, rows = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        output_json = os.path.join(temporary.name, "final.json")
        output_md = os.path.join(temporary.name, "final.md")
        latex_dir = os.path.join(temporary.name, "latex")
        assemble(
            manifest_path,
            rows,
            output_json,
            output_md,
            output_latex_dir=latex_dir,
        )
        combined = os.path.join(latex_dir, "TABLES_scanrefer_ablations_all.tex")
        document = os.path.join(latex_dir, "compile_test.tex")
        with open(combined, "r") as handle:
            tables = handle.read()
        with open(document, "w") as handle:
            handle.write(
                "\\documentclass[10pt,twocolumn]{article}\n"
                "\\usepackage[margin=0.75in]{geometry}\n"
                "\\usepackage{booktabs,amssymb}\n"
                "\\begin{document}\n"
                + tables
                + "\\end{document}\n"
            )
        completed = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "compile_test.tex"],
            cwd=latex_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertNotIn("Overfull \\hbox", completed.stdout)

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
