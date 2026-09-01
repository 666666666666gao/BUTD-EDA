import hashlib
import json
import os
import tempfile
import unittest

from finalize_dual_machine_tables import finalize
import test_assemble_final_tables as assemble_test_helpers


MACHINE_ROWS = {
    "machine35608": ("S2", "S0", "S3"),
    "machine50630": ("R1", "R3", "R0", "R2"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


class FinalizeDualMachineTablesTest(unittest.TestCase):
    def make_fixture(self):
        base = assemble_test_helpers.AssembleFinalTablesTest()
        temporary, manifest, row_paths = base.make_fixture()
        by_row = {}
        for path in row_paths:
            with open(path, "r") as handle:
                by_row[json.load(handle)["row"]] = path
        markers = []
        for machine, rows in MACHINE_ROWS.items():
            marker = os.path.join(temporary.name, "{}_READY".format(machine))
            with open(marker, "w") as handle:
                handle.write("timestamp=2026-09-07T12:00:00+08:00\n")
                handle.write("machine={}\n".format(machine))
                handle.write("row_count={}\n".format(len(rows)))
                for row in rows:
                    handle.write(
                        "{}  /remote/control/audited_rows/{}.json\n".format(
                            sha256(by_row[row]), row
                        )
                    )
            markers.append(marker)
        return temporary, manifest, row_paths, markers

    def test_verifies_two_manifests_and_builds_complete_output_directory(self):
        temporary, manifest, rows, markers = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        output = os.path.join(temporary.name, "final_bundle")

        result = finalize(manifest, markers, rows, output)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["machine_rows"], MACHINE_ROWS)
        self.assertTrue(result["main_monotonicity"]["overall_025_non_decreasing"])
        self.assertIn("internal_claim_summary", result)
        self.assertTrue(
            result["internal_claim_summary"][
                "sacr_all_variants_full_not_lower_overall_025"
            ]
        )
        for name in (
            "final_tables.json",
            "final_tables.md",
            "FINAL_TABLES_RECEIPT.json",
            os.path.join("latex", "TABLE_3_main_modules.tex"),
            os.path.join("latex", "TABLE_4_sacr_internal.tex"),
            os.path.join("latex", "TABLE_5_rapf_internal.tex"),
            os.path.join("latex", "TABLES_scanrefer_ablations_all.tex"),
        ):
            self.assertTrue(os.path.isfile(os.path.join(output, name)), name)
        with open(os.path.join(output, "FINAL_TABLES_RECEIPT.json"), "r") as handle:
            receipt = json.load(handle)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(set(receipt["machine_manifest_sha256"]), set(MACHINE_ROWS))
        self.assertEqual(len(receipt["row_json_sha256"]), 7)

    def test_rejects_marker_sha_mismatch_without_partial_output(self):
        temporary, manifest, rows, markers = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        output = os.path.join(temporary.name, "final_bundle")
        with open(markers[0], "r") as handle:
            marker = handle.read()
        marker = marker.replace(marker.splitlines()[3].split()[0], "0" * 64, 1)
        with open(markers[0], "w") as handle:
            handle.write(marker)

        with self.assertRaisesRegex(ValueError, "declared SHA256 mismatch"):
            finalize(manifest, markers, rows, output)
        self.assertFalse(os.path.exists(output))

    def test_rejects_wrong_machine_assignment(self):
        temporary, manifest, rows, markers = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        output = os.path.join(temporary.name, "final_bundle")
        with open(markers[0], "r") as handle:
            marker = handle.read()
        marker = marker.replace("/S2.json", "/R1.json", 1)
        with open(markers[0], "w") as handle:
            handle.write(marker)

        with self.assertRaisesRegex(ValueError, "row assignment mismatch"):
            finalize(manifest, markers, rows, output)
        self.assertFalse(os.path.exists(output))

    def test_refuses_to_overwrite_existing_output_directory(self):
        temporary, manifest, rows, markers = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        output = os.path.join(temporary.name, "final_bundle")
        os.makedirs(output)

        with self.assertRaisesRegex(ValueError, "output directory already exists"):
            finalize(manifest, markers, rows, output)


if __name__ == "__main__":
    unittest.main()
