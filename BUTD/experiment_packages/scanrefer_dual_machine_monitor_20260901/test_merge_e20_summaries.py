import hashlib
import json
import os
import tempfile
import unittest

from merge_e20_summaries import merge


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)
        handle.write("\n")


class MergeE20SummariesTest(unittest.TestCase):
    def make_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        paths = []
        for index, machine in enumerate(("machine35608", "machine50630")):
            metrics = {
                "unique_025": 0.80 + index * 0.01,
                "unique_050": 0.60 + index * 0.01,
                "multiple_025": 0.45 + index * 0.01,
                "multiple_050": 0.34 + index * 0.01,
                "overall_025": 0.50 + index * 0.01,
                "overall_050": 0.38 + index * 0.01,
            }
            payload = {
                "counts": {"multiple": 8089, "total": 9508, "unique": 1419},
                "epoch": 20,
                "eval_log": "/tmp/{}/eval_epoch_20.log".format(machine),
                "eval_log_sha256": "{:064x}".format(index + 1),
                "machine": machine,
                "metrics": metrics,
                "percent": {name: value * 100.0 for name, value in metrics.items()},
                "ready_file": "/tmp/{}/E20_READY".format(machine),
                "ready_timestamp": "2026-09-02T01:0{}:00+08:00".format(index),
            }
            path = os.path.join(temporary.name, "{}.json".format(machine))
            write_json(path, payload)
            paths.append(path)
        return temporary, paths

    def test_merges_exactly_two_machine_summaries(self):
        temporary, paths = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        output_json = os.path.join(temporary.name, "combined.json")
        output_md = os.path.join(temporary.name, "combined.md")

        result = merge(paths, output_json, output_md)

        self.assertEqual(result["status"], "both_e20_complete")
        self.assertEqual(result["rows"]["S2"]["machine"], "machine35608")
        self.assertEqual(result["rows"]["R1"]["machine"], "machine50630")
        self.assertEqual(result["rows"]["R1"]["percent"]["overall_050"], 39.0)
        self.assertTrue(result["provisional_only_no_queue_action"])
        for path in paths:
            with open(path, "rb") as handle:
                expected = hashlib.sha256(handle.read()).hexdigest()
            with open(path, "r") as handle:
                machine = json.load(handle)["machine"]
            self.assertEqual(result["source_summary_sha256"][machine], expected)
        with open(output_md, "r") as handle:
            markdown = handle.read()
        self.assertIn("| S2 | machine35608 | E20 |", markdown)
        self.assertIn("| R1 | machine50630 | E20 |", markdown)
        self.assertIn("provisional", markdown.lower())

    def test_rejects_missing_machine(self):
        temporary, paths = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        with open(paths[1], "r") as handle:
            payload = json.load(handle)
        payload["machine"] = "machine35608"
        write_json(paths[1], payload)

        with self.assertRaisesRegex(ValueError, "expected exactly machines"):
            merge(
                paths,
                os.path.join(temporary.name, "combined.json"),
                os.path.join(temporary.name, "combined.md"),
            )

    def test_rejects_non_e20_or_metric_drift(self):
        temporary, paths = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        with open(paths[0], "r") as handle:
            payload = json.load(handle)
        payload["epoch"] = 15
        write_json(paths[0], payload)
        with self.assertRaisesRegex(ValueError, "not epoch 20"):
            merge(
                paths,
                os.path.join(temporary.name, "combined.json"),
                os.path.join(temporary.name, "combined.md"),
            )

        payload["epoch"] = 20
        payload["percent"]["overall_025"] += 0.1
        write_json(paths[0], payload)
        with self.assertRaisesRegex(ValueError, "percent/metric mismatch"):
            merge(
                paths,
                os.path.join(temporary.name, "combined.json"),
                os.path.join(temporary.name, "combined.md"),
            )


if __name__ == "__main__":
    unittest.main()
