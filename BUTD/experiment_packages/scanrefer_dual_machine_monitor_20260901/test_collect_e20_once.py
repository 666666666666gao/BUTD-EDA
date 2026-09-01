import hashlib
import json
import os
import tempfile
import unittest

from collect_e20_once import collect


METRICS = {
    "last__bbs_acc0.25_top1": 0.53,
    "last__bbs_acc0.50_top1": 0.40,
    "last__bbs_unique_acc0.25_top1": 0.84,
    "last__bbs_unique_acc0.50_top1": 0.65,
    "last__bbs_multiple_acc0.25_top1": 0.48,
    "last__bbs_multiple_acc0.50_top1": 0.35,
    "last__bbs_unique_count_acc0.25": 1419,
    "last__bbs_unique_count_acc0.50": 1419,
    "last__bbs_multiple_count_acc0.25": 8089,
    "last__bbs_multiple_count_acc0.50": 8089,
}


class CollectE20OnceTest(unittest.TestCase):
    def make_fixture(self, remove_key=None, wrong_sha=False):
        root = tempfile.TemporaryDirectory()
        log = os.path.join(root.name, "eval_epoch_20.log")
        with open(log, "w") as handle:
            for key, value in METRICS.items():
                if key != remove_key:
                    handle.write("{}: {}\n".format(key, value))
        with open(log, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if wrong_sha:
            digest = "0" * 64
        ready = os.path.join(root.name, "E20_READY")
        with open(ready, "w") as handle:
            handle.write("timestamp=2026-09-02T01:00:00+08:00\n")
            handle.write("machine=machine35608\n")
            handle.write("milestone={}\n".format(log))
            handle.write("required_metric_keys=10\n")
            handle.write("{}  {}\n".format(digest, log))
        return root, ready

    def test_collects_and_validates_complete_e20(self):
        root, ready = self.make_fixture()
        self.addCleanup(root.cleanup)
        output = os.path.join(root.name, "summary.json")
        result = collect(ready, output)
        self.assertEqual(result["machine"], "machine35608")
        self.assertEqual(result["epoch"], 20)
        self.assertEqual(result["counts"], {"multiple": 8089, "total": 9508, "unique": 1419})
        self.assertAlmostEqual(result["metrics"]["overall_025"], 0.53)
        self.assertAlmostEqual(result["percent"]["overall_050"], 40.0)
        with open(output, "r") as handle:
            self.assertEqual(json.load(handle), result)

    def test_rejects_sha_mismatch(self):
        root, ready = self.make_fixture(wrong_sha=True)
        self.addCleanup(root.cleanup)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            collect(ready, os.path.join(root.name, "summary.json"))

    def test_rejects_incomplete_metric_set(self):
        root, ready = self.make_fixture(remove_key="last__bbs_multiple_acc0.50_top1")
        self.addCleanup(root.cleanup)
        with self.assertRaisesRegex(ValueError, "missing metric keys"):
            collect(ready, os.path.join(root.name, "summary.json"))


if __name__ == "__main__":
    unittest.main()
