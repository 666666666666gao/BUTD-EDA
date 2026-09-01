import hashlib
import json
import os
import tempfile
import unittest

from collect_completed_row import collect


METRICS = {
    "last__bbs_acc0.25_top1": 0.6,
    "last__bbs_acc0.50_top1": 0.4,
    "last__bbs_unique_acc0.25_top1": 0.6,
    "last__bbs_unique_acc0.50_top1": 0.4,
    "last__bbs_multiple_acc0.25_top1": 0.6,
    "last__bbs_multiple_acc0.50_top1": 0.4,
    "last__bbs_unique_count_acc0.25": 1419.0,
    "last__bbs_unique_count_acc0.50": 1419.0,
    "last__bbs_multiple_count_acc0.25": 8089.0,
    "last__bbs_multiple_count_acc0.50": 8089.0,
}


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def write_eval(path, metrics=None):
    values = METRICS if metrics is None else metrics
    with open(path, "w") as handle:
        for key, value in values.items():
            handle.write("{}: {:.10f}\n".format(key, value))


class CollectCompletedRowTest(unittest.TestCase):
    def make_completed_row(self):
        temporary = tempfile.TemporaryDirectory()
        run = os.path.join(temporary.name, "run")
        os.makedirs(run)
        checkpoint = os.path.join(run, "ckpt_best_primary.pth")
        with open(checkpoint, "wb") as handle:
            handle.write(b"checkpoint fixture")
        with open(checkpoint, "rb") as handle:
            checkpoint_sha = hashlib.sha256(handle.read()).hexdigest()
        best = {
            "checkpoint": checkpoint,
            "comparison": "strict_greater_than",
            "epoch": 20,
            "metric": "last__bbs_acc0.25_top1",
            "min_delta": 0.0,
            "mode": "max",
            "score": 0.6,
        }
        protocol = {
            "rng_seed": 0,
            "batch_size": 24,
            "max_epoch": 65,
            "val_freq": 5,
            "lr": 0.0001,
            "lr_backbone": 0.001,
            "lr_decay_epochs": [55, 60],
        }
        write_json(os.path.join(run, "best_primary.json"), best)
        write_json(os.path.join(run, "config.json"), protocol)
        write_eval(os.path.join(run, "eval_epoch_20.log"))
        write_eval(os.path.join(run, "eval_epoch_last.log"))
        receipt = {
            "row": "S2",
            "run": run,
            "best_primary": best,
            "checkpoint": checkpoint,
            "checkpoint_size": os.path.getsize(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "protocol": protocol,
        }
        receipt_path = os.path.join(temporary.name, "S2.json")
        output_path = os.path.join(temporary.name, "S2_collected.json")
        write_json(receipt_path, receipt)
        return temporary, receipt_path, output_path

    def test_collects_a_complete_strict_best_row(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)

        result = collect(receipt_path, output_path)

        self.assertEqual(result["row"], "S2")
        self.assertEqual(result["best_epoch"], 20)
        self.assertEqual(result["percent"]["overall_025"], 60.0)
        self.assertEqual(result["counts"], {"unique": 1419, "multiple": 8089, "total": 9508})
        self.assertTrue(result["reload_parity"])
        self.assertTrue(os.path.isfile(output_path))

    def test_rejects_checkpoint_sha_mismatch(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)
        with open(receipt_path, "r") as handle:
            receipt = json.load(handle)
        receipt["checkpoint_sha256"] = "0" * 64
        write_json(receipt_path, receipt)

        with self.assertRaisesRegex(ValueError, "checkpoint SHA256 mismatch"):
            collect(receipt_path, output_path)

        self.assertFalse(os.path.exists(output_path))

    def test_rejects_reload_metric_divergence(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)
        with open(receipt_path, "r") as handle:
            receipt = json.load(handle)
        divergent = dict(METRICS)
        divergent["last__bbs_acc0.50_top1"] = 0.39
        write_eval(os.path.join(receipt["run"], "eval_epoch_last.log"), divergent)

        with self.assertRaisesRegex(ValueError, "reload parity failed"):
            collect(receipt_path, output_path)

        self.assertFalse(os.path.exists(output_path))

    def test_rejects_best_primary_score_not_in_eval_log(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)
        with open(receipt_path, "r") as handle:
            receipt = json.load(handle)
        receipt["best_primary"]["score"] = 0.61
        write_json(receipt_path, receipt)
        write_json(
            os.path.join(receipt["run"], "best_primary.json"),
            receipt["best_primary"],
        )

        with self.assertRaisesRegex(ValueError, "best score does not match"):
            collect(receipt_path, output_path)

        self.assertFalse(os.path.exists(output_path))

    def test_rejects_protocol_drift(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)
        with open(receipt_path, "r") as handle:
            receipt = json.load(handle)
        receipt["protocol"]["batch_size"] = 16
        write_json(receipt_path, receipt)
        write_json(os.path.join(receipt["run"], "config.json"), receipt["protocol"])

        with self.assertRaisesRegex(ValueError, "protocol mismatch: batch_size"):
            collect(receipt_path, output_path)

        self.assertFalse(os.path.exists(output_path))

    def test_rejects_wrong_scanrefer_sample_contract(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)
        with open(receipt_path, "r") as handle:
            receipt = json.load(handle)
        wrong = dict(METRICS)
        wrong["last__bbs_unique_count_acc0.25"] = 1418.0
        wrong["last__bbs_unique_count_acc0.50"] = 1418.0
        write_eval(os.path.join(receipt["run"], "eval_epoch_20.log"), wrong)
        write_eval(os.path.join(receipt["run"], "eval_epoch_last.log"), wrong)

        with self.assertRaisesRegex(ValueError, "unexpected sample contract"):
            collect(receipt_path, output_path)

        self.assertFalse(os.path.exists(output_path))

    def test_rejects_an_extra_weight_file(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)
        with open(receipt_path, "r") as handle:
            receipt = json.load(handle)
        with open(os.path.join(receipt["run"], "ckpt_epoch_20.pth"), "wb") as handle:
            handle.write(b"redundant weight")

        with self.assertRaisesRegex(ValueError, "expected exactly one weight"):
            collect(receipt_path, output_path)

        self.assertFalse(os.path.exists(output_path))

    def test_rejects_overall_subset_inconsistency(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)
        with open(receipt_path, "r") as handle:
            receipt = json.load(handle)
        inconsistent = dict(METRICS)
        inconsistent["last__bbs_acc0.50_top1"] = 0.41
        write_eval(os.path.join(receipt["run"], "eval_epoch_20.log"), inconsistent)
        write_eval(os.path.join(receipt["run"], "eval_epoch_last.log"), inconsistent)

        with self.assertRaisesRegex(ValueError, "Overall/subset mismatch"):
            collect(receipt_path, output_path)

        self.assertFalse(os.path.exists(output_path))

    def test_rejects_receipt_best_primary_drift(self):
        temporary, receipt_path, output_path = self.make_completed_row()
        self.addCleanup(temporary.cleanup)
        with open(receipt_path, "r") as handle:
            receipt = json.load(handle)
        receipt["best_primary"]["epoch"] = 25
        write_json(receipt_path, receipt)

        with self.assertRaisesRegex(ValueError, "receipt best_primary differs"):
            collect(receipt_path, output_path)

        self.assertFalse(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
