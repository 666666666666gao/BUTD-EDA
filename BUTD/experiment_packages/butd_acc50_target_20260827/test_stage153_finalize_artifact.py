#!/usr/bin/env python3
import json
import os
import tempfile
import unittest

import stage153_build_final_artifact as artifact
import stage153_finalize_artifact as finalizer
from test_stage153_build_final_artifact import ArtifactBuilderTest


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class FinalizerTest(unittest.TestCase):
    def fixture(self, root, selected_hits=(5200, 4040)):
        args = ArtifactBuilderTest().fixture(root)
        artifact.build_artifact(args)
        fresh = artifact.read_json(args.validation_result)
        fresh["metrics"]["selected"]["hits025"] = selected_hits[0]
        fresh["metrics"]["selected"]["hits050"] = selected_hits[1]
        fresh_path = os.path.join(root, "fresh.json")
        write_json(fresh_path, fresh)
        receipt_path = os.path.join(root, "receipt.json")
        write_json(receipt_path, {
            "stage": "stage153d_fresh_bundle_reload",
            "no_ground_truth_used_for_inference": True,
            "primary_stage150_hits": [5234, 3979],
            "alternate_stage135c_hits": [5215, 3956],
            "selected_hits": list(selected_hits),
            "evaluator_restored_sha256": (
                "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
            ),
            "locked_validation_result_sha256": artifact.sha256(fresh_path),
        })
        return args, fresh_path, receipt_path

    def test_finalize_success(self):
        with tempfile.TemporaryDirectory() as root:
            args, fresh, receipt = self.fixture(root)
            manifest = finalizer.finalize(args.output_dir, fresh, receipt)
            self.assertEqual(
                manifest["formal_status"],
                "complete_fresh_bundle_inference_reload",
            )
            self.assertTrue(manifest["fresh_bundle_reload"]["strict_goal_met"])

    def test_rejects_metric_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            args, fresh, receipt = self.fixture(root, selected_hits=(5199, 4040))
            with self.assertRaises(AssertionError):
                finalizer.finalize(args.output_dir, fresh, receipt)

    def test_accepts_stage158_fresh_reload_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            args, fresh, receipt = self.fixture(root)
            payload = artifact.read_json(receipt)
            payload["stage"] = "stage158d_fresh_bundle_reload"
            write_json(receipt, payload)
            manifest = finalizer.finalize(args.output_dir, fresh, receipt)
            self.assertEqual(
                manifest["formal_status"],
                "complete_fresh_bundle_inference_reload",
            )

    def test_accepts_stage162_fresh_reload_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            args, fresh, receipt = self.fixture(root)
            payload = artifact.read_json(receipt)
            payload["stage"] = "stage162d_fresh_bundle_reload"
            payload["primary_stage150_hits"] = None
            payload["primary_inference_required_for_deployment"] = False
            write_json(receipt, payload)
            manifest = finalizer.finalize(args.output_dir, fresh, receipt)
            self.assertEqual(
                manifest["formal_status"],
                "complete_fresh_bundle_inference_reload",
            )

    def test_accepts_stage163_fresh_reload_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            args, fresh, receipt = self.fixture(root)
            payload = artifact.read_json(receipt)
            payload["stage"] = "stage163d_fresh_bundle_reload"
            payload["primary_stage150_hits"] = None
            payload["primary_inference_required_for_deployment"] = False
            write_json(receipt, payload)
            manifest = finalizer.finalize(args.output_dir, fresh, receipt)
            self.assertEqual(
                manifest["formal_status"],
                "complete_fresh_bundle_inference_reload",
            )

    def test_accepts_stage165_fresh_reload_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            args, fresh, receipt = self.fixture(root)
            payload = artifact.read_json(receipt)
            payload["stage"] = "stage165d_fresh_bundle_reload"
            payload["primary_stage150_hits"] = None
            payload["primary_inference_required_for_deployment"] = False
            write_json(receipt, payload)
            manifest = finalizer.finalize(args.output_dir, fresh, receipt)
            self.assertEqual(
                manifest["formal_status"],
                "complete_fresh_bundle_inference_reload",
            )

    def test_accepts_stage167_fresh_reload_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            args, fresh, receipt = self.fixture(root)
            manifest_path = os.path.join(args.output_dir, "manifest.json")
            manifest = artifact.read_json(manifest_path)
            manifest["selector_stage"] = (
                "167_train_only_oof_stage154_stage165_meta_selector"
            )
            write_json(manifest_path, manifest)
            build_receipt_path = os.path.join(
                args.output_dir, "build_receipt.json"
            )
            build_receipt = artifact.read_json(build_receipt_path)
            build_receipt["manifest_sha256"] = artifact.sha256(manifest_path)
            write_json(build_receipt_path, build_receipt)
            fresh_payload = artifact.read_json(fresh)
            fresh_payload["stage"] = (
                "167_stage154_stage165_meta_selector_validation_eval"
            )
            write_json(fresh, fresh_payload)
            payload = artifact.read_json(receipt)
            payload["stage"] = "stage167d_fresh_bundle_reload"
            payload["primary_inference_required_for_deployment"] = True
            payload["locked_validation_result_sha256"] = artifact.sha256(fresh)
            write_json(receipt, payload)
            manifest = finalizer.finalize(args.output_dir, fresh, receipt)
            self.assertEqual(
                manifest["formal_status"],
                "complete_fresh_bundle_inference_reload",
            )


if __name__ == "__main__":
    unittest.main()
