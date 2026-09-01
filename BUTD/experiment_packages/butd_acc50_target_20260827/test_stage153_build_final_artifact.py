#!/usr/bin/env python3
import json
import os
import tempfile
import types
import unittest

import torch

import stage153_build_final_artifact as artifact


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)


class ArtifactBuilderTest(unittest.TestCase):
    def fixture(self, root, goal_met=True, unexpected_change=False):
        os.makedirs(root, exist_ok=True)
        base_model = {"shared.weight": torch.ones(2)}
        shapes = ((81920,), (128,), (512,), (4,))
        for name, shape in zip(artifact.ALTERNATE_HEAD_KEYS, shapes):
            base_model[name] = torch.zeros(shape)
        primary_model = {name: value.clone() for name, value in base_model.items()}
        alternate_model = {name: value.clone() for name, value in base_model.items()}
        for name in artifact.ALTERNATE_HEAD_KEYS:
            alternate_model[name].add_(1.0)
        if unexpected_change:
            alternate_model["shared.weight"].add_(1.0)
        alternate_ckpt = os.path.join(root, "alternate.pth")
        primary_ckpt = os.path.join(root, "primary.pth")
        torch.save({"model": alternate_model}, alternate_ckpt)
        torch.save({"model": primary_model}, primary_ckpt)

        files = {}
        for name in ("binary", "ordinal", "pointwise", "selector", "runtime"):
            path = os.path.join(root, name + ".txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(name + "\n")
            files[name] = path
        stage31_path = os.path.join(root, "stage31.json")
        stage33_path = os.path.join(root, "stage33.json")
        stage142_path = os.path.join(root, "stage142.json")
        selector_path = os.path.join(root, "selector.json")
        validation_path = os.path.join(root, "validation.json")
        write_json(stage31_path, {
            "binary_model": files["binary"],
            "binary_model_sha256": artifact.sha256(files["binary"]),
            "ordinal_model": files["ordinal"],
            "ordinal_model_sha256": artifact.sha256(files["ordinal"]),
        })
        write_json(stage33_path, {
            "model_path": files["pointwise"],
            "model_sha256": artifact.sha256(files["pointwise"]),
        })
        write_json(stage142_path, {
            "validation_labels_used_for_selection": False,
            "selection_data_scope": "scanrefer_train_scene_hash_dev_only",
        })
        write_json(selector_path, {
            "stage": "153_train_only_stage142_stage150_source_selector",
            "script": files["runtime"],
            "validation_labels_used_for_selection": False,
            "validation_evaluation_authorized": True,
            "internal_gate_pass": True,
            "models": [{
                "path": files["selector"],
                "sha256": artifact.sha256(files["selector"]),
            }],
        })
        write_json(validation_path, {
            "stage": "153_source_selector_validation_eval",
            "validation_labels_used_for_selection": False,
            "strict_goal_met_offline": goal_met,
            "policy_lock_sha256": artifact.sha256(selector_path),
            "strict_goal_hits": {"acc025": 5126, "acc050": 4033},
            "metrics": {"selected": {
                "hits025": 5200, "hits050": 4040,
                "acc025": 5200 / 9508, "acc050": 4040 / 9508,
            }},
        })
        return types.SimpleNamespace(
            alternate_checkpoint=alternate_ckpt,
            primary_checkpoint=primary_ckpt,
            stage31_lock=stage31_path,
            stage33_lock=stage33_path,
            stage142_lock=stage142_path,
            selector_lock=selector_path,
            validation_result=validation_path,
            output_dir=os.path.join(root, "artifact"),
            runtime_file=[files["runtime"]],
        )

    def test_build_and_validate_success(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            manifest_path = artifact.build_artifact(args)
            self.assertTrue(os.path.isfile(manifest_path))
            manifest = artifact.validate_artifact(args.output_dir)
            self.assertEqual(
                manifest["tensor_identity_audit"]["changed_parameter_elements"],
                82564,
            )
            self.assertFalse(manifest["training_dumps_packaged"])
            self.assertEqual(manifest["formal_status"],
                             "offline_goal_met_pending_fresh_bundle_inference_reload")
            stage31 = artifact.read_json(os.path.join(
                args.output_dir, "locks", "stage31.json"
            ))
            stage33 = artifact.read_json(os.path.join(
                args.output_dir, "locks", "stage33.json"
            ))
            selector = artifact.read_json(os.path.join(
                args.output_dir, "locks", "stage153.json"
            ))
            bundled_paths = [
                stage31["binary_model"], stage31["ordinal_model"],
                stage33["model_path"], selector["models"][0]["path"],
            ]
            for path in bundled_paths:
                self.assertTrue(os.path.isfile(path), path)
                self.assertEqual(
                    os.path.commonpath([args.output_dir, path]), args.output_dir
                )

    def test_rejects_offline_goal_failure(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root, goal_met=False)
            with self.assertRaises(AssertionError):
                artifact.build_artifact(args)

    def test_rejects_unexpected_weight_change(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root, unexpected_change=True)
            with self.assertRaises(AssertionError):
                artifact.build_artifact(args)

    def test_accepts_preregistered_stage154_identity(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            selector = artifact.read_json(args.selector_lock)
            selector["stage"] = (
                "154_train_only_scene_oof_stage142_stage150_source_selector"
            )
            write_json(args.selector_lock, selector)
            validation = artifact.read_json(args.validation_result)
            validation["stage"] = "154_oof_source_selector_validation_eval"
            validation["policy_lock_sha256"] = artifact.sha256(args.selector_lock)
            write_json(args.validation_result, validation)
            manifest_path = artifact.build_artifact(args)
            manifest = artifact.read_json(manifest_path)
            self.assertEqual(manifest["selector_stage"], selector["stage"])

    def test_accepts_preregistered_stage155_identity(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            selector = artifact.read_json(args.selector_lock)
            selector["stage"] = (
                "155_train_only_fold_routed_oof_stage142_stage150_selector"
            )
            write_json(args.selector_lock, selector)
            validation = artifact.read_json(args.validation_result)
            validation["stage"] = (
                "155_fold_routed_oof_selector_validation_eval"
            )
            validation["policy_lock_sha256"] = artifact.sha256(
                args.selector_lock
            )
            write_json(args.validation_result, validation)
            manifest_path = artifact.build_artifact(args)
            manifest = artifact.read_json(manifest_path)
            self.assertEqual(manifest["selector_stage"], selector["stage"])

    def test_accepts_preregistered_stage156_identity(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            selector = artifact.read_json(args.selector_lock)
            selector["stage"] = (
                "156_train_only_five_fold_mean_oof_stage142_stage150_selector"
            )
            write_json(args.selector_lock, selector)
            validation = artifact.read_json(args.validation_result)
            validation["stage"] = (
                "156_five_fold_mean_selector_validation_eval"
            )
            validation["policy_lock_sha256"] = artifact.sha256(
                args.selector_lock
            )
            write_json(args.validation_result, validation)
            manifest_path = artifact.build_artifact(args)
            manifest = artifact.read_json(manifest_path)
            self.assertEqual(manifest["selector_stage"], selector["stage"])

    def test_accepts_preregistered_stage157_identity(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            selector = artifact.read_json(args.selector_lock)
            selector["stage"] = "157_all_train_refit_stage154_fix_break_selector"
            write_json(args.selector_lock, selector)
            validation = artifact.read_json(args.validation_result)
            validation["stage"] = (
                "157_all_train_refit_selector_validation_eval"
            )
            validation["policy_lock_sha256"] = artifact.sha256(
                args.selector_lock
            )
            write_json(args.validation_result, validation)
            manifest_path = artifact.build_artifact(args)
            manifest = artifact.read_json(manifest_path)
            self.assertEqual(manifest["selector_stage"], selector["stage"])

    def test_accepts_regularized_stage158_identity(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            selector = artifact.read_json(args.selector_lock)
            selector["stage"] = (
                "158_regularized_scene_oof_stage142_stage150_source_selector"
            )
            write_json(args.selector_lock, selector)
            validation = artifact.read_json(args.validation_result)
            validation["stage"] = "158_regularized_selector_validation_eval"
            validation["policy_lock_sha256"] = artifact.sha256(
                args.selector_lock
            )
            write_json(args.validation_result, validation)
            manifest_path = artifact.build_artifact(args)
            manifest = artifact.read_json(manifest_path)
            self.assertEqual(manifest["selector_stage"], selector["stage"])

    def test_accepts_stage162_single_option_ranker_schema(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            selector = artifact.read_json(args.selector_lock)
            model = selector.pop("models")[0]
            selector["stage"] = "162_stage135c_same_domain_tier3_option_ranker"
            selector["model_path"] = model["path"]
            selector["model_sha256"] = model["sha256"]
            write_json(args.selector_lock, selector)
            validation = artifact.read_json(args.validation_result)
            validation["stage"] = "162_tier3_option_ranker_validation_eval"
            validation["policy_lock_sha256"] = artifact.sha256(
                args.selector_lock
            )
            write_json(args.validation_result, validation)
            manifest_path = artifact.build_artifact(args)
            manifest = artifact.read_json(manifest_path)
            bundled = artifact.read_json(os.path.join(
                args.output_dir, "locks", "stage153.json"
            ))
            self.assertEqual(manifest["selector_stage"], selector["stage"])
            self.assertTrue(bundled["model_path"].startswith(args.output_dir))
            self.assertEqual(
                artifact.sha256(bundled["model_path"]),
                selector["model_sha256"],
            )

    def test_accepts_stage163_residual_option_ranker_schema(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            selector = artifact.read_json(args.selector_lock)
            model = selector.pop("models")[0]
            selector["stage"] = (
                "163_stage142_plus_tier3_fixed_residual_option_ranker"
            )
            selector["model_path"] = model["path"]
            selector["model_sha256"] = model["sha256"]
            write_json(args.selector_lock, selector)
            validation = artifact.read_json(args.validation_result)
            validation["stage"] = (
                "163_tier3_residual_blend_validation_eval"
            )
            validation["policy_lock_sha256"] = artifact.sha256(
                args.selector_lock
            )
            write_json(args.validation_result, validation)
            manifest_path = artifact.build_artifact(args)
            manifest = artifact.read_json(manifest_path)
            bundled = artifact.read_json(os.path.join(
                args.output_dir, "locks", "stage153.json"
            ))
            self.assertEqual(manifest["selector_stage"], selector["stage"])
            self.assertTrue(bundled["model_path"].startswith(args.output_dir))
            self.assertEqual(
                artifact.sha256(bundled["model_path"]),
                selector["model_sha256"],
            )
            self.assertTrue(any(
                item["path"] == "models/stage163_tier3_option_ranker.txt"
                for item in manifest["models"]
            ))

    def test_accepts_stage165_no_extra_model_and_rewrites_nested_policy(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            selector = artifact.read_json(args.selector_lock)
            selector["stage"] = (
                "165_stage164_nested_blend_train_dev_90pct_change_cap"
            )
            selector["models"] = []
            selector["nested_policy"] = os.path.abspath(args.stage142_lock)
            selector["nested_policy_sha256"] = artifact.sha256(
                args.stage142_lock
            )
            write_json(args.selector_lock, selector)
            validation = artifact.read_json(args.validation_result)
            validation["stage"] = (
                "165_capped_same_domain_trio_validation_eval"
            )
            validation["policy_lock_sha256"] = artifact.sha256(
                args.selector_lock
            )
            write_json(args.validation_result, validation)
            manifest_path = artifact.build_artifact(args)
            manifest = artifact.read_json(manifest_path)
            bundled_selector = artifact.read_json(os.path.join(
                args.output_dir, "locks", "stage153.json"
            ))
            bundled_policy = os.path.join(
                args.output_dir, "locks", "stage142.json"
            )
            self.assertEqual(manifest["selector_stage"], selector["stage"])
            self.assertEqual(len(manifest["models"]), 3)
            self.assertEqual(
                bundled_selector["nested_policy"], bundled_policy
            )
            self.assertEqual(
                bundled_selector["nested_policy_sha256"],
                artifact.sha256(bundled_policy),
            )


if __name__ == "__main__":
    unittest.main()
