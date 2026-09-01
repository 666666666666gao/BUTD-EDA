#!/usr/bin/env python3
import json
import os
import tempfile
import types
import unittest

import torch

import stage153_build_final_artifact as base
import stage167_build_final_artifact as artifact


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def write_text(root, name):
    path = os.path.join(root, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(name + "\n")
    return path


def ranker_locks(root, prefix):
    binary = write_text(root, prefix + "_binary.txt")
    ordinal = write_text(root, prefix + "_ordinal.txt")
    pointwise = write_text(root, prefix + "_pointwise.txt")
    stage31 = os.path.join(root, prefix + "_stage31.json")
    stage33 = os.path.join(root, prefix + "_stage33.json")
    write_json(stage31, {
        "binary_model": binary,
        "binary_model_sha256": base.sha256(binary),
        "ordinal_model": ordinal,
        "ordinal_model_sha256": base.sha256(ordinal),
    })
    write_json(stage33, {
        "model_path": pointwise,
        "model_sha256": base.sha256(pointwise),
    })
    return stage31, stage33


def nested_lock(root, name, stage31, stage33, script):
    s31 = base.read_json(stage31)
    s33 = base.read_json(stage33)
    path = os.path.join(root, name + ".json")
    write_json(path, {
        "stage": name,
        "script": script,
        "validation_labels_used_for_selection": False,
        "selection_data_scope": "scanrefer_train_scene_hash_dev_only",
        "provenance": {
            "stage31_lock": stage31,
            "stage31_lock_sha256": base.sha256(stage31),
            "stage33_lock": stage33,
            "stage33_lock_sha256": base.sha256(stage33),
            "sources": {
                "binary": {"path": s31["binary_model"],
                           "sha256": s31["binary_model_sha256"]},
                "ordinal": {"path": s31["ordinal_model"],
                            "sha256": s31["ordinal_model_sha256"]},
                "pointwise": {"path": s33["model_path"],
                              "sha256": s33["model_sha256"]},
            },
        },
    })
    return path


class Stage167ArtifactTest(unittest.TestCase):
    def fixture(self, root):
        runtime = write_text(root, "runtime.py")
        original31, original33 = ranker_locks(root, "original")
        stage142 = nested_lock(
            root, "stage142", original31, original33, runtime
        )
        selector154_model = write_text(root, "stage154_selector.txt")
        stage154 = os.path.join(root, "stage154.json")
        stage142_payload = base.read_json(stage142)
        write_json(stage154, {
            "stage": "154_train_only_scene_oof_stage142_stage150_source_selector",
            "script": runtime,
            "validation_labels_used_for_selection": False,
            "stage142_lock": stage142,
            "stage142_lock_sha256": base.sha256(stage142),
            "models": [{
                "path": selector154_model,
                "sha256": base.sha256(selector154_model),
            }],
            "provenance": stage142_payload["provenance"],
        })

        stage164_31, stage164_33 = ranker_locks(root, "stage164")
        stage164_parent = nested_lock(
            root, "stage164_parent", stage164_31, stage164_33, runtime
        )
        stage165 = os.path.join(root, "stage165.json")
        stage165_payload = base.read_json(stage164_parent)
        stage165_payload.update({
            "stage": "165_stage164_nested_blend_train_dev_90pct_change_cap",
            "parent_policy": stage164_parent,
            "parent_policy_sha256": base.sha256(stage164_parent),
        })
        write_json(stage165, stage165_payload)

        selector167_model = write_text(root, "stage167_selector.txt")
        selector167 = os.path.join(root, "stage167.json")
        write_json(selector167, {
            "stage": "167_train_only_oof_stage154_stage165_meta_selector",
            "script": runtime,
            "validation_labels_used_for_selection": False,
            "validation_evaluation_authorized": True,
            "internal_gate_pass": True,
            "stage154_lock": stage154,
            "stage154_lock_sha256": base.sha256(stage154),
            "stage165_policy": stage165,
            "stage165_policy_sha256": base.sha256(stage165),
            "stage31_lock_sha256": base.sha256(original31),
            "stage33_lock_sha256": base.sha256(original33),
            "stage142_lock_sha256": base.sha256(stage142),
            "models": [{
                "path": selector167_model,
                "sha256": base.sha256(selector167_model),
            }],
        })
        validation = os.path.join(root, "validation.json")
        write_json(validation, {
            "stage": "167_stage154_stage165_meta_selector_validation_eval",
            "validation_labels_used_for_selection": False,
            "strict_goal_met_offline": True,
            "policy_lock_sha256": base.sha256(selector167),
            "strict_goal_hits": {"acc025": 5126, "acc050": 4033},
            "metrics": {"selected": {
                "hits025": 5200, "hits050": 4040,
                "acc025": 5200 / 9508, "acc050": 4040 / 9508,
            }},
        })

        model = {"shared.weight": torch.ones(2)}
        shapes = ((81920,), (128,), (512,), (4,))
        for name, shape in zip(base.ALTERNATE_HEAD_KEYS, shapes):
            model[name] = torch.zeros(shape)
        primary = os.path.join(root, "primary.pth")
        alternate = os.path.join(root, "alternate.pth")
        torch.save({"model": {k: v.clone() for k, v in model.items()}}, primary)
        changed = {k: v.clone() for k, v in model.items()}
        for name in base.ALTERNATE_HEAD_KEYS:
            changed[name].add_(1.0)
        torch.save({"model": changed}, alternate)

        return types.SimpleNamespace(
            alternate_checkpoint=alternate,
            primary_checkpoint=primary,
            stage31_lock=original31,
            stage33_lock=original33,
            stage142_lock=stage142,
            stage154_lock=stage154,
            stage164_stage31_lock=stage164_31,
            stage164_stage33_lock=stage164_33,
            stage164_parent_lock=stage164_parent,
            stage165_lock=stage165,
            selector_lock=selector167,
            validation_result=validation,
            output_dir=os.path.join(root, "artifact"),
            runtime_file=[runtime],
        )

    def test_packages_both_dependency_stacks_with_internal_paths(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.fixture(root)
            manifest_path = artifact.build_artifact(args)
            manifest = base.validate_artifact(args.output_dir)
            self.assertEqual(
                manifest["selector_stage"],
                "167_train_only_oof_stage154_stage165_meta_selector",
            )
            self.assertEqual(len(manifest["models"]), 8)
            for lock_name in (
                "stage31", "stage33", "stage142", "stage154",
                "stage164_stage31", "stage164_stage33",
                "stage164_parent", "stage165", "stage167",
            ):
                self.assertIn(lock_name, manifest["locks"])
            for item in manifest["locks"].values():
                self.assertTrue(os.path.isfile(os.path.join(
                    args.output_dir, item["path"]
                )))
            stage167 = base.read_json(os.path.join(
                args.output_dir, "locks", "stage167.json"
            ))
            for key in ("stage154_lock", "stage165_policy"):
                self.assertEqual(
                    os.path.commonpath([args.output_dir, stage167[key]]),
                    args.output_dir,
                )
            self.assertTrue(os.path.isfile(manifest_path))


if __name__ == "__main__":
    unittest.main()
