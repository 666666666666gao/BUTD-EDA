#!/usr/bin/env python3
"""Build and audit a self-contained Stage153 deployable artifact bundle.

The builder is fail-closed: it only runs after the train-only selector has
authorized validation and the fixed validation policy has already met both
strict goals.  Training dumps and GT-derived features are never copied into
the inference bundle.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys

import torch


ALTERNATE_HEAD_KEYS = (
    "module.detector_policy_adapter.rerank_head.0.weight",
    "module.detector_policy_adapter.rerank_head.0.bias",
    "module.detector_policy_adapter.rerank_head.2.weight",
    "module.detector_policy_adapter.rerank_head.2.bias",
)

SELECTOR_RESULT_STAGES = {
    "153_train_only_stage142_stage150_source_selector": (
        "153_source_selector_validation_eval"
    ),
    "154_train_only_scene_oof_stage142_stage150_source_selector": (
        "154_oof_source_selector_validation_eval"
    ),
    "155_train_only_fold_routed_oof_stage142_stage150_selector": (
        "155_fold_routed_oof_selector_validation_eval"
    ),
    "156_train_only_five_fold_mean_oof_stage142_stage150_selector": (
        "156_five_fold_mean_selector_validation_eval"
    ),
    "157_all_train_refit_stage154_fix_break_selector": (
        "157_all_train_refit_selector_validation_eval"
    ),
    "158_regularized_scene_oof_stage142_stage150_source_selector": (
        "158_regularized_selector_validation_eval"
    ),
    "162_stage135c_same_domain_tier3_option_ranker": (
        "162_tier3_option_ranker_validation_eval"
    ),
    "163_stage142_plus_tier3_fixed_residual_option_ranker": (
        "163_tier3_residual_blend_validation_eval"
    ),
    "165_stage164_nested_blend_train_dev_90pct_change_cap": (
        "165_capped_same_domain_trio_validation_eval"
    ),
    "167_train_only_oof_stage154_stage165_meta_selector": (
        "167_stage154_stage165_meta_selector_validation_eval"
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def copy_component(source, output_dir, relative_name):
    assert os.path.isfile(source), source
    destination = os.path.join(output_dir, relative_name)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": relative_name.replace(os.sep, "/"),
        "sha256": sha256(destination),
        "size": os.path.getsize(destination),
    }


def write_json_component(payload, source, output_dir, relative_name):
    destination = os.path.join(output_dir, relative_name)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    atomic_json(destination, payload)
    return {
        "path": relative_name.replace(os.sep, "/"),
        "sha256": sha256(destination),
        "size": os.path.getsize(destination),
        "source_sha256": sha256(source),
    }


def link_primary(source, output_dir):
    destination = os.path.join(output_dir, "weights", "butd_stage153_primary.pth")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        os.link(source, destination)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        mode = "copy"
    return {
        "path": "weights/butd_stage153_primary.pth",
        "sha256": sha256(destination),
        "size": os.path.getsize(destination),
        "storage_mode": mode,
    }


def compare_checkpoints(alternate_path, primary_path):
    alternate = torch.load(alternate_path, map_location="cpu")
    primary = torch.load(primary_path, map_location="cpu")
    assert "model" in alternate and "model" in primary
    alternate_model = alternate["model"]
    primary_model = primary["model"]
    assert list(alternate_model) == list(primary_model)
    changed = [
        name for name in alternate_model
        if not torch.equal(alternate_model[name], primary_model[name])
    ]
    assert tuple(changed) == ALTERNATE_HEAD_KEYS, changed
    tensors = {
        name: alternate_model[name].detach().cpu().clone()
        for name in ALTERNATE_HEAD_KEYS
    }
    audit = {
        "model_tensor_count": len(primary_model),
        "changed_tensor_names": changed,
        "changed_tensor_count": len(changed),
        "changed_parameter_elements": int(sum(t.numel() for t in tensors.values())),
        "all_changes_are_rerank_head": True,
    }
    assert audit["changed_parameter_elements"] == 82564
    return tensors, audit


def validate_scientific_gates(selector_lock, validation_result, selector_lock_path):
    selector_stage = selector_lock["stage"]
    assert selector_stage in SELECTOR_RESULT_STAGES, selector_stage
    assert selector_lock["validation_labels_used_for_selection"] is False
    assert selector_lock["validation_evaluation_authorized"] is True
    assert selector_lock["internal_gate_pass"] is True
    assert validation_result["stage"] == SELECTOR_RESULT_STAGES[selector_stage]
    assert validation_result["validation_labels_used_for_selection"] is False
    assert validation_result["strict_goal_met_offline"] is True
    assert validation_result["policy_lock_sha256"] == sha256(selector_lock_path)
    metrics = validation_result["metrics"]["selected"]
    goals = validation_result["strict_goal_hits"]
    assert int(metrics["hits025"]) >= int(goals["acc025"])
    assert int(metrics["hits050"]) >= int(goals["acc050"])
    return {
        "selector_stage": selector_stage,
        "validation_result_stage": validation_result["stage"],
        "hits025": int(metrics["hits025"]),
        "hits050": int(metrics["hits050"]),
        "acc025": float(metrics["acc025"]),
        "acc050": float(metrics["acc050"]),
        "required_hits025": int(goals["acc025"]),
        "required_hits050": int(goals["acc050"]),
    }


def referenced_model_paths(stage31, stage33, selector):
    result = [
        (stage31["binary_model"], "models/stage31_binary.txt",
         stage31["binary_model_sha256"]),
        (stage31["ordinal_model"], "models/stage31_ordinal.txt",
         stage31["ordinal_model_sha256"]),
        (stage33["model_path"], "models/stage33_pointwise.txt",
         stage33["model_sha256"]),
    ]
    if "models" in selector:
        for index, item in enumerate(selector["models"]):
            result.append((
                item["path"],
                "models/stage153_selector_{:02d}.txt".format(index),
                item["sha256"],
            ))
    else:
        single_model_names = {
            "162_stage135c_same_domain_tier3_option_ranker": (
                "models/stage162_tier3_option_ranker.txt"
            ),
            "163_stage142_plus_tier3_fixed_residual_option_ranker": (
                "models/stage163_tier3_option_ranker.txt"
            ),
        }
        assert selector["stage"] in single_model_names, selector["stage"]
        result.append((
            selector["model_path"],
            single_model_names[selector["stage"]],
            selector["model_sha256"],
        ))
    return result


def build_artifact(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    stage31 = read_json(args.stage31_lock)
    stage33 = read_json(args.stage33_lock)
    stage142 = read_json(args.stage142_lock)
    selector = read_json(args.selector_lock)
    validation = read_json(args.validation_result)
    metrics = validate_scientific_gates(selector, validation, args.selector_lock)

    assert stage142["validation_labels_used_for_selection"] is False
    assert stage142["selection_data_scope"] == "scanrefer_train_scene_hash_dev_only"
    alternate_tensors, tensor_audit = compare_checkpoints(
        args.alternate_checkpoint, args.primary_checkpoint
    )

    staging = args.output_dir + ".building.{}".format(os.getpid())
    assert not os.path.exists(staging), staging
    os.makedirs(staging)
    try:
        primary_item = link_primary(args.primary_checkpoint, staging)
        primary_item["source_sha256"] = sha256(args.primary_checkpoint)
        primary_item["source_path_for_audit"] = os.path.abspath(args.primary_checkpoint)
        assert primary_item["sha256"] == primary_item["source_sha256"]

        alternate_payload = {
            "format": "stage153_alternate_rerank_head_v1",
            "source_checkpoint_sha256": sha256(args.alternate_checkpoint),
            "primary_checkpoint_sha256": primary_item["sha256"],
            "tensor_names": list(ALTERNATE_HEAD_KEYS),
            "tensors": alternate_tensors,
        }
        alternate_relative = "weights/stage135c_alternate_rerank_head.pth"
        alternate_destination = os.path.join(staging, alternate_relative)
        torch.save(alternate_payload, alternate_destination)
        alternate_item = {
            "path": alternate_relative,
            "sha256": sha256(alternate_destination),
            "size": os.path.getsize(alternate_destination),
            "source_checkpoint_sha256": alternate_payload["source_checkpoint_sha256"],
        }

        models = []
        bundled_model_paths = {}
        for source, relative, expected_sha in referenced_model_paths(
            stage31, stage33, selector
        ):
            assert sha256(source) == expected_sha, source
            models.append(copy_component(source, staging, relative))
            bundled_model_paths[os.path.abspath(source)] = os.path.abspath(
                os.path.join(args.output_dir, relative)
            )

        runtime = []
        bundled_runtime_paths = {}
        for source in args.runtime_file:
            basename = os.path.basename(source)
            assert basename not in bundled_runtime_paths, basename
            relative = "runtime/{}".format(basename)
            runtime.append(copy_component(source, staging, relative))
            bundled_runtime_paths[basename] = os.path.abspath(
                os.path.join(args.output_dir, relative)
            )

        stage31_bundle = dict(stage31)
        stage31_bundle["binary_model"] = bundled_model_paths[
            os.path.abspath(stage31["binary_model"])
        ]
        stage31_bundle["ordinal_model"] = bundled_model_paths[
            os.path.abspath(stage31["ordinal_model"])
        ]
        stage33_bundle = dict(stage33)
        stage33_bundle["model_path"] = bundled_model_paths[
            os.path.abspath(stage33["model_path"])
        ]
        stage31_item = write_json_component(
            stage31_bundle, args.stage31_lock, staging, "locks/stage31.json"
        )
        stage33_item = write_json_component(
            stage33_bundle, args.stage33_lock, staging, "locks/stage33.json"
        )

        stage142_bundle = json.loads(json.dumps(stage142))
        if "provenance" in stage142_bundle:
            provenance = stage142_bundle["provenance"]
            provenance["stage31_lock"] = os.path.abspath(
                os.path.join(args.output_dir, "locks/stage31.json")
            )
            provenance["stage31_lock_sha256"] = stage31_item["sha256"]
            provenance["stage33_lock"] = os.path.abspath(
                os.path.join(args.output_dir, "locks/stage33.json")
            )
            provenance["stage33_lock_sha256"] = stage33_item["sha256"]
            source_mapping = {
                "binary": stage31_bundle["binary_model"],
                "ordinal": stage31_bundle["ordinal_model"],
                "pointwise": stage33_bundle["model_path"],
            }
            for name, path in source_mapping.items():
                if name in provenance.get("sources", {}):
                    provenance["sources"][name]["path"] = path
        stage142_script_basename = os.path.basename(
            stage142_bundle.get("script", "")
        )
        if stage142_script_basename in bundled_runtime_paths:
            stage142_bundle["script"] = bundled_runtime_paths[
                stage142_script_basename
            ]
        elif "stage140_train_eval_nested_blend.py" in bundled_runtime_paths:
            stage142_bundle["script"] = bundled_runtime_paths[
                "stage140_train_eval_nested_blend.py"
            ]
        stage142_item = write_json_component(
            stage142_bundle, args.stage142_lock, staging, "locks/stage142.json"
        )

        selector_bundle = json.loads(json.dumps(selector))
        if "models" in selector_bundle:
            for original, bundled in zip(selector_bundle["models"], models[3:]):
                original["path"] = os.path.abspath(
                    os.path.join(args.output_dir, bundled["path"])
                )
        else:
            assert len(models[3:]) == 1
            selector_bundle["model_path"] = os.path.abspath(
                os.path.join(args.output_dir, models[3]["path"])
            )
        selector_script_basename = os.path.basename(selector["script"])
        if selector_script_basename in bundled_runtime_paths:
            selector_bundle["script"] = bundled_runtime_paths[
                selector_script_basename
            ]
        if selector_bundle["stage"] == (
            "165_stage164_nested_blend_train_dev_90pct_change_cap"
        ):
            selector_bundle["nested_policy"] = os.path.abspath(
                os.path.join(args.output_dir, "locks", "stage142.json")
            )
            selector_bundle["nested_policy_sha256"] = stage142_item["sha256"]
        selector_item = write_json_component(
            selector_bundle, args.selector_lock, staging, "locks/stage153.json"
        )
        locks = {
            "stage31": stage31_item,
            "stage33": stage33_item,
            "stage142": stage142_item,
            "stage153": selector_item,
            "validation_result": copy_component(
                args.validation_result, staging, "evidence/validation_result.json"
            ),
        }

        manifest = {
            "format": "butd_stage153_deployable_bundle_v1",
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "no_ground_truth_required_at_inference": True,
            "fixed_protocol": True,
            "selector_stage": selector["stage"],
            "validation_metrics": metrics,
            "primary_checkpoint": primary_item,
            "alternate_rerank_head": alternate_item,
            "tensor_identity_audit": tensor_audit,
            "locks": locks,
            "models": models,
            "runtime_files": runtime,
            "training_dumps_packaged": False,
            "formal_status": "offline_goal_met_pending_fresh_bundle_inference_reload",
        }
        manifest_path = os.path.join(staging, "manifest.json")
        atomic_json(manifest_path, manifest)
        manifest_sha = sha256(manifest_path)
        atomic_json(os.path.join(staging, "build_receipt.json"), {
            "manifest_sha256": manifest_sha,
            "artifact_format": manifest["format"],
            "validation_metrics": metrics,
            "fresh_bundle_inference_reload_required": True,
        })
        os.replace(staging, args.output_dir)
    except Exception:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        raise
    validate_artifact(args.output_dir)
    return os.path.join(args.output_dir, "manifest.json")


def safe_resolve(root, relative):
    root = os.path.abspath(root)
    path = os.path.abspath(os.path.join(root, relative))
    assert os.path.commonpath([root, path]) == root, relative
    return path


def validate_artifact(root):
    manifest_path = os.path.join(root, "manifest.json")
    manifest = read_json(manifest_path)
    assert manifest["format"] == "butd_stage153_deployable_bundle_v1"
    assert manifest["no_ground_truth_required_at_inference"] is True
    assert manifest["training_dumps_packaged"] is False
    items = [manifest["primary_checkpoint"], manifest["alternate_rerank_head"]]
    items.extend(manifest["locks"].values())
    items.extend(manifest["models"])
    items.extend(manifest["runtime_files"])
    if "fresh_bundle_reload" in manifest:
        items.append(manifest["fresh_bundle_reload"]["validation_result"])
        items.append(manifest["fresh_bundle_reload"]["receipt"])
    for item in items:
        path = safe_resolve(root, item["path"])
        assert os.path.isfile(path), path
        assert sha256(path) == item["sha256"], path

    primary = torch.load(
        safe_resolve(root, manifest["primary_checkpoint"]["path"]),
        map_location="cpu",
    )
    alternate = torch.load(
        safe_resolve(root, manifest["alternate_rerank_head"]["path"]),
        map_location="cpu",
    )
    assert alternate["format"] == "stage153_alternate_rerank_head_v1"
    assert tuple(alternate["tensor_names"]) == ALTERNATE_HEAD_KEYS
    assert set(alternate["tensors"]) == set(ALTERNATE_HEAD_KEYS)
    for name in ALTERNATE_HEAD_KEYS:
        assert name in primary["model"]
        assert alternate["tensors"][name].shape == primary["model"][name].shape
        assert alternate["tensors"][name].dtype == primary["model"][name].dtype
    receipt = read_json(os.path.join(root, "build_receipt.json"))
    assert receipt["manifest_sha256"] == sha256(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("alternate_checkpoint")
    build.add_argument("primary_checkpoint")
    build.add_argument("stage31_lock")
    build.add_argument("stage33_lock")
    build.add_argument("stage142_lock")
    build.add_argument("selector_lock")
    build.add_argument("validation_result")
    build.add_argument("output_dir")
    build.add_argument("--runtime-file", action="append", default=[])
    validate = subparsers.add_parser("validate")
    validate.add_argument("artifact_dir")
    args = parser.parse_args()
    if args.command == "build":
        path = build_artifact(args)
        print(json.dumps({"manifest": path, "sha256": sha256(path)}, indent=2))
    else:
        manifest = validate_artifact(args.artifact_dir)
        print(json.dumps({
            "artifact": os.path.abspath(args.artifact_dir),
            "format": manifest["format"],
            "validation_metrics": manifest["validation_metrics"],
        }, indent=2))


if __name__ == "__main__":
    main()
