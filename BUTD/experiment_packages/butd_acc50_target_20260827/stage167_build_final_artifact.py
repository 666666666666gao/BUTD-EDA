#!/usr/bin/env python3
"""Build a self-contained deployable bundle for the Stage167 policy stack."""

import argparse
import datetime
import json
import os
import shutil

import torch

import stage153_build_final_artifact as base


STAGE = "167_train_only_oof_stage154_stage165_meta_selector"
RESULT_STAGE = "167_stage154_stage165_meta_selector_validation_eval"


def absolute_bundle_path(output_dir, relative):
    return os.path.abspath(os.path.join(output_dir, relative))


def rewrite_runtime(payload, runtime_paths):
    script = os.path.basename(payload.get("script", ""))
    if script in runtime_paths:
        payload["script"] = runtime_paths[script]


def rewrite_ranker_lock(payload, model_paths):
    result = json.loads(json.dumps(payload))
    result["binary_model"] = model_paths[os.path.abspath(
        payload["binary_model"]
    )]
    result["ordinal_model"] = model_paths[os.path.abspath(
        payload["ordinal_model"]
    )]
    return result


def rewrite_pointwise_lock(payload, model_paths):
    result = json.loads(json.dumps(payload))
    result["model_path"] = model_paths[os.path.abspath(
        payload["model_path"]
    )]
    return result


def rewrite_provenance(payload, stage31_path, stage31_sha,
                       stage33_path, stage33_sha, source_paths):
    provenance = payload["provenance"]
    provenance["stage31_lock"] = stage31_path
    provenance["stage31_lock_sha256"] = stage31_sha
    provenance["stage33_lock"] = stage33_path
    provenance["stage33_lock_sha256"] = stage33_sha
    for name, path in source_paths.items():
        if name in provenance.get("sources", {}):
            provenance["sources"][name]["path"] = path


def rewrite_nested_lock(payload, stage31_item, stage33_item,
                        stage31_payload, stage33_payload, output_dir,
                        runtime_paths):
    result = json.loads(json.dumps(payload))
    rewrite_provenance(
        result,
        absolute_bundle_path(output_dir, stage31_item["path"]),
        stage31_item["sha256"],
        absolute_bundle_path(output_dir, stage33_item["path"]),
        stage33_item["sha256"],
        {
            "binary": stage31_payload["binary_model"],
            "ordinal": stage31_payload["ordinal_model"],
            "pointwise": stage33_payload["model_path"],
        },
    )
    rewrite_runtime(result, runtime_paths)
    return result


def add_models(entries, staging, output_dir):
    models = []
    paths = {}
    for source, relative, expected_sha in entries:
        assert base.sha256(source) == expected_sha, source
        assert os.path.abspath(source) not in paths, source
        item = base.copy_component(source, staging, relative)
        models.append(item)
        paths[os.path.abspath(source)] = absolute_bundle_path(
            output_dir, relative
        )
    return models, paths


def model_entries(stage31, stage33, stage154, stage164_31,
                  stage164_33, selector):
    entries = [
        (stage31["binary_model"], "models/original_binary.txt",
         stage31["binary_model_sha256"]),
        (stage31["ordinal_model"], "models/original_ordinal.txt",
         stage31["ordinal_model_sha256"]),
        (stage33["model_path"], "models/original_pointwise.txt",
         stage33["model_sha256"]),
    ]
    for index, item in enumerate(stage154["models"]):
        entries.append((
            item["path"],
            "models/stage154_selector_{:02d}.txt".format(index),
            item["sha256"],
        ))
    entries.extend([
        (stage164_31["binary_model"], "models/stage164_binary.txt",
         stage164_31["binary_model_sha256"]),
        (stage164_31["ordinal_model"], "models/stage164_ordinal.txt",
         stage164_31["ordinal_model_sha256"]),
        (stage164_33["model_path"], "models/stage164_pointwise.txt",
         stage164_33["model_sha256"]),
    ])
    for index, item in enumerate(selector["models"]):
        entries.append((
            item["path"],
            "models/stage167_selector_{:02d}.txt".format(index),
            item["sha256"],
        ))
    return entries


def validate_dependencies(args, stage142, stage154, stage164_parent,
                          stage165, selector):
    assert selector["stage"] == STAGE
    assert selector["stage154_lock_sha256"] == base.sha256(args.stage154_lock)
    assert selector["stage165_policy_sha256"] == base.sha256(args.stage165_lock)
    assert selector["stage31_lock_sha256"] == base.sha256(args.stage31_lock)
    assert selector["stage33_lock_sha256"] == base.sha256(args.stage33_lock)
    assert selector["stage142_lock_sha256"] == base.sha256(args.stage142_lock)
    assert stage154["stage142_lock_sha256"] == base.sha256(args.stage142_lock)
    assert stage154["validation_labels_used_for_selection"] is False
    assert stage165["validation_labels_used_for_selection"] is False
    assert stage165["parent_policy_sha256"] == base.sha256(
        args.stage164_parent_lock
    )
    assert stage142["validation_labels_used_for_selection"] is False
    assert stage142["selection_data_scope"] == (
        "scanrefer_train_scene_hash_dev_only"
    )
    assert stage164_parent["validation_labels_used_for_selection"] is False
    for payload, stage31_path, stage33_path in (
        (stage142, args.stage31_lock, args.stage33_lock),
        (stage154, args.stage31_lock, args.stage33_lock),
        (stage164_parent, args.stage164_stage31_lock,
         args.stage164_stage33_lock),
        (stage165, args.stage164_stage31_lock,
         args.stage164_stage33_lock),
    ):
        provenance = payload["provenance"]
        assert provenance["stage31_lock_sha256"] == base.sha256(stage31_path)
        assert provenance["stage33_lock_sha256"] == base.sha256(stage33_path)


def build_artifact(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    stage31 = base.read_json(args.stage31_lock)
    stage33 = base.read_json(args.stage33_lock)
    stage142 = base.read_json(args.stage142_lock)
    stage154 = base.read_json(args.stage154_lock)
    stage164_31 = base.read_json(args.stage164_stage31_lock)
    stage164_33 = base.read_json(args.stage164_stage33_lock)
    stage164_parent = base.read_json(args.stage164_parent_lock)
    stage165 = base.read_json(args.stage165_lock)
    selector = base.read_json(args.selector_lock)
    validation = base.read_json(args.validation_result)
    metrics = base.validate_scientific_gates(
        selector, validation, args.selector_lock
    )
    validate_dependencies(
        args, stage142, stage154, stage164_parent, stage165, selector
    )
    alternate_tensors, tensor_audit = base.compare_checkpoints(
        args.alternate_checkpoint, args.primary_checkpoint
    )

    staging = args.output_dir + ".building.{}".format(os.getpid())
    assert not os.path.exists(staging), staging
    os.makedirs(staging)
    try:
        primary_item = base.link_primary(args.primary_checkpoint, staging)
        primary_item["source_sha256"] = base.sha256(args.primary_checkpoint)
        primary_item["source_path_for_audit"] = os.path.abspath(
            args.primary_checkpoint
        )
        assert primary_item["sha256"] == primary_item["source_sha256"]

        alternate_payload = {
            "format": "stage153_alternate_rerank_head_v1",
            "source_checkpoint_sha256": base.sha256(
                args.alternate_checkpoint
            ),
            "primary_checkpoint_sha256": primary_item["sha256"],
            "tensor_names": list(base.ALTERNATE_HEAD_KEYS),
            "tensors": alternate_tensors,
        }
        alternate_relative = "weights/stage135c_alternate_rerank_head.pth"
        alternate_path = os.path.join(staging, alternate_relative)
        torch.save(alternate_payload, alternate_path)
        alternate_item = {
            "path": alternate_relative,
            "sha256": base.sha256(alternate_path),
            "size": os.path.getsize(alternate_path),
            "source_checkpoint_sha256": alternate_payload[
                "source_checkpoint_sha256"
            ],
        }

        models, model_paths = add_models(
            model_entries(
                stage31, stage33, stage154, stage164_31, stage164_33,
                selector,
            ),
            staging, args.output_dir,
        )
        runtime = []
        runtime_paths = {}
        for source in args.runtime_file:
            basename = os.path.basename(source)
            assert basename not in runtime_paths, basename
            relative = "runtime/" + basename
            runtime.append(base.copy_component(source, staging, relative))
            runtime_paths[basename] = absolute_bundle_path(
                args.output_dir, relative
            )

        stage31_bundle = rewrite_ranker_lock(stage31, model_paths)
        stage33_bundle = rewrite_pointwise_lock(stage33, model_paths)
        stage31_item = base.write_json_component(
            stage31_bundle, args.stage31_lock, staging,
            "locks/stage31.json",
        )
        stage33_item = base.write_json_component(
            stage33_bundle, args.stage33_lock, staging,
            "locks/stage33.json",
        )
        stage142_bundle = rewrite_nested_lock(
            stage142, stage31_item, stage33_item, stage31_bundle,
            stage33_bundle, args.output_dir, runtime_paths,
        )
        stage142_item = base.write_json_component(
            stage142_bundle, args.stage142_lock, staging,
            "locks/stage142.json",
        )

        stage154_bundle = json.loads(json.dumps(stage154))
        for item in stage154_bundle["models"]:
            item["path"] = model_paths[os.path.abspath(item["path"])]
        stage154_bundle["stage142_lock"] = absolute_bundle_path(
            args.output_dir, stage142_item["path"]
        )
        stage154_bundle["stage142_lock_sha256"] = stage142_item["sha256"]
        stage154_bundle["stage31_lock_sha256"] = stage31_item["sha256"]
        stage154_bundle["stage33_lock_sha256"] = stage33_item["sha256"]
        rewrite_provenance(
            stage154_bundle,
            absolute_bundle_path(args.output_dir, stage31_item["path"]),
            stage31_item["sha256"],
            absolute_bundle_path(args.output_dir, stage33_item["path"]),
            stage33_item["sha256"],
            {
                "binary": stage31_bundle["binary_model"],
                "ordinal": stage31_bundle["ordinal_model"],
                "pointwise": stage33_bundle["model_path"],
            },
        )
        rewrite_runtime(stage154_bundle, runtime_paths)
        stage154_item = base.write_json_component(
            stage154_bundle, args.stage154_lock, staging,
            "locks/stage154.json",
        )

        stage164_31_bundle = rewrite_ranker_lock(stage164_31, model_paths)
        stage164_33_bundle = rewrite_pointwise_lock(stage164_33, model_paths)
        stage164_31_item = base.write_json_component(
            stage164_31_bundle, args.stage164_stage31_lock, staging,
            "locks/stage164_stage31.json",
        )
        stage164_33_item = base.write_json_component(
            stage164_33_bundle, args.stage164_stage33_lock, staging,
            "locks/stage164_stage33.json",
        )
        stage164_parent_bundle = rewrite_nested_lock(
            stage164_parent, stage164_31_item, stage164_33_item,
            stage164_31_bundle, stage164_33_bundle, args.output_dir,
            runtime_paths,
        )
        stage164_parent_item = base.write_json_component(
            stage164_parent_bundle, args.stage164_parent_lock, staging,
            "locks/stage164_parent.json",
        )

        stage165_bundle = rewrite_nested_lock(
            stage165, stage164_31_item, stage164_33_item,
            stage164_31_bundle, stage164_33_bundle, args.output_dir,
            runtime_paths,
        )
        stage165_bundle["parent_policy"] = absolute_bundle_path(
            args.output_dir, stage164_parent_item["path"]
        )
        stage165_bundle["parent_policy_sha256"] = stage164_parent_item["sha256"]
        stage165_item = base.write_json_component(
            stage165_bundle, args.stage165_lock, staging,
            "locks/stage165.json",
        )

        selector_bundle = json.loads(json.dumps(selector))
        for item in selector_bundle["models"]:
            item["path"] = model_paths[os.path.abspath(item["path"])]
        selector_bundle["stage154_lock"] = absolute_bundle_path(
            args.output_dir, stage154_item["path"]
        )
        selector_bundle["stage154_lock_sha256"] = stage154_item["sha256"]
        selector_bundle["stage165_policy"] = absolute_bundle_path(
            args.output_dir, stage165_item["path"]
        )
        selector_bundle["stage165_policy_sha256"] = stage165_item["sha256"]
        selector_bundle["stage31_lock_sha256"] = stage31_item["sha256"]
        selector_bundle["stage33_lock_sha256"] = stage33_item["sha256"]
        selector_bundle["stage142_lock_sha256"] = stage142_item["sha256"]
        if "stage142_provenance" in selector_bundle:
            holder = {"provenance": selector_bundle["stage142_provenance"]}
            rewrite_provenance(
                holder,
                absolute_bundle_path(args.output_dir, stage31_item["path"]),
                stage31_item["sha256"],
                absolute_bundle_path(args.output_dir, stage33_item["path"]),
                stage33_item["sha256"],
                {
                    "binary": stage31_bundle["binary_model"],
                    "ordinal": stage31_bundle["ordinal_model"],
                    "pointwise": stage33_bundle["model_path"],
                },
            )
        rewrite_runtime(selector_bundle, runtime_paths)
        selector_item = base.write_json_component(
            selector_bundle, args.selector_lock, staging,
            "locks/stage167.json",
        )

        locks = {
            "stage31": stage31_item,
            "stage33": stage33_item,
            "stage142": stage142_item,
            "stage154": stage154_item,
            "stage164_stage31": stage164_31_item,
            "stage164_stage33": stage164_33_item,
            "stage164_parent": stage164_parent_item,
            "stage165": stage165_item,
            "stage167": selector_item,
            "validation_result": base.copy_component(
                args.validation_result, staging,
                "evidence/validation_result.json",
            ),
        }
        manifest = {
            "format": "butd_stage153_deployable_bundle_v1",
            "created_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
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
            "formal_status": (
                "offline_goal_met_pending_fresh_bundle_inference_reload"
            ),
        }
        manifest_path = os.path.join(staging, "manifest.json")
        base.atomic_json(manifest_path, manifest)
        base.atomic_json(os.path.join(staging, "build_receipt.json"), {
            "manifest_sha256": base.sha256(manifest_path),
            "artifact_format": manifest["format"],
            "validation_metrics": metrics,
            "fresh_bundle_inference_reload_required": True,
        })
        os.replace(staging, args.output_dir)
    except Exception:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        raise
    base.validate_artifact(args.output_dir)
    return os.path.join(args.output_dir, "manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alternate_checkpoint")
    parser.add_argument("primary_checkpoint")
    parser.add_argument("stage31_lock")
    parser.add_argument("stage33_lock")
    parser.add_argument("stage142_lock")
    parser.add_argument("stage154_lock")
    parser.add_argument("stage164_stage31_lock")
    parser.add_argument("stage164_stage33_lock")
    parser.add_argument("stage164_parent_lock")
    parser.add_argument("stage165_lock")
    parser.add_argument("selector_lock")
    parser.add_argument("validation_result")
    parser.add_argument("output_dir")
    parser.add_argument("--runtime-file", action="append", default=[])
    args = parser.parse_args()
    path = build_artifact(args)
    print(json.dumps({
        "manifest": path,
        "sha256": base.sha256(path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
