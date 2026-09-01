#!/usr/bin/env python3
"""Finalize a Stage153 bundle after fresh full-model inference and reload."""

import argparse
import json
import os
import shutil

import stage153_build_final_artifact as artifact


def finalize(bundle_dir, fresh_result_path, reload_receipt_path):
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    manifest = artifact.validate_artifact(bundle_dir)
    assert manifest["formal_status"] == (
        "offline_goal_met_pending_fresh_bundle_inference_reload"
    )
    fresh = artifact.read_json(fresh_result_path)
    receipt = artifact.read_json(reload_receipt_path)
    expected_result_stage = artifact.SELECTOR_RESULT_STAGES[
        manifest["selector_stage"]
    ]
    assert fresh["stage"] == expected_result_stage
    assert fresh["strict_goal_met_offline"] is True
    assert receipt["stage"] in (
        "stage153d_fresh_bundle_reload",
        "stage154d_fresh_bundle_reload",
        "stage155d_fresh_bundle_reload",
        "stage156d_fresh_bundle_reload",
        "stage157d_fresh_bundle_reload",
        "stage158d_fresh_bundle_reload",
        "stage162d_fresh_bundle_reload",
        "stage163d_fresh_bundle_reload",
        "stage165d_fresh_bundle_reload",
        "stage167d_fresh_bundle_reload",
    )
    assert receipt["no_ground_truth_used_for_inference"] is True
    if receipt["stage"] in (
        "stage162d_fresh_bundle_reload",
        "stage163d_fresh_bundle_reload",
        "stage165d_fresh_bundle_reload",
    ):
        assert receipt["primary_stage150_hits"] is None
        assert receipt["primary_inference_required_for_deployment"] is False
    else:
        assert receipt["primary_stage150_hits"] == [5234, 3979]
    assert receipt["alternate_stage135c_hits"] == [5215, 3956]
    assert receipt["evaluator_restored_sha256"] == (
        "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86"
    )
    selected = fresh["metrics"]["selected"]
    goals = fresh["strict_goal_hits"]
    assert int(selected["hits025"]) >= int(goals["acc025"])
    assert int(selected["hits050"]) >= int(goals["acc050"])
    assert receipt["selected_hits"] == [
        int(selected["hits025"]), int(selected["hits050"])
    ]
    expected = manifest["validation_metrics"]
    assert int(selected["hits025"]) == int(expected["hits025"])
    assert int(selected["hits050"]) == int(expected["hits050"])
    assert receipt["locked_validation_result_sha256"] == artifact.sha256(
        fresh_result_path
    )

    evidence_dir = os.path.join(bundle_dir, "evidence")
    fresh_destination = os.path.join(
        evidence_dir, "fresh_bundle_reload_validation_result.json"
    )
    receipt_destination = os.path.join(
        evidence_dir, "fresh_bundle_reload_receipt.json"
    )
    assert not os.path.exists(fresh_destination)
    assert not os.path.exists(receipt_destination)
    shutil.copy2(fresh_result_path, fresh_destination)
    shutil.copy2(reload_receipt_path, receipt_destination)

    manifest["fresh_bundle_reload"] = {
        "validation_result": {
            "path": "evidence/fresh_bundle_reload_validation_result.json",
            "sha256": artifact.sha256(fresh_destination),
            "size": os.path.getsize(fresh_destination),
        },
        "receipt": {
            "path": "evidence/fresh_bundle_reload_receipt.json",
            "sha256": artifact.sha256(receipt_destination),
            "size": os.path.getsize(receipt_destination),
        },
        "selected_hits025": int(selected["hits025"]),
        "selected_hits050": int(selected["hits050"]),
        "selected_acc025": float(selected["acc025"]),
        "selected_acc050": float(selected["acc050"]),
        "strict_goal_met": True,
    }
    manifest["formal_status"] = "complete_fresh_bundle_inference_reload"
    artifact.atomic_json(manifest_path, manifest)
    build_receipt_path = os.path.join(bundle_dir, "build_receipt.json")
    build_receipt = artifact.read_json(build_receipt_path)
    build_receipt["manifest_sha256"] = artifact.sha256(manifest_path)
    build_receipt["fresh_bundle_inference_reload_required"] = False
    build_receipt["formal_status"] = manifest["formal_status"]
    build_receipt["formal_metrics"] = manifest["fresh_bundle_reload"]
    artifact.atomic_json(build_receipt_path, build_receipt)
    validated = artifact.validate_artifact(bundle_dir)
    assert validated["formal_status"] == "complete_fresh_bundle_inference_reload"
    return validated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir")
    parser.add_argument("fresh_result")
    parser.add_argument("reload_receipt")
    args = parser.parse_args()
    manifest = finalize(args.bundle_dir, args.fresh_result, args.reload_receipt)
    print(json.dumps({
        "bundle": os.path.abspath(args.bundle_dir),
        "manifest_sha256": artifact.sha256(
            os.path.join(args.bundle_dir, "manifest.json")
        ),
        "formal_status": manifest["formal_status"],
        "formal_metrics": manifest["fresh_bundle_reload"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
