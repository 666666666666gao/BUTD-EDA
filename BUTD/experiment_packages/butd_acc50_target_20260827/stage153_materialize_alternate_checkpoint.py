#!/usr/bin/env python3
"""Materialize the alternate Stage135c rerank-head checkpoint from a bundle."""

import argparse
import json
import os

import torch

import stage153_build_final_artifact as artifact


def materialize(bundle_dir, output_path):
    assert not os.path.exists(output_path), output_path
    manifest = artifact.validate_artifact(bundle_dir)
    assert manifest["formal_status"] == (
        "offline_goal_met_pending_fresh_bundle_inference_reload"
    )
    primary_path = artifact.safe_resolve(
        bundle_dir, manifest["primary_checkpoint"]["path"]
    )
    alternate_path = artifact.safe_resolve(
        bundle_dir, manifest["alternate_rerank_head"]["path"]
    )
    checkpoint = torch.load(primary_path, map_location="cpu")
    alternate = torch.load(alternate_path, map_location="cpu")
    before = {}
    for name in artifact.ALTERNATE_HEAD_KEYS:
        before[name] = checkpoint["model"][name].detach().cpu().clone()
        replacement = alternate["tensors"][name]
        assert replacement.shape == checkpoint["model"][name].shape
        assert replacement.dtype == checkpoint["model"][name].dtype
        checkpoint["model"][name] = replacement.detach().cpu().clone()
        assert not torch.equal(before[name], checkpoint["model"][name])
    checkpoint["stage153_materialization"] = {
        "format": "stage153_materialized_stage135c_alternate_head_v1",
        "bundle_manifest_sha256": artifact.sha256(
            os.path.join(bundle_dir, "manifest.json")
        ),
        "primary_checkpoint_sha256": manifest["primary_checkpoint"]["sha256"],
        "alternate_head_sha256": manifest["alternate_rerank_head"]["sha256"],
        "replaced_tensor_names": list(artifact.ALTERNATE_HEAD_KEYS),
    }
    temporary = output_path + ".tmp.{}".format(os.getpid())
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    reloaded = torch.load(output_path, map_location="cpu")
    for name in artifact.ALTERNATE_HEAD_KEYS:
        assert torch.equal(
            reloaded["model"][name], alternate["tensors"][name]
        )
    return {
        "checkpoint": os.path.abspath(output_path),
        "sha256": artifact.sha256(output_path),
        "size": os.path.getsize(output_path),
        "materialization": reloaded["stage153_materialization"],
        "temporary_and_removable_after_fresh_inference": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir")
    parser.add_argument("output_checkpoint")
    args = parser.parse_args()
    print(json.dumps(
        materialize(args.bundle_dir, args.output_checkpoint),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
