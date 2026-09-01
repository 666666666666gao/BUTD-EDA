#!/usr/bin/env python3
"""Authorize and wrap the fixed same-domain three-ranker policy."""

import argparse
import hashlib
import json
import math
import os


STAGE = "164_stage135c_same_domain_retrained_trio_nested_blend"
RESULT_STAGE = "164_same_domain_trio_validation_eval"


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


def authorize(stage142_path, new_policy_path, output_path):
    old = read_json(stage142_path)
    new = read_json(new_policy_path)
    expected_protocol = (
        "scanrefer_train_only_scene_hash_dev_locked_nested_blend_v1"
    )
    assert old["protocol"] == expected_protocol
    assert new["protocol"] == expected_protocol
    assert old["validation_labels_used_for_selection"] is False
    assert new["validation_labels_used_for_selection"] is False
    assert old["train_dump_sha256"] == new["train_dump_sha256"]
    assert old["split_sizes"] == new["split_sizes"]
    old_test = old["internal_scene_hash_test"]
    new_test = new["internal_scene_hash_test"]
    old_selected = old_test["selected"]
    new_selected = new_test["selected"]
    old_baseline = old_test["baseline"]
    new_baseline = new_test["baseline"]
    assert int(old_selected["count"]) == int(new_selected["count"])
    assert int(old_baseline["hits025"]) == int(new_baseline["hits025"])
    assert int(old_baseline["hits050"]) == int(new_baseline["hits050"])
    count = int(new_selected["count"])
    allowed_loss025 = int(math.floor(0.002 * count))
    delta025 = int(new_selected["hits025"]) - int(old_selected["hits025"])
    delta050 = int(new_selected["hits050"]) - int(old_selected["hits050"])
    checks = {
        "acc025_loss_within_0p2pp": delta025 >= -allowed_loss025,
        "acc050_net_hits_at_least_5": delta050 >= 5,
        "changed_ratio_at_most_0p98": (
            float(new_test["changed_ratio"]) <= 0.98
        ),
    }
    gate_pass = all(checks.values())
    payload = {
        "stage": STAGE,
        "script": os.path.abspath(__file__),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "validation_labels_used_for_selection": False,
        "selection_data_scope": "scanrefer_train_scene_hash_dev_only",
        "internal_gate_pass": gate_pass,
        "validation_evaluation_authorized": gate_pass,
        "models": [],
        "stage142_policy": os.path.abspath(stage142_path),
        "stage142_policy_sha256": sha256(stage142_path),
        "nested_policy": os.path.abspath(new_policy_path),
        "nested_policy_sha256": sha256(new_policy_path),
        "train_dump_sha256": new["train_dump_sha256"],
        "internal_comparison": {
            "count": count,
            "allowed_loss025_hits": allowed_loss025,
            "stage142_hits": [
                int(old_selected["hits025"]),
                int(old_selected["hits050"]),
            ],
            "new_hits": [
                int(new_selected["hits025"]),
                int(new_selected["hits050"]),
            ],
            "delta_hits": [delta025, delta050],
            "new_changed_ratio": float(new_test["changed_ratio"]),
            "checks": checks,
        },
    }
    assert not os.path.exists(output_path), output_path
    atomic_json(output_path, payload)
    return payload


def wrap_validation(authorization_path, raw_result_path, output_path):
    authorization = read_json(authorization_path)
    raw = read_json(raw_result_path)
    assert authorization["stage"] == STAGE
    assert authorization["internal_gate_pass"] is True
    assert authorization["validation_evaluation_authorized"] is True
    assert authorization["validation_labels_used_for_selection"] is False
    assert raw["stage"] == "140_train_only_frozen_nested_blend_on_stage135c"
    assert raw["validation_labels_used_for_selection"] is False
    assert raw["policy_lock_sha256"] == authorization["nested_policy_sha256"]
    payload = dict(raw)
    payload["stage"] = RESULT_STAGE
    payload["policy_lock"] = os.path.abspath(authorization_path)
    payload["policy_lock_sha256"] = sha256(authorization_path)
    payload["nested_policy_lock"] = os.path.abspath(
        authorization["nested_policy"]
    )
    payload["nested_policy_lock_sha256"] = raw["policy_lock_sha256"]
    payload["validation_labels_used_for_selection"] = False
    assert not os.path.exists(output_path), output_path
    atomic_json(output_path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    auth = sub.add_parser("authorize")
    auth.add_argument("stage142_policy")
    auth.add_argument("new_policy")
    auth.add_argument("output_json")
    wrap = sub.add_parser("wrap-validation")
    wrap.add_argument("authorization")
    wrap.add_argument("raw_result")
    wrap.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "authorize":
        payload = authorize(
            args.stage142_policy, args.new_policy, args.output_json
        )
    else:
        payload = wrap_validation(
            args.authorization, args.raw_result, args.output_json
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
