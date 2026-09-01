#!/usr/bin/env python3
"""Lock a train-only 90%-change cap on the Stage164 nested ensemble."""

import argparse
import copy
import hashlib
import json
import math
import os

import numpy as np

import stage140_train_eval_nested_blend as nested
from train_joint_option_ranker import (
    build_dataset,
    load_rows,
    materialize,
    split_indices,
)


STAGE = "165_stage164_nested_blend_train_dev_90pct_change_cap"
RESULT_STAGE = "165_capped_same_domain_trio_validation_eval"
MAX_CHANGED_RATIO = 0.90


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


def choose_capped_gate(scores, ious, groups, baselines):
    _, _, gaps = nested.group_decisions(
        scores, ious, groups, baselines, float("-inf")
    )
    finite = gaps[np.isfinite(gaps)]
    assert finite.size
    thresholds = list(np.unique(
        np.quantile(finite, np.linspace(0.0, 1.0, 201))
    ))
    thresholds.append(float("inf"))
    candidates = []
    for threshold in thresholds:
        result = nested.evaluate_arrays(
            scores, ious, groups, baselines, float(threshold)
        )
        if result["changed_ratio"] > MAX_CHANGED_RATIO + 1e-12:
            continue
        if result["selected"]["hits025"] < result["baseline"]["hits025"]:
            continue
        candidates.append((float(threshold), result))
    assert candidates
    threshold, result = max(candidates, key=lambda item: (
        item[1]["selected"]["hits050"],
        item[1]["selected"]["hits025"],
        item[1]["selected"]["mean_iou"],
        -item[1]["changed_ratio"],
    ))
    gate = {
        "threshold": threshold,
        "max_changed_ratio": MAX_CHANGED_RATIO,
        "selection_scope": "train_scene_hash_dev_only",
        "selection_rule": (
            "max_acc050_then_acc025_then_mean_iou_then_lower_changed_ratio"
            "_subject_to_acc025_not_below_adapter_and_changed_ratio_le_0p90"
        ),
        "dev": result,
    }
    return gate


def lock_policy(train_dump, parent_policy_path, stage142_path,
                output_policy_path, output_auth_path):
    parent = read_json(parent_policy_path)
    old = read_json(stage142_path)
    expected_protocol = (
        "scanrefer_train_only_scene_hash_dev_locked_nested_blend_v1"
    )
    assert parent["protocol"] == expected_protocol
    assert old["protocol"] == expected_protocol
    assert parent["validation_labels_used_for_selection"] is False
    assert old["validation_labels_used_for_selection"] is False
    assert sha256(train_dump) == parent["train_dump_sha256"]
    assert parent["train_dump_sha256"] == old["train_dump_sha256"]
    assert parent["split_sizes"] == old["split_sizes"]

    rows = load_rows(train_dump)
    group_features, group_ious, metas = build_dataset(
        rows, max_candidates=8, require_scene=True
    )
    splits = split_indices(metas)
    arrays = {
        split: materialize(
            group_features, group_ious, metas, splits[split]
        )
        for split in ("dev", "test")
    }
    boosters = nested.load_boosters(parent["provenance"])
    dev_x, _, dev_ious, dev_groups, dev_baselines = arrays["dev"]
    inner_dev, pointwise_dev = nested.component_scores(
        dev_x, dev_groups, boosters, parent["provenance"]
    )
    selected = parent["selected"]
    dev_scores = (
        float(selected["inner_weight"]) * inner_dev
        + float(selected["pointwise_weight"]) * pointwise_dev
    )
    gate = choose_capped_gate(
        dev_scores, dev_ious, dev_groups, dev_baselines
    )

    test_x, _, test_ious, test_groups, test_baselines = arrays["test"]
    inner_test, pointwise_test = nested.component_scores(
        test_x, test_groups, boosters, parent["provenance"]
    )
    test_scores = (
        float(selected["inner_weight"]) * inner_test
        + float(selected["pointwise_weight"]) * pointwise_test
    )
    test_result = nested.evaluate_arrays(
        test_scores, test_ious, test_groups, test_baselines,
        float(gate["threshold"]),
    )

    policy = copy.deepcopy(parent)
    policy["stage"] = STAGE
    policy["script"] = os.path.abspath(__file__)
    policy["script_sha256"] = sha256(os.path.abspath(__file__))
    policy["parent_policy"] = os.path.abspath(parent_policy_path)
    policy["parent_policy_sha256"] = sha256(parent_policy_path)
    policy["selected"]["gate"] = gate
    policy["selected"]["dev"] = gate["dev"]
    policy["internal_scene_hash_test"] = test_result
    policy["gate_protocol"] = "train_dev_fixed_90pct_change_cap_v1"
    policy["validation_labels_used_for_selection"] = False
    assert not os.path.exists(output_policy_path), output_policy_path
    atomic_json(output_policy_path, policy)

    old_test = old["internal_scene_hash_test"]
    old_selected = old_test["selected"]
    new_selected = test_result["selected"]
    assert int(old_selected["count"]) == int(new_selected["count"])
    assert int(old_test["baseline"]["hits025"]) == int(
        test_result["baseline"]["hits025"]
    )
    assert int(old_test["baseline"]["hits050"]) == int(
        test_result["baseline"]["hits050"]
    )
    count = int(new_selected["count"])
    allowed_loss025 = int(math.floor(0.002 * count))
    delta025 = int(new_selected["hits025"]) - int(old_selected["hits025"])
    delta050 = int(new_selected["hits050"]) - int(old_selected["hits050"])
    checks = {
        "acc025_loss_within_0p2pp": delta025 >= -allowed_loss025,
        "acc050_net_hits_at_least_5": delta050 >= 5,
        "test_changed_ratio_at_most_0p90": (
            float(test_result["changed_ratio"]) <= MAX_CHANGED_RATIO + 1e-12
        ),
    }
    gate_pass = all(checks.values())
    authorization = {
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
        "nested_policy": os.path.abspath(output_policy_path),
        "nested_policy_sha256": sha256(output_policy_path),
        "parent_policy_sha256": sha256(parent_policy_path),
        "train_dump_sha256": policy["train_dump_sha256"],
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
            "new_changed_ratio": float(test_result["changed_ratio"]),
            "checks": checks,
        },
    }
    assert not os.path.exists(output_auth_path), output_auth_path
    atomic_json(output_auth_path, authorization)
    return {"policy": policy, "authorization": authorization}


def wrap_validation(authorization_path, raw_result_path, output_path):
    authorization = read_json(authorization_path)
    raw = read_json(raw_result_path)
    assert authorization["stage"] == STAGE
    assert authorization["internal_gate_pass"] is True
    assert authorization["validation_evaluation_authorized"] is True
    assert raw["stage"] == "140_train_only_frozen_nested_blend_on_stage135c"
    assert raw["validation_labels_used_for_selection"] is False
    assert raw["policy_lock_sha256"] == authorization["nested_policy_sha256"]
    result = dict(raw)
    result["stage"] = RESULT_STAGE
    result["policy_lock"] = os.path.abspath(authorization_path)
    result["policy_lock_sha256"] = sha256(authorization_path)
    result["nested_policy_lock"] = authorization["nested_policy"]
    result["nested_policy_lock_sha256"] = raw["policy_lock_sha256"]
    result["validation_labels_used_for_selection"] = False
    assert not os.path.exists(output_path), output_path
    atomic_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock")
    lock.add_argument("train_dump")
    lock.add_argument("parent_policy")
    lock.add_argument("stage142_policy")
    lock.add_argument("output_policy")
    lock.add_argument("output_authorization")
    wrap = sub.add_parser("wrap-validation")
    wrap.add_argument("authorization")
    wrap.add_argument("raw_result")
    wrap.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "lock":
        payload = lock_policy(
            args.train_dump, args.parent_policy, args.stage142_policy,
            args.output_policy, args.output_authorization,
        )
    else:
        payload = wrap_validation(
            args.authorization, args.raw_result, args.output_json
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
