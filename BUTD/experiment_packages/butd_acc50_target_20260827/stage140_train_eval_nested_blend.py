#!/usr/bin/env python3
"""Freeze and evaluate a train-only nested blend of frozen option models.

The outer blend weight and gate are selected exclusively on a deterministic
scene-disjoint split of the ScanRefer training dump.  The validation dump is
consumed only by the separate ``evaluate`` command after the policy lock has
been written.
"""

import argparse
import hashlib
import json
import math
import os

import lightgbm as lgb
import numpy as np

from train_joint_option_ranker import (
    ACTIONS,
    FEATURE_NAMES,
    MATCH_POWERS,
    build_dataset,
    choose_gate,
    group_decisions,
    load_rows,
    materialize,
    metrics,
    normalize_group_scores,
    split_indices,
)


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
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def summarize(ious):
    result = metrics(ious)
    result.update({
        "hits025": int((ious >= 0.25).sum()),
        "hits050": int((ious >= 0.50).sum()),
    })
    return result


def fix_break(baseline, selected, changed):
    result = {"changed": int(changed.sum())}
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        before = baseline >= threshold
        after = selected >= threshold
        result["fix_" + suffix] = int((~before & after).sum())
        result["break_" + suffix] = int((before & ~after).sum())
        result["net_" + suffix] = (
            result["fix_" + suffix] - result["break_" + suffix]
        )
    return result


def validate_lock_schema(blend_lock, point_lock):
    assert blend_lock["feature_names"] == FEATURE_NAMES
    assert point_lock["feature_names"] == FEATURE_NAMES
    assert int(blend_lock["max_candidates"]) == 8
    assert int(point_lock["max_candidates"]) == 8
    for lock in (blend_lock, point_lock):
        if "match_powers" in lock:
            assert tuple(lock["match_powers"]) == MATCH_POWERS
        if "actions" in lock:
            assert tuple(lock["actions"]) == ACTIONS
    assert blend_lock["protocol"].startswith(
        "train_dev_locked_group_standardized"
    )
    assert point_lock["protocol"] == (
        "scene_hash_train_only_equal_group_weight_pointwise"
    )


def verified_sources(blend_lock_path, point_lock_path):
    blend_lock = read_json(blend_lock_path)
    point_lock = read_json(point_lock_path)
    validate_lock_schema(blend_lock, point_lock)
    sources = {
        "ordinal": {
            "path": blend_lock["ordinal_model"],
            "sha256": blend_lock["ordinal_model_sha256"],
            "iteration": int(blend_lock["ordinal_iteration"]),
        },
        "binary": {
            "path": blend_lock["binary_model"],
            "sha256": blend_lock["binary_model_sha256"],
            "iteration": int(blend_lock["binary_iteration"]),
        },
        "pointwise": {
            "path": point_lock["model_path"],
            "sha256": point_lock["model_sha256"],
            "iteration": int(point_lock["best_iteration"]),
        },
    }
    for source in sources.values():
        assert os.path.isfile(source["path"]), source["path"]
        assert sha256(source["path"]) == source["sha256"], source["path"]
    selected = blend_lock["selected"]
    inner = {
        "ordinal_weight": float(selected["ordinal_weight"]),
        "binary_weight": float(selected["binary_weight"]),
    }
    assert abs(inner["ordinal_weight"] + inner["binary_weight"] - 1.0) < 1e-8
    provenance = {
        "stage31_lock": os.path.abspath(blend_lock_path),
        "stage31_lock_sha256": sha256(blend_lock_path),
        "stage33_lock": os.path.abspath(point_lock_path),
        "stage33_lock_sha256": sha256(point_lock_path),
        "sources": sources,
        "inner_blend": inner,
    }
    return provenance


def load_boosters(provenance):
    return {
        name: lgb.Booster(model_file=source["path"])
        for name, source in provenance["sources"].items()
    }


def component_scores(features, groups, boosters, provenance):
    raw = {
        name: boosters[name].predict(
            features,
            num_iteration=int(provenance["sources"][name]["iteration"]),
        )
        for name in ("ordinal", "binary", "pointwise")
    }
    ordinal = normalize_group_scores(raw["ordinal"], groups)
    binary = normalize_group_scores(raw["binary"], groups)
    pointwise = normalize_group_scores(raw["pointwise"], groups)
    inner = (
        provenance["inner_blend"]["ordinal_weight"] * ordinal
        + provenance["inner_blend"]["binary_weight"] * binary
    )
    # Stage139 used a group-standardized Stage31 score before the outer blend.
    inner = normalize_group_scores(inner, groups)
    return inner, pointwise


def evaluate_arrays(scores, ious, groups, baselines, threshold):
    selected, baseline, gaps = group_decisions(
        scores, ious, groups, baselines, float(threshold)
    )
    changed = gaps >= float(threshold)
    return {
        "selected": summarize(selected),
        "baseline": summarize(baseline),
        "changed_ratio": float(changed.mean()),
        "score_gap_mean": float(gaps.mean()),
        "fix_break": fix_break(baseline, selected, changed),
    }


def train(args):
    assert not os.path.exists(args.output_lock), args.output_lock
    provenance = verified_sources(args.stage31_lock, args.stage33_lock)
    rows = load_rows(args.train_dump)
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
    boosters = load_boosters(provenance)
    dev_x, _, dev_ious, dev_groups, dev_baselines = arrays["dev"]
    inner_dev, pointwise_dev = component_scores(
        dev_x, dev_groups, boosters, provenance
    )
    candidates = []
    for pointwise_weight in np.linspace(0.0, 1.0, 21):
        inner_weight = 1.0 - pointwise_weight
        scores = inner_weight * inner_dev + pointwise_weight * pointwise_dev
        gate, baseline_metrics = choose_gate(
            scores, dev_ious, dev_groups, dev_baselines
        )
        result = evaluate_arrays(
            scores, dev_ious, dev_groups, dev_baselines, gate["threshold"]
        )
        candidates.append({
            "inner_weight": float(inner_weight),
            "pointwise_weight": float(pointwise_weight),
            "gate": gate,
            "dev": result,
            "dev_baseline_from_gate_search": baseline_metrics,
        })
        print(
            "CANDIDATE pointwise_weight={:.2f} hits={}/{} threshold={:.9g}".format(
                pointwise_weight,
                result["selected"]["hits025"],
                result["selected"]["hits050"],
                float(gate["threshold"]),
            ),
            flush=True,
        )
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]["dev"]["selected"]["acc050"],
        candidates[index]["dev"]["selected"]["acc025"],
        candidates[index]["dev"]["selected"]["mean_iou"],
        -candidates[index]["dev"]["changed_ratio"],
    ))
    selected_config = candidates[selected_index]

    test_x, _, test_ious, test_groups, test_baselines = arrays["test"]
    inner_test, pointwise_test = component_scores(
        test_x, test_groups, boosters, provenance
    )
    test_scores = (
        selected_config["inner_weight"] * inner_test
        + selected_config["pointwise_weight"] * pointwise_test
    )
    internal_test = evaluate_arrays(
        test_scores,
        test_ious,
        test_groups,
        test_baselines,
        selected_config["gate"]["threshold"],
    )
    lock = {
        "protocol": "scanrefer_train_only_scene_hash_dev_locked_nested_blend_v1",
        "selection_data_scope": "scanrefer_train_scene_hash_dev_only",
        "validation_labels_used_for_selection": False,
        "train_dump": os.path.abspath(args.train_dump),
        "train_dump_sha256": sha256(args.train_dump),
        "script": os.path.abspath(__file__),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "max_candidates": 8,
        "feature_names": FEATURE_NAMES,
        "match_powers": list(MATCH_POWERS),
        "actions": list(ACTIONS),
        "provenance": provenance,
        "outer_weight_grid": [
            float(value) for value in np.linspace(0.0, 1.0, 21)
        ],
        "gate_search": "train_joint_option_ranker.choose_gate_201_quantiles",
        "selection_criterion": (
            "dev_acc050_then_acc025_then_mean_iou_then_lower_changed_ratio"
        ),
        "selected_index": int(selected_index),
        "selected": selected_config,
        "candidates": candidates,
        "internal_scene_hash_test": internal_test,
        "split_sizes": {
            name: int(len(indices)) for name, indices in splits.items()
        },
    }
    atomic_json(args.output_lock, lock)
    print(json.dumps({
        "output_lock": os.path.abspath(args.output_lock),
        "output_lock_sha256": sha256(args.output_lock),
        "selected_index": int(selected_index),
        "selected": selected_config,
        "internal_scene_hash_test": internal_test,
    }, indent=2, sort_keys=True), flush=True)


def evaluate(args):
    lock = read_json(args.policy_lock)
    assert lock["protocol"] == (
        "scanrefer_train_only_scene_hash_dev_locked_nested_blend_v1"
    )
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["feature_names"] == FEATURE_NAMES
    assert tuple(lock["match_powers"]) == MATCH_POWERS
    assert tuple(lock["actions"]) == ACTIONS
    assert sha256(lock["script"]) == lock["script_sha256"]
    provenance = lock["provenance"]
    for source in provenance["sources"].values():
        assert sha256(source["path"]) == source["sha256"]
    rows = load_rows(args.dump)
    group_features, group_ious, metas = build_dataset(
        rows, max_candidates=8, require_scene=False
    )
    features = np.concatenate(group_features, axis=0)
    ious = np.concatenate(group_ious, axis=0)
    groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
    baselines = np.asarray(
        [meta.baseline_index for meta in metas], dtype=np.int32
    )
    boosters = load_boosters(provenance)
    inner, pointwise = component_scores(
        features, groups, boosters, provenance
    )
    selected_config = lock["selected"]
    scores = (
        float(selected_config["inner_weight"]) * inner
        + float(selected_config["pointwise_weight"]) * pointwise
    )
    result_metrics = evaluate_arrays(
        scores,
        ious,
        groups,
        baselines,
        float(selected_config["gate"]["threshold"]),
    )
    count = len(rows)
    strict_hits025 = math.floor(0.5391 * count) + 1
    strict_hits050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "140_train_only_frozen_nested_blend_on_stage135c",
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "dump": os.path.abspath(args.dump),
        "dump_sha256": sha256(args.dump),
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": sha256(args.policy_lock),
        "selection_data_scope": lock["selection_data_scope"],
        "validation_labels_used_for_selection": False,
        "selected_config": selected_config,
        "metrics": result_metrics,
        "strict_goal_hits": {
            "acc025": strict_hits025,
            "acc050": strict_hits050,
        },
        "strict_goal_met_offline": bool(
            result_metrics["selected"]["hits025"] >= strict_hits025
            and result_metrics["selected"]["hits050"] >= strict_hits050
        ),
    }
    assert not os.path.exists(args.output_json), args.output_json
    atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_dump")
    train_parser.add_argument("stage31_lock")
    train_parser.add_argument("stage33_lock")
    train_parser.add_argument("output_lock")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("dump")
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
