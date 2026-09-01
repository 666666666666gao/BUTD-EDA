#!/usr/bin/env python3
"""Fixed 0.1 residual blend of Stage142 and a same-domain Tier-3 ranker."""

import argparse
import json
import math
import os

import lightgbm as lgb
import numpy as np

import stage140_train_eval_nested_blend as nested
import stage162_tier3_option_ranker as tier3
import train_joint_option_ranker as option


STAGE = "163_stage142_plus_tier3_fixed_residual_option_ranker"
TIER3_WEIGHT = 0.1


def fixed_blend(stage142_scores, tier3_scores):
    return (
        (1.0 - TIER3_WEIGHT) * np.asarray(stage142_scores, dtype=np.float32)
        + TIER3_WEIGHT * np.asarray(tier3_scores, dtype=np.float32)
    )


def internal_gate_pass(result, stage142_selected):
    count = int(stage142_selected["count"])
    tolerance025 = max(1, int(math.ceil(0.002 * count)))
    return bool(
        int(result["selected"]["hits050"])
        >= int(stage142_selected["hits050"]) + 5
        and int(result["selected"]["hits025"])
        >= int(stage142_selected["hits025"]) - tolerance025
        and float(result["changed_ratio"]) <= 0.98
    )


def load_stage142(stage31_lock, stage33_lock, stage142_lock):
    lock = json.load(open(stage142_lock, encoding="utf-8"))
    assert lock["validation_labels_used_for_selection"] is False
    provenance = nested.verified_sources(stage31_lock, stage33_lock)
    boosters = nested.load_boosters(provenance)
    selected = lock["selected"]
    assert abs(float(selected["inner_weight"]) - 0.95) < 1e-12
    assert abs(float(selected["pointwise_weight"]) - 0.05) < 1e-12
    return lock, provenance, boosters


def stage142_scores(features, groups, boosters, provenance, selected):
    inner, pointwise = nested.component_scores(
        features, groups, boosters, provenance
    )
    return (
        float(selected["inner_weight"]) * inner
        + float(selected["pointwise_weight"]) * pointwise
    )


def evaluate_all(
    features, ious, groups, baselines, tier3_booster, iteration,
    nested_scores, nested_threshold, blend_threshold,
):
    raw_tier3 = tier3_booster.predict(features, num_iteration=int(iteration))
    normalized_tier3 = option.normalize_group_scores(raw_tier3, groups)
    blended = fixed_blend(nested_scores, normalized_tier3)
    stage142_result = nested.evaluate_arrays(
        nested_scores, ious, groups, baselines, float(nested_threshold)
    )
    blended_result = tier3.evaluate_scores(
        blended, ious, groups, baselines, float(blend_threshold)
    )
    return stage142_result, blended_result


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    rows = option.load_rows(args.train_dump)
    assert len(rows) == 36665
    group_features, group_ious, metas = option.build_dataset(
        rows, max_candidates=8, require_scene=True
    )
    splits = option.split_indices(metas)
    arrays = {
        split: tier3.materialize_tier3(
            group_features, group_ious, metas, splits[split]
        )
        for split in ("train", "dev", "test")
    }
    x_train, y_train, _, groups_train, _ = arrays["train"]
    x_dev, y_dev, iou_dev, groups_dev, baseline_dev = arrays["dev"]
    ranker = tier3.make_ranker(args.num_threads)
    ranker.fit(
        x_train, y_train,
        group=groups_train.tolist(),
        eval_set=[(x_dev, y_dev)],
        eval_group=[groups_dev.tolist()],
        eval_at=[1],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)],
        feature_name=option.FEATURE_NAMES,
    )
    iteration = int(ranker.best_iteration_ or ranker.n_estimators)
    stage142_lock, provenance, boosters = load_stage142(
        args.stage31_lock, args.stage33_lock, args.stage142_lock
    )
    selected_config = stage142_lock["selected"]
    dev_nested = stage142_scores(
        x_dev, groups_dev, boosters, provenance, selected_config
    )
    dev_tier3 = option.normalize_group_scores(
        ranker.booster_.predict(x_dev, num_iteration=iteration), groups_dev
    )
    dev_blend = fixed_blend(dev_nested, dev_tier3)
    gate, _ = option.choose_gate(
        dev_blend, iou_dev, groups_dev, baseline_dev
    )
    dev_stage142, dev_result = evaluate_all(
        x_dev, iou_dev, groups_dev, baseline_dev,
        ranker.booster_, iteration, dev_nested,
        selected_config["gate"]["threshold"], gate["threshold"],
    )
    dev_result.pop("selected_ious")

    x_test, _, iou_test, groups_test, baseline_test = arrays["test"]
    test_nested = stage142_scores(
        x_test, groups_test, boosters, provenance, selected_config
    )
    test_stage142, test_result = evaluate_all(
        x_test, iou_test, groups_test, baseline_test,
        ranker.booster_, iteration, test_nested,
        selected_config["gate"]["threshold"], gate["threshold"],
    )
    test_result.pop("selected_ious")
    authorized = internal_gate_pass(test_result, test_stage142["selected"])

    model_path = os.path.join(args.output_dir, "tier3_residual_ranker.txt")
    ranker.booster_.save_model(model_path, num_iteration=iteration)
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_fixed_residual_blend_lock",
        "protocol": (
            "stage142_fixed_0p9_plus_same_domain_tier3_fixed_0p1_"
            "scene70_train_scene15_dev_scene15_test_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": authorized,
        "internal_gate_pass": authorized,
        "script": os.path.abspath(__file__),
        "script_sha256": option.sha256(os.path.abspath(__file__)),
        "train_dump": os.path.abspath(args.train_dump),
        "train_dump_sha256": option.sha256(args.train_dump),
        "stage31_lock": os.path.abspath(args.stage31_lock),
        "stage31_lock_sha256": option.sha256(args.stage31_lock),
        "stage33_lock": os.path.abspath(args.stage33_lock),
        "stage33_lock_sha256": option.sha256(args.stage33_lock),
        "stage142_lock": os.path.abspath(args.stage142_lock),
        "stage142_lock_sha256": option.sha256(args.stage142_lock),
        "model_path": os.path.abspath(model_path),
        "model_sha256": option.sha256(model_path),
        "best_iteration": iteration,
        "max_candidates": 8,
        "match_powers": list(option.MATCH_POWERS),
        "actions": list(option.ACTIONS),
        "feature_names": option.FEATURE_NAMES,
        "tier_boundaries": [0.25, 0.50],
        "label_gain": [0, 1, 8],
        "stage142_weight": 0.9,
        "tier3_weight": TIER3_WEIGHT,
        "gate": gate,
        "stage142_selected_config": selected_config,
        "internal": {
            "dev": {"stage142": dev_stage142, "selected": dev_result},
            "test": {"stage142": test_stage142, "selected": test_result},
        },
        "split_group_counts": {
            split: len(indices) for split, indices in splits.items()
        },
    }
    lock_path = os.path.join(args.output_dir, "locked_residual_blend_policy.json")
    tier3.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": option.sha256(lock_path),
        "best_iteration": iteration,
        "gate": gate,
        "internal": lock["internal"],
        "internal_gate_pass": authorized,
        "validation_evaluation_authorized": authorized,
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = json.load(open(args.policy_lock, encoding="utf-8"))
    assert lock["stage"] == STAGE
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert option.sha256(lock["script"]) == lock["script_sha256"]
    assert option.sha256(lock["model_path"]) == lock["model_sha256"]
    rows = option.load_rows(args.dump)
    group_features, group_ious, metas = option.build_dataset(
        rows, max_candidates=lock["max_candidates"], require_scene=False
    )
    x = np.concatenate(group_features, axis=0)
    ious = np.concatenate(group_ious, axis=0)
    groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
    baselines = np.asarray(
        [meta.baseline_index for meta in metas], dtype=np.int32
    )
    stage142_lock, provenance, boosters = load_stage142(
        args.stage31_lock, args.stage33_lock, args.stage142_lock
    )
    nested_scores = stage142_scores(
        x, groups, boosters, provenance, stage142_lock["selected"]
    )
    ranker = lgb.Booster(model_file=lock["model_path"])
    stage142_result, result = evaluate_all(
        x, ious, groups, baselines, ranker, lock["best_iteration"],
        nested_scores, stage142_lock["selected"]["gate"]["threshold"],
        lock["gate"]["threshold"],
    )
    selected = result.pop("selected_ious")
    unique = np.asarray([
        bool(row.get("is_unique_label_only", False)) for row in rows
    ])
    result["unique"] = tier3.subset_summary(selected, unique)
    result["multiple"] = tier3.subset_summary(selected, ~unique)
    count = len(rows)
    strict025 = math.floor(0.5391 * count) + 1
    strict050 = math.floor(0.4241 * count) + 1
    output = {
        "stage": "163_tier3_residual_blend_validation_eval",
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": option.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "stage142": stage142_result,
        "metrics": result,
        "strict_goal_hits": {"acc025": strict025, "acc050": strict050},
        "strict_goal_met_offline": bool(
            result["selected"]["hits025"] >= strict025
            and result["selected"]["hits050"] >= strict050
        ),
    }
    assert not os.path.exists(args.output_json), args.output_json
    tier3.atomic_json(args.output_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))


def add_locks(parser):
    parser.add_argument("stage31_lock")
    parser.add_argument("stage33_lock")
    parser.add_argument("stage142_lock")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("train_dump")
    add_locks(train_parser)
    train_parser.add_argument("output_dir")
    train_parser.add_argument("--num-threads", type=int, default=16)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("dump")
    add_locks(eval_parser)
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
