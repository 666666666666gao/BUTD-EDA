#!/usr/bin/env python3
"""Same-domain tier-3 option ranker for Stage135c BUTD predictions.

The label hierarchy is aligned with the two ScanRefer metrics:
Tier 2 (IoU >= .50) > Tier 1 (.25 <= IoU < .50) > Tier 0 (IoU < .25).
Only ScanRefer train scenes choose the model iteration and safety gate.
"""

import argparse
import json
import math
import os

import lightgbm as lgb
import numpy as np

import train_joint_option_ranker as option


STAGE = "162_stage135c_same_domain_tier3_option_ranker"


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def tier3_labels(ious):
    ious = np.asarray(ious, dtype=np.float32)
    labels = np.zeros(len(ious), dtype=np.int32)
    labels[ious >= 0.25] = 1
    labels[ious >= 0.50] = 2
    return labels


def summarize(ious):
    ious = np.asarray(ious, dtype=np.float32)
    return {
        "count": int(len(ious)),
        "hits025": int((ious >= 0.25).sum()),
        "hits050": int((ious >= 0.50).sum()),
        "acc025": float((ious >= 0.25).mean()),
        "acc050": float((ious >= 0.50).mean()),
        "mean_iou": float(ious.mean()),
    }


def evaluate_scores(scores, ious, groups, baselines, threshold):
    selected, baseline, gaps = option.group_decisions(
        scores, ious, groups, baselines, float(threshold)
    )
    return {
        "selected": summarize(selected),
        "baseline": summarize(baseline),
        "changed_ratio": float(np.mean(gaps >= float(threshold))),
        "score_gap_mean": float(np.mean(gaps)),
        "selected_ious": selected,
    }


def internal_gate_pass(result):
    return bool(
        result["selected"]["acc050"]
        >= result["baseline"]["acc050"] + 0.01 - 1e-12
        and result["selected"]["acc025"]
        >= result["baseline"]["acc025"] - 0.002 - 1e-12
        and result["changed_ratio"] <= 0.98
    )


def materialize_tier3(group_features, group_ious, metas, indices):
    x, _, ious, groups, baselines = option.materialize(
        group_features, group_ious, metas, indices
    )
    return x, tier3_labels(ious), ious, groups, baselines


def make_ranker(num_threads):
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 8],
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=16201,
        n_jobs=int(num_threads),
        verbosity=-1,
    )


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
        split: materialize_tier3(
            group_features, group_ious, metas, splits[split]
        )
        for split in ("train", "dev", "test")
    }
    x_train, y_train, _, groups_train, _ = arrays["train"]
    x_dev, y_dev, iou_dev, groups_dev, baseline_dev = arrays["dev"]
    ranker = make_ranker(args.num_threads)
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
    dev_scores = ranker.booster_.predict(x_dev, num_iteration=iteration)
    gate, _ = option.choose_gate(
        dev_scores, iou_dev, groups_dev, baseline_dev
    )
    dev_result = evaluate_scores(
        dev_scores, iou_dev, groups_dev, baseline_dev, gate["threshold"]
    )
    dev_result.pop("selected_ious")
    x_test, _, iou_test, groups_test, baseline_test = arrays["test"]
    test_scores = ranker.booster_.predict(x_test, num_iteration=iteration)
    test_result = evaluate_scores(
        test_scores, iou_test, groups_test, baseline_test, gate["threshold"]
    )
    test_result.pop("selected_ious")
    authorized = internal_gate_pass(test_result)
    model_path = os.path.join(args.output_dir, "tier3_option_ranker.txt")
    ranker.booster_.save_model(model_path, num_iteration=iteration)
    split_scenes = {
        split: sorted({metas[index].scene_id for index in indices})
        for split, indices in splits.items()
    }
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_tier3_option_lock",
        "protocol": (
            "stage135c_same_domain_scene70_train_scene15_dev_"
            "scene15_test_tier3_gain_0_1_8_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": authorized,
        "internal_gate_pass": authorized,
        "script": os.path.abspath(__file__),
        "script_sha256": option.sha256(os.path.abspath(__file__)),
        "train_dump": os.path.abspath(args.train_dump),
        "train_dump_sha256": option.sha256(args.train_dump),
        "model_path": os.path.abspath(model_path),
        "model_sha256": option.sha256(model_path),
        "best_iteration": iteration,
        "max_candidates": 8,
        "match_powers": list(option.MATCH_POWERS),
        "actions": list(option.ACTIONS),
        "feature_names": option.FEATURE_NAMES,
        "tier_boundaries": [0.25, 0.50],
        "label_gain": [0, 1, 8],
        "gate": gate,
        "internal": {"dev": dev_result, "test": test_result},
        "split_group_counts": {
            split: len(indices) for split, indices in splits.items()
        },
        "split_scene_counts": {
            split: len(scenes) for split, scenes in split_scenes.items()
        },
    }
    lock_path = os.path.join(args.output_dir, "locked_tier3_option_policy.json")
    atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": option.sha256(lock_path),
        "best_iteration": iteration,
        "gate": gate,
        "internal": lock["internal"],
        "internal_gate_pass": authorized,
        "validation_evaluation_authorized": authorized,
    }, indent=2, sort_keys=True))


def subset_summary(selected, mask):
    mask = np.asarray(mask, dtype=bool)
    assert len(mask) == len(selected) and int(mask.sum()) > 0
    return summarize(np.asarray(selected)[mask])


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
    booster = lgb.Booster(model_file=lock["model_path"])
    scores = booster.predict(x, num_iteration=int(lock["best_iteration"]))
    metrics = evaluate_scores(
        scores, ious, groups, baselines, lock["gate"]["threshold"]
    )
    selected = metrics.pop("selected_ious")
    unique = np.asarray([
        bool(row.get("is_unique_label_only", False)) for row in rows
    ])
    metrics["unique"] = subset_summary(selected, unique)
    metrics["multiple"] = subset_summary(selected, ~unique)
    count = len(rows)
    strict025 = math.floor(0.5391 * count) + 1
    strict050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "162_tier3_option_ranker_validation_eval",
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": option.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "metrics": metrics,
        "strict_goal_hits": {"acc025": strict025, "acc050": strict050},
        "strict_goal_met_offline": bool(
            metrics["selected"]["hits025"] >= strict025
            and metrics["selected"]["hits050"] >= strict050
        ),
    }
    assert not os.path.exists(args.output_json), args.output_json
    atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("train_dump")
    train_parser.add_argument("output_dir")
    train_parser.add_argument("--num-threads", type=int, default=16)
    eval_parser = sub.add_parser("evaluate")
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
