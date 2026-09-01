#!/usr/bin/env python
"""Refit a locked train-only LightGBM option ranker on all training scenes.

Hyperparameters, iteration count, feature schema, and the conservative gate
threshold come from a policy selected entirely inside the ScanRefer training
split.  This script does not inspect or accept validation data.
"""

import argparse
import json
import os

import lightgbm as lgb

import train_joint_option_ranker as joint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dump")
    parser.add_argument("source_model")
    parser.add_argument("source_lock")
    parser.add_argument("output_dir")
    parser.add_argument("--num-threads", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(args.output_dir)
    source_lock = json.load(open(args.source_lock, "r", encoding="utf-8"))
    if joint.sha256(args.source_model) != source_lock["model_sha256"]:
        raise ValueError("source model hash does not match locked policy")
    if source_lock["feature_names"] != joint.FEATURE_NAMES:
        raise ValueError("feature schema differs from locked policy")
    if list(source_lock["actions"]) != list(joint.ACTIONS):
        raise ValueError("action schema differs from locked policy")
    if list(source_lock["match_powers"]) != list(joint.MATCH_POWERS):
        raise ValueError("detector-match schema differs from locked policy")

    config = dict(source_lock["selected_config"])
    iteration = int(source_lock["best_iteration"])
    threshold = float(source_lock["gate"]["threshold"])
    max_candidates = int(source_lock["max_candidates"])

    rows = joint.load_rows(args.train_dump)
    group_features, group_ious, metas = joint.build_dataset(
        rows, max_candidates=max_candidates, require_scene=True
    )
    indices = list(range(len(metas)))
    x_train, y_train, iou_train, groups_train, baseline_train = (
        joint.materialize(
            group_features, group_ious, metas, indices,
            label_mode="binary50",
        )
    )

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=iteration,
        learning_rate=0.05,
        num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]),
        min_child_samples=int(config["min_child_samples"]),
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=int(source_lock["selected_config_index"]),
        n_jobs=args.num_threads,
        verbosity=-1,
    )
    ranker.fit(
        x_train,
        y_train,
        group=groups_train.tolist(),
        feature_name=joint.FEATURE_NAMES,
        callbacks=[lgb.log_evaluation(20)],
    )

    os.makedirs(args.output_dir, exist_ok=False)
    model_path = os.path.join(
        args.output_dir, "binary50_option_ranker_full_train.txt"
    )
    ranker.booster_.save_model(model_path, num_iteration=iteration)
    train_audit = joint.evaluate_split(
        ranker.booster_, x_train, iou_train, groups_train, baseline_train,
        threshold, iteration,
    )
    receipt = {
        "protocol": "scanrefer_train_only_locked_config_full_refit_v1",
        "validation_used_for_training_or_selection": False,
        "train_dump": os.path.abspath(args.train_dump),
        "train_dump_sha256": joint.sha256(args.train_dump),
        "source_model": os.path.abspath(args.source_model),
        "source_model_sha256": joint.sha256(args.source_model),
        "source_lock": os.path.abspath(args.source_lock),
        "source_lock_sha256": joint.sha256(args.source_lock),
        "feature_names": joint.FEATURE_NAMES,
        "max_candidates": max_candidates,
        "match_powers": list(joint.MATCH_POWERS),
        "actions": list(joint.ACTIONS),
        "selected_config_index": int(source_lock["selected_config_index"]),
        "selected_config": config,
        "best_iteration": iteration,
        "gate": dict(source_lock["gate"]),
        "full_train_group_count": len(metas),
        "full_train_option_count": int(x_train.shape[0]),
        "full_train_scene_count": len({meta.scene_id for meta in metas}),
        "full_train_audit": train_audit,
        "model_path": os.path.abspath(model_path),
    }
    receipt["model_sha256"] = joint.sha256(model_path)
    lock_path = os.path.join(
        args.output_dir, "locked_binary50_full_train_policy.json"
    )
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
