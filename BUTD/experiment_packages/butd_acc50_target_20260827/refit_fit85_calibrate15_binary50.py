#!/usr/bin/env python
"""Fit a locked option ranker on 85% of train scenes and calibrate on 15%.

The model configuration, feature schema, and iteration count are inherited
from the previously locked train-only policy.  The held-out 15% of ScanRefer
training scenes is used exactly once to set the conservative fallback gate.
No ScanRefer validation example is accepted by this script.
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
    max_candidates = int(source_lock["max_candidates"])
    rows = joint.load_rows(args.train_dump)
    group_features, group_ious, metas = joint.build_dataset(
        rows, max_candidates=max_candidates, require_scene=True
    )
    splits = joint.split_indices(metas)
    fit_indices = list(splits["train"]) + list(splits["dev"])
    calibration_indices = list(splits["test"])
    fit_scenes = {metas[index].scene_id for index in fit_indices}
    calibration_scenes = {
        metas[index].scene_id for index in calibration_indices
    }
    if not fit_scenes.isdisjoint(calibration_scenes):
        raise ValueError("fit/calibration scene leakage")

    x_fit, y_fit, iou_fit, groups_fit, baseline_fit = joint.materialize(
        group_features, group_ious, metas, fit_indices,
        label_mode="binary50",
    )
    x_cal, _, iou_cal, groups_cal, baseline_cal = joint.materialize(
        group_features, group_ious, metas, calibration_indices,
        label_mode="binary50",
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
        x_fit,
        y_fit,
        group=groups_fit.tolist(),
        feature_name=joint.FEATURE_NAMES,
        callbacks=[lgb.log_evaluation(20)],
    )
    calibration_scores = ranker.booster_.predict(
        x_cal, num_iteration=iteration
    )
    gate, calibration_baseline = joint.choose_gate(
        calibration_scores, iou_cal, groups_cal, baseline_cal
    )
    calibration_audit = joint.evaluate_split(
        ranker.booster_, x_cal, iou_cal, groups_cal, baseline_cal,
        gate["threshold"], iteration,
    )
    fit_audit = joint.evaluate_split(
        ranker.booster_, x_fit, iou_fit, groups_fit, baseline_fit,
        gate["threshold"], iteration,
    )

    os.makedirs(args.output_dir, exist_ok=False)
    model_path = os.path.join(
        args.output_dir, "binary50_option_ranker_fit85.txt"
    )
    ranker.booster_.save_model(model_path, num_iteration=iteration)
    receipt = {
        "protocol": "scanrefer_train_only_scene_hash_fit85_gate15_v1",
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
        "gate": gate,
        "fit_group_count": len(fit_indices),
        "fit_scene_count": len(fit_scenes),
        "calibration_group_count": len(calibration_indices),
        "calibration_scene_count": len(calibration_scenes),
        "fit_audit": fit_audit,
        "calibration_baseline": calibration_baseline,
        "calibration_audit": calibration_audit,
        "model_path": os.path.abspath(model_path),
    }
    receipt["model_sha256"] = joint.sha256(model_path)
    lock_path = os.path.join(
        args.output_dir, "locked_binary50_fit85_policy.json"
    )
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
