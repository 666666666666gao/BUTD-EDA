#!/usr/bin/env python
"""Train the locked rank-2 rescue selector with clean+augmented train views.

The source option rankers and selector architecture are inherited from the
locked Stage55 protocol.  Augmented views are admitted only for scene-disjoint
selector fitting; early stopping and gate calibration use clean training
views.  ScanRefer validation data is not accepted by this script.
"""

import argparse
import importlib.util
import json
import os
import sys

import lightgbm as lgb
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(HERE, "train_top2_delta_rescue_selector.py")
SPEC = importlib.util.spec_from_file_location("stage55_delta", SOURCE_PATH)
STAGE55 = importlib.util.module_from_spec(SPEC)
sys.modules["stage55_delta"] = STAGE55
SPEC.loader.exec_module(STAGE55)
BASE = STAGE55.BASE


def prepare_rows(rows, top1_module, top2_module, booster1, booster2,
                 lock1, lock2):
    old1 = top1_module.load_rows
    old2 = top2_module.load_rows
    top1_module.load_rows = lambda _path: rows
    top2_module.load_rows = lambda _path: rows
    try:
        return STAGE55.prepare(
            "memory://filtered-train-rows",
            top1_module,
            top2_module,
            booster1,
            booster2,
            lock1,
            lock2,
            require_scene=True,
        )
    finally:
        top1_module.load_rows = old1
        top2_module.load_rows = old2


def filter_source_holdout(rows, top1_module):
    selected = []
    for row in rows:
        scene_id = str(row.get("scene_id", ""))
        if not scene_id:
            raise ValueError("missing scene_id in training row")
        if top1_module.scene_bucket(scene_id) >= 85:
            selected.append(row)
    return selected


def subset_indices(payload, training):
    result = []
    for index, meta in enumerate(payload["metas"]):
        is_train = STAGE55.selector_bucket(meta.scene_id) < 70
        if bool(is_train) == bool(training):
            result.append(index)
    return result


def non_tie_positions(payload, subset):
    candidate_hit = payload["candidate_ious"] >= 0.50
    incumbent_hit = payload["incumbent_ious"] >= 0.50
    delta = candidate_hit.astype(np.int8) - incumbent_hit.astype(np.int8)
    row_to_position = {
        int(row): position
        for position, row in enumerate(payload["candidate_rows"])
    }
    positions = [
        row_to_position[index]
        for index in subset
        if index in row_to_position and delta[index] != 0
    ]
    candidate_rows = payload["candidate_rows"]
    labels = (delta[candidate_rows] > 0).astype(np.int32)
    return positions, labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_train_dump")
    parser.add_argument("augmented_train_dump")
    parser.add_argument("output_dir")
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--top1-script", required=True)
    parser.add_argument("--top2-script", required=True)
    parser.add_argument("--top1-model", required=True)
    parser.add_argument("--top2-model", required=True)
    parser.add_argument("--top1-lock", required=True)
    parser.add_argument("--top2-lock", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(args.output_dir)
    module1, module2, lock1, lock2, booster1, booster2 = (
        BASE.load_sources(args)
    )
    clean_rows = filter_source_holdout(
        module1.load_rows(args.clean_train_dump), module1
    )
    clean = prepare_rows(
        clean_rows, module1, module2, booster1, booster2, lock1, lock2
    )
    augmented_rows = filter_source_holdout(
        module1.load_rows(args.augmented_train_dump), module1
    )
    augmented = prepare_rows(
        augmented_rows, module1, module2, booster1, booster2, lock1, lock2
    )

    clean_train = subset_indices(clean, training=True)
    clean_dev = subset_indices(clean, training=False)
    augmented_train = subset_indices(augmented, training=True)
    train_scenes = {clean["metas"][i].scene_id for i in clean_train}
    train_scenes.update(
        augmented["metas"][i].scene_id for i in augmented_train
    )
    dev_scenes = {clean["metas"][i].scene_id for i in clean_dev}
    if not train_scenes.isdisjoint(dev_scenes):
        raise ValueError("selector train/dev scene leakage")

    clean_train_pos, clean_labels = non_tie_positions(clean, clean_train)
    clean_dev_pos, _ = non_tie_positions(clean, clean_dev)
    aug_train_pos, aug_labels = non_tie_positions(
        augmented, augmented_train
    )
    if not clean_train_pos or not clean_dev_pos or not aug_train_pos:
        raise ValueError("empty non-tie selector split")
    x_train = np.concatenate([
        clean["x"][clean_train_pos],
        augmented["x"][aug_train_pos],
    ], axis=0)
    y_train = np.concatenate([
        clean_labels[clean_train_pos],
        aug_labels[aug_train_pos],
    ], axis=0)
    x_dev = clean["x"][clean_dev_pos]
    y_dev = clean_labels[clean_dev_pos]
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("mixed selector requires both label classes")

    config = {
        "name": "stage55_balanced_locked",
        "num_leaves": 15,
        "max_depth": 6,
        "min_child_samples": 20,
        "reg_lambda": 3.0,
    }
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=config["num_leaves"],
        max_depth=config["max_depth"],
        min_child_samples=config["min_child_samples"],
        reg_lambda=config["reg_lambda"],
        reg_alpha=1.0,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=20260830,
        n_jobs=args.num_threads,
        scale_pos_weight=float(negatives / positives),
        verbosity=-1,
    )
    classifier.fit(
        x_train,
        y_train,
        eval_set=[(x_dev, y_dev)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(60, verbose=True)],
    )
    iteration = int(classifier.best_iteration_ or 500)
    clean_probabilities = classifier.predict_proba(
        clean["x"], num_iteration=iteration
    )[:, 1]
    gate, incumbent_dev = BASE.choose_gate(
        clean, clean_probabilities, clean_dev
    )
    selected_dev, overrides = BASE.apply_gate(
        clean, clean_probabilities, gate["threshold"], subset=clean_dev
    )
    candidate_dev = clean["candidate_ious"][clean_dev]
    incumbent_ious_dev = clean["incumbent_ious"][clean_dev]
    oracle_dev = np.maximum(candidate_dev, incumbent_ious_dev)

    os.makedirs(args.output_dir, exist_ok=False)
    model_path = os.path.join(args.output_dir, "top2_delta_selector.txt")
    classifier.booster_.save_model(model_path, num_iteration=iteration)
    lock = {
        "protocol": "scene_hash_train_only_top2_delta_clean_aug_fit_v1",
        "candidate_selection": "best_scored_match_rank_1_option",
        "training_label": "candidate_hit50_minus_incumbent_hit50_non_ties",
        "feature_augmentation": (
            "candidate_minus_incumbent_and_baseline_plus_train_view"
        ),
        "clean_train_dump": args.clean_train_dump,
        "clean_train_dump_sha256": BASE.sha256(args.clean_train_dump),
        "augmented_train_dump": args.augmented_train_dump,
        "augmented_train_dump_sha256": BASE.sha256(
            args.augmented_train_dump
        ),
        "top1_script": args.top1_script,
        "top1_script_sha256": BASE.sha256(args.top1_script),
        "top2_script": args.top2_script,
        "top2_script_sha256": BASE.sha256(args.top2_script),
        "top1_model": args.top1_model,
        "top1_model_sha256": BASE.sha256(args.top1_model),
        "top2_model": args.top2_model,
        "top2_model_sha256": BASE.sha256(args.top2_model),
        "top1_lock": args.top1_lock,
        "top1_lock_sha256": BASE.sha256(args.top1_lock),
        "top2_lock": args.top2_lock,
        "top2_lock_sha256": BASE.sha256(args.top2_lock),
        "selector_model": model_path,
        "best_iteration": iteration,
        "feature_count": int(clean["x"].shape[1]),
        "gate": gate,
        "selected_config": config,
        "selector_groups": {
            "clean_source_holdout": len(clean_rows),
            "augmented_source_holdout": len(augmented_rows),
            "clean_train": len(clean_train),
            "augmented_train": len(augmented_train),
            "clean_dev": len(clean_dev),
            "clean_non_tie_train": len(clean_train_pos),
            "augmented_non_tie_train": len(aug_train_pos),
            "clean_non_tie_dev": len(clean_dev_pos),
            "mixed_positive_train": positives,
            "clean_positive_dev": int(y_dev.sum()),
        },
        "selector_dev": {
            "incumbent": incumbent_dev,
            "candidate": BASE.metrics(candidate_dev),
            "oracle_upper_bound": BASE.metrics(oracle_dev),
            "selected": BASE.metrics(selected_dev),
            "override_ratio": float(np.mean(overrides)),
        },
    }
    lock_path = os.path.join(
        args.output_dir, "locked_top2_delta_policy.json"
    )
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print(json.dumps(lock, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
