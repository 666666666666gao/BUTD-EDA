#!/usr/bin/env python
"""Stage55: learn whether explicit rank-2 rescue beats Stage29 at IoU 0.50."""

import hashlib
import importlib.util
import json
import os
import sys

import lightgbm as lgb
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
STAGE53_PATH = os.path.join(HERE, "train_top2_candidate_rescue_selector.py")
SPEC = importlib.util.spec_from_file_location("explicit_top2_rescue", STAGE53_PATH)
STAGE53 = importlib.util.module_from_spec(SPEC)
sys.modules["explicit_top2_rescue"] = STAGE53
SPEC.loader.exec_module(STAGE53)
BASE = STAGE53.BASE


def prepare(dump, top1_module, top2_module, top1_booster, top2_booster,
            top1_lock, top2_lock, require_scene):
    payload = STAGE53.prepare(
        dump, top1_module, top2_module, top1_booster, top2_booster,
        top1_lock, top2_lock, require_scene
    )
    x = payload["x"]
    n2 = len(top2_module.FEATURE_NAMES)
    n1 = len(top1_module.FEATURE_NAMES)
    candidate = x[:, :n2]
    incumbent = x[:, n2:n2 + n1]
    baseline = x[:, n2 + n1:n2 + 2 * n1]
    index2 = {name: index for index, name in enumerate(top2_module.FEATURE_NAMES)}
    common2 = np.asarray(
        [index2[name] for name in top1_module.FEATURE_NAMES], dtype=np.int64
    )
    relative_to_incumbent = candidate[:, common2] - incumbent
    relative_to_baseline = candidate[:, common2] - baseline
    payload["x"] = np.concatenate([
        x, relative_to_incumbent, relative_to_baseline
    ], axis=1).astype(np.float32)
    return payload


def selector_bucket(scene_id):
    token = ("top2_delta_rescue_v1|" + scene_id).encode("utf-8")
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 100


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    module1, module2, lock1, lock2, booster1, booster2 = BASE.load_sources(args)
    payload = prepare(
        args.train_dump, module1, module2, booster1, booster2,
        lock1, lock2, require_scene=True
    )
    source_test = [
        index for index, meta in enumerate(payload["metas"])
        if module1.scene_bucket(meta.scene_id) >= 85
    ]
    selector_train = [
        index for index in source_test
        if selector_bucket(payload["metas"][index].scene_id) < 70
    ]
    selector_train_set = set(selector_train)
    selector_dev = [
        index for index in source_test if index not in selector_train_set
    ]
    train_scenes = {payload["metas"][i].scene_id for i in selector_train}
    dev_scenes = {payload["metas"][i].scene_id for i in selector_dev}
    assert train_scenes.isdisjoint(dev_scenes)
    row_to_position = {
        int(row): position
        for position, row in enumerate(payload["candidate_rows"])
    }
    candidate_hit = payload["candidate_ious"] >= 0.50
    incumbent_hit = payload["incumbent_ious"] >= 0.50
    delta = candidate_hit.astype(np.int8) - incumbent_hit.astype(np.int8)
    train_positions = [
        row_to_position[index] for index in selector_train
        if index in row_to_position and delta[index] != 0
    ]
    dev_positions = [
        row_to_position[index] for index in selector_dev
        if index in row_to_position and delta[index] != 0
    ]
    assert train_positions and dev_positions
    candidate_rows = payload["candidate_rows"]
    labels = (delta[candidate_rows] > 0).astype(np.int32)
    configs = [
        {"name": "small", "num_leaves": 7, "max_depth": 4,
         "min_child_samples": 20, "reg_lambda": 3.0},
        {"name": "balanced", "num_leaves": 15, "max_depth": 6,
         "min_child_samples": 20, "reg_lambda": 3.0},
        {"name": "regularized", "num_leaves": 15, "max_depth": 6,
         "min_child_samples": 40, "reg_lambda": 8.0},
    ]
    candidates = []
    for config_index, config in enumerate(configs):
        train_labels = labels[train_positions]
        positives = int(train_labels.sum())
        negatives = int(len(train_labels) - positives)
        classifier = lgb.LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.025,
            num_leaves=config["num_leaves"], max_depth=config["max_depth"],
            min_child_samples=config["min_child_samples"],
            reg_lambda=config["reg_lambda"], reg_alpha=1.0,
            colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
            random_state=20260829 + config_index, n_jobs=args.num_threads,
            scale_pos_weight=float(negatives / max(positives, 1)),
            verbosity=-1,
        )
        classifier.fit(
            payload["x"][train_positions], train_labels,
            eval_set=[(payload["x"][dev_positions], labels[dev_positions])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(60, verbose=True)],
        )
        iteration = int(classifier.best_iteration_ or 500)
        probabilities = classifier.predict_proba(
            payload["x"], num_iteration=iteration
        )[:, 1]
        gate, incumbent_dev = BASE.choose_gate(
            payload, probabilities, selector_dev
        )
        selected_dev, overrides = BASE.apply_gate(
            payload, probabilities, gate["threshold"], subset=selector_dev
        )
        candidates.append({
            "config": config,
            "iteration": iteration,
            "classifier": classifier,
            "probabilities": probabilities,
            "gate": gate,
            "dev": {
                "incumbent": incumbent_dev,
                "selected": BASE.metrics(selected_dev),
                "override_ratio": float(np.mean(overrides)),
            },
        })
    chosen = max(candidates, key=lambda item: (
        item["dev"]["selected"]["acc050"],
        item["dev"]["selected"]["acc025"],
        item["dev"]["selected"]["mean_iou"],
        -item["dev"]["override_ratio"],
    ))
    model_path = os.path.join(args.output_dir, "top2_delta_selector.txt")
    chosen["classifier"].booster_.save_model(
        model_path, num_iteration=chosen["iteration"]
    )
    candidate_dev = payload["candidate_ious"][selector_dev]
    incumbent_dev = payload["incumbent_ious"][selector_dev]
    oracle_dev = np.maximum(candidate_dev, incumbent_dev)
    lock = {
        "protocol": "scene_hash_train_only_internal_test_top2_delta_v1",
        "candidate_selection": "best_scored_match_rank_1_option",
        "training_label": "candidate_hit50_minus_incumbent_hit50_non_ties",
        "train_dump": args.train_dump,
        "train_dump_sha256": BASE.sha256(args.train_dump),
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
        "best_iteration": chosen["iteration"],
        "feature_count": int(payload["x"].shape[1]),
        "feature_augmentation": "candidate_minus_incumbent_and_baseline",
        "gate": chosen["gate"],
        "selected_config": chosen["config"],
        "all_config_dev": [
            {"config": item["config"], "iteration": item["iteration"],
             "gate": item["gate"], "dev": item["dev"]}
            for item in candidates
        ],
        "selector_groups": {
            "source_internal_test": len(source_test),
            "train": len(selector_train), "dev": len(selector_dev),
            "non_tie_train": len(train_positions),
            "non_tie_dev": len(dev_positions),
            "positive_train": int(labels[train_positions].sum()),
            "positive_dev": int(labels[dev_positions].sum()),
        },
        "selector_dev": {
            "incumbent": BASE.metrics(incumbent_dev),
            "candidate": BASE.metrics(candidate_dev),
            "oracle_upper_bound": BASE.metrics(oracle_dev),
            "selected": chosen["dev"]["selected"],
            "override_ratio": chosen["dev"]["override_ratio"],
        },
    }
    lock_path = os.path.join(args.output_dir, "locked_top2_delta_policy.json")
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print(json.dumps(lock, indent=2, sort_keys=True), flush=True)


def evaluate(args):
    module1, module2, lock1, lock2, booster1, booster2 = BASE.load_sources(args)
    lock = BASE.load_json(args.selector_lock)
    checks = {
        args.top1_script: lock["top1_script_sha256"],
        args.top2_script: lock["top2_script_sha256"],
        args.top1_model: lock["top1_model_sha256"],
        args.top2_model: lock["top2_model_sha256"],
        args.top1_lock: lock["top1_lock_sha256"],
        args.top2_lock: lock["top2_lock_sha256"],
    }
    for path, expected in checks.items():
        assert BASE.sha256(path) == expected, path
    payload = prepare(
        args.dump, module1, module2, booster1, booster2,
        lock1, lock2, require_scene=False
    )
    selector = lgb.Booster(model_file=args.selector_model)
    probabilities = selector.predict(
        payload["x"], num_iteration=int(lock["best_iteration"])
    )
    selected, overrides = BASE.apply_gate(
        payload, probabilities, float(lock["gate"]["threshold"])
    )
    oracle = np.maximum(payload["incumbent_ious"], payload["candidate_ious"])
    result = {
        "diagnostic_only": True,
        "protocol": lock["protocol"],
        "dump": args.dump,
        "dump_sha256": BASE.sha256(args.dump),
        "selector_lock": args.selector_lock,
        "selector_lock_sha256": BASE.sha256(args.selector_lock),
        "selector_model_sha256": BASE.sha256(args.selector_model),
        "incumbent": BASE.metrics(payload["incumbent_ious"]),
        "selected": BASE.metrics(selected),
        "override_ratio": float(np.mean(overrides)),
        "rank2_oracle_upper_bound": BASE.metrics(oracle),
    }
    result["goal_achieved_offline"] = bool(
        result["selected"]["acc025"] > 0.5391
        and result["selected"]["acc050"] > 0.4241
    )
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


ORIGINAL_SELF_TEST = BASE.self_test


def self_test():
    ORIGINAL_SELF_TEST()
    candidate = np.asarray([True, False, True])
    incumbent = np.asarray([False, True, True])
    delta = candidate.astype(np.int8) - incumbent.astype(np.int8)
    assert delta.tolist() == [1, -1, 0]
    print("TOP2_DELTA_RESCUE_SELECTOR_SELFTEST_PASS")


BASE.prepare = prepare
BASE.train = train
BASE.evaluate = evaluate
BASE.self_test = self_test


if __name__ == "__main__":
    BASE.main()
