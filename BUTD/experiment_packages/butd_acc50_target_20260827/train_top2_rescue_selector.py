#!/usr/bin/env python
"""Lock a train-only gate for top-2 detector rescues over Stage29.

The two source rankers are already locked from scene-disjoint partitions of
the ScanRefer training dump.  This selector uses only the source rankers'
previously untouched internal-test scenes, splits those scenes again for
selector fitting/early stopping, and never reads ScanRefer val during lock.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import lightgbm as lgb
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def scene_bucket(scene_id, salt):
    token = (salt + "|" + scene_id).encode("utf-8")
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 100


def metrics(ious):
    ious = np.asarray(ious, dtype=np.float32)
    return {
        "acc025": float(np.mean(ious >= 0.25)),
        "acc050": float(np.mean(ious >= 0.50)),
        "mean_iou": float(np.mean(ious)),
        "count": int(len(ious)),
    }


def assert_aligned(metas_a, metas_b, ious_a, ious_b):
    assert len(metas_a) == len(metas_b)
    for index, (meta_a, meta_b) in enumerate(zip(metas_a, metas_b)):
        assert meta_a.scene_id == meta_b.scene_id, index
        assert meta_a.example_id == meta_b.example_id, index
        base_a = float(ious_a[index][meta_a.baseline_index])
        base_b = float(ious_b[index][meta_b.baseline_index])
        assert abs(base_a - base_b) < 1e-5, (index, base_a, base_b)


def score_groups(module, group_features, metas, booster, lock):
    all_x = np.concatenate(group_features, axis=0)
    all_scores = booster.predict(
        all_x, num_iteration=int(lock["best_iteration"])
    )
    threshold = float(lock["gate"]["threshold"])
    records = []
    cursor = 0
    for x, meta in zip(group_features, metas):
        size = int(meta.size)
        scores = np.asarray(all_scores[cursor:cursor + size], dtype=np.float32)
        baseline = int(meta.baseline_index)
        raw_best = int(np.argmax(scores))
        raw_gap = float(scores[raw_best] - scores[baseline])
        chosen = raw_best if raw_gap >= threshold else baseline
        order = np.argsort(-scores, kind="stable")
        margin = (
            float(scores[order[0]] - scores[order[1]])
            if size > 1 else float("inf")
        )
        mean = float(scores.mean())
        std = float(scores.std() + 1e-6)
        records.append({
            "baseline": baseline,
            "raw_best": raw_best,
            "chosen": int(chosen),
            "raw_gap": raw_gap,
            "margin": margin,
            "best_score": float(scores[raw_best]),
            "baseline_score": float(scores[baseline]),
            "chosen_score": float(scores[chosen]),
            "best_z": float((scores[raw_best] - mean) / std),
            "baseline_z": float((scores[baseline] - mean) / std),
            "size": size,
        })
        cursor += size
    assert cursor == len(all_scores)
    return records


def selector_feature(x_top1, x_top2, rec_top1, rec_top2):
    candidate = np.asarray(x_top2[rec_top2["raw_best"]], dtype=np.float32)
    incumbent = np.asarray(x_top1[rec_top1["chosen"]], dtype=np.float32)
    baseline = np.asarray(x_top1[rec_top1["baseline"]], dtype=np.float32)
    same_query = float(np.allclose(candidate[:18], incumbent[:18], atol=1e-5))
    summary = np.asarray([
        rec_top2["raw_gap"], rec_top2["margin"],
        rec_top2["best_score"], rec_top2["baseline_score"],
        rec_top2["best_z"], rec_top2["baseline_z"],
        rec_top1["raw_gap"], rec_top1["margin"],
        rec_top1["best_score"], rec_top1["baseline_score"],
        rec_top1["chosen_score"], rec_top1["best_z"],
        rec_top1["baseline_z"],
        float(rec_top1["chosen"] == rec_top1["baseline"]),
        same_query,
        float(rec_top2["size"]) / 300.0,
        float(rec_top1["size"]) / 150.0,
    ], dtype=np.float32)
    vector = np.concatenate([candidate, incumbent, baseline, summary])
    return np.nan_to_num(vector, nan=0.0, posinf=20.0, neginf=-20.0)


def prepare(dump, top1_module, top2_module, top1_booster, top2_booster,
            top1_lock, top2_lock, require_scene):
    rows = top1_module.load_rows(dump)
    features1, ious1, metas1 = top1_module.build_dataset(
        rows, int(top1_lock["max_candidates"]), require_scene=require_scene
    )
    features2, ious2, metas2 = top2_module.build_dataset(
        rows, int(top2_lock["max_candidates"]), require_scene=require_scene
    )
    assert_aligned(metas1, metas2, ious1, ious2)
    records1 = score_groups(
        top1_module, features1, metas1, top1_booster, top1_lock
    )
    records2 = score_groups(
        top2_module, features2, metas2, top2_booster, top2_lock
    )
    rank_index = top2_module.FEATURE_NAMES.index("match_rank")
    selector_x = []
    eligible = np.zeros(len(metas1), dtype=bool)
    candidate_ious = np.zeros(len(metas1), dtype=np.float32)
    incumbent_ious = np.zeros(len(metas1), dtype=np.float32)
    candidate_rows = []
    for index, (x1, x2, y1, y2, r1, r2) in enumerate(zip(
            features1, features2, ious1, ious2, records1, records2)):
        incumbent_ious[index] = float(y1[r1["chosen"]])
        raw2 = int(r2["raw_best"])
        candidate_ious[index] = float(y2[raw2])
        is_rank2 = float(x2[raw2, rank_index]) > 0.5
        if is_rank2 and raw2 != r2["baseline"]:
            eligible[index] = True
            candidate_rows.append(index)
            selector_x.append(selector_feature(x1, x2, r1, r2))
    if selector_x:
        selector_x = np.stack(selector_x).astype(np.float32)
    else:
        width = len(top2_module.FEATURE_NAMES) + 2 * len(
            top1_module.FEATURE_NAMES
        ) + 17
        selector_x = np.zeros((0, width), dtype=np.float32)
    return {
        "x": selector_x,
        "candidate_rows": np.asarray(candidate_rows, dtype=np.int64),
        "eligible": eligible,
        "candidate_ious": candidate_ious,
        "incumbent_ious": incumbent_ious,
        "metas": metas1,
    }


def apply_gate(payload, probabilities, threshold, subset=None):
    selected = payload["incumbent_ious"].copy()
    override = np.zeros(len(selected), dtype=bool)
    rows = payload["candidate_rows"]
    accepted = probabilities >= float(threshold)
    override[rows[accepted]] = True
    selected[override] = payload["candidate_ious"][override]
    if subset is None:
        subset = np.arange(len(selected), dtype=np.int64)
    subset = np.asarray(subset, dtype=np.int64)
    return selected[subset], override[subset]


def choose_gate(payload, probabilities, dev_indices):
    dev_indices = np.asarray(dev_indices, dtype=np.int64)
    candidate_probability = {
        int(row): float(probability)
        for row, probability in zip(payload["candidate_rows"], probabilities)
    }
    finite = np.asarray([
        candidate_probability[int(index)]
        for index in dev_indices if int(index) in candidate_probability
    ], dtype=np.float32)
    thresholds = list(np.unique(np.quantile(finite, np.linspace(0, 1, 201))))
    thresholds += [float("-inf"), float("inf")]
    incumbent = metrics(payload["incumbent_ious"][dev_indices])
    rows = []
    all_prob = np.asarray(probabilities, dtype=np.float32)
    for threshold in thresholds:
        selected, overrides = apply_gate(
            payload, all_prob, float(threshold), subset=dev_indices
        )
        result = metrics(selected)
        result["threshold"] = float(threshold)
        result["override_ratio"] = float(np.mean(overrides))
        result["preserves_acc025"] = bool(
            result["acc025"] >= incumbent["acc025"] - 0.001
        )
        rows.append(result)
    feasible = [row for row in rows if row["preserves_acc025"]]
    return max(feasible, key=lambda row: (
        row["acc050"], row["acc025"], row["mean_iou"],
        -row["override_ratio"]
    )), incumbent


def load_sources(args):
    module1 = load_module("ranker_top1", args.top1_script)
    module2 = load_module("ranker_top2", args.top2_script)
    lock1 = load_json(args.top1_lock)
    lock2 = load_json(args.top2_lock)
    assert lock1["feature_names"] == module1.FEATURE_NAMES
    assert lock2["feature_names"] == module2.FEATURE_NAMES
    assert int(lock2.get("match_topk", 0)) == 2
    booster1 = lgb.Booster(model_file=args.top1_model)
    booster2 = lgb.Booster(model_file=args.top2_model)
    return module1, module2, lock1, lock2, booster1, booster2


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    module1, module2, lock1, lock2, booster1, booster2 = load_sources(args)
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
        if scene_bucket(payload["metas"][index].scene_id, "top2_rescue_v1") < 70
    ]
    selector_train_set = set(selector_train)
    selector_dev = [
        index for index in source_test if index not in selector_train_set
    ]
    train_set = set(selector_train)
    dev_set = set(selector_dev)
    row_to_position = {
        int(row): position
        for position, row in enumerate(payload["candidate_rows"])
    }
    train_positions = [
        row_to_position[index] for index in selector_train
        if index in row_to_position
    ]
    dev_positions = [
        row_to_position[index] for index in selector_dev
        if index in row_to_position
    ]
    assert train_positions and dev_positions
    train_scenes = {payload["metas"][i].scene_id for i in train_set}
    dev_scenes = {payload["metas"][i].scene_id for i in dev_set}
    assert train_scenes.isdisjoint(dev_scenes)
    labels = (payload["candidate_ious"][
        payload["candidate_rows"]
    ] >= 0.50).astype(np.int32)
    positives = int(labels[train_positions].sum())
    negatives = int(len(train_positions) - positives)
    scale_pos_weight = float(negatives / max(positives, 1))
    classifier = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.03,
        num_leaves=15, max_depth=6, min_child_samples=80,
        reg_lambda=2.0, colsample_bytree=0.8, subsample=0.8,
        subsample_freq=1, random_state=20260829, n_jobs=args.num_threads,
        scale_pos_weight=scale_pos_weight, verbosity=-1,
    )
    classifier.fit(
        payload["x"][train_positions], labels[train_positions],
        eval_set=[(payload["x"][dev_positions], labels[dev_positions])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=True)],
    )
    iteration = int(classifier.best_iteration_ or 500)
    probabilities = classifier.predict_proba(
        payload["x"], num_iteration=iteration
    )[:, 1]
    gate, dev_incumbent = choose_gate(payload, probabilities, selector_dev)
    selected_dev, override_dev = apply_gate(
        payload, probabilities, gate["threshold"], subset=selector_dev
    )
    candidate_dev = payload["candidate_ious"][selector_dev]
    incumbent_dev = payload["incumbent_ious"][selector_dev]
    oracle_dev = np.maximum(candidate_dev, incumbent_dev)
    model_path = os.path.join(args.output_dir, "top2_rescue_selector.txt")
    classifier.booster_.save_model(model_path, num_iteration=iteration)
    lock_path = os.path.join(args.output_dir, "locked_top2_rescue_policy.json")
    lock = {
        "protocol": "scene_hash_train_only_internal_test_top2_rescue_v1",
        "train_dump": args.train_dump,
        "train_dump_sha256": sha256(args.train_dump),
        "top1_script": args.top1_script,
        "top1_script_sha256": sha256(args.top1_script),
        "top2_script": args.top2_script,
        "top2_script_sha256": sha256(args.top2_script),
        "top1_model": args.top1_model,
        "top1_model_sha256": sha256(args.top1_model),
        "top2_model": args.top2_model,
        "top2_model_sha256": sha256(args.top2_model),
        "top1_lock": args.top1_lock,
        "top1_lock_sha256": sha256(args.top1_lock),
        "top2_lock": args.top2_lock,
        "top2_lock_sha256": sha256(args.top2_lock),
        "selector_model": model_path,
        "best_iteration": iteration,
        "feature_count": int(payload["x"].shape[1]),
        "gate": gate,
        "selector_groups": {
            "source_internal_test": len(source_test),
            "train": len(selector_train), "dev": len(selector_dev),
            "eligible_train": len(train_positions),
            "eligible_dev": len(dev_positions),
        },
        "selector_dev": {
            "incumbent": dev_incumbent,
            "selected": metrics(selected_dev),
            "override_ratio": float(np.mean(override_dev)),
            "candidate": metrics(candidate_dev),
            "oracle_upper_bound": metrics(oracle_dev),
        },
    }
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print(json.dumps(lock, indent=2, sort_keys=True), flush=True)


def evaluate(args):
    module1, module2, lock1, lock2, booster1, booster2 = load_sources(args)
    selector_lock = load_json(args.selector_lock)
    checks = {
        args.top1_script: selector_lock["top1_script_sha256"],
        args.top2_script: selector_lock["top2_script_sha256"],
        args.top1_model: selector_lock["top1_model_sha256"],
        args.top2_model: selector_lock["top2_model_sha256"],
        args.top1_lock: selector_lock["top1_lock_sha256"],
        args.top2_lock: selector_lock["top2_lock_sha256"],
    }
    for path, expected in checks.items():
        assert sha256(path) == expected, path
    payload = prepare(
        args.dump, module1, module2, booster1, booster2,
        lock1, lock2, require_scene=False
    )
    selector = lgb.Booster(model_file=args.selector_model)
    probabilities = selector.predict(
        payload["x"], num_iteration=int(selector_lock["best_iteration"])
    )
    selected, overrides = apply_gate(
        payload, probabilities, float(selector_lock["gate"]["threshold"])
    )
    oracle = np.maximum(payload["incumbent_ious"], payload["candidate_ious"])
    result = {
        "diagnostic_only": True,
        "protocol": selector_lock["protocol"],
        "dump": args.dump,
        "dump_sha256": sha256(args.dump),
        "selector_lock": args.selector_lock,
        "selector_lock_sha256": sha256(args.selector_lock),
        "selector_model_sha256": sha256(args.selector_model),
        "incumbent": metrics(payload["incumbent_ious"]),
        "selected": metrics(selected),
        "override_ratio": float(np.mean(overrides)),
        "eligible_ratio": float(np.mean(payload["eligible"])),
        "rank2_oracle_upper_bound": metrics(oracle),
    }
    result["goal_achieved_offline"] = bool(
        result["selected"]["acc025"] > 0.5391
        and result["selected"]["acc050"] > 0.4241
    )
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def self_test():
    values = np.asarray([0.2, 0.7, 0.4], dtype=np.float32)
    result = metrics(values)
    assert result["acc025"] == 2 / 3
    assert result["acc050"] == 1 / 3
    assert scene_bucket("scene0000_00", "a") == scene_bucket(
        "scene0000_00", "a"
    )
    print("TOP2_RESCUE_SELECTOR_SELFTEST_PASS")


def add_sources(parser):
    parser.add_argument("--top1-script", required=True)
    parser.add_argument("--top2-script", required=True)
    parser.add_argument("--top1-model", required=True)
    parser.add_argument("--top2-model", required=True)
    parser.add_argument("--top1-lock", required=True)
    parser.add_argument("--top2-lock", required=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("train_dump")
    train_parser.add_argument("output_dir")
    train_parser.add_argument("--num-threads", type=int, default=16)
    add_sources(train_parser)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("dump")
    eval_parser.add_argument("selector_model")
    eval_parser.add_argument("selector_lock")
    eval_parser.add_argument("output_json")
    add_sources(eval_parser)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
