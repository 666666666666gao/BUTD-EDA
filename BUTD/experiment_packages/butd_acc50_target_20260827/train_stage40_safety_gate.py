#!/usr/bin/env python3
"""Train a train-scene-only safety gate for the locked Stage29 ranker.

The gate predicts whether accepting Stage29's non-baseline choice is a @0.50
fix, tie, or break.  It never sees ScanRefer validation during fitting or gate
calibration.  Stage29 remains the default policy; the gate can only reject a
Stage29 change and fall back to the original baseline choice.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import lightgbm as lgb
import numpy as np


META_NAMES = [
    "candidate_z", "baseline_z", "candidate_minus_baseline_z",
    "candidate_rank", "baseline_rank", "locked_score_gap",
    "top_score_margin", "group_size_norm", "stage29_changed",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("joint_ranker_stage40", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def secondary_bucket(scene_id):
    token = (str(scene_id) + "|stage40_safety").encode("utf-8")
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 2


def normalized_rank(values):
    order = np.argsort(-np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float32)
    ranks[order] = np.arange(len(order), dtype=np.float32)
    return ranks / max(1, len(order) - 1)


def safety_feature(x, scores, candidate, baseline):
    x = np.asarray(x, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    z = (scores - scores.mean()) / (scores.std() + 1e-6)
    ranks = normalized_rank(scores)
    top = np.sort(scores)[-2:]
    margin = float(top[-1] - top[-2]) if len(top) > 1 else 0.0
    meta = np.asarray([
        z[candidate], z[baseline], z[candidate] - z[baseline],
        ranks[candidate], ranks[baseline],
        scores[candidate] - scores[baseline], margin,
        min(len(x), 160) / 160.0, float(candidate != baseline),
    ], dtype=np.float32)
    # Only deployable candidate tensors and locked-model scores are used.
    return np.concatenate([
        x[candidate], x[baseline], x[candidate] - x[baseline], meta
    ]).astype(np.float32)


def filter_rows(rows, predicate):
    return [row for row in rows if predicate(str(row.get("scene_id", "")))]


def build_policy_data(joint, rows, lock, booster, require_scene=True):
    gf, gi, metas = joint.build_dataset(
        rows, int(lock["max_candidates"]), require_scene=require_scene
    )
    features = []
    candidate_iou = []
    baseline_iou = []
    changed = []
    scenes = []
    threshold = float(lock["gate"]["threshold"])
    for index, (x, ious, meta) in enumerate(zip(gf, gi, metas)):
        scores = np.asarray(booster.predict(
            x, num_iteration=int(lock["best_iteration"])
        ), dtype=np.float32)
        baseline = int(meta.baseline_index)
        best = int(np.argmax(scores))
        candidate = best if scores[best] - scores[baseline] >= threshold else baseline
        features.append(safety_feature(x, scores, candidate, baseline))
        candidate_iou.append(float(ious[candidate]))
        baseline_iou.append(float(ious[baseline]))
        changed.append(candidate != baseline)
        scenes.append(meta.scene_id)
        if (index + 1) % 1000 == 0:
            print("safety_groups={}/{}".format(index + 1, len(rows)), flush=True)
    return {
        "x": np.stack(features).astype(np.float32),
        "candidate_iou": np.asarray(candidate_iou, dtype=np.float32),
        "baseline_iou": np.asarray(baseline_iou, dtype=np.float32),
        "changed": np.asarray(changed, dtype=bool),
        "scenes": np.asarray(scenes),
    }


def class_labels(data):
    candidate = data["candidate_iou"] >= 0.50
    baseline = data["baseline_iou"] >= 0.50
    delta = candidate.astype(np.int8) - baseline.astype(np.int8)
    return (delta + 1).astype(np.int32)  # break=0, tie=1, fix=2


def stage29_ious(data):
    return np.where(data["changed"], data["candidate_iou"],
                    data["baseline_iou"])


def metrics(ious):
    return {
        "acc025": float(np.mean(ious >= 0.25)),
        "acc050": float(np.mean(ious >= 0.50)),
        "mean_iou": float(np.mean(ious)), "count": int(len(ious)),
    }


def choose_threshold(data, utility):
    default = stage29_ious(data)
    default_metrics = metrics(default)
    active = utility[data["changed"]]
    thresholds = list(np.unique(np.quantile(active, np.linspace(0, 1, 301))))
    thresholds += [float("-inf"), float("inf")]
    rows = []
    for threshold in thresholds:
        accept = data["changed"] & (utility >= threshold)
        selected = np.where(accept, data["candidate_iou"],
                            data["baseline_iou"])
        row = metrics(selected)
        row.update({
            "threshold": float(threshold),
            "accepted_ratio": float(np.mean(accept)),
            "preserves_stage29_acc025": bool(
                row["acc025"] >= default_metrics["acc025"] - 0.001
            ),
        })
        rows.append(row)
    feasible = [row for row in rows if row["preserves_stage29_acc025"]]
    return max(feasible, key=lambda row: (
        row["acc050"], row["acc025"], row["mean_iou"],
        -row["accepted_ratio"],
    ))


def evaluate_data(data, utility, threshold):
    default = stage29_ious(data)
    accept = data["changed"] & (utility >= threshold)
    selected = np.where(accept, data["candidate_iou"], data["baseline_iou"])
    return {
        "selected": metrics(selected), "default_stage29": metrics(default),
        "original_baseline": metrics(data["baseline_iou"]),
        "stage29_changed_ratio": float(np.mean(data["changed"])),
        "accepted_ratio": float(np.mean(accept)),
        "fix25": int(np.sum((data["baseline_iou"] < 0.25) &
                            (selected >= 0.25))),
        "break25": int(np.sum((data["baseline_iou"] >= 0.25) &
                              (selected < 0.25))),
        "fix50": int(np.sum((data["baseline_iou"] < 0.50) &
                            (selected >= 0.50))),
        "break50": int(np.sum((data["baseline_iou"] >= 0.50) &
                              (selected < 0.50))),
    }


def predictor_utility(model, data, iteration):
    probabilities = model.predict_proba(data["x"], num_iteration=iteration)
    return probabilities[:, 2] - probabilities[:, 0]


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    joint = load_module(args.joint_script)
    lock = load_json(args.stage29_lock)
    booster = lgb.Booster(model_file=lock["model_path"])
    clean_rows = joint.load_rows(args.clean_dump)
    aug_rows = joint.load_rows(args.augmented_dump)
    assert len(clean_rows) == len(aug_rows)
    for clean, aug in zip(clean_rows, aug_rows):
        assert (str(clean.get("scene_id")), str(clean.get("ann_id")),
                str(clean.get("object_id"))) == (
                    str(aug.get("scene_id")), str(aug.get("ann_id")),
                    str(aug.get("object_id")))
    is_dev = lambda scene: 70 <= joint.scene_bucket(scene) < 85
    is_test = lambda scene: joint.scene_bucket(scene) >= 85
    clean_dev_rows = filter_rows(clean_rows, is_dev)
    aug_dev_rows = filter_rows(aug_rows, is_dev)
    test_rows = filter_rows(clean_rows, is_test)
    clean_dev = build_policy_data(joint, clean_dev_rows, lock, booster)
    aug_dev = build_policy_data(joint, aug_dev_rows, lock, booster)
    test = build_policy_data(joint, test_rows, lock, booster)
    calib_mask = np.asarray([secondary_bucket(s) == 0 for s in test["scenes"]])
    audit_mask = ~calib_mask
    assert set(test["scenes"][calib_mask]).isdisjoint(
        set(test["scenes"][audit_mask]))

    train_x = np.concatenate([
        clean_dev["x"][clean_dev["changed"]],
        aug_dev["x"][aug_dev["changed"]],
    ])
    train_y = np.concatenate([
        class_labels(clean_dev)[clean_dev["changed"]],
        class_labels(aug_dev)[aug_dev["changed"]],
    ])
    domain_weight = np.concatenate([
        np.ones(int(clean_dev["changed"].sum()), dtype=np.float32),
        np.full(int(aug_dev["changed"].sum()), 0.25, dtype=np.float32),
    ])
    counts = np.bincount(train_y, minlength=3).astype(np.float64)
    class_weight = np.clip(len(train_y) / (3.0 * np.maximum(counts, 1)),
                           0.25, 8.0)
    weights = domain_weight * class_weight[train_y]

    calib = {key: value[calib_mask] for key, value in test.items()}
    audit = {key: value[audit_mask] for key, value in test.items()}
    calib_changed = calib["changed"]
    names = (["candidate_" + n for n in joint.FEATURE_NAMES] +
             ["baseline_" + n for n in joint.FEATURE_NAMES] +
             ["delta_" + n for n in joint.FEATURE_NAMES] + META_NAMES)
    assert train_x.shape[1] == len(names) == 177

    classifier = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=500,
        learning_rate=0.03, num_leaves=15, max_depth=5,
        min_child_samples=50, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=2.0, random_state=args.seed,
        n_jobs=args.num_threads, verbosity=-1,
    )
    classifier.fit(
        train_x, train_y, sample_weight=weights,
        eval_set=[(calib["x"][calib_changed],
                   class_labels(calib)[calib_changed])],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(20)],
        feature_name=names,
    )
    iteration = int(classifier.best_iteration_ or classifier.n_estimators)
    calib_utility = predictor_utility(classifier, calib, iteration)
    gate = choose_threshold(calib, calib_utility)
    audit_utility = predictor_utility(classifier, audit, iteration)
    model_path = os.path.join(args.output_dir, "stage40_safety_gate.txt")
    classifier.booster_.save_model(model_path, num_iteration=iteration)
    receipt = {
        "protocol": "train_scene_disjoint_stage29_break_safety_gate",
        "clean_dump": os.path.abspath(args.clean_dump),
        "clean_dump_sha256": sha256(args.clean_dump),
        "augmented_dump": os.path.abspath(args.augmented_dump),
        "augmented_dump_sha256": sha256(args.augmented_dump),
        "stage29_lock": os.path.abspath(args.stage29_lock),
        "stage29_lock_sha256": sha256(args.stage29_lock),
        "joint_script": os.path.abspath(args.joint_script),
        "joint_script_sha256": sha256(args.joint_script),
        "model_path": os.path.abspath(model_path),
        "model_sha256": sha256(model_path), "best_iteration": iteration,
        "gate": gate, "feature_names": names,
        "train_class_counts": counts.astype(int).tolist(),
        "train_class_weights": class_weight.tolist(),
        "sample_counts": {
            "clean_dev": len(clean_dev_rows), "augmented_dev": len(aug_dev_rows),
            "calib": int(calib_mask.sum()), "audit": int(audit_mask.sum()),
        },
        "scene_counts": {
            "clean_dev": len(set(clean_dev["scenes"])),
            "calib": len(set(calib["scenes"])),
            "audit": len(set(audit["scenes"])),
        },
        "internal": {
            "calib": evaluate_data(calib, calib_utility, gate["threshold"]),
            "audit": evaluate_data(audit, audit_utility, gate["threshold"]),
        },
    }
    lock_path = os.path.join(args.output_dir, "locked_stage40_safety.json")
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def evaluate(args):
    lock = load_json(args.safety_lock)
    joint = load_module(args.joint_script)
    rank_lock = load_json(args.stage29_lock)
    ranker = lgb.Booster(model_file=rank_lock["model_path"])
    rows = joint.load_rows(args.dump)
    data = build_policy_data(joint, rows, rank_lock, ranker,
                             require_scene=False)
    classifier = lgb.LGBMClassifier()
    classifier._Booster = lgb.Booster(model_file=lock["model_path"])
    classifier._n_features = data["x"].shape[1]
    classifier._n_classes = 3
    classifier.fitted_ = True
    # Booster prediction is stable and avoids relying on sklearn wrapper internals.
    probabilities = classifier._Booster.predict(
        data["x"], num_iteration=int(lock["best_iteration"])
    )
    utility = probabilities[:, 2] - probabilities[:, 0]
    result = evaluate_data(data, utility, float(lock["gate"]["threshold"]))
    receipt = {
        "protocol": lock["protocol"], "diagnostic_only": True,
        "dump": os.path.abspath(args.dump), "dump_sha256": sha256(args.dump),
        "safety_lock": os.path.abspath(args.safety_lock),
        "safety_lock_sha256": sha256(args.safety_lock),
        "model_sha256": sha256(lock["model_path"]),
        "threshold": float(lock["gate"]["threshold"]), **result,
    }
    receipt["goal_achieved_offline"] = bool(
        receipt["selected"]["acc025"] > 0.5391 and
        receipt["selected"]["acc050"] > 0.4241
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def self_test():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(7, 56)).astype(np.float32)
    scores = rng.normal(size=7).astype(np.float32)
    feature_a = safety_feature(x, scores, 2, 5)
    fake_iou_a = rng.random(7)
    fake_iou_b = 1.0 - fake_iou_a
    feature_b = safety_feature(x, scores, 2, 5)
    assert np.array_equal(feature_a, feature_b)
    assert not np.array_equal(fake_iou_a, fake_iou_b)
    assert len(feature_a) == 177
    print("STAGE40_SAFETY_GATE_SELFTEST_PASS features=177")


def common(parser):
    parser.add_argument("--joint-script", required=True)
    parser.add_argument("--stage29-lock", required=True)
    parser.add_argument("--dump", required=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    p = sub.add_parser("train")
    p.add_argument("--joint-script", required=True)
    p.add_argument("--stage29-lock", required=True)
    p.add_argument("--clean-dump", required=True)
    p.add_argument("--augmented-dump", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num-threads", type=int, default=16)
    p = sub.add_parser("evaluate"); common(p)
    p.add_argument("--safety-lock", required=True)
    p.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.command == "self-test": self_test()
    elif args.command == "train": train(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
