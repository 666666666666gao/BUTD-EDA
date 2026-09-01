import hashlib
import json
import os
import sys

import lightgbm as lgb
import numpy as np


package, train_dump, candidate_model, candidate_lock_path, val_dump, out = (
    sys.argv[1:]
)
sys.path.insert(0, package)
import train_joint_option_ranker as ranker  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate_examples(x, ious, groups, baselines, candidate_scores):
    features = []
    baseline_values = []
    candidate_values = []
    candidate_changed = []
    cursor = 0
    for size, baseline_local in zip(groups, baselines):
        size = int(size)
        baseline_local = int(baseline_local)
        end = cursor + size
        local_scores = candidate_scores[cursor:end]
        candidate_local = int(np.argmax(local_scores))
        order = np.argsort(local_scores)[::-1]
        second_local = int(order[1]) if size > 1 else candidate_local
        candidate_feature = x[cursor + candidate_local]
        baseline_feature = x[cursor + baseline_local]
        delta = candidate_feature - baseline_feature
        scalars = np.asarray(
            [
                local_scores[candidate_local],
                local_scores[baseline_local],
                local_scores[candidate_local] - local_scores[baseline_local],
                local_scores[candidate_local] - local_scores[second_local],
                float(local_scores.mean()),
                float(local_scores.std()),
                float(size),
                float(candidate_local != baseline_local),
            ],
            dtype=np.float32,
        )
        features.append(
            np.concatenate(
                [candidate_feature, baseline_feature, delta, scalars], axis=0
            ).astype(np.float32)
        )
        baseline_values.append(float(ious[cursor + baseline_local]))
        candidate_values.append(float(ious[cursor + candidate_local]))
        candidate_changed.append(candidate_local != baseline_local)
        cursor = end
    assert cursor == len(candidate_scores)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(baseline_values, dtype=np.float32),
        np.asarray(candidate_values, dtype=np.float32),
        np.asarray(candidate_changed, dtype=bool),
    )


def action_metrics(
    gate_scores, threshold, baseline_ious, candidate_ious, changed_mask
):
    accepted = changed_mask & (gate_scores >= threshold)
    selected = np.where(accepted, candidate_ious, baseline_ious)
    result = ranker.metrics(selected)
    result.update(
        threshold=float(threshold),
        accepted_ratio=float(np.mean(accepted)),
        fix050=int(np.sum((baseline_ious < 0.50) & (selected >= 0.50))),
        break050=int(np.sum((baseline_ious >= 0.50) & (selected < 0.50))),
        fix025=int(np.sum((baseline_ious < 0.25) & (selected >= 0.25))),
        break025=int(np.sum((baseline_ious >= 0.25) & (selected < 0.25))),
    )
    return result


def choose_threshold(scores, baseline_ious, candidate_ious, changed_mask):
    finite = scores[np.isfinite(scores)]
    thresholds = list(
        np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 401)))
    )
    thresholds.extend([float("-inf"), float("inf")])
    baseline = ranker.metrics(baseline_ious)
    rows = [
        action_metrics(
            scores,
            threshold,
            baseline_ious,
            candidate_ious,
            changed_mask,
        )
        for threshold in thresholds
    ]
    feasible = [
        row for row in rows
        if row["acc025"] >= baseline["acc025"] - 0.001
    ]
    return max(
        feasible,
        key=lambda row: (
            row["acc050"],
            row["acc025"],
            row["mean_iou"],
            -row["accepted_ratio"],
        ),
    )


def predict_gate(learner, kind, features, iteration):
    if kind == "classifier":
        return learner.predict_proba(
            features, num_iteration=iteration
        )[:, 1]
    return learner.predict(features, num_iteration=iteration)


os.makedirs(out, exist_ok=False)
candidate_lock = json.load(open(candidate_lock_path, encoding="utf-8"))
assert sha256(candidate_model) == candidate_lock["model_sha256"]
candidate_booster = lgb.Booster(model_file=candidate_model)

train_rows = ranker.load_rows(train_dump)
group_features, group_ious, metas = ranker.build_dataset(
    train_rows, int(candidate_lock["max_candidates"]), require_scene=True
)
splits = ranker.split_indices(metas)
arrays = {
    split: ranker.materialize(
        group_features, group_ious, metas, splits[split],
        label_mode="binary50",
    )
    for split in ("dev", "test")
}

gate_data = {}
for split in ("dev", "test"):
    x, _, ious, groups, baselines = arrays[split]
    candidate_scores = candidate_booster.predict(
        x, num_iteration=int(candidate_lock["best_iteration"])
    )
    gate_data[split] = gate_examples(
        x, ious, groups, baselines, candidate_scores
    )

dev_x, dev_base, dev_candidate, dev_changed = gate_data["dev"]
test_x, test_base, test_candidate, test_changed = gate_data["test"]
dev_fix = ((dev_base < 0.50) & (dev_candidate >= 0.50)).astype(np.int32)
dev_utility = (
    4.0
    * (
        (dev_candidate >= 0.50).astype(np.float32)
        - (dev_base >= 0.50).astype(np.float32)
    )
    + 1.0
    * (
        (dev_candidate >= 0.25).astype(np.float32)
        - (dev_base >= 0.25).astype(np.float32)
    )
    + 0.1 * (dev_candidate - dev_base)
)

specs = (
    {
        "name": "fix_classifier_d3",
        "kind": "classifier",
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 80,
        "positive_weight": 4.0,
        "break_weight": 4.0,
    },
    {
        "name": "fix_classifier_d5",
        "kind": "classifier",
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 60,
        "positive_weight": 6.0,
        "break_weight": 6.0,
    },
    {
        "name": "fix_classifier_d6",
        "kind": "classifier",
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 40,
        "positive_weight": 8.0,
        "break_weight": 8.0,
    },
    {
        "name": "utility_regression_d3",
        "kind": "regressor",
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 80,
        "positive_weight": 4.0,
        "break_weight": 4.0,
    },
    {
        "name": "utility_regression_d5",
        "kind": "regressor",
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 60,
        "positive_weight": 6.0,
        "break_weight": 6.0,
    },
)

break050 = (dev_base >= 0.50) & (dev_candidate < 0.50)
break025 = (dev_base >= 0.25) & (dev_candidate < 0.25)
candidates = []
for index, spec in enumerate(specs):
    weights = np.ones(len(dev_x), dtype=np.float32)
    weights[dev_fix.astype(bool)] = spec["positive_weight"]
    weights[break050] = spec["break_weight"]
    weights[break025] = max(spec["break_weight"], 10.0)
    common = dict(
        n_estimators=240,
        learning_rate=0.03,
        num_leaves=spec["num_leaves"],
        max_depth=spec["max_depth"],
        min_child_samples=spec["min_child_samples"],
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=index,
        n_jobs=16,
        verbosity=-1,
    )
    if spec["kind"] == "classifier":
        learner = lgb.LGBMClassifier(objective="binary", **common)
        target = dev_fix
    else:
        learner = lgb.LGBMRegressor(objective="huber", **common)
        target = dev_utility
    learner.fit(dev_x, target, sample_weight=weights)
    test_scores = predict_gate(learner, spec["kind"], test_x, 240)
    threshold = choose_threshold(
        test_scores, test_base, test_candidate, test_changed
    )
    candidates.append(
        {
            "spec": spec,
            "threshold": threshold,
            "booster": learner.booster_,
        }
    )
    print(
        "gate_candidate",
        json.dumps(
            {"spec": spec, "threshold": threshold}, sort_keys=True
        ),
        flush=True,
    )

selected_index = max(
    range(len(candidates)),
    key=lambda index: (
        candidates[index]["threshold"]["acc050"],
        candidates[index]["threshold"]["acc025"],
        candidates[index]["threshold"]["mean_iou"],
        -candidates[index]["threshold"]["accepted_ratio"],
    ),
)
selected = candidates[selected_index]
gate_model_path = os.path.join(out, "learned_override_gate.txt")
selected["booster"].save_model(gate_model_path, num_iteration=240)

val_rows = ranker.load_rows(val_dump)
val_features, val_ious, val_metas = ranker.build_dataset(
    val_rows, int(candidate_lock["max_candidates"]), require_scene=False
)
val_x = np.concatenate(val_features, axis=0)
val_iou = np.concatenate(val_ious, axis=0)
val_groups = np.asarray(
    [meta.size for meta in val_metas], dtype=np.int32
)
val_baselines = np.asarray(
    [meta.baseline_index for meta in val_metas], dtype=np.int32
)
val_candidate_scores = candidate_booster.predict(
    val_x, num_iteration=int(candidate_lock["best_iteration"])
)
val_gate_x, val_base, val_candidate, val_changed = gate_examples(
    val_x, val_iou, val_groups, val_baselines, val_candidate_scores
)
gate_booster = lgb.Booster(model_file=gate_model_path)
if selected["spec"]["kind"] == "classifier":
    val_gate_scores = gate_booster.predict(val_gate_x, num_iteration=240)
else:
    val_gate_scores = gate_booster.predict(val_gate_x, num_iteration=240)
val_result = action_metrics(
    val_gate_scores,
    selected["threshold"]["threshold"],
    val_base,
    val_candidate,
    val_changed,
)
baseline = ranker.metrics(val_base)
assert abs(baseline["acc025"] - 0.5477492637778713) < 1e-12
assert abs(baseline["acc050"] - 0.41533445519562473) < 1e-12

lock = {
    "stage": "114",
    "protocol": "train_only_scene_split_learned_override_gate",
    "train_dump": os.path.abspath(train_dump),
    "train_dump_sha256": sha256(train_dump),
    "val_dump_sha256": sha256(val_dump),
    "candidate_model": os.path.abspath(candidate_model),
    "candidate_model_sha256": sha256(candidate_model),
    "candidate_lock_sha256": sha256(candidate_lock_path),
    "candidate_best_iteration": int(candidate_lock["best_iteration"]),
    "gate_model": os.path.abspath(gate_model_path),
    "gate_model_sha256": sha256(gate_model_path),
    "gate_iteration": 240,
    "selected_spec_index": selected_index,
    "selected_spec": selected["spec"],
    "selected_internal_test": selected["threshold"],
    "all_internal_test_candidates": [
        {"spec": item["spec"], "threshold": item["threshold"]}
        for item in candidates
    ],
    "val": {
        "baseline": baseline,
        "selected": val_result,
    },
    "strict_goal": {"acc025_gt": 0.5391, "acc050_gt": 0.4241},
    "strict_goal_met_offline": bool(
        val_result["acc025"] > 0.5391
        and val_result["acc050"] > 0.4241
    ),
    "diagnostic_only_until_integrated_and_reloaded": True,
}
lock_path = os.path.join(out, "locked_learned_override_gate.json")
with open(lock_path, "w", encoding="utf-8") as f:
    json.dump(lock, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(lock, indent=2, sort_keys=True))
