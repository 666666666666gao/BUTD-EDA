import hashlib
import json
import os
import sys

import lightgbm as lgb
import numpy as np


package, train_dump, val_dump, out = sys.argv[1:]
sys.path.insert(0, package)
import train_joint_option_ranker as ranker  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_weights(ious, groups, baselines, hard_weight):
    weights = np.ones(len(ious), dtype=np.float32)
    cursor = 0
    hard_groups = 0
    repairable_groups = 0
    for size, baseline in zip(groups, baselines):
        size = int(size)
        baseline = int(baseline)
        end = cursor + size
        baseline_iou = float(ious[cursor + baseline])
        repairable = bool(np.any(ious[cursor:end] >= 0.50))
        if baseline_iou < 0.50:
            hard_groups += 1
            if repairable:
                repairable_groups += 1
                weights[cursor:end] = float(hard_weight)
        cursor = end
    assert cursor == len(ious)
    return weights, hard_groups, repairable_groups


def evaluate(booster, iteration, x, ious, groups, baselines, threshold):
    scores = booster.predict(x, num_iteration=iteration)
    selected, baseline_ious, gaps = ranker.group_decisions(
        scores, ious, groups, baselines, threshold
    )
    result = {
        "baseline": ranker.metrics(baseline_ious),
        "selected": ranker.metrics(selected),
        "changed_ratio": float(np.mean(gaps >= threshold)),
        "fix050": int(
            np.sum((baseline_ious < 0.50) & (selected >= 0.50))
        ),
        "break050": int(
            np.sum((baseline_ious >= 0.50) & (selected < 0.50))
        ),
        "fix025": int(
            np.sum((baseline_ious < 0.25) & (selected >= 0.25))
        ),
        "break025": int(
            np.sum((baseline_ious >= 0.25) & (selected < 0.25))
        ),
    }
    return result


os.makedirs(out, exist_ok=False)
rows = ranker.load_rows(train_dump)
group_features, group_ious, metas = ranker.build_dataset(
    rows, 8, require_scene=True
)
splits = ranker.split_indices(metas)
arrays = {
    split: ranker.materialize(
        group_features,
        group_ious,
        metas,
        splits[split],
        label_mode="binary50",
    )
    for split in ("train", "dev", "test")
}
x_train, y_train, iou_train, groups_train, baseline_train = arrays["train"]
x_dev, y_dev, iou_dev, groups_dev, baseline_dev = arrays["dev"]
x_test, _, iou_test, groups_test, baseline_test = arrays["test"]

hard_weights = (1.0, 2.0, 4.0, 8.0, 16.0)
candidates = []
for config_index, hard_weight in enumerate(hard_weights):
    sample_weight, hard_count, repairable_count = group_weights(
        iou_train, groups_train, baseline_train, hard_weight
    )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=config_index,
        n_jobs=16,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        group=groups_train.tolist(),
        eval_set=[(x_dev, y_dev)],
        eval_group=[groups_dev.tolist()],
        eval_at=[1],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)],
        feature_name=ranker.FEATURE_NAMES,
    )
    iteration = int(model.best_iteration_ or model.n_estimators)
    dev_scores = model.booster_.predict(x_dev, num_iteration=iteration)
    gate, _ = ranker.choose_gate(
        dev_scores, iou_dev, groups_dev, baseline_dev
    )
    test_result = evaluate(
        model.booster_,
        iteration,
        x_test,
        iou_test,
        groups_test,
        baseline_test,
        float(gate["threshold"]),
    )
    candidate = {
        "hard_weight": hard_weight,
        "hard_train_groups": hard_count,
        "repairable_hard_train_groups": repairable_count,
        "iteration": iteration,
        "dev_gate": gate,
        "internal_test": test_result,
        "booster": model.booster_,
    }
    candidates.append(candidate)
    print(
        "hardfocus_candidate",
        json.dumps(
            {key: value for key, value in candidate.items() if key != "booster"},
            sort_keys=True,
        ),
        flush=True,
    )

selected_index = max(
    range(len(candidates)),
    key=lambda index: (
        candidates[index]["internal_test"]["selected"]["acc050"],
        candidates[index]["internal_test"]["selected"]["acc025"],
        candidates[index]["internal_test"]["selected"]["mean_iou"],
        -candidates[index]["internal_test"]["changed_ratio"],
    ),
)
selected = candidates[selected_index]
model_path = os.path.join(out, "hardfocus_binary50_ranker.txt")
selected["booster"].save_model(
    model_path, num_iteration=selected["iteration"]
)

val_rows = ranker.load_rows(val_dump)
val_group_features, val_group_ious, val_metas = ranker.build_dataset(
    val_rows, 8, require_scene=False
)
val_x = np.concatenate(val_group_features, axis=0)
val_ious = np.concatenate(val_group_ious, axis=0)
val_groups = np.asarray([meta.size for meta in val_metas], dtype=np.int32)
val_baselines = np.asarray(
    [meta.baseline_index for meta in val_metas], dtype=np.int32
)
val_result = evaluate(
    selected["booster"],
    selected["iteration"],
    val_x,
    val_ious,
    val_groups,
    val_baselines,
    float(selected["dev_gate"]["threshold"]),
)
assert abs(val_result["baseline"]["acc025"] - 0.5477492637778713) < 1e-12
assert abs(val_result["baseline"]["acc050"] - 0.41533445519562473) < 1e-12

receipt_candidates = [
    {key: value for key, value in candidate.items() if key != "booster"}
    for candidate in candidates
]
receipt = {
    "stage": "115",
    "protocol": "train_only_scene_split_repairable_hard_group_weighting",
    "train_dump": os.path.abspath(train_dump),
    "train_dump_sha256": sha256(train_dump),
    "val_dump": os.path.abspath(val_dump),
    "val_dump_sha256": sha256(val_dump),
    "hard_weights_predeclared": list(hard_weights),
    "selected_index": selected_index,
    "selected_hard_weight": selected["hard_weight"],
    "selected_iteration": selected["iteration"],
    "selected_dev_gate": selected["dev_gate"],
    "all_internal_candidates": receipt_candidates,
    "model": os.path.abspath(model_path),
    "model_sha256": sha256(model_path),
    "val": val_result,
    "strict_goal": {"acc025_gt": 0.5391, "acc050_gt": 0.4241},
    "strict_goal_met_offline": bool(
        val_result["selected"]["acc025"] > 0.5391
        and val_result["selected"]["acc050"] > 0.4241
    ),
    "diagnostic_only_until_integrated_and_reloaded": True,
}
with open(
    os.path.join(out, "locked_hardfocus_policy.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
