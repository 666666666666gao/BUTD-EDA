import hashlib
import json
import os
import sys

import lightgbm as lgb
import numpy as np


(
    package,
    stage18_train_dump,
    stage95_train_dump,
    stage18_val_dump,
    stage95_val_dump,
    output_dir,
) = sys.argv[1:]
sys.path.insert(0, package)
import train_joint_option_ranker as ranker  # noqa: E402


os.makedirs(output_dir, exist_ok=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entropy_and_margin(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    shifted = values - np.max(values)
    probabilities = np.exp(np.clip(shifted, -50.0, 50.0))
    probabilities /= np.maximum(probabilities.sum(), 1e-8)
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-8)))
    order = np.sort(values)[::-1]
    margin = float(order[0] - order[1]) if len(order) > 1 else 0.0
    return entropy, margin


def detector_match_features(box, row):
    detected = np.asarray(row.get("detected_box", []), dtype=np.float32)
    confidence = np.asarray(
        row.get("detected_target_confidence", []), dtype=np.float32
    )
    if detected.ndim != 2 or detected.shape[-1] != 6 or len(detected) == 0:
        return [0.0] * 10
    if len(confidence) != len(detected):
        confidence = np.zeros(len(detected), dtype=np.float32)
    confidence = np.clip(np.nan_to_num(confidence), 0.0, 1.0)
    support = ranker.pair_iou(box, detected)
    index = int(np.argmax(support))
    matched = detected[index]
    box_size = np.maximum(np.abs(box[3:]), 1e-5)
    matched_size = np.maximum(np.abs(matched[3:]), 1e-5)
    center_delta = (matched[:3] - box[:3]) / (
        0.5 * (matched_size + box_size) + 1e-5
    )
    size_ratio = np.log(matched_size / box_size)
    return [
        float(support[index]),
        float(confidence[index]),
        float(support[index] * confidence[index]),
        *[float(value) for value in center_delta],
        *[float(value) for value in size_ratio],
        float(len(detected) / 80.0),
    ]


def row_representation(row):
    adapter_scores = np.nan_to_num(
        np.asarray(row[ranker.SCORE_KEYS[4]], dtype=np.float32)
    )
    position = int(np.argmax(adapter_scores))
    candidate_queries = list(row["adapter_candidate_query"])
    box = np.asarray(row["adapter_box_at_candidate"], dtype=np.float32)[
        position
    ]
    values = []
    for key in ranker.SCORE_KEYS:
        scores = np.nan_to_num(np.asarray(row[key], dtype=np.float32))
        entropy, margin = entropy_and_margin(scores)
        values.extend(
            [
                float(scores[position]),
                float(np.mean(scores)),
                float(np.std(scores)),
                entropy,
                margin,
            ]
        )
    size = np.maximum(np.abs(box[3:]), 1e-5)
    values.extend([float(value) for value in np.log(size)])
    values.append(float(np.log(np.prod(size) + 1e-8)))
    values.append(float(len(candidate_queries) / 15.0))
    values.append(float(position / max(len(candidate_queries) - 1, 1)))
    values.extend(detector_match_features(box, row))
    values.extend(
        [
            float(row.get("text_target_cid") is not None),
            float(bool(row.get("is_unique_label_only", False))),
            float(str(row.get("decomposition_status", "")) == "ok"),
            float(str(row.get("decomposition_status", "")) == "repaired"),
            float(str(row.get("decomposition_status", "")) == "global_only"),
        ]
    )
    return np.asarray(values, dtype=np.float32), box, position


def paired_examples(stage18_rows, stage95_rows, require_scene):
    assert len(stage18_rows) == len(stage95_rows)
    features = []
    iou18 = []
    iou95 = []
    scenes = []
    for index, (row18, row95) in enumerate(zip(stage18_rows, stage95_rows)):
        assert int(row18.get("example_id", index)) == int(
            row95.get("example_id", index)
        )
        target18 = np.asarray(row18["gt_box"], dtype=np.float32)
        target95 = np.asarray(row95["gt_box"], dtype=np.float32)
        assert np.allclose(target18, target95, atol=1e-6)
        rep18, box18, position18 = row_representation(row18)
        rep95, box95, position95 = row_representation(row95)
        box18_size = np.maximum(np.abs(box18[3:]), 1e-5)
        box95_size = np.maximum(np.abs(box95[3:]), 1e-5)
        pair = np.asarray(
            [
                float(ranker.aligned_iou(box18[None], box95[None])[0]),
                *(
                    (box18[:3] - box95[:3])
                    / (0.5 * (box18_size + box95_size) + 1e-5)
                ).tolist(),
                *np.log(box18_size / box95_size).tolist(),
                float(position18 == position95),
            ],
            dtype=np.float32,
        )
        features.append(
            np.concatenate([rep18, rep95, rep18 - rep95, pair], axis=0)
        )
        iou18.append(
            float(ranker.aligned_iou(box18[None], target18[None])[0])
        )
        iou95.append(
            float(ranker.aligned_iou(box95[None], target95[None])[0])
        )
        scene = str(row95.get("scene_id", index))
        if require_scene:
            assert scene
        scenes.append(scene)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(iou18, dtype=np.float32),
        np.asarray(iou95, dtype=np.float32),
        scenes,
    )


def action_metrics(scores, threshold, iou18, iou95):
    choose18 = scores >= threshold
    selected = np.where(choose18, iou18, iou95)
    result = ranker.metrics(selected)
    result.update(
        threshold=float(threshold),
        choose18_ratio=float(np.mean(choose18)),
        fix050=int(np.sum((iou95 < 0.50) & (selected >= 0.50))),
        break050=int(np.sum((iou95 >= 0.50) & (selected < 0.50))),
        fix025=int(np.sum((iou95 < 0.25) & (selected >= 0.25))),
        break025=int(np.sum((iou95 >= 0.25) & (selected < 0.25))),
    )
    return result


def choose_threshold(scores, iou18, iou95):
    finite = scores[np.isfinite(scores)]
    thresholds = list(
        np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 501)))
    )
    thresholds.extend([float("-inf"), float("inf")])
    baseline = ranker.metrics(iou95)
    rows = [action_metrics(scores, value, iou18, iou95) for value in thresholds]
    feasible = [
        row for row in rows if row["acc025"] >= baseline["acc025"] - 0.001
    ]
    return max(
        feasible,
        key=lambda row: (
            row["acc050"],
            row["acc025"],
            row["mean_iou"],
            -row["choose18_ratio"],
        ),
    )


train18_rows = ranker.load_rows(stage18_train_dump)
train95_rows = ranker.load_rows(stage95_train_dump)
x, iou18, iou95, scenes = paired_examples(
    train18_rows, train95_rows, require_scene=True
)
split_indices = {"train": [], "dev": [], "test": []}
for index, scene in enumerate(scenes):
    bucket = ranker.scene_bucket(scene)
    split = "train" if bucket < 70 else ("dev" if bucket < 85 else "test")
    split_indices[split].append(index)
split_indices = {
    key: np.asarray(value, dtype=np.int64) for key, value in split_indices.items()
}

fix050 = ((iou95 < 0.50) & (iou18 >= 0.50)).astype(np.int32)
beneficial = (
    4.0 * ((iou18 >= 0.50).astype(np.float32) - (iou95 >= 0.50))
    + 1.0 * ((iou18 >= 0.25).astype(np.float32) - (iou95 >= 0.25))
    + 0.1 * (iou18 - iou95)
)
break050 = (iou95 >= 0.50) & (iou18 < 0.50)
break025 = (iou95 >= 0.25) & (iou18 < 0.25)

specs = []
for depth, leaves, child in ((3, 7, 100), (5, 15, 70), (6, 31, 50)):
    specs.append(
        {
            "name": f"fix_classifier_d{depth}",
            "kind": "fix_classifier",
            "max_depth": depth,
            "num_leaves": leaves,
            "min_child_samples": child,
            "positive_weight": 8.0,
            "break_weight": 16.0,
        }
    )
for depth, leaves, child in ((3, 7, 100), (5, 15, 70)):
    specs.append(
        {
            "name": f"benefit_classifier_d{depth}",
            "kind": "benefit_classifier",
            "max_depth": depth,
            "num_leaves": leaves,
            "min_child_samples": child,
            "positive_weight": 6.0,
            "break_weight": 12.0,
        }
    )
    specs.append(
        {
            "name": f"utility_regression_d{depth}",
            "kind": "utility_regression",
            "max_depth": depth,
            "num_leaves": leaves,
            "min_child_samples": child,
            "positive_weight": 6.0,
            "break_weight": 12.0,
        }
    )

train_index = split_indices["train"]
dev_index = split_indices["dev"]
test_index = split_indices["test"]
candidates = []
for index, spec in enumerate(specs):
    weights = np.ones(len(train_index), dtype=np.float32)
    train_fix = fix050[train_index].astype(bool)
    train_break50 = break050[train_index]
    train_break25 = break025[train_index]
    weights[train_fix] = spec["positive_weight"]
    weights[train_break50] = spec["break_weight"]
    weights[train_break25] = max(spec["break_weight"], 24.0)
    common = dict(
        n_estimators=360,
        learning_rate=0.025,
        num_leaves=spec["num_leaves"],
        max_depth=spec["max_depth"],
        min_child_samples=spec["min_child_samples"],
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=index,
        n_jobs=16,
        verbosity=-1,
    )
    if spec["kind"] == "fix_classifier":
        learner = lgb.LGBMClassifier(objective="binary", **common)
        target = fix050[train_index]
    elif spec["kind"] == "benefit_classifier":
        learner = lgb.LGBMClassifier(objective="binary", **common)
        target = (beneficial[train_index] > 0.0).astype(np.int32)
    else:
        learner = lgb.LGBMRegressor(objective="huber", **common)
        target = beneficial[train_index]
    learner.fit(x[train_index], target, sample_weight=weights)
    if spec["kind"].endswith("classifier"):
        dev_scores = learner.predict_proba(x[dev_index])[:, 1]
    else:
        dev_scores = learner.predict(x[dev_index])
    dev = choose_threshold(
        dev_scores, iou18[dev_index], iou95[dev_index]
    )
    candidates.append({"spec": spec, "dev": dev, "booster": learner.booster_})
    print("candidate", json.dumps({"spec": spec, "dev": dev}, sort_keys=True))

selected_index = max(
    range(len(candidates)),
    key=lambda index: (
        candidates[index]["dev"]["acc050"],
        candidates[index]["dev"]["acc025"],
        candidates[index]["dev"]["mean_iou"],
        -candidates[index]["dev"]["choose18_ratio"],
    ),
)
selected = candidates[selected_index]
model_path = os.path.join(output_dir, "cross_checkpoint_safe_gate.txt")
selected["booster"].save_model(model_path, num_iteration=360)
booster = lgb.Booster(model_file=model_path)
test_scores = booster.predict(x[test_index], num_iteration=360)
test_result = action_metrics(
    test_scores,
    selected["dev"]["threshold"],
    iou18[test_index],
    iou95[test_index],
)

val18_rows = ranker.load_rows(stage18_val_dump)
val95_rows = ranker.load_rows(stage95_val_dump)
val_x, val18, val95, _ = paired_examples(
    val18_rows, val95_rows, require_scene=False
)
val_scores = booster.predict(val_x, num_iteration=360)
val_result = action_metrics(
    val_scores, selected["dev"]["threshold"], val18, val95
)
val_baseline = ranker.metrics(val95)
assert abs(val_baseline["acc025"] - 0.5477492637778713) < 1e-12
assert abs(val_baseline["acc050"] - 0.41533445519562473) < 1e-12

lock = {
    "stage": "128",
    "protocol": "train_fit_dev_locked_test_confirmed_cross_checkpoint_safe_gate",
    "inputs": {
        "stage18_train_sha256": sha256(stage18_train_dump),
        "stage95_train_sha256": sha256(stage95_train_dump),
        "stage18_val_sha256": sha256(stage18_val_dump),
        "stage95_val_sha256": sha256(stage95_val_dump),
    },
    "selected_spec_index": selected_index,
    "selected_spec": selected["spec"],
    "selected_dev": selected["dev"],
    "test": test_result,
    "all_dev_candidates": [
        {"spec": item["spec"], "dev": item["dev"]} for item in candidates
    ],
    "gate_model": os.path.abspath(model_path),
    "gate_model_sha256": sha256(model_path),
    "val": {"baseline": val_baseline, "selected": val_result},
    "strict_goal_met_offline": bool(
        val_result["acc025"] > 0.5391 and val_result["acc050"] > 0.4241
    ),
    "diagnostic_only_until_integrated_and_reloaded": True,
}
lock_path = os.path.join(output_dir, "locked_cross_checkpoint_safe_gate.json")
with open(lock_path, "w", encoding="utf-8") as handle:
    json.dump(lock, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(lock, indent=2, sort_keys=True))
