import json
import os
import sys

import numpy as np


package, train_dump, val_dump, output_dir = sys.argv[1:]
sys.path.insert(0, package)
import train_joint_option_ranker as ranker  # noqa: E402


os.makedirs(output_dir, exist_ok=False)


def selected_boxes(rows):
    predicted = []
    target = []
    scene_ids = []
    score_key = ranker.SCORE_KEYS[4]
    for index, row in enumerate(rows):
        scores = np.nan_to_num(np.asarray(row[score_key], dtype=np.float32))
        position = int(np.argmax(scores))
        boxes = np.asarray(row["adapter_box_at_candidate"], dtype=np.float32)
        predicted.append(boxes[position])
        target.append(np.asarray(row["gt_box"], dtype=np.float32))
        scene_ids.append(str(row.get("scene_id", index)))
    return (
        np.asarray(predicted, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        scene_ids,
    )


def scaled_ious(predicted, target, scale_xyz):
    boxes = predicted.copy()
    boxes[:, 3:] *= np.asarray(scale_xyz, dtype=np.float32)[None]
    boxes[:, 3:] = np.maximum(np.abs(boxes[:, 3:]), 1e-5)
    return ranker.aligned_iou(boxes, target)


def metrics_for(predicted, target, indices, scale_xyz):
    return ranker.metrics(
        scaled_ious(predicted[indices], target[indices], scale_xyz)
    )


train_rows = ranker.load_rows(train_dump)
train_pred, train_target, train_scenes = selected_boxes(train_rows)
split_indices = {"train": [], "dev": [], "test": []}
for index, scene_id in enumerate(train_scenes):
    bucket = ranker.scene_bucket(scene_id)
    split = "train" if bucket < 70 else ("dev" if bucket < 85 else "test")
    split_indices[split].append(index)
split_indices = {
    key: np.asarray(value, dtype=np.int64) for key, value in split_indices.items()
}

policies = []
isotropic = np.round(np.arange(0.70, 1.401, 0.025), 3)
for scale in isotropic:
    scale_xyz = [float(scale)] * 3
    policies.append(
        {
            "family": "isotropic",
            "scale_xyz": scale_xyz,
            "dev": metrics_for(
                train_pred, train_target, split_indices["dev"], scale_xyz
            ),
            "test": metrics_for(
                train_pred, train_target, split_indices["test"], scale_xyz
            ),
        }
    )

horizontal = np.round(np.arange(0.80, 1.201, 0.05), 3)
vertical = np.round(np.arange(0.75, 1.251, 0.05), 3)
for xy_scale in horizontal:
    for z_scale in vertical:
        scale_xyz = [float(xy_scale), float(xy_scale), float(z_scale)]
        policies.append(
            {
                "family": "horizontal_vertical",
                "scale_xyz": scale_xyz,
                "dev": metrics_for(
                    train_pred, train_target, split_indices["dev"], scale_xyz
                ),
                "test": metrics_for(
                    train_pred, train_target, split_indices["test"], scale_xyz
                ),
            }
        )

baseline = next(
    row for row in policies if row["scale_xyz"] == [1.0, 1.0, 1.0]
)
feasible = [
    row
    for row in policies
    if row["dev"]["acc025"] >= baseline["dev"]["acc025"] - 0.001
]
selected = max(
    feasible,
    key=lambda row: (
        row["dev"]["acc050"],
        row["dev"]["acc025"],
        row["dev"]["mean_iou"],
        -sum(abs(value - 1.0) for value in row["scale_xyz"]),
    ),
)

val_rows = ranker.load_rows(val_dump)
val_pred, val_target, _ = selected_boxes(val_rows)
val_baseline_ious = scaled_ious(val_pred, val_target, [1.0, 1.0, 1.0])
val_selected_ious = scaled_ious(val_pred, val_target, selected["scale_xyz"])
val_baseline = ranker.metrics(val_baseline_ious)
val_selected = ranker.metrics(val_selected_ious)
val_selected.update(
    fix050=int(
        np.sum((val_baseline_ious < 0.50) & (val_selected_ious >= 0.50))
    ),
    break050=int(
        np.sum((val_baseline_ious >= 0.50) & (val_selected_ious < 0.50))
    ),
    fix025=int(
        np.sum((val_baseline_ious < 0.25) & (val_selected_ious >= 0.25))
    ),
    break025=int(
        np.sum((val_baseline_ious >= 0.25) & (val_selected_ious < 0.25))
    ),
)
assert abs(val_baseline["acc025"] - 0.5477492637778713) < 1e-12
assert abs(val_baseline["acc050"] - 0.41533445519562473) < 1e-12

result = {
    "stage": "126",
    "protocol": "clean_train_scene_dev_locked_fixed_box_size_scale",
    "train_dump": os.path.abspath(train_dump),
    "train_dump_sha256": ranker.sha256(train_dump),
    "val_dump": os.path.abspath(val_dump),
    "val_dump_sha256": ranker.sha256(val_dump),
    "train_baseline": baseline,
    "selected_policy": selected,
    "all_train_policies": policies,
    "val": {"baseline": val_baseline, "selected": val_selected},
    "strict_goal_met_offline": bool(
        val_selected["acc025"] > 0.5391
        and val_selected["acc050"] > 0.4241
    ),
    "diagnostic_only_until_integrated_and_reloaded": True,
}
path = os.path.join(output_dir, "locked_fixed_box_scale_policy.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
