import json
import os
import sys

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


def aligned_selected_boxes(left_rows, right_rows):
    assert len(left_rows) == len(right_rows)
    left_boxes = []
    right_boxes = []
    targets = []
    scenes = []
    score_key = ranker.SCORE_KEYS[4]
    for index, (left, right) in enumerate(zip(left_rows, right_rows)):
        assert int(left.get("example_id", index)) == int(
            right.get("example_id", index)
        )
        left_target = np.asarray(left["gt_box"], dtype=np.float32)
        right_target = np.asarray(right["gt_box"], dtype=np.float32)
        assert np.allclose(left_target, right_target, atol=1e-6)
        left_position = int(
            np.argmax(np.nan_to_num(np.asarray(left[score_key], dtype=np.float32)))
        )
        right_position = int(
            np.argmax(np.nan_to_num(np.asarray(right[score_key], dtype=np.float32)))
        )
        left_boxes.append(
            np.asarray(left["adapter_box_at_candidate"], dtype=np.float32)[
                left_position
            ]
        )
        right_boxes.append(
            np.asarray(right["adapter_box_at_candidate"], dtype=np.float32)[
                right_position
            ]
        )
        targets.append(left_target)
        scenes.append(str(right.get("scene_id", index)))
    return (
        np.asarray(left_boxes, dtype=np.float32),
        np.asarray(right_boxes, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        scenes,
    )


def box_ious(boxes, target):
    boxes = boxes.copy()
    boxes[:, 3:] = np.maximum(np.abs(boxes[:, 3:]), 1e-5)
    return ranker.aligned_iou(boxes, target)


def blend_boxes(stage18, stage95, stage18_weight):
    boxes = (
        float(stage18_weight) * stage18
        + (1.0 - float(stage18_weight)) * stage95
    )
    boxes[:, 3:] = np.maximum(np.abs(boxes[:, 3:]), 1e-5)
    return boxes


stage18_train_rows = ranker.load_rows(stage18_train_dump)
stage95_train_rows = ranker.load_rows(stage95_train_dump)
train18, train95, train_target, train_scenes = aligned_selected_boxes(
    stage18_train_rows, stage95_train_rows
)
splits = {"train": [], "dev": [], "test": []}
for index, scene in enumerate(train_scenes):
    bucket = ranker.scene_bucket(scene)
    split = "train" if bucket < 70 else ("dev" if bucket < 85 else "test")
    splits[split].append(index)
splits = {key: np.asarray(value, dtype=np.int64) for key, value in splits.items()}

train18_iou = box_ious(train18, train_target)
train95_iou = box_ious(train95, train_target)
train_policies = []
for weight in np.round(np.arange(0.0, 1.001, 0.025), 3):
    ious = box_ious(blend_boxes(train18, train95, weight), train_target)
    train_policies.append(
        {
            "stage18_weight": float(weight),
            "dev": ranker.metrics(ious[splits["dev"]]),
            "test": ranker.metrics(ious[splits["test"]]),
        }
    )

baseline = next(row for row in train_policies if row["stage18_weight"] == 0.0)
feasible = [
    row
    for row in train_policies
    if row["dev"]["acc025"] >= baseline["dev"]["acc025"] - 0.001
]
selected = max(
    feasible,
    key=lambda row: (
        row["dev"]["acc050"],
        row["dev"]["acc025"],
        row["dev"]["mean_iou"],
        -row["stage18_weight"],
    ),
)

stage18_val_rows = ranker.load_rows(stage18_val_dump)
stage95_val_rows = ranker.load_rows(stage95_val_dump)
val18, val95, val_target, _ = aligned_selected_boxes(
    stage18_val_rows, stage95_val_rows
)
val18_iou = box_ious(val18, val_target)
val95_iou = box_ious(val95, val_target)
val_blend_iou = box_ious(
    blend_boxes(val18, val95, selected["stage18_weight"]), val_target
)
val_oracle_iou = np.maximum(val18_iou, val95_iou)

val_blend = ranker.metrics(val_blend_iou)
val_blend.update(
    fix050=int(np.sum((val95_iou < 0.50) & (val_blend_iou >= 0.50))),
    break050=int(np.sum((val95_iou >= 0.50) & (val_blend_iou < 0.50))),
    fix025=int(np.sum((val95_iou < 0.25) & (val_blend_iou >= 0.25))),
    break025=int(np.sum((val95_iou >= 0.25) & (val_blend_iou < 0.25))),
)
stage18_as_action = {
    "metrics": ranker.metrics(val18_iou),
    "fix050": int(np.sum((val95_iou < 0.50) & (val18_iou >= 0.50))),
    "break050": int(np.sum((val95_iou >= 0.50) & (val18_iou < 0.50))),
    "fix025": int(np.sum((val95_iou < 0.25) & (val18_iou >= 0.25))),
    "break025": int(np.sum((val95_iou >= 0.25) & (val18_iou < 0.25))),
}

result = {
    "stage": "127",
    "protocol": "cross_checkpoint_train_dev_locked_box_blend",
    "inputs": {
        "stage18_train_sha256": ranker.sha256(stage18_train_dump),
        "stage95_train_sha256": ranker.sha256(stage95_train_dump),
        "stage18_val_sha256": ranker.sha256(stage18_val_dump),
        "stage95_val_sha256": ranker.sha256(stage95_val_dump),
    },
    "train": {
        "stage18_dev": ranker.metrics(train18_iou[splits["dev"]]),
        "stage95_dev": ranker.metrics(train95_iou[splits["dev"]]),
        "stage18_test": ranker.metrics(train18_iou[splits["test"]]),
        "stage95_test": ranker.metrics(train95_iou[splits["test"]]),
        "selected_policy": selected,
        "all_blend_policies": train_policies,
    },
    "val": {
        "stage18": stage18_as_action,
        "stage95": ranker.metrics(val95_iou),
        "selected_blend": val_blend,
        "two_checkpoint_oracle": ranker.metrics(val_oracle_iou),
    },
    "strict_goal_met_offline": bool(
        val_blend["acc025"] > 0.5391 and val_blend["acc050"] > 0.4241
    ),
    "diagnostic_only_until_integrated_and_reloaded": True,
}
assert abs(result["val"]["stage95"]["acc025"] - 0.5477492637778713) < 1e-12
assert abs(result["val"]["stage95"]["acc050"] - 0.41533445519562473) < 1e-12
path = os.path.join(output_dir, "stage127_cross_checkpoint_box_audit.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
