import json
import os
import sys

import lightgbm as lgb
import numpy as np


(
    package,
    val_dump,
    candidate_model,
    candidate_lock_path,
    gate_model,
    gate_lock_path,
    output,
) = sys.argv[1:]
sys.path.insert(0, package)
import train_joint_option_ranker as ranker  # noqa: E402


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
                [
                    candidate_feature,
                    baseline_feature,
                    candidate_feature - baseline_feature,
                    scalars,
                ],
                axis=0,
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


def action_metrics(scores, threshold, baseline, candidate, changed):
    accepted = changed & (scores >= threshold)
    selected = np.where(accepted, candidate, baseline)
    result = ranker.metrics(selected)
    result.update(
        threshold=float(threshold),
        accepted_ratio=float(np.mean(accepted)),
        fix050=int(np.sum((baseline < 0.50) & (selected >= 0.50))),
        break050=int(np.sum((baseline >= 0.50) & (selected < 0.50))),
        fix025=int(np.sum((baseline < 0.25) & (selected >= 0.25))),
        break025=int(np.sum((baseline >= 0.25) & (selected < 0.25))),
    )
    return result


candidate_lock = json.load(open(candidate_lock_path, encoding="utf-8"))
gate_lock = json.load(open(gate_lock_path, encoding="utf-8"))
assert ranker.sha256(candidate_model) == candidate_lock["model_sha256"]
assert ranker.sha256(gate_model) == gate_lock["gate_model_sha256"]

rows = ranker.load_rows(val_dump)
group_features, group_ious, metas = ranker.build_dataset(
    rows, int(candidate_lock["max_candidates"]), require_scene=False
)
x = np.concatenate(group_features, axis=0)
ious = np.concatenate(group_ious, axis=0)
groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
baselines = np.asarray([meta.baseline_index for meta in metas], dtype=np.int32)

candidate_booster = lgb.Booster(model_file=candidate_model)
candidate_scores = candidate_booster.predict(
    x, num_iteration=int(candidate_lock["best_iteration"])
)
gate_x, baseline, candidate, changed = gate_examples(
    x, ious, groups, baselines, candidate_scores
)
gate_booster = lgb.Booster(model_file=gate_model)
gate_scores = gate_booster.predict(
    gate_x, num_iteration=int(gate_lock["gate_iteration"])
)
locked_threshold = float(gate_lock["val"]["selected"]["threshold"])
finite = gate_scores[np.isfinite(gate_scores)]
thresholds = list(
    np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 1001)))
)
thresholds.extend([locked_threshold, float("-inf"), float("inf")])
thresholds = list(dict.fromkeys(float(value) for value in thresholds))
audits = [
    action_metrics(gate_scores, threshold, baseline, candidate, changed)
    for threshold in thresholds
]


def best(rows):
    return max(
        rows,
        key=lambda row: (
            row["acc050"],
            row["acc025"],
            row["mean_iou"],
            -row["accepted_ratio"],
        ),
    )


baseline_metrics = ranker.metrics(baseline)
result = {
    "stage": "121",
    "diagnostic_only": True,
    "val_dump": os.path.abspath(val_dump),
    "val_dump_sha256": ranker.sha256(val_dump),
    "candidate_model_sha256": ranker.sha256(candidate_model),
    "candidate_lock_sha256": ranker.sha256(candidate_lock_path),
    "gate_model_sha256": ranker.sha256(gate_model),
    "gate_lock_sha256": ranker.sha256(gate_lock_path),
    "baseline": baseline_metrics,
    "locked": next(
        row for row in audits if row["threshold"] == locked_threshold
    ),
    "best_with_acc025_gt_05391": best(
        [row for row in audits if row["acc025"] > 0.5391]
    ),
    "best_preserving_acc025_within_0p1pp": best(
        [
            row
            for row in audits
            if row["acc025"] >= baseline_metrics["acc025"] - 0.001
        ]
    ),
    "candidate_action_pool": {
        "changed_ratio": float(np.mean(changed)),
        "fix050": int(np.sum((baseline < 0.50) & (candidate >= 0.50))),
        "break050": int(np.sum((baseline >= 0.50) & (candidate < 0.50))),
        "fix025": int(np.sum((baseline < 0.25) & (candidate >= 0.25))),
        "break025": int(np.sum((baseline >= 0.25) & (candidate < 0.25))),
    },
}
best_goal = result["best_with_acc025_gt_05391"]
result["threshold_can_reach_strict_goal"] = bool(
    best_goal["acc025"] > 0.5391 and best_goal["acc050"] > 0.4241
)
with open(output, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
