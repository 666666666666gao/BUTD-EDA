import hashlib
import json
import os
import sys

import lightgbm as lgb
import numpy as np


package, dump, model, lock_path, output = sys.argv[1:]
sys.path.insert(0, package)
import train_joint_option_ranker as ranker  # noqa: E402


lock = json.load(open(lock_path, encoding="utf-8"))
assert ranker.sha256(model) == lock["model_sha256"]
rows = ranker.load_rows(dump)
group_features, group_ious, metas = ranker.build_dataset(
    rows, int(lock["max_candidates"]), require_scene=False
)
x = np.concatenate(group_features, axis=0)
ious = np.concatenate(group_ious, axis=0)
groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
baselines = np.asarray(
    [meta.baseline_index for meta in metas], dtype=np.int32
)
booster = lgb.Booster(model_file=model)
scores = booster.predict(x, num_iteration=int(lock["best_iteration"]))
raw_selected, baseline_ious, gaps = ranker.group_decisions(
    scores, ious, groups, baselines, float("-inf")
)
finite = gaps[np.isfinite(gaps)]
thresholds = list(
    np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 1001)))
)
thresholds += [float("-inf"), float("inf")]
audits = []
for threshold in thresholds:
    selected, _, _ = ranker.group_decisions(
        scores, ious, groups, baselines, float(threshold)
    )
    metrics = ranker.metrics(selected)
    metrics.update(
        threshold=float(threshold),
        changed_ratio=float(np.mean(gaps >= threshold)),
        fix050=int(np.sum((baseline_ious < 0.50) & (selected >= 0.50))),
        break050=int(np.sum((baseline_ious >= 0.50) & (selected < 0.50))),
        fix025=int(np.sum((baseline_ious < 0.25) & (selected >= 0.25))),
        break025=int(np.sum((baseline_ious >= 0.25) & (selected < 0.25))),
    )
    audits.append(metrics)


def best(rows):
    return max(
        rows,
        key=lambda row: (
            row["acc050"],
            row["acc025"],
            row["mean_iou"],
            -row["changed_ratio"],
        ),
    )


baseline = ranker.metrics(baseline_ious)
result = {
    "stage": "113",
    "diagnostic_only": True,
    "dump": os.path.abspath(dump),
    "dump_sha256": ranker.sha256(dump),
    "model": os.path.abspath(model),
    "model_sha256": ranker.sha256(model),
    "lock_sha256": ranker.sha256(lock_path),
    "baseline": baseline,
    "locked": next(
        row
        for row in audits
        if abs(row["threshold"] - float(lock["gate"]["threshold"])) < 1e-12
    ),
    "raw_model_choice": ranker.metrics(raw_selected),
    "best_with_acc025_gt_05391": best(
        [row for row in audits if row["acc025"] > 0.5391]
    ),
    "best_preserving_acc025_within_0p1pp": best(
        [row for row in audits if row["acc025"] >= baseline["acc025"] - 0.001]
    ),
}
result["threshold_can_reach_strict_goal"] = bool(
    result["best_with_acc025_gt_05391"]["acc050"] > 0.4241
)
with open(output, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
