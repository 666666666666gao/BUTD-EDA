import json
import os
import sys

import numpy as np


package, train_dump, val_dump, output_dir = sys.argv[1:]
sys.path.insert(0, package)
import train_joint_option_ranker as ranker  # noqa: E402


os.makedirs(output_dir, exist_ok=False)
idx_baseline = ranker.FEATURE_NAMES.index("is_baseline")
idx_power = ranker.FEATURE_NAMES.index("match_conf_power")
idx_alpha = ranker.FEATURE_NAMES.index("alpha")


def policy_ious(group_features, group_ious, indices, power, alpha):
    selected = []
    for index in indices:
        features = group_features[index]
        ious = group_ious[index]
        mask = (
            np.isclose(features[:, idx_baseline], 1.0)
            & np.isclose(features[:, idx_power], power)
            & np.isclose(features[:, idx_alpha], alpha)
        )
        positions = np.flatnonzero(mask)
        assert len(positions) == 1, (index, power, alpha, positions)
        selected.append(float(ious[int(positions[0])]))
    return np.asarray(selected, dtype=np.float32)


train_rows = ranker.load_rows(train_dump)
train_features, train_ious, train_metas = ranker.build_dataset(
    train_rows, max_candidates=8, require_scene=True
)
splits = ranker.split_indices(train_metas)
policies = []
for power in ranker.MATCH_POWERS:
    for alpha in ranker.ACTIONS:
        dev_ious = policy_ious(
            train_features, train_ious, splits["dev"], power, alpha
        )
        test_ious = policy_ious(
            train_features, train_ious, splits["test"], power, alpha
        )
        policies.append(
            {
                "match_power": float(power),
                "alpha": float(alpha),
                "dev": ranker.metrics(dev_ious),
                "test": ranker.metrics(test_ious),
            }
        )

baseline = next(
    row
    for row in policies
    if row["match_power"] == float(ranker.MATCH_POWERS[0])
    and row["alpha"] == 0.0
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
        -row["alpha"],
        -row["match_power"],
    ),
)

val_rows = ranker.load_rows(val_dump)
val_features, val_ious, val_metas = ranker.build_dataset(
    val_rows, max_candidates=8, require_scene=False
)
val_indices = list(range(len(val_metas)))
val_baseline_ious = policy_ious(
    val_features,
    val_ious,
    val_indices,
    float(ranker.MATCH_POWERS[0]),
    0.0,
)
val_selected_ious = policy_ious(
    val_features,
    val_ious,
    val_indices,
    selected["match_power"],
    selected["alpha"],
)
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
    "stage": "125",
    "protocol": "clean_train_scene_dev_locked_fixed_baseline_query_box_policy",
    "train_dump": os.path.abspath(train_dump),
    "train_dump_sha256": ranker.sha256(train_dump),
    "val_dump": os.path.abspath(val_dump),
    "val_dump_sha256": ranker.sha256(val_dump),
    "all_train_policies": policies,
    "train_baseline": baseline,
    "selected_policy": selected,
    "val": {"baseline": val_baseline, "selected": val_selected},
    "strict_goal_met_offline": bool(
        val_selected["acc025"] > 0.5391
        and val_selected["acc050"] > 0.4241
    ),
    "diagnostic_only_until_integrated_and_reloaded": True,
}
path = os.path.join(output_dir, "locked_fixed_box_policy.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
