import json
import os
import sys

import numpy as np


package, dump, output, probability_text, limit_text = sys.argv[1:]
sys.path.insert(0, package)
import train_joint_option_ranker as ranker  # noqa: E402


limit = int(limit_text)
rows = ranker.load_rows(dump)
if limit > 0:
    rows = rows[:limit]

baseline_ious = []
oracle_ious = []
candidate_counts = []
for row in rows:
    _, ious, baseline_index = ranker.row_options(row, max_candidates=8)
    baseline_ious.append(float(ious[int(baseline_index)]))
    oracle_ious.append(float(np.max(ious)))
    candidate_counts.append(int(len(ious)))

baseline_ious = np.asarray(baseline_ious, dtype=np.float32)
oracle_ious = np.asarray(oracle_ious, dtype=np.float32)
result = {
    "stage": "122",
    "diagnostic_only": True,
    "corruption_probability": float(probability_text),
    "count": int(len(rows)),
    "dump": os.path.abspath(dump),
    "dump_sha256": ranker.sha256(dump),
    "baseline": ranker.metrics(baseline_ious),
    "candidate_oracle": ranker.metrics(oracle_ious),
    "candidate_count_mean": float(np.mean(candidate_counts)),
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(result, indent=2, sort_keys=True))
