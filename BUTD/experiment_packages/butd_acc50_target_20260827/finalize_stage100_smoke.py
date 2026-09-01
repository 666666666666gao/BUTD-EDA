import hashlib
import json
import os
import sys

import torch


live_path, dump_metrics_path, dump_path, out, status_path = sys.argv[1:]
with open(live_path, encoding="utf-8") as f:
    live = json.load(f)
with open(dump_metrics_path, encoding="utf-8") as f:
    dumped = json.load(f)
dump_row_count = dumped.pop("detector_topk_compact_rows")
assert dump_row_count == 96, dump_row_count
assert live == dumped, {
    key: (live.get(key), dumped.get(key))
    for key in sorted(set(live) | set(dumped))
    if live.get(key) != dumped.get(key)
}

payload = torch.load(dump_path, map_location="cpu")
rows = payload["rows"]
assert len(rows) == 96, len(rows)
for required in (
    "adapter_candidate_query",
    "adapter_hit50_logit_at_candidate",
    "adapter_box_at_candidate",
    "gt_box",
    "detected_box",
):
    assert all(required in row for row in rows), required

receipt = {
    "stage": "100",
    "status": "smoke_parity_pass",
    "samples": len(rows),
    "live_evaluator_sha256": "50bb300e4ddee8234c5be041a7d74429fdcfdeabcff432df8e68d36de0076d86",
    "dump_evaluator_sha256": "cc5a662474b1de9ab5eceed737a4348e0231b54b460f24dad4a9a5ad5f99724f",
    "acc025": live["last__bbs_acc0.25_top1"],
    "acc050": live["last__bbs_acc0.50_top1"],
    "all_shared_eval_metrics_exact_match": True,
    "expected_dump_only_metric": {"detector_topk_compact_rows": dump_row_count},
    "executed_runner_sha256": "35a8a777ad77aaf2abc186ee010d6d92b7887cf9663fceb9b5ab1a877cf33611",
}
for key, path in (
    ("live_metrics_sha256", live_path),
    ("dump_metrics_sha256", dump_metrics_path),
    ("dump_sha256", dump_path),
):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    receipt[key] = digest.hexdigest()

receipt_path = os.path.join(out, "stage100_smoke_parity_receipt.json")
with open(receipt_path, "w", encoding="utf-8") as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write("\n")
with open(status_path, "w", encoding="utf-8") as f:
    f.write("stage100_smoke_parity_pass\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
