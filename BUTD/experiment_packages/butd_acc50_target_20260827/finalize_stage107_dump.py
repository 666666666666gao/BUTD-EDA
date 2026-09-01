import hashlib
import json
import os
import sys

import torch


dump, out, expected_text, status_path = sys.argv[1:]
expected = int(expected_text)
rows = torch.load(dump, map_location="cpu")["rows"]
assert len(rows) == expected, (len(rows), expected)
required = (
    "scene_id",
    "adapter_candidate_query",
    "adapter_box_at_candidate",
    "adapter_hit50_logit_at_candidate",
    "gt_box",
    "detected_box",
)
for key in required:
    assert all(key in row for row in rows), key
scene_count = len({row["scene_id"] for row in rows})
assert scene_count >= (1 if expected < 1000 else 100), scene_count
digest = hashlib.sha256()
with open(dump, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)
receipt = {
    "stage": "107",
    "status": "complete",
    "rows": len(rows),
    "scene_count": scene_count,
    "dump": dump,
    "dump_sha256": digest.hexdigest(),
    "source_checkpoint_sha256": "f1fc08314ef1143d2d9dd83f47d8b0773e5fbd5b0f809e1a23104c7eb82e6811",
    "dump_evaluator_sha256": "0e9c5c6474385274d97a602780118e3c0c6e094d9e805c2323ab1ec55ec8bbe6",
}
path = os.path.join(out, "stage107_receipt.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write("\n")
with open(status_path, "w", encoding="utf-8") as f:
    f.write("stage107_complete\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
