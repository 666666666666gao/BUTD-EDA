import hashlib
import json
import os
import runpy
import sys


(
    source,
    package,
    augmented_train_dump,
    candidate_model,
    candidate_lock,
    val_dump,
    out,
) = sys.argv[1:]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


expected_source_sha = (
    "d791e55adebb4bc68be8ddbeb86c42dbe7baea8a1bb1b1ce3c3b6f6addadbdca"
)
assert sha256(source) == expected_source_sha

sys.argv = [
    source,
    package,
    augmented_train_dump,
    candidate_model,
    candidate_lock,
    val_dump,
    out,
]
runpy.run_path(source, run_name="__main__")

lock_path = os.path.join(out, "locked_learned_override_gate.json")
lock = json.load(open(lock_path, encoding="utf-8"))
assert lock["stage"] == "114"
lock.update(
    stage="120",
    protocol="stage117_mixed_candidate_augmented_scene_split_learned_override_gate",
    source_gate_script=os.path.abspath(source),
    source_gate_script_sha256=expected_source_sha,
)
with open(lock_path, "w", encoding="utf-8") as handle:
    json.dump(lock, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(
    json.dumps(
        {
            "stage": "120",
            "lock": lock_path,
            "lock_sha256": sha256(lock_path),
            "selected": lock["val"]["selected"],
            "strict_goal_met_offline": lock["strict_goal_met_offline"],
        },
        indent=2,
        sort_keys=True,
    )
)
