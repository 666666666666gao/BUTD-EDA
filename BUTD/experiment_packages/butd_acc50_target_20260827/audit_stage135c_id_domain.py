#!/usr/bin/env python3
"""Audit that Stage29 dump IDs equal ScanRefer loader order."""

import argparse
import json

import torch


def _key(scene_id, object_id, ann_id):
    return str(scene_id), str(object_id), str(ann_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--scan-ids", required=True)
    parser.add_argument("--dump", required=True)
    args = parser.parse_args()

    with open(args.scan_ids, encoding="utf-8") as handle:
        allowed = {line.strip() for line in handle if line.strip()}
    with open(args.annotations, encoding="utf-8") as handle:
        reader = json.load(handle)

    loader_keys = []
    for source_index, anno in enumerate(reader):
        if anno["scene_id"] not in allowed:
            continue
        ann_id = anno.get("ann_id", anno.get("ann_id_key", str(source_index)))
        loader_keys.append(_key(anno["scene_id"], anno["object_id"], ann_id))

    rows = torch.load(args.dump, map_location="cpu")["rows"]
    dump_keys = [
        _key(row["scene_id"], row["object_id"], row["ann_id"])
        for row in rows
    ]
    dump_ids = [int(row["example_id"]) for row in rows]

    mismatches = [
        index
        for index, (left, right) in enumerate(zip(loader_keys, dump_keys))
        if left != right
    ]
    report = {
        "annotation_rows": len(reader),
        "filtered_scanrefer_rows": len(loader_keys),
        "dump_rows": len(rows),
        "dump_example_id_min": min(dump_ids),
        "dump_example_id_max": max(dump_ids),
        "dump_example_id_unique": len(set(dump_ids)),
        "stable_key_mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "joint_train_rows_observed": 48655,
        "joint_det_extra_rows": 48655 - len(loader_keys),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    assert len(loader_keys) == len(rows) == 36665
    assert dump_ids == list(range(36665))
    assert not mismatches
    assert report["joint_det_extra_rows"] == 11990
    assert report["joint_det_extra_rows"] % 10 == 0
    print("STAGE135C_ID_DOMAIN_AUDIT_PASS")


if __name__ == "__main__":
    main()
