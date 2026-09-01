#!/usr/bin/env python3
"""Add inference-time ScanRefer scene IDs to an existing ordered raw dump."""

import argparse
import hashlib
import json
import os

import torch

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filtered_scene_ids(annotations, scan_ids):
    allowed = {str(scene_id) for scene_id in scan_ids}
    result = []
    for annotation in annotations:
        scene_id = str(annotation.get("scene_id", ""))
        assert scene_id
        if scene_id in allowed:
            result.append(scene_id)
    return result


def atomic_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dump")
    parser.add_argument("annotations_json")
    parser.add_argument("scan_ids_txt")
    parser.add_argument("output_dump")
    parser.add_argument("receipt_json")
    args = parser.parse_args()
    assert not os.path.exists(args.output_dump), args.output_dump
    assert not os.path.exists(args.receipt_json), args.receipt_json

    payload = torch.load(args.raw_dump, map_location="cpu")
    rows = payload["rows"]
    with open(args.annotations_json, encoding="utf-8") as handle:
        annotations = json.load(handle)
    with open(args.scan_ids_txt, encoding="utf-8") as handle:
        scan_ids = [line.strip() for line in handle if line.strip()]
    ordered = filtered_scene_ids(annotations, scan_ids)
    assert len(rows) == len(ordered) == 9508
    assert all(not row.get("scene_id", "") for row in rows)
    assert [int(row.get("example_id", index)) for index, row in enumerate(rows)] \
        == list(range(len(rows)))

    enriched_rows = []
    for row, scene_id in zip(rows, ordered):
        item = dict(row)
        item["scene_id"] = scene_id
        enriched_rows.append(item)
    enriched = dict(payload)
    enriched["rows"] = enriched_rows
    enriched["scene_id_provenance"] = {
        "source": "Joint3DDataset.load_scanrefer_annos reader order",
        "annotations_json": os.path.abspath(args.annotations_json),
        "annotations_sha256": sha256(args.annotations_json),
        "scan_ids_txt": os.path.abspath(args.scan_ids_txt),
        "scan_ids_sha256": sha256(args.scan_ids_txt),
        "validation_labels_used_for_policy_selection": False,
    }
    temporary = args.output_dump + ".tmp"
    torch.save(enriched, temporary)
    os.replace(temporary, args.output_dump)

    reloaded = torch.load(args.output_dump, map_location="cpu")
    assert len(reloaded["rows"]) == 9508
    assert [row["scene_id"] for row in reloaded["rows"]] == ordered
    receipt = {
        "status": "complete",
        "row_count": len(ordered),
        "unique_scene_count": len(set(ordered)),
        "input_raw_sha256": sha256(args.raw_dump),
        "annotations_sha256": sha256(args.annotations_json),
        "scan_ids_sha256": sha256(args.scan_ids_txt),
        "output_sha256": sha256(args.output_dump),
        "row_order_contract": (
            "ScanRefer refined JSON reader order filtered by official val scan IDs"
        ),
        "validation_labels_used_for_policy_selection": False,
    }
    atomic_json(args.receipt_json, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
