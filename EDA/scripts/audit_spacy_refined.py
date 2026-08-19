#!/usr/bin/env python
"""Audit refined *_spacy decomposition files for weak relation buckets."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

EDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDA_ROOT))

from src.joint_det_dataset import Joint3DDataset


JSON_FIELDS = {
    "rel_slots": [],
    "attr_slot": {},
    "coverage_stats": {},
    "slot_mask": {},
}

REFINED_PATTERNS = (
    "*spacy_refined.json",
    "*spacy_refined.csv",
)


def _json_load(value, default):
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _infer_dataset(path):
    name = path.name.lower()
    if "scanrefer" in name:
        return "scanrefer_spacy"
    if "nr3d" in name:
        return "nr3d_spacy"
    if "sr3d" in name:
        return "sr3d_spacy"
    return "unknown_spacy"


def _utterance(row):
    if row.get("utterance"):
        return str(row["utterance"])
    if row.get("description"):
        return str(row["description"])
    tokens = row.get("tokens", row.get("token", []))
    if isinstance(tokens, list):
        return " ".join(str(token) for token in tokens)
    return str(tokens or "")


def _relation_slots(row):
    rel_slots = row.get("rel_slots", [])
    if not isinstance(rel_slots, list):
        rel_slots = [rel_slots]
    return [slot for slot in rel_slots if Joint3DDataset._spacy_relation_text(slot).strip()]


def _has_spatial_attribute(row):
    attr_slot = row.get("attr_slot", {})
    items = attr_slot.get("items", []) if isinstance(attr_slot, dict) else []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text = Joint3DDataset._spacy_spatial_attr_text(item).lower()
        padded = " " + text + " "
        if item.get("type") == "spatial_attribute":
            return True
        for term in (
            "top",
            "bottom",
            "upper",
            "lower",
            "center",
            "centre",
            "middle",
            "corner",
            "side",
            "edge",
            "above",
            "below",
            "under",
            "over",
            "beneath",
        ):
            if " " + term + " " in padded:
                return True

    coverage = row.get("coverage_stats", {})
    return (
        isinstance(coverage, dict)
        and Joint3DDataset._spacy_positive_count(
            coverage.get("spatial_attribute_rows", 0)
        )
    )


def _has_weak_parse_signal(row):
    coverage = row.get("coverage_stats", {})
    if not isinstance(coverage, dict):
        return False
    weak_keys = (
        "candidate_relation_count",
        "invalid_relation_count",
        "spatial_attribute_rows",
        "spatial_info_routed_to_attr",
        "refined_dropped_relation_count",
    )
    return any(
        Joint3DDataset._spacy_positive_count(coverage.get(key, 0))
        for key in weak_keys
    )


def _normalise_row(row, dataset):
    normalised = dict(row)
    for field, default in JSON_FIELDS.items():
        normalised[field] = _json_load(normalised.get(field), default)
    normalised["dataset"] = dataset
    normalised["utterance"] = _utterance(normalised)
    return normalised


def _bool_field(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def audit_rows(rows, dataset):
    stats = Counter()
    for raw in rows:
        row = _normalise_row(raw, dataset)
        utterance = row["utterance"]
        rel_slots = _relation_slots(row)
        relation_free = len(rel_slots) == 0
        raw_view = Joint3DDataset._spacy_has_raw_view_word(utterance)
        compass = Joint3DDataset._spacy_has_compass_direction_word(utterance)
        spatial_attr = _has_spatial_attribute(row)
        weak_signal = _has_weak_parse_signal(row)

        stats["rows"] += 1
        stats[f"{dataset}_rows"] += 1
        stats["relation_rows"] += int(not relation_free)
        stats["relation_free_rows"] += int(relation_free)
        stats["raw_view_rows"] += int(raw_view)
        stats["compass_rows"] += int(compass)
        stats["spatial_attribute_rows"] += int(spatial_attr)
        stats["weak_parse_rows"] += int(relation_free and weak_signal)
        stats["weak_raw_view_rows"] += int(relation_free and weak_signal and raw_view)
        stats["weak_spatial_attribute_rows"] += int(
            relation_free and weak_signal and spatial_attr
        )
        stats["train_global_only_rows"] += int(
            _bool_field(row.get("decomp_train_global_only_mask", False))
        )
        stats["train_weak_generic_rows"] += int(
            _bool_field(row.get("decomp_train_weak_generic_mask", False))
        )
        stats["protected_rotation_rows"] += int(
            Joint3DDataset._spacy_rotation_mode(row) == "none"
        )
        stats["yaw_relation_rows"] += int(
            Joint3DDataset._spacy_rotation_mode(row) == "yaw_only"
        )
    return stats


def _read_file(path):
    dataset = _infer_dataset(path)
    if path.suffix == ".json":
        with path.open() as f:
            return audit_rows(json.load(f), dataset)
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            return audit_rows(csv.DictReader(f), dataset)
    return Counter()


def audit_path(path):
    path = Path(path)
    if path.is_file():
        return _read_file(path)

    stats = Counter()
    for pattern in REFINED_PATTERNS:
        for refined_path in sorted(path.rglob(pattern)):
            stats.update(_read_file(refined_path))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    stats = audit_path(args.path)
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
