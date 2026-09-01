#!/usr/bin/env python3
"""Report Unique/Multiple metrics for the already-locked Stage154 policy."""

import argparse
import hashlib
import json
import os


STAGE154 = "154_train_only_scene_oof_stage142_stage150_source_selector"
EXPECTED_VALIDATION_RESULT_SHA256 = (
    "983b74dd3dfb816992ffe8a854c9428836721620d52b24c6693e2d3ab653f816"
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def select_fixed_source_ious(scores, stage142_ious, stage150_ious, threshold):
    scores = [float(value) for value in scores]
    old = [float(value) for value in stage142_ious]
    new = [float(value) for value in stage150_ious]
    if not (len(scores) == len(old) == len(new)):
        raise ValueError("scores and source IoUs must have equal length")
    threshold = float(threshold)
    return [
        new_iou if score >= threshold else old_iou
        for score, old_iou, new_iou in zip(scores, old, new)
    ]


def subgroup_metrics(is_unique, selected_ious):
    flags = [bool(value) for value in is_unique]
    ious = [float(value) for value in selected_ious]
    if len(flags) != len(ious) or not flags:
        raise ValueError("subgroup labels and selected IoUs must have equal nonzero length")

    unique_count = sum(flags)
    multiple_count = len(flags) - unique_count
    if unique_count == 0 or multiple_count == 0:
        raise ValueError("both Unique and Multiple subsets must be nonempty")

    counts = {
        "unique": unique_count,
        "multiple": multiple_count,
        "total": len(flags),
    }

    def threshold_result(threshold):
        hits = [iou > threshold for iou in ious]
        unique_hits = sum(hit for hit, flag in zip(hits, flags) if flag)
        multiple_hits = sum(hit for hit, flag in zip(hits, flags) if not flag)
        overall_hits = unique_hits + multiple_hits
        return (
            {"unique": unique_hits, "multiple": multiple_hits, "overall": overall_hits},
            {
                "unique": unique_hits / float(unique_count),
                "multiple": multiple_hits / float(multiple_count),
                "overall": overall_hits / float(len(flags)),
            },
        )

    hits025, acc025 = threshold_result(0.25)
    hits050, acc050 = threshold_result(0.50)
    return {
        "counts": counts,
        "hits025": hits025,
        "hits050": hits050,
        "acc025": acc025,
        "acc050": acc050,
    }


def validate_stage154_contract(metrics, expected_hits):
    expected_counts = {"unique": 1419, "multiple": 8089, "total": 9508}
    if metrics["counts"] != expected_counts:
        raise ValueError(
            "Stage154 subgroup sample contract mismatch: expected={} actual={}".format(
                expected_counts, metrics["counts"]
            )
        )
    observed = {
        "acc025": int(metrics["hits025"]["overall"]),
        "acc050": int(metrics["hits050"]["overall"]),
    }
    if observed != expected_hits:
        raise ValueError(
            "locked Overall hit mismatch: expected={} actual={}".format(
                expected_hits, observed
            )
        )
    return True


def derive(args):
    import numpy as np

    import stage153_train_source_selector as base

    if os.path.exists(args.output_json):
        raise ValueError("output already exists: {}".format(args.output_json))
    validation_result_sha = sha256(args.validation_result)
    if validation_result_sha != EXPECTED_VALIDATION_RESULT_SHA256:
        raise ValueError(
            "Stage154 validation-result SHA256 mismatch: expected={} actual={}".format(
                EXPECTED_VALIDATION_RESULT_SHA256, validation_result_sha
            )
        )
    validation = read_json(args.validation_result)
    if validation.get("status") != "complete":
        raise ValueError("Stage154 validation result is not complete")
    if validation.get("validation_labels_used_for_selection") is not False:
        raise ValueError("Stage154 selector provenance is not train-only")

    lock = base.read_json(args.policy_lock)
    if lock.get("stage") != STAGE154:
        raise ValueError("unexpected Stage154 policy stage")
    if base.sha256(args.policy_lock) != validation["policy_lock_sha256"]:
        raise ValueError("policy lock SHA256 differs from locked validation result")
    if lock.get("validation_labels_used_for_selection") is not False:
        raise ValueError("policy lock used validation labels for selection")

    raw_rows = base.option_ranker.load_rows(args.stage142_dump)
    if len(raw_rows) != 9508:
        raise ValueError("unexpected Stage154 validation row count")
    stage142 = base.build_stage142_arrays(
        raw_rows,
        args.stage142_lock,
        args.stage31_lock,
        args.stage33_lock,
    )
    matrix, stage150_ious, feature_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump,
        args.stage150_compact_dump,
        raw_rows,
        stage142,
        locked_feature_names=lock["source_feature_names"],
    )
    if feature_names != lock["feature_names"]:
        raise ValueError("Stage154 full feature identity mismatch")
    if source_feature_names != lock["source_feature_names"]:
        raise ValueError("Stage154 source feature identity mismatch")

    candidate = base.load_candidate(lock)
    scores = base.booster_predict(candidate, matrix)
    threshold = float(lock["selected_oof"]["threshold"])
    locked_threshold = float(validation["metrics"]["threshold"])
    if abs(threshold - locked_threshold) > 1e-15:
        raise ValueError("Stage154 threshold differs from validation result")
    selected_ious = select_fixed_source_ious(
        scores,
        stage142["ious"],
        stage150_ious,
        threshold,
    )

    labels = []
    for index, row in enumerate(raw_rows):
        if "is_unique_label_only" not in row:
            raise ValueError("row {} lacks is_unique_label_only".format(index))
        labels.append(bool(row["is_unique_label_only"]))
    metrics = subgroup_metrics(labels, selected_ious)
    expected_hits = {
        "acc025": int(validation["metrics"]["selected"]["hits025"]),
        "acc050": int(validation["metrics"]["selected"]["hits050"]),
    }
    validate_stage154_contract(metrics, expected_hits)
    for threshold_name, validation_name in (("acc025", "acc025"), ("acc050", "acc050")):
        observed = float(metrics[threshold_name]["overall"])
        expected = float(validation["metrics"]["selected"][validation_name])
        if abs(observed - expected) > 1e-15:
            raise ValueError(
                "Stage154 Overall accuracy parity failed at {}".format(threshold_name)
            )

    changed = int((np.asarray(scores) >= threshold).sum())
    payload = {
        "stage": "154_locked_validation_subgroup_report",
        "status": "complete",
        "evaluation_scope": "posthoc_subgroup_reporting_only",
        "model_or_threshold_reselected": False,
        "validation_labels_used_for_selection": False,
        "validation_labels_used_for_subgroup_reporting": True,
        "validation_result": os.path.abspath(args.validation_result),
        "validation_result_sha256": validation_result_sha,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": base.sha256(args.policy_lock),
        "selected_candidate": lock["selected_candidate"],
        "threshold": threshold,
        "feature_count": len(feature_names),
        "changed": changed,
        "changed_ratio": changed / float(len(raw_rows)),
        "sources": {
            "stage142_dump": os.path.abspath(args.stage142_dump),
            "stage142_dump_sha256": base.sha256(args.stage142_dump),
            "stage150_source_dump": os.path.abspath(args.stage150_source_dump),
            "stage150_source_dump_sha256": base.sha256(args.stage150_source_dump),
            "stage150_compact_dump": os.path.abspath(args.stage150_compact_dump),
            "stage150_compact_dump_sha256": base.sha256(args.stage150_compact_dump),
        },
        "metrics": metrics,
        "percent": {
            "unique_025": metrics["acc025"]["unique"] * 100.0,
            "unique_050": metrics["acc050"]["unique"] * 100.0,
            "multiple_025": metrics["acc025"]["multiple"] * 100.0,
            "multiple_050": metrics["acc050"]["multiple"] * 100.0,
            "overall_025": metrics["acc025"]["overall"] * 100.0,
            "overall_050": metrics["acc050"]["overall"] * 100.0,
        },
    }
    base.atomic_json(args.output_json, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage142_dump")
    parser.add_argument("stage150_source_dump")
    parser.add_argument("stage150_compact_dump")
    parser.add_argument("stage31_lock")
    parser.add_argument("stage33_lock")
    parser.add_argument("stage142_lock")
    parser.add_argument("policy_lock")
    parser.add_argument("validation_result")
    parser.add_argument("output_json")
    args = parser.parse_args()
    result = derive(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
