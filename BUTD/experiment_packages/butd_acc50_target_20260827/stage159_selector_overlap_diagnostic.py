#!/usr/bin/env python3
"""Read-only overlap diagnostic for Stage154 and Stage158 selectors."""

import argparse
import json
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof


def policy_masks(scores154, threshold154, scores158, threshold158):
    mask154 = np.asarray(scores154) >= float(threshold154)
    mask158 = np.asarray(scores158) >= float(threshold158)
    return {
        "stage154": mask154,
        "stage158": mask158,
        "intersection": mask154 & mask158,
        "union": mask154 | mask158,
        "stage154_only": mask154 & ~mask158,
        "stage158_only": mask158 & ~mask154,
        "neither": ~mask154 & ~mask158,
    }


def mask_effect(mask, old_ious, new_ious):
    mask = np.asarray(mask, dtype=bool)
    old_ious = np.asarray(old_ious)
    new_ious = np.asarray(new_ious)
    selected = np.where(mask, new_ious, old_ious)
    changed_old = old_ious[mask]
    changed_new = new_ious[mask]
    return {
        "selected_policy": base.summarize(selected),
        "changed_count": int(mask.sum()),
        "changed_ratio": float(mask.mean()),
        "changed_rows_fix_break": base.fix_break(
            changed_old, changed_new, np.ones(int(mask.sum()), dtype=bool)
        ),
        "overall_fix_break": base.fix_break(old_ious, selected, mask),
    }


def diagnose(args):
    raw_rows = base.option_ranker.load_rows(args.stage142_dump)
    stage142 = base.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, feature_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142,
    )
    lock154 = base.read_json(args.stage154_lock)
    lock158 = base.read_json(args.stage158_lock)
    assert lock154["validation_labels_used_for_selection"] is False
    assert lock158["validation_labels_used_for_selection"] is False
    assert feature_names == lock154["feature_names"] == lock158["feature_names"]
    assert source_feature_names == lock154["source_feature_names"]
    assert source_feature_names == lock158["source_feature_names"]
    indices = np.arange(len(raw_rows), dtype=np.int64)
    if args.scope == "internal_test":
        _, indices = oof.development_and_test(stage142["metas"])
    old_ious = stage142["ious"][indices]
    new_ious = new_ious[indices]
    features = matrix[indices]
    model154 = base.load_candidate(lock154)
    model158 = base.load_candidate(lock158)
    scores154 = base.booster_predict(model154, features)
    scores158 = base.booster_predict(model158, features)
    masks = policy_masks(
        scores154, lock154["selected_oof"]["threshold"],
        scores158, lock158["selected_oof"]["threshold"],
    )
    effects = {
        name: mask_effect(mask, old_ious, new_ious)
        for name, mask in masks.items()
    }
    disagreement050 = base.hit(old_ious, 0.50) != base.hit(new_ious, 0.50)
    report = {
        "stage": "159_stage154_stage158_overlap_diagnostic",
        "diagnostic_only": True,
        "scope": args.scope,
        "row_count": int(len(indices)),
        "validation_labels_used_for_policy_selection": False,
        "default_stage142": base.summarize(old_ious),
        "stage150": base.summarize(new_ious),
        "source_disagreement050_count": int(disagreement050.sum()),
        "stage154_lock_sha256": base.sha256(args.stage154_lock),
        "stage158_lock_sha256": base.sha256(args.stage158_lock),
        "thresholds": {
            "stage154": float(lock154["selected_oof"]["threshold"]),
            "stage158": float(lock158["selected_oof"]["threshold"]),
        },
        "score_correlation": float(np.corrcoef(scores154, scores158)[0, 1]),
        "effects": effects,
    }
    assert not os.path.exists(args.output_json), args.output_json
    base.atomic_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage142_dump")
    parser.add_argument("stage150_source_dump")
    parser.add_argument("stage150_compact_dump")
    parser.add_argument("stage31_lock")
    parser.add_argument("stage33_lock")
    parser.add_argument("stage142_lock")
    parser.add_argument("stage154_lock")
    parser.add_argument("stage158_lock")
    parser.add_argument("output_json")
    parser.add_argument("--scope", choices=("all", "internal_test"), default="all")
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
