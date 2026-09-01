#!/usr/bin/env python3
"""Read-only overlap audit between the fixed Stage154 and Stage165 policies."""

import argparse
import json
import os

import numpy as np

import stage140_train_eval_nested_blend as nested
import stage153_train_source_selector as source
import stage154_oof_source_selector as oof
from train_joint_option_ranker import build_dataset, materialize


def pair_overlap(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    assert first.shape == second.shape
    oracle = np.maximum(first, second)
    result = {
        "stage154": source.summarize(first),
        "stage165": source.summarize(second),
        "pair_oracle": source.summarize(oracle),
        "stage165_relative_to_stage154": source.fix_break(
            first, second, np.ones(first.shape[0], dtype=bool)
        ),
        "threshold_partitions": {},
    }
    for threshold, name in ((0.25, "025"), (0.50, "050")):
        hit_first = source.hit(first, threshold)
        hit_second = source.hit(second, threshold)
        result["threshold_partitions"][name] = {
            "both_hit": int((hit_first & hit_second).sum()),
            "stage154_only_hit": int((hit_first & ~hit_second).sum()),
            "stage165_only_hit": int((~hit_first & hit_second).sum()),
            "neither_hit": int((~hit_first & ~hit_second).sum()),
        }
    return result


def stage165_ious(raw_rows, policy_path):
    policy = source.read_json(policy_path)
    assert policy["stage"] == (
        "165_stage164_nested_blend_train_dev_90pct_change_cap"
    )
    assert policy["validation_labels_used_for_selection"] is False
    group_features, group_ious, metas = build_dataset(
        raw_rows, max_candidates=8, require_scene=False
    )
    indices = np.arange(len(metas), dtype=np.int64)
    features, _, ious, groups, baselines = materialize(
        group_features, group_ious, metas, indices
    )
    boosters = nested.load_boosters(policy["provenance"])
    inner, pointwise = nested.component_scores(
        features, groups, boosters, policy["provenance"]
    )
    selected = policy["selected"]
    scores = (
        float(selected["inner_weight"]) * inner
        + float(selected["pointwise_weight"]) * pointwise
    )
    selected_ious, baseline_ious, gaps = nested.group_decisions(
        scores, ious, groups, baselines,
        float(selected["gate"]["threshold"]),
    )
    example_ids = np.asarray([meta.example_id for meta in metas], dtype=np.int64)
    return {
        "selected_ious": selected_ious,
        "baseline_ious": baseline_ious,
        "gaps": gaps,
        "example_ids": example_ids,
    }


def diagnose(args):
    raw_rows = source.option_ranker.load_rows(args.stage142_dump)
    stage142 = source.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, stage150_ious, feature_names, source_feature_names = (
        source.build_matrix(
            args.stage150_source_dump, args.stage150_compact_dump,
            raw_rows, stage142,
        )
    )
    lock154 = source.read_json(args.stage154_lock)
    assert lock154["validation_labels_used_for_selection"] is False
    assert feature_names == lock154["feature_names"]
    assert source_feature_names == lock154["source_feature_names"]
    model154 = source.load_candidate(lock154)
    scores154 = source.booster_predict(model154, matrix)
    mask154 = scores154 >= float(lock154["selected_oof"]["threshold"])
    selected154 = np.where(mask154, stage150_ious, stage142["ious"])

    stage165 = stage165_ious(raw_rows, args.stage165_policy)
    expected_ids = np.asarray([
        int(row.get("example_id", index)) for index, row in enumerate(raw_rows)
    ], dtype=np.int64)
    assert np.array_equal(stage165["example_ids"], expected_ids)
    selected165 = stage165["selected_ious"]
    assert selected154.shape == selected165.shape

    indices = np.arange(len(raw_rows), dtype=np.int64)
    if args.scope == "internal_test":
        _, indices = oof.development_and_test(stage142["metas"])
    report = {
        "stage": "166_stage154_stage165_overlap_diagnostic",
        "diagnostic_only": True,
        "scope": args.scope,
        "row_count": int(len(indices)),
        "validation_labels_used_for_policy_selection": False,
        "stage154_lock_sha256": source.sha256(args.stage154_lock),
        "stage165_policy_sha256": source.sha256(args.stage165_policy),
        "stage154_changed_count": int(mask154[indices].sum()),
        "stage165_changed_count_from_adapter": int(
            (stage165["gaps"][indices] >= float(
                source.read_json(args.stage165_policy)["selected"]["gate"]["threshold"]
            )).sum()
        ),
        "overlap": pair_overlap(selected154[indices], selected165[indices]),
    }
    assert not os.path.exists(args.output_json), args.output_json
    source.atomic_json(args.output_json, report)
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
    parser.add_argument("stage165_policy")
    parser.add_argument("output_json")
    parser.add_argument("--scope", choices=("all", "internal_test"), default="all")
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
