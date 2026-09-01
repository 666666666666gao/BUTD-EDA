#!/usr/bin/env python3
"""Mean ensemble of the five train-only Stage155 OOF fold selectors."""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof
import stage155_fold_routed_oof_selector as routed


STAGE = "156_train_only_five_fold_mean_oof_stage142_stage150_selector"


def ensemble_predictions(predictors, features, predict):
    assert set(predictors) == set(range(routed.FOLD_COUNT)), sorted(predictors)
    predictions = []
    for fold in range(routed.FOLD_COUNT):
        values = np.asarray(
            predict(predictors[fold], features), dtype=np.float32
        ).reshape(-1)
        assert len(values) == len(features)
        assert np.isfinite(values).all()
        predictions.append(values)
    scores = np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float32)
    assert np.isfinite(scores).all()
    return scores


def build_lock(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    upstream = base.read_json(args.stage155_lock)
    assert upstream["stage"] == (
        "155_train_only_fold_routed_oof_stage142_stage150_selector"
    )
    assert upstream["validation_labels_used_for_selection"] is False
    assert base.sha256(upstream["script"]) == upstream["script_sha256"]
    os.makedirs(args.output_dir)
    raw_rows = base.option_ranker.load_rows(args.stage142_train_dump)
    assert len(raw_rows) == 36665
    stage142 = base.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, feature_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142,
        locked_feature_names=upstream["source_feature_names"],
    )
    assert feature_names == upstream["feature_names"]
    assert source_feature_names == upstream["source_feature_names"]
    _, test = oof.development_and_test(stage142["metas"])
    candidates = routed.load_fold_candidates(upstream)
    scores = ensemble_predictions(
        candidates, matrix[test], base.booster_predict
    )
    internal_test = base.evaluate_scores(
        scores, stage142["ious"][test], new_ious[test],
        upstream["selected_oof"]["threshold"],
    )
    gate = routed.internal_gate(internal_test, len(test))
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_five_fold_mean_lock",
        "protocol": (
            "scanrefer_train_stage155_locked_models_mean_probability_"
            "same_threshold_scene15_test_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": gate,
        "script": os.path.abspath(__file__),
        "script_sha256": base.sha256(os.path.abspath(__file__)),
        "stage155_lock": os.path.abspath(args.stage155_lock),
        "stage155_lock_sha256": base.sha256(args.stage155_lock),
        "stage142_train_dump": os.path.abspath(args.stage142_train_dump),
        "stage142_train_dump_sha256": base.sha256(args.stage142_train_dump),
        "stage150_source_dump": os.path.abspath(args.stage150_source_dump),
        "stage150_source_dump_sha256": base.sha256(args.stage150_source_dump),
        "stage150_compact_dump": os.path.abspath(args.stage150_compact_dump),
        "stage150_compact_dump_sha256": base.sha256(args.stage150_compact_dump),
        "stage142_lock": os.path.abspath(args.stage142_lock),
        "stage142_lock_sha256": base.sha256(args.stage142_lock),
        "stage31_lock_sha256": base.sha256(args.stage31_lock),
        "stage33_lock_sha256": base.sha256(args.stage33_lock),
        "provenance": stage142["provenance"],
        "safe_config": stage142["safe_config"],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "source_feature_names": source_feature_names,
        "selected_candidate": upstream["selected_candidate"],
        "models": upstream["models"],
        "selected_oof": upstream["selected_oof"],
        "aggregation": "arithmetic_mean_of_five_fold_probabilities",
        "internal_scene_test": internal_test,
        "internal_gate_pass": gate,
        "split_sizes": upstream["split_sizes"],
    }
    lock_path = os.path.join(
        args.output_dir, "locked_five_fold_mean_selector.json"
    )
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "internal_scene_test": internal_test,
        "internal_gate_pass": gate,
        "validation_evaluation_authorized": gate,
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = base.read_json(args.policy_lock)
    assert lock["stage"] == STAGE
    assert lock["aggregation"] == (
        "arithmetic_mean_of_five_fold_probabilities"
    )
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert base.sha256(lock["script"]) == lock["script_sha256"]
    raw_rows = base.option_ranker.load_rows(args.stage142_dump)
    stage142 = base.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, feature_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142,
        locked_feature_names=lock["source_feature_names"],
    )
    assert feature_names == lock["feature_names"]
    assert source_feature_names == lock["source_feature_names"]
    candidates = routed.load_fold_candidates(lock)
    scores = ensemble_predictions(candidates, matrix, base.booster_predict)
    metrics = base.evaluate_scores(
        scores, stage142["ious"], new_ious,
        lock["selected_oof"]["threshold"],
    )
    count = len(raw_rows)
    strict025 = math.floor(0.5391 * count) + 1
    strict050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "156_five_fold_mean_selector_validation_eval",
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": base.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "metrics": metrics,
        "strict_goal_hits": {"acc025": strict025, "acc050": strict050},
        "strict_goal_met_offline": bool(
            metrics["selected"]["hits025"] >= strict025
            and metrics["selected"]["hits050"] >= strict050
        ),
    }
    assert not os.path.exists(args.output_json), args.output_json
    base.atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def add_common(parser, train=False):
    parser.add_argument("stage142_train_dump" if train else "stage142_dump")
    parser.add_argument("stage150_source_dump")
    parser.add_argument("stage150_compact_dump")
    parser.add_argument("stage31_lock")
    parser.add_argument("stage33_lock")
    parser.add_argument("stage142_lock")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock_parser = subparsers.add_parser("lock")
    add_common(lock_parser, train=True)
    lock_parser.add_argument("stage155_lock")
    lock_parser.add_argument("output_dir")
    eval_parser = subparsers.add_parser("evaluate")
    add_common(eval_parser)
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "lock":
        build_lock(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
