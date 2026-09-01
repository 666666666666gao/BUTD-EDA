#!/usr/bin/env python3
"""Refit the authorized Stage154 selector on all ScanRefer train scenes."""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof


STAGE = "157_all_train_refit_stage154_fix_break_selector"
UPSTREAM_STAGE = "154_train_only_scene_oof_stage142_stage150_source_selector"


def validate_upstream_lock(lock):
    assert lock["stage"] == UPSTREAM_STAGE
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["internal_gate_pass"] is True
    assert lock["validation_evaluation_authorized"] is True
    return {
        "selected_candidate": lock["selected_candidate"],
        "threshold": float(lock["selected_oof"]["threshold"]),
    }


def refit(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    upstream = base.read_json(args.stage154_lock)
    identity = validate_upstream_lock(upstream)
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
    candidates, train_labels = base.fit_candidates(
        matrix, stage142["ious"], new_ious
    )
    selected = oof.candidate_by_name(
        candidates, identity["selected_candidate"]
    )
    model_items = base.save_candidate(selected, args.output_dir)
    reloaded = base.load_candidate({
        "selected_candidate": identity["selected_candidate"],
        "models": model_items,
    })
    smoke = base.booster_predict(reloaded, matrix[:32])
    assert len(smoke) == 32 and np.isfinite(smoke).all()
    label_counts = {
        name: int(np.asarray(value).sum())
        for name, value in train_labels.items()
        if name != "utility"
    }
    label_counts.update({
        "rows": len(raw_rows),
        "utility_positive": int((train_labels["utility"] > 0).sum()),
        "utility_negative": int((train_labels["utility"] < 0).sum()),
    })
    lock = {
        "stage": STAGE,
        "status": "complete_all_train_refit_lock",
        "protocol": (
            "stage154_scene_oof_selected_and_scene15_authorized_"
            "same_candidate_same_threshold_refit_all_train_v1"
        ),
        "selection_data_scope": "scanrefer_all_train_scenes_after_holdout_gate",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": True,
        "script": os.path.abspath(__file__),
        "script_sha256": base.sha256(os.path.abspath(__file__)),
        "stage154_lock": os.path.abspath(args.stage154_lock),
        "stage154_lock_sha256": base.sha256(args.stage154_lock),
        "inherited_internal_gate": upstream["internal_scene_test"],
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
        "selected_candidate": identity["selected_candidate"],
        "models": model_items,
        "selected_oof": upstream["selected_oof"],
        "locked_threshold": identity["threshold"],
        "refit_row_count": len(raw_rows),
        "train_label_counts": label_counts,
        "internal_gate_pass": True,
    }
    lock_path = os.path.join(args.output_dir, "locked_all_train_refit.json")
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "selected_candidate": identity["selected_candidate"],
        "locked_threshold": identity["threshold"],
        "refit_row_count": len(raw_rows),
        "validation_evaluation_authorized": True,
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = base.read_json(args.policy_lock)
    assert lock["stage"] == STAGE
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert lock["refit_row_count"] == 36665
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
    candidate = base.load_candidate(lock)
    scores = base.booster_predict(candidate, matrix)
    metrics = base.evaluate_scores(
        scores, stage142["ious"], new_ious, lock["locked_threshold"]
    )
    count = len(raw_rows)
    strict025 = math.floor(0.5391 * count) + 1
    strict050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "157_all_train_refit_selector_validation_eval",
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
    refit_parser = subparsers.add_parser("refit")
    add_common(refit_parser, train=True)
    refit_parser.add_argument("stage154_lock")
    refit_parser.add_argument("output_dir")
    eval_parser = subparsers.add_parser("evaluate")
    add_common(eval_parser)
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "refit":
        refit(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
