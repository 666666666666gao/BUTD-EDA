#!/usr/bin/env python3
"""Hard predicted-Multiple gate for the fixed Stage171 selector.

The only change from Stage171 is an inference-safe semantic guard: Stage165
may replace Stage154 only when the detector exposes at least two boxes whose
predicted class matches the parsed text target class.  No model is retrained
and the Stage171 OOF score threshold is reused unchanged.
"""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof
import stage155_fold_routed_oof_selector as routed
import stage167_stage154_stage165_meta_selector as stage167
import stage168_risk_capped_meta_selector as stage168
import stage169_fold_routed_risk_selector as stage169
import stage170_explicit_fix_break_risk as stage170
import stage171_invariant_fix_break_risk as stage171


STAGE = "172_train_only_predicted_multiple_gate_stage154_stage165"
RESULT_STAGE = "172_predicted_multiple_gate_validation_eval"
MULTIPLICITY_FEATURE = "compact__text_target_detector_match_count"
MIN_MATCHED_TARGET_DETECTIONS = 2.0


def predicted_multiple_mask(counts):
    counts = np.asarray(counts, dtype=np.float32)
    assert counts.ndim == 1
    assert np.isfinite(counts).all()
    assert (counts >= 0.0).all()
    return counts >= MIN_MATCHED_TARGET_DETECTIONS


def multiplicity_feature_index(feature_names):
    matches = [
        index for index, name in enumerate(feature_names)
        if name == MULTIPLICITY_FEATURE
    ]
    assert len(matches) == 1, matches
    return int(matches[0])


def gate_scores(scores, counts):
    scores = np.asarray(scores, dtype=np.float32)
    eligible = predicted_multiple_mask(counts)
    assert scores.ndim == 1 and scores.shape == eligible.shape
    result = scores.copy()
    result[~eligible] = -np.inf
    return result


def annotate_fixed_oof_gate(report):
    fb = report["fix_break"]
    report["preserves_acc025"] = bool(
        report["selected"]["hits025"]
        >= report["default_stage142"]["hits025"]
    )
    report["risk_cap_gate"] = bool(
        report["changed_ratio"]
        <= stage168.MAX_OOF_CHANGED_RATIO + 1e-12
    )
    report["high_precision_gate"] = bool(
        fb["fix_050"] >= fb["break_050"] + 5
        and fb["fix_050"] >= (
            stage168.MIN_OOF_FIX_BREAK_RATIO
            * max(1, fb["break_050"])
        )
    )
    return report


def build_scored_inputs(args, policy):
    data = stage167.build_inputs(args)
    reduced, reduced_names, reduced_indices = (
        stage171.select_invariant_features(
            data["features"], data["feature_names"]
        )
    )
    assert reduced_names == policy["feature_names"]
    assert reduced_indices == policy["selected_feature_indices"]
    count_index = multiplicity_feature_index(data["feature_names"])
    counts = np.asarray(data["features"][:, count_index], dtype=np.float32)
    assert (counts >= 0.0).all()
    predictors = routed.load_fold_candidates(policy)
    raw_scores, fold_ids = routed.routed_predictions(
        predictors,
        reduced,
        data["metas"],
        stage170.booster_predict,
    )
    return data, gate_scores(raw_scores, counts), counts, fold_ids


def validate_parent(args):
    policy = base.read_json(args.stage171_policy)
    assert policy["stage"] == stage171.STAGE
    assert policy["validation_labels_used_for_selection"] is False
    assert policy["internal_gate_pass"] is False
    assert policy["validation_evaluation_authorized"] is False
    assert base.sha256(policy["script"]) == policy["script_sha256"]
    return policy


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    parent = validate_parent(args)
    data, scores, counts, fold_ids = build_scored_inputs(args, parent)
    old_ious = data["stage154_ious"]
    new_ious = data["stage165_ious"]
    development, test = oof.development_and_test(data["metas"])
    threshold = float(parent["selected_oof"]["threshold"])
    development_report = annotate_fixed_oof_gate(base.evaluate_scores(
        scores[development],
        old_ious[development],
        new_ious[development],
        threshold,
    ))
    internal_test = base.evaluate_scores(
        scores[test], old_ious[test], new_ious[test], threshold
    )
    gate = bool(
        development_report["preserves_acc025"]
        and development_report["risk_cap_gate"]
        and development_report["high_precision_gate"]
        and stage169.internal_gate(internal_test)
    )
    eligible = predicted_multiple_mask(counts)
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_predicted_multiple_gate_lock",
        "protocol": (
            "scanrefer_train_scene_stage171_fixed_threshold_detector_"
            "predicted_multiple_count_ge_2_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": gate,
        "internal_gate_pass": gate,
        "script": os.path.abspath(__file__),
        "script_sha256": base.sha256(os.path.abspath(__file__)),
        "parent_policy": os.path.abspath(args.stage171_policy),
        "parent_policy_sha256": base.sha256(args.stage171_policy),
        "feature_names": parent["feature_names"],
        "feature_count": parent["feature_count"],
        "selected_feature_indices": parent["selected_feature_indices"],
        "source_feature_names": parent["source_feature_names"],
        "fold_count": parent["fold_count"],
        "routing": parent["routing"],
        "selected_candidate": parent["selected_candidate"],
        "models": parent["models"],
        "selected_oof": development_report,
        "internal_scene_test": internal_test,
        "multiplicity_gate": {
            "feature": MULTIPLICITY_FEATURE,
            "minimum_matched_target_detections": (
                MIN_MATCHED_TARGET_DETECTIONS
            ),
            "uses_ground_truth": False,
            "rule_selected_without_label_scan": True,
            "parent_score_threshold_unchanged": threshold,
            "development_eligible_count": int(eligible[development].sum()),
            "development_eligible_ratio": float(eligible[development].mean()),
            "internal_eligible_count": int(eligible[test].sum()),
            "internal_eligible_ratio": float(eligible[test].mean()),
        },
        "internal_test_fold_counts": {
            str(fold): int((fold_ids[test] == fold).sum())
            for fold in range(routed.FOLD_COUNT)
        },
        "risk_protocol": {
            "score": parent["risk_protocol"]["score"],
            "threshold": threshold,
            "threshold_reselected": False,
            "only_change_from_stage171": (
                "hard_predicted_multiple_gate_target_class_detector_count_ge_2"
            ),
            "max_oof_changed_ratio": stage168.MAX_OOF_CHANGED_RATIO,
            "max_internal_changed_ratio": stage168.MAX_INTERNAL_CHANGED_RATIO,
            "min_oof_fix_break_ratio": stage168.MIN_OOF_FIX_BREAK_RATIO,
            "min_internal_fix_break_ratio": (
                stage168.MIN_INTERNAL_FIX_BREAK_RATIO
            ),
        },
        "stage154_lock": parent["stage154_lock"],
        "stage154_lock_sha256": parent["stage154_lock_sha256"],
        "stage165_policy": parent["stage165_policy"],
        "stage165_policy_sha256": parent["stage165_policy_sha256"],
        "stage31_lock_sha256": parent["stage31_lock_sha256"],
        "stage33_lock_sha256": parent["stage33_lock_sha256"],
        "stage142_lock_sha256": parent["stage142_lock_sha256"],
        "stage142_provenance": parent["stage142_provenance"],
        "stage142_safe_config": parent["stage142_safe_config"],
        "stage142_dump_sha256": parent["stage142_dump_sha256"],
        "stage150_source_dump_sha256": parent[
            "stage150_source_dump_sha256"
        ],
        "stage150_compact_dump_sha256": parent[
            "stage150_compact_dump_sha256"
        ],
        "threshold154": parent["threshold154"],
        "threshold165": parent["threshold165"],
        "split_sizes": parent["split_sizes"],
    }
    lock_path = os.path.join(
        args.output_dir, "locked_predicted_multiple_gate.json"
    )
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "multiplicity_gate": lock["multiplicity_gate"],
        "selected_oof": development_report,
        "internal_scene_test": internal_test,
        "internal_gate_pass": gate,
        "validation_evaluation_authorized": gate,
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = base.read_json(args.policy_lock)
    assert lock["stage"] == STAGE
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert base.sha256(lock["script"]) == lock["script_sha256"]
    parent = base.read_json(args.stage171_policy)
    assert base.sha256(args.stage171_policy) == lock["parent_policy_sha256"]
    assert parent["models"] == lock["models"]
    data, scores, counts, fold_ids = build_scored_inputs(args, parent)
    metrics = base.evaluate_scores(
        scores,
        data["stage154_ious"],
        data["stage165_ious"],
        lock["selected_oof"]["threshold"],
    )
    count = len(data["raw_rows"])
    goals = {
        "acc025": math.floor(0.5391 * count) + 1,
        "acc050": math.floor(0.4241 * count) + 1,
    }
    result = {
        "stage": RESULT_STAGE,
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": base.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "metrics": metrics,
        "validation_predicted_multiple_count": int(
            predicted_multiple_mask(counts).sum()
        ),
        "validation_fold_counts": {
            str(fold): int((fold_ids == fold).sum())
            for fold in range(routed.FOLD_COUNT)
        },
        "strict_goal_hits": goals,
        "strict_goal_met_offline": bool(
            metrics["selected"]["hits025"] >= goals["acc025"]
            and metrics["selected"]["hits050"] >= goals["acc050"]
        ),
    }
    assert not os.path.exists(args.output_json), args.output_json
    base.atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def add_inputs(parser):
    parser.add_argument("stage142_dump")
    parser.add_argument("stage150_source_dump")
    parser.add_argument("stage150_compact_dump")
    parser.add_argument("stage31_lock")
    parser.add_argument("stage33_lock")
    parser.add_argument("stage142_lock")
    parser.add_argument("stage154_lock")
    parser.add_argument("stage165_policy")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    add_inputs(train_parser)
    train_parser.add_argument("stage171_policy")
    train_parser.add_argument("output_dir")
    eval_parser = subparsers.add_parser("evaluate")
    add_inputs(eval_parser)
    eval_parser.add_argument("stage171_policy")
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
