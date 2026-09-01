#!/usr/bin/env python3
"""Explicit train-only fix/break risk selector for Stage154 vs Stage165.

The inference score is a fixed log-odds ratio between two independently
trained heads: P(Stage165 fixes an @0.50 miss) and P(Stage165 breaks an
@0.50 hit).  All features, scene splits, routing, thresholds and safety gates
remain identical to Stage169.
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


STAGE = "170_train_only_explicit_fix_break_risk_stage154_stage165"
RESULT_STAGE = "170_explicit_fix_break_risk_validation_eval"
CANDIDATE_NAME = "explicit_fix_break_odds"


def log_odds_ratio(fix_probability, break_probability):
    fix_probability = np.clip(
        np.asarray(fix_probability, dtype=np.float32), 1e-5, 1.0 - 1e-5
    )
    break_probability = np.clip(
        np.asarray(break_probability, dtype=np.float32), 1e-5, 1.0 - 1e-5
    )
    fix_logit = np.log(fix_probability / (1.0 - fix_probability))
    break_logit = np.log(break_probability / (1.0 - break_probability))
    scores = fix_logit - break_logit
    assert np.isfinite(scores).all()
    return scores.astype(np.float32)


def fit_candidate(features, old_ious, new_ious):
    labels = base.labels(old_ious, new_ious)
    fix_model = base.lgb.LGBMClassifier(
        objective="binary", **base.learner_common(17001)
    )
    break_model = base.lgb.LGBMClassifier(
        objective="binary", **base.learner_common(17002)
    )
    # Keep unweighted empirical probabilities.  The OOF risk threshold handles
    # rarity; class reweighting would destroy the intended odds interpretation.
    fix_model.fit(features, labels["fix050"].astype(np.int32))
    break_model.fit(features, labels["break050"].astype(np.int32))
    return {
        "name": CANDIDATE_NAME,
        "models": [fix_model, break_model],
    }


def predict_candidate(candidate, features):
    assert candidate["name"] == CANDIDATE_NAME
    fix_model, break_model = candidate["models"]
    fix_probability = fix_model.predict_proba(features)[:, 1]
    break_probability = break_model.predict_proba(features)[:, 1]
    return log_odds_ratio(fix_probability, break_probability)


def booster_predict(candidate, features):
    assert candidate["name"] == CANDIDATE_NAME
    fix_model, break_model = candidate["models"]
    return log_odds_ratio(
        fix_model.predict(features, num_iteration=500),
        break_model.predict(features, num_iteration=500),
    )


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    data = stage167.build_inputs(args)
    old_ious = data["stage154_ious"]
    new_ious = data["stage165_ious"]
    development, test = oof.development_and_test(data["metas"])
    folds = oof.oof_folds(
        data["metas"], development, fold_count=routed.FOLD_COUNT
    )
    oof_scores = np.full(len(old_ious), np.nan, dtype=np.float32)
    fold_candidates = {}
    fold_reports = []
    for item in folds:
        fold = int(item["fold"])
        fit, heldout = item["fit"], item["heldout"]
        candidate = fit_candidate(
            data["features"][fit], old_ious[fit], new_ious[fit]
        )
        fold_candidates[fold] = candidate
        oof_scores[heldout] = predict_candidate(
            candidate, data["features"][heldout]
        )
        fold_reports.append({
            "fold": fold,
            "fit_rows": int(len(fit)),
            "heldout_rows": int(len(heldout)),
            "fit_scenes": int(len({
                data["metas"][int(index)].scene_id for index in fit
            })),
            "heldout_scenes": int(len({
                data["metas"][int(index)].scene_id for index in heldout
            })),
        })

    development_scores = oof_scores[development]
    assert np.isfinite(development_scores).all()
    selected_oof = stage168.choose_risk_capped_threshold(
        development_scores,
        old_ious[development],
        new_ious[development],
    )
    print(json.dumps({
        "name": CANDIDATE_NAME, "oof": selected_oof,
    }, sort_keys=True), flush=True)
    models = routed.save_fold_candidates(fold_candidates, args.output_dir)

    test_metas = [data["metas"][int(index)] for index in test]
    test_scores, test_fold_ids = routed.routed_predictions(
        fold_candidates,
        data["features"][test],
        test_metas,
        predict_candidate,
    )
    internal_test = base.evaluate_scores(
        test_scores,
        old_ious[test],
        new_ious[test],
        selected_oof["threshold"],
    )
    gate = bool(
        selected_oof["high_precision_gate"]
        and stage169.internal_gate(internal_test)
    )
    train_labels = base.labels(old_ious[development], new_ious[development])
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_explicit_fix_break_risk_lock",
        "protocol": (
            "scanrefer_train_scene85_five_scene_oof_stage154_stage165_"
            "explicit_fix_break_odds_fold_routed_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": gate,
        "internal_gate_pass": gate,
        "script": os.path.abspath(__file__),
        "script_sha256": base.sha256(os.path.abspath(__file__)),
        "feature_names": data["feature_names"],
        "feature_count": len(data["feature_names"]),
        "source_feature_names": data["source_feature_names"],
        "fold_count": routed.FOLD_COUNT,
        "routing": "scene_bucket_mod_5_to_excluded_fold_model",
        "selected_candidate": CANDIDATE_NAME,
        "models": models,
        "selected_oof": selected_oof,
        "fold_reports": fold_reports,
        "internal_scene_test": internal_test,
        "internal_test_fold_counts": {
            str(fold): int((test_fold_ids == fold).sum())
            for fold in range(routed.FOLD_COUNT)
        },
        "risk_protocol": {
            "score": "logit_p_fix050_minus_logit_p_break050",
            "class_weights": "none_empirical_probability",
            "max_oof_changed_ratio": stage168.MAX_OOF_CHANGED_RATIO,
            "max_internal_changed_ratio": stage168.MAX_INTERNAL_CHANGED_RATIO,
            "min_oof_fix_break_ratio": stage168.MIN_OOF_FIX_BREAK_RATIO,
            "min_internal_fix_break_ratio": (
                stage168.MIN_INTERNAL_FIX_BREAK_RATIO
            ),
            "only_change_from_stage169": (
                "utility_regression_replaced_by_explicit_fix_break_odds"
            ),
        },
        "stage154_lock": os.path.abspath(args.stage154_lock),
        "stage154_lock_sha256": base.sha256(args.stage154_lock),
        "stage165_policy": os.path.abspath(args.stage165_policy),
        "stage165_policy_sha256": base.sha256(args.stage165_policy),
        "stage31_lock_sha256": base.sha256(args.stage31_lock),
        "stage33_lock_sha256": base.sha256(args.stage33_lock),
        "stage142_lock_sha256": base.sha256(args.stage142_lock),
        "stage142_provenance": data["stage142_provenance"],
        "stage142_safe_config": data["stage142_safe_config"],
        "stage142_dump_sha256": base.sha256(args.stage142_dump),
        "stage150_source_dump_sha256": base.sha256(
            args.stage150_source_dump
        ),
        "stage150_compact_dump_sha256": base.sha256(
            args.stage150_compact_dump
        ),
        "threshold154": data["threshold154"],
        "threshold165": data["threshold165"],
        "split_sizes": {
            "development_oof": int(len(development)),
            "internal_test": int(len(test)),
        },
        "development_label_counts": {
            "rows": int(len(development)),
            "fix050": int(train_labels["fix050"].sum()),
            "break050": int(train_labels["break050"].sum()),
            "disagree050": int(train_labels["disagree050"].sum()),
        },
    }
    lock_path = os.path.join(
        args.output_dir, "locked_explicit_fix_break_risk.json"
    )
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "selected_oof": selected_oof,
        "internal_scene_test": internal_test,
        "internal_gate_pass": gate,
        "validation_evaluation_authorized": gate,
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = base.read_json(args.policy_lock)
    assert lock["stage"] == STAGE
    assert lock["fold_count"] == routed.FOLD_COUNT
    assert lock["routing"] == "scene_bucket_mod_5_to_excluded_fold_model"
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert base.sha256(lock["script"]) == lock["script_sha256"]
    data = stage167.build_inputs(args)
    assert data["feature_names"] == lock["feature_names"]
    predictors = routed.load_fold_candidates(lock)
    scores, fold_ids = routed.routed_predictions(
        predictors,
        data["features"],
        data["metas"],
        booster_predict,
    )
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
    train_parser.add_argument("output_dir")
    eval_parser = subparsers.add_parser("evaluate")
    add_inputs(eval_parser)
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
