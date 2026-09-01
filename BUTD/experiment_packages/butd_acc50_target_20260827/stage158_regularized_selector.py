#!/usr/bin/env python3
"""Low-capacity scene-OOF selector for Stage142 versus Stage150.

Stage157 showed that the original depth-5, 500-tree selector became too
aggressive when refit on more data.  This stage changes model capacity rather
than the validation threshold: three preregistered, strongly regularized
fix-vs-break classifiers are selected using scene-disjoint OOF predictions on
the 85% development split.  The untouched 15% train-scene split remains the
only authorization gate for validation evaluation.
"""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof


STAGE = "158_regularized_scene_oof_stage142_stage150_source_selector"


def candidate_specs():
    """Return the fixed low-capacity search space used before validation."""
    return [
        {
            "variant": "depth2_leaves3",
            "n_estimators": 120,
            "learning_rate": 0.03,
            "num_leaves": 3,
            "max_depth": 2,
            "min_child_samples": 400,
            "colsample_bytree": 0.50,
            "reg_alpha": 2.0,
            "reg_lambda": 20.0,
            "seed": 15801,
        },
        {
            "variant": "depth3_leaves7",
            "n_estimators": 160,
            "learning_rate": 0.02,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 300,
            "colsample_bytree": 0.50,
            "reg_alpha": 2.0,
            "reg_lambda": 20.0,
            "seed": 15802,
        },
        {
            "variant": "depth2_leaves4",
            "n_estimators": 200,
            "learning_rate": 0.02,
            "num_leaves": 4,
            "max_depth": 2,
            "min_child_samples": 240,
            "colsample_bytree": 0.40,
            "reg_alpha": 4.0,
            "reg_lambda": 30.0,
            "seed": 15803,
        },
    ]


def fit_variant(features, old_ious, new_ious, spec):
    labels = base.labels(old_ious, new_ious)
    disagreement = labels["disagree050"]
    assert int(disagreement.sum()) > 100
    target = labels["fix050"][disagreement].astype(np.int32)
    weights = np.ones(len(target), dtype=np.float32)
    weights[target == 0] = 1.5
    learner = base.lgb.LGBMClassifier(
        objective="binary",
        n_estimators=spec["n_estimators"],
        learning_rate=spec["learning_rate"],
        num_leaves=spec["num_leaves"],
        max_depth=spec["max_depth"],
        min_child_samples=spec["min_child_samples"],
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=spec["colsample_bytree"],
        reg_alpha=spec["reg_alpha"],
        reg_lambda=spec["reg_lambda"],
        random_state=spec["seed"],
        n_jobs=16,
        verbosity=-1,
    )
    learner.fit(features[disagreement], target, sample_weight=weights)
    return {"name": "fix_vs_break_classifier", "models": [learner]}


def save_variant(candidate, spec, output_dir):
    path = os.path.join(output_dir, "selector_model_00.txt")
    candidate["models"][0].booster_.save_model(
        path, num_iteration=spec["n_estimators"]
    )
    return [{
        "path": os.path.abspath(path),
        "sha256": base.sha256(path),
        "iteration": int(spec["n_estimators"]),
    }]


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    raw_rows = base.option_ranker.load_rows(args.stage142_train_dump)
    assert len(raw_rows) == 36665
    stage142 = base.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, feature_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142,
    )
    development, test = oof.development_and_test(stage142["metas"])
    folds = oof.oof_folds(stage142["metas"], development, fold_count=5)
    specs = candidate_specs()
    oof_scores = {
        spec["variant"]: np.full(len(raw_rows), np.nan, dtype=np.float32)
        for spec in specs
    }
    fold_reports = []
    for item in folds:
        fit, heldout = item["fit"], item["heldout"]
        for spec in specs:
            candidate = fit_variant(
                matrix[fit], stage142["ious"][fit], new_ious[fit], spec
            )
            oof_scores[spec["variant"]][heldout] = base.predict_candidate(
                candidate, matrix[heldout]
            ).astype(np.float32)
        fold_reports.append({
            "fold": int(item["fold"]),
            "fit_rows": int(len(fit)),
            "heldout_rows": int(len(heldout)),
            "fit_scenes": int(len({
                stage142["metas"][index].scene_id for index in fit
            })),
            "heldout_scenes": int(len({
                stage142["metas"][index].scene_id for index in heldout
            })),
        })

    reports = []
    for spec in specs:
        scores = oof_scores[spec["variant"]][development]
        assert np.isfinite(scores).all()
        locked = base.choose_threshold(
            scores, stage142["ious"][development], new_ious[development]
        )
        reports.append({"variant": spec["variant"], "spec": spec, "oof": locked})
        print(json.dumps({
            "variant": spec["variant"], "oof": locked
        }, sort_keys=True))
    selected = max(reports, key=lambda item: (
        item["oof"]["selected"]["hits050"],
        item["oof"]["selected"]["hits025"],
        item["oof"]["selected"]["mean_iou"],
        -item["oof"]["changed_ratio"],
    ))

    final_candidate = fit_variant(
        matrix[development], stage142["ious"][development],
        new_ious[development], selected["spec"],
    )
    model_items = save_variant(final_candidate, selected["spec"], args.output_dir)
    reloaded = base.load_candidate({
        "selected_candidate": "fix_vs_break_classifier", "models": model_items,
    })
    test_scores = base.booster_predict(reloaded, matrix[test])
    internal_test = base.evaluate_scores(
        test_scores, stage142["ious"][test], new_ious[test],
        selected["oof"]["threshold"],
    )
    tolerance025 = max(1, int(math.ceil(0.001 * len(test))))
    fb = internal_test["fix_break"]
    internal_gate_pass = bool(
        internal_test["selected"]["hits050"]
        >= internal_test["default_stage142"]["hits050"] + 5
        and fb["fix_050"] > fb["break_050"]
        and internal_test["selected"]["hits025"]
        >= internal_test["default_stage142"]["hits025"] - tolerance025
        and internal_test["changed_ratio"] <= 0.20
    )
    development_labels = base.labels(
        stage142["ious"][development], new_ious[development]
    )
    label_counts = {
        name: int(np.asarray(value).sum())
        for name, value in development_labels.items()
        if name != "utility"
    }
    label_counts.update({
        "rows": int(len(development)),
        "utility_positive": int((development_labels["utility"] > 0).sum()),
        "utility_negative": int((development_labels["utility"] < 0).sum()),
    })
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_regularized_oof_lock",
        "protocol": (
            "post_stage154_engineering_low_capacity_fix_break_"
            "scene85_five_oof_scene15_gate_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": internal_gate_pass,
        "script": os.path.abspath(__file__),
        "script_sha256": base.sha256(os.path.abspath(__file__)),
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
        "selected_candidate": "fix_vs_break_classifier",
        "selected_variant": selected["variant"],
        "selected_spec": selected["spec"],
        "models": model_items,
        "selected_oof": selected["oof"],
        "all_oof_candidates": reports,
        "fold_reports": fold_reports,
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "split_sizes": {
            "development_oof": int(len(development)),
            "internal_test": int(len(test)),
        },
        "development_label_counts": label_counts,
    }
    lock_path = os.path.join(args.output_dir, "locked_regularized_selector.json")
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "selected_variant": selected["variant"],
        "selected_oof": selected["oof"],
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "validation_evaluation_authorized": internal_gate_pass,
        "feature_count": len(feature_names),
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = base.read_json(args.policy_lock)
    assert lock["stage"] == STAGE
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert base.sha256(lock["script"]) == lock["script_sha256"]
    raw_rows = base.option_ranker.load_rows(args.stage142_dump)
    stage142 = base.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, feature_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142, locked_feature_names=lock["source_feature_names"],
    )
    assert feature_names == lock["feature_names"]
    assert source_feature_names == lock["source_feature_names"]
    candidate = base.load_candidate(lock)
    scores = base.booster_predict(candidate, matrix)
    metrics = base.evaluate_scores(
        scores, stage142["ious"], new_ious,
        lock["selected_oof"]["threshold"],
    )
    count = len(raw_rows)
    strict025 = math.floor(0.5391 * count) + 1
    strict050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "158_regularized_selector_validation_eval",
        "status": "complete",
        "post_stage154_engineering_refinement": True,
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": base.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "selected_variant": lock["selected_variant"],
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
    train_parser = subparsers.add_parser("train")
    add_common(train_parser, train=True)
    train_parser.add_argument("output_dir")
    eval_parser = subparsers.add_parser("evaluate")
    add_common(eval_parser)
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
