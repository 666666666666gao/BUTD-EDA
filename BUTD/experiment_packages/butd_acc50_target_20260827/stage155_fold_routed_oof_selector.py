#!/usr/bin/env python3
"""Train-only scene-OOF selector with fold-routed inference.

Stage154 locks a threshold from scene-disjoint OOF predictions, then refits one
model on all development scenes.  This preregistered fallback preserves the
OOF calibration contract at inference: each scene is routed by the same stable
scene hash to the corresponding model that excluded that hash bucket.
"""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof


STAGE = "155_train_only_fold_routed_oof_stage142_stage150_selector"
FOLD_COUNT = 5


def routed_predictions(predictors, features, metas, predict):
    """Return predictions from the hash-routed fold model for every row."""
    assert set(predictors) == set(range(FOLD_COUNT)), sorted(predictors)
    assert len(features) == len(metas)
    folds = np.asarray([
        base.option_ranker.scene_bucket(meta.scene_id) % FOLD_COUNT
        for meta in metas
    ], dtype=np.int64)
    scores = np.full(len(metas), np.nan, dtype=np.float32)
    for fold in range(FOLD_COUNT):
        indices = np.flatnonzero(folds == fold)
        if len(indices) == 0:
            continue
        values = np.asarray(
            predict(predictors[fold], features[indices]), dtype=np.float32
        ).reshape(-1)
        assert len(values) == len(indices)
        scores[indices] = values
    assert np.isfinite(scores).all()
    return scores, folds


def group_model_items(items, fold_count=FOLD_COUNT):
    """Validate and group flat lock-file model entries by routed fold."""
    grouped = []
    for fold in range(int(fold_count)):
        group = [item for item in items if int(item["fold"]) == fold]
        assert len(group) > 0, fold
        group = sorted(group, key=lambda item: int(item["fold_model_index"]))
        assert [int(item["fold_model_index"]) for item in group] == list(
            range(len(group))
        )
        grouped.append(group)
    assert sum(len(group) for group in grouped) == len(items)
    return grouped


def save_fold_candidates(candidates, output_dir):
    assert set(candidates) == set(range(FOLD_COUNT))
    model_items = []
    for fold in range(FOLD_COUNT):
        fold_dir = os.path.join(output_dir, "fold_{:02d}".format(fold))
        os.makedirs(fold_dir)
        saved = base.save_candidate(candidates[fold], fold_dir)
        for index, item in enumerate(saved):
            entry = dict(item)
            entry["fold"] = fold
            entry["fold_model_index"] = index
            model_items.append(entry)
    group_model_items(model_items)
    return model_items


def load_fold_candidates(lock):
    groups = group_model_items(lock["models"])
    return {
        fold: base.load_candidate({
            "selected_candidate": lock["selected_candidate"],
            "models": group,
        })
        for fold, group in enumerate(groups)
    }


def internal_gate(report, count):
    tolerance025 = max(1, int(math.ceil(0.001 * count)))
    fb = report["fix_break"]
    return bool(
        report["selected"]["hits050"]
        >= report["default_stage142"]["hits050"] + 5
        and fb["fix_050"] > fb["break_050"]
        and report["selected"]["hits025"]
        >= report["default_stage142"]["hits025"] - tolerance025
        and report["changed_ratio"] <= 0.20
    )


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
    folds = oof.oof_folds(
        stage142["metas"], development, fold_count=FOLD_COUNT
    )
    candidate_names = (
        "utility_regression", "benefit_classifier",
        "fix_vs_break_classifier", "dual_hit050_advantage",
    )
    oof_scores = {
        name: np.full(len(raw_rows), np.nan, dtype=np.float32)
        for name in candidate_names
    }
    fold_candidates = {name: {} for name in candidate_names}
    fold_reports = []
    for item in folds:
        fold = int(item["fold"])
        fit = item["fit"]
        heldout = item["heldout"]
        candidates, _ = base.fit_candidates(
            matrix[fit], stage142["ious"][fit], new_ious[fit]
        )
        for name in candidate_names:
            candidate = oof.candidate_by_name(candidates, name)
            fold_candidates[name][fold] = candidate
            oof_scores[name][heldout] = base.predict_candidate(
                candidate, matrix[heldout]
            ).astype(np.float32)
        fold_reports.append({
            "fold": fold,
            "fit_rows": int(len(fit)),
            "heldout_rows": int(len(heldout)),
            "fit_scenes": int(len({
                stage142["metas"][index].scene_id for index in fit
            })),
            "heldout_scenes": int(len({
                stage142["metas"][index].scene_id for index in heldout
            })),
        })

    candidate_reports = []
    for name in candidate_names:
        scores = oof_scores[name][development]
        assert np.isfinite(scores).all()
        locked = base.choose_threshold(
            scores, stage142["ious"][development], new_ious[development]
        )
        candidate_reports.append({"name": name, "oof": locked})
        print(json.dumps({"name": name, "oof": locked}, sort_keys=True))
    selected = max(candidate_reports, key=lambda item: (
        item["oof"]["selected"]["hits050"],
        item["oof"]["selected"]["hits025"],
        item["oof"]["selected"]["mean_iou"],
        -item["oof"]["changed_ratio"],
    ))
    selected_folds = fold_candidates[selected["name"]]
    model_items = save_fold_candidates(selected_folds, args.output_dir)

    test_metas = [stage142["metas"][int(index)] for index in test]
    test_scores, test_fold_ids = routed_predictions(
        selected_folds, matrix[test], test_metas, base.predict_candidate
    )
    internal_test = base.evaluate_scores(
        test_scores, stage142["ious"][test], new_ious[test],
        selected["oof"]["threshold"],
    )
    gate = internal_gate(internal_test, len(test))
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
        "status": "complete_train_only_fold_routed_oof_lock",
        "protocol": (
            "scanrefer_train_scene85_five_scene_oof_lock_"
            "scene_hash_routed_fold_model_scene15_test_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": gate,
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
        "fold_count": FOLD_COUNT,
        "routing": "scene_bucket_mod_5",
        "selected_candidate": selected["name"],
        "models": model_items,
        "selected_oof": selected["oof"],
        "all_oof_candidates": candidate_reports,
        "fold_reports": fold_reports,
        "internal_scene_test": internal_test,
        "internal_test_fold_counts": {
            str(fold): int((test_fold_ids == fold).sum())
            for fold in range(FOLD_COUNT)
        },
        "internal_gate_pass": gate,
        "split_sizes": {
            "development_oof": int(len(development)),
            "internal_test": int(len(test)),
        },
        "development_label_counts": label_counts,
    }
    lock_path = os.path.join(
        args.output_dir, "locked_fold_routed_oof_selector.json"
    )
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "selected_candidate": selected["name"],
        "selected_oof": selected["oof"],
        "internal_scene_test": internal_test,
        "internal_gate_pass": gate,
        "validation_evaluation_authorized": gate,
        "feature_count": len(feature_names),
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = base.read_json(args.policy_lock)
    assert lock["stage"] == STAGE
    assert lock["fold_count"] == FOLD_COUNT
    assert lock["routing"] == "scene_bucket_mod_5"
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
    candidates = load_fold_candidates(lock)
    scores, fold_ids = routed_predictions(
        candidates, matrix, stage142["metas"], base.booster_predict
    )
    metrics = base.evaluate_scores(
        scores, stage142["ious"], new_ious,
        lock["selected_oof"]["threshold"],
    )
    count = len(raw_rows)
    strict025 = math.floor(0.5391 * count) + 1
    strict050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "155_fold_routed_oof_selector_validation_eval",
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": base.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "metrics": metrics,
        "validation_fold_counts": {
            str(fold): int((fold_ids == fold).sum())
            for fold in range(FOLD_COUNT)
        },
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
