#!/usr/bin/env python3
"""Scene-OOF fallback selector for Stage142 versus Stage150.

This fallback is preregistered before Stage153 validation completes.  It uses
the same inference-only feature matrix, preserves the original 15% scene test,
and replaces the single dev split with five scene-disjoint OOF folds over the
remaining 85% of ScanRefer train scenes.
"""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base


STAGE = "154_train_only_scene_oof_stage142_stage150_source_selector"


def development_and_test(metas):
    original = base.option_ranker.split_indices(metas)
    development = np.asarray(
        list(original["train"]) + list(original["dev"]), dtype=np.int64
    )
    test = np.asarray(original["test"], dtype=np.int64)
    assert len(development) + len(test) == len(metas)
    development_scenes = {metas[index].scene_id for index in development}
    test_scenes = {metas[index].scene_id for index in test}
    assert development_scenes.isdisjoint(test_scenes)
    return development, test


def oof_folds(metas, development, fold_count=5):
    development = np.asarray(development, dtype=np.int64)
    folds = []
    scene_to_fold = {}
    for index in development:
        scene = metas[int(index)].scene_id
        fold = base.option_ranker.scene_bucket(scene) % int(fold_count)
        scene_to_fold.setdefault(scene, fold)
        assert scene_to_fold[scene] == fold
    for fold in range(fold_count):
        heldout = np.asarray([
            int(index) for index in development
            if scene_to_fold[metas[int(index)].scene_id] == fold
        ], dtype=np.int64)
        fit = np.asarray([
            int(index) for index in development
            if scene_to_fold[metas[int(index)].scene_id] != fold
        ], dtype=np.int64)
        assert len(heldout) > 0 and len(fit) > 0
        heldout_scenes = {metas[index].scene_id for index in heldout}
        fit_scenes = {metas[index].scene_id for index in fit}
        assert heldout_scenes.isdisjoint(fit_scenes)
        folds.append({"fold": fold, "fit": fit, "heldout": heldout})
    covered = np.concatenate([item["heldout"] for item in folds])
    assert sorted(covered.tolist()) == sorted(development.tolist())
    return folds


def candidate_by_name(candidates, name):
    matches = [candidate for candidate in candidates if candidate["name"] == name]
    assert len(matches) == 1, (name, [item["name"] for item in candidates])
    return matches[0]


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
    development, test = development_and_test(stage142["metas"])
    folds = oof_folds(stage142["metas"], development, fold_count=5)
    candidate_names = (
        "utility_regression", "benefit_classifier",
        "fix_vs_break_classifier", "dual_hit050_advantage",
    )
    oof_scores = {
        name: np.full(len(raw_rows), np.nan, dtype=np.float32)
        for name in candidate_names
    }
    fold_reports = []
    for item in folds:
        fit = item["fit"]
        heldout = item["heldout"]
        candidates, _ = base.fit_candidates(
            matrix[fit], stage142["ious"][fit], new_ious[fit]
        )
        for name in candidate_names:
            candidate = candidate_by_name(candidates, name)
            oof_scores[name][heldout] = base.predict_candidate(
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
        del candidates

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

    final_candidates, development_labels = base.fit_candidates(
        matrix[development], stage142["ious"][development],
        new_ious[development],
    )
    final_candidate = candidate_by_name(final_candidates, selected["name"])
    model_items = base.save_candidate(final_candidate, args.output_dir)
    locked_candidate = base.load_candidate({
        "selected_candidate": selected["name"], "models": model_items,
    })
    test_scores = base.booster_predict(locked_candidate, matrix[test])
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
        "status": "complete_train_only_oof_lock",
        "protocol": (
            "scanrefer_train_scene85_five_scene_oof_lock_"
            "scene15_test_rich_conservative_selector_v1"
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
        "selected_candidate": selected["name"],
        "models": model_items,
        "selected_oof": selected["oof"],
        "all_oof_candidates": candidate_reports,
        "fold_reports": fold_reports,
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "split_sizes": {
            "development_oof": int(len(development)),
            "internal_test": int(len(test)),
        },
        "development_label_counts": label_counts,
    }
    lock_path = os.path.join(args.output_dir, "locked_oof_source_selector.json")
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "selected_candidate": selected["name"],
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
        raw_rows, stage142,
        locked_feature_names=lock["source_feature_names"],
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
        "stage": "154_oof_source_selector_validation_eval",
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
