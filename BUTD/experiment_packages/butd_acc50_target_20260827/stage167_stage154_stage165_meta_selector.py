#!/usr/bin/env python3
"""Train-only scene-OOF selector between fixed Stage154 and Stage165."""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof
import stage166_stage154_stage165_overlap as overlap


STAGE = "167_train_only_oof_stage154_stage165_meta_selector"
META_FEATURE_NAMES = (
    "meta__stage154_score",
    "meta__stage154_margin",
    "meta__stage154_changed",
    "meta__stage165_gap",
    "meta__stage165_margin",
    "meta__stage165_changed",
)


def augment_features(matrix, scores154, threshold154, gaps165, threshold165):
    scores154 = np.asarray(scores154, dtype=np.float32)
    gaps165 = np.asarray(gaps165, dtype=np.float32)
    assert matrix.shape[0] == len(scores154) == len(gaps165)
    extras = np.column_stack((
        scores154,
        scores154 - float(threshold154),
        (scores154 >= float(threshold154)).astype(np.float32),
        gaps165,
        gaps165 - float(threshold165),
        (gaps165 >= float(threshold165)).astype(np.float32),
    )).astype(np.float32)
    extras = np.nan_to_num(extras, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.concatenate((np.asarray(matrix, dtype=np.float32), extras), axis=1)


def build_inputs(args):
    raw_rows = base.option_ranker.load_rows(args.stage142_dump)
    stage142 = base.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, stage150_ious, feature_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142,
    )
    lock154 = base.read_json(args.stage154_lock)
    lock165 = base.read_json(args.stage165_policy)
    assert lock154["validation_labels_used_for_selection"] is False
    assert lock165["validation_labels_used_for_selection"] is False
    assert feature_names == lock154["feature_names"]
    assert source_feature_names == lock154["source_feature_names"]
    candidate154 = base.load_candidate(lock154)
    scores154 = base.booster_predict(candidate154, matrix).astype(np.float32)
    threshold154 = float(lock154["selected_oof"]["threshold"])
    selected154 = np.where(
        scores154 >= threshold154, stage150_ious, stage142["ious"]
    ).astype(np.float32)

    stage165 = overlap.stage165_ious(raw_rows, args.stage165_policy)
    threshold165 = float(lock165["selected"]["gate"]["threshold"])
    selected165 = np.asarray(stage165["selected_ious"], dtype=np.float32)
    assert selected154.shape == selected165.shape
    augmented = augment_features(
        matrix, scores154, threshold154, stage165["gaps"], threshold165
    )
    augmented_names = list(feature_names) + list(META_FEATURE_NAMES)
    assert augmented.shape[1] == len(augmented_names)
    return {
        "raw_rows": raw_rows,
        "metas": stage142["metas"],
        "features": augmented,
        "feature_names": augmented_names,
        "source_feature_names": source_feature_names,
        "stage154_ious": selected154,
        "stage165_ious": selected165,
        "threshold154": threshold154,
        "threshold165": threshold165,
        "stage142_provenance": stage142["provenance"],
        "stage142_safe_config": stage142["safe_config"],
    }


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    data = build_inputs(args)
    old_ious = data["stage154_ious"]
    new_ious = data["stage165_ious"]
    development, test = oof.development_and_test(data["metas"])
    folds = oof.oof_folds(data["metas"], development, fold_count=5)
    candidate_names = (
        "utility_regression", "benefit_classifier",
        "fix_vs_break_classifier", "dual_hit050_advantage",
    )
    oof_scores = {
        name: np.full(len(old_ious), np.nan, dtype=np.float32)
        for name in candidate_names
    }
    fold_reports = []
    for item in folds:
        fit = item["fit"]
        heldout = item["heldout"]
        candidates, _ = base.fit_candidates(
            data["features"][fit], old_ious[fit], new_ious[fit]
        )
        for name in candidate_names:
            candidate = oof.candidate_by_name(candidates, name)
            oof_scores[name][heldout] = base.predict_candidate(
                candidate, data["features"][heldout]
            ).astype(np.float32)
        fold_reports.append({
            "fold": int(item["fold"]),
            "fit_rows": int(len(fit)),
            "heldout_rows": int(len(heldout)),
        })

    reports = []
    for name in candidate_names:
        scores = oof_scores[name][development]
        assert np.isfinite(scores).all()
        locked = base.choose_threshold(
            scores, old_ious[development], new_ious[development]
        )
        reports.append({"name": name, "oof": locked})
        print(json.dumps(reports[-1], sort_keys=True), flush=True)
    selected = max(reports, key=lambda item: (
        item["oof"]["selected"]["hits050"],
        item["oof"]["selected"]["hits025"],
        item["oof"]["selected"]["mean_iou"],
        -item["oof"]["changed_ratio"],
    ))

    final_candidates, labels = base.fit_candidates(
        data["features"][development],
        old_ious[development], new_ious[development],
    )
    final_candidate = oof.candidate_by_name(final_candidates, selected["name"])
    models = base.save_candidate(final_candidate, args.output_dir)
    locked_candidate = base.load_candidate({
        "selected_candidate": selected["name"], "models": models,
    })
    test_scores = base.booster_predict(
        locked_candidate, data["features"][test]
    )
    internal_test = base.evaluate_scores(
        test_scores, old_ious[test], new_ious[test],
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
    train_labels = base.labels(old_ious[development], new_ious[development])
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_oof_lock",
        "protocol": "scanrefer_train_scene85_five_scene_oof_stage154_stage165_v1",
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "validation_evaluation_authorized": internal_gate_pass,
        "internal_gate_pass": internal_gate_pass,
        "script": os.path.abspath(__file__),
        "script_sha256": base.sha256(os.path.abspath(__file__)),
        "feature_names": data["feature_names"],
        "feature_count": len(data["feature_names"]),
        "source_feature_names": data["source_feature_names"],
        "selected_candidate": selected["name"],
        "models": models,
        "selected_oof": selected["oof"],
        "all_oof_candidates": reports,
        "fold_reports": fold_reports,
        "internal_scene_test": internal_test,
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
        "stage150_source_dump_sha256": base.sha256(args.stage150_source_dump),
        "stage150_compact_dump_sha256": base.sha256(args.stage150_compact_dump),
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
    lock_path = os.path.join(args.output_dir, "locked_meta_selector.json")
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "selected_candidate": selected["name"],
        "selected_oof": selected["oof"],
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "validation_evaluation_authorized": internal_gate_pass,
    }, indent=2, sort_keys=True))


def evaluate(args):
    lock = base.read_json(args.policy_lock)
    assert lock["stage"] == STAGE
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert base.sha256(lock["script"]) == lock["script_sha256"]
    data = build_inputs(args)
    assert data["feature_names"] == lock["feature_names"]
    candidate = base.load_candidate(lock)
    scores = base.booster_predict(candidate, data["features"])
    result_metrics = base.evaluate_scores(
        scores, data["stage154_ious"], data["stage165_ious"],
        lock["selected_oof"]["threshold"],
    )
    count = len(data["raw_rows"])
    goals = {
        "acc025": math.floor(0.5391 * count) + 1,
        "acc050": math.floor(0.4241 * count) + 1,
    }
    result = {
        "stage": "167_stage154_stage165_meta_selector_validation_eval",
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": base.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "metrics": result_metrics,
        "strict_goal_hits": goals,
        "strict_goal_met_offline": bool(
            result_metrics["selected"]["hits025"] >= goals["acc025"]
            and result_metrics["selected"]["hits050"] >= goals["acc050"]
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
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    add_inputs(train_parser)
    train_parser.add_argument("output_dir")
    eval_parser = sub.add_parser("evaluate")
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
