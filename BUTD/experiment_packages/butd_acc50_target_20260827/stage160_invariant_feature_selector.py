#!/usr/bin/env python3
"""Train a scene-OOF source selector on inference-invariant feature groups.

The full Stage154 selector uses 1,116 inference-safe fields, including absolute
box centers and sizes whose distributions can drift between ScanRefer train
and validation scenes.  This stage holds the classifier and scientific gates
fixed while selecting among three preregistered feature groups that exclude
absolute geometry.
"""

import argparse
import json
import math
import os

import numpy as np

import stage153_train_source_selector as base
import stage154_oof_source_selector as oof


STAGE = "160_invariant_feature_scene_oof_stage142_stage150_selector"
SOURCE_NAMES = (
    "acd", "base", "contrastive_base", "fused", "quality",
    "target_detector", "target_detector_logit",
)


def is_absolute_geometry(name):
    if name.startswith("compact__selected_log_size_"):
        return True
    if name == "compact__selected_log_volume":
        return True
    if name.startswith("stage142_option__option_log_size_"):
        return True
    if name == "stage142_option__option_log_volume":
        return True
    for source in SOURCE_NAMES:
        prefix = "source__{}_top_".format(source)
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if (
                suffix.startswith("size_")
                or suffix.startswith("center_")
                or suffix == "volume"
            ):
                return True
    return False


def is_geometry(name):
    return any(token in name for token in (
        "center", "size", "volume", "box_iou", "log_size_ratio",
    ))


def feature_groups(feature_names):
    feature_names = list(feature_names)
    relative = [
        name for name in feature_names if not is_absolute_geometry(name)
    ]
    score_only = [name for name in relative if not is_geometry(name)]
    compact_relative = [
        name for name in relative if not name.startswith("source__")
    ]
    groups = {
        "score_only": score_only,
        "relative_geometry": relative,
        "compact_relative_geometry": compact_relative,
    }
    assert all(groups.values())
    assert len({tuple(value) for value in groups.values()}) == len(groups)
    return groups


def feature_indices(all_names, selected_names):
    position = {name: index for index, name in enumerate(all_names)}
    assert len(position) == len(all_names)
    result = np.asarray([position[name] for name in selected_names], dtype=np.int64)
    assert len(result) == len(selected_names)
    return result


def fit_classifier(features, old_ious, new_ious):
    labels = base.labels(old_ious, new_ious)
    disagreement = labels["disagree050"]
    assert int(disagreement.sum()) > 100
    target = labels["fix050"][disagreement].astype(np.int32)
    weights = np.ones(len(target), dtype=np.float32)
    weights[target == 0] = 1.5
    learner = base.lgb.LGBMClassifier(
        objective="binary", **base.learner_common(16001)
    )
    learner.fit(features[disagreement], target, sample_weight=weights)
    return {"name": "fix_vs_break_classifier", "models": [learner]}


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    raw_rows = base.option_ranker.load_rows(args.stage142_train_dump)
    assert len(raw_rows) == 36665
    stage142 = base.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, all_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142,
    )
    groups = feature_groups(all_names)
    group_indices = {
        name: feature_indices(all_names, names) for name, names in groups.items()
    }
    development, test = oof.development_and_test(stage142["metas"])
    folds = oof.oof_folds(stage142["metas"], development, fold_count=5)
    oof_scores = {
        name: np.full(len(raw_rows), np.nan, dtype=np.float32)
        for name in groups
    }
    fold_reports = []
    for item in folds:
        fit, heldout = item["fit"], item["heldout"]
        for name, indices in group_indices.items():
            candidate = fit_classifier(
                matrix[fit][:, indices], stage142["ious"][fit], new_ious[fit]
            )
            oof_scores[name][heldout] = base.predict_candidate(
                candidate, matrix[heldout][:, indices]
            ).astype(np.float32)
        fold_reports.append({
            "fold": int(item["fold"]),
            "fit_rows": int(len(fit)),
            "heldout_rows": int(len(heldout)),
        })

    reports = []
    for name in groups:
        scores = oof_scores[name][development]
        assert np.isfinite(scores).all()
        locked = base.choose_threshold(
            scores, stage142["ious"][development], new_ious[development]
        )
        reports.append({
            "group": name,
            "feature_count": len(groups[name]),
            "oof": locked,
        })
        print(json.dumps(reports[-1], sort_keys=True))
    selected = max(reports, key=lambda item: (
        item["oof"]["selected"]["hits050"],
        item["oof"]["selected"]["hits025"],
        item["oof"]["selected"]["mean_iou"],
        -item["oof"]["changed_ratio"],
    ))
    selected_names = groups[selected["group"]]
    selected_indices = group_indices[selected["group"]]
    final_candidate = fit_classifier(
        matrix[development][:, selected_indices],
        stage142["ious"][development], new_ious[development],
    )
    model_items = base.save_candidate(final_candidate, args.output_dir)
    reloaded = base.load_candidate({
        "selected_candidate": "fix_vs_break_classifier", "models": model_items,
    })
    test_scores = base.booster_predict(
        reloaded, matrix[test][:, selected_indices]
    )
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
    lock = {
        "stage": STAGE,
        "status": "complete_train_only_invariant_feature_oof_lock",
        "protocol": (
            "post_stage159_absolute_geometry_removed_fixed_classifier_"
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
        "feature_names": all_names,
        "feature_count": len(all_names),
        "source_feature_names": source_feature_names,
        "selected_feature_group": selected["group"],
        "selected_feature_names": selected_names,
        "selected_feature_indices": selected_indices.tolist(),
        "selected_feature_count": len(selected_names),
        "all_feature_group_counts": {
            name: len(names) for name, names in groups.items()
        },
        "selected_candidate": "fix_vs_break_classifier",
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
    }
    lock_path = os.path.join(
        args.output_dir, "locked_invariant_feature_selector.json"
    )
    base.atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": base.sha256(lock_path),
        "selected_feature_group": selected["group"],
        "selected_feature_count": len(selected_names),
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
    raw_rows = base.option_ranker.load_rows(args.stage142_dump)
    stage142 = base.build_stage142_arrays(
        raw_rows, args.stage142_lock, args.stage31_lock, args.stage33_lock
    )
    matrix, new_ious, all_names, source_feature_names = base.build_matrix(
        args.stage150_source_dump, args.stage150_compact_dump,
        raw_rows, stage142, locked_feature_names=lock["source_feature_names"],
    )
    assert all_names == lock["feature_names"]
    assert source_feature_names == lock["source_feature_names"]
    indices = np.asarray(lock["selected_feature_indices"], dtype=np.int64)
    assert [all_names[index] for index in indices] == lock["selected_feature_names"]
    candidate = base.load_candidate(lock)
    scores = base.booster_predict(candidate, matrix[:, indices])
    metrics = base.evaluate_scores(
        scores, stage142["ious"], new_ious,
        lock["selected_oof"]["threshold"],
    )
    count = len(raw_rows)
    strict025 = math.floor(0.5391 * count) + 1
    strict050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "160_invariant_feature_selector_validation_eval",
        "status": "complete",
        "post_stage159_engineering_refinement": True,
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": base.sha256(args.policy_lock),
        "validation_labels_used_for_selection": False,
        "selected_feature_group": lock["selected_feature_group"],
        "selected_feature_count": lock["selected_feature_count"],
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
