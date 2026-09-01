#!/usr/bin/env python3
"""Train a class-balanced fix-vs-break gate over Stage143 action pairs."""

import argparse
import hashlib
import json
import math
import os

import lightgbm as lgb
import numpy as np

import stage140_train_eval_nested_blend as blend
import stage143_same_checkpoint_complement_gate as complement
from train_joint_option_ranker import (
    ACTIONS,
    FEATURE_NAMES,
    MATCH_POWERS,
    build_dataset,
    load_rows,
    materialize,
    split_indices,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def build_pair_dataset(action_x, action_ious, action_groups, safe_offsets):
    pair_x = []
    alternative_ious = []
    pair_safe_ious = []
    pair_groups = []
    safe_group_ious = []
    cursor = 0
    for size_value, safe_offset_value in zip(action_groups, safe_offsets):
        size = int(size_value)
        safe_offset = int(safe_offset_value)
        group_x = action_x[cursor:cursor + size]
        group_ious = action_ious[cursor:cursor + size]
        safe_iou = float(group_ious[safe_offset])
        alternatives = 0
        for option_index in range(size):
            if option_index == safe_offset:
                continue
            pair_x.append(group_x[option_index])
            alternative_ious.append(float(group_ious[option_index]))
            pair_safe_ious.append(safe_iou)
            alternatives += 1
        pair_groups.append(alternatives)
        safe_group_ious.append(safe_iou)
        cursor += size
    assert cursor == len(action_x)
    return (
        np.asarray(pair_x, dtype=np.float32),
        np.asarray(alternative_ious, dtype=np.float32),
        np.asarray(pair_safe_ious, dtype=np.float32),
        np.asarray(pair_groups, dtype=np.int32),
        np.asarray(safe_group_ious, dtype=np.float32),
    )


def pair_labels(alternative_ious, safe_ious):
    fix050 = (safe_ious < 0.50) & (alternative_ious >= 0.50)
    break050 = (safe_ious >= 0.50) & (alternative_ious < 0.50)
    break025 = (safe_ious >= 0.25) & (alternative_ious < 0.25)
    utility = (
        4.0 * (
            (alternative_ious >= 0.50).astype(np.float32)
            - (safe_ious >= 0.50).astype(np.float32)
        )
        + 1.0 * (
            (alternative_ious >= 0.25).astype(np.float32)
            - (safe_ious >= 0.25).astype(np.float32)
        )
        + 0.1 * (alternative_ious - safe_ious)
    )
    beneficial = utility > 0.0
    return {
        "fix050": fix050,
        "break050": break050,
        "break025": break025,
        "beneficial": beneficial,
        "utility": utility.astype(np.float32),
    }


def label_summary(labels):
    return {
        "pairs": int(len(labels["utility"])),
        "fix050": int(labels["fix050"].sum()),
        "break050": int(labels["break050"].sum()),
        "break025": int(labels["break025"].sum()),
        "beneficial": int(labels["beneficial"].sum()),
        "harmful": int((labels["utility"] < 0.0).sum()),
        "neutral": int((labels["utility"] == 0.0).sum()),
    }


def pair_decisions(
    scores,
    alternative_ious,
    pair_groups,
    safe_group_ious,
    threshold,
):
    selected = []
    changed = []
    best_scores = []
    cursor = 0
    for alternatives_value, safe_iou in zip(pair_groups, safe_group_ious):
        alternatives = int(alternatives_value)
        if alternatives == 0:
            selected.append(float(safe_iou))
            changed.append(False)
            best_scores.append(float("-inf"))
            continue
        group_scores = scores[cursor:cursor + alternatives]
        group_ious = alternative_ious[cursor:cursor + alternatives]
        best = int(np.argmax(group_scores))
        best_score = float(group_scores[best])
        use_alternative = best_score >= float(threshold)
        selected.append(
            float(group_ious[best]) if use_alternative else float(safe_iou)
        )
        changed.append(use_alternative)
        best_scores.append(best_score)
        cursor += alternatives
    assert cursor == len(scores)
    return (
        np.asarray(selected, dtype=np.float32),
        np.asarray(changed, dtype=bool),
        np.asarray(best_scores, dtype=np.float32),
    )


def pair_oracle(alternative_ious, pair_groups, safe_group_ious):
    selected = []
    cursor = 0
    for alternatives_value, safe_iou in zip(pair_groups, safe_group_ious):
        alternatives = int(alternatives_value)
        best = float(safe_iou)
        if alternatives:
            best = max(
                best,
                float(np.max(alternative_ious[cursor:cursor + alternatives])),
            )
            cursor += alternatives
        selected.append(best)
    assert cursor == len(alternative_ious)
    return complement.summarize(np.asarray(selected, dtype=np.float32))


def evaluate_pair_scores(
    scores,
    alternative_ious,
    pair_groups,
    safe_group_ious,
    threshold,
):
    selected, changed, best_scores = pair_decisions(
        scores,
        alternative_ious,
        pair_groups,
        safe_group_ious,
        threshold,
    )
    return {
        "selected": complement.summarize(selected),
        "safe": complement.summarize(safe_group_ious),
        "changed_ratio": float(changed.mean()),
        "best_score_mean_finite": float(
            best_scores[np.isfinite(best_scores)].mean()
        ),
        "fix_break": complement.fix_break(
            safe_group_ious, selected, changed
        ),
        "pair_oracle": pair_oracle(
            alternative_ious, pair_groups, safe_group_ious
        ),
    }


def choose_threshold(
    scores, alternative_ious, pair_groups, safe_group_ious
):
    _, _, best_scores = pair_decisions(
        scores,
        alternative_ious,
        pair_groups,
        safe_group_ious,
        float("inf"),
    )
    finite = best_scores[np.isfinite(best_scores)]
    thresholds = list(
        np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 401)))
    )
    thresholds.extend([float("-inf"), float("inf")])
    safe025 = float(np.mean(safe_group_ious >= 0.25))
    rows = []
    for threshold in thresholds:
        result = evaluate_pair_scores(
            scores,
            alternative_ious,
            pair_groups,
            safe_group_ious,
            float(threshold),
        )
        result["threshold"] = float(threshold)
        result["preserves_acc025"] = bool(
            result["selected"]["acc025"] >= safe025 - 0.001
        )
        rows.append(result)
    feasible = [row for row in rows if row["preserves_acc025"]]
    return max(feasible, key=lambda row: (
        row["selected"]["acc050"],
        row["selected"]["acc025"],
        row["selected"]["mean_iou"],
        -row["changed_ratio"],
    ))


def prepare_split(arrays, boosters, provenance, safe_config):
    action_arrays = complement.prepare_split(
        arrays, boosters, provenance, safe_config
    )
    return build_pair_dataset(*action_arrays[:4])


def fit_candidate(spec, train_x, train_labels, seed):
    weights = np.full(len(train_x), 0.25, dtype=np.float32)
    weights[train_labels["beneficial"]] = spec["positive_weight"]
    weights[train_labels["fix050"]] = spec["fix_weight"]
    weights[train_labels["break050"]] = spec["break050_weight"]
    weights[train_labels["break025"]] = spec["break025_weight"]
    common = dict(
        n_estimators=420,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=16,
        verbosity=-1,
    )
    if spec["kind"] == "fix_classifier":
        learner = lgb.LGBMClassifier(objective="binary", **common)
        target = train_labels["fix050"].astype(np.int32)
    elif spec["kind"] == "benefit_classifier":
        learner = lgb.LGBMClassifier(objective="binary", **common)
        target = train_labels["beneficial"].astype(np.int32)
    else:
        learner = lgb.LGBMRegressor(objective="huber", **common)
        target = train_labels["utility"]
    learner.fit(train_x, target, sample_weight=weights)
    return learner


def predict_candidate(learner, kind, features):
    if kind.endswith("classifier"):
        return learner.predict_proba(features)[:, 1]
    return learner.predict(features)


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    provenance = blend.verified_sources(args.stage31_lock, args.stage33_lock)
    stage142_lock = read_json(args.stage142_lock)
    safe_config = complement.validate_stage142_lock(
        stage142_lock, provenance
    )
    rows = load_rows(args.train_dump)
    group_features, group_ious, metas = build_dataset(
        rows, max_candidates=8, require_scene=True
    )
    splits = split_indices(metas)
    arrays = {
        split: materialize(group_features, group_ious, metas, splits[split])
        for split in ("train", "dev", "test")
    }
    boosters = blend.load_boosters(provenance)
    pair_arrays = {
        split: prepare_split(arrays[split], boosters, provenance, safe_config)
        for split in ("train", "dev", "test")
    }
    train_x, train_alt, train_safe_pairs, _, _ = pair_arrays["train"]
    train_labels = pair_labels(train_alt, train_safe_pairs)
    dev_x, dev_alt, _, dev_groups, dev_safe = pair_arrays["dev"]

    specs = [
        {
            "name": "fix050_classifier",
            "kind": "fix_classifier",
            "positive_weight": 4.0,
            "fix_weight": 16.0,
            "break050_weight": 32.0,
            "break025_weight": 64.0,
        },
        {
            "name": "benefit_classifier",
            "kind": "benefit_classifier",
            "positive_weight": 8.0,
            "fix_weight": 16.0,
            "break050_weight": 32.0,
            "break025_weight": 64.0,
        },
        {
            "name": "utility_regression",
            "kind": "utility_regression",
            "positive_weight": 8.0,
            "fix_weight": 16.0,
            "break050_weight": 32.0,
            "break025_weight": 64.0,
        },
    ]
    candidates = []
    for index, spec in enumerate(specs):
        learner = fit_candidate(spec, train_x, train_labels, index)
        dev_scores = predict_candidate(learner, spec["kind"], dev_x)
        dev = choose_threshold(
            dev_scores, dev_alt, dev_groups, dev_safe
        )
        candidates.append({"spec": spec, "dev": dev, "learner": learner})
        print(json.dumps({"spec": spec, "dev": dev}, sort_keys=True), flush=True)
    selected_index = max(range(len(candidates)), key=lambda index: (
        candidates[index]["dev"]["selected"]["acc050"],
        candidates[index]["dev"]["selected"]["acc025"],
        candidates[index]["dev"]["selected"]["mean_iou"],
        -candidates[index]["dev"]["changed_ratio"],
    ))
    selected = candidates[selected_index]
    model_path = os.path.join(args.output_dir, "fixbreak_gate.txt")
    selected["learner"].booster_.save_model(model_path, num_iteration=420)
    booster = lgb.Booster(model_file=model_path)

    test_x, test_alt, _, test_groups, test_safe = pair_arrays["test"]
    test_scores = booster.predict(test_x, num_iteration=420)
    internal_test = evaluate_pair_scores(
        test_scores,
        test_alt,
        test_groups,
        test_safe,
        float(selected["dev"]["threshold"]),
    )
    internal_gate_pass = bool(
        internal_test["selected"]["hits050"]
        > internal_test["safe"]["hits050"]
        and internal_test["selected"]["acc025"]
        >= internal_test["safe"]["acc025"] - 0.001
    )
    lock = {
        "stage": "144_same_checkpoint_fixbreak_gate",
        "status": "complete_train_only_lock",
        "protocol": (
            "scanrefer_train_scene70_fit_scene15_dev_lock_scene15_test_"
            "confirmed_class_balanced_fixbreak_pair_gate_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "train_dump": os.path.abspath(args.train_dump),
        "train_dump_sha256": sha256(args.train_dump),
        "stage31_lock_sha256": sha256(args.stage31_lock),
        "stage33_lock_sha256": sha256(args.stage33_lock),
        "stage142_lock_sha256": sha256(args.stage142_lock),
        "script": os.path.abspath(__file__),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "stage143_script": os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "stage143_same_checkpoint_complement_gate.py",
        ),
        "stage143_script_sha256": sha256(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "stage143_same_checkpoint_complement_gate.py",
        )),
        "trainer_script_sha256": sha256(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "train_joint_option_ranker.py",
        )),
        "stage140_script_sha256": sha256(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "stage140_train_eval_nested_blend.py",
        )),
        "provenance": provenance,
        "safe_config": safe_config,
        "feature_names": complement.COMPLEMENT_FEATURE_NAMES,
        "feature_count": len(complement.COMPLEMENT_FEATURE_NAMES),
        "model": os.path.abspath(model_path),
        "model_sha256": sha256(model_path),
        "model_iteration": 420,
        "selected_spec_index": int(selected_index),
        "selected_spec": selected["spec"],
        "selected_dev": selected["dev"],
        "all_dev_candidates": [
            {"spec": item["spec"], "dev": item["dev"]}
            for item in candidates
        ],
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "validation_evaluation_authorized": internal_gate_pass,
        "split_sizes": {
            name: int(len(indices)) for name, indices in splits.items()
        },
        "pair_counts": {
            name: int(len(pair_arrays[name][0])) for name in pair_arrays
        },
        "train_label_summary": label_summary(train_labels),
    }
    lock_path = os.path.join(args.output_dir, "locked_fixbreak_gate.json")
    atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": sha256(lock_path),
        "selected_spec_index": int(selected_index),
        "selected_spec": selected["spec"],
        "selected_dev": selected["dev"],
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "validation_evaluation_authorized": internal_gate_pass,
        "train_label_summary": label_summary(train_labels),
    }, indent=2, sort_keys=True), flush=True)


def evaluate(args):
    lock = read_json(args.policy_lock)
    assert lock["stage"] == "144_same_checkpoint_fixbreak_gate"
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert sha256(lock["script"]) == lock["script_sha256"]
    assert sha256(lock["stage143_script"]) == lock["stage143_script_sha256"]
    assert sha256(lock["model"]) == lock["model_sha256"]
    package_dir = os.path.dirname(lock["script"])
    assert sha256(os.path.join(package_dir, "train_joint_option_ranker.py")) == (
        lock["trainer_script_sha256"]
    )
    assert sha256(os.path.join(
        package_dir, "stage140_train_eval_nested_blend.py"
    )) == lock["stage140_script_sha256"]
    provenance = lock["provenance"]
    for source in provenance["sources"].values():
        assert sha256(source["path"]) == source["sha256"]

    rows = load_rows(args.dump)
    group_features, group_ious, metas = build_dataset(
        rows, max_candidates=8, require_scene=False
    )
    features = np.concatenate(group_features, axis=0)
    ious = np.concatenate(group_ious, axis=0)
    groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
    baselines = np.asarray(
        [meta.baseline_index for meta in metas], dtype=np.int32
    )
    boosters = blend.load_boosters(provenance)
    inner, pointwise = blend.component_scores(
        features, groups, boosters, provenance
    )
    action_arrays = complement.build_action_dataset(
        features,
        ious,
        groups,
        baselines,
        inner,
        pointwise,
        lock["safe_config"],
    )
    pair_x, alternative_ious, _, pair_groups, safe_group_ious = (
        build_pair_dataset(*action_arrays[:4])
    )
    booster = lgb.Booster(model_file=lock["model"])
    scores = booster.predict(
        pair_x, num_iteration=int(lock["model_iteration"])
    )
    result_metrics = evaluate_pair_scores(
        scores,
        alternative_ious,
        pair_groups,
        safe_group_ious,
        float(lock["selected_dev"]["threshold"]),
    )
    count = len(rows)
    strict_hits025 = math.floor(0.5391 * count) + 1
    strict_hits050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "144_same_checkpoint_fixbreak_gate_eval",
        "status": "complete",
        "diagnostic_only_until_integrated_and_independently_reloaded": True,
        "dump": os.path.abspath(args.dump),
        "dump_sha256": sha256(args.dump),
        "policy_lock": os.path.abspath(args.policy_lock),
        "policy_lock_sha256": sha256(args.policy_lock),
        "selection_data_scope": lock["selection_data_scope"],
        "validation_labels_used_for_selection": False,
        "metrics": result_metrics,
        "strict_goal_hits": {
            "acc025": strict_hits025,
            "acc050": strict_hits050,
        },
        "strict_goal_met_offline": bool(
            result_metrics["selected"]["hits025"] >= strict_hits025
            and result_metrics["selected"]["hits050"] >= strict_hits050
        ),
    }
    assert not os.path.exists(args.output_json), args.output_json
    atomic_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_dump")
    train_parser.add_argument("stage31_lock")
    train_parser.add_argument("stage33_lock")
    train_parser.add_argument("stage142_lock")
    train_parser.add_argument("output_dir")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("dump")
    eval_parser.add_argument("policy_lock")
    eval_parser.add_argument("output_json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
