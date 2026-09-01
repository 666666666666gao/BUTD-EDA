#!/usr/bin/env python3
"""Train a scene-split gate over complementary Stage31/Stage33 actions.

The model is fitted only on ScanRefer training scenes.  A deterministic
scene-disjoint dev split selects the gate threshold, an internal scene test
split authorizes one validation evaluation, and ScanRefer validation labels
are never used for model or policy selection.
"""

import argparse
import hashlib
import json
import math
import os

import lightgbm as lgb
import numpy as np

import stage140_train_eval_nested_blend as blend
from train_joint_option_ranker import (
    ACTIONS,
    FEATURE_NAMES,
    MATCH_POWERS,
    build_dataset,
    load_rows,
    materialize,
    metrics,
    relevance,
    split_indices,
)


SOURCE_NAMES = ("baseline", "safe", "stage31", "symmetric", "stage33")


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


def summarize(ious):
    result = metrics(ious)
    result.update({
        "hits025": int((ious >= 0.25).sum()),
        "hits050": int((ious >= 0.50).sum()),
    })
    return result


def fix_break(before, after, changed):
    result = {"changed": int(changed.sum())}
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        old = before >= threshold
        new = after >= threshold
        fixes = int((~old & new).sum())
        breaks = int((old & ~new).sum())
        result["fix_" + suffix] = fixes
        result["break_" + suffix] = breaks
        result["net_" + suffix] = fixes - breaks
    return result


def softmax_entropy(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    shifted = values - np.max(values)
    probabilities = np.exp(np.clip(shifted, -50.0, 50.0))
    probabilities /= np.maximum(probabilities.sum(), 1e-8)
    return float(-np.sum(probabilities * np.log(probabilities + 1e-8)))


def top_margin(values):
    ordered = np.sort(np.asarray(values, dtype=np.float32))[::-1]
    return float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0


def rank_fractions(values):
    values = np.asarray(values, dtype=np.float32)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(len(values) - 1, 1)


def correlation(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if len(left) < 2 or left.std() < 1e-6 or right.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def complement_feature_names():
    names = []
    names += ["candidate_" + name for name in FEATURE_NAMES]
    names += ["safe_" + name for name in FEATURE_NAMES]
    names += ["candidate_minus_safe_" + name for name in FEATURE_NAMES]
    score_names = ("stage31", "stage33", "symmetric", "safe_blend")
    names += ["candidate_score_" + name for name in score_names]
    names += ["safe_score_" + name for name in score_names]
    names += ["candidate_minus_safe_score_" + name for name in score_names]
    names += ["candidate_rank_" + name for name in score_names]
    names += ["safe_rank_" + name for name in score_names]
    names += ["candidate_minus_safe_rank_" + name for name in score_names]
    names += [
        "group_size_norm",
        "stage31_stage33_correlation",
        "stage31_margin",
        "stage33_margin",
        "symmetric_margin",
        "safe_blend_margin",
        "stage31_entropy",
        "stage33_entropy",
        "symmetric_entropy",
        "safe_blend_entropy",
        "safe_blend_gap",
        "top_stage31_equals_stage33",
        "top_stage31_equals_symmetric",
        "top_stage33_equals_symmetric",
        "candidate_is_safe",
        "candidate_is_raw_baseline",
    ]
    names += ["source_" + name for name in SOURCE_NAMES]
    return names


COMPLEMENT_FEATURE_NAMES = complement_feature_names()


def validate_stage142_lock(lock, provenance):
    assert lock["protocol"] == (
        "scanrefer_train_only_scene_hash_dev_locked_nested_blend_v1"
    )
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["feature_names"] == FEATURE_NAMES
    assert tuple(lock["match_powers"]) == MATCH_POWERS
    assert tuple(lock["actions"]) == ACTIONS
    expected = provenance["sources"]
    actual = lock["provenance"]["sources"]
    for name in ("ordinal", "binary", "pointwise"):
        assert expected[name]["sha256"] == actual[name]["sha256"]
        assert int(expected[name]["iteration"]) == int(actual[name]["iteration"])
    selected = lock["selected"]
    inner_weight = float(selected["inner_weight"])
    pointwise_weight = float(selected["pointwise_weight"])
    assert abs(inner_weight + pointwise_weight - 1.0) < 1e-8
    return {
        "inner_weight": inner_weight,
        "pointwise_weight": pointwise_weight,
        "threshold": float(selected["gate"]["threshold"]),
    }


def build_action_dataset(
    features, ious, groups, baselines, inner, pointwise, safe_config
):
    action_features = []
    action_ious = []
    action_groups = []
    safe_offsets = []
    source_counts = {name: 0 for name in SOURCE_NAMES}
    cursor = 0
    for size_value, baseline_value in zip(groups, baselines):
        size = int(size_value)
        baseline_index = int(baseline_value)
        group_x = features[cursor:cursor + size]
        group_iou = ious[cursor:cursor + size]
        group_inner = inner[cursor:cursor + size]
        group_pointwise = pointwise[cursor:cursor + size]
        group_symmetric = 0.5 * group_inner + 0.5 * group_pointwise
        group_safe = (
            safe_config["inner_weight"] * group_inner
            + safe_config["pointwise_weight"] * group_pointwise
        )
        safe_best = int(np.argmax(group_safe))
        safe_gap = float(group_safe[safe_best] - group_safe[baseline_index])
        safe_index = (
            safe_best
            if safe_gap >= safe_config["threshold"]
            else baseline_index
        )
        proposed = {
            "baseline": baseline_index,
            "safe": safe_index,
            "stage31": int(np.argmax(group_inner)),
            "symmetric": int(np.argmax(group_symmetric)),
            "stage33": int(np.argmax(group_pointwise)),
        }
        candidates = []
        flags = {}
        for source_name in SOURCE_NAMES:
            option_index = proposed[source_name]
            source_counts[source_name] += 1
            if option_index not in flags:
                candidates.append(option_index)
                flags[option_index] = {name: 0.0 for name in SOURCE_NAMES}
            flags[option_index][source_name] = 1.0

        score_sets = (
            group_inner,
            group_pointwise,
            group_symmetric,
            group_safe,
        )
        rank_sets = tuple(rank_fractions(values) for values in score_sets)
        top_inner = int(np.argmax(group_inner))
        top_pointwise = int(np.argmax(group_pointwise))
        top_symmetric = int(np.argmax(group_symmetric))
        context = [
            float(size / 144.0),
            correlation(group_inner, group_pointwise),
            *[top_margin(values) for values in score_sets],
            *[softmax_entropy(values) for values in score_sets],
            safe_gap,
            float(top_inner == top_pointwise),
            float(top_inner == top_symmetric),
            float(top_pointwise == top_symmetric),
        ]
        safe_raw = group_x[safe_index]
        safe_scores = [float(values[safe_index]) for values in score_sets]
        safe_ranks = [float(values[safe_index]) for values in rank_sets]
        safe_action_offset = None
        for local_action_index, option_index in enumerate(candidates):
            candidate_raw = group_x[option_index]
            candidate_scores = [
                float(values[option_index]) for values in score_sets
            ]
            candidate_ranks = [
                float(values[option_index]) for values in rank_sets
            ]
            vector = []
            vector.extend(candidate_raw.tolist())
            vector.extend(safe_raw.tolist())
            vector.extend((candidate_raw - safe_raw).tolist())
            vector.extend(candidate_scores)
            vector.extend(safe_scores)
            vector.extend(
                [left - right for left, right in zip(candidate_scores, safe_scores)]
            )
            vector.extend(candidate_ranks)
            vector.extend(safe_ranks)
            vector.extend(
                [left - right for left, right in zip(candidate_ranks, safe_ranks)]
            )
            vector.extend(context)
            vector.extend([
                float(option_index == safe_index),
                float(option_index == baseline_index),
            ])
            vector.extend([flags[option_index][name] for name in SOURCE_NAMES])
            assert len(vector) == len(COMPLEMENT_FEATURE_NAMES), (
                len(vector), len(COMPLEMENT_FEATURE_NAMES)
            )
            if option_index == safe_index:
                safe_action_offset = local_action_index
            action_features.append(vector)
            action_ious.append(float(group_iou[option_index]))
        assert safe_action_offset is not None
        action_groups.append(len(candidates))
        safe_offsets.append(safe_action_offset)
        cursor += size
    assert cursor == len(features)
    return (
        np.asarray(action_features, dtype=np.float32),
        np.asarray(action_ious, dtype=np.float32),
        np.asarray(action_groups, dtype=np.int32),
        np.asarray(safe_offsets, dtype=np.int32),
        source_counts,
    )


def action_oracle(action_ious, groups):
    selected = []
    cursor = 0
    for size_value in groups:
        size = int(size_value)
        selected.append(float(np.max(action_ious[cursor:cursor + size])))
        cursor += size
    assert cursor == len(action_ious)
    return summarize(np.asarray(selected, dtype=np.float32))


def action_decisions(scores, action_ious, groups, safe_offsets, threshold):
    selected = []
    safe = []
    changed = []
    gaps = []
    cursor = 0
    for size_value, safe_offset_value in zip(groups, safe_offsets):
        size = int(size_value)
        safe_offset = int(safe_offset_value)
        group_scores = scores[cursor:cursor + size]
        group_ious = action_ious[cursor:cursor + size]
        best = int(np.argmax(group_scores))
        gap = float(group_scores[best] - group_scores[safe_offset])
        choose = best if gap >= float(threshold) else safe_offset
        selected.append(float(group_ious[choose]))
        safe.append(float(group_ious[safe_offset]))
        changed.append(choose != safe_offset)
        gaps.append(gap)
        cursor += size
    assert cursor == len(scores)
    return (
        np.asarray(selected, dtype=np.float32),
        np.asarray(safe, dtype=np.float32),
        np.asarray(changed, dtype=bool),
        np.asarray(gaps, dtype=np.float32),
    )


def evaluate_action_scores(scores, action_ious, groups, safe_offsets, threshold):
    selected, safe, changed, gaps = action_decisions(
        scores, action_ious, groups, safe_offsets, threshold
    )
    return {
        "selected": summarize(selected),
        "safe": summarize(safe),
        "changed_ratio": float(changed.mean()),
        "score_gap_mean": float(gaps.mean()),
        "fix_break": fix_break(safe, selected, changed),
        "action_oracle": action_oracle(action_ious, groups),
    }


def choose_threshold(scores, action_ious, groups, safe_offsets):
    _, safe, _, gaps = action_decisions(
        scores, action_ious, groups, safe_offsets, float("inf")
    )
    safe_metrics = metrics(safe)
    finite = gaps[np.isfinite(gaps)]
    thresholds = list(
        np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 301)))
    )
    thresholds.extend([float("-inf"), float("inf")])
    rows = []
    for threshold in thresholds:
        result = evaluate_action_scores(
            scores, action_ious, groups, safe_offsets, float(threshold)
        )
        result["threshold"] = float(threshold)
        result["preserves_acc025"] = bool(
            result["selected"]["acc025"] >= safe_metrics["acc025"] - 0.001
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
    features, _, ious, groups, baselines = arrays
    inner, pointwise = blend.component_scores(
        features, groups, boosters, provenance
    )
    return build_action_dataset(
        features,
        ious,
        groups,
        baselines,
        inner,
        pointwise,
        safe_config,
    )


def train(args):
    assert not os.path.exists(args.output_dir), args.output_dir
    os.makedirs(args.output_dir)
    provenance = blend.verified_sources(args.stage31_lock, args.stage33_lock)
    stage142_lock = read_json(args.stage142_lock)
    safe_config = validate_stage142_lock(stage142_lock, provenance)

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
    action_arrays = {
        split: prepare_split(arrays[split], boosters, provenance, safe_config)
        for split in ("train", "dev", "test")
    }
    train_x, train_ious, train_groups, _, _ = action_arrays["train"]
    dev_x, dev_ious, dev_groups, dev_safe_offsets, _ = action_arrays["dev"]
    train_y = relevance(train_ious, mode="ordinal")
    dev_y = relevance(dev_ious, mode="ordinal")

    learner = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 4, 24, 36],
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=120,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        random_state=0,
        n_jobs=args.num_threads,
        verbosity=-1,
    )
    learner.fit(
        train_x,
        train_y,
        group=train_groups.tolist(),
        eval_set=[(dev_x, dev_y)],
        eval_group=[dev_groups.tolist()],
        eval_at=[1],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    best_iteration = int(learner.best_iteration_ or 900)
    dev_scores = learner.predict(dev_x, num_iteration=best_iteration)
    selected_dev = choose_threshold(
        dev_scores, dev_ious, dev_groups, dev_safe_offsets
    )

    model_path = os.path.join(args.output_dir, "same_checkpoint_gate.txt")
    learner.booster_.save_model(model_path, num_iteration=best_iteration)
    booster = lgb.Booster(model_file=model_path)
    test_x, test_ious, test_groups, test_safe_offsets, _ = action_arrays["test"]
    test_scores = booster.predict(test_x, num_iteration=best_iteration)
    internal_test = evaluate_action_scores(
        test_scores,
        test_ious,
        test_groups,
        test_safe_offsets,
        float(selected_dev["threshold"]),
    )
    internal_gate_pass = bool(
        internal_test["selected"]["hits050"]
        > internal_test["safe"]["hits050"]
        and internal_test["selected"]["acc025"]
        >= internal_test["safe"]["acc025"] - 0.001
    )
    lock = {
        "stage": "143_same_checkpoint_complement_gate",
        "status": "complete_train_only_lock",
        "protocol": (
            "scanrefer_train_scene70_fit_scene15_dev_lock_scene15_test_confirmed_"
            "stage31_stage33_sample_gate_v1"
        ),
        "selection_data_scope": "scanrefer_train_scenes_only",
        "validation_labels_used_for_selection": False,
        "train_dump": os.path.abspath(args.train_dump),
        "train_dump_sha256": sha256(args.train_dump),
        "stage31_lock": os.path.abspath(args.stage31_lock),
        "stage31_lock_sha256": sha256(args.stage31_lock),
        "stage33_lock": os.path.abspath(args.stage33_lock),
        "stage33_lock_sha256": sha256(args.stage33_lock),
        "stage142_lock": os.path.abspath(args.stage142_lock),
        "stage142_lock_sha256": sha256(args.stage142_lock),
        "script": os.path.abspath(__file__),
        "script_sha256": sha256(os.path.abspath(__file__)),
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
        "candidate_sources": list(SOURCE_NAMES),
        "symmetric_weight": 0.5,
        "feature_names": COMPLEMENT_FEATURE_NAMES,
        "feature_count": len(COMPLEMENT_FEATURE_NAMES),
        "model": os.path.abspath(model_path),
        "model_sha256": sha256(model_path),
        "best_iteration": best_iteration,
        "model_config": {
            "objective": "lambdarank",
            "label_gain": [0, 1, 4, 24, 36],
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 120,
            "reg_lambda": 3.0,
            "seed": 0,
        },
        "threshold_selection": (
            "scene_dev_acc050_then_acc025_then_mean_iou_with_safe_acc025_"
            "minus_0p001_floor"
        ),
        "selected_dev": selected_dev,
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "validation_evaluation_authorized": internal_gate_pass,
        "split_sizes": {
            name: int(len(indices)) for name, indices in splits.items()
        },
        "split_action_counts": {
            name: int(action_arrays[name][0].shape[0])
            for name in action_arrays
        },
        "train_action_source_proposals": action_arrays["train"][4],
    }
    lock_path = os.path.join(args.output_dir, "locked_same_checkpoint_gate.json")
    atomic_json(lock_path, lock)
    print(json.dumps({
        "lock": os.path.abspath(lock_path),
        "lock_sha256": sha256(lock_path),
        "best_iteration": best_iteration,
        "selected_dev": selected_dev,
        "internal_scene_test": internal_test,
        "internal_gate_pass": internal_gate_pass,
        "validation_evaluation_authorized": internal_gate_pass,
    }, indent=2, sort_keys=True), flush=True)


def evaluate(args):
    lock = read_json(args.policy_lock)
    assert lock["stage"] == "143_same_checkpoint_complement_gate"
    assert lock["validation_labels_used_for_selection"] is False
    assert lock["validation_evaluation_authorized"] is True
    assert sha256(lock["script"]) == lock["script_sha256"]
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
    action_x, action_ious, action_groups, safe_offsets, _ = (
        build_action_dataset(
            features,
            ious,
            groups,
            baselines,
            inner,
            pointwise,
            lock["safe_config"],
        )
    )
    assert action_x.shape[1] == int(lock["feature_count"])
    booster = lgb.Booster(model_file=lock["model"])
    scores = booster.predict(
        action_x, num_iteration=int(lock["best_iteration"])
    )
    result_metrics = evaluate_action_scores(
        scores,
        action_ious,
        action_groups,
        safe_offsets,
        float(lock["selected_dev"]["threshold"]),
    )
    count = len(rows)
    strict_hits025 = math.floor(0.5391 * count) + 1
    strict_hits050 = math.floor(0.4241 * count) + 1
    result = {
        "stage": "143_same_checkpoint_complement_gate_eval",
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
    train_parser.add_argument("--num-threads", type=int, default=16)
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
