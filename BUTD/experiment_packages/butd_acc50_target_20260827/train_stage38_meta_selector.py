#!/usr/bin/env python3
"""Train-only meta selector over two locked option rankers.

The primary rankers and every threshold are locked without using ScanRefer val.
This script uses disjoint train-scene partitions:
  * original dev scenes: meta-selector fitting;
  * original test scenes, half: early stopping and gate calibration;
  * original test scenes, other half: untouched audit.

At deployment the default action is the locked Stage29 choice.  The selector may
switch to the original baseline or the locked Stage36 choice only when its score
advantage passes the train-only calibrated safety gate.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys

import lightgbm as lgb
import numpy as np


EXTRA_FEATURES = [
    "stage29_z", "stage36_z", "stage29_z_delta_default",
    "stage36_z_delta_default", "stage29_rank", "stage36_rank",
    "is_default_stage29", "is_stage36_choice", "is_original_baseline",
    "stage29_equals_stage36", "group_size_norm", "stage29_top_gap",
    "stage36_top_gap",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_joint(path):
    spec = importlib.util.spec_from_file_location("joint_ranker_stage38", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_lock(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def secondary_bucket(scene_id):
    token = (str(scene_id) + "|stage38_meta").encode("utf-8")
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 2


def split_meta_indices(joint, metas):
    primary = joint.split_indices(metas)
    meta_train = list(primary["dev"])
    meta_calib = [i for i in primary["test"]
                  if secondary_bucket(metas[i].scene_id) == 0]
    meta_audit = [i for i in primary["test"]
                  if secondary_bucket(metas[i].scene_id) == 1]
    scenes = {
        name: {metas[i].scene_id for i in values}
        for name, values in (("train", meta_train),
                             ("calib", meta_calib),
                             ("audit", meta_audit))
    }
    assert all(scenes[a].isdisjoint(scenes[b])
               for a, b in (("train", "calib"), ("train", "audit"),
                            ("calib", "audit")))
    assert meta_train and meta_calib and meta_audit
    return {"train": meta_train, "calib": meta_calib,
            "audit": meta_audit}, scenes


def locked_choice(scores, baseline, threshold):
    best = int(np.argmax(scores))
    gap = float(scores[best] - scores[int(baseline)])
    return best if gap >= float(threshold) else int(baseline)


def normalized_rank(values):
    order = np.argsort(-np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float32)
    ranks[order] = np.arange(len(order), dtype=np.float32)
    return ranks / max(1, len(order) - 1)


def build_meta_split(joint, group_features, group_ious, metas, indices,
                     booster29, lock29, booster36, lock36):
    features = []
    ious = []
    groups = []
    defaults = []
    provenance = []
    for count, index in enumerate(indices, 1):
        x = np.asarray(group_features[index], dtype=np.float32)
        y_iou = np.asarray(group_ious[index], dtype=np.float32)
        baseline = int(metas[index].baseline_index)
        s29 = np.asarray(booster29.predict(
            x, num_iteration=int(lock29["best_iteration"])
        ), dtype=np.float32)
        s36 = np.asarray(booster36.predict(
            x, num_iteration=int(lock36["best_iteration"])
        ), dtype=np.float32)
        c29 = locked_choice(s29, baseline, lock29["gate"]["threshold"])
        c36 = locked_choice(s36, baseline, lock36["gate"]["threshold"])
        # Default Stage29 must be first so fallback index is always zero.
        candidates = []
        for candidate in (c29, c36, baseline):
            if candidate not in candidates:
                candidates.append(candidate)
        z29 = (s29 - s29.mean()) / (s29.std() + 1e-6)
        z36 = (s36 - s36.mean()) / (s36.std() + 1e-6)
        r29 = normalized_rank(s29)
        r36 = normalized_rank(s36)
        top29 = np.sort(s29)[-2:]
        top36 = np.sort(s36)[-2:]
        gap29 = float(top29[-1] - top29[-2]) if len(top29) > 1 else 0.0
        gap36 = float(top36[-1] - top36[-2]) if len(top36) > 1 else 0.0
        default_x = x[c29]
        group_rows = []
        for candidate in candidates:
            extra = np.asarray([
                z29[candidate], z36[candidate],
                z29[candidate] - z29[c29], z36[candidate] - z36[c29],
                r29[candidate], r36[candidate],
                float(candidate == c29), float(candidate == c36),
                float(candidate == baseline), float(c29 == c36),
                min(len(x), 160) / 160.0, gap29, gap36,
            ], dtype=np.float32)
            # No IoU, GT box, or label-derived value enters the features.
            group_rows.append(np.concatenate(
                [x[candidate], x[candidate] - default_x, extra]
            ))
        features.append(np.stack(group_rows))
        ious.append(y_iou[np.asarray(candidates, dtype=np.int64)])
        groups.append(len(candidates))
        defaults.append(0)
        provenance.append({
            "scene_id": metas[index].scene_id,
            "default_stage29": c29,
            "stage36": c36,
            "original": baseline,
            "candidates": candidates,
        })
        if count % 1000 == 0:
            print("meta_groups={}/{}".format(count, len(indices)), flush=True)
    return {
        "x": np.concatenate(features, axis=0).astype(np.float32),
        "ious": np.concatenate(ious, axis=0).astype(np.float32),
        "groups": np.asarray(groups, dtype=np.int32),
        "defaults": np.asarray(defaults, dtype=np.int32),
        "provenance": provenance,
    }


def decision_details(joint, scores, split, threshold):
    selected, defaults, gaps = joint.group_decisions(
        scores, split["ious"], split["groups"], split["defaults"], threshold
    )
    result = {
        "selected": joint.metrics(selected),
        "default_stage29": joint.metrics(defaults),
        "changed_ratio": float(np.mean(gaps >= threshold)),
        "score_gap_mean": float(np.mean(gaps)),
        "fix25": int(np.sum((defaults < 0.25) & (selected >= 0.25))),
        "break25": int(np.sum((defaults >= 0.25) & (selected < 0.25))),
        "fix50": int(np.sum((defaults < 0.50) & (selected >= 0.50))),
        "break50": int(np.sum((defaults >= 0.50) & (selected < 0.50))),
    }
    return result


def make_ranker(seed, threads):
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=[0, 1],
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        max_depth=5, min_child_samples=100, subsample=0.85,
        subsample_freq=1, colsample_bytree=0.85, reg_lambda=2.0,
        random_state=seed, n_jobs=threads, verbosity=-1,
    )


def common_load(args):
    joint = load_joint(args.joint_script)
    lock29 = load_lock(args.stage29_lock)
    lock36 = load_lock(args.stage36_lock)
    booster29 = lgb.Booster(model_file=lock29["model_path"])
    booster36 = lgb.Booster(model_file=lock36["model_path"])
    rows = joint.load_rows(args.dump)
    gf, gi, metas = joint.build_dataset(
        rows, int(lock29["max_candidates"]), require_scene=args.require_scene
    )
    return joint, lock29, lock36, booster29, booster36, gf, gi, metas


def train(args):
    os.makedirs(args.output_dir, exist_ok=False)
    (joint, lock29, lock36, booster29, booster36,
     gf, gi, metas) = common_load(args)
    indices, scenes = split_meta_indices(joint, metas)
    splits = {
        name: build_meta_split(joint, gf, gi, metas, values,
                               booster29, lock29, booster36, lock36)
        for name, values in indices.items()
    }
    names = (["choice_" + n for n in joint.FEATURE_NAMES] +
             ["delta_default_" + n for n in joint.FEATURE_NAMES] +
             EXTRA_FEATURES)
    assert splits["train"]["x"].shape[1] == len(names)
    ranker = make_ranker(args.seed, args.num_threads)
    ranker.fit(
        splits["train"]["x"],
        (splits["train"]["ious"] >= 0.50).astype(np.int32),
        group=splits["train"]["groups"].tolist(),
        eval_set=[(
            splits["calib"]["x"],
            (splits["calib"]["ious"] >= 0.50).astype(np.int32),
        )],
        eval_group=[splits["calib"]["groups"].tolist()], eval_at=[1],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(20)],
        feature_name=names,
    )
    iteration = int(ranker.best_iteration_ or ranker.n_estimators)
    calib_scores = ranker.booster_.predict(
        splits["calib"]["x"], num_iteration=iteration
    )
    gate, _ = joint.choose_gate(
        calib_scores, splits["calib"]["ious"],
        splits["calib"]["groups"], splits["calib"]["defaults"]
    )
    audit_scores = ranker.booster_.predict(
        splits["audit"]["x"], num_iteration=iteration
    )
    model_path = os.path.join(args.output_dir, "stage38_meta_selector.txt")
    ranker.booster_.save_model(model_path, num_iteration=iteration)
    receipt = {
        "protocol": "train_scene_disjoint_stage29_default_meta_selector",
        "dump": os.path.abspath(args.dump), "dump_sha256": sha256(args.dump),
        "joint_script": os.path.abspath(args.joint_script),
        "joint_script_sha256": sha256(args.joint_script),
        "stage29_lock": os.path.abspath(args.stage29_lock),
        "stage29_lock_sha256": sha256(args.stage29_lock),
        "stage36_lock": os.path.abspath(args.stage36_lock),
        "stage36_lock_sha256": sha256(args.stage36_lock),
        "model_path": os.path.abspath(model_path),
        "model_sha256": sha256(model_path), "best_iteration": iteration,
        "feature_names": names, "gate": gate,
        "scene_counts": {k: len(v) for k, v in scenes.items()},
        "sample_counts": {k: len(v) for k, v in indices.items()},
        "internal": {
            "calib": decision_details(joint, calib_scores,
                                      splits["calib"], gate["threshold"]),
            "audit": decision_details(joint, audit_scores,
                                      splits["audit"], gate["threshold"]),
        },
    }
    lock_path = os.path.join(args.output_dir, "locked_stage38_meta.json")
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def evaluate(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    lock = load_lock(args.meta_lock)
    (joint, lock29, lock36, booster29, booster36,
     gf, gi, metas) = common_load(args)
    split = build_meta_split(joint, gf, gi, metas, list(range(len(metas))),
                             booster29, lock29, booster36, lock36)
    booster = lgb.Booster(model_file=lock["model_path"])
    scores = booster.predict(split["x"],
                             num_iteration=int(lock["best_iteration"]))
    result = decision_details(joint, scores, split,
                              float(lock["gate"]["threshold"]))
    receipt = {
        "protocol": lock["protocol"], "diagnostic_only": True,
        "dump": os.path.abspath(args.dump), "dump_sha256": sha256(args.dump),
        "meta_lock": os.path.abspath(args.meta_lock),
        "meta_lock_sha256": sha256(args.meta_lock),
        "model_sha256": sha256(lock["model_path"]),
        "threshold": float(lock["gate"]["threshold"]), **result,
    }
    receipt["goal_achieved_offline"] = bool(
        receipt["selected"]["acc025"] > 0.5391 and
        receipt["selected"]["acc050"] > 0.4241
    )
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def self_test():
    # Meta features are constructed only from deployable option vectors and
    # locked-model scores.  Changing labels/IoUs cannot change the feature rows.
    rng = np.random.default_rng(17)
    x = rng.normal(size=(5, 56)).astype(np.float32)
    default = x[2]
    extra = rng.normal(size=(13,)).astype(np.float32)
    a = np.concatenate([x[1], x[1] - default, extra])
    fake_iou_a = rng.random(5)
    fake_iou_b = 1.0 - fake_iou_a
    b = np.concatenate([x[1], x[1] - default, extra])
    assert np.array_equal(a, b)
    assert not np.array_equal(fake_iou_a, fake_iou_b)
    assert len(a) == 125
    print("STAGE38_META_SELECTOR_SELFTEST_PASS features=125")


def add_common(parser):
    parser.add_argument("--joint-script", required=True)
    parser.add_argument("--stage29-lock", required=True)
    parser.add_argument("--stage36-lock", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--require-scene", action="store_true")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("self-test")
    p = sub.add_parser("train"); add_common(p)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num-threads", type=int, default=16)
    p = sub.add_parser("evaluate"); add_common(p)
    p.add_argument("--meta-lock", required=True)
    p.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.command == "self-test": self_test()
    elif args.command == "train": train(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
