#!/usr/bin/env python
"""Stage53: gate the best explicit rank-2 detector option over Stage29.

This keeps the Stage51 protocol but proposes the highest-scoring option among
all match_rank=1 options in every group, instead of proposing it only when a
rank-2 option happens to beat every rank-1 option in the source ranker.
"""

import importlib.util
import json
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(HERE, "train_top2_rescue_selector.py")
SPEC = importlib.util.spec_from_file_location("top2_rescue_base", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
sys.modules["top2_rescue_base"] = BASE
SPEC.loader.exec_module(BASE)


def score_groups(group_features, metas, booster, lock):
    all_x = np.concatenate(group_features, axis=0)
    all_scores = booster.predict(
        all_x, num_iteration=int(lock["best_iteration"])
    )
    threshold = float(lock["gate"]["threshold"])
    records = []
    cursor = 0
    for x, meta in zip(group_features, metas):
        size = int(meta.size)
        scores = np.asarray(all_scores[cursor:cursor + size], dtype=np.float32)
        baseline = int(meta.baseline_index)
        raw_best = int(np.argmax(scores))
        raw_gap = float(scores[raw_best] - scores[baseline])
        chosen = raw_best if raw_gap >= threshold else baseline
        order = np.argsort(-scores, kind="stable")
        margin = (
            float(scores[order[0]] - scores[order[1]])
            if size > 1 else float("inf")
        )
        mean = float(scores.mean())
        std = float(scores.std() + 1e-6)
        records.append({
            "baseline": baseline,
            "raw_best": raw_best,
            "chosen": int(chosen),
            "raw_gap": raw_gap,
            "margin": margin,
            "best_score": float(scores[raw_best]),
            "baseline_score": float(scores[baseline]),
            "chosen_score": float(scores[chosen]),
            "best_z": float((scores[raw_best] - mean) / std),
            "baseline_z": float((scores[baseline] - mean) / std),
            "size": size,
            "scores": scores,
            "score_mean": mean,
            "score_std": std,
        })
        cursor += size
    assert cursor == len(all_scores)
    return records


def candidate_record(record, candidate, candidate_positions):
    result = dict(record)
    scores = record["scores"]
    candidate_scores = scores[candidate_positions]
    order = np.argsort(-candidate_scores, kind="stable")
    margin = (
        float(candidate_scores[order[0]] - candidate_scores[order[1]])
        if len(candidate_positions) > 1 else float("inf")
    )
    result.update({
        "raw_best": int(candidate),
        "raw_gap": float(scores[candidate] - scores[record["baseline"]]),
        "margin": margin,
        "best_score": float(scores[candidate]),
        "best_z": float(
            (scores[candidate] - record["score_mean"]) / record["score_std"]
        ),
    })
    return result


def prepare(dump, top1_module, top2_module, top1_booster, top2_booster,
            top1_lock, top2_lock, require_scene):
    rows = top1_module.load_rows(dump)
    features1, ious1, metas1 = top1_module.build_dataset(
        rows, int(top1_lock["max_candidates"]), require_scene=require_scene
    )
    features2, ious2, metas2 = top2_module.build_dataset(
        rows, int(top2_lock["max_candidates"]), require_scene=require_scene
    )
    BASE.assert_aligned(metas1, metas2, ious1, ious2)
    records1 = score_groups(features1, metas1, top1_booster, top1_lock)
    records2 = score_groups(features2, metas2, top2_booster, top2_lock)
    rank_index = top2_module.FEATURE_NAMES.index("match_rank")
    selector_x = []
    eligible = np.zeros(len(metas1), dtype=bool)
    candidate_ious = np.zeros(len(metas1), dtype=np.float32)
    incumbent_ious = np.zeros(len(metas1), dtype=np.float32)
    candidate_rows = []
    for index, (x1, x2, y1, y2, r1, r2) in enumerate(zip(
            features1, features2, ious1, ious2, records1, records2)):
        incumbent_ious[index] = float(y1[r1["chosen"]])
        rank2_positions = np.flatnonzero(x2[:, rank_index] > 0.5)
        if len(rank2_positions) == 0:
            candidate_ious[index] = incumbent_ious[index]
            continue
        candidate = int(rank2_positions[
            np.argmax(r2["scores"][rank2_positions])
        ])
        rescue = candidate_record(r2, candidate, rank2_positions)
        candidate_ious[index] = float(y2[candidate])
        eligible[index] = True
        candidate_rows.append(index)
        selector_x.append(BASE.selector_feature(x1, x2, r1, rescue))
    if selector_x:
        selector_x = np.stack(selector_x).astype(np.float32)
    else:
        width = len(top2_module.FEATURE_NAMES) + 2 * len(
            top1_module.FEATURE_NAMES
        ) + 17
        selector_x = np.zeros((0, width), dtype=np.float32)
    return {
        "x": selector_x,
        "candidate_rows": np.asarray(candidate_rows, dtype=np.int64),
        "eligible": eligible,
        "candidate_ious": candidate_ious,
        "incumbent_ious": incumbent_ious,
        "metas": metas1,
    }


ORIGINAL_TRAIN = BASE.train
ORIGINAL_SELF_TEST = BASE.self_test


def train(args):
    ORIGINAL_TRAIN(args)
    lock_path = os.path.join(
        args.output_dir, "locked_top2_rescue_policy.json"
    )
    with open(lock_path, "r", encoding="utf-8") as handle:
        lock = json.load(handle)
    lock["protocol"] = (
        "scene_hash_train_only_internal_test_explicit_rank2_rescue_v2"
    )
    lock["candidate_selection"] = "best_scored_match_rank_1_option"
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
    print("STAGE53_LOCK_PROTOCOL_UPDATED", flush=True)


def self_test():
    ORIGINAL_SELF_TEST()
    scores = np.asarray([0.2, 0.1, 0.8], dtype=np.float32)
    positions = np.asarray([0, 2], dtype=np.int64)
    assert int(positions[np.argmax(scores[positions])]) == 2
    print("TOP2_EXPLICIT_CANDIDATE_RESCUE_SELFTEST_PASS")


BASE.prepare = prepare
BASE.train = train
BASE.self_test = self_test


if __name__ == "__main__":
    BASE.main()
