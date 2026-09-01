#!/usr/bin/env python3
"""Exact threshold/fix-break audit for a frozen Stage29 option ranker."""

import argparse
import hashlib
import json
import math
import os

import lightgbm as lgb
import numpy as np

from train_joint_option_ranker import build_dataset, load_rows, metrics


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decision_stats(baseline, alternative, changed):
    result = {"changed": int(changed.sum())}
    for threshold, name in ((0.25, "025"), (0.50, "050")):
        before = baseline >= threshold
        after = alternative >= threshold
        result["fix_" + name] = int((~before & after).sum())
        result["break_" + name] = int((before & ~after).sum())
        result["net_" + name] = (
            result["fix_" + name] - result["break_" + name]
        )
    return result


def summarize(ious):
    result = metrics(ious)
    result.update({
        "hits025": int((ious >= 0.25).sum()),
        "hits050": int((ious >= 0.50).sum()),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("model")
    parser.add_argument("lock_json")
    parser.add_argument("output_json")
    args = parser.parse_args()

    lock = json.load(open(args.lock_json, encoding="utf-8"))
    assert sha256(args.model) == lock["model_sha256"]
    rows = load_rows(args.dump)
    group_features, group_ious, metas = build_dataset(
        rows, max_candidates=int(lock["max_candidates"]), require_scene=False
    )
    features = np.concatenate(group_features, axis=0)
    ious = np.concatenate(group_ious, axis=0)
    groups = np.asarray([meta.size for meta in metas], dtype=np.int32)
    baselines = np.asarray(
        [meta.baseline_index for meta in metas], dtype=np.int32
    )
    booster = lgb.Booster(model_file=args.model)
    scores = booster.predict(
        features, num_iteration=int(lock["best_iteration"])
    )

    baseline_ious = []
    alternative_ious = []
    oracle_ious = []
    gaps = []
    cursor = 0
    for size, baseline in zip(groups, baselines):
        size = int(size)
        group_scores = scores[cursor:cursor + size]
        group_ious_values = ious[cursor:cursor + size]
        best = int(np.argmax(group_scores))
        baseline = int(baseline)
        baseline_ious.append(float(group_ious_values[baseline]))
        alternative_ious.append(float(group_ious_values[best]))
        oracle_ious.append(float(group_ious_values.max()))
        gaps.append(float(group_scores[best] - group_scores[baseline]))
        cursor += size
    assert cursor == len(scores)
    baseline_ious = np.asarray(baseline_ious, dtype=np.float32)
    alternative_ious = np.asarray(alternative_ious, dtype=np.float32)
    oracle_ious = np.asarray(oracle_ious, dtype=np.float32)
    gaps = np.asarray(gaps, dtype=np.float64)

    locked_threshold = float(lock["gate"]["threshold"])
    locked_changed = gaps >= locked_threshold
    locked_ious = np.where(
        locked_changed, alternative_ious, baseline_ious
    )

    # Exact sweep over every unique gap.  Start at +inf (all baseline), then
    # switch equal-gap groups atomically as the threshold decreases.
    order = np.argsort(-gaps, kind="mergesort")
    running = baseline_ious.copy()
    sweep = [{
        "threshold": "inf",
        "changed": 0,
        **summarize(running),
    }]
    position = 0
    while position < len(order):
        gap = gaps[order[position]]
        end = position + 1
        while end < len(order) and gaps[order[end]] == gap:
            end += 1
        indices = order[position:end]
        running[indices] = alternative_ious[indices]
        sweep.append({
            "threshold": float(gap),
            "changed": int(end),
            **summarize(running),
        })
        position = end

    strict_hits025 = math.floor(0.5391 * len(rows)) + 1
    strict_hits050 = math.floor(0.4241 * len(rows)) + 1
    feasible_min = [row for row in sweep if row["hits025"] >= strict_hits025]
    feasible_keep5440 = [
        row for row in sweep
        if row["hits025"] >= math.floor(0.5440 * len(rows)) + 1
    ]

    def best_row(candidates):
        return max(candidates, key=lambda row: (
            row["hits050"], row["hits025"], row["mean_iou"],
            -row["changed"],
        ))

    best_min = best_row(feasible_min)
    best_keep5440 = best_row(feasible_keep5440)
    top = sorted(feasible_min, key=lambda row: (
        row["hits050"], row["hits025"], row["mean_iou"],
    ), reverse=True)[:10]

    report = {
        "stage": "stage29_exact_threshold_audit",
        "dump": os.path.abspath(args.dump),
        "dump_sha256": sha256(args.dump),
        "model": os.path.abspath(args.model),
        "model_sha256": sha256(args.model),
        "lock": os.path.abspath(args.lock_json),
        "lock_sha256": sha256(args.lock_json),
        "count": len(rows),
        "strict_goal_hits": {
            "acc025": strict_hits025,
            "acc050": strict_hits050,
        },
        "baseline": summarize(baseline_ious),
        "locked": {
            "threshold": locked_threshold,
            **summarize(locked_ious),
            "fix_break": decision_stats(
                baseline_ious, locked_ious, locked_changed
            ),
            "strict_goal_met": bool(
                (locked_ious >= 0.25).sum() >= strict_hits025
                and (locked_ious >= 0.50).sum() >= strict_hits050
            ),
        },
        "best_threshold_acc025_gt_5391": best_min,
        "best_threshold_acc025_gt_5440": best_keep5440,
        "top10_thresholds": top,
        "threshold_family_can_meet_strict_goal": bool(
            best_min["hits050"] >= strict_hits050
        ),
        "oracle": {
            **summarize(oracle_ious),
            "fix_break": decision_stats(
                baseline_ious, oracle_ious,
                oracle_ious != baseline_ious,
            ),
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
