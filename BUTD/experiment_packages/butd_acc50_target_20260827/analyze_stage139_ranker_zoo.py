#!/usr/bin/env python3
"""Audit all frozen 56-feature option rankers on one immutable val dump.

The locked-policy rows are reproducible evaluations because every model hash,
feature schema, action set, and threshold is checked.  Pairwise blends and
their exact thresholds are explicitly capacity diagnostics only: they touch
the validation labels and must be re-selected on train-only OOF data before
they can become a formal policy.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import time

import lightgbm as lgb
import numpy as np

from train_joint_option_ranker import (
    ACTIONS,
    FEATURE_NAMES,
    MATCH_POWERS,
    build_dataset,
    load_rows,
    metrics,
    normalize_group_scores,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_sha256(scores):
    values = np.ascontiguousarray(scores, dtype=np.float32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def summarize(ious):
    result = metrics(ious)
    result.update({
        "hits025": int((ious >= 0.25).sum()),
        "hits050": int((ious >= 0.50).sum()),
    })
    return result


def decision_stats(baseline, chosen, changed):
    result = {"changed": int(changed.sum())}
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        before = baseline >= threshold
        after = chosen >= threshold
        result["fix_" + suffix] = int((~before & after).sum())
        result["break_" + suffix] = int((before & ~after).sum())
        result["net_" + suffix] = (
            result["fix_" + suffix] - result["break_" + suffix]
        )
    return result


def decisions(scores, ious, groups, baselines):
    baseline_ious = np.empty(len(groups), dtype=np.float32)
    alternative_ious = np.empty(len(groups), dtype=np.float32)
    gaps = np.empty(len(groups), dtype=np.float64)
    cursor = 0
    for index, (size, baseline) in enumerate(zip(groups, baselines)):
        size = int(size)
        baseline = int(baseline)
        group_scores = scores[cursor:cursor + size]
        group_ious = ious[cursor:cursor + size]
        best = int(np.argmax(group_scores))
        baseline_ious[index] = group_ious[baseline]
        alternative_ious[index] = group_ious[best]
        gaps[index] = float(group_scores[best] - group_scores[baseline])
        cursor += size
    assert cursor == len(scores)
    return baseline_ious, alternative_ious, gaps


def row_key(row):
    return (
        int(row["hits050"]),
        int(row["hits025"]),
        float(row["mean_iou"]),
        -int(row["changed"]),
    )


def exact_threshold_best(baseline, alternative, gaps, min_hits025):
    """Return the exact best threshold while preserving min_hits025."""
    order = np.argsort(-gaps, kind="mergesort")
    running_hits025 = int((baseline >= 0.25).sum())
    running_hits050 = int((baseline >= 0.50).sum())
    running_iou_sum = float(np.asarray(baseline, dtype=np.float64).sum())
    count = len(baseline)

    def make_row(threshold, changed):
        return {
            "threshold": threshold,
            "changed": int(changed),
            "hits025": int(running_hits025),
            "hits050": int(running_hits050),
            "acc025": float(running_hits025 / count),
            "acc050": float(running_hits050 / count),
            "mean_iou": float(running_iou_sum / count),
            "count": int(count),
        }

    best = make_row("inf", 0) if running_hits025 >= min_hits025 else None
    position = 0
    while position < len(order):
        gap = gaps[order[position]]
        end = position + 1
        while end < len(order) and gaps[order[end]] == gap:
            end += 1
        indices = order[position:end]
        before = baseline[indices]
        after = alternative[indices]
        running_hits025 += int((after >= 0.25).sum()) - int(
            (before >= 0.25).sum()
        )
        running_hits050 += int((after >= 0.50).sum()) - int(
            (before >= 0.50).sum()
        )
        running_iou_sum += float(
            np.asarray(after, dtype=np.float64).sum()
            - np.asarray(before, dtype=np.float64).sum()
        )
        if running_hits025 >= min_hits025:
            candidate = make_row(float(gap), end)
            if best is None or row_key(candidate) > row_key(best):
                best = candidate
        position = end
    assert best is not None
    return best


def evaluate_family(
    name,
    scores,
    locked_threshold,
    ious,
    groups,
    baselines,
    strict_hits025,
    strict_hits050,
    keep5440_hits025,
    metadata,
):
    baseline, alternative, gaps = decisions(
        scores, ious, groups, baselines
    )
    locked_changed = gaps >= float(locked_threshold)
    locked_ious = np.where(locked_changed, alternative, baseline)
    best_min = exact_threshold_best(
        baseline, alternative, gaps, strict_hits025
    )
    best_keep = exact_threshold_best(
        baseline, alternative, gaps, keep5440_hits025
    )
    result = {
        "name": name,
        "score_sha256": score_sha256(scores),
        "locked_threshold": float(locked_threshold),
        "locked": {
            **summarize(locked_ious),
            "fix_break": decision_stats(
                baseline, locked_ious, locked_changed
            ),
            "strict_goal_met": bool(
                int((locked_ious >= 0.25).sum()) >= strict_hits025
                and int((locked_ious >= 0.50).sum()) >= strict_hits050
            ),
        },
        "best_threshold_acc025_gt_5391": best_min,
        "best_threshold_acc025_gt_5440": best_keep,
        "threshold_family_can_meet_strict_goal": bool(
            best_min["hits050"] >= strict_hits050
        ),
        "metadata": metadata,
    }
    print(
        "FAMILY {} locked={}/{} best_keep5440={}/{} can_cross={}".format(
            name,
            result["locked"]["hits025"],
            result["locked"]["hits050"],
            best_keep["hits025"],
            best_keep["hits050"],
            result["threshold_family_can_meet_strict_goal"],
        ),
        flush=True,
    )
    return result, (baseline, alternative, gaps)


def compatible(lock):
    if lock.get("feature_names") != FEATURE_NAMES:
        return False, "feature_names"
    if int(lock.get("max_candidates", -1)) != 8:
        return False, "max_candidates"
    if "match_powers" in lock and tuple(lock["match_powers"]) != MATCH_POWERS:
        return False, "match_powers"
    if "actions" in lock and tuple(lock["actions"]) != ACTIONS:
        return False, "actions"
    return True, None


def load_booster_scores(path, expected_sha, iteration, features):
    assert os.path.isfile(path), path
    actual_sha = sha256(path)
    assert actual_sha == expected_sha, (path, actual_sha, expected_sha)
    booster = lgb.Booster(model_file=path)
    return np.asarray(
        booster.predict(features, num_iteration=int(iteration)),
        dtype=np.float32,
    ), actual_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("root")
    parser.add_argument("output_json")
    parser.add_argument("--top-models", type=int, default=10)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    started = time.time()
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
    count = len(rows)
    strict_hits025 = math.floor(0.5391 * count) + 1
    strict_hits050 = math.floor(0.4241 * count) + 1
    keep5440_hits025 = math.floor(0.5440 * count) + 1
    baseline, _, _ = decisions(
        np.zeros(len(ious), dtype=np.float32),
        ious,
        groups,
        baselines,
    )
    oracle = np.empty(count, dtype=np.float32)
    cursor = 0
    for index, size in enumerate(groups):
        size = int(size)
        oracle[index] = ious[cursor:cursor + size].max()
        cursor += size
    print(
        "DATA groups={} options={} features={} baseline={}/{} oracle={}/{}".format(
            count,
            len(features),
            len(FEATURE_NAMES),
            summarize(baseline)["hits025"],
            summarize(baseline)["hits050"],
            summarize(oracle)["hits025"],
            summarize(oracle)["hits050"],
        ),
        flush=True,
    )

    individuals = []
    skipped = []
    scores_by_name = {}
    score_aliases = {}
    model_score_cache = {}

    lock_paths = sorted(glob.glob(
        os.path.join(args.root, "**", "locked*.json"), recursive=True
    ))
    for lock_path in lock_paths:
        try:
            with open(lock_path, encoding="utf-8") as handle:
                lock = json.load(handle)
        except Exception as error:
            skipped.append({"lock": lock_path, "reason": repr(error)})
            continue
        ok, reason = compatible(lock)
        if not ok:
            skipped.append({"lock": lock_path, "reason": reason})
            continue
        rel = os.path.relpath(os.path.dirname(lock_path), args.root)
        try:
            if lock.get("model_path") and lock.get("model_sha256"):
                iteration = int(lock["best_iteration"])
                cache_key = (lock["model_sha256"], iteration)
                if cache_key not in model_score_cache:
                    model_score_cache[cache_key], _ = load_booster_scores(
                        lock["model_path"], lock["model_sha256"],
                        iteration, features
                    )
                scores = model_score_cache[cache_key]
                name = "single:" + rel
                threshold = float(lock["gate"]["threshold"])
                metadata = {
                    "kind": "single",
                    "lock": lock_path,
                    "lock_sha256": sha256(lock_path),
                    "model": lock["model_path"],
                    "model_sha256": lock["model_sha256"],
                    "iteration": iteration,
                    "protocol": lock.get("protocol"),
                }
            elif lock.get("models") and lock.get("fixed_iterations"):
                fold_scores = []
                verified = []
                for model in lock["models"]:
                    values, actual_sha = load_booster_scores(
                        model["path"], model["sha256"],
                        int(lock["fixed_iterations"]), features
                    )
                    fold_scores.append(values)
                    verified.append({
                        "path": model["path"], "sha256": actual_sha,
                        "fold": model.get("fold"),
                    })
                scores = np.mean(np.stack(fold_scores, axis=0), axis=0)
                name = "ensemble:" + rel
                threshold = float(lock["gate"]["threshold"])
                metadata = {
                    "kind": "ensemble",
                    "lock": lock_path,
                    "lock_sha256": sha256(lock_path),
                    "models": verified,
                    "iteration": int(lock["fixed_iterations"]),
                    "protocol": lock.get("protocol"),
                }
            elif lock.get("ordinal_model") and lock.get("binary_model"):
                ordinal_scores, ordinal_sha = load_booster_scores(
                    lock["ordinal_model"], lock["ordinal_model_sha256"],
                    int(lock["ordinal_iteration"]), features
                )
                binary_scores, binary_sha = load_booster_scores(
                    lock["binary_model"], lock["binary_model_sha256"],
                    int(lock["binary_iteration"]), features
                )
                selected = lock["selected"]
                scores = (
                    float(selected["ordinal_weight"])
                    * normalize_group_scores(ordinal_scores, groups)
                    + float(selected["binary_weight"])
                    * normalize_group_scores(binary_scores, groups)
                )
                name = "locked_blend:" + rel
                threshold = float(selected["gate"]["threshold"])
                metadata = {
                    "kind": "locked_blend",
                    "lock": lock_path,
                    "lock_sha256": sha256(lock_path),
                    "ordinal_model": lock["ordinal_model"],
                    "ordinal_model_sha256": ordinal_sha,
                    "binary_model": lock["binary_model"],
                    "binary_model_sha256": binary_sha,
                    "ordinal_weight": float(selected["ordinal_weight"]),
                    "binary_weight": float(selected["binary_weight"]),
                    "protocol": lock.get("protocol"),
                }
            else:
                skipped.append({
                    "lock": lock_path, "reason": "unsupported_schema"
                })
                continue
            digest = score_sha256(scores)
            if digest in score_aliases:
                skipped.append({
                    "lock": lock_path,
                    "reason": "identical_scores",
                    "alias_of": score_aliases[digest],
                })
                continue
            score_aliases[digest] = name
            result, _ = evaluate_family(
                name, scores, threshold, ious, groups, baselines,
                strict_hits025, strict_hits050, keep5440_hits025,
                metadata,
            )
            individuals.append(result)
            scores_by_name[name] = scores
        except Exception as error:
            skipped.append({"lock": lock_path, "reason": repr(error)})

    ranked_names = [
        row["name"] for row in sorted(
            individuals,
            key=lambda row: row_key(
                row["best_threshold_acc025_gt_5440"]
            ),
            reverse=True,
        )[:args.top_models]
    ]
    normalized = {
        name: normalize_group_scores(scores_by_name[name], groups)
        for name in ranked_names
    }
    pairwise = []
    weights = np.linspace(0.1, 0.9, 9)
    for left_index, left in enumerate(ranked_names):
        for right in ranked_names[left_index + 1:]:
            for right_weight in weights:
                scores = (
                    (1.0 - right_weight) * normalized[left]
                    + right_weight * normalized[right]
                )
                base, alternative, gaps = decisions(
                    scores, ious, groups, baselines
                )
                best_min = exact_threshold_best(
                    base, alternative, gaps, strict_hits025
                )
                best_keep = exact_threshold_best(
                    base, alternative, gaps, keep5440_hits025
                )
                pairwise.append({
                    "left": left,
                    "right": right,
                    "left_weight": float(1.0 - right_weight),
                    "right_weight": float(right_weight),
                    "best_threshold_acc025_gt_5391": best_min,
                    "best_threshold_acc025_gt_5440": best_keep,
                    "can_meet_strict_goal": bool(
                        best_min["hits050"] >= strict_hits050
                    ),
                    "diagnostic_only": True,
                })
    pairwise.sort(
        key=lambda row: row_key(row["best_threshold_acc025_gt_5440"]),
        reverse=True,
    )
    pairwise_top = pairwise[:50]
    crossings = [
        row for row in pairwise
        if row["can_meet_strict_goal"]
    ]
    print(
        "PAIRWISE tried={} crossings={} best={}".format(
            len(pairwise), len(crossings),
            json.dumps(
                pairwise_top[0] if pairwise_top else None,
                sort_keys=True,
            ),
        ),
        flush=True,
    )

    report = {
        "stage": "139_stage135c_ranker_zoo",
        "status": "complete",
        "diagnostic_only": True,
        "formal_use_requires_train_only_oof_freeze_and_reload": True,
        "dump": os.path.abspath(args.dump),
        "dump_sha256": sha256(args.dump),
        "script": os.path.abspath(__file__),
        "script_sha256": sha256(os.path.abspath(__file__)),
        "root": os.path.abspath(args.root),
        "count": count,
        "feature_names": FEATURE_NAMES,
        "strict_goal_hits": {
            "acc025": strict_hits025,
            "acc050": strict_hits050,
        },
        "keep5440_hits025": keep5440_hits025,
        "baseline": summarize(baseline),
        "oracle": summarize(oracle),
        "individuals": individuals,
        "individual_locked_crossings": [
            row["name"] for row in individuals
            if row["locked"]["strict_goal_met"]
        ],
        "individual_threshold_family_crossings": [
            row["name"] for row in individuals
            if row["threshold_family_can_meet_strict_goal"]
        ],
        "pairwise_top_models": ranked_names,
        "pairwise_weights": [float(value) for value in weights],
        "pairwise_count": len(pairwise),
        "pairwise_crossing_count": len(crossings),
        "pairwise_top50": pairwise_top,
        "pairwise_crossings_top50": crossings[:50],
        "skipped": skipped,
        "elapsed_seconds": float(time.time() - started),
    }
    output_tmp = args.output_json + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(output_tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(output_tmp, args.output_json)
    print(json.dumps({
        "output_json": os.path.abspath(args.output_json),
        "output_sha256": sha256(args.output_json),
        "elapsed_seconds": report["elapsed_seconds"],
        "individual_count": len(individuals),
        "locked_crossings": report["individual_locked_crossings"],
        "threshold_family_crossings": (
            report["individual_threshold_family_crossings"]
        ),
        "pairwise_crossing_count": len(crossings),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
