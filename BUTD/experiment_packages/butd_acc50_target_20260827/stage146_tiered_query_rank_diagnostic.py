#!/usr/bin/env python3
"""Train-only scene-split diagnostic for tiered query ranking.

The policy is fitted on ScanRefer training scenes only.  A deterministic
scene-disjoint dev split selects one reject threshold and a second internal
scene split gates the single validation evaluation.  No validation label is
used to fit the ranker or select its threshold.

Only deployable values already present in the frozen compact dump are used as
features.  IoU and GT boxes are labels/diagnostics only.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch


SCORE_KEYS = (
    "adapter_score_at_candidate",
    "adapter_fused_at_candidate",
    "adapter_delta_at_candidate",
    "adapter_hit25_logit_at_candidate",
    "adapter_hit50_logit_at_candidate",
    "adapter_rescue_logit_at_candidate",
)

QUALITY_KEYS = (
    "quality_topk_score",
    "base_at_quality_topk",
    "fused_at_quality_topk",
    "contrastive_at_quality_topk",
    "detector_class_at_quality_topk",
    "detector_conf20_at_quality_topk",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def scene_bucket(scene_id):
    digest = hashlib.sha256(str(scene_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def tier_label(iou):
    if float(iou) >= 0.50:
        return 2
    if float(iou) >= 0.25:
        return 1
    return 0


def standardize(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def descending_rank(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    order = np.argsort(-values, kind="stable")
    rank = np.empty(len(values), dtype=np.float32)
    rank[order] = np.arange(len(values), dtype=np.float32)
    return 1.0 - rank / max(len(values) - 1, 1)


def entropy(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    shifted = np.clip(values - values.max(), -50.0, 50.0)
    probability = np.exp(shifted)
    probability /= max(float(probability.sum()), 1e-8)
    return float(-(probability * np.log(probability + 1e-8)).sum())


def top_margin(values):
    values = np.sort(np.nan_to_num(np.asarray(values, dtype=np.float32)))[::-1]
    return float(values[0] - values[1]) if len(values) > 1 else 0.0


def safe_count(value):
    if value is None:
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def valid_nonnegative_id(value):
    if value is None:
        return False
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def quality_lookup(row):
    query_ids = list(row.get("quality_topk_query", ()))
    lookup = {}
    for offset, query_id in enumerate(query_ids):
        values = []
        for key in QUALITY_KEYS:
            source = row.get(key, ())
            values.append(float(source[offset]) if offset < len(source) else 0.0)
        lookup[int(query_id)] = values
    return lookup


def row_group(row, candidate_k=5):
    scores = np.asarray(row["adapter_score_at_candidate"], dtype=np.float32)
    ious = np.asarray(row["adapter_iou_at_candidate"], dtype=np.float32)
    boxes = np.asarray(row["adapter_box_at_candidate"], dtype=np.float32)
    query_ids = np.asarray(row["adapter_candidate_query"], dtype=np.int64)
    if not (len(scores) == len(ious) == len(boxes) == len(query_ids)):
        raise ValueError("candidate field lengths do not match")
    if len(scores) == 0:
        raise ValueError("candidate set is empty")
    order = np.argsort(-scores, kind="stable")[: min(candidate_k, len(scores))]
    source_values = {
        key: np.asarray(row[key], dtype=np.float32)[order]
        for key in SCORE_KEYS
    }
    source_z = {key: standardize(value) for key, value in source_values.items()}
    source_rank = {
        key: descending_rank(value) for key, value in source_values.items()
    }
    selected_boxes = boxes[order]
    selected_ious = ious[order]
    selected_queries = query_ids[order]
    baseline_box = selected_boxes[0]
    baseline_size = np.maximum(np.abs(baseline_box[3:]), 1e-3)
    quality = quality_lookup(row)

    status = str(row.get("decomposition_status", ""))
    profile = str(row.get("spacy_profile_bucket", ""))
    sample_context = [
        float(len(scores) / 16.0),
        float(math.log1p(safe_count(row.get("detector_class_count", 0)))),
        float(math.log1p(safe_count(row.get("detector_conf20_count", 0)))),
        float(math.log1p(safe_count(row.get("detector_conf50_count", 0)))),
        float(valid_nonnegative_id(row.get("text_target_cid", -1))),
        float(status == "ok"),
        float("repair" in status),
        float("global" in status),
        float("weak" in status),
        float("relation_free" in profile),
        float("view" in profile),
        float("direction" in profile),
    ]
    for key in (
        "adapter_score_at_candidate",
        "adapter_fused_at_candidate",
        "adapter_hit50_logit_at_candidate",
    ):
        sample_context.extend([
            top_margin(source_values[key]), entropy(source_values[key])
        ])

    features = []
    for local_index, (original_index, box, query_id) in enumerate(
        zip(order, selected_boxes, selected_queries)
    ):
        vector = []
        for key in SCORE_KEYS:
            raw = float(source_values[key][local_index])
            baseline_raw = float(source_values[key][0])
            vector.extend([
                raw,
                float(source_z[key][local_index]),
                float(source_rank[key][local_index]),
                raw - baseline_raw,
            ])
        relative_center = (box[:3] - baseline_box[:3]) / baseline_size
        relative_size = np.log(
            (np.maximum(np.abs(box[3:]), 1e-4)) / (baseline_size + 1e-4)
        )
        vector.extend(box.tolist())
        vector.extend(relative_center.tolist())
        vector.extend(relative_size.tolist())
        vector.extend([
            float(np.linalg.norm(relative_center)),
            float(np.log(np.maximum(np.prod(np.abs(box[3:])), 1e-8))),
            float(local_index / max(len(order) - 1, 1)),
            float(local_index == 0),
        ])
        quality_values = quality.get(int(query_id))
        vector.append(float(quality_values is not None))
        vector.extend(quality_values if quality_values is not None else [0.0] * len(QUALITY_KEYS))
        vector.extend(sample_context)
        features.append(vector)

    return {
        "features": np.asarray(features, dtype=np.float32),
        "labels": np.asarray([tier_label(value) for value in selected_ious], dtype=np.int32),
        "ious": selected_ious.astype(np.float32),
        "group": len(order),
        "baseline_iou": float(selected_ious[0]),
        "scene_id": str(row.get("scene_id", "")),
    }


def build_groups(rows, candidate_k=5):
    return [row_group(row, candidate_k=candidate_k) for row in rows]


def subset(groups, split):
    if split == "all":
        selected = groups
    else:
        selected = []
        for group in groups:
            bucket = scene_bucket(group["scene_id"])
            group_split = "dev" if bucket == 0 else "test" if bucket == 1 else "train"
            if group_split == split:
                selected.append(group)
    if not selected:
        raise ValueError("split {} is empty".format(split))
    return {
        "features": np.concatenate([group["features"] for group in selected]),
        "labels": np.concatenate([group["labels"] for group in selected]),
        "ious": np.concatenate([group["ious"] for group in selected]),
        "group_sizes": np.asarray([group["group"] for group in selected], dtype=np.int32),
        "baseline_ious": np.asarray([group["baseline_iou"] for group in selected], dtype=np.float32),
        "examples": len(selected),
        "scenes": len({group["scene_id"] for group in selected}),
    }


def predict_groups(model, arrays):
    flat = model.predict(arrays["features"], num_iteration=model.best_iteration_)
    result = []
    cursor = 0
    for size in arrays["group_sizes"]:
        size = int(size)
        result.append(np.asarray(flat[cursor:cursor + size], dtype=np.float32))
        cursor += size
    if cursor != len(flat):
        raise AssertionError("prediction groups do not consume all rows")
    return result


def apply_policy(predictions, arrays, threshold):
    chosen_ious = []
    changed = []
    gaps = []
    cursor = 0
    for prediction, size in zip(predictions, arrays["group_sizes"]):
        size = int(size)
        group_iou = arrays["ious"][cursor:cursor + size]
        best = int(np.argmax(prediction))
        gap = float(prediction[best] - prediction[0])
        use_best = best != 0 and gap >= float(threshold)
        chosen_ious.append(float(group_iou[best] if use_best else group_iou[0]))
        changed.append(use_best)
        gaps.append(gap if best != 0 else float("-inf"))
        cursor += size
    return (
        np.asarray(chosen_ious, dtype=np.float32),
        np.asarray(changed, dtype=bool),
        np.asarray(gaps, dtype=np.float32),
    )


def summarize(before, after, changed):
    result = {
        "count": int(len(before)),
        "changed": int(changed.sum()),
        "changed_ratio": float(changed.mean()),
        "mean_iou_before": float(before.mean()),
        "mean_iou_after": float(after.mean()),
    }
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        old = before >= threshold
        new = after >= threshold
        fixes = int((~old & new).sum())
        breaks = int((old & ~new).sum())
        result.update({
            "hits_{}_before".format(suffix): int(old.sum()),
            "hits_{}_after".format(suffix): int(new.sum()),
            "acc_{}_before".format(suffix): float(old.mean()),
            "acc_{}_after".format(suffix): float(new.mean()),
            "fix_{}".format(suffix): fixes,
            "break_{}".format(suffix): breaks,
            "net_{}".format(suffix): fixes - breaks,
        })
    return result


def evaluate(predictions, arrays, threshold):
    after, changed, _ = apply_policy(predictions, arrays, threshold)
    return summarize(arrays["baseline_ious"], after, changed)


def choose_threshold(predictions, arrays):
    _, _, gaps = apply_policy(predictions, arrays, float("inf"))
    finite = gaps[np.isfinite(gaps)]
    thresholds = [float("-inf"), float("inf")]
    if len(finite):
        thresholds.extend(
            float(value) for value in np.unique(
                np.quantile(finite, np.linspace(0.0, 1.0, 501))
            )
        )
    rows = []
    for threshold in thresholds:
        result = evaluate(predictions, arrays, threshold)
        result["threshold"] = float(threshold)
        result["preserves_acc025"] = bool(result["net_025"] >= 0)
        rows.append(result)
    feasible = [row for row in rows if row["preserves_acc025"]]
    return max(feasible, key=lambda row: (
        row["hits_050_after"],
        row["hits_025_after"],
        row["mean_iou_after"],
        -row["changed"],
    ))


def internal_gate(result):
    minimum_net50 = max(8, int(math.ceil(0.002 * result["count"])))
    checks = {
        "preserves_acc025": result["net_025"] >= 0,
        "positive_net050": result["net_050"] >= minimum_net50,
        "fixes_exceed_breaks050": result["fix_050"] > result["break_050"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_net050": minimum_net50,
    }


def load_rows(path):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "detector_topk_compact_v1":
        raise ValueError("unexpected dump format")
    return payload["rows"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dump", required=True)
    parser.add_argument("--val-dump", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-k", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(str(output))
    output.mkdir(parents=True)

    train_rows = load_rows(args.train_dump)
    train_groups = build_groups(train_rows, candidate_k=args.candidate_k)
    arrays = {name: subset(train_groups, name) for name in ("train", "dev", "test")}

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        label_gain=[0, 1, 5],
        n_estimators=360,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=0,
        n_jobs=4,
        verbosity=-1,
    )
    ranker.fit(
        arrays["train"]["features"],
        arrays["train"]["labels"],
        group=arrays["train"]["group_sizes"].tolist(),
        eval_set=[(arrays["dev"]["features"], arrays["dev"]["labels"])],
        eval_group=[arrays["dev"]["group_sizes"].tolist()],
        eval_at=[1],
        callbacks=[lgb.early_stopping(35, verbose=False)],
    )
    model_path = output / "tiered_query_ranker.txt"
    ranker.booster_.save_model(str(model_path))

    dev_predictions = predict_groups(ranker, arrays["dev"])
    selected = choose_threshold(dev_predictions, arrays["dev"])
    threshold = selected["threshold"]
    test_predictions = predict_groups(ranker, arrays["test"])
    test_result = evaluate(test_predictions, arrays["test"], threshold)
    gate = internal_gate(test_result)

    receipt = {
        "protocol": "scanrefer_train_scene_split_tiered_topk_rank_v1",
        "validation_labels_used_for_fit_or_threshold": False,
        "candidate_k": int(args.candidate_k),
        "tier_definition": {"tier2": "iou>=0.50", "tier1": "0.25<=iou<0.50", "tier0": "iou<0.25"},
        "feature_dimension": int(arrays["train"]["features"].shape[1]),
        "best_iteration": int(ranker.best_iteration_),
        "split_summary": {
            name: {"examples": value["examples"], "scenes": value["scenes"]}
            for name, value in arrays.items()
        },
        "dev_threshold_selection": selected,
        "internal_scene_test": test_result,
        "internal_gate": gate,
        "train_dump": {"path": args.train_dump, "sha256": sha256(args.train_dump)},
        "val_dump": {"path": args.val_dump, "sha256": sha256(args.val_dump)},
    }

    if gate["passed"]:
        val_rows = load_rows(args.val_dump)
        val_groups = build_groups(val_rows, candidate_k=args.candidate_k)
        val_arrays = subset(val_groups, "all")
        val_predictions = predict_groups(ranker, val_arrays)
        receipt["validation_single_evaluation"] = evaluate(
            val_predictions, val_arrays, threshold
        )
        receipt["validation_labels_consumed"] = True
    else:
        receipt["validation_single_evaluation"] = None
        receipt["validation_labels_consumed"] = False

    receipt_path = output / "tiered_query_rank_receipt.json"
    atomic_json(receipt_path, receipt)
    receipt["model_sha256"] = sha256(model_path)
    atomic_json(receipt_path, receipt)
    os.chmod(model_path, 0o444)
    os.chmod(receipt_path, 0o444)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
