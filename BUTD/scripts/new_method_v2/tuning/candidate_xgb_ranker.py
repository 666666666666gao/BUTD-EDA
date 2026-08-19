"""Train/evaluate an XGBoost ranker on dumped top-k candidate rows."""

import argparse
from collections import OrderedDict
import json
import os
import pickle

import numpy as np

from scripts.new_method_v2.tuning.source_choice_calibrator import (
    _load_rows,
    build_candidate_feature_matrix,
    evaluate_candidate_scores,
)


def _group_rows(rows):
    groups = OrderedDict()
    for row in rows:
        example_id = row.get("example_id")
        groups.setdefault(example_id, []).append(row)
    return list(groups.values())


def _flatten(groups):
    return [row for group in groups for row in group]


def build_ranker_training_data(rows, label_key="candidate_iou", columns=None):
    groups = _group_rows(rows)
    ordered_rows = _flatten(groups)
    x, _, columns = build_candidate_feature_matrix(
        ordered_rows, columns=columns
    )
    y = np.asarray(
        [float(row.get(label_key, 0.0)) for row in ordered_rows],
        dtype=np.float32,
    )
    group_sizes = [len(group) for group in groups]
    return x, y, columns, group_sizes


def _scores_for_key(rows, key):
    return [float(row.get(key, -1.0)) for row in rows]


def _metrics(ious):
    ious = np.asarray(ious, dtype=np.float32)
    return {
        "count": int(len(ious)),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "acc025": float((ious > 0.25).mean()) if len(ious) else 0.0,
        "acc050": float((ious > 0.50).mean()) if len(ious) else 0.0,
    }


def _choice_payload(rows, scores):
    groups = _group_rows(rows)
    payload = []
    offset = 0
    for group in groups:
        group_scores = np.asarray(
            scores[offset:offset + len(group)], dtype=np.float32
        )
        offset += len(group)
        quality_scores = np.asarray(
            [float(row.get("quality_score", -1.0)) for row in group],
            dtype=np.float32,
        )
        ious = np.asarray(
            [float(row.get("candidate_iou", -1.0)) for row in group],
            dtype=np.float32,
        )
        model_order = np.argsort(group_scores)[::-1]
        model_idx = int(model_order[0])
        second_score = (
            float(group_scores[int(model_order[1])])
            if len(model_order) > 1
            else float(group_scores[model_idx])
        )
        quality_idx = int(quality_scores.argmax())
        oracle_idx = int(ious.argmax())
        payload.append(
            {
                "model_iou": float(ious[model_idx]),
                "quality_iou": float(ious[quality_idx]),
                "oracle_iou": float(ious[oracle_idx]),
                "top_minus_quality": float(
                    group_scores[model_idx] - group_scores[quality_idx]
                ),
                "top_minus_second": float(
                    group_scores[model_idx] - second_score
                ),
            }
        )
    return payload


def _payload_metrics(payload, key):
    return _metrics([item[key] for item in payload])


def _thresholds(*arrays):
    values = np.concatenate(
        [np.asarray(array, dtype=np.float32) for array in arrays if len(array)]
    )
    if not len(values):
        return [0.0]
    return [float(value) for value in np.unique(np.quantile(values, np.linspace(0.0, 1.0, 401)))]


def _gate_metrics(payload, key, threshold):
    return _metrics(
        [
            item["model_iou"] if item[key] >= threshold else item["quality_iou"]
            for item in payload
        ]
    )


def _gate_2d_metrics(payload, threshold_quality, threshold_second):
    return _metrics(
        [
            (
                item["model_iou"]
                if (
                    item["top_minus_quality"] >= threshold_quality
                    and item["top_minus_second"] >= threshold_second
                )
                else item["quality_iou"]
            )
            for item in payload
        ]
    )


def _best_1d(train_payload, val_payload, key, thresholds, optimize):
    train_best = None
    val_best = None
    for threshold in thresholds:
        train_metrics = _gate_metrics(train_payload, key, threshold)
        val_metrics = _gate_metrics(val_payload, key, threshold)
        item = {
            "threshold": float(threshold),
            "train": train_metrics,
            "val": val_metrics,
        }
        train_key = (train_metrics[optimize], train_metrics["mean_iou"])
        val_key = (val_metrics[optimize], val_metrics["mean_iou"])
        if train_best is None or train_key > train_best[0]:
            train_best = (train_key, item)
        if val_best is None or val_key > val_best[0]:
            val_best = (val_key, item)
    return {"train_selected": train_best[1], "val_best_diagnostic": val_best[1]}


def _best_2d(
    train_payload,
    val_payload,
    thresholds_quality,
    thresholds_second,
    optimize,
):
    train_best = None
    val_best = None
    quality_grid = thresholds_quality[:: max(1, len(thresholds_quality) // 80)]
    second_grid = thresholds_second[:: max(1, len(thresholds_second) // 80)]
    for threshold_quality in quality_grid:
        for threshold_second in second_grid:
            train_metrics = _gate_2d_metrics(
                train_payload, threshold_quality, threshold_second
            )
            val_metrics = _gate_2d_metrics(
                val_payload, threshold_quality, threshold_second
            )
            item = {
                "threshold_top_minus_quality": float(threshold_quality),
                "threshold_top_minus_second": float(threshold_second),
                "train": train_metrics,
                "val": val_metrics,
            }
            train_key = (train_metrics[optimize], train_metrics["mean_iou"])
            val_key = (val_metrics[optimize], val_metrics["mean_iou"])
            if train_best is None or train_key > train_best[0]:
                train_best = (train_key, item)
            if val_best is None or val_key > val_best[0]:
                val_best = (val_key, item)
    return {"train_selected": train_best[1], "val_best_diagnostic": val_best[1]}


def _gate_sweep(train_payload, val_payload):
    train_tmq = [item["top_minus_quality"] for item in train_payload]
    train_tms = [item["top_minus_second"] for item in train_payload]
    val_tmq = [item["top_minus_quality"] for item in val_payload]
    val_tms = [item["top_minus_second"] for item in val_payload]
    thresholds_quality = _thresholds(train_tmq, val_tmq)
    thresholds_second = _thresholds(train_tms, val_tms)
    gates = {}
    for optimize in ("acc025", "acc050"):
        gates[optimize] = {
            "top_minus_quality": _best_1d(
                train_payload,
                val_payload,
                "top_minus_quality",
                thresholds_quality,
                optimize,
            ),
            "top_minus_second": _best_1d(
                train_payload,
                val_payload,
                "top_minus_second",
                thresholds_second,
                optimize,
            ),
            "conjunction_2d": _best_2d(
                train_payload,
                val_payload,
                thresholds_quality,
                thresholds_second,
                optimize,
            ),
        }
    return gates


def train_ranker(train_rows, val_rows, args):
    import xgboost as xgb

    x_train, y_train, columns, train_groups = build_ranker_training_data(
        train_rows, label_key=args.label_key
    )
    x_val, _, _, val_groups = build_ranker_training_data(
        val_rows, label_key=args.label_key, columns=columns
    )
    ordered_train_rows = _flatten(_group_rows(train_rows))
    ordered_val_rows = _flatten(_group_rows(val_rows))
    model = xgb.XGBRanker(
        objective=args.objective,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        tree_method=args.tree_method,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbosity=1,
    )
    model.fit(x_train, y_train, group=train_groups, verbose=False)
    train_scores = model.predict(x_train)
    val_scores = model.predict(x_val)
    train_payload = _choice_payload(ordered_train_rows, train_scores)
    val_payload = _choice_payload(ordered_val_rows, val_scores)
    metrics = {
        "row_type": "candidate_xgb_ranker",
        "label_key": args.label_key,
        "objective": args.objective,
        "num_features": len(columns),
        "feature_columns": columns,
        "train_candidate_rows": int(len(ordered_train_rows)),
        "val_candidate_rows": int(len(ordered_val_rows)),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "params": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "min_child_weight": float(args.min_child_weight),
            "reg_lambda": float(args.reg_lambda),
            "tree_method": args.tree_method,
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
        },
        "train": evaluate_candidate_scores(ordered_train_rows, train_scores),
        "val": evaluate_candidate_scores(ordered_val_rows, val_scores),
        "train_quality": evaluate_candidate_scores(
            ordered_train_rows, _scores_for_key(ordered_train_rows, "quality_score")
        ),
        "val_quality": evaluate_candidate_scores(
            ordered_val_rows, _scores_for_key(ordered_val_rows, "quality_score")
        ),
        "train_oracle": _payload_metrics(train_payload, "oracle_iou"),
        "val_oracle": _payload_metrics(val_payload, "oracle_iou"),
        "gates": _gate_sweep(train_payload, val_payload),
    }
    return model, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_key", default="candidate_iou")
    parser.add_argument("--objective", default="rank:pairwise")
    parser.add_argument("--n_estimators", type=int, default=160)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--min_child_weight", type=float, default=10.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--tree_method", default="hist")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_rows = _load_rows(args.train)
    val_rows = _load_rows(args.val)
    model, metrics = train_ranker(train_rows, val_rows, args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
