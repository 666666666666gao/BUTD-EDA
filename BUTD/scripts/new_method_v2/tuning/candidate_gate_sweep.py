"""Sweep quality-protected gates over dumped candidate calibrator scores."""

import argparse
import json
import os
import sys
import pickle

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.new_method_v2.tuning.source_choice_calibrator import (
    CandidateListwiseScorer,
    _load_rows,
    _predict_candidate_listwise_scores,
    build_candidate_feature_matrix,
)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "CandidateListwiseScorer":
            return CandidateListwiseScorer
        return super().find_class(module, name)


def _load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def _as_array(values):
    return np.asarray(values, dtype=np.float32)


def _row_iou(row):
    return float(row.get("candidate_iou", -1.0))


def _metrics(ious):
    ious = _as_array(ious)
    return {
        "count": int(len(ious)),
        "mean_iou": float(ious.mean()) if len(ious) else 0.0,
        "acc025": float((ious > 0.25).mean()) if len(ious) else 0.0,
        "acc050": float((ious > 0.50).mean()) if len(ious) else 0.0,
    }


def _groups(rows, scores):
    grouped = []
    current_id = None
    current_rows = []
    current_scores = []
    for row, score in zip(rows, scores):
        example_id = row.get("example_id")
        if current_id is None:
            current_id = example_id
        if example_id != current_id:
            grouped.append((current_id, current_rows, _as_array(current_scores)))
            current_id = example_id
            current_rows = []
            current_scores = []
        current_rows.append(row)
        current_scores.append(float(score))
    if current_rows:
        grouped.append((current_id, current_rows, _as_array(current_scores)))
    return grouped


def _choices(rows, scores):
    quality_ious = []
    model_ious = []
    oracle_ious = []
    top_minus_quality = []
    top_minus_second = []
    gate_payload = []

    for _, group_rows, group_scores in _groups(rows, scores):
        quality_scores = _as_array(
            [float(row.get("quality_score", 0.0)) for row in group_rows]
        )
        quality_idx = int(quality_scores.argmax())
        model_order = np.argsort(group_scores)[::-1]
        model_idx = int(model_order[0])
        second_score = (
            float(group_scores[int(model_order[1])])
            if len(model_order) > 1
            else float(group_scores[model_idx])
        )
        oracle_idx = int(
            np.asarray([_row_iou(row) for row in group_rows]).argmax()
        )

        quality_ious.append(_row_iou(group_rows[quality_idx]))
        model_ious.append(_row_iou(group_rows[model_idx]))
        oracle_ious.append(_row_iou(group_rows[oracle_idx]))
        diff_quality = float(group_scores[model_idx] - group_scores[quality_idx])
        diff_second = float(group_scores[model_idx] - second_score)
        top_minus_quality.append(diff_quality)
        top_minus_second.append(diff_second)
        gate_payload.append(
            {
                "quality_iou": _row_iou(group_rows[quality_idx]),
                "model_iou": _row_iou(group_rows[model_idx]),
                "top_minus_quality": diff_quality,
                "top_minus_second": diff_second,
            }
        )

    return {
        "quality": _metrics(quality_ious),
        "model": _metrics(model_ious),
        "oracle": _metrics(oracle_ious),
        "top_minus_quality": _as_array(top_minus_quality),
        "top_minus_second": _as_array(top_minus_second),
        "payload": gate_payload,
    }


def _thresholds(*arrays):
    values = np.concatenate([array for array in arrays if len(array)])
    if not len(values):
        return [0.0]
    qs = np.linspace(0.0, 1.0, 401)
    thresholds = np.unique(np.quantile(values, qs))
    return [float(value) for value in thresholds]


def _gate_metrics(choices, key, threshold):
    ious = [
        item["model_iou"] if item[key] >= threshold else item["quality_iou"]
        for item in choices["payload"]
    ]
    return _metrics(ious)


def _gate_2d_metrics(choices, threshold_quality, threshold_second):
    ious = [
        (
            item["model_iou"]
            if (
                item["top_minus_quality"] >= threshold_quality
                and item["top_minus_second"] >= threshold_second
            )
            else item["quality_iou"]
        )
        for item in choices["payload"]
    ]
    return _metrics(ious)


def _best_1d(train_choices, val_choices, key, thresholds, optimize):
    train_best = None
    val_best = None
    for threshold in thresholds:
        train_metrics = _gate_metrics(train_choices, key, threshold)
        val_metrics = _gate_metrics(val_choices, key, threshold)
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


def _best_2d(train_choices, val_choices, thresholds_quality, thresholds_second, optimize):
    train_best = None
    val_best = None
    tq_grid = thresholds_quality[:: max(1, len(thresholds_quality) // 80)]
    ts_grid = thresholds_second[:: max(1, len(thresholds_second) // 80)]
    for threshold_quality in tq_grid:
        for threshold_second in ts_grid:
            train_metrics = _gate_2d_metrics(
                train_choices, threshold_quality, threshold_second
            )
            val_metrics = _gate_2d_metrics(
                val_choices, threshold_quality, threshold_second
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


def _rf_scores(model, rows, columns):
    x, _, _ = build_candidate_feature_matrix(rows, columns=columns)
    return model.predict_proba(x)[:, 1]


def _listwise_scores(model, rows, columns, device):
    model = model.to(device)
    return _predict_candidate_listwise_scores(model, rows, columns, device)


def _scores(model, model_type, rows, columns, device):
    if model_type == "rf":
        return _rf_scores(model, rows, columns)
    if model_type == "listwise":
        return _listwise_scores(model, rows, columns, device)
    raise ValueError("unsupported model type: {}".format(model_type))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--model_type", choices=["rf", "listwise"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--listwise_device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    metric_data = _load_metrics(args.metrics)
    columns = metric_data["feature_columns"]
    with open(args.model, "rb") as f:
        model = _CompatUnpickler(f).load()

    train_rows = _load_rows(args.train)
    val_rows = _load_rows(args.val)
    device = (
        "cuda"
        if args.listwise_device == "auto" and torch.cuda.is_available()
        else args.listwise_device
    )
    train_scores = _scores(model, args.model_type, train_rows, columns, device)
    val_scores = _scores(model, args.model_type, val_rows, columns, device)
    train_choices = _choices(train_rows, train_scores)
    val_choices = _choices(val_rows, val_scores)

    tq_thresholds = _thresholds(
        train_choices["top_minus_quality"],
        val_choices["top_minus_quality"],
    )
    ts_thresholds = _thresholds(
        train_choices["top_minus_second"],
        val_choices["top_minus_second"],
    )
    output = {
        "model_type": args.model_type,
        "train": {
            "quality": train_choices["quality"],
            "model": train_choices["model"],
            "oracle": train_choices["oracle"],
        },
        "val": {
            "quality": val_choices["quality"],
            "model": val_choices["model"],
            "oracle": val_choices["oracle"],
        },
        "gates": {},
    }
    for optimize in ("acc025", "acc050"):
        output["gates"][optimize] = {
            "top_minus_quality": _best_1d(
                train_choices, val_choices, "top_minus_quality",
                tq_thresholds, optimize,
            ),
            "top_minus_second": _best_1d(
                train_choices, val_choices, "top_minus_second",
                ts_thresholds, optimize,
            ),
            "conjunction_2d": _best_2d(
                train_choices, val_choices, tq_thresholds,
                ts_thresholds, optimize,
            ),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
