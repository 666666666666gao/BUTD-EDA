#!/usr/bin/env python
"""Parse ScanRefer eval logs for exploratory RAPF tuning.

This module is intentionally small and dependency-free so eval-only sweep,
Optuna short-runs, and top-k reruns use the same metric names and objective.
Tuning outputs are exploratory and are not official paper results.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import re
from pathlib import Path


MISSING = "NA"

BBS_ACC025_KEY = "last__bbs_acc0.25_top1"
BBS_ACC050_KEY = "last__bbs_acc0.50_top1"
BBF_ACC025_KEY = "last__bbf_acc0.25_top1"
BBF_ACC050_KEY = "last__bbf_acc0.50_top1"
DISAGREE_KEY = "last_bbs_vs_bbf_top1_disagree_ratio"
BBS_IOU_KEY = "last_bbs_top1_iou"
BBF_IOU_KEY = "last_bbf_top1_iou"

SOURCE_KEYS = (
    "eval_primary_score_source",
    "eval_bbs_score_source",
    "eval_bbf_score_source",
)

TUNING_METRIC_KEYS = (
    BBS_ACC025_KEY,
    BBS_ACC050_KEY,
    BBF_ACC025_KEY,
    BBF_ACC050_KEY,
    DISAGREE_KEY,
    BBS_IOU_KEY,
    BBF_IOU_KEY,
) + SOURCE_KEYS


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def parse_value(raw):
    value = raw.strip()
    if not value:
        return value
    try:
        number = float(value)
        if math.isfinite(number):
            return number
        return value
    except ValueError:
        return value


def parse_eval_text(text):
    metrics = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([^:\n]+?)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        key = match.group(1).strip()
        value = parse_value(match.group(2))
        metrics[key] = value
    return metrics


def parse_eval_file(path):
    return parse_eval_text(read_text(path))


def contains_failure(text):
    failure_markers = (
        "Traceback (most recent call last)",
        "RuntimeError:",
        "AssertionError:",
        "ValueError:",
        "CUDA out of memory",
        "nan loss",
    )
    return any(marker in text for marker in failure_markers)


def as_float(metrics, key):
    value = metrics.get(key)
    if isinstance(value, (float, int)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except ValueError:
            return None
    return None


def as_text(metrics, key):
    value = metrics.get(key)
    if value is None:
        return MISSING
    if isinstance(value, float):
        return "{:.6g}".format(value)
    return str(value)


def balanced_objective(acc025, acc050, weight_025=0.5, weight_050=0.5):
    if acc025 is None or acc050 is None:
        return None
    return float(weight_025) * acc025 + float(weight_050) * acc050


def objective_with_disagree_penalty(
        acc025, acc050, disagree, weight_025=0.5, weight_050=0.5):
    objective = balanced_objective(acc025, acc050, weight_025, weight_050)
    if objective is None:
        return None
    if disagree is None:
        return None
    return objective - 0.2 * max(0.0, disagree - 0.30)


def metric_row(metrics, weight_025=0.5, weight_050=0.5):
    acc025 = as_float(metrics, BBS_ACC025_KEY)
    acc050 = as_float(metrics, BBS_ACC050_KEY)
    bbf025 = as_float(metrics, BBF_ACC025_KEY)
    bbf050 = as_float(metrics, BBF_ACC050_KEY)
    disagree = as_float(metrics, DISAGREE_KEY)
    row = {
        "last_bbs_acc025": acc025,
        "last_bbs_acc050": acc050,
        "last_bbf_acc025": bbf025,
        "last_bbf_acc050": bbf050,
        "bbs_minus_bbf_acc025": (
            acc025 - bbf025 if acc025 is not None and bbf025 is not None else None
        ),
        "bbs_minus_bbf_acc050": (
            acc050 - bbf050 if acc050 is not None and bbf050 is not None else None
        ),
        "last_bbs_vs_bbf_top1_disagree_ratio": disagree,
        "last_bbs_top1_iou": as_float(metrics, BBS_IOU_KEY),
        "last_bbf_top1_iou": as_float(metrics, BBF_IOU_KEY),
        "eval_primary_score_source": as_text(metrics, "eval_primary_score_source"),
        "eval_bbs_score_source": as_text(metrics, "eval_bbs_score_source"),
        "eval_bbf_score_source": as_text(metrics, "eval_bbf_score_source"),
    }
    row["balanced_objective"] = balanced_objective(
        acc025, acc050, weight_025, weight_050
    )
    row["objective_with_disagree_penalty"] = objective_with_disagree_penalty(
        acc025, acc050, disagree, weight_025, weight_050
    )
    return row


def is_number(value):
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def csv_value(value):
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isfinite(value):
            return "{:.8g}".format(value)
        return MISSING
    return str(value)


def sort_rows(rows, key, reverse=True):
    def rank_value(row):
        value = row.get(key)
        if is_number(value):
            return float(value)
        return float("-inf") if reverse else float("inf")

    return sorted(rows, key=rank_value, reverse=reverse)


def markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(label for label, _ in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append(
            "| " + " | ".join(csv_value(row.get(key)) for _, key in columns) + " |"
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse a ScanRefer eval log and print tuning metrics as JSON."
    )
    parser.add_argument("log_path")
    parser.add_argument("--objective-weight-025", type=float, default=0.5)
    parser.add_argument("--objective-weight-050", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.log_path)
    metrics = parse_eval_file(path)
    row = metric_row(
        metrics,
        weight_025=args.objective_weight_025,
        weight_050=args.objective_weight_050,
    )
    row["log_path"] = os.fspath(path)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
