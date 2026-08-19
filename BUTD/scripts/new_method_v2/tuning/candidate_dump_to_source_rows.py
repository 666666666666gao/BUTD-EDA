"""Convert top-k candidate dumps into source-level top1 rows."""

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.new_method_v2.tuning.source_choice_calibrator import (  # noqa: E402
    DEFAULT_SOURCE_NAMES,
    _load_rows_with_metadata,
    _normalize_source_names,
)
from src.grounding_evaluator import GroundingEvaluator  # noqa: E402


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_groups(rows):
    current_id = None
    current_rows = []
    for row in rows:
        example_id = row.get("example_id")
        if current_id is None:
            current_id = example_id
        if example_id != current_id:
            yield current_id, current_rows
            current_id = example_id
            current_rows = []
        current_rows.append(row)
    if current_rows:
        yield current_id, current_rows


def _top_row(group_rows, source):
    available_rows = [
        row for row in group_rows
        if _as_float(row.get(f"{source}_available", 0.0)) > 0.5
    ]
    if not available_rows:
        return None
    return min(
        available_rows,
        key=lambda row: (
            _as_float(row.get(f"{source}_rank", 1.0e9)),
            _as_float(row.get(f"{source}_delta_to_top", 1.0e9)),
            -_as_float(row.get(f"{source}_score", 0.0)),
        ),
    )


def _score_at_rank(group_rows, source, rank):
    for row in group_rows:
        if int(round(_as_float(row.get(f"{source}_rank", 0.0)))) == rank:
            return _as_float(row.get(f"{source}_score", 0.0))
    return None


def _copy_prefixes(out, row, prefixes):
    for key, value in row.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            out[key] = _as_float(value)


def _convert_group(example_id, group_rows, source_names):
    first = group_rows[0]
    out = {"example_id": _as_float(example_id)}
    _copy_prefixes(out, first, ("context_", "layer_stability_"))

    top_rows = {}
    top_ious = []
    available = []
    for source in source_names:
        row = _top_row(group_rows, source)
        if row is None:
            available.append(False)
            top_ious.append(-1.0)
            out[f"{source}_available"] = 0.0
            out[f"{source}_top_query"] = -1.0
            out[f"{source}_top_score"] = 0.0
            out[f"{source}_top_margin"] = 0.0
            out[f"{source}_top_iou"] = -1.0
            continue

        top_rows[source] = row
        available.append(True)
        top_iou = _as_float(row.get("candidate_iou", -1.0))
        top_ious.append(top_iou)
        top_score = _as_float(row.get(f"{source}_score", 0.0))
        second_score = _score_at_rank(group_rows, source, 2)
        top_margin = 0.0 if second_score is None else top_score - second_score
        out[f"{source}_available"] = 1.0
        out[f"{source}_top_query"] = _as_float(row.get("candidate_query", -1.0))
        out[f"{source}_top_score"] = top_score
        out[f"{source}_top_margin"] = top_margin
        out[f"{source}_top_iou"] = top_iou

        for key, value in row.items():
            if key.startswith(f"{source}_top_"):
                out[key] = _as_float(value)
            elif key.startswith("rapf_"):
                out[f"{source}_{key}"] = _as_float(value)

    for top_source in source_names:
        top = top_rows.get(top_source)
        for score_source in source_names:
            key = f"{score_source}_score_at_{top_source}_top"
            out[key] = (
                0.0 if top is None else _as_float(top.get(f"{score_source}_score", 0.0))
            )

    for left_idx, left in enumerate(source_names):
        left_row = top_rows.get(left)
        for right in source_names[left_idx + 1:]:
            right_row = top_rows.get(right)
            same = (
                left_row is not None
                and right_row is not None
                and int(round(_as_float(left_row.get("candidate_query", -1.0))))
                == int(round(_as_float(right_row.get("candidate_query", -2.0))))
            )
            out[f"{left}_{right}_same_query"] = 1.0 if same else 0.0

    GroundingEvaluator._source_choice_pairwise_numeric_deltas(
        out, source_names, ("top_score", "top_margin"), include_abs=True
    )
    GroundingEvaluator._source_choice_cross_score_deltas(out, source_names)
    GroundingEvaluator._source_choice_top_geometry_deltas(out, source_names)

    top_iou_tensor = torch.as_tensor(top_ious, dtype=torch.float32)
    available_tensor = torch.as_tensor(available, dtype=torch.bool)
    masked_iou = top_iou_tensor.masked_fill(~available_tensor, -1.0)
    oracle_source = int(masked_iou.argmax().item())
    threshold_utility = (
        masked_iou
        + (masked_iou > 0.25).float()
        + (masked_iou > 0.50).float()
    ).masked_fill(~available_tensor, -1.0)
    threshold_source = int(threshold_utility.argmax().item())
    oracle_iou = float(masked_iou[oracle_source].item())
    out["oracle_source_id"] = float(oracle_source)
    out["threshold_utility_source_id"] = float(threshold_source)
    out["oracle_iou"] = oracle_iou
    out["oracle_hit025"] = 1.0 if oracle_iou > 0.25 else 0.0
    out["oracle_hit050"] = 1.0 if oracle_iou > 0.50 else 0.0
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source_names",
        default="",
        help=(
            "Comma-separated source-name order to keep in the converted "
            "source rows."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, metadata = _load_rows_with_metadata(args.input)
    if args.source_names:
        source_names = _normalize_source_names(args.source_names)
    else:
        source_names = _normalize_source_names(
            metadata.get("source_names", DEFAULT_SOURCE_NAMES)
        )
    source_rows = [
        _convert_group(example_id, group_rows, source_names)
        for example_id, group_rows in _iter_groups(rows)
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "rows": source_rows,
            "source_names": list(source_names),
            "row_type": "source",
        },
        args.output,
    )
    print(
        "converted {} candidate rows into {} source rows -> {}".format(
            len(rows), len(source_rows), args.output
        )
    )


if __name__ == "__main__":
    main()
