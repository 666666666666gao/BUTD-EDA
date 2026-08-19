"""Add source-pair delta features to existing source-choice dumps."""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.grounding_evaluator import GroundingEvaluator


SOURCE_NAMES = ("base", "fused", "quality", "contrastive_base", "acd")


def _load_rows(path):
    data = torch.load(path, map_location="cpu")
    rows = data["rows"] if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a row list")
    return data, rows


def _source_names_for_row(row):
    sources = []
    for source in SOURCE_NAMES:
        if (
            f"{source}_available" in row
            or f"{source}_score" in row
            or f"{source}_top_score" in row
        ):
            sources.append(source)
    return tuple(sources)


def _row_type(row, requested):
    if requested != "auto":
        return requested
    return "candidate" if "candidate_query" in row else "top1"


def augment_rows(rows, row_type="auto"):
    if not rows:
        return {"row_type": row_type, "rows": 0, "keys_before": 0,
                "keys_after": 0}

    inferred_type = _row_type(rows[0], row_type)
    keys_before = len(rows[0])
    for row in rows:
        source_names = _source_names_for_row(row)
        if inferred_type == "candidate":
            GroundingEvaluator._source_choice_pairwise_numeric_deltas(
                row,
                source_names,
                ("score", "rank", "delta_to_top", "in_topk"),
            )
            GroundingEvaluator._source_choice_top_score_deltas(
                row, source_names
            )
            GroundingEvaluator._source_choice_top_geometry_deltas(
                row, source_names
            )
        elif inferred_type == "top1":
            GroundingEvaluator._source_choice_pairwise_numeric_deltas(
                row,
                source_names,
                ("top_score", "top_margin"),
                include_abs=True,
            )
            GroundingEvaluator._source_choice_cross_score_deltas(
                row, source_names
            )
            GroundingEvaluator._source_choice_top_geometry_deltas(
                row, source_names
            )
        else:
            raise ValueError(f"unknown row type: {inferred_type}")

    return {
        "row_type": inferred_type,
        "rows": len(rows),
        "keys_before": keys_before,
        "keys_after": len(rows[0]),
    }


def save_rows(original_data, rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(original_data, dict):
        data = dict(original_data)
        data["rows"] = rows
    else:
        data = {"rows": rows}
    torch.save(data, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--row_type",
        default="auto",
        choices=["auto", "top1", "candidate"],
    )
    args = parser.parse_args()

    data, rows = _load_rows(args.input)
    summary = augment_rows(rows, row_type=args.row_type)
    save_rows(data, rows, args.output)
    print(
        "augmented {input} -> {output}: {rows} {row_type} rows, "
        "{keys_before} -> {keys_after} keys".format(
            input=args.input,
            output=args.output,
            **summary,
        )
    )


if __name__ == "__main__":
    main()
