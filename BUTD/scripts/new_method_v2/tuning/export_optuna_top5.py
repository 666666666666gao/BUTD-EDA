#!/usr/bin/env python
"""Export global Optuna top-5 reports from a shared ScanRefer tuning study."""

from __future__ import print_function

import argparse
import sys
from pathlib import Path

import optuna_scanrefer_two_stage_full as tuning


OUTPUT_NAMES = (
    "optuna_scanrefer_two_stage_full_trials.csv",
    "optuna_scanrefer_two_stage_full_best.json",
    "optuna_scanrefer_two_stage_full_top5.json",
    "optuna_scanrefer_two_stage_full_top5.csv",
    "optuna_scanrefer_two_stage_full_summary.md",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export top-5 reports from a shared ScanRefer Optuna study."
    )
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--output-dir", default="reports/tuning")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--objective-weight-025", type=float, default=0.5)
    parser.add_argument("--objective-weight-050", type=float, default=0.5)
    args = parser.parse_args()
    args.report_dir = args.output_dir
    return args


def dry_run(args):
    print("[dry-run] study_name = {}".format(args.study_name))
    print("[dry-run] storage = {}".format(tuning.mask_storage_url(args.storage)))
    print("[dry-run] output files:")
    for name in OUTPUT_NAMES:
        print("  {}".format(Path(args.output_dir) / name))
    return 0


def main():
    args = parse_args()
    if args.dry_run:
        return dry_run(args)
    try:
        import optuna
    except ImportError:
        print(
            "ERROR: Optuna is not installed. Install it with `pip install optuna` "
            "or add it to the active environment before exporting reports.",
            file=sys.stderr,
        )
        return 2
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    rows = tuning.collect_records(study)
    tuning.write_all_outputs(args, rows)
    print("Exported global Optuna tuning reports under {}".format(args.output_dir))
    print("Study: {}".format(args.study_name))
    print("Storage: {}".format(tuning.mask_storage_url(args.storage)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
